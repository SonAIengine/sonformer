"""
02a-rope: Decoder-only transformer with **Rotary Position Embedding (RoPE)**.

Diff vs 00-baseline:
    - Removed `self.pos_emb` (no learned absolute PE)
    - Added RoPE buffers (cos/sin) and apply rotation to q,k inside Attention
    - `head_dim` must be even (RoPE rotates pairs)
    - Pre-compute RoPE cache to a large length so extrapolation eval works
      without rebuilding the model.

RoPE convention: LLaMA-style "split-half" rotation
    Given x of last-dim D, split into x1 (first D/2) and x2 (last D/2).
    Rotate: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    max_seq_len: int = 512
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_hidden: int = 1536
    dropout: float = 0.1
    pad_id: int = 0
    rope_base: float = 10000.0
    rope_cache_len: int = 8192   # pre-computed cos/sin length (for extrapolation eval)


# ───────────────────────────────────────────────────────────────────────────
# RoPE primitives
# ───────────────────────────────────────────────────────────────────────────

def precompute_rope(head_dim: int, max_len: int, base: float = 10000.0,
                    device=None, dtype=torch.float32):
    """Returns (cos, sin) of shape (max_len, head_dim)."""
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    pos = torch.arange(max_len, device=device, dtype=dtype)
    freqs = torch.outer(pos, inv_freq)               # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)          # (T, D)  duplicated halves
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half and rotate: concat([-x2, x1])."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, H, T, D)
    cos: (T, D), sin: (T, D)  — sliced to current T before passing in
    """
    cos = cos.unsqueeze(0).unsqueeze(0)              # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


# ───────────────────────────────────────────────────────────────────────────
# Modules
# ───────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head self-attention with RoPE applied to q, k."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # (B, H, T, D)

        # ── RoPE: rotate q and k (NOT v) ───────────────────────────────────
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FFN(nn.Module):
    """2-layer ReLU MLP (identical to baseline)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.relu(self.up(x))))


class Block(nn.Module):
    """Pre-LN transformer block; passes RoPE cos/sin through to attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Sonformer(nn.Module):
    """Decoder-only transformer LM with RoPE."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # NOTE: no pos_emb — RoPE replaces absolute positional embedding
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight    # weight tying

        self.apply(self._init_weights)

        # Pre-compute RoPE cache up to rope_cache_len (for extrapolation eval).
        head_dim = cfg.d_model // cfg.n_heads
        cache_len = max(cfg.max_seq_len, cfg.rope_cache_len)
        cos, sin = precompute_rope(head_dim, cache_len, base=cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Causal mask up to rope_cache_len so extrapolation works without rebuild.
        mask = torch.tril(torch.ones(cache_len, cache_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask, persistent=False)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _maybe_extend_cache(self, T: int, device, dtype):
        """If T exceeds cached length, recompute (cheap)."""
        if T <= self.rope_cos.shape[0]:
            return
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = precompute_rope(head_dim, T, base=self.cfg.rope_base, device=device, dtype=dtype)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        self._maybe_extend_cache(T, idx.device, self.rope_cos.dtype)

        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)
        mask = self.causal_mask[:, :, :T, :T].to(idx.device)

        x = self.drop(self.tok_emb(idx))             # NO pos_emb addition
        for block in self.blocks:
            x = block(x, cos, sin, mask)

        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=self.cfg.pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> tuple[torch.Tensor, list]:
        self.eval()
        for _ in range(max_new_tokens):
            # RoPE는 외삽 가능하지만 학습 길이 한참 넘어가면 품질 저하 → 자르기
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx, []

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
