"""
Encoder-Decoder Transformer — original 2017 paper ("Attention is All You Need").
Single-file reference (정답지) for hand-coding exercises in this folder.

Key difference from `practice/sonlm-from-scratch/solution.py`:
    - Two stacks: Encoder (bidirectional self-attn) + Decoder (causal self-attn + cross-attn)
    - Decoder layers have THREE sub-layers (self-attn, cross-attn, FFN)
    - Sinusoidal positional encoding (paper-original) instead of learned PE
    - Designed for seq2seq (translation, summarization, etc.) — here demoed via
      a sequence-reversal task: src=[a,b,c,d,EOS] → tgt=[BOS,d,c,b,a,EOS]

Run:
    python solution.py        # 약 1500 step에 reverse task 학습 (~30s on H100)
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Vocab + special tokens
    vocab_size: int = 32
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3              # encoder layers = decoder layers (편의상)
    ffn_hidden: int = 512          # 4× d_model
    max_seq_len: int = 16
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    max_steps: int = 1500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.1   # original paper의 trick
    log_interval: int = 100
    seed: int = 42


# ───────────────────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding (paper-original)
# ───────────────────────────────────────────────────────────────────────

def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """
    PE_(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE_(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Returns: (max_len, d_model)
    """
    pos = torch.arange(max_len).unsqueeze(1).float()                  # (max_len, 1)
    i = torch.arange(0, d_model, 2).float()                            # (d_model/2,)
    div = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ───────────────────────────────────────────────────────────────────────
# Multi-Head Attention — general (supports self- AND cross-attention)
# ───────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Q from one source, K/V from another (or same).
    self-attention:  q_in = k_in = v_in
    cross-attention: q_in = decoder hidden, k_in = v_in = encoder output
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Tq, C = q_in.shape
        Tk = k_in.shape[1]
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(q_in).view(B, Tq, H, D).transpose(1, 2)     # (B, H, Tq, D)
        k = self.k_proj(k_in).view(B, Tk, H, D).transpose(1, 2)     # (B, H, Tk, D)
        v = self.v_proj(v_in).view(B, Tk, H, D).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D)              # (B, H, Tq, Tk)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Tq, C)
        return self.out_proj(out)


# ───────────────────────────────────────────────────────────────────────
# FFN
# ───────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.relu(self.up(x))))


# ───────────────────────────────────────────────────────────────────────
# Encoder Layer — 2 sub-layers: self-attn, FFN
# ───────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """Pre-LN. self-attention is bidirectional (no causal mask)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, src_mask)            # self-attn
        x = x + self.ffn(self.norm2(x))                 # FFN
        return x


# ───────────────────────────────────────────────────────────────────────
# Decoder Layer — 3 sub-layers: causal self-attn, cross-attn, FFN
# ───────────────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    Pre-LN. Three sub-layers in order:
      1. causal self-attention (decoder는 이전 자기 출력만 봄)
      2. cross-attention (decoder Q × encoder K/V — 입력 시퀀스를 참조)
      3. FFN
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Causal self-attention (decoder hidden → decoder hidden, masked future)
        n = self.norm1(x)
        x = x + self.self_attn(n, n, n, tgt_mask)

        # 2. Cross-attention: q from decoder, k/v from encoder output
        n = self.norm2(x)
        x = x + self.cross_attn(n, enc_out, enc_out, src_mask)

        # 3. FFN
        x = x + self.ffn(self.norm3(x))
        return x


# ───────────────────────────────────────────────────────────────────────
# Full Encoder-Decoder Transformer
# ───────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Encoder-decoder transformer (original 2017 paper).
    Shared vocab between src/tgt (간소화).
    Embeddings tied with output projection.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer(
            "pe", sinusoidal_pe(cfg.max_seq_len * 2, cfg.d_model), persistent=False
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm_enc = nn.LayerNorm(cfg.d_model)
        self.norm_dec = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.out_proj.weight = self.tok_emb.weight              # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)                   # 원조 paper convention
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        # 원조 paper: embedding × √d_model 으로 scale 한 뒤 PE 더함
        x = self.tok_emb(ids) * math.sqrt(self.cfg.d_model)
        x = x + self.pe[: ids.shape[1]].to(x.device)
        return self.dropout(x)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.norm_enc(x)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        src_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self._embed(tgt)
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm_dec(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return self.out_proj(dec_out)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Greedy decoding: <BOS>로 시작해 <EOS>까지."""
        self.eval()
        enc_out = self.encode(src, src_mask)                     # encoder 한 번만 돌림
        B = src.size(0)
        out = torch.full((B, 1), self.cfg.bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            T = out.size(1)
            tgt_mask = make_tgt_mask(out, self.cfg.pad_id, T, src.device)
            dec_out = self.decode(out, enc_out, tgt_mask, src_mask)
            next_id = self.out_proj(dec_out[:, -1]).argmax(-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
            if (next_id == self.cfg.eos_id).all():
                break
        return out


# ───────────────────────────────────────────────────────────────────────
# Mask utilities
# ───────────────────────────────────────────────────────────────────────

def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Padding mask for encoder. (B, 1, 1, T) — 1=attend, 0=ignore."""
    return (src != pad_id)[:, None, None, :].long()


def make_tgt_mask(tgt: torch.Tensor, pad_id: int, T: int, device) -> torch.Tensor:
    """Padding + causal mask for decoder self-attention. (B, 1, T, T)."""
    pad = (tgt != pad_id)[:, None, None, :].long()                          # (B, 1, 1, T)
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.long))[None, None]  # (1, 1, T, T)
    return pad & causal


# ───────────────────────────────────────────────────────────────────────
# Synthetic task: sequence reversal
# ───────────────────────────────────────────────────────────────────────

def make_reverse_batch(
    batch_size: int,
    cfg: Config,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    src:  [a, b, c, d, EOS, PAD, ...]
    tgt:  [BOS, d, c, b, a, EOS, PAD, ...]   (reversed)
    토큰 범위: 3 ~ vocab_size-1 (PAD/BOS/EOS 제외)
    """
    seq_lens = torch.randint(3, cfg.max_seq_len // 2, (batch_size,))
    max_src_len = seq_lens.max().item() + 1                     # +1 for EOS
    max_tgt_len = seq_lens.max().item() + 2                     # +2 for BOS, EOS

    src = torch.full((batch_size, max_src_len), cfg.pad_id, dtype=torch.long)
    tgt = torch.full((batch_size, max_tgt_len), cfg.pad_id, dtype=torch.long)

    for i, L in enumerate(seq_lens.tolist()):
        seq = torch.randint(3, cfg.vocab_size, (L,))
        src[i, :L] = seq
        src[i, L] = cfg.eos_id
        tgt[i, 0] = cfg.bos_id
        tgt[i, 1 : L + 1] = seq.flip(0)                          # 뒤집기
        tgt[i, L + 1] = cfg.eos_id

    return src.to(device), tgt.to(device)


# ───────────────────────────────────────────────────────────────────────
# LR schedule (warmup + cosine)
# ───────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


# ───────────────────────────────────────────────────────────────────────
# Smoke test — train on reverse task
# ───────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params ({n_params / 1e6:.2f}M)")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),                                       # 원조 paper 값
    )

    losses: list[float] = []
    t0 = time.time()
    for step in range(cfg.max_steps):
        src, tgt = make_reverse_batch(cfg.batch_size, cfg, device)
        # Teacher forcing: input=tgt[:-1], target=tgt[1:]
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, cfg.pad_id)
        tgt_mask = make_tgt_mask(tgt_in, cfg.pad_id, tgt_in.size(1), device)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            tgt_out.reshape(-1),
            ignore_index=cfg.pad_id,
            label_smoothing=cfg.label_smoothing,
        )

        lr = get_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        losses.append(loss.item())
        if step % cfg.log_interval == 0:
            window = losses[-cfg.log_interval :]
            avg = sum(window) / max(1, len(window))
            print(f"step {step:5d} | lr {lr:.6f} | loss {avg:.4f} | {time.time() - t0:.1f}s")

    # ── Final test: 모델이 reverse를 진짜 학습했나? ───────────────────
    print("\n[final] testing reversal generalization...")
    model.eval()
    src, tgt = make_reverse_batch(8, cfg, device)
    src_mask = make_src_mask(src, cfg.pad_id)
    pred = model.generate(src, src_mask, max_len=cfg.max_seq_len)

    correct = 0
    for i in range(src.size(0)):
        # src에서 EOS 전까지가 입력
        src_seq = src[i].tolist()
        eos_idx = src_seq.index(cfg.eos_id) if cfg.eos_id in src_seq else len(src_seq)
        in_seq = src_seq[:eos_idx]
        # tgt와 pred에서 BOS 다음 ~ EOS 전까지가 출력
        tgt_seq = tgt[i].tolist()
        pred_seq = pred[i].tolist()
        tgt_out = tgt_seq[1 : tgt_seq.index(cfg.eos_id) if cfg.eos_id in tgt_seq[1:] else len(tgt_seq)]
        pred_after_bos = pred_seq[1:] if pred_seq[0] == cfg.bos_id else pred_seq
        pred_out = pred_after_bos[: pred_after_bos.index(cfg.eos_id)] if cfg.eos_id in pred_after_bos else pred_after_bos

        ok = pred_out == tgt_out
        if ok:
            correct += 1
        marker = "✓" if ok else "✗"
        print(f"  {marker} src={in_seq}  →  pred={pred_out}  (expected: {tgt_out})")

    print(f"\n[accuracy] {correct}/{src.size(0)} sequences reversed correctly")


if __name__ == "__main__":
    smoke_test()
