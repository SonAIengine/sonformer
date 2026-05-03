"""
BERT — Encoder-only Transformer with MLM pre-training. Single-file reference.

Family:
    decoder-only (sonlm)        → autoregressive 생성
    encoder-decoder (Transformer) → 입력→출력 매핑 (번역/요약)
    encoder-only (BERT, this)   → 입력 표현 학습 (분류/임베딩/MLM)

Key differences from the other two:
    - Bidirectional self-attention (NO causal mask)
    - MLM pre-training: 15% 위치를 masking → 원래 토큰 예측
    - [CLS] 토큰의 hidden을 분류 task에 활용 (pooled output)
    - GELU FFN, learned PE, Embedding 뒤 LayerNorm (BERT 컨벤션)

Smoke task: bigram cycle MLM
    cycle: [4, 5, 6, ..., V-1, 4, 5, ...] of period V-4
    sequence: random window of length T from cycle
    mask 15% positions → predict from bidirectional context

Run:
    python solution.py        # ~10s on H100, MLM accuracy → ~100% on cycle task
"""

from __future__ import annotations

import math
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
    # Vocab + special tokens (BERT 표준)
    vocab_size: int = 32
    pad_id: int = 0
    cls_id: int = 1                # [CLS] (sequence-level representation)
    sep_id: int = 2                # [SEP] (segment separator)
    mask_id: int = 3                # [MASK] (MLM target marker)

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ffn_hidden: int = 512           # 4× d_model
    max_seq_len: int = 16
    dropout: float = 0.1
    type_vocab_size: int = 2        # segment A vs B (NSP용 — 여기선 사용 안 함, 구조만 보임)

    # MLM masking
    mask_prob: float = 0.15
    mask_replace_prob: float = 0.8  # 80% [MASK]
    mask_random_prob: float = 0.1   # 10% random token (10% keep original)

    # Training
    batch_size: int = 64
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    max_steps: int = 1500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 100
    seed: int = 42


# ───────────────────────────────────────────────────────────────────────
# Embeddings — BERT 스타일 (token + position + segment, 합친 뒤 LN)
# ───────────────────────────────────────────────────────────────────────

class Embeddings(nn.Module):
    """token + position + (optional) segment → LayerNorm → dropout."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)        # learned PE
        self.seg_emb = nn.Embedding(cfg.type_vocab_size, cfg.d_model)    # segment A/B
        self.norm = nn.LayerNorm(cfg.d_model)                             # BERT는 합 직후 LN
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, ids: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        if segment_ids is None:
            segment_ids = torch.zeros_like(ids)
        x = self.tok_emb(ids) + self.pos_emb(pos) + self.seg_emb(segment_ids)
        return self.dropout(self.norm(x))


# ───────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention (no causal — BERT는 bidirectional)
# ───────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))            # padding only — NO causal
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ───────────────────────────────────────────────────────────────────────
# FFN — BERT는 GELU 사용
# ───────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))                # GELU (not ReLU)


# ───────────────────────────────────────────────────────────────────────
# Encoder Layer — Pre-LN (원조 BERT는 Post-LN. Pre-LN이 학습 안정성 ↑)
# ───────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ───────────────────────────────────────────────────────────────────────
# BERT — Encoder-only + MLM head + (optional) classifier head
# ───────────────────────────────────────────────────────────────────────

class BERT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embeddings = Embeddings(cfg)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

        # MLM head: hidden → dense → GELU → LN → tied output projection
        # (HuggingFace BERT의 cls.predictions.transform + decoder 구조)
        self.mlm_dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.mlm_norm = nn.LayerNorm(cfg.d_model)
        self.mlm_decoder = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        self.mlm_decoder.weight = self.embeddings.tok_emb.weight          # weight tying

        # Pooler (for classification): CLS hidden → tanh(linear)
        self.pooler = nn.Linear(cfg.d_model, cfg.d_model)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns hidden states (B, T, C)."""
        x = self.embeddings(ids, segment_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def mlm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, T, V) — MLM token predictions."""
        x = F.gelu(self.mlm_dense(hidden))
        x = self.mlm_norm(x)
        return self.mlm_decoder(x)

    def pooled_output(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, C) — [CLS] hidden을 tanh(linear)로. 분류 task에 사용."""
        return torch.tanh(self.pooler(hidden[:, 0]))


# ───────────────────────────────────────────────────────────────────────
# Mask utility
# ───────────────────────────────────────────────────────────────────────

def make_padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, 1, 1, T) — 1=attend, 0=ignore. NO causal — BERT는 bidirectional."""
    return (ids != pad_id)[:, None, None, :].long()


# ───────────────────────────────────────────────────────────────────────
# MLM masking — 15% selected, 80% [MASK] / 10% random / 10% keep
# ───────────────────────────────────────────────────────────────────────

def apply_mlm_masking(ids: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        masked_ids: ids에서 일부 위치를 [MASK]/random/keep으로 변환한 입력
        labels:     원래 토큰 (선택 안 된 위치는 -100 → cross_entropy ignore_index)

    BERT 원조 recipe (15% select, then 80/10/10 split) — 섞인 학습이 robustness ↑
    """
    labels = ids.clone()
    prob = torch.full(ids.shape, cfg.mask_prob, device=ids.device)
    selected = torch.bernoulli(prob).bool()
    labels[~selected] = -100                                             # 미선택 위치는 loss 무시

    rand = torch.rand(ids.shape, device=ids.device)
    masked_ids = ids.clone()

    # 80% → [MASK]
    replace_mask = selected & (rand < cfg.mask_replace_prob)
    masked_ids[replace_mask] = cfg.mask_id

    # 10% → random token (special token 회피해서 4..V-1 범위)
    replace_rand = selected & (rand >= cfg.mask_replace_prob) & (
        rand < cfg.mask_replace_prob + cfg.mask_random_prob
    )
    random_tokens = torch.randint(4, cfg.vocab_size, ids.shape, device=ids.device)
    masked_ids[replace_rand] = random_tokens[replace_rand]

    # 10% → 그대로 유지 (이미 ids 값이라 변경 없음)
    return masked_ids, labels


# ───────────────────────────────────────────────────────────────────────
# Synthetic task — bigram cycle MLM
# ───────────────────────────────────────────────────────────────────────

def make_cycle_batch(batch_size: int, cfg: Config, device) -> torch.Tensor:
    """
    cycle: [4, 5, 6, ..., V-1, 4, 5, ...] of period (V - 4)
    각 sample: random start에서 max_seq_len개 연속 토큰
    이 패턴이면 마스크된 위치를 양쪽 이웃으로부터 예측 가능 (bidirectional 가치 입증)
    """
    period = cfg.vocab_size - 4
    starts = torch.randint(0, period, (batch_size,))
    offsets = torch.arange(cfg.max_seq_len)
    seqs = ((starts.unsqueeze(1) + offsets) % period) + 4
    return seqs.to(device)


# ───────────────────────────────────────────────────────────────────────
# LR schedule + smoke test
# ───────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def smoke_test() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    model = BERT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params ({n_params / 1e6:.2f}M)")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),                                              # BERT 원조 값
    )

    losses: list[float] = []
    t0 = time.time()
    for step in range(cfg.max_steps):
        ids = make_cycle_batch(cfg.batch_size, cfg, device)
        masked_ids, labels = apply_mlm_masking(ids, cfg)
        # padding 없는 task — mask 생략 (또는 모두 1)
        hidden = model(masked_ids)
        logits = model.mlm_logits(hidden)

        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,                                           # 선택 안 된 위치 무시
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

    # ── Final evaluation ───────────────────────────────────────────────
    print("\n[final] testing MLM prediction...")
    model.eval()
    ids = make_cycle_batch(8, cfg, device)
    masked_ids, labels = apply_mlm_masking(ids, cfg)
    with torch.no_grad():
        logits = model.mlm_logits(model(masked_ids))
        preds = logits.argmax(-1)

    masked_positions = labels != -100
    correct = (preds == labels) & masked_positions
    acc = correct.sum().item() / max(1, masked_positions.sum().item())

    # 첫 두 샘플 시각화
    for i in range(2):
        orig = ids[i].tolist()
        masked = masked_ids[i].tolist()
        pred = preds[i].tolist()
        marks = ["✓" if labels[i, j] != -100 and preds[i, j] == labels[i, j] else
                 "✗" if labels[i, j] != -100 else " " for j in range(len(orig))]
        print(f"  sample {i}:")
        print(f"    orig:   {orig}")
        print(f"    masked: {masked}    ([MASK]={cfg.mask_id})")
        print(f"    pred:   {pred}")
        print(f"    marks:  {marks}    (✓=맞춘 마스크, ✗=틀린 마스크, 공백=마스크 아님)")

    print(f"\n[accuracy] MLM masked-position accuracy: {acc:.2%}")


if __name__ == "__main__":
    smoke_test()
