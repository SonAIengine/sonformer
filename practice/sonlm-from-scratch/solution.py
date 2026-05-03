"""
sonlm — single-file reference implementation (정답지).

End-to-end vanilla decoder-only transformer LM in one file.
Educational reference for hand-coding exercises in this folder.

Pipeline: JSONL → BPE → (x, y) shift → SonLM → cross-entropy → AdamW → generate.

Usage:
    python solution.py                                              # smoke test (random data, ~5s)
    python solution.py train <train.jsonl> <val.jsonl> <tk.json>    # real training

먼저 본인 손으로 (config.py / model.py / dataset.py / train.py)를 만들어보고,
막힐 때만 이 파일을 펴서 막힌 부분을 비교. 통째로 베끼지 말 것.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ───────────────────────────────────────────────────────────────────────
# Config — 모델/학습 파라미터 한 곳에
# ───────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    vocab_size: int = 4096
    max_seq_len: int = 128
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_hidden: int = 768       # sonlm/ 컨벤션 (2× d_model). 4× 도 흔함.
    dropout: float = 0.1
    pad_id: int = 0

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 1500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    log_interval: int = 50
    seed: int = 42


# ───────────────────────────────────────────────────────────────────────
# Model — Attention / FFN / Block / SonLM
# ───────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head self-attention. Fused QKV projection."""

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
        # (B, T, 3C) → (B, T, 3, H, D) → (3, B, H, T, D)
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                         # each (B, H, T, D)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))    # causal
        attn = self.dropout(F.softmax(attn, dim=-1))             # (B, H, T, T)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FFN(nn.Module):
    """2-layer ReLU MLP."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.relu(self.up(x))))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SonLM(nn.Module):
    """Vanilla decoder-only transformer LM."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                # weight tying

        # Causal mask buffer — 매 forward 재생성 안 하도록 캐시
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))[None, None]
        self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))     # (B, T, C)
        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)
        logits = self.lm_head(self.norm(x))                       # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=self.cfg.pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.7, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature              # 마지막 위치만
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ───────────────────────────────────────────────────────────────────────
# Data — JSONL ({"text": "..."}) → (x, y) 한 칸 shift → padding
# ───────────────────────────────────────────────────────────────────────

class JSONLDataset(Dataset):
    """JSONL each line {"text": "..."}. Tokenize at load."""

    def __init__(self, path: str, tokenizer, max_len: int):
        self.samples = []
        with open(path) as f:
            for line in f:
                ids = tokenizer.encode(json.loads(line)["text"]).ids
                if len(ids) > max_len:
                    ids = ids[:max_len]
                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids = self.samples[i]
        # ⭐ 한 칸 shift: x[i]를 보고 y[i] (= x[i+1]) 예측하도록 학습
        return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long)


def collate_fn(batch, pad_id: int = 0):
    """Pad each sample in batch to max length with pad_id."""
    xs, ys = zip(*batch)
    L = max(len(x) for x in xs)
    px = torch.full((len(xs), L), pad_id, dtype=torch.long)
    py = torch.full((len(ys), L), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        px[i, :len(x)] = x
        py[i, :len(y)] = y
    return px, py


# ───────────────────────────────────────────────────────────────────────
# Training — warmup + cosine LR, AdamW, optional AMP
# ───────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: Config) -> float:
    """Linear warmup → cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def train(cfg: Config, train_loader: DataLoader, val_loader: DataLoader | None = None,
          device: str = "cuda") -> SonLM:
    torch.manual_seed(cfg.seed)
    model = SonLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params ({n_params / 1e6:.2f}M)")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    step = 0
    losses: list[float] = []
    t0 = time.time()
    model.train()
    while step < cfg.max_steps:
        for x, y in train_loader:
            if step >= cfg.max_steps:
                break
            x, y = x.to(device), y.to(device)
            lr = get_lr(step, cfg)
            for pg in opt.param_groups:
                pg["lr"] = lr

            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if step % cfg.log_interval == 0:
                window = losses[-cfg.log_interval:]
                avg = sum(window) / max(1, len(window))
                print(f"step {step:6d} | lr {lr:.6f} | train {avg:.4f} | {time.time() - t0:.1f}s")

            if val_loader is not None and step > 0 and step % cfg.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss, n = 0.0, 0
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        _, vl = model(vx, vy)
                        val_loss += vl.item()
                        n += 1
                        if n >= 30:
                            break
                model.train()
                avg_val = val_loss / max(1, n)
                print(f"          eval {avg_val:.4f} | ppl {math.exp(avg_val):.2f}")
            step += 1

    print(f"[done] {time.time() - t0:.1f}s")
    return model


# ───────────────────────────────────────────────────────────────────────
# Entry — smoke test (random data, no real dataset needed)
# ───────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    """Verify forward + backward with random data — runs in ~5s on any GPU."""
    cfg = Config(
        vocab_size=64, max_seq_len=16, d_model=32, n_layers=2, n_heads=4,
        ffn_hidden=64, max_steps=20, warmup_steps=5, log_interval=5,
        eval_interval=999, seed=0, batch_size=8,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    torch.manual_seed(0)
    raw = [torch.randint(1, cfg.vocab_size, (cfg.max_seq_len + 1,)) for _ in range(64)]

    class FakeDS(Dataset):
        def __len__(self):
            return len(raw)

        def __getitem__(self, i):
            return raw[i][:-1], raw[i][1:]

    loader = DataLoader(
        FakeDS(),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, cfg.pad_id),
    )
    train(cfg, loader, device=device)


def real_train(train_path: str, val_path: str, tk_path: str) -> None:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tk_path)
    cfg = Config(vocab_size=tk.get_vocab_size())
    train_ds = JSONLDataset(train_path, tk, cfg.max_seq_len)
    val_ds = JSONLDataset(val_path, tk, cfg.max_seq_len)
    print(f"[data] train={len(train_ds):,}  val={len(val_ds):,}")
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, cfg.pad_id))
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, cfg.pad_id))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(cfg, train_ld, val_ld, device=device)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        if len(sys.argv) < 5:
            print("Usage: python solution.py train <train.jsonl> <val.jsonl> <tokenizer.json>")
            sys.exit(1)
        real_train(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        smoke_test()
