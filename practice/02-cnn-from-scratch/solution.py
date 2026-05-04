"""
1D CNN for sequence classification (TextCNN) — single-file reference (정답지).

2014 Kim Yoon "Convolutional Neural Networks for Sentence Classification".
이미지 CNN의 텍스트판. kernel = "n-gram 패턴 detector".

핵심 흐름:
    token ids → embedding → (B, E, T)로 transpose
    → 여러 kernel_size의 Conv1d 병렬 적용 → ReLU
    → max-over-time pooling (시간축 전체에 max — "어디든 한 번 잡혔으면 OK")
    → concat → MLP → 분류

학습 포인트:
    - Conv1d 의 kernel은 그 자체로 "어떤 n-gram에 강하게 반응"하는 detector
    - max pooling은 위치 정보를 버리고 "있냐/없냐"만 남김
    - RNN과 달리 모든 시간 위치를 *병렬로* 처리 → GPU 활용 ↑
    - 한계: receptive field가 kernel_size에 묶임 → long-range는 약함

학습 task: bigram [1, 2]가 시퀀스 어딘가에 있으면 label=1, 없으면 0.
        kernel_size=2 짜리 filter 하나가 거의 완벽히 풂.

Run: python solution.py        # 약 800 step (~5s)
"""

from __future__ import annotations

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
    vocab_size: int = 8
    embed_dim: int = 32
    seq_len: int = 16
    kernel_sizes: tuple[int, ...] = (2, 3, 4)   # 여러 n-gram 동시 감지
    n_filters: int = 16                         # kernel_size 마다 16개 필터
    n_classes: int = 2                          # binary classification
    dropout: float = 0.2

    batch_size: int = 64
    learning_rate: float = 1e-3
    max_steps: int = 800
    log_interval: int = 100
    seed: int = 42

    # 데이터 task: 이 bigram이 있나?
    target_bigram: tuple[int, int] = (1, 2)


# ───────────────────────────────────────────────────────────────────────
# TextCNN
# ───────────────────────────────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    Multi-kernel 1D CNN classifier.

    각 kernel_size 마다 별도 Conv1d를 만들고, 결과를 max-pool 한 뒤 concat.
    "병렬로 여러 n-gram detector를 켜놓고, 가장 강하게 반응한 값만 통과"
    라는 직관.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)

        # kernel_size 마다 Conv1d 한 개씩
        # in_channels = embed_dim (각 시간 위치의 vector 차원)
        # out_channels = n_filters (이 크기의 필터 개수)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=cfg.embed_dim,
                out_channels=cfg.n_filters,
                kernel_size=k,
            )
            for k in cfg.kernel_sizes
        ])

        # 모든 kernel 결과 concat 후 차원 = n_filters × len(kernel_sizes)
        total_features = cfg.n_filters * len(cfg.kernel_sizes)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(total_features, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) 정수 토큰 ids
        return: (B, n_classes) logits
        """
        # 1) Embedding lookup → (B, T, E)
        e = self.embed(x)

        # 2) Conv1d는 channel-first 를 기대: (B, T, E) → (B, E, T)
        e = e.transpose(1, 2)

        # 3) 각 kernel 별로 conv + ReLU + max-over-time
        pooled = []
        for conv in self.convs:
            # conv(e): (B, n_filters, T - k + 1)
            h = F.relu(conv(e))
            # 시간축에 max: 시퀀스 어디서든 "가장 강한 활성화" 한 값
            # (B, n_filters, T') → (B, n_filters)
            h = h.max(dim=-1).values
            pooled.append(h)

        # 4) 모든 kernel 결과 concat → (B, n_filters × n_kernels)
        z = torch.cat(pooled, dim=-1)

        # 5) dropout + linear → 분류 logits
        return self.head(self.dropout(z))


# ───────────────────────────────────────────────────────────────────────
# Synthetic data: bigram 감지
# ───────────────────────────────────────────────────────────────────────

def make_bigram_batch(
    B: int,
    T: int,
    V: int,
    target: tuple[int, int],
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    각 시퀀스: V vocab에서 임의로 뽑은 길이 T 시퀀스.
    label = 1 이면 target bigram을 임의 위치에 강제 삽입.
    label = 0 이면 우연히 들어갈 수도 있지만 평균적으로 거의 없음 (V=8이면 1/64 확률).

    학습이 잘 되면 model이 "이 bigram이 있으면 1, 없으면 0"을 거의 완벽히 분류.
    """
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, 2, (B,))                                # 50/50 label

    # label=1인 샘플에 target bigram을 임의 위치에 삽입
    for i in range(B):
        if y[i] == 1:
            pos = torch.randint(0, T - 1, (1,)).item()
            x[i, pos] = target[0]
            x[i, pos + 1] = target[1]
        else:
            # label=0인데 우연히 target bigram이 있으면 깨버림 (clean label)
            for p in range(T - 1):
                if x[i, p].item() == target[0] and x[i, p + 1].item() == target[1]:
                    x[i, p + 1] = (target[1] + 1) % V

    return x.to(device), y.to(device)


# ───────────────────────────────────────────────────────────────────────
# Smoke test
# ───────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[task] bigram {cfg.target_bigram} 감지 (시퀀스 길이 {cfg.seq_len})")

    model = TextCNN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    losses: list[float] = []
    accs: list[float] = []
    t0 = time.time()
    for step in range(cfg.max_steps):
        x, y = make_bigram_batch(
            cfg.batch_size, cfg.seq_len, cfg.vocab_size, cfg.target_bigram, device
        )
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        accs.append((logits.argmax(-1) == y).float().mean().item())
        if step % cfg.log_interval == 0:
            avg_l = sum(losses[-cfg.log_interval:]) / max(1, len(losses[-cfg.log_interval:]))
            avg_a = sum(accs[-cfg.log_interval:]) / max(1, len(accs[-cfg.log_interval:]))
            print(f"step {step:4d} | loss {avg_l:.4f} | acc {avg_a:.3f} | {time.time()-t0:.1f}s")

    # 최종 평가
    model.eval()
    with torch.no_grad():
        x, y = make_bigram_batch(
            512, cfg.seq_len, cfg.vocab_size, cfg.target_bigram, device
        )
        pred = model(x).argmax(-1)
        acc = (pred == y).float().mean().item()
    print(f"\n[final] bigram detection accuracy on 512 samples: {acc:.3f}")
    print(f"        (random chance = 0.500)")


if __name__ == "__main__":
    smoke_test()
