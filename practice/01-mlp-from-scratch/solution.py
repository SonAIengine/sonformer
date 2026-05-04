"""
MLP (Multi-Layer Perceptron) — single-file reference (정답지).

autograd 위에 처음 얹는 신경망. 다음 흐름을 한 파일에서 모두 다룸:

    nn.Module → nn.Linear + ReLU → CrossEntropyLoss → Adam → train/eval

학습 task: two-moons 2D 분류 (선형 분류기는 절대 못 푸는 데이터)
        → MLP의 비선형성이 진짜 필요하다는 걸 직관으로 체감.

Run: python solution.py        # ~3s on CPU
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
    n_samples: int = 1000
    noise: float = 0.15           # 데이터 노이즈 강도

    in_dim: int = 2               # 2D 점
    hidden: int = 32
    out_dim: int = 2              # 2-class 분류
    dropout: float = 0.0          # 작은 모델이라 0이 깔끔. 0.1로 바꿔 비교해도 됨.

    batch_size: int = 64
    learning_rate: float = 1e-2
    max_steps: int = 300
    log_interval: int = 50
    seed: int = 42


# ───────────────────────────────────────────────────────────────────────
# Two-moons 데이터 (sklearn 없이 직접)
# ───────────────────────────────────────────────────────────────────────

def make_moons(n: int, noise: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    초승달 모양 두 클래스를 평면에 배치.
    클래스 0: 위 반원 (원점 근처)
    클래스 1: 아래 반원, 살짝 오른쪽으로 평행이동
    """
    g = torch.Generator().manual_seed(seed)
    n0 = n // 2
    n1 = n - n0

    # 클래스 0: 위 반원
    t0 = torch.rand(n0, generator=g) * math.pi
    x0 = torch.stack([torch.cos(t0), torch.sin(t0)], dim=1)

    # 클래스 1: 아래 반원 + (1, -0.5) 이동
    t1 = torch.rand(n1, generator=g) * math.pi
    x1 = torch.stack([1 - torch.cos(t1), -torch.sin(t1) - 0.5], dim=1)

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)])

    # 가우시안 노이즈
    X = X + noise * torch.randn(X.shape, generator=g)

    # 셔플
    perm = torch.randperm(n, generator=g)
    return X[perm], y[perm]


# ───────────────────────────────────────────────────────────────────────
# MLP — nn.Module 표준 구조
# ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    2-layer ReLU MLP.

    nn.Module 의 두 가지 약속:
        1. __init__ 안에서 학습 파라미터 가진 모듈을 self.* 로 등록
           → PyTorch가 자동으로 .parameters()에 모음 (optimizer가 이걸 받음)
        2. forward 에서 입력 → 출력 흐름 정의
           → 호출은 model(x). __call__이 forward를 부르면서 hook 처리도 함.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, cfg.hidden)
        self.fc3 = nn.Linear(cfg.hidden, cfg.out_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ★ ReLU 없이 fc1 → fc2 → fc3 만 쌓으면 (W3 W2 W1)x + ... 로
        #    결국 한 개 Linear와 똑같다. 비선형성이 없으면 깊이가 의미 없음.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)                          # logits (softmax는 loss에 포함됨)


# ───────────────────────────────────────────────────────────────────────
# 학습 / 평가
# ───────────────────────────────────────────────────────────────────────

def train_one_step(model, X, y, opt) -> tuple[float, float]:
    model.train()                                    # dropout 등을 학습 모드로
    logits = model(X)
    loss = F.cross_entropy(logits, y)                # softmax + NLL을 한 번에
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return loss.item(), acc


@torch.no_grad()
def evaluate(model, X, y) -> tuple[float, float]:
    model.eval()                                     # dropout 끔
    logits = model(X)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


# ───────────────────────────────────────────────────────────────────────
# Decision boundary 시각화 (ASCII)
# ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def print_decision_boundary(model, X, y, grid: int = 30) -> None:
    """
    2D 평면을 grid×grid 로 잘라서 model이 각 점을 어느 클래스로 예측하는지 출력.
    .  = 클래스 0 영역
    #  = 클래스 1 영역
    o, x = 실제 데이터 포인트 (각각 클래스 0/1)
    """
    model.eval()
    x_min, x_max = X[:, 0].min().item() - 0.3, X[:, 0].max().item() + 0.3
    y_min, y_max = X[:, 1].min().item() - 0.3, X[:, 1].max().item() + 0.3

    xs = torch.linspace(x_min, x_max, grid)
    ys = torch.linspace(y_min, y_max, grid)

    print()
    for j in range(grid - 1, -1, -1):                # y는 위에서 아래로
        row = []
        for i in range(grid):
            pt = torch.tensor([[xs[i], ys[j]]])
            pred = model(pt).argmax(-1).item()
            row.append("#" if pred == 1 else ".")
        # 데이터 포인트 overlay
        for k in range(X.size(0)):
            xi = int((X[k, 0].item() - x_min) / (x_max - x_min) * (grid - 1) + 0.5)
            yi = int((X[k, 1].item() - y_min) / (y_max - y_min) * (grid - 1) + 0.5)
            if yi == j and 0 <= xi < grid:
                row[xi] = "x" if y[k].item() == 1 else "o"
        print("  " + "".join(row))


# ───────────────────────────────────────────────────────────────────────
# Smoke test
# ───────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # 학습/검증 데이터
    X_train, y_train = make_moons(cfg.n_samples, cfg.noise, seed=cfg.seed)
    X_test, y_test = make_moons(cfg.n_samples, cfg.noise, seed=cfg.seed + 1)

    model = MLP(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params} params")

    # Adam: 손으로 SGD 한 단계 위. lr 자동 적응.
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    t0 = time.time()
    for step in range(cfg.max_steps):
        # 배치 무작위 추출
        idx = torch.randint(0, cfg.n_samples, (cfg.batch_size,))
        Xb, yb = X_train[idx], y_train[idx]
        train_loss, train_acc = train_one_step(model, Xb, yb, opt)

        if step % cfg.log_interval == 0:
            test_loss, test_acc = evaluate(model, X_test, y_test)
            print(
                f"step {step:4d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"test loss {test_loss:.4f} acc {test_acc:.3f} | "
                f"{time.time() - t0:.1f}s"
            )

    # 최종 평가 + decision boundary 출력
    test_loss, test_acc = evaluate(model, X_test, y_test)
    print(f"\n[final] test acc = {test_acc:.3f}")

    print("\n[decision boundary] (. = class 0 영역, # = class 1 영역, o/x = 실제 데이터)")
    print_decision_boundary(model, X_test[:80], y_test[:80])


if __name__ == "__main__":
    smoke_test()
