"""
LSTM (Long Short-Term Memory) — single-file reference (정답지).

Vanilla RNN의 vanishing gradient를 완화하기 위한 1997년 구조.
hidden state h_t 외에 cell state c_t를 별도 채널로 두고, 4개의 gate로
정보 흐름을 동적 제어.

핵심 식:
    i_t = σ(W_i · [x; h_{t-1}])    input gate     — 새 정보 얼마나 쓸까
    f_t = σ(W_f · [x; h_{t-1}])    forget gate    — 과거 cell 얼마나 유지할까
    o_t = σ(W_o · [x; h_{t-1}])    output gate    — cell → hidden 얼마나 내보낼까
    g_t = tanh(W_g · [x; h_{t-1}]) candidate      — 새로 추가할 후보 정보

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        ★ cell state (장기 기억)
    h_t = o_t ⊙ tanh(c_t)                  hidden state (외부로 노출되는 출력)

★ 왜 vanishing gradient에 강한가:
    c_t는 c_{t-1}에 forget gate만 곱하고 + 새 정보를 *더하는* 구조.
    forget=1 일 때 c는 사실상 항등 통과 → gradient가 길게 흘러도 살아남음.

학습 task: delayed copy. [a, X, X, ..., X, ?] 마지막에 a를 복원.
        길이를 늘려가면 vanilla RNN은 깨지지만 LSTM은 비교적 견딤.

Run: python solution.py        # 약 2000 step (~15s)
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
    vocab_size: int = 8           # {0=NOISE, 1~7=실제 신호}
    hidden_size: int = 64
    seq_len: int = 20             # 신호 + noise 길이 (RNN보다 더 긴 의존성 시험)
    batch_size: int = 32
    learning_rate: float = 5e-3
    max_steps: int = 2000
    log_interval: int = 100
    grad_clip: float = 1.0
    seed: int = 42


# ───────────────────────────────────────────────────────────────────────
# LSTM Cell — 4 gate, fused 한 번 행렬곱
# ───────────────────────────────────────────────────────────────────────

class LSTMCell(nn.Module):
    """
    4 gate를 따로 4번 연산하는 대신, 한 번에 4×H 차원으로 뽑아서 split하면
    빠르고 코드도 짧아진다 (PyTorch 내장 LSTMCell도 같은 방식).

    [x_t; h_{t-1}]을 입력으로 받아 (i, f, g, o) 4개 vector 동시 계산.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        # 입력 (x: V) + 이전 hidden (h: H) → 4×H (4개 gate 한 번에)
        self.gates = nn.Linear(cfg.vocab_size + cfg.hidden_size, 4 * cfg.hidden_size)

        # ★ LSTM 표준 트릭: forget gate bias를 1로 초기화.
        # 이유: bias=0이면 sigmoid(W·x + 0) ≈ 0.5 → cell이 매 step마다 절반씩 지워짐
        #       → vanishing 재발. bias=1이면 sigmoid(1) ≈ 0.73 으로 시작해
        #       "기본적으로 기억하자"는 prior를 학습 초기에 박아둠.
        # gates 출력 순서: [i, f, g, o] 이므로 두번째 chunk가 forget bias.
        with torch.no_grad():
            H = cfg.hidden_size
            self.gates.bias[H : 2 * H].fill_(1.0)

    def forward(
        self,
        x_onehot: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x_onehot: (B, V), h_prev: (B, H), c_prev: (B, H)

        # 입력과 이전 hidden 을 concat → 한 번의 행렬곱으로 4 gate 모두 계산
        z = self.gates(torch.cat([x_onehot, h_prev], dim=-1))    # (B, 4H)
        i, f, g, o = z.chunk(4, dim=-1)                           # 각 (B, H)

        # gate 활성화: i/f/o는 sigmoid (0~1, "얼마나 통과"), g는 tanh (-1~1, "내용물")
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # ★ 핵심 한 줄: cell state 갱신
        # 이전 cell × forget (지울 거 지움) + 새 후보 × input (받을 거 받음)
        c_new = f * c_prev + i * g

        # hidden은 cell을 tanh로 squash 한 뒤 output gate로 필터링
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


# ───────────────────────────────────────────────────────────────────────
# Delayed-copy classifier (마지막 step에서 첫 토큰을 분류)
# ───────────────────────────────────────────────────────────────────────

class LSTMCopier(nn.Module):
    """
    시퀀스 끝 hidden 으로 "첫 토큰이 무엇이었나"를 분류.
    cell state가 첫 토큰 정보를 시퀀스 끝까지 운반하는 능력 시험.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.cell = LSTMCell(cfg)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) 정수 토큰 ids
        return: (B, V) — 마지막 step의 logits (첫 토큰 분류)
        """
        B, T = x.shape
        H = self.cfg.hidden_size
        h = torch.zeros(B, H, device=x.device)
        c = torch.zeros(B, H, device=x.device)
        for t in range(T):
            x_t = F.one_hot(x[:, t], self.cfg.vocab_size).float()       # (B, V)
            h, c = self.cell(x_t, h, c)
        return self.head(h)                                              # (B, V)


# ───────────────────────────────────────────────────────────────────────
# Synthetic data: delayed copy
# ───────────────────────────────────────────────────────────────────────

def make_copy_batch(B: int, T: int, V: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    각 시퀀스: [signal, NOISE, NOISE, ..., NOISE]
        - position 0: 1~V-1 중 임의 (실제 신호 토큰)
        - position 1..T-1: 모두 0 (NOISE)
    target: signal (position 0의 값)을 복원

    LSTM은 signal을 cell state에 forget gate로 보존해 끝까지 운반해야 함.
    """
    signal = torch.randint(1, V, (B,))                  # 1 ~ V-1 (0은 noise 표시용)
    x = torch.zeros(B, T, dtype=torch.long)
    x[:, 0] = signal
    return x.to(device), signal.to(device)


# ───────────────────────────────────────────────────────────────────────
# Smoke test
# ───────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[task] delayed copy: 첫 토큰을 길이 {cfg.seq_len} 시퀀스 끝에서 복원")

    model = LSTMCopier(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    losses: list[float] = []
    accs: list[float] = []
    t0 = time.time()
    for step in range(cfg.max_steps):
        x, y = make_copy_batch(cfg.batch_size, cfg.seq_len, cfg.vocab_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
        x, y = make_copy_batch(256, cfg.seq_len, cfg.vocab_size, device)
        pred = model(x).argmax(-1)
        acc = (pred == y).float().mean().item()
    print(f"\n[final] copy accuracy on 256 samples: {acc:.3f}")
    print(f"        (random chance = {1.0 / (cfg.vocab_size - 1):.3f})")


if __name__ == "__main__":
    smoke_test()
