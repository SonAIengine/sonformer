"""
Vanilla RNN (Elman) — single-file reference (정답지).

Transformer 이전 시퀀스 모델의 모태. 시간을 따라 hidden state h_t를 갱신:

    h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)     # 기억 갱신
    y_t = W_hy · h_t + b_y                            # 출력

핵심 학습 포인트:
    - 모든 step이 같은 가중치(W_xh, W_hh) 공유 — parameter sharing
    - hidden state h가 "지금까지 본 모든 것의 압축된 요약" 역할
    - 한계: 시퀀스가 길어지면 gradient가 vanish/explode → LSTM이 풀려고 한 문제

학습 task: 작은 어휘에서 반복 패턴 char-LM (next-token = (cur+1) % V).
        1-step 의존성이라 vanilla RNN이 잘 풀어야 함.

Run: python solution.py        # 약 1000 step에 수렴 (~10s)
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
    vocab_size: int = 8           # 작은 어휘 (synthetic char set)
    hidden_size: int = 64
    seq_len: int = 16             # 학습 시 시퀀스 길이
    batch_size: int = 32
    learning_rate: float = 5e-3
    max_steps: int = 1000
    log_interval: int = 100
    grad_clip: float = 1.0
    seed: int = 42


# ───────────────────────────────────────────────────────────────────────
# RNN Cell — 단일 time step
# ───────────────────────────────────────────────────────────────────────

class RNNCell(nn.Module):
    """
    한 토큰을 받아 hidden state를 업데이트하는 단위.
    이게 모든 시간 step에서 *같은 인스턴스로* 재사용된다 (weight sharing).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        # x → hidden 으로 가는 입력 변환 (bias는 W_hh 쪽에만)
        self.W_xh = nn.Linear(cfg.vocab_size, cfg.hidden_size, bias=False)
        # hidden → hidden 으로 가는 재귀 변환 (bias 포함)
        self.W_hh = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def forward(self, x_onehot: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x_onehot: (B, V), h_prev: (B, H)
        # 두 선형변환을 더한 뒤 tanh로 [-1, 1] 범위로 squash
        return torch.tanh(self.W_xh(x_onehot) + self.W_hh(h_prev))


# ───────────────────────────────────────────────────────────────────────
# RNN Language Model — manual time unroll
# ───────────────────────────────────────────────────────────────────────

class RNNLM(nn.Module):
    """
    char-level LM. PyTorch의 nn.RNN을 안 쓰고 손으로 펼쳐서 학습 의도 명확화.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.cell = RNNCell(cfg)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) 정수 토큰 ids
        return: (B, T, V) — 각 step에서 다음 토큰 예측 logits
        """
        B, T = x.shape
        # 초기 hidden은 0. (실전에선 학습 가능한 파라미터로 두기도 함)
        h = torch.zeros(B, self.cfg.hidden_size, device=x.device)
        outs = []
        for t in range(T):
            # 정수 ID → one-hot 벡터로 변환 (간단히 하려고 embedding 대신)
            x_t = F.one_hot(x[:, t], self.cfg.vocab_size).float()      # (B, V)
            h = self.cell(x_t, h)                                       # (B, H)
            outs.append(self.head(h))                                   # (B, V)
        return torch.stack(outs, dim=1)                                 # (B, T, V)

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, max_new: int) -> torch.Tensor:
        """
        prefix 토큰을 모두 흘려 hidden을 만든 뒤, 거기서 한 토큰씩 argmax 생성.

        ★ 주의: 흔한 버그가 "warm-up으로 prefix를 다 흘린 뒤,
                다시 prefix[-1]을 또 cell에 먹이고 예측"하는 것.
                그러면 마지막 prefix 토큰이 두 번 흡수돼서 분포가 망가짐.
                올바른 패턴은: warm-up 후 h에서 *바로* 다음 토큰 예측,
                그 prediction을 다음 입력으로 사용.
        """
        self.eval()
        B = prefix.size(0)
        h = torch.zeros(B, self.cfg.hidden_size, device=prefix.device)

        # 1) prefix 전체를 cell에 흘림 — h는 prefix 전부를 본 상태
        for t in range(prefix.size(1)):
            x_t = F.one_hot(prefix[:, t], self.cfg.vocab_size).float()
            h = self.cell(x_t, h)

        # 2) 현재 h에서 다음 토큰을 예측 → 그 prediction을 다시 입력으로
        out = [prefix]
        for _ in range(max_new):
            next_id = self.head(h).argmax(-1, keepdim=True)         # (B, 1)
            out.append(next_id)
            x_t = F.one_hot(next_id.squeeze(-1), self.cfg.vocab_size).float()
            h = self.cell(x_t, h)
        return torch.cat(out, dim=1)


# ───────────────────────────────────────────────────────────────────────
# Synthetic data: 반복 패턴
# ───────────────────────────────────────────────────────────────────────

def make_pattern_batch(B: int, T: int, V: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    각 시퀀스: 임의 시작점 s에서 [s, s+1, s+2, ...] mod V 로 진행.
    next-token = (cur + 1) % V — 1-step 의존성이라 RNN으로 충분히 풀림.
    """
    starts = torch.randint(0, V, (B,))                              # (B,)
    # broadcasting으로 한 줄: (B, 1) + (1, T+1) → (B, T+1)
    seq = (starts.unsqueeze(1) + torch.arange(T + 1).unsqueeze(0)) % V
    x = seq[:, :-1]         # 입력
    y = seq[:, 1:]          # 정답 (한 칸 shift)
    return x.to(device), y.to(device)


# ───────────────────────────────────────────────────────────────────────
# Smoke test
# ───────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    model = RNNLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    losses: list[float] = []
    t0 = time.time()
    for step in range(cfg.max_steps):
        x, y = make_pattern_batch(cfg.batch_size, cfg.seq_len, cfg.vocab_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # RNN은 BPTT에서 gradient가 폭발하기 쉬움 → clip 필수
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(loss.item())
        if step % cfg.log_interval == 0:
            avg = sum(losses[-cfg.log_interval:]) / max(1, len(losses[-cfg.log_interval:]))
            print(f"step {step:4d} | loss {avg:.4f} | {time.time()-t0:.1f}s")

    # 패턴을 진짜 학습했는지 generate로 확인
    print("\n[gen] 0,1,2 다음에 무엇이 올까?")
    prefix = torch.tensor([[0, 1, 2]], device=device)
    out = model.generate(prefix, max_new=10)
    print(f"     model: {out.tolist()[0]}")
    print(f"     정답:  [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4]")


if __name__ == "__main__":
    smoke_test()
