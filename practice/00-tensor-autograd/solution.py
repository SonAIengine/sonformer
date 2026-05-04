"""
Tensor + Autograd 기초 — single-file reference (정답지).

PyTorch의 진짜 바닥. 다음 두 가지를 손으로 확인:
    1. 텐서 연산 + broadcasting 동작
    2. autograd가 chain rule을 *정확히* 따라가는지 (손 gradient와 비교)

이걸 통과해야 위 layer (MLP/CNN/RNN/Transformer)가 블랙박스가 아니게 됨.

Run: python solution.py
"""

from __future__ import annotations

import torch


# ───────────────────────────────────────────────────────────────────────
# Part 1. 텐서 기초 — shape, dtype, broadcasting
# ───────────────────────────────────────────────────────────────────────

def part1_tensor_basics() -> None:
    print("\n[Part 1] 텐서 기초")
    print("─" * 60)

    # 생성
    a = torch.tensor([1.0, 2.0, 3.0])               # shape (3,), dtype float32
    b = torch.zeros(2, 3)                           # shape (2, 3), 0으로 채움
    c = torch.randn(2, 3)                           # 표준정규에서 샘플
    print(f"a = {a}, shape={a.shape}, dtype={a.dtype}")
    print(f"b shape: {b.shape}, c shape: {c.shape}")

    # ── Broadcasting ─────────────────────────────────────────────
    # shape이 달라도 마지막 차원부터 맞춰서 자동 확장.
    # 규칙: 각 차원이 같거나, 한 쪽이 1이면 OK.
    x = torch.ones(3, 1)            # (3, 1)
    y = torch.ones(1, 4)            # (1, 4)
    z = x + y                       # (3, 4) — 양쪽이 broadcast 됨
    print(f"\n(3,1) + (1,4) → broadcast → {z.shape}")

    # 흔한 함정: shape이 헷갈리면 명시적으로 unsqueeze/reshape 권장
    v = torch.tensor([1.0, 2.0, 3.0])      # (3,)
    M = torch.ones(3, 4)                    # (3, 4)
    # v.unsqueeze(1) → (3, 1) → broadcast하여 각 행에 v[i] 곱셈
    out = M * v.unsqueeze(1)
    print(f"(3,4) * (3,1) row-wise scale: {out.shape}, first row sum = {out[0].sum():.1f}")

    # device 이동
    if torch.cuda.is_available():
        a_gpu = a.to("cuda")
        print(f"\na를 GPU로: device = {a_gpu.device}")


# ───────────────────────────────────────────────────────────────────────
# Part 2. Autograd — gradient를 손으로 검증
# ───────────────────────────────────────────────────────────────────────

def part2_autograd_verify() -> None:
    """
    아주 간단한 함수에서 손 gradient ↔ autograd gradient 비교.

    함수: f(x) = x^2 · sin(x)
    수식: f'(x) = 2x · sin(x) + x^2 · cos(x)
    """
    print("\n[Part 2] autograd가 chain rule을 정확히 따라가는가")
    print("─" * 60)

    x = torch.tensor(1.5, requires_grad=True)       # leaf, gradient 추적 켜기
    y = x ** 2 * torch.sin(x)
    y.backward()                                     # x.grad에 dy/dx 채움

    manual = 2 * 1.5 * torch.sin(torch.tensor(1.5)).item() + 1.5 ** 2 * torch.cos(torch.tensor(1.5)).item()
    autograd = x.grad.item()
    print(f"x = 1.5 에서")
    print(f"  손 gradient   = {manual:.10f}")
    print(f"  autograd      = {autograd:.10f}")
    print(f"  차이          = {abs(manual - autograd):.2e}  ← 0에 가까우면 OK")
    assert abs(manual - autograd) < 1e-6, "autograd가 깨졌다 (있을 수 없는 일)"


# ───────────────────────────────────────────────────────────────────────
# Part 3. 선형 회귀 — gradient 손풀이 + autograd 동시 검증 + SGD 학습
# ───────────────────────────────────────────────────────────────────────

def part3_linear_regression() -> None:
    """
    데이터: y = 3x + 1 + noise
    모델:   ŷ = w·x + b
    손실:   L = mean((ŷ - y)^2)

    손 gradient:
        ∂L/∂w = (2/N) Σ (ŵx+b - y) · x
        ∂L/∂b = (2/N) Σ (ŵx+b - y)

    1) 손 vs autograd gradient 일치 확인
    2) SGD로 (w, b) → (3, 1) 수렴 확인
    """
    print("\n[Part 3] 선형 회귀: 손 gradient ↔ autograd 일치 + SGD 수렴")
    print("─" * 60)

    torch.manual_seed(42)
    N = 200
    x = torch.linspace(-2, 2, N)
    y = 3 * x + 1 + 0.1 * torch.randn(N)            # 약한 noise

    # 학습 파라미터 (둘 다 0에서 시작)
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # ── 1) gradient 일치 확인 (한 step만) ───────────────────────
    pred = w * x + b
    err = pred - y                                   # (N,)
    loss = (err ** 2).mean()
    loss.backward()

    # 손으로 푼 gradient
    manual_dw = 2.0 * (err * x).mean().item()
    manual_db = 2.0 * err.mean().item()

    print(f"  손 ∂L/∂w     = {manual_dw:.8f}")
    print(f"  autograd ∂w  = {w.grad.item():.8f}")
    print(f"  손 ∂L/∂b     = {manual_db:.8f}")
    print(f"  autograd ∂b  = {b.grad.item():.8f}")
    assert abs(manual_dw - w.grad.item()) < 1e-5
    assert abs(manual_db - b.grad.item()) < 1e-5
    print("  → 일치 ✓")

    # ── 2) SGD로 학습 ────────────────────────────────────────────
    print("\n  SGD 학습 (lr=0.1, 100 step):")
    lr = 0.1
    for step in range(100):
        # forward
        pred = w * x + b
        loss = ((pred - y) ** 2).mean()

        # backward — *주의: 이 전에 grad를 0으로 비워야 누적 안 됨*
        if w.grad is not None:
            w.grad.zero_()
            b.grad.zero_()
        loss.backward()

        # parameter update — 이 부분은 autograd가 추적하면 안 되므로 .data 또는 no_grad
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        if step % 20 == 0:
            print(f"    step {step:3d} | w = {w.item():.4f} | b = {b.item():.4f} | loss = {loss.item():.5f}")

    print(f"\n  최종: w = {w.item():.4f}  (정답 3.0)")
    print(f"        b = {b.item():.4f}  (정답 1.0)")


# ───────────────────────────────────────────────────────────────────────
# Part 4. 흔한 함정 3가지
# ───────────────────────────────────────────────────────────────────────

def part4_common_pitfalls() -> None:
    print("\n[Part 4] 흔한 함정")
    print("─" * 60)

    # 함정 1: grad 누적
    x = torch.tensor(2.0, requires_grad=True)
    (x ** 2).backward()
    print(f"  첫 backward 후 x.grad = {x.grad.item()}  (정답 4.0)")
    (x ** 2).backward()
    print(f"  두 번째 backward 후 x.grad = {x.grad.item()}  ★ 누적되어 8.0!")
    print(f"  → 학습 루프에서 항상 zero_grad() 필요\n")

    # 함정 2: 중간 텐서는 .grad 없음
    a = torch.tensor(3.0, requires_grad=True)
    b = a * 2                                        # 중간 텐서 (non-leaf)
    c = b ** 2
    c.backward()
    print(f"  leaf a.grad     = {a.grad.item()}  (정상)")
    print(f"  중간 b.grad     = {b.grad}  ★ None — 중간 텐서는 default로 grad 보존 안 됨")
    print(f"  → 보고 싶으면 b.retain_grad() 호출 필요\n")

    # 함정 3: in-place로 leaf 수정하면 그래프 깨짐
    w = torch.tensor(1.0, requires_grad=True)
    try:
        w += 1                                       # in-place — 그래프 추적 중인 leaf에 금지
        print("  in-place 수정 통과 (PyTorch 버전에 따라 다름)")
    except RuntimeError as e:
        print(f"  in-place 수정 → RuntimeError: {str(e)[:80]}...")
    print("  → with torch.no_grad(): 안에서 수정하거나 .data로 우회")


# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    part1_tensor_basics()
    part2_autograd_verify()
    part3_linear_regression()
    part4_common_pitfalls()
    print("\n" + "=" * 60)
    print("모두 통과. 다음: practice/01-mlp-from-scratch/")
    print("=" * 60)
