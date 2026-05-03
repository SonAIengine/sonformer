"""
GuppyLM training loop.

────────────────────────────────────────────────────────────────────────────
[학습 루프 한 눈에 보기]

  매 step마다 반복:
      1) DataLoader에서 (x, y) 배치 꺼내기            ← dataset.py
      2) GPU로 옮기기
      3) 이번 step의 learning rate 계산 (warmup+cosine)
      4) Forward: logits, loss = model(x, y)         ← model.py
      5) Backward: loss.backward()                   ← gradient 계산
      6) Gradient clipping (폭주 방지)
      7) Optimizer step (실제로 weight 업데이트)
      8) zero_grad (다음 step을 위해 gradient 초기화)
      9) 주기적으로 평가 + 체크포인트 저장

핵심 트릭:
  - Warmup + Cosine LR:  학습 초반엔 천천히 올렸다가 나중엔 부드럽게 내림
  - AMP (mixed precision): float16으로 계산해서 GPU 메모리/속도 ↑
  - Grad clip:            gradient norm을 1.0으로 제한해 학습 안정성 ↑
  - AdamW + weight decay: GPT 계열 표준 옵티마이저 설정
────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import os
import time

import torch

from .config import GuppyConfig, TrainConfig
from .dataset import get_dataloader
from .model import GuppyLM


def get_device(config):
    """CUDA > MPS(애플 실리콘) > CPU 순으로 자동 선택."""
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


def get_lr(step, config):
    """
    Learning rate 스케줄: Linear warmup + Cosine decay.

      lr
       │      ╭─────╮
       │     ╱       ╲___
       │    ╱             ╲___
       │   ╱                  ╲___
       │  ╱                       ╲___ → min_lr
       └──┴────────────────────────────── step
        warmup       cosine decay

    - 0 ~ warmup_steps: 0에서 learning_rate까지 선형 증가
        이유: 초기에 random weight는 gradient가 거칠어서, 큰 LR로 시작하면 발산.
    - 그 이후: cosine 곡선을 따라 min_lr까지 부드럽게 감소
        이유: 후반엔 작은 step으로 미세조정이 잘 됨.
    """
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps

    # warmup 끝난 뒤의 진행도 (0.0 → 1.0)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    # cos(0)=1, cos(π)=-1 이므로 coeff는 1.0 → 0.0으로 부드럽게 감소
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * coeff


@torch.no_grad()   # 평가 중엔 gradient 계산 안 함 (메모리/속도 ↑)
def evaluate(model, loader, device, max_batches=50):
    """
    Eval set에서 평균 loss 측정.
    eval mode로 전환 (dropout 끔) → loss 계산 → 다시 train mode로 복원.
    """
    model.eval()                                    # dropout, BN 등 평가 모드로
    total_loss, n = 0, 0
    for x, y in loader:
        if n >= max_batches:                        # 전체 eval set이 크면 일부만 샘플링
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)                       # forward만 (backward 없음)
        total_loss += loss.item()
        n += 1
    model.train()                                   # 학습 모드 복귀 (중요!)
    return total_loss / max(1, n)


def train():
    # ── 0) 설정 로드 ────────────────────────────────────────────────────────
    mc = GuppyConfig()                              # 모델 하이퍼파라미터
    tc = TrainConfig()                              # 학습 하이퍼파라미터
    device = get_device(tc)
    torch.manual_seed(tc.seed)                      # 재현성을 위한 시드 고정

    print(f"Device: {device}")

    # ── 1) 모델 생성 ────────────────────────────────────────────────────────
    tokenizer_path = os.path.join(tc.data_dir, "tokenizer.json")
    model = GuppyLM(mc).to(device)                  # 모델을 GPU(또는 CPU)로 이동
    print(model.param_summary())

    # ── 2) DataLoader 생성 (train / eval) ───────────────────────────────────
    train_loader = get_dataloader(
        os.path.join(tc.data_dir, "train.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, shuffle=True,    # 학습은 매 epoch마다 섞기
    )
    eval_loader = get_dataloader(
        os.path.join(tc.data_dir, "eval.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, shuffle=False,   # 평가는 순서 고정
    )
    print(f"Train: {len(train_loader.dataset):,}, Eval: {len(eval_loader.dataset):,}")

    # ── 3) Optimizer ───────────────────────────────────────────────────────
    # AdamW: Adam + weight decay 분리 (GPT-2/3 표준).
    # betas=(0.9, 0.95): GPT 계열에서 쓰는 값. 기본값 (0.9, 0.999)보다 second moment를
    #                    빠르게 적응시켜 큰 모델 학습에 유리.
    # weight_decay=0.1: L2 정규화 효과 (overfitting 방지).
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate,
        weight_decay=tc.weight_decay, betas=(0.9, 0.95),
    )

    # ── 4) Mixed Precision 설정 (CUDA only) ─────────────────────────────────
    # AMP = Automatic Mixed Precision.
    # float32 대신 float16/bfloat16으로 계산해서 GPU 메모리 절반, 속도 1.5~2배.
    # 단, float16은 표현 범위가 좁아서 gradient가 너무 작으면 0으로 underflow됨.
    # → GradScaler가 loss에 큰 수를 곱해 gradient를 키워두고 step 직전에 다시 나눔.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # 체크포인트 폴더 + config 저장 (나중에 추론할 때 같은 설정으로 모델 재구성하려고)
    os.makedirs(tc.output_dir, exist_ok=True)
    with open(os.path.join(tc.output_dir, "config.json"), "w") as f:
        json.dump({"model": vars(mc), "train": vars(tc)}, f, indent=2)

    # ── 5) 학습 루프 시작 ───────────────────────────────────────────────────
    model.train()                                   # dropout 켜기
    step, best_eval = 0, float("inf")
    losses = []
    t0 = time.time()

    print(f"\nTraining for {tc.max_steps} steps...")
    print(f"{'Step':>6} | {'LR':>10} | {'Train':>10} | {'Eval':>10} | {'Time':>8}")
    print("-" * 56)

    # 바깥 while: 한 epoch이 끝나도 max_steps에 도달 못 했으면 다시 epoch 돌림
    while step < tc.max_steps:
        for x, y in train_loader:                   # x, y: (B, T_max) — dataset.py 참고
            if step >= tc.max_steps:
                break

            # ── 5-a) 데이터 GPU로 ──────────────────────────────────────────
            x, y = x.to(device), y.to(device)

            # ── 5-b) 이번 step의 LR 적용 ───────────────────────────────────
            lr = get_lr(step, tc)
            for pg in optimizer.param_groups:       # optimizer에 LR 직접 주입
                pg["lr"] = lr

            # ── 5-c) Forward + Backward + Step ─────────────────────────────
            if use_amp:
                # autocast 안에서는 float16/bfloat16으로 자동 변환되어 계산
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)           # forward → loss (scalar)
                # loss를 키운 뒤 backward (underflow 방지)
                scaler.scale(loss).backward()
                # step 전에 gradient를 원래 스케일로 되돌림 (clip하려면 필수)
                scaler.unscale_(optimizer)
                # gradient norm 제한: ‖∇‖ > 1.0이면 비율로 줄임 → 학습 폭주 방지
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                # 실제 weight 업데이트 (단, scaler가 inf/nan 감지하면 skip)
                scaler.step(optimizer)
                scaler.update()                     # 다음 step의 scale 조정
            else:
                # CPU/MPS: AMP 없이 일반 float32 학습
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()

            # gradient 초기화 (set_to_none=True가 0으로 채우는 것보다 빠름).
            # 이걸 안 하면 다음 step의 gradient가 누적돼서 학습 망함.
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            # ── 5-d) 로깅 (100 step마다) ───────────────────────────────────
            if step % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg:10.4f} | {'--':>10} | {elapsed:7.1f}s")

            # ── 5-e) 평가 + 베스트 모델 저장 (eval_interval마다) ───────────
            if step > 0 and step % tc.eval_interval == 0:
                el = evaluate(model, eval_loader, device)
                avg_train = sum(losses[-tc.eval_interval:]) / min(len(losses), tc.eval_interval)
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg_train:10.4f} | {el:10.4f} | {elapsed:7.1f}s")

                # eval loss가 역대 최저면 "best_model.pt"로 갱신
                # → 추론 시 이 파일을 로드해서 사용 (overfitting된 마지막 모델 대신)
                if el < best_eval:
                    best_eval = el
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": vars(mc),
                        "eval_loss": el,
                    }, os.path.join(tc.output_dir, "best_model.pt"))
                    print(f"  -> Best model (eval={el:.4f})")

            # ── 5-f) 주기적 스냅샷 (save_interval마다) ─────────────────────
            # 학습이 중단됐을 때 재개 가능하도록 step별 체크포인트도 저장
            if step > 0 and step % tc.save_interval == 0:
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": vars(mc),
                }, os.path.join(tc.output_dir, f"step_{step}.pt"))

            step += 1

    # ── 6) 학습 종료 후 마지막 모델 저장 ────────────────────────────────────
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": vars(mc),
        "train_losses": losses,                     # loss 곡선도 같이 저장 (분석용)
    }, os.path.join(tc.output_dir, "final_model.pt"))

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.0f}s, best eval: {best_eval:.4f}")


if __name__ == "__main__":
    train()
