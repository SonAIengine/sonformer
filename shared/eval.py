"""
공통 평가 유틸.

여기에 들어가는 metric은 모든 실험에서 동일하게 쓰여야 함
(다르면 ablation 비교 불가).

Metric:
    - perplexity (eval cross-entropy → exp)
    - extrapolation_perplexity (학습 길이 넘는 시퀀스에서 PPL — RoPE/ALiBi 비교용)
    - generate_samples (정성평가용 샘플 출력)
"""

from __future__ import annotations

import math
import time
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(
    model,
    loader,
    device,
    max_batches: Optional[int] = None,
    pad_id: int = 0,
) -> dict:
    """
    Eval set의 평균 cross-entropy와 perplexity 계산.

    Returns:
        {"loss": float, "ppl": float, "n_tokens": int, "n_batches": int}
    """
    model.eval()
    total_loss, total_tokens, n = 0.0, 0, 0
    for x, y in loader:
        if max_batches is not None and n >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        # pad가 아닌 토큰 수 (정확한 PPL 계산용)
        mask = (y != pad_id)
        n_tok = mask.sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
        n += 1
    avg_loss = total_loss / max(1, total_tokens)
    return {
        "loss": avg_loss,
        "ppl": math.exp(avg_loss),
        "n_tokens": total_tokens,
        "n_batches": n,
    }


@torch.no_grad()
def extrapolation_perplexity(
    model,
    tokenizer,
    long_texts: Iterable[str],
    device,
    seq_lens: List[int] = (256, 512, 1024, 2048),
    max_samples_per_len: int = 32,
    pad_id: int = 0,
) -> dict:
    """
    학습 시 max_seq_len을 넘는 길이에서 PPL 측정.
    Position encoding ablation의 핵심 metric (Learned PE는 외삽 시 망함, RoPE/ALiBi는 견딤).

    각 seq_len마다 평균 PPL 반환.
    """
    model.eval()
    results = {}
    for L in seq_lens:
        losses, n_tok = 0.0, 0
        used = 0
        for text in long_texts:
            if used >= max_samples_per_len:
                break
            ids = tokenizer.encode(text).ids
            if len(ids) < L + 1:
                continue
            window = ids[: L + 1]
            x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([window[1:]], dtype=torch.long, device=device)
            try:
                _, loss = model(x, y)
            except Exception as e:
                # 모델이 외삽 자체가 안 되는 경우 (예: learned PE의 max_seq_len 초과)
                results[L] = {"loss": float("inf"), "ppl": float("inf"), "error": str(e)}
                used = max_samples_per_len  # 다음 seq_len으로
                break
            mask = (y != pad_id)
            t = mask.sum().item()
            losses += loss.item() * t
            n_tok += t
            used += 1

        if n_tok > 0 and L not in results:
            avg = losses / n_tok
            results[L] = {"loss": avg, "ppl": math.exp(min(avg, 30)), "n_samples": used}
        elif L not in results:
            results[L] = {"loss": float("nan"), "ppl": float("nan"), "n_samples": 0}
    return results


@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    prompts: List[str],
    device,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 50,
) -> List[dict]:
    """정성평가용 — 프롬프트마다 한 번 생성."""
    model.eval()
    out = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.time()
        idx, _ = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        elapsed = time.time() - t0
        decoded = tokenizer.decode(idx[0].tolist()[len(ids):])
        out.append({
            "prompt": prompt,
            "generated": decoded,
            "tokens_generated": idx.shape[1] - len(ids),
            "wallclock_s": elapsed,
            "tok_per_s": (idx.shape[1] - len(ids)) / max(1e-6, elapsed),
        })
    return out


def measure_throughput(
    model,
    loader,
    device,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> dict:
    """
    학습 step당 wallclock + 메모리 측정.
    Optimizer step은 skip하고 forward+backward만 — 모델 자체의 속도를 봄.
    """
    model.train()
    it = iter(loader)
    # warmup
    for _ in range(n_warmup):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    n_steps = 0
    n_tokens = 0
    for _ in range(n_measure):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        n_steps += 1
        n_tokens += x.numel()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    out = {
        "step_per_s": n_steps / elapsed,
        "tok_per_s": n_tokens / elapsed,
        "ms_per_step": 1000 * elapsed / n_steps,
    }
    if device.type == "cuda":
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    return out
