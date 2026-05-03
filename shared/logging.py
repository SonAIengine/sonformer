"""
실험 로깅 유틸.

원칙:
    모든 실험이 같은 포맷의 CSV를 쓰면 나중에 비교 표 자동 생성 가능.
    wandb는 선택 — 환경변수 SONFORMER_WANDB=1일 때만 사용.

CSV 포맷:
    train_log.csv:  step, lr, train_loss, eval_loss, eval_ppl, wallclock_s, vram_mb
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional


class CSVLogger:
    """append-only CSV logger."""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames

        # 새 파일이면 헤더 쓰기
        is_new = not self.path.exists()
        self._fp = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=fieldnames)
        if is_new:
            self._writer.writeheader()
            self._fp.flush()

    def log(self, **row: Any) -> None:
        # 빠진 필드는 빈 값
        for k in self.fieldnames:
            row.setdefault(k, "")
        self._writer.writerow(row)
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class WandbLogger:
    """wandb 래퍼 (옵션). 환경변수 SONFORMER_WANDB=1일 때만 active."""

    def __init__(self, project: str, run_name: str, config: dict):
        self.active = os.environ.get("SONFORMER_WANDB") == "1"
        self._wandb = None
        if self.active:
            try:
                import wandb  # type: ignore
                self._wandb = wandb
                wandb.init(project=project, name=run_name, config=config)
            except ImportError:
                print("[wandb] not installed, skipping (pip install wandb to enable)")
                self.active = False

    def log(self, data: dict, step: Optional[int] = None) -> None:
        if self.active and self._wandb is not None:
            self._wandb.log(data, step=step)

    def finish(self) -> None:
        if self.active and self._wandb is not None:
            self._wandb.finish()


def save_json(path: str | Path, data: dict) -> None:
    """평가 결과 등을 결정적 포맷으로 저장."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_samples(path: str | Path, samples: list[dict]) -> None:
    """generate_samples 결과를 사람이 읽기 좋은 텍스트로 저장."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            f.write(f"=== sample {i} ===\n")
            f.write(f"prompt:    {s.get('prompt', '')!r}\n")
            f.write(f"generated: {s.get('generated', '')!r}\n")
            f.write(f"tok/s:     {s.get('tok_per_s', 0):.1f}\n\n")


class Timer:
    """간단한 컨텍스트 매니저 타이머."""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.t0
