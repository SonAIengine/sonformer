"""
공통 데이터셋 로더.

설계:
    각 실험이 같은 데이터를 부르려면 단일 진입점이 필요함.
    `load_dataset("tinystories")` 한 줄로 어디서든 동일한 split을 받게 만든다.
    캐시는 프로젝트 루트의 `data/<name>/`에 저장 → 한 번 받으면 재사용.

지원 데이터셋 (커리큘럼 단계별):
    Tier 0 (sanity):   shakespeare       — 1MB, 5분 학습용
    Tier 1 (메인):     tinystories        — 500MB, Phase 1-2 ablation 메인
                       wikitext-103       — 표준 LM 벤치마크 (논문 비교용)
    Tier 3 (long):     pg19               — 외삽 능력 평가

사용 예:
    from shared.datasets import load_dataset
    train_texts = load_dataset("tinystories", split="train")
    val_texts   = load_dataset("tinystories", split="validation")

반환값:
    문자열 리스트 (각 원소 = 한 샘플/문서). 토큰화는 별도(`shared.tokenizers`).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

# 프로젝트 루트의 data/ 폴더가 기본 캐시 위치
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data"


# ───────────────────────────────────────────────────────────────────────────
# 공개 API
# ───────────────────────────────────────────────────────────────────────────

def load_dataset(
    name: str,
    split: str = "train",
    cache_dir: Optional[str | Path] = None,
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    데이터셋 이름으로 텍스트 리스트를 반환.

    Args:
        name: "tinystories", "shakespeare", "wikitext-103", "pg19"
        split: "train", "validation", "test" (데이터셋마다 다름)
        cache_dir: 캐시 위치 (기본 = 프로젝트/data)
        max_samples: 디버깅용 — 처음 N개만 로드

    Returns:
        list[str]. 각 원소는 한 문서/샘플 텍스트.
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)

    if name == "shakespeare":
        texts = _load_shakespeare(split, cache)
    elif name == "tinystories":
        texts = _load_tinystories(split, cache)
    elif name == "wikitext-103":
        texts = _load_wikitext103(split, cache)
    elif name == "pg19":
        texts = _load_pg19(split, cache)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported: shakespeare, tinystories, wikitext-103, pg19"
        )

    if max_samples is not None:
        texts = texts[:max_samples]
    return texts


def dataset_info(name: str) -> dict:
    """데이터셋 메타정보 (학습 전 sanity check용)."""
    return _DATASET_INFO.get(name, {})


_DATASET_INFO = {
    "shakespeare": {
        "size_mb": 1,
        "approx_tokens": 300_000,
        "domain": "literary (Shakespeare plays)",
        "tier": 0,
        "use": "sanity check (5분 학습)",
    },
    "tinystories": {
        "size_mb": 500,
        "approx_tokens": 500_000_000,
        "domain": "synthetic children's stories",
        "tier": 1,
        "use": "Phase 1-2 메인 ablation",
    },
    "wikitext-103": {
        "size_mb": 500,
        "approx_tokens": 100_000_000,
        "domain": "English Wikipedia",
        "tier": 1,
        "use": "표준 LM 비교 (논문 baseline)",
    },
    "pg19": {
        "size_mb": 11_000,
        "approx_tokens": 2_000_000_000,
        "domain": "Project Gutenberg books (long)",
        "tier": 3,
        "use": "long-context 외삽 평가",
    },
}


# ───────────────────────────────────────────────────────────────────────────
# 데이터셋별 로더
# ───────────────────────────────────────────────────────────────────────────

def _load_shakespeare(split: str, cache: Path) -> List[str]:
    """
    Tiny Shakespeare (Karpathy의 char-rnn 데이터).
    train/val을 9:1로 나눔 (split이 'validation'이면 마지막 10%).
    """
    out_dir = cache / "shakespeare"
    out_dir.mkdir(exist_ok=True)
    txt_path = out_dir / "input.txt"

    if not txt_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"[shakespeare] downloading {url} ...")
        urllib.request.urlretrieve(url, txt_path)
        print(f"[shakespeare] saved to {txt_path}")

    text = txt_path.read_text(encoding="utf-8")
    n = len(text)
    train_end = int(n * 0.9)
    if split == "train":
        body = text[:train_end]
    elif split in ("validation", "val", "test"):
        body = text[train_end:]
    else:
        raise ValueError(f"Unknown split '{split}'")

    # 빈 줄 두 개 이상으로 자른 뒤 빈 chunk 제거 → "샘플" 단위로 만듦
    chunks = [c.strip() for c in body.split("\n\n") if c.strip()]
    return chunks


def _load_tinystories(split: str, cache: Path) -> List[str]:
    """
    TinyStories (Microsoft 2023). HuggingFace `datasets` 라이브러리 우선,
    실패 시 직접 다운로드로 폴백.
    """
    out_dir = cache / "tinystories"
    out_dir.mkdir(exist_ok=True)

    # 1) HuggingFace datasets 시도
    try:
        from datasets import load_dataset as hf_load  # type: ignore

        hf_split = {"train": "train", "validation": "validation", "val": "validation"}.get(split, split)
        ds = hf_load("roneneldan/TinyStories", split=hf_split, cache_dir=str(out_dir))
        return [str(x) for x in ds["text"]]
    except ImportError:
        pass
    except Exception as e:
        print(f"[tinystories] HF load failed ({e}), falling back to direct download")

    # 2) 폴백: HuggingFace에서 raw 파일 직접 다운로드
    files = {
        "train": "TinyStoriesV2-GPT4-train.txt",
        "validation": "TinyStoriesV2-GPT4-valid.txt",
        "val": "TinyStoriesV2-GPT4-valid.txt",
    }
    fname = files.get(split)
    if fname is None:
        raise ValueError(f"Unknown split '{split}'")

    fpath = out_dir / fname
    if not fpath.exists():
        url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{fname}"
        print(f"[tinystories] downloading {url} (this may take a while) ...")
        urllib.request.urlretrieve(url, fpath)
        print(f"[tinystories] saved to {fpath}")

    text = fpath.read_text(encoding="utf-8")
    # TinyStories는 "<|endoftext|>"로 스토리 구분
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
    return stories


def _load_wikitext103(split: str, cache: Path) -> List[str]:
    """WikiText-103. HuggingFace datasets로만 지원 (분리된 파일 형태)."""
    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except ImportError:
        raise RuntimeError(
            "wikitext-103 requires `pip install datasets`."
        )

    hf_split = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}.get(split, split)
    ds = hf_load("wikitext", "wikitext-103-raw-v1", split=hf_split, cache_dir=str(cache / "wikitext-103"))
    # 빈 줄 거르기
    return [str(x) for x in ds["text"] if str(x).strip()]


def _load_pg19(split: str, cache: Path) -> List[str]:
    """PG-19 (long context 평가용). HF datasets로만 지원."""
    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except ImportError:
        raise RuntimeError("pg19 requires `pip install datasets`.")

    hf_split = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}.get(split, split)
    ds = hf_load("deepmind/pg19", split=hf_split, cache_dir=str(cache / "pg19"))
    return [str(x) for x in ds["text"]]


# ───────────────────────────────────────────────────────────────────────────
# 토큰화된 JSONL로 변환 (학습 파이프라인이 바로 받게)
# ───────────────────────────────────────────────────────────────────────────

def to_jsonl(texts: Iterable[str], out_path: str | Path, key: str = "text") -> int:
    """
    텍스트 리스트를 `{"text": ...}` JSONL 형식으로 저장.
    기존 dataset.py 파이프라인과 호환.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({key: t}, ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    # 빠른 점검: shakespeare는 인터넷만 있으면 즉시 동작
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "shakespeare"
    print(f"[info] {name}: {dataset_info(name)}")
    train = load_dataset(name, split="train", max_samples=3)
    print(f"[info] {name} train sample (first 3):")
    for i, t in enumerate(train):
        print(f"  [{i}] {t[:120]!r}{'...' if len(t) > 120 else ''}")
