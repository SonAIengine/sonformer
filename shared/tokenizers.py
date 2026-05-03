"""
공통 BPE 토크나이저 학습/로드.

설계:
    실험마다 토크나이저가 다르면 perplexity 비교가 무의미해짐
    (토큰 수가 다르니까). 같은 데이터셋엔 같은 토크나이저를 공유.

캐시 정책:
    `data/<dataset>/tokenizer-<vocab>.json`에 저장.
    vocab_size + 데이터셋이 같으면 재학습 안 하고 로드만.

사용:
    from shared.tokenizers import get_tokenizer
    tk = get_tokenizer(dataset="tinystories", vocab_size=4096)
    ids = tk.encode("hello world").ids
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data"

SPECIAL_TOKENS = ["<|pad|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]


def get_tokenizer(
    dataset: str,
    vocab_size: int = 4096,
    cache_dir: Optional[str | Path] = None,
    train_corpus: Optional[Iterable[str]] = None,
    force_retrain: bool = False,
) -> Tokenizer:
    """
    캐시된 토크나이저를 로드, 없으면 학습 후 캐시.

    Args:
        dataset: 데이터셋 식별자 (캐시 경로 키)
        vocab_size: BPE vocab 크기
        cache_dir: 기본 = data/
        train_corpus: 캐시에 없을 때 학습에 쓸 텍스트 iterable.
                      None이면 자동으로 `shared.datasets.load_dataset`로 학습 split 로드.
        force_retrain: True면 캐시 무시하고 새로 학습

    Returns:
        tokenizers.Tokenizer
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    out_dir = cache / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    tk_path = out_dir / f"tokenizer-{vocab_size}.json"

    if tk_path.exists() and not force_retrain:
        return Tokenizer.from_file(str(tk_path))

    # 학습 데이터 준비
    if train_corpus is None:
        from .datasets import load_dataset
        train_corpus = load_dataset(dataset, split="train")

    print(f"[tokenizer] training BPE (vocab={vocab_size}) on {dataset} ...")
    tk = _train_bpe(train_corpus, vocab_size)
    tk.save(str(tk_path))
    print(f"[tokenizer] saved to {tk_path}")
    return tk


def _train_bpe(corpus: Iterable[str], vocab_size: int) -> Tokenizer:
    """BPE 학습 (byte-level, GPT-2 스타일)."""
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    # Ġ/Ċ 같은 byte-level marker를 사람이 읽는 문자로 되돌리는 decoder
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def encode_batch(tokenizer: Tokenizer, texts: List[str]) -> List[List[int]]:
    """텍스트 배치를 ID 리스트로."""
    encoded = tokenizer.encode_batch(texts)
    return [e.ids for e in encoded]


def special_token_ids(tokenizer: Tokenizer) -> dict:
    """학습/추론에서 자주 쓰는 special token ID들."""
    return {
        "pad": tokenizer.token_to_id("<|pad|>"),
        "bos": tokenizer.token_to_id("<|im_start|>"),
        "eos": tokenizer.token_to_id("<|im_end|>"),
        "eot": tokenizer.token_to_id("<|endoftext|>"),
    }
