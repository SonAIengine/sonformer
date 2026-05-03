# sonformer experiments

> 각 폴더 = 자기완결 실험. baseline vs ablation 비교 포맷이 통일됨.

## 디자인 원칙

1. **Baseline은 박제** — `experiments/00-baseline`은 모든 후속 ablation의 비교 기준. 한 번 측정하고 봉인.
2. **한 실험 = 한 변경** — `model.py`에서 한 컴포넌트만 바꾸기. data/train/optimizer는 모두 동일.
3. **자기완결** — 폴더 하나만 보면 무엇을 했고 어떤 결과인지 완전히 이해 가능.
4. **공정 비교** — 같은 데이터, 같은 토크나이저, 같은 seed, 같은 LR 스케줄.
5. **Reproducible anywhere** — `bash run.sh` 한 줄로 어디서든 동일한 결과.

## 실험 인덱스

| 폴더 | 변경 | 상태 | 핵심 metric |
|---|---|---|---|
| [00-baseline](00-baseline/) | (none — 기준점) | 🟡 ready, 미실행 | eval PPL |
| [02a-rope](02a-rope/) | Learned PE → RoPE | 🟡 ready, 미실행 | extrapolation PPL @ 1k/2k |
| _02b-swiglu_ | ReLU FFN → SwiGLU | ⚪ TODO | eval PPL, params |
| _02c-rmsnorm_ | LayerNorm → RMSNorm | ⚪ TODO | eval PPL, wallclock |
| _02d-gqa_ | MHA → GQA | ⚪ TODO | wallclock, VRAM |
| _02e-flashattn_ | naive → SDPA / Flash | ⚪ TODO | wallclock, VRAM |

(상세 로드맵은 루트의 [../README.md](../README.md) 참조)

## 폴더 구조 표준

```
NN-name/
├── README.md          가설 / 변경점 / 결과 표
├── config.yaml        모든 hyperparameter
├── model.py           이 실험의 모델 (자기완결)
├── train.py           학습 + eval 스크립트
├── run.sh             진입점 (`bash run.sh`)
├── requirements.txt   버전 고정
└── results/           학습 후 자동 생성
    ├── train_log.csv
    ├── eval.json
    ├── samples.txt
    ├── best.pt
    ├── config.json
    └── model_meta.json
```

## 새 ablation 추가 흐름

```bash
# 1. baseline 폴더 복사
cp -r 00-baseline 02b-swiglu

# 2. README.md 업데이트 (가설, 변경점)
# 3. model.py만 수정 (FFN을 SwiGLU로)
# 4. config.yaml의 experiment.name만 수정
# 5. 실행
cd 02b-swiglu && bash run.sh

# 6. 결과 표를 README.md에 채우고 commit
```

## 결과 비교 (모든 실험 끝난 뒤)

각 실험의 `results/eval.json`을 모아 비교 표를 만들 수 있도록
나중에 `tools/compare_experiments.py`를 추가 예정.

지금은 각 폴더의 `README.md`에 baseline 대비 표를 직접 채움.

## 데이터셋 / 평가 인프라

공통 인프라는 `../shared/`에 있음:
- `shared/datasets.py` — `load_dataset("tinystories")` 등 자동 다운로드/캐시
- `shared/tokenizers.py` — BPE 학습/로드 (실험 간 동일 토크나이저 강제)
- `shared/eval.py` — perplexity, extrapolation_perplexity, generate_samples
- `shared/logging.py` — CSV / wandb 로거

각 실험은 이 모듈들을 import해서 데이터 로딩과 평가를 수행. 코드 중복 없이
공정 비교 보장.

## 의존성

각 실험 폴더의 `requirements.txt`에 명시. 기본 스택:
- `torch >= 2.1`
- `tokenizers >= 0.15`
- `datasets >= 2.14` (HuggingFace, TinyStories/WikiText 자동 다운로드용)
- `PyYAML`

`SONFORMER_WANDB=1`로 wandb 로깅 활성화 (선택, `pip install wandb` 필요).
