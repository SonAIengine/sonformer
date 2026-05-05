# sonformer

> Transformer를 코드로 직접 따라가며 공부하는 개인 학습 저장소.

[arman-bd/guppylm](https://github.com/arman-bd/guppylm)을 베이스로, 대학원 과정에서 transformer 아키텍처와 LLM 학습 파이프라인을 한 줄씩 이해하고 본인 손으로 발전시키는 것이 목표입니다.

이 저장소는 **두 가지 모드**로 운용됩니다:

| 모드 | 폴더 | 목적 |
|---|---|---|
| 🖐️ **학습 트랙** | [`practice/`](practice/) | 빈 폴더에서부터 모델을 직접 손코딩. autograd 기초 → MLP/CNN/RNN/LSTM → Transformer 가족까지 8단계 |
| 🔬 **실험 트랙** | [`experiments/`](experiments/) | vanilla transformer baseline 위에 모던 기법을 한 가지씩 ablation, PPL/속도/메모리 정량 비교 |

> **동선** — 두 트랙은 *연결*돼 있음. 같은 기법을 `practice/`에서 먼저 손코딩(구조 이해) → `experiments/`에서 ablation(효과 측정).
> 예: RoPE 공부하고 싶다 → `practice/llama-from-scratch/`(예정)에서 손으로 구현 → [`experiments/02a-rope/`](experiments/02a-rope/)에서 vanilla 대비 PPL 변화 측정.

---

## Quick Start

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 학습 트랙 — 가장 작은 단계부터
python practice/00-tensor-autograd/solution.py     # autograd가 chain rule 정확히 따르는지 손으로 검증

# 3) 실험 트랙 — vanilla baseline + RoPE ablation
python -m sonlm prepare && python -m sonlm train   # sonlm reference 학습
cd experiments/02a-rope && bash run.sh             # 첫 ablation
```

---

## 🖐️ 학습 트랙 (practice/)

빈 [`practice.py`](practice/06-encoder-decoder-from-scratch/practice.py)에서 시작해 [`solution.py`](practice/00-tensor-autograd/solution.py)와 비교하며 직접 손코딩. 각 폴더는 self-contained — synthetic data로 다운로드 없이 즉시 돌아감.

### 8단계 커리큘럼

| # | 폴더 | 학습 핵심 |
|---|---|---|
| 🌰 0 | [00-tensor-autograd](practice/00-tensor-autograd/) | 손 gradient ↔ autograd 일치 검증, broadcasting, 흔한 함정 3가지 |
| 🌰 0 | [01-mlp-from-scratch](practice/01-mlp-from-scratch/) | MLP on two-moons. nn.Module / ReLU 비선형성 / decision boundary |
| 🌱 1 | [02-cnn-from-scratch](practice/02-cnn-from-scratch/) | TextCNN. 1D conv = n-gram 패턴 detector |
| 🌱 1 | [03-rnn-from-scratch](practice/03-rnn-from-scratch/) | Vanilla Elman RNN. hidden state 누적, BPTT |
| 🌱 1 | [04-lstm-from-scratch](practice/04-lstm-from-scratch/) | 4-gate LSTM. cell state, forget bias=1 트릭 |
| 🌳 2 | [05-sonlm-from-scratch](practice/05-sonlm-from-scratch/) | Decoder-only (GPT 계열). causal self-attn, autoregressive |
| 🌳 2 | [06-encoder-decoder-from-scratch](practice/06-encoder-decoder-from-scratch/) | 원조 2017 paper. cross-attention, sinusoidal PE, teacher forcing |
| 🌳 2 | [07-bert-from-scratch](practice/07-bert-from-scratch/) | Encoder-only + MLM. 양방향 attention, [CLS] |

상세 안내 + 학습 시나리오: [practice/README.md](practice/README.md)

---

## 🔬 실험 트랙 (experiments/)

각 실험은 **자기완결** 폴더 — `bash run.sh` 한 줄로 어디서든(Colab/local/Lambda) 동일 결과 재현. 모든 실험은 [`experiments/00-baseline`](experiments/00-baseline/) 대비 ablation 비교.

```bash
# 새 ablation 폴더 스폰
bash tools/new_experiment.sh 02b-swiglu "FFN: ReLU → SwiGLU"

# 학습 실행 (의존성 이미 있으면 스킵)
cd experiments/02b-swiglu && SONFORMER_SKIP_DEPS=1 bash run.sh

# 모든 완료된 실험 결과 비교 표
python tools/compare.py
```

상세 인덱스: [experiments/README.md](experiments/README.md)

### Claude Code와 함께

[`CLAUDE.md`](CLAUDE.md)에 협업 방식이 박혀있어 다음 발화로 새 실험을 시작할 수 있습니다:

| 발화 | 결과 |
|---|---|
| "**SwiGLU 공부하고 싶어**" | `experiments/02b-swiglu/` 자동 스폰 + 가설부터 함께 작성 |
| "**RoPE 어떻게 동작해?**" | 수학적 직관 + 코드 라인별 설명 |
| "**왜 PPL 차이가 이렇게 나?**" | loss curve / attention map 분석 |

---

## sonlm — 학습 reference 패키지

`practice/05-sonlm-from-scratch/solution.py`의 다파일 버전. 한국어 주석으로 **shape 추적 + 동작 원리 + 설계 의도**를 담은 vanilla decoder-only LM. 실험 트랙의 baseline 역할.

| | |
|---|---|
| 파라미터 | 8.7M |
| Layers / Heads / d_model | 6 / 6 / 384 |
| FFN | 768 (ReLU) |
| Vocab | 4,096 (BPE) |
| Max sequence | 128 |
| Norm / PE / LM head | LayerNorm Pre-LN / Learned / Weight-tied |

RoPE, GQA, SwiGLU 등 모던 기법은 의도적으로 제외 — **단순함이 학습 목적에 더 중요**. 모던 기법은 `experiments/` 에서 한 가지씩 비교.

```bash
python -m sonlm prepare    # 합성 데이터 + BPE tokenizer 생성
python -m sonlm train      # 학습 (T4 GPU 기준 ~5분)
python -m sonlm chat       # 학습된 모델로 대화
```

---

## Roadmap (11 Phase)

수학 기초부터 SOTA까지 — 이 한 트랙을 다 통과하면 현대 LLM의 모든 핵심을 직접 구현/측정/배포할 수 있습니다.

| Phase | 영역 | 핵심 항목 |
|---|---|---|
| 0 | 수학 기초 | 선형대수 · SVD · chain rule · entropy/KL · convex optimization |
| 1 | ML/DL 기초 | regression · backprop · init/norm/dropout · 디버깅 (sanity, gradient check) |
| 2 | 시퀀스 모델 | CNN/RNN/LSTM/GRU · Bahdanau attn · Transformer 가족 (decoder/encoder/enc-dec) |
| 3 | Transformer 변형 | RoPE/ALiBi · MQA/GQA/MLA · FlashAttn 1/2/3 · SwiGLU/MoE · RMSNorm |
| 4 | 학습 최적화 + 시스템 | Lion/Sophia/Muon · WSD/μP · bf16/fp8 · ZeRO/FSDP/TP/PP/EP · scaling laws · **CUDA/Triton, GPU memory hierarchy, NCCL, NVLink/IB/RDMA, nsight profiling** |
| 5 | 데이터/토크나이저 | BPE/WordPiece/SP · dedup/filter · FineWeb/RedPajama · synthetic (Phi 스타일) |
| 6 | Post-training | LoRA/QLoRA/DoRA · PPO/DPO/KTO/GRPO · R1-style RL · CAI · red-teaming |
| 7 | 추론 최적화 | KV cache · GPTQ/AWQ/BitNet · Speculative/EAGLE · vLLM/SGLang/llama.cpp |
| 8 | 평가/안전/해석 | lm-eval-harness · ARC-AGI · LLM-as-judge · SAE · mech interp |
| 9 | 최신 트렌드 | Mamba/Jamba · MTP · multimodal · RAG/GraphRAG · agents · diffusion LLM |
| 10 | 한국어/본인 색깔 | sonformer-ko · KMMLU · 모델 카드 · HF Hub 데모 |

전체 체크리스트 + reference: [ROADMAP.md](ROADMAP.md)

---

## 출처 및 라이선스

[arman-bd/guppylm](https://github.com/arman-bd/guppylm) (MIT)을 학습 베이스로 사용합니다. 원본의 `LICENSE`는 그대로 보존되어 있습니다.

> 자기만의 언어모델을 학습시키는 일은 마법이 아니다.
> 박사 학위도, 대규모 GPU 클러스터도 필요 없다.
> 노트북 한 권, 5분, 그리고 처음부터 끝까지 직접 만든 LLM.

이 메시지에 공감해 학습 베이스로 채택했고, sonformer는 여기에 한국어 주석/학습 노트/개인 실험을 누적해가는 저장소입니다.

License: MIT
