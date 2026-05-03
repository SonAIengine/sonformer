# sonformer

> Transformer를 코드로 직접 따라가며 공부하는 개인 학습 저장소.

[arman-bd/guppylm](https://github.com/arman-bd/guppylm)을 베이스로, 대학원 과정에서 transformer 아키텍처와 LLM 학습 파이프라인 전체를 한 줄씩 이해하고 본인 손으로 발전시키는 것을 목표로 합니다.

모든 핵심 파일에는 **shape 추적 + 동작 원리 + 설계 의도**를 담은 한국어 주석이 달려있습니다.

---

## 프로젝트 구조

```
sonformer/
├── guppylm/              ⭐ FROZEN baseline — 한국어 주석 잔뜩, "교과서" 역할
├── shared/               공통 인프라 (실험 간 공유)
│   ├── datasets.py         load_dataset("tinystories"), ("shakespeare"), ...
│   ├── tokenizers.py       BPE 학습/로드 (실험 간 동일 토크나이저 강제)
│   ├── eval.py             perplexity, extrapolation_perplexity, generate_samples
│   └── logging.py          CSV / wandb 로거
├── data/                 다운로드된 데이터 캐시 (.gitignore)
├── experiments/          ⭐ ablation kit — 여기에 작업 누적
│   ├── 00-baseline/        vanilla 기준점 (frozen)
│   ├── 02a-rope/           Learned PE → RoPE
│   └── README.md           실험 인덱스
└── docs/, tools/, ...    원본 유산 (export_onnx 등)
```

각 실험은 **자기완결**이라서 폴더 하나만 보면 무엇을 했는지/어떤 결과인지 완전히 파악 가능. `bash run.sh` 한 줄로 어디서든 (Colab/local/Lambda) 동일한 결과를 재현.

상세 사용법: [experiments/README.md](experiments/README.md)

---

## 학습 워크플로 (Claude와 함께)

이 저장소는 [Claude Code](https://claude.com/claude-code)와 짝을 이뤄 쓰도록 설계됐습니다 — [`CLAUDE.md`](CLAUDE.md)에 협업 방식이 박혀있어서, 다음과 같은 발화로 새 실험을 시작할 수 있습니다.

| 발화 | 결과 |
|---|---|
| "**SwiGLU 공부하고 싶어**" | `experiments/02b-swiglu/` 자동 스폰 + 가설부터 함께 작성 |
| "**RoPE 어떻게 동작해?**" | 수학적 직관 + 코드 라인별 설명 |
| "**FlashAttention 직접 구현해보자**" | `02e-flashattn/` 스폰 + 한 줄씩 함께 |
| "**왜 PPL 차이가 이렇게 나?**" | loss curve / attention map 분석 |

### 도구

```bash
# 새 ablation 폴더 스폰 (baseline 복사 + README 템플릿 + config 갱신)
bash tools/new_experiment.sh 02b-swiglu "FFN: ReLU → SwiGLU"

# 모든 완료된 실험 결과를 한 표로
python tools/compare.py
# 또는 CSV로
python tools/compare.py --csv summary.csv

# 학습 실행 (의존성 이미 있으면 스킵)
cd experiments/02b-swiglu && SONFORMER_SKIP_DEPS=1 bash run.sh
```

### 진행 현황 (한눈에)

| 실험 | 데이터 | Eval PPL | Δ vs baseline | 상태 |
|---|---|---|---|---|
| [00-baseline](experiments/00-baseline/) | Shakespeare | 37.47 | (기준) | ✅ |
| [02a-rope](experiments/02a-rope/) | Shakespeare | 28.47 | **-24.0%** | ✅ |
| 02b-swiglu | _ | _ | _ | ⚪ TODO |
| 02c-rmsnorm | _ | _ | _ | ⚪ TODO |
| ... | _ | _ | _ | ⚪ TODO |

전체 로드맵 (8 Phase, 100+ 항목)은 아래 [발전 계획](#발전-계획-roadmap) 참고.

---

## 학습 커리큘럼

코드를 읽는 권장 순서. "데이터가 어떻게 흘러가는지" 따라가는 방향으로 작은 것부터 큰 것 순서.

### 0. [config.py](guppylm/config.py) — 하이퍼파라미터 (1분)

모델/학습 설정값을 먼저 한 번 훑어두면, 다른 파일에서 `d_model=384`, `n_heads=6` 같은 숫자가 어디서 오는지 바로 보입니다.

### 1. [model.py](guppylm/model.py) ⭐ — Transformer 구조 (핵심)

읽는 순서:
1. `GuppyLM.forward` — 전체 흐름 (토큰 → 임베딩 → 블록 → logits)
2. `Block.forward` — Pre-LN + residual 패턴 (4줄, transformer의 정수)
3. `Attention.forward` — QKV 분리, scaled dot-product, causal mask
4. `FFN.forward` — 2-layer MLP
5. `generate` — autoregressive 샘플링, temperature/top-k

**여기서 배우는 것**
- Multi-head self-attention의 shape 변환 (B, T, C → B, H, T, D)
- Causal mask가 왜 필요한지, `√d_k` 스케일링의 의미
- Pre-LN vs Post-LN, weight tying, learned positional embedding
- temperature/top-k가 생성 결과에 미치는 영향

### 2. [dataset.py](guppylm/dataset.py) — 데이터 파이프라인

- 텍스트 → BPE 토큰 ID 변환
- **`x = ids[:-1]`, `y = ids[1:]` — 한 칸 shift** (language modeling의 본질)
- Padding과 `collate_fn`

**여기서 배우는 것**
- LM이 한 번의 forward로 어떻게 T개 위치를 동시에 학습하는지 (causal mask 덕분)
- `pad_id=0`과 `cross_entropy(ignore_index=0)`의 연결

### 3. [train.py](guppylm/train.py) — 학습 루프

- AdamW (`betas=(0.9, 0.95)`) + warmup + cosine LR decay
- AMP (mixed precision) + GradScaler
- Gradient clipping (norm = 1.0)
- 체크포인트 전략 (best / step / final)

**여기서 배우는 것**
- 학습 안정화 트릭: 왜 warmup, 왜 grad clip, 왜 AMP
- `forward → backward → unscale → clip → step → zero_grad` 순서가 왜 그 순서인지
- `set_to_none=True`의 미묘한 의미

### 4. [inference.py](guppylm/inference.py) — 추론

- ChatML 프롬프트 포맷 (`<|im_start|>...<|im_end|>`)
- Autoregressive 생성 루프
- 후처리 (특수 토큰 누수 차단)

**여기서 배우는 것**
- 학습과 추론은 본질적으로 같은 일 — 둘 다 "다음 토큰 예측"
- 프롬프트 엔지니어링의 시작점 — 모델이 학습한 패턴을 그대로 활용하는 방식

### 5. (선택) [prepare_data.py](guppylm/prepare_data.py) + [generate_data.py](guppylm/generate_data.py)

BPE tokenizer 학습과 합성 데이터 생성. tokenizer 동작 원리, 데이터 엔지니어링이 궁금하면.

---

## 모델 사양

| | |
|---|---|
| 파라미터 | 8.7M |
| Layers | 6 |
| Hidden dim | 384 |
| Heads | 6 |
| FFN | 768 (ReLU) |
| Vocab | 4,096 (BPE) |
| Max sequence | 128 토큰 |
| Norm | LayerNorm (Pre-LN) |
| Position | Learned embeddings |
| LM head | Weight-tied |

Vanilla decoder-only transformer. RoPE, GQA, SwiGLU 같은 모던 기법은 의도적으로 제외 — **기본 구조 학습이 목적**이라 단순함이 더 중요합니다.

---

## 실행

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 학습 (T4 GPU 기준 ~5분)
```bash
python -m guppylm prepare    # 합성 데이터 + BPE tokenizer 생성
python -m guppylm train      # 학습 시작
python -m guppylm chat       # 학습된 모델로 대화
```

> 패키지 이름은 원본 그대로 `guppylm`으로 유지. repo 이름(sonformer)은 **본인 학습 프로젝트로서의 정체성**이고, 코드 패키지 이름은 원본 호환성을 위해 그대로 둡니다.

---

## 발전 계획 (Roadmap)

Vanilla transformer를 출발점으로, **최신 트렌드까지 모든 항목을 직접 구현하고 측정**해 본인 색깔의 모델로 발전시킵니다.
사이클: `구현 → 학습 → ablation/벤치마크 → 정리(노트/블로그)`.

### Phase 1 — 기본기 다지기

코드를 손으로 따라가며 기초 체력을 굳히는 단계.

- [ ] Attention shape 변환을 손으로 추적하며 노트 작성
- [ ] Causal mask / padding mask 동작 시각화
- [ ] `√d_k` 스케일링 유무에 따른 gradient norm 비교
- [ ] Weight tying ON/OFF — perplexity & 파라미터 수 비교
- [ ] Loss curve, attention heatmap 자동 로깅 도구 (TensorBoard / wandb)
- [ ] BPE merge 과정 직접 추적 (tokenizer 내부 시각화)

### Phase 2 — 아키텍처 변형

vanilla 대비 **perplexity / wallclock / VRAM**으로 ablation.

**Positional Encoding**
- [ ] Learned PE (현재) vs Sinusoidal PE
- [ ] **RoPE** (Rotary) — LLaMA 계열 표준
- [ ] **ALiBi** — 외삽 가능한 bias
- [ ] **NoPE** — PE 없이 학습 가능성 검증
- [ ] 장문 외삽 비교: NTK-aware RoPE / YaRN / LongRoPE

**Attention 변형**
- [ ] MHA → **MQA (Multi-Query)**
- [ ] MHA → **GQA (Grouped-Query)** — LLaMA 2/3 표준
- [ ] **MLA (Multi-head Latent Attention)** — DeepSeek-V2/V3
- [ ] **Sliding Window Attention** — Mistral
- [ ] **FlashAttention** 도입 (`F.scaled_dot_product_attention`)
- [ ] **Native Sparse Attention** — DeepSeek 2025

**FFN / Activation**
- [ ] ReLU → GELU → **SwiGLU** / GeGLU 비교
- [ ] **Mixture of Experts (MoE)** — Switch / Mixtral 스타일 top-k routing
- [ ] Aux loss / load balancing 구현

**Normalization**
- [ ] LayerNorm → **RMSNorm** — LLaMA 표준
- [ ] **QK-Norm** (query/key에 추가 norm)
- [ ] DeepNorm (deep 모델 안정화)
- [ ] Pre-LN vs Post-LN vs Sandwich (NormFormer)
- [ ] **Parallel residual** (PaLM / GPT-J 스타일)

### Phase 3 — 학습 최적화

**Optimizer**
- [ ] AdamW (현재) → **Lion**
- [ ] **Sophia** (second-order 근사)
- [ ] **Muon** (orthogonal momentum, 2024)
- [ ] Adafactor — 저메모리 학습

**LR / Init**
- [ ] Cosine → **WSD** (Warmup-Stable-Decay)
- [ ] **μP (maximal update parametrization)** — width-invariant scaling
- [ ] Z-loss / auxiliary loss
- [ ] Init scale ablation (small init, depth-scaled init)

**메모리 / 속도**
- [ ] fp16 → **bf16** → **fp8** (H100/H200) 비교
- [ ] Gradient checkpointing
- [ ] Gradient accumulation
- [ ] **Sequence packing** (짧은 샘플 묶어 throughput ↑)
- [ ] **DeepSpeed ZeRO-1/2/3** 단계별 비교
- [ ] **FSDP** (PyTorch 네이티브)
- [ ] Tensor / Pipeline / Sequence parallel (multi-GPU 시)

**Curriculum**
- [ ] Length warmup — 짧은 시퀀스에서 긴 시퀀스로
- [ ] 데이터 품질 필터링 효과 측정 (post-Chinchilla 후속들)
- [ ] Scaling laws 직접 측정 (작은 grid로 Chinchilla 곡선 재현)

### Phase 4 — Post-training (Alignment & Reasoning)

**SFT (Supervised Fine-Tuning)**
- [ ] Instruction tuning 데이터셋 구성
- [ ] **LoRA / QLoRA / DoRA** 저비용 fine-tuning
- [ ] Adapter / prefix tuning 비교

**Preference Learning**
- [ ] Reward model 학습
- [ ] **PPO (RLHF)** — 클래식 baseline
- [ ] **DPO** — reward model 없는 직접 최적화
- [ ] **KTO / IPO / ORPO / SimPO** 비교
- [ ] **GRPO** (DeepSeek) — group relative policy optimization
- [ ] Online DPO / iterative DPO

**Reasoning (o1 / R1 계열)**
- [ ] Chain-of-Thought 데이터로 SFT
- [ ] **R1-Zero 스타일** RL on reasoning traces
- [ ] **Test-time compute scaling** — best-of-N, self-consistency, MCTS
- [ ] Reasoning **distillation** — 큰 R1 → 작은 모델
- [ ] Verifier / process reward model

### Phase 5 — 추론 최적화

- [ ] **KV cache** 직접 구현 (paged attention 류)
- [ ] **Continuous batching**
- [ ] Quantization: INT8 → INT4 (**GPTQ / AWQ**) → FP8 → **1-bit (BitNet)**
- [ ] ONNX export 개선 (현재 `tools/export_onnx.py` 베이스)
- [ ] **Speculative decoding** — draft model로 가속
- [ ] **EAGLE / Medusa** — 자체 decode head로 가속
- [ ] llama.cpp / ggml export
- [ ] vLLM / SGLang 비교 벤치마크

### Phase 6 — 평가 인프라

- [ ] Perplexity / loss curve 자동 리포트
- [ ] **lm-evaluation-harness** 통합
- [ ] **HumanEval / MBPP** — 코드
- [ ] **MMLU / GSM8K / MATH** — 지식 / 수학
- [ ] **IFEval** — instruction following
- [ ] **LLM-as-judge** (bias 분석 포함)
- [ ] Long-context 평가 (Needle-in-Haystack, RULER)
- [ ] 한국어 평가셋 (KMMLU, HAE-RAE)

### Phase 7 — 최신 트렌드 / 장기 실험 (2024–2026)

- [ ] **Mamba / Mamba-2** 직접 구현 + transformer 비교
- [ ] **Hybrid 아키텍처** — Jamba 스타일 (Transformer + Mamba)
- [ ] **Multi-Token Prediction (MTP)** — DeepSeek-V3
- [ ] **Linear attention / RetNet** 계열
- [ ] **Long context** 1M+ 토큰 학습/평가
- [ ] **Multimodal**: CLIP 인코더 붙여 LLaVA 스타일 VLM
- [ ] **Whisper 스타일** 음성 인코더 결합
- [ ] **RAG** (Retrieval-Augmented Generation) end-to-end
- [ ] **Agent / Tool use** — function calling 학습
- [ ] **MoE 라우팅 분석** — 어떤 expert가 어떤 토큰을 처리하는지

### Phase 8 — 한국어 / 본인 색깔

- [ ] 한국어 합성 데이터로 재학습 (sonformer-ko)
- [ ] 한국어 BPE / SentencePiece tokenizer 직접 학습
- [ ] 한국어 reasoning 데이터셋 합성
- [ ] 본인이 발견한 ablation 결과로 **모델 카드** 작성
- [ ] HuggingFace Hub 업로드 + 데모

### 산출물 형식

각 실험은 일관된 포맷으로 정리:
- `experiments/<phase>/<topic>/` — 학습 스크립트, config, 결과
- 학습 곡선 + ablation 표 + 메모리/속도 비교
- README 또는 블로그 글
- 재현 가능한 seed 고정 + 환경 명세

---

## 출처 및 라이선스

이 저장소는 [arman-bd/guppylm](https://github.com/arman-bd/guppylm)을 학습 베이스로 사용합니다. MIT 라이선스이며 원본의 `LICENSE` 파일은 그대로 보존되어 있습니다.

**원본의 핵심 가치 (인용)**
> 자기만의 언어모델을 학습시키는 일은 마법이 아니다.
> 박사 학위도, 대규모 GPU 클러스터도 필요 없다.
> 노트북 한 권, 5분, 그리고 처음부터 끝까지 직접 만든 LLM.

이 메시지에 공감해 학습 베이스로 채택했고, sonformer는 여기에 한국어 주석/학습 노트/개인 실험을 누적해가는 저장소입니다.

---

License: MIT
