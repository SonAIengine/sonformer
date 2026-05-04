# sonformer Roadmap

> Vanilla transformer를 출발점으로, **최신 트렌드까지 모든 항목을 직접 구현하고 측정**해 본인 색깔의 모델로 발전시킨다.
> 사이클: `구현 → 학습 → ablation/벤치마크 → 정리(노트/블로그)`.

축약본은 [README.md](README.md#roadmap-8-phase) 표 참고. 이 파일은 각 Phase의 풀 체크리스트.

---

## Phase 1 — 기본기 다지기

코드를 손으로 따라가며 기초 체력을 굳히는 단계.

- [ ] Attention shape 변환을 손으로 추적하며 노트 작성
- [ ] Causal mask / padding mask 동작 시각화
- [ ] `√d_k` 스케일링 유무에 따른 gradient norm 비교
- [ ] Weight tying ON/OFF — perplexity & 파라미터 수 비교
- [ ] Loss curve, attention heatmap 자동 로깅 도구 (TensorBoard / wandb)
- [ ] BPE merge 과정 직접 추적 (tokenizer 내부 시각화)

---

## Phase 2 — 아키텍처 변형

vanilla 대비 **perplexity / wallclock / VRAM**으로 ablation.

### Positional Encoding
- [ ] Learned PE (현재) vs Sinusoidal PE
- [x] **RoPE** (Rotary) — LLaMA 계열 표준 [(02a-rope)](experiments/02a-rope/)
- [ ] **ALiBi** — 외삽 가능한 bias
- [ ] **NoPE** — PE 없이 학습 가능성 검증
- [ ] 장문 외삽 비교: NTK-aware RoPE / YaRN / LongRoPE

### Attention 변형
- [ ] MHA → **MQA (Multi-Query)**
- [ ] MHA → **GQA (Grouped-Query)** — LLaMA 2/3 표준
- [ ] **MLA (Multi-head Latent Attention)** — DeepSeek-V2/V3
- [ ] **Sliding Window Attention** — Mistral
- [ ] **FlashAttention** 도입 (`F.scaled_dot_product_attention`)
- [ ] **Native Sparse Attention** — DeepSeek 2025

### FFN / Activation
- [ ] ReLU → GELU → **SwiGLU** / GeGLU 비교
- [ ] **Mixture of Experts (MoE)** — Switch / Mixtral 스타일 top-k routing
- [ ] Aux loss / load balancing 구현

### Normalization
- [ ] LayerNorm → **RMSNorm** — LLaMA 표준
- [ ] **QK-Norm** (query/key에 추가 norm)
- [ ] DeepNorm (deep 모델 안정화)
- [ ] Pre-LN vs Post-LN vs Sandwich (NormFormer)
- [ ] **Parallel residual** (PaLM / GPT-J 스타일)

---

## Phase 3 — 학습 최적화

### Optimizer
- [ ] AdamW (현재) → **Lion**
- [ ] **Sophia** (second-order 근사)
- [ ] **Muon** (orthogonal momentum, 2024)
- [ ] Adafactor — 저메모리 학습

### LR / Init
- [ ] Cosine → **WSD** (Warmup-Stable-Decay)
- [ ] **μP (maximal update parametrization)** — width-invariant scaling
- [ ] Z-loss / auxiliary loss
- [ ] Init scale ablation (small init, depth-scaled init)

### 메모리 / 속도
- [ ] fp16 → **bf16** → **fp8** (H100/H200) 비교
- [ ] Gradient checkpointing
- [ ] Gradient accumulation
- [ ] **Sequence packing** (짧은 샘플 묶어 throughput ↑)
- [ ] **DeepSpeed ZeRO-1/2/3** 단계별 비교
- [ ] **FSDP** (PyTorch 네이티브)
- [ ] Tensor / Pipeline / Sequence parallel (multi-GPU 시)

### Curriculum
- [ ] Length warmup — 짧은 시퀀스에서 긴 시퀀스로
- [ ] 데이터 품질 필터링 효과 측정 (post-Chinchilla 후속들)
- [ ] Scaling laws 직접 측정 (작은 grid로 Chinchilla 곡선 재현)

---

## Phase 4 — Post-training (Alignment & Reasoning)

### SFT (Supervised Fine-Tuning)
- [ ] Instruction tuning 데이터셋 구성
- [ ] **LoRA / QLoRA / DoRA** 저비용 fine-tuning
- [ ] Adapter / prefix tuning 비교

### Preference Learning
- [ ] Reward model 학습
- [ ] **PPO (RLHF)** — 클래식 baseline
- [ ] **DPO** — reward model 없는 직접 최적화
- [ ] **KTO / IPO / ORPO / SimPO** 비교
- [ ] **GRPO** (DeepSeek) — group relative policy optimization
- [ ] Online DPO / iterative DPO

### Reasoning (o1 / R1 계열)
- [ ] Chain-of-Thought 데이터로 SFT
- [ ] **R1-Zero 스타일** RL on reasoning traces
- [ ] **Test-time compute scaling** — best-of-N, self-consistency, MCTS
- [ ] Reasoning **distillation** — 큰 R1 → 작은 모델
- [ ] Verifier / process reward model

---

## Phase 5 — 추론 최적화

- [ ] **KV cache** 직접 구현 (paged attention 류)
- [ ] **Continuous batching**
- [ ] Quantization: INT8 → INT4 (**GPTQ / AWQ**) → FP8 → **1-bit (BitNet)**
- [ ] ONNX export 개선 (현재 `tools/export_onnx.py` 베이스)
- [ ] **Speculative decoding** — draft model로 가속
- [ ] **EAGLE / Medusa** — 자체 decode head로 가속
- [ ] llama.cpp / ggml export
- [ ] vLLM / SGLang 비교 벤치마크

---

## Phase 6 — 평가 인프라

- [ ] Perplexity / loss curve 자동 리포트
- [ ] **lm-evaluation-harness** 통합
- [ ] **HumanEval / MBPP** — 코드
- [ ] **MMLU / GSM8K / MATH** — 지식 / 수학
- [ ] **IFEval** — instruction following
- [ ] **LLM-as-judge** (bias 분석 포함)
- [ ] Long-context 평가 (Needle-in-Haystack, RULER)
- [ ] 한국어 평가셋 (KMMLU, HAE-RAE)

---

## Phase 7 — 최신 트렌드 / 장기 실험 (2024–2026)

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

---

## Phase 8 — 한국어 / 본인 색깔

- [ ] 한국어 합성 데이터로 재학습 (sonformer-ko)
- [ ] 한국어 BPE / SentencePiece tokenizer 직접 학습
- [ ] 한국어 reasoning 데이터셋 합성
- [ ] 본인이 발견한 ablation 결과로 **모델 카드** 작성
- [ ] HuggingFace Hub 업로드 + 데모

---

## 산출물 형식

각 실험은 일관된 포맷으로 정리:
- `experiments/<phase>/<topic>/` — 학습 스크립트, config, 결과
- 학습 곡선 + ablation 표 + 메모리/속도 비교
- README 또는 블로그 글
- 재현 가능한 seed 고정 + 환경 명세
