# sonformer Roadmap — LLM 풀스택 마스터 커리큘럼

> 이 한 파일을 다 통과하면 현대 LLM의 모든 핵심 컴포넌트를 **이해하고, 직접 구현하고, 측정하고, 배포**까지 할 수 있다.
> 사이클: `개념 → 손코딩 → 학습/측정 → 비교/정리`.

**구성 원칙**
- 각 항목은 *체크 가능한* 단위 (개념 1개 / 구현 1개 / 측정 1개)
- 핵심 항목엔 권위있는 reference (paper/repo) 명시
- `practice/` 폴더로 손코딩 가능한 항목엔 ✅ 와 폴더 번호
- `experiments/` 폴더로 ablation 가능한 항목 표기

축약 표는 [README.md](README.md#roadmap-11-phase) 참고.

---

## Phase 0 — 수학 기초

전체 LLM의 *진짜* 바닥. 이걸 모르면 위 모든 게 마법으로 보임.

### 선형대수
- [ ] 벡터/행렬 연산, 내적/외적, 노름 (L1/L2/Frobenius)
- [ ] 고유값분해 (eigendecomposition), 대각화
- [ ] **SVD** (Singular Value Decomposition) — LoRA의 수학적 기반
- [ ] **행렬 미분** (matrix calculus) — gradient 손풀이의 무기
- [ ] Einstein notation (`einsum`) — `B H T D` 추적의 표준

### 미적분 / 최적화
- [ ] 편미분, gradient, Jacobian, Hessian
- [ ] **Chain rule** — backprop 의 본질
- [ ] Convex optimization 기초 (왜 convex가 좋은지)
- [ ] **Gradient descent 변형 계보**: SGD → Momentum → Nesterov → Adagrad → RMSProp → **Adam → AdamW**
- [ ] Second-order: Newton, Gauss-Newton, **K-FAC** (Sophia/Shampoo의 기반)

### 확률 / 정보이론
- [ ] 조건부 확률, Bayes' theorem
- [ ] **MLE / MAP** — language modeling의 출발점
- [ ] 분포: Gaussian, Bernoulli, Categorical, Dirichlet
- [ ] **Entropy / Cross-entropy / KL divergence** — LM loss의 본질
- [ ] **Mutual information** — RLHF/contrastive learning에서 등장
- [ ] **Gumbel-softmax** — 이산 sampling을 미분 가능하게

### Reference
- 📚 *Mathematics for Machine Learning* (Deisenroth) — 무료 PDF
- 📚 *Pattern Recognition and Machine Learning* (Bishop)
- 📺 3Blue1Brown — Linear Algebra / Calculus 시리즈
- 📺 Stanford CS229 — Andrew Ng

---

## Phase 1 — ML/DL 기초

### Classical ML (개념만)
- [ ] Linear regression (closed-form vs gradient descent)
- [ ] Logistic regression — *first neural network*
- [ ] **Bias-variance tradeoff** — overfitting/underfitting의 정체
- [ ] Cross-validation, train/val/test 분할
- [ ] Regularization: L1/L2, dropout, early stopping, data augmentation
- [ ] Ensemble: bagging, boosting (개념만)

### Deep Learning Fundamentals
- [x] **Tensor + Autograd 손풀이** — [practice/00-tensor-autograd](practice/00-tensor-autograd/)
- [x] **MLP + backprop** — [practice/01-mlp-from-scratch](practice/01-mlp-from-scratch/)
- [ ] 활성화 함수 계보: sigmoid → tanh → **ReLU** → LeakyReLU → GELU → SiLU/Swish → **SwiGLU**
- [ ] **Initialization**: Xavier (Glorot), He (Kaiming), depth-scaled, μP
- [ ] **Normalization 가족**: BatchNorm → **LayerNorm** → GroupNorm → **RMSNorm** → Weight Standardization
- [ ] Dropout (vanilla, DropConnect, attention dropout)
- [ ] **Skip connections / Residual** — ResNet의 통찰, Transformer의 backbone
- [ ] **Vanishing / Exploding gradient** — LSTM → Transformer 진화 동기

### 디버깅 / 진단 능력 (실전 무기)
- [ ] gradient norm 추적 (각 layer별)
- [ ] activation distribution 추적 (dead neurons 찾기)
- [ ] Loss curve 분석 (과적합/언더피트 패턴)
- [ ] **NaN/Inf 추적** (어디서 터지는지)
- [ ] **Sanity check**: tiny data로 overfit 시켜보기
- [ ] **Gradient check** (수치 미분 vs autograd 비교)

### Reference
- 📚 Goodfellow *Deep Learning* (free online)
- 🎓 Stanford CS231n / CS224n
- 📺 **Andrej Karpathy "Neural Networks: Zero to Hero"** ⭐

---

## Phase 2 — 시퀀스 모델 (Pre-Transformer → Transformer)

### Pre-Transformer (왜 attention이 필요했나)
- [x] **CNN for sequence (TextCNN)** — [practice/02-cnn](practice/02-cnn-from-scratch/)
- [x] **Vanilla RNN (Elman)** — [practice/03-rnn](practice/03-rnn-from-scratch/)
- [x] **LSTM** 4-gate — [practice/04-lstm](practice/04-lstm-from-scratch/)
- [ ] **GRU** — LSTM 단순화 변형
- [ ] **Bidirectional RNN/LSTM**
- [ ] Seq2seq encoder-decoder (RNN 시대)
- [ ] **Bahdanau / Luong attention** — original attention paper (2014/2015)

### Transformer 본체
- [x] **Decoder-only** (GPT 계열) — [practice/05-sonlm](practice/05-sonlm-from-scratch/)
- [x] **Encoder-Decoder** (원조 2017) — [practice/06-encoder-decoder](practice/06-encoder-decoder-from-scratch/)
- [x] **Encoder-only / BERT + MLM** — [practice/07-bert](practice/07-bert-from-scratch/)
- [ ] Multi-head Attention 상세 — head별 attention pattern 시각화
- [ ] **Causal mask vs Padding mask** 정확히 구분
- [ ] **Pre-LN vs Post-LN** — 학습 안정성 차이
- [ ] **Weight tying** — 입력 embedding ↔ 출력 분류기

### 핵심 paper 읽고 노트 정리
- [ ] *Attention Is All You Need* (Vaswani 2017)
- [ ] *BERT* (Devlin 2018)
- [ ] *GPT-2 / GPT-3* (Radford 2019, Brown 2020)
- [ ] *Scaling Laws for Neural Language Models* (Kaplan 2020)
- [ ] *T5: Text-to-Text Transfer Transformer* (Raffel 2020)

---

## Phase 3 — Transformer 아키텍처 변형 (모던 LLM의 표준)

vanilla 대비 **PPL / wallclock / VRAM**으로 ablation. 모든 항목 `experiments/` 폴더로 측정.

### Positional Encoding
- [ ] Learned PE vs **Sinusoidal PE** (paper original)
- [x] **RoPE (Rotary)** — LLaMA 표준 [(02a-rope ✅)](experiments/02a-rope/)
- [ ] **ALiBi** — 외삽 가능 bias (BLOOM)
- [ ] **NoPE** — PE 없이 학습 가능한지
- [ ] **장문 외삽**: PI (Position Interpolation), NTK-aware RoPE, **YaRN**, **LongRoPE**
- [ ] **Attention Sinks** — StreamingLLM (긴 입력에서 첫 토큰의 역할)

### Attention 변형
- [ ] MHA → **MQA** (Multi-Query) — KV head 1개로 통일
- [ ] MHA → **GQA** (Grouped-Query) — LLaMA 2/3 표준
- [ ] **MLA** (Multi-head Latent Attention) — DeepSeek-V2/V3
- [ ] **Sliding Window Attention** — Mistral
- [ ] **Sparse Attention**: Longformer, BigBird
- [ ] **Native Sparse Attention (NSA)** — DeepSeek 2025
- [ ] **FlashAttention 1/2/3** — IO-aware exact attention
- [ ] **Ring Attention** — sequence parallel for long context
- [ ] **PagedAttention** — vLLM의 KV cache 최적화
- [ ] **Linear Attention / Performer** — 부드러운 근사

### FFN / Activation
- [ ] ReLU → GELU → **SwiGLU** / GeGLU 비교
- [ ] **Mixture of Experts (MoE)** — Switch / Mixtral 스타일 top-k routing
- [ ] MoE routing: top-k, **expert choice**, **soft MoE**
- [ ] Aux loss / load balancing 구현
- [ ] **Fine-grained MoE** — DeepSeek 256-expert 구조
- [ ] **Shared experts** (DeepSeek 변형)
- [ ] **Upcycling**: dense → MoE 변환

### Normalization
- [ ] LayerNorm → **RMSNorm** — 더 빠르고 동등한 성능
- [ ] **QK-Norm** — query/key에 추가 norm (학습 안정화)
- [ ] **DeepNorm** — 깊은 모델 안정화 (Microsoft)
- [ ] Pre-LN vs Post-LN vs **Sandwich Norm** (NormFormer)
- [ ] **Parallel Residual** (PaLM/GPT-J 스타일) — attn과 ffn을 병렬

### Reference
- 📄 RoFormer (RoPE), GQA paper
- 📄 FlashAttention 1/2/3
- 📄 Mixtral 8x7B technical report
- 📄 DeepSeek-V3 technical report (2024)

---

## Phase 4 — 학습 최적화

### Optimizer 계보
- [ ] AdamW (현재 baseline) → **Lion** (sign-based, 메모리 ↓)
- [ ] **Sophia** (second-order Hessian 근사)
- [ ] **Muon** (orthogonal momentum, 2024) — 최신 SOTA optimizer
- [ ] **Shampoo** (matrix-aware preconditioner)
- [ ] Adafactor — 저메모리 학습
- [ ] **Distributed Shampoo** (실용 버전)

### LR Schedule / Init
- [ ] Linear warmup + cosine decay (현재 표준)
- [ ] **WSD** (Warmup-Stable-Decay) — MiniCPM/DeepSeek
- [ ] **μP** (maximal update parametrization) — width-invariant scaling
- [ ] **z-loss** / auxiliary loss
- [ ] Init scale ablation: small init, depth-scaled init, μP init

### Mixed Precision / 메모리
- [ ] fp32 → fp16 → **bf16** → **fp8** (H100/H200) 비교
- [ ] Loss scaling (fp16) vs bf16 (불필요)
- [ ] **Gradient checkpointing** (메모리 ↔ 속도 tradeoff)
- [ ] **Gradient accumulation** (effective batch size ↑)
- [ ] **Sequence packing** (짧은 샘플 묶어 throughput ↑)
- [ ] **Activation offloading** (CPU/NVMe로 swap)
- [ ] CUDA OOM 진단 + memory profiler 활용

### Distributed Training (병렬화)
- [ ] **Data Parallel (DP)** — 모델 복제, 가장 단순
- [ ] **DDP** (DistributedDataParallel) — DP의 효율 버전
- [ ] **DeepSpeed ZeRO-1/2/3** — optimizer/grad/param sharding
- [ ] **FSDP** (PyTorch 네이티브)
- [ ] **Tensor Parallel (TP)** — Megatron-LM 스타일 (행/열 분할)
- [ ] **Pipeline Parallel (PP)** — bubble overhead 이해
- [ ] **Sequence Parallel (SP)** — Megatron-LM
- [ ] **Expert Parallel (EP)** — MoE 전용
- [ ] **3D / 4D parallelism** — 여러 차원 결합
- [ ] **Communication primitives**: AllReduce, AllGather, ReduceScatter
- [ ] **Comm-compute overlap** — 통신/계산 동시 진행

### Scaling Laws
- [ ] **Chinchilla** 곡선 직접 측정 (작은 grid)
- [ ] **Compute-optimal** training — 데이터/모델/compute 비율
- [ ] **Emergent abilities** — phase transition 측정
- [ ] **Inference scaling laws** (test-time compute scaling)
- [ ] Model 크기 vs 데이터 vs FLOPs 트레이드오프

### GPU 하드웨어 / 메모리 hierarchy
"학습이 왜 이 속도인가"를 답하려면 하드웨어 구조 이해가 필수.
- [ ] **GPU 메모리 계층**: HBM (80GB / 3TB/s) → L2 cache → SMEM/L1 → registers
      각 layer의 capacity / latency / bandwidth 외워두기
- [ ] **SM (Streaming Multiprocessor)** 구조 — H100 132 SMs, 각 SM의 warp scheduler
- [ ] **Thread hierarchy**: thread → **warp (32)** → block → grid
- [ ] **Tensor Cores** — fp16/bf16/fp8/int8 matrix engine, mma instruction
- [ ] **Roofline model** — compute-bound vs memory-bound 식별
- [ ] **Arithmetic intensity** 계산 (FLOPs / byte)
- [ ] H100 vs A100 vs H200 vs B200 spec 비교 (FLOPS, 메모리, 대역폭)

### CUDA / Kernel 직접 작성
- [ ] **CUDA C++** 기초 — vector add → reduction → matrix multiply
- [ ] 메모리 **coalescing** — global memory access 최적화
- [ ] **Bank conflict** — shared memory 접근 패턴
- [ ] **Warp-level primitives**: shuffle, vote, ballot
- [ ] **CUDA streams** + async copy (memcpy_async)
- [ ] **CUDA Graph** — kernel launch overhead 제거
- [ ] **Triton** — Python에서 CUDA 커널 작성 (NVIDIA 직접 추천)
- [ ] **FlashAttention** Triton 버전 직접 구현 (SRAM-aware tiling)
- [ ] **Fused kernel** 작성: Linear + GELU, RMSNorm + RoPE 등
- [ ] cuBLAS / cuDNN vs custom kernel 벤치마크

### 통신 / Network (멀티노드 학습 핵심)
- [ ] **NVLink / NVSwitch** — intra-node GPU↔GPU (H100 900 GB/s)
- [ ] **InfiniBand (IB)** 기초 — HCA, switch, routing
- [ ] **RDMA** — kernel bypass, zero-copy 통신
- [ ] **GPUDirect RDMA** — GPU 메모리 → IB → 다른 GPU 직접 (CPU 우회)
- [ ] **NCCL** 내부 — ring vs tree vs **double-binary tree** allreduce 알고리즘
- [ ] **NCCL collective**: AllReduce, AllGather, ReduceScatter, Broadcast
- [ ] **Topology-aware** allreduce (NVLink intra + IB inter)
- [ ] **Comm-compute overlap** — backward와 allreduce를 동시 실행
- [ ] **Bandwidth/latency 측정**: `nccl-tests` 직접 돌리기
- [ ] **EFA (AWS), Slingshot (HPE)** 등 IB 대체

### Profiling / 측정 도구
"느리다"를 증명하려면 도구 다룰 줄 알아야 함.
- [ ] **nsight-systems** (`nsys`) — timeline 분석, kernel/comm/cpu overlap 시각화
- [ ] **nsight-compute** (`ncu`) — kernel 단위 metric (occupancy, memory throughput)
- [ ] **PyTorch Profiler** + Chrome trace
- [ ] **`nvidia-smi` / `nvtop`** — 실시간 GPU 상태
- [ ] **`nvprof` / DCGM** — production 모니터링
- [ ] **Triton autotuner** — kernel 자동 튜닝
- [ ] CUDA event 기반 wallclock 측정 vs `time.time()` 차이

### Reference
- 📚 *Programming Massively Parallel Processors* (Kirk & Hwu) — CUDA 표준 교과서
- 📚 *CUDA C++ Best Practices Guide* (NVIDIA 공식)
- 📄 *Training Compute-Optimal LLMs* (Chinchilla)
- 📄 *μTransfer* / *Sophia* / *Muon* / *Shampoo*
- 📄 *Megatron-LM* / *DeepSpeed* / *FSDP* 논문
- 📄 *FlashAttention* — 메모리 hierarchy를 모델 설계에 반영한 모범 사례
- 📄 *Bringing HPC techniques to Deep Learning* (Baidu, allreduce 알고리즘 origin)
- 📦 **NVIDIA `cutlass`** — CUDA 매트릭스 연산 reference
- 📦 **`nccl-tests`** — collective bandwidth 측정 표준 도구
- 📦 PyTorch FSDP tutorial, DeepSpeed tutorial
- 📺 NVIDIA GTC 세션: GPU architecture deep dive (매년)

---

## Phase 5 — 데이터 / 토크나이저

데이터가 모델 결과의 80%를 결정. 자주 무시되는 핵심.

### 토크나이저
- [ ] **BPE** (Byte-Pair Encoding) — 직접 구현
- [ ] **WordPiece** (BERT) — BPE 변형
- [ ] **SentencePiece** / **Unigram LM** — 언어 무관
- [ ] **Tiktoken** (GPT-4) 분석
- [ ] **Byte-level BPE** — `<0xE2><0x97>` 같은 처리
- [ ] **Vocabulary size** ablation (PPL vs 학습 비용)
- [ ] 한국어 토크나이저 — 형태소 vs subword 비교

### 데이터 처리 파이프라인
- [ ] **Deduplication**: exact, near-dup (MinHashLSH)
- [ ] Quality filtering: perplexity, classifier-based, rule-based
- [ ] **Data mixing** weights (RedPajama, FineWeb 사례)
- [ ] **Document packing** (concatenation with EOS)
- [ ] **Contamination check** (eval set leakage)
- [ ] Multi-source 비율: web, code, math, books, papers

### Public 데이터셋 (학습 가능)
- [ ] **The Pile** / **RedPajama** / **FineWeb (Edu)** / **Dolma**
- [ ] Code: **The Stack v2**
- [ ] Math: **Open-Web-Math**, **Proof-Pile-2**
- [ ] 한국어: 모두의 말뭉치, AI Hub, **HAE-RAE**

### Synthetic Data (2024-2025 핵심 트렌드)
- [ ] **Phi 시리즈** 접근 (textbook quality)
- [ ] **Self-instruct** (모델로 데이터 만들기)
- [ ] **Distillation from frontier** (GPT-4/Claude로 데이터 생성)
- [ ] **Persona-driven synthesis** (PERSONA Hub)
- [ ] **Reverse instructing** (output→input)

### Reference
- 📄 *FineWeb* / *Dolma* / *RedPajama* technical reports
- 📄 *Phi-3 / Phi-4* technical reports
- 📦 `datasets` (HuggingFace), `tokenizers` 라이브러리

---

## Phase 6 — Post-training (Alignment & Reasoning)

### SFT (Supervised Fine-Tuning)
- [ ] Instruction tuning 데이터셋 구성 + 필터링
- [ ] **Conversation format**: ChatML, LLaMA-3 chat template
- [ ] **Loss masking** (system/user 발화는 loss 제외)
- [ ] **Multi-turn** 학습 전략

### PEFT (Parameter-Efficient Fine-Tuning)
- [ ] **LoRA** — 직접 구현 + rank/alpha ablation
- [ ] **QLoRA** — 4-bit base + LoRA adapter
- [ ] **DoRA** (weight decomposed)
- [ ] **Prompt tuning** / **Prefix tuning** / **P-tuning v2**
- [ ] **Adapter modules** (Houlsby, Pfeiffer)
- [ ] **(IA)³** — adapter의 sparse 버전
- [ ] **GaLore** — gradient low-rank projection

### Preference Learning
- [ ] **Reward model** 학습 (Bradley-Terry)
- [ ] **PPO (RLHF)** — 클래식 baseline. Schulman 2017 + InstructGPT
- [ ] **DPO** — reward model 없는 직접 최적화 (Rafailov 2023)
- [ ] **KTO** (Kahneman-Tversky) — binary feedback
- [ ] **IPO** (Identity Preference Optimization)
- [ ] **ORPO** — SFT + preference 합침
- [ ] **SimPO** — reference-free
- [ ] **GRPO** (DeepSeek) — group relative policy
- [ ] **RLAIF** / **Constitutional AI** (Anthropic)
- [ ] Online DPO / iterative DPO

### Reasoning (o1 / R1 계열) — 2024-2025의 주축
- [ ] CoT (Chain-of-Thought) 데이터로 SFT
- [ ] **R1-Zero 스타일** RL (verifiable reward만)
- [ ] **Test-time compute scaling**: best-of-N, self-consistency, **MCTS**
- [ ] **Reasoning distillation**: 큰 R1 → 작은 모델
- [ ] **Process Reward Model (PRM)** vs Outcome Reward Model (ORM)
- [ ] **Speculative reasoning** (early exit when confident)
- [ ] **Self-correction / Self-refine**

### Safety / Alignment
- [ ] **Red-teaming** automation: GCG, PAIR, AdvBench
- [ ] Refusal training (helpful vs harmless tradeoff)
- [ ] **Constitutional AI** approach
- [ ] Jailbreak 분석 (DAN, prompt injection patterns)
- [ ] **Watermarking** generated text
- [ ] **Sleeper agent** / backdoor 탐지

### Reference
- 📄 *InstructGPT*, *DPO*, *Constitutional AI*
- 📄 **DeepSeek-R1** technical report (2025)
- 📄 *o1 system card* (OpenAI)
- 📄 *Llama-3 / Llama-3.1* post-training reports

---

## Phase 7 — 추론 최적화 / 서빙

### KV Cache 직접 구현
- [ ] **Naive KV cache** — autoregressive 가속의 출발
- [ ] **Paged KV cache** (vLLM) — fragmentation 해결
- [ ] **Sliding window** + cache eviction
- [ ] **Multi-query batching** + KV cache sharing
- [ ] **Prefix caching** — 공통 system prompt 재사용

### Quantization
- [ ] **INT8**: LLM.int8, SmoothQuant
- [ ] **INT4**: **GPTQ**, **AWQ**, **GGUF** (llama.cpp)
- [ ] **FP8** (H100/H200 native)
- [ ] **1-bit / 1.58-bit**: BitNet
- [ ] **Mixed precision per layer** (sensitive layers 보호)
- [ ] **PTQ vs QAT** (post-training vs quant-aware training)

### Speculative / Decoding 가속
- [ ] **Speculative decoding** — draft model + verify (Leviathan 2023)
- [ ] **Self-speculative** (큰 모델의 early layer가 draft)
- [ ] **EAGLE / EAGLE-2** (자체 decode head)
- [ ] **Medusa** (multi-head decoding)
- [ ] **Lookahead decoding** (n-gram cache)

### Serving 인프라
- [ ] **Continuous batching** vs static batching
- [ ] **vLLM** 아키텍처 분석 (PagedAttention, scheduler)
- [ ] **TGI** (HuggingFace Text Generation Inference)
- [ ] **SGLang** — RadixAttention prefix sharing
- [ ] **llama.cpp / GGUF** workflow (CPU/Mac inference)
- [ ] **TensorRT-LLM** (NVIDIA, 최고 성능)
- [ ] **MLX** (Apple Silicon)

### Sampling 전략
- [ ] **Greedy / Beam search** (왜 LLM에선 보통 안 씀)
- [ ] **Temperature / Top-k / Top-p (nucleus)**
- [ ] **Min-p sampling** (2024)
- [ ] **DRY** sampling (반복 방지)
- [ ] **Logit bias / Constrained generation** (JSON schema 등)
- [ ] **Guided decoding** (Outlines, lm-format-enforcer)

### Reference
- 📄 *FlashAttention*, *PagedAttention* (vLLM), *EAGLE*
- 📄 *GPTQ* / *AWQ* / *BitNet*
- 📦 **vLLM, llama.cpp 코드 직접 읽기** ⭐

---

## Phase 8 — 평가 + 안전 + 해석성

### 평가 인프라
- [ ] Perplexity / loss curve 자동 리포트
- [ ] **lm-evaluation-harness** 통합
- [ ] **HumanEval / MBPP / LiveCodeBench** — 코드
- [ ] **MMLU / MMLU-Pro / GPQA** — 지식
- [ ] **GSM8K / MATH / AIME** — 수학
- [ ] **IFEval** — instruction following
- [ ] **ARC-AGI** — abstract reasoning
- [ ] **Long-context**: Needle-in-Haystack, **RULER**, ZeroSCROLLS, BABILong
- [ ] **한국어**: KMMLU, HAE-RAE, KoBEST, KoBigBench
- [ ] **Multimodal**: MMMU, MMBench, MathVista

### LLM-as-Judge
- [ ] **Chatbot Arena** (LMSYS) 평가 방식
- [ ] **Pairwise comparison** judge bias 분석 (position, length, style)
- [ ] **Calibration** of judge
- [ ] **MT-Bench** — multi-turn 평가

### 해석성 (Interpretability)
- [ ] **Probing** (linear probes on hidden states)
- [ ] **Attention pattern 분석**: induction heads, name mover heads
- [ ] **Activation patching** / Causal mediation analysis
- [ ] **Sparse Autoencoders (SAE)** — feature decomposition (Anthropic 2024)
- [ ] **Circuit analysis** — mechanistic interpretability
- [ ] **Logit lens** / **Tuned lens**
- [ ] **Steering vectors** — activation을 조작해 행동 제어

### Reference
- 📦 `lm-evaluation-harness` (EleutherAI)
- 📄 Anthropic *Scaling Monosemanticity* (SAE, 2024)
- 📺 **Neel Nanda mech interp 강의** ⭐
- 📦 `TransformerLens` (mech interp 도구)

---

## Phase 9 — 최신 트렌드 + Multimodal + Agent (2024-2026)

### 비-Transformer 아키텍처
- [ ] **Mamba / Mamba-2** — Selective State Space Models
- [ ] **Hyena / H3** — long convolution
- [ ] **RWKV** — RNN 부활
- [ ] **RetNet** — retention 메커니즘
- [ ] **Hybrid**: **Jamba** (Transformer + Mamba), **Zamba**
- [ ] **TTT (Test-Time Training)** layers
- [ ] **xLSTM** — LSTM의 modern 부활

### Long Context (긴 문맥)
- [ ] **Ring Attention** — sequence parallel
- [ ] **YaRN / LongRoPE** — RoPE 외삽 기법
- [ ] **Attention Sinks / StreamingLLM**
- [ ] **Context compression**: AutoCompressor, GIST tokens, LLMLingua
- [ ] **1M+ context** 학습/평가 (Gemini 1.5, Claude 100k+)

### Multi-Token Prediction / 가속 학습
- [ ] **MTP** — DeepSeek-V3 (학습 + 추론 가속)
- [ ] **Medusa** (post-hoc MTP)
- [ ] Loss balancing for multi-position predictions

### Multimodal
- [ ] **Vision Encoder + LM**: LLaVA, MiniCPM-V, Qwen-VL
- [ ] **CLIP** contrastive (vision ↔ text)
- [ ] **SigLIP** — sigmoid loss 변형
- [ ] **Audio**: Whisper encoder + LM
- [ ] **Native multimodal**: Chameleon, **GPT-4o**, **Gemini 1.5/2**
- [ ] **Image generation** + LM 결합 (Diffusion + Transformer)
- [ ] **Video understanding** — temporal modeling
- [ ] **Speech-in / Speech-out** (Moshi, GPT-4o voice)

### RAG (Retrieval-Augmented Generation)
- [ ] Embedding model 학습 (contrastive, **BGE**, **E5**)
- [ ] Vector DB: FAISS, Qdrant, Chroma, **Milvus**
- [ ] **Reranker** (cross-encoder)
- [ ] **Hybrid search** (dense + sparse BM25)
- [ ] **GraphRAG** — knowledge graph 결합 (Microsoft 2024)
- [ ] **HyDE / Query rewriting**
- [ ] End-to-end retrieval-LM joint training (RAG, REPLUG)
- [ ] **Long-context vs RAG** 비교

### Agent / Tool Use
- [ ] **Function calling** 학습 (JSON schema 출력)
- [ ] **ReAct** (reasoning + acting interleaved)
- [ ] **Toolformer** — self-supervised tool 호출
- [ ] **Computer use** (Anthropic 2024) — GUI 조작
- [ ] **Multi-agent orchestration** (planner + worker)
- [ ] **AutoGen / CrewAI / LangGraph** 패턴
- [ ] **MCP** (Model Context Protocol) — Anthropic 표준

### Diffusion LLMs (실험적, 2025)
- [ ] **LLaDA** (2025) — discrete diffusion for text
- [ ] Continuous diffusion text models
- [ ] Comparison vs autoregressive (속도, 품질)

### Synthetic Data 생성 / 셀프 향상
- [ ] **Self-rewarding LMs** (Yuan 2024)
- [ ] **STaR / V-STaR** — self-taught reasoner
- [ ] **rStar-Math** — Monte Carlo 기반 reasoning 데이터 생성

### Reference
- 📄 Mamba, Mamba-2, Jamba
- 📄 **DeepSeek-V3 / R1** technical reports (2024-2025)
- 📄 LLaVA-1.5/1.6, GPT-4o system card
- 📄 *Computer use* (Anthropic), *MCP* spec

---

## Phase 10 — 한국어 + 본인 색깔

### 한국어 모델
- [ ] 한국어 합성 데이터로 재학습 (sonformer-ko)
- [ ] 한국어 BPE / SentencePiece 직접 학습 (형태소 기반 vs subword 비교)
- [ ] 한국어 reasoning 데이터셋 합성
- [ ] **KMMLU / HAE-RAE / KoBEST** 평가

### 산출물 형식
- [ ] **모델 카드** 작성 (의도된 사용 / 한계 / 편향 / 학습 데이터)
- [ ] **HuggingFace Hub** 업로드 + 데모 (Spaces)
- [ ] 본인 ablation 결과 정리 (블로그/노트)
- [ ] 재현 가능 환경 명세 (seed, hardware, time, requirements)

### 응용 프로젝트 (선택)
- [ ] 한국어 코드 어시스턴트 (Ko-Code)
- [ ] 한국어 reasoning 모델 (Ko-R1)
- [ ] 도메인 특화: 법률 / 의료 / 논문

---

## 산출물 형식 (전체 공통)

각 실험은 일관된 포맷으로 정리:
- `experiments/<phase>/<topic>/` — 학습 스크립트, config, 결과
- 학습 곡선 + ablation 표 + 메모리/속도 비교
- README 또는 블로그 글
- 재현 가능한 seed 고정 + 환경 명세

---

## 추천 학습 순서

| 시나리오 | 우선순위 |
|---|---|
| **빠른 LLM 마스터 (3~6개월)** | 0(빠르게) → 1 → 2(practice 0~7) → 3(주요만: RoPE/GQA/SwiGLU/RMSNorm) → 6(LoRA + DPO) |
| **연구자 트랙** | 0 → 1 → 2 → 3 → 4(scaling laws) → 8(interp/SAE) → 9 |
| **응용 엔지니어** | 1 → 2 → 6(LoRA/SFT) → 7(serving: vLLM/llama.cpp) → 9(RAG/Agent) |
| **풀스택 (1년+)** | 0 → 1 → 2 → ... → 10 모두 |

---

## 참고 — 권위있는 외부 커리큘럼

본인 진도 점검에 비교하면 좋음:
- 📺 **Andrej Karpathy "Neural Networks: Zero to Hero"** — 모든 게 여기서 시작
- 📚 *Stanford CS336: Language Modeling from Scratch* (2024+)
- 📚 *Speech and Language Processing* (Jurafsky & Martin) — 무료
- 📦 *nanoGPT* / *minGPT* (Karpathy)
- 📦 *llm-course* (mlabonne) — visual roadmap
- 📦 *Awesome-LLM* (Hannibal046) — paper 모음
