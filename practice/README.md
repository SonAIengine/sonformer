# practice/ — 손코딩 학습 공간

> 각 폴더 = 하나의 transformer 변형/응용을 빈 폴더에서부터 직접 구현해보는 공간.
> 옆에 `solution.py`(정답지)가 있어 막힐 때만 비교. 베끼지 말고 손으로 새기기.

---

## 현재 인덱스

> **Tier 0 → Tier 2 순으로** 따라가면 "맨바닥 텐서 → 트랜스포머"까지 빈틈 없음.
> 이미 ML 경험이 있으면 Tier 0/1을 빠르게 훑고 Tier 2부터 시작해도 OK.

### 🌰 Tier 0 — 진짜 기초 (autograd / nn.Module)

| 폴더 | 무엇 | 학습 핵심 | 상태 |
|---|---|---|---|
| [00-tensor-autograd](00-tensor-autograd/) | **Tensor + autograd** 손풀이 | 손 gradient ↔ autograd 일치 검증, broadcasting, 흔한 함정 | ✅ 정답지 + 가이드 |
| [01-mlp-from-scratch](01-mlp-from-scratch/) | **MLP** on two-moons | nn.Module, ReLU 비선형성, optimizer, decision boundary | ✅ 정답지 + 가이드 |

### 🌱 Tier 1 — Pre-Transformer (왜 attention이 필요했나)

| 폴더 | 무엇 | 학습 핵심 | 상태 |
|---|---|---|---|
| [02-cnn-from-scratch](02-cnn-from-scratch/) | **1D CNN** for sequence (TextCNN) | kernel=n-gram detector, max-pool, 병렬성 | ✅ 정답지 + 가이드 |
| [03-rnn-from-scratch](03-rnn-from-scratch/) | **Vanilla RNN** (Elman) | hidden 누적, BPTT, vanishing gradient 동기 | ✅ 정답지 + 가이드 |
| [04-lstm-from-scratch](04-lstm-from-scratch/) | **LSTM** 4-gate | cell state, forget gate, 장기 의존성 | ✅ 정답지 + 가이드 |

### 🌳 Tier 2 — Transformer 가족

| 폴더 | 무엇 | 학습 핵심 | 상태 |
|---|---|---|---|
| [05-sonlm-from-scratch](05-sonlm-from-scratch/) | **Decoder-only** transformer (GPT 계열) | causal self-attn, autoregressive 생성, x/y shift | ✅ 정답지 + 가이드 |
| [06-encoder-decoder-from-scratch](06-encoder-decoder-from-scratch/) | **Encoder-Decoder** (원조 2017 paper) | **cross-attention**, sinusoidal PE, teacher forcing | ✅ 정답지 + 가이드 |
| [07-bert-from-scratch](07-bert-from-scratch/) | **Encoder-only** (BERT) + MLM | 양방향 attention, 15% 마스킹, [CLS] | ✅ 정답지 |

---

## 가족 매핑 — 한 장에 정리

```
                    ┌─────────────────────┐
                    │  Original Transformer│   (encoder + decoder)
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
        ┌─────────▼──────────┐   ┌──────────▼─────────┐
        │   Decoder-only     │   │   Encoder-only     │
        │  (causal self-attn)│   │ (bidir self-attn)  │
        │                    │   │                    │
        │  GPT, LLaMA, sonlm │   │   BERT, RoBERTa,   │
        │   → autoregressive │   │   ViT, CLIP encoder│
        │   생성 (continuation)│   │  → MLM, 분류, 임베딩│
        └────────────────────┘   └────────────────────┘

Cross-attention이 들어가면:
        Encoder-Decoder (T5, BART, Whisper, 번역 모델들)
        → 입력 시퀀스 X → 출력 시퀀스 Y
```

이 세 가지가 transformer 가족의 **세 기둥**. 거의 모든 모던 모델이 이 셋 중 하나.

---

## 다음 가능한 방향

직접 손코딩해볼 만한 것들. 난이도 / 비교 의의 / 학습 가치 순으로 묶음.

### 🟢 Tier 3 — 모던 LLM 변형 (decoder-only를 더 깊게)

본인이 만든 sonlm baseline 위에 모던 기법을 한 번에 다 얹은 LLaMA-style 모델을 한 파일로:

- [ ] **`practice/llama-from-scratch/`** — RoPE + RMSNorm + SwiGLU + GQA를 한 번에. 현재 LLaMA 3 / Mistral 계열의 표준 구조.
- [ ] **`practice/gpt2-from-scratch/`** — 원조 GPT-2 그대로 (learned PE + GELU + LayerNorm + MHA). nanoGPT 스타일.
- [ ] **`practice/mamba-from-scratch/`** — State Space Model. transformer 너머의 새 가족.

### 🟡 Tier 4 — Vision / Multimodal

- [ ] **`practice/vit-from-scratch/`** — Vision Transformer. 이미지를 patch로 잘라 토큰 취급. encoder-only(BERT 계열)와 거의 같음.
- [ ] **`practice/clip-from-scratch/`** — Vision encoder + Text encoder + contrastive loss. 두 modality embedding alignment.
- [ ] **`practice/whisper-mini/`** — Audio encoder + Text decoder (encoder-decoder 응용). 음성 인식.

### 🟠 Tier 5 — 학습 / 추론 기법 (구조 외)

- [ ] **`practice/lora-from-scratch/`** — pre-trained 모델에 low-rank adapter 붙여 fine-tuning.
- [ ] **`practice/kv-cache-from-scratch/`** — 추론 속도 최적화. 매 step마다 K, V를 캐시해서 재계산 안 함.
- [ ] **`practice/beam-search-from-scratch/`** — greedy 대신 top-k path 동시 탐색.
- [ ] **`practice/speculative-decoding/`** — small draft model로 여러 토큰 미리 만들고 큰 model로 검증.

### 🔴 Tier 6 — Post-training (Alignment)

- [ ] **`practice/dpo-from-scratch/`** — preference data로 직접 최적화. RLHF 없이 정렬.
- [ ] **`practice/ppo-from-scratch/`** — 클래식 RLHF. policy + value network + reward model.
- [ ] **`practice/grpo-from-scratch/`** — DeepSeek-R1 스타일 reasoning RL.

### 🟣 Tier 7 — 평가 / 분석

- [ ] **`practice/attention-viz/`** — attention map을 그려서 모델이 무엇에 집중하는지 시각화.
- [ ] **`practice/embedding-probing/`** — embedding의 기하 구조 분석 (nearest neighbors, t-SNE).
- [ ] **`practice/scaling-laws-mini/`** — Chinchilla 곡선 작은 grid로 직접 측정.

---

## 어떤 순서로 가는 게 좋은가

| 시나리오 | 추천 순서 |
|---|---|
| **🌰 진짜 처음 (권장)** | 00 → 01 → 02 → 03 → 04 → 05 (정공법 순차) |
| **시퀀스 모델 역사 따라가기** | 02 → 03 → 04 → 05 → 06 |
| **Transformer 가족 다 보기** | 05 → 06 → 07 → **vit** → **whisper-mini** |
| **모던 LLM 풀스택 마스터** | 05 → **llama** (Tier 3) → **kv-cache** → **lora** → **dpo** |
| **연구자 트랙** | 05 → **scaling-laws-mini** → **attention-viz** → 관심 분야 깊게 |
| **응용 엔지니어 트랙** | 05 → **lora** → **kv-cache** → **speculative-decoding** |

본인 페이스에 맞춰 골라잡기. 막히면 Claude에게 "다음 거 시작하자" 말씀하시면 새 폴더 스폰 + 가이드 작성.

---

## 작업 흐름 (모든 polder 공통)

```
1. README.md 읽기 — 무엇 / 왜 / 어떤 단계로
2. solution.py를 한 번 돌려 baseline 동작 확인 (정답지 작동 확인)
3. solution.py 닫고, 빈 파일에서 직접 작성 시작
4. 막히면:
   - 같은 폴더의 solution.py 참고 (큰 흐름)
   - 다른 폴더의 solution.py와 diff (가족 간 차이)
   - sonlm/ 학습 reference (한국어 주석 풍부 버전)
   - Claude에게 구체 질문 ("이 mask shape가 왜 안 맞아?")
5. 끝나면 commit + 다음 폴더로
```

---

## practice/ vs experiments/

| | `practice/` | `experiments/` |
|---|---|---|
| 목적 | 한 아키텍처 변형을 처음부터 끝까지 구현 | 한 컴포넌트 ablation으로 효과 측정 |
| 단위 | 보통 단일 파일 (self-contained) | 폴더 단위 (model + train + config + results) |
| 데이터 | synthetic / 작은 toy task | TinyStories / WikiText / PG-19 |
| 결과 | "동작 확인" (smoke test, accuracy) | PPL / wallclock / VRAM 비교 표 |
| 길이 | 짧음 (수 분 ~ 수십 분) | 길음 (시간 단위) |
| 누적 | 학습 노트 | 실험 데이터 |

practice는 "코드를 손에 새기는" 곳, experiments는 "기법의 효과를 숫자로 정량화"하는 곳.
