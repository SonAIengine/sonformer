# Encoder-Decoder Transformer 손코딩 (from scratch)

> 원조 2017 "Attention is All You Need" 의 **encoder-decoder** transformer를 빈 폴더에서부터 직접 구현하는 작업 공간.
> 막히면 `solution.py` 또는 그래도 안 되면 Claude에게 구체 질문.

---

## 왜 이걸 따로 공부하나?

`practice/sonlm-from-scratch/`에서 만든 건 **decoder-only** (GPT 계열). 이건 현재 LLM의 주류지만, **transformer 가족의 절반**일 뿐. 원조 transformer는 encoder-decoder 구조였고, 이게 아직도 살아있는 곳:

- 번역 (T5, mBART, NLLB)
- 요약 (BART, Pegasus)
- 음성 인식 (Whisper의 decoder 부분)
- 이미지 캡셔닝 (BLIP-2 등)

**핵심 차이점은 cross-attention.** decoder가 매 step마다 encoder의 출력을 참조하는 메커니즘. decoder-only로는 표현하기 어려운 "고정된 입력 → 가변 출력" 패턴을 자연스럽게 다룸.

---

## 두 transformer 비교

| | decoder-only (sonlm) | encoder-decoder (이 폴더) |
|---|---|---|
| Layer 구성 | self-attn + FFN | encoder: self-attn + FFN<br>decoder: self-attn + **cross-attn** + FFN |
| Mask | causal (모든 layer) | encoder: padding only<br>decoder self: causal+padding<br>cross: src padding |
| 입력/출력 | 한 시퀀스 (continuation) | 두 시퀀스 (src → tgt) |
| PE | learned (sonlm) | sinusoidal (paper-original) |
| 학습 패턴 | 다음 토큰 예측 | teacher forcing (BOS + tgt[:-1] → tgt[1:]) |
| 추론 | 단일 stream autoregressive | encoder 한 번 + decoder autoregressive |
| 대표 모델 | GPT, LLaMA, Mistral | T5, BART, mBART, Whisper |

---

## 목표

이 폴더 안에 다음을 직접 만든다:

```
practice/encoder-decoder-from-scratch/
├── README.md        ← (이 파일) 단계별 가이드
├── solution.py      ← 정답지 — 단일 파일 reference (~330 lines)
│
├── config.py        ← Config dataclass             (본인이 작성)
├── model.py         ← MHA / Encoder / Decoder / Transformer (본인이 작성)
├── masks.py         ← src_mask, tgt_mask 헬퍼      (본인이 작성)
├── data.py          ← reverse task 합성 배치 생성  (본인이 작성)
└── train.py         ← 학습 루프                     (본인이 작성)
```

**검증 task: sequence reversal**
- src: `[a, b, c, d, EOS]`
- tgt: `[BOS, d, c, b, a, EOS]`

cross-attention이 동작하지 않으면 절대 못 푸는 task. decoder-only로는 어려움 (위치 정보를 명시적으로 매핑해야).

---

## 사전 준비

```bash
pip install torch     # 외부 데이터 안 씀, tokenizers도 안 씀 (synthetic)
python -c "import torch; print(torch.cuda.is_available())"
```

`solution.py`를 먼저 한 번 돌려서 baseline 체감하기:
```bash
python solution.py
```
→ ~20초에 8/8 reverse accuracy 달성하는 모습 확인.

---

# Phase A. config.py — Config dataclass

**무엇:** 모델/학습 + 특수 토큰 ID까지 한 곳에.

**필드 (대략)**
```python
vocab_size: int = 32             # 작은 task라 32면 충분
d_model: int = 128
n_heads: int = 4
n_layers: int = 3                # encoder/decoder 둘 다
ffn_hidden: int = 512            # 4× d_model
max_seq_len: int = 16
dropout: float = 0.1

pad_id: int = 0
bos_id: int = 1
eos_id: int = 2

# 학습
batch_size: int = 64
learning_rate: float = 3e-4
warmup_steps: int = 100
max_steps: int = 1500
label_smoothing: float = 0.1
```

→ 막히면: `solution.py:Config`

---

# Phase B. PE (Sinusoidal Positional Encoding)

**무엇:** `(max_len, d_model)` 텐서 한 번 계산해서 buffer로 캐시.

**원조 공식**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**구현 힌트**
```python
pos = torch.arange(max_len).unsqueeze(1).float()        # (max_len, 1)
i = torch.arange(0, d_model, 2).float()                 # (d_model/2,)
div = torch.exp(-math.log(10000.0) * i / d_model)
pe = torch.zeros(max_len, d_model)
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div)
```

**왜 sin/cos?** 위치 차이가 회전으로 표현돼 외삽 가능. learned PE보다 학습 길이 외부에서도 안정.

→ 막히면: `solution.py:sinusoidal_pe`

---

# Phase C. MultiHeadAttention (general)

decoder-only의 attention과 다르게, **q/k/v를 다른 시퀀스에서 받을 수 있어야** cross-attention이 가능. 따라서 q, k, v projection을 **분리**.

**시그니처**
```python
def forward(self, q_in, k_in, v_in, mask=None):
    # q_in: (B, Tq, C)
    # k_in, v_in: (B, Tk, C)
    # 반환: (B, Tq, C)
```

**구현 흐름**
- `q = q_proj(q_in).view(B, Tq, H, D).transpose(1, 2)` → (B, H, Tq, D)
- k, v도 동일하게 (Tk 사용)
- attention scores `(B, H, Tq, Tk)` = `q @ k.transpose(-2, -1) / sqrt(D)`
- mask 적용 (`masked_fill(mask == 0, -inf)`)
- softmax + dropout
- `attn @ v` → transpose → contiguous().view → out_proj

**호출 패턴**
- self-attention: `attn(x, x, x, mask)`
- cross-attention: `attn(decoder_x, enc_out, enc_out, src_mask)`

→ 막히면: `solution.py:MultiHeadAttention`

---

# Phase D. EncoderLayer / DecoderLayer

**EncoderLayer (2 sub-layers, Pre-LN)**
```
x = x + self_attn(LN(x))     # bidirectional, src padding mask만 적용
x = x + ffn(LN(x))
```

**DecoderLayer (3 sub-layers, Pre-LN)** ⭐ — 핵심
```
x = x + self_attn(LN(x), tgt_mask)              # causal + tgt padding
x = x + cross_attn(LN(x), enc_out, enc_out, src_mask)   # ← 새로움
x = x + ffn(LN(x))
```

**막히기 쉬운 곳**
- DecoderLayer는 **3개의 LayerNorm**이 필요 (각 sub-layer 앞에 하나씩)
- cross-attention의 mask는 `src_mask` (encoder padding) — `tgt_mask`가 아님
- forward 시그니처에 `enc_out`, `tgt_mask`, `src_mask` 모두 받기

→ 막히면: `solution.py:EncoderLayer`, `solution.py:DecoderLayer`

---

# Phase E. Transformer (전체)

**구성**
- `tok_emb`: 공유 vocab이므로 src/tgt 같은 임베딩 (간소화)
- PE buffer: register_buffer로 한 번 계산
- `encoder`: ModuleList of EncoderLayer × n_layers
- `decoder`: ModuleList of DecoderLayer × n_layers
- 각 stack 끝에 final `LayerNorm` (Pre-LN 컨벤션)
- `out_proj`: `Linear(d_model, vocab_size, bias=False)` — weight tied with `tok_emb`

**메소드**
- `_embed(ids)`: `tok_emb(ids) * sqrt(d_model) + pe[:T]` → dropout
- `encode(src, src_mask)`: embedding → encoder layers → final LN
- `decode(tgt, enc_out, tgt_mask, src_mask)`: embedding → decoder layers → final LN
- `forward(src, tgt, src_mask, tgt_mask)`: encode + decode + out_proj
- `generate(src, src_mask, max_len)`: encoder 한 번 + greedy decoder loop

**왜 embedding × √d_model?** 원조 paper convention. embedding scale을 PE와 비슷하게 맞추는 효과.

→ 막히면: `solution.py:Transformer`

---

# Phase F. Mask utilities

**`make_src_mask(src, pad_id)`**: encoder padding mask
```python
return (src != pad_id)[:, None, None, :].long()    # (B, 1, 1, T_src)
```

**`make_tgt_mask(tgt, pad_id, T, device)`**: decoder padding + causal
```python
pad = (tgt != pad_id)[:, None, None, :].long()
causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.long))[None, None]
return pad & causal     # (B, 1, T_tgt, T_tgt)
```

**왜 mask가 두 종류?**
- encoder는 미래 봐도 됨 (입력 전체 양방향) → padding만 막으면 됨
- decoder self-attn은 미래 못 봐 (autoregressive) → causal + padding
- cross-attn에서 decoder가 encoder를 볼 때, encoder의 padding 위치는 무시해야 → src_mask 그대로 사용

→ 막히면: `solution.py:make_src_mask`, `make_tgt_mask`

---

# Phase G. Reverse task 데이터 생성

```python
def make_reverse_batch(batch_size, cfg, device):
    # 각 샘플마다 길이 3~max_seq_len/2 사이 랜덤
    # src = [a, b, c, ..., EOS, PAD, ...]
    # tgt = [BOS, ...reversed..., EOS, PAD, ...]
    # 토큰 범위: 3 ~ vocab_size-1 (PAD/BOS/EOS 회피)
    ...
```

batch마다 새 random seq → 일반화 능력 확인 가능.

→ 막히면: `solution.py:make_reverse_batch`

---

# Phase H. 학습 루프

**Teacher forcing:**
```python
tgt_in = tgt[:, :-1]    # decoder 입력: BOS, d, c, b, a
tgt_out = tgt[:, 1:]    # 정답:        d, c, b, a, EOS
logits = model(src, tgt_in, src_mask, tgt_mask)
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    tgt_out.reshape(-1),
    ignore_index=pad_id,
    label_smoothing=0.1,        # 원조 paper trick
)
```

**Optimizer**
- AdamW, `betas=(0.9, 0.98)` (원조 paper 값. decoder-only는 보통 0.95)
- weight_decay 0.01
- warmup + cosine LR

**검증**
학습 끝에 `model.generate`로 4-8개 샘플 reverse → 정답과 비교.
1500 step에서 거의 100% 정확도가 정상.

→ 막히면: `solution.py:smoke_test`

---

# 검증 체크리스트

각 Phase 끝나면:
- [ ] A. `Config()` instantiate 됨
- [ ] B. `sinusoidal_pe(16, 32).shape == (16, 32)`, 첫 row가 sin/cos 패턴
- [ ] C. `MultiHeadAttention(cfg)(q, k, v, mask)` shape OK (cross-attn shape도)
- [ ] D. `EncoderLayer(cfg)(x, mask)`, `DecoderLayer(cfg)(x, enc_out, tm, sm)` OK
- [ ] E. `Transformer(cfg)(src, tgt, sm, tm)` 가 `(B, T_tgt, V)` 반환
- [ ] F. `make_tgt_mask`가 lower-triangular + padding 조합으로 정확
- [ ] G. `make_reverse_batch`가 src/tgt shape 일관성 유지
- [ ] H. 학습 후 `model.generate`로 4/4 이상 정확히 reverse 됨

전체 통과하면, **encoder-decoder transformer를 처음부터 만들어 cross-attention의 동작을 두 눈으로 확인**한 것.

---

# 막힐 때 어떻게 도움 받기

1. **30분 직접 시도** — 답답한 만큼 손에 새겨짐.
2. 정답지 비교 (둘 중 적합한 것):
   - `solution.py` (이 폴더) — 큰 흐름이 헷갈릴 때
   - decoder-only와 어떻게 다른지 헷갈리면 `../sonlm-from-scratch/solution.py`와 diff
3. Claude에게 묻기 — 단, "이 부분에서 mask shape가 왜 이렇게?" 식의 구체 질문.

---

# 끝낸 뒤

이 코드를 sonlm-from-scratch와 한 줄씩 diff해보면, **decoder-only가 본질적으로 encoder-decoder의 어떤 부분을 잘라내고 단순화한 건지** 명확해짐:

- encoder stack 제거
- decoder의 cross-attention 제거
- decoder의 self-attention만 남음 → 이게 사실상 LM의 forward

그래서 GPT/LLaMA 같은 decoder-only 모델이 더 단순한 거고, scaling이 잘 되는 이유 중 하나.

다음 단계로 시도해볼 만한 것:
- 진짜 번역 task (간단한 영-한 parallel data, ~10K samples)
- Whisper 스타일 (audio encoder + text decoder)
- BART 스타일 pre-training (denoising autoencoder)
