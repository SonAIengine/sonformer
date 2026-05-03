# sonlm 손코딩 (from scratch)

> Vanilla decoder-only transformer를 **빈 폴더에서부터** 직접 손으로 구현하는 작업 공간.
> 막히면 `solution.py` (단일 파일 정답지) 또는 `../../sonlm/` (모듈 분리 정답지)를 비교.
> 단, 가능하면 끝까지 안 보고 만들어보기.

---

## 목표

이 폴더 안에 다음을 직접 만든다:

```
practice/sonlm-from-scratch/
├── README.md        ← (이 파일) 단계별 가이드
├── solution.py      ← 정답지 — 모든 걸 한 파일에 담은 nanoGPT 스타일 reference
│
├── config.py        ← 하이퍼파라미터 dataclass         (본인이 작성)
├── model.py         ← Attention / FFN / Block / SonLM (본인이 작성)
├── dataset.py       ← JSONL 로드 + (x, y) shift       (본인이 작성)
├── train.py         ← optimizer + LR schedule + 루프  (본인이 작성)
└── inference.py     ← (옵션) chat 루프                 (본인이 작성)
```

완성 후 `python -c "from model import SonLM; ..."` 로 forward가 도는 게 검증 끝.

---

## 정답지 두 가지

| 정답지 | 스타일 | 언제 보면 좋은가 |
|---|---|---|
| **`solution.py`** (이 폴더) | 단일 파일, 개발자스러운 압축 (nanoGPT 풍) | 큰 그림 한 번에 보고 싶을 때, 학습 루프까지 한 흐름으로 따라가고 싶을 때 |
| **`../../sonlm/`** | 파일 분리 + 한국어 주석 풍부 | 특정 컴포넌트 (`model.py`만, `dataset.py`만)에서 막혔을 때 |

`solution.py`는 그 자체로 동작하는 reference: `python solution.py` 실행 시 random
데이터로 smoke test (~0.5초). 실제 데이터로 학습하려면 `python solution.py train ...`.

---

## 학습 방식

1. **순서대로 한 파일씩** 구현. 한 파일 끝나면 작은 import 테스트로 동작 확인.
2. **막히면 단계별 hint를 먼저 본다.** 그래도 안 되면 정답지 비교.
3. **베끼지 않는다.** 정답 코드를 보더라도 닫고 다시 직접 친다 — 손에 새기는 게 목적.
4. **각 Phase 끝나면 commit.** 진행 흔적이 남으면 어디서 막혔는지 나중에 보임.

---

## 사전 준비

```bash
pip install torch tokenizers     # 둘 다 시스템에 이미 깔려있을 가능성 큼
python -c "import torch; print(torch.cuda.is_available())"
```

학습 데이터/토크나이저는 일단 신경 안 써도 됨 — Phase F에서 합류.

---

# Phase A. config.py — 하이퍼파라미터

**무엇:** 모든 모델/학습 파라미터를 한 군데로 묶기.

**왜:** 흩어진 매직 넘버는 학습 불가능. dataclass 한 곳 → 모델로 흘리는 패턴.

**힌트**
- `from dataclasses import dataclass` 사용
- 필요한 필드 (대략): `vocab_size`, `max_seq_len`, `d_model`, `n_layers`, `n_heads`,
  `ffn_hidden`, `dropout`, `pad_id`
- `d_model % n_heads == 0` 이어야 함 (assert로 막아두면 좋음)

**검증**
```python
from config import SonLMConfig
cfg = SonLMConfig()
print(cfg.d_model, cfg.n_heads)
```

→ 막히면: `sonlm/config.py`

---

# Phase B. model.py — Attention

**무엇:** Multi-head self-attention 모듈 하나.

**입력 shape:** `x: (B, T, C)`, optional `mask`
**출력 shape:** `(B, T, C)`

**구현 순서**
1. `__init__`: `qkv = Linear(C, 3*C)`, `out = Linear(C, C)`, `head_dim = C // n_heads`
2. `forward`:
   - `qkv(x)` → reshape → permute로 q, k, v 분리 (각각 `(B, H, T, D)`)
   - `q @ k.transpose(-2,-1) / sqrt(head_dim)` → `(B, H, T, T)`
   - mask 있으면 `masked_fill(mask==0, -inf)` (causal)
   - softmax → dropout
   - `attn @ v` → transpose → contiguous().view → `(B, T, C)`
   - `self.out(...)` 리턴

**막히기 쉬운 곳**
- `permute(2, 0, 3, 1, 4)` 같은 재배열 — 종이에 shape 그려가며 따라가기
- mask broadcasting: `(1, 1, T, T)` 모양으로 넣어야 `(B, H, T, T)`에 broadcast됨

**검증**
```python
import torch
from config import SonLMConfig
from model import Attention
cfg = SonLMConfig(d_model=32, n_heads=4)
attn = Attention(cfg)
x = torch.randn(2, 8, 32)
out = attn(x)
print(out.shape)  # (2, 8, 32)
```

→ 막히면: `sonlm/model.py`의 `Attention`

---

# Phase C. model.py — FFN, Block

**FFN (가장 쉬움):** `Linear → ReLU → Linear → Dropout`. 중간 차원은 `ffn_hidden` (보통 4× d_model).

**Block (Pre-LN):**
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
return x
```

→ 막히면: `sonlm/model.py`의 `FFN`, `Block`

---

# Phase D. model.py — SonLM (전체)

**무엇:** 토큰 입력을 받아 logits을 내는 전체 모델.

**구성**
- `tok_emb`: `Embedding(vocab_size, d_model)`
- `pos_emb`: `Embedding(max_seq_len, d_model)` (learned PE)
- `blocks`: `ModuleList([Block(cfg) for _ in range(n_layers)])`
- `norm`: 마지막 `LayerNorm`
- `lm_head`: `Linear(d_model, vocab_size, bias=False)`
- **weight tying**: `self.lm_head.weight = self.tok_emb.weight`
- `_init_weights`: `nn.init.normal_(..., mean=0., std=0.02)`

**forward**
- `idx: (B, T)` → `tok_emb(idx) + pos_emb(arange(T))` → `(B, T, C)`
- causal mask: `torch.tril(torch.ones(T, T))` → `(1, 1, T, T)`로 reshape
- blocks 순회
- final `norm` → `lm_head` → `(B, T, vocab)`
- `targets`이 있으면 `F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=pad_id)`

**검증**
```python
import torch
from config import SonLMConfig
from model import SonLM
cfg = SonLMConfig(vocab_size=64, max_seq_len=16, d_model=32, n_layers=2,
                  n_heads=4, ffn_hidden=64, dropout=0.0)
m = SonLM(cfg)
x = torch.randint(0, 64, (2, 8))
y = torch.randint(0, 64, (2, 8))
logits, loss = m(x, y)
print(logits.shape, float(loss))  # (2, 8, 64), ~4.something
```

→ 막히면: `sonlm/model.py`의 `SonLM`

---

# Phase E. model.py — generate

**무엇:** autoregressive 샘플링 (한 토큰씩 만들어 이어붙이기).

**구현**
```
@torch.no_grad
def generate(self, idx, max_new_tokens, temperature, top_k):
    self.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]   # context 자르기
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature   # 마지막 위치만
        if top_k > 0: 상위 k개 외에는 -inf
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == eos_id: break
    return idx
```

**막히기 쉬운 곳**
- `logits[:, -1, :]` — 마지막 토큰의 분포만 필요한데 전체 시퀀스 logits을 받게 됨
- top-k filtering: `torch.topk` → 임계값보다 작은 logit을 `-inf`로

→ 막히면: `sonlm/model.py`의 `generate`

---

# Phase F. dataset.py — 데이터 + (x, y) shift

**무엇:** JSONL → 토큰 ID → `(x, y)` 페어 → 배치.

**Dataset 클래스**
```python
class SonLMDataset(Dataset):
    def __init__(self, path, tokenizer_path, max_len):
        # JSONL 한 줄씩 읽어 tokenizer.encode → ids 리스트
        # 너무 길면 자르기, 너무 짧으면(<2) 버리기
        ...
    def __getitem__(self, idx):
        ids = self.samples[idx]
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])  # ⭐ 한 칸 shift
```

**collate_fn (padding)**
```python
def collate(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    # (B, max_len) 텐서를 pad_id로 채우고 앞부분만 실제 토큰으로
    ...
```

**핵심 깨달음:** `x = ids[:-1]`, `y = ids[1:]` — 한 칸 shift로 모든 위치에서 "다음 토큰 예측"을 동시에 학습.

→ 막히면: `sonlm/dataset.py`

---

# Phase G. train.py — 학습 루프

**무엇:** DataLoader 돌면서 forward → loss → backward → step.

**스켈레톤**
```python
1. config 로드
2. tokenizer 로드, DataLoader 만들기
3. 모델 + optimizer (AdamW, betas=(0.9, 0.95))
4. (옵션) AMP scaler
5. for step in range(max_steps):
       for x, y in loader:
           x, y = x.to(device), y.to(device)
           lr = warmup_cosine(step, ...)
           # forward
           # backward + grad clip + step
           optimizer.zero_grad()
           if step % log_interval == 0: print loss
           if step % eval_interval == 0: eval + save best
```

**LR schedule (warmup + cosine):**
```
if step < warmup: lr = peak * step / warmup
else: progress = (step - warmup) / (max_steps - warmup)
      lr = min_lr + (peak - min_lr) * 0.5 * (1 + cos(pi * progress))
```

**막히기 쉬운 곳**
- `optimizer.zero_grad(set_to_none=True)` — 안 하면 gradient 누적돼서 학습 망가짐
- AMP 쓰면 `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → clip → `scaler.step` → `scaler.update`

→ 막히면: `sonlm/train.py`

---

# Phase H. inference.py — 추론 (옵션)

체크포인트 로드 → `model.generate` 호출. ChatML 포맷이 신경 쓰이면
`sonlm/inference.py`의 `_format_prompt` 참고. 단순히 토큰 생성만 보고 싶으면
`generate`만 호출해도 충분.

---

# 검증 체크리스트

각 Phase 끝나면:
- [ ] A. `from config import SonLMConfig` 됨
- [ ] B. `Attention(cfg)(x)` 가 올바른 shape 반환
- [ ] C. `FFN(cfg)(x)`, `Block(cfg)(x, mask)` 동작
- [ ] D. `SonLM(cfg)(x, y)` 가 `(logits, loss)` 반환
- [ ] E. `model.generate(...)` 가 새 토큰 반환
- [ ] F. `Dataset[i]` 가 `(x, y)` 쌍 반환, `len(x) == len(y)`, `y[0] == x[1]` (한 칸 shift 확인)
- [ ] G. 학습 루프 한 step 통과 → loss가 숫자로 찍힘

전체 통과하면, 데이터로 실제 학습을 돌려서 baseline (`experiments/00-baseline`)과
같은 PPL이 나오는지 비교해도 좋음.

---

# 막힐 때 어떻게 도움 받기

1. **먼저 30분 직접 시도** — 답답한 만큼 손에 새겨짐.
2. 30분 넘게 막히면 정답지 비교 (둘 중 적합한 것):
   - `solution.py` — 큰 흐름이 헷갈릴 때 (어떻게 연결되는지)
   - `../../sonlm/<이름>.py` — 특정 함수/클래스가 안 풀릴 때
3. 정답지를 봐도 베끼지 말고, **막힌 부분만** 보고 닫은 뒤 다시 직접 친다.
4. 그래도 안 풀리면 Claude에게 묻기 — 단, "정답 보여줘"가 아니라
   "이 부분에서 shape가 왜 이렇게 안 맞아?" 식의 구체 질문이 효과적.

---

# 다 끝낸 뒤

이 폴더의 코드를 sonlm/과 비교해보면, 같은 결과를 내는 데도 미묘하게 다른
스타일/구조가 있을 거임. 그 차이를 노트로 남기면 다음 단계의 양분이 됨.

그 다음엔 `experiments/02b-swiglu/` 같은 ablation으로 — 본인이 만든 baseline
위에 한 가지 변경을 얹어보는 단계로 넘어가면 자연스러움.
