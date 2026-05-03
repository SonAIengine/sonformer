# 00-baseline — Vanilla Decoder-Only Transformer

> 모든 후속 ablation의 비교 기준점. **이 폴더는 박제됨** — 첫 측정 후 수정 금지.

---

## 1. 가설

baseline 자체엔 가설 없음. 이 폴더의 역할은 **모든 ablation이 같은 출발선에서 출발한 fixed reference**가 되는 것.

---

## 2. 모델 구조

### 2-1. 핵심 구성요소

| 구성요소 | 선택 | 기준 |
|---|---|---|
| 아키텍처 | Decoder-only | GPT 계열 표준 |
| Norm | LayerNorm, **Pre-LN** | 학습 안정성 (Post-LN보다 우수) |
| Position | **Learned embedding** | GPT-2 스타일, 가장 단순 |
| Attention | **MHA** (vanilla, fused QKV) | 모던 변형(GQA/MLA) 없이 깨끗하게 |
| FFN | Linear → **ReLU** → Linear | 가장 단순한 activation (의도적으로 GELU 안 씀) |
| FFN 비율 | 4× d_model | 표준 |
| LM head | bias 없음, **weight-tied** with tok_emb | 파라미터 절약 + 표준 |
| Init | N(0, 0.02) | GPT-2 스타일 |
| Dropout | embedding/attn-weights/FFN 출력 | 원조 transformer 스타일 |

### 2-2. 상위 [`sonlm/model.py`](../../sonlm/model.py) 대비 변경점

`sonlm/`의 학습용 코드는 한국어 주석으로 가득한 "교과서" 버전. 실험 폴더의 `model.py`는 같은 아키텍처를 **clean re-implementation** 한 것이며, 다음 변경:

| 변경 | 이유 |
|---|---|
| `register_buffer("causal_mask", ...)`로 마스크 캐시 | 매 forward 재생성 비효율 제거 |
| `dataclass Config`로 인자 일원화 | 하이퍼파라미터를 YAML→dataclass→model로 깨끗히 흘리기 |
| 한국어 주석 → 영문 docstring 위주 | diff/grep 친화. 학습 노트는 README로 분리 |
| 클래스명 `SonLM` → `Sonformer` | 실험 폴더의 ablation 변형들이 공통으로 쓸 이름 |

**기능적/수학적으론 동일** — `sonlm/model.py`와 같은 결과를 내는 vanilla decoder-only transformer.

### 2-3. 실제 사용 하이퍼파라미터 (이번 run 기준)

```yaml
data:        shakespeare, vocab=1024, max_len=256
model:       d_model=256, n_layers=4, n_heads=4, ffn_hidden=1024 (4×), dropout=0.1
train:       batch=32, lr=3e-4 (warmup 100, cosine to 3e-5), max_steps=1500
optim:       AdamW (β=(0.9, 0.95), wd=0.1, grad_clip=1.0)
amp:         CUDA fp16 + GradScaler
seed:        42
device:      H100 80GB
```

→ **3,487,232 params (3.49M)**

---

## 3. 데이터셋: Shakespeare (Tier 0 — sanity check)

### 3-1. 왜 Shakespeare인가

이 첫 run의 목적은 **"파이프라인이 끝까지 돌아가는가"**의 검증. 따라서:

- ✅ **빠른 다운로드** (1MB, urllib만 사용 — `datasets` 라이브러리 의존성 없음)
- ✅ **빠른 학습** (전체 1500 step이 H100에서 ~17초)
- ✅ **널리 알려진 baseline** (nanoGPT 기본 데이터, 결과 비교 가능)
- ❌ 너무 작아 모던 기법(SwiGLU, RoPE 외삽 등)의 진짜 효과는 잘 안 보임 — 그건 다음 run에서 TinyStories로

### 3-2. 데이터 처리

| 단계 | 설명 |
|---|---|
| 다운로드 | `https://raw.githubusercontent.com/karpathy/char-rnn/.../tinyshakespeare/input.txt` |
| 캐시 위치 | `data/shakespeare/input.txt` |
| Train/Val 분할 | 텍스트의 앞 90% / 뒤 10% (문자 단위) |
| 청크 분리 | 빈 줄 두 개(`\n\n`) 기준으로 자른 뒤 빈 chunk 제거 |
| 학습 chunk 수 | 6,283 |
| 평가 chunk 수 | 200 (config의 `max_eval_samples`로 제한) |

### 3-3. 토크나이저

- **BPE (byte-level, GPT-2 스타일)**, vocab=1024
- `data/shakespeare/tokenizer-1024.json`에 캐시 (학습 데이터 + vocab 같으면 재학습 안 함)
- Special tokens: `<|pad|>`, `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`
- ⚠️ 첫 학습 시 decoder를 `ByteLevel()`로 안 잡아 둔 버그가 있어서 후속 fix 적용함 (생성 샘플의 `Ġ`/`Ċ`가 사람 가독 텍스트로 디코딩되도록). 캐시된 JSON은 패치되어 그대로 재사용 가능.

### 3-4. (x, y) 페어

[`dataset.py`의 한 칸 shift](../../sonlm/dataset.py)와 동일:
```
ids = [A, B, C, D, E]
x   = [A, B, C, D]      ← 입력
y   = [B, C, D, E]      ← 한 칸 shift 정답
```

---

## 4. 학습 결과

### 4-1. 핵심 지표

| Metric | Value |
|---|---|
| 학습 step | 1,500 |
| 총 wallclock | **17.6 s** (H100) |
| Peak VRAM | **679.6 MB** |
| Final eval loss | **3.6236** |
| Final eval PPL | **37.47** |
| Best eval loss | **3.6177** (step 1400) |
| Best eval PPL | **37.25** |

### 4-2. Loss curve 요약

step | lr | train_loss | eval_loss | eval_ppl
:--:|:--:|:--:|:--:|:--:
0 | 0.000000 | 7.03 | — | —
200 | 0.000297 | 5.54 | 4.62 | 101.5
400 | 0.000271 | 4.36 | 4.09 | 59.7
600 | 0.000224 | 4.00 | 3.86 | 47.5
800 | 0.000165 | 3.82 | 3.76 | 42.9
1000 | 0.000106 | 3.71 | 3.67 | 39.4
1200 | 0.000059 | 3.64 | 3.64 | 38.3
1400 | 0.000033 | 3.59 | **3.62** | **37.3**

수렴이 매끈함 (warmup 끝나고 cosine 따라 자연스럽게 내려감, train/eval 갭 작음 → overfitting 아직 아님).

전체 곡선: [`results/train_log.csv`](results/train_log.csv)

### 4-3. 생성 샘플

```
prompt:    'ROMEO:'
generated: "
            O! I sawlt thou ask not!
            Marry, that'st thou body sinair, best of yours!
            Doever, bo! thou, withst thou not thou?
            ..."

prompt:    'To be, or not to be,'
generated: " heir; forken, wherefore the villain!
            What is yours? Cours shall die?
            That we had so be gone!'Tis the first fieve? You are they say
            ..."

prompt:    'What is '
generated: "Keepeper:
            How, O, Sorrus?' the justed Kinger of his cristi',
            And yet I, and you,
            Of that cause to-t the seatured. Well you,
            ..."
```

전체: [`results/samples.txt`](results/samples.txt)

**해석**:
- 셰익스피어식 운율/어휘는 흉내내고 있음
- 단어 단위에선 새 토큰 조합(`sinair`, `Doever`, `Sorrus`)이 나옴 → 1500 step + 3.5M params로는 어휘 일관성까진 학습 못 함
- 줄바꿈/대화 형식 등 표면적 구조는 잡혀 있음

이는 baseline으로서 **충분** — 후속 ablation들은 이 PPL=37.47에서 출발해 얼마나 더 떨어뜨리는지 측정.

---

## 5. 산출물 (`results/`)

| 파일 | 내용 |
|---|---|
| `train_log.csv` | step별 lr, train_loss, eval_loss, eval_ppl, wallclock, VRAM |
| `eval.json` | final + best eval, n_params, total_steps, wallclock |
| `samples.txt` | 생성 샘플 |
| `best.pt` | best eval loss 시점 체크포인트 |
| `config.json` | 사용된 config 전체 |
| `model_meta.json` | 파라미터 수 |

---

## 6. 다음 단계

- [02a-rope](../02a-rope/) — Learned PE → RoPE
- (예정) 02b-swiglu, 02c-rmsnorm, 02d-gqa, ...

각 ablation은 이 폴더를 복사해서 `model.py`만 수정. **나머지(데이터/스케줄/seed)는 모두 동일 유지** → 비교 공정성.

```bash
cp -r 00-baseline 02b-swiglu
# 02b-swiglu/model.py만 수정 (FFN: ReLU → SwiGLU)
# config.yaml의 experiment.name만 수정
cd 02b-swiglu && bash run.sh
```

---

## 7. 한계 / 주의

- **Shakespeare는 sanity 용도** (Tier 0). 본격 ablation은 TinyStories(Tier 1)로 가야 더 깨끗한 PPL 차이가 보임.
- **1500 step은 짧음** — overfitting 안 함 동시에 수렴도 다 안 됨. ablation 비교는 동일 step이면 충분히 공정하지만, 절대값을 신뢰하려면 5000+ step 권장.
- **fp16 AMP** 사용. fp32와 비교는 별도 항목.
- **단일 seed (42)**. 진지한 ablation 보고는 3 seeds 평균 권장.
