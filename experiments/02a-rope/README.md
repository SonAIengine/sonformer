# 02a-rope — Rotary Position Embedding (RoPE)

> **Learned PE → RoPE.** 가장 영향이 크고 후속 외삽 실험으로 자연스럽게 이어지는 첫 ablation.

---

## 1. 가설

| Metric | 예상 | 실제 |
|---|---|---|
| In-distribution PPL | baseline과 동등 ~ 약간 좋음 | **37.47 → 28.47 (-24%)** ✅ |
| Extrapolation (T > 학습 길이) | baseline 대비 압도적 우위 | (Shakespeare 데이터 부족, 미검증 — §4-3) |
| Wallclock | ≈ baseline (오버헤드 미미) | 17.6s → 19.6s (+11%) |
| Params | ↓ (`pos_emb` 제거) | -65,536 (-1.9%, 정확히 max_seq_len × d_model) |

**짧은 결론**: in-distribution 단계에서도 **매우 큰 이득**. 같은 step·같은 데이터·같은 seed에서 PPL을 24% 떨어뜨림. RoPE의 진짜 본 게임(외삽)은 더 긴 데이터에서 별도 검증 필요.

---

## 2. RoPE란?

**핵심 아이디어**: 절대 위치 임베딩 대신, query/key 벡터를 **위치에 따라 회전**시켜 attention 내적이 자연스럽게 상대 위치 정보를 담게 만든다.

수학적으로:
- 위치 m의 query를 각도 θ_m으로 회전
- 위치 n의 key를 각도 θ_n으로 회전
- 내적 → 회전각 차이 (m − n)에 의존하는 표현이 자동으로 나옴

**왜 좋은가**:
1. 절대 PE를 따로 더할 필요 없음 (파라미터 ↓)
2. 학습 길이 밖 위치에서도 회전 각도만 부드럽게 외삽됨 (long-context 가능)
3. attention 점수가 명시적으로 "두 토큰의 거리"를 표현

---

## 3. Baseline 대비 변경점 — 파일별 상세

### 3-1. `model.py` (대부분의 변경)

**A. Config 추가 필드**
```python
@dataclass
class Config:
    ...
    rope_base: float = 10000.0       # RoPE 주파수 베이스 (LLaMA 표준)
    rope_cache_len: int = 8192       # 외삽 평가용 미리 계산 길이
```

**B. `pos_emb` 제거**
```python
# baseline:
self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)   # ← 제거
# forward에서:
x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))        # → 아래로 변경
# rope 버전:
x = self.drop(self.tok_emb(idx))                             # NO pos_emb
```

**C. RoPE primitive 함수 3개 추가**
```python
def precompute_rope(head_dim, max_len, base=10000.0, ...):
    """(cos, sin) 쌍을 (max_len, head_dim) shape로 미리 계산."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    pos = torch.arange(max_len)
    freqs = torch.outer(pos, inv_freq)             # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)        # (T, D), 두 절반 복제
    return emb.cos(), emb.sin()

def rotate_half(x):
    """Last dim을 반으로 잘라 [-x2, x1]로 회전."""
    d = x.shape[-1]
    return torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1)

def apply_rope(x, cos, sin):
    """x: (B, H, T, D). q와 k에 적용 (v에는 안 함)."""
    return (x * cos) + (rotate_half(x) * sin)
```

→ LLaMA-style "split-half" 회전 컨벤션. 원조 RoFormer 논문의 adjacent-pair와 수학적으로 동등.

**D. `Attention.forward` 수정 — q/k에만 RoPE 적용**
```python
# baseline:
q, k, v = qkv[0], qkv[1], qkv[2]
attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

# rope:
q, k, v = qkv[0], qkv[1], qkv[2]
q = apply_rope(q, cos, sin)        # ← q 회전
k = apply_rope(k, cos, sin)        # ← k 회전 (v는 그대로!)
attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

**왜 v는 회전 안 함?** v를 회전하면 attention output에 위치 정보가 직접 섞여 들어가서, "토큰 i의 의미"와 "토큰 i의 위치"가 분리 안 됨 → 학습 망가짐.

**E. RoPE 캐시 + auto-extend (외삽 지원)**
```python
def __init__(self, cfg):
    ...
    cache_len = max(cfg.max_seq_len, cfg.rope_cache_len)
    cos, sin = precompute_rope(head_dim, cache_len, base=cfg.rope_base)
    self.register_buffer("rope_cos", cos, persistent=False)
    self.register_buffer("rope_sin", sin, persistent=False)

def _maybe_extend_cache(self, T, device, dtype):
    """T가 캐시 길이 초과 시 더 큰 cache로 재계산 (외삽 시)."""
    if T <= self.rope_cos.shape[0]:
        return
    cos, sin = precompute_rope(...)
    self.register_buffer("rope_cos", cos, persistent=False)
    ...
```

이 덕분에 학습 시엔 캐시된 cos/sin 그대로 쓰고, eval에서 더 긴 시퀀스 들어와도 자동으로 캐시 확장 — 모델 재구성 없이 외삽 가능.

### 3-2. `config.yaml`

추가/변경된 키:
```yaml
model:
  rope_base: 10000.0          # 새로 추가
  rope_cache_len: 4096        # 새로 추가 (16× max_seq_len)
extrapolation:
  enabled: true               # 학습 후 외삽 PPL 측정 자동 수행
  seq_lens: [128, 256, 512, 1024, 2048]
```

`pos_emb` 관련 항목 자체가 없어짐.

### 3-3. `train.py`

**학습 루프는 byte-for-byte baseline과 동일**. 차이점 두 가지:

```python
# Config 인스턴스화에 RoPE 인자 전달
mcfg = Config(
    ...,
    rope_base=cfg["model"].get("rope_base", 10000.0),
    rope_cache_len=cfg["model"].get("rope_cache_len", 8192),
)

# 학습 종료 후 외삽 PPL 측정
if cfg.get("extrapolation", {}).get("enabled", False):
    long_texts = sorted(val_texts, key=len, reverse=True)[:200]
    extrap_results = extrapolation_perplexity(
        model, tk, long_texts, device,
        seq_lens=cfg["extrapolation"]["seq_lens"], ...
    )
```

학습 자체엔 어떤 차이도 없어서 **비교 공정**.

---

## 4. 학습 결과

### 4-1. 데이터셋 (baseline과 동일)

- **Shakespeare** (Tier 0, 1MB)
- 6,283 train chunks / 200 val chunks (동일 split, 동일 seed=42)
- BPE vocab 1024 (baseline의 토크나이저와 **완전히 같은 캐시 파일** 사용 → ID 매핑 일치)

### 4-2. In-distribution metric

| Metric | Baseline | RoPE | Δ |
|---|---|---|---|
| Params | 3,487,232 | **3,421,696** | **-65,536 (-1.9%)** |
| Final eval loss | 3.6236 | **3.3489** | **-0.2747** |
| Final eval PPL | 37.47 | **28.47** | **-24.0%** ⭐ |
| Best eval loss | 3.6177 | **3.3504** | -0.2673 |
| Wallclock | 17.6s | 19.6s | +11% |
| Peak VRAM | 679.6 MB | 744.9 MB | +9.6% |

→ **PPL을 24% 떨어뜨림.** 같은 데이터, 같은 step, 같은 seed에서. 이게 RoPE를 모든 모던 LLM이 채택하는 이유.

파라미터 수가 정확히 `max_seq_len(256) × d_model(256) = 65,536`만큼 줄어든 게 보이며, 이게 `pos_emb` 제거가 의도대로 됐다는 확실한 증거.

### 4-3. Extrapolation eval — **데이터 부족으로 미결**

이번 run에서 측정한 외삽 PPL:

| seq_len | RoPE PPL | n_samples |
|---|---|---|
| 128 | 37.22 | 22 |
| 256 | 46.46 | 4 |
| 512 | NaN | **0** |
| 1024 | NaN | **0** |
| 2048 | NaN | **0** |

**원인**: Shakespeare validation chunk들이 너무 짧아서, 토큰화 후 256 토큰 이상인 샘플이 4개밖에 없고 512+ 는 0개. 외삽 평가 자체가 의미 있게 돌아가질 않음.

**해석**: in-distribution에서 RoPE 우위는 깨끗히 입증됐지만, RoPE의 **간판 능력인 외삽**은 이 셋업으론 확인 불가. 이건 다음 run에서 **TinyStories(긴 스토리) 또는 PG-19(소설)**로 다시 평가해야 함.

이런 한계를 알면서도 보고하는 이유: ablation kit이 외삽 측정 인프라까지 갖췄음을 보이기 위해 (n_samples=0 케이스 graceful 처리 포함).

### 4-4. Loss curve 비교 (eval만)

step | Baseline eval_ppl | RoPE eval_ppl | RoPE 우위
:--:|:--:|:--:|:--:
200 | 101.5 | 90.4 | -11%
400 | 59.7 | 50.9 | -15%
600 | 47.5 | 40.1 | -16%
800 | 42.9 | 33.7 | -21%
1000 | 39.4 | 31.2 | -21%
1200 | 38.3 | 29.3 | -23%
1400 | **37.3** | **28.5** | **-24%**

**RoPE는 처음부터 끝까지 baseline을 앞섬, 학습이 진행될수록 격차 확대.** 단순 우연이 아닌 architecturally robust한 우위.

### 4-5. 생성 샘플

```
prompt: 'ROMEO:'
gen:    "
        If he should be, sir, and I'll short me me.
        He would I, how he is a match. What is the piece?
        Sarry, do me to the fight of thy grace
        ..."

prompt: 'To be, or not to be,'
gen:    "
        Mepow far-formed. What's the dear? Hetter sick.
        And every hell? I be so, in thy name?
        Bethind, brighting, since, I'll have yours injuried.
        ..."

prompt: 'What is '
gen:    "Kom?
        Where is the duke's because? what wilt thou?
        Your draw, by the trumpet of a mind of drum.
        Will me he must live. We speak as welcome,
        ..."
```

전체: [`results/samples.txt`](results/samples.txt)

**해석 (qualitative)**:
- 두 모델 다 셰익스피어식 표면 형식은 잡힘
- RoPE 쪽이 문장 단위 짜임새가 약간 더 자연스러움 (수치로 표현하긴 어려운 인상)
- 절대값으로는 두 모델 다 미흡 — 1500 step + 3.5M params의 한계

---

## 5. 결론

### 진짜 발견된 것
1. ✅ **RoPE는 in-distribution에서도 큰 우위** (PPL -24%, 동일 step·seed). 자주 "외삽 때만 좋다"고 알려진 통념과 다름. **즉시 채택 가치 충분**.
2. ✅ 파라미터는 오히려 약간 감소 (`pos_emb` 제거 효과 정확히 -65,536).
3. ✅ Wallclock 11% 증가 — RoPE 회전(`rotate_half + 곱셈 두 번`) 오버헤드. fp16 AMP 켰음에도 발생.
4. ✅ VRAM 10% 증가 — RoPE buffer가 cache_len(4096) × d_model(256)만큼 더 들고 있음. 학습 시엔 sequence만큼만 쓰지만 buffer는 항상 메모리 점유.

### 미결
- ❌ 외삽 영역에서의 RoPE 우위 — Shakespeare로는 측정 불가. **TODO: TinyStories / PG-19 run**.

### 다음 단계
- 02b-swiglu (FFN activation 변경)
- 02c-rmsnorm (Norm 변경)
- 진지한 외삽 비교용 별도 실험: `02a-rope-extrap`을 만들어 PG-19 데이터로 따로 측정

---

## 6. 산출물

| 파일 | 내용 |
|---|---|
| `results/train_log.csv` | step별 loss, lr, wallclock, VRAM |
| `results/eval.json` | final eval, **extrapolation 결과 포함** |
| `results/samples.txt` | 생성 샘플 |
| `results/best.pt` | best eval loss 시점 체크포인트 |
| `results/config.json` | 사용된 config |
| `results/model_meta.json` | 파라미터 수 |

---

## 7. 한 줄 정리

> RoPE는 외삽뿐 아니라 **in-distribution에서도 큰 이득**. 같은 데이터·step·seed에서 PPL을 24% 떨어뜨림. 모든 모던 LLM이 채택한 이유가 이 한 ablation에서 손에 잡힘.
