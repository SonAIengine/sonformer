# sonformer

> Transformer를 코드로 직접 따라가며 공부하는 개인 학습 저장소.

[arman-bd/guppylm](https://github.com/arman-bd/guppylm)을 베이스로, 대학원 과정에서 transformer 아키텍처와 LLM 학습 파이프라인 전체를 한 줄씩 이해하고 본인 손으로 발전시키는 것을 목표로 합니다.

모든 핵심 파일에는 **shape 추적 + 동작 원리 + 설계 의도**를 담은 한국어 주석이 달려있습니다.

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

## 발전 계획

대학원 과정과 함께 진행할 학습/실험 항목:

- [ ] Attention shape 변환을 손으로 직접 따라가며 정리 (블로그/노트)
- [ ] Learned PE → RoPE로 교체해보기
- [ ] ReLU FFN → SwiGLU로 교체
- [ ] KV cache로 추론 속도 개선
- [ ] FlashAttention(또는 `scaled_dot_product_attention`) 도입
- [ ] 한국어 합성 데이터로 재학습
- [ ] Multi-head → Grouped-Query Attention 비교
- [ ] Loss curve, attention map 시각화

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
