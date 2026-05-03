# 00-baseline

> Vanilla decoder-only transformer. **Frozen reference** — 모든 후속 ablation의 비교 기준점.

## 가설

특별한 가설 없음. 이 폴더는 **다른 모든 실험이 측정될 baseline**입니다.

## 아키텍처

| 구성요소 | 값 |
|---|---|
| Norm | LayerNorm (Pre-LN) |
| Position | Learned embedding |
| Attention | MHA (vanilla, fused QKV) |
| FFN | 2-layer ReLU |
| LM head | Weight-tied, no bias |
| Init | N(0, 0.02) |

`config.yaml` 기본값:
- d_model = 384, n_layers = 6, n_heads = 6
- ffn_hidden = 1536 (4× d_model)
- max_seq_len = 512, vocab = 4096
- batch = 32, lr = 3e-4 (AdamW + warmup + cosine), max_steps = 5000

## 실행

```bash
bash run.sh
```

기본은 TinyStories에서 학습. 다른 데이터셋으로 바꾸려면 `config.yaml`의 `data.dataset`을 `shakespeare`(빠름) 또는 `wikitext-103`으로.

```bash
# wandb 로깅
SONFORMER_WANDB=1 bash run.sh

# 의존성 이미 설치돼 있으면 pip install 스킵
SONFORMER_SKIP_DEPS=1 bash run.sh
```

## 결과 (run 후 채워짐)

| Metric | Value |
|---|---|
| Final eval loss | _TBD_ |
| Final eval PPL | _TBD_ |
| Best eval loss | _TBD_ |
| Wallclock | _TBD_ |
| Peak VRAM | _TBD_ |
| Generation 품질 | `results/samples.txt` 참조 |

학습 곡선: `results/train_log.csv` (step, lr, train_loss, eval_loss, eval_ppl, wallclock_s, vram_mb)

## Ablation 만드는 법 (이 폴더 복사)

```bash
cp -r 00-baseline 02a-rope
# 02a-rope/model.py 만 수정. config.yaml은 변경 사항 명시 (예: extrapolation.enabled=true).
# 그 외 (data, train, run.sh, requirements.txt)는 그대로 유지 — 비교 공정성.
```

원칙: **한 실험에 한 변경.** model.py에서 한 컴포넌트만 바꾸고, config 외 다른 건 건들지 않기.
