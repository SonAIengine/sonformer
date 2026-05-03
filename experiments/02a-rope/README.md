# 02a-rope — Rotary Position Embedding (RoPE)

> Learned positional embedding을 **RoPE**로 교체.

## 가설

| Metric | 예상 |
|---|---|
| In-distribution PPL (T ≤ 학습 길이) | baseline과 동등 ~ 약간 좋음 |
| **Extrapolation PPL** (T > 학습 길이) | **baseline 대비 압도적 우위** |
| Wallclock | ≈ baseline (오버헤드 미미) |
| 파라미터 수 | **약간 감소** (`pos_emb` 제거 → `max_seq_len × d_model` 만큼) |

이유: learned PE는 학습 길이를 벗어나면 random vector를 보는 것과 같음.
RoPE는 절대 위치를 명시적 임베딩이 아닌 query/key의 회전으로 표현하므로,
학습 길이를 벗어나도 회전 각도만 부드럽게 외삽됨.

## 변경점 (vs 00-baseline)

| 파일 | 변경 |
|---|---|
| `model.py` | `pos_emb` 제거; `precompute_rope`/`rotate_half`/`apply_rope` 추가; `Attention`이 q,k에 RoPE 적용; RoPE buffer를 `rope_cache_len`까지 미리 계산 |
| `config.yaml` | `model.rope_base`, `model.rope_cache_len` 추가; `extrapolation.enabled: true` |
| `train.py` | `Config` 인스턴스 생성 시 RoPE 인자 전달; 학습 종료 후 `extrapolation_perplexity` 호출 |

`train.py`의 학습 루프 자체는 baseline과 **byte-for-byte 동일** — 비교 공정성 유지.

## RoPE 핵심 코드

```python
def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)             # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)

def rotate_half(x):
    d = x.shape[-1]
    return torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1)
```

q, k에만 적용 (v는 그대로). v를 회전하면 attention output에 위치 정보가
원치 않게 섞여 들어가서 학습 망가짐.

## 실행

```bash
bash run.sh
```

baseline과 동일한 데이터/스케줄/seed로 학습 → `results/`에 결과 저장.
Extrapolation eval은 학습 종료 후 자동 수행 (config의 `extrapolation.enabled: true`).

## 결과 (run 후 채워짐)

### In-distribution
| Metric | 00-baseline | 02a-rope | Δ |
|---|---|---|---|
| Final eval loss | _ | _ | _ |
| Final eval PPL | _ | _ | _ |
| Wallclock | _ | _ | _ |
| Params | _ | _ | _ |

### Extrapolation (긴 시퀀스에서 PPL — 학습 시 max_seq_len = 512)
| Seq len | 00-baseline | 02a-rope |
|---|---|---|
| 256 | _ | _ |
| 512 | _ | _ |
| 1024 | **error / inf** (learned PE 한계) | _ |
| 2048 | **error / inf** | _ |

baseline은 1024 이상에서 `pos_emb`이 학습되지 않은 위치 임베딩을 참조해 의미 없는 출력을 냄 (사실상 측정 불가). RoPE는 이 영역에서도 동작하는지가 핵심 평가.

## 한 줄 정리

> RoPE는 in-distribution에서는 비슷하거나 약간 더 좋지만, **외삽에서 격차가 폭발적**. 그게 이걸 쓰는 이유.
