# cnn-from-scratch — 1D CNN 손코딩 (TextCNN)

> 시퀀스에 1D convolution을 적용해 **local pattern**(n-gram)을 감지하는 분류기.
> 이미지 CNN의 텍스트판. 2014 Kim Yoon "Convolutional Neural Networks for Sentence Classification".

---

## 무엇을 배우나

- **1D Conv = "n-gram 패턴 detector"** 라는 직관
- kernel_size에 따라 어떤 길이의 패턴이 잡히는지
- **multi-kernel 병렬** — 다양한 n-gram을 동시에 보기
- **max-over-time pooling** — "시퀀스 어디든 한 번이라도 잡혔으면 OK"
- CNN의 한계: receptive field 안의 local 패턴은 잘 잡지만, **long-range dependency는 약함**

---

## 핵심 식

```
시퀀스 (B, T, d_emb)을 (B, d_emb, T)로 transpose
→ Conv1d(in=d_emb, out=n_filters, kernel_size=k)
   → 각 위치에서 길이 k 짜리 슬라이딩 윈도우에 같은 가중치 적용
→ ReLU
→ MaxPool1d over time (T 차원 전체에 max)
   → "이 패턴이 어디든 한 번이라도 강하게 활성화됐는가"
→ 여러 kernel_size 결과를 concat → MLP → 분류
```

---

## RNN/LSTM과의 비교

| | RNN/LSTM | CNN | Transformer |
|---|---|---|---|
| 처리 방식 | sequential | 병렬 (전 위치 동시) | 병렬 (전 위치 동시) |
| receptive field | 전 시퀀스 (이론상) | kernel_size × layer 깊이 | 전 시퀀스 (1 layer로) |
| local 패턴 | 잡지만 느림 | ★ 본업 | OK |
| long-range | 약함 | 약함 (얕으면) | ★ 본업 |
| GPU 활용 | 낮음 | 높음 | 높음 |

CNN은 **transformer가 결국 흡수한 "병렬화" 장점**을 RNN보다 먼저 시퀀스에 가져온 모델.

---

## 학습 task

작은 어휘 V=8에서 **bigram 패턴 감지 분류**:
- positive (label=1): 시퀀스 어딘가에 `[1, 2]` 가 인접해 있음
- negative (label=0): 그렇지 않음

kernel_size=2 짜리 filter 하나가 거의 완벽히 푸는 게 정상. CNN이 왜 이 task에 강한지 직접 확인.

---

## 단계

1. token → embedding (`nn.Embedding`)
2. transpose: `(B, T, E) → (B, E, T)` (Conv1d는 channel-first)
3. 여러 kernel_size로 `Conv1d` 병렬 — 예: [2, 3, 4]
4. 각 출력에 ReLU
5. `max(dim=-1)` 으로 시간축 pooling
6. concat → MLP → 2-way 분류
7. `BCEWithLogitsLoss` 또는 `CrossEntropy`

---

## 그 다음

- **RNN/LSTM** ([../03-rnn-from-scratch/](../03-rnn-from-scratch/), [../04-lstm-from-scratch/](../04-lstm-from-scratch/)) — 같은 시퀀스 분류 task를 sequential 방식으로
- **Transformer** — attention이 어떻게 CNN의 "병렬성"과 RNN의 "전역성"을 동시에 가져갔는지
