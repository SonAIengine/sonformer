# lstm-from-scratch — LSTM 손코딩

> Vanilla RNN의 **vanishing gradient**를 완화하기 위해 1997년 등장한 4-gate 구조.

---

## 무엇을 배우나

- **gate**라는 개념 — 정보를 얼마나 통과시킬지 sigmoid로 동적 결정
- **cell state c_t** — hidden과 분리된 "장기 기억" 채널
- 4 gate (input / forget / output / candidate)의 역할 분담
- **왜 forget gate가 vanishing gradient를 완화하는가** (gradient가 곱셈 chain이 아닌 덧셈 path를 갖게 됨)

---

## 핵심 식

```
i_t = σ(W_i · [x_t; h_{t-1}])    # input gate: 새 정보 얼마나 쓸까
f_t = σ(W_f · [x_t; h_{t-1}])    # forget gate: 과거 cell 얼마나 유지할까
o_t = σ(W_o · [x_t; h_{t-1}])    # output gate: cell의 얼마를 hidden으로 내보낼까
g_t = tanh(W_g · [x_t; h_{t-1}]) # candidate: 새로 추가할 후보 정보

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t  # ★ cell state — 곱셈 + 덧셈 mix
h_t = o_t ⊙ tanh(c_t)            # hidden state
```

**핵심**: c_t는 c_{t-1}에 forget gate만 곱하고 더하는 구조라, gradient가 길게 흘러도 살아남기 쉬움.

---

## RNN과의 비교 (학습 task 동일)

| | Vanilla RNN | LSTM |
|---|---|---|
| 파라미터 | W_xh, W_hh (2개) | W_i, W_f, W_o, W_g (4배) |
| 시간 메모리 | h_t 하나 | h_t + c_t (두 채널) |
| 짧은 의존성 | 잘 풀림 | 잘 풀림 |
| 긴 의존성 | 깨짐 | 비교적 잘 풀림 |
| 속도 | 빠름 | 약 4배 연산 |

이 폴더에서는 **RNN과 같은 task + 더 긴 시퀀스**로 비교해서 LSTM의 효과를 직접 측정.

---

## 학습 task

작은 어휘 V=8에서 **delayed copy task**:
- 입력: `[a, X, X, X, ..., X]` (첫 토큰 후 noise N개)
- 정답: 마지막에 `a` 를 다시 출력

길이 N을 늘려가면서 RNN이 깨지는 지점에서 LSTM은 살아남는지 확인.

---

## 단계

1. `LSTMCell` 4-gate 구현 — 각 gate를 따로 선형변환 (또는 fused 4×H로 한 번에)
2. cell state c와 hidden state h 두 개 모두 흘리기
3. `LSTMLM` (또는 분류기) — `for t in range(T)` unroll
4. 학습 루프
5. 시퀀스 길이 늘려가며 RNN과 비교

---

## 그 다음

- **Transformer** ([../05-sonlm-from-scratch/](../05-sonlm-from-scratch/)) — sequential dependency 자체를 attention으로 우회 → 병렬화 + long-range 둘 다 잡음
- LSTM은 여전히 sequential. 학습 시 GPU 활용도 낮음. 이게 transformer가 이긴 결정적 이유.
