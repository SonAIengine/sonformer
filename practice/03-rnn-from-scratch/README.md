# rnn-from-scratch — Vanilla RNN 손코딩

> Transformer의 **모태**가 되는 시퀀스 모델. 시간순으로 한 hidden state에 정보를 누적.

---

## 무엇을 배우나

- 시퀀스를 **시간 차원으로 unroll** 하는 감각
- hidden state라는 "기억 변수"가 어떻게 흘러가는지
- BPTT (Backpropagation Through Time) — 시간을 거슬러 gradient 전파
- **vanishing gradient의 실체** — 왜 LSTM이 발명됐고, 왜 결국 transformer로 넘어갔는지

---

## 핵심 식 (한 줄)

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)     # hidden 업데이트
y_t = W_hy · h_t + b_y                            # 출력 (각 step)
```

**모든 시간 step이 같은 W_xh, W_hh를 공유** (parameter sharing). 시퀀스가 길어도 파라미터 수는 일정.

---

## 학습 task

작은 어휘 V=8에서 **반복 패턴 char-LM**:
- 입력: `[2, 3, 4, 5, 6, 7, 0, 1, ...]` (mod V로 +1씩)
- 정답: 다음 토큰 `(cur + 1) % V`

이 task는 1-step 의존성이라 vanilla RNN이 잘 풀어야 함. 깊어지면 못 풀게 되는 게 다음 단계 (LSTM에서 길이를 늘려 비교).

---

## 단계

1. `RNNCell` 직접 구현 — `tanh(Linear(x) + Linear(h))` 한 줄
2. `RNNLM` — `for t in range(T)` 로 manual unroll
3. one-hot 인코딩으로 토큰 → 벡터
4. 마지막 hidden을 head로 vocab 분류
5. 학습 루프 (Adam + clip)
6. `generate()` — prefix를 한 토큰씩 흘려 hidden 만든 뒤 sampling

PyTorch에는 `nn.RNN` 모듈이 있지만 **여기선 의도적으로 cell을 직접 풀어** unrolling 감각 익히기.

---

## 직접 손코딩 →

`practice.py` 만들어서 빈 상태부터 시작. 막히면 `solution.py` 비교 (단, 베끼지 말 것).

---

## 그 다음

- **LSTM** ([../04-lstm-from-scratch/](../04-lstm-from-scratch/)) — vanishing gradient를 4-gate로 완화
- **Transformer** ([../05-sonlm-from-scratch/](../05-sonlm-from-scratch/)) — sequential dependency 자체를 attention으로 대체
