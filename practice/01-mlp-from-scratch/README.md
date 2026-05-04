# mlp-from-scratch — Multi-Layer Perceptron 손코딩

> autograd 위에 *최소한의* 신경망. nn.Module / 활성화 함수 / 손실 / optimizer가 어떻게 맞물려 도는지 한 번에 보기.

---

## 무엇을 배우나

1. **`nn.Module`** 의 의미 — `__init__`에서 layer 등록, `forward`에서 흐름 정의
2. **`nn.Linear`** = `Wx + b` 한 줄
3. **활성화 함수**(ReLU, sigmoid, tanh)가 왜 필요한가 — 없으면 아무리 깊어도 한 개의 linear와 같음
4. **`nn.CrossEntropyLoss`** vs `BCEWithLogitsLoss` 차이
5. **Optimizer**(SGD, Adam) — `optimizer.step()` 한 줄이 손으로 짜던 `w -= lr * w.grad` 를 대체
6. **train/eval 모드** — `model.train()` / `model.eval()` (dropout, batchnorm에 영향)
7. **decision boundary 시각화** — 모델이 무엇을 학습했는지 그림으로

---

## 학습 task: Two Moons 분류

2D 평면에 초승달 모양 두 클래스가 얽힌 데이터. linear classifier로는 절대 못 풀고, ReLU MLP로는 쉽게 풀림 → "비선형성의 위력" 직관 체험.

---

## 모델

```
Input (2)
  ↓ Linear(2, 32) + ReLU
  ↓ Linear(32, 32) + ReLU
  ↓ Linear(32, 2)         ← 2-class logits
Output (2)
```

작아서 CPU로도 1초 안에 학습.

---

## 단계

1. `make_moons` 합성 데이터 생성 (sklearn 없이 numpy/torch로)
2. `class MLP(nn.Module)` 작성 (init + forward)
3. `nn.CrossEntropyLoss`, `torch.optim.Adam`
4. 학습 루프
5. (선택) decision boundary를 ASCII로 출력 또는 확률 grid 출력

---

## 학습 포인트

- ReLU 빼면 학습이 안 됨 → 직접 빼보고 확인
- hidden size를 1로 줄이면 학습이 안 됨 → underfit 직접 확인
- 학습 step 수를 늘려보면 train loss는 계속 떨어지지만 test acc는 어느 순간 평탄 → overfitting의 첫 만남

---

## 그 다음

- [02-cnn-from-scratch](../02-cnn-from-scratch/) — Linear 대신 Conv1d, 시퀀스 입력
- [03-rnn-from-scratch](../03-rnn-from-scratch/) — 시간 차원 unroll
