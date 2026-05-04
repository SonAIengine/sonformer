# tensor-autograd — PyTorch 텐서 + autograd 손으로 익히기

> **모든 딥러닝의 진짜 바닥**. 이걸 한 번 손으로 풀어보지 않으면 위에 쌓는 모든 모델이 블랙박스가 됨.

---

## 무엇을 배우나

1. **Tensor 생성 / shape / dtype / device** — `torch.tensor`, `.shape`, `.float()`, `.to("cuda")`
2. **Broadcasting** — shape이 달라도 연산이 되는 규칙
3. **Autograd** — `requires_grad=True` 켜면 모든 연산이 그래프로 기록됨
4. **Backward** — `loss.backward()` 가 어떻게 leaf tensor의 `.grad`를 채우는가
5. **수치 검증** — autograd가 계산한 gradient를 *손으로 푼 수식*과 비교

---

## 핵심 시연: 선형 회귀

데이터: $y = 3x + 1 + \text{noise}$

모델: $\hat y = w x + b$  (학습할 파라미터 2개)

손실: $\mathcal L = \frac{1}{N}\sum (\hat y - y)^2$

손으로 푼 gradient:
```
∂L/∂w = (2/N) Σ (ŵx + b - y) · x
∂L/∂b = (2/N) Σ (ŵx + b - y)
```

이 두 식이 `loss.backward()` 후 `w.grad`, `b.grad`와 **소수점 6자리까지 일치**하는지 확인 → autograd가 마법이 아니라 chain rule을 정확히 따라간다는 확신.

---

## 단계

1. 합성 데이터 생성 (`y = 3x + 1 + ε`)
2. `w, b = torch.tensor(0.0, requires_grad=True), ...` 로 파라미터
3. forward → loss → `loss.backward()`
4. 손 gradient와 `.grad` 비교 (assert)
5. `w.data -= lr * w.grad`, `w.grad.zero_()` 로 SGD 한 step
6. 1000 step 돌려 (w, b)가 (3, 1)에 수렴하는지

---

## 학습 포인트 (이걸 못 답하면 다음 단계 가지 말 것)

- `requires_grad=True`인 leaf tensor에만 `.grad`가 채워진다 (중간 텐서는 default로 안 채워짐)
- `.backward()`는 **누적**한다 — `optimizer.zero_grad()`를 안 부르면 grad가 더해짐
- `.detach()` / `with torch.no_grad():` 로 그래프에서 분리 가능 (eval/inference 시)
- `loss.backward()`를 두 번 부르면 default로 에러 → 학습 그래프는 한 번 쓰고 버려짐

---

## 그 다음

- [01-mlp-from-scratch](../01-mlp-from-scratch/) — 같은 autograd로 multi-layer 모델 학습
