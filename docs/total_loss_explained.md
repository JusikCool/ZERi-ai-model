# 최종 손실 함수 수식 설명

---

## 1. 수식 원문

$$\mathcal{L} = \frac{1}{|Q|}\sum_{q \in Q}\sum_t \lambda_t \cdot L_q(y_t, \hat{y}_t^{(q)}) + \gamma \cdot \mathcal{L}_{cross}$$

- $Q$ : 분위수 집합 {0.1, 0.5, 0.9}, $|Q| = 3$
- $q$ : 개별 분위수
- $t$ : 시점 (거래일)
- $\lambda_t$ : 시점 t의 동적 가중치 (VIX·σ 연동)
- $L_q$ : 분위수 q에 대한 핀볼 로스
- $\gamma$ : Crossing Penalty 반영 비율 (Optuna 탐색)
- $\mathcal{L}_{cross}$ : 분위수 역전 패널티

---

## 2. 수식을 두 부분으로 나누면

$$\mathcal{L} = \underbrace{\frac{1}{|Q|}\sum_{q \in Q}\sum_t \lambda_t \cdot L_q(y_t, \hat{y}_t^{(q)})}_{\text{Adaptive Pinball Loss 평균}} + \underbrace{\gamma \cdot \mathcal{L}_{cross}}_{\text{역전 방지 패널티}}$$

---

## 3. 첫 번째 항 — Adaptive Pinball Loss 평균

$$\frac{1}{|Q|}\sum_{q \in Q}\sum_t \lambda_t \cdot L_q$$

한 번의 학습에서 아래 순서로 계산된다.

**Step 1.** Q0.1, Q0.5, Q0.9 각각에 대해 모든 시점 t의 $\lambda_t \cdot L_q$ 를 합산

**Step 2.** 분위수 개수(3)로 나눠서 평균

즉 세 분위수가 **동등한 비중**으로 학습에 기여한다.

### 코드에서의 실제 계산:

```python
for i, q in enumerate([0.1, 0.5, 0.9]):
    lambda_t = compute_lambda(vix, sigma, q)   # 동적 가중치
    pb = pinball(y_pred[..., i], y_true, q)    # 핀볼 로스
    total_loss += (lambda_t * pb).mean()       # 시점 평균 후 합산

total_loss /= 3  # 분위수 평균
```

---

## 4. 두 번째 항 — Crossing Penalty (분위수 역전 방지)

$$\gamma \cdot \mathcal{L}_{cross}$$

### 분위수 역전이란?

정상적인 예측:

$$\hat{y}^{(0.1)} < \hat{y}^{(0.5)} < \hat{y}^{(0.9)}$$

역전된 예측 (비정상):

$$\hat{y}^{(0.1)} > \hat{y}^{(0.5)}$$

Q0.1이 Q0.5보다 높으면 "하방 경계가 중앙값보다 위에 있다"는 모순이 발생한다.

### Crossing Penalty 계산:

$$\mathcal{L}_{cross} = \text{ReLU}(\hat{y}^{(0.1)} - \hat{y}^{(0.5)}) + \text{ReLU}(\hat{y}^{(0.5)} - \hat{y}^{(0.9)})$$

- 역전이 없으면 → ReLU 결과가 0 → 패널티 없음
- 역전이 발생하면 → 역전된 만큼 패널티 부과

$\gamma$는 Optuna로 탐색하며, 패널티 강도를 조절한다.

---

## 5. 전체 흐름 요약

```
매 학습 스텝마다:

1. Q0.1, Q0.5, Q0.9 각각
   → 시점별 λt 계산 (VIX, σ 기반)
   → λt × 핀볼 로스 계산
   → 시점 평균

2. 세 분위수 평균 → Adaptive Pinball Loss

3. 분위수 역전 여부 확인 → Crossing Penalty

4. 최종 손실 = Adaptive Pinball Loss + γ × Crossing Penalty
```

- 최종 손실값이 클수록 모델 파라미터를 더 크게 업데이트해서 다음 예측을 개선하도록 강제하는 것이고, 이게 패널티이다.
- 업데이트 되는 TFT 내부 파라미터는 다음과 같다.
    - Variable Selection Network의 가중치 — 어떤 피처가 중요한지 결정
    - Multi-head Attention의 가중치 — 어떤 시점에 집중할지 결정
    - GRN(Gated Residual Network)의 가중치 — 비선형 패턴 학습
    - LSTM의 hidden state 가중치 — 시계열 순서 의존성 학습

- 이것들은 파라미터가 아님 (Optuna에 의해 결정된 고정값임):
    - α, β, γ — Optuna가 학습 전에 결정, 학습 중 고정
    - VIX 임계값(25), $s_v$ — 사전 설정값
    - 분위수(0.1, 0.5, 0.9) — 고정

즉 손실이 클수록 TFT 내부 수만~수십만 개의 가중치가 역전파(Backpropagation)로 업데이트된다.

---

## 6. γ (Crossing 계수)는 왜 필요한가?

Crossing Penalty가 너무 크면 모델이 역전 방지에만 집중해서 예측 정확도가 떨어진다.
너무 작으면 역전이 자주 발생한다.
Optuna가 [0.01, 0.5] 범위에서 적절한 γ값을 찾아 균형을 맞춘다.
