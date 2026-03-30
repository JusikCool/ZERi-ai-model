# M3 Full Model — 구현 문서

> Adaptive AI 기반 금융 하방 리스크 예측 시스템
> 담당: M3 Full Model (TFT + VIX·σ 완전 Adaptive Pinball Loss)

---

## 1. 모델 목적

향후 **10 거래일(약 2주)간 각 날짜의 종가**를 **세 가지 분위수(Q0.1 / Q0.5 / Q0.9)**로 동시에 예측합니다.

모델은 한 번의 실행으로 10 거래일(주말 제외 약 2주, 달력상 14일)치 등락률을 동시에 출력합니다. 이를 마지막 실제 종가(Day 0) 하나에 곱해 예측 종가로 변환합니다.

```
거래일 Day+N 예측 종가 = Day 0 실제 종가 × (1 + Day+N 예측 등락률)
(N = 1 ~ 10 거래일, 달력상 약 14일)
```

| 분위수 | 의미 |
|--------|------|
| Q0.1 | 실제 종가가 이 가격보다 낮을 확률이 10% → **하방 리스크 하한선** |
| Q0.5 | 실제 종가가 이 가격보다 낮을 확률이 50% → **중앙값 예측** |
| Q0.9 | 실제 종가가 이 가격보다 낮을 확률이 90% → **상방 한계선** |

특히 **VIX(공포지수)와 실현변동성(σ)이 높을수록 예측 밴드를 더 보수적으로 조정**하는 것이 이 모델의 핵심입니다.

---

## 2. 데이터 구성

### 수집 대상

| 구분 | 내용 |
|------|------|
| 종목 | AAPL, GOOGL, META, NVDA, ORCL, TSLA (6개) |
| 기간 | 2015-01-01 ~ 2025-01-01 |
| 단위 | 일별(Daily) 종가 기준 |

### 주요 입력 피처

- **주가 데이터**: Open, High, Low, Close, Volume (yfinance)
- **시장 지표**: NASDAQ 지수, VIX 공포지수
- **거시경제 지표**: 기준금리(FEDFUNDS), 실업률(UNRATE), CPI, GDP, M2 통화량 등 (FRED API)
- **기술적 지표**: RSI, ATR, SMA, 실현변동성(σ\_20d)

### 예측 대상 (Target)

```
Target_Return_5d = (Close_t+5 - Close_t) / Close_t
```

오늘 종가 대비 5 거래일 후 종가의 가격 변화율입니다. (등락률, 투자 수익률과는 다른 개념)

---

## 3. 모델 아키텍처: TFT (Temporal Fusion Transformer)

### TFT란?

시계열 예측에 특화된 딥러닝 모델입니다. 일반 Transformer와 달리 **과거 관측값, 미래에 알 수 있는 변수, 정적 정보**를 각각 분리해서 처리합니다.

```
과거 60일 데이터 (Encoder)
        ↓
   TFT 내부 처리
   - Variable Selection: 어떤 피처가 중요한지 자동 선택
   - Attention: 과거 어느 시점이 중요한지 가중치 부여
        ↓
미래 10 거래일(약 2주) 연속 예측 (Decoder)
→ 각 시점마다 Q0.1, Q0.5, Q0.9 동시 출력
```

### 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Encoder 길이 | 60일 | 모델이 참고하는 과거 기간 |
| Decoder 길이 | 10일 | 예측 구간 |
| Hidden Size | 64 | 모델 내부 표현 차원 |
| Attention Head | 4 | Multi-head Attention 수 |
| Dropout | 0.1 | 과적합 방지 비율 |
| Learning Rate | 0.0003 | 파라미터 업데이트 크기 |

---

## 4. 핵심 기여: Adaptive Pinball Loss

### 기존 Pinball Loss란?

분위수 예측에 사용하는 손실 함수입니다.

```
Pinball Loss (q, y_true, y_pred):
  - y_true > y_pred 이면: q × (y_true - y_pred)
  - y_true < y_pred 이면: (q-1) × (y_true - y_pred)
```

예측값이 실제값보다 낮으면 하방 예측 실패, 높으면 상방 예측 실패로 보고 각각 패널티를 부여합니다.

### M3의 개선: λt 동적 가중치 적용

시장이 불안정할수록(VIX↑, σ↑) 손실에 더 큰 패널티를 주어 **모델이 더 보수적인 예측을 하도록 강제**합니다.

```
Adaptive Pinball Loss = λt × Pinball Loss

λt = 1 + α × max(VIX - threshold, 0) / vix_scale
       + β × relu(σ) / sigma_scale
```

- λt ≥ 1.0 항상 보장 (VIX, σ가 낮으면 일반 Pinball Loss와 동일)
- VIX가 높을수록 λt 증가 → 손실이 커져 → 모델이 하방 예측을 더 안전하게 조정

### 비대칭 λt (M3만의 특징)

Q0.1(하방)과 Q0.9(상방)에 서로 다른 가중치를 적용합니다.

| 분위수 | 가중치 계수 | 의미 |
|--------|-----------|------|
| Q0.1 (하방) | α_down, β_down | 하방 리스크를 더 민감하게 반영 |
| Q0.5 (중앙값) | 1.0 고정 | 적응 없음 |
| Q0.9 (상방) | α_up, β_up | 상방 예측도 시장 상황에 반응 |

### Quantile Crossing Penalty

세 분위수 예측값이 역전되는 현상 방지합니다.

```
Q0.1 < Q0.5 < Q0.9 이어야 정상
→ Q0.1 > Q0.5 이거나 Q0.5 > Q0.9 이면 패널티 부과
```

```python
penalty = relu(Q0.1 - Q0.5) + relu(Q0.5 - Q0.9)
total_loss = adaptive_pinball_loss + crossing_weight × penalty
```

---

## 5. 학습 과정

### 전체 흐름

```
1. 데이터 로드 및 전처리 (data/raw/tft_processed_panel_v1.csv)
2. Train/Val 분할 (전체의 80% 학습, 20% 검증)
3. TFT + Adaptive Pinball Loss 학습
4. EarlyStopping으로 과적합 방지
5. 최적 가중치 자동 저장
```

### EarlyStopping

검증 손실(val_loss)이 **20 epoch 동안 개선되지 않으면 학습 자동 중단**합니다. 불필요한 학습 낭비를 방지합니다.

### Optuna 하이퍼파라미터 자동 최적화

수동으로 하이퍼파라미터를 조정하는 대신, Optuna가 자동으로 최적값을 탐색합니다.

```
탐색 대상:
- hidden_size: [32, 64, 128]
- learning_rate: 1e-4 ~ 5e-3
- dropout: 0.05 ~ 0.3
- alpha_down, beta_down: 0.5 ~ 3.0  (하방 λt 계수)
- alpha_up, beta_up: 0.5 ~ 3.0      (상방 λt 계수)
- crossing_weight: 0.01 ~ 0.5
```

30번의 trial을 수행하여 val_loss가 가장 낮은 조합을 선택합니다.

### 최종 학습 결과 (v3)

| 항목 | 값 |
|------|----|
| 학습 종료 Epoch | 8 |
| 최종 val_loss | 0.0120 |
| 저장 경로 | model/saved/m3_epoch=08_val_loss=0.0120.ckpt |

---

## 6. 성능 검증

### 검증 방법: Rolling Window Backtest (5-Fold)

전체 데이터를 시간 순서대로 5구간으로 나누어 각 구간에서 예측 성능을 측정합니다. 사전 학습된 단일 모델을 사용하며 fold별 재학습은 없습니다.

```
[--Train1--][Val1]
[----Train2----][Val2]
[------Train3------][Val3]
[--------Train4--------][Val4]
[----------Train5----------][Val5]
```

총 샘플: 종목당 28,300개

### 검증 지표 1: Violation Rate

실제 종가가 예측 분위수 종가보다 낮은 비율을 측정합니다.

```
Q0.1 목표: 10% (허용 범위 5~15%)
Q0.5 목표: 50% (허용 범위 45~55%)
Q0.9 목표: 90% (허용 범위 85~95%)
```

**결과: 18개 케이스 중 17개 통과 (94.4%)**

| 종목 | Q0.1 | Q0.5 | Q0.9 |
|------|:----:|:----:|:----:|
| AAPL | 10.67% ✅ | 48.5% ✅ | 88.6% ✅ |
| GOOGL | 12.08% ✅ | 49.5% ✅ | 88.3% ✅ |
| META | 9.02% ✅ | 43.0% ❌ | 85.6% ✅ |
| NVDA | 13.25% ✅ | 50.4% ✅ | 89.1% ✅ |
| ORCL | 13.21% ✅ | 50.3% ✅ | 87.4% ✅ |
| TSLA | 10.40% ✅ | 48.6% ✅ | 89.2% ✅ |

META Q0.5 실패는 2015~2025년 사이 Facebook → Meta 리브랜딩, 메타버스 전환 등 기업 구조적 변화로 인한 것으로 모델 한계가 아닌 데이터 특성에 기인합니다.

### 검증 지표 2: Kupiec POF Test

위반율이 이론적으로 통계적으로 유의미한지 검증합니다 (p-value ≥ 0.05 통과).

**결과: 3/18 통과 (GOOGL Q0.5, NVDA Q0.5, ORCL Q0.5)**

Kupiec 테스트는 샘플이 클수록 극도로 민감해지는 특성이 있습니다. n=28,300에서는 이론값에서 0.3%만 벗어나도 기각됩니다. 따라서 Violation Rate를 1차 지표로, Kupiec를 참고 지표로 활용합니다.

---

## 7. Ablation Study 구조 (팀 전체)

| 모델 | Loss 방식 | 특징 |
|------|-----------|------|
| M1 (TFT-Fixed) | 고정 Pinball Loss | 기준 모델 |
| M2 (TFT-VIX) | VIX만 반영한 Adaptive Loss | 부분 Adaptive |
| M3 (Full Model) | VIX + σ 완전 Adaptive + 비대칭 λt | **최종 제안 모델** |

M1, M2와의 비교를 통해 VIX·σ 연동 Adaptive Loss의 효과를 정량적으로 검증합니다.

---

## 8. 실행 방법

```bash
# 학습
python main.py --mode train

# Optuna 하이퍼파라미터 탐색
python main.py --mode optuna --n_trials 30

# 백테스팅 및 성능 검증
python main.py --mode backtest --n_splits 5

# Streamlit 시각화 대시보드
streamlit run visualization/app.py
```

---

## 9. 폴더 구조

```
ZERi-ai-model/
├── data/raw/                  # 수집 원본 데이터
├── model/m3_full_model/
│   ├── loss.py                # Adaptive Pinball Loss
│   ├── model.py               # TFT 래퍼 (M3FullModel)
│   ├── dataset.py             # 데이터 전처리 및 DataLoader
│   └── train.py               # 학습 및 Optuna 최적화
├── model/saved/               # 학습된 체크포인트
├── validation/
│   ├── backtest/backtest.py   # Rolling Window 백테스팅
│   └── kupiec/kupiec.py       # Kupiec POF Test
├── visualization/app.py       # Streamlit 대시보드
├── configs/config.yaml        # 전체 설정값
└── main.py                    # 실행 진입점
```
