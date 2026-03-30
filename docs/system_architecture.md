# M3 Full Model — 시스템 구조도

```mermaid
flowchart TD
    subgraph COLLECT["📥 데이터 수집 (Data Collection)"]
        A1["yfinance API\n- OHLCV (6개 종목)\n- VIX 공포지수\n- NASDAQ 지수"]
        A2["FRED API\n- 기준금리 FEDFUNDS\n- 실업률 UNRATE\n- CPI, GDP, M2 등\n  (총 12개 거시지표)"]
    end

    subgraph PREPROCESS["⚙️ 데이터 전처리 (Preprocessing)"]
        B1["기술적 지표 계산\n- RSI_14 (pandas-ta)\n- ATR_14\n- SMA_20\n- Returns\n- Realized_Vol_20d (σ)"]
        B2["학습 타깃 생성\nTarget_Return_5d\n= (Close_t+5 - Close_t) / Close_t"]
        B3["CSV 저장\ntft_processed_panel_v1.csv\n(6종목 × 약 2,500일)"]
    end

    subgraph TRAIN["🧠 모델 학습 (Training)"]
        C1["데이터 분할\nTrain 80% / Val 20%\nGroupNormalizer (z-score)"]
        C2["Optuna 하이퍼파라미터 탐색\n- hidden_size: 32/64/128\n- learning_rate: 1e-4 ~ 5e-3\n- alpha_down/up, beta_down/up\n- crossing_weight\n(30 trials)"]
        C3["M3FullModel 학습\nTFT + Adaptive Pinball Loss\n- λt = 1 + α·VIX + β·σ\n- 비대칭 λt (Q0.1 / Q0.9)\n- Quantile Crossing Penalty\n- EarlyStopping (patience=20)\n- GradientClip (val=0.1)"]
        C4["체크포인트 저장\nm3_epoch=08_val_loss=0.0120.ckpt"]
    end

    subgraph VALIDATE["✅ 성능 검증 (Validation)"]
        D1["Rolling Window Backtest\n5-Fold\n(n=28,300 per ticker)"]
        D2["Violation Rate\nQ0.1: 5~15% 목표\nQ0.5: 45~55% 목표\nQ0.9: 85~95% 목표"]
        D3["Kupiec POF Test\np-value ≥ 0.05\n(참고 지표)"]
        D4["최종 결과\n18개 케이스 중 17개 통과\n(94.4%)"]
    end

    subgraph SERVE["📊 서빙 및 시각화 (Serving)"]
        E1["Streamlit Dashboard\nvisualization/app.py"]
        E2["Quantile Band 그래프\n- 실제 종가 (전체 기간)\n- Q0.1 / Q0.5 / Q0.9 예측 밴드\n- 미래 2주 예측 (초록)"]
        E3["λt 동향 그래프\n- VIX / σ 원시값\n- λt Q0.1 / Q0.9 추이"]
    end

    subgraph DEPLOY["🐳 배포 (Deployment)"]
        F1["Docker Container\n- Python 3.10\n- PyTorch\n- pytorch_forecasting\n- Streamlit"]
        F2["docker-compose.yml\nport: 8501\nvolume: model/saved/\nvolume: data/raw/"]
    end

    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> D1
    D1 --> D2
    D1 --> D3
    D2 --> D4
    D3 --> D4
    C4 --> E1
    B3 --> E1
    E1 --> E2
    E1 --> E3
    E1 --> F1
    F1 --> F2
```

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| **언어** | Python 3.10 |
| **딥러닝** | PyTorch, pytorch_forecasting |
| **데이터 수집** | yfinance, fredapi |
| **데이터 처리** | pandas, numpy, pandas-ta |
| **하이퍼파라미터 최적화** | Optuna |
| **학습 프레임워크** | PyTorch Lightning |
| **시각화** | Streamlit, Plotly |
| **통계 검증** | scipy, statsmodels |
| **배포** | Docker, Docker Compose |

---

## 일별 운영 흐름 (실서비스 기준)

```mermaid
flowchart LR
    G1["매일 오전 6시 (KST)\n장 마감 후"]
    --> G2["yfinance / FRED API\n당일 데이터 수집"]
    --> G3["RSI, ATR, σ 등\n기술적 지표 재계산"]
    --> G4["CSV 업데이트\n새 행 추가"]
    --> G5["모델 추론\n최근 60일 → 2주 예측"]
    --> G6["Streamlit\n예측 결과 업데이트"]
```

> 재학습은 매월 or 분기 1회 수행 (성능 저하 감지 시)

---

## Docker 배포 구성 (예시)

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./model/saved:/app/model/saved
      - ./data/raw:/app/data/raw
    command: streamlit run visualization/app.py --server.port 8501
```

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "visualization/app.py"]
```
