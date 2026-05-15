"""
predict_kronos.py — Kronos zero-shot 추론
50 NASDAQ 종목, T+1 ~ T+30 의 5일 누적 수익률 분위수 (Q0.05 ~ Q0.95, 19개)

출력 (TFT 의 predict_with_xai.py 와 동일 schema):
  model/saved/kronos/
    predict_kronos_returns.csv             # long form
    xai/raw/{TICKER}_predictions.csv       # wide form (30 × 19)

ensemble.py 에서 그대로 merge 가능:
  tft = pd.read_csv("model/saved/predict_xai_returns.csv")
  kronos = pd.read_csv("model/saved/kronos/predict_kronos_returns.csv")
  merged = tft.merge(kronos, on=["group_id", "future_day", "Date"],
                     suffixes=("_tft", "_kronos"))

데이터 흐름:
  1) yfinance → 종목별 OHLCV (full history)
  2) 마지막 5거래일 trim → TFT 의 T 와 정렬 (TFT 는 Target_Return_5d 의 NaN 으로 자동 trim)
  3) 마지막 60거래일을 Kronos 입력 window 로 사용
  4) Kronos 로 35 거래일 OHLCV 의 10 stochastic sample path 생성
     (pred_len = 30 horizon + 5 buffer for 5-day forward return)
  5) 각 path 의 close 에서 5-day forward return 계산: r[t] = close[t+5] / close[t] - 1
  6) 10 path 분포의 경험적 분위수 (Q0.05 ~ Q0.95, 19개) 산출

준비:
  git clone https://github.com/shiyu-coder/Kronos
  pip install -r Kronos/requirements.txt
  Kronos 폴더가 본 스크립트와 같은 위치에 있다고 가정 (KRONOS_REPO 로 조정)

실행:
  python predict_kronos.py

Kronos-small + 50종목 × 10 sample 기준 예상 시간:
  GPU (RTX 3060+): ~10-15분
  CPU: ~25-30분
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

# Kronos repo 경로
KRONOS_REPO = Path("./Kronos")
sys.path.insert(0, str(KRONOS_REPO))
from model import Kronos, KronosTokenizer, KronosPredictor

# =============================================================
# 설정
# =============================================================
TICKERS = [
    # 기존 6개
    "AAPL", "GOOGL", "TSLA", "META", "NVDA", "ORCL",
    # 메가캡 테크 (10)
    "MSFT", "AMZN", "AVGO", "NFLX", "CSCO", "ADBE", "INTC", "AMD", "QCOM", "TXN",
    # 소프트웨어/SaaS (5)
    "INTU", "ADSK", "CTSH", "CDNS", "SNPS",
    # 반도체 (9)
    "AMAT", "LRCX", "KLAC", "MCHP", "MRVL", "MU", "ASML", "NXPI", "ON",
    # 인터넷/게임 (4)
    "EBAY", "BKNG", "EA", "TTWO",
    # 헬스케어/바이오 (7)
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ISRG", "IDXX",
    # 소비재 (5)
    "SBUX", "COST", "MDLZ", "PEP", "MAR",
    # 금융/결제 (1)
    "PAYX",
    # 유틸리티/통신 (3)
    "CMCSA", "CHTR", "TMUS",
]

WINDOW_SIZE = 60                       # 인코더 window (TFT 와 동일)
HORIZON = 30                           # 예측 시점 수 (TFT 와 동일)
N_BUFFER_5D = 5                        # 5일 forward return 계산용 추가 예측
PRED_LEN = HORIZON + N_BUFFER_5D       # = 35
SAMPLE_COUNT = 10                      # stochastic path 개수 (속도/분위수 안정성 trade-off)
QUANTILES = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05, 0.10, ..., 0.95

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kronos sampling 파라미터
TEMPERATURE = 1.0
TOP_P = 0.9

# 마지막 N 거래일 trim → TFT 의 dropna(Target_Return_5d) 와 정렬
TRIM_LAST_N = 5

# 입력 fetch buffer (gaps/holidays 고려)
FETCH_PERIOD = "1y"

OUTPUT_DIR = Path("model/saved/kronos")
RAW_DIR = OUTPUT_DIR / "xai" / "raw"


# =============================================================
# 데이터 수집
# =============================================================
def fetch_ohlcv(ticker: str) -> pd.DataFrame | None:
    """
    yfinance 에서 OHLCV fetch → 정규화 → 마지막 5거래일 trim → 마지막 WINDOW_SIZE 거래일.

    TFT 와 T 를 동일하게 맞추기 위해 마지막 5거래일을 manual trim
    (TFT 는 Target_Return_5d = Close.shift(-5) 의 NaN 으로 자동 trim).
    """
    try:
        df = yf.Ticker(ticker).history(period=FETCH_PERIOD)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
        df.index.name = None
        df = df.rename(columns=str.lower)
        df = df[["open", "high", "low", "close", "volume"]]
        df["amount"] = df["close"] * df["volume"]  # turnover (Kronos optional)

        # Drop NaN rows (yfinance 가 가끔 빈 행을 넣음)
        df = df.dropna(subset=["open", "high", "low", "close"])

        if TRIM_LAST_N > 0:
            df = df.iloc[:-TRIM_LAST_N]
        if len(df) < WINDOW_SIZE:
            return None
        return df.tail(WINDOW_SIZE)
    except Exception as e:
        print(f"  ❌ fetch 실패: {e}")
        return None


# =============================================================
# Kronos sample paths
# =============================================================
def get_sample_paths(
    predictor: KronosPredictor,
    df: pd.DataFrame,
    pred_len: int,
    sample_count: int,
) -> list[pd.DataFrame]:
    """
    Kronos 로 sample_count 개의 독립 stochastic path 추출.

    KronosPredictor.predict(sample_count=N) 가 평균 DataFrame 만 반환하는 경우가 있어
    안전하게 sample_count=1 로 N 번 호출하여 raw 한 path 를 직접 수집.

    Returns:
        list of pd.DataFrame (length=sample_count), each (pred_len, 6) [O,H,L,C,V,A]
    """
    x_timestamp = pd.Series(df.index)
    last_date = df.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=pred_len
    )
    y_timestamp = pd.Series(future_dates)

    df_input = df.reset_index(drop=True)

    paths = []
    for _ in range(sample_count):
        pred = predictor.predict(
            df=df_input,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=TEMPERATURE,
            top_p=TOP_P,
            sample_count=1,
        )
        paths.append(pred)
    return paths


def paths_to_quantile_returns(
    paths: list[pd.DataFrame],
    quantiles: list[float],
    horizon: int,
    n_buffer: int,
) -> np.ndarray:
    """
    각 path 의 close 에서 5-day forward return 계산 후, 경험적 분위수 산출.

    Math:
      closes[i, j]   = path i 의 close at future day j+1   (j=0..pred_len-1)
      return_5d[i, j] = closes[i, j+5] / closes[i, j] - 1, for j=0..horizon-1
      → "T+(j+1) 시점에서 본 5일 누적 수익률"

    Returns:
        (horizon, n_quantiles) ndarray
    """
    closes = np.stack([p["close"].to_numpy() for p in paths], axis=0)
    n_samples, pred_len = closes.shape
    assert pred_len >= horizon + n_buffer, (
        f"pred_len={pred_len} < horizon+buffer={horizon+n_buffer}"
    )

    base = closes[:, :horizon]                          # close at T+1..T+horizon
    future = closes[:, n_buffer : n_buffer + horizon]   # close at T+1+5..T+horizon+5
    returns = future / base - 1.0                       # (n_samples, horizon)

    q = np.quantile(returns, quantiles, axis=0)         # (n_quantiles, horizon)
    return q.T                                          # (horizon, n_quantiles)


# =============================================================
# Main
# =============================================================
def load_kronos_predictor() -> KronosPredictor:
    """모델/토크나이저 로드 + 디바이스 셋업."""
    print(f"[device] {DEVICE}")
    print(f"[Kronos] model={KRONOS_MODEL_NAME}, tokenizer={KRONOS_TOKENIZER_NAME}")

    tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_NAME)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()

    # KronosPredictor 의 device 파라미터 지원 여부에 따라 분기
    try:
        predictor = KronosPredictor(
            model, tokenizer, max_context=MAX_CONTEXT, device=DEVICE
        )
    except TypeError:
        predictor = KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)
    return predictor


def main() -> None:
    predictor = load_kronos_predictor()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict] = []
    successes: list[str] = []
    failures: list[str] = []

    t_start = time.time()
    for i, ticker in enumerate(TICKERS, 1):
        t0 = time.time()
        print(f"[{i:2d}/{len(TICKERS)}] {ticker:6s} ... ", end="", flush=True)

        df = fetch_ohlcv(ticker)
        if df is None:
            print("❌ data insufficient")
            failures.append(ticker)
            continue

        try:
            paths = get_sample_paths(predictor, df, PRED_LEN, SAMPLE_COUNT)
            q_returns = paths_to_quantile_returns(
                paths, QUANTILES, HORIZON, N_BUFFER_5D
            )
        except Exception as e:
            print(f"❌ predict 실패: {e}")
            failures.append(ticker)
            continue

        # Future business days (T+1 .. T+HORIZON)
        last_date = df.index[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=HORIZON
        )

        # (1) wide form: xai/raw/{ticker}_predictions.csv
        df_wide = pd.DataFrame(
            q_returns,
            index=[d.date() for d in future_dates],
            columns=[f"Q{q}" for q in QUANTILES],
        )
        df_wide.index.name = "Date"
        df_wide.to_csv(RAW_DIR / f"{ticker}_predictions.csv")

        # (2) long form rows → predict_kronos_returns.csv
        for t in range(HORIZON):
            row = {
                "group_id": ticker,
                "future_day": t + 1,
                "Date": future_dates[t].date(),
                "window_end_estimate": (
                    future_dates[t] + pd.tseries.offsets.BDay(N_BUFFER_5D)
                ).date(),
            }
            for qi, q in enumerate(QUANTILES):
                row[f"Q{q}_return_5d"] = float(q_returns[t, qi])
            long_rows.append(row)

        elapsed = time.time() - t0
        median_avg = float(np.mean(q_returns[:, QUANTILES.index(0.5)]))
        print(
            f"✅ T={last_date.date()}, Q0.5 평균={median_avg*100:+.2f}%, "
            f"{elapsed:.1f}s"
        )
        successes.append(ticker)

    # Save long form CSV
    if long_rows:
        df_long = pd.DataFrame(long_rows)
        long_path = OUTPUT_DIR / "predict_kronos_returns.csv"
        df_long.to_csv(long_path, index=False)
        total_elapsed = time.time() - t_start
        print(
            f"\n=== 완료 ({len(successes)}/{len(TICKERS)} 종목, "
            f"총 {total_elapsed/60:.1f}분) ==="
        )
        print(f"  - {long_path}")
        print(f"  - {RAW_DIR}/<TICKER>_predictions.csv ({len(successes)} files)")
        if failures:
            print(f"  ⚠️ 실패 종목 ({len(failures)}): {failures}")
        print(f"\nensemble.py 에서 merge 키:")
        print(f"  ['group_id', 'future_day', 'Date']")
    else:
        print("\n❌ 결과 없음 — 모든 종목 실패")


if __name__ == "__main__":
    main()