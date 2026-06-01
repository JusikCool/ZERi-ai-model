"""
학습된 TFT 체크포인트로 매일 production 추론:
  - yfinance + FRED 에서 50 종목의 최근 데이터를 직접 수집
  - fresh data 의 last_date T 기준, 향후 N영업일(T+1 ~ T+N) 5일 누적 수익률 분위수
    (T = today - 5 거래일. Target_Return_5d 가 valid 한 마지막 일자.
     처음 ~5일 예측은 실현 시나리오와 비교 가능, 마지막 ~5일은 진짜 미래)
    N = config["data"]["horizon"]
  - 같은 추론 결과의 TFT XAI(변수 중요도 + attention)
  - 분위수/변수 중요도/attention 패턴을 자동으로 한국어 줄글로 해설

데이터 흐름:
  - 학습 CSV (dataset.py 의 DATA_PATH 가 가리키는 파일): train_ds 템플릿 복원용.
    GroupNormalizer 가 학습 시점 종목별 mean/std 로 fit 되어 있고,
    체크포인트 weights 와 정합되므로 normalizer 상태가 반드시 필요.
  - fetch_fresh_data(): yfinance + FRED 에서 직접 수집.
    build_dataset_50tickers.py 와 동일한 처리 (풀 history → dropna → 마지막 N 거래일 슬라이스).
    실제 추론 입력으로 사용. CSV 와 동일한 컬럼/형식.
  - vol_group 컬럼: vol_group_map.json 으로 종목 → group 매핑 추가 (M4-Combined 필수)

추론 흐름:
  1. df_train = load_data()   # 학습 CSV → train_ds (normalizer 상태)
  2. df = fetch_fresh_data()  # 신선한 입력 데이터 (today - 5 거래일까지)
  3. df["vol_group"] 추가     # vol_group_map 으로 매핑
  4. df_ext = extend_with_future_rows(df, horizon)  # T+1 ~ T+horizon 행 추가
  5. predict_ds = TimeSeriesDataSet.from_dataset(train_ds, df_ext, predict=True)
     → train_ds 의 fitted normalizer 가 fresh data 에 적용됨
  6. tft.predict(mode="raw") 로 분위수 + variable selection + attention 동시 추출

Production 사용:
  매일 시장 마감 후 python predict_with_xai.py 실행.
  CSV 갱신 불필요 (yfinance/FRED 가 자동으로 최신 데이터 제공).

준비:
  - pip install yfinance fredapi ta
  - FRED API key 발급 (무료): https://fred.stlouisfed.org/docs/api/api_key.html
  - 본 파일 상단의 FRED_API_KEY 상수에 발급받은 key 입력
  - data/raw/vol_group_map.json 존재 (build 스크립트에서 생성)

출력:
  model/saved/
    predict_xai_returns.csv             # 50 종목 × horizon 일 × n 분위수
    predict_xai_summary_<TICKER>.png    # 종목별 통합 figure
    predict_xai_summary_<TICKER>.txt    # 종목별 텍스트 해설
    predict_xai_report.txt              # 50 종목 통합 보고서
    xai/
      xai_ALL_*.png, xai_<TICKER>_*.png # 변수 중요도/attention
      xai_summary.csv

실행 위치: 프로젝트 루트
실행:
  python predict_with_xai.py
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import yfinance as yf
import ta
from fredapi import Fred
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# run_full_pipeline.py 와 동일한 attention mask device patch
_orig_get_attention_mask = TemporalFusionTransformer.get_attention_mask


def _patched_get_attention_mask(self, encoder_lengths, **kwargs):
    mask = _orig_get_attention_mask(self, encoder_lengths, **kwargs)
    return mask.to(encoder_lengths.device)


TemporalFusionTransformer.get_attention_mask = _patched_get_attention_mask

from model.m4.dataset import (
    GROUP_ID,
    TIME_IDX,
    build_dataset,
    get_vix_stats,
    load_data,
)
from model.m4.model import M4FullModel
from validation.backtest.backtest_m4 import (
    _build_model_kwargs,
    _load_vol_group_map,
    GROUP_LABELS,
)

CONFIG_PATH = Path("configs/config.yaml")
CHECKPOINT_DIR = Path("model/saved")
OUTPUT_DIR = Path("model/saved")
XAI_DIR = OUTPUT_DIR / "xai"

TARGET = "Target_Return_5d"

# ============================================================
# Fresh data fetching (production daily inference)
# ============================================================
FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"   # ← 본인 key 로 교체 (또는 환경변수에서 로드)

# 학습 시 사용한 50 종목 (반드시 학습 데이터셋과 동일해야 normalizer 가 매핑됨)
TICKERS_FRESH = [
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

# 학습 시 window_size=60 으로 학습됐으므로 인코더 60일 유지.
# config.yaml 의 data.window_size 와 반드시 동일해야 함.
KEEP_LAST_N_TRADING_DAYS = 60

FRED_SERIES_IDS = [
    "FEDFUNDS", "UNRATE", "DTWEXBGS", "CPIAUCSL", "PCEPI",
    "GDP", "M2SL", "GS10", "T10Y2Y", "PAYEMS", "CSUSHPISA", "INDPRO",
]


# ------------------------------------------------------------
# 변수 카테고리 매핑 (XAI 텍스트 해설용)
# ------------------------------------------------------------
VARIABLE_LABELS = {
    # 가격/기술 지표
    "Target_Return_5d": ("자기자신의 5일 수익률", "price_tech"),
    "Open": ("시가", "price_tech"),
    "High": ("고가", "price_tech"),
    "Low": ("저가", "price_tech"),
    "Close": ("종가", "price_tech"),
    "Volume": ("거래량", "price_tech"),
    "Dividends": ("배당", "price_tech"),
    "Stock Splits": ("주식분할", "price_tech"),
    "RSI_14": ("RSI(14) 모멘텀", "price_tech"),
    "ATR_14": ("ATR(14) 변동성", "price_tech"),
    "SMA_20": ("20일 이동평균", "price_tech"),
    "Returns": ("일간 수익률", "price_tech"),
    "Realized_Vol_20d": ("20일 실현 변동성", "price_tech"),
    "GARCH_Variance": ("GARCH 추정 분산", "price_tech"),
    # 시장 리스크
    "NASDAQ_Close": ("나스닥 종가", "market_risk"),
    "VIX_Close": ("VIX (시장 공포지수)", "market_risk"),
    # 거시 경제
    "FEDFUNDS": ("연준 기준금리", "macro"),
    "UNRATE": ("실업률", "macro"),
    "DTWEXBGS": ("무역가중 달러지수", "macro"),
    "CPIAUCSL": ("CPI (소비자물가)", "macro"),
    "PCEPI": ("PCE 물가지수", "macro"),
    "GDP": ("GDP", "macro"),
    "M2SL": ("M2 통화량", "macro"),
    "GS10": ("10년 국채 수익률", "macro"),
    "T10Y2Y": ("장단기 금리 스프레드", "macro"),
    "PAYEMS": ("비농업 고용", "macro"),
    "CSUSHPISA": ("주택가격지수", "macro"),
    "INDPRO": ("산업생산지수", "macro"),
    # 달력
    "Month": ("월", "calendar"),
    "Day_of_Week": ("요일", "calendar"),
    # 정적
    "group_id": ("종목 식별자", "static"),
    "vol_group": ("변동성 그룹", "static"),
}

CATEGORY_LABELS = {
    "price_tech": "가격/기술 지표",
    "market_risk": "시장 리스크",
    "macro": "거시 경제 변수",
    "calendar": "달력 변수",
    "static": "정적 변수",
    "other": "기타",
}


def _categorize(var_name: str) -> tuple[str, str]:
    """변수 이름 → (한글 라벨, 카테고리)"""
    if var_name in VARIABLE_LABELS:
        return VARIABLE_LABELS[var_name]
    if "encoder_length" in var_name or "decoder_length" in var_name:
        return (var_name, "other")
    if "scale" in var_name or "center" in var_name:
        return (var_name, "other")
    return (var_name, "other")


def generate_text_explanation(
    g: str,
    fc_g: pd.DataFrame,
    hist_g: pd.DataFrame,
    interp_g: dict,
    encoder_vars: list[str],
    last_obs_date: pd.Timestamp,
) -> str:
    """
    종목별 prose 해설 생성.
    수익률 분위수 / 변수 중요도 / attention 패턴을 자연어로 설명.
    horizon 길이에 동적으로 적응 (10일이든 30일이든 자동 표기).
    """
    n_horizon = len(fc_g)           # 예측 시점 수 (10, 30 등)
    weeks = max(1, round(n_horizon / 5))  # 5거래일 = 1주

    parts: list[str] = []
    parts.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    parts.append(f"  [{g}] 향후 {n_horizon}영업일 5일 누적 수익률 전망 분석")
    parts.append(f"  (기준일 T = {last_obs_date.date()}, 예측 구간 = T+1 ~ T+{n_horizon})")
    parts.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ---- 1. 수익률 전망 (Q0.05 ~ Q0.5 하방 시나리오 + 중앙) ----
    q_low = fc_g["Q0.05_return_5d"].to_numpy()
    q5 = fc_g["Q0.5_return_5d"].to_numpy()

    median_avg = float(np.mean(q5))
    median_min, median_max = float(np.min(q5)), float(np.max(q5))
    q_low_avg = float(np.mean(q_low))
    q_low_min, q_low_max = float(np.min(q_low)), float(np.max(q_low))
    downside_band_avg = float(np.mean(q5 - q_low))

    if median_avg > 0.005:
        direction = "상승"
    elif median_avg < -0.005:
        direction = "하락"
    else:
        direction = "횡보"

    hist_std = float(hist_g["Target_Return_5d"].std()) if len(hist_g) > 0 else 0.0
    band_ratio_desc = ""
    if hist_std > 0:
        ratio = downside_band_avg / (1.645 * hist_std)
        if ratio > 1.2:
            band_ratio_desc = (
                f" 정규분포 가정 대비 약 {ratio:.1f}배로 넓어 "
                f"하방 리스크에 대한 모델의 우려가 큽니다."
            )
        elif ratio < 0.8:
            band_ratio_desc = (
                f" 정규분포 가정 대비 약 {ratio:.1f}배로 좁아 "
                f"하방 리스크에 대한 모델 확신이 비교적 강합니다."
            )
        else:
            band_ratio_desc = f" 최근 60일 실현 변동성과 유사한 수준입니다."

    parts.append("")
    parts.append("◆ 수익률 전망 (중앙 시나리오 + 하방 분위수)")
    parts.append(
        f"  Q0.5(중앙 시나리오) {n_horizon}일 평균은 {median_avg*100:+.2f}% 로, "
        f"향후 약 {weeks}주 동안 5일 단위 누적 수익률이 대체로 {direction} 흐름을 보일 것으로 모델이 추정합니다. "
        f"개별 시점별 Q0.5 는 {median_min*100:+.2f}% ~ {median_max*100:+.2f}% 범위에 위치합니다."
    )
    parts.append(
        f"  Q0.05(하위 5% 극단 비관 시나리오) {n_horizon}일 평균은 {q_low_avg*100:+.2f}%, "
        f"개별 시점 범위 {q_low_min*100:+.2f}% ~ {q_low_max*100:+.2f}% 입니다."
    )
    parts.append(
        f"  Q0.05~Q0.5 하방 밴드의 평균 폭은 {downside_band_avg*100:.2f}%p 입니다."
        + band_ratio_desc
    )

    # ---- 2. 변수 중요도 (encoder) ----
    parts.append("")
    parts.append("◆ 변수 중요도 (Encoder 기준 상위 5)")
    if "encoder_variables" in interp_g:
        df_imp = _var_importance_df(interp_g["encoder_variables"], encoder_vars)
        df_imp["category"] = df_imp["variable"].apply(lambda x: _categorize(x)[1])

        top5 = df_imp.head(5)
        bullet_lines = []
        for _, row in top5.iterrows():
            label, cat = _categorize(row["variable"])
            cat_kr = CATEGORY_LABELS.get(cat, "기타")
            bullet_lines.append(
                f"    {row['variable']:24s} {row['importance']:.4f}  "
                f"({label}, {cat_kr})"
            )
        parts.extend(bullet_lines)

        total_imp = df_imp["importance"].sum()
        if total_imp > 0:
            cat_share = (
                df_imp.groupby("category")["importance"].sum() / total_imp
            ).sort_values(ascending=False)
            top_cats = cat_share.head(3)
            cat_str = ", ".join(
                f"{CATEGORY_LABELS.get(c, c)} {v*100:.1f}%" for c, v in top_cats.items()
            )
            parts.append(f"  카테고리별 비중 (상위 3): {cat_str}")

            top_cat = top_cats.index[0]
            top_cat_kr = CATEGORY_LABELS.get(top_cat, top_cat)
            parts.append(
                f"  → {top_cat_kr}가 예측에 가장 큰 영향을 주고 있으며, "
                f"이는 {('자기상관/추세 성분' if top_cat == 'price_tech' else '거시 환경 변화' if top_cat == 'macro' else '시장 리스크 신호' if top_cat == 'market_risk' else '주기성/계절성')}이 "
                f"이번 예측의 주된 근거임을 의미합니다."
            )

    # ---- 3. Attention 패턴 ----
    parts.append("")
    parts.append("◆ Attention (모델이 과거 어느 시점을 참조했는가)")
    if "attention" in interp_g:
        att = interp_g["attention"].detach().cpu().numpy()
        if att.ndim == 1:
            flat = att
        elif att.ndim == 2:
            flat = att.mean(axis=0)
        else:
            flat = att.mean(axis=tuple(range(att.ndim - 1)))

        n_enc = len(flat)
        top_idx = np.argsort(flat)[-3:][::-1]
        peak_lines = []
        for rank, idx in enumerate(top_idx, 1):
            rel = idx - n_enc
            offset_back = abs(rel)
            try:
                actual_date = pd.bdate_range(
                    end=last_obs_date, periods=offset_back + 1
                )[0]
                date_str = actual_date.date().isoformat()
            except Exception:
                date_str = "N/A"
            peak_lines.append(
                f"    {rank}위: 약 {offset_back}거래일 전 ({date_str}), "
                f"가중치 {flat[idx]:.4f}"
            )
        parts.extend(peak_lines)

        recent = float(np.mean(flat[-10:])) if n_enc >= 10 else float(np.mean(flat))
        mid = (
            float(np.mean(flat[-30:-10]))
            if n_enc >= 30
            else float(np.mean(flat[: max(1, n_enc - 10)]))
        )
        far = float(np.mean(flat[:-30])) if n_enc > 30 else 0.0

        if recent > mid and recent > far:
            zone = "최근 10일 (T-10 ~ T-1) 에 가장 집중"
        elif mid > recent and mid > far:
            zone = "10~30일 전 중간 과거 구간에 집중"
        elif far > recent and far > mid and far > 0:
            zone = "30일 이상 먼 과거에 집중"
        else:
            zone = "특정 시점에 치우치지 않고 비교적 분산"
        parts.append(f"  시간대 분포: {zone}.")
        parts.append(
            f"  (구간별 평균 — 최근10일: {recent:.4f}, 10~30일전: {mid:.4f}, 30일전이상: {far:.4f})"
        )

    parts.append("")
    return "\n".join(parts)


# ------------------------------------------------------------
# Fresh data fetching (yfinance + FRED → load_data() 호환 형식)
# ------------------------------------------------------------
def fetch_fresh_data() -> pd.DataFrame:
    """
    yfinance + FRED 에서 데이터를 직접 수집해서, dataset.py 의 load_data() 와
    동일한 컬럼 / dtype / 정렬 / time_idx 구조로 DataFrame 을 반환.

    핵심 원칙: build_dataset_50tickers.py 의 검증된 fetch 패턴 100% 동일.
      - FRED: fred.get_series(sid)
      - yf.download: start="1980-01-01"
      - yf.Ticker.history: period="max"

    그 다음:
      - 기술지표 (RSI/ATR/SMA, Realized_Vol_20d) 계산
      - Target_Return_5d = Close.shift(-5) / Close - 1
      - dropna(): warmup 초기 + 마지막 5일 trailing 모두 자동 제거
      - balanced panel: 가장 늦은 시작일에 맞춤
      - 마지막에 group 별로 KEEP_LAST_N_TRADING_DAYS 만 keep
      - time_idx: load_data 와 동일하게 group 별 cumcount
    """
    if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
        raise ValueError(
            "FRED_API_KEY 가 설정되지 않았습니다. "
            "predict_with_xai.py 상단의 FRED_API_KEY 상수에 본인 key 를 입력하세요. "
            "(https://fred.stlouisfed.org/docs/api/api_key.html 에서 무료 발급)"
        )

    print(
        f"\n[fresh fetch] {len(TICKERS_FRESH)} 종목 "
        f"(전체 history fetch → 마지막 {KEEP_LAST_N_TRADING_DAYS} 거래일만 keep)"
    )

    # ---- FRED 거시지표 ----
    print("  [1/3] FRED 거시지표...")
    fred = Fred(api_key=FRED_API_KEY)
    fred_frames = []
    for sid in FRED_SERIES_IDS:
        try:
            s = fred.get_series(sid)
            if not s.empty:
                fred_frames.append(s.to_frame(name=sid))
            else:
                print(f"    ⚠️ {sid}: empty")
        except Exception as e:
            print(f"    ⚠️ {sid}: {e}")
    df_fred = pd.concat(fred_frames, axis=1, sort=True)
    df_fred.index.name = "Date"
    df_fred.index = pd.to_datetime(df_fred.index)

    # ---- VIX, NASDAQ ----
    print("  [2/3] VIX / NASDAQ...")
    market_data = yf.download(
        ["^VIX", "^IXIC"],
        start="1980-01-01",
        progress=False,
        auto_adjust=False,
    )
    if isinstance(market_data.columns, pd.MultiIndex):
        market_close = market_data["Close"]
    else:
        market_close = market_data[[c for c in market_data.columns if "Close" in c]]
    market_close = market_close.rename(
        columns={"^VIX": "VIX_Close", "^IXIC": "NASDAQ_Close"}
    )
    market_close.index = pd.to_datetime(
        market_close.index, utc=True
    ).tz_localize(None).normalize()

    # ---- 종목별 처리 ----
    print(f"  [3/3] 종목별 처리 ({len(TICKERS_FRESH)})...")
    datasets = []
    for ticker in TICKERS_FRESH:
        try:
            df_stock = yf.Ticker(ticker).history(period="max")
            if df_stock.empty:
                print(f"    ❌ {ticker}: empty")
                continue
            df_stock.index = pd.to_datetime(
                df_stock.index, utc=True
            ).tz_localize(None).normalize()
            df_stock.index.name = "Date"

            df = df_stock.join(market_close, how="left")
            df = df.join(df_fred, how="left")
            df.ffill(inplace=True)
            df.bfill(inplace=True)

            df["RSI_14"] = ta.momentum.RSIIndicator(
                close=df["Close"], window=14
            ).rsi()
            df["ATR_14"] = ta.volatility.AverageTrueRange(
                high=df["High"], low=df["Low"], close=df["Close"], window=14
            ).average_true_range()
            df["SMA_20"] = ta.trend.SMAIndicator(
                close=df["Close"], window=20
            ).sma_indicator()

            df["Returns"] = df["Close"].pct_change()
            df["Realized_Vol_20d"] = df["Returns"].rolling(window=20).std()

            df["Month"] = df.index.month.astype(str)
            df["Day_of_Week"] = df.index.dayofweek.astype(str)

            df["Target_Return_5d"] = df["Close"].shift(-5) / df["Close"] - 1.0

            if "VIXCLS" in df.columns:
                df.drop(columns=["VIXCLS"], inplace=True)

            df.dropna(inplace=True)
            df = df.reset_index()
            df["group_id"] = ticker

            print(
                f"    ✅ {ticker}: {len(df)}행 "
                f"({df['Date'].min().date()} ~ {df['Date'].max().date()})"
            )
            datasets.append(df)
        except Exception as e:
            print(f"    ❌ {ticker}: {e}")

    if not datasets:
        raise RuntimeError("fresh fetch 실패: 종목 데이터가 하나도 없습니다.")

    final_df = pd.concat(datasets, ignore_index=True)
    min_dates = final_df.groupby("group_id")["Date"].min()
    common_start = min_dates.max()
    final_df = final_df[final_df["Date"] >= common_start].copy()

    final_df = final_df.sort_values(["group_id", "Date"]).reset_index(drop=True)
    final_df = (
        final_df.groupby("group_id", group_keys=False)
                .tail(KEEP_LAST_N_TRADING_DAYS)
                .reset_index(drop=True)
    )

    final_df["group_id"] = final_df["group_id"].astype(str)
    final_df["Month"] = final_df["Month"].astype(str)
    final_df["Day_of_Week"] = final_df["Day_of_Week"].astype(str)
    final_df["time_idx"] = final_df.groupby("group_id").cumcount()

    print(
        f"  📊 마지막 {KEEP_LAST_N_TRADING_DAYS} 거래일 슬라이스: "
        f"{len(final_df)} rows, {final_df['group_id'].nunique()} 종목"
    )
    print(
        f"     기간: {final_df['Date'].min().date()} ~ {final_df['Date'].max().date()}"
    )
    return final_df


# ------------------------------------------------------------
# 미래 행 확장 (방어적 NaN 처리)
# ------------------------------------------------------------
def extend_with_future_rows(df: pd.DataFrame, n_future: int = 10) -> pd.DataFrame:
    """
    각 group_id 별로 마지막 거래일 다음 n_future 영업일 행을 추가.

    NaN 이 절대 남지 않도록 다중 안전장치 적용:
      1) 미래 각 행을 dict 로 명시적으로 구성
         - Date: 영업일 캘린더
         - Month, Day_of_Week: 날짜에서 계산
         - Target_Return_5d: 0.0 (placeholder)
         - vol_group 등 static: last_row 값 복사
         - 그 외: last_row 값 (NaN 이면 0.0 fallback)
      2) 원본과 dtype 일치
      3) 최종 fillna(0.0) 로 모든 numeric NaN 제거
    """
    pieces = []
    for g, sub in df.groupby(GROUP_ID, sort=False):
        sub = sub.sort_values("Date").reset_index(drop=True)
        last_date = sub["Date"].iloc[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=n_future
        )
        last_row = sub.iloc[-1]

        future_records = []
        for date in future_dates:
            rec = {}
            for col in sub.columns:
                if col == "Date":
                    rec[col] = date
                elif col == GROUP_ID:
                    rec[col] = g
                elif col == "Month":
                    rec[col] = str(date.month)
                elif col == "Day_of_Week":
                    rec[col] = str(date.dayofweek)
                elif col == TARGET:
                    rec[col] = 0.0
                elif col == TIME_IDX:
                    rec[col] = 0
                else:
                    val = last_row[col]
                    rec[col] = 0.0 if pd.isna(val) else val
            future_records.append(rec)

        future_df = pd.DataFrame(future_records, columns=sub.columns)

        for col in sub.columns:
            try:
                future_df[col] = future_df[col].astype(sub[col].dtype)
            except (ValueError, TypeError):
                pass

        pieces.append(pd.concat([sub, future_df], ignore_index=True))

    extended = pd.concat(pieces, ignore_index=True)
    extended[TIME_IDX] = extended.groupby(GROUP_ID).cumcount()

    numeric_cols = extended.select_dtypes(include=[np.number]).columns.tolist()
    nan_before = int(extended[numeric_cols].isna().sum().sum())
    if nan_before > 0:
        per_col = extended[numeric_cols].isna().sum()
        print(f"[INFO] {nan_before} NaN values found in numeric cols, filling with 0.0:")
        print(per_col[per_col > 0])
        extended[numeric_cols] = extended[numeric_cols].fillna(0.0)

    target_nan = int(extended[TARGET].isna().sum())
    total_nan = int(extended.isna().sum().sum())
    print(f"[CHECK] df_ext shape={extended.shape}  total_nan={total_nan}  target_nan={target_nan}")
    assert target_nan == 0, f"Target still has {target_nan} NaN after fill"
    assert total_nan == 0, f"DataFrame still has {total_nan} NaN after fill"

    return extended


# ------------------------------------------------------------
# XAI 헬퍼
# ------------------------------------------------------------
def _slice_output(output: dict, idx: slice) -> dict:
    sliced = {}
    for k, v in output.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            sliced[k] = v[idx]
        else:
            sliced[k] = v
    return sliced


def _var_importance_df(importance: torch.Tensor, names: list[str]) -> pd.DataFrame:
    arr = importance.detach().cpu().numpy().flatten()
    if len(arr) != len(names):
        m = min(len(arr), len(names))
        arr, names = arr[:m], names[:m]
    df = pd.DataFrame({"variable": names, "importance": arr})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _save_var_importance_plot(df_imp: pd.DataFrame, title: str, out_path: Path) -> None:
    df_plot = df_imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(df_plot) * 0.28)))
    ax.barh(df_plot["variable"], df_plot["importance"], color="C0")
    ax.set_title(title)
    ax.set_xlabel("Importance (sum of selection weights)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_attention_plot(attention: torch.Tensor, title: str, out_path: Path) -> None:
    arr = attention.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    if arr.ndim == 1:
        ax.plot(np.arange(-len(arr), 0), arr, marker="o")
        ax.set_xlabel("Encoder time step (relative to forecast start)")
        ax.set_ylabel("Attention weight")
    elif arr.ndim == 2:
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_xlabel("Encoder time step")
        ax.set_ylabel("Decoder step")
        plt.colorbar(im, ax=ax, label="Attention")
    else:
        flat = arr.mean(axis=tuple(range(arr.ndim - 1)))
        ax.plot(np.arange(-len(flat), 0), flat, marker="o")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_raw_xai_csv(
    g: str,
    out_g: dict,
    encoder_vars: list[str],
    decoder_vars: list[str],
    static_vars: list[str],
    quantiles: list[float],
    last_obs_date: pd.Timestamp,
    raw_dir: Path,
) -> None:
    """종목별 raw XAI 텐서를 정제 전 형태로 CSV 로 저장."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _to_2d(t):
        arr = t.detach().cpu().numpy()
        arr = np.squeeze(arr)
        if arr.ndim >= 3:
            arr = arr.mean(axis=0)
        return arr

    if "prediction" in out_g and isinstance(out_g["prediction"], torch.Tensor):
        pred = _to_2d(out_g["prediction"])
        if pred.ndim == 2:
            future_dates = pd.bdate_range(
                start=last_obs_date + pd.Timedelta(days=1), periods=pred.shape[0]
            )
            cols = [f"Q{q}" for q in quantiles[:pred.shape[1]]]
            df_p = pd.DataFrame(
                pred, index=[d.date() for d in future_dates], columns=cols
            )
            df_p.index.name = "Date"
            df_p.to_csv(raw_dir / f"{g}_predictions.csv")

    if "encoder_variables" in out_g and isinstance(
        out_g["encoder_variables"], torch.Tensor
    ):
        enc = _to_2d(out_g["encoder_variables"])
        if enc.ndim == 2:
            n_time, n_var = enc.shape
            cols = encoder_vars[:n_var] if encoder_vars else [f"var_{i}" for i in range(n_var)]
            idx = [f"T-{n_time - i}" for i in range(n_time)]
            df_e = pd.DataFrame(enc, index=idx, columns=cols)
            df_e.index.name = "encoder_step"
            df_e.to_csv(raw_dir / f"{g}_encoder_variable_weights.csv")

    if "decoder_variables" in out_g and isinstance(
        out_g["decoder_variables"], torch.Tensor
    ):
        dec = _to_2d(out_g["decoder_variables"])
        if dec.ndim == 2 and dec.shape[1] > 0:
            n_time, n_var = dec.shape
            cols = decoder_vars[:n_var] if decoder_vars else [f"var_{i}" for i in range(n_var)]
            future_dates = pd.bdate_range(
                start=last_obs_date + pd.Timedelta(days=1), periods=n_time
            )
            df_d = pd.DataFrame(
                dec, index=[d.date() for d in future_dates], columns=cols
            )
            df_d.index.name = "Date"
            df_d.to_csv(raw_dir / f"{g}_decoder_variable_weights.csv")

    if "static_variables" in out_g and isinstance(
        out_g["static_variables"], torch.Tensor
    ):
        stat = out_g["static_variables"].detach().cpu().numpy()
        if stat.ndim >= 2:
            stat = stat.mean(axis=0)
        stat = stat.flatten()
        if static_vars and len(stat) > 0:
            n = min(len(stat), len(static_vars))
            df_s = pd.DataFrame({
                "variable": static_vars[:n],
                "weight": stat[:n],
            })
            df_s.to_csv(raw_dir / f"{g}_static_variable_weights.csv", index=False)


def _save_combined_summary(
    g: str,
    fc: pd.DataFrame,
    hist: pd.DataFrame,
    interp_g: dict,
    encoder_vars: list[str],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1])

    ax_ret = fig.add_subplot(gs[0, :])
    ax_ret.plot(
        hist["Date"], hist["Target_Return_5d"],
        color="black", alpha=0.6, label="Realized 5d return (last 60d)",
    )
    ax_ret.axhline(0, color="grey", linewidth=0.8, alpha=0.5)
    ax_ret.axvline(
        hist["Date"].iloc[-1], linestyle="--", color="grey",
        alpha=0.6, label="last observed (T)",
    )
    x_future = pd.to_datetime(fc["Date"])
    ax_ret.plot(
        x_future, fc["Q0.5_return_5d"],
        color="C0", marker="o", label="Q0.5 forecast (median)",
    )
    ax_ret.fill_between(
        x_future, fc["Q0.05_return_5d"], fc["Q0.5_return_5d"],
        color="C0", alpha=0.2, label="Q0.05 ~ Q0.5 downside band",
    )
    ax_ret.set_title(
        f"{g}: T+1 ~ T+{len(fc)} forecast — Q0.5 + Q0.05~Q0.5 downside band"
    )
    ax_ret.set_ylabel("5-day return")
    ax_ret.legend(loc="best", fontsize=9)
    ax_ret.grid(True, alpha=0.3)
    plt.setp(ax_ret.get_xticklabels(), rotation=20)

    ax_var = fig.add_subplot(gs[1, 0])
    if "encoder_variables" in interp_g:
        df_enc = _var_importance_df(interp_g["encoder_variables"], encoder_vars)
        df_top = df_enc.head(10).iloc[::-1]
        ax_var.barh(df_top["variable"], df_top["importance"], color="C2")
        ax_var.set_title(f"{g}: Top-10 encoder variable importance")
        ax_var.set_xlabel("Importance")
        ax_var.grid(True, alpha=0.3, axis="x")

    ax_att = fig.add_subplot(gs[1, 1])
    if "attention" in interp_g:
        arr = interp_g["attention"].detach().cpu().numpy()
        if arr.ndim == 1:
            ax_att.plot(np.arange(-len(arr), 0), arr, marker="o", color="C3")
            ax_att.set_xlabel("Encoder time step (relative)")
            ax_att.set_ylabel("Attention")
        elif arr.ndim == 2:
            im = ax_att.imshow(arr, aspect="auto", cmap="viridis")
            ax_att.set_xlabel("Encoder time step")
            ax_att.set_ylabel("Decoder step")
            plt.colorbar(im, ax=ax_att, label="Attention")
        else:
            flat = arr.mean(axis=tuple(range(arr.ndim - 1)))
            ax_att.plot(np.arange(-len(flat), 0), flat, marker="o", color="C3")
        ax_att.set_title(f"{g}: Attention pattern")
        ax_att.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]

    horizon = data_cfg["horizon"]
    window_size = data_cfg["window_size"]
    quantiles = model_cfg["quantiles"]

    # ---- (A) 학습 CSV → train_ds 템플릿 (normalizer 상태만 필요) ----
    print("\n[학습 CSV 로드] train_ds 템플릿 (normalizer / scaler) 복원용")
    df_train = load_data()
    print(f"  학습 데이터: {len(df_train)} rows, "
          f"종목 {sorted(df_train[GROUP_ID].unique())}")
    print(f"  기간: {df_train['Date'].min().date()} ~ {df_train['Date'].max().date()}")
    print(f"  인코더 길이: {window_size}, 예측 horizon: {horizon}")

    ckpts = glob.glob(str(CHECKPOINT_DIR / "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("model/saved/ 에 체크포인트가 없습니다.")
    ckpt_path = sorted(ckpts)[-1]
    print(f"  체크포인트: {ckpt_path}")

    vix_mean, vix_std = get_vix_stats(df_train)
    train_ds, _ = build_dataset(
        df_train,
        max_encoder_length=window_size,
        max_prediction_length=horizon,
    )

    # M4-Combined 모드: vol_group_map 로드 + ckpt 기반 architecture 자동 추정
    mode = "m4_combined"
    vol_group_map = _load_vol_group_map()
    if vol_group_map is None:
        raise FileNotFoundError(
            "data/raw/vol_group_map.json 이 없습니다. "
            "먼저 build 스크립트를 실행하여 vol_group_map.json 을 생성하세요."
        )
    print(f"  mode: {mode}")
    print(f"  vol_group_map: {len(vol_group_map)} tickers")

    model_kwargs = _build_model_kwargs(
        model_cfg, vix_mean, vix_std,
        mode=mode, df=df_train, vol_group_map=vol_group_map,
        ckpt_path=ckpt_path,
    )
    model = M4FullModel.from_dataset(dataset=train_ds, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tft: TemporalFusionTransformer = model.tft

    # ---- (B) yfinance + FRED 에서 신선한 데이터 수집 → 실제 추론 입력 ----
    df = fetch_fresh_data()

    # vol_group 컬럼 추가 (m4_combined 학습 시 static_categorical 로 사용됨)
    default_vol_group = GROUP_LABELS[len(GROUP_LABELS) // 2]  # mid_vol
    df["vol_group"] = (
        df["group_id"].map(vol_group_map).fillna(default_vol_group).astype(str)
    )
    vol_group_counts = df.groupby("vol_group")["group_id"].nunique().to_dict()
    print(f"  vol_group 컬럼 추가: {vol_group_counts}")

    # 학습 CSV 의 컬럼 구성과 일치시키기.
    # fetch_fresh_data 에서 FRED API 실패 등으로 누락된 컬럼이 있을 수 있음.
    # 학습 CSV 의 마지막값으로 보충 (FRED 거시 변수는 모든 종목 동일하므로 글로벌 마지막값으로 충분).
    train_cols = set(df_train.columns)
    fresh_cols = set(df.columns)
    missing = sorted(train_cols - fresh_cols)
    if missing:
        print(f"  ⚠️ fresh data 에서 누락된 컬럼 {len(missing)}개 → 학습 CSV 값으로 보충")
        df_train_sorted = df_train.sort_values("Date")
        for col in missing:
            try:
                fallback = df_train_sorted[col].dropna().iloc[-1]
            except (IndexError, KeyError):
                fallback = 0.0
            df[col] = fallback
            print(f"     {col} = {fallback}")

    # 반대로 fresh 에만 있고 학습엔 없는 컬럼은 제거
    extra = sorted(fresh_cols - train_cols - {"Date"})
    if extra:
        print(f"  ⚠️ fresh data 에만 있는 컬럼 {len(extra)}개 제거: {extra}")
        df = df.drop(columns=extra)

    # 컬럼 순서를 학습 CSV 와 동일하게 정렬
    common_cols = [c for c in df_train.columns if c in df.columns]
    df = df[common_cols]
    print(f"  컬럼 정합성 OK: fresh {len(df.columns)}개 vs 학습 {len(df_train.columns)}개")


    # ---- 미래 행 확장 → predict 데이터셋 ----
    print(f"\n미래 {horizon}영업일 행 추가 중...")
    df_ext = extend_with_future_rows(df, n_future=horizon)

    # train_ds 의 fitted normalizer 가 fresh data 에 적용됨
    predict_ds = TimeSeriesDataSet.from_dataset(
        train_ds, df_ext, predict=True, stop_randomization=True
    )
    predict_loader = predict_ds.to_dataloader(
        train=False, batch_size=64, num_workers=0
    )

    # ---- 한 번의 raw 추론으로 예측 + XAI 모두 확보 ----
    print("\nraw 추론 (예측 + XAI 동시 추출)...")
    raw_predictions = tft.predict(predict_loader, mode="raw", return_x=True)
    output = raw_predictions.output
    x_batch = raw_predictions.x

    pred_tensor = output["prediction"]
    if isinstance(pred_tensor, torch.Tensor):
        preds = pred_tensor.detach().cpu().numpy()
    else:
        preds = np.asarray(pred_tensor)
    print(f"예측 텐서 shape: {preds.shape}  (n_groups, horizon, n_quantiles)")

    group_mapping = {i: g for i, g in enumerate(sorted(df[GROUP_ID].unique()))}
    group_ints = x_batch["groups"][:, 0].cpu().numpy()
    all_groups = [group_mapping[g] for g in group_ints]

    encoder_vars = list(tft.encoder_variables)
    decoder_vars = list(tft.decoder_variables)
    static_vars = list(tft.static_variables)

    # ---- (1) 수익률 결과 정리: T+1 ~ T+horizon ----
    rows = []
    for i, g in enumerate(all_groups):
        sub = df[df[GROUP_ID] == g].sort_values("Date")
        last_date = sub["Date"].iloc[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=preds.shape[1]
        )
        for t in range(preds.shape[1]):
            ret_row = {
                "group_id": g,
                "future_day": t + 1,
                "Date": future_dates[t].date(),
                "window_end_estimate": (
                    future_dates[t] + pd.tseries.offsets.BDay(5)
                ).date(),
            }
            for qi, q in enumerate(quantiles):
                ret_row[f"Q{q}_return_5d"] = float(preds[i, t, qi])
            rows.append(ret_row)

    df_ret = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    ret_path = OUTPUT_DIR / "predict_xai_returns.csv"
    df_ret.to_csv(ret_path, index=False)

    print(f"\n=== T+1 ~ T+{horizon} 5일 누적 수익률 분위수 예측 ===")
    print("(각 행: 'Date 시점에서 본 향후 5거래일 누적 수익률' 분위수)")
    print(df_ret.to_string(index=False))
    print(f"\n수익률 CSV: {ret_path}")

    # ---- (2) 전체 평균 XAI ----
    print("\n전체 평균(reduction='sum') interpretation 계산...")
    interp_all = tft.interpret_output(output, reduction="sum")
    xai_summary_rows: list[pd.DataFrame] = []

    if "encoder_variables" in interp_all:
        df_enc_all = _var_importance_df(interp_all["encoder_variables"], encoder_vars)
        _save_var_importance_plot(
            df_enc_all,
            "Encoder variable importance (avg over all groups)",
            XAI_DIR / "xai_ALL_encoder_variables.png",
        )
        df_enc_all["scope"] = "encoder"; df_enc_all["group_id"] = "ALL"
        xai_summary_rows.append(df_enc_all)

    if "decoder_variables" in interp_all and len(decoder_vars) > 0:
        df_dec_all = _var_importance_df(interp_all["decoder_variables"], decoder_vars)
        _save_var_importance_plot(
            df_dec_all,
            "Decoder variable importance (avg over all groups)",
            XAI_DIR / "xai_ALL_decoder_variables.png",
        )
        df_dec_all["scope"] = "decoder"; df_dec_all["group_id"] = "ALL"
        xai_summary_rows.append(df_dec_all)

    if "static_variables" in interp_all and len(static_vars) > 0:
        df_stat_all = _var_importance_df(interp_all["static_variables"], static_vars)
        _save_var_importance_plot(
            df_stat_all,
            "Static variable importance (avg over all groups)",
            XAI_DIR / "xai_ALL_static_variables.png",
        )
        df_stat_all["scope"] = "static"; df_stat_all["group_id"] = "ALL"
        xai_summary_rows.append(df_stat_all)

    if "attention" in interp_all:
        _save_attention_plot(
            interp_all["attention"],
            "Attention (avg over all groups, decoder→encoder)",
            XAI_DIR / "xai_ALL_attention.png",
        )

    # ---- (3) 종목별 XAI + 통합 figure ----
    print("\n종목별 interpretation + 통합 figure 생성...")
    text_explanations: list[str] = []
    for g in sorted(df[GROUP_ID].unique()):
        idxs = [i for i, name in enumerate(all_groups) if name == g]
        if not idxs:
            continue
        sl = slice(idxs[0], idxs[-1] + 1)
        out_g = _slice_output(output, sl)
        interp_g = tft.interpret_output(out_g, reduction="sum")

        if "encoder_variables" in interp_g:
            df_enc = _var_importance_df(interp_g["encoder_variables"], encoder_vars)
            _save_var_importance_plot(
                df_enc, f"{g}: Encoder variable importance",
                XAI_DIR / f"xai_{g}_encoder_variables.png",
            )
            df_enc["scope"] = "encoder"; df_enc["group_id"] = g
            xai_summary_rows.append(df_enc)

        if "decoder_variables" in interp_g and len(decoder_vars) > 0:
            df_dec = _var_importance_df(interp_g["decoder_variables"], decoder_vars)
            _save_var_importance_plot(
                df_dec, f"{g}: Decoder variable importance",
                XAI_DIR / f"xai_{g}_decoder_variables.png",
            )
            df_dec["scope"] = "decoder"; df_dec["group_id"] = g
            xai_summary_rows.append(df_dec)

        if "static_variables" in interp_g and len(static_vars) > 0:
            df_stat = _var_importance_df(interp_g["static_variables"], static_vars)
            _save_var_importance_plot(
                df_stat, f"{g}: Static variable importance",
                XAI_DIR / f"xai_{g}_static_variables.png",
            )
            df_stat["scope"] = "static"; df_stat["group_id"] = g
            xai_summary_rows.append(df_stat)

        if "attention" in interp_g:
            _save_attention_plot(
                interp_g["attention"],
                f"{g}: Attention (decoder→encoder)",
                XAI_DIR / f"xai_{g}_attention.png",
            )

        fc_g = df_ret[df_ret["group_id"] == g].sort_values("Date")
        hist_g = df[df[GROUP_ID] == g].sort_values("Date").tail(60)
        _save_combined_summary(
            g, fc_g, hist_g, interp_g, encoder_vars,
            OUTPUT_DIR / f"predict_xai_summary_{g}.png",
        )

        last_obs_date = df[df[GROUP_ID] == g]["Date"].max()

        _save_raw_xai_csv(
            g, out_g, encoder_vars, decoder_vars, static_vars,
            quantiles, last_obs_date, XAI_DIR / "raw",
        )

        explanation = generate_text_explanation(
            g, fc_g, hist_g, interp_g, encoder_vars, last_obs_date
        )
        txt_path = OUTPUT_DIR / f"predict_xai_summary_{g}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(explanation)
        text_explanations.append(explanation)
        print(f"\n{explanation}")

    if xai_summary_rows:
        summary = pd.concat(xai_summary_rows, ignore_index=True)
        summary = summary[["group_id", "scope", "variable", "importance"]]
        summary_path = XAI_DIR / "xai_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nXAI summary CSV: {summary_path}")

        print("\n=== 종목별 Encoder variable Top 5 ===")
        for g, sub in summary[summary["scope"] == "encoder"].groupby("group_id"):
            if g == "ALL":
                continue
            top = sub.sort_values("importance", ascending=False).head(5)
            print(f"\n[{g}]")
            print(top[["variable", "importance"]].to_string(index=False))

    if text_explanations:
        report_path = OUTPUT_DIR / "predict_xai_report.txt"
        header = (
            "TFT 예측 + XAI 자동 해설 보고서\n"
            f"생성: predict_with_xai.py\n"
            f"horizon: {horizon} 영업일\n"
            f"종목 수: {len(text_explanations)}\n"
            "=" * 60 + "\n\n"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n\n".join(text_explanations))
        print(f"\n텍스트 해설 통합 보고서: {report_path}")

    print(f"\n결과: {OUTPUT_DIR}")
    print(f"  - predict_xai_returns.csv      (T+1 ~ T+{horizon} 분위수 수익률)")
    print(f"  - predict_xai_summary_*.png    (종목별 통합 figure)")
    print(f"  - predict_xai_summary_*.txt    (종목별 텍스트 해설)")
    print(f"  - predict_xai_report.txt       (모든 종목 통합 보고서)")
    print(f"  - xai/                         (변수 중요도/attention)")
    print(f"  - xai/raw/                     (정제 전 raw XAI 텐서 CSV)")
    print("\n완료.")


if __name__ == "__main__":
    main()