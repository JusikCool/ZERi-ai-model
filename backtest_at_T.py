"""
backtest_at_T.py — 단일 위기 시점 T 에서 세 모델 평가

사용:
    python backtest_at_T.py --date 2020-03-23
    python backtest_at_T.py --date 2022-02-24 --output_dir model/saved/crisis/

데이터 source: tft_processed_panel_v3.csv (이미 모든 시점 데이터 보유)
  - TFT 입력: T 까지의 60 BD × 27 feature
  - Kronos 입력: T 까지의 60 BD × OHLCV
  - 실제 미래값: panel 의 T+1 ~ T+35 close

산출물 (output_dir):
  {date}_predictions.csv  — ticker × future_day × 분위수 × 모델 (TFT/Kronos/Ensemble)
  {date}_actuals.csv      — ticker × future_day × 실제 5일 forward return
  {date}_summary.csv      — 모델별 분위수별 violation count + Kupiec test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# =============================================================
# Kronos import — 우리 model 폴더와 이름 충돌 방지
# Kronos 의 model.py 를 먼저 import 한 뒤 sys.modules 에서 분리
# =============================================================
KRONOS_REPO = Path("./Kronos")
sys.path.insert(0, str(KRONOS_REPO))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402
# sys.modules["model"] = Kronos repo 의 model.py 가 캐싱됨
# 이걸 다른 이름으로 백업 후 제거 → 이제 우리 model 패키지 import 가능
sys.modules["_kronos_model_module"] = sys.modules.pop("model")
sys.path.pop(0)

# =============================================================
# 우리 model 패키지 + 기타 모듈 import
# =============================================================
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer  # noqa: E402

# get_attention_mask 디바이스 패치
_orig_get_attention_mask = TemporalFusionTransformer.get_attention_mask
def _patched_get_attention_mask(self, encoder_lengths, **kwargs):
    mask = _orig_get_attention_mask(self, encoder_lengths, **kwargs)
    return mask.to(encoder_lengths.device)
TemporalFusionTransformer.get_attention_mask = _patched_get_attention_mask

from model.m4.dataset import load_data, build_dataset, get_vix_stats, GROUP_ID, TIME_IDX  # noqa: E402
from model.m4.model import M4FullModel  # noqa: E402
from validation.kupiec.kupiec import kupiec_pof_test  # noqa: E402
from validation.kupiec.coverage_tests import christoffersen_cc_test  # noqa: E402

# =============================================================
# 설정 — ensemble.py 와 동일 default
# =============================================================
QUANTILES = [round(0.05 * i, 2) for i in range(1, 20)]  # 19개
HORIZON = 30
KRONOS_BUFFER = 5
KRONOS_PRED_LEN = HORIZON + KRONOS_BUFFER

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_SAMPLE_COUNT = 10
KRONOS_TEMPERATURE = 1.0
KRONOS_TOP_P = 0.9
KRONOS_MAX_CONTEXT = 512

# Ensemble α 공식
ALPHA_BASE = 0.5
ALPHA_SENSITIVITY = -0.03
VIX_CENTER = 20.0
ALPHA_MIN = 0.2
ALPHA_MAX = 0.8

CHECKPOINT_DIR = Path("model/saved")
VOL_GROUP_MAP_PATH = Path("data/raw/vol_group_map.json")
GROUP_LABELS = ["low_vol", "mid_vol", "high_vol"]
CONFIG_PATH = Path("configs/config.yaml")


# =============================================================
# TFT M4-Combined 모델 로드 (backtest_m4.py 와 동일 패턴)
# =============================================================
def _build_ticker_to_vol_group_idx(panel_df, vol_group_map):
    if vol_group_map is None:
        return None
    sorted_tickers = sorted(panel_df[GROUP_ID].unique())
    label_to_idx = {l: i for i, l in enumerate(GROUP_LABELS)}
    default = GROUP_LABELS[len(GROUP_LABELS) // 2]
    return torch.tensor(
        [label_to_idx[vol_group_map.get(t, default)] for t in sorted_tickers],
        dtype=torch.long,
    )


def load_tft_model(df, config, mode="m4_combined"):
    """M4-Combined 모델 로드."""
    import glob
    pattern = str(CHECKPOINT_DIR / f"{mode}_best_*.ckpt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        raise FileNotFoundError(f"체크포인트 없음: {pattern}")
    ckpt_path = sorted(ckpts)[-1]

    best_params_path = CHECKPOINT_DIR / f"best_params_{mode}.json"
    with open(best_params_path) as f:
        best = json.load(f)

    with open(VOL_GROUP_MAP_PATH) as f:
        vol_group_map = json.load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    vix_mean, vix_std = get_vix_stats(df)

    train_ds, _ = build_dataset(
        df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )

    ticker_to_vol_group_idx = _build_ticker_to_vol_group_idx(df, vol_group_map)

    kwargs = dict(
        learning_rate=best.get("learning_rate", model_cfg["learning_rate"]),
        hidden_size=best.get("hidden_size", model_cfg["hidden_size"]),
        attention_head_size=best.get("attention_head_size", model_cfg["attention_head_size"]),
        dropout=best.get("dropout", model_cfg["dropout"]),
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
        vix_mean=vix_mean,
        vix_std=vix_std,
        crossing_weight=best.get("crossing_weight", 0.1),
        use_garch_sigma=True,
        group_labels=GROUP_LABELS,
        ticker_to_vol_group_idx=ticker_to_vol_group_idx,
        alpha_down_by_group={g: best[f"alpha_down_{g}"] for g in GROUP_LABELS},
        beta_down_by_group={g: best[f"beta_down_{g}"] for g in GROUP_LABELS},
        alpha_up_by_group={g: best[f"alpha_up_{g}"] for g in GROUP_LABELS},
        beta_up_by_group={g: best[f"beta_up_{g}"] for g in GROUP_LABELS},
    )

    model = M4FullModel.from_dataset(dataset=train_ds, **kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, train_ds


# =============================================================
# TFT 단일 T 추론
# =============================================================
def predict_tft_at_T(df_panel, model, train_ds, T_date, config):
    """
    T 까지의 데이터로 T+1 ~ T+30 분위수 예측.

    Returns:
        dict[ticker] = np.ndarray shape (30, 19) — 분위수 예측
    """
    data_cfg = config["data"]
    T = pd.Timestamp(T_date)

    # T 까지의 데이터만 사용 (data leakage 방지)
    df_T = df_panel[df_panel["Date"] <= T].copy()

    # 각 종목 마다 마지막 60 BD 만 추출 → predict mode dataset 생성
    pred_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        df_T,
        predict=True,
        stop_randomization=True,
    )
    pred_loader = pred_ds.to_dataloader(
        train=False,
        batch_size=config["model"]["batch_size"] * 2,
        num_workers=0,
    )

    group_mapping = {i: g for i, g in enumerate(sorted(df_panel[GROUP_ID].unique()))}
    tft_predictions = {}

    with torch.no_grad():
        for batch in pred_loader:
            x, _ = batch
            y_pred = model.predict(x).numpy()  # (batch, horizon, n_quantiles)
            group_ints = x["groups"][:, 0].numpy()
            for bi, gi in enumerate(group_ints):
                ticker = group_mapping[gi]
                tft_predictions[ticker] = y_pred[bi]

    return tft_predictions


# =============================================================
# Kronos 단일 T 추론
# =============================================================
def load_kronos_predictor():
    print(f"[Kronos] Loading {KRONOS_MODEL_NAME} ...")
    tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_NAME)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    try:
        return KronosPredictor(model, tokenizer, max_context=KRONOS_MAX_CONTEXT, device=device)
    except TypeError:
        return KronosPredictor(model, tokenizer, max_context=KRONOS_MAX_CONTEXT)


def predict_kronos_at_T(df_panel, predictor, T_date, window=60):
    """
    각 종목별로 T 까지의 60 BD OHLCV 로 Kronos 추론 → 분위수 예측.

    Returns:
        dict[ticker] = np.ndarray shape (30, 19) — 5일 forward return 분위수
    """
    T = pd.Timestamp(T_date)
    kronos_predictions = {}
    tickers = sorted(df_panel[GROUP_ID].unique())

    for i, ticker in enumerate(tickers, 1):
        df_t = df_panel[
            (df_panel[GROUP_ID] == ticker) & (df_panel["Date"] <= T)
        ].sort_values("Date").tail(window).copy()

        if len(df_t) < window:
            print(f"  [{i:2d}/{len(tickers)}] {ticker:6s}: ❌ 데이터 부족 ({len(df_t)} BD)")
            continue

        df_t = df_t.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df_t["amount"] = df_t["close"] * df_t["volume"]
        df_t.index = pd.DatetimeIndex(df_t["Date"].values)

        x_timestamp = pd.Series(df_t.index)
        future_dates = pd.bdate_range(
            start=df_t.index[-1] + pd.Timedelta(days=1), periods=KRONOS_PRED_LEN
        )
        y_timestamp = pd.Series(future_dates)
        df_input = df_t[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)

        # SAMPLE_COUNT 개 stochastic paths
        paths = []
        for _ in range(KRONOS_SAMPLE_COUNT):
            pred = predictor.predict(
                df=df_input,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=KRONOS_PRED_LEN,
                T=KRONOS_TEMPERATURE,
                top_p=KRONOS_TOP_P,
                sample_count=1,
            )
            paths.append(pred)

        # 5일 forward return 분위수
        closes = np.stack([p["close"].to_numpy() for p in paths], axis=0)
        base = closes[:, :HORIZON]
        future = closes[:, KRONOS_BUFFER : KRONOS_BUFFER + HORIZON]
        returns = future / base - 1.0
        q = np.quantile(returns, QUANTILES, axis=0)  # (n_q, horizon)
        kronos_predictions[ticker] = q.T              # (horizon, n_q)
        print(f"  [{i:2d}/{len(tickers)}] {ticker:6s}: ✅")

    return kronos_predictions


# =============================================================
# Ensemble (VIX-aware α)
# =============================================================
def ensemble_predictions(df_panel, tft_pred, kronos_pred, T_date):
    """
    각 종목 분위수 예측을 α(VIX) 로 가중 평균.

    Returns:
        ensemble_pred: dict[ticker] = np.ndarray (30, 19)
        alpha: scalar (T 시점 VIX 기반)
    """
    T = pd.Timestamp(T_date)
    vix_at_T = df_panel[df_panel["Date"] <= T]["VIX_Close"].iloc[-1]
    alpha = ALPHA_BASE + ALPHA_SENSITIVITY * (vix_at_T - VIX_CENTER)
    alpha = max(ALPHA_MIN, min(ALPHA_MAX, alpha))

    ensemble = {}
    common_tickers = set(tft_pred.keys()) & set(kronos_pred.keys())
    for ticker in common_tickers:
        ensemble[ticker] = alpha * tft_pred[ticker] + (1 - alpha) * kronos_pred[ticker]

    return ensemble, alpha, vix_at_T


# =============================================================
# 실제 미래값 fetch
# =============================================================
def get_actuals(df_panel, T_date):
    """
    각 종목의 T+1 ~ T+30 5일 forward return.

    Returns:
        dict[ticker] = np.ndarray shape (30,) — 5일 forward return
    """
    T = pd.Timestamp(T_date)
    actuals = {}
    tickers = sorted(df_panel[GROUP_ID].unique())

    for ticker in tickers:
        df_t = df_panel[
            (df_panel[GROUP_ID] == ticker) & (df_panel["Date"] > T)
        ].sort_values("Date")

        if len(df_t) < HORIZON + KRONOS_BUFFER:
            continue

        closes = df_t["Close"].values[:HORIZON + KRONOS_BUFFER]
        base = closes[:HORIZON]
        future = closes[KRONOS_BUFFER : KRONOS_BUFFER + HORIZON]
        actuals[ticker] = future / base - 1.0

    return actuals


# =============================================================
# 비교 + 검증
# =============================================================
def compare_models(predictions_dict, actuals, model_name):
    """
    predictions_dict[ticker] = (30, 19)
    actuals[ticker] = (30,)

    Returns:
        분위수별 violation count + Kupiec test 결과 DataFrame
    """
    rows = []
    common_tickers = set(predictions_dict.keys()) & set(actuals.keys())

    for q_i, q in enumerate(QUANTILES):
        violations = 0
        total = 0
        for ticker in common_tickers:
            pred_q = predictions_dict[ticker][:, q_i]  # (30,)
            actual = actuals[ticker]                    # (30,)
            if q < 0.5:
                viol = (actual < pred_q).sum()
            elif q > 0.5:
                viol = (actual > pred_q).sum()
            else:
                viol = (actual < pred_q).sum()  # 중앙값은 하방으로 카운트
            violations += int(viol)
            total += len(pred_q)

        # 명목 violation rate
        nominal_rate = q if q <= 0.5 else (1 - q)
        actual_rate = violations / total if total > 0 else 0

        # Kupiec POF test
        # 모든 종목 × 모든 future_day 의 violation 을 한 시계열로 보고 test
        try:
            n = total
            x = violations
            if x == 0 or x == n:
                kupiec_p = np.nan
            else:
                expected = nominal_rate
                lr_uc = -2 * (
                    x * np.log(expected) + (n - x) * np.log(1 - expected)
                    - x * np.log(x / n) - (n - x) * np.log(1 - x / n)
                )
                from scipy.stats import chi2
                kupiec_p = 1 - chi2.cdf(lr_uc, df=1)
        except Exception:
            kupiec_p = np.nan

        rows.append({
            "model": model_name,
            "quantile": q,
            "nominal_rate": round(nominal_rate, 3),
            "violations": violations,
            "total": total,
            "actual_rate": round(actual_rate, 4),
            "kupiec_p": round(kupiec_p, 4) if not np.isnan(kupiec_p) else np.nan,
            "kupiec_pass": bool(kupiec_p > 0.05) if not np.isnan(kupiec_p) else False,
        })
    return pd.DataFrame(rows)


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="위기 시점 T (YYYY-MM-DD)")
    parser.add_argument("--output_dir", default="model/saved/crisis", type=Path)
    parser.add_argument("--skip_kronos", action="store_true", help="Kronos 추론 생략 (디버그용)")
    args = parser.parse_args()

    T_date = args.date
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Crisis backtest @ T = {T_date}")
    print(f"{'='*60}\n")

    # 데이터 + config 로드
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    df = load_data()
    print(f"[panel] {df['group_id'].nunique()} 종목, {len(df)} rows, "
          f"{df['Date'].min().date()} ~ {df['Date'].max().date()}")

    # === 1. TFT 추론 ===
    print(f"\n[1/4] TFT (M4-Combined) 추론 ...")
    tft_model, train_ds = load_tft_model(df, config, mode="m4_combined")
    tft_pred = predict_tft_at_T(df, tft_model, train_ds, T_date, config)
    print(f"  ✓ {len(tft_pred)} 종목 예측 완료")

    # === 2. Kronos 추론 ===
    if args.skip_kronos:
        print(f"\n[2/4] Kronos SKIP")
        kronos_pred = {}
    else:
        print(f"\n[2/4] Kronos 추론 (예상 ~5분/종목 × 50) ...")
        predictor = load_kronos_predictor()
        kronos_pred = predict_kronos_at_T(df, predictor, T_date)
        print(f"  ✓ {len(kronos_pred)} 종목 예측 완료")

    # === 3. Ensemble ===
    print(f"\n[3/4] Ensemble (VIX-aware α) ...")
    if kronos_pred:
        ensemble_pred, alpha, vix = ensemble_predictions(df, tft_pred, kronos_pred, T_date)
        print(f"  VIX(T)={vix:.2f}, α={alpha:.3f}")
        print(f"  ✓ {len(ensemble_pred)} 종목 ensemble 완료")
    else:
        ensemble_pred = {}
        alpha = None

    # === 4. 실제값 + 비교 ===
    print(f"\n[4/4] 실제 미래값 + 검증 ...")
    actuals = get_actuals(df, T_date)
    print(f"  ✓ {len(actuals)} 종목 실제값 추출")

    reports = []
    reports.append(compare_models(tft_pred, actuals, "TFT (M4-Combined)"))
    if kronos_pred:
        reports.append(compare_models(kronos_pred, actuals, "Kronos"))
    if ensemble_pred:
        reports.append(compare_models(ensemble_pred, actuals, "Ensemble"))

    summary = pd.concat(reports, ignore_index=True)
    summary["T_date"] = T_date

    # 저장
    out_path = args.output_dir / f"{T_date}_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n[저장] {out_path}")

    # 핵심 표 출력
    print(f"\n{'='*60}")
    print(f"  결과 — Kupiec POF Pass (19 quantile 중)")
    print(f"{'='*60}")
    for model_name in summary["model"].unique():
        sub = summary[summary["model"] == model_name]
        n_pass = sub["kupiec_pass"].sum()
        print(f"  {model_name:25s}: {n_pass:2d} / 19")

    print(f"\n{'='*60}")
    print(f"  분위수별 상세")
    print(f"{'='*60}")
    pivot = summary.pivot(index="quantile", columns="model", values="kupiec_pass")
    print(pivot.to_string())


if __name__ == "__main__":
    main()