"""
학습된 DeepAR 체크포인트로 매일 production 추론 + SHAP XAI.

predict_with_xai.py (TFT + 내장 interpret_output) 의 DeepAR 버전.
DeepAR 은 TFT 의 variable selection / attention 같은 내장 XAI 가 없으므로
모델-불가지론적 (model-agnostic) 인 SHAP KernelExplainer 로 변수 중요도를 구한다.

데이터 흐름:
  1. df_train = load_data()                                          # 학습 CSV → train_ds (normalizer 상태)
  2. df_fresh = fetch_fresh_data() (또는 --use_csv 시 df_train tail)  # 최근 N 거래일 입력
  3. df_ext = extend_with_future_rows(df_fresh, horizon)              # 미래 행 추가 (known reals 자동 채움)
  4. predict_ds = TimeSeriesDataSet.from_dataset(train_ds, df_ext, predict=True)
  5. model.predict(x) → (B, T, Q) quantile 텐서
  6. SHAP: encoder window 의 "마지막 timestep 의 real feature 값" 을 perturb 해서
          종목별/전체 SHAP 값을 산출 → 어떤 변수가 예측에 가장 큰 영향을 주는가?

출력 디렉토리: model/saved/
  - predict_deepar_xai_returns.csv         # 종목 × horizon × quantiles
  - predict_deepar_summary_<TICKER>.png    # 종목별 통합 figure (Q0.5 + 하방 밴드 + SHAP)
  - predict_deepar_summary_<TICKER>.txt    # 종목별 텍스트 해설
  - predict_deepar_xai_report.txt          # 전체 통합 보고서
  - xai_shap/
      shap_global_summary.png              # 전체 종목 SHAP summary (mean(|shap|))
      shap_<TICKER>_bar.png                # 종목별 SHAP bar plot
      shap_<TICKER>_values.csv             # 종목별 raw SHAP 값
      shap_summary.csv                     # 통합 (group_id × variable × shap_value)

실행:
  python predict_with_xai_deepar.py                     # CSV 기반 (FRED API key 불필요)
  python predict_with_xai_deepar.py --fresh             # yfinance/FRED 신선 데이터 사용
  python predict_with_xai_deepar.py --shap_nsamples 200 # SHAP 정밀도 ↑ (대신 느림)
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import yaml
from pytorch_forecasting import TimeSeriesDataSet

from model.m3_full_model.dataset import (
    GROUP_ID,
    TIME_IDX,
    build_dataset_for_deepar,
    get_vix_stats,
    load_data,
)
from model.m3_full_model.deepar_model import M3FullModel
from validation.backtest.deepar_backtest import (
    _build_model_kwargs,
    _find_deepar_checkpoint,
)

CONFIG_PATH = Path("configs/config.yaml")
OUTPUT_DIR = Path("model/saved")
SHAP_DIR = OUTPUT_DIR / "xai_shap"

TARGET = "Target_Return_5d"
KEEP_LAST_N_TRADING_DAYS = 60

# ============================================================
# 변수 카테고리 매핑 (텍스트 해설용 — predict_with_xai.py 와 동일)
# ============================================================
VARIABLE_LABELS = {
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
    "NASDAQ_Close": ("나스닥 종가", "market_risk"),
    "VIX_Close": ("VIX (시장 공포지수)", "market_risk"),
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
}

CATEGORY_LABELS = {
    "price_tech": "가격/기술 지표",
    "market_risk": "시장 리스크",
    "macro": "거시 경제 변수",
    "other": "기타",
}


def _categorize(var_name: str) -> tuple[str, str]:
    if var_name in VARIABLE_LABELS:
        return VARIABLE_LABELS[var_name]
    return (var_name, "other")


# ============================================================
# 미래 행 확장 (predict_with_xai.py 와 동일 로직)
# ============================================================
def extend_with_future_rows(df: pd.DataFrame, n_future: int = 10) -> pd.DataFrame:
    """각 group_id 별로 마지막 거래일 다음 n_future 영업일 행을 추가.
    DeepAR 은 time_varying_known_reals 가 미래에도 필요하므로
    last_row 값을 복사 (캘린더 변수만 새로 계산)."""
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
        print(f"[INFO] {nan_before} NaN in numeric cols → fillna(0.0)")
        extended[numeric_cols] = extended[numeric_cols].fillna(0.0)

    assert extended.isna().sum().sum() == 0, "df_ext still has NaN after fill"
    return extended


# ============================================================
# SHAP 헬퍼
# ============================================================
def _clone_batch_at(x_batch: dict, group_idx: int, n_reps: int) -> dict:
    """x_batch[group_idx:group_idx+1] 을 n_reps 만큼 복제한 새 batch 를 반환.

    SHAP wrapper 가 perturbation 들을 한 번의 model.predict 로 평가할 수 있도록
    하나의 종목 windou 를 n_reps 만큼 펼친 batch 를 만든다.
    """
    cloned: dict = {}
    for k, v in x_batch.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            sample = v[group_idx:group_idx + 1].clone()
            repeat_dims = [n_reps] + [1] * (sample.dim() - 1)
            cloned[k] = sample.repeat(*repeat_dims)
        else:
            cloned[k] = v
    return cloned


def _get_real_var_names(model: M3FullModel, n_reals: int) -> list[str]:
    """encoder_cont 의 마지막 차원에 대응하는 real 변수 이름 리스트."""
    candidates = []
    deepar = model.deepar
    for attr in ("reals", "encoder_variables"):
        if hasattr(deepar, attr):
            names = list(getattr(deepar, attr))
            if len(names) == n_reals:
                return names
            candidates.append((attr, len(names), names))
    # 폴백: 길이를 맞춰서 자르거나 패딩
    print(f"[WARN] reals 이름 매칭 실패. 후보: {[(a, n) for a, n, _ in candidates]}")
    if candidates:
        attr, n, names = candidates[0]
        if n >= n_reals:
            return names[:n_reals]
        return names + [f"feat_{i}" for i in range(n, n_reals)]
    return [f"feat_{i}" for i in range(n_reals)]


def compute_shap_for_ticker(
    model: M3FullModel,
    x_batch: dict,
    ticker_idx: int,
    background: np.ndarray,
    explain_input: np.ndarray,
    quantiles: list[float],
    nsamples: int,
    horizon: int,
    device: torch.device,
) -> np.ndarray:
    """
    하나의 종목에 대한 SHAP 값을 계산.

    - background: (n_bg, n_reals) — 다른 종목/시점의 'encoder 마지막 step' 스냅샷
    - explain_input: (n_reals,) — 이 종목의 실제 'encoder 마지막 step' 값
    - 반환: (n_reals,) SHAP 값. f(x) = horizon 평균 Q0.5 forecast.
    """
    q_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    encoder_cont = x_batch["encoder_cont"]
    cont_dtype = encoder_cont.dtype

    def f(perturbed: np.ndarray) -> np.ndarray:
        # perturbed: (m, n_reals)
        m = perturbed.shape[0]
        if m == 0:
            return np.zeros(0, dtype=np.float32)
        batch_clone = _clone_batch_at(x_batch, ticker_idx, m)
        new_vals = torch.as_tensor(perturbed, dtype=cont_dtype, device=device)
        batch_clone["encoder_cont"][:, -1, :] = new_vals
        for k, v in batch_clone.items():
            if isinstance(v, torch.Tensor):
                batch_clone[k] = v.to(device)
        with torch.no_grad():
            preds = model.predict(batch_clone)  # (m, T, Q)
        return preds[:, :horizon, q_idx].mean(dim=1).cpu().numpy()

    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(
        explain_input.reshape(1, -1),
        nsamples=nsamples,
        silent=True,
    )
    # shap_values: list 또는 (1, n_reals) ndarray
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.asarray(shap_values).flatten()


# ============================================================
# Plotting
# ============================================================
def _save_shap_bar_plot(
    shap_vals: np.ndarray, var_names: list[str], title: str, out_path: Path, top_k: int = 15
) -> None:
    order = np.argsort(np.abs(shap_vals))[::-1][:top_k]
    vals = shap_vals[order]
    names = [var_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.3)))
    colors = ["C3" if v < 0 else "C0" for v in vals[::-1]]
    ax.barh(names[::-1], vals[::-1], color=colors)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("SHAP value (= 예측 median return 에 기여한 양)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_combined_summary(
    g: str,
    fc: pd.DataFrame,
    hist: pd.DataFrame,
    shap_vals: np.ndarray,
    var_names: list[str],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    ax_ret = fig.add_subplot(gs[0, 0])
    ax_ret.plot(
        hist["Date"], hist["Target_Return_5d"],
        color="black", alpha=0.6, label="Realized 5d return (last 60d)",
    )
    ax_ret.axhline(0, color="grey", linewidth=0.8, alpha=0.5)
    if len(hist) > 0:
        ax_ret.axvline(
            hist["Date"].iloc[-1], linestyle="--", color="grey",
            alpha=0.6, label="last observed (T)",
        )
    x_future = pd.to_datetime(fc["Date"])
    ax_ret.plot(
        x_future, fc["Q0.5_return_5d"],
        color="C0", marker="o", label="Q0.5 forecast (median)",
    )
    if "Q0.05_return_5d" in fc.columns:
        ax_ret.fill_between(
            x_future, fc["Q0.05_return_5d"], fc["Q0.5_return_5d"],
            color="C0", alpha=0.2, label="Q0.05 ~ Q0.5 downside band",
        )
    ax_ret.set_title(f"{g}: DeepAR forecast — Q0.5 + Q0.05~Q0.5 downside band")
    ax_ret.set_ylabel("5-day return")
    ax_ret.legend(loc="best", fontsize=9)
    ax_ret.grid(True, alpha=0.3)
    plt.setp(ax_ret.get_xticklabels(), rotation=20)

    ax_sh = fig.add_subplot(gs[1, 0])
    top_k = 10
    order = np.argsort(np.abs(shap_vals))[::-1][:top_k]
    vals = shap_vals[order]
    names = [var_names[i] for i in order]
    colors = ["C3" if v < 0 else "C2" for v in vals[::-1]]
    ax_sh.barh(names[::-1], vals[::-1], color=colors)
    ax_sh.axvline(0, color="black", linewidth=0.6)
    ax_sh.set_title(f"{g}: Top-{top_k} SHAP feature attribution (median forecast)")
    ax_sh.set_xlabel("SHAP value")
    ax_sh.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_global_shap_summary(
    all_shap: dict[str, np.ndarray], var_names: list[str], out_path: Path, top_k: int = 20
) -> None:
    """모든 종목 SHAP 의 mean(|shap|) → 전역 변수 중요도."""
    mat = np.stack(list(all_shap.values()), axis=0)  # (n_groups, n_reals)
    mean_abs = np.abs(mat).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    vals = mean_abs[order]
    names = [var_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.3)))
    ax.barh(names[::-1], vals[::-1], color="C0")
    ax.set_title("Global SHAP feature importance (mean |shap| over all tickers)")
    ax.set_xlabel("mean |SHAP value|")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================
# 텍스트 해설
# ============================================================
def generate_text_explanation(
    g: str,
    fc_g: pd.DataFrame,
    hist_g: pd.DataFrame,
    shap_vals: np.ndarray,
    var_names: list[str],
    last_obs_date: pd.Timestamp,
) -> str:
    parts: list[str] = []
    parts.append("━" * 50)
    parts.append(f"  [{g}] DeepAR 분위수 예측 + SHAP XAI")
    parts.append(f"  (기준일 T = {last_obs_date.date()}, 예측 구간 = T+1 이후)")
    parts.append("━" * 50)

    # ---- 1. 수익률 전망 ----
    q5 = fc_g["Q0.5_return_5d"].to_numpy()
    median_avg = float(np.mean(q5))
    median_min, median_max = float(np.min(q5)), float(np.max(q5))

    if median_avg > 0.005:
        direction = "상승"
    elif median_avg < -0.005:
        direction = "하락"
    else:
        direction = "횡보"

    parts.append("")
    parts.append("◆ 수익률 전망 (DeepAR)")
    parts.append(
        f"  Q0.5 평균 {median_avg*100:+.2f}% → 5일 누적 수익률이 {direction} 흐름 추정. "
        f"개별 시점 Q0.5 는 {median_min*100:+.2f}% ~ {median_max*100:+.2f}% 범위."
    )

    if "Q0.05_return_5d" in fc_g.columns:
        q_low = fc_g["Q0.05_return_5d"].to_numpy()
        downside_band_avg = float(np.mean(q5 - q_low))
        hist_std = float(hist_g["Target_Return_5d"].std()) if len(hist_g) > 0 else 0.0
        band_ratio_desc = ""
        if hist_std > 0:
            ratio = downside_band_avg / (1.645 * hist_std)
            if ratio > 1.2:
                band_ratio_desc = f" 정규분포 가정 대비 약 {ratio:.1f}배로 넓어 하방 리스크 우려가 큽니다."
            elif ratio < 0.8:
                band_ratio_desc = f" 정규분포 가정 대비 약 {ratio:.1f}배로 좁아 하방 확신이 비교적 강합니다."
            else:
                band_ratio_desc = " 최근 실현 변동성과 유사한 수준입니다."
        parts.append(
            f"  Q0.05~Q0.5 하방 밴드 평균 폭 {downside_band_avg*100:.2f}%p." + band_ratio_desc
        )

    # ---- 2. SHAP 변수 중요도 ----
    parts.append("")
    parts.append("◆ SHAP 변수 기여도 (encoder 마지막 시점의 변수 값이 Q0.5 예측에 미친 영향)")
    abs_vals = np.abs(shap_vals)
    order = np.argsort(abs_vals)[::-1]
    top5 = order[:5]

    by_cat: dict[str, float] = {}
    for i in top5:
        name = var_names[i]
        val = shap_vals[i]
        label, cat = _categorize(name)
        cat_kr = CATEGORY_LABELS.get(cat, "기타")
        sign = "↑" if val > 0 else "↓"
        parts.append(
            f"    {name:24s} SHAP={val:+.5f} {sign}  ({label}, {cat_kr})"
        )

    # 카테고리 점유율
    for i in range(len(var_names)):
        _, cat = _categorize(var_names[i])
        by_cat[cat] = by_cat.get(cat, 0.0) + abs_vals[i]
    total = sum(by_cat.values())
    if total > 0:
        top_cats = sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)[:3]
        cat_str = ", ".join(
            f"{CATEGORY_LABELS.get(c, c)} {v/total*100:.1f}%" for c, v in top_cats
        )
        parts.append(f"  카테고리별 비중 (상위 3): {cat_str}")

        top_cat = top_cats[0][0]
        top_cat_kr = CATEGORY_LABELS.get(top_cat, top_cat)
        reason_map = {
            "price_tech": "자기상관/추세 성분",
            "market_risk": "시장 리스크 신호",
            "macro": "거시 환경 변화",
        }
        parts.append(
            f"  → {top_cat_kr}가 가장 큰 영향. "
            f"이번 예측의 주된 근거는 {reason_map.get(top_cat, '기타 요인')} 으로 추정."
        )

    parts.append("")
    return "\n".join(parts)


# ============================================================
# 메인
# ============================================================
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh", action="store_true",
        help="yfinance + FRED 에서 fresh data fetch (predict_with_xai.py 와 동일 로직). "
             "지정하지 않으면 학습 CSV 의 마지막 N 거래일을 입력으로 사용.",
    )
    parser.add_argument(
        "--shap_nsamples", type=int, default=100,
        help="KernelExplainer 의 nsamples (높을수록 정밀, 느림)",
    )
    parser.add_argument(
        "--shap_background", type=int, default=50,
        help="SHAP background 샘플 수 (학습 데이터 last-step 분포에서 추출)",
    )
    parser.add_argument(
        "--shap_horizon", type=int, default=10,
        help="SHAP 의 target = 첫 N 영업일 Q0.5 평균 forecast",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="추론 device (cuda/cpu)",
    )
    args = parser.parse_args()

    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]

    horizon = data_cfg["horizon"]
    window_size = data_cfg["window_size"]
    quantiles = model_cfg["quantiles"]
    device = torch.device(args.device)

    # ---- (A) 학습 CSV → train_ds 템플릿 ----
    print("\n[학습 CSV 로드] train_ds 템플릿 (normalizer / scaler) 복원용")
    df_train = load_data()
    print(f"  학습 데이터: {len(df_train)} rows, 종목 {df_train[GROUP_ID].nunique()}개")
    print(f"  기간: {df_train['Date'].min().date()} ~ {df_train['Date'].max().date()}")
    print(f"  인코더 길이: {window_size}, 예측 horizon: {horizon}")

    ckpt_path = _find_deepar_checkpoint()
    print(f"  체크포인트: {ckpt_path}")

    vix_mean, vix_std = get_vix_stats(df_train)
    train_ds, _ = build_dataset_for_deepar(
        df_train,
        max_encoder_length=window_size,
        max_prediction_length=horizon,
    )

    model_kwargs = _build_model_kwargs(model_cfg, vix_mean, vix_std)
    model = M3FullModel.from_dataset(dataset=train_ds, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    # ---- (B) 추론 입력 데이터 준비 ----
    if args.fresh:
        try:
            from predict_with_xai import fetch_fresh_data
            df = fetch_fresh_data()
        except ImportError as e:
            raise RuntimeError(
                "fresh fetch 를 위해서는 predict_with_xai.py 의 fetch_fresh_data 가 필요. "
                f"원인: {e}"
            )
    else:
        print("\n[CSV 기반] 학습 CSV 의 마지막 거래일 윈도우를 입력으로 사용 (--fresh 옵션 시 yfinance/FRED)")
        df = (
            df_train.sort_values([GROUP_ID, "Date"])
                    .groupby(GROUP_ID, group_keys=False)
                    .tail(KEEP_LAST_N_TRADING_DAYS)
                    .reset_index(drop=True)
        )
        df[TIME_IDX] = df.groupby(GROUP_ID).cumcount()
        print(
            f"  {len(df)} rows, {df[GROUP_ID].nunique()} 종목, "
            f"기간 {df['Date'].min().date()} ~ {df['Date'].max().date()}"
        )

    # ---- (C) 미래 행 확장 → predict 데이터셋 ----
    print(f"\n미래 {horizon} 영업일 행 추가 중...")
    df_ext = extend_with_future_rows(df, n_future=horizon)
    predict_ds = TimeSeriesDataSet.from_dataset(
        train_ds, df_ext, predict=True, stop_randomization=True
    )
    predict_loader = predict_ds.to_dataloader(
        train=False, batch_size=len(df_ext[GROUP_ID].unique()), num_workers=0
    )

    # ---- (D) 한 batch 로 모든 종목 추론 + SHAP ----
    x_batch, _ = next(iter(predict_loader))
    x_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in x_batch.items()}

    with torch.no_grad():
        preds = model.predict(x_batch).cpu().numpy()  # (B, T, Q)
    print(f"예측 텐서 shape: {preds.shape}  (n_groups, horizon, n_quantiles)")

    # 그룹 매핑
    group_mapping = {i: g for i, g in enumerate(sorted(df[GROUP_ID].unique()))}
    group_ints = x_batch["groups"][:, 0].cpu().numpy()
    all_groups = [group_mapping[g] for g in group_ints]

    n_reals = x_batch["encoder_cont"].shape[-1]
    var_names = _get_real_var_names(model, n_reals)
    print(f"encoder real features: {n_reals}개  ({var_names[:5]} ...)")

    # ---- (E) 수익률 결과 정리 ----
    rows = []
    for i, g in enumerate(all_groups):
        sub = df[df[GROUP_ID] == g].sort_values("Date")
        last_date = sub["Date"].iloc[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=preds.shape[1]
        )
        for t in range(preds.shape[1]):
            row = {
                "group_id": g,
                "future_day": t + 1,
                "Date": future_dates[t].date(),
            }
            for qi, q in enumerate(quantiles):
                row[f"Q{q}_return_5d"] = float(preds[i, t, qi])
            rows.append(row)
    df_ret = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    ret_path = OUTPUT_DIR / "predict_deepar_xai_returns.csv"
    df_ret.to_csv(ret_path, index=False)
    print(f"\n수익률 CSV: {ret_path}")

    # ---- (F) SHAP background 샘플 준비 ----
    # 학습 데이터의 encoder window 마지막 step 값들에서 background 추출.
    # train_ds.to_dataloader 로 한 batch 만 뽑아서 last-step 모음.
    print(f"\nSHAP background 수집 ({args.shap_background} 샘플)...")
    train_loader = train_ds.to_dataloader(train=False, batch_size=args.shap_background, num_workers=0)
    bg_batch, _ = next(iter(train_loader))
    background = bg_batch["encoder_cont"][:, -1, :].cpu().numpy().astype(np.float32)
    print(f"  background shape: {background.shape}")

    # ---- (G) 종목별 SHAP 계산 + 결과 저장 ----
    print(f"\n종목별 SHAP 계산 시작 (nsamples={args.shap_nsamples}, "
          f"shap_horizon={args.shap_horizon} 영업일 Q0.5 평균)")
    shap_horizon = min(args.shap_horizon, preds.shape[1])

    all_shap: dict[str, np.ndarray] = {}
    text_explanations: list[str] = []
    shap_rows: list[pd.DataFrame] = []

    for i, g in enumerate(sorted(set(all_groups))):
        try:
            ticker_idx = all_groups.index(g)
            explain_input = x_batch["encoder_cont"][ticker_idx, -1, :].cpu().numpy().astype(np.float32)

            shap_vals = compute_shap_for_ticker(
                model=model,
                x_batch=x_batch,
                ticker_idx=ticker_idx,
                background=background,
                explain_input=explain_input,
                quantiles=quantiles,
                nsamples=args.shap_nsamples,
                horizon=shap_horizon,
                device=device,
            )
            all_shap[g] = shap_vals
            print(f"  [{i+1:2d}/{len(set(all_groups))}] {g}: ||shap||_1 = {np.abs(shap_vals).sum():.4f}")

            # 저장
            _save_shap_bar_plot(
                shap_vals, var_names, f"{g}: SHAP feature attribution (DeepAR median forecast)",
                SHAP_DIR / f"shap_{g}_bar.png",
            )
            df_s = pd.DataFrame({
                "variable": var_names,
                "shap_value": shap_vals,
                "abs_shap": np.abs(shap_vals),
            }).sort_values("abs_shap", ascending=False).reset_index(drop=True)
            df_s.to_csv(SHAP_DIR / f"shap_{g}_values.csv", index=False)

            df_summary = df_s.copy()
            df_summary["group_id"] = g
            shap_rows.append(df_summary)

            # 통합 figure (forecast + SHAP)
            fc_g = df_ret[df_ret["group_id"] == g].sort_values("Date")
            hist_g = df[df[GROUP_ID] == g].sort_values("Date").tail(60)
            _save_combined_summary(
                g, fc_g, hist_g, shap_vals, var_names,
                OUTPUT_DIR / f"predict_deepar_summary_{g}.png",
            )

            # 텍스트 해설
            last_obs_date = df[df[GROUP_ID] == g]["Date"].max()
            explanation = generate_text_explanation(
                g, fc_g, hist_g, shap_vals, var_names, last_obs_date
            )
            txt_path = OUTPUT_DIR / f"predict_deepar_summary_{g}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(explanation)
            text_explanations.append(explanation)
        except Exception as e:
            print(f"  [{g}] SHAP 계산 실패: {e}")
            continue

    # ---- (H) 전체 통합 결과 ----
    if all_shap:
        _save_global_shap_summary(
            all_shap, var_names,
            SHAP_DIR / "shap_global_summary.png",
        )
        summary = pd.concat(shap_rows, ignore_index=True)
        summary = summary[["group_id", "variable", "shap_value", "abs_shap"]]
        summary_path = SHAP_DIR / "shap_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSHAP 통합 CSV: {summary_path}")

        # 전체 SHAP Top 5 by variable
        global_imp = (
            summary.groupby("variable")["abs_shap"].mean()
                   .sort_values(ascending=False).head(10)
        )
        print("\n=== 전체 종목 평균 SHAP Top 10 변수 ===")
        print(global_imp.to_string())

    # 통합 보고서
    if text_explanations:
        report_path = OUTPUT_DIR / "predict_deepar_xai_report.txt"
        header = (
            "DeepAR 예측 + SHAP XAI 자동 해설 보고서\n"
            "생성: predict_with_xai_deepar.py\n"
            f"종목 수: {len(text_explanations)}\n"
            f"SHAP nsamples={args.shap_nsamples}, background={args.shap_background}, "
            f"horizon(평균)={shap_horizon}\n"
            + "=" * 60 + "\n\n"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n\n".join(text_explanations))
        print(f"텍스트 해설 통합 보고서: {report_path}")

    print(f"\n결과 디렉토리: {OUTPUT_DIR}")
    print("  - predict_deepar_xai_returns.csv       (종목 × horizon × quantiles)")
    print("  - predict_deepar_summary_*.png/.txt    (종목별 forecast + SHAP 통합)")
    print("  - predict_deepar_xai_report.txt        (통합 보고서)")
    print("  - xai_shap/shap_global_summary.png     (전체 변수 중요도)")
    print("  - xai_shap/shap_<TICKER>_bar.png       (종목별 SHAP)")
    print("  - xai_shap/shap_summary.csv            (통합 raw)")
    print("\n완료.")


if __name__ == "__main__":
    main()
