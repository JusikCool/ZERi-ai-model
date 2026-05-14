"""
ensemble.py — TFT + Kronos 분위수 가중평균 (late fusion) with VIX-aware dynamic α

설계 근거 (TFT 백테스트에서 확인된 약점):
  - TFT 는 우리 학습 데이터(주로 평상시 구간) 에 특화 → 평상시 정확도 우수
    그러나 학습에 포함되지 않은 고변동성 / 공포 구간에서 일반화 성능 저하 확인됨
  - Kronos 는 45개 글로벌 거래소 12B K-line 사전학습 (2008 금융위기, COVID, 각종 시장 충격 포함)
    → 다양한 시장 국면에 강건. 특히 위기 구간에서 상대적 우위 기대
  - 두 모델의 강점이 서로 다른 시장 국면 → VIX 로 국면 판단 → 동적 가중 합성

수식:
  α(VIX) = base + sensitivity · (VIX - center), clamp [α_min, α_max]
  Q_ensemble[q, t] = α(VIX[T_t]) · Q_tft[q, t] + (1 - α(VIX[T_t])) · Q_kronos[q, t]

  base = 0.5, sensitivity = -0.03, center = 20
  → VIX < center (공포지수 낮음) → α 큼 → TFT 가중
  → VIX = center (평상시)         → α = 0.5 (균등)
  → VIX > center (공포지수 높음) → α 작음 → Kronos 가중

α 매핑 (default 파라미터):
  VIX=10 → α=0.80 (clamp), TFT 80% / Kronos 20%   (극도 평온)
  VIX=15 → α=0.65,        TFT 65% / Kronos 35%   (평온)
  VIX=20 → α=0.50,        TFT 50% / Kronos 50%   (평상시 = 균등)
  VIX=25 → α=0.35,        TFT 35% / Kronos 65%   (가벼운 우려)
  VIX=30 → α=0.20 (clamp), TFT 20% / Kronos 80%   (공포)
  VIX=35+ → α=0.20 (clamp), TFT 20% / Kronos 80%   (위기)

"각 날짜별로" 의 의미:
  각 row 의 inference date T 를 derive (T = Date - future_day BDays)
  T 시점의 VIX 로 그 row 의 α 결정
  - production 단일 run: 모든 row 가 같은 T → 같은 α (한 값)
  - backtest 다중 T:    row 별 T 가 달라서 row 별 α 자동 분기

Inputs:
  model/saved/predict_xai_returns.csv             # TFT
  model/saved/kronos/predict_kronos_returns.csv   # Kronos

Outputs:
  model/saved/ensemble/predict_ensemble_returns.csv          # long form
  model/saved/ensemble/xai/raw/{TICKER}_predictions.csv      # wide form (30 × 19)

출력 schema 는 TFT/Kronos 와 동일 + 메타 컬럼 2개 추가:
  - alpha     : 각 row 에 적용된 α 값 (투명성 / 분석용)
  - vix_at_T  : 각 row 의 inference date 시점 VIX close

다운스트림 호환:
  kupiec.py / backtest.py 는 분위수 컬럼만 사용하므로 메타 컬럼 추가 영향 없음.

사용:
  python ensemble.py                              # default dynamic α (VIX-based)
  python ensemble.py --alpha 0.5                  # static α override (실험/비교용)
  python ensemble.py --alpha-base 0.55            # 평상시 TFT 쪽으로 살짝 bias
  python ensemble.py --alpha-sensitivity -0.04    # VIX 민감도 강화
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# =============================================================
# 설정
# =============================================================
TFT_CSV = Path("model/saved/predict_xai_returns.csv")
KRONOS_CSV = Path("model/saved/kronos/predict_kronos_returns.csv")
OUTPUT_DIR = Path("model/saved/ensemble")

# 분위수 (TFT/Kronos 동일)
QUANTILES = [round(0.05 * i, 2) for i in range(1, 20)]
QUANTILE_COLS = [f"Q{q}_return_5d" for q in QUANTILES]

MERGE_KEYS = ["group_id", "future_day", "Date"]

# -------------------------------------------------------------
# Dynamic α 공식 파라미터 (CLI 로 override 가능)
# α = ALPHA_BASE + ALPHA_SENSITIVITY · (VIX - VIX_CENTER), clamp [ALPHA_MIN, ALPHA_MAX]
# -------------------------------------------------------------
ALPHA_BASE = 0.5             # VIX = center 일 때의 α (평상시 균등)
ALPHA_SENSITIVITY = -0.03    # 음수 → VIX↑ 시 α↓ (Kronos 가중 증가)
VIX_CENTER = 20.0            # VIX 중립 기준 (역사적 long-term 평균 근처)
ALPHA_MIN = 0.2              # α 하한 → Kronos 가중 상한 80%
ALPHA_MAX = 0.8              # α 상한 → TFT 가중 상한 80%

VIX_FETCH_LOOKBACK_DAYS = 10  # yfinance fetch 시 buffer (휴장일 대응)
FALLBACK_ALPHA = 0.5          # VIX fetch 실패 시 fallback


# =============================================================
# Dynamic α 계산
# =============================================================
def compute_alpha(
    vix: float,
    base: float = ALPHA_BASE,
    sensitivity: float = ALPHA_SENSITIVITY,
    center: float = VIX_CENTER,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> float:
    """
    VIX 기반 동적 α 계산.

    α = base + sensitivity · (VIX - center), clamped to [alpha_min, alpha_max]
    sensitivity < 0 → VIX↑ → α↓ → Kronos 가중 증가 (사용자 합의)
    """
    alpha = base + sensitivity * (vix - center)
    return float(max(alpha_min, min(alpha_max, alpha)))


def compute_inference_dates(df: pd.DataFrame) -> pd.Series:
    """
    각 row 의 inference date T derive: T = Date - future_day BDays

    BDay 산술이라 휴장일/주말 자동 처리.
    production 단일 run: 모든 row 가 같은 T
    backtest 다중 run:    row 별 T 다름
    """
    dates = pd.to_datetime(df["Date"])
    fdays = df["future_day"].astype(int)

    inferences = pd.Series(index=df.index, dtype="datetime64[ns]")
    for f in fdays.unique():
        mask = (fdays == f).to_numpy()
        sub_dates = dates.loc[mask]
        inferences.loc[mask] = (sub_dates - pd.tseries.offsets.BDay(int(f))).values
    return inferences


def fetch_vix_for_dates(unique_dates: pd.DatetimeIndex) -> dict:
    """
    각 unique inference date 의 VIX close 값 fetch.

    Returns:
        {Timestamp(normalized): vix_value}
        해당 일자 또는 직전 거래일의 VIX close 사용 (휴장일 대응).
    """
    if len(unique_dates) == 0:
        return {}

    start = unique_dates.min() - pd.Timedelta(days=VIX_FETCH_LOOKBACK_DAYS)
    end = unique_dates.max() + pd.Timedelta(days=1)

    try:
        vix_data = yf.Ticker("^VIX").history(
            start=start, end=end, auto_adjust=False
        )
        if vix_data.empty:
            print(f"  ⚠️ VIX 데이터 empty ({start.date()} ~ {end.date()})")
            return {}
    except Exception as e:
        print(f"  ⚠️ yfinance VIX fetch 실패: {e}")
        return {}

    vix_close = vix_data["Close"].dropna()
    vix_close.index = pd.to_datetime(
        vix_close.index, utc=True
    ).tz_localize(None).normalize()

    result = {}
    for d in unique_dates:
        d_norm = pd.Timestamp(d).normalize()
        valid = vix_close[vix_close.index <= d_norm]
        if len(valid) > 0:
            result[d_norm] = float(valid.iloc[-1])
    return result


# =============================================================
# Load / Save
# =============================================================
def load_predictions(path: Path, source_name: str) -> pd.DataFrame:
    """예측 CSV 로드 + schema 검증."""
    if not path.exists():
        raise FileNotFoundError(f"{source_name} 결과 파일이 없음: {path}")

    df = pd.read_csv(path)

    missing_keys = [k for k in MERGE_KEYS if k not in df.columns]
    if missing_keys:
        raise ValueError(f"{source_name}: 필수 merge 키 누락 → {missing_keys}")

    missing_q = [q for q in QUANTILE_COLS if q not in df.columns]
    if missing_q:
        raise ValueError(
            f"{source_name}: 분위수 컬럼 {len(missing_q)}개 누락 → "
            f"{missing_q[:3]}{'...' if len(missing_q) > 3 else ''}"
        )

    return df


def save_wide_form(df_long: pd.DataFrame, out_dir: Path) -> int:
    """long → 종목별 wide (TFT/Kronos xai/raw 와 동일 형식)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_cols = [f"Q{q}" for q in QUANTILES]
    n_saved = 0

    for ticker, sub in df_long.groupby("group_id"):
        sub = sub.sort_values("Date")
        df_wide = pd.DataFrame(
            sub[QUANTILE_COLS].to_numpy(),
            index=pd.to_datetime(sub["Date"]).dt.date,
            columns=wide_cols,
        )
        df_wide.index.name = "Date"
        df_wide.to_csv(out_dir / f"{ticker}_predictions.csv")
        n_saved += 1

    return n_saved


# =============================================================
# Ensemble
# =============================================================
def make_ensemble(
    tft: pd.DataFrame,
    kronos: pd.DataFrame,
    alpha_static: float | None = None,
    alpha_params: dict | None = None,
) -> pd.DataFrame:
    """
    TFT + Kronos 분위수 가중평균.

    Args:
        alpha_static: 지정하면 모든 row 에 같은 α (백테스트 / 실험용)
                      None 이면 VIX 기반 dynamic α 사용
        alpha_params: dynamic α 공식 파라미터
                      (base, sensitivity, center, alpha_min, alpha_max)
    """
    merged = tft.merge(
        kronos, on=MERGE_KEYS, suffixes=("_tft", "_kronos"), how="inner"
    )

    n_tft, n_kronos, n_merged = len(tft), len(kronos), len(merged)
    print(
        f"[merge] TFT: {n_tft} rows, Kronos: {n_kronos} rows, "
        f"교집합: {n_merged} rows"
    )
    if n_merged == 0:
        raise RuntimeError(
            "Merge 결과 0 rows. (group_id, future_day, Date) 키가 일치하는지, "
            "두 모델의 T(기준일) 가 같은지 확인."
        )
    if n_merged < n_tft or n_merged < n_kronos:
        print(
            f"  ⚠️ 불일치: TFT-only={n_tft - n_merged}, "
            f"Kronos-only={n_kronos - n_merged}"
        )

    # α 결정
    if alpha_static is not None:
        if not 0.0 <= alpha_static <= 1.0:
            raise ValueError(
                f"alpha 는 0.0 ~ 1.0 범위여야 함 (입력: {alpha_static})"
            )
        print(f"\n[α 모드] static α = {alpha_static:.3f} (override)")
        merged["alpha"] = alpha_static
        merged["vix_at_T"] = np.nan
    else:
        print(f"\n[α 모드] dynamic α (VIX-based)")
        params = alpha_params or {}
        print(
            f"  공식: α = {params.get('base', ALPHA_BASE):.3f} "
            f"+ {params.get('sensitivity', ALPHA_SENSITIVITY):.4f} "
            f"× (VIX - {params.get('center', VIX_CENTER):.1f}), "
            f"clamp [{params.get('alpha_min', ALPHA_MIN):.2f}, "
            f"{params.get('alpha_max', ALPHA_MAX):.2f}]"
        )

        inf_dates = compute_inference_dates(merged)
        unique_inf = pd.DatetimeIndex(sorted(inf_dates.dt.normalize().unique()))
        print(
            f"  inference dates: {len(unique_inf)} unique "
            f"({unique_inf.min().date()} ~ {unique_inf.max().date()})"
        )

        vix_dict = fetch_vix_for_dates(unique_inf)

        if not vix_dict:
            print(
                f"  ⚠️ VIX fetch 실패 → fallback static α = {FALLBACK_ALPHA}"
            )
            merged["alpha"] = FALLBACK_ALPHA
            merged["vix_at_T"] = np.nan
        else:
            alpha_map = {
                d: compute_alpha(v, **params) for d, v in vix_dict.items()
            }

            inf_norm = inf_dates.dt.normalize()
            merged["vix_at_T"] = inf_norm.map(vix_dict)
            merged["alpha"] = inf_norm.map(alpha_map)

            n_nan = merged["alpha"].isna().sum()
            if n_nan > 0:
                print(
                    f"  ⚠️ {n_nan} rows: VIX 매핑 실패 → "
                    f"fallback α={FALLBACK_ALPHA}"
                )
                merged["alpha"] = merged["alpha"].fillna(FALLBACK_ALPHA)

            # α 분포 진단
            unique_alphas = sorted(set(alpha_map.values()))
            print(
                f"  α 분포: {min(unique_alphas):.3f} ~ "
                f"{max(unique_alphas):.3f} "
                f"(VIX {min(vix_dict.values()):.2f} ~ "
                f"{max(vix_dict.values()):.2f})"
            )
            # 적은 inference date 면 다 출력 (production single-T 케이스)
            if len(unique_inf) <= 5:
                for d in sorted(unique_inf):
                    d_norm = pd.Timestamp(d).normalize()
                    if d_norm in vix_dict:
                        v = vix_dict[d_norm]
                        a = alpha_map[d_norm]
                        print(
                            f"    T={d_norm.date()}: VIX={v:.2f} → "
                            f"α={a:.3f} (TFT={a:.2f}, Kronos={1-a:.2f})"
                        )

    # 분위수 가중평균
    for col in QUANTILE_COLS:
        merged[col] = (
            merged["alpha"] * merged[f"{col}_tft"]
            + (1 - merged["alpha"]) * merged[f"{col}_kronos"]
        )

    # 출력 컬럼 정리
    out_cols = MERGE_KEYS.copy()
    if "window_end_estimate_tft" in merged.columns:
        merged["window_end_estimate"] = merged["window_end_estimate_tft"]
        out_cols.append("window_end_estimate")
    out_cols += ["alpha", "vix_at_T"]  # 메타 컬럼 (투명성)
    out_cols += QUANTILE_COLS

    return merged[out_cols].copy()


# =============================================================
# Diagnostic
# =============================================================
def print_summary(
    df_ensemble: pd.DataFrame,
    df_tft: pd.DataFrame,
    df_kronos: pd.DataFrame,
) -> None:
    n_tickers = df_ensemble["group_id"].nunique()
    n_rows = len(df_ensemble)

    print(f"\n=== Ensemble 요약 ===")
    print(f"  종목 수: {n_tickers}, 행 수: {n_rows}")

    # α 분포
    alpha_vals = df_ensemble["alpha"]
    print(f"\n  α (TFT 가중치) 분포:")
    print(f"    unique values: {alpha_vals.nunique()}")
    print(
        f"    범위: {alpha_vals.min():.3f} ~ {alpha_vals.max():.3f}, "
        f"평균: {alpha_vals.mean():.3f}"
    )

    # VIX 분포 (dynamic 모드인 경우)
    if not df_ensemble["vix_at_T"].isna().all():
        vix = df_ensemble["vix_at_T"]
        print(f"\n  VIX[T] 분포:")
        print(
            f"    범위: {vix.min():.2f} ~ {vix.max():.2f}, "
            f"평균: {vix.mean():.2f}"
        )

    # 모델별 Q0.5 비교
    median_col = "Q0.5_return_5d"
    if median_col in df_tft.columns and median_col in df_kronos.columns:
        tft_med = df_tft[median_col].mean()
        kronos_med = df_kronos[median_col].mean()
        ens_med = df_ensemble[median_col].mean()
        print(f"\n  전체 평균 Q0.5 (5일 누적 수익률):")
        print(f"    TFT      : {tft_med*100:+.3f}%")
        print(f"    Kronos   : {kronos_med*100:+.3f}%")
        print(f"    Ensemble : {ens_med*100:+.3f}%")

    # 모델별 Q0.05 비교 (하방 시나리오)
    q05_col = "Q0.05_return_5d"
    if q05_col in df_tft.columns and q05_col in df_kronos.columns:
        tft_q05 = df_tft[q05_col].mean()
        kronos_q05 = df_kronos[q05_col].mean()
        ens_q05 = df_ensemble[q05_col].mean()
        print(f"\n  전체 평균 Q0.05 (하위 5% 비관 시나리오):")
        print(f"    TFT      : {tft_q05*100:+.3f}%")
        print(f"    Kronos   : {kronos_q05*100:+.3f}%")
        print(f"    Ensemble : {ens_q05*100:+.3f}%")

    # 분위수 단조성
    q_means = df_ensemble[QUANTILE_COLS].mean()
    non_mono = sum(
        1 for i in range(len(QUANTILE_COLS) - 1)
        if q_means.iloc[i] > q_means.iloc[i + 1]
    )
    if non_mono == 0:
        print(f"\n  ✓ 분위수 단조성 OK")
    else:
        print(f"\n  ⚠️ 분위수 비단조: {non_mono} 인접쌍")

    # 종목별 Q0.5 평균
    if median_col in df_ensemble.columns:
        per_t = (
            df_ensemble.groupby("group_id")[median_col].mean().sort_values()
        )
        print(f"\n  종목별 Q0.5 평균:")
        print(f"    하위 5: " + ", ".join(
            f"{t} ({v*100:+.2f}%)" for t, v in per_t.head(5).items()
        ))
        print(f"    상위 5: " + ", ".join(
            f"{t} ({v*100:+.2f}%)" for t, v in per_t.tail(5)[::-1].items()
        ))


# =============================================================
# Main
# =============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TFT + Kronos VIX-aware dynamic ensemble"
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="static α override (0.0~1.0). 미지정 시 VIX 기반 dynamic α 사용"
    )
    parser.add_argument(
        "--alpha-base", type=float, default=ALPHA_BASE,
        help=f"dynamic α base (VIX=center 일 때 α, default {ALPHA_BASE})"
    )
    parser.add_argument(
        "--alpha-sensitivity", type=float, default=ALPHA_SENSITIVITY,
        help=f"dynamic α slope (default {ALPHA_SENSITIVITY}, 음수=VIX↑→Kronos↑)"
    )
    parser.add_argument(
        "--vix-center", type=float, default=VIX_CENTER,
        help=f"VIX 중립 기준 (default {VIX_CENTER})"
    )
    parser.add_argument(
        "--alpha-min", type=float, default=ALPHA_MIN,
        help=f"α 하한 (default {ALPHA_MIN})"
    )
    parser.add_argument(
        "--alpha-max", type=float, default=ALPHA_MAX,
        help=f"α 상한 (default {ALPHA_MAX})"
    )
    parser.add_argument(
        "--tft", type=Path, default=TFT_CSV,
        help=f"TFT 결과 CSV (default: {TFT_CSV})"
    )
    parser.add_argument(
        "--kronos", type=Path, default=KRONOS_CSV,
        help=f"Kronos 결과 CSV (default: {KRONOS_CSV})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"출력 디렉토리 (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()

    # Load
    print(f"[load] TFT:    {args.tft}")
    df_tft = load_predictions(args.tft, "TFT")
    print(f"  → {len(df_tft)} rows, {df_tft['group_id'].nunique()} 종목")

    print(f"[load] Kronos: {args.kronos}")
    df_kronos = load_predictions(args.kronos, "Kronos")
    print(f"  → {len(df_kronos)} rows, {df_kronos['group_id'].nunique()} 종목")

    # Ensemble
    alpha_params = {
        "base": args.alpha_base,
        "sensitivity": args.alpha_sensitivity,
        "center": args.vix_center,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
    }
    df_ensemble = make_ensemble(
        df_tft, df_kronos,
        alpha_static=args.alpha,
        alpha_params=alpha_params,
    )

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.output_dir / "predict_ensemble_returns.csv"
    df_ensemble.to_csv(long_path, index=False)
    print(f"\n[save] long form: {long_path}")

    raw_dir = args.output_dir / "xai" / "raw"
    n_saved = save_wide_form(df_ensemble, raw_dir)
    print(f"[save] wide form: {raw_dir}/<TICKER>_predictions.csv ({n_saved} files)")

    print_summary(df_ensemble, df_tft, df_kronos)
    print("\n완료.")


if __name__ == "__main__":
    main()
