import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from model.m4.dataset import (
    build_dataset,
    get_vix_stats,
    GROUP_ID,
    TIME_IDX,
)
from model.m4.model import M4FullModel
from validation.kupiec.kupiec import run_validation_by_group
from validation.kupiec.coverage_tests import (
    run_validation_by_group_extended,
    print_extended_summary,
)

CHECKPOINT_DIR = Path("model/saved")
COVID_START = "2020-01-01"
COVID_END = "2021-01-01"
UKRAINE_INFLATION_START = "2022-02-01"
UKRAINE_INFLATION_END = "2022-07-01"
TRUMP_TARIFF_START = "2025-04-01"
TRUMP_TARIFF_END = "2025-05-31"

# ===== M4: mode 별 파일 경로 =====
VOL_GROUP_MAP_PATH = Path("data/raw/vol_group_map.json")
GROUP_LABELS = ["low_vol", "mid_vol", "high_vol"]


def _best_params_path(mode: str = None) -> Path:
    if mode is None:
        return CHECKPOINT_DIR / "best_params.json"
    return CHECKPOINT_DIR / f"best_params_{mode}.json"


def _load_best_params(mode: str = None) -> dict:
    """mode 별 best_params 로드."""
    path = _best_params_path(mode)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            best = json.load(f)
        print(f"[backtest] best_params 로드: {path}")
        return best
    print(f"[backtest] ⚠️ {path} 없음 → config default 사용")
    return {}


def _load_vol_group_map() -> dict:
    if VOL_GROUP_MAP_PATH.exists():
        with open(VOL_GROUP_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def _build_ticker_to_vol_group_idx(
    panel_df: pd.DataFrame, vol_group_map: dict
) -> torch.Tensor:
    """run_full_pipeline_m4 와 동일 로직 — alphabetic ticker order 기준."""
    if vol_group_map is None:
        return None
    panel_tickers_sorted = sorted(panel_df["group_id"].unique())
    label_to_idx = {l: i for i, l in enumerate(GROUP_LABELS)}
    default_label = GROUP_LABELS[len(GROUP_LABELS) // 2]

    vg_list = []
    for ticker in panel_tickers_sorted:
        vg_str = vol_group_map.get(ticker, default_label)
        if vg_str not in label_to_idx:
            vg_str = default_label
        vg_list.append(label_to_idx[vg_str])
    return torch.tensor(vg_list, dtype=torch.long)


def _find_checkpoint(mode: str = None) -> str:
    """mode 별 checkpoint 패턴 검색."""
    if mode is not None:
        pattern = str(CHECKPOINT_DIR / f"{mode}_best_*.ckpt")
        ckpts = glob.glob(pattern)
        if ckpts:
            return sorted(ckpts)[-1]
        print(f"[backtest] ⚠️ '{pattern}' 매치 없음 → 전체 검색 fallback")
    ckpts = glob.glob(str(CHECKPOINT_DIR / "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"{CHECKPOINT_DIR} 에 체크포인트 없음")
    return sorted(ckpts)[-1]


def _build_model_kwargs(
    model_cfg: dict,
    vix_mean: float,
    vix_std: float,
    mode: str = None,
    df: pd.DataFrame = None,
    vol_group_map: dict = None,
) -> dict:
    """
    M4FullModel.from_dataset 의 kwargs 생성. mode 별 분기.
    우선순위: best_params_{mode}.json > config["model"]
    """
    best = _load_best_params(mode)

    def pick(key, default=None):
        if key in best:
            return best[key]
        return model_cfg.get(key, default)

    kwargs = dict(
        learning_rate=pick("learning_rate", model_cfg["learning_rate"]),
        hidden_size=pick("hidden_size", model_cfg["hidden_size"]),
        attention_head_size=pick(
            "attention_head_size", model_cfg["attention_head_size"]
        ),
        dropout=pick("dropout", model_cfg["dropout"]),
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
        vix_mean=vix_mean,
        vix_std=vix_std,
        use_garch_sigma=True,
        group_labels=GROUP_LABELS,
    )

    if "hidden_continuous_size" in best:
        kwargs["hidden_continuous_size"] = best["hidden_continuous_size"]
    elif "hidden_continuous_size" in model_cfg:
        kwargs["hidden_continuous_size"] = model_cfg["hidden_continuous_size"]

    if "crossing_weight" in best:
        kwargs["crossing_weight"] = best["crossing_weight"]

    # mode 별 α/β
    if mode == "m4_combined":
        try:
            kwargs["alpha_down_by_group"] = {g: best[f"alpha_down_{g}"] for g in GROUP_LABELS}
            kwargs["beta_down_by_group"] = {g: best[f"beta_down_{g}"] for g in GROUP_LABELS}
            kwargs["alpha_up_by_group"] = {g: best[f"alpha_up_{g}"] for g in GROUP_LABELS}
            kwargs["beta_up_by_group"] = {g: best[f"beta_up_{g}"] for g in GROUP_LABELS}
        except KeyError as e:
            raise KeyError(
                f"m4_combined best_params 에 group 별 α/β key 누락: {e}. "
                f"best_params_m4_combined.json 을 확인하세요."
            )
        kwargs["ticker_to_vol_group_idx"] = _build_ticker_to_vol_group_idx(
            df, vol_group_map
        )
    else:
        # m4_garch (또는 mode=None 기본)
        for k in ("alpha_down", "beta_down", "alpha_up", "beta_up"):
            if k in best:
                kwargs[k] = best[k]
        kwargs["ticker_to_vol_group_idx"] = None

    return kwargs


def _build_validation_reports(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    groups_all: np.ndarray,
    model_cfg: dict,
    val_cfg: dict,
    title: str = "",
) -> dict:
    basic_report = run_validation_by_group(
        y_true_all,
        y_pred_all,
        groups_all,
        quantiles=model_cfg["quantiles"],
        vr_threshold=val_cfg["violation_rate_threshold"],
        pvalue_threshold=val_cfg["kupiec_pvalue_threshold"],
    )

    extended_report = run_validation_by_group_extended(
        y_true_all,
        y_pred_all,
        groups_all,
        quantiles=model_cfg["quantiles"],
        vr_threshold=val_cfg["violation_rate_threshold"],
        pvalue_threshold=val_cfg["kupiec_pvalue_threshold"],
        dq_n_lags=val_cfg.get("dq_n_lags", 4),
    )

    if title:
        print_extended_summary(extended_report, title=title)

    return {"basic": basic_report, "extended": extended_report}


def rolling_window_backtest(
    df: pd.DataFrame,
    config: dict,
    n_splits: int = 5,
    return_extended: bool = False,
    mode: str = None,
) -> pd.DataFrame:
    """
    Rolling-window 백테스트.

    Args:
        mode: 'm4_garch' / 'm4_combined' / None
              모드별 checkpoint + best_params + vol_group_map 자동 로드
    """
    data_cfg = config["data"]
    model_cfg = config["model"]

    print(f"\n[backtest] mode={mode}")
    ckpt_path = _find_checkpoint(mode)
    print(f"[backtest] checkpoint: {ckpt_path}")

    vol_group_map = _load_vol_group_map() if mode == "m4_combined" else None
    if mode == "m4_combined":
        if vol_group_map is None:
            raise ValueError(f"mode='m4_combined' 인데 {VOL_GROUP_MAP_PATH} 없음")
        print(f"[backtest] vol_group_map 로드: {len(vol_group_map)} tickers")

    vix_mean, vix_std = get_vix_stats(df)
    train_ds, _ = build_dataset(
        df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )

    model_kwargs = _build_model_kwargs(
        model_cfg, vix_mean, vix_std, mode=mode, df=df, vol_group_map=vol_group_map
    )
    model = M4FullModel.from_dataset(dataset=train_ds, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    group_mapping = {i: g for i, g in enumerate(sorted(df[GROUP_ID].unique()))}

    max_time = df[TIME_IDX].max()
    min_time = df[TIME_IDX].min()
    fold_size = (max_time - min_time) // (n_splits + 1)

    all_y_true = []
    all_y_pred = []
    all_groups = []

    for i in range(n_splits):
        train_end = min_time + fold_size * (i + 1)
        val_end = train_end + fold_size

        fold_df = df[
            (df[TIME_IDX] > train_end - data_cfg["window_size"])
            & (df[TIME_IDX] <= val_end)
        ].copy()

        val_ds = TimeSeriesDataSet.from_dataset(
            train_ds,
            fold_df,
            predict=False,
            stop_randomization=True,
        )
        val_loader = val_ds.to_dataloader(
            train=False,
            batch_size=model_cfg["batch_size"] * 2,
            num_workers=0,
        )

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                y_true = y[0].numpy()
                y_pred = model.predict(x).numpy()

                group_ints = x["groups"][:, 0].numpy()
                group_names = np.array([group_mapping[g] for g in group_ints])

                all_y_true.append(y_true[:, -1])
                all_y_pred.append(y_pred[:, -1, :])
                all_groups.append(group_names)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    groups_all = np.concatenate(all_groups)

    val_cfg = config["validation"]
    reports = _build_validation_reports(
        y_true_all, y_pred_all, groups_all,
        model_cfg, val_cfg,
        title=f"Rolling-window Backtest (mode={mode}, {n_splits} splits)",
    )

    return reports if return_extended else reports["basic"]


def period_backtest(
    df: pd.DataFrame,
    config: dict,
    start: str,
    end: str,
    return_extended: bool = False,
    mode: str = None,
) -> pd.DataFrame:
    data_cfg = config["data"]
    model_cfg = config["model"]

    print(f"\n[backtest] mode={mode}, period={start}~{end}")
    ckpt_path = _find_checkpoint(mode)

    vol_group_map = _load_vol_group_map() if mode == "m4_combined" else None
    if mode == "m4_combined" and vol_group_map is None:
        raise ValueError(f"mode='m4_combined' 인데 {VOL_GROUP_MAP_PATH} 없음")

    vix_mean, vix_std = get_vix_stats(df)
    train_ds, _ = build_dataset(
        df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )

    model_kwargs = _build_model_kwargs(
        model_cfg, vix_mean, vix_std, mode=mode, df=df, vol_group_map=vol_group_map
    )
    model = M4FullModel.from_dataset(dataset=train_ds, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    group_mapping = {i: g for i, g in enumerate(sorted(df[GROUP_ID].unique()))}

    covid_time_idx = df[
        (df["Date"] >= start) & (df["Date"] < end)
    ][TIME_IDX].unique()

    if len(covid_time_idx) == 0:
        raise ValueError(f"구간({start}~{end}) 데이터 없음")

    t_min = int(covid_time_idx.min())
    t_max = int(covid_time_idx.max())

    fold_df = df[
        (df[TIME_IDX] >= t_min - data_cfg["window_size"])
        & (df[TIME_IDX] <= t_max)
    ].copy()

    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        fold_df,
        predict=False,
        stop_randomization=True,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=model_cfg["batch_size"] * 2,
        num_workers=0,
    )

    all_y_true = []
    all_y_pred = []
    all_groups = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_true = y[0].numpy()
            y_pred = model.predict(x).numpy()

            group_ints = x["groups"][:, 0].numpy()
            group_names = np.array([group_mapping[g] for g in group_ints])

            all_y_true.append(y_true[:, -1])
            all_y_pred.append(y_pred[:, -1, :])
            all_groups.append(group_names)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    groups_all = np.concatenate(all_groups)

    val_cfg = config["validation"]
    reports = _build_validation_reports(
        y_true_all, y_pred_all, groups_all,
        model_cfg, val_cfg,
        title=f"Period Backtest (mode={mode}, {start} ~ {end})",
    )

    reports["basic"]["period"] = f"{start} ~ {end}"
    reports["extended"]["period"] = f"{start} ~ {end}"

    return reports if return_extended else reports["basic"]


def covid_backtest(df, config, start=COVID_START, end=COVID_END, return_extended=False, mode=None):
    return period_backtest(df, config, start, end, return_extended=return_extended, mode=mode)


def ukraine_inflation_backtest(df, config, start=UKRAINE_INFLATION_START, end=UKRAINE_INFLATION_END, return_extended=False, mode=None):
    return period_backtest(df, config, start, end, return_extended=return_extended, mode=mode)


def trump_tariff_backtest(df, config, start=TRUMP_TARIFF_START, end=TRUMP_TARIFF_END, return_extended=False, mode=None):
    return period_backtest(df, config, start, end, return_extended=return_extended, mode=mode)