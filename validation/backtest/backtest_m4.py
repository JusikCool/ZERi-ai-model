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
BEST_PARAMS_PATH = CHECKPOINT_DIR / "best_params.json"
COVID_START = "2020-01-01"
COVID_END = "2021-01-01"
UKRAINE_INFLATION_START = "2022-02-01"
UKRAINE_INFLATION_END = "2022-07-01"
TRUMP_TARIFF_START = "2025-04-01"
TRUMP_TARIFF_END = "2025-05-31"


def _load_best_params() -> dict:
    """학습 시 저장된 TFT best params 로드. 없으면 빈 dict."""
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _build_model_kwargs(model_cfg: dict, vix_mean: float, vix_std: float) -> dict:
    """
    TFT용 from_dataset kwargs 빌더.
    우선순위: best_params.json > config["model"]
    학습 시 사용한 하이퍼파라미터와 동일한 구조로 모델을 만들어야 체크포인트 로드 가능.
    """
    best = _load_best_params()

    def pick(key, default=None):
        if key in best:
            return best[key]
        return model_cfg.get(key, default)

    kwargs = dict(
        learning_rate=pick("learning_rate", model_cfg["learning_rate"]),
        hidden_size=pick("hidden_size", model_cfg["hidden_size"]),
        attention_head_size=pick("attention_head_size", model_cfg["attention_head_size"]),
        dropout=pick("dropout", model_cfg["dropout"]),
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
        vix_mean=vix_mean,
        vix_std=vix_std,
    )

    # hidden_continuous_size 같은 추가 파라미터도 best_params에 있으면 반영
    if "hidden_continuous_size" in best:
        kwargs["hidden_continuous_size"] = best["hidden_continuous_size"]
    elif "hidden_continuous_size" in model_cfg:
        kwargs["hidden_continuous_size"] = model_cfg["hidden_continuous_size"]

    # AdaptivePinballLoss 가중치 best_params에 있으면 반영
    for opt_key in ("alpha_down", "beta_down", "alpha_up", "beta_up", "crossing_weight"):
        if opt_key in best:
            kwargs[opt_key] = best[opt_key]

    return kwargs


# =============================================================
# 내부 helper: 기존 보고서 + extended 보고서 동시 생성
# =============================================================
def _build_validation_reports(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    groups_all: np.ndarray,
    model_cfg: dict,
    val_cfg: dict,
    title: str = "",
) -> dict:
    """
    동일 (y_true, y_pred, groups) 에 대해:
      - basic_report : 기존 Kupiec UC test (호환성 유지)
      - extended_report : UC + CC + DQ test (Buczyński & Chlebus 2024 framework)

    extended_report 가 basic_report 의 superset이므로, 분석/저장 용도에 따라 선택 사용.
    """
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

    # 콘솔 요약 출력 (Phase 1 효과 즉시 가시화)
    if title:
        print_extended_summary(extended_report, title=title)

    return {"basic": basic_report, "extended": extended_report}


def rolling_window_backtest(
    df: pd.DataFrame,
    config: dict,
    n_splits: int = 5,
    return_extended: bool = False,
) -> pd.DataFrame:
    """
    Rolling-window 백테스트. 기본 반환은 기존과 동일 (basic Kupiec report).

    Args:
        return_extended: True 면 {"basic", "extended"} dict 반환,
                         False 면 기존 호환성 유지 (basic DataFrame 만).
                         두 경우 모두 콘솔에 extended summary 출력.
    """
    data_cfg = config["data"]
    model_cfg = config["model"]

    ckpts = glob.glob(str(CHECKPOINT_DIR / "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("model/saved/ 에 체크포인트가 없습니다.")
    ckpt_path = sorted(ckpts)[-1]

    vix_mean, vix_std = get_vix_stats(df)

    train_ds, _ = build_dataset(
        df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )

    model_kwargs = _build_model_kwargs(model_cfg, vix_mean, vix_std)
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
        title=f"Rolling-window Backtest ({n_splits} splits)",
    )

    return reports if return_extended else reports["basic"]


def period_backtest(
    df: pd.DataFrame,
    config: dict,
    start: str,
    end: str,
    return_extended: bool = False,
) -> pd.DataFrame:
    """
    구간 (start ~ end) 백테스트. 기존 호환성 유지 + extended 옵션.
    """
    data_cfg = config["data"]
    model_cfg = config["model"]

    ckpts = glob.glob(str(CHECKPOINT_DIR / "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("model/saved/ 에 체크포인트가 없습니다.")
    ckpt_path = sorted(ckpts)[-1]

    vix_mean, vix_std = get_vix_stats(df)

    train_ds, _ = build_dataset(
        df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )

    model_kwargs = _build_model_kwargs(model_cfg, vix_mean, vix_std)
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
        title=f"Period Backtest ({start} ~ {end})",
    )

    reports["basic"]["period"] = f"{start} ~ {end}"
    reports["extended"]["period"] = f"{start} ~ {end}"

    return reports if return_extended else reports["basic"]


def covid_backtest(
    df: pd.DataFrame,
    config: dict,
    start: str = COVID_START,
    end: str = COVID_END,
    return_extended: bool = False,
) -> pd.DataFrame:
    return period_backtest(df, config, start, end, return_extended=return_extended)


def ukraine_inflation_backtest(
    df: pd.DataFrame,
    config: dict,
    start: str = UKRAINE_INFLATION_START,
    end: str = UKRAINE_INFLATION_END,
    return_extended: bool = False,
) -> pd.DataFrame:
    return period_backtest(df, config, start, end, return_extended=return_extended)


def trump_tariff_backtest(
    df: pd.DataFrame,
    config: dict,
    start: str = TRUMP_TARIFF_START,
    end: str = TRUMP_TARIFF_END,
    return_extended: bool = False,
) -> pd.DataFrame:
    return period_backtest(df, config, start, end, return_extended=return_extended)
