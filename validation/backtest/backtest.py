import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from model.m3_full_model.dataset import (
    build_dataset,
    get_vix_stats,
    GROUP_ID,
    TIME_IDX,
)
from model.m3_full_model.model import M3FullModel
from validation.kupiec.kupiec import run_validation_by_group

CHECKPOINT_DIR = Path("model/saved")
COVID_START = "2020-01-01"
COVID_END = "2021-01-01"
UKRAINE_INFLATION_START = "2022-02-01"
UKRAINE_INFLATION_END = "2022-07-01"
TRUMP_TARIFF_START = "2025-04-01"
TRUMP_TARIFF_END = "2025-05-31"


def rolling_window_backtest(
    df: pd.DataFrame,
    config: dict,
    n_splits: int = 5,
) -> pd.DataFrame:
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

    model = M3FullModel.from_dataset(
        dataset=train_ds,
        learning_rate=model_cfg["learning_rate"],
        hidden_size=model_cfg["hidden_size"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
        vix_mean=vix_mean,
        vix_std=vix_std,
    )
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
    report = run_validation_by_group(
        y_true_all,
        y_pred_all,
        groups_all,
        quantiles=model_cfg["quantiles"],
        vr_threshold=val_cfg["violation_rate_threshold"],
        pvalue_threshold=val_cfg["kupiec_pvalue_threshold"],
    )
    return report

def period_backtest(
    df: pd.DataFrame,
    config: dict,
    start: str,
    end: str,
) -> pd.DataFrame:
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

    model = M3FullModel.from_dataset(
        dataset=train_ds,
        learning_rate=model_cfg["learning_rate"],
        hidden_size=model_cfg["hidden_size"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
        vix_mean=vix_mean,
        vix_std=vix_std,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    group_mapping = {i: g for i, g in enumerate(sorted(df[GROUP_ID].unique()))}

    covid_time_idx = df[
        (df["Date"] >= start) & (df["Date"] < end)
    ][TIME_IDX].unique()

    if len(covid_time_idx) == 0:
        raise ValueError(f"COVID 구간({start}~{end}) 데이터 없음")

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
    report = run_validation_by_group(
        y_true_all,
        y_pred_all,
        groups_all,
        quantiles=model_cfg["quantiles"],
        vr_threshold=val_cfg["violation_rate_threshold"],
        pvalue_threshold=val_cfg["kupiec_pvalue_threshold"],
    )
    report["period"] = f"{start} ~ {end}"
    return report
    
def covid_backtest(
    df: pd.DataFrame,
    config: dict,
    start: str = COVID_START,
    end: str = COVID_END,
) -> pd.DataFrame:
    return period_backtest(df, config, start, end)

def ukraine_inflation_backtest(
    df: pd.DataFrame,
    config: dict,
    start: str = UKRAINE_INFLATION_START,
    end: str = UKRAINE_INFLATION_END,
) -> pd.DataFrame:
    return period_backtest(df, config, start, end)
 
 
def trump_tariff_backtest(
    df: pd.DataFrame,
    config: dict,
    start: str = TRUMP_TARIFF_START,
    end: str = TRUMP_TARIFF_END,
) -> pd.DataFrame:
    return period_backtest(df, config, start, end)