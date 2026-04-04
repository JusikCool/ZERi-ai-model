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
                pred_len = y_true.shape[1]
                groups_expanded = np.repeat(group_names, pred_len)

                all_y_true.append(y_true.ravel())
                all_y_pred.append(y_pred.reshape(-1, y_pred.shape[-1]))
                all_groups.append(groups_expanded)

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
