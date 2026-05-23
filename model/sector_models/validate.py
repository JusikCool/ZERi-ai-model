from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from model.m3_full_model.dataset import GROUP_ID, TIME_IDX
from validation.kupiec.kupiec import run_validation, run_validation_by_group

from .dataset import DEFAULT_ENCODER_LEN, DEFAULT_VAL_RATIO
from .train import QUANTILES, load_trained_model, sector_dir

VR_THRESHOLD = 0.05
PVALUE_THRESHOLD = 0.05
EVAL_BATCH_SIZE = 128


def _collect_predictions(
    model,
    train_ds: TimeSeriesDataSet,
    df: pd.DataFrame,
    batch_size: int = EVAL_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cutoff = int(df[TIME_IDX].max() * (1 - DEFAULT_VAL_RATIO))
    eval_df = df[df[TIME_IDX] > cutoff - DEFAULT_ENCODER_LEN].copy()

    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds, eval_df, predict=False, stop_randomization=True
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    group_mapping = {i: g for i, g in enumerate(sorted(df[GROUP_ID].unique()))}

    y_trues, y_preds, groups_all = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_true = y[0].numpy()
            y_pred = model.predict(x).numpy()
            group_ints = x["groups"][:, 0].numpy()
            group_names = np.array([group_mapping[int(g)] for g in group_ints])
            y_trues.append(y_true[:, -1])
            y_preds.append(y_pred[:, -1, :])
            groups_all.append(group_names)

    return (
        np.concatenate(y_trues),
        np.concatenate(y_preds, axis=0),
        np.concatenate(groups_all),
    )


def evaluate_sector(
    sector_id: str,
    vr_threshold: float = VR_THRESHOLD,
    pvalue_threshold: float = PVALUE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, df, train_ds, _ = load_trained_model(sector_id)
    y_true, y_pred, groups = _collect_predictions(model, train_ds, df)

    per_ticker = run_validation_by_group(
        y_true,
        y_pred,
        groups,
        quantiles=QUANTILES,
        vr_threshold=vr_threshold,
        pvalue_threshold=pvalue_threshold,
    )
    per_ticker.insert(0, "sector", sector_id)

    sector_level = run_validation(
        y_true,
        y_pred,
        quantiles=QUANTILES,
        vr_threshold=vr_threshold,
        pvalue_threshold=pvalue_threshold,
    )
    sector_level.insert(0, "sector", sector_id)

    out_dir = sector_dir(sector_id)
    per_ticker.to_csv(out_dir / "validation_by_ticker.csv", index=False)
    sector_level.to_csv(out_dir / "validation_sector.csv", index=False)

    return per_ticker, sector_level
