from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from model.m3_full_model.dataset import GROUP_ID, TIME_IDX
from model.sector_models.sectors import SECTOR_IDS, SECTOR_MAP, get_sector
from model.sector_models.train import (
    DEFAULT_ENCODER_LEN,
    QUANTILES,
    load_trained_model,
)

DEFAULT_BATCH_SIZE = 64


def _denormalize_quantiles(
    y_pred: torch.Tensor, target_scale
) -> torch.Tensor:
    if target_scale is None:
        return y_pred
    if isinstance(target_scale, (list, tuple)):
        target_scale = target_scale[0]
    ts = target_scale.to(y_pred.device).float()
    if ts.dim() == 2:
        center = ts[..., 0].view(-1, 1, 1)
        scale = ts[..., 1].view(-1, 1, 1)
    else:
        center = ts[..., 0:1]
        scale = ts[..., 1:2]
    return y_pred * scale + center


@lru_cache(maxsize=len(SECTOR_IDS))
def _cached_sector_model(sector_id: str):
    return load_trained_model(sector_id)


def load_model(ticker: str):
    sector = get_sector(ticker)
    return _cached_sector_model(sector)


def _prepare_eval_df(
    df: pd.DataFrame, ticker: str, encoder_len: int
) -> pd.DataFrame:
    sub = df[df[GROUP_ID] == ticker].copy()
    if sub.empty:
        raise ValueError(f"입력 데이터에 {ticker} 데이터가 없습니다.")
    sub = sub.sort_values(TIME_IDX).reset_index(drop=True)
    sub[TIME_IDX] = sub.groupby(GROUP_ID).cumcount()
    return sub


def predict(
    ticker: str,
    data: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    return_original_scale: bool = True,
) -> dict:
    sector = get_sector(ticker)
    model, _, train_ds, _ = _cached_sector_model(sector)

    eval_df = _prepare_eval_df(data, ticker, DEFAULT_ENCODER_LEN)
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds, eval_df, predict=True, stop_randomization=True
    )
    loader = val_ds.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    preds = []
    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            y_pred = model.predict(x)
            if return_original_scale:
                y_pred = _denormalize_quantiles(y_pred, x.get("target_scale"))
            preds.append(y_pred)
    quantile_tensor = torch.cat(preds, dim=0).cpu().numpy()

    return {
        "ticker": ticker,
        "sector": sector,
        "quantiles": QUANTILES,
        "predictions": quantile_tensor,
        "scale": "original_return" if return_original_scale else "normalized",
    }


def predict_batch(
    tickers: list[str],
    data: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    return_original_scale: bool = True,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for ticker in tickers:
        results[ticker] = predict(
            ticker, data,
            batch_size=batch_size,
            return_original_scale=return_original_scale,
        )
    return results


def list_sectors() -> list[str]:
    return list(SECTOR_IDS)


def list_tickers() -> list[str]:
    return sorted(SECTOR_MAP.keys())
