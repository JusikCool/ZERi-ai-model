from pathlib import Path

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from model.m4.dataset import (
    GROUP_ID,
    TARGET,
    TIME_IDX,
    build_crisis_aware_dataloaders,
    build_dataloaders,
    build_dataset,
    load_data,
)

from .sectors import TICKERS_BY_SECTOR, get_tickers

SECTOR_DATA_PATH_M4 = Path("data/raw/tft_processed_panel_v5.csv")
DEFAULT_ENCODER_LEN = 60
DEFAULT_PREDICTION_LEN = 10
DEFAULT_VAL_RATIO = 0.2


def filter_sector(df: pd.DataFrame, sector_id: str) -> pd.DataFrame:
    tickers = get_tickers(sector_id)
    sub = df[df[GROUP_ID].isin(tickers)].copy()
    if sub.empty:
        raise ValueError(
            f"섹터 '{sector_id}' 종목 ({tickers}) 가 데이터에 존재하지 않습니다."
        )
    sub = sub.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    sub[TIME_IDX] = sub.groupby(GROUP_ID).cumcount()
    return sub


def load_sector_data(
    sector_id: str, data_path: Path = SECTOR_DATA_PATH_M4
) -> pd.DataFrame:
    df = load_data(data_path)
    return filter_sector(df, sector_id)


def build_sector_dataset(
    sector_id: str,
    max_encoder_length: int = DEFAULT_ENCODER_LEN,
    max_prediction_length: int = DEFAULT_PREDICTION_LEN,
    val_ratio: float = DEFAULT_VAL_RATIO,
    data_path: Path = SECTOR_DATA_PATH_M4,
) -> tuple[pd.DataFrame, TimeSeriesDataSet, TimeSeriesDataSet]:
    df = load_sector_data(sector_id, data_path=data_path)
    train_ds, val_ds = build_dataset(
        df,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        val_ratio=val_ratio,
    )
    return df, train_ds, val_ds


def build_sector_dataloaders(
    train_ds: TimeSeriesDataSet,
    val_ds: TimeSeriesDataSet,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    return build_dataloaders(
        train_ds, val_ds, batch_size=batch_size, num_workers=num_workers
    )


def build_sector_crisis_dataloaders(
    train_ds: TimeSeriesDataSet,
    val_ds: TimeSeriesDataSet,
    df: pd.DataFrame,
    encoder_length: int,
    batch_size: int = 64,
    num_workers: int = 0,
    vix_threshold: float = 25.0,
    high_vix_multiplier: float = 3.0,
) -> tuple[DataLoader, DataLoader]:
    return build_crisis_aware_dataloaders(
        train_ds, val_ds, df,
        encoder_length=encoder_length,
        batch_size=batch_size,
        num_workers=num_workers,
        vix_threshold=vix_threshold,
        high_vix_multiplier=high_vix_multiplier,
    )
