from pathlib import Path

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader

DATA_PATH = Path("data/raw/tft_processed_panel_v1.csv")

TARGET = "Target_Return_5d"
TIME_IDX = "time_idx"
GROUP_ID = "group_id"

TIME_VARYING_KNOWN_CATEGORICALS = ["Month", "Day_of_Week"]

TIME_VARYING_UNKNOWN_REALS = [
    "Open", "High", "Low", "Close", "Volume",
    "Dividends", "Stock Splits",
    "NASDAQ_Close", "VIX_Close",
    "FEDFUNDS", "UNRATE", "DTWEXBGS", "CPIAUCSL", "PCEPI",
    "GDP", "M2SL", "GS10", "T10Y2Y", "PAYEMS", "CSUSHPISA", "INDPRO",
    "RSI_14", "ATR_14", "SMA_20",
    "Returns", "Realized_Vol_20d",
]


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    df[GROUP_ID] = df[GROUP_ID].astype(str)
    df["Month"] = df["Month"].astype(str)
    df["Day_of_Week"] = df["Day_of_Week"].astype(str)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df[TIME_IDX] = df.groupby(GROUP_ID).cumcount()
    return df


def build_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 10,
    val_ratio: float = 0.2,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    cutoff = int(df[TIME_IDX].max() * (1 - val_ratio))

    train_dataset = TimeSeriesDataSet(
        df[df[TIME_IDX] <= cutoff],
        time_idx=TIME_IDX,
        target=TARGET,
        group_ids=[GROUP_ID],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[GROUP_ID],
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(
            groups=[GROUP_ID], transformation=None
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        df[df[TIME_IDX] > cutoff - max_encoder_length],
        predict=True,
        stop_randomization=True,
    )

    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset: TimeSeriesDataSet,
    val_dataset: TimeSeriesDataSet,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_loader = train_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=num_workers
    )
    return train_loader, val_loader
