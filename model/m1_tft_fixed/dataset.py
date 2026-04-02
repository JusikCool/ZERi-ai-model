from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet

try:
    from .config import TFTFixedConfig
    from .data import ensure_processed_panel
except ImportError:  # pragma: no cover - direct script execution
    from config import TFTFixedConfig
    from data import ensure_processed_panel

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - fallback for older environments
    import pytorch_lightning as pl


@dataclass
class DatasetBundle:
    train_dataset: TimeSeriesDataSet
    validation_dataset: TimeSeriesDataSet
    test_dataset: TimeSeriesDataSet
    prediction_dataset: TimeSeriesDataSet


class TFTFixedDataModule(pl.LightningDataModule):
    def __init__(self, config: TFTFixedConfig):
        super().__init__()
        self.config = config
        self.dataframe: Optional[pd.DataFrame] = None
        self.datasets: Optional[DatasetBundle] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.datasets is not None:
            return

        df = load_data(self.config.data_path)
        validate_data(df, self.config)
        self.dataframe = df
        self.datasets = build_dataset(df, self.config)

    def train_dataloader(self):
        self.setup()
        return self.datasets.train_dataset.to_dataloader(
            train=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        self.setup()
        return self.datasets.validation_dataset.to_dataloader(
            train=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        self.setup()
        return self.datasets.test_dataset.to_dataloader(
            train=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self):
        self.setup()
        return self.datasets.prediction_dataset.to_dataloader(
            train=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )


def load_data(data_path: str) -> pd.DataFrame:
    ensure_processed_panel(output_path=Path(data_path))
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)
    df["group_id"] = df["group_id"].astype(str)
    df["Month"] = df["Month"].astype(str)
    df["Day_of_Week"] = df["Day_of_Week"].astype(str)
    df = df.dropna(subset=[config_target_column()]).reset_index(drop=True)
    df["time_idx"] = df.groupby("group_id").cumcount()
    return df


def config_target_column() -> str:
    return TFTFixedConfig().target_column


def validate_data(df: pd.DataFrame, config: TFTFixedConfig) -> None:
    required_columns = (
        config.static_categoricals
        + config.time_varying_known_categoricals
        + config.time_varying_known_reals
        + config.time_varying_unknown_reals
        + [config.target_column, "Date"]
    )
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Prepared panel is missing required columns: {missing}")

    banned_in_features = set(config.excluded_model_columns) & set(
        config.static_categoricals
        + config.time_varying_known_categoricals
        + config.time_varying_known_reals
        + config.time_varying_unknown_reals
    )
    if banned_in_features:
        raise ValueError(
            f"Excluded columns leaked into model features: {sorted(banned_in_features)}"
        )

    numeric_columns = (
        config.time_varying_known_reals
        + config.time_varying_unknown_reals
        + [config.target_column]
    )
    numeric_frame = df[numeric_columns]
    if numeric_frame.isna().any().any():
        summary = numeric_frame.isna().sum()
        summary = summary[summary > 0].to_dict()
        raise ValueError(f"Prepared panel contains NaN values: {summary}")

    numeric_values = numeric_frame.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Prepared panel contains inf or -inf values.")

    group_lengths = df.groupby("group_id")["time_idx"].nunique()
    min_required_length = (
        config.max_encoder_length + config.max_prediction_length
    )
    too_short = group_lengths[group_lengths < min_required_length]
    if not too_short.empty:
        raise ValueError(
            "Some groups are too short for the configured encoder/prediction lengths: "
            f"{too_short.to_dict()}"
        )


def build_dataset(df: pd.DataFrame, config: TFTFixedConfig) -> DatasetBundle:
    cutoff = int(df["time_idx"].max() * (1 - config.val_ratio))

    train_dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff].copy(),
        time_idx="time_idx",
        target=config.target_column,
        group_ids=config.group_ids,
        max_encoder_length=config.max_encoder_length,
        max_prediction_length=config.max_prediction_length,
        static_categoricals=config.static_categoricals,
        time_varying_known_categoricals=config.time_varying_known_categoricals,
        time_varying_known_reals=config.time_varying_known_reals,
        time_varying_unknown_reals=config.time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=config.group_ids, center=False),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        df[df["time_idx"] > cutoff - config.max_encoder_length].copy(),
        predict=True,
        stop_randomization=True,
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        df[df["time_idx"] > cutoff - config.max_encoder_length].copy(),
        predict=True,
        stop_randomization=True,
    )
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        df[df["time_idx"] > cutoff - config.max_encoder_length].copy(),
        predict=True,
        stop_randomization=True,
    )

    return DatasetBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        prediction_dataset=prediction_dataset,
    )
