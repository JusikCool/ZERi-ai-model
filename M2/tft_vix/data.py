from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet

from .compat import pl
from .config import TFTVIXConfig


@dataclass
class DatasetBundle:
    train_dataset: TimeSeriesDataSet
    validation_dataset: TimeSeriesDataSet
    test_dataset: TimeSeriesDataSet
    prediction_dataset: TimeSeriesDataSet


class TFTVIXDataModule(pl.LightningDataModule):
    def __init__(self, config: TFTVIXConfig):
        super().__init__()
        self.config = config
        self.dataframe: Optional[pd.DataFrame] = None
        self.datasets: Optional[DatasetBundle] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.datasets is not None:
            return

        df = load_prepared_panel(self.config.data_path)
        validate_prepared_panel(df, self.config)
        datasets = build_datasets(df, self.config)

        self.dataframe = df
        self.datasets = datasets

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


def load_prepared_panel(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)
    df["group_id"] = df["group_id"].astype(str)
    df["Month"] = df["Month"].astype(str)
    df["Day_of_Week"] = df["Day_of_Week"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)
    return df


def validate_prepared_panel(df: pd.DataFrame, config: TFTVIXConfig) -> None:
    required_columns = (
        config.static_categoricals
        + config.time_varying_known_categoricals
        + config.time_varying_known_reals
        + config.time_varying_unknown_reals
        + [config.target_column, "Date", "split"]
    )
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Prepared panel is missing required columns: {missing}")

    numeric_columns = config.time_varying_known_reals + config.time_varying_unknown_reals + [config.target_column]
    numeric_frame = df[numeric_columns]
    if numeric_frame.isna().any().any():
        summary = numeric_frame.isna().sum()
        summary = summary[summary > 0].to_dict()
        raise ValueError(f"Prepared panel contains NaN values: {summary}")

    numeric_values = numeric_frame.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Prepared panel contains inf or -inf values.")

    for split_name in ("train", "validation", "test"):
        if split_name not in set(df["split"].unique()):
            raise ValueError(f"Prepared panel is missing split '{split_name}'.")

    group_lengths = df.groupby("group_id")["time_idx"].nunique()
    min_required_length = config.max_encoder_length + config.max_prediction_length
    too_short = group_lengths[group_lengths < min_required_length]
    if not too_short.empty:
        raise ValueError(
            "Some groups are too short for the configured encoder/prediction lengths: "
            f"{too_short.to_dict()}"
        )


def build_datasets(df: pd.DataFrame, config: TFTVIXConfig) -> DatasetBundle:
    train_cutoff = int(df.loc[df["split"] == "train", "time_idx"].max())
    validation_cutoff = int(df.loc[df["split"] == "validation", "time_idx"].max())

    training_dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff].copy(),
        time_idx="time_idx",
        target=config.target_column,
        group_ids=config.group_ids,
        min_encoder_length=config.max_encoder_length,
        max_encoder_length=config.max_encoder_length,
        min_prediction_length=config.max_prediction_length,
        max_prediction_length=config.max_prediction_length,
        static_categoricals=config.static_categoricals,
        time_varying_known_categoricals=config.time_varying_known_categoricals,
        time_varying_known_reals=config.time_varying_known_reals,
        time_varying_unknown_reals=config.time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=config.group_ids, center=False),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df[df["time_idx"] <= validation_cutoff].copy(),
        min_prediction_idx=train_cutoff + 1,
        stop_randomization=True,
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df.copy(),
        min_prediction_idx=validation_cutoff + 1,
        stop_randomization=True,
    )
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df.copy(),
        min_prediction_idx=validation_cutoff + 1,
        predict=False,
        stop_randomization=True,
    )

    return DatasetBundle(
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        prediction_dataset=prediction_dataset,
    )
