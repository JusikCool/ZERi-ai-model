from pathlib import Path

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader

# ===== M4 변경: v3 panel (GARCH_Variance + vol_group 포함) =====
DATA_PATH = Path("data/raw/tft_processed_panel_v3.csv")

TARGET = "Target_Return_5d"
TIME_IDX = "time_idx"
GROUP_ID = "group_id"

TIME_VARYING_KNOWN_CATEGORICALS = ["Month", "Day_of_Week"]

TIME_VARYING_UNKNOWN_REALS = [
    "Target_Return_5d", "Open", "High", "Low", "Close", "Volume",
    "Dividends", "Stock Splits",
    "NASDAQ_Close", "VIX_Close",
    "FEDFUNDS", "UNRATE", "DTWEXBGS", "CPIAUCSL", "PCEPI",
    "GDP", "M2SL", "GS10", "T10Y2Y", "PAYEMS", "CSUSHPISA", "INDPRO",
    "RSI_14", "ATR_14", "SMA_20",
    "Returns", "Realized_Vol_20d",
    "GARCH_Variance",   # ===== M4 추가: GARCHNet 차용 =====
]

# ===== M4 추가: static categorical 로 사용할 변동성 그룹 =====
STATIC_CATEGORICALS = [GROUP_ID, "vol_group"]


def get_vix_stats(df: pd.DataFrame) -> tuple[float, float]:
    vix = df["VIX_Close"].dropna()
    return float(vix.mean()), float(vix.std())


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)
    df[GROUP_ID] = df[GROUP_ID].astype(str)
    df["Month"] = df["Month"].astype(str)
    df["Day_of_Week"] = df["Day_of_Week"].astype(str)
    # ===== M4 추가: vol_group 도 categorical string 으로 =====
    if "vol_group" in df.columns:
        df["vol_group"] = df["vol_group"].astype(str)
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
        static_categoricals=STATIC_CATEGORICALS,   # ===== M4: vol_group 추가 =====
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(
            groups=[GROUP_ID], transformation=None
        ),
        add_relative_time_idx=False,
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


def build_dataset_for_tide(
    df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 10,
    val_ratio: float = 0.2,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    TiDE 전용 dataset 빌더.
    TiDE는 future covariate (time_varying_known_*) 처리에 dataset 구성과 충돌하는
    이슈가 있어서, future covariate (Month, Day_of_Week)를 제거한 dataset을 만든다.
    나머지 설정은 build_dataset()과 동일.
    """
    cutoff = int(df[TIME_IDX].max() * (1 - val_ratio))

    train_dataset = TimeSeriesDataSet(
        df[df[TIME_IDX] <= cutoff],
        time_idx=TIME_IDX,
        target=TARGET,
        group_ids=[GROUP_ID],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=STATIC_CATEGORICALS,   # ===== M4: vol_group 추가 =====
        # time_varying_known_categoricals 제거 (TiDE 호환성)
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(
            groups=[GROUP_ID], transformation=None
        ),
        add_relative_time_idx=False,
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


def build_dataset_for_deepar(
    df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 10,
    val_ratio: float = 0.2,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    DeepAR 전용 dataset 빌더.
    DeepAR은 'encoder/decoder variables가 target 외에 동일해야 한다'는 제약이 있어서,
    target을 제외한 모든 covariate를 time_varying_known_reals로 옮긴다.
    실제로 미래에 안다는 의미가 아니라, DeepAR의 autoregressive 동작 방식상
    encoder와 decoder에 동일한 covariate를 제공해야 하기 때문.
    """
    cutoff = int(df[TIME_IDX].max() * (1 - val_ratio))

    # target만 unknown으로 남기고, 나머지는 known reals로
    target_only_unknown = [TARGET]
    other_reals_as_known = [v for v in TIME_VARYING_UNKNOWN_REALS if v != TARGET]

    train_dataset = TimeSeriesDataSet(
        df[df[TIME_IDX] <= cutoff],
        time_idx=TIME_IDX,
        target=TARGET,
        group_ids=[GROUP_ID],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=STATIC_CATEGORICALS,   # ===== M4: vol_group 추가 =====
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_known_reals=other_reals_as_known,
        time_varying_unknown_reals=target_only_unknown,
        target_normalizer=GroupNormalizer(
            groups=[GROUP_ID], transformation=None
        ),
        add_relative_time_idx=False,
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
