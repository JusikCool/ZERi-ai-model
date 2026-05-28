from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader, WeightedRandomSampler

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


def _extract_sample_keys(
    train_ds: TimeSeriesDataSet,
    probe_batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    train_ds 의 각 sample 에 대해 (group_id, prediction_start_time_idx) 추출.

    pytorch-forecasting 버전마다 train_ds.index 구조가 다르고 (특히 group_id 가
    별도 categorical encoder 에 들어있는 경우), 공식 public API 인
    train_ds.x_to_index(batch_x) 를 sequential dataloader 로 한 번 훑어서
    안정적으로 수집한다.

    Fallback:
      1) train_ds.decoded_index 가 group_id + time 컬럼을 가지고 있으면 그것 사용 (빠름)
      2) 위가 안 되면 x_to_index 경로 (모든 pytorch-forecasting 0.10+ 호환)
    """
    decoded = getattr(train_ds, "decoded_index", None)
    if decoded is not None and isinstance(decoded, pd.DataFrame):
        gcol = GROUP_ID if GROUP_ID in decoded.columns else None
        tcol = "time" if "time" in decoded.columns else (
            TIME_IDX if TIME_IDX in decoded.columns else None
        )
        if gcol and tcol:
            return (
                decoded[gcol].astype(str).to_numpy(),
                decoded[tcol].astype(int).to_numpy(),
            )

    if not hasattr(train_ds, "x_to_index"):
        raise RuntimeError(
            "train_ds.x_to_index() 가 없습니다. pytorch-forecasting 0.10+ 권장."
        )

    print(f"[sampler] x_to_index 로 sample 키 추출 (batch_size={probe_batch_size}) ...")
    probe_loader = train_ds.to_dataloader(
        train=False, batch_size=probe_batch_size, num_workers=0
    )
    groups_all: list[str] = []
    times_all: list[int] = []
    for batch in probe_loader:
        x, _ = batch
        idx_df = train_ds.x_to_index(x)
        gcol = GROUP_ID if GROUP_ID in idx_df.columns else None
        tcol = TIME_IDX if TIME_IDX in idx_df.columns else (
            "time" if "time" in idx_df.columns else None
        )
        if gcol is None or tcol is None:
            raise RuntimeError(
                f"x_to_index() 결과에서 group/time 컬럼 추출 실패. "
                f"available columns={list(idx_df.columns)}"
            )
        groups_all.extend(idx_df[gcol].astype(str).tolist())
        times_all.extend(idx_df[tcol].astype(int).tolist())

    print(f"[sampler] {len(groups_all)} samples 키 수집 완료")
    return (
        np.asarray(groups_all, dtype=object),
        np.asarray(times_all, dtype=np.int64),
    )


def compute_crisis_weights(
    train_ds: TimeSeriesDataSet,
    df: pd.DataFrame,
    encoder_length: int,
    vix_col: str = "VIX_Close",
    vix_threshold: float = 25.0,
    high_vix_multiplier: float = 3.0,
) -> np.ndarray:
    """
    각 학습 sample 의 encoder window 내 'VIX > threshold 비율' 기반 가중치 계산.

    weight_i = 1 + (high_vix_multiplier - 1) * (high_vix_ratio in window_i)

    공포 구간(VIX>25) 비율이 50%인 윈도우는 weight = 1 + 2*0.5 = 2.0 (high_vix_multiplier=3 일 때)
    평시 윈도우 (VIX>25 가 0%)는 weight = 1.0
    """
    df_sorted = df.sort_values([GROUP_ID, TIME_IDX]).reset_index(drop=True)

    def _rolling_high_vix_ratio(s: pd.Series) -> pd.Series:
        is_high = (s > vix_threshold).astype(float)
        return is_high.rolling(window=encoder_length, min_periods=1).mean()

    df_sorted["_hv_ratio"] = (
        df_sorted.groupby(GROUP_ID)[vix_col]
        .transform(_rolling_high_vix_ratio)
    )

    ratio_lookup = (
        df_sorted.set_index([GROUP_ID, TIME_IDX])["_hv_ratio"].to_dict()
    )

    groups, time_starts = _extract_sample_keys(train_ds)

    weights = np.ones(len(groups), dtype=np.float32)
    miss = 0
    for i, (g, t) in enumerate(zip(groups, time_starts)):
        key = (str(g), int(t) - 1)
        ratio = ratio_lookup.get(key, None)
        if ratio is None:
            miss += 1
            ratio = 0.0
        weights[i] = 1.0 + (high_vix_multiplier - 1.0) * float(ratio)

    high_w = float((weights > 1.5).mean())
    print(
        f"[crisis-aware sampler] n_samples={len(weights)}, "
        f"mean_w={weights.mean():.3f}, "
        f"max_w={weights.max():.3f}, "
        f"high_vix_window_fraction(w>1.5)={high_w:.3f}, "
        f"missing_keys={miss}"
    )
    return weights


def build_crisis_aware_dataloaders(
    train_dataset: TimeSeriesDataSet,
    val_dataset: TimeSeriesDataSet,
    df: pd.DataFrame,
    encoder_length: int,
    batch_size: int = 64,
    num_workers: int = 0,
    vix_threshold: float = 25.0,
    high_vix_multiplier: float = 3.0,
) -> tuple[DataLoader, DataLoader]:
    """
    train_loader 는 WeightedRandomSampler 로 공포 구간 윈도우를 oversampling.
    val_loader 는 그대로 (편향 없는 검증).
    """
    weights = compute_crisis_weights(
        train_dataset, df,
        encoder_length=encoder_length,
        vix_threshold=vix_threshold,
        high_vix_multiplier=high_vix_multiplier,
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    train_loader = train_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=num_workers
    )
    return train_loader, val_loader
