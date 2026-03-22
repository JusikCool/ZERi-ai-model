from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "NASDAQ_Close",
    "VIX_Close",
    "RSI_14",
    "ATR_14",
    "SMA_20",
    "Returns",
    "Realized_Vol_20d",
    "Month",
    "Day_of_Week",
    "Target_Return_5d",
    "group_id",
    "time_idx",
]

STATIC_CATEGORICALS = ["group_id"]
TIME_VARYING_KNOWN_CATEGORICALS = ["Month", "Day_of_Week"]
TIME_VARYING_KNOWN_REALS = ["time_idx"]
TIME_VARYING_UNKNOWN_REALS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "NASDAQ_Close",
    "VIX_Close",
    "FEDFUNDS",
    "UNRATE",
    "DTWEXBGS",
    "CPIAUCSL",
    "PCEPI",
    "GDP",
    "M2SL",
    "GS10",
    "T10Y2Y",
    "PAYEMS",
    "CSUSHPISA",
    "INDPRO",
    "RSI_14",
    "ATR_14",
    "SMA_20",
    "Returns",
    "Realized_Vol_20d",
]
TARGET_COLUMN = "Target_Return_5d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a TFT-VIX training panel from the processed panel CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            r"C:\Users\user\Desktop\JERi\data-pipeline-base\dataset\tft_processed_panel_v1.csv"
        ),
        help="Path to the processed panel CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\user\Desktop\JERi\ZERi-ai-model\M2\artifacts"),
        help="Directory where processed outputs will be written.",
    )
    parser.add_argument(
        "--encoder-length",
        type=int,
        default=60,
        help="Context length reserved for TFT encoder.",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=5,
        help="Forward trading-day return horizon used for target validation.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio based on unique dates.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio based on unique dates.",
    )
    return parser.parse_args()


def validate_split_ratios(train_ratio: float, val_ratio: float) -> float:
    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train/validation/test ratios must all be positive.")
    return test_ratio


def load_panel(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    return df


def sort_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["group_id", "Date"]).reset_index(drop=True)

    df["group_id"] = df["group_id"].astype(str)
    df["Month"] = df["Month"].astype(str)
    df["Day_of_Week"] = df["Day_of_Week"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)

    return df


def enforce_balanced_calendar(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("Date")["group_id"].nunique()
    expected = df["group_id"].nunique()
    invalid_dates = counts[counts != expected]
    if not invalid_dates.empty:
        # Drop incomplete panel dates so every remaining date has the same ticker set.
        df = df[~df["Date"].isin(invalid_dates.index)].copy()
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = [
        column
        for column in df.columns
        if column not in {"Date", "group_id", "Month", "Day_of_Week"}
    ]

    # Fill within each ticker so one asset never borrows information from another.
    df[numeric_columns] = df.groupby("group_id")[numeric_columns].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    df[numeric_columns] = df.groupby("group_id")[numeric_columns].transform(
        lambda s: s.ffill().bfill()
    )

    if df[numeric_columns].isna().any().any():
        remaining = df[numeric_columns].isna().sum()
        remaining = remaining[remaining > 0].to_dict()
        raise ValueError(f"Missing values remain after fill: {remaining}")

    return df


def regenerate_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    future_close = df.groupby("group_id")["Close"].shift(-horizon)
    return future_close / df["Close"] - 1.0


def validate_target(df: pd.DataFrame, horizon: int, tolerance: float = 1e-10) -> None:
    regenerated = regenerate_target(df, horizon=horizon)
    comparison = pd.concat(
        [df[TARGET_COLUMN], regenerated.rename("regenerated_target")], axis=1
    ).dropna()
    max_abs_error = (
        comparison[TARGET_COLUMN] - comparison["regenerated_target"]
    ).abs().max()
    if pd.isna(max_abs_error) or max_abs_error > tolerance:
        raise ValueError(
            f"Target validation failed. Max absolute difference was {max_abs_error}."
        )


def filter_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    allowed_columns = (
        ["Date"]
        + STATIC_CATEGORICALS
        + TIME_VARYING_KNOWN_CATEGORICALS
        + TIME_VARYING_KNOWN_REALS
        + [column for column in TIME_VARYING_UNKNOWN_REALS if column in df.columns]
        + [TARGET_COLUMN]
    )
    return df.loc[:, allowed_columns].copy()


def assign_split_labels(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> tuple[pd.DataFrame, dict[str, str]]:
    unique_dates = sorted(df["Date"].unique())
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))

    # Keep at least one full date in each split.
    train_end_idx = min(max(train_end_idx, 1), n_dates - 2)
    val_end_idx = min(max(val_end_idx, train_end_idx + 1), n_dates - 1)

    train_end_date = unique_dates[train_end_idx - 1]
    val_end_date = unique_dates[val_end_idx - 1]

    split_map: dict[pd.Timestamp, str] = {}
    for date in unique_dates[:train_end_idx]:
        split_map[date] = "train"
    for date in unique_dates[train_end_idx:val_end_idx]:
        split_map[date] = "validation"
    for date in unique_dates[val_end_idx:]:
        split_map[date] = "test"

    df = df.copy()
    df["split"] = df["Date"].map(split_map)

    boundaries = {
        "train_end_date": pd.Timestamp(train_end_date).strftime("%Y-%m-%d"),
        "validation_end_date": pd.Timestamp(val_end_date).strftime("%Y-%m-%d"),
        "test_end_date": pd.Timestamp(unique_dates[-1]).strftime("%Y-%m-%d"),
    }
    return df, boundaries


def build_metadata(
    df: pd.DataFrame,
    input_path: Path,
    encoder_length: int,
    prediction_horizon: int,
    boundaries: dict[str, str],
) -> dict[str, object]:
    return {
        "input_path": str(input_path),
        "rows": int(len(df)),
        "tickers": sorted(df["group_id"].unique().tolist()),
        "date_min": df["Date"].min().strftime("%Y-%m-%d"),
        "date_max": df["Date"].max().strftime("%Y-%m-%d"),
        "encoder_length": encoder_length,
        "decoder_length": 1,
        "prediction_horizon": prediction_horizon,
        "target_column": TARGET_COLUMN,
        "static_categoricals": STATIC_CATEGORICALS,
        "time_varying_known_categoricals": TIME_VARYING_KNOWN_CATEGORICALS,
        "time_varying_known_reals": TIME_VARYING_KNOWN_REALS,
        "time_varying_unknown_reals": [
            column for column in TIME_VARYING_UNKNOWN_REALS if column in df.columns
        ],
        "train_rows": int((df["split"] == "train").sum()),
        "validation_rows": int((df["split"] == "validation").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "split_boundaries": boundaries,
    }


def save_outputs(df: pd.DataFrame, output_dir: Path, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / "tft_vix_panel_ready.csv"
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"
    metadata_path = output_dir / "dataset_meta.json"

    df.to_csv(all_path, index=False)
    df[df["split"] == "train"].to_csv(train_path, index=False)
    df[df["split"] == "validation"].to_csv(val_path, index=False)
    df[df["split"] == "test"].to_csv(test_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_split_ratios(args.train_ratio, args.val_ratio)

    df = load_panel(args.input)
    df = sort_and_cast(df)
    df = enforce_balanced_calendar(df)
    df = fill_missing_values(df)
    validate_target(df, horizon=args.prediction_horizon)
    df = filter_leakage_columns(df)
    df, boundaries = assign_split_labels(
        df, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    metadata = build_metadata(
        df=df,
        input_path=args.input,
        encoder_length=args.encoder_length,
        prediction_horizon=args.prediction_horizon,
        boundaries=boundaries,
    )
    save_outputs(df=df, output_dir=args.output_dir, metadata=metadata)

    print(f"Saved TFT-ready panel to: {args.output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
