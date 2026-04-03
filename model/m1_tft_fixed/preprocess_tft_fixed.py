from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
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
    "Month",
    "Day_of_Week",
    "RSI_14",
    "ATR_14",
    "SMA_20",
    "Returns",
    "Realized_Vol_20d",
    "Target_Return_5d",
    "group_id",
    "time_idx",
]
TARGET_COLUMN = "Target_Return_5d"


def infer_target_horizon(target_column: str = TARGET_COLUMN) -> int:
    match = re.search(r"Target_Return_(\d+)d", target_column)
    if match is None:
        raise ValueError(
            f"Could not infer target horizon from target column: {target_column}"
        )
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent.parent
    workspace_root = project_root.parent
    parser = argparse.ArgumentParser(
        description="Prepare the shared model input CSV used by m3_full_model and m1_tft_fixed."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=workspace_root
        / "data-pipeline-base"
        / "dataset"
        / "tft_processed_panel_v1.csv",
        help="Path to the processed panel CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "data" / "raw" / "tft_processed_panel_v1.csv",
        help="Path where the processed panel CSV will be written.",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=infer_target_horizon(),
    )
    return parser.parse_args()


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
    df["time_idx"] = df.groupby("group_id").cumcount()
    return df


def enforce_balanced_calendar(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("Date")["group_id"].nunique()
    expected = df["group_id"].nunique()
    invalid_dates = counts[counts != expected]
    if not invalid_dates.empty:
        df = df[~df["Date"].isin(invalid_dates.index)].copy()
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = [
        column
        for column in df.columns
        if column not in {"Date", "group_id", "Month", "Day_of_Week"}
    ]
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


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, REQUIRED_COLUMNS].copy()


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    df = load_panel(args.input)
    df = sort_and_cast(df)
    df = enforce_balanced_calendar(df)
    df = fill_missing_values(df)
    validate_target(df, horizon=args.prediction_horizon)
    df = select_output_columns(df)
    save_output(df=df, output_path=args.output_path)

    summary = {
        "output_path": str(args.output_path),
        "rows": int(len(df)),
        "tickers": sorted(df["group_id"].unique().tolist()),
        "date_min": df["Date"].min().strftime("%Y-%m-%d"),
        "date_max": df["Date"].max().strftime("%Y-%m-%d"),
        "prediction_horizon": args.prediction_horizon,
        "contains_vix": "VIX_Close" in df.columns,
        "contains_sigma": "Realized_Vol_20d" in df.columns,
    }
    print(f"Saved shared training panel to: {args.output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
