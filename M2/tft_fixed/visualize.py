from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _latest_metrics_path(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "logs").glob("version_*/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv files found under {run_dir / 'logs'}")
    return candidates[-1]


def save_loss_curve(run_dir: Path, output_path: Path) -> Path:
    metrics_path = _latest_metrics_path(run_dir)
    metrics_df = pd.read_csv(metrics_path)

    train_df = (
        metrics_df.dropna(subset=["train_loss_epoch"])[["epoch", "train_loss_epoch"]]
        .drop_duplicates(subset=["epoch"])
        .sort_values("epoch")
    )
    val_df = (
        metrics_df.dropna(subset=["val_loss"])[["epoch", "val_loss"]]
        .drop_duplicates(subset=["epoch"])
        .sort_values("epoch")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    if not train_df.empty:
        plt.plot(train_df["epoch"], train_df["train_loss_epoch"], label="train_loss")
    if not val_df.empty:
        plt.plot(val_df["epoch"], val_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TFT-FIXED Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_prediction_plot(
    panel_df: pd.DataFrame,
    prediction_path: Path,
    output_path: Path,
    target_column: str,
    ticker: str,
    title_prefix: str = "TFT-FIXED",
    max_points: int = 120,
) -> Path:
    prediction_df = pd.read_csv(prediction_path)
    merge_columns = ["time_idx", "group_id"]
    actual_df = panel_df[merge_columns + ["Date", target_column]].copy()
    merged = prediction_df.merge(actual_df, on=merge_columns, how="left", validate="one_to_one")
    merged = merged[merged["group_id"] == ticker].sort_values("time_idx").tail(max_points)
    if merged.empty:
        raise ValueError(f"No prediction rows found for ticker {ticker}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(merged["Date"], merged[target_column], label="actual", linewidth=2)
    plt.plot(merged["Date"], merged["pred_q50"], label="pred_q50", linewidth=1.8)
    plt.plot(merged["Date"], merged["pred_q25"], label="pred_q25", linewidth=1.2)
    plt.plot(merged["Date"], merged["pred_q10"], label="pred_q10", linewidth=1.2)
    plt.fill_between(
        merged["Date"],
        merged["pred_q10"],
        merged["pred_q25"],
        alpha=0.2,
        label="q10-q25 band",
    )
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Date")
    plt.ylabel(target_column)
    plt.title(f"{title_prefix} Quantile Forecasts: {ticker}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path
