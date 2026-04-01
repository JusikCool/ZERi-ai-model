from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import TFTFixedConfig
from .data import TFTFixedDataModule
from .inference import generate_prediction_frame
from .metrics import build_group_quantile_metrics, build_quantile_metrics, build_summary_metrics
from .model import load_tft_model_from_checkpoint
from .visualize import save_prediction_plot


def build_evaluation_frame(panel_df: pd.DataFrame, prediction_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    actual_df = panel_df[["time_idx", "group_id", "Date", target_column]].copy()
    actual_df = actual_df.rename(columns={target_column: "y_true"})
    merged = prediction_df.merge(actual_df, on=["time_idx", "group_id"], how="left", validate="one_to_one")
    ordered_columns = ["pred_q10", "pred_q50", "pred_q90", "y_true", "time_idx", "group_id", "Date"]
    return merged.loc[:, ordered_columns].copy()


def evaluate_tft_fixed_model(
    checkpoint_path: Path,
    config: TFTFixedConfig,
    output_dir: Path,
    plot_ticker: str | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    data_module = TFTFixedDataModule(config)
    data_module.setup()

    model = load_tft_model_from_checkpoint(str(checkpoint_path), config)
    raw_prediction_df = generate_prediction_frame(
        model=model,
        dataloader=data_module.predict_dataloader(),
        config=config,
    )
    evaluation_df = build_evaluation_frame(
        panel_df=data_module.dataframe,
        prediction_df=raw_prediction_df,
        target_column=config.target_column,
    )

    prediction_csv_path = predictions_dir / "test_predictions.csv"
    evaluation_df.to_csv(prediction_csv_path, index=False)

    quantile_metrics = build_quantile_metrics(evaluation_df, config.quantiles)
    quantile_metrics_csv_path = metrics_dir / "quantile_metrics.csv"
    quantile_metrics.to_csv(quantile_metrics_csv_path, index=False)

    group_quantile_metrics = build_group_quantile_metrics(evaluation_df, config.quantiles)
    group_quantile_metrics_csv_path = metrics_dir / "group_quantile_metrics.csv"
    group_quantile_metrics.to_csv(group_quantile_metrics_csv_path, index=False)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "prediction_csv_path": str(prediction_csv_path),
        "quantile_metrics_csv_path": str(quantile_metrics_csv_path),
        "group_quantile_metrics_csv_path": str(group_quantile_metrics_csv_path),
        "summary_metrics": build_summary_metrics(quantile_metrics),
        "quantiles": list(config.quantiles),
        "uses_vix": False,
        "uses_sigma": False,
        "uses_adaptive_loss": False,
    }
    summary_json_path = metrics_dir / "evaluation_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_paths: list[str] = []
    for ticker in sorted(evaluation_df["group_id"].unique().tolist()):
        plot_path = save_prediction_plot(
            panel_df=data_module.dataframe,
            prediction_path=prediction_csv_path,
            output_path=plots_dir / f"prediction_plot_{ticker}.png",
            target_column=config.target_column,
            ticker=ticker,
            title_prefix="TFT-FIXED Evaluation",
        )
        plot_paths.append(str(plot_path))

    result = {
        "prediction_csv_path": str(prediction_csv_path),
        "quantile_metrics_csv_path": str(quantile_metrics_csv_path),
        "group_quantile_metrics_csv_path": str(group_quantile_metrics_csv_path),
        "summary_json_path": str(summary_json_path),
        "primary_plot_path": str(plots_dir / f"prediction_plot_{plot_ticker or config.default_plot_ticker}.png"),
        "all_plot_paths": plot_paths,
    }
    result_json_path = output_dir / "evaluation_result.json"
    result_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
