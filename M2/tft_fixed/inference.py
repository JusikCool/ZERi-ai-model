from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TFTFixedConfig
from .data import TFTFixedDataModule
from .model import load_tft_model_from_checkpoint
from .visualize import save_prediction_plot


def generate_prediction_frame(model, dataloader, config: TFTFixedConfig) -> pd.DataFrame:
    raw_predictions = model.predict(
        dataloader,
        mode="quantiles",
        return_index=True,
        trainer_kwargs={"accelerator": config.accelerator, "devices": config.devices},
    )

    index_df = raw_predictions.index.copy()
    predictions = raw_predictions.output

    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()

    if predictions.ndim == 1:
        predictions = predictions[:, None]

    if predictions.ndim == 2:
        prediction_columns = {
            f"pred_q{int(quantile * 100):02d}": predictions[:, idx]
            for idx, quantile in enumerate(config.quantiles)
        }
        return pd.concat([index_df.reset_index(drop=True), pd.DataFrame(prediction_columns)], axis=1)

    if predictions.ndim != 3:
        raise ValueError(f"Unsupported prediction shape: {predictions.shape}")

    horizon = predictions.shape[1]
    quantile_columns = [f"pred_q{int(quantile * 100):02d}" for quantile in config.quantiles]

    repeated_index = index_df.loc[index_df.index.repeat(horizon)].reset_index(drop=True).copy()
    repeated_index["horizon_step"] = np.tile(np.arange(1, horizon + 1), len(index_df))
    repeated_index["time_idx"] = repeated_index["time_idx"].astype(int) + repeated_index["horizon_step"] - 1

    flat_predictions = predictions.reshape(-1, predictions.shape[2])
    flat_prediction_df = pd.DataFrame(flat_predictions, columns=quantile_columns)
    expanded = pd.concat([repeated_index, flat_prediction_df], axis=1)

    aggregated = (
        expanded.groupby(["time_idx", "group_id"], as_index=False)[quantile_columns]
        .mean()
        .sort_values(["group_id", "time_idx"])
        .reset_index(drop=True)
    )
    return aggregated


def save_predictions(model, dataloader, output_path: Path, config: TFTFixedConfig) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df = generate_prediction_frame(model=model, dataloader=dataloader, config=config)
    prediction_df.to_csv(output_path, index=False)
    return output_path


def run_inference(
    checkpoint_path: Path,
    config: TFTFixedConfig,
    output_dir: Path,
    plot_ticker: str | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_module = TFTFixedDataModule(config)
    data_module.setup()

    model = load_tft_model_from_checkpoint(str(checkpoint_path), config)
    predictions_path = save_predictions(
        model=model,
        dataloader=data_module.predict_dataloader(),
        output_path=output_dir / "predictions.csv",
        config=config,
    )
    prediction_plot_path = save_prediction_plot(
        panel_df=data_module.dataframe,
        prediction_path=predictions_path,
        output_path=output_dir / "prediction_plot.png",
        target_column=config.target_column,
        ticker=plot_ticker or config.default_plot_ticker,
        title_prefix="TFT-FIXED",
    )
    return {
        "predictions_path": str(predictions_path),
        "prediction_plot_path": str(prediction_plot_path),
    }
