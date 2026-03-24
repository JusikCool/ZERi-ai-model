from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TFTFixedConfig
from .data import TFTFixedDataModule
from .model import load_tft_model_from_checkpoint
from .visualize import save_prediction_plot


def save_predictions(model, dataloader, output_path: Path, config: TFTFixedConfig) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    if predictions.ndim == 3 and predictions.shape[1] == 1:
        predictions = predictions[:, 0, :]
    elif predictions.ndim == 1:
        predictions = predictions[:, None]

    prediction_columns = {
        f"pred_q{int(quantile * 100):02d}": predictions[:, idx]
        for idx, quantile in enumerate(config.quantiles)
    }
    prediction_df = pd.concat([index_df.reset_index(drop=True), pd.DataFrame(prediction_columns)], axis=1)
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
