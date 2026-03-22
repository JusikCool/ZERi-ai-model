from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TFTVIXConfig


def save_predictions(model, dataloader, output_path: Path, config: TFTVIXConfig) -> Path:
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
