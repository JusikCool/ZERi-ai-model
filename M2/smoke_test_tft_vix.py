from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tft_vix.config import TFTVIXConfig
from tft_vix.train import fit_tft_vix_model


def main() -> None:
    config = TFTVIXConfig(
        output_dir=Path(r"C:\Users\user\Desktop\JERi\ZERi-ai-model\M2\runs\tft_vix_smoke"),
        max_epochs=2,
        batch_size=64,
        limit_train_batches=4,
        limit_val_batches=2,
    )
    result = fit_tft_vix_model(config)

    prediction_path = Path(result["predictions_path"])
    prediction_df = pd.read_csv(prediction_path)
    prediction_columns = [column for column in prediction_df.columns if column.startswith("pred_q")]

    if prediction_df.empty:
        raise RuntimeError("Prediction output is empty.")
    if not prediction_columns:
        raise RuntimeError("Prediction output columns are missing.")

    values = prediction_df[prediction_columns].to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise RuntimeError("Prediction output contains NaN.")
    if not np.isfinite(values).all():
        raise RuntimeError("Prediction output contains inf or -inf.")

    payload = {
        "prediction_rows": int(len(prediction_df)),
        "prediction_columns": prediction_columns,
        "prediction_shape": list(values.shape),
        "best_model_path": result["trainer"].checkpoint_callback.best_model_path,
        "prediction_path": str(prediction_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
