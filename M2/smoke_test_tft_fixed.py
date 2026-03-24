from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tft_fixed.config import TFTFixedConfig
from tft_fixed.evaluate import evaluate_tft_fixed_model
from tft_fixed.train import fit_tft_fixed_model


def main() -> None:
    config = TFTFixedConfig(
        output_dir=Path(r"C:\Users\user\Desktop\JERi\ZERi-ai-model\M2\runs\tft_fixed_smoke"),
        max_epochs=2,
        batch_size=64,
        limit_train_batches=4,
        limit_val_batches=2,
    )
    result = fit_tft_fixed_model(config)
    evaluation = evaluate_tft_fixed_model(
        checkpoint_path=Path(result["best_model_path"]),
        config=config,
        output_dir=Path(r"C:\Users\user\Desktop\JERi\ZERi-ai-model\M2\runs\tft_fixed_eval_smoke"),
        plot_ticker="AAPL",
    )

    prediction_path = Path(evaluation["prediction_csv_path"])
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
        "loss_name": "quantile",
        "best_model_path": result["best_model_path"],
        "prediction_path": str(prediction_path),
        "quantile_metrics_csv_path": evaluation["quantile_metrics_csv_path"],
        "summary_json_path": evaluation["summary_json_path"],
        "primary_plot_path": evaluation["primary_plot_path"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
