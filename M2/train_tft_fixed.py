from __future__ import annotations

import argparse
import json
from pathlib import Path

from tft_fixed.config import TFTFixedConfig
from tft_fixed.train import fit_tft_fixed_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TFT-FIXED model.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--plot-ticker", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TFTFixedConfig()

    if args.data_path is not None:
        config.data_path = args.data_path
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.smoke_test:
        config.max_epochs = config.smoke_test_epochs
        config.limit_train_batches = 4
        config.limit_val_batches = 2
        config.output_dir = config.output_dir / "smoke_test"

    result = fit_tft_fixed_model(config, plot_ticker=args.plot_ticker)
    payload = {
        "predictions_path": str(result["predictions_path"]),
        "best_model_path": result["trainer"].checkpoint_callback.best_model_path,
        "loss_curve_path": str(result["loss_curve_path"]),
        "prediction_plot_path": str(result["prediction_plot_path"]),
        "summary_path": str(result["summary_path"]),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
