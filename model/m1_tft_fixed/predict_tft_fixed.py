from __future__ import annotations

import argparse
import json
from pathlib import Path

from tft_fixed.config import TFTFixedConfig
from tft_fixed.inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the TFT-FIXED model.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-ticker", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TFTFixedConfig()
    if args.data_path is not None:
        config.data_path = args.data_path
    result = run_inference(
        checkpoint_path=args.checkpoint_path,
        config=config,
        output_dir=args.output_dir,
        plot_ticker=args.plot_ticker,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
