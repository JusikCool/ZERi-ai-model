from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .config import TFTFixedConfig
    from .preprocess_tft_fixed import (
        enforce_balanced_calendar,
        fill_missing_values,
        infer_target_horizon,
        load_panel,
        save_output,
        select_output_columns,
        sort_and_cast,
        validate_target,
    )
except ImportError:  # pragma: no cover - direct script execution
    from config import TFTFixedConfig
    from preprocess_tft_fixed import (
        enforce_balanced_calendar,
        fill_missing_values,
        infer_target_horizon,
        load_panel,
        save_output,
        select_output_columns,
        sort_and_cast,
        validate_target,
    )


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_SOURCE_PATH = (
    WORKSPACE_ROOT / "data-pipeline-base" / "dataset" / "tft_processed_panel_v1.csv"
)


def build_processed_panel(
    source_path: Path = DEFAULT_SOURCE_PATH,
    output_path: Path | None = None,
    prediction_horizon: int = infer_target_horizon(),
) -> Path:
    config = TFTFixedConfig()
    output_path = output_path or config.data_path

    if not source_path.exists():
        raise FileNotFoundError(
            "Could not build data/raw/tft_processed_panel_v1.csv because the source "
            f"file does not exist: {source_path}"
        )

    df = load_panel(source_path)
    df = sort_and_cast(df)
    df = enforce_balanced_calendar(df)
    df = fill_missing_values(df)
    validate_target(df, horizon=prediction_horizon)
    df = select_output_columns(df)
    save_output(df=df, output_path=output_path)
    return output_path


def ensure_processed_panel(
    output_path: Path | None = None,
    source_path: Path = DEFAULT_SOURCE_PATH,
    prediction_horizon: int = infer_target_horizon(),
) -> Path:
    config = TFTFixedConfig()
    output_path = output_path or config.data_path
    if output_path.exists():
        return output_path
    return build_processed_panel(
        source_path=source_path,
        output_path=output_path,
        prediction_horizon=prediction_horizon,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create data/raw/tft_processed_panel_v1.csv for m1_tft_fixed."
    )
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=infer_target_horizon(),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path or TFTFixedConfig().data_path
    if args.force or not output_path.exists():
        result_path = build_processed_panel(
            source_path=args.source_path,
            output_path=output_path,
            prediction_horizon=args.prediction_horizon,
        )
    else:
        result_path = output_path

    payload = {
        "output_path": str(result_path),
        "exists": result_path.exists(),
        "source_path": str(args.source_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
