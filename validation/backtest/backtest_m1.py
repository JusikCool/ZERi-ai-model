from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "model" / "m1_tft_fixed" / "runs" / "tft_fixed" / "checkpoints"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model" / "m1_tft_fixed" / "runs" / "tft_fixed" / "backtest"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.m1_tft_fixed.config import TFTFixedConfig
from model.m1_tft_fixed.dataset import build_dataset, load_data
from model.m1_tft_fixed.model import load_tft_model_from_checkpoint


def load_kupiec_functions():
    kupiec_path = PROJECT_ROOT / "validation" / "kupiec" / "kupiec.py"
    spec = importlib.util.spec_from_file_location("kupiec_module_m1_backtest", kupiec_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.run_validation, module.run_validation_by_group


run_validation, run_validation_by_group = load_kupiec_functions()


def load_validation_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    return checkpoints[-1]


def collect_fold_predictions(
    model,
    train_dataset,
    df: pd.DataFrame,
    config: TFTFixedConfig,
    n_splits: int,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    max_time = int(df["time_idx"].max())
    min_time = int(df["time_idx"].min())
    fold_size = (max_time - min_time) // (n_splits + 1)
    if fold_size <= 0:
        raise ValueError("Fold size must be positive. Increase data length or reduce n_splits.")

    all_y_true: list[pd.Series] = []
    all_y_pred: list[pd.DataFrame] = []
    all_groups: list[pd.Series] = []

    for i in range(n_splits):
        train_end = min_time + fold_size * (i + 1)
        val_end = train_end + fold_size
        fold_df = df[
            (df["time_idx"] > train_end - config.max_encoder_length)
            & (df["time_idx"] <= val_end)
        ].copy()

        val_dataset = train_dataset.from_dataset(
            train_dataset,
            fold_df,
            predict=False,
            stop_randomization=True,
        )
        val_loader = val_dataset.to_dataloader(
            train=False,
            batch_size=config.batch_size * 2,
            num_workers=config.num_workers,
        )

        predictions = model.predict(
            val_loader,
            mode="quantiles",
            return_index=True,
            return_y=True,
            trainer_kwargs={"accelerator": config.accelerator, "devices": config.devices},
        )

        y_pred = predictions.output
        if hasattr(y_pred, "detach"):
            y_pred = y_pred.detach().cpu().numpy()

        y_true = predictions.y
        if isinstance(y_true, tuple):
            y_true = y_true[0]
        if hasattr(y_true, "detach"):
            y_true = y_true.detach().cpu().numpy()

        group_values = predictions.index["group_id"].astype(str).reset_index(drop=True)

        if y_pred.ndim == 3:
            horizon = y_pred.shape[1]
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            group_values = pd.Series(group_values.repeat(horizon).to_numpy())
        else:
            group_values = pd.Series(group_values.to_numpy())

        y_true_series = pd.Series(y_true.reshape(-1))
        y_pred_frame = pd.DataFrame(
            y_pred,
            columns=[f"pred_q{int(q * 100):02d}" for q in config.quantiles],
        )

        all_y_true.append(y_true_series)
        all_y_pred.append(y_pred_frame)
        all_groups.append(group_values)

    return (
        pd.concat(all_y_true, ignore_index=True),
        pd.concat(all_y_pred, ignore_index=True),
        pd.concat(all_groups, ignore_index=True),
    )


def rolling_window_backtest_m1(
    checkpoint_path: Path,
    config: TFTFixedConfig,
    n_splits: int = 5,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    thresholds: dict | None = None,
) -> dict[str, str]:
    thresholds = thresholds or load_validation_config()["validation"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(config.data_path)
    datasets = build_dataset(df, config)
    train_dataset = datasets.train_dataset
    model = load_tft_model_from_checkpoint(str(checkpoint_path), config)

    y_true, y_pred, groups = collect_fold_predictions(
        model=model,
        train_dataset=train_dataset,
        df=df,
        config=config,
        n_splits=n_splits,
    )

    overall_report = run_validation(
        y_true=y_true.to_numpy(),
        y_pred=y_pred.to_numpy(),
        quantiles=list(config.quantiles),
        vr_threshold=thresholds["violation_rate_threshold"],
        pvalue_threshold=thresholds["kupiec_pvalue_threshold"],
    )
    by_group_report = run_validation_by_group(
        y_true=y_true.to_numpy(),
        y_pred=y_pred.to_numpy(),
        groups=groups.to_numpy(),
        quantiles=list(config.quantiles),
        vr_threshold=thresholds["violation_rate_threshold"],
        pvalue_threshold=thresholds["kupiec_pvalue_threshold"],
    )

    overall_path = output_dir / "backtest_kupiec_overall.csv"
    by_group_path = output_dir / "backtest_kupiec_by_group.csv"
    overall_report.to_csv(overall_path, index=False)
    by_group_report.to_csv(by_group_path, index=False)

    result = {
        "checkpoint_path": str(checkpoint_path),
        "overall_report_path": str(overall_path),
        "by_group_report_path": str(by_group_path),
        "n_splits": str(n_splits),
    }
    (output_dir / "backtest_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-fold rolling window backtest for m1_tft_fixed."
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TFTFixedConfig()
    if args.data_path is not None:
        config.data_path = args.data_path

    checkpoint_path = args.checkpoint_path or find_latest_checkpoint()
    result = rolling_window_backtest_m1(
        checkpoint_path=checkpoint_path,
        config=config,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
