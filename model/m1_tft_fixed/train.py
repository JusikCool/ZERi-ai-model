from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

try:
    from .config import TFTFixedConfig
    from .data import ensure_processed_panel
    from .dataset import TFTFixedDataModule
    from .model import build_tft_model
except ImportError:  # pragma: no cover - direct script execution
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from config import TFTFixedConfig
    from data import ensure_processed_panel
    from dataset import TFTFixedDataModule
    from model import build_tft_model

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger
except ImportError:  # pragma: no cover - fallback for older environments
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import CSVLogger


def load_run_validation():
    project_root = Path(__file__).resolve().parent.parent.parent
    kupiec_path = project_root / "validation" / "kupiec" / "kupiec.py"
    spec = importlib.util.spec_from_file_location("kupiec_module", kupiec_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.run_validation, module.run_validation_by_group


run_validation, run_validation_by_group = load_run_validation()


def build_trainer(config: TFTFixedConfig, run_dir: Path) -> pl.Trainer:
    module_dir = Path(__file__).resolve().parent
    log_dir = module_dir / "lightning_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, mode="min"),
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="tft-fixed-{epoch:02d}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = CSVLogger(save_dir=str(log_dir), name="", version=None)

    return pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(module_dir),
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        enable_model_summary=True,
        enable_progress_bar=True,
    )


def fit_tft_fixed_model(config: TFTFixedConfig):
    pl.seed_everything(config.seed, workers=True)

    run_dir = config.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_processed_panel(output_path=config.data_path)

    data_module = TFTFixedDataModule(config)
    data_module.setup()

    model = build_tft_model(data_module.datasets.train_dataset, config)
    trainer = build_trainer(config, run_dir)

    trainer.fit(model, datamodule=data_module)
    validation_metrics = trainer.validate(
        model,
        datamodule=data_module,
        ckpt_path="best",
        verbose=False,
        weights_only=False,
    )
    kupiec_result_paths = run_kupiec_validation(
        model=model,
        data_module=data_module,
        config=config,
        run_dir=run_dir,
    )

    return {
        "model": model,
        "trainer": trainer,
        "data_module": data_module,
        "best_model_path": trainer.checkpoint_callback.best_model_path,
        "validation_metrics": validation_metrics,
        "kupiec_result_path": str(kupiec_result_paths["overall"]),
        "kupiec_by_group_result_path": str(kupiec_result_paths["by_group"]),
    }


def run_kupiec_validation(
    model,
    data_module: TFTFixedDataModule,
    config: TFTFixedConfig,
    run_dir: Path,
) -> dict[str, Path]:
    predictions = model.predict(
        data_module.val_dataloader(),
        mode="quantiles",
        return_y=True,
        return_index=True,
        trainer_kwargs={"accelerator": config.accelerator, "devices": config.devices},
    )

    y_pred = predictions.output
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = np.asarray(y_pred)

    y_true = predictions.y
    if isinstance(y_true, tuple):
        y_true = y_true[0]
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)

    group_values = predictions.index["group_id"].astype(str).to_numpy()

    if y_pred.ndim == 3:
        horizon = y_pred.shape[1]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        group_values = np.repeat(group_values, horizon)
    if y_true.ndim > 1:
        y_true = y_true.reshape(-1)

    overall_df = run_validation(
        y_true=y_true,
        y_pred=y_pred,
        quantiles=list(config.quantiles),
    )
    by_group_df = run_validation_by_group(
        y_true=y_true,
        y_pred=y_pred,
        groups=group_values,
        quantiles=list(config.quantiles),
    )

    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    overall_path = validation_dir / "kupiec_validation.csv"
    by_group_path = validation_dir / "kupiec_validation_by_group.csv"
    overall_df.to_csv(overall_path, index=False)
    by_group_df.to_csv(by_group_path, index=False)
    return {
        "overall": overall_path,
        "by_group": by_group_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TFT-FIXED model.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
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

    result = fit_tft_fixed_model(config)
    payload = {
        "best_model_path": result["best_model_path"],
        "validation_metrics": result["validation_metrics"],
        "kupiec_result_path": result["kupiec_result_path"],
        "kupiec_by_group_result_path": result["kupiec_by_group_result_path"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
