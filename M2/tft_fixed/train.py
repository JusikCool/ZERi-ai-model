from __future__ import annotations

import json
from pathlib import Path

from .compat import CSVLogger, EarlyStopping, LearningRateMonitor, ModelCheckpoint, pl
from .config import TFTFixedConfig
from .data import TFTFixedDataModule
from .inference import save_predictions
from .model import build_tft_model, load_tft_model_from_checkpoint
from .visualize import save_loss_curve, save_prediction_plot


def build_trainer(config: TFTFixedConfig, run_dir: Path) -> pl.Trainer:
    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min"),
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="tft-fixed-{epoch:02d}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = CSVLogger(save_dir=str(run_dir), name="logs")

    return pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        enable_model_summary=True,
        enable_progress_bar=False,
    )


def fit_tft_fixed_model(config: TFTFixedConfig, plot_ticker: str | None = None):
    pl.seed_everything(config.seed, workers=True)

    run_dir = config.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    data_module = TFTFixedDataModule(config)
    data_module.setup()

    model = build_tft_model(data_module.datasets.train_dataset, config)
    trainer = build_trainer(config, run_dir)

    trainer.fit(model, datamodule=data_module)
    validation_metrics = trainer.validate(
        model,
        datamodule=data_module,
        ckpt_path="best",
        weights_only=False,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    prediction_model = (
        load_tft_model_from_checkpoint(best_model_path, config)
        if best_model_path
        else model
    )

    predictions_path = save_predictions(
        model=prediction_model,
        dataloader=data_module.predict_dataloader(),
        output_path=run_dir / "predictions" / "test_predictions.csv",
        config=config,
    )

    plots_dir = run_dir / "plots"
    loss_curve_path = save_loss_curve(run_dir=run_dir, output_path=plots_dir / "loss_curve.png")
    prediction_plot_path = save_prediction_plot(
        panel_df=data_module.dataframe,
        prediction_path=predictions_path,
        output_path=plots_dir / "prediction_plot.png",
        target_column=config.target_column,
        ticker=plot_ticker or config.default_plot_ticker,
        title_prefix="TFT-FIXED",
    )

    summary = {
        "best_model_path": best_model_path,
        "predictions_path": str(predictions_path),
        "loss_curve_path": str(loss_curve_path),
        "prediction_plot_path": str(prediction_plot_path),
        "validation_metrics": validation_metrics,
        "loss_name": "quantile",
        "uses_vix": False,
        "uses_sigma": False,
        "uses_adaptive_loss": False,
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "model": prediction_model,
        "trainer": trainer,
        "data_module": data_module,
        "predictions_path": predictions_path,
        "loss_curve_path": loss_curve_path,
        "prediction_plot_path": prediction_plot_path,
        "summary_path": summary_path,
        "validation_metrics": validation_metrics,
    }
