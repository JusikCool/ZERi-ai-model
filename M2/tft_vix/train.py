from __future__ import annotations

from pathlib import Path

from .compat import CSVLogger, EarlyStopping, LearningRateMonitor, ModelCheckpoint, pl
from .config import TFTVIXConfig
from .data import TFTVIXDataModule
from .inference import save_predictions
from .model import build_tft_model


def build_trainer(config: TFTVIXConfig, run_dir: Path) -> pl.Trainer:
    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min"),
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="tft-vix-{epoch:02d}-{val_loss:.5f}",
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


def fit_tft_vix_model(config: TFTVIXConfig):
    pl.seed_everything(config.seed, workers=True)

    run_dir = config.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    data_module = TFTVIXDataModule(config)
    data_module.setup()

    model = build_tft_model(data_module.datasets.train_dataset, config)
    trainer = build_trainer(config, run_dir)

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)

    predictions_path = save_predictions(
        model=model,
        dataloader=data_module.predict_dataloader(),
        output_path=run_dir / "predictions" / "test_predictions.csv",
        config=config,
    )

    return {
        "model": model,
        "trainer": trainer,
        "data_module": data_module,
        "predictions_path": predictions_path,
    }
