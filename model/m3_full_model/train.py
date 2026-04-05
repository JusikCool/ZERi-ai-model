from pathlib import Path

import optuna
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .dataset import build_dataset, build_dataloaders, get_vix_stats, load_data
from .model import M3FullModel

CONFIG_PATH = Path("configs/config.yaml")
CHECKPOINT_DIR = Path("model/saved")
LOG_DIR = Path("logs")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config: dict, trial: optuna.Trial | None = None) -> float:
    data_cfg = config["data"]
    model_cfg = config["model"]

    hidden_size = (
        trial.suggest_categorical("hidden_size", [32, 64, 128])
        if trial else model_cfg["hidden_size"]
    )
    attention_head_size = (
        trial.suggest_categorical("attention_head_size", [1, 2, 4])
        if trial else model_cfg["attention_head_size"]
    )
    dropout = (
        trial.suggest_float("dropout", 0.05, 0.3)
        if trial else model_cfg["dropout"]
    )
    learning_rate = (
        trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        if trial else model_cfg["learning_rate"]
    )
    alpha_down = (
        trial.suggest_float("alpha_down", 0.5, 3.0)
        if trial else 1.0
    )
    beta_down = (
        trial.suggest_float("beta_down", 0.5, 3.0)
        if trial else 1.0
    )
    alpha_up = (
        trial.suggest_float("alpha_up", 0.5, 3.0)
        if trial else 1.0
    )
    beta_up = (
        trial.suggest_float("beta_up", 0.5, 3.0)
        if trial else 1.0
    )
    crossing_weight = (
        trial.suggest_float("crossing_weight", 0.01, 0.5)
        if trial else 0.1
    )

    df = load_data()
    vix_mean, vix_std = get_vix_stats(df)
    train_ds, val_ds = build_dataset(
        df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )
    train_loader, val_loader = build_dataloaders(
        train_ds, val_ds, batch_size=model_cfg["batch_size"]
    )

    model = M3FullModel.from_dataset(
        dataset=train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
        vix_mean=vix_mean,
        vix_std=vix_std,
        alpha_down=alpha_down,
        beta_down=beta_down,
        alpha_up=alpha_up,
        beta_up=beta_up,
        crossing_weight=crossing_weight,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, mode="min"),
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename="m3_{epoch:02d}_{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=model_cfg["max_epochs"],
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=pl.loggers.CSVLogger(LOG_DIR, name="m3_full_model"),
        gradient_clip_val=0.1,
    )

    trainer.fit(model, train_loader, val_loader)

    return float(trainer.callback_metrics.get("val_loss", float("inf")))


def run_optuna(n_trials: int = 30, config_path: Path = CONFIG_PATH) -> optuna.Study:
    config = load_config(config_path)

    study = optuna.create_study(
        direction="minimize",
        study_name="m3_full_model",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        lambda trial: train(config, trial),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n[Optuna 결과]")
    print(f"  최적 val_loss : {study.best_value:.6f}")
    print(f"  최적 파라미터 : {study.best_params}")

    return study


if __name__ == "__main__":
    config = load_config()
    train(config)
