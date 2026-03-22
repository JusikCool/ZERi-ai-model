import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from model.m3_full_model.dataset import (
    build_dataset,
    build_dataloaders,
    TIME_IDX,
    TARGET,
)
from model.m3_full_model.model import M3FullModel
from validation.kupiec.kupiec import run_validation


def rolling_window_backtest(
    df: pd.DataFrame,
    config: dict,
    n_splits: int = 5,
) -> pd.DataFrame:
    data_cfg = config["data"]
    model_cfg = config["model"]

    max_time = df[TIME_IDX].max()
    min_time = df[TIME_IDX].min()
    total_len = max_time - min_time

    fold_size = total_len // (n_splits + 1)

    all_y_true = []
    all_y_pred = []

    for i in range(n_splits):
        train_end = min_time + fold_size * (i + 1)
        val_end = train_end + fold_size

        train_df = df[df[TIME_IDX] <= train_end]
        val_df = df[
            (df[TIME_IDX] > train_end - data_cfg["window_size"])
            & (df[TIME_IDX] <= val_end)
        ]

        train_ds, val_ds = build_dataset(
            pd.concat([train_df, val_df]).reset_index(drop=True),
            max_encoder_length=data_cfg["window_size"],
            max_prediction_length=data_cfg["horizon"],
            val_ratio=fold_size / (fold_size * (i + 1) + fold_size),
        )
        _, val_loader = build_dataloaders(
            train_ds, val_ds, batch_size=model_cfg["batch_size"]
        )

        model = M3FullModel.from_dataset(
            dataset=train_ds,
            learning_rate=model_cfg["learning_rate"],
            hidden_size=model_cfg["hidden_size"],
            attention_head_size=model_cfg["attention_head_size"],
            dropout=model_cfg["dropout"],
            quantiles=model_cfg["quantiles"],
            vix_threshold=model_cfg["vix_threshold"],
        )

        trainer = pl.Trainer(
            max_epochs=model_cfg["max_epochs"],
            callbacks=[EarlyStopping(monitor="val_loss", patience=8, mode="min")],
            enable_progress_bar=False,
            logger=False,
            gradient_clip_val=0.1,
        )

        _, val_loader_train = build_dataloaders(
            train_ds, val_ds, batch_size=model_cfg["batch_size"]
        )
        trainer.fit(model, val_loader_train, val_loader)

        fold_preds = []
        fold_trues = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                y_true = y[0].numpy()
                y_pred = model.predict(x).numpy()
                fold_trues.append(y_true.reshape(-1, y_true.shape[-1]))
                fold_preds.append(y_pred.reshape(-1, y_pred.shape[-1]))

        all_y_true.append(np.concatenate(fold_trues, axis=0))
        all_y_pred.append(np.concatenate(fold_preds, axis=0))

    y_true_all = np.concatenate(all_y_true, axis=0).ravel()
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    y_pred_all = y_pred_all.reshape(-1, y_pred_all.shape[-1])

    val_cfg = config["validation"]
    report = run_validation(
        y_true_all,
        y_pred_all,
        quantiles=model_cfg["quantiles"],
        vr_threshold=val_cfg["violation_rate_threshold"],
        pvalue_threshold=val_cfg["kupiec_pvalue_threshold"],
    )
    return report
