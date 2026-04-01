import pandas as pd
import torch
import optuna

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

class AdaptivePinballLoss(QuantileLoss):
    def __init__(
        self,
        quantiles: list = [0.1, 0.5, 0.9],
        alpha_down: float = 1.0,
        alpha_up: float = 2.0,
        vix_threshold: float = 20.0,
        crossing_weight: float = 0.1,
    ):
        super().__init__(quantiles=quantiles)
        self.alpha_down = alpha_down
        self.alpha_up = alpha_up
        self.vix_threshold = vix_threshold
        self.crossing_weight = crossing_weight

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []

        # --- per-quantile pinball loss ---
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            pinball = torch.where(errors >= 0, q * errors, (q - 1) * errors)
            losses.append(pinball)

        losses = torch.stack(losses, dim=-1)  # (batch, time, n_q)

        if len(self.quantiles) >= 2:
            pred_spread = (y_pred[..., -1] - y_pred[..., 0]).detach()  # (batch, time)

            spread_normalized = torch.sigmoid(pred_spread * 50)
            weight = (
                spread_normalized * self.alpha_up
                + (1 - spread_normalized) * self.alpha_down
            ).unsqueeze(-1)  # (batch, time, 1)
        else:
            weight = self.alpha_up

        weighted_loss = (weight * losses).mean()

        # --- quantile crossing penalty ---
        crossing_loss = 0.0
        for i in range(len(self.quantiles) - 1):
            crossing = torch.relu(y_pred[..., i] - y_pred[..., i + 1])
            crossing_loss += crossing.mean()
        crossing_loss *= self.crossing_weight

        return weighted_loss + crossing_loss

MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 10
DATA_CSV_PATH = "./tft_processed_panel_v1.csv"

df = pd.read_csv(DATA_CSV_PATH, parse_dates=["Date"])
df = df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)
df["group_id"] = df["group_id"].astype(str)

df["Month"] = df["Month"].astype(str)
df["Day_of_Week"] = df["Day_of_Week"].astype(str)

df = df.dropna(subset=["Target_Return_5d"]).reset_index(drop=True)
df["time_idx"] = df.groupby("group_id").cumcount()

training_cutoff = int(df["time_idx"].max() * 0.8)

unknown_reals = [
    "Open", "High", "Low", "Close", "Volume",
    "NASDAQ_Close", "VIX_Close",
    "FEDFUNDS", "UNRATE", "DTWEXBGS",
    "CPIAUCSL", "PCEPI", "GDP", "M2SL",
    "GS10", "T10Y2Y", "PAYEMS", "CSUSHPISA", "INDPRO",
    "RSI_14", "ATR_14", "SMA_20",
    "Returns", "Realized_Vol_20d",
]

train_dataset = TimeSeriesDataSet(
    df[df["time_idx"] <= training_cutoff],
    time_idx="time_idx",
    target="Target_Return_5d",
    group_ids=["group_id"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    static_categoricals=["group_id"],
    time_varying_known_categoricals=["Month", "Day_of_Week"],
    time_varying_unknown_reals=unknown_reals,
    target_normalizer=GroupNormalizer(
        groups=["group_id"],
        transformation="softplus",
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

val_dataset = TimeSeriesDataSet.from_dataset(
    train_dataset,
    df[df["time_idx"] > training_cutoff - MAX_ENCODER_LENGTH],
    predict=True,
    stop_randomization=True,
)

train_dataloader = train_dataset.to_dataloader(
    train=True,
    batch_size=64,
    num_workers=0
)
val_dataloader = val_dataset.to_dataloader(
    train=False,
    batch_size=128,
    num_workers=0
)

def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    alpha_down = trial.suggest_float("alpha_down", 0.5, 3.0)
    alpha_up = trial.suggest_float("alpha_up", 0.5, 3.0)
    crossing_weight = trial.suggest_float("crossing_weight", 0.01, 0.5)

    hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [8, 16, 32])

    if hidden_size % attention_head_size != 0:
        raise optuna.TrialPruned()

    loss = AdaptivePinballLoss(
        quantiles=[0.1, 0.5, 0.9],
        alpha_down=alpha_down,
        alpha_up=alpha_up,
        vix_threshold=20.0,
        crossing_weight=crossing_weight,
    )

    model = TemporalFusionTransformer.from_dataset(
        dataset=train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=loss,
        output_size=3,
        log_interval=10,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min"
        ),
        ModelCheckpoint(
            dirpath=f"./model/m2/trial_{trial.number}",
            filename="m2_{epoch:02d}_{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )
    ]

    trainer = Trainer(
        max_epochs=50,
        callbacks=callbacks,
        enable_progress_bar=True,
        gradient_clip_val=0.1,
        accelerator="gpu",
        devices=1,
        logger=True,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf")))
    return val_loss.item()

study = optuna.create_study(
    direction="minimize",
    study_name="m2",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5
    ),
)

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True,
)

print("\n" + "="*50)
print("Best Trial:")
print(f"  val_loss: {study.best_value:.6f}")
print("  Params:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")

best = study.best_params

loss = AdaptivePinballLoss(
    quantiles=[0.1, 0.5, 0.9],
    alpha_down=best["alpha_down"],
    alpha_up=best["alpha_up"],
    vix_threshold=20.0,
    crossing_weight=best["crossing_weight"],
)

final_model = TemporalFusionTransformer.from_dataset(
    dataset=train_dataset,
    learning_rate=best["learning_rate"],
    hidden_size=best["hidden_size"],
    attention_head_size=best["attention_head_size"],
    dropout=best["dropout"],
    hidden_continuous_size=best.get("hidden_continuous_size", 16),
    loss=loss,
    output_size=3,
    log_interval=10,
)

final_callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
    ),
    ModelCheckpoint(
        dirpath="./model/m2",
        filename="m2_final_best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    ),
    LearningRateMonitor(logging_interval="epoch"),
]

final_trainer = Trainer(
    max_epochs=50,
    callbacks=final_callbacks,
    enable_progress_bar=True,
    gradient_clip_val=0.1,
    accelerator="gpu",
    devices=1,
)

final_trainer.fit(final_model, train_dataloader, val_dataloader)

print(f"val_loss: {final_trainer.callback_metrics['val_loss'].item():.6f}")