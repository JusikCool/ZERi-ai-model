import json
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer

from model.m4.dataset import GROUP_ID, TIME_IDX, get_vix_stats
from model.m4.model import M4FullModel

from .dataset import (
    DEFAULT_ENCODER_LEN,
    DEFAULT_PREDICTION_LEN,
    DEFAULT_VAL_RATIO,
    build_sector_crisis_dataloaders,
    build_sector_dataloaders,
    build_sector_dataset,
)


_orig_get_attention_mask = TemporalFusionTransformer.get_attention_mask


def _patched_get_attention_mask(self, encoder_lengths, **kwargs):
    mask = _orig_get_attention_mask(self, encoder_lengths, **kwargs)
    return mask.to(encoder_lengths.device)


TemporalFusionTransformer.get_attention_mask = _patched_get_attention_mask


QUANTILES: list[float] = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95,
]

TAIL_QUANTILES: set[float] = {0.05, 0.10, 0.90, 0.95}
TAIL_WEIGHT: float = 3.0
QUANTILE_WEIGHTS: list[float] = [
    TAIL_WEIGHT if q in TAIL_QUANTILES else 1.0 for q in QUANTILES
]

VIX_THRESHOLD: float = 25.0
DEFAULT_MAX_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 64
SEED: int = 42

CRISIS_VIX_THRESHOLD: float = 25.0
CRISIS_HIGH_VIX_MULTIPLIER: float = 3.0

GROUP_LABELS = ["low_vol", "mid_vol", "high_vol"]

SECTOR_SAVE_ROOT = Path("model/saved")
LOG_DIR = Path("logs")


def sector_dir(sector_id: str) -> Path:
    p = SECTOR_SAVE_ROOT / f"sector_m4_{sector_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def best_params_path(sector_id: str) -> Path:
    return sector_dir(sector_id) / "best_params.json"


def best_ckpt_path(sector_id: str) -> Path:
    return sector_dir(sector_id) / "best.ckpt"


def _accelerator() -> tuple[str, int]:
    if torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


def _build_model(
    train_ds, params: dict, vix_mean: float, vix_std: float
) -> M4FullModel:
    return M4FullModel.from_dataset(
        dataset=train_ds,
        learning_rate=float(params["learning_rate"]),
        hidden_size=int(params["hidden_size"]),
        attention_head_size=int(params.get("attention_head_size", 4)),
        dropout=float(params["dropout"]),
        quantiles=QUANTILES,
        vix_threshold=VIX_THRESHOLD,
        vix_mean=vix_mean,
        vix_std=vix_std,
        alpha_down=float(params.get("alpha_down", 1.0)),
        beta_down=float(params.get("beta_down", 1.0)),
        alpha_up=float(params.get("alpha_up", 1.0)),
        beta_up=float(params.get("beta_up", 1.0)),
        crossing_weight=float(params.get("crossing_weight", 0.1)),
        use_garch_sigma=True,
        ticker_to_vol_group_idx=None,
        group_labels=GROUP_LABELS,
        quantile_weights=QUANTILE_WEIGHTS,
    )


def _build_loaders(
    sector_id: str,
    train_ds,
    val_ds,
    df,
    batch_size: int,
    crisis_sampling: bool,
):
    if crisis_sampling:
        print(
            f"[sector_m4_{sector_id}] crisis-aware sampler ON "
            f"(VIX>{CRISIS_VIX_THRESHOLD} ×{CRISIS_HIGH_VIX_MULTIPLIER})"
        )
        return build_sector_crisis_dataloaders(
            train_ds, val_ds, df,
            encoder_length=DEFAULT_ENCODER_LEN,
            batch_size=batch_size,
            vix_threshold=CRISIS_VIX_THRESHOLD,
            high_vix_multiplier=CRISIS_HIGH_VIX_MULTIPLIER,
        )
    return build_sector_dataloaders(train_ds, val_ds, batch_size=batch_size)


def run_optuna(
    sector_id: str,
    n_trials: int = 30,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    crisis_sampling: bool = True,
) -> dict:
    pl.seed_everything(SEED, workers=True)
    df, train_ds, val_ds = build_sector_dataset(
        sector_id,
        max_encoder_length=DEFAULT_ENCODER_LEN,
        max_prediction_length=DEFAULT_PREDICTION_LEN,
        val_ratio=DEFAULT_VAL_RATIO,
    )
    vix_mean, vix_std = get_vix_stats(df)
    train_loader, val_loader = _build_loaders(
        sector_id, train_ds, val_ds, df, batch_size, crisis_sampling
    )

    accelerator, devices = _accelerator()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "attention_head_size": trial.suggest_categorical(
                "attention_head_size", [1, 2, 4]
            ),
            "dropout": trial.suggest_float("dropout", 0.05, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "crossing_weight": trial.suggest_float("crossing_weight", 1e-3, 1.0, log=True),
            "alpha_down": trial.suggest_float("alpha_down", 0.5, 3.0),
            "beta_down": trial.suggest_float("beta_down", 0.5, 3.0),
            "alpha_up": trial.suggest_float("alpha_up", 0.5, 3.0),
            "beta_up": trial.suggest_float("beta_up", 0.5, 3.0),
        }
        model = _build_model(train_ds, params, vix_mean, vix_std)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min")],
            enable_progress_bar=True,
            logger=pl.loggers.CSVLogger(LOG_DIR, name=f"sector_m4_{sector_id}_optuna"),
            gradient_clip_val=0.1,
            accelerator=accelerator,
            devices=devices,
        )
        trainer.fit(model, train_loader, val_loader)
        val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))

        trial.report(val_loss, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return val_loss

    study = optuna.create_study(
        direction="minimize",
        study_name=f"sector_m4_{sector_id}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = dict(study.best_params)
    best["val_loss"] = float(study.best_value)

    out_path = best_params_path(sector_id)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print(f"\n[Optuna - sector_m4_{sector_id}] 최적 파라미터 저장: {out_path}")
    for k, v in best.items():
        print(f"  {k}: {v}")
    return best


def train_with_params(
    sector_id: str,
    params: dict,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    crisis_sampling: bool = True,
) -> Path:
    pl.seed_everything(SEED, workers=True)
    df, train_ds, val_ds = build_sector_dataset(
        sector_id,
        max_encoder_length=DEFAULT_ENCODER_LEN,
        max_prediction_length=DEFAULT_PREDICTION_LEN,
        val_ratio=DEFAULT_VAL_RATIO,
    )
    vix_mean, vix_std = get_vix_stats(df)
    train_loader, val_loader = _build_loaders(
        sector_id, train_ds, val_ds, df, batch_size, crisis_sampling
    )

    model = _build_model(train_ds, params, vix_mean, vix_std)

    accelerator, devices = _accelerator()

    out_dir = sector_dir(sector_id)
    checkpoint_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20, mode="min"),
            checkpoint_cb,
        ],
        enable_progress_bar=True,
        logger=pl.loggers.CSVLogger(LOG_DIR, name=f"sector_m4_{sector_id}_final"),
        gradient_clip_val=0.1,
        accelerator=accelerator,
        devices=devices,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = Path(checkpoint_cb.best_model_path)
    final_ckpt = best_ckpt_path(sector_id)
    if best_ckpt.exists() and best_ckpt.resolve() != final_ckpt.resolve():
        if final_ckpt.exists():
            final_ckpt.unlink()
        best_ckpt.replace(final_ckpt)
    print(f"\n[sector_m4_{sector_id}] 체크포인트 저장: {final_ckpt}")
    print(f"[sector_m4_{sector_id}] val_loss: {checkpoint_cb.best_model_score:.6f}")
    return final_ckpt


def load_best_params(sector_id: str) -> dict:
    path = best_params_path(sector_id)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 run_optuna 를 실행하세요."
        )
    with open(path, encoding="utf-8") as f:
        params = json.load(f)
    params.pop("val_loss", None)
    return params


def load_trained_model(sector_id: str):
    params = load_best_params(sector_id)
    df, train_ds, val_ds = build_sector_dataset(
        sector_id,
        max_encoder_length=DEFAULT_ENCODER_LEN,
        max_prediction_length=DEFAULT_PREDICTION_LEN,
        val_ratio=DEFAULT_VAL_RATIO,
    )
    vix_mean, vix_std = get_vix_stats(df)
    model = _build_model(train_ds, params, vix_mean, vix_std)

    ckpt_path = best_ckpt_path(sector_id)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} 가 없습니다. 먼저 train_with_params 를 실행하세요."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, df, train_ds, val_ds
