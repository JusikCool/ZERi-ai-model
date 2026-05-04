"""
GPU 서버용 전체 파이프라인 (NHiTS 버전)
  1단계: Optuna 30 trials → 최적 하이퍼파라미터 탐색
  2단계: 최적 파라미터로 재학습
  3단계: Rolling Window 백테스트 + Kupiec 결과 출력

실행:
  python run_full_pipeline_nhits.py
  python run_full_pipeline_nhits.py --n_trials 50   # trial 수 조절
  python run_full_pipeline_nhits.py --skip_optuna   # Optuna 생략, config.yaml 값으로 바로 학습
"""

import argparse
import json
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model.m3_full_model.dataset import (
    build_dataset,
    build_dataloaders,
    get_vix_stats,
    load_data,
)
from model.m3_full_model.nhits_model import M3FullModel
from validation.backtest.nhits_backtest import rolling_window_backtest, covid_backtest

CONFIG_PATH = Path("configs/config.yaml")
CHECKPOINT_DIR = Path("model/saved")
LOG_DIR = Path("logs")
BEST_PARAMS_PATH = Path("model/saved/best_params_nhits.json")


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_optuna(config: dict, n_trials: int) -> dict:
    data_cfg = config["data"]
    model_cfg = config["model"]

    df = load_data()
    vix_mean, vix_std = get_vix_stats(df)
    train_ds, val_ds = build_dataset(
        df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=model_cfg["batch_size"])

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        alpha_down = trial.suggest_float("alpha_down", 0.5, 3.0)
        beta_down = trial.suggest_float("beta_down", 0.5, 3.0)
        alpha_up = trial.suggest_float("alpha_up", 0.5, 3.0)
        beta_up = trial.suggest_float("beta_up", 0.5, 3.0)
        crossing_weight = trial.suggest_float("crossing_weight", 0.01, 0.5)

        model = M3FullModel.from_dataset(
            dataset=train_ds,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            n_layers=n_layers,
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

        trainer = pl.Trainer(
            max_epochs=model_cfg["max_epochs"],
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min")],
            enable_progress_bar=False,
            logger=pl.loggers.CSVLogger(LOG_DIR, name="optuna_trials_nhits"),
            gradient_clip_val=0.1,
            accelerator="gpu",
            devices=1,
        )
        trainer.fit(model, train_loader, val_loader)

        val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))

        trial.report(val_loss, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_loss

    study = optuna.create_study(
        direction="minimize",
        study_name="m3_full_model_nhits",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["val_loss"] = study.best_value

    print("\n" + "=" * 50)
    print("[Optuna 완료 - NHiTS] 최적 파라미터")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print("=" * 50 + "\n")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"최적 파라미터 저장: {BEST_PARAMS_PATH}")

    return best


def train_with_params(config: dict, params: dict) -> Path:
    data_cfg = config["data"]
    model_cfg = config["model"]

    df = load_data()
    vix_mean, vix_std = get_vix_stats(df)
    train_ds, val_ds = build_dataset(
        df,
        max_encoder_length=data_cfg["window_size"],
        max_prediction_length=data_cfg["horizon"],
    )
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=model_cfg["batch_size"])

    model = M3FullModel.from_dataset(
        dataset=train_ds,
        learning_rate=params.get("learning_rate", model_cfg["learning_rate"]),
        hidden_size=params.get("hidden_size", model_cfg["hidden_size"]),
        n_layers=params.get("n_layers", model_cfg.get("n_layers", 2)),
        dropout=params.get("dropout", model_cfg["dropout"]),
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
        vix_mean=vix_mean,
        vix_std=vix_std,
        alpha_down=params.get("alpha_down", 1.0),
        beta_down=params.get("beta_down", 1.0),
        alpha_up=params.get("alpha_up", 1.0),
        beta_up=params.get("beta_up", 1.0),
        crossing_weight=params.get("crossing_weight", 0.1),
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="m3_nhits_best_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=model_cfg["max_epochs"],
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20, mode="min"),
            checkpoint_cb,
        ],
        enable_progress_bar=True,
        logger=pl.loggers.CSVLogger(LOG_DIR, name="m3_final_nhits"),
        gradient_clip_val=0.1,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = Path(checkpoint_cb.best_model_path)
    print(f"\n최종 체크포인트: {best_ckpt}")
    print(f"최종 val_loss: {checkpoint_cb.best_model_score:.6f}")
    return best_ckpt


def run_backtest(config: dict) -> None:
    df = load_data()
    report = rolling_window_backtest(df, config, n_splits=5)

    print("\n" + "=" * 60)
    print("[백테스트 결과 - NHiTS] Rolling Window 5-Fold Kupiec POF Test")
    print("=" * 60)
    print(report.to_string(index=False))

    vr_pass_count = report["vr_pass"].sum()
    kupiec_pass_count = report["kupiec_pass"].sum()
    total = len(report)
    print(f"\nViolation Rate Pass: {vr_pass_count}/{total}")
    print(f"Kupiec Pass:         {kupiec_pass_count}/{total}")

    results_path = Path("model/saved/backtest_results_nhits.csv")
    report.to_csv(results_path, index=False)
    print(f"\n결과 저장: {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--skip_optuna", action="store_true")
    args = parser.parse_args()

    config = load_config()

    if args.skip_optuna:
        model_cfg = config["model"]
        params = {
            "hidden_size": model_cfg["hidden_size"],
            "n_layers": model_cfg.get("n_layers", 2),
            "dropout": model_cfg["dropout"],
            "learning_rate": model_cfg["learning_rate"],
            "alpha_down": 1.0,
            "beta_down": 1.0,
            "alpha_up": 1.0,
            "beta_up": 1.0,
            "crossing_weight": 0.1,
        }
        print("[Optuna 생략] config.yaml 기본값으로 학습합니다.")
    else:
        print(f"[1단계] Optuna 탐색 시작 ({args.n_trials} trials)")
        params = run_optuna(config, n_trials=args.n_trials)

    print("\n[2단계] 최적 파라미터로 재학습")
    train_with_params(config, params)

    print("\n[3단계] 백테스트 실행")
    run_backtest(config)

    print("\n[4단계] COVID 구간 백테스트 (2020-01-01 ~ 2021-01-01)")
    df = load_data()
    covid_report = covid_backtest(df, config)
    print("\n" + "=" * 60)
    print("[COVID 구간 백테스트 결과 - NHiTS] Kupiec POF Test")
    print("=" * 60)
    print(covid_report.to_string(index=False))

    covid_vr_pass = covid_report["vr_pass"].sum()
    covid_kupiec_pass = covid_report["kupiec_pass"].sum()
    covid_total = len(covid_report)
    print(f"\nViolation Rate Pass: {covid_vr_pass}/{covid_total}")
    print(f"Kupiec Pass:         {covid_kupiec_pass}/{covid_total}")

    covid_results_path = Path("model/saved/covid_backtest_results_nhits.csv")
    covid_report.to_csv(covid_results_path, index=False)
    print(f"\n결과 저장: {covid_results_path}")

    print("\n[5단계] 우크라이나 전쟁 + 인플레이션 구간 백테스트 (2022-02-01 ~ 2022-07-01)")
    ukraine_report = ukraine_inflation_backtest(df, config)
    print("\n" + "=" * 60)
    print("[우크라이나 전쟁 + 인플레이션 구간 백테스트 결과 - NHiTS] Kupiec POF Test")
    print("=" * 60)
    print(ukraine_report.to_string(index=False))
 
    ukraine_vr_pass = ukraine_report["vr_pass"].sum()
    ukraine_kupiec_pass = ukraine_report["kupiec_pass"].sum()
    ukraine_total = len(ukraine_report)
    print(f"\nViolation Rate Pass: {ukraine_vr_pass}/{ukraine_total}")
    print(f"Kupiec Pass:         {ukraine_kupiec_pass}/{ukraine_total}")
 
    ukraine_results_path = Path("model/saved/ukraine_inflation_backtest_results.csv")
    ukraine_report.to_csv(ukraine_results_path, index=False)
    print(f"\n결과 저장: {ukraine_results_path}")
 
    print("\n[6단계] 트럼프 관세 충격 구간 백테스트 (2025-04-01 ~ 2025-05-31)")
    tariff_report = trump_tariff_backtest(df, config)
    print("\n" + "=" * 60)
    print("[트럼프 관세 충격 구간 백테스트 결과 - NHiTS] Kupiec POF Test")
    print("=" * 60)
    print(tariff_report.to_string(index=False))
 
    tariff_vr_pass = tariff_report["vr_pass"].sum()
    tariff_kupiec_pass = tariff_report["kupiec_pass"].sum()
    tariff_total = len(tariff_report)
    print(f"\nViolation Rate Pass: {tariff_vr_pass}/{tariff_total}")
    print(f"Kupiec Pass:         {tariff_kupiec_pass}/{tariff_total}")
 
    tariff_results_path = Path("model/saved/trump_tariff_backtest_results.csv")
    tariff_report.to_csv(tariff_results_path, index=False)
    print(f"\n결과 저장: {tariff_results_path}")

    print("\n전체 파이프라인 완료 (NHiTS).")


if __name__ == "__main__":
    main()