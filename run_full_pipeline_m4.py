"""
GPU 서버용 전체 파이프라인 (M4 확장)
  1단계: Optuna 30 trials → 최적 하이퍼파라미터 탐색
  2단계: 최적 파라미터로 재학습
  3단계: Rolling Window 백테스트 + Kupiec/CC/DQ 결과 출력
  4~6단계: COVID / 우크라이나 / 트럼프 관세 구간 백테스트

실행:
  python run_full_pipeline.py --mode m4_garch                  # M4-GARCH only
  python run_full_pipeline.py --mode m4_combined --n_trials 60 # M4-Combined (권장 trial ↑)
  python run_full_pipeline.py --mode m4_garch --skip_optuna    # Optuna 생략

ablation 비교:
  python run_full_pipeline.py --mode m4_garch
  python run_full_pipeline.py --mode m4_combined --n_trials 60
  → 두 모드 결과 모두 model/saved/ 에 저장됨 (파일명에 mode 접두)
"""

import argparse
import json
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer

_orig_get_attention_mask = TemporalFusionTransformer.get_attention_mask

def _patched_get_attention_mask(self, encoder_lengths, **kwargs):
    mask = _orig_get_attention_mask(self, encoder_lengths, **kwargs)
    return mask.to(encoder_lengths.device)

TemporalFusionTransformer.get_attention_mask = _patched_get_attention_mask

from model.m4.dataset import (
    build_dataset,
    build_dataloaders,
    get_vix_stats,
    load_data,
)
from model.m4.model import M4FullModel
from validation.backtest.backtest_m4 import (
    rolling_window_backtest,
    covid_backtest,
    ukraine_inflation_backtest,
    trump_tariff_backtest,
)

CONFIG_PATH = Path("configs/config.yaml")
CHECKPOINT_DIR = Path("model/saved")
LOG_DIR = Path("logs")

# ===== M4: vol_group 매핑 파일 위치 =====
VOL_GROUP_MAP_PATH = Path("vol_group_map.json")
# 또는 data/raw/vol_group_map.json — 사용자 환경에 따라 조정

# ===== M4: vol_group 라벨 (loss.py group_labels 와 일치) =====
GROUP_LABELS = ["low_vol", "mid_vol", "high_vol"]


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_vol_group_map() -> dict:
    """vol_group_map.json 로드. 파일 없으면 None."""
    if not VOL_GROUP_MAP_PATH.exists():
        return None
    with open(VOL_GROUP_MAP_PATH, encoding="utf-8") as f:
        return json.load(f)


def _get_paths(mode: str) -> dict:
    """모드별 파일 경로 dict 반환."""
    return {
        "best_params": CHECKPOINT_DIR / f"best_params_{mode}.json",
        "ckpt_pattern": f"{mode}_best_{{epoch:02d}}_{{val_loss:.4f}}",
        "backtest_csv": Path(f"model/saved/backtest_results_{mode}.csv"),
        "covid_csv": Path(f"model/saved/covid_backtest_results_{mode}.csv"),
        "ukraine_csv": Path(f"model/saved/ukraine_inflation_backtest_results_{mode}.csv"),
        "tariff_csv": Path(f"model/saved/trump_tariff_backtest_results_{mode}.csv"),
        "optuna_log": f"optuna_trials_{mode}",
        "final_log": f"final_{mode}",
    }


def _build_from_dataset_kwargs(
    params: dict,
    config: dict,
    train_ds,
    vix_mean: float,
    vix_std: float,
    mode: str,
    vol_group_map: dict = None,
) -> dict:
    """
    mode 에 따라 M4FullModel.from_dataset 의 인자 dict 구성.
      m4_garch    : use_garch_sigma=True + scalar α/β
      m4_combined : use_garch_sigma=True + group 별 α/β + vol_group_map
    """
    model_cfg = config["model"]

    common = dict(
        dataset=train_ds,
        learning_rate=params.get("learning_rate", model_cfg["learning_rate"]),
        hidden_size=params.get("hidden_size", model_cfg["hidden_size"]),
        attention_head_size=params.get(
            "attention_head_size", model_cfg["attention_head_size"]
        ),
        dropout=params.get("dropout", model_cfg["dropout"]),
        quantiles=model_cfg["quantiles"],
        vix_threshold=model_cfg["vix_threshold"],
        vix_mean=vix_mean,
        vix_std=vix_std,
        crossing_weight=params.get("crossing_weight", 0.1),
        use_garch_sigma=True,            # 두 모드 모두 GARCH
        group_labels=GROUP_LABELS,
    )

    if mode == "m4_garch":
        # Scalar α/β (M3 동일, σ 자리만 GARCH)
        common.update(
            alpha_down=params.get("alpha_down", 1.0),
            beta_down=params.get("beta_down", 1.0),
            alpha_up=params.get("alpha_up", 1.0),
            beta_up=params.get("beta_up", 1.0),
            vol_group_map=None,           # M4-GARCH only mode
        )
    elif mode == "m4_combined":
        # Group 별 α/β dict 구성
        alpha_down_by_group = {
            g: params[f"alpha_down_{g}"] for g in GROUP_LABELS
        }
        beta_down_by_group = {
            g: params[f"beta_down_{g}"] for g in GROUP_LABELS
        }
        alpha_up_by_group = {
            g: params[f"alpha_up_{g}"] for g in GROUP_LABELS
        }
        beta_up_by_group = {
            g: params[f"beta_up_{g}"] for g in GROUP_LABELS
        }

        if vol_group_map is None:
            raise ValueError(
                "m4_combined 모드에 vol_group_map 이 필요. "
                f"{VOL_GROUP_MAP_PATH} 가 없거나 비어있음."
            )

        common.update(
            alpha_down_by_group=alpha_down_by_group,
            beta_down_by_group=beta_down_by_group,
            alpha_up_by_group=alpha_up_by_group,
            beta_up_by_group=beta_up_by_group,
            vol_group_map=vol_group_map,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return common


def run_optuna(
    config: dict, n_trials: int, mode: str, vol_group_map: dict = None
) -> dict:
    data_cfg = config["data"]
    model_cfg = config["model"]
    paths = _get_paths(mode)

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

    def objective(trial: optuna.Trial) -> float:
        # 공통 hyperparameter
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "attention_head_size": trial.suggest_categorical(
                "attention_head_size", [1, 2, 4]
            ),
            "dropout": trial.suggest_float("dropout", 0.05, 0.3),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 5e-3, log=True
            ),
            "crossing_weight": trial.suggest_float(
                "crossing_weight", 1e-3, 1.0, log=True
            ),
        }

        # 모드별 α/β
        if mode == "m4_garch":
            params["alpha_down"] = trial.suggest_float("alpha_down", 0.5, 3.0)
            params["beta_down"] = trial.suggest_float("beta_down", 0.5, 3.0)
            params["alpha_up"] = trial.suggest_float("alpha_up", 0.5, 3.0)
            params["beta_up"] = trial.suggest_float("beta_up", 0.5, 3.0)
        elif mode == "m4_combined":
            # group 별 α/β (12 sectoral params)
            for g in GROUP_LABELS:
                params[f"alpha_down_{g}"] = trial.suggest_float(
                    f"alpha_down_{g}", 0.5, 3.0
                )
                params[f"beta_down_{g}"] = trial.suggest_float(
                    f"beta_down_{g}", 0.5, 3.0
                )
                params[f"alpha_up_{g}"] = trial.suggest_float(
                    f"alpha_up_{g}", 0.5, 3.0
                )
                params[f"beta_up_{g}"] = trial.suggest_float(
                    f"beta_up_{g}", 0.5, 3.0
                )

        from_dataset_kwargs = _build_from_dataset_kwargs(
            params, config, train_ds, vix_mean, vix_std, mode, vol_group_map
        )
        model = M4FullModel.from_dataset(**from_dataset_kwargs)

        trainer = pl.Trainer(
            max_epochs=model_cfg["max_epochs"],
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min")],
            enable_progress_bar=True,
            logger=pl.loggers.CSVLogger(LOG_DIR, name=paths["optuna_log"]),
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
        study_name=f"m4_{mode}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["val_loss"] = study.best_value

    print("\n" + "=" * 50)
    print(f"[Optuna 완료] mode={mode} 최적 파라미터")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print("=" * 50 + "\n")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths["best_params"], "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"최적 파라미터 저장: {paths['best_params']}")

    return best


def train_with_params(
    config: dict, params: dict, mode: str, vol_group_map: dict = None
) -> Path:
    data_cfg = config["data"]
    model_cfg = config["model"]
    paths = _get_paths(mode)

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

    from_dataset_kwargs = _build_from_dataset_kwargs(
        params, config, train_ds, vix_mean, vix_std, mode, vol_group_map
    )
    model = M4FullModel.from_dataset(**from_dataset_kwargs)

    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=paths["ckpt_pattern"],
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
        logger=pl.loggers.CSVLogger(LOG_DIR, name=paths["final_log"]),
        gradient_clip_val=0.1,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = Path(checkpoint_cb.best_model_path)
    print(f"\n[mode={mode}] 최종 체크포인트: {best_ckpt}")
    print(f"최종 val_loss: {checkpoint_cb.best_model_score:.6f}")
    return best_ckpt


def _print_and_save_report(report, mode: str, title: str, csv_path: Path) -> None:
    """공통 백테스트 결과 출력 + 저장."""
    print("\n" + "=" * 60)
    print(f"[{title}] (mode={mode})")
    print("=" * 60)
    print(report.to_string(index=False))

    vr_pass = report["vr_pass"].sum() if "vr_pass" in report.columns else "N/A"
    kupiec_pass = report["kupiec_pass"].sum() if "kupiec_pass" in report.columns else "N/A"
    total = len(report)
    print(f"\nViolation Rate Pass: {vr_pass}/{total}")
    print(f"Kupiec Pass:         {kupiec_pass}/{total}")

    report.to_csv(csv_path, index=False)
    print(f"\n결과 저장: {csv_path}")


def run_backtest(config: dict, mode: str) -> None:
    """3~6단계 백테스트 통합 실행. 모드별 CSV 파일명 분리."""
    paths = _get_paths(mode)
    df = load_data()

    # 3단계: Rolling Window
    report = rolling_window_backtest(df, config, n_splits=5)
    _print_and_save_report(
        report, mode,
        "Rolling Window 5-Fold Backtest (Kupiec POF + Extended CC/DQ in 콘솔)",
        paths["backtest_csv"],
    )

    # 4단계: COVID
    covid_report = covid_backtest(df, config)
    _print_and_save_report(
        covid_report, mode,
        "COVID 구간 (2020-01-01 ~ 2021-01-01)",
        paths["covid_csv"],
    )

    # 5단계: 우크라이나 + 인플레이션
    ukraine_report = ukraine_inflation_backtest(df, config)
    _print_and_save_report(
        ukraine_report, mode,
        "우크라이나 전쟁 + 인플레이션 구간 (2022-02-01 ~ 2022-07-01)",
        paths["ukraine_csv"],
    )

    # 6단계: 트럼프 관세
    tariff_report = trump_tariff_backtest(df, config)
    _print_and_save_report(
        tariff_report, mode,
        "트럼프 관세 충격 구간 (2025-04-01 ~ 2025-05-31)",
        paths["tariff_csv"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["m4_garch", "m4_combined"],
        default="m4_garch",
        help="m4_garch: scalar α/β, m4_combined: group 별 α/β",
    )
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--skip_optuna", action="store_true")
    args = parser.parse_args()

    mode = args.mode
    print(f"\n{'='*60}")
    print(f"  M4 Pipeline — mode = {mode}")
    print(f"{'='*60}\n")

    # m4_combined 일 때 trial 권장 사항
    if mode == "m4_combined" and args.n_trials < 50 and not args.skip_optuna:
        print(
            f"⚠️ mode=m4_combined 은 17개 hyperparameter 를 탐색합니다.\n"
            f"   현재 n_trials={args.n_trials}. 50~80 권장. 계속하려면 Enter, 중단하려면 Ctrl+C."
        )
        try:
            input()
        except KeyboardInterrupt:
            return

    config = load_config()
    vol_group_map = load_vol_group_map()

    if mode == "m4_combined":
        if vol_group_map is None:
            print(
                f"❌ {VOL_GROUP_MAP_PATH} 가 없습니다. "
                f"먼저 build_dataset_50tickers.py 를 실행하여 vol_group_map.json 을 생성하세요."
            )
            return
        print(f"  vol_group_map 로드: {len(vol_group_map)} tickers")

    if args.skip_optuna:
        model_cfg = config["model"]
        base_params = {
            "hidden_size": model_cfg["hidden_size"],
            "attention_head_size": model_cfg["attention_head_size"],
            "dropout": model_cfg["dropout"],
            "learning_rate": model_cfg["learning_rate"],
            "crossing_weight": 0.1,
        }
        if mode == "m4_garch":
            base_params.update(
                alpha_down=1.0, beta_down=1.0, alpha_up=1.0, beta_up=1.0,
            )
        else:  # m4_combined
            for g in GROUP_LABELS:
                base_params[f"alpha_down_{g}"] = 1.0
                base_params[f"beta_down_{g}"] = 1.0
                base_params[f"alpha_up_{g}"] = 1.0
                base_params[f"beta_up_{g}"] = 1.0
        params = base_params
        print(f"[Optuna 생략] config.yaml + 기본값 (α/β=1.0) 으로 학습.")
    else:
        print(f"[1단계] Optuna 탐색 시작 ({args.n_trials} trials, mode={mode})")
        params = run_optuna(config, args.n_trials, mode, vol_group_map)

    print(f"\n[2단계] 최적 파라미터로 재학습 (mode={mode})")
    train_with_params(config, params, mode, vol_group_map)

    print(f"\n[3~6단계] 백테스트 실행 (mode={mode})")
    run_backtest(config, mode)

    print(f"\n전체 파이프라인 완료 (mode={mode}).")


if __name__ == "__main__":
    main()
