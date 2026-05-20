import argparse
from pathlib import Path

from model.m4.dataset import load_data
from model.m3_full_model.train import load_config, run_optuna, train
from validation.kupiec.kupiec import run_validation
from validation.backtest.backtest_m4 import rolling_window_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--mode",
        choices=["train", "optuna", "backtest"],
        default="train",
    )
    # ===== M4 추가: backtest 시 어느 모델 모드를 평가할지 =====
    parser.add_argument(
        "--backtest_mode",
        choices=["m4_garch", "m4_combined"],
        default="m4_combined",
        help="backtest 모드 — 학습한 모델과 동일해야 함",
    )
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_splits", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.mode == "train":
        val_loss = train(config)
        print(f"val_loss: {val_loss:.6f}")

    elif args.mode == "optuna":
        study = run_optuna(n_trials=args.n_trials, config_path=args.config)
        print(study.best_params)

    elif args.mode == "backtest":
        df = load_data()
        report = rolling_window_backtest(
            df, config, n_splits=args.n_splits, mode=args.backtest_mode
        )
        out_path = Path(f"model/saved/backtest_results_{args.backtest_mode}.csv")
        report.to_csv(out_path, index=False)
        print(f"\n결과 저장: {out_path}")
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()