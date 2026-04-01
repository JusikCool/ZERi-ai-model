import argparse
from pathlib import Path

from model.m3_full_model.dataset import load_data
from model.m3_full_model.train import load_config, run_optuna, train
from validation.kupiec.kupiec import run_validation
from validation.backtest.backtest import rolling_window_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--mode",
        choices=["train", "optuna", "backtest"],
        default="train",
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
        report = rolling_window_backtest(df, config, n_splits=args.n_splits)
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
