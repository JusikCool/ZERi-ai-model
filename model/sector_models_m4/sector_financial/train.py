import argparse

from model.sector_models_m4.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_EPOCHS,
    load_best_params,
    run_optuna,
    train_with_params,
)
from model.sector_models_m4.validate import evaluate_sector

SECTOR_ID = "financial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--skip_optuna", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no_crisis_sampling", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    args = parser.parse_args()
    crisis_sampling = not args.no_crisis_sampling

    if args.skip_optuna:
        params = load_best_params(SECTOR_ID)
    else:
        params = run_optuna(
            SECTOR_ID,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            crisis_sampling=crisis_sampling,
        )
        params = {k: v for k, v in params.items() if k != "val_loss"}

    train_with_params(
        SECTOR_ID,
        params,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        crisis_sampling=crisis_sampling,
    )

    if not args.no_eval:
        per_ticker, sector_level = evaluate_sector(SECTOR_ID)
        print(f"\n[m4_{SECTOR_ID}] 섹터 종합")
        print(sector_level.to_string(index=False))
        print(f"\n[m4_{SECTOR_ID}] 종목별")
        print(per_ticker.to_string(index=False))


if __name__ == "__main__":
    main()
