import argparse
from pathlib import Path

import pandas as pd

from model.sector_models_m4.sectors import SECTOR_IDS
from model.sector_models_m4.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_EPOCHS,
    load_best_params,
    run_optuna,
    train_with_params,
)
from model.sector_models_m4.validate import evaluate_sector

RESULTS_DIR = Path("model/saved")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument(
        "--sectors", nargs="*", default=None,
        help=f"실행할 섹터 ID 목록 (기본: 전체 {len(SECTOR_IDS)}개)",
    )
    parser.add_argument(
        "--skip_optuna", action="store_true",
        help="저장된 best_params.json 으로 학습 단계부터 시작",
    )
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--no_crisis_sampling", action="store_true",
        help="VIX>25 윈도우 oversampling 끔 (기본: on)",
    )
    parser.add_argument(
        "--no_eval", action="store_true",
        help="섹터별 학습 후 Kupiec 검증 생략",
    )
    return parser.parse_args()


def _print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _run_sector(
    sector_id: str,
    n_trials: int,
    skip_optuna: bool,
    max_epochs: int,
    batch_size: int,
    crisis_sampling: bool,
    no_eval: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    _print_header(f"[섹터 m4_{sector_id}] 파이프라인 시작")

    if skip_optuna:
        params = load_best_params(sector_id)
        print(f"[1단계 생략] best_params.json 로드 → 학습 단계부터 시작")
    else:
        print(f"[1단계] Optuna 탐색 ({n_trials} trials)")
        params = run_optuna(
            sector_id,
            n_trials=n_trials,
            max_epochs=max_epochs,
            batch_size=batch_size,
            crisis_sampling=crisis_sampling,
        )
        params = {k: v for k, v in params.items() if k != "val_loss"}

    print(f"\n[2단계] sector_m4_{sector_id} 최종 재학습")
    train_with_params(
        sector_id,
        params,
        max_epochs=max_epochs,
        batch_size=batch_size,
        crisis_sampling=crisis_sampling,
    )

    if no_eval:
        return None, None

    print(f"\n[3단계] sector_m4_{sector_id} 검증")
    per_ticker, sector_level = evaluate_sector(sector_id)
    print(f"\n[m4_{sector_id}] 종목별 결과")
    print(per_ticker.to_string(index=False))
    print(f"\n[m4_{sector_id}] 섹터 종합 결과")
    print(sector_level.to_string(index=False))
    return per_ticker, sector_level


def _summarize(reports_per_ticker: list[pd.DataFrame]) -> None:
    if not reports_per_ticker:
        return
    combined = pd.concat(reports_per_ticker, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "sector_m4_validation_all.csv", index=False)

    _print_header("[전체 종합] Q0.10 Violation Rate (목표 0.05~0.15)")
    q10 = combined[combined["quantile"] == 0.10].copy()
    cols = ["sector", "group_id", "violation_rate", "p_value", "vr_pass", "kupiec_pass"]
    print(q10[cols].to_string(index=False))

    _print_header("[섹터별 평균] Q0.10 Violation Rate")
    sector_avg = (
        q10.groupby("sector")
        .agg(
            n_tickers=("group_id", "nunique"),
            mean_vr=("violation_rate", "mean"),
            std_vr=("violation_rate", "std"),
            vr_pass_count=("vr_pass", "sum"),
            kupiec_pass_count=("kupiec_pass", "sum"),
        )
        .reset_index()
    )
    print(sector_avg.to_string(index=False))
    sector_avg.to_csv(RESULTS_DIR / "sector_m4_validation_summary_q10.csv", index=False)
    print(f"\n저장: {RESULTS_DIR / 'sector_m4_validation_all.csv'}")
    print(f"저장: {RESULTS_DIR / 'sector_m4_validation_summary_q10.csv'}")


def main() -> None:
    args = _parse_args()
    crisis_sampling = not args.no_crisis_sampling

    targets = args.sectors or list(SECTOR_IDS)
    unknown = [s for s in targets if s not in SECTOR_IDS]
    if unknown:
        raise SystemExit(f"알 수 없는 섹터 ID: {unknown}. 가능: {SECTOR_IDS}")

    _print_header(
        f"M4 8섹터 파이프라인 시작 — "
        f"crisis_sampling={crisis_sampling}, sectors={len(targets)}"
    )

    all_per_ticker = []
    for sector_id in targets:
        per_ticker, _ = _run_sector(
            sector_id,
            n_trials=args.n_trials,
            skip_optuna=args.skip_optuna,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            crisis_sampling=crisis_sampling,
            no_eval=args.no_eval,
        )
        if per_ticker is not None:
            all_per_ticker.append(per_ticker)

    if not args.no_eval:
        _summarize(all_per_ticker)

    _print_header(f"전체 파이프라인 완료 ({len(targets)} 섹터)")


if __name__ == "__main__":
    main()
