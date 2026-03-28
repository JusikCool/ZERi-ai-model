from __future__ import annotations

import math
from typing import Sequence

import pandas as pd
from scipy.stats import chi2


def _prediction_column(quantile: float) -> str:
    return f"pred_q{int(quantile * 100):02d}"


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> float:
    errors = y_true - y_pred
    loss = pd.concat([quantile * errors, (quantile - 1.0) * errors], axis=1).max(axis=1)
    return float(loss.mean())


def empirical_coverage(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true <= y_pred).mean())


def violation_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true < y_pred).mean())


def kupiec_pof_test(y_true: pd.Series, y_pred: pd.Series, alpha: float) -> dict[str, float]:
    violations = int((y_true < y_pred).sum())
    n_obs = int(len(y_true))
    observed_rate = violations / n_obs if n_obs > 0 else float("nan")
    eps = 1e-12
    observed_rate_clipped = min(max(observed_rate, eps), 1.0 - eps)
    alpha_clipped = min(max(alpha, eps), 1.0 - eps)

    log_null = (n_obs - violations) * math.log(1.0 - alpha_clipped) + violations * math.log(alpha_clipped)
    log_alt = (n_obs - violations) * math.log(1.0 - observed_rate_clipped) + violations * math.log(observed_rate_clipped)
    lr_pof = float(max(0.0, -2.0 * (log_null - log_alt)))
    p_value = float(1.0 - chi2.cdf(lr_pof, df=1))

    return {
        "n_obs": n_obs,
        "violations": violations,
        "expected_violation_rate": float(alpha),
        "observed_violation_rate": float(observed_rate),
        "lr_pof": lr_pof,
        "p_value": p_value,
    }


def build_quantile_metrics(prediction_df: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    y_true = prediction_df["y_true"]
    for quantile in quantiles:
        pred_col = _prediction_column(quantile)
        y_pred = prediction_df[pred_col]
        kupiec = kupiec_pof_test(y_true=y_true, y_pred=y_pred, alpha=quantile)
        rows.append(
            {
                "quantile": float(quantile),
                "pred_column": pred_col,
                "pinball_loss": pinball_loss(y_true=y_true, y_pred=y_pred, quantile=quantile),
                "empirical_coverage": empirical_coverage(y_true=y_true, y_pred=y_pred),
                "coverage_gap": empirical_coverage(y_true=y_true, y_pred=y_pred) - float(quantile),
                "violation_rate": violation_rate(y_true=y_true, y_pred=y_pred),
                "kupiec_lr_pof": kupiec["lr_pof"],
                "kupiec_p_value": kupiec["p_value"],
                "n_obs": kupiec["n_obs"],
                "violations": kupiec["violations"],
            }
        )
    return pd.DataFrame(rows)


def build_group_quantile_metrics(
    prediction_df: pd.DataFrame,
    quantiles: Sequence[float],
    group_column: str = "group_id",
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for group_value, group_df in prediction_df.groupby(group_column):
        group_metrics = build_quantile_metrics(group_df, quantiles).copy()
        group_metrics.insert(0, group_column, str(group_value))
        rows.extend(group_metrics.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_summary_metrics(quantile_metrics: pd.DataFrame) -> dict[str, float]:
    q10_row = quantile_metrics.loc[quantile_metrics["quantile"] == 0.10].iloc[0]
    return {
        "mean_pinball_loss": float(quantile_metrics["pinball_loss"].mean()),
        "violation_rate_q10": float(q10_row["violation_rate"]),
        "empirical_coverage_q10": float(q10_row["empirical_coverage"]),
        "kupiec_lr_pof_q10": float(q10_row["kupiec_lr_pof"]),
        "kupiec_p_value_q10": float(q10_row["kupiec_p_value"]),
    }
