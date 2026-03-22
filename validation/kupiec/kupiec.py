import numpy as np
import pandas as pd
from scipy.stats import chi2


def violation_rate(y_true: np.ndarray, q_pred: np.ndarray) -> float:
    return float(np.mean(y_true < q_pred))


def kupiec_pof_test(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    quantile: float = 0.1,
) -> dict:
    n = len(y_true)
    violations = np.sum(y_true < q_pred)
    t0 = n - violations
    t1 = violations
    p = t1 / n

    if t1 == 0 or t1 == n:
        return {"violations": int(t1), "n": n, "violation_rate": p, "lr_stat": np.nan, "p_value": np.nan}

    lr = -2 * (
        t1 * np.log(quantile / p) + t0 * np.log((1 - quantile) / (1 - p))
    )
    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "violations": int(t1),
        "n": n,
        "violation_rate": round(p, 6),
        "lr_stat": round(lr, 6),
        "p_value": round(p_value, 6),
    }


def run_validation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: list[float] = None,
    vr_threshold: float = 0.05,
    pvalue_threshold: float = 0.05,
) -> pd.DataFrame:
    quantiles = quantiles or [0.1, 0.5, 0.9]
    rows = []

    for i, q in enumerate(quantiles):
        result = kupiec_pof_test(y_true, y_pred[:, i], quantile=q)
        result["quantile"] = q
        result["vr_pass"] = result["violation_rate"] <= vr_threshold
        result["kupiec_pass"] = (
            result["p_value"] >= pvalue_threshold
            if not np.isnan(result["p_value"])
            else False
        )
        rows.append(result)

    df = pd.DataFrame(rows)[
        ["quantile", "violations", "n", "violation_rate", "lr_stat", "p_value", "vr_pass", "kupiec_pass"]
    ]
    return df
