"""
coverage_tests.py — Extended VaR coverage tests (Phase 1)

Buczyński & Chlebus (2024) GARCHNet 논문의 평가 framework 적용.
기존 kupiec.py (UC test) 에 추가:
  - Conditional Coverage (CC) Test — Christoffersen (1998)
  - Dynamic Quantile (DQ) Test — Engle & Manganelli (2004)

이 두 검정은 violation 의 **시계열적 특성** 을 검사하므로,
y_true / q_pred 가 한 종목 내에서 시간 순서대로 정렬되어 있어야 함.
backtest.py 의 rolling_window 방식은 이 조건을 만족.

기존 kupiec.py 와의 관계:
  - kupiec.kupiec_pof_test(...) → UC test 만 (violation 비율)
  - 본 모듈의 christoffersen_cc_test(...) → UC + Independence
  - 본 모듈의 engle_dq_test(...) → 가장 엄격 (autocorrelation 포함)

표준 VaR validation 의 통상 사용:
  UC test 통과 (필요조건) → CC test 통과 (clustering 없음) → DQ test 통과 (가장 엄격)
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

# 기존 kupiec 함수 재사용
try:
    from .kupiec import kupiec_pof_test
except ImportError:
    try:
        # validation.kupiec.kupiec 형태 (backtest.py 와 동일)
        from validation.kupiec.kupiec import kupiec_pof_test
    except ImportError:
        # standalone 실행 시 (이 파일과 같은 디렉토리에 kupiec.py)
        from kupiec import kupiec_pof_test


# =============================================================
# Christoffersen (1998) Conditional Coverage Test
# =============================================================
def christoffersen_cc_test(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    quantile: float = 0.05,
) -> dict:
    """
    Christoffersen (1998) Conditional Coverage Test.

    LR_cc = LR_uc + LR_ind, ~chi²(2)
      LR_uc  : Unconditional Coverage (Kupiec POF 와 동일)
      LR_ind : Independence — violation 이 군집 (clustering) 발생하지 않는가
      LR_cc  : 두 검정의 결합

    위기 구간에 violation 이 몰려있으면 LR_ind 가 크게 나와 fail.
    따라서 본 검정은 "위기 대응력" 의 직접적 검증.

    Args:
        y_true:    실제 수익률 시계열 (1D)
        q_pred:    분위수 예측 시계열 (1D)
        quantile:  명목 분위수 (예: 0.05)

    Returns:
        dict with LR_uc, LR_ind, LR_cc, p_value, reject
    """
    y_true = np.asarray(y_true).ravel()
    q_pred = np.asarray(q_pred).ravel()
    assert len(y_true) == len(q_pred)

    violations = (y_true < q_pred).astype(int)
    T = len(violations)
    n1 = violations.sum()
    n0 = T - n1

    # --- Unconditional Coverage (UC) ---
    if n1 == 0 or n1 == T:
        return {
            "LR_uc": np.nan, "LR_ind": np.nan, "LR_cc": np.nan,
            "p_value_cc": np.nan, "reject_cc": False,
            "n": T, "violations": int(n1),
        }

    pi_hat = n1 / T
    LR_uc = -2.0 * (
        n1 * np.log(quantile) + n0 * np.log(1 - quantile)
        - n1 * np.log(pi_hat) - n0 * np.log(1 - pi_hat)
    )

    # --- Independence (transition counts) ---
    # n_ij = number of transitions from state i to state j (i, j ∈ {0, 1})
    prev = violations[:-1]
    curr = violations[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    # Transition probabilities
    denom_0 = n00 + n01
    denom_1 = n10 + n11

    if denom_0 == 0 or denom_1 == 0:
        # 한쪽 state 가 전혀 안 나오면 independence 검정 불가
        LR_ind = 0.0
    else:
        pi_01 = n01 / denom_0
        pi_11 = n11 / denom_1
        pi_uncond = (n01 + n11) / (n00 + n01 + n10 + n11)

        # log-likelihood under independence null (single pi)
        # vs alternative (pi_01 != pi_11)
        eps = 1e-12  # 로그 0 방지

        log_l_null = (
            (n00 + n10) * np.log(max(1 - pi_uncond, eps))
            + (n01 + n11) * np.log(max(pi_uncond, eps))
        )
        log_l_alt = (
            n00 * np.log(max(1 - pi_01, eps))
            + n01 * np.log(max(pi_01, eps))
            + n10 * np.log(max(1 - pi_11, eps))
            + n11 * np.log(max(pi_11, eps))
        )
        LR_ind = -2.0 * (log_l_null - log_l_alt)
        LR_ind = max(LR_ind, 0.0)  # numerical safety

    LR_cc = LR_uc + LR_ind
    p_value_cc = 1.0 - chi2.cdf(LR_cc, df=2)

    return {
        "n": T,
        "violations": int(n1),
        "LR_uc": round(LR_uc, 6),
        "LR_ind": round(LR_ind, 6),
        "LR_cc": round(LR_cc, 6),
        "p_value_cc": round(p_value_cc, 6),
        "reject_cc": bool(p_value_cc < 0.05),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
    }


# =============================================================
# Engle & Manganelli (2004) Dynamic Quantile Test
# =============================================================
def engle_dq_test(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    quantile: float = 0.05,
    n_lags: int = 4,
) -> dict:
    """
    Engle & Manganelli (2004) Dynamic Quantile Test.

    Hit_t = violations_t - quantile (centered hit variable)
    회귀: Hit_t = β_0 + Σ_{k=1}^{K} β_k · Hit_{t-k} + β_{K+1} · VaR_t + u_t
    H_0:  β_0 = β_1 = ... = β_{K+1} = 0  (violation 의 시계열 패턴 없음)

    Test statistic: DQ = β̂'(X'X)β̂ / [α(1-α)],  ~ chi²(K+2)

    CC 보다 강력: VaR 자체와의 상관까지 검정.
    GARCHNet 논문에서 표준 VaR 평가의 가장 엄격한 검정으로 사용.

    Args:
        y_true:    실제 수익률 시계열 (1D)
        q_pred:    분위수 예측 시계열 (1D, VaR 추정치 역할)
        quantile:  명목 분위수
        n_lags:    lag hit 개수 (논문 표준: 4)

    Returns:
        dict with DQ_stat, p_value, reject
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    q_pred = np.asarray(q_pred, dtype=float).ravel()
    assert len(y_true) == len(q_pred)

    T = len(y_true)
    if T <= n_lags + 2:
        # 데이터 너무 적으면 검정 불가
        return {
            "n": T, "n_lags": n_lags, "DQ_stat": np.nan,
            "p_value_dq": np.nan, "reject_dq": False,
            "note": "insufficient_data",
        }

    violations = (y_true < q_pred).astype(float)
    hits = violations - quantile  # centered

    # Design matrix:
    #   [1, Hit_{t-1}, Hit_{t-2}, ..., Hit_{t-K}, VaR_t]
    valid_T = T - n_lags
    K = n_lags
    X = np.ones((valid_T, K + 2))
    for k in range(1, K + 1):
        X[:, k] = hits[K - k : T - k]
    X[:, -1] = q_pred[K:]

    y = hits[K:]

    # OLS via solve (X'X 가 singular 일 가능성 처리)
    try:
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return {
            "n": T, "n_lags": n_lags, "DQ_stat": np.nan,
            "p_value_dq": np.nan, "reject_dq": False,
            "note": "singular_design_matrix",
        }

    # DQ statistic
    dq_numer = float(beta @ X.T @ X @ beta)
    dq_denom = quantile * (1.0 - quantile)
    DQ_stat = dq_numer / dq_denom
    p_value_dq = 1.0 - chi2.cdf(DQ_stat, df=K + 2)

    return {
        "n": T,
        "n_lags": n_lags,
        "DQ_stat": round(DQ_stat, 6),
        "p_value_dq": round(p_value_dq, 6),
        "reject_dq": bool(p_value_dq < 0.05),
    }


# =============================================================
# 통합 검증 보고서
# =============================================================
def run_validation_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: list[float] = None,
    vr_threshold: float = 0.05,
    pvalue_threshold: float = 0.05,
    dq_n_lags: int = 4,
) -> pd.DataFrame:
    """
    Kupiec UC + Christoffersen CC + Engle-Manganelli DQ 통합 보고.

    Args:
        y_true:     (T,) 실제 수익률
        y_pred:     (T, n_quantiles) 분위수 예측
        quantiles:  분위수 리스트 (default [0.1, 0.5, 0.9])
        vr_threshold:    violation rate 허용 오차
        pvalue_threshold: 검정 p-value 임계값 (보통 0.05)
        dq_n_lags:  DQ test 의 lag 수

    Returns:
        pd.DataFrame columns:
          quantile, n, violations, violation_rate,
          p_value_uc, uc_pass,
          LR_cc, p_value_cc, cc_pass,
          DQ_stat, p_value_dq, dq_pass,
          all_pass (UC AND CC AND DQ)
    """
    quantiles = quantiles or [0.1, 0.5, 0.9]
    rows = []

    for i, q in enumerate(quantiles):
        # 하방 분위수에 대해서만 적합 (Q < 0.5 가 표준 VaR setting).
        # Q > 0.5 면 부호 뒤집어서 검정 (upper tail VaR).
        if q <= 0.5:
            y_t = y_true
            y_p = y_pred[:, i]
            q_eff = q
        else:
            # Upper tail: y_true > q_pred 가 violation.
            # 동일 검정 framework 을 위해 부호 뒤집기:
            y_t = -y_true
            y_p = -y_pred[:, i]
            q_eff = 1.0 - q

        uc = kupiec_pof_test(y_t, y_p, quantile=q_eff)
        cc = christoffersen_cc_test(y_t, y_p, quantile=q_eff)
        dq = engle_dq_test(y_t, y_p, quantile=q_eff, n_lags=dq_n_lags)

        # Pass 조건들
        vr_pass = abs(uc["violation_rate"] - q_eff) <= vr_threshold
        uc_pass = (
            not np.isnan(uc["p_value"])
            and uc["p_value"] >= pvalue_threshold
        )
        cc_pass = (
            not np.isnan(cc["p_value_cc"])
            and cc["p_value_cc"] >= pvalue_threshold
        )
        dq_pass = (
            not np.isnan(dq["p_value_dq"])
            and dq["p_value_dq"] >= pvalue_threshold
        )

        rows.append({
            "quantile": q,
            "n": uc["n"],
            "violations": uc["violations"],
            "violation_rate": uc["violation_rate"],
            "vr_pass": vr_pass,
            "p_value_uc": uc["p_value"],
            "uc_pass": uc_pass,
            "LR_cc": cc["LR_cc"],
            "p_value_cc": cc["p_value_cc"],
            "cc_pass": cc_pass,
            "DQ_stat": dq["DQ_stat"],
            "p_value_dq": dq["p_value_dq"],
            "dq_pass": dq_pass,
            "all_pass": uc_pass and cc_pass and dq_pass,
        })

    return pd.DataFrame(rows)


def run_validation_by_group_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    quantiles: list[float] = None,
    vr_threshold: float = 0.05,
    pvalue_threshold: float = 0.05,
    dq_n_lags: int = 4,
) -> pd.DataFrame:
    """
    종목 (group) 별로 run_validation_extended 적용.
    backtest.py 의 기존 run_validation_by_group 와 동일 인터페이스.

    한 종목 안에서 시간 순 정렬되어 있어야 CC/DQ 정상 작동.
    """
    quantiles = quantiles or [0.1, 0.5, 0.9]
    rows = []

    for group in np.unique(groups):
        mask = groups == group
        report = run_validation_extended(
            y_true[mask],
            y_pred[mask, :],
            quantiles=quantiles,
            vr_threshold=vr_threshold,
            pvalue_threshold=pvalue_threshold,
            dq_n_lags=dq_n_lags,
        )
        report["group_id"] = group
        rows.append(report)

    df = pd.concat(rows, ignore_index=True)
    # group_id 를 앞으로
    cols = ["group_id"] + [c for c in df.columns if c != "group_id"]
    return df[cols]


# =============================================================
# 요약 보고서 헬퍼
# =============================================================
def summarize_extended_report(report: pd.DataFrame) -> dict:
    """
    그룹별 extended report 의 집계 통계.

    각 분위수 × 검정 별로 통과 종목 수 / 비율 출력.
    """
    summary = {}
    if "group_id" not in report.columns:
        # 단일 그룹 보고서
        return {"total": len(report), "report": report}

    n_groups = report["group_id"].nunique()
    for q in report["quantile"].unique():
        sub = report[report["quantile"] == q]
        summary[f"Q{q}"] = {
            "n_groups": n_groups,
            "uc_pass": int(sub["uc_pass"].sum()),
            "cc_pass": int(sub["cc_pass"].sum()),
            "dq_pass": int(sub["dq_pass"].sum()),
            "all_pass": int(sub["all_pass"].sum()),
            "uc_pass_pct": round(sub["uc_pass"].mean() * 100, 1),
            "cc_pass_pct": round(sub["cc_pass"].mean() * 100, 1),
            "dq_pass_pct": round(sub["dq_pass"].mean() * 100, 1),
            "all_pass_pct": round(sub["all_pass"].mean() * 100, 1),
        }
    return summary


def print_extended_summary(report: pd.DataFrame, title: str = "Extended Validation"):
    """콘솔용 요약 출력."""
    summary = summarize_extended_report(report)
    if "n_groups" in str(summary):
        # 그룹별 보고
        print(f"\n=== {title} ===")
        print(f"{'Quantile':>10} | {'UC':>10} | {'CC':>10} | {'DQ':>10} | {'ALL':>10}")
        print("-" * 60)
        for qk, v in summary.items():
            print(
                f"{qk:>10} | "
                f"{v['uc_pass']:>3}/{v['n_groups']:>3} ({v['uc_pass_pct']:>4.0f}%) | "
                f"{v['cc_pass']:>3}/{v['n_groups']:>3} ({v['cc_pass_pct']:>4.0f}%) | "
                f"{v['dq_pass']:>3}/{v['n_groups']:>3} ({v['dq_pass_pct']:>4.0f}%) | "
                f"{v['all_pass']:>3}/{v['n_groups']:>3} ({v['all_pass_pct']:>4.0f}%)"
            )
    else:
        print(report.to_string(index=False))


# =============================================================
# 단위 자체 검증 (실행 시 sanity check)
# =============================================================
if __name__ == "__main__":
    """
    합성 데이터로 sanity check.
    Case 1: 잘 calibrated → 모든 검정 통과 기대
    Case 2: violation clustering → CC fail 기대
    """
    np.random.seed(42)
    n = 1000
    quantile = 0.05

    # === Case 1: well-calibrated ===
    print("=" * 60)
    print("Case 1: Well-calibrated (violation rate ≈ 5%, no clustering)")
    print("=" * 60)
    y_true = np.random.randn(n) * 0.02
    q_pred = np.percentile(y_true, quantile * 100) * np.ones(n)
    report1 = run_validation_extended(
        y_true, q_pred[:, np.newaxis], quantiles=[quantile]
    )
    print(report1.T.to_string())

    # === Case 2: clustered violations ===
    print("\n" + "=" * 60)
    print("Case 2: Clustered violations (UC 통과 가능, CC 실패 기대)")
    print("=" * 60)
    y_true2 = np.random.randn(n) * 0.02
    q_pred2 = np.percentile(y_true2, quantile * 100) * np.ones(n)
    # 50 개의 violation 을 처음 100 거래일에 몰아넣음 (clustering)
    cluster_start = 50
    cluster_size = 50
    bad_period = slice(cluster_start, cluster_start + cluster_size)
    y_true2[bad_period] = q_pred2[bad_period] - 0.001  # 강제 violation
    # 나머지는 거의 violation 없음
    elsewhere = np.r_[
        np.arange(0, cluster_start),
        np.arange(cluster_start + cluster_size, n)
    ]
    y_true2[elsewhere] = np.maximum(y_true2[elsewhere], q_pred2[elsewhere] + 0.001)

    report2 = run_validation_extended(
        y_true2, q_pred2[:, np.newaxis], quantiles=[quantile]
    )
    print(report2.T.to_string())

    print("\n해석:")
    print("  Case 1 은 UC/CC/DQ 모두 통과 기대 (well-calibrated).")
    print("  Case 2 는 violation rate 가 5%여도 한 곳에 몰려있어")
    print("  CC test 가 clustering 을 잡아내야 함 (reject_cc=True).")
