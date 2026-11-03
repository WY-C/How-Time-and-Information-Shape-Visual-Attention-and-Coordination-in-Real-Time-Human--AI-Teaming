from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import helmert
from statsmodels.stats.multitest import multipletests


ALPHA = 0.05


def helmert_orthonormal(k: int) -> np.ndarray:
    """Return (k-1) x k orthonormal Helmert contrasts."""
    if k < 2:
        raise ValueError("k must be >= 2")
    return np.asarray(helmert(k, full=False), dtype=float)


def gg_epsilon_from_scores(scores: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon from orthonormal effect contrast scores.

    scores: n_subjects x effect_df matrix. For a factor with k levels,
    effect_df = k - 1. For an interaction, effect_df is the product of the
    component dfs.
    """
    x = np.asarray(scores, dtype=float)
    if x.ndim != 2:
        raise ValueError("scores must be a 2D array")
    d = x.shape[1]
    if d <= 1:
        return 1.0
    s = np.cov(x, rowvar=False, ddof=1)
    if np.ndim(s) == 0:
        return 1.0
    tr = float(np.trace(s))
    tr2 = float(np.trace(s @ s))
    if tr2 <= 0 or not np.isfinite(tr2):
        return 1.0
    eps = (tr * tr) / (d * tr2)
    return float(np.clip(eps, 1.0 / d, 1.0))


def mauchly_from_scores(scores: np.ndarray, alpha: float = ALPHA) -> dict:
    """Mauchly's test of sphericity from orthonormal effect contrast scores.

    This implements the standard Mauchly W statistic on the covariance matrix
    of the effect contrast scores. For a one-way factor with k levels, the
    score dimension d is k-1. For a multi-factor within-subject interaction,
    d is the interaction df (e.g., 4 for a 3x3 interaction).

    If d == 1 (e.g., a two-level factor), sphericity is necessarily satisfied
    and no Mauchly test is required.
    """
    x = np.asarray(scores, dtype=float)
    if x.ndim != 2:
        raise ValueError("scores must be a 2D array")
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("scores contain NaN/Inf; use complete cases before testing")

    n, d = x.shape
    if n < 3:
        raise ValueError("At least 3 complete participants are required")
    if d <= 1:
        return {
            "mauchly_applicable": False,
            "mauchly_W": 1.0,
            "mauchly_chi2": np.nan,
            "mauchly_df": 0.0,
            "mauchly_p": 1.0,
            "sphericity_violated": False,
        }

    s = np.cov(x, rowvar=False, ddof=1)
    eig = np.linalg.eigvalsh(s)
    tol = max(np.finfo(float).eps * max(abs(eig).max(), 1.0) * d, 1e-15)
    if np.any(eig <= tol):
        # Degenerate covariance implies W -> 0. Treat as a violation rather
        # than silently returning an undefined statistic.
        return {
            "mauchly_applicable": True,
            "mauchly_W": 0.0,
            "mauchly_chi2": np.inf,
            "mauchly_df": float(d * (d + 1) / 2 - 1),
            "mauchly_p": 0.0,
            "sphericity_violated": True,
        }

    w = float(np.prod(eig) / (eig.mean() ** d))
    w = float(np.clip(w, np.finfo(float).tiny, 1.0))

    # Standard chi-square approximation. Writing k=d+1 gives the familiar
    # correction factor f = (2d^2 + d + 2) / (6d(n-1)).
    f = (2 * d * d + d + 2) / (6 * d * (n - 1))
    chi2 = float((f - 1.0) * (n - 1) * np.log(w))
    df = float(d * (d + 1) / 2 - 1)
    p = float(stats.chi2.sf(chi2, df))

    return {
        "mauchly_applicable": True,
        "mauchly_W": w,
        "mauchly_chi2": chi2,
        "mauchly_df": df,
        "mauchly_p": p,
        "sphericity_violated": bool(p < alpha),
    }


def _apply_sphericity_rule(row: dict, scores: np.ndarray, alpha: float) -> dict:
    sph = mauchly_from_scores(scores, alpha=alpha)
    eps = gg_epsilon_from_scores(scores)

    row.update(sph)
    row["epsilon_GG"] = eps
    row["df1_GG"] = row["df1"] * eps
    row["df2_GG"] = row["df2"] * eps
    row["p_GG"] = float(stats.f.sf(row["F"], row["df1_GG"], row["df2_GG"]))

    if sph["sphericity_violated"]:
        row["correction_used"] = "Greenhouse-Geisser"
        row["df1_report"] = row["df1_GG"]
        row["df2_report"] = row["df2_GG"]
        row["p_report"] = row["p_GG"]
    else:
        row["correction_used"] = "none"
        row["df1_report"] = row["df1"]
        row["df2_report"] = row["df2"]
        row["p_report"] = row["p_uncorrected"]
    return row


def rm_anova_1way(
    data: pd.DataFrame,
    subject: str,
    factor: str,
    dv: str,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """One-way fully within-subject RM-ANOVA with conditional GG correction."""
    d = data[[subject, factor, dv]].dropna().copy()
    if d.duplicated([subject, factor]).any():
        raise ValueError(f"Duplicate participant-condition cells for {dv}")

    levels = sorted(d[factor].unique())
    pivot = d.pivot(index=subject, columns=factor, values=dv).reindex(columns=levels).dropna()
    y = pivot.to_numpy(dtype=float)
    n, k = y.shape
    if k < 2:
        raise ValueError(f"{factor} must have at least two levels")

    grand = y.mean()
    mean_level = y.mean(axis=0)
    mean_subject = y.mean(axis=1)

    ss_effect = n * np.sum((mean_level - grand) ** 2)
    error = y - mean_subject[:, None] - mean_level[None, :] + grand
    ss_error = np.sum(error**2)

    df1 = k - 1
    df2 = (n - 1) * df1
    f_value = (ss_effect / df1) / (ss_error / df2)
    p_unc = float(stats.f.sf(f_value, df1, df2))
    eta = float(ss_effect / (ss_effect + ss_error))

    h = helmert_orthonormal(k)
    scores = y @ h.T

    row = {
        "term": factor,
        "F": float(f_value),
        "df1": float(df1),
        "df2": float(df2),
        "p_uncorrected": p_unc,
        "partial_eta2": eta,
        "n": int(n),
    }
    row = _apply_sphericity_rule(row, scores, alpha)
    return pd.DataFrame([row])


def rm_anova_2way(
    data: pd.DataFrame,
    subject: str,
    factor_a: str,
    factor_b: str,
    dv: str,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Balanced two-way fully within-subject RM-ANOVA.

    One row per participant x factor_a x factor_b is required. Sphericity is
    tested separately for A, B, and A x B using orthonormal effect contrast
    scores. Greenhouse-Geisser correction is used only for effects whose
    Mauchly test rejects sphericity at alpha.
    """
    d = data[[subject, factor_a, factor_b, dv]].dropna().copy()
    if d.duplicated([subject, factor_a, factor_b]).any():
        raise ValueError(f"Duplicate participant-condition cells for {dv}")

    subjects = sorted(d[subject].unique())
    a_levels = sorted(d[factor_a].unique())
    b_levels = sorted(d[factor_b].unique())
    index = pd.MultiIndex.from_product(
        [subjects, a_levels, b_levels], names=[subject, factor_a, factor_b]
    )
    series = d.set_index([subject, factor_a, factor_b])[dv].reindex(index)
    complete = series.groupby(level=0).apply(lambda x: x.notna().all())
    subjects = [s for s in subjects if bool(complete.loc[s])]

    y = series.loc[(subjects, slice(None), slice(None))].to_numpy(dtype=float)
    y = y.reshape(len(subjects), len(a_levels), len(b_levels))
    n, a, b = y.shape

    grand = y.mean()
    mean_s = y.mean(axis=(1, 2))
    mean_a = y.mean(axis=(0, 2))
    mean_b = y.mean(axis=(0, 1))
    mean_ab = y.mean(axis=0)
    mean_sa = y.mean(axis=2)
    mean_sb = y.mean(axis=1)

    ss_a = n * b * np.sum((mean_a - grand) ** 2)
    err_a = mean_sa - mean_s[:, None] - mean_a[None, :] + grand
    ss_sa = b * np.sum(err_a**2)

    ss_b = n * a * np.sum((mean_b - grand) ** 2)
    err_b = mean_sb - mean_s[:, None] - mean_b[None, :] + grand
    ss_sb = a * np.sum(err_b**2)

    effect_ab = mean_ab - mean_a[:, None] - mean_b[None, :] + grand
    ss_ab = n * np.sum(effect_ab**2)
    err_ab = (
        y
        - mean_sa[:, :, None]
        - mean_sb[:, None, :]
        - mean_ab[None, :, :]
        + mean_s[:, None, None]
        + mean_a[None, :, None]
        + mean_b[None, None, :]
        - grand
    )
    ss_sab = np.sum(err_ab**2)

    h_a = helmert_orthonormal(a)
    h_b = helmert_orthonormal(b)
    score_a = mean_sa @ h_a.T
    score_b = mean_sb @ h_b.T
    score_ab = np.array([(h_a @ y[i] @ h_b.T).ravel() for i in range(n)])

    specs = [
        (factor_a, ss_a, ss_sa, a - 1, score_a),
        (factor_b, ss_b, ss_sb, b - 1, score_b),
        (f"{factor_a}×{factor_b}", ss_ab, ss_sab, (a - 1) * (b - 1), score_ab),
    ]

    rows = []
    for term, ss_effect, ss_error, df1, scores in specs:
        df2 = (n - 1) * df1
        f_value = (ss_effect / df1) / (ss_error / df2)
        row = {
            "term": term,
            "F": float(f_value),
            "df1": float(df1),
            "df2": float(df2),
            "p_uncorrected": float(stats.f.sf(f_value, df1, df2)),
            "partial_eta2": float(ss_effect / (ss_effect + ss_error)),
            "n": int(n),
        }
        rows.append(_apply_sphericity_rule(row, scores, alpha))

    return pd.DataFrame(rows)


def paired_posthoc(
    condition_means: pd.DataFrame,
    outcome: str,
    factor: str,
    subject: str = "pid",
) -> pd.DataFrame:
    """Marginal paired comparisons with Holm correction."""
    marginal = condition_means.groupby([subject, factor], as_index=False)[outcome].mean()
    pivot = marginal.pivot(index=subject, columns=factor, values=outcome)
    rows = []
    for level_1, level_2 in combinations(sorted(pivot.columns), 2):
        pair = pivot[[level_1, level_2]].dropna()
        diff = pair[level_2] - pair[level_1]
        t_value, p_value = stats.ttest_rel(pair[level_2], pair[level_1])
        sd = diff.std(ddof=1)
        rows.append(
            {
                "outcome": outcome,
                "factor": factor,
                "level_1": level_1,
                "level_2": level_2,
                "mean_1": pair[level_1].mean(),
                "mean_2": pair[level_2].mean(),
                "difference_2_minus_1": diff.mean(),
                "t": float(t_value),
                "df": int(len(pair) - 1),
                "p_uncorrected": float(p_value),
                "cohens_dz": float(diff.mean() / sd) if sd > 0 else np.nan,
                "n": int(len(pair)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        reject, p_holm, _, _ = multipletests(out["p_uncorrected"], method="holm")
        out["p_holm"] = p_holm
        out["significant_holm"] = reject
    return out


def logit_adjust(p, n_reference: int) -> np.ndarray:
    """Continuity-adjusted logit for proportions that may contain 0 or 1."""
    p = np.asarray(p, dtype=float)
    q = (p * (n_reference - 1) + 0.5) / n_reference
    return np.log(q / (1 - q))


def print_report_table(anova: pd.DataFrame) -> None:
    cols = [
        "outcome",
        "term",
        "F",
        "df1_report",
        "df2_report",
        "p_report",
        "partial_eta2",
        "mauchly_p",
        "correction_used",
    ]
    cols = [c for c in cols if c in anova.columns]
    print(anova[cols].to_string(index=False))
