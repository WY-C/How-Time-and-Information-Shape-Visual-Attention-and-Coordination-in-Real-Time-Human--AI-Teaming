from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning


@dataclass
class LMMFit:
    result: object
    data: pd.DataFrame
    formula: str
    group_col: str
    trial_vc: bool
    optimizer: str


def fit_lmm(
    data: pd.DataFrame,
    formula: str,
    *,
    group_col: str = "pid",
    trial_vc: bool = False,
    trial_col: str = "trial_id",
    required_cols: Sequence[str] | None = None,
) -> LMMFit:
    """Fit an ML linear mixed-effects model, failing closed on non-convergence.

    Random structure:
      - participant random intercept via ``groups=pid``
      - optional nested trial variance component via ``0 + C(trial_id)``

    Fixed-effect inference is based on Wald chi-square tests from the converged
    full model. Sum coding should be used in ``formula`` for Type-III-like term
    tests in the presence of interactions.
    """
    req = [group_col]
    if trial_vc:
        req.append(trial_col)
    if required_cols:
        req.extend(required_cols)
    req = list(dict.fromkeys(req))
    d = data.dropna(subset=[c for c in req if c in data.columns]).copy()
    if d.empty:
        raise RuntimeError(f"No usable rows for LMM: {formula}")
    if d[group_col].nunique() < 2:
        raise RuntimeError(f"LMM requires >=2 {group_col} groups: {formula}")

    vc = {"Trial": f"0 + C({trial_col})"} if trial_vc else None
    last_error: Exception | None = None
    last_result = None
    for optimizer in ("lbfgs", "powell", "bfgs", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.filterwarnings("ignore", message="Random effects covariance is singular")
                warnings.filterwarnings("ignore", message="The random effects covariance matrix is singular")
                warnings.filterwarnings("ignore", message="The MLE may be on the boundary of the parameter space")
                model = smf.mixedlm(
                    formula,
                    d,
                    groups=d[group_col],
                    re_formula="1",
                    vc_formula=vc,
                )
                result = model.fit(reml=False, method=optimizer, maxiter=2000, disp=False)
            last_result = result
            finite_fit = (
                np.isfinite(float(getattr(result, "llf", np.nan)))
                and np.isfinite(float(getattr(result, "aic", np.nan)))
                and np.isfinite(float(getattr(result, "bic", np.nan)))
            )
            if bool(getattr(result, "converged", False)) and finite_fit:
                return LMMFit(result=result, data=d, formula=formula, group_col=group_col, trial_vc=trial_vc, optimizer=optimizer)
        except Exception as exc:  # keep structure fixed; only optimizer changes
            last_error = exc
            continue
    if last_result is not None:
        raise RuntimeError(
            f"LMM failed to produce a converged finite-likelihood fit under fixed random structure: {formula}"
        )
    raise RuntimeError(f"LMM fit failed: {formula}; last error={last_error}")


def _term_name(raw: str) -> str:
    name = raw.replace("C(", "").replace(", Sum)", "")
    name = name.replace(":", "×")
    return name


def lmm_term_table(fit: LMMFit, *, analysis: str, aoi: str | None = None, target_terms: Iterable[str] | None = None) -> pd.DataFrame:
    """Return fixed-effect Wald chi-square tests, robust to boundary singularity.

    statsmodels' ``wald_test_terms`` can fail when a random-effect variance lies
    on the boundary (common in some narrow temporal-bin models). The fixed-effect
    model is still estimable, so term tests are computed directly from the
    fixed-effect coefficient vector and covariance submatrix using a Moore–Penrose
    pseudoinverse. Degrees of freedom equal the numerical rank of that covariance
    block.
    """
    result = fit.result
    fe_names = list(result.model.exog_names)
    beta = result.fe_params.reindex(fe_names).to_numpy(dtype=float)
    cov_all = result.cov_params().loc[fe_names, fe_names]
    design_info = result.model.data.design_info

    rows = []
    for raw_term, sl in design_info.term_name_slices.items():
        idx = np.arange(len(fe_names))[sl]
        b = beta[idx]
        v = cov_all.to_numpy(dtype=float)[np.ix_(idx, idx)]
        if len(idx) == 0:
            continue
        if not np.isfinite(v).all() or not np.isfinite(b).all():
            stat = np.nan
            pval = np.nan
            df = 0
        else:
            try:
                rank = int(np.linalg.matrix_rank(v, tol=1e-10))
            except np.linalg.LinAlgError:
                rank = 0
            if rank <= 0:
                stat = np.nan
                pval = np.nan
                df = 0
            else:
                stat = float(b.T @ np.linalg.pinv(v, rcond=1e-10) @ b)
                df = rank
                pval = float(chi2.sf(stat, df))
        rows.append({"raw_term": raw_term, "Wald_chi2": stat, "p_value": pval, "df": df})
    table = pd.DataFrame(rows)
    table["term"] = table["raw_term"].map(_term_name)
    target = set(target_terms or [])

    participant_var = float(result.cov_re.iloc[0, 0]) if getattr(result, "cov_re", None) is not None and result.cov_re.size else np.nan
    trial_var = np.nan
    if fit.trial_vc and hasattr(result, "vcomp") and len(result.vcomp):
        trial_var = float(result.vcomp[0])

    table.insert(0, "analysis", analysis)
    if aoi is not None:
        table.insert(1, "AOI", aoi)
    table["role"] = table["term"].map(lambda x: "target" if x in target else ("intercept" if x == "Intercept" else "nuisance"))
    table["n_obs"] = int(result.nobs)
    table["n_participants"] = int(fit.data[fit.group_col].nunique())
    table["n_trials"] = int(fit.data["trial_id"].nunique()) if "trial_id" in fit.data.columns else np.nan
    table["converged"] = bool(result.converged)
    table["optimizer"] = fit.optimizer
    table["log_likelihood"] = float(result.llf)
    table["AIC"] = float(result.aic)
    table["BIC"] = float(result.bic)
    table["participant_random_intercept_var"] = participant_var
    table["trial_random_intercept_var"] = trial_var
    boundary = (np.isfinite(participant_var) and participant_var < 1e-8) or (np.isfinite(trial_var) and trial_var < 1e-8)
    table["random_effect_boundary"] = bool(boundary)
    table["residual_var"] = float(result.scale)
    table["formula"] = fit.formula
    table["random_effects"] = "(1|Participant) + (1|Trial)" if fit.trial_vc else "(1|Participant)"
    return table

def _design_mean(fit: LMMFit, grid: pd.DataFrame) -> np.ndarray:
    design_info = fit.result.model.data.design_info
    mat = patsy.build_design_matrices([design_info], grid, return_type="dataframe")[0]
    mat = mat.reindex(columns=fit.result.model.exog_names, fill_value=0.0)
    return mat.to_numpy(dtype=float).mean(axis=0)


def _grid(data: pd.DataFrame, fixed: dict, marginal_over: Sequence[str]) -> pd.DataFrame:
    levels = []
    for col in marginal_over:
        vals = list(pd.unique(data[col].dropna()))
        try:
            vals = sorted(vals)
        except TypeError:
            vals = sorted(vals, key=str)
        levels.append(vals)
    combos = list(itertools.product(*levels)) if levels else [()]
    rows = []
    for combo in combos:
        rec = dict(fixed)
        for col, value in zip(marginal_over, combo):
            rec[col] = value
        rows.append(rec)
    return pd.DataFrame(rows)


def lmm_emm_contrast(
    fit: LMMFit,
    *,
    factor: str,
    level1,
    level2,
    fixed: dict | None = None,
    marginal_over: Sequence[str] = (),
) -> dict:
    """Estimated-marginal-mean contrast on the model's transformed scale.

    Contrast is level2 - level1. Nuisance factors listed in ``marginal_over``
    are equally averaged across their observed levels, preserving a balanced
    factorial interpretation rather than weighting by row frequency.
    """
    fixed = dict(fixed or {})
    g1 = _grid(fit.data, {**fixed, factor: level1}, marginal_over)
    g2 = _grid(fit.data, {**fixed, factor: level2}, marginal_over)
    x1 = _design_mean(fit, g1)
    x2 = _design_mean(fit, g2)
    c = x2 - x1

    fe_names = list(fit.result.model.exog_names)
    beta = fit.result.fe_params.reindex(fe_names).to_numpy(dtype=float)
    cov_all = fit.result.cov_params()
    cov = cov_all.loc[fe_names, fe_names].to_numpy(dtype=float)
    est = float(c @ beta)
    var = float(c @ cov @ c)
    se = math.sqrt(max(var, 0.0))
    z = est / se if se > 0 else np.nan
    p = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "factor": factor,
        "level1": level1,
        "level2": level2,
        "estimate_level2_minus_level1_logit": est,
        "SE": se,
        "z": z,
        "p_uncorrected": p,
    }


def holm_adjust(rows: list[dict], alpha: float = 0.05) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    valid = out["p_uncorrected"].notna()
    out["p_holm"] = np.nan
    out["significant_holm"] = False
    if valid.any():
        reject, padj, _, _ = multipletests(out.loc[valid, "p_uncorrected"], alpha=alpha, method="holm")
        out.loc[valid, "p_holm"] = padj
        out.loc[valid, "significant_holm"] = reject.astype(bool)
    return out
