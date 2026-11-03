#!/usr/bin/env python3
"""Behavioral reanalysis: score, raw collision count, timeout, and response time.

Canonical behavioral contract
=============================
Collision
- Primary DV is raw collision count x.
- No log(1+x), square-root, or other transformation is applied.
- Each trial keeps its raw ``num_collisions`` value.
- The three layout repetitions are averaged within participant x Time x Horizon,
  and that raw participant-condition mean enters RM-ANOVA/post-hoc tests.

Timeout
- Preserve the original trial-level timeout definition exactly:
      timeout_rate = num_human_idle / decision_steps_completed
- The adjusted-logit transform is applied at the TRIAL level first, then the
  three layout repetitions are averaged within participant x condition.

Response time
- successful decision -> recorded response_time_ms, clipped to [0, budget]
- timeout decision (is_human_idle == 1) -> full available budget (200/400/600 ms)
- a missing RT on a non-timeout decision is treated as inconsistent source data
  and fails closed instead of being silently reclassified as a timeout.
- log(RT) is computed at the decision level, averaged to trial level, then to
  participant-condition level for inference. Raw ms are retained for descriptives.

Raw decision and trial values are always saved before participant aggregation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    HORIZON_LEVELS, TIME_LEVELS, adjusted_logit, ensure_output_tree,
    group_mean_ci95, load_steps, load_trials, save_csv,
)
from rm_stats import paired_posthoc, rm_anova_2way


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def build_decision_values(steps: pd.DataFrame, trial: pd.DataFrame) -> pd.DataFrame:
    meta = trial[["trial_id", "pid", "Time", "Horizon", "Layout"]]
    d = (
        steps[steps["human_action_due"].eq(1)]
        .drop(columns=["pid"], errors="ignore")
        .merge(meta, on="trial_id", how="inner", validate="many_to_one")
    )

    rt = pd.to_numeric(d["response_time_ms"], errors="coerce")
    timeout = d["is_human_idle"].eq(1)

    # Do not silently reinterpret an unexplained missing RT as timeout. The
    # canonical timeout indicator is is_human_idle, while the published timeout
    # rate itself comes from trial.csv (num_human_idle / decision_steps_completed).
    bad_missing = (~timeout) & rt.isna()
    if bool(bad_missing.any()):
        sample = d.loc[bad_missing, ["pid", "trial_id", "step", "response_time_ms", "is_human_idle"]].head(20)
        raise RuntimeError(
            "Found non-timeout human_action_due rows with missing response_time_ms. "
            "Refusing to silently count them as timeout:\n" + sample.to_string(index=False)
        )

    d["timeout"] = timeout.astype(int)
    d["rt_timeout_imputed"] = timeout.astype(int)

    successful_capped = rt.clip(lower=0)
    successful_capped = np.minimum(successful_capped, d["Time"].astype(float))
    d["response_time_with_timeout_ms"] = np.where(
        timeout,
        d["Time"].astype(float),
        successful_capped,
    )
    d["response_time_fraction_of_budget"] = d["response_time_with_timeout_ms"] / d["Time"]
    d["log_response_time_with_timeout"] = np.log(d["response_time_with_timeout_ms"].clip(lower=1.0))

    keep = [
        "pid", "trial_id", "step", "Time", "Horizon", "Layout", "state_onset_ms",
        "response_time_ms", "is_human_idle", "timeout", "rt_timeout_imputed",
        "response_time_with_timeout_ms", "response_time_fraction_of_budget",
        "log_response_time_with_timeout",
    ]
    return d[keep].copy()


def build_trial_values(trial: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "score_final",
        "num_collisions",
        "num_human_idle",
        "decision_steps_completed",
    ]
    missing = [c for c in needed if c not in trial.columns]
    if missing:
        raise ValueError(f"trial.csv missing behavioral columns: {missing}")

    step_agg = decisions.groupby(["pid", "trial_id"], as_index=False).agg(
        step_n_decisions=("step", "size"),
        step_n_timeouts=("timeout", "sum"),
        step_timeout_rate=("timeout", "mean"),
        mean_response_time_with_timeout_ms=("response_time_with_timeout_ms", "mean"),
        median_response_time_with_timeout_ms=("response_time_with_timeout_ms", "median"),
        mean_rt_fraction_of_budget=("response_time_fraction_of_budget", "mean"),
        mean_log_response_time_with_timeout=("log_response_time_with_timeout", "mean"),
    )

    base = trial[
        [
            "pid", "trial_id", "Time", "Horizon", "Layout",
            "score_final", "num_collisions", "num_human_idle", "decision_steps_completed",
        ]
    ].copy()

    for c in ["score_final", "num_collisions", "num_human_idle", "decision_steps_completed"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    invalid_denom = base["decision_steps_completed"].isna() | base["decision_steps_completed"].le(0)
    if bool(invalid_denom.any()):
        raise RuntimeError(
            "Invalid decision_steps_completed in completed trials:\n"
            + base.loc[invalid_denom, ["pid", "trial_id", "decision_steps_completed"]].head(20).to_string(index=False)
        )

    # Canonical timeout measure: preserve the original trial.csv definition.
    base["timeout_rate"] = base["num_human_idle"] / base["decision_steps_completed"]
    if ((base["timeout_rate"] < 0) | (base["timeout_rate"] > 1)).any():
        raise RuntimeError("trial.csv produced timeout_rate outside [0,1]")

    # Preserve original transformation order: transform each trial proportion,
    # then average transformed trial values across layouts at participant-condition level.
    base["timeout_logit"] = adjusted_logit(base["timeout_rate"])

    out = base.merge(step_agg, on=["pid", "trial_id"], how="left", validate="one_to_one")
    if out[["step_n_decisions", "mean_response_time_with_timeout_ms"]].isna().any().any():
        raise RuntimeError("Some completed trials have no human decision rows")

    # Keep source-vs-step consistency diagnostics in raw trial output. The
    # trial.csv quantities remain canonical for timeout inference regardless.
    out["decision_count_delta_step_minus_trial"] = out["step_n_decisions"] - out["decision_steps_completed"]
    out["timeout_count_delta_step_minus_trial"] = out["step_n_timeouts"] - out["num_human_idle"]
    out["timeout_rate_delta_step_minus_trial"] = out["step_timeout_rate"] - out["timeout_rate"]

    return out


def participant_inputs(raw: pd.DataFrame) -> pd.DataFrame:
    # Average the three trial/layout repetitions only AFTER trial-level raw values
    # and transformed timeout/RT values have been constructed.
    cols = [
        "score_final",
        "num_collisions",  # raw x, intentionally untransformed
        "timeout_rate",
        "timeout_logit",
        "mean_response_time_with_timeout_ms",
        "median_response_time_with_timeout_ms",
        "mean_rt_fraction_of_budget",
        "mean_log_response_time_with_timeout",
    ]
    return raw.groupby(["pid", "Time", "Horizon"], as_index=False)[cols].mean()


def _rm_posthoc_families(cells: pd.DataFrame, dv: str, label: str) -> pd.DataFrame:
    """Marginal and simple-effect paired post-hoc families with Holm correction."""
    rows = []

    for factor in ["Time", "Horizon"]:
        q = paired_posthoc(cells, dv, factor)
        q["analysis"] = "Behavior RM-ANOVA post-hoc"
        q["outcome"] = label
        q["posthoc_family"] = f"{label}: marginal {factor}"
        q["comparison_scope"] = "marginal"
        q["conditioning_factor"] = ""
        q["conditioning_level"] = ""
        rows.append(q)

    for h in HORIZON_LEVELS:
        q = paired_posthoc(cells[cells["Horizon"].eq(h)], dv, "Time")
        q["analysis"] = "Behavior RM-ANOVA post-hoc"
        q["outcome"] = label
        q["posthoc_family"] = f"{label}: Time within Horizon={h}"
        q["comparison_scope"] = "simple_effect"
        q["conditioning_factor"] = "Horizon"
        q["conditioning_level"] = h
        rows.append(q)

    for t in TIME_LEVELS:
        q = paired_posthoc(cells[cells["Time"].eq(t)], dv, "Horizon")
        q["analysis"] = "Behavior RM-ANOVA post-hoc"
        q["outcome"] = label
        q["posthoc_family"] = f"{label}: Horizon within Time={t}"
        q["comparison_scope"] = "simple_effect"
        q["conditioning_factor"] = "Time"
        q["conditioning_level"] = t
        rows.append(q)

    return pd.concat(rows, ignore_index=True)


def statistics(cells: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {
        "Task score": "score_final",
        "Collision count (raw x)": "num_collisions",
        "Deadline timeout rate adjusted logit": "timeout_logit",
        "Response time with timeout=budget (log ms)": "mean_log_response_time_with_timeout",
    }
    anovas, posts = [], []
    for label, dv in specs.items():
        r = rm_anova_2way(cells, "pid", "Time", "Horizon", dv, alpha=alpha)
        r.insert(0, "outcome", label)
        anovas.append(r)
        posts.append(_rm_posthoc_families(cells, dv, label))
    return pd.concat(anovas, ignore_index=True), pd.concat(posts, ignore_index=True)

def descriptives(cells: pd.DataFrame) -> pd.DataFrame:
    raw = [
        "score_final",
        "num_collisions",
        "timeout_rate",
        "mean_response_time_with_timeout_ms",
    ]
    rows = []
    for c in raw:
        g = group_mean_ci95(cells, ["Time", "Horizon"], c)
        g["outcome"] = c
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def plots(cells: pd.DataFrame, figdir: Path) -> None:
    specs = [
        ("score_final", "Task score", "Score"),
        ("num_collisions", "Collision count", "Count"),
        ("timeout_rate", "Deadline timeout rate", "Timeout (%)"),
        ("mean_response_time_with_timeout_ms", "Response time (timeout = budget)", "Response time (ms)"),
    ]
    for col, title, ylabel in specs:
        fig, ax = plt.subplots(figsize=(7, 5))
        for h in HORIZON_LEVELS:
            q = group_mean_ci95(cells[cells["Horizon"].eq(h)], ["Time"], col)
            scale = 100.0 if col == "timeout_rate" else 1.0
            ax.errorbar(
                q["Time"], q["mean"] * scale,
                yerr=q["ci95"] * scale,
                marker="o", capsize=4, elinewidth=1.0,
                label=f"H{h}",
            )
        ax.set_xticks(TIME_LEVELS)
        ax.set_xlabel("Available decision time (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — mean ± 95% CI")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / f"behavior_{col}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = ensure_output_tree(args.out)
    trial = load_trials(args.roots, strict_main=not args.allow_incomplete)
    steps = load_steps(args.roots, trial)

    decisions = build_decision_values(steps, trial)
    save_csv(decisions, paths["raw"] / "behavior_decision_values.csv")

    raw = build_trial_values(trial, decisions)
    save_csv(raw, paths["raw"] / "behavior_trial_values.csv")

    cells = participant_inputs(raw)
    save_csv(cells, paths["inputs"] / "behavior_participant_condition_values.csv")

    a, p = statistics(cells, args.alpha)
    save_csv(a, paths["stats"] / "behavior_rm_anova.csv")
    save_csv(p, paths["stats"] / "behavior_posthoc_holm.csv")

    save_csv(descriptives(cells), paths["desc"] / "behavior_descriptives.csv")
    plots(cells, paths["fig"])

    qc = pd.DataFrame([{
        "n_decisions": len(decisions),
        "n_trials": len(raw),
        "n_timeout_imputed_decisions": int(decisions["rt_timeout_imputed"].sum()),
        "min_capped_rt_ms": float(decisions["response_time_with_timeout_ms"].min()),
        "max_rt_minus_budget_ms": float((decisions["response_time_with_timeout_ms"] - decisions["Time"]).max()),
        "max_abs_decision_count_delta_step_minus_trial": float(raw["decision_count_delta_step_minus_trial"].abs().max()),
        "max_abs_timeout_count_delta_step_minus_trial": float(raw["timeout_count_delta_step_minus_trial"].abs().max()),
        "max_abs_timeout_rate_delta_step_minus_trial": float(raw["timeout_rate_delta_step_minus_trial"].abs().max()),
        "collision_inference_scale": "raw_x",
        "timeout_inference_source": "trial_num_human_idle_over_decision_steps_completed",
        "rt_timeout_rule": "is_human_idle==1 -> budget",
    }])
    save_csv(qc, paths["qc"] / "behavior_pipeline_qc.csv")
    print(f"Behavior reanalysis complete: {paths['root']}")


if __name__ == "__main__":
    main()
