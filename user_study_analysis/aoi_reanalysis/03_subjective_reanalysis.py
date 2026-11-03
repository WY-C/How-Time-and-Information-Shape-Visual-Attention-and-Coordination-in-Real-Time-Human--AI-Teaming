#!/usr/bin/env python3
"""Subjective analysis retained as 3(Time) x 3(Horizon) repeated-measures ANOVA."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import (
    HORIZON_LEVELS, TIME_LEVELS, ensure_output_tree,
    group_mean_ci95, load_trials, save_csv,
)
from rm_stats import paired_posthoc, rm_anova_2way


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--subjective", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def find_column(columns, text: str) -> str:
    matches = [c for c in columns if text in str(c)]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify questionnaire column containing {text!r}; matches={matches}")
    return matches[0]


def _rm_posthoc_families(cell: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
    for factor in ["Time", "Horizon"]:
        q = paired_posthoc(cell, outcome, factor)
        q["analysis"] = "Subjective RM-ANOVA post-hoc"
        q["outcome"] = outcome
        q["posthoc_family"] = f"{outcome}: marginal {factor}"
        q["comparison_scope"] = "marginal"
        q["conditioning_factor"] = ""
        q["conditioning_level"] = ""
        rows.append(q)
    for h in HORIZON_LEVELS:
        q = paired_posthoc(cell[cell["Horizon"].eq(h)], outcome, "Time")
        q["analysis"] = "Subjective RM-ANOVA post-hoc"
        q["outcome"] = outcome
        q["posthoc_family"] = f"{outcome}: Time within Horizon={h}"
        q["comparison_scope"] = "simple_effect"
        q["conditioning_factor"] = "Horizon"
        q["conditioning_level"] = h
        rows.append(q)
    for t in TIME_LEVELS:
        q = paired_posthoc(cell[cell["Time"].eq(t)], outcome, "Horizon")
        q["analysis"] = "Subjective RM-ANOVA post-hoc"
        q["outcome"] = outcome
        q["posthoc_family"] = f"{outcome}: Horizon within Time={t}"
        q["comparison_scope"] = "simple_effect"
        q["conditioning_factor"] = "Time"
        q["conditioning_level"] = t
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args(); paths = ensure_output_tree(args.out)
    trial = load_trials(args.roots, strict_main=True)
    required_trial = ["condition_id"]
    missing = [c for c in required_trial if c not in trial.columns]
    if missing:
        raise ValueError(f"trial.csv missing columns needed for subjective mapping: {missing}")
    lookup = trial[["pid", "condition_id", "Layout", "Time", "Horizon"]].copy()
    lookup["Condition"] = lookup["condition_id"].astype(str).str.upper()
    lookup["Layout"] = lookup["Layout"].astype(str).str.upper()
    lookup = lookup[["pid", "Condition", "Layout", "Time", "Horizon"]].drop_duplicates()

    q = pd.read_excel(args.subjective)
    q["pid"] = q["Participant ID"].astype(str).str.strip().str.upper()
    q["Condition"] = q["Condition Num"].astype(str).str.strip().str.upper()
    q["Layout"] = q["Layout"].astype(str).str.strip().str.upper()
    if "타임스탬프" in q.columns:
        q["_timestamp"] = pd.to_datetime(q["타임스탬프"], errors="coerce")
        q = q.sort_values("_timestamp")
    q = q.drop_duplicates(["pid", "Condition", "Layout"], keep="last")

    cmap = {
        find_column(q.columns, "정신적으로 부담스러웠다"): "Mental demand",
        find_column(q.columns, "의사결정 시간은 행동을 결정하기에 충분했다"): "Time sufficiency",
        find_column(q.columns, "AI 파트너의 다음 행동을 예측하기 쉬웠다"): "AI predictability",
        find_column(q.columns, "정보의 양은 주어진 의사결정 시간에 비해"): "Perceived information amount",
    }
    q = q.rename(columns=cmap)
    for c in cmap.values(): q[c] = pd.to_numeric(q[c], errors="coerce")
    q["Information appropriateness |rating-4|"] = (q["Perceived information amount"] - 4).abs()
    q = q.merge(lookup, on=["pid", "Condition", "Layout"], how="inner", validate="one_to_one")
    save_csv(q, paths["raw"] / "subjective_mapped_trial_values.csv")

    outcomes = list(cmap.values()) + ["Information appropriateness |rating-4|"]
    cells_all, anovas, posts, descs = [], [], [], []
    for outcome in outcomes:
        cell = q.groupby(["pid", "Time", "Horizon"], as_index=False)[outcome].mean()
        cell["outcome"] = outcome; cells_all.append(cell.rename(columns={outcome: "value"}))
        r = rm_anova_2way(cell, "pid", "Time", "Horizon", outcome, alpha=args.alpha)
        r.insert(0, "outcome", outcome); anovas.append(r)
        posts.append(_rm_posthoc_families(cell, outcome))
        d = group_mean_ci95(cell, ["Time", "Horizon"], outcome); d["outcome"] = outcome; descs.append(d)

        fig, ax = plt.subplots(figsize=(7,5))
        for h in HORIZON_LEVELS:
            z = group_mean_ci95(cell[cell["Horizon"].eq(h)], ["Time"], outcome)
            ax.errorbar(
                z["Time"], z["mean"], yerr=z["ci95"],
                marker="o", capsize=4, elinewidth=1.0, label=f"H{h}"
            )
        ax.set_xticks(TIME_LEVELS); ax.set_xlabel("Available decision time (ms)"); ax.set_ylabel("Rating"); ax.set_title(f"{outcome} — mean ± 95% CI"); ax.legend()
        fig.tight_layout(); safe = "".join(ch if ch.isalnum() else "_" for ch in outcome).strip("_").lower()
        fig.savefig(paths["fig"] / f"subjective_{safe}.png", dpi=200); plt.close(fig)

    save_csv(pd.concat(cells_all, ignore_index=True), paths["inputs"] / "subjective_participant_condition_values.csv")
    save_csv(pd.concat(anovas, ignore_index=True), paths["stats"] / "subjective_rm_anova.csv")
    save_csv(pd.concat(posts, ignore_index=True), paths["stats"] / "subjective_posthoc_holm.csv")
    save_csv(pd.concat(descs, ignore_index=True), paths["desc"] / "subjective_descriptives.csv")
    print(f"Subjective reanalysis complete: {paths['root']}")


if __name__ == "__main__":
    main()
