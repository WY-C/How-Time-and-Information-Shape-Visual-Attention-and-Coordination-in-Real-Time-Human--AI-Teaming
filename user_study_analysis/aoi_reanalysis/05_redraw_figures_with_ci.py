#!/usr/bin/env python3
"""Redraw all paper-facing figures from an existing results/ directory.

Uncertainty rule
----------------
All error bars are participant-level two-sided t-based 95% confidence intervals:
    mean ± t_(.975, n-1) * SD / sqrt(n)
The plotting unit is the participant, not the trial or gaze bin.

100% stacked composition figures are retained without segment error bars because
individual segment CIs are not interpretable on a cumulative stack axis. Their
statistically interpretable uncertainty is shown in companion CI figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

TIME_LEVELS = [200, 400, 600]
HORIZON_LEVELS = [0, 2, 4]
FUTURE_STEPS = [1, 2, 3, 4]
AOIS = ["Self", "Partner", "Information", "Other"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Existing results directory")
    p.add_argument("--figdir", default=None, help="Output directory; default=<results>/04_figures")
    return p.parse_args()


def mean_ci95(data: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    g = (
        data.groupby(groups, dropna=False)[value]
        .agg(count="count", mean="mean", std="std")
        .reset_index()
    )
    g["sem"] = g["std"] / np.sqrt(g["count"])
    crit = np.where(
        g["count"].to_numpy() > 1,
        student_t.ppf(0.975, g["count"].to_numpy() - 1),
        np.nan,
    )
    g["ci95"] = g["sem"] * crit
    return g


def errorbar(ax, g: pd.DataFrame, x: str, *, scale=1.0, label=None,
             marker="o", capsize=4.0, alpha=1.0) -> None:
    ax.errorbar(
        g[x], g["mean"] * scale,
        yerr=g["ci95"] * scale,
        marker=marker, capsize=capsize,
        elinewidth=1.0, alpha=alpha, label=label,
    )


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def participant_overall(inp: pd.DataFrame) -> pd.DataFrame:
    cols = [f"{a.lower()}_prop" for a in AOIS]
    return inp.groupby(["pid", "Time", "Horizon"], as_index=False)[cols].mean()


def draw_behavior(results: Path, figdir: Path) -> None:
    d = pd.read_csv(results / "01_analysis_inputs" / "behavior_participant_condition_values.csv")
    specs = [
        ("score_final", "Task score", "Score", 1.0),
        ("num_collisions", "Collision count", "Count", 1.0),
        ("timeout_rate", "Deadline timeout rate", "Timeout (%)", 100.0),
        ("mean_response_time_with_timeout_ms", "Response time (timeout = budget)", "Response time (ms)", 1.0),
    ]
    for col, title, ylabel, scale in specs:
        fig, ax = plt.subplots(figsize=(7, 5))
        for h in HORIZON_LEVELS:
            g = mean_ci95(d[d["Horizon"].eq(h)], ["Time"], col)
            errorbar(ax, g, "Time", scale=scale, label=f"H{h}")
        ax.set_xticks(TIME_LEVELS)
        ax.set_xlabel("Available decision time (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — mean ± 95% CI")
        ax.legend()
        save(fig, figdir / f"behavior_{col}.png")


def draw_overall(results: Path, figdir: Path) -> None:
    raw = pd.read_csv(results / "01_analysis_inputs" / "overall_trial_lmm_input.csv")
    d = participant_overall(raw)
    props = [f"{a.lower()}_prop" for a in AOIS]
    means = d.groupby(["Time", "Horizon"], as_index=False)[props].mean()
    labels = [f"{int(r.Time)}ms\nH{int(r.Horizon)}" for r in means.itertuples(index=False)]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(means))
    for a in AOIS:
        vals = means[f"{a.lower()}_prop"].to_numpy() * 100
        ax.bar(labels, vals, bottom=bottom, label=a)
        bottom += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Proportion of fixation dwell (%)")
    ax.set_title("Overall gaze composition (uncertainty shown in companion CI plot)")
    ax.legend(ncol=4)
    save(fig, figdir / "overall_aoi_100pct_stacked.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, h in zip(axes, HORIZON_LEVELS):
        sub = d[d["Horizon"].eq(h)]
        for a in AOIS:
            g = mean_ci95(sub, ["Time"], f"{a.lower()}_prop")
            errorbar(ax, g, "Time", scale=100, label=a)
        ax.set_title(f"H{h}")
        ax.set_xlabel("Available decision time (ms)")
        ax.set_xticks(TIME_LEVELS)
    axes[0].set_ylabel("Fixation dwell proportion (%)")
    axes[-1].legend(ncol=2)
    fig.suptitle("Overall AOI allocation — mean ± 95% CI")
    save(fig, figdir / "overall_aoi_change_by_time.png")

    # Information only, H2 vs H4.
    fig, ax = plt.subplots(figsize=(7, 5))
    for h in [2, 4]:
        g = mean_ci95(d[d["Horizon"].eq(h)], ["Time"], "information_prop")
        errorbar(ax, g, "Time", scale=100, label=f"H{h}")
    ax.set_xticks(TIME_LEVELS)
    ax.set_xlabel("Available decision time (ms)")
    ax.set_ylabel("Information dwell proportion (%)")
    ax.set_title("Action preview gaze — mean ± 95% CI")
    ax.legend()
    save(fig, figdir / "information_gaze_h2_vs_h4.png")


def draw_temporal(results: Path, figdir: Path, filename: str, resolution: str) -> None:
    d = pd.read_csv(results / "01_analysis_inputs" / filename)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, t in zip(axes, TIME_LEVELS):
        sub = d[d["Time"].eq(t)]
        for a in AOIS:
            g = mean_ci95(sub, ["BinStartMs"], f"{a.lower()}_prop")
            fine = resolution != "100ms"
            errorbar(
                ax, g, "BinStartMs", scale=100,
                label=a, marker="o" if not fine else None,
                capsize=0 if fine else 3,
                alpha=0.85 if fine else 1.0,
            )
        ax.set_title(f"{t} ms budget")
        ax.set_xlabel("Elapsed time from decision onset (ms)")
        if fine:
            ax.set_xticks(np.arange(0, t + 1, 100))
    axes[0].set_ylabel("Fixation dwell proportion (%)")
    axes[-1].legend(ncol=2)
    fig.suptitle(f"Temporal evolution of 4 AOIs — {resolution}, mean ± 95% CI")
    out = "temporal_4aoi_100ms.png" if resolution == "100ms" else "temporal_4aoi_16_67ms.png"
    save(fig, figdir / out)


def draw_response(results: Path, figdir: Path, mode: str) -> None:
    trial = pd.read_csv(results / "01_analysis_inputs" / f"response_{mode}_trial_phase_lmm_input.csv")
    props = [f"{a.lower()}_prop" for a in AOIS]
    d = trial.groupby(["pid", "Time", "Mode", "Phase"], as_index=False)[props].mean()
    phases = ["Pre", "Post"]

    # Composition view.
    means = d.groupby(["Time", "Phase"], as_index=False)[props].mean()
    labels, rows = [], []
    for t in TIME_LEVELS:
        for ph in phases:
            q = means[(means["Time"].eq(t)) & (means["Phase"].eq(ph))]
            if len(q):
                labels.append(f"{t}ms\n{ph}")
                rows.append(q.iloc[0])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(rows))
    for a in AOIS:
        vals = np.array([r[f"{a.lower()}_prop"] for r in rows]) * 100
        ax.bar(labels, vals, bottom=bottom, label=a)
        bottom += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Fixation dwell proportion (%)")
    ax.set_title(f"Response-locked gaze composition — {mode} (CI in companion figure)")
    ax.legend(ncol=4)
    save(fig, figdir / f"response_pre_post_100pct_{mode}.png")

    # CI view.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, a in zip(axes.ravel(), AOIS):
        for ph in phases:
            g = mean_ci95(d[d["Phase"].eq(ph)], ["Time"], f"{a.lower()}_prop")
            errorbar(ax, g, "Time", scale=100, label=ph)
        ax.set_title(a)
        ax.set_xticks(TIME_LEVELS)
        ax.set_xlabel("Available decision time (ms)")
        ax.set_ylabel("Fixation dwell proportion (%)")
        ax.legend()
    fig.suptitle(f"Response-locked PRE/POST gaze — {mode}, mean ± 95% CI")
    save(fig, figdir / f"response_pre_post_ci_{mode}.png")


def draw_h1h4(results: Path, figdir: Path) -> None:
    raw = pd.read_csv(results / "01_analysis_inputs" / "h1_h4_trial_lmm_input.csv")
    cell = raw.groupby(["pid", "Time", "FutureStep"], as_index=False).agg(
        dwell_prop_total=("dwell_prop_total", "mean"),
        relative_allocation=("relative_allocation", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for h in FUTURE_STEPS:
        g = mean_ci95(cell[cell["FutureStep"].eq(h)], ["Time"], "dwell_prop_total")
        errorbar(ax, g, "Time", scale=100, label=f"H{h}")
    ax.set_xticks(TIME_LEVELS)
    ax.set_xlabel("Available decision time (ms)")
    ax.set_ylabel("Proportion of total fixation (%)")
    ax.set_title("H1–H4 dwell proportion of total fixation — mean ± 95% CI")
    ax.legend()
    save(fig, figdir / "h1_h4_prop_total_fixation.png")

    rel = pd.read_csv(results / "01_analysis_inputs" / "h1_h4_relative_participant_lmm_input.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in FUTURE_STEPS:
        g = mean_ci95(rel[rel["FutureStep"].eq(h)], ["Time"], "relative_allocation")
        errorbar(ax, g, "Time", scale=100, label=f"H{h}")
    ax.set_xticks(TIME_LEVELS)
    ax.set_xlabel("Available decision time (ms)")
    ax.set_ylabel("Share of Information dwell (%)")
    ax.set_title("H1–H4 relative allocation within Information — mean ± 95% CI")
    ax.legend()
    save(fig, figdir / "h1_h4_relative_allocation.png")

    nfraw = pd.read_csv(results / "01_analysis_inputs" / "h1_h4_near_far_trial_lmm_input.csv")
    nf = nfraw.groupby(["pid", "Time", "Range"], as_index=False)["dwell_prop_total"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    for rng in ["Near", "Far"]:
        g = mean_ci95(nf[nf["Range"].eq(rng)], ["Time"], "dwell_prop_total")
        errorbar(ax, g, "Time", scale=100, label=rng)
    ax.set_xticks(TIME_LEVELS)
    ax.set_xlabel("Available decision time (ms)")
    ax.set_ylabel("Proportion of total fixation (%)")
    ax.set_title("Near (H1+H2) vs Far (H3+H4) — mean ± 95% CI")
    ax.legend()
    save(fig, figdir / "h1_h4_near_far_total_fixation.png")


def draw_subjective(results: Path, figdir: Path) -> None:
    d = pd.read_csv(results / "01_analysis_inputs" / "subjective_participant_condition_values.csv")
    for outcome in d["outcome"].drop_duplicates():
        sub = d[d["outcome"].eq(outcome)]
        fig, ax = plt.subplots(figsize=(7, 5))
        for h in HORIZON_LEVELS:
            g = mean_ci95(sub[sub["Horizon"].eq(h)], ["Time"], "value")
            errorbar(ax, g, "Time", label=f"H{h}")
        ax.set_xticks(TIME_LEVELS)
        ax.set_xlabel("Available decision time (ms)")
        ax.set_ylabel("Rating")
        ax.set_title(f"{outcome} — mean ± 95% CI")
        ax.legend()
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in outcome).strip("_").lower()
        save(fig, figdir / f"subjective_{safe_name}.png")


def main() -> None:
    args = parse_args()
    results = Path(args.results).expanduser().resolve()
    figdir = Path(args.figdir).expanduser().resolve() if args.figdir else results / "04_figures"
    figdir.mkdir(parents=True, exist_ok=True)

    draw_behavior(results, figdir)
    draw_overall(results, figdir)
    draw_temporal(results, figdir, "temporal_100ms_participant_lmm_input.csv", "100ms")
    draw_temporal(results, figdir, "temporal_16.67ms_participant_lmm_input.csv", "16.67ms")
    draw_response(results, figdir, "whole")
    draw_response(results, figdir, "symmetric")
    draw_h1h4(results, figdir)
    draw_subjective(results, figdir)
    print(f"Redrew figures with participant-level 95% CI: {figdir}")


if __name__ == "__main__":
    main()
