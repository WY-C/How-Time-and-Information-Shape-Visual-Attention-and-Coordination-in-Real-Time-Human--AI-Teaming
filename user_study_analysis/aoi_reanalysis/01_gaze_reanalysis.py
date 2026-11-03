#!/usr/bin/env python3
"""Canonical gaze reanalysis under the 100%-partition AOI contract.

Four top-level AOIs are mutually exhaustive after fractional overlap allocation:
    Self + Partner + Information + Other = 100%.

Overlap rule:
- Information alone on a non-self/non-partner location -> 100% Information.
- Self + Information -> 50/50.
- Partner + Information -> 50/50.
- No Self/Partner/Information match -> 100% Other.
- H1-H4 overlap is divided equally *within the Information dwell allocated above*.

Paper-facing gaze families:
1. Overall AOI composition (trial-wide): all four AOIs together.
2. Temporal evolution at 16.67 ms (1000/60) bins.
3. Temporal evolution at 100 ms bins.
4. Response-locked PRE/POST gaze (whole-window + symmetric 100-ms sensitivity).
5. Information gaze across Time x displayed Horizon (H2 vs H4).
6. Hierarchical H1-H4 dwell proportion of total fixation in H4 trials.
7. Hierarchical H1-H4 relative allocation within top-level Information dwell.
8. Hierarchical Near (H1+H2) vs Far (H3+H4) allocation.

Every analysis saves pre-aggregation raw/intermediate values before descriptives/statistics.

All inferential gaze analyses use adjusted-logit Linear Mixed-Effects Models (LMMs).
Behavioral and subjective RM-ANOVA are implemented in separate modules.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    AOIS, COARSE_BIN_MS, EXPECTED_MAIN_TRIALS, FINE_BIN_MS, FUTURE_STEPS,
    HORIZON_LEVELS, TIME_LEVELS, adjusted_logit, allocate_future_steps,
    allocate_top_level, audit_composition, discover_files, ensure_output_tree,
    load_steps, load_trials, matched_future_steps, overlap_ms, parse_segments,
    save_csv, segment_has_information, group_mean_ci95,
)
from lmm_stats import fit_lmm, holm_adjust, lmm_emm_contrast, lmm_term_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--symmetric-ms", type=float, default=100.0)
    p.add_argument("--min-symmetric-ms", type=float, default=50.0)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def build_segment_allocations(roots: list[str], trial: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    meta = trial.set_index("trial_id")[["pid", "Time", "Horizon", "Layout"]].to_dict("index")
    onset = {(r.trial_id, int(r.step)): float(r.state_onset_ms) for r in steps.itertuples(index=False)}
    valid = set(meta)
    rows = []
    segment_id = 0

    for f in discover_files(roots, "fixation.csv"):
        head = pd.read_csv(f, nrows=0)
        if not {"trial_id", "context_segments"}.issubset(head.columns):
            continue
        x = pd.read_csv(f, usecols=["trial_id", "context_segments"])
        x = x[x["trial_id"].isin(valid)]
        for tid, value in zip(x["trial_id"], x["context_segments"]):
            m = meta[tid]
            for seg in parse_segments(value):
                try:
                    s = float(seg.get("start_ms")); e = float(seg.get("end_ms"))
                except Exception:
                    continue
                dur = e - s
                if not np.isfinite(dur) or dur <= 0:
                    continue
                segment_id += 1
                alloc = allocate_top_level(seg, dur)
                future = allocate_future_steps(seg, alloc["Information"])
                try:
                    step = int(seg.get("step"))
                except Exception:
                    step = np.nan
                state_onset = onset.get((tid, int(step))) if np.isfinite(step) else np.nan
                rec = {
                    "segment_id": segment_id,
                    "pid": m["pid"], "trial_id": tid, "Time": int(m["Time"]),
                    "Horizon": int(m["Horizon"]), "Layout": m["Layout"],
                    "step": step, "state_onset_ms": state_onset,
                    "start_ms": s, "end_ms": e, "duration_ms": dur,
                    "base_aoi_logged": seg.get("base_aoi"),
                    "information_match": int(segment_has_information(seg)),
                    "matched_future_steps": json.dumps(matched_future_steps(seg)),
                }
                for a in AOIS:
                    rec[f"{a.lower()}_ms"] = alloc[a]
                    rec[f"{a.lower()}_weight"] = alloc[a] / dur
                for h in FUTURE_STEPS:
                    rec[f"h{h}_ms"] = future[h]
                    rec[f"h{h}_weight"] = future[h] / dur
                rows.append(rec)
    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError("No usable fixation context segments were found")
    err = (d[[f"{a.lower()}_ms" for a in AOIS]].sum(axis=1) - d["duration_ms"]).abs().max()
    if err > 1e-6:
        raise RuntimeError(f"Segment AOI conservation failed: max error={err}")
    return d


def build_overall(seg: pd.DataFrame, trial: pd.DataFrame) -> pd.DataFrame:
    agg = {"duration_ms": "sum"}
    for a in AOIS:
        agg[f"{a.lower()}_ms"] = "sum"
    for h in FUTURE_STEPS:
        agg[f"h{h}_ms"] = "sum"
    x = seg.groupby(["pid", "trial_id", "Time", "Horizon", "Layout"], as_index=False).agg(agg)
    x = x.rename(columns={"duration_ms": "total_fixation_ms"})
    for a in AOIS:
        c = a.lower()
        x[f"{c}_prop"] = x[f"{c}_ms"] / x["total_fixation_ms"]
    for h in FUTURE_STEPS:
        x[f"h{h}_prop_total"] = x[f"h{h}_ms"] / x["total_fixation_ms"]
    audit_composition(x, "overall")
    # Ensure every completed trial is represented.
    missing = sorted(set(trial["trial_id"]) - set(x["trial_id"]))
    if missing:
        raise RuntimeError(f"Completed trials without fixation dwell: {missing[:10]} (n={len(missing)})")
    return x


def _bin_count(time_ms: int, width_ms: float) -> int:
    return int(round(float(time_ms) / float(width_ms)))


def build_temporal(seg: pd.DataFrame, width_ms: float, label: str) -> pd.DataFrame:
    acc: dict[tuple, dict] = {}
    for r in seg.itertuples(index=False):
        if not np.isfinite(r.step) or not np.isfinite(r.state_onset_ms):
            continue
        onset = float(r.state_onset_ms); T = int(r.Time)
        cs = max(float(r.start_ms), onset); ce = min(float(r.end_ms), onset + T)
        if ce <= cs:
            continue
        rel_s, rel_e = cs - onset, ce - onset
        n_bins = _bin_count(T, width_ms)
        b0 = max(0, int(math.floor(rel_s / width_ms + 1e-12)))
        b1 = min(n_bins - 1, int(math.floor((rel_e - 1e-9) / width_ms)))
        for b in range(b0, b1 + 1):
            bs, be = b * width_ms, min((b + 1) * width_ms, T)
            ov = overlap_ms(rel_s, rel_e, bs, be)
            if ov <= 0:
                continue
            key = (r.pid, r.trial_id, int(r.Time), int(r.Horizon), r.Layout, b)
            a = acc.setdefault(key, {"total_fixation_ms": 0.0, **{f"{z.lower()}_ms": 0.0 for z in AOIS}})
            a["total_fixation_ms"] += ov
            for z in AOIS:
                a[f"{z.lower()}_ms"] += ov * float(getattr(r, f"{z.lower()}_weight"))

    # Emit every structurally available bin so zero-dwell bins are explicit in raw output.
    trial_meta = seg[["pid", "trial_id", "Time", "Horizon", "Layout"]].drop_duplicates()
    rows = []
    for m in trial_meta.itertuples(index=False):
        n_bins = _bin_count(int(m.Time), width_ms)
        for b in range(n_bins):
            key = (m.pid, m.trial_id, int(m.Time), int(m.Horizon), m.Layout, b)
            a = acc.get(key, {"total_fixation_ms": 0.0, **{f"{z.lower()}_ms": 0.0 for z in AOIS}})
            rec = {
                "pid": m.pid, "trial_id": m.trial_id, "Time": int(m.Time),
                "Horizon": int(m.Horizon), "Layout": m.Layout,
                "Resolution": label, "Bin": b,
                "BinStartMs": b * width_ms,
                "BinEndMs": min((b + 1) * width_ms, int(m.Time)),
                **a,
            }
            total = float(a["total_fixation_ms"])
            for z in AOIS:
                rec[f"{z.lower()}_prop"] = a[f"{z.lower()}_ms"] / total if total > 0 else np.nan
            rows.append(rec)
    d = pd.DataFrame(rows)
    audit_composition(d, f"temporal_{label}", allow_empty=True)
    return d


def overall_lmm_inputs(overall: pd.DataFrame) -> pd.DataFrame:
    d = overall.copy()
    for a in AOIS:
        c = a.lower()
        d[f"{c}_logit"] = adjusted_logit(d[f"{c}_prop"])
    return d


def participant_overall_descriptives(overall: pd.DataFrame) -> pd.DataFrame:
    cols = [f"{a.lower()}_prop" for a in AOIS]
    return overall.groupby(["pid", "Time", "Horizon"], as_index=False)[cols].mean()


def _raw_marginal_mean(data: pd.DataFrame, value: str, *, factor: str, level, fixed: dict | None = None) -> float:
    d = data.copy()
    fixed = fixed or {}
    d = d[d[factor].eq(level)]
    for col, val in fixed.items():
        d = d[d[col].eq(val)]
    if d.empty:
        return np.nan
    # Participant is the descriptive unit: average observations within participant first.
    return float(d.groupby("pid")[value].mean().mean())


def _finalize_lmm_family(rows: list[dict], alpha: float, *, family: str, scope: str,
                         conditioning_factor: str = "", conditioning_level="") -> pd.DataFrame:
    out = holm_adjust(rows, alpha)
    if out.empty:
        return out
    out.insert(0, "posthoc_family", family)
    out.insert(1, "comparison_scope", scope)
    out.insert(2, "conditioning_factor", conditioning_factor)
    out.insert(3, "conditioning_level", conditioning_level)
    return out


def overall_statistics(lmm_input: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overall gaze LMMs plus marginal and simple-effect post-hoc tests.

    The full 3x3 Information model is intentionally NOT fit because H0 is a
    structural zero for Information. Information inference is exclusively the
    H2-vs-H4 primary LMM.
    """
    stats_rows, post_rows = [], []

    # Self / Partner / Other: full 3 x 3 model.
    for a in [x for x in AOIS if x != "Information"]:
        c = a.lower()
        fit = fit_lmm(
            lmm_input,
            f"{c}_logit ~ C(Time, Sum) * C(Horizon, Sum) + C(Layout, Sum)",
            required_cols=[f"{c}_logit", "Time", "Horizon", "Layout"],
        )
        analysis = "Overall 4-AOI composition (3x3)"
        stats_rows.append(lmm_term_table(
            fit, analysis=analysis, aoi=a,
            target_terms=["Time", "Horizon", "Time×Horizon"],
        ))

        # Marginal Time and Horizon pairwise comparisons.
        for factor, levels, marginal in [
            ("Time", TIME_LEVELS, ["Horizon", "Layout"]),
            ("Horizon", HORIZON_LEVELS, ["Time", "Layout"]),
        ]:
            fam = []
            for l1, l2 in combinations_local(levels):
                r = lmm_emm_contrast(fit, factor=factor, level1=l1, level2=l2, marginal_over=marginal)
                r.update({
                    "analysis": analysis, "AOI": a,
                    "raw_mean_level1": _raw_marginal_mean(lmm_input, f"{c}_prop", factor=factor, level=l1),
                    "raw_mean_level2": _raw_marginal_mean(lmm_input, f"{c}_prop", factor=factor, level=l2),
                })
                r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
                fam.append(r)
            post_rows.append(_finalize_lmm_family(
                fam, alpha, family=f"{a}: marginal {factor}", scope="marginal"
            ))

        # Simple Time effects within each Horizon.
        for h in HORIZON_LEVELS:
            fam = []
            for l1, l2 in combinations_local(TIME_LEVELS):
                r = lmm_emm_contrast(
                    fit, factor="Time", level1=l1, level2=l2,
                    fixed={"Horizon": h}, marginal_over=["Layout"],
                )
                r.update({
                    "analysis": analysis, "AOI": a,
                    "raw_mean_level1": _raw_marginal_mean(lmm_input, f"{c}_prop", factor="Time", level=l1, fixed={"Horizon": h}),
                    "raw_mean_level2": _raw_marginal_mean(lmm_input, f"{c}_prop", factor="Time", level=l2, fixed={"Horizon": h}),
                })
                r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
                fam.append(r)
            post_rows.append(_finalize_lmm_family(
                fam, alpha, family=f"{a}: Time within Horizon={h}", scope="simple_effect",
                conditioning_factor="Horizon", conditioning_level=h,
            ))

        # Simple Horizon effects within each Time.
        for t in TIME_LEVELS:
            fam = []
            for l1, l2 in combinations_local(HORIZON_LEVELS):
                r = lmm_emm_contrast(
                    fit, factor="Horizon", level1=l1, level2=l2,
                    fixed={"Time": t}, marginal_over=["Layout"],
                )
                r.update({
                    "analysis": analysis, "AOI": a,
                    "raw_mean_level1": _raw_marginal_mean(lmm_input, f"{c}_prop", factor="Horizon", level=l1, fixed={"Time": t}),
                    "raw_mean_level2": _raw_marginal_mean(lmm_input, f"{c}_prop", factor="Horizon", level=l2, fixed={"Time": t}),
                })
                r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
                fam.append(r)
            post_rows.append(_finalize_lmm_family(
                fam, alpha, family=f"{a}: Horizon within Time={t}", scope="simple_effect",
                conditioning_factor="Time", conditioning_level=t,
            ))

    # Primary Information inference excludes structural H0.
    info = lmm_input[lmm_input["Horizon"].isin([2, 4])].copy()
    fit = fit_lmm(
        info,
        "information_logit ~ C(Time, Sum) * C(Horizon, Sum) + C(Layout, Sum)",
        required_cols=["information_logit", "Time", "Horizon", "Layout"],
    )
    analysis = "Information gaze primary (H2 vs H4)"
    stats_rows.append(lmm_term_table(
        fit, analysis=analysis, aoi="Information",
        target_terms=["Time", "Horizon", "Time×Horizon"],
    ))

    for factor, levels, marginal in [
        ("Time", TIME_LEVELS, ["Horizon", "Layout"]),
        ("Horizon", [2, 4], ["Time", "Layout"]),
    ]:
        fam = []
        for l1, l2 in combinations_local(levels):
            r = lmm_emm_contrast(fit, factor=factor, level1=l1, level2=l2, marginal_over=marginal)
            r.update({
                "analysis": analysis, "AOI": "Information",
                "raw_mean_level1": _raw_marginal_mean(info, "information_prop", factor=factor, level=l1),
                "raw_mean_level2": _raw_marginal_mean(info, "information_prop", factor=factor, level=l2),
            })
            r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
            fam.append(r)
        post_rows.append(_finalize_lmm_family(
            fam, alpha, family=f"Information: marginal {factor}", scope="marginal"
        ))

    for h in [2, 4]:
        fam = []
        for l1, l2 in combinations_local(TIME_LEVELS):
            r = lmm_emm_contrast(
                fit, factor="Time", level1=l1, level2=l2,
                fixed={"Horizon": h}, marginal_over=["Layout"],
            )
            r.update({
                "analysis": analysis, "AOI": "Information",
                "raw_mean_level1": _raw_marginal_mean(info, "information_prop", factor="Time", level=l1, fixed={"Horizon": h}),
                "raw_mean_level2": _raw_marginal_mean(info, "information_prop", factor="Time", level=l2, fixed={"Horizon": h}),
            })
            r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
            fam.append(r)
        post_rows.append(_finalize_lmm_family(
            fam, alpha, family=f"Information: Time within Horizon={h}", scope="simple_effect",
            conditioning_factor="Horizon", conditioning_level=h,
        ))

    for t in TIME_LEVELS:
        fam = []
        r = lmm_emm_contrast(
            fit, factor="Horizon", level1=2, level2=4,
            fixed={"Time": t}, marginal_over=["Layout"],
        )
        r.update({
            "analysis": analysis, "AOI": "Information",
            "raw_mean_level1": _raw_marginal_mean(info, "information_prop", factor="Horizon", level=2, fixed={"Time": t}),
            "raw_mean_level2": _raw_marginal_mean(info, "information_prop", factor="Horizon", level=4, fixed={"Time": t}),
        })
        r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
        fam.append(r)
        post_rows.append(_finalize_lmm_family(
            fam, alpha, family=f"Information: H4 vs H2 within Time={t}", scope="simple_effect",
            conditioning_factor="Time", conditioning_level=t,
        ))

    return (
        pd.concat(stats_rows, ignore_index=True),
        pd.concat([x for x in post_rows if len(x)], ignore_index=True),
    )

def combinations_local(levels):
    levels = list(levels)
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            yield levels[i], levels[j]

def temporal_participant_inputs(d: pd.DataFrame) -> pd.DataFrame:
    # Participant-level values are descriptive/plotting summaries only, not inferential inputs.
    nonempty = d[d["total_fixation_ms"] > 0].copy()
    prop_cols = [f"{a.lower()}_prop" for a in AOIS]
    out = nonempty.groupby(["pid", "Time", "Resolution", "Bin", "BinStartMs", "BinEndMs"], as_index=False).agg(
        **{c: (c, "mean") for c in prop_cols},
        n_trial_bins_contributing=("trial_id", "size"),
    )
    for a in AOIS:
        c = a.lower()
        out[f"{c}_logit"] = adjusted_logit(out[f"{c}_prop"])
    return out


def temporal_statistics(inputs: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant-trajectory LMM temporal inference plus automatic post-hoc.

    100-ms resolution: all pairwise bin contrasts, Holm corrected within each
    AOI x Time family. 16.67-ms resolution: adjacent-bin contrasts only, Holm
    corrected within each AOI x Time family; exhaustive frame-pair testing would
    be unnecessarily large (up to 630 comparisons per family).
    """
    within_rows, post_rows = [], []
    for a in AOIS:
        c = a.lower()
        for (res, time), g in inputs.groupby(["Resolution", "Time"]):
            if g["Bin"].nunique() < 2:
                continue
            fit = fit_lmm(
                g,
                f"{c}_logit ~ C(Bin, Sum)",
                required_cols=[f"{c}_logit", "Bin"],
            )
            tab = lmm_term_table(
                fit, analysis="Within-budget temporal change", aoi=a, target_terms=["Bin"],
            )
            tab.insert(1, "Resolution", res)
            tab.insert(2, "Time", int(time))
            within_rows.append(tab)

            levels = sorted(g["Bin"].unique())
            if str(res) == "100ms":
                pairs = list(combinations_local(levels))
                scope = "all_pairwise_bins"
            else:
                pairs = list(zip(levels[:-1], levels[1:]))
                scope = "adjacent_bins"

            fam = []
            for b1, b2 in pairs:
                r = lmm_emm_contrast(fit, factor="Bin", level1=b1, level2=b2)
                r.update({
                    "analysis": "Temporal-bin LMM contrast", "Resolution": res,
                    "Time": int(time), "AOI": a, "Bin1": int(b1), "Bin2": int(b2),
                    "Start1Ms": float(g.loc[g["Bin"].eq(b1), "BinStartMs"].iloc[0]),
                    "Start2Ms": float(g.loc[g["Bin"].eq(b2), "BinStartMs"].iloc[0]),
                    "raw_mean_level1": float(g.loc[g["Bin"].eq(b1), f"{c}_prop"].mean()),
                    "raw_mean_level2": float(g.loc[g["Bin"].eq(b2), f"{c}_prop"].mean()),
                })
                r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
                fam.append(r)
            if fam:
                post_rows.append(_finalize_lmm_family(
                    fam, alpha,
                    family=f"{a}: {res} bins within Time={int(time)}",
                    scope=scope, conditioning_factor="Time", conditioning_level=int(time),
                ))

    return (
        pd.concat(within_rows, ignore_index=True) if within_rows else pd.DataFrame(),
        pd.concat([x for x in post_rows if len(x)], ignore_index=True) if post_rows else pd.DataFrame(),
    )

def build_response_events(steps: pd.DataFrame, trial: pd.DataFrame) -> pd.DataFrame:
    meta = trial[["trial_id", "pid", "Time", "Horizon", "Layout"]]
    e = steps[steps["human_action_due"].eq(1)].drop(columns=["pid"], errors="ignore").merge(meta, on="trial_id", how="inner", validate="many_to_one")
    e["eligible_prepost"] = e["response_time_ms"].notna() & e["response_time_ms"].gt(0) & e["response_time_ms"].lt(e["Time"])
    e["response_abs_ms"] = e["state_onset_ms"] + e["response_time_ms"]
    e["deadline_abs_ms"] = e["state_onset_ms"] + e["Time"]
    return e


def build_response_decision_phase(seg: pd.DataFrame, events: pd.DataFrame, mode: str,
                                  symmetric_ms: float, min_symmetric_ms: float) -> pd.DataFrame:
    e = events[events["eligible_prepost"]].copy()
    if mode == "symmetric":
        e["window_ms"] = np.minimum.reduce([
            np.full(len(e), float(symmetric_ms)), e["response_time_ms"].to_numpy(float),
            (e["Time"] - e["response_time_ms"]).to_numpy(float),
        ])
        e = e[e["window_ms"] >= min_symmetric_ms].copy()
    event_map = {(r.trial_id, int(r.step)): r for r in e.itertuples(index=False)}
    acc = {}
    for key in event_map:
        for phase in ["Pre", "Post"]:
            acc[(key[0], key[1], phase)] = {"total_fixation_ms": 0.0, **{f"{a.lower()}_ms": 0.0 for a in AOIS}}
    for r in seg.itertuples(index=False):
        if not np.isfinite(r.step):
            continue
        key = (r.trial_id, int(r.step))
        if key not in event_map:
            continue
        ev = event_map[key]
        if mode == "whole":
            windows = [("Pre", float(ev.state_onset_ms), float(ev.response_abs_ms)),
                       ("Post", float(ev.response_abs_ms), float(ev.deadline_abs_ms))]
        else:
            w = float(ev.window_ms)
            windows = [("Pre", float(ev.response_abs_ms)-w, float(ev.response_abs_ms)),
                       ("Post", float(ev.response_abs_ms), float(ev.response_abs_ms)+w)]
        for phase, left, right in windows:
            ov = overlap_ms(float(r.start_ms), float(r.end_ms), left, right)
            if ov <= 0: continue
            a = acc[(r.trial_id, int(r.step), phase)]
            a["total_fixation_ms"] += ov
            for z in AOIS:
                a[f"{z.lower()}_ms"] += ov * float(getattr(r, f"{z.lower()}_weight"))
    rows = []
    for (tid, step, phase), a in acc.items():
        ev = event_map[(tid, step)]
        rec = {"pid": ev.pid, "trial_id": tid, "step": step, "Time": int(ev.Time),
               "Horizon": int(ev.Horizon), "Layout": ev.Layout, "Mode": mode, "Phase": phase, **a}
        total = a["total_fixation_ms"]
        for z in AOIS:
            rec[f"{z.lower()}_prop"] = a[f"{z.lower()}_ms"] / total if total > 0 else np.nan
        rows.append(rec)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    # Keep decisions with positive fixation dwell in both phases.
    valid = (d[d["total_fixation_ms"] > 0].groupby(["trial_id", "step"])["Phase"].nunique())
    valid_pairs = set(valid[valid.eq(2)].index)
    keys = list(zip(d["trial_id"], d["step"]))
    d = d[pd.Series(keys, index=d.index).isin(valid_pairs)].copy()
    audit_composition(d, f"response_{mode}")
    return d


def response_participant_inputs(decision_phase: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prop_cols = [f"{a.lower()}_prop" for a in AOIS]
    # Equal-weight decisions -> trial-phase value. This is the inferential LMM input.
    trial_phase = decision_phase.groupby(["pid", "trial_id", "Time", "Horizon", "Layout", "Mode", "Phase"], as_index=False).agg(
        **{c: (c, "mean") for c in prop_cols}, n_decisions=("step", "nunique")
    )
    for a in AOIS:
        c = a.lower()
        trial_phase[f"{c}_logit"] = adjusted_logit(trial_phase[f"{c}_prop"])
    # Participant summaries are descriptive/plotting only.
    participant = trial_phase.groupby(["pid", "Time", "Mode", "Phase"], as_index=False)[prop_cols].mean()
    return trial_phase, participant


def response_statistics(trial_phase: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Response LMM plus marginal and interaction simple-effect contrasts."""
    stats_rows, pair_rows = [], []
    for mode, gm in trial_phase.groupby("Mode"):
        for a in AOIS:
            c = a.lower()
            fit = fit_lmm(
                gm,
                f"{c}_logit ~ C(Phase, Sum) * C(Time, Sum) + C(Horizon, Sum) + C(Layout, Sum)",
                trial_vc=True,
                required_cols=[f"{c}_logit", "Phase", "Time", "Horizon", "Layout", "trial_id"],
            )
            analysis = "Response PRE/POST Phase x Time"
            stats_rows.append(lmm_term_table(
                fit, analysis=analysis, aoi=a,
                target_terms=["Phase", "Time", "Phase×Time"],
            ).assign(Mode=mode))

            # Marginal Phase effect.
            fam = []
            r = lmm_emm_contrast(
                fit, factor="Phase", level1="Pre", level2="Post",
                marginal_over=["Time", "Horizon", "Layout"],
            )
            r.update({
                "analysis": "Response LMM contrast", "Mode": mode, "AOI": a,
                "raw_mean_level1": _raw_marginal_mean(gm, f"{c}_prop", factor="Phase", level="Pre"),
                "raw_mean_level2": _raw_marginal_mean(gm, f"{c}_prop", factor="Phase", level="Post"),
            })
            r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
            fam.append(r)
            pair_rows.append(_finalize_lmm_family(
                fam, alpha, family=f"{mode}/{a}: marginal Phase", scope="marginal"
            ))

            # Marginal Time comparisons.
            fam = []
            for t1, t2 in combinations_local(TIME_LEVELS):
                r = lmm_emm_contrast(
                    fit, factor="Time", level1=t1, level2=t2,
                    marginal_over=["Phase", "Horizon", "Layout"],
                )
                r.update({
                    "analysis": "Response LMM contrast", "Mode": mode, "AOI": a,
                    "raw_mean_level1": _raw_marginal_mean(gm, f"{c}_prop", factor="Time", level=t1),
                    "raw_mean_level2": _raw_marginal_mean(gm, f"{c}_prop", factor="Time", level=t2),
                })
                r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
                fam.append(r)
            pair_rows.append(_finalize_lmm_family(
                fam, alpha, family=f"{mode}/{a}: marginal Time", scope="marginal"
            ))

            # PRE vs POST within each Time; Holm across the three planned simple effects.
            fam = []
            for time in TIME_LEVELS:
                r = lmm_emm_contrast(
                    fit, factor="Phase", level1="Pre", level2="Post",
                    fixed={"Time": time}, marginal_over=["Horizon", "Layout"],
                )
                pre = _raw_marginal_mean(gm, f"{c}_prop", factor="Phase", level="Pre", fixed={"Time": time})
                post = _raw_marginal_mean(gm, f"{c}_prop", factor="Phase", level="Post", fixed={"Time": time})
                r.update({
                    "analysis": "Response LMM contrast", "Mode": mode, "AOI": a,
                    "Time": int(time), "pre_mean_prop": pre, "post_mean_prop": post,
                    "post_minus_pre": post - pre,
                })
                fam.append(r)
            pair_rows.append(_finalize_lmm_family(
                fam, alpha, family=f"{mode}/{a}: Phase within Time",
                scope="simple_effect", conditioning_factor="Time", conditioning_level="200|400|600",
            ))

            # Time pairwise within each Phase; Holm within each phase family.
            for phase in ["Pre", "Post"]:
                fam = []
                for t1, t2 in combinations_local(TIME_LEVELS):
                    r = lmm_emm_contrast(
                        fit, factor="Time", level1=t1, level2=t2,
                        fixed={"Phase": phase}, marginal_over=["Horizon", "Layout"],
                    )
                    r.update({
                        "analysis": "Response LMM contrast", "Mode": mode, "AOI": a, "Phase": phase,
                        "raw_mean_level1": _raw_marginal_mean(gm, f"{c}_prop", factor="Time", level=t1, fixed={"Phase": phase}),
                        "raw_mean_level2": _raw_marginal_mean(gm, f"{c}_prop", factor="Time", level=t2, fixed={"Phase": phase}),
                    })
                    r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
                    fam.append(r)
                pair_rows.append(_finalize_lmm_family(
                    fam, alpha, family=f"{mode}/{a}: Time within Phase={phase}",
                    scope="simple_effect", conditioning_factor="Phase", conditioning_level=phase,
                ))

    return (
        pd.concat(stats_rows, ignore_index=True),
        pd.concat([x for x in pair_rows if len(x)], ignore_index=True) if pair_rows else pd.DataFrame(),
    )

def future_step_inputs(overall: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical hierarchical H1-H4 input.

    The top-level AOI partition is resolved first so that
    Self + Partner + Information + Other = 100%. H1-H4 are then a subdivision
    of the already-allocated Information dwell. Therefore, within every H4
    trial, H1+H2+H3+H4 attributed dwell must equal top-level Information dwell
    (up to floating-point tolerance).
    """
    h4 = overall[overall["Horizon"].eq(4)].copy()
    rows = []
    for r in h4.itertuples(index=False):
        info_ms = float(r.information_ms)
        for h in FUTURE_STEPS:
            dwell_ms = float(getattr(r, f"h{h}_ms"))
            rows.append({
                "pid": r.pid, "trial_id": r.trial_id, "Time": int(r.Time), "Layout": r.Layout,
                "FutureStep": h, "total_fixation_ms": float(r.total_fixation_ms),
                "information_ms": info_ms, "dwell_ms": dwell_ms,
                "dwell_prop_total": float(getattr(r, f"h{h}_prop_total")),
                "relative_allocation": dwell_ms / info_ms if info_ms > 0 else np.nan,
            })
    raw = pd.DataFrame(rows)
    raw["dwell_prop_logit"] = adjusted_logit(raw["dwell_prop_total"])
    raw["relative_allocation_logit"] = adjusted_logit(raw["relative_allocation"])
    cell = raw.groupby(["pid", "Time", "FutureStep"], as_index=False).agg(
        dwell_prop_total=("dwell_prop_total", "mean"),
        relative_allocation=("relative_allocation", "mean"),
        dwell_ms=("dwell_ms", "mean"),
    )
    return raw, cell


def future_step_statistics(raw: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = fit_lmm(
        raw,
        "dwell_prop_logit ~ C(Time, Sum) * C(FutureStep, Sum) + C(Layout, Sum)",
        trial_vc=True,
        required_cols=["dwell_prop_logit", "Time", "FutureStep", "Layout", "trial_id"],
    )
    analysis = "H1-H4 proportion of total fixation (H4 trials)"
    stat = lmm_term_table(
        fit, analysis=analysis,
        target_terms=["Time", "FutureStep", "Time×FutureStep"],
    )
    families = []

    # Marginal main-effect comparisons.
    for factor, levels, marginal in [
        ("Time", TIME_LEVELS, ["FutureStep", "Layout"]),
        ("FutureStep", FUTURE_STEPS, ["Time", "Layout"]),
    ]:
        fam = []
        for l1, l2 in combinations_local(levels):
            r = lmm_emm_contrast(fit, factor=factor, level1=l1, level2=l2, marginal_over=marginal)
            r.update({
                "analysis": analysis,
                "raw_mean_level1": _raw_marginal_mean(raw, "dwell_prop_total", factor=factor, level=l1),
                "raw_mean_level2": _raw_marginal_mean(raw, "dwell_prop_total", factor=factor, level=l2),
            })
            r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
            fam.append(r)
        families.append(_finalize_lmm_family(
            fam, alpha, family=f"H1-H4: marginal {factor}", scope="marginal"
        ))

    # Time simple effects within each H1-H4 step.
    for h in FUTURE_STEPS:
        fam = []
        for t1, t2 in combinations_local(TIME_LEVELS):
            r = lmm_emm_contrast(
                fit, factor="Time", level1=t1, level2=t2,
                fixed={"FutureStep": h}, marginal_over=["Layout"],
            )
            r.update({
                "analysis": analysis,
                "raw_mean_level1": _raw_marginal_mean(raw, "dwell_prop_total", factor="Time", level=t1, fixed={"FutureStep": h}),
                "raw_mean_level2": _raw_marginal_mean(raw, "dwell_prop_total", factor="Time", level=t2, fixed={"FutureStep": h}),
            })
            r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
            fam.append(r)
        families.append(_finalize_lmm_family(
            fam, alpha, family=f"H1-H4: Time within H{h}", scope="simple_effect",
            conditioning_factor="FutureStep", conditioning_level=h,
        ))

    # FutureStep simple effects within each Time.
    for t in TIME_LEVELS:
        fam = []
        for h1, h2 in combinations_local(FUTURE_STEPS):
            r = lmm_emm_contrast(
                fit, factor="FutureStep", level1=h1, level2=h2,
                fixed={"Time": t}, marginal_over=["Layout"],
            )
            r.update({
                "analysis": analysis,
                "raw_mean_level1": _raw_marginal_mean(raw, "dwell_prop_total", factor="FutureStep", level=h1, fixed={"Time": t}),
                "raw_mean_level2": _raw_marginal_mean(raw, "dwell_prop_total", factor="FutureStep", level=h2, fixed={"Time": t}),
            })
            r["raw_mean_level2_minus_level1"] = r["raw_mean_level2"] - r["raw_mean_level1"]
            fam.append(r)
        families.append(_finalize_lmm_family(
            fam, alpha, family=f"H1-H4: FutureStep within Time={t}", scope="simple_effect",
            conditioning_factor="Time", conditioning_level=t,
        ))

    return stat, pd.concat([x for x in families if len(x)], ignore_index=True)


def future_relative_participant_input(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.groupby(["pid", "Time", "FutureStep"], as_index=False)["relative_allocation"].mean()
    d["relative_allocation_logit"] = adjusted_logit(d["relative_allocation"])
    return d


def future_relative_statistics(raw: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Relative H1-H4 allocation inside top-level Information dwell.

    Each trial's four relative shares sum to one exactly. To avoid a redundant
    trial random intercept, layouts are first averaged within participant x Time
    x FutureStep, then an adjusted-logit LMM with participant random intercept is fit.
    """
    d = future_relative_participant_input(raw)
    analysis = "Hierarchical H1-H4 relative allocation within Information dwell"
    fit = fit_lmm(
        d, "relative_allocation_logit ~ C(Time, Sum) * C(FutureStep, Sum)",
        required_cols=["relative_allocation_logit", "Time", "FutureStep"],
    )
    stat = lmm_term_table(fit, analysis=analysis, target_terms=["Time", "FutureStep", "Time×FutureStep"])
    families = []
    for factor, levels, marginal in [("Time", TIME_LEVELS, ["FutureStep"]), ("FutureStep", FUTURE_STEPS, ["Time"])]:
        fam=[]
        for l1,l2 in combinations_local(levels):
            r=lmm_emm_contrast(fit,factor=factor,level1=l1,level2=l2,marginal_over=marginal)
            r.update({
                "analysis":analysis,
                "raw_mean_level1":_raw_marginal_mean(d,"relative_allocation",factor=factor,level=l1),
                "raw_mean_level2":_raw_marginal_mean(d,"relative_allocation",factor=factor,level=l2),
            })
            r["raw_mean_level2_minus_level1"]=r["raw_mean_level2"]-r["raw_mean_level1"]
            fam.append(r)
        families.append(_finalize_lmm_family(fam,alpha,family=f"H1-H4 relative allocation: marginal {factor}",scope="marginal"))
    for h in FUTURE_STEPS:
        fam=[]
        for t1,t2 in combinations_local(TIME_LEVELS):
            r=lmm_emm_contrast(fit,factor="Time",level1=t1,level2=t2,fixed={"FutureStep":h})
            r.update({
                "analysis":analysis,
                "raw_mean_level1":_raw_marginal_mean(d,"relative_allocation",factor="Time",level=t1,fixed={"FutureStep":h}),
                "raw_mean_level2":_raw_marginal_mean(d,"relative_allocation",factor="Time",level=t2,fixed={"FutureStep":h}),
            })
            r["raw_mean_level2_minus_level1"]=r["raw_mean_level2"]-r["raw_mean_level1"]
            fam.append(r)
        families.append(_finalize_lmm_family(fam,alpha,family=f"H1-H4 relative allocation: Time within H{h}",scope="simple_effect",conditioning_factor="FutureStep",conditioning_level=h))
    for t in TIME_LEVELS:
        fam=[]
        for h1,h2 in combinations_local(FUTURE_STEPS):
            r=lmm_emm_contrast(fit,factor="FutureStep",level1=h1,level2=h2,fixed={"Time":t})
            r.update({
                "analysis":analysis,
                "raw_mean_level1":_raw_marginal_mean(d,"relative_allocation",factor="FutureStep",level=h1,fixed={"Time":t}),
                "raw_mean_level2":_raw_marginal_mean(d,"relative_allocation",factor="FutureStep",level=h2,fixed={"Time":t}),
            })
            r["raw_mean_level2_minus_level1"]=r["raw_mean_level2"]-r["raw_mean_level1"]
            fam.append(r)
        families.append(_finalize_lmm_family(fam,alpha,family=f"H1-H4 relative allocation: FutureStep within Time={t}",scope="simple_effect",conditioning_factor="Time",conditioning_level=t))
    return d, stat, pd.concat([x for x in families if len(x)], ignore_index=True)


def near_far_inputs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = raw.pivot_table(
        index=["pid","trial_id","Time","Layout","total_fixation_ms","information_ms"],
        columns="FutureStep", values="dwell_ms", aggfunc="sum", fill_value=0.0,
    ).reset_index()
    rows=[]
    for _,r in wide.iterrows():
        near_ms=float(r.get(1,0.0))+float(r.get(2,0.0))
        far_ms=float(r.get(3,0.0))+float(r.get(4,0.0))
        info_ms=float(r["information_ms"])
        for rng,ms in [("Near",near_ms),("Far",far_ms)]:
            rows.append({
                "pid":r["pid"],"trial_id":r["trial_id"],"Time":int(r["Time"]),"Layout":r["Layout"],
                "Range":rng,"total_fixation_ms":float(r["total_fixation_ms"]),"information_ms":info_ms,
                "dwell_ms":ms,"dwell_prop_total":ms/float(r["total_fixation_ms"]),
                "relative_allocation":ms/info_ms if info_ms>0 else np.nan,
            })
    d=pd.DataFrame(rows)
    d["dwell_prop_logit"]=adjusted_logit(d["dwell_prop_total"])
    d["relative_allocation_logit"]=adjusted_logit(d["relative_allocation"])
    cell=d.groupby(["pid","Time","Range"],as_index=False).agg(
        dwell_prop_total=("dwell_prop_total","mean"), relative_allocation=("relative_allocation","mean")
    )
    return d,cell


def near_far_statistics(raw: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    analysis="Hierarchical Near vs Far proportion of total fixation (H4 trials)"
    fit=fit_lmm(
        raw,"dwell_prop_logit ~ C(Time, Sum) * C(Range, Sum) + C(Layout, Sum)",trial_vc=True,
        required_cols=["dwell_prop_logit","Time","Range","Layout","trial_id"],
    )
    stat=lmm_term_table(fit,analysis=analysis,target_terms=["Time","Range","Time×Range"])
    families=[]
    for factor,levels,marginal in [("Time",TIME_LEVELS,["Range","Layout"]),("Range",["Near","Far"],["Time","Layout"])]:
        fam=[]
        for l1,l2 in combinations_local(levels):
            r=lmm_emm_contrast(fit,factor=factor,level1=l1,level2=l2,marginal_over=marginal)
            r.update({"analysis":analysis,
                "raw_mean_level1":_raw_marginal_mean(raw,"dwell_prop_total",factor=factor,level=l1),
                "raw_mean_level2":_raw_marginal_mean(raw,"dwell_prop_total",factor=factor,level=l2)})
            r["raw_mean_level2_minus_level1"]=r["raw_mean_level2"]-r["raw_mean_level1"]
            fam.append(r)
        families.append(_finalize_lmm_family(fam,alpha,family=f"Near/Far hierarchical: marginal {factor}",scope="marginal"))
    for rng in ["Near","Far"]:
        fam=[]
        for t1,t2 in combinations_local(TIME_LEVELS):
            r=lmm_emm_contrast(fit,factor="Time",level1=t1,level2=t2,fixed={"Range":rng},marginal_over=["Layout"])
            r.update({"analysis":analysis,
                "raw_mean_level1":_raw_marginal_mean(raw,"dwell_prop_total",factor="Time",level=t1,fixed={"Range":rng}),
                "raw_mean_level2":_raw_marginal_mean(raw,"dwell_prop_total",factor="Time",level=t2,fixed={"Range":rng})})
            r["raw_mean_level2_minus_level1"]=r["raw_mean_level2"]-r["raw_mean_level1"]
            fam.append(r)
        families.append(_finalize_lmm_family(fam,alpha,family=f"Near/Far hierarchical: Time within {rng}",scope="simple_effect",conditioning_factor="Range",conditioning_level=rng))
    for t in TIME_LEVELS:
        r=lmm_emm_contrast(fit,factor="Range",level1="Far",level2="Near",fixed={"Time":t},marginal_over=["Layout"])
        r.update({"analysis":analysis,
            "raw_mean_level1":_raw_marginal_mean(raw,"dwell_prop_total",factor="Range",level="Far",fixed={"Time":t}),
            "raw_mean_level2":_raw_marginal_mean(raw,"dwell_prop_total",factor="Range",level="Near",fixed={"Time":t})})
        r["raw_mean_level2_minus_level1"]=r["raw_mean_level2"]-r["raw_mean_level1"]
        families.append(_finalize_lmm_family([r],alpha,family=f"Near/Far hierarchical: Near vs Far within Time={t}",scope="simple_effect",conditioning_factor="Time",conditioning_level=t))
    return stat,pd.concat([x for x in families if len(x)],ignore_index=True)


def descriptives_from_cells(cells: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in value_cols:
        g = group_mean_ci95(cells, group_cols, col)
        g["measure"] = col
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def _errorbar_line(ax, summary: pd.DataFrame, x: str, *, scale: float = 1.0,
                   marker: str | None = "o", label: str | None = None,
                   capsize: float = 4.0, alpha: float = 1.0) -> None:
    """Plot mean ± participant-level t-based 95% CI."""
    ax.errorbar(
        summary[x], summary["mean"] * scale,
        yerr=summary["ci95"] * scale,
        marker=marker, label=label,
        capsize=capsize, elinewidth=1.0, alpha=alpha,
    )


def plot_overall(cells: pd.DataFrame, figdir: Path) -> None:
    # Composition-only stacked figure. Individual AOI uncertainty is shown in
    # overall_aoi_change_by_time.png; error bars on stacked segment heights are
    # not drawn because they are not interpretable on the cumulative stack axis.
    prop_cols = [f"{a.lower()}_prop" for a in AOIS]
    means = cells.groupby(["Time", "Horizon"], as_index=False)[prop_cols].mean()
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
    fig.tight_layout()
    fig.savefig(figdir / "overall_aoi_100pct_stacked.png", dpi=200)
    plt.close(fig)

    # Participant-level mean ± 95% CI companion plot.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, h in zip(axes, HORIZON_LEVELS):
        sub = cells[cells["Horizon"].eq(h)]
        for a in AOIS:
            g = group_mean_ci95(sub, ["Time"], f"{a.lower()}_prop")
            _errorbar_line(ax, g, "Time", scale=100.0, label=a)
        ax.set_title(f"H{h}")
        ax.set_xlabel("Available decision time (ms)")
        ax.set_xticks(TIME_LEVELS)
    axes[0].set_ylabel("Fixation dwell proportion (%)")
    axes[-1].legend(ncol=2)
    fig.suptitle("Overall AOI allocation — mean ± 95% CI")
    fig.tight_layout()
    fig.savefig(figdir / "overall_aoi_change_by_time.png", dpi=200)
    plt.close(fig)


def plot_temporal(inputs: pd.DataFrame, resolution: str, figdir: Path) -> None:
    d = inputs[inputs["Resolution"].eq(resolution)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, t in zip(axes, TIME_LEVELS):
        sub = d[d["Time"].eq(t)]
        for a in AOIS:
            g = group_mean_ci95(sub, ["BinStartMs"], f"{a.lower()}_prop")
            # Fine-resolution panels are dense: retain every CI but use thin,
            # uncapped vertical error bars to avoid hiding the temporal trace.
            fine = resolution != "100ms"
            _errorbar_line(
                ax, g, "BinStartMs", scale=100.0,
                marker="o" if not fine else None,
                label=a, capsize=0.0 if fine else 3.0,
                alpha=0.85 if fine else 1.0,
            )
        ax.set_title(f"{t} ms budget")
        ax.set_xlabel("Elapsed time from decision onset (ms)")
        if resolution != "100ms":
            ax.set_xticks(np.arange(0, t + 1, 100))
    axes[0].set_ylabel("Fixation dwell proportion (%)")
    axes[-1].legend(ncol=2)
    fig.suptitle(f"Temporal evolution of 4 AOIs — {resolution}, mean ± 95% CI")
    fig.tight_layout()
    fig.savefig(figdir / f"temporal_4aoi_{resolution.replace('.', '_')}.png", dpi=200)
    plt.close(fig)


def plot_information(cells: pd.DataFrame, figdir: Path) -> None:
    d = cells[cells["Horizon"].isin([2, 4])]
    fig, ax = plt.subplots(figsize=(7, 5))
    for h in [2, 4]:
        g = group_mean_ci95(d[d["Horizon"].eq(h)], ["Time"], "information_prop")
        _errorbar_line(ax, g, "Time", scale=100.0, label=f"H{h}")
    ax.set_xticks(TIME_LEVELS)
    ax.set_xlabel("Available decision time (ms)")
    ax.set_ylabel("Information dwell proportion (%)")
    ax.set_title("Action preview gaze — mean ± 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "information_gaze_h2_vs_h4.png", dpi=200)
    plt.close(fig)


def plot_future(cell: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in FUTURE_STEPS:
        g = group_mean_ci95(cell[cell["FutureStep"].eq(h)], ["Time"], "dwell_prop_total")
        _errorbar_line(ax, g, "Time", scale=100.0, label=f"H{h}")
    ax.set_xticks(TIME_LEVELS)
    ax.set_xlabel("Available decision time (ms)")
    ax.set_ylabel("Proportion of total fixation (%)")
    ax.set_title("H1–H4 dwell proportion of total fixation — mean ± 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "h1_h4_prop_total_fixation.png", dpi=200)
    plt.close(fig)


def plot_future_relative(cell: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in FUTURE_STEPS:
        g = group_mean_ci95(cell[cell["FutureStep"].eq(h)], ["Time"], "relative_allocation")
        _errorbar_line(ax, g, "Time", scale=100.0, label=f"H{h}")
    ax.set_xticks(TIME_LEVELS)
    ax.set_xlabel("Available decision time (ms)")
    ax.set_ylabel("Share of Information dwell (%)")
    ax.set_title("H1–H4 relative allocation within Information — mean ± 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "h1_h4_relative_allocation.png", dpi=200)
    plt.close(fig)


def plot_near_far(cell: pd.DataFrame, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for rng in ["Near", "Far"]:
        g = group_mean_ci95(cell[cell["Range"].eq(rng)], ["Time"], "dwell_prop_total")
        _errorbar_line(ax, g, "Time", scale=100.0, label=rng)
    ax.set_xticks(TIME_LEVELS)
    ax.set_xlabel("Available decision time (ms)")
    ax.set_ylabel("Proportion of total fixation (%)")
    ax.set_title("Near (H1+H2) vs Far (H3+H4) — mean ± 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "h1_h4_near_far_total_fixation.png", dpi=200)
    plt.close(fig)


def plot_response(participant: pd.DataFrame, mode: str, figdir: Path) -> None:
    d = participant[participant["Mode"].eq(mode)]

    # Keep the composition figure for the 100% partition view.
    means = d.groupby(["Time", "Phase"], as_index=False)[[f"{a.lower()}_prop" for a in AOIS]].mean()
    phases = ["Pre", "Post"]
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
    fig.tight_layout()
    fig.savefig(figdir / f"response_pre_post_100pct_{mode}.png", dpi=200)
    plt.close(fig)

    # Statistically interpretable CI view: one panel per AOI, with participant
    # means and 95% CI for PRE and POST at each Time level.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, a in zip(axes.ravel(), AOIS):
        for ph in phases:
            g = group_mean_ci95(d[d["Phase"].eq(ph)], ["Time"], f"{a.lower()}_prop")
            _errorbar_line(ax, g, "Time", scale=100.0, label=ph)
        ax.set_title(a)
        ax.set_xticks(TIME_LEVELS)
        ax.set_xlabel("Available decision time (ms)")
        ax.set_ylabel("Fixation dwell proportion (%)")
        ax.legend()
    fig.suptitle(f"Response-locked PRE/POST gaze — {mode}, mean ± 95% CI")
    fig.tight_layout()
    fig.savefig(figdir / f"response_pre_post_ci_{mode}.png", dpi=200)
    plt.close(fig)

def main() -> None:
    args = parse_args(); paths = ensure_output_tree(args.out)
    trial = load_trials(args.roots, strict_main=not args.allow_incomplete)
    steps = load_steps(args.roots, trial)

    seg = build_segment_allocations(args.roots, trial, steps)
    save_csv(seg, paths["raw"] / "gaze_segment_allocations.csv")

    overall = build_overall(seg, trial)
    save_csv(overall, paths["raw"] / "overall_trial_aoi_values.csv")
    overall_lmm = overall_lmm_inputs(overall)
    save_csv(overall_lmm, paths["inputs"] / "overall_trial_lmm_input.csv")
    overall_desc_cells = participant_overall_descriptives(overall)
    lmm_stats, post = overall_statistics(overall_lmm, args.alpha)
    save_csv(lmm_stats, paths["stats"] / "overall_aoi_lmm.csv")
    save_csv(post, paths["stats"] / "overall_aoi_lmm_posthoc_holm.csv")
    save_csv(descriptives_from_cells(overall_desc_cells, ["Time","Horizon"], [f"{a.lower()}_prop" for a in AOIS]), paths["desc"] / "overall_aoi_descriptives.csv")
    plot_overall(overall_desc_cells, paths["fig"]); plot_information(overall_desc_cells, paths["fig"])

    temporal_inputs = []
    for width, label in [(FINE_BIN_MS, "16.67ms"), (COARSE_BIN_MS, "100ms")]:
        td = build_temporal(seg, width, label)
        save_csv(td, paths["raw"] / f"temporal_{label}_trial_bin_values.csv")
        ti = temporal_participant_inputs(td); temporal_inputs.append(ti)
        save_csv(ti, paths["inputs"] / f"temporal_{label}_participant_lmm_input.csv")
        plot_temporal(ti, label, paths["fig"])
    temporal_all = pd.concat(temporal_inputs, ignore_index=True)
    within, adj = temporal_statistics(temporal_all, args.alpha)
    save_csv(within, paths["stats"] / "temporal_within_budget_lmm.csv")
    save_csv(adj, paths["stats"] / "temporal_posthoc_lmm_holm.csv")
    if len(adj):
        adjacent_only = adj[adj["comparison_scope"].eq("adjacent_bins")].copy()
        save_csv(adjacent_only, paths["stats"] / "temporal_adjacent_bin_lmm_holm.csv")
    save_csv(descriptives_from_cells(temporal_all, ["Resolution","Time","Bin","BinStartMs","BinEndMs"], [f"{a.lower()}_prop" for a in AOIS]), paths["desc"] / "temporal_aoi_descriptives.csv")

    events = build_response_events(steps, trial)
    save_csv(events, paths["raw"] / "response_event_values.csv")
    for mode in ["whole", "symmetric"]:
        rd = build_response_decision_phase(seg, events, mode, args.symmetric_ms, args.min_symmetric_ms)
        save_csv(rd, paths["raw"] / f"response_{mode}_decision_phase_values.csv")
        if rd.empty: continue
        trial_phase, rp = response_participant_inputs(rd)
        save_csv(trial_phase, paths["raw"] / f"response_{mode}_trial_phase_values.csv")
        save_csv(trial_phase, paths["inputs"] / f"response_{mode}_trial_phase_lmm_input.csv")
        ra, rpost = response_statistics(trial_phase, args.alpha)
        save_csv(ra, paths["stats"] / f"response_{mode}_lmm.csv")
        save_csv(rpost, paths["stats"] / f"response_{mode}_posthoc_lmm_holm.csv")
        # Backward-compatible alias; contents now include all response post-hoc families.
        save_csv(rpost, paths["stats"] / f"response_{mode}_prepost_lmm_holm.csv")
        save_csv(descriptives_from_cells(rp, ["Time","Phase"], [f"{a.lower()}_prop" for a in AOIS]), paths["desc"] / f"response_{mode}_descriptives.csv")
        plot_response(rp, mode, paths["fig"])

    future_raw, future_cell = future_step_inputs(overall)
    save_csv(future_raw, paths["raw"] / "h1_h4_trial_values.csv")
    save_csv(future_raw, paths["inputs"] / "h1_h4_trial_lmm_input.csv")
    fa, fp = future_step_statistics(future_raw, args.alpha)
    save_csv(fa, paths["stats"] / "h1_h4_lmm.csv")
    save_csv(fp, paths["stats"] / "h1_h4_lmm_posthoc_holm.csv")
    save_csv(descriptives_from_cells(future_cell, ["Time","FutureStep"], ["dwell_prop_total"]), paths["desc"] / "h1_h4_descriptives.csv")
    plot_future(future_cell, paths["fig"])

    relative_input, relative_stat, relative_post = future_relative_statistics(future_raw, args.alpha)
    save_csv(relative_input, paths["inputs"] / "h1_h4_relative_participant_lmm_input.csv")
    save_csv(relative_stat, paths["stats"] / "h1_h4_relative_lmm.csv")
    save_csv(relative_post, paths["stats"] / "h1_h4_relative_lmm_posthoc_holm.csv")
    save_csv(descriptives_from_cells(relative_input, ["Time","FutureStep"], ["relative_allocation"]), paths["desc"] / "h1_h4_relative_descriptives.csv")
    plot_future_relative(relative_input, paths["fig"])

    near_far_raw, near_far_cell = near_far_inputs(future_raw)
    save_csv(near_far_raw, paths["raw"] / "h1_h4_near_far_trial_values.csv")
    save_csv(near_far_raw, paths["inputs"] / "h1_h4_near_far_trial_lmm_input.csv")
    nf_stat, nf_post = near_far_statistics(near_far_raw, args.alpha)
    save_csv(nf_stat, paths["stats"] / "h1_h4_near_far_lmm.csv")
    save_csv(nf_post, paths["stats"] / "h1_h4_near_far_lmm_posthoc_holm.csv")
    save_csv(descriptives_from_cells(near_far_cell, ["Time","Range"], ["dwell_prop_total","relative_allocation"]), paths["desc"] / "h1_h4_near_far_descriptives.csv")
    plot_near_far(near_far_cell, paths["fig"])

    qc = pd.DataFrame([{
        "n_segments": len(seg), "n_trials": overall["trial_id"].nunique(),
        "max_segment_composition_error_ms": float((seg[[f"{a.lower()}_ms" for a in AOIS]].sum(axis=1)-seg["duration_ms"]).abs().max()),
        "overall_max_prop_sum_error": float((overall[[f"{a.lower()}_prop" for a in AOIS]].sum(axis=1)-1).abs().max()),
        "information_overlap_segments": int(((seg["information_match"].eq(1)) & (seg["base_aoi_logged"].isin(["Self","Partner"]))).sum()),
        "multi_future_step_overlap_segments": int(seg["matched_future_steps"].map(lambda x: len(json.loads(x)) > 1).sum()),
        "fine_bin_ms_exact": FINE_BIN_MS, "coarse_bin_ms": COARSE_BIN_MS,
    }])
    save_csv(qc, paths["qc"] / "gaze_pipeline_qc.csv")
    print(f"Gaze reanalysis complete: {paths['root']}")


if __name__ == "__main__":
    main()
