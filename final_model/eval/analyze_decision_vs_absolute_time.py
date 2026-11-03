"""Compare within-decision and episode-absolute AOI allocation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    HIT_COLUMNS,
    attach_outcomes,
    load_human,
    load_model,
    new_fixations,
)
from final_model.eval.summarize_early_checkpoint import sha256


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "final_model/eval/results"
SOURCE_SUMMARY = RESULTS / "benchmark_staged_pluslocal_10m_20m_v8_30seed_20260906"
OUTPUT = SOURCE_SUMMARY / "decision_vs_absolute_time_v2"
ABSOLUTE_BINS_MS = tuple(range(0, 120_001, 20_000))
MODELS = (
    (
        "StagedPlusLocal10M_30seed",
        RESULTS / "humanfix_plus_local_gaze10m_v8_exponential_30ep_20260905",
    ),
    (
        "StagedPlusLocal20M_30seed",
        RESULTS / "benchmark_20m_staged_pluslocal_v8_30seed_20260906",
    ),
)


def _human_elapsed(decisions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for participant in sorted((PROJECT / "data" / "processed").iterdir()):
        source = participant / "decision_dataset.csv"
        if not source.exists():
            continue
        frames.append(pd.read_csv(
            source,
            usecols=["pid", "trial_id", "step", "state_onset_ms"],
        ))
    states = pd.concat(frames, ignore_index=True)
    states["decision_uid"] = (
        states.pid.astype(str) + "|" + states.trial_id.astype(str)
        + "|" + states.step.astype(str)
    )
    states["actual_elapsed_ms"] = (
        states.state_onset_ms
        - states.groupby(["pid", "trial_id"], observed=True).state_onset_ms.transform("min")
    )
    result = decisions.merge(
        states[["decision_uid", "actual_elapsed_ms"]],
        on="decision_uid",
        how="left",
        validate="one_to_one",
    )
    if result.actual_elapsed_ms.isna().any():
        raise RuntimeError("human elapsed-time mapping is incomplete")
    result["nominal_decision_clock_ms"] = result["step"] * result["T"]
    return result


def _model_elapsed(decisions: pd.DataFrame) -> pd.DataFrame:
    result = decisions.copy()
    result["actual_elapsed_ms"] = result.decision_onset_ms
    result["nominal_decision_clock_ms"] = result["decision"] * result["T"]
    if not np.allclose(
        result.actual_elapsed_ms, result.nominal_decision_clock_ms, atol=1e-8
    ):
        raise RuntimeError("model absolute and nominal clocks differ")
    return result


def _absolute_bins(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for basis, column in (
        ("actual_trial_elapsed", "actual_elapsed_ms"),
        ("nominal_decision_clock", "nominal_decision_clock_ms"),
    ):
        frame = decisions.copy()
        frame["bin_start_ms"] = pd.cut(
            frame[column],
            bins=ABSOLUTE_BINS_MS,
            right=False,
            labels=ABSOLUTE_BINS_MS[:-1],
        ).astype(float)
        frame = frame.dropna(subset=["bin_start_ms"])
        frame["bin_start_ms"] = frame.bin_start_ms.astype(int)
        for (model, t, h, start), subset in frame.groupby(
            ["model", "T", "H", "bin_start_ms"], observed=True
        ):
            cluster_means = subset.groupby("cluster", observed=True)[
                [*HIT_COLUMNS.values(), "n_new_fixations"]
            ].mean()
            row = {
                "time_basis": basis,
                "model": model,
                "T": int(t),
                "H": int(h),
                "bin_start_ms": int(start),
                "bin_end_ms": int(start + 20_000),
                "decisions": len(subset),
                "clusters": subset.cluster.nunique(),
                "new_fixations_per_decision": cluster_means.n_new_fixations.mean(),
            }
            for aoi, outcome in HIT_COLUMNS.items():
                row[f"{aoi}_decision_coverage"] = cluster_means[outcome].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def _marginalize_h(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (basis, model, t, start), frame in table.groupby(
        ["time_basis", "model", "T", "bin_start_ms"], observed=True
    ):
        for aoi in HIT_COLUMNS:
            eligible = frame[frame.H.isin([2, 4])] if aoi == "Preview" else frame
            weights = eligible.decisions.astype(float)
            rows.append({
                "time_basis": basis,
                "model": model,
                "T": int(t),
                "aoi": aoi,
                "h_subset": "H2_H4" if aoi == "Preview" else "all_H",
                "bin_start_ms": int(start),
                "bin_end_ms": int(start + 20_000),
                "decisions": int(eligible.decisions.sum()),
                "clusters": int(eligible.clusters.max()),
                "decision_coverage": np.average(
                    eligible[f"{aoi}_decision_coverage"], weights=weights
                ),
                "new_fixations_per_decision": np.average(
                    eligible.new_fixations_per_decision, weights=weights
                ),
            })
    return pd.DataFrame(rows)


def _absolute_event_bins(
    events: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    clocks = decisions[[
        "model", "decision_uid", "actual_elapsed_ms", "nominal_decision_clock_ms"
    ]]
    frame = events.merge(
        clocks, on=["model", "decision_uid"], how="left", validate="many_to_one"
    )
    relative_fixation_ms = frame.fixation_start_ms - frame.decision_start_ms
    frame["actual_fixation_elapsed_ms"] = frame.actual_elapsed_ms + relative_fixation_ms
    frame["nominal_fixation_clock_ms"] = (
        frame.nominal_decision_clock_ms + relative_fixation_ms
    )
    rows = []
    for basis, column in (
        ("actual_trial_elapsed", "actual_fixation_elapsed_ms"),
        ("nominal_decision_clock", "nominal_fixation_clock_ms"),
    ):
        basis_frame = frame.copy()
        basis_frame["bin_start_ms"] = pd.cut(
            basis_frame[column],
            bins=ABSOLUTE_BINS_MS,
            right=False,
            labels=ABSOLUTE_BINS_MS[:-1],
        ).astype(float)
        basis_frame = basis_frame.dropna(subset=["bin_start_ms"])
        basis_frame["bin_start_ms"] = basis_frame.bin_start_ms.astype(int)
        for (model, t, h, start), subset in basis_frame.groupby(
            ["model", "T", "H", "bin_start_ms"], observed=True
        ):
            row = {
                "time_basis": basis,
                "model": model,
                "T": int(t),
                "H": int(h),
                "bin_start_ms": int(start),
                "bin_end_ms": int(start + 20_000),
                "new_fixations": len(subset),
                "clusters": subset.cluster.nunique(),
            }
            for aoi, outcome in HIT_COLUMNS.items():
                row[f"{aoi}_per_fixation"] = subset[outcome].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    human_decisions, human_segments = load_human(
        PROJECT / "data" / "processed", RESULTS / "human_benchmark_20260901_contract_v2"
    )
    human_decisions = human_decisions[human_decisions.comparison.eq("plus")].copy()
    human_segments = human_segments[human_segments.comparison.eq("plus")].copy()
    decision_frames = [_human_elapsed(human_decisions)]
    segment_frames = [human_segments]
    for label, root in MODELS:
        decisions, segments = load_model(root, label=label, comparison="plus")
        decision_frames.append(_model_elapsed(decisions))
        segment_frames.append(segments)

    decisions = pd.concat(decision_frames, ignore_index=True)
    segments = pd.concat(segment_frames, ignore_index=True)
    events = new_fixations(segments)
    decisions = attach_outcomes(decisions, segments, events)
    cells = _absolute_bins(decisions)
    marginal = _marginalize_h(cells)
    event_bins = _absolute_event_bins(events, decisions)
    within_decision = pd.read_csv(
        SOURCE_SUMMARY / "within_decision_timebin_coverage.csv"
    )
    profile = pd.read_csv(SOURCE_SUMMARY / "t600_timebin_profile.csv")

    outputs = {
        "absolute_timebin_by_T_H.csv": cells,
        "absolute_timebin_H_marginal.csv": marginal,
        "absolute_timebin_per_fixation.csv": event_bins,
        "within_decision_timebin_coverage.csv": within_decision,
        "t600_within_decision_profile.csv": profile,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / name, index=False)
    manifest = {
        "within_decision_definition": "AOI segment overlaps 0-200, 200-400, or 400-600 ms after decision onset",
        "absolute_actual_definition": "decision AOI coverage assigned by elapsed milliseconds from the first state onset in that trial/episode",
        "absolute_nominal_definition": "decision AOI coverage assigned by decision index times T; excludes human response/transition overhead",
        "absolute_bins_ms": list(ABSOLUTE_BINS_MS),
        "human_actual_time_warning": "Actual human trial time includes response and transition overhead and is not mechanically identical to model virtual time.",
        "outputs": {name: sha256(OUTPUT / name) for name in outputs},
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
