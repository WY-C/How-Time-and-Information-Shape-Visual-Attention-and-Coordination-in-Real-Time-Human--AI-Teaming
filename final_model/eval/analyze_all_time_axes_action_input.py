"""Joint full/absolute/decision-relative/action-input timing analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    HIT_COLUMNS,
    first_fixation_table,
    fit_adjusted_models,
    load_human,
    load_model,
    new_fixations,
    per_fixation_table,
)
from final_model.eval.summarize_early_checkpoint import sha256


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "final_model/eval/results"
SUMMARY_ROOT = RESULTS / "benchmark_staged_pluslocal_10m_20m_v8_30seed_20260906"
OUTPUT = SUMMARY_ROOT / "all_time_axes_action_input"
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


def _human_timing(decisions: pd.DataFrame) -> pd.DataFrame:
    pids = set(decisions.pid.unique())
    frames = []
    for pid in sorted(pids):
        frames.append(pd.read_csv(
            PROJECT / "data" / "processed" / pid / "decision_dataset.csv",
            usecols=[
                "pid", "trial_id", "step", "state_onset_ms",
                "human_input_ms", "action_execution_ms",
            ],
        ))
    timing = pd.concat(frames, ignore_index=True)
    timing["decision_uid"] = (
        timing.pid.astype(str) + "|" + timing.trial_id.astype(str)
        + "|" + timing.step.astype(str)
    )
    timing["selected_input_latency_ms"] = (
        timing.human_input_ms - timing.state_onset_ms
    )
    timing["selected_input_abs_ms"] = timing.human_input_ms
    result = decisions.merge(
        timing[[
            "decision_uid", "selected_input_latency_ms", "selected_input_abs_ms"
        ]],
        on="decision_uid", how="left", validate="one_to_one",
    )
    result["input_observed"] = result.selected_input_abs_ms.notna().astype(np.int8)
    return result


def _model_timing(decisions: pd.DataFrame, root: Path) -> pd.DataFrame:
    timing = pd.read_csv(
        root / "raw/decision_log.csv",
        usecols=[
            "T", "H", "seed", "episode", "decision", "forced_stay",
            "selected_action_input_time_ms", "selected_action_input_source",
        ],
    ).rename(columns={
        "selected_action_input_time_ms": "selected_input_latency_ms",
        "selected_action_input_source": "selected_input_source",
    })
    result = decisions.merge(
        timing,
        on=["T", "H", "seed", "episode", "decision"],
        how="left", validate="one_to_one",
    )
    result["selected_input_abs_ms"] = (
        result.decision_onset_ms + result.selected_input_latency_ms
    )
    result["input_observed"] = result.selected_input_abs_ms.notna().astype(np.int8)
    mismatch = result.input_observed.ne(result.forced_stay.eq(0))
    if mismatch.any():
        raise RuntimeError("model selected-input/forced-STAY mismatch")
    return result


def _input_summary(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligibility = []
    latency = []
    for (model, t, h), frame in decisions.groupby(
        ["model", "T", "H"], observed=True
    ):
        cluster_rates = frame.groupby("cluster", observed=True).input_observed.mean()
        observed = frame[frame.input_observed.eq(1)]
        cluster_latency = observed.groupby(
            "cluster", observed=True
        ).selected_input_latency_ms.mean()
        eligibility.append({
            "model": model, "T": int(t), "H": int(h),
            "decisions": len(frame), "clusters": frame.cluster.nunique(),
            "input_observed_rate": cluster_rates.mean(),
            "timeout_rate": 1.0 - cluster_rates.mean(),
        })
        latency.append({
            "model": model, "T": int(t), "H": int(h),
            "observed_inputs": len(observed), "clusters": observed.cluster.nunique(),
            "selected_input_latency_ms": cluster_latency.mean(),
            "selected_input_latency_median_ms": observed.selected_input_latency_ms.median(),
        })
    return pd.DataFrame(eligibility), pd.DataFrame(latency)


def _phase_data(
    decisions: pd.DataFrame, segments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = decisions[decisions.input_observed.eq(1)].copy()
    clocks = eligible[["comparison", "model", "decision_uid", "selected_input_abs_ms"]]
    segments = segments.merge(
        clocks,
        on=["comparison", "model", "decision_uid"],
        how="inner", validate="many_to_one",
    )
    segment_end = segments.segment_start_ms + segments.duration_ms
    events = new_fixations(segments)
    events["phase"] = np.where(
        events.fixation_start_ms.lt(events.selected_input_abs_ms), "pre", "post"
    )

    decision_frames = []
    event_frames = []
    per_fixation_frames = []
    first_frames = []
    adjusted_frames = []
    keys = ["comparison", "model", "decision_uid"]
    for phase in ("pre", "post"):
        if phase == "pre":
            phase_segments = segments[segments.segment_start_ms.lt(
                segments.selected_input_abs_ms
            )].copy()
        else:
            phase_segments = segments[segment_end.gt(
                segments.selected_input_abs_ms
            )].copy()
        phase_events = events[events.phase.eq(phase)].copy()
        counts = phase_events.groupby(keys, observed=True).size().rename(
            "n_new_fixations"
        )
        coverage = phase_segments.groupby(keys, observed=True)[
            list(HIT_COLUMNS.values())
        ].max()
        phase_decisions = eligible.merge(counts, on=keys, how="left").merge(
            coverage, on=keys, how="left"
        )
        phase_decisions["n_new_fixations"] = (
            phase_decisions.n_new_fixations.fillna(0).astype(int)
        )
        for column in HIT_COLUMNS.values():
            phase_decisions[column] = phase_decisions[column].fillna(0).astype(np.int8)
        phase_decisions["phase"] = phase
        phase_events["phase"] = phase
        decision_frames.append(phase_decisions)
        event_frames.append(phase_events)

        per_fixation = per_fixation_table(phase_events)
        per_fixation.insert(3, "phase", phase)
        per_fixation_frames.append(per_fixation)
        first = first_fixation_table(phase_decisions, phase_events)
        first.insert(3, "phase", phase)
        first_frames.append(first)
        adjusted = fit_adjusted_models(phase_decisions)
        adjusted.insert(3, "phase", phase)
        adjusted_frames.append(adjusted)
    return (
        pd.concat(decision_frames, ignore_index=True),
        pd.concat(event_frames, ignore_index=True),
        pd.concat(per_fixation_frames, ignore_index=True),
        pd.concat(first_frames, ignore_index=True),
        pd.concat(adjusted_frames, ignore_index=True),
    )


def _headline(adjusted: pd.DataFrame) -> pd.DataFrame:
    coefficient = "C(T, Treatment(reference=200))[T.600]"
    selected = adjusted[adjusted.coefficient.eq(coefficient)]
    return selected.pivot_table(
        index=["model", "phase"], columns="aoi", values="odds_ratio"
    ).reset_index()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    human_decisions, human_segments = load_human(
        PROJECT / "data" / "processed", RESULTS / "human_benchmark_20260901_contract_v2"
    )
    human_decisions = _human_timing(
        human_decisions[human_decisions.comparison.eq("plus")].copy()
    )
    human_segments = human_segments[human_segments.comparison.eq("plus")].copy()
    decision_frames = [human_decisions]
    segment_frames = [human_segments]
    for label, root in MODELS:
        decisions, segments = load_model(root, label=label, comparison="plus")
        decision_frames.append(_model_timing(decisions, root))
        segment_frames.append(segments)
    decisions = pd.concat(decision_frames, ignore_index=True)
    segments = pd.concat(segment_frames, ignore_index=True)
    eligibility, latency = _input_summary(decisions)
    phase = _phase_data(decisions, segments)
    phase_decisions, phase_events, per_fixation, first, adjusted = phase

    full_summary = pd.read_csv(SUMMARY_ROOT / "benchmark_summary.csv")
    first36 = pd.read_csv(
        SUMMARY_ROOT / "first36s_absolute/opportunity_controlled_summary_first36s.csv"
    )
    within_decision = pd.read_csv(SUMMARY_ROOT / "t600_timebin_profile.csv")
    absolute = pd.read_csv(
        SUMMARY_ROOT / "decision_vs_absolute_time_v2/absolute_timebin_H_marginal.csv"
    )
    outputs = {
        "action_input_eligibility.csv": eligibility,
        "action_input_latency.csv": latency,
        "action_phase_decisions.csv": phase_decisions,
        "action_phase_new_fixations.csv": phase_events,
        "action_phase_per_fixation_aoi.csv": per_fixation,
        "action_phase_first_new_fixation_aoi.csv": first,
        "action_phase_adjusted_models.csv": adjusted,
        "action_phase_headline.csv": _headline(adjusted),
        "full_window_headline.csv": full_summary,
        "first36s_headline.csv": first36,
        "within_decision_timebins.csv": within_decision,
        "absolute_episode_timebins.csv": absolute,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / name, index=False)
    manifest = {
        "source": "saved actual-environment contract-v8 rerollouts; no query replay",
        "action_input_contract": "first arrival at which the eventual final selected action became current",
        "model_input_delay_ms": 99.95279386919466,
        "input_observed_scope": "pre/post phase analysis includes non-timeout decisions only; eligibility is reported separately",
        "phase_event_rule": "new fixation onset before selected input is pre; otherwise post",
        "phase_coverage_rule": "a fixation segment overlapping a phase contributes decision-level AOI coverage to that phase",
        "combined_axes": ["full decision", "decision-relative", "episode-absolute", "first 36 seconds", "selected-action input phase"],
        "outputs": {name: sha256(OUTPUT / name) for name in outputs},
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
