"""Analyze equal [0, 36 s) exposure across T=200/400/600."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    HIT_COLUMNS,
    attach_outcomes,
    first_fixation_table,
    fit_adjusted_models,
    load_human,
    load_model,
    new_fixations,
    opportunity_summary,
    per_fixation_table,
    dwell_table,
)
from final_model.eval.summarize_early_checkpoint import sha256


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "final_model/eval/results"
OUTPUT = (
    RESULTS / "benchmark_staged_pluslocal_10m_20m_v8_30seed_20260906"
    / "first36s_absolute"
)
CUTOFF_MS = 36_000.0
SUB_BINS = ((0, 12_000), (12_000, 24_000), (24_000, 36_000))
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


def _human_sources() -> pd.DataFrame:
    frames = []
    for participant in sorted((PROJECT / "data" / "processed").iterdir()):
        source = participant / "decision_dataset.csv"
        if source.exists():
            frames.append(pd.read_csv(
                source,
                usecols=["pid", "trial_id", "step", "state_onset_ms", "reward"],
            ))
    states = pd.concat(frames, ignore_index=True)
    states["decision_uid"] = (
        states.pid.astype(str) + "|" + states.trial_id.astype(str)
        + "|" + states.step.astype(str)
    )
    states["trial_start_ms"] = states.groupby(
        ["pid", "trial_id"], observed=True
    ).state_onset_ms.transform("min")
    states["actual_onset_ms"] = states.state_onset_ms - states.trial_start_ms
    return states


def _clip_human(
    decisions: pd.DataFrame, segments: pd.DataFrame, states: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clocks = states[[
        "decision_uid", "trial_start_ms", "actual_onset_ms", "reward"
    ]]
    decisions = decisions.merge(
        clocks, on="decision_uid", how="left", validate="one_to_one"
    )
    decisions = decisions[decisions.actual_onset_ms.lt(CUTOFF_MS)].copy()
    segments = segments.merge(
        clocks[["decision_uid", "trial_start_ms"]],
        on="decision_uid", how="left", validate="many_to_one",
    )
    segments["segment_start_ms"] -= segments.trial_start_ms
    segments["fixation_start_ms"] -= segments.trial_start_ms
    segments["decision_start_ms"] -= segments.trial_start_ms
    segments["decision_end_ms"] = np.minimum(
        segments.decision_end_ms - segments.trial_start_ms, CUTOFF_MS
    )
    segment_end = segments.segment_start_ms + segments.duration_ms
    segments["duration_ms"] = (
        np.minimum(segment_end, CUTOFF_MS) - np.maximum(segments.segment_start_ms, 0)
    ).clip(lower=0)
    segments["segment_start_ms"] = segments.segment_start_ms.clip(lower=0)
    segments = segments[
        segments.decision_uid.isin(decisions.decision_uid)
        & segments.duration_ms.gt(0)
        & segments.segment_start_ms.lt(CUTOFF_MS)
    ].copy()
    return decisions, segments


def _clip_model(
    decisions: pd.DataFrame, segments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = decisions[decisions.decision_onset_ms.lt(CUTOFF_MS)].copy()
    decisions["actual_onset_ms"] = decisions.decision_onset_ms
    segments["decision_end_ms"] = np.minimum(
        segments.decision_end_ms, CUTOFF_MS
    )
    segment_end = segments.segment_start_ms + segments.duration_ms
    segments["duration_ms"] = (
        np.minimum(segment_end, CUTOFF_MS) - np.maximum(segments.segment_start_ms, 0)
    ).clip(lower=0)
    segments["segment_start_ms"] = segments.segment_start_ms.clip(lower=0)
    segments = segments[
        segments.decision_uid.isin(decisions.decision_uid)
        & segments.duration_ms.gt(0)
        & segments.segment_start_ms.lt(CUTOFF_MS)
    ].copy()
    return decisions, segments


def _performance(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, t, h), frame in decisions.groupby(
        ["model", "T", "H"], observed=True
    ):
        episode_column = "episode_unit"
        units = frame.groupby(episode_column, observed=True).agg(
            decisions=("decision_uid", "size"),
            sparse_reward=("reward" if model == "Human" else "sparse_team_reward", "sum"),
        )
        units["deliveries"] = units.sparse_reward / 20.0
        rows.append({
            "model": model,
            "T": int(t),
            "H": int(h),
            "episodes_or_trials": len(units),
            "mean_decisions_in_first36s": units.decisions.mean(),
            "deliveries_per_first36s": units.deliveries.mean(),
            "sparse_reward_per_first36s": units.sparse_reward.mean(),
        })
    return pd.DataFrame(rows)


def _sub_bins(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start, end in SUB_BINS:
        frame = decisions[
            decisions.actual_onset_ms.ge(start) & decisions.actual_onset_ms.lt(end)
        ]
        for (model, t), subset in frame.groupby(["model", "T"], observed=True):
            cluster_means = subset.groupby("cluster", observed=True)[
                [*HIT_COLUMNS.values(), "n_new_fixations"]
            ].mean()
            row = {
                "model": model,
                "T": int(t),
                "bin_start_ms": start,
                "bin_end_ms": end,
                "decisions": len(subset),
                "clusters": subset.cluster.nunique(),
                "new_fixations_per_decision": cluster_means.n_new_fixations.mean(),
            }
            for aoi, outcome in HIT_COLUMNS.items():
                eligible = subset[subset.H.isin([2, 4])] if aoi == "Preview" else subset
                row[f"{aoi}_decision_coverage"] = (
                    eligible.groupby("cluster", observed=True)[outcome].mean().mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    human_decisions, human_segments = load_human(
        PROJECT / "data" / "processed", RESULTS / "human_benchmark_20260901_contract_v2"
    )
    human_decisions = human_decisions[human_decisions.comparison.eq("plus")].copy()
    human_segments = human_segments[human_segments.comparison.eq("plus")].copy()
    human_decisions, human_segments = _clip_human(
        human_decisions, human_segments, _human_sources()
    )
    decision_frames = [human_decisions]
    segment_frames = [human_segments]
    for label, root in MODELS:
        decisions, segments = load_model(root, label=label, comparison="plus")
        outcomes = pd.read_csv(
            root / "raw/decision_log.csv",
            usecols=[
                "T", "H", "seed", "episode", "decision",
                "deliveries", "sparse_team_reward",
            ],
        )
        decisions = decisions.merge(
            outcomes,
            on=["T", "H", "seed", "episode", "decision"],
            how="left",
            validate="one_to_one",
        )
        decisions, segments = _clip_model(decisions, segments)
        decision_frames.append(decisions)
        segment_frames.append(segments)

    decisions = pd.concat(decision_frames, ignore_index=True)
    segments = pd.concat(segment_frames, ignore_index=True)
    events = new_fixations(segments)
    human_counts = (
        events[events.model.eq("Human")]
        .groupby("decision_uid", observed=True)
        .size()
    )
    human_mask = decisions.model.eq("Human")
    decisions.loc[human_mask, "new_fixations"] = (
        decisions.loc[human_mask, "decision_uid"]
        .map(human_counts)
        .fillna(0)
        .astype(int)
    )
    decisions = attach_outcomes(decisions, segments, events)
    per_fixation = per_fixation_table(events)
    first = first_fixation_table(decisions, events)
    adjusted = fit_adjusted_models(decisions)
    dwell = dwell_table(segments.copy(), decisions)
    summary = opportunity_summary(per_fixation, first, adjusted, dwell, decisions)
    outputs = {
        "performance_first36s.csv": _performance(decisions),
        "per_fixation_aoi_probability_first36s.csv": per_fixation,
        "first_new_fixation_aoi_first36s.csv": first,
        "fixation_count_adjusted_models_first36s.csv": adjusted,
        "opportunity_controlled_summary_first36s.csv": summary,
        "absolute_12s_subbins.csv": _sub_bins(decisions),
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / name, index=False)
    manifest = {
        "window": "[0, 36000) ms from actual trial/episode onset",
        "boundary_handling": "segments crossing 36 s are clipped exactly at the boundary",
        "model_decisions_per_episode": {"T200": 180, "T400": 90, "T600": 60},
        "human_clock": "state_onset_ms minus first state_onset_ms in each trial",
        "aoi_contract": "plus 5-tile; overlapping Self/Teammate/Preview and exclusive Other",
        "adjusted_model": "decision coverage ~ categorical T + categorical H + n_new_fixations; GEE clustered by participant/seed",
        "outputs": {name: sha256(OUTPUT / name) for name in outputs},
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
