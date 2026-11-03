"""Consolidate the staged plus+local 10M/20M 30-seed benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    attach_outcomes,
    dwell_table,
    first_fixation_table,
    fit_adjusted_models,
    load_human,
    load_model,
    new_fixations,
    per_fixation_table,
)
from final_model.eval.analyze_pairwise_t_contrasts import (
    _adjusted_contrasts,
    _aoi_contrasts,
    _cluster_directions,
    _performance_contrasts,
)
from final_model.eval.analyze_within_decision_timebins import _coverage, _profile
from final_model.eval.summarize_early_checkpoint import (
    fit_per_fixation_slopes,
    performance_summary,
    seed_metrics,
    sha256,
)


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "final_model/eval/results"
OUTPUT = RESULTS / "benchmark_staged_pluslocal_10m_20m_v8_30seed_20260906"
HUMAN_ROOT = RESULTS / "human_benchmark_20260901_contract_v2"
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


def _headline(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    adjusted: pd.DataFrame,
    performances: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    coefficient = "C(T, Treatment(reference=200))[T.600]"
    rows = []
    for model in decisions.model.unique():
        model_decisions = decisions[decisions.model.eq(model)]
        model_events = events[events.model.eq(model)]
        performance = performances.get(model)
        row = {
            "model": model,
            "clusters": model_decisions.cluster.nunique(),
            "episodes": len(performance) if performance is not None else pd.NA,
            "decisions": len(model_decisions),
            "deliveries_per_episode": (
                performance.deliveries.mean() if performance is not None else pd.NA
            ),
            "sparse_reward_per_episode": (
                performance.sparse_team_reward.mean() if performance is not None else pd.NA
            ),
            "new_fixations_per_decision": len(model_events) / len(model_decisions),
        }
        for aoi in ("Self", "Teammate", "Preview", "Other"):
            match = adjusted[
                adjusted.model.eq(model)
                & adjusted.aoi.eq(aoi)
                & adjusted.coefficient.eq(coefficient)
            ].iloc[0]
            row[f"{aoi}_adjusted_or_T600_vs_T200"] = match.odds_ratio
            row[f"{aoi}_adjusted_or_ci_low"] = match.ci_low
            row[f"{aoi}_adjusted_or_ci_high"] = match.ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def _provenance(label: str, root: Path) -> dict[str, object]:
    pair = json.loads((root / "benchmark/checkpoint_pair.json").read_text())
    action = pair["action_checkpoint"]
    gaze = pair["gaze_checkpoint"]
    metadata = action["metadata"]
    return {
        "model": label,
        "checkpoint": action["path"],
        "checkpoint_sha256": action["sha256"],
        "step": action["completed_steps"],
        "action_actor_sha256": action["actor_sha256"],
        "gaze_actor_sha256": gaze["gaze_sha256"],
        "source_action_sha256": metadata["action_checkpoint_sha256"],
        "partner_sha256": metadata["fixed_partner_sha256"],
        "emma_mode": metadata["emma_mode"],
        "peripheral_mode": metadata["peripheral_mode"],
        "peripheral_sigma_deg": metadata.get("peripheral_sigma_deg"),
        "memory_decay_mode": metadata["memory_decay_mode"],
        "memory_lifetime_ms": metadata["memory_lifetime_ms"],
        "pairing_status": pair["pairing_status"],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    human_decisions, human_segments = load_human(
        PROJECT / "data" / "processed", HUMAN_ROOT
    )
    human_decisions = human_decisions[human_decisions.comparison.eq("plus")].copy()
    human_segments = human_segments[human_segments.comparison.eq("plus")].copy()

    raw_datasets = [
        (human_decisions, human_segments, "Human", "plus", "plus_5_tile")
    ]
    decision_frames = [human_decisions]
    segment_frames = [human_segments]
    performances: dict[str, pd.DataFrame] = {}
    performance_frames = []
    provenance_rows = []
    configs = []
    for label, root in MODELS:
        decisions, segments = load_model(root, label=label, comparison="plus")
        if decisions.seed.nunique() != 30 or len(decisions) != 48600:
            raise RuntimeError(f"{label}: expected 30 seeds and 48,600 decisions")
        decision_frames.append(decisions)
        segment_frames.append(segments)
        raw_datasets.append((decisions, segments, label, "plus", "plus_5_tile"))
        performance = pd.read_csv(root / "summaries/performance.csv")
        if performance.seed.nunique() != 30 or len(performance) != 270:
            raise RuntimeError(f"{label}: expected 30 seeds and 270 episodes")
        performances[label] = performance
        summary = performance_summary(performance)
        summary.insert(0, "model", label)
        performance_frames.append(summary)
        provenance_rows.append(_provenance(label, root))
        configs.append(json.loads((root / "config.json").read_text()))

    decisions = pd.concat(decision_frames, ignore_index=True)
    segments = pd.concat(segment_frames, ignore_index=True)
    events = new_fixations(segments)
    decisions = attach_outcomes(decisions, segments, events)
    per_fixation = per_fixation_table(events)
    first_fixation = first_fixation_table(decisions, events)
    dwell = dwell_table(segments.copy(), decisions)
    adjusted = fit_adjusted_models(decisions)
    pairwise_adjusted = _adjusted_contrasts(decisions)
    cluster_rows, direction_summary = _cluster_directions(events)

    slopes = []
    for model in decisions.model.unique():
        frame = fit_per_fixation_slopes(events[events.model.eq(model)], model)
        if model == "Human":
            frame["cluster"] = "participant"
        slopes.append(frame)

    seed_frames = []
    for label, _ in MODELS:
        frame = seed_metrics(
            performances[label],
            decisions[decisions.model.eq(label)],
            events[events.model.eq(label)],
        )
        frame.insert(0, "model", label)
        seed_frames.append(frame)

    timebins = pd.concat(
        [
            _coverage(
                raw_decisions,
                raw_segments,
                model=model,
                comparison=comparison,
                footprint=footprint,
            )
            for raw_decisions, raw_segments, model, comparison, footprint in raw_datasets
        ],
        ignore_index=True,
    )

    outputs = {
        "benchmark_summary.csv": _headline(decisions, events, adjusted, performances),
        "performance_by_condition.csv": pd.concat(performance_frames, ignore_index=True),
        "per_fixation_aoi_probability.csv": per_fixation,
        "first_new_fixation_aoi.csv": first_fixation,
        "fixation_count_adjusted_models.csv": adjusted,
        "per_fixation_slopes.csv": pd.concat(slopes, ignore_index=True),
        "seed_level_metrics.csv": pd.concat(seed_frames, ignore_index=True),
        "pairwise_T_contrasts.csv": _aoi_contrasts(
            decisions, per_fixation, first_fixation, dwell, pairwise_adjusted
        ),
        "pairwise_adjusted_models.csv": pairwise_adjusted,
        "pairwise_performance_contrasts.csv": _performance_contrasts(
            decisions, performances
        ),
        "per_cluster_per_fixation_contrasts.csv": cluster_rows,
        "per_cluster_direction_summary.csv": direction_summary,
        "within_decision_timebin_coverage.csv": timebins,
        "t600_timebin_profile.csv": _profile(timebins),
        "checkpoint_provenance.csv": pd.DataFrame(provenance_rows),
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT / filename, index=False)

    invariant_keys = (
        "decision_times", "horizons", "episodes", "episode_length", "seed_start",
        "evaluation_action_mode", "commitment_rule", "action_input_delay_ms", "tau",
        "tau_start", "tau_min", "tau_kappa_ms", "benchmark_contract_version",
        "benchmark_contract_sha256", "aoi_contract",
    )
    invariants = {key: configs[0][key] for key in invariant_keys}
    for config in configs[1:]:
        mismatch = {key for key, value in invariants.items() if config[key] != value}
        if mismatch:
            raise RuntimeError(f"evaluation contract mismatch: {sorted(mismatch)}")
    manifest = {
        "purpose": "30-seed staged plus+local 10M/20M benchmark against human plus AOIs",
        "rerollout_type": "actual_environment_rerollout",
        "common_evaluation_contract": invariants,
        "model_seeds": list(range(810000, 810030)),
        "pairwise_T_contrasts_ms": [[200, 400], [400, 600], [200, 600]],
        "within_decision_bins_ms": [[0, 200], [200, 400], [400, 600]],
        "preview_subset": [2, 4],
        "aoi_note": "Plus 5-tile footprint; overlapping Self/Teammate/Preview, exclusive Other",
        "outputs": {name: sha256(OUTPUT / name) for name in outputs},
    }
    (OUTPUT / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
