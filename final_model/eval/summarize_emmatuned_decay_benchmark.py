"""Package the 2026-09-06 EMMA-tuned memory-decay benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    attach_outcomes,
    load_human,
    load_model,
    new_fixations,
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
OUTPUT = RESULTS / "benchmark_20m_emmatuned_decay_comparison_v8_5seed_20260906"
MODELS = (
    {
        "model": "StagedPlusLocal20M",
        "root": RESULTS / "benchmark_20m_staged_pluslocal_v8_5seed_20260906",
        "analysis": RESULTS / "benchmark_20m_analysis_stagedplus_vs_exact3x3_20260906",
        "perception": "plus_local_sigma3",
    },
    {
        "model": "EmmaTunedDecay2400_20M",
        "root": RESULTS / "benchmark_20m_emmatuned_decay2400_noperipheral_v8_5seed_20260906",
        "analysis": RESULTS / "benchmark_20m_analysis_emmatuned_decay2400_20260906",
        "perception": "plus_no_peripheral",
    },
    {
        "model": "EmmaTunedOverDecay1200_20M",
        "root": RESULTS / "benchmark_20m_emmatuned_overdecay1200_noperipheral_v8_5seed_20260906",
        "analysis": RESULTS / "benchmark_20m_analysis_emmatuned_overdecay1200_20260906",
        "perception": "plus_no_peripheral",
    },
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    summary_rows: list[dict[str, object]] = []
    performance_frames = []
    opportunity_frames = []
    slope_frames = []
    seed_frames = []
    provenance_rows = []
    timebin_datasets = []
    configs = []

    human_decisions, human_segments = load_human(
        PROJECT / "data" / "processed", RESULTS / "human_benchmark_20260901_contract_v2"
    )
    timebin_datasets.append(
        (human_decisions, human_segments, "Human", "plus", "plus_5_tile")
    )

    human_opportunity = pd.read_csv(
        MODELS[1]["analysis"] / "opportunity_controlled_summary.csv"
    )
    human_opportunity = human_opportunity[
        human_opportunity.model.eq("Human")
        & human_opportunity.comparison.eq("plus")
    ].copy()
    opportunity_frames.append(human_opportunity)
    human_events = new_fixations(human_segments[human_segments.comparison.eq("plus")])
    human_slopes = fit_per_fixation_slopes(human_events, "Human")
    human_slopes.insert(1, "perception", "plus_5_tile")
    slope_frames.append(human_slopes)
    human_row: dict[str, object] = {
        "model": "Human",
        "perception": "plus_5_tile",
        "seeds_or_participants": int(human_decisions.cluster.nunique()),
        "episodes": pd.NA,
        "deliveries_per_episode": pd.NA,
        "sparse_reward_per_episode": pd.NA,
        "new_fixations_per_decision": len(human_events)
        / len(human_decisions[human_decisions.comparison.eq("plus")]),
    }
    for aoi in ("Self", "Teammate", "Preview", "Other"):
        opportunity = human_opportunity[human_opportunity.aoi.eq(aoi)].iloc[0]
        slope = human_slopes[human_slopes.aoi.eq(aoi)].iloc[0]
        human_row[f"{aoi}_adjusted_or"] = opportunity.adjusted_T600_vs_T200_odds_ratio
        human_row[f"{aoi}_per_fix_slope_pp_per_100ms"] = (
            slope.marginal_slope_pp_per_100ms
        )
    summary_rows.append(human_row)

    for spec in MODELS:
        root = spec["root"]
        performance = pd.read_csv(root / "summaries/performance.csv")
        perf = performance_summary(performance)
        perf.insert(0, "model", spec["model"])
        performance_frames.append(perf)

        decisions, segments = load_model(root, label=spec["model"], comparison="plus")
        events = new_fixations(segments)
        timebin_datasets.append(
            (decisions.copy(), segments, spec["model"], "plus", "plus_5_tile")
        )
        decisions["new_fixations"] = 0
        decisions = attach_outcomes(decisions, segments, events)

        slopes = fit_per_fixation_slopes(events, spec["model"])
        slopes.insert(1, "perception", spec["perception"])
        slope_frames.append(slopes)
        seeds = seed_metrics(performance, decisions, events)
        seeds.insert(0, "model", spec["model"])
        seed_frames.append(seeds)

        opportunity = pd.read_csv(spec["analysis"] / "opportunity_controlled_summary.csv")
        opportunity = opportunity[
            opportunity.model.eq(spec["model"])
            & opportunity.comparison.eq("plus")
        ].copy()
        opportunity.insert(2, "perception", spec["perception"])
        opportunity_frames.append(opportunity)

        row: dict[str, object] = {
            "model": spec["model"],
            "perception": spec["perception"],
            "seeds_or_participants": int(performance.seed.nunique()),
            "episodes": len(performance),
            "deliveries_per_episode": performance.deliveries.mean(),
            "sparse_reward_per_episode": performance.sparse_team_reward.mean(),
            "new_fixations_per_decision": len(events) / len(decisions),
        }
        for aoi in ("Self", "Teammate", "Preview", "Other"):
            aoi_row = opportunity[opportunity.aoi.eq(aoi)].iloc[0]
            slope_row = slopes[slopes.aoi.eq(aoi)].iloc[0]
            row[f"{aoi}_adjusted_or"] = aoi_row.adjusted_T600_vs_T200_odds_ratio
            row[f"{aoi}_per_fix_slope_pp_per_100ms"] = (
                slope_row.marginal_slope_pp_per_100ms
            )
        summary_rows.append(row)

        pair = json.loads((root / "benchmark/checkpoint_pair.json").read_text())
        metadata = pair["action_checkpoint"]["metadata"]
        provenance_rows.append(
            {
                "model": spec["model"],
                "checkpoint": pair["action_checkpoint"]["path"],
                "checkpoint_sha256": pair["action_checkpoint"]["sha256"],
                "step": pair["action_checkpoint"]["completed_steps"],
                "action_actor_sha256": pair["action_checkpoint"]["actor_sha256"],
                "gaze_actor_sha256": pair["gaze_checkpoint"]["gaze_sha256"],
                "source_action_sha256": metadata["action_checkpoint_sha256"],
                "partner_sha256": metadata["fixed_partner_sha256"],
                "emma_mode": metadata["emma_mode"],
                "emma_timing_contract_id": metadata["emma_timing_contract_id"],
                "peripheral_mode": metadata["peripheral_mode"],
                "memory_decay_mode": metadata["memory_decay_mode"],
                "memory_lifetime_ms": metadata["memory_lifetime_ms"],
                "training_stage": metadata["training_stage"],
                "training_contract": metadata["training_contract"],
            }
        )
        configs.append(json.loads((root / "config.json").read_text()))

    timebin_summary = pd.concat(
        [
            _coverage(
                decisions,
                segments,
                model=model,
                comparison=comparison,
                footprint=footprint,
            )
            for decisions, segments, model, comparison, footprint in timebin_datasets
        ],
        ignore_index=True,
    )
    timebin_profile = _profile(timebin_summary)

    outputs = {
        "benchmark_summary.csv": pd.DataFrame(summary_rows),
        "performance_by_condition.csv": pd.concat(performance_frames, ignore_index=True),
        "aoi_opportunity_controlled.csv": pd.concat(opportunity_frames, ignore_index=True),
        "per_fixation_slopes.csv": pd.concat(slope_frames, ignore_index=True),
        "seed_level_metrics.csv": pd.concat(seed_frames, ignore_index=True),
        "checkpoint_provenance.csv": pd.DataFrame(provenance_rows),
        "within_decision_timebin_coverage.csv": timebin_summary,
        "t600_timebin_profile.csv": timebin_profile,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT / filename, index=False)

    invariant_keys = (
        "decision_times",
        "horizons",
        "episodes",
        "episode_length",
        "seed_start",
        "evaluation_action_mode",
        "commitment_rule",
        "action_input_delay_ms",
        "tau",
        "tau_start",
        "tau_min",
        "tau_kappa_ms",
        "benchmark_contract_version",
        "benchmark_contract_sha256",
        "aoi_contract",
    )
    invariants = {key: configs[0][key] for key in invariant_keys}
    if any(
        config[key] != value
        for config in configs[1:]
        for key, value in invariants.items()
    ):
        raise RuntimeError("benchmark evaluation contracts do not match")
    manifest = {
        "purpose": "5-seed benchmark of completed EMMA-tuned decay and overdecay gaze runs",
        "rerollout_type": "actual_environment_rerollout",
        "common_evaluation_contract": invariants,
        "seeds": list(range(810000, 810005)),
        "primary_contrast": "EMMA tuned, plus, no peripheral: memory 2400 ms versus 1200 ms",
        "context_reference_note": (
            "StagedPlusLocal20M is contextual only because it uses local peripheral sigma=3."
        ),
        "inference_warning": "n=5 seeds; use direction and magnitude, not strong significance claims",
        "outputs": {name: sha256(OUTPUT / name) for name in outputs},
    }
    (OUTPUT / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
