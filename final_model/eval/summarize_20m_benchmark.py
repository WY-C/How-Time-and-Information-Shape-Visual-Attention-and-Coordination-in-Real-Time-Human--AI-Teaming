"""Consolidate the 2026-09-06 three-model 20M benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    attach_outcomes,
    load_model,
    new_fixations,
)
from final_model.eval.summarize_early_checkpoint import (
    fit_per_fixation_slopes,
    performance_summary,
    seed_metrics,
    sha256,
)


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "final_model/eval/results"
OUTPUT = RESULTS / "benchmark_20m_threeway_v8_5seed_20260906"
MODELS = (
    {
        "model": "DirectJoint20M",
        "curriculum": "direct_joint_from_full",
        "perception": "plus_local_sigma3",
        "comparison": "plus",
        "root": RESULTS / "benchmark_20m_directjoint_pluslocal_v8_5seed_20260906",
        "analysis": RESULTS / "benchmark_20m_analysis_direct_vs_exact3x3_20260906",
    },
    {
        "model": "StagedPlusLocal20M",
        "curriculum": "staged_action_then_gaze",
        "perception": "plus_local_sigma3",
        "comparison": "plus",
        "root": RESULTS / "benchmark_20m_staged_pluslocal_v8_5seed_20260906",
        "analysis": RESULTS / "benchmark_20m_analysis_stagedplus_vs_exact3x3_20260906",
    },
    {
        "model": "StagedExact3x3_20M",
        "curriculum": "staged_action_then_gaze",
        "perception": "exact3x3_no_peripheral",
        "comparison": "exact3x3",
        "root": RESULTS / "benchmark_20m_staged_exact3x3_v8_5seed_20260906",
        "analysis": RESULTS / "benchmark_20m_analysis_direct_vs_exact3x3_20260906",
    },
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    benchmark_rows = []
    performance_frames = []
    opportunity_frames = []
    slope_frames = []
    seed_frames = []
    provenance_rows = []
    configs = []

    for spec in MODELS:
        root = spec["root"]
        performance = pd.read_csv(root / "summaries/performance.csv")
        perf = performance_summary(performance)
        perf.insert(0, "model", spec["model"])
        performance_frames.append(perf)

        decisions, segments = load_model(
            root, label=spec["model"], comparison=spec["comparison"]
        )
        events = new_fixations(segments)
        decisions["new_fixations"] = 0
        decisions = attach_outcomes(decisions, segments, events)

        slopes = fit_per_fixation_slopes(events, spec["model"])
        slopes.insert(1, "perception", spec["perception"])
        slope_frames.append(slopes)
        seeds = seed_metrics(performance, decisions, events)
        seeds.insert(0, "model", spec["model"])
        seed_frames.append(seeds)

        opportunity = pd.read_csv(
            spec["analysis"] / "opportunity_controlled_summary.csv"
        )
        opportunity = opportunity[
            opportunity.model.eq(spec["model"])
            & opportunity.comparison.eq(spec["comparison"])
        ].copy()
        opportunity.insert(2, "perception", spec["perception"])
        opportunity_frames.append(opportunity)

        row = {
            "model": spec["model"],
            "curriculum": spec["curriculum"],
            "perception": spec["perception"],
            "seeds": performance.seed.nunique(),
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
            row[f"{aoi}_first_fix_delta_T600_minus_T200"] = (
                aoi_row.first_fixation_delta
            )
        benchmark_rows.append(row)

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
                "fixation_shapes": json.dumps(metadata["fixation_shapes"]),
                "peripheral_mode": metadata["peripheral_mode"],
                "peripheral_sigma_deg": metadata.get("peripheral_sigma_deg"),
                "memory_decay_mode": metadata["memory_decay_mode"],
                "memory_lifetime_ms": metadata["memory_lifetime_ms"],
                "training_stage": metadata["training_stage"],
                "training_contract": metadata["training_contract"],
            }
        )
        configs.append(json.loads((root / "config.json").read_text()))

    outputs = {
        "benchmark_summary.csv": pd.DataFrame(benchmark_rows),
        "performance_by_condition.csv": pd.concat(performance_frames, ignore_index=True),
        "aoi_opportunity_controlled.csv": pd.concat(opportunity_frames, ignore_index=True),
        "per_fixation_slopes.csv": pd.concat(slope_frames, ignore_index=True),
        "seed_level_metrics.csv": pd.concat(seed_frames, ignore_index=True),
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
    if any(any(config[key] != value for key, value in invariants.items()) for config in configs[1:]):
        raise RuntimeError("benchmark evaluation contracts do not match")
    manifest = {
        "purpose": "5-seed exploratory benchmark of three ready 20M checkpoints",
        "rerollout_type": "actual_environment_rerollout",
        "common_evaluation_contract": invariants,
        "seeds": list(range(810000, 810005)),
        "native_footprint_note": (
            "DirectJoint20M and StagedPlusLocal20M AOIs use a plus 5-tile footprint; "
            "StagedExact3x3_20M AOIs use its native exact 3x3 9-tile footprint."
        ),
        "inference_warning": "n=5 seeds; use direction and magnitude, not strong significance claims",
        "outputs": {name: sha256(OUTPUT / name) for name in outputs},
    }
    (OUTPUT / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
