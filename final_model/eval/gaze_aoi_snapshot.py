#!/usr/bin/env python3
"""Small canonical rollout for inspecting completed GazeNet acquisitions by AOI."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.eval.common import create_output_directory, write_csv, write_json
from final_model.eval.fixation_duration import (
    FIXATION_DURATION_CONTRACT_ID,
    fixation_duration_metrics,
)
from final_model.eval.partial_rollout import PartialEvaluationConfig, evaluate_partial


def _center_key_results(events, episodes, *, weight_field=None):
    by_seed = defaultdict(list)
    for row in events:
        by_seed[(int(row["T"]), int(row["H"]), int(row["seed"]))].append(row)

    seed_rows = []
    for episode in episodes:
        time_ms = int(episode["T"])
        horizon = int(episode["H"])
        seed = int(episode["seed"])
        rows = by_seed[(time_ms, horizon, seed)]
        duration_rows = [
            {
                **row,
                "included_in_fixation_duration": row.get(
                    "included_in_fixation_duration", 1
                ),
            }
            for row in rows
        ]
        classified_rows = (
            [
                row for row in duration_rows
                if int(row["included_in_fixation_duration"]) == 1
            ]
            if weight_field is not None else rows
        )
        event_count = len(rows)
        denominator = sum(
            1.0 if weight_field is None else float(row[weight_field])
            for row in classified_rows
        )

        def weight(row):
            return 1.0 if weight_field is None else float(row[weight_field])

        def exclusive_share(aoi):
            return sum(
                weight(row)
                for row in classified_rows
                if row["exclusive_center_aoi"] == aoi
            ) / max(denominator, 1.0)

        duration_fields = (
            fixation_duration_metrics(duration_rows)
            if weight_field is not None else {}
        )
        seed_rows.append({
            "T": time_ms,
            "H": horizon,
            "episode": int(episode["episode"]),
            "seed": seed,
            "score": float(episode["sparse_team_reward"]),
            "deliveries": int(episode["deliveries"]),
            "decisions": int(episode["decisions"]),
            "collision_count": int(episode["collision_count"]),
            "collision_rate": float(episode["collision_rate"]),
            "center_event_count": event_count,
            "center_duration_ms": (
                denominator if weight_field is not None else None
            ),
            "self_center_share": exclusive_share("self"),
            "teammate_center_share": exclusive_share("partner"),
            "preview_center_share": (
                exclusive_share("preview") if horizon in (2, 4) else None
            ),
            "other_center_share": exclusive_share("other"),
            "center_overlap_share": sum(
                weight(row)
                for row in classified_rows
                if int(row["self_center"])
                + int(row["partner_center"])
                + int(row["preview_center"]) > 1
            ) / max(denominator, 1.0),
            **duration_fields,
        })

    condition_rows = []
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["T"], row["H"])].append(row)
    for (time_ms, horizon), rows in sorted(grouped.items()):
        decisions = sum(row["decisions"] for row in rows)
        condition_row = {
            "T": time_ms,
            "H": horizon,
            "seeds": len(rows),
            "score_mean": float(np.mean([row["score"] for row in rows])),
            "deliveries_mean": float(np.mean([row["deliveries"] for row in rows])),
            "collision_count": sum(row["collision_count"] for row in rows),
            "collision_rate": sum(row["collision_count"] for row in rows)
            / max(decisions, 1),
            "self_center_share_mean": float(np.mean([
                row["self_center_share"] for row in rows
            ])),
            "teammate_center_share_mean": float(np.mean([
                row["teammate_center_share"] for row in rows
            ])),
            "preview_center_share_mean": (
                float(np.mean([row["preview_center_share"] for row in rows]))
                if horizon in (2, 4) else None
            ),
            "other_center_share_mean": float(np.mean([
                row["other_center_share"] for row in rows
            ])),
            "center_overlap_share_mean": float(np.mean([
                row["center_overlap_share"] for row in rows
            ])),
        }
        if weight_field is not None:
            for aoi in ("self", "teammate", "preview", "other"):
                condition_row[f"fixation_duration_ms_{aoi}"] = sum(
                    row[f"fixation_duration_ms_{aoi}"] for row in rows
                )
                condition_row[f"fixation_prop_{aoi}_mean"] = float(np.mean([
                    row[f"fixation_prop_{aoi}"] for row in rows
                ]))
            for field in (
                "saccade_duration_ms", "classified_fixation_duration_ms",
                "accounted_gaze_duration_ms",
            ):
                condition_row[field] = sum(row[field] for row in rows)
        condition_rows.append(condition_row)
    return seed_rows, condition_rows


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--decision-times", type=int, nargs="+", default=(200, 400, 600))
    value.add_argument("--horizons", type=int, nargs="+", default=(0, 2, 4))
    value.add_argument("--episodes", type=int, default=1)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--seed-start", type=int, default=20260826)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--gaze-mode", choices=("stochastic", "argmax"), default="stochastic")
    value.add_argument(
        "--aoi-weighting",
        choices=("acquisition_count", "fixation_duration"),
        default="acquisition_count",
    )
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    output = create_output_directory(args.output_dir, prefix="gaze_aoi_snapshot")
    loaded = load_partial_checkpoint_for_evaluation(
        args.checkpoint, device=args.device
    )
    all_events = []
    all_segments = []
    all_summary = []
    all_episodes = []
    cell_results = []
    for decision_time in args.decision_times:
        for horizon in args.horizons:
            result = evaluate_partial(
                PartialEvaluationConfig(
                    checkpoint=args.checkpoint,
                    decision_time=decision_time,
                    horizon=horizon,
                    fixation_shape=loaded.metadata["fixation_shapes"][0],
                    gaze_type="learned",
                    episodes=args.episodes,
                    episode_length=args.episode_length,
                    seed_start=args.seed_start,
                    device=args.device,
                    gaze_deterministic=args.gaze_mode == "argmax",
                    action_deterministic=True,
                    aoi_diagnostic=True,
                ),
                loaded=loaded,
            )
            all_events.extend(result["aoi_acquisitions"])
            all_segments.extend(result["fixation_segments"])
            all_episodes.extend(
                {"T": decision_time, "H": horizon, **row}
                for row in result["episodes"]
            )
            all_summary.extend(
                {"T": decision_time, "H": horizon, **row}
                for row in result["aoi_summary"]
            )
            cell_results.append({
                "T": decision_time,
                "H": horizon,
                "score": result["summary"]["mean_sparse_team_reward"],
                "deliveries": result["summary"]["mean_deliveries"],
                "collision_count": result["summary"]["collision_count"],
                "collision_rate": result["summary"]["collision_rate"],
                "acquisitions": result["summary"]["aoi_acquisition_count"],
            })
            print(
                f"T{decision_time} H{horizon}: "
                f"acquisitions={result['summary']['aoi_acquisition_count']}, "
                f"deliveries={result['summary']['mean_deliveries']:.2f}",
                flush=True,
            )
    write_csv(output / "aoi_acquisitions.csv", all_events)
    write_csv(output / "fixation_segments.csv", all_segments)
    write_csv(output / "aoi_summary.csv", all_summary)
    write_csv(output / "episodes.csv", all_episodes)
    write_csv(output / "conditions.csv", cell_results)
    duration_weighted = args.aoi_weighting == "fixation_duration"
    seed_results, condition_results = _center_key_results(
        all_segments if duration_weighted else all_events,
        all_episodes,
        weight_field="duration_ms" if duration_weighted else None,
    )
    write_csv(output / "key_results_by_seed.csv", seed_results)
    write_csv(output / "key_results_by_condition.csv", condition_results)
    write_json(output / "manifest.json", {
        "checkpoint": str(loaded.path),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_step": loaded.completed_steps,
        "decision_times": args.decision_times,
        "horizons": args.horizons,
        "episodes": args.episodes,
        "episode_length": args.episode_length,
        "seed_start": args.seed_start,
        "device": args.device,
        "gaze_mode": args.gaze_mode,
        "task_action_mode": "deterministic_argmax",
        "score_metric": "sparse_team_reward",
        "collision_contract_id": "human_compatible_joint_transition_collision_v1",
        "aoi_definition": "exclusive_fixation_center_tile",
        "aoi_priority": ["preview", "self", "teammate", "other"],
        "aoi_weighting": args.aoi_weighting,
        "aoi_denominator": (
            "fixation_duration_ms_excluding_exec_saccade"
            if args.aoi_weighting == "fixation_duration"
            else "completed_acquisition_events"
        ),
        "fixation_duration_contract_id": FIXATION_DURATION_CONTRACT_ID,
        "fixation_duration_contract": (
            "simulator_phase_central_fixation_prep_source_exec_unclassified_"
            "post_landing_target_same_fixation_current_boundary_split_v1"
        ),
        "zero_classified_fixation_convention": "all_fixation_proportions_zero",
        "preview_denominator": "H2_and_H4_only",
        "action_query_contract": loaded.metadata.get("acquisition_action_contract_id"),
        "event_count": len(all_events),
        "fixation_segment_count": len(all_segments),
        "fixation_duration_ms": sum(
            float(row["duration_ms"])
            for row in all_segments
            if int(row["included_in_fixation_duration"]) == 1
        ),
        "saccade_duration_ms": sum(
            float(row["duration_ms"])
            for row in all_segments
            if int(row["included_in_fixation_duration"]) == 0
        ),
        "window_duration_ms": sum(
            float(row["duration_ms"]) for row in all_segments
        ),
    })
    print(f"output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
