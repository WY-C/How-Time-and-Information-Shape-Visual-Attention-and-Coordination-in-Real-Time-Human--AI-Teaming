"""Aggregate canonical optimizer-free Partial evaluation across 0--1M checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import torch

from final_model.common.partners import parameter_hash
from final_model.common.paths import file_sha256
from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.eval.common import FINAL_ANALYSIS_SEED_START, evaluation_seeds
from final_model.eval.partial_rollout import PartialEvaluationConfig, evaluate_partial


CHECKPOINT_NAMES = (
    "step_000000000.pt",
    "step_000200000.pt",
    "step_000400000.pt",
    "step_000600000.pt",
    "step_000800000.pt",
    "step_001000000.pt",
)
HORIZONS = (0, 2, 4)
EPISODES_PER_HORIZON = 30
EPISODE_LENGTH = 400
DECISION_TIME = 600
FIXATION_SHAPE = "plus"
GAZE_TYPE = "rule_based"


def _rate(rows: list[Mapping[str, Any]], numerator: str, denominator: str) -> float:
    top = sum(float(row.get(numerator, 0.0)) for row in rows)
    bottom = sum(float(row.get(denominator, 0.0)) for row in rows)
    return top / max(bottom, 1.0)


def summarize_horizon(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(result["episodes"])
    deliveries = [float(row["deliveries"]) for row in rows]
    sparse = [float(row["sparse_team_reward"]) for row in rows]
    acquisitions = sum(float(row.get("acquisitions", 0.0)) for row in rows)
    return {
        "episodes": len(rows),
        "deliveries_per_episode_mean": mean(deliveries),
        "deliveries_per_episode_std": pstdev(deliveries),
        "sparse_reward_per_episode_mean": mean(sparse),
        "forced_stay_rate": _rate(rows, "forced_stays", "task_decisions"),
        "actor_active_rate": _rate(rows, "executed_proposals", "task_decisions"),
        "acquisitions_per_decision": _rate(rows, "acquisitions", "task_decisions"),
        "valid_memory_fraction": mean(
            float(row["valid_memory_fraction"]) for row in rows
        ),
        "gaze_requests_per_decision": _rate(
            rows, "gaze_requests", "task_decisions"
        ),
        "same_target_acquisition_rate": (
            sum(float(row.get("same_target_acquisitions", 0.0)) for row in rows)
            / max(acquisitions, 1.0)
        ),
        "new_target_acquisition_rate": (
            sum(float(row.get("new_target_acquisitions", 0.0)) for row in rows)
            / max(acquisitions, 1.0)
        ),
        "action_distribution": None,
        "interaction_count": None,
        "episode_seeds": [int(row["seed"]) for row in rows],
        "action_trajectory_sha256": [
            row["action_trajectory_sha256"] for row in rows
        ],
        "optimizer_free": bool(result["summary"]["optimizer_free"]),
        "parameter_hash_preserved": bool(
            result["summary"]["parameter_hash_preserved"]
        ),
        "fixed_partner_identity": result["summary"]["fixed_partner_identity"],
        "fixed_partner_sha256": result["summary"]["fixed_partner_sha256"],
    }


def _evaluate_checkpoint(task: tuple[str, str, int, int]) -> dict[str, Any]:
    checkpoint_text, device, seed_start, torch_threads = task
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    checkpoint = Path(checkpoint_text)
    content_before = file_sha256(checkpoint)
    loaded = load_partial_checkpoint_for_evaluation(checkpoint, device=device)
    actor_before = loaded.actor_sha256
    critic_before = loaded.critic_sha256
    expected_seeds = list(evaluation_seeds(seed_start, EPISODES_PER_HORIZON))
    horizons = {}
    for horizon in HORIZONS:
        result = evaluate_partial(
            PartialEvaluationConfig(
                checkpoint=checkpoint,
                decision_time=DECISION_TIME,
                horizon=horizon,
                fixation_shape=FIXATION_SHAPE,
                gaze_type=GAZE_TYPE,
                episodes=EPISODES_PER_HORIZON,
                episode_length=EPISODE_LENGTH,
                seed_start=seed_start,
                device=device,
                gaze_deterministic=True,
                action_deterministic=True,
                freshness_diagnostic=False,
                output_dir=None,
            ),
            loaded=loaded,
        )
        summary = summarize_horizon(result)
        if summary["episode_seeds"] != expected_seeds:
            raise RuntimeError("held-out episode seed list changed")
        horizons[f"h{horizon}"] = summary
    loaded.assert_unchanged()
    actor_after = parameter_hash(loaded.actor)
    critic_after = parameter_hash(loaded.critic)
    content_after = file_sha256(checkpoint)
    if content_before != content_after:
        raise RuntimeError(f"evaluation modified checkpoint: {checkpoint}")
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_absolute_path": str(checkpoint.resolve()),
        "checkpoint_sha256_before": content_before,
        "checkpoint_sha256_after": content_after,
        "completed_steps": loaded.completed_steps,
        "actor_parameter_sha256_before": actor_before,
        "actor_parameter_sha256_after": actor_after,
        "critic_parameter_sha256_before": critic_before,
        "critic_parameter_sha256_after": critic_after,
        "cooking_dynamics_id": loaded.metadata["cooking_dynamics_id"],
        "memory_lifetime_ms": 2400,
        "horizons": horizons,
    }


def compact_row(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    horizons = checkpoint["horizons"]
    delivery_values = [
        horizons[f"h{horizon}"]["deliveries_per_episode_mean"]
        for horizon in HORIZONS
    ]
    return {
        "checkpoint": checkpoint["checkpoint"],
        "completed_steps": checkpoint["completed_steps"],
        "H0_deliveries": delivery_values[0],
        "H2_deliveries": delivery_values[1],
        "H4_deliveries": delivery_values[2],
        "mean_deliveries": mean(delivery_values),
        "worst_horizon": min(delivery_values),
        "horizon_spread": max(delivery_values) - min(delivery_values),
        "forced_stay_rate": mean(
            horizons[f"h{horizon}"]["forced_stay_rate"] for horizon in HORIZONS
        ),
        "actor_active_rate": mean(
            horizons[f"h{horizon}"]["actor_active_rate"] for horizon in HORIZONS
        ),
        "acquisitions_per_decision": mean(
            horizons[f"h{horizon}"]["acquisitions_per_decision"]
            for horizon in HORIZONS
        ),
        "valid_memory_fraction": mean(
            horizons[f"h{horizon}"]["valid_memory_fraction"]
            for horizon in HORIZONS
        ),
    }


def classify_progression(table: list[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    first, last = table[0], table[-1]
    delta = float(last["mean_deliveries"]) - float(first["mean_deliveries"])
    baseline = float(first["mean_deliveries"])
    percentage = None if baseline == 0.0 else 100.0 * delta / baseline
    horizon_changes = {
        label: float(last[f"{label}_deliveries"])
        - float(first[f"{label}_deliveries"])
        for label in ("H0", "H2", "H4")
    }
    collapsed = [
        label
        for label in ("H0", "H2", "H4")
        if float(last[f"{label}_deliveries"])
        < 0.75 * float(first[f"{label}_deliveries"])
        and sum(change > 0.0 for change in horizon_changes.values()) >= 2
    ]
    series = [float(row["mean_deliveries"]) for row in table]
    peak_before_last = max(series[:-1])
    scale = max(abs(baseline), 1.0)
    if collapsed:
        label = "E. one-horizon collapse"
    elif delta < -0.05 * scale or float(last["mean_deliveries"]) < 0.85 * peak_before_last:
        label = "D. unstable/regressing"
    elif abs(delta) <= 0.05 * scale:
        label = "C. flat"
    elif delta >= 0.15 * scale and float(last["mean_deliveries"]) >= 0.95 * max(series):
        label = "A. clearly improving"
    else:
        label = "B. improving but unfinished"
    return label, {
        "step0_to_step1m_absolute_change": delta,
        "step0_to_step1m_percentage_change": percentage,
        "per_horizon_absolute_change": horizon_changes,
        "collapsed_horizons": collapsed,
        "checkpoint_mean_series": series,
        "classification_rule": (
            "endpoint task-performance heuristic recorded in "
            "final_model.server_parallel.partial_progression_eval"
        ),
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Partial plus 0--1M held-out evaluation",
        "",
        (
            f"Verdict: **{report['interpretation']}**. Primary selection metric is "
            "held-out sparse deliveries per episode; this is not final checkpoint selection."
        ),
        "",
        "| checkpoint | H0_deliveries | H2_deliveries | H4_deliveries | mean_deliveries | worst_horizon | forced_stay_rate | acquisitions_per_decision | valid_memory_fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["table"]:
        lines.append(
            "| {checkpoint} | {H0_deliveries:.4f} | {H2_deliveries:.4f} | "
            "{H4_deliveries:.4f} | {mean_deliveries:.4f} | {worst_horizon:.4f} | "
            "{forced_stay_rate:.6f} | {acquisitions_per_decision:.6f} | "
            "{valid_memory_fraction:.6f} |".format(**row)
        )
    comparison = report["step0_vs_step1m"]
    percent = comparison["percentage_change"]
    percent_text = "undefined (step0 was zero)" if percent is None else f"{percent:.2f}%"
    lines.extend([
        "",
        "## Step 0 vs step 1M",
        "",
        f"- Absolute mean-deliveries change: {comparison['absolute_change']:.6f}",
        f"- Percentage mean-deliveries change: {percent_text}",
        f"- H0/H2/H4 spread at step 0: {report['table'][0]['horizon_spread']:.6f}",
        f"- H0/H2/H4 spread at step 1M: {report['table'][-1]['horizon_spread']:.6f}",
        "",
        "## Per-horizon secondary metrics",
        "",
        "| checkpoint | horizon | deliveries mean | deliveries std | sparse reward mean | forced-STAY | actor-active | acquisitions/decision | valid memory | gaze requests/decision | same target | new target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for checkpoint in report["checkpoints"]:
        for horizon in HORIZONS:
            row = checkpoint["horizons"][f"h{horizon}"]
            lines.append(
                f"| {checkpoint['checkpoint']} | H{horizon} | "
                f"{row['deliveries_per_episode_mean']:.4f} | "
                f"{row['deliveries_per_episode_std']:.4f} | "
                f"{row['sparse_reward_per_episode_mean']:.4f} | "
                f"{row['forced_stay_rate']:.6f} | {row['actor_active_rate']:.6f} | "
                f"{row['acquisitions_per_decision']:.6f} | "
                f"{row['valid_memory_fraction']:.6f} | "
                f"{row['gaze_requests_per_decision']:.6f} | "
                f"{row['same_target_acquisition_rate']:.6f} | "
                f"{row['new_target_acquisition_rate']:.6f} |"
            )
    lines.extend([
        "",
        "## Contract",
        "",
        f"- Held-out seeds: {report['held_out_seed_start']}--{report['held_out_seed_end']} (same 30 seeds for every checkpoint/horizon)",
        "- Evaluator: `final_model.eval.partial_rollout.evaluate_partial`",
        "- Optimizer/PPO updates: none",
        "- Deterministic policy evaluation: true",
        "- Primary reward: sparse team reward; shaping is not used for evaluation selection",
        "- Checkpoint and actor/critic parameter hashes: preserved",
        "- Action distribution and interaction count: unavailable from the existing canonical evaluator and therefore not invented",
        "",
    ])
    return "\n".join(lines)


def run(args) -> dict[str, Any]:
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoints = [checkpoint_dir / name for name in CHECKPOINT_NAMES]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required checkpoints: {missing}")
    if args.episodes != EPISODES_PER_HORIZON:
        raise ValueError(f"this audit requires exactly {EPISODES_PER_HORIZON} episodes")
    seeds = list(evaluation_seeds(args.seed_start, args.episodes))
    tasks = [
        (str(path), args.device, args.seed_start, args.torch_threads)
        for path in checkpoints
    ]
    rows = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {executor.submit(_evaluate_checkpoint, task): task[0] for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"evaluated {row['checkpoint']}", flush=True)
    rows.sort(key=lambda row: int(row["completed_steps"]))
    table = [compact_row(row) for row in rows]
    interpretation, basis = classify_progression(table)
    first, last = table[0], table[-1]
    absolute = float(last["mean_deliveries"]) - float(first["mean_deliveries"])
    percentage = (
        None
        if float(first["mean_deliveries"]) == 0.0
        else 100.0 * absolute / float(first["mean_deliveries"])
    )
    report = {
        "analysis": "partial_plus_0_1m_progression_sanity_check",
        "not_for_final_checkpoint_selection": True,
        "canonical_evaluator": "final_model.eval.partial_rollout.evaluate_partial",
        "evaluation_device": args.device,
        "evaluation_parallelism": {
            "axis": "independent checkpoints only",
            "workers": args.workers,
            "torch_threads_per_worker": args.torch_threads,
            "worker_completion_order_affects_results": False,
        },
        "gaze_type": GAZE_TYPE,
        "fixation_shape": FIXATION_SHAPE,
        "decision_time_ms": DECISION_TIME,
        "memory_lifetime_ms": 2400,
        "horizons": list(HORIZONS),
        "episode_length": EPISODE_LENGTH,
        "episodes_per_horizon_per_checkpoint": args.episodes,
        "total_evaluation_episodes": len(checkpoints) * len(HORIZONS) * args.episodes,
        "held_out_seed_partition": "FINAL_ANALYSIS_SEED_START",
        "held_out_seed_start": args.seed_start,
        "held_out_seed_end": seeds[-1],
        "held_out_seeds": seeds,
        "training_seed_used_for_evaluation": False,
        "deterministic": True,
        "primary_metric": "sparse deliveries per episode",
        "reward_shaping_used_for_primary_evaluation": False,
        "optimizer_free": True,
        "ppo_updates": 0,
        "checkpoint_files_preserved": all(
            row["checkpoint_sha256_before"] == row["checkpoint_sha256_after"]
            for row in rows
        ),
        "parameter_hashes_preserved": all(
            row["actor_parameter_sha256_before"] == row["actor_parameter_sha256_after"]
            and row["critic_parameter_sha256_before"] == row["critic_parameter_sha256_after"]
            for row in rows
        ),
        "checkpoints": rows,
        "table": table,
        "step0_vs_step1m": {
            "step0_mean_deliveries": first["mean_deliveries"],
            "step1m_mean_deliveries": last["mean_deliveries"],
            "absolute_change": absolute,
            "percentage_change": percentage,
        },
        "interpretation": interpretation,
        "interpretation_basis": basis,
        "unavailable_secondary_metrics": {
            "action_distribution": "not emitted by the existing canonical Partial evaluator",
            "interaction_count": "not emitted by the existing canonical Partial evaluator",
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with args.markdown_output.open("x", encoding="utf-8") as handle:
        handle.write(markdown(report))
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint-dir", type=Path, required=True)
    value.add_argument("--json-output", type=Path, required=True)
    value.add_argument("--markdown-output", type=Path, required=True)
    value.add_argument("--episodes", type=int, default=EPISODES_PER_HORIZON)
    value.add_argument("--seed-start", type=int, default=FINAL_ANALYSIS_SEED_START)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--workers", type=int, default=6)
    value.add_argument("--torch-threads", type=int, default=2)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    report = run(args)
    print(json.dumps({
        "interpretation": report["interpretation"],
        "step0_vs_step1m": report["step0_vs_step1m"],
        "total_evaluation_episodes": report["total_evaluation_episodes"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
