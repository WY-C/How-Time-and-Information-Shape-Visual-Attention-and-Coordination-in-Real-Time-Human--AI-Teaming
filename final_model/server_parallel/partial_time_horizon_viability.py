"""Read-only Time x Horizon viability evaluation for a frozen Partial ActionNet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from final_model.common.overcooked_backend import DirectOvercookedBackend, cooking_contract
from final_model.common.partners import FixedHSPPartnerRepository, parameter_hash
from final_model.common.paths import file_sha256
from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.eval.common import (
    CANONICAL_PARTNER,
    FINAL_ANALYSIS_SEED_START,
    evaluation_seeds,
)
from final_model.partial_ver.gaze import GazePhase
from final_model.partial_ver.memory import MEMORY_LIFETIME_MS
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.rule_based import DEFAULT_RULE_GAZE_PROBABILITIES


DECISION_TIMES_MS = (200, 400, 600)
HORIZONS = (0, 2, 4)
CONDITIONS = tuple(
    (decision_time, horizon)
    for decision_time in DECISION_TIMES_MS
    for horizon in HORIZONS
)
FIXATION_SHAPE = "plus"
GAZE_TYPE = "rule_based"
EPISODES_PER_CONDITION = 30
EPISODE_LENGTH = 400
RULE_GAZE_PROBABILITIES = {
    "task": 0.60,
    "stay": 0.20,
    "stale": 0.15,
    "random": 0.05,
}
STAY_ACTION = 4


def action_decision_record(
    collected: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    decision_index: int,
    decision_time_ms: int,
) -> dict[str, Any]:
    """Describe when the ultimately executed action was decided."""
    budget = int(decision_time_ms)
    if budget not in DECISION_TIMES_MS:
        raise ValueError(f"decision_time_ms must be one of {DECISION_TIMES_MS}")
    deadline = float(collected["deadline_ms"])
    decision_start = deadline - float(budget)
    latest = collected["latest_proposal"]
    forced_idle = bool(result["forced_stay"])
    action = int(result["executed_action"])
    if not 0 <= action < 6:
        raise RuntimeError("executed action is outside the primitive action space")
    if forced_idle != (latest is None):
        raise RuntimeError("forced-idle/proposal contract mismatch")

    proposal_latency = None
    if latest is not None:
        if int(latest["action"]) != action:
            raise RuntimeError("executed action does not match latest proposal")
        proposal_latency = float(latest["acquisition_time_ms"]) - decision_start
        if not 0.0 <= proposal_latency <= float(budget):
            raise RuntimeError("final proposal timestamp is outside its decision")

    idle = action == STAY_ACTION
    if not idle and proposal_latency is None:
        raise RuntimeError("non-idle action requires a proposal timestamp")
    decision_latency = float(budget) if idle else float(proposal_latency)
    idle_type = "forced" if forced_idle else ("chosen" if idle else None)
    return {
        "decision_index": int(decision_index),
        "executed_action": action,
        "action_decision_latency_ms": decision_latency,
        "final_proposal_latency_ms": proposal_latency,
        "idle_type": idle_type,
        "proposal_count": len(collected["proposals"]),
    }


def _rate(
    rows: Sequence[Mapping[str, Any]], numerator: str, denominator: str
) -> float:
    top = sum(float(row.get(numerator, 0.0)) for row in rows)
    bottom = sum(float(row.get(denominator, 0.0)) for row in rows)
    return top / max(bottom, 1.0)


def summarize_condition(
    rows: Sequence[Mapping[str, Any]],
    *,
    decision_time_ms: int,
    horizon: int,
) -> dict[str, Any]:
    episodes = list(rows)
    if not episodes:
        raise ValueError("condition summary requires at least one episode")
    deliveries = [float(row["deliveries"]) for row in episodes]
    sparse = [float(row["sparse_team_reward"]) for row in episodes]
    acquisitions = sum(float(row.get("acquisitions", 0.0)) for row in episodes)
    new_targets = sum(
        float(row.get("new_target_acquisitions", 0.0)) for row in episodes
    )
    decisions = sum(float(row.get("task_decisions", 0.0)) for row in episodes)
    action_decisions = [
        record
        for row in episodes
        for record in row.get("action_decisions", ())
    ]
    if len(action_decisions) != int(decisions):
        raise RuntimeError("every task decision must have one action-time record")
    action_latencies = [
        float(record["action_decision_latency_ms"])
        for record in action_decisions
    ]
    non_idle_latencies = [
        float(record["action_decision_latency_ms"])
        for record in action_decisions
        if record["idle_type"] is None
    ]
    forced_idle_count = sum(
        record["idle_type"] == "forced" for record in action_decisions
    )
    chosen_idle_count = sum(
        record["idle_type"] == "chosen" for record in action_decisions
    )
    idle_count = forced_idle_count + chosen_idle_count
    return {
        "decision_time_ms": int(decision_time_ms),
        "horizon": int(horizon),
        "episodes": len(episodes),
        "episode_seeds": [int(row["seed"]) for row in episodes],
        "deliveries_per_episode_mean": mean(deliveries),
        "deliveries_per_episode_std": pstdev(deliveries),
        "sparse_reward_per_episode_mean": mean(sparse),
        "action_decision_latency_ms_mean": mean(action_latencies),
        "action_decision_latency_ms_median": median(action_latencies),
        "action_decision_latency_ms_std": pstdev(action_latencies),
        "non_idle_action_decision_latency_ms_mean": (
            mean(non_idle_latencies) if non_idle_latencies else None
        ),
        "idle_count": idle_count,
        "idle_rate": idle_count / max(decisions, 1.0),
        "forced_idle_count": forced_idle_count,
        "forced_idle_rate": forced_idle_count / max(decisions, 1.0),
        "chosen_idle_count": chosen_idle_count,
        "chosen_idle_rate": chosen_idle_count / max(decisions, 1.0),
        "actor_active_rate": _rate(
            episodes, "executed_proposals", "task_decisions"
        ),
        "forced_stay_rate": _rate(episodes, "forced_stays", "task_decisions"),
        "no_acquisition_decisions": int(
            sum(int(row.get("no_acquisition_decisions", 0)) for row in episodes)
        ),
        "no_acquisition_decision_rate": _rate(
            episodes, "no_acquisition_decisions", "task_decisions"
        ),
        "actionnet_queries_per_decision": _rate(
            episodes, "actionnet_queries", "task_decisions"
        ),
        "recurrent_commits_per_decision": _rate(
            episodes, "recurrent_commits", "task_decisions"
        ),
        "gaze_requests_per_decision": _rate(
            episodes, "gaze_requests", "task_decisions"
        ),
        "acquisitions_per_decision": acquisitions / max(decisions, 1.0),
        "same_target_acquisition_rate": (
            sum(
                float(row.get("same_target_acquisitions", 0.0))
                for row in episodes
            )
            / max(acquisitions, 1.0)
        ),
        "new_target_acquisition_rate": new_targets / max(acquisitions, 1.0),
        "mean_saccade_distance_degrees": (
            sum(
                float(row.get("saccade_distance_sum_degrees", 0.0))
                for row in episodes
            )
            / max(new_targets, 1.0)
        ),
        "valid_memory_fraction": mean(
            float(row["valid_memory_fraction"]) for row in episodes
        ),
        "freshness_normalized_mean": mean(
            float(row["freshness_normalized_mean"]) for row in episodes
        ),
        "memory_expiry_rate": _rate(
            episodes, "memory_expiry_count", "task_decisions"
        ),
        "prep_count": int(sum(int(row.get("prep_count", 0)) for row in episodes)),
        "exec_count": int(sum(int(row.get("exec_count", 0)) for row in episodes)),
        "dwell_count": int(
            sum(int(row.get("dwell_count", 0)) for row in episodes)
        ),
        "cross_decision_gaze_operations": int(
            sum(
                int(row.get("cross_decision_gaze_operations", 0))
                for row in episodes
            )
        ),
        "cross_decision_gaze_operation_rate": _rate(
            episodes, "cross_decision_gaze_operations", "task_decisions"
        ),
        "task_decisions": int(decisions),
        "fixed_partner_guard_failures": int(
            sum(
                int(row.get("partner_step_guard_failures", 0))
                + int(row.get("partner_full_hash_failures", 0))
                for row in episodes
            )
        ),
    }


def summarize_by_time(conditions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for decision_time in DECISION_TIMES_MS:
        selected = [
            row for row in conditions if int(row["decision_time_ms"]) == decision_time
        ]
        if len(selected) != len(HORIZONS):
            raise RuntimeError("Time summary requires exactly H0/H2/H4")
        result.append({
            "decision_time_ms": decision_time,
            "deliveries_per_episode_mean_across_horizons": mean(
                float(row["deliveries_per_episode_mean"]) for row in selected
            ),
            "action_decision_latency_ms_mean_across_horizons": mean(
                float(row["action_decision_latency_ms_mean"])
                for row in selected
            ),
            "idle_rate_mean_across_horizons": mean(
                float(row["idle_rate"]) for row in selected
            ),
            "forced_idle_rate_mean_across_horizons": mean(
                float(row["forced_idle_rate"]) for row in selected
            ),
            "chosen_idle_rate_mean_across_horizons": mean(
                float(row["chosen_idle_rate"]) for row in selected
            ),
            "forced_stay_rate_mean_across_horizons": mean(
                float(row["forced_stay_rate"]) for row in selected
            ),
            "actor_active_rate_mean_across_horizons": mean(
                float(row["actor_active_rate"]) for row in selected
            ),
            "acquisitions_per_decision_mean_across_horizons": mean(
                float(row["acquisitions_per_decision"]) for row in selected
            ),
            "actionnet_queries_per_decision_mean_across_horizons": mean(
                float(row["actionnet_queries_per_decision"]) for row in selected
            ),
            "valid_memory_fraction_mean_across_horizons": mean(
                float(row["valid_memory_fraction"]) for row in selected
            ),
        })
    return result


def viability_interpretation(
    conditions: Sequence[Mapping[str, Any]],
    by_time: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    zero_conditions = [
        f"T{row['decision_time_ms']}/H{row['horizon']}"
        for row in conditions
        if float(row["deliveries_per_episode_mean"]) == 0.0
    ]
    horizon_means = {
        f"h{horizon}": mean(
            float(row["deliveries_per_episode_mean"])
            for row in conditions
            if int(row["horizon"]) == horizon
        )
        for horizon in HORIZONS
    }
    time_means = {
        f"t{row['decision_time_ms']}": float(
            row["deliveries_per_episode_mean_across_horizons"]
        )
        for row in by_time
    }
    nonzero_horizons = sum(value > 0.0 for value in horizon_means.values())
    collapsed_horizons = [
        label
        for label, value in horizon_means.items()
        if value == 0.0 and nonzero_horizons > 0
    ]
    t200_t400_nonviable = (
        time_means["t200"] == 0.0
        and time_means["t400"] == 0.0
        and time_means["t600"] > 0.0
    )
    recommendation = (
        "GazeNet training should NOT begin yet; consider ActionNet Time "
        "adaptation first."
        if t200_t400_nonviable
        else (
            "No automatic GazeNet-training PASS is declared. Interpret the "
            "nine held-out task results and behavioral diagnostics jointly."
        )
    )
    return {
        "automatic_scientific_pass": None,
        "obvious_task_collapse": len(zero_conditions) == len(CONDITIONS),
        "zero_delivery_conditions": zero_conditions,
        "one_horizon_zero_delivery_collapse": collapsed_horizons,
        "t200_t400_clearly_nonviable_vs_t600": t200_t400_nonviable,
        "time_delivery_means": time_means,
        "horizon_delivery_means": horizon_means,
        "strong_time_degradation_threshold_applied": False,
        "excessive_forced_stay_threshold_applied": False,
        "comparable_viability_threshold_applied": False,
        "maximum_forced_stay_rate": max(
            float(row["forced_stay_rate"]) for row in conditions
        ),
        "maximum_no_acquisition_decision_rate": max(
            float(row["no_acquisition_decision_rate"]) for row in conditions
        ),
        "recommendation": recommendation,
    }


def _validate_checkpoint_contract(loaded) -> None:
    metadata = loaded.metadata
    mismatch = {}
    expected = {
        "observation_mode": "partial",
        "training_stage": "action",
        "memory_lifetime_ms": int(MEMORY_LIFETIME_MS),
        "fixed_partner_sampling_disabled": True,
        **cooking_contract(),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatch[key] = (metadata.get(key), value)
    if not set(HORIZONS).issubset({int(value) for value in metadata.get("horizons", ())}):
        mismatch["horizons"] = (metadata.get("horizons"), list(HORIZONS))
    if FIXATION_SHAPE not in metadata.get("fixation_shapes", ()):
        mismatch["fixation_shapes"] = (
            metadata.get("fixation_shapes"),
            [FIXATION_SHAPE],
        )
    probabilities = metadata.get("rule_gaze_probabilities")
    if probabilities != RULE_GAZE_PROBABILITIES:
        mismatch["rule_gaze_probabilities"] = (
            probabilities,
            RULE_GAZE_PROBABILITIES,
        )
    if mismatch:
        raise RuntimeError(f"incompatible frozen Partial Action checkpoint: {mismatch}")


def _evaluate_condition(
    loaded,
    repository,
    *,
    decision_time_ms: int,
    horizon: int,
    seeds: Sequence[int],
    episode_length: int,
) -> dict[str, Any]:
    rows = []
    with torch.no_grad():
        for episode, seed in enumerate(seeds):
            backend = DirectOvercookedBackend(
                seed=int(seed),
                horizon=int(horizon),
                episode_length=int(episode_length),
                device=loaded.device,
                partner_repository=repository,
            )
            adapter = PartialObservationAdapter(
                backend,
                decision_time=int(decision_time_ms),
                fixation_shape=FIXATION_SHAPE,
                gaze_policy=None,
                rule_gaze_seed=int(seed) + 99_001,
                rule_gaze_probabilities=RULE_GAZE_PROBABILITIES,
            )
            hidden = np.zeros((1, 64), dtype=np.float32)
            mask = np.zeros((1,), dtype=np.float32)
            sparse = 0.0
            deliveries = 0
            actions = []
            action_decisions = []
            cross_decision = 0
            backend.verify_partner_parameters()
            for _decision in range(int(episode_length)):
                pre_hidden = hidden.copy()
                pre_mask = mask.copy()

                def proposal(observation, _critic, available, _timestamp):
                    probabilities, candidate = loaded.actor.get_probs(
                        observation[None],
                        pre_hidden[None],
                        pre_mask[None],
                        available[None],
                    )
                    action = probabilities.argmax(-1)
                    return {
                        "action": int(action.item()),
                        "candidate_hidden": candidate.detach().cpu().numpy()[0],
                    }

                collected = adapter.collect_decision(proposal)
                if adapter.gaze.phase is not GazePhase.READY:
                    cross_decision += 1
                result = adapter.commit_collected(collected)
                latest = collected["latest_proposal"]
                action_decisions.append(
                    action_decision_record(
                        collected,
                        result,
                        decision_index=_decision,
                        decision_time_ms=decision_time_ms,
                    )
                )
                if latest is not None:
                    hidden = latest["candidate_hidden"].copy()
                    mask.fill(1.0)
                sparse += float(result["sparse_team_reward"])
                deliveries += int(result["deliveries"])
                actions.append(int(result["executed_action"]))
                if result["done"]:
                    break
            backend.verify_partner_parameters()
            metrics = adapter.metrics()
            rows.append({
                "episode": episode,
                "seed": int(seed),
                "deliveries": deliveries,
                "sparse_team_reward": sparse,
                "action_trajectory_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
                "action_decisions": action_decisions,
                "cross_decision_gaze_operations": cross_decision,
                "saccade_distance_sum_degrees": float(adapter.distance_sum_degrees),
                **metrics,
            })
    summary = summarize_condition(
        rows,
        decision_time_ms=decision_time_ms,
        horizon=horizon,
    )
    return {"summary": summary, "episodes": rows}


def _checkpoint_id(path: Path, checkpoint_sha256: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._-") or "checkpoint"
    return f"{stem}_{checkpoint_sha256[:12]}"


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen Partial ActionNet Time x Horizon viability",
        "",
        (
            "No thresholded scientific PASS is assigned. Primary results are held-out "
            "sparse deliveries; behavior and memory diagnostics are read-only."
        ),
        "",
        "| T | H | deliveries/episode | action decision ms | idle | forced idle | chosen idle | actor-active | acquisitions/decision | ActionNet queries/decision | valid memory |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["conditions"]:
        lines.append(
            f"| {row['decision_time_ms']} | H{row['horizon']} | "
            f"{row['deliveries_per_episode_mean']:.4f} | "
            f"{row['action_decision_latency_ms_mean']:.3f} | "
            f"{row['idle_rate']:.6f} | {row['forced_idle_rate']:.6f} | "
            f"{row['chosen_idle_rate']:.6f} | {row['actor_active_rate']:.6f} | "
            f"{row['acquisitions_per_decision']:.6f} | "
            f"{row['actionnet_queries_per_decision']:.6f} | "
            f"{row['valid_memory_fraction']:.6f} |"
        )
    lines.extend([
        "",
        "## Time means across H0/H2/H4",
        "",
        "| T | deliveries/episode | action decision ms | idle | forced idle | chosen idle | actor-active | acquisitions/decision | ActionNet queries/decision | valid memory |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["by_time"]:
        lines.append(
            f"| {row['decision_time_ms']} | "
            f"{row['deliveries_per_episode_mean_across_horizons']:.4f} | "
            f"{row['action_decision_latency_ms_mean_across_horizons']:.3f} | "
            f"{row['idle_rate_mean_across_horizons']:.6f} | "
            f"{row['forced_idle_rate_mean_across_horizons']:.6f} | "
            f"{row['chosen_idle_rate_mean_across_horizons']:.6f} | "
            f"{row['actor_active_rate_mean_across_horizons']:.6f} | "
            f"{row['acquisitions_per_decision_mean_across_horizons']:.6f} | "
            f"{row['actionnet_queries_per_decision_mean_across_horizons']:.6f} | "
            f"{row['valid_memory_fraction_mean_across_horizons']:.6f} |"
        )
    interpretation = report["interpretation"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Obvious all-condition task collapse: {interpretation['obvious_task_collapse']}",
        f"- Zero-delivery conditions: {interpretation['zero_delivery_conditions']}",
        f"- Zero-delivery horizon collapse: {interpretation['one_horizon_zero_delivery_collapse']}",
        f"- T200/T400 clearly nonviable while T600 remains viable: {interpretation['t200_t400_clearly_nonviable_vs_t600']}",
        f"- Maximum forced-STAY rate: {interpretation['maximum_forced_stay_rate']:.6f}",
        f"- Maximum no-acquisition rate: {interpretation['maximum_no_acquisition_decision_rate']:.6f}",
        f"- Recommendation: {interpretation['recommendation']}",
        "- Strong-degradation/comparable-viability thresholds were intentionally not invented.",
        "",
        "## Contract",
        "",
        f"- Held-out seeds: {report['held_out_seeds'][0]}--{report['held_out_seeds'][-1]} (identical for all nine conditions)",
        "- ActionNet and critic parameter hashes: unchanged",
        "- Optimizer/PPO/GazeNet: absent",
        "- Action-decision latency: final executed non-idle proposal timestamp relative to decision start",
        "- Idle latency: full T200/T400/T600 budget for both forced and chosen STAY; types remain separate",
        "- Gaze: canonical rule-based 0.60/0.20/0.15/0.05",
        "- Memory: canonical 2400 ms; footprint: plus",
        "- Partner: canonical fixed HSP S2-s24 seed3",
        "- Cross-decision gaze count: pending committed gaze operations observed at task-decision deadlines",
        "",
    ])
    return "\n".join(lines)


def run(args) -> dict[str, Any]:
    checkpoint_path = args.action_checkpoint.expanduser().resolve()
    checkpoint_content_before = file_sha256(checkpoint_path)
    loaded = load_partial_checkpoint_for_evaluation(
        checkpoint_path, device=args.device
    )
    _validate_checkpoint_contract(loaded)
    loaded.actor.eval().requires_grad_(False)
    loaded.critic.eval().requires_grad_(False)
    actor_before = parameter_hash(loaded.actor)
    critic_before = parameter_hash(loaded.critic)
    seeds = evaluation_seeds(args.seed_start, args.episodes)
    repository = FixedHSPPartnerRepository(
        device=loaded.device,
        checkpoint=Path(CANONICAL_PARTNER).resolve(),
    )
    conditions = []
    episode_results = {}
    for decision_time_ms, horizon in CONDITIONS:
        result = _evaluate_condition(
            loaded,
            repository,
            decision_time_ms=decision_time_ms,
            horizon=horizon,
            seeds=seeds,
            episode_length=args.episode_length,
        )
        if result["summary"]["episode_seeds"] != list(seeds):
            raise RuntimeError("held-out seed list changed between conditions")
        conditions.append(result["summary"])
        episode_results[f"t{decision_time_ms}_h{horizon}"] = result["episodes"]
        print(f"evaluated T{decision_time_ms}/H{horizon}", flush=True)
    loaded.assert_unchanged()
    actor_after = parameter_hash(loaded.actor)
    critic_after = parameter_hash(loaded.critic)
    checkpoint_content_after = file_sha256(checkpoint_path)
    if (
        actor_before != actor_after
        or critic_before != critic_after
        or checkpoint_content_before != checkpoint_content_after
    ):
        raise RuntimeError("read-only viability evaluation changed frozen state")
    by_time = summarize_by_time(conditions)
    report = {
        "analysis": (
            "frozen_partial_action_time_horizon_viability_v2_"
            "action_decision_latency"
        ),
        "checkpoint_absolute_path": str(checkpoint_path),
        "checkpoint_sha256_before": checkpoint_content_before,
        "checkpoint_sha256_after": checkpoint_content_after,
        "checkpoint_completed_steps": loaded.completed_steps,
        "checkpoint_training_decision_times": loaded.metadata.get("decision_times"),
        "actor_parameter_sha256_before": actor_before,
        "actor_parameter_sha256_after": actor_after,
        "critic_parameter_sha256_before": critic_before,
        "critic_parameter_sha256_after": critic_after,
        "decision_times_ms": list(DECISION_TIMES_MS),
        "horizons": list(HORIZONS),
        "fixation_shape": FIXATION_SHAPE,
        "gaze_type": GAZE_TYPE,
        "rule_gaze_probabilities": RULE_GAZE_PROBABILITIES,
        "memory_lifetime_ms": MEMORY_LIFETIME_MS,
        "episode_length": args.episode_length,
        "episodes_per_condition": args.episodes,
        "total_episodes": len(CONDITIONS) * args.episodes,
        "held_out_seed_partition": "FINAL_ANALYSIS_SEED_START",
        "held_out_seeds": list(seeds),
        "deterministic_action_selection": True,
        "action_decision_latency_contract": {
            "unit": "milliseconds",
            "non_idle": (
                "final executed proposal acquisition timestamp minus decision start"
            ),
            "idle": "full decision-time budget",
            "idle_types": ["forced", "chosen"],
            "environment_commit": "decision deadline; reported separately by contract",
        },
        "optimizer_free": True,
        "ppo_updates": 0,
        "gazenet_used": False,
        "conditions": conditions,
        "by_time": by_time,
        "interpretation": viability_interpretation(conditions, by_time),
        "episode_results": episode_results,
        "parameter_hashes_preserved": True,
        "checkpoint_file_preserved": True,
        "fixed_partner_provenance": repository.provenance(),
        "cooking_contract": cooking_contract(),
    }
    checkpoint_id = _checkpoint_id(checkpoint_path, checkpoint_content_before)
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"partial_time_horizon_viability_{checkpoint_id}.json"
    markdown_path = report_dir / f"partial_time_horizon_viability_{checkpoint_id}.md"
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with markdown_path.open("x", encoding="utf-8") as handle:
        handle.write(markdown(report))
    return {
        **report,
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--action-checkpoint", type=Path, required=True)
    value.add_argument("--episodes", type=int, default=EPISODES_PER_CONDITION)
    value.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    value.add_argument("--seed-start", type=int, default=FINAL_ANALYSIS_SEED_START)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--torch-threads", type=int, default=2)
    value.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
    )
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episode_length != EPISODE_LENGTH:
        raise ValueError(f"--episode-length must remain {EPISODE_LENGTH}")
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    report = run(args)
    print(json.dumps({
        "json_report": report["json_report"],
        "markdown_report": report["markdown_report"],
        "total_episodes": report["total_episodes"],
        "interpretation": report["interpretation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
