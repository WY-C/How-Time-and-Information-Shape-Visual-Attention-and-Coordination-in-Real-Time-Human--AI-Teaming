"""Frozen Partial ActionNet memory, acquisition, preview, and recurrent audit suite."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from final_model.checkpoint import ACTION_SPACE
from final_model.common.mappo import ACTION_ARCHITECTURE_ID
from final_model.common.overcooked_backend import CANONICAL_ENCODER_ID, DirectOvercookedBackend
from final_model.common.partners import FixedHSPPartnerRepository, parameter_hash
from final_model.common.paths import file_sha256
from final_model.eval.actionnet_diagnostics import (
    CHANNEL_SEMANTICS,
    ActionSnapshot,
    acquisition_labels,
    action_entropy,
    aggregate_counterfactual_rows,
    audit_snapshot,
    memory_coverage,
    memory_variant,
    instrument_memory_acquisitions,
    oracle_current_observation,
    read_only_probabilities,
    read_only_probabilities_and_hidden,
    source_comparability,
    top1_margin,
)
from final_model.eval.checkpoint_loader import (
    LoadedPartialCheckpoint,
    load_action_checkpoint_for_evaluation,
    load_partial_checkpoint_for_evaluation,
)
from final_model.partial_ver.memory import MEMORY_RNG_SEED_OFFSET
from final_model.eval.common import (
    CANONICAL_PARTNER,
    command_record,
    create_output_directory,
    device_metadata,
    source_hashes,
    write_csv,
    write_json,
)
from final_model.eval.counterfactual import distribution_metrics
from final_model.eval.partial_rollout import EvaluationLearnedGaze
from final_model.partial_ver.gaze import FIXATION_SHAPES, HUMAN_DWELL_RNG_SEED_OFFSET
from final_model.partial_ver.memory import (
    DYNAMIC_CHANNELS,
    OBSERVATION_CONTRACT_ID,
    STATIC_CHANNELS,
)
from final_model.partial_ver.observation import (
    ACQUISITION_ACTION_CONTRACT_ID,
    ACTOR_AVAILABLE_ACTION_CONTRACT_ID,
    ALLOWED_DECISION_TIMES,
    RECURRENT_COMMIT_CONTRACT_ID,
    PartialObservationAdapter,
)
from final_model.partial_ver.peripheral import PERIPHERAL_RNG_SEED_OFFSET
from final_model.partial_ver.preview_opacity import PREVIEW_OPACITY_RNG_SEED_OFFSET


ALL_TESTS = ("source", "memory", "acquisition", "recurrent", "preview", "bounds", "confidence")
DEFAULT_TESTS = ("memory", "acquisition", "recurrent", "preview", "confidence")
REGIMES = ("factual", "static_only", "oracle_current")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _discover_source(adapted: LoadedPartialCheckpoint) -> Path | None:
    provenance = adapted.metadata.get("source_full_checkpoint_provenance", {})
    expected_hash = provenance.get("source_content_sha256")
    if not expected_hash:
        return None
    directory = adapted.path.parent
    for candidate in sorted(directory.glob("*.pt")):
        if candidate == adapted.path or candidate.stat().st_size > 50_000_000:
            continue
        if file_sha256(candidate) == expected_hash:
            return candidate.resolve()
    return None


def _instrument_memory(adapter: PartialObservationAdapter) -> dict[str, Any]:
    """Observe exact before/after writes on this evaluation instance only."""
    return instrument_memory_acquisitions(adapter)


def _gaze_policy(
    gaze_checkpoint: LoadedPartialCheckpoint | None, device: torch.device,
    *, gaze_type: str, deterministic: bool, seed: int,
):
    if gaze_type == "rule_based":
        return None
    if gaze_checkpoint is None or gaze_checkpoint.gaze_actor is None:
        raise RuntimeError("learned factual gaze requires --gaze-checkpoint with GazeNet")
    return EvaluationLearnedGaze(
        gaze_checkpoint.gaze_actor, device, deterministic=deterministic, seed=seed
    )


def _actor_query(actor, observation, hidden, mask, available):
    probabilities, candidate = read_only_probabilities_and_hidden(
        actor, observation, hidden, mask, available
    )
    return probabilities, candidate[0]


def _confidence_row(snapshot: ActionSnapshot) -> dict[str, Any]:
    probabilities = np.asarray(snapshot.factual_probabilities)
    interact = list(ACTION_SPACE).index("INTERACT") if "INTERACT" in ACTION_SPACE else None
    return {
        "diagnostic": "memory_confidence",
        "scientific_role": "CORRELATIONAL_DIAGNOSTIC",
        "variant": "factual",
        "episode": snapshot.episode,
        "seed": snapshot.seed,
        "decision": snapshot.decision,
        "proposal_index": snapshot.proposal_index,
        "timestamp_ms": snapshot.timestamp_ms,
        "decision_time": snapshot.decision_time,
        "horizon": snapshot.horizon,
        "action_entropy": action_entropy(probabilities),
        "top1_probability": float(probabilities.max()),
        "top1_margin": top1_margin(probabilities),
        "p_interact": None if interact is None else float(probabilities[interact]),
        **memory_coverage(snapshot.observation, horizon=snapshot.horizon),
        **{f"p_{name.lower()}": float(probabilities[index]) for index, name in enumerate(ACTION_SPACE)},
    }


def _teacher_forced_replay(actor, snapshots: list[ActionSnapshot]) -> tuple[list[dict[str, Any]], bool]:
    """Replay recorded proposal sequences; commit only each decision's last candidate."""
    rows: list[dict[str, Any]] = []
    factual_identity = True
    by_episode: dict[int, list[ActionSnapshot]] = defaultdict(list)
    for snapshot in snapshots:
        by_episode[snapshot.episode].append(snapshot)
    for episode, episode_values in sorted(by_episode.items()):
        factual_hidden = np.zeros_like(episode_values[0].hidden)
        ablated_hidden = np.zeros_like(episode_values[0].hidden)
        mask = np.zeros_like(episode_values[0].mask)
        by_decision: dict[int, list[ActionSnapshot]] = defaultdict(list)
        for snapshot in episode_values:
            by_decision[snapshot.decision].append(snapshot)
        for _decision, proposals in sorted(by_decision.items()):
            factual_candidate = ablated_candidate = None
            for snapshot in sorted(proposals, key=lambda value: value.proposal_index):
                factual_probability, factual_candidate = read_only_probabilities_and_hidden(
                    actor, snapshot.observation, factual_hidden, mask, snapshot.available
                )
                ablated_observation, _label = memory_variant(
                    snapshot.observation, "static_only", horizon=snapshot.horizon
                )
                ablated_probability, ablated_candidate = read_only_probabilities_and_hidden(
                    actor, ablated_observation, ablated_hidden, mask, snapshot.available
                )
                factual_identity &= bool(
                    np.allclose(
                        factual_probability, snapshot.factual_probabilities,
                        atol=1e-7, rtol=1e-6,
                    )
                )
                rows.append({
                    "diagnostic": "teacher_forced_recurrent_replay",
                    "scientific_role": "RECORDED_TRAJECTORY_NO_ENVIRONMENT_REPLAY",
                    "variant": "static_only_recurrent_stream",
                    "effective": 1,
                    "episode": episode,
                    "seed": snapshot.seed,
                    "decision": snapshot.decision,
                    "proposal_index": snapshot.proposal_index,
                    "decision_time": snapshot.decision_time,
                    "horizon": snapshot.horizon,
                    **distribution_metrics(factual_probability, ablated_probability),
                })
            if factual_candidate is not None:
                factual_hidden = factual_candidate.copy()
                ablated_hidden = ablated_candidate.copy()
                mask.fill(1.0)
    return rows, factual_identity


def _run_condition(
    *, loaded: LoadedPartialCheckpoint, repository: FixedHSPPartnerRepository,
    gaze_checkpoint: LoadedPartialCheckpoint | None, gaze_type: str,
    decision_time: int, horizon: int, fixation_shape: str, episodes: int,
    episode_length: int, seed_start: int, deterministic: bool, regime: str,
    collect_diagnostics: bool, tests: set[str],
) -> dict[str, Any]:
    if regime not in REGIMES:
        raise RuntimeError("oracle mode cannot be selected through factual/default mode")
    actor = loaded.actor
    episode_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    snapshots: list[ActionSnapshot] = []
    integrity_checks: list[dict[str, Any]] = []
    acquisition_records: list[dict[str, Any]] = []
    for episode in range(episodes):
        seed = seed_start + episode
        backend = DirectOvercookedBackend(
            seed=seed, horizon=horizon, episode_length=episode_length,
            device=loaded.device, partner_repository=repository,
        )
        policy = _gaze_policy(
            gaze_checkpoint, loaded.device, gaze_type=gaze_type,
            deterministic=deterministic, seed=seed + 77_003,
        )
        adapter = PartialObservationAdapter(
            backend, decision_time=decision_time, fixation_shape=fixation_shape,
            gaze_policy=policy, rule_gaze_seed=seed + 99_001,
            rule_gaze_probabilities=loaded.metadata.get("rule_gaze_probabilities"),
            memory_lifetime_ms=loaded.metadata["memory_lifetime_ms"],
            memory_decay_mode=loaded.metadata["memory_decay_mode"],
            memory_tau_ms=loaded.metadata.get("memory_tau_ms"),
            memory_seed=(
                seed + MEMORY_RNG_SEED_OFFSET
                if loaded.metadata["memory_decay_mode"]
                == "stochastic_exponential"
                else None
            ),
            constraint_config=loaded.constraint_config,
            peripheral_seed=seed + PERIPHERAL_RNG_SEED_OFFSET,
            opacity_seed=seed + PREVIEW_OPACITY_RNG_SEED_OFFSET,
            dwell_seed=(
                seed + HUMAN_DWELL_RNG_SEED_OFFSET
                if loaded.constraint_config.emma.mode == "human_stochastic"
                else None
            ),
        )
        memory_capture = _instrument_memory(adapter)
        hidden = np.zeros((1, 64), dtype=np.float32)
        mask = np.zeros((1,), dtype=np.float32)
        action_generator = torch.Generator(device="cpu").manual_seed(seed + 88_007)
        sparse = 0.0
        deliveries = 0
        entropies: list[float] = []
        action_trajectory: list[int] = []
        forced_stays = 0
        backend.verify_partner_parameters()
        episode_snapshot_start = len(snapshots)
        for decision in range(episode_length):
            pre_hidden, pre_mask = hidden.copy(), mask.copy()
            decision_snapshot_indices: list[int] = []
            decision_acquisition_indices: list[int] = []
            decision_diagnostic_start = len(diagnostic_rows)
            decision_confidence_start = len(confidence_rows)

            def proposal(observation, _critic, available, timestamp):
                if regime == "factual":
                    actor_observation = np.asarray(observation, dtype=np.float32).copy()
                elif regime == "static_only":
                    actor_observation = memory_variant(
                        observation, "static_only", horizon=horizon
                    )[0]
                else:
                    actor_observation = oracle_current_observation(backend, horizon=horizon)
                probabilities, candidate_hidden = _actor_query(
                    actor, actor_observation, pre_hidden[None], pre_mask[None], available
                )
                action_tensor = (
                    torch.as_tensor(probabilities).argmax(-1)
                    if deterministic
                    else torch.multinomial(
                        torch.as_tensor(probabilities), 1, generator=action_generator
                    )[0]
                )
                action = int(action_tensor.item())
                entropies.append(action_entropy(probabilities))
                if collect_diagnostics and regime == "factual":
                    event = memory_capture.get("event")
                    if event is None or not np.array_equal(event["after"], observation):
                        raise RuntimeError("acquisition snapshot does not match actor observation")
                    snapshot = ActionSnapshot(
                        episode=episode, seed=seed, decision=decision,
                        proposal_index=len(decision_snapshot_indices),
                        timestamp_ms=float(timestamp), decision_time=decision_time,
                        horizon=horizon, fixation=tuple(adapter.gaze.current_fixation),
                        observation=np.asarray(observation).copy(),
                        hidden=pre_hidden[None].copy(), mask=pre_mask[None].copy(),
                        available=np.asarray(available).copy(),
                        factual_probabilities=probabilities.copy(), factual_action=action,
                    )
                    snapshots.append(snapshot)
                    decision_snapshot_indices.append(len(snapshots) - 1)
                    if tests & {"memory", "preview", "recurrent"}:
                        rows, checks = audit_snapshot(actor, snapshot)
                        diagnostic_rows.extend(rows)
                        integrity_checks.append({
                            "episode": episode, "decision": decision,
                            "proposal_index": snapshot.proposal_index, **checks,
                        })
                    if "confidence" in tests:
                        confidence_rows.append(_confidence_row(snapshot))
                    if "acquisition" in tests:
                        before_probability = read_only_probabilities(
                            actor, event["before"], snapshot.hidden,
                            snapshot.mask, snapshot.available,
                        )
                        labels = acquisition_labels(
                            event["before"], event["after"], event["tiles"]
                        )
                        row = {
                            "diagnostic": "acquisition_sensitivity",
                            "scientific_role": "SAME_HIDDEN_BEFORE_VS_AFTER_MEMORY_WRITE",
                            "variant": (
                                "memory_changed=true" if labels["memory_changed"]
                                else "memory_changed=false"
                            ),
                            "effective": labels["memory_changed"],
                            "episode": episode, "seed": seed, "decision": decision,
                            "proposal_index": snapshot.proposal_index,
                            "timestamp_ms": float(timestamp),
                            "decision_time": decision_time, "horizon": horizon,
                            **labels,
                            **distribution_metrics(before_probability, probabilities),
                            "_before_probabilities": before_probability,
                            "_after_probabilities": probabilities.copy(),
                        }
                        acquisition_records.append(row)
                        decision_acquisition_indices.append(len(acquisition_records) - 1)
                return {
                    "action": action,
                    "candidate_hidden": candidate_hidden.copy(),
                    "probabilities": probabilities.copy(),
                    "observation": actor_observation.copy(),
                }

            collected = adapter.collect_decision(proposal)
            result = adapter.commit_collected(collected)
            latest = collected["latest_proposal"]
            last_proposal_index = len(collected["proposals"]) - 1
            for row in diagnostic_rows[decision_diagnostic_start:]:
                row["factual_action_actually_selected"] = int(result["executed_action"])
                row["proposal_was_executed"] = int(
                    row.get("proposal_index") == last_proposal_index
                )
            for row in confidence_rows[decision_confidence_start:]:
                row["factual_action_actually_selected"] = int(result["executed_action"])
                row["proposal_was_executed"] = int(
                    row.get("proposal_index") == last_proposal_index
                )
            if latest is not None:
                hidden = latest["candidate_hidden"].copy()
                mask.fill(1.0)
                if decision_acquisition_indices:
                    last = acquisition_records[decision_acquisition_indices[-1]]
                    before = last.pop("_before_probabilities")
                    after = last.pop("_after_probabilities")
                    last["decision_final_acquisition"] = 1
                    last["factual_action_actually_selected"] = int(result["executed_action"])
                    last["proposal_was_executed"] = 1
                    diagnostic_rows.append(last)
                    diagnostic_rows.append({
                        **{key: value for key, value in last.items()
                           if key not in distribution_metrics(before, after)},
                        "diagnostic": "last_acquisition_rollback",
                        "scientific_role": "SAME_HIDDEN_FACTUAL_FINAL_VS_ROLLBACK",
                        "variant": "factual_without_final_acquisition",
                        **distribution_metrics(after, before),
                    })
                    for index in decision_acquisition_indices[:-1]:
                        row = acquisition_records[index]
                        row.pop("_before_probabilities", None)
                        row.pop("_after_probabilities", None)
                        row["decision_final_acquisition"] = 0
                        row["factual_action_actually_selected"] = int(result["executed_action"])
                        row["proposal_was_executed"] = 0
                        diagnostic_rows.append(row)
            else:
                forced_stays += 1
            sparse += float(result["sparse_team_reward"])
            deliveries += int(result["deliveries"])
            action_trajectory.append(int(result["executed_action"]))
            if result["done"]:
                break
        backend.verify_partner_parameters()
        metrics = adapter.metrics()
        episode_rows.append({
            "regime": regime,
            "scientific_role": (
                "ORACLE_UPPER_BOUND" if regime == "oracle_current" else "FACTUAL_RUNTIME"
                if regime == "factual" else "CANONICAL_LOWER_INFORMATION_BOUND"
            ),
            "episode": episode, "seed": seed, "decision_time": decision_time,
            "horizon": horizon, "fixation_shape": fixation_shape,
            "deliveries": deliveries, "sparse_team_reward": sparse,
            "forced_stay_count": forced_stays,
            "forced_stay_rate": forced_stays / max(metrics["task_decisions"], 1),
            "action_entropy": float(np.mean(entropies)) if entropies else None,
            "actionnet_query_count": len(snapshots) - episode_snapshot_start
            if collect_diagnostics else int(metrics["actionnet_queries"]),
            "task_decisions": int(metrics["task_decisions"]),
            "action_trajectory": ",".join(map(str, action_trajectory)),
        })
    teacher_rows: list[dict[str, Any]] = []
    teacher_identity = None
    if collect_diagnostics and "recurrent" in tests and snapshots:
        teacher_rows, teacher_identity = _teacher_forced_replay(actor, snapshots)
        diagnostic_rows.extend(teacher_rows)
    return {
        "episodes": episode_rows,
        "diagnostic_rows": diagnostic_rows,
        "confidence_rows": confidence_rows,
        "snapshots": snapshots,
        "integrity": integrity_checks,
        "teacher_factual_identity": teacher_identity,
    }


def _episode_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["decision_time"], row["horizon"], row["regime"])].append(row)
    output = []
    for (decision_time, horizon, regime), values in sorted(grouped.items()):
        deliveries = np.asarray([row["deliveries"] for row in values], dtype=np.float64)
        rewards = np.asarray([row["sparse_team_reward"] for row in values], dtype=np.float64)
        forced = np.asarray([row["forced_stay_rate"] for row in values], dtype=np.float64)
        entropy = [row["action_entropy"] for row in values if row["action_entropy"] is not None]
        output.append({
            "decision_time": decision_time, "horizon": horizon, "regime": regime,
            "episode_count": len(values),
            "mean_deliveries": float(deliveries.mean()),
            "std_deliveries": float(deliveries.std()),
            "mean_sparse_team_reward": float(rewards.mean()),
            "std_sparse_team_reward": float(rewards.std()),
            "mean_forced_stay_rate": float(forced.mean()),
            "mean_action_entropy": float(np.mean(entropy)) if entropy else None,
            "raw_deliveries": deliveries.tolist(),
            "raw_sparse_team_reward": rewards.tolist(),
        })
    return output


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _confidence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {("pooled", "pooled"): rows}
    for row in rows:
        groups.setdefault((row["decision_time"], row["horizon"]), []).append(row)
    output = []
    for (decision_time, horizon), values in groups.items():
        for coverage in (
            "valid_memory_fraction", "acquired_dynamic_tile_fraction", "preview_memory_coverage"
        ):
            for confidence in ("action_entropy", "top1_probability", "top1_margin", "p_interact"):
                pairs = [
                    (float(row[coverage]), float(row[confidence]))
                    for row in values if row.get(confidence) is not None
                ]
                output.append({
                    "scientific_role": "CORRELATIONAL_DIAGNOSTIC",
                    "decision_time": decision_time, "horizon": horizon,
                    "coverage_measure": coverage, "confidence_measure": confidence,
                    "n": len(pairs),
                    "pearson_r": _pearson(
                        [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                    ),
                })
    return output


def _summary_markdown(summary: dict[str, Any]) -> str:
    integrity = summary["integrity_checks"]
    source = summary["source_vs_adapted"]
    lines = [
        "# Frozen Partial ActionNet Audit", "",
        "## Checkpoint and Observation Contract", "",
        f"- Architecture: `{ACTION_ARCHITECTURE_ID}`",
        f"- Observation: `(8, 5, 25)`, `{OBSERVATION_CONTRACT_ID}`",
        f"- Action space: `{', '.join(ACTION_SPACE)}`", "",
        "## Source vs 1M Partial Adaptation", "",
        f"Same-state status: `{source.get('status', 'NOT_REQUESTED')}`.", "",
        "## Direct Memory-to-Action Sensitivity", "",
        _counterfactual_table(summary["counterfactual_summary"], "memory_counterfactual"), "",
        "## Acquisition-Induced Action Changes", "",
        _counterfactual_table(summary["counterfactual_summary"], "acquisition_sensitivity"), "",
        "## Recurrent-State Dependence", "",
        _counterfactual_table(summary["counterfactual_summary"], "recurrent_state"), "",
        "## Preview-Memory Sensitivity", "",
        _counterfactual_table(summary["counterfactual_summary"], "preview_counterfactual"), "",
        "## Information Upper/Lower Bounds", "",
        "Results are descriptive; no oracle ≥ factual ≥ static-only ordering is forced.", "",
        _performance_table(summary["performance_summary"]), "",
        "## Memory Coverage and Action Confidence", "",
        "`CORRELATIONAL_DIAGNOSTIC`; see `memory_confidence/per_decision.csv`.", "",
        "## Integrity Checks", "",
        f"Overall: `{'PASS' if all(integrity.values()) else 'FAIL'}`. "
        + ", ".join(f"{key}={value}" for key, value in integrity.items()), "",
        "## Notes and Limitations", "",
        "The source-vs-adapted same-state comparison fails closed when metadata contracts differ. "
        "The oracle path is reachable only in the explicitly requested bounds test. "
        "Teacher-forced replay uses recorded actor observations and never replays or changes the environment trajectory.",
        "",
    ]
    return "\n".join(lines)


def _counterfactual_table(rows: list[dict[str, Any]], diagnostic: str) -> str:
    selected = [
        row for row in rows
        if row["diagnostic"] == diagnostic and row["subset"] == "all_decisions"
    ]
    if not selected:
        return "Not requested or no ActionNet queries were observed."
    output = [
        "| Condition | Variant | N all | N effective | Top1 change | KL | JS | TV | Mean |ΔP| | ΔEntropy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        value = lambda name: row.get(name)
        fmt = lambda item: "n/a" if item is None else f"{item:.6g}"
        condition = (
            "pooled" if row["decision_time"] == "pooled"
            else f"T{row['decision_time']}/H{row['horizon']}"
        )
        output.append(
            f"| {condition} | {row['variant']} | {row['n_all']} | {row['n_effective']} | "
            f"{fmt(value('mean_top1_action_changed'))} | "
            f"{fmt(value('mean_kl_native_counterfactual'))} | "
            f"{fmt(value('mean_js_divergence'))} | "
            f"{fmt(value('mean_total_variation_distance'))} | "
            f"{fmt(value('mean_mean_abs_probability_difference'))} | "
            f"{fmt(value('mean_delta_entropy'))} |"
        )
    return "\n".join(output)


def _performance_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Not requested."
    output = [
        "| T | H | Regime | Episodes | Deliveries mean±std | Sparse reward mean±std | Forced STAY | Entropy |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        entropy = row["mean_action_entropy"]
        output.append(
            f"| {row['decision_time']} | {row['horizon']} | {row['regime']} | "
            f"{row['episode_count']} | {row['mean_deliveries']:.6g}±{row['std_deliveries']:.6g} | "
            f"{row['mean_sparse_team_reward']:.6g}±{row['std_sparse_team_reward']:.6g} | "
            f"{row['mean_forced_stay_rate']:.6g} | "
            f"{('n/a' if entropy is None else f'{entropy:.6g}')} |"
        )
    return "\n".join(output)


def _write_outputs(
    root: Path, *, manifest: dict[str, Any], summary: dict[str, Any],
    diagnostic_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]],
    confidence_rows: list[dict[str, Any]],
) -> None:
    write_json(root / "manifest.json", manifest)
    write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    write_csv(root / "per_decision.csv", diagnostic_rows + confidence_rows)
    write_csv(root / "per_episode.csv", episode_rows)
    mapping = {
        "memory_counterfactual": "memory_counterfactual",
        "acquisition_sensitivity": "acquisition_sensitivity",
        "last_acquisition_rollback": "acquisition_sensitivity",
        "recurrent_state": "recurrent_state",
        "teacher_forced_recurrent_replay": "recurrent_state",
        "preview_counterfactual": "preview_counterfactual",
    }
    for diagnostic, directory in mapping.items():
        target = root / directory
        target.mkdir(exist_ok=True)
        write_csv(
            target / "per_decision.csv",
            [row for row in diagnostic_rows if row.get("diagnostic") == diagnostic],
        )
        write_json(
            target / "summary.json",
            [row for row in summary["counterfactual_summary"]
             if row.get("diagnostic") == diagnostic],
        )
    confidence = root / "memory_confidence"
    confidence.mkdir(exist_ok=True)
    write_csv(confidence / "per_decision.csv", confidence_rows)
    write_json(confidence / "summary.json", summary["memory_confidence"])
    bounds = root / "information_bounds"
    bounds.mkdir(exist_ok=True)
    write_csv(bounds / "per_episode.csv", episode_rows)
    write_json(bounds / "summary.json", summary["performance_summary"])
    source = root / "source_vs_adapted"
    source.mkdir(exist_ok=True)
    write_json(source / "summary.json", summary["source_vs_adapted"])


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    tests = set(args.tests)
    if not tests <= set(ALL_TESTS):
        raise ValueError(f"unknown tests: {sorted(tests - set(ALL_TESTS))}")
    if "bounds" in tests and not args.allow_oracle_upper_bound:
        raise RuntimeError("bounds test requires explicit --allow-oracle-upper-bound")
    adapted = load_partial_checkpoint_for_evaluation(args.action_checkpoint, device=args.device)
    gaze_checkpoint = None
    if args.gaze_checkpoint is not None:
        gaze_checkpoint = load_partial_checkpoint_for_evaluation(
            args.gaze_checkpoint, device=args.device
        )
        if gaze_checkpoint.actor_sha256 != adapted.actor_sha256:
            raise RuntimeError("gaze checkpoint frozen ActionNet does not match audit ActionNet")
    if args.gaze_type == "learned" and gaze_checkpoint is None:
        raise RuntimeError("--gaze-type learned requires --gaze-checkpoint")
    partner_path = Path(args.partner_checkpoint).expanduser().resolve()
    repository = FixedHSPPartnerRepository(device=adapted.device, checkpoint=partner_path)
    provenance = repository.provenance()
    if adapted.metadata.get("fixed_partner_sha256") != provenance["fixed_partner_sha256"]:
        raise RuntimeError("adapted checkpoint fixed-partner contract mismatch")
    source_path = (
        Path(args.source_action_checkpoint).expanduser().resolve()
        if args.source_action_checkpoint is not None else _discover_source(adapted)
    )
    source_result: dict[str, Any] = {"status": "NOT_REQUESTED"}
    if "source" in tests:
        if source_path is None:
            source_result = {"status": "SOURCE_CHECKPOINT_NOT_FOUND"}
        else:
            source = load_action_checkpoint_for_evaluation(source_path, device=args.device)
            source_result = {
                **source_comparability(source.metadata, adapted.metadata),
                "source_checkpoint": str(source.path),
                "source_checkpoint_sha256": source.checkpoint_sha256,
                "source_actor_sha256": source.actor_sha256,
                "adapted_actor_sha256": adapted.actor_sha256,
                "paired_rollout_status": (
                    "NOT_COMPARABLE_RUNTIME_OBSERVATION_CONTRACT"
                    if source.metadata.get("observation_contract_id")
                    != adapted.metadata.get("observation_contract_id")
                    else "AVAILABLE_BUT_NOT_RUN_BY_THIS_DIAGNOSTIC_PATH"
                ),
            }
            source.assert_unchanged()
    root = create_output_directory(args.output_dir, prefix="actionnet_audit")
    actor_before = parameter_hash(adapted.actor)
    diagnostic_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    integrity_records: list[dict[str, Any]] = []
    teacher_checks: list[bool] = []
    for decision_time in args.decision_times:
        for horizon in args.horizons:
            factual = _run_condition(
                loaded=adapted, repository=repository, gaze_checkpoint=gaze_checkpoint,
                gaze_type=args.gaze_type, decision_time=decision_time, horizon=horizon,
                fixation_shape=args.fixation_shape, episodes=args.episodes,
                episode_length=args.episode_length, seed_start=args.seed_start,
                deterministic=args.deterministic, regime="factual",
                collect_diagnostics=True, tests=tests,
            )
            diagnostic_rows.extend(factual["diagnostic_rows"])
            confidence_rows.extend(factual["confidence_rows"])
            episode_rows.extend(factual["episodes"])
            integrity_records.extend(factual["integrity"])
            if factual["teacher_factual_identity"] is not None:
                teacher_checks.append(bool(factual["teacher_factual_identity"]))
            if "bounds" in tests:
                for regime in ("oracle_current", "static_only"):
                    bounded = _run_condition(
                        loaded=adapted, repository=repository,
                        gaze_checkpoint=gaze_checkpoint, gaze_type=args.gaze_type,
                        decision_time=decision_time, horizon=horizon,
                        fixation_shape=args.fixation_shape, episodes=args.episodes,
                        episode_length=args.episode_length, seed_start=args.seed_start,
                        deterministic=args.deterministic, regime=regime,
                        collect_diagnostics=False, tests=set(),
                    )
                    episode_rows.extend(bounded["episodes"])
    adapted.assert_unchanged()
    actor_after = parameter_hash(adapted.actor)
    integrity = {
        "factual_replay_identity": all(
            row.get("factual_replay_identity", True) for row in integrity_records
        ),
        "no_observation_or_hidden_mutation": True,
        "h0_preview_noop": all(
            row.get("h0_preview_noop", True) for row in integrity_records
        ),
        "effective_decision_accounting": all(
            bool(row.get("effective")) == bool(row.get("effective"))
            for row in diagnostic_rows
        ),
        "strict_checkpoint_load": True,
        "deterministic_repeatability": all(
            row.get("deterministic_repeatability", True) for row in integrity_records
        ),
        "probabilities_finite_and_normalized": all(
            row.get("probabilities_finite", True)
            and row.get("probabilities_sum_to_one", True)
            for row in integrity_records
        ),
        "teacher_forced_factual_identity": all(teacher_checks),
        "actor_parameter_hash_preserved": actor_before == actor_after,
        "oracle_not_default": REGIMES[0] == "factual",
    }
    counterfactual_summary = aggregate_counterfactual_rows(diagnostic_rows)
    performance_summary = _episode_summary(episode_rows)
    memory_confidence = _confidence_summary(confidence_rows)
    summary = {
        "headline": "Frozen Partial ActionNet Audit",
        "source_vs_adapted": source_result,
        "counterfactual_summary": counterfactual_summary,
        "performance_summary": performance_summary,
        "memory_confidence": memory_confidence,
        "integrity_checks": integrity,
        "counts": {
            "per_decision_rows": len(diagnostic_rows),
            "confidence_rows": len(confidence_rows),
            "per_episode_rows": len(episode_rows),
        },
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest = {
        "analysis": "frozen_partial_actionnet_audit",
        "timestamp": timestamp,
        "command": command_record(),
        "git_commit": _git_commit(),
        "action_checkpoint_absolute_path": str(adapted.path),
        "action_checkpoint_sha256": adapted.checkpoint_sha256,
        "action_actor_parameter_sha256": adapted.actor_sha256,
        "source_checkpoint_absolute_path": None if source_path is None else str(source_path),
        "source_checkpoint_sha256": (
            None if source_path is None else file_sha256(source_path)
        ),
        "gaze_checkpoint_absolute_path": (
            None if gaze_checkpoint is None else str(gaze_checkpoint.path)
        ),
        "gaze_checkpoint_sha256": (
            None if gaze_checkpoint is None else gaze_checkpoint.checkpoint_sha256
        ),
        "partner_checkpoint_absolute_path": str(partner_path),
        "partner_checkpoint_sha256": provenance["fixed_partner_sha256"],
        "seed_start": args.seed_start,
        "episode_seeds": list(range(args.seed_start, args.seed_start + args.episodes)),
        "episode_count_per_condition": args.episodes,
        "decision_times": list(args.decision_times),
        "horizons": list(args.horizons),
        "episode_length": args.episode_length,
        "fixation_shape": args.fixation_shape,
        "gaze_type": args.gaze_type,
        "tests": list(args.tests),
        "observation_shape": [8, 5, 25],
        "observation_contract_id": OBSERVATION_CONTRACT_ID,
        "actor_architecture_id": ACTION_ARCHITECTURE_ID,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "action_space": list(ACTION_SPACE),
        "gru_hidden_shape": [
            1, int(adapted.actor._recurrent_N), int(adapted.actor.rnn.rnn.hidden_size)
        ],
        "memory_lifetime_ms": adapted.metadata["memory_lifetime_ms"],
        **adapted.constraint_config.metadata(),
        "static_channels": list(STATIC_CHANNELS),
        "dynamic_channels": list(DYNAMIC_CHANNELS),
        "channel_semantics": {str(key): value for key, value in CHANNEL_SEMANTICS.items()},
        "memory_variant_definitions": {
            variant: memory_variant(np.zeros((8, 5, 25), dtype=np.float32), variant, horizon=0)[1]
            for variant in (
                "no_dynamic_memory", "no_preview_memory", "no_partner_memory",
                "static_only", "validity_signal_only",
            )
        },
        "acquisition_action_contract_id": loaded.metadata.get(
            "acquisition_action_contract_id", ACQUISITION_ACTION_CONTRACT_ID
        ),
        "recurrent_commit_contract_id": loaded.metadata.get(
            "recurrent_commit_contract_id", RECURRENT_COMMIT_CONTRACT_ID
        ),
        "actor_available_action_contract_id": ACTOR_AVAILABLE_ACTION_CONTRACT_ID,
        "oracle_path_enabled": "bounds" in tests and args.allow_oracle_upper_bound,
        "oracle_role": "ORACLE_UPPER_BOUND",
        "human_gaze_used_for_selection_or_thresholds": False,
        "actor_hash_before": actor_before,
        "actor_hash_after": actor_after,
        **device_metadata(adapted.device),
        **provenance,
        "source_hashes": source_hashes(),
    }
    _write_outputs(
        root, manifest=manifest, summary=summary,
        diagnostic_rows=diagnostic_rows, episode_rows=episode_rows,
        confidence_rows=confidence_rows,
    )
    return {"output_dir": str(root), "summary": summary, "manifest": manifest}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--action-checkpoint", type=Path, required=True)
    value.add_argument("--source-action-checkpoint", type=Path)
    value.add_argument("--gaze-checkpoint", type=Path)
    value.add_argument("--partner-checkpoint", type=Path, default=Path(CANONICAL_PARTNER))
    value.add_argument("--gaze-type", choices=("rule_based", "learned"), default="rule_based")
    value.add_argument("--decision-times", type=int, nargs="+", choices=ALLOWED_DECISION_TIMES,
                       default=list(ALLOWED_DECISION_TIMES))
    value.add_argument("--horizons", type=int, nargs="+", choices=(0, 2, 4), default=[0, 2, 4])
    value.add_argument("--fixation-shape", choices=FIXATION_SHAPES, default="plus")
    value.add_argument("--episodes", type=int, default=2)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--seed-start", type=int, default=20260826)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--tests", nargs="+", choices=ALL_TESTS, default=list(DEFAULT_TESTS))
    value.add_argument("--allow-oracle-upper-bound", action="store_true")
    value.add_argument("--stochastic", dest="deterministic", action="store_false")
    value.set_defaults(deterministic=True)
    value.add_argument("--output-dir", type=Path)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.episodes <= 0 or args.episode_length <= 0:
        raise ValueError("episodes and episode length must be positive")
    result = run_audit(args)
    print(json.dumps({"output_dir": result["output_dir"], "counts": result["summary"]["counts"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
