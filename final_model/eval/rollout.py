"""One read-only rollout engine shared by every Full Action evaluator."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.common.partners import FixedHSPPartnerRepository, parameter_hash
from final_model.common.paths import file_sha256
from final_model.eval.checkpoint_loader import (
    LoadedActionCheckpoint,
    load_action_checkpoint_for_evaluation,
)
from final_model.eval.common import (
    CANONICAL_PARTNER,
    FINAL_ANALYSIS_SEED_START,
    array_sha256,
    command_record,
    create_output_directory,
    device_metadata,
    evaluation_seeds,
    mask_full_observation,
    source_hashes,
    write_csv,
    write_json,
)
from final_model.eval.counterfactual import CounterfactualCollector
from final_model.eval.task_events import COUNT_FIELDS, TaskEventLogger
from final_model.simple_ver.observation import FullObservationAdapter


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint: Path
    evaluation_horizon: int
    episodes: int = 10
    seed_start: int = FINAL_ANALYSIS_SEED_START
    episode_length: int = 400
    device: str = "cpu"
    preview_mode: str = "native"
    learner_action_mode: str = "deterministic"
    behavior_events: bool = False
    counterfactual: bool = False
    output_dir: Path | None = None
    allow_fixed_horizon_ablation: bool = False


@dataclass
class EvaluationResult:
    config: EvaluationConfig
    checkpoint: LoadedActionCheckpoint
    episode_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    behavior_rows: list[dict[str, Any]]
    events: list[dict[str, Any]]
    counterfactual_rows: list[dict[str, Any]]
    counterfactual_summary: dict[str, Any] | None
    manifest: dict[str, Any]
    output_dir: Path | None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deliveries = np.asarray([row["deliveries"] for row in rows], dtype=np.float64)
    sparse = np.asarray([row["sparse_team_reward"] for row in rows], dtype=np.float64)
    return {
        "episode_count": len(rows),
        "mean_deliveries": float(deliveries.mean()),
        "std_deliveries": float(deliveries.std()),
        "median_deliveries": float(np.median(deliveries)),
        "mean_sparse_team_reward": float(sparse.mean()),
        "std_sparse_team_reward": float(sparse.std()),
    }


def aggregate_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {field: sum(int(row[field]) for row in rows) for field in COUNT_FIELDS}
    placements = counts["learner_onion_counter_place"]
    pickups = counts["learner_onion_pickup"]
    interactions = counts["learner_interaction_count"] + counts["partner_interaction_count"]
    return {
        **counts,
        "episode_count": len(rows),
        "handoff_success_rate": (
            counts["partner_pickup_of_learner_placed_onion"] / placements
            if placements else 0.0
        ),
        "self_repickup_rate": (
            counts["learner_repickup_of_own_placed_onion"] / placements
            if placements else 0.0
        ),
        "learner_direct_pot_rate": counts["learner_onion_pot_insert"] / max(pickups, 1),
        "interaction_efficiency": counts["team_delivery"] / max(interactions, 1),
    }


def _state_hash(state) -> str:
    return hashlib.sha256(
        json.dumps(state.to_dict(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _manifest(
    *, config: EvaluationConfig, loaded: LoadedActionCheckpoint,
    seeds: tuple[int, ...], partner_path: Path, partner_hash_before: str,
    partner_hash_after: str, actor_after: str, critic_after: str,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_record(),
        "config": {
            **asdict(config),
            "checkpoint": str(Path(config.checkpoint).expanduser().resolve()),
            "output_dir": None if config.output_dir is None else str(Path(config.output_dir).resolve()),
        },
        "checkpoint_absolute_path": str(loaded.path),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_completed_step": loaded.completed_steps,
        "checkpoint_horizon_mode": loaded.horizon_mode,
        "checkpoint_horizons": list(loaded.horizons),
        "evaluation_horizon": int(config.evaluation_horizon),
        "preview_mode": config.preview_mode,
        "episode_count": int(config.episodes),
        "episode_seeds": list(seeds),
        "learner_action_mode": config.learner_action_mode,
        "partner_checkpoint_path": str(partner_path),
        "partner_checkpoint_sha256": file_sha256(partner_path),
        "actor_hash_before": loaded.actor_sha256,
        "actor_hash_after": actor_after,
        "critic_hash_before": loaded.critic_sha256,
        "critic_hash_after": critic_after,
        "partner_hash_before": partner_hash_before,
        "partner_hash_after": partner_hash_after,
        "source_hashes": source_hashes(),
        **device_metadata(loaded.device),
    }


def evaluate(
    config: EvaluationConfig, *, loaded: LoadedActionCheckpoint | None = None
) -> EvaluationResult:
    if config.learner_action_mode not in {"deterministic", "stochastic"}:
        raise ValueError("learner_action_mode must be deterministic or stochastic")
    if int(config.episode_length) <= 0:
        raise ValueError("episode_length must be positive")
    checkpoint = loaded or load_action_checkpoint_for_evaluation(
        config.checkpoint, device=config.device
    )
    horizon = checkpoint.validate_evaluation_horizon(
        config.evaluation_horizon,
        preview_ablation=config.allow_fixed_horizon_ablation,
    )
    seeds = evaluation_seeds(config.seed_start, config.episodes)
    partner_path = Path(CANONICAL_PARTNER).resolve()
    repository = FixedHSPPartnerRepository(
        device=checkpoint.device, checkpoint=partner_path
    )
    partner_actor = repository.actor(repository.manifest[0]["name"])
    partner_hash_before = parameter_hash(partner_actor)
    actor_before = parameter_hash(checkpoint.actor)
    critic_before = parameter_hash(checkpoint.critic)
    episode_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    counterfactual = CounterfactualCollector() if config.counterfactual else None

    for episode_index, seed in enumerate(seeds):
        backend = DirectOvercookedBackend(
            seed=seed,
            horizon=horizon,
            episode_length=config.episode_length,
            device=checkpoint.device,
            partner_repository=repository,
        )
        adapter = FullObservationAdapter(backend)
        logger = (
            TaskEventLogger(episode=episode_index, evaluation_horizon=horizon)
            if config.behavior_events else None
        )
        hidden = np.zeros(
            (1, int(checkpoint.actor._recurrent_N), int(checkpoint.actor.hidden_size)),
            dtype=np.float32,
        )
        mask = np.zeros((1, 1), dtype=np.float32)
        learner_rng = torch.Generator(device="cpu").manual_seed(seed + 701)
        sparse = shaped = 0.0
        deliveries = interactions = 0
        action_trace = []
        reward_trace = []
        recurrent_digest = hashlib.sha256()
        backend.verify_partner_parameters()
        for timestep in range(config.episode_length):
            observation, _critic_observation, available = adapter.prepare_decision()
            observation = mask_full_observation(
                observation,
                evaluation_horizon=horizon,
                preview_mode=config.preview_mode,
            )
            hidden_before = hidden.copy()
            mask_before = mask.copy()
            with torch.no_grad():
                probabilities_tensor, returned_hidden = checkpoint.actor.get_probs(
                    observation[None], hidden_before, mask_before, available[None]
                )
            probabilities = probabilities_tensor.detach().cpu().numpy()[0]
            if config.learner_action_mode == "deterministic":
                learner_action = int(probabilities.argmax())
            else:
                learner_action = int(
                    torch.multinomial(
                        torch.as_tensor(probabilities), 1, generator=learner_rng
                    ).item()
                )
            if counterfactual is not None:
                counterfactual.observe(
                    actor=checkpoint.actor,
                    observation=observation,
                    hidden=hidden_before,
                    mask=mask_before,
                    available=available,
                    native_probabilities=probabilities,
                    evaluation_horizon=horizon,
                    episode=episode_index,
                    timestep=timestep,
                    state=backend.state,
                    mdp=backend.mdp,
                )
            hidden = returned_hidden.detach().cpu().numpy()
            mask.fill(1.0)
            partner_observation, partner_available = backend.partner_batch_input()
            partner_hash = backend.begin_partner_step()
            partner_action = backend.partners.action(
                partner_observation, partner_available
            )
            pre_state = copy.deepcopy(backend.state)
            result = adapter.commit_decision_with_partner(
                learner_action, partner_action, partner_hash
            )
            if logger is not None:
                logger.observe(
                    timestep=timestep,
                    pre_state=pre_state,
                    post_state=backend.state,
                    actions=(partner_action, learner_action),
                    mdp=backend.mdp,
                    result=result,
                )
            sparse += float(result["sparse_team_reward"])
            shaped += float(result["native_shaped_reward"])
            deliveries += int(result["deliveries"])
            interactions += int(partner_action == 5) + int(learner_action == 5)
            action_trace.append((int(partner_action), int(learner_action)))
            reward_trace.append(float(result["sparse_team_reward"]))
            recurrent_digest.update(np.ascontiguousarray(hidden).tobytes())
            if result["done"]:
                break
        backend.verify_partner_parameters()
        if len(action_trace) != config.episode_length:
            raise RuntimeError("evaluation episode terminated at an unexpected length")
        row = {
            "checkpoint_path": str(checkpoint.path),
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "checkpoint_completed_step": checkpoint.completed_steps,
            "checkpoint_horizon_mode": checkpoint.horizon_mode,
            "evaluation_horizon": horizon,
            "preview_mode": config.preview_mode,
            "evaluation_seed": seed,
            "episode_index": episode_index,
            "episode_length": len(action_trace),
            "deliveries": deliveries,
            "sparse_team_reward": sparse,
            "native_shaped_reward": shaped,
            "total_interactions": interactions,
            "actor_hash": checkpoint.actor_sha256,
            "partner_hash": partner_hash_before,
            "joint_action_trajectory_sha256": hashlib.sha256(
                json.dumps(action_trace).encode("utf-8")
            ).hexdigest(),
            "reward_trajectory_sha256": hashlib.sha256(
                json.dumps(reward_trace).encode("utf-8")
            ).hexdigest(),
            "actor_recurrent_trajectory_sha256": recurrent_digest.hexdigest(),
            "terminal_state_sha256": _state_hash(backend.state),
        }
        episode_rows.append(row)
        if logger is not None:
            behavior_rows.append({"episode_index": episode_index, **logger.summary()})
            event_rows.extend(logger.events)

    checkpoint.assert_unchanged()
    actor_after = parameter_hash(checkpoint.actor)
    critic_after = parameter_hash(checkpoint.critic)
    partner_hash_after = parameter_hash(partner_actor)
    if actor_before != actor_after or critic_before != critic_after:
        raise RuntimeError("evaluation changed policy parameters")
    if partner_hash_before != partner_hash_after:
        raise RuntimeError("evaluation changed fixed partner parameters")
    summary = {
        **_summary(episode_rows),
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_completed_step": checkpoint.completed_steps,
        "checkpoint_horizon_mode": checkpoint.horizon_mode,
        "evaluation_horizon": horizon,
        "preview_mode": config.preview_mode,
        "learner_action_mode": config.learner_action_mode,
    }
    counterfactual_summary = None if counterfactual is None else counterfactual.summary()
    manifest = _manifest(
        config=config,
        loaded=checkpoint,
        seeds=seeds,
        partner_path=partner_path,
        partner_hash_before=partner_hash_before,
        partner_hash_after=partner_hash_after,
        actor_after=actor_after,
        critic_after=critic_after,
    )
    output_dir = None
    if config.output_dir is not None:
        output_dir = create_output_directory(config.output_dir, prefix="performance")
        write_csv(output_dir / "episodes.csv", episode_rows)
        write_json(output_dir / "summary.json", summary)
        write_csv(output_dir / "summary.csv", [summary])
        if behavior_rows:
            behavior_summary = aggregate_behavior(behavior_rows)
            write_csv(output_dir / "behavior_episodes.csv", behavior_rows)
            write_csv(output_dir / "behavior_summary.csv", [behavior_summary])
            write_json(output_dir / "behavior_summary.json", behavior_summary)
            with (output_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
                for event in event_rows:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
        if counterfactual is not None:
            write_csv(output_dir / "counterfactual.csv", counterfactual.rows)
            write_json(
                output_dir / "counterfactual_summary.json", counterfactual_summary
            )
        write_json(output_dir / "manifest.json", manifest)
    return EvaluationResult(
        config=config,
        checkpoint=checkpoint,
        episode_rows=episode_rows,
        summary=summary,
        behavior_rows=behavior_rows,
        events=event_rows,
        counterfactual_rows=[] if counterfactual is None else counterfactual.rows,
        counterfactual_summary=counterfactual_summary,
        manifest=manifest,
        output_dir=output_dir,
    )
