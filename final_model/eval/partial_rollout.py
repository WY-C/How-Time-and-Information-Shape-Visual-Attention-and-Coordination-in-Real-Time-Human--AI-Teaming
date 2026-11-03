"""Optimizer-free acquisition-driven Partial policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.common.partners import FixedHSPPartnerRepository, parameter_hash
from final_model.eval.checkpoint_loader import (
    LoadedPartialCheckpoint,
    load_partial_checkpoint_for_evaluation,
)
from final_model.eval.actionnet_diagnostics import instrument_memory_acquisitions
from final_model.eval.freshness_counterfactual import freshness_counterfactual
from final_model.eval.fixation_duration import (
    FIXATION_DURATION_CONTRACT_ID,
    fixation_duration_metrics,
    split_fixation_window,
)
from final_model.eval.common import (
    CANONICAL_PARTNER,
    create_output_directory,
    write_csv,
    write_json,
)
from final_model.partial_ver.gaze import (
    FIXATION_SHAPES,
    HUMAN_DWELL_RNG_SEED_OFFSET,
    acquisition_tiles,
)
from final_model.partial_ver.observation import ALLOWED_DECISION_TIMES, PartialObservationAdapter
from final_model.partial_ver.memory import MEMORY_RNG_SEED_OFFSET
from final_model.partial_ver.peripheral import PERIPHERAL_RNG_SEED_OFFSET
from final_model.partial_ver.preview_opacity import PREVIEW_OPACITY_RNG_SEED_OFFSET
from final_model.partial_ver.training import LearnedGazeController
from final_model.server_parallel.collision_observability import (
    human_compatible_joint_transition_collision,
)
from final_model.server_parallel.gaze_v1 import action_index_to_tile
from final_model.simple_ver.observation import preview4


AOI_NAMES = ("self", "partner", "preview", "h1", "h2", "h3", "h4", "near", "far")


def _aoi_masks(backend: DirectOvercookedBackend) -> dict[str, np.ndarray]:
    masks = {name: np.zeros((8, 5), dtype=bool) for name in AOI_NAMES}
    masks["self"][tuple(map(int, backend.state.players[1].position))] = True
    masks["partner"][tuple(map(int, backend.state.players[0].position))] = True
    displayed = preview4(backend.preview_planes(), backend.horizon) > 0.0
    for index, name in enumerate(("h1", "h2", "h3", "h4")):
        masks[name] = displayed[..., index]
    masks["near"] = masks["h1"] | masks["h2"]
    masks["far"] = masks["h3"] | masks["h4"]
    masks["preview"] = masks["near"] | masks["far"]
    return masks


def _aoi_event_fields(backend, fixation, shape: str) -> dict[str, Any]:
    target = tuple(map(int, fixation))
    footprint = acquisition_tiles(target, shape)
    masks = _aoi_masks(backend)
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        result[f"{name}_center"] = int(bool(mask[target]))
        result[f"{name}_footprint"] = int(
            any(bool(mask[tile]) for tile in footprint)
        )
    for definition in ("center", "footprint"):
        result[f"exclusive_{definition}_aoi"] = next(
            (
                name
                for name in ("preview", "self", "partner")
                if result[f"{name}_{definition}"]
            ),
            "other",
        )
    return result


def _enrich_fixation_segment(backend, shape, row):
    target = (int(row["fixation_x"]), int(row["fixation_y"]))
    return {
        **row,
        **_aoi_event_fields(backend, target, shape),
    }


def _aoi_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for definition in ("center", "footprint"):
        exclusive = f"exclusive_{definition}_aoi"
        for name in (*AOI_NAMES, "other"):
            if name == "other":
                hits = sum(row[exclusive] == name for row in rows)
                kind = "exclusive"
            else:
                hits = sum(int(row[f"{name}_{definition}"]) for row in rows)
                kind = "overlapping"
            result.append({
                "definition": definition,
                "aoi": name,
                "count": hits,
                "share": hits / max(len(rows), 1),
                "event_count": len(rows),
                "count_kind": kind,
            })
    return result


class EvaluationLearnedGaze:
    def __init__(self, actor, device, *, deterministic: bool, seed: int) -> None:
        self.actor, self.device = actor, device
        self.deterministic = bool(deterministic)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def reset(self):
        return None

    def set_event_time(self, _timestamp):
        return None

    def __call__(self, memory, fixation, context):
        spatial = LearnedGazeController.spatial_input(memory, fixation)
        with torch.no_grad():
            distribution = self.actor.distribution(
                torch.as_tensor(spatial[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(context[None], dtype=torch.float32, device=self.device),
            )
            action = (
                distribution.probs.argmax(-1)
                if self.deterministic
                else torch.multinomial(
                    distribution.probs.detach().cpu()[0], 1,
                    generator=self.generator,
                )
            )
        index = int(action.item())
        self.last_probabilities = (
            distribution.probs.detach().cpu().numpy()[0].copy()
        )
        self.last_action_index = index
        self.last_memory = np.asarray(memory, dtype=np.float32).copy()
        self.last_fixation = tuple(map(int, fixation))
        self.last_context = np.asarray(context, dtype=np.float32).copy()
        return action_index_to_tile(index)


@dataclass(frozen=True)
class PartialEvaluationConfig:
    checkpoint: Path
    decision_time: int
    horizon: int
    fixation_shape: str
    gaze_type: str
    episodes: int = 10
    episode_length: int = 400
    seed_start: int = 720_000
    device: str = "cpu"
    gaze_deterministic: bool = True
    action_deterministic: bool = True
    freshness_diagnostic: bool = False
    aoi_diagnostic: bool = False
    output_dir: Path | None = None


def _validate(config, loaded) -> None:
    if config.decision_time not in ALLOWED_DECISION_TIMES:
        raise ValueError("invalid Partial evaluation decision time")
    if config.horizon not in (0, 2, 4):
        raise ValueError("invalid Partial evaluation horizon")
    if config.fixation_shape not in FIXATION_SHAPES:
        raise ValueError("invalid Partial evaluation footprint")
    if config.gaze_type not in {"rule_based", "learned"}:
        raise ValueError("invalid Partial evaluation gaze type")
    metadata = loaded.metadata
    checks = {
        "decision_time": (config.decision_time, metadata.get("decision_times", [])),
        "horizon": (config.horizon, metadata.get("horizons", [])),
        "fixation_shape": (config.fixation_shape, metadata.get("fixation_shapes", [])),
    }
    mismatch = {key: values for key, values in checks.items() if values[0] not in values[1]}
    if mismatch:
        raise RuntimeError(f"Partial checkpoint does not cover evaluation condition: {mismatch}")
    if config.gaze_type == "learned" and loaded.gaze_actor is None:
        raise RuntimeError("learned-gaze evaluation requires a checkpoint with GazeNet")


def evaluate_partial(
    config: PartialEvaluationConfig,
    *, loaded: LoadedPartialCheckpoint | None = None,
) -> dict[str, Any]:
    checkpoint = loaded or load_partial_checkpoint_for_evaluation(
        config.checkpoint, device=config.device
    )
    _validate(config, checkpoint)
    actor_before, critic_before = checkpoint.actor_sha256, checkpoint.critic_sha256
    gaze_before = checkpoint.gaze_sha256
    partner_path = Path(CANONICAL_PARTNER).resolve()
    repository = FixedHSPPartnerRepository(
        device=checkpoint.device, checkpoint=partner_path
    )
    partner_provenance = repository.provenance()
    if checkpoint.metadata.get("fixed_partner_sha256") != partner_provenance[
        "fixed_partner_sha256"
    ]:
        raise RuntimeError("Partial evaluation fixed-partner checkpoint mismatch")
    episode_rows = []
    counterfactual_rows = []
    aoi_rows: list[dict[str, Any]] = []
    fixation_rows: list[dict[str, Any]] = []
    decision_gaze_rows: list[dict[str, Any]] = []
    total_metrics: dict[str, float] = {}
    rule_branches = {name: 0 for name in ("task", "stay", "stale", "random")}
    requested_targets = {f"{x},{y}": 0 for x in range(8) for y in range(5)}
    landing_targets = dict(requested_targets)
    with torch.no_grad():
        for episode in range(config.episodes):
            episode_fixation_start = len(fixation_rows)
            backend = DirectOvercookedBackend(
                seed=config.seed_start + episode, horizon=config.horizon,
                episode_length=config.episode_length, device=checkpoint.device,
                partner_repository=repository,
            )
            gaze_policy = None
            if config.gaze_type == "learned":
                gaze_policy = EvaluationLearnedGaze(
                    checkpoint.gaze_actor, checkpoint.device,
                    deterministic=config.gaze_deterministic,
                    seed=config.seed_start + episode + 77_003,
                )
            adapter = PartialObservationAdapter(
                backend, decision_time=config.decision_time,
                fixation_shape=config.fixation_shape, gaze_policy=gaze_policy,
                rule_gaze_seed=config.seed_start + episode + 99_001,
                rule_gaze_probabilities=checkpoint.metadata.get("rule_gaze_probabilities"),
                memory_lifetime_ms=checkpoint.metadata["memory_lifetime_ms"],
                memory_decay_mode=checkpoint.metadata["memory_decay_mode"],
                memory_tau_ms=checkpoint.metadata.get("memory_tau_ms"),
                memory_seed=(
                    config.seed_start + episode + MEMORY_RNG_SEED_OFFSET
                    if checkpoint.metadata["memory_decay_mode"]
                    == "stochastic_exponential"
                    else None
                ),
                constraint_config=checkpoint.constraint_config,
                peripheral_seed=(
                    config.seed_start + episode + PERIPHERAL_RNG_SEED_OFFSET
                ),
                opacity_seed=(
                    config.seed_start + episode + PREVIEW_OPACITY_RNG_SEED_OFFSET
                ),
                dwell_seed=(
                    config.seed_start + episode + HUMAN_DWELL_RNG_SEED_OFFSET
                    if checkpoint.constraint_config.emma.mode == "human_stochastic"
                    else None
                ),
                ground_truth_policy_input=bool(
                    checkpoint.metadata.get("ground_truth_policy_input", False)
                ),
            )
            if config.aoi_diagnostic:
                def record_aoi(event):
                    fixation = action_index_to_tile(int(event["fixation_index"]))
                    aoi_rows.append({
                        "T": config.decision_time,
                        "H": config.horizon,
                        "episode": episode,
                        "seed": config.seed_start + episode,
                        "decision": decision,
                        "acquisition_index": int(event["event_index"]),
                        "timestamp_ms": float(event["timestamp_ms"]),
                        "fixation_x": fixation[0],
                        "fixation_y": fixation[1],
                        **_aoi_event_fields(
                            backend, fixation, config.fixation_shape
                        ),
                    })

                instrument_memory_acquisitions(adapter, on_event=record_aoi)
            gaze_phase_events = []
            original_advance_event = adapter.gaze.advance_event

            def recording_advance_event():
                fixation_before = tuple(adapter.gaze.current_fixation)
                event = original_advance_event()
                gaze_phase_events.append({
                    "timestamp_ms": float(event.timestamp_ms),
                    "completed_phase": event.completed_phase.value,
                    "fixation_before": fixation_before,
                })
                return event

            adapter.gaze.advance_event = recording_advance_event
            phase_event_cursor = 0
            hidden = np.zeros((1, 64), dtype=np.float32)
            mask = np.zeros((1,), dtype=np.float32)
            sparse = 0.0
            deliveries = 0
            collision_count = 0
            actions = []
            backend.verify_partner_parameters()
            action_generator = torch.Generator(device="cpu").manual_seed(
                config.seed_start + episode + 88_007
            )
            for decision in range(config.episode_length):
                decision_onset = float(adapter.current_time_ms)
                segment_start = decision_onset
                pre_hidden = hidden.copy()
                pre_mask = mask.copy()

                def proposal(observation, _critic, available, _timestamp):
                    probabilities, candidate = checkpoint.actor.get_probs(
                        observation[None], pre_hidden[None], pre_mask[None], available[None]
                    )
                    action = (
                        probabilities.argmax(-1)
                        if config.action_deterministic
                        else torch.multinomial(
                            probabilities.detach().cpu()[0], 1,
                            generator=action_generator,
                        )
                    )
                    native = probabilities.detach().cpu().numpy()[0]
                    if config.freshness_diagnostic:
                        rows = freshness_counterfactual(
                            actor=checkpoint.actor, observation=observation,
                            hidden=pre_hidden[None], mask=pre_mask[None],
                            available=available, native_probabilities=native,
                        )
                        counterfactual_rows.extend(
                            {"episode": episode, "decision": decision, **row} for row in rows
                        )
                    return {
                        "action": int(action.item()),
                        "candidate_hidden": candidate.detach().cpu().numpy()[0],
                        "probabilities": native,
                    }

                collected = adapter.collect_decision(proposal)
                if config.aoi_diagnostic:
                    decision_segments = [
                        _enrich_fixation_segment(
                            backend, config.fixation_shape, row
                        )
                        for row in split_fixation_window(
                            start_ms=segment_start,
                            end_ms=float(collected["deadline_ms"]),
                            phase_events=gaze_phase_events[phase_event_cursor:],
                            final_phase=adapter.gaze.phase.value,
                            final_fixation=adapter.gaze.current_fixation,
                        )
                    ]
                    phase_event_cursor = len(gaze_phase_events)
                    for row in decision_segments:
                        row.update({
                            "T": config.decision_time,
                            "H": config.horizon,
                            "episode": episode,
                            "seed": config.seed_start + episode,
                            "decision": decision,
                        })
                    fixation_rows.extend(decision_segments)
                    decision_gaze_rows.append({
                        "T": config.decision_time,
                        "H": config.horizon,
                        "episode": episode,
                        "seed": config.seed_start + episode,
                        "decision": decision,
                        **fixation_duration_metrics(decision_segments),
                    })
                    covered = sum(row["duration_ms"] for row in decision_segments)
                    if not np.isclose(covered, config.decision_time, atol=1e-8):
                        raise RuntimeError(
                            "fixation segments do not cover the decision window"
                        )
                latest = collected["latest_proposal"]
                learner_action = 4 if latest is None else int(latest["action"])
                partner_observation, partner_available = backend.partner_batch_input()
                partner_hash_before = backend.begin_partner_step()
                partner_action = backend.partners.action(
                    partner_observation, partner_available
                )
                collision_count += int(
                    human_compatible_joint_transition_collision(
                        backend.mdp,
                        backend.state,
                        (
                            backend.Action.INDEX_TO_ACTION[int(partner_action)],
                            backend.Action.INDEX_TO_ACTION[learner_action],
                        ),
                    )
                )
                result = adapter.commit_collected(
                    collected,
                    partner_action=partner_action,
                    partner_hash_before=partner_hash_before,
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
            for name, count in metrics.get("rule_branch_counts", {}).items():
                rule_branches[name] += int(count)
            for name, count in metrics["requested_target_distribution"].items():
                requested_targets[name] += int(count)
            for name, count in metrics["landing_target_distribution"].items():
                landing_targets[name] += int(count)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    total_metrics[key] = total_metrics.get(key, 0.0) + float(value)
            episode_segments = fixation_rows[episode_fixation_start:]
            episode_gaze = (
                fixation_duration_metrics(episode_segments)
                if config.aoi_diagnostic else {}
            )
            episode_rows.append({
                "episode": episode, "seed": config.seed_start + episode,
                "deliveries": deliveries, "sparse_team_reward": sparse,
                "decisions": len(actions),
                "collision_count": collision_count,
                "collision_rate": collision_count / max(len(actions), 1),
                "action_trajectory_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
                "actor_active_sample_rate": metrics["executed_proposals"] / max(metrics["task_decisions"], 1),
                **episode_gaze,
                **metrics,
            })
    checkpoint.assert_unchanged()
    if parameter_hash(checkpoint.actor) != actor_before or parameter_hash(checkpoint.critic) != critic_before:
        raise RuntimeError("Partial evaluation changed Action policy parameters")
    if checkpoint.gaze_actor is not None and checkpoint.gaze_sha256 != gaze_before:
        raise RuntimeError("Partial evaluation changed gaze policy parameters")
    episodes = max(config.episodes, 1)
    total_collisions = sum(row["collision_count"] for row in episode_rows)
    total_decisions = sum(row["decisions"] for row in episode_rows)
    aggregate_gaze = (
        fixation_duration_metrics(fixation_rows)
        if config.aoi_diagnostic else {}
    )
    summary = {
        "episode_count": config.episodes,
        "mean_deliveries": float(np.mean([row["deliveries"] for row in episode_rows])),
        "mean_sparse_team_reward": float(np.mean([row["sparse_team_reward"] for row in episode_rows])),
        "collision_count": total_collisions,
        "collision_rate": total_collisions / max(total_decisions, 1),
        "collision_contract_id": "human_compatible_joint_transition_collision_v1",
        "decision_time": config.decision_time, "horizon": config.horizon,
        "fixation_shape": config.fixation_shape, "gaze_type": config.gaze_type,
        **{key: value / episodes for key, value in total_metrics.items()},
        "acquisitions_per_decision": total_metrics.get("acquisitions", 0.0)
        / max(total_metrics.get("task_decisions", 0.0), 1.0),
        "no_acquisition_rate": total_metrics.get("no_acquisition_decisions", 0.0)
        / max(total_metrics.get("task_decisions", 0.0), 1.0),
        "rule_branch_distribution": rule_branches,
        "requested_target_distribution": requested_targets,
        "landing_target_distribution": landing_targets,
        "counterfactual_query_count": len(counterfactual_rows),
        "aoi_acquisition_count": len(aoi_rows),
        "aoi_fixation_duration_ms": sum(
            row["duration_ms"]
            for row in fixation_rows
            if row["included_in_fixation_duration"]
        ),
        "aoi_saccade_duration_ms": sum(
            row["duration_ms"]
            for row in fixation_rows
            if not row["included_in_fixation_duration"]
        ),
        "aoi_window_duration_ms": sum(
            row["duration_ms"] for row in fixation_rows
        ),
        **aggregate_gaze,
        "optimizer_free": True, "parameter_hash_preserved": True,
        "fixed_partner_identity": partner_provenance["fixed_partner_identity"],
        "fixed_partner_sha256": partner_provenance["fixed_partner_sha256"],
    }
    manifest = {
        "analysis": "canonical_partial_acquisition_driven_evaluation",
        "checkpoint_absolute_path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_completed_step": checkpoint.completed_steps,
        "decision_time": config.decision_time, "horizon": config.horizon,
        "fixation_shape": config.fixation_shape, "gaze_type": config.gaze_type,
        "episodes": config.episodes, "episode_length": config.episode_length,
        "seed_start": config.seed_start,
        "gaze_selection": (
            "deterministic_argmax"
            if config.gaze_deterministic else "stochastic_categorical"
        ),
        "task_action_selection": (
            "deterministic_argmax"
            if config.action_deterministic else "stochastic_categorical"
        ),
        "freshness_diagnostic": config.freshness_diagnostic,
        "aoi_diagnostic": config.aoi_diagnostic,
        "fixation_duration_contract_id": FIXATION_DURATION_CONTRACT_ID,
        "fixation_aoi_definition": "mutually_exclusive_central_tile",
        "fixation_aoi_priority": ["preview", "self", "teammate", "other"],
        "zero_classified_fixation_convention": "all_fixation_proportions_zero",
        "actor_hash_before": actor_before,
        "actor_hash_after": parameter_hash(checkpoint.actor),
        "critic_hash_before": critic_before,
        "critic_hash_after": parameter_hash(checkpoint.critic),
        "gaze_hash_before": gaze_before,
        "gaze_hash_after": (
            None if checkpoint.gaze_actor is None
            else checkpoint.gaze_sha256
        ),
        "optimizer_free": True,
        **checkpoint.constraint_config.metadata(),
        **partner_provenance,
    }
    output_dir = None
    if config.output_dir is not None:
        output_dir = create_output_directory(config.output_dir, prefix="partial")
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "manifest.json", manifest)
        write_csv(output_dir / "episodes.csv", episode_rows)
        if counterfactual_rows:
            write_csv(output_dir / "freshness_counterfactual.csv", counterfactual_rows)
        if aoi_rows:
            write_csv(output_dir / "aoi_acquisitions.csv", aoi_rows)
            write_csv(output_dir / "aoi_summary.csv", _aoi_summary(aoi_rows))
        if fixation_rows:
            write_csv(output_dir / "fixation_segments.csv", fixation_rows)
            write_csv(
                output_dir / "fixation_duration_by_decision.csv",
                decision_gaze_rows,
            )
    return {
        "summary": summary, "episodes": episode_rows,
        "freshness_counterfactual": counterfactual_rows,
        "aoi_acquisitions": aoi_rows,
        "aoi_summary": _aoi_summary(aoi_rows),
        "fixation_segments": fixation_rows,
        "fixation_duration_by_decision": decision_gaze_rows,
        "manifest": manifest,
        "output_dir": output_dir,
    }
