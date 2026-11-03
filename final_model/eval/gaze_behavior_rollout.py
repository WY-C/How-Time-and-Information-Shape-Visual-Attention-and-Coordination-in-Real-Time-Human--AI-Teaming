"""Canonical runtime delegation and rich event recording for gaze evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.common.partners import FixedHSPPartnerRepository, parameter_hash
from final_model.eval.actionnet_diagnostics import (
    instrument_memory_acquisitions,
    read_only_probabilities,
    read_only_probabilities_and_hidden,
)
from final_model.eval.common import CANONICAL_PARTNER
from final_model.eval.gaze_behavior_artifact import (
    atomic_output_path,
    begin_rollout_artifact,
    finalize_rollout_artifact,
)
from final_model.eval.gaze_behavior_metrics import (
    AOIS,
    CONTENT_CHANNELS,
    classify_memory_change,
    decision_relative_timestamp,
    enrichment_values,
    exclusive_aoi,
    manhattan_distance,
    probability_metrics,
    proximity_bin,
    target_hit_mask,
    write_csv,
)
from final_model.eval.partial_rollout import EvaluationLearnedGaze
from final_model.partial_ver.gaze import (
    HUMAN_DWELL_RNG_SEED_OFFSET,
    ReferenceGeometry,
    acquisition_tiles,
)
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.memory import MEMORY_RNG_SEED_OFFSET
from final_model.partial_ver.peripheral import PERIPHERAL_RNG_SEED_OFFSET
from final_model.partial_ver.preview_opacity import PREVIEW_OPACITY_RNG_SEED_OFFSET
from final_model.server_parallel.gaze_v1 import (
    GAZE_ACTION_COUNT,
    GAZE_TILES,
    action_index_to_tile,
    tile_to_action_index,
)
from final_model.server_parallel.collision_observability import (
    human_compatible_joint_transition_collision,
)
from final_model.simple_ver.observation import preview4


@dataclass(frozen=True)
class RolloutConfig:
    model_name: str
    decision_times: tuple[int, ...]
    horizons: tuple[int, ...]
    fixation_shape: str
    episodes: int
    episode_length: int
    seed_start: int
    gaze_mode: str
    action_mode: str
    aoi_precedence: tuple[str, ...]


class RecordingGazePolicy:
    """Observe a canonical EvaluationLearnedGaze call after it has selected."""

    def __init__(self, delegate: EvaluationLearnedGaze, callback: Callable) -> None:
        self.delegate = delegate
        self.callback = callback

    def reset(self):
        return self.delegate.reset()

    def set_event_time(self, timestamp):
        return self.delegate.set_event_time(timestamp)

    def __call__(self, memory, fixation, context):
        target = self.delegate(memory, fixation, context)
        self.callback(self.delegate, target)
        return target


def _held_name(player) -> str | None:
    value = getattr(player, "held_object", None)
    return None if value is None else str(value.name)


def _actor_observation_before_acquisition(
    event: Mapping[str, Any] | None,
    observation: np.ndarray,
    *,
    ground_truth_policy_input: bool,
    actor_query_time_ms: float,
) -> np.ndarray:
    if event is None:
        raise RuntimeError("acquisition instrumentation lost canonical observation")
    if ground_truth_policy_input:
        # Gaze acquisitions update partial memory, but Full ActionNet input is
        # the gaze-independent ground-truth state.  Its same-state
        # counterfactual is therefore exactly the factual observation.
        return observation
    memory_before = event.get("_memory_before")
    memory_after = event.get("_memory_after")
    if memory_before is not None and memory_after is not None:
        expected_after = memory_after.snapshot(actor_query_time_ms)
        before_observation = memory_before.snapshot(actor_query_time_ms)
    else:
        expected_after = event["after"]
        before_observation = event["before"]
    if not np.array_equal(expected_after, observation):
        raise RuntimeError("acquisition instrumentation lost canonical observation")
    return before_observation


def ground_truth_masks(backend: DirectOvercookedBackend) -> dict[str, np.ndarray]:
    masks = {name: np.zeros((8, 5), dtype=bool) for name in AOIS}
    self_tile = tuple(map(int, backend.state.players[1].position))
    partner_tile = tuple(map(int, backend.state.players[0].position))
    masks["self"][self_tile] = True
    masks["partner"][partner_tile] = True
    displayed = preview4(backend.preview_planes(), backend.horizon) > 0.0
    for index, name in enumerate(("h1", "h2", "h3", "h4")):
        masks[name] = displayed[..., index]
    masks["near"] = masks["h1"] | masks["h2"]
    masks["far"] = masks["h3"] | masks["h4"]
    masks["preview"] = masks["near"] | masks["far"]
    if backend.horizon == 0 and any(np.any(masks[name]) for name in ("preview", "h1", "h2", "h3", "h4", "near", "far")):
        raise RuntimeError("H0 preview annotation leakage")
    return masks


def _state_annotations(backend: DirectOvercookedBackend) -> dict[str, Any]:
    self_tile = tuple(map(int, backend.state.players[1].position))
    partner_tile = tuple(map(int, backend.state.players[0].position))
    distance = manhattan_distance(self_tile, partner_tile)
    self_held = _held_name(backend.state.players[1])
    partner_held = _held_name(backend.state.players[0])
    canonical = backend.canonical_pair()[1]
    pot_tiles = canonical[..., 10] > 0
    pot_values = canonical[..., 15][pot_tiles] / 255.0
    return {
        "self_tile_x": self_tile[0], "self_tile_y": self_tile[1],
        "partner_tile_x": partner_tile[0], "partner_tile_y": partner_tile[1],
        "partner_manhattan_distance": distance,
        "partner_distance_bin": proximity_bin(distance),
        "partner_adjacent": int(distance == 1),
        "interaction_relevant_proximity": int(distance <= 2),
        "self_holding": self_held, "partner_holding": partner_held,
        "self_holding_any": int(self_held is not None),
        "partner_holding_any": int(partner_held is not None),
        "pot_onion_count_total": float(pot_values.sum()),
        "onion_resource_tile_count": int(np.count_nonzero(canonical[..., 19])),
        "dish_resource_tile_count": int(np.count_nonzero(canonical[..., 18])),
        "serve_resource_tile_count": int(np.count_nonzero(canonical[..., 14])),
        "partner_distance_metric": "tile_manhattan",
    }


def _aoi_fields(
    masks: Mapping[str, np.ndarray], target: Sequence[int], shape: str
) -> dict[str, int]:
    landing = tuple(map(int, target))
    footprint = acquisition_tiles(landing, shape)
    result = {}
    for name, mask in masks.items():
        result[f"{name}_center"] = int(bool(mask[landing]))
        result[f"{name}_footprint"] = int(any(bool(mask[tile]) for tile in footprint))
    return result


def _acquired_aoi_fields(
    masks: Mapping[str, np.ndarray],
    foveal_tiles: Sequence[Sequence[int]],
    peripheral_tiles: Sequence[Sequence[int]],
    opacity_successes: Sequence[tuple[Sequence[int], int]] = (),
    *, opacity_actual: bool = False,
) -> dict[str, int]:
    foveal = tuple(tuple(map(int, tile)) for tile in foveal_tiles)
    peripheral = tuple(tuple(map(int, tile)) for tile in peripheral_tiles)
    detected_preview = tuple(
        (tuple(map(int, tile)), int(step)) for tile, step in opacity_successes
    )
    result = {}
    for name, mask in masks.items():
        foveal_hit = any(bool(mask[tile]) for tile in foveal)
        if opacity_actual and name in {"preview", "h1", "h2", "h3", "h4"}:
            required_step = None if name == "preview" else int(name[1])
            peripheral_hit = any(
                bool(mask[tile])
                and (required_step is None or step == required_step)
                for tile, step in detected_preview
            )
        else:
            peripheral_hit = any(bool(mask[tile]) for tile in peripheral)
        result[f"{name}_foveal_acquired"] = int(foveal_hit)
        result[f"{name}_peripheral_acquired"] = int(peripheral_hit)
        result[f"{name}_total_acquired"] = int(foveal_hit or peripheral_hit)
    return result


def _request_enrichment_fields(
    masks: Mapping[str, np.ndarray], probabilities: np.ndarray,
    selected: int, shape: str,
) -> dict[str, Any]:
    result = {}
    for name, mask in masks.items():
        for definition, footprint in (("center", False), ("footprint", True)):
            prefix = f"{name}_{definition}"
            values = enrichment_values(
                probabilities,
                target_hit_mask(mask, shape=shape, footprint=footprint),
                selected,
            )
            result.update({f"{prefix}_{key}": value for key, value in values.items()})
            result[f"{prefix}_compatible_target_count"] = int(
                round(values["chance_mass"] * GAZE_ACTION_COUNT)
            )
    return result


def _candidate_tiles(mask: np.ndarray, *, shape: str, footprint: bool) -> list[list[int]]:
    hits = target_hit_mask(mask, shape=shape, footprint=footprint)
    return [list(GAZE_TILES[index]) for index in np.flatnonzero(hits)]


def _memory_fields(
    before: np.ndarray, after: np.ndarray, tiles: Sequence[Sequence[int]],
    *, saccade_distance: float,
) -> dict[str, Any]:
    footprint = tuple(tuple(map(int, tile)) for tile in tiles)
    pre_fresh = np.asarray(before[..., 20], dtype=np.float64) / 255.0
    post_fresh = np.asarray(after[..., 20], dtype=np.float64) / 255.0
    selected_pre = np.asarray([pre_fresh[tile] for tile in footprint])
    selected_post = np.asarray([post_fresh[tile] for tile in footprint])
    pre_valid, post_valid = pre_fresh > 0.0, post_fresh > 0.0
    content = classify_memory_change(before, after)
    gain = float(post_valid.mean() - pre_valid.mean())
    novelty = int(content["content_novelty_count"])
    return {
        "pre_valid_fraction": float(pre_valid.mean()),
        "post_valid_fraction": float(post_valid.mean()),
        "pre_mean_freshness": float(pre_fresh.mean()),
        "post_mean_freshness": float(post_fresh.mean()),
        "pre_target_mean_freshness": float(selected_pre.mean()),
        "pre_target_min_freshness": float(selected_pre.min()),
        "pre_target_valid_fraction": float(np.mean(selected_pre > 0.0)),
        "pre_target_stale_fraction": float(np.mean(selected_pre <= 0.0)),
        "footprint_tiles_refreshed": len(footprint),
        "newly_valid_count": int(np.count_nonzero(~pre_valid & post_valid)),
        "expired_count": int(np.count_nonzero(pre_valid & ~post_valid)),
        "marginal_valid_memory_gain": gain,
        "information_gain_per_saccade_degree": None if saccade_distance == 0 else gain / saccade_distance,
        "content_novelty_per_saccade_degree": None if saccade_distance == 0 else novelty / saccade_distance,
        **content,
    }


def _source_memory_fields(
    before: np.ndarray, after: np.ndarray,
    foveal_tiles: Sequence[Sequence[int]],
    peripheral_tiles: Sequence[Sequence[int]],
) -> dict[str, Any]:
    pre_fresh = np.asarray(before[..., 20], dtype=np.float64) / 255.0
    post_fresh = np.asarray(after[..., 20], dtype=np.float64) / 255.0

    def fields(label, tiles):
        selected = tuple(tuple(map(int, tile)) for tile in tiles)
        if not selected:
            return {
                f"{label}_newly_valid_count": 0,
                f"{label}_freshness_gain": 0.0,
                f"{label}_content_changed_tile_count": 0,
                f"{label}_information_changed": 0,
            }
        newly_valid = sum(
            pre_fresh[tile] <= 0.0 < post_fresh[tile] for tile in selected
        )
        freshness_gain = sum(
            max(float(post_fresh[tile] - pre_fresh[tile]), 0.0)
            for tile in selected
        )
        content_changed = sum(
            not np.array_equal(before[tile][list(CONTENT_CHANNELS)],
                               after[tile][list(CONTENT_CHANNELS)])
            for tile in selected
        )
        return {
            f"{label}_newly_valid_count": int(newly_valid),
            f"{label}_freshness_gain": float(freshness_gain),
            f"{label}_content_changed_tile_count": int(content_changed),
            f"{label}_information_changed": int(
                content_changed > 0 or freshness_gain > 0.0
            ),
        }

    result = {
        **fields("foveal", foveal_tiles),
        **fields("peripheral", peripheral_tiles),
    }
    result["acquisition_information_source"] = (
        "peripheral_success_change"
        if result["peripheral_information_changed"]
        else "foveal_only_change"
        if result["foveal_information_changed"]
        else "no_effective_change"
    )
    return result


def _write_gzip_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(path) as temporary:
        if not rows:
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                handle.write("")
            return
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)


def run_model_rollouts(
    *, loaded, config: RolloutConfig, model_dir: Path,
    reporter=None, model_index: int = 1, model_count: int = 1,
) -> dict[str, Any]:
    """Run each model/condition/seed exactly once using the existing adapter."""
    model_dir.mkdir(parents=True, exist_ok=True)
    begin_rollout_artifact(model_dir)
    repository = FixedHSPPartnerRepository(
        device=loaded.device, checkpoint=Path(CANONICAL_PARTNER).resolve()
    )
    provenance = repository.provenance()
    if loaded.metadata.get("fixed_partner_sha256") != provenance["fixed_partner_sha256"]:
        raise RuntimeError("Partial evaluation fixed-partner checkpoint mismatch")
    actor_before, critic_before, gaze_before = (
        loaded.actor_sha256, loaded.critic_sha256, loaded.gaze_sha256
    )
    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    gaze_memories: list[np.ndarray] = []
    gaze_fixations: list[tuple[int, int]] = []
    gaze_contexts: list[np.ndarray] = []
    geometry = ReferenceGeometry()
    model_started = (
        None
        if reporter is None
        else reporter.start_model(model_index, model_count, config.model_name)
    )
    conditions = [
        (decision_time, horizon)
        for decision_time in config.decision_times
        for horizon in config.horizons
    ]

    for condition_index, (decision_time, horizon) in enumerate(conditions, 1):
            condition_episode_start = len(episode_rows)
            condition_started = (
                None
                if reporter is None
                else reporter.start_condition(
                    index=condition_index, total=len(conditions),
                    decision_time=decision_time, horizon=horizon,
                    episodes=config.episodes,
                    episode_length=config.episode_length,
                )
            )
            for episode in range(config.episodes):
                if reporter is not None:
                    reporter.context["episode"] = (
                        f"{episode + 1}/{config.episodes}"
                    )
                seed = config.seed_start + episode
                backend = DirectOvercookedBackend(
                    seed=seed, horizon=horizon, episode_length=config.episode_length,
                    device=loaded.device, partner_repository=repository,
                )
                if loaded.gaze_actor is None:
                    raise RuntimeError("gaze behavior suite requires a compatible GazeNet")
                gaze = EvaluationLearnedGaze(
                    loaded.gaze_actor, loaded.device,
                    deterministic=config.gaze_mode == "argmax", seed=seed + 77_003,
                )
                runtime: dict[str, Any] = {
                    "decision": 0, "decision_start_ms": 0.0,
                    "decision_request_index": 0,
                }

                def on_gaze_request(policy, target):
                    probabilities = np.asarray(policy.last_probabilities, dtype=np.float64)
                    if probabilities.shape != (40,) or not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
                        raise RuntimeError("invalid GazeNet probability vector")
                    masks = ground_truth_masks(backend)
                    index = int(policy.last_action_index)
                    source = tuple(policy.last_fixation)
                    distance = geometry.visual_angle_degrees(source, target)
                    relative = decision_relative_timestamp(
                        adapter.current_time_ms, runtime["decision_start_ms"]
                    )
                    row = {
                        "request_row_id": len(request_rows),
                        "model_name": config.model_name,
                        "checkpoint_step": loaded.completed_steps,
                        "checkpoint_sha256": loaded.checkpoint_sha256,
                        "T": decision_time, "H": horizon, "episode": episode,
                        "seed": seed, "decision": runtime["decision"],
                        "request_index": runtime["decision_request_index"],
                        "absolute_timestamp_ms": adapter.current_time_ms,
                        "decision_start_ms": runtime["decision_start_ms"],
                        "decision_relative_timestamp_ms": relative,
                        "remaining_time_ms": decision_time - relative,
                        "current_fixation_x": source[0], "current_fixation_y": source[1],
                        "requested_target_x": target[0], "requested_target_y": target[1],
                        "same_target": int(tuple(source) == tuple(target)),
                        "new_target": int(tuple(source) != tuple(target)),
                        "saccade_distance_degrees": distance,
                        "gaze_action_index": index,
                        "gaze_entropy": float(-np.sum(probabilities * np.log(np.clip(probabilities, np.finfo(float).tiny, 1)))),
                        **_state_annotations(backend),
                        **_request_enrichment_fields(masks, probabilities, index, config.fixation_shape),
                        **{f"p{action}": float(probabilities[action]) for action in range(40)},
                    }
                    request_rows.append(row)
                    gaze_memories.append(policy.last_memory.copy())
                    gaze_fixations.append(source)
                    gaze_contexts.append(policy.last_context.copy())
                    runtime["decision_request_index"] += 1

                recording_policy = RecordingGazePolicy(gaze, on_gaze_request)
                adapter = PartialObservationAdapter(
                    backend, decision_time=decision_time,
                    fixation_shape=config.fixation_shape,
                    gaze_policy=recording_policy,
                    rule_gaze_seed=seed + 99_001,
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
                    ground_truth_policy_input=bool(
                        loaded.metadata.get("ground_truth_policy_input", False)
                    ),
                )
                memory_capture = instrument_memory_acquisitions(adapter)
                hidden = np.zeros((1, 64), dtype=np.float32)
                mask = np.zeros((1,), dtype=np.float32)
                action_generator = torch.Generator(device="cpu").manual_seed(seed + 88_007)
                sparse = 0.0
                deliveries = 0
                collision_count = 0
                actions: list[int] = []
                backend.verify_partner_parameters()
                for decision in range(config.episode_length):
                    runtime.update({
                        "decision": decision,
                        "decision_start_ms": float(adapter.current_time_ms),
                        "decision_request_index": 0,
                    })
                    start_fixation = tuple(adapter.gaze.current_fixation)
                    start_masks = ground_truth_masks(backend)
                    pre_hidden, pre_mask = hidden.copy(), mask.copy()
                    acquisition_start = len(acquisition_rows)
                    previous_sampled = previous_top1 = None

                    def proposal(observation, _critic, available, timestamp):
                        nonlocal previous_sampled, previous_top1
                        event = memory_capture.get("event")
                        before_observation = _actor_observation_before_acquisition(
                            event,
                            observation,
                            ground_truth_policy_input=adapter.ground_truth_policy_input,
                            actor_query_time_ms=float(timestamp),
                        )
                        probabilities, candidate = read_only_probabilities_and_hidden(
                            loaded.actor, observation, pre_hidden[None],
                            pre_mask[None], available,
                        )
                        sampled = int(
                            probabilities.argmax()
                            if config.action_mode == "argmax"
                            else torch.multinomial(
                                torch.as_tensor(probabilities), 1,
                                generator=action_generator,
                            ).item()
                        )
                        before_probability = read_only_probabilities(
                            loaded.actor, before_observation, pre_hidden[None],
                            pre_mask[None], available,
                        )
                        difference = probability_metrics(before_probability, probabilities)
                        masks = ground_truth_masks(backend)
                        landing = tuple(adapter.gaze.current_fixation)
                        peripheral_success_indices = tuple(
                            int(value)
                            for value in event.get("peripheral_success_indices", ())
                        )
                        peripheral_success_tiles = tuple(
                            action_index_to_tile(index)
                            for index in peripheral_success_indices
                        )
                        opacity_success_tiles = tuple(
                            (action_index_to_tile(int(tile_index)), int(step))
                            for tile_index, positive_steps, success in zip(
                                event.get("opacity_opportunity_tile_indices", ()),
                                event.get("opacity_opportunity_positive_steps", ()),
                                event.get("opacity_success_mask", ()),
                            )
                            if success
                            for step in positive_steps
                        )
                        saccade = float(adapter.gaze.last_distance_degrees)
                        relative = decision_relative_timestamp(
                            timestamp, runtime["decision_start_ms"]
                        )
                        top1 = int(probabilities.argmax())
                        aoi = _aoi_fields(masks, landing, config.fixation_shape)
                        row = {
                            "model_name": config.model_name,
                            "checkpoint_step": loaded.completed_steps,
                            "checkpoint_sha256": loaded.checkpoint_sha256,
                            "T": decision_time, "H": horizon, "episode": episode,
                            "seed": seed, "decision": decision,
                            "acquisition_index": len(acquisition_rows) - acquisition_start,
                            "proposal_index": len(acquisition_rows) - acquisition_start,
                            "absolute_timestamp_ms": float(timestamp),
                            "decision_start_ms": runtime["decision_start_ms"],
                            "decision_relative_timestamp_ms": relative,
                            "remaining_time_ms": decision_time - relative,
                            "fixation_x": landing[0], "fixation_y": landing[1],
                            "landing_action_index": tile_to_action_index(landing),
                            "acquisition_footprint_tiles": json.dumps([list(tile) for tile in event["tiles"]]),
                            "acquisition_all_success_tiles": json.dumps([
                                list(tile) for tile in event["all_success_tiles"]
                            ]),
                            **{
                                key: json.dumps(event.get(key, []))
                                for key in (
                                    "foveal_tile_indices",
                                    "peripheral_candidate_indices",
                                    "peripheral_candidate_eccentricity_deg",
                                    "peripheral_candidate_detection_probability",
                                    "peripheral_success_mask",
                                    "peripheral_success_indices",
                                    "peripheral_failure_indices",
                                    "opacity_opportunity_tile_indices",
                                    "opacity_opportunity_steps",
                                    "opacity_opportunity_probabilities",
                                    "opacity_opportunity_positive_steps",
                                    "opacity_opportunity_positive_step_counts",
                                    "opacity_opportunity_max_alphas",
                                    "opacity_attempt_tile_indices",
                                    "opacity_attempt_steps",
                                    "opacity_attempt_probabilities",
                                    "opacity_attempt_positive_steps",
                                    "opacity_attempt_max_alphas",
                                    "opacity_success_mask",
                                    "opacity_success_tile_indices",
                                    "opacity_success_steps",
                                    "opacity_miss_tile_indices",
                                    "opacity_miss_steps",
                                )
                            },
                            "peripheral_candidate_count": int(
                                event.get("peripheral_candidate_count", 0)
                            ),
                            "peripheral_sampled_new_fixation": int(
                                event.get(
                                    "peripheral_sampled_new_fixation", False
                                )
                            ),
                            "peripheral_success_count": int(
                                event.get("peripheral_success_count", 0)
                            ),
                            "peripheral_failure_count": int(
                                event.get("peripheral_failure_count", 0)
                            ),
                            "opacity_attempt_count": int(
                                event.get("opacity_attempt_count", 0)
                            ),
                            "opacity_opportunity_count": int(
                                event.get("opacity_opportunity_count", 0)
                            ),
                            "opacity_success_count": int(
                                event.get("opacity_success_count", 0)
                            ),
                            "opacity_miss_count": int(
                                event.get("opacity_miss_count", 0)
                            ),
                            "opacity_overlap_preview_tile_count": int(
                                event.get("opacity_overlap_preview_tile_count", 0)
                            ),
                            "opacity_overlap_step_instance_count": int(
                                event.get("opacity_overlap_step_instance_count", 0)
                            ),
                            "opacity_information_source": (
                                "peripheral_with_opacity_detected_preview"
                                if int(event.get("opacity_success_count", 0)) > 0
                                else "peripheral_without_opacity_detected_preview"
                                if int(event.get("peripheral_success_count", 0)) > 0
                                else "no_peripheral_success"
                            ),
                            "peripheral_attempt_eccentricity_sum_deg": float(sum(
                                event.get(
                                    "peripheral_candidate_eccentricity_deg", ()
                                )
                            )),
                            "peripheral_success_eccentricity_sum_deg": float(sum(
                                eccentricity
                                for eccentricity, success in zip(
                                    event.get(
                                        "peripheral_candidate_eccentricity_deg", ()
                                    ),
                                    event.get("peripheral_success_mask", ()),
                                )
                                if success
                            )),
                            "same_target": int(adapter._same_target),
                            "new_target": int(not adapter._same_target),
                            "saccade_distance_degrees": saccade,
                            **aoi,
                            **_acquired_aoi_fields(
                                masks, event["tiles"], peripheral_success_tiles,
                                opacity_success_tiles,
                                opacity_actual=(
                                    loaded.constraint_config.preview_opacity.mode
                                    == "actual"
                                ),
                            ),
                            "exclusive_center_aoi": exclusive_aoi(aoi, config.aoi_precedence, suffix="center"),
                            "exclusive_footprint_aoi": exclusive_aoi(aoi, config.aoi_precedence, suffix="footprint"),
                            **_state_annotations(backend),
                            **_memory_fields(event["before"], event["after"], event["tiles"], saccade_distance=saccade),
                            **_source_memory_fields(
                                event["before"], event["after"], event["tiles"],
                                peripheral_success_tiles,
                            ),
                            "counterfactual_top1_changed": difference["top1_changed"],
                            "counterfactual_kl": difference["kl"],
                            "counterfactual_js": difference["js"],
                            "counterfactual_tv": difference["tv"],
                            "counterfactual_mean_abs_delta_p": difference["mean_abs_delta_p"],
                            "counterfactual_delta_entropy": difference["delta_entropy"],
                            "top1_before": int(before_probability.argmax()),
                            "top1_after": top1,
                            "sampled_action": sampled,
                            "online_top1_action": top1,
                            "sampled_action_changed_from_previous": int(previous_sampled is not None and sampled != previous_sampled),
                            "online_top1_changed_from_previous": int(previous_top1 is not None and top1 != previous_top1),
                            **{f"before_p{action}": float(before_probability[action]) for action in range(6)},
                            **{f"after_p{action}": float(probabilities[action]) for action in range(6)},
                            **{f"online_p{action}": float(probabilities[action]) for action in range(6)},
                        }
                        acquisition_rows.append(row)
                        previous_sampled, previous_top1 = sampled, top1
                        return {
                            "action": sampled,
                            "candidate_hidden": candidate[0].copy(),
                            "probabilities": probabilities.copy(),
                        }

                    collected = adapter.collect_decision(proposal)
                    latest = collected["latest_proposal"]
                    learner_action = 4 if latest is None else int(latest["action"])
                    partner_observation, partner_available = (
                        backend.partner_batch_input()
                    )
                    partner_hash_before = backend.begin_partner_step()
                    partner_action = backend.partners.action(
                        partner_observation, partner_available
                    )
                    collision_count += int(
                        human_compatible_joint_transition_collision(
                            backend.mdp,
                            backend.state,
                            (
                                backend.Action.INDEX_TO_ACTION[
                                    int(partner_action)
                                ],
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
                    for row in acquisition_rows[acquisition_start:]:
                        row["final_executed_action"] = int(result["executed_action"])
                        row["final_proposal_index"] = len(collected["proposals"]) - 1
                        row["proposal_was_final"] = int(
                            row["proposal_index"] == len(collected["proposals"]) - 1
                        )
                    decision_row = {
                        "model_name": config.model_name, "T": decision_time,
                        "H": horizon, "episode": episode, "seed": seed,
                        "decision": decision,
                        "decision_start_ms": runtime["decision_start_ms"],
                        "decision_end_ms": float(adapter.current_time_ms),
                        "start_fixation_x": start_fixation[0],
                        "start_fixation_y": start_fixation[1],
                        "final_executed_action": int(result["executed_action"]),
                        "final_proposal_index": None if latest is None else len(collected["proposals"]) - 1,
                        "forced_stay": int(result["forced_stay"]),
                        "deliveries": int(result["deliveries"]),
                        "sparse_team_reward": float(result["sparse_team_reward"]),
                        "proposal_count": len(collected["proposals"]),
                    }
                    for definition, footprint in (("center", False), ("footprint", True)):
                        for label in ("near", "far"):
                            decision_row[f"{label}_{definition}_candidate_tiles"] = json.dumps(
                                _candidate_tiles(start_masks[label], shape=config.fixation_shape, footprint=footprint)
                            )
                    decision_rows.append(decision_row)
                    if result["done"]:
                        break
                backend.verify_partner_parameters()
                metrics = adapter.metrics()
                episode_row = {
                    "model_name": config.model_name, "T": decision_time,
                    "H": horizon, "episode": episode, "seed": seed,
                    "deliveries": deliveries, "sparse_team_reward": sparse,
                    "collision_count": collision_count,
                    "collision_rate": collision_count / max(len(actions), 1),
                    "action_trajectory_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
                    **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
                }
                episode_rows.append(episode_row)
                if reporter is not None:
                    condition_rows = episode_rows[condition_episode_start:]
                    acquisitions_total = sum(
                        int(row["acquisitions"]) for row in condition_rows
                    )
                    decisions_total = sum(
                        int(row["task_decisions"]) for row in condition_rows
                    )
                    reporter.episode(
                        model=config.model_name,
                        decision_time=decision_time, horizon=horizon,
                        episode=episode + 1, episodes=config.episodes,
                        deliveries_mean=float(np.mean([
                            row["deliveries"] for row in condition_rows
                        ])),
                        acquisitions_per_decision=(
                            acquisitions_total / max(decisions_total, 1)
                        ),
                        last_episode=episode_row,
                    )
            if reporter is not None:
                condition_rows = episode_rows[condition_episode_start:]
                acquisitions_total = sum(
                    int(row["acquisitions"]) for row in condition_rows
                )
                decisions_total = sum(
                    int(row["task_decisions"]) for row in condition_rows
                )
                forced_total = sum(
                    int(row["forced_stays"]) for row in condition_rows
                )
                next_condition = (
                    "none"
                    if condition_index == len(conditions)
                    else f"T{conditions[condition_index][0]} "
                         f"H{conditions[condition_index][1]}"
                )
                reporter.done_condition(
                    decision_time=decision_time, horizon=horizon,
                    episodes=config.episodes, started=condition_started,
                    deliveries=float(np.mean([
                        row["deliveries"] for row in condition_rows
                    ])),
                    acquisitions_per_decision=(
                        acquisitions_total / max(decisions_total, 1)
                    ),
                    forced_stay=forced_total / max(decisions_total, 1),
                    next_condition=next_condition,
                )

    if reporter is not None:
        reporter.model_done(
            config.model_name, len(conditions) * config.episodes,
            model_started,
        )

    loaded.assert_unchanged()
    if parameter_hash(loaded.actor) != actor_before or parameter_hash(loaded.critic) != critic_before:
        raise RuntimeError("evaluation changed ActionNet/critic parameters")
    if loaded.gaze_sha256 != gaze_before:
        raise RuntimeError("evaluation changed GazeNet parameters")
    if any(action_index_to_tile(index) != GAZE_TILES[index] for index in range(40)):
        raise RuntimeError("canonical gaze x-major mapping mismatch")
    if any(
        int(row["H"]) == 0 and any(float(row[f"preview_{definition}_{metric}"]) != 0.0 for definition in ("center", "footprint") for metric in ("chance_mass", "policy_mass", "sample_hit"))
        for row in request_rows
    ):
        raise RuntimeError("H0 preview leakage detected in saved dataset")

    write_csv(model_dir / "episodes.csv", episode_rows)
    write_csv(model_dir / "decisions.csv", decision_rows)
    _write_gzip_csv(model_dir / "gaze_requests.csv.gz", request_rows)
    _write_gzip_csv(model_dir / "acquisitions.csv.gz", acquisition_rows)
    with atomic_output_path(model_dir / "gaze_inputs.npz") as temporary:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                memory=np.asarray(gaze_memories, dtype=np.float32),
                fixation=np.asarray(gaze_fixations, dtype=np.int8),
                context=np.asarray(gaze_contexts, dtype=np.float32),
            )
    manifest = {
        "analysis": "canonical_gaze_behavior_rich_rollout_v1",
        "model_name": config.model_name,
        "checkpoint_absolute_path": str(loaded.path),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_completed_step": loaded.completed_steps,
        "actor_sha256": loaded.actor_sha256,
        "critic_sha256": loaded.critic_sha256,
        "gaze_sha256": loaded.gaze_sha256,
        "memory_lifetime_ms": loaded.metadata["memory_lifetime_ms"],
        "memory_decay_mode": loaded.metadata["memory_decay_mode"],
        "decision_times": list(config.decision_times), "horizons": list(config.horizons),
        "fixation_shape": config.fixation_shape, "episodes": config.episodes,
        "episode_length": config.episode_length, "seed_start": config.seed_start,
        "gaze_mode": config.gaze_mode, "action_mode": config.action_mode,
        "aoi_precedence": list(config.aoi_precedence),
        "aoi_precedence_name": "explicit_model_internal_exclusive_order",
        "optimizer_free": True, "parameter_hash_preserved": True,
        "ground_truth_policy_input": bool(
            loaded.metadata.get("ground_truth_policy_input", False)
        ),
        "simulation_count_per_model_condition_seed": 1,
        "collision_contract_id": (
            "human_compatible_joint_transition_collision_v1"
        ),
        "request_count": len(request_rows), "acquisition_count": len(acquisition_rows),
        **loaded.constraint_config.metadata(),
        "unsupported_semantics": {
            "semantic_action_revision": "no canonical planner uniquely maps a primitive direction to a task goal",
            "same_resource_target_contention": "partner and learner target intent is not exposed canonically",
            "human_inspired_role_switch_cases": "intent predicates cannot be derived exactly from state alone",
        },
        **{key: loaded.metadata.get(key) for key in (
            "observation_contract_id", "freshness_contract_id",
            "gaze_action_contract_id", "gaze_input_contract_id",
            "gaze_credit_assignment_contract_id", "entropy_schedule",
            "source_full_checkpoint_provenance",
        )},
        **provenance,
    }
    manifest = finalize_rollout_artifact(model_dir, manifest)
    return {"manifest": manifest, "episodes": episode_rows, "decisions": decision_rows}
