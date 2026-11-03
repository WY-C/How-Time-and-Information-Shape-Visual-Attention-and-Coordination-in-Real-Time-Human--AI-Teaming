"""Event-driven canonical Partial observation and proposal runtime."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.partial_ver.constraints import (
    BASELINE_CONSTRAINT_CONFIG,
    HumanLikeConstraintConfig,
    constraint_config_for_emma,
    validate_constraint_implementation,
    constraint_config_from_metadata,
)
from final_model.partial_ver.gaze import (
    EmmaTimingConfig,
    GazePhase,
    GazeStateMachine,
    VISUAL_ENCODING_EMMA_MODES,
    acquisition_tiles,
    visual_encoding_time_ms,
)
from final_model.partial_ver.memory import MEMORY_LIFETIME_MS, PersistentPartialMemory
from final_model.partial_ver.peripheral import (
    PeripheralDetection,
    sample_peripheral_detection,
)
from final_model.partial_ver.preview_opacity import (
    PreviewOpacityDetection,
    sample_preview_opacity,
)
from final_model.partial_ver.rule_based import RuleBasedGazePolicy
from final_model.server_parallel.gaze_v1 import tile_to_action_index
from final_model.simple_ver.observation import build_full_observation, preview4


ALLOWED_DECISION_TIMES = (200, 400, 600)
ACQUISITION_ACTION_CONTRACT_ID = "actionnet_query_after_visual_acquisition_only_v1"
RECURRENT_COMMIT_CONTRACT_ID = "same_predecision_hidden_latest_proposal_commit_v1"
DEADLINE_ACTION_CONTRACT_ID = "actionnet_single_sample_at_decision_deadline_v1"
DEADLINE_RECURRENT_COMMIT_CONTRACT_ID = "decision_deadline_candidate_hidden_commit_v1"
HELD_FOVEAL_REFRESH_CONTRACT_ID = (
    "post_environment_step_plus_refresh_while_human_hold_v1"
)
ACTOR_AVAILABLE_ACTION_CONTRACT_ID = "partial_actor_all_six_actions_no_privileged_mask_v1"
FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID = (
    "full_state20_freshness255_displayed_preview4_gaze_independent_v1"
)


def available_action_audit_report() -> dict[str, object]:
    return {
        "contract_id": ACTOR_AVAILABLE_ACTION_CONTRACT_ID,
        "backend_mask_actor_visible": False,
        "actor_mask": [1.0] * 6,
        "reason": (
            "the exact backend mask depends on hidden current learner position, "
            "orientation, occupancy, and terrain interaction state"
        ),
        "centralized_critic_privileged_state": True,
        "critic_state_passed_to_actor": False,
    }


def time_one_hot(value: int) -> tuple[float, float, float]:
    result = [0.0, 0.0, 0.0]
    result[ALLOWED_DECISION_TIMES.index(int(value))] = 1.0
    return tuple(result)


def horizon_one_hot(value: int) -> tuple[float, float, float]:
    allowed = (0, 2, 4)
    result = [0.0, 0.0, 0.0]
    result[allowed.index(int(value))] = 1.0
    return tuple(result)


@dataclass(frozen=True)
class PartialDecision:
    observation: np.ndarray
    critic: np.ndarray
    available_actions: np.ndarray
    action: int
    result: dict


class PartialObservationAdapter:
    def __init__(
        self,
        backend: DirectOvercookedBackend,
        *,
        decision_time: int,
        fixation_shape: str,
        gaze_policy=None,
        rule_gaze_seed: int = 0,
        rule_gaze_probabilities=None,
        defer_preview_until_acquisition: bool = False,
        memory_lifetime_ms: float | None = MEMORY_LIFETIME_MS,
        memory_decay_mode: str = "linear",
        memory_tau_ms: float | None = None,
        memory_seed: int | None = None,
        emma_timing: EmmaTimingConfig | None = None,
        constraint_config: HumanLikeConstraintConfig | None = None,
        peripheral_seed: int | None = None,
        opacity_seed: int | None = None,
        dwell_seed: int | None = None,
        ground_truth_policy_input: bool = False,
    ) -> None:
        if decision_time not in ALLOWED_DECISION_TIMES:
            raise ValueError(f"decision_time must be one of {ALLOWED_DECISION_TIMES}")
        self.backend = backend
        self.decision_time = int(decision_time)
        self.fixation_shape = str(fixation_shape)
        self._uses_rule_gaze = gaze_policy is None
        self.rule_gaze_seed = int(rule_gaze_seed)
        self.rule_gaze_probabilities = rule_gaze_probabilities
        self.defer_preview_until_acquisition = bool(
            defer_preview_until_acquisition
        )
        self.ground_truth_policy_input = bool(ground_truth_policy_input)
        self.gaze_policy = (
            RuleBasedGazePolicy(
                self.fixation_shape,
                seed=self.rule_gaze_seed,
                probabilities=self.rule_gaze_probabilities,
            )
            if gaze_policy is None
            else gaze_policy
        )
        self.memory = PersistentPartialMemory(
            memory_lifetime_ms=memory_lifetime_ms,
            memory_decay_mode=memory_decay_mode,
            memory_tau_ms=memory_tau_ms,
            memory_seed=memory_seed,
        )
        if emma_timing is not None and constraint_config is not None:
            raise ValueError("provide constraint_config or legacy emma_timing, not both")
        self.constraint_config = (
            BASELINE_CONSTRAINT_CONFIG
            if constraint_config is None and emma_timing is None
            else constraint_config
            if constraint_config is not None
            else constraint_config_for_emma(emma_timing)
        )
        validate_constraint_implementation(self.constraint_config)
        if (
            self.constraint_config.peripheral.mode != "none"
            and self.fixation_shape != "plus"
        ):
            raise ValueError("active peripheral vision requires fixation_shape=plus")
        if (
            self.constraint_config.peripheral.mode != "none"
            and peripheral_seed is None
        ):
            raise ValueError("active peripheral vision requires peripheral_seed")
        if (
            self.constraint_config.preview_opacity.mode == "actual"
            and opacity_seed is None
        ):
            raise ValueError("active preview opacity requires opacity_seed")
        self.emma_timing = self.constraint_config.emma
        if self.emma_timing.mode == "human_stochastic" and dwell_seed is None:
            raise ValueError("human_stochastic EMMA timing requires dwell_seed")
        if self.emma_timing.mode != "human_stochastic" and dwell_seed is not None:
            raise ValueError("dwell_seed is only valid for human_stochastic EMMA timing")
        self.action_query_contract_id = (
            DEADLINE_ACTION_CONTRACT_ID
            if self.emma_timing.mode == "human_stochastic"
            or self.emma_timing.mode in VISUAL_ENCODING_EMMA_MODES
            else ACQUISITION_ACTION_CONTRACT_ID
        )
        self.recurrent_commit_contract_id = (
            DEADLINE_RECURRENT_COMMIT_CONTRACT_ID
            if self.emma_timing.mode == "human_stochastic"
            or self.emma_timing.mode in VISUAL_ENCODING_EMMA_MODES
            else RECURRENT_COMMIT_CONTRACT_ID
        )
        self.held_foveal_refresh_contract_id = (
            HELD_FOVEAL_REFRESH_CONTRACT_ID
            if self.emma_timing.mode == "human_stochastic"
            else None
        )
        self.peripheral_seed = 0 if peripheral_seed is None else int(peripheral_seed)
        self.peripheral_rng = np.random.Generator(
            np.random.PCG64(self.peripheral_seed)
        )
        self.peripheral_sample_events = 0
        self.peripheral_rng_draw_count = 0
        self.opacity_seed = 0 if opacity_seed is None else int(opacity_seed)
        self.opacity_rng = np.random.Generator(np.random.PCG64(self.opacity_seed))
        self.opacity_rng_draw_count = 0
        self.gaze = GazeStateMachine(self.emma_timing, dwell_seed=dwell_seed)
        self.current_time_ms = 0.0
        self.decision_index = 0
        self.actionnet_online_forwards = 0
        self.task_decisions = 0
        self.gaze_requests = 0
        self.acquisitions = 0
        self.same_target_acquisitions = 0
        self.new_target_acquisitions = 0
        self.prep_count = 0
        self.exec_count = 0
        self.dwell_count = 0
        self.distance_sum_degrees = 0.0
        self.action_proposals = 0
        self.proposal_revisions = 0
        self.executed_proposals = 0
        self.forced_stays = 0
        self.no_acquisition_decisions = 0
        self.recurrent_commits = 0
        self.held_foveal_refreshes = 0
        self.memory_refresh_total = 0
        self.memory_expiry_total = 0
        self.freshness_valid_count = 0
        self.freshness_sample_count = 0
        self.freshness_normalized_sum = 0.0
        self.freshness_normalized_square_sum = 0.0
        self.freshness_normalized_min = np.inf
        self.freshness_normalized_max = -np.inf
        self.peripheral_attempts = 0
        self.peripheral_successes = 0
        self.peripheral_failures = 0
        self.peripheral_attempt_eccentricity_sum_degrees = 0.0
        self.peripheral_success_eccentricity_sum_degrees = 0.0
        self.opacity_attempts_by_step = np.zeros(4, dtype=np.int64)
        self.opacity_successes_by_step = np.zeros(4, dtype=np.int64)
        self.opacity_opportunities_by_step = np.zeros(4, dtype=np.int64)
        self.effective_preview_acquisitions_by_step = np.zeros(4, dtype=np.int64)
        self.opacity_preview_tiles = 0
        self.overlap_preview_tiles = 0
        self.overlap_step_instances = 0
        self.opacity_positive_step_instances = 0
        self.opacity_max_alpha_sum = 0.0
        self.opacity_max_probability_sum = 0.0
        self.last_acquisition_metadata = None
        self._last_peripheral_fixation = None
        self.requested_target_counts = np.zeros((8, 5), dtype=np.int64)
        self.landing_target_counts = np.zeros((8, 5), dtype=np.int64)
        self.events = deque(maxlen=256)
        self._same_target = True
        self._reset_observation_state(initial=True)

    def _visual(self):
        learner = self.backend.canonical_pair()[1]
        return learner, preview4(self.backend.preview_planes(), self.backend.horizon)

    def policy_observation(self) -> np.ndarray:
        if not self.ground_truth_policy_input:
            return self.memory.snapshot(self.current_time_ms)
        learner = self.backend.canonical_pair()[1]
        return build_full_observation(
            learner,
            self.backend.preview_planes(),
            self.backend.horizon,
        )

    def _reset_observation_state(self, *, initial: bool = False) -> None:
        learner = (
            self.backend.canonical_pair()[1]
            if self.defer_preview_until_acquisition
            else self._visual()[0]
        )
        if not initial:
            self.memory_refresh_total += self.memory.refresh_count
            self.memory_expiry_total += self.memory.expiry_count
        self.memory.reset(learner, now_ms=self.current_time_ms)
        self.decision_index = 0
        self.gaze.reset(self.backend.self_tile(), self.current_time_ms)
        self._last_peripheral_fixation = None
        if hasattr(self.gaze_policy, "reset"):
            self.gaze_policy.reset()
        if not initial:
            self.events.append(
                {"timestamp_ms": self.current_time_ms, "event": "episode_reset"}
            )

    def set_condition(self, *, decision_time: int, horizon: int, fixation_shape: str) -> None:
        if self.backend.episode_step != 0:
            raise RuntimeError("Partial condition may change only at an episode boundary")
        if int(decision_time) not in ALLOWED_DECISION_TIMES:
            raise ValueError("invalid Partial decision time")
        self.backend.set_horizon(int(horizon))
        changed_shape = str(fixation_shape) != self.fixation_shape
        self.decision_time = int(decision_time)
        self.fixation_shape = str(fixation_shape)
        if (
            self.constraint_config.peripheral.mode != "none"
            and self.fixation_shape != "plus"
        ):
            raise ValueError("active peripheral vision requires fixation_shape=plus")
        if changed_shape and self._uses_rule_gaze:
            old_state = self.gaze_policy.training_state()
            self.gaze_policy = RuleBasedGazePolicy(
                self.fixation_shape,
                seed=self.rule_gaze_seed,
                probabilities=self.rule_gaze_probabilities,
            )
            # Shape changes intentionally start a fresh episode-local selection history;
            # the isolated RNG continues from its previous state.
            self.gaze_policy.rng.bit_generator.state = copy.deepcopy(old_state["rng_state"])
            self.gaze_policy.selection_count = int(old_state["selection_count"])
            self.gaze_policy.branch_counts = copy.deepcopy(old_state["branch_counts"])
            self.gaze_policy.last_branch = str(old_state["last_branch"])
            self.gaze_policy.last_requested_target = old_state["last_requested_target"]
            self.gaze_policy.events = copy.deepcopy(old_state["events"])

    def reset_episode(self, condition=None) -> None:
        self.backend.reset()
        if condition is not None:
            self.set_condition(
                decision_time=condition.decision_time,
                horizon=condition.horizon,
                fixation_shape=condition.fixation_shape,
            )
        self._reset_observation_state()

    def gaze_context(self, deadline: float) -> np.ndarray:
        remaining = max(0.0, deadline - self.current_time_ms) / self.decision_time
        return np.asarray(
            (remaining, *time_one_hot(self.decision_time), *horizon_one_hot(self.backend.horizon)),
            dtype=np.float32,
        )

    def _select_if_ready(self, deadline: float) -> None:
        if self.gaze.phase is not GazePhase.READY or self.current_time_ms >= deadline:
            return
        self.memory.set_time(self.current_time_ms)
        if hasattr(self.gaze_policy, "set_event_time"):
            self.gaze_policy.set_event_time(self.current_time_ms)
        target = self.gaze_policy(
            self.policy_observation(),
            self.gaze.current_fixation,
            self.gaze_context(deadline),
        )
        self._same_target = tuple(target) == self.gaze.current_fixation
        self.requested_target_counts[tuple(target)] += 1
        self.gaze.select(target, self.current_time_ms)
        self.gaze_requests += 1
        if not self._same_target:
            self.distance_sum_degrees += self.gaze.last_distance_degrees

    def _actor_available_actions(self) -> np.ndarray:
        return np.ones(6, dtype=np.float32)

    def _record_freshness(self, observation: np.ndarray) -> None:
        raw = np.asarray(observation[..., 20], dtype=np.float64)
        if not np.isfinite(raw).all() or np.any(raw < 0.0) or np.any(raw > 255.0):
            raise RuntimeError("invalid continuous Partial freshness")
        semantic = raw / 255.0
        self.freshness_valid_count += int(np.count_nonzero(semantic > 0.0))
        self.freshness_sample_count += int(semantic.size)
        self.freshness_normalized_sum += float(semantic.sum())
        self.freshness_normalized_square_sum += float(np.square(semantic).sum())
        self.freshness_normalized_min = min(
            self.freshness_normalized_min, float(semantic.min())
        )
        self.freshness_normalized_max = max(
            self.freshness_normalized_max, float(semantic.max())
        )

    def _critic_observation(self) -> np.ndarray:
        partner, learner = self.backend.canonical_pair()
        return np.ascontiguousarray(
            np.concatenate((learner, partner), axis=-1), dtype=np.float32
        )

    def _peripheral_detection(self) -> PeripheralDetection:
        self.peripheral_sample_events += 1
        result = sample_peripheral_detection(
            mode=self.constraint_config.peripheral.mode,
            sigma_deg=self.constraint_config.peripheral.sigma_deg,
            extent=self.constraint_config.peripheral.extent,
            fixation=self.gaze.current_fixation,
            rng=self.peripheral_rng,
        )
        attempts = len(result.candidate_tiles)
        successes = len(result.success_tiles)
        self.peripheral_rng_draw_count += attempts
        self.peripheral_attempts += attempts
        self.peripheral_successes += successes
        self.peripheral_failures += attempts - successes
        self.peripheral_attempt_eccentricity_sum_degrees += sum(
            result.eccentricities_deg
        )
        self.peripheral_success_eccentricity_sum_degrees += sum(
            eccentricity
            for eccentricity, success in zip(
                result.eccentricities_deg, result.success_mask
            )
            if success
        )
        return result

    def _acquisition_metadata(
        self, fixation, foveal_tiles, peripheral: PeripheralDetection,
    ) -> dict[str, Any]:
        foveal_indices = sorted(tile_to_action_index(tile) for tile in foveal_tiles)
        candidate_indices = [
            tile_to_action_index(tile) for tile in peripheral.candidate_tiles
        ]
        success_indices = [
            index for index, success in zip(
                candidate_indices, peripheral.success_mask
            ) if success
        ]
        failure_indices = [
            index for index, success in zip(
                candidate_indices, peripheral.success_mask
            ) if not success
        ]
        result = {
            "fixation_index": tile_to_action_index(fixation),
            "foveal_tile_indices": foveal_indices,
            "peripheral_candidate_indices": candidate_indices,
            "peripheral_candidate_eccentricity_deg": list(
                peripheral.eccentricities_deg
            ),
            "peripheral_candidate_detection_probability": list(
                peripheral.probabilities
            ),
            "peripheral_success_mask": list(peripheral.success_mask),
            "peripheral_success_indices": success_indices,
            "peripheral_failure_indices": failure_indices,
            "peripheral_candidate_count": len(candidate_indices),
            "peripheral_success_count": len(success_indices),
            "peripheral_failure_count": len(failure_indices),
        }
        if self.emma_timing.mode in VISUAL_ENCODING_EMMA_MODES:
            geometry = self.gaze.geometry
            result.update({
                "foveal_encoding_time_ms": [
                    visual_encoding_time_ms(
                        geometry.visual_angle_degrees(fixation, tile)
                    )
                    for tile in foveal_tiles
                ],
                "peripheral_candidate_encoding_time_ms": [
                    visual_encoding_time_ms(eccentricity)
                    for eccentricity in peripheral.eccentricities_deg
                ],
                "encoding_K": self.emma_timing.encoding_K_seconds,
                "encoding_k": self.emma_timing.encoding_k,
                "encoding_f": self.emma_timing.encoding_f,
            })
        return result

    def _preview_opacity_detection(
        self, displayed_preview4: np.ndarray,
        peripheral: PeripheralDetection,
    ) -> PreviewOpacityDetection:
        result = sample_preview_opacity(
            mode=self.constraint_config.preview_opacity.mode,
            displayed_horizon=self.backend.horizon,
            displayed_preview4=displayed_preview4,
            peripheral_success_tiles=peripheral.success_tiles,
            rng=self.opacity_rng,
        )
        attempts = len(result.attempt_steps)
        self.opacity_rng_draw_count += attempts
        self.opacity_preview_tiles += len(result.opportunity_tile_indices)
        self.opacity_max_alpha_sum += sum(result.opportunity_max_alphas)
        self.opacity_max_probability_sum += sum(
            result.opportunity_probabilities
        )
        for positive_steps in result.opportunity_positive_steps:
            self.opacity_positive_step_instances += len(positive_steps)
            if len(positive_steps) > 1:
                self.overlap_preview_tiles += 1
                self.overlap_step_instances += len(positive_steps)
            for step in positive_steps:
                self.opacity_opportunities_by_step[step - 1] += 1
        for index, (step, success) in enumerate(zip(
            result.attempt_steps, result.success_mask
        )):
            self.opacity_attempts_by_step[step - 1] += 1
            if success:
                self.opacity_successes_by_step[step - 1] += 1
                for positive_step in result.opportunity_positive_steps[index]:
                    self.effective_preview_acquisitions_by_step[
                        positive_step - 1
                    ] += 1
        if self.constraint_config.preview_opacity.mode == "off":
            for positive_steps in result.opportunity_positive_steps:
                for step in positive_steps:
                    self.effective_preview_acquisitions_by_step[step - 1] += 1
        return result

    @staticmethod
    def _opacity_metadata(
        opacity: PreviewOpacityDetection,
    ) -> dict[str, Any]:
        return {
            "opacity_opportunity_tile_indices": list(
                opacity.opportunity_tile_indices
            ),
            "opacity_opportunity_steps": list(opacity.opportunity_steps),
            "opacity_opportunity_probabilities": list(
                opacity.opportunity_probabilities
            ),
            "opacity_opportunity_positive_steps": [
                list(steps) for steps in opacity.opportunity_positive_steps
            ],
            "opacity_opportunity_positive_step_counts": [
                len(steps) for steps in opacity.opportunity_positive_steps
            ],
            "opacity_opportunity_max_alphas": list(
                opacity.opportunity_max_alphas
            ),
            "opacity_attempt_tile_indices": list(opacity.attempt_tile_indices),
            "opacity_attempt_steps": list(opacity.attempt_steps),
            "opacity_attempt_probabilities": list(
                opacity.attempt_probabilities
            ),
            "opacity_attempt_positive_steps": [
                list(steps)
                for steps in opacity.opportunity_positive_steps[
                    :len(opacity.attempt_steps)
                ]
            ],
            "opacity_attempt_max_alphas": list(
                opacity.opportunity_max_alphas[:len(opacity.attempt_steps)]
            ),
            "opacity_success_mask": list(opacity.success_mask),
            "opacity_success_tile_indices": list(
                opacity.success_tile_indices
            ),
            "opacity_success_steps": list(opacity.success_steps),
            "opacity_miss_tile_indices": list(opacity.miss_tile_indices),
            "opacity_miss_steps": list(opacity.miss_steps),
            "opacity_attempt_count": len(opacity.attempt_steps),
            "opacity_opportunity_count": len(opacity.opportunity_steps),
            "opacity_overlap_preview_tile_count": sum(
                len(steps) > 1 for steps in opacity.opportunity_positive_steps
            ),
            "opacity_overlap_step_instance_count": sum(
                len(steps) for steps in opacity.opportunity_positive_steps
                if len(steps) > 1
            ),
            "opacity_success_count": sum(opacity.success_mask),
            "opacity_miss_count": len(opacity.attempt_steps)
            - sum(opacity.success_mask),
        }

    def advance_to(self, deadline: float, on_acquisition=None) -> list[dict[str, Any]]:
        deadline = float(deadline)
        if deadline < self.current_time_ms:
            raise ValueError("Partial deadline cannot move backward")
        proposals = []
        self._select_if_ready(deadline)
        while self.gaze.next_event_ms is not None and self.gaze.next_event_ms <= deadline:
            self.current_time_ms = float(self.gaze.next_event_ms)
            self.memory.set_time(self.current_time_ms)
            event = self.gaze.advance_event()
            if event.completed_phase is GazePhase.PREP:
                self.prep_count += 1
            elif event.completed_phase is GazePhase.EXEC:
                self.exec_count += 1
            elif event.completed_phase in {GazePhase.DWELL, GazePhase.ENCODE}:
                self.dwell_count += 1
            record = {
                "timestamp_ms": event.timestamp_ms,
                "completed_phase": event.completed_phase.value,
                "acquisition": event.acquisition,
            }
            if event.acquisition:
                state, displayed = self._visual()
                tiles = acquisition_tiles(
                    self.gaze.current_fixation, self.fixation_shape
                )
                peripheral_sampled = (
                    self.constraint_config.peripheral.mode != "none"
                    and self._last_peripheral_fixation
                    != self.gaze.current_fixation
                )
                peripheral = (
                    self._peripheral_detection()
                    if peripheral_sampled
                    else PeripheralDetection((), (), (), ())
                )
                if peripheral_sampled:
                    self._last_peripheral_fixation = tuple(
                        self.gaze.current_fixation
                    )
                opacity = self._preview_opacity_detection(displayed, peripheral)
                acquisition_metadata = self._acquisition_metadata(
                    self.gaze.current_fixation, tiles, peripheral
                )
                acquisition_metadata["peripheral_sampled_new_fixation"] = (
                    peripheral_sampled
                )
                acquisition_metadata.update(self._opacity_metadata(opacity))
                self.last_acquisition_metadata = acquisition_metadata
                self.memory.acquire(
                    state, opacity.perceived_preview4,
                    (*tiles, *peripheral.success_tiles),
                    timestamp_ms=self.current_time_ms
                )
                acquired_tiles = tuple(dict.fromkeys(
                    (*tiles, *peripheral.success_tiles)
                ))
                acquisition_metadata["acquired_tile_indices"] = sorted(
                    tile_to_action_index(tile) for tile in acquired_tiles
                )
                if self.memory.memory_decay_mode == "stochastic_exponential":
                    acquisition_metadata.update({
                        "memory_sampled_lifetime_ms": [
                            float(self.memory.sampled_lifetime_ms[tile])
                            for tile in acquired_tiles
                        ],
                        "memory_actual_lifetime_ms": [
                            float(self.memory.actual_lifetime_ms[tile])
                            for tile in acquired_tiles
                        ],
                        "memory_age_ms": [0.0] * len(acquired_tiles),
                        "memory_freshness": [1.0] * len(acquired_tiles),
                        "memory_deletion_reason": [None] * len(acquired_tiles),
                    })
                self.acquisitions += 1
                self.landing_target_counts[self.gaze.current_fixation] += 1
                if self._same_target:
                    self.same_target_acquisitions += 1
                else:
                    self.new_target_acquisitions += 1
                record["tiles"] = [list(tile) for tile in tiles]
                record.update(copy.deepcopy(acquisition_metadata))
                record["memory_updated_before_actionnet"] = True
                if hasattr(self.gaze_policy, "record_landing"):
                    self.gaze_policy.record_landing(
                        landing_target=self.gaze.current_fixation,
                        acquisition_success=True,
                        same_target=self._same_target,
                    )
                if on_acquisition is not None:
                    observation = self.policy_observation()
                    self._record_freshness(observation)
                    proposal = dict(
                        on_acquisition(
                            observation,
                            self._critic_observation(),
                            self._actor_available_actions(),
                            self.current_time_ms,
                        )
                    )
                    proposal.setdefault("observation", observation.copy())
                    proposal.setdefault("available_actions", self._actor_available_actions())
                    proposal["acquisition_time_ms"] = self.current_time_ms
                    proposals.append(proposal)
                    self.actionnet_online_forwards += 1
                    self.action_proposals += 1
                    record["actionnet_query"] = True
            self.events.append(record)
            self._select_if_ready(deadline)
        self.current_time_ms = deadline
        self.memory.set_time(deadline)
        return proposals

    def current(self):
        return (
            self.policy_observation(),
            self._critic_observation(),
            self._actor_available_actions(),
        )

    def _refresh_held_fovea_after_environment_step(self) -> bool:
        if (
            self.emma_timing.mode != "human_stochastic"
            or self.gaze.phase is not GazePhase.HOLD
        ):
            return False
        state, displayed = self._visual()
        tiles = acquisition_tiles(
            self.gaze.current_fixation, self.fixation_shape
        )
        metadata = self._acquisition_metadata(
            self.gaze.current_fixation,
            tiles,
            PeripheralDetection((), (), (), ()),
        )
        metadata.update({
            "acquisition_kind": "held_foveal_post_environment_step_refresh",
            "peripheral_sampled_new_fixation": False,
        })
        self.last_acquisition_metadata = metadata
        self.memory.acquire(
            state,
            displayed,
            tiles,
            timestamp_ms=self.current_time_ms,
        )
        self.held_foveal_refreshes += 1
        self.events.append({
            "timestamp_ms": self.current_time_ms,
            "event": "held_foveal_post_environment_step_refresh",
            "acquisition": False,
            "held_foveal_refresh": True,
            "tiles": [list(tile) for tile in tiles],
            "memory_updated_after_environment_step": True,
        })
        return True

    def prepare_decision(self):
        deadline = self.current_time_ms + self.decision_time
        self.advance_to(deadline)
        return self.current()

    def collect_decision(self, on_acquisition) -> dict[str, Any]:
        deadline = self.current_time_ms + self.decision_time
        acquisitions_before = self.acquisitions
        if (
            self.emma_timing.mode == "human_stochastic"
            or self.emma_timing.mode in VISUAL_ENCODING_EMMA_MODES
        ):
            self.advance_to(deadline)
            observation = self.policy_observation()
            self._record_freshness(observation)
            proposal = dict(on_acquisition(
                observation,
                self._critic_observation(),
                self._actor_available_actions(),
                deadline,
            ))
            proposal.setdefault("observation", observation.copy())
            proposal.setdefault(
                "available_actions", self._actor_available_actions()
            )
            proposal["action_query_time_ms"] = deadline
            self.actionnet_online_forwards += 1
            self.action_proposals += 1
            proposals = [proposal]
        else:
            proposals = self.advance_to(
                deadline, on_acquisition=on_acquisition
            )
        if len(proposals) > 1:
            self.proposal_revisions += len(proposals) - 1
        latest = None if not proposals else proposals[-1]
        return {
            "deadline_ms": deadline,
            "proposals": proposals,
            "latest_proposal": latest,
            "observation": self.policy_observation(),
            "critic": self._critic_observation(),
            "available_actions": self._actor_available_actions(),
            "acquisitions": self.acquisitions - acquisitions_before,
            "actor_sample_active": latest is not None,
            "forced_stay": latest is None,
        }

    def commit_collected(self, collected, *, partner_action=None, partner_hash_before=None):
        latest = collected["latest_proposal"]
        if latest is None:
            action = 4
            self.no_acquisition_decisions += 1
            self.forced_stays += 1
        else:
            action = int(latest["action"])
            self.executed_proposals += 1
            self.recurrent_commits += 1
        self.task_decisions += 1
        self.decision_index += 1
        if partner_action is None:
            result = self.backend.step(action)
        else:
            result = self.backend.step_with_partner_action(
                action,
                int(partner_action),
                partner_hash_before=str(partner_hash_before),
            )
        if not result["done"]:
            self._refresh_held_fovea_after_environment_step()
        return {
            **result,
            "executed_action": action,
            "actor_sample_active": latest is not None,
            "forced_stay": latest is None,
            "no_acquisition_decision": latest is None,
            "proposal_count": len(collected["proposals"]),
            "latest_proposal": latest,
        }

    def run_decision(self, policy: Callable[[np.ndarray, np.ndarray, np.ndarray], int]):
        def propose(observation, critic, available, _timestamp):
            return {"action": int(policy(observation, critic, available))}

        collected = self.collect_decision(propose)
        result = self.commit_collected(collected)
        latest = collected["latest_proposal"]
        observation = (
            collected["observation"] if latest is None else latest["observation"]
        )
        return PartialDecision(
            observation,
            collected["critic"],
            collected["available_actions"],
            result["executed_action"],
            result,
        )

    def metrics(self) -> dict[str, Any]:
        memory = self.memory.diagnostics()
        count = self.freshness_sample_count
        mean = self.freshness_normalized_sum / max(count, 1)
        variance = (
            self.freshness_normalized_square_sum / max(count, 1) - mean * mean
        )
        opacity_attempts = int(self.opacity_attempts_by_step.sum())
        opacity_successes = int(self.opacity_successes_by_step.sum())
        opacity_opportunities = int(self.opacity_opportunities_by_step.sum())
        effective_preview_detections = (
            opacity_opportunities
            if self.constraint_config.preview_opacity.mode == "off"
            else opacity_successes
        )
        result = {
            "gaze_requests": self.gaze_requests,
            "task_decisions": self.task_decisions,
            "acquisitions": self.acquisitions,
            "same_target_acquisitions": self.same_target_acquisitions,
            "new_target_acquisitions": self.new_target_acquisitions,
            "prep_count": self.prep_count,
            "exec_count": self.exec_count,
            "dwell_count": self.dwell_count,
            "mean_saccade_distance_degrees": self.distance_sum_degrees
            / max(self.new_target_acquisitions, 1),
            "actionnet_queries": self.actionnet_online_forwards,
            "action_proposals": self.action_proposals,
            "proposal_revisions": self.proposal_revisions,
            "executed_proposals": self.executed_proposals,
            "forced_stays": self.forced_stays,
            "no_acquisition_decisions": self.no_acquisition_decisions,
            "recurrent_commits": self.recurrent_commits,
            "held_foveal_refreshes": self.held_foveal_refreshes,
            **memory,
            "freshness_normalized_min": (
                0.0 if not count else float(self.freshness_normalized_min)
            ),
            "freshness_normalized_max": (
                0.0 if not count else float(self.freshness_normalized_max)
            ),
            "freshness_normalized_mean": float(mean),
            "freshness_normalized_variance": float(max(variance, 0.0)),
            "mean_freshness": float(mean),
            "valid_memory_fraction": (
                float(self.freshness_valid_count / count) if count else 0.0
            ),
            "memory_refresh_count": self.memory_refresh_total + self.memory.refresh_count,
            "memory_expiry_count": self.memory_expiry_total + self.memory.expiry_count,
            "requested_target_distribution": {
                f"{x},{y}": int(self.requested_target_counts[x, y])
                for x in range(8) for y in range(5)
            },
            "landing_target_distribution": {
                f"{x},{y}": int(self.landing_target_counts[x, y])
                for x in range(8) for y in range(5)
            },
            "peripheral_attempts": self.peripheral_attempts,
            "peripheral_successes": self.peripheral_successes,
            "peripheral_success_count": self.peripheral_successes,
            "peripheral_failures": self.peripheral_failures,
            "peripheral_sample_events": self.peripheral_sample_events,
            "peripheral_rng_draw_count": self.peripheral_rng_draw_count,
            "peripheral_attempt_eccentricity_sum_degrees": (
                self.peripheral_attempt_eccentricity_sum_degrees
            ),
            "peripheral_success_eccentricity_sum_degrees": (
                self.peripheral_success_eccentricity_sum_degrees
            ),
            "peripheral_success_rate": (
                self.peripheral_successes / max(self.peripheral_attempts, 1)
            ),
            "peripheral_successes_per_acquisition": (
                self.peripheral_successes / max(self.acquisitions, 1)
            ),
            "peripheral_successes_per_decision": (
                self.peripheral_successes / max(self.task_decisions, 1)
            ),
            "mean_peripheral_attempt_eccentricity_deg": (
                self.peripheral_attempt_eccentricity_sum_degrees
                / max(self.peripheral_attempts, 1)
            ),
            "mean_peripheral_success_eccentricity_deg": (
                self.peripheral_success_eccentricity_sum_degrees
                / max(self.peripheral_successes, 1)
            ),
            "opacity_attempts": opacity_attempts,
            "opacity_opportunities": opacity_opportunities,
            "opacity_successes": opacity_successes,
            "opacity_misses": opacity_attempts - opacity_successes,
            "opacity_success_rate": opacity_successes / max(opacity_attempts, 1),
            "opacity_rng_draw_count": self.opacity_rng_draw_count,
            "opacity_preview_tiles": self.opacity_preview_tiles,
            "overlap_preview_tiles": self.overlap_preview_tiles,
            "overlap_step_instances": self.overlap_step_instances,
            "opacity_positive_step_instances": (
                self.opacity_positive_step_instances
            ),
            "mean_opacity_max_alpha": (
                self.opacity_max_alpha_sum / max(self.opacity_preview_tiles, 1)
            ),
            "mean_opacity_max_probability": (
                self.opacity_max_probability_sum
                / max(self.opacity_preview_tiles, 1)
            ),
            "opacity_detected_previews_per_acquisition": (
                effective_preview_detections / max(self.acquisitions, 1)
            ),
            "opacity_detected_previews_per_decision": (
                effective_preview_detections / max(self.task_decisions, 1)
            ),
            "effective_preview_detections_per_acquisition": (
                effective_preview_detections / max(self.acquisitions, 1)
            ),
            "effective_preview_detections_per_decision": (
                effective_preview_detections / max(self.task_decisions, 1)
            ),
        }
        for step_index in range(4):
            attempts = int(self.opacity_attempts_by_step[step_index])
            successes = int(self.opacity_successes_by_step[step_index])
            opportunities = int(self.opacity_opportunities_by_step[step_index])
            effective = int(self.effective_preview_acquisitions_by_step[step_index])
            step = step_index + 1
            result[f"opacity_opportunities_h{step}"] = opportunities
            result[f"opacity_attempts_h{step}"] = attempts
            result[f"opacity_successes_h{step}"] = successes
            result[f"effective_preview_detections_h{step}"] = effective
            result[f"opacity_success_rate_h{step}"] = (
                successes / max(attempts, 1)
            )
        if hasattr(self.gaze_policy, "branch_counts"):
            result["rule_branch_counts"] = dict(self.gaze_policy.branch_counts)
        return result

    def training_state(self) -> dict[str, object]:
        policy_state = None
        if hasattr(self.gaze_policy, "training_state"):
            policy_state = self.gaze_policy.training_state()
        return {
            "decision_time": self.decision_time,
            "fixation_shape": self.fixation_shape,
            "ground_truth_policy_input": self.ground_truth_policy_input,
            "policy_observation_contract_id": (
                FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
                if self.ground_truth_policy_input
                else self.memory.observation_contract_id
            ),
            "memory": self.memory.training_state(),
            "gaze": self.gaze.training_state(),
            "current_time_ms": self.current_time_ms,
            "decision_index": self.decision_index,
            "metrics": {
                key: value
                for key, value in self.__dict__.items()
                if key in {
                    "actionnet_online_forwards", "task_decisions", "gaze_requests",
                    "acquisitions", "same_target_acquisitions", "new_target_acquisitions",
                    "prep_count", "exec_count", "dwell_count", "distance_sum_degrees",
                    "action_proposals", "proposal_revisions", "executed_proposals",
                    "forced_stays", "no_acquisition_decisions", "recurrent_commits",
                    "held_foveal_refreshes",
                    "memory_refresh_total", "memory_expiry_total",
                    "freshness_valid_count",
                    "freshness_sample_count", "freshness_normalized_sum",
                    "freshness_normalized_square_sum", "freshness_normalized_min",
                    "freshness_normalized_max",
                    "peripheral_attempts", "peripheral_successes",
                    "peripheral_failures", "peripheral_sample_events",
                    "peripheral_rng_draw_count",
                    "peripheral_attempt_eccentricity_sum_degrees",
                    "peripheral_success_eccentricity_sum_degrees",
                    "opacity_preview_tiles", "overlap_preview_tiles",
                    "overlap_step_instances", "opacity_positive_step_instances",
                    "opacity_max_alpha_sum", "opacity_max_probability_sum",
                }
            },
            "requested_target_counts": self.requested_target_counts.copy(),
            "landing_target_counts": self.landing_target_counts.copy(),
            "events": list(self.events),
            "same_target": self._same_target,
            "peripheral_seed": self.peripheral_seed,
            "peripheral_rng_state": copy.deepcopy(
                self.peripheral_rng.bit_generator.state
            ),
            "last_acquisition_metadata": copy.deepcopy(
                self.last_acquisition_metadata
            ),
            "last_peripheral_fixation": self._last_peripheral_fixation,
            "opacity_seed": self.opacity_seed,
            "opacity_rng_state": copy.deepcopy(
                self.opacity_rng.bit_generator.state
            ),
            "opacity_rng_draw_count": self.opacity_rng_draw_count,
            "opacity_attempts_by_step": self.opacity_attempts_by_step.copy(),
            "opacity_successes_by_step": self.opacity_successes_by_step.copy(),
            "opacity_opportunities_by_step": (
                self.opacity_opportunities_by_step.copy()
            ),
            "effective_preview_acquisitions_by_step": (
                self.effective_preview_acquisitions_by_step.copy()
            ),
            "gaze_policy": policy_state,
            **self.constraint_config.metadata(),
            "acquisition_action_contract_id": self.action_query_contract_id,
            "recurrent_commit_contract_id": self.recurrent_commit_contract_id,
            "actor_available_action_contract_id": ACTOR_AVAILABLE_ACTION_CONTRACT_ID,
            **(
                {
                    "held_foveal_refresh_contract_id": (
                        self.held_foveal_refresh_contract_id
                    )
                }
                if self.held_foveal_refresh_contract_id is not None
                else {}
            ),
        }

    def load_training_state(self, state: dict[str, object]) -> None:
        expected = {
            "decision_time": self.decision_time,
            "fixation_shape": self.fixation_shape,
            "ground_truth_policy_input": self.ground_truth_policy_input,
            "policy_observation_contract_id": (
                FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
                if self.ground_truth_policy_input
                else self.memory.observation_contract_id
            ),
            "acquisition_action_contract_id": self.action_query_contract_id,
            "recurrent_commit_contract_id": self.recurrent_commit_contract_id,
            "actor_available_action_contract_id": ACTOR_AVAILABLE_ACTION_CONTRACT_ID,
        }
        if "constraint_contract_id" in state:
            saved_config = constraint_config_from_metadata(state)
            if saved_config != self.constraint_config:
                raise RuntimeError(
                    "partial adapter resume constraint mismatch: "
                    f"checkpoint={saved_config.metadata()}, "
                    f"requested={self.constraint_config.metadata()}"
                )
        elif (
            self.constraint_config.peripheral.mode != "none"
            or self.constraint_config.preview_opacity.mode != "off"
        ):
            expected.update(self.constraint_config.metadata())
        elif (
            "emma_timing_contract_id" in state
            or "emma_mode" in state
            or self.emma_timing.mode != "default"
        ):
            expected.update(self.emma_timing.metadata())
        if self.held_foveal_refresh_contract_id is not None:
            expected["held_foveal_refresh_contract_id"] = (
                self.held_foveal_refresh_contract_id
            )
        mismatch = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"partial adapter resume contract mismatch: {mismatch}")
        self.memory.load_training_state(state["memory"])
        self.gaze.load_training_state(state["gaze"])
        self.current_time_ms = float(state["current_time_ms"])
        self.decision_index = int(state["decision_index"])
        for key, value in state["metrics"].items():
            if key in {
                "distance_sum_degrees", "freshness_normalized_sum",
                "freshness_normalized_square_sum", "freshness_normalized_min",
                "freshness_normalized_max",
                "peripheral_attempt_eccentricity_sum_degrees",
                "peripheral_success_eccentricity_sum_degrees",
                "opacity_max_alpha_sum", "opacity_max_probability_sum",
            }:
                value = float(value)
            else:
                value = int(value)
            setattr(self, key, value)
        for name in ("requested_target_counts", "landing_target_counts"):
            value = np.asarray(state[name], dtype=np.int64)
            if value.shape != (8, 5):
                raise RuntimeError(f"invalid Partial {name} resume state")
            setattr(self, name, value.copy())
        self.events = deque(state["events"], maxlen=256)
        self._same_target = bool(state["same_target"])
        if "peripheral_rng_state" in state:
            if int(state.get("peripheral_seed", self.peripheral_seed)) != self.peripheral_seed:
                raise RuntimeError("partial adapter peripheral seed mismatch")
            self.peripheral_rng.bit_generator.state = copy.deepcopy(
                state["peripheral_rng_state"]
            )
            self.last_acquisition_metadata = copy.deepcopy(
                state.get("last_acquisition_metadata")
            )
            saved_fixation = state.get("last_peripheral_fixation")
            if saved_fixation is None:
                if (
                    "last_peripheral_fixation" not in state
                    and self.constraint_config.peripheral.mode != "none"
                ):
                    raise RuntimeError(
                        "active peripheral resume has no fixation sampling state"
                    )
                self._last_peripheral_fixation = None
            else:
                saved_fixation = tuple(map(int, saved_fixation))
                if len(saved_fixation) != 2:
                    raise RuntimeError("invalid peripheral fixation resume state")
                self._last_peripheral_fixation = saved_fixation
        elif self.constraint_config.peripheral.mode != "none":
            raise RuntimeError("active peripheral resume has no RNG state")
        if "opacity_rng_state" in state:
            if int(state.get("opacity_seed", self.opacity_seed)) != self.opacity_seed:
                raise RuntimeError("partial adapter opacity seed mismatch")
            self.opacity_rng.bit_generator.state = copy.deepcopy(
                state["opacity_rng_state"]
            )
            self.opacity_rng_draw_count = int(
                state.get("opacity_rng_draw_count", 0)
            )
            for name in (
                "opacity_attempts_by_step", "opacity_successes_by_step",
                "opacity_opportunities_by_step",
                "effective_preview_acquisitions_by_step",
            ):
                value = np.asarray(state.get(name, np.zeros(4)), dtype=np.int64)
                if value.shape != (4,):
                    raise RuntimeError(f"invalid Partial {name} resume state")
                setattr(self, name, value.copy())
        elif self.constraint_config.preview_opacity.mode == "actual":
            raise RuntimeError("active preview opacity resume has no RNG state")
        if state.get("gaze_policy") is not None:
            if not hasattr(self.gaze_policy, "load_training_state"):
                raise RuntimeError("gaze policy cannot restore checkpoint runtime state")
            self.gaze_policy.load_training_state(state["gaze_policy"])
