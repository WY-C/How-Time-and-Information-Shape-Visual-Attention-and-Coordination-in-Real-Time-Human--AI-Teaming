"""Preallocated one-decision shared-memory buffers."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes

import numpy as np

from final_model.server_parallel.protocol import MAX_ACQUISITIONS_PER_DECISION


OBSERVATION_SHAPE = (8, 5, 25)
CRITIC_SHAPE = (8, 5, 40)
PARTNER_OBSERVATION_SHAPE = (8, 5, 20)


def _raw(context, ctype, count: int):
    return context.RawArray(ctype, int(count))


@dataclass
class SharedDecisionBuffers:
    n_envs: int
    acquisition_counts: object
    gaze_request_counts: object
    same_target_acquisition_counts: object
    new_target_acquisition_counts: object
    valid_memory_tile_counts: object
    freshness_sample_counts: object
    freshness_normalized_sums: object
    saccade_distance_sums_degrees: object
    acquisition_observations: object
    acquisition_timestamps: object
    final_observations: object
    critic_observations: object
    available_actions: object
    partner_observations: object
    partner_available_actions: object
    learner_actions: object
    partner_actions: object
    rewards: object
    deliveries: object
    joint_collisions: object
    dones: object
    reset_flags: object
    reset_decision_times: object
    reset_horizons: object
    reset_fixation_codes: object
    next_observations: object
    next_critic_observations: object
    next_available_actions: object

    @classmethod
    def create(cls, context, n_envs: int) -> "SharedDecisionBuffers":
        count = int(n_envs)
        if count <= 0:
            raise ValueError("shared decision buffers require environments")
        observation_size = int(np.prod(OBSERVATION_SHAPE))
        critic_size = int(np.prod(CRITIC_SHAPE))
        partner_size = int(np.prod(PARTNER_OBSERVATION_SHAPE))
        return cls(
            n_envs=count,
            acquisition_counts=_raw(context, ctypes.c_int32, count),
            gaze_request_counts=_raw(context, ctypes.c_int32, count),
            same_target_acquisition_counts=_raw(context, ctypes.c_int32, count),
            new_target_acquisition_counts=_raw(context, ctypes.c_int32, count),
            valid_memory_tile_counts=_raw(context, ctypes.c_int32, count),
            freshness_sample_counts=_raw(context, ctypes.c_int32, count),
            freshness_normalized_sums=_raw(context, ctypes.c_double, count),
            saccade_distance_sums_degrees=_raw(
                context, ctypes.c_double, count
            ),
            acquisition_observations=_raw(
                context,
                ctypes.c_float,
                count * MAX_ACQUISITIONS_PER_DECISION * observation_size,
            ),
            acquisition_timestamps=_raw(
                context, ctypes.c_double, count * MAX_ACQUISITIONS_PER_DECISION
            ),
            final_observations=_raw(context, ctypes.c_float, count * observation_size),
            critic_observations=_raw(context, ctypes.c_float, count * critic_size),
            available_actions=_raw(context, ctypes.c_float, count * 6),
            partner_observations=_raw(context, ctypes.c_float, count * partner_size),
            partner_available_actions=_raw(context, ctypes.c_float, count * 6),
            learner_actions=_raw(context, ctypes.c_int64, count),
            partner_actions=_raw(context, ctypes.c_int64, count),
            rewards=_raw(context, ctypes.c_double, count * 4),
            deliveries=_raw(context, ctypes.c_int32, count),
            joint_collisions=_raw(context, ctypes.c_uint8, count),
            dones=_raw(context, ctypes.c_uint8, count),
            reset_flags=_raw(context, ctypes.c_uint8, count),
            reset_decision_times=_raw(context, ctypes.c_int32, count),
            reset_horizons=_raw(context, ctypes.c_int32, count),
            reset_fixation_codes=_raw(context, ctypes.c_int8, count),
            next_observations=_raw(context, ctypes.c_float, count * observation_size),
            next_critic_observations=_raw(context, ctypes.c_float, count * critic_size),
            next_available_actions=_raw(context, ctypes.c_float, count * 6),
        )

    @staticmethod
    def _view(raw, dtype, shape):
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def views(self) -> dict[str, np.ndarray]:
        n = self.n_envs
        maximum = MAX_ACQUISITIONS_PER_DECISION
        return {
            "acquisition_counts": self._view(self.acquisition_counts, np.int32, (n,)),
            "gaze_request_counts": self._view(
                self.gaze_request_counts, np.int32, (n,)
            ),
            "same_target_acquisition_counts": self._view(
                self.same_target_acquisition_counts, np.int32, (n,)
            ),
            "new_target_acquisition_counts": self._view(
                self.new_target_acquisition_counts, np.int32, (n,)
            ),
            "valid_memory_tile_counts": self._view(
                self.valid_memory_tile_counts, np.int32, (n,)
            ),
            "freshness_sample_counts": self._view(
                self.freshness_sample_counts, np.int32, (n,)
            ),
            "freshness_normalized_sums": self._view(
                self.freshness_normalized_sums, np.float64, (n,)
            ),
            "saccade_distance_sums_degrees": self._view(
                self.saccade_distance_sums_degrees, np.float64, (n,)
            ),
            "acquisition_observations": self._view(
                self.acquisition_observations,
                np.float32,
                (n, maximum, *OBSERVATION_SHAPE),
            ),
            "acquisition_timestamps": self._view(
                self.acquisition_timestamps, np.float64, (n, maximum)
            ),
            "final_observations": self._view(
                self.final_observations, np.float32, (n, *OBSERVATION_SHAPE)
            ),
            "critic_observations": self._view(
                self.critic_observations, np.float32, (n, *CRITIC_SHAPE)
            ),
            "available_actions": self._view(
                self.available_actions, np.float32, (n, 6)
            ),
            "partner_observations": self._view(
                self.partner_observations,
                np.float32,
                (n, *PARTNER_OBSERVATION_SHAPE),
            ),
            "partner_available_actions": self._view(
                self.partner_available_actions, np.float32, (n, 6)
            ),
            "learner_actions": self._view(self.learner_actions, np.int64, (n,)),
            "partner_actions": self._view(self.partner_actions, np.int64, (n,)),
            "rewards": self._view(self.rewards, np.float64, (n, 4)),
            "deliveries": self._view(self.deliveries, np.int32, (n,)),
            "joint_collisions": self._view(
                self.joint_collisions, np.uint8, (n,)
            ),
            "dones": self._view(self.dones, np.uint8, (n,)),
            "reset_flags": self._view(self.reset_flags, np.uint8, (n,)),
            "reset_decision_times": self._view(
                self.reset_decision_times, np.int32, (n,)
            ),
            "reset_horizons": self._view(self.reset_horizons, np.int32, (n,)),
            "reset_fixation_codes": self._view(
                self.reset_fixation_codes, np.int8, (n,)
            ),
            "next_observations": self._view(
                self.next_observations, np.float32, (n, *OBSERVATION_SHAPE)
            ),
            "next_critic_observations": self._view(
                self.next_critic_observations, np.float32, (n, *CRITIC_SHAPE)
            ),
            "next_available_actions": self._view(
                self.next_available_actions, np.float32, (n, 6)
            ),
        }

    def clear_decision_metadata(self) -> None:
        views = self.views()
        views["acquisition_counts"].fill(0)
        views["gaze_request_counts"].fill(0)
        views["same_target_acquisition_counts"].fill(0)
        views["new_target_acquisition_counts"].fill(0)
        views["valid_memory_tile_counts"].fill(0)
        views["freshness_sample_counts"].fill(0)
        views["freshness_normalized_sums"].fill(0.0)
        views["saccade_distance_sums_degrees"].fill(0.0)
        views["acquisition_timestamps"].fill(np.nan)
        views["joint_collisions"].fill(0)
        views["dones"].fill(0)
        views["reset_flags"].fill(0)
