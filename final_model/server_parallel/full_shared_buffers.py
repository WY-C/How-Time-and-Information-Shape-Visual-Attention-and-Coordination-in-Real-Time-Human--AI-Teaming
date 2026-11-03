"""One-step shared arrays for the server-only Full rollout backend."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes

import numpy as np


STATE_SHAPE = (8, 5, 20)


def _raw(context, ctype, count: int):
    return context.RawArray(ctype, int(count))


@dataclass
class FullSharedStepBuffers:
    n_envs: int
    learner_observations: object
    partner_observations: object
    learner_available_actions: object
    partner_available_actions: object
    learner_actions: object
    partner_actions: object
    reset_flags: object
    reset_horizons: object

    @classmethod
    def create(cls, context, n_envs: int) -> "FullSharedStepBuffers":
        count = int(n_envs)
        if count <= 0:
            raise ValueError("Full shared buffers require environments")
        state_size = int(np.prod(STATE_SHAPE))
        return cls(
            n_envs=count,
            learner_observations=_raw(
                context, ctypes.c_float, count * state_size
            ),
            partner_observations=_raw(
                context, ctypes.c_float, count * state_size
            ),
            learner_available_actions=_raw(
                context, ctypes.c_float, count * 6
            ),
            partner_available_actions=_raw(
                context, ctypes.c_float, count * 6
            ),
            learner_actions=_raw(context, ctypes.c_int64, count),
            partner_actions=_raw(context, ctypes.c_int64, count),
            reset_flags=_raw(context, ctypes.c_uint8, count),
            reset_horizons=_raw(context, ctypes.c_int8, count),
        )

    @staticmethod
    def _view(raw, dtype, shape):
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def views(self) -> dict[str, np.ndarray]:
        count = self.n_envs
        return {
            "learner_observations": self._view(
                self.learner_observations,
                np.float32,
                (count, *STATE_SHAPE),
            ),
            "partner_observations": self._view(
                self.partner_observations,
                np.float32,
                (count, *STATE_SHAPE),
            ),
            "learner_available_actions": self._view(
                self.learner_available_actions, np.float32, (count, 6)
            ),
            "partner_available_actions": self._view(
                self.partner_available_actions, np.float32, (count, 6)
            ),
            "learner_actions": self._view(
                self.learner_actions, np.int64, (count,)
            ),
            "partner_actions": self._view(
                self.partner_actions, np.int64, (count,)
            ),
            "reset_flags": self._view(
                self.reset_flags, np.uint8, (count,)
            ),
            "reset_horizons": self._view(
                self.reset_horizons, np.int8, (count,)
            ),
        }

    def clear_resets(self) -> None:
        self.views()["reset_flags"].fill(0)
