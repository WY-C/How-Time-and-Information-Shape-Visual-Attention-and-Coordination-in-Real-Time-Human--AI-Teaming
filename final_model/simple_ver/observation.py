"""Full observation adapter over the shared direct backend."""

from __future__ import annotations

from typing import Callable

import numpy as np

from final_model.common.overcooked_backend import DirectOvercookedBackend


OBSERVATION_SCALE = 255.0
OBSERVATION_CONTRACT_ID = "full_state20_freshness255_displayed_preview4_v2"


def preview4(planes: np.ndarray, horizon: int) -> np.ndarray:
    source = np.asarray(planes, dtype=np.float32)
    if source.shape != (4, 8, 5):
        raise ValueError(f"predictor planes must be (4,8,5), got {source.shape}")
    result = np.moveaxis(source, 0, -1).copy()
    result[..., int(horizon):] = 0.0
    return np.ascontiguousarray(result, dtype=np.float32)


def build_full_observation(state20: np.ndarray, planes: np.ndarray, horizon: int) -> np.ndarray:
    state = np.asarray(state20, dtype=np.float32)
    if state.shape != (8, 5, 20):
        raise ValueError("canonical state must be (8,5,20)")
    result = np.concatenate(
        (
            state,
            np.full((8, 5, 1), OBSERVATION_SCALE, dtype=np.float32),
            preview4(planes, horizon),
        ),
        axis=-1,
    )
    return np.ascontiguousarray(result, dtype=np.float32)


class FullObservationAdapter:
    def __init__(self, backend: DirectOvercookedBackend):
        self.backend = backend
        self.actionnet_online_forwards = 0
        self.task_decisions = 0

    def reset_episode(self) -> None:
        self.backend.reset()

    def current(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        partner, learner = self.backend.canonical_pair()
        actor = build_full_observation(learner, self.backend.preview_planes(), self.backend.horizon)
        critic = np.ascontiguousarray(np.concatenate((learner, partner), axis=-1), dtype=np.float32)
        return actor, critic, self.backend.learner_available_actions()

    def prepare_decision(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.current()

    def commit_decision(self, action: int):
        self.actionnet_online_forwards += 1
        self.task_decisions += 1
        if self.actionnet_online_forwards != self.task_decisions:
            raise RuntimeError("ActionNet must run exactly once per decision")
        return self.backend.step(int(action))

    def commit_decision_with_partner(
        self,
        action: int,
        partner_action: int,
        partner_hash_before: str,
    ):
        self.actionnet_online_forwards += 1
        self.task_decisions += 1
        if self.actionnet_online_forwards != self.task_decisions:
            raise RuntimeError("ActionNet must run exactly once per decision")
        return self.backend.step_with_partner_action(
            int(action),
            int(partner_action),
            partner_hash_before=partner_hash_before,
        )

    def run_decision(self, policy: Callable[[np.ndarray, np.ndarray, np.ndarray], int]):
        observation, critic, available = self.prepare_decision()
        action = int(policy(observation, critic, available))
        return observation, critic, available, action, self.commit_decision(action)

    def training_state(self) -> dict[str, int]:
        return {
            "actionnet_online_forwards": self.actionnet_online_forwards,
            "task_decisions": self.task_decisions,
        }

    def load_training_state(self, state: dict[str, int]) -> None:
        self.actionnet_online_forwards = int(state["actionnet_online_forwards"])
        self.task_decisions = int(state["task_decisions"])
