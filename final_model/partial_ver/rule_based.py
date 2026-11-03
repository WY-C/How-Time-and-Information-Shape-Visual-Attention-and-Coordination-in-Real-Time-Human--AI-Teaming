"""Fixed stochastic, memory-only rule gaze for canonical Partial runtime."""

from __future__ import annotations

import copy
import math

import numpy as np

from final_model.partial_ver.gaze import ReferenceGeometry, acquisition_tiles, validate_tile


RULE_BASED_GAZE_POLICY_ID = "stochastic_memory_only_rule_gaze_v2"
DEFAULT_RULE_GAZE_PROBABILITIES = {
    "task": 0.60,
    "stay": 0.20,
    "stale": 0.15,
    "random": 0.05,
}
NEUTRAL_RULE_GAZE_PROBABILITIES = {
    "task": 0.00,
    "stay": 0.20,
    "stale": 0.00,
    "random": 0.80,
}


def validate_rule_probabilities(probabilities) -> dict[str, float]:
    expected = tuple(DEFAULT_RULE_GAZE_PROBABILITIES)
    values = {key: float(probabilities[key]) for key in expected}
    if set(probabilities) != set(expected) or any(value < 0.0 for value in values.values()):
        raise ValueError("rule gaze probabilities must define non-negative task/stay/stale/random")
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("rule gaze probabilities must sum to exactly one")
    return values


class RuleBasedGazePolicy:
    """Select using only static layout, current fixation, and legitimate memory."""

    policy_id = RULE_BASED_GAZE_POLICY_ID
    _SHAPE = (8, 5, 25)

    def __init__(
        self,
        fixation_shape: str,
        *,
        seed: int = 0,
        probabilities=None,
    ) -> None:
        acquisition_tiles((0, 0), fixation_shape)
        self.fixation_shape = fixation_shape
        self.seed = int(seed)
        self.probabilities = validate_rule_probabilities(
            DEFAULT_RULE_GAZE_PROBABILITIES if probabilities is None else probabilities
        )
        self.candidates = tuple((x, y) for y in range(5) for x in range(8))
        self.target_index = {tile: index for index, tile in enumerate(self.candidates)}
        self.footprint_masks = np.zeros((40, 8, 5), dtype=bool)
        for index, target in enumerate(self.candidates):
            for tile in acquisition_tiles(target, fixation_shape):
                self.footprint_masks[index][tile] = True
        geometry = ReferenceGeometry()
        self.visual_angles = np.asarray(
            [
                [geometry.visual_angle_degrees(source, target) for target in self.candidates]
                for source in self.candidates
            ],
            dtype=np.float32,
        )
        np.fill_diagonal(self.visual_angles, 0.0)
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> None:
        if not hasattr(self, "selection_count"):
            self.selection_count = 0
            self.branch_counts = {key: 0 for key in self.probabilities}
            self.events = []
        self.last_branch = "startup"
        self.last_requested_target = None

    @staticmethod
    def _task_priority(memory: np.ndarray) -> np.ndarray:
        known = memory[..., 20] > 0.0
        priority = np.zeros((8, 5), dtype=np.float32)
        priority += 5.0 * (memory[..., 10] != 0.0)
        priority += 3.0 * (memory[..., 12] != 0.0)
        priority += 3.0 * (memory[..., 13] != 0.0)
        priority += 4.0 * (memory[..., 14] != 0.0)
        priority += 5.0 * (np.any(memory[..., 15:20] != 0.0, axis=-1) & known)
        priority += 2.0 * (np.any(memory[..., 0:2] != 0.0, axis=-1) & known)
        priority += 2.0 * (np.any(memory[..., 21:25] != 0.0, axis=-1) & known)
        return priority

    def _argmax_target(self, scores: np.ndarray, current: tuple[int, int]):
        current_index = self.target_index[current]
        return self.candidates[
            max(
                range(40),
                key=lambda index: (
                    float(scores[index]),
                    -float(self.visual_angles[current_index, index]),
                    -index,
                ),
            )
        ]

    def _task_target(self, memory, current):
        priority = self._task_priority(memory)
        scores = np.sum(self.footprint_masks * priority[None], axis=(1, 2))
        return self._argmax_target(scores, current)

    def _stale_target(self, memory, current):
        priority = self._task_priority(memory)
        task_region = priority > 0.0
        if not np.any(task_region):
            task_region = np.any(memory[..., 10:15] != 0.0, axis=-1)
        if not np.any(task_region):
            task_region = np.ones((8, 5), dtype=bool)
        semantic_freshness = np.clip(memory[..., 20] / 255.0, 0.0, 1.0)
        staleness = (1.0 - semantic_freshness) * task_region
        scores = np.sum(self.footprint_masks * staleness[None], axis=(1, 2))
        return self._argmax_target(scores, current)

    def __call__(self, memory, fixation, context) -> tuple[int, int]:
        partial = np.asarray(memory, dtype=np.float32)
        if partial.shape != self._SHAPE:
            raise ValueError(f"Partial memory must be {self._SHAPE}, got {partial.shape}")
        current = validate_tile(fixation)
        keys = tuple(self.probabilities)
        branch = str(self.rng.choice(keys, p=[self.probabilities[key] for key in keys]))
        if branch == "stay":
            target = current
        elif branch == "random":
            target = self.candidates[int(self.rng.integers(len(self.candidates)))]
        elif branch == "stale":
            target = self._stale_target(partial, current)
        else:
            target = self._task_target(partial, current)
        self.selection_count += 1
        self.branch_counts[branch] += 1
        self.last_branch = branch
        self.last_requested_target = target
        self.events.append(
            {
                "branch": branch,
                "requested_target": target,
                "current_fixation": current,
                "freshness_semantic_min": float(partial[..., 20].min() / 255.0),
                "freshness_semantic_max": float(partial[..., 20].max() / 255.0),
                "freshness_semantic_mean": float(partial[..., 20].mean() / 255.0),
                "timing_context": np.asarray(context, dtype=np.float32).tolist(),
            }
        )
        return target

    def record_landing(self, *, landing_target, acquisition_success: bool, same_target: bool):
        if self.events:
            self.events[-1].update(
                {
                    "landing_target": tuple(landing_target),
                    "acquisition_success": bool(acquisition_success),
                    "same_target": bool(same_target),
                }
            )

    def training_state(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "fixation_shape": self.fixation_shape,
            "seed": self.seed,
            "probabilities": dict(self.probabilities),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "selection_count": self.selection_count,
            "branch_counts": dict(self.branch_counts),
            "last_branch": self.last_branch,
            "last_requested_target": self.last_requested_target,
            "events": copy.deepcopy(self.events[-128:]),
        }

    def load_training_state(self, state: dict[str, object]) -> None:
        if (
            state.get("policy_id") != self.policy_id
            or state.get("fixation_shape") != self.fixation_shape
            or int(state.get("seed", -1)) != self.seed
            or validate_rule_probabilities(state.get("probabilities", {}))
            != self.probabilities
        ):
            raise RuntimeError("rule-based gaze resume contract mismatch")
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self.selection_count = int(state["selection_count"])
        self.branch_counts = {
            key: int(state["branch_counts"][key]) for key in self.probabilities
        }
        self.last_branch = str(state["last_branch"])
        target = state["last_requested_target"]
        self.last_requested_target = None if target is None else validate_tile(target)
        self.events = copy.deepcopy(list(state["events"]))
