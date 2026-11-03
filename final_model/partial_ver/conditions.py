"""Generic isolated shuffled-block scheduler for Partial episode conditions."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from final_model.partial_ver.gaze import FIXATION_SHAPES


CONDITION_SCHEDULER_ID = "partial_cartesian_shuffled_episode_blocks_v1"
ALLOWED_DECISION_TIMES = (200, 400, 600)


@dataclass(frozen=True, order=True)
class PartialCondition:
    decision_time: int
    horizon: int
    fixation_shape: str


class PartialConditionScheduler:
    def __init__(
        self,
        *,
        decision_times,
        horizons,
        fixation_shapes,
        seed: int,
    ) -> None:
        self.decision_times = tuple(int(value) for value in decision_times)
        self.horizons = tuple(int(value) for value in horizons)
        self.fixation_shapes = tuple(str(value) for value in fixation_shapes)
        if not self.decision_times or not self.horizons or not self.fixation_shapes:
            raise ValueError("Partial condition axes must be non-empty")
        if len(set(self.decision_times)) != len(self.decision_times) or any(
            value not in ALLOWED_DECISION_TIMES for value in self.decision_times
        ):
            raise ValueError(f"decision times must be unique values in {ALLOWED_DECISION_TIMES}")
        if len(set(self.horizons)) != len(self.horizons) or any(
            value not in (0, 2, 4) for value in self.horizons
        ):
            raise ValueError("horizons must be unique values in (0,2,4)")
        if len(set(self.fixation_shapes)) != len(self.fixation_shapes) or any(
            value not in FIXATION_SHAPES for value in self.fixation_shapes
        ):
            raise ValueError(f"fixation shapes must be unique values in {FIXATION_SHAPES}")
        self.conditions = tuple(
            PartialCondition(*values)
            for values in product(
                self.decision_times, self.horizons, self.fixation_shapes
            )
        )
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.block_index = -1
        self.position_in_block = 0
        self.current_block = ()
        self.assignment_counts = {condition: 0 for condition in self.conditions}

    @property
    def shared(self) -> bool:
        return len(self.conditions) > 1

    def _start_block(self) -> None:
        order = self.rng.permutation(len(self.conditions))
        self.current_block = tuple(self.conditions[int(index)] for index in order)
        self.block_index += 1
        self.position_in_block = 0

    def next(self) -> PartialCondition:
        if not self.current_block or self.position_in_block >= len(self.current_block):
            self._start_block()
        condition = self.current_block[self.position_in_block]
        self.position_in_block += 1
        self.assignment_counts[condition] += 1
        return condition

    def state_dict(self) -> dict[str, object]:
        return {
            "scheduler_id": CONDITION_SCHEDULER_ID,
            "decision_times": list(self.decision_times),
            "horizons": list(self.horizons),
            "fixation_shapes": list(self.fixation_shapes),
            "seed": self.seed,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "block_index": self.block_index,
            "position_in_block": self.position_in_block,
            "current_block": [asdict(value) for value in self.current_block],
            "assignment_counts": [
                {**asdict(condition), "count": count}
                for condition, count in self.assignment_counts.items()
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {
            "scheduler_id": CONDITION_SCHEDULER_ID,
            "decision_times": list(self.decision_times),
            "horizons": list(self.horizons),
            "fixation_shapes": list(self.fixation_shapes),
            "seed": self.seed,
        }
        mismatch = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"Partial condition scheduler contract mismatch: {mismatch}")
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self.block_index = int(state["block_index"])
        self.position_in_block = int(state["position_in_block"])
        self.current_block = tuple(
            PartialCondition(**value) for value in state["current_block"]
        )
        if self.position_in_block < 0 or self.position_in_block > len(self.current_block):
            raise RuntimeError("invalid Partial condition scheduler block position")
        counts = {
            PartialCondition(
                int(value["decision_time"]),
                int(value["horizon"]),
                str(value["fixation_shape"]),
            ): int(value["count"])
            for value in state["assignment_counts"]
        }
        if set(counts) != set(self.conditions):
            raise RuntimeError("invalid Partial condition scheduler counts")
        self.assignment_counts = counts
