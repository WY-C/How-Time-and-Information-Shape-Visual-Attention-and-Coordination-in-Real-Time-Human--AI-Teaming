"""Deterministic exactly-balanced per-episode horizon assignment."""

from __future__ import annotations

from collections import Counter
import random
from typing import Any, Iterable

import numpy as np


SHARED_HORIZONS = (0, 2, 4)
HORIZON_MODE_FIXED = "fixed"
HORIZON_MODE_EPISODE_BALANCED = "episode_balanced"


def horizon_one_hot(horizon: int) -> np.ndarray:
    value = int(horizon)
    if value not in SHARED_HORIZONS:
        raise ValueError(f"horizon must be one of {SHARED_HORIZONS}")
    result = np.zeros(3, dtype=np.float32)
    result[SHARED_HORIZONS.index(value)] = 1.0
    return result


class BalancedEpisodeHorizonSampler:
    """Shuffle each H0/H2/H4 block with an isolated seeded RNG."""

    def __init__(self, *, seed: int, horizons: Iterable[int] = SHARED_HORIZONS):
        self.horizons = tuple(int(value) for value in horizons)
        if self.horizons != SHARED_HORIZONS:
            raise ValueError(f"shared horizons must be exactly {SHARED_HORIZONS}")
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._block: list[int] = []
        self._index = 0
        self.counts = Counter({value: 0 for value in self.horizons})

    def next(self) -> int:
        if self._index == len(self._block):
            self._block = list(self.horizons)
            self._rng.shuffle(self._block)
            self._index = 0
        value = int(self._block[self._index])
        self._index += 1
        self.counts[value] += 1
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "horizons": list(self.horizons),
            "rng_state": self._rng.getstate(),
            "pending_block": list(self._block),
            "pending_index": self._index,
            "episode_counts": {
                int(horizon): int(self.counts[horizon]) for horizon in self.horizons
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {"seed": self.seed, "horizons": list(self.horizons)}
        mismatch = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"shared horizon sampler contract mismatch: {mismatch}")
        block = [int(value) for value in state["pending_block"]]
        index = int(state["pending_index"])
        if sorted(block) != list(self.horizons) or not 0 <= index <= len(block):
            raise RuntimeError("invalid shared horizon sampler pending block")
        counts = {int(key): int(value) for key, value in state["episode_counts"].items()}
        if set(counts) != set(self.horizons) or any(value < 0 for value in counts.values()):
            raise RuntimeError("invalid shared horizon sampler episode counts")
        if sum(counts.values()) % len(self.horizons) != index % len(self.horizons):
            raise RuntimeError("shared horizon sampler count/index mismatch")
        self._rng.setstate(state["rng_state"])
        self._block = block
        self._index = index
        self.counts = Counter(counts)

