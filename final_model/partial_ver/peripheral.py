"""Canonical stochastic peripheral perception, independent of memory encoding."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from final_model.partial_ver.gaze import ReferenceGeometry, acquisition_tiles
from final_model.server_parallel.gaze_v1 import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GAZE_TILES,
    tile_to_action_index,
)


LEGACY_PERIPHERAL_PERCEPTION_CONTRACT_ID = (
    "canonical_plus_exact_simultaneous_peripheral_snapshot_v1"
)
PERIPHERAL_PERCEPTION_CONTRACT_ID = (
    "canonical_plus_new_fixation_first_acquisition_peripheral_snapshot_v2"
)
PERIPHERAL_PROBABILITY_CONTRACT_ID = (
    "gaussian_visual_eccentricity_independent_bernoulli_v1"
)
PERIPHERAL_RNG_CONTRACT_ID = "numpy_pcg64_per_adapter_seed_offset_v1"
PERIPHERAL_RNG_SEED_OFFSET = 97_000_021
P1_DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
PERIPHERAL_EXTENTS = ("diag3x3", "diamond13")
PLUS_EXACT_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
DIAG3X3_PERIPHERAL_OFFSETS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
DIAMOND13_PERIPHERAL_OFFSETS = (
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def peripheral_detection_probability(
    eccentricity_deg: float | Sequence[float] | np.ndarray,
    sigma_deg: float,
) -> float | np.ndarray:
    sigma = float(sigma_deg)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("peripheral sigma must be finite and positive")
    eccentricity = np.asarray(eccentricity_deg, dtype=np.float64)
    if np.any(~np.isfinite(eccentricity)) or np.any(eccentricity < 0.0):
        raise ValueError("peripheral eccentricity must be finite and nonnegative")
    probability = np.clip(
        np.exp(-0.5 * np.square(eccentricity / sigma)), 0.0, 1.0
    )
    if np.any(~np.isfinite(probability)):
        raise RuntimeError("non-finite peripheral detection probability")
    if probability.ndim == 0:
        return float(probability)
    return probability


def p1_peripheral_candidates(
    fixation: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    x, y = GAZE_TILES[tile_to_action_index(fixation)]
    candidates = [
        (x + dx, y + dy)
        for dx, dy in P1_DIAGONAL_OFFSETS
        if 0 <= x + dx < BOARD_WIDTH and 0 <= y + dy < BOARD_HEIGHT
    ]
    return tuple(sorted(candidates, key=tile_to_action_index))


def p2_peripheral_candidates(
    fixation: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    plus = set(acquisition_tiles(fixation, "plus"))
    return tuple(tile for tile in GAZE_TILES if tile not in plus)


def extent_peripheral_offsets(extent: str) -> tuple[tuple[int, int], ...]:
    if extent == "diag3x3":
        return DIAG3X3_PERIPHERAL_OFFSETS
    if extent == "diamond13":
        return DIAMOND13_PERIPHERAL_OFFSETS
    raise ValueError(f"peripheral extent must be one of {PERIPHERAL_EXTENTS}")


def extent_peripheral_candidates(
    fixation: Sequence[int], extent: str
) -> tuple[tuple[int, int], ...]:
    x, y = GAZE_TILES[tile_to_action_index(fixation)]
    candidates = {
        (x + dx, y + dy)
        for dx, dy in extent_peripheral_offsets(extent)
        if 0 <= x + dx < BOARD_WIDTH and 0 <= y + dy < BOARD_HEIGHT
    }
    return tuple(sorted(candidates, key=tile_to_action_index))


def peripheral_candidates(
    mode: str, fixation: Sequence[int], extent: str | None = None,
) -> tuple[tuple[int, int], ...]:
    if mode == "none":
        if extent is not None:
            raise ValueError("inactive peripheral mode must not have an extent")
        return ()
    if extent is not None:
        candidates = extent_peripheral_candidates(fixation, extent)
    elif mode == "local":
        candidates = p1_peripheral_candidates(fixation)
    elif mode == "global":
        candidates = p2_peripheral_candidates(fixation)
    else:
        raise ValueError("peripheral mode must be none, local, or global")
    plus = set(acquisition_tiles(fixation, "plus"))
    if plus.intersection(candidates):
        raise RuntimeError("peripheral candidates overlap canonical PLUS")
    if tuple(sorted(candidates, key=tile_to_action_index)) != candidates:
        raise RuntimeError("peripheral candidates are not in canonical x-major order")
    return candidates


@dataclass(frozen=True)
class PeripheralDetection:
    candidate_tiles: tuple[tuple[int, int], ...]
    eccentricities_deg: tuple[float, ...]
    probabilities: tuple[float, ...]
    success_mask: tuple[bool, ...]

    @property
    def success_tiles(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            tile for tile, success in zip(self.candidate_tiles, self.success_mask)
            if success
        )

    @property
    def failure_tiles(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            tile for tile, success in zip(self.candidate_tiles, self.success_mask)
            if not success
        )


def sample_peripheral_detection(
    *, mode: str, sigma_deg: float | None, fixation: Sequence[int],
    rng: np.random.Generator, geometry: ReferenceGeometry | None = None,
    extent: str | None = None,
) -> PeripheralDetection:
    candidates = peripheral_candidates(mode, fixation, extent)
    if mode == "none":
        if sigma_deg is not None:
            raise ValueError("inactive peripheral mode must not have sigma")
        return PeripheralDetection((), (), (), ())
    if sigma_deg is None:
        raise ValueError("active peripheral mode requires sigma")
    geometry = ReferenceGeometry() if geometry is None else geometry
    eccentricities = tuple(
        float(geometry.visual_angle_degrees(fixation, tile))
        for tile in candidates
    )
    probabilities_array = np.asarray(
        peripheral_detection_probability(eccentricities, sigma_deg),
        dtype=np.float64,
    )
    draws = np.asarray(rng.random(len(candidates)), dtype=np.float64)
    if draws.shape != probabilities_array.shape:
        raise RuntimeError("peripheral RNG returned an invalid draw shape")
    return PeripheralDetection(
        candidates,
        eccentricities,
        tuple(float(value) for value in probabilities_array),
        tuple(bool(value) for value in draws < probabilities_array),
    )
