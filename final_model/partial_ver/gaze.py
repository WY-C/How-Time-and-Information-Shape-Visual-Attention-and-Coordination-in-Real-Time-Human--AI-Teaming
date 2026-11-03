"""Portable committed event-driven gaze and exact acquisition footprints."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np


FIXATION_SHAPES = ("3x3", "2x2", "plus")
PREP_MS = 135.0
EXEC_BASE_MS = 70.0
EXEC_MS_PER_DEGREE = 2.0
DWELL_MS = 60.0
EMMA_TIMING_CONTRACT_ID = "emma_distance_dependent_execution_v1"
HUMAN_STOCHASTIC_EMMA_TIMING_CONTRACT_ID = "emma_human_stochastic_fixation_v2"
DEFAULT_ENCODING_EMMA_TIMING_CONTRACT_ID = (
    "emma_default_movement_parallel_visual_encoding_v1"
)
RELAXED_SACCADE_EMMA_TIMING_CONTRACT_ID = (
    "emma_half_saccade_and_encoding_parallel_visual_encoding_v2"
)
RELAXED_SACCADE_LATENCY_SCALE = 0.5
RELAXED_SACCADE_ENCODING_SCALE = 0.5
DEFAULT_ENCODING_K_SECONDS = 0.006
DEFAULT_ENCODING_K = 0.4
DEFAULT_ENCODING_FAMILIARITY = 0.1
VISUAL_ENCODING_EMMA_MODES = frozenset({
    "default_encoding", "relaxed_saccade"
})
DEFAULT_ENCODING_TEMPORAL_SEMANTICS = (
    "encoding starts with eye-movement preparation; at landing unfinished "
    "encoding is proportionally rescaled at zero eccentricity; the atomic "
    "fixation snapshot is acquired only after landing and encoding completion"
)
# Fit to 30,668 valid consecutive human fixation pairs after subtracting the
# tuned EMMA preparation and distance-dependent execution time.
HUMAN_DWELL_DISTRIBUTION = "floor_inflated_capped_lognormal"
HUMAN_DWELL_FLOOR_PROBABILITY = 0.06068214425590192
HUMAN_DWELL_FLOOR_MS = 60.0
HUMAN_DWELL_LOG_MEAN = 5.844786707880467
HUMAN_DWELL_LOG_SIGMA = 1.0338073724375734
HUMAN_DWELL_CAP_MS = 4000.0
HUMAN_DWELL_RNG_CONTRACT_ID = "numpy_pcg64_human_dwell_per_adapter_seed_offset_v1"
HUMAN_DWELL_RNG_SEED_OFFSET = 109_000_033
HUMAN_DWELL_CALIBRATION_ID = "counter_circuit_valid_fixation_residual_30668_v1"

MONITOR_DIAGONAL_MM = 685.8
MONITOR_ASPECT = (16.0, 9.0)
MONITOR_RASTER_PX = (2560.0, 1440.0)
VIEWPORT_TILE_PX = (90.0, 90.0)
EYE_PROJECTION_VIEWPORT_PX = (360.0, 225.0)
VIEWING_DISTANCE_MM = 600.0


@dataclass(frozen=True)
class EmmaTimingConfig:
    mode: str
    prep_ms: float
    exec_intercept_ms: float
    exec_slope_ms_per_degree: float
    dwell_ms: float
    contract_id: str = EMMA_TIMING_CONTRACT_ID
    dwell_distribution: str = "fixed"
    dwell_floor_probability: float = 0.0
    dwell_floor_ms: float = 0.0
    dwell_log_mean: float = 0.0
    dwell_log_sigma: float = 0.0
    dwell_cap_ms: float = 0.0
    encoding_K_seconds: float = 0.0
    encoding_k: float = 0.0
    encoding_f: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {
            "default", "tuned", "human_stochastic", *VISUAL_ENCODING_EMMA_MODES
        }:
            raise ValueError(
                "EMMA mode must be default, tuned, human_stochastic, or "
                "a visual-encoding mode"
            )
        expected_contract = (
            HUMAN_STOCHASTIC_EMMA_TIMING_CONTRACT_ID
            if self.mode == "human_stochastic"
            else RELAXED_SACCADE_EMMA_TIMING_CONTRACT_ID
            if self.mode == "relaxed_saccade"
            else DEFAULT_ENCODING_EMMA_TIMING_CONTRACT_ID
            if self.mode == "default_encoding"
            else EMMA_TIMING_CONTRACT_ID
        )
        if self.contract_id != expected_contract:
            raise ValueError("unsupported EMMA timing contract")
        for name in (
            "prep_ms", "exec_intercept_ms",
            "exec_slope_ms_per_degree", "dwell_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mode == "human_stochastic":
            if self.dwell_distribution != HUMAN_DWELL_DISTRIBUTION:
                raise ValueError("unsupported human dwell distribution")
            if not 0.0 <= self.dwell_floor_probability <= 1.0:
                raise ValueError("dwell_floor_probability must be in [0, 1]")
            for name in (
                "dwell_floor_ms", "dwell_log_mean", "dwell_log_sigma",
                "dwell_cap_ms",
            ):
                value = float(getattr(self, name))
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
        elif self.mode in VISUAL_ENCODING_EMMA_MODES:
            expected = (
                DEFAULT_ENCODING_K_SECONDS * (
                    RELAXED_SACCADE_ENCODING_SCALE
                    if self.mode == "relaxed_saccade"
                    else 1.0
                ),
                DEFAULT_ENCODING_K,
                DEFAULT_ENCODING_FAMILIARITY,
            )
            actual = (
                self.encoding_K_seconds, self.encoding_k, self.encoding_f
            )
            if actual != expected:
                raise ValueError(
                    "visual-encoding modes require canonical encoding constants"
                )
            if self.dwell_ms != 0.0:
                raise ValueError(
                    "visual-encoding modes replace fixed dwell with encoding"
                )
        elif (
            self.dwell_distribution != "fixed"
            or self.dwell_floor_probability != 0.0
            or self.dwell_floor_ms != 0.0
            or self.dwell_log_mean != 0.0
            or self.dwell_log_sigma != 0.0
            or self.dwell_cap_ms != 0.0
            or self.encoding_K_seconds != 0.0
            or self.encoding_k != 0.0
            or self.encoding_f != 0.0
        ):
            raise ValueError("default/tuned EMMA modes require fixed dwell")

    def execution_completion_ms(
        self, execution_start_ms: float, visual_angle_deg: float
    ) -> float:
        return (
            float(execution_start_ms)
            + self.exec_intercept_ms
            + self.exec_slope_ms_per_degree * float(visual_angle_deg)
        )

    def metadata(self) -> dict[str, object]:
        result = {
            "emma_timing_contract_id": self.contract_id,
            "emma_mode": self.mode,
            "emma_prep_ms": self.prep_ms,
            "emma_exec_intercept_ms": self.exec_intercept_ms,
            "emma_exec_slope_ms_per_degree": self.exec_slope_ms_per_degree,
            "emma_dwell_ms": self.dwell_ms,
        }
        if self.mode == "human_stochastic":
            result.update({
                "emma_dwell_distribution": self.dwell_distribution,
                "emma_dwell_floor_probability": self.dwell_floor_probability,
                "emma_dwell_floor_ms": self.dwell_floor_ms,
                "emma_dwell_log_mean": self.dwell_log_mean,
                "emma_dwell_log_sigma": self.dwell_log_sigma,
                "emma_dwell_cap_ms": self.dwell_cap_ms,
                "emma_dwell_rng_contract_id": HUMAN_DWELL_RNG_CONTRACT_ID,
                "emma_dwell_rng_seed_offset": HUMAN_DWELL_RNG_SEED_OFFSET,
                "emma_dwell_calibration_id": HUMAN_DWELL_CALIBRATION_ID,
            })
        elif self.mode in VISUAL_ENCODING_EMMA_MODES:
            result.update({
                "encoding_K": self.encoding_K_seconds,
                "encoding_k": self.encoding_k,
                "encoding_f": self.encoding_f,
                "emma_encoding_time_unit": "seconds",
                "emma_encoding_temporal_semantics": (
                    DEFAULT_ENCODING_TEMPORAL_SEMANTICS
                ),
            })
            if self.mode == "relaxed_saccade":
                result.update({
                    "saccade_latency_scale": RELAXED_SACCADE_LATENCY_SCALE,
                    "encoding_latency_scale": RELAXED_SACCADE_ENCODING_SCALE,
                    "saccade_latency_reference_mode": "default_encoding",
                    "encoding_latency_reference_mode": "default_encoding",
                })
        return result


DEFAULT_EMMA_TIMING = EmmaTimingConfig(
    mode="default", prep_ms=PREP_MS,
    exec_intercept_ms=EXEC_BASE_MS,
    exec_slope_ms_per_degree=EXEC_MS_PER_DEGREE,
    dwell_ms=DWELL_MS,
)

DEFAULT_ENCODING_EMMA_TIMING = EmmaTimingConfig(
    mode="default_encoding",
    prep_ms=PREP_MS,
    exec_intercept_ms=EXEC_BASE_MS,
    exec_slope_ms_per_degree=EXEC_MS_PER_DEGREE,
    dwell_ms=0.0,
    contract_id=DEFAULT_ENCODING_EMMA_TIMING_CONTRACT_ID,
    encoding_K_seconds=DEFAULT_ENCODING_K_SECONDS,
    encoding_k=DEFAULT_ENCODING_K,
    encoding_f=DEFAULT_ENCODING_FAMILIARITY,
)

RELAXED_SACCADE_EMMA_TIMING = EmmaTimingConfig(
    mode="relaxed_saccade",
    prep_ms=PREP_MS * RELAXED_SACCADE_LATENCY_SCALE,
    exec_intercept_ms=EXEC_BASE_MS * RELAXED_SACCADE_LATENCY_SCALE,
    exec_slope_ms_per_degree=(
        EXEC_MS_PER_DEGREE * RELAXED_SACCADE_LATENCY_SCALE
    ),
    dwell_ms=0.0,
    contract_id=RELAXED_SACCADE_EMMA_TIMING_CONTRACT_ID,
    encoding_K_seconds=(
        DEFAULT_ENCODING_K_SECONDS * RELAXED_SACCADE_ENCODING_SCALE
    ),
    encoding_k=DEFAULT_ENCODING_K,
    encoding_f=DEFAULT_ENCODING_FAMILIARITY,
)

HUMAN_STOCHASTIC_EMMA_TIMING = EmmaTimingConfig(
    mode="human_stochastic",
    prep_ms=135.0,
    exec_intercept_ms=44.8732080135,
    exec_slope_ms_per_degree=3.83832565515,
    dwell_ms=HUMAN_DWELL_FLOOR_MS,
    contract_id=HUMAN_STOCHASTIC_EMMA_TIMING_CONTRACT_ID,
    dwell_distribution=HUMAN_DWELL_DISTRIBUTION,
    dwell_floor_probability=HUMAN_DWELL_FLOOR_PROBABILITY,
    dwell_floor_ms=HUMAN_DWELL_FLOOR_MS,
    dwell_log_mean=HUMAN_DWELL_LOG_MEAN,
    dwell_log_sigma=HUMAN_DWELL_LOG_SIGMA,
    dwell_cap_ms=HUMAN_DWELL_CAP_MS,
)


def resolve_emma_timing(
    mode: str = "default", *, prep_ms: float | None = None,
    exec_intercept_ms: float | None = None,
    exec_slope_ms_per_degree: float | None = None,
    dwell_ms: float | None = None,
) -> EmmaTimingConfig:
    supplied = (prep_ms, exec_intercept_ms, exec_slope_ms_per_degree, dwell_ms)
    if mode == "default":
        if any(value is not None for value in supplied):
            raise ValueError("default EMMA mode rejects tuned-only timing values")
        return DEFAULT_EMMA_TIMING
    if mode == "human_stochastic":
        if any(value is not None for value in supplied):
            raise ValueError(
                "human_stochastic EMMA mode uses calibrated canonical timing "
                "and rejects explicit timing values"
            )
        return HUMAN_STOCHASTIC_EMMA_TIMING
    if mode == "default_encoding":
        if any(value is not None for value in supplied):
            raise ValueError(
                "default_encoding EMMA mode uses canonical movement and encoding "
                "constants and rejects explicit timing values"
            )
        return DEFAULT_ENCODING_EMMA_TIMING
    if mode == "relaxed_saccade":
        if any(value is not None for value in supplied):
            raise ValueError(
                "relaxed_saccade uses canonical half-saccade movement and "
                "encoding constants and rejects explicit timing values"
            )
        return RELAXED_SACCADE_EMMA_TIMING
    if mode != "tuned":
        raise ValueError(
            "EMMA mode must be default, tuned, human_stochastic, "
            "default_encoding, or relaxed_saccade"
        )
    if any(value is None for value in supplied):
        raise ValueError("tuned EMMA mode requires all four timing values")
    return EmmaTimingConfig(
        mode="tuned", prep_ms=float(prep_ms),
        exec_intercept_ms=float(exec_intercept_ms),
        exec_slope_ms_per_degree=float(exec_slope_ms_per_degree),
        dwell_ms=float(dwell_ms),
    )


def emma_timing_from_metadata(metadata: dict[str, object]) -> EmmaTimingConfig:
    if "emma_timing_contract_id" not in metadata and "emma_mode" not in metadata:
        return DEFAULT_EMMA_TIMING
    mode = str(metadata.get("emma_mode"))
    if mode == "human_stochastic":
        expected = HUMAN_STOCHASTIC_EMMA_TIMING.metadata()
        mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise RuntimeError(
                f"checkpoint human stochastic EMMA timing mismatch: {mismatch}"
            )
        return HUMAN_STOCHASTIC_EMMA_TIMING
    if mode == "default_encoding":
        expected = DEFAULT_ENCODING_EMMA_TIMING.metadata()
        mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise RuntimeError(
                f"checkpoint default encoding EMMA timing mismatch: {mismatch}"
            )
        return DEFAULT_ENCODING_EMMA_TIMING
    if mode == "relaxed_saccade":
        expected = RELAXED_SACCADE_EMMA_TIMING.metadata()
        mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise RuntimeError(
                f"checkpoint relaxed-saccade EMMA timing mismatch: {mismatch}"
            )
        return RELAXED_SACCADE_EMMA_TIMING
    if metadata.get("emma_timing_contract_id") != EMMA_TIMING_CONTRACT_ID:
        raise RuntimeError("checkpoint EMMA timing contract mismatch")
    values = {
        "prep_ms": metadata.get("emma_prep_ms"),
        "exec_intercept_ms": metadata.get("emma_exec_intercept_ms"),
        "exec_slope_ms_per_degree": metadata.get(
            "emma_exec_slope_ms_per_degree"
        ),
        "dwell_ms": metadata.get("emma_dwell_ms"),
    }
    if mode == "default":
        expected = DEFAULT_EMMA_TIMING.metadata()
        mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"checkpoint default EMMA timing mismatch: {mismatch}")
        return DEFAULT_EMMA_TIMING
    try:
        return resolve_emma_timing(mode, **values)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid checkpoint EMMA timing: {error}") from error


def validate_tile(tile: Sequence[int]) -> tuple[int, int]:
    if len(tile) != 2:
        raise ValueError("fixation must have two coordinates")
    x, y = int(tile[0]), int(tile[1])
    if float(tile[0]) != x or float(tile[1]) != y or not (0 <= x < 8 and 0 <= y < 5):
        raise ValueError(f"invalid fixation {tuple(tile)}")
    return x, y


def acquisition_tiles(fixation: Sequence[int], shape: str) -> tuple[tuple[int, int], ...]:
    x, y = validate_tile(fixation)
    if shape == "3x3":
        offsets = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    elif shape == "2x2":
        offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
    elif shape == "plus":
        offsets = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
    else:
        raise ValueError(f"shape must be one of {FIXATION_SHAPES}")
    return tuple((x + dx, y + dy) for dx, dy in offsets if 0 <= x + dx < 8 and 0 <= y + dy < 5)


class ReferenceGeometry:
    @property
    def pixel_pitch_mm(self):
        ax, ay = MONITOR_ASPECT
        norm = math.hypot(ax, ay)
        return (
            (MONITOR_DIAGONAL_MM * ax / norm) / MONITOR_RASTER_PX[0],
            (MONITOR_DIAGONAL_MM * ay / norm) / MONITOR_RASTER_PX[1],
        )

    def _ray(self, tile):
        x, y = validate_tile(tile)
        pitch_x, pitch_y = self.pixel_pitch_mm
        ray = (
            ((x + 0.5) * VIEWPORT_TILE_PX[0] - EYE_PROJECTION_VIEWPORT_PX[0]) * pitch_x,
            ((y + 0.5) * VIEWPORT_TILE_PX[1] - EYE_PROJECTION_VIEWPORT_PX[1]) * pitch_y,
            VIEWING_DISTANCE_MM,
        )
        norm = math.sqrt(sum(value * value for value in ray))
        return tuple(value / norm for value in ray)

    def visual_angle_degrees(self, source, target) -> float:
        first, second = self._ray(source), self._ray(target)
        dot = sum(a * b for a, b in zip(first, second))
        return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


class GazePhase(str, Enum):
    READY = "READY"
    PREP = "PREP"
    EXEC = "EXEC"
    DWELL = "DWELL"
    ENCODE = "ENCODE"
    HOLD = "HOLD"


@dataclass(frozen=True)
class GazeEvent:
    timestamp_ms: float
    completed_phase: GazePhase
    next_phase: GazePhase
    acquisition: bool


class GazeStateMachine:
    def __init__(
        self,
        timing: EmmaTimingConfig = DEFAULT_EMMA_TIMING,
        *,
        dwell_seed: int | None = None,
    ) -> None:
        self.timing = timing
        if timing.mode == "human_stochastic" and dwell_seed is None:
            raise ValueError("human_stochastic EMMA timing requires dwell_seed")
        self.dwell_seed = None if dwell_seed is None else int(dwell_seed)
        self.dwell_rng = (
            np.random.Generator(np.random.PCG64(self.dwell_seed))
            if timing.mode == "human_stochastic"
            else None
        )
        self.dwell_sample_count = 0
        self.last_sampled_dwell_ms = timing.dwell_ms
        self.fixation_hold_completion_ms = None
        self.geometry = ReferenceGeometry()
        self.phase = GazePhase.READY
        self.current_fixation = None
        self.requested_target = None
        self.next_event_ms = None
        self.last_distance_degrees = 0.0
        self.encoding_started_ms = None
        self.initial_encoding_ms = None
        self.last_encoding_ms = None

    def visual_encoding_time_ms(self, eccentricity_deg: float) -> float:
        if self.timing.mode not in VISUAL_ENCODING_EMMA_MODES:
            raise RuntimeError(
                "visual encoding is active only in visual-encoding modes"
            )
        return visual_encoding_time_ms(
            eccentricity_deg,
            K_seconds=self.timing.encoding_K_seconds,
            k=self.timing.encoding_k,
            familiarity=self.timing.encoding_f,
        )

    def _sample_dwell_ms(self) -> float:
        if self.timing.mode != "human_stochastic":
            return self.timing.dwell_ms
        assert self.dwell_rng is not None
        self.dwell_sample_count += 1
        if self.dwell_rng.random() < self.timing.dwell_floor_probability:
            value = self.timing.dwell_floor_ms
        else:
            sample = float(self.dwell_rng.lognormal(
                self.timing.dwell_log_mean,
                self.timing.dwell_log_sigma,
            ))
            value = min(
                max(sample, self.timing.dwell_floor_ms),
                self.timing.dwell_cap_ms,
            )
        self.last_sampled_dwell_ms = value
        return value

    def _schedule_post_landing(self, now_ms: float) -> None:
        if self.timing.mode in VISUAL_ENCODING_EMMA_MODES:
            encoding_ms = self.visual_encoding_time_ms(0.0)
            self.phase = GazePhase.ENCODE
            self.next_event_ms = float(now_ms) + encoding_ms
            self.encoding_started_ms = float(now_ms)
            self.initial_encoding_ms = encoding_ms
            self.last_encoding_ms = encoding_ms
            self.fixation_hold_completion_ms = None
            return
        if self.timing.mode != "human_stochastic":
            self.phase = GazePhase.DWELL
            self.next_event_ms = float(now_ms) + self.timing.dwell_ms
            self.fixation_hold_completion_ms = None
            return
        dwell_ms = self._sample_dwell_ms()
        self.phase = GazePhase.ENCODE
        self.next_event_ms = float(now_ms) + self.timing.dwell_floor_ms
        self.fixation_hold_completion_ms = float(now_ms) + dwell_ms

    def reset(self, self_tile, now_ms=0.0) -> None:
        self.current_fixation = validate_tile(self_tile)
        self.requested_target = self.current_fixation
        self._schedule_post_landing(now_ms)
        self.last_distance_degrees = 0.0

    def select(self, target, now_ms: float) -> None:
        if self.phase is not GazePhase.READY or self.current_fixation is None:
            raise RuntimeError("gaze operation is committed until READY")
        selected = validate_tile(target)
        self.requested_target = selected
        if selected == self.current_fixation:
            self._schedule_post_landing(now_ms)
            self.last_distance_degrees = 0.0
        else:
            self.phase = GazePhase.PREP
            self.next_event_ms = float(now_ms) + self.timing.prep_ms
            self.last_distance_degrees = self.geometry.visual_angle_degrees(
                self.current_fixation, selected
            )
            if self.timing.mode in VISUAL_ENCODING_EMMA_MODES:
                self.encoding_started_ms = float(now_ms)
                self.initial_encoding_ms = self.visual_encoding_time_ms(
                    self.last_distance_degrees
                )
                self.last_encoding_ms = self.initial_encoding_ms

    def advance_event(self) -> GazeEvent:
        if self.next_event_ms is None or self.requested_target is None:
            raise RuntimeError("no gaze event scheduled")
        timestamp = self.next_event_ms
        completed = self.phase
        acquisition = False
        if completed is GazePhase.PREP:
            self.phase = GazePhase.EXEC
            self.next_event_ms = self.timing.execution_completion_ms(
                timestamp, self.last_distance_degrees
            )
        elif completed is GazePhase.EXEC:
            self.current_fixation = self.requested_target
            if self.timing.mode in VISUAL_ENCODING_EMMA_MODES:
                if self.encoding_started_ms is None or self.initial_encoding_ms is None:
                    raise RuntimeError("default encoding movement has no encoding state")
                elapsed = timestamp - self.encoding_started_ms
                remaining_fraction = max(
                    0.0, 1.0 - elapsed / self.initial_encoding_ms
                )
                remaining_ms = (
                    remaining_fraction * self.visual_encoding_time_ms(0.0)
                )
                self.last_encoding_ms = remaining_ms
                if remaining_ms == 0.0:
                    self.phase = GazePhase.READY
                    self.next_event_ms = None
                    self.encoding_started_ms = None
                    self.initial_encoding_ms = None
                    acquisition = True
                else:
                    self.phase = GazePhase.ENCODE
                    self.next_event_ms = timestamp + remaining_ms
            else:
                self._schedule_post_landing(timestamp)
        elif completed is GazePhase.DWELL:
            self.phase = GazePhase.READY
            self.next_event_ms = None
            acquisition = True
        elif completed is GazePhase.ENCODE:
            acquisition = True
            if self.timing.mode in VISUAL_ENCODING_EMMA_MODES:
                self.phase = GazePhase.READY
                self.next_event_ms = None
                self.encoding_started_ms = None
                self.initial_encoding_ms = None
                return GazeEvent(timestamp, completed, self.phase, acquisition)
            hold_completion = self.fixation_hold_completion_ms
            if hold_completion is None or hold_completion <= timestamp:
                self.phase = GazePhase.READY
                self.next_event_ms = None
                self.fixation_hold_completion_ms = None
            else:
                self.phase = GazePhase.HOLD
                self.next_event_ms = hold_completion
        elif completed is GazePhase.HOLD:
            self.phase = GazePhase.READY
            self.next_event_ms = None
            self.fixation_hold_completion_ms = None
        else:
            raise RuntimeError("READY has no completion event")
        return GazeEvent(timestamp, completed, self.phase, acquisition)

    def training_state(self) -> dict[str, object]:
        result = {
            "phase": self.phase.value,
            "current_fixation": self.current_fixation,
            "requested_target": self.requested_target,
            "next_event_ms": self.next_event_ms,
            "last_distance_degrees": self.last_distance_degrees,
        }
        if self.timing.mode == "human_stochastic":
            assert self.dwell_rng is not None
            result.update({
                "dwell_seed": self.dwell_seed,
                "dwell_rng_contract_id": HUMAN_DWELL_RNG_CONTRACT_ID,
                "dwell_rng_state": copy.deepcopy(
                    self.dwell_rng.bit_generator.state
                ),
                "dwell_sample_count": self.dwell_sample_count,
                "last_sampled_dwell_ms": self.last_sampled_dwell_ms,
                "fixation_hold_completion_ms": self.fixation_hold_completion_ms,
            })
        elif self.timing.mode in VISUAL_ENCODING_EMMA_MODES:
            result.update({
                "encoding_started_ms": self.encoding_started_ms,
                "initial_encoding_ms": self.initial_encoding_ms,
                "last_encoding_ms": self.last_encoding_ms,
            })
        return result

    def load_training_state(self, state: dict[str, object]) -> None:
        self.phase = GazePhase(str(state["phase"]))
        self.current_fixation = validate_tile(state["current_fixation"])
        self.requested_target = validate_tile(state["requested_target"])
        next_event = state["next_event_ms"]
        self.next_event_ms = None if next_event is None else float(next_event)
        self.last_distance_degrees = float(state["last_distance_degrees"])
        if self.timing.mode == "human_stochastic":
            if state.get("dwell_rng_contract_id") != HUMAN_DWELL_RNG_CONTRACT_ID:
                raise RuntimeError("gaze dwell RNG contract mismatch")
            if state.get("dwell_seed") != self.dwell_seed:
                raise RuntimeError("gaze dwell RNG seed mismatch")
            if "dwell_rng_state" not in state:
                raise RuntimeError("human stochastic gaze resume has no dwell RNG state")
            assert self.dwell_rng is not None
            self.dwell_rng.bit_generator.state = copy.deepcopy(
                state["dwell_rng_state"]
            )
            self.dwell_sample_count = int(state["dwell_sample_count"])
            self.last_sampled_dwell_ms = float(state["last_sampled_dwell_ms"])
            hold_completion = state.get("fixation_hold_completion_ms")
            self.fixation_hold_completion_ms = (
                None if hold_completion is None else float(hold_completion)
            )
        elif self.timing.mode in VISUAL_ENCODING_EMMA_MODES:
            for name in (
                "encoding_started_ms", "initial_encoding_ms", "last_encoding_ms"
            ):
                value = state.get(name)
                setattr(self, name, None if value is None else float(value))


def visual_encoding_time_ms(
    eccentricity_deg: float,
    *,
    K_seconds: float = DEFAULT_ENCODING_K_SECONDS,
    k: float = DEFAULT_ENCODING_K,
    familiarity: float = DEFAULT_ENCODING_FAMILIARITY,
) -> float:
    eccentricity = float(eccentricity_deg)
    K_value = float(K_seconds)
    k_value = float(k)
    frequency = float(familiarity)
    if not math.isfinite(eccentricity) or eccentricity < 0.0:
        raise ValueError("encoding eccentricity must be finite and nonnegative")
    if not math.isfinite(K_value) or K_value <= 0.0:
        raise ValueError("encoding K must be finite and positive")
    if not math.isfinite(k_value) or k_value < 0.0:
        raise ValueError("encoding k must be finite and nonnegative")
    if not math.isfinite(frequency) or not 0.0 < frequency <= 1.0:
        raise ValueError("encoding familiarity must be in (0, 1]")
    return 1000.0 * K_value * (-math.log(frequency)) * math.exp(
        k_value * eccentricity
    )
