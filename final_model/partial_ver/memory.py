"""Timestamped Partial memory with exact static infrastructure only."""

from __future__ import annotations

import copy
from typing import Iterable, Sequence

import numpy as np


STATIC_CHANNELS = (10, 11, 12, 13, 14)
DYNAMIC_CHANNELS = tuple(index for index in range(20) if index not in STATIC_CHANNELS)
MEMORY_LIFETIME_MS = 2400.0
MEMORY_DECAY_MODES = ("linear", "none", "stochastic_exponential")
MEMORY_RNG_SEED_OFFSET = 131_000_039
RAW_FRESHNESS_SCALE = np.float32(255.0)
CONTINUOUS_FRESHNESS_CONTRACT_ID = (
    "continuous_linear_age_over_configured_lifetime_raw255_hard_expiry_v1"
)
NO_DECAY_FRESHNESS_CONTRACT_ID = (
    "persistent_binary_observed_freshness_raw255_no_expiry_v1"
)
STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID = (
    "stochastic_exponential_binary_alive_hard_cutoff_raw255_v2"
)
OBSERVATION_CONTRACT_ID = (
    "partial_state20_continuous_linear_freshness_configurable_expiry_raw255_"
    "preview4_v7"
)
NO_DECAY_OBSERVATION_CONTRACT_ID = (
    "partial_state20_persistent_binary_freshness_no_expiry_raw255_preview4_v1"
)
STOCHASTIC_EXPONENTIAL_OBSERVATION_CONTRACT_ID = (
    "partial_state20_stochastic_exponential_binary_alive_hard_cutoff_"
    "raw255_preview4_v2"
)
MEMORY_RNG_CONTRACT_ID = "numpy_pcg64_memory_lifetime_per_adapter_seed_offset_v1"


def observation_contract_id(
    memory_lifetime_ms: float | None,
    memory_decay_mode: str = "linear",
    memory_tau_ms: float | None = None,
) -> str:
    mode = str(memory_decay_mode)
    if mode == "none":
        if memory_lifetime_ms is not None:
            raise ValueError("no-decay memory must not have a finite lifetime")
        return NO_DECAY_OBSERVATION_CONTRACT_ID
    if mode == "stochastic_exponential":
        if memory_lifetime_ms is None or memory_tau_ms is None:
            raise ValueError(
                "stochastic exponential memory requires max lifetime and tau"
            )
        lifetime = float(memory_lifetime_ms)
        tau = float(memory_tau_ms)
        if not np.isfinite(lifetime) or lifetime <= 0.0:
            raise ValueError("memory max lifetime must be finite and positive")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("memory tau must be finite and positive")
        return STOCHASTIC_EXPONENTIAL_OBSERVATION_CONTRACT_ID
    if mode != "linear":
        raise ValueError(f"memory decay mode must be one of {MEMORY_DECAY_MODES}")
    if memory_tau_ms is not None:
        raise ValueError("linear memory decay does not use memory tau")
    if memory_lifetime_ms is None:
        raise ValueError("linear memory decay requires a finite lifetime")
    lifetime = float(memory_lifetime_ms)
    if not np.isfinite(lifetime) or lifetime <= 0.0:
        raise ValueError("memory lifetime must be finite and positive")
    return OBSERVATION_CONTRACT_ID


class PersistentPartialMemory:
    def __init__(
        self,
        *,
        memory_lifetime_ms: float | None = MEMORY_LIFETIME_MS,
        memory_decay_mode: str = "linear",
        memory_tau_ms: float | None = None,
        memory_seed: int | None = None,
    ) -> None:
        mode = str(memory_decay_mode)
        self.observation_contract_id = observation_contract_id(
            memory_lifetime_ms, mode, memory_tau_ms
        )
        self.memory_decay_mode = mode
        self.memory_lifetime_ms = (
            None if memory_lifetime_ms is None else float(memory_lifetime_ms)
        )
        self.memory_tau_ms = (
            None if memory_tau_ms is None else float(memory_tau_ms)
        )
        if mode == "stochastic_exponential" and memory_seed is None:
            raise ValueError("stochastic exponential memory requires memory_seed")
        if mode != "stochastic_exponential" and memory_seed is not None:
            raise ValueError("memory_seed is only valid for stochastic exponential memory")
        self.memory_seed = None if memory_seed is None else int(memory_seed)
        self.memory_rng = (
            np.random.Generator(np.random.PCG64(self.memory_seed))
            if mode == "stochastic_exponential"
            else None
        )
        self.freshness_contract_id = (
            NO_DECAY_FRESHNESS_CONTRACT_ID
            if mode == "none"
            else STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID
            if mode == "stochastic_exponential"
            else CONTINUOUS_FRESHNESS_CONTRACT_ID
        )
        self.state20 = np.zeros((8, 5, 20), dtype=np.float32)
        self.last_observed_time_ms = np.full((8, 5), -np.inf, dtype=np.float64)
        self.sampled_lifetime_ms = np.full((8, 5), np.inf, dtype=np.float64)
        self.actual_lifetime_ms = np.full((8, 5), np.inf, dtype=np.float64)
        self.current_model_time_ms = 0.0
        self.preview4 = np.zeros((8, 5, 4), dtype=np.float32)
        self.static_reference = np.zeros((8, 5, 5), dtype=np.float32)
        self.refresh_count = 0
        self.expiry_count = 0
        self.stochastic_deletion_count = 0
        self.hard_cutoff_deletion_count = 0
        self.lifetime_sample_count = 0

    def reset(self, current_state20: np.ndarray, *, now_ms: float = 0.0) -> np.ndarray:
        current = np.asarray(current_state20, dtype=np.float32)
        if current.shape != (8, 5, 20):
            raise ValueError("canonical state must be (8,5,20)")
        now = float(now_ms)
        if not np.isfinite(now) or now < 0.0:
            raise ValueError("model time must be finite and non-negative")
        self.state20.fill(0.0)
        self.last_observed_time_ms.fill(-np.inf)
        self.sampled_lifetime_ms.fill(np.inf)
        self.actual_lifetime_ms.fill(np.inf)
        self.current_model_time_ms = now
        self.preview4.fill(0.0)
        self.static_reference = np.ascontiguousarray(current[..., STATIC_CHANNELS].copy())
        self.state20[..., STATIC_CHANNELS] = self.static_reference
        self.refresh_count = 0
        self.expiry_count = 0
        self.stochastic_deletion_count = 0
        self.hard_cutoff_deletion_count = 0
        self.lifetime_sample_count = 0
        return self.snapshot()

    def _validate_time(self, now_ms: float) -> float:
        now = float(now_ms)
        if not np.isfinite(now) or now < self.current_model_time_ms:
            raise ValueError("model time must be finite and monotonic")
        return now

    def set_time(self, now_ms: float) -> None:
        self.current_model_time_ms = self._validate_time(now_ms)
        self._clear_expired()

    def acquire(
        self,
        current_state20,
        current_preview4,
        tiles: Iterable[Sequence[int]],
        *,
        timestamp_ms: float | None = None,
    ) -> None:
        current = np.asarray(current_state20, dtype=np.float32)
        preview = np.asarray(current_preview4, dtype=np.float32)
        if current.shape != (8, 5, 20) or preview.shape != (8, 5, 4):
            raise ValueError("invalid acquisition inputs")
        if not np.array_equal(current[..., STATIC_CHANNELS], self.static_reference):
            raise RuntimeError("static canonical geometry changed within episode")
        if timestamp_ms is not None:
            self.set_time(timestamp_ms)
        unique = tuple(sorted(
            {tuple(map(int, tile)) for tile in tiles},
            key=lambda tile: tile[0] * 5 + tile[1],
        ))
        for tile in unique:
            x, y = tile
            if not (0 <= x < 8 and 0 <= y < 5):
                raise ValueError(f"out-of-board acquisition tile {tile}")
            self.state20[x, y, DYNAMIC_CHANNELS] = current[x, y, DYNAMIC_CHANNELS]
            self.last_observed_time_ms[x, y] = self.current_model_time_ms
            self.preview4[x, y] = preview[x, y]
            if self.memory_decay_mode == "stochastic_exponential":
                assert self.memory_rng is not None
                sampled = float(self.memory_rng.exponential(self.memory_tau_ms))
                self.sampled_lifetime_ms[x, y] = sampled
                self.actual_lifetime_ms[x, y] = min(
                    sampled, self.memory_lifetime_ms
                )
                self.lifetime_sample_count += 1
        self.refresh_count += len(unique)

    def ages_ms(self) -> np.ndarray:
        observed = np.isfinite(self.last_observed_time_ms)
        ages = np.full((8, 5), np.inf, dtype=np.float64)
        ages[observed] = (
            self.current_model_time_ms - self.last_observed_time_ms[observed]
        )
        if np.any(ages[observed] < 0.0):
            raise RuntimeError("Partial memory contains a future observation timestamp")
        return ages

    def semantic_freshness(self, now_ms: float | None = None) -> np.ndarray:
        if now_ms is not None:
            self.set_time(now_ms)
        if self.memory_decay_mode == "none":
            return np.ascontiguousarray(
                np.isfinite(self.last_observed_time_ms), dtype=np.float32
            )
        ages = self.ages_ms()
        if self.memory_decay_mode == "stochastic_exponential":
            return np.ascontiguousarray(
                np.isfinite(self.last_observed_time_ms), dtype=np.float32
            )
        freshness = np.clip(
            1.0 - ages / self.memory_lifetime_ms,
            0.0,
            1.0,
        )
        return np.ascontiguousarray(freshness, dtype=np.float32)

    def raw_freshness_input(self, now_ms: float | None = None) -> np.ndarray:
        return np.ascontiguousarray(
            self.semantic_freshness(now_ms) * RAW_FRESHNESS_SCALE,
            dtype=np.float32,
        )

    def _clear_expired(self) -> None:
        if self.memory_decay_mode == "none":
            return
        ages = self.current_model_time_ms - self.last_observed_time_ms
        if self.memory_decay_mode == "stochastic_exponential":
            expired = np.isfinite(self.last_observed_time_ms) & (
                ages >= self.actual_lifetime_ms
            )
        else:
            expired = np.isfinite(self.last_observed_time_ms) & (
                ages >= self.memory_lifetime_ms
            )
        if not np.any(expired):
            return
        self.expiry_count += int(np.count_nonzero(expired))
        if self.memory_decay_mode == "stochastic_exponential":
            hard_cutoff = expired & (
                self.sampled_lifetime_ms >= self.memory_lifetime_ms
            )
            hard_count = int(np.count_nonzero(hard_cutoff))
            self.hard_cutoff_deletion_count += hard_count
            self.stochastic_deletion_count += (
                int(np.count_nonzero(expired)) - hard_count
            )
        for channel in DYNAMIC_CHANNELS:
            self.state20[..., channel][expired] = 0.0
        self.preview4[expired] = 0.0
        self.last_observed_time_ms[expired] = -np.inf
        self.sampled_lifetime_ms[expired] = np.inf
        self.actual_lifetime_ms[expired] = np.inf

    def snapshot(self, now_ms: float | None = None) -> np.ndarray:
        if now_ms is not None:
            self.set_time(now_ms)
        else:
            self._clear_expired()
        result = np.concatenate(
            (
                self.state20,
                self.raw_freshness_input()[..., None],
                self.preview4,
            ),
            axis=-1,
        )
        return np.ascontiguousarray(result, dtype=np.float32)

    def valid_fraction(self, now_ms: float | None = None) -> float:
        return float(np.mean(self.semantic_freshness(now_ms) > 0.0))

    def diagnostics(self) -> dict[str, object]:
        semantic = self.semantic_freshness()
        raw = semantic * RAW_FRESHNESS_SCALE
        result = {
            "freshness_raw_min": float(raw.min()),
            "freshness_raw_max": float(raw.max()),
            "freshness_normalized_min": float(semantic.min()),
            "freshness_normalized_max": float(semantic.max()),
            "freshness_normalized_mean": float(semantic.mean()),
            "freshness_normalized_variance": float(semantic.var()),
            "mean_freshness": float(semantic.mean()),
            "valid_memory_fraction": float(np.mean(semantic > 0)),
            "memory_refresh_count": self.refresh_count,
            "memory_expiry_count": self.expiry_count,
        }
        if self.memory_decay_mode == "stochastic_exponential":
            result.update({
                "memory_stochastic_deletion_count": self.stochastic_deletion_count,
                "memory_hard_cutoff_deletion_count": self.hard_cutoff_deletion_count,
                "memory_lifetime_sample_count": self.lifetime_sample_count,
            })
        return result

    def training_state(self) -> dict[str, object]:
        result = {
            "observation_contract_id": self.observation_contract_id,
            "freshness_contract_id": self.freshness_contract_id,
            "memory_decay_mode": self.memory_decay_mode,
            "memory_lifetime_ms": self.memory_lifetime_ms,
            "state20": self.state20.copy(),
            "last_observed_time_ms": self.last_observed_time_ms.copy(),
            "current_model_time_ms": self.current_model_time_ms,
            "preview4": self.preview4.copy(),
            "static_reference": self.static_reference.copy(),
            "refresh_count": self.refresh_count,
            "expiry_count": self.expiry_count,
        }
        if self.memory_decay_mode == "stochastic_exponential":
            assert self.memory_rng is not None
            result.update({
                "memory_tau_ms": self.memory_tau_ms,
                "memory_max_lifetime_ms": self.memory_lifetime_ms,
                "memory_seed": self.memory_seed,
                "memory_rng_contract_id": MEMORY_RNG_CONTRACT_ID,
                "memory_rng_state": copy.deepcopy(
                    self.memory_rng.bit_generator.state
                ),
                "sampled_lifetime_ms": self.sampled_lifetime_ms.copy(),
                "actual_lifetime_ms": self.actual_lifetime_ms.copy(),
                "stochastic_deletion_count": self.stochastic_deletion_count,
                "hard_cutoff_deletion_count": self.hard_cutoff_deletion_count,
                "lifetime_sample_count": self.lifetime_sample_count,
            })
        return result

    def load_training_state(self, state: dict[str, object]) -> None:
        saved_mode = str(state.get("memory_decay_mode", "linear"))
        if (
            state.get("observation_contract_id") != self.observation_contract_id
            or state.get("freshness_contract_id") != self.freshness_contract_id
            or saved_mode != self.memory_decay_mode
            or state.get("memory_lifetime_ms") != self.memory_lifetime_ms
            or (
                saved_mode == "stochastic_exponential"
                and state.get("memory_tau_ms") != self.memory_tau_ms
            )
        ):
            raise RuntimeError("Partial memory resume contract mismatch")
        arrays = {
            "state20": ((8, 5, 20), np.float32),
            "last_observed_time_ms": ((8, 5), np.float64),
            "preview4": ((8, 5, 4), np.float32),
            "static_reference": ((8, 5, 5), np.float32),
        }
        for name, (shape, dtype) in arrays.items():
            value = np.asarray(state[name], dtype=dtype)
            if value.shape != shape:
                raise RuntimeError(f"invalid resume memory {name} shape: {value.shape}")
            setattr(self, name, np.ascontiguousarray(value.copy()))
        self.current_model_time_ms = float(state["current_model_time_ms"])
        self.refresh_count = int(state["refresh_count"])
        self.expiry_count = int(state["expiry_count"])
        if self.memory_decay_mode == "stochastic_exponential":
            if state.get("memory_rng_contract_id") != MEMORY_RNG_CONTRACT_ID:
                raise RuntimeError("Partial memory RNG contract mismatch")
            if state.get("memory_seed") != self.memory_seed:
                raise RuntimeError("Partial memory RNG seed mismatch")
            assert self.memory_rng is not None
            self.memory_rng.bit_generator.state = copy.deepcopy(
                state["memory_rng_state"]
            )
            for name in ("sampled_lifetime_ms", "actual_lifetime_ms"):
                value = np.asarray(state[name], dtype=np.float64)
                if value.shape != (8, 5):
                    raise RuntimeError(f"invalid resume memory {name} shape")
                setattr(self, name, np.ascontiguousarray(value.copy()))
            self.stochastic_deletion_count = int(
                state["stochastic_deletion_count"]
            )
            self.hard_cutoff_deletion_count = int(
                state["hard_cutoff_deletion_count"]
            )
            self.lifetime_sample_count = int(state["lifetime_sample_count"])
        if not np.isfinite(self.current_model_time_ms) or self.current_model_time_ms < 0:
            raise RuntimeError("invalid Partial memory model time")
        finite = np.isfinite(self.last_observed_time_ms)
        if np.any(self.last_observed_time_ms[finite] > self.current_model_time_ms):
            raise RuntimeError("invalid Partial memory observation timestamp")
        if not np.array_equal(
            self.state20[..., STATIC_CHANNELS], self.static_reference
        ):
            raise RuntimeError("Partial memory static state changed in resume checkpoint")
        self._clear_expired()


def memory_contract_report(
    memory_lifetime_ms: float | None = MEMORY_LIFETIME_MS,
    memory_decay_mode: str = "linear",
    memory_tau_ms: float | None = None,
    memory_seed: int | None = None,
) -> dict[str, object]:
    memory = PersistentPartialMemory(
        memory_lifetime_ms=memory_lifetime_ms,
        memory_decay_mode=memory_decay_mode,
        memory_tau_ms=memory_tau_ms,
        memory_seed=memory_seed,
    )
    no_decay = memory.memory_decay_mode == "none"
    stochastic = memory.memory_decay_mode == "stochastic_exponential"
    result = {
        "observation_contract_id": memory.observation_contract_id,
        "freshness_contract_id": memory.freshness_contract_id,
        "memory_decay_mode": memory.memory_decay_mode,
        "memory_lifetime_ms": memory.memory_lifetime_ms,
        "semantic_freshness": (
            "1 if ever observed in this episode, otherwise 0"
            if no_decay
            else "1 while sampled entry is alive; 0 after deletion"
            if stochastic
            else "clip(1 - age_ms / memory_lifetime_ms, 0, 1)"
        ),
        "raw_freshness": "255 * semantic_freshness, float32",
        "always_exact_channels": list(STATIC_CHANNELS),
        "gaze_dependent_channels": list(DYNAMIC_CHANNELS) + [21, 22, 23, 24],
        "preview_semantics": (
            "renderer-visible max-alpha winner step only per tile; raw overlapping "
            "loser steps are not encoded"
        ),
        "held_object_flags": False,
        "derived_cooking_ready_flags": False,
        "reset": "static 10-14 exact; all dynamic, freshness, and preview zero",
        "time_source": "absolute simulation/model milliseconds",
        "expiration": (
            "none until episode reset"
            if no_decay
            else (
                "age >= min(sampled Exponential(tau), max_lifetime) clears "
                "dynamic state and preview"
            )
            if stochastic
            else (
                f"age >= {memory.memory_lifetime_ms:g}ms clears dynamic state "
                "and preview"
            )
        ),
    }
    if stochastic:
        result.update({
            "memory_tau_ms": memory.memory_tau_ms,
            "memory_max_lifetime_ms": memory.memory_lifetime_ms,
            "memory_rng_contract_id": MEMORY_RNG_CONTRACT_ID,
            "memory_seed": memory.memory_seed,
        })
    return result
