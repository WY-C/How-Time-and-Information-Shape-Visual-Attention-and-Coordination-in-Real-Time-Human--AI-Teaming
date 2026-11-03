"""Read-only event instrumentation for the bounded footprint memory audit."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


BOARD_TILES = 40
NOMINAL_TILES = {"3x3": 9, "2x2": 4, "plus": 5}


def _empty_phase() -> dict[str, object]:
    return {
        "sample_count": 0,
        "valid_counts": [],
        "freshness_normalized_sum": 0.0,
        "freshness_normalized_square_sum": 0.0,
        "freshness_tile_samples": 0,
        "preview_nonzero_tile_count": 0,
        "valid_preview_nonzero_tile_count": 0,
        "valid_tile_count": 0,
        "preview_channel_nonzero_counts": [0] * 4,
    }


class _AuditedRuleGazePolicy:
    """Delegate exactly, recording only the memory that the rule policy sees."""

    def __init__(self, policy, recorder: "FootprintAuditRecorder") -> None:
        self._policy = policy
        self._recorder = recorder

    def __call__(self, memory, fixation, context):
        self._recorder.record_sample("pre_gaze_selection", memory)
        return self._policy(memory, fixation, context)

    def __getattr__(self, name):
        return getattr(self._policy, name)


class FootprintAuditRecorder:
    """Collect sufficient statistics without reading hidden environment state."""

    def __init__(self, *, env_id: int, horizon: int, fixation_shape: str) -> None:
        if fixation_shape not in NOMINAL_TILES:
            raise ValueError(f"unsupported audit footprint {fixation_shape!r}")
        self.env_id = int(env_id)
        self.horizon = int(horizon)
        self.fixation_shape = str(fixation_shape)
        self.phases = {
            "pre_gaze_selection": _empty_phase(),
            "post_acquisition": _empty_phase(),
            "task_deadline": _empty_phase(),
        }
        self.actual_tile_counts: list[int] = []
        self.new_valid_counts: list[int] = []
        self.refreshed_counts: list[int] = []
        self.recovered_expired_or_unknown_counts: list[int] = []
        self.reacquired_after_expiry_counts: list[int] = []
        self.same_tile_intervals_ms: list[float] = []
        self.expiry_count = 0
        self.expired_tiles: set[tuple[int, int]] = set()
        self.reacquired_after_expiry_tiles: set[tuple[int, int]] = set()
        self.expiries_per_decision: list[int] = []
        self._current_decision_expiries = 0
        self._last_acquisition_ms: dict[tuple[int, int], float] = {}
        self.actor_mask_checks = 0
        self.actor_mask_failures = 0

    def install(self, adapter) -> None:
        """Install audit-only delegates after canonical state restoration."""
        adapter.gaze_policy = _AuditedRuleGazePolicy(adapter.gaze_policy, self)
        memory = adapter.memory
        original_set_time = memory.set_time
        original_acquire = memory.acquire

        def audited_set_time(now_ms: float) -> None:
            before = np.isfinite(memory.last_observed_time_ms)
            original_set_time(now_ms)
            after = np.isfinite(memory.last_observed_time_ms)
            expired = np.argwhere(before & ~after)
            count = int(expired.shape[0])
            if count:
                self.expiry_count += count
                self._current_decision_expiries += count
                self.expired_tiles.update(
                    (int(tile[0]), int(tile[1])) for tile in expired
                )

        def audited_acquire(
            current_state20,
            current_preview4,
            tiles: Iterable[Sequence[int]],
            *,
            timestamp_ms: float | None = None,
        ) -> None:
            unique = tuple(sorted({tuple(map(int, tile)) for tile in tiles}))
            if timestamp_ms is not None:
                # Preserve acquire semantics while measuring validity after any
                # expiration at the acquisition timestamp.
                memory.set_time(timestamp_ms)
            before = np.isfinite(memory.last_observed_time_ms)
            now_ms = float(memory.current_model_time_ms)
            original_acquire(
                current_state20,
                current_preview4,
                unique,
                timestamp_ms=None,
            )
            new_valid = 0
            refreshed = 0
            reacquired_after_expiry = 0
            for tile in unique:
                if before[tile]:
                    refreshed += 1
                else:
                    new_valid += 1
                    if tile in self.expired_tiles:
                        reacquired_after_expiry += 1
                        self.reacquired_after_expiry_tiles.add(tile)
                previous = self._last_acquisition_ms.get(tile)
                if previous is not None:
                    self.same_tile_intervals_ms.append(now_ms - previous)
                self._last_acquisition_ms[tile] = now_ms
            self.actual_tile_counts.append(len(unique))
            self.new_valid_counts.append(new_valid)
            self.refreshed_counts.append(refreshed)
            self.recovered_expired_or_unknown_counts.append(new_valid)
            self.reacquired_after_expiry_counts.append(reacquired_after_expiry)

        memory.set_time = audited_set_time
        memory.acquire = audited_acquire

    def record_sample(self, phase: str, observation) -> None:
        sample = np.asarray(observation, dtype=np.float32)
        if sample.shape != (8, 5, 25):
            raise RuntimeError(f"invalid audit memory sample shape {sample.shape}")
        row = self.phases[phase]
        raw = np.asarray(sample[..., 20], dtype=np.float64)
        if not np.isfinite(raw).all() or np.any(raw < 0.0) or np.any(raw > 255.0):
            raise RuntimeError("invalid continuous freshness in audit sample")
        normalized = raw / 255.0
        valid = normalized > 0.0
        preview = sample[..., 21:25]
        preview_nonzero = np.any(preview != 0.0, axis=-1)
        row["sample_count"] += 1
        row["valid_counts"].append(int(np.count_nonzero(valid)))
        row["freshness_normalized_sum"] += float(normalized.sum())
        row["freshness_normalized_square_sum"] += float(
            np.square(normalized).sum()
        )
        row["freshness_tile_samples"] += BOARD_TILES
        row["preview_nonzero_tile_count"] += int(
            np.count_nonzero(preview_nonzero)
        )
        row["valid_preview_nonzero_tile_count"] += int(
            np.count_nonzero(valid & preview_nonzero)
        )
        row["valid_tile_count"] += int(np.count_nonzero(valid))
        for channel in range(4):
            row["preview_channel_nonzero_counts"][channel] += int(
                np.count_nonzero(preview[..., channel] != 0.0)
            )

    def record_actionnet_query(self, observation, available_actions) -> None:
        self.record_sample("post_acquisition", observation)
        available = np.asarray(available_actions, dtype=np.float32)
        self.actor_mask_checks += 1
        if available.shape != (6,) or not np.array_equal(
            available, np.ones(6, dtype=np.float32)
        ):
            self.actor_mask_failures += 1

    def record_deadline(self, observation, available_actions) -> None:
        self.record_sample("task_deadline", observation)
        available = np.asarray(available_actions, dtype=np.float32)
        self.actor_mask_checks += 1
        if available.shape != (6,) or not np.array_equal(
            available, np.ones(6, dtype=np.float32)
        ):
            self.actor_mask_failures += 1
        self.expiries_per_decision.append(self._current_decision_expiries)
        self._current_decision_expiries = 0

    def payload(self) -> dict[str, object]:
        return {
            "env_id": self.env_id,
            "horizon": self.horizon,
            "fixation_shape": self.fixation_shape,
            "nominal_tiles_per_acquisition": NOMINAL_TILES[self.fixation_shape],
            "phases": self.phases,
            "acquisition": {
                "actual_tile_counts": self.actual_tile_counts,
                "new_valid_counts": self.new_valid_counts,
                "refreshed_counts": self.refreshed_counts,
                "recovered_expired_or_unknown_counts": (
                    self.recovered_expired_or_unknown_counts
                ),
                "reacquired_after_expiry_counts": (
                    self.reacquired_after_expiry_counts
                ),
                "same_tile_intervals_ms": self.same_tile_intervals_ms,
            },
            "expiry": {
                "count": self.expiry_count,
                "unique_expired_tiles": len(self.expired_tiles),
                "unique_reacquired_after_expiry_tiles": len(
                    self.reacquired_after_expiry_tiles
                ),
                "per_decision_counts": self.expiries_per_decision,
            },
            "actor_mask_checks": self.actor_mask_checks,
            "actor_mask_failures": self.actor_mask_failures,
            "hidden_state_reads": 0,
        }
