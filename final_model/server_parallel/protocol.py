"""Fixed server-parallel protocol constants and deterministic sharding."""

from __future__ import annotations

from dataclasses import dataclass

from final_model.partial_ver.gaze import (
    DEFAULT_ENCODING_K_SECONDS,
    DWELL_MS,
    FIXATION_SHAPES,
    RELAXED_SACCADE_ENCODING_SCALE,
    visual_encoding_time_ms,
)
from final_model.partial_ver.observation import ALLOWED_DECISION_TIMES


MINIMUM_ACQUISITION_INTERVAL_MS = min(
    DWELL_MS,
    visual_encoding_time_ms(
        0.0,
        K_seconds=(
            DEFAULT_ENCODING_K_SECONDS * RELAXED_SACCADE_ENCODING_SCALE
        ),
    ),
)
MAX_ACQUISITIONS_PER_DECISION = (
    int(max(ALLOWED_DECISION_TIMES) // MINIMUM_ACQUISITION_INTERVAL_MS) + 2
)
if MAX_ACQUISITIONS_PER_DECISION != 88:
    raise RuntimeError("canonical gaze timing changed; acquisition bound must be audited")

FIXATION_TO_CODE = {name: index for index, name in enumerate(FIXATION_SHAPES)}
CODE_TO_FIXATION = {value: key for key, value in FIXATION_TO_CODE.items()}


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    env_ids: tuple[int, ...]


def deterministic_assignments(
    n_rollout_threads: int, rollout_workers: int
) -> tuple[WorkerAssignment, ...]:
    environments = int(n_rollout_threads)
    workers = int(rollout_workers)
    if environments <= 0 or workers <= 0:
        raise ValueError("environment and worker counts must be positive")
    if workers > environments:
        raise ValueError("rollout workers may not exceed rollout environments")
    result = tuple(
        WorkerAssignment(
            worker_id=worker,
            env_ids=tuple(range(worker, environments, workers)),
        )
        for worker in range(workers)
    )
    flattened = sorted(env for assignment in result for env in assignment.env_ids)
    if flattened != list(range(environments)):
        raise RuntimeError("deterministic worker assignment is not exhaustive")
    return result
