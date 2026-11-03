"""Exact RNG-neutral Time x Horizon x fixation observability for Partial runs."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np

from final_model.partial_ver.conditions import PartialCondition


GROUP_NAMES = ("time_metrics", "horizon_metrics", "condition_metrics")
PUBLIC_METRIC_FIELDS = (
    "episode_count",
    "decisions",
    "sparse_reward",
    "sparse_reward_per_episode",
    "reward",
    "deliveries",
    "deliveries_per_episode",
    "forced_stays",
    "forced_stay_rate",
    "actor_active_rate",
    "acquisitions",
    "acquisitions_per_decision",
    "actionnet_query_count",
    "gaze_requests",
    "same_target_acquisitions",
    "new_target_acquisitions",
    "same_target_rate",
    "new_target_rate",
    "valid_memory_fraction",
    "mean_freshness",
    "mean_saccade_distance",
)
_INTEGER_ADDITIVE_FIELDS = (
    "deliveries",
    "task_decisions",
    "episodes",
    "forced_stays",
    "actor_active_decisions",
    "acquisitions",
    "actionnet_query_count",
    "gaze_requests",
    "same_target_acquisitions",
    "new_target_acquisitions",
    "valid_memory_tiles",
    "freshness_sample_count",
)
_FLOAT_ADDITIVE_FIELDS = (
    "optimization_reward",
    "sparse_team_reward",
    "freshness_normalized_sum",
    "saccade_distance_sum_degrees",
)


def condition_metric_id(condition: PartialCondition) -> str:
    return (
        f"t{int(condition.decision_time)}_h{int(condition.horizon)}_"
        f"{condition.fixation_shape}"
    )


def _new_bucket() -> dict[str, float | int]:
    return {
        **{field: 0 for field in _INTEGER_ADDITIVE_FIELDS},
        **{field: 0.0 for field in _FLOAT_ADDITIVE_FIELDS},
    }


def new_partial_metric_totals(
    conditions: Sequence[PartialCondition],
) -> dict[str, dict[str, dict[str, float | int]]]:
    active = tuple(conditions)
    if not active:
        raise ValueError("Partial observability requires active conditions")
    if len(set(active)) != len(active):
        raise ValueError("Partial observability conditions must be unique")
    return {
        "time_metrics": {
            f"t{value}": _new_bucket()
            for value in dict.fromkeys(
                int(condition.decision_time) for condition in active
            )
        },
        "horizon_metrics": {
            f"h{value}": _new_bucket()
            for value in dict.fromkeys(int(condition.horizon) for condition in active)
        },
        "condition_metrics": {
            condition_metric_id(condition): _new_bucket()
            for condition in active
        },
    }


def partial_observation_statistics(
    observations: Sequence[np.ndarray],
) -> dict[str, float | int]:
    """Reduce already-created ActionNet observations without new perception work."""
    values = tuple(observations)
    if not values:
        return {
            "valid_memory_tiles": 0,
            "freshness_sample_count": 0,
            "freshness_normalized_sum": 0.0,
        }
    arrays = [np.asarray(value) for value in values]
    if any(array.shape != (8, 5, 25) for array in arrays):
        raise RuntimeError("invalid Partial observation for observability")
    raw_freshness = np.stack([array[..., 20] for array in arrays]).astype(
        np.float64, copy=False
    )
    if not np.isfinite(raw_freshness).all() or np.any(
        raw_freshness < 0.0
    ) or np.any(
        raw_freshness > 255.0
    ):
        raise RuntimeError("invalid Partial freshness for observability")
    semantic_freshness = raw_freshness / 255.0
    return {
        "valid_memory_tiles": int(np.count_nonzero(semantic_freshness > 0.0)),
        "freshness_sample_count": int(semantic_freshness.size),
        "freshness_normalized_sum": float(semantic_freshness.sum()),
    }


def _array(
    value: Any,
    *,
    count: int,
    label: str,
    dtype: Any = None,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (count,):
        raise RuntimeError(f"invalid Partial observability {label} shape")
    return result


def accumulate_partial_metrics(
    totals: dict[str, dict[str, dict[str, float | int]]],
    *,
    conditions: Sequence[PartialCondition],
    rewards: np.ndarray,
    deliveries: Sequence[int],
    dones: Sequence[bool],
    acquisitions: Sequence[int],
    actor_active: Sequence[bool],
    actionnet_queries: Sequence[int] | None = None,
    gaze_requests: Sequence[int] | None = None,
    same_target_acquisitions: Sequence[int] | None = None,
    new_target_acquisitions: Sequence[int] | None = None,
    valid_memory_tiles: Sequence[int] | None = None,
    freshness_sample_counts: Sequence[int] | None = None,
    freshness_normalized_sums: Sequence[float] | None = None,
    saccade_distance_sums_degrees: Sequence[float] | None = None,
    require_acquisition_query_match: bool = True,
) -> None:
    """Accumulate one already-collected vectorized Partial environment step."""
    active = tuple(conditions)
    count = len(active)
    reward_array = np.asarray(rewards)
    if reward_array.ndim != 2 or reward_array.shape[0] != count or reward_array.shape[1] < 2:
        raise RuntimeError("invalid Partial observability reward shape")
    if not np.isfinite(reward_array[:, :2]).all():
        raise RuntimeError("non-finite Partial observability reward")
    delivery_array = _array(deliveries, count=count, label="deliveries", dtype=np.int64)
    done_array = _array(dones, count=count, label="dones", dtype=bool)
    acquisition_array = _array(
        acquisitions, count=count, label="acquisitions", dtype=np.int64
    )
    active_array = _array(actor_active, count=count, label="actor_active", dtype=bool)

    def optional(value, label, dtype, default):
        if value is None:
            return np.full(count, default, dtype=dtype)
        return _array(value, count=count, label=label, dtype=dtype)

    query_array = optional(
        actionnet_queries, "actionnet_queries", np.int64, 0
    )
    gaze_array = optional(gaze_requests, "gaze_requests", np.int64, 0)
    same_array = optional(
        same_target_acquisitions, "same_target_acquisitions", np.int64, 0
    )
    new_array = optional(
        new_target_acquisitions, "new_target_acquisitions", np.int64, 0
    )
    valid_array = optional(
        valid_memory_tiles, "valid_memory_tiles", np.int64, 0
    )
    freshness_count_array = optional(
        freshness_sample_counts, "freshness_sample_counts", np.int64, 0
    )
    freshness_sum_array = optional(
        freshness_normalized_sums, "freshness_normalized_sums", np.float64, 0.0
    )
    distance_array = optional(
        saccade_distance_sums_degrees,
        "saccade_distance_sums_degrees",
        np.float64,
        0.0,
    )
    integer_arrays = (
        delivery_array,
        acquisition_array,
        query_array,
        gaze_array,
        same_array,
        new_array,
        valid_array,
        freshness_count_array,
    )
    if any(np.any(value < 0) for value in integer_arrays):
        raise RuntimeError("negative Partial observability counter")
    if require_acquisition_query_match and np.any(
        query_array != acquisition_array
    ):
        raise RuntimeError("Partial acquisitions and ActionNet queries diverged")
    if np.any(same_array + new_array != acquisition_array):
        raise RuntimeError("Partial acquisition target counts diverged")
    if np.any(valid_array > freshness_count_array):
        raise RuntimeError("Partial valid-memory count exceeds samples")
    if not np.isfinite(freshness_sum_array).all() or not np.isfinite(
        distance_array
    ).all():
        raise RuntimeError("non-finite Partial observability diagnostic")

    for index, condition in enumerate(active):
        labels = (
            ("time_metrics", f"t{int(condition.decision_time)}"),
            ("horizon_metrics", f"h{int(condition.horizon)}"),
            ("condition_metrics", condition_metric_id(condition)),
        )
        values: Mapping[str, float | int] = {
            "optimization_reward": float(reward_array[index, 0]),
            "sparse_team_reward": float(reward_array[index, 1]),
            "deliveries": int(delivery_array[index]),
            "task_decisions": 1,
            "episodes": int(done_array[index]),
            "forced_stays": int(not active_array[index]),
            "actor_active_decisions": int(active_array[index]),
            "acquisitions": int(acquisition_array[index]),
            "actionnet_query_count": int(query_array[index]),
            "gaze_requests": int(gaze_array[index]),
            "same_target_acquisitions": int(same_array[index]),
            "new_target_acquisitions": int(new_array[index]),
            "valid_memory_tiles": int(valid_array[index]),
            "freshness_sample_count": int(freshness_count_array[index]),
            "freshness_normalized_sum": float(freshness_sum_array[index]),
            "saccade_distance_sum_degrees": float(distance_array[index]),
        }
        for group, label in labels:
            try:
                bucket = totals[group][label]
            except KeyError as error:
                raise RuntimeError(
                    f"inactive Partial condition observed: {group}/{label}"
                ) from error
            for field, value in values.items():
                bucket[field] += value


def _divide(numerator: float | int, denominator: float | int) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def finalize_partial_metrics(
    totals: Mapping[str, Mapping[str, Mapping[str, float | int]]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result = copy.deepcopy(totals)
    for group in GROUP_NAMES:
        for bucket in result[group].values():
            episodes = int(bucket["episodes"])
            decisions = int(bucket["task_decisions"])
            acquisitions = int(bucket["acquisitions"])
            samples = int(bucket["freshness_sample_count"])
            new_targets = int(bucket["new_target_acquisitions"])
            bucket.update({
                "reward": float(bucket["optimization_reward"]),
                "sparse_reward": float(bucket["sparse_team_reward"]),
                "decisions": decisions,
                "episode_count": episodes,
                "sparse_reward_per_episode": _divide(
                    bucket["sparse_team_reward"], episodes
                ),
                "deliveries_per_episode": _divide(bucket["deliveries"], episodes),
                "forced_stay_rate": _divide(bucket["forced_stays"], decisions),
                "actor_active_rate": _divide(
                    bucket["actor_active_decisions"], decisions
                ),
                "acquisitions_per_decision": _divide(acquisitions, decisions),
                "same_target_rate": _divide(
                    bucket["same_target_acquisitions"], acquisitions
                ),
                "new_target_rate": _divide(
                    bucket["new_target_acquisitions"], acquisitions
                ),
                "valid_memory_fraction": _divide(
                    bucket["valid_memory_tiles"], samples
                ),
                "mean_freshness": (
                    None
                    if samples == 0
                    else float(bucket["freshness_normalized_sum"]) / samples
                ),
                "mean_saccade_distance": _divide(
                    bucket["saccade_distance_sum_degrees"], new_targets
                ),
            })
    return result


def validate_partial_metric_totals(
    metrics: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    *,
    optimization_reward: float,
    sparse_reward: float,
    deliveries: int,
    episodes: int,
    decisions: int,
    forced_stays: int,
    acquisitions: int,
) -> None:
    expected_integer = {
        "deliveries": int(deliveries),
        "episodes": int(episodes),
        "task_decisions": int(decisions),
        "forced_stays": int(forced_stays),
        "acquisitions": int(acquisitions),
    }
    expected_float = {
        "optimization_reward": float(optimization_reward),
        "sparse_team_reward": float(sparse_reward),
    }
    for group in GROUP_NAMES:
        buckets = metrics[group].values()
        mismatch = {
            field: (sum(int(row[field]) for row in buckets), expected)
            for field, expected in expected_integer.items()
            if sum(int(row[field]) for row in metrics[group].values()) != expected
        }
        for field, expected in expected_float.items():
            actual = sum(float(row[field]) for row in metrics[group].values())
            if not np.isclose(actual, expected, rtol=1e-10, atol=1e-6):
                mismatch[field] = (actual, expected)
        if mismatch:
            raise RuntimeError(
                f"Partial {group} totals do not reconcile: {mismatch}"
            )


def flatten_partial_metric_namespaces(
    latest: Mapping[str, Any],
) -> dict[str, float | int]:
    """Create stable W&B/JSON slash keys from structured Partial summaries."""
    flattened: dict[str, float | int] = {}
    for group, prefix in (
        ("time_metrics", ""),
        ("horizon_metrics", ""),
        ("condition_metrics", "cell/"),
    ):
        for label, bucket in (latest.get(group) or {}).items():
            for field in PUBLIC_METRIC_FIELDS:
                value = bucket.get(field)
                if value is not None:
                    flattened[f"{prefix}{label}/{field}"] = value
    return flattened


__all__ = [
    "GROUP_NAMES",
    "accumulate_partial_metrics",
    "condition_metric_id",
    "finalize_partial_metrics",
    "flatten_partial_metric_namespaces",
    "new_partial_metric_totals",
    "partial_observation_statistics",
    "validate_partial_metric_totals",
]
