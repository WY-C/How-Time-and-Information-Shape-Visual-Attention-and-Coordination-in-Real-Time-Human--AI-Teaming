"""Episode-clustered statistics for the canonical gaze behavior suite."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_BOOTSTRAP_CHUNK_SIZE = 250


def _bootstrap_means(
    values: np.ndarray, *, iterations: int, seed: int,
    chunk_size: int = DEFAULT_BOOTSTRAP_CHUNK_SIZE, progress=None,
) -> np.ndarray:
    """Draw row means without materializing the full ``iterations x N`` matrix."""
    count = int(iterations)
    size = int(chunk_size)
    if count <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if size <= 0:
        raise ValueError("bootstrap chunk size must be positive")
    rng = np.random.default_rng(int(seed))
    result = np.empty(count, dtype=np.float64)
    for start in range(0, count, size):
        stop = min(start + size, count)
        draws = rng.integers(
            0, len(values), size=(stop - start, len(values))
        )
        result[start:stop] = values[draws].mean(axis=1)
        if progress is not None:
            progress(stop)
    return result


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(array, (2.5, 97.5))
    return float(low), float(high)


def cluster_bootstrap_mean(
    rows: Iterable[Mapping],
    value: Callable[[Mapping], float] | str,
    *,
    iterations: int,
    seed: int,
    cluster_fields: tuple[str, ...] = ("seed",),
    chunk_size: int = DEFAULT_BOOTSTRAP_CHUNK_SIZE,
    progress=None,
) -> dict[str, float | int]:
    """Bootstrap a row mean by resampling independent episode/seed clusters."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    getter = (lambda row: float(row[value])) if isinstance(value, str) else value
    event_count = 0
    for row in rows:
        try:
            candidate = getter(row)
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate):
            grouped[tuple(row[field] for field in cluster_fields)].append(candidate)
            event_count += 1
    cluster_values = np.asarray(
        [np.mean(values) for values in grouped.values()], dtype=np.float64
    )
    if not cluster_values.size:
        return {
            "mean": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "n_clusters": 0, "event_count": 0,
        }
    boot = _bootstrap_means(
        cluster_values, iterations=iterations, seed=seed,
        chunk_size=chunk_size, progress=progress,
    )
    low, high = percentile_interval(boot)
    return {
        "mean": float(cluster_values.mean()), "ci_low": low, "ci_high": high,
        "n_clusters": int(cluster_values.size), "event_count": event_count,
    }


def paired_cluster_bootstrap(
    left: Mapping[int, float],
    right: Mapping[int, float],
    *,
    iterations: int,
    seed: int,
    chunk_size: int = DEFAULT_BOOTSTRAP_CHUNK_SIZE,
    progress=None,
) -> dict[str, float | int]:
    """Return paired ``left - right`` contrast over common evaluation seeds."""
    common = sorted(set(left) & set(right))
    differences = np.asarray(
        [float(left[key]) - float(right[key]) for key in common], dtype=np.float64
    )
    differences = differences[np.isfinite(differences)]
    if not differences.size:
        return {
            "mean": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "n_pairs": 0,
        }
    boot = _bootstrap_means(
        differences, iterations=iterations, seed=seed,
        chunk_size=chunk_size, progress=progress,
    )
    low, high = percentile_interval(boot)
    return {
        "mean": float(differences.mean()), "ci_low": low,
        "ci_high": high, "n_pairs": int(differences.size),
    }


def paired_binary_bootstrap(
    pairs: Sequence[tuple[int, int, int]], *, iterations: int, seed: int,
    chunk_size: int = DEFAULT_BOOTSTRAP_CHUNK_SIZE, progress=None,
) -> dict[str, float | int]:
    """Bootstrap first-minus-second binary outcomes clustered by seed."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for cluster, first, second in pairs:
        grouped[int(cluster)].append(float(first) - float(second))
    values = {key: float(np.mean(items)) for key, items in grouped.items()}
    zero = {key: 0.0 for key in values}
    return paired_cluster_bootstrap(
        values, zero, iterations=iterations, seed=seed,
        chunk_size=chunk_size, progress=progress,
    )


def directional_status(ci_low: float, ci_high: float) -> str:
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "NOT_AVAILABLE"
    if ci_low > 0.0:
        return "POSITIVE"
    if ci_high < 0.0:
        return "NEGATIVE"
    return "NO_CLEAR_DIFFERENCE"
