"""Pure annotations and offline metrics for saved gaze behavior rollouts."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from final_model.eval.gaze_behavior_stats import (
    DEFAULT_BOOTSTRAP_CHUNK_SIZE,
    cluster_bootstrap_mean,
    directional_status,
    paired_binary_bootstrap,
    paired_cluster_bootstrap,
)
from final_model.eval.gaze_behavior_artifact import atomic_output_path
from final_model.partial_ver.gaze import ReferenceGeometry, acquisition_tiles
from final_model.partial_ver.preview_opacity import (
    rendered_preview_alphas,
    rendered_preview_probabilities,
)
from final_model.server_parallel.gaze_v1 import GAZE_ACTION_COUNT, GAZE_TILES


CONTENT_CHANNELS = (*range(20), *range(21, 25))
SELF_CHANNELS = (0, 2, 3, 4, 5)
PARTNER_CHANNELS = (1, 6, 7, 8, 9)
PREVIEW_CHANNELS = (21, 22, 23, 24)
FRESHNESS_CHANNEL = 20
AOIS = ("self", "partner", "preview", "h1", "h2", "h3", "h4", "near", "far")
ANALYSIS_STAGES = (
    "load_episodes", "load_decisions", "load_requests", "load_acquisitions",
    "response_annotation", "task_performance", "spatial_gaze", "aoi",
    "peripheral", "opacity", "enrichment",
    "acquisition_impact", "content_freshness", "response_time",
    "revision_association", "stabilization_monitoring", "proximity",
    "coordination_states", "semantic_revisions", "memory_economy",
    "cue_distance", "input_ablation", "bootstrap_ci", "trend_summary",
)


def decision_relative_timestamp(
    absolute_timestamp_ms: float, decision_start_ms: float
) -> float:
    value = float(absolute_timestamp_ms) - float(decision_start_ms)
    if not np.isfinite(value) or value < -1e-8:
        raise ValueError("decision-relative timestamp must be finite and non-negative")
    return max(value, 0.0)


def top1_stable_index(actions: Sequence[int]) -> int | None:
    if not actions:
        return None
    final = int(actions[-1])
    for index in range(len(actions)):
        if all(int(action) == final for action in actions[index:]):
            return index
    raise AssertionError("the final proposal must define a stable suffix")


def first_occurrence(actions: Sequence[int], target: int) -> int | None:
    return next((index for index, value in enumerate(actions) if int(value) == int(target)), None)


def _validate_probability(value: Sequence[float], *, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"probability vector must be finite with shape ({size},)")
    if not np.isclose(array.sum(), 1.0, atol=1e-6) or np.any(array < 0.0):
        raise ValueError("probabilities must be non-negative and sum to one")
    return array


def probability_metrics(left: Sequence[float], right: Sequence[float]) -> dict[str, float | int]:
    first = _validate_probability(left, size=len(left))
    second = _validate_probability(right, size=len(left))
    epsilon = np.finfo(np.float64).tiny
    safe_first = np.clip(first, epsilon, 1.0)
    safe_second = np.clip(second, epsilon, 1.0)
    middle = 0.5 * (safe_first + safe_second)
    absolute = np.abs(first - second)
    entropy_first = float(-np.sum(first * np.log(safe_first)))
    entropy_second = float(-np.sum(second * np.log(safe_second)))
    return {
        "top1_changed": int(first.argmax() != second.argmax()),
        "kl": float(max(np.sum(first * np.log(safe_first / safe_second)), 0.0)),
        "js": float(max(
            0.5 * np.sum(first * np.log(safe_first / middle))
            + 0.5 * np.sum(second * np.log(safe_second / middle)), 0.0
        )),
        "tv": float(0.5 * absolute.sum()),
        "mean_abs_delta_p": float(absolute.mean()),
        "entropy_before": entropy_first,
        "entropy_after": entropy_second,
        "delta_entropy": entropy_second - entropy_first,
    }


def classify_memory_change(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    left = np.asarray(before, dtype=np.float32)
    right = np.asarray(after, dtype=np.float32)
    if left.shape != (8, 5, 25) or right.shape != left.shape:
        raise ValueError("memory snapshots must have shape (8,5,25)")
    content_delta = left[..., CONTENT_CHANNELS] != right[..., CONTENT_CHANNELS]
    freshness_delta = left[..., FRESHNESS_CHANNEL] != right[..., FRESHNESS_CHANNEL]
    content_tiles = np.any(content_delta, axis=-1)
    content = bool(np.any(content_tiles))
    freshness = bool(np.any(freshness_delta))
    if content and freshness:
        label = "content_and_freshness_changed"
    elif content:
        label = "content_changed"
    elif freshness:
        label = "freshness_only_changed"
    else:
        label = "no_effective_change"
    changed_channels = set(
        np.flatnonzero(np.any(left != right, axis=(0, 1))).tolist()
    )
    return {
        "content_change_class": label,
        "content_changed": int(content),
        "freshness_changed": int(freshness),
        "dynamic_content_changed": int(bool(changed_channels & set(range(20)))),
        "self_content_changed": int(bool(changed_channels & set(SELF_CHANNELS))),
        "partner_content_changed": int(bool(changed_channels & set(PARTNER_CHANNELS))),
        "preview_content_changed": int(bool(changed_channels & set(PREVIEW_CHANNELS))),
        "content_novelty_count": int(np.count_nonzero(content_tiles)),
        "freshness_changed_count": int(np.count_nonzero(freshness_delta)),
        "freshness_only_changed_count": int(
            np.count_nonzero(freshness_delta & ~content_tiles)
        ),
    }


def target_hit_mask(aoi_mask: np.ndarray, *, shape: str, footprint: bool) -> np.ndarray:
    aoi = np.asarray(aoi_mask, dtype=bool)
    if aoi.shape != (8, 5):
        raise ValueError("AOI mask must have shape (8,5)")
    result = np.zeros(GAZE_ACTION_COUNT, dtype=bool)
    for index, target in enumerate(GAZE_TILES):
        result[index] = (
            bool(aoi[target])
            if not footprint
            else any(bool(aoi[tile]) for tile in acquisition_tiles(target, shape))
        )
    return result


def enrichment_values(
    probabilities: Sequence[float], hit_mask: Sequence[bool], sampled_index: int
) -> dict[str, float | int]:
    probability = _validate_probability(probabilities, size=GAZE_ACTION_COUNT)
    mask = np.asarray(hit_mask, dtype=bool)
    if mask.shape != (GAZE_ACTION_COUNT,):
        raise ValueError("hit mask must contain all 40 gaze actions")
    chance = float(mask.mean())
    policy = float(probability[mask].sum())
    sampled = int(mask[int(sampled_index)])
    return {
        "chance_mass": chance,
        "policy_mass": policy,
        "mass_enrichment": 0.0 if chance == 0.0 else policy / chance,
        "sample_hit": sampled,
        "sample_enrichment": 0.0 if chance == 0.0 else sampled / chance,
    }


def exclusive_aoi(row: Mapping[str, Any], precedence: Sequence[str], *, suffix: str) -> str:
    for label in precedence:
        if _integer(row.get(f"{label}_{suffix}", 0)):
            return label
    return "other"


def revision_risk(rows: Sequence[Mapping[str, Any]], exposure: str) -> dict[str, float | int]:
    eligible = [row for row in rows if _integer(row.get("proposal_index", 0)) > 0]
    exposed = [_integer(row["online_top1_changed_from_previous"]) for row in eligible if _integer(row.get(exposure, 0))]
    unexposed = [_integer(row["online_top1_changed_from_previous"]) for row in eligible if not _integer(row.get(exposure, 0))]
    risk_e = float(np.mean(exposed)) if exposed else float("nan")
    risk_u = float(np.mean(unexposed)) if unexposed else float("nan")
    return {
        "eligible_transitions": len(eligible), "exposed_n": len(exposed),
        "unexposed_n": len(unexposed), "risk_exposed": risk_e,
        "risk_unexposed": risk_u,
        "risk_difference": risk_e - risk_u,
        "risk_ratio": risk_e / risk_u if risk_u > 0.0 else float("nan"),
    }


def _chunked_cluster_counts(
    counts: np.ndarray, *, iterations: int, seed: int, progress=None,
    chunk_size: int = DEFAULT_BOOTSTRAP_CHUNK_SIZE,
):
    """Yield resampled sums of per-seed sufficient statistics."""
    rng = np.random.default_rng(int(seed))
    for start in range(0, int(iterations), int(chunk_size)):
        stop = min(start + int(chunk_size), int(iterations))
        indices = rng.integers(
            0, len(counts), size=(stop - start, len(counts))
        )
        yield stop, counts[indices].sum(axis=1)
        if progress is not None:
            progress(stop)


def _revision_seed_counts(rows, exposure) -> np.ndarray:
    clusters = []
    for values in _group(rows, ("seed",)).values():
        counts = [0, 0, 0, 0]
        for row in values:
            if _integer(row.get("proposal_index", 0)) <= 0:
                continue
            exposed = bool(_integer(row.get(exposure, 0)))
            changed = bool(_integer(row["online_top1_changed_from_previous"]))
            counts[(0 if exposed else 2) + (0 if changed else 1)] += 1
        clusters.append(counts)
    return np.asarray(clusters, dtype=np.int64)


def _revision_ratio_bootstrap(
    rows, exposure, *, iterations, seed, progress=None,
) -> np.ndarray:
    counts = _revision_seed_counts(rows, exposure)
    if not len(counts):
        return np.empty(0, dtype=np.float64)
    results = []
    for _stop, total in _chunked_cluster_counts(
        counts, iterations=iterations, seed=seed, progress=progress,
    ):
        a, b, c, d = total.T
        with np.errstate(divide="ignore", invalid="ignore"):
            risk_exposed = a / (a + b)
            risk_unexposed = c / (c + d)
            ratio = np.where(risk_unexposed > 0.0,
                             risk_exposed / risk_unexposed, np.nan)
        results.append(ratio[np.isfinite(ratio)])
    return np.concatenate(results) if results else np.empty(0, dtype=np.float64)


def stabilization_first_post_pairs(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping, Mapping]]:
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[_decision_key(row)].append(row)
    pairs = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda row: _integer(row["proposal_index"]))
        at = next((row for row in ordered if row.get("response_phase") == "AT_RESPONSE"), None)
        post = next((row for row in ordered if row.get("response_phase") == "POST"), None)
        if at is not None and post is not None:
            pairs.append((at, post))
    return pairs


def manhattan_distance(first: Sequence[int], second: Sequence[int]) -> int:
    return abs(int(first[0]) - int(second[0])) + abs(int(first[1]) - int(second[1]))


def proximity_bin(distance: int, bins: Sequence[int] = (1, 3)) -> str:
    near, far = map(int, bins)
    if distance <= near:
        return "adjacent"
    if distance <= far:
        return "near"
    return "far"


def minimum_gaze_distance(
    fixation: Sequence[int], candidates: Iterable[Sequence[int]]
) -> float | None:
    geometry = ReferenceGeometry()
    values = [geometry.visual_angle_degrees(fixation, candidate) for candidate in candidates]
    return None if not values else float(min(values))


def cue_distance_choice(
    *, fixation: Sequence[int], near_candidates: Sequence[Sequence[int]],
    far_candidates: Sequence[Sequence[int]], first_preview: str | None,
) -> dict[str, Any]:
    near = minimum_gaze_distance(fixation, near_candidates)
    far = minimum_gaze_distance(fixation, far_candidates)
    return {
        "near_distance_degrees": near,
        "far_distance_degrees": far,
        "far_closer": None if near is None or far is None else int(far < near),
        "first_preview_far": None if first_preview is None else int(first_preview == "far"),
    }


def _integer(value: Any) -> int:
    if value in (None, "", "None"):
        return 0
    return int(float(value))


def _float(value: Any) -> float:
    if value in (None, "", "None"):
        return float("nan")
    return float(value)


def _decision_key(row: Mapping[str, Any]) -> tuple:
    return (
        row.get("model_name"), _integer(row.get("T")), _integer(row.get("H")),
        _integer(row.get("seed")), _integer(row.get("decision")),
    )


def read_csv(
    path: Path, columns: Iterable[str] | None = None,
    exclude=None,
) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if columns is not None and exclude is not None:
            raise ValueError("columns and exclude are mutually exclusive")
        if exclude is not None:
            columns = [
                field for field in (reader.fieldnames or ())
                if not exclude(field)
            ]
        if columns is None:
            return list(reader)
        selected = tuple(dict.fromkeys(columns))
        missing = set(selected) - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"{path} is missing analysis columns: {sorted(missing)}"
            )
        return [
            {field: row[field] for field in selected}
            for row in reader
        ]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with atomic_output_path(path) as temporary:
            temporary.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)


def _mean(rows: Sequence[Mapping], field: str) -> float:
    values = [_float(row.get(field)) for row in rows]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _group(rows: Sequence[Mapping], fields: Sequence[str]) -> dict[tuple, list[Mapping]]:
    result: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        result[tuple(row.get(field) for field in fields)].append(row)
    return result


def annotate_response_phases(
    acquisitions: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_decision = defaultdict(list)
    for row in acquisitions:
        by_decision[_decision_key(row)].append(row)
    response_rows = []
    for decision in decisions:
        values = sorted(
            by_decision.get(_decision_key(decision), ()),
            key=lambda row: _integer(row["proposal_index"]),
        )
        top1 = [_integer(row["online_top1_action"]) for row in values]
        sampled = [_integer(row["sampled_action"]) for row in values]
        stable = top1_stable_index(top1)
        sampled_stable = top1_stable_index(sampled)
        for index, row in enumerate(values):
            row["response_phase"] = (
                "PRE" if stable is not None and index < stable
                else "AT_RESPONSE" if stable is not None and index == stable
                else "POST" if stable is not None
                else "NO_RESPONSE"
            )
        timestamp = None if stable is None else _float(values[stable]["decision_relative_timestamp_ms"])
        sampled_timestamp = None if sampled_stable is None else _float(values[sampled_stable]["decision_relative_timestamp_ms"])
        response_rows.append({
            **{key: decision.get(key) for key in ("model_name", "T", "H", "episode", "seed", "decision")},
            "response_defined": int(stable is not None),
            "top1_stable_proposal_index": stable,
            "top1_stable_rt_ms": timestamp,
            "top1_stable_rt_fraction": None if timestamp is None else timestamp / _float(decision["T"]),
            "top1_stable_remaining_ms": None if timestamp is None else _float(decision["T"]) - timestamp,
            "sampled_stable_proposal_index": sampled_stable,
            "sampled_stable_rt_ms": sampled_timestamp,
            "first_final_top1_index": None if not top1 else first_occurrence(top1, top1[-1]),
            "first_final_sampled_index": None if not sampled else first_occurrence(sampled, sampled[-1]),
            "pre_acquisition_count": sum(row.get("response_phase") == "PRE" for row in values),
            "at_response_acquisition_count": sum(row.get("response_phase") == "AT_RESPONSE" for row in values),
            "post_acquisition_count": sum(row.get("response_phase") == "POST" for row in values),
            "post_any": int(any(row.get("response_phase") == "POST" for row in values)),
        })
    return response_rows


def _condition_rows(rows: Sequence[Mapping]) -> dict[tuple[int, int], list[Mapping]]:
    result: dict[tuple[int, int], list[Mapping]] = defaultdict(list)
    for row in rows:
        result[(_integer(row["T"]), _integer(row["H"]))].append(row)
    return result


def analyze_saved_model(
    model_dir: Path,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    aoi_precedence: Sequence[str],
    gaze_actor=None,
    gaze_device=None,
    progress=None,
    progress_model_name: str = "model",
    projected: bool = True,
) -> dict[str, Any]:
    """Read rich CSV/NPZ data and write every analysis without environment access."""
    rollout_manifest = json.loads(
        (model_dir / "rollout_manifest.json").read_text(encoding="utf-8")
    )
    opacity_mode = str(rollout_manifest.get("preview_opacity_mode", "off"))

    def diagnostics(**extra):
        result = {
            "episodes": len(locals_data.get("episodes", ())),
            "decisions": len(locals_data.get("decisions", ())),
            "requests": len(locals_data.get("requests", ())),
            "acquisitions": len(locals_data.get("acquisitions", ())),
            "bootstrap": bootstrap_iterations,
        }
        result.update(extra)
        return result

    def start_stage(name: str, **extra):
        if progress is None:
            return None
        return progress.start_analysis(
            model=progress_model_name, name=name,
            index=ANALYSIS_STAGES.index(name) + 1,
            total=len(ANALYSIS_STAGES),
            diagnostics=diagnostics(**extra),
        )

    def done_stage(name: str, started, **extra) -> None:
        if progress is not None:
            progress.done_analysis(
                name, started, diagnostics=diagnostics(**extra)
            )

    locals_data = {}

    def load_rows(name, path, **kwargs):
        started = start_stage(name)
        rows = read_csv(path, **kwargs)
        locals_data[name.removeprefix("load_")] = rows
        done_stage(name, started, rows=len(rows), columns=len(rows[0]) if rows else 0)
        return rows

    episodes = load_rows("load_episodes", model_dir / "episodes.csv")
    decisions = load_rows("load_decisions", model_dir / "decisions.csv")
    requests = load_rows(
        "load_requests", model_dir / "gaze_requests.csv.gz",
        exclude=(lambda field: (
            field.startswith("p") and field[1:].isdigit()
        ) or field.endswith("_compatible_target_count")
        or field in {"checkpoint_step", "checkpoint_sha256"})
        if projected else None,
    )
    unused_acquisition_fields = {
        "checkpoint_step", "checkpoint_sha256", "absolute_timestamp_ms",
        "decision_start_ms", "remaining_time_ms", "fixation_x", "fixation_y",
        "acquisition_footprint_tiles", "acquisition_all_success_tiles",
        "foveal_tile_indices", "peripheral_candidate_indices",
        "peripheral_candidate_eccentricity_deg",
        "peripheral_candidate_detection_probability", "peripheral_success_mask",
        "peripheral_success_indices", "peripheral_failure_indices",
        "opacity_opportunity_tile_indices", "opacity_opportunity_steps",
        "opacity_opportunity_max_alphas", "opacity_attempt_tile_indices",
        "opacity_attempt_probabilities", "opacity_attempt_positive_steps",
        "opacity_attempt_max_alphas", "opacity_success_tile_indices",
        "opacity_success_steps", "opacity_miss_tile_indices",
        "opacity_miss_steps", "self_tile_x", "self_tile_y", "partner_tile_x",
        "partner_tile_y", "self_holding", "partner_holding",
        "pot_onion_count_total", "onion_resource_tile_count",
        "dish_resource_tile_count", "serve_resource_tile_count",
        "partner_distance_metric", "pre_valid_fraction", "post_valid_fraction",
        "pre_mean_freshness", "post_mean_freshness", "pre_target_stale_fraction",
        "footprint_tiles_refreshed", "expired_count", "top1_before", "top1_after",
        "sampled_action_changed_from_previous", "final_executed_action",
        "final_proposal_index", "proposal_was_final",
    }
    acquisitions = load_rows(
        "load_acquisitions", model_dir / "acquisitions.csv.gz",
        exclude=(lambda field: field in unused_acquisition_fields
        or field.startswith(("before_p", "after_p", "online_p")))
        if projected else None,
    )
    stage_started = start_stage("response_annotation")
    response = annotate_response_phases(acquisitions, decisions)
    done_stage("response_annotation", stage_started, rows=len(response))

    stage_started = start_stage("task_performance")
    task_rows = []
    for (time_ms, horizon), rows in _condition_rows(episodes).items():
        matching_decisions = [row for row in decisions if _integer(row["T"]) == time_ms and _integer(row["H"]) == horizon]
        matching_acq = [row for row in acquisitions if _integer(row["T"]) == time_ms and _integer(row["H"]) == horizon]
        task_rows.append({
            "T": time_ms, "H": horizon, "episodes": len(rows),
            "deliveries_mean": _mean(rows, "deliveries"),
            "deliveries_std": float(np.std([_float(row["deliveries"]) for row in rows])),
            "sparse_reward_mean": _mean(rows, "sparse_team_reward"),
            "collision_count": sum(_integer(row["collision_count"]) for row in rows),
            "collision_rate": (
                sum(_integer(row["collision_count"]) for row in rows)
                / max(sum(_integer(row["task_decisions"]) for row in rows), 1)
            ),
            "forced_stay_rate": _mean(matching_decisions, "forced_stay"),
            "actionnet_queries_per_decision": len(matching_acq) / max(len(matching_decisions), 1),
            "acquisitions_per_decision": len(matching_acq) / max(len(matching_decisions), 1),
        })
    task_aggregates = []
    for dimension, values in (("T", sorted({_integer(row["T"]) for row in episodes})), ("H", sorted({_integer(row["H"]) for row in episodes}))):
        for value in values:
            episode_subset = [row for row in episodes if _integer(row[dimension]) == value]
            decision_subset = [row for row in decisions if _integer(row[dimension]) == value]
            acquisition_subset = [row for row in acquisitions if _integer(row[dimension]) == value]
            task_aggregates.append({
                "T": value if dimension == "T" else "pooled",
                "H": value if dimension == "H" else "pooled",
                "aggregation": f"{dimension}_mean", "episodes": len(episode_subset),
                "deliveries_mean": _mean(episode_subset, "deliveries"),
                "deliveries_std": float(np.std([_float(row["deliveries"]) for row in episode_subset])),
                "sparse_reward_mean": _mean(episode_subset, "sparse_team_reward"),
                "collision_count": sum(
                    _integer(row["collision_count"]) for row in episode_subset
                ),
                "collision_rate": (
                    sum(_integer(row["collision_count"]) for row in episode_subset)
                    / max(sum(_integer(row["task_decisions"]) for row in episode_subset), 1)
                ),
                "forced_stay_rate": _mean(decision_subset, "forced_stay"),
                "actionnet_queries_per_decision": len(acquisition_subset) / max(len(decision_subset), 1),
                "acquisitions_per_decision": len(acquisition_subset) / max(len(decision_subset), 1),
            })
    write_csv(model_dir / "task_performance.csv", task_rows + task_aggregates)
    done_stage("task_performance", stage_started)

    stage_started = start_stage("spatial_gaze")
    spatial_rows = []
    distributions: dict[tuple[int, int], np.ndarray] = {}
    for condition, rows in _condition_rows(requests).items():
        counts = np.zeros(GAZE_ACTION_COUNT, dtype=np.float64)
        landing = np.zeros(GAZE_ACTION_COUNT, dtype=np.float64)
        for row in rows:
            counts[_integer(row["gaze_action_index"])] += 1
        acq_rows = [row for row in acquisitions if (_integer(row["T"]), _integer(row["H"])) == condition]
        for row in acq_rows:
            landing[_integer(row["landing_action_index"])] += 1
        distribution = counts / max(counts.sum(), 1.0)
        landing_distribution = landing / max(landing.sum(), 1.0)
        distributions[condition] = distribution
        positive = distribution[distribution > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        spatial_rows.append({
            "T": condition[0], "H": condition[1], "request_count": len(rows),
            "acquisition_count": len(acq_rows), "top1_target_share": float(distribution.max()),
            "top5_target_share": float(np.sort(distribution)[-5:].sum()),
            "empirical_target_entropy": entropy,
            "kl_from_uniform": math.log(GAZE_ACTION_COUNT) - entropy,
            "request_vs_landing_tv": float(0.5 * np.abs(distribution - landing_distribution).sum()),
            "mean_saccade_distance_degrees": _mean(rows, "saccade_distance_degrees"),
            "same_target_rate": _mean(rows, "same_target"),
            **{f"target_{index}_share": float(distribution[index]) for index in range(GAZE_ACTION_COUNT)},
        })
    write_csv(model_dir / "spatial_gaze.csv", spatial_rows)

    js_rows = []
    for left, left_dist in distributions.items():
        for right, right_dist in distributions.items():
            js_rows.append({
                "left_T": left[0], "left_H": left[1], "right_T": right[0], "right_H": right[1],
                "js": probability_metrics(left_dist, right_dist)["js"],
            })
    write_csv(model_dir / "spatial_js_matrix.csv", js_rows)
    done_stage("spatial_gaze", stage_started)

    stage_started = start_stage("aoi")
    aoi_rows = []
    for condition, rows in _condition_rows(acquisitions).items():
        decision_count = sum(_integer(row["T"]) == condition[0] and _integer(row["H"]) == condition[1] for row in decisions)
        for definition in ("center", "footprint"):
            for aoi in ("self", "partner", "preview", "h1", "h2", "h3", "h4", "near", "far"):
                field = f"{aoi}_{definition}"
                hit_decisions = {_decision_key(row) for row in rows if _integer(row.get(field, 0))}
                aoi_rows.append({
                    "T": condition[0], "H": condition[1], "definition": definition,
                    "aoi": aoi, "event_count": len(rows),
                    "acquisition_share": _mean(rows, field),
                    "any_acquisition_decision_rate": len(hit_decisions) / max(decision_count, 1),
                    "count_per_decision": sum(_integer(row.get(field, 0)) for row in rows) / max(decision_count, 1),
                    "exclusive_precedence": ">".join(aoi_precedence) + ">other",
                })
    write_csv(model_dir / "aoi_summary.csv", aoi_rows)
    done_stage("aoi", stage_started)

    stage_started = start_stage("peripheral")
    peripheral_rows = []
    peripheral_impact_rows = []
    for condition, rows in _condition_rows(acquisitions).items():
        condition_decisions = [
            row for row in decisions
            if (_integer(row["T"]), _integer(row["H"])) == condition
        ]
        decision_count = len(condition_decisions)
        for phase in ("ALL", "PRE", "AT_RESPONSE", "POST"):
            subset = (
                rows if phase == "ALL"
                else [row for row in rows if row.get("response_phase") == phase]
            )
            attempts = sum(
                _integer(row.get("peripheral_candidate_count", 0))
                for row in subset
            )
            successes = sum(
                _integer(row.get("peripheral_success_count", 0))
                for row in subset
            )
            values = {
                "T": condition[0], "H": condition[1],
                "response_phase": phase,
                "acquisition_count": len(subset),
                "peripheral_attempts": attempts,
                "peripheral_successes": successes,
                "peripheral_failures": attempts - successes,
                "peripheral_attempts_per_acquisition": (
                    attempts / max(len(subset), 1)
                ),
                "peripheral_successes_per_acquisition": (
                    successes / max(len(subset), 1)
                ),
                "peripheral_success_rate": successes / max(attempts, 1),
                "peripheral_successes_per_decision": (
                    successes / max(decision_count, 1)
                ),
                "mean_attempted_eccentricity_deg": (
                    sum(_float(row.get(
                        "peripheral_attempt_eccentricity_sum_deg", 0.0
                    )) for row in subset) / max(attempts, 1)
                ),
                "mean_successful_eccentricity_deg": (
                    sum(_float(row.get(
                        "peripheral_success_eccentricity_sum_deg", 0.0
                    )) for row in subset) / max(successes, 1)
                ),
            }
            for aoi in ("self", "partner", "preview", "h1", "h2", "h3", "h4"):
                for source in ("foveal", "peripheral", "total"):
                    field = f"{aoi}_{source}_acquired"
                    values[f"{field}_share"] = _mean(subset, field)
                peripheral_decisions = {
                    _decision_key(row) for row in subset
                    if _integer(row.get(f"{aoi}_peripheral_acquired", 0))
                }
                values[f"any_{aoi}_peripheral_acquisition_decision_rate"] = (
                    len(peripheral_decisions) / max(decision_count, 1)
                )
            peripheral_rows.append(values)

        for source in (
            "foveal_only_change", "peripheral_success_change",
            "no_effective_change",
        ):
            subset = [
                row for row in rows
                if row.get("acquisition_information_source") == source
            ]
            peripheral_impact_rows.append({
                "T": condition[0], "H": condition[1],
                "information_source": source,
                "event_count": len(subset),
                "top1_change_rate": _mean(
                    subset, "counterfactual_top1_changed"
                ),
                "tv": _mean(subset, "counterfactual_tv"),
                "kl": _mean(subset, "counterfactual_kl"),
                "delta_entropy": _mean(
                    subset, "counterfactual_delta_entropy"
                ),
            })
    write_csv(model_dir / "peripheral_summary.csv", peripheral_rows)
    write_csv(
        model_dir / "peripheral_actionnet_impact.csv", peripheral_impact_rows
    )
    done_stage("peripheral", stage_started)

    stage_started = start_stage("opacity")
    opacity_rows = []
    opacity_impact_rows = []

    def logged_list(row, field):
        value = row.get(field, [])
        if isinstance(value, str):
            value = json.loads(value) if value else []
        return list(value)

    for condition, rows in _condition_rows(acquisitions).items():
        condition_decisions = [
            row for row in decisions
            if (_integer(row["T"]), _integer(row["H"])) == condition
        ]
        decision_count = len(condition_decisions)
        alphas = rendered_preview_alphas(condition[1])
        probabilities = rendered_preview_probabilities(condition[1])
        for phase in ("ALL", "PRE", "AT_RESPONSE", "POST"):
            subset = (
                rows if phase == "ALL"
                else [row for row in rows if row.get("response_phase") == phase]
            )
            opportunities = sum(
                _integer(row.get("opacity_opportunity_count", 0))
                for row in subset
            )
            attempts = sum(
                _integer(row.get("opacity_attempt_count", 0)) for row in subset
            )
            successes = sum(
                _integer(row.get("opacity_success_count", 0)) for row in subset
            )
            effective = opportunities if opacity_mode == "off" else successes
            effective_h_steps = sum(
                sum(
                    len(positive_steps) if opacity_mode == "off" or success else 0
                    for positive_steps, success in zip(
                        logged_list(row, "opacity_opportunity_positive_steps"),
                        (
                            [True] * _integer(row.get("opacity_opportunity_count", 0))
                            if opacity_mode == "off"
                            else logged_list(row, "opacity_success_mask")
                        ),
                    )
                )
                for row in subset
            )
            max_probabilities = [
                float(probability)
                for row in subset
                for probability in logged_list(
                    row, "opacity_opportunity_probabilities"
                )
            ]
            values = {
                "T": condition[0], "H": condition[1],
                "response_phase": phase,
                "preview_opacity_mode": opacity_mode,
                "acquisition_count": len(subset),
                "peripheral_spatial_preview_opportunities": opportunities,
                "opacity_attempts": attempts,
                "opacity_successes": successes,
                "opacity_misses": attempts - successes,
                "opacity_success_rate": successes / max(attempts, 1),
                "one_attempt_per_preview_tile": int(
                    attempts == opportunities if opacity_mode == "actual"
                    else attempts == 0
                ),
                "effective_preview_detections": effective,
                "effective_h_step_acquisitions": effective_h_steps,
                "effective_preview_detections_per_acquisition": (
                    effective / max(len(subset), 1)
                ),
                "effective_preview_detections_per_decision": (
                    effective / max(decision_count, 1)
                ),
                "opacity_preview_tiles": opportunities,
                "overlap_preview_tiles": sum(
                    _integer(row.get("opacity_overlap_preview_tile_count", 0))
                    for row in subset
                ),
                "overlap_step_instances": sum(
                    _integer(row.get("opacity_overlap_step_instance_count", 0))
                    for row in subset
                ),
                "positive_step_instances": sum(
                    sum(logged_list(
                        row, "opacity_opportunity_positive_step_counts"
                    )) for row in subset
                ),
                "mean_max_alpha_probability": (
                    float(np.mean(max_probabilities)) if max_probabilities else 0.0
                ),
            }
            for step in range(1, 5):
                step_attempts = sum(
                    logged_list(row, "opacity_attempt_steps").count(step)
                    for row in subset
                )
                step_successes = sum(
                    sum(
                        int(success)
                        for logged_step, success in zip(
                            logged_list(row, "opacity_attempt_steps"),
                            logged_list(row, "opacity_success_mask"),
                        )
                        if int(logged_step) == step
                    )
                    for row in subset
                )
                step_opportunities = sum(
                    sum(step in positive_steps for positive_steps in logged_list(
                        row, "opacity_opportunity_positive_steps"
                    )) for row in subset
                )
                step_effective = sum(
                    sum(
                        step in positive_steps
                        and (opacity_mode == "off" or bool(success))
                        for positive_steps, success in zip(
                            logged_list(row, "opacity_opportunity_positive_steps"),
                            (
                                [True] * _integer(row.get("opacity_opportunity_count", 0))
                                if opacity_mode == "off"
                                else logged_list(row, "opacity_success_mask")
                            ),
                        )
                    ) for row in subset
                )
                values.update({
                    f"renderer_alpha_h{step}": alphas[step - 1],
                    f"nominal_opacity_probability_h{step}": (
                        probabilities[step - 1]
                    ),
                    f"opacity_opportunities_h{step}": step_opportunities,
                    f"opacity_attempts_h{step}": step_attempts,
                    f"opacity_successes_h{step}": step_successes,
                    f"opacity_success_rate_h{step}": (
                        step_successes / max(step_attempts, 1)
                    ),
                    f"effective_preview_detections_h{step}": step_effective,
                })
            opacity_rows.append(values)

        for source in (
            "peripheral_without_opacity_detected_preview",
            "peripheral_with_opacity_detected_preview",
            "no_peripheral_success",
        ):
            subset = [
                row for row in rows
                if row.get("opacity_information_source") == source
            ]
            opacity_impact_rows.append({
                "T": condition[0], "H": condition[1],
                "information_source": source, "event_count": len(subset),
                "top1_change_rate": _mean(subset, "counterfactual_top1_changed"),
                "tv": _mean(subset, "counterfactual_tv"),
                "kl": _mean(subset, "counterfactual_kl"),
                "delta_entropy": _mean(
                    subset, "counterfactual_delta_entropy"
                ),
            })
    write_csv(model_dir / "opacity_summary.csv", opacity_rows)
    write_csv(model_dir / "opacity_actionnet_impact.csv", opacity_impact_rows)
    done_stage("opacity", stage_started)

    stage_started = start_stage("enrichment")
    enrichment_rows = []
    for condition, rows in _condition_rows(requests).items():
        for definition in ("center", "footprint"):
            for aoi in AOIS:
                prefix = f"{aoi}_{definition}"
                chance = _mean(rows, f"{prefix}_chance_mass")
                policy = _mean(rows, f"{prefix}_policy_mass")
                hit = _mean(rows, f"{prefix}_sample_hit")
                enrichment_rows.append({
                    "T": condition[0], "H": condition[1], "definition": definition,
                    "aoi": aoi, "request_count": len(rows), "chance_mass": chance,
                    "policy_mass": policy,
                    "mass_enrichment": _mean(rows, f"{prefix}_mass_enrichment"),
                    "sample_hit_rate": hit,
                    "sample_enrichment": _mean(rows, f"{prefix}_sample_enrichment"),
                })
    write_csv(model_dir / "enrichment.csv", enrichment_rows)
    done_stage("enrichment", stage_started)

    stage_started = start_stage("acquisition_impact")
    impact_rows = []
    for (time_ms, horizon), rows in _condition_rows(acquisitions).items():
        for definition in ("center", "footprint"):
            for aoi in ("self", "partner", "preview", "other"):
                selected = [row for row in rows if (exclusive_aoi(row, aoi_precedence, suffix=definition) == aoi)]
                for phase in ("ALL", "PRE", "AT_RESPONSE", "POST"):
                    phase_rows = selected if phase == "ALL" else [row for row in selected if row.get("response_phase") == phase]
                    for content_class in ("ALL", "content_changed", "freshness_only_changed", "content_and_freshness_changed", "no_effective_change"):
                        subset = phase_rows if content_class == "ALL" else [row for row in phase_rows if row.get("content_change_class") == content_class]
                        impact_rows.append({
                            "T": time_ms, "H": horizon, "definition": definition,
                            "aoi": aoi, "response_phase": phase,
                            "content_change_class": content_class,
                            "event_count": len(subset),
                            "top1_change_rate": _mean(subset, "counterfactual_top1_changed"),
                            "kl": _mean(subset, "counterfactual_kl"), "js": _mean(subset, "counterfactual_js"),
                            "tv": _mean(subset, "counterfactual_tv"),
                            "mean_abs_delta_p": _mean(subset, "counterfactual_mean_abs_delta_p"),
                            "delta_entropy": _mean(subset, "counterfactual_delta_entropy"),
                        })
    write_csv(model_dir / "acquisition_impact.csv", impact_rows)
    done_stage("acquisition_impact", stage_started)

    stage_started = start_stage("content_freshness")
    content_rows = []
    for condition, rows in _condition_rows(acquisitions).items():
        for label in ("content_changed", "freshness_only_changed", "content_and_freshness_changed", "no_effective_change"):
            subset = [row for row in rows if row.get("content_change_class") == label]
            content_rows.append({
                "T": condition[0], "H": condition[1], "content_change_class": label,
                "event_count": len(subset), "frequency": len(subset) / max(len(rows), 1),
                "top1_change_rate": _mean(subset, "counterfactual_top1_changed"),
                "kl": _mean(subset, "counterfactual_kl"), "js": _mean(subset, "counterfactual_js"),
                "tv": _mean(subset, "counterfactual_tv"),
                "pre_rate": float(np.mean([
                    int(row.get("response_phase") == "PRE") for row in subset
                ])) if subset else float("nan"),
                "post_rate": float(np.mean([
                    int(row.get("response_phase") == "POST") for row in subset
                ])) if subset else float("nan"),
                "partner_content_changed_rate": _mean(subset, "partner_content_changed"),
                "preview_content_changed_rate": _mean(subset, "preview_content_changed"),
                "self_content_changed_rate": _mean(subset, "self_content_changed"),
            })
    write_csv(model_dir / "content_freshness.csv", content_rows)
    done_stage("content_freshness", stage_started)

    stage_started = start_stage("response_time")
    write_csv(model_dir / "response_time.csv", response)
    done_stage("response_time", stage_started)

    stage_started = start_stage("revision_association")
    revision_rows = []
    revision_group_count = len(_condition_rows(acquisitions)) * 6
    revision_tracker = (
        None if progress is None
        else progress.bootstrap_tracker(
            "revision_association", bootstrap_iterations,
        )
    )
    revision_group_index = 0
    for condition, rows in _condition_rows(acquisitions).items():
        for exposure in ("self_footprint", "partner_footprint", "preview_footprint", "self_center", "partner_center", "preview_center"):
            per_seed = {}
            for seed_value, seed_rows in _group(rows, ("seed",)).items():
                value = revision_risk(seed_rows, exposure)["risk_difference"]
                if np.isfinite(value):
                    per_seed[_integer(seed_value[0])] = value
            contrast = paired_cluster_bootstrap(
                per_seed, {key: 0.0 for key in per_seed},
                iterations=bootstrap_iterations, seed=bootstrap_seed,
            )
            def revision_progress(completed):
                if revision_tracker is not None:
                    revision_tracker.update(
                        (revision_group_index * bootstrap_iterations + completed)
                        // max(revision_group_count, 1)
                    )

            rr_boot = _revision_ratio_bootstrap(
                rows, exposure, iterations=bootstrap_iterations,
                seed=bootstrap_seed, progress=revision_progress,
            )
            rr_low, rr_high = (
                (float("nan"), float("nan"))
                if not rr_boot.size
                else tuple(map(float, np.percentile(rr_boot, (2.5, 97.5))))
            )
            revision_rows.append({
                "T": condition[0], "H": condition[1], "exposure": exposure,
                **revision_risk(rows, exposure),
                "risk_difference_ci_low": contrast["ci_low"],
                "risk_difference_ci_high": contrast["ci_high"],
                "risk_ratio_ci_low": rr_low,
                "risk_ratio_ci_high": rr_high,
                "bootstrap_seed_count": contrast["n_pairs"],
                "scientific_language": "ASSOCIATION",
            })
            revision_group_index += 1
    write_csv(model_dir / "revision_association.csv", revision_rows)
    if revision_tracker is not None:
        revision_tracker.done()
    done_stage("revision_association", stage_started)

    stage_started = start_stage("stabilization_monitoring")
    stabilization_rows = []
    pairs = stabilization_first_post_pairs(acquisitions)
    for condition in sorted(_condition_rows(acquisitions)):
        selected = [(at, post) for at, post in pairs if (_integer(at["T"]), _integer(at["H"])) == condition]
        condition_acq = _condition_rows(acquisitions)[condition]
        condition_response = [row for row in response if (_integer(row["T"]), _integer(row["H"])) == condition]
        for aoi in ("self_footprint", "partner_footprint", "preview_footprint"):
            binary_pairs = [(_integer(at["seed"]), _integer(at[aoi]), _integer(post[aoi])) for at, post in selected]
            contrast = paired_binary_bootstrap(binary_pairs, iterations=bootstrap_iterations, seed=bootstrap_seed)
            stabilization_rows.append({
                "T": condition[0], "H": condition[1], "aoi": aoi,
                "paired_decisions": len(selected), "at_response_rate": float(np.mean([_integer(at[aoi]) for at, _ in selected])) if selected else float("nan"),
                "first_post_rate": float(np.mean([_integer(post[aoi]) for _, post in selected])) if selected else float("nan"),
                "at_minus_post": contrast["mean"], "ci_low": contrast["ci_low"], "ci_high": contrast["ci_high"],
                "post_any": _mean(condition_response, "post_any"),
                "post_event_count": sum(row.get("response_phase") == "POST" for row in condition_acq),
            })
    write_csv(model_dir / "stabilization_monitoring.csv", stabilization_rows)
    done_stage("stabilization_monitoring", stage_started)

    stage_started = start_stage("proximity")
    proximity_rows = []
    for condition, rows in _condition_rows(acquisitions).items():
        for distance_group in ("adjacent", "near", "far"):
            subset = [row for row in rows if row.get("partner_distance_bin") == distance_group]
            request_subset = [
                row for row in requests
                if (_integer(row["T"]), _integer(row["H"])) == condition
                and row.get("partner_distance_bin") == distance_group
            ]
            proximity_rows.append({
                "T": condition[0], "H": condition[1], "distance_bin": distance_group,
                "event_count": len(subset), "mean_manhattan_distance": _mean(subset, "partner_manhattan_distance"),
                "partner_center_rate": _mean(subset, "partner_center"),
                "partner_footprint_rate": _mean(subset, "partner_footprint"),
                "partner_center_mass_enrichment": _mean(request_subset, "partner_center_mass_enrichment"),
                "partner_footprint_mass_enrichment": _mean(request_subset, "partner_footprint_mass_enrichment"),
                "revision_rate": _mean(subset, "online_top1_changed_from_previous"),
                "post_partner_monitoring_rate": _mean([row for row in subset if row.get("response_phase") == "POST"], "partner_footprint"),
            })
    write_csv(model_dir / "proximity.csv", proximity_rows)
    done_stage("proximity", stage_started)

    stage_started = start_stage("coordination_states")
    coordination_rows = []
    for condition, rows in _condition_rows(acquisitions).items():
        for state in ("partner_adjacent", "interaction_relevant_proximity", "partner_holding_any", "self_holding_any"):
            for value in (0, 1):
                subset = [row for row in rows if _integer(row.get(state, 0)) == value]
                request_subset = [
                    row for row in requests
                    if (_integer(row["T"]), _integer(row["H"])) == condition
                    and _integer(row.get(state, 0)) == value
                ]
                coordination_rows.append({
                    "T": condition[0], "H": condition[1], "state": state,
                    "state_value": value, "event_count": len(subset),
                    "partner_footprint_rate": _mean(subset, "partner_footprint"),
                    "preview_footprint_rate": _mean(subset, "preview_footprint"),
                    "partner_enrichment": _mean(request_subset, "partner_footprint_mass_enrichment"),
                    "preview_enrichment": _mean(request_subset, "preview_footprint_mass_enrichment"),
                    "revision_rate": _mean(subset, "online_top1_changed_from_previous"),
                    "stabilization_rt_ms": _mean([response_row for response_row in response if (_integer(response_row["T"]), _integer(response_row["H"])) == condition], "top1_stable_rt_ms"),
                })
    write_csv(model_dir / "coordination_states.csv", coordination_rows)
    done_stage("coordination_states", stage_started)

    stage_started = start_stage("semantic_revisions")
    write_csv(model_dir / "semantic_revisions.csv", [{"status": "unsupported", "reason": "primitive direction does not uniquely identify a task goal under a documented canonical planner"}])
    if progress is not None:
        progress.skip_analysis_stage(
            "semantic_revisions", "canonical semantic goal unavailable"
        )
    else:
        done_stage("semantic_revisions", stage_started)

    stage_started = start_stage("memory_economy")
    economy_rows = []
    for condition, rows in _condition_rows(acquisitions).items():
        for phase in ("ALL", "PRE", "AT_RESPONSE", "POST"):
            phase_rows = rows if phase == "ALL" else [row for row in rows if row.get("response_phase") == phase]
            for aoi in ("ALL", "self", "partner", "preview", "other"):
                subset = phase_rows if aoi == "ALL" else [row for row in phase_rows if exclusive_aoi(row, aoi_precedence, suffix="footprint") == aoi]
                economy_rows.append({
                    "T": condition[0], "H": condition[1], "response_phase": phase,
                    "aoi": aoi, "event_count": len(subset),
                    "pre_target_mean_freshness": _mean(subset, "pre_target_mean_freshness"),
                    "pre_target_min_freshness": _mean(subset, "pre_target_min_freshness"),
                    "pre_target_valid_fraction": _mean(subset, "pre_target_valid_fraction"),
                    "newly_valid_tiles_per_acquisition": _mean(subset, "newly_valid_count"),
                    "content_novelty_per_acquisition": _mean(subset, "content_novelty_count"),
                    "marginal_valid_memory_gain": _mean(subset, "marginal_valid_memory_gain"),
                    "information_gain_per_saccade_degree": _mean(subset, "information_gain_per_saccade_degree"),
                    "content_novelty_per_saccade_degree": _mean(subset, "content_novelty_per_saccade_degree"),
                    "foveal_newly_valid_tiles_per_acquisition": _mean(
                        subset, "foveal_newly_valid_count"
                    ),
                    "peripheral_newly_valid_tiles_per_acquisition": _mean(
                        subset, "peripheral_newly_valid_count"
                    ),
                    "foveal_freshness_gain_per_acquisition": _mean(
                        subset, "foveal_freshness_gain"
                    ),
                    "peripheral_freshness_gain_per_acquisition": _mean(
                        subset, "peripheral_freshness_gain"
                    ),
                })
    write_csv(model_dir / "memory_economy.csv", economy_rows)
    done_stage("memory_economy", stage_started)

    stage_started = start_stage("cue_distance")
    cue_rows = _cue_distance_analysis(
        decisions, acquisitions, bootstrap_iterations, bootstrap_seed,
        progress=progress,
    )
    write_csv(model_dir / "cue_distance.csv", cue_rows)
    done_stage("cue_distance", stage_started)

    stage_started = start_stage("input_ablation")
    ablation_requests = read_csv(
        model_dir / "gaze_requests.csv.gz",
        columns=("request_row_id", "T", "H", *(
            f"p{action}" for action in range(GAZE_ACTION_COUNT)
        )),
    )
    ablation_rows = _input_ablation(
        model_dir, ablation_requests,
        gaze_actor=gaze_actor, gaze_device=gaze_device,
        progress=progress,
    )
    del ablation_requests
    write_csv(model_dir / "input_ablation.csv", ablation_rows)
    done_stage("input_ablation", stage_started)

    stage_started = start_stage("bootstrap_ci")
    bootstrap_rows = _core_bootstrap(
        episodes, decisions, requests, response, acquisitions,
        iterations=bootstrap_iterations, seed=bootstrap_seed,
    )
    write_csv(model_dir / "bootstrap_ci.csv", bootstrap_rows)
    done_stage("bootstrap_ci", stage_started)

    stage_started = start_stage("trend_summary")
    trend_rows = _trend_rows(bootstrap_rows, spatial_rows, economy_rows, proximity_rows, cue_rows)
    write_csv(model_dir / "trend_summary.csv", trend_rows)
    done_stage("trend_summary", stage_started)

    condition_summary = []
    for row in task_rows:
        match_spatial = next((item for item in spatial_rows if item["T"] == row["T"] and item["H"] == row["H"]), {})
        condition_summary.append({**row, **{key: value for key, value in match_spatial.items() if key not in {"T", "H"} and not key.startswith("target_")}})
    write_csv(model_dir / "condition_summary.csv", condition_summary)
    return {
        "task": task_rows, "spatial": spatial_rows, "aoi": aoi_rows,
        "enrichment": enrichment_rows, "response": response,
        "revision": revision_rows, "stabilization": stabilization_rows,
        "content": content_rows, "proximity": proximity_rows,
        "coordination": coordination_rows, "economy": economy_rows,
        "peripheral": peripheral_rows,
        "peripheral_impact": peripheral_impact_rows,
        "opacity": opacity_rows, "opacity_impact": opacity_impact_rows,
        "cue": cue_rows, "ablation": ablation_rows,
        "bootstrap": bootstrap_rows, "trends": trend_rows,
        "condition_summary": condition_summary,
    }


def _cue_distance_analysis(
    decisions, acquisitions, iterations, seed, *, progress=None
):
    rows = []
    by_decision = defaultdict(list)
    for acquisition in acquisitions:
        by_decision[_decision_key(acquisition)].append(acquisition)
    tracker = (
        None if progress is None
        else progress.bootstrap_tracker("cue_distance", iterations)
    )
    group_index = 0
    for definition in ("center", "footprint"):
        eligible = []
        for decision in decisions:
            if _integer(decision["H"]) != 4:
                continue
            near = json.loads(decision[f"near_{definition}_candidate_tiles"])
            far = json.loads(decision[f"far_{definition}_candidate_tiles"])
            values = sorted(
                by_decision.get(_decision_key(decision), ()),
                key=lambda row: _integer(row["proposal_index"]),
            )
            first = next((row for row in values if _integer(row.get(f"near_{definition}", 0)) or _integer(row.get(f"far_{definition}", 0))), None)
            first_label = None
            if first is not None:
                near_hit = _integer(first.get(f"near_{definition}", 0))
                far_hit = _integer(first.get(f"far_{definition}", 0))
                if near_hit != far_hit:
                    first_label = "near" if near_hit else "far"
            result = cue_distance_choice(
                fixation=(_integer(decision["start_fixation_x"]), _integer(decision["start_fixation_y"])),
                near_candidates=near, far_candidates=far, first_preview=first_label,
            )
            if result["far_closer"] is not None and result["first_preview_far"] is not None:
                eligible.append({**decision, **result})
        for time_ms in (None, 200, 400, 600):
            subset = eligible if time_ms is None else [row for row in eligible if _integer(row["T"]) == time_ms]
            far_closer = [_integer(row["first_preview_far"]) for row in subset if _integer(row["far_closer"])]
            near_closer = [_integer(row["first_preview_far"]) for row in subset if not _integer(row["far_closer"])]
            p_far = float(np.mean(far_closer)) if far_closer else float("nan")
            p_near = float(np.mean(near_closer)) if near_closer else float("nan")
            def odds_ratio(values):
                a = sum(_integer(row["far_closer"]) and _integer(row["first_preview_far"]) for row in values)
                b = sum(_integer(row["far_closer"]) and not _integer(row["first_preview_far"]) for row in values)
                c = sum(not _integer(row["far_closer"]) and _integer(row["first_preview_far"]) for row in values)
                d = sum(not _integer(row["far_closer"]) and not _integer(row["first_preview_far"]) for row in values)
                return float((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)))
            odds = odds_ratio(subset) if subset else float("nan")
            clusters = list(_group(subset, ("seed",)).values())
            boot_chunks = []
            if clusters:
                cluster_counts = np.asarray([
                    [
                        sum(_integer(row["far_closer"]) and _integer(row["first_preview_far"]) for row in cluster),
                        sum(_integer(row["far_closer"]) and not _integer(row["first_preview_far"]) for row in cluster),
                        sum(not _integer(row["far_closer"]) and _integer(row["first_preview_far"]) for row in cluster),
                        sum(not _integer(row["far_closer"]) and not _integer(row["first_preview_far"]) for row in cluster),
                    ]
                    for cluster in clusters
                ], dtype=np.int64)
                for completed, total in _chunked_cluster_counts(
                    cluster_counts, iterations=iterations, seed=seed,
                ):
                    a, b, c, d = total.T
                    boot_chunks.append(
                        (a + 0.5) * (d + 0.5)
                        / ((b + 0.5) * (c + 0.5))
                    )
                    if tracker is not None:
                        tracker.update(
                            (group_index * iterations + completed) // 8
                        )
            else:
                if tracker is not None:
                    tracker.update((group_index + 1) * iterations // 8)
            boot = np.concatenate(boot_chunks) if boot_chunks else np.empty(0)
            low, high = (
                (float("nan"), float("nan"))
                if not boot.size else tuple(map(float, np.percentile(boot, (2.5, 97.5))))
            )
            rows.append({
                "definition": definition, "T": "pooled" if time_ms is None else time_ms,
                "eligible_decisions": len(subset),
                "p_first_far_given_far_closer": p_far,
                "p_first_far_given_near_closer": p_near,
                "odds_ratio": odds,
                "ci_low": low, "ci_high": high,
                "bootstrap_unit": "episode_seed",
                "distance_metric": "ReferenceGeometry.visual_angle_degrees",
            })
            group_index += 1
    if tracker is not None:
        tracker.done()
    return rows


def _input_ablation(
    model_dir, requests, *, gaze_actor, gaze_device, progress=None
):
    if gaze_actor is None:
        return [{"status": "not_available", "reason": "analysis requires the frozen checkpoint GazeNet"}]
    sidecar = np.load(model_dir / "gaze_inputs.npz")
    memory = sidecar["memory"]
    fixation = sidecar["fixation"]
    context = sidecar["context"]
    from final_model.partial_ver.training import LearnedGazeController
    import torch
    variants = ("no_time_one_hot", "no_horizon_one_hot", "no_freshness_signal", "no_preview_memory", "no_partner_memory", "no_self_memory", "static_only")
    total_queries = len(requests) * (len(variants) + 1)
    ablation_started = (
        None if progress is None
        else progress.ablation_plan(
            requests=len(requests), variants=len(variants),
            total_queries=total_queries,
        )
    )
    completed_queries = 0
    report_milestones = {
        int(math.ceil(total_queries * fraction / 10))
        for fraction in range(1, 11)
    }

    def query_completed():
        nonlocal completed_queries
        completed_queries += 1
        if progress is None:
            return
        if completed_queries in report_milestones:
            progress.ablation_update(
                completed_queries, total_queries, ablation_started
            )
    totals: dict[tuple, list[dict]] = defaultdict(list)
    for index, row in enumerate(requests):
        baseline = np.asarray([_float(row[f"p{action}"]) for action in range(40)])
        baseline_spatial = LearnedGazeController.spatial_input(
            memory[index], fixation[index]
        )
        with torch.no_grad():
            replay = gaze_actor.distribution(
                torch.as_tensor(
                    baseline_spatial[None], dtype=torch.float32,
                    device=gaze_device,
                ),
                torch.as_tensor(
                    context[index][None], dtype=torch.float32,
                    device=gaze_device,
                ),
            ).probs.detach().cpu().numpy()[0]
        if not np.allclose(replay, baseline, atol=1e-7, rtol=1e-6):
            raise RuntimeError("saved gaze request does not replay on the frozen GazeNet")
        query_completed()
        for variant in variants:
            changed_memory = memory[index].copy()
            changed_context = context[index].copy()
            if variant == "no_time_one_hot": changed_context[1:4] = 0
            elif variant == "no_horizon_one_hot": changed_context[4:7] = 0
            elif variant == "no_freshness_signal": changed_memory[..., 20] = 0
            elif variant == "no_preview_memory": changed_memory[..., 21:25] = 0
            elif variant == "no_partner_memory": changed_memory[..., [1, 6, 7, 8, 9]] = 0
            elif variant == "no_self_memory": changed_memory[..., [0, 2, 3, 4, 5]] = 0
            elif variant == "static_only": changed_memory[..., [*range(10), *range(15, 25)]] = 0
            spatial = LearnedGazeController.spatial_input(changed_memory, fixation[index])
            with torch.no_grad():
                probability = gaze_actor.distribution(
                    torch.as_tensor(spatial[None], dtype=torch.float32, device=gaze_device),
                    torch.as_tensor(changed_context[None], dtype=torch.float32, device=gaze_device),
                ).probs.detach().cpu().numpy()[0]
            totals[(_integer(row["T"]), _integer(row["H"]), variant)].append(probability_metrics(baseline, probability))
            query_completed()
    return [{
        "T": key[0], "H": key[1], "variant": key[2], "request_count": len(values),
        "off_manifold_warning": 1,
        **{metric: float(np.mean([value[metric] for value in values])) for metric in ("top1_changed", "kl", "js", "tv", "mean_abs_delta_p", "delta_entropy")},
    } for key, values in sorted(totals.items())]


def _seed_metric(rows, field, *, predicate=lambda row: True):
    result = {}
    for key, values in _group([row for row in rows if predicate(row)], ("seed",)).items():
        mean = _mean(values, field)
        if np.isfinite(mean):
            result[_integer(key[0])] = mean
    return result


def _contrast_row(metric, left, right, *, left_label, right_label, iterations, seed):
    result = paired_cluster_bootstrap(
        left, right, iterations=iterations, seed=seed
    )
    return {
        "metric": metric, "contrast": f"{left_label} - {right_label}",
        **result, "status": directional_status(result["ci_low"], result["ci_high"]),
        "bootstrap_unit": "episode_seed",
    }


def _decision_any_metric(decisions, acquisitions, field, *, time_ms=None, horizon=None):
    flags = set()
    for row in acquisitions:
        if _integer(row.get(field, 0)):
            flags.add(_decision_key(row))
    rows = []
    for decision in decisions:
        if time_ms is not None and _integer(decision["T"]) != time_ms:
            continue
        if horizon is not None and _integer(decision["H"]) != horizon:
            continue
        rows.append({
            "seed": decision["seed"],
            "value": int(_decision_key(decision) in flags),
        })
    return _seed_metric(rows, "value")


def _rate_per_seed(numerator_rows, denominator_rows, *, predicate=lambda row: True):
    numerator = Counter(_integer(row["seed"]) for row in numerator_rows if predicate(row))
    denominator = Counter(_integer(row["seed"]) for row in denominator_rows if predicate(row))
    return {key: numerator[key] / value for key, value in denominator.items() if value}


def _revision_rd_by_seed(rows, exposure):
    result = {}
    for key, values in _group(rows, ("seed",)).items():
        rd = revision_risk(values, exposure)["risk_difference"]
        if np.isfinite(rd):
            result[_integer(key[0])] = rd
    return result


def _core_bootstrap(episodes, decisions, requests, response, acquisitions, *, iterations, seed):
    rows = []
    for time_ms in (200, 400, 600):
        values = [row for row in response if _integer(row["T"]) == time_ms]
        result = cluster_bootstrap_mean(values, "top1_stable_rt_ms", iterations=iterations, seed=seed)
        rows.append({"metric": "top1_stable_rt_ms", "contrast": f"T{time_ms} mean", **result, "status": "DESCRIPTIVE", "bootstrap_unit": "episode_seed"})

    def time_seed_metric(source, field, time_ms):
        return _seed_metric(source, field, predicate=lambda row: _integer(row["T"]) == time_ms)

    rows.append(_contrast_row(
        "deliveries", time_seed_metric(episodes, "deliveries", 600), time_seed_metric(episodes, "deliveries", 200),
        left_label="T600", right_label="T200", iterations=iterations, seed=seed,
    ))
    rows.append(_contrast_row(
        "acquisitions_per_decision",
        _rate_per_seed(acquisitions, decisions, predicate=lambda row: _integer(row["T"]) == 600),
        _rate_per_seed(acquisitions, decisions, predicate=lambda row: _integer(row["T"]) == 200),
        left_label="T600", right_label="T200", iterations=iterations, seed=seed,
    ))
    for aoi in ("partner_footprint", "preview_footprint"):
        rows.append(_contrast_row(
            f"decision_any_{aoi}",
            _decision_any_metric(decisions, acquisitions, aoi, time_ms=600),
            _decision_any_metric(decisions, acquisitions, aoi, time_ms=200),
            left_label="T600", right_label="T200", iterations=iterations, seed=seed,
        ))
    for field in ("partner_footprint", "preview_footprint"):
        rows.append(_contrast_row(
            f"relative_allocation_{field}", time_seed_metric(acquisitions, field, 600),
            time_seed_metric(acquisitions, field, 200), left_label="T600", right_label="T200",
            iterations=iterations, seed=seed,
        ))
    for field in ("preview_footprint_mass_enrichment", "partner_center_mass_enrichment", "partner_footprint_mass_enrichment"):
        values = _seed_metric(requests, field)
        rows.append(_contrast_row(
            field, values, {key: 1.0 for key in values}, left_label="enrichment", right_label="chance=1",
            iterations=iterations, seed=seed,
        ))
    rows.append(_contrast_row(
        "raw_preview_footprint", _seed_metric(acquisitions, "preview_footprint", predicate=lambda row: _integer(row["H"]) == 4),
        _seed_metric(acquisitions, "preview_footprint", predicate=lambda row: _integer(row["H"]) == 2),
        left_label="H4", right_label="H2", iterations=iterations, seed=seed,
    ))
    rows.append(_contrast_row(
        "preview_footprint_mass_enrichment", _seed_metric(requests, "preview_footprint_mass_enrichment", predicate=lambda row: _integer(row["H"]) == 4),
        _seed_metric(requests, "preview_footprint_mass_enrichment", predicate=lambda row: _integer(row["H"]) == 2),
        left_label="H4", right_label="H2", iterations=iterations, seed=seed,
    ))
    rows.append(_contrast_row(
        "near_far_mass_enrichment", _seed_metric(requests, "near_footprint_mass_enrichment", predicate=lambda row: _integer(row["H"]) == 4),
        _seed_metric(requests, "far_footprint_mass_enrichment", predicate=lambda row: _integer(row["H"]) == 4),
        left_label="near", right_label="far", iterations=iterations, seed=seed,
    ))
    for exposure in ("partner_footprint", "preview_footprint"):
        values = _revision_rd_by_seed(acquisitions, exposure)
        rows.append(_contrast_row(
            f"revision_RD_{exposure}", values, {key: 0.0 for key in values},
            left_label="exposed", right_label="unexposed", iterations=iterations, seed=seed,
        ))
    rows.append(_contrast_row(
        "post_stabilization_monitoring", time_seed_metric(response, "post_any", 600),
        time_seed_metric(response, "post_any", 200), left_label="T600", right_label="T200",
        iterations=iterations, seed=seed,
    ))
    for field in ("partner_footprint", "preview_footprint"):
        rows.append(_contrast_row(
            f"PRE_POST_{field}",
            _seed_metric(acquisitions, field, predicate=lambda row: row.get("response_phase") == "PRE"),
            _seed_metric(acquisitions, field, predicate=lambda row: row.get("response_phase") == "POST"),
            left_label="PRE", right_label="POST", iterations=iterations, seed=seed,
        ))
    rows.append(_contrast_row(
        "memory_content_novelty", time_seed_metric(acquisitions, "content_novelty_count", 600),
        time_seed_metric(acquisitions, "content_novelty_count", 200), left_label="T600", right_label="T200",
        iterations=iterations, seed=seed,
    ))
    rows.append(_contrast_row(
        "proximity_partner_check",
        _seed_metric(acquisitions, "partner_footprint", predicate=lambda row: row.get("partner_distance_bin") in {"adjacent", "near"}),
        _seed_metric(acquisitions, "partner_footprint", predicate=lambda row: row.get("partner_distance_bin") == "far"),
        left_label="near", right_label="far", iterations=iterations, seed=seed,
    ))
    return rows


def _trend_rows(bootstrap, spatial, economy, proximity, cue):
    lookup = {(row["metric"], row["contrast"]): row for row in bootstrap}
    mapping = {
        "Time ↑ task performance": ("deliveries", "T600 - T200"),
        "Time ↑ acquisitions/decision": ("acquisitions_per_decision", "T600 - T200"),
        "Time ↑ Partner-any decision": ("decision_any_partner_footprint", "T600 - T200"),
        "Time ↑ Preview-any decision": ("decision_any_preview_footprint", "T600 - T200"),
        "Time ↑ Partner relative allocation": ("relative_allocation_partner_footprint", "T600 - T200"),
        "Time ↑ Preview relative allocation": ("relative_allocation_preview_footprint", "T600 - T200"),
        "Preview preference above chance": ("preview_footprint_mass_enrichment", "enrichment - chance=1"),
        "Partner center preference above chance": ("partner_center_mass_enrichment", "enrichment - chance=1"),
        "Partner footprint preference above chance": ("partner_footprint_mass_enrichment", "enrichment - chance=1"),
        "H4 raw Preview viewing > H2": ("raw_preview_footprint", "H4 - H2"),
        "H4 Preview enrichment > H2": ("preview_footprint_mass_enrichment", "H4 - H2"),
        "near-future H1/H2 > H3/H4": ("near_far_mass_enrichment", "near - far"),
        "Partner acquisition associated with action revision": ("revision_RD_partner_footprint", "exposed - unexposed"),
        "Preview acquisition associated with action revision": ("revision_RD_preview_footprint", "exposed - unexposed"),
        "Time ↑ post-stabilization monitoring": ("post_stabilization_monitoring", "T600 - T200"),
        "memory economy metrics": ("memory_content_novelty", "T600 - T200"),
        "proximity-related Partner checking": ("proximity_partner_check", "near - far"),
    }
    labels = (
        "Time ↑ task performance", "Time ↑ acquisitions/decision",
        "Time ↑ Partner-any decision", "Time ↑ Preview-any decision",
        "Time ↑ Partner relative allocation", "Time ↑ Preview relative allocation",
        "Preview preference above chance", "Partner center preference above chance",
        "Partner footprint preference above chance", "H4 raw Preview viewing > H2",
        "H4 Preview enrichment > H2", "near-future H1/H2 > H3/H4",
        "Partner acquisition associated with action revision", "Preview acquisition associated with action revision",
        "Time ↑ post-stabilization monitoring", "spatial T/H redistribution magnitude",
        "fixation concentration / top1 share", "memory economy metrics",
        "proximity-related Partner checking", "cue-distance near/far selection effect",
    )
    result = []
    for label in labels:
        row = lookup.get(mapping.get(label))
        if row is not None:
            result.append({"trend": label, "status": row["status"], "estimate": row["mean"], "ci_low": row["ci_low"], "ci_high": row["ci_high"], "basis": row["contrast"]})
        elif label == "spatial T/H redistribution magnitude" and spatial:
            result.append({"trend": label, "status": "NO_CLEAR_DIFFERENCE", "estimate": None, "ci_low": None, "ci_high": None, "basis": "descriptive JS matrix; no directional null"})
        elif label == "fixation concentration / top1 share" and spatial:
            result.append({"trend": label, "status": "NO_CLEAR_DIFFERENCE", "estimate": max(row["top1_target_share"] for row in spatial), "ci_low": None, "ci_high": None, "basis": "diagnostic concentration only; no arbitrary threshold"})
        elif label == "cue-distance near/far selection effect":
            pooled = next((row for row in cue if row["definition"] == "footprint" and row["T"] == "pooled"), None)
            status = "NOT_AVAILABLE"
            if pooled and np.isfinite(pooled["ci_low"]) and pooled["ci_low"] > 1:
                status = "POSITIVE"
            elif pooled and np.isfinite(pooled["ci_high"]) and pooled["ci_high"] < 1:
                status = "NEGATIVE"
            elif pooled and np.isfinite(pooled["odds_ratio"]):
                status = "NO_CLEAR_DIFFERENCE"
            result.append({"trend": label, "status": status, "estimate": None if pooled is None else pooled["odds_ratio"], "ci_low": None if pooled is None else pooled["ci_low"], "ci_high": None if pooled is None else pooled["ci_high"], "basis": "cluster-bootstrap odds ratio"})
        else:
            result.append({"trend": label, "status": "NOT_AVAILABLE", "estimate": None, "ci_low": None, "ci_high": None, "basis": "required contrast not estimable"})
    return result
