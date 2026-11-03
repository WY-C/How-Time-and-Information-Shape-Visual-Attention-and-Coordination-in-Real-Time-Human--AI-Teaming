"""Read-only ActionNet snapshot, memory-ablation, and aggregation utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from final_model.checkpoint import ACTION_SPACE
from final_model.eval.counterfactual import actor_probabilities, distribution_metrics
from final_model.partial_ver.memory import DYNAMIC_CHANNELS, STATIC_CHANNELS
from final_model.server_parallel.gaze_v1 import action_index_to_tile


OBSERVATION_SHAPE = (8, 5, 25)
PARTNER_CHANNELS = (1, 6, 7, 8, 9)
SELF_CHANNELS = (0, 2, 3, 4, 5)
OTHER_DYNAMIC_CHANNELS = (15, 16, 17, 18, 19)
PREVIEW_CHANNELS = (21, 22, 23, 24)
FRESHNESS_CHANNEL = 20
MEMORY_VARIANTS = (
    "no_dynamic_memory",
    "no_preview_memory",
    "no_partner_memory",
    "static_only",
    "validity_signal_only",
)

CHANNEL_SEMANTICS = {
    0: "self_position_memory",
    1: "partner_position_memory",
    2: "self_orientation_north_memory",
    3: "self_orientation_south_memory",
    4: "self_orientation_east_memory",
    5: "self_orientation_west_memory",
    6: "partner_orientation_north_memory",
    7: "partner_orientation_south_memory",
    8: "partner_orientation_east_memory",
    9: "partner_orientation_west_memory",
    10: "pot_location_static",
    11: "counter_location_static",
    12: "onion_dispenser_location_static",
    13: "dish_dispenser_location_static",
    14: "serving_location_static",
    15: "pot_onion_count_memory",
    16: "pot_cook_time_memory",
    17: "soup_location_memory",
    18: "dish_location_memory",
    19: "onion_location_memory",
    20: "timestamp_freshness_normalized_5_to_0",
    21: "remembered_displayed_partner_action_preview_h1",
    22: "remembered_displayed_partner_action_preview_h2",
    23: "remembered_displayed_partner_action_preview_h3",
    24: "remembered_displayed_partner_action_preview_h4",
}


def instrument_memory_acquisitions(adapter, *, on_event=None) -> dict[str, Any]:
    """Capture exact before/after acquisition snapshots without changing writes."""
    capture: dict[str, Any] = {"event": None, "events": [], "count": 0}
    original = adapter.memory.acquire

    def wrapped(_memory, current_state20, current_preview4, tiles, *, timestamp_ms=None):
        memory_before = copy.deepcopy(adapter.memory)
        before = adapter.memory.snapshot(
            adapter.memory.current_model_time_ms
            if timestamp_ms is None
            else timestamp_ms
        ).copy()
        all_success_tiles = tuple(tuple(map(int, tile)) for tile in tiles)
        metadata = copy.deepcopy(adapter.last_acquisition_metadata or {})
        foveal_indices = metadata.get("foveal_tile_indices")
        tile_values = (
            all_success_tiles
            if foveal_indices is None
            else tuple(action_index_to_tile(index) for index in foveal_indices)
        )
        original(
            current_state20,
            current_preview4,
            all_success_tiles,
            timestamp_ms=timestamp_ms,
        )
        after = adapter.memory.snapshot().copy()
        memory_after = copy.deepcopy(adapter.memory)
        capture["count"] += 1
        event = {
            "event_index": capture["count"] - 1,
            "timestamp_ms": float(adapter.memory.current_model_time_ms),
            "before": before,
            "after": after,
            "_memory_before": memory_before,
            "_memory_after": memory_after,
            "tiles": tile_values,
            "all_success_tiles": all_success_tiles,
            **metadata,
        }
        capture["event"] = event
        capture["events"].append(event)
        if on_event is not None:
            on_event(event)

    adapter.memory.acquire = MethodType(wrapped, adapter.memory)
    return capture


def _observation(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != OBSERVATION_SHAPE:
        raise ValueError(f"actor observation must be {OBSERVATION_SHAPE}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("actor observation must be finite")
    return result


def preview_variant(observation: np.ndarray, *, horizon: int, keep: int) -> np.ndarray:
    """Keep the first ``keep`` remembered preview planes and copy everything else."""
    source = _observation(observation)
    value, active = int(horizon), int(keep)
    if value not in (0, 2, 4) or active not in range(5) or active > value:
        raise ValueError("invalid preview counterfactual")
    if np.any(source[..., 21 + value : 25] != 0.0):
        raise RuntimeError("observation contains preview information above H")
    result = np.ascontiguousarray(source.copy())
    result[..., 21 + active : 25] = 0.0
    return result


def memory_variant(
    observation: np.ndarray, variant: str, *, horizon: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a representation-aware ablation without changing the source array."""
    source = _observation(observation)
    result = np.ascontiguousarray(source.copy())
    label: dict[str, Any] = {"scientific_role": "MAIN_COUNTERFACTUAL"}
    if variant == "no_dynamic_memory":
        result[..., list(DYNAMIC_CHANNELS)] = 0.0
        label["definition"] = "channels 0-9 and 15-19 canonical unknown; preview retained"
    elif variant == "no_preview_memory":
        result = preview_variant(result, horizon=horizon, keep=0)
        label["definition"] = "remembered displayed preview channels 21-24 removed"
    elif variant == "no_partner_memory":
        result[..., list(PARTNER_CHANNELS)] = 0.0
        label["definition"] = "partner position/orientation memory channels 1 and 6-9 removed"
    elif variant == "static_only":
        result[..., list(DYNAMIC_CHANNELS)] = 0.0
        result[..., FRESHNESS_CHANNEL] = 0.0
        result[..., list(PREVIEW_CHANNELS)] = 0.0
        label["definition"] = "only exact static infrastructure channels 10-14 retained"
    elif variant == "validity_signal_only":
        result[..., FRESHNESS_CHANNEL] = 0.0
        label.update(
            scientific_role="OOD_DIAGNOSTIC_ONLY",
            definition="contents retained while timestamp freshness channel 20 is removed",
        )
    else:
        raise ValueError(f"unknown memory variant: {variant!r}")
    return result, label


def oracle_current_observation(backend, *, horizon: int) -> np.ndarray:
    """Explicit ORACLE_UPPER_BOUND: current dynamic state plus displayed H preview."""
    learner = np.asarray(backend.canonical_pair()[1], dtype=np.float32)
    if learner.shape != (8, 5, 20):
        raise RuntimeError("canonical current state contract changed")
    displayed = np.moveaxis(np.asarray(backend.preview_planes(), dtype=np.float32), 0, -1)
    if displayed.shape != (8, 5, 4):
        raise RuntimeError("displayed preview contract changed")
    displayed[..., int(horizon) :] = 0.0
    freshness = np.ones((8, 5, 1), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate((learner, freshness, displayed), axis=-1))


def action_entropy(probabilities: np.ndarray) -> float:
    value = np.asarray(probabilities, dtype=np.float64)
    safe = np.clip(value, np.finfo(np.float64).tiny, 1.0)
    return float(-np.sum(value * np.log(safe)))


def top1_margin(probabilities: np.ndarray) -> float:
    ordered = np.sort(np.asarray(probabilities, dtype=np.float64))
    return float(ordered[-1] - ordered[-2])


def memory_coverage(observation: np.ndarray, *, horizon: int) -> dict[str, float]:
    value = _observation(observation)
    valid = value[..., FRESHNESS_CHANNEL] > 0.0
    dynamic_present = np.any(value[..., list(DYNAMIC_CHANNELS)] != 0.0, axis=-1)
    preview_present = (
        np.any(value[..., 21 : 21 + int(horizon)] != 0.0, axis=-1)
        if horizon
        else np.zeros((8, 5), dtype=bool)
    )
    return {
        "valid_memory_fraction": float(valid.mean()),
        "acquired_dynamic_tile_count": int(np.count_nonzero(valid & dynamic_present)),
        "acquired_dynamic_tile_fraction": float(np.mean(valid & dynamic_present)),
        "preview_memory_tile_count": int(np.count_nonzero(valid & preview_present)),
        "preview_memory_coverage": float(np.mean(valid & preview_present)),
    }


@dataclass(frozen=True)
class ActionSnapshot:
    episode: int
    seed: int
    decision: int
    proposal_index: int
    timestamp_ms: float
    decision_time: int
    horizon: int
    fixation: tuple[int, int]
    observation: np.ndarray
    hidden: np.ndarray
    mask: np.ndarray
    available: np.ndarray
    factual_probabilities: np.ndarray
    factual_action: int


def read_only_probabilities(
    actor, observation: np.ndarray, hidden: np.ndarray, mask: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    """Use the canonical actor path while discarding and isolating returned state."""
    obs_before = np.asarray(observation).copy()
    hidden_before = np.asarray(hidden).copy()
    available_before = np.asarray(available).copy()
    result = actor_probabilities(
        actor, obs_before, hidden_before, np.asarray(mask).copy(), available_before
    )
    if not np.array_equal(obs_before, observation):
        raise RuntimeError("diagnostic actor forward mutated factual observation")
    if not np.array_equal(hidden_before, hidden):
        raise RuntimeError("diagnostic actor forward mutated factual hidden state")
    if not np.array_equal(available_before, available):
        raise RuntimeError("diagnostic actor forward mutated action mask")
    if not np.all(np.isfinite(result)) or not np.isclose(result.sum(), 1.0, atol=1e-6):
        raise RuntimeError("diagnostic ActionNet probabilities are invalid")
    return result


def read_only_probabilities_and_hidden(
    actor, observation: np.ndarray, hidden: np.ndarray, mask: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical actor query returning a copied candidate state for replay only."""
    obs = _observation(observation).copy()
    hidden_before = np.asarray(hidden).copy()
    with torch.no_grad():
        probabilities, candidate = actor.get_probs(
            obs[None], hidden_before.copy(), np.asarray(mask).copy(),
            np.asarray(available)[None].copy(),
        )
    result = probabilities.detach().cpu().numpy()[0]
    next_hidden = candidate.detach().cpu().numpy().copy()
    if not np.array_equal(hidden_before, hidden):
        raise RuntimeError("replay query mutated source recurrent state")
    if not np.all(np.isfinite(result)) or not np.isclose(result.sum(), 1.0, atol=1e-6):
        raise RuntimeError("replay ActionNet probabilities are invalid")
    return result, next_hidden


def read_only_logits_probabilities_and_hidden(
    actor, observation: np.ndarray, hidden: np.ndarray, mask: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluation-only actor query exposing the discrete head's raw logits."""
    action_out = getattr(getattr(actor, "act", None), "action_out", None)
    linear = getattr(action_out, "linear", None)
    if linear is None:
        raise RuntimeError("ActionNet does not expose the canonical discrete logit head")
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().clone())

    handle = linear.register_forward_hook(capture)
    try:
        probabilities, next_hidden = read_only_probabilities_and_hidden(
            actor, observation, hidden, mask, available
        )
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (1, len(ACTION_SPACE)):
        raise RuntimeError("canonical ActionNet query did not emit one six-action logit row")
    logits = captured[0].cpu().numpy()[0].astype(np.float64, copy=True)
    availability = np.asarray(available).reshape(-1)
    if availability.shape != (len(ACTION_SPACE),):
        raise ValueError("available action mask must contain six values")
    logits[availability == 0] = -1e10
    shifted = logits - np.max(logits)
    reconstructed = np.exp(shifted) / np.exp(shifted).sum()
    if not np.allclose(reconstructed, probabilities, atol=1e-7, rtol=1e-6):
        raise RuntimeError("raw ActionNet logits do not reproduce reported probabilities")
    return probabilities, logits, next_hidden


def audit_snapshot(actor, snapshot: ActionSnapshot) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Replay a snapshot and run same-hidden memory, preview, and zero-hidden tests."""
    factual = np.asarray(snapshot.factual_probabilities, dtype=np.float64)
    replay = read_only_probabilities(
        actor, snapshot.observation, snapshot.hidden, snapshot.mask, snapshot.available
    )
    repeat = read_only_probabilities(
        actor, snapshot.observation, snapshot.hidden, snapshot.mask, snapshot.available
    )
    checks = {
        "factual_replay_identity": bool(np.allclose(factual, replay, atol=1e-7, rtol=1e-6)),
        "deterministic_repeatability": bool(np.array_equal(replay, repeat)),
        "probabilities_finite": bool(np.all(np.isfinite(replay))),
        "probabilities_sum_to_one": bool(np.isclose(replay.sum(), 1.0, atol=1e-6)),
    }
    rows: list[dict[str, Any]] = []
    common = {
        "episode": snapshot.episode,
        "seed": snapshot.seed,
        "decision": snapshot.decision,
        "proposal_index": snapshot.proposal_index,
        "timestamp_ms": snapshot.timestamp_ms,
        "decision_time": snapshot.decision_time,
        "horizon": snapshot.horizon,
        "fixation_x": snapshot.fixation[0],
        "fixation_y": snapshot.fixation[1],
        "proposal_action_sampled": snapshot.factual_action,
    }
    for variant in MEMORY_VARIANTS:
        changed, label = memory_variant(snapshot.observation, variant, horizon=snapshot.horizon)
        probabilities = read_only_probabilities(
            actor, changed, snapshot.hidden, snapshot.mask, snapshot.available
        )
        rows.append({
            **common,
            "diagnostic": "memory_counterfactual",
            "variant": variant,
            "effective": int(not np.array_equal(changed, snapshot.observation)),
            **label,
            **distribution_metrics(factual, probabilities),
        })
    keep_values = {0: (0,), 2: (1, 0), 4: (2, 1, 0)}[snapshot.horizon]
    for keep in keep_values:
        changed = preview_variant(snapshot.observation, horizon=snapshot.horizon, keep=keep)
        probabilities = read_only_probabilities(
            actor, changed, snapshot.hidden, snapshot.mask, snapshot.available
        )
        rows.append({
            **common,
            "diagnostic": "preview_counterfactual",
            "variant": "none" if keep == 0 else f"h1_h{keep}" if keep > 1 else "h1_only",
            "preview_planes_kept": keep,
            "effective": int(not np.array_equal(changed, snapshot.observation)),
            "scientific_role": "MAIN_COUNTERFACTUAL",
            **distribution_metrics(factual, probabilities),
        })
        if snapshot.horizon == 0:
            checks["h0_preview_noop"] = bool(np.array_equal(changed, snapshot.observation))
    zero_hidden = np.zeros_like(snapshot.hidden)
    probabilities = read_only_probabilities(
        actor, snapshot.observation, zero_hidden, snapshot.mask, snapshot.available
    )
    rows.append({
        **common,
        "diagnostic": "recurrent_state",
        "variant": "canonical_zero_hidden",
        "effective": int(not np.array_equal(zero_hidden, snapshot.hidden)),
        "scientific_role": "DIAGNOSTIC_ABLATION",
        **distribution_metrics(factual, probabilities),
    })
    return rows, checks


def acquisition_labels(before: np.ndarray, after: np.ndarray, tiles: Iterable) -> dict[str, Any]:
    left, right = _observation(before), _observation(after)
    tile_list = [tuple(map(int, tile)) for tile in tiles]
    first = all(left[x, y, FRESHNESS_CHANNEL] == 0.0 for x, y in tile_list)
    difference = right - left
    changed_channels = set(np.flatnonzero(np.any(difference != 0.0, axis=(0, 1))).tolist())
    return {
        "memory_changed": int(not np.array_equal(left, right)),
        "first_acquisition_of_all_tiles": int(first),
        "reacquisition_of_any_tile": int(not first),
        "self_information_changed": int(bool(changed_channels & set(SELF_CHANNELS))),
        "partner_information_changed": int(bool(changed_channels & set(PARTNER_CHANNELS))),
        "preview_information_changed": int(bool(changed_channels & set(PREVIEW_CHANNELS))),
        "other_dynamic_information_changed": int(
            bool(changed_channels & set(OTHER_DYNAMIC_CHANNELS))
        ),
        "freshness_changed": int(FRESHNESS_CHANNEL in changed_channels),
        "changed_channels": ",".join(map(str, sorted(changed_channels))),
        "acquired_tile_count": len(tile_list),
    }


def aggregate_counterfactual_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return pooled and T/H summaries, each split into all/effective decisions."""
    values = [dict(row) for row in rows]
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in values:
        keys = (
            ("pooled", "pooled", row.get("diagnostic"), row.get("variant")),
            (row.get("decision_time"), row.get("horizon"), row.get("diagnostic"), row.get("variant")),
        )
        for key in keys:
            groups.setdefault(key, []).append(row)
    output = []
    metric_names = (
        "top1_action_changed", "kl_native_counterfactual", "js_divergence",
        "total_variation_distance", "mean_abs_probability_difference",
        "max_abs_probability_difference", "delta_entropy", "delta_top1_margin",
        "abs_delta_P_INTERACT", "abs_delta_P_STAY", "delta_movement_probability",
    )
    for (decision_time, horizon, diagnostic, variant), group in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        effective = [row for row in group if bool(row.get("effective", True))]
        for subset_name, subset in (("all_decisions", group), ("effective_decisions", effective)):
            record = {
                "decision_time": decision_time,
                "horizon": horizon,
                "diagnostic": diagnostic,
                "variant": variant,
                "subset": subset_name,
                "n_all": len(group),
                "n_effective": len(effective),
                "effective_fraction": len(effective) / max(len(group), 1),
                "n": len(subset),
            }
            for name in metric_names:
                finite = [float(row[name]) for row in subset if row.get(name) is not None]
                record[f"mean_{name}"] = float(np.mean(finite)) if finite else None
            output.append(record)
    return output


def source_comparability(source_metadata: Mapping[str, Any], adapted_metadata: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("architecture_id", "observation_contract_id", "action_space")
    mismatches = {
        field: {"source": source_metadata.get(field), "adapted": adapted_metadata.get(field)}
        for field in fields if source_metadata.get(field) != adapted_metadata.get(field)
    }
    return {
        "status": "COMPARABLE_SAME_STATE" if not mismatches else "NOT_COMPARABLE_SAME_STATE",
        "mismatches": mismatches,
        "strict_loading_or_adapter_bypass_used": False,
    }


def validate_action_contract(metadata: Mapping[str, Any]) -> None:
    if list(metadata.get("action_space", ())) != list(ACTION_SPACE):
        raise RuntimeError("checkpoint action-space contract mismatch")
