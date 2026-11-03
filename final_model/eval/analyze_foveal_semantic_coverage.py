"""Reanalyze saved learned-gaze rollouts under foveal/peripheral scopes.

This module is intentionally offline-only: it reads the rich acquisition and
request traces produced by ``gaze_behavior_rollout`` and never loads or changes
the training checkpoints.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from final_model.common.zsceval_compat import install_zsceval
from final_model.partial_ver.gaze import ReferenceGeometry, acquisition_tiles
from final_model.partial_ver.peripheral import (
    PERIPHERAL_PERCEPTION_CONTRACT_ID,
    PERIPHERAL_PROBABILITY_CONTRACT_ID,
    p1_peripheral_candidates,
    peripheral_detection_probability,
)
from final_model.server_parallel.gaze_v1 import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GAZE_ACTION_COUNT,
    GAZE_TILES,
    action_index_to_tile,
    tile_to_action_index,
)


DEFAULT_RESULT_ROOT = Path(
    "final_model/eval/results/gaze20m_train_sampling_20260903"
)
MODEL_DIRECTORIES = (
    "no_decay_pheri_20m_train_sampling",
    "original_decay_pheri_20m_train_sampling",
)
DECISION_KEY = ["T", "H", "episode", "seed", "decision"]
EVENT_KEY = DECISION_KEY + ["acquisition_index"]
TARGETS = ("Self", "Partner", "Preview", "H1", "H2", "H3", "H4")
AOI_PRECEDENCE = (
    "Self", "Partner", "Preview", "Pot", "Onion", "Dish", "Serve",
    "Counter", "Floor",
)
RELATIONS = ("CENTER", "CARDINAL", "DIAGONAL", "FAR")


def spatial_relation(
    fixation: tuple[int, int], target: tuple[int, int]
) -> str:
    """Return the disjoint spatial relation using production footprint helpers."""
    fixation = action_index_to_tile(tile_to_action_index(fixation))
    target = action_index_to_tile(tile_to_action_index(target))
    if target == fixation:
        return "CENTER"
    if target in acquisition_tiles(fixation, "plus"):
        return "CARDINAL"
    if target in p1_peripheral_candidates(fixation):
        return "DIAGONAL"
    return "FAR"


def scope_probability(
    fixation: tuple[int, int], target: tuple[int, int], *, scope: str,
    sigma_deg: float, geometry: ReferenceGeometry,
) -> float:
    """Expected acquisition probability for one tile under a fixed scope."""
    relation = spatial_relation(fixation, target)
    if relation in {"CENTER", "CARDINAL"}:
        return 1.0
    if scope == "plus_none":
        return 0.0
    if scope == "exact_3x3":
        return float(relation == "DIAGONAL")
    if scope == "plus_local" and relation != "DIAGONAL":
        return 0.0
    if scope not in {"plus_local", "plus_global"}:
        raise ValueError(f"unknown perceptual scope: {scope}")
    eccentricity = geometry.visual_angle_degrees(fixation, target)
    return float(peripheral_detection_probability(eccentricity, sigma_deg))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_indices(values: Iterable[str]) -> np.ndarray:
    rows = list(values)
    result = np.zeros((len(rows), GAZE_ACTION_COUNT), dtype=bool)
    for row_index, value in enumerate(rows):
        for target_index in json.loads(value):
            result[row_index, int(target_index)] = True
    return result


def _target_hit_indices(target_index: int) -> np.ndarray:
    target = action_index_to_tile(target_index)
    return np.asarray([
        index for index, fixation in enumerate(GAZE_TILES)
        if target in acquisition_tiles(fixation, "plus")
    ], dtype=np.int64)


PLUS_TARGET_HITS = tuple(
    _target_hit_indices(target_index) for target_index in range(GAZE_ACTION_COUNT)
)


def recover_preview_tile_indices(requests: pd.DataFrame) -> dict[str, np.ndarray]:
    """Recover singleton H1-H4 tile locations from recorded policy masses.

    Each displayed future-step mask contains exactly one tile. The trace stores
    its center policy mass (the probability of that tile). In the rare event of
    equal center probabilities, the independently stored PLUS policy mass
    disambiguates the tile.
    """
    probabilities = requests[[f"p{i}" for i in range(GAZE_ACTION_COUNT)]].to_numpy(
        dtype=np.float64
    )
    recovered: dict[str, np.ndarray] = {}
    for step in range(1, 5):
        prefix = f"h{step}_center"
        counts = requests[f"{prefix}_compatible_target_count"].to_numpy(dtype=int)
        if not np.isin(counts, (0, 1)).all():
            raise RuntimeError(f"H{step} center mask is not singleton/absent")
        center_mass = requests[f"{prefix}_policy_mass"].to_numpy(dtype=np.float64)
        footprint_mass = requests[
            f"h{step}_footprint_policy_mass"
        ].to_numpy(dtype=np.float64)
        indices = np.full(len(requests), -1, dtype=np.int16)
        for row_index in np.flatnonzero(counts == 1):
            differences = np.abs(probabilities[row_index] - center_mass[row_index])
            candidates = np.flatnonzero(differences <= 1e-12)
            if len(candidates) != 1:
                candidates = np.asarray([
                    candidate for candidate in candidates
                    if abs(
                        probabilities[row_index, PLUS_TARGET_HITS[candidate]].sum()
                        - footprint_mass[row_index]
                    ) <= 1e-12
                ], dtype=np.int64)
            if len(candidates) != 1:
                raise RuntimeError(
                    f"cannot uniquely recover H{step} tile at row {row_index}"
                )
            indices[row_index] = int(candidates[0])
        recovered[f"H{step}"] = indices
    return recovered


def _indices_from_xy(x: pd.Series, y: pd.Series) -> np.ndarray:
    return np.asarray([
        tile_to_action_index((int(tx), int(ty))) for tx, ty in zip(x, y)
    ], dtype=np.int16)


def _relations(
    fixation_indices: np.ndarray, target_indices: np.ndarray,
    available: np.ndarray,
) -> dict[str, np.ndarray]:
    values = {name: np.zeros(len(target_indices), dtype=bool) for name in RELATIONS}
    for row_index in np.flatnonzero(available):
        relation = spatial_relation(
            action_index_to_tile(int(fixation_indices[row_index])),
            action_index_to_tile(int(target_indices[row_index])),
        )
        values[relation][row_index] = True
    return values


def _or(values: Iterable[np.ndarray], length: int) -> np.ndarray:
    arrays = list(values)
    return np.logical_or.reduce(arrays) if arrays else np.zeros(length, dtype=bool)


def _rate_row(
    *, model: str, target: str, available: np.ndarray,
    relations: Mapping[str, np.ndarray], foveal: np.ndarray,
    peripheral: np.ndarray, total: np.ndarray,
    diagonal_peripheral: np.ndarray, far_peripheral: np.ndarray,
) -> dict[str, object]:
    n = int(available.sum())
    if n == 0:
        raise RuntimeError(f"empty denominator for {model}/{target}")

    def count_rate(values: np.ndarray, label: str) -> dict[str, object]:
        count = int(np.count_nonzero(values & available))
        return {f"{label}_count": count, f"{label}_pct": 100.0 * count / n}

    plus = relations["CENTER"] | relations["CARDINAL"]
    strict_peripheral_only = peripheral & ~foveal
    return {
        "model": model,
        "target": target,
        "n": n,
        "denominator": (
            "all acquisitions" if target in {"Self", "Partner"}
            else "acquisitions where target is displayed"
        ),
        **count_rate(relations["CENTER"], "center"),
        **count_rate(plus, "plus_covered"),
        **count_rate(relations["DIAGONAL"], "diagonal_position"),
        **count_rate(relations["FAR"], "far_position"),
        **count_rate(foveal, "actual_foveal_acquired"),
        **count_rate(peripheral, "actual_peripheral_acquired"),
        **count_rate(strict_peripheral_only, "peripheral_only_acquired"),
        **count_rate(diagonal_peripheral, "diagonal_peripheral_success"),
        **count_rate(far_peripheral, "far_peripheral_success"),
        **count_rate(total, "any_acquired"),
    }


def _summary_row(
    frame: pd.DataFrame, *, model: str, target: str, scope: str,
    scope_value: str | int, available_column: str,
) -> dict[str, object] | None:
    subset = frame[frame[available_column].astype(bool)]
    if subset.empty:
        return None
    fields = (
        "center", "plus", "diagonal", "far", "peripheral_acquired", "acquired"
    )
    row: dict[str, object] = {
        "model": model, "target": target, "scope": scope,
        "scope_value": scope_value, "n": len(subset),
        "denominator": "all decisions where target is displayed",
    }
    for field in fields:
        column = f"{target}_{field}"
        count = int(subset[column].astype(bool).sum())
        row[f"ever_{field}_count"] = count
        row[f"ever_{field}_pct"] = 100.0 * count / len(subset)
    return row


def _load_terrain() -> tuple[tuple[str, ...], ...]:
    install_zsceval()
    from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
        OvercookedGridworld,
    )

    mdp = OvercookedGridworld.from_layout_name("random3")
    terrain = tuple(tuple(row) for row in mdp.terrain_mtx)
    if len(terrain) != BOARD_HEIGHT or any(len(row) != BOARD_WIDTH for row in terrain):
        raise RuntimeError("production random3 terrain is not 8x5")
    return terrain


def _fixture_name(terrain: tuple[tuple[str, ...], ...], tile_index: int) -> str:
    x, y = action_index_to_tile(int(tile_index))
    return {
        "P": "Pot", "O": "Onion", "D": "Dish", "S": "Serve",
        "X": "Counter", " ": "Floor",
    }[terrain[y][x]]


def _git_metadata(project_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _analyze_model(
    model_dir: Path, terrain: tuple[tuple[str, ...], ...]
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    manifest = json.loads((model_dir / "rollout_manifest.json").read_text())
    model = str(manifest["model_name"])
    sigma = float(manifest["peripheral_sigma_deg"])
    if manifest["fixation_shape"] != "plus":
        raise RuntimeError(f"{model}: analysis requires PLUS fixation")
    if manifest["peripheral_mode"] != "global":
        raise RuntimeError(f"{model}: source rollout is not global peripheral")
    if manifest["peripheral_perception_contract_id"] != PERIPHERAL_PERCEPTION_CONTRACT_ID:
        raise RuntimeError(f"{model}: peripheral perception contract mismatch")
    if manifest["peripheral_probability_contract_id"] != PERIPHERAL_PROBABILITY_CONTRACT_ID:
        raise RuntimeError(f"{model}: peripheral probability contract mismatch")

    acquisition_path = model_dir / "acquisitions.csv.gz"
    request_path = model_dir / "gaze_requests.csv.gz"
    decision_path = model_dir / "decisions.csv"
    acquisitions = pd.read_csv(acquisition_path)
    requests = pd.read_csv(request_path)
    decisions = pd.read_csv(decision_path)
    request_key = DECISION_KEY + ["request_index"]
    if len(acquisitions) != len(requests) or not np.array_equal(
        acquisitions[EVENT_KEY].to_numpy(), requests[request_key].to_numpy()
    ):
        raise RuntimeError(f"{model}: acquisition/request event keys are not 1:1")

    preview_indices = recover_preview_tile_indices(requests)
    fixation_indices = _indices_from_xy(
        acquisitions["fixation_x"], acquisitions["fixation_y"]
    )
    target_indices = {
        "Self": _indices_from_xy(acquisitions["self_tile_x"], acquisitions["self_tile_y"]),
        "Partner": _indices_from_xy(
            acquisitions["partner_tile_x"], acquisitions["partner_tile_y"]
        ),
        **preview_indices,
    }
    available = {
        "Self": np.ones(len(acquisitions), dtype=bool),
        "Partner": np.ones(len(acquisitions), dtype=bool),
        "H1": acquisitions["H"].to_numpy() >= 2,
        "H2": acquisitions["H"].to_numpy() >= 2,
        "H3": acquisitions["H"].to_numpy() >= 4,
        "H4": acquisitions["H"].to_numpy() >= 4,
    }
    available["Preview"] = acquisitions["H"].to_numpy() > 0
    relation = {
        target: _relations(fixation_indices, indices, available[target])
        for target, indices in target_indices.items()
    }

    # The independently recorded acquisition AOIs validate preview recovery and
    # the coordinate convention against the state at the same acquisition.
    recovery_checks = 0
    for target in ("Self", "Partner", "H1", "H2", "H3", "H4"):
        prefix = target.lower()
        expected_center = relation[target]["CENTER"]
        expected_plus = expected_center | relation[target]["CARDINAL"]
        mask = available[target]
        if not np.array_equal(
            expected_center[mask], acquisitions[f"{prefix}_center"].to_numpy(bool)[mask]
        ):
            raise RuntimeError(f"{model}: {target} center recovery mismatch")
        if not np.array_equal(
            expected_plus[mask], acquisitions[f"{prefix}_footprint"].to_numpy(bool)[mask]
        ):
            raise RuntimeError(f"{model}: {target} PLUS recovery mismatch")
        recovery_checks += int(mask.sum()) * 2

    success_matrix = _json_indices(acquisitions["peripheral_success_indices"])
    actual: dict[str, dict[str, np.ndarray]] = {}
    for target in ("Self", "Partner", "H1", "H2", "H3", "H4"):
        row = np.arange(len(acquisitions))
        peripheral = success_matrix[row, target_indices[target]] & available[target]
        prefix = target.lower()
        recorded_peripheral = acquisitions[
            f"{prefix}_peripheral_acquired"
        ].to_numpy(bool)
        if not np.array_equal(peripheral, recorded_peripheral):
            raise RuntimeError(f"{model}: {target} peripheral provenance mismatch")
        foveal = acquisitions[f"{prefix}_foveal_acquired"].to_numpy(bool)
        total = acquisitions[f"{prefix}_total_acquired"].to_numpy(bool)
        if not np.array_equal(total, foveal | peripheral):
            raise RuntimeError(f"{model}: {target} total acquisition mismatch")
        actual[target] = {
            "foveal": foveal,
            "peripheral": peripheral,
            "total": total,
            "diagonal_peripheral": peripheral & relation[target]["DIAGONAL"],
            "far_peripheral": peripheral & relation[target]["FAR"],
        }

    preview_relations = {
        name: _or((relation[f"H{step}"][name] for step in range(1, 5)), len(acquisitions))
        for name in RELATIONS
    }
    actual["Preview"] = {
        field: _or((actual[f"H{step}"][field] for step in range(1, 5)), len(acquisitions))
        for field in (
            "foveal", "peripheral", "total", "diagonal_peripheral",
            "far_peripheral",
        )
    }
    relation["Preview"] = preview_relations
    for field, recorded in (
        ("foveal", "preview_foveal_acquired"),
        ("peripheral", "preview_peripheral_acquired"),
        ("total", "preview_total_acquired"),
    ):
        if not np.array_equal(
            actual["Preview"][field], acquisitions[recorded].to_numpy(bool)
        ):
            raise RuntimeError(f"{model}: ANY Preview {field} mismatch")

    acquisition_rows = []
    for target in TARGETS:
        acquisition_rows.append(_rate_row(
            model=model, target=target, available=available[target],
            relations=relation[target], foveal=actual[target]["foveal"],
            peripheral=actual[target]["peripheral"],
            total=actual[target]["total"],
            diagonal_peripheral=actual[target]["diagonal_peripheral"],
            far_peripheral=actual[target]["far_peripheral"],
        ))

    event = acquisitions[DECISION_KEY].copy()
    for target in TARGETS:
        event[f"{target}_available"] = available[target]
        event[f"{target}_center"] = relation[target]["CENTER"]
        event[f"{target}_plus"] = (
            relation[target]["CENTER"] | relation[target]["CARDINAL"]
        )
        event[f"{target}_diagonal"] = relation[target]["DIAGONAL"]
        event[f"{target}_far"] = relation[target]["FAR"]
        event[f"{target}_peripheral_acquired"] = actual[target]["peripheral"]
        event[f"{target}_acquired"] = actual[target]["total"]
    decision_any = event.groupby(DECISION_KEY, as_index=False).max()
    decision_frame = decisions[DECISION_KEY].merge(
        decision_any, on=DECISION_KEY, how="left"
    )
    metric_columns = [
        column for column in decision_frame
        if column not in DECISION_KEY
    ]
    decision_frame[metric_columns] = decision_frame[metric_columns].fillna(False)
    # Availability is a property of the condition, not of whether a visual
    # acquisition occurred. Forced/no-acquisition decisions remain in the
    # denominator with all ``ever_*`` outcomes false.
    decision_frame["Self_available"] = True
    decision_frame["Partner_available"] = True
    decision_frame["Preview_available"] = decision_frame["H"] > 0
    decision_frame["H1_available"] = decision_frame["H"] >= 2
    decision_frame["H2_available"] = decision_frame["H"] >= 2
    decision_frame["H3_available"] = decision_frame["H"] >= 4
    decision_frame["H4_available"] = decision_frame["H"] >= 4

    decision_rows: list[dict[str, object]] = []
    decision_time_rows: list[dict[str, object]] = []
    decision_horizon_rows: list[dict[str, object]] = []
    for target in TARGETS:
        row = _summary_row(
            decision_frame, model=model, target=target, scope="overall",
            scope_value="ALL", available_column=f"{target}_available",
        )
        if row is not None:
            decision_rows.append(row)
        for time_ms, subset in decision_frame.groupby("T"):
            row = _summary_row(
                subset, model=model, target=target, scope="T",
                scope_value=int(time_ms), available_column=f"{target}_available",
            )
            if row is not None:
                decision_time_rows.append(row)
        for horizon, subset in decision_frame.groupby("H"):
            row = _summary_row(
                subset, model=model, target=target, scope="H",
                scope_value=int(horizon), available_column=f"{target}_available",
            )
            if row is not None:
                decision_horizon_rows.append(row)

    semantic = np.empty(len(acquisitions), dtype=object)
    unassigned = np.ones(len(acquisitions), dtype=bool)
    for label, field in (
        ("Self", "self_center"), ("Partner", "partner_center"),
        ("Preview", "preview_center"),
    ):
        selected = unassigned & acquisitions[field].to_numpy(bool)
        semantic[selected] = label
        unassigned &= ~selected
    for row_index in np.flatnonzero(unassigned):
        semantic[row_index] = _fixture_name(terrain, int(fixation_indices[row_index]))
    center_frame = acquisitions[["T", "H"]].copy()
    center_frame["semantic_aoi"] = semantic
    center_rows = []
    for scope, value, subset in [
        ("overall", "ALL", center_frame),
        *(("T", int(key), group) for key, group in center_frame.groupby("T")),
        *(("H", int(key), group) for key, group in center_frame.groupby("H")),
    ]:
        counts = subset["semantic_aoi"].value_counts()
        for label in AOI_PRECEDENCE:
            count = int(counts.get(label, 0))
            center_rows.append({
                "model": model, "scope": scope, "scope_value": value,
                "semantic_aoi": label, "n": len(subset), "count": count,
                "pct": 100.0 * count / len(subset),
                "precedence": ">".join(AOI_PRECEDENCE),
            })

    exact_other = ~(
        acquisitions["self_center"].to_numpy(bool)
        | acquisitions["partner_center"].to_numpy(bool)
        | acquisitions["preview_center"].to_numpy(bool)
    )
    self_plus = acquisitions["self_footprint"].to_numpy(bool)
    partner_plus = acquisitions["partner_footprint"].to_numpy(bool)
    preview_plus = acquisitions["preview_footprint"].to_numpy(bool)
    other_rows = []
    center_categories = np.asarray([
        _fixture_name(terrain, int(index)) for index in fixation_indices
    ], dtype=object)
    for category in ("ALL", "Counter", "Floor", "Pot", "Onion", "Dish", "Serve"):
        subset_all = exact_other & (
            np.ones(len(acquisitions), dtype=bool)
            if category == "ALL" else center_categories == category
        )
        subset_preview = subset_all & available["Preview"]
        n_all = int(subset_all.sum())
        n_preview = int(subset_preview.sum())
        if n_all == 0:
            continue
        other_rows.append({
            "model": model, "center_semantic_category": category,
            "n_all": n_all, "n_preview_available": n_preview,
            "self_in_plus_pct": 100.0 * np.mean(self_plus[subset_all]),
            "partner_in_plus_pct": 100.0 * np.mean(partner_plus[subset_all]),
            "self_or_partner_in_plus_pct": 100.0 * np.mean(
                (self_plus | partner_plus)[subset_all]
            ),
            "preview_in_plus_pct": (
                100.0 * np.mean(preview_plus[subset_preview]) if n_preview else np.nan
            ),
            "self_or_partner_or_preview_in_plus_pct": (
                100.0 * np.mean(
                    (self_plus | partner_plus | preview_plus)[subset_preview]
                ) if n_preview else np.nan
            ),
            "preview_denominator": "H>0 exact-Other acquisitions",
            "other_metrics_denominator": "all exact-Other acquisitions",
        })

    geometry = ReferenceGeometry()
    angle = np.zeros((GAZE_ACTION_COUNT, GAZE_ACTION_COUNT), dtype=np.float64)
    for fixation_index, fixation in enumerate(GAZE_TILES):
        for target_index, target in enumerate(GAZE_TILES):
            angle[fixation_index, target_index] = geometry.visual_angle_degrees(
                fixation, target
            )
    gaussian = np.asarray(
        peripheral_detection_probability(angle, sigma), dtype=np.float64
    )
    scopes = ("plus_none", "plus_local", "exact_3x3", "plus_global")
    counterfactual_rows = []
    for target in TARGETS:
        mask = available[target]
        probabilities = {scope: np.zeros(len(acquisitions)) for scope in scopes}
        if target != "Preview":
            for row_index in np.flatnonzero(mask):
                fixation = action_index_to_tile(int(fixation_indices[row_index]))
                target_tile = action_index_to_tile(int(target_indices[target][row_index]))
                for scope in scopes:
                    probabilities[scope][row_index] = scope_probability(
                        fixation, target_tile, scope=scope, sigma_deg=sigma,
                        geometry=geometry,
                    )
        else:
            for row_index in np.flatnonzero(mask):
                unique_targets = {
                    int(target_indices[f"H{step}"][row_index])
                    for step in range(1, 5)
                    if available[f"H{step}"][row_index]
                }
                for scope in scopes:
                    per_tile = []
                    for target_index in unique_targets:
                        per_tile.append(scope_probability(
                            action_index_to_tile(int(fixation_indices[row_index])),
                            action_index_to_tile(target_index), scope=scope,
                            sigma_deg=sigma, geometry=geometry,
                        ))
                    probabilities[scope][row_index] = 1.0 - float(np.prod(
                        1.0 - np.asarray(per_tile, dtype=np.float64)
                    ))
        row = {
            "model": model, "target": target, "n": int(mask.sum()),
            "denominator": (
                "all acquisitions" if target in {"Self", "Partner"}
                else "acquisitions where target is displayed"
            ),
            "interpretation": "fixed Global trajectory; counterfactual scope only",
        }
        for scope in scopes:
            row[f"{scope}_expected_pct"] = 100.0 * probabilities[scope][mask].mean()
        counterfactual_rows.append(row)

    successful_preview_angles = []
    strict_peripheral_only_preview_angles = []
    exact_preview_success_checks = 0
    for row_index in np.flatnonzero(available["Preview"]):
        unique_preview = {
            int(target_indices[f"H{step}"][row_index])
            for step in range(1, 5) if available[f"H{step}"][row_index]
        }
        for target_index in unique_preview:
            target_relation = spatial_relation(
                action_index_to_tile(int(fixation_indices[row_index])),
                action_index_to_tile(target_index),
            )
            if target_relation in {"CENTER", "CARDINAL"}:
                continue
            if success_matrix[row_index, target_index]:
                value = angle[int(fixation_indices[row_index]), target_index]
                successful_preview_angles.append(value)
                if not actual["Preview"]["foveal"][row_index]:
                    strict_peripheral_only_preview_angles.append(value)
                exact_preview_success_checks += 1
    preview_angles = np.asarray(successful_preview_angles, dtype=np.float64)
    strict_preview_angles = np.asarray(
        strict_peripheral_only_preview_angles, dtype=np.float64
    )
    if preview_angles.size == 0:
        raise RuntimeError(f"{model}: no peripheral Preview successes")
    if strict_preview_angles.size == 0:
        raise RuntimeError(f"{model}: no strict peripheral-only Preview successes")
    sanity = {
        "model": model,
        "event_count": len(acquisitions),
        "decision_count": len(decisions),
        "preview_recovery_center_plus_assertions": recovery_checks,
        "peripheral_preview_exact_match_pct": 100.0,
        "peripheral_preview_unique_tile_success_count": exact_preview_success_checks,
        "all_peripheral_hit_preview_success_eccentricity_median_deg": float(
            np.median(preview_angles)
        ),
        "all_peripheral_hit_preview_success_gt4_pct": float(
            100.0 * np.mean(preview_angles > 4.0)
        ),
        "all_peripheral_hit_preview_success_gt6_pct": float(
            100.0 * np.mean(preview_angles > 6.0)
        ),
        "strict_peripheral_only_preview_success_tile_count": int(
            strict_preview_angles.size
        ),
        "strict_peripheral_only_preview_success_eccentricity_median_deg": float(
            np.median(strict_preview_angles)
        ),
        "strict_peripheral_only_preview_success_gt4_pct": float(
            100.0 * np.mean(strict_preview_angles > 4.0)
        ),
        "strict_peripheral_only_preview_success_gt6_pct": float(
            100.0 * np.mean(strict_preview_angles > 6.0)
        ),
        "eligible_denominator_rate_check": {
            target: {
                "foveal_pct": float(100.0 * actual[target]["foveal"][available[target]].mean()),
                "peripheral_pct": float(100.0 * actual[target]["peripheral"][available[target]].mean()),
                "total_pct": float(100.0 * actual[target]["total"][available[target]].mean()),
            } for target in ("Self", "Partner", "Preview")
        },
        "legacy_all_acquisition_denominator_rate_check": {
            target: {
                "foveal_pct": float(100.0 * actual[target]["foveal"].mean()),
                "peripheral_pct": float(100.0 * actual[target]["peripheral"].mean()),
                "total_pct": float(100.0 * actual[target]["total"].mean()),
            } for target in ("Self", "Partner", "Preview")
        },
    }
    outputs = {
        "acquisition": acquisition_rows,
        "decision": decision_rows,
        "decision_time": decision_time_rows,
        "decision_horizon": decision_horizon_rows,
        "center": center_rows,
        "other": other_rows,
        "counterfactual": counterfactual_rows,
    }
    source = {
        "model": model,
        "model_directory": str(model_dir.resolve()),
        "checkpoint_path": manifest["checkpoint_absolute_path"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_files": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (acquisition_path, request_path, decision_path, model_dir / "rollout_manifest.json")
        },
        "rollout_contract": {
            key: manifest.get(key) for key in (
                "gaze_mode", "action_mode", "fixation_shape", "peripheral_mode",
                "peripheral_sigma_deg", "peripheral_contract_id",
                "peripheral_perception_contract_id",
                "peripheral_probability_contract_id", "peripheral_rng_contract_id",
                "memory_decay_mode", "memory_lifetime_ms", "preview_opacity_mode",
            )
        },
        "sanity": sanity,
    }
    return outputs, source


def run_analysis(result_root: Path, output_dir: Path) -> None:
    result_root = result_root.resolve()
    model_root = result_root / "models"
    model_dirs = [model_root / name for name in MODEL_DIRECTORIES]
    missing = [str(path) for path in model_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing source model directories: {missing}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    # Explicit production-contract regression guards requested by the analysis.
    if tile_to_action_index((3, 1)) != 16:
        raise RuntimeError("gaze serialization is not x-major x*5+y")
    if [tile_to_action_index(tile) for tile in acquisition_tiles((3, 1), "plus")] != [
        16, 15, 17, 11, 21
    ]:
        raise RuntimeError("production PLUS footprint changed")

    terrain = _load_terrain()
    combined = {
        key: [] for key in (
            "acquisition", "decision", "decision_time", "decision_horizon",
            "center", "other", "counterfactual",
        )
    }
    sources = []
    for model_dir in model_dirs:
        outputs, source = _analyze_model(model_dir, terrain)
        for key, rows in outputs.items():
            combined[key].extend(rows)
        sources.append(source)

    output_dir.mkdir(parents=True)
    filenames = {
        "acquisition": "acquisition_level_summary.csv",
        "decision": "decision_level_summary.csv",
        "decision_time": "decision_level_by_time.csv",
        "decision_horizon": "decision_level_by_horizon.csv",
        "center": "center_semantic_aoi_summary.csv",
        "other": "other_center_plus_coverage.csv",
        "counterfactual": "counterfactual_perceptual_coverage.csv",
    }
    for key, filename in filenames.items():
        pd.DataFrame(combined[key]).to_csv(output_dir / filename, index=False)

    project_root = Path(__file__).resolve().parents[2]
    metadata = {
        "analysis_id": "global_gaze_foveal_semantic_coverage_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": str(result_root),
        "output_directory": str(output_dir.resolve()),
        "sources": sources,
        "coordinate_convention": {
            "name": "x-major",
            "formula": "index = x * 5 + y",
            "verified_example": {
                "fixation": [3, 1], "center_index": 16,
                "plus_raw_indices": [16, 15, 17, 11, 21],
                "plus_set_sorted": [11, 15, 16, 17, 21],
            },
            "production_helpers": [
                "tile_to_action_index", "action_index_to_tile", "acquisition_tiles",
            ],
        },
        "spatial_definitions": {
            "CENTER": "target tile equals fixation center",
            "PLUS_COVERED": "CENTER or cardinal neighbor from acquisition_tiles(..., 'plus')",
            "DIAGONAL": "production p1_peripheral_candidates(fixation)",
            "FAR": "outside the 3x3 neighborhood",
        },
        "aoi_precedence": list(AOI_PRECEDENCE),
        "denominators": {
            "acquisition_self_partner": "all successful visual acquisitions",
            "acquisition_preview": "H>0 acquisitions; H0 excluded",
            "acquisition_h1_h2": "H2/H4 acquisitions",
            "acquisition_h3_h4": "H4 acquisitions",
            "decision": "all game decisions where target is displayed; no-acquisition decisions count as false",
            "other_center_preview": "H>0 exact-Other acquisitions",
        },
        "counterfactual": {
            "trajectory": "fixed saved Global gaze trajectory; no policy/environment rerun",
            "plus_none": "PLUS=1, outside=0",
            "exact_3x3": "PLUS and diagonal=1, outside=0",
            "plus_local": "PLUS=1; production local diagonal Gaussian; outside=0",
            "plus_global": "PLUS=1; every outside tile uses production Gaussian",
            "preview_any": "probability of at least one unique displayed Preview tile; independent Bernoulli detections",
        },
        "assumptions_and_limitations": [
            "H1-H4 singleton tiles are exactly reconstructed from same-event gaze-request center/PLUS policy masses and validated against acquisition center/PLUS flags.",
            "Actual acquisition provenance uses current-event foveal/peripheral fields and peripheral_success_indices, never persistent memory contents.",
            "Onion and Dish semantic-center labels mean fixed dispensers. Loose moving onion/dish object positions are not stored in this rollout trace and cannot be recovered without a new state trace.",
            "ANY Preview spatial columns are non-exclusive because different future steps may occupy different spatial relation classes in one acquisition.",
            "ANY Preview actual_peripheral_acquired may coexist with foveal acquisition of another future step; peripheral_only_acquired excludes such rows.",
        ],
        "git": _git_metadata(project_root),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    acquisition = pd.DataFrame(combined["acquisition"])
    core = acquisition[acquisition["target"].isin(["Self", "Partner", "Preview"])]
    print("\nACQUISITION COVERAGE (%)")
    print(core[[
        "model", "target", "center_pct", "plus_covered_pct",
        "actual_peripheral_acquired_pct", "peripheral_only_acquired_pct",
        "any_acquired_pct",
    ]].round(2).to_string(index=False))
    print("\nOTHER-CENTER PLUS COVERAGE (%)")
    other = pd.DataFrame(combined["other"])
    print(other[other["center_semantic_category"] == "ALL"][[
        "model", "n_all", "self_in_plus_pct", "partner_in_plus_pct",
        "self_or_partner_in_plus_pct", "preview_in_plus_pct",
        "self_or_partner_or_preview_in_plus_pct",
    ]].round(2).to_string(index=False))
    print("\nCOUNTERFACTUAL EXPECTED COVERAGE (%)")
    counterfactual = pd.DataFrame(combined["counterfactual"])
    print(counterfactual[counterfactual["target"].isin(["Self", "Partner", "Preview"])][[
        "model", "target", "plus_none_expected_pct", "plus_local_expected_pct",
        "exact_3x3_expected_pct", "plus_global_expected_pct",
    ]].round(2).to_string(index=False))
    print(f"\nSaved: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.result_root / "foveal_semantic_analysis"
    run_analysis(args.result_root, output_dir)


if __name__ == "__main__":
    main()
