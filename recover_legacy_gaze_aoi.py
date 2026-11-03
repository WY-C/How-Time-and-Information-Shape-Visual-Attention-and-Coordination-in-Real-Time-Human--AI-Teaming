#!/usr/bin/env python3
"""Recover legacy gaze AOIs without modifying the source experiment logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from experiment_logging import build_ivt_fixation_rows
from ivt import IVTConfig


COOKING_TERRAIN_TYPES = {"O", "T", "D", "P"}
LAYOUT_ROOT = Path("lib/overcooked_ai/overcooked_ai_py/data/layouts")


def _git_calibrations() -> dict[str, dict[float, dict[str, Any]]]:
    output = subprocess.check_output(
        ["git", "rev-list", "--objects", "--all"], text=True
    )
    calibrations: dict[str, dict[float, dict[str, Any]]] = {}
    seen_blobs = set()
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or "gaze_calibration/" not in parts[1] or not parts[1].endswith(".json"):
            continue
        blob_id = parts[0]
        if blob_id in seen_blobs:
            continue
        seen_blobs.add(blob_id)
        try:
            payload = json.loads(
                subprocess.check_output(["git", "cat-file", "blob", blob_id])
            )
            layout = str(payload["layout_name"])
            rmse = float(payload["fit_rmse"])
            samples = payload.get("samples", [])
            if len(samples) < 3 or not all("surface_point" in sample for sample in samples):
                continue
            payload["git_blob"] = blob_id
            calibrations.setdefault(layout, {}).setdefault(rmse, payload)
        except (KeyError, TypeError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError):
            continue
    return calibrations


def _select_calibration(
    candidates: dict[float, dict[str, Any]], trial_rmse: float
) -> tuple[dict[str, Any], str]:
    selected_rmse = min(candidates, key=lambda rmse: abs(rmse - trial_rmse))
    mode = "exact" if abs(selected_rmse - trial_rmse) <= 1e-12 else "approximate_nearest_rmse"
    return candidates[selected_rmse], mode


def _surface_transform(payload: dict[str, Any]) -> np.ndarray:
    targets = np.asarray(
        [sample["target_display_point"] for sample in payload["samples"]],
        dtype=np.float64,
    )
    surfaces = np.asarray(
        [sample["surface_point"] for sample in payload["samples"]],
        dtype=np.float64,
    )
    design = np.column_stack([targets, np.ones(len(targets), dtype=np.float64)])
    coefficients, _, _, _ = np.linalg.lstsq(design, surfaces, rcond=None)
    return coefficients


def _infer_legacy_window_origin(
    trial_raw: pd.DataFrame,
    calibration_matrix: np.ndarray,
    *,
    game_window_size: tuple[float, float],
) -> tuple[float, float]:
    required = {
        "gaze_x",
        "gaze_y",
        "gaze_surface_x",
        "gaze_surface_y",
        "surface_width_px",
        "surface_height_px",
        "map_display_width_px",
        "map_display_height_px",
        "display_width_px",
        "display_height_px",
    }
    if not required.issubset(trial_raw.columns):
        raise ValueError("Legacy window origin cannot be recovered from this raw gaze schema.")

    usable = trial_raw.dropna(subset=list(required))
    if usable.empty:
        raise ValueError("Legacy window origin cannot be recovered without logged surface samples.")
    corrected = np.column_stack(
        [usable["gaze_x"], usable["gaze_y"], np.ones(len(usable), dtype=np.float64)]
    ) @ calibration_matrix.T
    display_width = usable["display_width_px"].to_numpy(dtype=np.float64)
    display_height = usable["display_height_px"].to_numpy(dtype=np.float64)
    map_width = usable["map_display_width_px"].to_numpy(dtype=np.float64)
    map_height = usable["map_display_height_px"].to_numpy(dtype=np.float64)
    surface_width = usable["surface_width_px"].to_numpy(dtype=np.float64)
    surface_height = usable["surface_height_px"].to_numpy(dtype=np.float64)
    map_left = (float(game_window_size[0]) - map_width) / 2.0
    map_top = (float(game_window_size[1]) - map_height) / 2.0
    origin_x = (
        corrected[:, 0] * display_width
        - map_left
        - usable["gaze_surface_x"].to_numpy(dtype=np.float64) * map_width / surface_width
    )
    origin_y = (
        corrected[:, 1] * display_height
        - map_top
        - usable["gaze_surface_y"].to_numpy(dtype=np.float64) * map_height / surface_height
    )
    return float(np.median(origin_x)), float(np.median(origin_y))


def _reproject_legacy_surface_point(
    corrected_display_point: np.ndarray,
    *,
    display_size: tuple[float, float],
    game_window_origin: tuple[float, float],
    game_window_size: tuple[float, float],
    map_display_size: tuple[float, float],
    surface_size: tuple[float, float],
) -> np.ndarray:
    display_width, display_height = display_size
    window_width, window_height = game_window_size
    map_width, map_height = map_display_size
    surface_width, surface_height = surface_size
    map_left = (window_width - map_width) / 2.0
    map_top = (window_height - map_height) / 2.0
    global_x = float(corrected_display_point[0]) * display_width
    global_y = float(corrected_display_point[1]) * display_height
    return np.asarray(
        [
            (global_x - game_window_origin[0] - map_left) * surface_width / map_width,
            (global_y - game_window_origin[1] - map_top) * surface_height / map_height,
        ],
        dtype=np.float64,
    )


def _load_terrain(layout: str) -> list[list[str]]:
    text = (LAYOUT_ROOT / f"{layout}.layout").read_text(encoding="utf-8")
    match = re.search(r'"grid"\s*:\s*"""(.*?)"""', text, re.DOTALL)
    if match is None:
        raise ValueError(f"Grid not found in layout {layout!r}.")
    rows = [line.strip() for line in match.group(1).strip().splitlines()]
    width = len(rows[0])
    if not rows or any(len(row) != width for row in rows):
        raise ValueError(f"Layout {layout!r} is not rectangular.")
    return [[" " if tile in {"1", "2"} else tile for tile in row] for row in rows]


def _json_value(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _classify_context(
    tile: tuple[int, int] | None,
    terrain: list[list[str]],
    step_row: pd.Series | None,
) -> dict[str, Any]:
    context = {
        "base_aoi": "Outside",
        "base_position": None,
        "overlay_flag": 0,
        "overlay_type": "none",
        "overlay_horizon": None,
        "path_rank": None,
        "overlay_underlying_aoi": None,
        "overlay_position": None,
        "overlay_matches": [],
        "counter_object_present": 0,
        "counter_object_type": None,
    }
    if tile is None or step_row is None:
        return context

    x, y = tile
    if y < 0 or y >= len(terrain) or x < 0 or x >= len(terrain[y]):
        return context

    human_position = tuple(_json_value(step_row.get("human_position"), []))
    agent_position = tuple(_json_value(step_row.get("agent_position"), []))
    if tile == human_position:
        base_aoi = "Self"
    elif tile == agent_position:
        base_aoi = "Partner"
    else:
        terrain_type = terrain[y][x]
        if terrain_type in COOKING_TERRAIN_TYPES:
            base_aoi = "Cooking_Area"
        elif terrain_type == "S":
            base_aoi = "Serving_Area"
        elif terrain_type == "X":
            base_aoi = "Counter"
        elif terrain_type == " ":
            base_aoi = "Corridor"
        else:
            base_aoi = "Outside"
    context["base_aoi"] = base_aoi
    context["base_position"] = [x, y]

    trajectory = _json_value(step_row.get("future_trajectory"), {})
    matching_entries = [
        entry for entry in trajectory.get("entries", []) if tuple(entry.get("tile", ())) == tile
    ]
    if matching_entries:
        selected = matching_entries[0]
        context.update(
            {
                "overlay_flag": 1,
                "overlay_type": selected.get("overlay_type", "path"),
                "overlay_horizon": selected.get("overlay_horizon"),
                "path_rank": selected.get("path_rank"),
                "overlay_underlying_aoi": base_aoi,
                "overlay_position": selected.get("overlay_position", [x, y]),
                "overlay_matches": [
                    {
                        "overlay_type": entry.get("overlay_type", "path"),
                        "overlay_horizon": entry.get("overlay_horizon"),
                        "path_rank": entry.get("path_rank"),
                        "overlay_underlying_aoi": base_aoi,
                        "overlay_position": entry.get("overlay_position", [x, y]),
                    }
                    for entry in matching_entries
                ],
            }
        )

    if base_aoi == "Counter":
        state = _json_value(step_row.get("env_state"), {})
        for obj in state.get("objects", []):
            if tuple(obj.get("position", ())) == tile:
                context["counter_object_present"] = 1
                context["counter_object_type"] = obj.get("name")
                break
    return context


def recover(input_dir: Path, output_dir: Path) -> None:
    raw = pd.read_csv(input_dir / "raw_gaze.csv")
    steps = pd.read_csv(input_dir / "step.csv")
    trials = pd.read_csv(input_dir / "trial.csv")
    metadata_path = input_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata_trials = metadata.get("trials", {})
    has_logged_surface = {
        "gaze_surface_x",
        "gaze_surface_y",
        "surface_width_px",
        "surface_height_px",
    }.issubset(raw.columns)
    calibration_window_size = (
        int(pd.to_numeric(trials["screen_width_px"]).max()),
        int(pd.to_numeric(trials["screen_height_px"]).max()),
    )
    calibrations = {} if has_logged_surface else _git_calibrations()
    output_dir.mkdir(parents=True, exist_ok=True)

    step_lookup = {
        (str(row.trial_id), int(row.step)): row
        for _, row in steps.iterrows()
    }
    trial_lookup = trials.set_index("trial_id")
    terrain_cache: dict[str, list[list[str]]] = {}
    recovered_parts = []
    fixation_rows = []
    manifest_rows = []

    for trial_id, trial_raw in raw.groupby("trial_id", sort=False):
        trial = trial_lookup.loc[trial_id]
        layout = str(trial["layout"])
        trial_rmse = float(trial["calibration_rmse"])
        terrain = terrain_cache.setdefault(layout, _load_terrain(layout))
        if has_logged_surface:
            calibration = metadata_trials[str(trial_id)]["calibration"]
            calibration_matrix = np.asarray(calibration["transform_matrix"], dtype=np.float64)
            selected_rmse = float(calibration["fit_rmse"])
            calibration_blob = ""
            mode = "exact_corrected_window_reprojection"
            surface_matrix = None
            surface_width = int(trial_raw["surface_width_px"].dropna().iloc[0])
            surface_height = int(trial_raw["surface_height_px"].dropna().iloc[0])
            game_window_size = (
                float(trial["screen_width_px"]),
                float(trial["screen_height_px"]),
            )
            game_window_origin = _infer_legacy_window_origin(
                trial_raw,
                calibration_matrix,
                game_window_size=game_window_size,
            )
        else:
            payload, mode = _select_calibration(calibrations[layout], trial_rmse)
            selected_rmse = float(payload["fit_rmse"])
            calibration_matrix = np.asarray(payload["transform_matrix"], dtype=np.float64)
            calibration_blob = payload["git_blob"]
            surface_matrix = _surface_transform(payload)
            surface_width = len(terrain[0]) * 75
            surface_height = len(terrain) * 75

        recovered_trial = trial_raw.copy()
        recovered_columns = {
            "corrected_display_x": [],
            "corrected_display_y": [],
            "recovered_surface_x": [],
            "recovered_surface_y": [],
            "recovered_tile_x": [],
            "recovered_tile_y": [],
            "base_aoi": [],
            "overlay_flag": [],
            "overlay_horizon": [],
            "counter_object_present": [],
            "counter_object_type": [],
        }
        ivt_samples = []
        fallback_sample_count = 0
        valid_surface_sample_count = 0
        for _, row in trial_raw.iterrows():
            valid = bool(row.get("gaze_validity", 0)) and pd.notna(row.get("gaze_x")) and pd.notna(row.get("gaze_y"))
            corrected = None
            surface = None
            tile = None
            if valid:
                raw_point = np.asarray([row["gaze_x"], row["gaze_y"], 1.0], dtype=np.float64)
                corrected = calibration_matrix @ raw_point
                if has_logged_surface:
                    valid_surface_sample_count += int(
                        pd.notna(row.get("gaze_surface_x")) and pd.notna(row.get("gaze_surface_y"))
                    )
                    surface = _reproject_legacy_surface_point(
                        corrected,
                        display_size=(
                            float(row["display_width_px"]),
                            float(row["display_height_px"]),
                        ),
                        game_window_origin=game_window_origin,
                        game_window_size=game_window_size,
                        map_display_size=(
                            float(row["map_display_width_px"]),
                            float(row["map_display_height_px"]),
                        ),
                        surface_size=(surface_width, surface_height),
                    )
                elif not has_logged_surface:
                    surface = np.asarray([corrected[0], corrected[1], 1.0]) @ surface_matrix
                if (
                    surface is not None
                    and 0 <= surface[0] < surface_width
                    and 0 <= surface[1] < surface_height
                ):
                    tile = (
                        int(surface[0] * len(terrain[0]) / surface_width),
                        int(surface[1] * len(terrain) / surface_height),
                    )
            step_row = None
            if pd.notna(row.get("step")):
                step_row = step_lookup.get((str(trial_id), int(row["step"])))
            context = _classify_context(tile, terrain, step_row)
            values = {
                "corrected_display_x": None if corrected is None else corrected[0],
                "corrected_display_y": None if corrected is None else corrected[1],
                "recovered_surface_x": None if surface is None else surface[0],
                "recovered_surface_y": None if surface is None else surface[1],
                "recovered_tile_x": None if tile is None else tile[0],
                "recovered_tile_y": None if tile is None else tile[1],
                "base_aoi": context["base_aoi"],
                "overlay_flag": context["overlay_flag"],
                "overlay_horizon": context["overlay_horizon"],
                "counter_object_present": context["counter_object_present"],
                "counter_object_type": context["counter_object_type"],
            }
            for name, value in values.items():
                recovered_columns[name].append(value)
            sample = {
                "timestamp_ms": int(row["timestamp_ms"]),
                "x": row.get("gaze_x"),
                "y": row.get("gaze_y"),
                "valid": valid,
            }
            if valid and step_row is not None:
                sample.update({"step": int(row["step"]), **context})
            ivt_samples.append(sample)

        for name, values in recovered_columns.items():
            recovered_trial[name] = values
        recovered_trial["recovery_mode"] = mode
        recovered_trial["trial_calibration_rmse"] = trial_rmse
        recovered_trial["selected_calibration_rmse"] = selected_rmse
        recovered_trial["calibration_git_blob"] = calibration_blob
        recovered_parts.append(recovered_trial)

        rows = build_ivt_fixation_rows(
            ivt_samples,
            pid=str(trial["pid"]),
            trial_id=str(trial_id),
            config=IVTConfig(
                viewing_distance_mm=float(trial.get("ivt_viewing_distance_mm", 600.0))
            ),
        )
        for fixation_row in rows:
            fixation_row.update(
                {
                    "layout": layout,
                    "time_constraint_ms": int(trial["time_constraint_ms"]),
                    "information_steps": int(trial["information_steps"]),
                    "recovery_mode": mode,
                }
            )
        fixation_rows.extend(rows)
        manifest_rows.append(
            {
                "trial_id": trial_id,
                "global_trial_index": int(trial["global_trial_index"]),
                "layout": layout,
                "trial_calibration_rmse": trial_rmse,
                "selected_calibration_rmse": selected_rmse,
                "rmse_difference": abs(selected_rmse - trial_rmse),
                "recovery_mode": mode,
                "calibration_git_blob": calibration_blob,
                "calibration_window_width_px": calibration_window_size[0],
                "calibration_window_height_px": calibration_window_size[1],
                "recovered_window_left_px": game_window_origin[0] if has_logged_surface else np.nan,
                "recovered_window_top_px": game_window_origin[1] if has_logged_surface else np.nan,
                "valid_surface_sample_count": valid_surface_sample_count,
                "fallback_sample_count": fallback_sample_count,
                "fallback_sample_ratio": (
                    fallback_sample_count / valid_surface_sample_count
                    if valid_surface_sample_count
                    else np.nan
                ),
            }
        )

    recovered_raw = pd.concat(recovered_parts, ignore_index=True)
    recovered_fixations = pd.DataFrame(fixation_rows)
    manifest = pd.DataFrame(manifest_rows).sort_values("global_trial_index")
    recovered_raw.to_csv(output_dir / "raw_gaze_recovered.csv", index=False)
    recovered_fixations.to_csv(output_dir / "fixation_recovered.csv", index=False)
    manifest.to_csv(output_dir / "trial_recovery_manifest.csv", index=False)

    dwell_rows = []
    for scope, selected in (
        ("all_trials", recovered_fixations),
        (
            "exact_only",
            recovered_fixations[recovered_fixations.recovery_mode.str.startswith("exact")],
        ),
    ):
        grouped = selected.groupby("base_aoi", dropna=False).fixation_duration_ms.sum()
        total = grouped.sum()
        for aoi, duration in grouped.items():
            dwell_rows.append(
                {
                    "scope": scope,
                    "base_aoi": aoi or "Outside",
                    "fixation_duration_ms": int(duration),
                    "dwell_ratio": float(duration / total) if total else np.nan,
                }
            )
    pd.DataFrame(dwell_rows).to_csv(output_dir / "aoi_dwell_summary.csv", index=False)

    exact_count = int(manifest.recovery_mode.str.startswith("exact").sum())
    print(f"Recovered {len(manifest)} trials: exact={exact_count}, approximate={len(manifest)-exact_count}")
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recover(args.input_dir, args.output_dir)
