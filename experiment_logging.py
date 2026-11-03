import csv
from dataclasses import replace
from datetime import datetime
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import Direction
from ivt import (
    IVTConfig,
    IVTFixation,
    classify_ivt_fixations,
    classify_vision_zone,
    visual_eccentricity_deg,
)


PATH_VISUALIZATION_ENDPOINT = "endpoint"
PATH_VISUALIZATION_ENDPOINT_PLUS_FIRST = "endpoint_plus_first"
PATH_VISUALIZATION_FULL = "full"
PATH_VISUALIZATION_NONE = "none"
PATH_RENDER_STYLE_HIGHLIGHT = "highlight"
AOI_SCHEMA_VERSION = "2.3"

STEP_LEGACY_FIELDS = [
    "pid",
    "trial_id",
    "step",
    "state_onset_ms",
    "human_action",
    "agent_action",
    "joint_action",
    "env_state",
    "human_position",
    "agent_position",
    "human_held_object",
    "agent_held_object",
    "pot_states",
    "is_collision",
    "human_action_due",
    "ai_action_due",
    "is_human_idle",
    "accumulated_score",
    "reward",
    "response_time_ms",
    "human_input_ms",
    "future_trajectory",
]
STEP_FIELDS_V2 = STEP_LEGACY_FIELDS + ["action_execution_ms", "done"]
STEP_INPUT_REVISION_FIELDS = [
    "first_human_action",
    "first_human_input_ms",
    "final_human_action",
    "final_human_input_ms",
    "human_input_changed",
    "response_input_source",
]
STEP_FIELDS = STEP_FIELDS_V2 + STEP_INPUT_REVISION_FIELDS

COOKING_TERRAIN_TYPES = {"O", "T", "D", "P"}
TRIAL_ANALYSIS_FIELDS = [
    "participant_index",
    "global_trial_index",
    "block_index",
    "block_trial_index",
    "condition_id",
    "information_steps",
    "layout_code",
    "williams_sequence_index",
    "layout_sequence_id",
    "layout_assignment_id",
    "trial_start_time",
    "trial_end_time",
    "trial_duration_ms",
    "completed",
    "total_gaze_samples",
    "valid_gaze_samples",
    "invalid_gaze_samples",
    "calibration_rmse",
]
TRIAL_IVT_FIELDS_V1 = [
    "ivt_screen_diagonal_inches",
    "ivt_screen_aspect_ratio",
    "ivt_viewing_distance_mm",
    "ivt_median_window_samples",
    "ivt_velocity_window_ms",
    "ivt_velocity_threshold_deg_s",
    "ivt_merge_gap_ms",
    "ivt_merge_angle_deg",
    "ivt_min_fixation_duration_ms",
    "ivt_foveal_threshold_deg",
    "ivt_parafoveal_threshold_deg",
]
TRIAL_IVT_FIELDS_V2 = TRIAL_IVT_FIELDS_V1 + [
    "ivt_viewing_distance_source",
    "ivt_viewing_distance_sample_count",
    "ivt_viewing_distance_mean_mm",
    "ivt_viewing_distance_median_mm",
    "ivt_viewing_distance_std_mm",
    "ivt_viewing_distance_window_sec",
]
TRIAL_GAZE_QUALITY_FIELDS = [
    "calibration_point_count",
    "calibration_quality_status",
    "calibration_quality_passed",
    "drift_check_status",
    "drift_check_angle_deg",
    "drift_check_sample_count",
    "drift_check_passed",
]
TRIAL_GAZE_PREVIEW_FIELDS = [
    "preview_sample_count",
    "preview_valid",
    "preview_accepted_ms",
]
TRIAL_IVT_FIELDS = (
    TRIAL_IVT_FIELDS_V2
    + TRIAL_GAZE_QUALITY_FIELDS
    + TRIAL_GAZE_PREVIEW_FIELDS
)
TRIAL_LEGACY_FIELDS = [
    "pid",
    "trial_id",
    "condition",
    "time_constraint_ms",
    "decision_horizon",
    "decision_steps_completed",
    "visualization_mode",
    "num_paths_shown",
    "layout",
    "score_final",
    "num_collisions",
    "num_human_idle",
    "screen_width_px",
    "screen_height_px",
]
TRIAL_PREDICTION_FIELDS = [
    field_name
    for horizon in range(1, 5)
    for field_name in (
        f"prediction_samples@{horizon}",
        f"path_acc@{horizon}",
        f"position_acc@{horizon}",
        f"action_acc@{horizon}",
    )
]
TRIAL_BASE_FIELDS = TRIAL_LEGACY_FIELDS + TRIAL_PREDICTION_FIELDS + ["mean_gaze_validity"]
TRIAL_FIELDS_V3 = TRIAL_BASE_FIELDS + TRIAL_ANALYSIS_FIELDS + TRIAL_IVT_FIELDS
TRIAL_PER_EYE_VALIDITY_FIELDS = [
    field_name
    for eye_prefix in ("left", "right")
    for field_name in (
        f"{eye_prefix}_gaze_point_sample_count",
        f"{eye_prefix}_valid_gaze_point_count",
        f"mean_{eye_prefix}_gaze_point_validity",
    )
]
TRIAL_FIELDS_WITH_PER_EYE_VALIDITY = TRIAL_FIELDS_V3 + TRIAL_PER_EYE_VALIDITY_FIELDS
TRIAL_SCORE_AUC_FIELDS = ["score_auc", "mean_cumulative_score"]
TRIAL_FIELDS_WITH_SCORE_AUC = TRIAL_FIELDS_WITH_PER_EYE_VALIDITY + TRIAL_SCORE_AUC_FIELDS
TRIAL_RUNTIME_FIELDS = [
    "target_game_fps",
    "render_frame_count",
    "environment_step_count",
    "mean_render_fps",
    "mean_environment_steps_per_second",
]
TRIAL_FIELDS = TRIAL_FIELDS_WITH_SCORE_AUC + TRIAL_RUNTIME_FIELDS
FIXATION_LEGACY_BASE_FIELDS = [
    "pid",
    "trial_id",
    "fixation_start_step",
    "fixation_end_step",
    "fixation_id",
    "fixation_start_ms",
    "fixation_end_ms",
    "fixation_duration_ms",
    "base_aoi",
    "base_position",
    "previous_base_aoi",
    "next_base_aoi",
    "overlay_flag",
    "overlay_type",
    "overlay_horizon",
    "path_rank",
    "overlay_underlying_aoi",
    "overlay_position",
    "overlay_matches",
    "counter_object_present",
    "counter_object_type",
]
FIXATION_BASE_FIELDS = FIXATION_LEGACY_BASE_FIELDS[:8] + [
    "fixation_x",
    "fixation_y",
    "fixation_sample_count",
    "mean_velocity_deg_s",
    "peak_velocity_deg_s",
    *FIXATION_LEGACY_BASE_FIELDS[8:],
    "aoi_matches",
]
FIXATION_FIELDS_V2 = FIXATION_BASE_FIELDS + ["record_type"]
FIXATION_FIELDS = FIXATION_BASE_FIELDS + ["context_segments", "record_type"]
RAW_GAZE_BASE_FIELDS = [
    "pid",
    "trial_id",
    "step",
    "current_state_onset_ms",
    "timestamp_ms",
    "gaze_x",
    "gaze_y",
    "gaze_validity",
]
RAW_GAZE_LEGACY_FIELDS = RAW_GAZE_BASE_FIELDS + [
    "callback_perf_counter",
    "tobii_device_timestamp_us",
    "tobii_system_timestamp_us",
]
RAW_GAZE_GEOMETRY_FIELDS = [
    "gaze_surface_x",
    "gaze_surface_y",
    "surface_width_px",
    "surface_height_px",
    "map_display_width_px",
    "map_display_height_px",
    "display_width_px",
    "display_height_px",
]
RAW_GAZE_ECCENTRICITY_FIELDS = [
    field_name
    for horizon in range(1, 5)
    for field_name in (f"h{horizon}_eccentricity_deg", f"h{horizon}_vision_zone")
]
RAW_GAZE_FIELDS_V2 = RAW_GAZE_LEGACY_FIELDS + RAW_GAZE_GEOMETRY_FIELDS + RAW_GAZE_ECCENTRICITY_FIELDS
RAW_GAZE_ORIGIN_FIELDS = [
    field_name
    for eye_prefix in ("left", "right")
    for field_name in (
        f"{eye_prefix}_gaze_origin_validity",
        f"{eye_prefix}_gaze_origin_x_mm",
        f"{eye_prefix}_gaze_origin_y_mm",
        f"{eye_prefix}_gaze_origin_z_mm",
    )
]
RAW_GAZE_FIELDS_V3 = RAW_GAZE_FIELDS_V2 + RAW_GAZE_ORIGIN_FIELDS
RAW_GAZE_POINT_VALIDITY_FIELDS = [
    "left_gaze_point_validity",
    "right_gaze_point_validity",
]
RAW_GAZE_FIELDS = RAW_GAZE_FIELDS_V3 + RAW_GAZE_POINT_VALIDITY_FIELDS


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_position(position: Optional[Sequence[int]]) -> Optional[List[int]]:
    if position is None:
        return None
    return [int(position[0]), int(position[1])]


def _normalize_logged_overlay_type(overlay_type: Optional[str]) -> Optional[str]:
    if overlay_type == "marker":
        return "path"
    return overlay_type


def _sanitize_session_name_part(value: Optional[Any], fallback: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return fallback
    sanitized = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in text).strip("-_")
    return sanitized or fallback


def _condition_to_info_amount_label(condition: str) -> str:
    text = str(condition).strip().upper()
    if len(text) != 3 or text[1] != "-" or text[2] not in "012345678":
        return "unknown_info"
    condition_index = int(text[2])
    info_index = condition_index % 3
    if info_index == 0:
        return "0step"
    if info_index == 1:
        return "2step"
    return "4step"


def _ensure_csv_schema(
    file_path: str,
    target_fields: Sequence[str],
    known_schemas: Sequence[Sequence[str]],
) -> None:
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return

    with open(file_path, "r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.reader(file_obj))
    if not rows:
        return

    header = rows[0]
    nonempty_rows = [row for row in rows[1:] if row]
    if header == list(target_fields) and all(len(row) == len(target_fields) for row in nonempty_rows):
        return

    schemas_by_length = {len(schema): list(schema) for schema in known_schemas}
    migrated_rows = []
    for row in rows[1:]:
        if not row:
            migrated_rows.append([])
            continue
        if row[0] == "pid":
            continue
        if len(row) == len(header):
            source_fields = header
        else:
            source_fields = schemas_by_length.get(len(row))
        if source_fields is None:
            raise ValueError(
                f"Cannot migrate {file_path}: unsupported row width {len(row)}."
            )
        source_row = dict(zip(source_fields, row))
        migrated_rows.append([source_row.get(field_name, "") for field_name in target_fields])

    backup_path = f"{file_path}.pre_schema_migration.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
    with open(file_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(target_fields)
        writer.writerows(migrated_rows)


def _get_experiment_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _checkpoint_path(model: Any) -> Optional[str]:
    if model is None:
        return None
    for attribute_name in ("checkpoint_path", "model_path"):
        value = getattr(model, attribute_name, None)
        if value:
            return os.path.abspath(os.fspath(value))
    return None


def serialize_calibration_info(calibration: Any) -> Dict[str, Any]:
    if calibration is None:
        return {
            "enabled": False,
            "layout_name": None,
            "participant_id": None,
            "calibration_scope": None,
            "format_version": None,
            "loaded": False,
            "has_transform": False,
            "calibration_path": None,
            "loaded_from_path": None,
            "required_points": None,
            "sample_count": 0,
            "fit_rmse": None,
            "quality_status": "unavailable",
            "can_start": False,
            "target_span": None,
            "transform_matrix": None,
        }

    transform_matrix = getattr(calibration, "transform_matrix", None)
    if transform_matrix is not None:
        transform_matrix = transform_matrix.tolist()
    return {
        "enabled": True,
        "layout_name": getattr(calibration, "layout_name", None),
        "participant_id": getattr(calibration, "participant_id", None),
        "calibration_scope": getattr(calibration, "calibration_scope", None),
        "format_version": getattr(calibration, "format_version", None),
        "loaded": bool(getattr(calibration, "loaded_from_path", None)),
        "has_transform": bool(getattr(calibration, "has_transform", False)),
        "calibration_path": getattr(calibration, "calibration_path", None),
        "loaded_from_path": getattr(calibration, "loaded_from_path", None),
        "required_points": getattr(calibration, "required_points", None),
        "sample_count": int(getattr(calibration, "sample_count", 0)),
        "fit_rmse": getattr(calibration, "fit_rmse", None),
        "quality_status": getattr(calibration, "quality_status", "unavailable"),
        "can_start": bool(getattr(calibration, "can_start", False)),
        "target_span": list(getattr(calibration, "target_span", (0.0, 0.0))),
        "transform_matrix": transform_matrix,
    }


def _serialize_action(action: Any) -> str:
    if action == Action.INTERACT:
        return "interact"
    if action == Action.STAY:
        return "stay"
    if action == Direction.NORTH:
        return "north"
    if action == Direction.SOUTH:
        return "south"
    if action == Direction.WEST:
        return "west"
    if action == Direction.EAST:
        return "east"
    return str(action)


def resolve_response_input(
    first_action: Optional[Any],
    first_input_ms: Optional[int],
    final_action: Optional[Any],
    final_input_ms: Optional[int],
) -> Tuple[Optional[int], Optional[str], bool]:
    """Select the timestamp used for reaction time without losing revisions."""

    if first_input_ms is None and final_input_ms is None:
        return None, None, False
    if first_input_ms is None:
        return final_input_ms, "final", False
    if final_input_ms is None:
        return first_input_ms, "first", False
    changed = first_action != final_action
    if changed:
        return final_input_ms, "final", True
    return first_input_ms, "first", False


def _serialize_object_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_serialize_object_state(component) for component in value]
    if isinstance(value, list):
        return [_serialize_object_state(component) for component in value]
    if isinstance(value, dict):
        return {str(key): _serialize_object_state(component) for key, component in value.items()}
    return value


def _serialize_held_object(player: Any) -> Optional[Dict[str, Any]]:
    if player is None or not player.has_object():
        return None
    held_object = player.get_object()
    return {
        "name": held_object.name,
        "position": _normalize_position(held_object.position),
        "state": _serialize_object_state(getattr(held_object, "state", None)),
    }


def _simplify_pot_states(mdp: Any, state: Any) -> Dict[str, Any]:
    raw_pot_states = mdp.get_pot_states(state)
    simplified = {
        "empty": [_normalize_position(position) for position in raw_pot_states.get("empty", [])],
        "onion": {},
        "tomato": {},
    }
    for soup_type in ("onion", "tomato"):
        for key, positions in raw_pot_states.get(soup_type, {}).items():
            simplified[soup_type][key] = [_normalize_position(position) for position in positions]
    return simplified


def _normalize_visualized_path(
    path: Sequence[Any],
    max_steps: int,
    stop_at_interaction: bool,
) -> Tuple[Tuple[Tuple[int, int], bool, Optional[Tuple[int, int]]], ...]:
    normalized_steps = []
    for step in path[:max_steps]:
        if isinstance(step, dict):
            position = (int(step["position"][0]), int(step["position"][1]))
            interact_flag = bool(step.get("interact", False))
            interact_position = None
            if interact_flag:
                interact_position = tuple(int(component) for component in step.get("interact_position", position))
            normalized_steps.append((position, interact_flag, interact_position))
            if stop_at_interaction and interact_flag:
                break
            continue

        position = (int(step[0]), int(step[1]))
        interact_flag = len(step) >= 3 and step[2] == "interact"
        interact_position = position if interact_flag else None
        normalized_steps.append((position, interact_flag, interact_position))
        if stop_at_interaction and interact_flag:
            break

    return tuple(normalized_steps)


def _build_visible_path_signature_key(
    normalized_path: Sequence[Tuple[Tuple[int, int], bool, Optional[Tuple[int, int]]]],
    include_interactions: bool,
) -> Tuple[Any, ...]:
    visible_steps = []
    for position, interact_flag, interact_position in normalized_path:
        signature = (
            (position, interact_flag, interact_position)
            if include_interactions
            else position
        )
        if not visible_steps or signature != visible_steps[-1]:
            visible_steps.append(signature)
    return tuple(visible_steps)


def select_ranked_normalized_entries(
    raw_paths: Sequence[Any],
    raw_weights: Optional[Sequence[float]],
    greedy_count: int,
    include_interactions: bool,
    max_ranked_paths: int,
    max_steps: int,
    stop_at_interaction: bool,
) -> List[Dict[str, Any]]:
    if not raw_paths or max_ranked_paths <= 0:
        return []

    if raw_weights is None or len(raw_weights) != len(raw_paths):
        normalized_weights = [1.0 / max(1, len(raw_paths))] * len(raw_paths)
    else:
        total_weight = float(sum(max(0.0, float(weight)) for weight in raw_weights))
        if total_weight <= 0:
            normalized_weights = [1.0 / max(1, len(raw_paths))] * len(raw_paths)
        else:
            normalized_weights = [max(0.0, float(weight)) / total_weight for weight in raw_weights]

    normalized_paths = [
        _normalize_visualized_path(path, max_steps, stop_at_interaction)
        for path in raw_paths
    ]

    ranked_entries = []
    if greedy_count > 0 and normalized_paths:
        greedy_path = normalized_paths[0]
        greedy_key = _build_visible_path_signature_key(greedy_path, include_interactions)
        ranked_entries.append(
            {
                "normalized_path": greedy_path,
                "weight": float(normalized_weights[0]),
            }
        )

        unique_sampled_paths: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for path, weight in zip(normalized_paths[greedy_count:], normalized_weights[greedy_count:]):
            path_key = _build_visible_path_signature_key(path, include_interactions)
            if path_key == greedy_key:
                continue
            existing = unique_sampled_paths.get(path_key)
            if existing is None:
                unique_sampled_paths[path_key] = {
                    "normalized_path": path,
                    "weight": float(weight),
                }
            else:
                existing["weight"] += float(weight)

        ranked_entries.extend(
            sorted(unique_sampled_paths.values(), key=lambda item: item["weight"], reverse=True)[: max(0, max_ranked_paths - 1)]
        )
    else:
        unique_paths: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for path, weight in zip(normalized_paths, normalized_weights):
            path_key = _build_visible_path_signature_key(path, include_interactions)
            existing = unique_paths.get(path_key)
            if existing is None:
                unique_paths[path_key] = {
                    "normalized_path": path,
                    "weight": float(weight),
                }
            else:
                existing["weight"] += float(weight)
        ranked_entries = sorted(unique_paths.values(), key=lambda item: item["weight"], reverse=True)[:max_ranked_paths]

    for index, entry in enumerate(ranked_entries, start=1):
        entry["rank"] = index
    return ranked_entries


def _normalized_path_to_steps(
    normalized_path: Sequence[Tuple[Tuple[int, int], bool, Optional[Tuple[int, int]]]]
) -> List[Dict[str, Any]]:
    steps = []
    for position, interact_flag, interact_position in normalized_path:
        if interact_flag:
            steps.append(
                {
                    "position": tuple(position),
                    "interact": True,
                    "interact_position": tuple(interact_position if interact_position is not None else position),
                }
            )
        else:
            steps.append(
                {
                    "position": tuple(position),
                    "interact": False,
                }
            )
    return steps


def build_visible_overlay_entries(
    rollout_results: Optional[Dict[str, Any]],
    visualization_mode: str,
    num_steps: int,
    stop_path_at_interaction: bool,
    max_ranked_paths,
    show_interaction_overlay: bool = True,
    path_render_style: str = PATH_RENDER_STYLE_HIGHLIGHT,
) -> List[Dict[str, Any]]:
    if rollout_results is None or visualization_mode == PATH_VISUALIZATION_NONE:
        return []

    if (
        not rollout_results.get("ai_paths")
        and rollout_results.get("ai_position_probabilities") is not None
        and rollout_results.get("ai_valid_positions") is not None
    ):
        position_probabilities = rollout_results["ai_position_probabilities"][:num_steps]
        valid_positions = rollout_results["ai_valid_positions"]
        argmax_indices = [max(range(len(step_probabilities)), key=lambda idx: float(step_probabilities[idx])) for step_probabilities in position_probabilities]
        derived_steps = []
        for horizon, position_index in enumerate(argmax_indices, start=1):
            position = tuple(int(component) for component in valid_positions[position_index])
            derived_steps.append(
                {
                    "tile": position,
                    "overlay_position": _normalize_position(position),
                    "overlay_type": "path",
                    "overlay_horizon": horizon,
                    "path_rank": 1,
                }
            )
        if derived_steps and visualization_mode == PATH_VISUALIZATION_ENDPOINT:
            endpoint = dict(derived_steps[-1])
            endpoint["overlay_type"] = "marker"
            return [endpoint]
        if derived_steps and visualization_mode == PATH_VISUALIZATION_ENDPOINT_PLUS_FIRST:
            endpoint = dict(derived_steps[-1])
            endpoint["overlay_type"] = "marker"
            return [endpoint, derived_steps[0]]
        return derived_steps

    ranked_entries = select_ranked_normalized_entries(
        rollout_results.get("ai_paths", []),
        rollout_results.get("ai_path_weights"),
        int(rollout_results.get("ai_greedy_path_count", 0)),
        include_interactions=(show_interaction_overlay or visualization_mode != PATH_VISUALIZATION_FULL),
        max_ranked_paths=max(0, min(2, int(max_ranked_paths))),
        max_steps=num_steps,
        stop_at_interaction=stop_path_at_interaction,
    )

    overlay_entries: List[Dict[str, Any]] = []
    for ranked_entry in ranked_entries:
        rank = int(ranked_entry["rank"])
        steps = _normalized_path_to_steps(ranked_entry["normalized_path"])
        if not steps:
            continue

        if visualization_mode == PATH_VISUALIZATION_ENDPOINT:
            endpoint_step = steps[-1]
            endpoint_tile = (
                tuple(endpoint_step["interact_position"])
                if endpoint_step.get("interact", False)
                else tuple(endpoint_step["position"])
            )
            overlay_entries.append(
                {
                    "tile": endpoint_tile,
                    "overlay_position": _normalize_position(endpoint_tile),
                    "overlay_type": "marker",
                    "overlay_horizon": len(steps),
                    "path_rank": rank,
                }
            )
            continue

        if visualization_mode == PATH_VISUALIZATION_ENDPOINT_PLUS_FIRST:
            endpoint_step = steps[-1]
            endpoint_tile = (
                tuple(endpoint_step["interact_position"])
                if endpoint_step.get("interact", False)
                else tuple(endpoint_step["position"])
            )
            overlay_entries.append(
                {
                    "tile": endpoint_tile,
                    "overlay_position": _normalize_position(endpoint_tile),
                    "overlay_type": "marker",
                    "overlay_horizon": len(steps),
                    "path_rank": rank,
                }
            )
            first_step = steps[0]
            first_tile = (
                tuple(first_step["interact_position"])
                if path_render_style == PATH_RENDER_STYLE_HIGHLIGHT and first_step.get("interact", False)
                else tuple(first_step["position"])
            )
            overlay_entries.append(
                {
                    "tile": first_tile,
                    "overlay_position": _normalize_position(first_tile),
                    "overlay_type": "path",
                    "overlay_horizon": 1,
                    "path_rank": rank,
                }
            )
            if (
                path_render_style != PATH_RENDER_STYLE_HIGHLIGHT
                and show_interaction_overlay
                and first_step.get("interact", False)
            ):
                interact_tile = tuple(first_step["interact_position"])
                overlay_entries.append(
                    {
                        "tile": interact_tile,
                        "overlay_position": _normalize_position(interact_tile),
                        "overlay_type": "path",
                        "overlay_horizon": 1,
                        "path_rank": rank,
                    }
                )
            continue

        for horizon, step in enumerate(steps, start=1):
            tile = (
                tuple(step["interact_position"])
                if path_render_style == PATH_RENDER_STYLE_HIGHLIGHT and step.get("interact", False)
                else tuple(step["position"])
            )
            overlay_entries.append(
                {
                    "tile": tile,
                    "overlay_position": _normalize_position(tile),
                    "overlay_type": "path",
                    "overlay_horizon": horizon,
                    "path_rank": rank,
                }
            )
            if (
                path_render_style != PATH_RENDER_STYLE_HIGHLIGHT
                and show_interaction_overlay
                and step.get("interact", False)
            ):
                interact_tile = tuple(step["interact_position"])
                overlay_entries.append(
                    {
                        "tile": interact_tile,
                        "overlay_position": _normalize_position(interact_tile),
                        "overlay_type": "path",
                        "overlay_horizon": horizon,
                        "path_rank": rank,
                    }
                )

    deduped_entries: List[Dict[str, Any]] = []
    seen = set()
    for entry in overlay_entries:
        entry_key = (
            tuple(entry["tile"]),
            entry["overlay_type"],
            int(entry["overlay_horizon"]),
            int(entry["path_rank"]),
        )
        if entry_key in seen:
            continue
        seen.add(entry_key)
        deduped_entries.append(entry)

    return deduped_entries


def _classify_base_aoi(
    tile: Optional[Tuple[int, int]],
    terrain_mtx: Optional[Sequence[Sequence[str]]],
    state: Any,
    self_player_index: int,
    partner_player_index: int,
) -> Tuple[str, Optional[List[int]]]:
    if tile is None or terrain_mtx is None or state is None:
        return "Outside", None

    player_positions = [tuple(player.position) for player in state.players]
    if self_player_index < len(player_positions) and tile == player_positions[self_player_index]:
        return "Self", _normalize_position(tile)
    if partner_player_index < len(player_positions) and tile == player_positions[partner_player_index]:
        return "Partner", _normalize_position(tile)

    tile_x, tile_y = tile
    if tile_y < 0 or tile_y >= len(terrain_mtx):
        return "Outside", None
    if tile_x < 0 or tile_x >= len(terrain_mtx[tile_y]):
        return "Outside", None

    terrain_type = terrain_mtx[tile_y][tile_x]
    if terrain_type in COOKING_TERRAIN_TYPES:
        return "Cooking_Area", _normalize_position(tile)
    if terrain_type == "S":
        return "Serving_Area", _normalize_position(tile)
    if terrain_type == "X":
        return "Counter", _normalize_position(tile)
    if terrain_type == " ":
        return "Corridor", _normalize_position(tile)
    return "Outside", _normalize_position(tile)


def _fixation_context(fixation: IVTFixation) -> Dict[str, Any]:
    context_samples = [sample for sample in fixation.samples if "base_aoi" in sample]
    if not context_samples:
        return {
            "fixation_start_step": "",
            "fixation_end_step": "",
            "base_aoi": "",
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
            "aoi_matches": [],
        }

    signature_counts: Dict[Tuple[Any, ...], int] = {}
    signature_samples: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    aoi_counts: Dict[Tuple[str, Optional[Tuple[int, int]]], int] = {}
    overlay_matches: List[Dict[str, Any]] = []
    seen_overlay_matches = set()
    for sample in context_samples:
        base_position = tuple(sample["base_position"]) if sample.get("base_position") is not None else None
        signature = (
            sample["base_aoi"],
            base_position,
            int(sample.get("overlay_flag", 0)),
            sample.get("overlay_type", "none"),
            sample.get("overlay_horizon"),
            sample.get("path_rank"),
            sample.get("overlay_underlying_aoi"),
            tuple(sample["overlay_position"]) if sample.get("overlay_position") is not None else None,
            int(sample.get("counter_object_present", 0)),
            sample.get("counter_object_type"),
        )
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        signature_samples.setdefault(signature, sample)
        aoi_key = (sample["base_aoi"], base_position)
        aoi_counts[aoi_key] = aoi_counts.get(aoi_key, 0) + 1
        for overlay_match in sample.get("overlay_matches", []):
            serialized_match = _json_dumps(overlay_match)
            if serialized_match not in seen_overlay_matches:
                seen_overlay_matches.add(serialized_match)
                overlay_matches.append(overlay_match)

    dominant_signature = max(signature_counts, key=signature_counts.get)
    dominant = signature_samples[dominant_signature]
    total_context_samples = len(context_samples)
    aoi_matches = [
        {
            "base_aoi": base_aoi,
            "base_position": _normalize_position(base_position),
            "sample_count": count,
            "sample_fraction": count / total_context_samples,
        }
        for (base_aoi, base_position), count in sorted(
            aoi_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1] or (-1, -1)),
        )
    ]
    return {
        "fixation_start_step": int(context_samples[0]["step"]),
        "fixation_end_step": int(context_samples[-1]["step"]),
        "base_aoi": dominant["base_aoi"],
        "base_position": dominant.get("base_position"),
        "overlay_flag": int(dominant.get("overlay_flag", 0)),
        "overlay_type": dominant.get("overlay_type", "none"),
        "overlay_horizon": dominant.get("overlay_horizon"),
        "path_rank": dominant.get("path_rank"),
        "overlay_underlying_aoi": dominant.get("overlay_underlying_aoi"),
        "overlay_position": dominant.get("overlay_position"),
        "overlay_matches": overlay_matches,
        "counter_object_present": int(dominant.get("counter_object_present", 0)),
        "counter_object_type": dominant.get("counter_object_type"),
        "aoi_matches": aoi_matches,
    }


def _fixation_context_segments(fixation: IVTFixation) -> List[Dict[str, Any]]:
    """Keep contiguous sample-level AOI contexts inside one fixation."""

    segments: List[Dict[str, Any]] = []
    current_payload = None
    current_signature = None
    current_start_ms = None

    def close_segment(end_ms: int) -> None:
        if current_payload is None or current_start_ms is None or end_ms <= current_start_ms:
            return
        segments.append(
            {
                "start_ms": int(current_start_ms),
                "end_ms": int(end_ms),
                **current_payload,
            }
        )

    for sample in fixation.samples:
        timestamp_ms = int(sample["timestamp_ms"])
        if "base_aoi" not in sample:
            payload = None
        else:
            payload = {
                "step": int(sample["step"]),
                "base_aoi": sample["base_aoi"],
                "base_position": sample.get("base_position"),
                "overlay_flag": int(sample.get("overlay_flag", 0)),
                "overlay_type": sample.get("overlay_type", "none"),
                "overlay_horizon": sample.get("overlay_horizon"),
                "path_rank": sample.get("path_rank"),
                "overlay_underlying_aoi": sample.get("overlay_underlying_aoi"),
                "overlay_position": sample.get("overlay_position"),
                "overlay_matches": sample.get("overlay_matches", []),
                "counter_object_present": int(sample.get("counter_object_present", 0)),
                "counter_object_type": sample.get("counter_object_type"),
            }
        signature = None if payload is None else _json_dumps(payload)
        if signature == current_signature:
            continue
        close_segment(timestamp_ms)
        current_payload = payload
        current_signature = signature
        current_start_ms = timestamp_ms if payload is not None else None

    close_segment(fixation.end_ms)
    return segments


def build_ivt_fixation_rows(
    samples: Sequence[Dict[str, Any]],
    *,
    pid: str,
    trial_id: str,
    config: Optional[IVTConfig] = None,
) -> List[Dict[str, Any]]:
    fixations = classify_ivt_fixations(samples, config)
    contexts = [_fixation_context(fixation) for fixation in fixations]
    rows = []
    for index, (fixation, context) in enumerate(zip(fixations, contexts)):
        previous_base_aoi = contexts[index - 1]["base_aoi"] if index > 0 else ""
        next_base_aoi = contexts[index + 1]["base_aoi"] if index + 1 < len(contexts) else ""
        rows.append(
            {
                "pid": pid,
                "trial_id": trial_id,
                "fixation_start_step": context["fixation_start_step"],
                "fixation_end_step": context["fixation_end_step"],
                "fixation_id": index + 1,
                "fixation_start_ms": fixation.start_ms,
                "fixation_end_ms": fixation.end_ms,
                "fixation_duration_ms": fixation.duration_ms,
                "fixation_x": fixation.centroid_x,
                "fixation_y": fixation.centroid_y,
                "fixation_sample_count": fixation.sample_count,
                "mean_velocity_deg_s": fixation.mean_velocity_deg_s,
                "peak_velocity_deg_s": fixation.peak_velocity_deg_s,
                "base_aoi": context["base_aoi"],
                "base_position": _json_dumps(context["base_position"]),
                "previous_base_aoi": previous_base_aoi,
                "next_base_aoi": next_base_aoi,
                "overlay_flag": context["overlay_flag"],
                "overlay_type": context["overlay_type"],
                "overlay_horizon": (
                    "" if context["overlay_horizon"] is None else int(context["overlay_horizon"])
                ),
                "path_rank": "" if context["path_rank"] is None else int(context["path_rank"]),
                "overlay_underlying_aoi": context["overlay_underlying_aoi"] or "",
                "overlay_position": _json_dumps(context["overlay_position"]),
                "overlay_matches": _json_dumps(context["overlay_matches"]),
                "counter_object_present": context["counter_object_present"],
                "counter_object_type": context["counter_object_type"] or "",
                "aoi_matches": _json_dumps(context["aoi_matches"]),
                "context_segments": _json_dumps(_fixation_context_segments(fixation)),
                "record_type": "ivt_fixation",
            }
        )
    return rows


class ExperimentLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        log_dir: str,
        pid: str,
        trial_id: Optional[str],
        condition: str,
        layout: str,
        time_constraint_ms: int,
        decision_horizon: int,
        visualization_mode: str,
        num_paths_shown: int,
        num_steps: int,
        stop_path_at_interaction: bool,
        trial_metadata: Optional[Dict[str, Any]] = None,
        ivt_config: Optional[IVTConfig] = None,
        ai_model: Any = None,
        path_predictor: Any = None,
        gaze_calibration: Any = None,
        target_game_fps: Optional[float] = None,
    ):
        self.enabled = bool(enabled)
        self.layout = layout
        self.pid = pid
        self.condition = condition
        self.time_constraint_ms = int(time_constraint_ms)
        self.decision_horizon = int(decision_horizon)
        self.visualization_mode = visualization_mode
        self.num_paths_shown = int(num_paths_shown)
        self.num_steps = int(num_steps)
        self.stop_path_at_interaction = bool(stop_path_at_interaction)
        self.trial_metadata = dict(trial_metadata or {})
        self.ivt_config = ivt_config or IVTConfig()
        self.viewing_distance_info = {
            "source": "configured_fallback",
            "applied_distance_mm": self.ivt_config.viewing_distance_mm,
            "sample_count": 0,
            "mean_mm": None,
            "median_mm": None,
            "std_mm": None,
            "window_sec": None,
        }
        self.gaze_quality_info = {
            "calibration_point_count": 0,
            "calibration_quality_status": "unavailable",
            "calibration_quality_passed": False,
            "drift_check_status": "not_run",
            "drift_check_angle_deg": None,
            "drift_check_sample_count": 0,
            "drift_check_passed": False,
            "preview_sample_count": 0,
            "preview_valid": False,
            "preview_accepted_ms": None,
        }
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        default_trial_id = "_".join(
            [
                timestamp,
                _sanitize_session_name_part(pid, "no_pid"),
                _sanitize_session_name_part(condition, "no_condition"),
                _sanitize_session_name_part(layout, "no_layout"),
                str(self.time_constraint_ms),
                _condition_to_info_amount_label(condition),
            ]
        )
        self.trial_id = trial_id or default_trial_id
        self.session_name = _sanitize_session_name_part(pid, "no_pid")
        self.session_dir = os.path.join(os.path.abspath(log_dir), self.session_name)
        self._screen_width_px = None
        self._screen_height_px = None
        self._gaze_sample_count = 0
        self._valid_gaze_sample_count = 0
        self._per_eye_gaze_point_counts = {
            eye_prefix: {"total": 0, "valid": 0}
            for eye_prefix in ("left", "right")
        }
        self._trial_start_time = None
        self._trial_start_perf_counter = None
        self._score_auc = 0.0
        self._score_auc_sample_count = 0
        self._target_game_fps = target_game_fps
        self._render_frame_count = 0
        self._environment_step_count = 0
        self._raw_fixation_samples: List[Dict[str, Any]] = []
        self._pending_path_predictions: Dict[int, List[Dict[str, Any]]] = {}
        self._path_prediction_metrics = {
            horizon: {
                "count": 0,
                "path_correct": 0,
                "position_correct": 0,
                "action_count": 0,
                "action_correct": 0,
            }
            for horizon in range(1, 5)
        }

        self._trial_file = None
        self._step_file = None
        self._raw_gaze_file = None
        self._fixation_file = None
        self._prediction_accuracy_file = None
        self._trial_writer = None
        self._step_writer = None
        self._raw_gaze_writer = None
        self._fixation_writer = None
        self._prediction_accuracy_writer = None

        if not self.enabled:
            return

        os.makedirs(self.session_dir, exist_ok=True)
        _ensure_csv_schema(
            os.path.join(self.session_dir, "trial.csv"),
            TRIAL_FIELDS,
            (
                TRIAL_LEGACY_FIELDS,
                TRIAL_LEGACY_FIELDS + ["mean_gaze_validity"],
                TRIAL_BASE_FIELDS,
                TRIAL_BASE_FIELDS + TRIAL_ANALYSIS_FIELDS,
                TRIAL_BASE_FIELDS + TRIAL_ANALYSIS_FIELDS + TRIAL_IVT_FIELDS_V1,
                TRIAL_BASE_FIELDS + TRIAL_ANALYSIS_FIELDS + TRIAL_IVT_FIELDS_V2,
                TRIAL_BASE_FIELDS
                + TRIAL_ANALYSIS_FIELDS
                + TRIAL_IVT_FIELDS_V2
                + TRIAL_GAZE_QUALITY_FIELDS,
                TRIAL_FIELDS_V3,
                TRIAL_FIELDS_WITH_PER_EYE_VALIDITY,
                TRIAL_FIELDS_WITH_SCORE_AUC,
                TRIAL_FIELDS,
            ),
        )
        _ensure_csv_schema(
            os.path.join(self.session_dir, "fixation.csv"),
            FIXATION_FIELDS,
            (
                FIXATION_LEGACY_BASE_FIELDS,
                FIXATION_LEGACY_BASE_FIELDS + ["record_type"],
                FIXATION_BASE_FIELDS,
                FIXATION_FIELDS_V2,
                FIXATION_FIELDS,
            ),
        )
        _ensure_csv_schema(
            os.path.join(self.session_dir, "raw_gaze.csv"),
            RAW_GAZE_FIELDS,
            (
                RAW_GAZE_BASE_FIELDS,
                RAW_GAZE_LEGACY_FIELDS,
                RAW_GAZE_FIELDS_V2,
                RAW_GAZE_FIELDS_V3,
                RAW_GAZE_FIELDS,
            ),
        )
        _ensure_csv_schema(
            os.path.join(self.session_dir, "step.csv"),
            STEP_FIELDS,
            (STEP_LEGACY_FIELDS, STEP_FIELDS_V2, STEP_FIELDS),
        )
        self._trial_file, write_trial_header = self._open_trial_file("trial.csv")
        self._step_file, write_step_header = self._open_trial_file("step.csv")
        self._raw_gaze_file, write_raw_gaze_header = self._open_trial_file("raw_gaze.csv")
        self._fixation_file, write_fixation_header = self._open_trial_file("fixation.csv")
        self._prediction_accuracy_file, write_prediction_header = self._open_trial_file(
            "prediction_accuracy.csv"
        )

        self._trial_writer = csv.DictWriter(
            self._trial_file,
            fieldnames=TRIAL_FIELDS,
        )
        self._step_writer = csv.DictWriter(
            self._step_file,
            fieldnames=STEP_FIELDS,
        )
        self._raw_gaze_writer = csv.DictWriter(
            self._raw_gaze_file,
            fieldnames=RAW_GAZE_FIELDS,
        )
        self._fixation_writer = csv.DictWriter(
            self._fixation_file,
            fieldnames=FIXATION_FIELDS,
        )
        self._prediction_accuracy_writer = csv.DictWriter(
            self._prediction_accuracy_file,
            fieldnames=[
                "pid",
                "trial_id",
                "prediction_step",
                "actual_step",
                "horizon",
                "predicted_position",
                "actual_position",
                "position_correct",
                "predicted_action",
                "actual_action",
                "action_correct",
                "predicted_interact",
                "actual_interact",
                "interact_correct",
                "predicted_interact_position",
                "actual_interact_position",
                "interact_target_correct",
                "path_correct",
            ],
        )
        if write_trial_header:
            self._trial_writer.writeheader()
        if write_step_header:
            self._step_writer.writeheader()
        if write_raw_gaze_header:
            self._raw_gaze_writer.writeheader()
        if write_fixation_header:
            self._fixation_writer.writeheader()
        if write_prediction_header:
            self._prediction_accuracy_writer.writeheader()
        self.write_metadata(
            ai_checkpoint=_checkpoint_path(ai_model),
            path_predictor_checkpoint=_checkpoint_path(path_predictor),
            calibration=serialize_calibration_info(gaze_calibration),
        )

    def _open_trial_file(self, file_name: str):
        file_path = os.path.join(self.session_dir, file_name)
        has_existing_rows = os.path.exists(file_path) and os.path.getsize(file_path) > 0
        file_obj = open(file_path, "a", newline="", encoding="utf-8")
        if has_existing_rows:
            file_obj.write("\n")
            file_obj.flush()
        return file_obj, not has_existing_rows

    def set_screen_size(self, width: Optional[int], height: Optional[int]) -> None:
        if not self.enabled:
            return
        if width is not None:
            self._screen_width_px = int(width)
        if height is not None:
            self._screen_height_px = int(height)

    def set_viewing_distance(self, estimate: Dict[str, Any]) -> None:
        applied_distance = float(estimate["applied_distance_mm"])
        if applied_distance <= 0:
            raise ValueError("applied viewing distance must be positive")
        self.ivt_config = replace(
            self.ivt_config,
            viewing_distance_mm=applied_distance,
        )
        self.viewing_distance_info = {
            "source": str(estimate.get("source", "configured_fallback")),
            "applied_distance_mm": applied_distance,
            "sample_count": int(estimate.get("sample_count", 0)),
            "mean_mm": estimate.get("mean_mm"),
            "median_mm": estimate.get("median_mm"),
            "std_mm": estimate.get("std_mm"),
            "window_sec": estimate.get("window_sec"),
        }
        self.write_metadata()

    def set_calibration_quality(self, calibration: Any) -> None:
        if calibration is None:
            return
        quality_status = str(getattr(calibration, "quality_status", "unavailable"))
        self.gaze_quality_info.update(
            {
                "calibration_point_count": int(getattr(calibration, "sample_count", 0)),
                "calibration_quality_status": quality_status,
                "calibration_quality_passed": quality_status in {"pass", "warning"},
            }
        )
        self.write_metadata(calibration=serialize_calibration_info(calibration))

    def set_drift_check(self, result: Dict[str, Any]) -> None:
        self.gaze_quality_info.update(
            {
                "drift_check_status": str(result.get("status", "not_run")),
                "drift_check_angle_deg": result.get("angle_deg"),
                "drift_check_sample_count": int(result.get("sample_count", 0)),
                "drift_check_passed": bool(result.get("passed", False)),
            }
        )
        self.write_metadata()

    def set_gaze_preview(self, result: Dict[str, Any], *, accepted_ms: int) -> None:
        self.gaze_quality_info.update(
            {
                "preview_sample_count": int(result.get("sample_count", 0)),
                "preview_valid": bool(result.get("valid", False)),
                "preview_accepted_ms": int(accepted_ms),
            }
        )
        self.write_metadata()

    def write_metadata(
        self,
        *,
        ai_checkpoint: Optional[str] = None,
        path_predictor_checkpoint: Optional[str] = None,
        calibration: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        metadata_path = os.path.join(self.session_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file_obj:
                metadata = json.load(file_obj)
        else:
            metadata = {}

        metadata["experiment_git_commit"] = _get_experiment_git_commit()
        metadata["aoi_schema_version"] = AOI_SCHEMA_VERSION
        metadata.setdefault("metadata_schema_version", 1)
        trials = metadata.setdefault("trials", {})
        trial_metadata = trials.setdefault(self.trial_id, {})
        trial_metadata.update(
            {
                "participant_id": self.pid,
                "condition": self.condition,
                "layout": self.layout,
                "agent_observation_layout": self.trial_metadata.get(
                    "agent_observation_layout",
                    self.layout,
                ),
                "information_steps": self.trial_metadata.get("information_steps", self.num_steps),
                "viewing_distance": self.viewing_distance_info,
                "gaze_quality": self.gaze_quality_info,
            }
        )
        if "onion_cook_ticks_per_env_step" in self.trial_metadata:
            trial_metadata["onion_cook_ticks_per_env_step"] = int(
                self.trial_metadata["onion_cook_ticks_per_env_step"]
            )
        if ai_checkpoint is not None or "ai_checkpoint" not in trial_metadata:
            trial_metadata["ai_checkpoint"] = (
                None if ai_checkpoint is None else os.path.abspath(os.fspath(ai_checkpoint))
            )
        if path_predictor_checkpoint is not None or "path_predictor_checkpoint" not in trial_metadata:
            trial_metadata["path_predictor_checkpoint"] = (
                None
                if path_predictor_checkpoint is None
                else os.path.abspath(os.fspath(path_predictor_checkpoint))
            )
        if calibration is not None or "calibration" not in trial_metadata:
            trial_metadata["calibration"] = calibration or serialize_calibration_info(None)

        temporary_path = f"{metadata_path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2, ensure_ascii=False)
        os.replace(temporary_path, metadata_path)

    def mark_trial_started(self) -> None:
        if not self.enabled or self._trial_start_perf_counter is not None:
            return
        self._trial_start_time = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self._trial_start_perf_counter = time.perf_counter()

    def record_render_frame(self) -> None:
        if self.enabled and self._trial_start_perf_counter is not None:
            self._render_frame_count += 1

    def record_environment_step(self) -> None:
        if self.enabled and self._trial_start_perf_counter is not None:
            self._environment_step_count += 1

    def _write_row(self, writer: Optional[csv.DictWriter], file_obj: Any, row: Dict[str, Any]) -> None:
        if not self.enabled or writer is None or file_obj is None:
            return
        writer.writerow(row)
        file_obj.flush()

    def _summarize_future_trajectory(self, rollout_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        overlay_entries = self._get_overlay_entries(rollout_results)
        normalized_entries = []
        for entry in overlay_entries:
            normalized_entry = dict(entry)
            normalized_entry["overlay_type"] = _normalize_logged_overlay_type(entry.get("overlay_type"))
            normalized_entries.append(normalized_entry)
        return {
            "ai_start_position": _normalize_position(rollout_results.get("ai_start_position")) if rollout_results else None,
            "ai_start_orientation": _serialize_action(rollout_results.get("ai_start_orientation")) if rollout_results else None,
            "entries": normalized_entries,
        }

    def _get_overlay_entries(self, rollout_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if rollout_results is None:
            return []
        if "visible_overlay_entries" in rollout_results:
            return rollout_results["visible_overlay_entries"]
        return build_visible_overlay_entries(
            rollout_results,
            self.visualization_mode,
            self.num_steps,
            self.stop_path_at_interaction,
            max_ranked_paths=self.num_paths_shown,
        )

    def _gaze_eccentricity_fields(
        self,
        *,
        surface_point: Optional[Tuple[float, float]],
        surface_size: Optional[Tuple[int, int]],
        map_display_size: Optional[Tuple[int, int]],
        display_size: Optional[Tuple[int, int]],
        terrain_mtx: Optional[Sequence[Sequence[str]]],
        rollout_results: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        fields = {field_name: "" for field_name in RAW_GAZE_ECCENTRICITY_FIELDS}
        if (
            surface_point is None
            or surface_size is None
            or map_display_size is None
            or display_size is None
            or not terrain_mtx
            or not terrain_mtx[0]
        ):
            return fields

        row_count = len(terrain_mtx)
        column_count = len(terrain_mtx[0])
        eccentricities_by_horizon: Dict[int, List[float]] = {}
        for entry in self._get_overlay_entries(rollout_results):
            horizon = int(entry["overlay_horizon"])
            if horizon < 1 or horizon > 4:
                continue
            tile_x, tile_y = entry["tile"]
            target_surface_point = (
                (float(tile_x) + 0.5) * surface_size[0] / column_count,
                (float(tile_y) + 0.5) * surface_size[1] / row_count,
            )
            eccentricity = visual_eccentricity_deg(
                surface_point,
                target_surface_point,
                surface_size,
                map_display_size,
                display_size,
                self.ivt_config,
            )
            eccentricities_by_horizon.setdefault(horizon, []).append(eccentricity)

        for horizon, eccentricities in eccentricities_by_horizon.items():
            minimum_eccentricity = min(eccentricities)
            fields[f"h{horizon}_eccentricity_deg"] = minimum_eccentricity
            fields[f"h{horizon}_vision_zone"] = classify_vision_zone(
                minimum_eccentricity,
                self.ivt_config,
            )
        return fields

    def log_step(
        self,
        *,
        state: Any,
        mdp: Any,
        step: int,
        state_onset_ms: int,
        human_action: Any,
        agent_action: Any,
        joint_action: Sequence[Any],
        human_player_index: int,
        agent_player_index: int,
        is_collision: bool,
        human_action_due: bool,
        ai_action_due: bool,
        is_human_idle: bool,
        accumulated_score: float,
        reward: float,
        response_time_ms: Optional[int],
        human_input_ms: Optional[int],
        first_human_action: Optional[Any],
        first_human_input_ms: Optional[int],
        final_human_action: Optional[Any],
        final_human_input_ms: Optional[int],
        human_input_changed: bool,
        response_input_source: Optional[str],
        action_execution_ms: int,
        done: bool,
        rollout_results: Optional[Dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return

        accumulated_score = float(accumulated_score)
        self._score_auc += accumulated_score
        self._score_auc_sample_count += 1

        human_player = state.players[human_player_index]
        agent_player = state.players[agent_player_index]
        row = {
            "pid": self.pid,
            "trial_id": self.trial_id,
            "step": int(step),
            "state_onset_ms": int(state_onset_ms),
            "human_action": _serialize_action(human_action),
            "agent_action": _serialize_action(agent_action),
            "joint_action": _json_dumps([_serialize_action(action) for action in joint_action]),
            "env_state": _json_dumps(state.to_dict()),
            "human_position": _json_dumps(_normalize_position(human_player.position)),
            "agent_position": _json_dumps(_normalize_position(agent_player.position)),
            "human_held_object": _json_dumps(_serialize_held_object(human_player)),
            "agent_held_object": _json_dumps(_serialize_held_object(agent_player)),
            "pot_states": _json_dumps(_simplify_pot_states(mdp, state)),
            "is_collision": int(bool(is_collision)),
            "human_action_due": int(bool(human_action_due)),
            "ai_action_due": int(bool(ai_action_due)),
            "is_human_idle": int(bool(is_human_idle)),
            "accumulated_score": accumulated_score,
            "reward": float(reward),
            "response_time_ms": "" if response_time_ms is None else int(response_time_ms),
            "human_input_ms": "" if human_input_ms is None else int(human_input_ms),
            "future_trajectory": _json_dumps(self._summarize_future_trajectory(rollout_results)),
            "action_execution_ms": int(action_execution_ms),
            "done": int(bool(done)),
            "first_human_action": (
                "" if first_human_action is None else _serialize_action(first_human_action)
            ),
            "first_human_input_ms": (
                "" if first_human_input_ms is None else int(first_human_input_ms)
            ),
            "final_human_action": (
                "" if final_human_action is None else _serialize_action(final_human_action)
            ),
            "final_human_input_ms": (
                "" if final_human_input_ms is None else int(final_human_input_ms)
            ),
            "human_input_changed": int(bool(human_input_changed)),
            "response_input_source": response_input_source or "",
        }
        self._write_row(self._step_writer, self._step_file, row)

    def record_path_prediction(
        self,
        *,
        prediction_step: int,
        rollout_results: Optional[Dict[str, Any]],
    ) -> None:
        if not self.enabled or not rollout_results:
            return
        paths = rollout_results.get("ai_paths") or []
        if not paths:
            return

        normalized_path = _normalize_visualized_path(
            paths[0],
            max_steps=min(4, self.num_steps),
            stop_at_interaction=self.stop_path_at_interaction,
        )
        greedy_action_indices = rollout_results.get("ai_greedy_action_indices") or []
        for horizon, (position, interact, interact_position) in enumerate(normalized_path, start=1):
            predicted_action = None
            if horizon <= len(greedy_action_indices):
                predicted_action = Action.INDEX_TO_ACTION[int(greedy_action_indices[horizon - 1])]
            target_step = int(prediction_step) + horizon - 1
            self._pending_path_predictions.setdefault(target_step, []).append(
                {
                    "prediction_step": int(prediction_step),
                    "horizon": horizon,
                    "predicted_position": position,
                    "predicted_action": predicted_action,
                    "predicted_interact": bool(interact),
                    "predicted_interact_position": interact_position,
                }
            )

    def log_path_prediction_accuracy(
        self,
        *,
        actual_step: int,
        actual_player: Any,
        actual_action: Any,
    ) -> None:
        if not self.enabled:
            return
        predictions = self._pending_path_predictions.pop(int(actual_step), [])
        if not predictions:
            return

        actual_position = tuple(int(component) for component in actual_player.position)
        actual_interact = actual_action == Action.INTERACT
        actual_interact_position = None
        if actual_interact:
            actual_interact_position = tuple(
                int(component)
                for component in Action.move_in_direction(actual_position, actual_player.orientation)
            )

        for prediction in predictions:
            predicted_position = prediction["predicted_position"]
            predicted_interact = prediction["predicted_interact"]
            predicted_interact_position = prediction["predicted_interact_position"]
            predicted_action = prediction["predicted_action"]
            position_correct = predicted_position == actual_position
            interact_correct = predicted_interact == actual_interact
            interact_target_correct = None
            if predicted_interact and actual_interact:
                interact_target_correct = predicted_interact_position == actual_interact_position
            path_correct = (
                position_correct
                and interact_correct
                and (not actual_interact or bool(interact_target_correct))
            )
            action_correct = None if predicted_action is None else predicted_action == actual_action

            self._write_row(
                self._prediction_accuracy_writer,
                self._prediction_accuracy_file,
                {
                    "pid": self.pid,
                    "trial_id": self.trial_id,
                    "prediction_step": prediction["prediction_step"],
                    "actual_step": int(actual_step),
                    "horizon": prediction["horizon"],
                    "predicted_position": _json_dumps(_normalize_position(predicted_position)),
                    "actual_position": _json_dumps(_normalize_position(actual_position)),
                    "position_correct": int(position_correct),
                    "predicted_action": "" if predicted_action is None else _serialize_action(predicted_action),
                    "actual_action": _serialize_action(actual_action),
                    "action_correct": "" if action_correct is None else int(action_correct),
                    "predicted_interact": int(predicted_interact),
                    "actual_interact": int(actual_interact),
                    "interact_correct": int(interact_correct),
                    "predicted_interact_position": _json_dumps(
                        _normalize_position(predicted_interact_position)
                    ),
                    "actual_interact_position": _json_dumps(
                        _normalize_position(actual_interact_position)
                    ),
                    "interact_target_correct": (
                        "" if interact_target_correct is None else int(interact_target_correct)
                    ),
                    "path_correct": int(path_correct),
                },
            )

            metrics = self._path_prediction_metrics[prediction["horizon"]]
            metrics["count"] += 1
            metrics["path_correct"] += int(path_correct)
            metrics["position_correct"] += int(position_correct)
            if action_correct is not None:
                metrics["action_count"] += 1
                metrics["action_correct"] += int(action_correct)

    def _surface_point_to_tile(
        self,
        surface_point: Optional[Tuple[float, float]],
        surface_size: Optional[Tuple[int, int]],
        terrain_mtx: Optional[Sequence[Sequence[str]]],
    ) -> Optional[Tuple[int, int]]:
        if surface_point is None or surface_size is None or terrain_mtx is None:
            return None
        if surface_size[0] <= 0 or surface_size[1] <= 0 or not terrain_mtx or not terrain_mtx[0]:
            return None
        tile_x = int(surface_point[0] * len(terrain_mtx[0]) / float(surface_size[0]))
        tile_y = int(surface_point[1] * len(terrain_mtx) / float(surface_size[1]))
        if tile_x < 0 or tile_y < 0:
            return None
        if tile_y >= len(terrain_mtx) or tile_x >= len(terrain_mtx[tile_y]):
            return None
        return tile_x, tile_y

    def _classify_overlay(
        self,
        tile: Optional[Tuple[int, int]],
        rollout_results: Optional[Dict[str, Any]],
        visualization_mode: str,
        num_steps: int,
        stop_path_at_interaction: bool,
        underlying_aoi: str,
    ) -> Dict[str, Any]:
        if tile is None:
            return {
                "overlay_flag": 0,
                "overlay_type": "none",
                "overlay_horizon": None,
                "path_rank": None,
                "overlay_underlying_aoi": None,
                "overlay_position": None,
                "overlay_matches": [],
            }

        overlay_entries = self._get_overlay_entries(rollout_results)
        matching_entries = [entry for entry in overlay_entries if tuple(entry["tile"]) == tile]
        if not matching_entries:
            return {
                "overlay_flag": 0,
                "overlay_type": "none",
                "overlay_horizon": None,
                "path_rank": None,
                "overlay_underlying_aoi": None,
                "overlay_position": None,
                "overlay_matches": [],
            }

        matching_entries.sort(
            key=lambda entry: (
                0 if entry["overlay_type"] == "marker" else 1,
                int(entry["path_rank"]),
                int(entry["overlay_horizon"]),
            )
        )
        selected = matching_entries[0]
        overlay_matches = [
            {
                "overlay_type": _normalize_logged_overlay_type(entry["overlay_type"]),
                "overlay_horizon": int(entry["overlay_horizon"]),
                "path_rank": int(entry["path_rank"]),
                "overlay_underlying_aoi": underlying_aoi,
                "overlay_position": entry["overlay_position"],
            }
            for entry in matching_entries
        ]
        return {
            "overlay_flag": 1,
            "overlay_type": _normalize_logged_overlay_type(selected["overlay_type"]),
            "overlay_horizon": int(selected["overlay_horizon"]),
            "path_rank": int(selected["path_rank"]),
            "overlay_underlying_aoi": underlying_aoi,
            "overlay_position": selected["overlay_position"],
            "overlay_matches": overlay_matches,
        }

    def _classify_sample(
        self,
        *,
        step: int,
        timestamp_ms: int,
        state_onset_ms: int,
        surface_point: Optional[Tuple[float, float]],
        surface_size: Optional[Tuple[int, int]],
        terrain_mtx: Optional[Sequence[Sequence[str]]],
        state: Any,
        self_player_index: int,
        partner_player_index: int,
        rollout_results: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tile = self._surface_point_to_tile(surface_point, surface_size, terrain_mtx)
        base_aoi, base_position = _classify_base_aoi(
            tile,
            terrain_mtx,
            state,
            self_player_index,
            partner_player_index,
        )
        overlay_info = self._classify_overlay(
            tile,
            rollout_results,
            self.visualization_mode,
            self.num_steps,
            self.stop_path_at_interaction,
            base_aoi,
        )

        counter_object_present = 0
        counter_object_type = None
        if base_aoi == "Counter" and tile is not None and state is not None and state.has_object(tile):
            counter_object_present = 1
            counter_object_type = state.get_object(tile).name

        return {
            "step": int(step),
            "timestamp_ms": int(timestamp_ms),
            "current_state_onset_ms": int(state_onset_ms),
            "base_aoi": base_aoi,
            "base_position": base_position,
            "overlay_flag": int(overlay_info["overlay_flag"]),
            "overlay_type": overlay_info["overlay_type"],
            "overlay_horizon": overlay_info["overlay_horizon"],
            "path_rank": overlay_info["path_rank"],
            "overlay_underlying_aoi": overlay_info["overlay_underlying_aoi"],
            "overlay_position": overlay_info["overlay_position"],
            "overlay_matches": overlay_info["overlay_matches"],
            "counter_object_present": counter_object_present,
            "counter_object_type": counter_object_type,
        }

    def log_raw_gaze(
        self,
        *,
        step: Optional[int],
        current_state_onset_ms: Optional[int],
        timestamp_ms: int,
        gaze_x: Optional[float],
        gaze_y: Optional[float],
        gaze_validity: bool,
        surface_point: Optional[Tuple[float, float]],
        surface_size: Optional[Tuple[int, int]],
        map_display_size: Optional[Tuple[int, int]],
        display_size: Optional[Tuple[int, int]],
        terrain_mtx: Optional[Sequence[Sequence[str]]],
        state: Any,
        self_player_index: int,
        partner_player_index: int,
        rollout_results: Optional[Dict[str, Any]],
        classify_aoi: bool = True,
        callback_perf_counter: Optional[float] = None,
        tobii_device_timestamp_us: Optional[int] = None,
        tobii_system_timestamp_us: Optional[int] = None,
        gaze_point_validities: Optional[Dict[str, Optional[bool]]] = None,
        gaze_origins: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not self.enabled:
            return

        eccentricity_fields = self._gaze_eccentricity_fields(
            surface_point=surface_point if gaze_validity else None,
            surface_size=surface_size,
            map_display_size=map_display_size,
            display_size=display_size,
            terrain_mtx=terrain_mtx,
            rollout_results=rollout_results if classify_aoi else None,
        )
        gaze_point_validities = gaze_point_validities or {}
        gaze_origins = gaze_origins or {}
        row = {
            "pid": self.pid,
            "trial_id": self.trial_id,
            "step": "" if step is None else int(step),
            "current_state_onset_ms": "" if current_state_onset_ms is None else int(current_state_onset_ms),
            "timestamp_ms": int(timestamp_ms),
            "gaze_x": "" if gaze_x is None else float(gaze_x),
            "gaze_y": "" if gaze_y is None else float(gaze_y),
            "gaze_validity": int(bool(gaze_validity)),
            "callback_perf_counter": (
                "" if callback_perf_counter is None else float(callback_perf_counter)
            ),
            "tobii_device_timestamp_us": (
                "" if tobii_device_timestamp_us is None else int(tobii_device_timestamp_us)
            ),
            "tobii_system_timestamp_us": (
                "" if tobii_system_timestamp_us is None else int(tobii_system_timestamp_us)
            ),
            "gaze_surface_x": "" if surface_point is None else float(surface_point[0]),
            "gaze_surface_y": "" if surface_point is None else float(surface_point[1]),
            "surface_width_px": "" if surface_size is None else int(surface_size[0]),
            "surface_height_px": "" if surface_size is None else int(surface_size[1]),
            "map_display_width_px": "" if map_display_size is None else int(map_display_size[0]),
            "map_display_height_px": "" if map_display_size is None else int(map_display_size[1]),
            "display_width_px": "" if display_size is None else int(display_size[0]),
            "display_height_px": "" if display_size is None else int(display_size[1]),
            **eccentricity_fields,
        }
        for eye_prefix in ("left", "right"):
            point_validity = gaze_point_validities.get(eye_prefix)
            row[f"{eye_prefix}_gaze_point_validity"] = (
                "" if point_validity is None else int(bool(point_validity))
            )
            if point_validity is not None:
                self._per_eye_gaze_point_counts[eye_prefix]["total"] += 1
                self._per_eye_gaze_point_counts[eye_prefix]["valid"] += int(
                    bool(point_validity)
                )
            origin = gaze_origins.get(eye_prefix, {})
            point = origin.get("point")
            row[f"{eye_prefix}_gaze_origin_validity"] = int(bool(origin.get("valid", False)))
            for coordinate_index, coordinate_name in enumerate(("x", "y", "z")):
                row[f"{eye_prefix}_gaze_origin_{coordinate_name}_mm"] = (
                    "" if point is None else float(point[coordinate_index])
                )
        self._gaze_sample_count += 1
        self._valid_gaze_sample_count += int(bool(gaze_validity))
        # raw gaze는 invalid여도 저장한다.
        self._write_row(self._raw_gaze_writer, self._raw_gaze_file, row)

        ivt_sample = {
            "timestamp_ms": int(timestamp_ms),
            "x": gaze_x,
            "y": gaze_y,
            "valid": bool(gaze_validity and gaze_x is not None and gaze_y is not None),
        }
        if (
            not ivt_sample["valid"]
            or not classify_aoi
            or step is None
            or current_state_onset_ms is None
        ):
            self._raw_fixation_samples.append(ivt_sample)
            return

        sample = self._classify_sample(
            step=step,
            timestamp_ms=timestamp_ms,
            state_onset_ms=current_state_onset_ms,
            surface_point=surface_point,
            surface_size=surface_size,
            terrain_mtx=terrain_mtx,
            state=state,
            self_player_index=self_player_index,
            partner_player_index=partner_player_index,
            rollout_results=rollout_results,
        )
        ivt_sample.update(sample)
        self._raw_fixation_samples.append(ivt_sample)

    def log_trial_end(
        self,
        *,
        score_final: float,
        num_collisions: int,
        num_human_idle: int,
        decision_steps_completed: int,
        completed: bool,
        calibration_rmse: Optional[float],
    ) -> None:
        if not self.enabled:
            return

        trial_end_time = datetime.now().astimezone().isoformat(timespec="milliseconds")
        trial_duration_ms = None
        if self._trial_start_perf_counter is not None:
            trial_duration_ms = round((time.perf_counter() - self._trial_start_perf_counter) * 1000.0, 3)
        active_seconds = (
            None
            if trial_duration_ms is None or trial_duration_ms <= 0
            else trial_duration_ms / 1000.0
        )

        row = {
            "pid": self.pid,
            "trial_id": self.trial_id,
            "condition": self.condition,
            "time_constraint_ms": self.time_constraint_ms,
            "decision_horizon": self.decision_horizon,
            "decision_steps_completed": int(decision_steps_completed),
            "visualization_mode": self.visualization_mode,
            "num_paths_shown": self.num_paths_shown,
            "layout": self.layout,
            "score_final": float(score_final),
            "score_auc": self._score_auc,
            "mean_cumulative_score": (
                ""
                if self._score_auc_sample_count == 0
                else self._score_auc / self._score_auc_sample_count
            ),
            "target_game_fps": (
                "" if self._target_game_fps is None else float(self._target_game_fps)
            ),
            "render_frame_count": self._render_frame_count,
            "environment_step_count": self._environment_step_count,
            "mean_render_fps": (
                "" if active_seconds is None else self._render_frame_count / active_seconds
            ),
            "mean_environment_steps_per_second": (
                "" if active_seconds is None else self._environment_step_count / active_seconds
            ),
            "num_collisions": int(num_collisions),
            "num_human_idle": int(num_human_idle),
            "screen_width_px": "" if self._screen_width_px is None else int(self._screen_width_px),
            "screen_height_px": "" if self._screen_height_px is None else int(self._screen_height_px),
            "mean_gaze_validity": (
                ""
                if self._gaze_sample_count == 0
                else self._valid_gaze_sample_count / self._gaze_sample_count
            ),
            "participant_index": self.trial_metadata.get("participant_index", ""),
            "global_trial_index": self.trial_metadata.get("global_trial_index", ""),
            "block_index": self.trial_metadata.get("block_index", ""),
            "block_trial_index": self.trial_metadata.get("block_trial_index", ""),
            "condition_id": self.trial_metadata.get("condition_id", ""),
            "information_steps": self.trial_metadata.get("information_steps", self.num_steps),
            "layout_code": self.trial_metadata.get("layout_code", ""),
            "williams_sequence_index": self.trial_metadata.get("williams_sequence_index", ""),
            "layout_sequence_id": self.trial_metadata.get("layout_sequence_id", ""),
            "layout_assignment_id": self.trial_metadata.get("layout_assignment_id", ""),
            "trial_start_time": self._trial_start_time or "",
            "trial_end_time": trial_end_time,
            "trial_duration_ms": "" if trial_duration_ms is None else trial_duration_ms,
            "completed": int(bool(completed)),
            "total_gaze_samples": self._gaze_sample_count,
            "valid_gaze_samples": self._valid_gaze_sample_count,
            "invalid_gaze_samples": self._gaze_sample_count - self._valid_gaze_sample_count,
            "calibration_rmse": "" if calibration_rmse is None else float(calibration_rmse),
            "ivt_screen_diagonal_inches": self.ivt_config.screen_diagonal_inches,
            "ivt_screen_aspect_ratio": self.ivt_config.aspect_ratio_label,
            "ivt_viewing_distance_mm": self.ivt_config.viewing_distance_mm,
            "ivt_median_window_samples": self.ivt_config.median_window_samples,
            "ivt_velocity_window_ms": self.ivt_config.velocity_window_ms,
            "ivt_velocity_threshold_deg_s": self.ivt_config.velocity_threshold_deg_s,
            "ivt_merge_gap_ms": self.ivt_config.merge_gap_ms,
            "ivt_merge_angle_deg": self.ivt_config.merge_angle_deg,
            "ivt_min_fixation_duration_ms": self.ivt_config.min_fixation_duration_ms,
            "ivt_foveal_threshold_deg": self.ivt_config.foveal_threshold_deg,
            "ivt_parafoveal_threshold_deg": self.ivt_config.parafoveal_threshold_deg,
            "ivt_viewing_distance_source": self.viewing_distance_info["source"],
            "ivt_viewing_distance_sample_count": self.viewing_distance_info["sample_count"],
            "ivt_viewing_distance_mean_mm": (
                ""
                if self.viewing_distance_info["mean_mm"] is None
                else self.viewing_distance_info["mean_mm"]
            ),
            "ivt_viewing_distance_median_mm": (
                ""
                if self.viewing_distance_info["median_mm"] is None
                else self.viewing_distance_info["median_mm"]
            ),
            "ivt_viewing_distance_std_mm": (
                ""
                if self.viewing_distance_info["std_mm"] is None
                else self.viewing_distance_info["std_mm"]
            ),
            "ivt_viewing_distance_window_sec": (
                ""
                if self.viewing_distance_info["window_sec"] is None
                else self.viewing_distance_info["window_sec"]
            ),
            "calibration_point_count": self.gaze_quality_info["calibration_point_count"],
            "calibration_quality_status": self.gaze_quality_info["calibration_quality_status"],
            "calibration_quality_passed": int(
                bool(self.gaze_quality_info["calibration_quality_passed"])
            ),
            "drift_check_status": self.gaze_quality_info["drift_check_status"],
            "drift_check_angle_deg": (
                ""
                if self.gaze_quality_info["drift_check_angle_deg"] is None
                else self.gaze_quality_info["drift_check_angle_deg"]
            ),
            "drift_check_sample_count": self.gaze_quality_info["drift_check_sample_count"],
            "drift_check_passed": int(bool(self.gaze_quality_info["drift_check_passed"])),
            "preview_sample_count": self.gaze_quality_info["preview_sample_count"],
            "preview_valid": int(bool(self.gaze_quality_info["preview_valid"])),
            "preview_accepted_ms": (
                ""
                if self.gaze_quality_info["preview_accepted_ms"] is None
                else self.gaze_quality_info["preview_accepted_ms"]
            ),
        }
        for eye_prefix, counts in self._per_eye_gaze_point_counts.items():
            row[f"{eye_prefix}_gaze_point_sample_count"] = counts["total"]
            row[f"{eye_prefix}_valid_gaze_point_count"] = counts["valid"]
            row[f"mean_{eye_prefix}_gaze_point_validity"] = (
                "" if counts["total"] == 0 else counts["valid"] / counts["total"]
            )
        for horizon, metrics in self._path_prediction_metrics.items():
            count = metrics["count"]
            action_count = metrics["action_count"]
            row[f"prediction_samples@{horizon}"] = count
            row[f"path_acc@{horizon}"] = "" if count == 0 else metrics["path_correct"] / count
            row[f"position_acc@{horizon}"] = "" if count == 0 else metrics["position_correct"] / count
            row[f"action_acc@{horizon}"] = (
                "" if action_count == 0 else metrics["action_correct"] / action_count
            )
        self._write_row(self._trial_writer, self._trial_file, row)

    def _flush_fixations(self) -> None:
        if not self.enabled or not self._raw_fixation_samples:
            return
        rows = build_ivt_fixation_rows(
            self._raw_fixation_samples,
            pid=self.pid,
            trial_id=self.trial_id,
            config=self.ivt_config,
        )
        for row in rows:
            self._write_row(self._fixation_writer, self._fixation_file, row)

    def close(self) -> None:
        if not self.enabled:
            return

        self._flush_fixations()
        for file_obj in (
            self._trial_file,
            self._step_file,
            self._raw_gaze_file,
            self._fixation_file,
            self._prediction_accuracy_file,
        ):
            if file_obj is not None and not file_obj.closed:
                file_obj.close()
