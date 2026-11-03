"""Build decision-level model datasets without modifying source experiment logs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LOGGER = logging.getLogger(__name__)

WINDOW_FULL = "full_window"
WINDOW_PRE_ACTION = "pre_action_window"
DEFAULT_UNKNOWN_AOI_POLICY = "outside"
PATH_HORIZONS = (1, 2, 3, 4)
PATH_RANKS = (1, 2, 3, 4)
VISION_ZONES = ("foveal", "parafoveal", "peripheral")

BASE_AOI_MAPPING = {
    "self": "Self",
    "partner": "Partner",
    "counter": "Counter",
    "corridor": "Corridor",
    "cooking": "Cooking",
    "cooking_area": "Cooking",
    "serving": "Serving",
    "serving_area": "Serving",
    "outside": "Outside",
}

SEGMENT_FIELDS = [
    "pid",
    "trial_id",
    "step",
    "fixation_id",
    "window_type",
    "decision_start_ms",
    "decision_end_ms",
    "fixation_start_ms",
    "fixation_end_ms",
    "overlap_start_ms",
    "overlap_end_ms",
    "overlap_duration_ms",
    "base_aoi",
    "path_aoi",
    "overlay_flag",
    "overlay_horizon",
    "base_position",
    "overlay_type",
    "path_rank",
    "overlay_underlying_aoi",
    "overlay_position",
    "overlay_matches",
    "counter_object_present",
    "counter_object_type",
    "aoi_matches",
    "continued_from_previous_step",
    "continues_to_next_step",
]

DECISION_FIELDS = [
    "pid",
    "trial_id",
    "step",
    "time_condition",
    "information_steps",
    "layout",
    "state_onset_ms",
    "human_input_ms",
    "first_human_action",
    "first_human_input_ms",
    "final_human_action",
    "final_human_input_ms",
    "human_input_changed",
    "response_input_source",
    "response_time_ms",
    "action_execution_ms",
    "pre_action_gaze_sample_count",
    "pre_action_valid_gaze_count",
    "pre_action_gaze_validity",
    "full_window_gaze_sample_count",
    "full_window_valid_gaze_count",
    "full_window_gaze_validity",
    "longest_invalid_gap_ms",
    *[
        f"{window}_{eye_prefix}_{metric}"
        for window in ("pre_action", "full_window")
        for eye_prefix in ("left", "right")
        for metric in (
            "gaze_sample_count",
            "valid_gaze_count",
            "gaze_validity",
        )
    ],
    "env_state",
    "previous_human_action",
    "displayed_future_trajectory",
    "pre_action_aoi_sequence",
    "pre_action_aoi_durations",
    "full_window_aoi_sequence",
    "full_window_aoi_durations",
    "first_aoi",
    "last_pre_action_aoi",
    "num_fixations",
    "num_aoi_transitions",
    "pre_action_fixation_details",
    "full_window_fixation_details",
    *[
        f"{window}_h{horizon}_dwell_ms"
        for window in ("pre_action", "full_window")
        for horizon in PATH_HORIZONS
    ],
    *[
        f"{window}_rank{rank}_dwell_ms"
        for window in ("pre_action", "full_window")
        for rank in PATH_RANKS
    ],
    *[
        f"{window}_h{horizon}_{zone}_ms"
        for window in ("pre_action", "full_window")
        for horizon in PATH_HORIZONS
        for zone in VISION_ZONES
    ],
    "human_action",
    "agent_action",
    "is_human_idle",
    "is_collision",
    "reward",
    "done",
]

QC_FIELDS = [
    "pid",
    "completed_trial_count",
    "total_decision_count",
    "done_error_count",
    "timestamp_order_error_count",
    "gaze_validity",
    "left_gaze_point_validity",
    "right_gaze_point_validity",
    "decision_without_fixation_ratio",
    "missing_future_trajectory_count",
    "information_condition_mismatch_count",
    "stay_ratio",
    "action_up_count",
    "action_down_count",
    "action_left_count",
    "action_right_count",
    "action_interact_count",
    "action_stay_count",
]

CONDITION_COUNT_FIELDS = [
    "pid",
    "time_condition",
    "information_steps",
    "layout",
    "trial_count",
    "decision_count",
]

ISSUE_FIELDS = ["severity", "code", "pid", "trial_id", "step", "message"]


@dataclass
class IssueCollector:
    """Collect processing warnings and errors while also emitting console logs."""

    issues: List[Dict[str, Any]] = field(default_factory=list)
    _logged_by_code: Dict[str, int] = field(default_factory=dict)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        pid: str = "",
        trial_id: str = "",
        step: Any = "",
    ) -> None:
        row = {
            "severity": severity,
            "code": code,
            "pid": pid,
            "trial_id": trial_id,
            "step": step,
            "message": message,
        }
        self.issues.append(row)
        logged_count = self._logged_by_code.get(code, 0)
        self._logged_by_code[code] = logged_count + 1
        if logged_count >= 5:
            if logged_count == 5:
                LOGGER.warning("%s: further messages are recorded only in processing_issues.csv", code)
            return
        log_method = LOGGER.error if severity == "error" else LOGGER.warning
        log_method("%s: %s", code, message)


@dataclass
class Decision:
    """Parsed decision row and its two gaze windows."""

    pid: str
    trial_id: str
    step: int
    start_ms: int
    action_execution_ms: Optional[int]
    human_input_ms: Optional[int]
    first_human_action: str
    first_human_input_ms: Optional[int]
    final_human_action: str
    final_human_input_ms: Optional[int]
    human_input_changed: bool
    response_input_source: str
    response_time_ms: Optional[int]
    pre_action_end_ms: Optional[int]
    time_condition: Optional[int]
    information_steps: Optional[int]
    layout: str
    visualization_mode: str
    env_state_json: str
    future_trajectory_json: str
    future_trajectory_value: Any
    human_action: str
    agent_action: str
    is_human_idle: bool
    is_collision: bool
    reward: Optional[float]
    done: Optional[bool]
    source_row: Dict[str, str]
    timestamp_error: bool = False
    information_mismatch: bool = False


@dataclass
class Fixation:
    """Fixation interval with normalized spatial and path AOIs."""

    pid: str
    trial_id: str
    fixation_id: str
    start_ms: int
    end_ms: int
    base_aoi: str
    path_aoi: str
    overlay_flag: bool
    overlay_horizon: Optional[int]
    base_position: Any
    overlay_type: str
    path_rank: Optional[int]
    overlay_underlying_aoi: str
    overlay_position: Any
    overlay_matches: List[Dict[str, Any]]
    counter_object_present: bool
    counter_object_type: str
    aoi_matches: List[Dict[str, Any]]
    context_segments: List[Dict[str, Any]]


@dataclass
class BuildResult:
    """Summary and output paths returned by :func:`build_model_dataset`."""

    participant_count: int
    trial_count: int
    decision_count: int
    fixation_segment_count: int
    timestamp_error_count: int
    decisions_without_fixation: int
    qc_warning_count: int
    output_paths: Dict[str, Path]


def interval_overlap(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> Optional[Tuple[int, int, int]]:
    """Return the positive overlap of two half-open intervals, or ``None``."""

    overlap_start = max(first_start, second_start)
    overlap_end = min(first_end, second_end)
    overlap_duration = max(0, overlap_end - overlap_start)
    if overlap_duration <= 0:
        return None
    return overlap_start, overlap_end, overlap_duration


def normalize_base_aoi(
    value: Any,
    issues: IssueCollector,
    *,
    unknown_policy: str = DEFAULT_UNKNOWN_AOI_POLICY,
    pid: str = "",
    trial_id: str = "",
) -> str:
    """Normalize legacy AOI labels using an explicit mapping table."""

    normalized_key = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    if normalized_key in BASE_AOI_MAPPING:
        return BASE_AOI_MAPPING[normalized_key]

    message = f"Unknown base_aoi {value!r} in trial {trial_id or '<unknown>'}."
    issues.add("warning", "UNKNOWN_BASE_AOI", message, pid=pid, trial_id=trial_id)
    if unknown_policy == "error":
        raise ValueError(message)
    if unknown_policy == "unknown":
        return "Unknown"
    return "Outside"


def derive_path_aoi(
    overlay_flag: bool,
    overlay_horizon: Optional[int],
    issues: IssueCollector,
    *,
    pid: str = "",
    trial_id: str = "",
) -> str:
    """Map overlay fields to ``None``, ``Near Path``, or ``Far Path``."""

    if not overlay_flag or overlay_horizon is None:
        return "None"
    if overlay_horizon in (1, 2):
        return "Near Path"
    if overlay_horizon in (3, 4):
        return "Far Path"
    issues.add(
        "warning",
        "OVERLAY_HORIZON_OUT_OF_RANGE",
        f"overlay_horizon={overlay_horizon} is outside 1..4.",
        pid=pid,
        trial_id=trial_id,
    )
    return "None"


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "nan", "none", "null"}


def _optional_int(value: Any) -> Optional[int]:
    if _is_blank(value):
        return None
    return int(float(str(value).strip()))


def _optional_float(value: Any) -> Optional[float]:
    if _is_blank(value):
        return None
    return float(str(value).strip())


def _parse_bool(value: Any) -> Optional[bool]:
    if _is_blank(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _normalize_action(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "up": "up",
        "north": "up",
        "down": "down",
        "south": "down",
        "left": "left",
        "west": "left",
        "right": "right",
        "east": "right",
        "interact": "interact",
        "stay": "stay",
        "none": "stay",
        "": "stay",
    }
    return aliases.get(normalized, normalized)


def _json_array(values: Sequence[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _canonical_json(
    value: Any,
    issues: IssueCollector,
    *,
    field_name: str,
    pid: str,
    trial_id: str,
    step: Any,
) -> Tuple[str, Any]:
    if _is_blank(value):
        return "null", None
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        issues.add(
            "warning",
            "INVALID_JSON",
            f"Cannot parse {field_name}: {exc}",
            pid=pid,
            trial_id=trial_id,
            step=step,
        )
        return "null", None
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":")), parsed


def _parse_json_value(
    value: Any,
    default: Any,
    issues: IssueCollector,
    *,
    field_name: str,
    pid: str,
    trial_id: str,
) -> Any:
    if _is_blank(value):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        issues.add(
            "warning",
            "INVALID_JSON",
            f"Cannot parse fixation {field_name}: {exc}",
            pid=pid,
            trial_id=trial_id,
        )
        return default


def _trajectory_has_content(value: Any) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return False
    if isinstance(value, dict) and "entries" in value:
        return bool(value.get("entries"))
    return True


def _max_overlay_horizon(value: Any) -> Optional[int]:
    horizons: List[int] = []

    def visit(component: Any) -> None:
        if isinstance(component, dict):
            if "overlay_horizon" in component:
                try:
                    horizon = _optional_int(component.get("overlay_horizon"))
                except (TypeError, ValueError):
                    horizon = None
                if horizon is not None:
                    horizons.append(horizon)
            for child in component.values():
                visit(child)
        elif isinstance(component, list):
            for child in component:
                visit(child)

    visit(value)
    return max(horizons) if horizons else None


def _read_csv(path: Path, issues: IssueCollector) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(not _is_blank(value) for value in row.values() if value is not None):
                continue
            if None in row:
                issues.add(
                    "warning",
                    "MALFORMED_CSV_ROW",
                    f"{path.name}:{line_number} has extra columns.",
                )
            first_field = reader.fieldnames[0]
            if row.get(first_field) == first_field:
                continue
            rows.append({field_name: row.get(field_name, "") or "" for field_name in reader.fieldnames})
    return rows


def discover_input_file(input_dir: Path, kind: str, explicit_path: Optional[Path] = None) -> Optional[Path]:
    """Find a canonical or numbered input file without recursively scanning outputs."""

    if explicit_path is not None:
        resolved = explicit_path if explicit_path.is_absolute() else input_dir / explicit_path
        if not resolved.exists():
            raise FileNotFoundError(f"Explicit {kind} file not found: {resolved}")
        return resolved.resolve()

    extension = ".json" if kind == "metadata" else ".csv"
    exact = input_dir / f"{kind}{extension}"
    if exact.exists():
        return exact.resolve()

    pattern = re.compile(rf"^{re.escape(kind)}(?:\s*\(\d+\)|[ _-]\d+)?$", re.IGNORECASE)
    candidates = sorted(
        path.resolve()
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == extension and pattern.fullmatch(path.stem)
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple candidates found for {kind}: {candidates}. Specify --{kind.replace('_', '-')}-file."
        )
    return candidates[0]


def _build_contexts(
    trial_rows: Sequence[Mapping[str, str]],
    schedule_rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
    issues: IssueCollector,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    schedule_lookup: Dict[Tuple[str, int], Mapping[str, str]] = {}
    for row in schedule_rows:
        try:
            global_index = _optional_int(row.get("global_trial_index"))
        except (TypeError, ValueError):
            global_index = None
        if global_index is not None:
            schedule_lookup[(row.get("participant_id", ""), global_index)] = row

    metadata_trials = metadata.get("trials", {}) if isinstance(metadata, dict) else {}
    contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in trial_rows:
        pid = row.get("pid", "")
        trial_id = row.get("trial_id", "")
        try:
            global_index = _optional_int(row.get("global_trial_index"))
        except (TypeError, ValueError):
            global_index = None
        schedule = schedule_lookup.get((pid, global_index), {}) if global_index is not None else {}
        metadata_trial = metadata_trials.get(trial_id, {}) if isinstance(metadata_trials, dict) else {}

        source_info: Dict[str, int] = {}
        for source_name, value in (
            ("trial.csv", row.get("information_steps")),
            ("schedule.csv", schedule.get("information_steps")),
            ("metadata.json", metadata_trial.get("information_steps") if isinstance(metadata_trial, dict) else None),
        ):
            try:
                parsed = _optional_int(value)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None:
                source_info[source_name] = parsed
        source_values = set(source_info.values())
        source_mismatch = len(source_values) > 1
        if source_mismatch:
            issues.add(
                "warning",
                "INFORMATION_SOURCE_MISMATCH",
                f"information_steps disagree for {trial_id}: {source_info}",
                pid=pid,
                trial_id=trial_id,
            )

        try:
            time_condition = _optional_int(row.get("time_constraint_ms"))
        except (TypeError, ValueError):
            time_condition = None
        if time_condition is None:
            try:
                time_condition = _optional_int(schedule.get("time_ms"))
            except (TypeError, ValueError):
                time_condition = None

        contexts[(pid, trial_id)] = {
            "time_condition": time_condition,
            "information_steps": next(iter(source_info.values()), None),
            "layout": row.get("layout") or schedule.get("layout_name", ""),
            "visualization_mode": str(row.get("visualization_mode", "")).strip().lower(),
            "completed": row.get("completed"),
            "information_source_mismatch": source_mismatch,
        }
    return contexts


def _build_decisions(
    step_rows: Sequence[Dict[str, str]],
    contexts: Mapping[Tuple[str, str], Mapping[str, Any]],
    issues: IssueCollector,
) -> List[Decision]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in step_rows:
        grouped.setdefault((row.get("pid", ""), row.get("trial_id", "")), []).append(row)

    decisions: List[Decision] = []
    for (pid, trial_id), rows in grouped.items():
        try:
            rows.sort(key=lambda row: int(row.get("step", "")))
        except ValueError as exc:
            raise ValueError(f"Invalid step number in {pid}/{trial_id}") from exc
        context = contexts.get((pid, trial_id), {})
        trial_decisions: List[Decision] = []
        for index, row in enumerate(rows):
            step = int(row["step"])
            try:
                start_ms = _optional_int(row.get("state_onset_ms"))
            except (TypeError, ValueError):
                start_ms = None
            if start_ms is None:
                issues.add(
                    "error",
                    "MISSING_STATE_ONSET",
                    "state_onset_ms is missing or invalid.",
                    pid=pid,
                    trial_id=trial_id,
                    step=step,
                )
                continue

            try:
                execution_ms = _optional_int(row.get("action_execution_ms"))
            except (TypeError, ValueError):
                execution_ms = None
            if execution_ms is None:
                if index + 1 < len(rows):
                    try:
                        execution_ms = _optional_int(rows[index + 1].get("state_onset_ms"))
                    except (TypeError, ValueError):
                        execution_ms = None
                    if execution_ms is not None:
                        issues.add(
                            "warning",
                            "ACTION_EXECUTION_FALLBACK",
                            "action_execution_ms missing; used next state_onset_ms.",
                            pid=pid,
                            trial_id=trial_id,
                            step=step,
                        )
                if execution_ms is None:
                    issues.add(
                        "error",
                        "ACTION_EXECUTION_UNRECOVERABLE",
                        "action_execution_ms missing and no next state_onset_ms is available.",
                        pid=pid,
                        trial_id=trial_id,
                        step=step,
                    )

            try:
                human_input_ms = _optional_int(row.get("human_input_ms"))
            except (TypeError, ValueError):
                human_input_ms = None

            try:
                first_human_input_ms = _optional_int(row.get("first_human_input_ms"))
                final_human_input_ms = _optional_int(row.get("final_human_input_ms"))
            except (TypeError, ValueError):
                first_human_input_ms = final_human_input_ms = None
            first_human_action = (
                ""
                if _is_blank(row.get("first_human_action"))
                else _normalize_action(row.get("first_human_action"))
            )
            final_human_action = (
                ""
                if _is_blank(row.get("final_human_action"))
                else _normalize_action(row.get("final_human_action"))
            )
            # Legacy logs contain only the effective input timestamp/action.
            if first_human_input_ms is None and human_input_ms is not None:
                first_human_input_ms = human_input_ms
                first_human_action = _normalize_action(row.get("human_action"))
            if final_human_input_ms is None and human_input_ms is not None:
                final_human_input_ms = human_input_ms
                final_human_action = _normalize_action(row.get("human_action"))
            if first_human_input_ms is None:
                first_human_action = ""
            if final_human_input_ms is None:
                final_human_action = ""

            parsed_changed = _parse_bool(row.get("human_input_changed"))
            human_input_changed = bool(
                parsed_changed
                if parsed_changed is not None
                else (
                    first_human_input_ms is not None
                    and final_human_input_ms is not None
                    and first_human_action != final_human_action
                )
            )
            response_input_source = str(row.get("response_input_source") or "").strip()
            if not response_input_source and human_input_ms is not None:
                response_input_source = "final" if human_input_changed else "first"
            try:
                response_time_ms = _optional_int(row.get("response_time_ms"))
            except (TypeError, ValueError):
                response_time_ms = None

            timestamp_error = execution_ms is None or execution_ms < start_ms
            if execution_ms is not None and execution_ms < start_ms:
                issues.add(
                    "error",
                    "TIMESTAMP_ORDER",
                    f"state_onset_ms={start_ms} exceeds action_execution_ms={execution_ms}.",
                    pid=pid,
                    trial_id=trial_id,
                    step=step,
                )
            pre_action_end_ms = execution_ms
            if human_input_ms is not None:
                if execution_ms is None or human_input_ms < start_ms or human_input_ms > execution_ms:
                    timestamp_error = True
                    pre_action_end_ms = None
                    issues.add(
                        "error",
                        "TIMESTAMP_ORDER",
                        f"human_input_ms={human_input_ms} is outside the decision window.",
                        pid=pid,
                        trial_id=trial_id,
                        step=step,
                    )
                else:
                    pre_action_end_ms = human_input_ms

            for input_label, input_ms in (
                ("first_human_input_ms", first_human_input_ms),
                ("final_human_input_ms", final_human_input_ms),
            ):
                if input_ms is None:
                    continue
                if execution_ms is None or input_ms < start_ms or input_ms > execution_ms:
                    timestamp_error = True
                    issues.add(
                        "error",
                        "TIMESTAMP_ORDER",
                        f"{input_label}={input_ms} is outside the decision window.",
                        pid=pid,
                        trial_id=trial_id,
                        step=step,
                    )
            if (
                first_human_input_ms is not None
                and final_human_input_ms is not None
                and first_human_input_ms > final_human_input_ms
            ):
                timestamp_error = True
                issues.add(
                    "error",
                    "TIMESTAMP_ORDER",
                    "first_human_input_ms exceeds final_human_input_ms.",
                    pid=pid,
                    trial_id=trial_id,
                    step=step,
                )

            env_state_json, _ = _canonical_json(
                row.get("env_state"),
                issues,
                field_name="env_state",
                pid=pid,
                trial_id=trial_id,
                step=step,
            )
            future_json, future_value = _canonical_json(
                row.get("future_trajectory"),
                issues,
                field_name="future_trajectory",
                pid=pid,
                trial_id=trial_id,
                step=step,
            )
            visualization_mode = str(context.get("visualization_mode", "")).lower()
            actually_displayed = visualization_mode != "none" and _trajectory_has_content(future_value)
            if not actually_displayed:
                future_json = "null"
                future_value = None

            try:
                done = _parse_bool(row.get("done"))
            except ValueError:
                done = None
                issues.add(
                    "warning",
                    "INVALID_DONE",
                    f"Invalid done value {row.get('done')!r}.",
                    pid=pid,
                    trial_id=trial_id,
                    step=step,
                )

            decision = Decision(
                pid=pid,
                trial_id=trial_id,
                step=step,
                start_ms=start_ms,
                action_execution_ms=execution_ms,
                human_input_ms=human_input_ms,
                first_human_action=first_human_action,
                first_human_input_ms=first_human_input_ms,
                final_human_action=final_human_action,
                final_human_input_ms=final_human_input_ms,
                human_input_changed=human_input_changed,
                response_input_source=response_input_source,
                response_time_ms=response_time_ms,
                pre_action_end_ms=pre_action_end_ms,
                time_condition=context.get("time_condition"),
                information_steps=context.get("information_steps"),
                layout=str(context.get("layout", "")),
                visualization_mode=visualization_mode,
                env_state_json=env_state_json,
                future_trajectory_json=future_json,
                future_trajectory_value=future_value,
                human_action=_normalize_action(row.get("human_action")),
                agent_action=_normalize_action(row.get("agent_action")),
                is_human_idle=bool(_parse_bool(row.get("is_human_idle")) or False),
                is_collision=bool(_parse_bool(row.get("is_collision")) or False),
                reward=_optional_float(row.get("reward")),
                done=done,
                source_row=row,
                timestamp_error=timestamp_error,
                information_mismatch=bool(context.get("information_source_mismatch", False)),
            )
            decisions.append(decision)
            trial_decisions.append(decision)

        for current, following in zip(trial_decisions, trial_decisions[1:]):
            if current.action_execution_ms is not None and current.action_execution_ms > following.start_ms:
                current.timestamp_error = True
                issues.add(
                    "error",
                    "TIMESTAMP_ORDER",
                    f"action_execution_ms={current.action_execution_ms} exceeds next state_onset_ms={following.start_ms}.",
                    pid=pid,
                    trial_id=trial_id,
                    step=current.step,
                )

    return sorted(decisions, key=lambda item: (item.pid, item.trial_id, item.step))


def _build_fixations(
    rows: Sequence[Mapping[str, str]],
    issues: IssueCollector,
    unknown_aoi_policy: str,
) -> List[Fixation]:
    fixations: List[Fixation] = []
    for row in rows:
        if row.get("record_type") and row.get("record_type") != "ivt_fixation":
            continue
        pid = row.get("pid", "")
        trial_id = row.get("trial_id", "")
        try:
            start_ms = _optional_int(row.get("fixation_start_ms"))
            end_ms = _optional_int(row.get("fixation_end_ms"))
        except (TypeError, ValueError):
            start_ms = end_ms = None
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            issues.add(
                "warning",
                "INVALID_FIXATION_INTERVAL",
                f"Invalid fixation interval {start_ms}..{end_ms}.",
                pid=pid,
                trial_id=trial_id,
            )
            continue
        try:
            overlay_flag = bool(_parse_bool(row.get("overlay_flag")) or False)
        except ValueError:
            overlay_flag = False
        try:
            overlay_horizon = _optional_int(row.get("overlay_horizon"))
        except (TypeError, ValueError):
            overlay_horizon = None
        try:
            path_rank = _optional_int(row.get("path_rank"))
        except (TypeError, ValueError):
            path_rank = None
        try:
            counter_object_present = bool(
                _parse_bool(row.get("counter_object_present")) or False
            )
        except ValueError:
            counter_object_present = False
        base_position = _parse_json_value(
            row.get("base_position"),
            None,
            issues,
            field_name="base_position",
            pid=pid,
            trial_id=trial_id,
        )
        overlay_position = _parse_json_value(
            row.get("overlay_position"),
            None,
            issues,
            field_name="overlay_position",
            pid=pid,
            trial_id=trial_id,
        )
        overlay_matches = _parse_json_value(
            row.get("overlay_matches"),
            [],
            issues,
            field_name="overlay_matches",
            pid=pid,
            trial_id=trial_id,
        )
        aoi_matches = _parse_json_value(
            row.get("aoi_matches"),
            [],
            issues,
            field_name="aoi_matches",
            pid=pid,
            trial_id=trial_id,
        )
        raw_context_segments = _parse_json_value(
            row.get("context_segments"),
            [],
            issues,
            field_name="context_segments",
            pid=pid,
            trial_id=trial_id,
        )
        context_segments = []
        for context in raw_context_segments if isinstance(raw_context_segments, list) else []:
            if not isinstance(context, dict):
                continue
            try:
                context_start = _optional_int(context.get("start_ms"))
                context_end = _optional_int(context.get("end_ms"))
                context_horizon = _optional_int(context.get("overlay_horizon"))
                context_rank = _optional_int(context.get("path_rank"))
                context_overlay_flag = bool(
                    _parse_bool(context.get("overlay_flag")) or False
                )
                context_counter_present = bool(
                    _parse_bool(context.get("counter_object_present")) or False
                )
            except (TypeError, ValueError):
                continue
            if context_start is None or context_end is None or context_end <= context_start:
                continue
            context_segments.append(
                {
                    "start_ms": max(start_ms, context_start),
                    "end_ms": min(end_ms, context_end),
                    "base_aoi": normalize_base_aoi(
                        context.get("base_aoi"),
                        issues,
                        unknown_policy=unknown_aoi_policy,
                        pid=pid,
                        trial_id=trial_id,
                    ),
                    "base_position": context.get("base_position"),
                    "overlay_flag": context_overlay_flag,
                    "overlay_type": str(context.get("overlay_type") or "none"),
                    "overlay_horizon": context_horizon,
                    "path_rank": context_rank,
                    "overlay_underlying_aoi": str(
                        context.get("overlay_underlying_aoi") or ""
                    ),
                    "overlay_position": context.get("overlay_position"),
                    "overlay_matches": context.get("overlay_matches") or [],
                    "counter_object_present": context_counter_present,
                    "counter_object_type": str(context.get("counter_object_type") or ""),
                    "aoi_matches": [],
                }
            )
        context_segments = [
            context
            for context in context_segments
            if context["end_ms"] > context["start_ms"]
        ]
        fixations.append(
            Fixation(
                pid=pid,
                trial_id=trial_id,
                fixation_id=str(row.get("fixation_id", "")),
                start_ms=start_ms,
                end_ms=end_ms,
                base_aoi=normalize_base_aoi(
                    row.get("base_aoi"),
                    issues,
                    unknown_policy=unknown_aoi_policy,
                    pid=pid,
                    trial_id=trial_id,
                ),
                path_aoi=derive_path_aoi(
                    overlay_flag,
                    overlay_horizon,
                    issues,
                    pid=pid,
                    trial_id=trial_id,
                ),
                overlay_flag=overlay_flag,
                overlay_horizon=overlay_horizon,
                base_position=base_position,
                overlay_type=str(row.get("overlay_type") or "none"),
                path_rank=path_rank,
                overlay_underlying_aoi=str(row.get("overlay_underlying_aoi") or ""),
                overlay_position=overlay_position,
                overlay_matches=overlay_matches if isinstance(overlay_matches, list) else [],
                counter_object_present=counter_object_present,
                counter_object_type=str(row.get("counter_object_type") or ""),
                aoi_matches=aoi_matches if isinstance(aoi_matches, list) else [],
                context_segments=context_segments,
            )
        )

    fixations.sort(key=lambda item: (item.pid, item.trial_id, item.start_ms, item.end_ms, item.fixation_id))
    grouped: Dict[Tuple[str, str], List[Fixation]] = {}
    for fixation in fixations:
        grouped.setdefault((fixation.pid, fixation.trial_id), []).append(fixation)
    for (pid, trial_id), trial_fixations in grouped.items():
        previous = None
        for fixation in trial_fixations:
            if previous is not None and fixation.start_ms < previous.end_ms:
                issues.add(
                    "warning",
                    "OVERLAPPING_FIXATIONS",
                    f"Fixations {previous.fixation_id} and {fixation.fixation_id} overlap.",
                    pid=pid,
                    trial_id=trial_id,
                )
            if previous is None or fixation.end_ms > previous.end_ms:
                previous = fixation
    return fixations


def split_fixations_by_decision(
    fixations: Sequence[Fixation],
    decisions: Sequence[Decision],
) -> List[Dict[str, Any]]:
    """Split each fixation across full and pre-action half-open decision windows."""

    decisions_by_trial: Dict[Tuple[str, str], List[Decision]] = {}
    for decision in decisions:
        decisions_by_trial.setdefault((decision.pid, decision.trial_id), []).append(decision)

    segments: List[Dict[str, Any]] = []
    for fixation in fixations:
        contexts = fixation.context_segments or [
            {
                "start_ms": fixation.start_ms,
                "end_ms": fixation.end_ms,
                "base_aoi": fixation.base_aoi,
                "base_position": fixation.base_position,
                "overlay_flag": fixation.overlay_flag,
                "overlay_type": fixation.overlay_type,
                "overlay_horizon": fixation.overlay_horizon,
                "path_rank": fixation.path_rank,
                "overlay_underlying_aoi": fixation.overlay_underlying_aoi,
                "overlay_position": fixation.overlay_position,
                "overlay_matches": fixation.overlay_matches,
                "counter_object_present": fixation.counter_object_present,
                "counter_object_type": fixation.counter_object_type,
                "aoi_matches": fixation.aoi_matches,
            }
        ]
        for context in contexts:
            context_horizon = context.get("overlay_horizon")
            context_overlay_flag = bool(context.get("overlay_flag", False))
            if not context_overlay_flag or context_horizon is None:
                context_path_aoi = "None"
            elif context_horizon in (1, 2):
                context_path_aoi = "Near Path"
            elif context_horizon in (3, 4):
                context_path_aoi = "Far Path"
            else:
                context_path_aoi = "None"
            for decision in decisions_by_trial.get((fixation.pid, fixation.trial_id), []):
                windows = (
                    (WINDOW_FULL, decision.start_ms, decision.action_execution_ms),
                    (WINDOW_PRE_ACTION, decision.start_ms, decision.pre_action_end_ms),
                )
                for window_type, window_start, window_end in windows:
                    if window_end is None or window_end <= window_start:
                        continue
                    overlap = interval_overlap(
                        int(context["start_ms"]),
                        int(context["end_ms"]),
                        window_start,
                        window_end,
                    )
                    if overlap is None:
                        continue
                    overlap_start, overlap_end, overlap_duration = overlap
                    segments.append(
                        {
                            "pid": decision.pid,
                            "trial_id": decision.trial_id,
                            "step": decision.step,
                            "fixation_id": fixation.fixation_id,
                            "window_type": window_type,
                            "decision_start_ms": window_start,
                            "decision_end_ms": window_end,
                            "fixation_start_ms": fixation.start_ms,
                            "fixation_end_ms": fixation.end_ms,
                            "overlap_start_ms": overlap_start,
                            "overlap_end_ms": overlap_end,
                            "overlap_duration_ms": overlap_duration,
                            "base_aoi": context["base_aoi"],
                            "path_aoi": context_path_aoi,
                            "overlay_flag": int(context_overlay_flag),
                            "overlay_horizon": "" if context_horizon is None else context_horizon,
                            "base_position": json.dumps(
                                context.get("base_position"), ensure_ascii=False, separators=(",", ":")
                            ),
                            "overlay_type": context.get("overlay_type", "none"),
                            "path_rank": "" if context.get("path_rank") is None else context["path_rank"],
                            "overlay_underlying_aoi": context.get("overlay_underlying_aoi", ""),
                            "overlay_position": json.dumps(
                                context.get("overlay_position"), ensure_ascii=False, separators=(",", ":")
                            ),
                            "overlay_matches": json.dumps(
                                context.get("overlay_matches", []), ensure_ascii=False, separators=(",", ":")
                            ),
                            "counter_object_present": int(
                                bool(context.get("counter_object_present", False))
                            ),
                            "counter_object_type": context.get("counter_object_type", ""),
                            "aoi_matches": json.dumps(
                                context.get("aoi_matches", []), ensure_ascii=False, separators=(",", ":")
                            ),
                            "continued_from_previous_step": int(fixation.start_ms < window_start),
                            "continues_to_next_step": int(fixation.end_ms > window_end),
                        }
                    )
    return sorted(
        segments,
        key=lambda row: (
            row["pid"],
            row["trial_id"],
            int(row["step"]),
            int(row["overlap_start_ms"]),
            row["window_type"],
            str(row["fixation_id"]),
        ),
    )


def _sequence_from_segments(segments: Sequence[Mapping[str, Any]]) -> Tuple[List[str], List[int]]:
    sequence: List[str] = []
    durations: List[int] = []
    ordered = sorted(
        segments,
        key=lambda row: (
            int(row["overlap_start_ms"]),
            int(row["overlap_end_ms"]),
            str(row["fixation_id"]),
        ),
    )
    for segment in ordered:
        token = f"{segment['base_aoi']}|{segment['path_aoi']}"
        duration = int(segment["overlap_duration_ms"])
        if sequence and sequence[-1] == token:
            durations[-1] += duration
        else:
            sequence.append(token)
            durations.append(duration)
    return sequence, durations


def _decoded_json(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if _is_blank(value):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _segment_overlay_values(segment: Mapping[str, Any], field_name: str) -> set[int]:
    values = set()
    for match in _decoded_json(segment.get("overlay_matches"), []):
        if not isinstance(match, dict):
            continue
        try:
            value = _optional_int(match.get(field_name))
        except (TypeError, ValueError):
            value = None
        if value is not None:
            values.add(value)
    try:
        selected = _optional_int(segment.get(field_name))
    except (TypeError, ValueError):
        selected = None
    if bool(int(segment.get("overlay_flag", 0) or 0)) and selected is not None:
        values.add(selected)
    return values


def _exact_fixation_details(segments: Sequence[Mapping[str, Any]]) -> str:
    details = []
    for segment in sorted(
        segments,
        key=lambda row: (
            int(row["overlap_start_ms"]),
            int(row["overlap_end_ms"]),
            str(row["fixation_id"]),
        ),
    ):
        details.append(
            {
                "fixation_id": segment["fixation_id"],
                "start_ms": int(segment["overlap_start_ms"]),
                "end_ms": int(segment["overlap_end_ms"]),
                "duration_ms": int(segment["overlap_duration_ms"]),
                "base_aoi": segment["base_aoi"],
                "base_position": _decoded_json(segment.get("base_position"), None),
                "overlay_flag": int(segment.get("overlay_flag", 0) or 0),
                "overlay_type": segment.get("overlay_type", "none"),
                "overlay_horizon": _optional_int(segment.get("overlay_horizon")),
                "path_rank": _optional_int(segment.get("path_rank")),
                "overlay_underlying_aoi": segment.get("overlay_underlying_aoi", ""),
                "overlay_position": _decoded_json(segment.get("overlay_position"), None),
                "overlay_matches": _decoded_json(segment.get("overlay_matches"), []),
                "counter_object_present": int(
                    segment.get("counter_object_present", 0) or 0
                ),
                "counter_object_type": segment.get("counter_object_type", ""),
                "aoi_matches": _decoded_json(segment.get("aoi_matches"), []),
            }
        )
    return json.dumps(details, ensure_ascii=False, separators=(",", ":"))


def _overlay_dwell_fields(
    segments: Sequence[Mapping[str, Any]],
    window_prefix: str,
) -> Dict[str, int]:
    fields = {
        f"{window_prefix}_h{horizon}_dwell_ms": 0 for horizon in PATH_HORIZONS
    }
    fields.update(
        {f"{window_prefix}_rank{rank}_dwell_ms": 0 for rank in PATH_RANKS}
    )
    for segment in segments:
        duration = int(segment["overlap_duration_ms"])
        if not bool(int(segment.get("overlay_flag", 0) or 0)):
            continue
        try:
            horizon = _optional_int(segment.get("overlay_horizon"))
            rank = _optional_int(segment.get("path_rank"))
        except (TypeError, ValueError):
            continue
        if horizon in PATH_HORIZONS:
            fields[f"{window_prefix}_h{horizon}_dwell_ms"] += duration
        if rank in PATH_RANKS:
            fields[f"{window_prefix}_rank{rank}_dwell_ms"] += duration
    return fields


def _empty_visual_zone_fields() -> Dict[str, int]:
    return {
        f"{window}_h{horizon}_{zone}_ms": 0
        for window in ("pre_action", "full_window")
        for horizon in PATH_HORIZONS
        for zone in VISION_ZONES
    }


def _build_visual_zone_dwell(
    raw_gaze_rows: Sequence[Mapping[str, str]],
    decisions: Sequence[Decision],
) -> Dict[Tuple[str, str, int], Dict[str, int]]:
    decision_lookup = {
        (decision.pid, decision.trial_id, decision.step): decision
        for decision in decisions
    }
    rows_by_trial: Dict[Tuple[str, str], List[Mapping[str, str]]] = {}
    rows_by_decision: Dict[Tuple[str, str, int], List[Mapping[str, str]]] = {}
    for row in raw_gaze_rows:
        try:
            step = _optional_int(row.get("step"))
            timestamp_ms = _optional_int(row.get("timestamp_ms"))
        except (TypeError, ValueError):
            continue
        if step is None or timestamp_ms is None:
            continue
        key = (row.get("pid", ""), row.get("trial_id", ""), step)
        if key not in decision_lookup:
            continue
        enriched_row = dict(row)
        enriched_row["_timestamp_ms"] = timestamp_ms
        rows_by_decision.setdefault(key, []).append(enriched_row)
        rows_by_trial.setdefault(key[:2], []).append(enriched_row)

    typical_interval_by_trial: Dict[Tuple[str, str], int] = {}
    for trial_key, rows in rows_by_trial.items():
        timestamps = sorted(int(row["_timestamp_ms"]) for row in rows)
        intervals = [
            following - current
            for current, following in zip(timestamps, timestamps[1:])
            if 0 < following - current <= 50
        ]
        typical_interval_by_trial[trial_key] = max(
            1,
            int(round(median(intervals))) if intervals else 17,
        )

    dwell_by_decision: Dict[Tuple[str, str, int], Dict[str, int]] = {}
    for key, rows in rows_by_decision.items():
        decision = decision_lookup[key]
        rows.sort(key=lambda row: int(row["_timestamp_ms"]))
        typical_interval = typical_interval_by_trial.get(key[:2], 17)
        fields = _empty_visual_zone_fields()
        for index, row in enumerate(rows):
            sample_start = int(row["_timestamp_ms"])
            sample_end = sample_start + typical_interval
            if index + 1 < len(rows):
                following_timestamp = int(rows[index + 1]["_timestamp_ms"])
                if following_timestamp > sample_start:
                    sample_end = min(following_timestamp, sample_start + 50)
            for window_prefix, window_end in (
                ("full_window", decision.action_execution_ms),
                ("pre_action", decision.pre_action_end_ms),
            ):
                if window_end is None:
                    continue
                overlap = interval_overlap(
                    sample_start,
                    sample_end,
                    decision.start_ms,
                    window_end,
                )
                if overlap is None:
                    continue
                duration = overlap[2]
                candidates = []
                for horizon in PATH_HORIZONS:
                    zone = str(row.get(f"h{horizon}_vision_zone", "")).strip().lower()
                    if zone not in VISION_ZONES:
                        continue
                    try:
                        eccentricity = _optional_float(
                            row.get(f"h{horizon}_eccentricity_deg")
                        )
                    except (TypeError, ValueError):
                        eccentricity = None
                    candidates.append(
                        (
                            float("inf") if eccentricity is None else eccentricity,
                            horizon,
                            zone,
                        )
                    )
                if candidates:
                    _, horizon, zone = min(candidates)
                    fields[f"{window_prefix}_h{horizon}_{zone}_ms"] += duration
        dwell_by_decision[key] = fields
    return dwell_by_decision


def _build_gaze_validity_metrics(
    raw_gaze_rows: Sequence[Mapping[str, str]],
    decisions: Sequence[Decision],
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    decision_lookup = {
        (decision.pid, decision.trial_id, decision.step): decision
        for decision in decisions
    }
    decision_trial_keys = {key[:2] for key in decision_lookup}
    rows_by_trial: Dict[
        Tuple[str, str],
        List[Tuple[int, bool, Optional[bool], Optional[bool]]],
    ] = {}
    for row in raw_gaze_rows:
        try:
            timestamp_ms = _optional_int(row.get("timestamp_ms"))
            valid = _parse_bool(row.get("gaze_validity"))
            left_valid = _parse_bool(row.get("left_gaze_point_validity"))
            right_valid = _parse_bool(row.get("right_gaze_point_validity"))
        except (TypeError, ValueError):
            continue
        if timestamp_ms is None or valid is None:
            continue
        trial_key = (row.get("pid", ""), row.get("trial_id", ""))
        if trial_key not in decision_trial_keys:
            continue
        sample = (timestamp_ms, bool(valid), left_valid, right_valid)
        rows_by_trial.setdefault(trial_key, []).append(sample)

    typical_interval_by_trial: Dict[Tuple[str, str], int] = {}
    for trial_key, samples in rows_by_trial.items():
        timestamps = sorted(sample[0] for sample in samples)
        intervals = [
            following - current
            for current, following in zip(timestamps, timestamps[1:])
            if 0 < following - current <= 50
        ]
        typical_interval_by_trial[trial_key] = max(
            1,
            int(round(median(intervals))) if intervals else 17,
        )

    def summarize_window(
        samples: Sequence[Tuple[int, bool, Optional[bool], Optional[bool]]],
        start_ms: int,
        end_ms: Optional[int],
        typical_interval_ms: int,
    ) -> Tuple[int, int, Any, Any]:
        if end_ms is None or end_ms <= start_ms:
            return 0, 0, "", ""
        window_samples = sorted(
            (sample[0], sample[1])
            for sample in samples
            if start_ms <= sample[0] < end_ms
        )
        sample_count = len(window_samples)
        if sample_count == 0:
            return 0, 0, "", end_ms - start_ms
        valid_count = sum(int(valid) for _, valid in window_samples)

        # Allow one normal sampling interval at each boundary. Any longer
        # callback silence is treated as invalid gaze, even when no explicit
        # invalid sample was emitted by the tracker.
        current_invalid_gap_ms = max(
            0,
            window_samples[0][0] - start_ms - typical_interval_ms,
        )
        longest_invalid_gap_ms = current_invalid_gap_ms
        for index, (timestamp, valid) in enumerate(window_samples):
            next_timestamp = (
                window_samples[index + 1][0]
                if index + 1 < sample_count
                else end_ms
            )
            sample_end = min(end_ms, next_timestamp, timestamp + typical_interval_ms)
            sample_duration_ms = max(0, sample_end - timestamp)
            if valid:
                current_invalid_gap_ms = 0
            else:
                current_invalid_gap_ms += sample_duration_ms
                longest_invalid_gap_ms = max(
                    longest_invalid_gap_ms,
                    current_invalid_gap_ms,
                )
            callback_silence_ms = max(0, min(end_ms, next_timestamp) - sample_end)
            if callback_silence_ms:
                current_invalid_gap_ms += callback_silence_ms
                longest_invalid_gap_ms = max(
                    longest_invalid_gap_ms,
                    current_invalid_gap_ms,
                )
        return (
            sample_count,
            valid_count,
            valid_count / sample_count,
            longest_invalid_gap_ms,
        )

    def summarize_eye_window(
        samples: Sequence[Tuple[int, bool, Optional[bool], Optional[bool]]],
        start_ms: int,
        end_ms: Optional[int],
        eye_index: int,
    ) -> Tuple[int, int, Any]:
        if end_ms is None or end_ms <= start_ms:
            return 0, 0, ""
        validity_values = [
            sample[eye_index]
            for sample in samples
            if start_ms <= sample[0] < end_ms and sample[eye_index] is not None
        ]
        sample_count = len(validity_values)
        valid_count = sum(int(valid) for valid in validity_values)
        return (
            sample_count,
            valid_count,
            "" if sample_count == 0 else valid_count / sample_count,
        )

    metrics_by_decision: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for key, decision in decision_lookup.items():
        samples = rows_by_trial.get(key[:2], [])
        typical_interval = typical_interval_by_trial.get(key[:2], 17)
        pre_count, pre_valid, pre_ratio, _ = summarize_window(
            samples,
            decision.start_ms,
            decision.pre_action_end_ms,
            typical_interval,
        )
        full_count, full_valid, full_ratio, longest_invalid_gap = summarize_window(
            samples,
            decision.start_ms,
            decision.action_execution_ms,
            typical_interval,
        )
        metrics_by_decision[key] = {
            "pre_action_gaze_sample_count": pre_count,
            "pre_action_valid_gaze_count": pre_valid,
            "pre_action_gaze_validity": pre_ratio,
            "full_window_gaze_sample_count": full_count,
            "full_window_valid_gaze_count": full_valid,
            "full_window_gaze_validity": full_ratio,
            "longest_invalid_gap_ms": longest_invalid_gap,
        }
        for eye_prefix, eye_index in (("left", 2), ("right", 3)):
            for window_name, window_end in (
                ("pre_action", decision.pre_action_end_ms),
                ("full_window", decision.action_execution_ms),
            ):
                eye_count, eye_valid, eye_ratio = summarize_eye_window(
                    samples,
                    decision.start_ms,
                    window_end,
                    eye_index,
                )
                metrics_by_decision[key].update(
                    {
                        f"{window_name}_{eye_prefix}_gaze_sample_count": eye_count,
                        f"{window_name}_{eye_prefix}_valid_gaze_count": eye_valid,
                        f"{window_name}_{eye_prefix}_gaze_validity": eye_ratio,
                    }
                )
    return metrics_by_decision


def _validate_segment_durations(
    fixations: Sequence[Fixation],
    decisions: Sequence[Decision],
    segments: Sequence[Mapping[str, Any]],
    issues: IssueCollector,
) -> None:
    full_by_fixation: Dict[Tuple[str, str, str], int] = {}
    by_window: Dict[Tuple[str, str, int, str], int] = {}
    for segment in segments:
        if segment["window_type"] == WINDOW_FULL:
            fixation_key = (segment["pid"], segment["trial_id"], str(segment["fixation_id"]))
            full_by_fixation[fixation_key] = full_by_fixation.get(fixation_key, 0) + int(
                segment["overlap_duration_ms"]
            )
        window_key = (
            segment["pid"],
            segment["trial_id"],
            int(segment["step"]),
            segment["window_type"],
        )
        by_window[window_key] = by_window.get(window_key, 0) + int(segment["overlap_duration_ms"])

    for fixation in fixations:
        duration = fixation.end_ms - fixation.start_ms
        segmented = full_by_fixation.get((fixation.pid, fixation.trial_id, fixation.fixation_id), 0)
        if segmented > duration:
            issues.add(
                "error",
                "FIXATION_DURATION_EXCEEDED",
                f"Full-window segments total {segmented}ms but fixation duration is {duration}ms.",
                pid=fixation.pid,
                trial_id=fixation.trial_id,
            )

    for decision in decisions:
        for window_type, end_ms in (
            (WINDOW_FULL, decision.action_execution_ms),
            (WINDOW_PRE_ACTION, decision.pre_action_end_ms),
        ):
            if end_ms is None:
                continue
            duration = max(0, end_ms - decision.start_ms)
            segmented = by_window.get((decision.pid, decision.trial_id, decision.step, window_type), 0)
            if segmented > duration:
                issues.add(
                    "warning",
                    "WINDOW_DWELL_EXCEEDED",
                    f"{window_type} segment durations total {segmented}ms but window duration is {duration}ms; overlapping fixations likely exist.",
                    pid=decision.pid,
                    trial_id=decision.trial_id,
                    step=decision.step,
                )


def _mark_information_mismatches(
    decisions: Sequence[Decision],
    segments: Sequence[Mapping[str, Any]],
    issues: IssueCollector,
) -> None:
    max_segment_horizon: Dict[Tuple[str, str, int], int] = {}
    for segment in segments:
        if segment["window_type"] != WINDOW_FULL or _is_blank(segment.get("overlay_horizon")):
            continue
        key = (segment["pid"], segment["trial_id"], int(segment["step"]))
        max_segment_horizon[key] = max(
            max_segment_horizon.get(key, 0), int(segment["overlay_horizon"])
        )

    for decision in decisions:
        info = decision.information_steps
        content = _trajectory_has_content(decision.future_trajectory_value)
        trajectory_horizon = _max_overlay_horizon(decision.future_trajectory_value) or 0
        observed_horizon = max(
            trajectory_horizon,
            max_segment_horizon.get((decision.pid, decision.trial_id, decision.step), 0),
        )
        mismatch = decision.information_mismatch
        if info == 0 and content:
            mismatch = True
        if info in (2, 4) and observed_horizon > info:
            mismatch = True
        if mismatch:
            decision.information_mismatch = True
            issues.add(
                "warning",
                "INFORMATION_CONDITION_MISMATCH",
                f"information_steps={info}, displayed={content}, max_horizon={observed_horizon}.",
                pid=decision.pid,
                trial_id=decision.trial_id,
                step=decision.step,
            )
        elif info in (2, 4) and content and 0 < observed_horizon < info:
            issues.add(
                "warning",
                "SHORT_DISPLAYED_PATH",
                f"Displayed path ends at horizon {observed_horizon} under {info}-step condition.",
                pid=decision.pid,
                trial_id=decision.trial_id,
                step=decision.step,
            )


def _build_decision_rows(
    decisions: Sequence[Decision],
    segments: Sequence[Mapping[str, Any]],
    visual_zone_dwell: Mapping[Tuple[str, str, int], Mapping[str, int]],
    gaze_validity_metrics: Mapping[Tuple[str, str, int], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    segments_by_window: Dict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = {}
    for segment in segments:
        key = (segment["pid"], segment["trial_id"], int(segment["step"]), segment["window_type"])
        segments_by_window.setdefault(key, []).append(segment)

    previous_actions: Dict[Tuple[str, str], str] = {}
    rows: List[Dict[str, Any]] = []
    for decision in decisions:
        key = (decision.pid, decision.trial_id, decision.step)
        full_segments = segments_by_window.get((*key, WINDOW_FULL), [])
        pre_segments = segments_by_window.get((*key, WINDOW_PRE_ACTION), [])
        full_sequence, full_durations = _sequence_from_segments(full_segments)
        pre_sequence, pre_durations = _sequence_from_segments(pre_segments)
        first_aoi = full_sequence[0] if full_sequence else "null"
        if pre_segments:
            last_segment = max(
                pre_segments,
                key=lambda row: (int(row["overlap_end_ms"]), int(row["overlap_start_ms"])),
            )
            last_pre_action_aoi = f"{last_segment['base_aoi']}|{last_segment['path_aoi']}"
        else:
            last_pre_action_aoi = "null"
        trial_key = (decision.pid, decision.trial_id)
        previous_action = previous_actions.get(trial_key, "None")
        previous_actions[trial_key] = decision.human_action
        row = {
                "pid": decision.pid,
                "trial_id": decision.trial_id,
                "step": decision.step,
                "time_condition": "" if decision.time_condition is None else decision.time_condition,
                "information_steps": "" if decision.information_steps is None else decision.information_steps,
                "layout": decision.layout,
                "state_onset_ms": decision.start_ms,
                "human_input_ms": "" if decision.human_input_ms is None else decision.human_input_ms,
                "first_human_action": decision.first_human_action,
                "first_human_input_ms": (
                    "" if decision.first_human_input_ms is None else decision.first_human_input_ms
                ),
                "final_human_action": decision.final_human_action,
                "final_human_input_ms": (
                    "" if decision.final_human_input_ms is None else decision.final_human_input_ms
                ),
                "human_input_changed": int(decision.human_input_changed),
                "response_input_source": decision.response_input_source,
                "response_time_ms": (
                    "" if decision.response_time_ms is None else decision.response_time_ms
                ),
                "action_execution_ms": (
                    "" if decision.action_execution_ms is None else decision.action_execution_ms
                ),
                "env_state": decision.env_state_json,
                "previous_human_action": previous_action,
                "displayed_future_trajectory": decision.future_trajectory_json,
                "pre_action_aoi_sequence": _json_array(pre_sequence),
                "pre_action_aoi_durations": _json_array(pre_durations),
                "full_window_aoi_sequence": _json_array(full_sequence),
                "full_window_aoi_durations": _json_array(full_durations),
                "first_aoi": first_aoi,
                "last_pre_action_aoi": last_pre_action_aoi,
                "num_fixations": len({str(segment["fixation_id"]) for segment in full_segments}),
                "num_aoi_transitions": max(0, len(full_sequence) - 1),
                "pre_action_fixation_details": _exact_fixation_details(pre_segments),
                "full_window_fixation_details": _exact_fixation_details(full_segments),
                "human_action": decision.human_action,
                "agent_action": decision.agent_action,
                "is_human_idle": int(decision.is_human_idle),
                "is_collision": int(decision.is_collision),
                "reward": "" if decision.reward is None else decision.reward,
                "done": "" if decision.done is None else int(decision.done),
            }
        row.update(_overlay_dwell_fields(pre_segments, "pre_action"))
        row.update(_overlay_dwell_fields(full_segments, "full_window"))
        row.update(_empty_visual_zone_fields())
        row.update(visual_zone_dwell.get(key, {}))
        row.update(gaze_validity_metrics.get(key, {}))
        rows.append(row)
    return rows


def _build_qc_rows(
    decisions: Sequence[Decision],
    decision_rows: Sequence[Mapping[str, Any]],
    trial_contexts: Mapping[Tuple[str, str], Mapping[str, Any]],
    raw_gaze_rows: Sequence[Mapping[str, str]],
    issues: IssueCollector,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    decision_row_lookup = {
        (row["pid"], row["trial_id"], int(row["step"])): row for row in decision_rows
    }
    gaze_counts: Dict[str, List[int]] = {}
    per_eye_gaze_counts: Dict[str, Dict[str, List[int]]] = {}
    validity_field = None
    if raw_gaze_rows:
        for candidate in ("gaze_validity", "validity", "valid"):
            if candidate in raw_gaze_rows[0]:
                validity_field = candidate
                break
    if validity_field:
        LOGGER.info(
            "Gaze validity uses raw_gaze.csv:%s; truthy values are 1/true/yes.",
            validity_field,
        )
        for row in raw_gaze_rows:
            try:
                valid = _parse_bool(row.get(validity_field))
            except ValueError:
                valid = None
            if valid is None:
                continue
            counts = gaze_counts.setdefault(row.get("pid", ""), [0, 0])
            counts[0] += 1
            counts[1] += int(valid)
            for eye_prefix in ("left", "right"):
                try:
                    eye_valid = _parse_bool(row.get(f"{eye_prefix}_gaze_point_validity"))
                except ValueError:
                    eye_valid = None
                if eye_valid is None:
                    continue
                eye_counts = per_eye_gaze_counts.setdefault(row.get("pid", ""), {}).setdefault(
                    eye_prefix,
                    [0, 0],
                )
                eye_counts[0] += 1
                eye_counts[1] += int(eye_valid)
    elif raw_gaze_rows:
        issues.add("warning", "MISSING_GAZE_VALIDITY", "No gaze validity column was found.")

    decisions_by_pid: Dict[str, List[Decision]] = {}
    for decision in decisions:
        decisions_by_pid.setdefault(decision.pid, []).append(decision)

    qc_rows: List[Dict[str, Any]] = []
    condition_groups: Dict[Tuple[str, Any, Any, str], List[Decision]] = {}
    for pid, participant_decisions in sorted(decisions_by_pid.items()):
        trials: Dict[str, List[Decision]] = {}
        for decision in participant_decisions:
            trials.setdefault(decision.trial_id, []).append(decision)
            condition_groups.setdefault(
                (pid, decision.time_condition, decision.information_steps, decision.layout), []
            ).append(decision)

        done_errors = 0
        completed_from_done = 0
        for trial_id, trial_decisions in trials.items():
            trial_decisions.sort(key=lambda item: item.step)
            done_rows = [item for item in trial_decisions if item.done is True]
            if len(done_rows) != 1 or done_rows[0] is not trial_decisions[-1]:
                done_errors += 1
            else:
                completed_from_done += 1

        completed_values = []
        for (context_pid, _), context in trial_contexts.items():
            if context_pid != pid:
                continue
            try:
                completed = _parse_bool(context.get("completed"))
            except ValueError:
                completed = None
            if completed is not None:
                completed_values.append(bool(completed))
        completed_count = sum(completed_values) if completed_values else completed_from_done

        rows_for_pid = [
            decision_row_lookup[(item.pid, item.trial_id, item.step)] for item in participant_decisions
        ]
        no_fixation_count = sum(int(row["num_fixations"] == 0) for row in rows_for_pid)
        missing_future = sum(
            int(
                (decision.information_steps or 0) > 0
                and not _trajectory_has_content(decision.future_trajectory_value)
            )
            for decision in participant_decisions
        )
        action_counts = {name: 0 for name in ("up", "down", "left", "right", "interact", "stay")}
        for decision in participant_decisions:
            if decision.human_action in action_counts:
                action_counts[decision.human_action] += 1
            else:
                issues.add(
                    "warning",
                    "UNKNOWN_ACTION",
                    f"Unknown human action {decision.human_action!r}.",
                    pid=pid,
                    trial_id=decision.trial_id,
                    step=decision.step,
                )
        gaze_total, gaze_valid = gaze_counts.get(pid, [0, 0])
        left_total, left_valid = per_eye_gaze_counts.get(pid, {}).get("left", [0, 0])
        right_total, right_valid = per_eye_gaze_counts.get(pid, {}).get("right", [0, 0])
        total = len(participant_decisions)
        qc_rows.append(
            {
                "pid": pid,
                "completed_trial_count": completed_count,
                "total_decision_count": total,
                "done_error_count": done_errors,
                "timestamp_order_error_count": sum(item.timestamp_error for item in participant_decisions),
                "gaze_validity": "" if gaze_total == 0 else gaze_valid / gaze_total,
                "left_gaze_point_validity": (
                    "" if left_total == 0 else left_valid / left_total
                ),
                "right_gaze_point_validity": (
                    "" if right_total == 0 else right_valid / right_total
                ),
                "decision_without_fixation_ratio": 0.0 if total == 0 else no_fixation_count / total,
                "missing_future_trajectory_count": missing_future,
                "information_condition_mismatch_count": sum(
                    item.information_mismatch for item in participant_decisions
                ),
                "stay_ratio": 0.0 if total == 0 else action_counts["stay"] / total,
                "action_up_count": action_counts["up"],
                "action_down_count": action_counts["down"],
                "action_left_count": action_counts["left"],
                "action_right_count": action_counts["right"],
                "action_interact_count": action_counts["interact"],
                "action_stay_count": action_counts["stay"],
            }
        )

    condition_rows = []
    for (pid, time_condition, information_steps, layout), grouped_decisions in sorted(
        condition_groups.items(),
        key=lambda item: (
            item[0][0],
            -1 if item[0][1] is None else item[0][1],
            -1 if item[0][2] is None else item[0][2],
            item[0][3],
        ),
    ):
        condition_rows.append(
            {
                "pid": pid,
                "time_condition": "" if time_condition is None else time_condition,
                "information_steps": "" if information_steps is None else information_steps,
                "layout": layout,
                "trial_count": len({item.trial_id for item in grouped_decisions}),
                "decision_count": len(grouped_decisions),
            }
        )
    return qc_rows, condition_rows


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_completed_global_trial_indices(trial_csv_path: Path, pid: str) -> set[int]:
    """Return completed scheduled trial indices recorded for one participant."""

    if not trial_csv_path.exists():
        return set()
    completed_indices = set()
    with trial_csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if not row or row.get("pid") != pid:
                continue
            try:
                completed = _parse_bool(row.get("completed"))
                global_trial_index = _optional_int(row.get("global_trial_index"))
            except (TypeError, ValueError):
                continue
            if completed and global_trial_index is not None:
                completed_indices.add(global_trial_index)
    return completed_indices


def _filter_to_completed_trials(
    trial_rows: Sequence[Dict[str, str]],
    row_groups: Sequence[Sequence[Dict[str, str]]],
    issues: IssueCollector,
) -> Tuple[List[Dict[str, str]], List[List[Dict[str, str]]]]:
    if not trial_rows or "completed" not in trial_rows[0]:
        return list(trial_rows), [list(rows) for rows in row_groups]

    selected_by_schedule_cell: Dict[Tuple[str, int], Dict[str, str]] = {}
    selected_without_index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in trial_rows:
        try:
            completed = _parse_bool(row.get("completed"))
            global_index = _optional_int(row.get("global_trial_index"))
        except (TypeError, ValueError):
            completed = False
            global_index = None
        if not completed:
            continue
        if global_index is None:
            selected_without_index[(row.get("pid", ""), row.get("trial_id", ""))] = row
        else:
            selected_by_schedule_cell[(row.get("pid", ""), global_index)] = row

    selected_rows = list(selected_without_index.values()) + list(selected_by_schedule_cell.values())
    completed_keys = {
        (row.get("pid", ""), row.get("trial_id", ""))
        for row in selected_rows
    }
    all_trial_keys = {
        (row.get("pid", ""), row.get("trial_id", ""))
        for rows in row_groups
        for row in rows
    }
    excluded_keys = all_trial_keys - completed_keys
    if excluded_keys:
        issues.add(
            "warning",
            "INCOMPLETE_TRIALS_EXCLUDED",
            f"Excluded {len(excluded_keys)} incomplete or orphan trial attempt(s).",
        )

    filtered_groups = [
        [
            row
            for row in rows
            if (row.get("pid", ""), row.get("trial_id", "")) in completed_keys
        ]
        for rows in row_groups
    ]
    return selected_rows, filtered_groups


def build_model_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    input_files: Optional[Mapping[str, Path]] = None,
    unknown_aoi_policy: str = DEFAULT_UNKNOWN_AOI_POLICY,
) -> BuildResult:
    """Create derived model and QC CSVs from one participant log directory."""

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("--output-dir must differ from --input-dir; source logs are never overwritten.")
    if unknown_aoi_policy not in {"outside", "unknown", "error"}:
        raise ValueError("unknown_aoi_policy must be outside, unknown, or error")

    explicit = dict(input_files or {})
    issues = IssueCollector()
    paths = {
        kind: discover_input_file(input_dir, kind, explicit.get(kind))
        for kind in ("step", "fixation", "raw_gaze", "trial", "schedule", "metadata")
    }
    for required_kind in ("step", "fixation"):
        if paths[required_kind] is None:
            raise FileNotFoundError(f"Required input file not found: {required_kind}.csv")
    for optional_kind in ("raw_gaze", "trial", "schedule", "metadata"):
        if paths[optional_kind] is None:
            issues.add(
                "warning",
                "OPTIONAL_INPUT_MISSING",
                f"{optional_kind} input was not found; related fields and QC may be incomplete.",
            )

    step_rows = _read_csv(paths["step"], issues)
    fixation_rows = _read_csv(paths["fixation"], issues)
    raw_gaze_rows = _read_csv(paths["raw_gaze"], issues) if paths["raw_gaze"] else []
    trial_rows = _read_csv(paths["trial"], issues) if paths["trial"] else []
    schedule_rows = _read_csv(paths["schedule"], issues) if paths["schedule"] else []
    metadata: Dict[str, Any] = {}
    if paths["metadata"]:
        try:
            with paths["metadata"].open("r", encoding="utf-8") as file_obj:
                metadata = json.load(file_obj)
        except json.JSONDecodeError as exc:
            issues.add("warning", "INVALID_JSON", f"Cannot parse metadata.json: {exc}")

    if trial_rows:
        trial_rows, filtered_groups = _filter_to_completed_trials(
            trial_rows,
            (step_rows, fixation_rows, raw_gaze_rows),
            issues,
        )
        step_rows, fixation_rows, raw_gaze_rows = filtered_groups

    contexts = _build_contexts(trial_rows, schedule_rows, metadata, issues)
    decisions = _build_decisions(step_rows, contexts, issues)
    fixations = _build_fixations(fixation_rows, issues, unknown_aoi_policy)
    segments = split_fixations_by_decision(fixations, decisions)
    _validate_segment_durations(fixations, decisions, segments, issues)
    _mark_information_mismatches(decisions, segments, issues)
    visual_zone_dwell = _build_visual_zone_dwell(raw_gaze_rows, decisions)
    gaze_validity_metrics = _build_gaze_validity_metrics(raw_gaze_rows, decisions)
    decision_rows = _build_decision_rows(
        decisions,
        segments,
        visual_zone_dwell,
        gaze_validity_metrics,
    )
    qc_rows, condition_rows = _build_qc_rows(
        decisions,
        decision_rows,
        contexts,
        raw_gaze_rows,
        issues,
    )

    output_paths = {
        "segments": output_dir / "decision_fixation_segments.csv",
        "decisions": output_dir / "decision_dataset.csv",
        "participant_qc": output_dir / "participant_qc.csv",
        "condition_counts": output_dir / "participant_condition_counts.csv",
        "issues": output_dir / "processing_issues.csv",
    }
    _write_csv(output_paths["segments"], SEGMENT_FIELDS, segments)
    _write_csv(output_paths["decisions"], DECISION_FIELDS, decision_rows)
    _write_csv(output_paths["participant_qc"], QC_FIELDS, qc_rows)
    _write_csv(output_paths["condition_counts"], CONDITION_COUNT_FIELDS, condition_rows)
    _write_csv(output_paths["issues"], ISSUE_FIELDS, issues.issues)

    participant_ids = {decision.pid for decision in decisions}
    trial_ids = {(decision.pid, decision.trial_id) for decision in decisions}
    timestamp_errors = sum(decision.timestamp_error for decision in decisions)
    decisions_without_fixation = sum(int(row["num_fixations"] == 0) for row in decision_rows)
    return BuildResult(
        participant_count=len(participant_ids),
        trial_count=len(trial_ids),
        decision_count=len(decisions),
        fixation_segment_count=len(segments),
        timestamp_error_count=timestamp_errors,
        decisions_without_fixation=decisions_without_fixation,
        qc_warning_count=len(issues.issues),
        output_paths=output_paths,
    )
