#!/usr/bin/env python3
import argparse
from bisect import bisect_right
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lib" / "overcooked_ai"))

from experiment_logging import (  # noqa: E402
    FIXATION_FIELDS,
    TRIAL_FIELDS,
    build_ivt_fixation_rows,
)
from ivt import IVTConfig  # noqa: E402


def _read_dict_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = [row for row in reader if row and any(value for value in row.values())]
        return list(reader.fieldnames or []), rows


def _parse_json(value: Optional[str], fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _legacy_context(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "base_aoi": row.get("base_aoi", ""),
        "base_position": _parse_json(row.get("base_position"), None),
        "overlay_flag": int(row.get("overlay_flag") or 0),
        "overlay_type": row.get("overlay_type") or "none",
        "overlay_horizon": _parse_optional_int(row.get("overlay_horizon")),
        "path_rank": _parse_optional_int(row.get("path_rank")),
        "overlay_underlying_aoi": row.get("overlay_underlying_aoi") or None,
        "overlay_position": _parse_json(row.get("overlay_position"), None),
        "overlay_matches": _parse_json(row.get("overlay_matches"), []),
        "counter_object_present": int(row.get("counter_object_present") or 0),
        "counter_object_type": row.get("counter_object_type") or None,
    }


def _load_context_intervals(path: Path) -> Dict[str, Tuple[List[int], List[Dict[str, Any]]]]:
    _, rows = _read_dict_rows(path)
    intervals_by_trial: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("record_type", "aoi_segment") != "aoi_segment":
            continue
        trial_id = row.get("trial_id", "")
        interval = {
            "start_ms": int(row["fixation_start_ms"]),
            "end_ms": int(row["fixation_end_ms"]),
            "context": _legacy_context(row),
        }
        intervals_by_trial.setdefault(trial_id, []).append(interval)

    indexed = {}
    for trial_id, intervals in intervals_by_trial.items():
        intervals.sort(key=lambda interval: interval["start_ms"])
        indexed[trial_id] = ([interval["start_ms"] for interval in intervals], intervals)
    return indexed


def _find_context(
    timestamp_ms: int,
    indexed_intervals: Optional[Tuple[List[int], List[Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    if indexed_intervals is None:
        return None
    starts, intervals = indexed_intervals
    index = bisect_right(starts, timestamp_ms) - 1
    if index < 0 or timestamp_ms > intervals[index]["end_ms"]:
        return None
    return intervals[index]["context"]


def _raw_samples_by_trial(
    raw_gaze_path: Path,
    context_intervals: Dict[str, Tuple[List[int], List[Dict[str, Any]]]],
) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    _, rows = _read_dict_rows(raw_gaze_path)
    trial_order: List[str] = []
    samples_by_trial: Dict[str, List[Dict[str, Any]]] = {}
    pid_by_trial: Dict[str, str] = {}
    for row in rows:
        trial_id = row["trial_id"]
        if trial_id not in samples_by_trial:
            trial_order.append(trial_id)
            samples_by_trial[trial_id] = []
            pid_by_trial[trial_id] = row.get("pid", "")
        valid = row.get("gaze_validity") == "1" and row.get("gaze_x") not in (None, "") and row.get(
            "gaze_y"
        ) not in (None, "")
        sample: Dict[str, Any] = {
            "timestamp_ms": int(row["timestamp_ms"]),
            "x": float(row["gaze_x"]) if valid else None,
            "y": float(row["gaze_y"]) if valid else None,
            "valid": valid,
        }
        context = _find_context(sample["timestamp_ms"], context_intervals.get(trial_id))
        if context is not None:
            sample.update(context)
            sample["step"] = int(row["step"]) if row.get("step") not in (None, "") else 0
        samples_by_trial[trial_id].append(sample)
    return trial_order, samples_by_trial, pid_by_trial


def _backup_once(path: Path) -> Path:
    backup_path = path.with_name(f"{path.name}.pre_ivt.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def _write_fixations(
    output_path: Path,
    trial_order: Sequence[str],
    rows_by_trial: Dict[str, List[Dict[str, Any]]],
) -> None:
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=FIXATION_FIELDS)
        writer.writeheader()
        for trial_index, trial_id in enumerate(trial_order):
            if trial_index > 0:
                file_obj.write("\n")
            writer.writerows(rows_by_trial[trial_id])
    temp_path.replace(output_path)


def _update_trial_config(trial_path: Path, config: IVTConfig) -> None:
    _, rows = _read_dict_rows(trial_path)
    config_values = {
        "ivt_screen_diagonal_inches": config.screen_diagonal_inches,
        "ivt_screen_aspect_ratio": config.aspect_ratio_label,
        "ivt_viewing_distance_mm": config.viewing_distance_mm,
        "ivt_median_window_samples": config.median_window_samples,
        "ivt_velocity_window_ms": config.velocity_window_ms,
        "ivt_velocity_threshold_deg_s": config.velocity_threshold_deg_s,
        "ivt_merge_gap_ms": config.merge_gap_ms,
        "ivt_merge_angle_deg": config.merge_angle_deg,
        "ivt_min_fixation_duration_ms": config.min_fixation_duration_ms,
        "ivt_foveal_threshold_deg": config.foveal_threshold_deg,
        "ivt_parafoveal_threshold_deg": config.parafoveal_threshold_deg,
    }
    temp_path = trial_path.with_suffix(f"{trial_path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=TRIAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row_index, row in enumerate(rows):
            if row_index > 0:
                file_obj.write("\n")
            row.update(config_values)
            writer.writerow({field_name: row.get(field_name, "") for field_name in TRIAL_FIELDS})
    temp_path.replace(trial_path)


def reprocess_session(session_dir: Path, config: IVTConfig, write: bool) -> Dict[str, int]:
    raw_gaze_path = session_dir / "raw_gaze.csv"
    fixation_path = session_dir / "fixation.csv"
    trial_path = session_dir / "trial.csv"
    for required_path in (raw_gaze_path, fixation_path, trial_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required log file not found: {required_path}")

    context_source = fixation_path
    _, current_fixation_rows = _read_dict_rows(fixation_path)
    if current_fixation_rows and not any(
        row.get("record_type", "aoi_segment") == "aoi_segment" for row in current_fixation_rows
    ):
        backup_candidate = fixation_path.with_name(f"{fixation_path.name}.pre_ivt.bak")
        if not backup_candidate.exists():
            raise ValueError("Legacy AOI segments are unavailable; cannot restore AOI labels.")
        context_source = backup_candidate

    context_intervals = _load_context_intervals(context_source)
    trial_order, samples_by_trial, pid_by_trial = _raw_samples_by_trial(raw_gaze_path, context_intervals)
    rows_by_trial = {
        trial_id: build_ivt_fixation_rows(
            samples_by_trial[trial_id],
            pid=pid_by_trial[trial_id],
            trial_id=trial_id,
            config=config,
        )
        for trial_id in trial_order
    }
    summary = {
        "trials": len(trial_order),
        "raw_samples": sum(len(samples) for samples in samples_by_trial.values()),
        "fixations": sum(len(rows) for rows in rows_by_trial.values()),
    }
    if write:
        _backup_once(fixation_path)
        _backup_once(trial_path)
        _write_fixations(fixation_path, trial_order, rows_by_trial)
        _update_trial_config(trial_path, config)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild fixation.csv with the Tobii I-VT preset.")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--screen-diagonal-inches", type=float, default=27.0)
    parser.add_argument("--viewing-distance-mm", type=float, default=600.0)
    parser.add_argument("--write", action="store_true", help="Back up and replace fixation.csv and trial.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = IVTConfig(
        screen_diagonal_inches=args.screen_diagonal_inches,
        viewing_distance_mm=args.viewing_distance_mm,
    )
    summary = reprocess_session(args.session_dir.resolve(), config, args.write)
    mode = "written" if args.write else "preview"
    print(
        f"I-VT {mode}: trials={summary['trials']} raw_samples={summary['raw_samples']} "
        f"fixations={summary['fixations']} screen={config.screen_diagonal_inches:g}in "
        f"distance={config.viewing_distance_mm:g}mm"
    )


if __name__ == "__main__":
    main()
