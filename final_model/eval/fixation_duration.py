"""Strict simulator-defined model fixation-duration accounting."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


FIXATION_DURATION_CONTRACT_ID = (
    "simulator_phase_central_fixation_exec_excluded_preview_first_v1"
)
FIXATION_AOIS = ("self", "teammate", "preview", "other")


def split_fixation_window(
    *,
    start_ms: float,
    end_ms: float,
    phase_events: Iterable[Mapping[str, Any]],
    final_phase: str,
    final_fixation: Sequence[int],
) -> list[dict[str, Any]]:
    """Split one decision window at the simulator's gaze phase boundaries."""
    start = float(start_ms)
    end = float(end_ms)
    if end < start:
        raise ValueError("fixation window cannot end before it starts")
    cursor = start
    rows: list[dict[str, Any]] = []

    def append_segment(segment_end, phase, fixation):
        nonlocal cursor
        segment_end = float(segment_end)
        if segment_end < cursor or segment_end > end:
            raise ValueError("gaze phase event lies outside or reverses the window")
        if segment_end > cursor:
            tile = tuple(map(int, fixation))
            phase_name = str(getattr(phase, "value", phase))
            rows.append({
                "segment_start_ms": cursor,
                "segment_end_ms": segment_end,
                "duration_ms": segment_end - cursor,
                "gaze_phase": phase_name,
                "included_in_fixation_duration": int(phase_name != "EXEC"),
                "fixation_x": tile[0],
                "fixation_y": tile[1],
            })
        cursor = segment_end

    for event in phase_events:
        append_segment(
            event["timestamp_ms"],
            event["completed_phase"],
            event["fixation_before"],
        )
    append_segment(end, final_phase, final_fixation)
    return rows


def fixation_duration_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    aoi_field: str = "exclusive_center_aoi",
) -> dict[str, float]:
    """Aggregate mutually-exclusive central-tile AOIs and exclude EXEC."""
    durations = {name: 0.0 for name in FIXATION_AOIS}
    saccade = 0.0
    for row in rows:
        duration = float(row["duration_ms"])
        if int(row["included_in_fixation_duration"]) == 0:
            saccade += duration
            continue
        aoi = str(row[aoi_field])
        aoi = {
            "partner": "teammate",
            "Player": "self",
            "Teammate": "teammate",
            "Action Preview": "preview",
            "Other": "other",
        }.get(aoi, aoi)
        if aoi not in durations:
            raise ValueError(f"invalid mutually-exclusive fixation AOI: {aoi}")
        durations[aoi] += duration

    classified = sum(durations.values())
    # Preserve the evaluator's established empty-unit convention: all AOI
    # shares are 0 when there is no classified fixation duration.
    denominator = classified if classified > 0.0 else 1.0
    result = {
        f"fixation_duration_ms_{name}": duration
        for name, duration in durations.items()
    }
    result.update({
        "saccade_duration_ms": saccade,
        "classified_fixation_duration_ms": classified,
        "accounted_gaze_duration_ms": classified + saccade,
        **{
            f"fixation_prop_{name}": durations[name] / denominator
            for name in FIXATION_AOIS
        },
    })
    return result
