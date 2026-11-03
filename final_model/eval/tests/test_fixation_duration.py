import pytest

from final_model.eval.fixation_duration import (
    fixation_duration_metrics,
    split_fixation_window,
)


SOURCE = (1, 1)
TARGET = (4, 2)


def _event(timestamp, phase, fixation):
    return {
        "timestamp_ms": timestamp,
        "completed_phase": phase,
        "fixation_before": fixation,
    }


def _classify(rows, *, source="self", target="teammate"):
    result = []
    for row in rows:
        tile = (row["fixation_x"], row["fixation_y"])
        result.append({
            **row,
            "exclusive_center_aoi": source if tile == SOURCE else target,
        })
    return result


def test_prep_exec_and_post_landing_have_strict_temporal_accounting():
    rows = split_fixation_window(
        start_ms=0.0,
        end_ms=230.0,
        phase_events=(
            _event(100.0, "PREP", SOURCE),
            _event(150.0, "EXEC", SOURCE),
            _event(230.0, "ENCODE", TARGET),
        ),
        final_phase="READY",
        final_fixation=TARGET,
    )
    metrics = fixation_duration_metrics(_classify(rows))

    assert metrics["fixation_duration_ms_self"] == 100.0
    assert metrics["fixation_duration_ms_teammate"] == 80.0
    assert metrics["saccade_duration_ms"] == 50.0
    assert metrics["classified_fixation_duration_ms"] == 180.0


def test_same_fixation_encoding_is_entirely_current_fixation():
    rows = split_fixation_window(
        start_ms=0.0,
        end_ms=13.816,
        phase_events=(),
        final_phase="ENCODE",
        final_fixation=SOURCE,
    )
    metrics = fixation_duration_metrics(_classify(rows))
    assert metrics["fixation_duration_ms_self"] == pytest.approx(13.816)
    assert metrics["saccade_duration_ms"] == 0.0


def test_exec_is_excluded_on_both_sides_of_decision_boundary():
    first = split_fixation_window(
        start_ms=0.0, end_ms=25.0, phase_events=(),
        final_phase="EXEC", final_fixation=SOURCE,
    )
    second = split_fixation_window(
        start_ms=25.0, end_ms=50.0,
        phase_events=(_event(50.0, "EXEC", SOURCE),),
        final_phase="ENCODE", final_fixation=TARGET,
    )
    for rows in (first, second):
        metrics = fixation_duration_metrics(_classify(rows))
        assert metrics["classified_fixation_duration_ms"] == 0.0
        assert metrics["saccade_duration_ms"] == 25.0


def test_prep_is_split_to_source_across_decision_boundary():
    first = split_fixation_window(
        start_ms=0.0, end_ms=60.0, phase_events=(),
        final_phase="PREP", final_fixation=SOURCE,
    )
    second = split_fixation_window(
        start_ms=60.0, end_ms=100.0,
        phase_events=(_event(100.0, "PREP", SOURCE),),
        final_phase="EXEC", final_fixation=SOURCE,
    )
    assert fixation_duration_metrics(_classify(first))[
        "fixation_duration_ms_self"
    ] == 60.0
    assert fixation_duration_metrics(_classify(second))[
        "fixation_duration_ms_self"
    ] == 40.0


def test_post_landing_encoding_is_split_to_target_across_boundary():
    first = split_fixation_window(
        start_ms=0.0, end_ms=50.0, phase_events=(),
        final_phase="ENCODE", final_fixation=TARGET,
    )
    second = split_fixation_window(
        start_ms=50.0, end_ms=80.0,
        phase_events=(_event(80.0, "ENCODE", TARGET),),
        final_phase="READY", final_fixation=TARGET,
    )
    assert fixation_duration_metrics(_classify(first))[
        "fixation_duration_ms_teammate"
    ] == 50.0
    assert fixation_duration_metrics(_classify(second))[
        "fixation_duration_ms_teammate"
    ] == 30.0


def test_preview_priority_is_exclusive_when_teammate_overlaps():
    row = {
        "duration_ms": 20.0,
        "included_in_fixation_duration": 1,
        "partner_center": 1,
        "preview_center": 1,
        "exclusive_center_aoi": "preview",
    }
    metrics = fixation_duration_metrics([row])
    assert metrics["fixation_duration_ms_preview"] == 20.0
    assert metrics["fixation_duration_ms_teammate"] == 0.0
    assert metrics["classified_fixation_duration_ms"] == 20.0


def test_timeline_identity_and_positive_denominator_proportions():
    rows = split_fixation_window(
        start_ms=0.0,
        end_ms=230.0,
        phase_events=(
            _event(100.0, "PREP", SOURCE),
            _event(150.0, "EXEC", SOURCE),
            _event(230.0, "ENCODE", TARGET),
        ),
        final_phase="READY",
        final_fixation=TARGET,
    )
    metrics = fixation_duration_metrics(_classify(rows))
    assert metrics["accounted_gaze_duration_ms"] == pytest.approx(230.0)
    assert (
        metrics["classified_fixation_duration_ms"]
        + metrics["saccade_duration_ms"]
    ) == pytest.approx(230.0)
    assert sum(
        metrics[f"fixation_prop_{aoi}"]
        for aoi in ("self", "teammate", "preview", "other")
    ) == pytest.approx(1.0)


def test_empty_classified_duration_preserves_zero_share_convention():
    row = {
        "duration_ms": 50.0,
        "included_in_fixation_duration": 0,
    }
    metrics = fixation_duration_metrics([row])
    assert metrics["classified_fixation_duration_ms"] == 0.0
    assert all(
        metrics[f"fixation_prop_{aoi}"] == 0.0
        for aoi in ("self", "teammate", "preview", "other")
    )
