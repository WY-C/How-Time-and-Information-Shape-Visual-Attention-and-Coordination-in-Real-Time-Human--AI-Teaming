from __future__ import annotations

import json

import numpy as np
import pytest

from final_model.eval.fit_emma_timing import (
    construct_transitions,
    fit_linear,
    participant_cluster_bootstrap,
    run_fit,
)
from final_model.partial_ver.gaze import ReferenceGeometry


def transition(participant, x, y, **extra):
    return {
        "participant": participant,
        "visual_angle_deg": float(x),
        "observed_shift_ms": float(y),
        **extra,
    }


def fixation(participant, trial, fixation_id, start, end, tile, **extra):
    return {
        "participant": participant, "trial": trial,
        "fixation_id": fixation_id, "start_ms": start, "end_ms": end,
        "duration_ms": end - start, "centroid_x": 0.5, "centroid_y": 0.5,
        "tile": tile, "context_segment_count": 1,
        "time_ms": 200, "horizon": 0, "layout": "layout", "completed": 1,
        **extra,
    }


def test_linear_fit_recovers_known_coefficients():
    rows = [transition(f"P{i % 3}", x, 42.0 + 3.5 * x) for i, x in enumerate(range(12))]
    result = fit_linear(rows)
    assert result["execution_intercept_ms"] == pytest.approx(42.0)
    assert result["execution_slope_ms_per_degree"] == pytest.approx(3.5)


def test_participant_cluster_bootstrap_is_reproducible():
    rows = [transition(f"P{i % 4}", x, 30.0 + 2.0 * x + i % 2) for i, x in enumerate(range(20))]
    first = participant_cluster_bootstrap(rows, iterations=100, seed=9)
    second = participant_cluster_bootstrap(rows, iterations=100, seed=9)
    assert first == second


def test_transitions_never_cross_participant_or_trial():
    rows = [
        fixation("P01", "a", 1, 0, 100, (0, 0)),
        fixation("P01", "a", 2, 150, 250, (1, 0)),
        fixation("P01", "b", 1, 300, 400, (2, 0)),
        fixation("P02", "a", 1, 450, 550, (3, 0)),
    ]
    result = construct_transitions(rows, max_shift_ms=150)
    assert len(result) == 1
    assert result[0]["trial"] == "a"


def test_context_segments_do_not_create_physical_shifts():
    rows = [
        fixation("P01", "a", 1, 0, 100, (0, 0), context_segment_count=3),
        fixation("P01", "a", 2, 150, 250, (1, 0), context_segment_count=2),
    ]
    result = construct_transitions(rows, max_shift_ms=150)
    assert len(result) == 1
    assert result[0]["source_fixation_id"] == 1
    assert result[0]["destination_fixation_id"] == 2


@pytest.mark.parametrize(("start", "reason"), [(100, "nonpositive_or_overlapping_interval"), (90, "nonpositive_or_overlapping_interval")])
def test_zero_and_negative_intervals_are_explicitly_excluded(start, reason):
    rows = [
        fixation("P01", "a", 1, 0, 100, (0, 0)),
        fixation("P01", "a", 2, start, 200, (1, 0)),
    ]
    result = construct_transitions(rows, max_shift_ms=150)
    assert result[0]["primary_valid"] == 0
    assert result[0]["exclusion_reason"] == reason


def test_visual_angle_uses_canonical_geometry():
    rows = [
        fixation("P01", "a", 1, 0, 100, (0, 0)),
        fixation("P01", "a", 2, 150, 250, (3, 2)),
    ]
    result = construct_transitions(rows, max_shift_ms=150)
    assert result[0]["visual_angle_deg"] == pytest.approx(
        ReferenceGeometry().visual_angle_degrees((0, 0), (3, 2))
    )


def test_condition_labels_are_not_fit_predictors():
    rows = [transition("P01", x, 20 + x, time_ms=200, horizon=0) for x in range(5)]
    baseline = fit_linear(rows)
    for row in rows:
        row["time_ms"], row["horizon"] = 600, 4
    assert fit_linear(rows) == baseline


def test_time_horizon_replication_does_not_change_coefficients():
    base = [transition("P01", x, 20 + 2 * x, time_ms=200, horizon=0) for x in range(5)]
    replicated = [
        {**row, "participant": f"P{index}", "time_ms": time, "horizon": horizon}
        for index, (time, horizon) in enumerate(((200, 0), (400, 2), (600, 4)), 1)
        for row in base
    ]
    assert fit_linear(replicated)["execution_slope_ms_per_degree"] == pytest.approx(2.0)


def test_negative_unconstrained_coefficient_is_not_clipped():
    rows = [transition("P01", x, 100 - 4 * x) for x in range(10)]
    result = fit_linear(rows)
    assert result["execution_slope_ms_per_degree"] == pytest.approx(-4.0)


def test_long_gap_and_missing_tile_exclusions_are_reported():
    rows = [
        fixation("P01", "a", 1, 0, 100, (0, 0)),
        fixation("P01", "a", 2, 300, 400, (1, 0)),
        fixation("P01", "a", 3, 450, 550, None),
    ]
    result = construct_transitions(rows, max_shift_ms=150)
    assert [row["exclusion_reason"] for row in result] == [
        "gap_exceeds_execution_window", "missing_or_outside_canonical_tile"
    ]


def test_fitting_output_contains_source_provenance(tmp_path, monkeypatch):
    rows = []
    for participant in ("P01", "P02"):
        rows.extend([
            fixation(participant, "trial", 1, 0, 100, (0, 0)),
            fixation(participant, "trial", 2, 140, 240, (1, 0)),
            fixation(participant, "trial", 3, 285, 385, (3, 1)),
            fixation(participant, "trial", 4, 435, 535, (7, 4)),
        ])
    provenance = {
        "input_root": "/canonical/human",
        "participant_count": 2,
        "trial_count": 2,
        "excluded_incomplete_trial_count": 0,
        "fixation_count": len(rows),
        "fixations_with_multiple_context_segments": 0,
        "files": [{"path": "/canonical/human/P01/fixation.csv", "sha256": "a" * 64}],
    }
    monkeypatch.setattr(
        "final_model.eval.fit_emma_timing.load_human_fixations",
        lambda _path: (rows, provenance),
    )
    monkeypatch.setattr(
        "final_model.eval.fit_emma_timing._write_plot", lambda *_args: None
    )
    run_fit(
        input_root=tmp_path, output_dir=tmp_path / "output",
        bootstrap_iterations=20, seed=9, max_shift_ms=150,
    )
    saved = json.loads(
        (tmp_path / "output" / "fit_summary.json").read_text()
    )
    assert saved["source_provenance"] == provenance
    assert saved["tuned_cli_fragment"].startswith("--emma-mode tuned")
