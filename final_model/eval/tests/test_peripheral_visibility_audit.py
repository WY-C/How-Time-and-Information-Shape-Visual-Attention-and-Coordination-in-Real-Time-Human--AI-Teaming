from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from final_model.eval.peripheral_visibility_audit import (
    build_audit,
    detection_probability,
    eccentricity_matrix,
    expected_visibility,
    p1_candidates,
    p2_candidates,
    plus_footprint,
    run_audit,
)
from final_model.server_parallel.gaze_v1 import GAZE_TILES, tile_to_action_index


def test_probability_at_zero_is_one_and_values_are_valid():
    assert detection_probability(0.0, 2.0) == 1.0
    values = detection_probability(np.linspace(0.0, 30.0, 301), 2.0)
    assert np.isfinite(values).all()
    assert np.all((0.0 <= values) & (values <= 1.0))


def test_probability_declines_with_eccentricity_and_grows_with_sigma():
    eccentricities = np.linspace(0.0, 20.0, 201)
    small = detection_probability(eccentricities, 2.0)
    large = detection_probability(eccentricities, 6.0)
    assert np.all(np.diff(small) <= 0.0)
    assert np.all(large >= small)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("inf"), float("nan")])
def test_probability_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        detection_probability(1.0, sigma)


def test_p1_candidates_are_exactly_clipped_diagonals():
    assert set(p1_candidates((3, 2))) == {
        (2, 1), (2, 3), (4, 1), (4, 3)
    }
    assert set(p1_candidates((0, 0))) == {(1, 1)}
    assert set(p1_candidates((0, 2))) == {(1, 1), (1, 3)}


def test_p2_candidates_are_exactly_non_plus_and_all_peripheral_sets_are_disjoint():
    board = set(GAZE_TILES)
    for fixation in GAZE_TILES:
        plus = set(plus_footprint(fixation))
        p1 = set(p1_candidates(fixation))
        p2 = set(p2_candidates(fixation))
        assert p2 == board - plus
        assert plus.isdisjoint(p1)
        assert plus.isdisjoint(p2)


def test_eccentricity_is_symmetric_and_diagonal_exceeds_orthogonal():
    matrix = eccentricity_matrix()
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(40))
    source = tile_to_action_index((3, 2))
    horizontal = matrix[source, tile_to_action_index((4, 2))]
    vertical = matrix[source, tile_to_action_index((3, 3))]
    diagonal = matrix[source, tile_to_action_index((4, 3))]
    assert diagonal > horizontal
    assert diagonal > vertical


def test_expected_visibility_formulas():
    probabilities = np.asarray([0.2, 0.4, 0.7])
    metrics = expected_visibility(probabilities)
    assert metrics["expected_successes"] == pytest.approx(probabilities.sum())
    assert metrics["probability_any_success"] == pytest.approx(
        1.0 - np.prod(1.0 - probabilities)
    )
    assert metrics["expected_fraction_detected"] == pytest.approx(
        probabilities.mean()
    )


def test_audit_does_not_require_rng(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("geometry audit used RNG")

    monkeypatch.setattr(np.random, "default_rng", fail)
    monkeypatch.setattr(np.random, "random", fail)
    audit = build_audit([2.0, 3.0])
    assert audit["matrix"].shape == (40, 40)


def test_import_does_not_pull_in_training_checkpoint_or_actionnet_modules():
    script = """
import sys
import final_model.eval.peripheral_visibility_audit
for name in (
    'final_model.partial_ver.training',
    'final_model.common.mappo',
    'final_model.checkpoint',
):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_output_is_deterministic_and_complete(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_audit([2.0, 3.0], first, make_figures=False)
    run_audit([3.0, 2.0], second, make_figures=False)
    expected = {
        "eccentricity_matrix.csv",
        "geometry_summary.csv",
        "plus_footprints.csv",
        "p1_candidates.csv",
        "p2_candidates.csv",
        "probability_by_eccentricity.csv",
        "p1_sigma_summary.csv",
        "p2_sigma_summary.csv",
        "distance_band_expected_success.csv",
        "per_fixation_sigma_metrics.csv",
        "summary.md",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert {path.name for path in second.iterdir()} == expected
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()
