from __future__ import annotations

from final_model.eval.preview_overlap_audit import (
    DisplayedStep,
    _combination_rows,
    analyze_assignments,
    displayed_step_from_prediction,
    x_major_tile_index,
)


def _analyze(*steps: DisplayedStep, horizon: int = 4):
    return analyze_assignments(steps, horizon=horizon, width=8, height=5)


def test_one_tile_one_step_has_no_overlap():
    result = _analyze(DisplayedStep(1, (2, 3)))
    assert result["unique_tiles"] == 1
    assert result["overlap_tiles"] == 0
    assert result["has_overlap"] is False


def test_same_tile_h1_h2_is_overlap_and_different_tiles_are_not():
    same = _analyze(
        DisplayedStep(1, (2, 3)), DisplayedStep(2, (2, 3)), horizon=2
    )
    different = _analyze(
        DisplayedStep(1, (2, 3)), DisplayedStep(2, (2, 4)), horizon=2
    )
    assert same["overlap_tiles"] == 1
    assert same["overlap_rows"][0]["steps"] == (1, 2)
    assert different["overlap_tiles"] == 0


def test_exact_combination_and_multiplicity_counting():
    result = _analyze(
        DisplayedStep(1, (0, 0)),
        DisplayedStep(2, (0, 0)),
        DisplayedStep(3, (0, 0)),
        DisplayedStep(4, (1, 0)),
    )
    assert result["multiplicity"] == {1: 1, 2: 0, 3: 1, 4: 0}
    assert result["overlap_rows"][0]["steps"] == (1, 2, 3)

    decisions = [{
        "condition": "H4", "time_constraint_ms": 200,
        "layout": "counter_circuit", "analysis": result,
    }]
    rows = _combination_rows(decisions)
    overall = {row["step_combination"]: row for row in rows if row["scope"] == "overall"}
    assert overall["H1+H2+H3"]["overlap_tile_count"] == 1
    assert overall["H1+H2"]["overlap_tile_count"] == 0


def test_h2_excludes_h3_h4():
    result = _analyze(
        DisplayedStep(1, (0, 0)),
        DisplayedStep(2, (1, 0)),
        DisplayedStep(3, (0, 0)),
        DisplayedStep(4, (0, 0)),
        horizon=2,
    )
    assert result["positive_step_instances"] == 2
    assert result["overlap_tiles"] == 0


def test_renderer_alpha_is_max_while_model_probabilities_remain_per_step():
    h2 = _analyze(
        DisplayedStep(1, (0, 0)), DisplayedStep(2, (0, 0)), horizon=2
    )["overlap_rows"][0]
    assert h2["raw_alphas"] == (250, 70)
    assert h2["renderer_max_alpha"] == 250
    assert h2["probabilities"] == (250 / 255.0, 70 / 255.0)

    h4 = _analyze(
        DisplayedStep(1, (0, 0)), DisplayedStep(4, (0, 0))
    )["overlap_rows"][0]
    assert h4["raw_alphas"] == (250, 70)
    assert h4["renderer_max_alpha"] == 250
    assert h4["probabilities"] == (250 / 255.0, 70 / 255.0)


def test_output_order_is_deterministic_and_x_major():
    first = _analyze(
        DisplayedStep(1, (1, 0)),
        DisplayedStep(2, (0, 4)),
        DisplayedStep(3, (0, 1)),
    )
    second = _analyze(
        DisplayedStep(3, (0, 1)),
        DisplayedStep(1, (1, 0)),
        DisplayedStep(2, (0, 4)),
    )
    assert [row["tile"] for row in first["tile_rows"]] == [
        (0, 1), (0, 4), (1, 0)
    ]
    assert first["tile_rows"] == second["tile_rows"]
    assert x_major_tile_index((3, 2), width=8, height=5) == 17


def test_repeated_identical_step_does_not_become_overlap():
    repeated = DisplayedStep(1, (2, 2))
    result = _analyze(repeated, repeated)
    assert result["positive_step_instances"] == 1
    assert result["multiplicity"][1] == 1
    assert result["overlap_tiles"] == 0


def test_interaction_target_remapping_can_create_overlap():
    movement = displayed_step_from_prediction(
        step=1, predicted_position=(3, 2), predicted_interact=False,
        predicted_interact_position=None,
    )
    interaction = displayed_step_from_prediction(
        step=2, predicted_position=(2, 2), predicted_interact=True,
        predicted_interact_position=(3, 2),
    )
    result = _analyze(movement, interaction, horizon=2)
    overlap = result["overlap_rows"][0]
    assert overlap["provenance_class"] == "interaction-containing"
    assert overlap["interaction_remap_created"] is True
    assert result["no_remap_overlap_tiles"] == 0

