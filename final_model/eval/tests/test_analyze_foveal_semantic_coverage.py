from __future__ import annotations

import numpy as np
import pandas as pd

from final_model.eval.analyze_foveal_semantic_coverage import (
    recover_preview_tile_indices,
    scope_probability,
    spatial_relation,
)
from final_model.partial_ver.gaze import ReferenceGeometry, acquisition_tiles
from final_model.partial_ver.peripheral import peripheral_detection_probability
from final_model.server_parallel.gaze_v1 import (
    GAZE_ACTION_COUNT,
    GAZE_TILES,
    tile_to_action_index,
)


def test_x_major_plus_and_spatial_relations_use_production_helpers():
    assert tile_to_action_index((3, 1)) == 16
    assert sorted(tile_to_action_index(tile) for tile in acquisition_tiles((3, 1), "plus")) == [
        11, 15, 16, 17, 21,
    ]
    assert spatial_relation((3, 1), (3, 1)) == "CENTER"
    assert spatial_relation((3, 1), (4, 1)) == "CARDINAL"
    assert spatial_relation((3, 1), (4, 2)) == "DIAGONAL"
    assert spatial_relation((3, 1), (7, 4)) == "FAR"


def test_counterfactual_scopes_match_production_gaussian():
    geometry = ReferenceGeometry()
    fixation = (3, 1)
    diagonal = (4, 2)
    far = (7, 4)
    sigma = 3.0
    diagonal_probability = peripheral_detection_probability(
        geometry.visual_angle_degrees(fixation, diagonal), sigma
    )
    far_probability = peripheral_detection_probability(
        geometry.visual_angle_degrees(fixation, far), sigma
    )
    assert scope_probability(
        fixation, diagonal, scope="plus_none", sigma_deg=sigma, geometry=geometry
    ) == 0.0
    assert scope_probability(
        fixation, diagonal, scope="exact_3x3", sigma_deg=sigma, geometry=geometry
    ) == 1.0
    assert np.isclose(scope_probability(
        fixation, diagonal, scope="plus_local", sigma_deg=sigma, geometry=geometry
    ), diagonal_probability)
    assert scope_probability(
        fixation, far, scope="plus_local", sigma_deg=sigma, geometry=geometry
    ) == 0.0
    assert np.isclose(scope_probability(
        fixation, far, scope="plus_global", sigma_deg=sigma, geometry=geometry
    ), far_probability)


def test_preview_tile_recovery_uses_center_and_plus_policy_masses():
    probabilities = np.linspace(0.001, 0.040, GAZE_ACTION_COUNT)
    probabilities /= probabilities.sum()
    target_index = tile_to_action_index((4, 2))
    row = {f"p{i}": probabilities[i] for i in range(GAZE_ACTION_COUNT)}
    for step in range(1, 5):
        active = step <= 2
        row[f"h{step}_center_compatible_target_count"] = int(active)
        row[f"h{step}_center_policy_mass"] = (
            probabilities[target_index] if active else 0.0
        )
        row[f"h{step}_footprint_policy_mass"] = (
            probabilities[[
                index for index, fixation in enumerate(GAZE_TILES)
                if (4, 2) in acquisition_tiles(fixation, "plus")
            ]].sum() if active else 0.0
        )
    recovered = recover_preview_tile_indices(pd.DataFrame([row]))
    assert recovered["H1"].tolist() == [target_index]
    assert recovered["H2"].tolist() == [target_index]
    assert recovered["H3"].tolist() == [-1]
    assert recovered["H4"].tolist() == [-1]
