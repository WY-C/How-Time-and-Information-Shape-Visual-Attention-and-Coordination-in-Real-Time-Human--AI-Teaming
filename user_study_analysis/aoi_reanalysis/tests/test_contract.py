import math

from common import AOIS, FINE_BIN_MS, allocate_future_steps, allocate_top_level


def test_information_only_is_not_other():
    seg = {"base_aoi": "Other", "overlay_flag": 1, "overlay_type": "path", "overlay_horizon": 1}
    a = allocate_top_level(seg, 100.0)
    assert a == {"Self": 0.0, "Partner": 0.0, "Information": 100.0, "Other": 0.0}


def test_self_information_overlap_splits_half():
    seg = {"base_aoi": "Self", "overlay_flag": 1, "overlay_type": "path", "overlay_horizon": 1}
    a = allocate_top_level(seg, 100.0)
    assert a["Self"] == 50.0
    assert a["Information"] == 50.0
    assert sum(a.values()) == 100.0


def test_partner_without_info_is_100_partner():
    seg = {"base_aoi": "Partner", "overlay_flag": 0}
    a = allocate_top_level(seg, 80.0)
    assert a["Partner"] == 80.0
    assert sum(a.values()) == 80.0


def test_other_is_residual():
    seg = {"base_aoi": "Other", "overlay_flag": 0}
    a = allocate_top_level(seg, 75.0)
    assert a["Other"] == 75.0
    assert sum(a.values()) == 75.0


def test_horizon_overlap_splits_allocated_information_not_full_segment():
    seg = {
        "base_aoi": "Self", "overlay_flag": 1, "overlay_type": "path",
        "overlay_matches": [
            {"overlay_type": "path", "overlay_horizon": 1},
            {"overlay_type": "path", "overlay_horizon": 2},
        ],
    }
    top = allocate_top_level(seg, 100.0)
    h = allocate_future_steps(seg, top["Information"])
    assert top["Information"] == 50.0
    assert h[1] == 25.0 and h[2] == 25.0
    assert sum(h.values()) == 50.0


def test_fine_bin_is_60hz():
    assert math.isclose(FINE_BIN_MS, 1000.0 / 60.0)
    assert round(200 / FINE_BIN_MS) == 12
    assert round(400 / FINE_BIN_MS) == 24
    assert round(600 / FINE_BIN_MS) == 36
