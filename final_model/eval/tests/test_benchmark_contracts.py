import math

from final_model.eval.benchmark_contracts import (
    allocate_preview_horizons,
    fixation_onset_in_phase,
    is_carryover_fixation,
    is_new_for_carryover_decomposition,
    is_new_for_visual_sampling,
    manuscript_aoi,
    resolve_response_input,
    strict_source_check,
    v6_aoi_allocation,
)


def test_selected_action_resolver_uses_first_when_final_is_same():
    resolved = resolve_response_input("east", 40, "east", 160)
    assert (resolved.action, resolved.timestamp_ms) == ("east", 40)


def test_selected_action_resolver_uses_first_time_eventual_final_was_selected():
    resolved = resolve_response_input("east", 40, "north", 100)
    assert (resolved.action, resolved.timestamp_ms) == ("north", 100)


def test_selected_action_resolver_handles_one_or_no_recorded_input():
    assert resolve_response_input("east", 40, None, None).timestamp_ms == 40
    assert resolve_response_input(None, None, "north", 80).timestamp_ms == 80
    assert resolve_response_input(None, None, None, None).source == "timeout"


def test_analysis_specific_new_fixation_equality_is_not_unified():
    assert not is_carryover_fixation(100, 100)
    assert is_new_for_carryover_decomposition(100, 100)
    assert not is_new_for_visual_sampling(100, 100)
    assert is_new_for_visual_sampling(100.001, 100)


def test_strict_check_uses_half_open_onset_and_60ms_source_dwell():
    assert fixation_onset_in_phase(100, 100, 200)
    assert not fixation_onset_in_phase(200, 100, 200)
    assert strict_source_check(100, 100, 200, 60)
    assert not strict_source_check(100, 100, 200, 59.999)


def test_primary_and_v6_aoi_contracts_remain_distinct():
    assert manuscript_aoi("Partner", True) == "Action Preview"
    allocation = v6_aoi_allocation("Partner", True, 80)
    assert allocation["Teammate"] == 40
    assert allocation["Action Preview"] == 40


def test_overlapping_preview_horizons_split_only_preview_duration():
    allocation = allocate_preview_horizons(90, [1, 2, 2, 4])
    assert allocation == {1: 30, 2: 30, 4: 30}
    assert math.isclose(sum(allocation.values()), 90)
