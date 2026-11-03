from __future__ import annotations

import numpy as np
import pytest

from final_model.eval.common import mask_full_observation, valid_preview_modes


def observation(horizon):
    value = np.zeros((8, 5, 25), dtype=np.float32)
    value[..., :21] = np.arange(21, dtype=np.float32)
    value[..., 21:21 + horizon] = np.arange(1, horizon + 1, dtype=np.float32)
    return value


def test_h0_native_preview_zero():
    assert np.all(mask_full_observation(observation(0), evaluation_horizon=0, preview_mode="native")[..., 21:] == 0)


def test_h2_upper_preview_zero():
    result = mask_full_observation(observation(2), evaluation_horizon=2, preview_mode="native")
    assert np.all(result[..., 23:25] == 0)


def test_h4_native_preserved():
    source = observation(4)
    assert np.array_equal(mask_full_observation(source, evaluation_horizon=4, preview_mode="native"), source)


def test_h4_h12_exact_mask():
    result = mask_full_observation(observation(4), evaluation_horizon=4, preview_mode="h12")
    assert np.all(result[..., 21:23] != 0) and np.all(result[..., 23:25] == 0)


def test_h4_h1_exact_mask():
    result = mask_full_observation(observation(4), evaluation_horizon=4, preview_mode="h1")
    assert np.all(result[..., 21] != 0) and np.all(result[..., 22:25] == 0)


@pytest.mark.parametrize("horizon", (0, 2, 4))
def test_none_masks_all_preview(horizon):
    assert np.all(mask_full_observation(observation(horizon), evaluation_horizon=horizon, preview_mode="none")[..., 21:25] == 0)


def test_upward_information_leakage_fails():
    source = observation(2)
    source[..., 24] = 7
    with pytest.raises(RuntimeError, match="above"):
        mask_full_observation(source, evaluation_horizon=2, preview_mode="native")


def test_mask_does_not_modify_nonpreview_or_source():
    source = observation(4)
    before = source.copy()
    result = mask_full_observation(source, evaluation_horizon=4, preview_mode="none")
    assert np.array_equal(source, before)
    assert np.array_equal(result[..., :21], source[..., :21])


def test_valid_modes_are_horizon_bounded():
    assert valid_preview_modes(4) == ("native", "h12", "h1", "none")
    assert valid_preview_modes(2) == ("native", "h1", "none")
    assert valid_preview_modes(0) == ("native", "none")
