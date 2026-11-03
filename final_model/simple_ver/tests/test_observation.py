import numpy as np

from final_model.simple_ver.observation import build_full_observation, preview4


def test_full_observation_contract_and_horizon_masks():
    state = np.arange(8 * 5 * 20, dtype=np.float32).reshape(8, 5, 20)
    planes = np.zeros((4, 8, 5), dtype=np.float32)
    for index in range(4):
        planes[index, index, 0] = 255.0
    for horizon, expected in ((0, 0), (2, 2), (4, 4)):
        observation = build_full_observation(state, planes, horizon)
        assert observation.shape == (8, 5, 25)
        assert observation.dtype == np.float32
        assert observation.flags.c_contiguous
        assert np.array_equal(observation[..., :20], state)
        assert np.all(observation[..., 20] == 255.0)
        assert sum(np.count_nonzero(observation[..., 21 + i]) > 0 for i in range(4)) == expected


def test_full_freshness_never_decays_or_depends_on_time_or_gaze():
    planes = np.arange(4 * 8 * 5, dtype=np.float32).reshape(4, 8, 5)
    forbidden_partial_levels = {0.0, 51.0, 102.0, 153.0, 204.0}
    for horizon in (0, 2, 4):
        for task_step in range(8):
            state = np.full((8, 5, 20), task_step, dtype=np.float32)
            observation = build_full_observation(state, planes, horizon)
            freshness = observation[..., 20]
            assert freshness.shape == (8, 5)
            assert np.all(freshness == 255.0)
            assert not any(value in forbidden_partial_levels for value in np.unique(freshness))


def test_preview_plane_axis_conversion_is_exact():
    planes = np.arange(4 * 8 * 5, dtype=np.float32).reshape(4, 8, 5)
    assert np.array_equal(preview4(planes, 4), np.moveaxis(planes, 0, -1))
