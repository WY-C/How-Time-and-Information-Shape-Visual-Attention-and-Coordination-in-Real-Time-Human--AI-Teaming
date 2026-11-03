from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from final_model.partial_ver.constraints import (
    HumanLikeConstraintConfig,
    PeripheralVisionConfig,
    PreviewOpacityConfig,
)
from final_model.partial_ver.gaze import DEFAULT_EMMA_TIMING
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.peripheral import (
    PERIPHERAL_RNG_SEED_OFFSET,
    peripheral_detection_probability,
)
from final_model.partial_ver.tests.test_runtime import (
    FakeBackend,
    SameTarget,
    SameThenFar,
)


class TileBackend(FakeBackend):
    def canonical_pair(self):
        partner, learner = super().canonical_pair()
        for x in range(8):
            for y in range(5):
                learner[x, y, 0] = 1 + x * 5 + y + self.version
                learner[x, y, 19] = 100 + x * 5 + y + self.version
        return partner, learner

    def preview_planes(self):
        result = np.zeros((4, 8, 5), dtype=np.float32)
        for channel in range(self.horizon):
            for x in range(8):
                for y in range(5):
                    result[channel, x, y] = 10 * (channel + 1) + x * 5 + y
        return result


def config(mode: str, sigma: float = 3.0):
    return HumanLikeConstraintConfig(
        profile="custom",
        peripheral=PeripheralVisionConfig(mode, sigma),
        emma=DEFAULT_EMMA_TIMING,
        preview_opacity=PreviewOpacityConfig("off"),
    )


def adapter(mode: str, *, seed: int, horizon: int = 2):
    return PartialObservationAdapter(
        TileBackend(horizon), decision_time=200, fixation_shape="plus",
        gaze_policy=SameTarget(), constraint_config=config(mode),
        peripheral_seed=seed,
    )


def test_probability_formula_and_canonical_sigma():
    eccentricities = np.asarray([0.0, 1.0, 2.0, 4.0, 16.0])
    actual = peripheral_detection_probability(eccentricities, 3.0)
    np.testing.assert_allclose(
        actual, np.exp(-0.5 * np.square(eccentricities / 3.0))
    )
    assert actual[0] == 1.0
    assert np.all(np.diff(actual) <= 0.0)
    assert np.isfinite(actual).all()
    assert np.all((0.0 <= actual) & (actual <= 1.0))


@pytest.mark.parametrize(
    ("eccentricity", "sigma"),
    [(-1.0, 3.0), (float("inf"), 3.0), (float("nan"), 3.0),
     (1.0, 0.0), (1.0, -3.0), (1.0, float("inf"))],
)
def test_probability_rejects_invalid_inputs(eccentricity, sigma):
    with pytest.raises(ValueError):
        peripheral_detection_probability(eccentricity, sigma)


def test_none_consumes_no_peripheral_rng_and_has_zero_metrics():
    baseline = PartialObservationAdapter(
        TileBackend(), decision_time=200, fixation_shape="plus",
        gaze_policy=SameTarget(), peripheral_seed=123,
    )
    state_before = copy.deepcopy(baseline.peripheral_rng.bit_generator.state)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    baseline.collect_decision(lambda *_args: {"action": 4})
    assert baseline.peripheral_rng.bit_generator.state == state_before
    assert baseline.peripheral_sample_events == 0
    assert baseline.peripheral_rng_draw_count == 0
    assert random.getstate() == python_before
    assert all(
        np.array_equal(left, right)
        for left, right in zip(np.random.get_state(), numpy_before)
    )
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    metrics = baseline.metrics()
    for field in (
        "peripheral_attempts", "peripheral_successes", "peripheral_failures",
        "peripheral_sample_events", "peripheral_rng_draw_count",
        "peripheral_success_count",
        "peripheral_success_rate", "peripheral_successes_per_acquisition",
        "peripheral_successes_per_decision",
        "mean_peripheral_attempt_eccentricity_deg",
        "mean_peripheral_success_eccentricity_deg",
    ):
        assert metrics[field] == 0
    event = next(row for row in baseline.events if row.get("acquisition"))
    assert event["peripheral_candidate_indices"] == []
    assert event["peripheral_success_indices"] == []
    assert event["peripheral_failure_indices"] == []


def test_exact_3x3_acquires_nine_foveal_tiles_without_peripheral_sampling():
    runtime = PartialObservationAdapter(
        TileBackend(), decision_time=200, fixation_shape="3x3",
        gaze_policy=SameTarget(),
    )
    runtime.gaze.reset((3, 2), runtime.current_time_ms)
    rng_before = copy.deepcopy(runtime.peripheral_rng.bit_generator.state)

    runtime.advance_to(60)

    event = next(row for row in runtime.events if row.get("acquisition"))
    expected_tiles = {
        (x, y) for x in (2, 3, 4) for y in (1, 2, 3)
    }
    assert set(map(tuple, event["tiles"])) == expected_tiles
    assert len(event["foveal_tile_indices"]) == 9
    assert event["peripheral_candidate_indices"] == []
    assert event["peripheral_success_indices"] == []
    assert event["peripheral_sampled_new_fixation"] is False
    assert runtime.peripheral_sample_events == 0
    assert runtime.peripheral_rng_draw_count == 0
    assert runtime.peripheral_rng.bit_generator.state == rng_before


def test_p1_success_writes_exact_memory_freshness_and_preview():
    runtime = adapter("local", seed=1, horizon=2)
    runtime.advance_to(60)
    event = runtime.events[-1]
    assert event["foveal_tile_indices"] == [0, 1, 5]
    assert event["peripheral_candidate_indices"] == [6]
    assert event["peripheral_success_mask"] == [True]
    observation = runtime.memory.snapshot()
    current = runtime.backend.canonical_pair()[1]
    assert observation[1, 1, 0] == current[1, 1, 0]
    assert observation[1, 1, 19] == current[1, 1, 19]
    assert observation[1, 1, 20] == 255.0
    assert observation[1, 1, 21:25].tolist() == [16.0, 26.0, 0.0, 0.0]
    assert runtime.memory.last_observed_time_ms[1, 1] == 60.0


def test_p1_failure_preserves_memory_timestamp_freshness_and_preview():
    runtime = adapter("local", seed=4, horizon=2)
    old_state = runtime.backend.canonical_pair()[1].copy()
    old_preview = np.moveaxis(runtime.backend.preview_planes(), 0, -1)
    runtime.memory.acquire(old_state, old_preview, [(1, 1)], timestamp_ms=0.0)
    remembered = runtime.memory.state20[1, 1].copy()
    remembered_preview = runtime.memory.preview4[1, 1].copy()
    runtime.backend.version += 20
    runtime.advance_to(60)
    event = runtime.events[-1]
    assert event["peripheral_success_mask"] == [False]
    np.testing.assert_array_equal(runtime.memory.state20[1, 1], remembered)
    np.testing.assert_array_equal(runtime.memory.preview4[1, 1], remembered_preview)
    assert runtime.memory.last_observed_time_ms[1, 1] == 0.0
    assert runtime.memory.semantic_freshness()[1, 1] == pytest.approx(0.975)
    for tile in ((0, 0), (0, 1), (1, 0)):
        assert runtime.memory.last_observed_time_ms[tile] == 60.0


def test_h0_peripheral_success_has_no_preview_leakage():
    runtime = adapter("local", seed=1, horizon=0)
    runtime.advance_to(60)
    assert runtime.events[-1]["peripheral_success_mask"] == [True]
    assert np.count_nonzero(runtime.memory.snapshot()[..., 21:25]) == 0


def test_same_seed_repeats_different_seed_and_env_streams_diverge():
    first = adapter("global", seed=100)
    repeated = adapter("global", seed=100)
    different = adapter("global", seed=101)
    for runtime in (first, repeated, different):
        runtime.advance_to(600)
    first_masks = [
        row["peripheral_success_mask"] for row in first.events
        if row.get("acquisition")
    ]
    repeated_masks = [
        row["peripheral_success_mask"] for row in repeated.events
        if row.get("acquisition")
    ]
    different_masks = [
        row["peripheral_success_mask"] for row in different.events
        if row.get("acquisition")
    ]
    assert first_masks == repeated_masks
    assert first_masks != different_masks
    assert 100 + PERIPHERAL_RNG_SEED_OFFSET != 101 + PERIPHERAL_RNG_SEED_OFFSET


def test_active_peripheral_does_not_consume_global_rng_streams():
    runtime = adapter("global", seed=333)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    runtime.advance_to(600)
    assert random.getstate() == python_before
    assert all(
        np.array_equal(left, right)
        for left, right in zip(np.random.get_state(), numpy_before)
    )
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_peripheral_samples_once_per_new_fixation_not_repeated_same_tile():
    runtime = PartialObservationAdapter(
        TileBackend(), decision_time=600, fixation_shape="plus",
        gaze_policy=SameThenFar(), constraint_config=config("global"),
        peripheral_seed=333,
    )
    runtime.advance_to(600)
    acquisitions = [row for row in runtime.events if row.get("acquisition")]
    sampled = [
        row["peripheral_sampled_new_fixation"] for row in acquisitions
    ]
    assert sampled[:3] == [True, False, True]
    assert all(not value for value in sampled[3:])
    expected_draws = sum(
        row["peripheral_candidate_count"] for row in acquisitions
    )
    assert runtime.peripheral_rng_draw_count == expected_draws
    assert runtime.peripheral_sample_events == sum(sampled)
    assert runtime.metrics()["peripheral_success_count"] == (
        runtime.peripheral_successes
    )
    assert acquisitions[1]["peripheral_candidate_indices"] == []


def test_adapter_save_resume_restores_peripheral_rng_exactly():
    uninterrupted = adapter("global", seed=222)
    uninterrupted.advance_to(180)
    saved = uninterrupted.training_state()
    uninterrupted.advance_to(420)

    resumed = adapter("global", seed=222)
    resumed.load_training_state(saved)
    resumed.advance_to(420)
    assert resumed.peripheral_rng.bit_generator.state == (
        uninterrupted.peripheral_rng.bit_generator.state
    )
    assert resumed.peripheral_rng_draw_count == uninterrupted.peripheral_rng_draw_count
    assert resumed.metrics() == uninterrupted.metrics()
    assert list(resumed.events) == list(uninterrupted.events)


def test_resume_rejects_mode_or_sigma_mismatch():
    state = adapter("local", seed=5).training_state()
    with pytest.raises(RuntimeError, match="constraint mismatch"):
        adapter("global", seed=5).load_training_state(state)
    sigma4 = PartialObservationAdapter(
        TileBackend(), decision_time=200, fixation_shape="plus",
        gaze_policy=SameTarget(), constraint_config=config("local", 4.0),
        peripheral_seed=5,
    )
    with pytest.raises(RuntimeError, match="constraint mismatch"):
        sigma4.load_training_state(state)


def test_active_peripheral_requires_canonical_plus_shape():
    with pytest.raises(ValueError, match="fixation_shape=plus"):
        PartialObservationAdapter(
            TileBackend(), decision_time=200, fixation_shape="3x3",
            gaze_policy=SameTarget(), constraint_config=config("local"),
        )
