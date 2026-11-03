from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random

import torch

import numpy as np
import pytest

from final_model.partial_ver.constraints import (
    BASELINE_CONSTRAINT_CONFIG,
    CANONICAL_TUNED_EMMA_TIMING,
    HumanLikeConstraintConfig,
    PeripheralVisionConfig,
    PreviewOpacityConfig,
)
from final_model.partial_ver.gaze import DEFAULT_EMMA_TIMING
from final_model.partial_ver.memory import OBSERVATION_CONTRACT_ID
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.preview_opacity import (
    PREVIEW_OPACITY_RNG_SEED_OFFSET,
    PREVIEW_RENDERER_GIT_BLOB,
    PREVIEW_RENDERER_SHA256,
    alpha_to_detection_probability,
    preview_opacity_metadata,
    rendered_preview_alphas,
    rendered_preview_probabilities,
    sample_preview_opacity,
)
from final_model.partial_ver.tests.test_peripheral_runtime import TileBackend
from final_model.partial_ver.tests.test_runtime import SameTarget


class PreviewBackend(TileBackend):
    def __init__(self, horizon=4):
        super().__init__(horizon)
        self.preview = np.zeros((4, 8, 5), dtype=np.float32)

    def preview_planes(self):
        return self.preview.copy()


class FixedRng:
    def __init__(self, value):
        self.value = float(value)
        self.draws = 0

    def random(self, size=None):
        count = 1 if size is None else int(size)
        self.draws += count
        values = np.full(count, self.value, dtype=np.float64)
        return float(values[0]) if size is None else values


def config(mode="global", opacity="actual"):
    return HumanLikeConstraintConfig(
        profile="custom", peripheral=PeripheralVisionConfig(mode, 3.0),
        emma=DEFAULT_EMMA_TIMING,
        preview_opacity=PreviewOpacityConfig(opacity),
    )


def adapter(backend, *, seed=41, mode="global", opacity="actual"):
    return PartialObservationAdapter(
        backend, decision_time=200, fixation_shape="plus",
        gaze_policy=SameTarget(), constraint_config=config(mode, opacity),
        peripheral_seed=seed, opacity_seed=seed + PREVIEW_OPACITY_RNG_SEED_OFFSET,
    )


def test_actual_renderer_alpha_and_provenance_are_exact():
    assert OBSERVATION_CONTRACT_ID == (
        "partial_state20_continuous_linear_freshness_configurable_expiry_"
        "raw255_preview4_v7"
    )
    assert rendered_preview_alphas(2) == (250, 70, 0, 0)
    assert rendered_preview_alphas(4) == (250, 190, 130, 70)
    assert rendered_preview_alphas(1) == (250, 0, 0, 0)
    assert rendered_preview_probabilities(2) == pytest.approx(
        (250 / 255, 70 / 255, 0, 0)
    )
    metadata = preview_opacity_metadata()
    assert metadata["preview_renderer_git_blob"] == PREVIEW_RENDERER_GIT_BLOB
    assert metadata["preview_renderer_sha256"] == PREVIEW_RENDERER_SHA256
    assert metadata["preview_renderer_movement_interaction_alpha_shared"] is True
    assert metadata["preview_renderer_overlap_composition"] == (
        "maximum_source_alpha_per_tile"
    )
    renderer = Path(__file__).resolve().parents[3] / "main.py"
    assert hashlib.sha256(renderer.read_bytes()).hexdigest() == (
        PREVIEW_RENDERER_SHA256
    )


@pytest.mark.parametrize("alpha", (0, 26, 70, 130, 190, 250, 255))
def test_alpha_probability_is_direct_uint8_division(alpha):
    assert alpha_to_detection_probability(alpha) == pytest.approx(alpha / 255)


@pytest.mark.parametrize("alpha", (-1, 256, float("inf"), float("nan")))
def test_alpha_probability_rejects_invalid_values(alpha):
    with pytest.raises(ValueError):
        alpha_to_detection_probability(alpha)


def test_plus_h4_preview_is_exact_and_consumes_no_opacity_draw():
    backend = PreviewBackend(horizon=4)
    backend.preview[3, 0, 0] = 255.0
    runtime = adapter(backend)
    runtime.opacity_rng = FixedRng(1.0)
    runtime.advance_to(60)
    assert runtime.memory.preview4[0, 0, 3] == 255.0
    assert runtime.opacity_rng_draw_count == 0
    assert runtime.events[-1]["opacity_attempt_steps"] == []


def test_spatial_failure_bypasses_opacity_and_preserves_memory():
    backend = PreviewBackend(horizon=2)
    backend.preview[0, 1, 1] = 255.0
    runtime = adapter(backend, mode="local")
    runtime.memory.preview4[1, 1, 0] = 77.0
    runtime.memory.last_observed_time_ms[1, 1] = 0.0
    runtime.peripheral_rng = FixedRng(1.0)
    runtime.opacity_rng = FixedRng(0.0)
    runtime.advance_to(60)
    assert runtime.events[-1]["peripheral_success_indices"] == []
    assert runtime.events[-1]["opacity_attempt_steps"] == []
    assert runtime.opacity_rng.draws == 0
    assert runtime.memory.preview4[1, 1, 0] == 77.0
    assert runtime.memory.last_observed_time_ms[1, 1] == 0.0


@pytest.mark.parametrize(("opacity_draw", "expected"), ((0.0, 255.0), (1.0, 0.0)))
def test_spatial_success_opacity_success_or_miss_writes_perceived_preview(
    opacity_draw, expected,
):
    backend = PreviewBackend(horizon=2)
    backend.preview[0, 1, 1] = 255.0
    runtime = adapter(backend, mode="local")
    runtime.memory.preview4[1, 1, 0] = 77.0
    runtime.memory.last_observed_time_ms[1, 1] = 0.0
    runtime.peripheral_rng = FixedRng(0.0)
    runtime.opacity_rng = FixedRng(opacity_draw)
    runtime.advance_to(60)
    assert runtime.events[-1]["peripheral_success_indices"] == [6]
    assert runtime.events[-1]["opacity_attempt_steps"] == [1]
    assert runtime.memory.preview4[1, 1, 0] == expected
    assert runtime.memory.last_observed_time_ms[1, 1] == 60.0
    assert runtime.memory.state20[1, 1, 0] == backend.canonical_pair()[1][1, 1, 0]


def test_zero_cues_h0_and_h2_upper_steps_consume_no_unnecessary_draws():
    backend = PreviewBackend(horizon=2)
    backend.preview[2:, 1, 1] = 255.0
    runtime = adapter(backend, mode="local")
    runtime.peripheral_rng = FixedRng(0.0)
    runtime.opacity_rng = FixedRng(0.0)
    runtime.advance_to(60)
    assert runtime.opacity_rng.draws == 0
    assert np.count_nonzero(runtime.memory.preview4[1, 1, 2:]) == 0

    h0 = PreviewBackend(horizon=0)
    h0.preview[:, 1, 1] = 255.0
    runtime = adapter(h0, mode="local")
    runtime.peripheral_rng = FixedRng(0.0)
    runtime.opacity_rng = FixedRng(0.0)
    runtime.advance_to(60)
    assert runtime.opacity_rng.draws == 0
    assert np.count_nonzero(runtime.memory.preview4) == 0


def test_sampling_order_is_tile_x_major_with_one_shared_event_per_tile():
    displayed = np.zeros((8, 5, 4), dtype=np.float32)
    displayed[2, 2] = 255.0
    displayed[1, 1] = 255.0
    result = sample_preview_opacity(
        mode="actual", displayed_horizon=4,
        displayed_preview4=displayed,
        peripheral_success_tiles=((2, 2), (1, 1)), rng=FixedRng(0.0),
    )
    assert result.attempt_tile_indices == (6, 12)
    assert result.attempt_steps == (1, 1)


def test_opacity_rng_is_reproducible_independent_and_global_rng_free():
    displayed = np.ones((8, 5, 4), dtype=np.float32) * 255.0
    first = np.random.Generator(np.random.PCG64(51))
    repeated = np.random.Generator(np.random.PCG64(51))
    different = np.random.Generator(np.random.PCG64(52))
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    def sequence(rng):
        return [
            sample_preview_opacity(
                mode="actual", displayed_horizon=4,
                displayed_preview4=displayed,
                peripheral_success_tiles=((1, 1),), rng=rng,
            ).success_mask
            for _ in range(20)
        ]

    first_values = sequence(first)
    assert first_values == sequence(repeated)
    assert first_values != sequence(different)
    assert random.getstate() == python_before
    assert all(
        np.array_equal(left, right)
        for left, right in zip(np.random.get_state(), numpy_before)
    )
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_opacity_draws_do_not_change_spatial_peripheral_rng_outcome():
    backend_off = PreviewBackend(horizon=4)
    backend_off.preview[:, :, :] = 255.0
    off = adapter(backend_off, seed=444, opacity="off")
    backend_actual = PreviewBackend(horizon=4)
    backend_actual.preview[:, :, :] = 255.0
    actual = adapter(backend_actual, seed=444, opacity="actual")
    off.advance_to(60)
    actual.advance_to(60)
    assert off.events[-1]["peripheral_success_mask"] == (
        actual.events[-1]["peripheral_success_mask"]
    )
    assert off.peripheral_rng.bit_generator.state == (
        actual.peripheral_rng.bit_generator.state
    )
    assert off.opacity_rng_draw_count == 0
    assert actual.opacity_rng_draw_count > 0


def test_opacity_off_preserves_preview_and_consumes_zero_draws():
    displayed = np.ones((8, 5, 4), dtype=np.float32) * 255.0
    rng = FixedRng(0.0)
    result = sample_preview_opacity(
        mode="off", displayed_horizon=4,
        displayed_preview4=displayed,
        peripheral_success_tiles=((1, 1),), rng=rng,
    )
    np.testing.assert_array_equal(
        result.perceived_preview4, displayed
    )
    assert result.attempt_steps == ()
    assert rng.draws == 0


def test_plus_overlap_preserves_all_steps_exactly_without_opacity_draw():
    backend = PreviewBackend(horizon=4)
    backend.preview[1, 0, 0] = 255.0
    backend.preview[3, 0, 0] = 255.0
    runtime = adapter(backend)
    runtime.opacity_rng = FixedRng(1.0)
    runtime.advance_to(60)
    assert runtime.memory.preview4[0, 0].tolist() == [0.0, 255.0, 0.0, 255.0]
    assert runtime.opacity_rng.draws == 0


@pytest.mark.parametrize(
    ("opacity_draw", "expected_h2"), ((0.0, 255.0), (1.0, 0.0))
)
def test_peripheral_overlap_uses_one_shared_max_alpha_opacity_draw(
    opacity_draw, expected_h2,
):
    backend = PreviewBackend(horizon=4)
    backend.preview[1, 1, 1] = 255.0
    backend.preview[3, 1, 1] = 255.0
    runtime = adapter(backend, mode="local")
    runtime.peripheral_rng = FixedRng(0.0)
    runtime.opacity_rng = FixedRng(opacity_draw)
    runtime.advance_to(60)
    assert runtime.opacity_rng.draws == 1
    assert runtime.events[-1]["opacity_attempt_steps"] == [2]
    assert runtime.memory.preview4[1, 1].tolist() == [
        0.0, expected_h2, 0.0, expected_h2
    ]
    assert runtime.memory.last_observed_time_ms[1, 1] == 60.0
    assert runtime.events[-1]["opacity_opportunity_positive_steps"] == [[2, 4]]
    assert runtime.events[-1]["opacity_opportunity_max_alphas"] == [190]
    assert runtime.events[-1]["opacity_attempt_probabilities"] == [
        pytest.approx(190 / 255)
    ]
    assert runtime.metrics()["opacity_attempts"] == runtime.metrics()[
        "opacity_preview_tiles"
    ]


def test_all_four_overlapping_steps_share_one_max_alpha_event():
    backend = PreviewBackend(horizon=4)
    backend.preview[:, 1, 1] = 255.0
    runtime = adapter(backend, mode="local")
    runtime.peripheral_rng = FixedRng(0.0)
    runtime.opacity_rng = FixedRng(0.0)
    runtime.advance_to(60)
    assert runtime.opacity_rng.draws == 1
    assert runtime.events[-1]["opacity_opportunity_positive_steps"] == [
        [1, 2, 3, 4]
    ]
    assert runtime.events[-1]["opacity_opportunity_max_alphas"] == [250]
    assert runtime.events[-1]["opacity_attempt_probabilities"] == [
        pytest.approx(250 / 255)
    ]
    assert runtime.memory.preview4[1, 1].tolist() == [255.0] * 4
    assert runtime.metrics()["effective_preview_detections_h4"] == 1


def test_opacity_off_preserves_peripheral_overlap_without_draw():
    backend = PreviewBackend(horizon=4)
    backend.preview[1, 1, 1] = 255.0
    backend.preview[3, 1, 1] = 255.0
    runtime = adapter(backend, mode="local", opacity="off")
    runtime.peripheral_rng = FixedRng(0.0)
    runtime.opacity_rng = FixedRng(0.0)
    runtime.advance_to(60)
    assert runtime.opacity_rng.draws == 0
    assert runtime.events[-1]["opacity_attempt_steps"] == []
    assert runtime.memory.preview4[1, 1].tolist() == [0.0, 255.0, 0.0, 255.0]


@pytest.mark.parametrize(
    "constraint",
    [
        BASELINE_CONSTRAINT_CONFIG,
        config("local", "off"),
        config("global", "off"),
        HumanLikeConstraintConfig(
            profile="custom", peripheral=PeripheralVisionConfig("none"),
            emma=CANONICAL_TUNED_EMMA_TIMING,
            preview_opacity=PreviewOpacityConfig("off"),
        ),
    ],
)
def test_b0_p1_p2_e_off_are_independent_of_opacity_rng(constraint):
    runtimes = [
        PartialObservationAdapter(
            PreviewBackend(horizon=4), decision_time=400,
            fixation_shape="plus", gaze_policy=SameTarget(),
            constraint_config=constraint, peripheral_seed=222,
            opacity_seed=opacity_seed,
        )
        for opacity_seed in (1, 999_999)
    ]
    for runtime in runtimes:
        runtime.advance_to(1_200)
        assert runtime.opacity_rng_draw_count == 0
        assert runtime.metrics()["opacity_attempts"] == 0
    assert list(runtimes[0].events) == list(runtimes[1].events)
    np.testing.assert_array_equal(
        runtimes[0].memory.snapshot(), runtimes[1].memory.snapshot()
    )
    assert runtimes[0].peripheral_rng.bit_generator.state == (
        runtimes[1].peripheral_rng.bit_generator.state
    )


def test_opacity_save_resume_is_exact():
    backend = PreviewBackend(horizon=4)
    backend.preview[:, :, :] = 255.0
    uninterrupted = adapter(backend, seed=90)
    uninterrupted.advance_to(180)
    state = copy.deepcopy(uninterrupted.training_state())
    uninterrupted.advance_to(420)

    resumed_backend = PreviewBackend(horizon=4)
    resumed_backend.preview[:, :, :] = 255.0
    resumed = adapter(resumed_backend, seed=90)
    resumed.load_training_state(state)
    resumed.advance_to(420)
    assert resumed.opacity_rng.bit_generator.state == (
        uninterrupted.opacity_rng.bit_generator.state
    )
    assert resumed.opacity_rng_draw_count == uninterrupted.opacity_rng_draw_count
    assert resumed.metrics() == uninterrupted.metrics()
    assert list(resumed.events) == list(uninterrupted.events)


def test_synthetic_conditional_and_two_stage_probabilities():
    rng = np.random.Generator(np.random.PCG64(20260828))
    count = 300_000
    spatial_probability = 0.64
    opacity_probability = alpha_to_detection_probability(130)
    spatial = rng.random(count) < spatial_probability
    opacity = rng.random(count) < opacity_probability
    assert opacity[spatial].mean() == pytest.approx(opacity_probability, abs=0.004)
    assert np.mean(spatial & opacity) == pytest.approx(
        spatial_probability * opacity_probability, abs=0.004
    )
