import copy
import random

import numpy as np
import pytest
import torch

from final_model.partial_ver.gaze import (
    DEFAULT_EMMA_TIMING,
    DWELL_MS,
    EXEC_BASE_MS,
    EXEC_MS_PER_DEGREE,
    PREP_MS,
    EmmaTimingConfig,
    GazePhase,
    GazeStateMachine,
    HUMAN_DWELL_CAP_MS,
    HUMAN_DWELL_FLOOR_MS,
    HUMAN_STOCHASTIC_EMMA_TIMING,
    ReferenceGeometry,
    emma_timing_from_metadata,
    resolve_emma_timing,
)


class StubDwellRng:
    def __init__(self, *, branch: float, lognormal: float):
        self.branch = float(branch)
        self.lognormal_value = float(lognormal)

    def random(self):
        return self.branch

    def lognormal(self, _mean, _sigma):
        return self.lognormal_value


def test_default_configuration_is_exact_and_immutable():
    assert DEFAULT_EMMA_TIMING.metadata() == {
        "emma_timing_contract_id": "emma_distance_dependent_execution_v1",
        "emma_mode": "default",
        "emma_prep_ms": PREP_MS,
        "emma_exec_intercept_ms": EXEC_BASE_MS,
        "emma_exec_slope_ms_per_degree": EXEC_MS_PER_DEGREE,
        "emma_dwell_ms": DWELL_MS,
    }
    with pytest.raises(Exception):
        DEFAULT_EMMA_TIMING.prep_ms = 1.0


def test_tuned_state_machine_uses_config_without_changing_phase_semantics():
    timing = EmmaTimingConfig(
        mode="tuned", prep_ms=11.0, exec_intercept_ms=13.0,
        exec_slope_ms_per_degree=3.0, dwell_ms=17.0,
    )
    gaze = GazeStateMachine(timing)
    gaze.reset((0, 0), 5.0)
    assert gaze.advance_event().timestamp_ms == 22.0
    gaze.select((0, 0), 22.0)
    same = gaze.advance_event()
    assert same.timestamp_ms == 39.0
    assert same.completed_phase is GazePhase.DWELL
    gaze.select((7, 4), 39.0)
    distance = ReferenceGeometry().visual_angle_degrees((0, 0), (7, 4))
    events = [gaze.advance_event(), gaze.advance_event(), gaze.advance_event()]
    assert [event.completed_phase for event in events] == [
        GazePhase.PREP, GazePhase.EXEC, GazePhase.DWELL
    ]
    assert events[-1].acquisition
    assert events[-1].timestamp_ms == pytest.approx(
        39.0 + 11.0 + 13.0 + 3.0 * distance + 17.0
    )


def test_default_event_timestamps_are_bit_exact_to_legacy_arithmetic():
    gaze = GazeStateMachine()
    now = 123456789.125
    gaze.reset((3, 2), now)
    assert gaze.advance_event().timestamp_ms == now + DWELL_MS
    now += DWELL_MS
    for target in ((7, 4), (0, 0), (3, 2)):
        source = gaze.current_fixation
        gaze.select(target, now)
        prep = gaze.advance_event()
        assert prep.timestamp_ms == now + PREP_MS
        distance = ReferenceGeometry().visual_angle_degrees(source, target)
        execution = gaze.advance_event()
        legacy_execution_completion = (
            prep.timestamp_ms + EXEC_BASE_MS
            + EXEC_MS_PER_DEGREE * distance
        )
        assert execution.timestamp_ms == legacy_execution_completion
        dwell = gaze.advance_event()
        assert dwell.timestamp_ms == legacy_execution_completion + DWELL_MS
        now = dwell.timestamp_ms


def test_resolver_validation_and_legacy_checkpoint_policy():
    assert emma_timing_from_metadata({}) is DEFAULT_EMMA_TIMING
    with pytest.raises(ValueError, match="rejects"):
        resolve_emma_timing("default", prep_ms=1.0)
    with pytest.raises(ValueError, match="all four"):
        resolve_emma_timing("tuned", prep_ms=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        resolve_emma_timing(
            "tuned", prep_ms=1.0, exec_intercept_ms=-1.0,
            exec_slope_ms_per_degree=1.0, dwell_ms=1.0,
        )
    for invalid in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            resolve_emma_timing(
                "tuned", prep_ms=invalid, exec_intercept_ms=1.0,
                exec_slope_ms_per_degree=1.0, dwell_ms=1.0,
            )
    corrupt = copy.deepcopy(DEFAULT_EMMA_TIMING.metadata())
    corrupt["emma_prep_ms"] = 1.0
    with pytest.raises(RuntimeError, match="default EMMA timing mismatch"):
        emma_timing_from_metadata(corrupt)


def test_human_stochastic_timing_is_canonical_and_checkpoint_guarded():
    timing = resolve_emma_timing("human_stochastic")
    assert timing is HUMAN_STOCHASTIC_EMMA_TIMING
    assert emma_timing_from_metadata(timing.metadata()) is timing
    with pytest.raises(ValueError, match="rejects explicit"):
        resolve_emma_timing("human_stochastic", dwell_ms=60.0)
    corrupt = copy.deepcopy(timing.metadata())
    corrupt["emma_dwell_cap_ms"] = 3000.0
    with pytest.raises(RuntimeError, match="human stochastic EMMA timing mismatch"):
        emma_timing_from_metadata(corrupt)


def test_human_stochastic_dwell_is_broad_seeded_and_capped():
    count = 20_000
    gaze = GazeStateMachine(HUMAN_STOCHASTIC_EMMA_TIMING, dwell_seed=123)
    samples = np.asarray([gaze._sample_dwell_ms() for _ in range(count)])
    assert gaze.dwell_sample_count == count
    assert samples.min() == HUMAN_DWELL_FLOOR_MS
    assert samples.max() <= HUMAN_DWELL_CAP_MS
    assert 0.08 < np.mean(samples == HUMAN_DWELL_FLOOR_MS) < 0.13
    assert 280.0 < np.median(samples) < 350.0
    assert 500.0 < np.mean(samples) < 575.0
    assert 1100.0 < np.quantile(samples, 0.9) < 1400.0
    second = GazeStateMachine(HUMAN_STOCHASTIC_EMMA_TIMING, dwell_seed=123)
    assert np.array_equal(
        samples,
        np.asarray([second._sample_dwell_ms() for _ in range(count)]),
    )


@pytest.mark.parametrize(
    ("branch", "raw_positive", "expected_residence"),
    [
        (0.0, 500.0, 60.0),
        (1.0, 30.0, 60.0),
        (1.0, 500.0, 500.0),
        (1.0, 5000.0, 4000.0),
    ],
)
def test_human_stochastic_residual_is_encode_plus_additional_hold(
    branch, raw_positive, expected_residence
):
    gaze = GazeStateMachine(HUMAN_STOCHASTIC_EMMA_TIMING, dwell_seed=1)
    gaze.dwell_rng = StubDwellRng(branch=branch, lognormal=raw_positive)
    gaze.reset((3, 2), now_ms=10.0)
    assert gaze.last_sampled_dwell_ms == expected_residence
    assert gaze.phase is GazePhase.ENCODE
    assert gaze.next_event_ms == 70.0

    encode = gaze.advance_event()
    assert encode.completed_phase is GazePhase.ENCODE
    assert encode.acquisition
    if expected_residence == HUMAN_DWELL_FLOOR_MS:
        assert encode.next_phase is GazePhase.READY
        assert gaze.next_event_ms is None
    else:
        assert encode.next_phase is GazePhase.HOLD
        assert gaze.next_event_ms == 10.0 + expected_residence
        hold = gaze.advance_event()
        assert hold.completed_phase is GazePhase.HOLD
        assert hold.next_phase is GazePhase.READY
        assert hold.timestamp_ms == 10.0 + expected_residence


def test_human_stochastic_different_environment_seeds_are_independent():
    left = GazeStateMachine(HUMAN_STOCHASTIC_EMMA_TIMING, dwell_seed=100)
    same = GazeStateMachine(HUMAN_STOCHASTIC_EMMA_TIMING, dwell_seed=100)
    right = GazeStateMachine(HUMAN_STOCHASTIC_EMMA_TIMING, dwell_seed=101)
    left_values = np.asarray([left._sample_dwell_ms() for _ in range(100)])
    same_values = np.asarray([same._sample_dwell_ms() for _ in range(100)])
    right_values = np.asarray([right._sample_dwell_ms() for _ in range(100)])
    assert np.array_equal(left_values, same_values)
    assert not np.array_equal(left_values, right_values)


def test_human_stochastic_resume_restores_exact_future_dwell_stream():
    timing = HUMAN_STOCHASTIC_EMMA_TIMING
    original = GazeStateMachine(timing, dwell_seed=456)
    original.reset((3, 2), 0.0)
    encoded = original.advance_event()
    assert encoded.completed_phase is GazePhase.ENCODE
    assert encoded.acquisition
    assert encoded.timestamp_ms == HUMAN_DWELL_FLOOR_MS
    state = original.training_state()

    resumed = GazeStateMachine(timing, dwell_seed=456)
    resumed.load_training_state(state)
    now = encoded.timestamp_ms
    for _ in range(40):
        if original.phase is GazePhase.READY:
            original.select(original.current_fixation, now)
            resumed.select(resumed.current_fixation, now)
        left = original.advance_event()
        right = resumed.advance_event()
        assert left == right
        now = left.timestamp_ms
    assert original.training_state() == resumed.training_state()


def test_human_stochastic_requires_seed_but_fixed_state_format_is_unchanged():
    with pytest.raises(ValueError, match="requires dwell_seed"):
        GazeStateMachine(HUMAN_STOCHASTIC_EMMA_TIMING)
    fixed = GazeStateMachine().training_state()
    assert not any(key.startswith("dwell_") for key in fixed)


def test_resolving_timing_consumes_no_global_rng():
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    resolve_emma_timing(
        "tuned", prep_ms=1.0, exec_intercept_ms=2.0,
        exec_slope_ms_per_degree=3.0, dwell_ms=4.0,
    )
    assert random.getstate() == python_before
    assert all(
        np.array_equal(left, right)
        for left, right in zip(np.random.get_state(), numpy_before)
    )
    assert torch.equal(torch.random.get_rng_state(), torch_before)
