import copy

import numpy as np
import pytest

from final_model.partial_ver.gaze import (
    DWELL_MS,
    EXEC_BASE_MS,
    EXEC_MS_PER_DEGREE,
    PREP_MS,
    GazePhase,
    GazeStateMachine,
    ReferenceGeometry,
    acquisition_tiles,
)
from final_model.partial_ver.memory import (
    CONTINUOUS_FRESHNESS_CONTRACT_ID,
    DYNAMIC_CHANNELS,
    MEMORY_LIFETIME_MS,
    NO_DECAY_FRESHNESS_CONTRACT_ID,
    NO_DECAY_OBSERVATION_CONTRACT_ID,
    OBSERVATION_CONTRACT_ID,
    STATIC_CHANNELS,
    PersistentPartialMemory,
    memory_contract_report,
)
from final_model.partial_ver.rule_based import (
    NEUTRAL_RULE_GAZE_PROBABILITIES,
    RuleBasedGazePolicy,
    validate_rule_probabilities,
)


def state_and_preview():
    state = np.zeros((8, 5, 20), dtype=np.float32)
    state[..., STATIC_CHANNELS] = np.arange(10, 15, dtype=np.float32)
    state[2, 3, 0] = 255.0
    state[2, 3, 15:20] = np.arange(5, dtype=np.float32) + 1
    preview = np.zeros((8, 5, 4), dtype=np.float32)
    preview[2, 3] = (1, 2, 3, 4)
    return state, preview


def test_exact_fixation_shapes_and_clipping():
    assert set(acquisition_tiles((3, 2), "3x3")) == {
        (x, y) for x in (2, 3, 4) for y in (1, 2, 3)
    }
    assert acquisition_tiles((3, 2), "2x2") == (
        (3, 2), (4, 2), (3, 3), (4, 3)
    )
    assert set(acquisition_tiles((3, 2), "plus")) == {
        (3, 2), (3, 1), (3, 3), (2, 2), (4, 2)
    }
    assert acquisition_tiles((7, 4), "2x2") == ((7, 4),)


def test_reset_has_only_static_exact_and_no_self_leakage():
    state, _preview = state_and_preview()
    memory = PersistentPartialMemory()
    observation = memory.reset(state)
    assert np.array_equal(observation[..., STATIC_CHANNELS], state[..., STATIC_CHANNELS])
    assert np.count_nonzero(observation[..., DYNAMIC_CHANNELS]) == 0
    assert np.count_nonzero(observation[..., 20:25]) == 0


@pytest.mark.parametrize(
    ("age_ms", "expected"),
    (
        (0, 1.0), (240, 0.9), (300, 0.875), (600, 0.75),
        (1200, 0.5), (1800, 0.25), (2399, 1.0 / 2400.0),
        (2400, 0.0), (3000, 0.0),
    ),
)
def test_continuous_linear_freshness_2400(age_ms, expected):
    state, preview = state_and_preview()
    memory = PersistentPartialMemory()
    memory.reset(state)
    memory.acquire(state, preview, [(2, 3)], timestamp_ms=0)
    assert memory.semantic_freshness(age_ms)[2, 3] == pytest.approx(expected)
    assert memory.snapshot()[2, 3, 20] == pytest.approx(255.0 * expected)


@pytest.mark.parametrize(
    ("age_ms", "expected"),
    ((0, 1.0), (120, 0.9), (300, 0.75), (600, 0.5), (900, 0.25), (1200, 0.0)),
)
def test_continuous_linear_freshness_1200(age_ms, expected):
    state, preview = state_and_preview()
    memory = PersistentPartialMemory(memory_lifetime_ms=1200)
    memory.reset(state)
    memory.acquire(state, preview, [(2, 3)], timestamp_ms=0)
    assert memory.semantic_freshness(age_ms)[2, 3] == pytest.approx(expected)
    assert memory.snapshot()[2, 3, 20] == pytest.approx(255.0 * expected)


def test_no_decay_keeps_observed_state_fresh_until_episode_reset():
    state, preview = state_and_preview()
    memory = PersistentPartialMemory(
        memory_lifetime_ms=None, memory_decay_mode="none"
    )
    memory.reset(state)
    memory.acquire(state, preview, [(2, 3)], timestamp_ms=0)
    memory.set_time(10_000_000)
    observation = memory.snapshot()
    assert observation[2, 3, 0] == 255.0
    assert observation[2, 3, 20] == 255.0
    assert observation[2, 3, 21:25].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert memory.expiry_count == 0
    reset = memory.reset(state, now_ms=10_000_000)
    assert reset[2, 3, 0] == 0.0
    assert reset[2, 3, 20] == 0.0


def test_no_decay_has_an_explicit_distinct_contract_and_resume_state():
    state, preview = state_and_preview()
    source = PersistentPartialMemory(
        memory_lifetime_ms=None, memory_decay_mode="none"
    )
    source.reset(state)
    source.acquire(state, preview, [(2, 3)], timestamp_ms=100)
    saved = source.training_state()
    assert saved["memory_decay_mode"] == "none"
    assert saved["memory_lifetime_ms"] is None
    assert saved["freshness_contract_id"] == NO_DECAY_FRESHNESS_CONTRACT_ID
    assert saved["observation_contract_id"] == NO_DECAY_OBSERVATION_CONTRACT_ID
    restored = PersistentPartialMemory(
        memory_lifetime_ms=None, memory_decay_mode="none"
    )
    restored.load_training_state(saved)
    assert np.array_equal(restored.snapshot(99_999), source.snapshot(99_999))
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        PersistentPartialMemory().load_training_state(saved)


def test_reacquisition_resets_timestamp_and_expiry_clears_dynamic_not_static():
    state, preview = state_and_preview()
    memory = PersistentPartialMemory()
    memory.reset(state)
    memory.acquire(state, preview, [(2, 3)], timestamp_ms=0)
    memory.set_time(2000)
    assert memory.semantic_freshness()[2, 3] == pytest.approx(1.0 / 6.0)
    changed = state.copy()
    changed[2, 3, 0] = 99
    memory.acquire(changed, preview, [(2, 3)], timestamp_ms=2000)
    assert memory.semantic_freshness()[2, 3] == 1.0
    memory.set_time(4400)
    observation = memory.snapshot()
    assert np.count_nonzero(observation[2, 3, DYNAMIC_CHANNELS]) == 0
    assert observation[2, 3, 20] == 0
    assert np.array_equal(
        observation[2, 3, STATIC_CHANNELS], state[2, 3, STATIC_CHANNELS]
    )
    assert memory.expiry_count == 1


def test_self_memory_expires_and_time_law_is_decision_time_independent():
    state, preview = state_and_preview()
    snapshots = []
    for decision_time in (200, 400, 600):
        memory = PersistentPartialMemory()
        memory.reset(state)
        memory.acquire(state, preview, [(2, 3)], timestamp_ms=0)
        memory.set_time(1800)
        snapshots.append((memory.semantic_freshness()[2, 3], memory.snapshot()[2, 3, 0]))
    assert snapshots == [(0.25, 255.0)] * 3
    memory.set_time(MEMORY_LIFETIME_MS)
    assert memory.snapshot()[2, 3, 0] == 0.0


def test_memory_resume_round_trip_and_old_contract_fails_closed():
    state, preview = state_and_preview()
    source = PersistentPartialMemory()
    source.reset(state)
    source.acquire(state, preview, [(2, 3)], timestamp_ms=100)
    source.set_time(700)
    restored = PersistentPartialMemory()
    restored.load_training_state(source.training_state())
    assert np.array_equal(restored.snapshot(), source.snapshot())
    assert restored.last_observed_time_ms[2, 3] == 100
    legacy = source.training_state()
    legacy["observation_contract_id"] = "decision_step_memory"
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        restored.load_training_state(legacy)


def test_memory_resume_rejects_a_different_configured_lifetime():
    state, preview = state_and_preview()
    source = PersistentPartialMemory(memory_lifetime_ms=2501)
    source.reset(state)
    source.acquire(state, preview, [(2, 3)], timestamp_ms=0)
    restored = PersistentPartialMemory(memory_lifetime_ms=2501)
    restored.load_training_state(source.training_state())
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        PersistentPartialMemory(memory_lifetime_ms=3000).load_training_state(
            source.training_state()
        )


def test_memory_contract_is_continuous_raw255_and_privilege_free():
    report = memory_contract_report()
    assert report["memory_lifetime_ms"] == 2400
    assert report["observation_contract_id"] == OBSERVATION_CONTRACT_ID
    assert report["freshness_contract_id"] == CONTINUOUS_FRESHNESS_CONTRACT_ID
    assert report["raw_freshness"] == "255 * semantic_freshness, float32"
    assert report["always_exact_channels"] == [10, 11, 12, 13, 14]
    assert not report["held_object_flags"]
    assert not report["derived_cooking_ready_flags"]


def test_exact_startup_same_and_new_target_timing():
    gaze = GazeStateMachine()
    gaze.reset((3, 2), 0.0)
    assert gaze.phase is GazePhase.DWELL
    assert gaze.next_event_ms == 60.0
    assert gaze.advance_event().acquisition
    gaze.select((3, 2), 60.0)
    assert gaze.next_event_ms == 60.0 + DWELL_MS
    gaze.advance_event()
    distance = ReferenceGeometry().visual_angle_degrees((3, 2), (7, 4))
    gaze.select((7, 4), 120.0)
    gaze.advance_event()
    assert gaze.next_event_ms == pytest.approx(
        120.0 + PREP_MS + EXEC_BASE_MS + EXEC_MS_PER_DEGREE * distance
    )


def test_rule_probability_validation_and_isolated_reproducible_rng():
    with pytest.raises(ValueError, match="sum"):
        validate_rule_probabilities({"task": .6, "stay": .2, "stale": .1, "random": .05})
    memory = np.zeros((8, 5, 25), dtype=np.float32)
    memory[..., 10] = 1
    np.random.seed(999)
    global_before = np.random.get_state()
    first = RuleBasedGazePolicy("plus", seed=33)
    second = RuleBasedGazePolicy("plus", seed=33)
    trace1 = [first(memory, (3, 2), np.zeros(7)) for _ in range(30)]
    trace2 = [second(memory, (3, 2), np.zeros(7)) for _ in range(30)]
    assert trace1 == trace2
    global_after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(global_before, global_after))


def test_neutral_rule_gaze_preserves_stay_and_removes_semantic_branches():
    memory = np.zeros((8, 5, 25), dtype=np.float32)
    memory[7, 4, 10] = 255
    memory[0, 0, 20] = 255
    policy = RuleBasedGazePolicy(
        "plus", seed=33, probabilities=NEUTRAL_RULE_GAZE_PROBABILITIES,
    )
    for _ in range(200):
        policy(memory, (3, 2), np.zeros(7))
    assert policy.probabilities == {
        "task": 0.0, "stay": 0.2, "stale": 0.0, "random": 0.8,
    }
    assert policy.branch_counts["task"] == 0
    assert policy.branch_counts["stale"] == 0


def test_rule_gaze_uses_memory_only_and_resume_restores_rng():
    memory = np.zeros((8, 5, 25), dtype=np.float32)
    memory[7, 4, 10] = 255
    first = RuleBasedGazePolicy("3x3", seed=91)
    _ = [first(memory, (0, 0), np.zeros(7)) for _ in range(7)]
    state = first.training_state()
    expected = [first(memory, (0, 0), np.zeros(7)) for _ in range(20)]
    restored = RuleBasedGazePolicy("3x3", seed=91)
    restored.load_training_state(copy.deepcopy(state))
    actual = [restored(memory, (0, 0), np.zeros(7)) for _ in range(20)]
    assert actual == expected

    hidden_world_a = {"self": (0, 0), "pot": 0}
    hidden_world_b = {"self": (7, 4), "pot": 20}
    assert hidden_world_a != hidden_world_b
    same_a = RuleBasedGazePolicy("plus", seed=12)
    same_b = RuleBasedGazePolicy("plus", seed=12)
    assert same_a(memory, (2, 2), np.zeros(7)) == same_b(
        memory.copy(), (2, 2), np.zeros(7)
    )
