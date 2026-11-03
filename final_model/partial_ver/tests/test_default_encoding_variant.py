import math
from pathlib import Path

import numpy as np
import pytest

from final_model.partial_ver.constraints import (
    HumanLikeConstraintConfig,
    PeripheralVisionConfig,
    PreviewOpacityConfig,
)
from final_model.partial_ver.gaze import (
    DEFAULT_ENCODING_EMMA_TIMING,
    DEFAULT_ENCODING_FAMILIARITY,
    DEFAULT_ENCODING_K,
    DEFAULT_ENCODING_K_SECONDS,
    RELAXED_SACCADE_EMMA_TIMING,
    RELAXED_SACCADE_EMMA_TIMING_CONTRACT_ID,
    RELAXED_SACCADE_ENCODING_SCALE,
    RELAXED_SACCADE_LATENCY_SCALE,
    GazePhase,
    GazeStateMachine,
    ReferenceGeometry,
    emma_timing_from_metadata,
    visual_encoding_time_ms,
)
from final_model.partial_ver.memory import (
    MEMORY_RNG_CONTRACT_ID,
    PersistentPartialMemory,
    STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID,
)
from final_model.partial_ver.observation import (
    DEADLINE_ACTION_CONTRACT_ID,
    DEADLINE_RECURRENT_COMMIT_CONTRACT_ID,
    PartialObservationAdapter,
)
from final_model.partial_ver.peripheral import (
    DIAG3X3_PERIPHERAL_OFFSETS,
    DIAMOND13_PERIPHERAL_OFFSETS,
    PLUS_EXACT_OFFSETS,
    extent_peripheral_candidates,
    peripheral_detection_probability,
)
from final_model.partial_ver.tests.test_peripheral_runtime import TileBackend
from final_model.partial_ver.tests.test_runtime import SameTarget
from final_model.train import parser, validate_combination
from final_model.server_parallel.protocol import MAX_ACQUISITIONS_PER_DECISION


PARTNER = (
    Path(__file__).resolve().parents[3]
    / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
)


def _state_and_preview():
    state = np.zeros((8, 5, 20), dtype=np.float32)
    state[..., 10:15] = np.arange(5, dtype=np.float32)
    state[2, 3, 0] = 255.0
    preview = np.zeros((8, 5, 4), dtype=np.float32)
    return state, preview


def _memory(maximum, tau, seed=7):
    state, _ = _state_and_preview()
    memory = PersistentPartialMemory(
        memory_decay_mode="stochastic_exponential",
        memory_lifetime_ms=maximum,
        memory_tau_ms=tau,
        memory_seed=seed,
    )
    memory.reset(state)
    return memory


def _variant_config(extent, emma=DEFAULT_ENCODING_EMMA_TIMING):
    return HumanLikeConstraintConfig(
        profile="custom",
        peripheral=PeripheralVisionConfig("local", 3.0, extent=extent),
        emma=emma,
        preview_opacity=PreviewOpacityConfig("off"),
    )


def test_factorial_peripheral_geometry_and_boundary_clipping():
    assert set(PLUS_EXACT_OFFSETS) == {
        (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)
    }
    assert len(PLUS_EXACT_OFFSETS) == len(set(PLUS_EXACT_OFFSETS)) == 5
    assert len(DIAG3X3_PERIPHERAL_OFFSETS) == (
        len(set(DIAG3X3_PERIPHERAL_OFFSETS))
    ) == 4
    assert len(DIAMOND13_PERIPHERAL_OFFSETS) == (
        len(set(DIAMOND13_PERIPHERAL_OFFSETS))
    ) == 8
    assert set(PLUS_EXACT_OFFSETS).isdisjoint(DIAG3X3_PERIPHERAL_OFFSETS)
    assert set(PLUS_EXACT_OFFSETS).isdisjoint(DIAMOND13_PERIPHERAL_OFFSETS)
    assert len(set(PLUS_EXACT_OFFSETS) | set(DIAG3X3_PERIPHERAL_OFFSETS)) == 9
    assert len(set(PLUS_EXACT_OFFSETS) | set(DIAMOND13_PERIPHERAL_OFFSETS)) == 13
    assert set(extent_peripheral_candidates((0, 0), "diag3x3")) == {(1, 1)}
    assert set(extent_peripheral_candidates((0, 0), "diamond13")) == {
        (2, 0), (1, 1), (0, 2)
    }


def test_extent_probability_is_shared_monotonic_and_plus_is_not_a_candidate():
    theta = np.linspace(0.0, 20.0, 100)
    probability = peripheral_detection_probability(theta, 3.0)
    assert np.all(np.diff(probability) <= 0.0)
    fixation = (3, 2)
    geometry = ReferenceGeometry()
    shared = geometry.visual_angle_degrees(fixation, (4, 3))
    for extent in ("diag3x3", "diamond13"):
        assert peripheral_detection_probability(shared, 3.0) == pytest.approx(
            math.exp(-(shared ** 2) / (2.0 * 3.0 ** 2))
        )
        candidates = set(extent_peripheral_candidates(fixation, extent))
        plus = {(3, 2), (2, 2), (4, 2), (3, 1), (3, 3)}
        assert candidates.isdisjoint(plus)


def test_default_encoding_formula_fixed_familiarity_and_metadata():
    expected_zero = (
        1000.0 * DEFAULT_ENCODING_K_SECONDS
        * -math.log(DEFAULT_ENCODING_FAMILIARITY)
    )
    assert visual_encoding_time_ms(0.0) == pytest.approx(expected_zero)
    assert visual_encoding_time_ms(5.0) > visual_encoding_time_ms(1.0)
    assert visual_encoding_time_ms(3.0) == visual_encoding_time_ms(3.0)
    metadata = DEFAULT_ENCODING_EMMA_TIMING.metadata()
    assert metadata["encoding_K"] == DEFAULT_ENCODING_K_SECONDS
    assert metadata["encoding_k"] == DEFAULT_ENCODING_K
    assert metadata["encoding_f"] == DEFAULT_ENCODING_FAMILIARITY
    assert "semantic" not in metadata["emma_encoding_temporal_semantics"]


def test_default_encoding_runs_in_parallel_with_canonical_movement():
    gaze = GazeStateMachine(DEFAULT_ENCODING_EMMA_TIMING)
    gaze.reset((0, 0), 0.0)
    first = gaze.advance_event()
    assert first.completed_phase is GazePhase.ENCODE
    assert first.timestamp_ms == pytest.approx(visual_encoding_time_ms(0.0))
    start = first.timestamp_ms
    target = (7, 4)
    distance = ReferenceGeometry().visual_angle_degrees((0, 0), target)
    initial_encoding = visual_encoding_time_ms(distance)
    movement = 135.0 + 70.0 + 2.0 * distance
    gaze.select(target, start)
    assert gaze.advance_event().completed_phase is GazePhase.PREP
    landing = gaze.advance_event()
    assert landing.timestamp_ms == pytest.approx(start + movement)
    remaining = max(0.0, 1.0 - movement / initial_encoding) * (
        visual_encoding_time_ms(0.0)
    )
    if remaining:
        assert not landing.acquisition
        encoded = gaze.advance_event()
        assert encoded.timestamp_ms == pytest.approx(start + movement + remaining)
        assert encoded.acquisition
    else:
        assert landing.acquisition


def test_relaxed_saccade_halves_movement_and_encoding_and_is_strictly_tagged():
    relaxed = RELAXED_SACCADE_EMMA_TIMING
    default = DEFAULT_ENCODING_EMMA_TIMING
    assert relaxed.prep_ms == default.prep_ms * RELAXED_SACCADE_LATENCY_SCALE
    assert relaxed.exec_intercept_ms == (
        default.exec_intercept_ms * RELAXED_SACCADE_LATENCY_SCALE
    )
    assert relaxed.exec_slope_ms_per_degree == (
        default.exec_slope_ms_per_degree * RELAXED_SACCADE_LATENCY_SCALE
    )
    assert relaxed.encoding_K_seconds == (
        default.encoding_K_seconds * RELAXED_SACCADE_ENCODING_SCALE
    )
    assert relaxed.encoding_k == default.encoding_k
    assert relaxed.encoding_f == default.encoding_f

    metadata = relaxed.metadata()
    assert metadata["emma_timing_contract_id"] == (
        RELAXED_SACCADE_EMMA_TIMING_CONTRACT_ID
    )
    assert metadata["saccade_latency_scale"] == 0.5
    assert metadata["encoding_latency_scale"] == 0.5
    assert emma_timing_from_metadata(metadata) is relaxed
    with pytest.raises(RuntimeError, match="relaxed-saccade.*mismatch"):
        emma_timing_from_metadata({**metadata, "emma_prep_ms": 67.6})


def test_relaxed_resolved_component_values_are_exact():
    bounded = DEFAULT_ENCODING_EMMA_TIMING
    relaxed = RELAXED_SACCADE_EMMA_TIMING
    assert bounded.prep_ms == 135.0
    assert relaxed.prep_ms == 67.5
    assert bounded.execution_completion_ms(0.0, 0.0) == 70.0
    assert relaxed.execution_completion_ms(0.0, 0.0) == 35.0
    assert bounded.execution_completion_ms(0.0, 4.0) == 78.0
    assert relaxed.execution_completion_ms(0.0, 4.0) == 39.0
    for epsilon in (0.0, 1.5, 4.0, 16.0):
        assert relaxed.encoding_K_seconds == 0.003
        assert visual_encoding_time_ms(
            epsilon, K_seconds=relaxed.encoding_K_seconds
        ) == pytest.approx(
            0.5 * visual_encoding_time_ms(
                epsilon, K_seconds=bounded.encoding_K_seconds
            )
        )

    bounded_gaze = GazeStateMachine(bounded)
    relaxed_gaze = GazeStateMachine(relaxed)
    bounded_gaze.reset((3, 2), 0.0)
    relaxed_gaze.reset((3, 2), 0.0)
    bounded_ready = bounded_gaze.advance_event().timestamp_ms
    relaxed_ready = relaxed_gaze.advance_event().timestamp_ms
    bounded_gaze.select((3, 2), bounded_ready)
    relaxed_gaze.select((3, 2), relaxed_ready)
    assert (
        relaxed_gaze.advance_event().timestamp_ms - relaxed_ready
    ) == pytest.approx(
        0.5 * (bounded_gaze.advance_event().timestamp_ms - bounded_ready)
    )


@pytest.mark.parametrize("target", ((1, 0), (7, 4)))
@pytest.mark.parametrize(
    "timing", (DEFAULT_ENCODING_EMMA_TIMING, RELAXED_SACCADE_EMMA_TIMING)
)
def test_parallel_remaining_work_covers_encoding_before_and_after_landing(
    target, timing,
):
    gaze = GazeStateMachine(timing)
    gaze.reset((0, 0), 0.0)
    start = gaze.advance_event().timestamp_ms
    gaze.select(target, start)
    distance = gaze.last_distance_degrees
    initial_encoding = gaze.initial_encoding_ms
    movement = (
        timing.prep_ms + timing.exec_intercept_ms
        + timing.exec_slope_ms_per_degree * distance
    )
    assert (initial_encoding > movement) == (target == (7, 4))

    gaze.advance_event()
    landing = gaze.advance_event()
    expected_remaining = max(0.0, 1.0 - movement / initial_encoding) * (
        gaze.visual_encoding_time_ms(0.0)
    )
    assert gaze.last_encoding_ms == pytest.approx(expected_remaining)
    if expected_remaining == 0.0:
        assert landing.acquisition
        assert gaze.phase is GazePhase.READY
    else:
        assert not landing.acquisition
        assert gaze.phase is GazePhase.ENCODE
        encoded = gaze.advance_event()
        assert encoded.timestamp_ms == pytest.approx(
            landing.timestamp_ms + expected_remaining
        )
        assert encoded.acquisition


def test_relaxed_gaze_process_continues_across_decision_boundary():
    gaze = GazeStateMachine(RELAXED_SACCADE_EMMA_TIMING)
    gaze.reset((0, 0), 0.0)
    selection_time = gaze.advance_event().timestamp_ms
    gaze.select((7, 4), selection_time)
    boundary = selection_time + 50.0

    assert gaze.phase is GazePhase.PREP
    assert gaze.next_event_ms > boundary
    assert gaze.current_fixation == (0, 0)
    prep = gaze.advance_event()
    assert prep.timestamp_ms == pytest.approx(selection_time + 67.5)
    assert prep.completed_phase is GazePhase.PREP
    assert gaze.phase is GazePhase.EXEC
    landing = gaze.advance_event()
    assert landing.timestamp_ms > boundary
    assert gaze.current_fixation == (7, 4)


def test_relaxed_changes_only_gaze_timing_in_bounded_runtime_config():
    bounded = PartialObservationAdapter(
        TileBackend(), decision_time=400, fixation_shape="plus",
        gaze_policy=SameTarget(),
        constraint_config=_variant_config(
            "diamond13", DEFAULT_ENCODING_EMMA_TIMING
        ),
        peripheral_seed=11,
        memory_decay_mode="stochastic_exponential",
        memory_lifetime_ms=2400.0, memory_tau_ms=800.0, memory_seed=12,
    )
    relaxed = PartialObservationAdapter(
        TileBackend(), decision_time=400, fixation_shape="plus",
        gaze_policy=SameTarget(),
        constraint_config=_variant_config(
            "diamond13", RELAXED_SACCADE_EMMA_TIMING
        ),
        peripheral_seed=11,
        memory_decay_mode="stochastic_exponential",
        memory_lifetime_ms=2400.0, memory_tau_ms=800.0, memory_seed=12,
    )

    assert bounded.constraint_config.peripheral == (
        relaxed.constraint_config.peripheral
    )
    assert bounded.constraint_config.preview_opacity == (
        relaxed.constraint_config.preview_opacity
    )
    assert bounded.constraint_config.peripheral.sigma_deg == 3.0
    assert bounded.constraint_config.peripheral.extent == "diamond13"
    assert bounded.memory.memory_tau_ms == relaxed.memory.memory_tau_ms == 800.0
    assert (
        bounded.memory.memory_lifetime_ms
        == relaxed.memory.memory_lifetime_ms == 2400.0
    )


def test_relaxed_saccade_preserves_parallel_encoding_semantics():
    default = GazeStateMachine(DEFAULT_ENCODING_EMMA_TIMING)
    relaxed = GazeStateMachine(RELAXED_SACCADE_EMMA_TIMING)
    target = (7, 4)
    default.reset((0, 0), 0.0)
    relaxed.reset((0, 0), 0.0)
    default_initial = default.advance_event()
    relaxed_initial = relaxed.advance_event()
    assert relaxed_initial.timestamp_ms == pytest.approx(
        default_initial.timestamp_ms * 0.5
    )

    default.select(target, default_initial.timestamp_ms)
    relaxed.select(target, relaxed_initial.timestamp_ms)

    def acquisition_time(gaze):
        while True:
            event = gaze.advance_event()
            if event.acquisition:
                return event.timestamp_ms

    default_acquisition = acquisition_time(default)
    relaxed_acquisition = acquisition_time(relaxed)
    assert relaxed_acquisition - relaxed_initial.timestamp_ms == pytest.approx(
        (default_acquisition - default_initial.timestamp_ms) * 0.5
    )


@pytest.mark.parametrize(
    "emma", (DEFAULT_ENCODING_EMMA_TIMING, RELAXED_SACCADE_EMMA_TIMING)
)
def test_server_acquisition_capacity_covers_fastest_encoding_at_600ms(emma):
    gaze = GazeStateMachine(emma)
    gaze.reset((3, 2), 0.0)
    acquisitions = 0
    while gaze.next_event_ms is not None and gaze.next_event_ms <= 600.0:
        event = gaze.advance_event()
        if event.acquisition:
            acquisitions += 1
            gaze.select(gaze.current_fixation, event.timestamp_ms)
    assert acquisitions > 10
    assert acquisitions <= MAX_ACQUISITIONS_PER_DECISION


@pytest.mark.parametrize("decision_time", (200, 400, 600))
@pytest.mark.parametrize(
    "emma", (DEFAULT_ENCODING_EMMA_TIMING, RELAXED_SACCADE_EMMA_TIMING)
)
def test_visual_encoding_modes_query_action_once_at_decision_deadline(
    decision_time, emma,
):
    adapter = PartialObservationAdapter(
        TileBackend(),
        decision_time=decision_time,
        fixation_shape="plus",
        gaze_policy=SameTarget(),
        constraint_config=_variant_config("diag3x3", emma),
        peripheral_seed=11,
        memory_decay_mode="stochastic_exponential",
        memory_lifetime_ms=2400.0,
        memory_tau_ms=800.0,
        memory_seed=12,
    )
    calls = []

    def action_at_deadline(observation, _critic, available, timestamp):
        calls.append((timestamp, observation.copy(), available.copy()))
        return {"action": 2, "candidate_hidden": np.asarray([timestamp])}

    collected = adapter.collect_decision(action_at_deadline)

    assert collected["acquisitions"] >= 1
    assert len(calls) == len(collected["proposals"]) == 1
    assert calls[0][0] == collected["deadline_ms"] == float(decision_time)
    assert np.array_equal(calls[0][1], collected["observation"])
    assert np.array_equal(calls[0][2], collected["available_actions"])
    assert collected["latest_proposal"]["action_query_time_ms"] == float(
        decision_time
    )
    assert not collected["forced_stay"]
    assert adapter.actionnet_online_forwards == 1
    assert adapter.action_query_contract_id == DEADLINE_ACTION_CONTRACT_ID
    assert (
        adapter.recurrent_commit_contract_id
        == DEADLINE_RECURRENT_COMMIT_CONTRACT_ID
    )


@pytest.mark.parametrize(("maximum", "tau"), ((1200.0, 400.0), (2400.0, 800.0)))
def test_stochastic_memory_survival_distribution_and_hard_cutoff(maximum, tau):
    state, preview = _state_and_preview()
    memory = _memory(maximum, tau, seed=12345)
    samples = []
    for index in range(30_000):
        memory.acquire(state, preview, [(2, 3)], timestamp_ms=float(index))
        samples.append(memory.sampled_lifetime_ms[2, 3])
    samples = np.asarray(samples)
    assert np.mean(samples > tau) == pytest.approx(math.exp(-1.0), abs=0.012)
    assert np.mean(samples > 2.0 * tau) == pytest.approx(math.exp(-2.0), abs=0.009)
    assert np.mean(np.minimum(samples, maximum) >= maximum) == pytest.approx(
        math.exp(-3.0), abs=0.006
    )

    cutoff = _memory(maximum, tau, seed=4)
    cutoff.acquire(state, preview, [(2, 3)], timestamp_ms=0.0)
    cutoff.actual_lifetime_ms[2, 3] = maximum
    cutoff.sampled_lifetime_ms[2, 3] = maximum + 1.0
    assert cutoff.semantic_freshness(maximum - 1.0)[2, 3] == 1.0
    assert cutoff.snapshot()[2, 3, 20] == 255.0
    assert cutoff.semantic_freshness(maximum)[2, 3] == 0.0
    assert cutoff.snapshot()[2, 3, 20] == 0.0
    assert cutoff.hard_cutoff_deletion_count == 1


def test_stochastic_memory_freshness_is_binary_until_sampled_deletion():
    state, preview = _state_and_preview()
    memory = _memory(1200.0, 400.0, seed=17)
    memory.acquire(state, preview, [(2, 3)], timestamp_ms=0.0)
    memory.sampled_lifetime_ms[2, 3] = 125.0
    memory.actual_lifetime_ms[2, 3] = 125.0

    assert memory.semantic_freshness(124.999)[2, 3] == 1.0
    assert memory.snapshot()[2, 3, 20] == 255.0
    assert memory.snapshot()[2, 3, 0] == state[2, 3, 0]

    assert memory.semantic_freshness(125.0)[2, 3] == 0.0
    assert memory.snapshot()[2, 3, 20] == 0.0
    assert memory.snapshot()[2, 3, 0] == 0.0
    assert memory.stochastic_deletion_count == 1


def test_stochastic_memory_reacquisition_seeded_resume_and_mode_isolation():
    state, preview = _state_and_preview()
    first = _memory(1200.0, 400.0, seed=99)
    first.acquire(state, preview, [(2, 3)], timestamp_ms=0.0)
    old_lifetime = first.sampled_lifetime_ms[2, 3]
    first.acquire(state, preview, [(2, 3)], timestamp_ms=10.0)
    assert first.last_observed_time_ms[2, 3] == 10.0
    assert first.semantic_freshness()[2, 3] == 1.0
    assert first.snapshot()[2, 3, 20] == 255.0
    assert first.lifetime_sample_count == 2
    assert first.sampled_lifetime_ms[2, 3] != old_lifetime
    saved = first.training_state()
    assert saved["memory_rng_contract_id"] == MEMORY_RNG_CONTRACT_ID

    restored = _memory(1200.0, 400.0, seed=99)
    restored.load_training_state(saved)
    for timestamp in (20.0, 30.0, 40.0):
        first.acquire(state, preview, [(2, 3)], timestamp_ms=timestamp)
        restored.acquire(state, preview, [(2, 3)], timestamp_ms=timestamp)
        assert restored.sampled_lifetime_ms[2, 3] == (
            first.sampled_lifetime_ms[2, 3]
        )
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        _memory(2400.0, 800.0, seed=99).load_training_state(saved)


@pytest.mark.parametrize(("extent", "maximum", "tau"), [
    ("diag3x3", 1200.0, 400.0),
    ("diag3x3", 2400.0, 800.0),
    ("diamond13", 1200.0, 400.0),
    ("diamond13", 2400.0, 800.0),
])
def test_four_factorial_configs_reach_adapter(extent, maximum, tau):
    runtime = PartialObservationAdapter(
        TileBackend(),
        decision_time=200,
        fixation_shape="plus",
        gaze_policy=SameTarget(),
        constraint_config=_variant_config(extent),
        peripheral_seed=11,
        memory_decay_mode="stochastic_exponential",
        memory_lifetime_ms=maximum,
        memory_tau_ms=tau,
        memory_seed=12,
    )
    runtime.gaze.reset((3, 2), 0.0)
    runtime.advance_to(200.0)
    assert runtime.constraint_config.peripheral.extent == extent
    assert runtime.memory.memory_lifetime_ms == maximum
    assert runtime.memory.memory_tau_ms == tau
    event = next(row for row in runtime.events if row.get("acquisition"))
    assert len(event["foveal_tile_indices"]) == 5
    assert "foveal_encoding_time_ms" in event


def test_factorial_cli_records_independent_extent_tau_and_maximum():
    args = parser().parse_args([
        "--observation-mode", "partial",
        "--gaze-type", "rule_based",
        "--train-components", "action",
        "--horizon", "0",
        "--decision-time", "200",
        "--fixation-shape", "plus",
        "--steps", "2",
        "--rollout-length", "2",
        "--n-rollout-threads", "1",
        "--fixed-partner-checkpoint", str(PARTNER),
        "--memory-decay-mode", "stochastic_exponential",
        "--memory-tau-ms", "400",
        "--memory-max-lifetime-ms", "1200",
        "--constraint-profile", "custom",
        "--peripheral-mode", "local",
        "--peripheral-sigma-deg", "3",
        "--peripheral-extent", "diamond13",
        "--emma-mode", "default_encoding",
        "--preview-opacity-mode", "off",
    ])
    validate_combination(args)
    assert args.memory_lifetime_ms == args.memory_max_lifetime_ms == 1200.0
    assert args.memory_tau_ms == 400.0
    assert args.peripheral_extent == "diamond13"
    assert args.emma_mode == "default_encoding"
    assert args._resolved_constraint_config.metadata()["peripheral_extent"] == (
        "diamond13"
    )
    relaxed = parser().parse_args([
        "--observation-mode", "partial",
        "--gaze-type", "rule_based",
        "--train-components", "action",
        "--horizon", "0",
        "--decision-time", "200",
        "--fixation-shape", "plus",
        "--steps", "2",
        "--rollout-length", "2",
        "--n-rollout-threads", "1",
        "--fixed-partner-checkpoint", str(PARTNER),
        "--memory-decay-mode", "stochastic_exponential",
        "--memory-tau-ms", "400",
        "--memory-max-lifetime-ms", "1200",
        "--constraint-profile", "custom",
        "--peripheral-mode", "local",
        "--peripheral-sigma-deg", "3",
        "--peripheral-extent", "diamond13",
        "--emma-mode", "relaxed_saccade",
        "--preview-opacity-mode", "off",
    ])
    validate_combination(relaxed)
    assert relaxed.emma_mode == "relaxed_saccade"
    assert relaxed.emma_prep_ms == 67.5
    assert relaxed.emma_exec_intercept_ms == 35.0
    assert relaxed.emma_exec_slope_ms_per_degree == 1.0
    assert relaxed._resolved_constraint_config.emma is (
        RELAXED_SACCADE_EMMA_TIMING
    )
    with pytest.raises(ValueError, match="tau = max lifetime / 3"):
        bad = parser().parse_args([
            "--observation-mode", "partial", "--gaze-type", "rule_based",
            "--train-components", "action", "--horizon", "0",
            "--decision-time", "200", "--fixation-shape", "plus",
            "--steps", "2", "--rollout-length", "2",
            "--n-rollout-threads", "1", "--fixed-partner-checkpoint", str(PARTNER),
            "--memory-decay-mode", "stochastic_exponential",
            "--memory-tau-ms", "401", "--memory-max-lifetime-ms", "1200",
        ])
        validate_combination(bad)
