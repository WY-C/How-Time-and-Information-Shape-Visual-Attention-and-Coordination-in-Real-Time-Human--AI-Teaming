from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from final_model.common.paths import asset
from final_model.common.resume import capture_global_rng_state, restore_global_rng_state
from final_model.partial_ver.canonical_training import PartialTrainingSession
from final_model.train import parser, validate_combination


PARTNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")
RULE = {"task": 0.60, "stay": 0.20, "stale": 0.15, "random": 0.05}


def _options(**overrides):
    result = dict(
        gaze_type="rule_based",
        train_action=True,
        train_gaze=False,
        decision_times=[200],
        horizons=[0, 2, 4],
        fixation_shapes=["plus"],
        seed=20260825,
        rollout_length=1,
        episode_length=1,
        n_rollout_threads=3,
        device="cpu",
        source_checkpoint=None,
        fixed_partner_checkpoint=PARTNER,
        configured_total_steps=9,
        checkpoint_interval=9,
        reward_shaping_horizon_steps=20_000_000,
        rule_gaze_probabilities=RULE,
        ppo_epochs=1,
        num_mini_batches=1,
    )
    result.update(overrides)
    return result


@pytest.fixture(scope="module")
def predictor_parity(tmp_path_factory):
    output = tmp_path_factory.mktemp("gaze_predictor") / "partial_action.pt"
    source = PartialTrainingSession(**_options())
    source.run_update()
    source.save(output)

    learned = dict(
        gaze_type="learned",
        train_action=False,
        train_gaze=True,
        decision_times=[200, 400, 600],
        horizons=[0, 2, 4],
        fixation_shapes=["plus"],
        n_rollout_threads=9,
        source_checkpoint=output,
        parity_audit=True,
    )
    legacy = PartialTrainingSession(**_options(
        **learned,
        inference_batching="legacy",
    ))
    optimized = PartialTrainingSession(**_options(
        **learned,
        inference_batching="predictor",
        inference_shadow_audit=True,
    ))
    rng = capture_global_rng_state()
    legacy_result = legacy.run_update()
    restore_global_rng_state(rng)
    optimized_result = optimized.run_update()
    return legacy, optimized, legacy_result, optimized_result


def _assert_nested_exact(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    elif isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_exact(first, second)
    else:
        assert left == right


def test_predictor_batching_parity(predictor_parity):
    _legacy, _optimized, legacy_result, optimized_result = predictor_parity
    report = optimized_result["predictor_batch_parity"]
    assert report["env_mapping_exact"]
    assert report["h0_predictor_forward_count"] == 0
    assert report["displayed_action_exact"]
    assert report["displayed_preview_exact"]
    assert report["first_displayed_preview_divergence"] is None
    # Legacy invokes the shared oracle even for H0 and then displays zero planes.
    assert legacy_result["predictor_environment_count"] == 9
    assert optimized_result["predictor_environment_count"] == 6
    assert legacy_result["predictor_forward_count"] == 9
    assert optimized_result["predictor_forward_count"] == 1


def test_gaze_batching_parity_is_fail_closed():
    with pytest.raises(ValueError, match="global Torch RNG"):
        PartialTrainingSession(**_options(
            gaze_type="learned",
            train_action=False,
            train_gaze=True,
            source_checkpoint="unused.pt",
            inference_batching="gaze",
        ))


def test_action_batching_hidden_state_parity_is_fail_closed():
    args = parser().parse_args([
        "--observation-mode", "partial",
        "--gaze-type", "learned",
        "--train-components", "gaze",
        "--horizon", "0",
        "--decision-time", "200",
        "--fixation-shape", "plus",
        "--source-checkpoint", "partial.pt",
        "--fixed-partner-checkpoint", str(PARTNER),
        "--inference-batching", "action",
    ])
    with pytest.raises(ValueError, match="global Torch RNG"):
        validate_combination(args)


def test_gaze_timing_parity(predictor_parity):
    legacy, optimized, _legacy_result, _optimized_result = predictor_parity
    for first, second in zip(legacy.adapters, optimized.adapters):
        assert first.current_time_ms == second.current_time_ms
        assert first.gaze.training_state() == second.gaze.training_state()
        assert first.events == second.events
        assert first.requested_target_counts.tolist() == second.requested_target_counts.tolist()
        assert first.landing_target_counts.tolist() == second.landing_target_counts.tolist()


def test_acquisition_deadline_ordering(predictor_parity):
    legacy, optimized, _legacy_result, _optimized_result = predictor_parity
    for sessions in (legacy.adapters, optimized.adapters):
        for adapter in sessions:
            deadline = float(adapter.decision_time)
            deadline_acquisitions = [
                event
                for event in adapter.events
                if event.get("acquisition") and event["timestamp_ms"] == deadline
            ]
            if deadline_acquisitions:
                assert adapter.actionnet_online_forwards >= len(deadline_acquisitions)


def test_memory_parity(predictor_parity):
    legacy, optimized, _legacy_result, _optimized_result = predictor_parity
    for first, second in zip(legacy.adapters, optimized.adapters):
        _assert_nested_exact(
            first.memory.training_state(),
            second.memory.training_state(),
        )


def test_rollout_parity(predictor_parity):
    legacy, optimized, legacy_result, optimized_result = predictor_parity
    assert legacy.last_parity_audit == optimized.last_parity_audit
    for key in (
        "optimization_reward",
        "sparse_team_reward",
        "native_shaped_reward",
        "weighted_shaped_reward",
        "deliveries",
        "episodes",
        "gaze_requests",
        "acquisitions",
        "same_target_acquisitions",
        "new_target_acquisitions",
        "actionnet_queries",
        "actionnet_forward_count",
        "gaze_forward_count",
        "actor_hash_after",
        "gaze_hash_after",
    ):
        assert legacy_result[key] == optimized_result[key]


def test_ppo_input_parity(predictor_parity):
    _legacy, _optimized, legacy_result, optimized_result = predictor_parity
    for key in (
        "ppo_action_buffer",
        "ppo_gaze_records",
        "ppo_gaze_advantages",
        "ppo_gaze_returns",
        "rng_preupdate",
    ):
        assert legacy_result.get("predictor_batch_parity") is None
        assert predictor_parity[0].last_parity_audit[key] == predictor_parity[1].last_parity_audit[key]
    for result in (legacy_result, optimized_result):
        assert result["gaze_preupdate_parity_pass"] == 1.0
        assert result["gaze_preupdate_max_abs_logprob_difference"] <= 1e-5
        assert result["gaze_preupdate_ratio_min"] == pytest.approx(1.0, abs=1e-5)
        assert result["gaze_preupdate_ratio_max"] == pytest.approx(1.0, abs=1e-5)


def test_episode_reset_parity(predictor_parity):
    legacy, optimized, _legacy_result, _optimized_result = predictor_parity
    assert all(backend.episode_step == 0 for backend in legacy.backends)
    assert all(backend.episode_step == 0 for backend in optimized.backends)
    for first, second in zip(legacy.backends, optimized.backends):
        _assert_nested_exact(first.training_state(), second.training_state())


def test_profile_values_are_finite_nonnegative(predictor_parity):
    _legacy, _optimized, legacy_result, optimized_result = predictor_parity
    keys = (
        "update_wall_seconds",
        "rollout_collection_seconds",
        "gaze_inference_seconds",
        "actionnet_inference_seconds",
        "predictor_inference_seconds",
        "partner_inference_seconds",
        "environment_runtime_seconds",
        "observation_memory_seconds",
        "ppo_rollout_bookkeeping_seconds",
        "preupdate_audit_seconds",
        "ppo_update_seconds",
    )
    for result in (legacy_result, optimized_result):
        assert result["actionnet_forward_count"] == result["actionnet_queries"]
        assert result["gaze_forward_count"] == result["gaze_requests"]
        for key in keys:
            assert math.isfinite(result[key]) and result[key] >= 0.0
