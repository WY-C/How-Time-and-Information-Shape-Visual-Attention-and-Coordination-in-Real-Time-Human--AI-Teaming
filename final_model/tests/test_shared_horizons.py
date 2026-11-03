from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from final_model.checkpoint import initialize_partial_from_full
from final_model.common.action_training import ActionTrainingSession
from final_model.common.mappo import build_actionnet
from final_model.common.partners import parameter_hash
from final_model.common.paths import asset
from final_model.common.shared_horizons import (
    BalancedEpisodeHorizonSampler,
    SHARED_HORIZONS,
    horizon_one_hot,
)
from final_model.simple_ver.observation import build_full_observation
from final_model.train import _wandb_metrics, parser, validate_combination


LEARNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt")
PARTNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")


def _cli(*extra):
    return [
        "--observation-mode", "full",
        "--training-stage", "action",
        "--shared-horizons", "0", "2", "4",
        "--hsp-initial-checkpoint", str(LEARNER),
        "--fixed-partner-checkpoint", str(PARTNER),
        *extra,
    ]


def _session(
    *, seed=73, rollout_length=3, episode_length=2, n_rollout_threads=3,
    resume_checkpoint: Path | None = None,
):
    return ActionTrainingSession(
        observation_mode="full",
        horizon=None,
        shared_horizons=SHARED_HORIZONS,
        decision_time=None,
        fixation_shape=None,
        seed=seed,
        rollout_length=rollout_length,
        episode_length=episode_length,
        device="cpu",
        source_checkpoint=None,
        resume_checkpoint=resume_checkpoint,
        hsp_initial_checkpoint=None if resume_checkpoint else LEARNER,
        fixed_partner_checkpoint=PARTNER,
        n_rollout_threads=n_rollout_threads,
        inference_batching="predictor",
        ppo_epochs=1,
        num_mini_batches=1,
    )


def test_fixed_cli_unchanged_and_shared_cli_is_additive():
    for horizon in SHARED_HORIZONS:
        fixed = parser().parse_args(
            [
                "--observation-mode", "full", "--training-stage", "action",
                "--horizon", str(horizon), "--hsp-initial-checkpoint", str(LEARNER),
                "--fixed-partner-checkpoint", str(PARTNER),
            ]
        )
        validate_combination(fixed)
        assert fixed.horizon == horizon and fixed.shared_horizons is None
    shared = parser().parse_args(_cli())
    validate_combination(shared)
    assert shared.horizon is None
    assert shared.shared_horizons == [0, 2, 4]
    assert shared.inference_batching == "predictor"
    reordered = parser().parse_args(
        ["--observation-mode", "full", "--training-stage", "action",
         "--shared-horizons", "4", "0", "2",
         "--hsp-initial-checkpoint", str(LEARNER),
         "--fixed-partner-checkpoint", str(PARTNER)]
    )
    validate_combination(reordered)
    assert reordered.shared_horizons == [0, 2, 4]


@pytest.mark.parametrize("steps", (120_000, 1_200_000, 20_000_000, 60_000_000, 120_000_000))
def test_shared_budget_uses_generic_steps_without_schedule_reinterpretation(steps):
    args = parser().parse_args(_cli("--steps", str(steps)))
    validate_combination(args)
    assert args.steps == steps
    assert args.entropy_coefs == [0.2, 0.05, 0.01]
    assert args.entropy_coef_horizons == [0, 10_000_000, 20_000_000]


def test_cli_rejects_mixed_invalid_and_partial_shared_modes():
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                "--observation-mode", "full", "--training-stage", "action",
                "--horizon", "0", "--shared-horizons", "0", "2", "4",
            ]
        )
    invalid_set = parser().parse_args(
        ["--observation-mode", "full", "--training-stage", "action",
         "--shared-horizons", "0", "4"]
    )
    with pytest.raises(ValueError, match="exactly"):
        validate_combination(invalid_set)
    partial = parser().parse_args(
        ["--observation-mode", "partial", "--training-stage", "action",
         "--shared-horizons", "0", "2", "4",
         "--decision-time", "200", "--fixation-shape", "3x3",
         "--fixed-partner-checkpoint", str(PARTNER)]
    )
    validate_combination(partial)
    assert partial.partial_horizons == [0, 2, 4]


def test_balanced_sampler_blocks_counts_seed_and_resume():
    first = BalancedEpisodeHorizonSampler(seed=91)
    sequence = [first.next() for _ in range(300)]
    assert all(set(sequence[start:start + 3]) == set(SHARED_HORIZONS)
               for start in range(0, 300, 3))
    assert {value: sequence.count(value) for value in SHARED_HORIZONS} == {
        0: 100, 2: 100, 4: 100,
    }
    same = BalancedEpisodeHorizonSampler(seed=91)
    assert sequence == [same.next() for _ in range(300)]

    uninterrupted = BalancedEpisodeHorizonSampler(seed=92)
    prefix = [uninterrupted.next() for _ in range(17)]
    state = uninterrupted.state_dict()
    expected = [uninterrupted.next() for _ in range(40)]
    resumed = BalancedEpisodeHorizonSampler(seed=92)
    resumed.load_state_dict(state)
    assert [resumed.next() for _ in range(40)] == expected
    assert len(prefix) == 17


@pytest.mark.parametrize("horizon", SHARED_HORIZONS)
def test_canonical_mask_and_shared_fixed_observation_parity(horizon):
    state = np.arange(8 * 5 * 20, dtype=np.float32).reshape(8, 5, 20)
    planes = np.arange(1, 4 * 8 * 5 + 1, dtype=np.float32).reshape(4, 8, 5)
    fixed = build_full_observation(state, planes, horizon)
    shared = build_full_observation(state, planes, horizon)
    assert np.array_equal(shared, fixed)
    assert np.array_equal(horizon_one_hot(horizon), np.eye(3, dtype=np.float32)[SHARED_HORIZONS.index(horizon)])
    assert np.all(shared[..., 21 + horizon:25] == 0.0)
    if horizon:
        assert np.array_equal(shared[..., 21:21 + horizon], np.moveaxis(planes[:horizon], 0, -1))


def test_horizon_is_per_env_immutable_until_reset_and_rows_do_not_leak():
    session = _session(episode_length=2)
    try:
        initial = session.active_horizons.copy()
        assert set(initial) == set(SHARED_HORIZONS)
        first_obs = session._prepared[0]
        for env_index, horizon in enumerate(initial):
            assert np.all(first_obs[env_index, ..., 21 + horizon:25] == 0.0)
        session._commit_decisions(np.full((3, 1), 4, dtype=np.int64))
        session._prepare_decisions()
        assert np.array_equal(session.active_horizons, initial)
        results = session._commit_decisions(np.full((3, 1), 4, dtype=np.int64))
        assert all(result["done"] for result in results)
        for env_index in range(3):
            session.adapters[env_index].reset_episode()
            session._assign_next_horizon(env_index)
        assert not np.array_equal(session.active_horizons, initial)
        session._prepare_decisions()
        assert np.array_equal(
            session.active_horizon_one_hot,
            np.stack([horizon_one_hot(value) for value in session.active_horizons]),
        )
    finally:
        session._executor.shutdown(wait=True)


def test_shared_initial_hashes_match_fixed_full_initialization():
    shared = _session(seed=74)
    fixed = ActionTrainingSession(
        observation_mode="full", horizon=4, decision_time=None, fixation_shape=None,
        seed=74, rollout_length=3, episode_length=2, device="cpu", source_checkpoint=None,
        hsp_initial_checkpoint=LEARNER, fixed_partner_checkpoint=PARTNER,
        n_rollout_threads=3, inference_batching="predictor", ppo_epochs=1,
        num_mini_batches=1,
    )
    try:
        assert parameter_hash(shared.policy.actor) == parameter_hash(fixed.policy.actor)
        assert parameter_hash(shared.policy.critic) == parameter_hash(fixed.policy.critic)
    finally:
        shared._executor.shutdown(wait=True)
        fixed._executor.shutdown(wait=True)


def test_one_shared_ppo_metrics_partner_freeze_and_aggregates():
    session = _session()
    partner_before = session.backend.partners.frozen_parameter_hash
    try:
        result = session.run_update()
        buckets = result["shared_horizon_metrics"]
        assert all(result["ppo_sample_counts_by_horizon"][key] > 0 for key in buckets)
        assert result["actor_optimizer_count"] == result["critic_optimizer_count"] == 1
        assert result["actionnet_online_forwards"] == result["task_decisions"]
        assert result["actionnet_batched_forward_calls"] == session.rollout_length
        assert result["actor_grad_norm"] > 0 and result["critic_grad_norm"] > 0
        assert np.isfinite(result["approx_kl"])
        assert session.backend.partners.frozen_parameter_hash == partner_before
        for key in ("task_decisions", "episodes", "deliveries", "sparse_team_reward",
                    "native_shaped_reward", "optimization_reward"):
            aggregate_key = "task_decisions" if key == "task_decisions" else key
            assert sum(bucket[key] for bucket in buckets.values()) == pytest.approx(
                result[aggregate_key]
            )
        wandb = _wandb_metrics(result, "action")
        for label in ("h0", "h2", "h4"):
            assert f"train/{label}/deliveries_per_episode" in wandb
            assert f"train/shared_horizon_decision_count_{label}" in wandb
    finally:
        session._executor.shutdown(wait=True)


def test_shared_checkpoint_resume_stream_and_mode_mismatch_fail_closed(tmp_path):
    shared = _session(seed=75)
    shared.run_update()
    checkpoint = tmp_path / "shared.pt"
    metadata = shared.save(checkpoint)
    assert metadata["horizon_mode"] == "episode_balanced"
    assert metadata["horizons"] == [0, 2, 4]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "shared_horizon" in payload["training_state"]
    expected = shared.run_update()
    expected_hashes = (
        parameter_hash(shared.policy.actor), parameter_hash(shared.policy.critic)
    )

    resumed = _session(seed=75, resume_checkpoint=checkpoint)
    actual = resumed.run_update()
    try:
        assert expected_hashes == (
            parameter_hash(resumed.policy.actor), parameter_hash(resumed.policy.critic)
        )
        assert actual["active_horizons"] == expected["active_horizons"]
        assert actual["shared_horizon_episode_counts"] == expected["shared_horizon_episode_counts"]
        assert actual["shared_horizon_decision_counts"] == expected["shared_horizon_decision_counts"]
        with pytest.raises(RuntimeError):
            ActionTrainingSession(
                observation_mode="full", horizon=0, decision_time=None,
                fixation_shape=None, seed=75, rollout_length=3, episode_length=2,
                device="cpu", source_checkpoint=None, resume_checkpoint=checkpoint,
                fixed_partner_checkpoint=PARTNER, n_rollout_threads=3,
                inference_batching="predictor", ppo_epochs=1, num_mini_batches=1,
            )
        initialized = build_actionnet(seed=999)
        provenance = initialize_partial_from_full(
            checkpoint, actor=initialized, horizon=0
        )
        assert provenance["initialization"] == "shared_full_checkpoint"
        assert parameter_hash(initialized) == provenance[
            "source_actor_parameter_sha256"
        ]
    finally:
        shared._executor.shutdown(wait=True)
        resumed._executor.shutdown(wait=True)


def test_fixed_checkpoint_cannot_resume_shared(tmp_path):
    fixed = ActionTrainingSession(
        observation_mode="full", horizon=0, decision_time=None, fixation_shape=None,
        seed=76, rollout_length=3, episode_length=2, device="cpu", source_checkpoint=None,
        hsp_initial_checkpoint=LEARNER, fixed_partner_checkpoint=PARTNER,
        n_rollout_threads=3, inference_batching="predictor", ppo_epochs=1,
        num_mini_batches=1,
    )
    fixed.run_update()
    checkpoint = tmp_path / "fixed.pt"
    fixed.save(checkpoint)
    try:
        with pytest.raises(RuntimeError):
            _session(seed=76, resume_checkpoint=checkpoint)
    finally:
        fixed._executor.shutdown(wait=True)
