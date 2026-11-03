from __future__ import annotations

from pathlib import Path

import pytest
import torch

from final_model.common.action_training import ActionTrainingSession
from final_model.common.partners import parameter_hash
from final_model.common.paths import asset
from final_model.common.reward import (
    HSP_REWARD_SHAPING_HORIZON,
    hsp_reward_shaping_factor,
)
from final_model.train import parser, validate_combination


LEARNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt")
PARTNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")


def _fixed_session(
    *, reward_horizon=None, resume_checkpoint: Path | None = None, seed=181,
):
    kwargs = {}
    if reward_horizon is not None:
        kwargs["reward_shaping_horizon_steps"] = reward_horizon
    return ActionTrainingSession(
        observation_mode="full",
        horizon=0,
        decision_time=None,
        fixation_shape=None,
        seed=seed,
        rollout_length=4,
        episode_length=8,
        device="cpu",
        source_checkpoint=None,
        resume_checkpoint=resume_checkpoint,
        partner="mep5_init",
        n_rollout_threads=1,
        ppo_epochs=1,
        num_mini_batches=1,
        **kwargs,
    )


def test_default_and_custom_reward_shaping_factors():
    assert HSP_REWARD_SHAPING_HORIZON == 20_000_000
    assert hsp_reward_shaping_factor(0) == 1.0
    assert hsp_reward_shaping_factor(10_000_000) == 0.5
    assert hsp_reward_shaping_factor(20_000_000) == 0.0
    assert hsp_reward_shaping_factor(21_000_000) == 0.0
    assert hsp_reward_shaping_factor(0, 60_000_000) == 1.0
    assert hsp_reward_shaping_factor(30_000_000, 60_000_000) == 0.5
    assert hsp_reward_shaping_factor(60_000_000, 60_000_000) == 0.0
    assert hsp_reward_shaping_factor(60_000_000, 120_000_000) == 0.5
    assert hsp_reward_shaping_factor(120_000_000, 120_000_000) == 0.0
    assert hsp_reward_shaping_factor(0, 0) == 0.0
    assert hsp_reward_shaping_factor(1, 0) == 0.0
    with pytest.raises(ValueError, match="negative"):
        hsp_reward_shaping_factor(0, -1)


@pytest.mark.parametrize("steps", (20_000_000, 60_000_000, 120_000_000))
def test_cli_keeps_budget_and_reward_horizon_independent(steps):
    args = parser().parse_args(
        [
            "--observation-mode", "full", "--training-stage", "action",
            "--shared-horizons", "0", "2", "4",
            "--hsp-initial-checkpoint", str(LEARNER),
            "--fixed-partner-checkpoint", str(PARTNER),
            "--steps", str(steps),
            "--reward-shaping-horizon-steps", str(steps),
        ]
    )
    validate_combination(args)
    assert args.steps == steps
    assert args.reward_shaping_horizon_steps == steps


def test_cli_default_disabled_and_invalid_reward_horizon():
    args = parser().parse_args(
        [
            "--observation-mode", "full", "--training-stage", "action",
            "--shared-horizons", "0", "2", "4",
            "--hsp-initial-checkpoint", str(LEARNER),
            "--fixed-partner-checkpoint", str(PARTNER),
        ]
    )
    assert args.reward_shaping_horizon_steps == 20_000_000
    args.reward_shaping_horizon_steps = 0
    validate_combination(args)
    assert args.reward_shaping_horizon_steps == 0
    args.reward_shaping_horizon_steps = -1
    with pytest.raises(ValueError, match="negative"):
        validate_combination(args)


def test_zero_reward_horizon_is_strict_sparse():
    session = _fixed_session(reward_horizon=0)
    result = session.run_update()
    assert result["reward_shaping_horizon_steps"] == 0
    assert result["reward_shaping_factor"] == 0.0
    assert result["weighted_shaped_reward"] == 0.0
    assert result["optimization_reward"] == result["sparse_team_reward"]


def test_fixed_default_training_matches_explicit_20m():
    default = _fixed_session()
    default_result = default.run_update()
    default_hashes = (
        parameter_hash(default.policy.actor), parameter_hash(default.policy.critic)
    )
    explicit = _fixed_session(reward_horizon=20_000_000)
    explicit_result = explicit.run_update()
    assert default_hashes == (
        parameter_hash(explicit.policy.actor), parameter_hash(explicit.policy.critic)
    )
    for key in (
        "reward", "optimization_reward", "sparse_team_reward",
        "native_shaped_reward", "weighted_shaped_reward", "reward_shaping_factor",
        "actor_grad_norm", "critic_grad_norm",
    ):
        assert explicit_result[key] == default_result[key]


def test_custom_horizon_checkpoint_resume_and_mismatch_fail_closed(tmp_path):
    source = _fixed_session(reward_horizon=60_000_000, seed=182)
    source.run_update()
    checkpoint = tmp_path / "custom_reward_horizon.pt"
    metadata = source.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert metadata["reward_shaping_horizon_steps"] == 60_000_000
    assert payload["training_state"]["contract"][
        "reward_shaping_horizon_steps"
    ] == 60_000_000
    assert payload["training_state"]["schedule_state"][
        "reward_shaping_horizon_steps"
    ] == 60_000_000

    resumed = _fixed_session(
        reward_horizon=60_000_000, resume_checkpoint=checkpoint, seed=182
    )
    assert resumed.reward_shaping_horizon_steps == 60_000_000
    assert resumed.backend.reward_shaping_factor == hsp_reward_shaping_factor(
        resumed.completed_steps, 60_000_000
    )
    with pytest.raises(RuntimeError, match="reward shaping horizon|contract mismatch"):
        _fixed_session(resume_checkpoint=checkpoint, seed=182)


def test_shared_training_accepts_custom_reward_horizon():
    session = ActionTrainingSession(
        observation_mode="full",
        horizon=None,
        shared_horizons=(0, 2, 4),
        decision_time=None,
        fixation_shape=None,
        seed=183,
        rollout_length=3,
        episode_length=2,
        device="cpu",
        source_checkpoint=None,
        hsp_initial_checkpoint=LEARNER,
        fixed_partner_checkpoint=PARTNER,
        reward_shaping_horizon_steps=120_000_000,
        n_rollout_threads=3,
        inference_batching="predictor",
        ppo_epochs=1,
        num_mini_batches=1,
    )
    try:
        result = session.run_update()
        assert result["reward_shaping_horizon_steps"] == 120_000_000
        assert result["reward_shaping_factor"] == hsp_reward_shaping_factor(
            result["environment_steps"], 120_000_000
        )
        assert all(
            value > 0 for value in result["ppo_sample_counts_by_horizon"].values()
        )
    finally:
        session._executor.shutdown(wait=True)
