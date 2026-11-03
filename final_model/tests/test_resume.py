from __future__ import annotations

from pathlib import Path

import pytest
import torch

from final_model.common.action_training import ActionTrainingSession
from final_model.common.overcooked_backend import cooking_contract
from final_model.common.partners import parameter_hash
from final_model.common.paths import asset
from final_model.train import parser, validate_combination


PPO = dict(
    actor_lr=5e-4,
    critic_lr=5e-4,
    clip_param=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    entropy_coefs=(0.2, 0.05, 0.01),
    entropy_coef_horizons=(0, 40_000_000, 80_000_000),
    ppo_epochs=1,
    num_mini_batches=2,
    max_grad_norm=10.0,
)


def _action_session(*, resume_checkpoint: Path | None = None):
    return ActionTrainingSession(
        observation_mode="full",
        horizon=0,
        decision_time=None,
        fixation_shape=None,
        seed=31,
        rollout_length=16,
        episode_length=40,
        device="cpu",
        source_checkpoint=None,
        resume_checkpoint=resume_checkpoint,
        partner="mep5_init",
        n_rollout_threads=1,
        **PPO,
    )


def test_action_resume_reproduces_uninterrupted_next_update(tmp_path):
    uninterrupted = _action_session()
    uninterrupted.run_update()
    checkpoint = tmp_path / "action_resume.pt"
    metadata = uninterrupted.save(checkpoint)
    assert {key: metadata[key] for key in cooking_contract()} == cooking_contract()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert {
        key: payload["training_state"]["contract"][key]
        for key in cooking_contract()
    } == cooking_contract()

    expected = uninterrupted.run_update()
    expected_actor = parameter_hash(uninterrupted.policy.actor)
    expected_critic = parameter_hash(uninterrupted.policy.critic)

    resumed = _action_session(resume_checkpoint=checkpoint)
    assert resumed.completed_steps == 16
    assert resumed.update_count == 1
    actual = resumed.run_update()

    assert parameter_hash(resumed.policy.actor) == expected_actor
    assert parameter_hash(resumed.policy.critic) == expected_critic
    for key in (
        "environment_steps",
        "updates",
        "reward",
        "sparse_team_reward",
        "native_shaped_reward",
        "weighted_shaped_reward",
        "deliveries",
        "active_partners",
    ):
        assert actual[key] == expected[key]

    payload["metadata"]["cooking_dynamics_id"] = "legacy_one_tick_per_step"
    incompatible = tmp_path / "action_incompatible_cooking.pt"
    torch.save(payload, incompatible)
    with pytest.raises(RuntimeError, match="checkpoint contract mismatch"):
        _action_session(resume_checkpoint=incompatible)


def test_resume_cli_is_fail_closed_and_gaze_resume_needs_no_source():
    gaze_resume = parser().parse_args(
        [
            "--observation-mode", "partial",
            "--training-stage", "gaze",
            "--horizon", "4",
            "--decision-time", "600",
            "--fixation-shape", "plus",
            "--resume-checkpoint", "gaze.pt",
            "--fixed-partner-checkpoint",
            str(asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")),
        ]
    )
    validate_combination(gaze_resume)

    both = parser().parse_args(
        [
            "--observation-mode", "partial",
            "--training-stage", "action",
            "--horizon", "0",
            "--decision-time", "200",
            "--fixation-shape", "3x3",
            "--source-checkpoint", "source.pt",
            "--resume-checkpoint", "resume.pt",
            "--fixed-partner-checkpoint",
            str(asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")),
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_combination(both)
