import numpy as np
import pytest

from final_model.common.action_training import ActionTrainingSession
from final_model.common.reward import hsp_reward_shaping_factor


PPO = dict(ppo_epochs=1, num_mini_batches=2)


def test_shared_action_loop_one_forward_actual_metrics_and_native_reward(tmp_path):
    session = ActionTrainingSession(
        observation_mode="full", horizon=0, decision_time=None, fixation_shape=None,
        seed=11, rollout_length=16, episode_length=32, device="cpu",
        source_checkpoint=None, partner="mep5_init", n_rollout_threads=2, **PPO,
    )
    result = session.run_update()
    assert result["actionnet_online_forwards"] == result["task_decisions"] == 32
    assert result["actionnet_batched_forward_calls"] == 16
    assert result["environment_steps"] == 32
    assert result["n_rollout_threads"] == 2
    assert result["online_forwards_per_decision"] == 1.0
    assert result["partner_step_guard_checks"] == 64
    assert result["partner_step_guard_failures"] == 0
    assert result["partner_full_hash_checks"] == 4
    assert result["partner_full_hash_failures"] == 0
    assert result["partner_full_hash_baseline"] == result["partner_full_hash_actual"]
    assert result["partner_optimizer_isolation_verified"]
    assert result["reward"] == result["optimization_reward"]
    assert result["optimization_reward"] == pytest.approx(
        result["sparse_team_reward"] + result["weighted_shaped_reward"]
    )
    assert result["reward_shaping_factor"] == hsp_reward_shaping_factor(32)
    assert np.isfinite(result["approx_kl"])
    assert np.isfinite(result["dist_entropy"])
    assert 0 <= result["clip_fraction"] <= 1


def test_action_episode_horizon_uses_hsp_time_limit_bad_mask():
    session = ActionTrainingSession(
        observation_mode="full", horizon=0, decision_time=None, fixation_shape=None,
        seed=15, rollout_length=16, episode_length=16, device="cpu",
        source_checkpoint=None, partner="mep5_init", n_rollout_threads=2, **PPO,
    )
    result = session.run_update()
    assert result["episodes"] == 2
    assert session.args.use_proper_time_limits
    assert np.array_equal(
        session.buffer.bad_masks[0], np.zeros((2, 1), dtype=np.float32)
    )
