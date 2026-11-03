import random

import numpy as np
import pytest
import torch

from final_model.server_parallel.parallel_action_training import (
    _accumulate_horizon_metrics,
    _finalize_horizon_metrics,
    _new_horizon_metrics,
    _validate_horizon_metric_totals,
)


def assert_numpy_state_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_condition_aggregation_reconciles_and_preserves_rng():
    random.seed(301)
    np.random.seed(302)
    torch.manual_seed(303)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    horizons = np.asarray([0, 2, 4, 0, 2, 4], dtype=np.int64)
    rewards = np.asarray([
        [1.0, 10.0, 0.0, 0.0],
        [2.0, 20.0, 0.0, 0.0],
        [3.0, 30.0, 0.0, 0.0],
        [4.0, 40.0, 0.0, 0.0],
        [5.0, 50.0, 0.0, 0.0],
        [6.0, 60.0, 0.0, 0.0],
    ], dtype=np.float32)
    deliveries = np.asarray([1, 0, 2, 3, 1, 0], dtype=np.int64)
    dones = np.ones(6, dtype=bool)
    totals = _new_horizon_metrics([0, 2, 4])
    _accumulate_horizon_metrics(totals, horizons, rewards, deliveries, dones)
    result = _finalize_horizon_metrics(totals)
    _validate_horizon_metric_totals(
        result,
        optimization_reward=float(rewards[:, 0].sum()),
        sparse_reward=float(rewards[:, 1].sum()),
        deliveries=int(deliveries.sum()),
        episodes=int(dones.sum()),
        decisions=len(horizons),
    )

    assert result["h0"] == {
        "optimization_reward": 5.0,
        "sparse_team_reward": 50.0,
        "deliveries": 4,
        "task_decisions": 2,
        "episodes": 2,
        "reward": 5.0,
        "sparse_reward": 50.0,
        "decisions": 2,
        "episode_count": 2,
        "deliveries_per_episode": 2.0,
    }
    assert result["h2"]["deliveries_per_episode"] == 0.5
    assert result["h4"]["deliveries_per_episode"] == 1.0
    decision_counts = [result[label]["task_decisions"] for label in ("h0", "h2", "h4")]
    assert max(decision_counts) - min(decision_counts) == 0
    assert random.getstate() == python_before
    assert_numpy_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_deliveries_per_episode_is_undefined_without_completed_episode():
    totals = _new_horizon_metrics([0, 2, 4])
    horizons = np.asarray([0, 2, 4], dtype=np.int64)
    rewards = np.zeros((3, 4), dtype=np.float32)
    _accumulate_horizon_metrics(
        totals,
        horizons,
        rewards,
        np.zeros(3, dtype=np.int64),
        np.zeros(3, dtype=bool),
    )
    result = _finalize_horizon_metrics(totals)
    assert all(row["deliveries_per_episode"] is None for row in result.values())
    assert all(row["episode_count"] == 0 for row in result.values())


def test_reconciliation_fails_closed_on_incompatible_total():
    totals = _finalize_horizon_metrics(_new_horizon_metrics([0, 2, 4]))
    with pytest.raises(RuntimeError, match="optimization reward"):
        _validate_horizon_metric_totals(
            totals,
            optimization_reward=1.0,
            sparse_reward=0.0,
            deliveries=0,
            episodes=0,
            decisions=0,
        )
