import pytest

from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
)
from final_model.server_parallel.protocol import deterministic_assignments


def test_uneven_assignment_is_deterministic_and_exhaustive():
    assignments = deterministic_assignments(100, 16)
    assert [len(item.env_ids) for item in assignments[:4]] == [7, 7, 7, 7]
    assert [len(item.env_ids) for item in assignments[4:]] == [6] * 12
    assert sorted(env for item in assignments for env in item.env_ids) == list(range(100))


def test_worker_count_does_not_change_trajectory_or_update(session_options):
    options = session_options(
        rollout_length=2,
        episode_length=1,
        n_rollout_threads=3,
        configured_total_steps=6,
        horizons=[0, 2, 4],
    )
    first = ServerParallelActionTrainingSession(
        rollout_workers=1, server_trace=True, **options
    )
    try:
        first_result = first.run_update()
        first_trace = first.last_rollout_trace
    finally:
        first.close()
    second = ServerParallelActionTrainingSession(
        rollout_workers=2, server_trace=True, **options
    )
    try:
        second_result = second.run_update()
        assert first_trace == second.last_rollout_trace
        assert first_result["actor_hash_after"] == second_result["actor_hash_after"]
        assert first_result["condition_decision_counts"] == second_result[
            "condition_decision_counts"
        ]
        assert first_result["shared_horizon_metrics"] == second_result[
            "shared_horizon_metrics"
        ]
        for key in (
            "reward", "sparse_team_reward", "native_shaped_reward",
            "weighted_shaped_reward", "deliveries", "acquisitions",
            "actionnet_queries", "proposal_revisions", "recurrent_commits",
            "collision_count", "collision_rate",
        ):
            assert first_result[key] == second_result[key]
        shared = first_result["shared_horizon_metrics"]
        assert sum(row["optimization_reward"] for row in shared.values()) == pytest.approx(
            first_result["reward"], abs=1e-6
        )
        assert sum(row["sparse_team_reward"] for row in shared.values()) == pytest.approx(
            first_result["sparse_team_reward"], abs=1e-6
        )
        assert sum(row["deliveries"] for row in shared.values()) == first_result[
            "deliveries"
        ]
        assert sum(row["episodes"] for row in shared.values()) == first_result["episodes"]
        decision_counts = [row["task_decisions"] for row in shared.values()]
        assert sum(decision_counts) == first_result["task_decisions"]
        assert max(decision_counts) - min(decision_counts) == 0
        for group in ("time_metrics", "horizon_metrics", "condition_metrics"):
            assert sum(
                row["collision_count"]
                for row in first_result[group].values()
            ) == first_result["collision_count"]
            assert sum(
                row["task_decisions"]
                for row in first_result[group].values()
            ) == first_result["task_decisions"]
    finally:
        second.close()
