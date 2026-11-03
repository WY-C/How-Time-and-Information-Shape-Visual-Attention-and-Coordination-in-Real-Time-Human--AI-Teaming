from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
)


def test_partial_workers1_vs16_trajectory_regression(session_options):
    options = session_options(
        rollout_length=1,
        episode_length=1,
        n_rollout_threads=16,
        configured_total_steps=16,
        horizons=[0, 2, 4],
    )
    first = ServerParallelActionTrainingSession(
        rollout_workers=1, server_trace=True, **options
    )
    try:
        expected = first.run_update()
        first_trace = first.last_rollout_trace
    finally:
        first.close()
    parallel = ServerParallelActionTrainingSession(
        rollout_workers=16, server_trace=True, **options
    )
    try:
        actual = parallel.run_update()
        assert first_trace == parallel.last_rollout_trace
        assert expected["actor_hash_after"] == actual["actor_hash_after"]
        assert expected["acquisitions"] == expected["actionnet_queries"]
        assert actual["acquisitions"] == actual["actionnet_queries"]
        assert expected["partner_step_guard_failures"] == 0
        assert actual["partner_step_guard_failures"] == 0
    finally:
        parallel.close()
