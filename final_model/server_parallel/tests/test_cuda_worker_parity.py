import pytest
import torch

from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_worker_count_preserves_discrete_trajectory(session_options):
    options = session_options(
        device="cuda",
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
        assert first_result["actor_hash_before"] == second_result["actor_hash_before"]
        assert first_result["preupdate_max_abs_logprob_difference"] <= 1e-5
        assert second_result["preupdate_max_abs_logprob_difference"] <= 1e-5
    finally:
        second.close()
