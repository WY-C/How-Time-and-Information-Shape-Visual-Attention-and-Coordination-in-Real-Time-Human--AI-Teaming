import numpy as np
import pytest
import torch

from final_model.common.mappo import build_actionnet
from final_model.partial_ver.canonical_training import (
    PREUPDATE_LOGPROB_ATOL,
    PartialTrainingSession,
)
from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
    _canonical_batch1_rollout_actor_outputs,
)


def _inputs():
    generator = np.random.default_rng(20260825)
    observations = generator.normal(size=(2, 3, 8, 5, 25)).astype(np.float32)
    pre_actor = generator.normal(size=(2, 1, 64)).astype(np.float32)
    recurrent_masks = np.ones((2, 1), dtype=np.float32)
    available_actions = np.ones((1, 6), dtype=np.float32)
    rows = [(0, 0), (0, 2), (1, 1), (1, 2)]
    return observations, rows, pre_actor, recurrent_masks, available_actions


def _reference_outputs(
    actor, observations, rows, pre_actor, recurrent_masks, available_actions
):
    actions = []
    log_probabilities = []
    candidates = []
    with torch.no_grad():
        for env_id, acquisition in rows:
            action, log_probability, candidate = actor(
                observations[env_id, acquisition][None],
                pre_actor[env_id][None],
                recurrent_masks[env_id][None],
                available_actions,
                deterministic=False,
            )
            actions.append(int(action.item()))
            log_probabilities.append(float(log_probability.item()))
            candidates.append(candidate.detach().cpu().numpy()[0].copy())
    return (
        np.asarray(actions, dtype=np.int64),
        np.asarray(log_probabilities, dtype=np.float32),
        np.asarray(candidates, dtype=np.float32),
    )


def _assert_reference_parity(device):
    actor = build_actionnet(seed=20260825, device=device)
    actor.eval()
    inputs = _inputs()
    cpu_before = torch.get_rng_state().clone()
    cuda_before = (
        torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
    )

    expected = _reference_outputs(actor, *inputs)
    cpu_after_reference = torch.get_rng_state().clone()
    cuda_after_reference = (
        torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
    )

    torch.set_rng_state(cpu_before)
    if cuda_before is not None:
        torch.cuda.set_rng_state(cuda_before, device)
    calls = []
    original_forward = actor.forward

    def recording_forward(
        observation, rnn_state, mask, available_actions, *, deterministic
    ):
        calls.append({
            "observation": observation.copy(),
            "rnn_state": rnn_state.copy(),
            "mask": mask.copy(),
            "available_actions": available_actions.copy(),
            "deterministic": deterministic,
        })
        return original_forward(
            observation,
            rnn_state,
            mask,
            available_actions,
            deterministic=deterministic,
        )

    def forbidden_get_probs(*_args, **_kwargs):
        raise AssertionError("stale get_probs rollout path was restored")

    actor.forward = recording_forward
    actor.get_probs = forbidden_get_probs
    actual = _canonical_batch1_rollout_actor_outputs(actor, *inputs)

    assert len(calls) == len(inputs[1])
    assert all(call["observation"].shape[0] == 1 for call in calls)
    assert all(call["rnn_state"].shape[0] == 1 for call in calls)
    assert all(call["mask"].shape[0] == 1 for call in calls)
    assert all(call["available_actions"].shape == (1, 6) for call in calls)
    assert all(call["deterministic"] is False for call in calls)
    observations, rows, pre_actor, recurrent_masks, available_actions = inputs
    for call, (env_id, acquisition) in zip(calls, rows):
        assert np.array_equal(
            call["observation"], observations[env_id, acquisition][None]
        )
        assert np.array_equal(call["rnn_state"], pre_actor[env_id][None])
        assert np.array_equal(call["mask"], recurrent_masks[env_id][None])
        assert np.array_equal(call["available_actions"], available_actions)
    for actual_value, expected_value in zip(actual, expected):
        assert np.array_equal(actual_value, expected_value)
    assert torch.equal(torch.get_rng_state(), cpu_after_reference)
    if cuda_after_reference is not None:
        assert torch.equal(
            torch.cuda.get_rng_state(device), cuda_after_reference
        )


def test_optimized_cpu_outputs_and_rng_are_exactly_reference():
    _assert_reference_parity("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_optimized_cuda_outputs_and_rng_are_exactly_reference():
    _assert_reference_parity("cuda")


def test_candidate_action_and_log_prob_come_from_each_same_forward():
    class TaggedActor:
        def __init__(self):
            self.calls = 0

        def __call__(self, observation, rnn_state, mask, available, deterministic):
            tag = self.calls
            self.calls += 1
            assert observation.shape[0] == rnn_state.shape[0] == mask.shape[0] == 1
            assert available.shape == (1, 6)
            assert deterministic is False
            return (
                torch.tensor([[tag % 6]], dtype=torch.int64),
                torch.tensor([[tag + 0.25]], dtype=torch.float32),
                torch.full((1, 1, 64), float(tag), dtype=torch.float32),
            )

    actor = TaggedActor()
    actions, logs, candidates = _canonical_batch1_rollout_actor_outputs(
        actor, *_inputs()
    )
    assert actor.calls == 4
    assert np.array_equal(actions, np.asarray([0, 1, 2, 3]))
    assert np.array_equal(logs, np.asarray([0.25, 1.25, 2.25, 3.25]))
    assert all(np.all(candidates[index] == index) for index in range(4))


def test_optimization_is_rollout_only_and_audit_tolerance_is_unchanged():
    helper_name = "_canonical_batch1_rollout_actor_outputs"
    assert helper_name in ServerParallelActionTrainingSession.run_update.__code__.co_names
    assert helper_name not in PartialTrainingSession._preupdate_audit.__code__.co_names
    assert helper_name not in PartialTrainingSession._action_ppo_update.__code__.co_names
    assert PREUPDATE_LOGPROB_ATOL == 1e-5
