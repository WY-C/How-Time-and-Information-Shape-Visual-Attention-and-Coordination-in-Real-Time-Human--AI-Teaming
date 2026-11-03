from types import SimpleNamespace

import numpy as np
import pytest
import torch

from final_model.common.mappo import (
    build_training_stack,
    install_rollout_context_stable_actor,
)
from final_model.common.paths import asset
from final_model.partial_ver.canonical_training import (
    PREUPDATE_LOGPROB_ATOL,
    PartialTrainingSession,
)


class RecordingActor:
    def __init__(self):
        self.calls = []

    def evaluate_actions(
        self, observations, rnn_states, actions, masks, available_actions,
        active_masks,
    ):
        self.calls.append({
            "batch_size": len(observations),
            "observation_ids": np.asarray(observations)[:, 0].tolist(),
        })
        values = (
            np.asarray(observations)[:, :1] * 0.01
            + np.asarray(rnn_states)[:, :1] * 0.001
            + np.asarray(actions)[:, :1] * 0.0001
        ).astype(np.float32)
        return torch.from_numpy(values), torch.tensor(0.0), None


def audit_fixture():
    actor = RecordingActor()
    session = object.__new__(PartialTrainingSession)
    session.policy = SimpleNamespace(actor=actor)
    session.ppo = {"clip_param": 0.2}
    observations = np.asarray([
        [[11.0], [12.0], [13.0]],
        [[21.0], [22.0], [23.0]],
        [[0.0], [0.0], [0.0]],
    ], dtype=np.float32)
    rnn_states = observations * 2.0
    masks = np.ones((3, 3, 1), dtype=np.float32)
    available = np.ones((3, 3, 6), dtype=np.float32)
    actions = np.asarray([
        [[1], [2], [3]],
        [[4], [5], [0]],
    ], dtype=np.int64)
    active_masks = np.asarray([
        [[1.0], [0.0], [1.0]],
        [[0.0], [1.0], [1.0]],
        [[1.0], [1.0], [1.0]],
    ], dtype=np.float32)
    active = active_masks[:-1, :, 0] > 0.5
    expected = (
        observations[:-1][active, :1] * 0.01
        + rnn_states[:-1][active, :1] * 0.001
        + actions[active, :1] * 0.0001
    ).astype(np.float32)
    old_log_probs = np.zeros((2, 3, 1), dtype=np.float32)
    old_log_probs[active] = expected
    session.buffer = SimpleNamespace(
        active_masks=active_masks,
        obs=observations,
        rnn_states=rnn_states,
        actions=actions,
        masks=masks,
        available_actions=available,
        action_log_probs=old_log_probs,
    )
    return session, actor, expected


def test_preupdate_audit_replays_every_active_transition_batch1_in_order():
    session, actor, expected = audit_fixture()
    result = session._preupdate_audit()

    assert result["preupdate_parity_pass"] == 1.0
    assert len(actor.calls) == len(expected) == 4
    assert [call["batch_size"] for call in actor.calls] == [1, 1, 1, 1]
    assert [call["observation_ids"][0] for call in actor.calls] == [11, 13, 22, 23]
    assert result["preupdate_max_abs_logprob_difference"] == 0.0


def test_audit_helper_matches_explicit_manual_batch1_loop_exactly():
    session, actor, _expected = audit_fixture()
    active = session.buffer.active_masks[:-1, :, 0] > 0.5
    arrays = (
        session.buffer.obs[:-1][active],
        session.buffer.rnn_states[:-1][active],
        session.buffer.actions[active],
        session.buffer.masks[:-1][active],
        session.buffer.available_actions[:-1][active],
    )
    actual = session._audit_batch1_action_log_probs(*arrays)
    manual = []
    for index in range(len(arrays[0])):
        value, _entropy, _ = actor.evaluate_actions(
            *(array[index : index + 1] for array in arrays),
            np.ones((1, 1), np.float32),
        )
        manual.append(value)
    assert torch.equal(actual, torch.cat(manual, dim=0))


def test_preupdate_batch1_replay_preserves_cpu_and_cuda_rng():
    session, _actor, _expected = audit_fixture()
    cpu_before = torch.get_rng_state().clone()
    cuda_before = (
        [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None
    )
    session._preupdate_audit()
    assert torch.equal(torch.get_rng_state(), cpu_before)
    if cuda_before is not None:
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_before)
        )


def test_preupdate_audit_remains_fail_closed_at_exact_tolerance():
    session, _actor, _expected = audit_fixture()
    assert PREUPDATE_LOGPROB_ATOL == 1e-5
    session.buffer.action_log_probs[0, 0, 0] += 2.0 * PREUPDATE_LOGPROB_ATOL
    with pytest.raises(RuntimeError, match="log-prob parity failed"):
        session._preupdate_audit()


def test_audit_helper_is_not_used_by_ppo_update():
    assert "_audit_batch1_action_log_probs" in (
        PartialTrainingSession._preupdate_audit.__code__.co_names
    )
    assert "_audit_batch1_action_log_probs" not in (
        PartialTrainingSession._action_ppo_update.__code__.co_names
    )
    assert "evaluate_actions" in PartialTrainingSession._action_ppo_update.__code__.co_names


def test_ppo_evaluate_actions_keeps_existing_non_batch1_path():
    session = PartialTrainingSession(
        gaze_type="rule_based",
        train_action=True,
        train_gaze=False,
        decision_times=[600],
        horizons=[0],
        fixation_shapes=["plus"],
        seed=907,
        rollout_length=2,
        episode_length=2,
        n_rollout_threads=1,
        device="cpu",
        source_checkpoint=None,
        configured_total_steps=2,
        fixed_partner_checkpoint=asset(
            "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
        ),
        checkpoint_interval=2,
        reward_shaping_horizon_steps=20_000_000,
        rule_gaze_probabilities={
            "task": 0.60, "stay": 0.20, "stale": 0.15, "random": 0.05,
        },
        ppo_epochs=1,
        num_mini_batches=1,
    )
    batch_sizes = []
    original = session.policy.actor.evaluate_actions

    def recording_evaluate(observations, *args, **kwargs):
        batch_sizes.append(len(observations))
        return original(observations, *args, **kwargs)

    session.policy.actor.evaluate_actions = recording_evaluate
    result = session.run_update()
    audit_count = int(result["preupdate_transition_count"])

    assert audit_count == 2
    assert batch_sizes[:audit_count] == [1] * audit_count
    assert any(size > 1 for size in batch_sizes[audit_count:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_representative_actor_audit_matches_manual_batch1_exactly():
    _args, policy, _trainer, *_spaces = build_training_stack(
        seed=911, rollout_length=2, n_rollout_threads=2, device="cuda"
    )
    install_rollout_context_stable_actor(policy.actor, microbatch_size=100)
    session = object.__new__(PartialTrainingSession)
    session.policy = policy
    generator = np.random.default_rng(912)
    observations = generator.normal(size=(3, 8, 5, 25)).astype(np.float32)
    rnn_states = np.zeros((3, 1, 64), dtype=np.float32)
    actions = np.asarray([[0], [3], [5]], dtype=np.int64)
    masks = np.ones((3, 1), dtype=np.float32)
    available = np.ones((3, 6), dtype=np.float32)
    arrays = (observations, rnn_states, actions, masks, available)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()]

    actual = session._audit_batch1_action_log_probs(*arrays)
    manual = []
    with torch.no_grad():
        for index in range(3):
            value, _entropy, _ = policy.actor.evaluate_actions(
                *(array[index : index + 1] for array in arrays),
                np.ones((1, 1), np.float32),
            )
            manual.append(value)

    assert torch.equal(actual, torch.cat(manual, dim=0))
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert all(
        torch.equal(current, expected)
        for current, expected in zip(torch.cuda.get_rng_state_all(), cuda_before)
    )
