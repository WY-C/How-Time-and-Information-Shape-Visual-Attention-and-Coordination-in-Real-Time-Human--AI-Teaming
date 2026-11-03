from types import SimpleNamespace

import numpy as np
import pytest
import torch

from final_model.common.action_training import ActionTrainingSession
from final_model.common.inference_batching import (
    batched_partner_actions,
    partner_shadow_parity,
)
from final_model.train import parser, validate_combination


def test_canonical_inference_batching_defaults_are_mode_specific():
    full = parser().parse_args(
        [
            "--observation-mode", "full",
            "--training-stage", "action",
            "--horizon", "4",
            "--hsp-initial-checkpoint",
            str(asset("policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt")),
            "--fixed-partner-checkpoint",
            str(asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")),
        ]
    )
    validate_combination(full)
    assert full.inference_batching == "predictor"

    partial = parser().parse_args(
        [
            "--observation-mode", "partial",
            "--training-stage", "action",
            "--horizon", "4",
            "--decision-time", "200",
            "--fixation-shape", "3x3",
            "--fixed-partner-checkpoint",
            str(asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")),
        ]
    )
    validate_combination(partial)
    assert partial.inference_batching == "legacy"
from final_model.common.partners import parameter_hash
from final_model.common.paths import asset


class FakePartnerActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        linear = torch.nn.Linear(2, 6, bias=False)
        with torch.no_grad():
            linear.weight.copy_(
                torch.tensor(
                    [
                        [0.1, -0.2],
                        [0.2, 0.1],
                        [-0.3, 0.4],
                        [0.5, -0.1],
                        [-0.2, -0.3],
                        [0.0, 0.2],
                    ]
                )
            )
        self.linear = linear
        self.act = SimpleNamespace(action_out=SimpleNamespace(linear=self.linear))

    def get_probs(self, observations, hidden, masks, legal_masks):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        legal_masks = torch.as_tensor(legal_masks, dtype=torch.float32)
        logits = self.act.action_out.linear(observations)
        logits[legal_masks == 0] = -1e10
        probabilities = torch.softmax(logits, dim=-1)
        delta = observations[:, :1].reshape(-1, 1, 1)
        return probabilities, hidden * masks.reshape(-1, 1, 1) + delta


class FakePool:
    def __init__(self, actor, env_index):
        self.actor = actor
        self.device = torch.device("cpu")
        self.hidden = torch.full((1, 1, 2), float(env_index))
        self.mask = torch.ones((1, 1))
        self.torch_rng = torch.Generator().manual_seed(5000 + env_index)


def _fake_batch(count=100):
    actor = FakePartnerActor()
    pools = [FakePool(actor, env_index) for env_index in range(count)]
    observations = np.stack(
        [np.asarray((env_index + 1, -(env_index + 1)), dtype=np.float32)
         for env_index in range(count)]
    )
    legal_masks = np.ones((count, 6), dtype=np.float32)
    legal_masks[np.arange(count), np.arange(count) % 6] = 0.0
    return actor, pools, observations, legal_masks


def test_partner_batch100_order_hidden_isolation_legal_mask_and_rng():
    actor, pools, observations, legal_masks = _fake_batch()
    actor_hash = parameter_hash(actor)
    generator_states = [pool.torch_rng.get_state().clone() for pool in pools]
    actions, outputs = batched_partner_actions(pools, observations, legal_masks)

    assert actions.shape == (100,)
    assert outputs["next_hidden"].shape == (100, 1, 2)
    assert np.all(actions != np.arange(100) % 6)
    for env_index, pool in enumerate(pools):
        expected = env_index + env_index + 1
        assert torch.equal(
            pool.hidden,
            torch.full((1, 1, 2), float(expected)),
        )
        expected_generator = torch.Generator().set_state(generator_states[env_index])
        torch.multinomial(
            torch.as_tensor(outputs["probabilities"][env_index]),
            1,
            generator=expected_generator,
        )
        assert torch.equal(pool.torch_rng.get_state(), expected_generator.get_state())
    assert parameter_hash(actor) == actor_hash


def test_one_partner_env_reset_does_not_reset_other_hidden_state():
    _actor, pools, observations, legal_masks = _fake_batch()
    batched_partner_actions(pools, observations, legal_masks)
    before = [pool.hidden.clone() for pool in pools]
    pools[17].hidden.zero_()
    pools[17].mask.zero_()
    for env_index, pool in enumerate(pools):
        if env_index != 17:
            assert torch.equal(pool.hidden, before[env_index])
            assert torch.equal(pool.mask, torch.ones_like(pool.mask))
    batched_partner_actions(pools, observations, legal_masks)
    assert torch.equal(
        pools[17].hidden,
        torch.full((1, 1, 2), 18.0),
    )
    assert all(
        not torch.equal(pools[env_index].hidden, pools[17].hidden)
        for env_index in range(100)
        if env_index != 17
    )


def test_partner_shadow_parity_preserves_sampling_without_mutating_state():
    _actor, pools, observations, legal_masks = _fake_batch()
    hidden_before = [pool.hidden.clone() for pool in pools]
    rng_before = [pool.torch_rng.get_state().clone() for pool in pools]
    report = partner_shadow_parity(pools, observations, legal_masks)
    assert report["environment_count"] == 100
    assert report["env_mapping_exact"]
    assert report["legal_mask_exact"]
    assert report["partner_top1_action_agreement"] == 1.0
    assert report["partner_paired_sampled_action_agreement"] == 1.0
    assert report["partner_max_abs_logit_difference"] == 0.0
    assert report["partner_max_abs_probability_difference"] == 0.0
    assert report["partner_max_abs_next_hidden_difference"] == 0.0
    assert all(torch.equal(pool.hidden, hidden_before[index]) for index, pool in enumerate(pools))
    assert all(torch.equal(pool.torch_rng.get_state(), rng_before[index]) for index, pool in enumerate(pools))


def _production_session(horizon):
    return ActionTrainingSession(
        observation_mode="full",
        horizon=horizon,
        decision_time=None,
        fixation_shape=None,
        seed=20260821,
        rollout_length=10,
        episode_length=10,
        device="cpu",
        source_checkpoint=None,
        hsp_initial_checkpoint=asset("policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"),
        fixed_partner_checkpoint=asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"),
        n_rollout_threads=4,
        inference_batching="partner_predictor",
        ppo_epochs=1,
        num_mini_batches=1,
    )


@pytest.mark.parametrize("horizon", (2, 4))
def test_predictor_batch_preview_target_singleton_order_and_cache(horizon):
    session = _production_session(horizon)
    try:
        report = session.predictor_parity
        assert report["predictor_model_shared_instance_count"] == 1
        assert report["env_mapping_exact"]
        assert report["predictor_target_agent_index"] == 0
        assert report["actual_partner_agent_index"] == 0
        assert report["predictor_partner_alignment_verified"]
        assert report["displayed_preview_agreement_by_horizon"][:horizon] == [
            1.0
        ] * horizon
        assert report["first_displayed_preview_divergence"] is None
        initial_generations = [backend.state_generation for backend in session.backends]
        assert all(
            backend._preview_cache_generation == initial_generations[env_index]
            for env_index, backend in enumerate(session.backends)
        )

        session._commit_decisions(np.full((4, 1), 4, dtype=np.int64))
        assert all(backend._preview_cache is None for backend in session.backends)
        session._prepare_decisions()
        assert all(
            backend._preview_cache_generation == backend.state_generation
            and backend.state_generation == initial_generations[env_index] + 1
            for env_index, backend in enumerate(session.backends)
        )
    finally:
        session._executor.shutdown(wait=True)
