from pathlib import Path
import copy

import numpy as np
import pytest
import torch

import final_model.common.action_training as action_training
import final_model.common.mappo as mappo
from final_model.checkpoint import state_dict_hash
from final_model.common.hsp_initialization import (
    FIRST_CONV_KEY,
    FullFreshnessWeightGuard,
    migrate_hsp_s2_actor,
)
from final_model.common.mappo import build_training_stack
from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.common.partners import (
    FIXED_HSP_PARTNER_ID,
    FIXED_HSP_PARTNER_SHA256,
    FixedHSPPartnerRepository,
    FrozenPartnerPool,
    parameter_hash,
)
from final_model.common.paths import asset, file_sha256
from final_model.common.reward import hsp_reward_shaping_factor
from final_model.common.tracking import FULL_ACTION_MANDATORY_KEYS
from final_model.train import _wandb_metrics, parser, validate_combination


def learner_checkpoint() -> Path:
    return asset("policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt")


def partner_checkpoint() -> Path:
    return asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")


class BoundedFixedBackend:
    """Small deterministic backend fixture; PPO and partner actors remain production."""

    def __init__(
        self,
        *,
        seed,
        horizon,
        episode_length,
        device,
        partner_repository,
        prediction_oracle=None,
        **_kwargs,
    ):
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.episode_length = int(episode_length)
        self.environment_steps = 0
        self.episode_step = 0
        self.reward_shaping_factor = 1.0
        self.partners = FrozenPartnerPool(
            seed=seed, device=device, repository=partner_repository
        )
        self.oracle = prediction_oracle if prediction_oracle is not None else (
            object() if horizon else None
        )
        self.partner_step_guard_checks = 0
        self.partner_step_guard_failures = 0
        self.partner_full_hash_checks = 0
        self.partner_full_hash_failures = 0
        self.last_partner_full_hash_baseline = self.partners.frozen_parameter_hash
        self.last_partner_full_hash_actual = self.partners.frozen_parameter_hash

    @property
    def active_partner(self):
        return self.partners.name

    def canonical_pair(self):
        state = np.zeros((8, 5, 20), dtype=np.float32)
        for channel in range(20):
            index = (self.environment_steps * 7 + channel * 11) % 40
            state[index // 5, index % 5, channel] = 255.0
        return state.copy(), state

    def preview_planes(self):
        result = np.zeros((4, 8, 5), dtype=np.float32)
        for offset in range(4):
            index = (self.environment_steps + offset * 9) % 40
            result[offset, index // 5, index % 5] = 255.0
        return result

    def learner_available_actions(self):
        return np.ones(6, dtype=np.float32)

    def predictor_partner_contract(self):
        return {
            "displayed_preview_partner_identity": self.active_partner,
            "predictor_target_agent_index": 0,
            "predictor_targets_learner": False,
            "predictor_partner_alignment_verified": True,
        }

    def set_reward_shaping_factor(self, value):
        self.reward_shaping_factor = float(value)

    def step(self, _learner_action):
        before = self.partners.assert_frozen()
        self.partner_step_guard_checks += 1
        self.partners.action(self.canonical_pair()[0], np.ones(6, dtype=np.float32))
        after = self.partners.assert_frozen()
        self.partner_step_guard_checks += 1
        assert before == after
        self.environment_steps += 1
        self.episode_step += 1
        return {
            "optimization_reward": self.reward_shaping_factor,
            "sparse_team_reward": 0.0,
            "native_shaped_reward": 1.0,
            "weighted_shaped_reward": self.reward_shaping_factor,
            "deliveries": 0,
            "done": self.episode_step >= self.episode_length,
        }

    def reset(self):
        self.partners.load_next()
        self.episode_step = 0

    def verify_partner_parameters(self):
        self.partner_full_hash_checks += 1
        actual = self.partners.assert_frozen(full_hash=True)
        self.last_partner_full_hash_actual = actual
        return actual

    def training_state(self):
        return {
            "environment_steps": self.environment_steps,
            "episode_step": self.episode_step,
            "reward_shaping_factor": self.reward_shaping_factor,
            "partners": self.partners.training_state(),
            "partner_step_guard_checks": self.partner_step_guard_checks,
            "partner_step_guard_failures": self.partner_step_guard_failures,
            "partner_full_hash_checks": self.partner_full_hash_checks,
            "partner_full_hash_failures": self.partner_full_hash_failures,
            "last_partner_full_hash_baseline": self.last_partner_full_hash_baseline,
            "last_partner_full_hash_actual": self.last_partner_full_hash_actual,
        }

    def load_training_state(self, state):
        self.environment_steps = int(state["environment_steps"])
        self.episode_step = int(state["episode_step"])
        self.reward_shaping_factor = float(state["reward_shaping_factor"])
        self.partners.load_training_state(copy.deepcopy(state["partners"]))
        for key in (
            "partner_step_guard_checks",
            "partner_step_guard_failures",
            "partner_full_hash_checks",
            "partner_full_hash_failures",
        ):
            setattr(self, key, int(state[key]))
        self.last_partner_full_hash_baseline = state["last_partner_full_hash_baseline"]
        self.last_partner_full_hash_actual = state["last_partner_full_hash_actual"]


def test_s36_seed4_learner_and_s24_seed3_partner_are_distinct_canonical_sources():
    learner_sha = file_sha256(learner_checkpoint())
    partner_sha = file_sha256(partner_checkpoint())
    assert learner_sha == "73b0be7e6e7f818d97a8292e0b9d040e6e6bbcaeda4dd8e6954f7cee8797ea15"
    assert partner_sha == FIXED_HSP_PARTNER_SHA256
    assert learner_sha != partner_sha

    repository = FixedHSPPartnerRepository(
        device="cpu", checkpoint=partner_checkpoint()
    )
    provenance = repository.provenance()
    assert len(repository.manifest) == 1
    assert provenance["fixed_partner_seed"] == 3
    assert provenance["fixed_partner_population"] == 24
    assert provenance["fixed_partner_pool_size"] == 1
    assert provenance["fixed_partner_sampling_disabled"] is True
    actor = repository.actor(FIXED_HSP_PARTNER_ID)
    assert tuple(actor.base.cnn[0].weight.shape) == (32, 20, 3, 3)
    assert actor._recurrent_N == 1 and actor.hidden_size == 64
    assert not actor.training
    assert not any(parameter.requires_grad for parameter in actor.parameters())
    assert repository.assert_frozen(FIXED_HSP_PARTNER_ID, full_hash=True) == parameter_hash(actor)


def test_fixed_partner_rejects_seed4_and_pool_never_selects_another_actor():
    with pytest.raises(RuntimeError, match="seed-3"):
        FixedHSPPartnerRepository(device="cpu", checkpoint=learner_checkpoint())
    repository = FixedHSPPartnerRepository(device="cpu", checkpoint=partner_checkpoint())
    pool = FrozenPartnerPool(seed=20260821, device="cpu", repository=repository)
    rng_before = copy.deepcopy(pool.rng.bit_generator.state)
    for _ in range(8):
        pool.load_next()
        assert pool.name == FIXED_HSP_PARTNER_ID
    assert pool.rng.bit_generator.state == rng_before
    assert set(pool.counts) == {FIXED_HSP_PARTNER_ID}


@pytest.mark.parametrize("horizon", (0, 2, 4))
def test_freshness_is_frozen_zero_while_only_active_preview_channels_learn(horizon):
    _args, policy, _trainer, *_spaces = build_training_stack(
        seed=20260821, rollout_length=16, ppo_epochs=1, num_mini_batches=2
    )
    migrate_hsp_s2_actor(learner_checkpoint(), target_actor=policy.actor)
    guard = FullFreshnessWeightGuard(policy.actor, policy.actor_optimizer)
    guard.initialize_zero()
    generator = torch.Generator().manual_seed(91 + horizon)
    observation = (
        torch.rand((64, 8, 5, 25), generator=generator) < 0.02
    ).float() * 255.0
    observation[..., 20] = 255.0
    observation[..., 21:25] = (
        torch.rand((64, 8, 5, 4), generator=generator) < 0.05
    ).float() * 255.0
    observation[..., 21 + horizon :] = 0.0
    if horizon == 0:
        observation[..., 21:25] = 0.0
    hidden = torch.zeros((64, 1, 64))
    masks = torch.ones((64, 1))
    available = torch.ones((64, 6))
    before = policy.actor.state_dict()[FIRST_CONV_KEY].detach().clone()

    policy.actor_optimizer.zero_grad(set_to_none=True)
    features = policy.actor.base(observation)
    features, _hidden = policy.actor.rnn(features, hidden, masks)
    logits = policy.actor.act.action_out.linear(features)
    (logits.square().mean() + logits[:, 0].mean()).backward()
    gradient = policy.actor.base.cnn[0].weight.grad
    assert torch.count_nonzero(gradient[:, 20]).item() == 0
    for channel in range(21, 25):
        if channel < 21 + horizon:
            assert torch.count_nonzero(gradient[:, channel]).item() > 0
        else:
            assert torch.count_nonzero(gradient[:, channel]).item() == 0
    policy.actor_optimizer.step()
    guard.after_optimizer_update()
    after = policy.actor.state_dict()[FIRST_CONV_KEY].detach()
    assert torch.count_nonzero(after[:, 20]).item() == 0
    for channel in range(21, 25):
        changed = torch.count_nonzero(after[:, channel] - before[:, channel]).item()
        assert (changed > 0) == (channel < 21 + horizon)
    metrics = guard.metrics()
    assert metrics["freshness_weight_norm"] == 0.0


def test_nonzero_freshness_weight_fails_closed_instead_of_being_repaired():
    _args, policy, _trainer, *_spaces = build_training_stack(
        seed=7, rollout_length=16, ppo_epochs=1, num_mini_batches=2
    )
    guard = FullFreshnessWeightGuard(policy.actor, policy.actor_optimizer)
    guard.initialize_zero()
    with torch.no_grad():
        policy.actor.base.cnn[0].weight[0, 20, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="frozen-zero contract violation"):
        guard.assert_zero("resume_checkpoint")


@pytest.mark.parametrize("horizon", (0, 2, 4))
def test_bounded_real_ppo_update_preserves_freshness_and_fixed_partner(
    monkeypatch, horizon
):
    monkeypatch.setattr(
        action_training, "DirectOvercookedBackend", BoundedFixedBackend
    )
    session = action_training.ActionTrainingSession(
        observation_mode="full",
        horizon=horizon,
        decision_time=None,
        fixation_shape=None,
        seed=20260821,
        rollout_length=16,
        episode_length=16,
        device="cpu",
        source_checkpoint=None,
        hsp_initial_checkpoint=learner_checkpoint(),
        fixed_partner_checkpoint=partner_checkpoint(),
        n_rollout_threads=1,
        ppo_epochs=1,
        num_mini_batches=2,
    )
    before_partner = parameter_hash(session.backend.partners.actor)
    before = session.input_channel_weight_metrics()
    assert set(before.values()) == {0.0}
    result = session.run_update()
    assert result["environment_steps"] == 16
    assert result["freshness_weight_norm"] == 0.0
    assert parameter_hash(session.backend.partners.actor) == before_partner
    assert result["partner_step_guard_failures"] == 0
    assert result["partner_full_hash_failures"] == 0
    assert result["reward_identity_error"] == pytest.approx(0.0, abs=1e-10)
    assert result["shaping_factor"] == hsp_reward_shaping_factor(16)
    assert result["entropy_coef"] == pytest.approx(0.2, abs=1e-5)
    assert result["preupdate_parity_pass"] == 1.0
    assert result["preupdate_max_abs_logprob_difference"] <= 1e-5
    assert result["preupdate_ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert result["preupdate_approx_kl"] == pytest.approx(0.0, abs=1e-10)
    assert result["preupdate_clip_fraction"] == 0.0
    assert result["preupdate_max_legal_mask_difference"] == 0.0
    for key in (
        "approx_kl", "clip_fraction", "policy_loss", "value_loss",
        "dist_entropy", "entropy_coef", "explained_variance",
        "actor_grad_norm", "critic_grad_norm", "grad_clip_fraction",
        "steps_per_second", "rollout_collection_seconds", "ppo_update_seconds",
    ):
        assert np.isfinite(result[key]), key
    wandb = _wandb_metrics(result, "action")
    assert set(FULL_ACTION_MANDATORY_KEYS) <= set(wandb)
    assert all(np.isfinite(wandb[key]) for key in FULL_ACTION_MANDATORY_KEYS)
    assert set(result["sampled_partner_identities"]) == {FIXED_HSP_PARTNER_ID}
    for offset in range(1, 5):
        norm = result[f"preview_weight_norm_h{offset}"]
        assert (norm > 0.0) == (offset <= horizon)


def test_epoch1_minibatch1_h0_preupdate_parity_is_production_equivalent(
    monkeypatch,
):
    monkeypatch.setattr(
        action_training, "DirectOvercookedBackend", BoundedFixedBackend
    )
    corrected_audit = action_training.preupdate_policy_consistency_audit
    legacy = {}

    def audit_and_measure_legacy_order(**kwargs):
        corrected = corrected_audit(**kwargs)
        buffer = kwargs["buffer"]
        rng_state = torch.get_rng_state()
        try:
            sample = next(
                mappo._ZSCSeparatedReplayBuffer.recurrent_generator(
                    buffer,
                    np.zeros_like(buffer.rewards),
                    1,
                    kwargs["data_chunk_length"],
                )
            )
        finally:
            torch.set_rng_state(rng_state)
        replay_log_prob = mappo._actor_log_probs(
            kwargs["policy"], sample[1], sample[2], sample[4],
            sample[7], sample[11], sample[8],
        )
        log_ratio = replay_log_prob.astype(np.float64) - sample[9].astype(np.float64)
        ratio = np.exp(log_ratio)
        legacy.update(
            {
                "max_abs_logprob_difference": float(np.abs(log_ratio).max()),
                "approx_kl": float(((ratio - 1.0) - log_ratio).mean()),
                "clip_fraction": float((np.abs(ratio - 1.0) > 0.2).mean()),
            }
        )
        return corrected

    monkeypatch.setattr(
        action_training,
        "preupdate_policy_consistency_audit",
        audit_and_measure_legacy_order,
    )
    session = action_training.ActionTrainingSession(
        observation_mode="full",
        horizon=0,
        decision_time=None,
        fixation_shape=None,
        seed=20260821,
        rollout_length=20,
        episode_length=7,
        device="cpu",
        source_checkpoint=None,
        hsp_initial_checkpoint=learner_checkpoint(),
        fixed_partner_checkpoint=partner_checkpoint(),
        n_rollout_threads=4,
        actor_lr=5e-4,
        critic_lr=5e-4,
        entropy_coefs=(0.2, 0.05, 0.01),
        entropy_coef_horizons=(0, 10_000_000, 20_000_000),
        ppo_epochs=1,
        num_mini_batches=1,
    )
    result = session.run_update()
    assert session.args.data_chunk_length == 10
    assert session.policy.actor_optimizer.param_groups[0]["lr"] == 5e-4
    assert result["preupdate_transition_count"] == 80
    assert result["preupdate_max_abs_logprob_difference"] <= 1e-5
    assert result["preupdate_ratio_min"] == pytest.approx(1.0, abs=1e-5)
    assert result["preupdate_ratio_max"] == pytest.approx(1.0, abs=1e-5)
    assert result["preupdate_ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert result["preupdate_approx_kl"] == pytest.approx(0.0, abs=1e-10)
    assert result["preupdate_clip_fraction"] == 0.0
    assert result["preupdate_lower_clip_fraction"] == 0.0
    assert result["preupdate_upper_clip_fraction"] == 0.0
    assert result["preupdate_observation_hash_match"] == 1.0
    assert result["preupdate_observation_exact_match"] == 1.0
    assert result["preupdate_immediate_insert_checks"] == 80
    assert result["preupdate_post_actor_forward_checks"] == 80
    assert result["preupdate_source_buffer_alias_count"] == 0
    assert result["preupdate_capture_buffer_alias_count"] == 0
    assert result["preupdate_transition_id_exact_match"] == 1.0
    assert result["preupdate_action_exact_match"] == 1.0
    assert result["preupdate_rnn_state_exact_match"] == 1.0
    assert result["preupdate_recurrent_mask_exact_match"] == 1.0
    assert result["preupdate_legal_mask_exact_match"] == 1.0
    assert result["preupdate_max_legal_mask_difference"] == 0.0
    assert result["ratio"] == pytest.approx(1.0, abs=1e-5)
    assert result["approx_kl"] == pytest.approx(0.0, abs=1e-10)
    assert result["clip_fraction"] == 0.0
    assert legacy["max_abs_logprob_difference"] > 1.0
    assert legacy["approx_kl"] > 0.1
    assert legacy["clip_fraction"] > 0.5
    replay_steps = {
        transition_id // session.n_rollout_threads
        for transition_id in result["preupdate_direct_replay_transition_ids"]
    }
    assert {0, 6, 7, 8, 10, 19} <= replay_steps


def test_preupdate_audit_failure_prevents_every_optimizer_step(monkeypatch):
    monkeypatch.setattr(
        action_training, "DirectOvercookedBackend", BoundedFixedBackend
    )
    session = action_training.ActionTrainingSession(
        observation_mode="full", horizon=0, decision_time=None,
        fixation_shape=None, seed=20260821, rollout_length=10,
        episode_length=7, device="cpu", source_checkpoint=None,
        hsp_initial_checkpoint=learner_checkpoint(),
        fixed_partner_checkpoint=partner_checkpoint(), n_rollout_threads=2,
        ppo_epochs=1, num_mini_batches=1,
    )
    actor_before = parameter_hash(session.policy.actor)
    optimizer_steps = {"actor": 0, "critic": 0}
    original_actor_step = session.policy.actor_optimizer.step
    original_critic_step = session.policy.critic_optimizer.step

    def actor_step(*args, **kwargs):
        optimizer_steps["actor"] += 1
        return original_actor_step(*args, **kwargs)

    def critic_step(*args, **kwargs):
        optimizer_steps["critic"] += 1
        return original_critic_step(*args, **kwargs)

    def fail_audit(**_kwargs):
        raise RuntimeError(
            "PREUPDATE_POLICY_PARITY_FAILURE tensor=legal_action_mask"
        )

    monkeypatch.setattr(session.policy.actor_optimizer, "step", actor_step)
    monkeypatch.setattr(session.policy.critic_optimizer, "step", critic_step)
    monkeypatch.setattr(
        action_training, "preupdate_policy_consistency_audit", fail_audit
    )
    with pytest.raises(RuntimeError, match="PREUPDATE_POLICY_PARITY_FAILURE"):
        session.run_update()
    assert optimizer_steps == {"actor": 0, "critic": 0}
    assert parameter_hash(session.policy.actor) == actor_before


def test_flattened_forward_context_is_diagnostic_not_canonical(monkeypatch):
    monkeypatch.setattr(
        action_training, "DirectOvercookedBackend", BoundedFixedBackend
    )
    original_actor_log_probs = mappo._actor_log_probs

    def context_sensitive_replay(
        policy, obs, rnn_states, actions, masks, available, active
    ):
        result = original_actor_log_probs(
            policy, obs, rnn_states, actions, masks, available, active
        )
        # Only the legacy all-transition one-step context has one hidden per row.
        if obs.shape[0] == 80 and rnn_states.shape[0] == 80:
            result[9, 0] += 0.22166180610656738
        return result

    monkeypatch.setattr(mappo, "_actor_log_probs", context_sensitive_replay)
    session = action_training.ActionTrainingSession(
        observation_mode="full", horizon=0, decision_time=None,
        fixation_shape=None, seed=20260821, rollout_length=20,
        episode_length=7, device="cpu", source_checkpoint=None,
        hsp_initial_checkpoint=learner_checkpoint(),
        fixed_partner_checkpoint=partner_checkpoint(), n_rollout_threads=4,
        ppo_epochs=1, num_mini_batches=1,
    )
    result = session.run_update()
    assert result["preupdate_parity_pass"] == 1.0
    assert result["preupdate_max_abs_logprob_difference"] <= 1e-5
    assert result[
        "preupdate_flattened_context_max_abs_logprob_difference"
    ] == pytest.approx(0.22166180610656738)
    assert result["preupdate_flattened_context_transition_id"] == 9
    assert result["preupdate_flattened_context_is_canonical"] == 0.0
    assert result["preupdate_diagnostic_classification"] == (
        "F_ANOTHER_AUDIT_BUG_FLATTENED_FORWARD_CONTEXT"
    )
    report = result["preupdate_detailed_actor_context"]
    assert report["failure"]["time"] == 2
    assert report["failure"]["env"] == 1
    assert report["first_pipeline_divergence"] is None
    assert report["actual_rnn_argument_parity"]["actual_vs_buffer"][
        "exact_equal"
    ]
    assert all(
        row["exact_hidden_argument_parity"]["actual_vs_buffer"]["exact_equal"]
        for row in report["env_rnn_reset_trace"]
    )


def test_full_checkpoint_save_and_resume_preserve_zero_and_reject_nonzero(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        action_training, "DirectOvercookedBackend", BoundedFixedBackend
    )

    def session(*, resume_checkpoint=None):
        return action_training.ActionTrainingSession(
            observation_mode="full",
            horizon=2,
            decision_time=None,
            fixation_shape=None,
            seed=20260821,
            rollout_length=16,
            episode_length=16,
            device="cpu",
            source_checkpoint=None,
            resume_checkpoint=resume_checkpoint,
            hsp_initial_checkpoint=(
                learner_checkpoint() if resume_checkpoint is None else None
            ),
            fixed_partner_checkpoint=partner_checkpoint(),
            n_rollout_threads=1,
            ppo_epochs=1,
            num_mini_batches=2,
        )

    original = session()
    original.save_initialization_checkpoint(tmp_path / "step0.pt")
    original.run_update()
    checkpoint = tmp_path / "full_resume.pt"
    original.save(checkpoint)
    expected = original.run_update()
    expected_actor = parameter_hash(original.policy.actor)
    expected_critic = parameter_hash(original.policy.critic)
    resumed = session(resume_checkpoint=checkpoint)
    assert resumed.completed_steps == 16
    assert resumed.input_channel_weight_metrics()["freshness_weight_norm"] == 0.0
    actual = resumed.run_update()
    assert parameter_hash(resumed.policy.actor) == expected_actor
    assert parameter_hash(resumed.policy.critic) == expected_critic
    for key in (
        "environment_steps",
        "update_count",
        "optimization_reward",
        "sparse_team_reward",
        "native_shaped_reward",
        "weighted_shaped_reward",
        "deliveries",
        "preupdate_max_abs_logprob_difference",
        "preupdate_ratio_mean",
        "preupdate_approx_kl",
        "preupdate_clip_fraction",
    ):
        assert actual[key] == expected[key]
    assert resumed.input_channel_weight_metrics()["freshness_weight_norm"] == 0.0

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["actor"][FIRST_CONV_KEY][0, 20, 0, 0] = 1.0
    payload["metadata"]["actor_parameter_sha256"] = state_dict_hash(payload["actor"])
    invalid = tmp_path / "full_resume_invalid.pt"
    torch.save(payload, invalid)
    with pytest.raises(RuntimeError, match="frozen-zero contract violation"):
        session(resume_checkpoint=invalid)


def test_10m_resume_preserves_global_entropy_and_shaping_schedules(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        action_training, "DirectOvercookedBackend", BoundedFixedBackend
    )

    def make(*, resume_checkpoint=None, configured_total_steps=20_000_000):
        return action_training.ActionTrainingSession(
            observation_mode="full", horizon=2, decision_time=None,
            fixation_shape=None, seed=20260821, rollout_length=16,
            episode_length=16, device="cpu", source_checkpoint=None,
            resume_checkpoint=resume_checkpoint,
            hsp_initial_checkpoint=(learner_checkpoint() if resume_checkpoint is None else None),
            fixed_partner_checkpoint=partner_checkpoint(), n_rollout_threads=1,
            configured_total_steps=configured_total_steps, checkpoint_interval=1_000_000,
            entropy_coefs=(0.2, 0.05, 0.01),
            entropy_coef_horizons=(0, 10_000_000, 20_000_000),
            ppo_epochs=1, num_mini_batches=2,
        )

    source = make()
    source.save_initialization_checkpoint(tmp_path / "resume_step0.pt")
    source.completed_steps = 10_000_000
    source.update_count = 10_000_000 // 16
    source.backend.environment_steps = 10_000_000
    source.backend.set_reward_shaping_factor(0.500005)
    checkpoint = tmp_path / "resume_10m.pt"
    metadata = source.save(checkpoint)
    assert metadata["configured_total_steps"] == 20_000_000
    assert metadata["checkpoint_interval"] == 1_000_000
    assert metadata["entropy_schedule"] == [0.2, 0.05, 0.01]
    assert metadata["entropy_schedule_horizons"] == [0, 10_000_000, 20_000_000]
    assert metadata["reward_shaping_schedule"] == "linear"
    assert metadata["reward_shaping_horizon_steps"] == 20_000_000
    assert metadata["train_time_evaluation"] is False
    assert metadata["checkpoint_entropy_coef"] == pytest.approx(0.05)
    assert metadata["checkpoint_reward_shaping_factor"] == pytest.approx(0.5)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training_state"]["schedule_state"] == {
        "global_environment_step": 10_000_000,
        "entropy_coef": 0.05,
        "reward_shaping_factor": 0.5,
        "reward_shaping_horizon_steps": 20_000_000,
    }

    resumed = make(resume_checkpoint=checkpoint)
    assert resumed.completed_steps == 10_000_000
    assert resumed.trainer.entropy_coef == pytest.approx(0.05)
    assert resumed.backend.reward_shaping_factor == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="checkpoint contract mismatch"):
        make(resume_checkpoint=checkpoint, configured_total_steps=21_000_000)
    payload["training_state"]["schedule_state"]["entropy_coef"] = 0.2
    corrupt = tmp_path / "resume_10m_bad_schedule.pt"
    torch.save(payload, corrupt)
    with pytest.raises(RuntimeError, match="resume schedule state mismatch"):
        make(resume_checkpoint=corrupt)


def test_predictor_subject_is_fixed_agent0_partner_not_seed4_learner():
    backend = object.__new__(DirectOvercookedBackend)
    backend.fixed_partner_identity = FIXED_HSP_PARTNER_ID
    backend.partners = type("Partners", (), {"name": FIXED_HSP_PARTNER_ID})()
    contract = backend.predictor_partner_contract()
    assert contract["actual_partner_agent_index"] == 0
    assert contract["predictor_target_agent_index"] == 0
    assert contract["learner_agent_index"] == 1
    assert contract["predictor_targets_learner"] is False
    assert contract["predictor_partner_alignment_verified"] is True


def test_full_cli_requires_exact_seed4_learner_and_seed3_partner():
    base = [
        "--observation-mode", "full",
        "--training-stage", "action",
        "--horizon", "4",
    ]
    valid = parser().parse_args(
        [
            *base,
            "--hsp-initial-checkpoint", str(learner_checkpoint()),
            "--fixed-partner-checkpoint", str(partner_checkpoint()),
        ]
    )
    validate_combination(valid)
    with pytest.raises(ValueError, match="fixed-partner-checkpoint"):
        validate_combination(
            parser().parse_args(
                [*base, "--hsp-initial-checkpoint", str(learner_checkpoint())]
            )
        )
    with pytest.raises(ValueError, match="seed-4"):
        validate_combination(
            parser().parse_args(
                [
                    *base,
                    "--hsp-initial-checkpoint", str(partner_checkpoint()),
                    "--fixed-partner-checkpoint", str(partner_checkpoint()),
                ]
            )
        )
