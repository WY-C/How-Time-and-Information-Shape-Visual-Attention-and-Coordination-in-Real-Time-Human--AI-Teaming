from pathlib import Path

import numpy as np
import pytest
import torch

import final_model.common.action_training as action_training
from final_model.common.hsp_initialization import (
    FIRST_CONV_KEY,
    HSP_INITIALIZATION_TYPE,
    HSP_SOURCE_CHECKPOINT_SHA256,
    build_canonical_hsp_actor,
    migrate_hsp_s2_actor,
    preview_learnability_audit,
)
from final_model.common.mappo import build_actionnet, build_training_stack
from final_model.common.partners import parameter_hash
from final_model.common.paths import asset
from final_model.train import parser, validate_combination


def canonical_hsp_checkpoint() -> Path:
    return asset("policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt")


def fixed_partner_checkpoint() -> Path:
    return asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")


def test_canonical_hsp_s2_is_accepted_and_incompatible_sources_are_rejected(tmp_path):
    source = canonical_hsp_checkpoint()
    actor, metadata, state = build_canonical_hsp_actor(
        source, device=torch.device("cpu")
    )
    assert metadata["hsp_source_sha256"] == HSP_SOURCE_CHECKPOINT_SHA256
    assert metadata["source_population"] == 36
    assert metadata["source_seed"] == 4
    assert metadata["input_shape"] == (8, 5, 20)
    assert metadata["recurrent"] is True
    assert metadata["recurrent_n"] == 1
    assert metadata["hidden_size"] == 64
    assert metadata["action_count"] == 6
    assert tuple(state[FIRST_CONV_KEY].shape) == (32, 20, 3, 3)
    assert actor._recurrent_N == 1

    damaged = tmp_path / "damaged.pt"
    torch.save(state, damaged)
    with pytest.raises(RuntimeError, match="canonical identity"):
        build_canonical_hsp_actor(damaged, device=torch.device("cpu"))

    feed_forward = asset("policy_pool/random3/hsp/s1/hsp/hsp113_final_w1_actor.pt")
    with pytest.raises(RuntimeError, match="canonical identity"):
        build_canonical_hsp_actor(feed_forward, device=torch.device("cpu"))


def test_migration_copies_every_source_weight_and_zeroes_only_extra_channels():
    source_actor, _metadata, source = build_canonical_hsp_actor(
        canonical_hsp_checkpoint(), device=torch.device("cpu")
    )
    target = build_actionnet(seed=20260821)
    provenance = migrate_hsp_s2_actor(canonical_hsp_checkpoint(), target_actor=target)
    migrated = target.state_dict()

    assert provenance["initialization_type"] == HSP_INITIALIZATION_TYPE
    assert provenance["learner_hsp_source_population"] == 36
    assert provenance["migration_verified"] is True
    assert tuple(migrated[FIRST_CONV_KEY].shape) == (32, 25, 3, 3)
    assert torch.equal(migrated[FIRST_CONV_KEY][:, :20], source[FIRST_CONV_KEY])
    assert torch.count_nonzero(migrated[FIRST_CONV_KEY][:, 20:]).item() == 0
    for key in source:
        if key != FIRST_CONV_KEY:
            assert torch.equal(migrated[key], source[key]), key
    assert all(parameter.requires_grad for parameter in target.parameters())
    assert not source_actor.training


def test_migration_preserves_function_for_h0_h2_h4_and_initial_hashes_match():
    hashes = []
    for horizon in (0, 2, 4):
        target = build_actionnet(seed=20260821)
        provenance = migrate_hsp_s2_actor(canonical_hsp_checkpoint(), target_actor=target)
        audit = provenance["function_preservation"]
        assert audit["verified"] is True
        assert audit["top1_action_agreement"] == 100.0
        assert audit["max_abs_logit_difference"] <= audit["atol"]
        assert audit["max_abs_probability_difference"] <= audit["atol"]
        assert audit["kl_original_to_migrated"] <= audit["atol"]
        assert audit["max_abs_next_hidden_difference"] <= audit["atol"]
        hashes.append(parameter_hash(target))
    assert len(set(hashes)) == 1


def test_hsp_initialization_leaves_critic_and_optimizers_fresh():
    kwargs = dict(seed=41, rollout_length=16, ppo_epochs=1, num_mini_batches=2)
    _args, reference, _trainer, *_spaces = build_training_stack(**kwargs)
    _args, initialized, _trainer, *_spaces = build_training_stack(**kwargs)
    critic_before = parameter_hash(initialized.critic)
    provenance = migrate_hsp_s2_actor(
        canonical_hsp_checkpoint(), target_actor=initialized.actor
    )
    assert parameter_hash(initialized.critic) == critic_before == parameter_hash(reference.critic)
    assert initialized.actor_optimizer.state == {}
    assert initialized.critic_optimizer.state == {}
    assert provenance["initial_actor_parameter_sha256"] == parameter_hash(initialized.actor)


def test_bounded_h0_h2_h4_session_initialization_starts_at_zero(monkeypatch, tmp_path):
    class Partners:
        def __init__(self):
            self.actor = torch.nn.Linear(1, 1)

    class Repository:
        def __init__(self, *, device):
            self.partners = Partners()

    class Backend:
        def __init__(self, *, horizon, partner_repository, **kwargs):
            self.horizon = horizon
            self.oracle = None
            name = str(partner_repository.manifest[0]["name"])
            self.partners = type(
                "FixedPartners",
                (),
                {"actor": partner_repository.actor(name)},
            )()
            self.environment_steps = 0

        def canonical_pair(self):
            state = np.zeros((8, 5, 20), dtype=np.float32)
            return state, state.copy()

        def preview_planes(self):
            return np.ones((4, 8, 5), dtype=np.float32) * 255.0

        def learner_available_actions(self):
            return np.ones(6, dtype=np.float32)

        def training_state(self):
            return {"environment_steps": 0}

        def predictor_partner_contract(self):
            return {
                "displayed_preview_partner_identity": "hsp_s2_s24_seed3",
                "predictor_target_agent_index": 0,
                "predictor_targets_learner": False,
                "predictor_partner_alignment_verified": True,
            }

    monkeypatch.setattr(action_training, "FrozenPartnerRepository", Repository)
    monkeypatch.setattr(action_training, "DirectOvercookedBackend", Backend)
    hashes = {}
    critic_hashes = {}
    for horizon in (0, 2, 4):
        session = action_training.ActionTrainingSession(
            observation_mode="full",
            horizon=horizon,
            decision_time=None,
            fixation_shape=None,
            seed=20260821,
            rollout_length=16,
            episode_length=400,
            device="cpu",
            source_checkpoint=None,
            hsp_initial_checkpoint=canonical_hsp_checkpoint(),
            fixed_partner_checkpoint=fixed_partner_checkpoint(),
            partner=None,
            n_rollout_threads=1,
            ppo_epochs=1,
            num_mini_batches=2,
        )
        assert session.completed_steps == 0
        assert session.update_count == 0
        assert session.policy.actor_optimizer.state == {}
        assert session.policy.critic_optimizer.state == {}
        assert np.all(session.buffer.obs[0, ..., 20] == 255.0)
        hashes[horizon] = parameter_hash(session.policy.actor)
        critic_hashes[horizon] = parameter_hash(session.policy.critic)
        metadata = session.save(tmp_path / f"h{horizon}.pt")
        assert metadata["initialization_type"] == HSP_INITIALIZATION_TYPE
        assert metadata["input_channels"] == 25
        assert metadata["channel_20_value"] == 255.0
        step0_path = tmp_path / f"h{horizon}_step0.pt"
        step0 = session.save_initialization_checkpoint(step0_path)
        payload = torch.load(step0_path, map_location="cpu", weights_only=False)
        assert step0["checkpoint_role"] == "pre_training_initialization"
        assert step0["step"] == 0
        assert step0["fixed_partner_seed"] == 3
        assert step0["learner_hsp_source_seed"] == 4
        assert step0["learner_source_equals_partner"] is False
        assert step0["configured_total_steps"] == 20_000_000
        assert step0["checkpoint_interval"] == 1_000_000
        assert step0["entropy_schedule"] == [0.2, 0.05, 0.01]
        assert step0["entropy_schedule_horizons"] == [0, 10_000_000, 20_000_000]
        assert step0["reward_shaping_schedule"] == "linear"
        assert step0["reward_shaping_initial"] == 1.0
        assert step0["reward_shaping_final"] == 0.0
        assert step0["reward_shaping_horizon_steps"] == 20_000_000
        assert step0["full_channel20_input"] == 255.0
        assert step0["full_channel20_weight_frozen_zero"] is True
        assert step0["preview_weight_norm_logging"] is True
        assert step0["step0_checkpoint_saved"] is True
        assert step0["train_time_evaluation"] is False
        assert step0["checkpoint_entropy_coef"] == 0.2
        assert step0["checkpoint_reward_shaping_factor"] == 1.0
        assert payload["training_state"]["schedule_state"] == {
            "global_environment_step": 0,
            "entropy_coef": 0.2,
            "reward_shaping_factor": 1.0,
            "reward_shaping_horizon_steps": 20_000_000,
        }
        assert payload["training_state"]["completed_steps"] == 0
        assert payload["training_state"]["update_count"] == 0
        assert payload["training_state"]["actor_optimizer"]["state"] == {}
        assert payload["training_state"]["critic_optimizer"]["state"] == {}
    assert len(set(hashes.values())) == 1
    assert len(set(critic_hashes.values())) == 1


def test_preview_channels_receive_gradients_and_move_after_one_bounded_update():
    result = preview_learnability_audit(canonical_hsp_checkpoint())
    assert result["H2"]["active_preview_channels"] == [21, 22]
    assert result["H4"]["active_preview_channels"] == [21, 22, 23, 24]
    for metrics in result.values():
        assert metrics["nonzero_active_gradient_count"] > 0
        assert metrics["changed_active_weight_count"] > 0
        assert metrics["bounded_optimizer_steps"] == 1


@pytest.mark.parametrize("horizon", (0, 2, 4))
def test_cli_accepts_full_action_hsp_initialization(horizon):
    args = parser().parse_args(
        [
            "--observation-mode", "full",
            "--training-stage", "action",
            "--horizon", str(horizon),
            "--hsp-initial-checkpoint", str(canonical_hsp_checkpoint()),
            "--fixed-partner-checkpoint", str(fixed_partner_checkpoint()),
        ]
    )
    validate_combination(args)


@pytest.mark.parametrize(
    "arguments",
    (
        [
            "--observation-mode", "partial", "--training-stage", "action",
            "--horizon", "0", "--decision-time", "200", "--fixation-shape", "3x3",
        ],
        [
            "--observation-mode", "partial", "--training-stage", "gaze",
            "--horizon", "0", "--decision-time", "200", "--fixation-shape", "3x3",
            "--source-checkpoint", "partial.pt",
        ],
        [
            "--observation-mode", "full", "--training-stage", "action", "--horizon", "0",
            "--resume-checkpoint", "resume.pt",
        ],
        [
            "--observation-mode", "partial", "--training-stage", "action",
            "--horizon", "0", "--decision-time", "200", "--fixation-shape", "3x3",
            "--source-checkpoint", "source.pt",
        ],
    ),
)
def test_cli_rejects_hsp_initialization_conflicts(arguments):
    args = parser().parse_args(
        [*arguments, "--hsp-initial-checkpoint", str(canonical_hsp_checkpoint())]
    )
    with pytest.raises(ValueError):
        validate_combination(args)
