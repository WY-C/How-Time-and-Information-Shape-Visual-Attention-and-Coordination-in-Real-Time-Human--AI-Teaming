from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from final_model.checkpoint import save_action_checkpoint
from final_model.common.mappo import build_training_stack
from final_model.common.overcooked_backend import cooking_contract
from final_model.common.partners import (
    FIXED_HSP_PARTNER_ID,
    FixedHSPPartnerRepository,
    parameter_hash,
)
from final_model.common.paths import asset
from final_model.common.resume import capture_global_rng_state, restore_global_rng_state
from final_model.partial_ver.canonical_training import PartialTrainingSession
from final_model.partial_ver.canonical_training import (
    DIRECT_JOINT_INITIALIZATION_ID,
)
from final_model.partial_ver.constraints import (
    BASELINE_CONSTRAINT_CONFIG,
    FITTED_EMMA_CONSTRAINT_CONFIG,
    HumanLikeConstraintConfig,
    PeripheralVisionConfig,
    PreviewOpacityConfig,
    CANONICAL_TUNED_EMMA_TIMING,
    FULL_CONSTRAINT_CONFIG,
)
from final_model.partial_ver.gaze import (
    DEFAULT_EMMA_TIMING,
    HUMAN_DWELL_RNG_SEED_OFFSET,
    HUMAN_STOCHASTIC_EMMA_TIMING,
    resolve_emma_timing,
)
from final_model.partial_ver.memory import (
    CONTINUOUS_FRESHNESS_CONTRACT_ID,
    NO_DECAY_FRESHNESS_CONTRACT_ID,
    NO_DECAY_OBSERVATION_CONTRACT_ID,
    OBSERVATION_CONTRACT_ID,
)
from final_model.partial_ver.observation import (
    FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID,
    HELD_FOVEAL_REFRESH_CONTRACT_ID,
)
from final_model.train import _wandb_metrics


RULE = {"task": 0.60, "stay": 0.20, "stale": 0.15, "random": 0.05}
PARTNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")


def peripheral_config(
    mode, sigma=3.0, *, opacity="off", emma=DEFAULT_EMMA_TIMING,
):
    return HumanLikeConstraintConfig(
        profile="custom", peripheral=PeripheralVisionConfig(mode, sigma),
        emma=emma,
        preview_opacity=PreviewOpacityConfig(opacity),
    )


def kwargs(**overrides):
    result = dict(
        gaze_type="rule_based", train_action=True, train_gaze=False,
        decision_times=[200], horizons=[0], fixation_shapes=["3x3"],
        seed=31, rollout_length=2, episode_length=2, n_rollout_threads=1,
        device="cpu", source_checkpoint=None, configured_total_steps=6,
        fixed_partner_checkpoint=PARTNER,
        checkpoint_interval=2, reward_shaping_horizon_steps=60_000_000,
        rule_gaze_probabilities=RULE, ppo_epochs=1, num_mini_batches=1,
    )
    result.update(overrides)
    return result


def test_acquisition_rollout_ppo_and_freshness_gradient():
    session = PartialTrainingSession(**kwargs())
    result = session.run_update()
    assert result["environment_steps"] == result["task_decisions"] == 2
    assert result["actionnet_queries"] == result["acquisitions"]
    assert result["action_proposals"] == result["actionnet_queries"]
    assert result["executed_proposals"] + result["forced_stays"] == 2
    assert result["preupdate_parity_pass"] == 1
    assert result["preupdate_max_abs_logprob_difference"] <= 1e-5
    assert result["freshness_gradient_norm"] > 0
    assert result["freshness_parameter_delta_norm"] > 0
    assert result["reward_shaping_horizon_steps"] == 60_000_000
    assert result["active_partners"] == ["hsp_s2_s24_seed3"]
    assert result["fixed_partner_sampling_disabled"] is True
    assert set(result["time_metrics"]) == {"t200"}
    assert set(result["horizon_metrics"]) == {"h0"}
    assert set(result["condition_metrics"]) == {"t200_h0_3x3"}
    assert result["time_metrics"]["t200"]["decisions"] == 2
    assert result["condition_metrics"]["t200_h0_3x3"]["decisions"] == 2
    assert result["peripheral_attempts"] == 0
    assert result["peripheral_successes"] == 0
    assert result["peripheral_success_rate"] == 0
    wandb = _wandb_metrics(result, "action")
    assert wandb["perception/peripheral_attempts"] == 0
    assert wandb["perception/peripheral_success_rate"] == 0


def test_constraint_framework_default_is_fitted_emma():
    options = kwargs(decision_times=[400], horizons=[2])
    implicit = PartialTrainingSession(**options)
    implicit_result = implicit.run_update()
    explicit = PartialTrainingSession(**{
        **options, "constraint_config": FITTED_EMMA_CONSTRAINT_CONFIG,
    })
    explicit_result = explicit.run_update()
    for key in (
        "actor_hash_before", "actor_hash_after", "acquisitions",
        "gaze_requests", "same_target_acquisitions",
        "new_target_acquisitions", "action_proposals", "executed_proposals",
        "forced_stays", "sparse_team_reward", "native_shaped_reward",
        "optimization_reward", "reward", "deliveries",
    ):
        assert implicit_result[key] == explicit_result[key]
    assert list(implicit.adapter.events) == list(explicit.adapter.events)
    assert np.array_equal(
        implicit.adapter.memory.snapshot(), explicit.adapter.memory.snapshot()
    )
    assert np.array_equal(
        implicit.adapter.requested_target_counts,
        explicit.adapter.requested_target_counts,
    )
    assert np.array_equal(
        implicit.adapter.landing_target_counts,
        explicit.adapter.landing_target_counts,
    )
    assert implicit.backend.state == explicit.backend.state
    assert pickle.dumps(implicit.adapter.gaze_policy.training_state()) == (
        pickle.dumps(explicit.adapter.gaze_policy.training_state())
    )


def test_configurable_memory_lifetime_reaches_contract_and_adapters():
    session = PartialTrainingSession(**kwargs(memory_lifetime_ms=2501))
    assert session._contract()["memory_lifetime_ms"] == 2501
    assert session._contract()["freshness_contract_id"] == (
        CONTINUOUS_FRESHNESS_CONTRACT_ID
    )
    assert session.adapter.memory.memory_lifetime_ms == 2501
    assert session.adapter.memory.freshness_contract_id == (
        CONTINUOUS_FRESHNESS_CONTRACT_ID
    )


def test_no_decay_reaches_checkpoint_contract_and_adapters():
    session = PartialTrainingSession(**kwargs(
        memory_decay_mode="none", memory_lifetime_ms=None,
    ))
    contract = session._contract()
    assert contract["memory_decay_mode"] == "none"
    assert contract["memory_lifetime_ms"] is None
    assert contract["freshness_contract_id"] == NO_DECAY_FRESHNESS_CONTRACT_ID
    assert contract["memory_contract_id"] == NO_DECAY_OBSERVATION_CONTRACT_ID
    assert session.adapter.memory.memory_decay_mode == "none"


def test_fixed_t600_and_shared_nine_cell_metrics_are_first_class():
    fixed = PartialTrainingSession(**kwargs(
        decision_times=[600],
        horizons=[4],
        fixation_shapes=["plus"],
        rollout_length=1,
        episode_length=1,
        configured_total_steps=1,
    )).run_update()
    assert set(fixed["time_metrics"]) == {"t600"}
    assert set(fixed["condition_metrics"]) == {"t600_h4_plus"}
    assert "t600/deliveries_per_episode" in _wandb_metrics(fixed, "action")

    shared = PartialTrainingSession(**kwargs(
        decision_times=[200, 400, 600],
        horizons=[0, 2, 4],
        fixation_shapes=["plus"],
        rollout_length=1,
        episode_length=1,
        n_rollout_threads=9,
        configured_total_steps=9,
    )).run_update()
    cells = {
        f"t{decision_time}_h{horizon}_plus"
        for decision_time in (200, 400, 600)
        for horizon in (0, 2, 4)
    }
    assert set(shared["time_metrics"]) == {"t200", "t400", "t600"}
    assert set(shared["condition_metrics"]) == cells
    for group in ("time_metrics", "horizon_metrics", "condition_metrics"):
        assert sum(
            bucket["decisions"] for bucket in shared[group].values()
        ) == shared["task_decisions"]
        assert sum(
            bucket["deliveries"] for bucket in shared[group].values()
        ) == shared["deliveries"]
        assert sum(
            bucket["forced_stays"] for bucket in shared[group].values()
        ) == shared["forced_stays"]
        assert sum(
            bucket["sparse_reward"] for bucket in shared[group].values()
        ) == pytest.approx(shared["sparse_team_reward"])
    wandb = _wandb_metrics(shared, "action")
    assert all(f"t{value}/forced_stay_rate" in wandb for value in (200, 400, 600))
    assert all(f"cell/{cell}/acquisitions_per_decision" in wandb for cell in cells)


def test_partial_uses_full_fixed_partner_across_threads_and_episode_resets():
    session = PartialTrainingSession(**kwargs(
        episode_length=1, n_rollout_threads=2, configured_total_steps=4
    ))
    expected = FixedHSPPartnerRepository(device="cpu", checkpoint=PARTNER)
    expected_hash = parameter_hash(expected.actor(FIXED_HSP_PARTNER_ID))
    assert all(
        backend.active_partner == FIXED_HSP_PARTNER_ID
        and parameter_hash(backend.partners.actor) == expected_hash
        for backend in session.backends
    )
    result = session.run_update()
    assert result["active_partners"] == [FIXED_HSP_PARTNER_ID] * 2
    assert all(
        set(backend.partners.counts) == {FIXED_HSP_PARTNER_ID}
        and backend.partners.counts[FIXED_HSP_PARTNER_ID] >= 2
        for backend in session.backends
    )


def test_shared_full_initialization_is_actor_bit_exact(tmp_path):
    _args, policy, _trainer, *_spaces = build_training_stack(
        seed=5, rollout_length=2, n_rollout_threads=1
    )
    source = tmp_path / "shared_full.pt"
    save_action_checkpoint(
        source, actor=policy.actor, critic=policy.critic, mode="full",
        horizon=None, horizon_mode="episode_balanced", horizons=[0, 2, 4],
        step=2, seed=5, condition=cooking_contract(),
    )
    source_hash = parameter_hash(policy.actor)
    tuned = resolve_emma_timing(
        "tuned", prep_ms=135, exec_intercept_ms=44,
        exec_slope_ms_per_degree=3.8, dwell_ms=60,
    )
    session = PartialTrainingSession(**kwargs(
        source_checkpoint=source, horizons=[0, 2, 4], fixation_shapes=["2x2"],
        emma_timing=tuned,
    ))
    assert parameter_hash(session.policy.actor) == source_hash
    assert session.source_provenance["source_absolute_path"] == str(source.resolve())
    assert session.source_provenance["source_shared_horizons"] == [0, 2, 4]
    assert session.source_provenance["source_emma_timing"] == (
        DEFAULT_EMMA_TIMING.metadata()
    )
    assert session.source_provenance["target_emma_timing"] == tuned.metadata()
    assert session.source_provenance["source_constraint_config"][
        "constraint_variant_id"
    ] == "b0"
    assert session.source_provenance["target_constraint_config"][
        "constraint_variant_id"
    ] == "custom"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["metadata"]["cooking_dynamics_id"] = "legacy_cooking"
    incompatible = tmp_path / "legacy_shared_full.pt"
    torch.save(payload, incompatible)
    with pytest.raises(RuntimeError, match="cooking contract mismatch"):
        PartialTrainingSession(**kwargs(
            source_checkpoint=incompatible, horizons=[0, 2, 4]
        ))


def test_full_ground_truth_gaze_freezes_full_actionnet_and_trains_fresh_gaze(tmp_path):
    _args, policy, _trainer, *_spaces = build_training_stack(
        seed=51, rollout_length=4, n_rollout_threads=1
    )
    source = tmp_path / "full_action.pt"
    save_action_checkpoint(
        source,
        actor=policy.actor,
        critic=policy.critic,
        mode="full",
        horizon=0,
        step=4,
        seed=51,
        condition=cooking_contract(),
    )
    source_actor_hash = parameter_hash(policy.actor)
    source_critic_hash = parameter_hash(policy.critic)
    session = PartialTrainingSession(**kwargs(
        gaze_type="learned",
        train_action=False,
        train_gaze=True,
        ground_truth_policy_input=True,
        source_checkpoint=source,
        rollout_length=4,
        episode_length=8,
        configured_total_steps=4,
    ))
    gaze_before = {
        key: value.detach().clone()
        for key, value in session.gaze_actor.state_dict().items()
    }
    result = session.run_update()

    assert result["ground_truth_policy_input"] is True
    assert result["policy_observation_contract_id"] == (
        FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
    )
    assert result["actor_hash_before"] == result["actor_hash_after"] == source_actor_hash
    assert result["critic_hash_before"] == result["critic_hash_after"] == source_critic_hash
    assert result["action_actor_optimizer_present"] is False
    assert result["action_critic_optimizer_present"] is False
    assert result["freshness_normalized_min"] == 1.0
    assert result["freshness_normalized_max"] == 1.0
    assert any(
        not torch.equal(gaze_before[key], value)
        for key, value in session.gaze_actor.state_dict().items()
    )
    checkpoint = tmp_path / "full_gaze.pt"
    metadata = session.save(checkpoint)
    assert metadata["ground_truth_policy_input"] is True
    assert metadata["policy_observation_contract_id"] == (
        FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
    )
    resumed = PartialTrainingSession(**kwargs(
        gaze_type="learned",
        train_action=False,
        train_gaze=True,
        ground_truth_policy_input=True,
        source_checkpoint=None,
        resume_checkpoint=checkpoint,
        rollout_length=4,
        episode_length=8,
        configured_total_steps=4,
    ))
    assert resumed.completed_steps == 4
    assert resumed.initial_actor_hash == source_actor_hash


def test_direct_joint_from_full_is_explicit_bit_exact_and_trainable(tmp_path):
    _args, policy, _trainer, *_spaces = build_training_stack(
        seed=5, rollout_length=2, n_rollout_threads=1
    )
    source = tmp_path / "shared_full_direct_joint.pt"
    save_action_checkpoint(
        source, actor=policy.actor, critic=policy.critic, mode="full",
        horizon=None, horizon_mode="episode_balanced", horizons=[0, 2, 4],
        step=2, seed=5, condition=cooking_contract(),
    )
    options = kwargs(
        gaze_type="learned", train_action=True, train_gaze=True,
        source_checkpoint=source, horizons=[0, 2, 4],
    )
    with pytest.raises(RuntimeError, match="learned gaze must initialize"):
        PartialTrainingSession(**options)

    session = PartialTrainingSession(
        **options, direct_joint_from_full=True,
    )
    assert session.initial_actor_hash == parameter_hash(policy.actor)
    assert session.source_provenance["initialization"] == (
        DIRECT_JOINT_INITIALIZATION_ID
    )
    assert session.source_provenance["direct_joint_from_full"] is True
    assert session.source_provenance["action_actor_initialization"] == (
        "bit_exact_source_checkpoint"
    )
    assert session.source_provenance["action_critic_initialization"] == (
        "fresh_partial_critic"
    )
    assert session.source_provenance["gaze_actor_initialization"] == "fresh_gazenet"
    result = session.run_update()
    assert result["actor_hash_before"] != result["actor_hash_after"]
    assert result["gaze_hash_before"] != result["gaze_hash_after"]
    output = tmp_path / "direct_joint.pt"
    metadata = session.save(output)
    assert metadata["direct_joint_from_full"] is True
    assert metadata["source_checkpoint_provenance"]["initialization"] == (
        DIRECT_JOINT_INITIALIZATION_ID
    )
    resumed = PartialTrainingSession(**{
        **options,
        "source_checkpoint": None,
        "resume_checkpoint": output,
        "direct_joint_from_full": True,
    })
    assert resumed.completed_steps == session.completed_steps
    assert resumed.initial_actor_hash == metadata["actor_parameter_sha256"]
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        PartialTrainingSession(**{
            **options,
            "source_checkpoint": None,
            "resume_checkpoint": output,
        })


def test_checkpoint_resume_is_exact_and_preserves_custom_schedule(tmp_path):
    options = kwargs(decision_times=[200, 400], fixation_shapes=["plus"])
    uninterrupted = PartialTrainingSession(**options)
    uninterrupted.run_update()
    checkpoint = tmp_path / "partial.pt"
    metadata = uninterrupted.save(checkpoint)
    assert metadata["reward_shaping_horizon_steps"] == 60_000_000
    assert metadata["fixed_partner_identity"] == "hsp_s2_s24_seed3"
    resumed = PartialTrainingSession(**{**options, "resume_checkpoint": checkpoint})
    assert resumed.completed_steps == 2
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    rng = capture_global_rng_state()
    expected = uninterrupted.run_update()
    restore_global_rng_state(rng)
    actual = resumed.run_update()
    assert actual["actor_hash_after"] == expected["actor_hash_after"]
    assert actual["optimization_reward"] == expected["optimization_reward"]
    assert actual["actionnet_queries"] == expected["actionnet_queries"]

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["metadata"]["fixed_partner_sha256"] = "incompatible"
    incompatible = tmp_path / "partial_incompatible_partner.pt"
    torch.save(payload, incompatible)
    with pytest.raises(RuntimeError, match="fixed-partner contract mismatch"):
        PartialTrainingSession(**{**options, "resume_checkpoint": incompatible})


def test_emma_checkpoint_metadata_legacy_resume_and_mismatch(tmp_path):
    options = kwargs(
        rollout_length=1, episode_length=1, configured_total_steps=2
    )
    session = PartialTrainingSession(**options)
    session.run_update()
    current = tmp_path / "current.pt"
    metadata = session.save(current)
    assert {
        key: metadata[key] for key in CANONICAL_TUNED_EMMA_TIMING.metadata()
    } == CANONICAL_TUNED_EMMA_TIMING.metadata()
    assert metadata["constraint_profile"] == "custom"
    assert metadata["constraint_variant_id"] == "e"

    payload = torch.load(current, map_location="cpu", weights_only=False)
    for key in session.constraint_config.metadata():
        payload["metadata"].pop(key)
        payload["training_state"]["contract"].pop(key)
        for adapter in payload["training_state"]["adapters"]:
            adapter.pop(key)
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    resumed = PartialTrainingSession(**{
        **options, "resume_checkpoint": legacy,
        "emma_timing": DEFAULT_EMMA_TIMING,
    })
    assert resumed.emma_timing is DEFAULT_EMMA_TIMING
    assert resumed.constraint_config.variant_id == "b0"

    with pytest.raises(RuntimeError, match="resume constraint config mismatch"):
        PartialTrainingSession(**{
            **options, "resume_checkpoint": current,
            "emma_timing": DEFAULT_EMMA_TIMING,
        })


def test_human_stochastic_emma_uses_per_environment_rng_and_resumes(tmp_path):
    config = peripheral_config(
        "none", None, emma=HUMAN_STOCHASTIC_EMMA_TIMING
    )
    options = kwargs(
        constraint_config=config,
        n_rollout_threads=2,
        rollout_length=2,
        episode_length=2,
        configured_total_steps=4,
    )
    session = PartialTrainingSession(**options)
    assert [adapter.gaze.dwell_seed for adapter in session.adapters] == [
        options["seed"] + HUMAN_DWELL_RNG_SEED_OFFSET + env_index
        for env_index in range(2)
    ]
    assert (
        session.adapters[0].gaze.last_sampled_dwell_ms
        != session.adapters[1].gaze.last_sampled_dwell_ms
    )
    result = session.run_update()
    assert result["actionnet_queries"] == result["task_decisions"]
    assert result["forced_stays"] == 0
    checkpoint = tmp_path / "human_stochastic.pt"
    metadata = session.save(checkpoint)
    assert metadata["emma_mode"] == "human_stochastic"
    assert metadata["emma_dwell_distribution"] == (
        "floor_inflated_capped_lognormal"
    )
    assert metadata["held_foveal_refresh_contract_id"] == (
        HELD_FOVEAL_REFRESH_CONTRACT_ID
    )

    resumed = PartialTrainingSession(**{
        **options, "resume_checkpoint": checkpoint,
    })
    for original, restored in zip(session.adapters, resumed.adapters):
        assert original.gaze.training_state() == restored.gaze.training_state()


def test_old_discrete_partial_checkpoint_cannot_resume_corrected_training(tmp_path):
    options = kwargs(rollout_length=1, episode_length=1, configured_total_steps=1)
    session = PartialTrainingSession(**options)
    session.run_update()
    current = tmp_path / "current_partial.pt"
    metadata = session.save(current)
    assert metadata["observation_contract_id"] == OBSERVATION_CONTRACT_ID
    assert metadata["freshness_contract_id"] == CONTINUOUS_FRESHNESS_CONTRACT_ID

    payload = torch.load(current, map_location="cpu", weights_only=False)
    payload["metadata"]["observation_contract_id"] = (
        "partial_state20_discrete_freshness_configurable_expiry_preview4_v6"
    )
    payload["metadata"].pop("freshness_contract_id")
    payload["training_state"]["contract"]["memory_contract_id"] = (
        "partial_state20_discrete_freshness_configurable_expiry_preview4_v6"
    )
    payload["training_state"]["contract"].pop("freshness_contract_id")
    legacy = tmp_path / "legacy_discrete_partial.pt"
    torch.save(payload, legacy)

    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        PartialTrainingSession(**{**options, "resume_checkpoint": legacy})


def test_learned_modes_freeze_only_untrained_component(tmp_path):
    action = PartialTrainingSession(**kwargs())
    action.run_update()
    action_path = tmp_path / "default_action.pt"
    action.save(action_path)
    action_payload = torch.load(action_path, map_location="cpu", weights_only=False)
    gaze = PartialTrainingSession(**kwargs(
        gaze_type="learned", train_action=False, train_gaze=True,
        source_checkpoint=action_path,
    ))
    assert all(
        torch.equal(gaze.policy.actor.state_dict()[key].cpu(), value.cpu())
        for key, value in action_payload["actor"].items()
    )
    assert all(
        torch.equal(gaze.policy.critic.state_dict()[key].cpu(), value.cpu())
        for key, value in action_payload["critic"].items()
    )
    assert gaze.initial_actor_hash == action_payload["metadata"][
        "actor_parameter_sha256"
    ]
    assert gaze.initial_critic_hash == action_payload["metadata"][
        "critic_parameter_sha256"
    ]
    assert gaze.policy.actor_optimizer is None
    assert gaze.policy.critic_optimizer is None
    assert gaze.action_optimizer_isolation_verified is True
    assert not any(parameter.requires_grad for parameter in gaze.policy.actor.parameters())
    assert not any(parameter.requires_grad for parameter in gaze.policy.critic.parameters())
    gaze_result = gaze.run_update()
    assert gaze_result["actor_hash_before"] == gaze_result["actor_hash_after"]
    assert gaze_result["critic_hash_before"] == gaze_result["critic_hash_after"]
    assert gaze_result["action_actor_optimizer_present"] is False
    assert gaze_result["action_critic_optimizer_present"] is False
    assert gaze_result["gaze_hash_before"] != gaze_result["gaze_hash_after"]
    gaze_path = tmp_path / "default_gaze.pt"
    gaze.save(gaze_path)
    learned_action = PartialTrainingSession(**kwargs(
        gaze_type="learned", train_action=True, train_gaze=False,
        source_checkpoint=gaze_path,
    ))
    result = learned_action.run_update()
    assert result["actor_hash_before"] != result["actor_hash_after"]
    assert result["gaze_hash_before"] == result["gaze_hash_after"]


def test_tuned_action_gaze_and_joint_short_smokes(tmp_path):
    tuned = resolve_emma_timing(
        "tuned", prep_ms=135, exec_intercept_ms=44.8732080135,
        exec_slope_ms_per_degree=3.83832565515, dwell_ms=60,
    )
    smoke = {
        "decision_times": [400], "horizons": [2], "emma_timing": tuned,
    }
    action = PartialTrainingSession(**kwargs(**smoke))
    action.run_update()
    action_path = tmp_path / "action.pt"
    action.save(action_path)
    gaze = PartialTrainingSession(**kwargs(
        gaze_type="learned", train_action=False, train_gaze=True,
        source_checkpoint=action_path,
        **smoke,
    ))
    gaze_result = gaze.run_update()
    assert gaze_result["actor_hash_before"] == gaze_result["actor_hash_after"]
    assert gaze_result["gaze_hash_before"] != gaze_result["gaze_hash_after"]
    gaze_path = tmp_path / "gaze.pt"
    gaze.save(gaze_path)
    gaze_payload = torch.load(gaze_path, map_location="cpu", weights_only=False)
    joint = PartialTrainingSession(**kwargs(
        gaze_type="learned", train_action=True, train_gaze=True,
        source_checkpoint=gaze_path,
        **smoke,
    ))
    assert joint.source_provenance["action_critic_initialization"] == (
        "partial_checkpoint"
    )
    assert all(
        torch.equal(joint.policy.critic.state_dict()[key].cpu(), value.cpu())
        for key, value in gaze_payload["critic"].items()
    )
    assert joint.initial_critic_hash == gaze_payload["metadata"][
        "critic_parameter_sha256"
    ]
    assert not joint.policy.critic_optimizer.state
    result = joint.run_update()
    assert result["actor_hash_before"] != result["actor_hash_after"]
    assert result["critic_hash_before"] != result["critic_hash_after"]
    assert result["gaze_hash_before"] != result["gaze_hash_after"]


def test_learned_action_rejects_source_without_gazenet(tmp_path):
    source = PartialTrainingSession(**kwargs())
    source.run_update()
    path = tmp_path / "action.pt"
    source.save(path)
    with pytest.raises(RuntimeError, match="learned-gaze initialization"):
        PartialTrainingSession(**kwargs(
            gaze_type="learned", train_action=True, train_gaze=False,
            source_checkpoint=path,
        ))


@pytest.mark.parametrize(("mode", "variant"), [("local", "p1"), ("global", "p2")])
def test_peripheral_source_resume_and_all_training_modes(tmp_path, mode, variant):
    smoke = {
        "decision_times": [400], "horizons": [2],
        "fixation_shapes": ["plus"], "rollout_length": 1,
        "episode_length": 1, "configured_total_steps": 3,
    }
    baseline = PartialTrainingSession(**kwargs(**smoke))
    baseline.run_update()
    baseline_path = tmp_path / f"e_{mode}.pt"
    baseline.save(baseline_path)

    config = peripheral_config(mode)
    action = PartialTrainingSession(**kwargs(
        **smoke, source_checkpoint=baseline_path, constraint_config=config,
    ))
    assert action.source_provenance["source_constraint_config"][
        "constraint_variant_id"
    ] == "e"
    assert action.source_provenance["target_constraint_config"][
        "constraint_variant_id"
    ] == variant
    action_result = action.run_update()
    assert action_result["peripheral_attempts"] > 0
    assert action_result["peripheral_successes"] > 0
    action_path = tmp_path / f"{variant}_action.pt"
    metadata = action.save(action_path)
    assert metadata["constraint_variant_id"] == variant
    assert metadata["peripheral_sigma_deg"] == 3.0
    assert metadata["peripheral_perception_contract_id"]
    assert metadata["peripheral_probability_contract_id"]
    assert metadata["peripheral_rng_contract_id"]

    resumed = PartialTrainingSession(**kwargs(
        **smoke, resume_checkpoint=action_path, constraint_config=config,
    ))
    assert resumed.adapters[0].peripheral_rng.bit_generator.state == (
        action.adapters[0].peripheral_rng.bit_generator.state
    )
    other = peripheral_config("global" if mode == "local" else "local")
    with pytest.raises(RuntimeError, match="resume constraint config mismatch"):
        PartialTrainingSession(**kwargs(
            **smoke, resume_checkpoint=action_path, constraint_config=other,
        ))
    with pytest.raises(RuntimeError, match="resume constraint config mismatch"):
        PartialTrainingSession(**kwargs(
            **smoke, resume_checkpoint=action_path,
            constraint_config=peripheral_config(mode, 4.0),
        ))

    gaze = PartialTrainingSession(**kwargs(
        **smoke, gaze_type="learned", train_action=False, train_gaze=True,
        source_checkpoint=action_path, constraint_config=config,
    ))
    gaze_result = gaze.run_update()
    assert gaze_result["peripheral_attempts"] > 0
    gaze_path = tmp_path / f"{variant}_gaze.pt"
    gaze.save(gaze_path)

    joint = PartialTrainingSession(**kwargs(
        **smoke, gaze_type="learned", train_action=True, train_gaze=True,
        source_checkpoint=gaze_path, constraint_config=config,
    ))
    joint_result = joint.run_update()
    assert joint_result["peripheral_attempts"] > 0


@pytest.mark.parametrize(
    ("target_config", "source_config", "variant"),
    [
        (
            peripheral_config("global", opacity="actual"),
            peripheral_config("global"),
            "p2o",
        ),
        (
            FULL_CONSTRAINT_CONFIG,
            HumanLikeConstraintConfig(
                profile="custom", peripheral=PeripheralVisionConfig("none"),
                emma=CANONICAL_TUNED_EMMA_TIMING,
                preview_opacity=PreviewOpacityConfig("off"),
            ),
            "full",
        ),
    ],
)
def test_opacity_source_resume_and_all_training_modes(
    tmp_path, target_config, source_config, variant,
):
    smoke = {
        "decision_times": [400], "horizons": [4],
        "fixation_shapes": ["plus"], "rollout_length": 8,
        "episode_length": 8, "configured_total_steps": 8,
    }
    source = PartialTrainingSession(**kwargs(
        **smoke, constraint_config=source_config,
    ))
    source.run_update()
    source_path = tmp_path / f"{variant}_source.pt"
    source.save(source_path)

    action = PartialTrainingSession(**kwargs(
        **smoke, source_checkpoint=source_path,
        constraint_config=target_config,
    ))
    assert action.source_provenance["target_constraint_config"][
        "constraint_variant_id"
    ] == variant
    action_result = action.run_update()
    assert action_result["opacity_attempts"] > 0
    action_path = tmp_path / f"{variant}_action.pt"
    metadata = action.save(action_path)
    assert metadata["constraint_variant_id"] == variant
    assert metadata["preview_opacity_mode"] == "actual"
    assert metadata["preview_alpha_probability_contract_id"]
    assert metadata["preview_opacity_rng_contract_id"]
    assert metadata["preview_renderer_alpha_h2"] == [250, 70, 0, 0]
    assert metadata["preview_renderer_alpha_h4"] == [250, 190, 130, 70]

    resumed = PartialTrainingSession(**kwargs(
        **smoke, resume_checkpoint=action_path,
        constraint_config=target_config,
    ))
    assert resumed.adapters[0].opacity_rng.bit_generator.state == (
        action.adapters[0].opacity_rng.bit_generator.state
    )
    with pytest.raises(RuntimeError, match="resume constraint config mismatch"):
        PartialTrainingSession(**kwargs(
            **smoke, resume_checkpoint=action_path,
            constraint_config=peripheral_config("global"),
        ))

    gaze = PartialTrainingSession(**kwargs(
        **smoke, gaze_type="learned", train_action=False, train_gaze=True,
        source_checkpoint=action_path, constraint_config=target_config,
    ))
    gaze.run_update()
    gaze_path = tmp_path / f"{variant}_gaze.pt"
    gaze.save(gaze_path)
    joint = PartialTrainingSession(**kwargs(
        **smoke, gaze_type="learned", train_action=True, train_gaze=True,
        source_checkpoint=gaze_path, constraint_config=target_config,
    ))
    result = joint.run_update()
    assert result["opacity_attempts"] > 0


def test_opacity_source_transfer_starts_fresh_target_rng(tmp_path):
    smoke = {
        "decision_times": [400], "horizons": [4],
        "fixation_shapes": ["plus"], "rollout_length": 8,
        "episode_length": 8, "configured_total_steps": 8,
    }
    p2o = peripheral_config("global", opacity="actual")
    source = PartialTrainingSession(**kwargs(
        **smoke, constraint_config=p2o,
    ))
    source.run_update()
    assert source.adapters[0].opacity_rng_draw_count > 0
    source_path = tmp_path / "p2o_source_rng.pt"
    source.save(source_path)

    transferred = PartialTrainingSession(**kwargs(
        **smoke, source_checkpoint=source_path,
        constraint_config=FULL_CONSTRAINT_CONFIG,
    ))
    fresh = PartialTrainingSession(**kwargs(
        **smoke, constraint_config=FULL_CONSTRAINT_CONFIG,
    ))
    assert transferred.adapters[0].opacity_rng.bit_generator.state == (
        fresh.adapters[0].opacity_rng.bit_generator.state
    )
    assert transferred.adapters[0].opacity_rng.bit_generator.state != (
        source.adapters[0].opacity_rng.bit_generator.state
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_p2o_joint_cuda_preserves_actionnet_training_lifecycle(tmp_path):
    smoke = {
        "decision_times": [400], "horizons": [2],
        "fixation_shapes": ["plus"], "rollout_length": 8,
        "episode_length": 8, "configured_total_steps": 8,
        "constraint_config": peripheral_config("global", opacity="actual"),
    }
    action = PartialTrainingSession(**kwargs(**smoke))
    action.run_update()
    action_path = tmp_path / "p2_cuda_action.pt"
    action.save(action_path)

    gaze = PartialTrainingSession(**kwargs(
        **smoke, gaze_type="learned", train_action=False, train_gaze=True,
        source_checkpoint=action_path,
    ))
    gaze.run_update()
    gaze_path = tmp_path / "p2_cuda_gaze.pt"
    gaze.save(gaze_path)

    joint = PartialTrainingSession(**kwargs(
        **smoke, gaze_type="learned", train_action=True, train_gaze=True,
        source_checkpoint=gaze_path, device="cuda",
    ))
    result = joint.run_update()
    assert result["actor_hash_before"] != result["actor_hash_after"]
    assert result["gaze_hash_before"] != result["gaze_hash_after"]
    assert result["peripheral_attempts"] > 0
