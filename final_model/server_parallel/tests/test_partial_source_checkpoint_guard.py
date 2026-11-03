import copy
from pathlib import Path

import pytest
import torch

from final_model.common.partners import parameter_hash
from final_model.common.paths import file_sha256
from final_model.partial_ver.constraints import (
    BASELINE_CONSTRAINT_CONFIG,
    HumanLikeConstraintConfig,
    PeripheralVisionConfig,
    PreviewOpacityConfig,
)
from final_model.partial_ver.gaze import DEFAULT_EMMA_TIMING, resolve_emma_timing
from final_model.partial_ver.canonical_training import PartialTrainingSession
from final_model.server_parallel.parallel_action_training import (
    PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID,
    PARTIAL_SOURCE_VALIDATION_CONTRACT_ID,
    SOURCE_VALIDATION_CONTRACT_ID,
    ServerParallelActionTrainingSession,
    validate_partial_action_source_checkpoint,
    validate_partial_full_source_checkpoint,
    validate_partial_gaze_source_checkpoint,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_120M = (
    ROOT
    / "final_model/checkpoint/action/legacy/action_full_action_network_120M.pt"
)
PRODUCTION_B0_WORKERS4 = (
    ROOT / "final_model/checkpoint/action/legacy/action_baseline_memory1200.pt"
)
DIAMOND_GAZE_13M = (
    ROOT
    / "final_model/checkpoint/gaze/cau_diamond13_memory2400_gaze_13m.pt"
)
PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
RULE = {"task": 0.60, "stay": 0.20, "stale": 0.15, "random": 0.05}


def _peripheral_config(*, opacity: str = "off"):
    return HumanLikeConstraintConfig(
        profile="custom",
        peripheral=PeripheralVisionConfig("global", 3.0),
        emma=DEFAULT_EMMA_TIMING,
        preview_opacity=PreviewOpacityConfig(opacity),
    )


def _diamond_config():
    return HumanLikeConstraintConfig(
        profile="custom",
        peripheral=PeripheralVisionConfig(
            "local", 3.0, extent="diamond13"
        ),
        emma=resolve_emma_timing("default_encoding"),
        preview_opacity=PreviewOpacityConfig("off"),
    )


def _source_options(**overrides):
    options = dict(
        gaze_type="rule_based",
        train_action=True,
        train_gaze=False,
        decision_times=[200],
        horizons=[0, 2, 4],
        fixation_shapes=["plus"],
        seed=20260821,
        rollout_length=1,
        episode_length=1,
        n_rollout_threads=4,
        device="cpu",
        source_checkpoint=SOURCE_120M,
        resume_checkpoint=None,
        fixed_partner_checkpoint=PARTNER,
        configured_total_steps=4,
        checkpoint_interval=4,
        reward_shaping_horizon_steps=20_000_000,
        rule_gaze_probabilities=RULE,
        memory_lifetime_ms=1200,
        constraint_config=BASELINE_CONSTRAINT_CONFIG,
        partner=None,
        ppo_epochs=1,
        num_mini_batches=1,
    )
    options.update(overrides)
    return options


@pytest.fixture(scope="module")
def b0_partial_source(tmp_path_factory):
    checkpoint = tmp_path_factory.mktemp("partial_source") / "b0_canonical.pt"
    session = PartialTrainingSession(
        **_source_options(
            n_rollout_threads=1,
            configured_total_steps=1,
            checkpoint_interval=1,
        )
    )
    session.run_update()
    session.save(checkpoint)
    return checkpoint


def _target_options(source, constraint_config, **overrides):
    options = _source_options(
        source_checkpoint=source,
        constraint_config=constraint_config,
        n_rollout_threads=2,
        configured_total_steps=2,
        checkpoint_interval=2,
    )
    options.update(overrides)
    return options


@pytest.mark.parametrize("source", (SOURCE_120M,))
def test_compatible_canonical_full_sources_are_accepted(source):
    provenance = validate_partial_full_source_checkpoint(
        source,
        requested_horizons=[0, 2, 4],
    )
    assert provenance["source_validation_contract_id"] == (
        SOURCE_VALIDATION_CONTRACT_ID
    )
    assert provenance["source_content_sha256"] == file_sha256(source)
    assert provenance["source_checkpoint_validated"] is True


def test_production_diamond_gaze_source_is_accepted_for_staged_joint():
    provenance = validate_partial_gaze_source_checkpoint(
        DIAMOND_GAZE_13M,
        requested_horizons=[0, 2, 4],
        requested_decision_times=[200, 400, 600],
        requested_fixation_shapes=["plus"],
        target_memory_lifetime_ms=2400.0,
        target_memory_decay_mode="stochastic_exponential",
        target_memory_tau_ms=800.0,
        target_constraint_config=_diamond_config(),
    )
    assert provenance["source_validation_contract_id"] == (
        PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID
    )
    assert provenance["initialization"] == (
        "staged_joint_from_partial_gaze"
    )
    assert provenance["source_checkpoint_step"] == 13_000_000
    assert provenance["source_constraint_config"]["peripheral_extent"] == (
        "diamond13"
    )


def test_diamond_gaze_source_rejects_non_diamond_joint_target():
    with pytest.raises(RuntimeError, match="checkpoint contract mismatch"):
        validate_partial_gaze_source_checkpoint(
            DIAMOND_GAZE_13M,
            requested_horizons=[0, 2, 4],
            requested_decision_times=[200, 400, 600],
            requested_fixation_shapes=["plus"],
            target_memory_lifetime_ms=2400.0,
            target_memory_decay_mode="stochastic_exponential",
            target_memory_tau_ms=800.0,
            target_constraint_config=HumanLikeConstraintConfig(
                profile="custom",
                peripheral=PeripheralVisionConfig(
                    "local", 3.0, extent="diag3x3"
                ),
                emma=resolve_emma_timing("default_encoding"),
                preview_opacity=PreviewOpacityConfig("off"),
            ),
        )


def _mutated_source(tmp_path, name, mutate):
    payload = torch.load(SOURCE_120M, map_location="cpu", weights_only=False)
    payload = copy.deepcopy(payload)
    mutate(payload)
    path = tmp_path / f"{name}.pt"
    torch.save(payload, path)
    return path


@pytest.mark.parametrize(
    ("name", "mutate", "match"),
    (
        (
            "architecture",
            lambda payload: payload["metadata"].__setitem__(
                "architecture_id", "wrong_full_architecture"
            ),
            "architecture_id",
        ),
        (
            "partner",
            lambda payload: payload["metadata"].__setitem__(
                "fixed_partner_sha256", "0" * 64
            ),
            "fixed_partner_sha256",
        ),
        (
            "cooking",
            lambda payload: payload["metadata"].__setitem__(
                "cooking_dynamics_id", "legacy_cooking"
            ),
            "cooking_dynamics_id",
        ),
        (
            "smoke",
            lambda payload: payload["metadata"].__setitem__("smoke", True),
            "production checkpoint",
        ),
        (
            "incomplete",
            lambda payload: payload.pop("training_state"),
            "production checkpoint",
        ),
    ),
)
def test_incompatible_full_sources_fail_closed(tmp_path, name, mutate, match):
    source = _mutated_source(tmp_path, name, mutate)
    with pytest.raises(RuntimeError, match=match):
        validate_partial_full_source_checkpoint(
            source,
            requested_horizons=[0, 2, 4],
        )


def test_selected_40m_migration_save_and_resume_provenance(
    session_options,
    tmp_path,
):
    source_payload = torch.load(SOURCE_120M, map_location="cpu", weights_only=False)
    source_actor_sha256 = source_payload["metadata"]["actor_parameter_sha256"]
    source_content_sha256 = file_sha256(SOURCE_120M)
    options = session_options(
        source_checkpoint=SOURCE_120M,
        decision_times=[600],
        horizons=[0],
        configured_total_steps=2,
    )
    checkpoint = tmp_path / "partial_40m.pt"
    session = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=False,
        **options,
    )
    try:
        assert session.initial_actor_hash == source_actor_sha256
        assert parameter_hash(session.policy.actor) == source_actor_sha256
        assert session.source_provenance["source_content_sha256"] == (
            source_content_sha256
        )
        session.run_update()
        session.save(checkpoint)
    finally:
        session.close()

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    provenance = saved["metadata"]["source_full_checkpoint_provenance"]
    server_state = saved["training_state"]["server_parallel"]
    assert provenance["source_content_sha256"] == source_content_sha256
    assert provenance["source_actor_parameter_sha256"] == source_actor_sha256
    assert server_state["source_content_sha256"] == source_content_sha256
    assert server_state["source_actor_parameter_sha256"] == source_actor_sha256

    resume_options = {
        **options,
        "source_checkpoint": None,
        "resume_checkpoint": checkpoint,
    }
    resumed = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=False,
        **resume_options,
    )
    try:
        assert resumed.source_provenance["source_content_sha256"] == (
            source_content_sha256
        )
    finally:
        resumed.close()

    incompatible = copy.deepcopy(saved)
    incompatible["training_state"]["server_parallel"][
        "source_content_sha256"
    ] = "0" * 64
    incompatible_path = tmp_path / "partial_different_source.pt"
    torch.save(incompatible, incompatible_path)
    with pytest.raises(RuntimeError, match="source provenance mismatch"):
        ServerParallelActionTrainingSession(
            rollout_workers=2,
            server_trace=False,
            **{
                **resume_options,
                "resume_checkpoint": incompatible_path,
            },
        )


@pytest.mark.parametrize(
    ("constraint_config", "target_variant"),
    (
        (_peripheral_config(), "p2"),
        (_peripheral_config(opacity="actual"), "p2o"),
    ),
)
def test_b0_partial_source_initializes_new_constraint_variant(
    b0_partial_source,
    constraint_config,
    target_variant,
):
    session = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=False,
        **_target_options(b0_partial_source, constraint_config),
    )
    try:
        assert session.source_provenance["source_constraint_config"][
            "constraint_variant_id"
        ] == "b0"
        assert session.source_provenance["target_constraint_config"][
            "constraint_variant_id"
        ] == target_variant
        assert session.source_provenance["source_kind"] == (
            "canonical_partial_action"
        )
        assert session.source_provenance["source_absolute_path"] == str(
            b0_partial_source.resolve()
        )
    finally:
        session.close()


def test_partial_source_actor_initialization_is_bit_exact(b0_partial_source):
    payload = torch.load(
        b0_partial_source, map_location="cpu", weights_only=False
    )
    session = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=False,
        **_target_options(b0_partial_source, _peripheral_config()),
    )
    try:
        loaded = session.policy.actor.state_dict()
        assert loaded.keys() == payload["actor"].keys()
        assert all(
            torch.equal(loaded[key].cpu(), payload["actor"][key].cpu())
            for key in loaded
        )
        expected_hash = payload["metadata"]["actor_parameter_sha256"]
        assert session.initial_actor_hash == expected_hash
        assert parameter_hash(session.policy.actor) == expected_hash
        assert session.source_provenance[
            "source_actor_parameter_sha256"
        ] == expected_hash
    finally:
        session.close()


def test_gaze_stage_loads_and_freezes_source_action_actor_and_critic(
    b0_partial_source,
):
    payload = torch.load(
        b0_partial_source, map_location="cpu", weights_only=False
    )
    session = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=False,
        **_target_options(
            b0_partial_source,
            _peripheral_config(),
            gaze_type="learned",
            train_action=False,
            train_gaze=True,
        ),
    )
    try:
        assert all(
            torch.equal(session.policy.actor.state_dict()[key].cpu(), value.cpu())
            for key, value in payload["actor"].items()
        )
        assert all(
            torch.equal(session.policy.critic.state_dict()[key].cpu(), value.cpu())
            for key, value in payload["critic"].items()
        )
        assert session.source_provenance[
            "source_critic_parameter_sha256"
        ] == payload["metadata"]["critic_parameter_sha256"]
        assert session.policy.actor_optimizer is None
        assert session.policy.critic_optimizer is None
        assert not any(
            parameter.requires_grad for parameter in session.policy.actor.parameters()
        )
        assert not any(
            parameter.requires_grad for parameter in session.policy.critic.parameters()
        )
        startup = session.startup_report()
        assert startup["action_requires_grad"] is False
        assert startup["action_critic_requires_grad"] is False
        assert startup["action_optimizer_present"] is False
        assert startup["action_critic_optimizer_present"] is False
        assert startup["action_optimizer_isolation_verified"] is True
    finally:
        session.close()


def test_partial_source_workers4_can_initialize_new_workers16_run(
):
    payload = torch.load(
        PRODUCTION_B0_WORKERS4, map_location="cpu", weights_only=False
    )
    assert payload["metadata"]["rollout_workers"] == 4
    session = ServerParallelActionTrainingSession(
        rollout_workers=16,
        server_trace=False,
        **_target_options(
            PRODUCTION_B0_WORKERS4,
            _peripheral_config(),
            n_rollout_threads=16,
            configured_total_steps=16,
            checkpoint_interval=16,
        ),
    )
    try:
        assert session.rollout_workers == 16
        assert session.completed_steps == 0
        assert session.source_provenance["source_kind"] == (
            "canonical_partial_action"
        )
    finally:
        session.close()


def test_partial_resume_still_rejects_worker_count_change():
    with pytest.raises(
        RuntimeError, match="resume execution contract mismatch"
    ):
        ServerParallelActionTrainingSession(
            rollout_workers=16,
            server_trace=False,
            **_source_options(
                source_checkpoint=None,
                resume_checkpoint=PRODUCTION_B0_WORKERS4,
                n_rollout_threads=16,
                configured_total_steps=20_000_000,
                checkpoint_interval=1_000_000,
            ),
        )


def _mutated_partial_source(tmp_path, source, name, mutate):
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload = copy.deepcopy(payload)
    mutate(payload)
    path = tmp_path / f"partial_{name}.pt"
    torch.save(payload, path)
    return path


@pytest.mark.parametrize(
    ("name", "mutate", "match"),
    (
        (
            "observation",
            lambda payload: payload["metadata"].__setitem__(
                "observation_contract_id", "old_partial_observation"
            ),
            "observation_contract_id",
        ),
        (
            "memory",
            lambda payload: payload["metadata"].__setitem__(
                "memory_lifetime_ms", 2400.0
            ),
            "memory_lifetime_ms",
        ),
        (
            "actor_hash",
            lambda payload: payload["metadata"].__setitem__(
                "actor_parameter_sha256", "0" * 64
            ),
            "actor parameter hash mismatch",
        ),
        (
            "critic_hash",
            lambda payload: payload["metadata"].__setitem__(
                "critic_parameter_sha256", "0" * 64
            ),
            "critic parameter hash mismatch",
        ),
        (
            "incomplete",
            lambda payload: payload.pop("training_state"),
            "complete production checkpoint",
        ),
        (
            "smoke",
            lambda payload: payload["metadata"].__setitem__("smoke", True),
            "production checkpoint",
        ),
        (
            "pre_training",
            lambda payload: payload["metadata"].__setitem__(
                "checkpoint_role", "pre_training_initialization"
            ),
            "production checkpoint",
        ),
    ),
)
def test_malformed_partial_sources_fail_closed(
    tmp_path,
    b0_partial_source,
    name,
    mutate,
    match,
):
    source = _mutated_partial_source(
        tmp_path, b0_partial_source, name, mutate
    )
    with pytest.raises(RuntimeError, match=match):
        validate_partial_action_source_checkpoint(
            source,
            requested_horizons=[0, 2, 4],
            target_memory_lifetime_ms=1200,
        )


def test_partial_source_provenance_uses_generic_and_legacy_keys(
    b0_partial_source,
    tmp_path,
):
    session = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=False,
        **_target_options(b0_partial_source, _peripheral_config()),
    )
    checkpoint = tmp_path / "p2_from_partial.pt"
    try:
        session.run_update()
        session.save(checkpoint)
    finally:
        session.close()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    generic = payload["metadata"]["source_checkpoint_provenance"]
    legacy = payload["metadata"]["source_full_checkpoint_provenance"]
    assert generic == legacy
    assert generic["source_validation_contract_id"] == (
        PARTIAL_SOURCE_VALIDATION_CONTRACT_ID
    )
    assert generic["source_content_sha256"] == file_sha256(b0_partial_source)
    assert generic["source_actor_parameter_sha256"] == torch.load(
        b0_partial_source, map_location="cpu", weights_only=False
    )["metadata"]["actor_parameter_sha256"]
    resumed = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=False,
        **_target_options(
            None,
            _peripheral_config(),
            resume_checkpoint=checkpoint,
        ),
    )
    try:
        assert resumed.source_provenance == generic
        assert resumed.completed_steps == 2
    finally:
        resumed.close()
