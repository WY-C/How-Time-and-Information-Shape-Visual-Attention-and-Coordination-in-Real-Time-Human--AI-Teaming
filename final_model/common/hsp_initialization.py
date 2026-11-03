"""Canonical HSP-S2-s36 seed-4 learner migration into the 25-channel Full ActionNet."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from final_model.common.mappo import ACTION_ARCHITECTURE_ID, ACTOR_FORWARD_CONTEXT_ID
from final_model.common.partners import parameter_hash
from final_model.common.paths import file_sha256, policy_config_directory
from final_model.common.reward import HSP_REWARD_SHAPING_HORIZON
from final_model.common.zsceval_compat import install_zsceval, load_policy_config
from final_model.simple_ver.observation import OBSERVATION_SCALE


HSP_INITIALIZATION_TYPE = "hsp_s2_s36_seed4_20ch_to_25ch_function_preserving"
HSP_SOURCE_ARCHITECTURE_ID = "canonical_hsp_s2_R_Actor_8x5x20_gru64_categorical6_v1"
HSP_SOURCE_CHECKPOINT_SHA256 = (
    "73b0be7e6e7f818d97a8292e0b9d040e6e6bbcaeda4dd8e6954f7cee8797ea15"
)
HSP_POLICY_CONFIG_SHA256 = (
    "734926487967c3c287389cd090cfdb22d76840c7c986ea83695778d873bc6906"
)
FIRST_CONV_KEY = "base.cnn.0.weight"
FIRST_CONV_BIAS_KEY = "base.cnn.0.bias"
SOURCE_INPUT_CHANNELS = 20
TARGET_INPUT_CHANNELS = 25
SOURCE_PARAMETER_COUNT = 158918
TARGET_PARAMETER_COUNT = 160358
MIGRATION_ATOL = 1e-6
MIGRATION_RTOL = 1e-6


class FullFreshnessWeightGuard:
    """Keep only Full ActionNet first-conv input channel 20 at exact zero."""

    def __init__(self, actor: torch.nn.Module, optimizer: torch.optim.Optimizer):
        parameters = dict(actor.named_parameters())
        if FIRST_CONV_KEY not in parameters:
            raise RuntimeError(f"Full ActionNet has no {FIRST_CONV_KEY}")
        self.weight = parameters[FIRST_CONV_KEY]
        if tuple(self.weight.shape) != (32, 25, 3, 3):
            raise RuntimeError(
                f"Full freshness guard shape mismatch: {tuple(self.weight.shape)}"
            )
        self.optimizer = optimizer
        self.gradient_hook_calls = 0
        self.weight.register_hook(self._mask_gradient)

    def _mask_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        result = gradient.clone()
        result[:, 20, :, :].zero_()
        self.gradient_hook_calls += 1
        return result

    def initialize_zero(self) -> None:
        with torch.no_grad():
            self.weight[:, 20, :, :].zero_()
        self.assert_zero("initialization")

    def _optimizer_channel_nonzero(self) -> list[str]:
        hits = []
        state = self.optimizer.state.get(self.weight, {})
        for name, value in state.items():
            if torch.is_tensor(value) and tuple(value.shape) == tuple(self.weight.shape):
                if torch.count_nonzero(value[:, 20, :, :]).item():
                    hits.append(str(name))
        return hits

    def assert_zero(self, context: str) -> None:
        count = int(torch.count_nonzero(self.weight[:, 20, :, :]).item())
        if count:
            raise RuntimeError(
                f"Full freshness frozen-zero contract violation at {context}: "
                f"{count} nonzero weights"
            )
        optimizer_hits = self._optimizer_channel_nonzero()
        if optimizer_hits:
            raise RuntimeError(
                "Full freshness optimizer-state contract violation at "
                f"{context}: {optimizer_hits}"
            )

    def after_optimizer_update(self) -> None:
        """Fail closed; the gradient hook must prevent any weight or state update."""
        self.assert_zero("after_optimizer_update")

    def metrics(self) -> dict[str, float]:
        weight = self.weight.detach()
        return {
            "freshness_weight_norm": float(torch.linalg.vector_norm(weight[:, 20]).cpu()),
            **{
                f"preview_weight_norm_h{offset}": float(
                    torch.linalg.vector_norm(weight[:, 20 + offset]).cpu()
                )
                for offset in range(1, 5)
            },
        }


def _canonical_policy_contract() -> tuple[Any, Any, Any, dict[str, Any]]:
    config_path = policy_config_directory() / "rnn_policy_config.pkl"
    config_sha256 = file_sha256(config_path)
    if config_sha256 != HSP_POLICY_CONFIG_SHA256:
        raise RuntimeError(
            "canonical HSP policy config SHA-256 mismatch: "
            f"{config_sha256} != {HSP_POLICY_CONFIG_SHA256}"
        )
    args, observation_space, _share_space, action_space = load_policy_config("rnn")
    actual = {
        "input_shape": tuple(observation_space.shape),
        "recurrent": bool(args.use_recurrent_policy),
        "recurrent_n": int(args.recurrent_N),
        "hidden_size": int(args.hidden_size),
        "action_count": int(action_space.n),
    }
    expected = {
        "input_shape": (8, 5, SOURCE_INPUT_CHANNELS),
        "recurrent": True,
        "recurrent_n": 1,
        "hidden_size": 64,
        "action_count": 6,
    }
    if actual != expected:
        raise RuntimeError(f"canonical HSP recurrent policy config mismatch: {actual}")
    return args, observation_space, action_space, {
        **actual,
        "policy_config_sha256": config_sha256,
        "policy_config_name": "random3/policy_config/rnn_policy_config.pkl",
    }


def _source_state(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, torch.Tensor], str]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"HSP initial checkpoint does not exist: {source}")
    source_sha256 = file_sha256(source)
    if source_sha256 != expected_sha256:
        raise RuntimeError(
            "HSP checkpoint does not match the required canonical identity: "
            f"sha256={source_sha256}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not payload:
        raise RuntimeError("canonical HSP checkpoint must be a non-empty actor state_dict")
    state = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise RuntimeError("canonical HSP checkpoint contains a non-tensor state entry")
        state[key] = value.detach().cpu()
    return state, source_sha256


def build_canonical_hsp_actor(
    checkpoint: Path,
    *,
    device: torch.device,
    expected_sha256: str = HSP_SOURCE_CHECKPOINT_SHA256,
    expected_seed: int = 4,
    expected_population: int = 36,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, torch.Tensor]]:
    source_state, source_sha256 = _source_state(
        checkpoint, expected_sha256=expected_sha256
    )
    args, observation_space, action_space, config = _canonical_policy_contract()
    install_zsceval()
    from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        actor = R_Actor(args, observation_space, action_space, device=device)
        actor.load_state_dict(source_state, strict=True)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    first_conv = actor.state_dict().get(FIRST_CONV_KEY)
    if first_conv is None or tuple(first_conv.shape) != (32, 20, 3, 3):
        raise RuntimeError("canonical HSP actor first convolution contract mismatch")
    first_conv_bias = actor.state_dict().get(FIRST_CONV_BIAS_KEY)
    if first_conv_bias is None or tuple(first_conv_bias.shape) != (32,):
        raise RuntimeError("canonical HSP actor first-convolution bias contract mismatch")
    if actor._recurrent_N != 1 or actor.hidden_size != 64:
        raise RuntimeError("canonical HSP actor is not GRU64 recurrent_N=1")
    actor.eval().requires_grad_(False)
    source_parameter_count = sum(value.numel() for value in source_state.values())
    if source_parameter_count != SOURCE_PARAMETER_COUNT:
        raise RuntimeError(
            f"canonical HSP actor parameter count mismatch: {source_parameter_count}"
        )
    return actor, {
        "hsp_source_sha256": source_sha256,
        "source_architecture_id": HSP_SOURCE_ARCHITECTURE_ID,
        "source_parameter_count": source_parameter_count,
        "source_seed": int(expected_seed),
        "source_population": int(expected_population),
        "source_environment_contract": "random3/counter_circuit_old_dynamics",
        "source_learner_agent_index": 1,
        "source_action_distribution": "categorical6_with_original_legal_action_mask",
        **config,
    }, source_state


def _actor_outputs(actor, observation, hidden, masks, available):
    device = next(actor.parameters()).device
    obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
    rnn = torch.as_tensor(hidden, dtype=torch.float32, device=device)
    mask = torch.as_tensor(masks, dtype=torch.float32, device=device)
    legal = torch.as_tensor(available, dtype=torch.float32, device=device)
    features = actor.base(obs)
    features, next_hidden = actor.rnn(features, rnn, mask)
    logits = actor.act.action_out.linear(features)
    probabilities = actor.act.action_out(features, legal).probs
    return logits, probabilities, next_hidden


def function_preservation_audit(source_actor, target_actor) -> dict[str, Any]:
    batch = 8
    generator = torch.Generator(device="cpu").manual_seed(20260821)
    state20 = (
        torch.rand((batch, 8, 5, 20), generator=generator) < 0.02
    ).float().mul(OBSERVATION_SCALE)
    preview = (
        torch.rand((batch, 8, 5, 4), generator=generator) < 0.05
    ).float().mul(OBSERVATION_SCALE)
    freshness = torch.full((batch, 8, 5, 1), OBSERVATION_SCALE)
    zeros = torch.zeros_like(preview)
    inputs = {
        "H0": torch.cat((state20, freshness, zeros), dim=-1),
        "H2": torch.cat((state20, freshness, preview.clone()), dim=-1),
        "H4": torch.cat((state20, freshness, preview), dim=-1),
    }
    inputs["H2"][..., 23:25] = 0.0
    hidden = torch.linspace(-0.5, 0.5, batch * 64).reshape(batch, 1, 64)
    masks = torch.ones((batch, 1), dtype=torch.float32)
    available = torch.ones((batch, 6), dtype=torch.float32)
    available[::2, 0] = 0.0
    source_was_training = source_actor.training
    target_was_training = target_actor.training
    source_actor.eval()
    target_actor.eval()
    per_horizon = {}
    with torch.no_grad():
        source_logits, source_probs, source_hidden = _actor_outputs(
            source_actor, state20, hidden, masks, available
        )
        for horizon, observation in inputs.items():
            logits, probabilities, next_hidden = _actor_outputs(
                target_actor, observation, hidden, masks, available
            )
            kl = torch.sum(
                source_probs
                * (
                    torch.log(source_probs.clamp_min(1e-12))
                    - torch.log(probabilities.clamp_min(1e-12))
                ),
                dim=-1,
            )
            per_horizon[horizon] = {
                "max_abs_logit_difference": float(
                    torch.max(torch.abs(source_logits - logits)).cpu()
                ),
                "max_abs_probability_difference": float(
                    torch.max(torch.abs(source_probs - probabilities)).cpu()
                ),
                "kl_original_to_migrated": float(torch.max(torch.abs(kl)).cpu()),
                "top1_action_agreement": float(
                    (source_probs.argmax(-1) == probabilities.argmax(-1))
                    .float()
                    .mean()
                    .mul(100.0)
                    .cpu()
                ),
                "max_abs_next_hidden_difference": float(
                    torch.max(torch.abs(source_hidden - next_hidden)).cpu()
                ),
            }
    source_actor.train(source_was_training)
    target_actor.train(target_was_training)
    maxima = {
        key: max(values[key] for values in per_horizon.values())
        for key in (
            "max_abs_logit_difference",
            "max_abs_probability_difference",
            "kl_original_to_migrated",
            "max_abs_next_hidden_difference",
        )
    }
    minimum_agreement = min(
        values["top1_action_agreement"] for values in per_horizon.values()
    )
    verified = (
        minimum_agreement == 100.0
        and maxima["max_abs_logit_difference"] <= MIGRATION_ATOL
        and maxima["max_abs_probability_difference"] <= MIGRATION_ATOL
        and maxima["kl_original_to_migrated"] <= MIGRATION_ATOL
        and maxima["max_abs_next_hidden_difference"] <= MIGRATION_ATOL
    )
    if not verified:
        raise RuntimeError(f"HSP function-preservation audit failed: {per_horizon}")
    return {
        **maxima,
        "top1_action_agreement": minimum_agreement,
        "atol": MIGRATION_ATOL,
        "rtol": MIGRATION_RTOL,
        "per_horizon": per_horizon,
        "verified": True,
    }


def migrate_hsp_s2_actor(checkpoint: Path, *, target_actor) -> dict[str, Any]:
    source_actor, source_metadata, source_state = build_canonical_hsp_actor(
        checkpoint, device=next(target_actor.parameters()).device
    )
    target_state = target_actor.state_dict()
    if set(source_state) != set(target_state):
        raise RuntimeError(
            "HSP/Full actor state keys differ: "
            f"source_only={sorted(set(source_state) - set(target_state))}, "
            f"target_only={sorted(set(target_state) - set(source_state))}"
        )
    migrated = {}
    for key, target_value in target_state.items():
        source_value = source_state[key]
        if key == FIRST_CONV_KEY:
            if tuple(source_value.shape) != (32, 20, 3, 3) or tuple(
                target_value.shape
            ) != (32, 25, 3, 3):
                raise RuntimeError("unexpected HSP/Full first-convolution shapes")
            value = torch.zeros_like(target_value)
            value[:, :SOURCE_INPUT_CHANNELS].copy_(
                source_value.to(device=value.device, dtype=value.dtype)
            )
            migrated[key] = value
        else:
            if source_value.shape != target_value.shape:
                raise RuntimeError(
                    f"unexpected HSP/Full actor shape mismatch at {key}: "
                    f"{tuple(source_value.shape)} != {tuple(target_value.shape)}"
                )
            migrated[key] = source_value.to(
                device=target_value.device, dtype=target_value.dtype
            ).clone()
    target_actor.load_state_dict(migrated, strict=True)
    migrated_state = target_actor.state_dict()
    if not torch.equal(
        migrated_state[FIRST_CONV_KEY][:, :SOURCE_INPUT_CHANNELS].cpu(),
        source_state[FIRST_CONV_KEY],
    ):
        raise RuntimeError("HSP first-convolution source channels were not copied exactly")
    if torch.count_nonzero(
        migrated_state[FIRST_CONV_KEY][:, SOURCE_INPUT_CHANNELS:]
    ).item() != 0:
        raise RuntimeError("HSP migration extra input-channel weights are not exact zero")
    for key, source_value in source_state.items():
        if key != FIRST_CONV_KEY and not torch.equal(
            migrated_state[key].detach().cpu(), source_value
        ):
            raise RuntimeError(f"HSP actor parameter was not copied exactly: {key}")
    target_parameter_count = sum(value.numel() for value in migrated_state.values())
    if target_parameter_count != TARGET_PARAMETER_COUNT:
        raise RuntimeError(
            f"Full actor parameter count mismatch after migration: {target_parameter_count}"
        )
    audit = function_preservation_audit(source_actor, target_actor)
    source = Path(checkpoint).expanduser().resolve()
    return {
        "initialization_type": HSP_INITIALIZATION_TYPE,
        "learner_hsp_source_checkpoint": str(source),
        "learner_hsp_source_sha256": source_metadata["hsp_source_sha256"],
        "learner_hsp_source_seed": 4,
        "learner_hsp_source_population": 36,
        **source_metadata,
        "source_input_channels": SOURCE_INPUT_CHANNELS,
        "target_input_channels": TARGET_INPUT_CHANNELS,
        "migration_parameter_key": FIRST_CONV_KEY,
        "first_conv_bias_key": FIRST_CONV_BIAS_KEY,
        "first_conv_bias_copied": True,
        "source_first_conv_shape": list(source_state[FIRST_CONV_KEY].shape),
        "target_first_conv_shape": list(migrated_state[FIRST_CONV_KEY].shape),
        "copied_channels": [0, 19],
        "zero_initialized_channels": [20, 24],
        "channel_20_semantics": "full freshness maximum",
        "channel_20_value": OBSERVATION_SCALE,
        "migration_verified": True,
        "function_preservation": audit,
        "target_architecture_id": ACTION_ARCHITECTURE_ID,
        "target_parameter_count": target_parameter_count,
        "initial_actor_parameter_sha256": parameter_hash(target_actor),
        "absolute_path_used_for_compatibility": False,
    }


def hsp_initialization_report() -> dict[str, Any]:
    return {
        "initialization_type": HSP_INITIALIZATION_TYPE,
        "canonical_source_relative_path": "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt",
        "canonical_source_sha256": HSP_SOURCE_CHECKPOINT_SHA256,
        "policy_config_sha256": HSP_POLICY_CONFIG_SHA256,
        "source_architecture_id": HSP_SOURCE_ARCHITECTURE_ID,
        "source_environment_contract": "random3/counter_circuit_old_dynamics",
        "source_learner_agent_index": 1,
        "source_action_distribution": "categorical6_with_original_legal_action_mask",
        "target_architecture_id": ACTION_ARCHITECTURE_ID,
        "migration_parameter_key": FIRST_CONV_KEY,
        "first_conv_bias_key": FIRST_CONV_BIAS_KEY,
        "first_conv_bias_copied": True,
        "source_first_conv_shape": [32, 20, 3, 3],
        "target_first_conv_shape": [32, 25, 3, 3],
        "source_parameter_count": SOURCE_PARAMETER_COUNT,
        "target_parameter_count": TARGET_PARAMETER_COUNT,
        "copied_channels": [0, 19],
        "zero_initialized_channels": [20, 24],
        "full_freshness_channel20_frozen_zero": True,
        "preview_channels_21_24_trainable": True,
        "tolerance": {"atol": MIGRATION_ATOL, "rtol": MIGRATION_RTOL},
    }


def preview_learnability_audit(checkpoint: Path) -> dict[str, Any]:
    """Run one bounded update and prove that active preview weights can leave zero."""
    from final_model.common.mappo import build_actionnet

    results = {}
    for horizon in (2, 4):
        actor = build_actionnet(seed=20260821, device="cpu")
        migrate_hsp_s2_actor(checkpoint, target_actor=actor)
        actor.train()
        optimizer = torch.optim.SGD(actor.parameters(), lr=1e-2)
        generator = torch.Generator(device="cpu").manual_seed(20260821 + horizon)
        state = (
            torch.rand((32, 8, 5, 20), generator=generator) < 0.02
        ).float().mul(OBSERVATION_SCALE)
        freshness = torch.full((32, 8, 5, 1), OBSERVATION_SCALE)
        preview = (
            torch.rand((32, 8, 5, 4), generator=generator) < 0.05
        ).float().mul(OBSERVATION_SCALE)
        preview[..., horizon:] = 0.0
        observation = torch.cat((state, freshness, preview), dim=-1)
        hidden = torch.zeros((32, 1, 64), dtype=torch.float32)
        masks = torch.ones((32, 1), dtype=torch.float32)
        available = torch.ones((32, 6), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits, _probabilities, _next_hidden = _actor_outputs(
            actor, observation, hidden, masks, available
        )
        loss = logits.square().mean() + logits[:, 0].mean()
        loss.backward()
        gradient = actor.base.cnn[0].weight.grad.detach().clone()
        active = gradient[:, 21 : 21 + horizon]
        if torch.count_nonzero(active).item() == 0:
            raise RuntimeError(f"H{horizon} active preview channels received zero gradient")
        before = actor.state_dict()[FIRST_CONV_KEY].detach().clone()
        optimizer.step()
        after = actor.state_dict()[FIRST_CONV_KEY].detach()
        changed = after[:, 21 : 21 + horizon] - before[:, 21 : 21 + horizon]
        if torch.count_nonzero(changed).item() == 0:
            raise RuntimeError(f"H{horizon} active preview weights did not update")
        results[f"H{horizon}"] = {
            "active_preview_channels": list(range(21, 21 + horizon)),
            "nonzero_active_gradient_count": int(torch.count_nonzero(active).item()),
            "max_abs_active_gradient": float(active.abs().max().cpu()),
            "changed_active_weight_count": int(torch.count_nonzero(changed).item()),
            "max_abs_active_weight_change": float(changed.abs().max().cpu()),
            "bounded_optimizer_steps": 1,
            "verified": True,
        }
    return results


def run_hsp_initialization_audit(checkpoint: Path) -> dict[str, Any]:
    from final_model.common.mappo import build_actionnet

    migrations = {}
    hashes = {}
    for horizon in (0, 2, 4):
        actor = build_actionnet(seed=20260821, device="cpu")
        migrations[f"H{horizon}"] = migrate_hsp_s2_actor(
            checkpoint, target_actor=actor
        )
        hashes[f"H{horizon}"] = parameter_hash(actor)
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"H0/H2/H4 migrated actor hashes differ: {hashes}")
    return {
        **hsp_initialization_report(),
        "source_checkpoint_used": "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt",
        "initial_actor_hashes": hashes,
        "identical_initial_actor_hashes": True,
        "function_preservation": migrations["H0"]["function_preservation"],
        "preview_learnability": preview_learnability_audit(checkpoint),
    }


def full_fixed_partner_contract_report(
    learner_checkpoint: Path,
    partner_checkpoint: Path,
    *,
    reward_shaping_horizon_steps: int = HSP_REWARD_SHAPING_HORIZON,
) -> dict[str, Any]:
    from final_model.common.partners import (
        FIXED_HSP_PARTNER_ID,
        FIXED_HSP_PARTNER_SHA256,
        FixedHSPPartnerRepository,
    )

    learner = Path(learner_checkpoint).expanduser().resolve()
    partner = Path(partner_checkpoint).expanduser().resolve()
    if int(reward_shaping_horizon_steps) < 0:
        raise ValueError("reward_shaping_horizon_steps cannot be negative")
    learner_sha256 = file_sha256(learner)
    partner_sha256 = file_sha256(partner)
    if learner_sha256 != HSP_SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("Full learner audit source is not canonical seed-4")
    if partner_sha256 != FIXED_HSP_PARTNER_SHA256:
        raise RuntimeError("Full partner audit source is not canonical seed-3")
    if learner_sha256 == partner_sha256:
        raise RuntimeError("Full learner and fixed partner checkpoints are identical")
    repository = FixedHSPPartnerRepository(device="cpu", checkpoint=partner)
    partner_actor = repository.actor(FIXED_HSP_PARTNER_ID)
    frozen_hash = repository.assert_frozen(FIXED_HSP_PARTNER_ID, full_hash=True)
    migration = run_hsp_initialization_audit(learner)
    return {
        "contract_id": "full_s36_seed4_init_fixed_s24_seed3_partner_freshness_frozen_v1",
        "learner_initialization_checkpoint": "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt",
        "learner_initialization_sha256": learner_sha256,
        "learner_seed": 4,
        "learner_population": 36,
        "fixed_partner_checkpoint": "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt",
        "fixed_partner_sha256": partner_sha256,
        "fixed_partner_seed": 3,
        "fixed_partner_population": 24,
        "learner_source_equals_partner": False,
        "fixed_partner_parameter_sha256": frozen_hash,
        "fixed_partner_frozen": not any(
            parameter.requires_grad for parameter in partner_actor.parameters()
        ),
        "fixed_partner_pool_size": 1,
        "fixed_partner_sampling_disabled": True,
        "migration": migration,
        "full_channel20": {
            "input_value": OBSERVATION_SCALE,
            "input_shape": [8, 5],
            "first_conv_weight_slice": "base.cnn.0.weight[:,20,:,:]",
            "initial_weight_norm": 0.0,
            "frozen_exact_zero": True,
            "resume_nonzero_behavior": "fail_closed",
        },
        "preview_weight_norm_definitions": {
            f"train/preview_weight_norm_h{offset}": (
                f"L2(base.cnn.0.weight[:,{20 + offset},:,:])"
            )
            for offset in range(1, 5)
        },
        "predictor_partner_identity": {
            "actual_partner_agent_index": 0,
            "predictor_target_agent_index": 0,
            "learner_agent_index": 1,
            "history": "actual fixed seed-3 agent-0 position history",
            "predictor_checkpoint_sha256": (
                "a7c833d8262fed44c9209e090e92023e3d50cf1f9ffbc722a7418786529d6474"
            ),
            "predictor_targets_learner_seed4": False,
        },
        "step0_checkpoint_contract": {
            "filename": "checkpoints/step_000000000.pt",
            "role": "pre_training_initialization",
            "saved_before_environment_or_optimizer_step": True,
            "production_artifacts_created_only_when_training_command_starts": True,
        },
        "production_training": {
            "configured_total_steps": 20_000_000,
            "n_rollout_threads": 100,
            "rollout_length": 400,
            "episode_length": 400,
            "transitions_per_update": 40_000,
            "ppo_updates": 500,
            "checkpoint_interval": 1_000_000,
            "entropy_schedule": [0.2, 0.05, 0.01],
            "entropy_schedule_horizons": [0, 10_000_000, 20_000_000],
            "reward_shaping_schedule": "linear",
            "reward_shaping_initial": 1.0,
            "reward_shaping_final": 0.0,
            "reward_shaping_horizon_steps": int(reward_shaping_horizon_steps),
            "checkpoint_schedule_state": [
                "global_environment_step",
                "entropy_coef",
                "reward_shaping_factor",
            ],
            "resume_schedule_state_validation": "fail_closed",
            "rollout_policy_consistency_audit": "pre_optimizer_fail_closed_v2",
            "observation_capture_storage": "independent_copy_and_uint8_sha256_v2",
            "recurrent_minibatch_order": "time_major_within_shuffled_env_chunks_v1",
            "actor_forward_context": ACTOR_FORWARD_CONTEXT_ID,
            "actor_forward_microbatch_size": 100,
            "train_time_evaluation": False,
        },
        "overall_verdict": "PASS",
    }
