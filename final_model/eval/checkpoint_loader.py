"""Strict optimizer-free loader for current fixed/shared Full Action checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch

from final_model.checkpoint import ACTION_SPACE, FORMAT_ID, state_dict_hash
from final_model.common.mappo import ACTION_ARCHITECTURE_ID, FULL_OBSERVATION_CONTRACT, build_actionnet
from final_model.common.overcooked_backend import CANONICAL_ENCODER_ID, cooking_contract
from final_model.common.partners import (
    FIXED_HSP_PARTNER_CONTRACT_ID,
    FIXED_HSP_PARTNER_ID,
    FIXED_HSP_PARTNER_SHA256,
    PARTNER_SET_ID,
    parameter_hash,
)
from final_model.common.paths import file_sha256
from final_model.common.zsceval_compat import load_policy_config
from final_model.eval.common import EVALUATION_HORIZONS
from final_model.partial_ver.canonical_training import PARTIAL_CHECKPOINT_FORMAT_ID
from final_model.partial_ver.constraints import (
    HumanLikeConstraintConfig,
    constraint_config_from_metadata,
)
from final_model.partial_ver.gaze import EmmaTimingConfig
from final_model.partial_ver.memory import (
    CONTINUOUS_FRESHNESS_CONTRACT_ID,
    NO_DECAY_FRESHNESS_CONTRACT_ID,
    OBSERVATION_CONTRACT_ID,
    STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID,
    observation_contract_id,
)
from final_model.partial_ver.network import GAZE_ARCHITECTURE_ID, GazeCritic, GazeNet
from final_model.partial_ver.observation import (
    FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID,
)
from final_model.partial_ver.training import (
    gaze_action_evaluation_status,
    gaze_credit_evaluation_status,
    gaze_input_evaluation_status,
)


@dataclass
class LoadedActionCheckpoint:
    path: Path
    metadata: dict[str, Any]
    training_state: dict[str, Any]
    actor: torch.nn.Module
    critic: torch.nn.Module
    checkpoint_sha256: str
    actor_sha256: str
    critic_sha256: str
    completed_steps: int
    horizon_mode: str
    horizons: tuple[int, ...]
    device: torch.device

    def validate_evaluation_horizon(
        self, horizon: int, *, preview_ablation: bool = False
    ) -> int:
        value = int(horizon)
        if value not in EVALUATION_HORIZONS:
            raise ValueError(f"evaluation horizon must be one of {EVALUATION_HORIZONS}")
        if self.horizon_mode == "fixed":
            native = self.horizons[0]
            if value != native and not (preview_ablation and value < native):
                raise RuntimeError(
                    f"fixed H{native} checkpoint cannot be evaluated natively as H{value}"
                )
        elif value not in self.horizons:
            raise RuntimeError(f"shared checkpoint does not support H{value}")
        return value

    def assert_unchanged(self) -> None:
        if parameter_hash(self.actor) != self.actor_sha256:
            raise RuntimeError("evaluation mutated actor parameters")
        if parameter_hash(self.critic) != self.critic_sha256:
            raise RuntimeError("evaluation mutated critic parameters")


@dataclass
class LoadedPartialCheckpoint:
    path: Path
    metadata: dict[str, Any]
    training_state: dict[str, Any]
    actor: torch.nn.Module
    critic: torch.nn.Module
    gaze_actor: torch.nn.Module | None
    gaze_critic: torch.nn.Module | None
    checkpoint_sha256: str
    actor_sha256: str
    critic_sha256: str
    gaze_sha256: str | None
    completed_steps: int
    constraint_config: HumanLikeConstraintConfig
    emma_timing: EmmaTimingConfig
    device: torch.device

    def assert_unchanged(self) -> None:
        if parameter_hash(self.actor) != self.actor_sha256:
            raise RuntimeError("Partial evaluation mutated ActionNet")
        if parameter_hash(self.critic) != self.critic_sha256:
            raise RuntimeError("Partial evaluation mutated task critic")
        if self.gaze_actor is not None and state_dict_hash(
            self.gaze_actor.state_dict()
        ) != self.gaze_sha256:
            raise RuntimeError("Partial evaluation mutated GazeNet")


def _build_critic(*, device: torch.device) -> torch.nn.Module:
    from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic

    args, _observation, share, _action = load_policy_config("rnn")
    return R_Critic(args, share, device=device)


def _checkpoint_contract(metadata: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    expected = {
        "format_id": FORMAT_ID,
        "training_stage": "action",
        "observation_mode": "full",
        "architecture_id": ACTION_ARCHITECTURE_ID,
        "observation_contract_id": FULL_OBSERVATION_CONTRACT,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "partner_set_id": PARTNER_SET_ID,
        "action_space": list(ACTION_SPACE),
        **cooking_contract(),
    }
    mismatch = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"unsupported evaluation checkpoint contract: {mismatch}")
    mode = metadata.get("horizon_mode")
    if mode is None and "horizon" in metadata:
        mode = "fixed"
    if mode == "fixed":
        horizon = metadata.get("horizon")
        if horizon not in EVALUATION_HORIZONS:
            raise RuntimeError("fixed checkpoint has invalid horizon metadata")
        return mode, (int(horizon),)
    if mode == "episode_balanced":
        horizons = tuple(int(value) for value in metadata.get("horizons", ()))
        if horizons != EVALUATION_HORIZONS:
            raise RuntimeError("shared checkpoint horizons must be [0, 2, 4]")
        return mode, horizons
    raise RuntimeError(f"unsupported horizon mode: {mode!r}")


def load_action_checkpoint_for_evaluation(
    path: Path, *, device: str | torch.device = "cpu"
) -> LoadedActionCheckpoint:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"checkpoint does not exist: {source}")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    try:
        payload = torch.load(source, map_location=target, weights_only=False)
    except Exception as error:
        raise RuntimeError(f"cannot load checkpoint {source}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise RuntimeError("checkpoint has no metadata mapping")
    metadata = dict(payload["metadata"])
    horizon_mode, horizons = _checkpoint_contract(metadata)
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise RuntimeError("Full Action evaluation requires complete training_state")
    if "completed_steps" not in training_state:
        raise RuntimeError("checkpoint training_state has no completed_steps")
    completed_steps = int(training_state["completed_steps"])
    if int(metadata.get("step", -1)) != completed_steps or completed_steps < 0:
        raise RuntimeError("checkpoint metadata/training_state step mismatch")
    if not isinstance(payload.get("actor"), dict) or not isinstance(payload.get("critic"), dict):
        raise RuntimeError("checkpoint has no actor/critic state")

    actor = build_actionnet(seed=0, device=target)
    critic = _build_critic(device=target)
    actor.load_state_dict(payload["actor"], strict=True)
    critic.load_state_dict(payload["critic"], strict=True)
    actor.eval().requires_grad_(False)
    critic.eval().requires_grad_(False)
    actor_hash = parameter_hash(actor)
    critic_hash = parameter_hash(critic)
    if metadata.get("actor_parameter_sha256") != actor_hash:
        raise RuntimeError("checkpoint actor content hash mismatch")
    if metadata.get("critic_parameter_sha256") != critic_hash:
        raise RuntimeError("checkpoint critic content hash mismatch")
    return LoadedActionCheckpoint(
        path=source,
        metadata=metadata,
        training_state=training_state,
        actor=actor,
        critic=critic,
        checkpoint_sha256=file_sha256(source),
        actor_sha256=actor_hash,
        critic_sha256=critic_hash,
        completed_steps=completed_steps,
        horizon_mode=horizon_mode,
        horizons=horizons,
        device=target,
    )


def load_partial_checkpoint_for_evaluation(
    path: Path, *, device: str | torch.device = "cpu"
) -> LoadedPartialCheckpoint:
    source = Path(path).expanduser().resolve()
    target = torch.device(device)
    if not source.is_file():
        raise RuntimeError(f"checkpoint does not exist: {source}")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = torch.load(source, map_location=target, weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    has_constraint_metadata = "constraint_contract_id" in metadata
    constraint_config = constraint_config_from_metadata(metadata)
    emma_timing = constraint_config.emma
    metadata.update(constraint_config.metadata())
    memory_decay_mode = str(metadata.get("memory_decay_mode", "linear"))
    metadata["memory_decay_mode"] = memory_decay_mode
    saved_lifetime = metadata.get("memory_lifetime_ms")
    if memory_decay_mode == "none":
        memory_lifetime_ms = None
        memory_contract_valid = saved_lifetime is None
        expected_freshness_contract = NO_DECAY_FRESHNESS_CONTRACT_ID
    else:
        try:
            memory_lifetime_ms = float(saved_lifetime)
        except (TypeError, ValueError):
            memory_lifetime_ms = -1.0
        if memory_decay_mode == "stochastic_exponential":
            saved_tau = metadata.get("memory_tau_ms")
            saved_max = metadata.get("memory_max_lifetime_ms")
            try:
                memory_tau_ms = float(saved_tau)
                memory_max_lifetime_ms = float(saved_max)
            except (TypeError, ValueError):
                memory_tau_ms = memory_max_lifetime_ms = -1.0
            memory_contract_valid = (
                math.isfinite(memory_lifetime_ms)
                and memory_lifetime_ms > 0.0
                and memory_max_lifetime_ms == memory_lifetime_ms
                and math.isfinite(memory_tau_ms)
                and memory_tau_ms > 0.0
            )
            expected_freshness_contract = (
                STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID
            )
        else:
            memory_tau_ms = None
            memory_contract_valid = (
                memory_decay_mode == "linear"
                and math.isfinite(memory_lifetime_ms)
                and memory_lifetime_ms > 0.0
            )
            expected_freshness_contract = CONTINUOUS_FRESHNESS_CONTRACT_ID
    expected_observation_contract = (
        observation_contract_id(
            memory_lifetime_ms, memory_decay_mode,
            metadata.get("memory_tau_ms"),
        )
        if memory_contract_valid
        else OBSERVATION_CONTRACT_ID
    )
    expected = {
        "format_id": FORMAT_ID,
        "partial_checkpoint_format_id": PARTIAL_CHECKPOINT_FORMAT_ID,
        "observation_mode": "partial",
        "architecture_id": ACTION_ARCHITECTURE_ID,
        "memory_decay_mode": memory_decay_mode,
        "freshness_contract_id": expected_freshness_contract,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "partner_set_id": PARTNER_SET_ID,
        "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
        "fixed_partner_identity": FIXED_HSP_PARTNER_ID,
        "fixed_partner_contract_id": FIXED_HSP_PARTNER_CONTRACT_ID,
        "action_space": list(ACTION_SPACE),
        **cooking_contract(),
    }
    mismatch = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if not memory_contract_valid:
        mismatch["memory_lifetime_ms"] = (
            metadata.get("memory_lifetime_ms"),
            (
                "null for no-decay memory"
                if memory_decay_mode == "none"
                else "finite positive milliseconds"
            ),
        )
    source_observation_contract = metadata.get("observation_contract_id")
    if source_observation_contract != expected_observation_contract:
        mismatch["observation_contract_id"] = (
            source_observation_contract, expected_observation_contract,
        )
    ground_truth_policy_input = metadata.get("ground_truth_policy_input", False)
    if not isinstance(ground_truth_policy_input, bool):
        mismatch["ground_truth_policy_input"] = (
            ground_truth_policy_input, "boolean"
        )
    elif ground_truth_policy_input and metadata.get(
        "policy_observation_contract_id"
    ) != FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID:
        mismatch["policy_observation_contract_id"] = (
            metadata.get("policy_observation_contract_id"),
            FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID,
        )
    if mismatch:
        raise RuntimeError(f"unsupported Partial evaluation checkpoint: {mismatch}")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise RuntimeError("Partial evaluation requires complete training_state")
    stored_contract = training_state.get("contract")
    if not isinstance(stored_contract, dict):
        raise RuntimeError("Partial evaluation requires a memory contract")
    memory_state_mismatch = {
        key: (stored_contract.get(key), metadata[key])
        for key in (
            "memory_decay_mode",
            "memory_lifetime_ms",
            "freshness_contract_id",
        )
        if (
            stored_contract.get(key, "linear")
            if key == "memory_decay_mode"
            else stored_contract.get(key)
        ) != metadata[key]
    }
    if memory_decay_mode == "stochastic_exponential":
        for key in ("memory_tau_ms", "memory_max_lifetime_ms"):
            if stored_contract.get(key) != metadata.get(key):
                memory_state_mismatch[key] = (
                    stored_contract.get(key), metadata.get(key)
                )
    if stored_contract.get("memory_contract_id") != metadata[
        "observation_contract_id"
    ]:
        memory_state_mismatch["memory_contract_id"] = (
            stored_contract.get("memory_contract_id"),
            metadata["observation_contract_id"],
        )
    if has_constraint_metadata:
        stored_constraint_config = constraint_config_from_metadata(
            stored_contract
        )
        if stored_constraint_config != constraint_config:
            raise RuntimeError(
                "Partial evaluation constraint contract mismatch: "
                f"checkpoint={stored_constraint_config.metadata()}, "
                f"metadata={constraint_config.metadata()}"
            )
    elif (
        any(key in stored_contract for key in emma_timing.metadata())
        or emma_timing.mode != "default"
    ):
        timing_state_mismatch = {
            key: (stored_contract.get(key), value)
            for key, value in emma_timing.metadata().items()
            if stored_contract.get(key) != value
        }
        if timing_state_mismatch:
            raise RuntimeError(
                "Partial evaluation EMMA timing contract mismatch: "
                f"{timing_state_mismatch}"
            )
    if memory_state_mismatch:
        raise RuntimeError(
            "Partial evaluation memory contract mismatch: "
            f"{memory_state_mismatch}"
        )
    completed_steps = int(training_state.get("completed_steps", -1))
    if int(metadata.get("step", -2)) != completed_steps or completed_steps < 0:
        raise RuntimeError("Partial checkpoint step mismatch")
    actor = build_actionnet(seed=0, device=target)
    critic = _build_critic(device=target)
    actor.load_state_dict(payload["actor"], strict=True)
    critic.load_state_dict(payload["critic"], strict=True)
    actor.eval().requires_grad_(False)
    critic.eval().requires_grad_(False)
    actor_hash, critic_hash = parameter_hash(actor), parameter_hash(critic)
    if actor_hash != metadata.get("actor_parameter_sha256"):
        raise RuntimeError("Partial ActionNet hash mismatch")
    if critic_hash != metadata.get("critic_parameter_sha256"):
        raise RuntimeError("Partial critic hash mismatch")
    gaze_actor = gaze_critic = None
    gaze_hash = None
    if "gaze_actor" in payload:
        if metadata.get("gaze_architecture_id") != GAZE_ARCHITECTURE_ID:
            raise RuntimeError("Partial GazeNet architecture mismatch")
        gaze_actor, gaze_critic = GazeNet().to(target), GazeCritic().to(target)
        gaze_actor.load_state_dict(payload["gaze_actor"], strict=True)
        gaze_critic.load_state_dict(payload["gaze_critic"], strict=True)
        gaze_actor.eval().requires_grad_(False)
        gaze_critic.eval().requires_grad_(False)
        gaze_hash = state_dict_hash(gaze_actor.state_dict())
        if gaze_hash != metadata.get("gaze_parameter_sha256"):
            raise RuntimeError("Partial GazeNet hash mismatch")
        metadata["evaluation_gaze_credit_assignment_status"] = (
            gaze_credit_evaluation_status(metadata)
        )
        metadata["evaluation_gaze_action_contract_status"] = (
            gaze_action_evaluation_status(metadata)
        )
        metadata["evaluation_gaze_input_contract_status"] = (
            gaze_input_evaluation_status(metadata)
        )
        metadata["legacy_gaze_action_contract_metadata"] = (
            None
            if metadata["evaluation_gaze_action_contract_status"] == "current"
            else metadata.get("gaze_action_contract_id")
        )
    return LoadedPartialCheckpoint(
        path=source, metadata=metadata, training_state=training_state,
        actor=actor, critic=critic, gaze_actor=gaze_actor, gaze_critic=gaze_critic,
        checkpoint_sha256=file_sha256(source), actor_sha256=actor_hash,
        critic_sha256=critic_hash, gaze_sha256=gaze_hash,
        completed_steps=completed_steps,
        constraint_config=constraint_config,
        emma_timing=emma_timing, device=target,
    )
