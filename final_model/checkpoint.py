"""Content- and contract-based portable checkpoint handling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from final_model.common.mappo import (
    ACTION_ARCHITECTURE_ID,
    FULL_OBSERVATION_CONTRACT,
    PARTIAL_OBSERVATION_CONTRACT,
)
from final_model.common.overcooked_backend import CANONICAL_ENCODER_ID, cooking_contract
from final_model.common.partners import PARTNER_SET_ID, parameter_hash
from final_model.common.paths import file_sha256
from final_model.partial_ver.network import GAZE_ARCHITECTURE_ID
from final_model.server_parallel.gaze_v1 import (
    GAZENET_V1_ACTION_CONTRACT_ID,
    GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID,
    GAZENET_V1_INPUT_CONTRACT_ID,
    GAZENET_V1_REWARD_CONTRACT_ID,
)


FORMAT_ID = "final_model_portable_checkpoint_v1"
ACTION_SPACE = ("NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT")


def state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _action_metadata(
    *, mode: str, horizon: int | None, step: int, seed: int,
    horizon_mode: str = "fixed", horizons=None,
) -> dict[str, Any]:
    metadata = {
        "format_id": FORMAT_ID,
        "training_stage": "action",
        "observation_mode": mode,
        "horizon_mode": horizon_mode,
        "step": int(step),
        "seed": int(seed),
        "architecture_id": ACTION_ARCHITECTURE_ID,
        "observation_contract_id": (
            FULL_OBSERVATION_CONTRACT if mode == "full" else PARTIAL_OBSERVATION_CONTRACT
        ),
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "partner_set_id": PARTNER_SET_ID,
        "action_space": list(ACTION_SPACE),
    }
    if horizon_mode == "fixed":
        if horizon is None:
            raise ValueError("fixed checkpoint metadata requires horizon")
        metadata["horizon"] = int(horizon)
    elif horizon_mode == "episode_balanced":
        values = [int(value) for value in horizons or ()]
        if values != [0, 2, 4]:
            raise ValueError("episode-balanced checkpoint requires horizons [0, 2, 4]")
        metadata["horizons"] = values
    else:
        raise ValueError(f"unsupported horizon_mode: {horizon_mode!r}")
    if mode == "full":
        metadata.update(
            {
                "input_channels": 25,
                "channels_0_19": "current_canonical_state",
                "channel_20": "full_dynamic_freshness",
                "channel_20_value": 255.0,
                "channels_21_24": "displayed_ai_action_preview_h1_h4",
            }
        )
    return metadata


def save_action_checkpoint(
    path, *, actor, critic, mode, horizon, step, seed, condition=None, training_state=None,
    horizon_mode="fixed", horizons=None,
):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _action_metadata(
        mode=mode, horizon=horizon, step=step, seed=seed,
        horizon_mode=horizon_mode, horizons=horizons,
    )
    metadata.update(condition or {})
    metadata["actor_parameter_sha256"] = parameter_hash(actor)
    metadata["critic_parameter_sha256"] = parameter_hash(critic)
    payload = {"metadata": metadata, "actor": actor.state_dict(), "critic": critic.state_dict()}
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, output)
    return {**metadata, "checkpoint_sha256": file_sha256(output)}


def load_action_checkpoint(
    path, *, actor, critic=None, expected_mode=None, expected_horizon=None,
    expected_horizon_mode=None, expected_horizons=None, expected_condition=None,
):
    source = Path(path)
    payload = torch.load(source, map_location=next(actor.parameters()).device, weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    expected = {
        "format_id": FORMAT_ID,
        "training_stage": "action",
        "architecture_id": ACTION_ARCHITECTURE_ID,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "partner_set_id": PARTNER_SET_ID,
        "action_space": list(ACTION_SPACE),
    }
    if expected_mode is not None:
        expected["observation_mode"] = expected_mode
        expected["observation_contract_id"] = (
            FULL_OBSERVATION_CONTRACT if expected_mode == "full" else PARTIAL_OBSERVATION_CONTRACT
        )
    if expected_horizon is not None:
        expected["horizon"] = int(expected_horizon)
    if expected_horizon_mode is not None:
        expected["horizon_mode"] = expected_horizon_mode
    if expected_horizons is not None:
        expected["horizons"] = [int(value) for value in expected_horizons]
    expected.update(expected_condition or {})
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise RuntimeError(f"checkpoint contract mismatch: {mismatch}")
    actor.load_state_dict(payload["actor"], strict=True)
    if critic is not None:
        critic.load_state_dict(payload["critic"], strict=True)
    if metadata.get("actor_parameter_sha256") != parameter_hash(actor):
        raise RuntimeError("checkpoint actor content hash mismatch")
    if critic is not None and metadata.get("critic_parameter_sha256") != parameter_hash(critic):
        raise RuntimeError("checkpoint critic content hash mismatch")
    return {**metadata, "checkpoint_sha256": file_sha256(source)}


def initialize_partial_from_full(path, *, actor, horizon: int | None = None, horizons=None):
    provenance = load_action_checkpoint(
        path,
        actor=actor,
        expected_mode="full",
        expected_condition=cooking_contract(),
    )
    mode = provenance.get("horizon_mode", "fixed")
    requested = {int(value) for value in (horizons or ())}
    if horizon is not None:
        requested.add(int(horizon))
    if not requested:
        raise ValueError("Partial initialization requires at least one target horizon")
    supported = (
        {int(provenance["horizon"])}
        if mode == "fixed"
        else {int(value) for value in provenance.get("horizons", ())}
    )
    if mode not in {"fixed", "episode_balanced"} or not requested.issubset(supported):
        raise RuntimeError(
            f"Full checkpoint horizons {sorted(supported)} do not cover "
            f"Partial targets {sorted(requested)}"
        )
    return {
        "initialization": (
            "matching_full_checkpoint" if mode == "fixed" else "shared_full_checkpoint"
        ),
        "source_absolute_path": str(Path(path).expanduser().resolve()),
        "source_content_sha256": provenance["checkpoint_sha256"],
        "source_actor_parameter_sha256": provenance["actor_parameter_sha256"],
        "source_horizon_mode": mode,
        "source_horizon": provenance.get("horizon"),
        "source_shared_horizons": provenance.get("horizons"),
        "source_contract": provenance["observation_contract_id"],
        "source_dynamics_contract": cooking_contract(),
        "compatibility_used_absolute_path": False,
    }


def save_gaze_checkpoint(
    path, *, action_actor, gaze_actor, gaze_critic, horizon, decision_time, fixation_shape, step, seed,
    source_provenance=None, training_state=None, condition=None,
):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_id": FORMAT_ID,
        "training_stage": "gaze",
        "observation_mode": "partial",
        "horizon": int(horizon),
        "decision_time": int(decision_time),
        "fixation_shape": fixation_shape,
        "step": int(step),
        "seed": int(seed),
        "action_architecture_id": ACTION_ARCHITECTURE_ID,
        "gaze_architecture_id": GAZE_ARCHITECTURE_ID,
        "observation_contract_id": PARTIAL_OBSERVATION_CONTRACT,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "partner_set_id": PARTNER_SET_ID,
        "action_space": list(ACTION_SPACE),
        "actionnet_frozen": True,
        "action_optimizer_steps": 0,
        "action_parameter_sha256": parameter_hash(action_actor),
        "gaze_parameter_sha256": state_dict_hash(gaze_actor.state_dict()),
        "gaze_critic_parameter_sha256": state_dict_hash(gaze_critic.state_dict()),
        "gaze_action_contract_id": GAZENET_V1_ACTION_CONTRACT_ID,
        "gaze_input_contract_id": GAZENET_V1_INPUT_CONTRACT_ID,
        "gaze_reward_contract_id": GAZENET_V1_REWARD_CONTRACT_ID,
        "gaze_credit_assignment_contract_id": (
            GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID
        ),
        "source_partial_action_provenance": dict(source_provenance or {}),
    }
    metadata.update(condition or {})
    payload = {
        "metadata": metadata,
        "action_actor": action_actor.state_dict(),
        "gaze_actor": gaze_actor.state_dict(),
        "gaze_critic": gaze_critic.state_dict(),
    }
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, output)
    return {**metadata, "checkpoint_sha256": file_sha256(output)}


def load_gaze_checkpoint(
    path, *, action_actor, gaze_actor, gaze_critic=None, horizon=None,
    decision_time=None, fixation_shape=None, expected_condition=None,
):
    source = Path(path)
    payload = torch.load(source, map_location=next(action_actor.parameters()).device, weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    expected = {
        "format_id": FORMAT_ID,
        "training_stage": "gaze",
        "observation_mode": "partial",
        "action_architecture_id": ACTION_ARCHITECTURE_ID,
        "gaze_architecture_id": GAZE_ARCHITECTURE_ID,
        "observation_contract_id": PARTIAL_OBSERVATION_CONTRACT,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "partner_set_id": PARTNER_SET_ID,
        "actionnet_frozen": True,
        "action_optimizer_steps": 0,
    }
    for key, value in (("horizon", horizon), ("decision_time", decision_time), ("fixation_shape", fixation_shape)):
        if value is not None:
            expected[key] = value
    expected.update(expected_condition or {})
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise RuntimeError(f"gaze checkpoint contract mismatch: {mismatch}")
    action_actor.load_state_dict(payload["action_actor"], strict=True)
    gaze_actor.load_state_dict(payload["gaze_actor"], strict=True)
    if gaze_critic is not None:
        gaze_critic.load_state_dict(payload["gaze_critic"], strict=True)
    if parameter_hash(action_actor) != metadata.get("action_parameter_sha256"):
        raise RuntimeError("gaze checkpoint frozen ActionNet hash mismatch")
    if state_dict_hash(gaze_actor.state_dict()) != metadata.get("gaze_parameter_sha256"):
        raise RuntimeError("gaze checkpoint GazeNet hash mismatch")
    if (
        gaze_critic is not None
        and state_dict_hash(gaze_critic.state_dict())
        != metadata.get("gaze_critic_parameter_sha256")
    ):
        raise RuntimeError("gaze checkpoint critic content hash mismatch")
    credit_status = (
        "current"
        if metadata.get("gaze_credit_assignment_contract_id")
        == GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID
        else "legacy_or_unknown"
    )
    action_status = (
        "current"
        if metadata.get("gaze_action_contract_id")
        == GAZENET_V1_ACTION_CONTRACT_ID
        else "legacy_or_unknown_metadata_runtime_x_major"
    )
    legacy_action_metadata = (
        None
        if action_status == "current"
        else metadata.get("gaze_action_contract_id")
    )
    input_status = (
        "current"
        if metadata.get("gaze_input_contract_id")
        == GAZENET_V1_INPUT_CONTRACT_ID
        else "legacy_or_unknown_input_scaling"
    )
    return {
        **metadata,
        "evaluation_gaze_action_contract_status": action_status,
        "legacy_gaze_action_contract_metadata": legacy_action_metadata,
        "evaluation_gaze_input_contract_status": input_status,
        "evaluation_gaze_credit_assignment_status": credit_status,
        "checkpoint_sha256": file_sha256(source),
    }


def checkpoint_contract_report() -> dict[str, object]:
    return {
        "format_id": FORMAT_ID,
        "full_to_partial": {"H0": "H0", "H2": "H2", "H4": "H4"},
        "partial_action_initialization": [
            "fresh_partial", "matching_full_checkpoint", "shared_full_checkpoint"
        ],
        "partial_gaze_source": "canonical Partial v2 checkpoint",
        "strict_state_dict": True,
        "training_resume": {
            "supported_stages": ["action", "gaze", "joint"],
            "boundary": "completed PPO update",
            "legacy_checkpoint_behavior": "fail closed when training_state is absent",
            "schedule_state": (
                "global step, entropy coefficient, and reward-shaping factor are saved, "
                "validated, and restored"
            ),
        },
        "absolute_path_compatibility": False,
        "partial_v2": {
            "gaze_types": ["rule_based", "learned"],
            "train_components": [["action"], ["gaze"], ["action", "gaze"]],
            "default_memory_lifetime_ms": 2400,
            "memory_lifetime_configurable": True,
            "freshness": "continuous linear semantic [0,1], raw float32 [0,255]",
            "condition_scheduler": "generic Cartesian shuffled episode blocks",
            "action_query": "completed visual acquisitions only",
            "recurrent_commit": "latest proposal candidate only",
        },
        "compatibility_fields": [
            "format_id", "training_stage", "observation_mode", "horizon_mode",
            "horizon/horizons", "architecture_id",
            "observation_contract_id", "canonical_encoder_id", "partner_set_id", "action_space",
            "gaze_action_contract_id",
            "gaze_input_contract_id",
            "content hashes", "configured_total_steps", "checkpoint_interval",
            "entropy_schedule", "entropy_schedule_horizons",
            "reward_shaping_schedule", "reward_shaping_horizon_steps",
            "cooking_dynamics_id", "onion_cook_ticks_per_environment_step",
            "soup_cooking_time", "soup_items_required",
        ],
    }
