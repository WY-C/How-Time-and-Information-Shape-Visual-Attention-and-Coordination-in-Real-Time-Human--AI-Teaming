"""Persistent-worker execution backend around canonical Partial Action PPO."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Any

import numpy as np
import torch

from final_model.checkpoint import ACTION_SPACE, FORMAT_ID, state_dict_hash
from final_model.common.mappo import (
    ACTION_ARCHITECTURE_ID,
    FULL_OBSERVATION_CONTRACT,
    entropy_coefficient,
    explained_variance,
    install_rollout_context_stable_actor,
    parameter_hash,
)
from final_model.common.partners import (
    FIXED_HSP_PARTNER_CONTRACT_ID,
    FIXED_HSP_PARTNER_ID,
    FIXED_HSP_PARTNER_SHA256,
    PARTNER_SET_ID,
)
from final_model.common.paths import file_sha256
from final_model.common.resume import (
    TRAINING_STATE_FORMAT_ID,
    capture_global_rng_state,
)
from final_model.common.reward import hsp_reward_shaping_factor
from final_model.common.overcooked_backend import (
    CANONICAL_ENCODER_ID,
    cooking_contract,
)
from final_model.partial_ver.canonical_training import (
    PARTIAL_CHECKPOINT_FORMAT_ID,
    PartialTrainingSession,
    STAY_ACTION_INDEX,
)
from final_model.partial_ver.constraints import (
    HumanLikeConstraintConfig,
    constraint_config_from_metadata,
)
from final_model.partial_ver.network import GAZE_ARCHITECTURE_ID
from final_model.partial_ver.training import (
    gaze_ppo_update,
    require_current_gaze_action_contract,
    require_current_gaze_credit_contract,
    require_current_gaze_input_contract,
)
from final_model.parallel_gaze_contract import (
    PARALLEL_DIRECT_JOINT_TRAINING_CONTRACT,
    PARALLEL_GAZE_CPU_INFERENCE_DTYPE,
    PARALLEL_GAZE_CPU_INFERENCE_PATH,
    PARALLEL_GAZE_CPU_INTER_OP_THREADS,
    PARALLEL_GAZE_CPU_INTRA_OP_THREADS,
    PARALLEL_GAZE_HOST_MODE,
    PARALLEL_GAZE_LEGACY_LOGPROB_AUDIT_VERSION,
    PARALLEL_GAZE_LOGPROB_AUDIT_VERSION,
    PARALLEL_GAZE_RNG_CONTRACT,
    PARALLEL_GAZE_TRAINING_CONTRACT,
    PARALLEL_STAGED_JOINT_TRAINING_CONTRACT,
    merge_gaze_transitions,
)
from final_model.eval.benchmark_provenance import git_provenance
from final_model.partial_ver.conditions import PartialCondition
from final_model.partial_ver.observability import (
    accumulate_partial_metrics,
    finalize_partial_metrics,
    new_partial_metric_totals,
    validate_partial_metric_totals,
)
from final_model.partial_ver.observation import (
    FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID,
)
from final_model.partial_ver.memory import (
    CONTINUOUS_FRESHNESS_CONTRACT_ID,
    MEMORY_LIFETIME_MS,
    NO_DECAY_FRESHNESS_CONTRACT_ID,
    STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID,
    observation_contract_id,
)
from final_model.partial_ver.gaze import VISUAL_ENCODING_EMMA_MODES
from final_model.server_parallel import SERVER_PARALLEL_VERSION
from final_model.server_parallel.collision_observability import (
    accumulate_collision_totals,
    finalize_collision_rates,
    initialize_collision_totals,
    validate_collision_conservation,
)
from final_model.server_parallel.protocol import (
    FIXATION_TO_CODE,
    MAX_ACQUISITIONS_PER_DECISION,
    deterministic_assignments,
)
from final_model.server_parallel.rollout_worker import rollout_worker_main
from final_model.server_parallel.shared_buffers import SharedDecisionBuffers


SOURCE_VALIDATION_CONTRACT_ID = "server_partial_full_source_contract_v1"
PARTIAL_SOURCE_VALIDATION_CONTRACT_ID = (
    "server_partial_action_source_contract_v2"
)
PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID = (
    "server_partial_gaze_source_contract_v1"
)
RESUME_PROVENANCE_CONTRACT_ID = "server_partial_resume_provenance_v1"
WORKER_RESPONSE_TIMEOUT_SECONDS = 300.0
_MATCHED_CPU_CONTRACT_KEYS = (
    "main_torch_intra_op_threads",
    "main_torch_inter_op_threads",
    "gaze_cpu_inference_dtype",
    "gaze_cpu_inference_path",
    "gaze_logprob_audit_version",
)


def configure_parallel_gaze_main_cpu_execution() -> dict[str, int | str]:
    """Apply and verify the main-process half of the matched CPU audit path."""
    expected_intra = PARALLEL_GAZE_CPU_INTRA_OP_THREADS
    expected_inter = PARALLEL_GAZE_CPU_INTER_OP_THREADS
    if torch.get_num_threads() != expected_intra:
        torch.set_num_threads(expected_intra)
    if torch.get_num_interop_threads() != expected_inter:
        try:
            torch.set_num_interop_threads(expected_inter)
        except RuntimeError as error:
            raise RuntimeError(
                "parallel Gaze requires main Torch CPU intra/inter-op=1/1; "
                "start it in a fresh Python process before parallel CPU work"
            ) from error
    actual_intra = int(torch.get_num_threads())
    actual_inter = int(torch.get_num_interop_threads())
    if (actual_intra, actual_inter) != (expected_intra, expected_inter):
        raise RuntimeError(
            "parallel Gaze main CPU thread contract mismatch: "
            f"actual={actual_intra}/{actual_inter} "
            f"expected={expected_intra}/{expected_inter}"
        )
    return {
        "torch_cpu_intra_op_threads": actual_intra,
        "torch_cpu_inter_op_threads": actual_inter,
        "gaze_cpu_inference_dtype": PARALLEL_GAZE_CPU_INFERENCE_DTYPE,
        "gaze_cpu_inference_path": PARALLEL_GAZE_CPU_INFERENCE_PATH,
    }


def _require_parallel_gaze_main_cpu_execution() -> dict[str, int | str]:
    actual = {
        "torch_cpu_intra_op_threads": int(torch.get_num_threads()),
        "torch_cpu_inter_op_threads": int(torch.get_num_interop_threads()),
        "gaze_cpu_inference_dtype": PARALLEL_GAZE_CPU_INFERENCE_DTYPE,
        "gaze_cpu_inference_path": PARALLEL_GAZE_CPU_INFERENCE_PATH,
    }
    expected = (
        PARALLEL_GAZE_CPU_INTRA_OP_THREADS,
        PARALLEL_GAZE_CPU_INTER_OP_THREADS,
    )
    if (
        actual["torch_cpu_intra_op_threads"],
        actual["torch_cpu_inter_op_threads"],
    ) != expected:
        raise RuntimeError(
            "parallel Gaze main CPU thread settings drifted before audit: "
            f"actual={actual['torch_cpu_intra_op_threads']}/"
            f"{actual['torch_cpu_inter_op_threads']} expected=1/1"
        )
    return actual


def validate_partial_resume_continuation(
    stored_contract: object,
    *,
    stored_environment_steps: int,
    requested_total_steps: int,
    requested_checkpoint_interval: int,
    update_size: int,
) -> dict[str, int]:
    """Validate the only two operational fields allowed to change on resume."""
    if not isinstance(stored_contract, dict):
        raise RuntimeError("Partial resume checkpoint has no training contract")
    stored_total = stored_contract.get("configured_total_steps")
    stored_interval = stored_contract.get("checkpoint_interval")
    if (
        not isinstance(stored_total, int)
        or stored_total < stored_environment_steps
        or not isinstance(stored_interval, int)
        or stored_interval <= 0
    ):
        raise RuntimeError(
            "Partial resume checkpoint has invalid operational contract"
        )
    target = int(requested_total_steps)
    interval = int(requested_checkpoint_interval)
    batch = int(update_size)
    if batch <= 0:
        raise ValueError("PPO update size must be positive")
    if target < stored_environment_steps:
        raise ValueError("--steps cannot be below completed resume steps")
    if (target - stored_environment_steps) % batch:
        raise ValueError("remaining --steps must be a whole PPO update")
    if interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    if interval % batch:
        raise ValueError(
            "--checkpoint-interval must align with complete PPO updates"
        )
    return {
        "stored_environment_steps": int(stored_environment_steps),
        "stored_configured_total_steps": int(stored_total),
        "stored_checkpoint_interval": int(stored_interval),
        "requested_total_steps": target,
        "requested_checkpoint_interval": interval,
        "additional_environment_steps": target - int(stored_environment_steps),
        "additional_updates": (target - int(stored_environment_steps)) // batch,
        "update_size": batch,
    }


def continuation_checkpoint_schedule(
    *,
    start_step: int,
    target_step: int,
    checkpoint_interval: int,
    update_size: int,
) -> tuple[int, ...]:
    """Return checkpoints produced by the actual post-update save predicate."""
    start = int(start_step)
    target = int(target_step)
    interval = int(checkpoint_interval)
    batch = int(update_size)
    if target < start:
        raise ValueError("target step cannot be below resume step")
    if batch <= 0 or (target - start) % batch:
        raise ValueError("continuation target must align with complete PPO updates")
    if interval <= 0 or interval % batch:
        raise ValueError("checkpoint interval must align with complete PPO updates")
    return tuple(
        step
        for step in range(start + batch, target + 1, batch)
        if step % interval == 0 or step == target
    )


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _gaze_record_trace(records) -> list[dict[str, object]]:
    return [{
        "env_index": int(record["env_index"]),
        "episode_id": int(record["episode_id"]),
        "record_sequence_id": int(record["record_sequence_id"]),
        "task_decision_id": int(record["task_decision_id"]),
        "policy_version": int(record.get("policy_version", 0)),
        "action": int(record["action"]),
        "log_prob": float(record["log_prob"]),
        "value": float(record["value"]),
        "next_value": float(record["next_value"]),
        "reward": float(record["reward"]),
        "done": bool(record["done"]),
        "env_step_boundary_count": int(record["env_step_boundary_count"]),
        "observation_sha256": _array_hash(record["observation"]),
        "context_sha256": _array_hash(record["context"]),
    } for record in records]


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_batch1_rollout_actor_outputs(
    actor,
    observations: np.ndarray,
    rows: list[tuple[int, int]],
    pre_actor: np.ndarray,
    recurrent_masks: np.ndarray,
    available_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run canonical batch-1 forwards, then transfer their outputs by field."""
    action_tensors = []
    log_probability_tensors = []
    candidate_tensors = []
    with torch.no_grad():
        for env_id, acquisition in rows:
            action, log_probability, candidate = actor(
                observations[env_id, acquisition][None],
                pre_actor[env_id][None],
                recurrent_masks[env_id][None],
                available_actions,
                deterministic=False,
            )
            action_tensors.append(action)
            log_probability_tensors.append(log_probability)
            candidate_tensors.append(candidate)

        actions = torch.cat(action_tensors, dim=0)
        log_probabilities = torch.cat(log_probability_tensors, dim=0)
        candidates = torch.cat(candidate_tensors, dim=0)

    return (
        actions.detach().cpu().numpy().reshape(-1).copy(),
        log_probabilities.detach().cpu().numpy().reshape(-1).copy(),
        candidates.detach().cpu().numpy().copy(),
    )


def _new_horizon_metrics(horizons) -> dict[str, dict[str, float | int]]:
    return {
        f"h{int(horizon)}": {
            "optimization_reward": 0.0,
            "sparse_team_reward": 0.0,
            "deliveries": 0,
            "task_decisions": 0,
            "episodes": 0,
        }
        for horizon in horizons
    }


def _accumulate_horizon_metrics(
    totals: dict[str, dict[str, float | int]],
    horizons: np.ndarray,
    rewards: np.ndarray,
    deliveries: np.ndarray,
    dones: np.ndarray,
) -> None:
    """Read already-collected transition results into per-horizon scalars."""
    if not (
        len(horizons) == rewards.shape[0] == len(deliveries) == len(dones)
        and rewards.ndim == 2
        and rewards.shape[1] >= 2
    ):
        raise RuntimeError("invalid per-horizon observability inputs")
    for label, bucket in totals.items():
        horizon = int(label[1:])
        selected = horizons == horizon
        bucket["optimization_reward"] += float(rewards[selected, 0].sum())
        bucket["sparse_team_reward"] += float(rewards[selected, 1].sum())
        bucket["deliveries"] += int(deliveries[selected].sum())
        bucket["task_decisions"] += int(selected.sum())
        bucket["episodes"] += int(dones[selected].sum())


def _finalize_horizon_metrics(
    totals: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int | None]]:
    result = copy.deepcopy(totals)
    for bucket in result.values():
        episodes = int(bucket["episodes"])
        bucket.update({
            "reward": float(bucket["optimization_reward"]),
            "sparse_reward": float(bucket["sparse_team_reward"]),
            "decisions": int(bucket["task_decisions"]),
            "episode_count": episodes,
            "deliveries_per_episode": (
                None
                if episodes == 0
                else float(bucket["deliveries"]) / float(episodes)
            ),
        })
    return result


def _validate_horizon_metric_totals(
    metrics: dict[str, dict[str, float | int | None]],
    *,
    optimization_reward: float,
    sparse_reward: float,
    deliveries: int,
    episodes: int,
    decisions: int,
) -> None:
    if not np.isclose(
        sum(float(row["optimization_reward"]) for row in metrics.values()),
        optimization_reward,
        rtol=1e-10,
        atol=1e-6,
    ):
        raise RuntimeError("per-horizon optimization reward does not reconcile")
    if not np.isclose(
        sum(float(row["sparse_team_reward"]) for row in metrics.values()),
        sparse_reward,
        rtol=1e-10,
        atol=1e-6,
    ):
        raise RuntimeError("per-horizon sparse reward does not reconcile")
    exact = {
        "deliveries": (sum(int(row["deliveries"]) for row in metrics.values()), deliveries),
        "episodes": (sum(int(row["episodes"]) for row in metrics.values()), episodes),
        "decisions": (
            sum(int(row["task_decisions"]) for row in metrics.values()), decisions
        ),
    }
    mismatch = {key: values for key, values in exact.items() if values[0] != values[1]}
    if mismatch:
        raise RuntimeError(f"per-horizon observability totals do not reconcile: {mismatch}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_partial_action_source_checkpoint(
    path: Path,
    *,
    requested_horizons,
    target_memory_lifetime_ms: float | None = None,
    target_memory_decay_mode: str = "linear",
    target_memory_tau_ms: float | None = None,
) -> dict[str, object]:
    """Validate a canonical Full or Partial Action source before workers start."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"Partial source checkpoint does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("metadata"), dict
    ):
        raise RuntimeError("Partial source checkpoint has no metadata contract")
    metadata = dict(payload["metadata"])
    if metadata.get("observation_mode") == "partial":
        target_mode = str(target_memory_decay_mode)
        target_lifetime = (
            None
            if target_mode == "none"
            else MEMORY_LIFETIME_MS
            if target_memory_lifetime_ms is None
            else float(target_memory_lifetime_ms)
        )
        target_freshness_contract = (
            NO_DECAY_FRESHNESS_CONTRACT_ID
            if target_mode == "none"
            else STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID
            if target_mode == "stochastic_exponential"
            else CONTINUOUS_FRESHNESS_CONTRACT_ID
        )
        expected = {
            "format_id": FORMAT_ID,
            "partial_checkpoint_format_id": PARTIAL_CHECKPOINT_FORMAT_ID,
            "training_stage": "action",
            "observation_mode": "partial",
            "gaze_type": "rule_based",
            "train_action": True,
            "train_gaze": False,
            "inference_batching": "legacy",
            "architecture_id": ACTION_ARCHITECTURE_ID,
            "observation_contract_id": observation_contract_id(
                target_lifetime, target_mode, target_memory_tau_ms
            ),
            "memory_decay_mode": target_mode,
            "memory_lifetime_ms": target_lifetime,
            "freshness_contract_id": target_freshness_contract,
            "canonical_encoder_id": CANONICAL_ENCODER_ID,
            "partner_set_id": PARTNER_SET_ID,
            "action_space": list(ACTION_SPACE),
            "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
            "fixed_partner_identity": FIXED_HSP_PARTNER_ID,
            "fixed_partner_contract_id": FIXED_HSP_PARTNER_CONTRACT_ID,
            **cooking_contract(),
        }
        if target_mode == "stochastic_exponential":
            expected.update({
                "memory_tau_ms": target_memory_tau_ms,
                "memory_max_lifetime_ms": target_lifetime,
            })
        mismatch = {
            key: (
                metadata.get(key, "linear")
                if key == "memory_decay_mode"
                else metadata.get(key),
                value,
            )
            for key, value in expected.items()
            if (
                metadata.get(key, "linear")
                if key == "memory_decay_mode"
                else metadata.get(key)
            ) != value
        }
        if mismatch:
            raise RuntimeError(
                f"Partial source checkpoint contract mismatch: {mismatch}"
            )

        if "horizons" in metadata:
            supported_horizons = [int(value) for value in metadata["horizons"]]
            horizon_mode = str(
                metadata.get(
                    "horizon_mode", metadata.get("condition_mode", "fixed")
                )
            )
        elif metadata.get("horizon") is not None:
            supported_horizons = [int(metadata["horizon"])]
            horizon_mode = str(metadata.get("horizon_mode", "fixed"))
        else:
            raise RuntimeError(
                "Partial source checkpoint has invalid horizon contract"
            )
        requested = {int(value) for value in requested_horizons}
        if not requested or not requested.issubset(set(supported_horizons)):
            raise RuntimeError(
                "Partial source horizons do not cover requested Partial horizons"
            )

        actor_state = payload.get("actor")
        if not isinstance(actor_state, dict) or not actor_state:
            raise RuntimeError("Partial source checkpoint has no Action actor")
        actor_sha256 = state_dict_hash(actor_state)
        if not _is_sha256(metadata.get("actor_parameter_sha256")):
            raise RuntimeError("Partial source actor parameter hash is not recorded")
        if metadata["actor_parameter_sha256"] != actor_sha256:
            raise RuntimeError("Partial source actor parameter hash mismatch")
        critic_state = payload.get("critic")
        if not isinstance(critic_state, dict) or not critic_state:
            raise RuntimeError("Partial source checkpoint has no Action critic")
        critic_sha256 = state_dict_hash(critic_state)
        if not _is_sha256(metadata.get("critic_parameter_sha256")):
            raise RuntimeError("Partial source critic parameter hash is not recorded")
        if metadata["critic_parameter_sha256"] != critic_sha256:
            raise RuntimeError("Partial source critic parameter hash mismatch")

        state = payload.get("training_state")
        if not isinstance(state, dict):
            raise RuntimeError(
                "Partial source is not a complete production checkpoint"
            )
        state_contract = state.get("contract")
        required_state_fields = {
            "action_actor_optimizer",
            "action_critic_optimizer",
            "value_normalizer",
            "backends",
            "adapters",
            "conditions",
            "condition_decision_counts",
            "scheduler",
            "buffer",
            "rng",
        }
        step = metadata.get("step")
        if (
            not isinstance(step, int)
            or step <= 0
            or state.get("format_id") != TRAINING_STATE_FORMAT_ID
            or state.get("completed_steps") != step
            or not isinstance(state_contract, dict)
            or not required_state_fields.issubset(state)
            or not isinstance(state_contract.get("configured_total_steps"), int)
            or state_contract["configured_total_steps"] < step
            or not isinstance(state_contract.get("checkpoint_interval"), int)
            or state_contract["checkpoint_interval"] <= 0
            or metadata.get("checkpoint_role") == "pre_training_initialization"
            or metadata.get("smoke") is True
            or metadata.get("long_training_started") is False
        ):
            raise RuntimeError(
                "Partial source is not a valid production checkpoint"
            )
        redundant_expected = {
            "partial_checkpoint_format_id": PARTIAL_CHECKPOINT_FORMAT_ID,
            "observation_mode": "partial",
            "gaze_type": "rule_based",
            "train_action": True,
            "train_gaze": False,
            "inference_batching": "legacy",
            "memory_contract_id": observation_contract_id(
                target_lifetime, target_mode, target_memory_tau_ms
            ),
            "memory_decay_mode": target_mode,
            "memory_lifetime_ms": target_lifetime,
            "freshness_contract_id": target_freshness_contract,
            "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
            "fixed_partner_identity": FIXED_HSP_PARTNER_ID,
            "fixed_partner_contract_id": FIXED_HSP_PARTNER_CONTRACT_ID,
            **cooking_contract(),
        }
        if target_mode == "stochastic_exponential":
            redundant_expected.update({
                "memory_tau_ms": target_memory_tau_ms,
                "memory_max_lifetime_ms": target_lifetime,
            })
        redundant_mismatch = {
            key: (
                state_contract.get(key, "linear")
                if key == "memory_decay_mode"
                else state_contract.get(key),
                value,
            )
            for key, value in redundant_expected.items()
            if (
                state_contract.get(key, "linear")
                if key == "memory_decay_mode"
                else state_contract.get(key)
            ) != value
        }
        state_horizons = state_contract.get("horizons")
        if not isinstance(state_horizons, list) or {
            int(value) for value in state_horizons
        } != set(supported_horizons):
            redundant_mismatch["horizons"] = (
                state_horizons,
                supported_horizons,
            )
        if redundant_mismatch:
            raise RuntimeError(
                "Partial source training-state contract mismatch: "
                f"{redundant_mismatch}"
            )

        checkpoint_sha256 = file_sha256(source)
        if not _is_sha256(checkpoint_sha256):
            raise RuntimeError(
                "Partial source checkpoint content hash is not recorded"
            )
        return {
            "source_validation_contract_id": (
                PARTIAL_SOURCE_VALIDATION_CONTRACT_ID
            ),
            "source_checkpoint_validated": True,
            "source_production_checkpoint": True,
            "initialization": "partial_checkpoint",
            "source_kind": "canonical_partial_action",
            "source_absolute_path": str(source),
            "source_content_sha256": checkpoint_sha256,
            "source_actor_parameter_sha256": actor_sha256,
            "source_critic_parameter_sha256": critic_sha256,
            "source_checkpoint_step": step,
            "source_observation_mode": "partial",
            "source_observation_contract_id": metadata[
                "observation_contract_id"
            ],
            "source_architecture_id": ACTION_ARCHITECTURE_ID,
            "source_partial_checkpoint_format_id": (
                PARTIAL_CHECKPOINT_FORMAT_ID
            ),
            "source_horizon_mode": horizon_mode,
            "source_horizons": supported_horizons,
            "source_shared_horizons": supported_horizons,
            "source_horizon": (
                supported_horizons[0] if len(supported_horizons) == 1 else None
            ),
            "source_memory_lifetime_ms": target_lifetime,
            "source_memory_decay_mode": target_mode,
            "source_freshness_contract_id": target_freshness_contract,
            **(
                {
                    "source_memory_tau_ms": target_memory_tau_ms,
                    "source_memory_max_lifetime_ms": target_lifetime,
                }
                if target_mode == "stochastic_exponential"
                else {}
            ),
            "source_fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
            "source_fixed_partner_identity": FIXED_HSP_PARTNER_ID,
            "source_fixed_partner_contract_id": (
                FIXED_HSP_PARTNER_CONTRACT_ID
            ),
            "source_dynamics_contract": cooking_contract(),
        }

    expected = {
        "format_id": FORMAT_ID,
        "training_stage": "action",
        "observation_mode": "full",
        "architecture_id": ACTION_ARCHITECTURE_ID,
        "target_architecture_id": ACTION_ARCHITECTURE_ID,
        "observation_contract_id": FULL_OBSERVATION_CONTRACT,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "partner_set_id": PARTNER_SET_ID,
        "action_space": list(ACTION_SPACE),
        "input_channels": 25,
        "target_input_channels": 25,
        "channels_0_19": "current_canonical_state",
        "channel_20": "full_dynamic_freshness",
        "channel_20_value": 255.0,
        "channels_21_24": "displayed_ai_action_preview_h1_h4",
        "full_channel20_weight_frozen_zero": True,
        "recurrent": True,
        "recurrent_n": 1,
        "hidden_size": 64,
        "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
        "fixed_partner_identity": FIXED_HSP_PARTNER_ID,
        "fixed_partner_contract_id": FIXED_HSP_PARTNER_CONTRACT_ID,
        "fixed_partner_seed": 3,
        "fixed_partner_population": 24,
        "fixed_partner_pool_size": 1,
        "fixed_partner_sampling_disabled": True,
        "displayed_preview_partner_identity": FIXED_HSP_PARTNER_ID,
        "predictor_partner_alignment_verified": True,
        "migration_verified": True,
        "step0_checkpoint_saved": True,
        "train_time_evaluation": False,
        **cooking_contract(),
    }
    mismatch = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Partial source checkpoint contract mismatch: {mismatch}")

    horizon_mode = metadata.get("horizon_mode", "fixed")
    if horizon_mode == "episode_balanced":
        supported_horizons = [int(value) for value in metadata.get("horizons", ())]
        if supported_horizons != [0, 2, 4]:
            raise RuntimeError(
                "episode-balanced Full source must use horizons [0, 2, 4]"
            )
    elif horizon_mode == "fixed" and metadata.get("horizon") is not None:
        supported_horizons = [int(metadata["horizon"])]
    else:
        raise RuntimeError("Partial source checkpoint has invalid horizon contract")
    requested = {int(value) for value in requested_horizons}
    if not requested or not requested.issubset(set(supported_horizons)):
        raise RuntimeError(
            "Full source horizons do not cover requested Partial horizons"
        )

    actor_state = payload.get("actor")
    if not isinstance(actor_state, dict) or not actor_state:
        raise RuntimeError("Partial source checkpoint has no Full Action actor")
    actor_sha256 = state_dict_hash(actor_state)
    if not _is_sha256(metadata.get("actor_parameter_sha256")):
        raise RuntimeError("Partial source actor parameter hash is not recorded")
    if metadata["actor_parameter_sha256"] != actor_sha256:
        raise RuntimeError("Partial source actor parameter hash mismatch")

    state = payload.get("training_state")
    if not isinstance(state, dict):
        raise RuntimeError("Partial source is not a complete production checkpoint")
    step = metadata.get("step")
    total_steps = metadata.get("configured_total_steps")
    checkpoint_interval = metadata.get("checkpoint_interval")
    if (
        not isinstance(step, int)
        or step <= 0
        or state.get("format_id") != TRAINING_STATE_FORMAT_ID
        or state.get("completed_steps") != step
        or not isinstance(total_steps, int)
        or total_steps < step
        or not isinstance(checkpoint_interval, int)
        or checkpoint_interval <= 0
        or metadata.get("checkpoint_role") == "pre_training_initialization"
        or metadata.get("smoke") is True
        or metadata.get("long_training_started") is False
    ):
        raise RuntimeError("Partial source is not a valid production checkpoint")
    state_contract = state.get("contract")
    if not isinstance(state_contract, dict):
        raise RuntimeError("Partial source has no production training contract")
    redundant_expected = {
        "training_stage": "action",
        "observation_mode": "full",
        "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
        **cooking_contract(),
    }
    if horizon_mode == "episode_balanced":
        redundant_expected["horizon_mode"] = horizon_mode
        redundant_expected["horizons"] = [0, 2, 4]
    else:
        redundant_expected["horizon"] = supported_horizons[0]
    redundant_mismatch = {
        key: (state_contract.get(key), value)
        for key, value in redundant_expected.items()
        if state_contract.get(key) != value
    }
    if redundant_mismatch:
        raise RuntimeError(
            "Partial source training-state contract mismatch: "
            f"{redundant_mismatch}"
        )

    checkpoint_sha256 = file_sha256(source)
    if not _is_sha256(checkpoint_sha256):
        raise RuntimeError("Partial source checkpoint content hash is not recorded")
    return {
        "source_validation_contract_id": SOURCE_VALIDATION_CONTRACT_ID,
        "source_checkpoint_validated": True,
        "source_production_checkpoint": True,
        "source_absolute_path": str(source),
        "source_content_sha256": checkpoint_sha256,
        "source_actor_parameter_sha256": actor_sha256,
        "source_checkpoint_step": step,
        "source_observation_contract_id": FULL_OBSERVATION_CONTRACT,
        "source_architecture_id": ACTION_ARCHITECTURE_ID,
        "source_horizon_mode": horizon_mode,
        "source_shared_horizons": (
            supported_horizons if horizon_mode == "episode_balanced" else None
        ),
        "source_horizon": (
            supported_horizons[0] if horizon_mode == "fixed" else None
        ),
        "source_fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
        "source_fixed_partner_identity": FIXED_HSP_PARTNER_ID,
        "source_dynamics_contract": cooking_contract(),
    }


def validate_partial_full_source_checkpoint(
    path: Path,
    *,
    requested_horizons,
) -> dict[str, object]:
    """Backward-compatible name for canonical Full source validation."""
    return validate_partial_action_source_checkpoint(
        path,
        requested_horizons=requested_horizons,
    )


def validate_partial_gaze_source_checkpoint(
    path: Path,
    *,
    requested_horizons,
    requested_decision_times,
    requested_fixation_shapes,
    target_memory_lifetime_ms: float | None,
    target_memory_decay_mode: str,
    target_memory_tau_ms: float | None,
    target_constraint_config: HumanLikeConstraintConfig,
    target_ground_truth_policy_input: bool = False,
) -> dict[str, object]:
    """Validate a production learned-Gaze checkpoint for staged joint PPO."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"Partial Gaze source checkpoint does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    state = payload.get("training_state") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise RuntimeError("Partial Gaze source is not a complete checkpoint")

    target_mode = str(target_memory_decay_mode)
    target_lifetime = (
        None
        if target_mode == "none"
        else float(target_memory_lifetime_ms)
    )
    target_freshness_contract = (
        NO_DECAY_FRESHNESS_CONTRACT_ID
        if target_mode == "none"
        else STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID
        if target_mode == "stochastic_exponential"
        else CONTINUOUS_FRESHNESS_CONTRACT_ID
    )
    target_observation_contract = observation_contract_id(
        target_lifetime, target_mode, target_memory_tau_ms
    )
    expected = {
        "format_id": FORMAT_ID,
        "partial_checkpoint_format_id": PARTIAL_CHECKPOINT_FORMAT_ID,
        "training_stage": "gaze",
        "observation_mode": "partial",
        "gaze_type": "learned",
        "train_action": False,
        "train_gaze": True,
        "inference_batching": "legacy",
        "architecture_id": ACTION_ARCHITECTURE_ID,
        "gaze_architecture_id": GAZE_ARCHITECTURE_ID,
        "observation_contract_id": target_observation_contract,
        "memory_decay_mode": target_mode,
        "memory_lifetime_ms": target_lifetime,
        "freshness_contract_id": target_freshness_contract,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "partner_set_id": PARTNER_SET_ID,
        "action_space": list(ACTION_SPACE),
        "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
        "fixed_partner_identity": FIXED_HSP_PARTNER_ID,
        "fixed_partner_contract_id": FIXED_HSP_PARTNER_CONTRACT_ID,
        "execution_backend": PARALLEL_GAZE_TRAINING_CONTRACT,
        "training_contract": PARALLEL_GAZE_TRAINING_CONTRACT,
        **target_constraint_config.metadata(),
        **cooking_contract(),
    }
    if target_ground_truth_policy_input:
        expected.update({
            "ground_truth_policy_input": True,
            "policy_observation_contract_id": (
                FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
            ),
        })
    if target_mode == "stochastic_exponential":
        expected.update({
            "memory_tau_ms": target_memory_tau_ms,
            "memory_max_lifetime_ms": target_lifetime,
        })
    mismatch = {
        key: (
            metadata.get(key, "linear")
            if key == "memory_decay_mode"
            else metadata.get(key),
            value,
        )
        for key, value in expected.items()
        if (
            metadata.get(key, "linear")
            if key == "memory_decay_mode"
            else metadata.get(key)
        ) != value
    }
    requested_axes = {
        "horizons": {int(value) for value in requested_horizons},
        "decision_times": {int(value) for value in requested_decision_times},
        "fixation_shapes": {str(value) for value in requested_fixation_shapes},
    }
    for key, requested in requested_axes.items():
        supported = metadata.get(key)
        if not isinstance(supported, list) or not requested.issubset(set(supported)):
            mismatch[key] = (supported, f"covers {sorted(requested)}")
    if mismatch:
        raise RuntimeError(f"Partial Gaze source checkpoint contract mismatch: {mismatch}")

    require_current_gaze_action_contract(
        metadata, operation="parallel staged joint initialization"
    )
    require_current_gaze_input_contract(
        metadata, operation="parallel staged joint initialization"
    )
    require_current_gaze_credit_contract(
        metadata, operation="parallel staged joint initialization"
    )
    hashes = {}
    for payload_key, metadata_key, result_key in (
        ("actor", "actor_parameter_sha256", "source_actor_parameter_sha256"),
        ("critic", "critic_parameter_sha256", "source_critic_parameter_sha256"),
        ("gaze_actor", "gaze_parameter_sha256", "source_gaze_parameter_sha256"),
        (
            "gaze_critic",
            "gaze_critic_parameter_sha256",
            "source_gaze_critic_parameter_sha256",
        ),
    ):
        parameters = payload.get(payload_key)
        if not isinstance(parameters, dict) or not parameters:
            raise RuntimeError(f"Partial Gaze source has no {payload_key}")
        actual = state_dict_hash(parameters)
        if metadata.get(metadata_key) != actual:
            raise RuntimeError(f"Partial Gaze source {payload_key} hash mismatch")
        hashes[result_key] = actual

    state_contract = state.get("contract")
    required_state_fields = {
        "action_actor_optimizer",
        "action_critic_optimizer",
        "gaze_actor_optimizer",
        "gaze_critic_optimizer",
        "value_normalizer",
        "backends",
        "adapters",
        "conditions",
        "condition_decision_counts",
        "scheduler",
        "buffer",
        "rng",
        "server_parallel",
    }
    step = metadata.get("step")
    if (
        not isinstance(step, int)
        or step <= 0
        or state.get("format_id") != TRAINING_STATE_FORMAT_ID
        or state.get("completed_steps") != step
        or not isinstance(state_contract, dict)
        or not required_state_fields.issubset(state)
        or state.get("action_actor_optimizer") is not None
        or state.get("action_critic_optimizer") is not None
        or not isinstance(state.get("gaze_actor_optimizer"), dict)
        or not isinstance(state.get("gaze_critic_optimizer"), dict)
        or not isinstance(state_contract.get("configured_total_steps"), int)
        or state_contract["configured_total_steps"] < step
        or not isinstance(state_contract.get("checkpoint_interval"), int)
        or state_contract["checkpoint_interval"] <= 0
        or metadata.get("checkpoint_role") == "pre_training_initialization"
        or metadata.get("smoke") is True
        or metadata.get("long_training_started") is False
    ):
        raise RuntimeError("Partial Gaze source is not a valid production checkpoint")
    redundant_expected = {
        "partial_checkpoint_format_id": PARTIAL_CHECKPOINT_FORMAT_ID,
        "observation_mode": "partial",
        "gaze_type": "learned",
        "train_action": False,
        "train_gaze": True,
        "inference_batching": "legacy",
        "memory_contract_id": target_observation_contract,
        "memory_decay_mode": target_mode,
        "memory_lifetime_ms": target_lifetime,
        "freshness_contract_id": target_freshness_contract,
        "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
        "fixed_partner_identity": FIXED_HSP_PARTNER_ID,
        "fixed_partner_contract_id": FIXED_HSP_PARTNER_CONTRACT_ID,
        "training_contract": PARALLEL_GAZE_TRAINING_CONTRACT,
        **target_constraint_config.metadata(),
        **cooking_contract(),
    }
    if target_ground_truth_policy_input:
        redundant_expected.update({
            "ground_truth_policy_input": True,
            "policy_observation_contract_id": (
                FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
            ),
        })
    if target_mode == "stochastic_exponential":
        redundant_expected.update({
            "memory_tau_ms": target_memory_tau_ms,
            "memory_max_lifetime_ms": target_lifetime,
        })
    redundant_mismatch = {
        key: (state_contract.get(key), value)
        for key, value in redundant_expected.items()
        if state_contract.get(key) != value
    }
    for key, requested in requested_axes.items():
        supported = state_contract.get(key)
        if not isinstance(supported, list) or not requested.issubset(set(supported)):
            redundant_mismatch[key] = (supported, f"covers {sorted(requested)}")
    if redundant_mismatch:
        raise RuntimeError(
            "Partial Gaze source training-state contract mismatch: "
            f"{redundant_mismatch}"
        )
    _validate_resume_source_provenance(payload)

    return {
        "source_validation_contract_id": (
            PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID
        ),
        "source_checkpoint_validated": True,
        "source_production_checkpoint": True,
        "initialization": "staged_joint_from_partial_gaze",
        "source_kind": "canonical_partial_learned_gaze",
        "source_absolute_path": str(source),
        "source_content_sha256": file_sha256(source),
        **hashes,
        "source_checkpoint_step": step,
        "source_observation_mode": "partial",
        "source_observation_contract_id": target_observation_contract,
        "source_architecture_id": ACTION_ARCHITECTURE_ID,
        "source_gaze_architecture_id": GAZE_ARCHITECTURE_ID,
        "source_partial_checkpoint_format_id": PARTIAL_CHECKPOINT_FORMAT_ID,
        "source_horizon_mode": metadata.get("horizon_mode", "fixed"),
        "source_horizons": [int(value) for value in metadata["horizons"]],
        "source_decision_times": [
            int(value) for value in metadata["decision_times"]
        ],
        "source_fixation_shapes": [
            str(value) for value in metadata["fixation_shapes"]
        ],
        "source_memory_lifetime_ms": target_lifetime,
        "source_memory_decay_mode": target_mode,
        "source_freshness_contract_id": target_freshness_contract,
        **(
            {
                "source_memory_tau_ms": target_memory_tau_ms,
                "source_memory_max_lifetime_ms": target_lifetime,
            }
            if target_mode == "stochastic_exponential"
            else {}
        ),
        "source_fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
        "source_fixed_partner_identity": FIXED_HSP_PARTNER_ID,
        "source_fixed_partner_contract_id": FIXED_HSP_PARTNER_CONTRACT_ID,
        "source_constraint_config": constraint_config_from_metadata(
            metadata
        ).metadata(),
        "target_constraint_config": target_constraint_config.metadata(),
        "source_dynamics_contract": cooking_contract(),
    }


def _validate_resume_source_provenance(payload: dict) -> None:
    metadata = payload.get("metadata")
    state = payload.get("training_state")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise RuntimeError("server_parallel resume has no complete provenance")
    generic_provenance = metadata.get("source_checkpoint_provenance")
    legacy_provenance = metadata.get("source_full_checkpoint_provenance")
    if (
        generic_provenance is not None
        and legacy_provenance is not None
        and generic_provenance != legacy_provenance
    ):
        raise RuntimeError(
            "server_parallel resume source provenance keys disagree"
        )
    provenance = (
        generic_provenance
        if generic_provenance is not None
        else legacy_provenance
    )
    server_state = state.get("server_parallel")
    if not isinstance(provenance, dict) or not isinstance(server_state, dict):
        raise RuntimeError("server_parallel resume has no source provenance")
    validation_contract_id = provenance.get("source_validation_contract_id")
    state_contract = state.get("contract")
    if validation_contract_id == PARTIAL_SOURCE_VALIDATION_CONTRACT_ID:
        target_lifetime = (
            state_contract.get("memory_lifetime_ms")
            if isinstance(state_contract, dict)
            else None
        )
        target_mode = (
            str(state_contract.get("memory_decay_mode", "linear"))
            if isinstance(state_contract, dict)
            else "linear"
        )
        target_tau = (
            state_contract.get("memory_tau_ms")
            if isinstance(state_contract, dict)
            else None
        )
        target_freshness_contract = (
            NO_DECAY_FRESHNESS_CONTRACT_ID
            if target_mode == "none"
            else STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID
            if target_mode == "stochastic_exponential"
            else CONTINUOUS_FRESHNESS_CONTRACT_ID
        )
        source_expected = {
            "initialization": "partial_checkpoint",
            "source_kind": "canonical_partial_action",
            "source_observation_mode": "partial",
            "source_partial_checkpoint_format_id": (
                PARTIAL_CHECKPOINT_FORMAT_ID
            ),
            "source_freshness_contract_id": target_freshness_contract,
            "source_memory_decay_mode": target_mode,
            "source_memory_lifetime_ms": target_lifetime,
            "source_fixed_partner_contract_id": (
                FIXED_HSP_PARTNER_CONTRACT_ID
            ),
        }
        if target_mode == "stochastic_exponential":
            source_expected.update({
                "source_memory_tau_ms": target_tau,
                "source_memory_max_lifetime_ms": target_lifetime,
            })
        try:
            source_lifetime = provenance["source_memory_lifetime_ms"]
            if target_mode != "none":
                source_lifetime = float(source_lifetime)
            source_observation_contract = observation_contract_id(
                source_lifetime, target_mode, target_tau
            )
        except (KeyError, TypeError, ValueError):
            source_observation_contract = None
        source_expected["source_observation_contract_id"] = (
            source_observation_contract
        )
    elif validation_contract_id == PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID:
        target_lifetime = (
            state_contract.get("memory_lifetime_ms")
            if isinstance(state_contract, dict)
            else None
        )
        target_mode = (
            str(state_contract.get("memory_decay_mode", "linear"))
            if isinstance(state_contract, dict)
            else "linear"
        )
        target_tau = (
            state_contract.get("memory_tau_ms")
            if isinstance(state_contract, dict)
            else None
        )
        target_freshness_contract = (
            NO_DECAY_FRESHNESS_CONTRACT_ID
            if target_mode == "none"
            else STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID
            if target_mode == "stochastic_exponential"
            else CONTINUOUS_FRESHNESS_CONTRACT_ID
        )
        source_expected = {
            "initialization": "staged_joint_from_partial_gaze",
            "source_kind": "canonical_partial_learned_gaze",
            "source_observation_mode": "partial",
            "source_partial_checkpoint_format_id": (
                PARTIAL_CHECKPOINT_FORMAT_ID
            ),
            "source_gaze_architecture_id": GAZE_ARCHITECTURE_ID,
            "source_freshness_contract_id": target_freshness_contract,
            "source_memory_decay_mode": target_mode,
            "source_memory_lifetime_ms": target_lifetime,
            "source_fixed_partner_contract_id": (
                FIXED_HSP_PARTNER_CONTRACT_ID
            ),
            "source_observation_contract_id": observation_contract_id(
                target_lifetime, target_mode, target_tau
            ),
        }
        if target_mode == "stochastic_exponential":
            source_expected.update({
                "source_memory_tau_ms": target_tau,
                "source_memory_max_lifetime_ms": target_lifetime,
            })
    elif validation_contract_id == SOURCE_VALIDATION_CONTRACT_ID:
        source_expected = {
            "source_observation_contract_id": FULL_OBSERVATION_CONTRACT,
        }
    else:
        source_expected = {}
    expected = {
        "source_validation_contract_id": validation_contract_id,
        "source_checkpoint_validated": True,
        "source_production_checkpoint": True,
        "source_architecture_id": ACTION_ARCHITECTURE_ID,
        "source_fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
        "source_fixed_partner_identity": FIXED_HSP_PARTNER_ID,
        "source_dynamics_contract": cooking_contract(),
        **source_expected,
    }
    mismatch = {
        key: (
            provenance.get(key, "linear")
            if key == "source_memory_decay_mode"
            else provenance.get(key),
            value,
        )
        for key, value in expected.items()
        if (
            provenance.get(key, "linear")
            if key == "source_memory_decay_mode"
            else provenance.get(key)
        ) != value
    }
    if validation_contract_id in {
        PARTIAL_SOURCE_VALIDATION_CONTRACT_ID,
        PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID,
    }:
        source_horizons = provenance.get("source_horizons")
        target_horizons = state_contract.get("horizons")
        if (
            not isinstance(source_horizons, list)
            or not isinstance(target_horizons, list)
            or not set(target_horizons).issubset(set(source_horizons))
        ):
            mismatch["source_horizons"] = (
                source_horizons,
                f"covers resume horizons {target_horizons}",
            )
        if not isinstance(provenance.get("source_checkpoint_step"), int) or (
            provenance["source_checkpoint_step"] <= 0
        ):
            mismatch["source_checkpoint_step"] = (
                provenance.get("source_checkpoint_step"),
                "positive completed step",
            )
    hash_keys = ["source_content_sha256", "source_actor_parameter_sha256"]
    if validation_contract_id in {
        PARTIAL_SOURCE_VALIDATION_CONTRACT_ID,
        PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID,
    }:
        hash_keys.append("source_critic_parameter_sha256")
    if validation_contract_id == PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID:
        hash_keys.extend((
            "source_gaze_parameter_sha256",
            "source_gaze_critic_parameter_sha256",
        ))
    for key in hash_keys:
        value = provenance.get(key)
        if not _is_sha256(value):
            mismatch[key] = (value, "recorded sha256")
        elif server_state.get(key) != value:
            mismatch[f"training_state.{key}"] = (server_state.get(key), value)
    if validation_contract_id not in {
        SOURCE_VALIDATION_CONTRACT_ID,
        PARTIAL_SOURCE_VALIDATION_CONTRACT_ID,
        PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID,
    }:
        mismatch["source_validation_contract_id"] = (
            validation_contract_id,
            "supported source validation contract",
        )
    if server_state.get("source_validation_contract_id") != validation_contract_id:
        mismatch["training_state.source_validation_contract_id"] = (
            server_state.get("source_validation_contract_id"),
            validation_contract_id,
        )
    if mismatch:
        raise RuntimeError(
            f"server_parallel resume source provenance mismatch: {mismatch}"
        )


class ServerParallelActionTrainingSession(PartialTrainingSession):
    """Synchronous Partial rollout workers for Action or learned-Gaze PPO."""

    def __init__(
        self,
        *,
        rollout_workers: int = 1,
        server_trace: bool = False,
        memory_audit_enabled: bool = False,
        host_mode: str = PARALLEL_GAZE_HOST_MODE,
        condition_name: str | None = None,
        memory_condition: str | None = None,
        perception_condition: str | None = None,
        **kwargs,
    ) -> None:
        action_mode = (
            kwargs.get("gaze_type") == "rule_based"
            and bool(kwargs.get("train_action"))
            and not bool(kwargs.get("train_gaze"))
        )
        gaze_mode = (
            kwargs.get("gaze_type") == "learned"
            and not bool(kwargs.get("train_action"))
            and bool(kwargs.get("train_gaze"))
        )
        joint_mode = (
            kwargs.get("gaze_type") == "learned"
            and bool(kwargs.get("train_action"))
            and bool(kwargs.get("train_gaze"))
        )
        if not (action_mode or gaze_mode or joint_mode):
            raise ValueError(
                "parallel rollout backend supports rule-based Action-only or "
                "learned-Gaze-only or learned staged/direct joint training"
            )
        direct_joint_mode = joint_mode and bool(
            kwargs.get("direct_joint_from_full")
        )
        staged_joint_mode = joint_mode and not direct_joint_mode
        self.parallel_joint = joint_mode
        self.parallel_gaze = gaze_mode or joint_mode
        self.parallel_training_contract = (
            PARALLEL_DIRECT_JOINT_TRAINING_CONTRACT
            if direct_joint_mode
            else PARALLEL_STAGED_JOINT_TRAINING_CONTRACT
            if staged_joint_mode
            else PARALLEL_GAZE_TRAINING_CONTRACT
            if gaze_mode
            else None
        )
        self._main_cpu_execution = (
            configure_parallel_gaze_main_cpu_execution()
            if self.parallel_gaze else None
        )
        self.execution_backend = (
            self.parallel_training_contract
            if self.parallel_gaze else SERVER_PARALLEL_VERSION
        )
        self.host_mode = str(host_mode)
        self.condition_name = condition_name
        self.memory_condition = memory_condition
        self.perception_condition = perception_condition
        if self.parallel_gaze and not self.host_mode:
            raise ValueError("parallel Gaze host_mode must be non-empty")
        if kwargs.get("partner") is not None:
            raise ValueError("server_parallel_v1 does not accept arbitrary partners")
        workers = int(rollout_workers)
        environments = int(kwargs["n_rollout_threads"])
        self.assignments = deterministic_assignments(environments, workers)
        self.rollout_workers = workers
        self.server_trace = bool(server_trace)
        self.memory_audit_enabled = bool(memory_audit_enabled)
        self.resume_checkpoint_provenance: dict[str, object] | None = None
        self.resume_continuation: dict[str, int] | None = None
        self.resume_cpu_execution_contract_migration: dict[str, object] | None = None
        source = kwargs.get("source_checkpoint")
        resume = kwargs.get("resume_checkpoint")
        fixed_partner = kwargs.get("fixed_partner_checkpoint")
        if (
            fixed_partner is None
            or not Path(fixed_partner).is_file()
            or file_sha256(fixed_partner) != FIXED_HSP_PARTNER_SHA256
        ):
            raise ValueError("server_parallel_v1 requires the canonical fixed partner")
        source_validation = None
        resume_gaze_generator_states: dict[int, torch.Tensor] = {}
        if resume is None:
            if source is None:
                raise ValueError(
                    "server_parallel_v1 requires a canonical source checkpoint"
                )
            if staged_joint_mode:
                source_validation = validate_partial_gaze_source_checkpoint(
                    source,
                    requested_horizons=kwargs.get("horizons") or (),
                    requested_decision_times=(
                        kwargs.get("decision_times") or ()
                    ),
                    requested_fixation_shapes=(
                        kwargs.get("fixation_shapes") or ()
                    ),
                    target_memory_lifetime_ms=kwargs.get(
                        "memory_lifetime_ms", MEMORY_LIFETIME_MS
                    ),
                    target_memory_decay_mode=kwargs.get(
                        "memory_decay_mode", "linear"
                    ),
                    target_memory_tau_ms=kwargs.get("memory_tau_ms"),
                    target_constraint_config=kwargs["constraint_config"],
                    target_ground_truth_policy_input=bool(
                        kwargs.get("ground_truth_policy_input", False)
                    ),
                )
            else:
                source_validation = validate_partial_action_source_checkpoint(
                    source,
                    requested_horizons=kwargs.get("horizons") or (),
                    target_memory_lifetime_ms=kwargs.get(
                        "memory_lifetime_ms", MEMORY_LIFETIME_MS
                    ),
                    target_memory_decay_mode=kwargs.get(
                        "memory_decay_mode", "linear"
                    ),
                    target_memory_tau_ms=kwargs.get("memory_tau_ms"),
                )
        else:
            resume_path = Path(resume).expanduser().resolve()
            payload = torch.load(
                resume_path, map_location="cpu", weights_only=False
            )
            metadata = payload.get("metadata", {})
            state = payload.get("training_state", {})
            expected = {
                "execution_backend": self.execution_backend,
            }
            if self.parallel_gaze:
                expected.update(
                    training_contract=self.parallel_training_contract,
                )
            else:
                expected["rollout_workers"] = workers
            mismatch = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if self.parallel_gaze and mismatch:
                raise RuntimeError(
                    "parallel_gaze_v1 refuses serial or foreign checkpoint "
                    f"resume; start a new run instead: {mismatch}"
                )
            _validate_resume_source_provenance(payload)
            if mismatch:
                raise RuntimeError(
                    f"server_parallel resume execution contract mismatch: {mismatch}"
                )
            resume_start_step = state.get("completed_steps")
            if (
                not isinstance(resume_start_step, int)
                or resume_start_step <= 0
                or metadata.get("step") != resume_start_step
            ):
                raise RuntimeError(
                    "server_parallel resume progress provenance mismatch"
                )
            self.resume_checkpoint_provenance = {
                "resume_provenance_contract_id": RESUME_PROVENANCE_CONTRACT_ID,
                "resume_checkpoint_absolute_path": str(resume_path),
                "resume_checkpoint_content_sha256": file_sha256(resume_path),
                "resume_start_step": resume_start_step,
            }
            if self.parallel_gaze:
                parallel_state = state.get("server_parallel", {})
                saved_generators = parallel_state.get("gaze_generator_states")
                if not isinstance(saved_generators, dict):
                    raise RuntimeError(
                        "parallel_gaze_v1 resume has no logical-env RNG states"
                    )
                resume_gaze_generator_states = {
                    int(key): value for key, value in saved_generators.items()
                }

        self._context = mp.get_context("spawn")
        self._shared = SharedDecisionBuffers.create(
            self._context, environments
        )
        self._views = self._shared.views()
        self._response_queue = self._context.Queue()
        self._command_queues = []
        self._workers = []
        self._serial = 0
        self._closed = False
        for assignment in self.assignments:
            commands = self._context.Queue()
            process = self._context.Process(
                target=rollout_worker_main,
                args=(
                    assignment.worker_id,
                    assignment.env_ids,
                    commands,
                    self._response_queue,
                    self._shared,
                ),
                name=f"partial-rollout-worker-{assignment.worker_id}",
                daemon=False,
            )
            process.start()
            self._command_queues.append(commands)
            self._workers.append(process)
        boot = self._wait_boot()
        self.worker_pids = [int(row["pid"]) for row in boot]
        if any(row["cuda_initialized"] for row in boot):
            self.close()
            raise RuntimeError("a rollout worker initialized CUDA")
        try:
            super().__init__(**kwargs)
            if source_validation is not None:
                comparison_keys = [
                    "source_content_sha256",
                    "source_actor_parameter_sha256",
                ]
                if "source_critic_parameter_sha256" in source_validation:
                    comparison_keys.append("source_critic_parameter_sha256")
                if "source_gaze_parameter_sha256" in source_validation:
                    comparison_keys.append("source_gaze_parameter_sha256")
                if "source_gaze_critic_parameter_sha256" in source_validation:
                    comparison_keys.append(
                        "source_gaze_critic_parameter_sha256"
                    )
                for key in comparison_keys:
                    if self.source_provenance.get(key) != source_validation[key]:
                        raise RuntimeError(
                            f"canonical Partial source migration mismatch: {key}"
                        )
                self.source_provenance.update(source_validation)
            install_rollout_context_stable_actor(
                self.policy.actor, microbatch_size=self.n_rollout_threads
            )
            if gaze_mode:
                self.policy.actor_optimizer = None
                self.policy.critic_optimizer = None
            backend_states = [backend.training_state() for backend in self.backends]
            adapter_states = [adapter.training_state() for adapter in self.adapters]
            self._partner_pools = [backend.partners for backend in self.backends]
            repositories = {
                id(pool.repository): pool.repository for pool in self._partner_pools
            }
            if len(repositories) != 1:
                raise RuntimeError("canonical fixed partner must share one repository")
            self._partner_repository = next(iter(repositories.values()))
            self._partner_parameter_sha256 = self._partner_pools[0].assert_frozen(
                full_hash=True
            )
            if any(
                pool.assert_frozen(full_hash=True) != self._partner_parameter_sha256
                for pool in self._partner_pools
            ):
                raise RuntimeError("fixed partner parameter identity differs across envs")

            environment_config = {}
            for env_id, condition in enumerate(self.conditions):
                worker_backend_state = copy.deepcopy(backend_states[env_id])
                worker_adapter_state = adapter_states[env_id]
                worker_condition = PartialCondition(
                    int(worker_adapter_state["decision_time"]),
                    int(worker_backend_state["horizon"]),
                    str(worker_adapter_state["fixation_shape"]),
                )
                if worker_condition != condition:
                    raise RuntimeError(
                        "worker startup active-condition contract mismatch: "
                        f"env={env_id} state={worker_condition} "
                        f"active={condition}"
                    )
                worker_backend_state["partners"]["torch_rng"] = (
                    torch.Generator(device="cpu")
                    .manual_seed(self.seed + env_id + 91)
                    .get_state()
                )
                environment_config[env_id] = {
                    "seed": self.seed + env_id,
                    "rule_gaze_seed": int(
                        worker_adapter_state["gaze_policy"].get(
                            "seed", self.seed + 83_000_009 + env_id
                        )
                    ),
                    "peripheral_seed": int(
                        worker_adapter_state["peripheral_seed"]
                    ),
                    "opacity_seed": int(worker_adapter_state["opacity_seed"]),
                    "dwell_seed": worker_adapter_state["gaze"].get("dwell_seed"),
                    "memory_seed": worker_adapter_state["memory"].get(
                        "memory_seed"
                    ),
                    "condition": {
                        "decision_time": condition.decision_time,
                        "horizon": condition.horizon,
                        "fixation_shape": condition.fixation_shape,
                    },
                    "backend_state": worker_backend_state,
                    "adapter_state": adapter_states[env_id],
                    "gaze_episode_index": int(
                        worker_adapter_state.get("gaze_policy", {}).get(
                            "episode_id", 0
                        )
                    ),
                    "gaze_generator_state": resume_gaze_generator_states.get(
                        env_id
                    ),
                }
            del self.backend, self.adapter, self.backends, self.adapters
            self.controllers = []
            gc.collect()
            common_worker_config = {
                "gaze_type": self.gaze_type,
                "master_seed": self.seed,
                "episode_length": self.episode_length,
                "enable_predictor": any(
                    value > 0 for value in self.scheduler.horizons
                ),
                "partner_parameter_sha256": self._partner_parameter_sha256,
                "partner_checkpoint_sha256": FIXED_HSP_PARTNER_SHA256,
                "rule_gaze_probabilities": self.rule_gaze_probabilities,
                "memory_lifetime_ms": self.memory_lifetime_ms,
                "memory_decay_mode": self.memory_decay_mode,
                "memory_tau_ms": self.memory_tau_ms,
                "ground_truth_policy_input": self.ground_truth_policy_input,
                "constraint_config": self.constraint_config,
                "trace_enabled": self.server_trace,
                "memory_audit_enabled": self.memory_audit_enabled,
            }
            if self.parallel_gaze:
                common_worker_config.update({
                    "worker_torch_intra_op_threads": (
                        PARALLEL_GAZE_CPU_INTRA_OP_THREADS
                    ),
                    "worker_torch_inter_op_threads": (
                        PARALLEL_GAZE_CPU_INTER_OP_THREADS
                    ),
                    "gaze_cpu_inference_dtype": (
                        PARALLEL_GAZE_CPU_INFERENCE_DTYPE
                    ),
                    "gaze_cpu_inference_path": (
                        PARALLEL_GAZE_CPU_INFERENCE_PATH
                    ),
                    "gaze_actor_state": {
                        key: value.detach().cpu().clone()
                        for key, value in self.gaze_actor.state_dict().items()
                    },
                    "gaze_critic_state": {
                        key: value.detach().cpu().clone()
                        for key, value in self.gaze_critic.state_dict().items()
                    },
                    "gaze_actor_sha256": state_dict_hash(
                        self.gaze_actor.state_dict()
                    ),
                    "gaze_critic_sha256": state_dict_hash(
                        self.gaze_critic.state_dict()
                    ),
                    "gaze_policy_version": self.update_count,
                })
            worker_configs = [{
                **common_worker_config,
                "environments": {
                    env_id: environment_config[env_id]
                    for env_id in assignment.env_ids
                },
            } for assignment in self.assignments]
            startup = self._initialize_workers(worker_configs)
            if any(row["cuda_initialized"] for row in startup):
                raise RuntimeError("a rollout worker initialized CUDA")
            if self.parallel_gaze:
                actor_hash = state_dict_hash(self.gaze_actor.state_dict())
                critic_hash = state_dict_hash(self.gaze_critic.state_dict())
                if any(
                    row.get("gaze_actor_sha256") != actor_hash
                    or row.get("gaze_critic_sha256") != critic_hash
                    or row.get("gaze_rng_contract") != PARALLEL_GAZE_RNG_CONTRACT
                    or row.get("torch_cpu_intra_op_threads") != (
                        self._main_cpu_execution["torch_cpu_intra_op_threads"]
                    )
                    or row.get("torch_cpu_inter_op_threads") != (
                        self._main_cpu_execution["torch_cpu_inter_op_threads"]
                    )
                    or row.get("gaze_cpu_inference_dtype") != (
                        PARALLEL_GAZE_CPU_INFERENCE_DTYPE
                    )
                    or row.get("gaze_cpu_inference_path") != (
                        PARALLEL_GAZE_CPU_INFERENCE_PATH
                    )
                    or row.get("gaze_actor_eval") is not True
                    or row.get("gaze_actor_requires_grad") is not False
                    for row in startup
                ):
                    raise RuntimeError("parallel Gaze worker startup contract mismatch")
                self._parallel_gaze_worker_execution = [
                    {
                        key: row[key]
                        for key in (
                            "worker_id",
                            "torch_cpu_intra_op_threads",
                            "torch_cpu_inter_op_threads",
                            "gaze_cpu_inference_dtype",
                            "gaze_cpu_inference_path",
                            "gaze_actor_eval",
                            "gaze_actor_requires_grad",
                        )
                    }
                    for row in startup
                ]
        except BaseException:
            self.close()
            raise
        self.last_rollout_trace: list[dict[str, object]] = []
        self.last_execution_timings: dict[str, float] = {}

    def _resume(self, path: Path) -> None:
        """Keep canonical exact validation, except for two operational fields."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("training_state")
        if not isinstance(state, dict):
            raise RuntimeError(
                "Partial resume checkpoint has no complete training_state"
            )
        stored_contract = state.get("contract")
        stored_steps = state.get("completed_steps")
        if not isinstance(stored_steps, int):
            raise RuntimeError(
                "Partial resume checkpoint has invalid completed steps"
            )
        requested_total = self.configured_total_steps
        requested_interval = self.checkpoint_interval
        continuation = validate_partial_resume_continuation(
            stored_contract,
            stored_environment_steps=stored_steps,
            requested_total_steps=requested_total,
            requested_checkpoint_interval=requested_interval,
            update_size=self.rollout_length * self.n_rollout_threads,
        )
        legacy_cpu_contract = False
        if self.parallel_gaze and isinstance(stored_contract, dict):
            present = [key in stored_contract for key in _MATCHED_CPU_CONTRACT_KEYS]
            if any(present) and not all(present):
                raise RuntimeError(
                    "parallel Gaze resume has an incomplete matched CPU contract"
                )
            legacy_cpu_contract = not any(present)
            if legacy_cpu_contract:
                parallel_state = state.get("server_parallel")
                legacy_expected = {
                    "worker_torch_intra_op_threads": (
                        PARALLEL_GAZE_CPU_INTRA_OP_THREADS
                    ),
                    "worker_torch_inter_op_threads": (
                        PARALLEL_GAZE_CPU_INTER_OP_THREADS
                    ),
                }
                mismatch = {
                    key: (stored_contract.get(key), expected)
                    for key, expected in legacy_expected.items()
                    if stored_contract.get(key) != expected
                }
                if not isinstance(parallel_state, dict) or parallel_state.get(
                    "logprob_audit_version"
                ) != PARALLEL_GAZE_LEGACY_LOGPROB_AUDIT_VERSION:
                    mismatch["logprob_audit_version"] = (
                        parallel_state.get("logprob_audit_version")
                        if isinstance(parallel_state, dict) else None,
                        PARALLEL_GAZE_LEGACY_LOGPROB_AUDIT_VERSION,
                    )
                if mismatch:
                    raise RuntimeError(
                        "parallel Gaze legacy CPU audit migration mismatch: "
                        f"{mismatch}"
                    )
                self.resume_cpu_execution_contract_migration = {
                    "from": PARALLEL_GAZE_LEGACY_LOGPROB_AUDIT_VERSION,
                    "to": PARALLEL_GAZE_LOGPROB_AUDIT_VERSION,
                    "scope": "main_cpu_thread_and_inference_contract_only",
                }
        saved_conditions = state.get("conditions")
        if not isinstance(saved_conditions, list) or len(saved_conditions) != (
            self.n_rollout_threads
        ):
            raise RuntimeError("Partial resume active-condition count mismatch")
        if not (
            len(self.backends)
            == len(self.adapters)
            == len(self.conditions)
            == self.n_rollout_threads
        ):
            raise RuntimeError("Partial resume runtime environment count mismatch")
        for env_id, (backend, adapter, saved) in enumerate(
            zip(self.backends, self.adapters, saved_conditions)
        ):
            condition = PartialCondition(**saved)
            if backend.episode_step != 0:
                raise RuntimeError(
                    "Partial resume condition restoration requires fresh "
                    f"episode boundary: env={env_id} "
                    f"episode_step={backend.episode_step}"
                )
            if adapter.backend is not backend:
                raise RuntimeError(
                    f"Partial resume backend/adapter ordering mismatch: env={env_id}"
                )
            backend.set_horizon(condition.horizon)
            adapter.set_condition(
                decision_time=condition.decision_time,
                horizon=condition.horizon,
                fixation_shape=condition.fixation_shape,
            )
            self.conditions[env_id] = condition
            adapter_condition = PartialCondition(
                int(adapter.decision_time),
                int(adapter.backend.horizon),
                str(adapter.fixation_shape),
            )
            if not (
                int(backend.horizon) == condition.horizon
                and adapter_condition == condition
                and self.conditions[env_id] == condition
            ):
                raise RuntimeError(
                    "Partial resume active-condition restoration mismatch: "
                    f"env={env_id} backend_horizon={backend.horizon} "
                    f"adapter={adapter_condition} "
                    f"active={self.conditions[env_id]} saved={condition}"
                )
        self.configured_total_steps = continuation[
            "stored_configured_total_steps"
        ]
        self.checkpoint_interval = continuation["stored_checkpoint_interval"]
        self._resume_legacy_cpu_execution_contract = legacy_cpu_contract
        try:
            super()._resume(path)
        finally:
            self._resume_legacy_cpu_execution_contract = False
            self.configured_total_steps = requested_total
            self.checkpoint_interval = requested_interval
        if self.completed_steps != continuation["stored_environment_steps"]:
            raise RuntimeError("Partial resume restored unexpected progress")
        self.resume_continuation = continuation

    def _get_response(self) -> dict[str, object]:
        try:
            response = self._response_queue.get(
                timeout=WORKER_RESPONSE_TIMEOUT_SECONDS
            )
        except queue.Empty as error:
            dead = [process.pid for process in self._workers if not process.is_alive()]
            raise RuntimeError(f"rollout worker response timeout; dead={dead}") from error
        if response.get("phase") == "error":
            raise RuntimeError(
                f"rollout worker {response.get('worker_id')} failed: "
                f"{response.get('error')}\n{response.get('traceback')}"
            )
        return response

    def _wait_boot(self):
        responses = [self._get_response() for _ in self._workers]
        if {row.get("phase") for row in responses} != {"boot"}:
            raise RuntimeError("invalid rollout worker boot response")
        ordered = sorted(responses, key=lambda row: int(row["worker_id"]))
        if [tuple(row["env_ids"]) for row in ordered] != [
            assignment.env_ids for assignment in self.assignments
        ]:
            raise RuntimeError("worker boot assignment mismatch")
        return ordered

    def _initialize_workers(self, worker_configs):
        self._serial += 1
        serial = self._serial
        for commands, config in zip(self._command_queues, worker_configs):
            commands.put(("initialize", serial, config))
        responses = [self._get_response() for _ in self._workers]
        ordered = sorted(responses, key=lambda row: int(row["worker_id"]))
        if (
            any(row.get("phase") != "startup" for row in ordered)
            or any(int(row.get("serial", -1)) != serial for row in ordered)
        ):
            raise RuntimeError("invalid rollout worker initialization response")
        return ordered

    def _phase(self, phase: str, *arguments) -> tuple[list[dict], float, float]:
        self._serial += 1
        serial = self._serial
        started = time.perf_counter()
        for commands in self._command_queues:
            commands.put((phase, serial, *arguments))
        responses = [self._get_response() for _ in self._workers]
        wall = time.perf_counter() - started
        expected_workers = set(range(self.rollout_workers))
        if (
            {int(row.get("worker_id", -1)) for row in responses} != expected_workers
            or any(row.get("phase") != phase for row in responses)
            or any(int(row.get("serial", -1)) != serial for row in responses)
        ):
            raise RuntimeError(f"invalid {phase} worker barrier response")
        ordered = sorted(responses, key=lambda row: int(row["worker_id"]))
        worker_elapsed = max(
            (float(row.get("elapsed_seconds", 0.0)) for row in ordered), default=0.0
        )
        return ordered, wall, max(0.0, wall - worker_elapsed)

    def startup_report(self) -> dict[str, object]:
        report = {
            "execution_backend": self.execution_backend,
            "training_contract": (
                self.parallel_training_contract
                if self.parallel_gaze else None
            ),
            "host_mode": self.host_mode,
            "rollout_workers": self.rollout_workers,
            "n_rollout_threads": self.n_rollout_threads,
            "worker_assignments": {
                str(item.worker_id): list(item.env_ids) for item in self.assignments
            },
            "source_checkpoint_sha256": self.source_provenance[
                "source_content_sha256"
            ],
            "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
            "fixed_partner_parameter_sha256": self._partner_parameter_sha256,
            "cooking_contract": cooking_contract(),
            "multiprocessing_start_method": self._context.get_start_method(),
            "device": str(self.device),
            "worker_pids": list(self.worker_pids),
            "maximum_acquisitions_per_decision": MAX_ACQUISITIONS_PER_DECISION,
            "memory_lifetime_ms": self.memory_lifetime_ms,
            "memory_decay_mode": self.memory_decay_mode,
            "freshness_contract_id": self.freshness_contract_id,
            "ground_truth_policy_input": self.ground_truth_policy_input,
            "memory_audit_enabled": self.memory_audit_enabled,
        }
        if self.parallel_gaze:
            report.update({
                "condition_name": self.condition_name,
                "memory_condition": self.memory_condition,
                "perception_condition": self.perception_condition,
                "logprob_audit_version": (
                    PARALLEL_GAZE_LOGPROB_AUDIT_VERSION
                ),
                "main_cpu_execution": dict(self._main_cpu_execution),
                "worker_cpu_execution": copy.deepcopy(
                    self._parallel_gaze_worker_execution
                ),
                "cpu_execution_settings_matched": True,
                "action_checkpoint_path": self.source_provenance[
                    "source_absolute_path"
                ],
                "action_optimizer_present": (
                    self.policy.actor_optimizer is not None
                ),
                "action_requires_grad": any(
                    parameter.requires_grad
                    for parameter in self.policy.actor.parameters()
                ),
                "action_critic_optimizer_present": (
                    self.policy.critic_optimizer is not None
                ),
                "action_critic_requires_grad": any(
                    parameter.requires_grad
                    for parameter in self.policy.critic.parameters()
                ),
                "action_optimizer_isolation_verified": (
                    self.action_optimizer_isolation_verified
                ),
                "action_actor_parameter_sha256": parameter_hash(
                    self.policy.actor
                ),
                "action_critic_parameter_sha256": parameter_hash(
                    self.policy.critic
                ),
            })
        if self.resume_checkpoint_provenance is not None:
            report["resume_partial_checkpoint_provenance"] = copy.deepcopy(
                self.resume_checkpoint_provenance
            )
        if self.resume_continuation is not None:
            report["resume_continuation"] = dict(self.resume_continuation)
        if self.resume_cpu_execution_contract_migration is not None:
            report["resume_cpu_execution_contract_migration"] = dict(
                self.resume_cpu_execution_contract_migration
            )
        return report

    def _runtime_reports(self) -> list[dict[str, object]]:
        responses, _wall, _ipc = self._phase("metrics")
        rows = [item for response in responses for item in response["payload"]]
        return sorted(rows, key=lambda row: int(row["env_id"]))

    def _snapshot_workers(self):
        responses, _wall, _ipc = self._phase("snapshot")
        rows = [item for response in responses for item in response["payload"]]
        ordered = sorted(rows, key=lambda row: int(row["env_id"]))
        if [int(row["env_id"]) for row in ordered] != list(
            range(self.n_rollout_threads)
        ):
            raise RuntimeError("worker snapshot environment ordering mismatch")
        backends, adapters = [], []
        self._last_gaze_generator_states: dict[int, torch.Tensor] = {}
        for env_id, row in enumerate(ordered):
            backend = copy.deepcopy(row["backend"])
            backend["partners"] = self._partner_pools[env_id].training_state()
            backends.append(backend)
            adapters.append(row["adapter"])
            if self.parallel_gaze:
                generator_state = row.get("gaze_generator_state")
                if not isinstance(generator_state, torch.Tensor):
                    raise RuntimeError(
                        "parallel Gaze snapshot has no logical-env RNG state"
                    )
                self._last_gaze_generator_states[env_id] = (
                    generator_state.detach().cpu().clone()
                )
        return backends, adapters

    def _contract(self) -> dict[str, Any]:
        contract = super()._contract()
        if self.parallel_gaze:
            contract.update({
                "training_contract": self.parallel_training_contract,
                "gaze_rng_scheme_version": PARALLEL_GAZE_RNG_CONTRACT,
                "logical_env_count": self.n_rollout_threads,
                "rollout_synchronization": "synchronous_frozen_policy_v1",
                "transition_merge_order": (
                    "logical_env_id_episode_id_record_sequence_id"
                ),
                "ppo_update_owner": "main_process_only",
                "worker_torch_intra_op_threads": 1,
                "worker_torch_inter_op_threads": 1,
                "condition_name": self.condition_name,
                "memory_condition": self.memory_condition,
                "perception_condition": self.perception_condition,
            })
            if not getattr(
                self, "_resume_legacy_cpu_execution_contract", False
            ):
                contract.update({
                    "main_torch_intra_op_threads": (
                        PARALLEL_GAZE_CPU_INTRA_OP_THREADS
                    ),
                    "main_torch_inter_op_threads": (
                        PARALLEL_GAZE_CPU_INTER_OP_THREADS
                    ),
                    "gaze_cpu_inference_dtype": (
                        PARALLEL_GAZE_CPU_INFERENCE_DTYPE
                    ),
                    "gaze_cpu_inference_path": (
                        PARALLEL_GAZE_CPU_INFERENCE_PATH
                    ),
                    "gaze_logprob_audit_version": (
                        PARALLEL_GAZE_LOGPROB_AUDIT_VERSION
                    ),
                })
        return contract

    def _training_state(self) -> dict[str, Any]:
        backends, adapters = self._snapshot_workers()
        state = {
            "format_id": TRAINING_STATE_FORMAT_ID,
            "contract": self._contract(),
            "completed_steps": self.completed_steps,
            "update_count": self.update_count,
            "action_actor_optimizer": (
                None if self.policy.actor_optimizer is None
                else self.policy.actor_optimizer.state_dict()
            ),
            "action_critic_optimizer": (
                None if self.policy.critic_optimizer is None
                else self.policy.critic_optimizer.state_dict()
            ),
            "value_normalizer": self.trainer.value_normalizer.state_dict(),
            "gaze_actor_optimizer": (
                None if self.gaze_optimizer is None
                else self.gaze_optimizer.state_dict()
            ),
            "gaze_critic_optimizer": (
                None if self.gaze_critic_optimizer is None
                else self.gaze_critic_optimizer.state_dict()
            ),
            "backends": backends,
            "adapters": adapters,
            "conditions": [condition.__dict__ for condition in self.conditions],
            "condition_decision_counts": [
                {**condition.__dict__, "count": count}
                for condition, count in self.condition_decision_counts.items()
            ],
            "scheduler": self.scheduler.state_dict(),
            "buffer": self._buffer_state(self.buffer),
            "rng": capture_global_rng_state(),
            "server_parallel": {
                "execution_backend": self.execution_backend,
                "training_contract": (
                    self.parallel_training_contract
                    if self.parallel_gaze else None
                ),
                "logprob_audit_version": (
                    PARALLEL_GAZE_LOGPROB_AUDIT_VERSION
                    if self.parallel_gaze else None
                ),
                "host_mode": self.host_mode,
                "main_cpu_execution": (
                    dict(self._main_cpu_execution)
                    if self.parallel_gaze else None
                ),
                "worker_cpu_execution": (
                    copy.deepcopy(self._parallel_gaze_worker_execution)
                    if self.parallel_gaze else None
                ),
                "cpu_execution_settings_matched": (
                    True if self.parallel_gaze else None
                ),
                "rollout_workers": self.rollout_workers,
                "source_validation_contract_id": (
                    self.source_provenance["source_validation_contract_id"]
                ),
                "source_content_sha256": self.source_provenance[
                    "source_content_sha256"
                ],
                "source_actor_parameter_sha256": self.source_provenance[
                    "source_actor_parameter_sha256"
                ],
                **(
                    {
                        "source_critic_parameter_sha256": self.source_provenance[
                            "source_critic_parameter_sha256"
                        ]
                    }
                    if "source_critic_parameter_sha256" in self.source_provenance
                    else {}
                ),
                **(
                    {
                        "source_gaze_parameter_sha256": self.source_provenance[
                            "source_gaze_parameter_sha256"
                        ],
                        "source_gaze_critic_parameter_sha256": (
                            self.source_provenance[
                                "source_gaze_critic_parameter_sha256"
                            ]
                        ),
                    }
                    if "source_gaze_parameter_sha256" in self.source_provenance
                    and "source_gaze_critic_parameter_sha256"
                    in self.source_provenance
                    else {}
                ),
                "assignments": {
                    str(item.worker_id): list(item.env_ids)
                    for item in self.assignments
                },
                "gaze_generator_states": (
                    {
                        str(env_id): value
                        for env_id, value in self._last_gaze_generator_states.items()
                    }
                    if self.parallel_gaze else None
                ),
            },
        }
        if self.resume_checkpoint_provenance is not None:
            state["server_parallel"][
                "resume_partial_checkpoint_provenance"
            ] = copy.deepcopy(self.resume_checkpoint_provenance)
        if self.resume_cpu_execution_contract_migration is not None:
            state["server_parallel"][
                "resume_cpu_execution_contract_migration"
            ] = dict(self.resume_cpu_execution_contract_migration)
        return state

    def save(self, path: Path, *, checkpoint_role: str | None = None):
        metadata = super().save(path, checkpoint_role=checkpoint_role)
        output = Path(path)
        payload = torch.load(output, map_location="cpu", weights_only=False)
        payload["metadata"].update(
            execution_backend=self.execution_backend,
            rollout_workers=self.rollout_workers,
            worker_assignment={
                str(item.worker_id): list(item.env_ids) for item in self.assignments
            },
        )
        if self.parallel_gaze:
            git = git_provenance()
            payload["metadata"].update({
                "training_contract": self.parallel_training_contract,
                "host_mode": self.host_mode,
                "worker_count": self.rollout_workers,
                "logical_env_count": self.n_rollout_threads,
                "n_rollout_envs": self.n_rollout_threads,
                "rollout_length": self.rollout_length,
                "master_seed": self.seed,
                "rng_scheme_version": PARALLEL_GAZE_RNG_CONTRACT,
                "logprob_audit_version": (
                    PARALLEL_GAZE_LOGPROB_AUDIT_VERSION
                ),
                "worker_torch_intra_op_threads": 1,
                "worker_torch_inter_op_threads": 1,
                "main_torch_intra_op_threads": (
                    self._main_cpu_execution["torch_cpu_intra_op_threads"]
                ),
                "main_torch_inter_op_threads": (
                    self._main_cpu_execution["torch_cpu_inter_op_threads"]
                ),
                "gaze_cpu_inference_dtype": (
                    PARALLEL_GAZE_CPU_INFERENCE_DTYPE
                ),
                "gaze_cpu_inference_path": (
                    PARALLEL_GAZE_CPU_INFERENCE_PATH
                ),
                "cpu_execution_settings_matched": True,
                "action_checkpoint_sha256": self.source_provenance[
                    "source_content_sha256"
                ],
                "action_checkpoint_path": self.source_provenance[
                    "source_absolute_path"
                ],
                "action_checkpoint_parameter_sha256": parameter_hash(
                    self.policy.actor
                ),
                "action_checkpoint_critic_parameter_sha256": parameter_hash(
                    self.policy.critic
                ),
                "action_optimizer_present": (
                    self.policy.actor_optimizer is not None
                ),
                "action_requires_grad": any(
                    parameter.requires_grad
                    for parameter in self.policy.actor.parameters()
                ),
                "action_critic_optimizer_present": (
                    self.policy.critic_optimizer is not None
                ),
                "action_critic_requires_grad": any(
                    parameter.requires_grad
                    for parameter in self.policy.critic.parameters()
                ),
                "condition_name": self.condition_name,
                "memory_condition": self.memory_condition,
                "perception_condition": self.perception_condition,
                "initial_gaze_checkpoint": None,
                "gaze_initialization": {
                    "mode": "fresh_parallel_gaze_v1",
                    "initialized_from_serial_checkpoint": None,
                },
                "ppo_config": {
                    **self.ppo,
                    "entropy_coefs": list(self.entropy_coefs),
                    "entropy_coef_horizons": list(
                        self.entropy_coef_horizons
                    ),
                    "actor_lr": float(
                        self.gaze_optimizer.param_groups[0]["lr"]
                    ),
                    "critic_lr": float(
                        self.gaze_critic_optimizer.param_groups[0]["lr"]
                    ),
                },
                "code_git_commit": git["git_commit_hash"],
                "git_worktree_dirty": git["git_worktree_dirty"],
            })
        if self.resume_checkpoint_provenance is not None:
            payload["metadata"][
                "resume_partial_checkpoint_provenance"
            ] = copy.deepcopy(self.resume_checkpoint_provenance)
        torch.save(payload, output)
        return {
            **payload["metadata"],
            "checkpoint_sha256": file_sha256(output),
        }

    def _gaze_preupdate_audit(self, records) -> dict[str, float | str]:
        if not self.parallel_gaze:
            return super()._gaze_preupdate_audit(records)
        policy_versions = {
            int(record.get("policy_version", -1)) for record in records
        }
        if policy_versions != {int(self.update_count)}:
            raise RuntimeError(
                "parallel Gaze pre-update policy version mismatch | "
                f"records={sorted(policy_versions)} expected={self.update_count}"
            )
        main_execution = _require_parallel_gaze_main_cpu_execution()
        invalid_record_dtypes = [
            index
            for index, record in enumerate(records)
            if np.asarray(record["observation"]).dtype != np.float32
            or np.asarray(record["context"]).dtype != np.float32
        ]
        if invalid_record_dtypes:
            raise RuntimeError(
                "parallel Gaze pre-update record dtype mismatch | "
                f"indices={invalid_record_dtypes[:8]} expected=float32"
            )
        audit_actor = (
            copy.deepcopy(self.gaze_actor).cpu().eval().requires_grad_(False)
        )
        if any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in audit_actor.state_dict().values()
        ):
            raise RuntimeError("parallel Gaze CPU audit actor dtype is not float32")
        main_hash = state_dict_hash(self.gaze_actor.state_dict())
        audit_hash = state_dict_hash(audit_actor.state_dict())
        if audit_hash != main_hash:
            raise RuntimeError("parallel Gaze CPU audit actor hash mismatch")
        result = super()._gaze_preupdate_audit(
            records,
            actor=audit_actor,
            device=torch.device("cpu"),
        )
        result.update({
            "gaze_logprob_audit_version": (
                PARALLEL_GAZE_LOGPROB_AUDIT_VERSION
            ),
            "gaze_logprob_rollout_device": "cpu",
            "gaze_logprob_audit_device": "cpu",
            "gaze_logprob_training_device": str(self.device),
            "gaze_logprob_audit_actor_sha256": audit_hash,
            "gaze_logprob_policy_version": int(self.update_count),
            "gaze_logprob_main_cpu_intra_op_threads": main_execution[
                "torch_cpu_intra_op_threads"
            ],
            "gaze_logprob_main_cpu_inter_op_threads": main_execution[
                "torch_cpu_inter_op_threads"
            ],
            "gaze_logprob_worker_cpu_intra_op_threads": (
                PARALLEL_GAZE_CPU_INTRA_OP_THREADS
            ),
            "gaze_logprob_worker_cpu_inter_op_threads": (
                PARALLEL_GAZE_CPU_INTER_OP_THREADS
            ),
            "gaze_logprob_inference_dtype": (
                PARALLEL_GAZE_CPU_INFERENCE_DTYPE
            ),
            "gaze_logprob_inference_path": (
                PARALLEL_GAZE_CPU_INFERENCE_PATH
            ),
            "gaze_logprob_execution_settings_matched": True,
        })
        return result

    def run_update(self) -> dict[str, Any]:
        actor_before = parameter_hash(self.policy.actor)
        critic_before = parameter_hash(self.policy.critic)
        gaze_before = (
            state_dict_hash(self.gaze_actor.state_dict())
            if self.parallel_gaze else None
        )
        gaze_critic_before = (
            state_dict_hash(self.gaze_critic.state_dict())
            if self.parallel_gaze else None
        )
        freshness_before = self.freshness_audit.begin()
        reports_before = self._runtime_reports()
        gaze_reward_before = sum(
            float(row.get("gaze_total_attributed_optimization_reward", 0.0))
            for row in reports_before
        )
        gaze_boundaries_before = sum(
            int(row.get("gaze_total_env_boundaries", 0))
            for row in reports_before
        )
        bootstrap_count_before = sum(
            int(row.get("bootstrap_unowned_reward_count_total", 0))
            for row in reports_before
        )
        bootstrap_reward_before = sum(
            float(row.get("bootstrap_unowned_reward_sum_total", 0.0))
            for row in reports_before
        )
        self._partner_repository.assert_frozen(
            FIXED_HSP_PARTNER_ID, full_hash=True
        )
        self._phase("verify")
        gaze_sync_seconds = 0.0
        if self.parallel_gaze:
            actor_state = {
                key: value.detach().cpu().clone()
                for key, value in self.gaze_actor.state_dict().items()
            }
            critic_state = {
                key: value.detach().cpu().clone()
                for key, value in self.gaze_critic.state_dict().items()
            }
            sync_responses, gaze_sync_seconds, _sync_ipc = self._phase(
                "sync_gaze",
                actor_state,
                critic_state,
                gaze_before,
                gaze_critic_before,
                self.update_count,
            )
            sync_rows = [
                item
                for response in sync_responses
                for item in response.get("payload", ())
            ]
            if len(sync_rows) != self.rollout_workers or any(
                row.get("gaze_policy_version") != self.update_count
                or row.get("gaze_actor_sha256") != gaze_before
                or row.get("gaze_critic_sha256") != gaze_critic_before
                for row in sync_rows
            ):
                raise RuntimeError("parallel Gaze rollout weight sync mismatch")
        optimization = sparse = shaped = weighted = 0.0
        deliveries = episodes = 0
        partial_metric_totals = new_partial_metric_totals(
            self.scheduler.conditions
        )
        initialize_collision_totals(partial_metric_totals)
        collision_count = 0
        gaze_records: list[dict[str, Any]] = []
        started = time.perf_counter()
        actor_seconds = partner_seconds = critic_seconds = 0.0
        actionnet_rollout_forward_count = 0
        worker_collect_seconds = worker_commit_seconds = worker_finalize_seconds = 0.0
        ipc_seconds = 0.0
        self.last_rollout_trace = []
        self.policy.actor.eval()
        self.policy.critic.eval()
        rollout_available_actions = np.ones((1, 6), dtype=np.float32)
        for rollout_step in range(self.rollout_length):
            decision_conditions = tuple(self.conditions)
            step_trace: dict[str, object] = {
                "rollout_step": rollout_step,
                "conditions": [condition.__dict__ for condition in self.conditions],
            }
            global_before = self.completed_steps + rollout_step * self.n_rollout_threads
            factor = hsp_reward_shaping_factor(
                global_before, self.reward_shaping_horizon_steps
            )
            self._shared.clear_decision_metadata()
            collect_responses, collect_wall, collect_ipc = self._phase("collect", factor)
            worker_collect_seconds += collect_wall
            ipc_seconds += collect_ipc
            query_counts = self._views["acquisition_counts"].copy()
            if np.any(query_counts < 0) or np.any(
                query_counts > MAX_ACQUISITIONS_PER_DECISION
            ):
                raise RuntimeError("worker ActionNet query count exceeded validated bound")
            rows = [
                (env_id, acquisition)
                for env_id in range(self.n_rollout_threads)
                for acquisition in range(int(query_counts[env_id]))
            ]
            if len(rows) != int(query_counts.sum()):
                raise RuntimeError("flattened ActionNet query ordering mismatch")
            buffer_step = self.buffer.step
            pre_actor = self.buffer.rnn_states[buffer_step].copy()
            recurrent_masks = self.buffer.masks[buffer_step].copy()
            next_actor = pre_actor.copy()
            learner_actions = np.full(
                self.n_rollout_threads, STAY_ACTION_INDEX, dtype=np.int64
            )
            log_probabilities = np.zeros(self.n_rollout_threads, dtype=np.float32)
            if rows:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                actor_started = time.perf_counter()
                sampled_actions, sampled_logs, candidate_rows = (
                    _canonical_batch1_rollout_actor_outputs(
                        self.policy.actor,
                        self._views["acquisition_observations"],
                        rows,
                        pre_actor,
                        recurrent_masks,
                        rollout_available_actions,
                    )
                )
                actionnet_rollout_forward_count += len(rows)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                actor_seconds += time.perf_counter() - actor_started
                for row_index, (env_id, acquisition) in enumerate(rows):
                    if acquisition == int(query_counts[env_id]) - 1:
                        learner_actions[env_id] = sampled_actions[row_index]
                        log_probabilities[env_id] = sampled_logs[row_index]
                        next_actor[env_id] = candidate_rows[row_index]
            self._views["learner_actions"][:] = learner_actions

            next_critic = np.zeros_like(self.buffer.rnn_states_critic[buffer_step])
            values = np.zeros_like(self.buffer.value_preds[buffer_step])
            critic_started = time.perf_counter()
            with torch.no_grad():
                for env_id in range(self.n_rollout_threads):
                    value, candidate = self.policy.critic(
                        self._views["critic_observations"][env_id][None],
                        self.buffer.rnn_states_critic[buffer_step, env_id][None],
                        recurrent_masks[env_id][None],
                    )
                    values[env_id] = value.detach().cpu().numpy()[0]
                    next_critic[env_id] = candidate.detach().cpu().numpy()[0]
            critic_seconds += time.perf_counter() - critic_started

            partner_started = time.perf_counter()
            partner_actions = np.empty(self.n_rollout_threads, dtype=np.int64)
            for env_id, pool in enumerate(self._partner_pools):
                pool.assert_frozen()
                partner_actions[env_id] = pool.action(
                    self._views["partner_observations"][env_id],
                    self._views["partner_available_actions"][env_id],
                )
            partner_seconds += time.perf_counter() - partner_started
            self._views["partner_actions"][:] = partner_actions

            actor_active = query_counts > 0
            observations_for_ppo = self._views["final_observations"].copy()
            for env_id in np.flatnonzero(actor_active):
                observations_for_ppo[env_id] = self._views[
                    "acquisition_observations"
                ][env_id, query_counts[env_id] - 1]
            self.buffer.obs[buffer_step] = observations_for_ppo
            self.buffer.share_obs[buffer_step] = self._views[
                "critic_observations"
            ]
            self.buffer.available_actions[buffer_step].fill(1.0)
            self.buffer.active_masks[buffer_step, :, 0] = actor_active.astype(np.float32)
            self.buffer.actions[buffer_step, :, 0] = learner_actions
            self.buffer.action_log_probs[buffer_step, :, 0] = log_probabilities
            self.buffer.value_preds[buffer_step] = values
            self.buffer.transition_ids[buffer_step] = np.arange(
                rollout_step * self.n_rollout_threads,
                (rollout_step + 1) * self.n_rollout_threads,
            )
            for condition in self.conditions:
                self.condition_decision_counts[condition] += 1

            _commit, commit_wall, commit_ipc = self._phase("commit")
            worker_commit_seconds += commit_wall
            ipc_seconds += commit_ipc
            rewards = self._views["rewards"].copy()
            dones = self._views["dones"].astype(bool).copy()
            optimization += float(rewards[:, 0].sum())
            sparse += float(rewards[:, 1].sum())
            shaped += float(rewards[:, 2].sum())
            weighted += float(rewards[:, 3].sum())
            deliveries += int(self._views["deliveries"].sum())
            episodes += int(dones.sum())
            collision_count += accumulate_collision_totals(
                partial_metric_totals,
                conditions=decision_conditions,
                collisions=self._views["joint_collisions"],
            )
            acquisition_counts = (
                self._views["same_target_acquisition_counts"]
                + self._views["new_target_acquisition_counts"]
            )
            accumulate_partial_metrics(
                partial_metric_totals,
                conditions=decision_conditions,
                rewards=rewards,
                deliveries=self._views["deliveries"],
                dones=dones,
                acquisitions=acquisition_counts,
                actor_active=actor_active,
                actionnet_queries=query_counts,
                gaze_requests=self._views["gaze_request_counts"],
                same_target_acquisitions=self._views[
                    "same_target_acquisition_counts"
                ],
                new_target_acquisitions=self._views[
                    "new_target_acquisition_counts"
                ],
                valid_memory_tiles=self._views["valid_memory_tile_counts"],
                freshness_sample_counts=self._views["freshness_sample_counts"],
                freshness_normalized_sums=self._views[
                    "freshness_normalized_sums"
                ],
                saccade_distance_sums_degrees=self._views[
                    "saccade_distance_sums_degrees"
                ],
                require_acquisition_query_match=(
                    self.emma_timing.mode != "human_stochastic"
                    and self.emma_timing.mode not in VISUAL_ENCODING_EMMA_MODES
                ),
            )
            next_masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
            bad_masks = np.ones_like(next_masks)
            self._views["reset_flags"].fill(0)
            for env_id in np.flatnonzero(dones):
                next_condition = self.scheduler.next()
                self.conditions[int(env_id)] = next_condition
                self._views["reset_flags"][env_id] = 1
                self._views["reset_decision_times"][env_id] = next_condition.decision_time
                self._views["reset_horizons"][env_id] = next_condition.horizon
                self._views["reset_fixation_codes"][env_id] = FIXATION_TO_CODE[
                    next_condition.fixation_shape
                ]
                self._partner_pools[int(env_id)].load_next()
                next_actor[env_id].fill(0.0)
                next_critic[env_id].fill(0.0)
                next_masks[env_id].fill(0.0)
                bad_masks[env_id].fill(0.0)
            finalize_responses, finalize_wall, finalize_ipc = self._phase("finalize")
            worker_finalize_seconds += finalize_wall
            ipc_seconds += finalize_ipc
            if self.parallel_gaze:
                gaze_records.extend(merge_gaze_transitions(
                    item
                    for response in finalize_responses
                    for item in response.get("payload", ())
                ))
            self.buffer.insert(
                self._views["next_critic_observations"].copy(),
                self._views["next_observations"].copy(),
                next_actor,
                next_critic,
                self.buffer.actions[buffer_step].copy(),
                self.buffer.action_log_probs[buffer_step].copy(),
                self.buffer.value_preds[buffer_step].copy(),
                rewards[:, 0:1].astype(np.float32),
                next_masks,
                bad_masks=bad_masks,
                active_masks=np.ones_like(next_masks),
                available_actions=self._views["next_available_actions"].copy(),
            )
            step_trace.update({
                "acquisition_counts": query_counts.tolist(),
                "acquisition_timestamps": [
                    self._views[
                        "acquisition_timestamps"
                    ][env_id, :query_counts[env_id]].tolist()
                    for env_id in range(self.n_rollout_threads)
                ],
                "acquisition_observation_sha256": [
                    [
                        _array_hash(self._views["acquisition_observations"][env_id, index])
                        for index in range(int(query_counts[env_id]))
                    ]
                    for env_id in range(self.n_rollout_threads)
                ],
                "learner_actions": learner_actions.tolist(),
                "partner_actions": partner_actions.tolist(),
                "rewards": rewards.tolist(),
                "deliveries": self._views["deliveries"].tolist(),
                "dones": dones.tolist(),
                "actor_available_all_six": bool(np.all(
                    self.buffer.available_actions[buffer_step] == 1.0
                )),
                "worker_trace": sorted(
                    [
                        item
                        for response in collect_responses
                        for item in response.get("payload", ())
                    ],
                    key=lambda item: int(item["env_id"]),
                ) if self.server_trace else [],
            })
            self.last_rollout_trace.append(step_trace)

        if self.parallel_gaze:
            gaze_records = merge_gaze_transitions([
                {"gaze_records": gaze_records}
            ])

        collection_seconds = time.perf_counter() - started
        partial_metrics = finalize_partial_metrics(partial_metric_totals)
        finalize_collision_rates(partial_metrics)
        horizon_metrics = partial_metrics["horizon_metrics"]
        self._phase("verify")
        self._partner_repository.assert_frozen(
            FIXED_HSP_PARTNER_ID, full_hash=True
        )
        with torch.no_grad():
            next_value = self.policy.get_values(
                self.buffer.share_obs[-1],
                self.buffer.rnn_states_critic[-1],
                self.buffer.masks[-1],
            )
        self.buffer.compute_returns(
            next_value.detach().cpu().numpy(), self.trainer.value_normalizer
        )
        value_predictions = self.trainer.value_normalizer.denormalize(
            self.buffer.value_preds[:-1]
        )
        variance = explained_variance(value_predictions, self.buffer.returns[:-1])
        upcoming = self.completed_steps + self.rollout_length * self.n_rollout_threads
        audit_started = time.perf_counter()
        action_audit = self._preupdate_audit()
        gaze_audit = (
            self._gaze_preupdate_audit(gaze_records)
            if self.parallel_gaze else {}
        )
        audit_seconds = time.perf_counter() - audit_started
        update_started = time.perf_counter()
        if self.parallel_gaze:
            if self.train_action:
                if self.device.type == "cuda":
                    self.policy.actor.train()
                    self.policy.critic.train()
                action_update = self._action_ppo_update(upcoming)
            else:
                action_update = {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "dist_entropy": 0.0,
                    "actor_grad_norm": 0.0,
                    "critic_grad_norm": 0.0,
                    "approx_kl": 0.0,
                    "clip_fraction": 0.0,
                    "entropy_coef": entropy_coefficient(
                        upcoming,
                        self.entropy_coefs,
                        self.entropy_coef_horizons,
                    ),
                    "grad_clip_fraction": 0.0,
                }
            gaze_ppo = dict(self.ppo)
            gaze_ppo["entropy_coef"] = action_update["entropy_coef"]
            raw_gaze_update = gaze_ppo_update(
                self.gaze_actor,
                self.gaze_critic,
                self.gaze_optimizer,
                self.gaze_critic_optimizer,
                gaze_records,
                device=self.device,
                **gaze_ppo,
            )
            gaze_update = {
                "gaze_policy_loss": raw_gaze_update["policy_loss"],
                "gaze_value_loss": raw_gaze_update["value_loss"],
                "gaze_entropy": raw_gaze_update["gaze_entropy"],
                "gaze_approx_kl": raw_gaze_update["approx_kl"],
                "gaze_clip_fraction": raw_gaze_update["clip_fraction"],
                "gaze_actor_grad_norm": raw_gaze_update["actor_grad_norm"],
                "gaze_critic_grad_norm": raw_gaze_update["critic_grad_norm"],
                "gaze_weighted_entropy": raw_gaze_update[
                    "weighted_gaze_entropy"
                ],
                "gaze_weighted_approx_kl": raw_gaze_update[
                    "weighted_approx_kl"
                ],
                "gaze_weighted_clip_fraction": raw_gaze_update[
                    "weighted_clip_fraction"
                ],
                "gaze_records_per_task_decision_mean": raw_gaze_update[
                    "records_per_task_decision_mean"
                ],
                "gaze_records_per_task_decision_max": raw_gaze_update[
                    "records_per_task_decision_max"
                ],
                "gaze_unique_task_decisions": raw_gaze_update[
                    "unique_task_decisions"
                ],
                "gaze_decision_weight_sum_min": raw_gaze_update[
                    "decision_weight_sum_min"
                ],
                "gaze_decision_weight_sum_max": raw_gaze_update[
                    "decision_weight_sum_max"
                ],
                "gaze_decision_weight_sum_mean": raw_gaze_update[
                    "decision_weight_sum_mean"
                ],
                "gaze_weighted_advantage_mean": raw_gaze_update[
                    "weighted_advantage_mean"
                ],
                "gaze_weighted_advantage_std": raw_gaze_update[
                    "weighted_advantage_std"
                ],
                "gaze_unweighted_advantage_mean": raw_gaze_update[
                    "unweighted_advantage_mean"
                ],
                "gaze_unweighted_advantage_std": raw_gaze_update[
                    "unweighted_advantage_std"
                ],
                "gaze_explained_variance": raw_gaze_update[
                    "explained_variance"
                ],
                "gaze_return_mean": raw_gaze_update["return_mean"],
                "gaze_return_std": raw_gaze_update["return_std"],
                "gaze_value_mean": raw_gaze_update["value_mean"],
                "gaze_value_std": raw_gaze_update["value_std"],
                "gaze_critic_grad_clip_fraction": raw_gaze_update[
                    "critic_grad_clip_fraction"
                ],
                "gaze_env_boundary_count_in_records": raw_gaze_update[
                    "env_boundary_count_in_records"
                ],
            }
        else:
            if self.device.type == "cuda":
                self.policy.actor.train()
                self.policy.critic.train()
            action_update = self._action_ppo_update(upcoming)
            gaze_update = {}
        update_seconds = time.perf_counter() - update_started
        self._clear_after_update()
        self.completed_steps = upcoming
        self.update_count += 1
        gc.collect()
        actor_after = parameter_hash(self.policy.actor)
        critic_after = parameter_hash(self.policy.critic)
        gaze_after = (
            state_dict_hash(self.gaze_actor.state_dict())
            if self.parallel_gaze else None
        )
        gaze_critic_after = (
            state_dict_hash(self.gaze_critic.state_dict())
            if self.parallel_gaze else None
        )
        if self.parallel_gaze and not self.train_action and (
            actor_after != actor_before or critic_after != critic_before
        ):
            raise RuntimeError(
                "frozen ActionNet actor or critic changed during parallel Gaze PPO"
            )
        if not np.isclose(optimization, sparse + weighted, rtol=1e-10, atol=1e-6):
            raise RuntimeError("Partial reward identity failed")
        reports_after = self._runtime_reports()
        gaze_reward_after = sum(
            float(row.get("gaze_total_attributed_optimization_reward", 0.0))
            for row in reports_after
        )
        gaze_boundaries_after = sum(
            int(row.get("gaze_total_env_boundaries", 0))
            for row in reports_after
        )
        bootstrap_count_after = sum(
            int(row.get("bootstrap_unowned_reward_count_total", 0))
            for row in reports_after
        )
        bootstrap_reward_after = sum(
            float(row.get("bootstrap_unowned_reward_sum_total", 0.0))
            for row in reports_after
        )
        deltas = []
        branch_counts = {name: 0 for name in ("task", "stay", "stale", "random")}
        for before, after_row in zip(reports_before, reports_after):
            before_adapter = before["adapter"]
            after = after_row["adapter"]
            deltas.append({
                key: after[key] - before_adapter.get(key, 0)
                for key in (
                    "gaze_requests", "acquisitions", "same_target_acquisitions",
                    "new_target_acquisitions", "prep_count", "exec_count", "dwell_count",
                    "actionnet_queries", "action_proposals", "proposal_revisions",
                    "executed_proposals", "forced_stays", "no_acquisition_decisions",
                    "recurrent_commits", "held_foveal_refreshes",
                    "memory_refresh_count", "memory_expiry_count",
                    "peripheral_attempts", "peripheral_successes",
                    "peripheral_failures",
                    "peripheral_sample_events", "peripheral_rng_draw_count",
                    "peripheral_attempt_eccentricity_sum_degrees",
                    "peripheral_success_eccentricity_sum_degrees",
                )
            })
            for name, count in after.get("rule_branch_counts", {}).items():
                branch_counts[name] += int(count) - int(
                    before_adapter.get("rule_branch_counts", {}).get(name, 0)
                )
        aggregate = {key: sum(row[key] for row in deltas) for key in deltas[0]}
        memory_reports = [row["adapter"] for row in reports_after]
        decisions = self.rollout_length * self.n_rollout_threads
        if self.parallel_gaze:
            attributed_reward = gaze_reward_after - gaze_reward_before
            attributed_boundaries = gaze_boundaries_after - gaze_boundaries_before
            bootstrap_count = bootstrap_count_after - bootstrap_count_before
            bootstrap_reward = bootstrap_reward_after - bootstrap_reward_before
            if not np.isclose(
                attributed_reward + bootstrap_reward,
                optimization,
                rtol=1e-10,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "parallel Gaze reward ownership does not conserve reward"
                )
            if attributed_boundaries + bootstrap_count != decisions:
                raise RuntimeError(
                    "parallel Gaze transition ownership boundary mismatch: "
                    f"{attributed_boundaries} + {bootstrap_count} != {decisions}"
                )
        validate_partial_metric_totals(
            partial_metrics,
            optimization_reward=optimization,
            sparse_reward=sparse,
            deliveries=deliveries,
            episodes=episodes,
            decisions=decisions,
            forced_stays=aggregate["forced_stays"],
            acquisitions=aggregate["acquisitions"],
        )
        validate_collision_conservation(
            partial_metrics,
            collision_count=collision_count,
            decisions=decisions,
        )
        self.last_execution_timings = {
            "gaze_weight_sync_seconds": gaze_sync_seconds,
            "rollout_collection_seconds": collection_seconds,
            "worker_collect_seconds": worker_collect_seconds,
            "worker_collect_barrier_seconds": worker_collect_seconds,
            "actionnet_rollout_seconds": actor_seconds,
            "actionnet_rollout_forward_count": actionnet_rollout_forward_count,
            "worker_commit_barrier_seconds": worker_commit_seconds,
            "worker_commit_seconds": worker_commit_seconds,
            "worker_finalize_barrier_seconds": worker_finalize_seconds,
            "worker_finalize_seconds": worker_finalize_seconds,
            "actor_inference_seconds": actor_seconds,
            "critic_inference_seconds": critic_seconds,
            "partner_inference_seconds": partner_seconds,
            "partner_seconds": partner_seconds,
            "ipc_barrier_overhead_seconds": ipc_seconds,
            "preupdate_audit_seconds": audit_seconds,
            "ppo_update_seconds": update_seconds,
            "ppo_seconds": update_seconds,
        }
        gaze_trace = _gaze_record_trace(gaze_records) if self.parallel_gaze else []
        self.last_gaze_transition_trace = gaze_trace
        result = {
            "observation_mode": "partial", "training_stage": self.training_stage,
            "gaze_type": self.gaze_type,
            "train_components": list(self.train_components),
            "train_action": self.train_action, "train_gaze": self.train_gaze,
            "direct_joint_from_full": self.direct_joint_from_full,
            "horizon": self.scheduler.horizons[0] if len(self.scheduler.horizons) == 1 else None,
            "horizons": list(self.scheduler.horizons),
            "decision_time": self.scheduler.decision_times[0] if len(self.scheduler.decision_times) == 1 else None,
            "decision_times": list(self.scheduler.decision_times),
            "fixation_shape": self.scheduler.fixation_shapes[0] if len(self.scheduler.fixation_shapes) == 1 else None,
            "fixation_shapes": list(self.scheduler.fixation_shapes),
            "environment_steps": self.completed_steps, "global_step": self.completed_steps,
            "task_decisions": decisions, "updates": self.update_count,
            "update_count": self.update_count, "episodes": episodes,
            "reward": optimization, "optimization_reward": optimization,
            "sparse_team_reward": sparse, "native_shaped_reward": shaped,
            "weighted_shaped_reward": weighted, "deliveries": deliveries,
            "deliveries_per_episode": deliveries / max(episodes, 1),
            "collision_count": collision_count,
            "collision_rate": collision_count / decisions,
            "reward_identity_error": abs(optimization - sparse - weighted),
            "reward_shaping_factor": hsp_reward_shaping_factor(
                upcoming, self.reward_shaping_horizon_steps
            ),
            "shaping_factor": hsp_reward_shaping_factor(
                upcoming, self.reward_shaping_horizon_steps
            ),
            "reward_shaping_horizon_steps": self.reward_shaping_horizon_steps,
            "actionnet_online_forwards": aggregate["actionnet_queries"],
            "online_forwards_per_decision": aggregate["actionnet_queries"] / decisions,
            "acquisitions_per_decision": aggregate["acquisitions"] / decisions,
            "held_foveal_refreshes": aggregate["held_foveal_refreshes"],
            "held_foveal_refreshes_per_decision": (
                aggregate["held_foveal_refreshes"] / decisions
            ),
            "actor_active_sample_rate": aggregate["executed_proposals"] / decisions,
            "forced_stay_rate": aggregate["forced_stays"] / decisions,
            "gaze_transition_count": len(gaze_records),
            "gaze_transition_trace_sha256": (
                _json_sha256(gaze_trace) if self.parallel_gaze else None
            ),
            "gaze_sampled_actions_sha256": (
                _json_sha256([row["action"] for row in gaze_trace])
                if self.parallel_gaze else None
            ),
            "gaze_rollout_rewards_sha256": (
                _json_sha256([row["reward"] for row in gaze_trace])
                if self.parallel_gaze else None
            ),
            "gaze_ppo_inputs_sha256": (
                _json_sha256(gaze_trace) if self.parallel_gaze else None
            ),
            "active_partner": FIXED_HSP_PARTNER_ID,
            "active_partners": [FIXED_HSP_PARTNER_ID] * self.n_rollout_threads,
            "partner_step_guard_checks": sum(
                int(after["partner_step_guard_checks"])
                - int(before["partner_step_guard_checks"])
                for before, after in zip(reports_before, reports_after)
            ),
            "partner_step_guard_failures": sum(
                int(row["partner_step_guard_failures"]) for row in reports_after
            ),
            "partner_full_hash_checks": sum(
                int(after["partner_full_hash_checks"])
                - int(before["partner_full_hash_checks"])
                for before, after in zip(reports_before, reports_after)
            ),
            "partner_full_hash_failures": sum(
                int(row["partner_full_hash_failures"]) for row in reports_after
            ),
            "partner_optimizer_isolation_verified": True,
            "fixed_partner_identity": FIXED_HSP_PARTNER_ID,
            "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
            "fixed_partner_sampling_disabled": True,
            "actor_hash_before": actor_before, "actor_hash_after": actor_after,
            "critic_hash_before": critic_before,
            "critic_hash_after": critic_after,
            "action_actor_requires_grad": any(
                parameter.requires_grad for parameter in self.policy.actor.parameters()
            ),
            "action_critic_requires_grad": any(
                parameter.requires_grad for parameter in self.policy.critic.parameters()
            ),
            "action_actor_optimizer_present": self.policy.actor_optimizer is not None,
            "action_critic_optimizer_present": self.policy.critic_optimizer is not None,
            "action_optimizer_isolation_verified": (
                self.action_optimizer_isolation_verified
            ),
            "gaze_hash_before": gaze_before, "gaze_hash_after": gaze_after,
            "gaze_critic_hash_before": gaze_critic_before,
            "gaze_critic_hash_after": gaze_critic_after,
            "explained_variance": variance,
            "freshness_normalized_min": min(r["freshness_normalized_min"] for r in memory_reports),
            "freshness_normalized_max": max(r["freshness_normalized_max"] for r in memory_reports),
            "freshness_normalized_mean": float(np.mean([r["freshness_normalized_mean"] for r in memory_reports])),
            "freshness_normalized_variance": float(np.mean([r["freshness_normalized_variance"] for r in memory_reports])),
            "mean_freshness": float(np.mean([r["mean_freshness"] for r in memory_reports])),
            "valid_memory_fraction": float(np.mean([r["valid_memory_fraction"] for r in memory_reports])),
            "rule_branch_counts": branch_counts,
            "time_metrics": partial_metrics["time_metrics"],
            "horizon_metrics": horizon_metrics,
            "condition_metrics": partial_metrics["condition_metrics"],
            "shared_horizon_metrics": horizon_metrics,
            "shared_horizon_decision_counts": {
                label: int(bucket["task_decisions"])
                for label, bucket in horizon_metrics.items()
            },
            "shared_horizon_episode_counts": {
                label: int(bucket["episodes"])
                for label, bucket in horizon_metrics.items()
            },
            "condition_decision_counts": {
                f"t{condition.decision_time}_h{condition.horizon}_{condition.fixation_shape}": count
                for condition, count in self.condition_decision_counts.items()
            },
            "mean_saccade_distance_degrees": float(np.mean([
                report["mean_saccade_distance_degrees"] for report in memory_reports
            ])),
            "execution_backend": self.execution_backend,
            "training_contract": (
                self.parallel_training_contract
                if self.parallel_gaze else None
            ),
            "host_mode": self.host_mode,
            "rollout_workers": self.rollout_workers,
            "worker_count": self.rollout_workers,
            "logical_env_count": self.n_rollout_threads,
            "gaze_rng_scheme_version": (
                PARALLEL_GAZE_RNG_CONTRACT if self.parallel_gaze else None
            ),
            "gaze_worker_weight_sync_verified": self.parallel_gaze,
            "gaze_policy_version": (
                self.update_count - 1 if self.parallel_gaze else None
            ),
            "gaze_attributed_optimization_reward": (
                gaze_reward_after - gaze_reward_before
                if self.parallel_gaze else 0.0
            ),
            "gaze_reward_conservation_error": (
                abs(
                    optimization
                    - (gaze_reward_after - gaze_reward_before)
                    - (bootstrap_reward_after - bootstrap_reward_before)
                )
                if self.parallel_gaze else 0.0
            ),
            "gaze_attributed_env_boundaries": (
                gaze_boundaries_after - gaze_boundaries_before
                if self.parallel_gaze else 0
            ),
            "bootstrap_unowned_reward_count": (
                bootstrap_count_after - bootstrap_count_before
                if self.parallel_gaze else 0
            ),
            "bootstrap_unowned_reward_sum": (
                bootstrap_reward_after - bootstrap_reward_before
                if self.parallel_gaze else 0.0
            ),
            "gaze_policy_calls_total": sum(
                int(row.get("gaze_policy_calls_total", 0))
                for row in reports_after
            ),
            "gaze_transition_count_total": sum(
                int(row.get("gaze_transition_count_total", 0))
                for row in reports_after
            ),
            "bootstrap_unowned_reward_count_total": bootstrap_count_after,
            "bootstrap_unowned_reward_sum_total": bootstrap_reward_after,
            "worker_pids": list(self.worker_pids),
            **aggregate,
            **action_audit,
            **gaze_audit,
            **action_update,
            **gaze_update,
            **self.freshness_audit.report(freshness_before),
            **self.last_execution_timings,
            **cooking_contract(),
        }
        if self.server_trace and self.parallel_gaze:
            result["gaze_transition_trace"] = gaze_trace
        result.update({
            "peripheral_success_count": aggregate["peripheral_successes"],
            "peripheral_success_rate": (
                aggregate["peripheral_successes"]
                / max(aggregate["peripheral_attempts"], 1)
            ),
            "peripheral_successes_per_acquisition": (
                aggregate["peripheral_successes"]
                / max(aggregate["acquisitions"], 1)
            ),
            "peripheral_successes_per_decision": (
                aggregate["peripheral_successes"] / max(decisions, 1)
            ),
            "mean_peripheral_attempt_eccentricity_deg": (
                aggregate["peripheral_attempt_eccentricity_sum_degrees"]
                / max(aggregate["peripheral_attempts"], 1)
            ),
            "mean_peripheral_success_eccentricity_deg": (
                aggregate["peripheral_success_eccentricity_sum_degrees"]
                / max(aggregate["peripheral_successes"], 1)
            ),
        })
        return result

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._phase("shutdown")
        except Exception:
            pass
        for process in self._workers:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for commands in self._command_queues:
            commands.close()
        self._response_queue.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ParallelGazeTrainingSession(ServerParallelActionTrainingSession):
    """Host-neutral `parallel_gaze_v1` learned-Gaze training session."""

    def __init__(self, **kwargs) -> None:
        if kwargs.get("gaze_type") != "learned":
            raise ValueError("parallel_gaze_v1 requires learned gaze")
        if bool(kwargs.get("train_action")) or not bool(kwargs.get("train_gaze")):
            raise ValueError("parallel_gaze_v1 trains GazeNet only")
        super().__init__(**kwargs)
