"""One shared ActionNet/critic/MAPPO implementation for Full and Partial."""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from types import MethodType
from typing import Any

import numpy as np
import torch

from final_model.common.partners import parameter_hash
from final_model.common.zsceval_compat import install_zsceval, load_policy_config
from final_model.partial_ver.memory import OBSERVATION_CONTRACT_ID


install_zsceval()
from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from zsceval.algorithms.r_mappo.r_mappo import R_MAPPO
from zsceval.utils.separated_buffer import SeparatedReplayBuffer as _ZSCSeparatedReplayBuffer


ACTION_ARCHITECTURE_ID = "canonical_R_Actor_8x5x25_gru64_categorical6_v1"
FULL_OBSERVATION_CONTRACT = "full_state20_freshness255_displayed_preview4_v2"
PARTIAL_OBSERVATION_CONTRACT = OBSERVATION_CONTRACT_ID
CANONICAL_ENTROPY_COEFS = (0.2, 0.05, 0.01)
CANONICAL_ENTROPY_HORIZONS = (0, 10_000_000, 20_000_000)
GAZE_ENTROPY_COEFS = (0.05, 0.02, 0.01, 0.01)
GAZE_ENTROPY_HORIZONS = (0, 5_000_000, 10_000_000, 20_000_000)
PREUPDATE_LOGPROB_ATOL = 1e-5
ACTOR_FORWARD_CONTEXT_ID = "rollout_batch_microbatch_and_one_step_gru_v1"


def _time_env_flat(values: np.ndarray) -> np.ndarray:
    """Flatten canonical [time, env, ...] storage to deterministic env-major order."""
    axes = (1, 0, *range(2, values.ndim))
    transposed = values.transpose(axes)
    return transposed.reshape(-1, *values.shape[2:])


def _sequence_batch(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Return [time, chunk, ...] flattened in the order expected by RNNLayer."""
    chunks = [values[int(start) : int(start) + length] for start in starts]
    return np.stack(chunks, axis=1).reshape(-1, *values.shape[1:])


def _first_difference(left: np.ndarray, right: np.ndarray) -> tuple[int, ...] | None:
    different = np.argwhere(left != right)
    return None if not len(different) else tuple(int(value) for value in different[0])


def _observation_digest(observation: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(observation)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.digest()


def _digest_bytes_array(observation: np.ndarray) -> np.ndarray:
    return np.frombuffer(_observation_digest(observation), dtype=np.uint8).copy()


def _tensor_cpu_copy(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu").clone().numpy()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _three_way_tensor_report(
    actual: np.ndarray, captured: np.ndarray, buffered: np.ndarray
) -> dict[str, Any]:
    def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
        return {
            "exact_equal": bool(np.array_equal(left, right)),
            "max_abs_difference": float(absolute.max(initial=0.0)),
        }

    return {
        "actual_actor_argument_sha256": _array_sha256(actual),
        "captured_pre_forward_sha256": _array_sha256(captured),
        "buffer_sha256": _array_sha256(buffered),
        "actual_vs_captured": comparison(actual, captured),
        "actual_vs_buffer": comparison(actual, buffered),
        "captured_vs_buffer": comparison(captured, buffered),
    }


def install_rollout_context_stable_actor(
    actor: torch.nn.Module, *, microbatch_size: int
) -> None:
    """Make PPO actor execution use the same batch and one-step GRU context as rollout."""
    size = int(microbatch_size)
    if size <= 0:
        raise ValueError("actor forward microbatch size must be positive")
    if getattr(actor, "_final_model_forward_context_id", None) is not None:
        raise RuntimeError("actor forward context is already installed")

    original_base_forward = actor.base.forward

    def stable_base_forward(_module, observation):
        if observation.size(0) <= size:
            return original_base_forward(observation)
        return torch.cat(
            [
                original_base_forward(observation[start : start + size])
                for start in range(0, observation.size(0), size)
            ],
            dim=0,
        )

    original_rnn_forward = actor.rnn.forward

    def stable_rnn_forward(_module, features, hidden, masks):
        rows = int(features.size(0))
        sequences = int(hidden.size(0))
        if rows == sequences:
            if rows <= size:
                return original_rnn_forward(features, hidden, masks)
            outputs = []
            returned = []
            for start in range(0, rows, size):
                output, next_hidden = original_rnn_forward(
                    features[start : start + size],
                    hidden[start : start + size],
                    masks[start : start + size],
                )
                outputs.append(output)
                returned.append(next_hidden)
            return torch.cat(outputs, dim=0), torch.cat(returned, dim=0)

        if rows % sequences:
            raise RuntimeError("recurrent actor rows are not divisible by sequences")
        time_steps = rows // sequences
        time_features = features.view(time_steps, sequences, -1)
        time_masks = masks.view(time_steps, sequences, -1)
        output_blocks = []
        returned_blocks = []
        for start in range(0, sequences, size):
            end = min(start + size, sequences)
            block_hidden = hidden[start:end]
            block_outputs = []
            for time_index in range(time_steps):
                output, block_hidden = original_rnn_forward(
                    time_features[time_index, start:end],
                    block_hidden,
                    time_masks[time_index, start:end],
                )
                block_outputs.append(output)
            output_blocks.append(torch.stack(block_outputs, dim=0))
            returned_blocks.append(block_hidden)
        time_output = torch.cat(output_blocks, dim=1)
        return time_output.reshape(rows, -1), torch.cat(returned_blocks, dim=0)

    actor.base.forward = MethodType(stable_base_forward, actor.base)
    actor.rnn.forward = MethodType(stable_rnn_forward, actor.rnn)
    actor._final_model_forward_context_id = ACTOR_FORWARD_CONTEXT_ID
    actor._final_model_forward_microbatch_size = size


class ObservationParityError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(
            "PREUPDATE_POLICY_PARITY_FAILURE tensor=observation "
            + json.dumps(report, sort_keys=True, separators=(",", ":"))
        )


class ActionLogProbParityError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        location = report["failure"]
        super().__init__(
            "PREUPDATE_POLICY_PARITY_FAILURE tensor=action_log_prob "
            f"transition_id={location['transition_id']} "
            f"time={location['time']} env={location['env']} "
            f"path={location['path']} "
            + json.dumps(report, sort_keys=True, separators=(",", ":"))
        )


def _actor_pipeline_snapshot(
    actor: torch.nn.Module,
    observation: np.ndarray,
    rnn_state: np.ndarray,
    recurrent_mask: np.ndarray,
    legal_action_mask: np.ndarray,
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replay one original rollout batch and expose every deterministic actor stage."""
    with torch.no_grad():
        obs = torch.as_tensor(observation).to(**actor.tpdv)
        hidden = torch.as_tensor(rnn_state).to(**actor.tpdv)
        mask = torch.as_tensor(recurrent_mask).to(**actor.tpdv)
        available = torch.as_tensor(legal_action_mask).to(**actor.tpdv)
        selected_action = torch.as_tensor(action).to(**actor.tpdv)
        base_output = actor.base(obs)
        gru_input = base_output
        actor_features, returned_hidden = actor.rnn(base_output, hidden, mask)
        gru_output = actor_features
        if actor._use_influence_policy:
            mlp_obs = actor.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if actor._layer_after_N > 0:
            actor_features = actor.mlp_after(actor_features)
        raw_logits = actor.act.action_out.linear(actor_features)
        distribution = actor.act.action_out(actor_features, available)
        selected_log_prob = distribution.log_probs(selected_action)
    return {
        "observation": _tensor_cpu_copy(obs),
        "rnn_state": _tensor_cpu_copy(hidden),
        "recurrent_mask": _tensor_cpu_copy(mask),
        "legal_action_mask": _tensor_cpu_copy(available),
        "base_output": _tensor_cpu_copy(base_output),
        "gru_input": _tensor_cpu_copy(gru_input),
        "gru_output": _tensor_cpu_copy(gru_output),
        "returned_rnn_state": _tensor_cpu_copy(returned_hidden),
        "raw_logits": _tensor_cpu_copy(raw_logits),
        "distribution_logits": _tensor_cpu_copy(distribution.logits),
        "probabilities": _tensor_cpu_copy(distribution.probs),
        "action": _tensor_cpu_copy(selected_action),
        "action_log_prob": _tensor_cpu_copy(selected_log_prob),
    }


def _actor_selected_context_snapshot(
    actor: torch.nn.Module,
    observation: np.ndarray,
    rnn_state: np.ndarray,
    recurrent_mask: np.ndarray,
    legal_action_mask: np.ndarray,
    action: np.ndarray,
    selected_row: int,
) -> dict[str, np.ndarray]:
    """Expose one row while retaining the surrounding forward batch/sequence context."""
    row = int(selected_row)
    with torch.no_grad():
        obs = torch.as_tensor(observation).to(**actor.tpdv)
        hidden = torch.as_tensor(rnn_state).to(**actor.tpdv)
        mask = torch.as_tensor(recurrent_mask).to(**actor.tpdv)
        available = torch.as_tensor(legal_action_mask).to(**actor.tpdv)
        selected_action = torch.as_tensor(action).to(**actor.tpdv)
        value = actor.base._build_cnn_input(obs, actor.base.cnn_keys)
        stages = {"cnn_input": _tensor_cpu_copy(value[row])}
        for layer_index, layer in enumerate(actor.base.cnn):
            value = layer(value)
            stages[
                f"cnn_layer_{layer_index:02d}_{layer.__class__.__name__}"
            ] = _tensor_cpu_copy(value[row])
        base_output = value
        actor_features, _returned_hidden = actor.rnn(base_output, hidden, mask)
        stages["gru_output"] = _tensor_cpu_copy(actor_features[row])
        if actor._use_influence_policy:
            mlp_obs = actor.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if actor._layer_after_N > 0:
            actor_features = actor.mlp_after(actor_features)
        raw_logits = actor.act.action_out.linear(actor_features)
        distribution = actor.act.action_out(actor_features, available)
        selected_log_prob = distribution.log_probs(selected_action)
        stages["raw_logits"] = _tensor_cpu_copy(raw_logits[row])
        stages["distribution_logits"] = _tensor_cpu_copy(
            distribution.logits[row]
        )
        stages["probabilities"] = _tensor_cpu_copy(distribution.probs[row])
        stages["action_log_prob"] = _tensor_cpu_copy(selected_log_prob[row])
    return stages


class RolloutPolicyAuditCapture:
    """Immutable evidence of the exact actor inputs and outputs used during rollout."""

    def __init__(self, buffer: _ZSCSeparatedReplayBuffer):
        time_steps, environments = buffer.actions.shape[:2]
        self.time_steps = int(time_steps)
        self.environments = int(environments)
        self.transition_ids = np.arange(
            self.time_steps * self.environments, dtype=np.int64
        ).reshape(self.time_steps, self.environments)
        self.observations = np.empty_like(buffer.obs[:-1])
        # A uint8 plane is required: NumPy S32 scalars strip trailing null bytes.
        self.observation_sha256 = np.empty(
            (self.time_steps, self.environments, 32), dtype=np.uint8
        )
        self.rnn_states = np.empty_like(buffer.rnn_states[:-1])
        self.recurrent_masks = np.empty_like(buffer.masks[:-1])
        self.legal_action_masks = np.empty_like(buffer.available_actions[:-1])
        self.pre_forward_rnn_states = np.empty_like(buffer.rnn_states[:-1])
        self.pre_forward_recurrent_masks = np.empty_like(buffer.masks[:-1])
        self.pre_forward_legal_action_masks = np.empty_like(
            buffer.available_actions[:-1]
        )
        self.actions = np.empty_like(buffer.actions)
        self.old_action_log_probs = np.empty_like(buffer.action_log_probs)
        hidden_size = int(buffer.rnn_states.shape[-1])
        action_count = int(buffer.available_actions.shape[-1])
        feature_shape = (self.time_steps, self.environments, hidden_size)
        self.base_outputs = np.empty(feature_shape, dtype=np.float32)
        self.gru_inputs = np.empty(feature_shape, dtype=np.float32)
        self.gru_outputs = np.empty(feature_shape, dtype=np.float32)
        self.returned_rnn_states = np.empty_like(buffer.rnn_states[:-1])
        self.raw_action_logits = np.empty(
            (self.time_steps, self.environments, action_count), dtype=np.float32
        )
        self.distribution_logits = np.empty_like(self.raw_action_logits)
        self.action_probabilities = np.empty_like(self.raw_action_logits)
        self.actor_training_modes = np.empty(self.time_steps, dtype=np.bool_)
        self.actor_parameter_hashes: list[str | None] = [None] * self.time_steps
        self.rollout_start_actor_hash: str | None = None
        self.rollout_end_actor_hash: str | None = None
        self.preupdate_actor_hash: str | None = None
        self.episode_ids = np.empty(
            (self.time_steps, self.environments), dtype=np.int64
        )
        self.episode_steps = np.empty_like(self.episode_ids)
        self.done_before = np.empty(
            (self.time_steps, self.environments), dtype=np.bool_
        )
        self.done_after = np.empty_like(self.done_before)
        self.rnn_hidden_before_reset = np.empty_like(buffer.rnn_states[:-1])
        self.rnn_hidden_after_reset = np.empty_like(buffer.rnn_states[:-1])
        self.next_recurrent_masks = np.empty_like(buffer.masks[:-1])
        self._pre_recorded = np.zeros(self.time_steps, dtype=np.bool_)
        self._post_recorded = np.zeros(self.time_steps, dtype=np.bool_)
        self._actor_hook_recorded = np.zeros(self.time_steps, dtype=np.bool_)
        self._environment_recorded = np.zeros(self.time_steps, dtype=np.bool_)
        self._active_actor_step: int | None = None
        self._actor_hook_fields: set[str] = set()
        self._actor_hook_handles = []
        self.immediate_insert_checks = 0
        self.post_actor_forward_checks = 0
        self.source_buffer_alias_count = 0
        self.capture_buffer_alias_count = 0

    def attach_actor(self, actor: torch.nn.Module) -> None:
        if self._actor_hook_handles:
            raise RuntimeError("rollout actor audit hooks already attached")
        self.rollout_start_actor_hash = parameter_hash(actor)

        def active() -> int | None:
            return self._active_actor_step

        def base_pre(_module, arguments):
            step = active()
            if step is not None:
                self.observations[step] = _tensor_cpu_copy(arguments[0])
                self._actor_hook_fields.add("observation")

        def base_post(_module, _arguments, output):
            step = active()
            if step is not None:
                self.base_outputs[step] = _tensor_cpu_copy(output)
                self._actor_hook_fields.add("base_output")

        def rnn_pre(_module, arguments):
            step = active()
            if step is not None:
                self.gru_inputs[step] = _tensor_cpu_copy(arguments[0])
                self.rnn_states[step] = _tensor_cpu_copy(arguments[1])
                self.recurrent_masks[step] = _tensor_cpu_copy(arguments[2])
                self._actor_hook_fields.update(
                    ("gru_input", "rnn_state", "recurrent_mask")
                )

        def rnn_post(_module, _arguments, output):
            step = active()
            if step is not None:
                self.gru_outputs[step] = _tensor_cpu_copy(output[0])
                self.returned_rnn_states[step] = _tensor_cpu_copy(output[1])
                self._actor_hook_fields.update(("gru_output", "returned_rnn"))

        def action_out_pre(_module, arguments):
            step = active()
            if step is not None:
                available = arguments[1]
                if available is None:
                    self.legal_action_masks[step].fill(1.0)
                else:
                    self.legal_action_masks[step] = _tensor_cpu_copy(available)
                self._actor_hook_fields.add("legal_action_mask")

        def linear_post(_module, _arguments, output):
            step = active()
            if step is not None:
                self.raw_action_logits[step] = _tensor_cpu_copy(output)
                self._actor_hook_fields.add("raw_logits")

        def distribution_post(_module, _arguments, output):
            step = active()
            if step is not None:
                self.distribution_logits[step] = _tensor_cpu_copy(output.logits)
                self.action_probabilities[step] = _tensor_cpu_copy(output.probs)
                self._actor_hook_fields.update(("distribution_logits", "probabilities"))

        def actor_post(_module, _arguments, output):
            step = active()
            if step is not None:
                self.actions[step] = _tensor_cpu_copy(output[0])
                self.old_action_log_probs[step] = _tensor_cpu_copy(output[1])
                actual_returned = _tensor_cpu_copy(output[2])
                if not np.array_equal(actual_returned, self.returned_rnn_states[step]):
                    raise RuntimeError(
                        "PREUPDATE_POLICY_PARITY_FAILURE actor/RNN returned state mismatch"
                    )
                self._actor_hook_fields.update(("action", "action_log_prob"))

        self._actor_hook_handles = [
            actor.base.register_forward_pre_hook(base_pre),
            actor.base.register_forward_hook(base_post),
            actor.rnn.register_forward_pre_hook(rnn_pre),
            actor.rnn.register_forward_hook(rnn_post),
            actor.act.action_out.register_forward_pre_hook(action_out_pre),
            actor.act.action_out.linear.register_forward_hook(linear_post),
            actor.act.action_out.register_forward_hook(distribution_post),
            actor.register_forward_hook(actor_post),
        ]

    def begin_actor_forward(self, step: int, actor: torch.nn.Module) -> None:
        index = int(step)
        if self._active_actor_step is not None:
            raise RuntimeError("nested learner actor call during rollout audit")
        self._active_actor_step = index
        self._actor_hook_fields.clear()
        self.actor_training_modes[index] = bool(actor.training)
        self.actor_parameter_hashes[index] = parameter_hash(actor)

    def end_rollout(self, actor: torch.nn.Module) -> None:
        if self._active_actor_step is not None:
            raise RuntimeError("learner actor audit ended during active forward")
        self.rollout_end_actor_hash = parameter_hash(actor)
        for handle in self._actor_hook_handles:
            handle.remove()
        self._actor_hook_handles.clear()

    def _observation_failure_report(
        self,
        *,
        phase: str,
        time_index: int,
        env_index: int,
        captured: np.ndarray,
        current: np.ndarray,
        buffer: _ZSCSeparatedReplayBuffer,
        search_completed_rollout: bool,
    ) -> dict[str, Any]:
        difference_mask = captured != current
        coordinates = np.argwhere(difference_mask)
        absolute = np.abs(
            captured.astype(np.float64) - current.astype(np.float64)
        )
        coordinate_rows = [
            {
                "x": int(x),
                "y": int(y),
                "channel": int(channel),
                "captured_value": float(captured[x, y, channel]),
                "buffer_value": float(current[x, y, channel]),
            }
            for x, y, channel in coordinates[:30]
        ]
        channel_counts = {
            str(channel): int(np.sum(coordinates[:, 2] == channel))
            for channel in np.unique(coordinates[:, 2])
        }
        transition_id = int(self.transition_ids[time_index, env_index])
        capture_digest = _digest_bytes_array(captured)
        buffer_digest = _digest_bytes_array(current)
        report: dict[str, Any] = {
            "phase": phase,
            "transition_id": transition_id,
            "time": int(time_index),
            "env": int(env_index),
            "capture_sha256": bytes(capture_digest).hex(),
            "buffer_sha256": bytes(buffer_digest).hex(),
            "max_abs_difference": float(absolute.max()),
            "mean_abs_difference": float(absolute.mean()),
            "nonzero_difference_count": int(coordinates.shape[0]),
            "differing_coordinates_first_30": coordinate_rows,
            "differing_coordinate_rows_truncated": bool(coordinates.shape[0] > 30),
            "channel_difference_count": channel_counts,
            "captured_data_pointer": int(captured.__array_interface__["data"][0]),
            "buffer_data_pointer": int(current.__array_interface__["data"][0]),
            "captured_buffer_shares_memory": bool(
                np.shares_memory(captured, current)
            ),
        }
        if search_completed_rollout:
            buffer_hashes = np.empty_like(self.observation_sha256)
            for search_time in range(self.time_steps):
                for search_env in range(self.environments):
                    buffer_hashes[search_time, search_env] = _digest_bytes_array(
                        buffer.obs[search_time, search_env]
                    )

            captured_found = []
            for found_time, found_env in np.argwhere(
                np.all(buffer_hashes == capture_digest, axis=-1)
            ):
                if np.array_equal(
                    captured, buffer.obs[int(found_time), int(found_env)]
                ):
                    captured_found.append(
                        {
                            "time": int(found_time),
                            "env": int(found_env),
                            "transition_id": int(
                                self.transition_ids[int(found_time), int(found_env)]
                            ),
                        }
                    )

            buffer_found = []
            for found_time, found_env in np.argwhere(
                np.all(self.observation_sha256 == buffer_digest, axis=-1)
            ):
                if np.array_equal(
                    current, self.observations[int(found_time), int(found_env)]
                ):
                    buffer_found.append(
                        {
                            "time": int(found_time),
                            "env": int(found_env),
                            "transition_id": int(
                                self.transition_ids[int(found_time), int(found_env)]
                            ),
                        }
                    )
            report["lost_observation_search"] = {
                "expected": {
                    "time": int(time_index),
                    "env": int(env_index),
                    "transition_id": transition_id,
                },
                "captured_observation_found_in_buffer": captured_found,
                "buffer_observation_found_in_capture": buffer_found,
                "classification": (
                    "A_ORDERING_INDEXING_PERMUTATION"
                    if captured_found
                    else "B_NOT_FOUND_MUTATION_OVERWRITE_RECONSTRUCTION"
                ),
            }
        return report

    def record_pre_forward(
        self,
        step: int,
        source_observation: np.ndarray,
        buffer: _ZSCSeparatedReplayBuffer,
    ) -> None:
        index = int(step)
        if self._pre_recorded[index]:
            raise RuntimeError(f"rollout policy audit step {index} recorded twice")
        buffer.transition_ids[index] = self.transition_ids[index]
        actor_observation = buffer.obs[index]
        for env_index in range(self.environments):
            source = np.asarray(source_observation[env_index])
            actor_input = actor_observation[env_index]
            self.immediate_insert_checks += 1
            self.source_buffer_alias_count += int(
                np.shares_memory(source, actor_input)
            )
            source_digest = _digest_bytes_array(source)
            immediate_buffer_digest = _digest_bytes_array(actor_input)
            if not np.array_equal(source, actor_input) or not np.array_equal(
                source_digest, immediate_buffer_digest
            ):
                raise ObservationParityError(
                    self._observation_failure_report(
                        phase="immediate_after_buffer_insert",
                        time_index=index,
                        env_index=env_index,
                        captured=source,
                        current=actor_input,
                        buffer=buffer,
                        search_completed_rollout=False,
                    )
                )
            self.observation_sha256[index, env_index] = source_digest
        if self.source_buffer_alias_count:
            raise RuntimeError(
                "PREUPDATE_POLICY_PARITY_FAILURE tensor=observation_alias "
                f"source_buffer_alias_count={self.source_buffer_alias_count} "
                "capture_buffer_alias_count=not_yet_captured"
            )
        self.pre_forward_rnn_states[index] = buffer.rnn_states[index].copy()
        self.pre_forward_recurrent_masks[index] = buffer.masks[index].copy()
        self.pre_forward_legal_action_masks[index] = buffer.available_actions[
            index
        ].copy()
        self._pre_recorded[index] = True

    def record_environment_before(
        self, step: int, backends, buffer: _ZSCSeparatedReplayBuffer
    ) -> None:
        index = int(step)
        for env_index, backend in enumerate(backends):
            environment_steps = int(backend.environment_steps)
            episode_step = int(backend.episode_step)
            completed = environment_steps - episode_step
            self.episode_ids[index, env_index] = completed // int(
                backend.episode_length
            )
            self.episode_steps[index, env_index] = episode_step
            self.done_before[index, env_index] = bool(
                buffer.masks[index, env_index, 0] == 0.0
            )

    def record_environment_after(
        self,
        step: int,
        results,
        returned_next_hidden: np.ndarray,
        hidden_after_reset: np.ndarray,
        next_masks: np.ndarray,
    ) -> None:
        index = int(step)
        self.done_after[index] = np.asarray(
            [bool(result["done"]) for result in results], dtype=np.bool_
        )
        self.rnn_hidden_before_reset[index] = np.asarray(
            returned_next_hidden
        ).copy()
        self.rnn_hidden_after_reset[index] = np.asarray(hidden_after_reset).copy()
        self.next_recurrent_masks[index] = np.asarray(next_masks).copy()
        self._environment_recorded[index] = True

    def verify_post_actor_forward(
        self, step: int, buffer: _ZSCSeparatedReplayBuffer
    ) -> None:
        index = int(step)
        required = {
            "observation", "base_output", "gru_input", "rnn_state",
            "recurrent_mask", "gru_output", "returned_rnn", "legal_action_mask",
            "raw_logits", "distribution_logits", "probabilities", "action",
            "action_log_prob",
        }
        missing = required - self._actor_hook_fields
        if missing:
            raise RuntimeError(
                "PREUPDATE_POLICY_PARITY_FAILURE incomplete actual actor capture "
                f"step={index} missing={sorted(missing)}"
            )
        for env_index in range(self.environments):
            self.post_actor_forward_checks += 1
            captured = self.observations[index, env_index]
            current = buffer.obs[index, env_index]
            self.capture_buffer_alias_count += int(
                np.shares_memory(captured, current)
            )
            if not np.array_equal(captured, current):
                raise ObservationParityError(
                    self._observation_failure_report(
                        phase="immediately_after_actor_forward",
                        time_index=index,
                        env_index=env_index,
                        captured=captured,
                        current=current,
                        buffer=buffer,
                        search_completed_rollout=False,
                    )
                )
        if self.capture_buffer_alias_count:
            raise RuntimeError(
                "PREUPDATE_POLICY_PARITY_FAILURE tensor=observation_alias "
                f"capture_buffer_alias_count={self.capture_buffer_alias_count}"
            )
        self._actor_hook_recorded[index] = True
        self._active_actor_step = None

    def record_post_forward(
        self, step: int, actions: np.ndarray, old_action_log_probs: np.ndarray
    ) -> None:
        index = int(step)
        if not self._pre_recorded[index] or self._post_recorded[index]:
            raise RuntimeError(f"invalid rollout policy audit output at step {index}")
        if not np.array_equal(self.actions[index], np.asarray(actions)):
            raise RuntimeError(
                f"PREUPDATE_POLICY_PARITY_FAILURE tensor=actual_actor_action step={index}"
            )
        if not np.array_equal(
            self.old_action_log_probs[index], np.asarray(old_action_log_probs)
        ):
            raise RuntimeError(
                "PREUPDATE_POLICY_PARITY_FAILURE "
                f"tensor=actual_actor_action_log_prob step={index}"
            )
        self._post_recorded[index] = True

    def rnn_reset_trace(
        self,
        *,
        time_index: int,
        env_index: int,
        buffer: _ZSCSeparatedReplayBuffer,
    ) -> list[dict[str, Any]]:
        rows = []
        start = max(0, int(time_index) - 4)
        end = min(self.time_steps, int(time_index) + 3)
        for trace_time in range(start, end):
            actual = self.rnn_states[trace_time, env_index]
            captured = self.pre_forward_rnn_states[trace_time, env_index]
            buffered = buffer.rnn_states[trace_time, env_index]
            rows.append(
                {
                    "time": trace_time,
                    "transition_id": int(
                        self.transition_ids[trace_time, env_index]
                    ),
                    "episode_id": int(self.episode_ids[trace_time, env_index]),
                    "episode_step": int(
                        self.episode_steps[trace_time, env_index]
                    ),
                    "done_before": bool(self.done_before[trace_time, env_index]),
                    "done_after": bool(self.done_after[trace_time, env_index]),
                    "recurrent_mask": float(
                        self.recurrent_masks[trace_time, env_index, 0]
                    ),
                    "rnn_hidden_before_reset_sha256": _array_sha256(
                        self.rnn_hidden_before_reset[trace_time, env_index]
                    ),
                    "rnn_hidden_after_reset_sha256": _array_sha256(
                        self.rnn_hidden_after_reset[trace_time, env_index]
                    ),
                    "next_recurrent_mask": float(
                        self.next_recurrent_masks[trace_time, env_index, 0]
                    ),
                    "exact_hidden_argument_parity": _three_way_tensor_report(
                        actual, captured, buffered
                    ),
                    "returned_next_hidden_sha256": _array_sha256(
                        self.returned_rnn_states[trace_time, env_index]
                    ),
                    "returned_vs_before_reset_exact": bool(
                        np.array_equal(
                            self.returned_rnn_states[trace_time, env_index],
                            self.rnn_hidden_before_reset[trace_time, env_index],
                        )
                    ),
                }
            )
        return rows

    def action_logprob_failure_report(
        self,
        *,
        buffer: _ZSCSeparatedReplayBuffer,
        actor: torch.nn.Module,
        time_index: int,
        env_index: int,
        path: str,
        buffer_batch_log_probs: np.ndarray,
        flattened_log_probs: np.ndarray,
        sequence_log_probs: np.ndarray,
        sequence_transition_ids: np.ndarray,
        sequence_sample: tuple[np.ndarray, ...],
    ) -> dict[str, Any]:
        time_index = int(time_index)
        env_index = int(env_index)
        transition_id = int(self.transition_ids[time_index, env_index])
        replay = _actor_pipeline_snapshot(
            actor,
            self.observations[time_index],
            self.rnn_states[time_index],
            self.recurrent_masks[time_index],
            self.legal_action_masks[time_index],
            self.actions[time_index],
        )
        rollout_stages = {
            "observation": self.observations[time_index],
            "rnn_state": self.rnn_states[time_index],
            "recurrent_mask": self.recurrent_masks[time_index],
            "legal_action_mask": self.legal_action_masks[time_index],
            "base_output": self.base_outputs[time_index],
            "gru_input": self.gru_inputs[time_index],
            "gru_output": self.gru_outputs[time_index],
            "returned_rnn_state": self.returned_rnn_states[time_index],
            "raw_logits": self.raw_action_logits[time_index],
            "distribution_logits": self.distribution_logits[time_index],
            "probabilities": self.action_probabilities[time_index],
            "action": self.actions[time_index],
            "action_log_prob": self.old_action_log_probs[time_index],
        }
        stage_comparisons = {}
        first_divergence = None
        for name, rollout_value in rollout_stages.items():
            replay_value = replay[name]
            absolute = np.abs(
                rollout_value.astype(np.float64) - replay_value.astype(np.float64)
            )
            comparison = {
                "exact_equal": bool(np.array_equal(rollout_value, replay_value)),
                "max_abs_difference": float(absolute.max(initial=0.0)),
                "mean_abs_difference": float(absolute.mean()),
            }
            stage_comparisons[name] = comparison
            if first_divergence is None and not comparison["exact_equal"]:
                first_divergence = name

        sequence_rows = np.flatnonzero(sequence_transition_ids == transition_id)
        sequence_value = (
            None
            if not len(sequence_rows)
            else float(sequence_log_probs[int(sequence_rows[0]), 0])
        )
        assert len(sequence_rows) == 1
        sequence_row = int(sequence_rows[0])
        flattened_observation = buffer.obs[:-1].reshape(
            self.time_steps * self.environments, *buffer.obs.shape[2:]
        )
        flattened_rnn = buffer.rnn_states[:-1].reshape(
            self.time_steps * self.environments, *buffer.rnn_states.shape[2:]
        )
        flattened_actions = buffer.actions.reshape(
            self.time_steps * self.environments, *buffer.actions.shape[2:]
        )
        flattened_masks = buffer.masks[:-1].reshape(-1, 1)
        flattened_available = buffer.available_actions[:-1].reshape(
            self.time_steps * self.environments,
            *buffer.available_actions.shape[2:],
        )
        canonical_context = _actor_selected_context_snapshot(
            actor,
            self.observations[time_index],
            self.rnn_states[time_index],
            self.recurrent_masks[time_index],
            self.legal_action_masks[time_index],
            self.actions[time_index],
            env_index,
        )
        flattened_context = _actor_selected_context_snapshot(
            actor,
            flattened_observation,
            flattened_rnn,
            flattened_masks,
            flattened_available,
            flattened_actions,
            transition_id,
        )
        sequence_context = _actor_selected_context_snapshot(
            actor,
            sequence_sample[1],
            sequence_sample[2],
            sequence_sample[7],
            sequence_sample[11],
            sequence_sample[4],
            sequence_row,
        )
        context_comparisons = {}
        first_flattened_divergence = None
        first_sequence_divergence = None
        for stage_name, canonical_value in canonical_context.items():
            flattened_absolute = np.abs(
                canonical_value.astype(np.float64)
                - flattened_context[stage_name].astype(np.float64)
            )
            sequence_absolute = np.abs(
                canonical_value.astype(np.float64)
                - sequence_context[stage_name].astype(np.float64)
            )
            flattened_exact = bool(
                np.array_equal(canonical_value, flattened_context[stage_name])
            )
            sequence_exact = bool(
                np.array_equal(canonical_value, sequence_context[stage_name])
            )
            if first_flattened_divergence is None and not flattened_exact:
                first_flattened_divergence = stage_name
            if first_sequence_divergence is None and not sequence_exact:
                first_sequence_divergence = stage_name
            context_comparisons[stage_name] = {
                "canonical_vs_flattened_exact": flattened_exact,
                "canonical_vs_flattened_max_abs_difference": float(
                    flattened_absolute.max(initial=0.0)
                ),
                "canonical_vs_sequence_exact": sequence_exact,
                "canonical_vs_sequence_max_abs_difference": float(
                    sequence_absolute.max(initial=0.0)
                ),
            }
        old_value = float(self.old_action_log_probs[time_index, env_index, 0])
        capture_replay_value = float(replay["action_log_prob"][env_index, 0])
        buffer_batch_value = float(buffer_batch_log_probs[time_index, env_index, 0])
        flattened_value = float(flattened_log_probs[transition_id, 0])
        if (
            abs(capture_replay_value - old_value) <= PREUPDATE_LOGPROB_ATOL
            and abs(buffer_batch_value - old_value) <= PREUPDATE_LOGPROB_ATOL
            and abs(flattened_value - old_value) <= PREUPDATE_LOGPROB_ATOL
            and (sequence_value is None or abs(sequence_value - old_value) <= PREUPDATE_LOGPROB_ATOL)
        ):
            classification = "NO_DIVERGENCE"
        elif (
            abs(capture_replay_value - old_value) <= PREUPDATE_LOGPROB_ATOL
            and abs(buffer_batch_value - old_value) <= PREUPDATE_LOGPROB_ATOL
            and abs(flattened_value - old_value) > PREUPDATE_LOGPROB_ATOL
            and sequence_value is not None
            and abs(sequence_value - old_value) > PREUPDATE_LOGPROB_ATOL
        ):
            classification = "G_CUDA_FORWARD_BATCH_CONTEXT_MISMATCH"
        elif (
            abs(capture_replay_value - old_value) <= PREUPDATE_LOGPROB_ATOL
            and abs(buffer_batch_value - old_value) <= PREUPDATE_LOGPROB_ATOL
            and abs(flattened_value - old_value) > PREUPDATE_LOGPROB_ATOL
        ):
            classification = "F_ANOTHER_AUDIT_BUG_FLATTENED_FORWARD_CONTEXT"
        elif first_divergence in ("raw_logits", "distribution_logits", "probabilities"):
            classification = "D_ACTOR_FORWARD_MODE_NONDETERMINISM"
        elif (
            stage_comparisons["distribution_logits"]["exact_equal"]
            and stage_comparisons["probabilities"]["exact_equal"]
            and not stage_comparisons["action_log_prob"]["exact_equal"]
        ):
            classification = "E_ACTION_LOGPROB_DISTRIBUTION_SEMANTICS"
        elif not _three_way_tensor_report(
            self.rnn_states[time_index, env_index],
            self.pre_forward_rnn_states[time_index, env_index],
            buffer.rnn_states[time_index, env_index],
        )["actual_vs_buffer"]["exact_equal"]:
            classification = "C_BUFFER_RNN_STATE_STORAGE_MISMATCH"
        else:
            classification = "G_UNCLASSIFIED_FORWARD_DIVERGENCE"
        return {
            "failure": {
                "transition_id": transition_id,
                "time": time_index,
                "env": env_index,
                "path": path,
                "old_action_log_prob": old_value,
                "capture_direct_action_log_prob": capture_replay_value,
                "buffer_batch_action_log_prob": buffer_batch_value,
                "flattened_buffer_action_log_prob": flattened_value,
                "sequence_action_log_prob": sequence_value,
            },
            "classification": classification,
            "first_pipeline_divergence": first_divergence,
            "actor_context": {
                "rollout_start_parameter_hash": self.rollout_start_actor_hash,
                "transition_parameter_hash": self.actor_parameter_hashes[time_index],
                "rollout_end_parameter_hash": self.rollout_end_actor_hash,
                "preupdate_parameter_hash": self.preupdate_actor_hash,
                "rollout_actor_training_mode": bool(
                    self.actor_training_modes[time_index]
                ),
                "replay_actor_training_mode": bool(actor.training),
            },
            "actual_rnn_argument_parity": _three_way_tensor_report(
                self.rnn_states[time_index, env_index],
                self.pre_forward_rnn_states[time_index, env_index],
                buffer.rnn_states[time_index, env_index],
            ),
            "actual_observation_argument_parity": _three_way_tensor_report(
                self.observations[time_index, env_index],
                self.observations[time_index, env_index],
                buffer.obs[time_index, env_index],
            ),
            "actual_recurrent_mask_argument_parity": _three_way_tensor_report(
                self.recurrent_masks[time_index, env_index],
                self.pre_forward_recurrent_masks[time_index, env_index],
                buffer.masks[time_index, env_index],
            ),
            "actual_legal_mask_argument_parity": _three_way_tensor_report(
                self.legal_action_masks[time_index, env_index],
                self.pre_forward_legal_action_masks[time_index, env_index],
                buffer.available_actions[time_index, env_index],
            ),
            "pipeline_stage_comparisons": stage_comparisons,
            "execution_context_pipeline": {
                "canonical_batch_size": self.environments,
                "flattened_batch_size": self.time_steps * self.environments,
                "sequence_chunk_batch_size": int(sequence_sample[2].shape[0]),
                "first_flattened_divergence": first_flattened_divergence,
                "first_sequence_divergence": first_sequence_divergence,
                "stage_comparisons": context_comparisons,
            },
            "env_rnn_reset_trace": self.rnn_reset_trace(
                time_index=time_index, env_index=env_index, buffer=buffer
            ),
        }

    def verify_buffer(
        self, buffer: _ZSCSeparatedReplayBuffer, *, mode: str, horizon
    ) -> dict[str, float]:
        if (
            not self._pre_recorded.all()
            or not self._post_recorded.all()
            or not self._actor_hook_recorded.all()
            or not self._environment_recorded.all()
        ):
            raise RuntimeError("PREUPDATE_POLICY_PARITY_FAILURE incomplete rollout capture")
        if self._actor_hook_handles:
            raise RuntimeError("PREUPDATE_POLICY_PARITY_FAILURE actor hooks remain attached")
        actor_hashes = {
            self.rollout_start_actor_hash,
            self.rollout_end_actor_hash,
            self.preupdate_actor_hash,
            *self.actor_parameter_hashes,
        }
        if None in actor_hashes or len(actor_hashes) != 1:
            raise RuntimeError(
                "PREUPDATE_POLICY_PARITY_FAILURE tensor=actor_parameters "
                f"start={self.rollout_start_actor_hash} "
                f"end={self.rollout_end_actor_hash} "
                f"preupdate={self.preupdate_actor_hash}"
            )
        if bool(self.actor_training_modes.any()):
            first_training_step = int(np.argwhere(self.actor_training_modes)[0, 0])
            raise RuntimeError(
                "PREUPDATE_POLICY_PARITY_FAILURE tensor=actor_mode "
                f"time={first_training_step} mode=train"
            )
        if not np.array_equal(self.base_outputs, self.gru_inputs):
            index = _first_difference(self.base_outputs, self.gru_inputs)
            assert index is not None
            time_index, env_index = index[:2]
            raise RuntimeError(
                "PREUPDATE_POLICY_PARITY_FAILURE tensor=base_to_gru_argument "
                f"transition_id={self.transition_ids[time_index, env_index]} "
                f"time={time_index} env={env_index}"
            )
        expected_ids = np.arange(
            self.time_steps * self.environments, dtype=np.int64
        ).reshape(self.time_steps, self.environments)
        if not np.array_equal(self.transition_ids, expected_ids):
            raise RuntimeError("PREUPDATE_POLICY_PARITY_FAILURE transition IDs changed")

        if not np.array_equal(buffer.transition_ids, expected_ids):
            index = _first_difference(buffer.transition_ids, expected_ids)
            assert index is not None
            time_index, env_index = index
            raise RuntimeError(
                "PREUPDATE_POLICY_PARITY_FAILURE tensor=transition_id "
                f"time={time_index} env={env_index} "
                f"buffer={buffer.transition_ids[index]} expected={expected_ids[index]}"
            )
        if not np.array_equal(self.observations, buffer.obs[:-1]):
            differing_transitions = np.argwhere(
                np.any(self.observations != buffer.obs[:-1], axis=(2, 3, 4))
            )
            time_index, env_index = (
                int(differing_transitions[0, 0]),
                int(differing_transitions[0, 1]),
            )
            raise ObservationParityError(
                self._observation_failure_report(
                    phase="completed_rollout_before_optimizer",
                    time_index=time_index,
                    env_index=env_index,
                    captured=self.observations[time_index, env_index],
                    current=buffer.obs[time_index, env_index],
                    buffer=buffer,
                    search_completed_rollout=True,
                )
            )

        for time_index in range(self.time_steps):
            for env_index in range(self.environments):
                if not np.array_equal(
                    self.observation_sha256[time_index, env_index],
                    _digest_bytes_array(self.observations[time_index, env_index]),
                ):
                    raise RuntimeError(
                        "PREUPDATE_POLICY_PARITY_FAILURE tensor=observation_digest "
                        f"transition_id={self.transition_ids[time_index, env_index]} "
                        f"time={time_index} env={env_index}"
                    )

        exact_tensors = {
            "action": (self.actions, buffer.actions),
            "old_action_log_prob": (
                self.old_action_log_probs,
                buffer.action_log_probs,
            ),
            "rnn_state_before_action": (self.rnn_states, buffer.rnn_states[:-1]),
            "recurrent_mask": (self.recurrent_masks, buffer.masks[:-1]),
            "legal_action_mask": (
                self.legal_action_masks,
                buffer.available_actions[:-1],
            ),
        }
        for name, (rollout_value, buffer_value) in exact_tensors.items():
            if not np.array_equal(rollout_value, buffer_value):
                index = _first_difference(rollout_value, buffer_value)
                assert index is not None
                time_index, env_index = index[:2]
                transition_id = int(self.transition_ids[time_index, env_index])
                raise RuntimeError(
                    f"PREUPDATE_POLICY_PARITY_FAILURE tensor={name} "
                    f"transition_id={transition_id} time={time_index} env={env_index} "
                    f"rollout={rollout_value[index]!r} buffer={buffer_value[index]!r}"
                )

        session_capture_tensors = {
            "session_pre_forward_rnn_state": (
                self.pre_forward_rnn_states,
                self.rnn_states,
            ),
            "session_pre_forward_recurrent_mask": (
                self.pre_forward_recurrent_masks,
                self.recurrent_masks,
            ),
            "session_pre_forward_legal_action_mask": (
                self.pre_forward_legal_action_masks,
                self.legal_action_masks,
            ),
        }
        for name, (session_value, actual_actor_argument) in session_capture_tensors.items():
            if not np.array_equal(session_value, actual_actor_argument):
                index = _first_difference(session_value, actual_actor_argument)
                assert index is not None
                time_index, env_index = index[:2]
                raise RuntimeError(
                    f"PREUPDATE_POLICY_PARITY_FAILURE tensor={name} "
                    f"transition_id={self.transition_ids[time_index, env_index]} "
                    f"time={time_index} env={env_index}"
                )

        observation = buffer.obs[:-1]
        if mode == "full":
            if not np.all(observation[..., 20] == 255.0):
                raise RuntimeError(
                    "PREUPDATE_POLICY_PARITY_FAILURE Full channel20 is not exactly 255"
                )
            horizons = np.asarray(horizon)
            if horizons.ndim == 0:
                first_inactive = 21 + int(horizons)
                if first_inactive < 25 and not np.all(
                    observation[..., first_inactive:25] == 0.0
                ):
                    raise RuntimeError(
                        "PREUPDATE_POLICY_PARITY_FAILURE inactive Full preview channel is nonzero"
                    )
            elif horizons.shape != observation.shape[:2] or not np.isin(
                horizons, (0, 2, 4)
            ).all():
                raise RuntimeError("PREUPDATE_POLICY_PARITY_FAILURE invalid horizon rows")
            else:
                for time_index, env_index in np.ndindex(horizons.shape):
                    first_inactive = 21 + int(horizons[time_index, env_index])
                    if first_inactive < 25 and not np.all(
                        observation[time_index, env_index, ..., first_inactive:25] == 0.0
                    ):
                        raise RuntimeError(
                            "PREUPDATE_POLICY_PARITY_FAILURE inactive Full preview channel is nonzero"
                        )

        legal_difference = np.abs(
            self.legal_action_masks - buffer.available_actions[:-1]
        )
        return {
            "preupdate_observation_hash_match": 1.0,
            "preupdate_observation_exact_match": 1.0,
            "preupdate_immediate_insert_checks": float(
                self.immediate_insert_checks
            ),
            "preupdate_post_actor_forward_checks": float(
                self.post_actor_forward_checks
            ),
            "preupdate_source_buffer_alias_count": float(
                self.source_buffer_alias_count
            ),
            "preupdate_capture_buffer_alias_count": float(
                self.capture_buffer_alias_count
            ),
            "preupdate_transition_id_exact_match": 1.0,
            "preupdate_actor_parameter_immutable": 1.0,
            "preupdate_actor_eval_mode_exact_match": 1.0,
            "preupdate_base_to_gru_exact_match": 1.0,
            "preupdate_action_exact_match": 1.0,
            "preupdate_rnn_state_exact_match": 1.0,
            "preupdate_recurrent_mask_exact_match": 1.0,
            "preupdate_legal_mask_exact_match": 1.0,
            "preupdate_max_legal_mask_difference": float(legal_difference.max()),
        }


def _recurrent_sample(
    buffer: _ZSCSeparatedReplayBuffer,
    advantages: np.ndarray,
    starts: np.ndarray,
    data_chunk_length: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Build one correctly ordered recurrent sample plus deterministic transition IDs."""
    time_steps, environments = buffer.rewards.shape[:2]
    if time_steps % data_chunk_length:
        raise RuntimeError(
            "recurrent data_chunk_length must divide rollout_length without crossing envs"
        )
    arrays = {
        "share_obs": _time_env_flat(buffer.share_obs[:-1]),
        "obs": _time_env_flat(buffer.obs[:-1]),
        "actions": _time_env_flat(buffer.actions),
        "value_preds": _time_env_flat(buffer.value_preds[:-1]),
        "returns": _time_env_flat(buffer.returns[:-1]),
        "masks": _time_env_flat(buffer.masks[:-1]),
        "active_masks": _time_env_flat(buffer.active_masks[:-1]),
        "old_logs": _time_env_flat(buffer.action_log_probs),
        "advantages": _time_env_flat(advantages),
    }
    available = (
        None
        if buffer.available_actions is None
        else _time_env_flat(buffer.available_actions[:-1])
    )
    rnn_states = _time_env_flat(buffer.rnn_states[:-1])
    critic_rnn_states = _time_env_flat(buffer.rnn_states_critic[:-1])
    if not hasattr(buffer, "transition_ids") or buffer.transition_ids.shape != (
        time_steps,
        environments,
    ):
        raise RuntimeError(
            "PREUPDATE_POLICY_PARITY_FAILURE transition ID buffer missing or malformed"
        )
    if np.any(buffer.transition_ids < 0):
        index = tuple(int(value) for value in np.argwhere(buffer.transition_ids < 0)[0])
        raise RuntimeError(
            "PREUPDATE_POLICY_PARITY_FAILURE tensor=transition_id "
            f"time={index[0]} env={index[1]} buffer={buffer.transition_ids[index]}"
        )
    env_major_ids = _time_env_flat(buffer.transition_ids[..., None])

    def sequence(value: np.ndarray) -> np.ndarray:
        return _sequence_batch(value, starts, data_chunk_length)

    sample = (
        sequence(arrays["share_obs"]),
        sequence(arrays["obs"]),
        np.stack([rnn_states[int(start)] for start in starts]),
        np.stack([critic_rnn_states[int(start)] for start in starts]),
        sequence(arrays["actions"]),
        sequence(arrays["value_preds"]),
        sequence(arrays["returns"]),
        sequence(arrays["masks"]),
        sequence(arrays["active_masks"]),
        sequence(arrays["old_logs"]),
        sequence(arrays["advantages"]),
        None if available is None else sequence(available),
    )
    transition_ids = sequence(env_major_ids).reshape(-1)
    initial_transition_ids = env_major_ids[starts].reshape(-1)
    return sample, transition_ids, initial_transition_ids


class SeparatedReplayBuffer(_ZSCSeparatedReplayBuffer):
    """Final-model buffer with correct [time, chunk] recurrent minibatch ordering."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_ids = np.full(self.actions.shape[:2], -1, dtype=np.int64)

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        time_steps, environments = self.rewards.shape[:2]
        if time_steps % data_chunk_length:
            raise RuntimeError(
                "recurrent data_chunk_length must divide rollout_length without crossing envs"
            )
        data_chunks = time_steps * environments // data_chunk_length
        if data_chunks < num_mini_batch:
            raise RuntimeError("not enough recurrent chunks for PPO minibatches")
        chunks_per_batch = data_chunks // num_mini_batch
        if chunks_per_batch * num_mini_batch != data_chunks:
            raise RuntimeError("recurrent chunks must divide evenly across PPO minibatches")
        permutation = torch.randperm(data_chunks).numpy()
        for batch_index in range(num_mini_batch):
            starts = permutation[
                batch_index * chunks_per_batch : (batch_index + 1) * chunks_per_batch
            ] * data_chunk_length
            sample, _transition_ids, _initial_ids = _recurrent_sample(
                self, advantages, starts, data_chunk_length
            )
            yield sample


def ppo_health_metrics(ratio: torch.Tensor, clip_param: float):
    log_ratio = torch.log(ratio.clamp_min(torch.finfo(ratio.dtype).tiny))
    approx_kl = ((ratio - 1.0) - log_ratio)
    clipped = (torch.abs(ratio - 1.0) > float(clip_param)).float()
    return approx_kl, clipped


def _assert_recurrent_sample_order(
    buffer: _ZSCSeparatedReplayBuffer,
    sample: tuple[np.ndarray, ...],
    transition_ids: np.ndarray,
    initial_transition_ids: np.ndarray,
) -> None:
    """Fail at the first generator field whose deterministic transition ID differs."""
    time_steps, environments = buffer.actions.shape[:2]
    canonical = {
        "share_observation": buffer.share_obs[:-1].reshape(
            time_steps * environments, *buffer.share_obs.shape[2:]
        ),
        "observation": buffer.obs[:-1].reshape(
            time_steps * environments, *buffer.obs.shape[2:]
        ),
        "action": buffer.actions.reshape(
            time_steps * environments, *buffer.actions.shape[2:]
        ),
        "value_prediction": buffer.value_preds[:-1].reshape(-1, 1),
        "return": buffer.returns[:-1].reshape(-1, 1),
        "recurrent_mask": buffer.masks[:-1].reshape(-1, 1),
        "active_mask": buffer.active_masks[:-1].reshape(-1, 1),
        "old_action_log_prob": buffer.action_log_probs.reshape(
            time_steps * environments, *buffer.action_log_probs.shape[2:]
        ),
    }
    fields = {
        "share_observation": sample[0],
        "observation": sample[1],
        "action": sample[4],
        "value_prediction": sample[5],
        "return": sample[6],
        "recurrent_mask": sample[7],
        "active_mask": sample[8],
        "old_action_log_prob": sample[9],
    }
    if buffer.available_actions is not None:
        canonical["legal_action_mask"] = buffer.available_actions[:-1].reshape(
            time_steps * environments, *buffer.available_actions.shape[2:]
        )
        fields["legal_action_mask"] = sample[11]
    for name, actual in fields.items():
        expected = canonical[name][transition_ids]
        if not np.array_equal(actual, expected):
            index = _first_difference(actual, expected)
            assert index is not None
            row = index[0]
            transition_id = int(transition_ids[row])
            time_index, env_index = divmod(transition_id, environments)
            raise RuntimeError(
                f"PREUPDATE_POLICY_PARITY_FAILURE recurrent_generator_tensor={name} "
                f"transition_id={transition_id} time={time_index} env={env_index}"
            )

    actor_rnn = buffer.rnn_states[:-1].reshape(
        time_steps * environments, *buffer.rnn_states.shape[2:]
    )
    critic_rnn = buffer.rnn_states_critic[:-1].reshape(
        time_steps * environments, *buffer.rnn_states_critic.shape[2:]
    )
    for name, actual, canonical_state in (
        ("chunk_initial_actor_rnn", sample[2], actor_rnn),
        ("chunk_initial_critic_rnn", sample[3], critic_rnn),
    ):
        expected = canonical_state[initial_transition_ids]
        if not np.array_equal(actual, expected):
            index = _first_difference(actual, expected)
            assert index is not None
            transition_id = int(initial_transition_ids[index[0]])
            time_index, env_index = divmod(transition_id, environments)
            raise RuntimeError(
                f"PREUPDATE_POLICY_PARITY_FAILURE recurrent_generator_tensor={name} "
                f"transition_id={transition_id} time={time_index} env={env_index}"
            )


def _actor_log_probs(policy, obs, rnn_states, actions, masks, available, active):
    with torch.no_grad():
        log_probs, _entropy, _values = policy.actor.evaluate_actions(
            obs, rnn_states, actions, masks, available, active
        )
    return log_probs.detach().cpu().numpy()


def _direct_replay_indices(masks: np.ndarray) -> list[tuple[int, int]]:
    time_steps, environments = masks.shape[:2]
    env_indices = {0, environments // 2, environments - 1}
    selected = {
        (time_index, env_index)
        for time_index in (0, time_steps // 2, time_steps - 1)
        for env_index in env_indices
    }
    boundaries = np.argwhere(masks[..., 0] == 0.0)
    for time_index, env_index in boundaries[:6]:
        for adjacent in (int(time_index) - 1, int(time_index), int(time_index) + 1):
            if 0 <= adjacent < time_steps:
                selected.add((adjacent, int(env_index)))
    return sorted(selected)


def preupdate_policy_consistency_audit(
    *,
    policy,
    buffer: _ZSCSeparatedReplayBuffer,
    capture: RolloutPolicyAuditCapture,
    clip_param: float,
    data_chunk_length: int,
    mode: str,
    horizon,
    tolerance: float = PREUPDATE_LOGPROB_ATOL,
) -> dict[str, Any]:
    """Audit the completed rollout before backward or optimizer.step can run."""
    if policy.actor.training:
        raise RuntimeError(
            "PREUPDATE_POLICY_PARITY_FAILURE actor must remain in rollout/eval mode"
        )
    capture.preupdate_actor_hash = parameter_hash(policy.actor)
    metrics: dict[str, Any] = capture.verify_buffer(
        buffer, mode=mode, horizon=horizon
    )
    actor_hash_before = parameter_hash(policy.actor)
    time_steps, environments = buffer.actions.shape[:2]
    transition_count = time_steps * environments

    direct_obs = buffer.obs[:-1].reshape(
        transition_count, *buffer.obs.shape[2:]
    )
    direct_rnn = buffer.rnn_states[:-1].reshape(
        transition_count, *buffer.rnn_states.shape[2:]
    )
    direct_actions = buffer.actions.reshape(
        transition_count, *buffer.actions.shape[2:]
    )
    direct_masks = buffer.masks[:-1].reshape(transition_count, 1)
    direct_available = buffer.available_actions[:-1].reshape(
        transition_count, *buffer.available_actions.shape[2:]
    )
    direct_active = buffer.active_masks[:-1].reshape(transition_count, 1)
    old_direct = buffer.action_log_probs.reshape(
        transition_count, *buffer.action_log_probs.shape[2:]
    )
    flattened_direct = _actor_log_probs(
        policy,
        direct_obs,
        direct_rnn,
        direct_actions,
        direct_masks,
        direct_available,
        direct_active,
    )
    flattened_difference = np.abs(flattened_direct - old_direct)

    buffer_batch_log_probs = np.empty_like(buffer.action_log_probs)
    for time_index in range(time_steps):
        buffer_batch_log_probs[time_index] = _actor_log_probs(
            policy,
            buffer.obs[time_index],
            buffer.rnn_states[time_index],
            buffer.actions[time_index],
            buffer.masks[time_index],
            buffer.available_actions[time_index],
            buffer.active_masks[time_index],
        )
    buffer_batch_difference = np.abs(
        buffer_batch_log_probs - buffer.action_log_probs
    )

    data_chunks = transition_count // int(data_chunk_length)
    starts = np.arange(data_chunks, dtype=np.int64) * int(data_chunk_length)
    dummy_advantages = np.zeros_like(buffer.rewards)
    sample, transition_ids, initial_ids = _recurrent_sample(
        buffer, dummy_advantages, starts, int(data_chunk_length)
    )
    _assert_recurrent_sample_order(
        buffer, sample, transition_ids, initial_ids
    )
    new_sequence = _actor_log_probs(
        policy,
        sample[1],
        sample[2],
        sample[4],
        sample[7],
        sample[11],
        sample[8],
    )
    old_sequence = sample[9]
    sequence_difference = np.abs(new_sequence - old_sequence)
    log_ratio = new_sequence.astype(np.float64) - old_sequence.astype(np.float64)
    ratio = np.exp(log_ratio)
    approx_kl = (ratio - 1.0) - log_ratio
    lower = ratio < 1.0 - float(clip_param)
    upper = ratio > 1.0 + float(clip_param)

    flattened_worst_transition = int(np.argmax(flattened_difference.reshape(-1)))
    diagnostic_time, diagnostic_env = divmod(
        flattened_worst_transition, environments
    )
    capture_replay = _actor_pipeline_snapshot(
        policy.actor,
        capture.observations[diagnostic_time],
        capture.rnn_states[diagnostic_time],
        capture.recurrent_masks[diagnostic_time],
        capture.legal_action_masks[diagnostic_time],
        capture.actions[diagnostic_time],
    )
    capture_replay_difference = np.abs(
        capture_replay["action_log_prob"]
        - capture.old_action_log_probs[diagnostic_time]
    )

    replay_ids: list[int] = []
    replay_max_difference = 0.0
    replay_worst: tuple[int, int, float, float] | None = None
    for time_index, env_index in _direct_replay_indices(buffer.masks[:-1]):
        replay = _actor_log_probs(
            policy,
            buffer.obs[time_index, env_index][None],
            buffer.rnn_states[time_index, env_index][None],
            buffer.actions[time_index, env_index][None],
            buffer.masks[time_index, env_index][None],
            buffer.available_actions[time_index, env_index][None],
            buffer.active_masks[time_index, env_index][None],
        )
        difference = float(
            np.max(
                np.abs(
                    replay - buffer.action_log_probs[time_index, env_index][None]
                )
            )
        )
        replay_max_difference = max(replay_max_difference, difference)
        if replay_worst is None or difference >= replay_worst[3]:
            replay_worst = (
                int(time_index),
                int(env_index),
                float(replay.reshape(-1)[0]),
                difference,
            )
        replay_ids.append(int(capture.transition_ids[time_index, env_index]))

    canonical_maximum_difference = max(
        float(buffer_batch_difference.max()),
        float(sequence_difference.max()),
        float(capture_replay_difference.max()),
    )
    if (
        float(flattened_difference.max()) > float(tolerance)
        or canonical_maximum_difference > float(tolerance)
    ):
        diagnostic_report = capture.action_logprob_failure_report(
            buffer=buffer,
            actor=policy.actor,
            time_index=diagnostic_time,
            env_index=diagnostic_env,
            path="flattened_stored_pre_forward_state_diagnostic",
            buffer_batch_log_probs=buffer_batch_log_probs,
            flattened_log_probs=flattened_direct,
            sequence_log_probs=new_sequence,
            sequence_transition_ids=transition_ids,
            sequence_sample=sample,
        )
    else:
        diagnostic_report = {
            "classification": "NO_DIVERGENCE",
            "actor_forward_context": ACTOR_FORWARD_CONTEXT_ID,
            "actor_forward_microbatch_size": int(
                policy.actor._final_model_forward_microbatch_size
            ),
        }
    if (
        canonical_maximum_difference > float(tolerance)
        or bool(lower.any())
        or bool(upper.any())
    ):
        buffer_batch_max = float(buffer_batch_difference.max())
        sequence_max = float(sequence_difference.max())
        capture_max = float(capture_replay_difference.max())
        if buffer_batch_max >= sequence_max and buffer_batch_max >= capture_max:
            worst_row = int(np.argmax(buffer_batch_difference.reshape(-1)))
            transition_id = worst_row
            time_index, env_index = divmod(transition_id, environments)
            divergence_path = "buffer_timestep_batch_direct_replay"
        elif sequence_max >= capture_max:
            worst_row = int(np.argmax(sequence_difference.reshape(-1)))
            transition_id = int(transition_ids[worst_row])
            time_index, env_index = divmod(transition_id, environments)
            divergence_path = "recurrent_minibatch_sequence"
        else:
            time_index = diagnostic_time
            env_index = int(np.argmax(capture_replay_difference.reshape(-1)))
            transition_id = int(capture.transition_ids[time_index, env_index])
            divergence_path = "capture_timestep_batch_direct_replay"
        raise ActionLogProbParityError(
            capture.action_logprob_failure_report(
                buffer=buffer,
                actor=policy.actor,
                time_index=time_index,
                env_index=env_index,
                path=divergence_path,
                buffer_batch_log_probs=buffer_batch_log_probs,
                flattened_log_probs=flattened_direct,
                sequence_log_probs=new_sequence,
                sequence_transition_ids=transition_ids,
                sequence_sample=sample,
            )
        )
    if parameter_hash(policy.actor) != actor_hash_before:
        raise RuntimeError(
            "PREUPDATE_POLICY_PARITY_FAILURE actor parameters changed during audit"
        )

    quantiles = np.quantile(ratio, (0.01, 0.05, 0.50, 0.95, 0.99))
    metrics.update(
        {
            "preupdate_parity_pass": 1.0,
            "preupdate_transition_count": int(transition_count),
            "preupdate_max_abs_logprob_difference": canonical_maximum_difference,
            "preupdate_mean_abs_logprob_difference": float(
                sequence_difference.mean()
            ),
            "preupdate_direct_state_max_abs_logprob_difference": float(
                buffer_batch_difference.max()
            ),
            "preupdate_buffer_batch_max_abs_logprob_difference": float(
                buffer_batch_difference.max()
            ),
            "preupdate_capture_direct_max_abs_logprob_difference": float(
                capture_replay_difference.max()
            ),
            "preupdate_flattened_context_max_abs_logprob_difference": float(
                flattened_difference.max()
            ),
            "preupdate_flattened_context_transition_id": int(
                flattened_worst_transition
            ),
            "preupdate_flattened_context_is_canonical": 0.0,
            "preupdate_sequence_max_abs_logprob_difference": float(
                sequence_difference.max()
            ),
            "preupdate_replay_max_abs_logprob_difference": replay_max_difference,
            "preupdate_singleton_context_max_abs_logprob_difference": (
                replay_max_difference
            ),
            "preupdate_singleton_context_is_canonical": 0.0,
            "preupdate_ratio_min": float(ratio.min()),
            "preupdate_ratio_p01": float(quantiles[0]),
            "preupdate_ratio_p05": float(quantiles[1]),
            "preupdate_ratio_p50": float(quantiles[2]),
            "preupdate_ratio_p95": float(quantiles[3]),
            "preupdate_ratio_p99": float(quantiles[4]),
            "preupdate_ratio_max": float(ratio.max()),
            "preupdate_ratio_mean": float(ratio.mean()),
            "preupdate_approx_kl": float(approx_kl.mean()),
            "preupdate_clip_fraction": float(np.logical_or(lower, upper).mean()),
            "preupdate_lower_clip_fraction": float(lower.mean()),
            "preupdate_upper_clip_fraction": float(upper.mean()),
            "preupdate_direct_replay_transition_ids": replay_ids,
            "preupdate_diagnostic_classification": diagnostic_report[
                "classification"
            ],
            "preupdate_detailed_actor_context": diagnostic_report,
            "preupdate_logprob_tolerance": float(tolerance),
        }
    )
    return metrics


def entropy_coefficient(
    environment_steps: int,
    coefficients=CANONICAL_ENTROPY_COEFS,
    horizons=CANONICAL_ENTROPY_HORIZONS,
) -> float:
    """Piecewise-linear entropy schedule indexed only by global environment steps."""
    steps = int(environment_steps)
    if steps < 0:
        raise ValueError("environment_steps cannot be negative")
    if len(coefficients) != len(horizons) or not coefficients:
        raise ValueError("entropy coefficients and horizons must have equal nonzero length")
    for index in range(len(horizons) - 1):
        start, end = int(horizons[index]), int(horizons[index + 1])
        if start <= steps < end:
            fraction = (steps - start) / (end - start)
            return float(
                (1.0 - fraction) * coefficients[index]
                + fraction * coefficients[index + 1]
            )
    return float(coefficients[-1])


def explained_variance(value_predictions, returns) -> float:
    """Standard 1 - Var(return - prediction) / Var(return), finite on constants."""
    predicted = np.asarray(value_predictions, dtype=np.float64)
    target = np.asarray(returns, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError("value predictions and returns must have identical shapes")
    target_variance = float(np.var(target))
    if not np.isfinite(target_variance) or target_variance <= 1e-12:
        return 0.0
    value = 1.0 - float(np.var(target - predicted)) / target_variance
    return value if np.isfinite(value) else 0.0


def _box_with_channels(canonical_space: Any, channels: int):
    shape = (8, 5, int(channels))
    return canonical_space.__class__(
        low=np.zeros(shape, dtype=np.float32),
        high=np.full(shape, np.inf, dtype=np.float32),
        dtype=np.float32,
    )


def build_actionnet(*, seed: int, device: str | torch.device = "cpu") -> R_Actor:
    args, canonical, _share, action = load_policy_config("rnn")
    torch.manual_seed(int(seed))
    return R_Actor(args, _box_with_channels(canonical, 25), action, device=torch.device(device))


def training_args(
    *,
    rollout_length: int,
    n_rollout_threads: int = 1,
    actor_lr: float = 5e-4,
    critic_lr: float = 5e-4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float | None = None,
    entropy_coefs: tuple[float, ...] | list[float] = CANONICAL_ENTROPY_COEFS,
    entropy_coef_horizons: tuple[int, ...] | list[int] = CANONICAL_ENTROPY_HORIZONS,
    ppo_epochs: int = 15,
    num_mini_batches: int = 1,
    max_grad_norm: float = 10.0,
):
    source, _obs, _share, _action = load_policy_config("rnn")
    args = copy.deepcopy(source)
    args.episode_length = int(rollout_length)
    args.n_rollout_threads = int(n_rollout_threads)
    if args.n_rollout_threads <= 0:
        raise ValueError("n_rollout_threads must be positive")
    args.lr = float(actor_lr)
    args.critic_lr = float(critic_lr)
    args.clip_param = float(clip_param)
    args.gamma = float(gamma)
    args.gae_lambda = float(gae_lambda)
    if entropy_coef is not None:
        args.entropy_coefs = [float(entropy_coef)]
        args.entropy_coef_horizons = [0]
    else:
        if len(entropy_coefs) != len(entropy_coef_horizons) or not entropy_coefs:
            raise ValueError("entropy schedule coefficients and horizons must have equal lengths")
        args.entropy_coefs = [float(value) for value in entropy_coefs]
        args.entropy_coef_horizons = [int(value) for value in entropy_coef_horizons]
        if args.entropy_coef_horizons[0] != 0 or any(
            right <= left
            for left, right in zip(
                args.entropy_coef_horizons, args.entropy_coef_horizons[1:]
            )
        ):
            raise ValueError("entropy schedule horizons must start at zero and increase")
    args.ppo_epoch = int(ppo_epochs)
    args.num_mini_batch = int(num_mini_batches)
    args.max_grad_norm = float(max_grad_norm)
    maximum_chunk = min(
        10, max(1, int(rollout_length) // int(num_mini_batches))
    )
    args.data_chunk_length = next(
        length
        for length in range(maximum_chunk, 0, -1)
        if int(rollout_length) % length == 0
    )
    args.share_policy = False
    if rollout_length // args.data_chunk_length < args.num_mini_batch:
        raise ValueError("rollout_length/data_chunk_length cannot supply every minibatch")
    return args


class AuditedRMAPPO(R_MAPPO):
    """Existing MAPPO plus health metrics from actual minibatch ratios."""

    def ppo_update(self, *args, **kwargs):
        result = super().ppo_update(*args, **kwargs)
        ratio = result[5].detach()
        approx_kl, clipped = ppo_health_metrics(ratio, self.clip_param)
        self._health["kl_sum"] += float(approx_kl.sum().cpu())
        self._health["clip_sum"] += float(clipped.sum().cpu())
        self._health["count"] += int(ratio.numel())
        actor_grad_norm = float(torch.as_tensor(result[4]).detach().cpu())
        critic_grad_norm = float(torch.as_tensor(result[1]).detach().cpu())
        self._health["grad_clip_count"] += float(
            max(actor_grad_norm, critic_grad_norm) > self.max_grad_norm
        )
        self._health["optimization_pass_count"] += 1
        return result

    def train(self, buffer, turn_on=True, **kwargs):
        self._health = defaultdict(float)
        result = dict(super().train(buffer, turn_on=turn_on, **kwargs))
        count = int(self._health["count"])
        if count <= 0:
            raise RuntimeError("MAPPO produced no ratio samples")
        result["approx_kl"] = self._health["kl_sum"] / count
        result["clip_fraction"] = self._health["clip_sum"] / count
        passes = int(self._health["optimization_pass_count"])
        if passes <= 0:
            raise RuntimeError("MAPPO produced no optimization passes")
        result["grad_clip_fraction"] = self._health["grad_clip_count"] / passes
        return result


def build_training_stack(
    *, seed: int, rollout_length: int, n_rollout_threads: int = 1, device="cpu", **overrides
):
    target = torch.device(device)
    args = training_args(
        rollout_length=rollout_length,
        n_rollout_threads=n_rollout_threads,
        **overrides,
    )
    _source, canonical, share, action = load_policy_config("rnn")
    observation = _box_with_channels(canonical, 25)
    torch.manual_seed(int(seed))
    policy = R_MAPPOPolicy(args, observation, share, action, device=target)
    trainer = AuditedRMAPPO(args, policy, device=target)
    trainer.adapt_entropy_coef(0)
    return args, policy, trainer, observation, share, action


def architecture_report(seed: int = 20260821) -> dict[str, object]:
    full = build_actionnet(seed=seed)
    partial = build_actionnet(seed=seed)
    full_shapes = {name: list(value.shape) for name, value in full.state_dict().items()}
    partial_shapes = {name: list(value.shape) for name, value in partial.state_dict().items()}
    return {
        "architecture_id": ACTION_ARCHITECTURE_ID,
        "parity": full_shapes == partial_shapes and str(full) == str(partial),
        "same_state_keys": list(full.state_dict()) == list(partial.state_dict()),
        "parameter_shape_differences": {},
        "parameter_shapes": full_shapes,
        "network": str(full),
        "actor_input": [8, 5, 25],
        "critic_input": [8, 5, 40],
        "action_count": 6,
    }


__all__ = [
    "ACTION_ARCHITECTURE_ID",
    "AuditedRMAPPO",
    "FULL_OBSERVATION_CONTRACT",
    "PARTIAL_OBSERVATION_CONTRACT",
    "CANONICAL_ENTROPY_COEFS",
    "CANONICAL_ENTROPY_HORIZONS",
    "GAZE_ENTROPY_COEFS",
    "GAZE_ENTROPY_HORIZONS",
    "ACTOR_FORWARD_CONTEXT_ID",
    "PREUPDATE_LOGPROB_ATOL",
    "RolloutPolicyAuditCapture",
    "SeparatedReplayBuffer",
    "architecture_report",
    "build_actionnet",
    "build_training_stack",
    "entropy_coefficient",
    "explained_variance",
    "install_rollout_context_stable_actor",
    "parameter_hash",
    "preupdate_policy_consistency_audit",
    "ppo_health_metrics",
]
