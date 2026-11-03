"""Partial Gaze PPO with a strictly frozen adapted ActionNet."""

from __future__ import annotations

import copy
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from final_model.checkpoint import (
    load_action_checkpoint,
    load_gaze_checkpoint,
    save_gaze_checkpoint,
    state_dict_hash,
)
from final_model.common.mappo import (
    build_actionnet,
    explained_variance,
    ppo_health_metrics,
)
from final_model.common.overcooked_backend import (
    DirectOvercookedBackend,
    cooking_contract,
)
from final_model.common.partners import (
    FrozenPartnerRepository,
    assert_optimizer_isolation,
    parameter_hash,
)
from final_model.common.reward import hsp_reward_shaping_factor
from final_model.common.resume import (
    TRAINING_STATE_FORMAT_ID,
    capture_global_rng_state,
    optimizer_to,
    require_training_state,
    restore_global_rng_state,
    validate_resume_contract,
)
from final_model.partial_ver.network import GazeCritic, GazeNet
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.server_parallel.gaze_v1 import (
    GAZENET_V1_ACTION_CONTRACT_ID,
    GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID,
    GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID,
    GAZENET_V1_INPUT_CONTRACT_ID,
    GAZENET_V1_REWARD_CONTRACT_ID,
    action_index_to_tile,
    build_spatial_input,
)


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


class LearnedGazeController:
    def __init__(
        self,
        actor: GazeNet,
        critic: GazeCritic,
        device: torch.device,
        *,
        env_index: int = 0,
        action_generator: torch.Generator | None = None,
    ) -> None:
        self.actor, self.critic, self.device = actor, critic, device
        self.env_index = int(env_index)
        self.action_generator = action_generator
        self.policy_version = None if action_generator is None else 0
        self.episode_id = 0
        self.current_task_decision_id: int | None = None
        self.next_record_sequence_id = 0
        self.event_time_ms = 0.0
        self.active: dict[str, Any] | None = None
        self.completed: list[dict[str, Any]] = []
        self.total_attributed_optimization_reward = 0.0
        self.total_env_boundaries = 0
        self.total_bootstrap_unowned_reward_count = 0
        self.total_bootstrap_unowned_reward_sum = 0.0
        self.inference_seconds = 0.0
        self.forward_calls = 0

    def reset(self) -> None:
        if self.active is not None:
            raise RuntimeError("active gaze transition must close before episode reset")
        self.current_task_decision_id = None

    def begin_task_decision(self, task_decision_id: int) -> None:
        value = int(task_decision_id)
        if value < 0:
            raise ValueError("task_decision_id must be non-negative")
        self.current_task_decision_id = value

    def set_event_time(self, timestamp_ms: float) -> None:
        self.event_time_ms = float(timestamp_ms)

    def set_policy_version(self, version: int) -> None:
        if self.action_generator is None:
            raise RuntimeError("policy versioning is parallel-Gaze-only")
        self.policy_version = int(version)

    @staticmethod
    def spatial_input(memory, fixation):
        return build_spatial_input(memory, fixation)

    def _finish(self, next_value: float, done: bool) -> None:
        if self.active is None:
            return
        self.active["next_value"] = float(next_value)
        self.active["done"] = bool(done)
        self.active["elapsed_ms"] = self.event_time_ms - self.active.pop("selection_time_ms")
        self.completed.append(self.active)
        self.active = None

    def __call__(self, memory, fixation, context):
        if self.current_task_decision_id is None:
            raise RuntimeError("begin_task_decision must precede GazeNet selection")
        observation = self.spatial_input(memory, fixation)
        started = time.perf_counter()
        with torch.no_grad():
            spatial = tensor(observation[None], self.device)
            global_context = tensor(context[None], self.device)
            distribution = self.actor.distribution(spatial, global_context)
            value = self.critic(spatial, global_context)
            action = (
                distribution.sample()
                if self.action_generator is None
                else torch.multinomial(
                    distribution.probs, 1, generator=self.action_generator
                ).squeeze(0)
            )
        self._finish(float(value.item()), False)
        index = int(action.item())
        decision_key = (
            self.env_index,
            self.episode_id,
            self.current_task_decision_id,
        )
        self.active = {
            "observation": observation,
            "context": np.ascontiguousarray(context.copy(), dtype=np.float32),
            "action": index,
            "log_prob": float(distribution.log_prob(action).item()),
            "value": float(value.item()),
            "reward": 0.0,
            "sparse_team_reward": 0.0,
            "native_shaped_reward": 0.0,
            "weighted_shaped_reward": 0.0,
            "deliveries": 0,
            "done": False,
            "selection_time_ms": self.event_time_ms,
            "env_index": self.env_index,
            "episode_id": self.episode_id,
            "task_decision_id": self.current_task_decision_id,
            "task_decision_key": decision_key,
            "record_sequence_id": self.next_record_sequence_id,
            "env_step_boundary": False,
            "env_step_boundary_count": 0,
            "boundary_task_decision_keys": [],
            "boundary_optimization_rewards": [],
        }
        if self.policy_version is not None:
            self.active["policy_version"] = self.policy_version
        self.next_record_sequence_id += 1
        self.inference_seconds += time.perf_counter() - started
        self.forward_calls += 1
        return action_index_to_tile(index)

    def consume_inference_profile(self) -> tuple[float, int]:
        seconds = float(self.inference_seconds)
        forwards = int(self.forward_calls)
        self.inference_seconds = 0.0
        self.forward_calls = 0
        return seconds, forwards

    def accrue(
        self, *, optimization_reward: float, sparse_team_reward: float,
        native_shaped_reward: float, weighted_shaped_reward: float, deliveries: int,
        task_decision_id: int | None = None,
    ) -> None:
        reward = float(optimization_reward)
        if self.active is None:
            self.total_bootstrap_unowned_reward_count += 1
            self.total_bootstrap_unowned_reward_sum += reward
            return
        decision_id = (
            self.current_task_decision_id
            if task_decision_id is None
            else int(task_decision_id)
        )
        if decision_id is None:
            raise RuntimeError("env-step boundary has no task decision identifier")
        boundary_key = (self.env_index, self.episode_id, decision_id)
        self.active["reward"] += reward
        self.active["sparse_team_reward"] += float(sparse_team_reward)
        self.active["native_shaped_reward"] += float(native_shaped_reward)
        self.active["weighted_shaped_reward"] += float(weighted_shaped_reward)
        self.active["deliveries"] += int(deliveries)
        self.active["env_step_boundary"] = True
        self.active["env_step_boundary_count"] += 1
        self.active["boundary_task_decision_keys"].append(boundary_key)
        self.active["boundary_optimization_rewards"].append(reward)
        self.total_attributed_optimization_reward += reward
        self.total_env_boundaries += 1

    def finish_episode(self, timestamp_ms: float) -> None:
        self.event_time_ms = float(timestamp_ms)
        self._finish(0.0, True)
        self.episode_id += 1
        self.current_task_decision_id = None

    def drain(self):
        last_ready = -1
        for index, record in enumerate(self.completed):
            if int(record.get("env_step_boundary_count", 0)) > 0 or record["done"]:
                last_ready = index
        if last_ready < 0:
            return []
        result = self.completed[:last_ready + 1]
        self.completed = self.completed[last_ready + 1:]
        return result

    def training_state(self) -> dict[str, Any]:
        state = {
            "event_time_ms": self.event_time_ms,
            "env_index": self.env_index,
            "episode_id": self.episode_id,
            "current_task_decision_id": self.current_task_decision_id,
            "next_record_sequence_id": self.next_record_sequence_id,
            "active": copy.deepcopy(self.active),
            "completed": copy.deepcopy(self.completed),
            "total_attributed_optimization_reward": (
                self.total_attributed_optimization_reward
            ),
            "total_env_boundaries": self.total_env_boundaries,
            "total_bootstrap_unowned_reward_count": (
                self.total_bootstrap_unowned_reward_count
            ),
            "total_bootstrap_unowned_reward_sum": (
                self.total_bootstrap_unowned_reward_sum
            ),
        }
        if self.policy_version is not None:
            state["policy_version"] = self.policy_version
        return state

    def load_training_state(self, state: dict[str, Any]) -> None:
        self.event_time_ms = float(state["event_time_ms"])
        if int(state["env_index"]) != self.env_index:
            raise RuntimeError("gaze controller environment index mismatch")
        self.episode_id = int(state["episode_id"])
        current = state["current_task_decision_id"]
        self.current_task_decision_id = None if current is None else int(current)
        self.next_record_sequence_id = int(state["next_record_sequence_id"])
        if self.action_generator is not None:
            self.policy_version = int(state.get("policy_version", 0))
        self.active = copy.deepcopy(state["active"])
        self.completed = copy.deepcopy(state["completed"])
        self.total_attributed_optimization_reward = float(
            state["total_attributed_optimization_reward"]
        )
        self.total_env_boundaries = int(state["total_env_boundaries"])
        self.total_bootstrap_unowned_reward_count = int(
            state.get("total_bootstrap_unowned_reward_count", 0)
        )
        self.total_bootstrap_unowned_reward_sum = float(
            state.get("total_bootstrap_unowned_reward_sum", 0.0)
        )


def gaze_credit_evaluation_status(metadata: dict[str, Any]) -> str:
    if metadata.get("gaze_credit_assignment_contract_id") == (
        GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID
    ):
        return "current"
    return "legacy_or_unknown"


def gaze_action_evaluation_status(metadata: dict[str, Any]) -> str:
    if metadata.get("gaze_action_contract_id") == GAZENET_V1_ACTION_CONTRACT_ID:
        return "current"
    return "legacy_or_unknown_metadata_runtime_x_major"


def gaze_input_evaluation_status(metadata: dict[str, Any]) -> str:
    if metadata.get("gaze_input_contract_id") == GAZENET_V1_INPUT_CONTRACT_ID:
        return "current"
    return "legacy_or_unknown_input_scaling"


def require_current_gaze_input_contract(
    metadata: dict[str, Any], *, operation: str
) -> None:
    actual = metadata.get("gaze_input_contract_id")
    if actual != GAZENET_V1_INPUT_CONTRACT_ID:
        raise RuntimeError(
            f"{operation} requires Gaze input contract "
            f"{GAZENET_V1_INPUT_CONTRACT_ID!r}; got {actual!r}"
        )


def require_current_gaze_action_contract(
    metadata: dict[str, Any], *, operation: str
) -> None:
    actual = metadata.get("gaze_action_contract_id")
    if actual != GAZENET_V1_ACTION_CONTRACT_ID:
        raise RuntimeError(
            f"{operation} requires gaze action-mapping contract "
            f"{GAZENET_V1_ACTION_CONTRACT_ID!r}; got {actual!r}"
        )


def require_current_gaze_credit_contract(
    metadata: dict[str, Any], *, operation: str
) -> None:
    actual = metadata.get("gaze_credit_assignment_contract_id")
    if actual != GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID:
        raise RuntimeError(
            f"{operation} requires gaze credit-assignment contract "
            f"{GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID!r}; got {actual!r}"
        )


def _task_decision_key(record: dict[str, Any]) -> tuple[int, int, int]:
    key = record.get("task_decision_key")
    if key is not None:
        if len(key) != 3:
            raise RuntimeError("invalid gaze task_decision_key")
        return tuple(int(value) for value in key)
    try:
        return (
            int(record["env_index"]),
            int(record["episode_id"]),
            int(record["task_decision_id"]),
        )
    except KeyError as error:
        raise RuntimeError("gaze record has no canonical task decision identifier") from error


def _boundary_count(record: dict[str, Any]) -> int:
    count = int(
        record.get(
            "env_step_boundary_count",
            1 if record.get("env_step_boundary", False) else 0,
        )
    )
    if count < 0 or bool(record.get("env_step_boundary", count > 0)) != (count > 0):
        raise RuntimeError("inconsistent gaze env-step boundary metadata")
    return count


def _temporally_discounted_reward(
    record: dict[str, Any], gamma: float, boundary_count: int
) -> float:
    boundary_rewards = record.get("boundary_optimization_rewards")
    if boundary_rewards is None:
        return float(record["reward"])
    if len(boundary_rewards) != boundary_count:
        raise RuntimeError("gaze boundary reward ledger/count mismatch")
    return float(sum(
        (float(gamma) ** index) * float(reward)
        for index, reward in enumerate(boundary_rewards)
    ))


def decision_sample_weights(records):
    if not records:
        raise RuntimeError("decision weights require gaze records")
    counts: dict[tuple[int, int, int], int] = {}
    keys = []
    for record in records:
        key = _task_decision_key(record)
        keys.append(key)
        counts[key] = counts.get(key, 0) + 1
    raw = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    scaled = raw * (len(records) / len(counts))
    decision_sums: dict[tuple[int, int, int], float] = {
        key: 0.0 for key in counts
    }
    for key, value in zip(keys, raw):
        decision_sums[key] += float(value)
    sums = np.asarray(list(decision_sums.values()), dtype=np.float64)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("gaze raw decision weights do not sum to one")
    per_decision = np.asarray(list(counts.values()), dtype=np.float64)
    diagnostics = {
        "records_per_task_decision_mean": float(per_decision.mean()),
        "records_per_task_decision_max": float(per_decision.max()),
        "unique_task_decisions": int(len(counts)),
        "decision_weight_sum_min": float(sums.min()),
        "decision_weight_sum_max": float(sums.max()),
        "decision_weight_sum_mean": float(sums.mean()),
    }
    return raw, scaled, diagnostics


def weighted_advantage_normalization(advantages, weights, *, epsilon=1e-8):
    values = np.asarray(advantages, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if values.shape != sample_weights.shape or values.ndim != 1:
        raise ValueError("advantages and weights must be matching vectors")
    total = float(sample_weights.sum())
    if not np.isfinite(values).all() or not np.isfinite(sample_weights).all() or total <= 0:
        raise RuntimeError("invalid weighted advantage normalization inputs")
    mean = float(np.dot(sample_weights, values) / total)
    variance = float(np.dot(sample_weights, np.square(values - mean)) / total)
    std = math.sqrt(max(0.0, variance))
    normalized = (values - mean) / math.sqrt(max(0.0, variance) + epsilon)
    return normalized.astype(np.float32), mean, std


def weighted_torch_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError("values and weights must have identical shapes")
    return (values * weights).mean()


def gaze_advantages(records, gamma, gae_lambda):
    advantages = np.zeros(len(records), dtype=np.float32)
    streams: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, record in enumerate(records):
        key = _task_decision_key(record)
        sequence = int(record.get("record_sequence_id", index))
        streams.setdefault(key[:2], []).append((sequence, index))
    for stream in streams.values():
        ordered = sorted(stream)
        if len({sequence for sequence, _index in ordered}) != len(ordered):
            raise RuntimeError("duplicate gaze record sequence identifier")
        carry = 0.0
        for _sequence, index in reversed(ordered):
            record = records[index]
            not_done = 0.0 if record["done"] else 1.0
            boundaries = _boundary_count(record)
            gamma_t = float(gamma) ** boundaries
            lambda_t = float(gae_lambda) ** boundaries
            delta = (
                _temporally_discounted_reward(record, gamma, boundaries)
                + gamma_t * float(record["next_value"]) * not_done
                - float(record["value"])
            )
            carry = delta + gamma_t * lambda_t * not_done * carry
            advantages[index] = carry
    values = np.asarray([record["value"] for record in records], dtype=np.float32)
    return advantages, advantages + values


def gaze_ppo_update(actor, critic, actor_optimizer, critic_optimizer, records, *, device, **ppo):
    if not records:
        raise RuntimeError("Gaze PPO requires completed transitions")
    advantages, returns = gaze_advantages(records, ppo["gamma"], ppo["gae_lambda"])
    raw_weights, scaled_weights, diagnostics = decision_sample_weights(records)
    unweighted_mean = float(np.mean(advantages))
    unweighted_std = float(np.std(advantages))
    rollout_values = np.asarray(
        [record["value"] for record in records], dtype=np.float32
    )
    advantages, weighted_mean, weighted_std = weighted_advantage_normalization(
        advantages, raw_weights
    )
    diagnostics.update({
        "weighted_advantage_mean": weighted_mean,
        "weighted_advantage_std": weighted_std,
        "unweighted_advantage_mean": unweighted_mean,
        "unweighted_advantage_std": unweighted_std,
        "explained_variance": explained_variance(rollout_values, returns),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "value_mean": float(np.mean(rollout_values)),
        "value_std": float(np.std(rollout_values)),
        "env_boundary_count_in_records": int(
            sum(_boundary_count(record) for record in records)
        ),
    })
    size = len(records)
    mini = max(1, math.ceil(size / min(ppo["num_mini_batches"], size)))
    aggregates = {key: [] for key in (
        "policy_loss", "value_loss", "gaze_entropy", "actor_grad_norm", "critic_grad_norm",
        "approx_kl", "clip_fraction", "weighted_gaze_entropy",
        "weighted_approx_kl", "weighted_clip_fraction",
        "critic_grad_clip_fraction",
    )}
    for _ in range(ppo["ppo_epochs"]):
        for indices in torch.randperm(size, device=device).split(mini):
            ids = indices.detach().cpu().tolist()
            spatial = tensor(np.stack([records[i]["observation"] for i in ids]), device)
            context = tensor(np.stack([records[i]["context"] for i in ids]), device)
            actions = torch.as_tensor([records[i]["action"] for i in ids], device=device)
            old_log = tensor([records[i]["log_prob"] for i in ids], device)
            advantage = tensor(advantages[ids], device)
            targets = tensor(returns[ids], device)
            sample_weights = tensor(scaled_weights[ids], device)
            distribution = actor.distribution(spatial, context)
            new_log = distribution.log_prob(actions)
            ratio = torch.exp(new_log - old_log)
            clipped = ratio.clamp(1.0 - ppo["clip_param"], 1.0 + ppo["clip_param"])
            policy_samples = torch.min(ratio * advantage, clipped * advantage)
            policy_loss = -weighted_torch_mean(policy_samples, sample_weights)
            entropy = distribution.entropy().mean()
            weighted_entropy = weighted_torch_mean(
                distribution.entropy(), sample_weights
            )
            actor_optimizer.zero_grad(set_to_none=True)
            (policy_loss - ppo["entropy_coef"] * weighted_entropy).backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), ppo["max_grad_norm"])
            actor_optimizer.step()
            values = critic(spatial, context)
            value_loss = weighted_torch_mean(
                (values - targets).square(), sample_weights
            )
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), ppo["max_grad_norm"])
            critic_optimizer.step()
            with torch.no_grad():
                approx_kl_samples, clipped = ppo_health_metrics(ratio, ppo["clip_param"])
                approx_kl = approx_kl_samples.mean()
                clip_fraction = clipped.mean()
                weighted_approx_kl = weighted_torch_mean(
                    approx_kl_samples, sample_weights
                )
                weighted_clip_fraction = weighted_torch_mean(
                    clipped, sample_weights
                )
            values_to_add = {
                "policy_loss": policy_loss, "value_loss": value_loss, "gaze_entropy": entropy,
                "actor_grad_norm": actor_grad, "critic_grad_norm": critic_grad,
                "approx_kl": approx_kl, "clip_fraction": clip_fraction,
                "weighted_gaze_entropy": weighted_entropy,
                "weighted_approx_kl": weighted_approx_kl,
                "weighted_clip_fraction": weighted_clip_fraction,
                "critic_grad_clip_fraction": torch.as_tensor(
                    critic_grad > ppo["max_grad_norm"], device=device
                ),
            }
            for key, value in values_to_add.items():
                aggregates[key].append(float(value.detach().cpu()))
    return {
        **{key: float(np.mean(values)) for key, values in aggregates.items()},
        **diagnostics,
    }


class GazeTrainingSession:
    def __init__(
        self, *, source_checkpoint: Path | None, horizon: int, decision_time: int, fixation_shape: str,
        seed: int, rollout_length: int, episode_length: int, device: str, partner: str | None = None,
        actor_lr=5e-4, critic_lr=5e-4, clip_param=0.2, gamma=0.99, gae_lambda=0.95,
        entropy_coef=None, entropy_coefs=(0.2, 0.05, 0.01),
        entropy_coef_horizons=(0, 40_000_000, 80_000_000),
        ppo_epochs=15, num_mini_batches=1, max_grad_norm=10.0,
        n_rollout_threads=1,
        resume_checkpoint: Path | None = None,
    ) -> None:
        raise RuntimeError(
            "legacy decision-driven GazeTrainingSession is retired; use "
            "canonical_training.PartialTrainingSession"
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.horizon, self.decision_time = int(horizon), int(decision_time)
        self.fixation_shape, self.seed = fixation_shape, int(seed)
        self.rollout_length, self.episode_length = int(rollout_length), int(episode_length)
        self.n_rollout_threads = int(n_rollout_threads)
        self.partner = partner
        if self.n_rollout_threads <= 0:
            raise ValueError("n_rollout_threads must be positive")
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        if source_checkpoint is not None and resume_checkpoint is not None:
            raise ValueError("--source-checkpoint and --resume-checkpoint are mutually exclusive")
        self.action_actor = build_actionnet(seed=self.seed, device=self.device)
        self.source_provenance = None
        if resume_checkpoint is None:
            if source_checkpoint is None:
                raise ValueError("Partial Gaze requires --source-checkpoint or --resume-checkpoint")
            self.source_provenance = load_action_checkpoint(
                source_checkpoint,
                actor=self.action_actor,
                expected_mode="partial",
                expected_horizon=horizon,
                expected_condition={
                    "decision_time": int(decision_time),
                    "fixation_shape": fixation_shape,
                    **cooking_contract(),
                },
            )
            adaptation = self.source_provenance.get("source_provenance")
            if not isinstance(adaptation, dict) or adaptation.get("source_horizon") != int(horizon):
                raise RuntimeError(
                    "canonical Gaze training requires a matching-condition Partial Action checkpoint "
                    "with initialization provenance"
                )
            initialization = adaptation.get("initialization")
            if initialization not in {None, "fresh_partial", "matching_full_checkpoint"}:
                raise RuntimeError(
                    f"unsupported Partial Action initialization provenance: {initialization!r}"
                )
        self.action_actor.eval().requires_grad_(False)
        self.gaze_actor, self.gaze_critic = GazeNet().to(self.device), GazeCritic().to(self.device)
        self.gaze_optimizer = torch.optim.Adam(self.gaze_actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.gaze_critic.parameters(), lr=critic_lr)
        if entropy_coef is not None:
            entropy_coefs, entropy_coef_horizons = (float(entropy_coef),), (0,)
        if len(entropy_coefs) != len(entropy_coef_horizons) or not entropy_coefs:
            raise ValueError("entropy schedule coefficients and horizons must have equal lengths")
        self.entropy_coefs = tuple(float(value) for value in entropy_coefs)
        self.entropy_coef_horizons = tuple(int(value) for value in entropy_coef_horizons)
        self.ppo = {
            "clip_param": float(clip_param), "gamma": float(gamma), "gae_lambda": float(gae_lambda),
            "entropy_coef": self.entropy_coefs[0], "ppo_epochs": int(ppo_epochs),
            "num_mini_batches": int(num_mini_batches), "max_grad_norm": float(max_grad_norm),
        }
        repository = FrozenPartnerRepository(device=self.device)
        self.backends = []
        shared_oracle = None
        for env_index in range(self.n_rollout_threads):
            backend = DirectOvercookedBackend(
                seed=seed + env_index,
                horizon=horizon,
                episode_length=episode_length,
                device=device,
                partner=partner,
                partner_repository=repository,
                prediction_oracle=shared_oracle,
            )
            self.backends.append(backend)
            shared_oracle = backend.oracle
        optimizers = {
            "gaze_actor_optimizer": self.gaze_optimizer,
            "gaze_critic_optimizer": self.critic_optimizer,
        }
        for actor in {id(backend.partners.actor): backend.partners.actor for backend in self.backends}.values():
            assert_optimizer_isolation(actor, optimizers)
        self.partner_optimizer_isolation_verified = True
        self.controllers = [
            LearnedGazeController(self.gaze_actor, self.gaze_critic, self.device)
            for _ in self.backends
        ]
        self.adapters = [
            PartialObservationAdapter(
                backend,
                decision_time=decision_time,
                fixation_shape=fixation_shape,
                gaze_policy=controller,
            )
            for backend, controller in zip(self.backends, self.controllers)
        ]
        self.backend = self.backends[0]
        self.controller = self.controllers[0]
        self.adapter = self.adapters[0]
        self.action_hidden = np.zeros((self.n_rollout_threads, 1, 64), dtype=np.float32)
        self.action_mask = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
        self._executor = (
            None
            if self.n_rollout_threads == 1
            else ThreadPoolExecutor(max_workers=self.n_rollout_threads)
        )
        self.completed_steps = self.update_count = 0
        if resume_checkpoint is not None:
            self._resume(resume_checkpoint)
        self.action_hash_start = parameter_hash(self.action_actor)

    def _resume_contract(self) -> dict[str, Any]:
        return {
            "training_stage": "gaze",
            "observation_mode": "partial",
            "horizon": self.horizon,
            "decision_time": self.decision_time,
            "fixation_shape": self.fixation_shape,
            "seed": self.seed,
            "rollout_length": self.rollout_length,
            "episode_length": self.episode_length,
            "n_rollout_threads": self.n_rollout_threads,
            "partner": self.partner,
            "device_type": self.device.type,
            "actor_lr": float(self.gaze_optimizer.param_groups[0]["lr"]),
            "critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
            "clip_param": self.ppo["clip_param"],
            "gamma": self.ppo["gamma"],
            "gae_lambda": self.ppo["gae_lambda"],
            "entropy_coefs": list(self.entropy_coefs),
            "entropy_coef_horizons": list(self.entropy_coef_horizons),
            "ppo_epochs": self.ppo["ppo_epochs"],
            "num_mini_batches": self.ppo["num_mini_batches"],
            "max_grad_norm": self.ppo["max_grad_norm"],
            **cooking_contract(),
        }

    def _training_state(self) -> dict[str, Any]:
        return {
            "format_id": TRAINING_STATE_FORMAT_ID,
            "contract": self._resume_contract(),
            "completed_steps": self.completed_steps,
            "update_count": self.update_count,
            "gaze_optimizer": self.gaze_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "backends": [backend.training_state() for backend in self.backends],
            "adapters": [adapter.training_state() for adapter in self.adapters],
            "action_hidden": self.action_hidden.copy(),
            "action_mask": self.action_mask.copy(),
            "rng": capture_global_rng_state(),
        }

    def _resume(self, path: Path) -> None:
        metadata = load_gaze_checkpoint(
            path,
            action_actor=self.action_actor,
            gaze_actor=self.gaze_actor,
            gaze_critic=self.gaze_critic,
            horizon=self.horizon,
            decision_time=self.decision_time,
            fixation_shape=self.fixation_shape,
            expected_condition=cooking_contract(),
        )
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = require_training_state(payload)
        validate_resume_contract(state.get("contract"), self._resume_contract())
        if int(metadata.get("step", -1)) != int(state.get("completed_steps", -2)):
            raise RuntimeError("checkpoint metadata/training-state step mismatch")
        self.gaze_optimizer.load_state_dict(state["gaze_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        optimizer_to(self.gaze_optimizer, self.device)
        optimizer_to(self.critic_optimizer, self.device)
        if len(state["backends"]) != len(self.backends) or len(state["adapters"]) != len(self.adapters):
            raise RuntimeError("resume rollout environment count mismatch")
        for backend, saved in zip(self.backends, state["backends"]):
            backend.load_training_state(saved)
        for adapter, saved in zip(self.adapters, state["adapters"]):
            adapter.load_training_state(saved)
        for name in ("action_hidden", "action_mask"):
            target = getattr(self, name)
            source = np.asarray(state[name], dtype=np.float32)
            if source.shape != target.shape:
                raise RuntimeError(f"resume {name} shape mismatch: {source.shape} != {target.shape}")
            target[...] = source
        self.completed_steps = int(state["completed_steps"])
        self.update_count = int(state["update_count"])
        update_steps = self.rollout_length * self.n_rollout_threads
        if (
            self.completed_steps <= 0
            or self.completed_steps % update_steps
            or self.update_count != self.completed_steps // update_steps
            or sum(backend.environment_steps for backend in self.backends)
            != self.completed_steps
        ):
            raise RuntimeError("invalid Gaze training progress in resume checkpoint")
        self.source_provenance = metadata.get("source_partial_action_provenance")
        if not isinstance(self.source_provenance, dict) or not self.source_provenance:
            raise RuntimeError("Gaze resume checkpoint has no Partial Action provenance")
        self.ppo["entropy_coef"] = self._entropy_coef(self.completed_steps)
        restore_global_rng_state(state["rng"])

    def _entropy_coef(self, steps: int) -> float:
        for index in range(len(self.entropy_coef_horizons) - 1):
            start, end = self.entropy_coef_horizons[index:index + 2]
            if start <= steps < end:
                fraction = (steps - start) / (end - start)
                return (
                    (1.0 - fraction) * self.entropy_coefs[index]
                    + fraction * self.entropy_coefs[index + 1]
                )
        return self.entropy_coefs[-1]

    def run_update(self):
        action_before = parameter_hash(self.action_actor)
        step_guard_checks_before = sum(
            backend.partner_step_guard_checks for backend in self.backends
        )
        full_hash_checks_before = sum(
            backend.partner_full_hash_checks for backend in self.backends
        )
        for backend in self.backends:
            backend.verify_partner_parameters()
        gaze_before = state_dict_hash(self.gaze_actor.state_dict())
        forwards_before = sum(adapter.actionnet_online_forwards for adapter in self.adapters)
        decisions_before = sum(adapter.task_decisions for adapter in self.adapters)
        partial_before = {
            key: sum(float(adapter.metrics()[key]) for adapter in self.adapters)
            for key in self.adapter.metrics()
        }
        distance_before = sum(adapter.distance_sum_degrees for adapter in self.adapters)
        records = []
        episodes = 0
        started = time.perf_counter()
        for rollout_step in range(self.rollout_length):
            reward_shaping_factor = hsp_reward_shaping_factor(
                self.completed_steps + rollout_step * self.n_rollout_threads
            )
            for backend in self.backends:
                backend.set_reward_shaping_factor(reward_shaping_factor)
            current = []
            for adapter in self.adapters:
                current.append(adapter.prepare_decision())
            observations = np.stack([item[0] for item in current])
            available = np.stack([item[2] for item in current])
            with torch.no_grad():
                probabilities, hidden = self.action_actor.get_probs(
                    observations, self.action_hidden, self.action_mask, available
                )
                actions = Categorical(probs=probabilities).sample().detach().cpu().numpy()
            self.action_hidden = hidden.detach().cpu().numpy()
            self.action_mask.fill(1.0)
            selected = [int(value) for value in actions.reshape(-1)]
            if self._executor is None:
                results = [self.adapter.commit_decision(selected[0])]
            else:
                results = list(
                    self._executor.map(
                        lambda pair: pair[0].commit_decision(pair[1]),
                        zip(self.adapters, selected),
                    )
                )
            for env_index, (adapter, controller, result) in enumerate(
                zip(self.adapters, self.controllers, results)
            ):
                controller.accrue(
                    optimization_reward=result["optimization_reward"],
                    sparse_team_reward=result["sparse_team_reward"],
                    native_shaped_reward=result["native_shaped_reward"],
                    weighted_shaped_reward=result["weighted_shaped_reward"],
                    deliveries=result["deliveries"],
                )
                if result["done"]:
                    episodes += 1
                    controller.finish_episode(adapter.current_time_ms)
                records.extend(controller.drain())
                if result["done"]:
                    adapter.reset_episode()
                    self.action_hidden[env_index].fill(0.0)
                    self.action_mask[env_index].fill(0.0)
        collection_seconds = time.perf_counter() - started
        for backend in self.backends:
            backend.verify_partner_parameters()
        gaze_transition_count = len(records)
        optimization = float(sum(record["reward"] for record in records))
        sparse = float(sum(record["sparse_team_reward"] for record in records))
        shaped = float(sum(record["native_shaped_reward"] for record in records))
        weighted_shaped = float(sum(record["weighted_shaped_reward"] for record in records))
        deliveries = int(sum(record["deliveries"] for record in records))
        if not np.isclose(
            optimization, sparse + weighted_shaped, rtol=1e-10, atol=1e-6
        ):
            raise RuntimeError("Gaze PPO reward components do not sum to optimization reward")
        update_started = time.perf_counter()
        upcoming_steps = self.completed_steps + self.rollout_length * self.n_rollout_threads
        self.ppo["entropy_coef"] = self._entropy_coef(upcoming_steps)
        update = gaze_ppo_update(
            self.gaze_actor, self.gaze_critic, self.gaze_optimizer, self.critic_optimizer,
            records, device=self.device, **self.ppo,
        )
        update_seconds = time.perf_counter() - update_started
        records.clear()
        self.completed_steps = upcoming_steps
        self.update_count += 1
        action_after = parameter_hash(self.action_actor)
        if action_after != action_before or action_after != self.action_hash_start:
            raise RuntimeError("frozen ActionNet changed during Gaze training")
        forwards = sum(adapter.actionnet_online_forwards for adapter in self.adapters) - forwards_before
        decisions = sum(adapter.task_decisions for adapter in self.adapters) - decisions_before
        partner_counts: dict[str, int] = {}
        for backend in self.backends:
            for name, count in backend.partners.counts.items():
                partner_counts[name] = partner_counts.get(name, 0) + int(count)
        result = {
            "observation_mode": "partial", "training_stage": "gaze", "horizon": self.horizon,
            "decision_time": self.decision_time, "fixation_shape": self.fixation_shape,
            "environment_steps": self.completed_steps, "task_decisions": decisions,
            "actionnet_online_forwards": forwards, "online_forwards_per_decision": forwards / decisions,
            "actionnet_batched_forward_calls": self.rollout_length,
            "n_rollout_threads": self.n_rollout_threads,
            "updates": self.update_count, "episodes": episodes,
            "reward": optimization, "optimization_reward": optimization,
            "sparse_team_reward": sparse, "native_shaped_reward": shaped, "deliveries": deliveries,
            "weighted_shaped_reward": weighted_shaped,
            "reward_shaping_factor": hsp_reward_shaping_factor(upcoming_steps),
            **cooking_contract(),
            "active_partner": self.backend.active_partner,
            "active_partners": [backend.active_partner for backend in self.backends],
            "sampled_partner_identities": partner_counts,
            "partner_step_guard_checks": sum(
                backend.partner_step_guard_checks for backend in self.backends
            ) - step_guard_checks_before,
            "partner_step_guard_failures": sum(
                backend.partner_step_guard_failures for backend in self.backends
            ),
            "partner_full_hash_checks": sum(
                backend.partner_full_hash_checks for backend in self.backends
            ) - full_hash_checks_before,
            "partner_full_hash_failures": sum(
                backend.partner_full_hash_failures for backend in self.backends
            ),
            "partner_full_hash_baseline": self.backend.last_partner_full_hash_baseline,
            "partner_full_hash_actual": self.backend.last_partner_full_hash_actual,
            "partner_optimizer_isolation_verified": self.partner_optimizer_isolation_verified,
            "action_hash_before": action_before, "action_hash_after": action_after,
            "action_hash_unchanged": action_before == action_after,
            "actionnet_requires_grad": any(p.requires_grad for p in self.action_actor.parameters()),
            "action_optimizer_steps": 0,
            "gaze_hash_before": gaze_before,
            "gaze_hash_after": state_dict_hash(self.gaze_actor.state_dict()),
            "gaze_parameters_updated": gaze_before != state_dict_hash(self.gaze_actor.state_dict()),
            "gaze_transition_count": gaze_transition_count,
            "rollout_collection_seconds": collection_seconds, "ppo_update_seconds": update_seconds,
            "decisions_per_second": decisions / max(collection_seconds, 1e-12),
            **update,
        }
        after = {
            key: sum(float(adapter.metrics()[key]) for adapter in self.adapters)
            for key in self.adapter.metrics()
        }
        for key in (
            "gaze_requests", "acquisitions", "same_target_acquisitions",
            "new_target_acquisitions", "prep_count", "exec_count", "dwell_count",
        ):
            result[key] = after[key] - partial_before[key]
        result["valid_memory_fraction"] = (
            after["valid_memory_fraction"] / self.n_rollout_threads
        )
        result["mean_saccade_distance_degrees"] = (
            sum(adapter.distance_sum_degrees for adapter in self.adapters) - distance_before
        ) / max(result["new_target_acquisitions"], 1)
        return result

    def save(self, path):
        return save_gaze_checkpoint(
            path, action_actor=self.action_actor, gaze_actor=self.gaze_actor,
            gaze_critic=self.gaze_critic, horizon=self.horizon, decision_time=self.decision_time,
            fixation_shape=self.fixation_shape, step=self.completed_steps, seed=self.seed,
            source_provenance=self.source_provenance,
            training_state=self._training_state(),
            condition=cooking_contract(),
        )
