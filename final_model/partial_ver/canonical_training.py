"""Canonical acquisition-driven Partial Action/Gaze training session."""

from __future__ import annotations

import copy
import gc
import hashlib
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from final_model.checkpoint import ACTION_SPACE, FORMAT_ID, state_dict_hash
from final_model.common.mappo import (
    ACTION_ARCHITECTURE_ID,
    CANONICAL_ENTROPY_COEFS,
    CANONICAL_ENTROPY_HORIZONS,
    GAZE_ENTROPY_COEFS,
    GAZE_ENTROPY_HORIZONS,
    PREUPDATE_LOGPROB_ATOL,
    SeparatedReplayBuffer,
    build_training_stack,
    entropy_coefficient,
    explained_variance,
    ppo_health_metrics,
)
from final_model.common.inference_batching import (
    PARTIAL_LEARNED_PRODUCTION_MODES,
)
from final_model.common.overcooked_backend import (
    CANONICAL_ENCODER_ID,
    DirectOvercookedBackend,
    cooking_contract,
)
from final_model.common.partners import (
    FIXED_HSP_PARTNER_CONTRACT_ID,
    FIXED_HSP_PARTNER_ID,
    FIXED_HSP_PARTNER_SHA256,
    PARTNER_SET_ID,
    FixedHSPPartnerRepository,
    assert_optimizer_isolation,
    parameter_hash,
)
from final_model.common.paths import file_sha256
from final_model.common.reward import HSP_REWARD_SHAPING_HORIZON, hsp_reward_shaping_factor
from final_model.common.resume import (
    TRAINING_STATE_FORMAT_ID,
    capture_global_rng_state,
    optimizer_to,
    restore_global_rng_state,
    validate_resume_contract,
)
from final_model.partial_ver.conditions import (
    CONDITION_SCHEDULER_ID,
    PartialCondition,
    PartialConditionScheduler,
)
from final_model.partial_ver.constraints import (
    BASELINE_CONSTRAINT_CONFIG,
    FITTED_EMMA_CONSTRAINT_CONFIG,
    HumanLikeConstraintConfig,
    constraint_config_for_emma,
    constraint_config_from_metadata,
    validate_constraint_implementation,
)
from final_model.partial_ver.gaze import (
    EmmaTimingConfig,
    HUMAN_DWELL_RNG_SEED_OFFSET,
    VISUAL_ENCODING_EMMA_MODES,
)
from final_model.partial_ver.memory import (
    CONTINUOUS_FRESHNESS_CONTRACT_ID,
    MEMORY_LIFETIME_MS,
    MEMORY_RNG_CONTRACT_ID,
    MEMORY_RNG_SEED_OFFSET,
    NO_DECAY_FRESHNESS_CONTRACT_ID,
    STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID,
    observation_contract_id,
)
from final_model.partial_ver.network import GAZE_ARCHITECTURE_ID, GazeCritic, GazeNet
from final_model.partial_ver.observability import (
    accumulate_partial_metrics,
    finalize_partial_metrics,
    new_partial_metric_totals,
    partial_observation_statistics,
    validate_partial_metric_totals,
)
from final_model.partial_ver.observation import (
    ACTOR_AVAILABLE_ACTION_CONTRACT_ID,
    FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID,
    PartialObservationAdapter,
)
from final_model.partial_ver.peripheral import PERIPHERAL_RNG_SEED_OFFSET
from final_model.partial_ver.preview_opacity import (
    PREVIEW_OPACITY_RNG_SEED_OFFSET,
)
from final_model.partial_ver.rule_based import RULE_BASED_GAZE_POLICY_ID
from final_model.partial_ver.training import (
    LearnedGazeController,
    GAZENET_V1_ACTION_CONTRACT_ID,
    GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID,
    GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID,
    GAZENET_V1_INPUT_CONTRACT_ID,
    GAZENET_V1_REWARD_CONTRACT_ID,
    gaze_advantages,
    gaze_ppo_update,
    require_current_gaze_action_contract,
    require_current_gaze_credit_contract,
    require_current_gaze_input_contract,
)


PARTIAL_CHECKPOINT_FORMAT_ID = "canonical_partial_training_checkpoint_v2"
PARTIAL_PPO_CONTRACT_ID = "latest_acquisition_proposal_direct_recurrent_replay_v1"
DIRECT_JOINT_INITIALIZATION_ID = (
    "full_action_actor_fresh_partial_critic_and_gaze_v1"
)
STAY_ACTION_INDEX = 4
FIRST_CONV_KEY = "base.cnn.0.weight"


def _as_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _strict_cooking(metadata: dict[str, Any]) -> None:
    mismatch = {
        key: (metadata.get(key), expected)
        for key, expected in cooking_contract().items()
        if metadata.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(f"source checkpoint cooking contract mismatch: {mismatch}")


def _load_payload(path: Path, device) -> tuple[Path, dict, dict]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"checkpoint does not exist: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise RuntimeError("checkpoint has no metadata contract")
    return source, payload, dict(payload["metadata"])


def _source_actor_state(payload: dict) -> dict:
    for key in ("actor", "action_actor"):
        state = payload.get(key)
        if isinstance(state, dict):
            return state
    raise RuntimeError("source checkpoint has no ActionNet state")


class FreshnessGradientAudit:
    def __init__(self, actor: torch.nn.Module) -> None:
        weight = dict(actor.named_parameters()).get(FIRST_CONV_KEY)
        if weight is None or weight.ndim != 4 or weight.shape[1] != 25:
            raise RuntimeError("ActionNet first convolution contract changed")
        self.weight = weight
        self.gradient_norms: list[float] = []
        self.handle = weight.register_hook(self._record) if weight.requires_grad else None

    def _record(self, gradient):
        self.gradient_norms.append(
            float(torch.linalg.vector_norm(gradient[:, 20]).detach().cpu())
        )
        return gradient

    def begin(self):
        self.gradient_norms.clear()
        return self.weight[:, 20].detach().cpu().clone()

    def report(self, before) -> dict[str, float]:
        after = self.weight[:, 20].detach().cpu()
        return {
            "freshness_weight_norm": float(torch.linalg.vector_norm(after)),
            "freshness_gradient_norm": float(max(self.gradient_norms, default=0.0)),
            "freshness_parameter_delta_norm": float(
                torch.linalg.vector_norm(after - before)
            ),
        }


class PartialTrainingSession:
    """One Partial runtime for rule/learned gaze and Action/Gaze/joint PPO."""

    def __init__(
        self,
        *,
        gaze_type: str,
        train_action: bool,
        train_gaze: bool,
        decision_times,
        horizons,
        fixation_shapes,
        seed: int,
        rollout_length: int,
        episode_length: int,
        n_rollout_threads: int,
        device: str,
        source_checkpoint: Path | None,
        resume_checkpoint: Path | None = None,
        fixed_partner_checkpoint: Path | None = None,
        configured_total_steps: int = 20_000_000,
        checkpoint_interval: int = 1_000_000,
        reward_shaping_horizon_steps: int = HSP_REWARD_SHAPING_HORIZON,
        rule_gaze_probabilities=None,
        direct_joint_from_full: bool = False,
        ground_truth_policy_input: bool = False,
        memory_lifetime_ms: float | None = MEMORY_LIFETIME_MS,
        memory_decay_mode: str = "linear",
        memory_tau_ms: float | None = None,
        emma_timing: EmmaTimingConfig | None = None,
        constraint_config: HumanLikeConstraintConfig | None = None,
        inference_batching: str = "legacy",
        inference_shadow_audit: bool = False,
        parity_audit: bool = False,
        partner: str | None = None,
        actor_lr=5e-4,
        critic_lr=5e-4,
        clip_param=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coefs=None,
        entropy_coef_horizons=None,
        ppo_epochs=15,
        num_mini_batches=1,
        max_grad_norm=10.0,
    ) -> None:
        self.gaze_type = str(gaze_type)
        self.train_action, self.train_gaze = bool(train_action), bool(train_gaze)
        self.direct_joint_from_full = bool(direct_joint_from_full)
        self.ground_truth_policy_input = bool(ground_truth_policy_input)
        if self.gaze_type not in {"rule_based", "learned"}:
            raise ValueError("invalid Partial gaze type")
        if self.gaze_type == "rule_based" and (not self.train_action or self.train_gaze):
            raise ValueError("rule_based gaze supports Action training only")
        if self.gaze_type == "learned" and not (self.train_action or self.train_gaze):
            raise ValueError("learned Partial runtime must train at least one component")
        if self.direct_joint_from_full and not (
            self.gaze_type == "learned" and self.train_action and self.train_gaze
        ):
            raise ValueError(
                "direct joint from Full requires learned gaze and both train components"
            )
        if self.ground_truth_policy_input and not (
            self.gaze_type == "learned" and self.train_gaze
        ):
            raise ValueError(
                "ground-truth policy input requires learned Gaze training"
            )
        if entropy_coefs is None and entropy_coef_horizons is None:
            if self.train_gaze and not self.train_action:
                entropy_coefs = GAZE_ENTROPY_COEFS
                entropy_coef_horizons = GAZE_ENTROPY_HORIZONS
            else:
                entropy_coefs = CANONICAL_ENTROPY_COEFS
                entropy_coef_horizons = CANONICAL_ENTROPY_HORIZONS
        elif entropy_coefs is None or entropy_coef_horizons is None:
            raise ValueError(
                "entropy coefficients and horizons must be provided together"
            )
        self.inference_batching = str(inference_batching)
        self.inference_shadow_audit = bool(inference_shadow_audit)
        self.parity_audit = bool(parity_audit)
        self.last_parity_audit = None
        if self.gaze_type == "rule_based" and self.inference_batching != "legacy":
            raise ValueError("rule-based Partial supports only legacy inference")
        if (
            self.gaze_type == "learned"
            and self.inference_batching not in PARTIAL_LEARNED_PRODUCTION_MODES
        ):
            raise ValueError(
                "learned Partial production excludes gaze/ActionNet batching because "
                "legacy global Torch RNG ordering cannot be preserved"
            )
        if self.inference_shadow_audit and self.inference_batching == "legacy":
            raise ValueError("inference shadow audit requires predictor batching")
        self.train_components = [
            name for name, enabled in (("action", self.train_action), ("gaze", self.train_gaze))
            if enabled
        ]
        self.training_stage = (
            "joint" if self.train_action and self.train_gaze
            else "action" if self.train_action else "gaze"
        )
        self.seed = int(seed)
        self.rollout_length = int(rollout_length)
        self.episode_length = int(episode_length)
        self.n_rollout_threads = int(n_rollout_threads)
        self.configured_total_steps = int(configured_total_steps)
        self.checkpoint_interval = int(checkpoint_interval)
        self.reward_shaping_horizon_steps = int(reward_shaping_horizon_steps)
        self.memory_decay_mode = str(memory_decay_mode)
        self.memory_lifetime_ms = (
            None if memory_lifetime_ms is None else float(memory_lifetime_ms)
        )
        self.memory_tau_ms = (
            None if memory_tau_ms is None else float(memory_tau_ms)
        )
        self.memory_random_seed = (
            self.seed + MEMORY_RNG_SEED_OFFSET
            if self.memory_decay_mode == "stochastic_exponential"
            else None
        )
        if emma_timing is not None and constraint_config is not None:
            raise ValueError("provide constraint_config or legacy emma_timing, not both")
        self.constraint_config = (
            FITTED_EMMA_CONSTRAINT_CONFIG
            if constraint_config is None and emma_timing is None
            else constraint_config
            if constraint_config is not None
            else constraint_config_for_emma(emma_timing)
        )
        validate_constraint_implementation(self.constraint_config)
        self.emma_timing = self.constraint_config.emma
        self.freshness_contract_id = (
            NO_DECAY_FRESHNESS_CONTRACT_ID
            if self.memory_decay_mode == "none"
            else STOCHASTIC_EXPONENTIAL_FRESHNESS_CONTRACT_ID
            if self.memory_decay_mode == "stochastic_exponential"
            else CONTINUOUS_FRESHNESS_CONTRACT_ID
        )
        self.memory_contract_id = observation_contract_id(
            self.memory_lifetime_ms, self.memory_decay_mode, self.memory_tau_ms
        )
        if partner is not None:
            raise ValueError("canonical Partial training does not accept partner selection")
        if fixed_partner_checkpoint is None:
            raise ValueError("canonical Partial training requires a fixed partner checkpoint")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if min(self.rollout_length, self.episode_length, self.n_rollout_threads) <= 0:
            raise ValueError("rollout, episode, and environment counts must be positive")
        if self.reward_shaping_horizon_steps < 0:
            raise ValueError("reward shaping horizon cannot be negative")
        if source_checkpoint is not None and resume_checkpoint is not None:
            raise ValueError("source and resume checkpoints are mutually exclusive")
        if self.gaze_type == "learned" and source_checkpoint is None and resume_checkpoint is None:
            raise ValueError("Partial Gaze requires a source or resume checkpoint")

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.scheduler = PartialConditionScheduler(
            decision_times=decision_times,
            horizons=horizons,
            fixation_shapes=fixation_shapes,
            seed=self.seed + 71_000_003,
        )
        self.conditions = [self.scheduler.next() for _ in range(self.n_rollout_threads)]
        self.condition_decision_counts = {
            condition: 0 for condition in self.scheduler.conditions
        }
        ppo = dict(
            actor_lr=actor_lr, critic_lr=critic_lr, clip_param=clip_param,
            gamma=gamma, gae_lambda=gae_lambda, entropy_coefs=entropy_coefs,
            entropy_coef_horizons=entropy_coef_horizons, ppo_epochs=ppo_epochs,
            num_mini_batches=num_mini_batches, max_grad_norm=max_grad_norm,
        )
        args, policy, trainer, obs_space, share_space, action_space = build_training_stack(
            seed=self.seed, rollout_length=self.rollout_length,
            n_rollout_threads=self.n_rollout_threads, device=self.device, **ppo,
        )
        self.args, self.policy, self.trainer = args, policy, trainer
        self.ppo = {
            "clip_param": float(clip_param), "gamma": float(gamma),
            "gae_lambda": float(gae_lambda), "ppo_epochs": int(ppo_epochs),
            "num_mini_batches": int(num_mini_batches),
            "max_grad_norm": float(max_grad_norm),
        }
        self.entropy_coefs = tuple(float(value) for value in entropy_coefs)
        self.entropy_coef_horizons = tuple(int(value) for value in entropy_coef_horizons)
        self.source_provenance = None
        self.predictor_parity = None
        self._actionnet_rollout_seconds = 0.0
        self._actionnet_rollout_forward_count = 0
        self._parity_action_records = None
        self.gaze_actor = self.gaze_critic = None
        self.gaze_optimizer = self.gaze_critic_optimizer = None
        if self.gaze_type == "learned":
            self.gaze_actor = GazeNet().to(self.device)
            self.gaze_critic = GazeCritic().to(self.device)
            self.gaze_optimizer = torch.optim.Adam(self.gaze_actor.parameters(), lr=actor_lr)
            self.gaze_critic_optimizer = torch.optim.Adam(
                self.gaze_critic.parameters(), lr=critic_lr
            )

        if resume_checkpoint is None:
            self._initialize_from_source(source_checkpoint)
        if not self.train_action:
            self.policy.actor.eval().requires_grad_(False)
            self.policy.critic.eval().requires_grad_(False)
            self.policy.actor_optimizer = None
            self.policy.critic_optimizer = None
        if self.gaze_type == "learned" and not self.train_gaze:
            self.gaze_actor.eval().requires_grad_(False)
            self.gaze_critic.eval().requires_grad_(False)

        repository = FixedHSPPartnerRepository(
            device=self.device, checkpoint=fixed_partner_checkpoint
        )
        self.fixed_partner_provenance = repository.provenance()
        self.backends = []
        shared_oracle = None
        enable_predictor = any(condition.horizon > 0 for condition in self.scheduler.conditions)
        for env_index, condition in enumerate(self.conditions):
            backend = DirectOvercookedBackend(
                seed=self.seed + env_index, horizon=condition.horizon,
                episode_length=self.episode_length, device=self.device,
                partner=None, partner_repository=repository,
                prediction_oracle=shared_oracle, environment_index=env_index,
                enable_predictor=enable_predictor,
            )
            self.backends.append(backend)
            shared_oracle = backend.oracle
        optimizers = {}
        if self.train_action:
            optimizers.update(action_actor_optimizer=self.policy.actor_optimizer,
                              action_critic_optimizer=self.policy.critic_optimizer)
        if self.train_gaze:
            optimizers.update(gaze_actor_optimizer=self.gaze_optimizer,
                              gaze_critic_optimizer=self.gaze_critic_optimizer)
        if not self.train_action:
            assert_optimizer_isolation(self.policy.actor, optimizers)
            assert_optimizer_isolation(self.policy.critic, optimizers)
        self.action_optimizer_isolation_verified = not self.train_action
        for actor in {id(b.partners.actor): b.partners.actor for b in self.backends}.values():
            assert_optimizer_isolation(actor, optimizers)
        self.partner_optimizer_isolation_verified = True
        self.controllers = (
            [None] * self.n_rollout_threads
            if self.gaze_type == "rule_based"
            else [
                LearnedGazeController(
                    self.gaze_actor,
                    self.gaze_critic,
                    self.device,
                    env_index=env_index,
                )
                for env_index, _backend in enumerate(self.backends)
            ]
        )
        self.rule_gaze_probabilities = rule_gaze_probabilities
        self.adapters = [
            PartialObservationAdapter(
                backend, decision_time=condition.decision_time,
                fixation_shape=condition.fixation_shape, gaze_policy=controller,
                rule_gaze_seed=self.seed + 83_000_009 + env_index,
                rule_gaze_probabilities=rule_gaze_probabilities,
                memory_lifetime_ms=self.memory_lifetime_ms,
                memory_decay_mode=self.memory_decay_mode,
                memory_tau_ms=self.memory_tau_ms,
                memory_seed=(
                    self.memory_random_seed + env_index
                    if self.memory_random_seed is not None
                    else None
                ),
                constraint_config=self.constraint_config,
                peripheral_seed=(
                    self.seed + PERIPHERAL_RNG_SEED_OFFSET + env_index
                ),
                opacity_seed=(
                    self.seed + PREVIEW_OPACITY_RNG_SEED_OFFSET + env_index
                ),
                dwell_seed=(
                    self.seed + HUMAN_DWELL_RNG_SEED_OFFSET + env_index
                    if self.emma_timing.mode == "human_stochastic"
                    else None
                ),
                ground_truth_policy_input=self.ground_truth_policy_input,
                defer_preview_until_acquisition=(
                    self.inference_batching == "predictor"
                ),
            )
            for env_index, (backend, condition, controller) in enumerate(
                zip(self.backends, self.conditions, self.controllers)
            )
        ]
        self.backend, self.adapter = self.backends[0], self.adapters[0]
        action_query_contract_ids = {
            adapter.action_query_contract_id for adapter in self.adapters
        }
        recurrent_commit_contract_ids = {
            adapter.recurrent_commit_contract_id for adapter in self.adapters
        }
        held_foveal_refresh_contract_ids = {
            adapter.held_foveal_refresh_contract_id for adapter in self.adapters
        }
        if len(action_query_contract_ids) != 1:
            raise RuntimeError("Partial adapters disagree on ActionNet query contract")
        if len(recurrent_commit_contract_ids) != 1:
            raise RuntimeError("Partial adapters disagree on recurrent commit contract")
        if len(held_foveal_refresh_contract_ids) != 1:
            raise RuntimeError("Partial adapters disagree on held-foveal refresh contract")
        self.action_query_contract_id = action_query_contract_ids.pop()
        self.recurrent_commit_contract_id = recurrent_commit_contract_ids.pop()
        self.held_foveal_refresh_contract_id = (
            held_foveal_refresh_contract_ids.pop()
        )
        self.controller = self.controllers[0]
        self.buffer = SeparatedReplayBuffer(args, obs_space, share_space, action_space)
        self.buffer.masks[0].fill(0.0)
        self.buffer.active_masks[0].fill(0.0)
        self.buffer.available_actions[0].fill(1.0)
        for env_index, adapter in enumerate(self.adapters):
            obs, share, available = adapter.current()
            self.buffer.obs[0, env_index] = obs
            self.buffer.share_obs[0, env_index] = share
            self.buffer.available_actions[0, env_index] = available
        self.completed_steps = self.update_count = 0
        self.freshness_audit = FreshnessGradientAudit(self.policy.actor)
        if resume_checkpoint is not None:
            self._resume(resume_checkpoint)
        self.initial_actor_hash = parameter_hash(self.policy.actor)
        self.initial_critic_hash = parameter_hash(self.policy.critic)

    def _initialize_from_source(self, path: Path | None) -> None:
        if path is None:
            self.source_provenance = {
                "initialization": "fresh_partial",
                "source_emma_timing": None,
                "target_emma_timing": self.emma_timing.metadata(),
                "source_constraint_config": None,
                "target_constraint_config": self.constraint_config.metadata(),
            }
            return
        source, payload, metadata = _load_payload(path, self.device)
        source_constraint_config = constraint_config_from_metadata(
            metadata, require_implemented=False
        )
        source_emma_timing = source_constraint_config.emma
        _strict_cooking(metadata)
        source_mode = metadata.get("observation_mode")
        source_stage = metadata.get("training_stage")
        direct_full_joint = (
            self.direct_joint_from_full
            and source_mode == "full"
            and source_stage == "action"
        )
        full_ground_truth_gaze = (
            self.ground_truth_policy_input
            and source_mode == "full"
            and source_stage == "action"
        )
        if self.direct_joint_from_full and not direct_full_joint:
            raise RuntimeError(
                "direct joint from Full requires a Full Action source checkpoint"
            )
        expected_common = {
            "format_id": FORMAT_ID,
            "canonical_encoder_id": CANONICAL_ENCODER_ID,
            "partner_set_id": PARTNER_SET_ID,
            "action_space": list(ACTION_SPACE),
        }
        mismatch = {
            key: (metadata.get(key), expected)
            for key, expected in expected_common.items()
            if metadata.get(key) != expected
        }
        architecture = metadata.get(
            "architecture_id", metadata.get("action_architecture_id")
        )
        if architecture != ACTION_ARCHITECTURE_ID:
            mismatch["architecture_id"] = (architecture, ACTION_ARCHITECTURE_ID)
        if source_mode == "partial":
            expected_memory = {
                "memory_decay_mode": self.memory_decay_mode,
                "memory_lifetime_ms": self.memory_lifetime_ms,
                "freshness_contract_id": self.freshness_contract_id,
            }
            if self.memory_decay_mode == "stochastic_exponential":
                expected_memory.update({
                    "memory_tau_ms": self.memory_tau_ms,
                    "memory_max_lifetime_ms": self.memory_lifetime_ms,
                })
            mismatch.update({
                key: (
                    metadata.get(key, "linear")
                    if key == "memory_decay_mode"
                    else metadata.get(key),
                    expected,
                )
                for key, expected in expected_memory.items()
                if (
                    metadata.get(key, "linear")
                    if key == "memory_decay_mode"
                    else metadata.get(key)
                ) != expected
            })
            source_observation_contract = metadata.get("observation_contract_id")
            if source_observation_contract != self.memory_contract_id:
                mismatch["observation_contract_id"] = (
                    source_observation_contract, self.memory_contract_id,
                )
        if mismatch:
            raise RuntimeError(f"source checkpoint contract mismatch: {mismatch}")
        allowed_horizons = (
            set(metadata["horizons"])
            if "horizons" in metadata
            else {metadata.get("horizon")}
        )
        if not set(self.scheduler.horizons).issubset(allowed_horizons):
            raise RuntimeError("source checkpoint does not cover requested Partial horizons")
        if self.gaze_type == "rule_based":
            valid_full = source_mode == "full" and source_stage == "action"
            valid_partial = (
                source_mode == "partial"
                and metadata.get("partial_checkpoint_format_id")
                == PARTIAL_CHECKPOINT_FORMAT_ID
            )
            if not (valid_full or valid_partial):
                raise RuntimeError(
                    "rule-based Partial Action source must be a Full Action or "
                    "canonical Partial checkpoint"
                )
        elif direct_full_joint or full_ground_truth_gaze:
            pass
        elif source_mode != "partial":
            raise RuntimeError("learned gaze must initialize from a Partial checkpoint")
        elif metadata.get("partial_checkpoint_format_id") != PARTIAL_CHECKPOINT_FORMAT_ID:
            raise RuntimeError(
                "learned gaze requires canonical Partial initialization provenance"
            )
        actor_state = _source_actor_state(payload)
        self.policy.actor.load_state_dict(actor_state, strict=True)
        source_actor_hash = parameter_hash(self.policy.actor)
        expected_actor_hash = metadata.get(
            "actor_parameter_sha256", metadata.get("action_parameter_sha256")
        )
        if expected_actor_hash != source_actor_hash:
            raise RuntimeError("source ActionNet content hash mismatch")
        source_critic_hash = None
        if source_mode == "partial" or full_ground_truth_gaze:
            critic_state = payload.get("critic")
            if not isinstance(critic_state, dict) or not critic_state:
                raise RuntimeError("source checkpoint has no Action critic")
            source_critic_hash = state_dict_hash(critic_state)
            expected_critic_hash = metadata.get("critic_parameter_sha256")
            if expected_critic_hash != source_critic_hash:
                raise RuntimeError("source Action critic content hash mismatch")
            if not self.train_action or self.train_gaze:
                self.policy.critic.load_state_dict(critic_state, strict=True)
                if parameter_hash(self.policy.critic) != source_critic_hash:
                    raise RuntimeError(
                        "Partial step-0 Action critic is not bit-exact to source"
                    )
        source_gaze_hash = None
        source_gaze_critic_hash = None
        if self.gaze_type == "learned" and not self.train_gaze:
            if "gaze_actor" not in payload:
                raise RuntimeError("learned Action training requires learned-gaze initialization")
            require_current_gaze_action_contract(
                metadata, operation="fixed GazeNet initialization"
            )
            require_current_gaze_input_contract(
                metadata, operation="fixed GazeNet initialization"
            )
            require_current_gaze_credit_contract(
                metadata, operation="fixed GazeNet initialization"
            )
            self.gaze_actor.load_state_dict(payload["gaze_actor"], strict=True)
            source_gaze_hash = state_dict_hash(self.gaze_actor.state_dict())
            if metadata.get("gaze_parameter_sha256") not in {
                None, source_gaze_hash
            }:
                raise RuntimeError("source GazeNet content hash mismatch")
            if "gaze_critic" in payload:
                self.gaze_critic.load_state_dict(payload["gaze_critic"], strict=True)
                source_gaze_critic_hash = state_dict_hash(
                    self.gaze_critic.state_dict()
                )
        elif (
            self.gaze_type == "learned"
            and self.train_action
            and self.train_gaze
            and not direct_full_joint
        ):
            if "gaze_actor" not in payload:
                raise RuntimeError("joint training requires learned-gaze initialization")
            require_current_gaze_action_contract(
                metadata, operation="joint GazeNet initialization"
            )
            require_current_gaze_input_contract(
                metadata, operation="joint GazeNet initialization"
            )
            require_current_gaze_credit_contract(
                metadata, operation="joint GazeNet initialization"
            )
            self.gaze_actor.load_state_dict(payload["gaze_actor"], strict=True)
            source_gaze_hash = state_dict_hash(self.gaze_actor.state_dict())
            if metadata.get("gaze_parameter_sha256") not in {
                None, source_gaze_hash
            }:
                raise RuntimeError("source GazeNet content hash mismatch")
            if "gaze_critic" in payload:
                self.gaze_critic.load_state_dict(payload["gaze_critic"], strict=True)
                source_gaze_critic_hash = state_dict_hash(
                    self.gaze_critic.state_dict()
                )
        self.source_provenance = {
            "initialization": (
                DIRECT_JOINT_INITIALIZATION_ID
                if direct_full_joint
                else "full_ground_truth_gaze_from_full_action"
                if full_ground_truth_gaze
                else "shared_full_checkpoint"
                if source_mode == "full"
                else "partial_checkpoint"
            ),
            "direct_joint_from_full": direct_full_joint,
            "action_actor_initialization": "bit_exact_source_checkpoint",
            "action_critic_initialization": (
                "bit_exact_full_checkpoint"
                if full_ground_truth_gaze
                else "fresh_partial_critic"
                if direct_full_joint
                else "partial_checkpoint"
                if source_mode == "partial" and (
                    not self.train_action or self.train_gaze
                )
                else "fresh_partial_critic"
            ),
            "gaze_actor_initialization": (
                "fresh_gazenet"
                if direct_full_joint or full_ground_truth_gaze
                else "partial_checkpoint"
                if self.gaze_type == "learned"
                else None
            ),
            "gaze_critic_initialization": (
                "fresh_gaze_critic"
                if direct_full_joint or full_ground_truth_gaze
                else "partial_checkpoint"
                if self.gaze_type == "learned"
                else None
            ),
            "source_absolute_path": str(source),
            "source_content_sha256": file_sha256(source),
            "source_actor_parameter_sha256": source_actor_hash,
            **(
                {"source_critic_parameter_sha256": source_critic_hash}
                if source_critic_hash is not None
                else {}
            ),
            **(
                {"source_gaze_parameter_sha256": source_gaze_hash}
                if source_gaze_hash is not None
                else {}
            ),
            **(
                {
                    "source_gaze_critic_parameter_sha256": (
                        source_gaze_critic_hash
                    )
                }
                if source_gaze_critic_hash is not None
                else {}
            ),
            "source_horizon_mode": metadata.get("horizon_mode", "fixed"),
            "source_shared_horizons": metadata.get("horizons"),
            "source_horizon": metadata.get("horizon"),
            "source_dynamics_contract": cooking_contract(),
            "source_emma_timing": source_emma_timing.metadata(),
            "target_emma_timing": self.emma_timing.metadata(),
            "source_constraint_config": source_constraint_config.metadata(),
            "target_constraint_config": self.constraint_config.metadata(),
        }
        if parameter_hash(self.policy.actor) != source_actor_hash:
            raise RuntimeError("Partial step-0 actor is not bit-exact to source")

    def _contract(self) -> dict[str, Any]:
        result = {
            "training_state_format_id": TRAINING_STATE_FORMAT_ID,
            "partial_checkpoint_format_id": PARTIAL_CHECKPOINT_FORMAT_ID,
            "observation_mode": "partial", "gaze_type": self.gaze_type,
            "train_components": list(self.train_components),
            "train_action": self.train_action, "train_gaze": self.train_gaze,
            "direct_joint_from_full": self.direct_joint_from_full,
            "ground_truth_policy_input": self.ground_truth_policy_input,
            "policy_observation_contract_id": (
                FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
                if self.ground_truth_policy_input
                else self.memory_contract_id
            ),
            "inference_batching": self.inference_batching,
            "seed": self.seed, "rollout_length": self.rollout_length,
            "episode_length": self.episode_length,
            "n_rollout_threads": self.n_rollout_threads,
            "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
            "fixed_partner_identity": FIXED_HSP_PARTNER_ID,
            "fixed_partner_contract_id": FIXED_HSP_PARTNER_CONTRACT_ID,
            "configured_total_steps": self.configured_total_steps,
            "checkpoint_interval": self.checkpoint_interval,
            "reward_shaping_horizon_steps": self.reward_shaping_horizon_steps,
            "decision_times": list(self.scheduler.decision_times),
            "horizons": list(self.scheduler.horizons),
            "fixation_shapes": list(self.scheduler.fixation_shapes),
            "condition_scheduler_id": CONDITION_SCHEDULER_ID,
            "rule_gaze_probabilities": self.rule_gaze_probabilities,
            "memory_contract_id": self.memory_contract_id,
            "memory_decay_mode": self.memory_decay_mode,
            "memory_lifetime_ms": self.memory_lifetime_ms,
            "freshness_contract_id": self.freshness_contract_id,
            **self.constraint_config.metadata(),
            "acquisition_action_contract_id": self.action_query_contract_id,
            "recurrent_commit_contract_id": self.recurrent_commit_contract_id,
            **(
                {
                    "held_foveal_refresh_contract_id": (
                        self.held_foveal_refresh_contract_id
                    )
                }
                if self.held_foveal_refresh_contract_id is not None
                else {}
            ),
            "actor_available_action_contract_id": ACTOR_AVAILABLE_ACTION_CONTRACT_ID,
            "partial_ppo_contract_id": PARTIAL_PPO_CONTRACT_ID,
            "gaze_action_contract_id": (
                GAZENET_V1_ACTION_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "gaze_input_contract_id": (
                GAZENET_V1_INPUT_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "gaze_reward_contract_id": (
                GAZENET_V1_REWARD_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "gaze_bootstrap_reward_contract_id": (
                GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "gaze_credit_assignment_contract_id": (
                GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "entropy_coefs": list(self.entropy_coefs),
            "entropy_coef_horizons": list(self.entropy_coef_horizons),
            **self.ppo, **cooking_contract(),
        }
        if self.memory_decay_mode == "stochastic_exponential":
            result.update({
                "memory_tau_ms": self.memory_tau_ms,
                "memory_max_lifetime_ms": self.memory_lifetime_ms,
                "memory_random_seed": self.memory_random_seed,
                "memory_rng_contract_id": MEMORY_RNG_CONTRACT_ID,
                "memory_rng_seed_offset": MEMORY_RNG_SEED_OFFSET,
            })
        return result

    @staticmethod
    def _buffer_state(buffer) -> dict[str, Any]:
        names = (
            "share_obs", "obs", "rnn_states", "rnn_states_critic", "value_preds",
            "returns", "actions", "action_log_probs", "rewards", "masks",
            "bad_masks", "active_masks", "available_actions", "transition_ids",
        )
        return {name: getattr(buffer, name).copy() for name in names} | {"step": buffer.step}

    def _training_state(self) -> dict[str, Any]:
        state = {
            "format_id": TRAINING_STATE_FORMAT_ID, "contract": self._contract(),
            "completed_steps": self.completed_steps, "update_count": self.update_count,
            "action_actor_optimizer": (
                None
                if self.policy.actor_optimizer is None
                else self.policy.actor_optimizer.state_dict()
            ),
            "action_critic_optimizer": (
                None
                if self.policy.critic_optimizer is None
                else self.policy.critic_optimizer.state_dict()
            ),
            "value_normalizer": self.trainer.value_normalizer.state_dict(),
            "gaze_actor_optimizer": None if self.gaze_optimizer is None else self.gaze_optimizer.state_dict(),
            "gaze_critic_optimizer": None if self.gaze_critic_optimizer is None else self.gaze_critic_optimizer.state_dict(),
            "backends": [backend.training_state() for backend in self.backends],
            "adapters": [adapter.training_state() for adapter in self.adapters],
            "conditions": [condition.__dict__ for condition in self.conditions],
            "condition_decision_counts": [
                {**condition.__dict__, "count": count}
                for condition, count in self.condition_decision_counts.items()
            ],
            "scheduler": self.scheduler.state_dict(), "buffer": self._buffer_state(self.buffer),
            "rng": capture_global_rng_state(),
        }
        return state

    def _resume(self, path: Path) -> None:
        _source, payload, metadata = _load_payload(path, self.device)
        checkpoint_constraint_config = constraint_config_from_metadata(metadata)
        if checkpoint_constraint_config != self.constraint_config:
            raise RuntimeError(
                "Partial resume constraint config mismatch: "
                f"checkpoint={checkpoint_constraint_config.metadata()}, "
                f"requested={self.constraint_config.metadata()}"
            )
        if metadata.get("partial_checkpoint_format_id") != PARTIAL_CHECKPOINT_FORMAT_ID:
            raise RuntimeError("resume requires a canonical Partial v2 checkpoint")
        if self.gaze_type == "learned":
            require_current_gaze_action_contract(
                metadata, operation="GazeNet training resume"
            )
            require_current_gaze_input_contract(
                metadata, operation="GazeNet training resume"
            )
            require_current_gaze_credit_contract(
                metadata, operation="GazeNet training resume"
            )
        expected_partner = {
            key: self.fixed_partner_provenance[key]
            for key in (
                "fixed_partner_sha256",
                "fixed_partner_identity",
                "fixed_partner_contract_id",
            )
        }
        partner_mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected_partner.items()
            if metadata.get(key) != value
        }
        if partner_mismatch:
            raise RuntimeError(
                f"Partial resume fixed-partner contract mismatch: {partner_mismatch}"
            )
        state = payload.get("training_state")
        if not isinstance(state, dict):
            raise RuntimeError("Partial resume checkpoint has no complete training_state")
        actual_contract = state.get("contract")
        if isinstance(actual_contract, dict):
            actual_contract = dict(actual_contract)
            for key in self.constraint_config.metadata():
                actual_contract.pop(key, None)
            actual_contract.update(checkpoint_constraint_config.metadata())
        validate_resume_contract(actual_contract, self._contract())
        self.policy.actor.load_state_dict(payload["actor"], strict=True)
        self.policy.critic.load_state_dict(payload["critic"], strict=True)
        if self.gaze_type == "learned":
            self.gaze_actor.load_state_dict(payload["gaze_actor"], strict=True)
            self.gaze_critic.load_state_dict(payload["gaze_critic"], strict=True)
        if self.train_action:
            self.policy.actor_optimizer.load_state_dict(
                state["action_actor_optimizer"]
            )
            self.policy.critic_optimizer.load_state_dict(
                state["action_critic_optimizer"]
            )
            optimizer_to(self.policy.actor_optimizer, self.device)
            optimizer_to(self.policy.critic_optimizer, self.device)
        elif not self.train_action and (
            state.get("action_actor_optimizer") is not None
            or state.get("action_critic_optimizer") is not None
        ):
            raise RuntimeError(
                "frozen ActionNet resume checkpoint contains Action optimizer state"
            )
        self.trainer.value_normalizer.load_state_dict(state["value_normalizer"])
        if self.gaze_optimizer is not None:
            self.gaze_optimizer.load_state_dict(state["gaze_actor_optimizer"])
            self.gaze_critic_optimizer.load_state_dict(state["gaze_critic_optimizer"])
            optimizer_to(self.gaze_optimizer, self.device)
            optimizer_to(self.gaze_critic_optimizer, self.device)
        self.scheduler.load_state_dict(state["scheduler"])
        if len(state["backends"]) != self.n_rollout_threads:
            raise RuntimeError("Partial resume environment count mismatch")
        resumed_conditions = [
            PartialCondition(**value) for value in state["conditions"]
        ]
        if len(resumed_conditions) != self.n_rollout_threads:
            raise RuntimeError("Partial resume active-condition count mismatch")
        for backend, adapter, condition, backend_saved, adapter_saved in zip(
            self.backends,
            self.adapters,
            resumed_conditions,
            state["backends"],
            state["adapters"],
        ):
            # A newly constructed shared-condition backend starts from the
            # scheduler's initial horizon. Restore the checkpoint's active
            # horizon before validating/loading its training state.
            backend.set_horizon(condition.horizon)
            adapter.set_condition(
                decision_time=condition.decision_time,
                horizon=condition.horizon,
                fixation_shape=condition.fixation_shape,
            )
            backend.load_training_state(backend_saved)
            adapter.load_training_state(adapter_saved)
        self.conditions = resumed_conditions
        restored_counts = {
            PartialCondition(
                int(value["decision_time"]), int(value["horizon"]),
                str(value["fixation_shape"]),
            ): int(value["count"])
            for value in state["condition_decision_counts"]
        }
        if set(restored_counts) != set(self.condition_decision_counts):
            raise RuntimeError("Partial resume condition-decision counts mismatch")
        self.condition_decision_counts = restored_counts
        for name, saved in state["buffer"].items():
            if name == "step":
                self.buffer.step = int(saved)
                continue
            target = getattr(self.buffer, name)
            source = np.asarray(saved, dtype=target.dtype)
            if source.shape != target.shape:
                raise RuntimeError(f"Partial resume buffer {name} shape mismatch")
            target[...] = source
        self.completed_steps = int(state["completed_steps"])
        self.update_count = int(state["update_count"])
        update_size = self.rollout_length * self.n_rollout_threads
        if self.completed_steps <= 0 or self.completed_steps % update_size:
            raise RuntimeError("invalid Partial resume progress")
        if self.update_count != self.completed_steps // update_size:
            raise RuntimeError("Partial resume update count mismatch")
        if sum(backend.environment_steps for backend in self.backends) != self.completed_steps:
            raise RuntimeError("Partial resume environment-step mismatch")
        self.source_provenance = metadata.get(
            "source_checkpoint_provenance",
            metadata.get("source_full_checkpoint_provenance"),
        )
        restore_global_rng_state(state["rng"])

    def _proposal_callback(self, env_index, pre_hidden, recurrent_mask):
        def callback(observation, _critic, available, timestamp_ms):
            started = time.perf_counter()
            with torch.no_grad():
                action, log_prob, candidate = self.policy.actor(
                    observation[None],
                    pre_hidden[None],
                    recurrent_mask[None],
                    available[None],
                    deterministic=False,
                )
            result = {
                "action": int(action.item()), "log_prob": float(log_prob.item()),
                "candidate_hidden": candidate.detach().cpu().numpy()[0].copy(),
                "pre_decision_hidden": pre_hidden.copy(),
                "recurrent_mask": recurrent_mask.copy(),
                "available_actions": available.copy(),
                "timestamp_ms": float(timestamp_ms),
            }
            self._actionnet_rollout_seconds += time.perf_counter() - started
            self._actionnet_rollout_forward_count += 1
            if self._parity_action_records is not None:
                self._parity_action_records.append({
                    "environment_index": int(env_index),
                    "observation": np.ascontiguousarray(observation).copy(),
                    "pre_decision_hidden": pre_hidden.copy(),
                    "recurrent_mask": recurrent_mask.copy(),
                    "available_actions": available.copy(),
                    "action": int(result["action"]),
                    "log_prob": float(result["log_prob"]),
                    "candidate_hidden": result["candidate_hidden"].copy(),
                    "timestamp_ms": float(timestamp_ms),
                })
            return result
        return callback

    @staticmethod
    def _predictor_inputs(backends):
        return (
            [backend.state for backend in backends],
            [backend.mdp for backend in backends],
            [backend.self_history for backend in backends],
            [backend.partner_history for backend in backends],
        )

    def _record_partial_predictor_parity(
        self,
        backends,
        batch_predictions,
        batch_logits,
        legacy_predictions,
        legacy_logits,
    ) -> None:
        if self.predictor_parity is not None:
            return
        difference = np.abs(
            np.stack(legacy_logits).astype(np.float64)
            - np.asarray(batch_logits, dtype=np.float64)
        )
        preview_equal = []
        action_equal = []
        first_divergence = None
        for row, (backend, legacy, batched) in enumerate(
            zip(backends, legacy_predictions, batch_predictions)
        ):
            horizon = int(backend.horizon)
            preview_match = np.array_equal(
                legacy.displayed_prediction_gt[:horizon],
                batched.displayed_prediction_gt[:horizon],
            )
            action_match = (
                legacy.action_indices[:horizon]
                == batched.action_indices[:horizon]
            )
            preview_equal.append(preview_match)
            action_equal.append(action_match)
            if first_divergence is None and not preview_match:
                first_divergence = {
                    "row": row,
                    "environment_index": int(backend.environment_index),
                    "horizon": horizon,
                }
        self.predictor_parity = {
            "environment_count": len(backends),
            "environment_indices": [
                int(backend.environment_index) for backend in backends
            ],
            "env_mapping_exact": True,
            "h0_predictor_forward_count": 0,
            "predictor_max_abs_logit_difference": float(difference.max()),
            "predictor_mean_abs_logit_difference": float(difference.mean()),
            "displayed_action_exact": bool(all(action_equal)),
            "displayed_preview_exact": bool(all(preview_equal)),
            "first_displayed_preview_divergence": first_divergence,
        }
        if not all(preview_equal):
            raise RuntimeError("Partial predictor batch preview parity failed")

    def _populate_partial_previews(self) -> tuple[float, int, int]:
        if self.inference_batching != "predictor":
            return 0.0, 0, 0
        for backend in self.backends:
            if backend.horizon == 0 and backend._preview_cache is None:
                backend.set_batched_preview(
                    np.zeros((4, 8, 5), dtype=np.float32),
                    generation=backend.state_generation,
                )
        backends = [
            backend
            for backend in self.backends
            if backend.horizon > 0 and backend._preview_cache is None
        ]
        if not backends:
            return 0.0, 0, 0
        expected = sorted(int(backend.environment_index) for backend in backends)
        actual = [int(backend.environment_index) for backend in backends]
        if actual != expected:
            raise RuntimeError("Partial predictor environment ordering changed")
        oracle = backends[0].oracle
        if oracle is None or any(backend.oracle is not oracle for backend in backends):
            raise RuntimeError("Partial predictor batching requires one shared oracle")
        inputs = self._predictor_inputs(backends)
        legacy_predictions = legacy_logits = None
        if self.inference_shadow_audit and self.predictor_parity is None:
            legacy_predictions = []
            legacy_logits = []
            for row in range(len(backends)):
                predictions, logits = oracle.predict_batch_with_logits(
                    (inputs[0][row],),
                    (inputs[1][row],),
                    (inputs[2][row],),
                    (inputs[3][row],),
                    int(backends[row].horizon),
                )
                legacy_predictions.append(predictions[0])
                legacy_logits.append(logits[0])
        started = time.perf_counter()
        predictions, logits = oracle.predict_batch_with_logits(*inputs, 4)
        elapsed = time.perf_counter() - started
        for backend, prediction in zip(backends, predictions):
            backend.set_batched_preview(
                prediction.displayed_prediction_gt,
                generation=backend.state_generation,
            )
        if legacy_predictions is not None and legacy_logits is not None:
            self._record_partial_predictor_parity(
                backends,
                predictions,
                logits,
                legacy_predictions,
                legacy_logits,
            )
        return elapsed, 1, len(backends)

    def _audit_batch1_action_log_probs(
        self,
        observations,
        rnn_states,
        actions,
        masks,
        available_actions,
    ) -> torch.Tensor:
        """Replay active transitions in the exact batch-1 rollout context."""
        count = len(observations)
        if count <= 0 or any(
            len(value) != count
            for value in (rnn_states, actions, masks, available_actions)
        ):
            raise RuntimeError("invalid Partial batch-1 audit replay inputs")
        logs = []
        with torch.no_grad():
            for index in range(count):
                log_prob, _entropy, _ = self.policy.actor.evaluate_actions(
                    observations[index : index + 1],
                    rnn_states[index : index + 1],
                    actions[index : index + 1],
                    masks[index : index + 1],
                    available_actions[index : index + 1],
                    np.ones((1, 1), np.float32),
                )
                logs.append(log_prob)
        return torch.cat(logs, dim=0)

    @staticmethod
    def _audit_digest(value) -> str:
        digest = hashlib.sha256()

        def update(item) -> None:
            if isinstance(item, torch.Tensor):
                update(item.detach().cpu().contiguous().numpy())
            elif isinstance(item, np.ndarray):
                array = np.ascontiguousarray(item)
                digest.update(b"array")
                digest.update(array.dtype.str.encode("ascii"))
                digest.update(repr(array.shape).encode("ascii"))
                digest.update(array.tobytes())
            elif isinstance(item, dict):
                digest.update(b"dict")
                for key in sorted(item, key=lambda current: repr(current)):
                    update(key)
                    update(item[key])
            elif isinstance(item, (list, tuple)):
                digest.update(type(item).__name__.encode("ascii"))
                for nested in item:
                    update(nested)
            elif isinstance(item, (str, bytes, int, float, bool, type(None))):
                digest.update(type(item).__name__.encode("ascii"))
                digest.update(repr(item).encode("utf-8"))
            else:
                digest.update(type(item).__qualname__.encode("utf-8"))
                digest.update(
                    pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
                )

        update(value)
        return digest.hexdigest()

    def _gaze_preupdate_audit(
        self,
        records,
        *,
        actor=None,
        device=None,
    ) -> dict[str, float]:
        if not records:
            raise RuntimeError("Gaze pre-update audit requires completed transitions")
        audit_actor = self.gaze_actor if actor is None else actor
        audit_device = self.device if device is None else torch.device(device)
        differences = []
        ratios = []
        with torch.no_grad():
            for record in records:
                distribution = audit_actor.distribution(
                    _as_tensor(record["observation"][None], audit_device),
                    _as_tensor(record["context"][None], audit_device),
                )
                action = torch.as_tensor(
                    [record["action"]], dtype=torch.long, device=audit_device
                )
                current = float(distribution.log_prob(action).item())
                old = float(record["log_prob"])
                differences.append(abs(current - old))
                ratios.append(float(np.exp(current - old)))
        max_difference = max(differences)
        if max_difference > PREUPDATE_LOGPROB_ATOL:
            raise RuntimeError(
                "Partial pre-update GazeNet log-prob parity failed | "
                f"max_abs={max_difference:.9g} tol={PREUPDATE_LOGPROB_ATOL:.9g}"
            )
        ratio_array = np.asarray(ratios, dtype=np.float64)
        return {
            "gaze_preupdate_parity_pass": 1.0,
            "gaze_preupdate_transition_count": len(records),
            "gaze_preupdate_max_abs_logprob_difference": float(max_difference),
            "gaze_preupdate_mean_abs_logprob_difference": float(
                np.mean(differences)
            ),
            "gaze_preupdate_ratio_min": float(ratio_array.min()),
            "gaze_preupdate_ratio_max": float(ratio_array.max()),
            "gaze_preupdate_approx_kl": float(
                np.mean((ratio_array - 1.0) - np.log(ratio_array))
            ),
            "gaze_preupdate_logprob_tolerance": PREUPDATE_LOGPROB_ATOL,
        }

    def _capture_parity_snapshot(
        self,
        gaze_records,
        transition_records,
        *,
        gaze_forward_count: int,
    ):
        advantages, returns = gaze_advantages(
            gaze_records,
            self.ppo["gamma"],
            self.ppo["gae_lambda"],
        )
        buffer_state = self._buffer_state(self.buffer)
        return {
            "environment": self._audit_digest([
                backend.training_state() for backend in self.backends
            ]),
            "gaze_memory_timing": self._audit_digest([
                adapter.training_state() for adapter in self.adapters
            ]),
            "frozen_actionnet_rollout": self._audit_digest(
                self._parity_action_records
            ),
            "partner_and_transition": self._audit_digest(transition_records),
            "ppo_action_buffer": self._audit_digest(buffer_state),
            "ppo_gaze_records": self._audit_digest(gaze_records),
            "ppo_gaze_advantages": self._audit_digest(advantages),
            "ppo_gaze_returns": self._audit_digest(returns),
            "rng_preupdate": self._audit_digest(capture_global_rng_state()),
            "actionnet_forward_count": self._actionnet_rollout_forward_count,
            "gaze_forward_count": int(gaze_forward_count),
        }

    def _preupdate_audit(self) -> dict[str, float]:
        active = self.buffer.active_masks[:-1, :, 0] > 0.5
        count = int(np.count_nonzero(active))
        if not count:
            return {
                "preupdate_parity_pass": 1.0,
                "preupdate_transition_count": 0,
                "preupdate_max_abs_logprob_difference": 0.0,
                "preupdate_mean_abs_logprob_difference": 0.0,
                "preupdate_ratio_min": 1.0, "preupdate_ratio_max": 1.0,
                "preupdate_ratio_mean": 1.0, "preupdate_approx_kl": 0.0,
                "preupdate_clip_fraction": 0.0,
                "preupdate_observation_exact_match": 1.0,
                "preupdate_rnn_state_exact_match": 1.0,
                "preupdate_action_exact_match": 1.0,
                "preupdate_legal_mask_exact_match": 1.0,
                "preupdate_logprob_tolerance": PREUPDATE_LOGPROB_ATOL,
            }
        logs = self._audit_batch1_action_log_probs(
            self.buffer.obs[:-1][active],
            self.buffer.rnn_states[:-1][active],
            self.buffer.actions[active],
            self.buffer.masks[:-1][active],
            self.buffer.available_actions[:-1][active],
        )
        current = logs.detach().cpu().numpy()
        old = self.buffer.action_log_probs[active]
        difference = np.abs(current - old)
        ratio = np.exp(current - old)
        approx_kl, clipped = ppo_health_metrics(
            torch.as_tensor(ratio), self.ppo["clip_param"]
        )
        max_difference = float(difference.max())
        mean_difference = float(difference.mean())
        if max_difference > PREUPDATE_LOGPROB_ATOL:
            raise RuntimeError(
                "Partial pre-update ActionNet log-prob parity failed | "
                f"max_abs={max_difference:.9g} "
                f"mean_abs={mean_difference:.9g} "
                f"ratio_min={float(ratio.min()):.9g} "
                f"ratio_max={float(ratio.max()):.9g} "
                f"tol={PREUPDATE_LOGPROB_ATOL:.9g}"
            )
        return {
            "preupdate_parity_pass": 1.0,
            "preupdate_transition_count": count,
            "preupdate_max_abs_logprob_difference": float(difference.max()),
            "preupdate_mean_abs_logprob_difference": float(difference.mean()),
            "preupdate_ratio_min": float(ratio.min()),
            "preupdate_ratio_max": float(ratio.max()),
            "preupdate_ratio_mean": float(ratio.mean()),
            "preupdate_approx_kl": float(approx_kl.mean()),
            "preupdate_clip_fraction": float(clipped.mean()),
            "preupdate_observation_exact_match": 1.0,
            "preupdate_rnn_state_exact_match": 1.0,
            "preupdate_action_exact_match": 1.0,
            "preupdate_legal_mask_exact_match": 1.0,
            "preupdate_logprob_tolerance": PREUPDATE_LOGPROB_ATOL,
        }

    def _action_ppo_update(self, upcoming_steps: int) -> dict[str, float]:
        if not self.train_action:
            raise RuntimeError("Action PPO update called while train_action=False")

        # Rollout intentionally runs ActionNet in eval mode.  cuDNN recurrent
        # modules must be returned to training mode BEFORE the differentiable
        # PPO forward that will later be backpropagated.
        self.policy.actor.train()
        self.policy.critic.train()

        active = self.buffer.active_masks[:-1, :, 0] > 0.5
        actor_count = int(np.count_nonzero(active))
        returns = self.buffer.returns[:-1]
        denormalized_values = self.trainer.value_normalizer.denormalize(
            self.buffer.value_preds[:-1]
        )
        advantages = returns - denormalized_values
        active_advantages = advantages[active]
        if actor_count:
            active_advantages = (active_advantages - active_advantages.mean()) / (
                active_advantages.std() + 1e-5
            )
        all_count = self.rollout_length * self.n_rollout_threads
        actor_batch = max(1, int(np.ceil(actor_count / self.ppo["num_mini_batches"])))
        critic_batch = max(1, int(np.ceil(all_count / self.ppo["num_mini_batches"])))
        entropy_coef = entropy_coefficient(
            upcoming_steps, self.entropy_coefs, self.entropy_coef_horizons
        )
        metrics = {key: [] for key in (
            "policy_loss", "value_loss", "dist_entropy", "actor_grad_norm",
            "critic_grad_norm", "approx_kl", "clip_fraction",
        )}
        actor_arrays = {
            "obs": self.buffer.obs[:-1][active],
            "rnn": self.buffer.rnn_states[:-1][active],
            "actions": self.buffer.actions[active],
            "masks": self.buffer.masks[:-1][active],
            "available": self.buffer.available_actions[:-1][active],
            "old": self.buffer.action_log_probs[active],
            "adv": active_advantages,
        }
        critic_arrays = {
            "share": self.buffer.share_obs[:-1].reshape(all_count, 8, 5, 40),
            "rnn": self.buffer.rnn_states_critic[:-1].reshape(all_count, 1, 64),
            "masks": self.buffer.masks[:-1].reshape(all_count, 1),
            "old_values": self.buffer.value_preds[:-1].reshape(all_count, 1),
            "returns": returns.reshape(all_count, 1),
        }
        for _ in range(self.ppo["ppo_epochs"]):
            for ids in torch.randperm(actor_count).split(actor_batch):
                index = ids.numpy()
                logs, entropy, _ = self.policy.actor.evaluate_actions(
                    actor_arrays["obs"][index], actor_arrays["rnn"][index],
                    actor_arrays["actions"][index], actor_arrays["masks"][index],
                    actor_arrays["available"][index], np.ones((len(index), 1), np.float32),
                )
                old = _as_tensor(actor_arrays["old"][index], self.device)
                advantage = _as_tensor(actor_arrays["adv"][index], self.device)
                ratio = torch.exp(logs - old)
                clipped_ratio = ratio.clamp(
                    1.0 - self.ppo["clip_param"], 1.0 + self.ppo["clip_param"]
                )
                policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
                self.policy.actor_optimizer.zero_grad(set_to_none=True)
                (policy_loss - entropy_coef * entropy).backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.ppo["max_grad_norm"]
                )
                self.policy.actor_optimizer.step()
                kl, clipped = ppo_health_metrics(ratio.detach(), self.ppo["clip_param"])
                for key, value in (
                    ("policy_loss", policy_loss), ("dist_entropy", entropy),
                    ("actor_grad_norm", actor_grad), ("approx_kl", kl.mean()),
                    ("clip_fraction", clipped.mean()),
                ):
                    metrics[key].append(float(torch.as_tensor(value).detach().cpu()))
            for ids in torch.randperm(all_count).split(critic_batch):
                index = ids.numpy()
                values, _ = self.policy.critic(
                    critic_arrays["share"][index], critic_arrays["rnn"][index],
                    critic_arrays["masks"][index],
                )
                value_loss = self.trainer.cal_value_loss(
                    self.trainer.value_normalizer, values,
                    _as_tensor(critic_arrays["old_values"][index], self.device),
                    _as_tensor(critic_arrays["returns"][index], self.device),
                    torch.ones((len(index), 1), device=self.device),
                )
                self.policy.critic_optimizer.zero_grad(set_to_none=True)
                (value_loss * self.trainer.value_loss_coef).backward()
                critic_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.ppo["max_grad_norm"]
                )
                self.policy.critic_optimizer.step()
                metrics["value_loss"].append(float(value_loss.detach().cpu()))
                metrics["critic_grad_norm"].append(
                    float(torch.as_tensor(critic_grad).detach().cpu())
                )
        result = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in metrics.items()
        }
        result.update(entropy_coef=entropy_coef, grad_clip_fraction=float(
            np.mean([
                value > self.ppo["max_grad_norm"]
                for key in ("actor_grad_norm", "critic_grad_norm")
                for value in metrics[key]
            ])
        ))
        return result

    def _clear_after_update(self) -> None:
        self.buffer.after_update()
        for name in (
            "share_obs", "obs", "rnn_states", "rnn_states_critic",
            "available_actions", "masks", "bad_masks", "active_masks",
        ):
            array = getattr(self.buffer, name)
            array[1:].fill(1.0 if name in {"available_actions", "masks", "bad_masks", "active_masks"} else 0.0)
        for name in ("value_preds", "returns", "actions", "action_log_probs", "rewards"):
            getattr(self.buffer, name).fill(0.0)
        self.buffer.transition_ids.fill(-1)
        self.buffer.step = 0

    def run_update(self) -> dict[str, Any]:
        update_wall_started = time.perf_counter()
        actor_before = parameter_hash(self.policy.actor)
        critic_before = parameter_hash(self.policy.critic)
        gaze_before = None if self.gaze_actor is None else state_dict_hash(self.gaze_actor.state_dict())
        freshness_before = self.freshness_audit.begin()
        metric_before = [copy.deepcopy(adapter.metrics()) for adapter in self.adapters]
        guard_before = sum(b.partner_step_guard_checks for b in self.backends)
        full_guard_before = sum(b.partner_full_hash_checks for b in self.backends)
        for backend in self.backends:
            backend.verify_partner_parameters()
        optimization = sparse = shaped = weighted = 0.0
        deliveries = episodes = 0
        partial_metric_totals = new_partial_metric_totals(
            self.scheduler.conditions
        )
        gaze_records = []
        gaze_reward_before = sum(
            controller.total_attributed_optimization_reward
            for controller in self.controllers
            if controller is not None
        )
        gaze_boundaries_before = sum(
            controller.total_env_boundaries
            for controller in self.controllers
            if controller is not None
        )
        bootstrap_count_before = sum(
            controller.total_bootstrap_unowned_reward_count
            for controller in self.controllers
            if controller is not None
        )
        bootstrap_reward_before = sum(
            controller.total_bootstrap_unowned_reward_sum
            for controller in self.controllers
            if controller is not None
        )
        transition_records = [] if self.parity_audit else None
        self._parity_action_records = [] if self.parity_audit else None
        self._actionnet_rollout_seconds = 0.0
        self._actionnet_rollout_forward_count = 0
        predictor_seconds = 0.0
        predictor_forward_count = 0
        predictor_environment_count = 0
        partner_seconds = 0.0
        environment_seconds = 0.0
        rollout_bookkeeping_seconds = 0.0
        for controller in self.controllers:
            if controller is not None:
                controller.consume_inference_profile()
        for backend in self.backends:
            backend.consume_predictor_profile()
        started = time.perf_counter()
        self.policy.actor.eval()
        self.policy.critic.eval()
        for rollout_step in range(self.rollout_length):
            elapsed, forwards, environments = self._populate_partial_previews()
            predictor_seconds += elapsed
            predictor_forward_count += forwards
            predictor_environment_count += environments
            decision_conditions = tuple(self.conditions)
            global_before = self.completed_steps + rollout_step * self.n_rollout_threads
            factor = hsp_reward_shaping_factor(
                global_before, self.reward_shaping_horizon_steps
            )
            for backend in self.backends:
                backend.set_reward_shaping_factor(factor)
            step = self.buffer.step
            results = []
            decision_acquisitions = []
            decision_actionnet_queries = []
            decision_actor_active = []
            decision_gaze_requests = []
            decision_same_targets = []
            decision_new_targets = []
            decision_valid_memory_tiles = []
            decision_freshness_sample_counts = []
            decision_freshness_sums = []
            decision_saccade_distance_sums = []
            next_actor = np.zeros_like(self.buffer.rnn_states[step])
            next_critic = np.zeros_like(self.buffer.rnn_states_critic[step])
            next_masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
            bad_masks = np.ones_like(next_masks)
            for env_index, (adapter, controller) in enumerate(zip(self.adapters, self.controllers)):
                self.condition_decision_counts[self.conditions[env_index]] += 1
                task_decision_id = int(adapter.decision_index)
                if controller is not None:
                    controller.begin_task_decision(task_decision_id)
                gaze_requests_before = adapter.gaze_requests
                same_targets_before = adapter.same_target_acquisitions
                new_targets_before = adapter.new_target_acquisitions
                distance_before = adapter.distance_sum_degrees
                pre_hidden = self.buffer.rnn_states[step, env_index].copy()
                recurrent_mask = self.buffer.masks[step, env_index].copy()
                collected = adapter.collect_decision(
                    self._proposal_callback(env_index, pre_hidden, recurrent_mask)
                )
                bookkeeping_started = time.perf_counter()
                latest = collected["latest_proposal"]
                actor_active = latest is not None
                observation = collected["observation"] if latest is None else latest["observation"]
                action = STAY_ACTION_INDEX if latest is None else int(latest["action"])
                log_prob = 0.0 if latest is None else float(latest["log_prob"])
                candidate = pre_hidden if latest is None else latest["candidate_hidden"]
                with torch.no_grad():
                    value, critic_candidate = self.policy.critic(
                        collected["critic"][None],
                        self.buffer.rnn_states_critic[step, env_index][None],
                        recurrent_mask[None],
                    )
                self.buffer.obs[step, env_index] = observation.copy()
                self.buffer.share_obs[step, env_index] = collected["critic"].copy()
                self.buffer.available_actions[step, env_index] = collected["available_actions"]
                self.buffer.active_masks[step, env_index, 0] = float(actor_active)
                self.buffer.actions[step, env_index, 0] = action
                self.buffer.action_log_probs[step, env_index, 0] = log_prob
                self.buffer.value_preds[step, env_index] = value.detach().cpu().numpy()[0]
                self.buffer.transition_ids[step, env_index] = (
                    rollout_step * self.n_rollout_threads + env_index
                )
                partner_observation, partner_available = adapter.backend.partner_batch_input()
                partner_hash_before = adapter.backend.begin_partner_step()
                rollout_bookkeeping_seconds += (
                    time.perf_counter() - bookkeeping_started
                )
                partner_started = time.perf_counter()
                partner_action = adapter.backend.partners.action(
                    partner_observation,
                    partner_available,
                )
                partner_seconds += time.perf_counter() - partner_started
                environment_started = time.perf_counter()
                result = adapter.commit_collected(
                    collected,
                    partner_action=partner_action,
                    partner_hash_before=partner_hash_before,
                )
                environment_seconds += time.perf_counter() - environment_started
                bookkeeping_started = time.perf_counter()
                results.append(result)
                if transition_records is not None:
                    transition_records.append({
                        "rollout_step": int(rollout_step),
                        "environment_index": int(env_index),
                        "learner_action": int(result["executed_action"]),
                        "partner_action": int(partner_action),
                        "optimization_reward": float(result["optimization_reward"]),
                        "sparse_team_reward": float(result["sparse_team_reward"]),
                        "native_shaped_reward": float(result["native_shaped_reward"]),
                        "weighted_shaped_reward": float(result["weighted_shaped_reward"]),
                        "deliveries": int(result["deliveries"]),
                        "done": bool(result["done"]),
                    })
                observation_stats = partial_observation_statistics(
                    tuple(
                        proposal["observation"]
                        for proposal in collected["proposals"]
                    )
                )
                decision_acquisitions.append(int(collected["acquisitions"]))
                decision_actionnet_queries.append(len(collected["proposals"]))
                decision_actor_active.append(actor_active)
                decision_gaze_requests.append(
                    int(adapter.gaze_requests - gaze_requests_before)
                )
                decision_same_targets.append(
                    int(adapter.same_target_acquisitions - same_targets_before)
                )
                decision_new_targets.append(
                    int(adapter.new_target_acquisitions - new_targets_before)
                )
                decision_valid_memory_tiles.append(
                    int(observation_stats["valid_memory_tiles"])
                )
                decision_freshness_sample_counts.append(
                    int(observation_stats["freshness_sample_count"])
                )
                decision_freshness_sums.append(
                    float(observation_stats["freshness_normalized_sum"])
                )
                decision_saccade_distance_sums.append(
                    float(adapter.distance_sum_degrees - distance_before)
                )
                next_actor[env_index] = candidate
                next_critic[env_index] = critic_candidate.detach().cpu().numpy()[0]
                if controller is not None:
                    controller.accrue(
                        optimization_reward=result["optimization_reward"],
                        sparse_team_reward=result["sparse_team_reward"],
                        native_shaped_reward=result["native_shaped_reward"],
                        weighted_shaped_reward=result["weighted_shaped_reward"],
                        deliveries=result["deliveries"],
                        task_decision_id=task_decision_id,
                    )
                if result["done"]:
                    episodes += 1
                    if controller is not None:
                        controller.finish_episode(adapter.current_time_ms)
                    next_condition = self.scheduler.next()
                    self.conditions[env_index] = next_condition
                    adapter.reset_episode(next_condition)
                    next_actor[env_index].fill(0.0)
                    next_critic[env_index].fill(0.0)
                    next_masks[env_index].fill(0.0)
                    bad_masks[env_index].fill(0.0)
                if controller is not None:
                    gaze_records.extend(controller.drain())
                rollout_bookkeeping_seconds += (
                    time.perf_counter() - bookkeeping_started
                )
            bookkeeping_started = time.perf_counter()
            for result in results:
                optimization += float(result["optimization_reward"])
                sparse += float(result["sparse_team_reward"])
                shaped += float(result["native_shaped_reward"])
                weighted += float(result["weighted_shaped_reward"])
                deliveries += int(result["deliveries"])
            accumulate_partial_metrics(
                partial_metric_totals,
                conditions=decision_conditions,
                rewards=np.asarray([
                    [
                        result["optimization_reward"],
                        result["sparse_team_reward"],
                    ]
                    for result in results
                ], dtype=np.float64),
                deliveries=[result["deliveries"] for result in results],
                dones=[result["done"] for result in results],
                acquisitions=decision_acquisitions,
                actor_active=decision_actor_active,
                actionnet_queries=decision_actionnet_queries,
                gaze_requests=decision_gaze_requests,
                same_target_acquisitions=decision_same_targets,
                new_target_acquisitions=decision_new_targets,
                valid_memory_tiles=decision_valid_memory_tiles,
                freshness_sample_counts=decision_freshness_sample_counts,
                freshness_normalized_sums=decision_freshness_sums,
                saccade_distance_sums_degrees=decision_saccade_distance_sums,
                require_acquisition_query_match=(
                    self.emma_timing.mode != "human_stochastic"
                    and self.emma_timing.mode not in VISUAL_ENCODING_EMMA_MODES
                ),
            )
            next_obs, next_share, next_available = [], [], []
            for adapter in self.adapters:
                current = adapter.current()
                next_obs.append(current[0]); next_share.append(current[1]); next_available.append(current[2])
            self.buffer.insert(
                np.stack(next_share), np.stack(next_obs), next_actor, next_critic,
                self.buffer.actions[step].copy(), self.buffer.action_log_probs[step].copy(),
                self.buffer.value_preds[step].copy(),
                np.asarray([[r["optimization_reward"]] for r in results], np.float32),
                next_masks, bad_masks=bad_masks,
                active_masks=np.ones_like(next_masks), available_actions=np.stack(next_available),
            )
            rollout_bookkeeping_seconds += time.perf_counter() - bookkeeping_started
        collection_seconds = time.perf_counter() - started
        for backend in self.backends:
            legacy_seconds, legacy_forwards = backend.consume_predictor_profile()
            predictor_seconds += legacy_seconds
            predictor_forward_count += legacy_forwards
            predictor_environment_count += legacy_forwards
        gaze_inference_seconds = 0.0
        gaze_forward_count = 0
        for controller in self.controllers:
            if controller is None:
                continue
            seconds, forwards = controller.consume_inference_profile()
            gaze_inference_seconds += seconds
            gaze_forward_count += forwards
        observation_memory_seconds = max(
            0.0,
            collection_seconds
            - self._actionnet_rollout_seconds
            - gaze_inference_seconds
            - predictor_seconds
            - partner_seconds
            - environment_seconds
            - rollout_bookkeeping_seconds,
        )
        for backend in self.backends:
            backend.verify_partner_parameters()
        with torch.no_grad():
            next_value = self.policy.get_values(
                self.buffer.share_obs[-1], self.buffer.rnn_states_critic[-1], self.buffer.masks[-1]
            )
        self.buffer.compute_returns(
            next_value.detach().cpu().numpy(), self.trainer.value_normalizer
        )
        if self.parity_audit:
            self.last_parity_audit = self._capture_parity_snapshot(
                gaze_records,
                transition_records,
                gaze_forward_count=gaze_forward_count,
            )
        else:
            self.last_parity_audit = None
        value_predictions = self.trainer.value_normalizer.denormalize(
            self.buffer.value_preds[:-1]
        )
        variance = explained_variance(value_predictions, self.buffer.returns[:-1])
        upcoming = self.completed_steps + self.rollout_length * self.n_rollout_threads
        audit_started = time.perf_counter()
        action_audit = self._preupdate_audit()
        gaze_audit = (
            self._gaze_preupdate_audit(gaze_records)
            if self.parity_audit and self.train_gaze
            else {}
        )
        audit_seconds = time.perf_counter() - audit_started
        update_started = time.perf_counter()
        if self.train_action:
            action_update = self._action_ppo_update(upcoming)
        else:
            action_update = {
                "policy_loss": 0.0, "value_loss": 0.0, "dist_entropy": 0.0,
                "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
                "approx_kl": 0.0, "clip_fraction": 0.0,
                "entropy_coef": entropy_coefficient(
                    upcoming, self.entropy_coefs, self.entropy_coef_horizons
                ), "grad_clip_fraction": 0.0,
            }
        gaze_update = {
            "gaze_policy_loss": 0.0, "gaze_value_loss": 0.0,
            "gaze_entropy": 0.0, "gaze_approx_kl": 0.0,
            "gaze_clip_fraction": 0.0,
            "gaze_weighted_entropy": 0.0,
            "gaze_weighted_approx_kl": 0.0,
            "gaze_weighted_clip_fraction": 0.0,
            "gaze_records_per_task_decision_mean": 0.0,
            "gaze_records_per_task_decision_max": 0.0,
            "gaze_unique_task_decisions": 0,
            "gaze_decision_weight_sum_min": 0.0,
            "gaze_decision_weight_sum_max": 0.0,
            "gaze_decision_weight_sum_mean": 0.0,
            "gaze_weighted_advantage_mean": 0.0,
            "gaze_weighted_advantage_std": 0.0,
            "gaze_unweighted_advantage_mean": 0.0,
            "gaze_unweighted_advantage_std": 0.0,
            "gaze_explained_variance": None,
            "gaze_return_mean": None,
            "gaze_return_std": None,
            "gaze_value_mean": None,
            "gaze_value_std": None,
            "gaze_critic_grad_clip_fraction": None,
            "gaze_env_boundary_count_in_records": 0,
        }
        if self.train_gaze:
            gaze_ppo = dict(self.ppo)
            gaze_ppo["entropy_coef"] = entropy_coefficient(
                upcoming, self.entropy_coefs, self.entropy_coef_horizons
            )
            raw = gaze_ppo_update(
                self.gaze_actor, self.gaze_critic, self.gaze_optimizer,
                self.gaze_critic_optimizer, gaze_records, device=self.device, **gaze_ppo,
            )
            gaze_update = {
                "gaze_policy_loss": raw["policy_loss"],
                "gaze_value_loss": raw["value_loss"],
                "gaze_entropy": raw["gaze_entropy"],
                "gaze_approx_kl": raw["approx_kl"],
                "gaze_clip_fraction": raw["clip_fraction"],
                "gaze_actor_grad_norm": raw["actor_grad_norm"],
                "gaze_critic_grad_norm": raw["critic_grad_norm"],
                "gaze_weighted_entropy": raw["weighted_gaze_entropy"],
                "gaze_weighted_approx_kl": raw["weighted_approx_kl"],
                "gaze_weighted_clip_fraction": raw["weighted_clip_fraction"],
                "gaze_records_per_task_decision_mean": raw[
                    "records_per_task_decision_mean"
                ],
                "gaze_records_per_task_decision_max": raw[
                    "records_per_task_decision_max"
                ],
                "gaze_unique_task_decisions": raw["unique_task_decisions"],
                "gaze_decision_weight_sum_min": raw["decision_weight_sum_min"],
                "gaze_decision_weight_sum_max": raw["decision_weight_sum_max"],
                "gaze_decision_weight_sum_mean": raw["decision_weight_sum_mean"],
                "gaze_weighted_advantage_mean": raw["weighted_advantage_mean"],
                "gaze_weighted_advantage_std": raw["weighted_advantage_std"],
                "gaze_unweighted_advantage_mean": raw[
                    "unweighted_advantage_mean"
                ],
                "gaze_unweighted_advantage_std": raw[
                    "unweighted_advantage_std"
                ],
                "gaze_explained_variance": raw["explained_variance"],
                "gaze_return_mean": raw["return_mean"],
                "gaze_return_std": raw["return_std"],
                "gaze_value_mean": raw["value_mean"],
                "gaze_value_std": raw["value_std"],
                "gaze_critic_grad_clip_fraction": raw[
                    "critic_grad_clip_fraction"
                ],
                "gaze_env_boundary_count_in_records": raw[
                    "env_boundary_count_in_records"
                ],
            }
        update_seconds = time.perf_counter() - update_started
        self._clear_after_update()
        self.completed_steps = upcoming
        self.update_count += 1
        gc.collect()
        actor_after = parameter_hash(self.policy.actor)
        critic_after = parameter_hash(self.policy.critic)
        gaze_after = None if self.gaze_actor is None else state_dict_hash(self.gaze_actor.state_dict())
        if not self.train_action and (
            actor_after != actor_before or critic_after != critic_before
        ):
            raise RuntimeError("frozen ActionNet actor or critic changed")
        if self.gaze_type == "learned" and not self.train_gaze and gaze_after != gaze_before:
            raise RuntimeError("frozen GazeNet changed")
        if not np.isclose(optimization, sparse + weighted, rtol=1e-10, atol=1e-6):
            raise RuntimeError("Partial reward identity failed")
        gaze_attributed_reward = sum(
            controller.total_attributed_optimization_reward
            for controller in self.controllers
            if controller is not None
        ) - gaze_reward_before
        gaze_env_boundary_count = sum(
            controller.total_env_boundaries
            for controller in self.controllers
            if controller is not None
        ) - gaze_boundaries_before
        bootstrap_unowned_reward_count = sum(
            controller.total_bootstrap_unowned_reward_count
            for controller in self.controllers
            if controller is not None
        ) - bootstrap_count_before
        bootstrap_unowned_reward_sum = sum(
            controller.total_bootstrap_unowned_reward_sum
            for controller in self.controllers
            if controller is not None
        ) - bootstrap_reward_before
        gaze_reward_conservation_error = (
            0.0
            if self.gaze_type == "rule_based"
            else abs(
                optimization
                - gaze_attributed_reward
                - bootstrap_unowned_reward_sum
            )
        )
        if self.gaze_type == "learned" and not np.isclose(
            gaze_attributed_reward + bootstrap_unowned_reward_sum,
            optimization,
            rtol=1e-10,
            atol=1e-6,
        ):
            raise RuntimeError("Gaze reward ownership conservation failed")
        if self.gaze_type == "learned" and (
            gaze_env_boundary_count + bootstrap_unowned_reward_count
            != self.rollout_length * self.n_rollout_threads
        ):
            raise RuntimeError("Gaze reward ownership boundary count failed")
        deltas = []
        branch_counts = {name: 0 for name in ("task", "stay", "stale", "random")}
        for adapter, before in zip(self.adapters, metric_before):
            after = adapter.metrics()
            deltas.append({
                key: after[key] - before.get(key, 0)
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
                    "opacity_attempts", "opacity_successes",
                    "opacity_opportunities",
                    "opacity_opportunities_h1", "opacity_opportunities_h2",
                    "opacity_opportunities_h3", "opacity_opportunities_h4",
                    "opacity_attempts_h1", "opacity_attempts_h2",
                    "opacity_attempts_h3", "opacity_attempts_h4",
                    "opacity_successes_h1", "opacity_successes_h2",
                    "opacity_successes_h3", "opacity_successes_h4",
                    "opacity_preview_tiles", "overlap_preview_tiles",
                    "overlap_step_instances", "opacity_positive_step_instances",
                    "opacity_rng_draw_count",
                )
            })
            for name, count in after.get("rule_branch_counts", {}).items():
                branch_counts[name] += int(count - before.get("rule_branch_counts", {}).get(name, 0))
        aggregate = {key: sum(row[key] for row in deltas) for key in deltas[0]}
        memory_reports = [adapter.metrics() for adapter in self.adapters]
        if self.ground_truth_policy_input:
            for report in memory_reports:
                if report["actionnet_queries"] and (
                    report["freshness_normalized_min"] != 1.0
                    or report["freshness_normalized_max"] != 1.0
                    or report["valid_memory_fraction"] != 1.0
                ):
                    raise RuntimeError(
                        "Full ground-truth policy input violated the all-fresh "
                        "observation contract"
                    )
        active_samples = aggregate["executed_proposals"]
        decisions = self.rollout_length * self.n_rollout_threads
        partial_metrics = finalize_partial_metrics(partial_metric_totals)
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
        horizon_metrics = partial_metrics["horizon_metrics"]
        result = {
            "observation_mode": "partial", "training_stage": self.training_stage,
            "gaze_type": self.gaze_type, "train_components": list(self.train_components),
            "train_action": self.train_action, "train_gaze": self.train_gaze,
            "direct_joint_from_full": self.direct_joint_from_full,
            "ground_truth_policy_input": self.ground_truth_policy_input,
            "policy_observation_contract_id": (
                FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
                if self.ground_truth_policy_input
                else self.memory_contract_id
            ),
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
            "deliveries_per_episode": (
                None if episodes == 0 else deliveries / episodes
            ),
            "reward_identity_error": abs(optimization - sparse - weighted),
            "reward_shaping_factor": hsp_reward_shaping_factor(upcoming, self.reward_shaping_horizon_steps),
            "shaping_factor": hsp_reward_shaping_factor(upcoming, self.reward_shaping_horizon_steps),
            "reward_shaping_horizon_steps": self.reward_shaping_horizon_steps,
            "actionnet_online_forwards": aggregate["actionnet_queries"],
            "online_forwards_per_decision": aggregate["actionnet_queries"] / decisions,
            "actor_active_sample_rate": active_samples / decisions,
            "held_foveal_refreshes": aggregate["held_foveal_refreshes"],
            "held_foveal_refreshes_per_decision": (
                aggregate["held_foveal_refreshes"] / decisions
            ),
            "gaze_transition_count": len(gaze_records),
            "gaze_attributed_optimization_reward": gaze_attributed_reward,
            "bootstrap_unowned_reward_count": bootstrap_unowned_reward_count,
            "bootstrap_unowned_reward_sum": bootstrap_unowned_reward_sum,
            "gaze_reward_conservation_error": gaze_reward_conservation_error,
            "gaze_env_boundary_count": gaze_env_boundary_count,
            "gaze_policy_calls_total": sum(
                controller.next_record_sequence_id
                for controller in self.controllers
                if controller is not None
            ),
            "gaze_transition_count_total": sum(
                controller.next_record_sequence_id
                - int(controller.active is not None)
                for controller in self.controllers
                if controller is not None
            ),
            "bootstrap_unowned_reward_count_total": sum(
                controller.total_bootstrap_unowned_reward_count
                for controller in self.controllers
                if controller is not None
            ),
            "bootstrap_unowned_reward_sum_total": sum(
                controller.total_bootstrap_unowned_reward_sum
                for controller in self.controllers
                if controller is not None
            ),
            "active_partner": self.backend.active_partner,
            "active_partners": [backend.active_partner for backend in self.backends],
            "partner_step_guard_checks": sum(b.partner_step_guard_checks for b in self.backends) - guard_before,
            "partner_step_guard_failures": sum(b.partner_step_guard_failures for b in self.backends),
            "partner_full_hash_checks": sum(b.partner_full_hash_checks for b in self.backends) - full_guard_before,
            "partner_full_hash_failures": sum(b.partner_full_hash_failures for b in self.backends),
            "partner_optimizer_isolation_verified": self.partner_optimizer_isolation_verified,
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
            "rollout_collection_seconds": collection_seconds,
            "preupdate_audit_seconds": audit_seconds, "ppo_update_seconds": update_seconds,
            "inference_batching": self.inference_batching,
            "gaze_inference_seconds": gaze_inference_seconds,
            "gaze_forward_count": gaze_forward_count,
            "actionnet_inference_seconds": self._actionnet_rollout_seconds,
            "actionnet_forward_count": self._actionnet_rollout_forward_count,
            "predictor_inference_seconds": predictor_seconds,
            "predictor_forward_count": predictor_forward_count,
            "predictor_environment_count": predictor_environment_count,
            "partner_inference_seconds": partner_seconds,
            "environment_runtime_seconds": environment_seconds,
            "observation_memory_seconds": observation_memory_seconds,
            "ppo_rollout_bookkeeping_seconds": rollout_bookkeeping_seconds,
            "predictor_batch_parity": self.predictor_parity,
            "decisions_per_second": decisions / max(collection_seconds, 1e-12),
            "steps_per_second": decisions / max(collection_seconds + audit_seconds + update_seconds, 1e-12),
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
            **aggregate, **action_audit, **action_update, **gaze_update,
            **gaze_audit,
            **self.freshness_audit.report(freshness_before), **cooking_contract(),
        }
        result["mean_saccade_distance_degrees"] = float(np.mean([
            adapter.metrics()["mean_saccade_distance_degrees"] for adapter in self.adapters
        ]))
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
            "opacity_misses": (
                aggregate["opacity_attempts"] - aggregate["opacity_successes"]
            ),
            "opacity_success_rate": (
                aggregate["opacity_successes"]
                / max(aggregate["opacity_attempts"], 1)
            ),
            "opacity_detected_previews_per_acquisition": (
                (
                    aggregate["opacity_opportunities"]
                    if self.constraint_config.preview_opacity.mode == "off"
                    else aggregate["opacity_successes"]
                )
                / max(aggregate["acquisitions"], 1)
            ),
            "opacity_detected_previews_per_decision": (
                (
                    aggregate["opacity_opportunities"]
                    if self.constraint_config.preview_opacity.mode == "off"
                    else aggregate["opacity_successes"]
                ) / max(decisions, 1)
            ),
            "effective_preview_detections_per_acquisition": (
                (
                    aggregate["opacity_opportunities"]
                    if self.constraint_config.preview_opacity.mode == "off"
                    else aggregate["opacity_successes"]
                ) / max(aggregate["acquisitions"], 1)
            ),
            "effective_preview_detections_per_decision": (
                (
                    aggregate["opacity_opportunities"]
                    if self.constraint_config.preview_opacity.mode == "off"
                    else aggregate["opacity_successes"]
                ) / max(decisions, 1)
            ),
        })
        for step in range(1, 5):
            attempts = aggregate[f"opacity_attempts_h{step}"]
            successes = aggregate[f"opacity_successes_h{step}"]
            opportunities = aggregate[f"opacity_opportunities_h{step}"]
            result[f"effective_preview_detections_h{step}"] = (
                opportunities
                if self.constraint_config.preview_opacity.mode == "off"
                else successes
            )
            result[f"opacity_success_rate_h{step}"] = (
                successes / max(attempts, 1)
            )
        result["update_wall_seconds"] = time.perf_counter() - update_wall_started
        self._parity_action_records = None
        return result

    def save(self, path: Path, *, checkpoint_role: str | None = None):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_id": FORMAT_ID,
            "partial_checkpoint_format_id": PARTIAL_CHECKPOINT_FORMAT_ID,
            "training_stage": self.training_stage, "observation_mode": "partial",
            "gaze_type": self.gaze_type, "train_components": list(self.train_components),
            "train_action": self.train_action, "train_gaze": self.train_gaze,
            "direct_joint_from_full": self.direct_joint_from_full,
            "ground_truth_policy_input": self.ground_truth_policy_input,
            "policy_observation_contract_id": (
                FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
                if self.ground_truth_policy_input
                else self.memory_contract_id
            ),
            "inference_batching": self.inference_batching,
            "inference_shadow_audit": self.inference_shadow_audit,
            "step": self.completed_steps, "seed": self.seed,
            "canonical_encoder_id": CANONICAL_ENCODER_ID,
            "partner_set_id": PARTNER_SET_ID,
            "architecture_id": ACTION_ARCHITECTURE_ID,
            "action_space": list(ACTION_SPACE),
            "observation_contract_id": self.memory_contract_id,
            "gaze_architecture_id": None if self.gaze_actor is None else GAZE_ARCHITECTURE_ID,
            "actor_parameter_sha256": parameter_hash(self.policy.actor),
            "critic_parameter_sha256": parameter_hash(self.policy.critic),
            "gaze_parameter_sha256": None if self.gaze_actor is None else state_dict_hash(self.gaze_actor.state_dict()),
            "gaze_critic_parameter_sha256": None if self.gaze_critic is None else state_dict_hash(self.gaze_critic.state_dict()),
            "condition_mode": "fixed" if len(self.scheduler.conditions) == 1 else "cartesian_shuffled_blocks",
            "decision_times": list(self.scheduler.decision_times),
            "horizons": list(self.scheduler.horizons),
            "fixation_shapes": list(self.scheduler.fixation_shapes),
            "memory_decay_mode": self.memory_decay_mode,
            "memory_lifetime_ms": self.memory_lifetime_ms,
            "freshness_contract_id": self.freshness_contract_id,
            "freshness_input_scaling": (
                "persistent_binary_semantic_times_255_float32"
                if self.memory_decay_mode == "none"
                else "continuous_semantic_times_255_float32"
            ),
            **self.constraint_config.metadata(),
            "rule_gaze_policy_id": RULE_BASED_GAZE_POLICY_ID if self.gaze_type == "rule_based" else None,
            "rule_gaze_probabilities": self.rule_gaze_probabilities,
            "condition_scheduler_id": CONDITION_SCHEDULER_ID,
            "acquisition_action_contract_id": self.action_query_contract_id,
            "recurrent_commit_contract_id": self.recurrent_commit_contract_id,
            **(
                {
                    "held_foveal_refresh_contract_id": (
                        self.held_foveal_refresh_contract_id
                    )
                }
                if self.held_foveal_refresh_contract_id is not None
                else {}
            ),
            "actor_available_action_contract_id": ACTOR_AVAILABLE_ACTION_CONTRACT_ID,
            "partial_ppo_contract_id": PARTIAL_PPO_CONTRACT_ID,
            "gaze_action_contract_id": (
                GAZENET_V1_ACTION_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "gaze_input_contract_id": (
                GAZENET_V1_INPUT_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "gaze_reward_contract_id": (
                GAZENET_V1_REWARD_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "gaze_bootstrap_reward_contract_id": (
                GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "gaze_credit_assignment_contract_id": (
                GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID
                if self.gaze_type == "learned"
                else None
            ),
            "source_checkpoint_provenance": self.source_provenance,
            "source_full_checkpoint_provenance": self.source_provenance,
            "reward_shaping_horizon_steps": self.reward_shaping_horizon_steps,
            **self.fixed_partner_provenance,
            **cooking_contract(),
        }
        if self.memory_decay_mode == "stochastic_exponential":
            metadata.update({
                "memory_tau_ms": self.memory_tau_ms,
                "memory_max_lifetime_ms": self.memory_lifetime_ms,
                "memory_random_seed": self.memory_random_seed,
                "memory_rng_contract_id": MEMORY_RNG_CONTRACT_ID,
                "memory_rng_seed_offset": MEMORY_RNG_SEED_OFFSET,
            })
        if checkpoint_role is not None:
            metadata["checkpoint_role"] = checkpoint_role
        payload = {
            "metadata": metadata, "actor": self.policy.actor.state_dict(),
            "critic": self.policy.critic.state_dict(),
            "training_state": self._training_state(),
        }
        if self.gaze_actor is not None:
            payload.update(gaze_actor=self.gaze_actor.state_dict(),
                           gaze_critic=self.gaze_critic.state_dict())
        torch.save(payload, output)
        return {**metadata, "checkpoint_sha256": file_sha256(output)}
