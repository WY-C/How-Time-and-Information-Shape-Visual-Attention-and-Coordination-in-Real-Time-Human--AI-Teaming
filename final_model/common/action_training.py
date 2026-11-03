"""The single shared Action PPO rollout/update loop for Full and Partial."""

from __future__ import annotations

import gc
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final_model.checkpoint import (
    initialize_partial_from_full,
    load_action_checkpoint,
    save_action_checkpoint,
)
from final_model.common.hsp_initialization import (
    FullFreshnessWeightGuard,
    migrate_hsp_s2_actor,
)
from final_model.common.inference_batching import (
    FULL_INFERENCE_BATCHING_MODES,
    batched_partner_actions,
    initialize_partner_trajectory_shadow,
    legacy_partner_shadow_step,
    partner_shadow_parity,
    reset_partner_trajectory_shadow,
)
from final_model.common.mappo import (
    ACTOR_FORWARD_CONTEXT_ID,
    RolloutPolicyAuditCapture,
    SeparatedReplayBuffer,
    build_training_stack,
    entropy_coefficient,
    explained_variance,
    install_rollout_context_stable_actor,
    preupdate_policy_consistency_audit,
)
from final_model.common.overcooked_backend import (
    DirectOvercookedBackend,
    cooking_contract,
)
from final_model.common.partners import (
    FrozenPartnerRepository,
    FixedHSPPartnerRepository,
    assert_optimizer_isolation,
    parameter_hash,
)
from final_model.common.reward import (
    HSP_REWARD_SHAPING_HORIZON,
    REWARD_SHAPING_FINAL,
    REWARD_SHAPING_INITIAL,
    REWARD_SHAPING_SCHEDULE,
    hsp_reward_shaping_factor,
)
from final_model.common.resume import (
    TRAINING_STATE_FORMAT_ID,
    capture_global_rng_state,
    optimizer_to,
    require_training_state,
    restore_global_rng_state,
    validate_resume_contract,
)
from final_model.common.shared_horizons import (
    BalancedEpisodeHorizonSampler,
    HORIZON_MODE_EPISODE_BALANCED,
    HORIZON_MODE_FIXED,
    SHARED_HORIZONS,
    horizon_one_hot,
)
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.rule_based import RULE_BASED_GAZE_POLICY_ID
from final_model.simple_ver.observation import FullObservationAdapter


CANONICAL_FULL_TOTAL_STEPS = 20_000_000
CANONICAL_CHECKPOINT_INTERVAL = 1_000_000


class ActionTrainingSession:
    def __init__(
        self,
        *,
        observation_mode: str,
        horizon: int | None,
        decision_time: int | None,
        fixation_shape: str | None,
        seed: int,
        rollout_length: int,
        episode_length: int,
        device: str,
        source_checkpoint: Path | None,
        resume_checkpoint: Path | None = None,
        hsp_initial_checkpoint: Path | None = None,
        fixed_partner_checkpoint: Path | None = None,
        configured_total_steps: int = CANONICAL_FULL_TOTAL_STEPS,
        checkpoint_interval: int = CANONICAL_CHECKPOINT_INTERVAL,
        reward_shaping_horizon_steps: int = HSP_REWARD_SHAPING_HORIZON,
        partner: str | None = None,
        n_rollout_threads: int = 1,
        inference_batching: str | None = None,
        inference_shadow_audit: bool = False,
        shared_horizons: tuple[int, ...] | list[int] | None = None,
        **ppo,
    ) -> None:
        if observation_mode == "partial":
            raise RuntimeError(
                "canonical Partial training is acquisition-driven; construct "
                "final_model.partial_ver.canonical_training.PartialTrainingSession"
            )
        self.mode = observation_mode
        self.shared_horizons = (
            None if shared_horizons is None else tuple(int(value) for value in shared_horizons)
        )
        if self.shared_horizons is not None:
            if horizon is not None:
                raise ValueError("fixed horizon and shared horizons are mutually exclusive")
            if self.shared_horizons != SHARED_HORIZONS:
                raise ValueError(f"shared horizons must be exactly {SHARED_HORIZONS}")
            if observation_mode != "full":
                raise ValueError("shared horizons are only supported for Full Action")
            self.horizon = None
            self.horizon_mode = HORIZON_MODE_EPISODE_BALANCED
        else:
            if horizon is None:
                raise ValueError("fixed Action training requires horizon")
            self.horizon = int(horizon)
            self.horizon_mode = HORIZON_MODE_FIXED
        self.decision_time = decision_time
        self.fixation_shape = fixation_shape
        self.seed = int(seed)
        self.rollout_length = int(rollout_length)
        self.episode_length = int(episode_length)
        self.n_rollout_threads = int(n_rollout_threads)
        self.inference_batching = (
            "predictor"
            if inference_batching is None and self.mode == "full"
            else "legacy" if inference_batching is None else str(inference_batching)
        )
        self.inference_shadow_audit = bool(inference_shadow_audit)
        self.partner = partner
        self.fixed_partner_checkpoint = fixed_partner_checkpoint
        self.configured_total_steps = int(configured_total_steps)
        self.checkpoint_interval = int(checkpoint_interval)
        self.reward_shaping_horizon_steps = int(reward_shaping_horizon_steps)
        if self.configured_total_steps <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("configured total steps and checkpoint interval must be positive")
        if self.reward_shaping_horizon_steps < 0:
            raise ValueError("reward_shaping_horizon_steps cannot be negative")
        initial_paths = tuple(
            path
            for path in (source_checkpoint, resume_checkpoint, hsp_initial_checkpoint)
            if path is not None
        )
        if len(initial_paths) > 1:
            raise ValueError(
                "--source-checkpoint, --resume-checkpoint, and "
                "--hsp-initial-checkpoint are mutually exclusive"
            )
        if hsp_initial_checkpoint is not None and self.mode != "full":
            raise ValueError("--hsp-initial-checkpoint is only valid for Full Action")
        if fixed_partner_checkpoint is not None and self.mode != "full":
            raise ValueError("--fixed-partner-checkpoint is only valid for Full Action")
        if fixed_partner_checkpoint is not None and partner is not None:
            raise ValueError("--fixed-partner-checkpoint and --partner are mutually exclusive")
        if hsp_initial_checkpoint is not None and fixed_partner_checkpoint is None:
            raise ValueError("HSP-initialized Full Action requires --fixed-partner-checkpoint")
        if self.n_rollout_threads <= 0:
            raise ValueError("n_rollout_threads must be positive")
        if self.inference_batching not in FULL_INFERENCE_BATCHING_MODES:
            raise ValueError(
                "Full Action inference_batching must be one of "
                f"{FULL_INFERENCE_BATCHING_MODES}"
            )
        if self.mode != "full" and self.inference_batching != "legacy":
            raise ValueError("inference batching is only implemented for Full Action")
        if self.inference_shadow_audit and self.inference_batching == "legacy":
            raise ValueError("inference shadow audit requires a batched inference mode")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        args, policy, trainer, obs_space, share_space, action_space = build_training_stack(
            seed=self.seed,
            rollout_length=self.rollout_length,
            n_rollout_threads=self.n_rollout_threads,
            device=self.device,
            **ppo,
        )
        self.args, self.policy, self.trainer = args, policy, trainer
        self.source_provenance = None
        self.initialization_provenance = None
        if self.mode == "partial" and resume_checkpoint is None:
            if source_checkpoint is None:
                self.source_provenance = {
                    "initialization": "fresh_partial",
                    "source_horizon": self.horizon,
                    "source_contract": None,
                    "compatibility_used_absolute_path": False,
                }
            else:
                self.source_provenance = initialize_partial_from_full(
                    source_checkpoint, actor=self.policy.actor, horizon=self.horizon
                )
        elif self.mode == "full" and source_checkpoint is not None:
            raise ValueError("Full Action starts fresh and does not accept --source-checkpoint")
        if self.mode == "full" and resume_checkpoint is None:
            if hsp_initial_checkpoint is None:
                self.initialization_provenance = {
                    "initialization_type": "random_fresh",
                    "absolute_path_used_for_compatibility": False,
                    "initial_actor_parameter_sha256": parameter_hash(self.policy.actor),
                }
            else:
                self.initialization_provenance = migrate_hsp_s2_actor(
                    hsp_initial_checkpoint, target_actor=self.policy.actor
                )
        self.full_freshness_guard = None
        if self.mode == "full":
            self.full_freshness_guard = FullFreshnessWeightGuard(
                self.policy.actor, self.policy.actor_optimizer
            )
            if resume_checkpoint is None:
                self.full_freshness_guard.initialize_zero()
        install_rollout_context_stable_actor(
            self.policy.actor, microbatch_size=self.n_rollout_threads
        )
        if fixed_partner_checkpoint is None:
            repository = FrozenPartnerRepository(device=self.device)
            self.fixed_partner_provenance = None
        else:
            repository = FixedHSPPartnerRepository(
                device=self.device, checkpoint=fixed_partner_checkpoint
            )
            self.fixed_partner_provenance = repository.provenance()
            if self.initialization_provenance is not None:
                learner_sha = self.initialization_provenance.get(
                    "learner_hsp_source_sha256"
                )
                partner_sha = self.fixed_partner_provenance["fixed_partner_sha256"]
                if learner_sha == partner_sha:
                    raise RuntimeError("learner initialization and fixed partner must differ")
                self.initialization_provenance.update(
                    {
                        **self.fixed_partner_provenance,
                        "learner_source_equals_partner": False,
                        "full_channel20_input_value": 255.0,
                        "full_channel20_weight_frozen_zero": True,
                        "preview_channels_trainable": True,
                        "step0_checkpoint_saved": False,
                    }
                )
        self.horizon_sampler = (
            None
            if self.shared_horizons is None
            else BalancedEpisodeHorizonSampler(
                seed=self.seed, horizons=self.shared_horizons
            )
        )
        self.active_horizons = np.asarray(
            [
                self.horizon if self.horizon_sampler is None else self.horizon_sampler.next()
                for _ in range(self.n_rollout_threads)
            ],
            dtype=np.int64,
        )
        self.horizon_decision_counts = {value: 0 for value in SHARED_HORIZONS}
        self.backends = []
        shared_oracle = None
        prediction_device = (
            self.device
            if self.inference_batching in {"predictor", "partner_predictor"}
            else torch.device("cpu")
        )
        for env_index in range(self.n_rollout_threads):
            backend = DirectOvercookedBackend(
                seed=self.seed + env_index,
                horizon=int(self.active_horizons[env_index]),
                episode_length=self.episode_length,
                device=self.device,
                partner=partner,
                partner_repository=repository,
                prediction_oracle=shared_oracle,
                prediction_device=prediction_device,
                environment_index=env_index,
                enable_predictor=self.horizon_sampler is not None,
            )
            self.backends.append(backend)
            shared_oracle = backend.oracle
        optimizers = {
            "action_actor_optimizer": self.policy.actor_optimizer,
            "action_critic_optimizer": self.policy.critic_optimizer,
        }
        for actor in {id(backend.partners.actor): backend.partners.actor for backend in self.backends}.values():
            assert_optimizer_isolation(actor, optimizers)
        self.partner_optimizer_isolation_verified = True
        if self.mode == "full":
            self.adapters = [FullObservationAdapter(backend) for backend in self.backends]
        else:
            self.adapters = [
                PartialObservationAdapter(
                    backend,
                    decision_time=int(decision_time),
                    fixation_shape=str(fixation_shape),
                )
                for backend in self.backends
            ]
        # Preserve the convenient single-environment handles used by bounded audits.
        self.backend = self.backends[0]
        self.adapter = self.adapters[0]
        if (
            self.mode == "full"
            and self.fixed_partner_provenance is not None
        ):
            predictor_contract = self.backend.predictor_partner_contract()
            if not predictor_contract["predictor_partner_alignment_verified"]:
                raise RuntimeError("displayed predictor is not aligned to fixed partner")
            if predictor_contract["predictor_targets_learner"]:
                raise RuntimeError("displayed predictor must not target the learner")
            if self.initialization_provenance is not None:
                self.initialization_provenance.update(
                    {
                        "displayed_preview_partner_identity": predictor_contract[
                            "displayed_preview_partner_identity"
                        ],
                        "predictor_target_agent_index": predictor_contract[
                            "predictor_target_agent_index"
                        ],
                        "predictor_targets_learner": False,
                        "predictor_partner_alignment_verified": True,
                    }
                )
        self._executor = (
            None
            if self.n_rollout_threads == 1
            else ThreadPoolExecutor(max_workers=self.n_rollout_threads)
        )
        self._inference_counters = {
            "partner_inference_seconds": 0.0,
            "predictor_inference_seconds": 0.0,
            "env_step_seconds": 0.0,
            "partner_forward_calls": 0,
            "predictor_forward_calls": 0,
            "predictor_cache_generation_checks": 0,
        }
        self.partner_parity = None
        self.predictor_parity = None
        self.partner_trajectory_shadow = None
        self.predictor_trajectory_parity = None
        self._predictor_shadow_oracle = None
        if (
            self.inference_shadow_audit
            and self.inference_batching in {"predictor", "partner_predictor"}
            and (self.horizon_sampler is not None or int(self.horizon) > 0)
        ):
            self._predictor_shadow_oracle = (
                self.backend._displayed.CounterCircuitDisplayedPredictionOracle(
                    self.backend.oracle.checkpoint_path, device="cpu"
                )
            )
        self.buffer = SeparatedReplayBuffer(args, obs_space, share_space, action_space)
        self.completed_steps = 0
        self.update_count = 0
        self._initialize_buffer()
        if resume_checkpoint is not None:
            self._resume(resume_checkpoint)
        if (
            self.mode == "full"
            and self.fixed_partner_provenance is not None
            and all(hasattr(backend, "partner_batch_input") for backend in self.backends)
        ):
            partner_observations, partner_legal_masks = self._partner_inputs()
            self.partner_parity = partner_shadow_parity(
                [backend.partners for backend in self.backends],
                partner_observations,
                partner_legal_masks,
            )
            if (
                self.inference_shadow_audit
                and self.inference_batching in {"partner", "partner_predictor"}
            ):
                self.partner_trajectory_shadow = initialize_partner_trajectory_shadow(
                    [backend.partners for backend in self.backends]
                )

    def _resume_contract(self) -> dict[str, Any]:
        contract = {
            "training_stage": "action",
            "observation_mode": self.mode,
            "decision_time": self.decision_time,
            "fixation_shape": self.fixation_shape,
            "seed": self.seed,
            "rollout_length": self.rollout_length,
            "episode_length": self.episode_length,
            "n_rollout_threads": self.n_rollout_threads,
            "partner": self.partner,
            "fixed_partner_sha256": (
                None
                if self.fixed_partner_provenance is None
                else self.fixed_partner_provenance["fixed_partner_sha256"]
            ),
            "device_type": self.device.type,
            "actor_lr": float(self.args.lr),
            "critic_lr": float(self.args.critic_lr),
            "clip_param": float(self.args.clip_param),
            "gamma": float(self.args.gamma),
            "gae_lambda": float(self.args.gae_lambda),
            "entropy_coefs": list(self.args.entropy_coefs),
            "entropy_coef_horizons": list(self.args.entropy_coef_horizons),
            "ppo_epochs": int(self.args.ppo_epoch),
            "num_mini_batches": int(self.args.num_mini_batch),
            "max_grad_norm": float(self.args.max_grad_norm),
            "reward_shaping_horizon_steps": self.reward_shaping_horizon_steps,
            **cooking_contract(),
        }
        if self.horizon_sampler is None:
            contract["horizon"] = self.horizon
        else:
            contract.update(
                {
                    "horizon_mode": self.horizon_mode,
                    "horizons": list(self.shared_horizons),
                }
            )
        if self.mode == "full":
            contract.update(self._full_production_metadata())
        return contract

    def _full_production_metadata(self) -> dict[str, Any]:
        return {
            "configured_total_steps": self.configured_total_steps,
            "checkpoint_interval": self.checkpoint_interval,
            "entropy_schedule": list(self.args.entropy_coefs),
            "entropy_schedule_horizons": list(self.args.entropy_coef_horizons),
            "reward_shaping_schedule": REWARD_SHAPING_SCHEDULE,
            "reward_shaping_initial": REWARD_SHAPING_INITIAL,
            "reward_shaping_final": REWARD_SHAPING_FINAL,
            "reward_shaping_horizon_steps": self.reward_shaping_horizon_steps,
            "full_channel20_input": 255.0,
            "full_channel20_weight_frozen_zero": True,
            "preview_weight_norm_logging": True,
            "train_time_evaluation": False,
            "rollout_policy_consistency_audit": "pre_optimizer_fail_closed_v2",
            "observation_capture_storage": "independent_copy_and_uint8_sha256_v2",
            "recurrent_minibatch_order": "time_major_within_shuffled_env_chunks_v1",
            "actor_forward_context": ACTOR_FORWARD_CONTEXT_ID,
            "actor_forward_microbatch_size": self.n_rollout_threads,
            **cooking_contract(),
        }

    def _inference_metadata(self) -> dict[str, Any]:
        prediction_enabled = self.horizon_sampler is not None or self.horizon != 0
        return {
            "inference_batching": self.inference_batching,
            "inference_shadow_audit": self.inference_shadow_audit,
            "partner_inference_schedule": (
                "batch1_per_env"
                if self.inference_batching in {"legacy", "predictor"}
                else "one_shared_env_batch"
            ),
            "predictor_inference_schedule": (
                "none"
                if not prediction_enabled
                else (
                    "one_shared_env_batch_per_environment_step"
                    if self.inference_batching in {"predictor", "partner_predictor"}
                    else "batch1_per_env_per_environment_step"
                )
            ),
            "partner_sampling_semantics": (
                "per-env torch.Generator sampled in canonical env order"
            ),
            "partner_rnn_state_ownership": "independent_per_environment",
            "predictor_target_agent_index": 0,
            "predictor_model_singleton": True,
        }

    def _buffer_training_state(self) -> dict[str, Any]:
        names = (
            "share_obs", "obs", "rnn_states", "rnn_states_critic", "masks",
            "bad_masks", "active_masks", "available_actions",
        )
        return {name: getattr(self.buffer, name)[0].copy() for name in names}

    def _schedule_state(self) -> dict[str, float | int]:
        return {
            "global_environment_step": self.completed_steps,
            "entropy_coef": entropy_coefficient(
                self.completed_steps,
                self.args.entropy_coefs,
                self.args.entropy_coef_horizons,
            ),
            "reward_shaping_factor": hsp_reward_shaping_factor(
                self.completed_steps, self.reward_shaping_horizon_steps
            ),
            "reward_shaping_horizon_steps": self.reward_shaping_horizon_steps,
        }

    def _load_buffer_training_state(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            target = getattr(self.buffer, name)[0]
            source = np.asarray(value, dtype=target.dtype)
            if source.shape != target.shape:
                raise RuntimeError(
                    f"resume buffer {name} shape mismatch: {source.shape} != {target.shape}"
                )
            target[...] = source
        self.buffer.step = 0

    def _training_state(self) -> dict[str, Any]:
        value_normalizer = self.trainer.value_normalizer
        state = {
            "format_id": TRAINING_STATE_FORMAT_ID,
            "contract": self._resume_contract(),
            "completed_steps": self.completed_steps,
            "update_count": self.update_count,
            "schedule_state": self._schedule_state(),
            "actor_optimizer": self.policy.actor_optimizer.state_dict(),
            "critic_optimizer": self.policy.critic_optimizer.state_dict(),
            "value_normalizer": (
                None if value_normalizer is None else value_normalizer.state_dict()
            ),
            "backends": [backend.training_state() for backend in self.backends],
            "adapters": [adapter.training_state() for adapter in self.adapters],
            "buffer": self._buffer_training_state(),
            "prepared": tuple(value.copy() for value in self._prepared),
            "rng": capture_global_rng_state(),
        }
        if self.horizon_sampler is not None:
            state["shared_horizon"] = {
                "active_horizons": self.active_horizons.copy(),
                "active_horizon_one_hot": np.stack(
                    [horizon_one_hot(value) for value in self.active_horizons]
                ),
                "sampler": self.horizon_sampler.state_dict(),
                "decision_counts": dict(self.horizon_decision_counts),
            }
        return state

    def _resume(self, path: Path) -> None:
        condition = cooking_contract()
        if self.mode == "partial":
            condition.update({
                "decision_time": self.decision_time,
                "fixation_shape": self.fixation_shape,
                "gaze_policy": RULE_BASED_GAZE_POLICY_ID,
            })
        elif self.fixed_partner_provenance is not None:
            condition.update({
                "fixed_partner_sha256": self.fixed_partner_provenance[
                    "fixed_partner_sha256"
                ],
                "full_channel20_weight_frozen_zero": True,
                **self._full_production_metadata(),
            })
        metadata = load_action_checkpoint(
            path,
            actor=self.policy.actor,
            critic=self.policy.critic,
            expected_mode=self.mode,
            expected_horizon=self.horizon,
            expected_horizon_mode=(
                None if self.horizon_sampler is None else self.horizon_mode
            ),
            expected_horizons=self.shared_horizons,
            expected_condition=condition,
        )
        checkpoint_reward_horizon = int(
            metadata.get(
                "reward_shaping_horizon_steps", HSP_REWARD_SHAPING_HORIZON
            )
        )
        if checkpoint_reward_horizon != self.reward_shaping_horizon_steps:
            raise RuntimeError(
                "checkpoint reward shaping horizon mismatch: "
                f"{checkpoint_reward_horizon} != {self.reward_shaping_horizon_steps}"
            )
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = require_training_state(payload)
        saved_contract = state.get("contract")
        if isinstance(saved_contract, dict):
            saved_contract = dict(saved_contract)
            saved_contract.setdefault(
                "reward_shaping_horizon_steps", HSP_REWARD_SHAPING_HORIZON
            )
        validate_resume_contract(saved_contract, self._resume_contract())
        if int(metadata.get("step", -1)) != int(state.get("completed_steps", -2)):
            raise RuntimeError("checkpoint metadata/training-state step mismatch")
        self.policy.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.policy.critic_optimizer.load_state_dict(state["critic_optimizer"])
        optimizer_to(self.policy.actor_optimizer, self.device)
        optimizer_to(self.policy.critic_optimizer, self.device)
        if self.full_freshness_guard is not None:
            self.full_freshness_guard.assert_zero("resume_checkpoint")
        normalizer_state = state["value_normalizer"]
        if (self.trainer.value_normalizer is None) != (normalizer_state is None):
            raise RuntimeError("ValueNorm resume contract mismatch")
        if self.trainer.value_normalizer is not None:
            self.trainer.value_normalizer.load_state_dict(normalizer_state)
        if len(state["backends"]) != len(self.backends) or len(state["adapters"]) != len(self.adapters):
            raise RuntimeError("resume rollout environment count mismatch")
        if self.horizon_sampler is not None:
            shared = state.get("shared_horizon")
            if not isinstance(shared, dict):
                raise RuntimeError("shared checkpoint has no shared horizon state")
            self.horizon_sampler.load_state_dict(shared["sampler"])
            active = np.asarray(shared["active_horizons"], dtype=np.int64)
            if active.shape != (self.n_rollout_threads,) or not np.isin(
                active, self.shared_horizons
            ).all():
                raise RuntimeError("invalid active horizons in shared checkpoint")
            expected_one_hot = np.stack([horizon_one_hot(value) for value in active])
            if not np.array_equal(
                np.asarray(shared["active_horizon_one_hot"], dtype=np.float32),
                expected_one_hot,
            ):
                raise RuntimeError("shared checkpoint horizon one-hot mismatch")
            decision_counts = {
                int(key): int(value) for key, value in shared["decision_counts"].items()
            }
            if set(decision_counts) != set(SHARED_HORIZONS) or any(
                value < 0 for value in decision_counts.values()
            ):
                raise RuntimeError("invalid shared horizon decision counts")
            if sum(decision_counts.values()) != int(state["completed_steps"]):
                raise RuntimeError("shared horizon decision counts/progress mismatch")
            self.active_horizons = active.copy()
            self.horizon_decision_counts = decision_counts
            for backend, active_horizon in zip(self.backends, active):
                backend.set_horizon(int(active_horizon))
        elif "shared_horizon" in state:
            raise RuntimeError("fixed run cannot resume shared horizon state")
        for backend, saved in zip(self.backends, state["backends"]):
            backend.load_training_state(saved)
        for adapter, saved in zip(self.adapters, state["adapters"]):
            adapter.load_training_state(saved)
        self._load_buffer_training_state(state["buffer"])
        prepared = tuple(np.asarray(value).copy() for value in state["prepared"])
        if len(prepared) != 3:
            raise RuntimeError("invalid prepared decision state")
        for prepared_value, buffer_name in zip(
            prepared, ("obs", "share_obs", "available_actions")
        ):
            if not np.array_equal(prepared_value, getattr(self.buffer, buffer_name)[0]):
                raise RuntimeError(f"resume prepared/{buffer_name} state mismatch")
        self._prepared = prepared
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
            raise RuntimeError("invalid Action training progress in resume checkpoint")
        schedule_state = state.get("schedule_state")
        if isinstance(schedule_state, dict):
            schedule_state = dict(schedule_state)
            schedule_state.setdefault(
                "reward_shaping_horizon_steps", HSP_REWARD_SHAPING_HORIZON
            )
        expected_schedule_state = self._schedule_state()
        if not isinstance(schedule_state, dict) or schedule_state != expected_schedule_state:
            raise RuntimeError(
                "resume schedule state mismatch: "
                f"{schedule_state!r} != {expected_schedule_state!r}"
            )
        if self.mode == "partial":
            self.source_provenance = metadata.get("source_provenance")
            if not isinstance(self.source_provenance, dict):
                raise RuntimeError("Partial Action resume checkpoint has no source provenance")
        else:
            self.initialization_provenance = metadata.get("initialization_provenance")
            if not isinstance(self.initialization_provenance, dict):
                self.initialization_provenance = {
                    "initialization_type": metadata.get(
                        "initialization_type", "legacy_full_checkpoint"
                    ),
                    "resume_source_actor_parameter_sha256": metadata.get(
                        "actor_parameter_sha256"
                    ),
                    "absolute_path_used_for_compatibility": False,
                }
            if self.fixed_partner_provenance is not None:
                if (
                    self.initialization_provenance.get("fixed_partner_sha256")
                    != self.fixed_partner_provenance["fixed_partner_sha256"]
                ):
                    raise RuntimeError("Full resume fixed-partner provenance mismatch")
        resumed_shaping_factor = float(schedule_state["reward_shaping_factor"])
        for backend in self.backends:
            backend.set_reward_shaping_factor(resumed_shaping_factor)
        self.trainer.entropy_coef = float(schedule_state["entropy_coef"])
        restore_global_rng_state(state["rng"])

    @property
    def active_horizon_one_hot(self) -> np.ndarray:
        return np.stack([horizon_one_hot(value) for value in self.active_horizons])

    def _assert_active_horizons(self) -> None:
        actual = np.asarray([backend.horizon for backend in self.backends], dtype=np.int64)
        if not np.array_equal(actual, self.active_horizons):
            raise RuntimeError("active horizon changed inside an episode")

    def _assign_next_horizon(self, env_index: int) -> None:
        if self.horizon_sampler is None:
            return
        backend = self.backends[env_index]
        if backend.episode_step != 0:
            raise RuntimeError("next horizon assigned before environment reset")
        value = self.horizon_sampler.next()
        backend.set_horizon(value)
        self.active_horizons[env_index] = value

    def _current(self):
        current = [adapter.current() for adapter in self.adapters]
        return tuple(np.stack(values, axis=0) for values in zip(*current))

    def _synchronize_inference_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _partner_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        rows = [backend.partner_batch_input() for backend in self.backends]
        observations, legal_masks = (
            np.stack(values, axis=0) for values in zip(*rows)
        )
        if observations.shape[0] != self.n_rollout_threads:
            raise RuntimeError("partner batch environment ordering changed")
        return observations, legal_masks

    def _predictor_inputs(self):
        expected = list(range(self.n_rollout_threads))
        actual = [backend.environment_index for backend in self.backends]
        if actual != expected:
            raise RuntimeError(
                f"predictor environment ordering changed: {actual} != {expected}"
            )
        return (
            [backend.state for backend in self.backends],
            [backend.mdp for backend in self.backends],
            [backend.self_history for backend in self.backends],
            [backend.partner_history for backend in self.backends],
        )

    def _record_predictor_parity(
        self,
        batch_predictions,
        batch_logits: np.ndarray,
        predictor_inputs,
    ) -> None:
        if self.predictor_parity is not None:
            return
        oracle = self.backend.oracle
        legacy_oracle = self._predictor_shadow_oracle
        if legacy_oracle is None:
            legacy_oracle = (
                self.backend._displayed.CounterCircuitDisplayedPredictionOracle(
                    oracle.checkpoint_path, device="cpu"
                )
            )
        legacy_predictions = []
        legacy_logits = []
        states, mdps, self_histories, partner_histories = predictor_inputs
        prediction_horizon = 4 if self.horizon_sampler is not None else int(self.horizon)
        for env_index in range(self.n_rollout_threads):
            predictions, logits = legacy_oracle.predict_batch_with_logits(
                (states[env_index],),
                (mdps[env_index],),
                (self_histories[env_index],),
                (partner_histories[env_index],),
                prediction_horizon,
            )
            legacy_predictions.append(predictions[0])
            legacy_logits.append(logits[0])
        legacy_logits_array = np.stack(legacy_logits)
        difference = np.abs(
            legacy_logits_array.astype(np.float64)
            - np.asarray(batch_logits, dtype=np.float64)
        )
        action_agreements = []
        preview_agreements = []
        first_divergence = None
        for horizon_index in range(4):
            action_equal = np.asarray(
                [
                    legacy.action_indices[horizon_index]
                    == batched.action_indices[horizon_index]
                    for legacy, batched in zip(
                        legacy_predictions, batch_predictions
                    )
                ],
                dtype=np.bool_,
            )
            preview_equal = np.asarray(
                [
                    np.array_equal(
                        legacy.displayed_prediction_gt[horizon_index],
                        batched.displayed_prediction_gt[horizon_index],
                    )
                    for legacy, batched in zip(
                        legacy_predictions, batch_predictions
                    )
                ],
                dtype=np.bool_,
            )
            action_agreements.append(float(action_equal.mean()))
            preview_agreements.append(float(preview_equal.mean()))
            if first_divergence is None and not bool(preview_equal.all()):
                first_divergence = {
                    "env": int(np.flatnonzero(~preview_equal)[0]),
                    "horizon": horizon_index + 1,
                }
        contract = self.backend.predictor_partner_contract()
        self.predictor_parity = {
            "environment_count": self.n_rollout_threads,
            "env_mapping_exact": True,
            "predictor_model_shared_instance_count": len(
                {id(backend.oracle) for backend in self.backends}
            ),
            "legacy_device": "cpu",
            "batched_device": str(oracle.device),
            "predictor_max_abs_logit_difference": float(difference.max()),
            "predictor_mean_abs_logit_difference": float(difference.mean()),
            "displayed_action_agreement_by_horizon": action_agreements,
            "displayed_preview_agreement_by_horizon": preview_agreements,
            "displayed_preview_agreement": float(min(preview_agreements)),
            "first_displayed_preview_divergence": first_divergence,
            "predictor_target_agent_index": contract["predictor_target_agent_index"],
            "actual_partner_agent_index": contract["actual_partner_agent_index"],
            "displayed_preview_partner_identity": contract[
                "displayed_preview_partner_identity"
            ],
            "predictor_partner_alignment_verified": contract[
                "predictor_partner_alignment_verified"
            ],
        }

    def _record_predictor_trajectory_parity(
        self,
        batch_predictions,
        batch_logits: np.ndarray,
        predictor_inputs,
    ) -> None:
        if self._predictor_shadow_oracle is None:
            return
        states, mdps, self_histories, partner_histories = predictor_inputs
        prediction_horizon = 4 if self.horizon_sampler is not None else int(self.horizon)
        legacy_predictions = []
        legacy_logits = []
        for env_index in range(self.n_rollout_threads):
            predictions, logits = self._predictor_shadow_oracle.predict_batch_with_logits(
                (states[env_index],),
                (mdps[env_index],),
                (self_histories[env_index],),
                (partner_histories[env_index],),
                prediction_horizon,
            )
            legacy_predictions.append(predictions[0])
            legacy_logits.append(logits[0])
        difference = np.abs(
            np.stack(legacy_logits).astype(np.float64)
            - np.asarray(batch_logits, dtype=np.float64)
        )
        if self.predictor_trajectory_parity is None:
            self.predictor_trajectory_parity = {
                "checked_state_batches": 0,
                "checked_env_states": 0,
                "logit_abs_difference_sum": 0.0,
                "logit_element_count": 0,
                "predictor_max_abs_logit_difference": 0.0,
                "action_equal_counts": [0, 0, 0, 0],
                "preview_equal_counts": [0, 0, 0, 0],
                "first_divergence": None,
            }
        report = self.predictor_trajectory_parity
        report["checked_state_batches"] += 1
        report["checked_env_states"] += self.n_rollout_threads
        report["logit_abs_difference_sum"] += float(difference.sum())
        report["logit_element_count"] += int(difference.size)
        report["predictor_max_abs_logit_difference"] = max(
            report["predictor_max_abs_logit_difference"], float(difference.max())
        )
        for horizon_index in range(4):
            for env_index, (legacy, batched) in enumerate(
                zip(legacy_predictions, batch_predictions)
            ):
                action_equal = (
                    legacy.action_indices[horizon_index]
                    == batched.action_indices[horizon_index]
                )
                preview_equal = np.array_equal(
                    legacy.displayed_prediction_gt[horizon_index],
                    batched.displayed_prediction_gt[horizon_index],
                )
                report["action_equal_counts"][horizon_index] += int(action_equal)
                report["preview_equal_counts"][horizon_index] += int(preview_equal)
                if report["first_divergence"] is None and not preview_equal:
                    report["first_divergence"] = {
                        "state_batch": report["checked_state_batches"] - 1,
                        "env": env_index,
                        "horizon": horizon_index + 1,
                    }

    def _predictor_trajectory_report(self):
        source = self.predictor_trajectory_parity
        if source is None:
            return None
        total = int(source["checked_env_states"])
        return {
            "checked_state_batches": int(source["checked_state_batches"]),
            "checked_env_states": total,
            "predictor_max_abs_logit_difference": float(
                source["predictor_max_abs_logit_difference"]
            ),
            "predictor_mean_abs_logit_difference": float(
                source["logit_abs_difference_sum"]
                / max(int(source["logit_element_count"]), 1)
            ),
            "displayed_action_agreement_by_horizon": [
                float(count / max(total, 1))
                for count in source["action_equal_counts"]
            ],
            "displayed_preview_agreement_by_horizon": [
                float(count / max(total, 1))
                for count in source["preview_equal_counts"]
            ],
            "first_divergence": source["first_divergence"],
            "env_mapping_exact": True,
            "cache_generation_identity_checked": True,
        }

    def _populate_full_previews(self) -> None:
        if self.horizon_sampler is None and self.horizon == 0:
            return
        if all(
            backend._preview_cache is not None
            and backend._preview_cache_generation == backend.state_generation
            for backend in self.backends
        ):
            return
        self._synchronize_inference_device()
        started = time.perf_counter()
        if self.inference_batching in {"predictor", "partner_predictor"}:
            predictor_inputs = self._predictor_inputs()
            prediction_horizon = (
                4 if self.horizon_sampler is not None else int(self.horizon)
            )
            predictions, logits = self.backend.oracle.predict_batch_with_logits(
                *predictor_inputs, prediction_horizon
            )
            self._record_predictor_parity(predictions, logits, predictor_inputs)
            self._record_predictor_trajectory_parity(
                predictions, logits, predictor_inputs
            )
            for env_index, (backend, prediction) in enumerate(
                zip(self.backends, predictions)
            ):
                if backend.environment_index != env_index:
                    raise RuntimeError("batched predictor output/env mapping mismatch")
                backend.set_batched_preview(
                    prediction.displayed_prediction_gt,
                    generation=backend.state_generation,
                )
                self._inference_counters["predictor_cache_generation_checks"] += 1
            forward_calls = 1
        else:
            if self._executor is None:
                self.backend.preview_planes()
            else:
                list(self._executor.map(lambda backend: backend.preview_planes(), self.backends))
            forward_calls = self.n_rollout_threads
        self._synchronize_inference_device()
        self._inference_counters["predictor_inference_seconds"] += (
            time.perf_counter() - started
        )
        self._inference_counters["predictor_forward_calls"] += forward_calls

    def _prepare_decisions(self):
        if self.horizon_sampler is not None:
            self._assert_active_horizons()
        if (
            self.mode == "full"
            and all(hasattr(backend, "state_generation") for backend in self.backends)
        ):
            self._populate_full_previews()
        if self._executor is None:
            current = [self.adapter.prepare_decision()]
        else:
            current = list(self._executor.map(lambda adapter: adapter.prepare_decision(), self.adapters))
        return tuple(np.stack(values, axis=0) for values in zip(*current))

    def _commit_decisions(self, actions: np.ndarray):
        selected = [int(value) for value in actions.reshape(-1)]
        if (
            self.mode == "full"
            and all(
                hasattr(adapter, "commit_decision_with_partner")
                and hasattr(adapter.backend, "partner_batch_input")
                for adapter in self.adapters
            )
        ):
            observations, legal_masks = self._partner_inputs()
            before_hashes = [
                backend.begin_partner_step() for backend in self.backends
            ]
            self._synchronize_inference_device()
            partner_started = time.perf_counter()
            if self.inference_batching in {"legacy", "predictor"}:
                if self._executor is None:
                    partner_actions = np.asarray(
                        [self.backend.partners.action(observations[0], legal_masks[0])]
                    )
                else:
                    partner_actions = np.asarray(
                        list(
                            self._executor.map(
                                lambda row: row[0].partners.action(row[1], row[2]),
                                zip(self.backends, observations, legal_masks),
                            )
                        ),
                        dtype=np.int64,
                    )
                partner_forward_calls = self.n_rollout_threads
            else:
                partner_actions, partner_outputs = batched_partner_actions(
                    [backend.partners for backend in self.backends],
                    observations,
                    legal_masks,
                )
                partner_forward_calls = 1
                shadow = self.partner_trajectory_shadow
                if shadow is not None and shadow["first_divergence"] is None:
                    legacy_actions, legacy_outputs = legacy_partner_shadow_step(
                        [backend.partners for backend in self.backends],
                        observations,
                        legal_masks,
                        shadow,
                    )
                    logit_difference = np.abs(
                        legacy_outputs["logits"].astype(np.float64)
                        - partner_outputs["logits"].astype(np.float64)
                    )
                    probability_difference = np.abs(
                        legacy_outputs["probabilities"].astype(np.float64)
                        - partner_outputs["probabilities"].astype(np.float64)
                    )
                    hidden_difference = np.abs(
                        legacy_outputs["next_hidden"].astype(np.float64)
                        - partner_outputs["next_hidden"].astype(np.float64)
                    )
                    shadow["max_abs_logit_difference"] = max(
                        float(shadow.get("max_abs_logit_difference", 0.0)),
                        float(logit_difference.max()),
                    )
                    shadow["max_abs_probability_difference"] = max(
                        float(shadow.get("max_abs_probability_difference", 0.0)),
                        float(probability_difference.max()),
                    )
                    shadow["max_abs_next_hidden_difference"] = max(
                        float(shadow.get("max_abs_next_hidden_difference", 0.0)),
                        float(hidden_difference.max()),
                    )
                    action_equal = legacy_actions == partner_actions
                    if not bool(action_equal.all()):
                        env_index = int(np.flatnonzero(~action_equal)[0])
                        shadow["first_divergence"] = {
                            "rollout_step": int(shadow["checked_steps"]) - 1,
                            "env": env_index,
                            "legacy_partner_action": int(legacy_actions[env_index]),
                            "batched_partner_action": int(partner_actions[env_index]),
                            "max_abs_logit_difference_at_step": float(
                                logit_difference.max()
                            ),
                            "max_abs_probability_difference_at_step": float(
                                probability_difference.max()
                            ),
                            "max_abs_next_hidden_difference_at_step": float(
                                hidden_difference.max()
                            ),
                        }
            self._synchronize_inference_device()
            self._inference_counters["partner_inference_seconds"] += (
                time.perf_counter() - partner_started
            )
            self._inference_counters["partner_forward_calls"] += partner_forward_calls
            env_started = time.perf_counter()
            rows = list(zip(self.adapters, selected, partner_actions, before_hashes))
            if self._executor is None:
                adapter, learner_action, partner_action, before_hash = rows[0]
                results = [
                    adapter.commit_decision_with_partner(
                        learner_action, partner_action, before_hash
                    )
                ]
            else:
                results = list(
                    self._executor.map(
                        lambda row: row[0].commit_decision_with_partner(
                            row[1], row[2], row[3]
                        ),
                        rows,
                    )
                )
            self._inference_counters["env_step_seconds"] += (
                time.perf_counter() - env_started
            )
            return results
        if self._executor is None:
            return [self.adapter.commit_decision(selected[0])]
        return list(
            self._executor.map(
                lambda pair: pair[0].commit_decision(pair[1]),
                zip(self.adapters, selected),
            )
        )

    def _initialize_buffer(self) -> None:
        obs, critic, available = self._prepare_decisions()
        self._prepared = (obs, critic, available)
        self.buffer.obs[0] = obs
        self.buffer.share_obs[0] = critic
        self.buffer.available_actions[0] = available
        self.buffer.masks[0].fill(0.0)

    def _clear(self) -> None:
        self.buffer.share_obs[1:].fill(0.0)
        self.buffer.obs[1:].fill(0.0)
        self.buffer.rnn_states[1:].fill(0.0)
        self.buffer.rnn_states_critic[1:].fill(0.0)
        self.buffer.value_preds.fill(0.0)
        self.buffer.returns.fill(0.0)
        self.buffer.actions.fill(0.0)
        self.buffer.action_log_probs.fill(0.0)
        self.buffer.rewards.fill(0.0)
        self.buffer.masks[1:].fill(1.0)
        self.buffer.bad_masks[1:].fill(1.0)
        self.buffer.active_masks[1:].fill(1.0)
        self.buffer.available_actions[1:].fill(1.0)
        self.buffer.transition_ids.fill(-1)

    def run_update(self) -> dict[str, Any]:
        if self.full_freshness_guard is not None:
            self.full_freshness_guard.assert_zero("before_ppo_update")
        actor_before = parameter_hash(self.policy.actor)
        step_guard_checks_before = sum(
            backend.partner_step_guard_checks for backend in self.backends
        )
        full_hash_checks_before = sum(
            backend.partner_full_hash_checks for backend in self.backends
        )
        for backend in self.backends:
            backend.verify_partner_parameters()
        forwards_before = sum(adapter.actionnet_online_forwards for adapter in self.adapters)
        decisions_before = sum(adapter.task_decisions for adapter in self.adapters)
        partial_before = (
            {
                key: sum(float(adapter.metrics()[key]) for adapter in self.adapters)
                for key in self.adapter.metrics()
            }
            if self.mode == "partial"
            else {}
        )
        distance_before = (
            sum(adapter.distance_sum_degrees for adapter in self.adapters)
            if self.mode == "partial"
            else 0.0
        )
        inference_before = dict(self._inference_counters)
        optimization = sparse = shaped = weighted_shaped = 0.0
        deliveries = episodes = 0
        horizon_totals = (
            {
                value: {
                    "episodes": 0,
                    "task_decisions": 0,
                    "deliveries": 0,
                    "sparse_team_reward": 0.0,
                    "native_shaped_reward": 0.0,
                    "optimization_reward": 0.0,
                }
                for value in SHARED_HORIZONS
            }
            if self.horizon_sampler is not None
            else None
        )
        rollout_horizon_rows = []
        collection_started = time.perf_counter()
        self.trainer.prep_rollout()
        rollout_audit = RolloutPolicyAuditCapture(self.buffer)
        rollout_audit.attach_actor(self.policy.actor)
        for rollout_step in range(self.rollout_length):
            if self.horizon_sampler is not None:
                self._assert_active_horizons()
                decision_horizons = self.active_horizons.copy()
                rollout_horizon_rows.append(decision_horizons)
            else:
                decision_horizons = None
            reward_shaping_factor = hsp_reward_shaping_factor(
                self.completed_steps + rollout_step * self.n_rollout_threads,
                self.reward_shaping_horizon_steps,
            )
            for backend in self.backends:
                backend.set_reward_shaping_factor(reward_shaping_factor)
            step = self.buffer.step
            observation, critic, available = self._prepared
            self.buffer.obs[step] = observation
            self.buffer.share_obs[step] = critic
            self.buffer.available_actions[step] = available
            rollout_audit.record_pre_forward(
                step,
                observation,
                self.buffer,
            )
            rollout_audit.record_environment_before(
                step, self.backends, self.buffer
            )
            rollout_audit.begin_actor_forward(step, self.policy.actor)
            with torch.no_grad():
                values, actions, logs, actor_rnn, critic_rnn = self.policy.get_actions(
                    self.buffer.share_obs[step], self.buffer.obs[step],
                    self.buffer.rnn_states[step], self.buffer.rnn_states_critic[step],
                    self.buffer.masks[step], self.buffer.available_actions[step],
                )
            rollout_audit.verify_post_actor_forward(step, self.buffer)
            pending = {
                "values": values.detach().cpu().numpy(),
                "actions": actions.detach().cpu().numpy(),
                "logs": logs.detach().cpu().numpy(),
                "actor_rnn": actor_rnn.detach().cpu().numpy(),
                "critic_rnn": critic_rnn.detach().cpu().numpy(),
            }
            rollout_audit.record_post_forward(
                step, pending["actions"], pending["logs"]
            )
            results = self._commit_decisions(pending["actions"])
            for env_index, result in enumerate(results):
                if self.horizon_sampler is not None:
                    active_horizon = int(decision_horizons[env_index])
                    bucket = horizon_totals[active_horizon]
                    bucket["task_decisions"] += 1
                    bucket["deliveries"] += int(result["deliveries"])
                    bucket["sparse_team_reward"] += float(result["sparse_team_reward"])
                    bucket["native_shaped_reward"] += float(result["native_shaped_reward"])
                    bucket["optimization_reward"] += float(result["optimization_reward"])
                    self.horizon_decision_counts[active_horizon] += 1
                optimization += float(result["optimization_reward"])
                sparse += float(result["sparse_team_reward"])
                shaped += float(result["native_shaped_reward"])
                weighted_shaped += float(result["weighted_shaped_reward"])
                deliveries += int(result["deliveries"])
            next_actor, next_critic = pending["actor_rnn"], pending["critic_rnn"]
            returned_actor_before_reset = next_actor.copy()
            next_mask = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
            bad_mask = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
            for env_index, result in enumerate(results):
                if result["done"]:
                    episodes += 1
                    if self.horizon_sampler is not None:
                        horizon_totals[int(decision_horizons[env_index])]["episodes"] += 1
                    self.adapters[env_index].reset_episode()
                    self._assign_next_horizon(env_index)
                    next_actor[env_index].fill(0.0)
                    next_critic[env_index].fill(0.0)
                    next_mask[env_index].fill(0.0)
                    bad_mask[env_index].fill(0.0)
                elif (
                    self.horizon_sampler is not None
                    and self.backends[env_index].horizon
                    != int(decision_horizons[env_index])
                ):
                    raise RuntimeError("horizon changed before episode termination")
            if (
                self.partner_trajectory_shadow is not None
                and self.partner_trajectory_shadow["first_divergence"] is None
            ):
                reset_partner_trajectory_shadow(
                    self.partner_trajectory_shadow,
                    [
                        env_index
                        for env_index, result in enumerate(results)
                        if result["done"]
                    ],
                )
            rollout_audit.record_environment_after(
                step,
                results,
                returned_actor_before_reset,
                next_actor,
                next_mask,
            )
            next_obs, next_share, next_available = self._prepare_decisions()
            self._prepared = (next_obs, next_share, next_available)
            self.buffer.insert(
                next_share, next_obs, next_actor, next_critic,
                pending["actions"], pending["logs"], pending["values"],
                np.asarray(
                    [[result["optimization_reward"]] for result in results],
                    dtype=np.float32,
                ),
                next_mask,
                bad_masks=bad_mask,
                active_masks=np.ones((self.n_rollout_threads, 1), dtype=np.float32),
                available_actions=next_available,
            )
        rollout_audit.end_rollout(self.policy.actor)
        collection_seconds = time.perf_counter() - collection_started
        for backend in self.backends:
            backend.verify_partner_parameters()
        with torch.no_grad():
            next_value = self.policy.get_values(
                self.buffer.share_obs[-1], self.buffer.rnn_states_critic[-1], self.buffer.masks[-1]
            )
        self.buffer.compute_returns(next_value.detach().cpu().numpy(), self.trainer.value_normalizer)
        value_predictions = self.buffer.value_preds[:-1]
        if self.trainer.value_normalizer is not None:
            value_predictions = self.trainer.value_normalizer.denormalize(
                value_predictions
            )
        rollout_explained_variance = explained_variance(
            value_predictions, self.buffer.returns[:-1]
        )
        audit_started = time.perf_counter()
        upcoming_steps = self.completed_steps + self.rollout_length * self.n_rollout_threads
        preupdate_audit = preupdate_policy_consistency_audit(
            policy=self.policy,
            buffer=self.buffer,
            capture=rollout_audit,
            clip_param=self.trainer.clip_param,
            data_chunk_length=self.trainer.data_chunk_length,
            mode=self.mode,
            horizon=(
                self.horizon
                if self.horizon_sampler is None
                else np.stack(rollout_horizon_rows, axis=0)
            ),
        )
        preupdate_audit_seconds = time.perf_counter() - audit_started
        self.trainer.adapt_entropy_coef(upcoming_steps)
        self.trainer.prep_training()
        update_started = time.perf_counter()
        update = dict(self.trainer.train(self.buffer))
        if self.full_freshness_guard is not None:
            self.full_freshness_guard.after_optimizer_update()
        update_seconds = time.perf_counter() - update_started
        self.buffer.after_update()
        self._clear()
        self.buffer.step = 0
        self.completed_steps = upcoming_steps
        self.update_count += 1
        gc.collect()
        forwards = sum(adapter.actionnet_online_forwards for adapter in self.adapters) - forwards_before
        decisions = sum(adapter.task_decisions for adapter in self.adapters) - decisions_before
        rollout_decisions = self.rollout_length * self.n_rollout_threads
        if forwards != decisions or decisions != rollout_decisions:
            raise RuntimeError("one ActionNet forward per decision contract failed")
        if horizon_totals is not None:
            if sum(bucket["task_decisions"] for bucket in horizon_totals.values()) != decisions:
                raise RuntimeError("aggregate/shared horizon decision metric mismatch")
            if sum(bucket["episodes"] for bucket in horizon_totals.values()) != episodes:
                raise RuntimeError("aggregate/shared horizon episode metric mismatch")
            for key, aggregate in (
                ("deliveries", deliveries),
                ("sparse_team_reward", sparse),
                ("native_shaped_reward", shaped),
                ("optimization_reward", optimization),
            ):
                if not np.isclose(
                    sum(bucket[key] for bucket in horizon_totals.values()), aggregate
                ):
                    raise RuntimeError(f"aggregate/shared horizon {key} metric mismatch")
        partner_counts: dict[str, int] = {}
        for backend in self.backends:
            for name, count in backend.partners.counts.items():
                partner_counts[name] = partner_counts.get(name, 0) + int(count)
        step_guard_checks = sum(
            backend.partner_step_guard_checks for backend in self.backends
        ) - step_guard_checks_before
        full_hash_checks = sum(
            backend.partner_full_hash_checks for backend in self.backends
        ) - full_hash_checks_before
        full_hash_failures = sum(
            backend.partner_full_hash_failures for backend in self.backends
        )
        step_guard_failures = sum(
            backend.partner_step_guard_failures for backend in self.backends
        )
        result = {
            "observation_mode": self.mode,
            "training_stage": "action",
            "horizon": self.horizon,
            "horizon_mode": self.horizon_mode,
            "horizons": (
                None if self.shared_horizons is None else list(self.shared_horizons)
            ),
            "decision_time": self.decision_time,
            "fixation_shape": self.fixation_shape,
            "environment_steps": self.completed_steps,
            "global_step": self.completed_steps,
            "task_decisions": decisions,
            "actionnet_online_forwards": forwards,
            "actionnet_batched_forward_calls": self.rollout_length,
            "n_rollout_threads": self.n_rollout_threads,
            "online_forwards_per_decision": forwards / decisions,
            "updates": self.update_count,
            "update_count": self.update_count,
            "episodes": episodes,
            "reward": optimization,
            "optimization_reward": optimization,
            "sparse_team_reward": sparse,
            "native_shaped_reward": shaped,
            "weighted_shaped_reward": weighted_shaped,
            "reward_shaping_factor": hsp_reward_shaping_factor(
                upcoming_steps, self.reward_shaping_horizon_steps
            ),
            "shaping_factor": hsp_reward_shaping_factor(
                upcoming_steps, self.reward_shaping_horizon_steps
            ),
            "reward_shaping_horizon_steps": self.reward_shaping_horizon_steps,
            **cooking_contract(),
            "reward_identity_error": abs(optimization - sparse - weighted_shaped),
            "deliveries": deliveries,
            "active_partner": self.backend.active_partner,
            "active_partners": [backend.active_partner for backend in self.backends],
            "sampled_partner_identities": partner_counts,
            "partner_step_guard_checks": step_guard_checks,
            "partner_step_guard_failures": step_guard_failures,
            "partner_full_hash_checks": full_hash_checks,
            "partner_full_hash_failures": full_hash_failures,
            "partner_full_hash_baseline": self.backend.last_partner_full_hash_baseline,
            "partner_full_hash_actual": self.backend.last_partner_full_hash_actual,
            "partner_optimizer_isolation_verified": self.partner_optimizer_isolation_verified,
            **self._inference_metadata(),
            "partner_inference_seconds": (
                self._inference_counters["partner_inference_seconds"]
                - inference_before["partner_inference_seconds"]
            ),
            "predictor_inference_seconds": (
                self._inference_counters["predictor_inference_seconds"]
                - inference_before["predictor_inference_seconds"]
            ),
            "env_step_seconds": (
                self._inference_counters["env_step_seconds"]
                - inference_before["env_step_seconds"]
            ),
            "partner_forward_calls": int(
                self._inference_counters["partner_forward_calls"]
                - inference_before["partner_forward_calls"]
            ),
            "predictor_forward_calls": int(
                self._inference_counters["predictor_forward_calls"]
                - inference_before["predictor_forward_calls"]
            ),
            "partner_batch_size": (
                1
                if self.inference_batching in {"legacy", "predictor"}
                else self.n_rollout_threads
            ),
            "predictor_batch_size": (
                0
                if self.horizon_sampler is None and self.horizon == 0
                else (
                    self.n_rollout_threads
                    if self.inference_batching in {"predictor", "partner_predictor"}
                    else 1
                )
            ),
            "predictor_cache_generation_checks": int(
                self._inference_counters["predictor_cache_generation_checks"]
                - inference_before["predictor_cache_generation_checks"]
            ),
            "partner_batch_parity": self.partner_parity,
            "predictor_batch_parity": self.predictor_parity,
            "partner_trajectory_parity": self.partner_trajectory_report(),
            "predictor_trajectory_parity": self._predictor_trajectory_report(),
            "actor_hash_before": actor_before,
            "actor_hash_after": parameter_hash(self.policy.actor),
            "rollout_collection_seconds": collection_seconds,
            "ppo_update_seconds": update_seconds,
            "decisions_per_second": decisions / max(collection_seconds, 1e-12),
            "steps_per_second": decisions
            / max(
                collection_seconds + preupdate_audit_seconds + update_seconds,
                1e-12,
            ),
            "explained_variance": rollout_explained_variance,
            "preupdate_audit_seconds": preupdate_audit_seconds,
            **preupdate_audit,
            **{key: float(value) for key, value in update.items()},
        }
        if self.horizon_sampler is not None:
            for horizon, bucket in horizon_totals.items():
                bucket["deliveries_per_episode"] = (
                    bucket["deliveries"] / bucket["episodes"]
                    if bucket["episodes"] else 0.0
                )
                bucket["sparse_team_reward_per_episode"] = (
                    bucket["sparse_team_reward"] / bucket["episodes"]
                    if bucket["episodes"] else 0.0
                )
            result["shared_horizon_metrics"] = {
                f"h{horizon}": bucket for horizon, bucket in horizon_totals.items()
            }
            result["shared_horizon_episode_counts"] = {
                f"h{horizon}": int(self.horizon_sampler.counts[horizon])
                for horizon in SHARED_HORIZONS
            }
            result["shared_horizon_decision_counts"] = {
                f"h{horizon}": int(self.horizon_decision_counts[horizon])
                for horizon in SHARED_HORIZONS
            }
            result["active_horizons"] = self.active_horizons.tolist()
            result["active_horizon_one_hot"] = self.active_horizon_one_hot.tolist()
            result["ppo_sample_counts_by_horizon"] = {
                f"h{horizon}": int(horizon_totals[horizon]["task_decisions"])
                for horizon in SHARED_HORIZONS
            }
            result["actor_optimizer_count"] = 1
            result["critic_optimizer_count"] = 1
        if self.full_freshness_guard is not None:
            result.update(self.full_freshness_guard.metrics())
        if self.mode == "partial":
            after = {
                key: sum(float(adapter.metrics()[key]) for adapter in self.adapters)
                for key in self.adapter.metrics()
            }
            for key in (
                "gaze_requests", "acquisitions", "same_target_acquisitions",
                "new_target_acquisitions", "prep_count", "exec_count", "dwell_count",
                "rule_exploration_selections", "rule_monitoring_selections",
            ):
                result[key] = after[key] - partial_before[key]
            result["gaze_policy"] = RULE_BASED_GAZE_POLICY_ID
            result["valid_memory_fraction"] = (
                after["valid_memory_fraction"] / self.n_rollout_threads
            )
            result["mean_saccade_distance_degrees"] = (
                sum(adapter.distance_sum_degrees for adapter in self.adapters) - distance_before
            ) / max(result["new_target_acquisitions"], 1)
        return result

    def input_channel_weight_metrics(self) -> dict[str, float]:
        if self.full_freshness_guard is None:
            return {}
        return self.full_freshness_guard.metrics()

    def partner_trajectory_report(self):
        shadow = self.partner_trajectory_shadow
        if shadow is None:
            return None
        return {
            "checked_steps": int(shadow["checked_steps"]),
            "checked_transitions": int(shadow["checked_transitions"]),
            "max_abs_logit_difference": float(
                shadow.get("max_abs_logit_difference", 0.0)
            ),
            "max_abs_probability_difference": float(
                shadow.get("max_abs_probability_difference", 0.0)
            ),
            "max_abs_next_hidden_difference": float(
                shadow.get("max_abs_next_hidden_difference", 0.0)
            ),
            "first_divergence": shadow["first_divergence"],
            "env_mapping_exact": True,
            "rnn_state_ownership": "independent_per_environment",
            "sampling_rng_semantics": "independent per-env legacy generator shadow",
        }

    def save(self, path: Path, *, checkpoint_role: str | None = None):
        if self.full_freshness_guard is not None:
            self.full_freshness_guard.assert_zero("before_checkpoint_save")
        condition = {}
        if self.mode == "partial":
            condition = {
                "decision_time": self.decision_time,
                "fixation_shape": self.fixation_shape,
                "gaze_policy": RULE_BASED_GAZE_POLICY_ID,
                "source_provenance": self.source_provenance,
                "reward_shaping_horizon_steps": self.reward_shaping_horizon_steps,
                **cooking_contract(),
            }
        else:
            schedule_state = self._schedule_state()
            condition = {
                **self.initialization_provenance,
                "initialization_provenance": self.initialization_provenance,
                **self.input_channel_weight_metrics(),
                **self._full_production_metadata(),
                **self._inference_metadata(),
                "step0_checkpoint_saved": bool(
                    self.initialization_provenance.get("step0_checkpoint_saved", False)
                ),
                "checkpoint_entropy_coef": schedule_state["entropy_coef"],
                "checkpoint_reward_shaping_factor": schedule_state[
                    "reward_shaping_factor"
                ],
            }
        if checkpoint_role is not None:
            condition["checkpoint_role"] = checkpoint_role
        metadata = save_action_checkpoint(
            path,
            actor=self.policy.actor,
            critic=self.policy.critic,
            mode=self.mode,
            horizon=self.horizon,
            horizon_mode=self.horizon_mode,
            horizons=self.shared_horizons,
            step=self.completed_steps,
            seed=self.seed,
            condition=condition,
            training_state=self._training_state(),
        )
        if self.full_freshness_guard is not None:
            self.full_freshness_guard.assert_zero("after_checkpoint_save")
        return metadata

    def save_initialization_checkpoint(self, path: Path):
        if self.mode != "full" or self.initialization_provenance is None:
            raise RuntimeError("step-0 initialization checkpoint is Full Action only")
        if self.completed_steps != 0 or self.update_count != 0:
            raise RuntimeError("step-0 checkpoint requested after training began")
        if self.policy.actor_optimizer.state or self.policy.critic_optimizer.state:
            raise RuntimeError("step-0 checkpoint optimizers are not fresh")
        if self.full_freshness_guard is None:
            raise RuntimeError("Full freshness frozen-zero guard is absent")
        self.full_freshness_guard.assert_zero("step0_checkpoint")
        self.initialization_provenance["step0_checkpoint_saved"] = True
        self.initialization_provenance["step0_checkpoint_path"] = str(Path(path))
        return self.save(path, checkpoint_role="pre_training_initialization")
