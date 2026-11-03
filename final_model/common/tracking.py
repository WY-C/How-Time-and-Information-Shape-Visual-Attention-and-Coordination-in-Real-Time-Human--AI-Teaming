"""Shared optional RNG-neutral W&B integration."""

from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Any, Mapping

import numpy as np
import torch


WANDB_MODES = ("disabled", "offline", "online")
MANDATORY_KEYS = (
    "train/reward",
    "train/optimization_reward",
    "train/sparse_team_reward",
    "train/native_shaped_reward",
    "train/deliveries",
    "train/approx_kl",
    "train/entropy",
    "train/clip_fraction",
)
FULL_ACTION_MANDATORY_KEYS = (
    "train/sparse_team_reward",
    "train/native_shaped_reward",
    "train/shaping_factor",
    "train/optimization_reward",
    "train/deliveries",
    "train/approx_kl",
    "train/clip_fraction",
    "train/policy_loss",
    "train/value_loss",
    "train/action_entropy",
    "train/entropy_coef",
    "train/explained_variance",
    "train/actor_grad_norm",
    "train/critic_grad_norm",
    "train/grad_clip_fraction",
    "train/global_step",
    "train/update_count",
    "system/steps_per_second",
    "system/rollout_seconds",
    "system/update_seconds",
    "system/partner_inference_seconds",
    "system/predictor_inference_seconds",
    "system/env_step_seconds",
    "system/partner_forward_calls",
    "system/predictor_forward_calls",
    "system/partner_batch_size",
    "system/predictor_batch_size",
    "audit/preupdate_approx_kl",
    "audit/preupdate_clip_fraction",
    "audit/preupdate_max_abs_logprob_difference",
    "audit/preupdate_observation_exact_match",
    "audit/preupdate_transition_id_exact_match",
    "audit/preupdate_max_legal_mask_difference",
    "audit/preupdate_buffer_batch_max_abs_logprob_difference",
    "audit/preupdate_capture_direct_max_abs_logprob_difference",
    "audit/preupdate_sequence_max_abs_logprob_difference",
    "train/freshness_weight_norm",
    "train/preview_weight_norm_h1",
    "train/preview_weight_norm_h2",
    "train/preview_weight_norm_h3",
    "train/preview_weight_norm_h4",
)


@contextmanager
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def canonical_run_name(
    mode: str, stage: str, horizon: int, decision_time, shape, seed: int,
    total_steps: int | None = None,
) -> str:
    condition = f"h{horizon}"
    if mode == "partial":
        condition += f"_t{decision_time}_{shape}"
    budget = "" if total_steps is None else f"_{int(total_steps) // 1_000_000}m"
    return f"final_model_{mode}_{stage}_{condition}_seed{seed}{budget}"


class WandbTracker:
    def __init__(self, *, mode: str, project: str, name: str, config: Mapping[str, Any]):
        if mode not in WANDB_MODES:
            raise ValueError(f"wandb mode must be one of {WANDB_MODES}")
        self.require_full_action_metrics = (
            config.get("observation_mode") == "full"
            and config.get("training_stage") == "action"
        )
        self.run = None
        if mode == "disabled":
            return
        try:
            import wandb
        except ModuleNotFoundError as error:
            raise RuntimeError("W&B mode requested but wandb is not installed") from error
        with preserve_rng_state():
            self.run = wandb.init(project=project, name=name, config=dict(config), mode=mode)

    def log(self, metrics: Mapping[str, Any], *, environment_steps: int) -> None:
        missing = [key for key in MANDATORY_KEYS if key not in metrics]
        if self.require_full_action_metrics:
            missing.extend(
                key for key in FULL_ACTION_MANDATORY_KEYS if key not in metrics
            )
        if missing:
            raise RuntimeError(f"mandatory W&B keys missing: {sorted(set(missing))}")
        self.log_auxiliary(metrics, environment_steps=environment_steps)

    def log_auxiliary(
        self, metrics: Mapping[str, Any], *, environment_steps: int
    ) -> None:
        if self.run is None:
            return
        detached = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu())
            elif isinstance(value, np.generic):
                value = value.item()
            detached[key] = value
        with preserve_rng_state():
            self.run.log(detached, step=int(environment_steps))

    def finish(self) -> None:
        if self.run is not None:
            with preserve_rng_state():
                self.run.finish()


def tracking_contract_report() -> dict[str, object]:
    return {
        "mandatory_keys": list(MANDATORY_KEYS),
        "full_action_mandatory_keys": list(FULL_ACTION_MANDATORY_KEYS),
        "action_training_key": "train/action_entropy",
        "gaze_training_key": "train/gaze_entropy",
        "global_step": "environment_steps",
        "cadence": "once per completed PPO update (40,000 environment steps canonically)",
        "reward": "rollout sum of actual optimization rewards inserted into PPO",
        "sparse_team_reward": "rollout sum over 40,000 transitions",
        "native_shaped_reward": "rollout sum of learner-agent native shaped reward",
        "shaping_factor": (
            "configured linear reward-shaping schedule value at the completed global step; "
            "20M is only the backward-compatible CLI default"
        ),
        "weighted_shaped_reward": "rollout sum of per-transition factor * native shaped reward",
        "deliveries": "rollout sum of 20-point sparse deliveries",
        "optimization_reward": (
            "rollout sum of sparse team reward + per-transition configured linear shaping factor "
            "* shaped_reward_by_agent[1]"
        ),
        "approx_kl": (
            "mean((ratio - 1) - log(ratio)) over ratios evaluated before each PPO "
            "optimization pass; later passes include updates from earlier passes"
        ),
        "clip_fraction": (
            "mean(abs(ratio - 1) > clip_param) over ratios evaluated before each PPO "
            "optimization pass; later passes include updates from earlier passes"
        ),
        "preupdate_policy_audit": {
            "timing": "after rollout and before prep_training/backward/optimizer.step",
            "scope": "every transition plus first/middle/last and episode-boundary direct replays",
            "logprob_tolerance": 1e-5,
            "ratio": "computed from current eval-mode actor and rollout old log-prob",
            "failure": "PPO update is not entered",
            "recurrent_order": "time-major within independently shuffled env-major chunks",
            "observation_storage": "independent exact copy plus uint8[32] SHA-256",
            "insert_time_parity": "source vs buffer.obs[t] exact values and SHA-256 for every transition",
            "post_actor_parity": "independent copy vs buffer.obs[t] after actor forward for every transition",
            "alias_check": "np.shares_memory(source, buffer) and np.shares_memory(capture, buffer) must both be false",
            "transition_ids": "stored integer plane t * n_envs + env and consumed by recurrent replay",
            "actor_forward_context": "rollout-sized CNN microbatches and one-step GRU calls preserve the rollout CUDA execution context",
            "canonical_direct_replay": "replay the complete original timestep batch; singleton and all-transition flattened calls are noncanonical diagnostics",
            "singleton_replay": "logged for diagnosis only because slicing one transition changes the actual actor call batch context",
        },
        "policy_loss": "mean over PPO epochs and minibatches",
        "value_loss": "mean over PPO epochs and minibatches",
        "action_entropy": "mean categorical learner entropy over PPO epochs/minibatches",
        "entropy_coef": "piecewise-linear schedule value at completed global environment step",
        "explained_variance": (
            "1 - Var(return - pre-update critic prediction) / Var(return) over rollout; "
            "0 for degenerate target variance"
        ),
        "actor_grad_norm": "mean pre-clip total actor gradient norm over PPO passes",
        "critic_grad_norm": "mean pre-clip total critic gradient norm over PPO passes",
        "grad_clip_fraction": (
            "fraction of PPO passes where actor or critic pre-clip norm exceeds max_grad_norm"
        ),
        "steps_per_second": (
            "rollout decisions / (rollout seconds + pre-update audit seconds + PPO update seconds)"
        ),
        "rollout_seconds": "wall time for rollout collection",
        "partner_inference_seconds": "wall time for fixed-partner inference schedules",
        "predictor_inference_seconds": "wall time for action-preview predictor schedules",
        "env_step_seconds": "wall time for the unchanged ThreadPoolExecutor environment transitions",
        "update_seconds": "wall time for PPO optimization",
        "preupdate_audit_seconds": "wall time for fail-closed rollout-policy parity audit",
        "full_input_weight_norms": {
            "train/freshness_weight_norm": "L2 norm of base.cnn.0.weight[:,20,:,:]; exact zero",
            "train/preview_weight_norm_h1": "L2 norm of base.cnn.0.weight[:,21,:,:]",
            "train/preview_weight_norm_h2": "L2 norm of base.cnn.0.weight[:,22,:,:]",
            "train/preview_weight_norm_h3": "L2 norm of base.cnn.0.weight[:,23,:,:]",
            "train/preview_weight_norm_h4": "L2 norm of base.cnn.0.weight[:,24,:,:]",
        },
        "logging_side_effects": "detached scalars only; RNG restored; no model/env calls",
        "train_time_evaluation": False,
    }
