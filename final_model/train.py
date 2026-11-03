#!/usr/bin/env python3
"""The one canonical production trainer for Full Action, Partial Action, and Partial Gaze."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

PROJECT_PARENT = Path(__file__).resolve().parents[1]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from tqdm.auto import tqdm

from final_model.common.action_training import (
    CANONICAL_CHECKPOINT_INTERVAL,
    CANONICAL_FULL_TOTAL_STEPS,
    ActionTrainingSession,
)
from final_model.common.mappo import (
    ACTOR_FORWARD_CONTEXT_ID,
    CANONICAL_ENTROPY_COEFS,
    CANONICAL_ENTROPY_HORIZONS,
    GAZE_ENTROPY_COEFS,
    GAZE_ENTROPY_HORIZONS,
    PREUPDATE_LOGPROB_ATOL,
)
from final_model.common.hsp_initialization import HSP_SOURCE_CHECKPOINT_SHA256
from final_model.common.inference_batching import (
    FULL_INFERENCE_BATCHING_MODES,
    INFERENCE_BATCHING_MODES,
    PARTIAL_LEARNED_PRODUCTION_MODES,
)
from final_model.common.overcooked_backend import cooking_contract
from final_model.common.partners import FIXED_HSP_PARTNER_SHA256
from final_model.common.paths import file_sha256
from final_model.common.reward import (
    HSP_REWARD_SHAPING_HORIZON,
    REWARD_SHAPING_FINAL,
    REWARD_SHAPING_INITIAL,
    REWARD_SHAPING_SCHEDULE,
)
from final_model.common.tracking import WANDB_MODES, WandbTracker, canonical_run_name
from final_model.partial_ver.constraints import (
    BASELINE_CONSTRAINT_CONFIG,
    resolve_constraint_config,
)
from final_model.partial_ver.gaze import FIXATION_SHAPES
from final_model.partial_ver.memory import (
    MEMORY_DECAY_MODES,
    MEMORY_LIFETIME_MS,
    MEMORY_RNG_CONTRACT_ID,
    MEMORY_RNG_SEED_OFFSET,
)
from final_model.partial_ver.peripheral import PERIPHERAL_EXTENTS
from final_model.partial_ver.observation import ALLOWED_DECISION_TIMES
from final_model.partial_ver.observability import (
    flatten_partial_metric_namespaces,
)
from final_model.partial_ver.rule_based import (
    RULE_BASED_GAZE_POLICY_ID,
    validate_rule_probabilities,
)
from final_model.partial_ver.canonical_training import PartialTrainingSession


DEFAULT_STEPS = CANONICAL_FULL_TOTAL_STEPS
DEFAULT_ROLLOUT_LENGTH = 400
DEFAULT_EPISODE_LENGTH = 400
DEFAULT_N_ROLLOUT_THREADS = 100
DEFAULT_CHECKPOINT_INTERVAL = CANONICAL_CHECKPOINT_INTERVAL
DEFAULT_ENTROPY_COEFS = CANONICAL_ENTROPY_COEFS
DEFAULT_ENTROPY_HORIZONS = CANONICAL_ENTROPY_HORIZONS
DEFAULT_REWARD_SHAPING_HORIZON_STEPS = HSP_REWARD_SHAPING_HORIZON


class _StoreExplicit(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--observation-mode", choices=("full", "partial"), required=True)
    value.add_argument("--training-stage", choices=("action", "gaze"))
    value.add_argument("--gaze-type", choices=("rule_based", "learned"))
    value.add_argument(
        "--train-components", choices=("action", "gaze"), nargs="+"
    )
    horizon_group = value.add_mutually_exclusive_group(required=True)
    horizon_group.add_argument("--horizon", type=int, choices=(0, 2, 4))
    horizon_group.add_argument("--shared-horizons", type=int, nargs="+")
    value.add_argument("--decision-time", type=int, choices=ALLOWED_DECISION_TIMES)
    value.add_argument(
        "--shared-decision-times", type=int, choices=ALLOWED_DECISION_TIMES, nargs="+"
    )
    value.add_argument("--fixation-shape", choices=FIXATION_SHAPES)
    value.add_argument(
        "--shared-fixation-shapes", choices=FIXATION_SHAPES, nargs="+"
    )
    value.add_argument(
        "--memory-decay-mode",
        choices=MEMORY_DECAY_MODES,
        default="linear",
        help=(
            "Partial memory freshness: linear expiry, persistent no-decay, or "
            "seeded stochastic-exponential expiry"
        ),
    )
    value.add_argument(
        "--memory-lifetime-ms",
        type=float,
        default=None,
        help=(
            "Partial linear dynamic-memory lifetime in model milliseconds "
            "(default: 2400; omit when --memory-decay-mode none)"
        ),
    )
    value.add_argument("--memory-tau-ms", type=float)
    value.add_argument("--memory-max-lifetime-ms", type=float)
    value.add_argument(
        "--constraint-profile", choices=("baseline", "full", "custom")
    )
    value.add_argument("--peripheral-mode", choices=("none", "local", "global"))
    value.add_argument("--peripheral-sigma-deg", type=float)
    value.add_argument("--peripheral-extent", choices=PERIPHERAL_EXTENTS)
    value.add_argument(
        "--emma-mode",
        choices=(
            "default", "tuned", "human_stochastic", "default_encoding",
            "relaxed_saccade",
        ),
        help=(
            "Partial EMMA timing; omitted uses the fitted tuned parameters, "
            "'default' explicitly selects legacy timing, and "
            "'human_stochastic' uses the calibrated broad dwell distribution; "
            "'default_encoding' adds fixed-familiarity EMMA encoding to "
            "canonical movement timing; 'relaxed_saccade' halves preparation, "
            "execution, and encoding latency"
        ),
    )
    value.add_argument("--emma-prep-ms", type=float)
    value.add_argument("--emma-exec-intercept-ms", type=float)
    value.add_argument("--emma-exec-slope-ms-per-degree", type=float)
    value.add_argument("--emma-dwell-ms", type=float)
    value.add_argument("--preview-opacity-mode", choices=("off", "actual"))
    value.add_argument("--rule-gaze-task-prob", type=float, default=0.60)
    value.add_argument("--rule-gaze-stay-prob", type=float, default=0.20)
    value.add_argument("--rule-gaze-stale-prob", type=float, default=0.15)
    value.add_argument("--rule-gaze-random-prob", type=float, default=0.05)
    value.add_argument("--source-checkpoint", type=Path)
    value.add_argument(
        "--direct-joint-from-full",
        action="store_true",
        help=(
            "explicit diagnostic: initialize a learned Partial joint run from "
            "a Full Action checkpoint with fresh Partial critic and GazeNet"
        ),
    )
    value.add_argument(
        "--hsp-initial-checkpoint",
        type=Path,
        help="initialize only a Full Action actor from canonical HSP S2 20-channel weights",
    )
    value.add_argument(
        "--fixed-partner-checkpoint",
        type=Path,
        help="fixed canonical HSP S2-s24 seed-3 partner for Full and Partial training",
    )
    value.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume an Action/Gaze training checkpoint at a completed PPO boundary",
    )
    value.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="final total environment-step target, including already completed resume steps",
    )
    value.add_argument("--rollout-length", type=int, default=DEFAULT_ROLLOUT_LENGTH)
    value.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH)
    value.add_argument(
        "--n-rollout-threads", type=int, default=DEFAULT_N_ROLLOUT_THREADS
    )
    value.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    value.add_argument(
        "--reward-shaping-horizon-steps",
        type=int,
        default=DEFAULT_REWARD_SHAPING_HORIZON_STEPS,
        help=(
            "independent linear reward-shaping horizon; 0 disables shaping "
            "entirely; not derived from --steps"
        ),
    )
    value.add_argument("--seed", type=int, default=20260821)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--partner")
    value.add_argument(
        "--inference-batching",
        choices=INFERENCE_BATCHING_MODES,
        default=None,
        help=(
            "Full Action inference schedule; defaults to the accepted predictor-only "
            "batching path (Partial defaults to legacy)"
        ),
    )
    value.add_argument(
        "--inference-shadow-audit",
        action="store_true",
        help="expensive bounded legacy shadow used only to locate batching divergence",
    )
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--actor-lr", type=float, default=5e-4)
    value.add_argument("--critic-lr", type=float, default=5e-4)
    value.add_argument("--clip-param", type=float, default=0.2)
    value.add_argument("--gamma", type=float, default=0.99)
    value.add_argument("--gae-lambda", type=float, default=0.95)
    value.add_argument(
        "--entropy-coefs", type=float, nargs="+", default=list(DEFAULT_ENTROPY_COEFS),
        action=_StoreExplicit,
    )
    value.add_argument(
        "--entropy-coef-horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENTROPY_HORIZONS),
        action=_StoreExplicit,
    )
    value.add_argument("--ppo-epochs", type=int, default=15)
    value.add_argument("--num-mini-batches", type=int, default=1)
    value.add_argument("--max-grad-norm", type=float, default=10.0)
    value.add_argument("--wandb-mode", choices=WANDB_MODES, default="disabled")
    value.add_argument("--wandb-project", default="overcooked-final-model")
    value.add_argument("--smoke", action="store_true")
    return value


def _canonicalize_training_components(args) -> None:
    if getattr(args, "_training_components_canonicalized", False):
        return
    legacy = args.training_stage
    supplied_components = args.train_components
    if supplied_components is not None:
        if len(set(supplied_components)) != len(supplied_components):
            raise ValueError("--train-components may not contain duplicates")
        components = frozenset(supplied_components)
    elif legacy is not None:
        components = frozenset((legacy,))
    else:
        raise ValueError("provide --train-components or legacy --training-stage")

    if args.observation_mode == "full":
        if components == {"action"} and args.gaze_type is None:
            canonical_gaze_type = None
            args.ground_truth_policy_input = False
        elif components in ({"gaze"}, {"action", "gaze"}) and args.gaze_type in (
            None, "learned"
        ):
            canonical_gaze_type = "learned"
            args.ground_truth_policy_input = True
        else:
            raise ValueError(
                "Full mode supports Action-only, learned Gaze-only, or learned joint training"
            )
    else:
        args.ground_truth_policy_input = False
        if supplied_components is not None and args.gaze_type is None:
            raise ValueError("Partial --train-components requires --gaze-type")
        canonical_gaze_type = args.gaze_type or (
            "rule_based" if legacy == "action" else "learned"
        )
        if canonical_gaze_type == "rule_based" and components != {"action"}:
            raise ValueError("rule_based gaze supports only train-components action")
        if canonical_gaze_type == "learned" and components not in (
            {"action"}, {"gaze"}, {"action", "gaze"}
        ):
            raise ValueError("invalid learned-gaze train-components")

    legacy_components = None if legacy is None else frozenset((legacy,))
    if legacy_components is not None and supplied_components is not None:
        if legacy_components != components:
            raise ValueError("legacy --training-stage conflicts with --train-components")
        legacy_gaze = "rule_based" if legacy == "action" else "learned"
        if args.observation_mode == "partial" and canonical_gaze_type != legacy_gaze:
            raise ValueError("legacy --training-stage conflicts with --gaze-type")

    args.gaze_type = canonical_gaze_type
    args.train_action = "action" in components
    args.train_gaze = "gaze" in components
    args.train_components = [
        value for value in ("action", "gaze") if value in components
    ]
    args.training_stage = (
        "joint" if args.train_action and args.train_gaze
        else "action" if args.train_action else "gaze"
    )
    if (
        args.train_gaze
        and not args.train_action
        and not getattr(args, "_entropy_coefs_explicit", False)
        and not getattr(args, "_entropy_coef_horizons_explicit", False)
    ):
        args.entropy_coefs = list(GAZE_ENTROPY_COEFS)
        args.entropy_coef_horizons = list(GAZE_ENTROPY_HORIZONS)
    args._training_components_canonicalized = True


def validate_combination(args) -> None:
    _canonicalize_training_components(args)
    full_action = (
        args.observation_mode == "full" and args.train_action and not args.train_gaze
    )
    full_gaze = args.observation_mode == "full" and args.train_gaze
    gaze_runtime = args.observation_mode == "partial" or full_gaze
    constraint_config = getattr(args, "_resolved_constraint_config", None)
    if constraint_config is None:
        constraint_config = resolve_constraint_config(args)
    if (
        full_action
        and constraint_config != BASELINE_CONSTRAINT_CONFIG
    ):
        raise ValueError("human-like constraints are only valid for Partial mode")
    if not hasattr(args, "_resolved_constraint_config"):
        args._resolved_constraint_config = constraint_config
        args.constraint_profile = constraint_config.profile
        args.peripheral_mode = constraint_config.peripheral.mode
        args.peripheral_sigma_deg = constraint_config.peripheral.sigma_deg
        args.peripheral_extent = constraint_config.peripheral.extent
        args.emma_mode = constraint_config.emma.mode
        args.emma_prep_ms = constraint_config.emma.prep_ms
        args.emma_exec_intercept_ms = constraint_config.emma.exec_intercept_ms
        args.emma_exec_slope_ms_per_degree = (
            constraint_config.emma.exec_slope_ms_per_degree
        )
        args.emma_dwell_ms = constraint_config.emma.dwell_ms
        args.preview_opacity_mode = constraint_config.preview_opacity.mode
    args.rule_gaze_probabilities = validate_rule_probabilities(
        {
            "task": args.rule_gaze_task_prob,
            "stay": args.rule_gaze_stay_prob,
            "stale": args.rule_gaze_stale_prob,
            "random": args.rule_gaze_random_prob,
        }
    )
    if args.reward_shaping_horizon_steps < 0:
        raise ValueError("--reward-shaping-horizon-steps cannot be negative")
    if args.memory_decay_mode == "none":
        if args.memory_lifetime_ms is not None:
            raise ValueError(
                "--memory-lifetime-ms cannot be combined with "
                "--memory-decay-mode none"
            )
        if args.memory_tau_ms is not None or args.memory_max_lifetime_ms is not None:
            raise ValueError("no-decay memory rejects tau and max lifetime")
    elif args.memory_decay_mode == "stochastic_exponential":
        if (
            args.memory_lifetime_ms is not None
            and args.memory_lifetime_ms != args.memory_max_lifetime_ms
        ):
            raise ValueError(
                "stochastic exponential memory uses --memory-max-lifetime-ms"
            )
        if args.memory_tau_ms is None or args.memory_max_lifetime_ms is None:
            raise ValueError(
                "stochastic exponential memory requires --memory-tau-ms and "
                "--memory-max-lifetime-ms"
            )
        if args.memory_max_lifetime_ms not in (1200.0, 2400.0):
            raise ValueError("memory max lifetime must be 1200 or 2400 ms")
        if not math.isclose(
            args.memory_tau_ms,
            args.memory_max_lifetime_ms / 3.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("stochastic memory requires tau = max lifetime / 3")
        args.memory_lifetime_ms = args.memory_max_lifetime_ms
    else:
        if args.memory_tau_ms is not None or args.memory_max_lifetime_ms is not None:
            raise ValueError("linear memory uses --memory-lifetime-ms only")
        if args.memory_lifetime_ms is None:
            args.memory_lifetime_ms = MEMORY_LIFETIME_MS
    if args.memory_decay_mode == "linear" and (
        not math.isfinite(args.memory_lifetime_ms)
        or args.memory_lifetime_ms <= 0.0
    ):
        raise ValueError("--memory-lifetime-ms must be finite and positive")
    if args.shared_horizons is not None:
        if len(set(args.shared_horizons)) != len(args.shared_horizons):
            raise ValueError("--shared-horizons may not contain duplicates")
        args.shared_horizons = sorted(args.shared_horizons)
        if args.observation_mode == "full" and args.shared_horizons != [0, 2, 4]:
            raise ValueError("Full --shared-horizons must be exactly: 0 2 4")
    if args.inference_batching is None:
        args.inference_batching = (
            "predictor"
            if full_action
            else "legacy"
        )
    initialization_paths = tuple(
        path
        for path in (
            args.source_checkpoint,
            args.resume_checkpoint,
            args.hsp_initial_checkpoint,
        )
        if path is not None
    )
    if len(initialization_paths) > 1:
        raise ValueError(
            "--source-checkpoint, --resume-checkpoint, and "
            "--hsp-initial-checkpoint are mutually exclusive"
        )
    if full_action:
        if args.inference_batching not in FULL_INFERENCE_BATCHING_MODES:
            raise ValueError(
                "Full Action inference batching must be one of "
                f"{FULL_INFERENCE_BATCHING_MODES}"
            )
    elif args.gaze_type == "learned":
        if args.inference_batching not in PARTIAL_LEARNED_PRODUCTION_MODES:
            raise ValueError(
                "learned Partial production excludes gaze/ActionNet batching because "
                "it cannot preserve the legacy global Torch RNG call order; use one of "
                f"{PARTIAL_LEARNED_PRODUCTION_MODES}"
            )
    elif args.inference_batching != "legacy":
        raise ValueError("rule-based Partial supports only legacy inference")
    if args.inference_shadow_audit and args.inference_batching == "legacy":
        raise ValueError("inference shadow audit requires a batched inference mode")
    if gaze_runtime:
        if args.shared_decision_times is not None and len(
            set(args.shared_decision_times)
        ) != len(args.shared_decision_times):
            raise ValueError("--shared-decision-times may not contain duplicates")
        if args.shared_fixation_shapes is not None and len(
            set(args.shared_fixation_shapes)
        ) != len(args.shared_fixation_shapes):
            raise ValueError("--shared-fixation-shapes may not contain duplicates")
        if (args.decision_time is None) == (args.shared_decision_times is None):
            raise ValueError(
                "Partial mode requires exactly one of --decision-time or --shared-decision-times"
            )
        if (args.fixation_shape is None) == (args.shared_fixation_shapes is None):
            raise ValueError(
                "Partial mode requires exactly one of --fixation-shape or --shared-fixation-shapes"
            )
        args.partial_decision_times = (
            [args.decision_time]
            if args.shared_decision_times is None
            else sorted(args.shared_decision_times)
        )
        args.partial_horizons = (
            [args.horizon] if args.shared_horizons is None else list(args.shared_horizons)
        )
        args.partial_fixation_shapes = (
            [args.fixation_shape]
            if args.shared_fixation_shapes is None
            else list(args.shared_fixation_shapes)
        )
        if (
            constraint_config.peripheral.mode != "none"
            and args.partial_fixation_shapes != ["plus"]
        ):
            raise ValueError(
                "active peripheral vision requires exactly --fixation-shape plus"
            )
    else:
        if (
            args.memory_decay_mode != "linear"
            or args.memory_lifetime_ms != MEMORY_LIFETIME_MS
        ):
            raise ValueError("memory decay options are only valid for Partial mode")
        if any(
            value is not None
            for value in (
                args.decision_time, args.shared_decision_times,
                args.fixation_shape, args.shared_fixation_shapes,
            )
        ):
            raise ValueError("Full mode does not accept Partial condition arguments")
        args.partial_decision_times = None
        args.partial_horizons = None
        args.partial_fixation_shapes = None
    if (
        gaze_runtime
        and args.gaze_type == "learned"
        and args.source_checkpoint is None
        and args.resume_checkpoint is None
    ):
        raise ValueError(
            "Partial Gaze requires valid learned --source-checkpoint initialization "
            "or --resume-checkpoint"
        )
    if args.direct_joint_from_full and not (
        args.observation_mode == "partial"
        and args.gaze_type == "learned"
        and args.train_action
        and args.train_gaze
    ):
        raise ValueError(
            "--direct-joint-from-full requires Partial learned "
            "--train-components action gaze"
        )
    if (
        args.direct_joint_from_full
        and args.resume_checkpoint is None
        and args.source_checkpoint is None
    ):
        raise ValueError("--direct-joint-from-full requires --source-checkpoint")
    if full_action and args.source_checkpoint is not None:
        raise ValueError("fresh Full Action does not accept --source-checkpoint")
    if args.hsp_initial_checkpoint is not None and (
        not full_action
    ):
        raise ValueError("--hsp-initial-checkpoint is only valid for Full Action")
    if args.observation_mode in {"full", "partial"}:
        if args.partner is not None:
            raise ValueError("canonical training does not accept --partner")
        if args.fixed_partner_checkpoint is None:
            raise ValueError("canonical training requires --fixed-partner-checkpoint")
        partner_sha256 = file_sha256(args.fixed_partner_checkpoint)
        if partner_sha256 != FIXED_HSP_PARTNER_SHA256:
            raise ValueError(
                "--fixed-partner-checkpoint must be canonical HSP S2-s24 seed-3"
            )
    if full_action:
        if args.resume_checkpoint is None:
            if args.hsp_initial_checkpoint is None:
                raise ValueError("new canonical Full Action requires --hsp-initial-checkpoint")
            learner_sha256 = file_sha256(args.hsp_initial_checkpoint)
            if learner_sha256 != HSP_SOURCE_CHECKPOINT_SHA256:
                raise ValueError(
                    "--hsp-initial-checkpoint must be canonical HSP S2-s36 seed-4"
                )
            if learner_sha256 == partner_sha256:
                raise ValueError("learner initialization and fixed partner must differ")
    update_steps = args.rollout_length * args.n_rollout_threads
    if (
        args.steps <= 0
        or args.rollout_length <= 0
        or args.n_rollout_threads <= 0
        or args.steps % update_steps
    ):
        raise ValueError(
            "steps must be a positive multiple of rollout-length * n-rollout-threads"
        )
    if len(args.entropy_coefs) != len(args.entropy_coef_horizons):
        raise ValueError("entropy schedule coefficients and horizons must have equal lengths")
    if args.entropy_coef_horizons[0] != 0 or any(
        right <= left
        for left, right in zip(
            args.entropy_coef_horizons, args.entropy_coef_horizons[1:]
        )
    ):
        raise ValueError("entropy schedule horizons must start at zero and increase")


def _ppo(args):
    return {
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "clip_param": args.clip_param,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "entropy_coefs": args.entropy_coefs,
        "entropy_coef_horizons": args.entropy_coef_horizons,
        "ppo_epochs": args.ppo_epochs,
        "num_mini_batches": args.num_mini_batches,
        "max_grad_norm": args.max_grad_norm,
    }


def _constraint_config(args):
    return getattr(args, "_resolved_constraint_config", None) or (
        resolve_constraint_config(args)
    )


def _emma_timing(args):
    return _constraint_config(args).emma


def _session(args):
    common = dict(
        seed=args.seed,
        rollout_length=args.rollout_length,
        episode_length=args.episode_length,
        n_rollout_threads=args.n_rollout_threads,
        device=args.device,
        partner=args.partner,
        **_ppo(args),
    )
    if args.observation_mode == "full" and args.train_action and not args.train_gaze:
        return ActionTrainingSession(
            observation_mode=args.observation_mode,
            horizon=args.horizon,
            decision_time=args.decision_time,
            fixation_shape=args.fixation_shape,
            source_checkpoint=args.source_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
            hsp_initial_checkpoint=args.hsp_initial_checkpoint,
            fixed_partner_checkpoint=args.fixed_partner_checkpoint,
            configured_total_steps=args.steps,
            checkpoint_interval=args.checkpoint_interval,
            reward_shaping_horizon_steps=args.reward_shaping_horizon_steps,
            inference_batching=args.inference_batching,
            inference_shadow_audit=args.inference_shadow_audit,
            shared_horizons=args.shared_horizons,
            **common,
        )
    return PartialTrainingSession(
        gaze_type=args.gaze_type,
        train_action=args.train_action,
        train_gaze=args.train_gaze,
        decision_times=args.partial_decision_times,
        horizons=args.partial_horizons,
        fixation_shapes=args.partial_fixation_shapes,
        source_checkpoint=args.source_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        fixed_partner_checkpoint=args.fixed_partner_checkpoint,
        configured_total_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        reward_shaping_horizon_steps=args.reward_shaping_horizon_steps,
        rule_gaze_probabilities=args.rule_gaze_probabilities,
        direct_joint_from_full=args.direct_joint_from_full,
        ground_truth_policy_input=args.ground_truth_policy_input,
        memory_lifetime_ms=args.memory_lifetime_ms,
        memory_decay_mode=args.memory_decay_mode,
        memory_tau_ms=args.memory_tau_ms,
        constraint_config=_constraint_config(args),
        inference_batching=args.inference_batching,
        inference_shadow_audit=args.inference_shadow_audit,
        **common,
    )


def _wandb_metrics(latest, stage):
    if latest["observation_mode"] == "partial":
        action_active = bool(latest["train_action"])
        entropy = latest["dist_entropy"] if action_active else latest["gaze_entropy"]
        policy_loss = latest["policy_loss"] if action_active else latest["gaze_policy_loss"]
        value_loss = latest["value_loss"] if action_active else latest["gaze_value_loss"]
        approx_kl = latest["approx_kl"] if action_active else latest["gaze_approx_kl"]
        clip_fraction = latest["clip_fraction"] if action_active else latest["gaze_clip_fraction"]
        metrics = {
            "train/reward": latest["reward"],
            "train/optimization_reward": latest["optimization_reward"],
            "train/sparse_team_reward": latest["sparse_team_reward"],
            "train/native_shaped_reward": latest["native_shaped_reward"],
            "train/weighted_shaped_reward": latest["weighted_shaped_reward"],
            "train/reward_shaping_factor": latest["reward_shaping_factor"],
            "train/deliveries": latest["deliveries"],
            "train/approx_kl": approx_kl,
            "train/entropy": entropy,
            "train/clip_fraction": clip_fraction,
            "train/actor_loss": policy_loss,
            "train/critic_loss": value_loss,
            "train/environment_steps": latest["environment_steps"],
            "train/task_decisions": latest["task_decisions"],
            "train/updates": latest["updates"],
            "train/episodes": latest["episodes"],
            "train/decisions_per_second": latest["decisions_per_second"],
        }
        for key in (
            "gaze_requests", "acquisitions", "same_target_acquisitions",
            "new_target_acquisitions", "actionnet_queries", "action_proposals",
            "proposal_revisions", "executed_proposals", "forced_stays",
            "no_acquisition_decisions", "recurrent_commits", "valid_memory_fraction",
            "mean_freshness", "memory_refresh_count", "memory_expiry_count",
            "freshness_weight_norm", "freshness_gradient_norm",
            "freshness_parameter_delta_norm", "actor_active_sample_rate",
        ):
            metrics[f"partial/{key}"] = latest[key]
        for source, target in (
            ("peripheral_attempts", "perception/peripheral_attempts"),
            ("peripheral_successes", "perception/peripheral_successes"),
            ("peripheral_success_rate", "perception/peripheral_success_rate"),
            (
                "peripheral_successes_per_decision",
                "perception/peripheral_successes_per_decision",
            ),
            (
                "mean_peripheral_success_eccentricity_deg",
                "perception/mean_success_eccentricity_deg",
            ),
        ):
            metrics[target] = latest[source]
        for source, target in (
            ("opacity_attempts", "perception/opacity_attempts"),
            ("opacity_successes", "perception/opacity_successes"),
            ("opacity_success_rate", "perception/opacity_success_rate"),
            (
                "opacity_detected_previews_per_decision",
                "perception/opacity_detected_previews_per_decision",
            ),
        ):
            metrics[target] = latest[source]
        for source, target in (
            ("gaze_records_per_task_decision_mean", "gaze/records_per_task_decision_mean"),
            ("gaze_records_per_task_decision_max", "gaze/records_per_task_decision_max"),
            ("gaze_unique_task_decisions", "gaze/unique_task_decisions"),
            ("gaze_decision_weight_sum_min", "gaze/decision_weight_sum_min"),
            ("gaze_decision_weight_sum_max", "gaze/decision_weight_sum_max"),
            ("gaze_decision_weight_sum_mean", "gaze/decision_weight_sum_mean"),
            ("gaze_reward_conservation_error", "gaze/reward_conservation_error"),
            ("gaze_env_boundary_count", "gaze/env_boundary_count"),
            ("gaze_weighted_advantage_mean", "gaze/weighted_advantage_mean"),
            ("gaze_weighted_advantage_std", "gaze/weighted_advantage_std"),
            ("gaze_unweighted_advantage_mean", "gaze/unweighted_advantage_mean"),
            ("gaze_unweighted_advantage_std", "gaze/unweighted_advantage_std"),
            ("gaze_weighted_approx_kl", "ppo/weighted_approx_kl"),
            ("gaze_weighted_clip_fraction", "ppo/weighted_clip_fraction"),
            ("gaze_weighted_entropy", "gaze/weighted_entropy"),
        ):
            metrics[target] = latest[source]
        metrics.update(flatten_partial_metric_namespaces(latest))
        return metrics
    entropy = latest["dist_entropy"] if stage == "action" else latest["gaze_entropy"]
    metrics = {
        "train/reward": latest["reward"],
        "train/optimization_reward": latest["optimization_reward"],
        "train/sparse_team_reward": latest["sparse_team_reward"],
        "train/native_shaped_reward": latest["native_shaped_reward"],
        "train/weighted_shaped_reward": latest["weighted_shaped_reward"],
        "train/reward_shaping_factor": latest["reward_shaping_factor"],
        "train/deliveries": latest["deliveries"],
        "train/approx_kl": latest["approx_kl"],
        "train/entropy": entropy,
        "train/clip_fraction": latest["clip_fraction"],
        "train/actor_loss": latest["policy_loss"],
        "train/critic_loss": latest["value_loss"],
        "train/environment_steps": latest["environment_steps"],
        "train/task_decisions": latest["task_decisions"],
        "train/updates": latest["updates"],
        "train/episodes": latest["episodes"],
        "train/decisions_per_second": latest["decisions_per_second"],
    }
    metrics[f"train/{stage}_entropy"] = entropy
    if stage == "action":
        metrics.update(
            {
                "train/reward_identity_error": latest["reward_identity_error"],
                "train/shaping_factor": latest["shaping_factor"],
                "train/policy_loss": latest["policy_loss"],
                "train/value_loss": latest["value_loss"],
                "train/entropy_coef": latest["entropy_coef"],
                "train/explained_variance": latest["explained_variance"],
                "train/actor_grad_norm": latest["actor_grad_norm"],
                "train/critic_grad_norm": latest["critic_grad_norm"],
                "train/grad_clip_fraction": latest["grad_clip_fraction"],
                "train/global_step": latest["global_step"],
                "train/update_count": latest["update_count"],
                "system/steps_per_second": latest["steps_per_second"],
                "system/rollout_seconds": latest["rollout_collection_seconds"],
                "system/update_seconds": latest["ppo_update_seconds"],
                "system/preupdate_audit_seconds": latest[
                    "preupdate_audit_seconds"
                ],
                "system/partner_inference_seconds": latest[
                    "partner_inference_seconds"
                ],
                "system/predictor_inference_seconds": latest[
                    "predictor_inference_seconds"
                ],
                "system/env_step_seconds": latest["env_step_seconds"],
                "system/partner_forward_calls": latest["partner_forward_calls"],
                "system/predictor_forward_calls": latest["predictor_forward_calls"],
                "system/partner_batch_size": latest["partner_batch_size"],
                "system/predictor_batch_size": latest["predictor_batch_size"],
            }
        )
        shared = latest.get("shared_horizon_metrics")
        if shared is not None:
            for label in ("h0", "h2", "h4"):
                bucket = shared[label]
                for key in (
                    "episodes", "task_decisions", "deliveries",
                    "deliveries_per_episode", "sparse_team_reward",
                    "sparse_team_reward_per_episode", "native_shaped_reward",
                    "optimization_reward",
                ):
                    metrics[f"train/{label}/{key}"] = bucket[key]
                metrics[f"train/shared_horizon_episode_count_{label}"] = latest[
                    "shared_horizon_episode_counts"
                ][label]
                metrics[f"train/shared_horizon_decision_count_{label}"] = latest[
                    "shared_horizon_decision_counts"
                ][label]
        partner_parity = latest.get("partner_batch_parity")
        if partner_parity:
            for key in (
                "partner_max_abs_logit_difference",
                "partner_mean_abs_logit_difference",
                "partner_max_abs_probability_difference",
                "partner_mean_abs_probability_difference",
                "partner_max_abs_next_hidden_difference",
                "partner_top1_action_agreement",
                "partner_paired_sampled_action_agreement",
            ):
                metrics[f"audit/{key}"] = partner_parity[key]
        predictor_parity = latest.get("predictor_batch_parity")
        if predictor_parity:
            metrics["audit/predictor_max_abs_logit_difference"] = predictor_parity[
                "predictor_max_abs_logit_difference"
            ]
            metrics["audit/predictor_mean_abs_logit_difference"] = predictor_parity[
                "predictor_mean_abs_logit_difference"
            ]
            metrics["audit/predictor_displayed_preview_agreement"] = (
                predictor_parity["displayed_preview_agreement"]
            )
        for key in (
            "preupdate_parity_pass",
            "preupdate_transition_count",
            "preupdate_max_abs_logprob_difference",
            "preupdate_mean_abs_logprob_difference",
            "preupdate_direct_state_max_abs_logprob_difference",
            "preupdate_sequence_max_abs_logprob_difference",
            "preupdate_replay_max_abs_logprob_difference",
            "preupdate_singleton_context_max_abs_logprob_difference",
            "preupdate_singleton_context_is_canonical",
            "preupdate_ratio_min",
            "preupdate_ratio_p01",
            "preupdate_ratio_p05",
            "preupdate_ratio_p50",
            "preupdate_ratio_p95",
            "preupdate_ratio_p99",
            "preupdate_ratio_max",
            "preupdate_ratio_mean",
            "preupdate_approx_kl",
            "preupdate_clip_fraction",
            "preupdate_lower_clip_fraction",
            "preupdate_upper_clip_fraction",
            "preupdate_observation_hash_match",
            "preupdate_observation_exact_match",
            "preupdate_immediate_insert_checks",
            "preupdate_post_actor_forward_checks",
            "preupdate_source_buffer_alias_count",
            "preupdate_capture_buffer_alias_count",
            "preupdate_transition_id_exact_match",
            "preupdate_actor_parameter_immutable",
            "preupdate_actor_eval_mode_exact_match",
            "preupdate_base_to_gru_exact_match",
            "preupdate_action_exact_match",
            "preupdate_rnn_state_exact_match",
            "preupdate_recurrent_mask_exact_match",
            "preupdate_legal_mask_exact_match",
            "preupdate_max_legal_mask_difference",
            "preupdate_buffer_batch_max_abs_logprob_difference",
            "preupdate_capture_direct_max_abs_logprob_difference",
            "preupdate_flattened_context_max_abs_logprob_difference",
            "preupdate_flattened_context_transition_id",
            "preupdate_flattened_context_is_canonical",
            "preupdate_logprob_tolerance",
        ):
            metrics[f"audit/{key}"] = latest[key]
    for key in (
        "partner_step_guard_checks", "partner_step_guard_failures",
        "partner_full_hash_checks", "partner_full_hash_failures",
    ):
        metrics[f"audit/{key}"] = latest[key]
    metrics["audit/partner_optimizer_isolation_verified"] = int(
        latest["partner_optimizer_isolation_verified"]
    )
    if latest["observation_mode"] == "partial":
        for key in (
            "gaze_requests", "acquisitions", "same_target_acquisitions",
            "new_target_acquisitions", "valid_memory_fraction",
            "mean_saccade_distance_degrees", "prep_count", "exec_count", "dwell_count",
        ):
            metrics[f"partial/{key}"] = latest[key]
        for key in ("rule_exploration_selections", "rule_monitoring_selections"):
            if key in latest:
                metrics[f"partial/{key}"] = latest[key]
    else:
        for key in (
            "freshness_weight_norm",
            "preview_weight_norm_h1",
            "preview_weight_norm_h2",
            "preview_weight_norm_h3",
            "preview_weight_norm_h4",
        ):
            metrics[f"train/{key}"] = latest[key]
    return metrics


def run_training(args) -> dict:
    validate_combination(args)
    constraint_config = _constraint_config(args)
    emma_timing = constraint_config.emma
    name = canonical_run_name(
        args.observation_mode, args.training_stage,
        args.horizon if args.shared_horizons is None else "shared_" + "_".join(map(str, args.shared_horizons)),
        args.decision_time if args.shared_decision_times is None else "shared_" + "_".join(map(str, args.shared_decision_times)),
        args.fixation_shape if args.shared_fixation_shapes is None else "shared_" + "_".join(args.shared_fixation_shapes),
        args.seed, args.steps,
    )
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
    elif (
        args.resume_checkpoint is not None
        and args.resume_checkpoint.resolve().parent.name == "checkpoints"
    ):
        run_dir = args.resume_checkpoint.resolve().parent.parent
    else:
        run_dir = (Path(__file__).resolve().parent / "runs" / name).resolve()
    for child in ("checkpoints", "logs", "metrics"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.pop("_resolved_constraint_config", None)
    config.update(
        {
            "horizon_mode": (
                "fixed"
                if args.shared_horizons is None
                else "episode_balanced"
                if args.observation_mode == "full" and args.train_action and not args.train_gaze
                else "cartesian_shuffled_blocks"
            ),
            "horizons": args.shared_horizons,
            "configured_total_steps": args.steps,
            "entropy_schedule": list(args.entropy_coefs),
            "entropy_schedule_horizons": list(args.entropy_coef_horizons),
            "reward_shaping_schedule": REWARD_SHAPING_SCHEDULE,
            "reward_shaping_initial": REWARD_SHAPING_INITIAL,
            "reward_shaping_final": REWARD_SHAPING_FINAL,
            "reward_shaping_horizon_steps": args.reward_shaping_horizon_steps,
            "train_time_evaluation": False,
            **cooking_contract(),
        }
    )
    if args.observation_mode == "full" and args.train_action and not args.train_gaze:
        config.update(
            {
                "rollout_policy_consistency_audit": "pre_optimizer_fail_closed_v2",
                "observation_capture_storage": (
                    "independent_copy_and_uint8_sha256_v2"
                ),
                "preupdate_logprob_absolute_tolerance": PREUPDATE_LOGPROB_ATOL,
                "recurrent_minibatch_order": (
                    "time_major_within_shuffled_env_chunks_v1"
                ),
                "actor_forward_context": ACTOR_FORWARD_CONTEXT_ID,
                "actor_forward_microbatch_size": args.n_rollout_threads,
            }
        )
    config["source_checkpoint"] = None if args.source_checkpoint is None else str(args.source_checkpoint)
    config["resume_checkpoint"] = None if args.resume_checkpoint is None else str(args.resume_checkpoint)
    config["hsp_initial_checkpoint"] = (
        None
        if args.hsp_initial_checkpoint is None
        else str(args.hsp_initial_checkpoint)
    )
    config["fixed_partner_checkpoint"] = (
        None
        if args.fixed_partner_checkpoint is None
        else str(args.fixed_partner_checkpoint)
    )
    config["run_dir"] = str(run_dir)
    if args.observation_mode == "partial" or args.ground_truth_policy_input:
        config["gaze_policy"] = (
            RULE_BASED_GAZE_POLICY_ID if args.gaze_type == "rule_based" else "learned_gaze_policy"
        )
        config.update(constraint_config.metadata())
        config.update({
            "constraint/profile": constraint_config.profile,
            "constraint/variant_id": constraint_config.variant_id,
            "constraint/peripheral_mode": constraint_config.peripheral.mode,
            "constraint/peripheral_sigma_deg": (
                constraint_config.peripheral.sigma_deg
            ),
            "constraint/emma_mode": emma_timing.mode,
            "constraint/preview_opacity_mode": constraint_config.preview_opacity.mode,
            "emma/mode": emma_timing.mode,
            "emma/prep_ms": emma_timing.prep_ms,
            "emma/exec_intercept_ms": emma_timing.exec_intercept_ms,
            "emma/exec_slope_ms_per_degree": emma_timing.exec_slope_ms_per_degree,
            "emma/dwell_ms": emma_timing.dwell_ms,
        })
        if args.memory_decay_mode == "stochastic_exponential":
            config.update({
                "memory_tau_ms": args.memory_tau_ms,
                "memory_max_lifetime_ms": args.memory_max_lifetime_ms,
                "memory_random_seed": args.seed + MEMORY_RNG_SEED_OFFSET,
                "memory_rng_contract_id": MEMORY_RNG_CONTRACT_ID,
                "memory_rng_seed_offset": MEMORY_RNG_SEED_OFFSET,
            })
        print(
            "[CONSTRAINTS]\n"
            f"profile={constraint_config.profile}\n"
            f"variant={constraint_config.variant_id}\n"
            f"peripheral={constraint_config.peripheral.mode}\n"
            f"peripheral_sigma_deg={constraint_config.peripheral.sigma_deg}\n"
            f"emma={emma_timing.mode}\n"
            f"opacity={constraint_config.preview_opacity.mode}"
        )
        if emma_timing.mode == "tuned":
            print(
                "[EMMA]\n"
                f"prep_ms={emma_timing.prep_ms:g}\n"
                f"exec_intercept_ms={emma_timing.exec_intercept_ms:g}\n"
                "exec_slope_ms_per_degree="
                f"{emma_timing.exec_slope_ms_per_degree:g}\n"
                f"dwell_ms={emma_timing.dwell_ms:g}"
            )
    session = _session(args)
    if (
        args.observation_mode == "partial" or args.ground_truth_policy_input
    ) and session.source_provenance:
        source_constraints = session.source_provenance.get(
            "source_constraint_config"
        )
        if source_constraints is not None:
            print(
                "[CONSTRAINT TRANSFER]\n"
                f"source_variant={source_constraints.get('constraint_variant_id')}\n"
                f"target_variant={constraint_config.variant_id}\n"
                f"source_peripheral={source_constraints.get('peripheral_mode')}\n"
                f"target_peripheral={constraint_config.peripheral.mode}\n"
                "source_opacity="
                f"{source_constraints.get('preview_opacity_mode')}\n"
                "target_opacity="
                f"{constraint_config.preview_opacity.mode}\n"
                "target_sigma_deg="
                f"{constraint_config.peripheral.sigma_deg}\n"
                f"source={json.dumps(source_constraints, sort_keys=True)}\n"
                "target="
                f"{json.dumps(constraint_config.metadata(), sort_keys=True)}"
            )
    if args.resume_checkpoint is not None and args.steps <= session.completed_steps:
        raise ValueError(
            "--steps is the final total environment-step target and must exceed the "
            f"resumed checkpoint step ({session.completed_steps})"
        )
    step0_checkpoint = None
    if (
        args.observation_mode == "full"
        and args.training_stage == "action"
        and args.resume_checkpoint is None
    ):
        step0_checkpoint = run_dir / "checkpoints/step_000000000.pt"
        if step0_checkpoint.exists():
            raise FileExistsError(f"refusing to overwrite step-0 checkpoint: {step0_checkpoint}")
        session.save_initialization_checkpoint(step0_checkpoint)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    tracker = WandbTracker(
        mode=args.wandb_mode, project=args.wandb_project, name=name, config=config
    )
    if step0_checkpoint is not None:
        tracker.log_auxiliary(
            {
                "train/shaping_factor": 1.0,
                "train/reward_shaping_factor": 1.0,
                "train/entropy_coef": float(session.trainer.entropy_coef),
                "train/global_step": 0,
                "train/update_count": 0,
                **{
                    f"train/{key}": value
                    for key, value in session.input_channel_weight_metrics().items()
                },
            },
            environment_steps=0,
        )
    metrics_path = run_dir / "metrics/metrics.jsonl"
    latest = None
    next_checkpoint = (
        session.completed_steps // args.checkpoint_interval + 1
    ) * args.checkpoint_interval
    started = time.perf_counter()
    try:
        with tqdm(
            total=args.steps,
            initial=session.completed_steps,
            desc=name,
            unit="decision",
            unit_scale=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as progress:
            while session.completed_steps < args.steps:
                before = session.completed_steps
                latest = session.run_update()
                progress.update(session.completed_steps - before)
                action_active = bool(latest.get("train_action", args.training_stage == "action"))
                entropy = latest["dist_entropy"] if action_active else latest["gaze_entropy"]
                progress.set_postfix(
                    reward=f"{latest['reward']:.2f}",
                    action_entropy=(f"{latest['dist_entropy']:.3f}" if action_active else "frozen"),
                    gaze_entropy=(f"{latest['gaze_entropy']:.3f}" if latest.get("train_gaze", False) else "n/a"),
                    partner=latest["active_partner"],
                )
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(latest, sort_keys=True) + "\n")
                tracker.log(
                    _wandb_metrics(latest, args.training_stage),
                    environment_steps=session.completed_steps,
                )
                if session.completed_steps >= next_checkpoint or session.completed_steps >= args.steps:
                    session.save(run_dir / "checkpoints" / f"step_{session.completed_steps:09d}.pt")
                    while next_checkpoint <= session.completed_steps:
                        next_checkpoint += args.checkpoint_interval
    finally:
        tracker.finish()
    summary = {
        "project": "final_model",
        "observation_mode": args.observation_mode,
        "training_stage": args.training_stage,
        "horizon_mode": (
            session.horizon_mode
            if args.observation_mode == "full" and args.train_action and not args.train_gaze
            else "fixed" if len(session.scheduler.horizons) == 1 else "cartesian_shared"
        ),
        "horizon": args.horizon,
        "horizons": args.shared_horizons,
        "environment_steps": session.completed_steps,
        "updates": session.update_count,
        "wall_clock_seconds": time.perf_counter() - started,
        "latest": latest,
        "long_training_started": not args.smoke,
        "resumed_from": (
            None if args.resume_checkpoint is None else str(args.resume_checkpoint.resolve())
        ),
        "step0_checkpoint": (
            None if step0_checkpoint is None else str(step0_checkpoint.resolve())
        ),
    }
    if args.observation_mode == "partial" or args.ground_truth_policy_input:
        summary.update(constraint_config.metadata())
        summary.update({
            "constraint/profile": constraint_config.profile,
            "constraint/variant_id": constraint_config.variant_id,
            "constraint/peripheral_mode": constraint_config.peripheral.mode,
            "constraint/peripheral_sigma_deg": (
                constraint_config.peripheral.sigma_deg
            ),
            "constraint/emma_mode": emma_timing.mode,
            "constraint/preview_opacity_mode": constraint_config.preview_opacity.mode,
            "emma/prep_ms": emma_timing.prep_ms,
            "emma/exec_intercept_ms": emma_timing.exec_intercept_ms,
            "emma/exec_slope_ms_per_degree": emma_timing.exec_slope_ms_per_degree,
            "emma/dwell_ms": emma_timing.dwell_ms,
        })
    (run_dir / "logs/summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run_training(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
