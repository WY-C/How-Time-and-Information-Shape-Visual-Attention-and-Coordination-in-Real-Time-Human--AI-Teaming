"""Persistent-worker Full/Partial Action and learned-Gaze training."""

from __future__ import annotations

import json
from pathlib import Path
import time

from final_model.server_parallel.parallel_action_training import (
    ParallelGazeTrainingSession,
    ServerParallelActionTrainingSession,
)
from final_model.parallel_gaze_contract import (
    PARALLEL_DIRECT_JOINT_TRAINING_CONTRACT,
    PARALLEL_GAZE_CPU_INTER_OP_THREADS,
    PARALLEL_GAZE_CPU_INTRA_OP_THREADS,
    PARALLEL_GAZE_TRAINING_CONTRACT,
    PARALLEL_STAGED_JOINT_TRAINING_CONTRACT,
)
from final_model.server_parallel.full_training_runner import (
    make_full_session,
    run_full_training,
)
from final_model.server_parallel.observability import (
    TrainingObserver,
    build_update_record,
    resolved_config,
)
from final_model.train import (
    _constraint_config,
    _ppo,
    parser as canonical_parser,
    validate_combination,
)


def parser():
    value = canonical_parser()
    value.description = __doc__
    value.add_argument(
        "--rollout-workers",
        type=int,
        default=None,
        help="persistent spawn workers; Full default 8, Partial default 1",
    )
    value.add_argument(
        "--host-mode",
        choices=("local", "server"),
        default="local",
        help="execution provenance only; it does not change the training contract",
    )
    value.add_argument("--condition-name", default=None)
    value.add_argument("--memory-condition", default=None)
    value.add_argument("--perception-condition", default=None)
    value.add_argument(
        "--torch-intra-op-threads",
        type=int,
        default=2,
        help="Full main-process torch intra-op threads (default: 2)",
    )
    value.add_argument(
        "--torch-inter-op-threads",
        type=int,
        default=1,
        help="Full main-process torch inter-op threads (default: 1)",
    )
    value.add_argument(
        "--fixed-partner-executor-workers",
        type=int,
        default=4,
        help="Full batch-1 fixed-partner executor workers (default: 4)",
    )
    value.add_argument(
        "--server-trace",
        action="store_true",
        help="bounded parity trace; disabled for production throughput",
    )
    progress = value.add_mutually_exclusive_group()
    progress.add_argument(
        "--progress", dest="progress", action="store_true",
        help="show one live tqdm progress update per completed PPO update (default)",
    )
    progress.add_argument(
        "--no-progress", dest="progress", action="store_false",
        help="disable tqdm for tests and non-interactive runs",
    )
    value.set_defaults(progress=True)
    return value


def validate_server_args(args) -> None:
    validate_combination(args)
    full_action = (
        args.observation_mode == "full" and args.train_action and not args.train_gaze
    )
    gaze_runtime = args.observation_mode == "partial" or (
        args.observation_mode == "full" and args.train_gaze
    )
    if args.rollout_workers is None:
        args.rollout_workers = 8 if args.observation_mode == "full" else 1
    if int(args.rollout_workers) <= 0:
        raise ValueError("--rollout-workers must be positive")
    if int(args.rollout_workers) > int(args.n_rollout_threads):
        raise ValueError("--rollout-workers may not exceed --n-rollout-threads")
    if int(args.torch_intra_op_threads) <= 0:
        raise ValueError("--torch-intra-op-threads must be positive")
    if int(args.torch_inter_op_threads) <= 0:
        raise ValueError("--torch-inter-op-threads must be positive")
    if int(args.fixed_partner_executor_workers) <= 0:
        raise ValueError("--fixed-partner-executor-workers must be positive")
    unsupported = {}
    if gaze_runtime:
        if int(args.checkpoint_interval) <= 0:
            raise ValueError("--checkpoint-interval must be positive")
        if (
            args.resume_checkpoint is not None
            and int(args.checkpoint_interval)
            % (int(args.rollout_length) * int(args.n_rollout_threads))
        ):
            raise ValueError(
                "--checkpoint-interval must align with complete PPO updates"
            )
        action_mode = (
            args.gaze_type == "rule_based"
            and args.train_action
            and not args.train_gaze
        )
        gaze_mode = (
            args.gaze_type == "learned"
            and not args.train_action
            and args.train_gaze
        )
        direct_joint_mode = (
            args.gaze_type == "learned"
            and args.train_action
            and args.train_gaze
            and args.direct_joint_from_full
        )
        staged_joint_mode = (
            args.gaze_type == "learned"
            and args.train_action
            and args.train_gaze
            and not args.direct_joint_from_full
        )
        if gaze_mode or direct_joint_mode or staged_joint_mode:
            args.torch_intra_op_threads = PARALLEL_GAZE_CPU_INTRA_OP_THREADS
            args.torch_inter_op_threads = PARALLEL_GAZE_CPU_INTER_OP_THREADS
        if not (
            action_mode or gaze_mode or direct_joint_mode or staged_joint_mode
        ):
            unsupported["train_components"] = args.train_components
        if args.inference_batching != "legacy" or args.inference_shadow_audit:
            unsupported["inference_batching"] = args.inference_batching
        if args.resume_checkpoint is None and args.source_checkpoint is None:
            raise ValueError(
                "new Partial server_parallel training requires --source-checkpoint"
            )
    elif full_action:
        if args.inference_batching != "predictor":
            unsupported["inference_batching"] = args.inference_batching
        if not args.train_action or args.train_gaze:
            unsupported["train_components"] = args.train_components
    if args.partner is not None:
        unsupported["partner"] = args.partner
    if unsupported:
        raise ValueError(
            f"server_parallel unsupported configuration: {unsupported}"
        )


def _session(args):
    if args.observation_mode == "full" and args.train_action and not args.train_gaze:
        return make_full_session(args)
    session_class = (
        ParallelGazeTrainingSession
        if args.gaze_type == "learned" and not args.train_action
        else ServerParallelActionTrainingSession
    )
    return session_class(
        rollout_workers=args.rollout_workers,
        host_mode=args.host_mode,
        condition_name=args.condition_name,
        memory_condition=args.memory_condition,
        perception_condition=args.perception_condition,
        server_trace=args.server_trace,
        gaze_type=args.gaze_type,
        train_action=args.train_action,
        train_gaze=args.train_gaze,
        decision_times=args.partial_decision_times,
        horizons=args.partial_horizons,
        fixation_shapes=args.partial_fixation_shapes,
        seed=args.seed,
        rollout_length=args.rollout_length,
        episode_length=args.episode_length,
        n_rollout_threads=args.n_rollout_threads,
        device=args.device,
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
        partner=args.partner,
        **_ppo(args),
    )


def _save_profiled_checkpoint(
    session,
    path: Path,
    *,
    step: int,
    checkpoint_role: str | None = None,
):
    started = time.perf_counter()
    if checkpoint_role is None:
        result = session.save(path)
    else:
        result = session.save(path, checkpoint_role=checkpoint_role)
    checkpoint_seconds = time.perf_counter() - started
    print(
        f"PROFILE_CHECKPOINT step={int(step)} seconds={checkpoint_seconds:.9f}",
        flush=True,
    )
    return result


def _save_new_partial_initialization(args, session, run_dir: Path) -> Path | None:
    """Persist the validated Full-to-Partial migration before any PPO update."""
    if args.resume_checkpoint is not None:
        return None
    checkpoint_dir = Path(run_dir) / "checkpoints"
    existing = sorted(checkpoint_dir.glob("*.pt"))
    if existing:
        raise FileExistsError(
            "refusing to reuse a Partial run with checkpoints: "
            + ", ".join(str(path) for path in existing)
        )
    step0 = checkpoint_dir / "step_000000000.pt"
    _save_profiled_checkpoint(
        session,
        step0,
        step=0,
        checkpoint_role="pre_training_initialization",
    )
    return step0


def run_training(args) -> dict[str, object]:
    validate_server_args(args)
    if args.observation_mode == "full" and args.train_action and not args.train_gaze:
        return run_full_training(args)
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else (
            Path(__file__).resolve().parent
            / "runs"
            / (
                f"partial_{args.gaze_type}_{args.training_stage}"
                f"_seed{args.seed}_steps{args.steps}"
                f"_envs{args.n_rollout_threads}_workers{args.rollout_workers}"
            )
        ).resolve()
    )
    for child in ("checkpoints", "logs", "reports"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    base_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    base_config["execution_backend"] = (
        PARALLEL_DIRECT_JOINT_TRAINING_CONTRACT
        if args.direct_joint_from_full
        else PARALLEL_STAGED_JOINT_TRAINING_CONTRACT
        if args.train_action and args.train_gaze
        else PARALLEL_GAZE_TRAINING_CONTRACT
        if args.gaze_type == "learned"
        else "server_parallel_v1"
    )
    started = time.perf_counter()
    latest = None
    latest_record = None
    observer = None
    session = _session(args)
    try:
        base_config.update(session.constraint_config.metadata())
        _save_new_partial_initialization(args, session, run_dir)
        startup_report = session.startup_report()
        print(json.dumps(startup_report, indent=2, sort_keys=True), flush=True)
        if args.steps <= session.completed_steps:
            raise ValueError("--steps must exceed completed resume steps")
        update_size = args.rollout_length * args.n_rollout_threads
        if (args.steps - session.completed_steps) % update_size:
            raise ValueError("remaining --steps must be a whole PPO update")
        config = resolved_config(
            base_config, startup_report, session.source_provenance
        )
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shape = (
            session.scheduler.fixation_shapes[0]
            if len(session.scheduler.fixation_shapes) == 1
            else "mixed"
        )
        observer = TrainingObserver(
            metrics_path=run_dir / "reports" / "metrics.jsonl",
            resuming=args.resume_checkpoint is not None,
            progress=args.progress,
            description=f"Partial {shape}",
            total_steps=args.steps,
            initial_steps=session.completed_steps,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_name=(
                f"parallel_partial_{args.training_stage}_{shape}"
                f"_seed{args.seed}_steps{args.steps}"
            ),
            config=config,
        )
        while session.completed_steps < args.steps:
            previous_steps = session.completed_steps
            update_started = time.perf_counter()
            latest = session.run_update()
            update_wall_seconds = time.perf_counter() - update_started
            latest_record = build_update_record(
                latest,
                wall_time_seconds=time.perf_counter() - started,
                update_wall_seconds=update_wall_seconds,
                previous_environment_steps=previous_steps,
            )
            observer.emit(latest_record)
            if (
                session.completed_steps % args.checkpoint_interval == 0
                or session.completed_steps == args.steps
            ):
                _save_profiled_checkpoint(
                    session,
                    run_dir / "checkpoints" / f"step_{session.completed_steps:09d}.pt",
                    step=session.completed_steps,
                )
        summary = {
            "execution_backend": session.execution_backend,
            "observation_mode": args.observation_mode,
            "ground_truth_policy_input": args.ground_truth_policy_input,
            "training_stage": args.training_stage,
            "environment_steps": session.completed_steps,
            "updates": session.update_count,
            "rollout_workers": args.rollout_workers,
            "n_rollout_threads": args.n_rollout_threads,
            "latest": latest,
            "latest_observability": latest_record,
            "wall_clock_seconds": time.perf_counter() - started,
            "long_training_started": not args.smoke,
            "resumed_from": (
                None
                if args.resume_checkpoint is None
                else str(args.resume_checkpoint.resolve())
            ),
            "resume_checkpoint_sha256": (
                None
                if session.resume_checkpoint_provenance is None
                else session.resume_checkpoint_provenance[
                    "resume_checkpoint_content_sha256"
                ]
            ),
            "resume_start_step": (
                None
                if session.resume_checkpoint_provenance is None
                else session.resume_checkpoint_provenance["resume_start_step"]
            ),
            "source_checkpoint_sha256": session.source_provenance[
                "source_content_sha256"
            ],
        }
        (run_dir / "logs" / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        try:
            if observer is not None:
                observer.close()
        finally:
            session.close()


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run_training(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
