"""Bounded/production runner for the server-only Full backend."""

from __future__ import annotations

import json
from pathlib import Path
import time

from final_model.common.overcooked_backend import cooking_contract
from final_model.common.paths import file_sha256
from final_model.common.tracking import canonical_run_name
from final_model.server_parallel.full_action_training import (
    FULL_SERVER_PARALLEL_VERSION,
    ServerParallelFullActionTrainingSession,
)
from final_model.server_parallel.observability import (
    TrainingObserver,
    build_update_record,
    resolved_config,
)
from final_model.train import _ppo


def make_full_session(args) -> ServerParallelFullActionTrainingSession:
    return ServerParallelFullActionTrainingSession(
        rollout_workers=args.rollout_workers,
        torch_intra_op_threads=args.torch_intra_op_threads,
        torch_inter_op_threads=args.torch_inter_op_threads,
        fixed_partner_executor_workers=args.fixed_partner_executor_workers,
        server_trace=args.server_trace,
        observation_mode="full",
        horizon=args.horizon,
        shared_horizons=args.shared_horizons,
        decision_time=None,
        fixation_shape=None,
        seed=args.seed,
        rollout_length=args.rollout_length,
        episode_length=args.episode_length,
        n_rollout_threads=args.n_rollout_threads,
        device=args.device,
        source_checkpoint=None,
        resume_checkpoint=args.resume_checkpoint,
        hsp_initial_checkpoint=args.hsp_initial_checkpoint,
        fixed_partner_checkpoint=args.fixed_partner_checkpoint,
        configured_total_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        reward_shaping_horizon_steps=args.reward_shaping_horizon_steps,
        partner=args.partner,
        inference_batching=args.inference_batching,
        inference_shadow_audit=args.inference_shadow_audit,
        **_ppo(args),
    )


def run_full_training(args) -> dict[str, object]:
    horizon_label = (
        str(args.horizon)
        if args.shared_horizons is None
        else "shared_" + "_".join(map(str, args.shared_horizons))
    )
    name = canonical_run_name(
        "full", "action", horizon_label, None, None, args.seed, args.steps
    ) + f"_workers{args.rollout_workers}"
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else (Path(__file__).resolve().parent / "runs" / name).resolve()
    )
    for child in ("checkpoints", "logs", "reports"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    base_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    base_config.update({
        "execution_backend": FULL_SERVER_PARALLEL_VERSION,
        "horizon_mode": (
            "fixed" if args.shared_horizons is None else "episode_balanced"
        ),
        "configured_total_steps": args.steps,
        "reward_shaping_horizon_steps": args.reward_shaping_horizon_steps,
        "inference_batching": "predictor",
        "partner_inference_schedule": "batch1_per_env",
        "torch_intra_op_threads": args.torch_intra_op_threads,
        "torch_inter_op_threads": args.torch_inter_op_threads,
        "fixed_partner_executor_workers": args.fixed_partner_executor_workers,
        **cooking_contract(),
    })
    session = make_full_session(args)
    startup_report = session.startup_report()
    print(json.dumps(startup_report, indent=2, sort_keys=True), flush=True)
    config = resolved_config(base_config, startup_report)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    latest = None
    latest_record = None
    step0_checkpoint = None
    observer = None
    try:
        if args.steps <= session.completed_steps:
            raise ValueError("--steps must exceed completed resume steps")
        update_size = args.rollout_length * args.n_rollout_threads
        if (args.steps - session.completed_steps) % update_size:
            raise ValueError("remaining --steps must be a whole PPO update")
        if args.resume_checkpoint is None:
            step0_checkpoint = run_dir / "checkpoints/step_000000000.pt"
            if step0_checkpoint.exists():
                raise FileExistsError(
                    f"refusing to overwrite step-0 checkpoint: {step0_checkpoint}"
                )
            session.save_initialization_checkpoint(step0_checkpoint)
        condition = (
            f"H{args.horizon}"
            if args.shared_horizons is None
            else "shared H0/H2/H4"
        )
        observer = TrainingObserver(
            metrics_path=run_dir / "reports/metrics.jsonl",
            resuming=args.resume_checkpoint is not None,
            progress=args.progress,
            description=f"Full {condition}",
            total_steps=args.steps,
            initial_steps=session.completed_steps,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_name=name,
            config=config,
        )
        next_checkpoint = (
            session.completed_steps // args.checkpoint_interval + 1
        ) * args.checkpoint_interval
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
                session.completed_steps >= next_checkpoint
                or session.completed_steps >= args.steps
            ):
                session.save(
                    run_dir
                    / "checkpoints"
                    / f"step_{session.completed_steps:09d}.pt"
                )
                while next_checkpoint <= session.completed_steps:
                    next_checkpoint += args.checkpoint_interval
        summary = {
            "execution_backend": FULL_SERVER_PARALLEL_VERSION,
            "observation_mode": "full",
            "training_stage": "action",
            "horizon": args.horizon,
            "horizons": args.shared_horizons,
            "environment_steps": session.completed_steps,
            "updates": session.update_count,
            "rollout_workers": args.rollout_workers,
            "n_rollout_threads": args.n_rollout_threads,
            "torch_intra_op_threads": args.torch_intra_op_threads,
            "torch_inter_op_threads": args.torch_inter_op_threads,
            "fixed_partner_executor_workers": args.fixed_partner_executor_workers,
            "latest": latest,
            "latest_observability": latest_record,
            "wall_clock_seconds": time.perf_counter() - started,
            "long_training_started": not args.smoke,
            "resumed_from": (
                None
                if args.resume_checkpoint is None
                else str(args.resume_checkpoint.resolve())
            ),
            "hsp_initial_checkpoint_sha256": (
                None
                if args.hsp_initial_checkpoint is None
                else file_sha256(args.hsp_initial_checkpoint)
            ),
            "step0_checkpoint": (
                None if step0_checkpoint is None else str(step0_checkpoint)
            ),
        }
        (run_dir / "logs/summary.json").write_text(
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
