"""One-update CUDA benchmark for canonical and server-parallel Full Action."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from final_model.common.action_training import ActionTrainingSession
from final_model.common.mappo import (
    CANONICAL_ENTROPY_COEFS,
    CANONICAL_ENTROPY_HORIZONS,
)
from final_model.server_parallel.benchmark import ResourceMonitor, _active_trainers
from final_model.server_parallel.full_action_training import (
    ServerParallelFullActionTrainingSession,
)


ROOT = Path(__file__).resolve().parents[2]
HSP_INITIAL = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"
FIXED_PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "reports/full_parallel_benchmark.json"
)


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _options(args) -> dict[str, object]:
    steps = int(args.rollout_length) * int(args.n_rollout_threads)
    return {
        "observation_mode": "full",
        "horizon": None,
        "shared_horizons": [0, 2, 4],
        "decision_time": None,
        "fixation_shape": None,
        "seed": int(args.seed),
        "rollout_length": int(args.rollout_length),
        "episode_length": int(args.episode_length),
        "n_rollout_threads": int(args.n_rollout_threads),
        "device": str(args.device),
        "source_checkpoint": None,
        "resume_checkpoint": None,
        "hsp_initial_checkpoint": HSP_INITIAL,
        "fixed_partner_checkpoint": FIXED_PARTNER,
        "configured_total_steps": steps,
        "checkpoint_interval": steps,
        "reward_shaping_horizon_steps": int(
            args.reward_shaping_horizon_steps
        ),
        "partner": None,
        "inference_batching": "predictor",
        "inference_shadow_audit": False,
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coefs": list(args.entropy_coefs),
        "entropy_coef_horizons": list(args.entropy_coef_horizons),
        "ppo_epochs": int(args.ppo_epochs),
        "num_mini_batches": 1,
        "max_grad_norm": 10.0,
    }


def _failed_row(
    backend: str,
    workers: int | None,
    args,
    error: BaseException,
    *,
    elapsed: float,
) -> dict[str, object]:
    return {
        "backend": backend,
        "rollout_workers": workers,
        "n_rollout_threads": int(args.n_rollout_threads),
        "rollout_length": int(args.rollout_length),
        "environment_steps_target": (
            int(args.n_rollout_threads) * int(args.rollout_length)
        ),
        "ppo_epochs": int(args.ppo_epochs),
        "completed": False,
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "elapsed_seconds_until_failure": float(elapsed),
        "audit_bypassed": False,
        "long_training_started": False,
    }


def _one(backend: str, workers: int | None, args) -> dict[str, object]:
    initialized = time.perf_counter()
    session = None
    try:
        options = _options(args)
        if backend == "canonical_full":
            session = ActionTrainingSession(**options)
            pids = [os.getpid()]
        else:
            session = ServerParallelFullActionTrainingSession(
                rollout_workers=int(workers), server_trace=False, **options
            )
            pids = [os.getpid(), *session.worker_pids]
        startup_seconds = time.perf_counter() - initialized
        _synchronize(args.device)
        with ResourceMonitor(pids) as monitor:
            update_started = time.perf_counter()
            try:
                result = session.run_update()
                _synchronize(args.device)
                error = None
            except BaseException as caught:
                result = None
                error = caught
            total_seconds = time.perf_counter() - update_started
        resources = monitor.report()
        if error is not None:
            return {
                **_failed_row(
                    backend, workers, args, error,
                    elapsed=time.perf_counter() - initialized,
                ),
                "startup_seconds": startup_seconds,
                "total_seconds_until_failure": total_seconds,
                **resources,
            }
        assert result is not None
        steps = int(result["environment_steps"])
        return {
            "backend": backend,
            "rollout_workers": workers,
            "n_rollout_threads": int(args.n_rollout_threads),
            "rollout_length": int(args.rollout_length),
            "environment_steps_target": (
                int(args.n_rollout_threads) * int(args.rollout_length)
            ),
            "environment_steps": steps,
            "joint_action_decisions": steps,
            "ppo_epochs": int(args.ppo_epochs),
            "startup_seconds": startup_seconds,
            "total_update_seconds": total_seconds,
            "environment_steps_per_second": steps / max(total_seconds, 1e-12),
            "joint_action_decisions_per_second": (
                steps / max(total_seconds, 1e-12)
            ),
            "rollout_collection_seconds": result["rollout_collection_seconds"],
            "ppo_update_seconds": result["ppo_update_seconds"],
            "preupdate_audit_seconds": result["preupdate_audit_seconds"],
            "learner_inference_seconds": result.get("learner_inference_seconds"),
            "partner_inference_seconds": result.get("partner_inference_seconds"),
            "predictor_inference_seconds": result.get(
                "predictor_inference_seconds"
            ),
            "env_step_seconds": result.get("env_step_seconds"),
            "ipc_barrier_overhead_seconds": result.get(
                "ipc_barrier_overhead_seconds"
            ),
            "main_process_wait_seconds": result.get(
                "main_process_wait_seconds"
            ),
            "worker_wait_seconds": result.get("worker_wait_seconds"),
            "preupdate_max_abs_logprob_difference": result[
                "preupdate_max_abs_logprob_difference"
            ],
            "preupdate_approx_kl": result["preupdate_approx_kl"],
            "preupdate_clip_fraction": result["preupdate_clip_fraction"],
            "predictor_forward_calls": result.get("predictor_forward_calls"),
            "partner_forward_calls": result.get("partner_forward_calls"),
            "partner_step_guard_failures": result[
                "partner_step_guard_failures"
            ],
            "worker_partner_guard_failures": result.get(
                "worker_partner_guard_failures"
            ),
            "actor_hash_after": result["actor_hash_after"],
            "critic_hash_after": result.get("critic_hash_after"),
            "completed": True,
            "audit_bypassed": False,
            "long_training_started": False,
            **resources,
        }
    except BaseException as error:
        return _failed_row(
            backend,
            workers,
            args,
            error,
            elapsed=time.perf_counter() - initialized,
        )
    finally:
        if session is not None and hasattr(session, "close"):
            session.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _candidate_command(args, backend: str, workers: int | None, output: Path):
    command = [
        sys.executable,
        "-m",
        "final_model.server_parallel.full_benchmark",
        "--candidate-backend",
        backend,
        "--candidate-output",
        str(output),
        "--seed",
        str(args.seed),
        "--n-rollout-threads",
        str(args.n_rollout_threads),
        "--rollout-length",
        str(args.rollout_length),
        "--episode-length",
        str(args.episode_length),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--device",
        str(args.device),
        "--reward-shaping-horizon-steps",
        str(args.reward_shaping_horizon_steps),
        "--entropy-coefs",
        *(str(value) for value in args.entropy_coefs),
        "--entropy-coef-horizons",
        *(str(value) for value in args.entropy_coef_horizons),
    ]
    if workers is not None:
        command.extend(("--candidate-workers", str(workers)))
    return command


def _isolated_candidate(
    args, backend: str, workers: int | None
) -> dict[str, object]:
    label = backend if workers is None else f"{backend}_workers{workers}"
    temporary = args.output.parent / f".{label}.candidate.json"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    command = _candidate_command(args, backend, workers, temporary)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            timeout=float(args.candidate_timeout_seconds),
        )
        if process.returncode != 0:
            raise RuntimeError(f"candidate subprocess exit code {process.returncode}")
        if not temporary.is_file():
            raise RuntimeError("candidate subprocess did not write its result")
        return json.loads(temporary.read_text(encoding="utf-8"))
    except BaseException as error:
        return _failed_row(
            backend,
            workers,
            args,
            error,
            elapsed=time.perf_counter() - started,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _recommend(rows: list[dict[str, object]]) -> tuple[int | None, float | None]:
    completed = [row for row in rows if row.get("completed")]
    if not completed:
        return None, None
    highest = max(float(row["environment_steps_per_second"]) for row in completed)
    eligible = [
        row for row in completed
        if float(row["environment_steps_per_second"]) >= 0.95 * highest
    ]
    recommended = min(int(row["rollout_workers"]) for row in eligible)
    return recommended, highest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16, 32])
    value.add_argument("--seed", type=int, default=20260821)
    value.add_argument("--n-rollout-threads", type=int, default=100)
    value.add_argument("--rollout-length", type=int, default=400)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--ppo-epochs", type=int, default=15)
    value.add_argument("--device", default="cuda")
    value.add_argument("--reward-shaping-horizon-steps", type=int, default=20_000_000)
    value.add_argument(
        "--entropy-coefs", type=float, nargs="+",
        default=list(CANONICAL_ENTROPY_COEFS),
    )
    value.add_argument(
        "--entropy-coef-horizons", type=int, nargs="+",
        default=list(CANONICAL_ENTROPY_HORIZONS),
    )
    value.add_argument("--skip-canonical", action="store_true")
    value.add_argument("--candidate-timeout-seconds", type=float, default=1800.0)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument(
        "--candidate-backend", choices=("canonical_full", "server_parallel_full")
    )
    value.add_argument("--candidate-workers", type=int)
    value.add_argument("--candidate-output", type=Path)
    return value


def _validate(args) -> None:
    if int(args.n_rollout_threads) <= 0 or int(args.rollout_length) <= 0:
        raise ValueError("rollout dimensions must be positive")
    if int(args.ppo_epochs) <= 0:
        raise ValueError("--ppo-epochs must be positive")
    if int(args.reward_shaping_horizon_steps) < 0:
        raise ValueError("--reward-shaping-horizon-steps cannot be negative")
    if any(int(worker) <= 0 for worker in args.workers):
        raise ValueError("workers must be positive")
    if any(int(worker) > int(args.n_rollout_threads) for worker in args.workers):
        raise ValueError("workers may not exceed rollout threads")
    if len(args.entropy_coefs) != len(args.entropy_coef_horizons):
        raise ValueError("entropy coefficients and horizons must have equal lengths")


def run(args) -> dict[str, object]:
    _validate(args)
    active = _active_trainers()
    if active:
        raise RuntimeError(f"benchmark aborted because trainers are active: {active}")
    server_rows = [
        _isolated_candidate(args, "server_parallel_full", int(workers))
        for workers in args.workers
    ]
    canonical = None
    if not args.skip_canonical:
        canonical = _isolated_candidate(args, "canonical_full", None)
    recommended, highest = _recommend(server_rows)
    worker1 = next(
        (row for row in server_rows if row.get("rollout_workers") == 1), None
    )
    selected = next(
        (
            row for row in server_rows
            if row.get("rollout_workers") == recommended
        ),
        None,
    )
    speedup_worker1 = (
        float(selected["environment_steps_per_second"])
        / float(worker1["environment_steps_per_second"])
        if selected is not None and worker1 is not None
        and selected.get("completed") and worker1.get("completed")
        else None
    )
    speedup_canonical = (
        float(selected["environment_steps_per_second"])
        / float(canonical["environment_steps_per_second"])
        if selected is not None and canonical is not None
        and selected.get("completed") and canonical.get("completed")
        else None
    )
    by_worker = {
        int(row["rollout_workers"]): row
        for row in server_rows if row.get("completed")
    }
    workers64_followup = (
        16 in by_worker and 32 in by_worker
        and float(by_worker[32]["environment_steps_per_second"])
        > 1.05 * float(by_worker[16]["environment_steps_per_second"])
    )
    steps = int(args.n_rollout_threads) * int(args.rollout_length)
    report = {
        "benchmark_contract": "one_exact_full_ppo_update_no_wandb_v1",
        "observation_mode": "full",
        "horizon_mode": "episode_balanced",
        "shared_horizons": [0, 2, 4],
        "seed": int(args.seed),
        "environment_steps_per_candidate": steps,
        "n_rollout_threads": int(args.n_rollout_threads),
        "rollout_length": int(args.rollout_length),
        "episode_length": int(args.episode_length),
        "ppo_epochs": int(args.ppo_epochs),
        "device": str(args.device),
        "reward_shaping_horizon_steps": int(args.reward_shaping_horizon_steps),
        "entropy_coefs": list(args.entropy_coefs),
        "entropy_coef_horizons": list(args.entropy_coef_horizons),
        "server_parallel": server_rows,
        "canonical_full": canonical,
        "recommended_rollout_workers": recommended,
        "recommendation_rule": (
            "smallest worker count within 5 percent of highest completed throughput"
        ),
        "highest_environment_steps_per_second": highest,
        "speedup_vs_worker1": speedup_worker1,
        "speedup_vs_canonical_full": speedup_canonical,
        "workers64_followup_threshold_met": workers64_followup,
        "long_training_started": False,
        "full_120m_training_started": False,
        "partial_1m_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    _validate(args)
    if args.candidate_backend is not None:
        if args.candidate_output is None:
            raise ValueError("candidate mode requires --candidate-output")
        if (
            args.candidate_backend == "server_parallel_full"
            and args.candidate_workers is None
        ):
            raise ValueError("server candidate requires --candidate-workers")
        row = _one(args.candidate_backend, args.candidate_workers, args)
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(row, indent=2, sort_keys=True))
        return 0 if row.get("completed") else 1
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
