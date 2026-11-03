"""One-update benchmark for the experimental Full CPU-partner pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from final_model.server_parallel.benchmark import ResourceMonitor, _active_trainers
from final_model.server_parallel.full_hybrid_action_training import (
    FULL_HYBRID_VERSION,
    ServerParallelFullHybridActionTrainingSession,
)


ROOT = Path(__file__).resolve().parents[2]
HSP_INITIAL = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"
FIXED_PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
BASELINE_SUMMARY = (
    Path(__file__).resolve().parent
    / "runs/full_shared_intra2_inter1_exec4_w8_40k/logs/summary.json"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "reports/full_hybrid_inference_benchmark.json"
)


def _configure_torch(intra: int, inter: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(int(intra))
    os.environ["MKL_NUM_THREADS"] = str(int(intra))
    torch.set_num_threads(int(intra))
    torch.set_num_interop_threads(int(inter))


def _options(args) -> dict[str, object]:
    steps = int(args.n_rollout_threads) * int(args.rollout_length)
    return {
        "observation_mode": "full",
        "horizon": None,
        "shared_horizons": [0, 2, 4],
        "decision_time": None,
        "fixation_shape": None,
        "seed": int(args.seed),
        "rollout_length": int(args.rollout_length),
        "episode_length": 400,
        "n_rollout_threads": int(args.n_rollout_threads),
        "device": "cuda",
        "source_checkpoint": None,
        "resume_checkpoint": None,
        "hsp_initial_checkpoint": HSP_INITIAL,
        "fixed_partner_checkpoint": FIXED_PARTNER,
        "configured_total_steps": steps,
        "checkpoint_interval": steps,
        "reward_shaping_horizon_steps": 120_000_000,
        "partner": None,
        "inference_batching": "predictor",
        "inference_shadow_audit": False,
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coefs": [0.05, 0.03, 0.01],
        "entropy_coef_horizons": [0, 60_000_000, 120_000_000],
        "ppo_epochs": 15,
        "num_mini_batches": 1,
        "max_grad_norm": 10.0,
    }


def _baseline() -> dict[str, object]:
    payload = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    latest = payload["latest"]
    return {
        "source": str(BASELINE_SUMMARY),
        "execution_backend": latest["execution_backend"],
        "fixed_partner_device": "cuda",
        "wall_seconds": float(payload["wall_clock_seconds"]),
        "steps_per_second": float(latest["steps_per_second"]),
        "rollout_collection_seconds": float(
            latest["rollout_collection_seconds"]
        ),
        "partner_inference_seconds": float(latest["partner_inference_seconds"]),
        "predictor_inference_seconds": float(
            latest["predictor_inference_seconds"]
        ),
        "learner_inference_seconds": float(latest["learner_inference_seconds"]),
        "env_step_seconds": float(latest["env_step_seconds"]),
        "ipc_barrier_overhead_seconds": float(
            latest["ipc_barrier_overhead_seconds"]
        ),
        "main_process_wait_seconds": float(latest["main_process_wait_seconds"]),
        "worker_wait_seconds": float(latest["worker_wait_seconds"]),
        "ppo_update_seconds": float(latest["ppo_update_seconds"]),
        "preupdate_audit_seconds": float(latest["preupdate_audit_seconds"]),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--seed", type=int, default=20260821)
    value.add_argument("--n-rollout-threads", type=int, default=100)
    value.add_argument("--rollout-length", type=int, default=400)
    value.add_argument("--rollout-workers", type=int, default=8)
    value.add_argument("--torch-intra", type=int, default=2)
    value.add_argument("--torch-inter", type=int, default=1)
    value.add_argument("--partner-executor", type=int, default=4)
    value.add_argument("--runtime-executor", type=int, default=4)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def run(args) -> dict[str, object]:
    if _active_trainers():
        raise RuntimeError("hybrid benchmark aborted because a trainer is active")
    if int(args.rollout_workers) > int(args.n_rollout_threads):
        raise ValueError("rollout workers may not exceed environments")
    if int(args.n_rollout_threads) * int(args.rollout_length) != 40_000:
        raise ValueError("hybrid benchmark must be exactly 40,000 steps")
    _configure_torch(args.torch_intra, args.torch_inter)
    session = ServerParallelFullHybridActionTrainingSession(
        rollout_workers=int(args.rollout_workers),
        partner_executor_workers=int(args.partner_executor),
        runtime_executor_workers=int(args.runtime_executor),
        server_trace=False,
        **_options(args),
    )
    try:
        pids = [os.getpid(), *session.worker_pids]
        torch.cuda.synchronize()
        with ResourceMonitor(pids) as monitor:
            started = time.perf_counter()
            result = session.run_update()
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
        resource = monitor.report()
        hybrid = {
            "execution_backend": FULL_HYBRID_VERSION,
            "completed": True,
            "environment_steps": int(result["environment_steps"]),
            "wall_seconds": wall,
            "wall_steps_per_second": 40_000 / max(wall, 1e-12),
            "steps_per_second": float(result["steps_per_second"]),
            "rollout_collection_seconds": float(
                result["rollout_collection_seconds"]
            ),
            "cpu_partner_inference_seconds": float(
                result["cpu_partner_inference_seconds"]
            ),
            "cpu_partner_executor_task_seconds": float(
                result["cpu_partner_executor_task_seconds"]
            ),
            "partner_commit_wait_seconds": float(
                result["partner_commit_wait_seconds"]
            ),
            "partner_pipeline_elapsed_seconds": float(
                result["partner_pipeline_elapsed_seconds"]
            ),
            "predictor_inference_seconds": float(
                result["predictor_inference_seconds"]
            ),
            "learner_inference_seconds": float(
                result["learner_inference_seconds"]
            ),
            "env_step_seconds": float(result["env_step_seconds"]),
            "ipc_barrier_overhead_seconds": float(
                result["ipc_barrier_overhead_seconds"]
            ),
            "main_process_wait_seconds": float(
                result["main_process_wait_seconds"]
            ),
            "worker_wait_seconds": float(result["worker_wait_seconds"]),
            "ppo_update_seconds": float(result["ppo_update_seconds"]),
            "preupdate_audit_seconds": float(result["preupdate_audit_seconds"]),
            "partner_forward_calls": int(result["partner_forward_calls"]),
            "partner_batch_size": int(result["partner_batch_size"]),
            "predictor_forward_calls": int(result["predictor_forward_calls"]),
            "partner_step_guard_failures": int(
                result["partner_step_guard_failures"]
            ),
            "worker_partner_guard_failures": int(
                result["worker_partner_guard_failures"]
            ),
            "preupdate_max_abs_logprob_difference": float(
                result["preupdate_max_abs_logprob_difference"]
            ),
            "preupdate_approx_kl": float(result["preupdate_approx_kl"]),
            **resource,
        }
        baseline = _baseline()
        report = {
            "contract": "experimental_full_hybrid_40k_v1",
            "settings": {
                "shared_horizons": [0, 2, 4],
                "environment_steps": 40_000,
                "n_rollout_threads": int(args.n_rollout_threads),
                "rollout_length": int(args.rollout_length),
                "rollout_workers": int(args.rollout_workers),
                "torch_intra": int(args.torch_intra),
                "torch_inter": int(args.torch_inter),
                "partner_executor": int(args.partner_executor),
                "runtime_executor": int(args.runtime_executor),
                "partner_batch_size": 1,
            },
            "current_cuda_baseline": baseline,
            "hybrid": hybrid,
            "hybrid_speed_ratio_vs_cuda_baseline": (
                hybrid["steps_per_second"] / baseline["steps_per_second"]
            ),
            "long_training_started": False,
            "full_120m_training_started": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        session.close()


def main(argv=None) -> int:
    print(json.dumps(run(parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
