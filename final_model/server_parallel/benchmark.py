"""Bounded server throughput benchmark; never launches multi-update training."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import torch

from final_model.partial_ver.canonical_training import PartialTrainingSession
from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "final_model/source_checkpoints/shared_full_seed20260821_step_002000000.pt"
PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
DEFAULT_OUTPUT = ROOT / "final_model/reports/server_parallel_rollout_benchmark.json"
RULE = {"task": 0.60, "stay": 0.20, "stale": 0.15, "random": 0.05}


def _active_trainers() -> list[str]:
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout
    blocked = []
    for line in output.splitlines():
        if "final_model/train.py" in line or "final_model.server_parallel.train_server" in line:
            blocked.append(line.strip())
    return blocked


def _process_cpu_seconds(pid: int) -> float:
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().split()
        ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return (int(fields[13]) + int(fields[14])) / ticks
    except (FileNotFoundError, IndexError, ValueError):
        return 0.0


def _process_rss_bytes(pid: int) -> int:
    try:
        pages = int(Path(f"/proc/{int(pid)}/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, ValueError):
        return 0


def _gpu_sample() -> tuple[float, float]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        utilization, memory = (float(value.strip()) for value in output.split(","))
        return utilization, memory
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return 0.0, 0.0


class ResourceMonitor:
    def __init__(self, pids: list[int]) -> None:
        self.pids = [int(value) for value in pids]
        self.stop_event = threading.Event()
        self.gpu_utilization = []
        self.gpu_memory_mib = []
        self.rss_bytes = {pid: [] for pid in self.pids}
        self.cpu_before = {pid: _process_cpu_seconds(pid) for pid in self.pids}
        self.started = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            utilization, memory = _gpu_sample()
            self.gpu_utilization.append(utilization)
            self.gpu_memory_mib.append(memory)
            for pid in self.pids:
                self.rss_bytes[pid].append(_process_rss_bytes(pid))
            self.stop_event.wait(0.25)

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread.start()
        return self

    def __exit__(self, _type, _value, _traceback):
        self.stop_event.set()
        self.thread.join(timeout=5.0)

    def report(self) -> dict[str, object]:
        wall = max(time.perf_counter() - self.started, 1e-12)
        cpu_after = {pid: _process_cpu_seconds(pid) for pid in self.pids}
        cpu_seconds = sum(
            max(0.0, cpu_after[pid] - self.cpu_before[pid]) for pid in self.pids
        )
        return {
            "monitored_pids": self.pids,
            "aggregate_cpu_seconds": cpu_seconds,
            "average_cpu_cores_used": cpu_seconds / wall,
            "peak_rss_mib_by_pid": {
                str(pid): max(values, default=0) / (1024 * 1024)
                for pid, values in self.rss_bytes.items()
            },
            "mean_gpu_utilization_percent": (
                sum(self.gpu_utilization) / max(len(self.gpu_utilization), 1)
            ),
            "peak_gpu_utilization_percent": max(self.gpu_utilization, default=0.0),
            "peak_gpu_memory_mib": max(self.gpu_memory_mib, default=0.0),
            "load_average_after": list(os.getloadavg()),
        }


def _options(*, rollout_length: int, ppo_epochs: int) -> dict[str, object]:
    return {
        "gaze_type": "rule_based",
        "train_action": True,
        "train_gaze": False,
        "decision_times": [600],
        "horizons": [0, 2, 4],
        "fixation_shapes": ["3x3"],
        "seed": 20260821,
        "rollout_length": int(rollout_length),
        "episode_length": 400,
        "n_rollout_threads": 100,
        "device": "cuda",
        "source_checkpoint": SOURCE,
        "resume_checkpoint": None,
        "fixed_partner_checkpoint": PARTNER,
        "configured_total_steps": int(rollout_length) * 100,
        "checkpoint_interval": int(rollout_length) * 100,
        "reward_shaping_horizon_steps": 20_000_000,
        "rule_gaze_probabilities": RULE,
        "partner": None,
        "ppo_epochs": int(ppo_epochs),
        "num_mini_batches": 1,
    }


def _one(backend: str, *, workers: int | None, rollout_length: int, ppo_epochs: int):
    options = _options(rollout_length=rollout_length, ppo_epochs=ppo_epochs)
    initialized = time.perf_counter()
    session = None
    try:
        if backend == "canonical_serial":
            session = PartialTrainingSession(**options)
            pids = [os.getpid()]
        else:
            session = ServerParallelActionTrainingSession(
                rollout_workers=int(workers), server_trace=False, **options
            )
            pids = [os.getpid(), *session.worker_pids]
        startup_seconds = time.perf_counter() - initialized
        with ResourceMonitor(pids) as monitor:
            update_started = time.perf_counter()
            try:
                result = session.run_update()
                error = None
            except Exception as caught:
                result = None
                error = caught
            total_seconds = time.perf_counter() - update_started
        resource = monitor.report()
        if error is not None:
            return {
                "backend": backend,
                "rollout_workers": workers,
                "rollout_length": rollout_length,
                "environment_steps": int(rollout_length) * 100,
                "startup_seconds": startup_seconds,
                "total_seconds_until_failure": total_seconds,
                "completed": False,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "long_training_started": False,
                **resource,
            }
        assert result is not None
        steps = int(result["environment_steps"])
        acquisitions = int(result["acquisitions"])
        return {
            "backend": backend,
            "rollout_workers": workers,
            "rollout_length": rollout_length,
            "environment_steps": steps,
            "startup_seconds": startup_seconds,
            "total_update_seconds": total_seconds,
            "environment_steps_per_second": steps / max(total_seconds, 1e-12),
            "acquisitions_per_second": acquisitions / max(total_seconds, 1e-12),
            "actor_inference_seconds": result.get("actor_inference_seconds"),
            "partner_inference_seconds": result.get("partner_inference_seconds"),
            "ipc_barrier_overhead_seconds": result.get("ipc_barrier_overhead_seconds"),
            "rollout_collection_seconds": result["rollout_collection_seconds"],
            "ppo_update_seconds": result["ppo_update_seconds"],
            "preupdate_audit_seconds": result["preupdate_audit_seconds"],
            "preupdate_max_abs_logprob_difference": result[
                "preupdate_max_abs_logprob_difference"
            ],
            "preupdate_approx_kl": result["preupdate_approx_kl"],
            "preupdate_clip_fraction": result["preupdate_clip_fraction"],
            "actor_hash_after": result["actor_hash_after"],
            "partner_guard_failures": result["partner_step_guard_failures"],
            "completed": True,
            "long_training_started": False,
            **resource,
        }
    except Exception as error:
        return {
            "backend": backend,
            "rollout_workers": workers,
            "rollout_length": rollout_length,
            "environment_steps": int(rollout_length) * 100,
            "startup_seconds": time.perf_counter() - initialized,
            "completed": False,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "long_training_started": False,
        }
    finally:
        if session is not None and hasattr(session, "close"):
            session.close()
        gc.collect()
        torch.cuda.empty_cache()


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16, 32])
    value.add_argument("--rollout-length", type=int, default=40)
    value.add_argument("--ppo-epochs", type=int, default=15)
    value.add_argument("--production-size", action="store_true")
    value.add_argument("--production-only", action="store_true")
    value.add_argument("--bounded-only", action="store_true")
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def run(args) -> dict[str, object]:
    active = _active_trainers()
    if active:
        raise RuntimeError(f"benchmark aborted because trainers are active: {active}")
    existing_production = []
    if args.bounded_only:
        if not args.output.is_file():
            raise RuntimeError("--bounded-only requires an existing benchmark JSON")
        existing_production = json.loads(args.output.read_text())["production_size"]
    if args.production_only:
        if not args.output.is_file():
            raise RuntimeError("--production-only requires an existing benchmark JSON")
        bounded = json.loads(args.output.read_text())["bounded"]
    else:
        bounded = [
            _one(
                "canonical_serial", workers=None,
                rollout_length=args.rollout_length, ppo_epochs=args.ppo_epochs,
            )
        ]
        for workers in args.workers:
            bounded.append(_one(
                "server_parallel", workers=workers,
                rollout_length=args.rollout_length, ppo_epochs=args.ppo_epochs,
            ))
    server_rows = sorted(
        (row for row in bounded[1:] if row["completed"]),
        key=lambda row: float(row["total_update_seconds"]),
    )
    production = existing_production
    if args.production_size or args.production_only:
        for row in server_rows[:2]:
            production.append(_one(
                "server_parallel", workers=int(row["rollout_workers"]),
                rollout_length=400, ppo_epochs=args.ppo_epochs,
            ))
    completed_production = [row for row in production if row["completed"]]
    ranking = sorted(
        completed_production or server_rows,
        key=lambda row: float(row["total_update_seconds"]),
    )
    best = ranking[0] if ranking else None
    serial = bounded[0]
    speedup = (
        float(serial["total_update_seconds"])
        / float(best["total_update_seconds"])
        if best is not None and serial["completed"] else None
    )
    recommended = None if best is None else int(best["rollout_workers"])
    report = {
        "benchmark_contract": "bounded_single_update_no_wandb_v1",
        "bounded": bounded,
        "production_size": production,
        "recommended_rollout_workers": recommended,
        "bounded_speedup_vs_canonical": speedup,
        "canonical_serial_completed": bool(serial["completed"]),
        "long_training_started": False,
        "recommended_command": None if recommended is None else (
            "python -m final_model.server_parallel.train_server "
            "--observation-mode partial --gaze-type rule_based "
            "--train-components action --shared-horizons 0 2 4 "
            "--decision-time 600 --fixation-shape 3x3 "
            f"--n-rollout-threads 100 --rollout-workers {recommended} "
            "--rollout-length 400 --device cuda --wandb-mode disabled "
            f"--source-checkpoint {SOURCE} --fixed-partner-checkpoint {PARTNER} "
            "--steps N"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
