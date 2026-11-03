#!/usr/bin/env python3
"""Print a compact, read-only snapshot of a training run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _latest_record(path: Path) -> dict[str, object]:
    latest = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                latest = line
    if not latest:
        raise RuntimeError(f"metrics file is empty: {path}")
    return json.loads(latest)


def _number(record: dict[str, object], key: str, default: float = 0.0) -> float:
    value = record.get(key)
    return default if value is None else float(value)


def _entropy_coefficient(
    steps: int, coefficients: list[float], horizons: list[int]
) -> float:
    for index in range(len(horizons) - 1):
        start, end = horizons[index], horizons[index + 1]
        if start <= steps < end:
            fraction = (steps - start) / (end - start)
            return (
                (1.0 - fraction) * coefficients[index]
                + fraction * coefficients[index + 1]
            )
    return coefficients[-1]


def _process_status(run_dir: Path) -> str:
    pid_path = run_dir / "production.pid"
    if not pid_path.is_file():
        return "PID file 없음"
    pid = int(pid_path.read_text().strip())
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return f"PID {pid} 미확인"
    return f"PID {pid} ALIVE"


def _condition_line(record: dict[str, object], prefix: str) -> str:
    return (
        f"{prefix:<4} "
        f"score/ep={_number(record, prefix + '/sparse_reward_per_episode'):7.2f}  "
        f"collision={100 * _number(record, prefix + '/collision_rate'):5.2f}%  "
        f"acq/decision={_number(record, prefix + '/acquisitions_per_decision'):5.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "reports/metrics.jsonl"
    config = json.loads(config_path.read_text())
    record = _latest_record(metrics_path)

    steps = int(record["environment_steps"])
    total = int(config["steps"])
    speed = _number(record, "steps_per_second")
    remaining = max(total - steps, 0)
    eta = remaining / speed if speed > 0 else 0.0
    age = max(0.0, time.time() - metrics_path.stat().st_mtime)
    coefficients = [float(value) for value in config["entropy_coefs"]]
    horizons = [int(value) for value in config["entropy_coef_horizons"]]
    entropy_coef = _entropy_coefficient(steps, coefficients, horizons)
    schedule = " -> ".join(
        f"{coefficient:g}@{horizon / 1_000_000:g}M"
        for coefficient, horizon in zip(coefficients, horizons)
    )
    action_frozen = (
        record.get("actor_hash_before") == record.get("actor_hash_after")
        and record.get("critic_hash_before") == record.get("critic_hash_after")
        and bool(record.get("action_optimizer_isolation_verified"))
    )

    source = Path(str(config.get("source_checkpoint", "unknown")))
    source_hash = str(config.get("source_checkpoint_sha256", "unknown"))
    components = ",".join(config.get("train_components", ()))

    print(f"RUN   {run_dir}")
    print(
        f"MODEL condition={config.get('condition_name')} | "
        f"gaze={config.get('gaze_type')} | train={components} | "
        f"seed={config.get('seed')}"
    )
    print(
        f"PERCP fixation={config.get('fixation_shape')} | "
        f"peripheral={config.get('peripheral_mode')}/"
        f"{config.get('peripheral_extent')} "
        f"sigma={config.get('peripheral_sigma_deg')}deg"
    )
    print(
        f"MEM   mode={config.get('memory_decay_mode')} | "
        f"tau={config.get('memory_tau_ms')}ms | "
        f"Tmax={config.get('memory_max_lifetime_ms')}ms | "
        f"EMMA={config.get('emma_mode')}"
    )
    print(f"SRC   {source.name} | sha256={source_hash[:12]}...")
    print(
        f"STATE {_process_status(run_dir)} | metrics_age={age:.0f}s | "
        f"workers={config['rollout_workers']}"
    )
    print(
        f"STEP  {steps:,}/{total:,} ({100 * steps / total:5.2f}%) | "
        f"update={record['update_index']} | {speed:6.1f} steps/s | "
        f"elapsed={_duration(_number(record, 'wall_time_seconds'))} | ETA={_duration(eta)}"
    )
    print(f"ENT   coef={entropy_coef:.5f} | policy_entropy={_number(record, 'entropy'):.4f}")
    print(f"      schedule: {schedule}")
    print(
        f"PPO   actor_loss={_number(record, 'actor_loss'):+.5f} | "
        f"critic_loss={_number(record, 'critic_loss'):.3f} | "
        f"KL={_number(record, 'approx_kl'):.5f} | "
        f"clip={100 * _number(record, 'clip_fraction'):5.2f}% | "
        f"EV={_number(record, 'explained_variance'):+.4f}"
    )
    print(
        f"TASK  sparse={_number(record, 'sparse_reward'):.0f} | "
        f"deliveries/ep={_number(record, 'deliveries_per_episode'):.2f} | "
        f"collision={100 * _number(record, 'collision_rate'):.2f}% "
        f"({int(_number(record, 'collision_count'))})"
    )
    print(
        f"GAZE  acq/decision={_number(record, 'acquisitions_per_decision'):.3f} | "
        f"new_target={100 * _number(record, 'new_target_acquisition_rate'):.2f}% | "
        f"saccade={_number(record, 'mean_saccade_distance'):.2f}deg | "
        f"valid_memory={100 * _number(record, 'valid_memory_fraction'):.2f}%"
    )
    print(
        f"AUDIT action_frozen={'PASS' if action_frozen else 'FAIL'} | "
        f"logprob_diff={_number(record, 'gaze_preupdate_max_abs_logprob_difference'):.2e} | "
        f"reward_error={_number(record, 'gaze_reward_conservation_error'):.2e} | "
        f"freshness=[{_number(record, 'freshness_normalized_min'):.0f},"
        f"{_number(record, 'freshness_normalized_max'):.0f}]"
    )
    print("\nBY DECISION TIME")
    for prefix in ("t200", "t400", "t600"):
        print(_condition_line(record, prefix))
    print("BY HORIZON")
    for prefix in ("h0", "h2", "h4"):
        print(_condition_line(record, prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
