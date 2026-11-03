"""Run sparse-only Gaze/Joint checkpoint selection and final evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


GAZE_TOTAL_STEPS = 20_000_000
GAZE_INTERVAL = 1_000_000
JOINT_TOTAL_STEPS = 1_000_000
JOINT_INTERVAL = 200_000
SELECTION_SEED_START = 300_000
HOLDOUT_SEED_START = 300_036
FINAL_SEED_START = 400_000
EPISODES = 36
EPISODE_LENGTH = 180


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step:09d}.pt"


def checkpoint_step(path: Path) -> int:
    return int(path.stem.removeprefix("step_"))


def latest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted(run_dir.glob("checkpoints/step_*.pt"), key=checkpoint_step)
    return paths[-1] if paths else None


def metrics_path(run_dir: Path) -> Path:
    return run_dir / "reports" / "metrics.jsonl"


def read_metrics(path: Path) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                by_step[int(row["environment_steps"])] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return [by_step[step] for step in sorted(by_step)]


def episode_count(row: dict[str, Any]) -> int:
    direct = sum(int(row.get(f"h{h}_episode_count", 0)) for h in (0, 2, 4))
    if direct:
        return direct
    horizons = row.get("horizon_metrics", {})
    return sum(int(horizons.get(f"h{h}", {}).get("episode_count", 0)) for h in (0, 2, 4))


def sparse_reward(row: dict[str, Any]) -> float:
    if "sparse_reward" in row:
        return float(row["sparse_reward"])
    horizons = row.get("horizon_metrics", {})
    return sum(float(horizons.get(f"h{h}", {}).get("sparse_reward", 0.0)) for h in (0, 2, 4))


def training_shortlist(
    run_dir: Path, *, interval: int, total_steps: int, count: int = 3
) -> list[dict[str, Any]]:
    rows = read_metrics(metrics_path(run_dir))
    candidates = []
    for step in range(interval, total_steps + 1, interval):
        checkpoint = checkpoint_path(run_dir, step)
        if not checkpoint.is_file():
            continue
        window = [
            row
            for row in rows
            if step - interval < int(row["environment_steps"]) <= step
        ]
        episodes = sum(episode_count(row) for row in window)
        reward = sum(sparse_reward(row) for row in window)
        if not window or episodes <= 0:
            raise RuntimeError(f"missing complete training window for {checkpoint}")
        candidates.append(
            {
                "step": step,
                "checkpoint": str(checkpoint.resolve()),
                "window_start_exclusive": step - interval,
                "window_end_inclusive": step,
                "rows": len(window),
                "episodes": episodes,
                "sparse_reward": reward,
                "sparse_reward_per_episode": reward / episodes,
            }
        )
    expected = total_steps // interval
    if len(candidates) != expected:
        raise RuntimeError(
            f"expected {expected} selectable checkpoints in {run_dir}, found {len(candidates)}"
        )
    return sorted(
        candidates,
        key=lambda item: (-item["sparse_reward_per_episode"], item["step"]),
    )[:count]


def model_name(prefix: str, step: int) -> str:
    return f"{prefix}_{step // 1_000_000}m" if step % 1_000_000 == 0 else f"{prefix}_{step // 1_000}k"


def evaluation_manifest(
    models: list[dict[str, Any]], *, prefix: str, seed_start: int, device: str
) -> dict[str, Any]:
    return {
        "models": [
            {
                "name": model_name(prefix, int(item["step"])),
                "checkpoint": item["checkpoint"],
            }
            for item in models
        ],
        "evaluation": {
            "decision_times": [200, 400, 600],
            "horizons": [0, 2, 4],
            "fixation_shape": "plus",
            "episodes": EPISODES,
            "episode_length": EPISODE_LENGTH,
            "seed_start": seed_start,
            "device": device,
            "gaze_mode": "stochastic",
            "action_mode": "argmax",
            "bootstrap_iterations": 20_000,
            "bootstrap_seed": 20260826,
            "aoi_precedence": ["preview", "self", "partner"],
        },
    }


def rollout_complete(output_dir: Path, names: Iterable[str]) -> bool:
    expected_rows = 3 * 3 * EPISODES
    for name in names:
        model_dir = output_dir / "models" / name
        if (model_dir / ".incomplete").exists():
            return False
        path = model_dir / "episodes.csv"
        if not path.is_file():
            return False
        with path.open(encoding="utf-8", newline="") as stream:
            if sum(1 for _ in csv.DictReader(stream)) != expected_rows:
                return False
    return True


def run_evaluation(
    python: Path,
    pipeline_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    prefix: str,
    phase: str,
    seed_start: int,
    device: str,
) -> Path:
    output_dir = pipeline_dir / phase
    manifest_path = pipeline_dir / f"{phase}_manifest.json"
    manifest = evaluation_manifest(
        candidates, prefix=prefix, seed_start=seed_start, device=device
    )
    names = [item["name"] for item in manifest["models"]]
    if rollout_complete(output_dir, names):
        log(f"reuse complete evaluation: {output_dir}")
        return output_dir
    write_json(manifest_path, manifest)
    command = [
        str(python),
        "-u",
        "-m",
        "final_model.eval.gaze_behavior_suite",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--rollout-only",
        "--overwrite",
        "--progress",
    ]
    log("run evaluation: " + " ".join(command))
    subprocess.run(command, check=True)
    if not rollout_complete(output_dir, names):
        raise RuntimeError(f"evaluation did not produce complete artifacts: {output_dir}")
    return output_dir


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def episode_rows(output_dir: Path, names: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    result = {}
    for name in names:
        rows = read_csv_rows(output_dir / "models" / name / "episodes.csv")
        for row in rows:
            sparse = float(row["sparse_team_reward"])
            deliveries = int(row["deliveries"])
            if not math.isclose(sparse, 20.0 * deliveries, abs_tol=1e-8):
                raise RuntimeError(
                    f"sparse reward contract failed for {name}: {sparse} != 20*{deliveries}"
                )
        result[name] = rows
    return result


def rank_models(
    rows_by_model: dict[str, list[dict[str, str]]], steps: dict[str, int]
) -> list[dict[str, Any]]:
    ranked = []
    for name, rows in rows_by_model.items():
        values = [float(row["sparse_team_reward"]) for row in rows]
        conditions: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in rows:
            conditions[(int(row["T"]), int(row["H"]))].append(
                float(row["sparse_team_reward"])
            )
        ranked.append(
            {
                "name": name,
                "step": steps[name],
                "episodes": len(values),
                "mean_sparse_reward": statistics.fmean(values),
                "minimum_condition_mean_sparse_reward": min(
                    statistics.fmean(condition) for condition in conditions.values()
                ),
                "std_sparse_reward": statistics.pstdev(values),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            -item["mean_sparse_reward"],
            -item["minimum_condition_mean_sparse_reward"],
            item["std_sparse_reward"],
            item["step"],
        ),
    )


def seed_means(rows: list[dict[str, str]]) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values[int(row["seed"])].append(float(row["sparse_team_reward"]))
    if any(len(scores) != 9 for scores in values.values()):
        raise RuntimeError("paired comparison requires nine T×H observations per seed")
    return {seed: statistics.fmean(scores) for seed, scores in values.items()}


def paired_ci(
    lead_rows: list[dict[str, str]], runner_rows: list[dict[str, str]]
) -> dict[str, float | int | bool]:
    lead = seed_means(lead_rows)
    runner = seed_means(runner_rows)
    seeds = sorted(set(lead) & set(runner))
    if len(seeds) < 2 or set(lead) != set(runner):
        raise RuntimeError("paired comparison requires identical seed sets")
    differences = [lead[seed] - runner[seed] for seed in seeds]
    mean = statistics.fmean(differences)
    sem = statistics.stdev(differences) / math.sqrt(len(differences))
    try:
        from scipy.stats import t as student_t

        critical = float(student_t.ppf(0.975, len(differences) - 1))
    except ImportError:
        critical = 1.96
    low, high = mean - critical * sem, mean + critical * sem
    return {
        "paired_seeds": len(seeds),
        "mean_difference": mean,
        "ci_low": low,
        "ci_high": high,
        "includes_zero": low <= 0.0 <= high,
    }


def select_by_benchmark(
    python: Path,
    pipeline_dir: Path,
    shortlist: list[dict[str, Any]],
    *,
    prefix: str,
    phase_prefix: str,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    steps = {model_name(prefix, item["step"]): int(item["step"]) for item in shortlist}
    primary_dir = run_evaluation(
        python,
        pipeline_dir,
        shortlist,
        prefix=prefix,
        phase=f"{phase_prefix}_primary_seeds_{SELECTION_SEED_START}_{SELECTION_SEED_START + EPISODES - 1}",
        seed_start=SELECTION_SEED_START,
        device=device,
    )
    primary = episode_rows(primary_dir, steps)
    primary_rank = rank_models(primary, steps)
    lead, runner = primary_rank[0]["name"], primary_rank[1]["name"]
    comparison = paired_ci(primary[lead], primary[runner])
    holdout_dir = None
    final_rows = primary
    if comparison["includes_zero"]:
        finalists = [item for item in shortlist if model_name(prefix, item["step"]) in {lead, runner}]
        holdout_dir = run_evaluation(
            python,
            pipeline_dir,
            finalists,
            prefix=prefix,
            phase=f"{phase_prefix}_holdout_seeds_{HOLDOUT_SEED_START}_{HOLDOUT_SEED_START + EPISODES - 1}",
            seed_start=HOLDOUT_SEED_START,
            device=device,
        )
        holdout = episode_rows(holdout_dir, (lead, runner))
        final_rows = {name: primary[name] + holdout.get(name, []) for name in primary}
    final_rank = rank_models(final_rows, steps)
    selected_name = final_rank[0]["name"]
    selected = next(
        item for item in shortlist if model_name(prefix, item["step"]) == selected_name
    )
    report = {
        "selection_metric": "per-episode sparse_team_reward only",
        "ranking_tiebreakers": [
            "higher overall mean",
            "higher minimum T×H mean",
            "lower overall population standard deviation",
            "earlier checkpoint",
        ],
        "primary_seed_range": [SELECTION_SEED_START, SELECTION_SEED_START + EPISODES - 1],
        "holdout_seed_range": (
            [HOLDOUT_SEED_START, HOLDOUT_SEED_START + EPISODES - 1]
            if holdout_dir is not None
            else None
        ),
        "primary_ranking": primary_rank,
        "primary_lead_pair_ci": comparison,
        "final_ranking": final_rank,
        "selected": selected,
        "primary_output": str(primary_dir),
        "holdout_output": None if holdout_dir is None else str(holdout_dir),
    }
    write_json(pipeline_dir / f"{phase_prefix}_selection.json", report)
    return selected, report


def append_option(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def append_many(command: list[str], flag: str, values: Iterable[Any] | None) -> None:
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


def training_command(
    python: Path,
    config: dict[str, Any],
    run_dir: Path,
    *,
    total_steps: int,
    checkpoint_interval: int,
    reward_shaping_horizon_steps: int,
    source_checkpoint: Path | None = None,
    resume_checkpoint: Path | None = None,
    joint: bool,
) -> list[str]:
    command = [str(python), "-u", "-m", "final_model.server_parallel.train_server"]
    command.extend(("--gaze-type", "learned"))
    command.append("--train-components")
    command.extend(("action", "gaze") if joint else ("gaze",))
    append_option(command, "--source-checkpoint", source_checkpoint)
    append_option(command, "--resume-checkpoint", resume_checkpoint)
    append_option(command, "--run-dir", run_dir)
    append_option(command, "--observation-mode", config["observation_mode"])
    append_many(command, "--shared-horizons", config["shared_horizons"])
    append_many(command, "--shared-decision-times", config["shared_decision_times"])
    append_option(command, "--fixation-shape", config["fixation_shape"])
    append_option(command, "--memory-decay-mode", config["memory_decay_mode"])
    append_option(command, "--memory-lifetime-ms", config["memory_lifetime_ms"])
    if config["memory_decay_mode"] == "stochastic_exponential":
        append_option(command, "--memory-tau-ms", config.get("memory_tau_ms"))
        append_option(command, "--memory-max-lifetime-ms", config.get("memory_max_lifetime_ms"))
    append_option(command, "--fixed-partner-checkpoint", config["fixed_partner_checkpoint"])
    command.extend(("--steps", str(total_steps)))
    for key, flag in (
        ("rollout_length", "--rollout-length"),
        ("episode_length", "--episode-length"),
        ("n_rollout_threads", "--n-rollout-threads"),
        ("rollout_workers", "--rollout-workers"),
    ):
        append_option(command, flag, config[key])
    command.extend(("--checkpoint-interval", str(checkpoint_interval)))
    command.extend(("--reward-shaping-horizon-steps", str(reward_shaping_horizon_steps)))
    for key, flag in (
        ("actor_lr", "--actor-lr"),
        ("critic_lr", "--critic-lr"),
        ("ppo_epochs", "--ppo-epochs"),
        ("num_mini_batches", "--num-mini-batches"),
        ("clip_param", "--clip-param"),
        ("gamma", "--gamma"),
        ("gae_lambda", "--gae-lambda"),
        ("max_grad_norm", "--max-grad-norm"),
        ("seed", "--seed"),
        ("device", "--device"),
        ("host_mode", "--host-mode"),
        ("inference_batching", "--inference-batching"),
        ("constraint_profile", "--constraint-profile"),
        ("peripheral_mode", "--peripheral-mode"),
        ("emma_mode", "--emma-mode"),
        ("preview_opacity_mode", "--preview-opacity-mode"),
        ("memory_condition", "--memory-condition"),
        ("perception_condition", "--perception-condition"),
        ("fixed_partner_executor_workers", "--fixed-partner-executor-workers"),
    ):
        append_option(command, flag, config.get(key))
    if config.get("peripheral_mode") != "none":
        append_option(command, "--peripheral-sigma-deg", config.get("peripheral_sigma_deg"))
        append_option(command, "--peripheral-extent", config.get("peripheral_extent"))
    append_many(command, "--entropy-coefs", config.get("entropy_coefs"))
    append_many(command, "--entropy-coef-horizons", config.get("entropy_coef_horizons"))
    command.extend(("--condition-name", run_dir.name, "--wandb-mode", "disabled", "--progress"))
    return command


def cuda_available(python: Path) -> bool:
    completed = subprocess.run(
        [str(python), "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def wait_for_cuda(python: Path, poll_seconds: int) -> None:
    while not cuda_available(python):
        log(f"CUDA unavailable; retry in {poll_seconds}s")
        time.sleep(poll_seconds)


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def training_complete(run_dir: Path, final_step: int) -> bool:
    final = checkpoint_path(run_dir, final_step)
    rows = read_metrics(metrics_path(run_dir)) if metrics_path(run_dir).is_file() else []
    latest_step = int(rows[-1]["environment_steps"]) if rows else 0
    return final.is_file() and latest_step >= final_step


def supervise_gaze_training(
    python: Path,
    gaze_run_dir: Path,
    config: dict[str, Any],
    poll_seconds: int,
    active_pid: int | None,
) -> None:
    last_reported = None
    while not training_complete(gaze_run_dir, GAZE_TOTAL_STEPS):
        rows = read_metrics(metrics_path(gaze_run_dir)) if metrics_path(gaze_run_dir).is_file() else []
        step = int(rows[-1]["environment_steps"]) if rows else 0
        if process_alive(active_pid):
            if step != last_reported:
                log(
                    f"existing Gaze PID {active_pid} alive; "
                    f"step={step}/{GAZE_TOTAL_STEPS}"
                )
                last_reported = step
            time.sleep(poll_seconds)
            continue
        active_pid = None
        resume = latest_checkpoint(gaze_run_dir)
        if resume is None:
            raise RuntimeError(
                f"cannot resume Gaze without a checkpoint: {gaze_run_dir}"
            )
        wait_for_cuda(python, poll_seconds)
        command = training_command(
            python,
            config,
            gaze_run_dir,
            total_steps=GAZE_TOTAL_STEPS,
            checkpoint_interval=GAZE_INTERVAL,
            reward_shaping_horizon_steps=GAZE_TOTAL_STEPS,
            resume_checkpoint=resume,
            joint=False,
        )
        log(f"resume Gaze from {resume}: " + " ".join(command))
        completed = subprocess.run(command)
        if completed.returncode != 0:
            log(
                f"Gaze resume exited with code {completed.returncode}; "
                f"retry latest checkpoint in {poll_seconds}s"
            )
            time.sleep(poll_seconds)
    log(f"Gaze 20M checkpoint and metrics verified: {gaze_run_dir}")


def wait_for_training(run_dir: Path, final_step: int, poll_seconds: int) -> None:
    final = checkpoint_path(run_dir, final_step)
    last_reported = None
    while True:
        rows = read_metrics(metrics_path(run_dir)) if metrics_path(run_dir).is_file() else []
        step = int(rows[-1]["environment_steps"]) if rows else 0
        if final.is_file() and step >= final_step:
            log(f"training complete: {run_dir} step={step}")
            return
        if step != last_reported:
            log(f"waiting for training: {run_dir} step={step}/{final_step}")
            last_reported = step
        time.sleep(poll_seconds)


def run_joint_training(
    python: Path,
    gaze_config: dict[str, Any],
    joint_run_dir: Path,
    selected_gaze: Path,
) -> None:
    final = checkpoint_path(joint_run_dir, JOINT_TOTAL_STEPS)
    if final.is_file():
        log(f"reuse complete Joint training: {joint_run_dir}")
        return
    resume = latest_checkpoint(joint_run_dir)
    command = training_command(
        python,
        gaze_config,
        joint_run_dir,
        total_steps=JOINT_TOTAL_STEPS,
        checkpoint_interval=JOINT_INTERVAL,
        reward_shaping_horizon_steps=0,
        source_checkpoint=None if resume else selected_gaze,
        resume_checkpoint=resume,
        joint=True,
    )
    log("start strict sparse Joint (shaping=0): " + " ".join(command))
    subprocess.run(command, check=True)


def promote(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    log(f"promoted {source} -> {destination}")


def final_report(output_dir: Path, name: str, destination: Path, pipeline_dir: Path) -> dict[str, Any]:
    episodes = read_csv_rows(output_dir / "models" / name / "episodes.csv")
    episode_rows(output_dir, (name,))
    acquisitions_path = output_dir / "models" / name / "acquisitions.csv.gz"
    aoi: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    with gzip.open(acquisitions_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            label = row.get("exclusive_center_aoi") or "other"
            aoi[(int(row["T"]), int(row["H"]))][label] += 1

    def summarize(rows: list[dict[str, str]], counts: Counter[str]) -> dict[str, Any]:
        total_acq = sum(counts.values())
        result = {
            "episodes": len(rows),
            "task_score": statistics.fmean(float(row["sparse_team_reward"]) for row in rows),
            "collision_count": sum(int(row["collision_count"]) for row in rows),
            "task_decisions": sum(int(row["task_decisions"]) for row in rows),
            "aoi_center_count": dict(counts),
        }
        result["collision_rate"] = result["collision_count"] / result["task_decisions"]
        result["aoi_center_rate"] = {
            label: counts.get(label, 0) / max(total_acq, 1)
            for label in ("preview", "self", "partner", "other")
        }
        return result

    by_cell = []
    for t in (200, 400, 600):
        for h in (0, 2, 4):
            rows = [row for row in episodes if int(row["T"]) == t and int(row["H"]) == h]
            by_cell.append({"T": t, "H": h, **summarize(rows, aoi[(t, h)])})
    by_time = []
    for t in (200, 400, 600):
        rows = [row for row in episodes if int(row["T"]) == t]
        counts = sum((aoi[(t, h)] for h in (0, 2, 4)), Counter())
        by_time.append({"T": t, **summarize(rows, counts)})
    all_counts = sum(aoi.values(), Counter())
    report = {
        "model_name": name,
        "checkpoint": str(destination.resolve()),
        "seed_range": [FINAL_SEED_START, FINAL_SEED_START + EPISODES - 1],
        "episodes_per_TxH": EPISODES,
        "episode_length_learner_decisions": EPISODE_LENGTH,
        "selection_metric": "sparse_team_reward only",
        "sparse_contract": "sparse_team_reward = 20 * deliveries",
        "aoi_definition": "exclusive fixation center; Preview > Self > Partner > Other",
        "overall": summarize(episodes, all_counts),
        "by_time": by_time,
        "by_time_horizon": by_cell,
    }
    write_json(pipeline_dir / "final_benchmark.json", report)
    lines = [
        f"# {name} 최종 sparse benchmark",
        "",
        f"- checkpoint: `{destination.resolve()}`",
        f"- seeds: `{FINAL_SEED_START}–{FINAL_SEED_START + EPISODES - 1}`",
        f"- 36 seeds/T×H, {EPISODE_LENGTH} learner decisions, stochastic Gaze / argmax Action",
        "- AOI: exclusive fixation center, Preview > Self > Partner > Other",
        "",
        "## T별",
        "",
        "| T (ms) | task score | collision | Preview AOI | Self AOI | Partner AOI | Other AOI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_time:
        rates = row["aoi_center_rate"]
        lines.append(
            f"| {row['T']} | {row['task_score']:.2f} | {100*row['collision_rate']:.2f}% | "
            f"{100*rates['preview']:.2f}% | {100*rates['self']:.2f}% | "
            f"{100*rates['partner']:.2f}% | {100*rates['other']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## T×H별",
            "",
            "| T (ms) | H | task score | collision | Preview AOI | Self AOI | Partner AOI | Other AOI |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_cell:
        rates = row["aoi_center_rate"]
        lines.append(
            f"| {row['T']} | {row['H']} | {row['task_score']:.2f} | {100*row['collision_rate']:.2f}% | "
            f"{100*rates['preview']:.2f}% | {100*rates['self']:.2f}% | "
            f"{100*rates['partner']:.2f}% | {100*rates['other']:.2f}% |"
        )
    (pipeline_dir / "final_benchmark_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--gaze-run-dir", type=Path, required=True)
    value.add_argument("--branch-name", required=True)
    value.add_argument("--final-checkpoint", type=Path, required=True)
    value.add_argument("--python", type=Path, default=Path(sys.executable))
    value.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument(
        "--active-gaze-pid",
        type=int,
        help="initial externally started Gaze PID; resume only after it exits",
    )
    value.add_argument("--dry-run", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    gaze_run_dir = args.gaze_run_dir.expanduser().resolve()
    python = args.python.expanduser().resolve()
    destination = args.final_checkpoint.expanduser().resolve()
    config = json.loads((gaze_run_dir / "config.json").read_text(encoding="utf-8"))
    pipeline_dir = gaze_run_dir / "post_training_sparse_pipeline"
    joint_run_dir = pipeline_dir / f"{args.branch_name}_joint1m_strict_sparse"
    contract = {
        "branch_name": args.branch_name,
        "gaze_run_dir": str(gaze_run_dir),
        "joint_run_dir": str(joint_run_dir),
        "final_checkpoint": str(destination),
        "gaze_training_window": GAZE_INTERVAL,
        "joint_training_window": JOINT_INTERVAL,
        "selection_seed_range": [SELECTION_SEED_START, SELECTION_SEED_START + EPISODES - 1],
        "holdout_seed_range": [HOLDOUT_SEED_START, HOLDOUT_SEED_START + EPISODES - 1],
        "final_seed_range": [FINAL_SEED_START, FINAL_SEED_START + EPISODES - 1],
        "joint_reward_shaping_horizon_steps": 0,
    }
    write_json(pipeline_dir / "contract.json", contract)
    if args.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0

    supervise_gaze_training(
        python,
        gaze_run_dir,
        config,
        args.poll_seconds,
        args.active_gaze_pid,
    )
    gaze_shortlist = training_shortlist(
        gaze_run_dir, interval=GAZE_INTERVAL, total_steps=GAZE_TOTAL_STEPS
    )
    write_json(pipeline_dir / "gaze_training_shortlist.json", gaze_shortlist)
    selected_gaze, _ = select_by_benchmark(
        python,
        pipeline_dir,
        gaze_shortlist,
        prefix="gaze",
        phase_prefix="gaze_selection",
        device=args.device,
    )

    run_joint_training(
        python, config, joint_run_dir, Path(selected_gaze["checkpoint"])
    )
    wait_for_training(joint_run_dir, JOINT_TOTAL_STEPS, args.poll_seconds)
    joint_shortlist = training_shortlist(
        joint_run_dir, interval=JOINT_INTERVAL, total_steps=JOINT_TOTAL_STEPS
    )
    write_json(pipeline_dir / "joint_training_shortlist.json", joint_shortlist)
    selected_joint, _ = select_by_benchmark(
        python,
        pipeline_dir,
        joint_shortlist,
        prefix="joint",
        phase_prefix="joint_selection",
        device=args.device,
    )
    promote(Path(selected_joint["checkpoint"]), destination)

    final_item = {"step": selected_joint["step"], "checkpoint": str(destination)}
    final_output = run_evaluation(
        python,
        pipeline_dir,
        [final_item],
        prefix=args.branch_name,
        phase=f"final_benchmark_seeds_{FINAL_SEED_START}_{FINAL_SEED_START + EPISODES - 1}",
        seed_start=FINAL_SEED_START,
        device=args.device,
    )
    name = model_name(args.branch_name, int(selected_joint["step"]))
    final_report(final_output, name, destination, pipeline_dir)
    write_json(pipeline_dir / "COMPLETE.json", {**contract, "selected_gaze": selected_gaze, "selected_joint": selected_joint})
    log(f"pipeline complete: {pipeline_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
