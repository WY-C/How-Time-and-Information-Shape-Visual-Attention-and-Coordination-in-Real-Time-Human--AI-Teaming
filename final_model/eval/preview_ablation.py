"""Paired-seed rollout ablations that zero only legitimate preview channels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from final_model.eval.checkpoint_loader import load_action_checkpoint_for_evaluation
from final_model.eval.common import (
    ABLATION_SEED_START,
    create_output_directory,
    default_output_path,
    source_hashes,
    valid_preview_modes,
    write_csv,
    write_json,
)
from final_model.eval.rollout import EvaluationConfig, aggregate_behavior, evaluate


def run_ablation(*, checkpoint: Path, evaluation_horizon: int, preview_modes: list[str],
                 episodes: int, seed_start: int, episode_length: int, device: str,
                 output_dir: Path) -> dict:
    allowed = valid_preview_modes(evaluation_horizon)
    invalid = [mode for mode in preview_modes if mode not in allowed]
    if invalid:
        raise ValueError(f"invalid preview modes for H{evaluation_horizon}: {invalid}")
    root = create_output_directory(output_dir, prefix="preview_ablation")
    loaded = load_action_checkpoint_for_evaluation(checkpoint, device=device)
    loaded.validate_evaluation_horizon(evaluation_horizon, preview_ablation=True)
    rows = []
    configs = []
    run_manifests = []
    for mode in preview_modes:
        config = EvaluationConfig(
            checkpoint=checkpoint, evaluation_horizon=evaluation_horizon,
            episodes=episodes, seed_start=seed_start, episode_length=episode_length,
            device=device, preview_mode=mode, behavior_events=True,
            output_dir=root / mode, allow_fixed_horizon_ablation=True,
        )
        result = evaluate(config, loaded=loaded)
        behavior = aggregate_behavior(result.behavior_rows)
        rows.append({**result.summary, **behavior})
        run_manifests.append(result.manifest)
        configs.append({**asdict(config), "checkpoint": str(checkpoint), "output_dir": str(root / mode)})
    write_csv(root / "ablation_summary.csv", rows)
    write_json(root / "ablation_summary.json", rows)
    first = run_manifests[0]
    write_json(root / "manifest.json", {
        "analysis": "paired_seed_preview_rollout_ablation",
        "timestamp": first["timestamp"],
        "command": first["command"],
        "config": configs,
        "checkpoint_absolute_path": str(loaded.path),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_completed_step": loaded.completed_steps,
        "checkpoint_horizon_mode": loaded.horizon_mode,
        "checkpoint_horizons": list(loaded.horizons),
        "evaluation_horizon": evaluation_horizon,
        "preview_mode": preview_modes,
        "preview_modes": preview_modes,
        "episode_count": episodes,
        "episode_seeds": list(range(seed_start, seed_start + episodes)),
        "learner_action_mode": "deterministic",
        "partner_checkpoint_path": first["partner_checkpoint_path"],
        "partner_checkpoint_sha256": first["partner_checkpoint_sha256"],
        "actor_hash_before": first["actor_hash_before"],
        "actor_hash_after": run_manifests[-1]["actor_hash_after"],
        "critic_hash_before": first["critic_hash_before"],
        "critic_hash_after": run_manifests[-1]["critic_hash_after"],
        "partner_hash_before": first["partner_hash_before"],
        "partner_hash_after": run_manifests[-1]["partner_hash_after"],
        "device": first["device"],
        "torch_version": first["torch_version"],
        "cuda_available": first["cuda_available"],
        "gpu_name": first["gpu_name"],
        "configs": configs,
        "source_hashes": source_hashes(),
    })
    return {"rows": rows, "output_dir": str(root)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--evaluation-horizon", type=int, choices=(0, 2, 4), required=True)
    value.add_argument("--preview-modes", nargs="+", required=True)
    value.add_argument("--episodes", type=int, default=10)
    value.add_argument("--seed-start", type=int, default=ABLATION_SEED_START)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--output-dir", type=Path)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = run_ablation(
        checkpoint=args.checkpoint, evaluation_horizon=args.evaluation_horizon,
        preview_modes=args.preview_modes, episodes=args.episodes,
        seed_start=args.seed_start, episode_length=args.episode_length,
        device=args.device,
        output_dir=args.output_dir or default_output_path("preview_ablation"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
