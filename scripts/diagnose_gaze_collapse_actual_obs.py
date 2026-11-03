#!/usr/bin/env python3
"""Measure GazeNet fusion collapse on actual evaluation observations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.eval.partial_rollout import PartialEvaluationConfig, evaluate_partial


def diagnose(path: Path, *, seed: int, episode_length: int) -> dict[str, object]:
    loaded = load_partial_checkpoint_for_evaluation(path, device="cpu")
    if loaded.gaze_actor is None:
        raise RuntimeError(f"checkpoint has no GazeNet: {path}")

    fusion_inputs: list[torch.Tensor] = []
    fusion_pre: list[torch.Tensor] = []
    fusion_relu: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []

    def linear_hook(_module, inputs, output):
        fusion_inputs.append(inputs[0].detach().cpu())
        fusion_pre.append(output.detach().cpu())

    def relu_hook(_module, _inputs, output):
        fusion_relu.append(output.detach().cpu())

    def head_hook(_module, _inputs, output):
        logits.append(output.detach().cpu())

    handles = [
        loaded.gaze_actor.fusion[0].register_forward_hook(linear_hook),
        loaded.gaze_actor.fusion[1].register_forward_hook(relu_hook),
        loaded.gaze_actor.head.register_forward_hook(head_hook),
    ]
    try:
        rollout = evaluate_partial(
            PartialEvaluationConfig(
                checkpoint=path,
                decision_time=400,
                horizon=2,
                fixation_shape=loaded.metadata["fixation_shapes"][0],
                gaze_type="learned",
                episodes=1,
                episode_length=episode_length,
                seed_start=seed,
                device="cpu",
                gaze_deterministic=False,
                action_deterministic=False,
            ),
            loaded=loaded,
        )
    finally:
        for handle in handles:
            handle.remove()

    x = torch.cat(fusion_inputs)
    pre = torch.cat(fusion_pre)
    relu = torch.cat(fusion_relu)
    raw_logits = torch.cat(logits)
    probability = raw_logits.softmax(-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    argmax = probability.argmax(-1)
    argmax_counts = torch.bincount(argmax, minlength=40)
    temporal_variation = (
        (x[1:] - x[:-1]).abs().mean().item() if len(x) > 1 else 0.0
    )
    summary = rollout["summary"]
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_step": loaded.completed_steps,
        "memory_max_lifetime_ms": loaded.metadata.get("memory_max_lifetime_ms"),
        "gaze_calls": int(len(relu)),
        "fusion_pre_mean": float(pre.mean()),
        "fusion_pre_std": float(pre.std(unbiased=False)),
        "fusion_relu_positive_fraction": float((relu > 0).float().mean()),
        "fusion_relu_allzero_row_fraction": float((relu == 0).all(-1).float().mean()),
        "fusion_input_temporal_variation": temporal_variation,
        "normalized_policy_entropy": float(entropy.mean() / math.log(40)),
        "argmax_unique_targets": int((argmax_counts > 0).sum()),
        "argmax_max_target_share": float(argmax_counts.max() / len(argmax)),
        "mean_sparse_team_reward": float(summary["mean_sparse_team_reward"]),
        "mean_deliveries": float(summary["mean_deliveries"]),
        "collision_rate": float(summary["collision_rate"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--episode-length", type=int, default=180)
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        print(json.dumps(
            diagnose(checkpoint, seed=args.seed, episode_length=args.episode_length),
            indent=2,
            sort_keys=True,
        ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
