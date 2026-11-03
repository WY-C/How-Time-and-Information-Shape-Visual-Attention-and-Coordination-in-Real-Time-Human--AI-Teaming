#!/usr/bin/env python3
"""Canonical sparse-task evaluation for Full and Partial checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_PARENT = Path(__file__).resolve().parents[1]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from final_model.checkpoint import load_action_checkpoint, load_gaze_checkpoint
from final_model.common.mappo import build_actionnet
from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.common.partners import parameter_hash
from final_model.partial_ver.network import GazeNet
from final_model.partial_ver.gaze import HUMAN_DWELL_RNG_SEED_OFFSET
from final_model.partial_ver.memory import MEMORY_RNG_SEED_OFFSET
from final_model.partial_ver.constraints import constraint_config_from_metadata
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.peripheral import PERIPHERAL_RNG_SEED_OFFSET
from final_model.partial_ver.preview_opacity import PREVIEW_OPACITY_RNG_SEED_OFFSET
from final_model.partial_ver.training import LearnedGazeController
from final_model.server_parallel.gaze_v1 import action_index_to_tile
from final_model.simple_ver.observation import FullObservationAdapter


class EvaluationGazeController(LearnedGazeController):
    def __call__(self, memory, fixation, context):
        observation = self.spatial_input(memory, fixation)
        with torch.no_grad():
            distribution = self.actor.distribution(
                torch.as_tensor(observation[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(context[None], dtype=torch.float32, device=self.device),
            )
            index = int(distribution.probs.argmax(-1).item())
        return action_index_to_tile(index)

    def reset(self):
        self.active = None
        self.completed = []


def evaluate(
    *, checkpoint: Path, observation_mode: str, training_stage: str, horizon: int,
    decision_time: int | None, fixation_shape: str | None, episodes: int, episode_length: int,
    seed: int, device: str, partner: str | None = None,
):
    target = torch.device(device)
    action_actor = build_actionnet(seed=seed, device=target)
    gaze_actor = None
    if training_stage == "action":
        metadata = load_action_checkpoint(
            checkpoint,
            actor=action_actor,
            expected_mode=observation_mode,
            expected_horizon=horizon,
            expected_condition=(
                {"decision_time": int(decision_time), "fixation_shape": fixation_shape}
                if observation_mode == "partial"
                else None
            ),
        )
    else:
        if observation_mode != "partial":
            raise ValueError("full + gaze evaluation is invalid")
        gaze_actor = GazeNet().to(target)
        metadata = load_gaze_checkpoint(
            checkpoint, action_actor=action_actor, gaze_actor=gaze_actor,
            horizon=horizon, decision_time=decision_time, fixation_shape=fixation_shape,
        )
    action_actor.eval()
    actor_before = parameter_hash(action_actor)
    backend = DirectOvercookedBackend(
        seed=seed, horizon=horizon, episode_length=episode_length, device=device, partner=partner
    )
    if observation_mode == "full":
        adapter = FullObservationAdapter(backend)
    else:
        controller = None
        if gaze_actor is not None:
            controller = EvaluationGazeController(gaze_actor, None, target)
        constraint_config = constraint_config_from_metadata(metadata)
        metadata.update(constraint_config.metadata())
        adapter = PartialObservationAdapter(
            backend, decision_time=int(decision_time), fixation_shape=str(fixation_shape),
            gaze_policy=controller,
            memory_lifetime_ms=metadata.get("memory_lifetime_ms", 2400.0),
            memory_decay_mode=metadata.get("memory_decay_mode", "linear"),
            memory_tau_ms=metadata.get("memory_tau_ms"),
            memory_seed=(
                seed + MEMORY_RNG_SEED_OFFSET
                if metadata.get("memory_decay_mode") == "stochastic_exponential"
                else None
            ),
            constraint_config=constraint_config,
            peripheral_seed=seed + PERIPHERAL_RNG_SEED_OFFSET,
            opacity_seed=seed + PREVIEW_OPACITY_RNG_SEED_OFFSET,
            dwell_seed=(
                seed + HUMAN_DWELL_RNG_SEED_OFFSET
                if constraint_config.emma.mode == "human_stochastic"
                else None
            ),
        )
    hidden = np.zeros((1, 1, 64), dtype=np.float32)
    mask = np.zeros((1, 1), dtype=np.float32)
    rows = []
    sparse = shaped = optimization = 0.0
    deliveries = steps = 0
    while len(rows) < episodes:
        def policy(observation, _critic, available):
            nonlocal hidden, mask
            with torch.no_grad():
                probabilities, next_hidden = action_actor.get_probs(
                    observation[None], hidden, mask, available[None]
                )
            hidden = next_hidden.detach().cpu().numpy()
            mask.fill(1.0)
            return int(probabilities.argmax(-1).item())

        transition = adapter.run_decision(policy)
        result = transition[-1] if observation_mode == "full" else transition.result
        sparse += result["sparse_team_reward"]
        shaped += result["native_shaped_reward"]
        optimization += result["optimization_reward"]
        deliveries += result["deliveries"]
        steps += 1
        if result["done"]:
            rows.append({
                "episode": len(rows), "steps": steps, "sparse_team_reward": sparse,
                "native_shaped_reward": shaped, "optimization_reward": optimization,
                "deliveries": deliveries,
            })
            if len(rows) < episodes:
                adapter.reset_episode()
                hidden.fill(0.0)
                mask.fill(0.0)
                sparse = shaped = optimization = 0.0
                deliveries = steps = 0
    actor_after = parameter_hash(action_actor)
    return {
        "checkpoint_metadata": metadata,
        "observation_mode": observation_mode,
        "training_stage": training_stage,
        "horizon": horizon,
        "decision_time": decision_time,
        "fixation_shape": fixation_shape,
        "episodes": episodes,
        "eval/reward": float(np.mean([row["sparse_team_reward"] for row in rows])),
        "eval/deliveries": float(np.mean([row["deliveries"] for row in rows])),
        "eval/native_shaped_reward": float(np.mean([row["native_shaped_reward"] for row in rows])),
        "eval/optimization_reward": float(np.mean([row["optimization_reward"] for row in rows])),
        "task_decisions": adapter.task_decisions,
        "actionnet_online_forwards": adapter.actionnet_online_forwards,
        "online_forwards_per_decision": adapter.actionnet_online_forwards / adapter.task_decisions,
        "actor_hash_before": actor_before,
        "actor_hash_after": actor_after,
        "actor_unchanged": actor_before == actor_after,
        "sampled_partner_identities": dict(backend.partners.counts),
        "episode_results": rows,
    }


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--observation-mode", choices=("full", "partial"), required=True)
    value.add_argument("--training-stage", choices=("action", "gaze"), required=True)
    value.add_argument("--horizon", type=int, choices=(0, 2, 4), required=True)
    value.add_argument("--decision-time", type=int, choices=(200, 400, 600))
    value.add_argument("--fixation-shape", choices=("3x3", "2x2", "plus"))
    value.add_argument("--episodes", type=int, default=10)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--seed", type=int, default=20260821)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--partner")
    value.add_argument("--output", type=Path)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.observation_mode == "partial" and (args.decision_time is None or args.fixation_shape is None):
        raise ValueError("Partial evaluation requires decision time and fixation shape")
    values = vars(args).copy()
    values.pop("output")
    result = evaluate(**values)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
