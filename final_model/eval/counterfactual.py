"""Same-state, same-RNN preview counterfactual action diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final_model.common.overcooked_backend import _imports
from final_model.eval.common import (
    COUNTERFACTUAL_SEED_START,
    ACTION_NAMES,
    mask_full_observation,
    valid_preview_modes,
    default_output_path,
)


Action, _Gridworld, _shaping, _displayed = _imports()
INTERACT_INDEX = int(Action.ACTION_TO_INDEX[Action.INTERACT])


def distribution_metrics(native: np.ndarray, counterfactual: np.ndarray) -> dict[str, Any]:
    left = np.asarray(native, dtype=np.float64)
    right = np.asarray(counterfactual, dtype=np.float64)
    if left.shape != (6,) or right.shape != (6,):
        raise ValueError("action probability vectors must have six entries")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("action probabilities must be finite")
    if not np.isclose(left.sum(), 1.0, atol=1e-6) or not np.isclose(
        right.sum(), 1.0, atol=1e-6
    ):
        raise ValueError("action probabilities must sum to one")
    epsilon = np.finfo(np.float64).tiny
    left_safe = np.clip(left, epsilon, 1.0)
    right_safe = np.clip(right, epsilon, 1.0)
    middle = 0.5 * (left_safe + right_safe)
    kl = float(np.sum(left * np.log(left_safe / right_safe)))
    js = float(
        0.5 * np.sum(left * np.log(left_safe / middle))
        + 0.5 * np.sum(right * np.log(right_safe / middle))
    )
    native_top = int(left.argmax())
    counter_top = int(right.argmax())
    absolute = np.abs(left - right)
    native_sorted = np.sort(left)
    counter_sorted = np.sort(right)
    native_entropy = float(-np.sum(left * np.log(left_safe)))
    counter_entropy = float(-np.sum(right * np.log(right_safe)))
    stay_index = ACTION_NAMES.index("STAY") if "STAY" in ACTION_NAMES else None
    movement_indices = [
        index for index, name in enumerate(ACTION_NAMES)
        if name in {"NORTH", "SOUTH", "EAST", "WEST"}
    ]
    return {
        "kl_native_counterfactual": max(kl, 0.0),
        "js_divergence": max(js, 0.0),
        "l1_probability_distance": float(absolute.sum()),
        "total_variation_distance": float(0.5 * absolute.sum()),
        "mean_abs_probability_difference": float(absolute.mean()),
        "max_abs_probability_difference": float(absolute.max()),
        "top1_action_changed": int(native_top != counter_top),
        "native_top1_action": native_top,
        "native_top1_action_name": ACTION_NAMES[native_top],
        "counterfactual_top1_action": counter_top,
        "counterfactual_top1_action_name": ACTION_NAMES[counter_top],
        "native_entropy": native_entropy,
        "counterfactual_entropy": counter_entropy,
        "delta_entropy": float(counter_entropy - native_entropy),
        "native_top1_margin": float(native_sorted[-1] - native_sorted[-2]),
        "counterfactual_top1_margin": float(
            counter_sorted[-1] - counter_sorted[-2]
        ),
        "delta_top1_margin": float(
            (counter_sorted[-1] - counter_sorted[-2])
            - (native_sorted[-1] - native_sorted[-2])
        ),
        "delta_P_INTERACT": float(right[INTERACT_INDEX] - left[INTERACT_INDEX]),
        "abs_delta_P_INTERACT": float(
            abs(right[INTERACT_INDEX] - left[INTERACT_INDEX])
        ),
        "delta_P_STAY": (
            None if stay_index is None else float(right[stay_index] - left[stay_index])
        ),
        "abs_delta_P_STAY": (
            None if stay_index is None else float(abs(right[stay_index] - left[stay_index]))
        ),
        "delta_movement_probability": float(
            right[movement_indices].sum() - left[movement_indices].sum()
        ),
        **{f"native_p_{name.lower()}": float(left[index]) for index, name in enumerate(ACTION_NAMES)},
        **{
            f"counterfactual_p_{name.lower()}": float(right[index])
            for index, name in enumerate(ACTION_NAMES)
        },
        **{
            f"delta_p_{name.lower()}": float(right[index] - left[index])
            for index, name in enumerate(ACTION_NAMES)
        },
    }


def opportunity_labels(state, mdp) -> dict[str, bool]:
    learner = state.players[1]
    held = getattr(learner, "held_object", None)
    held_name = None if held is None else str(held.name)
    target = tuple(
        int(value + direction)
        for value, direction in zip(learner.position, learner.orientation)
    )
    x, y = target
    terrain = (
        None
        if y < 0 or y >= len(mdp.terrain_mtx) or x < 0 or x >= len(mdp.terrain_mtx[y])
        else str(mdp.terrain_mtx[y][x])
    )
    return {
        "learner_holding_onion": held_name == "onion",
        "learner_holding_dish": held_name == "dish",
        "learner_holding_soup": held_name == "soup",
        "facing_counter": terrain == "X",
        "facing_pot": terrain == "P",
        "facing_dispenser": terrain in {"O", "D"},
        "facing_delivery": terrain == "S",
        "onion_counter_place_opportunity": (
            held_name == "onion" and terrain == "X" and target not in state.objects
        ),
    }


def actor_probabilities(actor, observation, hidden, mask, available) -> np.ndarray:
    with torch.no_grad():
        probabilities, _discarded_hidden = actor.get_probs(
            np.asarray(observation)[None],
            np.asarray(hidden).copy(),
            np.asarray(mask).copy(),
            np.asarray(available)[None],
        )
    return probabilities.detach().cpu().numpy()[0]


class CounterfactualCollector:
    def __init__(self):
        self.rows: list[dict[str, Any]] = []

    def observe(
        self, *, actor, observation, hidden, mask, available, native_probabilities,
        evaluation_horizon: int, episode: int, timestep: int, state, mdp,
    ) -> None:
        hidden_before = np.asarray(hidden).copy()
        labels = opportunity_labels(state, mdp)
        for mode in valid_preview_modes(evaluation_horizon):
            counterfactual_observation = mask_full_observation(
                observation,
                evaluation_horizon=evaluation_horizon,
                preview_mode=mode,
            )
            probabilities = actor_probabilities(
                actor, counterfactual_observation, hidden_before, mask, available
            )
            row = {
                "episode": int(episode),
                "timestep": int(timestep),
                "evaluation_horizon": int(evaluation_horizon),
                "counterfactual_preview_mode": mode,
                **labels,
                **distribution_metrics(native_probabilities, probabilities),
            }
            self.rows.append(row)
        if not np.array_equal(hidden_before, hidden):
            raise RuntimeError("counterfactual query mutated native recurrent state")

    def summary(self) -> dict[str, Any]:
        subsets = {
            "all_decisions": lambda row: True,
            "learner_holding_onion": lambda row: bool(row["learner_holding_onion"]),
            "onion_counter_place_opportunity": lambda row: bool(
                row["onion_counter_place_opportunity"]
            ),
        }
        output = {}
        for subset, predicate in subsets.items():
            by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in self.rows:
                if predicate(row):
                    by_mode[row["counterfactual_preview_mode"]].append(row)
            output[subset] = {}
            for mode, rows in by_mode.items():
                output[subset][mode] = {
                    "decision_count": len(rows),
                    "mean_kl": float(np.mean([row["kl_native_counterfactual"] for row in rows])),
                    "mean_js": float(np.mean([row["js_divergence"] for row in rows])),
                    "top1_change_rate": float(np.mean([row["top1_action_changed"] for row in rows])),
                    "mean_abs_delta_P_INTERACT": float(
                        np.mean([abs(row["delta_P_INTERACT"]) for row in rows])
                    ),
                }
        return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--evaluation-horizon", type=int, choices=(0, 2, 4), required=True)
    value.add_argument("--episodes", type=int, default=10)
    value.add_argument("--seed-start", type=int, default=COUNTERFACTUAL_SEED_START)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--output-dir", type=Path)
    return value


def main(argv=None) -> int:
    from final_model.eval.rollout import EvaluationConfig, evaluate

    args = parser().parse_args(argv)
    result = evaluate(
        EvaluationConfig(
            checkpoint=args.checkpoint,
            evaluation_horizon=args.evaluation_horizon,
            episodes=args.episodes,
            seed_start=args.seed_start,
            episode_length=args.episode_length,
            device=args.device,
            counterfactual=True,
            output_dir=args.output_dir or default_output_path("counterfactual"),
        )
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
