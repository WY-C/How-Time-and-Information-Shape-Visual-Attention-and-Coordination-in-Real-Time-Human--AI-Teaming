"""CLI for fixed-horizon Full Action performance and optional behavior logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_model.eval.common import FINAL_ANALYSIS_SEED_START, default_output_path
from final_model.eval.rollout import EvaluationConfig, evaluate


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--evaluation-horizon", type=int, choices=(0, 2, 4), required=True)
    value.add_argument("--episodes", type=int, default=10)
    value.add_argument("--seed-start", type=int, default=FINAL_ANALYSIS_SEED_START)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--learner-action-mode", choices=("deterministic", "stochastic"), default="deterministic")
    value.add_argument("--behavior-events", action="store_true")
    value.add_argument("--output-dir", type=Path)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = evaluate(EvaluationConfig(
        checkpoint=args.checkpoint,
        evaluation_horizon=args.evaluation_horizon,
        episodes=args.episodes,
        seed_start=args.seed_start,
        episode_length=args.episode_length,
        device=args.device,
        learner_action_mode=args.learner_action_mode,
        behavior_events=args.behavior_events,
        output_dir=args.output_dir or default_output_path("performance"),
    ))
    print(json.dumps({**result.summary, "output_dir": str(result.output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
