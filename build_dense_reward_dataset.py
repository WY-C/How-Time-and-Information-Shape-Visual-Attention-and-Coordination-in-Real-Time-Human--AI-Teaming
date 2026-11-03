#!/usr/bin/env python3
"""Replay experiment logs and build official Overcooked dense reward datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing.dense_reward import build_dense_reward_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-file", type=Path)
    parser.add_argument("--trial-file", type=Path)
    parser.add_argument(
        "--ai-player-index",
        type=int,
        choices=(0, 1),
        default=0,
        help="Player slot controlled by the AI in the recorded experiment.",
    )
    parser.add_argument(
        "--allow-sparse-reward-mismatch",
        action="store_true",
        help="Write outputs despite replay/log sparse reward mismatches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dense_reward_dataset(
        args.input_dir,
        args.output_dir,
        step_file=args.step_file,
        trial_file=args.trial_file,
        ai_player_index=args.ai_player_index,
        allow_sparse_reward_mismatch=args.allow_sparse_reward_mismatch,
    )
    print(f"trials: {result.trial_count}")
    print(f"decisions: {result.decision_count}")
    print(f"sparse reward mismatches: {result.sparse_reward_mismatch_count}")
    print("outputs:")
    for name, path in result.output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
