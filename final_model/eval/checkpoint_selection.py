"""Delivery-only ranking of candidate checkpoints from selection evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from final_model.eval.common import (
    FINAL_ANALYSIS_SEED_START,
    SELECTION_SEED_START,
    assert_seed_partitions_distinct,
    create_output_directory,
    default_output_path,
    source_hashes,
    write_csv,
    write_json,
)


ALLOWED_SELECTION_INPUTS = {
    "checkpoint_path", "checkpoint_sha256", "checkpoint_completed_step",
    "checkpoint_horizon_mode", "evaluation_horizon", "mean_deliveries",
    "std_deliveries", "mean_sparse_team_reward",
}


def select_checkpoints(rows: list[dict[str, Any]], *, shared: bool) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("selection requires at least one summary row")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for original in rows:
        row = {key: original[key] for key in ALLOWED_SELECTION_INPUTS if key in original}
        key = str(row.get("checkpoint_sha256") or row.get("checkpoint_path"))
        grouped.setdefault(key, []).append(row)
    candidates = []
    for key, group in grouped.items():
        by_horizon = {int(row["evaluation_horizon"]): row for row in group}
        if shared and set(by_horizon) != {0, 2, 4}:
            raise ValueError("each shared checkpoint requires H0/H2/H4 selection summaries")
        means = [float(row["mean_deliveries"]) for row in group]
        base = group[0]
        item = {
            "checkpoint_path": base.get("checkpoint_path"),
            "checkpoint_sha256": base.get("checkpoint_sha256", key),
            "checkpoint_completed_step": int(base.get("checkpoint_completed_step", 0)),
            "aggregate_mean_deliveries": float(np.mean(means)),
            "minimum_condition_mean_deliveries": float(np.min(means)),
            "aggregate_delivery_std": float(np.mean([float(row.get("std_deliveries", 0.0)) for row in group])),
            "aggregate_mean_sparse_team_reward": float(np.mean([float(row.get("mean_sparse_team_reward", 0.0)) for row in group])),
        }
        for horizon, row in sorted(by_horizon.items()):
            item[f"h{horizon}_mean_deliveries"] = float(row["mean_deliveries"])
        candidates.append(item)
    best = max(item["aggregate_mean_deliveries"] for item in candidates)
    threshold = 0.95 * best
    for item in candidates:
        item["high_performing"] = item["aggregate_mean_deliveries"] >= threshold
    ordered_steps = sorted(candidates, key=lambda item: item["checkpoint_completed_step"])
    for index, item in enumerate(ordered_steps):
        neighbors = ordered_steps[max(0, index - 1):index] + ordered_steps[index + 1:index + 2]
        item["locally_stable"] = bool(item["high_performing"] and neighbors and any(n["high_performing"] for n in neighbors))
    return sorted(candidates, key=lambda item: (
        -item["aggregate_mean_deliveries"],
        -item["minimum_condition_mean_deliveries"],
        item["aggregate_delivery_std"],
        item["checkpoint_completed_step"],
    ))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--summary", type=Path, action="append", required=True)
    value.add_argument("--shared", action="store_true")
    value.add_argument("--selection-seed-start", type=int, default=SELECTION_SEED_START)
    value.add_argument("--final-analysis-seed-start", type=int, default=FINAL_ANALYSIS_SEED_START)
    value.add_argument("--output-dir", type=Path)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    assert_seed_partitions_distinct(args.selection_seed_start, args.final_analysis_seed_start)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    ranked = select_checkpoints(rows, shared=args.shared)
    output = create_output_directory(args.output_dir or default_output_path("checkpoint_selection"), prefix="checkpoint_selection")
    write_csv(output / "ranked_checkpoints.csv", ranked)
    write_json(output / "ranked_checkpoints.json", ranked)
    write_json(output / "manifest.json", {
        "analysis": "checkpoint_selection",
        "selection_rule": "delivery mean; 95% high-performing; neighboring high-performing stability",
        "behavior_metrics_used": False,
        "shared": args.shared,
        "selection_seed_start": args.selection_seed_start,
        "final_analysis_seed_start": args.final_analysis_seed_start,
        "input_summaries": [str(path.resolve()) for path in args.summary],
        "source_hashes": source_hashes(),
    })
    print(json.dumps({"ranked": ranked, "output_dir": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
