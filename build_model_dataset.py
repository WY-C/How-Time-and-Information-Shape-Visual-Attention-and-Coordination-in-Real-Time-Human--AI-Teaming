#!/usr/bin/env python3
"""CLI for building model-ready decision and QC datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from preprocessing.model_dataset import build_model_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for kind in ("step", "fixation", "raw_gaze", "trial", "schedule", "metadata"):
        parser.add_argument(f"--{kind.replace('_', '-')}-file", type=Path)
    parser.add_argument(
        "--unknown-aoi-policy",
        choices=("outside", "unknown", "error"),
        default="outside",
        help="How to handle base AOI labels absent from the explicit mapping table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    explicit_files = {
        kind: getattr(args, f"{kind}_file")
        for kind in ("step", "fixation", "raw_gaze", "trial", "schedule", "metadata")
        if getattr(args, f"{kind}_file") is not None
    }
    result = build_model_dataset(
        args.input_dir,
        args.output_dir,
        input_files=explicit_files,
        unknown_aoi_policy=args.unknown_aoi_policy,
    )
    print(f"participants: {result.participant_count}")
    print(f"trials: {result.trial_count}")
    print(f"decisions: {result.decision_count}")
    print(f"fixation segments: {result.fixation_segment_count}")
    print(f"timestamp errors: {result.timestamp_error_count}")
    print(f"decisions without fixation: {result.decisions_without_fixation}")
    print(f"QC warnings: {result.qc_warning_count}")
    print("outputs:")
    for name, path in result.output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

