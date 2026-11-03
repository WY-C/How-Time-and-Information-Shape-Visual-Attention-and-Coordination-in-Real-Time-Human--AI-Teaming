#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
from pathlib import Path

import pandas as pd


def run_script(script: Path, args: list[str]) -> None:
    """Run one pipeline script in-process with isolated argv."""
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script), *args]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def consolidate_posthoc(out: str) -> None:
    stats = Path(out) / "02_statistics"
    files = [
        "overall_aoi_lmm_posthoc_holm.csv",
        "temporal_posthoc_lmm_holm.csv",
        "response_whole_posthoc_lmm_holm.csv",
        "response_symmetric_posthoc_lmm_holm.csv",
        "h1_h4_lmm_posthoc_holm.csv",
        "h1_h4_relative_lmm_posthoc_holm.csv",
        "h1_h4_near_far_lmm_posthoc_holm.csv",
        "behavior_posthoc_holm.csv",
        "subjective_posthoc_holm.csv",
    ]
    parts = []
    for name in files:
        f = stats / name
        if not f.exists() or f.stat().st_size == 0:
            continue
        d = pd.read_csv(f)
        if d.empty:
            continue
        d.insert(0, "source_file", name)
        parts.append(d)
    if not parts:
        raise RuntimeError("No post-hoc outputs were generated")
    out_df = pd.concat(parts, ignore_index=True, sort=False)
    out_df.to_csv(stats / "all_posthoc_holm.csv", index=False, encoding="utf-8-sig")
    print(f"Consolidated post-hoc: {len(out_df)} rows -> {stats / 'all_posthoc_holm.csv'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--subjective", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--symmetric-ms", type=float, default=100.0)
    p.add_argument("--min-symmetric-ms", type=float, default=50.0)
    args = p.parse_args()

    # Stable headless plotting and bounded linear-algebra threading.
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    here = Path(__file__).resolve().parent
    common = ["--roots", *args.roots, "--out", args.out, "--alpha", str(args.alpha)]
    jobs = [
        (here / "00_trial_eligibility.py", common),
        (here / "01_gaze_reanalysis.py", [*common, "--symmetric-ms", str(args.symmetric_ms), "--min-symmetric-ms", str(args.min_symmetric_ms)]),
        (here / "02_behavior_reanalysis.py", common),
        (here / "03_subjective_reanalysis.py", [*common, "--subjective", args.subjective]),
    ]

    for script, script_args in jobs:
        print(f"\n=== {script.name} ===", flush=True)
        t0 = time.perf_counter()
        run_script(script, script_args)
        print(f"Completed {script.name} in {time.perf_counter() - t0:.2f}s", flush=True)

    print("\n=== consolidate_posthoc ===", flush=True)
    t0 = time.perf_counter()
    consolidate_posthoc(args.out)
    print(f"Completed post-hoc consolidation in {time.perf_counter() - t0:.2f}s", flush=True)

    validator = here / "04_validate_outputs.py"
    print(f"\n=== {validator.name} ===", flush=True)
    t0 = time.perf_counter()
    run_script(validator, ["--out", args.out])
    print(f"Completed {validator.name} in {time.perf_counter() - t0:.2f}s", flush=True)

    print(f"\nAll analyses completed: {args.out}")


if __name__ == "__main__":
    main()
