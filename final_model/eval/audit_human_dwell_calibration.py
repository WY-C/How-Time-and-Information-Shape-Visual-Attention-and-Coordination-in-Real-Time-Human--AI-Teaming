#!/usr/bin/env python3
"""Reproduce and audit the 30,668-pair human dwell calibration cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from final_model.eval.human_benchmark import (
    load_human_data,
    parse_processed_segments,
    process_eligibility,
)
from final_model.partial_ver.gaze import ReferenceGeometry


PREP_MS = 135.0
# These are the exact decimal literals in the recovered calibration command.
# The source execution fit retained more digits (recorded in the manifest).
EXEC_INTERCEPT_MS = 44.8732080135
EXEC_SLOPE_MS_PER_DEGREE = 3.83832565515
FLOOR_MS = 60.0
CAP_MS = 4000.0
EXPECTED_PAIR_COUNT = 30_668
RECOVERED_SESSION_ID = "01a06111-58a1-76e2-970e-a99c1826b609"
RECOVERED_COMMAND_TIMESTAMP = "2026-09-04T10:57:38.938Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _position(value: object) -> tuple[int, int] | None:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, list) or len(parsed) != 2:
        return None
    try:
        return tuple(map(int, parsed))
    except (TypeError, ValueError):
        return None


def _visual_angle(
    geometry: ReferenceGeometry,
    source: tuple[int, int] | None,
    destination: tuple[int, int] | None,
) -> float:
    if source is None or destination is None:
        return float("nan")
    try:
        return geometry.visual_angle_degrees(source, destination)
    except (TypeError, ValueError):
        return float("nan")


def build_pairs(data_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    root, eligible, _, fixation_files = load_human_data(data_root)
    eligibility_audit, eligibility_summary, valid_trials = process_eligibility(eligible)
    metadata = eligible.loc[
        eligible.trial_id.isin(valid_trials), ["pid", "trial_id", "Layout"]
    ]
    _, fixations = parse_processed_segments(fixation_files)
    fixations = fixations[fixations.trial_id.isin(valid_trials)].merge(
        metadata[["trial_id", "Layout"]], on="trial_id", validate="many_to_one"
    )
    fixations = fixations[fixations.Layout.eq("counter_circuit")].sort_values(
        ["trial_id", "fixation_start_ms", "fixation_id"]
    )

    grouped = fixations.groupby("trial_id", sort=False)
    fixations["source_fixation_id"] = grouped.fixation_id.shift()
    fixations["source_position"] = grouped.base_position.shift().map(_position)
    fixations["destination_position"] = fixations.base_position.map(_position)
    fixations["onset_gap_ms"] = grouped.fixation_start_ms.diff()
    geometry = ReferenceGeometry()
    fixations["visual_angle_deg"] = [
        _visual_angle(geometry, source, destination)
        for source, destination in zip(
            fixations.source_position, fixations.destination_position
        )
    ]
    pairs = fixations[
        fixations.onset_gap_ms.gt(0.0) & fixations.visual_angle_deg.notna()
    ].copy()
    pairs["prep_ms"] = PREP_MS
    pairs["execution_ms"] = (
        EXEC_INTERCEPT_MS + EXEC_SLOPE_MS_PER_DEGREE * pairs.visual_angle_deg
    )
    pairs["residual_ms"] = (
        pairs.onset_gap_ms - pairs.prep_ms - pairs.execution_ms
    )
    pairs["non_positive_residual"] = pairs.residual_ms.le(0.0)
    pairs["at_or_below_floor"] = pairs.residual_ms.le(FLOOR_MS)
    pairs["source_fixation_id"] = pairs.source_fixation_id.astype(int)

    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_PAIR_COUNT:,} calibration pairs, got {len(pairs):,}"
        )

    input_files = sorted(
        path
        for pattern in ("P*/trial.csv", "P*/step.csv", "P*/fixation.csv")
        for path in root.glob(pattern)
    )
    project_root = Path(__file__).resolve().parents[2]
    dependency_files = [
        project_root / "final_model" / "eval" / "human_benchmark.py",
        project_root / "final_model" / "partial_ver" / "gaze.py",
        project_root
        / "final_model"
        / "eval"
        / "results"
        / "emma_human_fit_v1"
        / "fit_summary.json",
    ]
    provenance = {
        "data_root": str(root),
        "input_files": [
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in input_files
        ],
        "input_file_count": len(input_files),
        "code_and_parameter_dependencies": [
            {
                "path": str(path.relative_to(project_root)),
                "sha256": _sha256(path),
            }
            for path in dependency_files
        ],
        "canonical_eligible_trials": int(len(eligible)),
        "process_valid_trials": int(len(valid_trials)),
        "process_eligibility": eligibility_summary,
        "process_exclusion_counts": {
            "session_fail": int((~eligibility_audit.session_pass).sum()),
            "trial_fail": int((~eligibility_audit.trial_pass).sum()),
            "decision_count_fail": int(
                (~eligibility_audit.decision_count_pass).sum()
            ),
        },
        "recovered_original_execution": {
            "codex_session_id": RECOVERED_SESSION_ID,
            "command_timestamp_utc": RECOVERED_COMMAND_TIMESTAMP,
        },
        "filter_order": [
            "completed trials with exactly 180 decisions and canonical Time/Preview",
            "session mean gaze validity >= 0.80",
            "trial mean gaze validity >= 0.90",
            "Layout == counter_circuit",
            "consecutive fixation rows within trial after onset/id sort",
            "destination onset - source onset > 0 ms",
            "source and destination map to valid canonical tiles",
        ],
        "residual_definition": (
            "destination fixation onset - source fixation onset - 135 ms PREP "
            "- (44.8732080135 + 3.83832565515 * visual_angle_deg) ms EXEC"
        ),
        "execution_fit_full_precision": {
            "intercept_ms": 44.8732080135294,
            "slope_ms_per_degree": 3.838325655150673,
        },
    }
    return pairs, provenance


def summarize(pairs: pd.DataFrame, provenance: dict[str, object]) -> dict[str, object]:
    residual = pairs.residual_ms.to_numpy(dtype=np.float64)
    positive = residual[residual > 0.0]
    positive_mean = float(positive.mean())
    positive_median = float(np.median(positive))

    # The deployed values were not the log-sample MLE. They were chosen to
    # match the arithmetic mean and median of the positive residuals.
    log_mean = math.log(positive_median)
    log_sigma = math.sqrt(2.0 * math.log(positive_mean / positive_median))
    non_positive_count = int(np.count_nonzero(residual <= 0.0))
    floor_count = int(np.count_nonzero(residual <= FLOOR_MS))
    non_positive_mass = non_positive_count / len(residual)
    empirical_floor_mass = floor_count / len(residual)
    model_floor_mass = non_positive_mass + (1.0 - non_positive_mass) * norm.cdf(
        (math.log(FLOOR_MS) - log_mean) / log_sigma
    )
    mle_log_mean = float(np.log(positive).mean())
    mle_log_sigma = float(np.log(positive).std(ddof=0))
    mle_floor_mass = non_positive_mass + (1.0 - non_positive_mass) * norm.cdf(
        (math.log(FLOOR_MS) - mle_log_mean) / mle_log_sigma
    )

    return {
        "analysis": "counter_circuit_human_dwell_calibration_30668_audit_v1",
        "provenance": provenance,
        "timing_constants": {
            "prep_ms": PREP_MS,
            "execution_intercept_ms": EXEC_INTERCEPT_MS,
            "execution_slope_ms_per_degree": EXEC_SLOPE_MS_PER_DEGREE,
            "floor_ms": FLOOR_MS,
            "cap_ms": CAP_MS,
        },
        "cohort": {
            "pair_count": int(len(residual)),
            "participant_count": int(pairs.pid.nunique()),
            "trial_count": int(pairs.trial_id.nunique()),
        },
        "empirical": {
            "non_positive_residual_count": non_positive_count,
            "non_positive_residual_mass": non_positive_mass,
            "residual_at_or_below_60_ms_count": floor_count,
            "residual_at_or_below_60_ms_mass": empirical_floor_mass,
            "positive_residual_count": int(len(positive)),
            "positive_residual_mean_ms": positive_mean,
            "positive_residual_median_ms": positive_median,
            "residual_quantiles_ms": {
                str(q): float(np.quantile(residual, q))
                for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
            },
        },
        "recovered_deployed_fit_method": "positive-residual mean/median moment match",
        "recovered_deployed_fit": {
            "floor_probability": non_positive_mass,
            "log_mean": log_mean,
            "log_sigma": log_sigma,
            "cap_ms": CAP_MS,
            "model_implied_mass_at_60_ms_after_floor_clamp": model_floor_mass,
        },
        "positive_residual_lognormal_mle_diagnostic": {
            "log_mean": mle_log_mean,
            "log_sigma": mle_log_sigma,
            "model_implied_mass_at_60_ms_after_floor_clamp": mle_floor_mass,
            "empirical_difference_percentage_points": 100.0
            * (empirical_floor_mass - mle_floor_mass),
        },
        "floor_mass_audit": {
            "empirical_mass": empirical_floor_mass,
            "model_implied_mass": model_floor_mass,
            "difference_probability": empirical_floor_mass - model_floor_mass,
            "difference_percentage_points": 100.0
            * (empirical_floor_mass - model_floor_mass),
            "retain_current_implementation_under_10_3_percent_rule": False,
        },
        "terminology": {
            "recommended": "non-positive residual mass",
            "do_not_call_it": "zero-HOLD probability",
        },
    }


def run(data_root: Path, output_dir: Path) -> dict[str, object]:
    pairs, provenance = build_pairs(data_root)
    summary = summarize(pairs, provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "pid",
        "trial_id",
        "source_fixation_id",
        "fixation_id",
        "fixation_start_ms",
        "onset_gap_ms",
        "source_position",
        "destination_position",
        "visual_angle_deg",
        "prep_ms",
        "execution_ms",
        "residual_ms",
        "non_positive_residual",
        "at_or_below_floor",
    ]
    pairs[columns].to_csv(output_dir / "calibration_pairs.csv.gz", index=False)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = summary["floor_mass_audit"]
    empirical = summary["empirical"]
    lines = [
        "# Human dwell calibration provenance audit",
        "",
        f"- Recovered cohort: {summary['cohort']['pair_count']:,} pairs",
        f"- Non-positive residual mass: {empirical['non_positive_residual_count']:,} / "
        f"{summary['cohort']['pair_count']:,} = "
        f"{100 * empirical['non_positive_residual_mass']:.6f}%",
        f"- Empirical P(D <= 60 ms): {empirical['residual_at_or_below_60_ms_count']:,} / "
        f"{summary['cohort']['pair_count']:,} = "
        f"{100 * audit['empirical_mass']:.6f}%",
        f"- Deployed approximation implied mass at 60 ms: "
        f"{100 * audit['model_implied_mass']:.6f}%",
        f"- Difference: {audit['difference_percentage_points']:.6f} percentage points",
        "- Decision: do not retain the current approximation under the stated 10.3% rule.",
        "- Manuscript term: `non-positive residual mass`, not `zero-HOLD probability`.",
        "",
        "See `audit_summary.json` for the complete filter chain and SHA-256 input manifest,",
        "and `calibration_pairs.csv.gz` for the row-level cohort.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    value.add_argument(
        "--data-root", type=Path, default=project_root / "data" / "raw" / "experiment"
    )
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary = run(args.data_root, args.output_dir)
    empirical = summary["empirical"]
    audit = summary["floor_mass_audit"]
    print(f"pairs={summary['cohort']['pair_count']}")
    print(
        "non_positive_mass="
        f"{empirical['non_positive_residual_mass']:.12f} "
        f"({empirical['non_positive_residual_count']}/{summary['cohort']['pair_count']})"
    )
    print(
        f"empirical_p_D_le_60={audit['empirical_mass']:.12f} "
        f"({empirical['residual_at_or_below_60_ms_count']}/{summary['cohort']['pair_count']})"
    )
    print(f"model_implied_floor_mass={audit['model_implied_mass']:.12f}")
    print(f"difference_percentage_points={audit['difference_percentage_points']:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
