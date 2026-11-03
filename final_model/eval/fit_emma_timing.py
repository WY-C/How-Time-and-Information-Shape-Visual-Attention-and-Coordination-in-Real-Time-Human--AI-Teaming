#!/usr/bin/env python3
"""Fit pooled distance-dependent gaze-shift execution timing from human fixations."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from final_model.partial_ver.gaze import (
    DWELL_MS,
    EXEC_BASE_MS,
    EXEC_MS_PER_DEGREE,
    PREP_MS,
    ReferenceGeometry,
)


DEFAULT_BOOTSTRAP = 20_000
DEFAULT_SEED = 20260828
DEFAULT_MAX_SHIFT_MS = 150.0
SOURCE_GLOB = "P[0-9][0-9]"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("trial_id")]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tile(value: str | None) -> tuple[int, int] | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        parsed = json.loads(value)
        tile = tuple(map(int, parsed))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if len(tile) != 2 or not (0 <= tile[0] < 8 and 0 <= tile[1] < 5):
        return None
    return tile


def load_human_fixations(input_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(input_root).expanduser().resolve()
    session_dirs = sorted(path for path in root.glob(SOURCE_GLOB) if path.is_dir())
    if not session_dirs:
        raise RuntimeError(f"no canonical P01-P36 sessions found under {root}")
    trials: dict[tuple[str, str], dict[str, Any]] = {}
    incomplete_trials: set[tuple[str, str]] = set()
    fixations: list[dict[str, Any]] = []
    sources = []
    for session in session_dirs:
        fixation_path, trial_path = session / "fixation.csv", session / "trial.csv"
        if not fixation_path.is_file() or not trial_path.is_file():
            raise RuntimeError(f"incomplete human session: {session}")
        sources.extend((fixation_path, trial_path))
        for row in _read_csv(trial_path):
            key = (row["pid"], row["trial_id"])
            if key in trials or key in incomplete_trials:
                raise RuntimeError(f"duplicate human trial: {key}")
            metadata = {
                "time_ms": int(row["time_constraint_ms"]),
                "horizon": int(row["information_steps"]),
                "layout": row["layout"],
                "completed": int(row["completed"]),
            }
            if metadata["completed"]:
                trials[key] = metadata
            else:
                incomplete_trials.add(key)
        for row in _read_csv(fixation_path):
            key = (row["pid"], row["trial_id"])
            if key in incomplete_trials:
                continue
            if key not in trials:
                raise RuntimeError(f"fixation has no trial metadata: {key}")
            if row.get("record_type") != "ivt_fixation":
                raise RuntimeError(f"non-physical fixation record in {fixation_path}")
            context_segments = json.loads(row.get("context_segments") or "[]")
            fixations.append({
                "participant": row["pid"], "trial": row["trial_id"],
                "fixation_id": int(row["fixation_id"]),
                "start_ms": float(row["fixation_start_ms"]),
                "end_ms": float(row["fixation_end_ms"]),
                "duration_ms": float(row["fixation_duration_ms"]),
                "centroid_x": float(row["fixation_x"]),
                "centroid_y": float(row["fixation_y"]),
                "tile": _tile(row.get("base_position")),
                "context_segment_count": len(context_segments),
                **trials[key],
            })
    participants = sorted({row["participant"] for row in fixations})
    if participants != [f"P{index:02d}" for index in range(1, 37)]:
        raise RuntimeError(f"canonical source must contain P01-P36, got {participants}")
    provenance = {
        "input_root": str(root),
        "participant_count": len(participants),
        "trial_count": len(trials),
        "excluded_incomplete_trial_count": len(incomplete_trials),
        "excluded_incomplete_trials": [list(key) for key in sorted(incomplete_trials)],
        "fixation_count": len(fixations),
        "fixations_with_multiple_context_segments": sum(
            row["context_segment_count"] > 1 for row in fixations
        ),
        "files": [
            {"path": str(path), "sha256": _sha256(path)} for path in sources
        ],
        "physical_fixation_key": ["participant", "trial", "fixation_id"],
        "source_schema": {
            "participant_id": "pid", "trial_id": "trial_id",
            "condition_time": "time_constraint_ms",
            "condition_horizon": "information_steps", "layout": "layout",
            "physical_fixation_id": "fixation_id",
            "fixation_start_ms": "fixation_start_ms",
            "fixation_end_ms": "fixation_end_ms",
            "centroid_x": "fixation_x", "centroid_y": "fixation_y",
            "mapped_tile": "base_position",
            "nested_context_segments": "context_segments",
        },
        "timestamp_unit": "milliseconds",
        "context_split_reconstruction": (
            "source fixation.csv already stores one I-VT physical fixation per row; "
            "context_segments remain nested within that fixation"
        ),
    }
    return fixations, provenance


def construct_transitions(
    fixations: Sequence[Mapping[str, Any]], *, max_shift_ms: float
) -> list[dict[str, Any]]:
    if not math.isfinite(max_shift_ms) or max_shift_ms <= 0.0:
        raise ValueError("max_shift_ms must be finite and positive")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    fixation_keys = set()
    for row in fixations:
        key = (str(row["participant"]), str(row["trial"]))
        fixation_key = (*key, int(row["fixation_id"]))
        if fixation_key in fixation_keys:
            raise RuntimeError(f"duplicate physical fixation ID: {fixation_key}")
        fixation_keys.add(fixation_key)
        grouped.setdefault(key, []).append(row)
    geometry = ReferenceGeometry()
    transitions = []
    for (participant, trial), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: (float(row["start_ms"]), int(row["fixation_id"])))
        if any(int(right["fixation_id"]) <= int(left["fixation_id"])
               for left, right in zip(ordered, ordered[1:])):
            raise RuntimeError(f"non-increasing fixation IDs in {participant}/{trial}")
        for source, destination in zip(ordered, ordered[1:]):
            shift = float(destination["start_ms"]) - float(source["end_ms"])
            source_tile, destination_tile = source.get("tile"), destination.get("tile")
            distance = (
                None if source_tile is None or destination_tile is None
                else geometry.visual_angle_degrees(source_tile, destination_tile)
            )
            if distance is None:
                reason = "missing_or_outside_canonical_tile"
            elif shift <= 0.0:
                reason = "nonpositive_or_overlapping_interval"
            elif shift > max_shift_ms:
                reason = "gap_exceeds_execution_window"
            else:
                reason = ""
            transitions.append({
                "participant": participant, "trial": trial,
                "source_fixation_id": int(source["fixation_id"]),
                "destination_fixation_id": int(destination["fixation_id"]),
                "source_x": float(source["centroid_x"]),
                "source_y": float(source["centroid_y"]),
                "destination_x": float(destination["centroid_x"]),
                "destination_y": float(destination["centroid_y"]),
                "source_tile": "" if source_tile is None else json.dumps(source_tile),
                "destination_tile": "" if destination_tile is None else json.dumps(destination_tile),
                "visual_angle_deg": "" if distance is None else float(distance),
                "observed_shift_ms": shift,
                "source_end_ms": float(source["end_ms"]),
                "destination_start_ms": float(destination["start_ms"]),
                "time_ms": int(source["time_ms"]),
                "horizon": int(source["horizon"]),
                "layout": source["layout"],
                "source_context_segment_count": int(source["context_segment_count"]),
                "destination_context_segment_count": int(destination["context_segment_count"]),
                "primary_valid": int(not reason),
                "exclusion_reason": reason,
            })
    return transitions


def fit_linear(transitions: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if len(transitions) < 2:
        raise ValueError("at least two transitions are required")
    x = np.asarray([float(row["visual_angle_deg"]) for row in transitions])
    y = np.asarray([float(row["observed_shift_ms"]) for row in transitions])
    design = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ np.asarray((intercept, slope))
    residual = y - prediction
    total = float(np.square(y - y.mean()).sum())
    return {
        "execution_intercept_ms": float(intercept),
        "execution_slope_ms_per_degree": float(slope),
        "n_transitions": len(transitions),
        "n_participants": len({str(row["participant"]) for row in transitions}),
        "r_squared": 0.0 if total <= 0.0 else float(1.0 - np.square(residual).sum() / total),
        "rmse_ms": float(np.sqrt(np.mean(np.square(residual)))),
        "mae_ms": float(np.mean(np.abs(residual))),
        "residual_mean_ms": float(residual.mean()),
        "residual_sd_ms": float(residual.std()),
    }


def participant_cluster_bootstrap(
    transitions: Sequence[Mapping[str, Any]], *, iterations: int, seed: int
) -> list[dict[str, float | int]]:
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in transitions:
        grouped.setdefault(str(row["participant"]), []).append(row)
    participants = sorted(grouped)
    if len(participants) < 2:
        raise ValueError("participant-cluster bootstrap requires multiple participants")
    sufficient = []
    for participant in participants:
        x = np.asarray([float(row["visual_angle_deg"]) for row in grouped[participant]])
        y = np.asarray([float(row["observed_shift_ms"]) for row in grouped[participant]])
        sufficient.append((len(x), x.sum(), y.sum(), np.square(x).sum(), np.dot(x, y)))
    stats = np.asarray(sufficient, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    output = []
    for index in range(int(iterations)):
        selected = rng.integers(0, len(participants), len(participants))
        n, sum_x, sum_y, sum_xx, sum_xy = stats[selected].sum(axis=0)
        denominator = n * sum_xx - sum_x * sum_x
        slope = float("nan") if denominator == 0.0 else (n * sum_xy - sum_x * sum_y) / denominator
        intercept = float("nan") if not np.isfinite(slope) else (sum_y - slope * sum_x) / n
        output.append({
            "bootstrap_iteration": index,
            "execution_intercept_ms": float(intercept),
            "execution_slope_ms_per_degree": float(slope),
        })
    return output


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "n": len(data), "mean": float(data.mean()), "sd": float(data.std()),
        "median": float(np.median(data)),
        "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
        **{f"p{label}": float(value) for label, value in zip(
            (1, 5, 50, 95, 99), np.percentile(data, (1, 5, 50, 95, 99))
        )},
        "min": float(data.min()), "max": float(data.max()),
    }


def _residual_rows(valid: Sequence[Mapping[str, Any]], fit: Mapping[str, Any]) -> list[dict[str, Any]]:
    intercept = float(fit["execution_intercept_ms"])
    slope = float(fit["execution_slope_ms_per_degree"])
    rows = []
    for dimension in ("time_ms", "horizon", "layout", "participant"):
        values = sorted({str(row[dimension]) for row in valid})
        for value in values:
            selected = [row for row in valid if str(row[dimension]) == value]
            residual = np.asarray([
                float(row["observed_shift_ms"]) - intercept
                - slope * float(row["visual_angle_deg"]) for row in selected
            ])
            rows.append({
                "dimension": dimension, "level": value, "n": len(selected),
                "residual_mean_ms": float(residual.mean()),
                "mae_ms": float(np.mean(np.abs(residual))),
                "rmse_ms": float(np.sqrt(np.mean(np.square(residual)))),
            })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, gzip_output: bool = False) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output: {path}")
    fields = list(rows[0])
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, valid, fit, bootstrap) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.asarray([float(row["visual_angle_deg"]) for row in valid])
    y = np.asarray([float(row["observed_shift_ms"]) for row in valid])
    edges = np.linspace(x.min(), x.max(), 16)
    centers, means = [], []
    for left, right in zip(edges, edges[1:]):
        selected = y[(x >= left) & (x <= right if right == edges[-1] else x < right)]
        if selected.size:
            centers.append((left + right) / 2.0)
            means.append(selected.mean())
    grid = np.linspace(x.min(), x.max(), 200)
    boot_a = np.asarray([row["execution_intercept_ms"] for row in bootstrap])
    boot_b = np.asarray([row["execution_slope_ms_per_degree"] for row in bootstrap])
    prediction = boot_a[:, None] + boot_b[:, None] * grid[None, :]
    low, high = np.percentile(prediction, (2.5, 97.5), axis=0)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(centers, means, label="Human binned mean", color="black", s=25)
    axis.plot(grid, EXEC_BASE_MS + EXEC_MS_PER_DEGREE * grid, label="Current/default")
    axis.plot(grid, fit["execution_intercept_ms"] + fit["execution_slope_ms_per_degree"] * grid, label="Fitted")
    axis.fill_between(grid, low, high, alpha=0.2, label="Participant-bootstrap 95% CI")
    axis.set(xlabel="Visual angle (degree)", ylabel="Inter-fixation shift (ms)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_fit(*, input_root: Path, output_dir: Path, bootstrap_iterations: int,
            seed: int, max_shift_ms: float) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixations, provenance = load_human_fixations(input_root)
    transitions = construct_transitions(fixations, max_shift_ms=max_shift_ms)
    valid = [row for row in transitions if row["primary_valid"]]
    fit = fit_linear(valid)
    bootstrap = participant_cluster_bootstrap(valid, iterations=bootstrap_iterations, seed=seed)
    intercepts = np.asarray([row["execution_intercept_ms"] for row in bootstrap])
    slopes = np.asarray([row["execution_slope_ms_per_degree"] for row in bootstrap])
    x = np.asarray([float(row["visual_angle_deg"]) for row in valid])
    representative = np.percentile(x, (0, 25, 50, 75, 100))
    predicted = []
    for distance in representative:
        values = intercepts + slopes * distance
        predicted.append({
            "visual_angle_deg": float(distance),
            "fitted_exec_ms": float(fit["execution_intercept_ms"] + fit["execution_slope_ms_per_degree"] * distance),
            "ci_low_ms": float(np.percentile(values, 2.5)),
            "ci_high_ms": float(np.percentile(values, 97.5)),
        })
    exclusions: dict[str, int] = {}
    for row in transitions:
        exclusions[row["exclusion_reason"] or "primary_valid"] = exclusions.get(row["exclusion_reason"] or "primary_valid", 0) + 1
    all_shifts = np.asarray([float(row["observed_shift_ms"]) for row in transitions])
    timing_audit = {
        **_distribution(all_shifts),
        "candidate_transitions": len(transitions),
        "zero_ms": int(np.count_nonzero(all_shifts == 0.0)),
        "negative_ms": int(np.count_nonzero(all_shifts < 0.0)),
        "overlapping_intervals": int(np.count_nonzero(all_shifts < 0.0)),
        "over_250_ms": int(np.count_nonzero(all_shifts > 250.0)),
        "over_1000_ms": int(np.count_nonzero(all_shifts > 1000.0)),
        "quantized_shift_peaks_ms": [
            {"shift_ms": float(value), "count": int(count)}
            for value, count in Counter(all_shifts.tolist()).most_common(12)
        ],
        "exclusion_counts": exclusions,
    }
    runtime_compatible = (
        fit["execution_intercept_ms"] >= 0.0
        and fit["execution_slope_ms_per_degree"] >= 0.0
    )
    summary = {
        "analysis": "human_calibrated_distance_dependent_gaze_shift_execution_v1",
        "source_provenance": provenance,
        "preprocessing": {
            "physical_fixations": "I-VT physical fixation rows",
            "median_filter_samples": 3, "velocity_threshold_deg_s": 30.0,
            "merge_gap_ms": 75.0, "merge_angle_deg": 0.5,
            "minimum_fixation_duration_ms": 60.0,
        },
        "primary_validity_rule": {
            "same_participant_trial_consecutive_physical_fixations": True,
            "canonical_tiles_required": True,
            "minimum_shift_ms_exclusive": 0.0,
            "maximum_shift_ms_inclusive": max_shift_ms,
        },
        "timing_audit": timing_audit,
        "fit": fit,
        "bootstrap": {
            "unit": "participant", "iterations": bootstrap_iterations, "seed": seed,
            "execution_intercept_95_ci_ms": np.percentile(intercepts, (2.5, 97.5)).tolist(),
            "execution_slope_95_ci_ms_per_degree": np.percentile(slopes, (2.5, 97.5)).tolist(),
            "predicted_execution": predicted,
        },
        "default_timing": {
            "prep_ms": PREP_MS, "execution_intercept_ms": EXEC_BASE_MS,
            "execution_slope_ms_per_degree": EXEC_MS_PER_DEGREE,
            "dwell_ms": DWELL_MS,
        },
        "runtime_nonnegative_compatible": runtime_compatible,
        "limitations": [
            "Inter-fixation timing does not identify internal motor preparation.",
            "Human fixation duration is not model perceptual dwell.",
            "The 60 Hz tracker quantizes timing at approximately 16.7 ms.",
            "The fit calibrates timing only and uses no gaze allocation, condition trend, or task outcome.",
        ],
    }
    participant_rows = []
    for participant in sorted({row["participant"] for row in transitions}):
        candidate = [row for row in transitions if row["participant"] == participant]
        selected = [row for row in candidate if row["primary_valid"]]
        participant_rows.append({
            "participant": participant, "candidate_transitions": len(candidate),
            "valid_transitions": len(selected),
            **{
                f"shift_{key}": value
                for key, value in _distribution([
                    row["observed_shift_ms"] for row in selected
                ]).items()
            },
        })
    fixation_duration_rows = [{"scope": "overall", "level": "all", **_distribution([row["duration_ms"] for row in fixations])}]
    for dimension in ("participant", "time_ms", "horizon"):
        for level in sorted({str(row[dimension]) for row in fixations}):
            selected = [row["duration_ms"] for row in fixations if str(row[dimension]) == level]
            fixation_duration_rows.append({"scope": dimension, "level": level, **_distribution(selected)})
    grid = np.linspace(float(x.min()), float(x.max()), 101)
    current_rows = [{
        "visual_angle_deg": float(distance),
        "default_exec_ms": float(EXEC_BASE_MS + EXEC_MS_PER_DEGREE * distance),
        "fitted_exec_ms": float(fit["execution_intercept_ms"] + fit["execution_slope_ms_per_degree"] * distance),
        "difference_ms": float(
            fit["execution_intercept_ms"] + fit["execution_slope_ms_per_degree"] * distance
            - EXEC_BASE_MS - EXEC_MS_PER_DEGREE * distance
        ),
    } for distance in grid]
    _write_csv(output / "transitions.csv.gz", transitions, gzip_output=True)
    _write_csv(output / "bootstrap_coefficients.csv.gz", bootstrap, gzip_output=True)
    _write_csv(output / "participant_descriptives.csv", participant_rows)
    condition_rows = _residual_rows(valid, fit)
    _write_csv(output / "condition_diagnostics.csv", condition_rows)
    _write_csv(output / "current_vs_fitted.csv", current_rows)
    _write_csv(output / "fixation_duration_summary.csv", fixation_duration_rows)
    _write_plot(output / "emma_fit.png", valid, fit, bootstrap)
    cli = None if not runtime_compatible else (
        f"--emma-mode tuned \\\n--emma-prep-ms {PREP_MS:.12g} \\\n"
        f"--emma-exec-intercept-ms {fit['execution_intercept_ms']:.12g} \\\n"
        f"--emma-exec-slope-ms-per-degree {fit['execution_slope_ms_per_degree']:.12g} \\\n"
        f"--emma-dwell-ms {DWELL_MS:.12g}"
    )
    summary["tuned_cli_fragment"] = cli
    (output / "fit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Human-calibrated distance-dependent gaze-shift execution timing", "",
        f"- Source: `{provenance['input_root']}` ({provenance['participant_count']} participants, {provenance['trial_count']} trials)",
        f"- Physical fixation reconstruction: one `ivt_fixation` row per physical fixation; nested `context_segments` were not treated as shifts ({provenance['fixations_with_multiple_context_segments']} rows had multiple segments).",
        f"- Excluded incomplete trials: {provenance['excluded_incomplete_trial_count']}.",
        f"- Candidate/valid transitions: {len(transitions)} / {len(valid)}",
        f"- Primary rule: consecutive physical fixations, mapped canonical tiles, 0 < shift <= {max_shift_ms:g} ms; the upper bound removes tracker/interruption gaps from the execution target only.",
        f"- Dominant 60-Hz-quantized shift peaks: {timing_audit['quantized_shift_peaks_ms'][:6]}.",
        f"- Current EXEC: `{EXEC_BASE_MS:g} + {EXEC_MS_PER_DEGREE:g} * visual_angle_deg` ms.",
        f"- Fitted EXEC: `{fit['execution_intercept_ms']:.6g} + {fit['execution_slope_ms_per_degree']:.6g} * visual_angle_deg` ms.",
        f"- Intercept 95% CI: {summary['bootstrap']['execution_intercept_95_ci_ms']}",
        f"- Slope 95% CI: {summary['bootstrap']['execution_slope_95_ci_ms_per_degree']}",
        f"- R2={fit['r_squared']:.6g}, RMSE={fit['rmse_ms']:.6g} ms, MAE={fit['mae_ms']:.6g} ms.", "",
        "## Fixation-duration diagnostic", "",
        f"Physical fixations: N={fixation_duration_rows[0]['n']}, mean={fixation_duration_rows[0]['mean']:.6g} ms, "
        f"median={fixation_duration_rows[0]['median']:.6g} ms, SD={fixation_duration_rows[0]['sd']:.6g} ms, "
        f"IQR={fixation_duration_rows[0]['iqr']:.6g} ms. These values do not set model DWELL.", "",
        "## Condition residual diagnostics", "",
        "The one pooled fit is retained. Time/Horizon/layout/participant residual N, mean, MAE, and RMSE are saved in `condition_diagnostics.csv`; no condition-specific coefficient was fitted.", "",
        "## Limitations", "",
        "The 60 Hz sampling grid limits precise saccade-duration estimation. Physical fixation duration is diagnostic only and does not set PREP or DWELL. The pooled fit uses no Time/Horizon effects, gaze allocation, behavioral trend, or task outcome.", "",
        "## Exact tuned CLI", "", "```text", cli or "UNAVAILABLE: unconstrained fit violates nonnegative runtime timing constraints.", "```", "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2] / "data" / "raw" / "experiment"
    value.add_argument("--input-root", type=Path, default=default_root)
    value.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    value.add_argument("--seed", type=int, default=DEFAULT_SEED)
    value.add_argument("--max-shift-ms", type=float, default=DEFAULT_MAX_SHIFT_MS)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary = run_fit(
        input_root=args.input_root, output_dir=args.output_dir,
        bootstrap_iterations=args.bootstrap, seed=args.seed,
        max_shift_ms=args.max_shift_ms,
    )
    fit, boot = summary["fit"], summary["bootstrap"]
    print("=" * 60)
    print("EMMA HUMAN TIMING FIT")
    print("=" * 60)
    print(f"participants                    : {fit['n_participants']}")
    print(f"valid transitions               : {fit['n_transitions']}")
    print(f"execution intercept             : {fit['execution_intercept_ms']:.12g} ms")
    print(f"95% CI                          : {boot['execution_intercept_95_ci_ms']}")
    print(f"execution slope                 : {fit['execution_slope_ms_per_degree']:.12g} ms/degree")
    print(f"95% CI                          : {boot['execution_slope_95_ci_ms_per_degree']}")
    print(f"R2                              : {fit['r_squared']:.12g}")
    print(f"RMSE                            : {fit['rmse_ms']:.12g}")
    print(f"MAE                             : {fit['mae_ms']:.12g}")
    print("=" * 60)
    if summary["tuned_cli_fragment"] is None:
        print("Fit is not directly compatible with nonnegative runtime constraints; scientific review required.")
    else:
        print(summary["tuned_cli_fragment"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
