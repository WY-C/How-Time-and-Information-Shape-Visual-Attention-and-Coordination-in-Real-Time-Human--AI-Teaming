"""Decision-level AOI coverage in successive 200 ms intervals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    HIT_COLUMNS,
    load_human,
    load_model,
)


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "final_model/eval/results"
OUTPUT = RESULTS / "benchmark_20m_threeway_v8_5seed_20260906/timebin_analysis"
BINS = ((0, 200), (200, 400), (400, 600))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coverage(
    decisions: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    model: str,
    comparison: str,
    footprint: str,
) -> pd.DataFrame:
    decisions = decisions[
        decisions.model.eq(model) & decisions.comparison.eq(comparison)
    ].copy()
    segments = segments[
        segments.model.eq(model) & segments.comparison.eq(comparison)
    ].copy()
    segment_end = segments["segment_start_ms"] + segments["duration_ms"]
    segments["relative_start_ms"] = (
        segments["segment_start_ms"] - segments["decision_start_ms"]
    )
    segments["relative_end_ms"] = segment_end - segments["decision_start_ms"]

    rows = []
    keys = ["decision_uid"]
    for start, end in BINS:
        eligible = decisions[decisions["T"].ge(end)].copy()
        overlapping = segments[
            segments["T"].ge(end)
            & segments.relative_start_ms.lt(end)
            & segments.relative_end_ms.gt(start)
        ].copy()
        coverage = overlapping.groupby(keys, observed=True)[
            list(HIT_COLUMNS.values())
        ].max()
        frame = eligible.merge(coverage, on=keys, how="left")
        for column in HIT_COLUMNS.values():
            frame[column] = frame[column].fillna(0).astype(np.int8)
        observed = set(overlapping.decision_uid)
        frame["observed_segment"] = frame.decision_uid.isin(observed).astype(np.int8)
        frame["bin_start_ms"] = start
        frame["bin_end_ms"] = end

        for (t, h), subset in frame.groupby(["T", "H"], observed=True):
            rows.append(_row(
                subset, model, comparison, footprint, "T_H_cell", int(t), int(h),
                "specific_H", start, end,
            ))
        for t, subset in frame.groupby("T", observed=True):
            rows.append(_row(
                subset, model, comparison, footprint, "T_marginal", int(t), np.nan,
                "all_H", start, end,
            ))
        preview_subset = frame[frame.H.isin([2, 4])]
        for t, subset in preview_subset.groupby("T", observed=True):
            rows.append(_row(
                subset, model, comparison, footprint, "T_marginal", int(t), np.nan,
                "H2_H4", start, end,
            ))
    return pd.DataFrame(rows)


def _row(
    frame: pd.DataFrame,
    model: str,
    comparison: str,
    footprint: str,
    aggregation: str,
    t: int,
    h: int | float,
    h_subset: str,
    start: int,
    end: int,
) -> dict[str, object]:
    # Equal-weight cluster estimates match the existing opportunity analysis and
    # avoid weighting participants by their number of eligible human decisions.
    cluster_means = frame.groupby("cluster", observed=True)[
        [*HIT_COLUMNS.values(), "observed_segment"]
    ].mean()
    result: dict[str, object] = {
        "model": model,
        "comparison": comparison,
        "footprint_shape": footprint,
        "aggregation": aggregation,
        "T": t,
        "H": h,
        "h_subset": h_subset,
        "bin_start_ms": start,
        "bin_end_ms": end,
        "eligible_decisions": len(frame),
        "clusters": frame.cluster.nunique(),
        "observed_segment_fraction": cluster_means.observed_segment.mean(),
    }
    for aoi, column in HIT_COLUMNS.items():
        result[f"{aoi}_decision_coverage"] = cluster_means[column].mean()
    return result


def _profile(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, comparison, footprint), frame in summary.groupby(
        ["model", "comparison", "footprint_shape"], observed=True
    ):
        for aoi in HIT_COLUMNS:
            h_subset = "H2_H4" if aoi == "Preview" else "all_H"
            subset = frame[
                frame.aggregation.eq("T_marginal")
                & frame["T"].eq(600)
                & frame.h_subset.eq(h_subset)
            ].sort_values("bin_start_ms")
            values = dict(zip(
                subset.bin_start_ms.astype(int),
                subset[f"{aoi}_decision_coverage"].astype(float),
            ))
            rows.append(
                {
                    "model": model,
                    "comparison": comparison,
                    "footprint_shape": footprint,
                    "aoi": aoi,
                    "h_subset": h_subset,
                    "coverage_0_200": values[0],
                    "coverage_200_400": values[200],
                    "coverage_400_600": values[400],
                    "late_minus_early": values[400] - values[0],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    human_decisions, human_segments = load_human(
        PROJECT / "data" / "processed",
        RESULTS / "human_benchmark_20260901_contract_v2",
    )
    datasets = [
        (human_decisions, human_segments, "Human", "plus", "plus_5_tile"),
        (human_decisions, human_segments, "Human", "exact3x3", "exact_3x3_9_tile"),
    ]
    for root, model, comparison, footprint in (
        (
            RESULTS / "benchmark_20m_directjoint_pluslocal_v8_5seed_20260906",
            "DirectJoint20M", "plus", "plus_5_tile",
        ),
        (
            RESULTS / "benchmark_20m_staged_pluslocal_v8_5seed_20260906",
            "StagedPlusLocal20M", "plus", "plus_5_tile",
        ),
        (
            RESULTS / "benchmark_20m_staged_exact3x3_v8_5seed_20260906",
            "StagedExact3x3_20M", "exact3x3", "exact_3x3_9_tile",
        ),
    ):
        decisions, segments = load_model(root, label=model, comparison=comparison)
        datasets.append((decisions, segments, model, comparison, footprint))

    summary = pd.concat(
        [
            _coverage(
                decisions, segments, model=model, comparison=comparison,
                footprint=footprint,
            )
            for decisions, segments, model, comparison, footprint in datasets
        ],
        ignore_index=True,
    )
    profile = _profile(summary)
    summary_path = OUTPUT / "within_decision_timebin_coverage.csv"
    profile_path = OUTPUT / "t600_timebin_profile.csv"
    summary.to_csv(summary_path, index=False)
    profile.to_csv(profile_path, index=False)
    manifest = {
        "definition": (
            "Decision-level indicator that an annotated fixation segment overlaps "
            "the 200 ms bin and covers the AOI; estimates average cluster-level rates."
        ),
        "bins_ms": [list(value) for value in BINS],
        "primary_comparison": "within T=600 decisions across 0-200, 200-400, 400-600 ms",
        "preview_subset": [2, 4],
        "overlap_note": "Self, Teammate, and Preview AOIs may overlap; Other is exclusive per segment.",
        "native_footprint_note": (
            "Plus models use plus 5-tile AOIs; exact3x3 uses native 9-tile AOIs."
        ),
        "outputs": {
            summary_path.name: _sha256(summary_path),
            profile_path.name: _sha256(profile_path),
        },
    }
    (OUTPUT / "timebin_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
