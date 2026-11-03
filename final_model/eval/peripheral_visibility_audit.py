#!/usr/bin/env python3
"""Geometry-only audit of prospective peripheral tile visibility.

This module deliberately does not load policies, checkpoints, environments, or
human data.  It describes information availability implied by the canonical
board geometry and a caller-supplied Gaussian eccentricity curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from final_model.partial_ver.gaze import (
    EYE_PROJECTION_VIEWPORT_PX,
    MONITOR_ASPECT,
    MONITOR_DIAGONAL_MM,
    MONITOR_RASTER_PX,
    VIEWING_DISTANCE_MM,
    VIEWPORT_TILE_PX,
    ReferenceGeometry,
    acquisition_tiles,
)
from final_model.partial_ver.peripheral import (
    peripheral_detection_probability as detection_probability,
    p1_peripheral_candidates as p1_candidates,
    p2_peripheral_candidates as p2_candidates,
)
from final_model.server_parallel.gaze_v1 import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GAZE_ACTION_COUNT,
    GAZE_TILES,
    action_index_to_tile,
    tile_to_action_index,
)


DEFAULT_SIGMAS_DEG = (2.0, 3.0, 4.0, 5.0, 6.0)
PERCENTILES = (0, 5, 25, 50, 75, 95, 100)
REPRESENTATIVE_ECCENTRICITIES_DEG = (
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0
)
def normalize_sigmas(sigmas_deg: Iterable[float]) -> tuple[float, ...]:
    sigmas = tuple(sorted(float(value) for value in sigmas_deg))
    if not sigmas:
        raise ValueError("at least one sigma is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in sigmas):
        raise ValueError("sigmas must be finite and positive")
    if len(set(sigmas)) != len(sigmas):
        raise ValueError("sigmas must be unique")
    return sigmas


def validate_probability_curve(
    eccentricities_deg: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> None:
    eccentricities = np.asarray(eccentricities_deg, dtype=np.float64)
    values = np.asarray(probabilities, dtype=np.float64)
    if eccentricities.shape != values.shape:
        raise ValueError("eccentricity and probability shapes differ")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise RuntimeError("probability must be finite and in [0, 1]")
    order = np.argsort(eccentricities, kind="stable")
    if np.any(np.diff(values[order]) > 1e-12):
        raise RuntimeError("probability must not increase with eccentricity")


def plus_footprint(fixation: Sequence[int]) -> tuple[tuple[int, int], ...]:
    return acquisition_tiles(fixation, "plus")


def eccentricity_matrix(
    geometry: ReferenceGeometry | None = None,
) -> np.ndarray:
    geometry = ReferenceGeometry() if geometry is None else geometry
    matrix = np.asarray(
        [
            [geometry.visual_angle_degrees(source, target) for target in GAZE_TILES]
            for source in GAZE_TILES
        ],
        dtype=np.float64,
    )
    # Identical rays have exactly zero eccentricity by definition.  acos can
    # otherwise leave a sub-microdegree round-off value for some normalized rays.
    np.fill_diagonal(matrix, 0.0)
    if matrix.shape != (GAZE_ACTION_COUNT, GAZE_ACTION_COUNT):
        raise RuntimeError("canonical gaze target count is not 40")
    return matrix


def expected_visibility(probabilities: Sequence[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return {
        "expected_successes": float(values.sum()),
        "probability_any_success": (
            0.0 if values.size == 0 else float(1.0 - np.prod(1.0 - values))
        ),
        "expected_fraction_detected": (
            0.0 if values.size == 0 else float(values.mean())
        ),
    }


def _percentiles(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"p{percentile}": float(np.percentile(array, percentile))
        for percentile in PERCENTILES
    }


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        **_percentiles(array),
    }


def _fixation_group(tile: tuple[int, int]) -> str:
    x, y = tile
    x_boundary = x in (0, BOARD_WIDTH - 1)
    y_boundary = y in (0, BOARD_HEIGHT - 1)
    if x_boundary and y_boundary:
        return "corner"
    if x_boundary or y_boundary:
        return "edge"
    center_x = (BOARD_WIDTH - 1) / 2.0
    center_y = (BOARD_HEIGHT - 1) / 2.0
    minimum = min(
        (other[0] - center_x) ** 2 + (other[1] - center_y) ** 2
        for other in GAZE_TILES
    )
    distance = (x - center_x) ** 2 + (y - center_y) ** 2
    return "center-ish" if math.isclose(distance, minimum) else "interior"


def _representative_fixations() -> dict[str, tuple[int, int]]:
    return {
        group: next(tile for tile in GAZE_TILES if _fixation_group(tile) == group)
        for group in ("center-ish", "corner", "edge")
    }


def build_audit(sigmas_deg: Iterable[float]) -> dict[str, object]:
    sigmas = normalize_sigmas(sigmas_deg)
    geometry = ReferenceGeometry()
    matrix = eccentricity_matrix(geometry)
    plus_by_fixation = {
        tile: plus_footprint(tile) for tile in GAZE_TILES
    }
    p1_by_fixation = {tile: p1_candidates(tile) for tile in GAZE_TILES}
    p2_by_fixation = {tile: p2_candidates(tile) for tile in GAZE_TILES}

    nonzero = matrix[matrix > 0.0]
    orthogonal = []
    diagonal = []
    for x, y in GAZE_TILES:
        source_index = tile_to_action_index((x, y))
        for dx, dy, bucket in ((1, 0, orthogonal), (0, 1, orthogonal),
                               (1, 1, diagonal), (1, -1, diagonal)):
            target = (x + dx, y + dy)
            if target in GAZE_TILES:
                bucket.append(matrix[source_index, tile_to_action_index(target)])

    p1_candidate_eccentricities = np.asarray([
        matrix[tile_to_action_index(fixation), tile_to_action_index(candidate)]
        for fixation in GAZE_TILES
        for candidate in p1_by_fixation[fixation]
    ])
    p2_candidate_eccentricities = np.asarray([
        matrix[tile_to_action_index(fixation), tile_to_action_index(candidate)]
        for fixation in GAZE_TILES
        for candidate in p2_by_fixation[fixation]
    ])

    per_fixation_rows = []
    for mode, candidates_by_fixation in (
        ("P1", p1_by_fixation), ("P2", p2_by_fixation)
    ):
        for sigma in sigmas:
            for fixation in GAZE_TILES:
                fixation_index = tile_to_action_index(fixation)
                candidates = candidates_by_fixation[fixation]
                distances = np.asarray([
                    matrix[fixation_index, tile_to_action_index(candidate)]
                    for candidate in candidates
                ])
                probabilities = detection_probability(distances, sigma)
                validate_probability_curve(distances, probabilities)
                metrics = expected_visibility(probabilities)
                per_fixation_rows.append({
                    "mode": mode,
                    "sigma_deg": sigma,
                    "fixation_index": fixation_index,
                    "fixation_x": fixation[0],
                    "fixation_y": fixation[1],
                    "fixation_group": _fixation_group(fixation),
                    "plus_count": len(plus_by_fixation[fixation]),
                    "candidate_count": len(candidates),
                    **metrics,
                    "expected_total_visible_tiles": (
                        len(plus_by_fixation[fixation])
                        + metrics["expected_successes"]
                    ),
                })

    p1_summary_rows = []
    p2_summary_rows = []
    for sigma in sigmas:
        p1_rows = [
            row for row in per_fixation_rows
            if row["mode"] == "P1" and row["sigma_deg"] == sigma
        ]
        p2_rows = [
            row for row in per_fixation_rows
            if row["mode"] == "P2" and row["sigma_deg"] == sigma
        ]
        diagonal_probabilities = detection_probability(
            p1_candidate_eccentricities, sigma
        )
        p1_successes = np.asarray([row["expected_successes"] for row in p1_rows])
        p1_summary_rows.append({
            "sigma_deg": sigma,
            "mean_diagonal_eccentricity_deg": float(
                p1_candidate_eccentricities.mean()
            ),
            "mean_p_diagonal": float(diagonal_probabilities.mean()),
            "min_p_diagonal": float(diagonal_probabilities.min()),
            "max_p_diagonal": float(diagonal_probabilities.max()),
            "mean_expected_peripheral_successes": float(p1_successes.mean()),
            "median_expected_peripheral_successes": float(np.median(p1_successes)),
            "min_expected_peripheral_successes": float(p1_successes.min()),
            "max_expected_peripheral_successes": float(p1_successes.max()),
            "mean_probability_any_success": float(np.mean([
                row["probability_any_success"] for row in p1_rows
            ])),
        })

        p2_successes = np.asarray([row["expected_successes"] for row in p2_rows])
        p2_summary_rows.append({
            "sigma_deg": sigma,
            "mean_candidate_count": float(np.mean([
                row["candidate_count"] for row in p2_rows
            ])),
            "mean_expected_peripheral_successes": float(p2_successes.mean()),
            "median_expected_peripheral_successes": float(np.median(p2_successes)),
            "p25_expected_peripheral_successes": float(
                np.percentile(p2_successes, 25)
            ),
            "p75_expected_peripheral_successes": float(
                np.percentile(p2_successes, 75)
            ),
            "min_expected_peripheral_successes": float(p2_successes.min()),
            "max_expected_peripheral_successes": float(p2_successes.max()),
            "mean_expected_fraction_detected": float(np.mean([
                row["expected_fraction_detected"] for row in p2_rows
            ])),
            "mean_probability_any_success": float(np.mean([
                row["probability_any_success"] for row in p2_rows
            ])),
            "mean_total_visible_tiles": float(np.mean([
                row["expected_total_visible_tiles"] for row in p2_rows
            ])),
            **{
                f"mean_expected_successes_{group.replace('-', '_')}": float(np.mean([
                    row["expected_successes"] for row in p2_rows
                    if row["fixation_group"] == group
                ]))
                for group in ("center-ish", "corner", "edge")
            },
        })

    distance_band_rows = []
    maximum_eccentricity = float(matrix.max())
    for sigma in sigmas:
        band_successes = [[], [], [], []]
        band_counts = [[], [], [], []]
        far_probabilities = []
        for fixation in GAZE_TILES:
            fixation_index = tile_to_action_index(fixation)
            candidates = p2_by_fixation[fixation]
            ordered = sorted(
                (
                    matrix[fixation_index, tile_to_action_index(candidate)],
                    tile_to_action_index(candidate),
                )
                for candidate in candidates
            )
            for band_index, band in enumerate(np.array_split(ordered, 4)):
                distances = np.asarray(band[:, 0], dtype=np.float64)
                probabilities = detection_probability(distances, sigma)
                band_successes[band_index].append(float(probabilities.sum()))
                band_counts[band_index].append(len(distances))
                if band_index == 3:
                    far_probabilities.extend(probabilities.tolist())
        distance_band_rows.append({
            "sigma_deg": sigma,
            "maximum_observed_eccentricity_deg": maximum_eccentricity,
            "p_at_maximum_eccentricity": detection_probability(
                maximum_eccentricity, sigma
            ),
            "mean_p_farthest_quartile": float(np.mean(far_probabilities)),
            **{
                f"mean_expected_successes_q{index + 1}": float(
                    np.mean(band_successes[index])
                )
                for index in range(4)
            },
            **{
                f"mean_candidate_count_q{index + 1}": float(
                    np.mean(band_counts[index])
                )
                for index in range(4)
            },
        })

    representative_eccentricities = list(REPRESENTATIVE_ECCENTRICITIES_DEG)
    if not any(
        math.isclose(value, maximum_eccentricity)
        for value in representative_eccentricities
    ):
        representative_eccentricities.append(maximum_eccentricity)
    probability_rows = []
    for sigma in sigmas:
        probabilities = detection_probability(representative_eccentricities, sigma)
        validate_probability_curve(representative_eccentricities, probabilities)
        probability_rows.extend({
            "sigma_deg": sigma,
            "eccentricity_deg": eccentricity,
            "probability": probability,
            "is_maximum_observed": math.isclose(
                eccentricity, maximum_eccentricity
            ),
        } for eccentricity, probability in zip(
            representative_eccentricities, probabilities
        ))

    return {
        "sigmas_deg": sigmas,
        "matrix": matrix,
        "plus_by_fixation": plus_by_fixation,
        "p1_by_fixation": p1_by_fixation,
        "p2_by_fixation": p2_by_fixation,
        "per_fixation_rows": per_fixation_rows,
        "probability_rows": probability_rows,
        "p1_summary_rows": p1_summary_rows,
        "p2_summary_rows": p2_summary_rows,
        "distance_band_rows": distance_band_rows,
        "geometry_summary": {
            "nonzero_pairwise_eccentricity_deg": _summary(nonzero),
            "orthogonally_adjacent_eccentricity_deg": _summary(orthogonal),
            "diagonally_adjacent_eccentricity_deg": _summary(diagonal),
            "p1_candidate_count": _summary([
                len(value) for value in p1_by_fixation.values()
            ]),
            "p1_candidate_eccentricity_deg": _summary(
                p1_candidate_eccentricities
            ),
            "p2_candidate_count": _summary([
                len(value) for value in p2_by_fixation.values()
            ]),
            "p2_candidate_eccentricity_deg": _summary(
                p2_candidate_eccentricities
            ),
        },
        "representative_fixations": _representative_fixations(),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_eccentricity_matrix(path: Path, matrix: np.ndarray) -> None:
    target_columns = [
        f"target_{index:02d}_x{tile[0]}_y{tile[1]}"
        for index, tile in enumerate(GAZE_TILES)
    ]
    rows = []
    for index, tile in enumerate(GAZE_TILES):
        rows.append({
            "fixation_index": index,
            "fixation_x": tile[0],
            "fixation_y": tile[1],
            **{
                column: float(matrix[index, target_index])
                for target_index, column in enumerate(target_columns)
            },
        })
    _write_csv(path, rows)


def _geometry_summary_rows(
    geometry_summary: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    return [
        {"distribution": name, **statistics}
        for name, statistics in geometry_summary.items()
    ]


def _candidate_rows(
    candidates_by_fixation: Mapping[
        tuple[int, int], Sequence[tuple[int, int]]
    ],
    matrix: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for fixation in GAZE_TILES:
        fixation_index = tile_to_action_index(fixation)
        candidates = candidates_by_fixation[fixation]
        candidate_indices = [tile_to_action_index(tile) for tile in candidates]
        for rank, (candidate, candidate_index) in enumerate(
            zip(candidates, candidate_indices), start=1
        ):
            rows.append({
                "fixation_index": fixation_index,
                "fixation_x": fixation[0],
                "fixation_y": fixation[1],
                "candidate_count": len(candidates),
                "candidate_indices": json.dumps(candidate_indices, separators=(",", ":")),
                "candidate_rank": rank,
                "candidate_index": candidate_index,
                "candidate_x": candidate[0],
                "candidate_y": candidate[1],
                "eccentricity_deg": float(matrix[fixation_index, candidate_index]),
            })
    return rows


def _plus_rows(
    plus_by_fixation: Mapping[tuple[int, int], Sequence[tuple[int, int]]]
) -> list[dict[str, object]]:
    rows = []
    for fixation in GAZE_TILES:
        tiles = plus_by_fixation[fixation]
        indices = [tile_to_action_index(tile) for tile in tiles]
        rows.append({
            "fixation_index": tile_to_action_index(fixation),
            "fixation_x": fixation[0],
            "fixation_y": fixation[1],
            "plus_count": len(tiles),
            "plus_indices": json.dumps(indices, separators=(",", ":")),
            "plus_tiles": json.dumps(tiles, separators=(",", ":")),
        })
    return rows


def _sigma_label(sigma: float) -> str:
    return format(sigma, ".12g").replace("-", "m").replace(".", "p")


def _save_figures(output_dir: Path, audit: Mapping[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = audit["sigmas_deg"]
    matrix = audit["matrix"]
    maximum = float(np.max(matrix))
    x_values = np.linspace(0.0, maximum, 400)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for sigma in sigmas:
        axis.plot(
            x_values, detection_probability(x_values, sigma),
            label=f"sigma={sigma:g} deg",
        )
    axis.set(xlabel="Eccentricity (deg)", ylabel="Detection probability",
             xlim=(0.0, maximum), ylim=(0.0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "probability_curves.png", dpi=160)
    plt.close(figure)

    p2_rows = audit["per_fixation_rows"]
    distributions = [
        [
            row["expected_successes"] for row in p2_rows
            if row["mode"] == "P2" and row["sigma_deg"] == sigma
        ]
        for sigma in sigmas
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(distributions, labels=[f"{sigma:g}" for sigma in sigmas])
    for position, values in enumerate(distributions, start=1):
        axis.scatter(
            np.full(len(values), position), values, s=10, alpha=0.45, color="tab:blue"
        )
    axis.set(
        xlabel="Sigma (deg)",
        ylabel="Expected P2 peripheral successes / acquisition",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "p2_expected_success.png", dpi=160)
    plt.close(figure)

    representatives = audit["representative_fixations"]
    plus_by_fixation = audit["plus_by_fixation"]
    for sigma in sigmas:
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        image = None
        for axis, (group, fixation) in zip(axes, representatives.items()):
            fixation_index = tile_to_action_index(fixation)
            values = detection_probability(matrix[fixation_index], sigma).reshape(
                BOARD_WIDTH, BOARD_HEIGHT
            )
            peripheral = np.array(values, copy=True)
            plus = plus_by_fixation[fixation]
            for tile in plus:
                peripheral[tile] = np.nan
            image = axis.imshow(
                peripheral.T, origin="lower", vmin=0.0, vmax=1.0,
                cmap="viridis", extent=(-0.5, BOARD_WIDTH - 0.5,
                                        -0.5, BOARD_HEIGHT - 0.5),
            )
            axis.scatter(
                [tile[0] for tile in plus], [tile[1] for tile in plus],
                marker="+", s=100, linewidths=2.0, color="red", label="PLUS exact",
            )
            axis.set(
                title=f"{group}: idx {fixation_index}, tile {fixation}",
                xlabel="tile x", ylabel="tile y",
                xticks=range(BOARD_WIDTH), yticks=range(BOARD_HEIGHT),
            )
            axis.legend(loc="upper right", fontsize=7)
        figure.colorbar(image, ax=axes, label="Peripheral p_detect")
        figure.suptitle(f"P2 visibility, sigma={sigma:g} deg")
        figure.savefig(
            output_dir / f"p2_heatmaps_sigma_{_sigma_label(sigma)}.png", dpi=160
        )
        plt.close(figure)


def _markdown_table(
    rows: Sequence[Mapping[str, object]], columns: Sequence[tuple[str, str]]
) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---:" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_summary(path: Path, audit: Mapping[str, object]) -> None:
    geometry = audit["geometry_summary"]
    pairwise = geometry["nonzero_pairwise_eccentricity_deg"]
    orthogonal = geometry["orthogonally_adjacent_eccentricity_deg"]
    diagonal = geometry["diagonally_adjacent_eccentricity_deg"]
    p1_candidates_summary = geometry["p1_candidate_count"]
    p2_candidates_summary = geometry["p2_candidate_count"]
    monitor_width = MONITOR_DIAGONAL_MM * MONITOR_ASPECT[0] / math.hypot(*MONITOR_ASPECT)
    monitor_height = MONITOR_DIAGONAL_MM * MONITOR_ASPECT[1] / math.hypot(*MONITOR_ASPECT)
    representatives = audit["representative_fixations"]

    lines = [
        "# Peripheral visibility geometry audit",
        "",
        "## Canonical geometry and indexing",
        "",
        "- Geometry source: `final_model.partial_ver.gaze.ReferenceGeometry.visual_angle_degrees`.",
        "- Every matrix entry is computed with that helper; mathematically identical "
        "source/target tiles are normalized from floating-point `acos` round-off to "
        "exactly 0 deg.",
        f"- Screen: {MONITOR_RASTER_PX[0]:g}×{MONITOR_RASTER_PX[1]:g} px; "
        f"{MONITOR_DIAGONAL_MM:g} mm diagonal, {MONITOR_ASPECT[0]:g}:{MONITOR_ASPECT[1]:g} "
        f"({monitor_width:.6f}×{monitor_height:.6f} mm).",
        f"- Viewing distance: {VIEWING_DISTANCE_MM:g} mm.",
        f"- Board: {BOARD_WIDTH}×{BOARD_HEIGHT} tiles; each tile spans "
        f"{VIEWPORT_TILE_PX[0]:g}×{VIEWPORT_TILE_PX[1]:g} raster px, so the mapped "
        f"board is {BOARD_WIDTH * VIEWPORT_TILE_PX[0]:g}×"
        f"{BOARD_HEIGHT * VIEWPORT_TILE_PX[1]:g} px.",
        f"- Eye-projection origin: {EYE_PROJECTION_VIEWPORT_PX[0]:g}, "
        f"{EYE_PROJECTION_VIEWPORT_PX[1]:g} px. Tile centers use `(x+0.5, y+0.5)` "
        "times the canonical tile span.",
        "- Gaze indexing is x-major: `index = x*5 + y`; `GAZE_TILES` is the "
        "authoritative 40-tile sequence.",
        "- PLUS is exactly the current acquisition footprint: fixation center plus "
        "valid up/down/left/right tiles, clipped at board boundaries.",
        "- P1 contains only valid diagonal offsets (±1, ±1), disjoint from PLUS.",
        "- P2 contains every canonical board tile outside PLUS.",
        f"- Representative groups are geometry-defined: corners are both-axis "
        f"boundaries; edges are one-axis boundaries excluding corners; center-ish "
        f"tiles minimize distance to board center. Plot representatives: {representatives}.",
        "",
        "## Eccentricity and boundary effects",
        "",
        f"- Nonzero pairwise range: {pairwise['min']:.6f}–{pairwise['max']:.6f} deg.",
        f"- Nonzero pairwise p5/p25/p50/p75/p95 = {pairwise['p5']:.6f} / "
        f"{pairwise['p25']:.6f} / {pairwise['p50']:.6f} / {pairwise['p75']:.6f} / "
        f"{pairwise['p95']:.6f} deg.",
        f"- Orthogonally adjacent: min/mean/median/max = {orthogonal['min']:.6f} / "
        f"{orthogonal['mean']:.6f} / {orthogonal['median']:.6f} / "
        f"{orthogonal['max']:.6f} deg.",
        f"- Orthogonally adjacent p5/p25/p50/p75/p95 = {orthogonal['p5']:.6f} / "
        f"{orthogonal['p25']:.6f} / {orthogonal['p50']:.6f} / "
        f"{orthogonal['p75']:.6f} / {orthogonal['p95']:.6f} deg.",
        f"- Diagonally adjacent: min/mean/median/max = {diagonal['min']:.6f} / "
        f"{diagonal['mean']:.6f} / {diagonal['median']:.6f} / "
        f"{diagonal['max']:.6f} deg.",
        f"- Diagonally adjacent p5/p25/p50/p75/p95 = {diagonal['p5']:.6f} / "
        f"{diagonal['p25']:.6f} / {diagonal['p50']:.6f} / "
        f"{diagonal['p75']:.6f} / {diagonal['p95']:.6f} deg.",
        f"- P1 candidate count min/mean/median/max = {p1_candidates_summary['min']:.0f} / "
        f"{p1_candidates_summary['mean']:.3f} / {p1_candidates_summary['median']:.3f} / "
        f"{p1_candidates_summary['max']:.0f}.",
        f"- P2 candidate count min/mean/max = {p2_candidates_summary['min']:.0f} / "
        f"{p2_candidates_summary['mean']:.3f} / {p2_candidates_summary['max']:.0f}; "
        "variation comes only from PLUS boundary clipping.",
        "- Full percentile details are in `geometry_summary.csv`.",
        "",
        "## Probability by representative eccentricity",
        "",
        *_markdown_table(
            audit["probability_rows"],
            (("sigma_deg", "sigma (deg)"), ("eccentricity_deg", "e (deg)"),
             ("probability", "p_detect")),
        ),
        "",
        "## P1 expected visibility",
        "",
        *_markdown_table(
            audit["p1_summary_rows"],
            (("sigma_deg", "sigma"),
             ("mean_p_diagonal", "mean p(diagonal)"),
             ("mean_expected_peripheral_successes", "mean E[success]"),
             ("median_expected_peripheral_successes", "median E[success]"),
             ("mean_probability_any_success", "mean P(any)")),
        ),
        "",
        "## P2 expected visibility",
        "",
        *_markdown_table(
            audit["p2_summary_rows"],
            (("sigma_deg", "sigma"),
             ("mean_expected_peripheral_successes", "mean E[success]"),
             ("median_expected_peripheral_successes", "median E[success]"),
             ("mean_expected_fraction_detected", "mean fraction"),
             ("mean_probability_any_success", "mean P(any)"),
             ("mean_total_visible_tiles", "mean total visible")),
        ),
        "",
        "## Far-distance diagnostic",
        "",
        "Quartiles are formed independently for each fixation by sorting its P2 "
        "candidates by eccentricity (then target index for ties). Reported expected "
        "successes are the mean quartile sums over all 40 fixations.",
        "",
        *_markdown_table(
            audit["distance_band_rows"],
            (("sigma_deg", "sigma"),
             ("p_at_maximum_eccentricity", "p(max e)"),
             ("mean_p_farthest_quartile", "mean p(farthest Q)"),
             ("mean_expected_successes_q1", "E nearest Q"),
             ("mean_expected_successes_q2", "E second Q"),
             ("mean_expected_successes_q3", "E third Q"),
             ("mean_expected_successes_q4", "E farthest Q")),
        ),
        "",
        "No sigma was selected using human gaze behavior or RL performance.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    sigmas_deg: Iterable[float], output_dir: str | Path, *,
    make_figures: bool = True,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = build_audit(sigmas_deg)
    matrix = audit["matrix"]

    _write_eccentricity_matrix(output / "eccentricity_matrix.csv", matrix)
    _write_csv(
        output / "geometry_summary.csv",
        _geometry_summary_rows(audit["geometry_summary"]),
    )
    _write_csv(output / "plus_footprints.csv", _plus_rows(audit["plus_by_fixation"]))
    _write_csv(
        output / "p1_candidates.csv",
        _candidate_rows(audit["p1_by_fixation"], matrix),
    )
    _write_csv(
        output / "p2_candidates.csv",
        _candidate_rows(audit["p2_by_fixation"], matrix),
    )
    _write_csv(output / "probability_by_eccentricity.csv", audit["probability_rows"])
    _write_csv(output / "p1_sigma_summary.csv", audit["p1_summary_rows"])
    _write_csv(output / "p2_sigma_summary.csv", audit["p2_summary_rows"])
    _write_csv(
        output / "distance_band_expected_success.csv", audit["distance_band_rows"]
    )
    _write_csv(
        output / "per_fixation_sigma_metrics.csv", audit["per_fixation_rows"]
    )
    _write_summary(output / "summary.md", audit)
    if make_figures:
        _save_figures(output, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit peripheral visibility from canonical geometry only."
    )
    parser.add_argument(
        "--sigmas-deg", nargs="+", type=float, default=DEFAULT_SIGMAS_DEG,
        help="positive Gaussian visibility sigmas in degrees",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("final_model/eval/results/peripheral_visibility_audit"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        sigmas = normalize_sigmas(args.sigmas_deg)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    run_audit(sigmas, args.output_dir)
    print(f"Peripheral visibility audit written to {args.output_dir}")


if __name__ == "__main__":
    main()
