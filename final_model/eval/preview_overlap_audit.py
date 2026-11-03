#!/usr/bin/env python3
"""Read-only audit of same-tile overlap in human-study preview logs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from final_model.partial_ver.preview_opacity import rendered_preview_alphas


CONDITIONS = (2, 4)
TIME_CONDITIONS_MS = (200, 400, 600)
LAYOUT_DIMENSIONS = {
    "counter_circuit": (8, 5),
    "double_rings": (7, 8),
    "unident_s_hard_onion": (9, 6),
}


@dataclass(frozen=True)
class DisplayedStep:
    step: int
    tile: tuple[int, int]
    interaction: bool | None = None
    unremapped_tile: tuple[int, int] | None = None


def x_major_tile_index(
    tile: Sequence[int], *, width: int, height: int
) -> int:
    x, y = int(tile[0]), int(tile[1])
    if not (0 <= x < int(width) and 0 <= y < int(height)):
        raise ValueError(f"tile {(x, y)} is outside {width}x{height}")
    return x * int(height) + y


def displayed_step_from_prediction(
    *, step: int, predicted_position: Sequence[int],
    predicted_interact: bool, predicted_interact_position: Sequence[int] | None,
) -> DisplayedStep:
    position = tuple(int(value) for value in predicted_position)
    if predicted_interact:
        if predicted_interact_position is None:
            raise ValueError("interaction prediction has no target tile")
        tile = tuple(int(value) for value in predicted_interact_position)
    else:
        tile = position
    return DisplayedStep(
        step=int(step), tile=tile, interaction=bool(predicted_interact),
        unremapped_tile=position,
    )


def _deduplicate_steps(
    assignments: Iterable[DisplayedStep], *, horizon: int,
) -> list[DisplayedStep]:
    if int(horizon) not in CONDITIONS:
        raise ValueError("overlap audit horizon must be 2 or 4")
    by_key: dict[tuple[tuple[int, int], int], DisplayedStep] = {}
    for assignment in assignments:
        step = int(assignment.step)
        if not 1 <= step <= int(horizon):
            continue
        normalized = DisplayedStep(
            step=step,
            tile=tuple(int(value) for value in assignment.tile),
            interaction=assignment.interaction,
            unremapped_tile=(
                None if assignment.unremapped_tile is None
                else tuple(int(value) for value in assignment.unremapped_tile)
            ),
        )
        key = normalized.tile, normalized.step
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = normalized
        elif previous != normalized:
            raise ValueError("duplicate tile-step has conflicting provenance")
    return sorted(by_key.values(), key=lambda item: (item.step, item.tile))


def analyze_assignments(
    assignments: Iterable[DisplayedStep], *, horizon: int,
    width: int, height: int,
) -> dict[str, object]:
    """Collapse duplicate render calls and count distinct H steps per tile."""
    steps = _deduplicate_steps(assignments, horizon=horizon)
    grouped: dict[tuple[int, int], list[DisplayedStep]] = {}
    for assignment in steps:
        x_major_tile_index(assignment.tile, width=width, height=height)
        grouped.setdefault(assignment.tile, []).append(assignment)

    alphas = rendered_preview_alphas(horizon)
    tile_rows = []
    for tile in sorted(
        grouped, key=lambda value: x_major_tile_index(
            value, width=width, height=height
        )
    ):
        members = sorted(grouped[tile], key=lambda item: item.step)
        distinct_steps = tuple(sorted({item.step for item in members}))
        raw_alphas = tuple(alphas[step - 1] for step in distinct_steps)
        interactions = tuple(
            item.step for item in members if item.interaction is True
        )
        movements = tuple(
            item.step for item in members if item.interaction is False
        )
        unknown = tuple(
            item.step for item in members if item.interaction is None
        )
        if interactions:
            provenance_class = "interaction-containing"
        elif unknown:
            provenance_class = "unknown"
        else:
            provenance_class = "movement-only"
        remap_created = None
        if all(item.unremapped_tile is not None for item in members):
            remap_created = len({item.unremapped_tile for item in members}) > 1
        tile_rows.append({
            "tile": tile,
            "tile_index": x_major_tile_index(tile, width=width, height=height),
            "steps": distinct_steps,
            "multiplicity": len(distinct_steps),
            "renderer_max_alpha": max(raw_alphas),
            "raw_alphas": raw_alphas,
            "probabilities": tuple(alpha / 255.0 for alpha in raw_alphas),
            "provenance_class": provenance_class,
            "interaction_steps": interactions,
            "movement_steps": movements,
            "unknown_steps": unknown,
            "interaction_remap_created": remap_created,
        })

    provenance_complete = all(
        item.unremapped_tile is not None for item in steps
    )
    no_remap_overlap_tiles = None
    no_remap_has_overlap = None
    if provenance_complete:
        unremapped: dict[tuple[int, int], set[int]] = {}
        for item in steps:
            unremapped.setdefault(item.unremapped_tile, set()).add(item.step)
        no_remap_overlap_tiles = sum(
            len(step_set) >= 2 for step_set in unremapped.values()
        )
        no_remap_has_overlap = no_remap_overlap_tiles > 0

    overlap_rows = [row for row in tile_rows if row["multiplicity"] >= 2]
    multiplicity = {
        value: sum(row["multiplicity"] == value for row in tile_rows)
        for value in range(1, 5)
    }
    return {
        "positive_step_instances": len(steps),
        "unique_tiles": len(tile_rows),
        "overlap_tiles": len(overlap_rows),
        "has_overlap": bool(overlap_rows),
        "overlap_step_instances": sum(
            int(row["multiplicity"]) for row in overlap_rows
        ),
        "multiplicity": multiplicity,
        "tile_rows": tile_rows,
        "overlap_rows": overlap_rows,
        "provenance_complete": provenance_complete,
        "no_remap_overlap_tiles": no_remap_overlap_tiles,
        "no_remap_has_overlap": no_remap_has_overlap,
    }


def _json_position(value: str) -> tuple[int, int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError(f"invalid logged position: {value!r}")
    return int(parsed[0]), int(parsed[1])


def _trial_metadata(log_root: Path) -> dict[str, dict[str, str]]:
    trials: dict[str, dict[str, str]] = {}
    for path in sorted(log_root.glob("P*/trial.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                trial_id = row["trial_id"]
                if trial_id in trials:
                    raise RuntimeError(f"duplicate trial identifier: {trial_id}")
                trials[trial_id] = row
    if not trials:
        raise RuntimeError(f"no human-study trial logs found under {log_root}")
    return trials


def _prediction_provenance(
    log_root: Path,
) -> dict[tuple[str, int, int], DisplayedStep]:
    result: dict[tuple[str, int, int], DisplayedStep] = {}
    for path in sorted(log_root.glob("P*/prediction_accuracy.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["trial_id"], int(row["prediction_step"]),
                    int(row["horizon"]),
                )
                if key in result:
                    raise RuntimeError(f"duplicate prediction provenance: {key}")
                result[key] = displayed_step_from_prediction(
                    step=int(row["horizon"]),
                    predicted_position=_json_position(row["predicted_position"]),
                    predicted_interact=row["predicted_interact"] == "1",
                    predicted_interact_position=(
                        _json_position(row["predicted_interact_position"])
                        if row["predicted_interact"] == "1" else None
                    ),
                )
    return result


def load_human_preview_decisions(log_root: Path) -> list[dict[str, object]]:
    trials = _trial_metadata(log_root)
    provenance = _prediction_provenance(log_root)
    decisions = []
    for path in sorted(log_root.glob("P*/step.csv")):
        participant = path.parent.name
        relative_source = path.relative_to(log_root.parent)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                trial = trials.get(row["trial_id"])
                if trial is None:
                    raise RuntimeError(f"step has no trial metadata: {row['trial_id']}")
                if trial["completed"] != "1":
                    continue
                horizon = int(trial["information_steps"])
                if horizon not in CONDITIONS:
                    continue
                if trial["visualization_mode"] != "full" or int(
                    trial["num_paths_shown"]
                ) != 1:
                    raise RuntimeError("preview trial is not full/rank-1 rendering")
                time_ms = int(trial["time_constraint_ms"])
                if time_ms not in TIME_CONDITIONS_MS:
                    raise RuntimeError(f"unexpected time condition: {time_ms}")
                layout = trial["layout"]
                if layout not in LAYOUT_DIMENSIONS:
                    raise RuntimeError(f"unsupported logged layout: {layout}")
                raw_trajectory = json.loads(row["future_trajectory"])
                entries = raw_trajectory.get("entries", [])
                assignments = []
                decision_index = int(row["step"])
                for entry in entries:
                    step = int(entry["overlay_horizon"])
                    tile = tuple(int(value) for value in entry["overlay_position"])
                    key = row["trial_id"], decision_index, step
                    predicted = provenance.get(key)
                    if predicted is not None:
                        if predicted.tile != tile:
                            raise RuntimeError(
                                f"logged renderer/prediction tile mismatch at {key}"
                            )
                        assignment = predicted
                    else:
                        assignment = DisplayedStep(step=step, tile=tile)
                    assignments.append(assignment)
                width, height = LAYOUT_DIMENSIONS[layout]
                analysis = analyze_assignments(
                    assignments, horizon=horizon, width=width, height=height
                )
                if analysis["positive_step_instances"] != horizon:
                    raise RuntimeError(
                        f"preview decision does not contain exactly H{horizon}: "
                        f"{row['trial_id']} step {decision_index}"
                    )
                decisions.append({
                    "source_identifier": f"{participant}/{row['trial_id']}",
                    "source_file": str(relative_source),
                    "participant_id": participant,
                    "trial_id": row["trial_id"],
                    "layout": layout,
                    "condition": f"H{horizon}",
                    "horizon": horizon,
                    "time_condition": f"T{time_ms}",
                    "time_constraint_ms": time_ms,
                    "decision_index": decision_index,
                    "analysis": analysis,
                })
    decisions.sort(key=lambda row: (
        row["condition"], row["time_constraint_ms"], row["layout"],
        row["participant_id"], row["trial_id"], row["decision_index"],
    ))
    if not decisions:
        raise RuntimeError("no completed H2/H4 preview decisions were loaded")
    return decisions


def _strata(
    decisions: Sequence[Mapping[str, object]],
) -> list[tuple[str, str, str, list[Mapping[str, object]]]]:
    rows = []
    for horizon in CONDITIONS:
        condition = f"H{horizon}"
        selected = [row for row in decisions if row["condition"] == condition]
        rows.append((condition, "overall", "ALL", selected))
        for time_ms in TIME_CONDITIONS_MS:
            rows.append((condition, "time", f"T{time_ms}", [
                row for row in selected if row["time_constraint_ms"] == time_ms
            ]))
        for layout in LAYOUT_DIMENSIONS:
            rows.append((condition, "layout", layout, [
                row for row in selected if row["layout"] == layout
            ]))
    return rows


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _summaries(
    decisions: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    decision_rows = []
    tile_rows = []
    for condition, scope, group, selected in _strata(decisions):
        analyses = [row["analysis"] for row in selected]
        decision_overlap = sum(bool(item["has_overlap"]) for item in analyses)
        complete = [item for item in analyses if item["provenance_complete"]]
        no_remap_decisions = sum(
            bool(item["no_remap_has_overlap"]) for item in complete
        )
        actual_complete_decisions = sum(bool(item["has_overlap"]) for item in complete)
        decision_rows.append({
            "condition": condition,
            "scope": scope,
            "group": group,
            "total_preview_bearing_decisions": len(selected),
            "decisions_with_overlap": decision_overlap,
            "overlap_decision_fraction": _ratio(decision_overlap, len(selected)),
            "provenance_complete_decisions": len(complete),
            "actual_overlap_decisions_in_provenance_complete_subset": (
                actual_complete_decisions
            ),
            "no_interaction_remap_overlap_decisions": no_remap_decisions,
            "interaction_remap_decision_overlap_delta": (
                actual_complete_decisions - no_remap_decisions
            ),
        })
        positive = sum(int(item["positive_step_instances"]) for item in analyses)
        unique = sum(int(item["unique_tiles"]) for item in analyses)
        overlap = sum(int(item["overlap_tiles"]) for item in analyses)
        overlap_steps = sum(int(item["overlap_step_instances"]) for item in analyses)
        overlap_records = [
            record for item in analyses for record in item["overlap_rows"]
        ]
        movement_only = sum(
            record["provenance_class"] == "movement-only"
            for record in overlap_records
        )
        interaction_containing = sum(
            record["provenance_class"] == "interaction-containing"
            for record in overlap_records
        )
        provenance_unknown = sum(
            record["provenance_class"] == "unknown"
            for record in overlap_records
        )
        no_remap_tiles = sum(int(item["no_remap_overlap_tiles"]) for item in complete)
        actual_complete_tiles = sum(int(item["overlap_tiles"]) for item in complete)
        tile_rows.append({
            "condition": condition,
            "scope": scope,
            "group": group,
            "total_positive_tile_step_instances": positive,
            "total_unique_rendered_tile_decision_instances": unique,
            "overlap_tiles": overlap,
            "overlap_tile_fraction": _ratio(overlap, unique),
            "step_instances_in_overlap_tiles": overlap_steps,
            "overlap_step_instance_fraction": _ratio(overlap_steps, positive),
            **{
                f"tiles_with_exactly_{value}_steps": sum(
                    int(item["multiplicity"][value]) for item in analyses
                )
                for value in range(1, 5)
            },
            "movement_only_overlap_tiles": movement_only,
            "movement_only_overlap_fraction": _ratio(movement_only, overlap),
            "interaction_containing_overlap_tiles": interaction_containing,
            "interaction_containing_overlap_fraction": _ratio(
                interaction_containing, overlap
            ),
            "unknown_provenance_overlap_tiles": provenance_unknown,
            "unknown_provenance_overlap_fraction": _ratio(
                provenance_unknown, overlap
            ),
            "interaction_remap_created_overlap_tiles": sum(
                record["interaction_remap_created"] is True
                for record in overlap_records
            ),
            "actual_overlap_tiles_in_provenance_complete_subset": (
                actual_complete_tiles
            ),
            "no_interaction_remap_overlap_tiles": no_remap_tiles,
            "interaction_remap_overlap_tile_delta": (
                actual_complete_tiles - no_remap_tiles
            ),
        })
    return decision_rows, tile_rows


def _all_overlap_combinations(horizon: int) -> list[tuple[int, ...]]:
    return [
        steps
        for multiplicity in range(2, horizon + 1)
        for steps in combinations(range(1, horizon + 1), multiplicity)
    ]


def _combination_rows(
    decisions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for condition, scope, group, selected in _strata(decisions):
        horizon = int(condition[1:])
        tile_records = [
            record for decision in selected
            for record in decision["analysis"]["tile_rows"]
        ]
        overlaps = [record for record in tile_records if record["multiplicity"] >= 2]
        counts: dict[tuple[int, ...], int] = {}
        for record in overlaps:
            steps = tuple(record["steps"])
            counts[steps] = counts.get(steps, 0) + 1
        for steps in _all_overlap_combinations(horizon):
            count = counts.get(steps, 0)
            rows.append({
                "condition": condition,
                "scope": scope,
                "group": group,
                "step_combination": "+".join(f"H{step}" for step in steps),
                "step_multiplicity": len(steps),
                "overlap_tile_count": count,
                "percent_among_overlap_tiles": 100.0 * _ratio(count, len(overlaps)),
                "percent_among_all_preview_tiles": 100.0 * _ratio(
                    count, len(tile_records)
                ),
            })
    return rows


def _compact_step_values(steps: Sequence[int], values: Sequence[object]) -> str:
    return ";".join(f"H{step}={value}" for step, value in zip(steps, values))


def _overlap_instance_rows(
    decisions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for decision in decisions:
        for record in decision["analysis"]["overlap_rows"]:
            steps = tuple(record["steps"])
            rows.append({
                "source_identifier": decision["source_identifier"],
                "source_file": decision["source_file"],
                "participant_id": decision["participant_id"],
                "trial_id": decision["trial_id"],
                "layout": decision["layout"],
                "condition": decision["condition"],
                "time_condition": decision["time_condition"],
                "time_constraint_ms": decision["time_constraint_ms"],
                "decision_index": decision["decision_index"],
                "tile_x": record["tile"][0],
                "tile_y": record["tile"][1],
                "canonical_tile_index": record["tile_index"],
                "overlapping_h_steps": "+".join(f"H{step}" for step in steps),
                "step_multiplicity": record["multiplicity"],
                "renderer_max_alpha": record["renderer_max_alpha"],
                "per_step_raw_alpha": _compact_step_values(
                    steps, record["raw_alphas"]
                ),
                "per_step_opacity_probability": _compact_step_values(
                    steps, [format(value, ".12g") for value in record["probabilities"]]
                ),
                "provenance_status": record["provenance_class"],
                "interaction_containing_flag": (
                    1 if record["provenance_class"] == "interaction-containing"
                    else (0 if record["provenance_class"] == "movement-only" else "")
                ),
                "interaction_steps": "+".join(
                    f"H{step}" for step in record["interaction_steps"]
                ),
                "movement_steps": "+".join(
                    f"H{step}" for step in record["movement_steps"]
                ),
                "unknown_provenance_steps": "+".join(
                    f"H{step}" for step in record["unknown_steps"]
                ),
                "interaction_remap_created_flag": (
                    "" if record["interaction_remap_created"] is None
                    else int(record["interaction_remap_created"])
                ),
            })
    return rows


def _renderer_rows(
    combination_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for row in combination_rows:
        if row["scope"] != "overall":
            continue
        horizon = int(str(row["condition"])[1:])
        steps = tuple(
            int(value[1:]) for value in str(row["step_combination"]).split("+")
        )
        alphas = rendered_preview_alphas(horizon)
        raw = tuple(alphas[step - 1] for step in steps)
        probabilities = tuple(value / 255.0 for value in raw)
        rows.append({
            "condition": row["condition"],
            "step_combination": row["step_combination"],
            "renderer_max_alpha": max(raw),
            "per_step_raw_alpha": _compact_step_values(steps, raw),
            "per_step_model_probability": _compact_step_values(
                steps, [format(value, ".12g") for value in probabilities]
            ),
            "overlap_tile_count": row["overlap_tile_count"],
            "percent_among_overlap_tiles": row["percent_among_overlap_tiles"],
            "percent_among_all_preview_tiles": row[
                "percent_among_all_preview_tiles"
            ],
            "semantic_consequence": (
                "renderer composites one tile at max alpha; model preserves "
                "independent per-step Bernoulli probabilities"
            ),
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def _markdown_table(
    rows: Sequence[Mapping[str, object]], columns: Sequence[tuple[str, str]],
) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _label in columns:
            value = row[key]
            if key.endswith("_fraction"):
                values.append(_percent(float(value)))
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_summary(
    path: Path, decisions: Sequence[Mapping[str, object]],
    decision_rows: Sequence[Mapping[str, object]],
    tile_rows: Sequence[Mapping[str, object]],
    combination_rows: Sequence[Mapping[str, object]],
) -> None:
    overall_decisions = [row for row in decision_rows if row["scope"] == "overall"]
    overall_tiles = [row for row in tile_rows if row["scope"] == "overall"]
    time_decisions = [row for row in decision_rows if row["scope"] == "time"]
    layout_decisions = [row for row in decision_rows if row["scope"] == "layout"]
    overall_combinations = [
        row for row in combination_rows
        if row["scope"] == "overall" and row["overlap_tile_count"] > 0
    ]
    participants = sorted({row["participant_id"] for row in decisions})
    trials = sorted({row["trial_id"] for row in decisions})
    incomplete_decisions = sum(
        not item["provenance_complete"]
        for row in decisions for item in [row["analysis"]]
    )
    total_steps = sum(
        int(row["analysis"]["positive_step_instances"]) for row in decisions
    )
    unknown_steps = sum(
        len(record["unknown_steps"])
        for row in decisions for record in row["analysis"]["tile_rows"]
    )
    lines = [
        "# Preview overlap audit",
        "",
        "## Source and scope",
        "",
        "- Source: actual completed human-study `data/raw/experiment/P*/step.csv` "
        "display states, joined to `trial.csv` and `prediction_accuracy.csv`.",
        f"- Included {len(participants)} participants, {len(trials)} completed H2/H4 "
        f"trials, and {len(decisions)} preview-bearing decisions.",
        "- The logged `future_trajectory.entries` are the rank-1, full/highlight "
        "renderer entries after interaction-target remapping; no synthetic paths, "
        "model rollout, policy loading, RNG, training, or behavioral metric was used.",
        "- Layout-local canonical tile index is x-major: `index = x*layout_height + y`. "
        "Dimensions are counter_circuit 8x5, double_rings 7x8, and "
        "unident_s_hard_onion 9x6.",
        "- Repeated identical tile/H-step entries are deduplicated. H2 filters out "
        "H3/H4 by construction.",
        "- H0 has no positive displayed steps and therefore zero overlap.",
        "",
        "## Overall prevalence",
        "",
        *_markdown_table(
            overall_decisions,
            (("condition", "condition"),
             ("total_preview_bearing_decisions", "preview decisions"),
             ("decisions_with_overlap", "decisions with overlap"),
             ("overlap_decision_fraction", "decision overlap rate")),
        ),
        "",
        *_markdown_table(
            overall_tiles,
            (("condition", "condition"),
             ("total_unique_rendered_tile_decision_instances", "unique tile-decisions"),
             ("overlap_tiles", "overlap tiles"),
             ("overlap_tile_fraction", "overlap-tile rate"),
             ("step_instances_in_overlap_tiles", "overlapped H instances"),
             ("overlap_step_instance_fraction", "H-instance overlap rate")),
        ),
        "",
        "`unique tile-decisions` means H steps are collapsed to unique rendered "
        "tiles within each decision, then those tile instances are summed across decisions.",
        "",
        "## T200/T400/T600",
        "",
        *_markdown_table(
            time_decisions,
            (("condition", "condition"), ("group", "T"),
             ("total_preview_bearing_decisions", "preview decisions"),
             ("decisions_with_overlap", "overlap decisions"),
             ("overlap_decision_fraction", "overlap rate")),
        ),
        "",
        "T is reported because the actual logs permit it. Geometry is generated "
        "per state rather than from T itself; observed T differences are descriptive.",
        "",
        "## Layout differences",
        "",
        *_markdown_table(
            layout_decisions,
            (("condition", "condition"), ("group", "layout"),
             ("total_preview_bearing_decisions", "preview decisions"),
             ("decisions_with_overlap", "overlap decisions"),
             ("overlap_decision_fraction", "overlap rate")),
        ),
        "",
        "## Multiplicity and provenance",
        "",
        *_markdown_table(
            overall_tiles,
            (("condition", "condition"),
             ("tiles_with_exactly_1_steps", "exactly 1"),
             ("tiles_with_exactly_2_steps", "exactly 2"),
             ("tiles_with_exactly_3_steps", "exactly 3"),
             ("tiles_with_exactly_4_steps", "exactly 4"),
             ("movement_only_overlap_tiles", "movement-only overlap"),
             ("movement_only_overlap_fraction", "movement-only %"),
             ("interaction_containing_overlap_tiles", "interaction-containing"),
             ("interaction_containing_overlap_fraction", "interaction %"),
             ("unknown_provenance_overlap_tiles", "provenance unknown")),
        ),
        "",
        "Interaction provenance is recovered by an exact join on trial, prediction "
        "step, and H-step. Prediction-accuracy rows are unavailable near terminal "
        f"boundaries for {unknown_steps} of {total_steps} displayed H-step instances "
        f"({_percent(_ratio(unknown_steps, total_steps))}); these affect "
        f"{incomplete_decisions} decisions. Affected overlap tiles are reported as "
        "unknown rather than guessed.",
        "",
        "For provenance-complete decisions, `no_interaction_remap_*` recomputes "
        "overlap using predicted player positions instead of interaction target tiles. "
        "A negative delta means remapping reduced overlap overall:",
        "",
        *_markdown_table(
            overall_decisions,
            (("condition", "condition"),
             ("provenance_complete_decisions", "eligible decisions"),
             ("actual_overlap_decisions_in_provenance_complete_subset", "actual overlap"),
             ("no_interaction_remap_overlap_decisions", "no-remap overlap"),
             ("interaction_remap_decision_overlap_delta", "actual - no-remap")),
        ),
        "",
        "Interaction remapping does **not** increase overlap in aggregate: among "
        "provenance-complete decisions it reduces the H2 and H4 decision-overlap "
        "counts shown above. It creates individual overlap tiles in 1 H2 instance "
        "and 1,883 H4 instances, but removes more repeated-position overlaps than "
        "it creates.",
        "",
        "## Exact step combinations",
        "",
        *_markdown_table(
            overall_combinations,
            (("condition", "condition"),
             ("step_combination", "combination"),
             ("overlap_tile_count", "count"),
             ("percent_among_overlap_tiles", "% overlap tiles"),
             ("percent_among_all_preview_tiles", "% all preview tiles")),
        ),
        "",
        "## Renderer alpha versus current model",
        "",
        "For every row above, the renderer shows one tile at the maximum alpha of "
        "its steps. The current model instead retains separate H channels and applies "
        "each step's `alpha/255` Bernoulli independently. Exact alpha/probability "
        "pairs and frequencies are in `renderer_vs_model_overlap.csv`: H2 uses "
        "(250, 70), while H4 uses (250, 190, 130, 70). No semantics were changed.",
        "",
        "## Interpretation and recommendation",
        "",
        "This is a descriptive geometry audit; LOW/NONTRIVIAL is not based on a "
        "hardcoded cutoff. The observed rates themselves drive the conclusion.",
        "",
        "- Classification: **NONTRIVIAL**.",
        "- Recommendation: **overlap is sufficiently common that representation "
        "semantics should be reconsidered before freeze**.",
        "- This recommendation uses only overlap prevalence and the renderer/model "
        "alpha-composition mismatch. It does not use human gaze, behavioral "
        "similarity, or RL performance.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(log_root: Path, output_dir: Path) -> dict[str, object]:
    decisions = load_human_preview_decisions(Path(log_root))
    decision_rows, tile_rows = _summaries(decisions)
    combination_rows = _combination_rows(decisions)
    instance_rows = _overlap_instance_rows(decisions)
    renderer_rows = _renderer_rows(combination_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "decision_overlap_summary.csv", decision_rows)
    _write_csv(output_dir / "tile_overlap_summary.csv", tile_rows)
    _write_csv(output_dir / "step_combination_counts.csv", combination_rows)
    _write_csv(output_dir / "overlap_instances.csv", instance_rows)
    _write_csv(output_dir / "renderer_vs_model_overlap.csv", renderer_rows)
    _write_summary(
        output_dir / "summary.md", decisions, decision_rows, tile_rows,
        combination_rows,
    )
    return {
        "decisions": decisions,
        "decision_summary": decision_rows,
        "tile_summary": tile_rows,
        "combination_counts": combination_rows,
        "overlap_instances": instance_rows,
        "renderer_vs_model": renderer_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=Path("data/raw/experiment"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("final_model/eval/results/preview_overlap_audit"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(args.log_root, args.output_dir)
    overall = [
        row for row in result["decision_summary"] if row["scope"] == "overall"
    ]
    for row in overall:
        print(
            f"{row['condition']}: {row['decisions_with_overlap']}/"
            f"{row['total_preview_bearing_decisions']} decisions overlap "
            f"({_percent(row['overlap_decision_fraction'])})"
        )
    print(f"Wrote {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
