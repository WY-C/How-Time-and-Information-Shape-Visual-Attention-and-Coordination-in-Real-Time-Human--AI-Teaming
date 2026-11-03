"""Canonical processed-human extraction for the standardized benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_model.eval.benchmark_contracts import (
    BENCHMARK_CONTRACT_VERSION,
    CONTEXT_RQ3_STATUS,
    PRIMARY_AOI_CONTRACT,
    SENSITIVITY_AOI_CONTRACT,
    allocate_preview_horizons,
    benchmark_contract_sha256,
    benchmark_registry,
    euclidean_tile_distance,
    manuscript_aoi,
    v6_aoi_allocation,
)
from final_model.eval.common import write_json
from final_model.eval.human_task_events import (
    detect_task_events,
    fit_task_event_gee,
    task_event_regression_summary,
)


TIME_LEVELS = (200, 400, 600)
PREVIEW_LEVELS = (0, 2, 4)
LAYOUT_LEVELS = ("counter_circuit", "double_rings", "unident_s_hard_onion")
SELECTED_TARGETS = {200: 87.6076, 400: 211.5641, 600: 241.4401}
REVISION_TARGETS = {
    200: (754, 37_635), 400: (2_482, 51_456), 600: (3_156, 53_974)
}
POST_FIRST_TARGETS = {200: .3978498, 400: .4571253, 600: .5902670}
TASK_TARGETS = {
    "pot_completion": 2622,
    "ai_delivery": 3313,
    "redundant_onion": 655,
    "terminal_step_179_delivery": 19,
    "redundant_release_at_deposit": 146,
}


class HumanRegressionError(RuntimeError):
    pass


def _save(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def load_human_data(data_root: Path):
    root = Path(data_root).expanduser().resolve()
    trial_files = sorted(root.glob("P*/trial.csv"))
    step_files = sorted(root.glob("P*/step.csv"))
    fixation_files = sorted(root.glob("P*/fixation.csv"))
    if not (len(trial_files) == len(step_files) == len(fixation_files) == 36):
        raise RuntimeError("expected P01--P36 trial, step, and fixation files")
    trials = pd.concat([pd.read_csv(path) for path in trial_files], ignore_index=True)
    trials["Time"] = pd.to_numeric(trials.time_constraint_ms, errors="coerce")
    trials["Preview"] = pd.to_numeric(trials.information_steps, errors="coerce")
    eligible = trials[
        trials.completed.eq(1)
        & trials.decision_steps_completed.eq(180)
        & trials.Time.isin(TIME_LEVELS)
        & trials.Preview.isin(PREVIEW_LEVELS)
    ].copy()
    eligible = eligible.rename(columns={"layout": "Layout"})
    eligible["trial_order_c"] = eligible.global_trial_index - 14
    if len(eligible) != 972 or eligible.pid.nunique() != 36:
        raise RuntimeError("canonical human eligibility is not 36 participants / 972 trials")
    cell = eligible.groupby(["pid", "Time", "Preview"]).agg(
        trials=("trial_id", "size"), layouts=("Layout", "nunique")
    )
    if not (cell.trials.eq(3) & cell.layouts.eq(3)).all():
        raise RuntimeError("human Time x Preview cells are not layout-balanced")
    valid = set(eligible.trial_id)
    columns = [
        "pid", "trial_id", "step", "state_onset_ms", "human_action",
        "agent_action", "env_state", "human_held_object", "agent_held_object",
        "pot_states", "is_collision", "is_human_idle", "accumulated_score",
        "reward", "response_time_ms", "human_input_ms", "action_execution_ms",
        "first_human_action", "first_human_input_ms", "final_human_action",
        "final_human_input_ms", "human_input_changed",
    ]
    parts = []
    for path in step_files:
        part = pd.read_csv(path, usecols=columns)
        parts.append(part[part.trial_id.isin(valid)])
    steps = pd.concat(parts, ignore_index=True)
    meta = eligible[[
        "trial_id", "Time", "Preview", "Layout", "global_trial_index",
        "trial_order_c", "mean_gaze_validity", "score_final",
        "trial_duration_ms", "num_collisions", "num_human_idle",
    ]]
    steps = steps.merge(meta, on="trial_id", validate="many_to_one")
    if len(steps) != 174_960 or steps.duplicated(["trial_id", "step"]).any():
        raise RuntimeError("canonical human decisions are not unique 174,960 rows")
    return root, eligible, steps, fixation_files


def behavior_outputs(eligible, steps):
    selected = steps[steps.human_input_ms.notna()].copy()
    selected["selected_action_latency_ms"] = selected.human_input_ms - selected.state_onset_ms
    trial = selected.groupby(
        ["pid", "trial_id", "Time", "Preview", "Layout"], as_index=False
    ).selected_action_latency_ms.mean()
    cell = trial.groupby(
        ["pid", "Time", "Preview"], as_index=False
    ).selected_action_latency_ms.mean()
    selected_time = cell.groupby("Time", as_index=False).selected_action_latency_ms.mean()
    selected_time["metric"] = "selected_action_latency_ms"

    non_timeout = steps[steps.first_human_input_ms.notna()].copy()
    non_timeout["canonical_had_revision"] = non_timeout.human_input_changed.astype(int)
    revision = non_timeout.groupby("Time", as_index=False).agg(
        numerator=("canonical_had_revision", "sum"),
        denominator=("canonical_had_revision", "size"),
    )
    revision["value"] = revision.numerator / revision.denominator
    revision["metric"] = "canonical_revision_rate"

    decision = steps.copy()
    decision["forced_stay"] = decision.first_human_input_ms.isna().astype(int)
    decision["intentional_stay"] = (
        decision.first_human_input_ms.notna() & decision.human_action.eq("stay")
    ).astype(int)
    marginal = decision.groupby("Time", as_index=False).agg(
        decisions=("step", "size"),
        forced_stay=("forced_stay", "mean"),
        intentional_stay=("intentional_stay", "mean"),
        collision=("is_collision", "mean"),
    )
    cells = decision.groupby(["Time", "Preview"], as_index=False).agg(
        decisions=("step", "size"),
        forced_stay=("forced_stay", "mean"),
        intentional_stay=("intentional_stay", "mean"),
        collision=("is_collision", "mean"),
    )
    return selected_time, revision, marginal, cells


def process_eligibility(eligible):
    session = eligible.groupby("pid", as_index=False).mean_gaze_validity.mean()
    session["session_pass"] = session.mean_gaze_validity.ge(.80)
    audit = eligible.merge(session[["pid", "session_pass"]], on="pid")
    audit["trial_pass"] = audit.mean_gaze_validity.ge(.90)
    audit["decision_count_pass"] = audit.decision_steps_completed.eq(180)
    audit["process_included"] = audit.session_pass & audit.trial_pass & audit.decision_count_pass
    included = audit[audit.process_included]
    summary = {
        "participants": int(included.pid.nunique()),
        "trials": int(len(included)),
        "decisions": int(len(included) * 180),
        "excluded_participants": sorted(audit.loc[~audit.session_pass, "pid"].unique()),
        "rule": "session mean validity >= .80 AND trial mean validity >= .90 AND exactly 180 decisions",
    }
    return audit, summary, set(included.trial_id)


def parse_processed_segments(fixation_files):
    records = []
    columns = [
        "pid", "trial_id", "fixation_id", "fixation_start_step",
        "fixation_start_ms", "fixation_end_ms", "base_position", "context_segments",
    ]
    fixation_parts = []
    for path in fixation_files:
        fixations = pd.read_csv(path, usecols=columns)
        fixation_parts.append(fixations.drop(columns="context_segments"))
        for row in fixations.itertuples(index=False):
            for segment_index, segment in enumerate(json.loads(row.context_segments)):
                matches = segment.get("overlay_matches") or []
                horizons = sorted({
                    int(match["overlay_horizon"])
                    for match in matches
                    if match.get("overlay_horizon") is not None
                })
                records.append({
                    "pid": row.pid, "trial_id": row.trial_id,
                    "fixation_id": row.fixation_id,
                    "fixation_start_step": row.fixation_start_step,
                    "fixation_start_ms": float(row.fixation_start_ms),
                    "fixation_end_ms": float(row.fixation_end_ms),
                    "segment_index": segment_index,
                    "segment_start_ms": float(segment["start_ms"]),
                    "segment_end_ms": float(segment["end_ms"]),
                    "step": int(segment["step"]),
                    "base_aoi": segment.get("base_aoi"),
                    "base_position": json.dumps(segment.get("base_position")),
                    "overlay_flag": int(segment.get("overlay_flag", 0)),
                    "overlay_underlying_aoi": segment.get("overlay_underlying_aoi"),
                    "matched_horizons": json.dumps(horizons),
                })
    segments = pd.DataFrame.from_records(records)
    fixations = pd.concat(fixation_parts, ignore_index=True)
    if (segments.segment_end_ms <= segments.segment_start_ms).any():
        raise RuntimeError("processed human gaze has nonpositive context segments")
    return segments, fixations


def _overlap(data, left, right):
    return np.maximum(
        0.0,
        np.minimum(data.segment_end_ms.to_numpy(), np.asarray(right))
        - np.maximum(data.segment_start_ms.to_numpy(), np.asarray(left)),
    )


def attach_gaze_windows(segments, steps):
    columns = [
        "trial_id", "step", "state_onset_ms", "first_human_input_ms",
        "human_input_ms", "action_execution_ms", "Time", "Preview", "Layout",
        "trial_order_c",
    ]
    data = segments.merge(steps[columns], on=["trial_id", "step"], validate="many_to_one")
    data["full_ms"] = _overlap(data, data.state_onset_ms, data.action_execution_ms)
    data["fixed_200_ms"] = _overlap(
        data, data.state_onset_ms,
        np.minimum(data.state_onset_ms + 200, data.action_execution_ms),
    )
    first_start = data.first_human_input_ms.fillna(data.action_execution_ms)
    selected_start = data.human_input_ms.fillna(data.action_execution_ms)
    data["post_first_ms"] = _overlap(data, first_start, data.action_execution_ms)
    data["post_selected_ms"] = _overlap(data, selected_start, data.action_execution_ms)
    pre_first_end = data.first_human_input_ms.fillna(data.state_onset_ms + data.Time)
    data["pre_first_ms"] = _overlap(data, data.state_onset_ms, pre_first_end)
    data["primary_aoi"] = [
        manuscript_aoi(base, bool(overlay))
        for base, overlay in zip(data.base_aoi, data.overlay_flag)
    ]
    return data


def post_input_outputs(data):
    group = ["pid", "trial_id", "Time", "Preview"]
    durations = data.groupby(group, as_index=False)[
        ["full_ms", "post_first_ms", "post_selected_ms"]
    ].sum()
    cells = durations.groupby(["pid", "Time", "Preview"], as_index=False)[
        ["full_ms", "post_first_ms", "post_selected_ms"]
    ].sum()
    cells["post_first_input_fraction"] = cells.post_first_ms / cells.full_ms
    cells["post_selected_action_fraction"] = cells.post_selected_ms / cells.full_ms
    summary = cells.groupby("Time", as_index=False)[
        ["post_first_input_fraction", "post_selected_action_fraction"]
    ].mean()
    return cells, summary


def aoi_outputs(data):
    use = data[data.full_ms.gt(0)].copy()
    group = ["pid", "trial_id", "Time", "Preview", "Layout"]
    total = use.groupby(group, as_index=False).full_ms.sum().rename(columns={"full_ms": "classified_ms"})
    primary = use.groupby(group + ["primary_aoi"], as_index=False).full_ms.sum()
    wide = primary.pivot(index=group, columns="primary_aoi", values="full_ms").fillna(0).reset_index()
    wide = wide.merge(total, on=group, validate="one_to_one")
    for label in ("Player", "Teammate", "Action Preview", "Other"):
        if label not in wide:
            wide[label] = 0.0
        wide[label + "_share"] = wide[label] / wide.classified_ms
    primary_cells = wide.groupby(["pid", "Time", "Preview"], as_index=False)[
        [label + "_share" for label in ("Player", "Teammate", "Action Preview", "Other")]
    ].mean()

    v6_records = []
    for row in use.itertuples(index=False):
        allocation = v6_aoi_allocation(row.base_aoi, bool(row.overlay_flag), row.full_ms)
        v6_records.append({
            "pid": row.pid, "trial_id": row.trial_id, "Time": row.Time,
            "Preview": row.Preview, "Layout": row.Layout, **allocation,
        })
    v6 = pd.DataFrame(v6_records).groupby(group, as_index=False)[
        ["Player", "Teammate", "Action Preview", "Other"]
    ].sum()
    v6["classified_ms"] = v6[["Player", "Teammate", "Action Preview", "Other"]].sum(axis=1)
    for label in ("Player", "Teammate", "Action Preview", "Other"):
        v6[label + "_share"] = v6[label] / v6.classified_ms
    v6_cells = v6.groupby(["pid", "Time", "Preview"], as_index=False)[
        [label + "_share" for label in ("Player", "Teammate", "Action Preview", "Other")]
    ].mean()
    return wide, primary_cells, v6, v6_cells


def carryover_outputs(data):
    use = data[data.fixed_200_ms.gt(0)].copy()
    use["component"] = np.where(
        use.fixation_start_ms.lt(use.state_onset_ms), "carryover", "new"
    )
    use["ai_related"] = use.primary_aoi.isin(["Teammate", "Action Preview"])
    group = ["pid", "trial_id", "Time", "Preview", "Layout"]
    totals = use.groupby(group, as_index=False).fixed_200_ms.sum().rename(columns={"fixed_200_ms": "classified_ms"})
    components = use[use.ai_related].groupby(group + ["component"], as_index=False).fixed_200_ms.sum()
    wide = components.pivot(index=group, columns="component", values="fixed_200_ms").fillna(0).reset_index()
    wide = totals.merge(wide, on=group, how="left").fillna(0)
    for component in ("carryover", "new"):
        if component not in wide:
            wide[component] = 0.0
        wide[component + "_share"] = wide[component] / wide.classified_ms
    cells = wide.groupby(["pid", "Time", "Preview"], as_index=False)[
        ["carryover_share", "new_share"]
    ].mean()
    return wide, cells.groupby("Time", as_index=False)[["carryover_share", "new_share"]].mean()


def preview_horizon_outputs(data):
    use = data[data.full_ms.gt(0) & data.primary_aoi.eq("Action Preview") & data.Preview.gt(0)].copy()
    rows = []
    for row in use.itertuples(index=False):
        allocation = allocate_preview_horizons(row.full_ms, json.loads(row.matched_horizons))
        for horizon, duration in allocation.items():
            rows.append({
                "pid": row.pid, "trial_id": row.trial_id, "Time": row.Time,
                "Preview": row.Preview, "Layout": row.Layout,
                "future_step": horizon, "allocated_ms": duration,
            })
    allocated = pd.DataFrame(rows)
    group = ["pid", "trial_id", "Time", "Preview", "Layout"]
    total = data[data.full_ms.gt(0) & data.Preview.gt(0)].groupby(group, as_index=False).full_ms.sum().rename(columns={"full_ms": "classified_ms"})
    preview = allocated.groupby(group, as_index=False).allocated_ms.sum().rename(columns={"allocated_ms": "preview_ms"})
    trial = total.merge(preview, on=group, how="left").fillna({"preview_ms": 0})
    trial["preview_share"] = trial.preview_ms / trial.classified_ms
    trial["per_represented_step_share"] = trial.preview_share / trial.Preview
    cells = trial.groupby(["pid", "Time", "Preview"], as_index=False)[
        ["preview_share", "per_represented_step_share"]
    ].mean()
    return allocated, trial, cells


def visual_sampling_outputs(fixations, steps, valid_trials):
    data = fixations[fixations.trial_id.isin(valid_trials)].copy()
    data = data.sort_values(["trial_id", "fixation_id"])
    def parse_position(value):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, list) and len(parsed) == 2 else None
    data["position"] = data.base_position.map(parse_position)
    data["previous_position"] = data.groupby("trial_id").position.shift()
    decision = steps[steps.trial_id.isin(valid_trials)][[
        "pid", "trial_id", "step", "Time", "Preview", "Layout",
        "trial_order_c", "state_onset_ms", "first_human_input_ms",
        "action_execution_ms",
    ]].copy()
    joined = data.merge(
        decision,
        left_on=["pid", "trial_id", "fixation_start_step"],
        right_on=["pid", "trial_id", "step"],
        how="inner",
    )
    joined["pre_first_end"] = joined.first_human_input_ms.fillna(joined.state_onset_ms + joined.Time)
    joined["fixed_200_end"] = np.minimum(joined.state_onset_ms + 200, joined.action_execution_ms)
    keys = ["pid", "trial_id", "step", "Time", "Preview", "Layout"]
    decision_keys = decision[keys].drop_duplicates()
    count_rows, saccade_rows = [], []
    for window, end in (
        ("pre_first", "pre_first_end"),
        ("initial_200ms", "fixed_200_end"),
        ("full_window", "action_execution_ms"),
    ):
        candidates = joined[
            joined.fixation_start_ms.gt(joined.state_onset_ms)
            & joined.fixation_start_ms.lt(joined[end])
        ].copy()
        counts = candidates.groupby(keys).fixation_id.nunique().rename("new_fixations").reset_index()
        merged = decision_keys.merge(counts, on=keys, how="left").fillna({"new_fixations": 0})
        merged["window"] = window
        merged["at_least_1"] = merged.new_fixations.ge(1).astype(int)
        merged["at_least_2"] = merged.new_fixations.ge(2).astype(int)
        count_rows.append(merged)
        first = candidates.sort_values("fixation_start_ms").drop_duplicates(["trial_id", "step"])
        for row in first.itertuples(index=False):
            distance = euclidean_tile_distance(row.previous_position, row.position)
            if distance is not None:
                saccade_rows.append({
                    "pid": row.pid, "trial_id": row.trial_id, "step": row.step,
                    "Time": row.Time, "Preview": row.Preview, "Layout": row.Layout,
                    "window": window, "euclidean_tiles": distance,
                    "distance_gt_3": int(distance > 3),
                })
    counts = pd.concat(count_rows, ignore_index=True)
    count_summary = counts.groupby(["window", "Time"], as_index=False).agg(
        decisions=("new_fixations", "size"),
        mean_new_fixations=("new_fixations", "mean"),
        at_least_1=("at_least_1", "mean"),
        at_least_2=("at_least_2", "mean"),
    )
    saccades = pd.DataFrame(saccade_rows)
    saccade_summary = saccades.groupby(["window", "Time"], as_index=False).agg(
        evaluable=("distance_gt_3", "size"),
        distance_gt_3=("distance_gt_3", "mean"),
    )
    return counts, count_summary, saccades, saccade_summary


def strict_check_outputs(data, steps, valid_trials):
    keys = ["pid", "trial_id", "step", "Time", "Preview", "Layout", "trial_order_c"]
    decisions = steps[steps.trial_id.isin(valid_trials)][keys + [
        "state_onset_ms", "first_human_input_ms", "human_input_ms", "action_execution_ms"
    ]].copy()
    phases = {
        "initial_200ms": ("fixed_200_ms", "state_onset_ms", None),
        "full_window": ("full_ms", "state_onset_ms", "action_execution_ms"),
        "pre_first": ("pre_first_ms", "state_onset_ms", None),
    }
    outputs = []
    for phase, (duration, start_column, end_column) in phases.items():
        use = data[data.trial_id.isin(valid_trials) & data[duration].gt(0)].copy()
        start = use[start_column]
        if phase == "initial_200ms":
            end = np.minimum(use.state_onset_ms + 200, use.action_execution_ms)
        elif phase == "pre_first":
            end = use.first_human_input_ms.fillna(use.state_onset_ms + use.Time)
        else:
            end = use[end_column]
        use = use[use.fixation_start_ms.ge(start) & use.fixation_start_ms.lt(end)].copy()
        use["teammate_source"] = (
            (use.overlay_flag.eq(0) & use.base_aoi.eq("Partner"))
            | (use.overlay_flag.eq(1) & use.overlay_underlying_aoi.eq("Partner"))
        )
        use["preview_source"] = use.overlay_flag.eq(1)
        source_rows = []
        for label, indicator in (("Teammate_check", "teammate_source"), ("Preview_check", "preview_source")):
            part = use[use[indicator]][keys + ["fixation_id", duration]].copy()
            part["source"] = label
            source_rows.append(part)
        sources = pd.concat(source_rows, ignore_index=True)
        dwell = sources.groupby(keys + ["fixation_id", "source"], as_index=False)[duration].sum()
        strict = dwell[dwell[duration].ge(60)].groupby(keys + ["source"]).size().unstack(fill_value=0).reset_index()
        for label in ("Teammate_check", "Preview_check"):
            if label not in strict:
                strict[label] = 0
            strict[label] = strict[label].gt(0).astype(int)
        strict["Any_AI"] = strict[["Teammate_check", "Preview_check"]].max(axis=1)
        result = decisions[keys].merge(
            strict[keys + ["Teammate_check", "Preview_check", "Any_AI"]],
            on=keys, how="left",
        ).fillna({"Teammate_check": 0, "Preview_check": 0, "Any_AI": 0})
        result["phase"] = phase
        outputs.append(result)
    rows = pd.concat(outputs, ignore_index=True)
    summary = rows.groupby(["phase", "Time"], as_index=False).agg(
        decisions=("Any_AI", "size"), teammate_rate=("Teammate_check", "mean"),
        preview_rate=("Preview_check", "mean"), any_ai_rate=("Any_AI", "mean"),
    )
    return rows, summary


def performance_outputs(eligible):
    trial = eligible[[
        "pid", "trial_id", "Time", "Preview", "Layout", "score_final",
        "trial_duration_ms", "num_collisions", "num_human_idle",
    ]].copy()
    trial["deliveries_per_100_decisions"] = trial.score_final / 20 / 180 * 100
    trial["points_per_minute"] = trial.score_final / (trial.trial_duration_ms / 60_000)
    cells = trial.groupby(["pid", "Time", "Preview"], as_index=False)[[
        "score_final", "deliveries_per_100_decisions", "points_per_minute",
        "trial_duration_ms", "num_collisions", "num_human_idle",
    ]].mean()
    metrics = [
        "score_final", "deliveries_per_100_decisions", "points_per_minute",
        "trial_duration_ms", "num_collisions", "num_human_idle",
    ]
    return trial, cells.groupby("Time", as_index=False)[metrics].mean()


def _regression(selected, revision, post, process, task):
    checks = []
    for row in selected.itertuples(index=False):
        expected = SELECTED_TARGETS[int(row.Time)]
        checks.append({
            "target": f"selected_action_latency_T{int(row.Time)}",
            "observed": row.selected_action_latency_ms, "expected": expected,
            "tolerance": .001,
            "pass": abs(row.selected_action_latency_ms - expected) <= .001,
        })
    for row in revision.itertuples(index=False):
        expected_n, expected_d = REVISION_TARGETS[int(row.Time)]
        checks.extend([
            {"target": f"revision_numerator_T{int(row.Time)}", "observed": int(row.numerator), "expected": expected_n, "tolerance": 0, "pass": int(row.numerator) == expected_n},
            {"target": f"revision_denominator_T{int(row.Time)}", "observed": int(row.denominator), "expected": expected_d, "tolerance": 0, "pass": int(row.denominator) == expected_d},
        ])
    for row in post.itertuples(index=False):
        expected = POST_FIRST_TARGETS[int(row.Time)]
        checks.append({
            "target": f"post_first_input_fraction_T{int(row.Time)}",
            "observed": row.post_first_input_fraction, "expected": expected,
            "tolerance": 5e-7,
            "pass": abs(row.post_first_input_fraction - expected) <= 5e-7,
        })
    for key, expected in (("participants", 35), ("trials", 907), ("decisions", 163_260)):
        checks.append({"target": f"process_{key}", "observed": process[key], "expected": expected, "tolerance": 0, "pass": process[key] == expected})
    for key, expected in TASK_TARGETS.items():
        checks.append({"target": f"task_{key}", "observed": task[key], "expected": expected, "tolerance": 0, "pass": task[key] == expected})
    return pd.DataFrame(checks)


def run_human_benchmark(
    *, data_root: Path, output_dir: Path, run_task_gee: bool = True
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_out = output / "raw"; summaries = output / "summaries"; benchmark = output / "benchmark"
    raw_out.mkdir(); summaries.mkdir(); benchmark.mkdir()
    root, eligible, steps, fixation_files = load_human_data(data_root)
    selected, revision, behavior_time, behavior_cells = behavior_outputs(eligible, steps)
    process_audit, process_summary, process_trials = process_eligibility(eligible)
    segments, fixations = parse_processed_segments(fixation_files)
    gaze = attach_gaze_windows(segments, steps)
    post_cells, post_summary = post_input_outputs(gaze)
    aoi_trial, aoi_cells, v6_trial, v6_cells = aoi_outputs(gaze)
    carry_trial, carry_summary = carryover_outputs(gaze)
    horizon_allocated, horizon_trial, horizon_cells = preview_horizon_outputs(gaze)
    visual, visual_summary, saccades, saccade_summary = visual_sampling_outputs(
        fixations, steps, process_trials
    )
    checks, check_summary = strict_check_outputs(gaze, steps, process_trials)
    performance_trial, performance_summary = performance_outputs(eligible)
    task_events = detect_task_events(eligible, steps)
    task_summary = task_event_regression_summary(task_events)
    regression = _regression(
        selected, revision, post_summary, process_summary, task_summary
    )

    _save(selected, summaries / "selected_action_latency.csv")
    _save(revision, summaries / "revision.csv")
    _save(behavior_time, summaries / "decision_behavior.csv")
    _save(behavior_cells, summaries / "decision_behavior_time_horizon.csv")
    _save(process_audit, raw_out / "process_eligibility.csv")
    _save(post_cells, raw_out / "post_input_participant_cells.csv")
    _save(post_summary, summaries / "post_input_gaze.csv")
    _save(aoi_trial, raw_out / "aoi_full_window_manuscript_trials.csv")
    _save(aoi_cells, summaries / "aoi_full_window.csv")
    _save(v6_trial, raw_out / "aoi_full_window_v6_sensitivity_trials.csv")
    _save(v6_cells, summaries / "aoi_full_window_v6_sensitivity.csv")
    _save(carry_trial, raw_out / "carryover_trials.csv")
    _save(carry_summary, summaries / "carryover.csv")
    _save(horizon_allocated, raw_out / "future_step_allocations.csv")
    _save(horizon_trial, summaries / "preview_horizon.csv")
    _save(horizon_cells, summaries / "future_steps.csv")
    _save(visual, raw_out / "visual_sampling_decisions.csv")
    _save(visual_summary, summaries / "visual_sampling.csv")
    _save(saccades, raw_out / "first_new_saccade_decisions.csv")
    _save(saccade_summary, summaries / "first_new_saccade.csv")
    _save(checks, raw_out / "context_check_decisions.csv")
    _save(check_summary, summaries / "context_checks.csv")
    _save(performance_trial, raw_out / "performance_trials.csv")
    _save(performance_summary, summaries / "performance.csv")
    layout_rows = []
    for row in performance_trial.groupby(["Layout", "Time"], as_index=False)[[
        "score_final", "deliveries_per_100_decisions", "points_per_minute"
    ]].mean().itertuples(index=False):
        for metric in ("score_final", "deliveries_per_100_decisions", "points_per_minute"):
            layout_rows.append({
                "analysis": "performance", "layout": row.Layout,
                "Time": row.Time, "Preview": None,
                "metric": metric, "value": getattr(row, metric),
                "availability": "available",
            })
    aoi_layout = aoi_trial.groupby(["Layout", "Time", "Preview"], as_index=False)[[
        "Player_share", "Teammate_share", "Action Preview_share", "Other_share"
    ]].mean()
    for _, row in aoi_layout.iterrows():
        for metric in ("Player_share", "Teammate_share", "Action Preview_share", "Other_share"):
            layout_rows.append({
                "analysis": "aoi_full_window", "layout": row["Layout"],
                "Time": row["Time"], "Preview": row["Preview"],
                "metric": metric, "value": row[metric],
                "availability": "available",
            })
    _save(pd.DataFrame(layout_rows), summaries / "layout.csv")
    for name, frame in task_events.items():
        _save(frame, raw_out / f"task_events_{name}.csv")
    task_rates = pd.concat([
        frame.groupby("Time", as_index=False).agg(
            opportunities=("outcome", "size"), successes=("outcome", "sum"), rate=("outcome", "mean")
        ).assign(event_type=name)
        for name, frame in task_events.items()
    ], ignore_index=True)
    _save(task_rates, summaries / "task_events.csv")
    if run_task_gee:
        _save(fit_task_event_gee(task_events), summaries / "task_event_gee.csv")
    _save(regression, benchmark / "human_regression.csv")
    reference_rows = []
    for row in selected.itertuples(index=False):
        reference_rows.append({
            "source": "human", "aoi_contract": None, "Time": row.Time,
            "Horizon": None, "metric": "selected_action_latency_ms",
            "value": row.selected_action_latency_ms, "numerator": None,
            "denominator": None,
        })
    for row in revision.itertuples(index=False):
        reference_rows.append({
            "source": "human", "aoi_contract": None, "Time": row.Time,
            "Horizon": None, "metric": "canonical_revision_rate",
            "value": row.value, "numerator": row.numerator,
            "denominator": row.denominator,
        })
    for row in post_summary.itertuples(index=False):
        for metric in ("post_first_input_fraction", "post_selected_action_fraction"):
            reference_rows.append({
                "source": "human", "aoi_contract": PRIMARY_AOI_CONTRACT,
                "Time": row.Time, "Horizon": None, "metric": metric,
                "value": getattr(row, metric), "numerator": None,
                "denominator": None,
            })
    primary_aoi_reference = aoi_cells.groupby(["Time", "Preview"], as_index=False)[[
        "Player_share", "Teammate_share", "Action Preview_share", "Other_share"
    ]].mean()
    aoi_name = {
        "Player_share": "primary_player_share",
        "Teammate_share": "primary_teammate_share",
        "Action Preview_share": "primary_action_preview_share",
        "Other_share": "primary_other_share",
    }
    for _, row in primary_aoi_reference.iterrows():
        for column, metric in aoi_name.items():
            reference_rows.append({
                "source": "human", "aoi_contract": PRIMARY_AOI_CONTRACT,
                "Time": row["Time"], "Horizon": row["Preview"],
                "metric": metric, "value": row[column],
                "numerator": None, "denominator": None,
            })
    for row in carry_summary.itertuples(index=False):
        for metric in ("carryover_share", "new_share"):
            reference_rows.append({
                "source": "human", "aoi_contract": PRIMARY_AOI_CONTRACT,
                "Time": row.Time, "Horizon": None, "metric": metric,
                "value": getattr(row, metric), "numerator": None,
                "denominator": None,
            })
    horizon_reference = horizon_cells.groupby(["Time", "Preview"], as_index=False)[[
        "preview_share", "per_represented_step_share"
    ]].mean()
    for row in horizon_reference.itertuples(index=False):
        for metric in ("preview_share", "per_represented_step_share"):
            reference_rows.append({
                "source": "human", "aoi_contract": PRIMARY_AOI_CONTRACT,
                "Time": row.Time, "Horizon": row.Preview, "metric": metric,
                "value": getattr(row, metric), "numerator": None,
                "denominator": None,
            })
    for row in visual_summary.itertuples(index=False):
        for column in ("mean_new_fixations", "at_least_1", "at_least_2"):
            reference_rows.append({
                "source": "human", "aoi_contract": None,
                "Time": row.Time, "Horizon": None,
                "metric": f"visual_sampling_{row.window}_{column}",
                "value": getattr(row, column), "numerator": None,
                "denominator": row.decisions,
            })
    for row in check_summary.itertuples(index=False):
        for column in ("teammate_rate", "preview_rate", "any_ai_rate"):
            reference_rows.append({
                "source": "human", "aoi_contract": None,
                "Time": row.Time, "Horizon": None,
                "metric": f"strict_check_{row.phase}_{column}",
                "value": getattr(row, column), "numerator": None,
                "denominator": row.decisions,
            })
    for row in performance_summary.itertuples(index=False):
        for metric in ("score_final", "deliveries_per_100_decisions", "points_per_minute"):
            reference_rows.append({
                "source": "human", "aoi_contract": None,
                "Time": row.Time, "Horizon": None, "metric": metric,
                "value": getattr(row, metric), "numerator": None,
                "denominator": 36,
            })
    for row in task_rates.itertuples(index=False):
        reference_rows.append({
            "source": "human", "aoi_contract": None,
            "Time": row.Time, "Horizon": None,
            "metric": f"task_event_{row.event_type}_rate",
            "value": row.rate, "numerator": row.successes,
            "denominator": row.opportunities,
        })
    _save(pd.DataFrame(reference_rows), benchmark / "human_reference.csv")
    write_json(benchmark / "analysis_registry.json", benchmark_registry())
    write_json(benchmark / "human_reference.json", {
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "benchmark_contract_sha256": benchmark_contract_sha256(),
        "data_root": str(root),
        "processed_gaze_source": "P01-P36/fixation.csv context_segments",
        "raw_gaze_regenerated": False,
        "aoi_primary_contract": PRIMARY_AOI_CONTRACT,
        "aoi_sensitivity_contract": SENSITIVITY_AOI_CONTRACT,
        "context_rq3_status": CONTEXT_RQ3_STATUS,
        "process_eligibility": process_summary,
        "task_event_counts": task_summary,
        "regression_pass": bool(regression["pass"].all()),
    })
    failures = regression[~regression["pass"]]
    if len(failures):
        raise HumanRegressionError(
            "human regression mismatch; stop before model pilot:\n"
            + failures.to_string(index=False)
        )
    return {
        "output_dir": output,
        "regression_pass": True,
        "regression": regression,
        "process": process_summary,
        "task_events": task_summary,
    }
