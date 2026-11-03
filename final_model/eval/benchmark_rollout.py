"""Temporal human--model benchmark rollout using explicit checkpoint pairs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.common.partners import FixedHSPPartnerRepository
from final_model.eval.actionnet_diagnostics import read_only_probabilities_and_hidden
from final_model.eval.actionnet_diagnostics import (
    read_only_logits_probabilities_and_hidden,
)
from final_model.eval.benchmark_checkpoints import (
    BenchmarkCheckpointPair,
    load_benchmark_checkpoint_pair,
)
from final_model.eval.benchmark_contracts import (
    ACTION_NAMES,
    BENCHMARK_CONTRACT_VERSION,
    EVAL_ACTION_MODES,
    PRIMARY_AOI_CONTRACT,
    SENSITIVITY_AOI_CONTRACT,
    allocate_preview_horizons,
    manuscript_aoi,
    unavailable_model_layout_rows,
    v6_aoi_allocation,
)
from final_model.eval.benchmark_provenance import (
    file_sha256,
    git_provenance,
    model_evaluation_provenance,
)
from final_model.eval.common import CANONICAL_PARTNER, write_csv, write_json
from final_model.eval.gaze_behavior_rollout import (
    EvaluationLearnedGaze,
    RecordingGazePolicy,
    ground_truth_masks,
)
from final_model.eval.temporal_action import (
    CALIBRATED_COMMITMENT_RULES,
    COMMITMENT_RULES,
    PeriodicThresholdDecisionEngine,
    commitment_threshold,
)
from final_model.eval.human_task_events import detect_task_events
from final_model.eval.fixation_duration import (
    FIXATION_DURATION_CONTRACT_ID,
    fixation_duration_metrics,
    split_fixation_window,
)
from final_model.partial_ver.gaze import (
    GazePhase,
    HUMAN_DWELL_RNG_SEED_OFFSET,
    acquisition_tiles,
)
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.memory import MEMORY_RNG_SEED_OFFSET
from final_model.partial_ver.peripheral import PERIPHERAL_RNG_SEED_OFFSET
from final_model.partial_ver.preview_opacity import PREVIEW_OPACITY_RNG_SEED_OFFSET


@dataclass(frozen=True)
class BenchmarkRolloutConfig:
    action_checkpoint: Path
    gaze_checkpoint: Path
    output_dir: Path
    decision_times: tuple[int, ...] = (200, 400, 600)
    horizons: tuple[int, ...] = (0, 2, 4)
    episodes: int = 1
    episode_length: int = 180
    seed_start: int = 810_000
    tau: float = .5
    motor_delay_ms: float = 0.0
    fixation_shape: str = "plus"
    device: str = "cpu"
    eval_action_mode: str = "periodic_20ms_threshold"
    benchmark_contract_version: str = BENCHMARK_CONTRACT_VERSION
    model_name: str = "partial_action_gaze_pair"
    commitment_rule: str = "fixed"
    tau_start: float | None = None
    tau_min: float = 0.0
    tau_slope_per_ms: float | None = None
    tau_kappa_ms: float | None = None


def evaluation_seeds(seed_start: int, episodes: int) -> tuple[int, ...]:
    """Return the one seed list reused in every Time x Horizon cell."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    return tuple(seed_start + index for index in range(episodes))


def _write_gzip_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    fields = sorted({key for row in values for key in row})
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _held_name(player) -> str | None:
    held = getattr(player, "held_object", None)
    return None if held is None else str(held.name)


def _learner_signature(backend) -> tuple[Any, ...]:
    player = backend.state.players[1]
    return (
        tuple(map(int, player.position)),
        tuple(map(int, player.orientation)),
        _held_name(player),
    )


def _global_signature(backend) -> str:
    state = backend.state
    value = state.to_dict() if hasattr(state, "to_dict") else repr(state)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _pot_states(mdp, state) -> dict[str, Any]:
    source = mdp.get_pot_states(state)
    result: dict[str, Any] = {
        "empty": [list(value) for value in source["empty"]]
    }
    for ingredient in ("onion", "tomato"):
        result[ingredient] = {
            category: [list(value) for value in positions]
            for category, positions in source[ingredient].items()
        }
        for position in source[ingredient].get("partially_full", []):
            soup = state.objects.get(tuple(position))
            if soup is None or not isinstance(soup.state, tuple) or len(soup.state) < 2:
                raise RuntimeError("partially full pot has no canonical soup count")
            count = int(soup.state[1])
            result[ingredient].setdefault(f"{count}_items", []).append(
                list(position)
            )
    return result


def _held_json(player) -> str | None:
    held = getattr(player, "held_object", None)
    return None if held is None else json.dumps(held.to_dict(), sort_keys=True)


def _segment_aoi_fields(
    masks, tile, duration_ms: float, fixation_shape: str = "plus"
) -> dict[str, Any]:
    x, y = map(int, tile)
    overlay = bool(masks["preview"][x, y])
    base = "Self" if masks["self"][x, y] else "Partner" if masks["partner"][x, y] else None
    primary = manuscript_aoi(base, overlay)
    v6 = v6_aoi_allocation(base, overlay, duration_ms)
    horizons = [index for index in range(1, 5) if masks[f"h{index}"][x, y]]
    allocated = allocate_preview_horizons(
        duration_ms if primary == "Action Preview" else 0.0, horizons
    )
    plus_tiles = acquisition_tiles(tile, "plus")
    plus_hits = {
        name: any(bool(masks[name.lower()][plus_tile]) for plus_tile in plus_tiles)
        for name in ("Self", "Partner", "Preview")
    }
    plus_hits["Other"] = not any(plus_hits.values())
    footprint_tiles = acquisition_tiles(tile, fixation_shape)
    footprint_hits = {
        name: any(bool(masks[name.lower()][footprint_tile]) for footprint_tile in footprint_tiles)
        for name in ("Self", "Partner", "Preview")
    }
    footprint_hits["Other"] = not any(footprint_hits.values())
    return {
        "aoi_contract": PRIMARY_AOI_CONTRACT,
        "primary_aoi": primary,
        "base_aoi": base,
        "teammate_source": int(bool(masks["partner"][x, y])),
        "preview_source": int(overlay),
        "preview_horizons": json.dumps(horizons),
        "plus_coverage_contract": "nonexclusive_semantic_foveal_plus_v1",
        **{f"plus_{name.lower()}_hit": int(hit) for name, hit in plus_hits.items()},
        "footprint_shape": fixation_shape,
        **{
            f"footprint_{name.lower()}_hit": int(hit)
            for name, hit in footprint_hits.items()
        },
        **{f"primary_{label.replace(' ', '_').lower()}_ms": duration_ms if primary == label else 0.0 for label in ("Player", "Teammate", "Action Preview", "Other")},
        **{f"v6_{label.replace(' ', '_').lower()}_ms": value for label, value in v6.items()},
        **{f"h{index}_allocated_ms": allocated.get(index, 0.0) for index in range(1, 5)},
    }


def _learner_collision_fields(backend, state, partner_action: int, learner_action: int):
    if int(learner_action) not in range(4):
        return {
            "partner_collision": 0,
            "environment_block": 0,
            "collision_any": 0,
        }
    mdp = backend.mdp
    players = state.players
    partner_value = mdp._move_if_direction(
        players[0].position,
        players[0].orientation,
        backend.Action.INDEX_TO_ACTION[int(partner_action)],
    )[0]
    learner_value = mdp._move_if_direction(
        players[1].position,
        players[1].orientation,
        backend.Action.INDEX_TO_ACTION[int(learner_action)],
    )[0]
    old_positions = tuple(player.position for player in players)
    candidate_positions = (partner_value, learner_value)
    partner_collision = bool(
        mdp.is_transition_collision(old_positions, candidate_positions)
    )
    environment_block = bool(
        not partner_collision and learner_value == players[1].position
    )
    return {
        "partner_collision": int(partner_collision),
        "environment_block": int(environment_block),
        "collision_any": int(partner_collision or environment_block),
    }


def _overlap(start, end, left, right):
    return np.maximum(
        0.0,
        np.minimum(np.asarray(end), np.asarray(right))
        - np.maximum(np.asarray(start), np.asarray(left)),
    )


def _decision_behavior_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    decisions = decisions.copy()
    decisions["non_timeout"] = decisions.forced_stay.eq(0)
    decisions["canonical_revision_non_timeout"] = decisions.canonical_had_revision.where(
        decisions.non_timeout
    )
    decisions["no_action"] = decisions.forced_stay | decisions.intentional_stay
    return decisions.groupby(["T", "H"], as_index=False).agg(
        decisions=("decision", "size"),
        non_timeout_decisions=("non_timeout", "sum"),
        selected_action_latency_ms=("selected_action_input_time_ms", "mean"),
        canonical_revisions=("canonical_revision_non_timeout", "sum"),
        canonical_revision_rate=("canonical_revision_non_timeout", "mean"),
        temporal_change_rate=("temporal_had_action_change", "mean"),
        intentional_stay_rate=("intentional_stay", "mean"),
        forced_stay_rate=("forced_stay", "mean"),
        no_action_rate=("no_action", "mean"),
        partner_collision_count=("partner_collision", "sum"),
        partner_collision_rate=("partner_collision", "mean"),
        environment_block_count=("environment_block", "sum"),
        environment_block_rate=("environment_block", "mean"),
        collision_count=("collision_any", "sum"),
        collision_rate=("collision_any", "mean"),
        deliveries=("deliveries", "sum"),
        sparse_team_reward=("sparse_team_reward", "sum"),
        realized_transition_rate=("realized_transition", "mean"),
    )


PLUS_AOI_COLUMNS = {
    "Self": "plus_self_hit",
    "Partner": "plus_partner_hit",
    "Preview": "plus_preview_hit",
    "Other": "plus_other_hit",
}


def _write_plus_aoi_outputs(
    output: Path,
    decisions: pd.DataFrame,
    segments: pd.DataFrame,
) -> None:
    keys = ["T", "H", "episode", "seed", "decision"]
    hit_columns = list(PLUS_AOI_COLUMNS.values())
    decision_info = decisions[keys + [
        "decision_onset_ms", "decision_deadline_ms",
        "selected_action_input_time_ms", "forced_stay",
    ]]
    annotated = segments.merge(decision_info, on=keys, validate="many_to_one")

    coverage = annotated.groupby(keys, as_index=False)[hit_columns].max()
    coverage = decision_info[keys].merge(coverage, on=keys, how="left")
    coverage[hit_columns] = coverage[hit_columns].fillna(0).astype(int)
    _write_gzip_csv(
        output / "raw" / "plus_aoi_decision_coverage.csv.gz",
        coverage.to_dict("records"),
    )
    coverage_long = coverage.melt(
        id_vars=keys, value_vars=hit_columns,
        var_name="aoi_column", value_name="checked",
    )
    inverse = {column: aoi for aoi, column in PLUS_AOI_COLUMNS.items()}
    coverage_long["aoi"] = coverage_long.aoi_column.map(inverse)
    coverage_trials = coverage_long.groupby(
        ["T", "H", "episode", "seed", "aoi"], as_index=False
    ).agg(
        checked_decisions=("checked", "sum"),
        denominator_decisions=("decision", "size"),
    )
    coverage_trials["coverage_fraction"] = (
        coverage_trials.checked_decisions / coverage_trials.denominator_decisions
    )
    coverage_trials["canonical_denominator"] = 180
    coverage_trials["checked_decisions_over_180"] = np.where(
        coverage_trials.denominator_decisions.eq(180),
        coverage_trials.checked_decisions / 180.0,
        np.nan,
    )
    _save_frame(
        coverage_trials,
        output / "summaries" / "plus_aoi_decision_coverage_trials.csv",
    )
    coverage_summary = coverage_long.groupby(
        ["T", "H", "aoi"], as_index=False
    ).agg(
        checked_decisions=("checked", "sum"),
        decisions=("decision", "size"),
    )
    coverage_summary["coverage_fraction"] = (
        coverage_summary.checked_decisions / coverage_summary.decisions
    )
    coverage_summary["metric_family"] = "semantic_foveal_plus_coverage"
    _save_frame(
        coverage_summary,
        output / "summaries" / "plus_aoi_decision_coverage.csv",
    )

    new_fixations = annotated[
        annotated.fixation_start_ms.ge(annotated.decision_onset_ms)
        & annotated.fixation_start_ms.lt(annotated.decision_deadline_ms)
    ].sort_values(keys + ["fixation_start_ms", "fixation_id"])
    new_fixations = new_fixations.drop_duplicates(keys + ["fixation_id"])
    fixation_fields = keys + [
        "fixation_id", "fixation_start_ms", "target_x", "target_y", *hit_columns,
    ]
    _write_gzip_csv(
        output / "raw" / "plus_aoi_fixation_events.csv.gz",
        new_fixations[fixation_fields].to_dict("records"),
    )
    fixation_long = new_fixations[fixation_fields].melt(
        id_vars=keys + ["fixation_id", "fixation_start_ms", "target_x", "target_y"],
        value_vars=hit_columns, var_name="aoi_column", value_name="hit",
    )
    fixation_long["aoi"] = fixation_long.aoi_column.map(inverse)
    full = fixation_long.groupby(["T", "H", "aoi"], as_index=False).agg(
        aoi_hit_fixations=("hit", "sum"),
        all_fixations=("fixation_id", "size"),
    )
    durations = decision_info.assign(
        interval_duration_ms=(
            decision_info.decision_deadline_ms - decision_info.decision_onset_ms
        )
    ).groupby(["T", "H"], as_index=False).interval_duration_ms.sum()
    full = full.merge(durations, on=["T", "H"], validate="many_to_one")
    full["hit_share"] = full.aoi_hit_fixations / full.all_fixations
    full["hit_rate_per_ms"] = full.aoi_hit_fixations / full.interval_duration_ms
    full["hit_rate_per_second"] = full.hit_rate_per_ms * 1000.0
    full["metric_family"] = "semantic_foveal_plus_coverage_not_composition"
    _save_frame(full, output / "summaries" / "plus_aoi_fixation_hits.csv")

    eligible = decision_info[
        decision_info.selected_action_input_time_ms.notna()
        & decision_info.forced_stay.eq(0)
    ].copy()
    eligible["selected_action_input_abs_ms"] = (
        eligible.decision_onset_ms + eligible.selected_action_input_time_ms
    )
    eligible_fixations = new_fixations.merge(
        eligible[keys + ["selected_action_input_abs_ms"]],
        on=keys, how="inner", validate="many_to_one",
    )
    eligible_fixations["phase"] = np.where(
        eligible_fixations.fixation_start_ms.lt(
            eligible_fixations.selected_action_input_abs_ms
        ),
        "pre", "post",
    )
    event_long = eligible_fixations[fixation_fields + ["phase"]].melt(
        id_vars=keys + ["fixation_id", "fixation_start_ms", "target_x", "target_y", "phase"],
        value_vars=hit_columns, var_name="aoi_column", value_name="hit",
    )
    event_long["aoi"] = event_long.aoi_column.map(inverse)
    event_counts = event_long.groupby(keys + ["phase", "aoi"], as_index=False).agg(
        aoi_hit_fixations=("hit", "sum"),
        all_fixations=("fixation_id", "size"),
    )
    phase_grid = eligible[keys + ["selected_action_input_time_ms"]].merge(
        pd.DataFrame({"phase": ["pre", "post"]}), how="cross"
    ).merge(pd.DataFrame({"aoi": list(PLUS_AOI_COLUMNS)}), how="cross")
    phase_grid["interval_duration_ms"] = np.where(
        phase_grid.phase.eq("pre"),
        phase_grid.selected_action_input_time_ms,
        phase_grid["T"] - phase_grid.selected_action_input_time_ms,
    )
    pre_post = phase_grid.merge(
        event_counts, on=keys + ["phase", "aoi"], how="left"
    )
    pre_post[["aoi_hit_fixations", "all_fixations"]] = pre_post[
        ["aoi_hit_fixations", "all_fixations"]
    ].fillna(0).astype(int)
    pre_post["hit_share"] = np.where(
        pre_post.all_fixations.gt(0),
        pre_post.aoi_hit_fixations / pre_post.all_fixations,
        np.nan,
    )
    pre_post["hit_rate_per_ms"] = np.where(
        pre_post.interval_duration_ms.gt(0),
        pre_post.aoi_hit_fixations / pre_post.interval_duration_ms,
        np.nan,
    )
    pre_post["hit_rate_per_second"] = pre_post.hit_rate_per_ms * 1000.0
    _write_gzip_csv(
        output / "raw" / "plus_aoi_pre_post_decisions.csv.gz",
        pre_post.to_dict("records"),
    )
    pre_post_summary = pre_post.groupby(
        ["T", "H", "phase", "aoi"], as_index=False
    ).agg(
        aoi_hit_fixations=("aoi_hit_fixations", "sum"),
        all_fixations=("all_fixations", "sum"),
        interval_duration_ms=("interval_duration_ms", "sum"),
        eligible_decisions=("decision", "size"),
    )
    pre_post_summary["hit_share"] = (
        pre_post_summary.aoi_hit_fixations / pre_post_summary.all_fixations
    )
    pre_post_summary["hit_rate_per_ms"] = (
        pre_post_summary.aoi_hit_fixations
        / pre_post_summary.interval_duration_ms
    )
    pre_post_summary["hit_rate_per_second"] = (
        pre_post_summary.hit_rate_per_ms * 1000.0
    )
    pre_post_summary["metric_family"] = (
        "semantic_foveal_plus_coverage_not_composition"
    )
    _save_frame(
        pre_post_summary,
        output / "summaries" / "plus_aoi_pre_post.csv",
    )
    changes = pre_post_summary.pivot(
        index=["T", "H", "aoi"], columns="phase",
        values=["hit_share", "hit_rate_per_ms", "hit_rate_per_second"],
    )
    changes.columns = [f"{metric}_{phase}" for metric, phase in changes.columns]
    changes = changes.reset_index()
    for metric in ("hit_share", "hit_rate_per_ms", "hit_rate_per_second"):
        changes[f"{metric}_post_minus_pre"] = (
            changes[f"{metric}_post"] - changes[f"{metric}_pre"]
        )
    changes["metric_family"] = "semantic_foveal_plus_coverage_not_composition"
    _save_frame(
        changes,
        output / "summaries" / "plus_aoi_pre_post_change.csv",
    )


def _write_model_summaries(output: Path, decision_rows, gaze_rows, episode_rows) -> None:
    summaries = output / "summaries"
    decisions = pd.DataFrame(decision_rows)
    trace = pd.DataFrame(gaze_rows)
    timeline = trace[
        trace.row_type.isin(("fixation_segment", "saccade_segment"))
    ].copy()
    keys = ["T", "H", "episode", "seed", "decision"]

    decision_gaze = []
    for values, frame in timeline.groupby(keys, sort=False):
        metrics = fixation_duration_metrics(
            frame.to_dict("records"), aoi_field="primary_aoi"
        )
        expected = float(values[0])
        if not np.isclose(
            metrics["accounted_gaze_duration_ms"], expected, atol=1e-8
        ):
            raise RuntimeError("strict gaze segments do not cover decision window")
        decision_gaze.append({
            **dict(zip(keys, values)),
            **metrics,
        })
    decision_gaze_frame = pd.DataFrame(decision_gaze)
    _save_frame(
        decision_gaze_frame,
        summaries / "fixation_duration_by_decision.csv",
    )
    episode_keys = ["T", "H", "episode", "seed"]
    duration_columns = [
        "fixation_duration_ms_self", "fixation_duration_ms_teammate",
        "fixation_duration_ms_preview", "fixation_duration_ms_other",
        "saccade_duration_ms", "classified_fixation_duration_ms",
        "accounted_gaze_duration_ms",
    ]
    episode_gaze = decision_gaze_frame.groupby(
        episode_keys, as_index=False
    )[duration_columns].sum()
    for aoi in ("self", "teammate", "preview", "other"):
        episode_gaze[f"fixation_prop_{aoi}"] = np.where(
            episode_gaze.classified_fixation_duration_ms.gt(0),
            episode_gaze[f"fixation_duration_ms_{aoi}"]
            / episode_gaze.classified_fixation_duration_ms,
            0.0,
        )
    _save_frame(
        episode_gaze,
        summaries / "fixation_duration_by_episode.csv",
    )
    segments = timeline[timeline.row_type.eq("fixation_segment")].copy()

    behavior = _decision_behavior_summary(decisions)
    _save_frame(behavior, summaries / "decision_behavior.csv")
    task_trials = decisions.assign(
        no_action=decisions.forced_stay | decisions.intentional_stay,
        non_timeout=decisions.forced_stay.eq(0),
    ).groupby(["T", "H", "episode", "seed"], as_index=False).agg(
        decisions=("decision", "size"),
        deliveries=("deliveries", "sum"),
        sparse_team_reward=("sparse_team_reward", "sum"),
        selected_action_responses=("non_timeout", "sum"),
        selected_action_response_time_ms=("selected_action_input_time_ms", "mean"),
        forced_timeout_rate=("forced_stay", "mean"),
        intentional_stay_rate=("intentional_stay", "mean"),
        no_action_rate=("no_action", "mean"),
        partner_collision_count=("partner_collision", "sum"),
        partner_collision_rate=("partner_collision", "mean"),
        environment_block_count=("environment_block", "sum"),
        environment_block_rate=("environment_block", "mean"),
        collision_count=("collision_any", "sum"),
        collision_rate=("collision_any", "mean"),
    )
    _save_frame(task_trials, summaries / "task_behavior_trials.csv")
    _write_plus_aoi_outputs(output, decisions, segments)

    attached = segments.merge(
        decisions[keys + [
            "decision_onset_ms", "decision_deadline_ms",
            "literal_first_threshold_crossing_time_ms",
            "selected_action_input_time_ms",
        ]], on=keys, validate="many_to_one",
    )
    first_start = attached.decision_onset_ms + attached.literal_first_threshold_crossing_time_ms
    selected_start = attached.decision_onset_ms + attached.selected_action_input_time_ms
    first_start = first_start.fillna(attached.decision_deadline_ms)
    selected_start = selected_start.fillna(attached.decision_deadline_ms)
    attached["post_first_ms"] = _overlap(
        attached.segment_start_ms, attached.segment_end_ms,
        first_start, attached.decision_deadline_ms,
    )
    attached["post_selected_ms"] = _overlap(
        attached.segment_start_ms, attached.segment_end_ms,
        selected_start, attached.decision_deadline_ms,
    )
    episode_group = ["T", "H", "episode", "seed"]
    post = attached.groupby(episode_group, as_index=False)[
        ["duration_ms", "post_first_ms", "post_selected_ms"]
    ].sum()
    post["post_first_input_fraction"] = post.post_first_ms / post.duration_ms
    post["post_selected_action_fraction"] = post.post_selected_ms / post.duration_ms
    post_summary = post.groupby(["T", "H"], as_index=False)[[
        "post_first_input_fraction", "post_selected_action_fraction"
    ]].mean()
    _save_frame(post_summary, summaries / "post_input_gaze.csv")

    aoi_columns = [
        "primary_player_ms", "primary_teammate_ms",
        "primary_action_preview_ms", "primary_other_ms",
    ]
    aoi = attached.groupby(episode_group, as_index=False)[aoi_columns + ["duration_ms"]].sum()
    for column in aoi_columns:
        aoi[column.replace("_ms", "_share")] = aoi[column] / aoi.duration_ms
    _save_frame(aoi, summaries / "aoi_full_window.csv")
    aoi_summary = aoi.groupby(["T", "H"], as_index=False)[
        [column.replace("_ms", "_share") for column in aoi_columns]
    ].mean()

    v6_columns = [
        "v6_player_ms", "v6_teammate_ms", "v6_action_preview_ms", "v6_other_ms"
    ]
    v6 = attached.groupby(episode_group, as_index=False)[v6_columns + ["duration_ms"]].sum()
    for column in v6_columns:
        v6[column.replace("_ms", "_share")] = v6[column] / v6.duration_ms
    _save_frame(v6, summaries / "aoi_full_window_v6_sensitivity.csv")

    fixed_end = np.minimum(attached.decision_onset_ms + 200, attached.decision_deadline_ms)
    attached["initial_200ms"] = _overlap(
        attached.segment_start_ms, attached.segment_end_ms,
        attached.decision_onset_ms, fixed_end,
    )
    attached["carryover"] = attached.fixation_start_ms.lt(attached.decision_onset_ms)
    attached["ai_related"] = attached.primary_aoi.isin(["Teammate", "Action Preview"])
    carry = attached[attached.ai_related].copy()
    carry["carryover_ms"] = np.where(carry.carryover, carry.initial_200ms, 0.0)
    carry["new_ms"] = np.where(~carry.carryover, carry.initial_200ms, 0.0)
    carry = carry.groupby(episode_group, as_index=False)[
        ["initial_200ms", "carryover_ms", "new_ms"]
    ].sum()
    denominator = attached.groupby(episode_group, as_index=False).initial_200ms.sum().rename(columns={"initial_200ms": "classified_ms"})
    carry = denominator.merge(carry, on=episode_group, how="left").fillna(0)
    carry["carryover_share"] = carry.carryover_ms / carry.classified_ms
    carry["new_share"] = carry.new_ms / carry.classified_ms
    _save_frame(carry, summaries / "carryover.csv")

    horizon_columns = [f"h{index}_allocated_ms" for index in range(1, 5)]
    horizon = attached.groupby(episode_group, as_index=False)[horizon_columns + ["duration_ms"]].sum()
    horizon["preview_ms"] = horizon[horizon_columns].sum(axis=1)
    horizon["preview_share"] = horizon.preview_ms / horizon.duration_ms
    horizon["per_represented_step_share"] = horizon.preview_share / horizon.H.replace(0, np.nan)
    _save_frame(horizon, summaries / "preview_horizon.csv")

    visual_rows = []
    saccade_rows = []
    check_rows = []
    fixation_table = segments.sort_values(
        ["T", "H", "episode", "seed", "fixation_id"]
    ).drop_duplicates(["T", "H", "episode", "seed", "fixation_id"])
    fixation_table[["previous_target_x", "previous_target_y"]] = fixation_table.groupby(
        ["T", "H", "episode", "seed"]
    )[["target_x", "target_y"]].shift()
    previous = fixation_table[[
        "T", "H", "episode", "seed", "fixation_id",
        "previous_target_x", "previous_target_y",
    ]]
    attached = attached.merge(
        previous,
        on=["T", "H", "episode", "seed", "fixation_id"],
        how="left", validate="many_to_one",
    )
    for window, end in (
        ("initial_200ms", np.minimum(attached.decision_onset_ms + 200, attached.decision_deadline_ms)),
        ("full_window", attached.decision_deadline_ms),
    ):
        new = attached[
            attached.fixation_start_ms.gt(attached.decision_onset_ms)
            & attached.fixation_start_ms.lt(end)
        ]
        counts = new.groupby(keys).fixation_id.nunique().rename("new_fixations").reset_index()
        base = decisions[keys].merge(counts, on=keys, how="left").fillna({"new_fixations": 0})
        base["window"] = window
        base["at_least_1"] = base.new_fixations.ge(1).astype(int)
        base["at_least_2"] = base.new_fixations.ge(2).astype(int)
        visual_rows.append(base)
        first = new.sort_values("fixation_start_ms").drop_duplicates(keys)
        first = first[first.previous_target_x.notna() & first.previous_target_y.notna()].copy()
        first["euclidean_tiles"] = np.hypot(
            first.target_x - first.previous_target_x,
            first.target_y - first.previous_target_y,
        )
        first["distance_gt_3"] = first.euclidean_tiles.gt(3).astype(int)
        first["window"] = window
        saccade_rows.append(first[keys + ["window", "euclidean_tiles", "distance_gt_3"]])

        window_ms = _overlap(
            attached.segment_start_ms, attached.segment_end_ms,
            attached.decision_onset_ms, end,
        )
        phase = attached.copy()
        phase["window_ms"] = window_ms
        phase = phase[
            phase.fixation_start_ms.ge(phase.decision_onset_ms)
            & phase.fixation_start_ms.lt(end)
            & phase.window_ms.gt(0)
        ]
        sources = []
        for label, source in (("Teammate_check", "teammate_source"), ("Preview_check", "preview_source")):
            part = phase[phase[source].eq(1)][keys + ["fixation_id", "window_ms"]].copy()
            part["source"] = label
            sources.append(part)
        source = pd.concat(sources, ignore_index=True)
        dwell = source.groupby(keys + ["fixation_id", "source"], as_index=False).window_ms.sum()
        strict = dwell[dwell.window_ms.ge(60)].groupby(keys + ["source"]).size().unstack(fill_value=0).reset_index()
        for label in ("Teammate_check", "Preview_check"):
            if label not in strict:
                strict[label] = 0
            strict[label] = strict[label].gt(0).astype(int)
        strict["Any_AI"] = strict[["Teammate_check", "Preview_check"]].max(axis=1)
        strict = decisions[keys].merge(strict, on=keys, how="left").fillna(0)
        strict["window"] = window
        check_rows.append(strict)
    visual = pd.concat(visual_rows, ignore_index=True)
    _save_frame(
        visual.groupby(["window", "T", "H"], as_index=False).agg(
            decisions=("new_fixations", "size"),
            mean_new_fixations=("new_fixations", "mean"),
            at_least_1=("at_least_1", "mean"), at_least_2=("at_least_2", "mean"),
        ), summaries / "visual_sampling.csv",
    )
    saccades = pd.concat(saccade_rows, ignore_index=True)
    _save_frame(
        saccades.groupby(["window", "T", "H"], as_index=False).agg(
            evaluable=("distance_gt_3", "size"),
            mean_euclidean_tiles=("euclidean_tiles", "mean"),
            distance_gt_3=("distance_gt_3", "mean"),
        ), summaries / "first_new_saccade.csv",
    )
    checks = pd.concat(check_rows, ignore_index=True)
    _save_frame(
        checks.groupby(["window", "T", "H"], as_index=False).agg(
            decisions=("Any_AI", "size"), teammate_rate=("Teammate_check", "mean"),
            preview_rate=("Preview_check", "mean"), any_ai_rate=("Any_AI", "mean"),
        ), summaries / "context_checks.csv",
    )
    _save_frame(pd.DataFrame(episode_rows), summaries / "performance.csv")
    reference = []
    for row in behavior.itertuples(index=False):
        for metric in (
            "selected_action_latency_ms", "canonical_revision_rate",
            "intentional_stay_rate", "forced_stay_rate", "no_action_rate",
        ):
            non_timeout_metric = metric in {
                "selected_action_latency_ms", "canonical_revision_rate"
            }
            reference.append({
                "source": "model", "Time": row.T, "Horizon": row.H,
                "aoi_contract": None, "metric": metric,
                "value": getattr(row, metric),
                "numerator": row.canonical_revisions
                if metric == "canonical_revision_rate" else None,
                "denominator": row.non_timeout_decisions
                if non_timeout_metric else row.decisions,
            })
    for row in post_summary.itertuples(index=False):
        for metric in ("post_first_input_fraction", "post_selected_action_fraction"):
            reference.append({
                "source": "model", "Time": row.T, "Horizon": row.H,
                "aoi_contract": PRIMARY_AOI_CONTRACT, "metric": metric,
                "value": getattr(row, metric), "numerator": None,
                "denominator": None,
            })
    for _, row in aoi_summary.iterrows():
        for metric in [column.replace("_ms", "_share") for column in aoi_columns]:
            reference.append({
                "source": "model", "Time": row["T"], "Horizon": row["H"],
                "aoi_contract": PRIMARY_AOI_CONTRACT, "metric": metric,
                "value": row[metric], "numerator": None,
                "denominator": None,
            })
    _save_frame(pd.DataFrame(reference), output / "benchmark/model_reference.csv")


def _save_frame(frame, path):
    frame.to_csv(path, index=False)


def _format_runtime(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _validate_config(config: BenchmarkRolloutConfig) -> None:
    if config.eval_action_mode not in EVAL_ACTION_MODES:
        raise ValueError(f"eval_action_mode must be one of {EVAL_ACTION_MODES}")
    mode_contracts = {
        "periodic_20ms_threshold": "contract_v2",
        "periodic_20ms_margin_threshold": "contract_v3",
        "periodic_20ms_margin_motor_delay": "contract_v4",
        "periodic_20ms_margin_motor_delay_stochastic_gaze": "contract_v5",
        "periodic_20ms_margin_threshold_stochastic_gaze": "contract_v6",
        "periodic_20ms_logit_commitment_stochastic_gaze": "contract_v7",
        "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze": "contract_v8",
        "periodic_20ms_no_commitment_stochastic_gaze": "contract_v9",
    }
    if config.eval_action_mode not in mode_contracts:
        raise ValueError("benchmark_rollout implements periodic mode; use existing evaluator for legacy mode")
    expected_contract = mode_contracts[config.eval_action_mode]
    if config.benchmark_contract_version != expected_contract:
        raise ValueError(
            f"{config.eval_action_mode} requires {expected_contract}, got "
            f"{config.benchmark_contract_version}"
        )
    if not np.isfinite(config.motor_delay_ms) or config.motor_delay_ms < 0.0:
        raise ValueError("motor_delay_ms must be finite and nonnegative")
    if config.eval_action_mode in {
        "periodic_20ms_margin_motor_delay",
        "periodic_20ms_margin_motor_delay_stochastic_gaze",
    }:
        if not np.isclose(config.tau, .18):
            raise ValueError("motor-delay contracts require global margin tau=0.18")
        if not np.isclose(config.motor_delay_ms, 87.607603):
            raise ValueError(
                "motor-delay contracts require motor_delay_ms=87.607603"
            )
    elif config.eval_action_mode == "periodic_20ms_margin_threshold_stochastic_gaze":
        if not np.isclose(config.tau, .18):
            raise ValueError("no-motor-delay stochastic-gaze contract requires tau=0.18")
        if config.motor_delay_ms != 0.0:
            raise ValueError("no-motor-delay stochastic-gaze contract requires motor_delay_ms=0")
    elif config.eval_action_mode == "periodic_20ms_logit_commitment_stochastic_gaze":
        if not np.isclose(config.motor_delay_ms, 87.607603):
            raise ValueError("logit commitment contract requires motor_delay_ms=87.607603")
        if config.commitment_rule not in CALIBRATED_COMMITMENT_RULES:
            raise ValueError(
                "v7 commitment_rule must be one of "
                f"{CALIBRATED_COMMITMENT_RULES}"
            )
        PeriodicThresholdDecisionEngine(
            tau=config.tau,
            threshold_rule="top1_top2_logit_margin",
            motor_delay_ms=config.motor_delay_ms,
            commitment_rule=config.commitment_rule,
            tau_start=config.tau_start,
            tau_min=config.tau_min,
            tau_slope_per_ms=config.tau_slope_per_ms,
            tau_kappa_ms=config.tau_kappa_ms,
        )
    elif (
        config.eval_action_mode
        == "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze"
    ):
        if config.commitment_rule not in CALIBRATED_COMMITMENT_RULES:
            raise ValueError(
                "v8 commitment_rule must be one of "
                f"{CALIBRATED_COMMITMENT_RULES}"
            )
        PeriodicThresholdDecisionEngine(
            tau=config.tau,
            threshold_rule="top1_top2_logit_margin",
            motor_delay_ms=config.motor_delay_ms,
            commitment_rule=config.commitment_rule,
            tau_start=config.tau_start,
            tau_min=config.tau_min,
            tau_slope_per_ms=config.tau_slope_per_ms,
            tau_kappa_ms=config.tau_kappa_ms,
        )
    elif config.eval_action_mode == "periodic_20ms_no_commitment_stochastic_gaze":
        if config.commitment_rule != "none":
            raise ValueError("no-commitment mode requires commitment_rule=none")
        if config.tau != 0.0:
            raise ValueError("no-commitment mode requires the ignored tau placeholder=0")
        if config.motor_delay_ms != 0.0:
            raise ValueError("no-commitment mode requires action_input_delay_ms=0")
        PeriodicThresholdDecisionEngine(
            tau=config.tau,
            threshold_rule="top1_top2_logit_margin",
            motor_delay_ms=0.0,
            commitment_rule="none",
        )
    elif config.motor_delay_ms != 0.0:
        raise ValueError("motor delay requires a periodic margin motor-delay mode")
    if config.episodes <= 0 or config.episode_length <= 0:
        raise ValueError("episodes and episode_length must be positive")
    if any(value not in (200, 400, 600) for value in config.decision_times):
        raise ValueError("invalid decision time")
    if any(value not in (0, 2, 4) for value in config.horizons):
        raise ValueError("invalid horizon")


def run_benchmark_rollout(
    config: BenchmarkRolloutConfig,
    *,
    pair: BenchmarkCheckpointPair | None = None,
) -> dict[str, Any]:
    _validate_config(config)
    git = git_provenance()
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    summaries = output / "summaries"
    benchmark = output / "benchmark"
    raw.mkdir(); summaries.mkdir(); benchmark.mkdir()
    loaded = pair or load_benchmark_checkpoint_pair(
        action_checkpoint=config.action_checkpoint,
        gaze_checkpoint=config.gaze_checkpoint,
        device=config.device,
    )
    action = loaded.action
    repository = FixedHSPPartnerRepository(
        device=action.device, checkpoint=Path(CANONICAL_PARTNER).resolve()
    )
    partner = repository.provenance()
    if action.metadata.get("fixed_partner_sha256") != partner["fixed_partner_sha256"]:
        raise RuntimeError("fixed partner differs from ActionNet training contract")

    query_rows: list[dict[str, Any]] = []
    gaze_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    model_step_rows: list[dict[str, Any]] = []
    threshold_rule = (
        "top1_top2_logit_margin"
        if config.eval_action_mode in {
            "periodic_20ms_logit_commitment_stochastic_gaze",
            "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze",
            "periodic_20ms_no_commitment_stochastic_gaze",
        }
        else "top1_top2_margin"
        if config.eval_action_mode in {
            "periodic_20ms_margin_threshold",
            "periodic_20ms_margin_motor_delay",
            "periodic_20ms_margin_motor_delay_stochastic_gaze",
            "periodic_20ms_margin_threshold_stochastic_gaze",
        }
        else "max_probability"
    )
    engine = PeriodicThresholdDecisionEngine(
        tau=config.tau,
        threshold_rule=threshold_rule,
        motor_delay_ms=config.motor_delay_ms,
        commitment_rule=config.commitment_rule,
        tau_start=config.tau_start,
        tau_min=config.tau_min,
        tau_slope_per_ms=config.tau_slope_per_ms,
        tau_kappa_ms=config.tau_kappa_ms,
    )
    seeds = evaluation_seeds(config.seed_start, config.episodes)
    stochastic_gaze = config.eval_action_mode in {
        "periodic_20ms_margin_motor_delay_stochastic_gaze",
        "periodic_20ms_margin_threshold_stochastic_gaze",
        "periodic_20ms_logit_commitment_stochastic_gaze",
        "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze",
        "periodic_20ms_no_commitment_stochastic_gaze",
    }
    run_started = time.perf_counter()
    total_episodes = (
        len(config.decision_times) * len(config.horizons) * len(seeds)
    )
    total_planned_decisions = total_episodes * config.episode_length
    completed_episodes = 0
    completed_planned_decisions = 0
    progress_interval = min(10, config.episode_length)
    print(
        f"[BENCHMARK] episodes={total_episodes} "
        f"planned_decisions={total_planned_decisions} "
        f"cells={len(config.decision_times) * len(config.horizons)}",
        flush=True,
    )

    for decision_time in config.decision_times:
        for horizon in config.horizons:
            for episode, seed in enumerate(seeds):
                backend = DirectOvercookedBackend(
                    seed=seed, horizon=horizon, episode_length=config.episode_length,
                    device=action.device, partner_repository=repository,
                )
                gaze = EvaluationLearnedGaze(
                    loaded.gaze_actor, action.device,
                    deterministic=not stochastic_gaze,
                    seed=seed + 77_003,
                )
                request_runtime: dict[str, Any] = {"decision": 0, "onset": 0.0}

                def on_gaze_request(policy, target):
                    gaze_rows.append({
                        "row_type": "request",
                        "T": decision_time, "H": horizon, "episode": episode,
                        "seed": seed, "decision": request_runtime["decision"],
                        "absolute_timestamp_ms": float(adapter.current_time_ms),
                        "decision_relative_timestamp_ms": float(adapter.current_time_ms - request_runtime["onset"]),
                        "source_x": int(policy.last_fixation[0]),
                        "source_y": int(policy.last_fixation[1]),
                        "target_x": int(target[0]), "target_y": int(target[1]),
                    })

                adapter = PartialObservationAdapter(
                    backend, decision_time=decision_time,
                    fixation_shape=config.fixation_shape,
                    gaze_policy=RecordingGazePolicy(gaze, on_gaze_request),
                    memory_lifetime_ms=action.metadata["memory_lifetime_ms"],
                    memory_decay_mode=action.metadata["memory_decay_mode"],
                    memory_tau_ms=action.metadata.get("memory_tau_ms"),
                    memory_seed=(
                        seed + MEMORY_RNG_SEED_OFFSET
                        if action.metadata["memory_decay_mode"]
                        == "stochastic_exponential"
                        else None
                    ),
                    constraint_config=action.constraint_config,
                    peripheral_seed=seed + PERIPHERAL_RNG_SEED_OFFSET,
                    opacity_seed=seed + PREVIEW_OPACITY_RNG_SEED_OFFSET,
                    dwell_seed=(
                        seed + HUMAN_DWELL_RNG_SEED_OFFSET
                        if action.constraint_config.emma.mode == "human_stochastic"
                        else None
                    ),
                    ground_truth_policy_input=bool(
                        action.metadata.get("ground_truth_policy_input", False)
                    ),
                )
                # Evaluation trace retention only; it does not alter scheduler state.
                adapter.events = deque()
                event_cursor = 0
                phase_events: list[dict[str, Any]] = []
                phase_event_cursor = 0
                original_advance_event = adapter.gaze.advance_event

                def recording_advance_event():
                    fixation_before = tuple(adapter.gaze.current_fixation)
                    event = original_advance_event()
                    phase_events.append({
                        "timestamp_ms": float(event.timestamp_ms),
                        "completed_phase": event.completed_phase.value,
                        "fixation_before": fixation_before,
                    })
                    return event

                adapter.gaze.advance_event = recording_advance_event
                physical_fixation_id = 0
                physical_fixation_start = 0.0
                hidden = np.zeros((1, 64), dtype=np.float32)
                mask = np.zeros((1,), dtype=np.float32)
                actions = []
                sparse = 0.0
                deliveries = 0
                partner_collisions = 0
                environment_blocks = 0
                episode_started = time.perf_counter()
                backend.verify_partner_parameters()

                for decision in range(config.episode_length):
                    onset = float(adapter.current_time_ms)
                    deadline = onset + decision_time
                    request_runtime.update(decision=decision, onset=onset)
                    masks = ground_truth_masks(backend)
                    pre_hidden = hidden.copy()
                    pre_mask = mask.copy()
                    pre_learner = _learner_signature(backend)
                    pre_global = _global_signature(backend)
                    acquisitions = []

                    def capture_advance(timestamp):
                        nonlocal event_cursor
                        adapter.advance_to(timestamp)
                        events = list(adapter.events)
                        for event in events[event_cursor:]:
                            completed = str(event.get("completed_phase", ""))
                            event_time = float(event["timestamp_ms"])
                            event_kind = event.get("event")
                            row = {
                                "row_type": (
                                    "acquisition"
                                    if event.get("acquisition")
                                    else "held_foveal_refresh"
                                    if event_kind
                                    == "held_foveal_post_environment_step_refresh"
                                    else "phase_completion"
                                ),
                                "T": decision_time, "H": horizon,
                                "episode": episode, "seed": seed,
                                "decision": decision,
                                "absolute_timestamp_ms": event_time,
                                "decision_relative_timestamp_ms": event_time - onset,
                                "completed_phase": completed,
                                "acquisition": int(bool(event.get("acquisition"))),
                                "held_foveal_refresh": int(bool(
                                    event.get("held_foveal_refresh", False)
                                )),
                                "fixation_x": int(adapter.gaze.current_fixation[0]),
                                "fixation_y": int(adapter.gaze.current_fixation[1]),
                                "memory_updated_before_actionnet": int(bool(event.get("memory_updated_before_actionnet", False))),
                            }
                            gaze_rows.append(row)
                            if event.get("acquisition"):
                                acquisitions.append(event_time)
                        event_cursor = len(events)

                    def query_actionnet(absolute_timestamp):
                        observation, _critic, available = adapter.current()
                        if threshold_rule == "top1_top2_logit_margin":
                            probabilities, logits, candidate = (
                                read_only_logits_probabilities_and_hidden(
                                    action.actor, observation, pre_hidden[None],
                                    pre_mask[None], available,
                                )
                            )
                            return {
                                "probabilities": probabilities,
                                "logits": logits,
                                "candidate_hidden": candidate[0].copy(),
                                "available_actions": available,
                            }
                        probabilities, candidate = read_only_probabilities_and_hidden(
                            action.actor, observation, pre_hidden[None],
                            pre_mask[None], available,
                        )
                        return {
                            "probabilities": probabilities,
                            "candidate_hidden": candidate[0].copy(),
                            "available_actions": available,
                        }

                    selected = engine.run(
                        decision_onset_ms=onset,
                        decision_time_ms=decision_time,
                        advance_to=capture_advance,
                        query_actionnet=query_actionnet,
                    )
                    decision_phase_segments = split_fixation_window(
                        start_ms=onset,
                        end_ms=deadline,
                        phase_events=phase_events[phase_event_cursor:],
                        final_phase=adapter.gaze.phase.value,
                        final_fixation=adapter.gaze.current_fixation,
                    )
                    phase_event_cursor = len(phase_events)
                    for segment in decision_phase_segments:
                        tile = (
                            int(segment["fixation_x"]),
                            int(segment["fixation_y"]),
                        )
                        duration = float(segment["duration_ms"])
                        row = {
                            "row_type": (
                                "fixation_segment"
                                if segment["included_in_fixation_duration"]
                                else "saccade_segment"
                            ),
                            "T": decision_time, "H": horizon,
                            "episode": episode, "seed": seed,
                            "decision": decision,
                            "fixation_id": physical_fixation_id,
                            "fixation_start_ms": physical_fixation_start,
                            "target_x": tile[0], "target_y": tile[1],
                            **segment,
                        }
                        if segment["included_in_fixation_duration"]:
                            row.update(_segment_aoi_fields(
                                masks, tile, duration, config.fixation_shape
                            ))
                        gaze_rows.append(row)
                        if segment["gaze_phase"] == GazePhase.EXEC.value:
                            physical_fixation_id += 1
                            physical_fixation_start = float(
                                segment["segment_end_ms"]
                            )

                    for query_index, row in enumerate(selected.queries):
                        query_rows.append({
                            "model": config.model_name,
                            "action_checkpoint_sha256": action.checkpoint_sha256,
                            "gaze_checkpoint_sha256": loaded.gaze.checkpoint_sha256,
                            "tau": config.tau,
                            "T": decision_time, "H": horizon,
                            "episode": episode, "seed": seed,
                            "decision": decision, "query_index": query_index,
                            **row,
                        })
                    collected = {
                        "proposals": [],
                        "latest_proposal": (
                            None if selected.forced_stay else {
                                "action": selected.executed_action,
                                "candidate_hidden": selected.last_query_candidate_hidden,
                            }
                        ),
                    }
                    partner_observation, partner_available = backend.partner_batch_input()
                    partner_hash_before = backend.begin_partner_step()
                    partner_action = backend.partners.action(
                        partner_observation, partner_available
                    )
                    pre_state = backend.state
                    pre_state_json = json.dumps(
                        pre_state.to_dict(), sort_keys=True, default=str
                    )
                    pre_pot_states = json.dumps(
                        _pot_states(backend.mdp, pre_state), sort_keys=True
                    )
                    collision = _learner_collision_fields(
                        backend, pre_state, partner_action, selected.executed_action
                    )
                    partner_collisions += collision["partner_collision"]
                    environment_blocks += collision["environment_block"]
                    learner_held = _held_json(pre_state.players[1])
                    partner_held = _held_json(pre_state.players[0])
                    transition = adapter.commit_collected(
                        collected, partner_action=partner_action,
                        partner_hash_before=partner_hash_before,
                    )
                    # Consume the post-transition held-fovea refresh event in
                    # this decision; otherwise the next decision's cursor
                    # would treat it as pre-existing trace history.
                    capture_advance(deadline)
                    post_state_json = json.dumps(
                        backend.state.to_dict(), sort_keys=True, default=str
                    )
                    hidden = np.asarray(selected.last_query_candidate_hidden).copy()
                    mask.fill(1.0)
                    post_learner = _learner_signature(backend)
                    post_global = _global_signature(backend)
                    realized = pre_learner != post_learner
                    actions.append(selected.executed_action)
                    sparse += float(transition["sparse_team_reward"])
                    deliveries += int(transition["deliveries"])
                    decision_rows.append({
                        "model": config.model_name,
                        "action_checkpoint_sha256": action.checkpoint_sha256,
                        "gaze_checkpoint_sha256": loaded.gaze.checkpoint_sha256,
                        "pairing_status": loaded.status,
                        "tau": config.tau,
                        "motor_delay_ms": config.motor_delay_ms,
                        "action_input_delay_ms": config.motor_delay_ms,
                        "commitment_rule": config.commitment_rule,
                        "tau_start": config.tau_start,
                        "tau_min": config.tau_min,
                        "tau_slope_per_ms": config.tau_slope_per_ms,
                        "tau_kappa_ms": config.tau_kappa_ms,
                        "layout": "counter_circuit",
                        "T": decision_time, "H": horizon,
                        "episode": episode, "seed": seed, "decision": decision,
                        "decision_onset_ms": onset, "decision_deadline_ms": deadline,
                        "literal_first_threshold_crossing_time_ms": selected.literal_first_threshold_crossing_time_ms,
                        "first_selected_action": selected.first_selected_action,
                        "first_action_input_time_ms": selected.first_action_input_time_ms,
                        "selected_action_input_time_ms": selected.selected_action_input_time_ms,
                        "selected_action_input_source": selected.selected_action_input_source,
                        "final_selected_action": selected.final_selected_action,
                        "executed_proposal_query_time_ms": selected.executed_proposal_query_time_ms,
                        "executed_proposal_arrival_time_ms": selected.executed_proposal_arrival_time_ms,
                        "selected_action": selected.final_selected_action,
                        "executed_action": selected.executed_action,
                        "executed_action_name": ACTION_NAMES[selected.executed_action],
                        "action_change_count": selected.action_change_count,
                        "temporal_had_action_change": int(selected.temporal_had_action_change),
                        "canonical_had_revision": int(selected.canonical_had_revision),
                        "intentional_stay": int(selected.intentional_stay),
                        "forced_stay": int(selected.forced_stay),
                        **collision,
                        "realized_transition": int(realized),
                        "joint_state_transition": int(pre_global != post_global),
                        "acquisition_count": len(acquisitions),
                        "acquisition_times_ms": json.dumps([value - onset for value in acquisitions]),
                        "query_count": len(selected.queries),
                        "sparse_team_reward": float(transition["sparse_team_reward"]),
                        "deliveries": int(transition["deliveries"]),
                    })
                    transition_rows.append({
                        "model": config.model_name, "layout": "counter_circuit",
                        "T": decision_time, "H": horizon,
                        "episode": episode, "seed": seed, "decision": decision,
                        "partner_action": int(partner_action),
                        "partner_action_name": ACTION_NAMES[int(partner_action)],
                        "learner_action": selected.executed_action,
                        "learner_action_name": ACTION_NAMES[selected.executed_action],
                        "pre_state_json": pre_state_json,
                        "post_state_json": post_state_json,
                        "sparse_team_reward": float(transition["sparse_team_reward"]),
                        "deliveries": int(transition["deliveries"]),
                        "canonical_task_event_detector_status": "raw_transition_ready",
                    })
                    model_step_rows.append({
                        "pid": f"MODEL_SEED_{seed}",
                        "trial_id": f"T{decision_time}_H{horizon}_seed{seed}_episode{episode}",
                        "step": decision,
                        "human_action": ACTION_NAMES[selected.executed_action].lower(),
                        "agent_action": ACTION_NAMES[int(partner_action)].lower(),
                        "env_state": pre_state_json,
                        "human_held_object": learner_held,
                        "agent_held_object": partner_held,
                        "pot_states": pre_pot_states,
                        "reward": float(transition["sparse_team_reward"]),
                        "first_human_input_ms": (
                            np.nan
                            if selected.first_action_input_time_ms is None
                            else onset + selected.first_action_input_time_ms
                        ),
                        "Time": decision_time, "Preview": horizon,
                        "Layout": "counter_circuit", "trial_order_c": 0.0,
                    })
                    decisions_in_episode = decision + 1
                    if (
                        decisions_in_episode % progress_interval == 0
                        or transition["done"]
                    ):
                        completed_decisions = (
                            completed_planned_decisions + decisions_in_episode
                        )
                        run_elapsed = time.perf_counter() - run_started
                        decisions_per_second = completed_decisions / max(
                            run_elapsed, 1e-12
                        )
                        remaining_decisions = max(
                            total_planned_decisions - completed_decisions, 0
                        )
                        eta_seconds = remaining_decisions / max(
                            decisions_per_second, 1e-12
                        )
                        print(
                            f"[PROGRESS] {completed_decisions}/"
                            f"{total_planned_decisions} "
                            f"({completed_decisions / total_planned_decisions:.2%}) "
                            f"episodes={completed_episodes}/{total_episodes} "
                            f"T={decision_time} H={horizon} "
                            f"episode={episode + 1}/{config.episodes} seed={seed} "
                            f"elapsed={_format_runtime(run_elapsed)} "
                            f"eta={_format_runtime(eta_seconds)}",
                            flush=True,
                        )
                    if transition["done"]:
                        break

                backend.verify_partner_parameters()
                elapsed = time.perf_counter() - episode_started
                episode_rows.append({
                    "model": config.model_name, "layout": "counter_circuit",
                    "T": decision_time, "H": horizon,
                    "episode": episode, "seed": seed,
                    "decisions": len(actions), "runtime_seconds": elapsed,
                    "runtime_seconds_per_decision": elapsed / max(len(actions), 1),
                    "deliveries": deliveries, "score": sparse,
                    "sparse_team_reward": sparse,
                    "partner_collision_count": partner_collisions,
                    "environment_block_count": environment_blocks,
                    "collision_count": partner_collisions + environment_blocks,
                    "partner_collision_rate": partner_collisions / max(len(actions), 1),
                    "environment_block_rate": environment_blocks / max(len(actions), 1),
                    "collision_rate": (
                        partner_collisions + environment_blocks
                    ) / max(len(actions), 1),
                    "deliveries_per_100_decisions": deliveries / max(len(actions), 1) * 100,
                    "simulated_decision_time_ms": len(actions) * decision_time,
                    "points_per_simulated_minute": sparse / max(len(actions) * decision_time / 60_000, 1e-12),
                    "action_trajectory_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
                })
                completed_episodes += 1
                completed_planned_decisions += config.episode_length
                print(
                    f"[EPISODE] {completed_episodes}/{total_episodes} "
                    f"({completed_episodes / total_episodes:.2%}) "
                    f"T={decision_time} H={horizon} "
                    f"episode={episode + 1}/{config.episodes} seed={seed} "
                    f"decisions={len(actions)} runtime={_format_runtime(elapsed)} "
                    f"deliveries={deliveries} reward={sparse:.6f}",
                    flush=True,
                )

    action.assert_unchanged(); loaded.gaze.assert_unchanged()
    _write_gzip_csv(raw / "action_queries.csv.gz", query_rows)
    _write_gzip_csv(raw / "gaze_trace.csv.gz", gaze_rows)
    write_csv(raw / "decision_log.csv", decision_rows)
    write_csv(raw / "episode_log.csv", episode_rows)
    _write_gzip_csv(raw / "task_transition_log.csv.gz", transition_rows)
    _write_model_summaries(output, decision_rows, gaze_rows, episode_rows)
    model_steps = pd.DataFrame(model_step_rows)
    model_trials = model_steps[[
        "pid", "trial_id", "Time", "Preview", "Layout", "trial_order_c"
    ]].drop_duplicates("trial_id")
    model_events = detect_task_events(model_trials, model_steps)
    task_summary_rows = []
    for event_type, frame in model_events.items():
        _write_gzip_csv(
            raw / f"task_events_{event_type}.csv.gz", frame.to_dict("records")
        )
        if frame.empty:
            task_summary_rows.append({
                "event_type": event_type, "T": None, "H": None,
                "opportunities": 0, "successes": 0, "rate": None,
            })
            continue
        task_summary_rows.extend(
            frame.groupby(["Time", "Preview"], as_index=False).agg(
                opportunities=("outcome", "size"),
                successes=("outcome", "sum"), rate=("outcome", "mean"),
            ).rename(columns={"Time": "T", "Preview": "H"}).assign(
                event_type=event_type
            ).to_dict("records")
        )
    write_csv(summaries / "task_events.csv", task_summary_rows)
    seed_manifest = {
        "seed_start": config.seed_start,
        "reuse_rule": "same seed index reused in every Time x Horizon cell",
        "seeds": list(seeds),
    }
    seed_manifest_path = output / "seed_manifest.json"
    write_json(seed_manifest_path, seed_manifest)
    provenance = model_evaluation_provenance(
        git=git,
        action_checkpoint_sha256=action.checkpoint_sha256,
        gaze_checkpoint_sha256=loaded.gaze.checkpoint_sha256,
        tau=config.tau,
        motor_delay_ms=config.motor_delay_ms,
        seed_manifest_sha256=file_sha256(seed_manifest_path),
        evaluation_action_mode=config.eval_action_mode,
        benchmark_contract_version=config.benchmark_contract_version,
        commitment_rule=(
            config.commitment_rule
            if threshold_rule == "top1_top2_logit_margin" else None
        ),
        tau_start=config.tau_start,
        tau_min=config.tau_min,
        tau_slope_per_ms=config.tau_slope_per_ms,
        tau_kappa_ms=config.tau_kappa_ms,
    )
    provenance.update({
        "fixation_duration_contract_id": FIXATION_DURATION_CONTRACT_ID,
        "fixation_aoi_definition": "mutually_exclusive_central_tile",
        "fixation_aoi_priority": ["preview", "self", "teammate", "other"],
        "fixation_denominator": "classified_fixation_duration_excluding_exec",
        "zero_classified_fixation_convention": "all_fixation_proportions_zero",
    })
    config_record = {
        **config.__dict__,
        **provenance,
        "action_checkpoint": str(Path(config.action_checkpoint).resolve()),
        "gaze_checkpoint": str(Path(config.gaze_checkpoint).resolve()),
        "output_dir": str(output),
        "aoi_primary_contract": PRIMARY_AOI_CONTRACT,
        "aoi_sensitivity_contract": SENSITIVITY_AOI_CONTRACT,
        "gaze_evaluation_mode": (
            "stochastic" if stochastic_gaze else "deterministic"
        ),
        "action_selection_mode": "deterministic_top1",
        "recurrent_state_contract": "all within-decision queries share pre-decision hidden; final query candidate commits",
        "calibration_status": "PROVISIONAL" if loaded.status != "CONFIRMED_EXACT_PAIR" else "PAIR_CONFIRMED_TAU_NOT_CALIBRATED",
    }
    write_json(output / "config.json", config_record)
    write_json(benchmark / "evaluation_manifest.json", provenance)
    write_json(benchmark / "checkpoint_pair.json", loaded.manifest())
    if threshold_rule == "top1_top2_logit_margin":
        write_csv(benchmark / "threshold_trajectory.csv", [
            {
                "rule": config.commitment_rule,
                "T": decision_time,
                "t_ms": elapsed,
                "tau": (
                    None
                    if config.commitment_rule == "none"
                    else commitment_threshold(
                        rule=config.commitment_rule,
                        elapsed_ms=elapsed,
                        tau=config.tau,
                        tau_start=config.tau_start,
                        tau_min=config.tau_min,
                        tau_slope_per_ms=config.tau_slope_per_ms,
                        tau_kappa_ms=config.tau_kappa_ms,
                    )
                ),
            }
            for decision_time in config.decision_times
            for elapsed in range(0, decision_time, 20)
        ])
    write_csv(
        summaries / "layout.csv",
        unavailable_model_layout_rows(
            model=config.model_name,
            checkpoint=action.checkpoint_sha256,
            layouts=("counter_circuit", "unident_s_hard_onion", "double_rings"),
        ),
    )
    summary = {
        **provenance,
        "runtime_seconds": time.perf_counter() - run_started,
        "episodes": len(episode_rows),
        "decisions": len(decision_rows),
        "queries": len(query_rows),
        "gaze_rows": len(gaze_rows),
        "pairing_status": loaded.status,
        "parameter_hash_preserved": True,
    }
    write_json(benchmark / "benchmark_report.json", summary)
    print(
        f"[COMPLETE] episodes={len(episode_rows)} "
        f"decisions={len(decision_rows)} "
        f"runtime={_format_runtime(summary['runtime_seconds'])} "
        f"output={output}",
        flush=True,
    )
    return {"output_dir": output, "summary": summary, "episodes": episode_rows}
