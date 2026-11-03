"""Offline opportunity-controlled AOI analysis for saved human/model traces.

The analysis deliberately reuses the matched-footprint contract used by the
30-seed benchmark: overlapping Self/Teammate/Preview hits, with Other defined
only when none of those three sources is inside the footprint.  It never loads
a policy checkpoint and never performs a rollout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


AOIS = ("Self", "Teammate", "Preview", "Other")
HIT_COLUMNS = {
    "Self": "self_hit",
    "Teammate": "teammate_hit",
    "Preview": "preview_hit",
    "Other": "other_hit",
}
TIMES = (200, 400, 600)
HORIZONS = (0, 2, 4)


def _position(value: object) -> tuple[int, int] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        return None
    return int(parsed[0]), int(parsed[1])


def _human_targets(row: pd.Series) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    state = json.loads(row.env_state)
    teammate = tuple(int(v) for v in state["players"][0]["position"])
    self_position = tuple(int(v) for v in state["players"][1]["position"])
    preview: set[tuple[int, int]] = set()
    if isinstance(row.displayed_future_trajectory, str):
        trajectory = json.loads(row.displayed_future_trajectory)
        for entry in trajectory.get("entries", []):
            value = entry.get("overlay_position", entry.get("tile"))
            if value is not None:
                preview.add(tuple(int(v) for v in value))
    return self_position, teammate, preview


def _inside(
    fixation: tuple[int, int] | None,
    target: tuple[int, int] | set[tuple[int, int]],
    shape: str,
) -> bool:
    if fixation is None:
        return False
    targets = target if isinstance(target, set) else {target}
    fx, fy = fixation
    for tx, ty in targets:
        dx, dy = abs(tx - fx), abs(ty - fy)
        if shape == "plus" and dx + dy <= 1:
            return True
        if shape == "exact3x3" and dx <= 1 and dy <= 1:
            return True
    return False


def load_human(
    processed_root: Path, benchmark_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = pd.read_csv(benchmark_root / "raw/visual_sampling_decisions.csv")
    canonical = canonical[canonical.window.eq("full_window")].copy()
    canonical = canonical.rename(columns={"Time": "T", "Preview": "H"})
    canonical["decision_uid"] = (
        canonical.pid.astype(str) + "|" + canonical.trial_id.astype(str)
        + "|" + canonical.step.astype(str)
    )
    if canonical.decision_uid.duplicated().any():
        raise RuntimeError("duplicate human full-window decision")

    state_frames: list[pd.DataFrame] = []
    segment_frames: list[pd.DataFrame] = []
    for pid in sorted(canonical.pid.unique()):
        participant = processed_root / str(pid)
        state_frames.append(pd.read_csv(
            participant / "decision_dataset.csv",
            usecols=["pid", "trial_id", "step", "env_state", "displayed_future_trajectory"],
        ))
        segment_frames.append(pd.read_csv(
            participant / "decision_fixation_segments.csv",
            usecols=[
                "pid", "trial_id", "step", "fixation_id", "window_type",
                "decision_start_ms", "decision_end_ms", "fixation_start_ms",
                "overlap_start_ms", "overlap_duration_ms", "base_aoi",
                "overlay_flag", "base_position",
            ],
        ))
    states = pd.concat(state_frames, ignore_index=True)
    states["decision_uid"] = (
        states.pid.astype(str) + "|" + states.trial_id.astype(str)
        + "|" + states.step.astype(str)
    )
    states = states[states.decision_uid.isin(canonical.decision_uid)].copy()
    if states.decision_uid.duplicated().any() or len(states) != len(canonical):
        raise RuntimeError("human state rows do not map one-to-one to eligible decisions")
    targets = {
        row.decision_uid: _human_targets(row)
        for row in states.itertuples(index=False)
    }

    segments = pd.concat(segment_frames, ignore_index=True)
    segments = segments[segments.window_type.eq("full_window")].copy()
    segments["decision_uid"] = (
        segments.pid.astype(str) + "|" + segments.trial_id.astype(str)
        + "|" + segments.step.astype(str)
    )
    segments = segments[segments.decision_uid.isin(canonical.decision_uid)].copy()
    segments = segments.merge(
        canonical[["decision_uid", "T", "H"]], on="decision_uid", how="left",
        validate="many_to_one",
    )

    annotated: list[pd.DataFrame] = []
    for comparison, shape in (("plus", "plus"), ("exact3x3", "exact3x3")):
        frame = segments.copy()
        values = []
        for row in frame.itertuples(index=False):
            fixation = _position(row.base_position)
            self_position, teammate, preview = targets[row.decision_uid]
            # Preserve the canonical annotation at the center; state/trajectory
            # positions extend that annotation to neighboring footprint tiles.
            self_hit = row.base_aoi == "Self" or _inside(fixation, self_position, shape)
            teammate_hit = row.base_aoi == "Partner" or _inside(fixation, teammate, shape)
            preview_hit = bool(row.overlay_flag) or _inside(fixation, preview, shape)
            other_hit = not (self_hit or teammate_hit or preview_hit)
            values.append((self_hit, teammate_hit, preview_hit, other_hit))
        frame[["self_hit", "teammate_hit", "preview_hit", "other_hit"]] = np.asarray(
            values, dtype=np.int8
        )
        frame["comparison"] = comparison
        frame["model"] = "Human"
        frame["footprint_shape"] = "plus_5_tile" if shape == "plus" else "exact_3x3_9_tile"
        frame["cluster"] = frame.pid.astype(str)
        frame["trial_unit"] = frame.pid.astype(str) + "|" + frame.trial_id.astype(str)
        frame["segment_start_ms"] = frame.overlap_start_ms
        frame["duration_ms"] = frame.overlap_duration_ms
        annotated.append(frame)
    human_segments = pd.concat(annotated, ignore_index=True)

    decisions = []
    for comparison, shape in (("plus", "plus_5_tile"), ("exact3x3", "exact_3x3_9_tile")):
        frame = canonical[[
            "pid", "trial_id", "step", "decision_uid", "T", "H", "new_fixations"
        ]].copy()
        frame["comparison"] = comparison
        frame["model"] = "Human"
        frame["footprint_shape"] = shape
        frame["cluster"] = frame.pid.astype(str)
        frame["episode_unit"] = frame.pid.astype(str) + "|" + frame.trial_id.astype(str)
        decisions.append(frame)
    return pd.concat(decisions, ignore_index=True), human_segments


def load_model(root: Path, *, label: str, comparison: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = pd.read_csv(root / "raw/decision_log.csv", usecols=[
        "T", "H", "episode", "seed", "decision", "decision_onset_ms",
        "decision_deadline_ms",
    ])
    decisions["decision_uid"] = (
        decisions["T"].astype(str) + "|" + decisions["H"].astype(str)
        + "|" + decisions.seed.astype(str) + "|" + decisions.episode.astype(str)
        + "|" + decisions.decision.astype(str)
    )
    if decisions.decision_uid.duplicated().any():
        raise RuntimeError(f"duplicate model decisions in {root}")
    trace = pd.read_csv(root / "raw/gaze_trace.csv.gz")
    segments = trace[trace.row_type.eq("fixation_segment")].copy()
    segments["decision_uid"] = (
        segments["T"].astype(int).astype(str) + "|" + segments["H"].astype(int).astype(str)
        + "|" + segments.seed.astype(int).astype(str) + "|" + segments.episode.astype(int).astype(str)
        + "|" + segments.decision.astype(int).astype(str)
    )
    segments = segments.merge(
        decisions[["decision_uid", "decision_onset_ms", "decision_deadline_ms"]],
        on="decision_uid", how="left", validate="many_to_one",
    )
    segments["decision_start_ms"] = segments.decision_onset_ms
    segments["decision_end_ms"] = segments.decision_deadline_ms
    if comparison == "plus":
        source = {
            "self_hit": "plus_self_hit", "teammate_hit": "plus_partner_hit",
            "preview_hit": "plus_preview_hit", "other_hit": "plus_other_hit",
        }
        shape = "plus_5_tile"
    else:
        source = {
            "self_hit": "footprint_self_hit", "teammate_hit": "footprint_partner_hit",
            "preview_hit": "footprint_preview_hit", "other_hit": "footprint_other_hit",
        }
        shape = "exact_3x3_9_tile"
        if not segments.footprint_shape.eq("3x3").all():
            raise RuntimeError("3x3 run does not contain exact configured-footprint logging")
    for target, origin in source.items():
        segments[target] = segments[origin].astype(np.int8)
    segments["comparison"] = comparison
    segments["model"] = label
    segments["footprint_shape"] = shape
    segments["cluster"] = segments.seed.astype(int).astype(str)
    segments["trial_unit"] = (
        segments.seed.astype(int).astype(str) + "|" + segments.episode.astype(int).astype(str)
    )

    decisions["comparison"] = comparison
    decisions["model"] = label
    decisions["footprint_shape"] = shape
    decisions["cluster"] = decisions.seed.astype(int).astype(str)
    decisions["episode_unit"] = (
        decisions.seed.astype(int).astype(str) + "|" + decisions.episode.astype(int).astype(str)
    )
    return decisions, segments


def new_fixations(segments: pd.DataFrame) -> pd.DataFrame:
    new = segments[
        segments.fixation_start_ms.gt(segments.decision_start_ms)
        & segments.fixation_start_ms.lt(segments.decision_end_ms)
    ].copy() if "decision_start_ms" in segments else segments[
        segments.fixation_start_ms.gt(segments.decision_onset_ms)
        & segments.fixation_start_ms.lt(segments.decision_deadline_ms)
    ].copy()
    keys = ["comparison", "model", "decision_uid", "fixation_id"]
    new = new.sort_values(keys + ["segment_start_ms"])
    return new.drop_duplicates(keys, keep="first")


def attach_outcomes(
    decisions: pd.DataFrame, segments: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    keys = ["comparison", "model", "decision_uid"]
    counts = events.groupby(keys, observed=True).size().rename("derived_new_fixations")
    coverage = segments.groupby(keys, observed=True)[list(HIT_COLUMNS.values())].max()
    result = decisions.merge(counts, on=keys, how="left").merge(
        coverage, on=keys, how="left"
    )
    result["derived_new_fixations"] = result.derived_new_fixations.fillna(0).astype(int)
    for column in HIT_COLUMNS.values():
        result[column] = result[column].fillna(0).astype(np.int8)
    human = result.model.eq("Human")
    mismatch = human & result.new_fixations.ne(result.derived_new_fixations)
    if mismatch.any():
        raise RuntimeError(f"human canonical new-fixation mismatch: {int(mismatch.sum())}")
    result["n_new_fixations"] = result.derived_new_fixations
    return result


def _summary_slices(frame: pd.DataFrame):
    for (t, h), subset in frame.groupby(["T", "H"], observed=True):
        yield "T_H_cell", int(t), int(h), "specific_H", subset
    for t, subset in frame.groupby("T", observed=True):
        yield "T_marginal", int(t), np.nan, "all_H", subset
    restricted = frame[frame.H.isin([2, 4])]
    for t, subset in restricted.groupby("T", observed=True):
        yield "T_marginal", int(t), np.nan, "H2_H4", subset


def per_fixation_table(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparison, model, shape), frame in events.groupby(
        ["comparison", "model", "footprint_shape"], observed=True
    ):
        for aggregation, t, h, h_subset, subset in _summary_slices(frame):
            row = {
                "comparison": comparison, "model": model, "footprint_shape": shape,
                "aggregation": aggregation, "T": t, "H": h, "h_subset": h_subset,
                "new_fixations": len(subset), "clusters": subset.cluster.nunique(),
            }
            for aoi, column in HIT_COLUMNS.items():
                row[f"{aoi}_hit_fixations"] = int(subset[column].sum())
                row[f"{aoi}_per_fix"] = float(subset[column].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def first_fixation_table(decisions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    keys = ["comparison", "model", "decision_uid"]
    first = events.sort_values(keys + ["fixation_start_ms", "segment_start_ms"]).drop_duplicates(keys)
    rows = []
    for (comparison, model, shape), frame in decisions.groupby(
        ["comparison", "model", "footprint_shape"], observed=True
    ):
        eligible = first[(first.comparison == comparison) & (first.model == model)]
        for aggregation, t, h, h_subset, subset in _summary_slices(frame):
            ids = set(subset.decision_uid)
            observed = eligible[eligible.decision_uid.isin(ids)]
            row = {
                "comparison": comparison, "model": model, "footprint_shape": shape,
                "aggregation": aggregation, "T": t, "H": h, "h_subset": h_subset,
                "decisions_total": len(subset), "decisions_with_new_fix": len(observed),
                "excluded_no_new_fix": len(subset) - len(observed),
                "excluded_fraction": 1.0 - len(observed) / len(subset),
                "clusters": subset.cluster.nunique(),
            }
            for aoi, column in HIT_COLUMNS.items():
                row[aoi] = float(observed[column].mean()) if len(observed) else np.nan
                row[f"{aoi}_hits"] = int(observed[column].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def fit_adjusted_models(decisions: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable

    rows = []
    for (comparison, model, shape), frame in decisions.groupby(
        ["comparison", "model", "footprint_shape"], observed=True
    ):
        for aoi, outcome in HIT_COLUMNS.items():
            subset = frame.copy()
            if aoi == "Preview":
                subset = subset[subset.H.isin([2, 4])].copy()
                h_subset = "H2_H4"
                h_term = "C(H, Treatment(reference=2))"
            else:
                h_subset = "all_H"
                h_term = "C(H, Treatment(reference=0))"
            formula = (
                f"{outcome} ~ C(T, Treatment(reference=200)) + {h_term}"
                " + n_new_fixations"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = smf.gee(
                    formula, groups="cluster", data=subset,
                    family=sm.families.Binomial(), cov_struct=Exchangeable(),
                ).fit(maxiter=100)
            for coefficient in result.params.index:
                estimate = float(result.params[coefficient])
                robust_se = float(result.bse[coefficient])
                rows.append({
                    "comparison": comparison, "model": model,
                    "footprint_shape": shape, "aoi": aoi, "h_subset": h_subset,
                    "formula": formula, "n_obs": len(subset),
                    "n_clusters": subset.cluster.nunique(), "cluster": "participant"
                    if model == "Human" else "seed",
                    "working_correlation": "Exchangeable", "coefficient": coefficient,
                    "estimate": estimate, "robust_se": robust_se,
                    "odds_ratio": math.exp(estimate),
                    "ci_low": math.exp(estimate - 1.96 * robust_se),
                    "ci_high": math.exp(estimate + 1.96 * robust_se),
                    "p_value": float(result.pvalues[coefficient]),
                    "interpretation": "fixation-count-adjusted association",
                })
    return pd.DataFrame(rows)


def fixed_count_table(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparison, model, shape), frame in decisions.groupby(
        ["comparison", "model", "footprint_shape"], observed=True
    ):
        for count in (1, 2):
            stratum = frame[frame.n_new_fixations.eq(count)]
            for aggregation, t, h, h_subset, subset in _summary_slices(stratum):
                row = {
                    "comparison": comparison, "model": model, "footprint_shape": shape,
                    "n_new_fixations": count, "aggregation": aggregation, "T": t,
                    "H": h, "h_subset": h_subset, "decisions": len(subset),
                    "clusters": subset.cluster.nunique(),
                    "statistics": "descriptive_only",
                }
                for aoi, column in HIT_COLUMNS.items():
                    row[aoi] = float(subset[column].mean()) if len(subset) else np.nan
                    row[f"{aoi}_covered_decisions"] = int(subset[column].sum())
                rows.append(row)
    return pd.DataFrame(rows)


def dwell_table(segments: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    for aoi, column in HIT_COLUMNS.items():
        segments[f"{aoi}_dwell"] = segments.duration_ms * segments[column]
    dwell_columns = [f"{aoi}_dwell" for aoi in AOIS]
    pieces = []
    for (comparison, model, shape), frame in segments.groupby(
        ["comparison", "model", "footprint_shape"], observed=True
    ):
        trial = frame.groupby(
            ["cluster", "trial_unit", "T", "H"], observed=True
        )[dwell_columns + ["duration_ms"]].sum().reset_index()
        for aoi in AOIS:
            trial[f"{aoi}_share"] = trial[f"{aoi}_dwell"] / trial.duration_ms
        if model == "Human":
            unit = trial.groupby(["cluster", "T", "H"], observed=True)[
                [f"{aoi}_share" for aoi in AOIS]
            ].mean().reset_index()
        else:
            unit = trial
        unit["comparison"] = comparison
        unit["model"] = model
        unit["footprint_shape"] = shape
        pieces.append(unit)
    units = pd.concat(pieces, ignore_index=True)

    rows = []
    for (comparison, model, shape), frame in units.groupby(
        ["comparison", "model", "footprint_shape"], observed=True
    ):
        for aggregation, t, h, h_subset, subset in _summary_slices(frame):
            row = {
                "comparison": comparison, "model": model, "footprint_shape": shape,
                "aggregation": aggregation, "T": t, "H": h, "h_subset": h_subset,
                "analysis_units": len(subset), "clusters": subset.cluster.nunique(),
            }
            for aoi in AOIS:
                share = float(subset[f"{aoi}_share"].mean())
                row[f"{aoi}_share"] = share
                row[f"{aoi}_dwell_ms_per_decision"] = share * t
            rows.append(row)
    return pd.DataFrame(rows)


def _metric_delta(
    table: pd.DataFrame, comparison: str, model: str, subset_name: str,
    column: str,
) -> tuple[float, float, float]:
    rows = table[
        table.comparison.eq(comparison) & table.model.eq(model)
        & table.aggregation.eq("T_marginal") & table.h_subset.eq(subset_name)
    ].set_index("T")
    start, finish = float(rows.at[200, column]), float(rows.at[600, column])
    return start, finish, finish - start


def _direction(delta: float, tolerance: float) -> str:
    if delta > tolerance:
        return "increase"
    if delta < -tolerance:
        return "decrease"
    return "flat"


def opportunity_summary(
    per_fix: pd.DataFrame, first: pd.DataFrame, adjusted: pd.DataFrame,
    dwell: pd.DataFrame, decisions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    entities = per_fix[["comparison", "model", "footprint_shape"]].drop_duplicates()
    for entity in entities.itertuples(index=False):
        for aoi in AOIS:
            h_subset = "H2_H4" if aoi == "Preview" else "all_H"
            dwell_200, dwell_600, dwell_delta = _metric_delta(
                dwell, entity.comparison, entity.model, h_subset,
                f"{aoi}_dwell_ms_per_decision",
            )
            outcome = HIT_COLUMNS[aoi]
            raw_frame = decisions[
                decisions.comparison.eq(entity.comparison)
                & decisions.model.eq(entity.model)
            ].copy()
            if h_subset == "H2_H4":
                raw_frame = raw_frame[raw_frame.H.isin([2, 4])]
            cluster_cells = raw_frame.groupby(
                ["cluster", "T", "H"], observed=True
            )[outcome].mean().reset_index()
            raw_by_t = cluster_cells.groupby("T", observed=True)[outcome].mean()
            raw_200, raw_400, raw_600 = (
                float(raw_by_t.loc[t]) for t in TIMES
            )
            raw_delta = raw_600 - raw_200
            pf_200, pf_600, pf_delta = _metric_delta(
                per_fix, entity.comparison, entity.model, h_subset, f"{aoi}_per_fix"
            )
            first_200, first_600, first_delta = _metric_delta(
                first, entity.comparison, entity.model, h_subset, aoi
            )
            coefficient = adjusted[
                adjusted.comparison.eq(entity.comparison)
                & adjusted.model.eq(entity.model) & adjusted.aoi.eq(aoi)
                & adjusted.coefficient.str.contains(r"\[T\.600\]", regex=True)
            ].iloc[0]
            adjusted_direction = "not_detected"
            if coefficient.p_value < 0.05:
                adjusted_direction = "positive" if coefficient.estimate > 0 else "negative"
            raw_direction = _direction(raw_delta, 0.01)
            per_fix_direction = _direction(pf_delta, 0.01)
            first_direction = _direction(first_delta, 0.01)
            if (
                raw_direction == "increase" and per_fix_direction == "increase"
                and first_direction == "increase" and adjusted_direction == "positive"
            ):
                classification = "selective increase"
            elif raw_direction == "increase" and (
                per_fix_direction == "decrease" or first_direction == "decrease"
                or adjusted_direction == "negative"
            ):
                classification = "opposite direction"
            elif raw_direction == "increase" and adjusted_direction == "not_detected":
                classification = "opportunity-driven increase"
            elif raw_direction == "flat":
                classification = "flat"
            else:
                classification = "flat"
            rows.append({
                "comparison": entity.comparison, "model": entity.model,
                "footprint_shape": entity.footprint_shape, "aoi": aoi,
                "h_subset": h_subset, "raw_decision_coverage_T200": raw_200,
                "raw_decision_coverage_T400": raw_400,
                "raw_decision_coverage_T600": raw_600,
                "raw_decision_coverage_delta": raw_delta,
                "raw_T_trend": raw_direction,
                "absolute_dwell_T200_ms": dwell_200,
                "absolute_dwell_T600_ms": dwell_600,
                "absolute_dwell_delta_ms": dwell_delta,
                "per_fixation_T200": pf_200, "per_fixation_T600": pf_600,
                "per_fixation_delta": pf_delta, "per_fixation_T_trend": per_fix_direction,
                "first_fixation_T200": first_200, "first_fixation_T600": first_600,
                "first_fixation_delta": first_delta, "first_fixation_T_trend": first_direction,
                "adjusted_T600_vs_T200_log_odds": coefficient.estimate,
                "adjusted_T600_vs_T200_odds_ratio": coefficient.odds_ratio,
                "adjusted_T600_vs_T200_p": coefficient.p_value,
                "controlled": adjusted_direction,
                "classification": classification,
                "absolute_coverage_increases": raw_direction == "increase",
                "per_fixation_probability_increases": per_fix_direction == "increase",
                "first_fixation_probability_increases": first_direction == "increase",
                "adjusted_T_association_remains": adjusted_direction != "not_detected",
                "claim_scope": "fixation-count-adjusted association; not a causal effect",
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--human-root", type=Path,
        default=Path("final_model/eval/results/human_benchmark_20260901_contract_v2"),
    )
    parser.add_argument(
        "--plus-root", type=Path,
        default=Path("final_model/eval/results/humanfix_plus_local_gaze7m_v8_exponential_30ep_tiefix_20260905"),
    )
    parser.add_argument("--plus-label", default="+7M")
    parser.add_argument(
        "--grid-root", type=Path,
        default=Path("final_model/eval/results/humanfix_exact3x3_gaze5m_v8_exponential_30ep_footprint3x3_20260905"),
    )
    parser.add_argument("--grid-label", default="3x3 5M")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("final_model/eval/results/humanfix_opportunity_controlled_aoi_20260905"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    human_decisions, human_segments = load_human(args.processed_root, args.human_root)
    plus_decisions, plus_segments = load_model(
        args.plus_root, label=args.plus_label, comparison="plus"
    )
    grid_decisions, grid_segments = load_model(
        args.grid_root, label=args.grid_label, comparison="exact3x3"
    )
    decisions = pd.concat([human_decisions, plus_decisions, grid_decisions], ignore_index=True)
    segments = pd.concat([human_segments, plus_segments, grid_segments], ignore_index=True)
    events = new_fixations(segments)
    decisions = attach_outcomes(decisions, segments, events)

    per_fix = per_fixation_table(events)
    first = first_fixation_table(decisions, events)
    adjusted = fit_adjusted_models(decisions)
    strata = fixed_count_table(decisions)
    dwell = dwell_table(segments, decisions)
    summary = opportunity_summary(per_fix, first, adjusted, dwell, decisions)
    outputs = {
        "per_fixation_aoi_probability.csv": per_fix,
        "first_new_fixation_aoi.csv": first,
        "fixation_count_adjusted_models.csv": adjusted,
        "fixed_fixation_count_strata.csv": strata,
        "normalized_footprint_dwell.csv": dwell,
        "opportunity_controlled_summary.csv": summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    plus = summary[summary.comparison.eq("plus")]
    print(f"                 Human                         {args.plus_label}")
    print("AOI        raw T trend   controlled     raw T trend   controlled")
    for aoi in AOIS:
        human = plus[(plus.model == "Human") & (plus.aoi == aoi)].iloc[0]
        model = plus[(plus.model == args.plus_label) & (plus.aoi == aoi)].iloc[0]
        print(
            f"{aoi:<11}{human.raw_T_trend:<14}{human.controlled:<15}"
            f"{model.raw_T_trend:<14}{model.controlled}"
        )


if __name__ == "__main__":
    main()
