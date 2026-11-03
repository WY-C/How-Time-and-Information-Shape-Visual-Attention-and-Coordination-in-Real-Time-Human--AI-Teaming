"""Compare T=200/400/600 pairwise under the existing AOI contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    HIT_COLUMNS,
    attach_outcomes,
    dwell_table,
    first_fixation_table,
    load_human,
    load_model,
    new_fixations,
    per_fixation_table,
)
from final_model.eval.summarize_early_checkpoint import sha256


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "final_model/eval/results"
OUTPUT = (
    RESULTS
    / "benchmark_20m_emmatuned_decay_comparison_v8_5seed_20260906"
    / "pairwise_T_contrasts"
)
CONTRASTS = ((200, 400), (400, 600), (200, 600))
MODELS = (
    (
        "StagedPlusLocal20M",
        RESULTS / "benchmark_20m_staged_pluslocal_v8_5seed_20260906",
    ),
    (
        "EmmaTunedDecay2400_20M",
        RESULTS / "benchmark_20m_emmatuned_decay2400_noperipheral_v8_5seed_20260906",
    ),
    (
        "EmmaTunedOverDecay1200_20M",
        RESULTS / "benchmark_20m_emmatuned_overdecay1200_noperipheral_v8_5seed_20260906",
    ),
)


def _values(table: pd.DataFrame, column: str, h_subset: str) -> dict[int, float]:
    subset = table[
        table.aggregation.eq("T_marginal") & table.h_subset.eq(h_subset)
    ]
    return {
        int(row.T): float(getattr(row, column))
        for row in subset.itertuples(index=False)
    }


def _adjusted_contrasts(decisions: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable

    rows = []
    for (model, shape), frame in decisions.groupby(
        ["model", "footprint_shape"], observed=True
    ):
        for aoi, outcome in HIT_COLUMNS.items():
            subset = frame.copy()
            if aoi == "Preview":
                subset = subset[subset.H.isin([2, 4])].copy()
                h_subset = "H2_H4"
                h_reference = 2
            else:
                h_subset = "all_H"
                h_reference = 0
            formula = (
                f"{outcome} ~ C(T, Treatment(reference=200)) + "
                f"C(H, Treatment(reference={h_reference})) + n_new_fixations"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.gee(
                    formula,
                    groups="cluster",
                    data=subset,
                    family=sm.families.Binomial(),
                    cov_struct=Exchangeable(),
                ).fit(maxiter=100)
            names = list(fit.params.index)
            covariance = np.asarray(fit.cov_params())
            for low, high in CONTRASTS:
                contrast = np.zeros(len(names), dtype=float)
                if high != 200:
                    contrast[names.index(
                        f"C(T, Treatment(reference=200))[T.{high}]"
                    )] += 1.0
                if low != 200:
                    contrast[names.index(
                        f"C(T, Treatment(reference=200))[T.{low}]"
                    )] -= 1.0
                estimate = float(contrast @ np.asarray(fit.params))
                robust_se = float(np.sqrt(contrast @ covariance @ contrast))
                z_value = estimate / robust_se
                rows.append(
                    {
                        "model": model,
                        "footprint_shape": shape,
                        "aoi": aoi,
                        "h_subset": h_subset,
                        "T_from": low,
                        "T_to": high,
                        "interval_ms": high - low,
                        "n_obs": len(subset),
                        "n_clusters": subset.cluster.nunique(),
                        "cluster": "participant" if model == "Human" else "seed",
                        "formula": formula,
                        "log_odds_contrast": estimate,
                        "robust_se": robust_se,
                        "odds_ratio": math.exp(estimate),
                        "ci_low": math.exp(estimate - 1.96 * robust_se),
                        "ci_high": math.exp(estimate + 1.96 * robust_se),
                        "p_value": math.erfc(abs(z_value) / math.sqrt(2.0)),
                    }
                )
    return pd.DataFrame(rows)


def _aoi_contrasts(
    decisions: pd.DataFrame,
    per_fixation: pd.DataFrame,
    first_fixation: pd.DataFrame,
    dwell: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (model, shape), frame in decisions.groupby(
        ["model", "footprint_shape"], observed=True
    ):
        for aoi, outcome in HIT_COLUMNS.items():
            h_subset = "H2_H4" if aoi == "Preview" else "all_H"
            eligible = frame[frame.H.isin([2, 4])] if aoi == "Preview" else frame
            cluster_cells = (
                eligible.groupby(["cluster", "T", "H"], observed=True)[outcome]
                .mean()
                .reset_index()
            )
            raw = cluster_cells.groupby("T", observed=True)[outcome].mean().to_dict()
            entity_per_fix = per_fixation[
                per_fixation.model.eq(model)
                & per_fixation.footprint_shape.eq(shape)
            ]
            per_fix = _values(entity_per_fix, f"{aoi}_per_fix", h_subset)
            entity_first = first_fixation[
                first_fixation.model.eq(model)
                & first_fixation.footprint_shape.eq(shape)
            ]
            first = _values(entity_first, aoi, h_subset)
            entity_dwell = dwell[
                dwell.model.eq(model) & dwell.footprint_shape.eq(shape)
            ]
            dwell_values = _values(
                entity_dwell, f"{aoi}_dwell_ms_per_decision", h_subset
            )
            for low, high in CONTRASTS:
                model_row = adjusted[
                    adjusted.model.eq(model)
                    & adjusted.aoi.eq(aoi)
                    & adjusted.T_from.eq(low)
                    & adjusted.T_to.eq(high)
                ].iloc[0]
                interval_100ms = (high - low) / 100.0
                per_fix_delta = per_fix[high] - per_fix[low]
                rows.append(
                    {
                        "model": model,
                        "footprint_shape": shape,
                        "aoi": aoi,
                        "h_subset": h_subset,
                        "T_from": low,
                        "T_to": high,
                        "interval_ms": high - low,
                        "raw_decision_coverage_from": raw[low],
                        "raw_decision_coverage_to": raw[high],
                        "raw_decision_coverage_delta": raw[high] - raw[low],
                        "per_fixation_probability_from": per_fix[low],
                        "per_fixation_probability_to": per_fix[high],
                        "per_fixation_probability_delta": per_fix_delta,
                        "per_fixation_slope_pp_per_100ms": (
                            per_fix_delta * 100.0 / interval_100ms
                        ),
                        "first_new_fixation_probability_from": first[low],
                        "first_new_fixation_probability_to": first[high],
                        "first_new_fixation_probability_delta": first[high] - first[low],
                        "dwell_ms_per_decision_from": dwell_values[low],
                        "dwell_ms_per_decision_to": dwell_values[high],
                        "dwell_ms_per_decision_delta": dwell_values[high]
                        - dwell_values[low],
                        "adjusted_odds_ratio": model_row.odds_ratio,
                        "adjusted_or_ci_low": model_row.ci_low,
                        "adjusted_or_ci_high": model_row.ci_high,
                        "adjusted_or_p_value": model_row.p_value,
                        "n_clusters": model_row.n_clusters,
                    }
                )
    return pd.DataFrame(rows)


def _performance_contrasts(
    decisions: pd.DataFrame, performances: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for model in decisions.model.unique():
        frame = decisions[decisions.model.eq(model)]
        new_fixations = frame.groupby("T", observed=True).n_new_fixations.mean().to_dict()
        performance = performances.get(model)
        deliveries = (
            performance.groupby("T", observed=True).deliveries.mean().to_dict()
            if performance is not None
            else {}
        )
        rewards = (
            performance.groupby("T", observed=True).sparse_team_reward.mean().to_dict()
            if performance is not None
            else {}
        )
        for low, high in CONTRASTS:
            rows.append(
                {
                    "model": model,
                    "T_from": low,
                    "T_to": high,
                    "interval_ms": high - low,
                    "deliveries_per_episode_from": deliveries.get(low, np.nan),
                    "deliveries_per_episode_to": deliveries.get(high, np.nan),
                    "deliveries_per_episode_delta": deliveries.get(high, np.nan)
                    - deliveries.get(low, np.nan),
                    "sparse_reward_per_episode_from": rewards.get(low, np.nan),
                    "sparse_reward_per_episode_to": rewards.get(high, np.nan),
                    "sparse_reward_per_episode_delta": rewards.get(high, np.nan)
                    - rewards.get(low, np.nan),
                    "new_fixations_per_decision_from": new_fixations[low],
                    "new_fixations_per_decision_to": new_fixations[high],
                    "new_fixations_per_decision_delta": new_fixations[high]
                    - new_fixations[low],
                }
            )
    return pd.DataFrame(rows)


def _cluster_directions(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, shape, cluster), frame in events.groupby(
        ["model", "footprint_shape", "cluster"], observed=True
    ):
        for aoi, outcome in HIT_COLUMNS.items():
            eligible = frame[frame.H.isin([2, 4])] if aoi == "Preview" else frame
            probability = eligible.groupby("T", observed=True)[outcome].mean().to_dict()
            for low, high in CONTRASTS:
                delta = probability[high] - probability[low]
                rows.append(
                    {
                        "model": model,
                        "footprint_shape": shape,
                        "cluster": cluster,
                        "cluster_type": "participant" if model == "Human" else "seed",
                        "aoi": aoi,
                        "T_from": low,
                        "T_to": high,
                        "probability_from": probability[low],
                        "probability_to": probability[high],
                        "delta": delta,
                        "direction": "increase" if delta > 0 else "decrease" if delta < 0 else "flat",
                    }
                )
    cluster_rows = pd.DataFrame(rows)
    direction = (
        cluster_rows.groupby(
            ["model", "footprint_shape", "cluster_type", "aoi", "T_from", "T_to"],
            observed=True,
        )
        .direction.value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ("increase", "decrease", "flat"):
        if column not in direction:
            direction[column] = 0
    direction["clusters"] = direction[["increase", "decrease", "flat"]].sum(axis=1)
    return cluster_rows, direction


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    human_decisions, human_segments = load_human(
        PROJECT / "data" / "processed", RESULTS / "human_benchmark_20260901_contract_v2"
    )
    decision_frames = [human_decisions[human_decisions.comparison.eq("plus")].copy()]
    segment_frames = [human_segments[human_segments.comparison.eq("plus")].copy()]
    performances: dict[str, pd.DataFrame] = {}
    for model, root in MODELS:
        decisions, segments = load_model(root, label=model, comparison="plus")
        decision_frames.append(decisions)
        segment_frames.append(segments)
        performances[model] = pd.read_csv(root / "summaries/performance.csv")

    decisions = pd.concat(decision_frames, ignore_index=True)
    segments = pd.concat(segment_frames, ignore_index=True)
    events = new_fixations(segments)
    decisions = attach_outcomes(decisions, segments, events)
    per_fixation = per_fixation_table(events)
    first_fixation = first_fixation_table(decisions, events)
    dwell = dwell_table(segments.copy(), decisions)
    adjusted = _adjusted_contrasts(decisions)
    cluster_rows, directions = _cluster_directions(events)

    outputs = {
        "pairwise_aoi_contrasts.csv": _aoi_contrasts(
            decisions, per_fixation, first_fixation, dwell, adjusted
        ),
        "pairwise_adjusted_models.csv": adjusted,
        "pairwise_performance_contrasts.csv": _performance_contrasts(
            decisions, performances
        ),
        "per_cluster_per_fixation_contrasts.csv": cluster_rows,
        "per_cluster_direction_summary.csv": directions,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT / filename, index=False)
    manifest = {
        "purpose": "Separate T effects into 200-400, 400-600, and 200-600 contrasts",
        "source": "saved actual-environment v8 rerollouts; no policy replay",
        "adjusted_model": (
            "decision-level AOI coverage ~ categorical T + categorical H + new-fixation count; "
            "GEE exchangeable clustering by participant or seed"
        ),
        "preview_subset": [2, 4],
        "contrasts": [list(pair) for pair in CONTRASTS],
        "inference_warning": "Model estimates use five seeds; interpret direction and magnitude diagnostically",
        "outputs": {name: sha256(OUTPUT / name) for name in outputs},
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
