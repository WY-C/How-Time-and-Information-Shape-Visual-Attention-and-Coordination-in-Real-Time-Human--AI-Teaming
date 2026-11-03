"""Package an early-checkpoint rerollout and opportunity-controlled AOI analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    HIT_COLUMNS,
    attach_outcomes,
    load_model,
    new_fixations,
)


BASELINE_ANALYSES = {
    "+7M": Path(
        "final_model/eval/results/humanfix_opportunity_controlled_aoi_20260905"
        "/opportunity_controlled_summary.csv"
    ),
    "+10M": Path(
        "final_model/eval/results/humanfix_opportunity_controlled_aoi_10m_20260905"
        "/opportunity_controlled_summary.csv"
    ),
}
BASELINE_SLOPES = Path(
    "final_model/eval/results/humanfix_opportunity_controlled_aoi_10m_20260905"
    "/per_fixation_T_slopes.csv"
)
BASELINE_ROOTS = {
    "+7M": Path(
        "final_model/eval/results/"
        "humanfix_plus_local_gaze7m_v8_exponential_30ep_tiefix_20260905"
    ),
    "+10M": Path(
        "final_model/eval/results/"
        "humanfix_plus_local_gaze10m_v8_exponential_30ep_20260905"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def performance_summary(performance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    slices = [("overall", "all", performance)]
    slices += [("T", str(int(key)), frame) for key, frame in performance.groupby("T")]
    slices += [("H", str(int(key)), frame) for key, frame in performance.groupby("H")]
    slices += [
        ("T_H", f"T{int(t)}_H{int(h)}", frame)
        for (t, h), frame in performance.groupby(["T", "H"])
    ]
    for dimension, level, frame in slices:
        rows.append(
            {
                "dimension": dimension,
                "level": level,
                "episodes": len(frame),
                "decisions": int(frame.decisions.sum()),
                "deliveries": int(frame.deliveries.sum()),
                "deliveries_per_episode": float(frame.deliveries.mean()),
                "sparse_reward": float(frame.sparse_team_reward.sum()),
                "sparse_reward_per_episode": float(frame.sparse_team_reward.mean()),
            }
        )
    return pd.DataFrame(rows)


def fit_per_fixation_slopes(events: pd.DataFrame, model: str) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable

    rows = []
    for aoi, outcome in HIT_COLUMNS.items():
        subset = events.copy()
        if aoi == "Preview":
            subset = subset[subset.H.isin([2, 4])].copy()
            h_subset = "H2_H4"
            h_term = "C(H, Treatment(reference=2))"
        else:
            h_subset = "all_H"
            h_term = "C(H, Treatment(reference=0))"
        subset["T_100ms"] = subset["T"].astype(float) / 100.0
        formula = f"{outcome} ~ T_100ms + {h_term}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.gee(
                formula,
                groups="cluster",
                data=subset,
                family=sm.families.Binomial(),
                cov_struct=Exchangeable(),
            ).fit()
        estimate = float(fit.params["T_100ms"])
        robust_se = float(fit.bse["T_100ms"])
        raw = subset.groupby("T", observed=True)[outcome].mean()
        rows.append(
            {
                "model": model,
                "aoi": aoi,
                "h_subset": h_subset,
                "n_new_fixations": len(subset),
                "clusters": subset.cluster.nunique(),
                "formula": formula,
                "log_odds_slope_per_100ms": estimate,
                "robust_se": robust_se,
                "odds_ratio_per_100ms": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.96 * robust_se),
                "ci_high": math.exp(estimate + 1.96 * robust_se),
                "p_value": float(fit.pvalues["T_100ms"]),
                "marginal_probability_T200": float(raw.loc[200]),
                "marginal_probability_T400": float(raw.loc[400]),
                "marginal_probability_T600": float(raw.loc[600]),
                "marginal_slope_pp_per_100ms": float((raw.loc[600] - raw.loc[200]) * 25.0),
                "cluster": "seed",
            }
        )
    return pd.DataFrame(rows)


def seed_metrics(
    performance: pd.DataFrame, decisions: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rows = []
    for seed in sorted(performance.seed.unique()):
        perf = performance[performance.seed.eq(seed)]
        decision = decisions[decisions.seed.eq(seed)].copy()
        event = events[events.seed.eq(seed)].copy()
        row: dict[str, object] = {
            "seed": int(seed),
            "episodes": len(perf),
            "decisions": int(perf.decisions.sum()),
            "deliveries": int(perf.deliveries.sum()),
            "deliveries_per_episode": float(perf.deliveries.mean()),
            "sparse_reward_per_episode": float(perf.sparse_team_reward.mean()),
            "new_fixations": len(event),
            "new_fixations_per_decision": float(len(event) / len(decision)),
        }
        for aoi, outcome in HIT_COLUMNS.items():
            subset_event = event[event.H.isin([2, 4])] if aoi == "Preview" else event
            raw = subset_event.groupby("T", observed=True)[outcome].mean()
            row[f"{aoi}_per_fix_T200"] = float(raw.loc[200])
            row[f"{aoi}_per_fix_T600"] = float(raw.loc[600])
            row[f"{aoi}_per_fix_slope_pp_per_100ms"] = float(
                (raw.loc[600] - raw.loc[200]) * 25.0
            )

            subset = decision[decision.H.isin([2, 4])].copy() if aoi == "Preview" else decision.copy()
            h_ref = 2 if aoi == "Preview" else 0
            formula = (
                f"{outcome} ~ C(T, Treatment(reference=200)) + "
                f"C(H, Treatment(reference={h_ref})) + n_new_fixations"
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.glm(formula, data=subset, family=sm.families.Binomial()).fit()
                coefficient = "C(T, Treatment(reference=200))[T.600]"
                row[f"{aoi}_seed_glm_adjusted_or_T600_vs_T200"] = math.exp(
                    float(fit.params[coefficient])
                )
            except Exception:
                row[f"{aoi}_seed_glm_adjusted_or_T600_vs_T200"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(
    root: Path, direct_summary: pd.DataFrame, direct_performance: pd.DataFrame,
    direct_slopes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label in ("+7M", "+10M"):
        baseline = pd.read_csv(BASELINE_ANALYSES[label])
        aoi = baseline[(baseline.comparison.eq("plus")) & baseline.model.eq(label)]
        perf = pd.read_csv(BASELINE_ROOTS[label] / "summaries/performance.csv")
        slopes = pd.read_csv(BASELINE_SLOPES)
        row = {
            "checkpoint": f"Staged{label[1:]}",
            "evaluation_seeds": int(perf.seed.nunique()),
            "episodes": len(perf),
            "deliveries_per_episode": float(perf.deliveries.mean()),
            "sparse_reward_per_episode": float(perf.sparse_team_reward.mean()),
        }
        for name in HIT_COLUMNS:
            row[f"{name}_adjusted_or"] = float(
                aoi[aoi.aoi.eq(name)].adjusted_T600_vs_T200_odds_ratio.iloc[0]
            )
        for name in ("Teammate", "Preview"):
            row[f"{name}_per_fix_slope_pp_per_100ms"] = float(
                slopes[(slopes.model.eq(label)) & slopes.aoi.eq(name)]
                .marginal_slope_pp_per_100ms.iloc[0]
            )
        rows.append(row)

    direct = {
        "checkpoint": "DirectJoint1M",
        "evaluation_seeds": int(direct_performance.seed.nunique()),
        "episodes": len(direct_performance),
        "deliveries_per_episode": float(direct_performance.deliveries.mean()),
        "sparse_reward_per_episode": float(direct_performance.sparse_team_reward.mean()),
    }
    for name in HIT_COLUMNS:
        direct[f"{name}_adjusted_or"] = float(
            direct_summary[direct_summary.aoi.eq(name)]
            .adjusted_T600_vs_T200_odds_ratio.iloc[0]
        )
    for name in ("Teammate", "Preview"):
        direct[f"{name}_per_fix_slope_pp_per_100ms"] = float(
            direct_slopes[direct_slopes.aoi.eq(name)].marginal_slope_pp_per_100ms.iloc[0]
        )
    rows.append(direct)
    frame = pd.DataFrame(rows)
    staged = frame[frame.checkpoint.eq("Staged10M")].iloc[0]
    for name in HIT_COLUMNS:
        frame[f"{name}_or_delta_vs_staged10m"] = frame[f"{name}_adjusted_or"] - staged[
            f"{name}_adjusted_or"
        ]
        frame[f"{name}_distance_to_1"] = (frame[f"{name}_adjusted_or"] - 1.0).abs()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="StrictDirectJoint1M")
    args = parser.parse_args()
    root = args.root.resolve()
    analysis_root = root / "opportunity_controlled"

    performance = pd.read_csv(root / "summaries/performance.csv")
    decisions, segments = load_model(root, label=args.model, comparison="plus")
    events = new_fixations(segments)
    decisions["new_fixations"] = 0
    decisions = attach_outcomes(decisions, segments, events)

    performance_summary(performance).to_csv(root / "performance_summary.csv", index=False)
    for name in (
        "per_fixation_aoi_probability.csv",
        "first_new_fixation_aoi.csv",
        "fixation_count_adjusted_models.csv",
    ):
        frame = pd.read_csv(analysis_root / name)
        frame = frame[(frame.comparison.eq("plus")) & frame.model.eq(args.model)]
        frame.to_csv(root / name, index=False)

    slopes = fit_per_fixation_slopes(events, args.model)
    slopes.to_csv(root / "per_fixation_T_slopes.csv", index=False)
    seeds = seed_metrics(performance, decisions, events)
    seeds.to_csv(root / "seed_level_metrics.csv", index=False)

    opportunity = pd.read_csv(analysis_root / "opportunity_controlled_summary.csv")
    opportunity = opportunity[
        (opportunity.comparison.eq("plus")) & opportunity.model.eq(args.model)
    ].copy()
    comparison = comparison_table(root, opportunity, performance, slopes)
    comparison.to_csv(root / "early_checkpoint_comparison.csv", index=False)

    checkpoint_pair = json.loads((root / "benchmark/checkpoint_pair.json").read_text())
    benchmark_manifest = json.loads((root / "benchmark/evaluation_manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    metadata = checkpoint_pair["action_checkpoint"]["metadata"]
    manifest = {
        "purpose": "strict direct-joint 1M early diagnostic; not a final result",
        "rerollout_type": "actual_environment_rerollout",
        "checkpoint_provenance": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
            "step": checkpoint_pair["action_checkpoint"]["completed_steps"],
            "action_actor_sha256": checkpoint_pair["action_checkpoint"]["actor_sha256"],
            "gaze_actor_sha256": checkpoint_pair["gaze_checkpoint"]["gaze_sha256"],
            "source_full_action_path": metadata["action_checkpoint_path"],
            "source_full_action_sha256": metadata["action_checkpoint_sha256"],
            "partner_path": metadata["fixed_partner_checkpoint"],
            "partner_file_sha256": metadata["fixed_partner_sha256"],
            "partner_parameter_sha256": metadata["fixed_partner_parameter_sha256"],
            "training_contract": metadata["training_contract"],
            "training_stage": metadata["training_stage"],
            "execution_backend": metadata["execution_backend"],
            "emma_mode": metadata["emma_mode"],
            "emma_timing_contract_id": metadata["emma_timing_contract_id"],
        },
        "training_config": {
            "master_seed": metadata["master_seed"],
            "decision_times": metadata["decision_times"],
            "horizons": metadata["horizons"],
            "fixation_shapes": metadata["fixation_shapes"],
            "peripheral_mode": metadata["peripheral_mode"],
            "peripheral_sigma_deg": metadata["peripheral_sigma_deg"],
            "memory_decay_mode": metadata["memory_decay_mode"],
            "memory_lifetime_ms": metadata["memory_lifetime_ms"],
            "logical_env_count": metadata["logical_env_count"],
            "n_rollout_envs": metadata["n_rollout_envs"],
            "rollout_length": metadata["rollout_length"],
            "ppo_config": metadata["ppo_config"],
            "reward_shaping_horizon_steps": metadata["reward_shaping_horizon_steps"],
        },
        "evaluation_contract": config,
        "benchmark_manifest": benchmark_manifest,
        "seeds": sorted(int(seed) for seed in performance.seed.unique()),
        "episodes_per_T_H_cell": int(performance.groupby(["T", "H"]).size().iloc[0]),
        "decisions_per_episode": sorted(int(value) for value in performance.decisions.unique()),
        "aoi_analysis": {
            "script": "final_model/eval/analyze_opportunity_controlled_aoi.py",
            "footprint": "plus_5_tile",
            "preview_h_subset": [2, 4],
            "adjusted_model": "decision coverage ~ categorical T + categorical H + n_new_fixations; GEE clustered by seed",
            "inference_warning": "n=5 model seeds; direction and magnitude only",
        },
        "outputs": {},
    }
    outputs = (
        "performance_summary.csv", "per_fixation_aoi_probability.csv",
        "first_new_fixation_aoi.csv", "fixation_count_adjusted_models.csv",
        "per_fixation_T_slopes.csv", "seed_level_metrics.csv",
        "early_checkpoint_comparison.csv",
    )
    manifest["outputs"] = {name: sha256(root / name) for name in outputs}
    (root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
