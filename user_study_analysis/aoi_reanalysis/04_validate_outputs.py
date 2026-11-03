#!/usr/bin/env python3
"""Fail-closed validation of the AOI reanalysis output contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AOI_STEMS = ["self", "partner", "information", "other"]
MAIN_PIDS = {f"P{i:02d}" for i in range(1, 37)}
EXPECTED_TRIALS_PER_PARTICIPANT = 27
EXPECTED_DECISION_STEPS_PER_TRIAL = 180
EXPECTED_MAIN_TRIALS = 36 * EXPECTED_TRIALS_PER_PARTICIPANT
EXPECTED_FINE = {200: 12, 400: 24, 600: 36}
EXPECTED_COARSE = {200: 2, 400: 4, 600: 6}
TOL = 1e-8


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required output missing: {path}")
    return path


def max_prop_error(d: pd.DataFrame) -> float:
    x = d[d.get("total_fixation_ms", pd.Series(1.0, index=d.index)).fillna(0) > 0].copy()
    if x.empty:
        return 0.0
    cols = [f"{a}_prop" for a in AOI_STEMS]
    return float((x[cols].sum(axis=1) - 1.0).abs().max())


def validate_bin_counts(d: pd.DataFrame, expected: dict[int, int], label: str) -> float:
    counts = d.groupby(["trial_id", "Time"], as_index=False)["Bin"].nunique()
    counts["expected"] = counts["Time"].map(expected)
    bad = counts[counts["Bin"].ne(counts["expected"])]
    if len(bad):
        raise RuntimeError(f"{label} structural bin-count failure:\n{bad.head(20)}")
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.out)
    raw = root / "00_raw_intermediate"
    qcdir = root / "05_qc"
    qcdir.mkdir(parents=True, exist_ok=True)

    required_files = [
        raw / "trial_eligibility_audit.csv",
        root / "05_qc" / "trial_count_by_participant.csv",
        root / "05_qc" / "trial_condition_balance.csv",
        root / "05_qc" / "excluded_trials.csv",
        raw / "gaze_segment_allocations.csv",
        raw / "overall_trial_aoi_values.csv",
        raw / "temporal_16.67ms_trial_bin_values.csv",
        raw / "temporal_100ms_trial_bin_values.csv",
        raw / "h1_h4_trial_values.csv",
        raw / "h1_h4_near_far_trial_values.csv",
        raw / "behavior_decision_values.csv",
        raw / "behavior_trial_values.csv",
        raw / "subjective_mapped_trial_values.csv",
        root / "01_analysis_inputs" / "overall_trial_lmm_input.csv",
        root / "01_analysis_inputs" / "temporal_16.67ms_participant_lmm_input.csv",
        root / "01_analysis_inputs" / "temporal_100ms_participant_lmm_input.csv",
        root / "01_analysis_inputs" / "h1_h4_trial_lmm_input.csv",
        root / "01_analysis_inputs" / "h1_h4_relative_participant_lmm_input.csv",
        root / "01_analysis_inputs" / "h1_h4_near_far_trial_lmm_input.csv",
        root / "01_analysis_inputs" / "behavior_participant_condition_values.csv",
        root / "01_analysis_inputs" / "subjective_participant_condition_values.csv",
        root / "02_statistics" / "overall_aoi_lmm.csv",
        root / "02_statistics" / "temporal_within_budget_lmm.csv",
        root / "02_statistics" / "h1_h4_lmm.csv",
        root / "02_statistics" / "h1_h4_relative_lmm.csv",
        root / "02_statistics" / "h1_h4_near_far_lmm.csv",
        root / "02_statistics" / "behavior_rm_anova.csv",
        root / "02_statistics" / "subjective_rm_anova.csv",
        root / "02_statistics" / "overall_aoi_lmm_posthoc_holm.csv",
        root / "02_statistics" / "temporal_posthoc_lmm_holm.csv",
        root / "02_statistics" / "h1_h4_lmm_posthoc_holm.csv",
        root / "02_statistics" / "h1_h4_relative_lmm_posthoc_holm.csv",
        root / "02_statistics" / "h1_h4_near_far_lmm_posthoc_holm.csv",
        root / "02_statistics" / "behavior_posthoc_holm.csv",
        root / "02_statistics" / "subjective_posthoc_holm.csv",
        root / "02_statistics" / "all_posthoc_holm.csv",
    ]
    for f in required_files:
        require(f)

    rows: list[dict] = []

    # Canonical trial eligibility: only completed 180-decision trials enter analysis,
    # and every P01-P36 participant must retain exactly 27 valid trials.
    eligibility = pd.read_csv(raw / "trial_eligibility_audit.csv")
    eligible = eligibility[eligibility["eligible"].astype(str).str.lower().isin({"true", "1"})].copy()
    n_eligible = int(eligible["trial_id"].nunique())
    rows.append({"check": "eligible_trial_total_972", "value": float(n_eligible), "criterion": f"== {EXPECTED_MAIN_TRIALS}", "pass": n_eligible == EXPECTED_MAIN_TRIALS})
    decision_bad = int((pd.to_numeric(eligible["decision_steps_completed_numeric"], errors="coerce") != EXPECTED_DECISION_STEPS_PER_TRIAL).sum())
    rows.append({"check": "eligible_trials_exactly_180_decisions", "value": float(decision_bad), "criterion": "== 0", "pass": decision_bad == 0})

    pc = pd.read_csv(qcdir / "trial_count_by_participant.csv")
    pid_set_ok = set(pc["pid"].astype(str)) == MAIN_PIDS
    count_bad = int((pd.to_numeric(pc["n_valid_trials"], errors="coerce") != EXPECTED_TRIALS_PER_PARTICIPANT).sum())
    rows.append({"check": "participant_set_P01_P36", "value": float(pid_set_ok), "criterion": "== 1", "pass": bool(pid_set_ok)})
    rows.append({"check": "each_participant_exactly_27_trials", "value": float(count_bad), "criterion": "== 0", "pass": count_bad == 0})

    balance = pd.read_csv(qcdir / "trial_condition_balance.csv")
    balance_bad = int((~balance["cell_ok"].astype(str).str.lower().isin({"true", "1"})).sum())
    rows.append({"check": "each_participant_condition_3_trials_3_layouts", "value": float(balance_bad), "criterion": "== 0", "pass": balance_bad == 0})

    seg = pd.read_csv(raw / "gaze_segment_allocations.csv")
    seg_err = float((seg[[f"{a}_ms" for a in AOI_STEMS]].sum(axis=1) - seg["duration_ms"]).abs().max())
    rows.append({"check": "segment_AOI_ms_conservation", "value": seg_err, "criterion": f"<= {TOL}", "pass": seg_err <= TOL})

    overall = pd.read_csv(raw / "overall_trial_aoi_values.csv")
    e = max_prop_error(overall)
    rows.append({"check": "overall_4AOI_prop_sum", "value": e, "criterion": f"<= {TOL}", "pass": e <= TOL})

    for filename, expected, name in [
        ("temporal_16.67ms_trial_bin_values.csv", EXPECTED_FINE, "16.67ms"),
        ("temporal_100ms_trial_bin_values.csv", EXPECTED_COARSE, "100ms"),
    ]:
        d = pd.read_csv(raw / filename)
        e = max_prop_error(d)
        rows.append({"check": f"temporal_{name}_4AOI_prop_sum", "value": e, "criterion": f"<= {TOL}", "pass": e <= TOL})
        validate_bin_counts(d, expected, name)
        rows.append({"check": f"temporal_{name}_structural_bin_counts", "value": 0.0, "criterion": "exact", "pass": True})

    # Response files may legitimately be empty only if the source has no eligible actual responses.
    for mode in ["whole", "symmetric"]:
        f = raw / f"response_{mode}_decision_phase_values.csv"
        if f.exists() and f.stat().st_size > 0:
            d = pd.read_csv(f)
            if len(d):
                e = max_prop_error(d)
                rows.append({"check": f"response_{mode}_4AOI_prop_sum", "value": e, "criterion": f"<= {TOL}", "pass": e <= TOL})

    # Hierarchical contract: H1-H4 exactly partition top-level Information dwell in H4 trials.
    h = pd.read_csv(raw / "h1_h4_trial_values.csv")
    hsum = h.groupby("trial_id", as_index=False).agg(
        h1_h4_sum_ms=("dwell_ms", "sum"),
        relative_sum=("relative_allocation", "sum"),
    )
    info = overall[overall["Horizon"].eq(4)][["trial_id", "information_ms"]]
    z = info.merge(hsum, on="trial_id", how="left", validate="one_to_one").fillna({"h1_h4_sum_ms": 0.0})
    partition_err = float((z["h1_h4_sum_ms"] - z["information_ms"]).abs().max()) if len(z) else 0.0
    rows.append({"check": "H1_H4_exactly_partitions_Information", "value": partition_err, "criterion": f"<= {TOL}", "pass": partition_err <= TOL})
    positive_info_ids = set(info.loc[info["information_ms"] > 0, "trial_id"])
    hrel = h[h["trial_id"].isin(positive_info_ids)].groupby("trial_id")["relative_allocation"].sum(min_count=1)
    rel_err = float((hrel - 1.0).abs().max()) if len(hrel) else 0.0
    rows.append({"check": "H1_H4_relative_allocation_sums_to_1_when_Information_positive", "value": rel_err, "criterion": f"<= {TOL}", "pass": rel_err <= TOL})
    nf = pd.read_csv(raw / "h1_h4_near_far_trial_values.csv")
    nf_sum = nf.groupby("trial_id", as_index=False).agg(nf_ms=("dwell_ms","sum"), nf_relative=("relative_allocation","sum"))
    zz = info.merge(nf_sum, on="trial_id", how="left", validate="one_to_one")
    nf_err = float((zz["nf_ms"] - zz["information_ms"]).abs().max()) if len(zz) else 0.0
    rows.append({"check": "NearFar_exactly_partitions_Information", "value": nf_err, "criterion": f"<= {TOL}", "pass": nf_err <= TOL})
    nfrel = nf[nf["trial_id"].isin(positive_info_ids)].groupby("trial_id")["relative_allocation"].sum(min_count=1)
    nf_rel_err = float((nfrel - 1.0).abs().max()) if len(nfrel) else 0.0
    rows.append({"check": "NearFar_relative_allocation_sums_to_1_when_Information_positive", "value": nf_rel_err, "criterion": f"<= {TOL}", "pass": nf_rel_err <= TOL})

    # Gaze inference contract: all inferential gaze models are LMMs, not RM-ANOVAs.
    gaze_lmm_files = [
        root / "02_statistics" / "overall_aoi_lmm.csv",
        root / "02_statistics" / "temporal_within_budget_lmm.csv",
        root / "02_statistics" / "h1_h4_lmm.csv",
        root / "02_statistics" / "h1_h4_relative_lmm.csv",
        root / "02_statistics" / "h1_h4_near_far_lmm.csv",
    ]
    for mode in ["whole", "symmetric"]:
        rp = root / "02_statistics" / f"response_{mode}_lmm.csv"
        if rp.exists():
            gaze_lmm_files.append(rp)
            require(root / "02_statistics" / f"response_{mode}_posthoc_lmm_holm.csv")
    nonconv = 0
    wrong_random = 0
    nonfinite_fit = 0
    for f in gaze_lmm_files:
        d = pd.read_csv(require(f))
        if "converged" not in d.columns or "random_effects" not in d.columns:
            raise RuntimeError(f"LMM output missing convergence/random-effect metadata: {f}")
        nonconv += int((~d["converged"].astype(str).str.lower().isin({"true", "1"})).sum())
        target = d[d.get("role", pd.Series("", index=d.index)).eq("target")]
        bad_target = target["p_value"].isna() | ~np.isfinite(pd.to_numeric(target["p_value"], errors="coerce")) | pd.to_numeric(target["df"], errors="coerce").le(0)
        wrong_random += int(bad_target.sum())
        for col in ["log_likelihood", "AIC", "BIC"]:
            if col not in d.columns:
                nonfinite_fit += len(d)
            else:
                nonfinite_fit += int((~np.isfinite(pd.to_numeric(d[col], errors="coerce"))).sum())
    rows.append({"check": "all_gaze_LMM_models_converged", "value": float(nonconv), "criterion": "== 0", "pass": nonconv == 0})
    rows.append({"check": "all_target_LMM_Wald_tests_estimable", "value": float(wrong_random), "criterion": "== 0", "pass": wrong_random == 0})
    rows.append({"check": "all_gaze_LMM_fit_metrics_finite", "value": float(nonfinite_fit), "criterion": "== 0", "pass": nonfinite_fit == 0})

    old_gaze_rm = [p for p in (root / "02_statistics").glob("*_rm_anova.csv") if p.name not in {"behavior_rm_anova.csv", "subjective_rm_anova.csv"}]
    rows.append({"check": "no_gaze_RM_ANOVA_outputs", "value": float(len(old_gaze_rm)), "criterion": "== 0", "pass": len(old_gaze_rm) == 0})

    overall_lmm = pd.read_csv(root / "02_statistics" / "overall_aoi_lmm.csv")
    overall_re_ok = overall_lmm["random_effects"].eq("(1|Participant)").all()
    rows.append({"check": "overall_gaze_participant_random_intercept", "value": float(overall_re_ok), "criterion": "== 1", "pass": bool(overall_re_ok)})
    invalid_info_3x3 = int(((overall_lmm["analysis"] == "Overall 4-AOI composition (3x3)") & (overall_lmm.get("AOI", pd.Series("", index=overall_lmm.index)) == "Information")).sum())
    valid_info_primary = int(((overall_lmm["analysis"] == "Information gaze primary (H2 vs H4)") & (overall_lmm.get("AOI", pd.Series("", index=overall_lmm.index)) == "Information")).sum())
    rows.append({"check": "no_structural_H0_Information_3x3_LMM", "value": float(invalid_info_3x3), "criterion": "== 0", "pass": invalid_info_3x3 == 0})
    rows.append({"check": "Information_primary_H2_H4_LMM_present", "value": float(valid_info_primary), "criterion": "> 0", "pass": valid_info_primary > 0})

    h_lmm = pd.read_csv(root / "02_statistics" / "h1_h4_lmm.csv")
    h_re_ok = h_lmm["random_effects"].eq("(1|Participant) + (1|Trial)").all()
    rows.append({"check": "H1_H4_participant_and_trial_random_intercepts", "value": float(h_re_ok), "criterion": "== 1", "pass": bool(h_re_ok)})

    hrel = pd.read_csv(root / "02_statistics" / "h1_h4_relative_lmm.csv")
    hrel_ok = hrel["random_effects"].eq("(1|Participant)").all()
    rows.append({"check": "H1_H4_relative_participant_random_intercept", "value": float(hrel_ok), "criterion": "== 1", "pass": bool(hrel_ok)})
    hnf = pd.read_csv(root / "02_statistics" / "h1_h4_near_far_lmm.csv")
    hnf_ok = hnf["random_effects"].eq("(1|Participant) + (1|Trial)").all()
    rows.append({"check": "NearFar_participant_and_trial_random_intercepts", "value": float(hnf_ok), "criterion": "== 1", "pass": bool(hnf_ok)})

    temporal_lmm = pd.read_csv(root / "02_statistics" / "temporal_within_budget_lmm.csv")
    temporal_re_ok = temporal_lmm["random_effects"].eq("(1|Participant)").all()
    rows.append({"check": "temporal_within_budget_participant_random_intercept", "value": float(temporal_re_ok), "criterion": "== 1", "pass": bool(temporal_re_ok)})


    for mode in ["whole", "symmetric"]:
        rp = root / "02_statistics" / f"response_{mode}_lmm.csv"
        if rp.exists():
            rr = pd.read_csv(rp)
            ok = rr["random_effects"].eq("(1|Participant) + (1|Trial)").all()
            rows.append({"check": f"response_{mode}_participant_and_trial_random_intercepts", "value": float(ok), "criterion": "== 1", "pass": bool(ok)})

    behavior = pd.read_csv(raw / "behavior_decision_values.csv")
    timeout = behavior[behavior["timeout"].eq(1)]
    rt_err = float((timeout["response_time_with_timeout_ms"] - timeout["Time"]).abs().max()) if len(timeout) else 0.0
    rows.append({"check": "timeout_RT_equals_budget", "value": rt_err, "criterion": f"<= {TOL}", "pass": rt_err <= TOL})
    over_budget = float((behavior["response_time_with_timeout_ms"] - behavior["Time"]).max()) if len(behavior) else 0.0
    rows.append({"check": "no_RT_above_budget", "value": over_budget, "criterion": f"<= {TOL}", "pass": over_budget <= TOL})

    # Timeout must preserve the original trial-level definition.
    btrial = pd.read_csv(raw / "behavior_trial_values.csv")
    out_count = btrial.groupby("pid")["trial_id"].nunique().reindex(sorted(MAIN_PIDS), fill_value=0)
    output_count_bad = int((out_count != EXPECTED_TRIALS_PER_PARTICIPANT).sum())
    output_decision_bad = int((pd.to_numeric(btrial["decision_steps_completed"], errors="coerce") != EXPECTED_DECISION_STEPS_PER_TRIAL).sum())
    rows.append({"check": "behavior_output_each_participant_27_trials", "value": float(output_count_bad), "criterion": "== 0", "pass": output_count_bad == 0})
    rows.append({"check": "behavior_output_trials_180_decisions", "value": float(output_decision_bad), "criterion": "== 0", "pass": output_decision_bad == 0})
    timeout_expected = btrial["num_human_idle"] / btrial["decision_steps_completed"]
    timeout_def_err = float((btrial["timeout_rate"] - timeout_expected).abs().max()) if len(btrial) else 0.0
    rows.append({"check": "timeout_rate_matches_trial_definition", "value": timeout_def_err, "criterion": f"<= {TOL}", "pass": timeout_def_err <= TOL})

    # Collision inference must use raw x, with participant-condition values equal
    # to the arithmetic mean of trial-level num_collisions across layouts.
    bcells = pd.read_csv(root / "01_analysis_inputs" / "behavior_participant_condition_values.csv")
    expected_collision = (
        btrial.groupby(["pid", "Time", "Horizon"], as_index=False)["num_collisions"].mean()
        .rename(columns={"num_collisions": "expected_num_collisions"})
    )
    zz = bcells.merge(expected_collision, on=["pid", "Time", "Horizon"], how="inner", validate="one_to_one")
    collision_err = float((zz["num_collisions"] - zz["expected_num_collisions"]).abs().max()) if len(zz) else 0.0
    rows.append({"check": "collision_raw_x_participant_mean", "value": collision_err, "criterion": f"<= {TOL}", "pass": collision_err <= TOL})
    forbidden = [c for c in bcells.columns if "collision_log" in c.lower()]
    rows.append({"check": "collision_no_transform_columns", "value": float(len(forbidden)), "criterion": "== 0", "pass": len(forbidden) == 0})

    bstats = pd.read_csv(root / "02_statistics" / "behavior_rm_anova.csv")
    outcomes = set(bstats.get("outcome", pd.Series(dtype=str)).astype(str))
    collision_label_ok = "Collision count (raw x)" in outcomes and not any("Collision count log" in x for x in outcomes)
    rows.append({"check": "collision_statistics_label_raw_x", "value": float(collision_label_ok), "criterion": "== 1", "pass": bool(collision_label_ok)})

    # Every analysis family must emit post-hoc outputs in the same run.
    all_post = pd.read_csv(root / "02_statistics" / "all_posthoc_holm.csv")
    required_post_cols = {"posthoc_family", "comparison_scope", "p_uncorrected", "p_holm", "significant_holm", "source_file"}
    missing_post_cols = sorted(required_post_cols - set(all_post.columns))
    rows.append({"check": "all_posthoc_required_columns", "value": float(len(missing_post_cols)), "criterion": "== 0", "pass": len(missing_post_cols) == 0})
    if not missing_post_cols:
        p_unc = pd.to_numeric(all_post["p_uncorrected"], errors="coerce")
        p_holm = pd.to_numeric(all_post["p_holm"], errors="coerce")
        bad_p = int((p_unc.isna() | p_holm.isna() | (p_unc < 0) | (p_unc > 1) | (p_holm < 0) | (p_holm > 1)).sum())
        rows.append({"check": "all_posthoc_pvalues_valid", "value": float(bad_p), "criterion": "== 0", "pass": bad_p == 0})
        scopes = set(all_post["comparison_scope"].dropna().astype(str))
        rows.append({"check": "posthoc_has_marginal_and_simple_effects", "value": float({"marginal", "simple_effect"}.issubset(scopes)), "criterion": "== 1", "pass": {"marginal", "simple_effect"}.issubset(scopes)})
        temporal_scopes = set(all_post.loc[all_post["source_file"].astype(str).str.contains("temporal"), "comparison_scope"].dropna().astype(str))
        temporal_ok = {"all_pairwise_bins", "adjacent_bins"}.issubset(temporal_scopes)
        rows.append({"check": "temporal_posthoc_100ms_allpair_16ms_adjacent", "value": float(temporal_ok), "criterion": "== 1", "pass": temporal_ok})

    q = pd.DataFrame(rows)
    q.to_csv(qcdir / "final_contract_validation.csv", index=False, encoding="utf-8-sig")
    if not bool(q["pass"].all()):
        raise RuntimeError("Final contract validation failed:\n" + q.loc[~q["pass"]].to_string(index=False))

    contract = {
        "contract_version": "AOI_REANALYSIS_V6_HIERARCHICAL_INFORMATION_H1H4",
        "top_level_AOIs": ["Self", "Partner", "Information", "Other"],
        "trial_eligibility_rule": "P01-P36 only; completed==true; decision_steps_completed==180 exactly; valid Time/Horizon; exactly 27 valid trials per participant; exactly 3 trials/3 distinct layouts per Time x Horizon cell",
        "top_level_rule": "Every non-empty gaze window sums to exactly 100%; matched Self/Partner and Information AOIs split dwell equally; Other is residual when none match.",
        "future_step_rule": "Hierarchical composition: first enforce Self+Partner+Information+Other=100%; then subdivide only the already-allocated Information dwell equally across matched H1-H4 steps. In every H4 trial H1+H2+H3+H4 equals Information within floating-point tolerance.",
        "temporal_resolutions_ms": {"overall": "unbinned trial-wide composition", "fine": 1000.0 / 60.0, "coarse": 100.0},
        "response": {"whole": "PRE=[onset,response), POST=[response,deadline)", "symmetric": "up to +/-100 ms, minimum 50 ms each side by default"},
        "gaze_inference_rule": "all gaze inferential analyses use adjusted-logit LMMs; overall uses trial-level rows with (1|Participant); response, primary hierarchical H1-H4 total-fixation, and hierarchical Near/Far use (1|Participant)+(1|Trial); H1-H4 relative allocation uses participant x Time x FutureStep cells with (1|Participant) because the four within-Information shares sum to 1; temporal uses participant-level trajectories marginalized across Horizon/Layout with (1|Participant); Layout is a fixed nuisance effect where trial-level models retain layout",
        "gaze_fixed_effect_test": "sum-coded fixed effects with Wald chi-square omnibus tests; LMM post-hoc estimated-marginal-mean contrasts with Holm correction",
        "posthoc_rule": "single run emits marginal pairwise and interaction simple effects for overall/response/H1-H4; temporal uses all pairwise 100-ms bins and adjacent 16.67-ms bins; behavior/subjective emit marginal pairwise plus Time-within-Horizon and Horizon-within-Time; Holm correction is applied within each explicitly labeled comparison family",
        "collision_rule": "raw num_collisions x; no transformation; layouts averaged within participant-condition before RM-ANOVA",
        "timeout_rule": "trial timeout_rate = num_human_idle / decision_steps_completed; adjusted-logit at trial level then layouts averaged",
        "reaction_time_timeout_rule": "is_human_idle == 1 -> full available decision-time budget; non-timeout missing RT fails closed",
        "raw_lineage": {
            "trial_eligibility": "00_raw_intermediate/trial_eligibility_audit.csv",
            "gaze_segment": "00_raw_intermediate/gaze_segment_allocations.csv",
            "overall_trial": "00_raw_intermediate/overall_trial_aoi_values.csv",
            "temporal_16_67_trial_bin": "00_raw_intermediate/temporal_16.67ms_trial_bin_values.csv",
            "temporal_100_trial_bin": "00_raw_intermediate/temporal_100ms_trial_bin_values.csv",
            "response_event": "00_raw_intermediate/response_event_values.csv",
            "response_decision_phase": "00_raw_intermediate/response_*_decision_phase_values.csv",
            "h1_h4_trial": "00_raw_intermediate/h1_h4_trial_values.csv",
            "h1_h4_near_far_trial": "00_raw_intermediate/h1_h4_near_far_trial_values.csv",
            "behavior_decision": "00_raw_intermediate/behavior_decision_values.csv",
            "behavior_trial": "00_raw_intermediate/behavior_trial_values.csv",
            "subjective_trial": "00_raw_intermediate/subjective_mapped_trial_values.csv",
        },
    }
    (root / "analysis_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Final contract validation: PASS")


if __name__ == "__main__":
    main()
