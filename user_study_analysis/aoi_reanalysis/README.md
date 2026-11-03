# AOI Reanalysis v4 — 100% Four-AOI Contract + Gaze LMM + Automatic Post-hoc

This bundle is the canonical reanalysis pipeline for the P01–P36 main study.

## Trial eligibility

A trial is included only when all of the following hold:

- participant is P01–P36
- `completed == true`
- `decision_steps_completed == 180` exactly
- Time is 200/400/600 ms
- Horizon is 0/2/4 steps

After exclusion, the pipeline fails closed unless every participant has exactly 27 valid trials and every participant × Time × Horizon cell contains exactly 3 trials from 3 distinct layouts.

## Canonical AOI rule

Every gaze segment is allocated to exactly 100% across:

- Self
- Partner
- Information
- Other

`Other` is residual. Information on an otherwise non-Self/non-Partner location is 100% Information. If Information overlaps Self or Partner, dwell is divided equally across the matched top-level AOIs. Therefore every non-empty window satisfies:

`Self + Partner + Information + Other = 1.0`

For H1–H4, the **already allocated Information dwell** is divided equally across all matched future-step AOIs. H1–H4 therefore cannot exceed top-level Information dwell.

## Statistical split

### Behavioral + subjective outcomes: repeated-measures ANOVA

Behavioral outcomes:

- Task score: raw score
- Collision: **raw `x`**, no `log(1+x)` transformation
- Deadline timeout: trial rate `num_human_idle / decision_steps_completed`, adjusted-logit for inference
- Response time: timeout decisions receive their full 200/400/600-ms budget; non-timeouts retain recorded RT; log(ms) for inference

Subjective outcomes retain the 3×3 Time × Horizon repeated-measures ANOVA.

For RM-ANOVA, Mauchly's test is used and Greenhouse–Geisser correction is applied only when sphericity is violated. The same run automatically emits Holm-adjusted marginal pairwise comparisons plus interaction simple effects (`Time within each Horizon`, `Horizon within each Time`).

### All gaze inference: Linear Mixed-Effects Models (LMM)

Gaze proportions use adjusted-logit transformation for inference. Raw proportions are retained for descriptives and figures.

Fixed-effect omnibus tests use **sum coding + Wald chi-square tests**. LMM post-hoc comparisons are estimated-marginal-mean contrasts from the fitted fixed effects, with Holm correction.

#### Overall 4-AOI composition

Trial-level input; each AOI is modeled separately:

`AOI_logit ~ Time * Horizon + Layout + (1 | Participant)`

For Information, the primary inference excludes H0 because Information is structurally absent:

`Information_logit ~ Time * Horizon(2,4) + Layout + (1 | Participant)`

#### Response-locked PRE/POST gaze

Trial-phase input. Symmetric ±100-ms response windows are the intended primary analysis; whole PRE/POST windows are retained as sensitivity output.

`AOI_logit ~ Phase * Time + Horizon + Layout + (1 | Participant) + (1 | Trial)`

The same run automatically emits LMM estimated-marginal-mean contrasts for marginal Phase, marginal Time, PRE-vs-POST within each Time, and Time pairwise within each Phase. Holm correction is applied within each explicitly labeled comparison family.

#### Temporal evolution — 16.67 ms and 100 ms

For temporal inference, Horizon and Layout are first marginalized within each participant × Time × Bin. Then, separately for each decision-time budget and resolution:

`AOI_logit ~ Bin + (1 | Participant)`

Post-hoc contrasts are derived from the same LMM and Holm-corrected. At 100 ms, all pairwise bins are tested within each AOI × Time family. At 16.67 ms, adjacent-bin contrasts are used to avoid hundreds of low-value frame-pair tests per family. Both resolutions remain LMMs; 100 ms is the cleaner paper-facing resolution, while 16.67 ms is retained for fine-grained temporal characterization.

#### H1–H4 proportion of total fixation

H4 trials only, trial × FutureStep input:

`FutureStep_dwell_logit ~ Time * FutureStep + Layout + (1 | Participant) + (1 | Trial)`

The same run emits marginal Time/FutureStep pairwise contrasts and the interaction simple effects: Time pairwise within each H1–H4 step, and all H1–H4 pairwise contrasts within each Time. Holm correction is applied within each family.


## Automatic post-hoc contract

`run_all.py` performs omnibus tests and all planned follow-ups in the same run. Every post-hoc row contains `posthoc_family`, `comparison_scope`, uncorrected `p`, Holm-adjusted `p_holm`, and a significance flag. Holm correction is performed **within each explicitly labeled family**, not across unrelated outcomes.

- Overall Self/Partner/Other: marginal Time, marginal Horizon, Time within each Horizon, Horizon within each Time.
- Information: H2/H4 only; marginal Time, H4-vs-H2, Time within H2/H4, H4-vs-H2 within each Time. No structural-H0 Information LMM is fit.
- Response: marginal Phase, marginal Time, PRE-vs-POST within each Time, Time pairwise within Pre and within Post.
- Temporal: 100-ms all pairwise bins; 16.67-ms adjacent bins, separately within AOI × Time.
- H1–H4: marginal Time/FutureStep, Time within each future step, FutureStep pairwise within each Time.
- Behavior/Subjective: marginal Time/Horizon, Time within each Horizon, Horizon within each Time.

`02_statistics/all_posthoc_holm.csv` concatenates every post-hoc family into a single audit/reporting table while retaining the source filename.

## Output hierarchy

```text
00_raw_intermediate/
    trial_eligibility_audit.csv
    gaze_segment_allocations.csv
    overall_trial_aoi_values.csv
    temporal_16.67ms_trial_bin_values.csv
    temporal_100ms_trial_bin_values.csv
    response_event_values.csv
    response_*_decision_phase_values.csv
    response_*_trial_phase_values.csv
    h1_h4_trial_values.csv
    behavior_decision_values.csv
    behavior_trial_values.csv
    subjective_mapped_trial_values.csv

01_analysis_inputs/
    overall_trial_lmm_input.csv
    temporal_16.67ms_participant_lmm_input.csv
    temporal_100ms_participant_lmm_input.csv
    response_*_trial_phase_lmm_input.csv
    h1_h4_trial_lmm_input.csv
    behavior_participant_condition_values.csv
    subjective_participant_condition_values.csv

02_statistics/
    overall_aoi_lmm.csv
    overall_aoi_lmm_posthoc_holm.csv
    temporal_within_budget_lmm.csv
    temporal_posthoc_lmm_holm.csv
    temporal_adjacent_bin_lmm_holm.csv
    response_whole_lmm.csv
    response_whole_posthoc_lmm_holm.csv
    response_whole_prepost_lmm_holm.csv      # compatibility alias
    response_symmetric_lmm.csv
    response_symmetric_posthoc_lmm_holm.csv
    response_symmetric_prepost_lmm_holm.csv  # compatibility alias
    h1_h4_lmm.csv
    h1_h4_lmm_posthoc_holm.csv
    behavior_rm_anova.csv
    behavior_posthoc_holm.csv
    subjective_rm_anova.csv
    subjective_posthoc_holm.csv
    all_posthoc_holm.csv

03_descriptives/
    raw-scale participant-balanced means / SD / median / N

04_figures/
    paper-facing PNG figures

05_qc/
    excluded_trials.csv
    trial_count_by_participant.csv
    trial_condition_balance.csv
    gaze_pipeline_qc.csv
    behavior_pipeline_qc.csv
    final_contract_validation.csv

analysis_contract.json
    machine-readable canonical analysis contract
```

`00_raw_intermediate` preserves values before participant-level aggregation. `01_analysis_inputs` contains the actual inferential inputs: trial-level LMM rows for overall gaze, trial-phase rows for response gaze, trial×FutureStep rows for H1–H4, participant×Time×Bin trajectory rows for temporal gaze, and participant-condition RM-ANOVA rows for behavior/subjective outcomes.

## Run

From the project root:

```bash
python user_study_analysis/aoi_reanalysis/run_all.py \
  --roots data/raw/experiment \
  --subjective data/raw/questionnaires/subjective.xlsx \
  --out data/analysis/aoi_reanalysis/results
```

The final validator fails closed if trial balance, 100% AOI conservation, temporal bin structure, H1–H4 conservation, timeout RT, raw-collision rule, the all-gaze-LMM contract, finite LMM fit metrics, or automatic post-hoc outputs are violated.


## v6 hierarchical H1-H4 canonical rule

The top-level allocation is always resolved first:

`Self + Partner + Information + Other = 100%`.

Only the resulting top-level `Information` dwell is subdivided into H1-H4. If a segment's Information allocation is `I` ms and it spatially matches `k` displayed future steps, each matched step receives `I/k` ms. Therefore, for every H4 trial:

`H1 + H2 + H3 + H4 = Information`.

The primary H1-H4 total-fixation metric and all H1-H4 LMM/post-hoc outputs use this hierarchical quantity. The old full-overlay quantity (which could exceed top-level Information because Self/Partner overlap was counted again) is not a canonical output in v6.

Additional outputs:
- `h1_h4_relative_lmm.csv`: H1-H4 shares within Information, summing to 100%.
- `h1_h4_near_far_lmm.csv`: Near=(H1+H2), Far=(H3+H4), derived from the same hierarchical Information partition.

## Figure uncertainty (95% CI)

All mean-based figures use participant-level two-sided t-based 95% confidence intervals:

`mean ± t_(.975, n-1) * SD / sqrt(n)`

The participant is the plotting unit; repeated trials/bins are averaged within participant before CI calculation.
The 100% stacked composition figures are retained without segment error bars because individual segment CIs are not interpretable on a cumulative stack axis. Their uncertainty is provided in the companion CI figures (`overall_aoi_change_by_time.png` and `response_pre_post_ci_*.png`).

To redraw figures only from an existing results directory without rerunning preprocessing/statistics:

```bash
python 05_redraw_figures_with_ci.py --results ./results
```

This overwrites/regenerates the files under `results/04_figures/` and adds `response_pre_post_ci_whole.png` and `response_pre_post_ci_symmetric.png`.
