# Human–model temporal benchmark

This benchmark is evaluation-only. It does not modify training, checkpoint
payloads, or the legacy acquisition-triggered evaluator.

## Frozen contract

The current benchmark is `contract_v2`. Its machine-readable frozen snapshot is
`contracts/contract_v2.json`. Existing definitions and regression targets must
not be edited in place. Any change that alters a benchmark metric definition or
target requires a new contract snapshot and explicit version selection.

Every model run writes `benchmark/evaluation_manifest.json` containing the
contract version and snapshot hash, git commit and dirty-worktree flag, both
checkpoint hashes, tau, seed-manifest hash, primary AOI contract, and evaluation
action mode.

Model AOI composition uses simulator-defined fixation intervals under
`simulator_phase_central_fixation_exec_excluded_preview_first_v1`. PREP is
assigned to the source central tile, EXEC is emitted as an unclassified
`saccade_segment`, and post-landing encoding is assigned to the target central
tile. Decision boundaries split intervals without resetting gaze state. The
four exclusive AOIs use `Action Preview > Player > Teammate > Other`; PLUS and
configured acquisition-footprint hits remain separate coverage diagnostics.
`summaries/fixation_duration_by_decision.csv` and
`summaries/fixation_duration_by_episode.csv` contain the explicit duration,
saccade, classified-denominator, and fixation-proportion fields.

## Action contract

`periodic_20ms_threshold` queries ActionNet at `0, 20, ... < T` with all six
actions available. Gaze is advanced to the query timestamp before the actor is
called, so an acquisition completing exactly at a query is visible. Gaze then
continues to the deadline without a deadline ActionNet query. STAY may be an
intentional threshold selection; no crossing produces a distinct forced STAY.

Within a decision, every query uses the pre-decision recurrent state. The
candidate state returned by the final query is committed once at the decision
boundary. This preserves one recurrent commit per environment decision.

## Human contracts

- Primary AOI: `manuscript_preview_priority` (`Action Preview > Player >
  Teammate > Other`).
- Sensitivity AOI: `v6_overlap_split_sensitivity`; never mixed into the primary
  human reference.
- `post_first_input_fraction` and `post_selected_action_fraction` are separate.
- Process eligibility is computed from validity metadata, never participant ID.
- Context-dependent RQ3 is registered as `CANNOT_VERIFY_PROVENANCE` until the
  exact original held-object formula is recovered.
- Corrected task-event reconstruction checks AI facing, held object, target pot,
  and the actual `2 -> ready` transition.

## Commands

Human reproduction:

```bash
python -m final_model.eval.benchmark_cli human \
  --data-root data/raw/experiment \
  --output final_model/eval/results/human_benchmark_run
```

Small model pilot with an explicitly paired ActionNet and GazeNet:

```bash
python -m final_model.eval.benchmark_cli model \
  --action-checkpoint final_model/checkpoint/action/action_baseline_memory1200.pt \
  --gaze-checkpoint final_model/checkpoint/gaze/baseline_1200_17m.pt \
  --output final_model/eval/results/temporal_pilot \
  --times 200,400,600 --horizons 0,2,4 \
  --episodes 1 --episode-length 5 --seed-start 810000 \
  --tau 0.5 --device cpu
```

Join the primary-contract human and model reference tables:

```bash
python -m final_model.eval.benchmark_cli compare \
  --human-reference final_model/eval/results/human_benchmark_run/benchmark/human_reference.csv \
  --model-reference final_model/eval/results/temporal_pilot/benchmark/model_reference.csv \
  --output final_model/eval/results/temporal_pilot/benchmark/human_model_comparison.csv
```

After a matching 2400-ms GazeNet pair and one global tau have been frozen, the
fixed-seed production command is the same model command with
`--episodes 100 --episode-length 180 --seed-start 810000`. Do not run it with a
placeholder pair or an unfrozen tau.

The 2400-ms ActionNet has no embedded GazeNet. A tau calibration using it must
remain `PROVISIONAL` until its intended GazeNet pairing is explicitly supplied
and validated. The configurable tau grid/sweep belongs to Phase 5 and is not
activated in this Phase 2--4 implementation. No production sweep is started by
these commands.

## Top-1 minus top-2 margin contract

`contract_v3` preserves the benchmark definitions above but replaces the action
threshold score with `top1_probability - top2_probability`. The original
`contract_v2` max-probability mode remains the default and is not modified.

A margin-threshold pilot is selected explicitly. The CLI infers `contract_v3`
from this action mode and rejects a mismatched explicit contract version.

```bash
python -m final_model.eval.benchmark_cli model \
  --action-checkpoint ACTION.pt \
  --gaze-checkpoint GAZE.pt \
  --output final_model/eval/results/margin_pilot \
  --times 400 --horizons 2 \
  --episodes 1 --episode-length 5 --seed-start 800000 \
  --action-mode periodic_20ms_margin_threshold \
  --tau 0.3 --device cpu
```

Each action-query row records `argmax_probability`, `top2_probability`,
`top1_top2_margin`, `threshold_rule`, and the applied `threshold_score`. Margin
tau is provisional until calibrated with seeds disjoint from production.

## Fixed motor-delay margin contract

`contract_v4` keeps the 20-ms margin queries and freezes one rule for every
Time and Horizon condition: global margin tau `0.18` and motor delay
`87.607603 ms`. A threshold crossing at time `t` schedules its selected input
at `t + 87.607603 ms`. Only inputs arriving by the decision deadline can select
that decision's action; otherwise it is a forced STAY.

Forced STAY and intentional STAY remain separate diagnostics. The additional
`no_action_rate` is their union, so intentional model STAY choices are included
when comparing physical non-action behavior.

The v4 semantic/foveal PLUS outputs are nonexclusive coverage metrics, not an
AOI composition. Self, Partner, and Preview are independently marked when the
five-tile PLUS footprint intersects them; Other is marked only when none does.
A physical fixation start contributes one fixation hit. Decision coverage is
binary per AOI and full trials report checked decisions over the canonical 180.
Pre/post outputs split physical fixation starts at the delayed resolved
selected-action input and store hit share, hits per interval duration, and
post-minus-pre changes. Forced timeouts are excluded from this split, while an
intentional STAY with an input is retained.

Collision output separates partner collision from a static environment block
for cardinal learner movements. STAY and INTERACT are excluded. Episode and
condition summaries retain deliveries, sparse team reward, response time, and
all three forced/intentional/combined no-action rates.

```bash
python -m final_model.eval.benchmark_cli model \
  --action-checkpoint ACTION.pt \
  --gaze-checkpoint GAZE.pt \
  --output final_model/eval/results/margin_motor_delay_pilot \
  --times 200,400,600 --horizons 0,2,4 \
  --episodes 1 --episode-length 5 --seed-start 800000 \
  --action-mode periodic_20ms_margin_motor_delay \
  --tau 0.18 --device cpu
```

The CLI supplies the frozen motor delay for this mode. Supplying a different
tau or `--motor-delay-ms` value is rejected rather than silently changing the
contract.

## Evaluation-only logit commitment calibration

`contract_v7` adds an opt-in ActionNet commitment mode. It thresholds the raw
top-1 minus top-2 logit gap and keeps the motor delay fixed at `87.607603 ms`.
Fixed, linear, and exponential bounds share one global trajectory based on
absolute elapsed milliseconds; there are no T-specific parameters and no
`t/T` normalization. Existing modes and their defaults are unchanged.

Existing query logs that contain all six softmax probabilities can be replayed
offline because `z_top1 - z_top2 = log(p_top1 / p_top2)`. This replay is exact
for commitment decisions on the logged query trajectories. It is not an exact
counterfactual environment rollout after the newly executed action first
diverges, so selected candidates should subsequently be rerun with v7.

```bash
python -m final_model.eval.calibrate_commitment_rule \
  --queries MODEL_RUN/raw/action_queries.csv.gz \
  --human-decision-summary HUMAN_RUN/summaries/decision_behavior.csv \
  --human-latency-summary HUMAN_RUN/summaries/selected_action_latency.csv \
  --calibration-seeds 810000,810001,810002 \
  --validation-seeds 810003,810004 \
  --output final_model/eval/results/commitment_calibration
```

The calibration output contains `candidate_metrics.csv`, `model_summary.csv`,
`search_trace.csv`, `threshold_trajectory.csv`, and a hash-bearing manifest.
Only forced-STAY rate and conditional selected-action latency enter the
objective. Reward, revisions, AOI, gaze, fixation, preview, and task metrics
remain independent validation outcomes.

An exponential candidate can be rerun without changing either checkpoint:

```bash
python -m final_model.eval.benchmark_cli model \
  --action-checkpoint ACTION_GAZE_PAIR.pt \
  --gaze-checkpoint ACTION_GAZE_PAIR.pt \
  --output final_model/eval/results/logit_commitment_rerun \
  --times 200,400,600 --horizons 0,2,4 \
  --episodes 5 --episode-length 180 --seed-start 810000 \
  --action-mode periodic_20ms_logit_commitment_stochastic_gaze \
  --commitment-rule exponential --tau 0 \
  --tau-start 1.0854054592484133 --tau-min 0 \
  --tau-kappa-ms 219.05884145273762 --device cpu
```

## Joint action-input-delay calibration

`contract_v8` preserves the v7 evidence, absolute-time bounds, proposal,
deadline, STAY, and revision semantics. It jointly fits one global
`action_input_delay_ms` with each bound family. `tau_min` remains fixed at zero.
The legacy v7 mode still requires `87.607603 ms`; v8 is selected explicitly and
requires `--action-input-delay-ms`.

Family parameters and family selection use calibration seeds only. The
pre-specified rule selects exponential only when its calibration objective is
more than 5% lower than linear. Validation seeds are evaluated only after this
selection. The calibrator records optimizer restarts, delay-bound diagnostics,
seed-level metrics, objective components, and an optional direct comparison to
a fixed-delay v7 result directory.

```bash
python -m final_model.eval.calibrate_commitment_rule \
  --queries MODEL_RUN/raw/action_queries.csv.gz \
  --human-decision-summary HUMAN_RUN/summaries/decision_behavior.csv \
  --human-latency-summary HUMAN_RUN/summaries/selected_action_latency.csv \
  --calibration-seeds 810000,810001,810002 \
  --validation-seeds 810003,810004 \
  --delay-upper-ms 240 --stability-restarts 3 \
  --maxiter 60 --popsize 8 \
  --fixed-delay-reference FIXED_DELAY_V7_CALIBRATION \
  --output final_model/eval/results/joint_delay_v8_calibration
```

The exact fixed, linear, and exponential full rerollout commands for the 4M
fit are stored in `run_joint_delay_v8_full_rerollouts.sh`. Those commands use
the same checkpoint, T/H cells, five seeds, and 180-decision episode length.

## No-action-commitment comparator

`contract_v9` adds an evaluation-only comparator that removes the ActionNet
commitment threshold and action-input delay. Every 20 ms query immediately
replaces the latest top-1 proposal, including intentional STAY, and the
decision deadline executes the latest proposal. Deadline, gaze latency,
60 ms acquisition, bounded observation, memory, preview, and stochastic
GazeNet behavior remain unchanged. Existing v7 and v8 contracts are unchanged.

```bash
python -m final_model.eval.benchmark_cli model \
  --action-checkpoint ACTION_GAZE_PAIR.pt \
  --gaze-checkpoint ACTION_GAZE_PAIR.pt \
  --output final_model/eval/results/no_commitment_rerun \
  --times 200,400,600 --horizons 0,2,4 \
  --episodes 5 --episode-length 180 --seed-start 810000 \
  --action-mode periodic_20ms_no_commitment_stochastic_gaze \
  --commitment-rule none --tau 0 --motor-delay-ms 0 --device cpu
```

Pre/post-input partitions are timing diagnostics only and are not directly
comparable between delayed and zero-delay conditions. Use the full decision
window for the primary gaze, AOI, preview, and visual-sampling comparison.
