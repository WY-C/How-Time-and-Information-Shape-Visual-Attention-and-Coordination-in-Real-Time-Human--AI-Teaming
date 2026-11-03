# Full Action evaluation and behavioral diagnostics

This package evaluates current fixed-H0/H2/H4 and episode-balanced shared-H0/H2/H4 Full Action checkpoints without restoring an optimizer or changing policy parameters. The strict loader requires complete `training_state`, verifies the checkpoint contract and actor/critic content hashes, and rejects Partial/Gaze or unsupported checkpoints. Every CLI creates a new directory and writes a provenance `manifest.json`; existing directories are never overwritten.

## Analyses

Performance uses fixed evaluation conditions, deterministic learner argmax by default, the production backend, 400-step episodes by default, explicit consecutive episode seeds, and canonical fixed partner `policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt`. Deliveries are primary and sparse team reward is secondary. Enable transition events with `--behavior-events`:

```bash
python -m final_model.eval.performance \
  --checkpoint PATH --evaluation-horizon 4 --episodes 100 \
  --seed-start 400000 --device cuda --behavior-events \
  --output-dir final_model/eval/results/final_h4
```

The onion handoff logger assigns a fresh token when an interacting agent places an onion into an empty counter coordinate. Provenance persists while that coordinate contains the onion. Removal by the other agent is a handoff; removal by its placer is a self-repickup. Overcooked states are copied between transitions, so durable object identity is unavailable; a counter is a one-object slot, making coordinate plus generation token the strongest stable identity. Inconsistent and multi-candidate transitions are counted as unresolved or ambiguous rather than attributed.

Rollout preview ablation changes only observation channels 21–24. H4 supports `native h12 h1 none`; H2 supports `native h1 none`; H0 supports `native none`. It never creates information or an explicit horizon label. Modes use paired seeds:

```bash
python -m final_model.eval.preview_ablation \
  --checkpoint SHARED.pt --evaluation-horizon 4 \
  --preview-modes native h12 h1 none --episodes 100 --seed-start 500000
```

Same-state counterfactual analysis runs a native trajectory and re-queries the actor with the exact same non-preview input, pre-action RNN state, mask, available actions, and parameters. Only preview channels change, and returned counterfactual RNN states are discarded. It reports KL, JS, L1, top-1 changes, all action probabilities, and `delta_P_INTERACT`:

```bash
python -m final_model.eval.counterfactual \
  --checkpoint SHARED.pt --evaluation-horizon 4 \
  --episodes 100 --seed-start 600000
```

Rollout ablation is causal over a trajectory after actions diverge. Same-state counterfactual analysis freezes the current state/RNN and measures only the immediate distribution change.

Checkpoint selection consumes performance `summary.json` files from a dedicated selection seed partition. It ranks only mean deliveries, marks values at least 95% of the best as high-performing, and reports neighboring-step stability. Shared selection requires H0/H2/H4 rows and reports aggregate mean, minimum condition mean, and each condition. Behavior metrics are intentionally ignored:

```bash
python -m final_model.eval.checkpoint_selection \
  --shared --summary RUN1_H0/summary.json --summary RUN1_H2/summary.json \
  --summary RUN1_H4/summary.json --output-dir final_model/eval/results/selection
```

For multiple arbitrary checkpoints, provide `run_suite` JSON such as:

```json
{"episodes": 100, "seed_start": 400000, "targets": [
  {"label": "shared_candidate", "checkpoint": "PATH", "evaluation_horizons": [0, 2, 4]}
]}
```

Then run `python -m final_model.eval.run_suite --manifest suite.json --device cuda`. Checkpoint steps are never hard-coded.

Partial v2 checkpoints use the separate acquisition-driven evaluator and must
declare decision time, horizon, footprint, and gaze type. For example:

```json
{"episodes": 10, "episode_length": 400, "targets": [{
  "label": "partial_candidate", "observation_mode": "partial",
  "checkpoint": "PATH", "conditions": [
    {"decision_time": 400, "horizon": 2, "fixation_shape": "plus",
     "gaze_type": "rule_based", "freshness_diagnostic": true}
  ]
}]}
```

Partial evaluation uses no optimizers and verifies ActionNet, critic, GazeNet,
and partner hashes. It uses the same canonical fixed HSP S2-s24 seed-3 partner
as Full evaluation; population sampling is disabled. Its same-state diagnostic changes channel 20 only to
`1.0/0.8/0.4/0.0`, holds the other channels and pre-decision RNN state fixed,
and discards all counterfactual recurrent outputs.

## Gaze behavior rollout artifacts

`gaze_behavior_suite` commits a complete rollout artifact before offline
analysis. The artifact directory is
`OUTPUT/models/MODEL_NAME/`; its schema-versioned `rollout_manifest.json` is
published only after the CSV, compressed event tables, and frozen gaze inputs
have been written. An incomplete marker makes interrupted writes fail closed.

Offline analysis can be repeated without running the environment:

```bash
python -m final_model.eval.gaze_behavior_suite \
  --analysis-only --rollout-artifact OUTPUT/models/MODEL_NAME \
  --bootstrap 20000 --device cuda --output-dir NEW_OUTPUT
```

The loader verifies the checkpoint hash, evaluation conditions, episode and
decision counts, seed range, gaze/action modes, constraint contract, artifact
inventory, and per-file hashes for current-schema artifacts. Phase-3 bootstrap
indices are generated in deterministic chunks of 250 while retaining only the
final scalar statistics.

## Independent seed defaults

- checkpoint selection: 300000
- final performance/behavior: 400000
- rollout ablation: 500000
- same-state counterfactual: 600000

Comparisons within an analysis use paired seeds. Selection episodes must not be reused for final behavioral reporting.

Outputs include episode and summary CSV/JSON, optional behavioral counts/events, optional counterfactual rows/summaries, and hashes for checkpoints, parameters, partner, evaluation sources, and relevant production sources. Opportunity labels are limited to facts directly visible in state and terrain; no undocumented partner-accessibility heuristic is reported.
