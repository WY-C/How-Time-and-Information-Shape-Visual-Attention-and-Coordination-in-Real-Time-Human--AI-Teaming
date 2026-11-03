# GazeNet v1 deferred production integration plan

This plan is intentionally not wired into production while the Partial Action trainer is
active. The isolated reference implementation is `server_parallel/gaze_v1.py`.

## Reusable canonical mechanisms

- `partial_ver.gaze.GazeStateMachine`: keep READY/PREP/EXEC/DWELL timing and committed
  operations across task-decision boundaries.
- `partial_ver.observation.PartialObservationAdapter`: keep canonical 2400-ms memory,
  acquisition-before-ActionNet ordering, and ActionNet query/commit semantics.
- `partial_ver.conditions.PartialConditionScheduler`: configure decision times
  `[200,400,600]`, horizons `[0,2,4]`, and fixation shapes `[plus]` for shuffled,
  episode-balanced nine-condition exposure.
- `partial_ver.training.LearnedGazeController`: retain the open-event accumulation model:
  each distinct env-step task reward is accrued once, the previous transition closes at
  the next legitimate READY selection, and terminal flush closes the final transition.
- `partial_ver.training.gaze_ppo_update`: retain PPO ratio/clipping mechanics after adding
  the required audit and observability tests.
- Canonical Partial checkpoint/resume machinery: retain scheduler, adapter, backend,
  controller, optimizer, buffer, and global RNG state restoration.

## Authoritative mismatches to correct after the trainer exits

1. `partial_ver/training.py`: the current fixation plane is `255`; v1 requires exactly `1`.
2. `partial_ver/observation.py`: learned-gaze remaining time currently divides by the
   current decision time; v1 requires clamped absolute `remaining_ms / 600`.
3. `partial_ver/training.py` and `eval/partial_rollout.py`: current action decoding uses
   `(index // 5, index % 5)`; v1 requires `(index % 8, index // 8)`.
4. `partial_ver/network.py`: the layer sizes match, but its `/255` spatial preprocessing
   is coupled to the old `255` fixation plane. Adopt the authoritative v1 input scaling
   and assign a provenance ID that cannot be confused with old learned-gaze checkpoints.

## Files that will require production edits

- `partial_ver/network.py`: canonicalize the isolated v1 actor/critic and contract IDs.
- `partial_ver/training.py`: use the v1 builder/mapping while preserving event credit.
- `partial_ver/observation.py`: supply the v1 context only at legitimate READY queries;
  preserve committed operations across boundaries and rule-gaze trajectory parity.
- `partial_ver/canonical_training.py`: enforce gaze-only optimization with frozen ActionNet,
  configure the nine-condition scheduler, record v1 contracts, and restore open gaze
  transitions fail-closed.
- `eval/partial_rollout.py`: use the same v1 builder and row-major action mapping for learned
  gaze evaluation.
- `server_parallel/train_server.py`, `parallel_action_training.py`, `rollout_worker.py`,
  `protocol.py`, and `shared_buffers.py`: add a learned-gaze-only execution path after exact
  parity design. READY event observations must reach GPU GazeNet exactly once; PREP/EXEC/DWELL
  must never trigger a forward pass.
- `server_parallel/observability.py`: add overall and nine-condition read-only aggregation.
- `eval/checkpoint_loader.py` or a dedicated server validator: validate GazeNet v1 architecture,
  Action source provenance, partner/cooking/memory contracts, and hashes.

## Frozen ActionNet loading

Require `--action-checkpoint PATH`; never hard-code a step. Validate a complete production
Partial Action checkpoint, record file/actor/critic hashes, load actor (and critic only if
needed for diagnostics), call `eval().requires_grad_(False)`, exclude Action parameters from
all optimizers, and assert the actor hash before/after every update and at checkpoint save.
The fixed HSP partner remains frozen and hash-guarded.

## Event reward and PPO flow

Each READY selection opens one gaze transition. The controller stores observation, context,
action, old log probability, value, and a zero task-reward accumulator. Every env.step calls
`accrue` once with that step's canonical optimization reward. The next READY selection closes
the previous transition with its accumulated reward and bootstrapped next value. Terminal
flush closes it with done=true and zero bootstrap. No reward is copied across multiple gaze
actions, and no gaze-specific shaping term is added. Add an identity audit comparing the sum
of env-step rewards with the sum of closed plus still-open transition reward accumulators.

## Checkpoint and resume provenance

Record the frozen Partial Action checkpoint absolute path/content SHA, actor/critic hashes,
GazeNet actor/critic architecture and parameter hashes, task-only reward contract, row-major
action contract, input contract, nine-condition scheduler state/counts, gaze actor/critic
optimizers, open/completed per-environment gaze transitions, gaze timing state, adapter/backend
state, partner recurrent/RNG state, Python/NumPy/Torch CPU/CUDA RNG, update count, and cumulative
environment steps. Resume must fail closed on any mismatch.

## W&B and JSONL metrics

Emit existing rollouts only—no diagnostic forwards. Overall and each T200/T400/T600 x H0/H2/H4
bucket must include sparse reward, deliveries/episode, gaze actor/critic loss, entropy, approx KL,
clip fraction, explained variance, actor/critic grad norms, requests/decision,
acquisitions/decision, same/new-target rates, mean saccade distance, target entropy/frequencies,
valid-memory/freshness/expiry, cross-decision gaze-operation rate, no-acquisition rate, and
forced-STAY rate. W&B step remains cumulative environment steps.

## Validation before training

Run unit tests, event-credit conservation tests, READY-only forward-count tests, fixed-Action
hash tests, nine-condition balance tests, checkpoint/resume exactness, logging on/off trajectory
parity, bounded no-optimization evaluation, and a bounded Gaze PPO smoke. Do not start long
GazeNet training until the frozen ActionNet Time x Horizon viability report is reviewed.
