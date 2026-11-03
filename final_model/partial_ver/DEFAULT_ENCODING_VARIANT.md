# Default-encoding perception and stochastic-memory variant

This variant is opt-in. It does not change `default`, `tuned`,
`human_stochastic`, legacy `local`/`global` peripheral vision, linear memory,
or no-decay memory.

## Audited baseline

- Movement timing lives in `partial_ver/gaze.py`. Repository-default movement
  is preparation `135 ms`, followed by execution `70 + 2 * distance_deg ms`.
  Existing tuned and human-stochastic modes use preparation `135 ms` and the
  fitted execution `44.8732080135 + 3.83832565515 * distance_deg ms`.
- Existing fixed modes acquire after a `60 ms` dwell. Human-stochastic mode
  acquires after its `60 ms` encoding floor and may then remain in HOLD.
- Exact footprint candidates come from `gaze.acquisition_tiles`. Active
  peripheral candidates are composed in `peripheral.py`; legacy local uses the
  four PLUS diagonals and legacy global uses every board tile outside PLUS.
- Peripheral probability is shared Gaussian angular decay
  `exp(-theta^2 / (2 * sigma_deg^2))`, with independent PCG64 Bernoulli draws.
- Visual angle uses tile-center rays on an 8x5 grid, 90x90 viewport pixels per
  tile, a 2560x1440 16:9 685.8-mm monitor, and 600-mm viewing distance.
- Linear memory freshness is `clip(1-age/lifetime, 0, 1)` and deletion occurs
  at `age >= lifetime`. No-decay memory remains binary-valid until reset.
- Bounded observation channel 20 stores `255 * freshness`. ActionNet receives
  that channel directly; GazeNet normalizes it to `[0,1]` in
  `server_parallel/gaze_v1.py`.
- Local rollout constructs each adapter from the session configuration. Server
  rollout serializes the same constraint configuration and per-environment RNG
  seeds into each worker. Evaluation reconstructs it from checkpoint metadata.

## New EMMA timing

Select with `--emma-mode default_encoding`.

- Movement parameters: preparation `135 ms`; execution
  `70 + 2 * distance_deg ms`.
- Encoding: `T_enc = 0.006 * [-ln(0.1)] * exp(0.4 * epsilon)` seconds.
- Familiarity is fixed at `0.1` for every tile and semantic category.
- Encoding begins concurrently with movement preparation. If movement lands
  before encoding completes, the uncompleted proportion is multiplied by the
  new foveal encoding time, following Salvucci's EMMA remaining-work rule.
  The repository emits its atomic multi-tile fixation snapshot only after both
  landing and target encoding are complete. It never computes
  `prep + execution + full encoding`.

The source for the default constants and parallel control flow is D. D.
Salvucci, *An Integrated Model of Eye Movements and Visual Encoding*, Cognitive
Systems Research 1 (2001), 201-220:
https://www.cs.drexel.edu/~salvucci/publications/Salvucci-CSR01.pdf

## New spatial extents

The exact core is always PLUS 5. `--peripheral-extent diag3x3` adds only the
four diagonals. `--peripheral-extent diamond13` adds the Manhattan-distance-2
ring. Both use the same `--peripheral-sigma-deg`; extent changes support only.

## New stochastic memory

Select with `--memory-decay-mode stochastic_exponential`, explicitly providing
`--memory-tau-ms` and `--memory-max-lifetime-ms`. Supported experiment pairs
are `(400,1200)` and `(800,2400)` milliseconds.

Each acquisition samples `L ~ Exponential(tau)` from a per-adapter PCG64
stream and stores `min(L, T_max)`. Re-acquisition replaces content, resets age,
and samples a new lifetime. A live entry keeps its content intact while channel
20 is `255` (semantic freshness `1`). At the sampled lifetime it is cleared and
channel 20 becomes `0`; no entry can survive at or beyond `T_max`. The
exponential law is applied only to lifetime sampling, not a second time to the
freshness value. RNG state and per-tile lifetimes are checkpointed.

## Action decision boundary

In `default_encoding`, visual acquisitions update perception and memory only.
After the full task-decision interval has elapsed, ActionNet receives the final
bounded observation and is queried exactly once. That single sampled action and
its recurrent candidate state are committed at the decision boundary. Multiple
within-decision acquisitions therefore do not create or revise ActionNet
proposals. Existing EMMA modes retain their previous ActionNet query semantics.

## Factorial launcher

`run_default_encoding_factorial.sh` emits the four matched Action-then-Gaze
commands by default. It does not start training in its default `MODE=print`.
Use `MODE=smoke` for the bounded four-condition smoke and reserve `MODE=full`
for a separately authorized full run.

For the separately trained relaxed-saccade comparator, set
`EMMA_MODE=relaxed_saccade`. This preserves all perception/memory settings while
multiplying preparation, execution intercept, execution slope, and encoding
latency by exactly `0.5`: preparation `67.5 ms`; execution
`35 + 1 * distance_deg ms`; encoding `K=0.003 s`. Its checkpoints use a
distinct timing contract and cannot be silently mixed with default-encoding
checkpoints.
