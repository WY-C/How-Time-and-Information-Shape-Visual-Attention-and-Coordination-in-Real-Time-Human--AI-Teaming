# `parallel_gaze_v1` local validation

Status: smoke/parity/runtime passed; production replacement not started.

Frozen source for the first production run:

- Action checkpoint: `final_model/runs/p2_local_no_decay_production_20260831_150338/no_decay_action/checkpoints/step_020000000.pt`
- SHA-256: `f703842f6f2cb4b65243925ad8c8d5f393e91b03d2d111caab41ba5af4f27656`
- Condition: P2, no memory decay, EMMA tuned

## Contract checks

- Training contract: `parallel_gaze_v1`
- RNG contract: `logical_env_episode_torch_generator_sha256_v1`
- Gaze sampling RNG is keyed by master seed, logical env ID, and episode/reset index.
- Worker ID and process completion order are not sampling or PPO ordering inputs.
- Gaze actor and critic are read-only in workers and hash-verified at every rollout sync.
- PPO updates happen only in the main process after a synchronous rollout barrier.
- Transitions are canonically merged by logical env ID, episode ID, and record sequence ID.
- Serial/foreign checkpoints are rejected as resumes; a serial checkpoint can only be an explicitly recorded new-run initializer in a future contract extension.

## Worker-count parity

The production-shaped parity run used 100 logical envs, rollout length 20, one PPO update, and the same initial weights and master seed.

- 3,094 Gaze transitions in both runs
- sampled-action digest: exact
- rollout-reward digest: exact
- PPO-input digest: exact
- post-update Gaze actor tensors: bit-exact
- post-update Gaze critic tensors: bit-exact
- frozen ActionNet tensors/hash: exact and unchanged
- attributed reward and all 2,000 env-step boundaries: exact
- NaN/Inf: none

The worker=1 checkpoint was also resumed with worker=8 at an episode boundary. Its next rollout, PPO inputs, and post-update Gaze actor/critic were bit-exact to an uninterrupted worker=8 run.

## Runtime

Production-shaped local CPU rollout, after setting each learned-Gaze worker's Torch intra/inter-op pools to one thread:

| Metric | worker=1 | worker=8 | speedup |
| --- | ---: | ---: | ---: |
| Rollout collection | 6.488 s | 3.319 s | 1.95x |
| Whole update | 7.641 s | 4.769 s | 1.60x |

The earlier unconstrained worker=8 trial showed severe Torch thread oversubscription and was rejected. The thread-pool setting is part of the v1 checkpoint contract.

## Serial sanity

A separate 32-decision run from the same initial Action checkpoint and seed compared the old serial RNG contract with v1:

- Gaze transitions: 40 vs 40
- Gaze entropy: 3.688829 vs 3.688828
- acquisitions: 51 vs 52
- actor-active rate: 0.906 vs 0.938
- no NaN/Inf or reward-ownership failure

The difference is consistent with the intentionally different stochastic RNG contract; no abnormal behavioral divergence was observed in this bounded smoke.

## Evidence

- `final_model/pilots/parallel_gaze_v1_w1_runtime_threads1`
- `final_model/pilots/parallel_gaze_v1_w8_runtime_threads1`
- `final_model/pilots/parallel_gaze_v1_serial_sanity`
- `final_model/pilots/parallel_gaze_v1_resume_source_boundary`
- `final_model/pilots/parallel_gaze_v1_resume_boundary_w1_to_w8`
- `final_model/pilots/parallel_gaze_v1_continuous_w8_boundary`

The pre-existing serial production process was not stopped or overwritten during validation.

## Log-prob audit revision

`cpu_same_device_v2` changes only the fail-closed pre-update consistency
audit. Learned-Gaze rollout workers still sample and store canonical PPO
`old_log_prob` values on CPU, and the strict audit now recomputes those values
with a hash-verified CPU copy of the synchronized Gaze actor. The existing
`1e-5` tolerance is unchanged. CUDA remains the owner of the actual Gaze PPO
update, and stored rollout log-probabilities are neither replaced nor
recomputed for PPO.

The training contract remains `parallel_gaze_v1`; checkpoints record
`logprob_audit_version=cpu_same_device_v2` as audit-implementation provenance.
No ActionNet, GazeNet architecture, PPO, reward, environment, RNG, merge-order,
or benchmark `contract_v2` semantics changed.
