"""Bounded canonical/server Full parity matrix and machine-readable audit."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from final_model.common.action_training import ActionTrainingSession
from final_model.common.mappo import _actor_pipeline_snapshot, parameter_hash
from final_model.common.overcooked_backend import cooking_contract
from final_model.server_parallel.full_action_training import (
    ServerParallelFullActionTrainingSession,
    _array_hash,
    _object_hash,
)


ROOT = Path(__file__).resolve().parents[2]
HSP_INITIAL = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"
FIXED_PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "reports/full_parallel_parity.json"
)
PPO_BUFFER_FIELDS = (
    "share_obs", "obs", "rnn_states", "rnn_states_critic", "value_preds",
    "returns", "actions", "action_log_probs", "rewards", "masks",
    "bad_masks", "active_masks", "available_actions", "transition_ids",
)


def _prepared_trace(session, prepared):
    observations, critics, available = prepared
    environments = []
    for env_id, backend in enumerate(session.backends):
        horizon = int(backend.horizon)
        beyond = observations[env_id, ..., 21 + horizon :]
        if not np.array_equal(beyond, np.zeros_like(beyond)):
            raise RuntimeError("Full parity trace found preview beyond horizon")
        environments.append({
            "env_id": env_id,
            "horizon": horizon,
            "predictor_requested": bool(
                session.horizon_sampler is not None
                or int(session.horizon or 0) > 0
            ),
            "predictor_input_sha256": _object_hash((
                backend.state,
                backend.self_history,
                backend.partner_history,
            )),
            "predictor_output_sha256": _array_hash(backend.preview_planes()),
            "preview_channel_sha256": [
                _array_hash(observations[env_id, ..., 21 + index])
                for index in range(4)
            ],
            "actor_observation_sha256": _array_hash(observations[env_id]),
            "critic_observation_sha256": _array_hash(critics[env_id]),
            "available_actions_sha256": _array_hash(available[env_id]),
        })
    return {"environments": environments}


def _install_canonical_trace(session) -> None:
    session.last_rollout_trace = []
    session._current_prepared_trace = _prepared_trace(session, session._prepared)
    session._partner_action_streams = [[] for _ in session.backends]
    for env_id, backend in enumerate(session.backends):
        original_partner_action = backend.partners.action

        def traced_partner_action(
            observation, available, *, _env=env_id, _original=original_partner_action
        ):
            action = _original(observation, available)
            session._partner_action_streams[_env].append(int(action))
            return action

        backend.partners.action = traced_partner_action
    original_actions = session.policy.get_actions

    def traced_actions(*arguments, **keywords):
        output = original_actions(*arguments, **keywords)
        share, observation, actor_rnn, critic_rnn, masks, available = arguments
        values, actions, logs, next_actor, next_critic = output
        snapshot = _actor_pipeline_snapshot(
            session.policy.actor,
            observation,
            actor_rnn,
            masks,
            available,
            actions.detach().cpu().numpy(),
        )
        session._last_actor_trace = {
            "actor_observation_sha256": _array_hash(observation),
            "critic_observation_sha256": _array_hash(share),
            "actor_pre_rnn_sha256": _array_hash(actor_rnn),
            "critic_pre_rnn_sha256": _array_hash(critic_rnn),
            "recurrent_mask_sha256": _array_hash(masks),
            "available_actions_sha256": _array_hash(available),
            "raw_logits_sha256": _array_hash(snapshot["raw_logits"]),
            "probabilities_sha256": _array_hash(snapshot["probabilities"]),
            "actions": actions.detach().cpu().numpy().reshape(-1).tolist(),
            "action_log_probs": logs.detach().cpu().numpy().reshape(-1).tolist(),
            "actor_candidate_rnn_sha256": _array_hash(next_actor),
            "critic_candidate_rnn_sha256": _array_hash(next_critic),
            "values_sha256": _array_hash(values),
        }
        return output

    session.policy.get_actions = traced_actions
    original_prepare = session._prepare_decisions

    def traced_prepare(_session):
        prepared = original_prepare()
        _session._current_prepared_trace = _prepared_trace(_session, prepared)
        return prepared

    session._prepare_decisions = MethodType(traced_prepare, session)
    original_commit = session._commit_decisions

    def traced_commit(_session, actions):
        selected = np.asarray(actions).reshape(-1).astype(np.int64)
        observations, legal_masks = _session._partner_inputs()
        partner_pre_hidden = [
            _array_hash(backend.partners.hidden) for backend in _session.backends
        ]
        partner_rng_before = [
            _array_hash(backend.partners.torch_rng.get_state())
            for backend in _session.backends
        ]
        stream_lengths = [len(values) for values in _session._partner_action_streams]
        results = original_commit(actions)
        _session.last_rollout_trace.append({
            "rollout_step": int(_session.buffer.step),
            "prepared": copy.deepcopy(_session._current_prepared_trace),
            "actor": copy.deepcopy(_session._last_actor_trace),
            "partner_observation_sha256": [
                _array_hash(value) for value in observations
            ],
            "partner_available_actions_sha256": [
                _array_hash(value) for value in legal_masks
            ],
            "partner_pre_rnn_sha256": partner_pre_hidden,
            "partner_rng_before_sha256": partner_rng_before,
            "partner_actions": [
                int(_session._partner_action_streams[env_id][stream_lengths[env_id]])
                for env_id in range(len(_session.backends))
            ],
            "learner_actions": selected.tolist(),
            "results": [{
                "optimization_reward": float(result["optimization_reward"]),
                "sparse_team_reward": float(result["sparse_team_reward"]),
                "native_shaped_reward": float(result["native_shaped_reward"]),
                "weighted_shaped_reward": float(result["weighted_shaped_reward"]),
                "deliveries": int(result["deliveries"]),
                "done": bool(result["done"]),
            } for result in results],
            "backend_state_sha256": [
                _object_hash((
                    backend.state,
                    backend.self_history,
                    backend.partner_history,
                    backend.episode_step,
                    backend.environment_steps,
                    backend.state_generation,
                    backend.horizon,
                ))
                for backend in _session.backends
            ],
        })
        for env_id, backend in enumerate(_session.backends):
            before = partner_rng_before[env_id]
            if _array_hash(backend.partners.torch_rng.get_state()) == before:
                raise RuntimeError("canonical partner RNG did not advance")
        return results

    session._commit_decisions = MethodType(traced_commit, session)


def _install_ppo_capture(session) -> None:
    original = session.trainer.train

    def traced_train(buffer):
        session._parity_ppo_payload = {
            name: _array_hash(getattr(buffer, name)) for name in PPO_BUFFER_FIELDS
        }
        return original(buffer)

    session.trainer.train = traced_train


def _options(condition: str, *, environments: int, rollout_length: int):
    shared = condition == "shared"
    horizon = None if shared else int(condition[1:])
    return {
        "observation_mode": "full",
        "horizon": horizon,
        "shared_horizons": [0, 2, 4] if shared else None,
        "decision_time": None,
        "fixation_shape": None,
        "seed": 20260821,
        "rollout_length": int(rollout_length),
        "episode_length": 1,
        "n_rollout_threads": int(environments),
        "device": "cpu",
        "source_checkpoint": None,
        "resume_checkpoint": None,
        "hsp_initial_checkpoint": HSP_INITIAL,
        "fixed_partner_checkpoint": FIXED_PARTNER,
        "configured_total_steps": int(rollout_length) * int(environments),
        "checkpoint_interval": int(rollout_length) * int(environments),
        "reward_shaping_horizon_steps": 20_000_000,
        "partner": None,
        "inference_batching": "predictor",
        "inference_shadow_audit": False,
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coefs": [0.2, 0.05, 0.01],
        "entropy_coef_horizons": [0, 10_000_000, 20_000_000],
        "ppo_epochs": 1,
        "num_mini_batches": 1,
        "max_grad_norm": 10.0,
    }


def _result_record(session, result, initial_actor, initial_critic):
    trace = session.last_rollout_trace
    partner_actions = []
    if trace and not trace[0]["partner_actions"]:
        raise RuntimeError("partner action trace was not populated")
    return {
        "initial_actor_sha256": initial_actor,
        "initial_critic_sha256": initial_critic,
        "fixed_partner_parameter_sha256": session.backends[0].partners.assert_frozen(
            full_hash=True
        ),
        "cooking_contract": cooking_contract(),
        "trace_sha256": _object_hash(trace),
        "trace": trace,
        "ppo_payload_sha256": session._parity_ppo_payload,
        "actor_hash_before_optimizer": result["actor_hash_before"],
        "actor_hash_after": result["actor_hash_after"],
        "critic_hash_after": parameter_hash(session.policy.critic),
        "preupdate_parity_pass": result["preupdate_parity_pass"],
        "preupdate_max_abs_logprob_difference": result[
            "preupdate_max_abs_logprob_difference"
        ],
        "preupdate_ratio_min": result["preupdate_ratio_min"],
        "preupdate_ratio_max": result["preupdate_ratio_max"],
        "preupdate_approx_kl": result["preupdate_approx_kl"],
        "preupdate_clip_fraction": result["preupdate_clip_fraction"],
        "reward": result["reward"],
        "sparse_team_reward": result["sparse_team_reward"],
        "native_shaped_reward": result["native_shaped_reward"],
        "weighted_shaped_reward": result["weighted_shaped_reward"],
        "deliveries": result["deliveries"],
        "partner_guard_failures": result["partner_step_guard_failures"],
        "predictor_forward_calls": result["predictor_forward_calls"],
        "predictor_batch_parity": result["predictor_batch_parity"],
        "shared_horizon_decision_counts": result.get(
            "shared_horizon_decision_counts"
        ),
        "shared_horizon_episode_counts": result.get(
            "shared_horizon_episode_counts"
        ),
    }


def _run_canonical(options):
    session = ActionTrainingSession(**options)
    _install_canonical_trace(session)
    _install_ppo_capture(session)
    initial_actor = parameter_hash(session.policy.actor)
    initial_critic = parameter_hash(session.policy.critic)
    result = session.run_update()
    record = _result_record(session, result, initial_actor, initial_critic)
    executor = getattr(session, "_executor", None)
    if executor is not None:
        executor.shutdown(wait=True)
        session._executor = None
    return record


def _run_server(options, workers: int):
    session = ServerParallelFullActionTrainingSession(
        rollout_workers=int(workers), server_trace=True, **options
    )
    _install_ppo_capture(session)
    initial_actor = parameter_hash(session.policy.actor)
    initial_critic = parameter_hash(session.policy.critic)
    try:
        result = session.run_update()
        return _result_record(
            session, result, initial_actor, initial_critic
        )
    finally:
        session.close()


def _assert_equal(left, right, *, label: str) -> None:
    exact_keys = (
        "initial_actor_sha256", "initial_critic_sha256",
        "fixed_partner_parameter_sha256", "cooking_contract", "trace_sha256",
        "trace", "ppo_payload_sha256", "actor_hash_before_optimizer",
        "actor_hash_after", "critic_hash_after", "reward",
        "sparse_team_reward", "native_shaped_reward", "weighted_shaped_reward",
        "deliveries", "partner_guard_failures", "predictor_forward_calls",
        "shared_horizon_decision_counts", "shared_horizon_episode_counts",
    )
    mismatch = [key for key in exact_keys if left[key] != right[key]]
    if mismatch:
        raise RuntimeError(f"Full parity mismatch {label}: {mismatch}")
    for record in (left, right):
        if record["preupdate_parity_pass"] != 1.0:
            raise RuntimeError(f"Full preupdate parity failed {label}")
        if record["preupdate_max_abs_logprob_difference"] > 1e-5:
            raise RuntimeError(f"Full preupdate tolerance failed {label}")
        if abs(record["preupdate_approx_kl"]) > 1e-6:
            raise RuntimeError(f"Full preupdate KL failed {label}")
        if record["preupdate_clip_fraction"] != 0.0:
            raise RuntimeError(f"Full preupdate clip fraction failed {label}")


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16, 32])
    value.add_argument("--environments", type=int, default=32)
    value.add_argument("--rollout-length", type=int, default=3)
    value.add_argument(
        "--conditions", nargs="+", default=["h0", "h2", "h4", "shared"]
    )
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def run(args) -> dict[str, object]:
    if max(args.workers) > args.environments:
        raise ValueError("parity workers may not exceed environments")
    matrix = {}
    for condition in args.conditions:
        options = _options(
            condition,
            environments=args.environments,
            rollout_length=args.rollout_length,
        )
        canonical = _run_canonical(options)
        server = {str(workers): _run_server(options, workers) for workers in args.workers}
        _assert_equal(canonical, server[str(args.workers[0])], label=(
            f"{condition}:canonical-vs-workers{args.workers[0]}"
        ))
        reference = server[str(args.workers[0])]
        for workers in args.workers[1:]:
            _assert_equal(reference, server[str(workers)], label=(
                f"{condition}:workers{args.workers[0]}-vs-workers{workers}"
            ))
        predictor = server[str(args.workers[0])]["predictor_batch_parity"]
        if predictor is not None and (
            predictor["displayed_preview_agreement"] != 1.0
            or predictor["first_displayed_preview_divergence"] is not None
        ):
            raise RuntimeError(f"Full predictor batch parity failed: {condition}")
        if condition == "shared":
            expected = args.environments * args.rollout_length // 3
            counts = reference["shared_horizon_decision_counts"]
            if counts != {"h0": expected, "h2": expected, "h4": expected}:
                raise RuntimeError(f"shared horizon balance failed: {counts}")
        matrix[condition] = {
            "canonical_vs_workers1": True,
            "worker_count_parity": {str(value): True for value in args.workers},
            "trace_sha256": canonical["trace_sha256"],
            "actor_hash_after": canonical["actor_hash_after"],
            "critic_hash_after": canonical["critic_hash_after"],
            "ppo_payload_sha256": canonical["ppo_payload_sha256"],
            "predictor_batch_parity": predictor,
            "shared_horizon_decision_counts": canonical[
                "shared_horizon_decision_counts"
            ],
            "preupdate": {
                key: canonical[key]
                for key in (
                    "preupdate_max_abs_logprob_difference",
                    "preupdate_ratio_min", "preupdate_ratio_max",
                    "preupdate_approx_kl", "preupdate_clip_fraction",
                )
            },
        }
    report = {
        "contract": "full_server_parallel_parity_v1",
        "conditions": matrix,
        "workers": list(args.workers),
        "environments": args.environments,
        "rollout_length": args.rollout_length,
        "all_passed": True,
        "long_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    print(json.dumps(run(parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
