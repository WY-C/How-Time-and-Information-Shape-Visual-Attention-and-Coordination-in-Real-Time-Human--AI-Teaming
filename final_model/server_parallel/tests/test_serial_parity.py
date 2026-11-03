import hashlib
import math

import numpy as np
import torch

from final_model.common.partners import parameter_hash
from final_model.partial_ver.canonical_training import PartialTrainingSession
from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
)
from final_model.server_parallel.rollout_worker import _trace_delta


def _hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _instrument(session):
    records = [[] for _ in session.adapters]
    for env_id, adapter in enumerate(session.adapters):
        original_collect = adapter.collect_decision
        original_commit = adapter.commit_collected
        backend = adapter.backend
        original_step = backend.step_with_partner_action

        def collect(callback, *, _env=env_id, _original=original_collect):
            acquisitions = []

            def traced(observation, critic, available, timestamp):
                proposal = callback(observation, critic, available, timestamp)
                acquisitions.append({
                    "timestamp": float(timestamp),
                    "observation_sha256": _hash(observation),
                    "action": int(proposal["action"]),
                })
                return proposal

            collected = _original(traced)
            records[_env].append({"acquisitions": acquisitions})
            return collected

        def step(learner_action, partner_action, *, partner_hash_before,
                 _env=env_id, _original=original_step):
            records[_env][-1]["partner_action"] = int(partner_action)
            return _original(
                learner_action, partner_action,
                partner_hash_before=partner_hash_before,
            )

        def commit(collected, *, _env=env_id, _original=original_commit, **kwargs):
            result = _original(collected, **kwargs)
            records[_env][-1].update(
                learner_action=int(result["executed_action"]),
                rewards=[
                    float(result["optimization_reward"]),
                    float(result["sparse_team_reward"]),
                    float(result["native_shaped_reward"]),
                    float(result["weighted_shaped_reward"]),
                ],
                deliveries=int(result["deliveries"]),
                done=bool(result["done"]),
            )
            return result

        adapter.collect_decision = collect
        adapter.commit_collected = commit
        backend.step_with_partner_action = step
    return records


def _server_records(trace, n_envs):
    result = [[] for _ in range(n_envs)]
    for step in trace:
        for env_id in range(n_envs):
            count = step["acquisition_counts"][env_id]
            result[env_id].append({
                "acquisitions": [
                    {
                        "timestamp": step["acquisition_timestamps"][env_id][index],
                        "observation_sha256": step[
                            "acquisition_observation_sha256"
                        ][env_id][index],
                        "action": step["learner_actions"][env_id]
                        if index == count - 1 else None,
                    }
                    for index in range(count)
                ],
                "learner_action": step["learner_actions"][env_id],
                "partner_action": step["partner_actions"][env_id],
                "rewards": step["rewards"][env_id],
                "deliveries": step["deliveries"][env_id],
                "done": step["dones"][env_id],
            })
    return result


def test_canonical_serial_and_server_worker1_are_trajectory_exact(session_options):
    options = session_options()
    canonical = PartialTrainingSession(**options)
    canonical_records = _instrument(canonical)
    canonical_result = canonical.run_update()
    canonical_torch_rng_after = torch.get_rng_state().clone()
    server = ServerParallelActionTrainingSession(
        rollout_workers=1, server_trace=True, **options
    )
    forward_calls = 0
    original_forward = server.policy.actor.forward

    def canonical_forward(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*args, **kwargs)

    def forbidden_get_probs(*_args, **_kwargs):
        raise AssertionError("server learner rollout must use canonical actor forward")

    server.policy.actor.forward = canonical_forward
    server.policy.actor.get_probs = forbidden_get_probs
    try:
        server_result = server.run_update()
        server_records = _server_records(server.last_rollout_trace, 2)
        for env_id in range(2):
            expected = canonical_records[env_id][0]
            actual = server_records[env_id][0]
            assert [row["timestamp"] for row in actual["acquisitions"]] == [
                row["timestamp"] for row in expected["acquisitions"]
            ]
            assert [row["observation_sha256"] for row in actual["acquisitions"]] == [
                row["observation_sha256"] for row in expected["acquisitions"]
            ]
            assert actual["learner_action"] == expected["learner_action"]
            assert actual["partner_action"] == expected["partner_action"]
            assert actual["rewards"] == expected["rewards"]
            assert actual["deliveries"] == expected["deliveries"]
            assert actual["done"] == expected["done"]
        assert server_result["actor_hash_after"] == canonical_result["actor_hash_after"]
        assert server_result["preupdate_parity_pass"] == 1.0
        assert server_result["preupdate_max_abs_logprob_difference"] <= 1e-5
        assert server_result["acquisitions"] == server_result["actionnet_queries"]
        assert server_result["executed_proposals"] == server_result["recurrent_commits"]
        assert forward_calls == server_result["acquisitions"]
        assert server_result["actionnet_rollout_forward_count"] == forward_calls
        assert server_result["actionnet_rollout_forward_count"] == server_result[
            "actionnet_queries"
        ]
        for name in (
            "worker_collect_seconds",
            "actionnet_rollout_seconds",
            "partner_seconds",
            "worker_commit_seconds",
            "worker_finalize_seconds",
            "preupdate_audit_seconds",
            "ppo_seconds",
        ):
            assert math.isfinite(server_result[name])
            assert server_result[name] >= 0.0
        assert torch.equal(torch.get_rng_state(), canonical_torch_rng_after)
        assert server_result["partner_step_guard_checks"] == canonical_result[
            "partner_step_guard_checks"
        ]
        assert parameter_hash(server.policy.actor) == parameter_hash(canonical.policy.actor)
    finally:
        server.close()


def test_bounded_trimmed_worker_matches_untrimmed_canonical_trajectory(session_options):
    options = session_options(
        decision_times=[600],
        horizons=[0],
        fixation_shapes=["plus"],
        rollout_length=32,
        episode_length=400,
        n_rollout_threads=1,
        configured_total_steps=32,
        checkpoint_interval=32,
    )
    canonical = PartialTrainingSession(**options)
    canonical_records = _instrument(canonical)
    canonical_rule_deltas = []
    instrumented_collect = canonical.adapters[0].collect_decision

    def trace_canonical_rule_delta(callback):
        adapter = canonical.adapters[0]
        rule_count = len(adapter.gaze_policy.events)
        collected = instrumented_collect(callback)
        canonical_rule_deltas.append(
            _trace_delta(adapter, len(adapter.events), rule_count)["rule_events"]
        )
        return collected

    canonical.adapters[0].collect_decision = trace_canonical_rule_delta
    canonical_result = canonical.run_update()
    canonical_rng = torch.get_rng_state().clone()
    untrimmed_events = list(canonical.adapters[0].gaze_policy.events)
    assert len(untrimmed_events) > 128

    server = ServerParallelActionTrainingSession(
        rollout_workers=1, server_trace=True, **options
    )
    try:
        server_result = server.run_update()
        actual_records = _server_records(server.last_rollout_trace, 1)[0]
        for actual, expected in zip(actual_records, canonical_records[0]):
            assert [row["timestamp"] for row in actual["acquisitions"]] == [
                row["timestamp"] for row in expected["acquisitions"]
            ]
            assert [
                row["observation_sha256"] for row in actual["acquisitions"]
            ] == [
                row["observation_sha256"] for row in expected["acquisitions"]
            ]
            assert actual["learner_action"] == expected["learner_action"]
            assert actual["partner_action"] == expected["partner_action"]
            assert actual["rewards"] == expected["rewards"]
            assert actual["deliveries"] == expected["deliveries"]
            assert actual["done"] == expected["done"]
        assert server_result["reward"] == canonical_result["reward"]
        assert torch.equal(torch.get_rng_state(), canonical_rng)
        assert all(
            row["worker_trace"][0]["retained_rule_event_count"] <= 128
            for row in server.last_rollout_trace
        )
        generated_trace_events = [
            event
            for row in server.last_rollout_trace
            for event in row["worker_trace"][0]["rule_events"]
        ]
        assert generated_trace_events == [
            event for decision in canonical_rule_deltas for event in decision
        ]
        _backends, adapter_states = server._snapshot_workers()
        server_policy = adapter_states[0]["gaze_policy"]
        canonical_policy = canonical.adapters[0].gaze_policy.training_state()
        assert server_policy["events"] == canonical_policy["events"]
        assert server_policy["rng_state"] == canonical_policy["rng_state"]
        assert server_policy["selection_count"] == canonical_policy["selection_count"]
        assert server_policy["branch_counts"] == canonical_policy["branch_counts"]
    finally:
        server.close()
