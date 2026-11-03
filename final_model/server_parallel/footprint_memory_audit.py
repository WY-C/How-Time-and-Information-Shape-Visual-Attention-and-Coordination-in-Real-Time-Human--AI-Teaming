"""Bounded, no-optimization Partial-memory audit across fixation footprints."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.distributions import Categorical

from final_model.common.mappo import parameter_hash
from final_model.common.partners import FIXED_HSP_PARTNER_ID
from final_model.common.paths import file_sha256
from final_model.partial_ver.memory import (
    CONTINUOUS_FRESHNESS_CONTRACT_ID,
    MEMORY_LIFETIME_MS,
)
from final_model.partial_ver.rule_based import RULE_BASED_GAZE_POLICY_ID
from final_model.server_parallel.footprint_audit_recorder import (
    BOARD_TILES,
    NOMINAL_TILES,
)
from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
    validate_partial_full_source_checkpoint,
)
from final_model.server_parallel.protocol import MAX_ACQUISITIONS_PER_DECISION


ROOT = Path(__file__).resolve().parents[2]
SELECTED_SOURCE = (
    ROOT / "final_model/eval/checkpoints/seed20260821/step_040000000.pt"
)
SELECTED_SOURCE_SHA256 = (
    "c0f076269b279c25d9c80a7c1bfa0a0758a31432392e70059544ab6ac7c27cc5"
)
FIXED_PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
RULE_GAZE_PROBABILITIES = {
    "task": 0.60,
    "stay": 0.20,
    "stale": 0.15,
    "random": 0.05,
}
PHASES = ("pre_gaze_selection", "post_acquisition", "task_deadline")


def _hash_update(digest, value: Any) -> None:
    if value is None:
        digest.update(b"none")
    elif isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, (int, np.integer)):
        digest.update(f"int:{int(value)}".encode("ascii"))
    elif isinstance(value, (float, np.floating)):
        digest.update(f"float:{float(value).hex()}".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
    elif isinstance(value, Path):
        _hash_update(digest, str(value))
    elif torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(b"tensor:")
        _hash_update(digest, array)
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f"array:{array.dtype}:{array.shape}:".encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_update(digest, key)
            _hash_update(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(f"sequence:{len(value)}:[".encode("ascii"))
        for item in value:
            _hash_update(digest, item)
        digest.update(b"]")
    elif isinstance(value, set):
        digest.update(b"set{")
        for item in sorted(value, key=lambda item: repr(item)):
            _hash_update(digest, item)
        digest.update(b"}")
    elif isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}:".encode("ascii"))
        digest.update(value)
    elif hasattr(value, "__dict__"):
        digest.update(
            f"object:{type(value).__module__}.{type(value).__qualname__}:".encode(
                "utf-8"
            )
        )
        _hash_update(digest, vars(value))
    else:
        digest.update(
            f"repr:{type(value).__module__}.{type(value).__qualname__}:".encode(
                "utf-8"
            )
        )
        digest.update(repr(value).encode("utf-8"))


def stable_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return 0.0 if not array.size else float(array.mean())


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return 0.0 if not array.size else float(np.median(array))


def _fraction(condition) -> float:
    array = np.asarray(condition, dtype=bool)
    return 0.0 if not array.size else float(array.mean())


def _describe_phase(rows: list[dict[str, object]], horizon: int | None) -> dict:
    counts = np.asarray(
        [value for row in rows for value in row["valid_counts"]],
        dtype=np.float64,
    )
    coverage = counts / BOARD_TILES
    tile_samples = sum(int(row["freshness_tile_samples"]) for row in rows)
    normalized_sum = sum(
        float(row["freshness_normalized_sum"]) for row in rows
    )
    normalized_square_sum = sum(
        float(row["freshness_normalized_square_sum"]) for row in rows
    )
    normalized_mean = normalized_sum / max(tile_samples, 1)
    normalized_variance = max(
        0.0,
        normalized_square_sum / max(tile_samples, 1) - normalized_mean ** 2,
    )
    preview_nonzero = sum(
        int(row["preview_nonzero_tile_count"]) for row in rows
    )
    valid_preview_nonzero = sum(
        int(row["valid_preview_nonzero_tile_count"]) for row in rows
    )
    valid_tiles = sum(int(row["valid_tile_count"]) for row in rows)
    preview_channels = np.sum(
        np.asarray(
            [row["preview_channel_nonzero_counts"] for row in rows],
            dtype=np.int64,
        ),
        axis=0,
    ) if rows else np.zeros(4, dtype=np.int64)
    applicable = [] if horizon == 0 else list(range(1, (horizon or 4) + 1))
    percentiles = (
        np.percentile(coverage, [10, 25, 75, 90])
        if coverage.size else np.zeros(4, dtype=np.float64)
    )
    sample_count = int(counts.size)
    total_board_samples = sample_count * BOARD_TILES
    return {
        "sample_count": sample_count,
        "valid_memory_coverage": {
            "mean_fraction": _mean(coverage),
            "median_fraction": _median(coverage),
            "std_fraction": 0.0 if not coverage.size else float(coverage.std()),
            "p10_fraction": float(percentiles[0]),
            "p25_fraction": float(percentiles[1]),
            "p75_fraction": float(percentiles[2]),
            "p90_fraction": float(percentiles[3]),
            "mean_valid_tile_count_out_of_40": _mean(counts),
            "median_valid_tile_count_out_of_40": _median(counts),
        },
        "freshness": {
            "normalized_mean": normalized_mean,
            "normalized_variance": normalized_variance,
            "expired_or_unknown_fraction": float(
                1.0 - valid_tiles / max(tile_samples, 1)
            ),
        },
        "dynamic_memory_coverage": {
            "definition": "tile validity governed by channel 20; static channels 10-14 excluded",
            "mean_valid_dynamic_tile_fraction": _mean(coverage),
            "median_valid_dynamic_tile_fraction": _median(coverage),
            "p10_valid_dynamic_tile_fraction": float(percentiles[0]),
            "p90_valid_dynamic_tile_fraction": float(percentiles[3]),
        },
        "preview_memory_coverage": {
            "applicable": horizon != 0,
            "applicable_preview_channels": [f"H{index}" for index in applicable],
            "nonzero_preview_board_tile_fraction": float(
                preview_nonzero / max(total_board_samples, 1)
            ),
            "valid_memory_tiles_carrying_nonzero_preview_fraction": float(
                valid_preview_nonzero / max(valid_tiles, 1)
            ),
            "per_preview_channel_nonzero_board_tile_fraction": {
                f"H{index + 1}": float(
                    preview_channels[index] / max(total_board_samples, 1)
                )
                for index in range(4)
            },
            "zero_values_not_treated_as_unobserved": True,
        },
        "gazenet_learnability": {
            "fraction_valid_coverage_ge_0_90": _fraction(coverage >= 0.90),
            "fraction_valid_coverage_ge_0_80": _fraction(coverage >= 0.80),
            "fraction_valid_coverage_ge_0_70": _fraction(coverage >= 0.70),
            "fraction_valid_coverage_le_0_50": _fraction(coverage <= 0.50),
            "fraction_valid_coverage_le_0_30": _fraction(coverage <= 0.30),
        },
    }


def _merge_runtime(rows: list[dict], decisions: int) -> dict:
    adapters = [row["adapter"] for row in rows]
    sums = {
        key: sum(int(adapter[key]) for adapter in adapters)
        for key in (
            "gaze_requests",
            "task_decisions",
            "acquisitions",
            "same_target_acquisitions",
            "new_target_acquisitions",
            "prep_count",
            "exec_count",
            "dwell_count",
            "actionnet_queries",
            "action_proposals",
            "proposal_revisions",
            "executed_proposals",
            "forced_stays",
            "recurrent_commits",
        )
    }
    distance_sum = sum(
        float(adapter["mean_saccade_distance_degrees"])
        * int(adapter["new_target_acquisitions"])
        for adapter in adapters
    )
    return {
        **sums,
        "gaze_requests_per_task_decision": sums["gaze_requests"] / max(decisions, 1),
        "acquisitions_per_task_decision": sums["acquisitions"] / max(decisions, 1),
        "same_target_acquisition_rate": sums["same_target_acquisitions"]
        / max(sums["acquisitions"], 1),
        "new_target_acquisition_rate": sums["new_target_acquisitions"]
        / max(sums["acquisitions"], 1),
        "mean_saccade_distance_degrees": distance_sum
        / max(sums["new_target_acquisitions"], 1),
        "forced_stay_rate": sums["forced_stays"] / max(decisions, 1),
        "actor_active_rate": sums["executed_proposals"] / max(decisions, 1),
    }


def _summarize_group(
    recorder_rows: list[dict],
    runtime_rows: list[dict],
    *,
    horizon: int | None,
    decisions: int,
    episodes: int,
    sparse_reward: float,
    deliveries: int,
) -> dict:
    phases = {
        phase: _describe_phase(
            [row["phases"][phase] for row in recorder_rows], horizon
        )
        for phase in PHASES
    }
    acquisitions = [row["acquisition"] for row in recorder_rows]
    actual = np.asarray(
        [value for row in acquisitions for value in row["actual_tile_counts"]],
        dtype=np.float64,
    )
    new = np.asarray(
        [value for row in acquisitions for value in row["new_valid_counts"]],
        dtype=np.float64,
    )
    refreshed = np.asarray(
        [value for row in acquisitions for value in row["refreshed_counts"]],
        dtype=np.float64,
    )
    recovered = np.asarray(
        [
            value
            for row in acquisitions
            for value in row["recovered_expired_or_unknown_counts"]
        ],
        dtype=np.float64,
    )
    reacquired_after_expiry = np.asarray(
        [
            value
            for row in acquisitions
            for value in row["reacquired_after_expiry_counts"]
        ],
        dtype=np.float64,
    )
    intervals = [
        value for row in acquisitions for value in row["same_tile_intervals_ms"]
    ]
    expiry_rows = [row["expiry"] for row in recorder_rows]
    per_decision_expiry = np.asarray(
        [value for row in expiry_rows for value in row["per_decision_counts"]],
        dtype=np.int64,
    )
    expiry_count = sum(int(row["count"]) for row in expiry_rows)
    runtime = _merge_runtime(runtime_rows, decisions)
    nominal_values = {
        int(row["nominal_tiles_per_acquisition"]) for row in recorder_rows
    }
    nominal = next(iter(nominal_values)) if nominal_values else 0
    new_hist = {
        str(value): int(np.count_nonzero(new == value))
        for value in range(nominal + 1)
    }
    marginal = new / BOARD_TILES
    result = {
        "environment_count": len(recorder_rows),
        "episode_count": int(episodes),
        "task_decisions": int(decisions),
        "sample_event_contract": {
            "pre_gaze_samples_equal_gaze_requests": (
                phases["pre_gaze_selection"]["sample_count"]
                == runtime["gaze_requests"]
            ),
            "post_acquisition_samples_equal_completed_acquisitions": (
                phases["post_acquisition"]["sample_count"]
                == runtime["acquisitions"]
            ),
            "task_deadline_samples_equal_task_decisions": (
                phases["task_deadline"]["sample_count"] == decisions
            ),
            "pre_gaze_and_post_acquisition_counts_required_equal": False,
        },
        "memory_samples": phases,
        "gaze_and_acquisition": {
            **runtime,
            "nominal_max_tiles_per_acquisition": nominal,
            "mean_actual_boundary_clipped_tiles_per_acquisition": _mean(actual),
            "median_actual_boundary_clipped_tiles_per_acquisition": _median(actual),
            "mean_new_valid_tiles_per_acquisition": _mean(new),
            "median_new_valid_tiles_per_acquisition": _median(new),
            "mean_refreshed_tiles_per_acquisition": _mean(refreshed),
            "mean_recovered_expired_or_unknown_tiles_per_acquisition": _mean(
                recovered
            ),
            "mean_reacquired_after_known_expiry_tiles_per_acquisition": _mean(
                reacquired_after_expiry
            ),
            "new_information_fraction_of_acquired_tiles": float(
                new.sum() / max(actual.sum(), 1.0)
            ),
            "refresh_revisit_fraction_of_acquired_tiles": float(
                refreshed.sum() / max(actual.sum(), 1.0)
            ),
        },
        "memory_expiry_and_turnover": {
            "tile_validity_expiration_count": int(expiry_count),
            "expirations_per_task_decision": expiry_count / max(decisions, 1),
            "expiration_rate_per_tile_decision": expiry_count
            / max(decisions * BOARD_TILES, 1),
            "unique_environment_tiles_expired_at_least_once": sum(
                int(row["unique_expired_tiles"]) for row in expiry_rows
            ),
            "unique_environment_tiles_reacquired_after_expiry": sum(
                int(row["unique_reacquired_after_expiry_tiles"])
                for row in expiry_rows
            ),
            "mean_time_between_acquisitions_same_tile_ms": _mean(intervals),
            "same_tile_interval_sample_count": len(intervals),
            "fraction_decisions_with_zero_memory_expiry": _fraction(
                per_decision_expiry == 0
            ),
            "fraction_decisions_with_at_least_one_expired_tile": _fraction(
                per_decision_expiry > 0
            ),
        },
        "gazenet_learnability": {
            "fraction_acquisitions_yielding_0_new_valid_tiles": _fraction(new == 0),
            "fraction_acquisitions_yielding_1plus_new_valid_tiles": _fraction(new >= 1),
            "fraction_acquisitions_yielding_2plus_new_valid_tiles": _fraction(new >= 2),
            "fraction_acquisitions_yielding_4plus_new_valid_tiles": _fraction(new >= 4),
            "mean_marginal_valid_memory_fraction_increase_per_acquisition": _mean(
                marginal
            ),
            "marginal_information_gain_valid_fraction": {
                "mean": _mean(marginal),
                "median": _median(marginal),
                "std": 0.0 if not marginal.size else float(marginal.std()),
                "p10": 0.0 if not marginal.size else float(np.percentile(marginal, 10)),
                "p90": 0.0 if not marginal.size else float(np.percentile(marginal, 90)),
                "new_valid_tile_count_histogram": new_hist,
            },
        },
        "task_viability": {
            "deliveries": int(deliveries),
            "sparse_team_reward": float(sparse_reward),
            "deliveries_per_episode": deliveries / max(episodes, 1),
            "forced_stay_count": runtime["forced_stays"],
            "forced_stay_rate": runtime["forced_stay_rate"],
            "actor_active_count": runtime["executed_proposals"],
            "actor_active_rate": runtime["actor_active_rate"],
            "actionnet_query_count": runtime["actionnet_queries"],
            "action_proposals": runtime["action_proposals"],
            "revisions": runtime["proposal_revisions"],
            "recurrent_commits": runtime["recurrent_commits"],
            "environment_task_decisions": runtime["task_decisions"],
            "acquisitions_equal_actionnet_queries": (
                runtime["acquisitions"] == runtime["actionnet_queries"]
            ),
        },
        "instrumentation": {
            "actor_mask_checks": sum(
                int(row["actor_mask_checks"]) for row in recorder_rows
            ),
            "actor_mask_failures": sum(
                int(row["actor_mask_failures"]) for row in recorder_rows
            ),
            "hidden_state_reads": sum(
                int(row["hidden_state_reads"]) for row in recorder_rows
            ),
        },
    }
    return result


def _sample_learner_actions(session, actor_hidden, recurrent_masks):
    counts = session._views["acquisition_counts"].copy()
    rows = [
        (env_id, acquisition)
        for env_id in range(session.n_rollout_threads)
        for acquisition in range(int(counts[env_id]))
    ]
    next_actor = actor_hidden.copy()
    learner_actions = np.full(
        session.n_rollout_threads, 4, dtype=np.int64
    )
    if rows:
        observations = np.stack([
            session._views["acquisition_observations"][env_id, acquisition]
            for env_id, acquisition in rows
        ])
        hidden = np.stack([actor_hidden[env_id] for env_id, _ in rows])
        masks = np.stack([recurrent_masks[env_id] for env_id, _ in rows])
        available = np.ones((len(rows), 6), dtype=np.float32)
        real_rows = len(rows)
        if session.device.type == "cuda":
            padding = (-real_rows) % session.n_rollout_threads
            if padding:
                observations = np.concatenate((
                    observations,
                    np.repeat(observations[-1:], padding, axis=0),
                ))
                hidden = np.concatenate((
                    hidden,
                    np.repeat(hidden[-1:], padding, axis=0),
                ))
                masks = np.concatenate((
                    masks,
                    np.repeat(masks[-1:], padding, axis=0),
                ))
                available = np.concatenate((
                    available,
                    np.repeat(available[-1:], padding, axis=0),
                ))
        with torch.no_grad():
            probabilities, candidates = session.policy.actor.get_probs(
                observations, hidden, masks, available
            )
        sampled_actions = []
        for row_index in range(real_rows):
            distribution = Categorical(
                probs=probabilities[row_index : row_index + 1]
            )
            sampled_actions.append(int(distribution.sample().item()))
        candidate_array = candidates[:real_rows].detach().cpu().numpy()
        for row_index, (env_id, acquisition) in enumerate(rows):
            if acquisition == int(counts[env_id]) - 1:
                learner_actions[env_id] = sampled_actions[row_index]
                next_actor[env_id] = candidate_array[row_index]
    return counts, learner_actions, next_actor


def _initial_state_record(session) -> dict:
    backends, adapters = session._snapshot_workers()
    environment_states = []
    partner_states = []
    gaze_rng_states = []
    for backend, adapter in zip(backends, adapters):
        backend = copy.deepcopy(backend)
        partner_states.append(backend.pop("partners"))
        environment_states.append(backend)
        gaze_rng_states.append(adapter["gaze_policy"]["rng_state"])
    return {
        "environment_state_hashes": [stable_hash(row) for row in environment_states],
        "environment_state_component_hashes": [
            {key: stable_hash(value) for key, value in sorted(row.items())}
            for row in environment_states
        ],
        "partner_state_hashes": [stable_hash(row) for row in partner_states],
        "rule_gaze_rng_state_hashes": [stable_hash(row) for row in gaze_rng_states],
        "rule_gaze_seeds": [
            int(adapter["gaze_policy"]["seed"]) for adapter in adapters
        ],
        "conditions": [condition.__dict__ for condition in session.conditions],
        "torch_rng_state_sha256": stable_hash(torch.get_rng_state()),
    }


def _audit_one_shape(args, shape: str, provenance: dict) -> tuple[dict, dict]:
    options = dict(
        gaze_type="rule_based",
        train_action=True,
        train_gaze=False,
        decision_times=[600],
        horizons=[0, 2, 4],
        fixation_shapes=[shape],
        seed=args.seed,
        rollout_length=1,
        episode_length=args.decisions,
        n_rollout_threads=args.environments,
        device=args.device,
        source_checkpoint=args.source_checkpoint,
        resume_checkpoint=None,
        fixed_partner_checkpoint=args.fixed_partner_checkpoint,
        configured_total_steps=args.environments * args.decisions,
        checkpoint_interval=args.environments * args.decisions,
        reward_shaping_horizon_steps=20_000_000,
        rule_gaze_probabilities=RULE_GAZE_PROBABILITIES,
        partner=None,
        ppo_epochs=1,
        num_mini_batches=1,
    )
    started = time.perf_counter()
    with ServerParallelActionTrainingSession(
        rollout_workers=args.rollout_workers,
        server_trace=False,
        memory_audit_enabled=True,
        **options,
    ) as session:
        startup = session.startup_report()
        initial = _initial_state_record(session)
        horizons = [int(condition.horizon) for condition in session.conditions]
        actor_before = parameter_hash(session.policy.actor)
        critic_before = parameter_hash(session.policy.critic)
        partner_before = session._partner_repository.assert_frozen(
            FIXED_HSP_PARTNER_ID, full_hash=True
        )
        actor_optimizer_before = stable_hash(
            session.policy.actor_optimizer.state_dict()
        )
        critic_optimizer_before = stable_hash(
            session.policy.critic_optimizer.state_dict()
        )
        buffer_before = stable_hash(session._buffer_state(session.buffer))
        scheduler_before = stable_hash(session.scheduler.state_dict())
        actor_hidden = session.buffer.rnn_states[0].copy()
        recurrent_masks = session.buffer.masks[0].copy()
        horizon_sparse = {value: 0.0 for value in (0, 2, 4)}
        horizon_deliveries = {value: 0 for value in (0, 2, 4)}
        horizon_decisions = {value: 0 for value in (0, 2, 4)}
        terminal_timing_exact = True
        available_actions_all_six = True
        session.policy.actor.eval()
        session.policy.critic.eval()
        session._phase("verify")
        for decision_index in range(args.decisions):
            session._shared.clear_decision_metadata()
            session._phase("collect", 1.0)
            available_actions_all_six = available_actions_all_six and bool(
                np.all(session._views["available_actions"] == 1.0)
            )
            counts, learner_actions, next_actor = _sample_learner_actions(
                session, actor_hidden, recurrent_masks
            )
            if np.any(counts > MAX_ACQUISITIONS_PER_DECISION):
                raise RuntimeError("canonical acquisition bound exceeded")
            session._views["learner_actions"][:] = learner_actions
            partner_actions = np.empty(args.environments, dtype=np.int64)
            for env_id, pool in enumerate(session._partner_pools):
                pool.assert_frozen()
                partner_actions[env_id] = pool.action(
                    session._views["partner_observations"][env_id],
                    session._views["partner_available_actions"][env_id],
                )
            session._views["partner_actions"][:] = partner_actions
            session._phase("commit")
            rewards = session._views["rewards"].copy()
            dones = session._views["dones"].astype(bool).copy()
            expected_done = decision_index == args.decisions - 1
            terminal_timing_exact = terminal_timing_exact and bool(
                np.all(dones == expected_done)
            )
            for env_id, horizon in enumerate(horizons):
                horizon_sparse[horizon] += float(rewards[env_id, 1])
                horizon_deliveries[horizon] += int(
                    session._views["deliveries"][env_id]
                )
                horizon_decisions[horizon] += 1
            session._views["reset_flags"].fill(0)
            session._phase("finalize")
            actor_hidden = next_actor
            if args.progress_every and (
                (decision_index + 1) % args.progress_every == 0
                or decision_index + 1 == args.decisions
            ):
                print(
                    f"{shape}: {decision_index + 1}/{args.decisions} decisions/env",
                    flush=True,
                )
        session._phase("verify")
        runtime = session._runtime_reports()
        actor_after = parameter_hash(session.policy.actor)
        critic_after = parameter_hash(session.policy.critic)
        partner_after = session._partner_repository.assert_frozen(
            FIXED_HSP_PARTNER_ID, full_hash=True
        )
        actor_optimizer_after = stable_hash(
            session.policy.actor_optimizer.state_dict()
        )
        critic_optimizer_after = stable_hash(
            session.policy.critic_optimizer.state_dict()
        )
        buffer_after = stable_hash(session._buffer_state(session.buffer))
        scheduler_after = stable_hash(session.scheduler.state_dict())

        recorder_rows = [row["memory_audit"] for row in runtime]
        if any(row is None for row in recorder_rows):
            raise RuntimeError("memory audit instrumentation missing from worker")
        by_horizon = {}
        for horizon in (0, 2, 4):
            selected_recorders = [
                row for row in recorder_rows if int(row["horizon"]) == horizon
            ]
            selected_runtime = [
                row for row in runtime
                if horizons[int(row["env_id"])] == horizon
            ]
            by_horizon[f"H{horizon}"] = _summarize_group(
                selected_recorders,
                selected_runtime,
                horizon=horizon,
                decisions=horizon_decisions[horizon],
                episodes=len(selected_recorders),
                sparse_reward=horizon_sparse[horizon],
                deliveries=horizon_deliveries[horizon],
            )
        overall = _summarize_group(
            recorder_rows,
            runtime,
            horizon=None,
            decisions=args.environments * args.decisions,
            episodes=args.environments,
            sparse_reward=sum(horizon_sparse.values()),
            deliveries=sum(horizon_deliveries.values()),
        )
        h0_preview_zero = all(
            by_horizon["H0"]["memory_samples"][phase][
                "preview_memory_coverage"
            ]["nonzero_preview_board_tile_fraction"] == 0.0
            for phase in PHASES
        )
        condition_counts = {
            f"H{horizon}": int(horizon_decisions[horizon])
            for horizon in (0, 2, 4)
        }
        guard_failures = sum(
            int(row["partner_step_guard_failures"])
            + int(row["partner_full_hash_failures"])
            for row in runtime
        )
        environment_steps = sum(int(row["environment_steps"]) for row in runtime)
        acquisitions = overall["gaze_and_acquisition"]["acquisitions"]
        gaze_requests = overall["gaze_and_acquisition"]["gaze_requests"]
        queries = overall["task_viability"]["actionnet_query_count"]
        recurring = overall["task_viability"]["recurrent_commits"]
        pre_gaze_samples = overall["memory_samples"]["pre_gaze_selection"][
            "sample_count"
        ]
        post_acquisition_samples = overall["memory_samples"][
            "post_acquisition"
        ]["sample_count"]
        deadline_samples = overall["memory_samples"]["task_deadline"][
            "sample_count"
        ]
        mask_failures = overall["instrumentation"]["actor_mask_failures"]
        hidden_reads = overall["instrumentation"]["hidden_state_reads"]
        checks = {
            "selected_source_validated": (
                session.source_provenance["source_content_sha256"]
                == provenance["source_content_sha256"]
            ),
            "source_actor_migration_bit_exact": (
                session.initial_actor_hash
                == provenance["source_actor_parameter_sha256"]
                == actor_before
            ),
            "actor_hash_unchanged": actor_before == actor_after,
            "critic_hash_unchanged": critic_before == critic_after,
            "actor_optimizer_state_unchanged": (
                actor_optimizer_before == actor_optimizer_after
            ),
            "critic_optimizer_state_unchanged": (
                critic_optimizer_before == critic_optimizer_after
            ),
            "ppo_buffer_unchanged": buffer_before == buffer_after,
            "scheduler_state_unchanged": scheduler_before == scheduler_after,
            "optimizer_steps": 0,
            "ppo_updates": 0,
            "fixed_partner_parameter_hash_unchanged": partner_before == partner_after,
            "fixed_partner_guard_failures": guard_failures,
            "fixed_partner_sampling_disabled": all(
                row["active_partner"] == FIXED_HSP_PARTNER_ID for row in runtime
            ),
            "horizon_exposure_exactly_balanced": len(set(condition_counts.values())) == 1,
            "actor_available_actions_six_ones": (
                available_actions_all_six and mask_failures == 0
            ),
            "rule_gaze_policy_memory_only": RULE_BASED_GAZE_POLICY_ID,
            "hidden_ground_truth_reads_added": hidden_reads,
            "environment_step_exactly_once_per_task_decision": (
                environment_steps == args.environments * args.decisions
            ),
            "terminal_episode_timing_exact": terminal_timing_exact,
            "pre_gaze_samples_equal_gaze_requests": (
                pre_gaze_samples == gaze_requests
            ),
            "post_acquisition_samples_equal_completed_acquisitions": (
                post_acquisition_samples == acquisitions
            ),
            "task_deadline_samples_equal_task_decisions": (
                deadline_samples == args.environments * args.decisions
            ),
            "sample_event_contract_exact_by_horizon": all(
                all(
                    value is False
                    if key == "pre_gaze_and_post_acquisition_counts_required_equal"
                    else bool(value)
                    for key, value in summary["sample_event_contract"].items()
                )
                for summary in by_horizon.values()
            ),
            "acquisitions_equal_actionnet_queries": acquisitions == queries,
            "recurrent_commits_at_most_one_per_task_decision": (
                recurring <= args.environments * args.decisions
            ),
            "h0_preview_zero": h0_preview_zero,
            "long_training_started": False,
        }
        if not _all_true_contract_checks(checks):
            raise RuntimeError(f"footprint audit contract failed: {checks}")
        result = {
            "shape": shape,
            "wall_seconds": time.perf_counter() - started,
            "startup": startup,
            "initialization": initial,
            "parameter_and_state_hashes": {
                "actor_before": actor_before,
                "actor_after": actor_after,
                "critic_before": critic_before,
                "critic_after": critic_after,
                "fixed_partner_before": partner_before,
                "fixed_partner_after": partner_after,
                "actor_optimizer_before": actor_optimizer_before,
                "actor_optimizer_after": actor_optimizer_after,
                "critic_optimizer_before": critic_optimizer_before,
                "critic_optimizer_after": critic_optimizer_after,
                "ppo_buffer_before": buffer_before,
                "ppo_buffer_after": buffer_after,
                "scheduler_before": scheduler_before,
                "scheduler_after": scheduler_after,
            },
            "condition_decision_counts": condition_counts,
            "overall": overall,
            "by_horizon": by_horizon,
            "contract_checks": checks,
        }
    return result, initial


def _all_true_contract_checks(checks: dict[str, Any]) -> bool:
    return all(
        value is False if key == "long_training_started"
        else value == 0 if key in {
            "optimizer_steps",
            "ppo_updates",
            "fixed_partner_guard_failures",
            "hidden_ground_truth_reads_added",
        } else bool(value)
        for key, value in checks.items()
    )


def _comparison_rows(footprints: dict[str, dict]) -> list[dict[str, Any]]:
    rows = []
    for shape, result in footprints.items():
        overall = result["overall"]
        memory = overall["memory_samples"]
        pre = memory["pre_gaze_selection"]
        post = memory["post_acquisition"]
        deadline = memory["task_deadline"]
        gaze = overall["gaze_and_acquisition"]
        task = overall["task_viability"]
        rows.append({
            "shape": shape,
            "pre_gaze_valid_fraction": pre["valid_memory_coverage"]["mean_fraction"],
            "post_acq_valid_fraction": post["valid_memory_coverage"]["mean_fraction"],
            "deadline_valid_fraction": deadline["valid_memory_coverage"]["mean_fraction"],
            "mean_continuous_freshness": pre["freshness"]["normalized_mean"],
            "expired_fraction": pre["freshness"]["expired_or_unknown_fraction"],
            "new_tiles_per_acq": gaze["mean_new_valid_tiles_per_acquisition"],
            "refresh_tiles_per_acq": gaze["mean_refreshed_tiles_per_acquisition"],
            "acquisitions_per_decision": gaze["acquisitions_per_task_decision"],
            "preview_coverage": pre["preview_memory_coverage"]["nonzero_preview_board_tile_fraction"],
            "dynamic_coverage": pre["dynamic_memory_coverage"]["mean_valid_dynamic_tile_fraction"],
            "forced_stay_rate": task["forced_stay_rate"],
            "deliveries_per_episode": task["deliveries_per_episode"],
        })
    return rows


def _markdown(report: dict) -> str:
    rows = report["comparison_table"]
    columns = (
        "shape",
        "pre_gaze_valid_fraction",
        "post_acq_valid_fraction",
        "deadline_valid_fraction",
        "fresh_4_5_fraction",
        "expired_fraction",
        "new_tiles_per_acq",
        "refresh_tiles_per_acq",
        "acquisitions_per_decision",
        "preview_coverage",
        "dynamic_coverage",
        "forced_stay_rate",
        "deliveries_per_episode",
    )
    lines = [
        "# Footprint persistent-memory audit",
        "",
        (
            f"No-optimization audit: {report['scientific_config']['environments']} "
            f"environments × {report['scientific_config']['task_decisions_per_environment']} "
            "task decisions per footprint. No footprint is selected automatically."
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                str(row[column]) if column == "shape" else f"{row[column]:.6f}"
                for column in columns
            ) + " |"
        )
    lines.extend(["", "## Sampling and contracts", ""])
    for shape, result in report["footprints"].items():
        samples = result["overall"]["memory_samples"]
        checks = result["contract_checks"]
        lines.append(
            f"- {shape}: A/B/C samples = "
            f"{samples['pre_gaze_selection']['sample_count']}/"
            f"{samples['post_acquisition']['sample_count']}/"
            f"{samples['task_deadline']['sample_count']}; "
            f"guard failures = {checks['fixed_partner_guard_failures']}; "
            f"optimizer steps = {checks['optimizer_steps']}; PPO updates = "
            f"{checks['ppo_updates']}."
        )
    pre_values = {
        row["shape"]: row["pre_gaze_valid_fraction"] for row in rows
    }
    new_values = {row["shape"]: row["new_tiles_per_acq"] for row in rows}
    lines.extend([
        "",
        "## Descriptive conclusion",
        "",
        (
            "Pre-gaze valid-memory coverage ranged from "
            f"{min(pre_values.values()):.4f} to {max(pre_values.values()):.4f}; "
            "the per-shape values above show the observed persistence difference."
        ),
        (
            "Mean newly valid tiles per acquisition ranged from "
            f"{min(new_values.values()):.4f} to {max(new_values.values()):.4f}. "
            "The JSON report contains the requested saturation thresholds, expiry "
            "turnover, marginal-gain distribution, and H0/H2/H4 breakdowns."
        ),
        (
            "These are descriptive diagnostics of remaining information-acquisition "
            "room; no footprint is designated as the final choice."
        ),
        "",
        "long_training_started = false",
        "",
    ])
    return "\n".join(lines)


def run(args) -> dict:
    torch.set_num_threads(args.torch_intra_threads)
    torch.set_num_interop_threads(args.torch_inter_threads)
    source_sha256 = file_sha256(args.source_checkpoint)
    if Path(args.source_checkpoint).resolve() != SELECTED_SOURCE.resolve():
        raise RuntimeError("audit requires the selected seed20260821 step40M source")
    if source_sha256 != SELECTED_SOURCE_SHA256:
        raise RuntimeError("selected step40M source content SHA-256 mismatch")
    provenance = validate_partial_full_source_checkpoint(
        args.source_checkpoint,
        requested_horizons=[0, 2, 4],
    )
    footprints = {}
    initializations = {}
    for shape in args.shapes:
        print(f"starting bounded no-optimization audit: {shape}", flush=True)
        result, initial = _audit_one_shape(args, shape, provenance)
        footprints[shape] = result
        initializations[shape] = initial
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    first_shape = args.shapes[0]
    initialization_equal = {
        key: all(
            initializations[shape][key] == initializations[first_shape][key]
            for shape in args.shapes[1:]
        )
        for key in (
            "environment_state_hashes",
            "partner_state_hashes",
            "rule_gaze_rng_state_hashes",
            "rule_gaze_seeds",
            "torch_rng_state_sha256",
        )
    }
    condition_assignment_equal = all(
        [
            {key: value for key, value in row.items() if key != "fixation_shape"}
            for row in initializations[shape]["conditions"]
        ]
        == [
            {key: value for key, value in row.items() if key != "fixation_shape"}
            for row in initializations[first_shape]["conditions"]
        ]
        for shape in args.shapes[1:]
    )
    global_checks = {
        "same_environment_initial_states_across_footprints": initialization_equal[
            "environment_state_hashes"
        ],
        "same_partner_state_initialization_across_footprints": initialization_equal[
            "partner_state_hashes"
        ],
        "same_rule_gaze_rng_initialization_across_footprints": initialization_equal[
            "rule_gaze_rng_state_hashes"
        ],
        "same_rule_gaze_seed_construction_across_footprints": initialization_equal[
            "rule_gaze_seeds"
        ],
        "same_torch_rng_initialization_across_footprints": initialization_equal[
            "torch_rng_state_sha256"
        ],
        "same_non_footprint_condition_assignment_across_footprints": (
            condition_assignment_equal
        ),
        "all_footprint_contract_checks_passed": all(
            _all_true_contract_checks(result["contract_checks"])
            for result in footprints.values()
        ),
        "no_learning_verified": True,
        "long_training_started": False,
    }
    if not all(
        value is False if key == "long_training_started" else bool(value)
        for key, value in global_checks.items()
    ):
        raise RuntimeError(f"cross-footprint initialization contract failed: {global_checks}")
    report = {
        "scientific_config": {
            "observation_mode": "partial",
            "gaze_type": "rule_based",
            "rule_gaze_policy_id": RULE_BASED_GAZE_POLICY_ID,
            "rule_gaze_probabilities": RULE_GAZE_PROBABILITIES,
            "decision_time_ms": 600,
            "horizons": [0, 2, 4],
            "horizon_mode": "episode_balanced_fixed_per_episode",
            "footprints": list(args.shapes),
            "environments": args.environments,
            "task_decisions_per_environment": args.decisions,
            "task_decisions_per_footprint": args.environments * args.decisions,
            "episode_length_task_decisions": args.decisions,
            "memory_lifetime_ms": MEMORY_LIFETIME_MS,
            "freshness_contract_id": CONTINUOUS_FRESHNESS_CONTRACT_ID,
            "seed": args.seed,
            "rollout_workers": args.rollout_workers,
            "torch_intra_op_threads": args.torch_intra_threads,
            "torch_inter_op_threads": args.torch_inter_threads,
            "device": args.device,
            "no_optimization": True,
            "nominal_footprint_tiles": NOMINAL_TILES,
        },
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_sha256": source_sha256,
        "source_provenance": provenance,
        "fixed_partner_checkpoint": str(
            Path(args.fixed_partner_checkpoint).resolve()
        ),
        "no_learning_verified": True,
        "footprints": footprints,
        "comparison_table": _comparison_rows(footprints),
        "contract_checks": global_checks,
        "long_training_started": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report), flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=SELECTED_SOURCE)
    parser.add_argument(
        "--fixed-partner-checkpoint", type=Path, default=FIXED_PARTNER
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=tuple(NOMINAL_TILES),
        default=list(NOMINAL_TILES),
    )
    parser.add_argument("--environments", type=int, default=30)
    parser.add_argument("--decisions", type=int, default=400)
    parser.add_argument("--rollout-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-intra-threads", type=int, default=2)
    parser.add_argument("--torch-inter-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    reports = ROOT / "final_model/server_parallel/reports"
    parser.add_argument(
        "--output-json", type=Path, default=reports / "footprint_memory_audit.json"
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=reports / "footprint_memory_audit.md"
    )
    args = parser.parse_args(argv)
    if args.environments <= 0 or args.decisions <= 0:
        parser.error("environments and decisions must be positive")
    if args.rollout_workers <= 0 or args.rollout_workers > args.environments:
        parser.error("rollout-workers must be in [1, environments]")
    if len(set(args.shapes)) != len(args.shapes):
        parser.error("shapes must not contain duplicates")
    return args


if __name__ == "__main__":
    run(parse_args())
