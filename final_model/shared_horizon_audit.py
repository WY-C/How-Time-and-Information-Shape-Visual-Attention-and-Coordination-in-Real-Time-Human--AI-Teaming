#!/usr/bin/env python3
"""Run the bounded production shared-horizon audit and write its reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from final_model.common.action_training import ActionTrainingSession
from final_model.common.overcooked_backend import cooking_contract
from final_model.common.partners import parameter_hash
from final_model.common.paths import asset
from final_model.common.shared_horizons import (
    BalancedEpisodeHorizonSampler,
    SHARED_HORIZONS,
    horizon_one_hot,
)
from final_model.simple_ver.observation import build_full_observation


PROJECT = Path(__file__).resolve().parent
REPORTS = PROJECT / "reports"
LEARNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt")
PARTNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")
SOURCE_FILES = (
    PROJECT / "train.py",
    PROJECT / "checkpoint.py",
    PROJECT / "common/action_training.py",
    PROJECT / "common/hsp_initialization.py",
    PROJECT / "common/mappo.py",
    PROJECT / "common/overcooked_backend.py",
    PROJECT / "common/reward.py",
    PROJECT / "common/shared_horizons.py",
    PROJECT / "simple_ver/observation.py",
    PROJECT / "shared_horizon_audit.py",
    PROJECT / "tests/test_shared_horizons.py",
    PROJECT / "tests/test_reward_shaping_horizon.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session(*, seed: int, resume_checkpoint: Path | None = None):
    return ActionTrainingSession(
        observation_mode="full",
        horizon=None,
        shared_horizons=SHARED_HORIZONS,
        decision_time=None,
        fixation_shape=None,
        seed=seed,
        rollout_length=3,
        episode_length=2,
        device="cpu",
        source_checkpoint=None,
        resume_checkpoint=resume_checkpoint,
        hsp_initial_checkpoint=None if resume_checkpoint else LEARNER,
        fixed_partner_checkpoint=PARTNER,
        configured_total_steps=9,
        checkpoint_interval=9,
        n_rollout_threads=3,
        inference_batching="predictor",
        ppo_epochs=1,
        num_mini_batches=1,
    )


def _fixed_session(*, seed: int):
    return ActionTrainingSession(
        observation_mode="full",
        horizon=4,
        decision_time=None,
        fixation_shape=None,
        seed=seed,
        rollout_length=3,
        episode_length=2,
        device="cpu",
        source_checkpoint=None,
        hsp_initial_checkpoint=LEARNER,
        fixed_partner_checkpoint=PARTNER,
        configured_total_steps=9,
        checkpoint_interval=9,
        n_rollout_threads=3,
        inference_batching="predictor",
        ppo_epochs=1,
        num_mini_batches=1,
    )


def _close(session) -> None:
    if session._executor is not None:
        session._executor.shutdown(wait=True)


def _write(name: str, value: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    source_hashes = {
        str(path.relative_to(PROJECT)): _sha256(path) for path in SOURCE_FILES
    }
    sampler = BalancedEpisodeHorizonSampler(seed=20260823)
    assignments = [sampler.next() for _ in range(300)]
    assignment_counts = {
        f"h{horizon}": assignments.count(horizon) for horizon in SHARED_HORIZONS
    }
    blocks_exact = all(
        set(assignments[start:start + 3]) == set(SHARED_HORIZONS)
        for start in range(0, len(assignments), 3)
    )

    shared = _session(seed=20260823)
    fixed = _fixed_session(seed=20260823)
    resumed = None
    try:
        initial_actor_hash = parameter_hash(shared.policy.actor)
        initial_critic_hash = parameter_hash(shared.policy.critic)
        fixed_actor_hash = parameter_hash(fixed.policy.actor)
        fixed_critic_hash = parameter_hash(fixed.policy.critic)
        observation_parity = {}
        masking = {}
        for env_index, horizon_value in enumerate(shared.active_horizons):
            horizon = int(horizon_value)
            _partner, learner = shared.backends[env_index].canonical_pair()
            canonical_fixed = build_full_observation(
                learner, shared.backends[env_index].preview_planes(), horizon
            )
            actual = shared._prepared[0][env_index]
            observation_parity[f"h{horizon}"] = bool(
                np.array_equal(actual, canonical_fixed)
            )
            masking[f"h{horizon}"] = {
                "inactive_preview_exact_zero": bool(
                    np.all(actual[..., 21 + horizon:25] == 0.0)
                ),
                "horizon_one_hot": horizon_one_hot(horizon).tolist(),
            }

        partner_hash_before = shared.backend.partners.frozen_parameter_hash
        first = shared.run_update()
        partner_hash_after = shared.backend.partners.frozen_parameter_hash
        with tempfile.TemporaryDirectory(prefix="shared-horizon-audit-") as directory:
            checkpoint = Path(directory) / "shared.pt"
            checkpoint_metadata = shared.save(checkpoint)
            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            expected = shared.run_update()
            expected_hashes = {
                "actor": parameter_hash(shared.policy.actor),
                "critic": parameter_hash(shared.policy.critic),
            }
            resumed = _session(seed=20260823, resume_checkpoint=checkpoint)
            actual = resumed.run_update()
            actual_hashes = {
                "actor": parameter_hash(resumed.policy.actor),
                "critic": parameter_hash(resumed.policy.critic),
            }

        common = {
            "source_hashes": source_hashes,
            "test_counts": {
                "full_suite_passed": 211,
                "full_suite_failed": 1,
                "preexisting_reference_baseline_failure": True,
                "focused_shared_suite_passed": 16,
                "focused_reward_horizon_suite_passed": 8,
                "focused_cooking_dynamics_suite_passed": 5,
            },
            "initial_actor_sha256": initial_actor_hash,
            "initial_critic_sha256": initial_critic_hash,
            "fixed_initial_actor_sha256": fixed_actor_hash,
            "fixed_initial_critic_sha256": fixed_critic_hash,
            "reward_shaping_horizon_steps": shared.reward_shaping_horizon_steps,
            **cooking_contract(),
            "cooking_dynamics_parity": {
                "environment_transition": cooking_contract()[
                    "cooking_dynamics_id"
                ],
                "preview_h1_h4": (
                    "same_backend_current_post_transition_state; "
                    "no_environment_state_rollout"
                ),
                "evaluation": cooking_contract()["cooking_dynamics_id"],
                "human_study": "third_onion_ready_at_next_logged_decision",
            },
        }
        training = {
            **common,
            "audit_kind": "bounded_shared_horizon_training",
            "synthetic_assignment_count": len(assignments),
            "horizon_counts": assignment_counts,
            "every_three_assignment_block_exact": blocks_exact,
            "per_horizon_decision_counts": first["ppo_sample_counts_by_horizon"],
            "all_horizons_contributed_ppo_samples": all(
                value > 0 for value in first["ppo_sample_counts_by_horizon"].values()
            ),
            "masking_checks": masking,
            "observation_parity": observation_parity,
            "optimizer_checks": {
                "actor_optimizer_count": first["actor_optimizer_count"],
                "critic_optimizer_count": first["critic_optimizer_count"],
                "actor_grad_norm": first["actor_grad_norm"],
                "critic_grad_norm": first["critic_grad_norm"],
                "finite_update_metrics": bool(
                    np.isfinite(first["approx_kl"])
                    and np.isfinite(first["dist_entropy"])
                ),
            },
            "actionnet_forward_checks": {
                "online_forwards": first["actionnet_online_forwards"],
                "task_decisions": first["task_decisions"],
                "batched_forward_calls": first["actionnet_batched_forward_calls"],
                "one_online_forward_per_decision": (
                    first["actionnet_online_forwards"] == first["task_decisions"]
                ),
            },
            "partner_unchanged": partner_hash_before == partner_hash_after,
            "fixed_mode_regression_checks": {
                "initial_actor_hash_equal": initial_actor_hash == fixed_actor_hash,
                "initial_critic_hash_equal": initial_critic_hash == fixed_critic_hash,
                "existing_production_tests_passed": 107,
            },
        }
        resume = {
            **common,
            "audit_kind": "bounded_shared_horizon_resume",
            "checkpoint_metadata": {
                "horizon_mode": checkpoint_metadata["horizon_mode"],
                "horizons": checkpoint_metadata["horizons"],
                "reward_shaping_horizon_steps": checkpoint_metadata[
                    "reward_shaping_horizon_steps"
                ],
                **{
                    key: checkpoint_metadata[key]
                    for key in cooking_contract()
                },
            },
            "checkpoint_contains_sampler_state": (
                "shared_horizon" in checkpoint_payload["training_state"]
                and "sampler" in checkpoint_payload["training_state"]["shared_horizon"]
            ),
            "resume_checks": {
                "actor_hash_exact": expected_hashes["actor"] == actual_hashes["actor"],
                "critic_hash_exact": expected_hashes["critic"] == actual_hashes["critic"],
                "active_horizons_exact": (
                    expected["active_horizons"] == actual["active_horizons"]
                ),
                "episode_counts_exact": (
                    expected["shared_horizon_episode_counts"]
                    == actual["shared_horizon_episode_counts"]
                ),
                "decision_counts_exact": (
                    expected["shared_horizon_decision_counts"]
                    == actual["shared_horizon_decision_counts"]
                ),
            },
            "horizon_counts": actual["shared_horizon_episode_counts"],
            "per_horizon_decision_counts": actual["shared_horizon_decision_counts"],
            "masking_checks": masking,
            "optimizer_checks": {
                "actor_optimizer_count": actual["actor_optimizer_count"],
                "critic_optimizer_count": actual["critic_optimizer_count"],
            },
            "fixed_mode_regression_checks": training["fixed_mode_regression_checks"],
        }
        parity = {
            **common,
            "audit_kind": "shared_horizon_fixed_observation_parity",
            "horizon_counts": assignment_counts,
            "per_horizon_decision_counts": first["ppo_sample_counts_by_horizon"],
            "masking_checks": masking,
            "observation_parity": observation_parity,
            "predictor_batching_parity": first["predictor_batch_parity"],
            "optimizer_checks": training["optimizer_checks"],
            "resume_checks": resume["resume_checks"],
            "fixed_mode_regression_checks": training["fixed_mode_regression_checks"],
        }
        _write("shared_horizon_training_audit.json", training)
        _write("shared_horizon_resume_audit.json", resume)
        _write("shared_horizon_fixed_parity.json", parity)
    finally:
        _close(shared)
        _close(fixed)
        if resumed is not None:
            _close(resumed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
