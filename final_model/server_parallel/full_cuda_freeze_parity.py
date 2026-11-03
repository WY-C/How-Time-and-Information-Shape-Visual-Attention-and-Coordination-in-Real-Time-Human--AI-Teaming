"""Exact CUDA baseline/tuned parity matrix for the Full performance freeze."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from final_model.common.mappo import parameter_hash
from final_model.server_parallel.full_action_training import (
    ServerParallelFullActionTrainingSession,
    _array_hash,
    _object_hash,
)
from final_model.server_parallel.full_parity import (
    _install_ppo_capture,
    PPO_BUFFER_FIELDS,
)


ROOT = Path(__file__).resolve().parents[2]
HSP_INITIAL = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"
FIXED_PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "reports/full_cuda_performance_freeze_parity.json"
)
CONDITIONS = ("h0", "h2", "h4", "shared")


def _configure_validated_cuda() -> None:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


def _options(args, condition: str) -> dict[str, object]:
    shared = condition == "shared"
    steps = int(args.environments) * int(args.rollout_length)
    return {
        "observation_mode": "full",
        "horizon": None if shared else int(condition[1:]),
        "shared_horizons": [0, 2, 4] if shared else None,
        "decision_time": None,
        "fixation_shape": None,
        "seed": int(args.seed),
        "rollout_length": int(args.rollout_length),
        "episode_length": 1,
        "n_rollout_threads": int(args.environments),
        "device": "cuda",
        "source_checkpoint": None,
        "resume_checkpoint": None,
        "hsp_initial_checkpoint": HSP_INITIAL,
        "fixed_partner_checkpoint": FIXED_PARTNER,
        "configured_total_steps": steps,
        "checkpoint_interval": steps,
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


def _install_partner_post_trace(session) -> None:
    original = session._commit_decisions

    def traced_commit(actions):
        results = original(actions)
        row = session.last_rollout_trace[-1]
        row["partner_post_rnn_sha256"] = [
            _array_hash(backend.partners.hidden) for backend in session.backends
        ]
        row["partner_post_mask_sha256"] = [
            _array_hash(backend.partners.mask) for backend in session.backends
        ]
        row["partner_rng_after_sha256"] = [
            _array_hash(backend.partners.torch_rng.get_state())
            for backend in session.backends
        ]
        return results

    session._commit_decisions = traced_commit


def _candidate(args) -> dict[str, object]:
    _configure_validated_cuda()
    options = _options(args, args.condition)
    if args.candidate == "validated_cuda":
        session = ServerParallelFullActionTrainingSession(
            rollout_workers=int(args.workers), server_trace=True, **options
        )
        previous = session._executor
        if previous is not None:
            previous.shutdown(wait=True)
        session._executor = ThreadPoolExecutor(max_workers=4)
    else:
        session = ServerParallelFullActionTrainingSession(
            rollout_workers=int(args.workers),
            torch_intra_op_threads=2,
            torch_inter_op_threads=1,
            fixed_partner_executor_workers=4,
            server_trace=True,
            **options,
        )
    _install_partner_post_trace(session)
    _install_ppo_capture(session)
    initial_actor = parameter_hash(session.policy.actor)
    initial_critic = parameter_hash(session.policy.critic)
    try:
        startup = session.startup_report()
        result = session.run_update()
        return {
            "candidate": args.candidate,
            "condition": args.condition,
            "startup": startup,
            "initial_actor_sha256": initial_actor,
            "initial_critic_sha256": initial_critic,
            "fixed_partner_parameter_sha256": (
                session.backends[0].partners.assert_frozen(full_hash=True)
            ),
            "trace": copy.deepcopy(session.last_rollout_trace),
            "trace_sha256": _object_hash(session.last_rollout_trace),
            "ppo_payload_sha256": dict(session._parity_ppo_payload),
            "ppo_buffer_fields": list(PPO_BUFFER_FIELDS),
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
            "shared_horizon_decision_counts": result.get(
                "shared_horizon_decision_counts"
            ),
            "shared_horizon_episode_counts": result.get(
                "shared_horizon_episode_counts"
            ),
            "partner_batch_size": result["partner_batch_size"],
            "partner_guard_failures": result["partner_step_guard_failures"],
            "worker_partner_guard_failures": result[
                "worker_partner_guard_failures"
            ],
        }
    finally:
        session.close()


def _projection(trace, key):
    if key.startswith("prepared."):
        child = key.split(".", 1)[1]
        return [
            [environment[child] for environment in step["prepared"]["environments"]]
            for step in trace
        ]
    if key.startswith("actor."):
        child = key.split(".", 1)[1]
        return [step["actor"][child] for step in trace]
    return [step[key] for step in trace]


def _compare(reference, tuned):
    categories = {
        "partner_actions": ("partner_actions",),
        "partner_rnn_pre_post": (
            "partner_pre_rnn_sha256", "partner_post_rnn_sha256",
            "partner_post_mask_sha256",
        ),
        "partner_rng": (
            "partner_rng_before_sha256", "partner_rng_after_sha256",
        ),
        "partner_inputs": (
            "partner_observation_sha256", "partner_available_actions_sha256",
        ),
        "predictor_input_output": (
            "prepared.predictor_input_sha256",
            "prepared.predictor_output_sha256",
        ),
        "preview_channels": ("prepared.preview_channel_sha256",),
        "learner_observations_actions_rnn": (
            "prepared.actor_observation_sha256",
            "prepared.critic_observation_sha256",
            "prepared.available_actions_sha256",
            "actor.actor_pre_rnn_sha256", "actor.critic_pre_rnn_sha256",
            "actor.recurrent_mask_sha256", "actor.available_actions_sha256",
            "actor.raw_logits_sha256", "actor.probabilities_sha256",
            "actor.actions", "actor.action_log_probs",
            "actor.actor_candidate_rnn_sha256",
            "actor.critic_candidate_rnn_sha256", "learner_actions",
        ),
        "backend_states": ("backend_state_sha256",),
        "rewards": ("results",),
    }
    checks = {}
    for category, fields in categories.items():
        different = [
            field for field in fields
            if _projection(reference["trace"], field)
            != _projection(tuned["trace"], field)
        ]
        checks[category] = {"exact": not different, "different_fields": different}
    different_ppo = [
        field for field in PPO_BUFFER_FIELDS
        if reference["ppo_payload_sha256"].get(field)
        != tuned["ppo_payload_sha256"].get(field)
    ]
    checks["ppo_buffer"] = {
        "exact": not different_ppo,
        "different_fields": different_ppo,
    }
    exact_record_fields = (
        "initial_actor_sha256", "initial_critic_sha256",
        "fixed_partner_parameter_sha256", "trace_sha256",
        "actor_hash_before_optimizer", "actor_hash_after", "critic_hash_after",
        "shared_horizon_decision_counts", "shared_horizon_episode_counts",
    )
    different_records = [
        field for field in exact_record_fields if reference[field] != tuned[field]
    ]
    checks["complete_record"] = {
        "exact": not different_records,
        "different_fields": different_records,
    }
    return checks


def _strict_preupdate(record) -> bool:
    return (
        record["preupdate_parity_pass"] == 1.0
        and record["preupdate_max_abs_logprob_difference"] == 0.0
        and record["preupdate_ratio_min"] == 1.0
        and record["preupdate_ratio_max"] == 1.0
        and record["preupdate_approx_kl"] == 0.0
        and record["preupdate_clip_fraction"] == 0.0
    )


def _isolated(args, condition: str, candidate: str):
    temporary = args.output.parent / f".{condition}.{candidate}.cuda-freeze.json"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    command = [
        sys.executable, "-m",
        "final_model.server_parallel.full_cuda_freeze_parity",
        "--candidate", candidate,
        "--condition", condition,
        "--candidate-output", str(temporary),
        "--seed", str(args.seed),
        "--environments", str(args.environments),
        "--rollout-length", str(args.rollout_length),
        "--workers", str(args.workers),
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True, timeout=900)
        return json.loads(temporary.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--seed", type=int, default=20260821)
    value.add_argument("--environments", type=int, default=100)
    value.add_argument("--rollout-length", type=int, default=3)
    value.add_argument("--workers", type=int, default=8)
    value.add_argument("--candidate", choices=("validated_cuda", "tuned_cuda"))
    value.add_argument("--condition", choices=CONDITIONS)
    value.add_argument("--candidate-output", type=Path)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def run(args) -> dict[str, object]:
    matrix = {}
    all_passed = True
    for condition in CONDITIONS:
        reference = _isolated(args, condition, "validated_cuda")
        tuned = _isolated(args, condition, "tuned_cuda")
        checks = _compare(reference, tuned)
        preupdate_exact = _strict_preupdate(reference) and _strict_preupdate(tuned)
        shared_counts_exact = True
        if condition == "shared":
            expected = int(args.environments) * int(args.rollout_length) // 3
            expected_counts = {"h0": expected, "h2": expected, "h4": expected}
            shared_counts_exact = (
                reference["shared_horizon_decision_counts"] == expected_counts
                and tuned["shared_horizon_decision_counts"] == expected_counts
            )
        devices_exact = (
            reference["startup"]["fixed_partner_device"].startswith("cuda")
            and tuned["startup"]["fixed_partner_device"].startswith("cuda")
        )
        batch1_exact = (
            reference["partner_batch_size"] == 1
            and tuned["partner_batch_size"] == 1
        )
        condition_passed = (
            all(check["exact"] for check in checks.values())
            and preupdate_exact and shared_counts_exact
            and devices_exact and batch1_exact
        )
        all_passed = all_passed and condition_passed
        matrix[condition] = {
            "checks": checks,
            "preupdate_exact": preupdate_exact,
            "shared_counts_exact": shared_counts_exact,
            "partner_cuda_exact": devices_exact,
            "partner_batch1_exact": batch1_exact,
            "validated_trace_sha256": reference["trace_sha256"],
            "tuned_trace_sha256": tuned["trace_sha256"],
            "shared_horizon_decision_counts": tuned[
                "shared_horizon_decision_counts"
            ],
            "validated_preupdate": {
                key: reference[key] for key in (
                    "preupdate_max_abs_logprob_difference",
                    "preupdate_ratio_min", "preupdate_ratio_max",
                    "preupdate_approx_kl", "preupdate_clip_fraction",
                )
            },
            "tuned_preupdate": {
                key: tuned[key] for key in (
                    "preupdate_max_abs_logprob_difference",
                    "preupdate_ratio_min", "preupdate_ratio_max",
                    "preupdate_approx_kl", "preupdate_clip_fraction",
                )
            },
            "passed": condition_passed,
        }
    report = {
        "contract": "full_cuda_performance_freeze_exact_parity_v1",
        "settings": {
            "conditions": list(CONDITIONS),
            "environments": int(args.environments),
            "rollout_length": int(args.rollout_length),
            "rollout_workers": int(args.workers),
            "torch_intra_op_threads": 2,
            "torch_inter_op_threads": 1,
            "fixed_partner_executor_workers": 4,
        },
        "conditions": matrix,
        "all_scientific_fields_exact": all_passed,
        "verdict": "GO" if all_passed else "NO-GO",
        "long_training_started": False,
        "full_120m_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.candidate is not None:
        if args.condition is None or args.candidate_output is None:
            raise ValueError("candidate mode requires condition and output")
        record = _candidate(args)
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
