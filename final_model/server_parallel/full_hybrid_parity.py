"""Deterministic CUDA parity audit for the experimental CPU-partner pipeline."""

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
from final_model.server_parallel.full_hybrid_action_training import (
    ServerParallelFullHybridActionTrainingSession,
)
from final_model.server_parallel.full_parity import (
    _install_ppo_capture,
    PPO_BUFFER_FIELDS,
)


ROOT = Path(__file__).resolve().parents[2]
HSP_INITIAL = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"
FIXED_PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "reports/full_hybrid_inference_parity.json"
)


def _configure_torch() -> None:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


def _options(args) -> dict[str, object]:
    steps = int(args.environments) * int(args.rollout_length)
    return {
        "observation_mode": "full",
        "horizon": None,
        "shared_horizons": [0, 2, 4],
        "decision_time": None,
        "fixation_shape": None,
        "seed": int(args.seed),
        "rollout_length": int(args.rollout_length),
        "episode_length": 400,
        "n_rollout_threads": int(args.environments),
        "device": "cuda",
        "source_checkpoint": None,
        "resume_checkpoint": None,
        "hsp_initial_checkpoint": HSP_INITIAL,
        "fixed_partner_checkpoint": FIXED_PARTNER,
        "configured_total_steps": steps,
        "checkpoint_interval": steps,
        "reward_shaping_horizon_steps": 120_000_000,
        "partner": None,
        "inference_batching": "predictor",
        "inference_shadow_audit": False,
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coefs": [0.05, 0.03, 0.01],
        "entropy_coef_horizons": [0, 60_000_000, 120_000_000],
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
    _configure_torch()
    options = _options(args)
    if args.candidate == "validated_cuda":
        session = ServerParallelFullActionTrainingSession(
            rollout_workers=int(args.workers), server_trace=True, **options
        )
        old = session._executor
        if old is not None:
            old.shutdown(wait=True)
        session._executor = ThreadPoolExecutor(max_workers=4)
    else:
        session = ServerParallelFullHybridActionTrainingSession(
            rollout_workers=int(args.workers),
            partner_executor_workers=4,
            runtime_executor_workers=4,
            server_trace=True,
            **options,
        )
    _install_partner_post_trace(session)
    _install_ppo_capture(session)
    initial_actor = parameter_hash(session.policy.actor)
    initial_critic = parameter_hash(session.policy.critic)
    try:
        result = session.run_update()
        return {
            "candidate": args.candidate,
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
            "preupdate_max_abs_logprob_difference": result[
                "preupdate_max_abs_logprob_difference"
            ],
            "preupdate_ratio_min": result["preupdate_ratio_min"],
            "preupdate_ratio_max": result["preupdate_ratio_max"],
            "preupdate_approx_kl": result["preupdate_approx_kl"],
            "partner_guard_failures": result["partner_step_guard_failures"],
            "worker_partner_guard_failures": result[
                "worker_partner_guard_failures"
            ],
        }
    finally:
        session.close()


def _first_list_difference(left, right):
    if len(left) != len(right):
        return {"kind": "length", "left": len(left), "right": len(right)}
    for index, (lvalue, rvalue) in enumerate(zip(left, right)):
        if lvalue != rvalue:
            return {"index": index, "left": lvalue, "right": rvalue}
    return None


def _trace_projection(trace, key):
    if key.startswith("prepared."):
        child = key.split(".", 1)[1]
        return [
            [env[child] for env in step["prepared"]["environments"]]
            for step in trace
        ]
    if key.startswith("actor."):
        child = key.split(".", 1)[1]
        return [step["actor"][child] for step in trace]
    return [step[key] for step in trace]


def _compare(left: dict[str, object], right: dict[str, object]):
    categories = {
        "partner_actions": ("partner_actions",),
        "partner_rnn_pre": ("partner_pre_rnn_sha256",),
        "partner_rnn_post": ("partner_post_rnn_sha256",),
        "partner_rng_before": ("partner_rng_before_sha256",),
        "partner_rng_after": ("partner_rng_after_sha256",),
        "predictor_inputs": ("prepared.predictor_input_sha256",),
        "predictor_outputs": ("prepared.predictor_output_sha256",),
        "preview_channels": ("prepared.preview_channel_sha256",),
        "learner_observations": (
            "prepared.actor_observation_sha256",
            "prepared.critic_observation_sha256",
            "prepared.available_actions_sha256",
        ),
        "learner_actions_rnn": (
            "actor.actor_pre_rnn_sha256",
            "actor.critic_pre_rnn_sha256",
            "actor.raw_logits_sha256",
            "actor.probabilities_sha256",
            "actor.actions",
            "actor.action_log_probs",
            "actor.actor_candidate_rnn_sha256",
            "actor.critic_candidate_rnn_sha256",
        ),
        "backend_states": ("backend_state_sha256",),
        "rewards": ("results",),
    }
    checks = {}
    first_divergence = None
    for category, keys in categories.items():
        exact = True
        detail = None
        for key in keys:
            difference = _first_list_difference(
                _trace_projection(left["trace"], key),
                _trace_projection(right["trace"], key),
            )
            if difference is not None:
                exact = False
                detail = {"field": key, **difference}
                break
        checks[category] = {"exact": exact, "first_difference": detail}
        if not exact and first_divergence is None:
            first_divergence = {"category": category, **detail}
    ppo_exact = left["ppo_payload_sha256"] == right["ppo_payload_sha256"]
    checks["ppo_buffer"] = {
        "exact": ppo_exact,
        "different_fields": [
            key for key in PPO_BUFFER_FIELDS
            if left["ppo_payload_sha256"].get(key)
            != right["ppo_payload_sha256"].get(key)
        ],
    }
    initialization_exact = all(
        left[key] == right[key]
        for key in (
            "initial_actor_sha256",
            "initial_critic_sha256",
            "fixed_partner_parameter_sha256",
            "actor_hash_before_optimizer",
        )
    )
    checks["initialization"] = {"exact": initialization_exact}
    all_exact = initialization_exact and all(
        bool(value["exact"]) for value in checks.values()
    )
    action_or_trajectory_changed = not all(
        checks[key]["exact"]
        for key in (
            "partner_actions", "predictor_inputs", "predictor_outputs",
            "preview_channels", "learner_observations", "learner_actions_rnn",
            "backend_states", "rewards", "ppo_buffer",
        )
    )
    return checks, first_divergence, all_exact, action_or_trajectory_changed


def _isolated(args, candidate: str) -> dict[str, object]:
    output = args.output.parent / f".{candidate}.hybrid-parity-candidate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "final_model.server_parallel.full_hybrid_parity",
        "--candidate",
        candidate,
        "--candidate-output",
        str(output),
        "--seed",
        str(args.seed),
        "--environments",
        str(args.environments),
        "--rollout-length",
        str(args.rollout_length),
        "--workers",
        str(args.workers),
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True, timeout=600)
        return json.loads(output.read_text(encoding="utf-8"))
    finally:
        output.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--seed", type=int, default=20260821)
    value.add_argument("--environments", type=int, default=12)
    value.add_argument("--rollout-length", type=int, default=3)
    value.add_argument("--workers", type=int, default=8)
    value.add_argument(
        "--candidate", choices=("validated_cuda", "hybrid_cpu_partner")
    )
    value.add_argument("--candidate-output", type=Path)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def run(args) -> dict[str, object]:
    validated = _isolated(args, "validated_cuda")
    hybrid = _isolated(args, "hybrid_cpu_partner")
    checks, first, all_exact, changed = _compare(validated, hybrid)
    report = {
        "contract": "experimental_full_hybrid_exact_parity_v1",
        "settings": {
            "shared_horizons": [0, 2, 4],
            "seed": int(args.seed),
            "environments": int(args.environments),
            "rollout_length": int(args.rollout_length),
            "rollout_workers": int(args.workers),
            "torch_intra": 2,
            "torch_inter": 1,
            "partner_executor": 4,
        },
        "checks": checks,
        "first_divergence": first,
        "all_exact": all_exact,
        "sampled_action_or_trajectory_changed": changed,
        "verdict": "GO" if all_exact else "NO-GO",
        "validated_trace_sha256": validated["trace_sha256"],
        "hybrid_trace_sha256": hybrid["trace_sha256"],
        "validated_ppo_payload_sha256": validated["ppo_payload_sha256"],
        "hybrid_ppo_payload_sha256": hybrid["ppo_payload_sha256"],
        "validated_preupdate": {
            key: validated[key] for key in (
                "preupdate_max_abs_logprob_difference", "preupdate_ratio_min",
                "preupdate_ratio_max", "preupdate_approx_kl",
            )
        },
        "hybrid_preupdate": {
            key: hybrid[key] for key in (
                "preupdate_max_abs_logprob_difference", "preupdate_ratio_min",
                "preupdate_ratio_max", "preupdate_approx_kl",
            )
        },
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
        if args.candidate_output is None:
            raise ValueError("candidate mode requires --candidate-output")
        record = _candidate(args)
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
