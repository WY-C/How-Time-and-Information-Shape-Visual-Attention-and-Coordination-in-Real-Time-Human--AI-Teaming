"""One-update learned-Gaze inference parity and performance benchmark."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import time

import torch

from final_model.common.paths import file_sha256
from final_model.common.resume import capture_global_rng_state
from final_model.partial_ver.canonical_training import PartialTrainingSession


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODE_ORDER = (
    "legacy",
    "gaze",
    "predictor",
    "gaze_predictor",
    "action",
    "gaze_predictor_action",
)
EXCLUDED = {
    "gaze": (
        "async gaze requests and later request inputs depend on earlier sampled "
        "targets; batching changes the legacy global Torch RNG call order"
    ),
    "gaze_predictor": (
        "predictor is eligible, but the gaze component changes the legacy global "
        "Torch RNG call order"
    ),
    "action": (
        "ActionNet sampling is interleaved with GazeNet sampling; deferred batch "
        "sampling changes the legacy global Torch RNG call order"
    ),
    "gaze_predictor_action": (
        "both gaze and ActionNet components violate the legacy global Torch RNG "
        "call order"
    ),
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-checkpoint", type=Path, required=True)
    value.add_argument("--fixed-partner-checkpoint", type=Path, required=True)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    value.add_argument("--seed", type=int, default=20260825)
    value.add_argument("--n-rollout-threads", type=int, default=100)
    value.add_argument("--rollout-length", type=int, default=400)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--configured-total-steps", type=int, default=20_000_000)
    value.add_argument("--ppo-epochs", type=int, default=15)
    value.add_argument("--num-mini-batches", type=int, default=1)
    value.add_argument(
        "--benchmark-json",
        type=Path,
        default=REPORTS / "gaze_inference_benchmark.json",
    )
    value.add_argument(
        "--benchmark-markdown",
        type=Path,
        default=REPORTS / "gaze_inference_benchmark.md",
    )
    value.add_argument(
        "--parity-json",
        type=Path,
        default=REPORTS / "gaze_inference_parity.json",
    )
    value.add_argument(
        "--parity-markdown",
        type=Path,
        default=REPORTS / "gaze_inference_parity.md",
    )
    value.add_argument(
        "--legacy-json",
        type=Path,
        default=REPORTS / "gaze_inference_benchmark_legacy.json",
    )
    value.add_argument(
        "--legacy-markdown",
        type=Path,
        default=REPORTS / "gaze_inference_benchmark_legacy.md",
    )
    return value


def _session(args, mode: str) -> PartialTrainingSession:
    return PartialTrainingSession(
        gaze_type="learned",
        train_action=False,
        train_gaze=True,
        decision_times=[200, 400, 600],
        horizons=[0, 2, 4],
        fixation_shapes=["plus"],
        seed=args.seed,
        rollout_length=args.rollout_length,
        episode_length=args.episode_length,
        n_rollout_threads=args.n_rollout_threads,
        device=args.device,
        source_checkpoint=args.source_checkpoint,
        fixed_partner_checkpoint=args.fixed_partner_checkpoint,
        configured_total_steps=args.configured_total_steps,
        checkpoint_interval=1_000_000,
        reward_shaping_horizon_steps=20_000_000,
        inference_batching=mode,
        inference_shadow_audit=(mode == "predictor"),
        parity_audit=True,
        entropy_coefs=(0.2, 0.05, 0.01),
        entropy_coef_horizons=(0, 10_000_000, 20_000_000),
        ppo_epochs=args.ppo_epochs,
        num_mini_batches=args.num_mini_batches,
    )


def _run_mode(args, mode: str):
    session = _session(args, mode)
    initialization = {
        "action_actor": session.initial_actor_hash,
        "gaze_actor": session._audit_digest(session.gaze_actor.state_dict()),
        "rng": session._audit_digest(capture_global_rng_state()),
    }
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(session.device)
        torch.cuda.synchronize(session.device)
    started = time.perf_counter()
    result = session.run_update()
    if args.device == "cuda":
        torch.cuda.synchronize(session.device)
    elapsed = time.perf_counter() - started
    decisions = args.rollout_length * args.n_rollout_threads
    row = {
        "mode": mode,
        "status": "reference" if mode == "legacy" else "candidate",
        "parity": "REFERENCE" if mode == "legacy" else None,
        "update_wall_seconds": elapsed,
        "rollout_seconds": result["rollout_collection_seconds"],
        "environment_steps_per_second": decisions / max(elapsed, 1e-12),
        "decisions_per_second": decisions
        / max(result["rollout_collection_seconds"], 1e-12),
        "gaze_inference_seconds": result["gaze_inference_seconds"],
        "gaze_forward_count": result["gaze_forward_count"],
        "actionnet_inference_seconds": result["actionnet_inference_seconds"],
        "actionnet_forward_count": result["actionnet_forward_count"],
        "predictor_inference_seconds": result["predictor_inference_seconds"],
        "predictor_forward_count": result["predictor_forward_count"],
        "predictor_environment_count": result["predictor_environment_count"],
        "partner_inference_seconds": result["partner_inference_seconds"],
        "environment_runtime_seconds": result["environment_runtime_seconds"],
        "observation_memory_seconds": result["observation_memory_seconds"],
        "ppo_rollout_bookkeeping_seconds": result[
            "ppo_rollout_bookkeeping_seconds"
        ],
        "preupdate_audit_seconds": result["preupdate_audit_seconds"],
        "ppo_update_seconds": result["ppo_update_seconds"],
        "gpu_peak_memory_bytes": (
            None
            if args.device != "cuda"
            else int(torch.cuda.max_memory_allocated(session.device))
        ),
        "gaze_preupdate_max_abs_logprob_difference": result[
            "gaze_preupdate_max_abs_logprob_difference"
        ],
        "gaze_preupdate_approx_kl": result["gaze_preupdate_approx_kl"],
        "predictor_batch_parity": result["predictor_batch_parity"],
    }
    scientific = {
        key: result[key]
        for key in (
            "optimization_reward",
            "sparse_team_reward",
            "native_shaped_reward",
            "weighted_shaped_reward",
            "deliveries",
            "episodes",
            "gaze_requests",
            "acquisitions",
            "same_target_acquisitions",
            "new_target_acquisitions",
            "actionnet_queries",
            "actor_hash_after",
            "gaze_hash_after",
        )
    }
    audit = session.last_parity_audit
    gaze_state = {
        key: value.detach().cpu().clone()
        for key, value in session.gaze_actor.state_dict().items()
    }
    del session
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return row, initialization, scientific, audit, gaze_state


def _compare(reference, candidate) -> dict[str, object]:
    ref_init, ref_science, ref_audit, ref_gaze_state = reference
    new_init, new_science, new_audit, new_gaze_state = candidate
    scientific_checks = {
        key: ref_science[key] == new_science[key]
        for key in ref_science
        if key != "gaze_hash_after"
    }
    parameter_differences = torch.cat([
        (ref_gaze_state[key] - new_gaze_state[key]).abs().reshape(-1)
        for key in ref_gaze_state
    ])
    postupdate_max_abs = float(parameter_differences.max())
    postupdate_mean_abs = float(parameter_differences.mean())
    checks = {
        "initialization_exact": ref_init == new_init,
        "scientific_metrics_exact": bool(all(scientific_checks.values())),
        "environment_exact": ref_audit["environment"] == new_audit["environment"],
        "gaze_timing_memory_exact": (
            ref_audit["gaze_memory_timing"]
            == new_audit["gaze_memory_timing"]
        ),
        "actionnet_rollout_exact": (
            ref_audit["frozen_actionnet_rollout"]
            == new_audit["frozen_actionnet_rollout"]
        ),
        "partner_transition_exact": (
            ref_audit["partner_and_transition"]
            == new_audit["partner_and_transition"]
        ),
        "ppo_action_buffer_exact": (
            ref_audit["ppo_action_buffer"] == new_audit["ppo_action_buffer"]
        ),
        "ppo_gaze_records_exact": (
            ref_audit["ppo_gaze_records"] == new_audit["ppo_gaze_records"]
        ),
        "ppo_gaze_advantages_exact": (
            ref_audit["ppo_gaze_advantages"]
            == new_audit["ppo_gaze_advantages"]
        ),
        "ppo_gaze_returns_exact": (
            ref_audit["ppo_gaze_returns"] == new_audit["ppo_gaze_returns"]
        ),
        "preupdate_rng_exact": (
            ref_audit["rng_preupdate"] == new_audit["rng_preupdate"]
        ),
        "actionnet_forward_count_exact": (
            ref_audit["actionnet_forward_count"]
            == new_audit["actionnet_forward_count"]
        ),
        "gaze_forward_count_exact": (
            ref_audit["gaze_forward_count"]
            == new_audit["gaze_forward_count"]
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "scientific_metric_checks": scientific_checks,
        "scientific_metric_differences": {
            key: {"legacy": ref_science[key], "candidate": new_science[key]}
            for key, equal in scientific_checks.items()
            if not equal
        },
        "postupdate_gaze_hash_exact": (
            ref_science["gaze_hash_after"] == new_science["gaze_hash_after"]
        ),
        "postupdate_gaze_parameter_max_abs_difference": postupdate_max_abs,
        "postupdate_gaze_parameter_mean_abs_difference": postupdate_mean_abs,
        "postupdate_note": (
            "CUDA Gaze PPO convolution backward is not bitwise reproducible across "
            "independent legacy runs; exact rollout, pre-update RNG, PPO inputs, and "
            "explicit parameter tolerances are the adoption gate"
        ),
    }


def _apply_postupdate_gate(
    comparison: dict[str, object],
    *,
    tolerance: float,
) -> None:
    within_tolerance = (
        comparison["postupdate_gaze_parameter_max_abs_difference"]
        <= float(tolerance)
    )
    comparison["postupdate_tolerance"] = float(tolerance)
    comparison["checks"]["postupdate_gaze_parameters_within_legacy_envelope"] = (
        within_tolerance
    )
    comparison["pass"] = bool(all(comparison["checks"].values()))


def _markdown(report) -> str:
    rows = report["modes"]
    lines = [
        "# Learned Gaze inference benchmark",
        "",
        f"Execution: `{report['execution']['device']}`, "
        f"{report['execution']['n_rollout_threads']} envs × "
        f"{report['execution']['rollout_length']} steps.",
        "",
        "| mode | rollout s | speedup | env steps/s | gaze s | ActionNet s | predictor s | partner s | parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "excluded":
            lines.append(
                f"| {row['mode']} | — | — | — | — | — | — | — | EXCLUDED |"
            )
            continue
        lines.append(
            "| {mode} | {rollout:.6f} | {speedup:.3f}x | {throughput:.3f} | "
            "{gaze:.6f} | {action:.6f} | {predictor:.6f} | {partner:.6f} | {parity} |".format(
                mode=row["mode"],
                rollout=row["rollout_seconds"],
                speedup=row["speedup_vs_legacy"],
                throughput=row["environment_steps_per_second"],
                gaze=row["gaze_inference_seconds"],
                action=row["actionnet_inference_seconds"],
                predictor=row["predictor_inference_seconds"],
                partner=row["partner_inference_seconds"],
                parity=row["parity"],
            )
        )
    lines.extend(["", "## Exclusions", ""])
    for mode, reason in EXCLUDED.items():
        lines.append(f"- `{mode}`: {reason}.")
    lines.extend([
        "",
        "Recommended production mode: `--inference-batching predictor` "
        "only when the recorded parity result is PASS.",
        "",
    ])
    return "\n".join(lines)


def _parity_markdown(report) -> str:
    lines = ["# Learned Gaze inference parity", ""]
    for mode in MODE_ORDER:
        record = report["modes"][mode]
        lines.append(f"## {mode}")
        lines.append("")
        if record.get("excluded"):
            lines.append(f"EXCLUDED: {record['reason']}")
        elif mode == "legacy":
            lines.append("Canonical reference.")
        else:
            lines.append("PASS" if record["pass"] else "FAIL")
            lines.append("")
            for name, passed in record["checks"].items():
                lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
        lines.append("")
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    if args.n_rollout_threads <= 0 or args.rollout_length <= 0:
        raise ValueError("rollout dimensions must be positive")
    execution = {
        "device": args.device,
        "cuda_device_name": (
            None if args.device != "cuda" else torch.cuda.get_device_name(0)
        ),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "seed": args.seed,
        "n_rollout_threads": args.n_rollout_threads,
        "rollout_length": args.rollout_length,
        "episode_length": args.episode_length,
        "environment_steps": args.n_rollout_threads * args.rollout_length,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
        "fixed_partner_checkpoint": str(args.fixed_partner_checkpoint.resolve()),
        "fixed_partner_checkpoint_sha256": file_sha256(
            args.fixed_partner_checkpoint
        ),
    }
    legacy_row, legacy_init, legacy_science, legacy_audit, legacy_gaze_state = _run_mode(
        args, "legacy"
    )
    (
        _legacy_repeat_row,
        legacy_repeat_init,
        legacy_repeat_science,
        legacy_repeat_audit,
        legacy_repeat_gaze_state,
    ) = _run_mode(args, "legacy")
    predictor_row, predictor_init, predictor_science, predictor_audit, predictor_gaze_state = _run_mode(
        args, "predictor"
    )
    legacy_repeat = _compare(
        (legacy_init, legacy_science, legacy_audit, legacy_gaze_state),
        (
            legacy_repeat_init,
            legacy_repeat_science,
            legacy_repeat_audit,
            legacy_repeat_gaze_state,
        ),
    )
    legacy_envelope = max(
        2e-7,
        2.0 * legacy_repeat["postupdate_gaze_parameter_max_abs_difference"],
    )
    _apply_postupdate_gate(legacy_repeat, tolerance=legacy_envelope)
    comparison = _compare(
        (legacy_init, legacy_science, legacy_audit, legacy_gaze_state),
        (predictor_init, predictor_science, predictor_audit, predictor_gaze_state),
    )
    _apply_postupdate_gate(comparison, tolerance=legacy_envelope)
    predictor_row["parity"] = "PASS" if comparison["pass"] else "FAIL"
    legacy_row["speedup_vs_legacy"] = 1.0
    predictor_row["speedup_vs_legacy"] = (
        legacy_row["update_wall_seconds"]
        / max(predictor_row["update_wall_seconds"], 1e-12)
    )
    rows = []
    for mode in MODE_ORDER:
        if mode == "legacy":
            rows.append(legacy_row)
        elif mode == "predictor":
            rows.append(predictor_row)
        else:
            rows.append({
                "mode": mode,
                "status": "excluded",
                "parity": "EXCLUDED",
                "reason": EXCLUDED[mode],
            })
    benchmark = {
        "contract": "learned_gaze_one_update_inference_benchmark_v1",
        "execution": execution,
        "modes": rows,
        "recommended_mode": "predictor" if comparison["pass"] else "legacy",
    }
    parity = {
        "contract": "learned_gaze_exact_trajectory_parity_v1",
        "execution": execution,
        "modes": {
            "legacy": {"reference": True},
            "legacy_repeat": legacy_repeat,
            "predictor": comparison,
            **{
                mode: {"excluded": True, "reason": reason}
                for mode, reason in EXCLUDED.items()
            },
        },
    }
    legacy_report = {
        "contract": "learned_gaze_legacy_reference_profile_v1",
        "execution": execution,
        "mode": legacy_row,
    }
    _write(args.benchmark_json, json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
    _write(args.benchmark_markdown, _markdown(benchmark))
    _write(args.parity_json, json.dumps(parity, indent=2, sort_keys=True) + "\n")
    _write(args.parity_markdown, _parity_markdown(parity))
    _write(args.legacy_json, json.dumps(legacy_report, indent=2, sort_keys=True) + "\n")
    _write(args.legacy_markdown, _markdown({
        "execution": execution,
        "modes": [legacy_row],
    }))
    print(json.dumps({
        "recommended_mode": benchmark["recommended_mode"],
        "predictor_parity": predictor_row["parity"],
        "speedup_vs_legacy": predictor_row["speedup_vs_legacy"],
        "benchmark_json": str(args.benchmark_json),
        "parity_json": str(args.parity_json),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
