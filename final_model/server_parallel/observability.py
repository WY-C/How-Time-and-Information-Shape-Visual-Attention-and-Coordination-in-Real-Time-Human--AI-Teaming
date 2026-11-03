"""RNG-neutral update-level observability for server_parallel training."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

from final_model.common.tracking import WandbTracker
from final_model.partial_ver.observability import (
    flatten_partial_metric_namespaces,
)


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("observability accepts scalar tensors only")
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _value(latest: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in latest:
            return _scalar(latest[key])
    return None


def _rate(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _guard_failures(latest: Mapping[str, Any]) -> int | None:
    values = [
        _value(latest, key)
        for key in (
            "worker_partner_guard_failures",
            "partner_step_guard_failures",
            "partner_full_hash_failures",
        )
    ]
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


def build_update_record(
    latest: Mapping[str, Any],
    *,
    wall_time_seconds: float,
    update_wall_seconds: float,
    previous_environment_steps: int,
) -> dict[str, Any]:
    """Detach already-computed scalars into one JSON/W&B/progress record."""
    environment_steps = int(_value(latest, "environment_steps", "global_step"))
    step_delta = environment_steps - int(previous_environment_steps)
    measured_steps_per_second = (
        float(step_delta) / float(update_wall_seconds)
        if update_wall_seconds > 0.0
        else None
    )
    acquisitions = _value(latest, "acquisitions")
    same_target = _value(latest, "same_target_acquisitions")
    new_target = _value(latest, "new_target_acquisitions")
    phase_timings = {
        "worker_collect_seconds": _value(
            latest, "worker_collect_seconds", "worker_collect_barrier_seconds"
        ),
        "actionnet_rollout_seconds": _value(
            latest, "actionnet_rollout_seconds", "actor_inference_seconds"
        ),
        "partner_seconds": _value(latest, "partner_inference_seconds"),
        "worker_commit_seconds": _value(
            latest, "worker_commit_seconds", "worker_commit_barrier_seconds"
        ),
        "worker_finalize_seconds": _value(
            latest, "worker_finalize_seconds", "worker_finalize_barrier_seconds"
        ),
        "preupdate_audit_seconds": _value(latest, "preupdate_audit_seconds"),
        "ppo_seconds": _value(latest, "ppo_update_seconds"),
    }
    update_unattributed_seconds = None
    if all(value is not None for value in phase_timings.values()):
        update_unattributed_seconds = float(update_wall_seconds) - sum(
            float(value) for value in phase_timings.values()
        )
    record = {
        "environment_steps": environment_steps,
        "update_index": int(_value(latest, "update_count", "updates")),
        "wall_time_seconds": float(wall_time_seconds),
        "update_wall_seconds": float(update_wall_seconds),
        "steps_per_second": _value(latest, "steps_per_second")
        or measured_steps_per_second,
        "reward": _value(latest, "reward", "optimization_reward"),
        "raw_reward": _value(latest, "raw_reward"),
        "sparse_reward": _value(latest, "sparse_reward", "sparse_team_reward"),
        "actor_loss": _value(
            latest, "gaze_policy_loss", "actor_loss", "policy_loss"
        ),
        "critic_loss": _value(
            latest, "gaze_value_loss", "critic_loss", "value_loss"
        ),
        "entropy": _value(
            latest, "gaze_entropy", "entropy", "dist_entropy"
        ),
        "approx_kl": _value(latest, "gaze_approx_kl", "approx_kl"),
        "clip_fraction": _value(
            latest, "gaze_clip_fraction", "clip_fraction"
        ),
        "explained_variance": _value(latest, "explained_variance"),
        "action_explained_variance": _value(latest, "explained_variance"),
        "gaze_explained_variance": _value(
            latest, "gaze_explained_variance"
        ),
        "gaze_return_mean": _value(latest, "gaze_return_mean"),
        "gaze_return_std": _value(latest, "gaze_return_std"),
        "gaze_value_mean": _value(latest, "gaze_value_mean"),
        "gaze_value_std": _value(latest, "gaze_value_std"),
        "gaze_critic_grad_clip_fraction": _value(
            latest, "gaze_critic_grad_clip_fraction"
        ),
        "gaze_weighted_advantage_mean": _value(
            latest, "gaze_weighted_advantage_mean"
        ),
        "gaze_weighted_advantage_std": _value(
            latest, "gaze_weighted_advantage_std"
        ),
        "gaze_unweighted_advantage_mean": _value(
            latest, "gaze_unweighted_advantage_mean"
        ),
        "gaze_unweighted_advantage_std": _value(
            latest, "gaze_unweighted_advantage_std"
        ),
        "actor_grad_norm": _value(
            latest, "gaze_actor_grad_norm", "actor_grad_norm"
        ),
        "critic_grad_norm": _value(
            latest, "gaze_critic_grad_norm", "critic_grad_norm"
        ),
        "gaze_transition_count": _value(latest, "gaze_transition_count"),
        "gaze_policy_calls_total": _value(latest, "gaze_policy_calls_total"),
        "gaze_transition_count_total": _value(
            latest, "gaze_transition_count_total"
        ),
        "gaze_attributed_optimization_reward": _value(
            latest, "gaze_attributed_optimization_reward"
        ),
        "gaze_attributed_env_boundaries": _value(
            latest, "gaze_attributed_env_boundaries", "gaze_env_boundary_count"
        ),
        "gaze_reward_conservation_error": _value(
            latest, "gaze_reward_conservation_error"
        ),
        "bootstrap_unowned_reward_count": _value(
            latest, "bootstrap_unowned_reward_count"
        ),
        "bootstrap_unowned_reward_sum": _value(
            latest, "bootstrap_unowned_reward_sum"
        ),
        "bootstrap_unowned_reward_count_total": _value(
            latest, "bootstrap_unowned_reward_count_total"
        ),
        "bootstrap_unowned_reward_sum_total": _value(
            latest, "bootstrap_unowned_reward_sum_total"
        ),
        "gaze_records_per_task_decision_mean": _value(
            latest, "gaze_records_per_task_decision_mean"
        ),
        "gaze_records_per_task_decision_max": _value(
            latest, "gaze_records_per_task_decision_max"
        ),
        "training_contract": _value(latest, "training_contract"),
        "worker_count": _value(latest, "worker_count", "rollout_workers"),
        "gaze_worker_weight_sync_verified": _value(
            latest, "gaze_worker_weight_sync_verified"
        ),
        "actor_hash_before": _value(latest, "actor_hash_before"),
        "actor_hash_after": _value(latest, "actor_hash_after"),
        "critic_hash_before": _value(latest, "critic_hash_before"),
        "critic_hash_after": _value(latest, "critic_hash_after"),
        "action_actor_requires_grad": _value(
            latest, "action_actor_requires_grad"
        ),
        "action_critic_requires_grad": _value(
            latest, "action_critic_requires_grad"
        ),
        "action_actor_optimizer_present": _value(
            latest, "action_actor_optimizer_present"
        ),
        "action_critic_optimizer_present": _value(
            latest, "action_critic_optimizer_present"
        ),
        "action_optimizer_isolation_verified": _value(
            latest, "action_optimizer_isolation_verified"
        ),
        "gaze_preupdate_max_abs_logprob_difference": _value(
            latest, "gaze_preupdate_max_abs_logprob_difference"
        ),
        "gaze_preupdate_mean_abs_logprob_difference": _value(
            latest, "gaze_preupdate_mean_abs_logprob_difference"
        ),
        "gaze_preupdate_logprob_tolerance": _value(
            latest, "gaze_preupdate_logprob_tolerance"
        ),
        "gaze_logprob_audit_version": _value(
            latest, "gaze_logprob_audit_version"
        ),
        "gaze_logprob_execution_settings_matched": _value(
            latest, "gaze_logprob_execution_settings_matched"
        ),
        "gaze_logprob_main_cpu_intra_op_threads": _value(
            latest, "gaze_logprob_main_cpu_intra_op_threads"
        ),
        "gaze_logprob_main_cpu_inter_op_threads": _value(
            latest, "gaze_logprob_main_cpu_inter_op_threads"
        ),
        "gaze_logprob_worker_cpu_intra_op_threads": _value(
            latest, "gaze_logprob_worker_cpu_intra_op_threads"
        ),
        "gaze_logprob_worker_cpu_inter_op_threads": _value(
            latest, "gaze_logprob_worker_cpu_inter_op_threads"
        ),
        "gaze_logprob_inference_dtype": _value(
            latest, "gaze_logprob_inference_dtype"
        ),
        "gaze_logprob_inference_path": _value(
            latest, "gaze_logprob_inference_path"
        ),
        "deliveries": _value(latest, "deliveries"),
        "deliveries_per_episode": _value(latest, "deliveries_per_episode"),
        "collision_count": _value(latest, "collision_count"),
        "collision_rate": _value(latest, "collision_rate"),
        "decisions": _value(latest, "task_decisions", "decisions"),
        "actor_active_rate": _value(
            latest, "actor_active_rate", "actor_active_sample_rate"
        ),
        "forced_stay_rate": _value(latest, "forced_stay_rate"),
        "gaze_requests": _value(latest, "gaze_requests"),
        "acquisitions": acquisitions,
        "acquisitions_per_decision": _value(
            latest, "acquisitions_per_decision"
        ),
        "held_foveal_refreshes": _value(latest, "held_foveal_refreshes"),
        "held_foveal_refreshes_per_decision": _value(
            latest, "held_foveal_refreshes_per_decision"
        ),
        "same_target_acquisition_rate": _rate(same_target, acquisitions),
        "new_target_acquisition_rate": _rate(new_target, acquisitions),
        "valid_memory_fraction": _value(latest, "valid_memory_fraction"),
        "freshness_normalized_min": _value(latest, "freshness_normalized_min"),
        "freshness_normalized_max": _value(latest, "freshness_normalized_max"),
        "freshness_normalized_mean": _value(latest, "freshness_normalized_mean"),
        "freshness_normalized_variance": _value(
            latest, "freshness_normalized_variance"
        ),
        "mean_freshness": _value(latest, "mean_freshness"),
        "mean_saccade_distance": _value(
            latest, "mean_saccade_distance", "mean_saccade_distance_degrees"
        ),
        "actionnet_query_count": _value(
            latest, "actionnet_queries", "actionnet_online_forwards"
        ),
        "actionnet_rollout_forward_count": _value(
            latest, "actionnet_rollout_forward_count"
        ),
        "recurrent_commit_count": _value(latest, "recurrent_commits"),
        "fixed_partner_guard_failures": _guard_failures(latest),
        "leakage_guard_failures": _value(latest, "leakage_guard_failures"),
        "rollout_seconds": _value(latest, "rollout_collection_seconds"),
        "predictor_seconds": _value(latest, "predictor_inference_seconds"),
        "env_step_seconds": _value(latest, "env_step_seconds"),
        "ipc_wait_seconds": _value(
            latest, "ipc_barrier_overhead_seconds", "worker_wait_seconds"
        ),
        **phase_timings,
        "update_unattributed_seconds": update_unattributed_seconds,
    }
    shared = latest.get("shared_horizon_metrics") or {}
    for label in ("h0", "h2", "h4"):
        bucket = shared.get(label) or {}
        record[f"{label}_reward"] = _value(bucket, "optimization_reward", "reward")
        record[f"{label}_sparse_reward"] = _value(
            bucket, "sparse_team_reward", "sparse_reward"
        )
        record[f"{label}_deliveries"] = _value(bucket, "deliveries")
        record[f"{label}_deliveries_per_episode"] = _value(
            bucket, "deliveries_per_episode"
        )
        record[f"{label}_decisions"] = _value(
            bucket, "task_decisions", "decisions"
        )
        record[f"{label}_episode_count"] = _value(
            bucket, "episodes", "episode_count"
        )
    for group in ("time_metrics", "horizon_metrics", "condition_metrics"):
        record[group] = latest.get(group) or {}
    record.update(flatten_partial_metric_namespaces(latest))
    for group, prefix in (
        ("time_metrics", ""),
        ("horizon_metrics", ""),
        ("condition_metrics", "cell/"),
    ):
        for label, bucket in (latest.get(group) or {}).items():
            for field in ("collision_count", "collision_rate"):
                value = _value(bucket, field)
                if value is not None:
                    record[f"{prefix}{label}/{field}"] = value
    return record


def wandb_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map the same local record to the documented W&B namespaces."""
    mapping = {
        "train/reward": "reward",
        "train/raw_reward": "raw_reward",
        "train/sparse_reward": "sparse_reward",
        "train/deliveries": "deliveries",
        "train/deliveries_per_episode": "deliveries_per_episode",
        "train/collision_count": "collision_count",
        "diagnostics/collision_rate": "collision_rate",
        "train/decisions": "decisions",
        "ppo/actor_loss": "actor_loss",
        "ppo/critic_loss": "critic_loss",
        "ppo/entropy": "entropy",
        "ppo/approx_kl": "approx_kl",
        "ppo/clip_fraction": "clip_fraction",
        "ppo/explained_variance": "explained_variance",
        "ppo/action_explained_variance": "action_explained_variance",
        "gaze/explained_variance": "gaze_explained_variance",
        "gaze/return_mean": "gaze_return_mean",
        "gaze/return_std": "gaze_return_std",
        "gaze/value_mean": "gaze_value_mean",
        "gaze/value_std": "gaze_value_std",
        "gaze/critic_grad_clip_fraction": "gaze_critic_grad_clip_fraction",
        "gaze/weighted_advantage_mean": "gaze_weighted_advantage_mean",
        "gaze/weighted_advantage_std": "gaze_weighted_advantage_std",
        "gaze/unweighted_advantage_mean": "gaze_unweighted_advantage_mean",
        "gaze/unweighted_advantage_std": "gaze_unweighted_advantage_std",
        "ppo/actor_grad_norm": "actor_grad_norm",
        "ppo/critic_grad_norm": "critic_grad_norm",
        "gaze/gaze_requests": "gaze_requests",
        "gaze/acquisitions": "acquisitions",
        "gaze/acquisitions_per_decision": "acquisitions_per_decision",
        "gaze/held_foveal_refreshes": "held_foveal_refreshes",
        "gaze/held_foveal_refreshes_per_decision": (
            "held_foveal_refreshes_per_decision"
        ),
        "gaze/same_target_rate": "same_target_acquisition_rate",
        "gaze/new_target_rate": "new_target_acquisition_rate",
        "gaze/mean_saccade_distance": "mean_saccade_distance",
        "memory/valid_fraction": "valid_memory_fraction",
        "memory/freshness_raw_histogram": "freshness_raw_histogram",
        "memory/freshness_normalized_min": "freshness_normalized_min",
        "memory/freshness_normalized_max": "freshness_normalized_max",
        "memory/freshness_normalized_mean": "freshness_normalized_mean",
        "memory/freshness_normalized_variance": "freshness_normalized_variance",
        "memory/mean_freshness": "mean_freshness",
        "contracts/forced_stay_rate": "forced_stay_rate",
        "contracts/actor_active_rate": "actor_active_rate",
        "contracts/actionnet_query_count": "actionnet_query_count",
        "contracts/recurrent_commit_count": "recurrent_commit_count",
        "contracts/fixed_partner_guard_failures": "fixed_partner_guard_failures",
        "contracts/leakage_guard_failures": "leakage_guard_failures",
        "diagnostics/gaze_preupdate_max_abs_logprob_difference": (
            "gaze_preupdate_max_abs_logprob_difference"
        ),
        "diagnostics/gaze_preupdate_mean_abs_logprob_difference": (
            "gaze_preupdate_mean_abs_logprob_difference"
        ),
        "perf/steps_per_second": "steps_per_second",
        "perf/rollout_seconds": "rollout_seconds",
        "perf/ppo_seconds": "ppo_seconds",
        "perf/partner_seconds": "partner_seconds",
        "perf/predictor_seconds": "predictor_seconds",
        "perf/env_step_seconds": "env_step_seconds",
        "perf/ipc_wait_seconds": "ipc_wait_seconds",
    }
    for label in ("h0", "h2", "h4"):
        mapping[f"{label}/reward"] = f"{label}_reward"
        mapping[f"{label}/sparse_reward"] = f"{label}_sparse_reward"
        mapping[f"{label}/deliveries"] = f"{label}_deliveries"
        mapping[f"{label}/deliveries_per_episode"] = (
            f"{label}_deliveries_per_episode"
        )
        mapping[f"{label}/decisions"] = f"{label}_decisions"
        mapping[f"{label}/episode_count"] = f"{label}_episode_count"
    result = {
        metric: record[key]
        for metric, key in mapping.items()
        if record.get(key) is not None
    }
    result.update({
        key: value
        for key, value in record.items()
        if value is not None
        and (
            key.startswith(("t200/", "t400/", "t600/", "cell/"))
            or key.startswith(("h0/", "h2/", "h4/"))
        )
    })
    return result


class JsonlMetricsWriter:
    def __init__(self, path: Path, *, resuming: bool):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resuming else "x"
        self._handle = self.path.open(mode, encoding="utf-8")

    def append(self, record: Mapping[str, Any]) -> None:
        self._handle.write(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class ProgressReporter:
    def __init__(
        self, *, enabled: bool, description: str, total: int, initial: int
    ):
        self.enabled = bool(enabled)
        self._last_steps = int(initial)
        self._bar = None
        if self.enabled:
            from tqdm import tqdm

            self._bar = tqdm(
                total=int(total),
                initial=int(initial),
                desc=description,
                unit="environment_steps",
                dynamic_ncols=True,
                file=sys.stderr,
            )

    def update(self, record: Mapping[str, Any]) -> None:
        environment_steps = int(record["environment_steps"])
        if environment_steps < self._last_steps:
            raise RuntimeError("observability environment_steps moved backwards")
        if self._bar is not None:
            self._bar.update(environment_steps - self._last_steps)
            labels = {
                "R": record.get("reward"),
                "Del": record.get("deliveries_per_episode"),
                "KL": record.get("approx_kl"),
                "Ent": record.get("entropy"),
                "Clip": record.get("clip_fraction"),
                "Mem": record.get("valid_memory_fraction"),
                "Acq": record.get("acquisitions_per_decision"),
            }
            self._bar.set_postfix(
                {key: f"{value:.4g}" for key, value in labels.items() if value is not None},
                refresh=True,
            )
        self._last_steps = environment_steps

    def write(self, line: str) -> None:
        if self._bar is None:
            print(line, flush=True)
            return
        from tqdm import tqdm

        tqdm.write(line, file=sys.stdout)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


class TrainingObserver:
    """Own all update sinks so each completed update is emitted exactly once."""

    def __init__(
        self,
        *,
        metrics_path: Path,
        resuming: bool,
        progress: bool,
        description: str,
        total_steps: int,
        initial_steps: int,
        wandb_mode: str,
        wandb_project: str,
        wandb_name: str,
        config: Mapping[str, Any],
        tracker_factory=WandbTracker,
    ):
        self.writer = JsonlMetricsWriter(metrics_path, resuming=resuming)
        self.progress = ProgressReporter(
            enabled=progress,
            description=description,
            total=total_steps,
            initial=initial_steps,
        )
        try:
            self.tracker = tracker_factory(
                mode=wandb_mode,
                project=wandb_project,
                name=wandb_name,
                config=config,
            )
        except Exception:
            self.progress.close()
            self.writer.close()
            raise

    def emit(self, record: Mapping[str, Any]) -> None:
        self.writer.append(record)
        self.tracker.log_auxiliary(
            wandb_metrics(record),
            environment_steps=int(record["environment_steps"]),
        )
        self.progress.write(
            "TRAIN_METRICS "
            + json.dumps(record, sort_keys=True, allow_nan=False)
        )
        self.progress.update(record)

    def close(self) -> None:
        try:
            self.tracker.finish()
        finally:
            self.progress.close()
            self.writer.close()


def resolved_config(
    base: Mapping[str, Any],
    startup_report: Mapping[str, Any],
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(base)
    result["startup_report"] = dict(startup_report)
    result["source_checkpoint_sha256"] = startup_report.get(
        "source_checkpoint_sha256"
    )
    result["fixed_partner_sha256"] = startup_report.get("fixed_partner_sha256")
    result["fixed_partner_parameter_sha256"] = startup_report.get(
        "fixed_partner_parameter_sha256"
    )
    if source_provenance:
        result.update({
            "source_validation_contract_id": source_provenance.get(
                "source_validation_contract_id"
            ),
            "source_actor_parameter_sha256": source_provenance.get(
                "source_actor_parameter_sha256"
            ),
            "source_observation_contract_id": source_provenance.get(
                "source_observation_contract_id"
            ),
            "source_architecture_id": source_provenance.get(
                "source_architecture_id"
            ),
        })
    return result
