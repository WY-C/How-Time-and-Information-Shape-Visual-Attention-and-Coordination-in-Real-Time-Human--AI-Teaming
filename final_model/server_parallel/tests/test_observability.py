import builtins
import copy
import json
import math
import random
import sys

import numpy as np
import pytest
import torch

from final_model.server_parallel.observability import (
    JsonlMetricsWriter,
    ProgressReporter,
    TrainingObserver,
    build_update_record,
    resolved_config,
    wandb_metrics,
)
from final_model.server_parallel.train_server import parser, validate_server_args


def metric_bucket(
    *, reward, sparse, deliveries, decisions, episodes, forced_stays, acquisitions,
    collision_count=1,
):
    return {
        "optimization_reward": reward,
        "reward": reward,
        "sparse_team_reward": sparse,
        "sparse_reward": sparse,
        "deliveries": deliveries,
        "task_decisions": decisions,
        "decisions": decisions,
        "episodes": episodes,
        "episode_count": episodes,
        "sparse_reward_per_episode": sparse / episodes,
        "deliveries_per_episode": deliveries / episodes,
        "collision_count": collision_count,
        "collision_rate": collision_count / decisions,
        "forced_stays": forced_stays,
        "forced_stay_rate": forced_stays / decisions,
        "actor_active_rate": (decisions - forced_stays) / decisions,
        "acquisitions": acquisitions,
        "acquisitions_per_decision": acquisitions / decisions,
        "actionnet_query_count": acquisitions,
        "gaze_requests": acquisitions + 1,
        "same_target_acquisitions": acquisitions // 4,
        "new_target_acquisitions": acquisitions - acquisitions // 4,
        "same_target_rate": (acquisitions // 4) / acquisitions,
        "new_target_rate": (acquisitions - acquisitions // 4) / acquisitions,
        "valid_memory_fraction": 0.7,
        "mean_freshness": 0.6,
        "mean_saccade_distance": 4.5,
    }


def update(environment_steps=40_000, update_count=1):
    horizon_metrics = {
        "h0": metric_bucket(
            reward=3.0, sparse=2.0, deliveries=1, decisions=13_000,
            episodes=1, forced_stays=2_600, acquisitions=26,
        ),
        "h2": metric_bucket(
            reward=4.0, sparse=3.0, deliveries=1, decisions=13_000,
            episodes=1, forced_stays=2_600, acquisitions=26,
        ),
        "h4": metric_bucket(
            reward=5.5, sparse=5.0, deliveries=2, decisions=14_000,
            episodes=1, forced_stays=2_800, acquisitions=28,
        ),
    }
    time_metrics = {
        "t200": metric_bucket(
            reward=3.0, sparse=2.0, deliveries=1, decisions=13_000,
            episodes=1, forced_stays=3_900, acquisitions=20,
        ),
        "t400": metric_bucket(
            reward=4.0, sparse=3.0, deliveries=1, decisions=13_000,
            episodes=1, forced_stays=2_600, acquisitions=26,
        ),
        "t600": metric_bucket(
            reward=5.5, sparse=5.0, deliveries=2, decisions=14_000,
            episodes=1, forced_stays=1_400, acquisitions=34,
        ),
    }
    condition_metrics = {
        f"t{decision_time}_h{horizon}_plus": metric_bucket(
            reward=1.0,
            sparse=1.0,
            deliveries=1,
            decisions=1_000,
            episodes=1,
            forced_stays=100,
            acquisitions=10,
        )
        for decision_time in (200, 400, 600)
        for horizon in (0, 2, 4)
    }
    return {
        "environment_steps": environment_steps,
        "update_count": update_count,
        "reward": 12.5,
        "sparse_team_reward": 10.0,
        "policy_loss": -0.1,
        "value_loss": 0.2,
        "dist_entropy": 0.3,
        "approx_kl": 0.004,
        "clip_fraction": 0.01,
        "explained_variance": 0.9,
        "actor_grad_norm": 1.2,
        "critic_grad_norm": 2.3,
        "deliveries": 4,
        "deliveries_per_episode": 2.0,
        "collision_count": 3,
        "collision_rate": 3 / 40_000,
        "task_decisions": 40_000,
        "actor_active_sample_rate": 0.8,
        "forced_stay_rate": 0.2,
        "gaze_requests": 100,
        "gaze_transition_count": 90,
        "gaze_policy_calls_total": 110,
        "gaze_transition_count_total": 109,
        "gaze_attributed_optimization_reward": 11.5,
        "gaze_attributed_env_boundaries": 39_000,
        "gaze_reward_conservation_error": 0.0,
        "bootstrap_unowned_reward_count": 1_000,
        "bootstrap_unowned_reward_sum": 1.0,
        "bootstrap_unowned_reward_count_total": 1_500,
        "bootstrap_unowned_reward_sum_total": 1.5,
        "actor_hash_before": "a" * 64,
        "actor_hash_after": "a" * 64,
        "critic_hash_before": "b" * 64,
        "critic_hash_after": "b" * 64,
        "action_actor_requires_grad": False,
        "action_critic_requires_grad": False,
        "action_actor_optimizer_present": False,
        "action_critic_optimizer_present": False,
        "action_optimizer_isolation_verified": True,
        "acquisitions": 80,
        "acquisitions_per_decision": 0.002,
        "same_target_acquisitions": 20,
        "new_target_acquisitions": 60,
        "valid_memory_fraction": 0.7,
        "freshness_normalized_min": 0.0,
        "freshness_normalized_max": 1.0,
        "freshness_normalized_mean": 0.6,
        "freshness_normalized_variance": 0.04,
        "mean_saccade_distance_degrees": 4.5,
        "actionnet_queries": 80,
        "recurrent_commits": 40,
        "partner_step_guard_failures": 0,
        "rollout_collection_seconds": 2.0,
        "worker_collect_seconds": 0.4,
        "actionnet_rollout_seconds": 0.3,
        "actionnet_rollout_forward_count": 80,
        "worker_commit_seconds": 0.2,
        "worker_finalize_seconds": 0.1,
        "preupdate_audit_seconds": 0.25,
        "ppo_update_seconds": 1.0,
        "partner_inference_seconds": 0.5,
        "time_metrics": time_metrics,
        "horizon_metrics": horizon_metrics,
        "condition_metrics": condition_metrics,
        "shared_horizon_metrics": horizon_metrics,
    }


def record(latest=None, previous_environment_steps=0):
    return build_update_record(
        update() if latest is None else latest,
        wall_time_seconds=5.0,
        update_wall_seconds=4.0,
        previous_environment_steps=previous_environment_steps,
    )


def assert_numpy_state_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_jsonl_emits_exactly_one_valid_record_per_completed_update(tmp_path):
    writer = JsonlMetricsWriter(tmp_path / "reports/metrics.jsonl", resuming=False)
    writer.append(record())
    writer.append(record(update(80_000, 2), previous_environment_steps=40_000))
    writer.close()

    lines = (tmp_path / "reports/metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["environment_steps"] for line in lines] == [40_000, 80_000]
    assert [json.loads(line)["update_index"] for line in lines] == [1, 2]


def test_new_run_jsonl_is_fail_closed_and_resume_appends(tmp_path):
    path = tmp_path / "reports/metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"environment_steps": 1}\n')
    with pytest.raises(FileExistsError):
        JsonlMetricsWriter(path, resuming=False)

    writer = JsonlMetricsWriter(path, resuming=True)
    writer.append(record())
    writer.close()
    assert len(path.read_text().splitlines()) == 2


def test_record_contains_required_metrics_and_horizon_buckets():
    value = record()
    assert value["environment_steps"] == 40_000
    assert value["steps_per_second"] == 10_000.0
    assert value["same_target_acquisition_rate"] == 0.25
    assert value["new_target_acquisition_rate"] == 0.75
    assert value["fixed_partner_guard_failures"] == 0
    assert value["collision_count"] == 3
    assert value["collision_rate"] == 3 / 40_000
    assert value["gaze_policy_calls_total"] == 110
    assert value["gaze_transition_count_total"] == 109
    assert value["gaze_attributed_env_boundaries"] == 39_000
    assert value["bootstrap_unowned_reward_count"] == 1_000
    assert value["bootstrap_unowned_reward_count_total"] == 1_500
    assert value["gaze_reward_conservation_error"] == 0.0
    assert value["actor_hash_before"] == value["actor_hash_after"]
    assert value["critic_hash_before"] == value["critic_hash_after"]
    assert value["action_actor_optimizer_present"] is False
    assert value["action_critic_optimizer_present"] is False
    assert value["action_optimizer_isolation_verified"] is True
    assert value["h0_reward"] == 3.0
    assert value["h0_sparse_reward"] == 2.0
    assert value["h0_decisions"] == 13_000
    assert value["h0_episode_count"] == 1
    assert value["h4_deliveries"] == 2
    assert set(value["time_metrics"]) == {"t200", "t400", "t600"}
    assert len(value["condition_metrics"]) == 9
    assert value["t200/forced_stay_rate"] == 3_900 / 13_000
    assert value["t600/acquisitions_per_decision"] == 34 / 14_000
    assert value["cell/t400_h2_plus/deliveries_per_episode"] == 1.0
    assert value["t200/collision_count"] == 1
    assert value["h4/collision_rate"] == 1 / 14_000
    assert value["cell/t400_h2_plus/collision_count"] == 1
    expected_timings = {
        "update_wall_seconds": 4.0,
        "worker_collect_seconds": 0.4,
        "actionnet_rollout_seconds": 0.3,
        "partner_seconds": 0.5,
        "worker_commit_seconds": 0.2,
        "worker_finalize_seconds": 0.1,
        "preupdate_audit_seconds": 0.25,
        "ppo_seconds": 1.0,
        "update_unattributed_seconds": 1.25,
    }
    for name, expected in expected_timings.items():
        assert math.isfinite(value[name])
        assert value[name] >= 0.0
        assert value[name] == pytest.approx(expected)
    assert value["actionnet_rollout_forward_count"] == 80
    assert value["actionnet_rollout_forward_count"] == value[
        "actionnet_query_count"
    ]
    for label in ("h0", "h2", "h4"):
        for metric in (
            "reward", "sparse_reward", "deliveries",
            "deliveries_per_episode", "decisions", "episode_count",
        ):
            assert value[f"{label}_{metric}"] is not None
    assert sum(value[f"{label}_reward"] for label in ("h0", "h2", "h4")) == value["reward"]
    assert sum(value[f"{label}_sparse_reward"] for label in ("h0", "h2", "h4")) == value["sparse_reward"]
    assert sum(value[f"{label}_deliveries"] for label in ("h0", "h2", "h4")) == value["deliveries"]
    assert sum(value[f"{label}_decisions"] for label in ("h0", "h2", "h4")) == value["decisions"]


def test_parallel_gaze_record_reports_gaze_ppo_not_frozen_action_ppo():
    latest = update()
    latest.update({
        "training_contract": "parallel_gaze_v1",
        "worker_count": 8,
        "gaze_transition_count": 123,
        "gaze_worker_weight_sync_verified": True,
        "gaze_preupdate_max_abs_logprob_difference": 4.5e-6,
        "gaze_preupdate_mean_abs_logprob_difference": 1.7e-7,
        "gaze_preupdate_logprob_tolerance": 1e-5,
        "gaze_logprob_audit_version": "cpu_matched_threads_v3",
        "gaze_logprob_execution_settings_matched": True,
        "gaze_logprob_main_cpu_intra_op_threads": 1,
        "gaze_logprob_main_cpu_inter_op_threads": 1,
        "gaze_logprob_worker_cpu_intra_op_threads": 1,
        "gaze_logprob_worker_cpu_inter_op_threads": 1,
        "gaze_logprob_inference_dtype": "float32",
        "gaze_logprob_inference_path": "eval_no_grad_distribution_v1",
        "gaze_policy_loss": 1.25,
        "gaze_value_loss": 2.5,
        "gaze_entropy": 3.75,
        "gaze_approx_kl": 0.0125,
        "gaze_clip_fraction": 0.125,
        "gaze_actor_grad_norm": 4.5,
        "gaze_critic_grad_norm": 5.5,
        "gaze_explained_variance": 0.625,
        "gaze_return_mean": 7.0,
        "gaze_return_std": 2.0,
        "gaze_value_mean": 6.5,
        "gaze_value_std": 1.5,
        "gaze_critic_grad_clip_fraction": 0.75,
        "gaze_weighted_advantage_mean": 0.25,
        "gaze_weighted_advantage_std": 1.25,
        "gaze_unweighted_advantage_mean": 0.5,
        "gaze_unweighted_advantage_std": 1.5,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "dist_entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "actor_grad_norm": 0.0,
        "critic_grad_norm": 0.0,
    })
    value = record(latest)
    assert value["actor_loss"] == 1.25
    assert value["critic_loss"] == 2.5
    assert value["entropy"] == 3.75
    assert value["approx_kl"] == 0.0125
    assert value["clip_fraction"] == 0.125
    assert value["actor_grad_norm"] == 4.5
    assert value["critic_grad_norm"] == 5.5
    assert value["action_explained_variance"] == 0.9
    assert value["gaze_explained_variance"] == 0.625
    assert value["gaze_return_mean"] == 7.0
    assert value["gaze_return_std"] == 2.0
    assert value["gaze_value_mean"] == 6.5
    assert value["gaze_value_std"] == 1.5
    assert value["gaze_critic_grad_clip_fraction"] == 0.75
    assert value["gaze_weighted_advantage_mean"] == 0.25
    assert value["gaze_weighted_advantage_std"] == 1.25
    assert value["gaze_unweighted_advantage_mean"] == 0.5
    assert value["gaze_unweighted_advantage_std"] == 1.5
    assert value["gaze_transition_count"] == 123
    assert value["training_contract"] == "parallel_gaze_v1"
    assert value["worker_count"] == 8
    assert value["gaze_worker_weight_sync_verified"] is True
    assert value["gaze_preupdate_max_abs_logprob_difference"] == 4.5e-6
    assert value["gaze_preupdate_mean_abs_logprob_difference"] == 1.7e-7
    assert value["gaze_preupdate_logprob_tolerance"] == 1e-5
    assert value["gaze_logprob_audit_version"] == "cpu_matched_threads_v3"
    assert value["gaze_logprob_execution_settings_matched"] is True
    assert value["gaze_logprob_main_cpu_intra_op_threads"] == 1
    assert value["gaze_logprob_main_cpu_inter_op_threads"] == 1
    assert value["gaze_logprob_worker_cpu_intra_op_threads"] == 1
    assert value["gaze_logprob_worker_cpu_inter_op_threads"] == 1
    assert value["gaze_logprob_inference_dtype"] == "float32"
    assert value["gaze_logprob_inference_path"] == (
        "eval_no_grad_distribution_v1"
    )


def test_disabled_wandb_does_not_import_or_initialize_wandb(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("disabled W&B attempted an import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    observer = TrainingObserver(
        metrics_path=tmp_path / "reports/metrics.jsonl",
        resuming=False,
        progress=False,
        description="test",
        total_steps=40_000,
        initial_steps=0,
        wandb_mode="disabled",
        wandb_project="test",
        wandb_name="test",
        config={},
    )
    observer.emit(record())
    observer.close()


def test_wandb_uses_environment_steps_as_global_step(tmp_path):
    class FakeTracker:
        instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.logs = []
            self.finished = False
            FakeTracker.instance = self

        def log_auxiliary(self, metrics, *, environment_steps):
            self.logs.append((dict(metrics), environment_steps))

        def finish(self):
            self.finished = True

    observer = TrainingObserver(
        metrics_path=tmp_path / "reports/metrics.jsonl",
        resuming=False,
        progress=False,
        description="test",
        total_steps=40_000,
        initial_steps=0,
        wandb_mode="online",
        wandb_project="test",
        wandb_name="test",
        config={},
        tracker_factory=FakeTracker,
    )
    observer.emit(record())
    observer.close()

    assert FakeTracker.instance.logs[0][1] == 40_000
    assert FakeTracker.instance.logs[0][0]["train/reward"] == 12.5
    assert FakeTracker.instance.logs[0][0]["h4/reward"] == 5.5
    assert FakeTracker.instance.logs[0][0]["h4/sparse_reward"] == 5.0
    assert FakeTracker.instance.logs[0][0]["h4/deliveries"] == 2
    assert FakeTracker.instance.logs[0][0]["h4/deliveries_per_episode"] == 2.0
    assert FakeTracker.instance.logs[0][0]["h4/decisions"] == 14_000
    assert FakeTracker.instance.logs[0][0]["h4/episode_count"] == 1
    assert FakeTracker.instance.logs[0][0]["train/collision_count"] == 3
    assert FakeTracker.instance.logs[0][0]["diagnostics/collision_rate"] == 3 / 40_000
    assert FakeTracker.instance.logs[0][0]["t200/forced_stay_rate"] == 0.3
    assert (
        FakeTracker.instance.logs[0][0][
            "cell/t600_h4_plus/acquisitions_per_decision"
        ]
        == 0.01
    )
    assert FakeTracker.instance.finished


def test_progress_disabled_does_not_import_tqdm(monkeypatch):
    monkeypatch.setitem(sys.modules, "tqdm", None)
    progress = ProgressReporter(
        enabled=False, description="test", total=40_000, initial=0
    )
    progress.update(record())
    progress.close()


def test_observability_preserves_rng_and_update_inputs(tmp_path):
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    latest = update()
    before_latest = copy.deepcopy(latest)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    observer = TrainingObserver(
        metrics_path=tmp_path / "reports/metrics.jsonl",
        resuming=False,
        progress=False,
        description="test",
        total_steps=40_000,
        initial_steps=0,
        wandb_mode="disabled",
        wandb_project="test",
        wandb_name="test",
        config={},
    )
    observer.emit(record(latest))
    observer.close()

    assert latest == before_latest
    assert random.getstate() == python_before
    assert_numpy_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_wandb_namespaces_and_resolved_provenance():
    metrics = wandb_metrics(record())
    assert {
        "train/reward", "ppo/approx_kl", "gaze/acquisitions",
        "memory/valid_fraction", "contracts/fixed_partner_guard_failures",
        "h0/reward", "h0/sparse_reward", "h0/decisions", "h0/episode_count",
        "h2/reward", "h4/reward", "perf/steps_per_second",
        "t200/deliveries_per_episode", "t400/forced_stay_rate",
        "t600/acquisitions_per_decision",
        "cell/t200_h0_plus/reward", "cell/t600_h4_plus/episode_count",
    }.issubset(metrics)
    assert len([key for key in metrics if key.startswith("cell/")]) == 9 * 23
    config = resolved_config(
        {"seed": 3},
        {"source_checkpoint_sha256": "source", "fixed_partner_sha256": "partner"},
        {
            "source_actor_parameter_sha256": "actor",
            "source_validation_contract_id": "validation",
            "source_observation_contract_id": "observation",
            "source_architecture_id": "architecture",
        },
    )
    assert config["source_checkpoint_sha256"] == "source"
    assert config["source_actor_parameter_sha256"] == "actor"
    assert config["fixed_partner_sha256"] == "partner"


def test_progress_cli_default_and_explicit_disable():
    required = [
        "--observation-mode", "full", "--train-components", "action",
        "--horizon", "0",
    ]
    assert parser().parse_args(required).progress is True
    assert parser().parse_args(required + ["--no-progress"]).progress is False


def test_partial_accepts_offline_wandb_mode(session_options):
    args = parser().parse_args([
        "--observation-mode", "partial",
        "--train-components", "action",
        "--horizon", "0",
        "--gaze-type", "rule_based",
        "--decision-time", "200",
        "--fixation-shape", "3x3",
        "--source-checkpoint", str(session_options()["source_checkpoint"]),
        "--fixed-partner-checkpoint", str(session_options()["fixed_partner_checkpoint"]),
        "--wandb-mode", "offline",
    ])
    validate_server_args(args)
    assert args.wandb_mode == "offline"
