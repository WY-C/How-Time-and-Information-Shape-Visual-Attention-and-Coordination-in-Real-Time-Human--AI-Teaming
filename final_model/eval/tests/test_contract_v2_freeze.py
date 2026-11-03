import hashlib
import re

from final_model.eval.benchmark_contracts import (
    BENCHMARK_CONTRACT_VERSION,
    benchmark_contract_sha256,
    benchmark_contract_snapshot,
)
from final_model.eval.benchmark_provenance import (
    REQUIRED_MODEL_EVALUATION_FIELDS,
    file_sha256,
    git_provenance,
    model_evaluation_provenance,
)
from final_model.eval.common import write_json


CONTRACT_V2_SHA256 = "6e681eeb89850f442c4e51557ce7868c155b652aee84b3bada0239e8d367923e"
CONTRACT_V3_SHA256 = "9f5097e23bbb46f8d1b37ef82fe38dca0d6be547b69350ab7e54f5cb6418a840"
CONTRACT_V4_SHA256 = "607adba11e66b670e4aed41d129d84e69af202eb5740d5e3a3769250d5e9ffc7"
CONTRACT_V5_SHA256 = "7671d68237ee7a0515f6f155c2744090907173b2bde67050f84e5b9c2e0f0b5c"
CONTRACT_V6_SHA256 = "ba52a7a54f8a3ebb5f8e8cfeabc2d0ed3225ded3edcae3aa2f9df126e20891e7"
CONTRACT_V7_SHA256 = "5ad1fa14a1174a1f89b247ac03676c529a0c79d308da0e104192c7e52f5d1358"
CONTRACT_V8_SHA256 = "b24bc7df531c7e24b4d72af47e714c9042bdd60e823b9c2a1c1cf591525f662d"
CONTRACT_V9_SHA256 = "c1490c93e92c7859dcf6ac82f33a93f56cfa3e25a7a0791b9a3ec14eab2dec1d"


def test_contract_v2_snapshot_is_frozen():
    assert BENCHMARK_CONTRACT_VERSION == "contract_v2"
    assert benchmark_contract_sha256() == CONTRACT_V2_SHA256
    snapshot = benchmark_contract_snapshot()
    targets = snapshot["human_regression_targets"]
    assert targets["selected_action_latency_ms"] == {
        "200": 87.6076, "400": 211.5641, "600": 241.4401,
    }
    assert targets["post_first_input_fraction"] == {
        "200": 0.3978498, "400": 0.4571253, "600": 0.590267,
    }
    assert targets["task_events"] == {
        "ai_delivery": 3313,
        "pot_completion": 2622,
        "redundant_onion": 655,
        "redundant_release_at_deposit": 146,
        "terminal_step_179_delivery": 19,
    }


def test_contract_v3_snapshot_freezes_top1_top2_margin_rule():
    assert benchmark_contract_sha256("contract_v3") == CONTRACT_V3_SHA256
    snapshot = benchmark_contract_snapshot("contract_v3")
    assert snapshot["benchmark_contract_version"] == "contract_v3"
    assert snapshot["action_timing"]["evaluation_action_mode"] == (
        "periodic_20ms_margin_threshold"
    )
    assert snapshot["action_timing"]["threshold_rule"] == (
        "top1_probability - top2_probability >= global_tau"
    )


def test_contract_v4_snapshot_freezes_motor_delay_and_no_action_rule():
    assert benchmark_contract_sha256("contract_v4") == CONTRACT_V4_SHA256
    snapshot = benchmark_contract_snapshot("contract_v4")
    assert snapshot["action_timing"]["global_tau"] == .18
    assert snapshot["action_timing"]["motor_delay_ms"] == 87.607603
    assert snapshot["human_regression_targets"]["no_action_rate"] == {
        "200": .355315, "400": .117764, "600": .074674,
    }
    assert snapshot["stay"]["no_action_rate"] == (
        "forced_stay OR intentional_stay; reported in addition to both components"
    )
    assert snapshot["semantic_foveal_plus_coverage"]["coverage_type"] == (
        "nonexclusive semantic/foveal coverage; not an AOI composition"
    )


def test_contract_v5_changes_only_gaze_policy_sampling():
    assert benchmark_contract_sha256("contract_v5") == CONTRACT_V5_SHA256
    snapshot = benchmark_contract_snapshot("contract_v5")
    assert snapshot["benchmark_contract_version"] == "contract_v5"
    assert snapshot["action_timing"]["evaluation_action_mode"] == (
        "periodic_20ms_margin_motor_delay_stochastic_gaze"
    )
    assert snapshot["policy_sampling"]["gaze_policy"] == (
        "categorical sampling from GazeNet probabilities"
    )
    assert snapshot["policy_sampling"]["action_policy"] == (
        "deterministic top1 action after the top1-top2 margin threshold is satisfied"
    )
    assert snapshot["policy_sampling"]["action_sampling"] is False


def test_contract_v6_removes_only_motor_delay_from_stochastic_gaze_mode():
    assert benchmark_contract_sha256("contract_v6") == CONTRACT_V6_SHA256
    snapshot = benchmark_contract_snapshot("contract_v6")
    assert snapshot["action_timing"]["evaluation_action_mode"] == (
        "periodic_20ms_margin_threshold_stochastic_gaze"
    )
    assert snapshot["action_timing"]["global_tau"] == .18
    assert snapshot["action_timing"]["motor_delay_ms"] == 0.0
    assert snapshot["policy_sampling"]["gaze_policy"] == (
        "categorical sampling from GazeNet probabilities"
    )


def test_contract_v7_freezes_logit_commitment_and_absolute_time():
    assert benchmark_contract_sha256("contract_v7") == CONTRACT_V7_SHA256
    snapshot = benchmark_contract_snapshot("contract_v7")
    timing = snapshot["action_timing"]
    assert timing["evidence"].startswith("raw top1 ActionNet logit")
    assert timing["motor_delay_ms"] == 87.607603
    assert "never normalized" in timing["elapsed_time_definition"]
    assert set(timing["commitment_rules"]) == {"fixed", "linear", "exponential"}


def test_contract_v8_jointly_fits_one_global_action_input_delay():
    assert benchmark_contract_sha256("contract_v8") == CONTRACT_V8_SHA256
    snapshot = benchmark_contract_snapshot("contract_v8")
    timing = snapshot["action_timing"]
    assert timing["evaluation_action_mode"] == (
        "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze"
    )
    assert "one nonnegative global fitted D" in timing["action_input_delay"]
    assert snapshot["calibration"]["tau_min"] == 0.0
    assert "calibration objective only" in snapshot["calibration"]["family_selection"]


def test_contract_v9_freezes_no_commitment_comparator():
    assert benchmark_contract_sha256("contract_v9") == CONTRACT_V9_SHA256
    snapshot = benchmark_contract_snapshot("contract_v9")
    timing = snapshot["action_timing"]
    assert timing["evaluation_action_mode"] == (
        "periodic_20ms_no_commitment_stochastic_gaze"
    )
    assert timing["confidence_threshold"] is None
    assert timing["logit_margin_threshold"] is None
    assert timing["action_input_delay_ms"] == 0.0
    assert timing["forced_stay"].startswith("none from threshold failure")
    assert snapshot["policy_sampling"]["gaze_policy"] == (
        "categorical sampling from GazeNet probabilities"
    )


def test_model_evaluation_provenance_contains_every_required_field(tmp_path):
    seed_manifest = tmp_path / "seed_manifest.json"
    write_json(seed_manifest, {"seeds": [810000, 810001]})
    record = model_evaluation_provenance(
        git={"git_commit_hash": "a" * 40, "git_worktree_dirty": True},
        action_checkpoint_sha256="b" * 64,
        gaze_checkpoint_sha256="c" * 64,
        tau=.55,
        seed_manifest_sha256=file_sha256(seed_manifest),
        evaluation_action_mode="periodic_20ms_threshold",
    )
    assert set(REQUIRED_MODEL_EVALUATION_FIELDS).issubset(record)
    assert record["benchmark_contract_version"] == "contract_v2"
    assert record["benchmark_contract_sha256"] == CONTRACT_V2_SHA256
    assert record["seed_manifest_sha256"] == hashlib.sha256(
        seed_manifest.read_bytes()
    ).hexdigest()
    assert record["aoi_contract"] == "manuscript_preview_priority"


def test_margin_provenance_uses_contract_v3_snapshot(tmp_path):
    seed_manifest = tmp_path / "seed_manifest.json"
    write_json(seed_manifest, {"seeds": [800000]})
    record = model_evaluation_provenance(
        git={"git_commit_hash": "a" * 40, "git_worktree_dirty": True},
        action_checkpoint_sha256="b" * 64,
        gaze_checkpoint_sha256="c" * 64,
        tau=.3,
        seed_manifest_sha256=file_sha256(seed_manifest),
        evaluation_action_mode="periodic_20ms_margin_threshold",
        benchmark_contract_version="contract_v3",
    )
    assert record["benchmark_contract_version"] == "contract_v3"
    assert record["benchmark_contract_sha256"] == CONTRACT_V3_SHA256


def test_logit_commitment_provenance_records_all_rule_parameters(tmp_path):
    seed_manifest = tmp_path / "seed_manifest.json"
    write_json(seed_manifest, {"seeds": [810000]})
    record = model_evaluation_provenance(
        git={"git_commit_hash": "a" * 40, "git_worktree_dirty": True},
        action_checkpoint_sha256="b" * 64,
        gaze_checkpoint_sha256="c" * 64,
        tau=0.0,
        motor_delay_ms=87.607603,
        seed_manifest_sha256=file_sha256(seed_manifest),
        evaluation_action_mode="periodic_20ms_logit_commitment_stochastic_gaze",
        benchmark_contract_version="contract_v7",
        commitment_rule="linear",
        tau_start=.75,
        tau_min=0.0,
        tau_slope_per_ms=.00125,
    )

    assert record["commitment_rule"] == "linear"
    assert record["tau_start"] == .75
    assert record["tau_min"] == 0.0
    assert record["tau_slope_per_ms"] == .00125
    assert record["tau_kappa_ms"] is None


def test_git_commit_hash_is_available_for_benchmark_provenance():
    record = git_provenance()
    assert re.fullmatch(r"[0-9a-f]{40}", record["git_commit_hash"])
    assert isinstance(record["git_worktree_dirty"], bool)


def test_git_provenance_accepts_explicit_deployment_commit(monkeypatch, tmp_path):
    deployed_commit = "d" * 40
    monkeypatch.setenv("FINAL_MODEL_GIT_COMMIT", deployed_commit)
    record = git_provenance(tmp_path)
    assert record == {
        "git_commit_hash": deployed_commit,
        "git_worktree_dirty": True,
    }
