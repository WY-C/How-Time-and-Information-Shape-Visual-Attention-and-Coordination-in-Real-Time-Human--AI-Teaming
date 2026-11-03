import pytest

from final_model.eval.partial_rollout import PartialEvaluationConfig, evaluate_partial
from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.common.paths import asset
from final_model.partial_ver.canonical_training import PartialTrainingSession
from final_model.partial_ver.gaze import resolve_emma_timing


RULE = {"task": 0.60, "stay": 0.20, "stale": 0.15, "random": 0.05}
PARTNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")


def test_partial_eval_is_hash_preserving_and_counterfactual_observational(tmp_path):
    timing = resolve_emma_timing(
        "tuned", prep_ms=135, exec_intercept_ms=44,
        exec_slope_ms_per_degree=3.8, dwell_ms=60,
    )
    session = PartialTrainingSession(
        gaze_type="rule_based", train_action=True, train_gaze=False,
        decision_times=[200], horizons=[0], fixation_shapes=["plus"], seed=43,
        rollout_length=2, episode_length=2, n_rollout_threads=1, device="cpu",
        source_checkpoint=None, configured_total_steps=2, checkpoint_interval=2,
        fixed_partner_checkpoint=PARTNER,
        reward_shaping_horizon_steps=20_000_000,
        emma_timing=timing,
        rule_gaze_probabilities=RULE, ppo_epochs=1, num_mini_batches=1,
    )
    session.run_update()
    checkpoint = tmp_path / "partial.pt"
    session.save(checkpoint)
    loaded = load_partial_checkpoint_for_evaluation(checkpoint)
    assert loaded.constraint_config == session.constraint_config
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    emma_keys = set(timing.metadata())
    for key in set(session.constraint_config.metadata()) - emma_keys:
        payload["metadata"].pop(key)
        payload["training_state"]["contract"].pop(key)
        for adapter in payload["training_state"]["adapters"]:
            adapter.pop(key)
    legacy_tuned = tmp_path / "legacy_tuned.pt"
    torch.save(payload, legacy_tuned)
    legacy_loaded = load_partial_checkpoint_for_evaluation(legacy_tuned)
    assert legacy_loaded.constraint_config.profile == "custom"
    assert legacy_loaded.constraint_config.emma == timing
    base = dict(
        checkpoint=checkpoint, decision_time=200, horizon=0,
        fixation_shape="plus", gaze_type="rule_based", episodes=1,
        episode_length=3, seed_start=99,
    )
    native = evaluate_partial(PartialEvaluationConfig(
        **base, gaze_deterministic=False, action_deterministic=True,
        aoi_diagnostic=True,
    ))
    assert native["manifest"]["gaze_selection"] == "stochastic_categorical"
    assert native["manifest"]["task_action_selection"] == "deterministic_argmax"
    assert native["episodes"][0]["decisions"] == 3
    assert (
        native["summary"]["aoi_fixation_duration_ms"]
        + native["summary"]["aoi_saccade_duration_ms"]
        == 600.0
    )
    assert native["summary"]["aoi_window_duration_ms"] == 600.0
    assert native["summary"]["accounted_gaze_duration_ms"] == 600.0
    assert native["summary"]["classified_fixation_duration_ms"] == (
        native["summary"]["aoi_fixation_duration_ms"]
    )
    assert native["summary"]["saccade_duration_ms"] == (
        native["summary"]["aoi_saccade_duration_ms"]
    )
    if native["summary"]["classified_fixation_duration_ms"] > 0.0:
        assert sum(
            native["summary"][f"fixation_prop_{aoi}"]
            for aoi in ("self", "teammate", "preview", "other")
        ) == pytest.approx(1.0)
    assert len(native["fixation_duration_by_decision"]) == 3
    assert all(
        row["accounted_gaze_duration_ms"] == 200.0
        for row in native["fixation_duration_by_decision"]
    )
    assert sum(
        row["duration_ms"] for row in native["fixation_segments"]
    ) == 600.0
    assert all(
        row["included_in_fixation_duration"] == (row["gaze_phase"] != "EXEC")
        for row in native["fixation_segments"]
    )
    assert native["summary"]["collision_count"] in range(4)
    assert 0.0 <= native["summary"]["collision_rate"] <= 1.0
    assert (
        native["summary"]["collision_contract_id"]
        == "human_compatible_joint_transition_collision_v1"
    )
    audited = evaluate_partial(PartialEvaluationConfig(
        **base, freshness_diagnostic=True
    ))
    assert native["episodes"][0]["action_trajectory_sha256"] == audited["episodes"][0]["action_trajectory_sha256"]
    assert audited["summary"]["counterfactual_query_count"] > 0
    assert any(
        row["l1_probability_distance"] > 0
        for row in audited["freshness_counterfactual"]
    )
    assert all(
        {"kl_native_counterfactual", "js_divergence", "delta_P_INTERACT"}
        <= set(row)
        for row in audited["freshness_counterfactual"]
    )
    assert audited["summary"]["optimizer_free"]
    assert audited["summary"]["parameter_hash_preserved"]
    assert audited["summary"]["fixed_partner_identity"] == "hsp_s2_s24_seed3"
    assert audited["manifest"]["emma_mode"] == "tuned"
    assert audited["manifest"]["emma_exec_intercept_ms"] == 44
    assert audited["manifest"]["constraint_profile"] == "custom"
    assert audited["manifest"]["constraint_variant_id"] == "custom"
import torch
