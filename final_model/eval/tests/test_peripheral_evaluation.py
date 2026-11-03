from __future__ import annotations

import json

from final_model.common.paths import asset
from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.eval.gaze_behavior_metrics import analyze_saved_model, read_csv
from final_model.eval.gaze_behavior_rollout import RolloutConfig, run_model_rollouts
from final_model.partial_ver.canonical_training import PartialTrainingSession
from final_model.partial_ver.constraints import (
    HumanLikeConstraintConfig,
    PeripheralVisionConfig,
    PreviewOpacityConfig,
)
from final_model.partial_ver.gaze import DEFAULT_EMMA_TIMING


PARTNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")
RULE = {"task": 0.60, "stay": 0.20, "stale": 0.15, "random": 0.05}


def _config():
    return HumanLikeConstraintConfig(
        profile="custom", peripheral=PeripheralVisionConfig("global", 3.0),
        emma=DEFAULT_EMMA_TIMING,
        preview_opacity=PreviewOpacityConfig("actual"),
    )


def _session(**overrides):
    values = dict(
        gaze_type="rule_based", train_action=True, train_gaze=False,
        decision_times=[400], horizons=[2], fixation_shapes=["plus"],
        seed=703, rollout_length=8, episode_length=8, n_rollout_threads=1,
        device="cpu", source_checkpoint=None, configured_total_steps=8,
        fixed_partner_checkpoint=PARTNER, checkpoint_interval=1,
        reward_shaping_horizon_steps=20_000_000,
        rule_gaze_probabilities=RULE, ppo_epochs=1, num_mini_batches=1,
        constraint_config=_config(),
    )
    values.update(overrides)
    return PartialTrainingSession(**values)


def test_p2o_rollout_and_offline_metrics_preserve_foveal_semantics(tmp_path):
    action = _session()
    action.run_update()
    action_path = tmp_path / "p2_action.pt"
    action.save(action_path)
    gaze = _session(
        gaze_type="learned", train_action=False, train_gaze=True,
        source_checkpoint=action_path,
    )
    gaze.run_update()
    gaze_path = tmp_path / "p2_gaze.pt"
    gaze.save(gaze_path)

    loaded = load_partial_checkpoint_for_evaluation(gaze_path)
    model_dir = tmp_path / "model"
    run_model_rollouts(
        loaded=loaded,
        config=RolloutConfig(
            model_name="p2", decision_times=(400,), horizons=(2,),
            fixation_shape="plus", episodes=1, episode_length=2,
            seed_start=901, gaze_mode="argmax", action_mode="argmax",
            aoi_precedence=("self", "partner", "preview"),
        ),
        model_dir=model_dir,
    )
    acquisitions = read_csv(model_dir / "acquisitions.csv.gz")
    assert acquisitions
    for row in acquisitions:
        candidates = json.loads(row["peripheral_candidate_indices"])
        successes = json.loads(row["peripheral_success_indices"])
        failures = json.loads(row["peripheral_failure_indices"])
        foveal = json.loads(row["foveal_tile_indices"])
        assert candidates == sorted(candidates)
        if int(row["peripheral_sampled_new_fixation"]):
            assert set(candidates) == set(range(40)) - set(foveal)
        else:
            assert candidates == []
        assert len(successes) + len(failures) == len(candidates)
        assert row["partner_footprint"] == row["partner_foveal_acquired"]
        assert int(row["partner_total_acquired"]) == int(
            bool(int(row["partner_foveal_acquired"]))
            or bool(int(row["partner_peripheral_acquired"]))
        )

    analysis = analyze_saved_model(
        model_dir, bootstrap_iterations=1, bootstrap_seed=1,
        aoi_precedence=("self", "partner", "preview"),
        gaze_actor=loaded.gaze_actor, gaze_device=loaded.device,
    )
    assert analysis["peripheral"]
    assert analysis["peripheral_impact"]
    assert analysis["opacity"]
    assert analysis["opacity_impact"]
    assert (model_dir / "peripheral_summary.csv").is_file()
    assert (model_dir / "peripheral_actionnet_impact.csv").is_file()
    assert (model_dir / "opacity_summary.csv").is_file()
    assert (model_dir / "opacity_actionnet_impact.csv").is_file()
    opacity = read_csv(model_dir / "opacity_summary.csv")
    all_phase = next(row for row in opacity if row["response_phase"] == "ALL")
    assert int(all_phase["renderer_alpha_h1"]) == 250
    assert int(all_phase["renderer_alpha_h2"]) == 70
    assert int(all_phase["renderer_alpha_h3"]) == 0
    assert all_phase["preview_opacity_mode"] == "actual"
    assert int(all_phase["peripheral_spatial_preview_opportunities"]) >= int(
        all_phase["opacity_attempts"]
    )
    economy = read_csv(model_dir / "memory_economy.csv")
    assert "peripheral_freshness_gain_per_acquisition" in economy[0]
    loaded.assert_unchanged()
