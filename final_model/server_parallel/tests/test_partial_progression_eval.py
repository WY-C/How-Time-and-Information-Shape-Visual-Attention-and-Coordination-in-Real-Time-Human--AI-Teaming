import pytest

from final_model.server_parallel.partial_progression_eval import (
    classify_progression,
    compact_row,
    summarize_horizon,
)


def canonical_result(deliveries=(1, 3), sparse=(20.0, 60.0)):
    rows = []
    for index in range(2):
        rows.append({
            "seed": 400_000 + index,
            "deliveries": deliveries[index],
            "sparse_team_reward": sparse[index],
            "forced_stays": index,
            "executed_proposals": 7,
            "task_decisions": 10,
            "acquisitions": 8,
            "same_target_acquisitions": 2,
            "new_target_acquisitions": 6,
            "gaze_requests": 8,
            "valid_memory_fraction": 0.5 + 0.2 * index,
            "action_trajectory_sha256": str(index),
        })
    return {
        "episodes": rows,
        "summary": {
            "optimizer_free": True,
            "parameter_hash_preserved": True,
            "fixed_partner_identity": "hsp_s2_s24_seed3",
            "fixed_partner_sha256": "partner",
        },
    }


def test_summarize_horizon_uses_sparse_task_and_canonical_diagnostics():
    summary = summarize_horizon(canonical_result())
    assert summary["deliveries_per_episode_mean"] == 2.0
    assert summary["deliveries_per_episode_std"] == 1.0
    assert summary["sparse_reward_per_episode_mean"] == 40.0
    assert summary["forced_stay_rate"] == 0.05
    assert summary["actor_active_rate"] == 0.7
    assert summary["acquisitions_per_decision"] == 0.8
    assert summary["valid_memory_fraction"] == pytest.approx(0.6)
    assert summary["same_target_acquisition_rate"] == 0.25
    assert summary["new_target_acquisition_rate"] == 0.75
    assert summary["action_distribution"] is None


def test_compact_row_and_classification_are_horizon_balanced():
    checkpoint = {
        "checkpoint": "step.pt",
        "completed_steps": 0,
        "horizons": {
            "h0": {**summarize_horizon(canonical_result()), "deliveries_per_episode_mean": 1.0},
            "h2": {**summarize_horizon(canonical_result()), "deliveries_per_episode_mean": 2.0},
            "h4": {**summarize_horizon(canonical_result()), "deliveries_per_episode_mean": 3.0},
        },
    }
    first = compact_row(checkpoint)
    second = dict(first, checkpoint="step1m.pt", completed_steps=1_000_000)
    second.update(H0_deliveries=2.0, H2_deliveries=3.0, H4_deliveries=4.0,
                  mean_deliveries=3.0, worst_horizon=2.0)
    label, basis = classify_progression([first, second])
    assert first["mean_deliveries"] == 2.0
    assert first["worst_horizon"] == 1.0
    assert first["horizon_spread"] == 2.0
    assert label == "A. clearly improving"
    assert basis["step0_to_step1m_absolute_change"] == 1.0
