from pathlib import Path

import pytest

from final_model.server_parallel.partial_time_horizon_viability import (
    CONDITIONS,
    DECISION_TIMES_MS,
    HORIZONS,
    _checkpoint_id,
    action_decision_record,
    parser,
    summarize_by_time,
    summarize_condition,
    viability_interpretation,
)


def episode(seed, deliveries, *, decisions=400):
    action_decisions = [
        {
            "decision_index": index,
            "executed_action": 4 if index < 20 else 2,
            "action_decision_latency_ms": 200.0 if index < 20 else 100.0,
            "final_proposal_latency_ms": (
                None if index < 10 else (50.0 if index < 20 else 100.0)
            ),
            "idle_type": (
                "forced" if index < 10 else ("chosen" if index < 20 else None)
            ),
            "proposal_count": 0 if index < 10 else 1,
        }
        for index in range(decisions)
    ]
    return {
        "seed": seed,
        "deliveries": deliveries,
        "sparse_team_reward": deliveries * 20.0,
        "task_decisions": decisions,
        "executed_proposals": decisions - 10,
        "forced_stays": 10,
        "no_acquisition_decisions": 10,
        "actionnet_queries": 800,
        "recurrent_commits": decisions - 10,
        "gaze_requests": 810,
        "acquisitions": 800,
        "same_target_acquisitions": 600,
        "new_target_acquisitions": 200,
        "saccade_distance_sum_degrees": 1000.0,
        "valid_memory_fraction": 0.5,
        "freshness_normalized_mean": 0.4,
        "memory_expiry_count": 20,
        "prep_count": 200,
        "exec_count": 200,
        "dwell_count": 800,
        "cross_decision_gaze_operations": 30,
        "partner_step_guard_failures": 0,
        "partner_full_hash_failures": 0,
        "action_decisions": action_decisions,
    }


def collected(*, deadline, proposals):
    return {
        "deadline_ms": float(deadline),
        "proposals": proposals,
        "latest_proposal": None if not proposals else proposals[-1],
    }


def test_non_idle_action_time_uses_final_executed_proposal_timestamp():
    proposals = [
        {"action": 2, "acquisition_time_ms": 1050.0},
        {"action": 3, "acquisition_time_ms": 1180.0},
    ]
    record = action_decision_record(
        collected(deadline=1200, proposals=proposals),
        {"executed_action": 3, "forced_stay": False},
        decision_index=7,
        decision_time_ms=200,
    )
    assert record == {
        "decision_index": 7,
        "executed_action": 3,
        "action_decision_latency_ms": 180.0,
        "final_proposal_latency_ms": 180.0,
        "idle_type": None,
        "proposal_count": 2,
    }


@pytest.mark.parametrize("budget", DECISION_TIMES_MS)
def test_forced_idle_uses_full_time_budget(budget):
    record = action_decision_record(
        collected(deadline=1000 + budget, proposals=[]),
        {"executed_action": 4, "forced_stay": True},
        decision_index=0,
        decision_time_ms=budget,
    )
    assert record["action_decision_latency_ms"] == float(budget)
    assert record["final_proposal_latency_ms"] is None
    assert record["idle_type"] == "forced"


@pytest.mark.parametrize("budget", DECISION_TIMES_MS)
def test_chosen_idle_uses_full_budget_but_preserves_proposal_time(budget):
    start = 1000.0
    proposal = {"action": 4, "acquisition_time_ms": start + 50.0}
    record = action_decision_record(
        collected(deadline=start + budget, proposals=[proposal]),
        {"executed_action": 4, "forced_stay": False},
        decision_index=0,
        decision_time_ms=budget,
    )
    assert record["action_decision_latency_ms"] == float(budget)
    assert record["final_proposal_latency_ms"] == 50.0
    assert record["idle_type"] == "chosen"


def test_action_time_rejects_timestamp_outside_decision():
    proposal = {"action": 1, "acquisition_time_ms": 999.0}
    with pytest.raises(RuntimeError, match="outside"):
        action_decision_record(
            collected(deadline=1200, proposals=[proposal]),
            {"executed_action": 1, "forced_stay": False},
            decision_index=0,
            decision_time_ms=200,
        )


def test_condition_grid_is_exact_cartesian_time_by_horizon():
    assert DECISION_TIMES_MS == (200, 400, 600)
    assert HORIZONS == (0, 2, 4)
    assert CONDITIONS == tuple(
        (decision_time, horizon)
        for decision_time in DECISION_TIMES_MS
        for horizon in HORIZONS
    )


def test_condition_summary_uses_episode_and_decision_denominators_exactly():
    summary = summarize_condition(
        [episode(400_000, 2), episode(400_001, 4)],
        decision_time_ms=200,
        horizon=2,
    )
    assert summary["episode_seeds"] == [400_000, 400_001]
    assert summary["deliveries_per_episode_mean"] == 3.0
    assert summary["deliveries_per_episode_std"] == 1.0
    assert summary["sparse_reward_per_episode_mean"] == 60.0
    assert summary["action_decision_latency_ms_mean"] == 105.0
    assert summary["action_decision_latency_ms_median"] == 100.0
    assert summary["non_idle_action_decision_latency_ms_mean"] == 100.0
    assert summary["idle_count"] == 40
    assert summary["idle_rate"] == 40 / 800
    assert summary["forced_idle_count"] == 20
    assert summary["forced_idle_rate"] == 20 / 800
    assert summary["chosen_idle_count"] == 20
    assert summary["chosen_idle_rate"] == 20 / 800
    assert summary["forced_stay_rate"] == 20 / 800
    assert summary["actor_active_rate"] == 780 / 800
    assert summary["no_acquisition_decisions"] == 20
    assert summary["actionnet_queries_per_decision"] == 2.0
    assert summary["recurrent_commits_per_decision"] == 780 / 800
    assert summary["acquisitions_per_decision"] == 2.0
    assert summary["same_target_acquisition_rate"] == 0.75
    assert summary["new_target_acquisition_rate"] == 0.25
    assert summary["mean_saccade_distance_degrees"] == 5.0
    assert summary["memory_expiry_rate"] == 40 / 800
    assert summary["cross_decision_gaze_operation_rate"] == 60 / 800


def condition_rows(delivery_by_time=None):
    delivery_by_time = delivery_by_time or {200: 1.0, 400: 2.0, 600: 3.0}
    return [
        {
            "decision_time_ms": decision_time,
            "horizon": horizon,
            "deliveries_per_episode_mean": delivery_by_time[decision_time],
            "action_decision_latency_ms_mean": decision_time / 2.0,
            "idle_rate": 0.2,
            "forced_idle_rate": 0.1,
            "chosen_idle_rate": 0.1,
            "forced_stay_rate": 0.1,
            "actor_active_rate": 0.9,
            "acquisitions_per_decision": 2.0,
            "actionnet_queries_per_decision": 2.0,
            "valid_memory_fraction": 0.5,
            "no_acquisition_decision_rate": 0.1,
        }
        for decision_time, horizon in CONDITIONS
    ]


def test_time_summary_averages_exactly_three_horizons():
    rows = summarize_by_time(condition_rows())
    assert [row["decision_time_ms"] for row in rows] == [200, 400, 600]
    assert [
        row["deliveries_per_episode_mean_across_horizons"] for row in rows
    ] == [1.0, 2.0, 3.0]
    assert [
        row["action_decision_latency_ms_mean_across_horizons"] for row in rows
    ] == [100.0, 200.0, 300.0]


def test_interpretation_does_not_invent_scientific_pass_threshold():
    conditions = condition_rows()
    by_time = summarize_by_time(conditions)
    result = viability_interpretation(conditions, by_time)
    assert result["automatic_scientific_pass"] is None
    assert result["strong_time_degradation_threshold_applied"] is False
    assert result["comparable_viability_threshold_applied"] is False
    assert "No automatic" in result["recommendation"]


def test_zero_t200_t400_produces_defer_gazenet_recommendation():
    conditions = condition_rows({200: 0.0, 400: 0.0, 600: 3.0})
    result = viability_interpretation(conditions, summarize_by_time(conditions))
    assert result["t200_t400_clearly_nonviable_vs_t600"] is True
    assert "should NOT begin" in result["recommendation"]


def test_checkpoint_id_is_content_disambiguated_and_cli_requires_checkpoint():
    assert _checkpoint_id(Path("step 20M.pt"), "a" * 64) == "step_20M_aaaaaaaaaaaa"
    with pytest.raises(SystemExit):
        parser().parse_args([])
    args = parser().parse_args(["--action-checkpoint", "/tmp/action.pt"])
    assert args.action_checkpoint == Path("/tmp/action.pt")
    assert args.episodes == 30
    assert args.seed_start == 400_000
    assert args.episode_length == 400
