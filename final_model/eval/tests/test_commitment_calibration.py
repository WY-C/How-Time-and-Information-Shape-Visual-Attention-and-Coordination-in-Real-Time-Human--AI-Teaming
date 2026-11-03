import numpy as np
import pandas as pd
import pytest

from final_model.eval.calibrate_commitment_rule import (
    DecisionTrace,
    load_query_traces,
    simulate_traces,
    summarize_simulation,
)


def _probabilities_for_gap(gap):
    logits = np.asarray([gap, 0.0, -2.0, -3.0, -4.0, -5.0])
    values = np.exp(logits - logits.max())
    return values / values.sum()


def test_existing_probability_log_recovers_exact_logit_gap(tmp_path):
    rows = []
    for index, elapsed in enumerate(range(0, 200, 20)):
        probabilities = _probabilities_for_gap(1.25)
        rows.append({
            "T": 200, "H": 0, "episode": 0, "seed": 1, "decision": 0,
            "query_time_ms": elapsed, "argmax_action": 0,
            **{
                column: probabilities[action]
                for action, column in enumerate((
                    "p_NORTH", "p_SOUTH", "p_EAST", "p_WEST", "p_STAY",
                    "p_INTERACT",
                ))
            },
        })
    path = tmp_path / "queries.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    traces, source, _ = load_query_traces(path)

    assert source == "exact_log_probability_ratio_from_complete_softmax"
    assert len(traces) == 1
    assert traces[0].logit_margin == pytest.approx(np.full(10, 1.25))


def test_offline_replay_preserves_delay_latest_proposal_and_revision_semantics():
    trace = DecisionTrace(
        T=200,
        H=0,
        episode=0,
        seed=1,
        decision=0,
        query_time_ms=np.arange(0, 200, 20, dtype=float),
        logit_margin=np.asarray([2.0] * 10),
        action=np.asarray([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
    )

    decision = simulate_traces(
        [trace], rule="fixed", parameters={
            "action_input_delay_ms": 87.607603, "tau": 1.0, "tau_min": 0.0,
        },
    ).iloc[0]

    assert decision.forced_stay == 0
    assert decision.selected_action_latency_ms == pytest.approx(87.607603)
    assert decision.canonical_revision == 0
    assert decision.temporal_action_change == 1


def test_joint_delay_changes_arrival_latency_and_forced_stay():
    trace = DecisionTrace(
        T=200, H=0, episode=0, seed=1, decision=0,
        query_time_ms=np.arange(0, 200, 20, dtype=float),
        logit_margin=np.asarray([0.0] * 9 + [2.0]),
        action=np.zeros(10, dtype=int),
    )
    selected = simulate_traces(
        [trace], rule="fixed", parameters={
            "action_input_delay_ms": 20.0, "tau": 1.0, "tau_min": 0.0,
        },
    ).iloc[0]
    forced = simulate_traces(
        [trace], rule="fixed", parameters={
            "action_input_delay_ms": 20.000001, "tau": 1.0, "tau_min": 0.0,
        },
    ).iloc[0]

    assert selected.forced_stay == 0
    assert selected.selected_action_latency_ms == 200.0
    assert forced.forced_stay == 1


def test_summary_objective_uses_only_forced_rate_and_conditional_latency():
    decisions = pd.DataFrame([
        {"T": 200, "forced_stay": 0, "intentional_stay": 0,
         "canonical_revision": 1, "temporal_action_change": 1,
         "selected_action_latency_ms": 100.0},
        {"T": 200, "forced_stay": 1, "intentional_stay": 0,
         "canonical_revision": 0, "temporal_action_change": 0,
         "selected_action_latency_ms": np.nan},
    ])
    targets = {
        200: {
            "forced_stay_rate": .5,
            "conditional_action_latency_mean_ms": 100.0,
        }
    }

    summary, objective = summarize_simulation(decisions, targets=targets)

    assert objective == 0.0
    assert summary.iloc[0].canonical_revision_rate_non_timeout == 1.0


def test_all_timeout_cell_cannot_drop_latency_term_from_objective():
    decisions = pd.DataFrame([{
        "T": 200, "forced_stay": 1, "intentional_stay": 0,
        "canonical_revision": 0, "temporal_action_change": 0,
        "selected_action_latency_ms": np.nan,
    }])
    targets = {
        200: {
            "forced_stay_rate": .5,
            "conditional_action_latency_mean_ms": 100.0,
        }
    }

    summary, objective = summarize_simulation(decisions, targets=targets)

    assert np.isinf(objective)
    assert np.isinf(summary.iloc[0].latency_objective_component)
