import numpy as np
import pytest

from final_model.eval.temporal_action import PeriodicThresholdDecisionEngine


def _run(
    probabilities, *, decision_time=100, tau=.6, acquisition_time=None,
    threshold_rule="max_probability", motor_delay_ms=0.0, logits=None,
    commitment_rule="fixed", tau_start=None, tau_min=0.0,
    tau_slope_per_ms=None, tau_kappa_ms=None,
):
    visible = {"acquired": False}
    advances = []

    def advance(timestamp):
        advances.append(timestamp)
        if acquisition_time is not None and timestamp >= acquisition_time:
            visible["acquired"] = True

    def query(timestamp):
        index = int(round(timestamp / 20))
        value = probabilities[index](visible) if callable(probabilities[index]) else probabilities[index]
        response = {
            "probabilities": np.asarray(value, dtype=float),
            "available_actions": np.ones(6),
            "candidate_hidden": np.asarray([index]),
        }
        if logits is not None:
            response["logits"] = np.asarray(logits[index], dtype=float)
        return response

    result = PeriodicThresholdDecisionEngine(
        tau=tau, threshold_rule=threshold_rule, motor_delay_ms=motor_delay_ms,
        commitment_rule=commitment_rule, tau_start=tau_start, tau_min=tau_min,
        tau_slope_per_ms=tau_slope_per_ms, tau_kappa_ms=tau_kappa_ms,
    ).run(
        decision_onset_ms=0,
        decision_time_ms=decision_time,
        advance_to=advance,
        query_actionnet=query,
    )
    return result, advances


def _p(action, confidence=.8):
    value = np.full(6, (1 - confidence) / 5)
    value[action] = confidence
    return value


def _softmax(logits):
    shifted = np.asarray(logits, dtype=float) - np.max(logits)
    return np.exp(shifted) / np.exp(shifted).sum()


def test_queries_exactly_every_20ms_and_never_at_deadline():
    result, advances = _run([_p(0)] * 5)
    assert [row["query_time_ms"] for row in result.queries] == [0, 20, 40, 60, 80]
    assert advances == [0, 20, 40, 60, 80, 100]


def test_acquisition_at_query_timestamp_is_visible_before_query():
    def at_40(visible):
        return _p(2 if visible["acquired"] else 0)
    result, _ = _run([_p(0), _p(0), at_40, _p(2), _p(2)], acquisition_time=40)
    assert result.queries[2]["argmax_action"] == 2


def test_gaze_acquisition_time_is_not_quantized_to_action_grid():
    pending = [33.25]
    observed_events = []
    visible = {"value": False}

    def advance(timestamp):
        while pending and pending[0] <= timestamp:
            event_time = pending.pop(0)
            observed_events.append(event_time)
            visible["value"] = True

    def query(_timestamp):
        return {
            "probabilities": _p(2 if visible["value"] else 0),
            "available_actions": np.ones(6),
        }

    result = PeriodicThresholdDecisionEngine(tau=.6).run(
        decision_onset_ms=0, decision_time_ms=60,
        advance_to=advance, query_actionnet=query,
    )
    assert observed_events == [33.25]
    assert result.queries[1]["argmax_action"] == 0
    assert result.queries[2]["argmax_action"] == 2


def test_stay_crossing_is_intentional_but_no_crossing_is_forced():
    intentional, _ = _run([_p(4)] * 5)
    assert intentional.intentional_stay and not intentional.forced_stay
    forced, _ = _run([np.ones(6) / 6] * 5)
    assert forced.forced_stay and not forced.intentional_stay
    assert forced.executed_action == 4


def test_margin_threshold_uses_top1_minus_top2_probability():
    probabilities = np.asarray([.55, .35, .04, .03, .02, .01])

    max_probability, _ = _run([probabilities] * 5, tau=.3)
    margin_forced, _ = _run(
        [probabilities] * 5, tau=.3, threshold_rule="top1_top2_margin"
    )
    margin_selected, _ = _run(
        [probabilities] * 5, tau=.19, threshold_rule="top1_top2_margin"
    )

    assert not max_probability.forced_stay
    assert margin_forced.forced_stay
    assert not margin_selected.forced_stay
    assert margin_selected.queries[0]["top2_probability"] == .35
    assert np.isclose(margin_selected.queries[0]["top1_top2_margin"], .2)
    assert np.isclose(margin_selected.queries[0]["threshold_score"], .2)
    assert margin_selected.queries[0]["threshold_rule"] == "top1_top2_margin"


def test_logit_margin_threshold_uses_raw_top_two_difference():
    logits = np.asarray([2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    probabilities = _softmax(logits)
    result, _ = _run(
        [probabilities] * 5,
        logits=[logits] * 5,
        tau=.9,
        threshold_rule="top1_top2_logit_margin",
    )

    assert not result.forced_stay
    assert result.queries[0]["top1_logit"] == 2.0
    assert result.queries[0]["top2_logit"] == 1.0
    assert result.queries[0]["logit_margin"] == 1.0


def test_exact_logit_top1_tie_uses_probability_argmax_without_false_mismatch():
    logits = np.asarray([2.0, 0.0, -1.0, -2.0, -3.0, 2.0])
    probabilities = _softmax(logits)
    result, _ = _run(
        [probabilities] * 5,
        logits=[logits] * 5,
        tau=0.0,
        threshold_rule="top1_top2_logit_margin",
    )

    assert result.queries[0]["argmax_action"] == 0
    assert result.queries[0]["logit_margin"] == 0.0
    assert result.queries[0]["proposal_accepted"]


def test_linear_bound_uses_absolute_elapsed_milliseconds():
    logits = np.asarray([1.5, 0.0, -1.0, -2.0, -3.0, -4.0])
    probabilities = _softmax(logits)
    result, _ = _run(
        [probabilities] * 5,
        logits=[logits] * 5,
        tau=0.0,
        threshold_rule="top1_top2_logit_margin",
        commitment_rule="linear",
        tau_start=2.0,
        tau_slope_per_ms=.01,
    )

    assert [row["threshold_at_query"] for row in result.queries] == [
        2.0, 1.8, 1.6, 1.4, 1.2,
    ]
    assert result.literal_first_threshold_crossing_time_ms == 60.0


def test_exponential_bound_matches_global_trajectory():
    logits = np.asarray([2.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    probabilities = _softmax(logits)
    result, _ = _run(
        [probabilities] * 5,
        logits=[logits] * 5,
        tau=0.0,
        threshold_rule="top1_top2_logit_margin",
        commitment_rule="exponential",
        tau_start=4.0,
        tau_min=1.0,
        tau_kappa_ms=100.0,
    )

    expected = 1.0 + 3.0 * np.exp(-40.0 / 100.0)
    assert result.queries[2]["threshold_at_query"] == pytest.approx(expected)


def test_none_accepts_every_top1_immediately_and_never_forces_stay():
    result, _ = _run(
        [_p(0), _p(0), _p(2), _p(2), _p(4)],
        tau=0.0,
        commitment_rule="none",
    )

    assert result.literal_first_threshold_crossing_time_ms is None
    assert result.first_action_input_time_ms == 0.0
    assert result.selected_action_input_time_ms == 80.0
    assert result.final_selected_action == 4
    assert result.intentional_stay
    assert not result.forced_stay
    assert result.action_change_count == 2
    assert all(row["proposal_accepted"] for row in result.queries)
    assert all(np.isnan(row["threshold_at_query"]) for row in result.queries)
    assert [row["proposal_arrival_time_ms"] for row in result.queries] == [
        0.0, 20.0, 40.0, 60.0, 80.0,
    ]


def test_none_rejects_action_input_delay():
    with pytest.raises(ValueError, match="zero action-input delay"):
        PeriodicThresholdDecisionEngine(
            tau=0.0, commitment_rule="none", motor_delay_ms=1.0,
        )


def test_motor_delay_moves_input_time_without_moving_threshold_crossing():
    result, _ = _run([_p(2)] * 5, motor_delay_ms=37.5)

    assert result.literal_first_threshold_crossing_time_ms == 0
    assert result.first_action_input_time_ms == 37.5
    assert result.selected_action_input_time_ms == 37.5
    assert result.final_selected_action == 2
    assert result.queries[0]["scheduled_input_time_ms"] == 37.5
    assert result.queries[0]["input_arrives_before_deadline"]
    assert result.queries[0]["action_input_delay_ms"] == 37.5
    assert result.queries[0]["proposal_query_time_ms"] == 0.0
    assert result.queries[0]["proposal_arrival_time_ms"] == 37.5
    assert result.executed_proposal_query_time_ms == 60.0
    assert result.executed_proposal_arrival_time_ms == 97.5


def test_motor_input_after_deadline_is_forced_stay():
    result, _ = _run([_p(2)] * 5, motor_delay_ms=101.0)

    assert result.literal_first_threshold_crossing_time_ms == 0
    assert result.first_action_input_time_ms is None
    assert result.selected_action_input_time_ms is None
    assert result.forced_stay
    assert result.executed_action == 4
    assert not result.queries[0]["input_arrives_before_deadline"]


def test_late_revision_does_not_replace_input_that_arrived_before_deadline():
    result, _ = _run(
        [_p(0), _p(0), _p(0), _p(0), _p(2)], motor_delay_ms=37.5
    )

    assert result.first_selected_action == 0
    assert result.final_selected_action == 0
    assert result.selected_action_input_time_ms == 37.5
    assert not result.canonical_had_revision
    assert result.executed_proposal_query_time_ms == 60.0
    assert result.executed_proposal_arrival_time_ms == 97.5


def test_revision_persistence_and_same_action_repeats():
    below = np.ones(6) / 6
    result, _ = _run([_p(0), _p(0), _p(2), below, _p(2)])
    assert result.first_selected_action == 0
    assert result.final_selected_action == 2
    assert result.action_change_count == 1
    assert result.canonical_had_revision
    assert result.selected_action_input_time_ms == 40
    assert result.executed_action == 2


def test_gaze_process_advances_to_deadline_after_first_selection():
    result, advances = _run([_p(0)] * 5)
    assert result.literal_first_threshold_crossing_time_ms == 0
    assert advances[-1] == 100


def test_nonstay_action_is_not_reclassified_when_transition_is_blocked_elsewhere():
    result, _ = _run([_p(3)] * 5)
    simulated_realized_transition = False
    assert result.executed_action == 3
    assert not result.intentional_stay and not result.forced_stay
    assert not simulated_realized_transition


def test_a_to_b_to_a_keeps_change_diagnostic_but_not_canonical_revision():
    result, _ = _run([_p(0), _p(1), _p(0), _p(0), _p(0)])
    assert result.action_change_count == 2
    assert result.temporal_had_action_change
    assert not result.canonical_had_revision
    assert result.selected_action_input_time_ms == 0


def test_legal_action_mask_is_rejected():
    engine = PeriodicThresholdDecisionEngine(tau=.5)
    try:
        engine.run(
            decision_onset_ms=0,
            decision_time_ms=20,
            advance_to=lambda _: None,
            query_actionnet=lambda _: {
                "probabilities": _p(0),
                "available_actions": np.asarray([0, 1, 1, 1, 1, 1]),
            },
        )
    except RuntimeError as error:
        assert "masking" in str(error)
    else:
        raise AssertionError("masked ActionNet query was accepted")
