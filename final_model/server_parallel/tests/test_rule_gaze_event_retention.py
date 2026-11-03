from types import SimpleNamespace

from final_model.partial_ver.rule_based import RuleBasedGazePolicy
from final_model.server_parallel.rollout_worker import (
    RULE_GAZE_EVENT_HISTORY_LIMIT,
    _trace_delta,
    _trim_rule_gaze_event_history,
)


def adapter_with_events(events):
    return SimpleNamespace(
        events=[],
        gaze_policy=SimpleNamespace(events=events),
    )


def test_many_decisions_retain_at_most_checkpoint_tail_in_place():
    events = []
    adapter = adapter_with_events(events)
    identity = id(events)
    all_generated = []
    for decision in range(1_000):
        rule_count = len(events)
        generated = [
            {"decision": decision, "event": index}
            for index in range(decision % 10 + 1)
        ]
        events.extend(generated)
        all_generated.extend(generated)
        delta = _trace_delta(adapter, 0, rule_count)
        assert delta["rule_events"] == generated
        assert _trim_rule_gaze_event_history(adapter) <= 128
        assert id(adapter.gaze_policy.events) == identity
        assert events == all_generated[-128:]


def test_trace_delta_captures_current_decision_before_tail_trim():
    previous = [{"old": index} for index in range(128)]
    adapter = adapter_with_events(previous)
    rule_count = len(previous)
    generated = [{"current": index} for index in range(10)]
    previous.extend(generated)

    delta = _trace_delta(adapter, 0, rule_count)
    retained = _trim_rule_gaze_event_history(adapter)

    assert delta["rule_events"] == generated
    assert retained == RULE_GAZE_EVENT_HISTORY_LIMIT
    assert adapter.gaze_policy.events == (
        [{"old": index} for index in range(10, 128)] + generated
    )


def test_checkpoint_serialized_event_tail_is_unchanged_by_runtime_trim():
    policy = RuleBasedGazePolicy("plus", seed=41)
    policy.events = [{"event": index} for index in range(300)]
    expected = policy.training_state()["events"]
    adapter = SimpleNamespace(gaze_policy=policy)

    _trim_rule_gaze_event_history(adapter)

    assert policy.training_state()["events"] == expected
    assert policy.events == expected
