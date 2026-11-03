import numpy as np
import pytest

from final_model.partial_ver.constraints import (
    HumanLikeConstraintConfig,
    PeripheralVisionConfig,
    PreviewOpacityConfig,
)
from final_model.partial_ver.gaze import HUMAN_STOCHASTIC_EMMA_TIMING
from final_model.partial_ver.observation import (
    ACTOR_AVAILABLE_ACTION_CONTRACT_ID,
    DEADLINE_ACTION_CONTRACT_ID,
    FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID,
    HELD_FOVEAL_REFRESH_CONTRACT_ID,
    PartialObservationAdapter,
)


class FakeBackend:
    def __init__(self, horizon=0):
        self.version = 7
        self.horizon = int(horizon)
        self.episode_step = 0
        self.environment_steps = 0
        self.step_actions = []

    def canonical_pair(self):
        partner = np.zeros((8, 5, 20), dtype=np.float32)
        learner = np.zeros_like(partner)
        learner[..., 10:15] = np.arange(10, 15, dtype=np.float32)
        learner[0, 0, 0] = self.version
        learner[0, 0, 19] = 255
        return partner, learner

    def preview_planes(self):
        result = np.zeros((4, 8, 5), dtype=np.float32)
        for channel in range(4):
            result[channel, 0, 0] = (channel + 1) * 10.0 + self.version
        return result

    def self_tile(self):
        return (0, 0)

    def learner_available_actions(self):
        return np.asarray([0, 0, 0, 0, 1, 0], dtype=np.float32)

    def step(self, action):
        self.step_actions.append(int(action))
        self.version += 1
        self.episode_step += 1
        self.environment_steps += 1
        return {
            "reward": 0.0,
            "optimization_reward": 0.0,
            "sparse_team_reward": 0.0,
            "native_shaped_reward": 0.0,
            "weighted_shaped_reward": 0.0,
            "deliveries": 0,
            "done": False,
        }

    def reset(self):
        self.version = 7
        self.episode_step = 0

    def set_horizon(self, horizon):
        self.horizon = int(horizon)


class SameTarget:
    def reset(self):
        pass

    def __call__(self, _memory, fixation, _context):
        return fixation


class SameThenFar:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0

    def __call__(self, _memory, fixation, _context):
        self.calls += 1
        return fixation if self.calls == 1 else (7, 4)


def human_stochastic_config():
    return HumanLikeConstraintConfig(
        profile="custom",
        peripheral=PeripheralVisionConfig("none"),
        emma=HUMAN_STOCHASTIC_EMMA_TIMING,
        preview_opacity=PreviewOpacityConfig("off"),
    )


@pytest.mark.parametrize("decision_time", (200, 400, 600))
def test_human_stochastic_queries_action_once_at_each_decision_deadline(
    decision_time,
):
    adapter = PartialObservationAdapter(
        FakeBackend(),
        decision_time=decision_time,
        fixation_shape="plus",
        gaze_policy=SameTarget(),
        constraint_config=human_stochastic_config(),
        dwell_seed=123,
    )
    calls = []

    def proposal(observation, _critic, _available, timestamp):
        calls.append((timestamp, observation[0, 0, 0]))
        return {"action": 2, "candidate_hidden": np.asarray([timestamp])}

    first = adapter.collect_decision(proposal)
    assert calls == [(float(decision_time), 7.0)]
    assert len(first["proposals"]) == 1
    assert first["latest_proposal"]["action_query_time_ms"] == float(
        decision_time
    )
    assert not first["forced_stay"]
    assert adapter.action_query_contract_id == DEADLINE_ACTION_CONTRACT_ID
    adapter.commit_collected(first)

    second = adapter.collect_decision(proposal)
    assert calls[-1] == (float(2 * decision_time), 8.0)
    assert len(second["proposals"]) == 1
    assert not second["forced_stay"]


def test_human_stochastic_hold_refreshes_fovea_across_environment_steps():
    adapter = PartialObservationAdapter(
        FakeBackend(),
        decision_time=200,
        fixation_shape="plus",
        gaze_policy=SameTarget(),
        constraint_config=human_stochastic_config(),
        dwell_seed=1,
    )
    calls = []

    def proposal(observation, _critic, _available, timestamp):
        calls.append((timestamp, observation[0, 0, 0]))
        return {"action": 2, "candidate_hidden": np.asarray([timestamp])}

    first = adapter.collect_decision(proposal)
    assert adapter.gaze.last_sampled_dwell_ms > 400.0
    assert first["acquisitions"] == 1
    adapter.commit_collected(first)

    second = adapter.collect_decision(proposal)
    assert second["acquisitions"] == 0
    adapter.commit_collected(second)

    assert calls == [(200.0, 7.0), (400.0, 8.0)]
    assert adapter.current()[0][0, 0, 0] == 9.0
    assert adapter.acquisitions == 1
    assert adapter.gaze_requests == 0
    assert adapter.held_foveal_refreshes == 2
    refreshes = [
        event for event in adapter.events
        if event.get("held_foveal_refresh")
    ]
    assert [event["timestamp_ms"] for event in refreshes] == [200.0, 400.0]
    state = adapter.training_state()
    assert state["held_foveal_refresh_contract_id"] == (
        HELD_FOVEAL_REFRESH_CONTRACT_ID
    )


def test_fixed_emma_does_not_enable_held_foveal_refresh_contract():
    adapter = PartialObservationAdapter(
        FakeBackend(), decision_time=200, fixation_shape="plus",
        gaze_policy=SameTarget(),
    )
    assert adapter.held_foveal_refresh_contract_id is None
    assert "held_foveal_refresh_contract_id" not in adapter.training_state()


def test_reset_static_only_initial_fixation_does_not_leak_self():
    adapter = PartialObservationAdapter(
        FakeBackend(), decision_time=200, fixation_shape="plus", gaze_policy=SameTarget()
    )
    observation, _critic, available = adapter.current()
    assert adapter.gaze.current_fixation == (0, 0)
    assert observation[0, 0, 0] == 0.0
    assert observation[0, 0, 19] == 0.0
    assert observation[0, 0, 20] == 0.0
    assert np.count_nonzero(observation[..., 10:15]) > 0
    assert available.tolist() == [1.0] * 6


def test_full_ground_truth_policy_input_is_fresh_and_gaze_independent():
    backend = FakeBackend(horizon=2)
    adapter = PartialObservationAdapter(
        backend,
        decision_time=200,
        fixation_shape="3x3",
        gaze_policy=SameTarget(),
        ground_truth_policy_input=True,
    )
    first = adapter.policy_observation()
    adapter.memory.state20.fill(0.0)
    adapter.memory.last_observed_time_ms.fill(-np.inf)
    adapter.gaze.current_fixation = (7, 4)
    second = adapter.policy_observation()

    assert np.array_equal(first, second)
    assert np.array_equal(first[..., :20], backend.canonical_pair()[1])
    assert np.all(first[..., 20] == 255.0)
    assert np.count_nonzero(first[..., 21]) == 1
    assert np.count_nonzero(first[..., 22]) == 1
    assert np.count_nonzero(first[..., 23:25]) == 0
    state = adapter.training_state()
    assert state["ground_truth_policy_input"] is True
    assert state["policy_observation_contract_id"] == (
        FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
    )


def test_acquisition_updates_memory_then_queries_actionnet_once_per_acquisition():
    adapter = PartialObservationAdapter(
        FakeBackend(), decision_time=200, fixation_shape="plus", gaze_policy=SameTarget()
    )
    calls = []

    def proposal(observation, _critic, available, timestamp):
        calls.append((timestamp, observation[0, 0, 0], observation[0, 0, 20]))
        assert available.tolist() == [1.0] * 6
        return {"action": len(calls) % 6, "candidate_hidden": np.asarray([len(calls)])}

    collected = adapter.collect_decision(proposal)
    assert calls == [
        (60.0, 7.0, 255.0),
        (120.0, 7.0, 255.0),
        (180.0, 7.0, 255.0),
    ]
    assert len(collected["proposals"]) == adapter.actionnet_online_forwards == 3
    result = adapter.commit_collected(collected)
    assert result["executed_action"] == 3
    assert result["proposal_count"] == 3
    assert adapter.backend.step_actions == [3]
    assert adapter.proposal_revisions == 2
    assert adapter.recurrent_commits == 1


def test_all_within_decision_proposals_can_use_same_predecision_hidden():
    adapter = PartialObservationAdapter(
        FakeBackend(), decision_time=200, fixation_shape="3x3", gaze_policy=SameTarget()
    )
    predecision_hidden = np.asarray([[[4.0, 9.0]]], dtype=np.float32)
    received = []

    def proposal(_observation, _critic, _available, _timestamp):
        received.append(predecision_hidden.copy())
        return {
            "action": len(received),
            "predecision_hidden": predecision_hidden.copy(),
            "candidate_hidden": predecision_hidden + len(received),
        }

    collected = adapter.collect_decision(proposal)
    assert len(received) == 3
    assert all(np.array_equal(value, predecision_hidden) for value in received)
    assert np.array_equal(
        collected["latest_proposal"]["candidate_hidden"], predecision_hidden + 3
    )


def test_no_acquisition_forces_stay_does_not_repeat_previous_proposal():
    adapter = PartialObservationAdapter(
        FakeBackend(), decision_time=200, fixation_shape="plus", gaze_policy=SameThenFar()
    )
    first = adapter.collect_decision(
        lambda *_args: {"action": 5, "candidate_hidden": np.asarray([9])}
    )
    assert first["actor_sample_active"]
    adapter.commit_collected(first)
    second = adapter.collect_decision(
        lambda *_args: {"action": 5, "candidate_hidden": np.asarray([10])}
    )
    assert not second["actor_sample_active"]
    result = adapter.commit_collected(second)
    assert result["forced_stay"]
    assert result["executed_action"] == 4
    assert adapter.backend.step_actions == [5, 4]
    assert adapter.actionnet_online_forwards == 2
    assert adapter.recurrent_commits == 1


def test_deadline_equal_acquisition_is_visible_to_that_decision():
    adapter = PartialObservationAdapter(
        FakeBackend(), decision_time=600, fixation_shape="plus", gaze_policy=SameTarget()
    )
    timestamps = []
    collected = adapter.collect_decision(
        lambda _o, _c, _a, timestamp: timestamps.append(timestamp) or {"action": 4}
    )
    assert 600.0 in timestamps
    assert collected["latest_proposal"]["acquisition_time_ms"] == 600.0


def test_post_deadline_acquisition_cannot_change_previous_decision():
    adapter = PartialObservationAdapter(
        FakeBackend(), decision_time=200, fixation_shape="plus", gaze_policy=SameThenFar()
    )
    first = adapter.collect_decision(lambda *_args: {"action": 2})
    assert [p["action"] for p in first["proposals"]] == [2, 2]
    adapter.commit_collected(first)
    assert adapter.backend.step_actions == [2]
    adapter.advance_to(500, on_acquisition=lambda *_args: {"action": 1})
    assert adapter.backend.step_actions == [2]


def test_preview_mask_and_memory_expire_by_real_time():
    for horizon, expected in {
        0: [0.0, 0.0, 0.0, 0.0],
        2: [17.0, 27.0, 0.0, 0.0],
        4: [17.0, 27.0, 37.0, 47.0],
    }.items():
        adapter = PartialObservationAdapter(
            FakeBackend(horizon),
            decision_time=200,
            fixation_shape="plus",
            gaze_policy=SameTarget(),
        )
        adapter.advance_to(60)
        assert adapter.memory.snapshot()[0, 0, 21:25].tolist() == expected
        adapter.memory.set_time(2460)
        assert adapter.memory.snapshot()[0, 0, 20:25].tolist() == [0.0] * 5


def test_actor_mask_is_all_six_and_checkpoint_contract_records_it():
    adapter = PartialObservationAdapter(
        FakeBackend(), decision_time=400, fixation_shape="2x2", gaze_policy=SameTarget()
    )
    assert adapter.current()[2].tolist() == [1.0] * 6
    assert (
        adapter.training_state()["actor_available_action_contract_id"]
        == ACTOR_AVAILABLE_ACTION_CONTRACT_ID
    )
