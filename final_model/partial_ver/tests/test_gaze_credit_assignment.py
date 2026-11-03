from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from final_model.checkpoint import (
    ACTION_SPACE,
    FORMAT_ID,
    load_gaze_checkpoint,
    state_dict_hash,
)
from final_model.common.mappo import (
    ACTION_ARCHITECTURE_ID,
    PARTIAL_OBSERVATION_CONTRACT,
)
from final_model.common.overcooked_backend import CANONICAL_ENCODER_ID
from final_model.common.partners import PARTNER_SET_ID, parameter_hash
from final_model.common.paths import asset
from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.partial_ver.canonical_training import PartialTrainingSession
from final_model.partial_ver.constraints import (
    HumanLikeConstraintConfig,
    PeripheralVisionConfig,
    PreviewOpacityConfig,
)
from final_model.partial_ver.gaze import DWELL_MS, GazePhase, GazeStateMachine, PREP_MS
from final_model.partial_ver.gaze import HUMAN_STOCHASTIC_EMMA_TIMING
from final_model.partial_ver.network import GAZE_ARCHITECTURE_ID
from final_model.partial_ver.memory import CONTINUOUS_FRESHNESS_CONTRACT_ID
from final_model.partial_ver.training import (
    GAZENET_V1_ACTION_CONTRACT_ID,
    GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID,
    GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID,
    GAZENET_V1_INPUT_CONTRACT_ID,
    GAZENET_V1_REWARD_CONTRACT_ID,
    LearnedGazeController,
    decision_sample_weights,
    gaze_advantages,
    gaze_credit_evaluation_status,
    gaze_ppo_update,
    weighted_advantage_normalization,
    weighted_torch_mean,
)
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.tests.test_runtime import FakeBackend
from final_model.server_parallel.gaze_v1 import gaze_available_actions


def record(
    decision: int,
    sequence: int,
    *,
    reward: float = 0.0,
    value: float = 0.0,
    next_value: float = 0.0,
    boundary: bool = False,
    done: bool = False,
    env: int = 0,
    episode: int = 0,
):
    return {
        "env_index": env,
        "episode_id": episode,
        "task_decision_id": decision,
        "task_decision_key": (env, episode, decision),
        "record_sequence_id": sequence,
        "reward": reward,
        "value": value,
        "next_value": next_value,
        "env_step_boundary": boundary,
        "env_step_boundary_count": int(boundary),
        "done": done,
    }


def legacy_advantages(records, gamma, gae_lambda):
    result = np.zeros(len(records), dtype=np.float32)
    carry = 0.0
    for index in reversed(range(len(records))):
        item = records[index]
        not_done = 0.0 if item["done"] else 1.0
        delta = (
            item["reward"]
            + gamma * item["next_value"] * not_done
            - item["value"]
        )
        carry = delta + gamma * gae_lambda * not_done * carry
        result[index] = carry
    return result


def test_same_decision_subrecords_have_no_internal_discount():
    records = [record(0, index) for index in range(5)]
    records[-1].update(reward=1.0, env_step_boundary=True,
                       env_step_boundary_count=1, done=True)
    advantages, _returns = gaze_advantages(records, gamma=0.99, gae_lambda=0.95)
    np.testing.assert_allclose(advantages, np.ones(5), rtol=0.0, atol=1e-7)


def test_raw_decision_weight_is_record_count_invariant():
    records = [record(0, index) for index in range(2)]
    records += [record(1, index + 2) for index in range(10)]
    raw, _scaled, diagnostics = decision_sample_weights(records)
    assert raw[:2].sum() == pytest.approx(1.0)
    assert raw[2:].sum() == pytest.approx(1.0)
    assert diagnostics["decision_weight_sum_min"] == pytest.approx(1.0)
    assert diagnostics["decision_weight_sum_max"] == pytest.approx(1.0)


def test_scaled_decision_weights_have_global_mean_one():
    records = []
    sequence = 0
    for decision, count in enumerate((1, 2, 5, 10)):
        for _ in range(count):
            records.append(record(decision, sequence))
            sequence += 1
    _raw, scaled, _diagnostics = decision_sample_weights(records)
    assert scaled.mean() == pytest.approx(1.0, abs=1e-12)


def test_gaze_ppo_update_reports_preupdate_value_diagnostics_and_clipping():
    class TinyActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(1, 40)

        def distribution(self, spatial, context):
            return Categorical(logits=self.head(context[:, :1]))

    class TinyCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(1, 1)

        def forward(self, spatial, context):
            return self.head(context[:, :1]).squeeze(-1)

    actor, critic = TinyActor(), TinyCritic()
    records = []
    rewards = (1.0, 2.0, 3.0, 4.0)
    for index, reward in enumerate(rewards):
        item = record(
            index, index, reward=reward, value=reward,
            boundary=True, done=True, episode=index,
        )
        item.update({
            "observation": np.zeros((8, 5, 26), dtype=np.float32),
            "context": np.asarray(
                [float(index + 1), 0, 0, 0, 0, 0, 0], dtype=np.float32
            ),
            "action": index,
        })
        records.append(item)
    with torch.no_grad():
        spatial = torch.as_tensor(
            np.stack([item["observation"] for item in records])
        )
        context = torch.as_tensor(
            np.stack([item["context"] for item in records])
        )
        actions = torch.as_tensor([item["action"] for item in records])
        log_probs = actor.distribution(spatial, context).log_prob(actions)
    for item, log_prob in zip(records, log_probs):
        item["log_prob"] = float(log_prob)

    result = gaze_ppo_update(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        records,
        device=torch.device("cpu"),
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        entropy_coef=0.05,
        ppo_epochs=1,
        num_mini_batches=1,
        max_grad_norm=1e-12,
    )

    assert result["explained_variance"] == pytest.approx(1.0)
    assert result["return_mean"] == pytest.approx(2.5)
    assert result["return_std"] == pytest.approx(np.std(rewards))
    assert result["value_mean"] == pytest.approx(2.5)
    assert result["value_std"] == pytest.approx(np.std(rewards))
    assert result["critic_grad_clip_fraction"] == pytest.approx(1.0)


def test_weighted_advantage_normalization_is_decision_balanced():
    records = [record(0, 0)] + [record(1, index + 1) for index in range(9)]
    raw, _scaled, _diagnostics = decision_sample_weights(records)
    values = np.asarray([0.0] + [10.0] * 9)
    normalized, mean, std = weighted_advantage_normalization(values, raw)
    assert values.mean() == pytest.approx(9.0)
    assert mean == pytest.approx(5.0)
    assert std == pytest.approx(5.0)
    assert np.dot(raw, normalized) / raw.sum() == pytest.approx(0.0, abs=1e-7)


def test_duplicating_identical_subrecord_does_not_change_decision_objective():
    first_records = [record(0, 0), record(1, 1)]
    second_records = [record(0, 0)] + [record(1, index + 1) for index in range(9)]
    first_values = torch.tensor([2.0, 8.0])
    second_values = torch.tensor([2.0] + [8.0] * 9)
    _, first_weights, _ = decision_sample_weights(first_records)
    _, second_weights, _ = decision_sample_weights(second_records)
    first = weighted_torch_mean(first_values, torch.tensor(first_weights))
    second = weighted_torch_mean(second_values, torch.tensor(second_weights))
    assert first.item() == pytest.approx(5.0)
    assert second.item() == pytest.approx(5.0)


def test_one_record_per_decision_matches_legacy_record_level_gae():
    records = [
        record(0, 0, reward=0.2, value=0.4, next_value=0.5, boundary=True),
        record(1, 1, reward=-0.1, value=0.5, next_value=0.3, boundary=True),
        record(2, 2, reward=1.0, value=0.3, boundary=True, done=True),
    ]
    actual, _returns = gaze_advantages(records, gamma=0.99, gae_lambda=0.95)
    expected = legacy_advantages(records, gamma=0.99, gae_lambda=0.95)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    raw, scaled, _diagnostics = decision_sample_weights(records)
    np.testing.assert_array_equal(raw, np.ones(3))
    np.testing.assert_array_equal(scaled, np.ones(3))


def test_zero_reward_environment_step_is_explicit_boundary():
    boundary = [record(0, 0, reward=0.0, value=1.0, next_value=2.0,
                       boundary=True)]
    within = [record(0, 0, reward=0.0, value=1.0, next_value=2.0,
                     boundary=False)]
    boundary_advantage, _ = gaze_advantages(boundary, 0.5, 0.25)
    within_advantage, _ = gaze_advantages(within, 0.5, 0.25)
    assert boundary_advantage[0] == pytest.approx(0.0)
    assert within_advantage[0] == pytest.approx(1.0)
    multi_boundary = record(0, 0, reward=3.0, boundary=True, done=True)
    multi_boundary["env_step_boundary_count"] = 2
    multi_boundary["boundary_optimization_rewards"] = [1.0, 2.0]
    multi_advantage, _ = gaze_advantages([multi_boundary], 0.5, 0.25)
    assert multi_advantage[0] == pytest.approx(2.0)


class FixedActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(40))

    def distribution(self, spatial, context):
        return Categorical(logits=self.logits.expand(spatial.shape[0], -1))


class FixedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self, spatial, context):
        return self.value.expand(spatial.shape[0])


def test_reward_before_first_learned_gaze_is_bootstrap_diagnostic_only():
    controller = LearnedGazeController(
        FixedActor(), FixedCritic(), torch.device("cpu"), env_index=3
    )
    controller.begin_task_decision(0)
    controller.accrue(
        optimization_reward=1.25,
        sparse_team_reward=1.0,
        native_shaped_reward=0.5,
        weighted_shaped_reward=0.25,
        deliveries=1,
        task_decision_id=0,
    )

    assert controller.active is None
    assert controller.drain() == []
    assert controller.next_record_sequence_id == 0
    assert controller.total_attributed_optimization_reward == 0.0
    assert controller.total_env_boundaries == 0
    assert controller.total_bootstrap_unowned_reward_count == 1
    assert controller.total_bootstrap_unowned_reward_sum == pytest.approx(1.25)
    assert GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID == (
        "forced_initial_fixation_reward_excluded_from_gaze_ppo_v1"
    )

    saved = controller.training_state()
    restored = LearnedGazeController(
        FixedActor(), FixedCritic(), torch.device("cpu"), env_index=3
    )
    restored.load_training_state(saved)
    assert restored.total_bootstrap_unowned_reward_count == 1
    assert restored.total_bootstrap_unowned_reward_sum == pytest.approx(1.25)

    restored.begin_task_decision(1)
    restored(np.zeros((8, 5, 25), np.float32), (0, 0), np.zeros(7, np.float32))
    restored.accrue(
        optimization_reward=2.0,
        sparse_team_reward=2.0,
        native_shaped_reward=0.0,
        weighted_shaped_reward=0.0,
        deliveries=0,
        task_decision_id=1,
    )
    restored.finish_episode(200.0)
    records = restored.drain()
    assert len(records) == 1
    assert records[0]["boundary_optimization_rewards"] == [2.0]
    assert restored.total_attributed_optimization_reward == pytest.approx(2.0)
    assert restored.total_bootstrap_unowned_reward_sum == pytest.approx(1.25)


def test_initial_human_fixation_crosses_deadline_without_gaze_reward_owner():
    class SemanticFakeBackend(FakeBackend):
        def canonical_pair(self):
            partner = np.zeros((8, 5, 20), dtype=np.float32)
            learner = np.zeros_like(partner)
            learner[0, 0, 0] = 255.0
            learner[0, 0, 19] = 255.0
            return partner, learner

        def preview_planes(self):
            return np.zeros((4, 8, 5), dtype=np.float32)

    controller = LearnedGazeController(
        FixedActor(), FixedCritic(), torch.device("cpu"), env_index=0
    )
    constraints = HumanLikeConstraintConfig(
        profile="custom",
        peripheral=PeripheralVisionConfig("none"),
        emma=HUMAN_STOCHASTIC_EMMA_TIMING,
        preview_opacity=PreviewOpacityConfig("off"),
    )
    adapter = PartialObservationAdapter(
        SemanticFakeBackend(),
        decision_time=200,
        fixation_shape="plus",
        gaze_policy=controller,
        constraint_config=constraints,
        dwell_seed=1,
    )

    controller.begin_task_decision(0)
    collected = adapter.collect_decision(
        lambda *_args: {"action": 4, "candidate_hidden": np.zeros(1)}
    )
    assert adapter.gaze.last_sampled_dwell_ms > 200.0
    assert adapter.gaze_requests == 0
    adapter.commit_collected(collected)
    controller.accrue(
        optimization_reward=1.5,
        sparse_team_reward=1.0,
        native_shaped_reward=1.0,
        weighted_shaped_reward=0.5,
        deliveries=0,
        task_decision_id=0,
    )
    assert controller.total_bootstrap_unowned_reward_count == 1
    assert controller.drain() == []

    for decision_id in range(1, 10):
        controller.begin_task_decision(decision_id)
        collected = adapter.collect_decision(
            lambda *_args: {"action": 4, "candidate_hidden": np.zeros(1)}
        )
        adapter.commit_collected(collected)
        controller.accrue(
            optimization_reward=0.0,
            sparse_team_reward=0.0,
            native_shaped_reward=0.0,
            weighted_shaped_reward=0.0,
            deliveries=0,
            task_decision_id=decision_id,
        )
        if adapter.gaze_requests:
            break

    assert adapter.gaze_requests > 0
    assert controller.active is not None
    assert controller.active["env_step_boundary_count"] == 1
    assert controller.total_attributed_optimization_reward == 0.0

def test_open_event_reward_is_owned_once_and_conserved():
    controller = LearnedGazeController(
        FixedActor(), FixedCritic(), torch.device("cpu"), env_index=3
    )
    controller.begin_task_decision(7)
    controller(np.zeros((8, 5, 25), np.float32), (0, 0), np.zeros(7, np.float32))
    controller.set_event_time(60.0)
    controller(np.zeros((8, 5, 25), np.float32), (0, 0), np.zeros(7, np.float32))
    assert controller.drain() == []
    controller.accrue(
        optimization_reward=2.5,
        sparse_team_reward=2.0,
        native_shaped_reward=1.0,
        weighted_shaped_reward=0.5,
        deliveries=1,
        task_decision_id=7,
    )
    controller.begin_task_decision(8)
    controller.set_event_time(200.0)
    controller(np.zeros((8, 5, 25), np.float32), (0, 0), np.zeros(7, np.float32))
    records = controller.drain()
    assert sum(item["reward"] for item in records) == pytest.approx(2.5)
    assert len(records) == 2
    assert records[0]["boundary_optimization_rewards"] == []
    assert records[1]["boundary_optimization_rewards"] == [2.5]
    assert controller.total_attributed_optimization_reward == pytest.approx(2.5)
    assert controller.total_env_boundaries == 1
    assert GAZENET_V1_REWARD_CONTRACT_ID == (
        "task_reward_once_per_env_step_open_gaze_event_v1"
    )


def test_terminal_resets_carry_across_episodes_and_interleaved_environments():
    records = [
        record(0, 0, reward=1.0, boundary=True, done=True, env=0, episode=0),
        record(0, 0, reward=10.0, boundary=True, done=True, env=1, episode=0),
        record(0, 1, reward=3.0, boundary=True, done=True, env=0, episode=1),
    ]
    advantages, _returns = gaze_advantages(records, 0.99, 0.95)
    np.testing.assert_allclose(advantages, [1.0, 10.0, 3.0])


def test_same_target_remains_legal_and_all_40_actions_available():
    available = gaze_available_actions()
    assert available.shape == (40,)
    np.testing.assert_array_equal(available, np.ones(40, np.float32))
    assert available[0] == 1.0


def test_observation_action_and_emma_timing_contracts_are_unchanged():
    controller = LearnedGazeController(FixedActor(), FixedCritic(), torch.device("cpu"))
    spatial = controller.spatial_input(
        np.zeros((8, 5, 25), np.float32), (2, 3)
    )
    assert spatial.shape == (8, 5, 26)
    assert spatial[2, 3, -1] == 1.0
    assert FixedActor().distribution(
        torch.zeros((1, 8, 5, 26)), torch.zeros((1, 7))
    ).probs.shape == (1, 40)
    gaze = GazeStateMachine()
    gaze.reset((1, 1), now_ms=0.0)
    assert gaze.next_event_ms == DWELL_MS
    gaze.advance_event()
    gaze.select((1, 1), DWELL_MS)
    assert gaze.phase is GazePhase.DWELL
    assert gaze.next_event_ms == 2 * DWELL_MS
    gaze.advance_event()
    gaze.select((2, 1), 2 * DWELL_MS)
    assert gaze.phase is GazePhase.PREP
    assert gaze.next_event_ms == 2 * DWELL_MS + PREP_MS


def test_action_gae_contract_is_not_replaced_by_gaze_credit_contract():
    from final_model.common import mappo

    assert mappo.ppo_health_metrics is not None
    assert not hasattr(mappo, "GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID")


def test_legacy_gaze_checkpoint_remains_evaluation_loadable(tmp_path):
    source_action, target_action = nn.Linear(3, 2), nn.Linear(3, 2)
    source_gaze, target_gaze = nn.Linear(4, 3), nn.Linear(4, 3)
    payload = {
        "metadata": {
            "format_id": FORMAT_ID,
            "training_stage": "gaze",
            "observation_mode": "partial",
            "action_architecture_id": ACTION_ARCHITECTURE_ID,
            "gaze_architecture_id": GAZE_ARCHITECTURE_ID,
            "observation_contract_id": PARTIAL_OBSERVATION_CONTRACT,
            "canonical_encoder_id": CANONICAL_ENCODER_ID,
            "partner_set_id": PARTNER_SET_ID,
            "actionnet_frozen": True,
            "action_optimizer_steps": 0,
            "gaze_action_contract_id": "gaze_tiles_row_major_y8_plus_x_all40_v1",
            "action_parameter_sha256": parameter_hash(source_action),
            "gaze_parameter_sha256": state_dict_hash(source_gaze.state_dict()),
            "action_space": list(ACTION_SPACE),
        },
        "action_actor": source_action.state_dict(),
        "gaze_actor": source_gaze.state_dict(),
    }
    path = tmp_path / "legacy_gaze.pt"
    torch.save(payload, path)
    metadata = load_gaze_checkpoint(
        path, action_actor=target_action, gaze_actor=target_gaze
    )
    assert metadata["evaluation_gaze_action_contract_status"] == (
        "legacy_or_unknown_metadata_runtime_x_major"
    )
    assert metadata["legacy_gaze_action_contract_metadata"] == (
        "gaze_tiles_row_major_y8_plus_x_all40_v1"
    )
    assert metadata["evaluation_gaze_credit_assignment_status"] == "legacy_or_unknown"
    assert metadata["evaluation_gaze_input_contract_status"] == (
        "legacy_or_unknown_input_scaling"
    )
    assert gaze_credit_evaluation_status(metadata) == "legacy_or_unknown"


def test_legacy_or_missing_credit_contract_fails_actual_training_resume_closed(
    tmp_path,
):
    common = dict(
        decision_times=[200],
        horizons=[0],
        fixation_shapes=["plus"],
        seed=419,
        rollout_length=1,
        episode_length=1,
        n_rollout_threads=1,
        device="cpu",
        configured_total_steps=1,
        fixed_partner_checkpoint=asset(
            "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
        ),
        checkpoint_interval=1,
        reward_shaping_horizon_steps=60_000_000,
        ppo_epochs=1,
        num_mini_batches=1,
    )
    action = PartialTrainingSession(
        gaze_type="rule_based",
        train_action=True,
        train_gaze=False,
        source_checkpoint=None,
        **common,
    )
    action.run_update()
    action_path = tmp_path / "partial_action.pt"
    action.save(action_path)
    gaze = PartialTrainingSession(
        gaze_type="learned",
        train_action=False,
        train_gaze=True,
        source_checkpoint=action_path,
        **common,
    )
    gaze.run_update()
    current_path = tmp_path / "current_gaze.pt"
    current_metadata = gaze.save(current_path)
    assert current_metadata["gaze_action_contract_id"] == (
        GAZENET_V1_ACTION_CONTRACT_ID
    )
    assert current_metadata["gaze_input_contract_id"] == (
        GAZENET_V1_INPUT_CONTRACT_ID
    )
    assert current_metadata["observation_contract_id"] == (
        PARTIAL_OBSERVATION_CONTRACT
    )
    assert current_metadata["memory_lifetime_ms"] == 2400.0
    assert current_metadata["freshness_contract_id"] == (
        CONTINUOUS_FRESHNESS_CONTRACT_ID
    )
    assert current_metadata["gaze_credit_assignment_contract_id"] == (
        GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID
    )
    assert current_metadata["gaze_bootstrap_reward_contract_id"] == (
        GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID
    )
    payload = torch.load(current_path, map_location="cpu", weights_only=False)
    assert payload["training_state"]["contract"]["gaze_action_contract_id"] == (
        GAZENET_V1_ACTION_CONTRACT_ID
    )
    assert payload["training_state"]["contract"][
        "gaze_bootstrap_reward_contract_id"
    ] == GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID
    current_evaluated = load_partial_checkpoint_for_evaluation(current_path)
    assert current_evaluated.metadata[
        "evaluation_gaze_action_contract_status"
    ] == "current"
    assert current_evaluated.metadata[
        "evaluation_gaze_input_contract_status"
    ] == "current"

    input_legacy_payload = torch.load(
        current_path, map_location="cpu", weights_only=False
    )
    input_legacy_payload["metadata"].pop("gaze_input_contract_id")
    input_legacy_payload["training_state"]["contract"].pop(
        "gaze_input_contract_id"
    )
    input_legacy_path = tmp_path / "missing_input_contract_gaze.pt"
    torch.save(input_legacy_payload, input_legacy_path)
    input_evaluated = load_partial_checkpoint_for_evaluation(input_legacy_path)
    assert input_evaluated.metadata[
        "evaluation_gaze_input_contract_status"
    ] == "legacy_or_unknown_input_scaling"
    with pytest.raises(RuntimeError, match="Gaze input contract"):
        PartialTrainingSession(
            gaze_type="learned",
            train_action=False,
            train_gaze=True,
            source_checkpoint=None,
            resume_checkpoint=input_legacy_path,
            **common,
        )

    old_action_contract = "gaze_tiles_row_major_y8_plus_x_all40_v1"
    for label, legacy_contract in (("missing", None), ("old", old_action_contract)):
        action_legacy_payload = torch.load(
            current_path, map_location="cpu", weights_only=False
        )
        if legacy_contract is None:
            action_legacy_payload["metadata"].pop("gaze_action_contract_id")
            action_legacy_payload["training_state"]["contract"].pop(
                "gaze_action_contract_id"
            )
        else:
            action_legacy_payload["metadata"][
                "gaze_action_contract_id"
            ] = legacy_contract
            action_legacy_payload["training_state"]["contract"][
                "gaze_action_contract_id"
            ] = legacy_contract
        action_legacy_path = tmp_path / f"{label}_action_contract_gaze.pt"
        torch.save(action_legacy_payload, action_legacy_path)
        action_evaluated = load_partial_checkpoint_for_evaluation(action_legacy_path)
        assert action_evaluated.metadata[
            "evaluation_gaze_action_contract_status"
        ] == "legacy_or_unknown_metadata_runtime_x_major"
        assert action_evaluated.metadata[
            "legacy_gaze_action_contract_metadata"
        ] == legacy_contract
        with pytest.raises(RuntimeError, match="action-mapping contract"):
            PartialTrainingSession(
                gaze_type="learned",
                train_action=False,
                train_gaze=True,
                source_checkpoint=None,
                resume_checkpoint=action_legacy_path,
                **common,
            )

    payload["metadata"].pop("gaze_credit_assignment_contract_id")
    payload["training_state"]["contract"].pop(
        "gaze_credit_assignment_contract_id"
    )
    legacy_path = tmp_path / "legacy_gaze.pt"
    torch.save(payload, legacy_path)

    evaluated = load_partial_checkpoint_for_evaluation(legacy_path)
    assert evaluated.metadata[
        "evaluation_gaze_credit_assignment_status"
    ] == "legacy_or_unknown"
    with pytest.raises(RuntimeError, match="credit-assignment contract"):
        PartialTrainingSession(
            gaze_type="learned",
            train_action=False,
            train_gaze=True,
            source_checkpoint=None,
            resume_checkpoint=legacy_path,
            **common,
        )
