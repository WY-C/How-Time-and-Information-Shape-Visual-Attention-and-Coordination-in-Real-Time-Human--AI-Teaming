import math

import numpy as np
import pytest
import torch

from final_model.common.mappo import ACTION_ARCHITECTURE_ID, build_actionnet
from final_model.eval import EvaluationGazeController
from final_model.eval.partial_rollout import EvaluationLearnedGaze
from final_model.partial_ver.memory import PersistentPartialMemory
from final_model.partial_ver.network import GazeCritic, GazeNet
from final_model.partial_ver.training import LearnedGazeController
from final_model.server_parallel.gaze_v1 import (
    GAZENET_V1_INPUT_CONTRACT_ID,
    build_spatial_input,
    normalize_partial_memory,
)


TILE = (2, 3)
FIXATION = (4, 1)


def raw_snapshot(memory_lifetime_ms: float, age_ms: float) -> np.ndarray:
    state = np.zeros((8, 5, 20), dtype=np.float32)
    state[TILE][0] = 255.0
    state[TILE][15] = 255.0
    state[TILE][16] = 2550.0
    preview = np.zeros((8, 5, 4), dtype=np.float32)
    preview[TILE][0] = 255.0
    memory = PersistentPartialMemory(memory_lifetime_ms=memory_lifetime_ms)
    memory.reset(state)
    memory.acquire(state, preview, (TILE,), timestamp_ms=0.0)
    return memory.snapshot(age_ms)


def action_conv_input(raw: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    actor = build_actionnet(seed=71, device="cpu")
    received = []
    hook = actor.base.cnn[0].register_forward_pre_hook(
        lambda _module, inputs: received.append(inputs[0].detach().clone())
    )
    try:
        probabilities, _next_rnn_states = actor.get_probs(
            raw[None],
            np.zeros((1, 1, 64), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 6), dtype=np.float32),
        )
    finally:
        hook.remove()
    assert len(received) == 1
    return received[0], probabilities.detach()


@pytest.mark.parametrize(
    ("lifetime", "age", "expected_semantic", "expected_raw"),
    (
        (2400.0, 0.0, 1.0, 255.0),
        (2400.0, 300.0, 0.875, 223.125),
        (1200.0, 300.0, 0.75, 191.25),
        (2400.0, 1200.0, 0.5, 127.5),
    ),
)
def test_required_values_reach_actual_action_and_gaze_conv1(
    lifetime, age, expected_semantic, expected_raw
):
    raw = raw_snapshot(lifetime, age)
    assert raw.dtype == np.float32
    assert raw[TILE][20] == pytest.approx(expected_raw)

    action_received, action_probabilities = action_conv_input(raw)
    assert ACTION_ARCHITECTURE_ID == (
        "canonical_R_Actor_8x5x25_gru64_categorical6_v1"
    )
    assert action_received[0, 0, TILE[0], TILE[1]] == pytest.approx(1.0)
    assert action_received[0, 20, TILE[0], TILE[1]] == pytest.approx(
        expected_semantic
    )
    assert torch.isfinite(action_probabilities).all()
    assert action_probabilities.sum(-1).item() == pytest.approx(1.0)

    spatial = torch.from_numpy(build_spatial_input(raw, FIXATION)[None])
    context = torch.zeros((1, 7), dtype=torch.float32)
    for model in (GazeNet(), GazeCritic()):
        received = []
        hook = model.spatial.layers[0].register_forward_pre_hook(
            lambda _module, inputs: received.append(inputs[0].detach().clone())
        )
        try:
            model(spatial, context)
        finally:
            hook.remove()
        assert len(received) == 1
        conv_input = received[0]
        assert conv_input[0, 0, TILE[0], TILE[1]] == pytest.approx(1.0)
        assert conv_input[0, 20, TILE[0], TILE[1]] == pytest.approx(
            expected_semantic
        )
        assert conv_input[0, 25, FIXATION[0], FIXATION[1]] == pytest.approx(1.0)
        assert torch.equal(conv_input, spatial.permute(0, 3, 1, 2))


@pytest.mark.parametrize("raw_value", (223.125, 191.25, 127.5))
def test_arbitrary_finite_float32_freshness_is_accepted(raw_value):
    raw = np.zeros((8, 5, 25), dtype=np.float32)
    raw[TILE][20] = np.float32(raw_value)
    normalized = normalize_partial_memory(raw)
    assert normalized[TILE][20] == pytest.approx(raw_value / 255.0)


def test_fresh_gaze_actor_initialization_is_finite_and_near_uniform():
    torch.manual_seed(79)
    actor = GazeNet()
    raw = raw_snapshot(2400.0, 300.0)
    spatial = torch.from_numpy(build_spatial_input(raw, FIXATION)[None])
    context = torch.zeros((1, 7), dtype=torch.float32)
    logits = actor(spatial, context)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * probabilities.log()).sum(-1).item()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(probabilities).all()
    assert probabilities.sum(-1).item() == pytest.approx(1.0)
    assert abs(entropy - math.log(40.0)) < 0.1


class CapturingActor(GazeNet):
    def __init__(self):
        super().__init__()
        self.received = None

    def distribution(self, spatial, context):
        self.received = spatial.detach().cpu().numpy()[0].copy()
        return super().distribution(spatial, context)


def test_training_evaluation_and_server_gaze_preprocessing_are_identical():
    assert GAZENET_V1_INPUT_CONTRACT_ID.endswith("no_network_rescale_v2")
    context = np.zeros(7, dtype=np.float32)
    for lifetime, age in ((2400.0, 300.0), (1200.0, 300.0)):
        raw = raw_snapshot(lifetime, age)
        expected = build_spatial_input(raw, FIXATION)

        training_actor = CapturingActor()
        training = LearnedGazeController(
            training_actor, GazeCritic(), torch.device("cpu")
        )
        training.begin_task_decision(0)
        training(raw, FIXATION, context)

        legacy_actor = CapturingActor()
        legacy = EvaluationGazeController(
            legacy_actor, GazeCritic(), torch.device("cpu")
        )
        legacy(raw, FIXATION, context)

        evaluation_actor = CapturingActor()
        evaluation = EvaluationLearnedGaze(
            evaluation_actor,
            torch.device("cpu"),
            deterministic=True,
            seed=0,
        )
        evaluation(raw, FIXATION, context)

        np.testing.assert_array_equal(training_actor.received, expected)
        np.testing.assert_array_equal(legacy_actor.received, expected)
        np.testing.assert_array_equal(evaluation_actor.received, expected)
