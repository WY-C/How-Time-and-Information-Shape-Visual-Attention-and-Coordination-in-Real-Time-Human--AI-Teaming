import copy
import hashlib
import inspect
import random

import numpy as np
import pytest
import torch
from torch import nn

from final_model.partial_ver.gaze import GazePhase, GazeStateMachine
from final_model.server_parallel.gaze_v1 import (
    GAZE_TILES,
    GAZENET_V1_ACTION_CONTRACT_ID,
    GazeCriticV1,
    GazeNetV1,
    action_index_to_tile,
    build_context,
    build_gaze_observation,
    build_spatial_input,
    gaze_available_actions,
    normalize_partial_memory,
    tile_to_action_index,
)


def memory():
    raw = np.zeros((8, 5, 25), dtype=np.float32)
    presence_channels = (*range(15), *range(17, 20), *range(21, 25))
    for index, channel in enumerate(presence_channels):
        raw[index % 8, (index // 8) % 5, channel] = 255.0
    raw[0, 1, 15] = 0.0
    raw[1, 1, 15] = 255.0
    raw[2, 1, 15] = 510.0
    raw[3, 1, 15] = 765.0
    raw[0, 2, 16] = 0.0
    raw[1, 2, 16] = 2550.0
    raw[2, 2, 16] = 5100.0
    raw[:7, 3, 20] = np.asarray(
        (0.0, 25.5, 63.75, 127.5, 191.25, 223.125, 255.0),
        dtype=np.float32,
    )
    return raw


def assert_numpy_rng_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_spatial_and_context_shapes_and_fixation_plane():
    raw = memory()
    spatial, context = build_gaze_observation(
        raw, (3, 2), remaining_ms=300, decision_time_ms=400, horizon=2
    )
    assert spatial.shape == (8, 5, 26)
    assert context.shape == (7,)
    assert spatial.dtype == np.float32
    assert np.array_equal(spatial[..., :25], normalize_partial_memory(raw))
    assert 0.0 <= float(spatial.min()) <= float(spatial.max()) <= 1.0
    assert spatial[..., 25].sum() == 1.0
    assert spatial[3, 2, 25] == 1.0
    assert np.count_nonzero(spatial[..., 25]) == 1


def test_normalized_memory_shape_dtype_and_source_immutability():
    raw = memory()
    before = raw.copy()
    before_hash = hashlib.sha256(raw.tobytes()).hexdigest()
    normalized = normalize_partial_memory(raw)
    build_spatial_input(raw, (4, 1))
    assert normalized.shape == (8, 5, 25)
    assert normalized.dtype == np.float32
    assert normalized is not raw
    assert np.array_equal(raw, before)
    assert hashlib.sha256(raw.tobytes()).hexdigest() == before_hash


def test_presence_location_and_orientation_channels_normalize_exactly():
    raw = memory()
    normalized = normalize_partial_memory(raw)
    for channel in (*range(15), *range(17, 20)):
        assert np.array_equal(normalized[..., channel], raw[..., channel] / 255.0)
        assert set(np.unique(normalized[..., channel])).issubset({0.0, 1.0})
    for channel in range(2, 10):
        assert np.array_equal(normalized[..., channel], raw[..., channel] / 255.0)


def test_pot_onion_count_semantic_levels_normalize_exactly():
    normalized = normalize_partial_memory(memory())
    assert normalized[:4, 1, 15].tolist() == pytest.approx(
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    )


def test_pot_cook_time_endpoints_and_midpoint_normalize_exactly():
    normalized = normalize_partial_memory(memory())
    assert normalized[:3, 2, 16].tolist() == [0.0, 0.5, 1.0]


def test_continuous_raw_freshness_normalizes_exactly():
    raw = memory()
    normalized = normalize_partial_memory(raw)
    assert normalized[:7, 3, 20].tolist() == pytest.approx(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.875, 1.0]
    )


def test_preview_channels_normalize_exactly():
    raw = memory()
    normalized = normalize_partial_memory(raw)
    for channel in range(21, 25):
        assert np.array_equal(normalized[..., channel], raw[..., channel] / 255.0)
        assert set(np.unique(normalized[..., channel])).issubset({0.0, 1.0})


@pytest.mark.parametrize(
    ("channel", "bad_value"),
    ((0, 1.0), (15, 254.0), (16, 256.0), (20, -0.01),
     (20, 255.01), (21, -255.0)),
)
def test_normalization_fails_closed_on_out_of_contract_values(channel, bad_value):
    raw = memory()
    raw[7, 4, channel] = bad_value
    with pytest.raises(ValueError, match="out-of-contract"):
        normalize_partial_memory(raw)


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf))
def test_normalization_rejects_non_finite_values(bad_value):
    raw = memory()
    raw[0, 0, 20] = bad_value
    with pytest.raises(ValueError, match="finite"):
        normalize_partial_memory(raw)


def test_normalization_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(8,5,25\)"):
        normalize_partial_memory(np.zeros((8, 5, 24), dtype=np.float32))


@pytest.mark.parametrize(
    ("remaining_ms", "expected"),
    ((600, 1.0), (300, 0.5), (200, 1.0 / 3.0), (0, 0.0)),
)
def test_remaining_time_is_normalized_by_600(remaining_ms, expected):
    context = build_context(
        remaining_ms=remaining_ms, decision_time_ms=200, horizon=0
    )
    assert context[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("decision_time", "expected"),
    ((200, [1, 0, 0]), (400, [0, 1, 0]), (600, [0, 0, 1])),
)
def test_time_one_hot_ordering(decision_time, expected):
    context = build_context(
        remaining_ms=0, decision_time_ms=decision_time, horizon=0
    )
    assert context[1:4].tolist() == expected


@pytest.mark.parametrize(
    ("horizon", "expected"),
    ((0, [1, 0, 0]), (2, [0, 1, 0]), (4, [0, 0, 1])),
)
def test_horizon_one_hot_ordering(horizon, expected):
    context = build_context(
        remaining_ms=0, decision_time_ms=200, horizon=horizon
    )
    assert context[4:7].tolist() == expected


def test_observation_builder_has_no_hidden_ground_truth_dependency():
    signature = inspect.signature(build_gaze_observation)
    assert tuple(signature.parameters) == (
        "partial_memory",
        "current_fixation",
        "remaining_ms",
        "decision_time_ms",
        "horizon",
    )


def test_actor_and_critic_output_shapes_batching_and_finiteness():
    actor = GazeNetV1()
    critic = GazeCriticV1()
    spatial = torch.from_numpy(
        np.stack([build_spatial_input(memory(), (0, 0))] * 3)
    )
    context = torch.from_numpy(
        np.stack([
            build_context(remaining_ms=200, decision_time_ms=200, horizon=0),
            build_context(remaining_ms=300, decision_time_ms=400, horizon=2),
            build_context(remaining_ms=600, decision_time_ms=600, horizon=4),
        ])
    )
    logits = actor(spatial, context)
    values = critic(spatial, context)
    assert logits.shape == (3, 40)
    assert values.shape == (3, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("model_type", (GazeNetV1, GazeCriticV1))
def test_actor_and_critic_receive_normalized_spatial_without_blanket_scaling(
    model_type,
):
    model = model_type()
    spatial = torch.from_numpy(build_spatial_input(memory(), (3, 2))[None])
    context = torch.from_numpy(
        build_context(remaining_ms=300, decision_time_ms=400, horizon=2)[None]
    )
    received = []
    hook = model.encoder.spatial.layers[0].register_forward_pre_hook(
        lambda _module, inputs: received.append(inputs[0].detach().clone())
    )
    try:
        model(spatial, context)
    finally:
        hook.remove()
    assert len(received) == 1
    assert received[0][0, 0, 0, 0] == 1.0
    assert received[0][0, 25, 3, 2] == 1.0
    assert torch.equal(received[0], spatial.permute(0, 3, 1, 2))


def test_architecture_layers_match_authoritative_v1_exactly():
    actor = GazeNetV1()
    critic = GazeCriticV1()
    convolutions = [
        module for module in actor.modules() if isinstance(module, nn.Conv2d)
    ]
    assert [
        (layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding)
        for layer in convolutions
    ] == [
        (26, 32, (3, 3), (1, 1), (1, 1)),
        (32, 64, (3, 3), (1, 1), (1, 1)),
        (64, 32, (3, 3), (1, 1), (1, 1)),
    ]
    assert actor.encoder.spatial.layers[7].in_features == 1280
    assert actor.encoder.spatial.layers[7].out_features == 64
    assert actor.encoder.context[0].in_features == 7
    assert actor.encoder.context[0].out_features == 16
    assert actor.encoder.fusion[0].in_features == 80
    assert actor.encoder.fusion[0].out_features == 64
    assert actor.head.in_features == 64 and actor.head.out_features == 40
    assert critic.head.in_features == 64 and critic.head.out_features == 1


def test_index_tile_mapping_is_exact_for_all_40_actions():
    expected = tuple((x, y) for x in range(8) for y in range(5))
    assert GAZE_TILES == expected
    for index, tile in enumerate(expected):
        assert action_index_to_tile(index) == tile
        assert tile_to_action_index(tile) == index
        assert index == tile[0] * 5 + tile[1]


def test_index_tile_mapping_landmarks_are_x_major():
    assert GAZENET_V1_ACTION_CONTRACT_ID == (
        "gaze_tiles_x_major_x5_plus_y_all40_v2"
    )
    landmarks = {
        0: (0, 0),
        4: (0, 4),
        5: (1, 0),
        9: (1, 4),
        10: (2, 0),
        39: (7, 4),
    }
    for index, tile in landmarks.items():
        assert action_index_to_tile(index) == tile
        assert GAZE_TILES[index] == tile
        assert tile_to_action_index(tile) == index


def test_selecting_current_fixation_is_available():
    available = gaze_available_actions()
    current = (6, 3)
    assert available.shape == (40,)
    assert np.all(available == 1.0)
    assert available[tile_to_action_index(current)] == 1.0


def test_actor_and_critic_have_no_recurrent_modules_or_shared_parameters():
    actor = GazeNetV1()
    critic = GazeCriticV1()
    recurrent_types = (nn.RNNBase, nn.RNN, nn.GRU, nn.LSTM)
    assert not any(isinstance(module, recurrent_types) for module in actor.modules())
    assert not any(isinstance(module, recurrent_types) for module in critic.modules())
    actor_parameters = {id(parameter) for parameter in actor.parameters()}
    critic_parameters = {id(parameter) for parameter in critic.parameters()}
    assert actor_parameters.isdisjoint(critic_parameters)


def test_deterministic_forward_for_identical_weights_and_inputs():
    torch.manual_seed(17)
    first = GazeNetV1()
    second = GazeNetV1()
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    spatial = torch.from_numpy(build_spatial_input(memory(), (2, 1))[None])
    context = torch.from_numpy(
        build_context(remaining_ms=200, decision_time_ms=600, horizon=4)[None]
    )
    assert torch.equal(first(spatial, context), second(spatial, context))


def test_pure_observation_building_does_not_consume_python_numpy_or_torch_rng():
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    build_gaze_observation(
        memory(), (7, 4), remaining_ms=123, decision_time_ms=400, horizon=2
    )
    assert random.getstate() == python_before
    assert_numpy_rng_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_architecture_state_dict_round_trip(tmp_path):
    torch.manual_seed(21)
    actor = GazeNetV1()
    critic = GazeCriticV1()
    path = tmp_path / "gazenet_v1.pt"
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored_actor = GazeNetV1()
    restored_critic = GazeCriticV1()
    restored_actor.load_state_dict(payload["actor"], strict=True)
    restored_critic.load_state_dict(payload["critic"], strict=True)
    for original, restored in zip(actor.parameters(), restored_actor.parameters()):
        assert torch.equal(original, restored)
    for original, restored in zip(critic.parameters(), restored_critic.parameters()):
        assert torch.equal(original, restored)


def test_existing_gaze_operation_remains_committed_across_t200_boundary():
    gaze = GazeStateMachine()
    gaze.reset((0, 0), now_ms=0.0)
    assert gaze.advance_event().next_phase is GazePhase.READY
    gaze.select((7, 4), now_ms=60.0)
    assert gaze.phase is GazePhase.PREP
    prep = gaze.advance_event()
    assert prep.timestamp_ms == 195.0
    assert gaze.phase is GazePhase.EXEC
    assert gaze.next_event_ms > 200.0
    saved = gaze.training_state()
    assert saved["requested_target"] == (7, 4)
    assert saved["phase"] == "EXEC"
