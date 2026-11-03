"""Isolated authoritative GazeNet v1 input and feed-forward architecture."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


BOARD_WIDTH = 8
BOARD_HEIGHT = 5
PARTIAL_MEMORY_CHANNELS = 25
GAZE_SPATIAL_CHANNELS = 26
GAZE_CONTEXT_SIZE = 7
GAZE_ACTION_COUNT = 40
RAW_ENCODING_SCALE = np.float32(255.0)
PRESENCE_CHANNELS = (*range(15), *range(17, 20), *range(21, 25))
POT_ONION_COUNT_CHANNEL = 15
POT_COOK_TIME_CHANNEL = 16
FRESHNESS_CHANNEL = 20
POT_ONION_COUNT_MAX = 3
SOUP_COOKING_TIME = 20
_PRESENCE_RAW_VALUES = np.asarray((0.0, 255.0), dtype=np.float32)
_POT_ONION_COUNT_RAW_VALUES = (
    np.arange(POT_ONION_COUNT_MAX + 1, dtype=np.float32) * RAW_ENCODING_SCALE
)
_POT_COOK_TIME_RAW_VALUES = (
    np.arange(SOUP_COOKING_TIME + 1, dtype=np.float32) * RAW_ENCODING_SCALE
)
TIME_VALUES_MS = (200, 400, 600)
HORIZON_VALUES = (0, 2, 4)
GAZE_TILES = tuple(
    (x, y) for x in range(BOARD_WIDTH) for y in range(BOARD_HEIGHT)
)
GAZENET_V1_ARCHITECTURE_ID = (
    "server_partial_gazenet_8x5x26_global7_ff40_authoritative_v1"
)
GAZENET_V1_INPUT_CONTRACT_ID = (
    "raw_partial25_to_semantic01_continuous_freshness_fixation1_context7_"
    "no_network_rescale_v2"
)
GAZENET_V1_ACTION_CONTRACT_ID = "gaze_tiles_x_major_x5_plus_y_all40_v2"
GAZENET_V1_REWARD_CONTRACT_ID = "task_reward_once_per_env_step_open_gaze_event_v1"
GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID = (
    "forced_initial_fixation_reward_excluded_from_gaze_ppo_v1"
)
GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID = (
    "env_step_boundary_gae_decision_normalized_ppo_v1"
)


def validate_tile(tile: Sequence[int]) -> tuple[int, int]:
    if len(tile) != 2:
        raise ValueError("fixation must contain exactly two coordinates")
    x, y = int(tile[0]), int(tile[1])
    if (
        float(tile[0]) != x
        or float(tile[1]) != y
        or not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT)
    ):
        raise ValueError(f"invalid fixation tile {tuple(tile)}")
    return x, y


def tile_to_action_index(tile: Sequence[int]) -> int:
    x, y = validate_tile(tile)
    return x * BOARD_HEIGHT + y


def validate_action_index(action_index: int) -> int:
    index = int(action_index)
    if float(action_index) != index or not 0 <= index < GAZE_ACTION_COUNT:
        raise ValueError("gaze action index must be an integer in [0, 40)")
    return index


def action_index_to_tile(action_index: int) -> tuple[int, int]:
    index = validate_action_index(action_index)
    return index // BOARD_HEIGHT, index % BOARD_HEIGHT


def gaze_available_actions() -> np.ndarray:
    """Every board tile is a legitimate target, including current fixation."""
    return np.ones(GAZE_ACTION_COUNT, dtype=np.float32)


def _require_values(
    values: np.ndarray,
    allowed: np.ndarray,
    label: str,
) -> None:
    invalid = ~np.isin(values, allowed)
    if np.any(invalid):
        bad_value = float(values[invalid][0])
        raise ValueError(f"{label} contains out-of-contract value {bad_value}")


def normalize_partial_memory(partial_memory: np.ndarray) -> np.ndarray:
    """Return a validated semantic [0,1] copy of canonical Partial memory."""
    memory = np.array(partial_memory, dtype=np.float32, copy=True, order="C")
    if memory.shape != (BOARD_WIDTH, BOARD_HEIGHT, PARTIAL_MEMORY_CHANNELS):
        raise ValueError("Partial memory must have shape (8,5,25)")
    if not np.isfinite(memory).all():
        raise ValueError("Partial memory must contain only finite values")

    _require_values(
        memory[..., PRESENCE_CHANNELS],
        _PRESENCE_RAW_VALUES,
        "presence/orientation/preview channels",
    )
    _require_values(
        memory[..., POT_ONION_COUNT_CHANNEL],
        _POT_ONION_COUNT_RAW_VALUES,
        "pot onion count channel",
    )
    _require_values(
        memory[..., POT_COOK_TIME_CHANNEL],
        _POT_COOK_TIME_RAW_VALUES,
        "pot cook-time channel",
    )
    freshness = memory[..., FRESHNESS_CHANNEL]
    if np.any(freshness < 0.0) or np.any(freshness > RAW_ENCODING_SCALE):
        bad_value = float(
            freshness[(freshness < 0.0) | (freshness > RAW_ENCODING_SCALE)][0]
        )
        raise ValueError(
            f"freshness channel contains out-of-contract value {bad_value}"
        )

    normalized = memory
    normalized[..., PRESENCE_CHANNELS] /= RAW_ENCODING_SCALE
    normalized[..., POT_ONION_COUNT_CHANNEL] /= np.float32(
        POT_ONION_COUNT_MAX
    ) * RAW_ENCODING_SCALE
    normalized[..., POT_COOK_TIME_CHANNEL] /= np.float32(
        SOUP_COOKING_TIME
    ) * RAW_ENCODING_SCALE
    normalized[..., FRESHNESS_CHANNEL] /= RAW_ENCODING_SCALE
    return np.ascontiguousarray(normalized, dtype=np.float32)


def build_spatial_input(
    partial_memory: np.ndarray,
    current_fixation: Sequence[int],
) -> np.ndarray:
    memory = normalize_partial_memory(partial_memory)
    fixation = validate_tile(current_fixation)
    plane = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, 1), dtype=np.float32)
    plane[fixation] = 1.0
    return np.ascontiguousarray(np.concatenate((memory, plane), axis=-1))


def _one_hot(value: int, allowed: tuple[int, ...], label: str) -> tuple[float, ...]:
    selected = int(value)
    if selected not in allowed:
        raise ValueError(f"{label} must be one of {allowed}")
    return tuple(1.0 if selected == candidate else 0.0 for candidate in allowed)


def build_context(
    *,
    remaining_ms: float,
    decision_time_ms: int,
    horizon: int,
) -> np.ndarray:
    remaining = float(remaining_ms)
    if not math.isfinite(remaining):
        raise ValueError("remaining_ms must be finite")
    normalized_remaining = min(max(remaining / 600.0, 0.0), 1.0)
    context = (
        normalized_remaining,
        *_one_hot(decision_time_ms, TIME_VALUES_MS, "decision_time_ms"),
        *_one_hot(horizon, HORIZON_VALUES, "horizon"),
    )
    return np.ascontiguousarray(context, dtype=np.float32)


def build_gaze_observation(
    partial_memory: np.ndarray,
    current_fixation: Sequence[int],
    *,
    remaining_ms: float,
    decision_time_ms: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        build_spatial_input(partial_memory, current_fixation),
        build_context(
            remaining_ms=remaining_ms,
            decision_time_ms=decision_time_ms,
            horizon=horizon,
        ),
    )


def _orthogonal(layer: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class SpatialEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _orthogonal(nn.Conv2d(26, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            _orthogonal(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            _orthogonal(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            _orthogonal(nn.Linear(1280, 64)),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or tuple(spatial.shape[1:]) != (8, 5, 26):
            raise ValueError("spatial gaze input must have shape [batch,8,5,26]")
        return self.layers(spatial.float().permute(0, 3, 1, 2))


class _GazeFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = SpatialEncoder()
        self.context = nn.Sequential(
            _orthogonal(nn.Linear(7, 16)),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            _orthogonal(nn.Linear(80, 64)),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

    def forward(self, spatial: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        spatial_feature = self.spatial(spatial)
        if context.ndim != 2 or tuple(context.shape) != (
            spatial_feature.shape[0],
            GAZE_CONTEXT_SIZE,
        ):
            raise ValueError("global gaze context must have shape [batch,7]")
        context_feature = self.context(context.float())
        return self.fusion(torch.cat((spatial_feature, context_feature), dim=-1))


class GazeNetV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _GazeFusion()
        self.head = _orthogonal(nn.Linear(64, GAZE_ACTION_COUNT), gain=0.01)

    def forward(self, spatial: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(spatial, context))

    def distribution(
        self, spatial: torch.Tensor, context: torch.Tensor
    ) -> Categorical:
        return Categorical(logits=self(spatial, context))


class GazeCriticV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _GazeFusion()
        self.head = _orthogonal(nn.Linear(64, 1), gain=1.0)

    def forward(self, spatial: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(spatial, context))


__all__ = [
    "GAZE_ACTION_COUNT",
    "GAZE_CONTEXT_SIZE",
    "GAZE_SPATIAL_CHANNELS",
    "GAZE_TILES",
    "GAZENET_V1_ACTION_CONTRACT_ID",
    "GAZENET_V1_ARCHITECTURE_ID",
    "GAZENET_V1_CREDIT_ASSIGNMENT_CONTRACT_ID",
    "GAZENET_V1_INPUT_CONTRACT_ID",
    "GAZENET_V1_REWARD_CONTRACT_ID",
    "GAZENET_V1_BOOTSTRAP_REWARD_CONTRACT_ID",
    "GazeCriticV1",
    "GazeNetV1",
    "action_index_to_tile",
    "build_context",
    "build_gaze_observation",
    "build_spatial_input",
    "gaze_available_actions",
    "normalize_partial_memory",
    "tile_to_action_index",
    "validate_action_index",
]
