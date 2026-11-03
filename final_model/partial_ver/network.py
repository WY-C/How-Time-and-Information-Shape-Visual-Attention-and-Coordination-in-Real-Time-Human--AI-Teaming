"""Feed-forward GazeNet; ActionNet remains shared in common.mappo."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from final_model.common.mappo import ACTION_ARCHITECTURE_ID, build_actionnet


GAZE_ARCHITECTURE_ID = "partial_gazenet_8x5x26_global7_ff40_v1"


def _orthogonal(layer: nn.Module, gain: float = np.sqrt(2.0)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class SpatialEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _orthogonal(nn.Conv2d(26, 32, 3, padding=1)), nn.ReLU(),
            _orthogonal(nn.Conv2d(32, 64, 3, padding=1)), nn.ReLU(),
            _orthogonal(nn.Conv2d(64, 32, 3, padding=1)), nn.ReLU(),
            nn.Flatten(), _orthogonal(nn.Linear(1280, 64)), nn.ReLU(), nn.LayerNorm(64),
        )

    def forward(self, value):
        if value.ndim != 4 or tuple(value.shape[1:]) != (8, 5, 26):
            raise ValueError("spatial gaze input must be [batch,8,5,26]")
        return self.layers(value.float().permute(0, 3, 1, 2))


class GazeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = SpatialEncoder()
        self.global_mlp = nn.Sequential(_orthogonal(nn.Linear(7, 16)), nn.ReLU())
        self.fusion = nn.Sequential(_orthogonal(nn.Linear(80, 64)), nn.ReLU(), nn.LayerNorm(64))
        self.head = _orthogonal(nn.Linear(64, 40), gain=0.01)

    def forward(self, spatial, context):
        feature = self.spatial(spatial)
        if context.ndim != 2 or tuple(context.shape) != (feature.shape[0], 7):
            raise ValueError("global gaze context must be [batch,7]")
        return self.head(self.fusion(torch.cat((feature, self.global_mlp(context.float())), -1)))

    def distribution(self, spatial, context):
        return Categorical(logits=self(spatial, context))


class GazeCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = SpatialEncoder()
        self.global_mlp = nn.Sequential(_orthogonal(nn.Linear(7, 16)), nn.ReLU())
        self.fusion = nn.Sequential(_orthogonal(nn.Linear(80, 64)), nn.ReLU(), nn.LayerNorm(64))
        self.head = _orthogonal(nn.Linear(64, 1), gain=1.0)

    def forward(self, spatial, context):
        feature = self.spatial(spatial)
        return self.head(self.fusion(torch.cat((feature, self.global_mlp(context.float())), -1))).squeeze(-1)


__all__ = [
    "ACTION_ARCHITECTURE_ID", "GAZE_ARCHITECTURE_ID", "GazeCritic", "GazeNet", "build_actionnet"
]
