from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from final_model.eval import EvaluationGazeController
from final_model.eval.partial_rollout import EvaluationLearnedGaze
from final_model.partial_ver.training import LearnedGazeController
from final_model.server_parallel.gaze_v1 import action_index_to_tile


class IndexedActor(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = int(index)

    def distribution(self, spatial, _context):
        logits = torch.full(
            (spatial.shape[0], 40),
            -torch.inf,
            dtype=torch.float32,
            device=spatial.device,
        )
        logits[:, self.index] = 0.0
        return Categorical(logits=logits)


class ZeroCritic(nn.Module):
    def forward(self, spatial, _context):
        return torch.zeros(spatial.shape[0], device=spatial.device)


def test_training_and_evaluation_paths_match_canonical_mapping_for_all_actions():
    memory = np.zeros((8, 5, 25), dtype=np.float32)
    context = np.zeros(7, dtype=np.float32)
    device = torch.device("cpu")
    for index in range(40):
        expected = action_index_to_tile(index)

        training = LearnedGazeController(IndexedActor(index), ZeroCritic(), device)
        training.begin_task_decision(0)
        assert training(memory, (0, 0), context) == expected

        legacy_evaluation = EvaluationGazeController(
            IndexedActor(index), ZeroCritic(), device
        )
        assert legacy_evaluation(memory, (0, 0), context) == expected

        partial_evaluation = EvaluationLearnedGaze(
            IndexedActor(index), device, deterministic=True, seed=0
        )
        assert partial_evaluation(memory, (0, 0), context) == expected
