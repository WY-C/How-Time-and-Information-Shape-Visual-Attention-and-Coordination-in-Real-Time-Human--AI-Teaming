from __future__ import annotations

from pathlib import Path

import pytest
import torch

from final_model.checkpoint import save_action_checkpoint
from final_model.common.mappo import build_actionnet
from final_model.common.overcooked_backend import cooking_contract
from final_model.eval.checkpoint_loader import _build_critic


@pytest.fixture(scope="session")
def synthetic_checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval_checkpoints")
    actor = build_actionnet(seed=91, device="cpu")
    critic = _build_critic(device=torch.device("cpu"))
    values = {}
    for horizon in (0, 2, 4):
        path = root / f"fixed_h{horizon}.pt"
        save_action_checkpoint(
            path, actor=actor, critic=critic, mode="full", horizon=horizon,
            step=123, seed=91, condition=cooking_contract(),
            training_state={"completed_steps": 123},
        )
        values[f"h{horizon}"] = path
    shared = root / "shared.pt"
    save_action_checkpoint(
        shared, actor=actor, critic=critic, mode="full", horizon=None,
        step=456, seed=91, training_state={"completed_steps": 456},
        condition=cooking_contract(),
        horizon_mode="episode_balanced", horizons=(0, 2, 4),
    )
    values["shared"] = shared
    return values
