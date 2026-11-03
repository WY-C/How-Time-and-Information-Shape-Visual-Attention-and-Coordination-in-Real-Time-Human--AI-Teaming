"""Strict actor-only checkpoint loading shared by training and evaluation."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch


def load_actor_state_dict(path: Path | str) -> OrderedDict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Actor checkpoint must contain a mapping, got {type(payload)!r}"
        )
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Actor checkpoint has no state dict tensors")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("Actor state dict contains non-tensor values")
    return OrderedDict(
        (key, value.detach().cpu()) for key, value in state.items()
    )
