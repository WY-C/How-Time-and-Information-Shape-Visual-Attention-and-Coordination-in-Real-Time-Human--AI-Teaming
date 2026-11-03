"""Small, explicit helpers for exact update-boundary training resume."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


TRAINING_STATE_FORMAT_ID = "final_model_training_resume_v1"


def capture_global_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_global_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise RuntimeError(f"resume RNG state is incomplete: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for values in optimizer.state.values():
        for key, value in values.items():
            if torch.is_tensor(value):
                values[key] = value.to(device)


def require_training_state(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("training_state")
    if not isinstance(state, dict):
        raise RuntimeError(
            "checkpoint has no resumable training_state; use it only for initialization/evaluation"
        )
    if state.get("format_id") != TRAINING_STATE_FORMAT_ID:
        raise RuntimeError(
            "unsupported training_state format: "
            f"{state.get('format_id')!r}"
        )
    return state


def validate_resume_contract(actual: Any, expected: dict[str, Any]) -> None:
    if not isinstance(actual, dict):
        raise RuntimeError("checkpoint has no resume contract")
    mismatch = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"training resume contract mismatch: {mismatch}")
