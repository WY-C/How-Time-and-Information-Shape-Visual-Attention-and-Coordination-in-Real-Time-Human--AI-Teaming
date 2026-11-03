"""Explicit ActionNet/GazeNet pairing for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from final_model.eval.checkpoint_loader import (
    LoadedPartialCheckpoint,
    load_partial_checkpoint_for_evaluation,
)


PAIR_CONTRACT_KEYS = (
    "observation_contract_id",
    "freshness_contract_id",
    "decision_times",
    "horizons",
    "fixation_shapes",
    "fixed_partner_sha256",
    "action_space",
)


@dataclass(frozen=True)
class BenchmarkCheckpointPair:
    action: LoadedPartialCheckpoint
    gaze: LoadedPartialCheckpoint
    status: str
    differences: dict[str, tuple[Any, Any]]

    @property
    def gaze_actor(self):
        return self.gaze.gaze_actor

    def manifest(self) -> dict[str, Any]:
        return {
            "pairing_status": self.status,
            "pairing_differences": {
                key: {"action": left, "gaze": right}
                for key, (left, right) in self.differences.items()
            },
            "action_checkpoint": {
                "path": str(self.action.path),
                "sha256": self.action.checkpoint_sha256,
                "completed_steps": self.action.completed_steps,
                "actor_sha256": self.action.actor_sha256,
                "metadata": self.action.metadata,
                "training_contract": self.action.training_state.get("contract"),
            },
            "gaze_checkpoint": {
                "path": str(self.gaze.path),
                "sha256": self.gaze.checkpoint_sha256,
                "completed_steps": self.gaze.completed_steps,
                "gaze_sha256": self.gaze.gaze_sha256,
                "metadata": self.gaze.metadata,
                "training_contract": self.gaze.training_state.get("contract"),
            },
        }


def load_benchmark_checkpoint_pair(
    *, action_checkpoint: Path, gaze_checkpoint: Path, device: str = "cpu"
) -> BenchmarkCheckpointPair:
    action = load_partial_checkpoint_for_evaluation(action_checkpoint, device=device)
    gaze = load_partial_checkpoint_for_evaluation(gaze_checkpoint, device=device)
    if gaze.gaze_actor is None:
        raise RuntimeError("gaze_checkpoint does not contain GazeNet weights")
    incompatible = {
        key: (action.metadata.get(key), gaze.metadata.get(key))
        for key in PAIR_CONTRACT_KEYS
        if action.metadata.get(key) != gaze.metadata.get(key)
    }
    if incompatible:
        raise RuntimeError(f"incompatible ActionNet/GazeNet contracts: {incompatible}")
    differences = {}
    for key in ("memory_decay_mode", "memory_lifetime_ms", "constraint_variant_id"):
        if action.metadata.get(key) != gaze.metadata.get(key):
            differences[key] = (action.metadata.get(key), gaze.metadata.get(key))
    status = "CONFIRMED_EXACT_PAIR" if not differences else "PROVISIONAL"
    return BenchmarkCheckpointPair(
        action=action, gaze=gaze, status=status, differences=differences
    )
