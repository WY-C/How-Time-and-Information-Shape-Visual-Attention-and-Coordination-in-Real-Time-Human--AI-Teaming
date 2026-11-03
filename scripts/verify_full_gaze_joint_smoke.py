#!/usr/bin/env python3
"""Fail-closed verification for Full-Gaze and diamond staged-joint smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from final_model.checkpoint import state_dict_hash
from final_model.partial_ver.observation import (
    FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID,
)
from final_model.parallel_gaze_contract import (
    PARALLEL_GAZE_TRAINING_CONTRACT,
    PARALLEL_STAGED_JOINT_TRAINING_CONTRACT,
)
from final_model.server_parallel.parallel_action_training import (
    PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID,
    SOURCE_VALIDATION_CONTRACT_ID,
)


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict) and isinstance(payload.get("metadata"), dict)
    assert isinstance(payload.get("training_state"), dict)
    return payload


def _verified_hash(payload: dict, key: str, metadata_key: str) -> str:
    parameters = payload.get(key)
    assert isinstance(parameters, dict) and parameters, key
    actual = state_dict_hash(parameters)
    assert payload["metadata"].get(metadata_key) == actual, metadata_key
    return actual


def _optimizer_state(payload: dict, key: str, *, present: bool) -> None:
    value = payload["training_state"].get(key)
    assert isinstance(value, dict) if present else value is None, key


def verify_full(source: dict, initial: dict, trained: dict, resumed: dict) -> None:
    source_actor = _verified_hash(source, "actor", "actor_parameter_sha256")
    source_critic = _verified_hash(source, "critic", "critic_parameter_sha256")
    initial_gaze = _verified_hash(initial, "gaze_actor", "gaze_parameter_sha256")
    trained_gaze = _verified_hash(trained, "gaze_actor", "gaze_parameter_sha256")
    resumed_gaze = _verified_hash(resumed, "gaze_actor", "gaze_parameter_sha256")
    for payload in (initial, trained, resumed):
        metadata = payload["metadata"]
        assert metadata["training_stage"] == "gaze"
        assert metadata["ground_truth_policy_input"] is True
        assert metadata["policy_observation_contract_id"] == (
            FULL_GROUND_TRUTH_POLICY_INPUT_CONTRACT_ID
        )
        assert metadata["execution_backend"] == PARALLEL_GAZE_TRAINING_CONTRACT
        assert metadata["training_contract"] == PARALLEL_GAZE_TRAINING_CONTRACT
        assert _verified_hash(payload, "actor", "actor_parameter_sha256") == source_actor
        assert _verified_hash(payload, "critic", "critic_parameter_sha256") == source_critic
        _optimizer_state(payload, "action_actor_optimizer", present=False)
        _optimizer_state(payload, "action_critic_optimizer", present=False)
        _optimizer_state(payload, "gaze_actor_optimizer", present=True)
        _optimizer_state(payload, "gaze_critic_optimizer", present=True)
    provenance = resumed["metadata"]["source_checkpoint_provenance"]
    assert provenance["source_validation_contract_id"] == SOURCE_VALIDATION_CONTRACT_ID
    assert initial_gaze != trained_gaze != resumed_gaze


def verify_diamond(source: dict, initial: dict, trained: dict, resumed: dict) -> None:
    source_actor = _verified_hash(source, "actor", "actor_parameter_sha256")
    source_critic = _verified_hash(source, "critic", "critic_parameter_sha256")
    source_gaze = _verified_hash(source, "gaze_actor", "gaze_parameter_sha256")
    source_gaze_critic = _verified_hash(
        source, "gaze_critic", "gaze_critic_parameter_sha256"
    )
    initial_actor = _verified_hash(initial, "actor", "actor_parameter_sha256")
    initial_critic = _verified_hash(initial, "critic", "critic_parameter_sha256")
    initial_gaze = _verified_hash(initial, "gaze_actor", "gaze_parameter_sha256")
    initial_gaze_critic = _verified_hash(
        initial, "gaze_critic", "gaze_critic_parameter_sha256"
    )
    trained_actor = _verified_hash(trained, "actor", "actor_parameter_sha256")
    trained_gaze = _verified_hash(trained, "gaze_actor", "gaze_parameter_sha256")
    resumed_actor = _verified_hash(resumed, "actor", "actor_parameter_sha256")
    resumed_gaze = _verified_hash(resumed, "gaze_actor", "gaze_parameter_sha256")
    assert initial_actor == source_actor
    assert initial_critic != source_critic
    assert initial_gaze == source_gaze
    assert initial_gaze_critic == source_gaze_critic
    assert initial_actor != trained_actor != resumed_actor
    assert initial_gaze != trained_gaze != resumed_gaze
    for payload in (initial, trained, resumed):
        metadata = payload["metadata"]
        assert metadata["training_stage"] == "joint"
        assert metadata["peripheral_extent"] == "diamond13"
        assert metadata["execution_backend"] == (
            PARALLEL_STAGED_JOINT_TRAINING_CONTRACT
        )
        assert metadata["training_contract"] == (
            PARALLEL_STAGED_JOINT_TRAINING_CONTRACT
        )
        for key in (
            "action_actor_optimizer",
            "action_critic_optimizer",
            "gaze_actor_optimizer",
            "gaze_critic_optimizer",
        ):
            _optimizer_state(payload, key, present=True)
    provenance = resumed["metadata"]["source_checkpoint_provenance"]
    assert provenance["source_validation_contract_id"] == (
        PARTIAL_GAZE_SOURCE_VALIDATION_CONTRACT_ID
    )
    assert provenance["initialization"] == "staged_joint_from_partial_gaze"
    assert provenance["source_checkpoint_step"] == 13_000_000
    assert provenance["source_constraint_config"]["peripheral_extent"] == "diamond13"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("full", "diamond"))
    parser.add_argument("source", type=Path)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    checkpoints = args.run_dir / "checkpoints"
    source = _load(args.source)
    initial = _load(checkpoints / "step_000000000.pt")
    trained = _load(checkpoints / "step_000000002.pt")
    resumed = _load(checkpoints / "step_000000004.pt")
    assert trained["metadata"]["step"] == 2
    assert resumed["metadata"]["step"] == 4
    if args.mode == "full":
        verify_full(source, initial, trained, resumed)
    else:
        verify_diamond(source, initial, trained, resumed)
    summary = json.loads((args.run_dir / "logs/summary.json").read_text())
    assert summary["environment_steps"] == 4
    assert summary["resume_start_step"] == 2
    latest = summary["latest"]
    if args.mode == "full":
        assert latest["freshness_normalized_min"] == 1.0
        assert latest["freshness_normalized_max"] == 1.0
        assert latest["valid_memory_fraction"] == 1.0
        assert latest["actor_hash_before"] == latest["actor_hash_after"]
        assert latest["critic_hash_before"] == latest["critic_hash_after"]
        assert latest["gaze_hash_before"] != latest["gaze_hash_after"]
    else:
        assert latest["actor_hash_before"] != latest["actor_hash_after"]
        assert latest["gaze_hash_before"] != latest["gaze_hash_after"]
    print(f"[PASS] {args.mode} smoke, update, save, and resume contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
