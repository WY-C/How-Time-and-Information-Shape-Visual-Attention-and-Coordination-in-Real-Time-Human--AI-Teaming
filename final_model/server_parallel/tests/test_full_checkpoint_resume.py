from pathlib import Path

import pytest
import torch

from final_model.common.mappo import parameter_hash
from final_model.eval.checkpoint_loader import load_action_checkpoint_for_evaluation
from final_model.server_parallel.full_action_training import (
    ServerParallelFullActionTrainingSession,
)


ROOT = Path(__file__).resolve().parents[3]
HSP = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"
PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"


def _first_difference(left, right, path="root"):
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + ".keys", sorted(left), sorted(right)
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path + ".length", len(left), len(right)
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_value, right_value, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    return None if left == right else (path, left, right)


def _options(**overrides):
    result = {
        "observation_mode": "full",
        "horizon": None,
        "shared_horizons": [0, 2, 4],
        "decision_time": None,
        "fixation_shape": None,
        "seed": 20260821,
        "rollout_length": 1,
        "episode_length": 1,
        "n_rollout_threads": 3,
        "device": "cpu",
        "source_checkpoint": None,
        "resume_checkpoint": None,
        "hsp_initial_checkpoint": HSP,
        "fixed_partner_checkpoint": PARTNER,
        "configured_total_steps": 6,
        "checkpoint_interval": 3,
        "reward_shaping_horizon_steps": 60_000_000,
        "partner": None,
        "inference_batching": "predictor",
        "inference_shadow_audit": False,
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coefs": [0.05, 0.03, 0.01],
        "entropy_coef_horizons": [0, 3, 6],
        "ppo_epochs": 1,
        "num_mini_batches": 1,
        "max_grad_norm": 10.0,
    }
    result.update(overrides)
    return result


def test_full_checkpoint_resume_and_canonical_evaluation(tmp_path):
    options = _options()
    uninterrupted = ServerParallelFullActionTrainingSession(
        rollout_workers=2, server_trace=True, **options
    )
    checkpoint = tmp_path / "full_server.pt"
    try:
        uninterrupted.run_update()
        expected = uninterrupted.run_update()
        expected_trace = uninterrupted.last_rollout_trace
    finally:
        uninterrupted.close()

    split = ServerParallelFullActionTrainingSession(
        rollout_workers=2, server_trace=True, **options
    )
    try:
        split.run_update()
        metadata = split.save(checkpoint)
        assert metadata["execution_backend"] == "server_parallel_full_v1"
        assert metadata["rollout_workers"] == 2
        assert metadata["reward_shaping_horizon_steps"] == 60_000_000
    finally:
        split.close()

    resumed = ServerParallelFullActionTrainingSession(
        rollout_workers=2,
        server_trace=True,
        **{
            **options,
            "hsp_initial_checkpoint": None,
            "resume_checkpoint": checkpoint,
        },
    )
    try:
        actual = resumed.run_update()
        difference = _first_difference(resumed.last_rollout_trace, expected_trace)
        assert difference is None, difference
        assert actual["actor_hash_after"] == expected["actor_hash_after"]
        assert actual["environment_steps"] == expected["environment_steps"] == 6
        assert actual["shared_horizon_decision_counts"] == expected[
            "shared_horizon_decision_counts"
        ]
        assert parameter_hash(resumed.backends[0].partners.actor) == (
            resumed.backends[0].partners.frozen_parameter_hash
        )
    finally:
        resumed.close()

    loaded = load_action_checkpoint_for_evaluation(checkpoint, device="cpu")
    assert loaded.metadata["execution_backend"] == "server_parallel_full_v1"
    loaded.assert_unchanged()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training_state"]["server_parallel_state"][
        "rollout_workers"
    ] == 2

    with pytest.raises(RuntimeError, match="rollout_workers"):
        ServerParallelFullActionTrainingSession(
            rollout_workers=1,
            server_trace=False,
            **{
                **options,
                "hsp_initial_checkpoint": None,
                "resume_checkpoint": checkpoint,
            },
        )


@pytest.mark.parametrize("mismatch", ("partner", "predictor", "cooking"))
def test_full_resume_rejects_scientific_contract_mismatch(
    mismatch, tmp_path
):
    options = _options(configured_total_steps=3)
    session = ServerParallelFullActionTrainingSession(
        rollout_workers=1, server_trace=False, **options
    )
    valid = tmp_path / "valid.pt"
    try:
        session.run_update()
        session.save(valid)
    finally:
        session.close()
    payload = torch.load(valid, map_location="cpu", weights_only=False)
    if mismatch == "partner":
        payload["metadata"]["fixed_partner_sha256"] = "0" * 64
    elif mismatch == "predictor":
        payload["metadata"]["inference_batching"] = "partner"
    else:
        payload["training_state"]["contract"][
            "cooking_dynamics_id"
        ] = "legacy_cooking"
    invalid = tmp_path / f"invalid_{mismatch}.pt"
    torch.save(payload, invalid)
    with pytest.raises(RuntimeError):
        ServerParallelFullActionTrainingSession(
            rollout_workers=1,
            server_trace=False,
            **{
                **options,
                "hsp_initial_checkpoint": None,
                "resume_checkpoint": invalid,
            },
        )
