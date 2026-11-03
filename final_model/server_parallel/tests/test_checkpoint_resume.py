import copy

import numpy as np
import pytest
import torch

from final_model.common.paths import file_sha256
from final_model.common.partners import parameter_hash
from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.partial_ver.conditions import PartialCondition
from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
)


def assert_nested_exact(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_nested_exact(actual_item, expected_item)
    elif torch.is_tensor(expected):
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    else:
        assert actual == expected


def test_checkpoint_resume_and_canonical_evaluation_compatibility(
    session_options, tmp_path
):
    options = session_options(configured_total_steps=4)
    uninterrupted = ServerParallelActionTrainingSession(
        rollout_workers=2, server_trace=True, **options
    )
    split = ServerParallelActionTrainingSession(
        rollout_workers=2, server_trace=True, **options
    )
    checkpoint = tmp_path / "server_parallel.pt"
    try:
        uninterrupted.run_update()
        expected = uninterrupted.run_update()
        expected_trace = uninterrupted.last_rollout_trace
        split.run_update()
        metadata = split.save(checkpoint)
        assert metadata["execution_backend"] == "server_parallel_v1"
        assert metadata["rollout_workers"] == 2
    finally:
        uninterrupted.close()
        split.close()

    resumed = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=True,
        **{
            **options,
            "source_checkpoint": None,
            "resume_checkpoint": checkpoint,
        },
    )
    try:
        actual = resumed.run_update()
        assert resumed.last_rollout_trace == expected_trace
        assert actual["actor_hash_after"] == expected["actor_hash_after"]
        assert actual["environment_steps"] == expected["environment_steps"] == 4
        assert parameter_hash(resumed._partner_pools[0].actor) == resumed._partner_parameter_sha256
    finally:
        resumed.close()

    loaded = load_partial_checkpoint_for_evaluation(checkpoint, device="cpu")
    assert loaded.metadata["execution_backend"] == "server_parallel_v1"
    loaded.assert_unchanged()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training_state"]["server_parallel"]["rollout_workers"] == 2


def test_resume_allows_only_operational_target_and_interval_changes(
    session_options, tmp_path
):
    options = session_options(
        configured_total_steps=2,
        checkpoint_interval=2,
        horizons=[0, 2, 4],
    )
    session = ServerParallelActionTrainingSession(
        rollout_workers=2, server_trace=False, **options
    )
    checkpoint = tmp_path / "step_000000002.pt"
    try:
        session.run_update()
        session.save(checkpoint)
    finally:
        session.close()
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original_state = copy.deepcopy(original["training_state"])

    resumed = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=False,
        **{
            **options,
            "configured_total_steps": 4,
            "checkpoint_interval": 4,
            "source_checkpoint": None,
            "resume_checkpoint": checkpoint,
        },
    )
    continued_checkpoint = tmp_path / "continued.pt"
    try:
        assert resumed.completed_steps == 2
        assert resumed.update_count == 1
        assert resumed.configured_total_steps == 4
        assert resumed.checkpoint_interval == 4
        assert parameter_hash(resumed.policy.actor) == original["metadata"][
            "actor_parameter_sha256"
        ]
        assert parameter_hash(resumed.policy.critic) == original["metadata"][
            "critic_parameter_sha256"
        ]
        assert_nested_exact(
            resumed.policy.actor_optimizer.state_dict(),
            original_state["action_actor_optimizer"],
        )
        assert_nested_exact(
            resumed.policy.critic_optimizer.state_dict(),
            original_state["action_critic_optimizer"],
        )
        assert_nested_exact(
            resumed.scheduler.state_dict(), original_state["scheduler"]
        )
        resumed.save(continued_checkpoint)
    finally:
        resumed.close()

    continued = torch.load(
        continued_checkpoint, map_location="cpu", weights_only=False
    )
    provenance = continued["metadata"][
        "resume_partial_checkpoint_provenance"
    ]
    assert provenance["resume_checkpoint_absolute_path"] == str(
        checkpoint.resolve()
    )
    assert provenance["resume_checkpoint_content_sha256"] == file_sha256(checkpoint)
    assert provenance["resume_start_step"] == 2
    assert continued["training_state"]["server_parallel"][
        "resume_partial_checkpoint_provenance"
    ] == provenance
    assert continued["training_state"]["contract"][
        "configured_total_steps"
    ] == 4
    assert continued["training_state"]["contract"]["checkpoint_interval"] == 4
    assert continued["metadata"]["source_full_checkpoint_provenance"] == original[
        "metadata"
    ]["source_full_checkpoint_provenance"]


def test_resume_restores_saved_h4_before_backend_state_load(
    session_options, tmp_path
):
    options = session_options(
        configured_total_steps=2,
        checkpoint_interval=1,
        horizons=[0, 4],
        n_rollout_threads=1,
    )
    session = ServerParallelActionTrainingSession(
        rollout_workers=1, server_trace=False, **options
    )
    checkpoint = tmp_path / "saved_h4.pt"
    try:
        assert session.conditions == [PartialCondition(200, 0, "3x3")]
        session.run_update()
        assert session.conditions == [PartialCondition(200, 4, "3x3")]
        session.save(checkpoint)
    finally:
        session.close()

    resumed = ServerParallelActionTrainingSession(
        rollout_workers=1,
        server_trace=False,
        **{
            **options,
            "source_checkpoint": None,
            "resume_checkpoint": checkpoint,
        },
    )
    try:
        assert resumed.conditions == [PartialCondition(200, 4, "3x3")]
        backend_states, adapter_states = resumed._snapshot_workers()
        assert backend_states[0]["horizon"] == 4
        assert adapter_states[0]["decision_time"] == 200
        assert adapter_states[0]["fixation_shape"] == "3x3"
    finally:
        resumed.close()


def test_resume_restores_mixed_nine_cell_env_mapping_and_keeps_backend_guard(
    session_options, tmp_path
):
    options = session_options(
        configured_total_steps=18,
        checkpoint_interval=9,
        decision_times=[200, 400, 600],
        horizons=[0, 2, 4],
        fixation_shapes=["plus"],
        n_rollout_threads=9,
    )
    session = ServerParallelActionTrainingSession(
        rollout_workers=2, server_trace=False, **options
    )
    checkpoint = tmp_path / "mixed_nine_cell.pt"
    try:
        fresh_conditions = list(session.conditions)
        session.run_update()
        saved_conditions = list(session.conditions)
        assert saved_conditions != fresh_conditions
        session.save(checkpoint)
    finally:
        session.close()

    resumed = ServerParallelActionTrainingSession(
        rollout_workers=2,
        server_trace=False,
        **{
            **options,
            "source_checkpoint": None,
            "resume_checkpoint": checkpoint,
        },
    )
    try:
        assert resumed.conditions == saved_conditions
        backend_states, adapter_states = resumed._snapshot_workers()
        for env_id, condition in enumerate(saved_conditions):
            assert backend_states[env_id]["horizon"] == condition.horizon
            assert adapter_states[env_id]["decision_time"] == (
                condition.decision_time
            )
            assert adapter_states[env_id]["fixation_shape"] == (
                condition.fixation_shape
            )
    finally:
        resumed.close()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["training_state"]["backends"][0]["horizon"] = (
        4 if saved_conditions[0].horizon != 4 else 0
    )
    invalid = tmp_path / "mixed_nine_cell_backend_mismatch.pt"
    torch.save(payload, invalid)
    with pytest.raises(RuntimeError, match="backend resume contract mismatch"):
        ServerParallelActionTrainingSession(
            rollout_workers=2,
            server_trace=False,
            **{
                **options,
                "source_checkpoint": None,
                "resume_checkpoint": invalid,
            },
        )


@pytest.mark.parametrize("mismatch", ("source", "partner", "cooking"))
def test_resume_rejects_scientific_contract_mismatch(
    mismatch, session_options, tmp_path
):
    options = session_options(configured_total_steps=2)
    session = ServerParallelActionTrainingSession(
        rollout_workers=2, server_trace=False, **options
    )
    checkpoint = tmp_path / "valid.pt"
    try:
        session.run_update()
        session.save(checkpoint)
    finally:
        session.close()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if mismatch == "source":
        payload["metadata"]["source_full_checkpoint_provenance"][
            "source_content_sha256"
        ] = "0" * 64
    elif mismatch == "partner":
        payload["metadata"]["fixed_partner_sha256"] = "0" * 64
    else:
        payload["training_state"]["contract"][
            "cooking_dynamics_id"
        ] = "legacy_cooking"
    invalid = tmp_path / f"invalid_{mismatch}.pt"
    torch.save(payload, invalid)

    with pytest.raises(RuntimeError):
        ServerParallelActionTrainingSession(
            rollout_workers=2,
            server_trace=False,
            **{
                **options,
                "source_checkpoint": None,
                "resume_checkpoint": invalid,
            },
        )
