from __future__ import annotations

from pathlib import Path

import torch

from final_model.partial_ver.network import GazeCritic, GazeNet
from final_model.partial_ver.training import LearnedGazeController
from final_model.server_parallel.train_server import parser, validate_server_args
from final_model.parallel_gaze_contract import (
    PARALLEL_DIRECT_JOINT_TRAINING_CONTRACT,
    PARALLEL_GAZE_CPU_INFERENCE_DTYPE,
    PARALLEL_GAZE_CPU_INFERENCE_PATH,
    PARALLEL_GAZE_CPU_INTER_OP_THREADS,
    PARALLEL_GAZE_CPU_INTRA_OP_THREADS,
    PARALLEL_GAZE_LOGPROB_AUDIT_VERSION,
    PARALLEL_GAZE_RNG_CONTRACT,
    PARALLEL_GAZE_TRAINING_CONTRACT,
    PARALLEL_STAGED_JOINT_TRAINING_CONTRACT,
    logical_env_gaze_seed,
    merge_gaze_transitions,
)


def _samples(master_seed: int, logical_env_id: int, episode_index: int):
    generator = torch.Generator(device="cpu").manual_seed(
        logical_env_gaze_seed(master_seed, logical_env_id, episode_index)
    )
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])
    return [
        int(torch.multinomial(probabilities, 1, generator=generator).item())
        for _ in range(32)
    ]


def test_contract_identity_is_host_neutral():
    assert (
        PARALLEL_DIRECT_JOINT_TRAINING_CONTRACT
        == "parallel_direct_joint_from_full_v1"
    )
    assert PARALLEL_GAZE_TRAINING_CONTRACT == "parallel_gaze_v1"
    assert PARALLEL_STAGED_JOINT_TRAINING_CONTRACT == (
        "parallel_staged_joint_v1"
    )
    assert PARALLEL_GAZE_LOGPROB_AUDIT_VERSION == "cpu_matched_threads_v3"
    assert PARALLEL_GAZE_CPU_INTRA_OP_THREADS == 1
    assert PARALLEL_GAZE_CPU_INTER_OP_THREADS == 1
    assert PARALLEL_GAZE_CPU_INFERENCE_DTYPE == "float32"
    assert PARALLEL_GAZE_CPU_INFERENCE_PATH == "eval_no_grad_distribution_v1"
    assert "worker" not in PARALLEL_GAZE_RNG_CONTRACT


def test_logical_env_rng_is_independent_of_worker_schedule():
    expected = {
        env_id: _samples(20260821, env_id, 3) for env_id in range(8)
    }
    reverse_schedule = {
        env_id: _samples(20260821, env_id, 3)
        for env_id in reversed(range(8))
    }
    assert reverse_schedule == expected
    assert len({logical_env_gaze_seed(20260821, env_id, 3) for env_id in range(8)}) == 8


def test_episode_reset_changes_only_that_logical_env_stream():
    assert _samples(17, 4, 0) != _samples(17, 4, 1)
    assert _samples(17, 4, 1) == _samples(17, 4, 1)


def test_worker_completion_order_cannot_change_transition_order():
    records = [
        {"env_index": 2, "episode_id": 0, "record_sequence_id": 1},
        {"env_index": 0, "episode_id": 1, "record_sequence_id": 4},
        {"env_index": 0, "episode_id": 0, "record_sequence_id": 3},
        {"env_index": 2, "episode_id": 0, "record_sequence_id": 0},
    ]
    first = merge_gaze_transitions([
        {"gaze_records": records[:2]}, {"gaze_records": records[2:]}
    ])
    second = merge_gaze_transitions([
        {"gaze_records": records[2:]}, {"gaze_records": records[:2]}
    ])
    expected = [records[2], records[1], records[3], records[0]]
    assert first == second == expected


def test_duplicate_transition_ownership_fails_closed():
    record = {"env_index": 0, "episode_id": 0, "record_sequence_id": 0}
    try:
        merge_gaze_transitions([
            {"gaze_records": [record]}, {"gaze_records": [dict(record)]}
        ])
    except RuntimeError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate logical transition was accepted")


def test_serial_controller_state_contract_is_unchanged():
    controller = LearnedGazeController(
        GazeNet(), GazeCritic(), torch.device("cpu")
    )
    assert "policy_version" not in controller.training_state()


def test_parallel_entrypoint_accepts_learned_gaze_only():
    root = Path(__file__).resolve().parents[3]
    args = parser().parse_args([
        "--observation-mode", "partial",
        "--gaze-type", "learned",
        "--train-components", "gaze",
        "--shared-horizons", "0", "2", "4",
        "--shared-decision-times", "200", "400", "600",
        "--fixation-shape", "plus",
        "--source-checkpoint", str(
            root
            / "final_model/runs/p2_local_no_decay_production_20260831_150338"
            / "no_decay_action/checkpoints/step_020000000.pt"
        ),
        "--fixed-partner-checkpoint",
        str(root / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"),
        "--rollout-workers", "8",
        "--n-rollout-threads", "8",
    ])
    validate_server_args(args)
    assert args.gaze_type == "learned"
    assert args.train_gaze and not args.train_action
    assert args.torch_intra_op_threads == 1
    assert args.torch_inter_op_threads == 1


def test_parallel_entrypoint_accepts_explicit_direct_joint_from_full():
    root = Path(__file__).resolve().parents[3]
    args = parser().parse_args([
        "--observation-mode", "partial",
        "--gaze-type", "learned",
        "--train-components", "action", "gaze",
        "--direct-joint-from-full",
        "--shared-horizons", "0", "2", "4",
        "--shared-decision-times", "200", "400", "600",
        "--fixation-shape", "plus",
        "--source-checkpoint", str(
            root
            / "final_model/checkpoint/action/legacy"
            / "action_full_action_network_120M.pt"
        ),
        "--fixed-partner-checkpoint",
        str(root / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"),
        "--rollout-workers", "8",
        "--n-rollout-threads", "100",
    ])
    validate_server_args(args)
    assert args.gaze_type == "learned"
    assert args.train_action and args.train_gaze
    assert args.direct_joint_from_full
    assert args.rollout_workers == 8
    assert args.torch_intra_op_threads == 1
    assert args.torch_inter_op_threads == 1


def test_parallel_entrypoint_accepts_staged_joint_from_learned_gaze():
    root = Path(__file__).resolve().parents[3]
    args = parser().parse_args([
        "--observation-mode", "partial",
        "--gaze-type", "learned",
        "--train-components", "action", "gaze",
        "--shared-horizons", "0", "2", "4",
        "--shared-decision-times", "200", "400", "600",
        "--fixation-shape", "plus",
        "--source-checkpoint", str(
            root
            / "final_model/checkpoint/gaze"
            / "cau_diamond13_memory2400_gaze_13m.pt"
        ),
        "--fixed-partner-checkpoint",
        str(root / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"),
        "--memory-decay-mode", "stochastic_exponential",
        "--memory-tau-ms", "800",
        "--memory-max-lifetime-ms", "2400",
        "--constraint-profile", "custom",
        "--peripheral-mode", "local",
        "--peripheral-sigma-deg", "3",
        "--peripheral-extent", "diamond13",
        "--emma-mode", "default_encoding",
        "--preview-opacity-mode", "off",
        "--rollout-workers", "8",
        "--n-rollout-threads", "100",
    ])
    validate_server_args(args)
    assert args.gaze_type == "learned"
    assert args.train_action and args.train_gaze
    assert not args.direct_joint_from_full
    assert args.peripheral_extent == "diamond13"
    assert args.torch_intra_op_threads == 1
    assert args.torch_inter_op_threads == 1
