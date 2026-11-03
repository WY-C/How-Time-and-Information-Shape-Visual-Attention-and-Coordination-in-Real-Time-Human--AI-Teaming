from pathlib import Path

import pytest

from final_model.server_parallel.train_server import parser, validate_server_args


ROOT = Path(__file__).resolve().parents[3]
HSP = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"
PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
SOURCE = (
    ROOT
    / "final_model/source_checkpoints/"
    "shared_full_seed20260821_step_002000000.pt"
)


def _full(*extra):
    return parser().parse_args([
        "--observation-mode", "full",
        "--training-stage", "action",
        "--horizon", "0",
        "--hsp-initial-checkpoint", str(HSP),
        "--fixed-partner-checkpoint", str(PARTNER),
        "--inference-batching", "predictor",
        *extra,
    ])


def _partial(*extra):
    return parser().parse_args([
        "--observation-mode", "partial",
        "--gaze-type", "rule_based",
        "--train-components", "action",
        "--horizon", "0",
        "--decision-time", "200",
        "--fixation-shape", "3x3",
        "--source-checkpoint", str(SOURCE),
        "--fixed-partner-checkpoint", str(PARTNER),
        *extra,
    ])


def test_full_production_execution_defaults_are_frozen():
    args = _full()
    validate_server_args(args)
    assert args.rollout_workers == 8
    assert args.torch_intra_op_threads == 2
    assert args.torch_inter_op_threads == 1
    assert args.fixed_partner_executor_workers == 4


def test_partial_rollout_default_remains_one():
    args = _partial()
    validate_server_args(args)
    assert args.rollout_workers == 1


@pytest.mark.parametrize(
    "flag",
    (
        "--torch-intra-op-threads",
        "--torch-inter-op-threads",
        "--fixed-partner-executor-workers",
    ),
)
def test_nonpositive_execution_settings_fail_closed(flag):
    with pytest.raises(ValueError, match="must be positive"):
        validate_server_args(_full(flag, "0"))


def test_explicit_execution_settings_remain_configurable():
    args = _full(
        "--rollout-workers", "3",
        "--torch-intra-op-threads", "3",
        "--torch-inter-op-threads", "2",
        "--fixed-partner-executor-workers", "5",
    )
    validate_server_args(args)
    assert (
        args.rollout_workers,
        args.torch_intra_op_threads,
        args.torch_inter_op_threads,
        args.fixed_partner_executor_workers,
    ) == (3, 3, 2, 5)
