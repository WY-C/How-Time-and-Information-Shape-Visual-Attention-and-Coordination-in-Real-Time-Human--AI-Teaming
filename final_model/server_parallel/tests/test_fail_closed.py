from pathlib import Path

import pytest

from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
)
from final_model.server_parallel.train_server import parser, validate_server_args


def _cli(*extra):
    root = Path(__file__).resolve().parents[3]
    return parser().parse_args([
        "--observation-mode", "partial",
        "--gaze-type", "rule_based",
        "--train-components", "action",
        "--horizon", "0",
        "--decision-time", "200",
        "--fixation-shape", "3x3",
        "--source-checkpoint",
        str(root / "final_model/source_checkpoints/shared_full_seed20260821_step_002000000.pt"),
        "--fixed-partner-checkpoint",
        str(root / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"),
        *extra,
    ])


@pytest.mark.parametrize(
    "arguments",
    (
        ("--gaze-type", "learned"),
        ("--train-components", "action", "gaze"),
        ("--partner", "hsp5"),
    ),
)
def test_unsupported_modes_fail_closed(arguments):
    args = _cli(*arguments)
    with pytest.raises(ValueError):
        validate_server_args(args)


def test_missing_source_and_invalid_worker_counts_fail_closed():
    args = _cli()
    args.source_checkpoint = None
    with pytest.raises(ValueError, match="source-checkpoint"):
        validate_server_args(args)
    args = _cli("--rollout-workers", "3", "--n-rollout-threads", "2")
    with pytest.raises(ValueError, match="may not exceed"):
        validate_server_args(args)


def test_server_parser_accepts_custom_memory_lifetime():
    args = _cli("--memory-lifetime-ms", "2501")
    validate_server_args(args)
    assert args.memory_lifetime_ms == 2501


def test_resume_checkpoint_interval_must_be_positive_and_update_aligned():
    args = _cli("--checkpoint-interval", "0")
    with pytest.raises(ValueError, match="checkpoint-interval"):
        validate_server_args(args)

    args = _cli("--checkpoint-interval", "500000")
    args.resume_checkpoint = Path("resume.pt")
    args.source_checkpoint = None
    with pytest.raises(ValueError, match="complete PPO updates"):
        validate_server_args(args)


def test_wrong_source_or_partner_sha_fails_before_worker_start(session_options):
    options = session_options()
    with pytest.raises(RuntimeError, match="source checkpoint"):
        ServerParallelActionTrainingSession(
            rollout_workers=1,
            **{**options, "source_checkpoint": options["fixed_partner_checkpoint"]},
        )
    with pytest.raises(ValueError, match="canonical fixed partner"):
        ServerParallelActionTrainingSession(
            rollout_workers=1,
            **{**options, "fixed_partner_checkpoint": options["source_checkpoint"]},
        )
