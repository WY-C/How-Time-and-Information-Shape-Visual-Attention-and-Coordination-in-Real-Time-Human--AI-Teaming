from pathlib import Path

import pytest

import final_model.train as train_module
from final_model.server_parallel.train_server import parser, validate_server_args


ROOT = Path(__file__).resolve().parents[3]
HSP = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"
PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
FULL_120M = (
    ROOT
    / "final_model/checkpoint/action/legacy/action_full_action_network_120M.pt"
)


@pytest.fixture(autouse=True)
def canonical_checkpoint_hashes(monkeypatch):
    hashes = {
        HSP: train_module.HSP_SOURCE_CHECKPOINT_SHA256,
        PARTNER: train_module.FIXED_HSP_PARTNER_SHA256,
    }
    monkeypatch.setattr(train_module, "file_sha256", lambda path: hashes[Path(path)])


def _cli(*extra):
    return parser().parse_args([
        "--observation-mode", "full",
        "--training-stage", "action",
        "--horizon", "0",
        "--hsp-initial-checkpoint", str(HSP),
        "--fixed-partner-checkpoint", str(PARTNER),
        "--inference-batching", "predictor",
        *extra,
    ])


def test_full_predictor_only_configuration_is_accepted():
    args = _cli("--rollout-workers", "2", "--n-rollout-threads", "2")
    validate_server_args(args)
    assert args.inference_batching == "predictor"


def test_full_gaze_server_configuration_is_accepted():
    args = parser().parse_args([
        "--observation-mode", "full",
        "--training-stage", "gaze",
        "--horizon", "0",
        "--decision-time", "200",
        "--fixation-shape", "3x3",
        "--source-checkpoint", str(FULL_120M),
        "--fixed-partner-checkpoint", str(PARTNER),
        "--n-rollout-threads", "2",
        "--rollout-workers", "2",
    ])
    validate_server_args(args)
    assert args.ground_truth_policy_input is True
    assert args.inference_batching == "legacy"


def test_full_ground_truth_joint_server_configuration_is_accepted():
    args = parser().parse_args([
        "--observation-mode", "full",
        "--gaze-type", "learned",
        "--train-components", "action", "gaze",
        "--shared-horizons", "0", "2", "4",
        "--shared-decision-times", "200", "400", "600",
        "--fixation-shape", "3x3",
        "--source-checkpoint", str(FULL_120M),
        "--fixed-partner-checkpoint", str(PARTNER),
        "--n-rollout-threads", "2",
        "--rollout-workers", "2",
    ])
    validate_server_args(args)
    assert args.training_stage == "joint"
    assert args.train_action and args.train_gaze
    assert args.ground_truth_policy_input is True
    assert args.inference_batching == "legacy"


@pytest.mark.parametrize("mode", ("legacy", "partner", "partner_predictor"))
def test_full_non_predictor_inference_modes_fail_closed(mode):
    with pytest.raises(ValueError, match="inference_batching"):
        validate_server_args(_cli("--inference-batching", mode))


def test_full_gaze_runtime_and_invalid_worker_count_fail_closed():
    with pytest.raises(ValueError, match="Full mode supports"):
        validate_server_args(_cli("--gaze-type", "rule_based"))
    with pytest.raises(ValueError, match="may not exceed"):
        validate_server_args(
            _cli("--rollout-workers", "3", "--n-rollout-threads", "2")
        )


def test_full_wrong_learner_or_partner_content_fails_closed():
    args = _cli()
    args.hsp_initial_checkpoint = PARTNER
    with pytest.raises(ValueError, match="S2-s36"):
        validate_server_args(args)
    args = _cli()
    args.fixed_partner_checkpoint = HSP
    with pytest.raises(ValueError, match="S2-s24"):
        validate_server_args(args)
