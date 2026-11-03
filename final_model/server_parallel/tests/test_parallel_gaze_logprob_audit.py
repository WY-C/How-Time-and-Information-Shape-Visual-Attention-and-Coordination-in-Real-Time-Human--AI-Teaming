from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from final_model.parallel_gaze_contract import (
    PARALLEL_GAZE_LOGPROB_AUDIT_VERSION,
)
from final_model.partial_ver.network import GazeNet
from final_model.server_parallel.parallel_action_training import (
    ServerParallelActionTrainingSession,
)
from final_model.server_parallel.train_server import (
    _session as build_session,
    parser,
    validate_server_args,
)


POLICY_VERSION = 7
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def matched_main_cpu_thread_contract(monkeypatch, request):
    if request.node.name == (
        "test_worker1_checkpoint_resumes_with_worker8_and_preserves_parity"
    ):
        return
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)


def _session(actor: GazeNet, device: str = "cpu"):
    session = object.__new__(ServerParallelActionTrainingSession)
    session.parallel_gaze = True
    session.gaze_actor = actor.to(device)
    session.device = torch.device(device)
    session.update_count = POLICY_VERSION
    return session


def _records(actor: GazeNet, count: int = 8):
    generator = np.random.default_rng(20260902)
    records = []
    actor = actor.cpu().eval()
    with torch.no_grad():
        for _ in range(count):
            observation = generator.normal(size=(8, 5, 26)).astype(np.float32)
            context = generator.normal(size=7).astype(np.float32)
            distribution = actor.distribution(
                torch.from_numpy(observation[None]),
                torch.from_numpy(context[None]),
            )
            action = int(distribution.probs.argmax(dim=-1).item())
            records.append({
                "observation": observation,
                "context": context,
                "action": action,
                "log_prob": float(
                    distribution.log_prob(torch.tensor([action])).item()
                ),
                "policy_version": POLICY_VERSION,
            })
    return records


def _controlled_actor() -> GazeNet:
    actor = GazeNet().eval()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.head.bias.copy_(torch.linspace(-1.0, 1.0, 40))
    return actor


def test_parallel_gaze_strict_audit_recomputes_cpu_worker_records_on_cpu():
    actor = _controlled_actor()
    records = _records(actor)
    old_log_probs = [record["log_prob"] for record in records]

    result = _session(actor)._gaze_preupdate_audit(records)

    assert result["gaze_preupdate_parity_pass"] == 1.0
    assert result["gaze_preupdate_max_abs_logprob_difference"] <= 1e-5
    assert result["gaze_logprob_audit_version"] == (
        PARALLEL_GAZE_LOGPROB_AUDIT_VERSION
    )
    assert result["gaze_logprob_rollout_device"] == "cpu"
    assert result["gaze_logprob_audit_device"] == "cpu"
    assert result["gaze_logprob_execution_settings_matched"] is True
    assert result["gaze_logprob_main_cpu_intra_op_threads"] == 1
    assert result["gaze_logprob_main_cpu_inter_op_threads"] == 1
    assert result["gaze_logprob_worker_cpu_intra_op_threads"] == 1
    assert result["gaze_logprob_worker_cpu_inter_op_threads"] == 1
    assert result["gaze_logprob_inference_dtype"] == "float32"
    assert result["gaze_logprob_inference_path"] == (
        "eval_no_grad_distribution_v1"
    )
    assert [record["log_prob"] for record in records] == old_log_probs


def test_parallel_gaze_strict_audit_fails_for_stale_weights():
    rollout_actor = _controlled_actor()
    records = _records(rollout_actor)
    stale_main_actor = copy.deepcopy(rollout_actor)
    with torch.no_grad():
        stale_main_actor.head.bias[records[0]["action"]].add_(0.5)

    with pytest.raises(RuntimeError, match="log-prob parity failed"):
        _session(stale_main_actor)._gaze_preupdate_audit(records)


def test_parallel_gaze_strict_audit_fails_for_wrong_action_pairing():
    actor = _controlled_actor()
    records = _records(actor, count=1)
    records[0]["action"] = 0

    with pytest.raises(RuntimeError, match="log-prob parity failed"):
        _session(actor)._gaze_preupdate_audit(records)


def test_parallel_gaze_strict_audit_fails_for_stale_policy_version():
    actor = _controlled_actor()
    records = _records(actor, count=1)
    records[0]["policy_version"] = POLICY_VERSION - 1

    with pytest.raises(RuntimeError, match="policy version mismatch"):
        _session(actor)._gaze_preupdate_audit(records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_difference_is_diagnostic_not_strict_audit_input(capsys):
    torch.manual_seed(20260902)
    actor = GazeNet().cpu().eval()
    records = _records(actor, count=64)
    strict = _session(copy.deepcopy(actor), device="cuda")._gaze_preupdate_audit(
        records
    )

    differences = []
    cuda_actor = copy.deepcopy(actor).cuda().eval()
    with torch.no_grad():
        for record in records:
            action = torch.tensor([record["action"]], device="cuda")
            current = cuda_actor.distribution(
                torch.as_tensor(record["observation"][None], device="cuda"),
                torch.as_tensor(record["context"][None], device="cuda"),
            ).log_prob(action)
            differences.append(abs(float(current.item()) - record["log_prob"]))
    maximum = float(np.max(differences))
    mean = float(np.mean(differences))
    print(f"CPU_CUDA_LOGPROB_DIFF max={maximum:.9g} mean={mean:.9g}")

    assert strict["gaze_preupdate_parity_pass"] == 1.0
    assert strict["gaze_logprob_audit_device"] == "cpu"
    assert strict["gaze_logprob_training_device"] == "cuda"
    assert np.isfinite(maximum)
    assert np.isfinite(mean)
    assert "CPU_CUDA_LOGPROB_DIFF" in capsys.readouterr().out


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="resume checkpoint contains CUDA RNG state",
)
def test_legacy_checkpoint_without_source_critic_provenance_fails_closed():
    resume = (
        ROOT
        / "final_model/pilots/parallel_gaze_v1_resume_source_boundary/"
        "checkpoints/step_000000032.pt"
    )
    args = parser().parse_args([
        "--observation-mode", "partial",
        "--shared-horizons", "0", "2", "4",
        "--shared-decision-times", "200", "400", "600",
        "--fixation-shape", "plus",
        "--fixed-partner-checkpoint",
        str(ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"),
        "--memory-decay-mode", "none",
        "--steps", "64",
        "--rollout-length", "4",
        "--episode-length", "4",
        "--n-rollout-threads", "8",
        "--checkpoint-interval", "32",
        "--reward-shaping-horizon-steps", "20000000",
        "--device", "cpu",
        "--ppo-epochs", "1",
        "--num-mini-batches", "1",
        "--wandb-mode", "disabled",
        "--constraint-profile", "custom",
        "--peripheral-mode", "global",
        "--peripheral-sigma-deg", "3.0",
        "--emma-mode", "tuned",
        "--emma-prep-ms", "135",
        "--emma-exec-intercept-ms", "44.8732080135",
        "--emma-exec-slope-ms-per-degree", "3.83832565515",
        "--emma-dwell-ms", "60",
        "--preview-opacity-mode", "off",
        "--gaze-type", "learned",
        "--train-components", "gaze",
        "--resume-checkpoint", str(resume),
        "--rollout-workers", "8",
        "--host-mode", "local",
        "--seed", "20260821",
        "--no-progress",
    ])
    validate_server_args(args)
    with pytest.raises(RuntimeError, match="source_critic_parameter_sha256"):
        build_session(args)
