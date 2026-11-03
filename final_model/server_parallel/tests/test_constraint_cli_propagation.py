import json
from pathlib import Path

import pytest
import torch

from final_model.partial_ver.observation import DEADLINE_ACTION_CONTRACT_ID
from final_model.server_parallel.train_server import (
    _session,
    parser,
    run_training,
    validate_server_args,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = next(
    (
        path
        for path in (
            ROOT
            / "final_model/checkpoint/action/legacy/action_full_action_network_120M.pt",
            ROOT / "final_model/checkpoint/action/action_full_action_network_120M.pt",
        )
        if path.is_file()
    ),
    None,
)
if SOURCE is None:
    pytest.skip(
        "requires a legacy action source checkpoint not included in the final public set",
        allow_module_level=True,
    )
PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
TUNED = {
    "prep_ms": 135.0,
    "exec_intercept_ms": 44.8732080135,
    "exec_slope_ms_per_degree": 3.83832565515,
    "dwell_ms": 60.0,
}


def _args(*constraint_flags, run_dir=None, rollout_length=1):
    values = [
        "--observation-mode", "partial",
        "--gaze-type", "rule_based",
        "--train-components", "action",
        "--horizon", "4",
        "--decision-time", "400",
        "--fixation-shape", "plus",
        "--memory-lifetime-ms", "1200",
        "--source-checkpoint", str(SOURCE),
        "--fixed-partner-checkpoint", str(PARTNER),
        "--rollout-workers", "1",
        "--rollout-length", str(rollout_length),
        "--episode-length", str(rollout_length),
        "--n-rollout-threads", "1",
        "--steps", str(rollout_length),
        "--checkpoint-interval", str(rollout_length),
        "--ppo-epochs", "1",
        "--num-mini-batches", "1",
        "--seed", "31",
        "--device", "cpu",
        "--wandb-mode", "disabled",
        "--no-progress",
        *constraint_flags,
    ]
    if run_dir is not None:
        values.extend(("--run-dir", str(run_dir)))
    args = parser().parse_args(values)
    validate_server_args(args)
    return args


P2_FLAGS = (
    "--constraint-profile", "custom",
    "--peripheral-mode", "global",
    "--peripheral-sigma-deg", "3.0",
    "--emma-mode", "default",
    "--preview-opacity-mode", "off",
)
P2O_FLAGS = (*P2_FLAGS[:-1], "actual")
E_FLAGS = (
    "--constraint-profile", "custom",
    "--peripheral-mode", "none",
    "--emma-mode", "tuned",
    "--emma-prep-ms", str(TUNED["prep_ms"]),
    "--emma-exec-intercept-ms", str(TUNED["exec_intercept_ms"]),
    "--emma-exec-slope-ms-per-degree",
    str(TUNED["exec_slope_ms_per_degree"]),
    "--emma-dwell-ms", str(TUNED["dwell_ms"]),
    "--preview-opacity-mode", "off",
)
FULL_FLAGS = (
    "--constraint-profile", "custom",
    "--peripheral-mode", "global",
    "--peripheral-sigma-deg", "3.0",
    *E_FLAGS[4:-2],
    "--preview-opacity-mode", "actual",
)


@pytest.mark.parametrize(
    ("flags", "variant", "peripheral", "opacity"),
    (
        (P2_FLAGS, "p2", "global", "off"),
        (P2O_FLAGS, "p2o", "global", "actual"),
        (E_FLAGS, "e", "none", "off"),
        (FULL_FLAGS, "full", "global", "actual"),
    ),
)
def test_real_server_cli_propagates_resolved_constraint_before_rollout(
    flags,
    variant,
    peripheral,
    opacity,
):
    args = _args(*flags)
    session = _session(args)
    try:
        config = session.constraint_config
        assert config is args._resolved_constraint_config
        assert config.variant_id == variant
        assert config.peripheral.mode == peripheral
        assert config.peripheral.sigma_deg == (
            None if peripheral == "none" else 3.0
        )
        assert config.preview_opacity.mode == opacity
        if variant in {"e", "full"}:
            assert config.emma.prep_ms == TUNED["prep_ms"]
            assert config.emma.exec_intercept_ms == TUNED["exec_intercept_ms"]
            assert config.emma.exec_slope_ms_per_degree == TUNED[
                "exec_slope_ms_per_degree"
            ]
            assert config.emma.dwell_ms == TUNED["dwell_ms"]
    finally:
        session.close()


@pytest.mark.parametrize(
    ("flags", "variant", "expect_opacity"),
    (
        (P2_FLAGS, "p2", False),
        (P2O_FLAGS, "p2o", True),
    ),
)
def test_real_server_cli_rollout_checkpoint_and_config_agree(
    tmp_path,
    flags,
    variant,
    expect_opacity,
):
    run_dir = tmp_path / variant
    summary = run_training(
        _args(*flags, run_dir=run_dir, rollout_length=8)
    )
    assert summary["latest"]["environment_steps"] == 8
    checkpoint = torch.load(
        run_dir / "checkpoints/step_000000008.pt",
        map_location="cpu",
        weights_only=False,
    )
    metadata = checkpoint["metadata"]
    adapter = checkpoint["training_state"]["adapters"][0]
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )

    assert metadata["constraint_variant_id"] == variant
    assert metadata["peripheral_mode"] == "global"
    assert metadata["peripheral_sigma_deg"] == 3.0
    assert config["constraint_variant_id"] == metadata[
        "constraint_variant_id"
    ]
    assert config["peripheral_mode"] == metadata["peripheral_mode"]
    assert config["peripheral_sigma_deg"] == metadata[
        "peripheral_sigma_deg"
    ]
    latest = summary["latest"]
    assert latest["peripheral_attempts"] > 0
    assert latest["peripheral_sample_events"] > 0
    assert latest["peripheral_rng_draw_count"] == latest[
        "peripheral_attempts"
    ]
    assert latest["peripheral_success_count"] == latest[
        "peripheral_successes"
    ]
    assert latest["peripheral_attempts"] == (
        latest["peripheral_successes"] + latest["peripheral_failures"]
    )
    assert latest["peripheral_success_rate"] == pytest.approx(
        latest["peripheral_successes"] / latest["peripheral_attempts"]
    )
    assert latest["peripheral_successes_per_acquisition"] == pytest.approx(
        latest["peripheral_successes"] / max(latest["acquisitions"], 1)
    )
    assert latest["peripheral_successes_per_decision"] == pytest.approx(
        latest["peripheral_successes"] / latest["task_decisions"]
    )
    assert adapter["metrics"]["peripheral_attempts"] > 0
    assert adapter["metrics"]["peripheral_rng_draw_count"] > 0
    if expect_opacity:
        assert int(adapter["opacity_attempts_by_step"].sum()) > 0
        assert adapter["opacity_rng_draw_count"] > 0
    else:
        assert int(adapter["opacity_attempts_by_step"].sum()) == 0
        assert adapter["opacity_rng_draw_count"] == 0


def test_default_encoding_stochastic_memory_reaches_server_worker(tmp_path):
    run_dir = tmp_path / "default_encoding_diamond13_1200"
    values = [
        "--observation-mode", "partial",
        "--gaze-type", "rule_based",
        "--train-components", "action",
        "--horizon", "0",
        "--decision-time", "200",
        "--fixation-shape", "plus",
        "--memory-decay-mode", "stochastic_exponential",
        "--memory-tau-ms", "400",
        "--memory-max-lifetime-ms", "1200",
        "--source-checkpoint", str(SOURCE),
        "--fixed-partner-checkpoint", str(PARTNER),
        "--rollout-workers", "1",
        "--rollout-length", "2",
        "--episode-length", "2",
        "--n-rollout-threads", "1",
        "--steps", "2",
        "--checkpoint-interval", "2",
        "--ppo-epochs", "1",
        "--num-mini-batches", "1",
        "--seed", "31",
        "--device", "cpu",
        "--wandb-mode", "disabled",
        "--no-progress",
        "--smoke",
        "--constraint-profile", "custom",
        "--peripheral-mode", "local",
        "--peripheral-sigma-deg", "3",
        "--peripheral-extent", "diamond13",
        "--emma-mode", "default_encoding",
        "--preview-opacity-mode", "off",
        "--run-dir", str(run_dir),
    ]
    args = parser().parse_args(values)
    validate_server_args(args)
    assert args.memory_tau_ms == 400.0
    assert args.memory_max_lifetime_ms == 1200.0
    assert args.peripheral_extent == "diamond13"
    assert args.emma_mode == "default_encoding"

    run_training(args)
    payload = torch.load(
        run_dir / "checkpoints/step_000000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    metadata = payload["metadata"]
    contract = payload["training_state"]["contract"]
    adapter = payload["training_state"]["adapters"][0]
    memory = adapter["memory"]
    for mapping in (metadata, contract, adapter):
        assert mapping["peripheral_extent"] == "diamond13"
        assert mapping["emma_mode"] == "default_encoding"
        assert mapping["encoding_K"] == 0.006
        assert mapping["encoding_k"] == 0.4
        assert mapping["encoding_f"] == 0.1
    for mapping in (metadata, contract):
        assert mapping["memory_tau_ms"] == 400.0
        assert mapping["memory_max_lifetime_ms"] == 1200.0
        assert mapping["memory_random_seed"] == 131_000_070
    assert memory["memory_tau_ms"] == 400.0
    assert memory["memory_max_lifetime_ms"] == 1200.0
    assert memory["memory_seed"] == 131_000_070
    assert "memory_rng_state" in memory
    assert metadata["acquisition_action_contract_id"] == (
        DEADLINE_ACTION_CONTRACT_ID
    )
    assert contract["acquisition_action_contract_id"] == (
        DEADLINE_ACTION_CONTRACT_ID
    )
    assert adapter["metrics"]["task_decisions"] == 2
    assert adapter["metrics"]["actionnet_online_forwards"] == 2
    assert adapter["metrics"]["action_proposals"] == 2
    assert adapter["metrics"]["proposal_revisions"] == 0
    assert adapter["metrics"]["acquisitions"] > 2
