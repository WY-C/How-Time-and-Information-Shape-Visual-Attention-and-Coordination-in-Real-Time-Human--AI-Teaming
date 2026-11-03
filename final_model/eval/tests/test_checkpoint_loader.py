from __future__ import annotations

import torch
import pytest

from final_model.eval.checkpoint_loader import load_action_checkpoint_for_evaluation


@pytest.mark.parametrize("horizon", (0, 2, 4))
def test_fixed_metadata_parsing(synthetic_checkpoints, horizon):
    loaded = load_action_checkpoint_for_evaluation(synthetic_checkpoints[f"h{horizon}"])
    assert loaded.horizon_mode == "fixed"
    assert loaded.horizons == (horizon,)
    assert loaded.completed_steps == 123


def test_shared_metadata_parsing(synthetic_checkpoints):
    loaded = load_action_checkpoint_for_evaluation(synthetic_checkpoints["shared"])
    assert loaded.horizon_mode == "episode_balanced"
    assert loaded.horizons == (0, 2, 4)
    assert loaded.completed_steps == 456


def test_unsupported_checkpoint_fails_closed(synthetic_checkpoints, tmp_path):
    payload = torch.load(synthetic_checkpoints["h0"], weights_only=False)
    payload["metadata"]["observation_mode"] = "partial"
    path = tmp_path / "partial.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="unsupported"):
        load_action_checkpoint_for_evaluation(path)


def test_legacy_cooking_dynamics_checkpoint_fails_closed(
    synthetic_checkpoints, tmp_path
):
    payload = torch.load(synthetic_checkpoints["h0"], weights_only=False)
    payload["metadata"].pop("cooking_dynamics_id")
    path = tmp_path / "legacy_cooking.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="cooking_dynamics_id"):
        load_action_checkpoint_for_evaluation(path)


@pytest.mark.parametrize("horizon", (0, 2, 4))
def test_shared_supports_all_horizons(synthetic_checkpoints, horizon):
    loaded = load_action_checkpoint_for_evaluation(synthetic_checkpoints["shared"])
    assert loaded.validate_evaluation_horizon(horizon) == horizon


def test_fixed_requires_native_horizon(synthetic_checkpoints):
    loaded = load_action_checkpoint_for_evaluation(synthetic_checkpoints["h2"])
    assert loaded.validate_evaluation_horizon(2) == 2
    with pytest.raises(RuntimeError):
        loaded.validate_evaluation_horizon(4)


def test_fixed_explicit_downward_ablation(synthetic_checkpoints):
    loaded = load_action_checkpoint_for_evaluation(synthetic_checkpoints["h4"])
    assert loaded.validate_evaluation_horizon(2, preview_ablation=True) == 2


def test_incomplete_checkpoint_rejected(synthetic_checkpoints, tmp_path):
    payload = torch.load(synthetic_checkpoints["h0"], weights_only=False)
    payload.pop("training_state")
    path = tmp_path / "incomplete.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="training_state"):
        load_action_checkpoint_for_evaluation(path)


def test_step_mismatch_rejected(synthetic_checkpoints, tmp_path):
    payload = torch.load(synthetic_checkpoints["h0"], weights_only=False)
    payload["training_state"]["completed_steps"] = 999
    path = tmp_path / "mismatch.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="step mismatch"):
        load_action_checkpoint_for_evaluation(path)


def test_loader_constructs_no_optimizer(synthetic_checkpoints, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer constructed")
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    load_action_checkpoint_for_evaluation(synthetic_checkpoints["h0"])
