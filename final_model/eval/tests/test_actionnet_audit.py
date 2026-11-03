import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from final_model.eval.actionnet_audit import DEFAULT_TESTS, parser
from final_model.eval.actionnet_diagnostics import (
    ActionSnapshot,
    audit_snapshot,
    memory_variant,
    preview_variant,
    read_only_probabilities,
    read_only_logits_probabilities_and_hidden,
    source_comparability,
)
from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.eval.counterfactual import distribution_metrics
from final_model.partial_ver.memory import DYNAMIC_CHANNELS, STATIC_CHANNELS


class DummyActor:
    """Small deterministic get_probs-compatible actor with a recurrent dependency."""

    def __init__(self):
        self.hidden_shapes = []

    def get_probs(self, observation, hidden, mask, available):
        obs = torch.as_tensor(observation, dtype=torch.float32)
        recurrent = torch.as_tensor(hidden, dtype=torch.float32)
        self.hidden_shapes.append(tuple(recurrent.shape))
        available_tensor = torch.as_tensor(available, dtype=torch.float32)
        features = torch.stack(
            (
                obs[..., 0].sum(dim=(-2, -1)),
                obs[..., 1].sum(dim=(-2, -1)),
                obs[..., 15].sum(dim=(-2, -1)),
                obs[..., 21].sum(dim=(-2, -1)),
                obs[..., 20].sum(dim=(-2, -1)),
                obs[..., 22:25].sum(dim=(-3, -2, -1)),
            ),
            dim=-1,
        )
        logits = features * 0.01 + recurrent.reshape(recurrent.shape[0], -1)[:, :6] * 0.1
        logits = logits.masked_fill(available_tensor <= 0, -1e9)
        probabilities = torch.softmax(logits, dim=-1)
        candidate = recurrent + obs.mean().reshape(1, 1, 1)
        return probabilities, candidate


def observation(horizon=4):
    value = np.zeros((8, 5, 25), dtype=np.float32)
    value[..., STATIC_CHANNELS] = 1.0
    value[2, 3, 0] = 255.0
    value[4, 1, 1] = 255.0
    value[2, 3, 15] = 510.0
    value[2, 3, 20] = 1.0
    if horizon:
        value[2, 3, 21 : 21 + horizon] = 255.0
    return value


def snapshot(horizon=4):
    actor = DummyActor()
    obs = observation(horizon)
    hidden = np.arange(64, dtype=np.float32).reshape(1, 1, 64) / 64.0
    mask = np.ones((1, 1), dtype=np.float32)
    available = np.ones(6, dtype=np.float32)
    probabilities = read_only_probabilities(actor, obs, hidden, mask, available)
    return actor, ActionSnapshot(
        episode=0, seed=7, decision=3, proposal_index=0, timestamp_ms=60.0,
        decision_time=200, horizon=horizon, fixation=(2, 3), observation=obs,
        hidden=hidden, mask=mask, available=available,
        factual_probabilities=probabilities, factual_action=int(probabilities.argmax()),
    )


def test_factual_replay_produces_identical_probabilities_and_is_deterministic():
    actor, value = snapshot()
    rows, checks = audit_snapshot(actor, value)
    assert rows
    assert checks["factual_replay_identity"]
    assert checks["deterministic_repeatability"]


def test_counterfactual_forward_does_not_mutate_hidden_state():
    actor, value = snapshot()
    before = value.hidden.copy()
    audit_snapshot(actor, value)
    assert np.array_equal(value.hidden, before)


def test_counterfactual_forward_does_not_mutate_factual_observation():
    actor, value = snapshot()
    before = value.observation.copy()
    audit_snapshot(actor, value)
    assert np.array_equal(value.observation, before)


def test_no_preview_is_exact_noop_for_h0():
    source = observation(0)
    assert np.array_equal(preview_variant(source, horizon=0, keep=0), source)


@pytest.mark.parametrize(("horizon", "keep"), ((2, 0), (4, 2)))
def test_no_preview_changes_valid_h2_h4_observation(horizon, keep):
    source = observation(horizon)
    assert not np.array_equal(preview_variant(source, horizon=horizon, keep=keep), source)


def test_static_only_uses_canonical_unknown_for_dynamic_and_preview():
    source = observation(4)
    result, label = memory_variant(source, "static_only", horizon=4)
    assert np.count_nonzero(result[..., list(DYNAMIC_CHANNELS)]) == 0
    assert np.count_nonzero(result[..., 20:25]) == 0
    assert np.array_equal(result[..., list(STATIC_CHANNELS)], source[..., list(STATIC_CHANNELS)])
    assert label["scientific_role"] == "MAIN_COUNTERFACTUAL"


def test_effective_decision_detection_matches_actor_tensor_change():
    actor, value = snapshot(0)
    rows, _checks = audit_snapshot(actor, value)
    preview = [row for row in rows if row["diagnostic"] == "preview_counterfactual"]
    static = [row for row in rows if row.get("variant") == "static_only"]
    assert preview and all(row["effective"] == 0 for row in preview)
    assert static and all(row["effective"] == 1 for row in static)


def test_distribution_metrics_are_zero_for_identical_distributions():
    probabilities = np.asarray([.1, .2, .1, .2, .2, .2])
    metrics = distribution_metrics(probabilities, probabilities.copy())
    assert metrics["kl_native_counterfactual"] == pytest.approx(0.0)
    assert metrics["js_divergence"] == pytest.approx(0.0)
    assert metrics["total_variation_distance"] == pytest.approx(0.0)


def test_distribution_metrics_are_finite_for_nonidentical_distributions():
    left = np.asarray([1.0, 0, 0, 0, 0, 0])
    right = np.asarray([0, 0, 0, 0, 0, 1.0])
    metrics = distribution_metrics(left, right)
    for name in ("kl_native_counterfactual", "js_divergence", "total_variation_distance"):
        assert np.isfinite(metrics[name])


def test_zero_hidden_recurrent_diagnostic_uses_snapshot_hidden_shape():
    actor, value = snapshot()
    audit_snapshot(actor, value)
    assert actor.hidden_shapes
    assert all(shape == value.hidden.shape for shape in actor.hidden_shapes)


def test_checkpoint_contract_mismatch_fails_closed(tmp_path):
    path = tmp_path / "mismatch.pt"
    torch.save({"metadata": {"observation_mode": "full"}}, path)
    with pytest.raises(RuntimeError, match="unsupported Partial evaluation checkpoint"):
        load_partial_checkpoint_for_evaluation(path, device="cpu")


def test_source_same_state_contract_mismatch_is_not_comparable():
    source = {"architecture_id": "a", "observation_contract_id": "full", "action_space": ["x"]}
    adapted = {"architecture_id": "a", "observation_contract_id": "partial", "action_space": ["x"]}
    result = source_comparability(source, adapted)
    assert result["status"] == "NOT_COMPARABLE_SAME_STATE"
    assert not result["strict_loading_or_adapter_bypass_used"]


def test_oracle_mode_cannot_be_selected_as_factual_or_default():
    args = parser().parse_args(["--action-checkpoint", "unused.pt"])
    assert args.tests == list(DEFAULT_TESTS)
    assert "bounds" not in args.tests
    assert not args.allow_oracle_upper_bound


def test_per_decision_probabilities_sum_to_one():
    actor, value = snapshot()
    probabilities = read_only_probabilities(
        actor, value.observation, value.hidden, value.mask, value.available
    )
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-6)


def test_eval_only_raw_logit_hook_reconstructs_probabilities():
    class CanonicalHeadActor:
        def __init__(self):
            linear = torch.nn.Linear(6, 6, bias=False)
            with torch.no_grad():
                linear.weight.copy_(torch.eye(6))
            self.act = type("Act", (), {})()
            self.act.action_out = type("ActionOut", (), {})()
            self.act.action_out.linear = linear

        def get_probs(self, observation, hidden, mask, available):
            del observation, mask
            features = torch.as_tensor(hidden, dtype=torch.float32).reshape(1, -1)[:, :6]
            logits = self.act.action_out.linear(features)
            available_tensor = torch.as_tensor(available).reshape(1, -1)
            probabilities = torch.softmax(
                logits.masked_fill(available_tensor <= 0, -1e10), dim=-1,
            )
            return probabilities, torch.as_tensor(hidden, dtype=torch.float32)

    actor = CanonicalHeadActor()
    hidden = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
    probabilities, logits, candidate = read_only_logits_probabilities_and_hidden(
        actor,
        np.zeros((8, 5, 25), dtype=np.float32),
        hidden,
        np.ones((1, 1), dtype=np.float32),
        np.ones(6, dtype=np.float32),
    )

    assert np.array_equal(logits, np.arange(6, dtype=np.float64))
    shifted = logits - logits.max()
    assert np.allclose(probabilities, np.exp(shifted) / np.exp(shifted).sum())
    assert np.array_equal(candidate, hidden)
