from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from final_model.eval.counterfactual import CounterfactualCollector, distribution_metrics


class FakeActor:
    def get_probs(self, observation, hidden, mask, available):
        preview = torch.as_tensor(observation, dtype=torch.float32)[..., 21:].sum()
        logits = torch.tensor([[1., 2., 3., 4., 5., 6.]])
        logits[0, 5] += preview / 1000
        return torch.softmax(logits, dim=-1), torch.as_tensor(hidden) + 1


def state():
    learner = SimpleNamespace(held_object=None, position=(1, 1), orientation=(1, 0))
    partner = SimpleNamespace(held_object=None, position=(0, 0), orientation=(1, 0))
    return SimpleNamespace(players=(partner, learner), objects={})


def mdp():
    return SimpleNamespace(terrain_mtx=[list("    ") for _ in range(4)])


def test_native_native_kl_zero():
    metrics = distribution_metrics(np.ones(6) / 6, np.ones(6) / 6)
    assert metrics["kl_native_counterfactual"] == 0


def test_native_native_js_zero():
    metrics = distribution_metrics(np.ones(6) / 6, np.ones(6) / 6)
    assert metrics["js_divergence"] == 0


def test_metrics_include_all_six_actions_and_interact_delta():
    metrics = distribution_metrics(np.ones(6) / 6, np.array([.1, .1, .1, .1, .1, .5]))
    assert "native_p_interact" in metrics
    assert np.isclose(metrics["delta_P_INTERACT"], .5 - 1 / 6)


def test_counterfactual_does_not_mutate_native_hidden():
    actor = FakeActor()
    observation = np.zeros((8, 5, 25), dtype=np.float32)
    observation[..., 21:25] = 1
    hidden = np.zeros((1, 1, 3), dtype=np.float32)
    before = hidden.copy()
    native, _ = actor.get_probs(observation[None], hidden, np.zeros((1, 1)), np.ones((1, 6)))
    collector = CounterfactualCollector()
    collector.observe(
        actor=actor, observation=observation, hidden=hidden,
        mask=np.zeros((1, 1)), available=np.ones(6),
        native_probabilities=native.numpy()[0], evaluation_horizon=4,
        episode=0, timestep=0, state=state(), mdp=mdp(),
    )
    assert np.array_equal(hidden, before)


def test_h4_counterfactual_generates_four_modes():
    actor = FakeActor(); observation = np.zeros((8, 5, 25), dtype=np.float32)
    native, _ = actor.get_probs(observation[None], np.zeros((1, 1, 3)), np.zeros((1, 1)), np.ones((1, 6)))
    collector = CounterfactualCollector()
    collector.observe(actor=actor, observation=observation, hidden=np.zeros((1, 1, 3)), mask=np.zeros((1, 1)), available=np.ones(6), native_probabilities=native.numpy()[0], evaluation_horizon=4, episode=0, timestep=0, state=state(), mdp=mdp())
    assert [row["counterfactual_preview_mode"] for row in collector.rows] == ["native", "h12", "h1", "none"]


def test_counterfactual_summary_is_finite():
    actor = FakeActor(); observation = np.zeros((8, 5, 25), dtype=np.float32)
    native, _ = actor.get_probs(observation[None], np.zeros((1, 1, 3)), np.zeros((1, 1)), np.ones((1, 6)))
    collector = CounterfactualCollector()
    collector.observe(actor=actor, observation=observation, hidden=np.zeros((1, 1, 3)), mask=np.zeros((1, 1)), available=np.ones(6), native_probabilities=native.numpy()[0], evaluation_horizon=4, episode=0, timestep=0, state=state(), mdp=mdp())
    assert all(np.isfinite(values["mean_kl"]) for values in collector.summary()["all_decisions"].values())
