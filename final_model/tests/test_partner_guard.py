import ctypes
import types

import numpy as np
import pytest
import torch

import final_model.common.partners as partners_module
from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.common.partners import (
    FrozenPartnerPool,
    assert_optimizer_isolation,
    parameter_hash,
    tensor_guard_snapshot,
)


class ToyPartner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.register_buffer("persistent", torch.ones(2))


def frozen_pool(model: torch.nn.Module) -> FrozenPartnerPool:
    model.eval().requires_grad_(False)
    pool = object.__new__(FrozenPartnerPool)
    pool.actor = model
    pool.frozen_parameter_hash = parameter_hash(model)
    pool.frozen_tensor_guards = tensor_guard_snapshot(model)
    return pool


def test_normal_guard_records_all_required_metadata_without_value_copy(monkeypatch):
    pool = frozen_pool(ToyPartner())
    guards = pool.frozen_tensor_guards
    assert {guard.name for guard in guards} == {"parameter:weight", "buffer:persistent"}
    for guard in guards:
        assert guard.object_id
        assert guard.data_ptr
        assert isinstance(guard.version, int)
        assert guard.shape
        assert guard.dtype == torch.float32
        assert guard.device == torch.device("cpu")

    def forbidden_cpu(_tensor, *args, **kwargs):
        raise AssertionError("cheap guard called Tensor.cpu()")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    assert pool.assert_frozen() == pool.frozen_parameter_hash


def test_step_guard_detects_in_place_parameter_mutation():
    pool = frozen_pool(ToyPartner())
    with torch.no_grad():
        pool.actor.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="tensor guard changed"):
        pool.assert_frozen()


def test_step_guard_detects_parameter_object_replacement():
    pool = frozen_pool(ToyPartner())
    pool.actor.weight = torch.nn.Parameter(pool.actor.weight.detach().clone(), requires_grad=False)
    with pytest.raises(RuntimeError, match="tensor guard changed"):
        pool.assert_frozen()


def test_step_guard_detects_underlying_storage_replacement():
    pool = frozen_pool(ToyPartner())
    object_id = id(pool.actor.weight)
    old_pointer = pool.actor.weight.data_ptr()
    pool.actor.weight.data = pool.actor.weight.data.clone()
    assert id(pool.actor.weight) == object_id
    assert pool.actor.weight.data_ptr() != old_pointer
    with pytest.raises(RuntimeError, match="tensor guard changed"):
        pool.assert_frozen()


def test_rollout_full_hash_detects_raw_value_mutation_missed_by_version_guard():
    pool = frozen_pool(ToyPartner())
    before = tensor_guard_snapshot(pool.actor)
    cell = ctypes.c_float.from_address(pool.actor.weight.data_ptr())
    cell.value += 3.0
    assert tensor_guard_snapshot(pool.actor) == before
    assert pool.assert_frozen() == pool.frozen_parameter_hash
    with pytest.raises(RuntimeError, match="parameter hash changed"):
        pool.assert_frozen(full_hash=True)


def test_optimizer_isolation_fails_closed_on_partner_membership():
    partner = ToyPartner()
    unrelated = ToyPartner()
    good = torch.optim.Adam(unrelated.parameters(), lr=1e-3)
    assert_optimizer_isolation(partner, {"good_optimizer": good})
    bad = torch.optim.Adam(partner.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError, match="frozen partner parameter found in bad_optimizer"):
        assert_optimizer_isolation(partner, {"bad_optimizer": bad})


def test_partner_reload_rebuilds_hash_and_guard_baselines():
    backend = DirectOvercookedBackend(seed=17, horizon=0, episode_length=4, device="cpu")
    old_name = backend.active_partner
    old_actor_id = id(backend.partners.actor)
    old_guards = backend.partners.frozen_tensor_guards
    replacement = next(
        str(record["name"]) for record in backend.partners.manifest
        if record["name"] != old_name
    )
    backend.partners.requested = replacement
    backend.reset()
    assert backend.active_partner == replacement
    assert id(backend.partners.actor) != old_actor_id
    assert backend.partners.frozen_tensor_guards != old_guards
    assert backend.partners.assert_frozen(full_hash=True) == backend.partners.frozen_parameter_hash


def test_env_step_guard_never_calls_full_hash(monkeypatch):
    backend = DirectOvercookedBackend(seed=18, horizon=0, episode_length=4, device="cpu")
    calls = 0
    original = partners_module.parameter_hash

    def counted_hash(module):
        nonlocal calls
        calls += 1
        return original(module)

    monkeypatch.setattr(partners_module, "parameter_hash", counted_hash)
    backend.step(4)
    assert calls == 0
    assert backend.partner_step_guard_checks == 2
    backend.verify_partner_parameters()
    assert calls == 1
    assert backend.partner_full_hash_checks == 1


def test_optimized_guard_matches_legacy_hash_guard_trajectory():
    condition = dict(
        seed=29, horizon=0, episode_length=32, device="cpu", partner="mep5_init"
    )
    optimized = DirectOvercookedBackend(**condition)
    legacy = DirectOvercookedBackend(**condition)
    legacy_baseline = legacy.partners.frozen_parameter_hash

    def legacy_full_hash_guard(self, *, full_hash=False):
        actual = parameter_hash(self.actor)
        if actual != legacy_baseline:
            raise RuntimeError("legacy partner hash changed")
        return actual

    legacy.partners.assert_frozen = types.MethodType(legacy_full_hash_guard, legacy.partners)
    optimized_actions, legacy_actions = [], []
    optimized_action = optimized.partners.action
    legacy_action = legacy.partners.action

    def record_optimized(observation, available):
        action = optimized_action(observation, available)
        optimized_actions.append(action)
        return action

    def record_legacy(observation, available):
        action = legacy_action(observation, available)
        legacy_actions.append(action)
        return action

    optimized.partners.action = record_optimized
    legacy.partners.action = record_legacy
    learner_actions = (0, 2, 4, 5, 1, 3, 4, 0, 5, 2, 1, 4)
    optimized_trace, legacy_trace = [], []
    for learner_action in learner_actions:
        assert np.array_equal(optimized.canonical_pair()[1], legacy.canonical_pair()[1])
        first = optimized.step(learner_action)
        second = legacy.step(learner_action)
        optimized_trace.append(
            (optimized.state.to_dict(), first["reward"], first["sparse_team_reward"], first["native_shaped_reward"])
        )
        legacy_trace.append(
            (legacy.state.to_dict(), second["reward"], second["sparse_team_reward"], second["native_shaped_reward"])
        )

    assert optimized.active_partner == legacy.active_partner == "mep5_init"
    assert optimized_actions == legacy_actions
    assert optimized_trace == legacy_trace
