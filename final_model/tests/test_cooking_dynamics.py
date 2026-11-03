import copy
import csv
import json
from pathlib import Path

import pytest

from final_model.common.overcooked_backend import (
    COOKING_DYNAMICS_ID,
    ONION_COOK_TICKS_PER_ENVIRONMENT_STEP,
    SOUP_COOKING_TIME,
    SOUP_ITEMS_REQUIRED,
    DirectOvercookedBackend,
    configure_experiment_cooking_dynamics,
    cooking_contract,
)
from final_model.common.zsceval_compat import install_zsceval
from final_model.eval import rollout as evaluation_rollout
from final_model.partial_ver.memory import PersistentPartialMemory


def _gridworld_and_object_state():
    install_zsceval()
    from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
        ObjectState,
        OvercookedGridworld,
    )

    return OvercookedGridworld, ObjectState


def test_experiment_cooking_contract_is_explicit():
    assert cooking_contract() == {
        "cooking_dynamics_id": COOKING_DYNAMICS_ID,
        "onion_cook_ticks_per_environment_step": (
            ONION_COOK_TICKS_PER_ENVIRONMENT_STEP
        ),
        "soup_cooking_time": SOUP_COOKING_TIME,
        "soup_items_required": SOUP_ITEMS_REQUIRED,
    }
    assert ONION_COOK_TICKS_PER_ENVIRONMENT_STEP == SOUP_COOKING_TIME == 20
    assert SOUP_ITEMS_REQUIRED == 3


def test_only_a_full_onion_soup_in_a_pot_becomes_ready():
    Gridworld, ObjectState = _gridworld_and_object_state()
    mdp = configure_experiment_cooking_dynamics(
        Gridworld.from_layout_name("random3")
    )
    state = mdp.get_standard_start_state()
    state.add_object(ObjectState("soup", (3, 0), ("onion", 3, 0)))
    state.add_object(ObjectState("soup", (4, 0), ("tomato", 3, 0)))
    state.add_object(ObjectState("soup", (2, 0), ("onion", 3, 0)))
    state.add_object(ObjectState("soup", (5, 0), ("onion", 2, 0)))

    mdp.step_environment_effects(state)

    assert state.get_object((3, 0)).state == ("onion", 3, 20)
    assert state.get_object((4, 0)).state == ("tomato", 3, 1)
    assert state.get_object((2, 0)).state == ("onion", 3, 0)
    assert state.get_object((5, 0)).state == ("onion", 2, 0)


def test_third_onion_is_ready_after_its_joint_transition_and_pickable_next_step():
    _Gridworld, ObjectState = _gridworld_and_object_state()
    backend = DirectOvercookedBackend(
        seed=71, horizon=0, episode_length=8, device="cpu", partner="mep5_init"
    )
    pot = (3, 0)
    learner = backend.state.players[1]
    backend.state.add_object(ObjectState("soup", pot, ("onion", 2, 0)))
    learner.set_object(ObjectState("onion", learner.position))

    before = backend.begin_partner_step()
    backend.step_with_partner_action(5, 4, partner_hash_before=before)

    learner = backend.state.players[1]
    assert backend.state.get_object(pot).state == ("onion", 3, 20)
    assert not learner.has_object()
    ready_observation = backend.canonical_pair()[1]
    assert ready_observation[pot[0], pot[1], 16] == 20 * 255

    memory = PersistentPartialMemory()
    memory.reset(ready_observation)
    memory.acquire(
        ready_observation,
        backend.preview_planes().transpose(1, 2, 0),
        [pot],
    )
    assert memory.snapshot()[pot[0], pot[1], 16] == 20 * 255

    learner.set_object(ObjectState("dish", learner.position))
    before = backend.begin_partner_step()
    backend.step_with_partner_action(5, 4, partner_hash_before=before)

    learner = backend.state.players[1]
    assert not backend.state.has_object(pot)
    assert learner.get_object().state == ("onion", 3, 20)


def test_backend_checkpoint_state_records_and_validates_cooking_contract():
    backend = DirectOvercookedBackend(
        seed=72, horizon=0, episode_length=4, device="cpu", partner="mep5_init"
    )
    state = backend.training_state()
    assert {key: state[key] for key in cooking_contract()} == cooking_contract()

    restored = DirectOvercookedBackend(
        seed=72, horizon=0, episode_length=4, device="cpu", partner="mep5_init"
    )
    restored.load_training_state(copy.deepcopy(state))
    assert restored.state.to_dict() == backend.state.to_dict()

    legacy = copy.deepcopy(state)
    legacy.pop("cooking_dynamics_id")
    with pytest.raises(RuntimeError, match="backend resume contract mismatch"):
        restored.load_training_state(legacy)


def test_environment_preview_evaluation_and_human_decision_cooking_parity(
    monkeypatch,
):
    _Gridworld, ObjectState = _gridworld_and_object_state()
    backend = DirectOvercookedBackend(
        seed=73, horizon=4, episode_length=8, device="cpu", partner="mep5_init"
    )
    pot = (3, 0)
    learner = backend.state.players[1]
    backend.state.add_object(ObjectState("soup", pot, ("onion", 2, 0)))
    learner.set_object(ObjectState("onion", learner.position))
    before = backend.begin_partner_step()
    backend.step_with_partner_action(5, 4, partner_hash_before=before)
    assert backend.state.get_object(pot).state == ("onion", 3, 20)

    preview_inputs = []
    original_encoding = backend.mdp.lossless_state_encoding

    def capture_preview_input(state):
        preview_inputs.append(state.get_object(pot).state)
        return original_encoding(state)

    def forbidden_preview_transition(*_args, **_kwargs):
        raise AssertionError("H1-H4 preview must not run an environment transition")

    monkeypatch.setattr(backend.mdp, "lossless_state_encoding", capture_preview_input)
    monkeypatch.setattr(
        backend.mdp, "get_state_transition", forbidden_preview_transition
    )
    planes = backend.preview_planes()
    assert planes.shape == (4, 8, 5)
    assert planes.sum(axis=(1, 2)).tolist() == [255.0] * 4
    assert preview_inputs == [("onion", 3, 20)]

    assert evaluation_rollout.DirectOvercookedBackend is DirectOvercookedBackend
    evaluation_backend = evaluation_rollout.DirectOvercookedBackend(
        seed=74, horizon=0, episode_length=8, device="cpu", partner="mep5_init"
    )
    evaluation_learner = evaluation_backend.state.players[1]
    evaluation_backend.state.add_object(
        ObjectState("soup", pot, ("onion", 2, 0))
    )
    evaluation_learner.set_object(
        ObjectState("onion", evaluation_learner.position)
    )
    before = evaluation_backend.begin_partner_step()
    evaluation_backend.step_with_partner_action(5, 4, partner_hash_before=before)
    assert evaluation_backend.state.get_object(pot).state == ("onion", 3, 20)

    log_path = (
        Path(__file__).resolve().parents[2]
        / "data" / "raw" / "experiment" / "P08" / "step.csv"
    )
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = {
            int(row["step"]): row
            for row in csv.DictReader(handle)
            if row["trial_id"] == "20260730-150243_P08_trial-01_C8_B"
        }
    before_human = json.loads(rows[14]["pot_states"])
    after_human = json.loads(rows[15]["pot_states"])
    assert json.loads(rows[14]["joint_action"])[0] == "interact"
    assert before_human["onion"]["2_items"] == [[4, 2]]
    assert after_human["onion"]["ready"] == [[4, 2]]
    assert json.loads(rows[15]["agent_held_object"]) is None
