from __future__ import annotations

from types import SimpleNamespace

from final_model.eval.task_events import TaskEventLogger


COUNTER = (2, 2)


def obj(name):
    return SimpleNamespace(name=name)


def player(*, held=None, position=(0, 0), orientation=(0, 1)):
    return SimpleNamespace(
        held_object=None if held is None else obj(held),
        position=position, orientation=orientation,
    )


def state(partner_held=None, learner_held=None, objects=None):
    return SimpleNamespace(
        players=(
            player(held=partner_held, position=(1, 2), orientation=(1, 0)),
            player(held=learner_held, position=(2, 3), orientation=(0, -1)),
        ),
        objects={} if objects is None else objects,
    )


def mdp():
    terrain = [list("        ") for _ in range(5)]
    terrain[2][2] = "X"
    return SimpleNamespace(terrain_mtx=terrain)


def place(logger, timestep=0):
    logger.observe(
        timestep=timestep, pre_state=state(learner_held="onion"),
        post_state=state(objects={COUNTER: obj("onion")}),
        actions=(4, 5), mdp=mdp(), result={"deliveries": 0},
    )


def partner_pickup(logger, timestep=1):
    logger.observe(
        timestep=timestep, pre_state=state(objects={COUNTER: obj("onion")}),
        post_state=state(partner_held="onion"),
        actions=(5, 4), mdp=mdp(), result={"deliveries": 0},
    )


def test_learner_counter_placement_detected():
    logger = TaskEventLogger(episode=0, evaluation_horizon=4)
    place(logger)
    assert logger.counts["learner_onion_counter_place"] == 1


def test_partner_pickup_classified_as_handoff():
    logger = TaskEventLogger(episode=0, evaluation_horizon=4)
    place(logger); partner_pickup(logger)
    assert logger.counts["partner_pickup_of_learner_placed_onion"] == 1


def test_learner_pickup_classified_as_self_repickup():
    logger = TaskEventLogger(episode=0, evaluation_horizon=4)
    place(logger)
    logger.observe(
        timestep=1, pre_state=state(objects={COUNTER: obj("onion")}),
        post_state=state(learner_held="onion"), actions=(4, 5), mdp=mdp(),
        result={"deliveries": 0},
    )
    assert logger.counts["learner_repickup_of_own_placed_onion"] == 1


def test_new_onion_same_counter_gets_new_token():
    logger = TaskEventLogger(episode=0, evaluation_horizon=4)
    place(logger, 0); partner_pickup(logger, 1); place(logger, 2)
    tokens = [event["provenance_token"] for event in logger.events if event["event_type"] == "learner_onion_counter_place"]
    assert tokens == [1, 2]


def test_unrelated_counter_onion_not_attributed():
    logger = TaskEventLogger(episode=0, evaluation_horizon=4)
    partner_pickup(logger)
    assert logger.counts["partner_pickup_of_learner_placed_onion"] == 0
    assert logger.counts["partner_repickup_of_own_placed_onion"] == 0


def test_ambiguous_transition_counted():
    logger = TaskEventLogger(episode=0, evaluation_horizon=4)
    logger.observe(
        timestep=0, pre_state=state(), post_state=state(objects={COUNTER: obj("onion")}),
        actions=(4, 4), mdp=mdp(), result={"deliveries": 0},
    )
    assert logger.counts["ambiguous_event_count"] == 1


def test_derived_rates_have_requested_denominators():
    logger = TaskEventLogger(episode=0, evaluation_horizon=4)
    logger.counts.update({
        "learner_onion_counter_place": 4,
        "partner_pickup_of_learner_placed_onion": 3,
        "learner_repickup_of_own_placed_onion": 1,
        "learner_onion_pickup": 2,
        "learner_onion_pot_insert": 1,
        "learner_interaction_count": 3,
        "partner_interaction_count": 2,
        "team_delivery": 2,
    })
    summary = logger.summary()
    assert summary["handoff_success_rate"] == .75
    assert summary["self_repickup_rate"] == .25
    assert summary["learner_direct_pot_rate"] == .5
    assert summary["interaction_efficiency"] == .4


def test_zero_denominators_are_safe():
    summary = TaskEventLogger(episode=0, evaluation_horizon=0).summary()
    assert summary["handoff_success_rate"] == 0
    assert summary["interaction_efficiency"] == 0
