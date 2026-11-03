from types import SimpleNamespace

import numpy as np
import pytest

from final_model.partial_ver.conditions import PartialCondition
from final_model.partial_ver.observability import (
    accumulate_partial_metrics,
    finalize_partial_metrics,
    new_partial_metric_totals,
)
from final_model.server_parallel.collision_observability import (
    accumulate_collision_totals,
    finalize_collision_rates,
    human_compatible_joint_transition_collision,
    initialize_collision_totals,
    validate_collision_conservation,
)


class FakeMdp:
    DELTAS = {"left": (-1, 0), "right": (1, 0), "stay": (0, 0)}

    def __init__(self):
        self.calls = []

    def _move_if_direction(self, position, orientation, action):
        self.calls.append((tuple(position), orientation, action))
        dx, dy = self.DELTAS[action]
        return (position[0] + dx, position[1] + dy), orientation

    @staticmethod
    def is_transition_collision(old_positions, candidate_positions):
        return (
            candidate_positions[0] == candidate_positions[1]
            or candidate_positions == tuple(reversed(old_positions))
        )


def test_human_compatible_collision_uses_pretransition_candidates():
    mdp = FakeMdp()
    state = SimpleNamespace(players=(
        SimpleNamespace(position=(0, 0), orientation="east"),
        SimpleNamespace(position=(2, 0), orientation="west"),
    ))
    assert human_compatible_joint_transition_collision(
        mdp, state, ("right", "left")
    )
    assert mdp.calls == [
        ((0, 0), "east", "right"),
        ((2, 0), "west", "left"),
    ]


def test_collision_marginals_and_nine_cells_conserve_exactly():
    conditions = tuple(
        PartialCondition(time, horizon, "plus")
        for time in (200, 400, 600)
        for horizon in (0, 2, 4)
    )
    totals = new_partial_metric_totals(conditions)
    initialize_collision_totals(totals)
    accumulate_partial_metrics(
        totals,
        conditions=conditions,
        rewards=np.zeros((9, 2), dtype=np.float64),
        deliveries=np.zeros(9, dtype=np.int64),
        dones=np.zeros(9, dtype=bool),
        acquisitions=np.zeros(9, dtype=np.int64),
        actor_active=np.ones(9, dtype=bool),
    )
    count = accumulate_collision_totals(
        totals,
        conditions=conditions,
        collisions=(1, 0, 1, 0, 1, 0, 1, 0, 1),
    )
    metrics = finalize_partial_metrics(totals)
    finalize_collision_rates(metrics)
    validate_collision_conservation(metrics, collision_count=count, decisions=9)

    assert count == 5
    assert metrics["time_metrics"]["t200"]["collision_count"] == 2
    assert metrics["horizon_metrics"]["h0"]["collision_count"] == 2
    assert metrics["condition_metrics"]["t600_h4_plus"]["collision_count"] == 1
    assert metrics["condition_metrics"]["t600_h4_plus"]["collision_rate"] == 1.0


def test_collision_conservation_remains_fail_closed():
    condition = PartialCondition(600, 4, "plus")
    totals = new_partial_metric_totals((condition,))
    initialize_collision_totals(totals)
    accumulate_partial_metrics(
        totals,
        conditions=(condition,),
        rewards=np.zeros((1, 2), dtype=np.float64),
        deliveries=(0,),
        dones=(False,),
        acquisitions=(0,),
        actor_active=(True,),
    )
    accumulate_collision_totals(
        totals, conditions=(condition,), collisions=(1,)
    )
    metrics = finalize_partial_metrics(totals)
    finalize_collision_rates(metrics)
    metrics["horizon_metrics"]["h4"]["collision_count"] = 0
    with pytest.raises(RuntimeError, match="do not reconcile"):
        validate_collision_conservation(
            metrics, collision_count=1, decisions=1
        )
