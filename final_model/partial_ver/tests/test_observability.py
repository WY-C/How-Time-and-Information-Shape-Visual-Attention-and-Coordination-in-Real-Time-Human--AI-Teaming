import random

import numpy as np
import pytest
import torch

from final_model.partial_ver.conditions import PartialCondition
from final_model.partial_ver.observability import (
    accumulate_partial_metrics,
    condition_metric_id,
    finalize_partial_metrics,
    flatten_partial_metric_namespaces,
    new_partial_metric_totals,
    partial_observation_statistics,
    validate_partial_metric_totals,
)


def grid():
    return tuple(
        PartialCondition(decision_time, horizon, "plus")
        for decision_time in (200, 400, 600)
        for horizon in (0, 2, 4)
    )


def assert_numpy_state_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_shared_time_horizon_grid_has_all_time_and_cell_ids():
    totals = new_partial_metric_totals(grid())
    assert tuple(totals["time_metrics"]) == ("t200", "t400", "t600")
    assert tuple(totals["horizon_metrics"]) == ("h0", "h2", "h4")
    assert set(totals["condition_metrics"]) == {
        f"t{decision_time}_h{horizon}_plus"
        for decision_time in (200, 400, 600)
        for horizon in (0, 2, 4)
    }
    assert all(condition_metric_id(value) in totals["condition_metrics"] for value in grid())


def test_exact_marginals_conserve_and_do_not_average_rates():
    h0 = PartialCondition(200, 0, "plus")
    h2 = PartialCondition(200, 2, "plus")
    totals = new_partial_metric_totals((h0, h2))
    conditions = (h0, h2, h2, h2)
    rewards = np.asarray(
        [[4.0, 40.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    accumulate_partial_metrics(
        totals,
        conditions=conditions,
        rewards=rewards,
        deliveries=[4, 0, 0, 0],
        dones=[True, True, True, True],
        acquisitions=[4, 0, 0, 0],
        actor_active=[False, True, True, True],
        actionnet_queries=[4, 0, 0, 0],
        gaze_requests=[4, 1, 1, 1],
        same_target_acquisitions=[2, 0, 0, 0],
        new_target_acquisitions=[2, 0, 0, 0],
        valid_memory_tiles=[80, 0, 0, 0],
        freshness_sample_counts=[160, 0, 0, 0],
        freshness_normalized_sums=[80.0, 0.0, 0.0, 0.0],
        saccade_distance_sums_degrees=[10.0, 0.0, 0.0, 0.0],
    )
    metrics = finalize_partial_metrics(totals)
    time = metrics["time_metrics"]["t200"]
    assert time["forced_stay_rate"] == 1 / 4
    assert time["deliveries_per_episode"] == 1.0
    assert time["acquisitions_per_decision"] == 1.0
    assert time["same_target_rate"] == 0.5
    assert time["new_target_rate"] == 0.5
    assert time["valid_memory_fraction"] == 0.5
    assert time["mean_freshness"] == 0.5
    assert time["mean_saccade_distance"] == 5.0
    validate_partial_metric_totals(
        metrics,
        optimization_reward=4.0,
        sparse_reward=40.0,
        deliveries=4,
        episodes=4,
        decisions=4,
        forced_stays=1,
        acquisitions=4,
    )
    flattened = flatten_partial_metric_namespaces(metrics)
    assert flattened["t200/forced_stay_rate"] == 1 / 4
    assert flattened["h0/deliveries_per_episode"] == 4.0
    assert flattened["cell/t200_h2_plus/acquisitions_per_decision"] == 0.0


def test_fixed_t600_still_has_first_class_time_namespace():
    condition = PartialCondition(600, 4, "plus")
    metrics = finalize_partial_metrics(new_partial_metric_totals((condition,)))
    assert set(metrics["time_metrics"]) == {"t600"}
    assert set(metrics["horizon_metrics"]) == {"h4"}
    assert set(metrics["condition_metrics"]) == {"t600_h4_plus"}
    assert metrics["time_metrics"]["t600"]["decisions"] == 0
    assert metrics["time_metrics"]["t600"]["forced_stay_rate"] is None


def test_observation_statistics_use_existing_freshness_without_mutation():
    observation = np.zeros((8, 5, 25), dtype=np.float32)
    observation[:4, :, 20] = 127.5
    before = observation.copy()
    result = partial_observation_statistics((observation,))
    assert result == {
        "valid_memory_tiles": 20,
        "freshness_sample_count": 40,
        "freshness_normalized_sum": 10.0,
    }
    assert np.array_equal(observation, before)


def test_aggregation_is_rng_neutral_and_reconciliation_fails_closed():
    random.seed(601)
    np.random.seed(602)
    torch.manual_seed(603)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    condition = PartialCondition(600, 0, "plus")
    totals = new_partial_metric_totals((condition,))
    accumulate_partial_metrics(
        totals,
        conditions=(condition,),
        rewards=np.zeros((1, 2), dtype=np.float64),
        deliveries=[0],
        dones=[False],
        acquisitions=[1],
        actor_active=[True],
        actionnet_queries=[1],
        gaze_requests=[1],
        same_target_acquisitions=[1],
        new_target_acquisitions=[0],
    )
    metrics = finalize_partial_metrics(totals)
    with pytest.raises(RuntimeError, match="do not reconcile"):
        validate_partial_metric_totals(
            metrics,
            optimization_reward=1.0,
            sparse_reward=0.0,
            deliveries=0,
            episodes=0,
            decisions=1,
            forced_stays=0,
            acquisitions=1,
        )
    assert random.getstate() == python_before
    assert_numpy_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
