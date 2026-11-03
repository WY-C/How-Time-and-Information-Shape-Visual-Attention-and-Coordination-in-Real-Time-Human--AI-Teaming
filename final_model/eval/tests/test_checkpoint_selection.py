from __future__ import annotations

import pytest

from final_model.eval.checkpoint_selection import select_checkpoints
from final_model.eval.common import (
    FINAL_ANALYSIS_SEED_START,
    SELECTION_SEED_START,
    assert_seed_partitions_distinct,
)


def row(path, step, mean, horizon=4, **extra):
    return {
        "checkpoint_path": path, "checkpoint_sha256": path,
        "checkpoint_completed_step": step, "evaluation_horizon": horizon,
        "mean_deliveries": mean, "std_deliveries": 1,
        "mean_sparse_team_reward": mean * 20, **extra,
    }


def test_behavior_metrics_cannot_affect_selection():
    base = [row("a", 1, 9, handoff_success_rate=0), row("b", 2, 10, handoff_success_rate=1)]
    flipped = [row("a", 1, 9, handoff_success_rate=999), row("b", 2, 10, handoff_success_rate=-999)]
    assert [x["checkpoint_sha256"] for x in select_checkpoints(base, shared=False)] == [x["checkpoint_sha256"] for x in select_checkpoints(flipped, shared=False)]


def test_95_percent_region_and_neighbor_stability():
    ranked = select_checkpoints([row("a", 1, 9.5), row("b", 2, 10), row("c", 3, 8)], shared=False)
    by_name = {item["checkpoint_sha256"]: item for item in ranked}
    assert by_name["a"]["high_performing"] and by_name["b"]["high_performing"]
    assert by_name["a"]["locally_stable"] and by_name["b"]["locally_stable"]
    assert not by_name["c"]["high_performing"]


def test_shared_requires_and_reports_all_horizons():
    rows = [row("s", 7, mean, horizon=horizon) for horizon, mean in [(0, 3), (2, 5), (4, 7)]]
    selected = select_checkpoints(rows, shared=True)[0]
    assert selected["aggregate_mean_deliveries"] == 5
    assert selected["minimum_condition_mean_deliveries"] == 3
    assert selected["h0_mean_deliveries"] == 3


def test_shared_missing_horizon_fails():
    with pytest.raises(ValueError, match="H0/H2/H4"):
        select_checkpoints([row("s", 7, 3, horizon=0)], shared=True)


def test_selection_and_final_seeds_are_separate():
    assert SELECTION_SEED_START != FINAL_ANALYSIS_SEED_START
    assert_seed_partitions_distinct(SELECTION_SEED_START, FINAL_ANALYSIS_SEED_START)


def test_duplicate_seed_partitions_fail():
    with pytest.raises(ValueError):
        assert_seed_partitions_distinct(1, 1)
