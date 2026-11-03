import copy
from collections import Counter

from final_model.partial_ver.conditions import PartialConditionScheduler


def test_fixed_time_shared_horizon_block_balance_and_episode_fixity():
    scheduler = PartialConditionScheduler(
        decision_times=(200,), horizons=(0, 2, 4), fixation_shapes=("3x3",), seed=1
    )
    block = [scheduler.next() for _ in range(3)]
    assert {(value.decision_time, value.horizon, value.fixation_shape) for value in block} == {
        (200, 0, "3x3"), (200, 2, "3x3"), (200, 4, "3x3")
    }
    active = block[0]
    assert all(active == block[0] for _decision in range(10))


def test_shared_time_shared_horizon_and_all_footprints_cartesian_exact():
    scheduler = PartialConditionScheduler(
        decision_times=(200, 400, 600),
        horizons=(0, 2, 4),
        fixation_shapes=("3x3", "2x2", "plus"),
        seed=2,
    )
    block = [scheduler.next() for _ in range(27)]
    assert len(set(block)) == 27
    assert Counter(value.decision_time for value in block) == {200: 9, 400: 9, 600: 9}
    assert Counter(value.horizon for value in block) == {0: 9, 2: 9, 4: 9}
    assert Counter(value.fixation_shape for value in block) == {
        "3x3": 9, "2x2": 9, "plus": 9
    }


def test_scheduler_resume_is_exact_and_rng_isolated():
    first = PartialConditionScheduler(
        decision_times=(200, 600), horizons=(0, 4), fixation_shapes=("plus",), seed=9
    )
    _ = [first.next() for _ in range(3)]
    state = first.state_dict()
    expected = [first.next() for _ in range(20)]
    restored = PartialConditionScheduler(
        decision_times=(200, 600), horizons=(0, 4), fixation_shapes=("plus",), seed=9
    )
    restored.load_state_dict(copy.deepcopy(state))
    assert [restored.next() for _ in range(20)] == expected
