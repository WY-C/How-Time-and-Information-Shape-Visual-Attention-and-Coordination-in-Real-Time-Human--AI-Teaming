import pytest

from final_model.common.mappo import entropy_coefficient
from final_model.common.reward import hsp_reward_shaping_factor
from final_model.server_parallel.parallel_action_training import (
    continuation_checkpoint_schedule,
    validate_partial_resume_continuation,
)


START = 1_000_000
TARGET = 20_000_000
UPDATE_SIZE = 40_000
INTERVAL = 1_000_000


def stored_contract(**overrides):
    value = {
        "configured_total_steps": START,
        "checkpoint_interval": 200_000,
    }
    value.update(overrides)
    return value


def validate(*, target=TARGET, interval=INTERVAL, contract=None):
    return validate_partial_resume_continuation(
        stored_contract() if contract is None else contract,
        stored_environment_steps=START,
        requested_total_steps=target,
        requested_checkpoint_interval=interval,
        update_size=UPDATE_SIZE,
    )


def test_old_same_config_resume_operational_fields_still_pass():
    result = validate(target=START, interval=200_000)
    assert result["additional_environment_steps"] == 0
    assert result["additional_updates"] == 0


def test_total_steps_and_checkpoint_interval_may_increase_independently():
    result = validate()
    assert result["requested_total_steps"] == TARGET
    assert result["requested_checkpoint_interval"] == INTERVAL
    assert result["stored_checkpoint_interval"] == 200_000


def test_total_steps_below_stored_progress_fails():
    with pytest.raises(ValueError, match="below completed"):
        validate(target=START - UPDATE_SIZE)


@pytest.mark.parametrize("interval", (0, -1, 500_000))
def test_checkpoint_interval_must_be_positive_and_update_aligned(interval):
    with pytest.raises(ValueError, match="checkpoint-interval"):
        validate(interval=interval)


def test_target_must_be_a_whole_number_of_updates():
    with pytest.raises(ValueError, match="whole PPO update"):
        validate(target=TARGET - 1)


def test_1m_to_20m_is_exactly_19m_and_475_updates():
    result = validate()
    assert result["additional_environment_steps"] == 19_000_000
    assert result["additional_updates"] == 475
    assert START + result["additional_updates"] * UPDATE_SIZE == TARGET


def test_first_resumed_update_and_cumulative_schedules_are_exact():
    first_completed = START + UPDATE_SIZE
    coefficients = [0.2, 0.05, 0.01]
    horizons = [0, 10_000_000, 20_000_000]
    assert first_completed == 1_040_000
    assert entropy_coefficient(START, coefficients, horizons) == pytest.approx(0.185)
    assert hsp_reward_shaping_factor(START, 20_000_000) == pytest.approx(0.95)
    assert entropy_coefficient(first_completed, coefficients, horizons) == pytest.approx(
        0.1844
    )
    assert hsp_reward_shaping_factor(
        first_completed, 20_000_000
    ) == pytest.approx(0.948)
    assert entropy_coefficient(10_000_000, coefficients, horizons) == pytest.approx(
        0.05
    )
    assert entropy_coefficient(TARGET, coefficients, horizons) == pytest.approx(0.01)
    assert hsp_reward_shaping_factor(TARGET, 20_000_000) == pytest.approx(0.0)


def test_integer_million_checkpoint_schedule_is_exact_and_post_update_only():
    schedule = continuation_checkpoint_schedule(
        start_step=START,
        target_step=TARGET,
        checkpoint_interval=INTERVAL,
        update_size=UPDATE_SIZE,
    )
    assert schedule == tuple(range(2_000_000, 20_000_001, 1_000_000))
    assert START not in schedule
    assert len(schedule) == len(set(schedule)) == 19
    assert all((step - START) % UPDATE_SIZE == 0 for step in schedule)
