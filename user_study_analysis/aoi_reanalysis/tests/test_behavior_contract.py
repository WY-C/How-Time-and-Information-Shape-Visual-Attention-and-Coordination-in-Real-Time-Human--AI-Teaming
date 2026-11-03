import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "02_behavior_reanalysis.py"
spec = importlib.util.spec_from_file_location("behavior_reanalysis", MODULE_PATH)
behavior = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(behavior)


def _trial_df():
    return pd.DataFrame([
        {
            "trial_id": "T1", "pid": "P01", "Time": 200, "Horizon": 0, "Layout": "A",
            "score_final": 100, "num_collisions": 2, "num_human_idle": 1,
            "decision_steps_completed": 2,
        },
        {
            "trial_id": "T2", "pid": "P01", "Time": 200, "Horizon": 0, "Layout": "B",
            "score_final": 120, "num_collisions": 4, "num_human_idle": 0,
            "decision_steps_completed": 2,
        },
    ])


def _step_df():
    return pd.DataFrame([
        {"trial_id": "T1", "step": 0, "state_onset_ms": 0, "response_time_ms": 80.0, "is_human_idle": 0, "human_action_due": 1},
        {"trial_id": "T1", "step": 1, "state_onset_ms": 200, "response_time_ms": np.nan, "is_human_idle": 1, "human_action_due": 1},
        {"trial_id": "T2", "step": 0, "state_onset_ms": 0, "response_time_ms": 90.0, "is_human_idle": 0, "human_action_due": 1},
        {"trial_id": "T2", "step": 1, "state_onset_ms": 200, "response_time_ms": 100.0, "is_human_idle": 0, "human_action_due": 1},
    ])


def test_timeout_rt_uses_budget_and_trial_timeout_definition_is_preserved():
    trial = _trial_df()
    decisions = behavior.build_decision_values(_step_df(), trial)
    timeout_row = decisions[(decisions["trial_id"] == "T1") & (decisions["timeout"] == 1)].iloc[0]
    assert timeout_row["response_time_with_timeout_ms"] == 200

    raw = behavior.build_trial_values(trial, decisions)
    t1 = raw[raw["trial_id"] == "T1"].iloc[0]
    assert t1["timeout_rate"] == 0.5
    assert t1["step_timeout_rate"] == 0.5


def test_collision_is_raw_x_before_and_after_participant_aggregation():
    trial = _trial_df()
    decisions = behavior.build_decision_values(_step_df(), trial)
    raw = behavior.build_trial_values(trial, decisions)
    cells = behavior.participant_inputs(raw)
    assert "collision_log1p" not in cells.columns
    assert "num_collisions" in cells.columns
    assert cells.loc[0, "num_collisions"] == 3.0


def test_non_timeout_missing_rt_fails_closed():
    trial = _trial_df().iloc[[0]].copy()
    steps = _step_df().iloc[[0, 1]].copy()
    steps.loc[0, "response_time_ms"] = np.nan
    steps.loc[0, "is_human_idle"] = 0
    with pytest.raises(RuntimeError, match="non-timeout"):
        behavior.build_decision_values(steps, trial)
