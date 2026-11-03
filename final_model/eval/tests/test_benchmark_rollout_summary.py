import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from final_model.eval.benchmark_rollout import (
    BenchmarkRolloutConfig,
    _decision_behavior_summary,
    _learner_collision_fields,
    _segment_aoi_fields,
    _validate_config,
    _write_plus_aoi_outputs,
)


def _decision(*, decision, forced_stay, revised=False, latency=None):
    return {
        "T": 400,
        "H": 2,
        "decision": decision,
        "selected_action_input_time_ms": latency,
        "canonical_had_revision": revised,
        "temporal_had_action_change": revised,
        "intentional_stay": False,
        "forced_stay": forced_stay,
        "partner_collision": 0,
        "environment_block": 0,
        "collision_any": 0,
        "deliveries": 0,
        "sparse_team_reward": 0.0,
        "realized_transition": not forced_stay,
    }


def test_revision_rate_and_latency_exclude_forced_stays():
    decisions = pd.DataFrame([
        _decision(decision=0, forced_stay=False, revised=True, latency=100.0),
        _decision(decision=1, forced_stay=False, revised=False, latency=200.0),
        _decision(decision=2, forced_stay=True),
    ])

    row = _decision_behavior_summary(decisions).iloc[0]

    assert row.decisions == 3
    assert row.non_timeout_decisions == 2
    assert row.selected_action_latency_ms == 150.0
    assert row.canonical_revisions == 1
    assert row.canonical_revision_rate == 0.5
    assert row.forced_stay_rate == 1 / 3
    assert row.no_action_rate == 1 / 3


def test_all_timeout_cell_has_undefined_revision_rate():
    decisions = pd.DataFrame([
        _decision(decision=0, forced_stay=True),
        _decision(decision=1, forced_stay=True),
    ])

    row = _decision_behavior_summary(decisions).iloc[0]

    assert row.non_timeout_decisions == 0
    assert row.canonical_revisions == 0
    assert math.isnan(row.canonical_revision_rate)


def test_intentional_stay_is_included_in_no_action_rate():
    decisions = pd.DataFrame([
        {**_decision(decision=0, forced_stay=False, latency=100.0),
         "intentional_stay": True},
        _decision(decision=1, forced_stay=False, latency=120.0),
    ])

    row = _decision_behavior_summary(decisions).iloc[0]

    assert row.intentional_stay_rate == 0.5
    assert row.forced_stay_rate == 0.0
    assert row.no_action_rate == 0.5


def test_plus_semantic_hits_are_nonexclusive_and_other_is_complement():
    masks = {
        name: np.zeros((8, 5), dtype=bool)
        for name in ("self", "partner", "preview", "h1", "h2", "h3", "h4")
    }
    masks["self"][2, 2] = True
    masks["preview"][3, 2] = True

    overlapping = _segment_aoi_fields(masks, (2, 2), 10.0)
    other = _segment_aoi_fields(masks, (7, 4), 10.0)

    assert overlapping["plus_self_hit"] == 1
    assert overlapping["plus_preview_hit"] == 1
    assert overlapping["plus_partner_hit"] == 0
    assert overlapping["plus_other_hit"] == 0
    assert other["plus_other_hit"] == 1


def test_configured_3x3_footprint_includes_diagonal_tiles():
    masks = {
        name: np.zeros((8, 5), dtype=bool)
        for name in ("self", "partner", "preview", "h1", "h2", "h3", "h4")
    }
    masks["self"][3, 3] = True

    fields = _segment_aoi_fields(masks, (2, 2), 10.0, "3x3")

    assert fields["plus_self_hit"] == 0
    assert fields["footprint_self_hit"] == 1
    assert fields["footprint_shape"] == "3x3"
    assert fields["footprint_other_hit"] == 0


class _CollisionMdp:
    @staticmethod
    def _move_if_direction(position, orientation, action):
        del orientation
        dx, dy = action
        candidate = (position[0] + dx, position[1] + dy)
        if candidate[0] < 0:
            candidate = position
        return candidate, action

    @staticmethod
    def is_transition_collision(old_positions, candidate_positions):
        return (
            candidate_positions[0] == candidate_positions[1]
            or candidate_positions == tuple(reversed(old_positions))
        )


def _collision_backend():
    action = SimpleNamespace(
        INDEX_TO_ACTION=((0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), (0, 0))
    )
    return SimpleNamespace(mdp=_CollisionMdp(), Action=action)


def test_collision_categories_are_separate_and_ignore_stay_interact():
    backend = _collision_backend()
    state = SimpleNamespace(players=(
        SimpleNamespace(position=(1, 0), orientation=(0, 1)),
        SimpleNamespace(position=(3, 0), orientation=(0, -1)),
    ))

    partner = _learner_collision_fields(backend, state, 2, 3)
    environment_state = SimpleNamespace(players=(
        SimpleNamespace(position=(2, 0), orientation=(0, 1)),
        SimpleNamespace(position=(0, 0), orientation=(0, -1)),
    ))
    environment = _learner_collision_fields(
        backend, environment_state, 4, 3
    )
    stay = _learner_collision_fields(backend, state, 4, 4)

    assert partner == {
        "partner_collision": 1, "environment_block": 0, "collision_any": 1,
    }
    assert environment == {
        "partner_collision": 0, "environment_block": 1, "collision_any": 1,
    }
    assert stay == {
        "partner_collision": 0, "environment_block": 0, "collision_any": 0,
    }


def test_plus_outputs_count_nonexclusive_hits_and_trial_n_over_180(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "summaries").mkdir()
    decisions = []
    segments = []
    for decision in range(180):
        onset = float(decision * 200)
        decisions.append({
            "T": 200, "H": 2, "episode": 0, "seed": 1,
            "decision": decision, "decision_onset_ms": onset,
            "decision_deadline_ms": onset + 200.0,
            "selected_action_input_time_ms": 100.0, "forced_stay": 0,
        })
        semantic = decision < 90
        segments.append({
            "T": 200, "H": 2, "episode": 0, "seed": 1,
            "decision": decision, "fixation_id": decision,
            "fixation_start_ms": onset, "segment_start_ms": onset,
            "segment_end_ms": onset + 200.0, "duration_ms": 200.0,
            "target_x": 2, "target_y": 2,
            "plus_self_hit": int(semantic), "plus_partner_hit": 0,
            "plus_preview_hit": int(semantic),
            "plus_other_hit": int(not semantic),
        })

    _write_plus_aoi_outputs(
        tmp_path, pd.DataFrame(decisions), pd.DataFrame(segments)
    )

    trials = pd.read_csv(
        tmp_path / "summaries" / "plus_aoi_decision_coverage_trials.csv"
    ).set_index("aoi")
    hits = pd.read_csv(
        tmp_path / "summaries" / "plus_aoi_fixation_hits.csv"
    ).set_index("aoi")
    assert trials.loc["Self", "checked_decisions"] == 90
    assert trials.loc["Self", "checked_decisions_over_180"] == .5
    assert trials.loc["Preview", "checked_decisions_over_180"] == .5
    assert hits.loc["Self", "hit_share"] == .5
    assert hits.loc["Preview", "hit_share"] == .5
    assert hits.loc["Other", "hit_share"] == .5


def test_margin_action_mode_requires_contract_v3():
    config = BenchmarkRolloutConfig(
        action_checkpoint=Path("action.pt"),
        gaze_checkpoint=Path("gaze.pt"),
        output_dir=Path("output"),
        eval_action_mode="periodic_20ms_margin_threshold",
        benchmark_contract_version="contract_v3",
    )

    _validate_config(config)

    with pytest.raises(ValueError, match="requires contract_v3"):
        _validate_config(replace(config, benchmark_contract_version="contract_v2"))


def test_max_probability_action_mode_remains_contract_v2():
    config = BenchmarkRolloutConfig(
        action_checkpoint=Path("action.pt"),
        gaze_checkpoint=Path("gaze.pt"),
        output_dir=Path("output"),
    )

    _validate_config(config)

    with pytest.raises(ValueError, match="requires contract_v2"):
        _validate_config(replace(config, benchmark_contract_version="contract_v3"))


def test_margin_motor_delay_mode_freezes_v4_parameters():
    config = BenchmarkRolloutConfig(
        action_checkpoint=Path("action.pt"),
        gaze_checkpoint=Path("gaze.pt"),
        output_dir=Path("output"),
        tau=.18,
        motor_delay_ms=87.607603,
        eval_action_mode="periodic_20ms_margin_motor_delay",
        benchmark_contract_version="contract_v4",
    )

    _validate_config(config)

    with pytest.raises(ValueError, match="tau=0.18"):
        _validate_config(replace(config, tau=.2))
    with pytest.raises(ValueError, match="motor_delay_ms=87.607603"):
        _validate_config(replace(config, motor_delay_ms=80.0))


def test_stochastic_gaze_motor_delay_mode_requires_contract_v5():
    config = BenchmarkRolloutConfig(
        action_checkpoint=Path("action.pt"),
        gaze_checkpoint=Path("gaze.pt"),
        output_dir=Path("output"),
        tau=.18,
        motor_delay_ms=87.607603,
        eval_action_mode="periodic_20ms_margin_motor_delay_stochastic_gaze",
        benchmark_contract_version="contract_v5",
    )

    _validate_config(config)

    with pytest.raises(ValueError, match="requires contract_v5"):
        _validate_config(replace(config, benchmark_contract_version="contract_v4"))


def test_stochastic_gaze_no_motor_delay_mode_requires_contract_v6():
    config = BenchmarkRolloutConfig(
        action_checkpoint=Path("action.pt"),
        gaze_checkpoint=Path("gaze.pt"),
        output_dir=Path("output"),
        tau=.18,
        motor_delay_ms=0.0,
        eval_action_mode="periodic_20ms_margin_threshold_stochastic_gaze",
        benchmark_contract_version="contract_v6",
    )

    _validate_config(config)

    with pytest.raises(ValueError, match="requires contract_v6"):
        _validate_config(replace(config, benchmark_contract_version="contract_v5"))
    with pytest.raises(ValueError, match="requires tau=0.18"):
        _validate_config(replace(config, tau=.2))
    with pytest.raises(ValueError, match="requires motor_delay_ms=0"):
        _validate_config(replace(config, motor_delay_ms=1.0))


def test_logit_commitment_mode_requires_v7_and_fixed_motor_delay():
    config = BenchmarkRolloutConfig(
        action_checkpoint=Path("action.pt"),
        gaze_checkpoint=Path("gaze.pt"),
        output_dir=Path("output"),
        tau=0.0,
        motor_delay_ms=87.607603,
        eval_action_mode="periodic_20ms_logit_commitment_stochastic_gaze",
        benchmark_contract_version="contract_v7",
        commitment_rule="linear",
        tau_start=2.0,
        tau_min=0.0,
        tau_slope_per_ms=.005,
    )

    _validate_config(config)

    with pytest.raises(ValueError, match="requires contract_v7"):
        _validate_config(replace(config, benchmark_contract_version="contract_v6"))
    with pytest.raises(ValueError, match="requires motor_delay_ms=87.607603"):
        _validate_config(replace(config, motor_delay_ms=0.0))


def test_joint_delay_logit_commitment_mode_accepts_global_fitted_delay():
    config = BenchmarkRolloutConfig(
        action_checkpoint=Path("action.pt"),
        gaze_checkpoint=Path("gaze.pt"),
        output_dir=Path("output"),
        tau=0.0,
        motor_delay_ms=42.5,
        eval_action_mode=(
            "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze"
        ),
        benchmark_contract_version="contract_v8",
        commitment_rule="exponential",
        tau_start=1.0,
        tau_min=0.0,
        tau_kappa_ms=200.0,
    )

    _validate_config(config)

    with pytest.raises(ValueError, match="requires contract_v8"):
        _validate_config(replace(config, benchmark_contract_version="contract_v7"))
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_config(replace(config, motor_delay_ms=-1.0))


def test_no_commitment_mode_is_isolated_from_v8_and_requires_zero_delay():
    base = dict(
        action_checkpoint=Path("action.pt"),
        gaze_checkpoint=Path("gaze.pt"),
        output_dir=Path("output"),
        tau=0.0,
        motor_delay_ms=0.0,
        eval_action_mode="periodic_20ms_no_commitment_stochastic_gaze",
        benchmark_contract_version="contract_v9",
        commitment_rule="none",
    )
    _validate_config(BenchmarkRolloutConfig(**base))
    with pytest.raises(ValueError, match="v8 commitment_rule"):
        _validate_config(BenchmarkRolloutConfig(
            **{
                **base,
                "eval_action_mode": (
                    "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze"
                ),
                "benchmark_contract_version": "contract_v8",
            }
        ))
    with pytest.raises(ValueError, match="action_input_delay_ms=0"):
        _validate_config(BenchmarkRolloutConfig(
            **{**base, "motor_delay_ms": 1.0}
        ))
