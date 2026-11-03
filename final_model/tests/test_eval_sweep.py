import json
from pathlib import Path

import numpy as np
import pytest
import torch

from final_model.common.mappo import ACTION_ARCHITECTURE_ID, FULL_OBSERVATION_CONTRACT
from final_model.common.overcooked_backend import CANONICAL_ENCODER_ID, cooking_contract
from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.checkpoint import ACTION_SPACE
from final_model.common.partners import PARTNER_SET_ID
from final_model.common.partners import partner_manifest
from final_model.eval_sweep import (
    CheckpointEvaluator,
    CheckpointRecord,
    CanonicalHSPAdapter,
    BASELINE_OBSERVATION_MODE,
    BASELINE_RUN_LABEL,
    EPISODE_FIELDS,
    SUMMARY_FIELDS,
    baseline_run_plan,
    discover_run,
    evaluation_seed_set,
    raw_checkpoint_metadata,
    run_sweep,
    cross_run_actual_step_validation,
    summarize_target,
    EvaluationTarget,
    parser,
    load_hsp_s2_actor,
    validate_horizon_comparison,
    validate_run_arguments,
)
from final_model.checkpoint import FORMAT_ID, save_action_checkpoint
from final_model.common.mappo import build_training_stack


def metadata(step, horizon=0):
    return {
        "format_id": FORMAT_ID,
        "training_stage": "action",
        "observation_mode": "full",
        "horizon": horizon,
        "step": step,
        "observation_contract_id": FULL_OBSERVATION_CONTRACT,
        "partner_set_id": PARTNER_SET_ID,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "action_space": list(ACTION_SPACE),
        "architecture_id": ACTION_ARCHITECTURE_ID,
        **cooking_contract(),
    }


def fake_run(
    tmp_path, steps, configured_steps, *, name="run", rollout_length=1024, horizon=0
):
    run = tmp_path / name
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    for step in steps:
        torch.save({"metadata": metadata(step, horizon)}, checkpoints / f"anything_{step}.pt")
    (run / "config.json").write_text(
        json.dumps(
            {
                "observation_mode": "full",
                "training_stage": "action",
                "horizon": horizon,
                "decision_time": None,
                "fixation_shape": None,
                "steps": configured_steps,
                "rollout_length": rollout_length,
            }
        ),
        encoding="utf-8",
    )
    return checkpoints


def parsed_runs(count):
    arguments = []
    for index in range(count):
        arguments.extend(("--run", f"run{index}=/tmp/run{index}"))
    return parser().parse_args(arguments).run


@pytest.mark.parametrize("count", [2, 3])
def test_parser_accepts_two_or_three_runs(count):
    validate_run_arguments(parsed_runs(count))


@pytest.mark.parametrize("count", [1, 4])
def test_parser_rejects_one_or_four_runs(count):
    with pytest.raises(ValueError, match="two or three"):
        validate_run_arguments(parsed_runs(count))


def test_parser_rejects_zero_runs():
    with pytest.raises(ValueError, match="two or three"):
        validate_run_arguments(parser().parse_args([]).run)


def baseline_checkpoint():
    return (
        Path(__file__).resolve().parents[2]
        / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
    )


def test_parser_accepts_standalone_baseline_checkpoint():
    args = parser().parse_args(("--baseline-checkpoint", str(baseline_checkpoint())))
    assert args.run == []
    assert args.baseline_checkpoint == baseline_checkpoint()
    assert args.final_episodes == 10
    assert args.episode_length == 400


def test_baseline_cannot_be_combined_with_training_runs():
    args = parser().parse_args(
        (
            "--baseline-checkpoint", str(baseline_checkpoint()),
            "--run", "h0=/tmp/h0",
            "--run", "h4=/tmp/h4",
        )
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        run_sweep(args)


@pytest.mark.parametrize(
    "override",
    [("--final-episodes", "9"), ("--episode-length", "399")],
)
def test_baseline_protocol_is_fixed_to_ten_by_400(override):
    args = parser().parse_args(
        ("--baseline-checkpoint", str(baseline_checkpoint()), *override)
    )
    with pytest.raises(ValueError, match="10 final episodes.*400"):
        run_sweep(args)


def test_baseline_plan_reuses_final_target_and_output_schema():
    plan = baseline_run_plan(baseline_checkpoint())
    assert plan.label == BASELINE_RUN_LABEL
    assert plan.horizon == 0
    assert plan.final_status == "BASELINE_FINAL"
    assert len(plan.targets) == 1
    assert plan.targets[0].role == "final"
    assert plan.targets[0].episodes_per_partner == 10
    assert plan.targets[0].checkpoint.metadata["observation_mode"] == BASELINE_OBSERVATION_MODE
    assert "sparse_team_reward" in EPISODE_FIELDS
    assert "deliveries" in EPISODE_FIELDS
    assert "mean_sparse_team_reward" in SUMMARY_FIELDS
    assert "mean_deliveries" in SUMMARY_FIELDS


def test_hsp_s2_baseline_loads_as_canonical_20ch_recurrent_actor():
    actor = load_hsp_s2_actor(
        baseline_checkpoint(), device=torch.device("cpu"), construction_seed=20260821
    )
    assert actor.base.cnn[0].in_channels == 20
    assert actor._recurrent_N == 1
    assert actor.hidden_size == 64
    assert not any(parameter.requires_grad for parameter in actor.parameters())


def test_baseline_evaluator_zero_resets_gru_state_without_running_episodes():
    evaluator = CheckpointEvaluator(
        baseline_run_plan(baseline_checkpoint()).targets[0].checkpoint,
        device="cpu",
        construction_seed=20260821,
    )
    hidden, mask = evaluator.initial_recurrent_state()
    assert hidden.shape == (1, 1, 64)
    assert np.count_nonzero(hidden) == 0
    assert np.count_nonzero(mask) == 0
    evaluator.verify_unchanged()


def test_baseline_action_is_deterministic_argmax_under_original_legal_mask():
    evaluator = CheckpointEvaluator(
        baseline_run_plan(baseline_checkpoint()).targets[0].checkpoint,
        device="cpu",
        construction_seed=20260821,
    )
    observation = np.zeros((8, 5, 20), dtype=np.float32)
    available = np.asarray([0, 0, 1, 0, 0, 0], dtype=np.float32)
    hidden, mask = evaluator.initial_recurrent_state()
    first, _ = evaluator.deterministic_action(observation, available, hidden, mask)
    second, _ = evaluator.deterministic_action(observation, available, hidden, mask)
    assert first == second == 2


def test_baseline_adapter_uses_canonical_20ch_and_original_learner_mask():
    class Backend:
        def __init__(self):
            self.observation = np.zeros((8, 5, 20), dtype=np.float32)
            self.available = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float32)
            self.received_action = None

        def canonical_learner_input(self):
            return self.observation, self.available

        def step(self, action):
            self.received_action = action
            return {"done": False}

        def reset(self):
            raise AssertionError("reset is not needed in this contract test")

    backend = Backend()
    adapter = CanonicalHSPAdapter(backend)
    observed = {}

    def policy(observation, critic, available):
        observed.update(observation=observation, critic=critic, available=available)
        return 2

    transition = adapter.run_decision(policy)
    assert observed["observation"] is backend.observation
    assert observed["observation"].shape == (8, 5, 20)
    assert observed["critic"] is None
    assert np.array_equal(observed["available"], backend.available)
    assert not np.array_equal(observed["available"], np.ones(6, dtype=np.float32))
    assert backend.received_action == 2
    assert transition[-1] == {"done": False}


def test_backend_canonical_learner_input_uses_agent1_original_mask():
    backend = object.__new__(DirectOvercookedBackend)
    observation = np.zeros((8, 5, 20), dtype=np.float32)
    expected = np.asarray([0, 1, 1, 0, 1, 1], dtype=np.float32)
    backend.canonical_pair = lambda: (np.ones_like(observation), observation)
    requested = []
    backend._available = lambda agent_index: requested.append(agent_index) or expected
    actual_observation, actual_available = backend.canonical_learner_input()
    assert actual_observation is observation
    assert actual_available is expected
    assert requested == [1]


def plans_for_horizons(tmp_path, labeled_horizons):
    plans = []
    for index, (label, horizon) in enumerate(labeled_horizons):
        checkpoints = fake_run(
            tmp_path,
            [1_000_448, 2_000_000],
            2_000_000,
            name=f"run_{index}",
            horizon=horizon,
        )
        plans.append(
            discover_run(
                label, checkpoints, milestone_episodes=5, final_episodes=10,
                allow_latest_as_final=False,
            )
        )
    return plans


@pytest.mark.parametrize(
    "labeled_horizons",
    [(("h0", 0), ("h4", 4)), (("h0", 0), ("h2", 2), ("h4", 4))],
)
def test_distinct_two_or_three_horizons_are_accepted(tmp_path, labeled_horizons):
    validate_horizon_comparison(plans_for_horizons(tmp_path, labeled_horizons))


def test_duplicate_horizon_is_rejected(tmp_path):
    plans = plans_for_horizons(tmp_path, (("first", 0), ("second", 0)))
    with pytest.raises(RuntimeError, match="distinct"):
        validate_horizon_comparison(plans)


def test_horizon_label_must_match_metadata(tmp_path):
    plans = plans_for_horizons(tmp_path, (("h0", 0), ("h2", 4)))
    with pytest.raises(RuntimeError, match="conflicts"):
        validate_horizon_comparison(plans)


def test_exact_milestones_and_final_are_scheduled_without_duplication(tmp_path):
    checkpoints = fake_run(tmp_path, [1_000_000, 2_000_000, 3_000_000], 3_000_000)
    plan = discover_run(
        "h0", checkpoints, milestone_episodes=5, final_episodes=10,
        allow_latest_as_final=False,
    )
    assert [(target.checkpoint.step, target.role, target.episodes_per_partner) for target in plan.targets] == [
        (1_000_000, "milestone", 5),
        (2_000_000, "milestone", 5),
        (3_000_000, "final", 10),
    ]
    assert plan.final_status == "CANONICAL_FINAL"


def test_earliest_rollout_boundary_checkpoint_is_used_for_milestone(tmp_path):
    checkpoints = fake_run(
        tmp_path, [1_000_700, 1_000_448, 2_000_000, 3_000_000], 3_000_000
    )
    plan = discover_run(
        "h4", checkpoints, milestone_episodes=5, final_episodes=10,
        allow_latest_as_final=False,
    )
    milestone = next(target for target in plan.targets if target.role == "milestone")
    assert milestone.milestone_target_step == 1_000_000
    assert milestone.checkpoint.step == 1_000_448
    assert plan.ignored_checkpoint_steps == [1_000_700]


def test_checkpoint_at_upper_rollout_boundary_is_missing(tmp_path):
    checkpoints = fake_run(tmp_path, [1_001_024, 2_000_000, 3_000_000], 3_000_000)
    plan = discover_run(
        "h4", checkpoints, milestone_episodes=5, final_episodes=10,
        allow_latest_as_final=False,
    )
    row = next(row for row in plan.schedule_rows if row["milestone_target_step"] == 1_000_000)
    assert row["status"] == "MISSING_MILESTONE"
    assert row["checkpoint_actual_step"] is None


def test_latest_override_is_explicit_and_not_evaluated_twice(tmp_path):
    checkpoints = fake_run(tmp_path, [1_000_000, 2_000_000], 3_000_000)
    plan = discover_run(
        "stopped", checkpoints, milestone_episodes=5, final_episodes=10,
        allow_latest_as_final=True,
    )
    assert [(target.checkpoint.step, target.role) for target in plan.targets] == [
        (1_000_000, "milestone"),
        (2_000_000, "final"),
    ]
    assert plan.final_status == "LATEST_AS_FINAL_OVERRIDE"


def test_final_seed_prefix_is_identical_to_milestone_seed_set():
    milestone = evaluation_seed_set(20260821, 5)
    final = evaluation_seed_set(20260821, 10)
    assert final[:5] == milestone


def test_cross_run_actual_steps_are_compared_without_rejecting_mismatch(tmp_path):
    left = fake_run(
        tmp_path, [1_000_448, 2_000_000], 2_000_000, name="left"
    )
    right = fake_run(
        tmp_path, [1_000_512, 2_000_000], 2_000_000, name="right"
    )
    plans = [
        discover_run(label, path, milestone_episodes=5, final_episodes=10,
                     allow_latest_as_final=False)
        for label, path in (("left", left), ("right", right))
    ]
    statuses, comparisons = cross_run_actual_step_validation(plans)
    assert statuses[1_000_000] == "MISMATCH"
    assert comparisons[0]["actual_steps_by_run"] == {
        "left": 1_000_448,
        "right": 1_000_512,
    }


@pytest.mark.parametrize(
    "actual_steps, expected",
    [
        ((1_000_448, 1_000_448), "MATCH"),
        ((1_000_448, 1_000_512), "MISMATCH"),
        ((1_000_448, None), "UNAVAILABLE"),
        ((1_000_448, 1_000_448, 1_000_448), "MATCH"),
        ((1_000_448, 1_000_448, 1_000_512), "MISMATCH"),
        ((1_000_448, 1_000_448, None), "UNAVAILABLE"),
    ],
)
def test_cross_run_status_supports_two_and_three_runs(tmp_path, actual_steps, expected):
    plans = []
    for index, actual_step in enumerate(actual_steps):
        horizon = (0, 2, 4)[index]
        steps = [2_000_000]
        if actual_step is not None:
            steps.insert(0, actual_step)
        checkpoints = fake_run(
            tmp_path, steps, 2_000_000, name=f"cross_{index}", horizon=horizon
        )
        plans.append(
            discover_run(
                f"h{horizon}", checkpoints, milestone_episodes=5, final_episodes=10,
                allow_latest_as_final=False,
            )
        )
    statuses, comparisons = cross_run_actual_step_validation(plans)
    assert statuses[1_000_000] == expected
    assert comparisons[0]["cross_run_actual_step_status"] == expected


def test_incomplete_configured_final_is_reported(tmp_path):
    checkpoints = fake_run(tmp_path, [1_000_448, 2_000_000], 3_000_000)
    plan = discover_run(
        "h0", checkpoints, milestone_episodes=5, final_episodes=10,
        allow_latest_as_final=False,
    )
    assert plan.final_status == "FINAL_CHECKPOINT_INCOMPLETE"
    assert any(row["status"] == "FINAL_CHECKPOINT_INCOMPLETE" for row in plan.schedule_rows)


def test_seed_sequence_is_reused_for_every_run_checkpoint_and_partner():
    seeds = evaluation_seed_set(20260821, 10)
    assignments = {
        (run, checkpoint, partner): seeds[:episode_count]
        for run in ("h0", "h2", "h4")
        for checkpoint, episode_count in (("1m", 5), ("5m", 5), ("final", 10))
        for partner in ("mep5_init", "hsp5")
    }
    assert assignments[("h0", "1m", "hsp5")] == assignments[("h4", "5m", "hsp5")]
    assert assignments[("h2", "final", "mep5_init")][:5] == assignments[("h0", "1m", "mep5_init")]


def test_partner_manifest_matches_hsp_s2_s24_seed3_training_population():
    manifest = partner_manifest()
    assert len(manifest) == 24
    assert sum(row["family"] == "mep" for row in manifest) == 18
    assert sum(row["family"] == "hsp" for row in manifest) == 6
    assert {row["name"] for row in manifest if row["family"] == "hsp"} == {
        "hsp5", "hsp27", "hsp60", "hsp131", "hsp138", "hsp168",
    }


def test_summary_contains_family_and_partner_means():
    checkpoint = CheckpointRecord(Path("synthetic.pt"), metadata(1_000_000))
    target = EvaluationTarget("h0", checkpoint, "milestone", 1)
    rows = [
        {"partner": "mep1", "partner_family": "mep", "sparse_team_reward": 20, "deliveries": 1},
        {"partner": "hsp1", "partner_family": "hsp", "sparse_team_reward": 40, "deliveries": 2},
    ]
    summary = summarize_target(target, rows)
    assert summary["mean_sparse_team_reward"] == 30.0
    assert summary["mep_mean_sparse_team_reward"] == 20.0
    assert summary["hsp_mean_sparse_team_reward"] == 40.0
    assert summary["partner_means"]["mep1"]["mean_deliveries"] == 1.0
    assert summary["partner_means"]["mep1"]["partner_family"] == "mep"


def test_loaded_evaluator_uses_explicit_episode_seeds(tmp_path):
    _args, policy, _trainer, _obs, _share, _action = build_training_stack(
        seed=41, rollout_length=16, ppo_epochs=1, num_mini_batches=2
    )
    path = tmp_path / "full.pt"
    save_action_checkpoint(
        path, actor=policy.actor, critic=policy.critic, mode="full", horizon=0,
        step=1_000_000, seed=41, condition=cooking_contract(),
    )
    checkpoint = CheckpointRecord(path, raw_checkpoint_metadata(path))
    evaluator = CheckpointEvaluator(checkpoint, device="cpu", construction_seed=41)
    rows = evaluator.evaluate_partner(
        partner="mep5_init", evaluation_seeds=(700, 701), episode_length=2
    )
    assert [row["evaluation_seed"] for row in rows] == [700, 701]
    assert [row["steps"] for row in rows] == [2, 2]
    evaluator.verify_unchanged()


def test_mixed_checkpoint_conditions_fail_closed(tmp_path):
    checkpoints = fake_run(tmp_path, [1_000_000, 2_000_000], 2_000_000)
    path = checkpoints / "anything_2000000.pt"
    changed = metadata(2_000_000)
    changed["horizon"] = 4
    torch.save({"metadata": changed}, path)
    with pytest.raises(RuntimeError, match="mixed checkpoint conditions"):
        discover_run(
            "mixed", checkpoints, milestone_episodes=5, final_episodes=10,
            allow_latest_as_final=False,
        )
