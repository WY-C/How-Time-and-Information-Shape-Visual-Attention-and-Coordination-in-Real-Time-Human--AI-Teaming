from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from final_model.audit import source_audit
from final_model.checkpoint import initialize_partial_from_full, save_action_checkpoint
from final_model.common.mappo import (
    ObservationParityError,
    RolloutPolicyAuditCapture,
    SeparatedReplayBuffer,
    architecture_report,
    build_actionnet,
    build_training_stack,
    entropy_coefficient,
    explained_variance,
    ppo_health_metrics,
    training_args,
)
from final_model.common.mappo import _observation_digest
from final_model.common.overcooked_backend import DirectOvercookedBackend, cooking_contract
from final_model.common.partners import parameter_hash
from final_model.common.paths import (
    policy_config_directory,
    repository_root,
    zsceval_source_root,
)
from final_model.common.reward import (
    hsp_reward_shaping_factor,
    native_reward,
    native_reward_shaping_audit,
)
from final_model.common.tracking import MANDATORY_KEYS, canonical_run_name
from final_model.train import parser, validate_combination


def _complete_actor_capture(capture, buffer, actor):
    actor.eval()
    capture.attach_actor(actor)
    environments = buffer.actions.shape[1]
    backends = [
        SimpleNamespace(environment_steps=0, episode_step=0, episode_length=100)
        for _ in range(environments)
    ]
    for step in range(buffer.actions.shape[0]):
        capture.record_pre_forward(step, buffer.obs[step].copy(), buffer)
        for backend in backends:
            backend.environment_steps = step
            backend.episode_step = step
        capture.record_environment_before(step, backends, buffer)
        capture.begin_actor_forward(step, actor)
        with torch.no_grad():
            actions, logs, returned = actor(
                buffer.obs[step], buffer.rnn_states[step], buffer.masks[step],
                buffer.available_actions[step],
            )
        capture.verify_post_actor_forward(step, buffer)
        action_values = actions.detach().cpu().numpy()
        log_values = logs.detach().cpu().numpy()
        returned_values = returned.detach().cpu().numpy()
        buffer.actions[step] = action_values
        buffer.action_log_probs[step] = log_values
        capture.record_post_forward(step, action_values, log_values)
        capture.record_environment_after(
            step,
            [{"done": False} for _ in range(environments)],
            returned_values,
            returned_values.copy(),
            np.ones((environments, 1), dtype=np.float32),
        )
    capture.end_rollout(actor)
    capture.preupdate_actor_hash = parameter_hash(actor)


def test_path_resolution_environment_then_markers(monkeypatch, tmp_path):
    for marker in ("predict_path", "policy_pool", "model"):
        (tmp_path / marker).mkdir()
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path))
    assert repository_root() == tmp_path.resolve()
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError):
        repository_root()


def test_zsceval_paths_use_only_lib(monkeypatch, tmp_path):
    for marker in ("predict_path", "policy_pool", "model"):
        (tmp_path / marker).mkdir()
    source = tmp_path / "lib/zsc_eval"
    config = source / "zsceval/policy_pool/random3/policy_config"
    config.mkdir(parents=True)
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path))

    assert zsceval_source_root() == source.resolve()
    assert policy_config_directory() == config.resolve()


def test_native_reward_formula_and_complete_term_audit():
    reward = native_reward({"sparse_reward_by_agent": [20, 0], "shaped_reward_by_agent": [3, 5]})
    assert reward.sparse_team_reward == 20
    assert reward.learner_native_shaped_reward == 5
    assert reward.weighted_shaped_reward == 5
    assert reward.optimization_reward == 25
    assert reward.deliveries == 1
    audit = native_reward_shaping_audit()
    assert {term["source_key"] for term in audit["native_shaping_terms"]} == {
        "PLACEMENT_IN_POT_REW", "DISH_PICKUP_REWARD", "SOUP_PICKUP_REWARD",
        "DISH_DISP_DISTANCE_REW", "POT_DISTANCE_REW", "SOUP_DISTANCE_REW",
    }
    assert not audit["v3d_contribution_reward_used"]


def test_hsp_s2_reward_shaping_schedule_and_weighted_reward():
    assert hsp_reward_shaping_factor(0) == 1.0
    assert hsp_reward_shaping_factor(5_000_000) == 0.75
    assert hsp_reward_shaping_factor(10_000_000) == 0.5
    assert hsp_reward_shaping_factor(15_000_000) == 0.25
    assert hsp_reward_shaping_factor(20_000_000) == 0.0
    assert hsp_reward_shaping_factor(25_000_000) == 0.0
    reward = native_reward(
        {"sparse_reward_by_agent": [20, 0], "shaped_reward_by_agent": [3, 5]},
        reward_shaping_factor=0.5,
    )
    assert reward.weighted_shaped_reward == 2.5
    assert reward.optimization_reward == 22.5
    final = native_reward(
        {"sparse_reward_by_agent": [20, 0], "shaped_reward_by_agent": [3, 5]},
        reward_shaping_factor=hsp_reward_shaping_factor(20_000_000),
    )
    assert final.sparse_team_reward == 20
    assert final.optimization_reward == final.sparse_team_reward


def test_entropy_schedule_and_explained_variance_are_pure_and_finite():
    assert entropy_coefficient(0) == 0.20
    assert entropy_coefficient(10_000_000) == 0.05
    assert entropy_coefficient(20_000_000) == 0.01
    assert entropy_coefficient(30_000_000) == 0.01
    assert explained_variance([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert explained_variance([0, 0, 0], [1, 1, 1]) == 0.0


def test_recurrent_generator_preserves_time_env_ids_and_chunk_initial_hidden():
    args, _policy, _trainer, obs_space, share_space, action_space = build_training_stack(
        seed=3,
        rollout_length=20,
        n_rollout_threads=3,
        ppo_epochs=1,
        num_mini_batches=1,
    )
    assert args.data_chunk_length == 10
    buffer = SeparatedReplayBuffer(args, obs_space, share_space, action_space)
    time_steps, environments = buffer.actions.shape[:2]
    transition_ids = np.arange(time_steps * environments).reshape(
        time_steps, environments
    )
    buffer.transition_ids[:] = transition_ids
    for time_index in range(time_steps):
        for env_index in range(environments):
            transition_id = transition_ids[time_index, env_index]
            buffer.obs[time_index, env_index, 0, 0, 0] = transition_id
            buffer.actions[time_index, env_index, 0] = transition_id
            buffer.action_log_probs[time_index, env_index, 0] = transition_id
            buffer.rnn_states[time_index, env_index].fill(transition_id)
            buffer.available_actions[time_index, env_index, 0] = transition_id

    torch.manual_seed(91)
    sample = next(
        buffer.recurrent_generator(
            np.zeros_like(buffer.rewards),
            num_mini_batch=1,
            data_chunk_length=args.data_chunk_length,
        )
    )
    observation_ids = sample[1][:, 0, 0, 0].reshape(
        args.data_chunk_length, -1
    )
    action_ids = sample[4][:, 0].reshape(args.data_chunk_length, -1)
    old_log_ids = sample[9][:, 0].reshape(args.data_chunk_length, -1)
    legal_mask_ids = sample[11][:, 0].reshape(args.data_chunk_length, -1)
    assert np.array_equal(observation_ids, action_ids)
    assert np.array_equal(observation_ids, old_log_ids)
    assert np.array_equal(observation_ids, legal_mask_ids)
    assert np.all(np.diff(observation_ids, axis=0) == environments)
    assert np.array_equal(sample[2][:, 0, 0], observation_ids[0])


def test_rollout_capture_reports_first_legal_mask_transition_mismatch():
    args, policy, _trainer, obs_space, share_space, action_space = build_training_stack(
        seed=4, rollout_length=10, n_rollout_threads=2,
        ppo_epochs=1, num_mini_batches=1,
    )
    buffer = SeparatedReplayBuffer(args, obs_space, share_space, action_space)
    capture = RolloutPolicyAuditCapture(buffer)
    _complete_actor_capture(capture, buffer, policy.actor)
    buffer.available_actions[5, 1, 2] = 0.0
    with pytest.raises(
        RuntimeError,
        match=(
            "tensor=legal_action_mask transition_id=11 time=5 env=1"
        ),
    ):
        capture.verify_buffer(buffer, mode="partial", horizon=0)


def test_observation_digest_storage_preserves_trailing_null_byte():
    raw_digest = bytes(range(1, 32)) + b"\x00"
    legacy = np.empty(1, dtype="S32")
    legacy[0] = raw_digest
    assert bytes(legacy[0]) != raw_digest
    assert bytes(np.frombuffer(raw_digest, dtype=np.uint8).copy()) == raw_digest

    args, _policy, _trainer, obs_space, share_space, action_space = build_training_stack(
        seed=5, rollout_length=10, n_rollout_threads=1,
        ppo_epochs=1, num_mini_batches=1,
    )
    buffer = SeparatedReplayBuffer(args, obs_space, share_space, action_space)
    candidate = buffer.obs[0].copy()
    for value in range(100_000):
        candidate[0, 0, 0, 0] = value
        digest = _observation_digest(candidate[0])
        if digest[-1] == 0:
            break
    else:
        raise AssertionError("failed to construct a trailing-null SHA-256 test value")
    buffer.obs[0] = candidate
    capture = RolloutPolicyAuditCapture(buffer)
    capture.record_pre_forward(0, candidate.copy(), buffer)
    assert capture.observation_sha256.dtype == np.uint8
    assert bytes(capture.observation_sha256[0, 0]) == digest


def test_completed_rollout_observation_mismatch_reports_bidirectional_search():
    args, policy, _trainer, obs_space, share_space, action_space = build_training_stack(
        seed=6, rollout_length=10, n_rollout_threads=2,
        ppo_epochs=1, num_mini_batches=1,
    )
    buffer = SeparatedReplayBuffer(args, obs_space, share_space, action_space)
    capture = RolloutPolicyAuditCapture(buffer)
    for step in range(10):
        for env_index in range(2):
            buffer.obs[step, env_index].fill(step * 2 + env_index)
    _complete_actor_capture(capture, buffer, policy.actor)

    first = buffer.obs[3, 0].copy()
    second = buffer.obs[5, 1].copy()
    buffer.obs[3, 0] = second
    buffer.obs[5, 1] = first
    with pytest.raises(ObservationParityError) as failure:
        capture.verify_buffer(buffer, mode="partial", horizon=0)
    report = failure.value.report
    assert report["transition_id"] == 6
    assert report["time"] == 3
    assert report["env"] == 0
    assert report["nonzero_difference_count"] > 0
    search = report["lost_observation_search"]
    assert search["classification"] == "A_ORDERING_INDEXING_PERMUTATION"
    assert search["captured_observation_found_in_buffer"] == [
        {"time": 5, "env": 1, "transition_id": 11}
    ]
    assert search["buffer_observation_found_in_capture"] == [
        {"time": 5, "env": 1, "transition_id": 11}
    ]


def test_wandb_health_formulas_and_required_keys():
    ratio = torch.tensor([0.7, 1.0, 1.1, 1.3], dtype=torch.float64)
    kl_samples, clipped = ppo_health_metrics(ratio, 0.2)
    expected_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
    assert torch.allclose(kl_samples.mean(), expected_kl)
    assert clipped.mean().item() == 0.5
    assert set(MANDATORY_KEYS) == {
        "train/reward", "train/optimization_reward", "train/sparse_team_reward",
        "train/native_shaped_reward", "train/deliveries", "train/approx_kl",
        "train/entropy", "train/clip_fraction",
    }


def test_direct_backend_native_values_predictor_and_partner_freeze():
    backend = DirectOvercookedBackend(seed=4, horizon=2, episode_length=4, device="cpu")
    assert backend.mdp.reward_shaping_params == {
        "PLACEMENT_IN_POT_REW": 3, "DISH_PICKUP_REWARD": 3, "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0, "POT_DISTANCE_REW": 0, "SOUP_DISTANCE_REW": 0,
    }
    before_state = backend.state.to_dict()
    planes = backend.preview_planes()
    assert planes.shape == (4, 8, 5)
    assert before_state == backend.state.to_dict()
    before = parameter_hash(backend.partners.actor)
    backend.step(4)
    assert parameter_hash(backend.partners.actor) == before
    assert backend.partner_step_guard_failures == 0
    assert backend.partner_full_hash_failures == 0


def test_backend_observation_caches_invalidate_once_per_env_step(monkeypatch):
    backend = DirectOvercookedBackend(seed=5, horizon=0, episode_length=4, device="cpu")
    canonical_calls = 0
    original_encoding = backend.mdp.lossless_state_encoding

    def counted_encoding(state):
        nonlocal canonical_calls
        canonical_calls += 1
        return original_encoding(state)

    monkeypatch.setattr(backend.mdp, "lossless_state_encoding", counted_encoding)
    first = backend.canonical_pair()
    second = backend.canonical_pair()
    assert first is second
    assert canonical_calls == 1
    backend.step(4)
    backend.canonical_pair()
    assert canonical_calls == 2

    predicted = DirectOvercookedBackend(seed=6, horizon=4, episode_length=4, device="cpu")
    predictor_calls = 0
    original_predict = predicted.oracle.predict

    def counted_predict(*args, **kwargs):
        nonlocal predictor_calls
        predictor_calls += 1
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(predicted.oracle, "predict", counted_predict)
    assert np.array_equal(predicted.preview_planes(), predicted.preview_planes())
    assert predictor_calls == 1
    predicted.step(4)
    predicted.preview_planes()
    assert predictor_calls == 2


def test_partner_version_guard_detects_in_place_mutation():
    backend = DirectOvercookedBackend(seed=7, horizon=0, episode_length=4, device="cpu")
    with torch.no_grad():
        next(backend.partners.actor.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="tensor guard changed"):
        backend.step(4)
    assert backend.partner_step_guard_failures == 1


def test_action_architecture_and_checkpoint_horizon_fail_closed(tmp_path):
    report = architecture_report(seed=3)
    assert report["parity"] and report["actor_input"] == [8, 5, 25]
    _args, policy, _trainer, _obs, _share, _act = build_training_stack(
        seed=3, rollout_length=16, ppo_epochs=1, num_mini_batches=2
    )
    path = tmp_path / "full_h2.pt"
    save_action_checkpoint(
        path, actor=policy.actor, critic=policy.critic, mode="full", horizon=2,
        step=0, seed=3, condition=cooking_contract(),
    )
    target = build_actionnet(seed=9)
    initialize_partial_from_full(path, actor=target, horizon=2)
    with pytest.raises(RuntimeError):
        initialize_partial_from_full(path, actor=build_actionnet(seed=9), horizon=4)


def test_parser_supported_combinations():
    learner = "policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"
    partner = "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
    valid = [
        ["--observation-mode", "full", "--training-stage", "action", "--horizon", "0",
         "--hsp-initial-checkpoint", learner, "--fixed-partner-checkpoint", partner],
        ["--observation-mode", "partial", "--training-stage", "action", "--horizon", "0",
         "--decision-time", "200", "--fixation-shape", "3x3",
         "--fixed-partner-checkpoint", partner],
        ["--observation-mode", "partial", "--training-stage", "action", "--horizon", "2",
         "--decision-time", "400", "--fixation-shape", "2x2", "--source-checkpoint", "x",
         "--fixed-partner-checkpoint", partner],
        ["--observation-mode", "partial", "--training-stage", "gaze", "--horizon", "4",
         "--decision-time", "600", "--fixation-shape", "plus", "--source-checkpoint", "x",
         "--fixed-partner-checkpoint", partner],
    ]
    for argv in valid:
        validate_combination(parser().parse_args(argv))
    invalid = parser().parse_args([
        "--observation-mode", "full", "--training-stage", "gaze", "--horizon", "0"
    ])
    with pytest.raises(ValueError):
        validate_combination(invalid)
    missing_gaze_source = parser().parse_args([
        "--observation-mode", "partial", "--training-stage", "gaze", "--horizon", "4",
        "--decision-time", "600", "--fixation-shape", "plus",
    ])
    with pytest.raises(ValueError, match="Partial Gaze requires"):
        validate_combination(missing_gaze_source)


def test_every_training_mode_defaults_to_one_million_step_checkpoints():
    commands = [
        ["--observation-mode", "full", "--training-stage", "action", "--horizon", "0"],
        ["--observation-mode", "partial", "--training-stage", "action", "--horizon", "0",
         "--decision-time", "200", "--fixation-shape", "3x3"],
        ["--observation-mode", "partial", "--training-stage", "gaze", "--horizon", "0",
         "--decision-time", "200", "--fixation-shape", "3x3", "--source-checkpoint", "x"],
    ]
    assert {parser().parse_args(command).checkpoint_interval for command in commands} == {1_000_000}


def test_canonical_training_hyperparameter_defaults():
    args = parser().parse_args(
        ["--observation-mode", "full", "--training-stage", "action", "--horizon", "0"]
    )
    assert {
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "ppo_epochs": args.ppo_epochs,
        "num_mini_batches": args.num_mini_batches,
        "entropy_coefs": args.entropy_coefs,
        "entropy_coef_horizons": args.entropy_coef_horizons,
        "clip_param": args.clip_param,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "max_grad_norm": args.max_grad_norm,
        "rollout_length": args.rollout_length,
        "episode_length": args.episode_length,
        "n_rollout_threads": args.n_rollout_threads,
        "seed": args.seed,
        "steps": args.steps,
    } == {
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,
        "ppo_epochs": 15,
        "num_mini_batches": 1,
        "entropy_coefs": [0.2, 0.05, 0.01],
        "entropy_coef_horizons": [0, 10_000_000, 20_000_000],
        "clip_param": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "max_grad_norm": 10.0,
        "rollout_length": 400,
        "episode_length": 400,
        "n_rollout_threads": 100,
        "seed": 20260821,
        "steps": 20_000_000,
    }
    assert training_args(rollout_length=400).data_chunk_length == 10
    assert canonical_run_name(
        "full", "action", 4, None, None, 20260821, args.steps
    ).endswith("_20m")


def test_source_preservation_portability_and_zero_old_reward_dependencies():
    report = source_audit()
    assert report["references_unchanged"]
    assert report["hardcoded_user_path_pass"]
    assert report["v3e_production_dependency_zero"]
    assert report["v3d_contribution_production_dependency_zero"]
