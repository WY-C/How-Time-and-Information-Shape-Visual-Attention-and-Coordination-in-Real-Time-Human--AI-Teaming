from __future__ import annotations

import json

import pytest

from final_model.eval.checkpoint_loader import load_action_checkpoint_for_evaluation
from final_model.eval.rollout import EvaluationConfig, _summary, evaluate


def config(path, **values):
    defaults = dict(
        checkpoint=path, evaluation_horizon=0, episodes=1,
        seed_start=777000, episode_length=3, device="cpu",
    )
    defaults.update(values)
    return EvaluationConfig(**defaults)


def test_aggregate_statistics_exactly_match_rows():
    rows = [
        {"deliveries": 1, "sparse_team_reward": 20},
        {"deliveries": 3, "sparse_team_reward": 60},
    ]
    summary = _summary(rows)
    assert summary["mean_deliveries"] == 2
    assert summary["median_deliveries"] == 2
    assert summary["mean_sparse_team_reward"] == 40


@pytest.fixture(scope="module")
def parity_results(synthetic_checkpoints):
    path = synthetic_checkpoints["h0"]
    return (
        evaluate(config(path)),
        evaluate(config(path, behavior_events=True)),
        evaluate(config(path, counterfactual=True)),
    )


def signatures(result):
    return [(
        row["joint_action_trajectory_sha256"], row["reward_trajectory_sha256"],
        row["terminal_state_sha256"], row["actor_recurrent_trajectory_sha256"],
        row["deliveries"],
    ) for row in result.episode_rows]


def test_deterministic_repeated_evaluation_reproducible(parity_results):
    plain, behavior, _ = parity_results
    assert signatures(plain) == signatures(behavior)


def test_plain_vs_behavior_native_parity(parity_results):
    assert signatures(parity_results[0]) == signatures(parity_results[1])


def test_plain_vs_counterfactual_native_parity(parity_results):
    assert signatures(parity_results[0]) == signatures(parity_results[2])


def test_actor_critic_partner_hashes_unchanged(parity_results):
    manifest = parity_results[2].manifest
    assert manifest["actor_hash_before"] == manifest["actor_hash_after"]
    assert manifest["critic_hash_before"] == manifest["critic_hash_after"]
    assert manifest["partner_hash_before"] == manifest["partner_hash_after"]


def test_output_files_and_manifest(synthetic_checkpoints, tmp_path):
    result = evaluate(config(
        synthetic_checkpoints["h0"], behavior_events=True,
        counterfactual=True, output_dir=tmp_path / "output",
    ))
    expected = {"episodes.csv", "summary.json", "manifest.json", "behavior_episodes.csv", "behavior_summary.csv", "behavior_summary.json", "events.jsonl", "counterfactual.csv", "counterfactual_summary.json"}
    assert expected <= {path.name for path in result.output_dir.iterdir()}
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["checkpoint_sha256"] == result.checkpoint.checkpoint_sha256


def test_existing_output_directory_is_not_overwritten(synthetic_checkpoints, tmp_path):
    output = tmp_path / "exists"; output.mkdir()
    with pytest.raises(FileExistsError):
        evaluate(config(synthetic_checkpoints["h0"], output_dir=output))
