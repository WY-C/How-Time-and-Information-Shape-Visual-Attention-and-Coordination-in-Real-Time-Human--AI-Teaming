from __future__ import annotations

import json
import gzip
from pathlib import Path

import numpy as np
import pytest

from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.eval.gaze_behavior_metrics import (
    _cue_distance_analysis,
    _revision_ratio_bootstrap,
    classify_memory_change,
    cue_distance_choice,
    decision_relative_timestamp,
    enrichment_values,
    manhattan_distance,
    probability_metrics,
    revision_risk,
    stabilization_first_post_pairs,
    target_hit_mask,
    top1_stable_index,
)
from final_model.eval.gaze_behavior_rollout import (
    _acquired_aoi_fields,
    _actor_observation_before_acquisition,
    _aoi_fields,
    _source_memory_fields,
)
from final_model.eval.gaze_behavior_rollout import RolloutConfig, run_model_rollouts
from final_model.eval.gaze_behavior_metrics import read_csv
from final_model.eval.partial_rollout import PartialEvaluationConfig, evaluate_partial
from final_model.eval.gaze_behavior_stats import (
    DEFAULT_BOOTSTRAP_CHUNK_SIZE,
    cluster_bootstrap_mean,
    paired_cluster_bootstrap,
)
from final_model.eval.gaze_behavior_artifact import (
    INCOMPLETE_MARKER,
    begin_rollout_artifact,
    finalize_rollout_artifact,
    validate_rollout_artifact,
)
from final_model.eval.gaze_behavior_suite import validate_manifest
from final_model.partial_ver.gaze import acquisition_tiles
from final_model.server_parallel.gaze_v1 import (
    GAZE_TILES,
    action_index_to_tile,
    tile_to_action_index,
)


def test_response_stabilization_synthetic_sequences():
    assert top1_stable_index([4, 4, 4]) == 0
    assert top1_stable_index([4, 0, 0]) == 1
    assert top1_stable_index([4, 0, 4, 4]) == 2
    assert top1_stable_index([]) is None


def test_decision_relative_timestamp_conversion():
    assert decision_relative_timestamp(1460.0, 1200.0) == 260.0
    with pytest.raises(ValueError):
        decision_relative_timestamp(1199.0, 1200.0)


def test_h0_preview_enrichment_is_exact_zero():
    result = enrichment_values(np.full(40, 1 / 40), np.zeros(40, bool), 9)
    assert result == {
        "chance_mass": 0.0, "policy_mass": 0.0, "mass_enrichment": 0.0,
        "sample_hit": 0, "sample_enrichment": 0.0,
    }


def test_gaze_40_action_mapping_parity():
    assert len(GAZE_TILES) == 40
    for index, tile in enumerate(GAZE_TILES):
        assert action_index_to_tile(index) == tile
        assert tile_to_action_index(tile) == index


def test_enrichment_formula_synthetic():
    probability = np.zeros(40)
    probability[:4] = 0.2
    probability[4:] = 0.2 / 36
    mask = np.zeros(40, bool)
    mask[:4] = True
    result = enrichment_values(probability, mask, 0)
    assert result["chance_mass"] == pytest.approx(0.1)
    assert result["policy_mass"] == pytest.approx(0.8)
    assert result["mass_enrichment"] == pytest.approx(8.0)


def test_center_vs_footprint_aoi_synthetic():
    mask = np.zeros((8, 5), bool)
    mask[3, 2] = True
    fields = _aoi_fields({"partner": mask}, (2, 2), "plus")
    assert fields["partner_center"] == 0
    assert fields["partner_footprint"] == 1
    center_hits = target_hit_mask(mask, shape="plus", footprint=False)
    footprint_hits = target_hit_mask(mask, shape="plus", footprint=True)
    assert footprint_hits.sum() > center_hits.sum()


def test_foveal_peripheral_total_aoi_uses_logical_or_without_redefining_footprint():
    partner = np.zeros((8, 5), bool)
    preview = np.zeros((8, 5), bool)
    partner[3, 2] = True
    preview[5, 4] = True
    foveal = acquisition_tiles((2, 2), "plus")
    fields = _acquired_aoi_fields(
        {"partner": partner, "preview": preview}, foveal, ((5, 4),)
    )
    assert fields == {
        "partner_foveal_acquired": 1,
        "partner_peripheral_acquired": 0,
        "partner_total_acquired": 1,
        "preview_foveal_acquired": 0,
        "preview_peripheral_acquired": 1,
        "preview_total_acquired": 1,
    }
    assert _aoi_fields({"partner": partner}, (2, 2), "plus")[
        "partner_footprint"
    ] == 1


def test_opacity_actual_preview_aoi_requires_detected_step_but_state_does_not():
    partner = np.zeros((8, 5), bool)
    preview = np.zeros((8, 5), bool)
    h1 = np.zeros((8, 5), bool)
    h2 = np.zeros((8, 5), bool)
    partner[5, 4] = preview[5, 4] = h1[5, 4] = h2[5, 4] = True
    fields = _acquired_aoi_fields(
        {"partner": partner, "preview": preview, "h1": h1, "h2": h2},
        (), ((5, 4),), (((5, 4), 1),), opacity_actual=True,
    )
    assert fields["partner_peripheral_acquired"] == 1
    assert fields["preview_peripheral_acquired"] == 1
    assert fields["h1_peripheral_acquired"] == 1
    assert fields["h2_peripheral_acquired"] == 0


def test_memory_attribution_separates_foveal_and_successful_peripheral_tiles():
    before = np.zeros((8, 5, 25), np.float32)
    after = before.copy()
    after[1, 0, 0] = 255
    after[1, 0, 20] = 255
    after[4, 4, 21] = 255
    after[4, 4, 20] = 255
    result = _source_memory_fields(
        before, after, ((1, 0),), ((4, 4),)
    )
    assert result["foveal_newly_valid_count"] == 1
    assert result["peripheral_newly_valid_count"] == 1
    assert result["foveal_freshness_gain"] == 1.0
    assert result["peripheral_freshness_gain"] == 1.0
    assert result["acquisition_information_source"] == (
        "peripheral_success_change"
    )


@pytest.mark.parametrize(
    ("content", "freshness", "expected"),
    [
        (True, False, "content_changed"),
        (False, True, "freshness_only_changed"),
        (True, True, "content_and_freshness_changed"),
        (False, False, "no_effective_change"),
    ],
)
def test_content_vs_freshness_classification(content, freshness, expected):
    before = np.zeros((8, 5, 25), np.float32)
    after = before.copy()
    if content:
        after[1, 1, 1] = 255
    if freshness:
        after[1, 1, 20] = 255
    assert classify_memory_change(before, after)["content_change_class"] == expected


def test_same_hidden_distribution_counterfactual_metrics():
    before = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    after = np.array([0.2, 0.5, 0.1, 0.1, 0.05, 0.05])
    result = probability_metrics(before, after)
    assert result["top1_changed"] == 1
    assert result["tv"] == pytest.approx(0.3)
    assert result["js"] > 0


def test_full_policy_acquisition_counterfactual_is_gaze_independent():
    memory_after = np.zeros((8, 5, 25), np.float32)
    full_observation = np.ones((8, 5, 25), np.float32)
    event = {"before": memory_after.copy(), "after": memory_after}
    result = _actor_observation_before_acquisition(
        event, full_observation, ground_truth_policy_input=True,
        actor_query_time_ms=200.0,
    )
    assert result is full_observation


def test_partial_policy_acquisition_still_requires_exact_memory_snapshot():
    observation = np.ones((8, 5, 25), np.float32)
    event = {
        "before": np.zeros_like(observation),
        "after": observation.copy(),
    }
    assert np.array_equal(
        _actor_observation_before_acquisition(
            event, observation, ground_truth_policy_input=False,
            actor_query_time_ms=200.0,
        ),
        event["before"],
    )
    event["after"][0, 0, 0] = 0
    with pytest.raises(RuntimeError, match="canonical observation"):
        _actor_observation_before_acquisition(
            event, observation, ground_truth_policy_input=False,
            actor_query_time_ms=200.0,
        )


def test_acquisition_annotations_do_not_mutate_inputs():
    before = np.zeros((8, 5, 25), np.float32)
    after = before.copy()
    after[2, 2, 20] = 255
    left, right = before.copy(), after.copy()
    classify_memory_change(before, after)
    mask = np.zeros((8, 5), bool)
    mask[2, 2] = True
    target_hit_mask(mask, shape="plus", footprint=True)
    assert np.array_equal(before, left)
    assert np.array_equal(after, right)


def test_gaze_and_action_modes_validate_independently(tmp_path):
    payload = {
        "models": [{"name": "m", "checkpoint": "missing.pt"}],
        "evaluation": {"gaze_mode": "argmax", "action_mode": "stochastic"},
    }
    _, spec = validate_manifest(payload, source=tmp_path / "manifest.json")
    assert spec.gaze_mode == "argmax"
    assert spec.action_mode == "stochastic"


def test_cluster_bootstrap_reproducibility():
    rows = ([{"seed": seed, "x": float(seed % 3)} for seed in range(10)]
            + [{"seed": 11, "x": None}])
    first = cluster_bootstrap_mean(rows, "x", iterations=100, seed=7)
    second = cluster_bootstrap_mean(rows, "x", iterations=100, seed=7)
    assert first == second


def test_paired_bootstrap_uses_common_seed_grouping():
    result = paired_cluster_bootstrap(
        {1: 5.0, 2: 4.0, 3: 100.0}, {1: 2.0, 2: 3.0, 4: -100.0},
        iterations=100, seed=3,
    )
    assert result["n_pairs"] == 2
    assert result["mean"] == pytest.approx(2.0)


def test_chunked_bootstrap_is_exactly_equal_to_original_full_matrix():
    rows = [
        {"seed": seed, "x": value}
        for seed, values in {
            1: (1.0, 3.0), 2: (4.0,), 3: (8.0, 10.0),
        }.items()
        for value in values
    ]
    iterations, seed = 1003, 71
    cluster_values = np.asarray([2.0, 4.0, 9.0], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(cluster_values), size=(iterations, len(cluster_values))
    )
    old_boot = cluster_values[draws].mean(axis=1)
    old_low, old_high = np.percentile(old_boot, (2.5, 97.5))

    result = cluster_bootstrap_mean(
        rows, "x", iterations=iterations, seed=seed,
        chunk_size=DEFAULT_BOOTSTRAP_CHUNK_SIZE,
    )

    assert result["mean"] == float(cluster_values.mean())
    assert result["ci_low"] == float(old_low)
    assert result["ci_high"] == float(old_high)


def test_chunked_revision_ratio_matches_raw_event_resampling_exactly():
    rows = []
    for seed, values in {
        10: ((1, 1), (1, 0), (0, 1), (0, 0)),
        11: ((1, 1), (1, 1), (0, 0), (0, 0)),
        12: ((1, 0), (1, 1), (0, 1), (0, 0)),
    }.items():
        rows.extend({
            "seed": seed, "proposal_index": index + 1,
            "preview_footprint": exposed,
            "online_top1_changed_from_previous": changed,
        } for index, (exposed, changed) in enumerate(values))
    clusters = [
        [row for row in rows if row["seed"] == seed]
        for seed in (10, 11, 12)
    ]
    iterations, seed = 777, 19
    rng = np.random.default_rng(seed)
    old = []
    for _ in range(iterations):
        sampled = [
            clusters[index]
            for index in rng.integers(0, len(clusters), len(clusters))
        ]
        ratio = revision_risk(
            [row for cluster in sampled for row in cluster],
            "preview_footprint",
        )["risk_ratio"]
        if np.isfinite(ratio):
            old.append(ratio)

    new = _revision_ratio_bootstrap(
        rows, "preview_footprint", iterations=iterations, seed=seed,
    )

    assert np.array_equal(new, np.asarray(old))


def _write_artifact_fixture(path):
    begin_rollout_artifact(path)
    (path / "episodes.csv").write_text("T,H,seed\n200,0,7\n", encoding="utf-8")
    (path / "decisions.csv").write_text("T,H,seed\n200,0,7\n", encoding="utf-8")
    for name in ("gaze_requests.csv.gz", "acquisitions.csv.gz"):
        with gzip.open(path / name, "wt", encoding="utf-8") as handle:
            handle.write("T,H,seed\n200,0,7\n")
    with (path / "gaze_inputs.npz").open("wb") as handle:
        np.savez_compressed(
            handle, memory=np.zeros((1, 1)), fixation=np.zeros((1, 2)),
            context=np.zeros((1, 1)),
        )
    return {
        "model_name": "fixture", "checkpoint_absolute_path": "/tmp/model.pt",
        "decision_times": [200], "horizons": [0], "fixation_shape": "plus",
        "episodes": 1, "episode_length": 1, "seed_start": 7,
        "gaze_mode": "stochastic", "action_mode": "stochastic",
        "aoi_precedence": ["self", "partner", "preview"],
        "request_count": 1, "acquisition_count": 1,
    }


def test_rollout_artifact_atomic_save_load_and_corruption_rejection(tmp_path):
    manifest = _write_artifact_fixture(tmp_path)
    assert (tmp_path / INCOMPLETE_MARKER).is_file()
    assert json.loads((tmp_path / "rollout_manifest.json").read_text())[
        "rollout_artifact_status"
    ] == "incomplete"
    saved = finalize_rollout_artifact(tmp_path, manifest)
    assert saved["rollout_artifact_schema_version"] == 1
    assert validate_rollout_artifact(tmp_path) == saved
    assert not (tmp_path / INCOMPLETE_MARKER).exists()

    with (tmp_path / "episodes.csv").open("a", encoding="utf-8") as handle:
        handle.write("200,0,8\n")
    with pytest.raises(RuntimeError, match="episode count mismatch"):
        validate_rollout_artifact(tmp_path)


def test_incomplete_rollout_artifact_is_rejected(tmp_path):
    _write_artifact_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="marked incomplete"):
        validate_rollout_artifact(tmp_path)


def test_rollout_artifact_manifest_seed_mismatch_is_rejected(tmp_path):
    saved = finalize_rollout_artifact(
        tmp_path, _write_artifact_fixture(tmp_path)
    )
    saved["seed_start"] = 8
    (tmp_path / "rollout_manifest.json").write_text(
        json.dumps(saved), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="seed range"):
        validate_rollout_artifact(tmp_path)


def test_revision_rd_rr_synthetic():
    rows = [
        {"proposal_index": 1, "partner_footprint": 1, "online_top1_changed_from_previous": 1},
        {"proposal_index": 2, "partner_footprint": 1, "online_top1_changed_from_previous": 0},
        {"proposal_index": 1, "partner_footprint": 0, "online_top1_changed_from_previous": 0},
        {"proposal_index": 0, "partner_footprint": 1, "online_top1_changed_from_previous": 1},
    ]
    result = revision_risk(rows, "partner_footprint")
    assert result["eligible_transitions"] == 3
    assert result["risk_difference"] == pytest.approx(0.5)
    assert np.isnan(result["risk_ratio"])


def test_stabilization_first_post_pair_construction():
    common = {"model_name": "m", "T": 400, "H": 2, "seed": 1, "decision": 0}
    rows = [
        {**common, "proposal_index": 1, "response_phase": "AT_RESPONSE"},
        {**common, "proposal_index": 2, "response_phase": "POST"},
        {**common, "proposal_index": 3, "response_phase": "POST"},
    ]
    pairs = stabilization_first_post_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0][1]["proposal_index"] == 2


def test_partner_distance_annotation():
    assert manhattan_distance((1, 1), (4, 3)) == 5


def test_cue_distance_near_far_synthetic():
    result = cue_distance_choice(
        fixation=(0, 0), near_candidates=((7, 4),), far_candidates=((1, 0),),
        first_preview="far",
    )
    assert result["far_closer"] == 1
    assert result["first_preview_far"] == 1


def test_cue_distance_indexes_acquisitions_once():
    class CountingRows(list):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    common = {"model_name": "m", "T": 200, "H": 4, "seed": 1, "decision": 0}
    decisions = [{
        **common,
        "start_fixation_x": 0,
        "start_fixation_y": 0,
        "near_center_candidate_tiles": "[[0, 0]]",
        "far_center_candidate_tiles": "[[7, 4]]",
        "near_footprint_candidate_tiles": "[[0, 0]]",
        "far_footprint_candidate_tiles": "[[7, 4]]",
    }]
    acquisitions = CountingRows([{
        **common,
        "proposal_index": 1,
        "near_center": 1,
        "far_center": 0,
        "near_footprint": 1,
        "far_footprint": 0,
    }])

    rows = _cue_distance_analysis(
        decisions, acquisitions, iterations=1, seed=7
    )

    assert acquisitions.iterations == 1
    assert len(rows) == 8
    assert rows[0]["eligible_decisions"] == 1


def test_input_ablation_style_copy_leaves_trajectory_input_untouched():
    memory = np.ones((8, 5, 25), np.float32)
    changed = memory.copy()
    changed[..., 20] = 0
    assert np.all(memory[..., 20] == 1)


def test_multi_model_manifest_validation(tmp_path):
    payload = {
        "models": [
            {"name": "m1", "checkpoint": "one.pt"},
            {"name": "m2", "checkpoint": "two.pt"},
        ],
        "evaluation": {"decision_times": [200], "horizons": [0]},
    }
    models, spec = validate_manifest(payload, source=tmp_path / "manifest.json")
    assert [model.name for model in models] == ["m1", "m2"]
    assert spec.decision_times == (200,)


def test_manifest_rejects_ambiguous_aoi_precedence():
    payload = {
        "models": [{"name": "m", "checkpoint": "one.pt"}],
        "evaluation": {"aoi_precedence": ["self", "self", "preview"]},
    }
    with pytest.raises(ValueError):
        validate_manifest(payload)


def test_fail_closed_on_missing_checkpoint(tmp_path):
    with pytest.raises(RuntimeError, match="checkpoint does not exist"):
        load_partial_checkpoint_for_evaluation(tmp_path / "missing.pt")


def test_plus_footprint_uses_canonical_helper():
    assert acquisition_tiles((3, 2), "plus") == (
        (3, 2), (3, 1), (3, 3), (2, 2), (4, 2)
    )


def test_stochastic_rollout_parity_with_existing_partial_evaluator(tmp_path):
    checkpoint = Path(
        "final_model/runs/gaze_corrected_memory1200_20m_20260827_141207/"
        "checkpoints/step_017000000.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("repository 17M checkpoint is unavailable")
    common = dict(
        checkpoint=checkpoint, decision_time=400, horizon=2,
        fixation_shape="plus", gaze_type="learned", episodes=1,
        episode_length=3, seed_start=20260826, device="cpu",
        gaze_deterministic=False,
        action_deterministic=False,
    )
    baseline = evaluate_partial(PartialEvaluationConfig(**common))
    loaded = load_partial_checkpoint_for_evaluation(checkpoint, device="cpu")
    model_dir = tmp_path / "model"
    result = run_model_rollouts(
        loaded=loaded,
        config=RolloutConfig(
            model_name="parity", decision_times=(400,), horizons=(2,),
            fixation_shape="plus", episodes=1, episode_length=3,
            seed_start=20260826, gaze_mode="stochastic",
            action_mode="stochastic",
            aoi_precedence=("self", "partner", "preview"),
        ),
        model_dir=model_dir,
    )
    expected = baseline["episodes"][0]
    actual = result["episodes"][0]
    for field in (
        "deliveries", "sparse_team_reward", "action_trajectory_sha256",
        "acquisitions", "gaze_requests", "same_target_acquisitions",
        "new_target_acquisitions", "valid_memory_fraction", "mean_freshness",
        "mean_saccade_distance_degrees", "forced_stays",
        "no_acquisition_decisions",
    ):
        assert actual[field] == expected[field]
    requests = read_csv(model_dir / "gaze_requests.csv.gz")
    acquisitions = read_csv(model_dir / "acquisitions.csv.gz")
    requested = {key: 0 for key in expected["requested_target_distribution"]}
    landing = {key: 0 for key in expected["landing_target_distribution"]}
    for row in requests:
        requested[f"{row['requested_target_x']},{row['requested_target_y']}"] += 1
    for row in acquisitions:
        landing[f"{row['fixation_x']},{row['fixation_y']}"] += 1
    assert requested == expected["requested_target_distribution"]
    assert landing == expected["landing_target_distribution"]
