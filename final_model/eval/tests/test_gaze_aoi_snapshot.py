import pytest

from final_model.eval.gaze_aoi_snapshot import _center_key_results


def test_center_key_results_use_preview_first_exclusive_priority():
    episodes = [
        {
            "T": 200, "H": 0, "episode": 0, "seed": 10,
            "sparse_team_reward": 20.0, "deliveries": 1,
            "decisions": 2, "collision_count": 1, "collision_rate": 0.5,
        },
        {
            "T": 200, "H": 2, "episode": 0, "seed": 10,
            "sparse_team_reward": 40.0, "deliveries": 2,
            "decisions": 2, "collision_count": 0, "collision_rate": 0.0,
        },
    ]
    events = [
        {
            "T": 200, "H": 0, "seed": 10,
            "self_center": 1, "partner_center": 0, "preview_center": 0,
            "exclusive_center_aoi": "self",
        },
        {
            "T": 200, "H": 0, "seed": 10,
            "self_center": 0, "partner_center": 0, "preview_center": 0,
            "exclusive_center_aoi": "other",
        },
        {
            "T": 200, "H": 2, "seed": 10,
            "self_center": 0, "partner_center": 1, "preview_center": 1,
            "exclusive_center_aoi": "preview",
        },
        {
            "T": 200, "H": 2, "seed": 10,
            "self_center": 0, "partner_center": 0, "preview_center": 0,
            "exclusive_center_aoi": "other",
        },
    ]

    seeds, conditions = _center_key_results(events, episodes)

    assert seeds[0]["preview_center_share"] is None
    assert seeds[0]["self_center_share"] == pytest.approx(0.5)
    assert seeds[1]["teammate_center_share"] == pytest.approx(0.0)
    assert seeds[1]["preview_center_share"] == pytest.approx(0.5)
    assert sum(
        seeds[1][field]
        for field in (
            "self_center_share", "teammate_center_share",
            "preview_center_share", "other_center_share",
        )
    ) == pytest.approx(1.0)
    assert seeds[1]["center_overlap_share"] == pytest.approx(0.5)
    assert conditions[0]["collision_rate"] == pytest.approx(0.5)
    assert conditions[1]["preview_center_share_mean"] == pytest.approx(0.5)

    for row, duration in zip(events, (100.0, 300.0, 300.0, 100.0)):
        row["duration_ms"] = duration
    duration_seeds, _ = _center_key_results(
        events, episodes, weight_field="duration_ms"
    )
    assert duration_seeds[0]["self_center_share"] == pytest.approx(0.25)
    assert duration_seeds[1]["preview_center_share"] == pytest.approx(0.75)
