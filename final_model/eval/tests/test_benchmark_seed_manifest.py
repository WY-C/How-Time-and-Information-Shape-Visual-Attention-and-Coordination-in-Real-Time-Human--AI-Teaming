from final_model.eval.benchmark_rollout import evaluation_seeds


def test_same_seed_list_is_reused_for_every_time_horizon_cell():
    cells = {
        (time, horizon): evaluation_seeds(810_000, 3)
        for time in (200, 400, 600)
        for horizon in (0, 2, 4)
    }
    assert set(cells.values()) == {(810_000, 810_001, 810_002)}
