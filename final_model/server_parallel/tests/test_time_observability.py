import multiprocessing as mp

import numpy as np

from final_model.server_parallel.shared_buffers import SharedDecisionBuffers


def test_worker_time_metric_counters_are_preallocated_and_cleared():
    buffers = SharedDecisionBuffers.create(mp.get_context("spawn"), 3)
    views = buffers.views()
    names = (
        "gaze_request_counts",
        "same_target_acquisition_counts",
        "new_target_acquisition_counts",
        "valid_memory_tile_counts",
        "freshness_sample_counts",
        "freshness_normalized_sums",
        "saccade_distance_sums_degrees",
        "joint_collisions",
    )
    for name in names:
        assert views[name].shape == (3,)
        views[name].fill(7)
    buffers.clear_decision_metadata()
    for name in names:
        assert np.count_nonzero(views[name]) == 0
