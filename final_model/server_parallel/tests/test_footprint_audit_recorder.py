import numpy as np

from final_model.partial_ver.memory import PersistentPartialMemory
from final_model.server_parallel.footprint_audit_recorder import (
    FootprintAuditRecorder,
)


class _Policy:
    policy_id = "test_memory_only"

    def __call__(self, memory, fixation, context):
        assert memory.shape == (8, 5, 25)
        return fixation


class _Adapter:
    def __init__(self):
        self.gaze_policy = _Policy()
        self.memory = PersistentPartialMemory()
        self.memory.reset(np.zeros((8, 5, 20), dtype=np.float32))


def test_recorder_samples_exact_events_and_tracks_turnover():
    adapter = _Adapter()
    recorder = FootprintAuditRecorder(
        env_id=0, horizon=2, fixation_shape="2x2"
    )
    recorder.install(adapter)

    adapter.gaze_policy(
        adapter.memory.snapshot(), (0, 0), np.zeros(7, dtype=np.float32)
    )
    state = np.zeros((8, 5, 20), dtype=np.float32)
    preview = np.zeros((8, 5, 4), dtype=np.float32)
    preview[0, 0, 0] = 1.0
    adapter.memory.acquire(
        state, preview, ((0, 0), (1, 0), (0, 1), (1, 1)), timestamp_ms=0.0
    )
    acquired = adapter.memory.snapshot()
    recorder.record_actionnet_query(acquired, np.ones(6, dtype=np.float32))
    recorder.record_deadline(acquired, np.ones(6, dtype=np.float32))
    adapter.memory.set_time(2400.0)
    recorder.record_deadline(
        adapter.memory.snapshot(), np.ones(6, dtype=np.float32)
    )
    adapter.memory.acquire(
        state, preview, ((0, 0),), timestamp_ms=2400.0
    )

    payload = recorder.payload()
    assert payload["phases"]["pre_gaze_selection"]["sample_count"] == 1
    assert payload["phases"]["post_acquisition"]["valid_counts"] == [4]
    assert payload["phases"]["task_deadline"]["valid_counts"] == [4, 0]
    assert payload["acquisition"]["actual_tile_counts"] == [4, 1]
    assert payload["acquisition"]["new_valid_counts"] == [4, 1]
    assert payload["expiry"]["count"] == 4
    assert payload["expiry"]["unique_expired_tiles"] == 4
    assert payload["expiry"]["unique_reacquired_after_expiry_tiles"] == 1
    assert payload["actor_mask_failures"] == 0
    assert payload["hidden_state_reads"] == 0
