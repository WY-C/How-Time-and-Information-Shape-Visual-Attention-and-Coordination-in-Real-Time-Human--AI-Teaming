from pathlib import Path

import pytest

from final_model.eval.benchmark_checkpoints import load_benchmark_checkpoint_pair


ROOT = Path(__file__).resolve().parents[3]
ACTION_1200 = ROOT / "final_model/checkpoint/action/action_baseline_memory1200.pt"
ACTION_2400 = ROOT / "final_model/checkpoint/action/action_baseline_memory2400.pt"
GAZE_1200 = ROOT / "final_model/checkpoint/gaze/baseline_1200_17m.pt"


@pytest.mark.skipif(not GAZE_1200.is_file(), reason="benchmark checkpoints unavailable")
def test_explicit_same_memory_pair_is_confirmed_and_records_both_hashes():
    pair = load_benchmark_checkpoint_pair(
        action_checkpoint=ACTION_1200, gaze_checkpoint=GAZE_1200
    )
    assert pair.status == "CONFIRMED_EXACT_PAIR"
    manifest = pair.manifest()
    assert manifest["action_checkpoint"]["sha256"]
    assert manifest["gaze_checkpoint"]["sha256"]
    assert manifest["action_checkpoint"]["path"] != manifest["gaze_checkpoint"]["path"]


@pytest.mark.skipif(not GAZE_1200.is_file(), reason="benchmark checkpoints unavailable")
def test_2400_action_with_1200_gaze_is_explicitly_provisional():
    pair = load_benchmark_checkpoint_pair(
        action_checkpoint=ACTION_2400, gaze_checkpoint=GAZE_1200
    )
    assert pair.status == "PROVISIONAL"
    assert pair.differences["memory_lifetime_ms"] == (2400.0, 1200.0)
