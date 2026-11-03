from pathlib import Path

import pytest

from final_model.eval.human_benchmark import run_human_benchmark


DATA = Path(__file__).resolve().parents[3] / "data" / "raw" / "experiment"


@pytest.mark.skipif(not DATA.is_dir(), reason="canonical processed human dataset unavailable")
def test_canonical_human_regressions(tmp_path):
    result = run_human_benchmark(
        data_root=DATA,
        output_dir=tmp_path / "human",
        run_task_gee=False,
    )
    assert result["regression_pass"]
    assert result["process"] == {
        "participants": 35,
        "trials": 907,
        "decisions": 163260,
        "excluded_participants": ["P16"],
        "rule": "session mean validity >= .80 AND trial mean validity >= .90 AND exactly 180 decisions",
    }
    assert result["task_events"] == {
        "pot_completion": 2622,
        "ai_delivery": 3313,
        "redundant_onion": 655,
        "terminal_step_179_delivery": 19,
        "redundant_release_at_deposit": 146,
    }
