from __future__ import annotations

import io
from pathlib import Path
import random
import numpy as np
import pytest
import torch

from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.eval.gaze_behavior_metrics import read_csv
from final_model.eval.gaze_behavior_progress import ProgressReporter, format_duration
from final_model.eval.gaze_behavior_rollout import RolloutConfig, run_model_rollouts
from final_model.eval.gaze_behavior_suite import parser


CHECKPOINT = Path(
    "final_model/runs/gaze_corrected_memory1200_20m_20260827_141207/"
    "checkpoints/step_017000000.pt"
)


def _reporter(*, enabled=True, total=4, stream=None, clock=lambda: 0.0):
    return ProgressReporter(
        enabled=enabled, total_rollout_episodes=total,
        progress_every_episodes=1, bootstrap_progress_every=20,
        stream=io.StringIO() if stream is None else stream, clock=clock,
    )


def test_progress_enabled_disabled_rollout_trajectory_and_hash_parity(tmp_path):
    if not CHECKPOINT.is_file():
        pytest.skip("repository 17M checkpoint is unavailable")
    outputs = []
    for enabled in (False, True):
        loaded = load_partial_checkpoint_for_evaluation(CHECKPOINT, device="cpu")
        model_dir = tmp_path / str(enabled)
        reporter = _reporter(enabled=enabled, total=1)
        result = run_model_rollouts(
            loaded=loaded,
            config=RolloutConfig(
                model_name="parity", decision_times=(400,), horizons=(2,),
                fixation_shape="plus", episodes=1, episode_length=3,
                seed_start=20260826, gaze_mode="stochastic",
                action_mode="stochastic",
                aoi_precedence=("self", "partner", "preview"),
            ),
            model_dir=model_dir, reporter=reporter,
        )
        requests = read_csv(model_dir / "gaze_requests.csv.gz")
        acquisitions = read_csv(model_dir / "acquisitions.csv.gz")
        outputs.append({
            "episode": result["episodes"][0],
            "request_tiles": [
                (row["requested_target_x"], row["requested_target_y"])
                for row in requests
            ],
            "landing_tiles": [
                (row["fixation_x"], row["fixation_y"])
                for row in acquisitions
            ],
            "actor_hash": result["manifest"]["actor_sha256"],
            "critic_hash": result["manifest"]["critic_sha256"],
            "gaze_hash": result["manifest"]["gaze_sha256"],
        })
    assert outputs[0] == outputs[1]


def test_progress_helper_does_not_touch_rng_states():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    stream = io.StringIO()
    reporter = _reporter(stream=stream)
    reporter.phase(2, "ROLLOUT")
    reporter.start_condition(
        index=1, total=1, decision_time=400, horizon=2,
        episodes=2, episode_length=3,
    )
    reporter.episode(
        model="m", decision_time=400, horizon=2, episode=1, episodes=2,
        deliveries_mean=1.0, acquisitions_per_decision=1.5,
        last_episode={
            "deliveries": 1, "sparse_team_reward": 20,
            "acquisitions": 4, "gaze_requests": 4,
        },
    )
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_global_episode_count_percentage_and_eta():
    now = [0.0]
    stream = io.StringIO()
    reporter = _reporter(total=4, stream=stream, clock=lambda: now[0])
    now[0] = 10.0
    reporter.episode(
        model="m", decision_time=400, horizon=2, episode=1, episodes=2,
        deliveries_mean=2.0, acquisitions_per_decision=1.0,
        last_episode={
            "deliveries": 2, "sparse_team_reward": 40,
            "acquisitions": 3, "gaze_requests": 3,
        },
    )
    output = stream.getvalue()
    assert "global=1/4 (25.0%)" in output
    assert "condition=50.0%" in output
    assert "ETA=00:00:30" in output


def test_eta_formatting():
    assert format_duration(None) == "--:--:--"
    assert format_duration(0) == "00:00:00"
    assert format_duration(3661.9) == "01:01:01"


def test_analysis_only_marks_rollout_skipped():
    stream = io.StringIO()
    reporter = _reporter(stream=stream)
    reporter.skip_rollout("analysis-only mode")
    assert "[PHASE 2/4][ROLLOUT][SKIP] analysis-only mode" in stream.getvalue()


def test_rollout_only_marks_analysis_skipped():
    stream = io.StringIO()
    reporter = _reporter(stream=stream)
    reporter.skip_analysis("rollout-only mode")
    assert "[PHASE 3/4][ANALYSIS][SKIP] rollout-only mode" in stream.getvalue()


def test_reuse_reports_reuse_not_active_rollout():
    stream = io.StringIO()
    reporter = _reporter(stream=stream)
    reporter.reuse("m", "/tmp/data", episodes=4, acquisitions=9)
    output = stream.getvalue()
    assert "[ROLLOUT][REUSE]" in output
    assert "[ROLLOUT] model=" not in output


def test_failure_context_printed_and_exception_reraised():
    stream = io.StringIO()
    reporter = _reporter(stream=stream)
    reporter.context.update({
        "phase": "rollout", "model": "m",
        "condition": "T400/H2", "episode": "1/2",
    })
    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        except Exception as error:
            reporter.failure(error)
            raise
    output = stream.getvalue()
    assert "[FAIL]" in output
    assert "condition : T400/H2" in output
    assert "error     : boom" in output


def test_cli_progress_defaults_and_switches():
    common = [
        "--checkpoint", "model.pt", "--model-name", "m",
        "--output-dir", "/tmp/output",
    ]
    assert parser().parse_args(common).progress is True
    assert parser().parse_args([*common, "--no-progress"]).progress is False
    args = parser().parse_args([
        *common, "--progress-every-episodes", "3",
        "--bootstrap-progress-every", "50",
    ])
    assert args.progress_every_episodes == 3
    assert args.bootstrap_progress_every == 50


class FlushTrackingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        return super().flush()


def test_every_progress_print_flushes():
    stream = FlushTrackingStream()
    reporter = _reporter(stream=stream)
    reporter.emit("one")
    reporter.emit("two")
    assert stream.flush_count == 2
