from pathlib import Path
import math
import re
from types import SimpleNamespace

import pytest

from final_model.server_parallel.train_server import (
    _save_new_partial_initialization,
)


class _Session:
    def __init__(self):
        self.calls = []

    def save(self, path, *, checkpoint_role=None):
        output = Path(path)
        output.write_bytes(b"validated-step0")
        self.calls.append((output, checkpoint_role))


def test_new_partial_run_saves_fail_closed_step0(tmp_path, capsys):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    session = _Session()
    output = _save_new_partial_initialization(
        SimpleNamespace(resume_checkpoint=None), session, run_dir
    )
    assert output == run_dir / "checkpoints/step_000000000.pt"
    assert output.read_bytes() == b"validated-step0"
    assert session.calls == [(output, "pre_training_initialization")]
    match = re.fullmatch(
        r"PROFILE_CHECKPOINT step=0 seconds=([0-9.]+)\n",
        capsys.readouterr().out,
    )
    assert match is not None
    assert math.isfinite(float(match.group(1)))
    assert float(match.group(1)) >= 0.0

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        _save_new_partial_initialization(
            SimpleNamespace(resume_checkpoint=None), _Session(), run_dir
        )


def test_partial_resume_does_not_create_step0(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    session = _Session()
    assert _save_new_partial_initialization(
        SimpleNamespace(resume_checkpoint=Path("resume.pt")), session, run_dir
    ) is None
    assert session.calls == []
