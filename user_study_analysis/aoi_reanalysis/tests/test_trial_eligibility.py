from pathlib import Path

import pandas as pd
import pytest

from common import (
    EXPECTED_DECISION_STEPS_PER_TRIAL,
    build_trial_eligibility_audit,
    load_trials,
)


def _write_complete_design(root: Path, *, make_p01_one_incomplete: bool = False, add_extra_incomplete: bool = False):
    layouts = ["A", "B", "C"]
    for i in range(1, 37):
        pid = f"P{i:02d}"
        rows = []
        k = 0
        for t in [200, 400, 600]:
            for h in [0, 2, 4]:
                for layout in layouts:
                    k += 1
                    steps = EXPECTED_DECISION_STEPS_PER_TRIAL
                    if make_p01_one_incomplete and pid == "P01" and k == 1:
                        steps = EXPECTED_DECISION_STEPS_PER_TRIAL - 1
                    rows.append({
                        "trial_id": f"{pid}_T{t}_H{h}_{layout}",
                        "pid": pid,
                        "completed": 1,
                        "time_constraint_ms": t,
                        "information_steps": h,
                        "layout_code": layout,
                        "decision_steps_completed": steps,
                    })
        if add_extra_incomplete and pid == "P01":
            rows.append({
                "trial_id": f"{pid}_EXTRA_INCOMPLETE",
                "pid": pid,
                "completed": 1,
                "time_constraint_ms": 200,
                "information_steps": 0,
                "layout_code": "A",
                "decision_steps_completed": EXPECTED_DECISION_STEPS_PER_TRIAL - 1,
            })
        p = root / pid
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(p / "trial.csv", index=False)


def test_incomplete_extra_trial_is_excluded_but_complete_27_design_passes(tmp_path):
    _write_complete_design(tmp_path, add_extra_incomplete=True)
    audit = build_trial_eligibility_audit([str(tmp_path)])
    bad = audit[audit["trial_id"].eq("P01_EXTRA_INCOMPLETE")].iloc[0]
    assert not bool(bad["eligible"])
    assert "decision_steps_completed_not_180" in bad["exclusion_reason"]

    valid = load_trials([str(tmp_path)], strict_main=True)
    assert len(valid) == 36 * 27
    assert valid.groupby("pid")["trial_id"].nunique().eq(27).all()
    assert valid["decision_steps_completed"].eq(180).all()


def test_participant_with_only_26_valid_trials_fails_closed(tmp_path):
    _write_complete_design(tmp_path, make_p01_one_incomplete=True)
    with pytest.raises(RuntimeError, match="exactly 27"):
        load_trials([str(tmp_path)], strict_main=True)
