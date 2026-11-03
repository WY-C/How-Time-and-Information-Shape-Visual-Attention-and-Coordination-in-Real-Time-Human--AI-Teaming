#!/usr/bin/env python3
"""Preflight trial eligibility audit for the canonical 27-trial main-study design."""
from __future__ import annotations

import argparse
import pandas as pd

from common import (
    EXPECTED_DECISION_STEPS_PER_TRIAL,
    EXPECTED_MAIN_TRIALS,
    EXPECTED_TRIALS_PER_CONDITION,
    EXPECTED_TRIALS_PER_PARTICIPANT,
    HORIZON_LEVELS,
    MAIN_PIDS,
    TIME_LEVELS,
    build_trial_eligibility_audit,
    ensure_output_tree,
    save_csv,
    validate_trial_balance,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.05)  # accepted for run_all argument compatibility
    args = p.parse_args()

    paths = ensure_output_tree(args.out)
    audit = build_trial_eligibility_audit(args.roots)
    save_csv(audit, paths["raw"] / "trial_eligibility_audit.csv")

    eligible = audit[audit["eligible"]].copy()
    eligible["Time"] = pd.to_numeric(eligible["Time"], errors="coerce").astype(int)
    eligible["Horizon"] = pd.to_numeric(eligible["Horizon"], errors="coerce").astype(int)
    eligible["decision_steps_completed"] = pd.to_numeric(
        eligible["decision_steps_completed_numeric"], errors="raise"
    ).astype(int)

    # Participant counts are always written, even if strict validation then fails.
    participant_counts = []
    for pid in sorted(MAIN_PIDS):
        g = eligible[eligible["pid"].eq(pid)]
        participant_counts.append({
            "pid": pid,
            "n_valid_trials": int(g["trial_id"].nunique()),
            "expected_valid_trials": EXPECTED_TRIALS_PER_PARTICIPANT,
            "exactly_27": int(g["trial_id"].nunique()) == EXPECTED_TRIALS_PER_PARTICIPANT,
        })
    participant_counts = pd.DataFrame(participant_counts)
    save_csv(participant_counts, paths["qc"] / "trial_count_by_participant.csv")

    cell = (
        eligible.groupby(["pid", "Time", "Horizon"], as_index=False)
        .agg(n_trials=("trial_id", "nunique"), n_layouts=("Layout", "nunique"))
    )
    full_index = pd.MultiIndex.from_product(
        [sorted(MAIN_PIDS), TIME_LEVELS, HORIZON_LEVELS],
        names=["pid", "Time", "Horizon"],
    )
    cell = cell.set_index(["pid", "Time", "Horizon"]).reindex(full_index, fill_value=0).reset_index()
    cell["expected_trials"] = EXPECTED_TRIALS_PER_CONDITION
    cell["expected_layouts"] = EXPECTED_TRIALS_PER_CONDITION
    cell["cell_ok"] = (
        cell["n_trials"].eq(EXPECTED_TRIALS_PER_CONDITION)
        & cell["n_layouts"].eq(EXPECTED_TRIALS_PER_CONDITION)
    )
    save_csv(cell, paths["qc"] / "trial_condition_balance.csv")

    excluded = audit[~audit["eligible"]].copy()
    save_csv(excluded, paths["qc"] / "excluded_trials.csv")

    # This performs the fail-closed checks after audit artifacts have been saved.
    validate_trial_balance(eligible, strict_main=True)

    print(
        "Trial eligibility preflight: PASS | "
        f"eligible={len(eligible)}/{EXPECTED_MAIN_TRIALS}, "
        f"decision_steps_required={EXPECTED_DECISION_STEPS_PER_TRIAL}, "
        f"excluded={len(excluded)}"
    )


if __name__ == "__main__":
    main()
