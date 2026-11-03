"""Deterministic PID-based counterbalancing for the 27-trial experiment."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple


TIMES_MS = (200, 400, 600)
INFORMATION_STEPS = (0, 2, 4)
LAYOUT_NAMES = {
    "A": "counter_circuit",
    "B": "unident_s_hard_onion",
    "C": "double_rings",
}
LAYOUT_SEQUENCE_ORDERS = {
    "L1": ("S1", "S2", "S3"),
    "L2": ("S2", "S3", "S1"),
    "L3": ("S3", "S1", "S2"),
    "L4": ("S3", "S2", "S1"),
    "L5": ("S2", "S1", "S3"),
    "L6": ("S1", "S3", "S2"),
}
WILLIAMS_BLOCK_OFFSETS = (0, 6, 12)
TRIAL_COUNT = 27
_PID_PATTERN = re.compile(r"^[Pp](\d+)$")


@dataclass(frozen=True)
class TrialSpec:
    participant_id: str
    participant_index: int
    global_trial_index: int
    block_index: int
    block_trial_index: int
    condition_id: str
    time_ms: int
    information_steps: int
    layout_code: str
    layout_name: str
    williams_sequence_index: int
    layout_sequence_id: str
    layout_assignment_id: str

    @property
    def runtime_condition(self) -> str:
        """Return the legacy A-0 condition label consumed by main.py."""
        return f"{self.layout_code}-{int(self.condition_id[1:]) - 1}"


def parse_participant_id(pid: str) -> int:
    """Return a zero-based participant index for a PID such as P01."""
    text = str(pid).strip()
    match = _PID_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("Participant ID must use the form P01, P02, ...")
    participant_number = int(match.group(1))
    if participant_number < 1:
        raise ValueError("Participant ID number must be at least 1.")
    return participant_number - 1


def format_participant_id(participant_index: int) -> str:
    if participant_index < 0:
        raise ValueError("participant_index must be non-negative.")
    return f"P{participant_index + 1:02d}"


def generate_williams_sequences(n: int) -> List[List[int]]:
    """Generate a first-order balanced Williams design for ``n`` conditions."""
    if n < 2:
        raise ValueError("Williams designs require at least two conditions.")

    base = [0]
    low, high = 1, n - 1
    while low <= high:
        base.append(low)
        low += 1
        if low <= high:
            base.append(high)
            high -= 1

    sequences = [[(condition + shift) % n for condition in base] for shift in range(n)]
    if n % 2 == 1:
        sequences.extend([list(reversed(sequence)) for sequence in sequences])
    _validate_williams_sequences(sequences, n)
    return sequences


def _validate_williams_sequences(sequences: Sequence[Sequence[int]], n: int) -> None:
    expected_count = n * 2 if n % 2 else n
    if len(sequences) != expected_count:
        raise ValueError(f"Expected {expected_count} Williams sequences, got {len(sequences)}.")

    expected_conditions = list(range(n))
    for sequence_index, sequence in enumerate(sequences):
        if sorted(sequence) != expected_conditions:
            raise ValueError(f"Williams sequence {sequence_index} is not a permutation of 0..{n - 1}.")

    position_counts = Counter(
        (position, condition)
        for sequence in sequences
        for position, condition in enumerate(sequence)
    )
    expected_position_count = len(sequences) // n
    if set(position_counts.values()) != {expected_position_count}:
        raise ValueError("Williams condition-by-position counts are not balanced.")

    adjacent_counts = Counter(
        pair
        for sequence in sequences
        for pair in zip(sequence, sequence[1:])
    )
    if len(adjacent_counts) != n * (n - 1) or len(set(adjacent_counts.values())) != 1:
        raise ValueError("Williams ordered adjacent pairs are not balanced.")


def get_layout_assignment_tables() -> Dict[str, Tuple[Tuple[str, ...], ...]]:
    return {
        "S1": (("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B")),
        "S2": (("B", "C", "A"), ("C", "A", "B"), ("A", "B", "C")),
        "S3": (("C", "A", "B"), ("A", "B", "C"), ("B", "C", "A")),
    }


def get_layout_sequence_index(participant_index: int) -> int:
    """Pair repeated Williams sequences with opposite layout orders."""
    sequence_count = len(LAYOUT_SEQUENCE_ORDERS)
    cohort_index = participant_index // sequence_count
    return (participant_index % sequence_count + 3 * (cohort_index % 2)) % sequence_count


def get_participant_schedule(pid: str) -> List[TrialSpec]:
    participant_index = parse_participant_id(pid)
    participant_id = format_participant_id(participant_index)
    williams_sequences = generate_williams_sequences(9)
    layout_sequence_index = get_layout_sequence_index(participant_index)
    layout_sequence_id = f"L{layout_sequence_index + 1}"
    layout_assignments = LAYOUT_SEQUENCE_ORDERS[layout_sequence_id]
    assignment_tables = get_layout_assignment_tables()

    schedule = []
    global_trial_index = 1
    for block_offset, sequence_offset in enumerate(WILLIAMS_BLOCK_OFFSETS):
        sequence_index = (participant_index + sequence_offset) % len(williams_sequences)
        assignment_id = layout_assignments[block_offset]
        assignment_table = assignment_tables[assignment_id]
        for block_trial_index, condition_index in enumerate(williams_sequences[sequence_index], start=1):
            time_index, information_index = divmod(condition_index, 3)
            layout_code = assignment_table[time_index][information_index]
            schedule.append(
                TrialSpec(
                    participant_id=participant_id,
                    participant_index=participant_index,
                    global_trial_index=global_trial_index,
                    block_index=block_offset + 1,
                    block_trial_index=block_trial_index,
                    condition_id=f"C{condition_index + 1}",
                    time_ms=TIMES_MS[time_index],
                    information_steps=INFORMATION_STEPS[information_index],
                    layout_code=layout_code,
                    layout_name=LAYOUT_NAMES[layout_code],
                    williams_sequence_index=sequence_index,
                    layout_sequence_id=layout_sequence_id,
                    layout_assignment_id=assignment_id,
                )
            )
            global_trial_index += 1

    validate_participant_schedule(schedule)
    return schedule


def validate_participant_schedule(schedule: Sequence[TrialSpec]) -> None:
    if len(schedule) != TRIAL_COUNT:
        raise ValueError(f"Participant schedule must contain {TRIAL_COUNT} trials, got {len(schedule)}.")
    if len({trial.participant_id for trial in schedule}) != 1:
        raise ValueError("A participant schedule contains multiple participant IDs.")
    if [trial.global_trial_index for trial in schedule] != list(range(1, TRIAL_COUNT + 1)):
        raise ValueError("global_trial_index must run consecutively from 1 through 27.")

    expected_conditions = {f"C{index}" for index in range(1, 10)}
    for block_index in range(1, 4):
        block = [trial for trial in schedule if trial.block_index == block_index]
        if len(block) != 9:
            raise ValueError(f"Block {block_index} must contain 9 trials, got {len(block)}.")
        if {trial.condition_id for trial in block} != expected_conditions:
            raise ValueError(f"Block {block_index} does not contain C1 through C9 exactly once.")
        if [trial.block_trial_index for trial in block] != list(range(1, 10)):
            raise ValueError(f"Block {block_index} has invalid block_trial_index values.")
        if Counter(trial.layout_code for trial in block) != Counter({code: 3 for code in LAYOUT_NAMES}):
            raise ValueError(f"Block {block_index} must use layouts A, B, and C three times each.")

    factorial_cells = Counter(
        (trial.time_ms, trial.information_steps, trial.layout_code) for trial in schedule
    )
    expected_cells = set(product(TIMES_MS, INFORMATION_STEPS, LAYOUT_NAMES))
    if set(factorial_cells) != expected_cells or set(factorial_cells.values()) != {1}:
        raise ValueError("The 27 Time x Information x Layout cells are not present exactly once.")
    _require_counts(schedule, "time_ms", Counter({value: 9 for value in TIMES_MS}))
    _require_counts(schedule, "information_steps", Counter({value: 9 for value in INFORMATION_STEPS}))
    _require_counts(schedule, "layout_code", Counter({value: 9 for value in LAYOUT_NAMES}))

    condition_layout_counts = Counter((trial.condition_id, trial.layout_code) for trial in schedule)
    if len(condition_layout_counts) != 27 or set(condition_layout_counts.values()) != {1}:
        raise ValueError("Each Time x Information condition must use A, B, and C exactly once.")
    _require_counts(
        schedule,
        "layout_assignment_id",
        Counter({assignment_id: 9 for assignment_id in ("S1", "S2", "S3")}),
    )

    for start in range(len(schedule) - 2):
        if len({trial.layout_code for trial in schedule[start : start + 3]}) == 1:
            raise ValueError(f"Layout repeats three times starting at global trial {start + 1}.")
    for previous, current in zip(schedule, schedule[1:]):
        if previous.condition_id == current.condition_id:
            raise ValueError(
                f"Condition {current.condition_id} repeats at global trials "
                f"{previous.global_trial_index} and {current.global_trial_index}."
            )


def _require_counts(schedule: Sequence[TrialSpec], field_name: str, expected: Counter) -> None:
    actual = Counter(getattr(trial, field_name) for trial in schedule)
    if actual != expected:
        raise ValueError(f"Invalid {field_name} counts: expected {dict(expected)}, got {dict(actual)}.")


def validate_all_participants(max_pid: int = 36) -> Dict[str, int]:
    if max_pid < 1:
        raise ValueError("max_pid must be at least 1.")
    schedules = [get_participant_schedule(format_participant_id(index)) for index in range(max_pid)]

    if max_pid == 36:
        for block_index in range(1, 4):
            sequence_counts = Counter(
                schedule[(block_index - 1) * 9].williams_sequence_index for schedule in schedules
            )
            if sequence_counts != Counter({index: 2 for index in range(18)}):
                raise ValueError(f"Williams sequence usage is not balanced in block {block_index}.")

        layout_sequence_counts = Counter(schedule[0].layout_sequence_id for schedule in schedules)
        if layout_sequence_counts != Counter({f"L{index}": 6 for index in range(1, 7)}):
            raise ValueError("L1 through L6 are not assigned to six participants each.")

        for block_index in range(1, 4):
            assignment_counts = Counter(
                schedule[(block_index - 1) * 9].layout_assignment_id for schedule in schedules
            )
            if assignment_counts != Counter({f"S{index}": 12 for index in range(1, 4)}):
                raise ValueError(f"S1 through S3 are not balanced in block {block_index}.")

    position_counts = {
        factor: [Counter(getattr(schedule[position], factor) for schedule in schedules) for position in range(27)]
        for factor in ("time_ms", "information_steps", "layout_code")
    }
    if max_pid == 36:
        for factor in ("time_ms", "information_steps"):
            if any(set(counts.values()) != {12} for counts in position_counts[factor]):
                raise ValueError(f"{factor} is not balanced at every global trial position.")

        expected_layout_position_counts = Counter({code: 12 for code in LAYOUT_NAMES})
        for counts in position_counts["layout_code"]:
            if counts != expected_layout_position_counts:
                raise ValueError("Layout is not balanced at every global trial position.")

    layout_position_values = [
        count
        for position_count in position_counts["layout_code"]
        for count in position_count.values()
    ]
    return {
        "participants": max_pid,
        "schedules": len(schedules),
        "trials": sum(len(schedule) for schedule in schedules),
        "layout_position_min": min(layout_position_values),
        "layout_position_max": max(layout_position_values),
        "layout_position_max_spread": max(
            max(counts.values()) - min(counts.values())
            for counts in position_counts["layout_code"]
        ),
    }


def save_schedule_csv(schedule: Sequence[TrialSpec], output_path: str) -> None:
    validate_participant_schedule(schedule)
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent_dir, exist_ok=True)
    fieldnames = list(TrialSpec.__dataclass_fields__)
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(trial) for trial in schedule)


def print_schedule(schedule: Sequence[TrialSpec]) -> None:
    validate_participant_schedule(schedule)
    header = "Trial  Block  Pos  Cond  Time  Info  Layout  Name                WSeq  LSeq  Assign"
    print(header)
    print("-" * len(header))
    previous_block = None
    for trial in schedule:
        if previous_block is not None and trial.block_index != previous_block:
            print("-" * len(header))
        print(
            f"{trial.global_trial_index:>5}  {trial.block_index:>5}  "
            f"{trial.block_trial_index:>3}  {trial.condition_id:>4}  "
            f"{trial.time_ms:>4}  {trial.information_steps:>4}  "
            f"{trial.layout_code:^6}  {trial.layout_name:<18}  "
            f"{trial.williams_sequence_index:>4}  {trial.layout_sequence_id:>4}  "
            f"{trial.layout_assignment_id:>6}"
        )
        previous_block = trial.block_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", help="Participant ID to preview, such as P01.")
    parser.add_argument("--output", help="Optional CSV output path for --pid.")
    parser.add_argument(
        "--validate-all",
        nargs="?",
        const=36,
        type=int,
        metavar="N",
        help="Validate participants P01 through PN (default: 36).",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    try:
        if args.validate_all is not None:
            report = validate_all_participants(args.validate_all)
            print("Validation passed: " + ", ".join(f"{key}={value}" for key, value in report.items()))
        if args.pid:
            schedule = get_participant_schedule(args.pid)
            print_schedule(schedule)
            if args.output:
                save_schedule_csv(schedule, args.output)
                print(f"Saved schedule: {os.path.abspath(args.output)}")
    except ValueError as exc:
        raise SystemExit(f"Validation failed: {exc}") from None
    if args.validate_all is None and not args.pid:
        raise SystemExit("Specify --pid P01 or --validate-all 36.")


if __name__ == "__main__":
    _main()
