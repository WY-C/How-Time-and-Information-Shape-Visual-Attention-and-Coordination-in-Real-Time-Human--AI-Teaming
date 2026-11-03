#!/usr/bin/env python3
"""Verify the CAU diag3x3-memory2400 relaxed Action->Gaze smoke run."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    require(len(sys.argv) == 2, f"usage: {sys.argv[0]} SMOKE_OUTPUT_ROOT")
    root = Path(sys.argv[1]).expanduser().resolve() / "diag3x3_memory2400"

    expected = {
        "action": {
            "training_stage": "action",
            "gaze_type": "rule_based",
            "entropy_coefs": [0.2, 0.05, 0.01],
            "entropy_coef_horizons": [0, 10_000_000, 20_000_000],
        },
        "gaze": {
            "training_stage": "gaze",
            "gaze_type": "learned",
            "entropy_coefs": [0.05, 0.02, 0.01, 0.01],
            "entropy_coef_horizons": [0, 5_000_000, 10_000_000, 20_000_000],
        },
    }

    for stage, stage_expected in expected.items():
        run = root / stage
        config_path = run / "config.json"
        checkpoint_path = run / "checkpoints" / "step_000000008.pt"
        require(config_path.is_file(), f"missing {config_path}")
        require(checkpoint_path.is_file(), f"missing {checkpoint_path}")
        require(checkpoint_path.stat().st_size > 0, f"empty {checkpoint_path}")

        config = json.loads(config_path.read_text())
        common_expected = {
            "emma_mode": "relaxed_saccade",
            "peripheral_mode": "local",
            "peripheral_extent": "diag3x3",
            "memory_decay_mode": "stochastic_exponential",
            "memory_tau_ms": 800.0,
            "memory_max_lifetime_ms": 2400.0,
            "condition_name": "diag3x3_memory2400",
            "steps": 8,
        }
        for key, value in {**common_expected, **stage_expected}.items():
            require(
                config.get(key) == value,
                f"{stage} {key}: expected {value!r}, got {config.get(key)!r}",
            )
        print(f"PASS: {stage} config and 8-step checkpoint")

    print("PASS: relaxed diag3x3 memory2400 Action->Gaze smoke verified")


if __name__ == "__main__":
    main()
