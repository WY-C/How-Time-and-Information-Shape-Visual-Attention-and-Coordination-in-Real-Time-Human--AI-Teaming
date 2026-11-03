"""Required provenance record for every versioned model benchmark run."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from final_model.eval.benchmark_contracts import (
    BENCHMARK_CONTRACT_VERSION,
    PRIMARY_AOI_CONTRACT,
    benchmark_contract_sha256,
)


REQUIRED_MODEL_EVALUATION_FIELDS = (
    "benchmark_contract_version",
    "git_commit_hash",
    "action_checkpoint_sha256",
    "gaze_checkpoint_sha256",
    "tau",
    "motor_delay_ms",
    "seed_manifest_sha256",
    "aoi_contract",
    "evaluation_action_mode",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_provenance(project_root: Path | None = None) -> dict[str, Any]:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        deployed_commit = os.environ.get("FINAL_MODEL_GIT_COMMIT", "")
        if re.fullmatch(r"[0-9a-f]{40}", deployed_commit) is not None:
            return {
                "git_commit_hash": deployed_commit,
                "git_worktree_dirty": True,
            }
        raise RuntimeError("model benchmark requires a readable git commit") from error
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError(f"invalid git commit hash: {commit!r}")
    return {"git_commit_hash": commit, "git_worktree_dirty": bool(status.strip())}


def model_evaluation_provenance(
    *,
    git: dict[str, Any],
    action_checkpoint_sha256: str,
    gaze_checkpoint_sha256: str,
    tau: float,
    seed_manifest_sha256: str,
    evaluation_action_mode: str,
    motor_delay_ms: float = 0.0,
    benchmark_contract_version: str = BENCHMARK_CONTRACT_VERSION,
    commitment_rule: str | None = None,
    tau_start: float | None = None,
    tau_min: float | None = None,
    tau_slope_per_ms: float | None = None,
    tau_kappa_ms: float | None = None,
) -> dict[str, Any]:
    record = {
        "benchmark_contract_version": benchmark_contract_version,
        "benchmark_contract_sha256": benchmark_contract_sha256(
            benchmark_contract_version
        ),
        "git_commit_hash": git["git_commit_hash"],
        "git_worktree_dirty": bool(git.get("git_worktree_dirty", False)),
        "action_checkpoint_sha256": action_checkpoint_sha256,
        "gaze_checkpoint_sha256": gaze_checkpoint_sha256,
        "tau": float(tau),
        "motor_delay_ms": float(motor_delay_ms),
        "action_input_delay_ms": float(motor_delay_ms),
        "seed_manifest_sha256": seed_manifest_sha256,
        "aoi_contract": PRIMARY_AOI_CONTRACT,
        "evaluation_action_mode": evaluation_action_mode,
    }
    if commitment_rule is not None:
        record.update({
            "commitment_rule": commitment_rule,
            "tau_start": tau_start,
            "tau_min": tau_min,
            "tau_slope_per_ms": tau_slope_per_ms,
            "tau_kappa_ms": tau_kappa_ms,
        })
    missing = [key for key in REQUIRED_MODEL_EVALUATION_FIELDS if key not in record]
    if missing:
        raise RuntimeError(f"incomplete model evaluation provenance: {missing}")
    return record
