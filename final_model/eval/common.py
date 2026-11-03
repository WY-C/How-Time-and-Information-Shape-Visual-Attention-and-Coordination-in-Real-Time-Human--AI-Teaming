"""Shared contracts and output helpers for evaluation-only code."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from final_model.common.paths import asset, file_sha256


EVALUATION_HORIZONS = (0, 2, 4)
ACTION_NAMES = ("NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT")
SELECTION_SEED_START = 300_000
FINAL_ANALYSIS_SEED_START = 400_000
ABLATION_SEED_START = 500_000
COUNTERFACTUAL_SEED_START = 600_000
CANONICAL_PARTNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")
PRODUCTION_SOURCES = (
    "checkpoint.py",
    "common/mappo.py",
    "common/overcooked_backend.py",
    "common/partners.py",
    "simple_ver/observation.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def evaluation_seeds(seed_start: int, episodes: int) -> tuple[int, ...]:
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    return tuple(int(seed_start) + index for index in range(int(episodes)))


def valid_preview_modes(horizon: int) -> tuple[str, ...]:
    value = int(horizon)
    if value == 4:
        return ("native", "h12", "h1", "none")
    if value == 2:
        return ("native", "h1", "none")
    if value == 0:
        return ("native", "none")
    raise ValueError(f"evaluation horizon must be one of {EVALUATION_HORIZONS}")


def preview_channel_count(horizon: int, mode: str) -> int:
    if mode not in valid_preview_modes(horizon):
        raise ValueError(f"preview mode {mode!r} is invalid for H{horizon}")
    return {
        "native": int(horizon),
        "h12": 2,
        "h1": 1,
        "none": 0,
    }[mode]


def mask_full_observation(
    observation: np.ndarray, *, evaluation_horizon: int, preview_mode: str
) -> np.ndarray:
    source = np.asarray(observation, dtype=np.float32)
    if source.shape[-3:] != (8, 5, 25):
        raise ValueError(f"Full observation must end in (8,5,25), got {source.shape}")
    horizon = int(evaluation_horizon)
    if horizon not in EVALUATION_HORIZONS:
        raise ValueError(f"evaluation horizon must be one of {EVALUATION_HORIZONS}")
    if not np.all(source[..., 21 + horizon:25] == 0.0):
        raise RuntimeError("native observation contains information above evaluation horizon")
    active = preview_channel_count(horizon, preview_mode)
    result = np.ascontiguousarray(source.copy(), dtype=np.float32)
    result[..., 21 + active:25] = 0.0
    return result


def create_output_directory(path: Path | None, *, prefix: str) -> Path:
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path(__file__).resolve().parent / "results" / f"{prefix}_{stamp}"
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def default_output_path(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path(__file__).resolve().parent / "results" / f"{prefix}_{stamp}"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    project = package.parent
    files = sorted(
        path for path in package.rglob("*.py") if "__pycache__" not in path.parts
    )
    hashes = {
        str(path.relative_to(project)): file_sha256(path)
        for path in files
    }
    for relative in PRODUCTION_SOURCES:
        path = project / relative
        hashes[f"production/{relative}"] = file_sha256(path)
    return hashes


def device_metadata(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def command_record(argv: Iterable[str] | None = None) -> str:
    values = list(sys.argv if argv is None else argv)
    return shlex.join(values)


def assert_seed_partitions_distinct(*seed_starts: int) -> None:
    if len(set(int(value) for value in seed_starts)) != len(seed_starts):
        raise ValueError("selection and analysis seed partitions must be distinct")
