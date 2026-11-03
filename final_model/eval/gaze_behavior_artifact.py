"""Atomic, fail-closed rollout artifacts for offline gaze analysis."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from contextlib import contextmanager
import zipfile

import numpy as np


ROLLOUT_ARTIFACT_SCHEMA_VERSION = 1
ROLLOUT_MANIFEST = "rollout_manifest.json"
INCOMPLETE_MARKER = ".rollout_artifact.incomplete"
ROLLOUT_DATA_FILES = (
    "episodes.csv",
    "decisions.csv",
    "gaze_requests.csv.gz",
    "acquisitions.csv.gz",
    "gaze_inputs.npz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_output_path(path: Path):
    """Yield a same-directory temporary path and atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def begin_rollout_artifact(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path / INCOMPLETE_MARKER, "incomplete\n")
    atomic_write_text(
        path / ROLLOUT_MANIFEST,
        json.dumps({
            "rollout_artifact_schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
            "rollout_artifact_status": "incomplete",
        }, indent=2, sort_keys=True) + "\n",
    )


def _row_count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def _npz_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as handle:
                version = np.lib.format.read_magic(handle)
                shape, _fortran, _dtype = (
                    np.lib.format.read_array_header_1_0(handle)
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0(handle)
                )
            result[Path(member).stem] = tuple(shape)
    return result


def _validate_episode_contract(path: Path, manifest: Mapping[str, Any]) -> None:
    expected_conditions = {
        (int(time_ms), int(horizon))
        for time_ms in manifest["decision_times"]
        for horizon in manifest["horizons"]
    }
    expected_seeds = set(range(
        int(manifest["seed_start"]),
        int(manifest["seed_start"]) + int(manifest["episodes"]),
    ))
    observed = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            condition = (int(float(row["T"])), int(float(row["H"])))
            observed.setdefault(condition, set()).add(int(float(row["seed"])))
    if set(observed) != expected_conditions:
        raise RuntimeError("rollout T/H conditions do not match the manifest")
    if any(seeds != expected_seeds for seeds in observed.values()):
        raise RuntimeError("rollout seed range does not match the manifest")


def finalize_rollout_artifact(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    files = {}
    for name in ROLLOUT_DATA_FILES:
        candidate = path / name
        if not candidate.is_file():
            raise RuntimeError(f"rollout artifact file is missing: {candidate}")
        entry = {
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
        if name.endswith((".csv", ".csv.gz")):
            entry["rows"] = _row_count(candidate)
        files[name] = entry
    payload = dict(manifest)
    payload.update({
        "rollout_artifact_schema_version": ROLLOUT_ARTIFACT_SCHEMA_VERSION,
        "rollout_artifact_status": "complete",
        "rollout_artifact_files": files,
    })
    atomic_write_text(
        path / ROLLOUT_MANIFEST,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )
    (path / INCOMPLETE_MARKER).unlink(missing_ok=True)
    return payload


def validate_rollout_artifact(
    path: Path, *, allow_legacy: bool = True, verify_hashes: bool = True,
) -> dict[str, Any]:
    artifact = path.expanduser().resolve()
    if (artifact / INCOMPLETE_MARKER).exists():
        raise RuntimeError(f"rollout artifact is marked incomplete: {artifact}")
    manifest_path = artifact / ROLLOUT_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f"rollout artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"malformed rollout artifact manifest: {error}") from error
    version = manifest.get("rollout_artifact_schema_version")
    if version is None:
        if not allow_legacy:
            raise RuntimeError("legacy rollout artifact has no schema version")
    elif version != ROLLOUT_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported rollout artifact schema version: {version}")
    if version is not None and manifest.get("rollout_artifact_status") != "complete":
        raise RuntimeError("rollout artifact status is not complete")
    for name in ROLLOUT_DATA_FILES:
        if not (artifact / name).is_file():
            raise RuntimeError(f"rollout artifact file is missing: {artifact / name}")

    expected_episodes = (
        int(manifest["episodes"])
        * len(manifest["decision_times"])
        * len(manifest["horizons"])
    )
    actual_episodes = _row_count(artifact / "episodes.csv")
    if actual_episodes != expected_episodes:
        raise RuntimeError(
            f"rollout episode count mismatch: {actual_episodes} != {expected_episodes}"
        )
    _validate_episode_contract(artifact / "episodes.csv", manifest)
    expected_decisions = expected_episodes * int(manifest["episode_length"])
    actual_decisions = _row_count(artifact / "decisions.csv")
    if actual_decisions != expected_decisions:
        raise RuntimeError(
            f"rollout decision count mismatch: {actual_decisions} != {expected_decisions}"
        )
    for name, key in (
        ("gaze_requests.csv.gz", "request_count"),
        ("acquisitions.csv.gz", "acquisition_count"),
    ):
        actual = _row_count(artifact / name)
        if actual != int(manifest[key]):
            raise RuntimeError(
                f"rollout {key} mismatch: {actual} != {manifest[key]}"
            )
    shapes = _npz_shapes(artifact / "gaze_inputs.npz")
    if set(shapes) != {"memory", "fixation", "context"}:
        raise RuntimeError("rollout gaze input arrays are incomplete")
    if any(shape[:1] != (int(manifest["request_count"]),) for shape in shapes.values()):
        raise RuntimeError("rollout gaze input row count does not match requests")

    entries = manifest.get("rollout_artifact_files", {})
    if version is not None:
        if set(entries) != set(ROLLOUT_DATA_FILES):
            raise RuntimeError("rollout artifact file inventory is incomplete")
        for name, expected in entries.items():
            candidate = artifact / name
            if candidate.stat().st_size != int(expected["bytes"]):
                raise RuntimeError(f"rollout artifact size mismatch: {name}")
            if verify_hashes and sha256_file(candidate) != expected["sha256"]:
                raise RuntimeError(f"rollout artifact SHA256 mismatch: {name}")
    return manifest
