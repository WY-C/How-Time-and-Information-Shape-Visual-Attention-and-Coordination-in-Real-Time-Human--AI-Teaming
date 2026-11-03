"""Portable repository-root and asset resolution."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT_ENVIRONMENT_VARIABLE = "STUDY_ROOT"
REPOSITORY_MARKERS = ("predict_path", "policy_pool", "model")
ZSCEVAL_SOURCE = "lib/zsc_eval"
POLICY_CONFIG_DIRECTORY = "lib/zsc_eval/zsceval/policy_pool/random3/policy_config"
PREDICTOR_SHA256 = "a7c833d8262fed44c9209e090e92023e3d50cf1f9ffbc722a7418786529d6474"


def _validate_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    missing = [name for name in REPOSITORY_MARKERS if not (resolved / name).exists()]
    if missing:
        raise RuntimeError(f"repository root {resolved} is missing markers {missing}")
    return resolved


def repository_root(start: Path | None = None) -> Path:
    configured = os.environ.get(ROOT_ENVIRONMENT_VARIABLE)
    if configured:
        return _validate_root(Path(configured))
    origin = (start or Path(__file__)).resolve()
    candidates = (origin, *origin.parents)
    for candidate in candidates:
        if all((candidate / marker).exists() for marker in REPOSITORY_MARKERS):
            return candidate
    raise RuntimeError(
        f"cannot discover repository root from {origin}; set {ROOT_ENVIRONMENT_VARIABLE}"
    )


def asset(relative: str | Path, *, must_exist: bool = True) -> Path:
    path = (repository_root() / Path(relative)).resolve()
    if repository_root() not in (path, *path.parents):
        raise RuntimeError(f"asset escapes repository root: {relative}")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predictor_checkpoint() -> Path:
    path = asset("predict_path/output/counter_circuit_1000000.pt")
    actual = file_sha256(path)
    if actual != PREDICTOR_SHA256:
        raise RuntimeError(
            f"predictor checkpoint hash mismatch: expected={PREDICTOR_SHA256}, actual={actual}"
        )
    return path


def zsceval_source_root() -> Path:
    return asset(ZSCEVAL_SOURCE)


def policy_config_directory() -> Path:
    return asset(POLICY_CONFIG_DIRECTORY)


def portability_report() -> dict[str, object]:
    root = repository_root()
    return {
        "root_resolution_priority": [ROOT_ENVIRONMENT_VARIABLE, "repository marker discovery", "fail closed"],
        "resolved_root": str(root),
        "markers": list(REPOSITORY_MARKERS),
        "zsceval_source": ZSCEVAL_SOURCE,
        "resolved_zsceval_source": str(zsceval_source_root()),
        "policy_config_directory": POLICY_CONFIG_DIRECTORY,
        "resolved_policy_config_directory": str(policy_config_directory()),
        "predictor_relative_path": "predict_path/output/counter_circuit_1000000.pt",
        "predictor_sha256": PREDICTOR_SHA256,
    }
