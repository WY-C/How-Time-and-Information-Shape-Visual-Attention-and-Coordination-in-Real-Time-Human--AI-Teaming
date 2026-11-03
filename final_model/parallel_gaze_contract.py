"""Frozen execution contract for synchronous learned-Gaze rollout workers."""

from __future__ import annotations

import hashlib


PARALLEL_GAZE_TRAINING_CONTRACT = "parallel_gaze_v1"
PARALLEL_DIRECT_JOINT_TRAINING_CONTRACT = "parallel_direct_joint_from_full_v1"
PARALLEL_STAGED_JOINT_TRAINING_CONTRACT = "parallel_staged_joint_v1"
PARALLEL_GAZE_RNG_CONTRACT = "logical_env_episode_torch_generator_sha256_v1"
PARALLEL_GAZE_LOGPROB_AUDIT_VERSION = "cpu_matched_threads_v3"
PARALLEL_GAZE_LEGACY_LOGPROB_AUDIT_VERSION = "cpu_same_device_v2"
PARALLEL_GAZE_CPU_INTRA_OP_THREADS = 1
PARALLEL_GAZE_CPU_INTER_OP_THREADS = 1
PARALLEL_GAZE_CPU_INFERENCE_DTYPE = "float32"
PARALLEL_GAZE_CPU_INFERENCE_PATH = "eval_no_grad_distribution_v1"
PARALLEL_GAZE_HOST_MODE = "local"


def logical_env_gaze_seed(
    master_seed: int, logical_env_id: int, episode_index: int
) -> int:
    """Stable 63-bit seed independent of worker identity and scheduling."""
    values = (int(master_seed), int(logical_env_id), int(episode_index))
    if values[1] < 0 or values[2] < 0:
        raise ValueError("logical env and episode indices must be non-negative")
    payload = ":".join(map(str, values)).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def canonical_gaze_transition_key(record) -> tuple[int, int, int]:
    return (
        int(record["env_index"]),
        int(record["episode_id"]),
        int(record["record_sequence_id"]),
    )


def merge_gaze_transitions(worker_payloads) -> list[dict]:
    records = [
        record
        for payload in worker_payloads
        for record in payload.get("gaze_records", ())
    ]
    ordered = sorted(records, key=canonical_gaze_transition_key)
    keys = [canonical_gaze_transition_key(record) for record in ordered]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate parallel Gaze transition key")
    return ordered
