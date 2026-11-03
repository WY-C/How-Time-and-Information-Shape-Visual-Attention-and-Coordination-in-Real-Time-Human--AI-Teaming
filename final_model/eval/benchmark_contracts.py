"""Versioned contracts shared by the human--model benchmark.

This module deliberately keeps analysis-specific boundary definitions separate.
They are similar, but the manuscript analyses did not use one universal
definition of a "new" fixation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ACTION_NAMES = ("NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT")
STAY_ACTION = 4
QUERY_INTERVAL_MS = 20
EVAL_ACTION_MODES = (
    "acquisition_triggered_legacy",
    "periodic_20ms_threshold",
    "periodic_20ms_margin_threshold",
    "periodic_20ms_margin_motor_delay",
    "periodic_20ms_margin_motor_delay_stochastic_gaze",
    "periodic_20ms_margin_threshold_stochastic_gaze",
    "periodic_20ms_logit_commitment_stochastic_gaze",
    "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze",
    "periodic_20ms_no_commitment_stochastic_gaze",
)
PRIMARY_AOI_CONTRACT = "manuscript_preview_priority"
SENSITIVITY_AOI_CONTRACT = "v6_overlap_split_sensitivity"
AOI_LABELS = ("Player", "Teammate", "Action Preview", "Other")
CONTEXT_RQ3_STATUS = "CANNOT_VERIFY_PROVENANCE"
BENCHMARK_CONTRACT_VERSION = "contract_v2"
BENCHMARK_CONTRACT_VERSIONS = (
    "contract_v2", "contract_v3", "contract_v4", "contract_v5", "contract_v6",
    "contract_v7", "contract_v8", "contract_v9",
)
BENCHMARK_CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "contracts"
    / f"{BENCHMARK_CONTRACT_VERSION}.json"
)


def benchmark_contract_path(version: str = BENCHMARK_CONTRACT_VERSION) -> Path:
    if version not in BENCHMARK_CONTRACT_VERSIONS:
        raise ValueError(f"unsupported benchmark contract version: {version}")
    return Path(__file__).resolve().parent / "contracts" / f"{version}.json"


def benchmark_contract_snapshot(
    version: str = BENCHMARK_CONTRACT_VERSION,
) -> dict[str, Any]:
    snapshot = json.loads(benchmark_contract_path(version).read_text(encoding="utf-8"))
    if snapshot.get("benchmark_contract_version") != version:
        raise RuntimeError("benchmark contract snapshot/version mismatch")
    return snapshot


def benchmark_contract_sha256(version: str = BENCHMARK_CONTRACT_VERSION) -> str:
    return hashlib.sha256(benchmark_contract_path(version).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ResolvedInput:
    action: int | str | None
    timestamp_ms: float | None
    source: str


def resolve_response_input(
    first_action: int | str | None,
    first_timestamp_ms: float | None,
    final_action: int | str | None,
    final_action_first_selected_ms: float | None,
) -> ResolvedInput:
    """Canonical selected-action input semantics for human or model events.

    ``final_action_first_selected_ms`` is the first time the eventual final
    action became the current input, not the last query before the deadline.
    """
    first_present = first_action is not None and first_timestamp_ms is not None
    final_present = (
        final_action is not None and final_action_first_selected_ms is not None
    )
    if first_present and final_present:
        if first_action == final_action:
            return ResolvedInput(first_action, float(first_timestamp_ms), "first_same_as_final")
        return ResolvedInput(
            final_action, float(final_action_first_selected_ms), "final_after_revision"
        )
    if first_present:
        return ResolvedInput(first_action, float(first_timestamp_ms), "first_only")
    if final_present:
        return ResolvedInput(
            final_action, float(final_action_first_selected_ms), "final_only"
        )
    return ResolvedInput(None, None, "timeout")


def query_times(decision_time_ms: int, interval_ms: int = QUERY_INTERVAL_MS) -> tuple[int, ...]:
    decision_time = int(decision_time_ms)
    interval = int(interval_ms)
    if decision_time <= 0 or interval <= 0:
        raise ValueError("decision time and query interval must be positive")
    return tuple(range(0, decision_time, interval))


def is_carryover_fixation(fixation_start_ms: float, decision_onset_ms: float) -> bool:
    return float(fixation_start_ms) < float(decision_onset_ms)


def is_new_for_carryover_decomposition(
    fixation_start_ms: float, decision_onset_ms: float
) -> bool:
    return float(fixation_start_ms) >= float(decision_onset_ms)


def is_new_for_visual_sampling(
    fixation_start_ms: float, decision_onset_ms: float
) -> bool:
    return float(fixation_start_ms) > float(decision_onset_ms)


def fixation_onset_in_phase(
    fixation_start_ms: float, phase_start_ms: float, phase_end_ms: float
) -> bool:
    return float(phase_start_ms) <= float(fixation_start_ms) < float(phase_end_ms)


def strict_source_check(
    fixation_start_ms: float,
    phase_start_ms: float,
    phase_end_ms: float,
    source_specific_phase_dwell_ms: float,
) -> bool:
    return fixation_onset_in_phase(
        fixation_start_ms, phase_start_ms, phase_end_ms
    ) and float(source_specific_phase_dwell_ms) >= 60.0


def manuscript_aoi(base_aoi: str | None, overlay_flag: bool) -> str:
    """Exclusive manuscript priority: Preview > Player > Teammate > Other."""
    if bool(overlay_flag):
        return "Action Preview"
    if base_aoi == "Self":
        return "Player"
    if base_aoi == "Partner":
        return "Teammate"
    return "Other"


def v6_aoi_allocation(
    base_aoi: str | None, overlay_flag: bool, duration_ms: float
) -> dict[str, float]:
    """V6 sensitivity allocation; never used as the primary paper contract."""
    duration = float(duration_ms)
    result = {label: 0.0 for label in AOI_LABELS}
    base = "Player" if base_aoi == "Self" else "Teammate" if base_aoi == "Partner" else None
    if overlay_flag and base is not None:
        result[base] = duration / 2.0
        result["Action Preview"] = duration / 2.0
    elif overlay_flag:
        result["Action Preview"] = duration
    elif base is not None:
        result[base] = duration
    else:
        result["Other"] = duration
    return result


def allocate_preview_horizons(
    preview_duration_ms: float, matched_horizons: Iterable[int]
) -> dict[int, float]:
    matched = tuple(sorted({int(value) for value in matched_horizons}))
    if any(value not in (1, 2, 3, 4) for value in matched):
        raise ValueError("preview horizons must be H1--H4")
    if not matched:
        return {}
    share = float(preview_duration_ms) / len(matched)
    return {value: share for value in matched}


def euclidean_tile_distance(
    previous: Sequence[float] | None, current: Sequence[float] | None
) -> float | None:
    if previous is None or current is None:
        return None
    try:
        if len(previous) != 2 or len(current) != 2:
            return None
        dx = float(current[0]) - float(previous[0])
        dy = float(current[1]) - float(previous[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(dx) or not math.isfinite(dy):
        return None
    return (dx * dx + dy * dy) ** 0.5


def benchmark_registry() -> list[dict[str, Any]]:
    return [
        {
            "analysis": "context_dependent_rq3_gee",
            "status": CONTEXT_RQ3_STATUS,
            "known_components": [
                "participant-clustered binomial-logit GEE",
                "Exchangeable working correlation and robust covariance",
                "Time, Preview, Layout, centered trial order",
                "Manhattan teammate distance bins, AI Interact, collision, held objects",
            ],
            "missing_components": [
                "exact final formula",
                "exact held-object encoding",
                "original coefficient table/source",
            ],
            "primary_output_available": False,
        }
    ]


def unavailable_model_layout_rows(
    *, model: str, checkpoint: str, layouts: Iterable[str]
) -> list[Mapping[str, Any]]:
    supported = {"counter_circuit"}
    return [
        {
            "model": model,
            "checkpoint": checkpoint,
            "layout": layout,
            "availability": "available" if layout in supported else "unavailable",
            "value": None,
        }
        for layout in layouts
    ]
