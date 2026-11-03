"""Evaluation-only 20-ms ActionNet threshold state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from final_model.eval.benchmark_contracts import (
    ACTION_NAMES,
    QUERY_INTERVAL_MS,
    STAY_ACTION,
    query_times,
    resolve_response_input,
)


CALIBRATED_COMMITMENT_RULES = ("fixed", "linear", "exponential")
COMMITMENT_RULES = (*CALIBRATED_COMMITMENT_RULES, "none")


def commitment_threshold(
    *,
    rule: str,
    elapsed_ms: float,
    tau: float,
    tau_start: float | None = None,
    tau_min: float = 0.0,
    tau_slope_per_ms: float | None = None,
    tau_kappa_ms: float | None = None,
) -> float:
    """Return the global commitment bound at absolute elapsed decision time."""
    name = str(rule)
    elapsed = float(elapsed_ms)
    if name not in COMMITMENT_RULES:
        raise ValueError(f"commitment rule must be one of {COMMITMENT_RULES}")
    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("elapsed_ms must be finite and nonnegative")
    if name == "none":
        raise ValueError("none has no commitment threshold")
    if name == "fixed":
        value = float(tau)
    elif name == "linear":
        value = max(
            float(tau_min),
            float(tau_start) - float(tau_slope_per_ms) * elapsed,
        )
    else:
        value = float(tau_min) + (float(tau_start) - float(tau_min)) * np.exp(
            -elapsed / float(tau_kappa_ms)
        )
    if not np.isfinite(value) or value < 0.0:
        raise RuntimeError("commitment threshold must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class TemporalDecisionResult:
    queries: tuple[dict[str, Any], ...]
    literal_first_threshold_crossing_time_ms: float | None
    first_selected_action: int | None
    first_action_input_time_ms: float | None
    selected_action_input_time_ms: float | None
    selected_action_input_source: str
    final_selected_action: int | None
    executed_proposal_query_time_ms: float | None
    executed_proposal_arrival_time_ms: float | None
    action_change_count: int
    temporal_had_action_change: bool
    canonical_had_revision: bool
    intentional_stay: bool
    forced_stay: bool
    executed_action: int
    selected_candidate_hidden: Any
    last_query_candidate_hidden: Any


class PeriodicThresholdDecisionEngine:
    """Run a frozen-state decision without owning environment transition logic."""

    def __init__(
        self,
        *,
        tau: float,
        interval_ms: int = QUERY_INTERVAL_MS,
        threshold_rule: str = "max_probability",
        motor_delay_ms: float = 0.0,
        commitment_rule: str = "fixed",
        tau_start: float | None = None,
        tau_min: float = 0.0,
        tau_slope_per_ms: float | None = None,
        tau_kappa_ms: float | None = None,
    ) -> None:
        self.tau = float(tau)
        self.interval_ms = int(interval_ms)
        self.threshold_rule = str(threshold_rule)
        self.motor_delay_ms = float(motor_delay_ms)
        self.commitment_rule = str(commitment_rule)
        self.tau_start = None if tau_start is None else float(tau_start)
        self.tau_min = float(tau_min)
        self.tau_slope_per_ms = (
            None if tau_slope_per_ms is None else float(tau_slope_per_ms)
        )
        self.tau_kappa_ms = None if tau_kappa_ms is None else float(tau_kappa_ms)
        if not np.isfinite(self.tau) or self.tau < 0.0:
            raise ValueError("tau must be finite and nonnegative")
        if self.threshold_rule != "top1_top2_logit_margin" and self.tau > 1.0:
            raise ValueError("probability threshold tau must be in [0, 1]")
        if self.interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if not np.isfinite(self.motor_delay_ms) or self.motor_delay_ms < 0.0:
            raise ValueError("motor_delay_ms must be finite and nonnegative")
        if self.commitment_rule == "none" and self.motor_delay_ms != 0.0:
            raise ValueError("none commitment requires zero action-input delay")
        if self.threshold_rule not in {
            "max_probability", "top1_top2_margin", "top1_top2_logit_margin"
        }:
            raise ValueError(f"unsupported threshold rule: {self.threshold_rule}")
        if self.commitment_rule not in COMMITMENT_RULES:
            raise ValueError(f"commitment rule must be one of {COMMITMENT_RULES}")
        if not np.isfinite(self.tau_min) or self.tau_min < 0.0:
            raise ValueError("tau_min must be finite and nonnegative")
        if self.commitment_rule not in {"fixed", "none"}:
            if self.tau_start is None or not np.isfinite(self.tau_start):
                raise ValueError("collapsing rules require finite tau_start")
            if self.tau_start < self.tau_min:
                raise ValueError("tau_start must be at least tau_min")
        if self.commitment_rule == "linear":
            if (
                self.tau_slope_per_ms is None
                or not np.isfinite(self.tau_slope_per_ms)
                or self.tau_slope_per_ms < 0.0
            ):
                raise ValueError("linear rule requires nonnegative tau_slope_per_ms")
        if self.commitment_rule == "exponential":
            if (
                self.tau_kappa_ms is None
                or not np.isfinite(self.tau_kappa_ms)
                or self.tau_kappa_ms <= 0.0
            ):
                raise ValueError("exponential rule requires positive tau_kappa_ms")

    @staticmethod
    def _probabilities(value) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (6,):
            raise ValueError(f"ActionNet must return six probabilities, got {result.shape}")
        if not np.isfinite(result).all() or np.any(result < 0.0):
            raise ValueError("ActionNet probabilities must be finite and nonnegative")
        if not np.isclose(result.sum(), 1.0, atol=1e-6):
            raise ValueError("ActionNet probabilities must sum to one")
        return result

    def run(
        self,
        *,
        decision_onset_ms: float,
        decision_time_ms: int,
        advance_to: Callable[[float], Any],
        query_actionnet: Callable[[float], Mapping[str, Any]],
    ) -> TemporalDecisionResult:
        onset = float(decision_onset_ms)
        deadline = onset + int(decision_time_ms)
        rows: list[dict[str, Any]] = []
        current_action = None
        current_candidate = None
        first_action = None
        first_time = None
        action_change_count = 0
        first_time_by_action: dict[int, float] = {}
        last_candidate = None
        literal_first_crossing_time = None
        current_proposal_query_time = None
        current_proposal_arrival_time = None
        pending_inputs: list[tuple[float, float, int, Any]] = []

        def apply_pending(relative_ms: float) -> bool:
            nonlocal current_action, current_candidate, first_action, first_time
            nonlocal action_change_count
            nonlocal current_proposal_query_time, current_proposal_arrival_time
            changed = False
            while pending_inputs and pending_inputs[0][0] <= relative_ms:
                input_time, query_time, action, candidate = pending_inputs.pop(0)
                first_time_by_action.setdefault(action, input_time)
                current_proposal_query_time = query_time
                current_proposal_arrival_time = input_time
                if current_action is None:
                    current_action = action
                    current_candidate = candidate
                    first_action = action
                    first_time = input_time
                elif action != current_action:
                    current_action = action
                    current_candidate = candidate
                    action_change_count += 1
                    changed = True
                else:
                    current_candidate = candidate
            return changed

        for relative_ms in query_times(decision_time_ms, self.interval_ms):
            absolute_ms = onset + relative_ms
            # This ordering is the equality-boundary contract: acquisitions at
            # exactly absolute_ms are applied before the actor query.
            advance_to(absolute_ms)
            changed = apply_pending(float(relative_ms))
            response = dict(query_actionnet(absolute_ms))
            probabilities = self._probabilities(response["probabilities"])
            available = np.asarray(response.get("available_actions", np.ones(6)))
            if available.shape != (6,) or not np.all(available == 1):
                raise RuntimeError("periodic benchmark forbids legal-action masking")
            candidate = response.get("candidate_hidden")
            last_candidate = candidate
            argmax_action = int(np.argmax(probabilities))
            argmax_probability = float(probabilities[argmax_action])
            top2_probability = float(np.partition(probabilities, -2)[-2])
            top1_top2_margin = argmax_probability - top2_probability
            logits = response.get("logits")
            if logits is None:
                top1_logit = top2_logit = logit_margin = float("nan")
            else:
                logits = np.asarray(logits, dtype=np.float64)
                if logits.shape != (6,) or not np.all(np.isfinite(logits)):
                    raise ValueError("ActionNet logits must be six finite values")
                logit_order = np.argsort(logits)
                logit_argmax = int(logit_order[-1])
                if (
                    logit_argmax != argmax_action
                    and logits[logit_argmax] != logits[argmax_action]
                ):
                    raise RuntimeError(
                        "ActionNet logits and probabilities disagree on top1 | "
                        f"logits={logits.tolist()} probabilities={probabilities.tolist()} "
                        f"logit_top1={logit_argmax} probability_top1={argmax_action}"
                    )
                top1_logit = float(logits[logit_order[-1]])
                top2_logit = float(logits[logit_order[-2]])
                logit_margin = top1_logit - top2_logit
            if self.threshold_rule == "max_probability":
                threshold_score = argmax_probability
            elif self.threshold_rule == "top1_top2_margin":
                threshold_score = top1_top2_margin
            else:
                if logits is None:
                    raise ValueError("logit-margin threshold requires raw ActionNet logits")
                threshold_score = logit_margin
            no_commitment = self.commitment_rule == "none"
            threshold_at_query = (
                float("nan")
                if no_commitment
                else commitment_threshold(
                    rule=self.commitment_rule,
                    elapsed_ms=float(relative_ms),
                    tau=self.tau,
                    tau_start=self.tau_start,
                    tau_min=self.tau_min,
                    tau_slope_per_ms=self.tau_slope_per_ms,
                    tau_kappa_ms=self.tau_kappa_ms,
                )
            )
            crossed = no_commitment or threshold_score >= threshold_at_query
            scheduled_input_time = None
            if crossed:
                if not no_commitment and literal_first_crossing_time is None:
                    literal_first_crossing_time = float(relative_ms)
                scheduled_input_time = float(relative_ms) + self.motor_delay_ms
                pending_inputs.append(
                    (scheduled_input_time, float(relative_ms), argmax_action, candidate)
                )
                changed = apply_pending(float(relative_ms)) or changed
            row = {
                "query_time_ms": float(relative_ms),
                "absolute_query_time_ms": absolute_ms,
                "argmax_action": argmax_action,
                "argmax_action_name": ACTION_NAMES[argmax_action],
                "argmax_probability": argmax_probability,
                "top2_probability": top2_probability,
                "top1_top2_margin": top1_top2_margin,
                "top1_logit": top1_logit,
                "top2_logit": top2_logit,
                "logit_margin": logit_margin,
                "threshold_rule": self.threshold_rule,
                "threshold_score": threshold_score,
                "threshold_at_query": threshold_at_query,
                "threshold_crossing": bool(crossed),
                "proposal_accepted": bool(crossed),
                "action_input_delay_ms": self.motor_delay_ms,
                "proposal_query_time_ms": (
                    float(relative_ms) if crossed else None
                ),
                "proposal_arrival_time_ms": scheduled_input_time,
                "motor_delay_ms": self.motor_delay_ms,
                "scheduled_input_time_ms": scheduled_input_time,
                "scheduled_absolute_input_time_ms": (
                    None if scheduled_input_time is None else onset + scheduled_input_time
                ),
                "input_arrives_before_deadline": bool(
                    scheduled_input_time is not None
                    and scheduled_input_time <= decision_time_ms
                ),
                "selected_input_changed": changed,
                "current_selected_action": current_action,
            }
            row.update({f"p_{ACTION_NAMES[index]}": float(value) for index, value in enumerate(probabilities)})
            rows.append(row)

        # Gaze/memory continues to the deadline, but no ActionNet query occurs there.
        advance_to(deadline)
        apply_pending(float(decision_time_ms))
        final_action = current_action
        resolved = resolve_response_input(
            first_action,
            first_time,
            final_action,
            None if final_action is None else first_time_by_action[final_action],
        )
        forced = final_action is None
        executed = STAY_ACTION if forced else int(final_action)
        intentional = not forced and executed == STAY_ACTION
        canonical_revision = (
            first_action is not None
            and final_action is not None
            and first_action != final_action
        )
        return TemporalDecisionResult(
            queries=tuple(rows),
            literal_first_threshold_crossing_time_ms=literal_first_crossing_time,
            first_selected_action=first_action,
            first_action_input_time_ms=first_time,
            selected_action_input_time_ms=resolved.timestamp_ms,
            selected_action_input_source=resolved.source,
            final_selected_action=final_action,
            executed_proposal_query_time_ms=current_proposal_query_time,
            executed_proposal_arrival_time_ms=current_proposal_arrival_time,
            action_change_count=action_change_count,
            temporal_had_action_change=action_change_count > 0,
            canonical_had_revision=canonical_revision,
            intentional_stay=intentional,
            forced_stay=forced,
            executed_action=executed,
            selected_candidate_hidden=current_candidate,
            last_query_candidate_hidden=last_candidate,
        )
