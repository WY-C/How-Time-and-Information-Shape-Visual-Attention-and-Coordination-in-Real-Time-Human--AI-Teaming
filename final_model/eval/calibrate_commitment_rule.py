"""Fit evaluation-only logit-margin commitment rules from query logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from final_model.eval.benchmark_provenance import file_sha256
from final_model.eval.common import write_json
from final_model.eval.temporal_action import (
    CALIBRATED_COMMITMENT_RULES,
    commitment_threshold,
)


LEGACY_FIXED_DELAY_MS = 87.607603
DEFAULT_DELAY_UPPER_MS = 120.0
FORCED_ERROR_SCALE = 0.10
LATENCY_ERROR_SCALE_MS = 100.0
PROBABILITY_COLUMNS = (
    "p_NORTH", "p_SOUTH", "p_EAST", "p_WEST", "p_STAY", "p_INTERACT",
)
KEY_COLUMNS = ("T", "H", "episode", "seed", "decision")


@dataclass(frozen=True)
class DecisionTrace:
    T: int
    H: int
    episode: int
    seed: int
    decision: int
    query_time_ms: np.ndarray
    logit_margin: np.ndarray
    action: np.ndarray


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integer seeds") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _parameters(rule: str, values: Iterable[float]) -> dict[str, float]:
    vector = tuple(float(value) for value in values)
    if rule == "fixed":
        return {
            "action_input_delay_ms": vector[0],
            "tau": vector[1],
            "tau_min": 0.0,
        }
    if rule == "linear":
        return {
            "action_input_delay_ms": vector[0],
            "tau": 0.0,
            "tau_start": vector[1],
            "tau_min": 0.0,
            "tau_slope_per_ms": vector[2],
        }
    if rule == "exponential":
        return {
            "action_input_delay_ms": vector[0],
            "tau": 0.0,
            "tau_start": vector[1],
            "tau_min": 0.0,
            "tau_kappa_ms": vector[2],
        }
    raise ValueError(f"unsupported rule: {rule}")


def _thresholds(rule: str, elapsed_ms: np.ndarray, parameters: dict[str, float]):
    if rule == "fixed":
        return np.full(elapsed_ms.shape, parameters["tau"], dtype=np.float64)
    if rule == "linear":
        return np.maximum(
            parameters["tau_min"],
            parameters["tau_start"]
            - parameters["tau_slope_per_ms"] * elapsed_ms,
        )
    return parameters["tau_min"] + (
        parameters["tau_start"] - parameters["tau_min"]
    ) * np.exp(-elapsed_ms / parameters["tau_kappa_ms"])


def load_query_traces(path: Path) -> tuple[list[DecisionTrace], str, pd.DataFrame]:
    """Load one trace per decision and recover raw-logit margin when necessary."""
    frame = pd.read_csv(path)
    required = {*KEY_COLUMNS, "query_time_ms", "argmax_action"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"query log is missing required columns: {missing}")
    if frame[list(KEY_COLUMNS) + ["query_time_ms"]].isna().any().any():
        raise ValueError("query keys and timestamps must not contain missing values")

    if "logit_margin" in frame and frame.logit_margin.notna().all():
        evidence_source = "raw_top1_minus_top2_logits"
        margin = frame.logit_margin.to_numpy(dtype=np.float64)
    else:
        absent = sorted(set(PROBABILITY_COLUMNS).difference(frame.columns))
        if absent:
            raise ValueError(
                "query log has neither complete raw logit margins nor six probabilities: "
                f"missing {absent}"
            )
        probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0.0):
            raise ValueError("probabilities must be finite and strictly positive")
        order = np.sort(probabilities, axis=1)
        margin = np.log(order[:, -1]) - np.log(order[:, -2])
        evidence_source = "exact_log_probability_ratio_from_complete_softmax"
    if not np.isfinite(margin).all() or np.any(margin < -1e-10):
        raise ValueError("logit margins must be finite and nonnegative")
    frame = frame.copy()
    frame["reconstructed_logit_margin"] = np.maximum(margin, 0.0)
    frame = frame.sort_values([*KEY_COLUMNS, "query_time_ms"], kind="stable")

    traces = []
    for key, group in frame.groupby(list(KEY_COLUMNS), sort=False):
        T = int(key[0])
        times = group.query_time_ms.to_numpy(dtype=np.float64)
        expected = np.arange(0, T, 20, dtype=np.float64)
        if not np.array_equal(times, expected):
            raise ValueError(f"decision {key} does not contain the canonical 20-ms grid")
        traces.append(DecisionTrace(
            T=T,
            H=int(key[1]),
            episode=int(key[2]),
            seed=int(key[3]),
            decision=int(key[4]),
            query_time_ms=times,
            logit_margin=group.reconstructed_logit_margin.to_numpy(dtype=np.float64),
            action=group.argmax_action.to_numpy(dtype=np.int64),
        ))
    return traces, evidence_source, frame


def load_human_targets(
    decision_summary: Path, latency_summary: Path,
) -> dict[int, dict[str, float]]:
    decision = pd.read_csv(decision_summary)
    latency = pd.read_csv(latency_summary)
    time_column = "Time" if "Time" in decision else "T"
    latency_time_column = "Time" if "Time" in latency else "T"
    if "forced_stay" not in decision:
        raise ValueError("human decision summary must contain forced_stay")
    if "selected_action_latency_ms" not in latency:
        raise ValueError(
            "human latency summary must contain selected_action_latency_ms"
        )
    forced = decision.set_index(time_column).forced_stay.to_dict()
    selected = latency.set_index(latency_time_column).selected_action_latency_ms.to_dict()
    targets = {
        int(T): {
            "forced_stay_rate": float(forced[T]),
            "conditional_action_latency_mean_ms": float(selected[T]),
        }
        for T in sorted(set(forced).intersection(selected))
    }
    if set(targets) != {200, 400, 600}:
        raise ValueError("human summaries must provide T=200, 400, and 600")
    return targets


def simulate_traces(
    traces: Iterable[DecisionTrace],
    *,
    rule: str,
    parameters: dict[str, float],
    action_input_delay_ms: float | None = None,
) -> pd.DataFrame:
    """Replay commitment decisions on fixed, previously logged query trajectories."""
    delay = (
        float(parameters["action_input_delay_ms"])
        if action_input_delay_ms is None else float(action_input_delay_ms)
    )
    if not np.isfinite(delay) or delay < 0.0:
        raise ValueError("action_input_delay_ms must be finite and nonnegative")
    rows = []
    for trace in traces:
        threshold = _thresholds(rule, trace.query_time_ms, parameters)
        eligible = (
            (trace.logit_margin >= threshold)
            & (trace.query_time_ms + delay <= trace.T)
        )
        accepted = np.flatnonzero(eligible)
        if accepted.size == 0:
            rows.append({
                "T": trace.T,
                "H": trace.H,
                "episode": trace.episode,
                "seed": trace.seed,
                "decision": trace.decision,
                "action_input_delay_ms": delay,
                "forced_stay": 1,
                "intentional_stay": 0,
                "canonical_revision": 0,
                "temporal_action_change": 0,
                "selected_action_latency_ms": np.nan,
                "executed_proposal_query_time_ms": np.nan,
                "executed_proposal_arrival_time_ms": np.nan,
            })
            continue
        accepted_actions = trace.action[accepted]
        final_action = int(accepted_actions[-1])
        first_final_index = accepted[np.flatnonzero(accepted_actions == final_action)[0]]
        changes = int(np.count_nonzero(np.diff(accepted_actions)))
        rows.append({
            "T": trace.T,
            "H": trace.H,
            "episode": trace.episode,
            "seed": trace.seed,
            "decision": trace.decision,
            "action_input_delay_ms": delay,
            "forced_stay": 0,
            "intentional_stay": int(final_action == 4),
            "canonical_revision": int(accepted_actions[0] != final_action),
            "temporal_action_change": int(changes > 0),
            "selected_action_latency_ms": (
                float(trace.query_time_ms[first_final_index]) + delay
            ),
            "executed_proposal_query_time_ms": float(trace.query_time_ms[accepted[-1]]),
            "executed_proposal_arrival_time_ms": (
                float(trace.query_time_ms[accepted[-1]]) + delay
            ),
        })
    return pd.DataFrame(rows)


def summarize_simulation(
    decisions: pd.DataFrame,
    *,
    targets: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, float]:
    rows = []
    for T, group in decisions.groupby("T", sort=True):
        latency = group.selected_action_latency_ms.dropna()
        forced_rate = float(group.forced_stay.mean())
        latency_mean = float(latency.mean()) if len(latency) else float("nan")
        forced_error = forced_rate - targets[int(T)]["forced_stay_rate"]
        forced_component = 0.5 * (forced_error / FORCED_ERROR_SCALE) ** 2
        if len(latency):
            latency_error = (
                latency_mean
                - targets[int(T)]["conditional_action_latency_mean_ms"]
            )
            latency_component = 0.5 * (
                latency_error / LATENCY_ERROR_SCALE_MS
            ) ** 2
        else:
            latency_error = float("nan")
            latency_component = float("inf")
        objective = forced_component + latency_component
        rows.append({
            "T": int(T),
            "decisions": int(len(group)),
            "forced_stay_rate": forced_rate,
            "conditional_action_latency_mean_ms": latency_mean,
            "conditional_action_latency_median_ms": (
                float(latency.median()) if len(latency) else float("nan")
            ),
            "conditional_action_latency_sd_ms": (
                float(latency.std(ddof=1)) if len(latency) > 1 else float("nan")
            ),
            "forced_stay_target": targets[int(T)]["forced_stay_rate"],
            "conditional_action_latency_target_ms": targets[int(T)][
                "conditional_action_latency_mean_ms"
            ],
            "forced_stay_target_error": forced_error,
            "forced_stay_error": forced_error,
            "latency_target_error_ms": latency_error,
            "latency_error_ms": latency_error,
            "forced_stay_objective_component": forced_component,
            "latency_objective_component": latency_component,
            "combined_objective": objective,
            "intentional_stay_rate": float(group.intentional_stay.mean()),
            "canonical_revision_rate_non_timeout": (
                float(group.loc[group.forced_stay.eq(0), "canonical_revision"].mean())
                if group.forced_stay.eq(0).any() else float("nan")
            ),
            "temporal_action_change_rate": float(group.temporal_action_change.mean()),
        })
    summary = pd.DataFrame(rows)
    return summary, float(summary.combined_objective.mean())


def fit_rules(
    traces: list[DecisionTrace],
    *,
    calibration_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    query_frame: pd.DataFrame,
    targets: dict[int, dict[str, float]],
    optimizer_seed: int,
    maxiter: int,
    popsize: int,
    stability_restarts: int,
    delay_upper_ms: float,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]],
    pd.DataFrame, pd.DataFrame,
]:
    calibration = [
        trace for trace in traces if trace.seed in calibration_seeds
    ]
    validation = [
        trace for trace in traces if trace.seed in validation_seeds
    ]
    if not calibration or not validation:
        raise ValueError("both calibration and validation splits must contain decisions")

    upper = max(2.0, float(query_frame.reconstructed_logit_margin.quantile(.999)))
    fitted: dict[str, dict[str, float]] = {}
    trace_rows = []
    candidate_rows = []
    model_rows = []
    diagnostic_rows = []
    seed_rows = []

    for rule in CALIBRATED_COMMITMENT_RULES:
        evaluation_index = 0
        current_delay_upper = float(delay_upper_ms)
        expansion_count = 0
        while True:
            bounds = {
                "fixed": ((0.0, current_delay_upper), (0.0, upper)),
                "linear": (
                    (0.0, current_delay_upper), (0.0, upper),
                    (0.0, upper / 100.0),
                ),
                "exponential": (
                    (0.0, current_delay_upper), (0.0, upper), (20.0, 2000.0),
                ),
            }[rule]
            restart_results = []
            for restart in range(stability_restarts):
                restart_seed = optimizer_seed + restart

                def objective(values):
                    nonlocal evaluation_index
                    parameters = _parameters(rule, values)
                    decisions = simulate_traces(
                        calibration, rule=rule, parameters=parameters,
                    )
                    _, score = summarize_simulation(decisions, targets=targets)
                    trace_rows.append({
                        "rule": rule,
                        "restart": restart,
                        "optimizer_seed": restart_seed,
                        "evaluation": evaluation_index,
                        "parameters": json.dumps(parameters, sort_keys=True),
                        "calibration_objective": score,
                    })
                    evaluation_index += 1
                    return score

                result = differential_evolution(
                    objective,
                    bounds=bounds,
                    seed=restart_seed,
                    maxiter=maxiter,
                    popsize=popsize,
                    polish=False,
                    updating="immediate",
                    workers=1,
                )
                restart_results.append(result)
                diagnostic_rows.append({
                    "rule": rule,
                    "restart": restart,
                    "optimizer_seed": restart_seed,
                    "delay_upper_ms": current_delay_upper,
                    "calibration_objective": float(result.fun),
                    "optimizer_success": bool(result.success),
                    "optimizer_message": str(result.message),
                    "parameters": json.dumps(
                        _parameters(rule, result.x), sort_keys=True,
                    ),
                })
            result = min(restart_results, key=lambda item: float(item.fun))
            parameters = _parameters(rule, result.x)
            if (
                parameters["action_input_delay_ms"] >= .99 * current_delay_upper
                and expansion_count < 2
            ):
                current_delay_upper *= 2.0
                expansion_count += 1
                continue
            break

        fitted[rule] = parameters
        objectives = {}
        for split, split_traces in (
            ("calibration", calibration), ("validation", validation),
        ):
            decisions = simulate_traces(
                split_traces, rule=rule, parameters=parameters,
            )
            summary, score = summarize_simulation(decisions, targets=targets)
            objectives[split] = score
            summary.insert(0, "parameter_count", 2 if rule == "fixed" else 3)
            for name in (
                "action_input_delay_ms", "tau", "tau_start",
                "tau_slope_per_ms", "tau_kappa_ms",
            ):
                summary[name] = parameters.get(name, np.nan)
            summary.insert(0, "split", split)
            summary.insert(0, "parameters", json.dumps(parameters, sort_keys=True))
            summary.insert(0, "rule", rule)
            candidate_rows.extend(summary.to_dict("records"))
            for seed, seed_decisions in decisions.groupby("seed", sort=True):
                seed_summary, seed_score = summarize_simulation(
                    seed_decisions, targets=targets,
                )
                seed_summary.insert(0, "seed", int(seed))
                seed_summary.insert(0, "split", split)
                seed_summary.insert(0, "rule", rule)
                seed_summary["seed_level_objective"] = seed_score
                seed_rows.extend(seed_summary.to_dict("records"))
        delay_value = parameters["action_input_delay_ms"]
        delay_span = current_delay_upper
        delay_near_lower = delay_value <= .01 * delay_span
        delay_near_upper = delay_value >= .99 * delay_span
        model_rows.append({
            "rule": rule,
            "parameters": json.dumps(parameters, sort_keys=True),
            "parameter_count": 2 if rule == "fixed" else 3,
            "action_input_delay_ms": delay_value,
            "calibration_objective": objectives["calibration"],
            "validation_objective": objectives["validation"],
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "delay_search_lower_ms": 0.0,
            "delay_search_upper_ms": current_delay_upper,
            "delay_search_expansions": expansion_count,
            "delay_near_lower_boundary": delay_near_lower,
            "delay_near_upper_boundary": delay_near_upper,
        })

    models = pd.DataFrame(model_rows)
    non_exponential = models[models.rule.ne("exponential")].sort_values(
        ["calibration_objective", "parameter_count", "rule"]
    ).iloc[0]
    linear = models[models.rule.eq("linear")].iloc[0]
    exponential = models[models.rule.eq("exponential")].iloc[0]
    clear_exponential_gain = (
        exponential.calibration_objective
        < 0.95 * linear.calibration_objective
    )
    chosen_rule = (
        "exponential"
        if clear_exponential_gain
        and exponential.calibration_objective < non_exponential.calibration_objective
        else str(non_exponential.rule)
    )
    fixed = models[models.rule.eq("fixed")].iloc[0]
    models["relative_improvement_vs_fixed"] = (
        fixed.calibration_objective - models.calibration_objective
    ) / fixed.calibration_objective
    models["relative_improvement_vs_linear"] = (
        linear.calibration_objective - models.calibration_objective
    ) / linear.calibration_objective
    models["validation_relative_improvement_vs_fixed"] = (
        fixed.validation_objective - models.validation_objective
    ) / fixed.validation_objective
    models["validation_relative_improvement_vs_linear"] = (
        linear.validation_objective - models.validation_objective
    ) / linear.validation_objective
    models["selected"] = models.rule.eq(chosen_rule)
    models["best_model"] = models.rule.eq(chosen_rule)
    models["selection_rule"] = (
        "calibration only: exponential must improve over linear by >5%; otherwise "
        "best fixed/linear calibration objective (lower parameter count breaks ties); "
        "validation is confirmation only"
    )
    return (
        pd.DataFrame(candidate_rows), models, pd.DataFrame(trace_rows), fitted,
        pd.DataFrame(diagnostic_rows), pd.DataFrame(seed_rows),
    )


def threshold_trajectories(
    fitted: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for rule, parameters in fitted.items():
        for T in (200, 400, 600):
            for elapsed in range(0, T, 20):
                rows.append({
                    "rule": rule,
                    "T": T,
                    "t_ms": elapsed,
                    "tau": commitment_threshold(
                        rule=rule,
                        elapsed_ms=elapsed,
                        tau=parameters["tau"],
                        tau_start=parameters.get("tau_start"),
                        tau_min=parameters["tau_min"],
                        tau_slope_per_ms=parameters.get("tau_slope_per_ms"),
                        tau_kappa_ms=parameters.get("tau_kappa_ms"),
                    ),
                    "action_input_delay_ms": parameters["action_input_delay_ms"],
                })
    return pd.DataFrame(rows)


def _write_fixed_delay_comparison(
    *,
    reference: Path,
    candidates: pd.DataFrame,
    models: pd.DataFrame,
    output: Path,
) -> None:
    reference = Path(reference)
    old_candidates = pd.read_csv(reference / "candidate_metrics.csv")
    old_models = pd.read_csv(reference / "model_summary.csv")
    metric_columns = [
        "forced_stay_rate", "conditional_action_latency_mean_ms",
        "conditional_action_latency_median_ms", "conditional_action_latency_sd_ms",
        "forced_stay_target_error", "latency_target_error_ms",
        "combined_objective",
    ]
    comparison = old_candidates[["rule", "split", "T", *metric_columns]].merge(
        candidates[["rule", "split", "T", *metric_columns]],
        on=["rule", "split", "T"],
        suffixes=("_fixed_delay_v7", "_joint_delay_v8"),
        validate="one_to_one",
    )
    for metric in metric_columns:
        comparison[f"{metric}_change_joint_minus_fixed"] = (
            comparison[f"{metric}_joint_delay_v8"]
            - comparison[f"{metric}_fixed_delay_v7"]
        )
    comparison.to_csv(output / "fixed_delay_v7_comparison.csv", index=False)

    model_columns = ["rule", "parameters", "calibration_objective", "validation_objective"]
    model_comparison = old_models[model_columns].merge(
        models[model_columns], on="rule",
        suffixes=("_fixed_delay_v7", "_joint_delay_v8"), validate="one_to_one",
    )
    for metric in ("calibration_objective", "validation_objective"):
        model_comparison[f"{metric}_change_joint_minus_fixed"] = (
            model_comparison[f"{metric}_joint_delay_v8"]
            - model_comparison[f"{metric}_fixed_delay_v7"]
        )
    model_comparison.to_csv(
        output / "fixed_delay_v7_model_comparison.csv", index=False,
    )


def run_calibration(
    *,
    queries: Path,
    human_decision_summary: Path,
    human_latency_summary: Path,
    output: Path,
    calibration_seeds: tuple[int, ...] | None = None,
    validation_seeds: tuple[int, ...] | None = None,
    optimizer_seed: int = 20260905,
    maxiter: int = 20,
    popsize: int = 8,
    stability_restarts: int = 3,
    delay_upper_ms: float = DEFAULT_DELAY_UPPER_MS,
    fixed_delay_reference: Path | None = None,
) -> dict[str, object]:
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    traces, evidence_source, query_frame = load_query_traces(queries)
    targets = load_human_targets(human_decision_summary, human_latency_summary)
    available = tuple(sorted(int(value) for value in query_frame.seed.unique()))
    if calibration_seeds is None and validation_seeds is None:
        if len(available) < 2:
            raise ValueError("automatic seed split requires at least two model seeds")
        split = max(1, int(np.ceil(len(available) * .6)))
        split = min(split, len(available) - 1)
        calibration_seeds, validation_seeds = available[:split], available[split:]
    elif calibration_seeds is None or validation_seeds is None:
        raise ValueError("provide both calibration and validation seeds, or neither")
    calibration_seeds = tuple(calibration_seeds)
    validation_seeds = tuple(validation_seeds)
    if set(calibration_seeds).intersection(validation_seeds):
        raise ValueError("calibration and validation seeds must be disjoint")
    requested = set(calibration_seeds).union(validation_seeds)
    if not requested.issubset(available):
        raise ValueError(f"requested seeds absent from query log: {sorted(requested-set(available))}")

    if stability_restarts <= 0:
        raise ValueError("stability_restarts must be positive")
    if not np.isfinite(delay_upper_ms) or delay_upper_ms <= 0.0:
        raise ValueError("delay_upper_ms must be finite and positive")

    candidates, models, search, fitted, diagnostics, seed_metrics = fit_rules(
        traces,
        calibration_seeds=calibration_seeds,
        validation_seeds=validation_seeds,
        query_frame=query_frame,
        targets=targets,
        optimizer_seed=optimizer_seed,
        maxiter=maxiter,
        popsize=popsize,
        stability_restarts=stability_restarts,
        delay_upper_ms=delay_upper_ms,
    )
    candidates.to_csv(output / "candidate_metrics.csv", index=False)
    models.to_csv(output / "model_summary.csv", index=False)
    search.to_csv(output / "search_trace.csv", index=False)
    diagnostics.to_csv(output / "optimizer_diagnostics.csv", index=False)
    seed_metrics.to_csv(output / "seed_metrics.csv", index=False)
    final_diagnostics = diagnostics.merge(
        diagnostics.groupby("rule", as_index=False).delay_upper_ms.max(),
        on=["rule", "delay_upper_ms"], validate="many_to_one",
    )
    stability = final_diagnostics.groupby("rule", as_index=False).agg(
        restarts=("restart", "size"),
        objective_min=("calibration_objective", "min"),
        objective_max=("calibration_objective", "max"),
        objective_sd=("calibration_objective", "std"),
    )
    stability["objective_range"] = stability.objective_max - stability.objective_min
    stability.to_csv(output / "optimization_stability.csv", index=False)
    threshold_trajectories(fitted).to_csv(
        output / "threshold_trajectory.csv", index=False,
    )
    if fixed_delay_reference is not None:
        _write_fixed_delay_comparison(
            reference=fixed_delay_reference,
            candidates=candidates,
            models=models,
            output=output,
        )
    manifest = {
        "queries": str(Path(queries).resolve()),
        "queries_sha256": file_sha256(Path(queries)),
        "human_decision_summary": str(Path(human_decision_summary).resolve()),
        "human_decision_summary_sha256": file_sha256(Path(human_decision_summary)),
        "human_latency_summary": str(Path(human_latency_summary).resolve()),
        "human_latency_summary_sha256": file_sha256(Path(human_latency_summary)),
        "evidence_source": evidence_source,
        "evidence_identity": "top1_logit-top2_logit = log(p_top1/p_top2)",
        "action_input_delay_ms": "jointly fitted once per commitment-rule family",
        "action_input_delay_fitted": True,
        "action_input_delay_global_across": ["T", "Preview", "model variants"],
        "initial_delay_search_bounds_ms": [0.0, delay_upper_ms],
        "boundary_expansion_rule": (
            "double upper bound and refit, up to two times, when D is within 1% "
            "of the upper boundary"
        ),
        "calibration_seeds": list(calibration_seeds),
        "validation_seeds": list(validation_seeds),
        "human_participant_split": "unavailable: aggregate human target summaries used",
        "targets": targets,
        "objective": {
            "per_T": (
                "0.5*(forced_error/0.10)^2 + "
                "0.5*(latency_error_ms/100)^2"
            ),
            "model": "unweighted mean across T=200,400,600",
            "excluded": [
                "delivery", "reward", "revision", "AOI", "gaze allocation",
                "fixation/acquisition", "preview viewing", "task behavior",
            ],
        },
        "optimizer": {
            "name": "scipy.optimize.differential_evolution",
            "seed": optimizer_seed,
            "maxiter": maxiter,
            "popsize": popsize,
            "stability_restarts": stability_restarts,
        },
        "offline_scope": (
            "Exact commitment replay on logged query trajectories. Not an exact "
            "counterfactual environment rollout after an executed action diverges."
        ),
    }
    if fixed_delay_reference is not None:
        reference = Path(fixed_delay_reference).resolve()
        manifest["fixed_delay_v7_reference"] = {
            "directory": str(reference),
            "candidate_metrics_sha256": file_sha256(
                reference / "candidate_metrics.csv"
            ),
            "model_summary_sha256": file_sha256(reference / "model_summary.csv"),
        }
    write_json(output / "calibration_manifest.json", manifest)

    shown = models[[
        "rule", "parameters", "parameter_count", "action_input_delay_ms",
        "calibration_objective", "validation_objective", "selected",
    ]]
    print("\nJoint-delay commitment calibration (lower objective is better)")
    print(shown.to_string(index=False))
    print(f"\nSelected model: {models.loc[models.selected, 'rule'].iloc[0]}")
    print("Selection used calibration only; validation was held out.")
    print("Offline result is provisional because future environment trajectories are fixed.")
    for row in models.itertuples():
        print(
            f"{row.rule}: D={row.action_input_delay_ms:.6f} ms; "
            f"delay_bounds=[{row.delay_search_lower_ms:.1f}, "
            f"{row.delay_search_upper_ms:.1f}]; "
            f"near_boundary={bool(row.delay_near_lower_boundary or row.delay_near_upper_boundary)}"
        )
    print(f"Output: {output}")
    return {
        "output_dir": output,
        "candidate_metrics": candidates,
        "model_summary": models,
        "fitted_parameters": fitted,
        "optimizer_diagnostics": diagnostics,
        "seed_metrics": seed_metrics,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--queries", type=Path, required=True)
    result.add_argument("--human-decision-summary", type=Path, required=True)
    result.add_argument("--human-latency-summary", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--calibration-seeds", type=_parse_seeds)
    result.add_argument("--validation-seeds", type=_parse_seeds)
    result.add_argument("--optimizer-seed", type=int, default=20260905)
    result.add_argument("--maxiter", type=int, default=20)
    result.add_argument("--popsize", type=int, default=8)
    result.add_argument("--stability-restarts", type=int, default=3)
    result.add_argument("--delay-upper-ms", type=float, default=DEFAULT_DELAY_UPPER_MS)
    result.add_argument("--fixed-delay-reference", type=Path)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    run_calibration(
        queries=args.queries,
        human_decision_summary=args.human_decision_summary,
        human_latency_summary=args.human_latency_summary,
        output=args.output,
        calibration_seeds=args.calibration_seeds,
        validation_seeds=args.validation_seeds,
        optimizer_seed=args.optimizer_seed,
        maxiter=args.maxiter,
        popsize=args.popsize,
        stability_restarts=args.stability_restarts,
        delay_upper_ms=args.delay_upper_ms,
        fixed_delay_reference=args.fixed_delay_reference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
