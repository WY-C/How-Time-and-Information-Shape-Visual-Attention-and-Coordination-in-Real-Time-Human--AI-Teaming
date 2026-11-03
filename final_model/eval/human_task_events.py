"""Corrected canonical task-event detectors for the human benchmark.

All modeled events are rebuilt from eligible raw ``step.csv`` transitions.
The detector never infers the acting player from a simultaneous Human Interact.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd


def held_name(value) -> str | None:
    if pd.isna(value):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed.get("name") if isinstance(parsed, dict) else None


def facing_target(env_state, player_index: int) -> tuple[int, int]:
    player = json.loads(env_state)["players"][int(player_index)]
    return tuple(
        int(position + direction)
        for position, direction in zip(player["position"], player["orientation"])
    )


def pot_count_and_category(value, position) -> tuple[int, str]:
    states = json.loads(value)
    target = list(position)
    for category, positions in states.get("onion", {}).items():
        if target not in positions:
            continue
        if category.endswith("_items"):
            return int(category.split("_", 1)[0]), category
        if category in {"ready", "cooking"}:
            return 3, category
    if target in states.get("empty", []):
        return 0, "empty"
    return 0, "not_onion_pot"


def final_onion_transition_is_ai_owned(
    *,
    ai_held: str | None,
    ai_action: str,
    ai_facing_tile,
    pre_pot_states,
    post_pot_states,
    next_ai_held: str | None,
) -> bool:
    pre_count, _ = pot_count_and_category(pre_pot_states, ai_facing_tile)
    post_count, post_category = pot_count_and_category(
        post_pot_states, ai_facing_tile
    )
    return bool(
        ai_held == "onion"
        and ai_action == "interact"
        and pre_count == 2
        and post_count == 3
        and post_category == "ready"
        and next_ai_held != "onion"
    )


def _metadata(trials: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    columns = ["pid", "Time", "Preview", "Layout", "trial_order_c"]
    return trials.set_index("trial_id")[columns].to_dict("index")


def _common(metadata, trial_id, event_type):
    value = metadata[trial_id]
    return {"event_type": event_type, "trial_id": trial_id, **value}


def _input_diagnostics(group, anchor: int, terminal: int) -> dict[str, Any]:
    window = group.iloc[anchor : terminal + 1]
    has_input = window["first_human_input_ms"].notna()
    return {
        "event_length_decisions": int(len(window)),
        "event_window_inputs": int(has_input.sum()),
        "event_window_input_fraction": float(has_input.mean()),
        "any_input": int(has_input.any()),
        "all_input": int(has_input.all()),
    }


def _final_onion_deposits(steps: pd.DataFrame):
    deposits = []
    cache = {}
    for trial_id, original in steps.groupby("trial_id", sort=True):
        group = original.sort_values("step").reset_index(drop=True)
        human = [held_name(value) for value in group.human_held_object]
        ai = [held_name(value) for value in group.agent_held_object]
        cache[trial_id] = (group, human, ai)
        for index in range(len(group) - 1):
            target = facing_target(group.at[index, "env_state"], 0)
            if final_onion_transition_is_ai_owned(
                ai_held=ai[index],
                ai_action=group.at[index, "agent_action"],
                ai_facing_tile=target,
                pre_pot_states=group.at[index, "pot_states"],
                post_pot_states=group.at[index + 1, "pot_states"],
                next_ai_held=ai[index + 1],
            ):
                deposits.append(
                    {"trial_id": trial_id, "terminal": index, "target": target}
                )
    return deposits, cache


def _canonical_anchor(group, ai, terminal, target):
    candidates = [
        index
        for index in range(max(0, terminal - 3), terminal + 1)
        if ai[index] == "onion"
        and pot_count_and_category(group.at[index, "pot_states"], target)[0] == 2
    ]
    if not candidates:
        raise RuntimeError("validated final-onion deposit has no canonical anchor")
    return min(candidates)


def _find_human_release(human, anchor, terminal):
    for index in range(anchor, terminal + 1):
        if index + 1 < len(human) and human[index] == "onion" and human[index + 1] != "onion":
            return index
    return None


def detect_task_events(
    trials: pd.DataFrame, steps: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    metadata = _metadata(trials)
    deposits, cache = _final_onion_deposits(steps)
    pot_rows, redundant_rows = [], []
    for deposit in deposits:
        trial_id, terminal, target = (
            deposit["trial_id"], deposit["terminal"], deposit["target"]
        )
        group, human, ai = cache[trial_id]
        terminal_step = int(group.at[terminal, "step"])

        if terminal >= 3:
            candidates = []
            for anchor in range(terminal - 3, terminal):
                count, _ = pot_count_and_category(group.at[anchor, "pot_states"], target)
                continuous = all(ai[index] == "onion" for index in range(anchor, terminal + 1))
                if human[anchor] is None and ai[anchor] == "onion" and count == 2 and continuous:
                    candidates.append(anchor)
            if candidates:
                anchor = min(candidates)
                acquisitions = [
                    index
                    for index in range(anchor + 1, min(terminal + 2, len(group)))
                    if human[index] == "dish" and human[index - 1] != "dish"
                ]
                row = _common(metadata, trial_id, "pot_completion")
                row.update(
                    event_id=f"pot_completion::{trial_id}::{terminal_step}",
                    anchor_step=int(group.at[anchor, "step"]),
                    terminal_step=terminal_step,
                    outcome=int(bool(acquisitions)),
                    target_pot=json.dumps(target),
                    same_transition_success=int(bool(acquisitions) and acquisitions[0] == terminal + 1),
                    terminal_at_trial_boundary=0,
                )
                row.update(_input_diagnostics(group, anchor, terminal))
                pot_rows.append(row)

        anchor = _canonical_anchor(group, ai, terminal, target)
        if human[anchor] == "onion":
            release = _find_human_release(human, anchor, terminal)
            row = _common(metadata, trial_id, "redundant_onion_role")
            row.update(
                event_id=f"redundant_onion_role::{trial_id}::{terminal_step}",
                anchor_step=int(group.at[anchor, "step"]),
                terminal_step=terminal_step,
                outcome=int(release is not None),
                target_pot=json.dumps(target),
                release_step=np.nan if release is None else int(group.at[release, "step"]),
                same_transition_success=int(release == terminal),
                terminal_at_trial_boundary=0,
            )
            row.update(_input_diagnostics(group, anchor, terminal))
            redundant_rows.append(row)

    delivery_rows = []
    for trial_id, original in steps.groupby("trial_id", sort=True):
        group = original.sort_values("step").reset_index(drop=True)
        human = [held_name(value) for value in group.human_held_object]
        ai = [held_name(value) for value in group.agent_held_object]
        for terminal in range(len(group)):
            if not (
                ai[terminal] == "soup"
                and group.at[terminal, "agent_action"] == "interact"
                and float(group.at[terminal, "reward"]) > 0
                and (terminal == len(group) - 1 or ai[terminal + 1] != "soup")
            ):
                continue
            soup_start = terminal
            while soup_start > 0 and ai[soup_start - 1] == "soup":
                soup_start -= 1
            pickup = soup_start - 1
            if pickup < 0 or ai[pickup] != "dish":
                continue
            anchors = [
                index for index in range(soup_start, terminal + 1) if human[index] is None
            ]
            if not anchors:
                continue
            anchor = min(anchors)
            acquisitions = [
                index
                for index in range(anchor + 1, min(terminal + 2, len(group)))
                if human[index] == "onion" and human[index - 1] != "onion"
            ]
            terminal_step = int(group.at[terminal, "step"])
            row = _common(metadata, trial_id, "ai_delivery_handoff")
            row.update(
                event_id=f"ai_delivery_handoff::{trial_id}::{terminal_step}",
                anchor_step=int(group.at[anchor, "step"]),
                terminal_step=terminal_step,
                outcome=int(bool(acquisitions)),
                same_transition_success=int(bool(acquisitions) and acquisitions[0] == terminal + 1),
                terminal_at_trial_boundary=int(terminal == len(group) - 1),
            )
            row.update(_input_diagnostics(group, anchor, terminal))
            delivery_rows.append(row)

    result = {
        "pot_completion": pd.DataFrame(pot_rows),
        "ai_delivery_handoff": pd.DataFrame(delivery_rows),
        "redundant_onion_role": pd.DataFrame(redundant_rows),
    }
    for name, frame in result.items():
        if frame.empty:
            result[name] = pd.DataFrame(columns=[
                "event_type", "event_id", "pid", "trial_id", "Time",
                "Preview", "Layout", "trial_order_c", "anchor_step",
                "terminal_step", "outcome", "same_transition_success",
                "terminal_at_trial_boundary", "event_length_decisions",
                "event_window_inputs", "event_window_input_fraction",
                "any_input", "all_input",
            ])
            continue
        if frame.event_id.duplicated().any():
            raise RuntimeError(f"duplicate canonical {name} event")
    return result


def task_event_regression_summary(events: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    return {
        "pot_completion": len(events["pot_completion"]),
        "ai_delivery": len(events["ai_delivery_handoff"]),
        "redundant_onion": len(events["redundant_onion_role"]),
        "terminal_step_179_delivery": int(
            events["ai_delivery_handoff"].terminal_at_trial_boundary.sum()
        ),
        "redundant_release_at_deposit": int(
            events["redundant_onion_role"].same_transition_success.sum()
        ),
    }


def fit_task_event_gee(events: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Fit base and opportunity-adjusted recoverable GEE specifications."""
    import math
    from scipy.stats import norm
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable

    rows = []
    time = "C(Time, Treatment(reference=200))"
    preview = "C(Preview, Treatment(reference=0))"
    for event_type, frame in events.items():
        for model, extra in (
            ("base", ""),
            (
                "opportunity_adjusted",
                " + event_window_input_fraction + event_length_decisions",
            ),
        ):
            formula = f"outcome ~ {time} + {preview} + C(Layout) + trial_order_c{extra}"
            result = smf.gee(
                formula, groups="pid", data=frame,
                family=sm.families.Binomial(link=sm.families.links.logit()),
                cov_struct=Exchangeable(),
            ).fit()
            for coefficient in result.params.index:
                estimate = float(result.params[coefficient])
                se = float(result.bse[coefficient])
                rows.append(
                    {
                        "row_type": "coefficient",
                        "event_type": event_type,
                        "model": model,
                        "formula": formula,
                        "coefficient": coefficient,
                        "estimate": estimate,
                        "robust_se": se,
                        "OR": math.exp(estimate),
                        "CI_low": math.exp(estimate - 1.96 * se),
                        "CI_high": math.exp(estimate + 1.96 * se),
                        "p": float(result.pvalues[coefficient]),
                        "holm_p": np.nan,
                        "groups": "pid",
                        "working_correlation": "Exchangeable",
                        "covariance": str(result.cov_type),
                    }
                )
            names = list(result.params.index)
            time_names = [
                name for name in names if "C(Time" in name and ":" not in name
            ]
            name_400 = next(name for name in time_names if "[T.400]" in name)
            name_600 = next(name for name in time_names if "[T.600]" in name)
            index_400, index_600 = names.index(name_400), names.index(name_600)
            definitions = (
                ("400_vs_200", {index_400: 1.0}),
                ("600_vs_200", {index_600: 1.0}),
                ("600_vs_400", {index_600: 1.0, index_400: -1.0}),
            )
            contrast_rows = []
            parameters = np.asarray(result.params)
            covariance = np.asarray(result.cov_params())
            for contrast, weights in definitions:
                vector = np.zeros(len(names))
                for index, weight in weights.items():
                    vector[index] = weight
                estimate = float(vector @ parameters)
                se = math.sqrt(float(vector @ covariance @ vector))
                p_value = float(2 * norm.sf(abs(estimate / se)))
                contrast_rows.append({
                    "row_type": "time_contrast",
                    "event_type": event_type,
                    "model": model,
                    "formula": formula,
                    "coefficient": contrast,
                    "estimate": estimate,
                    "robust_se": se,
                    "OR": math.exp(estimate),
                    "CI_low": math.exp(estimate - 1.96 * se),
                    "CI_high": math.exp(estimate + 1.96 * se),
                    "p": p_value,
                    "groups": "pid",
                    "working_correlation": "Exchangeable",
                    "covariance": str(result.cov_type),
                })
            order = np.argsort([row["p"] for row in contrast_rows])
            adjusted = np.empty(3)
            running = 0.0
            for rank, index in enumerate(order):
                running = max(running, (3 - rank) * contrast_rows[index]["p"])
                adjusted[index] = min(1.0, running)
            for row, holm_p in zip(contrast_rows, adjusted):
                row["holm_p"] = float(holm_p)
            rows.extend(contrast_rows)
    return pd.DataFrame(rows)
