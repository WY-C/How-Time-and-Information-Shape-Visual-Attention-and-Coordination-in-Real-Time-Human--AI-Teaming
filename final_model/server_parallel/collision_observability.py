"""Read-only Human-compatible joint-transition collision observability."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from final_model.partial_ver.conditions import PartialCondition
from final_model.partial_ver.observability import GROUP_NAMES, condition_metric_id


def human_compatible_joint_transition_collision(
    mdp: Any,
    state: Any,
    joint_action: Sequence[Any],
) -> bool:
    """Match the Human-study pre-transition intended-position definition."""
    actions = tuple(joint_action)
    if len(actions) != len(state.players):
        raise RuntimeError("joint collision action/player count mismatch")
    old_positions = tuple(player.position for player in state.players)
    candidate_positions = tuple(
        mdp._move_if_direction(player.position, player.orientation, action)[0]
        for player, action in zip(state.players, actions)
    )
    return bool(mdp.is_transition_collision(old_positions, candidate_positions))


def initialize_collision_totals(
    totals: Mapping[str, Mapping[str, dict[str, float | int]]],
) -> None:
    for group in GROUP_NAMES:
        for bucket in totals[group].values():
            bucket["collision_count"] = 0


def accumulate_collision_totals(
    totals: Mapping[str, Mapping[str, dict[str, float | int]]],
    *,
    conditions: Sequence[PartialCondition],
    collisions: Sequence[bool | int],
) -> int:
    active = tuple(conditions)
    values = np.asarray(collisions)
    if values.shape != (len(active),):
        raise RuntimeError("invalid Partial collision observability shape")
    if not np.isin(values, (0, 1)).all():
        raise RuntimeError("Partial collision observability must be binary")
    values = values.astype(np.int64, copy=False)
    for condition, collision in zip(active, values):
        labels = (
            ("time_metrics", f"t{int(condition.decision_time)}"),
            ("horizon_metrics", f"h{int(condition.horizon)}"),
            ("condition_metrics", condition_metric_id(condition)),
        )
        for group, label in labels:
            try:
                totals[group][label]["collision_count"] += int(collision)
            except KeyError as error:
                raise RuntimeError(
                    f"inactive Partial collision condition: {group}/{label}"
                ) from error
    return int(values.sum())


def finalize_collision_rates(
    metrics: Mapping[str, Mapping[str, dict[str, float | int | None]]],
) -> None:
    for group in GROUP_NAMES:
        for bucket in metrics[group].values():
            count = int(bucket["collision_count"])
            decisions = int(bucket["task_decisions"])
            if count < 0 or count > decisions:
                raise RuntimeError("Partial collision count exceeds task decisions")
            bucket["collision_rate"] = (
                None if decisions == 0 else float(count) / float(decisions)
            )


def validate_collision_conservation(
    metrics: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    *,
    collision_count: int,
    decisions: int,
) -> None:
    expected_count = int(collision_count)
    expected_decisions = int(decisions)
    if expected_count < 0 or expected_count > expected_decisions:
        raise RuntimeError("overall collision count exceeds task decisions")
    for group in GROUP_NAMES:
        buckets = metrics[group].values()
        actual_count = sum(int(bucket["collision_count"]) for bucket in buckets)
        actual_decisions = sum(
            int(bucket["task_decisions"])
            for bucket in metrics[group].values()
        )
        if actual_count != expected_count or actual_decisions != expected_decisions:
            raise RuntimeError(
                f"Partial {group} collision totals do not reconcile: "
                f"count={(actual_count, expected_count)} "
                f"decisions={(actual_decisions, expected_decisions)}"
            )


__all__ = [
    "accumulate_collision_totals",
    "finalize_collision_rates",
    "human_compatible_joint_transition_collision",
    "initialize_collision_totals",
    "validate_collision_conservation",
]
