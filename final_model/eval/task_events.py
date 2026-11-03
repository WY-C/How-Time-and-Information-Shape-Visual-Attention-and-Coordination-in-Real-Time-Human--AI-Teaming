"""Transition-based task events with counter-coordinate onion provenance."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


AGENT_NAMES = {0: "partner", 1: "learner"}
COUNT_FIELDS = (
    "learner_onion_pickup", "partner_onion_pickup",
    "learner_onion_counter_place", "partner_onion_counter_place",
    "partner_pickup_of_learner_placed_onion",
    "learner_repickup_of_own_placed_onion",
    "learner_pickup_of_partner_placed_onion",
    "partner_repickup_of_own_placed_onion",
    "learner_onion_pot_insert", "partner_onion_pot_insert",
    "learner_dish_pickup", "partner_dish_pickup",
    "learner_soup_pickup", "partner_soup_pickup",
    "learner_delivery", "partner_delivery", "team_delivery",
    "learner_interaction_count", "partner_interaction_count",
    "unresolved_counter_provenance_count", "ambiguous_event_count",
)


def _held_name(player) -> str | None:
    value = getattr(player, "held_object", None)
    return None if value is None else str(value.name)


def _facing_coordinate(player) -> tuple[int, int]:
    return tuple(
        int(position + direction)
        for position, direction in zip(player.position, player.orientation)
    )


def _terrain(mdp, coordinate: tuple[int, int]) -> str | None:
    x, y = coordinate
    matrix = mdp.terrain_mtx
    if y < 0 or y >= len(matrix) or x < 0 or x >= len(matrix[y]):
        return None
    return str(matrix[y][x])


def _object_name(state, coordinate: tuple[int, int]) -> str | None:
    value = state.objects.get(tuple(coordinate))
    return None if value is None else str(value.name)


@dataclass(frozen=True)
class CounterProvenance:
    token: int
    coordinate: tuple[int, int]
    placed_by: str


class TaskEventLogger:
    """Attribute counter onions by slot transitions; slots hold one object at a time."""

    def __init__(self, *, episode: int, evaluation_horizon: int):
        self.episode = int(episode)
        self.evaluation_horizon = int(evaluation_horizon)
        self.counts = Counter({field: 0 for field in COUNT_FIELDS})
        self.events: list[dict[str, Any]] = []
        self.provenance: dict[tuple[int, int], CounterProvenance] = {}
        self._next_token = 1

    def _emit(
        self, timestep: int, actor: str, event_type: str, *,
        object_name: str | None = None,
        coordinate: tuple[int, int] | None = None,
        token: int | None = None,
    ) -> None:
        self.events.append(
            {
                "episode": self.episode,
                "timestep": int(timestep),
                "actor": actor,
                "event_type": event_type,
                "object": object_name,
                "counter_coordinate": (
                    None if coordinate is None else f"{coordinate[0]},{coordinate[1]}"
                ),
                "provenance_token": token,
                "active_evaluation_horizon": self.evaluation_horizon,
            }
        )

    def _candidate_agents(
        self, pre_state, post_state, actions: Sequence[int], coordinate, *, placing: bool
    ) -> list[int]:
        candidates = []
        for index, (before, after, action) in enumerate(
            zip(pre_state.players, post_state.players, actions)
        ):
            if int(action) != 5 or _facing_coordinate(before) != coordinate:
                continue
            before_name, after_name = _held_name(before), _held_name(after)
            if placing and before_name == "onion" and after_name is None:
                candidates.append(index)
            if not placing and before_name is None and after_name == "onion":
                candidates.append(index)
        return candidates

    def observe(
        self, *, timestep: int, pre_state, post_state, actions: Sequence[int], mdp,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        if len(actions) != 2:
            raise ValueError("event logger requires partner, learner joint actions")
        detected_deliveries_this_step = 0
        for index, action in enumerate(actions):
            actor = AGENT_NAMES[index]
            before = pre_state.players[index]
            after = post_state.players[index]
            before_name, after_name = _held_name(before), _held_name(after)
            if int(action) == 5:
                self.counts[f"{actor}_interaction_count"] += 1
            if before_name is None and after_name in {"onion", "dish"}:
                self.counts[f"{actor}_{after_name}_pickup"] += 1
                self._emit(timestep, actor, f"{actor}_{after_name}_pickup", object_name=after_name)
            if before_name == "dish" and after_name == "soup":
                self.counts[f"{actor}_soup_pickup"] += 1
                self._emit(timestep, actor, f"{actor}_soup_pickup", object_name="soup")
            target = _facing_coordinate(before)
            terrain = _terrain(mdp, target)
            if int(action) == 5 and before_name == "onion" and after_name is None and terrain == "P":
                self.counts[f"{actor}_onion_pot_insert"] += 1
                self._emit(timestep, actor, f"{actor}_onion_pot_insert", object_name="onion")
            if int(action) == 5 and before_name == "soup" and after_name is None and terrain == "S":
                self.counts[f"{actor}_delivery"] += 1
                detected_deliveries_this_step += 1
                self._emit(timestep, actor, f"{actor}_delivery", object_name="soup")

        counter_coordinates = {
            tuple(position)
            for position in set(pre_state.objects) | set(post_state.objects) | set(self.provenance)
            if _terrain(mdp, tuple(position)) == "X"
        }
        for coordinate in sorted(counter_coordinates):
            before_name = _object_name(pre_state, coordinate)
            after_name = _object_name(post_state, coordinate)
            token = self.provenance.get(coordinate)
            if token is not None and before_name != "onion":
                self.counts["unresolved_counter_provenance_count"] += 1
                self.provenance.pop(coordinate, None)
                token = None
            if before_name is None and after_name == "onion":
                candidates = self._candidate_agents(
                    pre_state, post_state, actions, coordinate, placing=True
                )
                if len(candidates) != 1:
                    self.counts["ambiguous_event_count"] += 1
                    continue
                actor = AGENT_NAMES[candidates[0]]
                provenance = CounterProvenance(self._next_token, coordinate, actor)
                self._next_token += 1
                self.provenance[coordinate] = provenance
                self.counts[f"{actor}_onion_counter_place"] += 1
                self._emit(
                    timestep, actor, f"{actor}_onion_counter_place",
                    object_name="onion", coordinate=coordinate, token=provenance.token,
                )
            elif before_name == "onion" and after_name is None:
                candidates = self._candidate_agents(
                    pre_state, post_state, actions, coordinate, placing=False
                )
                provenance = self.provenance.pop(coordinate, None)
                if len(candidates) != 1:
                    if provenance is not None:
                        self.counts["unresolved_counter_provenance_count"] += 1
                    self.counts["ambiguous_event_count"] += 1
                    continue
                picker = AGENT_NAMES[candidates[0]]
                if provenance is None:
                    continue
                if provenance.placed_by == "learner" and picker == "partner":
                    field = "partner_pickup_of_learner_placed_onion"
                elif provenance.placed_by == "learner" and picker == "learner":
                    field = "learner_repickup_of_own_placed_onion"
                elif provenance.placed_by == "partner" and picker == "learner":
                    field = "learner_pickup_of_partner_placed_onion"
                else:
                    field = "partner_repickup_of_own_placed_onion"
                self.counts[field] += 1
                self._emit(
                    timestep, picker, field, object_name="onion",
                    coordinate=coordinate, token=provenance.token,
                )
            elif before_name == "onion" and after_name not in {"onion", None}:
                if token is not None:
                    self.counts["unresolved_counter_provenance_count"] += 1
                    self.provenance.pop(coordinate, None)
                self.counts["ambiguous_event_count"] += 1

        result_deliveries = (
            detected_deliveries_this_step
            if result is None else int(result.get("deliveries", 0))
        )
        self.counts["team_delivery"] += result_deliveries
        if result is not None and detected_deliveries_this_step != result_deliveries:
            self.counts["ambiguous_event_count"] += 1

    def summary(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {
            field: int(self.counts[field]) for field in COUNT_FIELDS
        }
        placements = int(values["learner_onion_counter_place"])
        pickups = int(values["learner_onion_pickup"])
        interactions = int(values["learner_interaction_count"]) + int(
            values["partner_interaction_count"]
        )
        values.update(
            {
                "handoff_success_rate": (
                    float(values["partner_pickup_of_learner_placed_onion"]) / placements
                    if placements else 0.0
                ),
                "self_repickup_rate": (
                    float(values["learner_repickup_of_own_placed_onion"]) / placements
                    if placements else 0.0
                ),
                "learner_direct_pot_rate": (
                    float(values["learner_onion_pot_insert"]) / max(pickups, 1)
                ),
                "interaction_efficiency": (
                    float(values["team_delivery"]) / max(interactions, 1)
                ),
            }
        )
        return values
