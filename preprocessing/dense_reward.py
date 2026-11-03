"""Reconstruct official Overcooked shaping rewards from experiment step logs."""

from __future__ import annotations

import csv
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERCOOKED_ROOT = REPO_ROOT / "lib" / "overcooked_ai"
if str(OVERCOOKED_ROOT) not in sys.path:
    sys.path.insert(0, str(OVERCOOKED_ROOT))

from overcooked_ai_py.data.layouts import read_layout_dict  # noqa: E402
from overcooked_ai_py.mdp.actions import Action, Direction  # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import (  # noqa: E402
    BASE_REW_SHAPING_PARAMS,
    OvercookedGridworld,
    OvercookedState,
)


DENSE_REWARD_SCHEMA_VERSION = 1
EVENT_KEYS = (
    "ingredient_placement",
    "useful_dish_pickup",
    "soup_pickup",
    "delivery",
)
EVENT_WEIGHTS = {
    "ingredient_placement": float(BASE_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"]),
    "useful_dish_pickup": float(BASE_REW_SHAPING_PARAMS["DISH_PICKUP_REWARD"]),
    "soup_pickup": float(BASE_REW_SHAPING_PARAMS["SOUP_PICKUP_REWARD"]),
}

ACTION_BY_NAME = {
    "north": Direction.NORTH,
    "south": Direction.SOUTH,
    "east": Direction.EAST,
    "west": Direction.WEST,
    "stay": Action.STAY,
    "interact": Action.INTERACT,
}

DECISION_DERIVED_FIELDS = [
    "dense_reward_schema_version",
    "replay_sparse_reward",
    "sparse_reward_matches_log",
]
for role in ("human", "agent", "team"):
    DECISION_DERIVED_FIELDS.extend(f"{role}_{event}_count" for event in EVENT_KEYS)
DECISION_DERIVED_FIELDS.extend(
    [
        "human_shaped_reward",
        "agent_shaped_reward",
        "team_shaped_reward",
        "human_dense_reward",
        "agent_dense_reward",
        "team_dense_reward",
        "team_productive_event_count",
        "cumulative_team_dense_reward",
    ]
)

TRIAL_DERIVED_FIELDS = [
    "dense_reward_schema_version",
    "dense_decision_count",
    "sparse_reward_mismatch_count",
]
for role in ("human", "agent", "team"):
    TRIAL_DERIVED_FIELDS.extend(f"{role}_{event}_count" for event in EVENT_KEYS)
TRIAL_DERIVED_FIELDS.extend(
    [
        "human_shaped_reward",
        "agent_shaped_reward",
        "team_shaped_reward",
        "human_dense_reward",
        "agent_dense_reward",
        "team_dense_reward",
        "team_productive_event_count",
        "team_productive_action_rate",
    ]
)


@dataclass(frozen=True)
class DenseRewardBuildResult:
    decision_count: int
    trial_count: int
    sparse_reward_mismatch_count: int
    output_paths: Dict[str, Path]


def _read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc


def _parse_joint_action(row: Mapping[str, str], ai_player_index: int) -> tuple[Any, Any]:
    serialized = str(row.get("joint_action", "")).strip()
    if serialized:
        names = json.loads(serialized)
    else:
        names = ["stay", "stay"]
        names[ai_player_index] = row.get("agent_action", "stay")
        names[1 - ai_player_index] = row.get("human_action", "stay")
    if not isinstance(names, list) or len(names) != 2:
        raise ValueError(f"joint_action must contain two actions: {serialized!r}")
    try:
        return tuple(ACTION_BY_NAME[str(name).strip().lower()] for name in names)
    except KeyError as exc:
        raise ValueError(f"Unsupported action name in joint_action: {names!r}") from exc


def _build_mdp(layout_name: str) -> OvercookedGridworld:
    layout_params = read_layout_dict(layout_name)
    grid = [row.strip() for row in layout_params.pop("grid").split("\n")]
    valid_args = set(inspect.signature(OvercookedGridworld.__init__).parameters)
    base_params = {key: value for key, value in layout_params.items() if key in valid_args}
    base_params["layout_name"] = layout_name
    base_params["rew_shaping_params"] = dict(BASE_REW_SHAPING_PARAMS)
    return OvercookedGridworld.from_grid(grid, base_params, {})


def _interaction_terrain(mdp: OvercookedGridworld, state: OvercookedState, player_index: int) -> str:
    player = state.players[player_index]
    interaction_position = Action.move_in_direction(player.position, player.orientation)
    return mdp.get_terrain_type_at_pos(interaction_position)


def _assign_first_candidates(candidates: Sequence[int], count: int) -> List[int]:
    if count < 0 or count > len(candidates):
        raise ValueError(f"Cannot attribute {count} events to candidate slots {list(candidates)}")
    assigned = [0, 0]
    for player_index in candidates[:count]:
        assigned[player_index] += 1
    return assigned


def _integer_event_count(value: float, unit: float, event_name: str) -> int:
    if unit <= 0:
        raise ValueError(f"Invalid reward unit for {event_name}: {unit}")
    count = int(round(value / unit))
    if not math.isclose(value, count * unit, abs_tol=1e-6):
        raise ValueError(f"Reward {value} is not an integer number of {event_name} events ({unit} each)")
    return count


def _replay_row(
    row: Mapping[str, str],
    mdp: OvercookedGridworld,
    *,
    ai_player_index: int,
) -> Dict[str, Any]:
    try:
        state_dict = json.loads(row["env_state"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid env_state at trial={row.get('trial_id')} step={row.get('step')}"
        ) from exc
    state = OvercookedState.from_dict(state_dict)
    joint_action = _parse_joint_action(row, ai_player_index)
    next_state, sparse_reward, shaped_reward = mdp.get_state_transition(state, joint_action)
    ml_actions = list(getattr(next_state, "ml_actions", [None, None]))

    placement_by_player = []
    soup_pickup_by_player = []
    for player_index, action in enumerate(ml_actions):
        before_object = state.players[player_index].held_object
        after_object = next_state.players[player_index].held_object
        placement_by_player.append(
            int(
                action in {"put_onion_in_pot", "put_tomato_in_pot"}
                and before_object is not None
                and before_object.name in {"onion", "tomato"}
                and after_object is None
            )
        )
        soup_pickup_by_player.append(
            int(
                action == "fill_dish_with_soup"
                and before_object is not None
                and before_object.name == "dish"
                and after_object is not None
                and after_object.name == "soup"
            )
        )
    dish_candidates = [
        player_index
        for player_index, action in enumerate(joint_action)
        if action == Action.INTERACT
        and state.players[player_index].held_object is None
        and _interaction_terrain(mdp, state, player_index) == "D"
    ]
    known_shaped_reward = (
        sum(placement_by_player) * EVENT_WEIGHTS["ingredient_placement"]
        + sum(soup_pickup_by_player) * EVENT_WEIGHTS["soup_pickup"]
    )
    useful_dish_count = _integer_event_count(
        float(shaped_reward) - known_shaped_reward,
        EVENT_WEIGHTS["useful_dish_pickup"],
        "useful dish pickup",
    )
    useful_dish_by_player = _assign_first_candidates(dish_candidates, useful_dish_count)

    delivery_candidates = [index for index, action in enumerate(ml_actions) if action == "deliver_soup"]
    delivery_count = _integer_event_count(float(sparse_reward), float(mdp.delivery_reward), "delivery")
    delivery_by_player = _assign_first_candidates(delivery_candidates, delivery_count)

    event_counts_by_player = []
    for player_index in range(2):
        event_counts_by_player.append(
            {
                "ingredient_placement": placement_by_player[player_index],
                "useful_dish_pickup": useful_dish_by_player[player_index],
                "soup_pickup": soup_pickup_by_player[player_index],
                "delivery": delivery_by_player[player_index],
            }
        )

    human_index = 1 - ai_player_index
    role_counts = {
        "human": event_counts_by_player[human_index],
        "agent": event_counts_by_player[ai_player_index],
    }
    role_counts["team"] = {
        event: role_counts["human"][event] + role_counts["agent"][event]
        for event in EVENT_KEYS
    }

    result: Dict[str, Any] = {
        "dense_reward_schema_version": DENSE_REWARD_SCHEMA_VERSION,
        "replay_sparse_reward": float(sparse_reward),
    }
    logged_sparse_reward = _float(row.get("reward"), "reward")
    result["sparse_reward_matches_log"] = int(
        math.isclose(logged_sparse_reward, float(sparse_reward), abs_tol=1e-6)
    )
    for role, counts in role_counts.items():
        for event, count in counts.items():
            result[f"{role}_{event}_count"] = count

    for role in ("human", "agent"):
        counts = role_counts[role]
        shaped = sum(counts[event] * EVENT_WEIGHTS[event] for event in EVENT_WEIGHTS)
        sparse = counts["delivery"] * float(mdp.delivery_reward)
        result[f"{role}_shaped_reward"] = shaped
        result[f"{role}_dense_reward"] = sparse + shaped
    result["team_shaped_reward"] = result["human_shaped_reward"] + result["agent_shaped_reward"]
    result["team_dense_reward"] = result["human_dense_reward"] + result["agent_dense_reward"]
    result["team_productive_event_count"] = sum(role_counts["team"].values())

    if not math.isclose(result["team_shaped_reward"], float(shaped_reward), abs_tol=1e-6):
        raise ValueError(
            "Reconstructed shaping reward does not match the engine at "
            f"trial={row.get('trial_id')} step={row.get('step')}: "
            f"reconstructed={result['team_shaped_reward']} engine={shaped_reward}"
        )
    return result


def build_dense_reward_dataset(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    step_file: Path | str | None = None,
    trial_file: Path | str | None = None,
    ai_player_index: int = 0,
    allow_sparse_reward_mismatch: bool = False,
) -> DenseRewardBuildResult:
    """Replay completed experiment decisions and write dense reward derivatives."""
    if ai_player_index not in (0, 1):
        raise ValueError("ai_player_index must be 0 or 1")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    step_path = Path(step_file) if step_file is not None else input_dir / "step.csv"
    trial_path = Path(trial_file) if trial_file is not None else input_dir / "trial.csv"

    step_fields, step_rows = _read_csv(step_path)
    trial_fields, all_trial_rows = _read_csv(trial_path)
    required_step_fields = {"pid", "trial_id", "step", "env_state", "reward"}
    required_trial_fields = {"pid", "trial_id", "layout", "completed"}
    if missing := required_step_fields.difference(step_fields):
        raise ValueError(f"step CSV is missing fields: {sorted(missing)}")
    if missing := required_trial_fields.difference(trial_fields):
        raise ValueError(f"trial CSV is missing fields: {sorted(missing)}")

    trial_rows = [row for row in all_trial_rows if row.get("trial_id") and _is_truthy(row.get("completed"))]
    trial_by_id = {row["trial_id"]: row for row in trial_rows}
    if len(trial_by_id) != len(trial_rows):
        raise ValueError("Completed trial IDs must be unique")

    mdp_by_layout: Dict[str, OvercookedGridworld] = {}
    cumulative_dense_by_trial: Dict[str, float] = {}
    decision_rows: List[Dict[str, Any]] = []
    mismatch_count = 0
    for row in step_rows:
        trial_id = row.get("trial_id", "")
        trial = trial_by_id.get(trial_id)
        if trial is None:
            continue
        layout_name = trial["layout"]
        if layout_name not in mdp_by_layout:
            mdp_by_layout[layout_name] = _build_mdp(layout_name)
        derived = _replay_row(row, mdp_by_layout[layout_name], ai_player_index=ai_player_index)
        mismatch_count += int(not derived["sparse_reward_matches_log"])
        cumulative_dense_by_trial[trial_id] = (
            cumulative_dense_by_trial.get(trial_id, 0.0) + derived["team_dense_reward"]
        )
        derived["cumulative_team_dense_reward"] = cumulative_dense_by_trial[trial_id]
        decision_rows.append({**row, **derived})

    if mismatch_count and not allow_sparse_reward_mismatch:
        raise ValueError(
            f"Replayed sparse reward disagrees with the log in {mismatch_count} decisions. "
            "No output was written; inspect the layout/version before overriding this check."
        )

    decisions_by_trial: Dict[str, List[Mapping[str, Any]]] = {}
    for row in decision_rows:
        decisions_by_trial.setdefault(str(row["trial_id"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for trial in trial_rows:
        rows = decisions_by_trial.get(trial["trial_id"], [])
        summary: Dict[str, Any] = dict(trial)
        summary.update(
            {
                "dense_reward_schema_version": DENSE_REWARD_SCHEMA_VERSION,
                "dense_decision_count": len(rows),
                "sparse_reward_mismatch_count": sum(
                    int(not row["sparse_reward_matches_log"]) for row in rows
                ),
            }
        )
        for role in ("human", "agent", "team"):
            for event in EVENT_KEYS:
                field = f"{role}_{event}_count"
                summary[field] = sum(int(row[field]) for row in rows)
        for field in (
            "human_shaped_reward",
            "agent_shaped_reward",
            "team_shaped_reward",
            "human_dense_reward",
            "agent_dense_reward",
            "team_dense_reward",
            "team_productive_event_count",
        ):
            summary[field] = sum(float(row[field]) for row in rows)
        summary["team_productive_action_rate"] = (
            summary["team_productive_event_count"] / len(rows) if rows else ""
        )
        summary_rows.append(summary)

    decision_path = output_dir / "decision_dense_reward.csv"
    summary_path = output_dir / "trial_dense_reward_summary.csv"
    metadata_path = output_dir / "dense_reward_metadata.json"
    _write_csv(decision_path, [*step_fields, *DECISION_DERIVED_FIELDS], decision_rows)
    _write_csv(summary_path, [*trial_fields, *TRIAL_DERIVED_FIELDS], summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": DENSE_REWARD_SCHEMA_VERSION,
        "source_step_csv": str(step_path.resolve()),
        "source_trial_csv": str(trial_path.resolve()),
        "ai_player_index": ai_player_index,
        "human_player_index": 1 - ai_player_index,
        "weights": {
            **EVENT_WEIGHTS,
            "delivery": 20.0,
        },
        "decision_count": len(decision_rows),
        "trial_count": len(summary_rows),
        "sparse_reward_mismatch_count": mismatch_count,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    return DenseRewardBuildResult(
        decision_count=len(decision_rows),
        trial_count=len(summary_rows),
        sparse_reward_mismatch_count=mismatch_count,
        output_paths={
            "decisions": decision_path,
            "trials": summary_path,
            "metadata": metadata_path,
        },
    )
