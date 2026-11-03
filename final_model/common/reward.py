"""Canonical native Overcooked reward shared by every observation mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


LEARNER_AGENT_INDEX = 1
DELIVERY_REWARD = 20.0
HSP_REWARD_SHAPING_HORIZON = 20_000_000
REWARD_SHAPING_SCHEDULE = "linear"
REWARD_SHAPING_INITIAL = 1.0
REWARD_SHAPING_FINAL = 0.0
OPTIMIZATION_FORMULA = (
    "sum(sparse_reward_by_agent) + reward_shaping_factor * shaped_reward_by_agent[1]"
)


@dataclass(frozen=True)
class NativeReward:
    sparse_team_reward: float
    learner_native_shaped_reward: float
    weighted_shaped_reward: float
    reward_shaping_factor: float
    optimization_reward: float
    deliveries: int


def hsp_reward_shaping_factor(
    completed_environment_steps: int,
    horizon_steps: int = HSP_REWARD_SHAPING_HORIZON,
) -> float:
    steps = int(completed_environment_steps)
    if steps < 0:
        raise ValueError("completed_environment_steps cannot be negative")
    horizon = int(horizon_steps)
    if horizon < 0:
        raise ValueError("horizon_steps cannot be negative")
    if horizon == 0:
        return REWARD_SHAPING_FINAL
    progress = min(steps / horizon, 1.0)
    return REWARD_SHAPING_INITIAL + progress * (
        REWARD_SHAPING_FINAL - REWARD_SHAPING_INITIAL
    )


def native_reward(
    info: Mapping[str, Sequence[float]], *, reward_shaping_factor: float = 1.0
) -> NativeReward:
    sparse_by_agent = tuple(float(value) for value in info["sparse_reward_by_agent"])
    shaped_by_agent = tuple(float(value) for value in info["shaped_reward_by_agent"])
    if len(sparse_by_agent) != 2 or len(shaped_by_agent) != 2:
        raise ValueError("native Overcooked reward must contain exactly two agents")
    sparse = float(sum(sparse_by_agent))
    shaped = shaped_by_agent[LEARNER_AGENT_INDEX]
    factor = float(reward_shaping_factor)
    if not 0.0 <= factor <= 1.0:
        raise ValueError("reward_shaping_factor must be between zero and one")
    weighted_shaped = factor * shaped
    deliveries = int(round(sparse / DELIVERY_REWARD))
    if abs(sparse - deliveries * DELIVERY_REWARD) > 1e-10:
        raise ValueError("random3 sparse reward is not a canonical 20-point delivery total")
    return NativeReward(sparse, shaped, weighted_shaped, factor, sparse + weighted_shaped, deliveries)


def reward_dict(info: Mapping[str, Sequence[float]]) -> dict[str, float | int]:
    return asdict(native_reward(info))


def native_reward_shaping_audit() -> dict[str, object]:
    source = (
        "lib/zsc_eval/zsceval/envs/overcooked/"
        "overcooked_ai_py/mdp/overcooked_mdp.py"
    )
    terms = [
        {
            "term": "ingredient placement",
            "source_key": "PLACEMENT_IN_POT_REW",
            "mdp_no_shaping_default": 0,
            "canonical_zsceval_training_value": 3,
            "final_model_value": 3,
            "trigger": "interacting agent places onion/tomato into an empty or matching non-full pot",
            "recipient": "the interacting agent",
            "active_in_random3": True,
            "source_file": source,
            "source_lines": [365, 374, 863, 874],
        },
        {
            "term": "useful dish pickup",
            "source_key": "DISH_PICKUP_REWARD",
            "mdp_no_shaping_default": 0,
            "canonical_zsceval_training_value": 3,
            "final_model_value": 3,
            "trigger": "interacting agent takes a dispenser dish while a nearly-ready pot needs a dish and no counter dish exists",
            "recipient": "the interacting agent",
            "active_in_random3": True,
            "source_file": source,
            "source_lines": [366, 375, 841],
        },
        {
            "term": "ready soup pickup",
            "source_key": "SOUP_PICKUP_REWARD",
            "mdp_no_shaping_default": 0,
            "canonical_zsceval_training_value": 5,
            "final_model_value": 5,
            "trigger": "interacting agent uses a dish to take a fully cooked soup from a pot",
            "recipient": "the interacting agent",
            "active_in_random3": True,
            "source_file": source,
            "source_lines": [367, 376, 853],
        },
        {
            "term": "distance toward pot",
            "source_key": "POT_DISTANCE_REW",
            "mdp_no_shaping_default": 0,
            "canonical_zsceval_training_value": 0,
            "final_model_value": 0,
            "trigger": "distance helper only; transition call is commented out",
            "recipient": "none in get_state_transition",
            "active_in_random3": False,
            "source_file": source,
            "source_lines": [369, 378, 779, 1410],
        },
        {
            "term": "distance toward dish dispenser",
            "source_key": "DISH_DISP_DISTANCE_REW",
            "mdp_no_shaping_default": 0,
            "canonical_zsceval_training_value": 0,
            "final_model_value": 0,
            "trigger": "distance helper only; transition call is commented out",
            "recipient": "none in get_state_transition",
            "active_in_random3": False,
            "source_file": source,
            "source_lines": [368, 377, 779, 1426],
        },
        {
            "term": "distance toward serving",
            "source_key": "SOUP_DISTANCE_REW",
            "mdp_no_shaping_default": 0,
            "canonical_zsceval_training_value": 0,
            "final_model_value": 0,
            "trigger": "distance helper only; transition call is commented out",
            "recipient": "none in get_state_transition",
            "active_in_random3": False,
            "source_file": source,
            "source_lines": [370, 379, 779, 1443],
        },
    ]
    return {
        "layout": "random3",
        "agent_order": {"0": "frozen partner", "1": "learner"},
        "sparse_delivery": {
            "value": DELIVERY_REWARD,
            "trigger": "agent holding soup interacts with serving terrain",
            "recipient": "delivering agent in sparse_reward_by_agent; summed for team training reward",
            "active_in_random3": True,
            "source_file": source,
            "source_lines": [395, 745, 879, 891, 905],
        },
        "native_shaping_terms": terms,
        "production_schedule": {
            "initial_factor": 1.0,
            "horizon_environment_steps": HSP_REWARD_SHAPING_HORIZON,
            "end_factor": 0.0,
            "schedule": "linear",
            "reason": "canonical Full Action 20M adaptation schedule",
        },
        "production_formula": OPTIMIZATION_FORMULA,
        "v3d_contribution_reward_used": False,
    }
