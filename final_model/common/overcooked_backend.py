"""Direct bundled OvercookedGridworld backend shared by Full and Partial."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch

from final_model.common.partners import FrozenPartnerPool, FrozenPartnerRepository
from final_model.common.paths import predictor_checkpoint
from final_model.common.reward import native_reward
from final_model.common.zsceval_compat import install_zsceval


CANONICAL_ENCODER_ID = (
    "zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp."
    "OvercookedGridworld.lossless_state_encoding:20ch_old_dynamics"
)
COOKING_DYNAMICS_ID = "experiment_instant_onion_ready_after_transition_v1"
ONION_COOK_TICKS_PER_ENVIRONMENT_STEP = 20
SOUP_COOKING_TIME = 20
SOUP_ITEMS_REQUIRED = 3


def cooking_contract() -> dict[str, object]:
    return {
        "cooking_dynamics_id": COOKING_DYNAMICS_ID,
        "onion_cook_ticks_per_environment_step": (
            ONION_COOK_TICKS_PER_ENVIRONMENT_STEP
        ),
        "soup_cooking_time": SOUP_COOKING_TIME,
        "soup_items_required": SOUP_ITEMS_REQUIRED,
    }


def configure_experiment_cooking_dynamics(mdp):
    """Make a full onion pot ready after the completing joint transition."""
    if (
        int(mdp.soup_cooking_time) != SOUP_COOKING_TIME
        or int(mdp.num_items_for_soup) != SOUP_ITEMS_REQUIRED
    ):
        raise RuntimeError("canonical experiment cooking recipe changed")
    original = mdp.step_environment_effects

    def step_environment_effects(state):
        result = original(state)
        for obj in state.objects.values():
            if getattr(obj, "name", None) != "soup":
                continue
            soup_type, num_items, cook_time = obj.state
            x, y = obj.position
            if (
                soup_type == "onion"
                and num_items == mdp.num_items_for_soup
                and mdp.terrain_mtx[y][x] == "P"
                and cook_time < mdp.soup_cooking_time
            ):
                obj.state = soup_type, num_items, mdp.soup_cooking_time
        return result

    mdp.step_environment_effects = step_environment_effects
    return mdp


def _imports():
    install_zsceval()
    from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action
    from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
        BASE_REW_SHAPING_PARAMS,
        OvercookedGridworld,
    )
    from zsceval.utils import v3a2_displayed_prediction as displayed

    return Action, OvercookedGridworld, BASE_REW_SHAPING_PARAMS, displayed


class DirectOvercookedBackend:
    """Agent 0 is a frozen partner and agent 1 is the learner."""

    def __init__(
        self,
        *,
        seed: int,
        horizon: int,
        episode_length: int,
        device: str | torch.device,
        partner: str | None = None,
        partner_repository: FrozenPartnerRepository | None = None,
        prediction_oracle=None,
        prediction_device: str | torch.device = "cpu",
        environment_index: int | None = None,
        enable_predictor: bool = False,
    ) -> None:
        if horizon not in (0, 2, 4):
            raise ValueError("horizon must be 0, 2, or 4")
        Action, Gridworld, shaping, displayed = _imports()
        self.Action = Action
        self._displayed = displayed
        self.mdp = configure_experiment_cooking_dynamics(
            Gridworld.from_layout_name(
                "random3", rew_shaping_params=copy.deepcopy(shaping)
            )
        )
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.episode_length = int(episode_length)
        self.episode_step = 0
        self.environment_steps = 0
        self.environment_index = environment_index
        self.state_generation = -1
        self.reward_shaping_factor = 1.0
        self.partners = FrozenPartnerPool(
            seed=seed,
            device=device,
            requested=partner,
            repository=partner_repository,
        )
        self.partner_step_guard_checks = 0
        self.partner_step_guard_failures = 0
        self.partner_full_hash_checks = 0
        self.partner_full_hash_failures = 0
        self.last_partner_full_hash_baseline = self.partners.frozen_parameter_hash
        self.last_partner_full_hash_actual = self.partners.frozen_parameter_hash
        self.fixed_partner_identity = (
            str(self.partners.manifest[0]["name"])
            if len(self.partners.manifest) == 1
            else None
        )
        self._canonical_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._preview_cache: np.ndarray | None = None
        self._preview_cache_generation: int | None = None
        self.predictor_inference_seconds = 0.0
        self.predictor_forward_calls = 0
        if self.horizon or enable_predictor:
            if prediction_oracle is None:
                path = predictor_checkpoint()
                # The preserved oracle has a legacy module constant; replace it with the
                # independently hash-verified repository-relative asset before construction.
                displayed.ORIGINAL_PREDICTOR_CHECKPOINT = path
                prediction_oracle = displayed.CounterCircuitDisplayedPredictionOracle(
                    path, device=prediction_device
                )
            self.oracle = prediction_oracle
        else:
            self.oracle = None
        self.reset(initial=True)

    def set_horizon(self, horizon: int) -> None:
        value = int(horizon)
        if value not in (0, 2, 4):
            raise ValueError("horizon must be 0, 2, or 4")
        if value != self.horizon and self.episode_step != 0:
            raise RuntimeError("horizon may only change at an episode boundary")
        if value and self.oracle is None:
            raise RuntimeError("cannot activate a predicted horizon without a predictor")
        if value != self.horizon:
            self.horizon = value
            self._invalidate_observation_cache()

    @property
    def active_partner(self) -> str:
        return self.partners.name

    @property
    def partner_manifest(self):
        return tuple(self.partners.manifest)

    def reset(self, *, initial: bool = False) -> None:
        if not initial:
            self.partners.load_next()
        self.state = self.mdp.get_standard_start_state()
        self.episode_step = 0
        self.state_generation += 1
        self.self_history, self.partner_history = self._displayed.initialize_position_histories(
            self.state
        )
        self._invalidate_observation_cache()

    def _invalidate_observation_cache(self) -> None:
        self._canonical_cache = None
        self._preview_cache = None
        self._preview_cache_generation = None

    def canonical_pair(self) -> tuple[np.ndarray, np.ndarray]:
        if self._canonical_cache is not None:
            return self._canonical_cache
        pair = tuple(
            np.ascontiguousarray(value * 255.0, dtype=np.float32)
            for value in self.mdp.lossless_state_encoding(self.state)
        )
        if tuple(value.shape for value in pair) != ((8, 5, 20), (8, 5, 20)):
            raise RuntimeError("canonical encoder contract changed")
        self._canonical_cache = pair
        return self._canonical_cache

    def preview_planes(self) -> np.ndarray:
        if (
            self._preview_cache is not None
            and self._preview_cache_generation != self.state_generation
        ):
            raise RuntimeError("stale predictor cache generation")
        if self._preview_cache is None:
            if self.oracle is None:
                self._preview_cache = np.zeros((4, 8, 5), dtype=np.float32)
            else:
                if (
                    self.fixed_partner_identity is not None
                    and self.active_partner != self.fixed_partner_identity
                ):
                    raise RuntimeError("displayed preview subject is not the fixed partner")
                started = time.perf_counter()
                prediction = self.oracle.predict(
                    self.state,
                    self.mdp,
                    self.self_history,
                    self.partner_history,
                    self.horizon,
                )
                self.predictor_inference_seconds += time.perf_counter() - started
                self.predictor_forward_calls += 1
                self._preview_cache = np.ascontiguousarray(
                    prediction.displayed_prediction_gt, dtype=np.float32
                )
            self._preview_cache_generation = self.state_generation
        return self._preview_cache

    def consume_predictor_profile(self) -> tuple[float, int]:
        seconds = float(self.predictor_inference_seconds)
        forwards = int(self.predictor_forward_calls)
        self.predictor_inference_seconds = 0.0
        self.predictor_forward_calls = 0
        return seconds, forwards

    def set_batched_preview(self, planes: np.ndarray, *, generation: int) -> None:
        if int(generation) != self.state_generation:
            raise RuntimeError("batched predictor result belongs to a stale environment state")
        value = np.ascontiguousarray(planes, dtype=np.float32)
        if value.shape != (4, 8, 5):
            raise RuntimeError("batched predictor result must have shape (4,8,5)")
        self._preview_cache = value.copy()
        self._preview_cache_generation = self.state_generation

    def predictor_partner_contract(self) -> dict[str, object]:
        return {
            "actual_partner_agent_index": 0,
            "predictor_target_agent_index": 0,
            "predictor_history_source": "actual_agent0_partner_position_history",
            "displayed_preview_partner_identity": self.active_partner,
            "fixed_partner_identity": self.fixed_partner_identity,
            "learner_agent_index": 1,
            "predictor_targets_learner": False,
            "predictor_partner_alignment_verified": (
                self.fixed_partner_identity is not None
                and self.active_partner == self.fixed_partner_identity
            ),
        }

    def self_tile(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.state.players[1].position)

    def _available(self, agent_index: int) -> np.ndarray:
        return np.ascontiguousarray(
            self._displayed._original_available_actions(self.mdp, self.state, agent_index),
            dtype=np.float32,
        )

    def canonical_learner_input(self) -> tuple[np.ndarray, np.ndarray]:
        """Canonical HSP learner input with the original agent-1 legal-action mask."""
        observation = self.canonical_pair()[1]
        available = self._available(1)
        if observation.shape != (8, 5, 20) or available.shape != (6,):
            raise RuntimeError("canonical HSP learner input contract changed")
        return observation, available

    def learner_available_actions(self) -> np.ndarray:
        return self._available(1)

    def set_reward_shaping_factor(self, factor: float) -> None:
        value = float(factor)
        if not 0.0 <= value <= 1.0:
            raise ValueError("reward_shaping_factor must be between zero and one")
        self.reward_shaping_factor = value

    def partner_batch_input(self) -> tuple[np.ndarray, np.ndarray]:
        return self.canonical_pair()[0], self._available(0)

    def begin_partner_step(self) -> str:
        try:
            return self.partners.assert_frozen()
        except RuntimeError:
            self.partner_step_guard_failures += 1
            raise
        finally:
            self.partner_step_guard_checks += 1

    def step(self, learner_action: int) -> dict[str, Any]:
        if not 0 <= int(learner_action) < 6:
            raise ValueError("learner action must be one of all six primitive actions")
        partner_obs, partner_available = self.partner_batch_input()
        before = self.begin_partner_step()
        partner_action = self.partners.action(partner_obs, partner_available)
        return self.step_with_partner_action(
            learner_action, partner_action, partner_hash_before=before
        )

    def step_with_partner_action(
        self,
        learner_action: int,
        partner_action: int,
        *,
        partner_hash_before: str,
    ) -> dict[str, Any]:
        if not 0 <= int(learner_action) < 6 or not 0 <= int(partner_action) < 6:
            raise ValueError("joint action must contain two primitive actions")
        pre_state = self.state
        self.state, native_info = self.mdp.get_state_transition(
            pre_state,
            (
                self.Action.INDEX_TO_ACTION[partner_action],
                self.Action.INDEX_TO_ACTION[int(learner_action)],
            ),
        )
        try:
            after = self.partners.assert_frozen()
        except RuntimeError:
            self.partner_step_guard_failures += 1
            raise
        finally:
            self.partner_step_guard_checks += 1
        if partner_hash_before != after:
            self.partner_step_guard_failures += 1
            raise RuntimeError("frozen partner baseline changed within env.step")
        self.self_history, self.partner_history = self._displayed.advance_position_histories(
            self.self_history, self.partner_history, self.state
        )
        self.episode_step += 1
        self.environment_steps += 1
        self.state_generation += 1
        self._invalidate_observation_cache()
        reward = native_reward(
            native_info, reward_shaping_factor=self.reward_shaping_factor
        )
        return {
            "reward": reward.optimization_reward,
            "optimization_reward": reward.optimization_reward,
            "sparse_team_reward": reward.sparse_team_reward,
            "native_shaped_reward": reward.learner_native_shaped_reward,
            "weighted_shaped_reward": reward.weighted_shaped_reward,
            "reward_shaping_factor": reward.reward_shaping_factor,
            "deliveries": reward.deliveries,
            "native_info": native_info,
            "partner": self.active_partner,
            "done": self.episode_step >= self.episode_length,
        }

    def verify_partner_parameters(self) -> str:
        """Run the full SHA-256 audit at a rollout boundary."""
        self.last_partner_full_hash_baseline = self.partners.frozen_parameter_hash
        try:
            actual = self.partners.assert_frozen(full_hash=True)
        except RuntimeError:
            self.partner_full_hash_failures += 1
            raise
        finally:
            self.partner_full_hash_checks += 1
        self.last_partner_full_hash_actual = actual
        return actual

    def training_state(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "horizon": self.horizon,
            "episode_length": self.episode_length,
            "state": copy.deepcopy(self.state),
            "episode_step": self.episode_step,
            "environment_steps": self.environment_steps,
            "state_generation": self.state_generation,
            "reward_shaping_factor": self.reward_shaping_factor,
            "self_history": copy.deepcopy(self.self_history),
            "partner_history": copy.deepcopy(self.partner_history),
            "partners": self.partners.training_state(),
            "partner_step_guard_checks": self.partner_step_guard_checks,
            "partner_step_guard_failures": self.partner_step_guard_failures,
            "partner_full_hash_checks": self.partner_full_hash_checks,
            "partner_full_hash_failures": self.partner_full_hash_failures,
            "last_partner_full_hash_baseline": self.last_partner_full_hash_baseline,
            "last_partner_full_hash_actual": self.last_partner_full_hash_actual,
            **cooking_contract(),
        }

    def load_training_state(self, state: dict[str, Any]) -> None:
        expected = {
            "seed": self.seed,
            "horizon": self.horizon,
            "episode_length": self.episode_length,
            **cooking_contract(),
        }
        mismatch = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"backend resume contract mismatch: {mismatch}")
        self.state = copy.deepcopy(state["state"])
        self.episode_step = int(state["episode_step"])
        self.environment_steps = int(state["environment_steps"])
        self.state_generation = int(
            state.get("state_generation", self.environment_steps)
        )
        self.reward_shaping_factor = float(state["reward_shaping_factor"])
        self.self_history = copy.deepcopy(state["self_history"])
        self.partner_history = copy.deepcopy(state["partner_history"])
        self.partners.load_training_state(state["partners"])
        for key in (
            "partner_step_guard_checks", "partner_step_guard_failures",
            "partner_full_hash_checks", "partner_full_hash_failures",
        ):
            setattr(self, key, int(state[key]))
        self.last_partner_full_hash_baseline = str(state["last_partner_full_hash_baseline"])
        self.last_partner_full_hash_actual = str(state["last_partner_full_hash_actual"])
        self._invalidate_observation_cache()


def backend_contract_report() -> dict[str, object]:
    return {
        "backend": "DirectOvercookedBackend",
        "layout": "random3",
        "direct_imports": [
            "zsceval.envs.overcooked.overcooked_ai_py.mdp.actions.Action",
            "zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp.OvercookedGridworld",
        ],
        "agent_order": {"0": "frozen partner", "1": "learner"},
        "cooking": cooking_contract(),
        "preview_state_dynamics": {
            "environment_state_rollout": False,
            "state_input": "same_backend_current_post_transition_state",
            "h1_h4_semantics": "predicted_action_position_and_interact_tiles_only",
        },
        "frozen_partner_verification": {
            "per_env_step": "metadata-only tensor guard before and after inference",
            "rollout_boundary": "full canonical SHA-256",
            "step_guard_fields": [
                "name", "object_id", "data_ptr", "version", "shape", "dtype", "device",
            ],
            "optimizer_isolation": "verified once at session startup",
        },
        "v3e_runtime_dependency": False,
        "v3d_reward_dependency": False,
        "native_reward_parameters": {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        },
    }
