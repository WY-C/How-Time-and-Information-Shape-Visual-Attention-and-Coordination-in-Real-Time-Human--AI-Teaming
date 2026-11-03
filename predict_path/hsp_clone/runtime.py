import copy
import glob
import os
import pickle
import re
import sys
import types
import pathlib

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_OVERCOOKED_PATH = os.path.join(REPO_ROOT, "lib", "overcooked_ai")
LOCAL_ZSCEVAL_PATH = os.path.join(REPO_ROOT, "lib", "zsc_eval")
LOCAL_POLICY_POOL_PATH = os.path.join(REPO_ROOT, "policy_pool")
LOCAL_ZSCEVAL_POLICY_POOL_PATH = os.path.join(LOCAL_ZSCEVAL_PATH, "zsceval", "policy_pool")
POLICY_POOL_ROOTS = [LOCAL_POLICY_POOL_PATH, LOCAL_ZSCEVAL_POLICY_POOL_PATH]

os.environ.setdefault("POLICY_POOL", LOCAL_POLICY_POOL_PATH)


def _install_gym_stub():
    try:
        import gym  # noqa: F401
        return
    except ImportError:
        pass

    class Env:
        pass

    class Box:
        def __init__(self, low=None, high=None, shape=None, dtype=None):
            self.low = low
            self.high = high
            self._shape = shape
            self.dtype = dtype

        @property
        def shape(self):
            return getattr(self, "_shape", None)

    class Discrete:
        def __init__(self, n=None):
            self.n = n

        def contains(self, value):
            return 0 <= int(value) < self.n

        def sample(self):
            return np.random.randint(self.n)

    class MultiDiscrete:
        def __init__(self, shape=None):
            self._shape = shape

        @property
        def shape(self):
            return getattr(self, "_shape", None)

    class MultiBinary:
        def __init__(self, shape=None):
            self._shape = shape

        @property
        def shape(self):
            return getattr(self, "_shape", None)

    gym_module = types.ModuleType("gym")
    spaces_module = types.ModuleType("gym.spaces")
    spaces_box_module = types.ModuleType("gym.spaces.box")
    spaces_discrete_module = types.ModuleType("gym.spaces.discrete")
    spaces_multi_discrete_module = types.ModuleType("gym.spaces.multi_discrete")
    spaces_multi_binary_module = types.ModuleType("gym.spaces.multi_binary")
    envs_module = types.ModuleType("gym.envs")
    registration_module = types.ModuleType("gym.envs.registration")

    registration_module.register = lambda *args, **kwargs: None
    envs_module.registration = registration_module

    spaces_module.Box = Box
    spaces_module.Discrete = Discrete
    spaces_module.MultiDiscrete = MultiDiscrete
    spaces_module.MultiBinary = MultiBinary
    spaces_box_module.Box = Box
    spaces_discrete_module.Discrete = Discrete
    spaces_multi_discrete_module.MultiDiscrete = MultiDiscrete
    spaces_multi_binary_module.MultiBinary = MultiBinary

    gym_module.Env = Env
    gym_module.spaces = spaces_module
    gym_module.envs = envs_module

    sys.modules["gym"] = gym_module
    sys.modules["gym.spaces"] = spaces_module
    sys.modules["gym.spaces.box"] = spaces_box_module
    sys.modules["gym.spaces.discrete"] = spaces_discrete_module
    sys.modules["gym.spaces.multi_discrete"] = spaces_multi_discrete_module
    sys.modules["gym.spaces.multi_binary"] = spaces_multi_binary_module
    sys.modules["gym.envs"] = envs_module
    sys.modules["gym.envs.registration"] = registration_module


def _install_absl_stub():
    try:
        from absl import flags  # noqa: F401
        return
    except ImportError:
        pass

    absl_module = types.ModuleType("absl")
    flags_module = types.ModuleType("absl.flags")

    class _DummyFlags:
        def __call__(self, argv):
            return argv

    flags_module.FLAGS = _DummyFlags()
    absl_module.flags = flags_module

    sys.modules["absl"] = absl_module
    sys.modules["absl.flags"] = flags_module


def _install_loguru_stub():
    try:
        import loguru  # noqa: F401
        return
    except ImportError:
        pass

    loguru_module = types.ModuleType("loguru")

    class _DummyLogger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    loguru_module.logger = _DummyLogger()
    sys.modules["loguru"] = loguru_module


def _install_wandb_stub():
    try:
        import wandb  # noqa: F401
        return
    except ImportError:
        pass

    wandb_module = types.ModuleType("wandb")
    wandb_module.run = types.SimpleNamespace(dir="/tmp")
    wandb_module.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_module


def _install_tensorboardx_stub():
    try:
        import tensorboardX  # noqa: F401
        return
    except ImportError:
        pass

    tensorboardx_module = types.ModuleType("tensorboardX")

    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def close(self):
            pass

    tensorboardx_module.SummaryWriter = SummaryWriter
    sys.modules["tensorboardX"] = tensorboardx_module


def install_runtime_stubs():
    _install_gym_stub()
    _install_absl_stub()
    _install_loguru_stub()
    _install_wandb_stub()
    _install_tensorboardx_stub()


install_runtime_stubs()


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map PosixPath (created on POSIX systems) to PurePosixPath on Windows
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        return super().find_class(module, name)

if LOCAL_OVERCOOKED_PATH not in sys.path:
    sys.path.insert(0, LOCAL_OVERCOOKED_PATH)
if LOCAL_ZSCEVAL_PATH not in sys.path:
    sys.path.insert(0, LOCAL_ZSCEVAL_PATH)

from overcooked_ai_py.agents.agent import GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as BaseOvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Direction, OvercookedGridworld as BaseOvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.utils import load_dict_from_file
from zsceval.algorithms.population.utils import EvalPolicy
from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from zsceval.overcooked_config import OLD_LAYOUTS
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as TomatoOvercookedEnv
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld as TomatoOvercookedGridworld,
    OvercookedState as TomatoOvercookedState,
)


ACTION_DIM = len(Action.ALL_ACTIONS)
CANONICAL_TOMATO_ORDERS = (
    {"ingredients": ["onion", "onion", "onion"]},
    {"ingredients": ["tomato", "tomato", "tomato"]},
    {"ingredients": ["onion", "onion", "tomato"]},
    {"ingredients": ["onion", "tomato", "tomato"]},
)
BASE_LAYOUTS_DIR = os.path.join(LOCAL_OVERCOOKED_PATH, "overcooked_ai_py", "data", "layouts")
TOMATO_LAYOUTS_DIR = os.path.join(
    LOCAL_ZSCEVAL_PATH,
    "zsceval",
    "envs",
    "overcooked_new",
    "src",
    "overcooked_ai_py",
    "data",
    "layouts",
)
DEFAULT_POLICY_POOL_PARTNER_AGENTS = ("HSP", "MEP", "GREEDY")
DEFAULT_HSP_S2_SUBPATH = os.path.join("hsp", "s2")
DEFAULT_HSP_S2_RUN_NAME = "hsp-S2-s36"
DEFAULT_MEP_S2_SUBPATH = os.path.join("mep", "s2")
DEFAULT_MEP_S2_RUN_NAME = "mep-S2-s36"
POLICY_POOL_LAYOUT_NAME_MAP = {
    "cramped_room": "simple",
    "coordination_ring": "coordination_rings",
}
ZSCEVAL_LAYOUT_NAME_MAP = {
    "forced_coordination": "random0",
    "bothway_coordination": "random0_medium",
    "coordination_ring": "random1",
    "counter_circuit": "random3",
    "asymmetric_advantages": "unident_s",
}
DEFAULT_HSP_S2_LAYOUT_RECOMMENDATIONS = {
    "random0": {
        "default": ("hsp-S2-s24", 5),
        "hsp-S2-s12": 4,
        "hsp-S2-s24": 5,
        "hsp-S2-s36": 1,
    },
    "random1": {
        "default": ("hsp-S2-s36", 2),
        "hsp-S2-s12": 2,
        "hsp-S2-s24": 3,
        "hsp-S2-s36": 2,
    },
    "random3": {
        "default": ("hsp-S2-s36", 2),
        "hsp-S2-s12": 2,
        "hsp-S2-s24": 3,
        "hsp-S2-s36": 2,
    },
    "unident_s": {
        "default": ("hsp-S2-s36", 3),
        "hsp-S2-s12": 4,
        "hsp-S2-s24": 5,
        "hsp-S2-s36": 3,
    },
    "unident_s_hard": {
        "default": ("hsp-S2-s36", 1),
        "hsp-S2-s36": 1,
    },
    "unident_s_hard_service": {
        "default": ("hsp-S2-s36", 1),
        "hsp-S2-s36": 1,
    },
    "unident_s_hard_onion": {
        "default": ("hsp-S2-s36", 1),
        "hsp-S2-s36": 1,
    },
}
def parse_name_list(raw_value):
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def parse_int_list(raw_value):
    return [int(value.strip()) for value in raw_value.split(",") if value.strip()]


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def get_zsceval_layout_candidates(layout_name):
    candidates = [layout_name]
    mapped_layout = ZSCEVAL_LAYOUT_NAME_MAP.get(layout_name)
    if mapped_layout and mapped_layout not in candidates:
        candidates.append(mapped_layout)
    return candidates


def get_policy_pool_layout_candidates(layout_name):
    candidates = [layout_name]
    policy_pool_layout = POLICY_POOL_LAYOUT_NAME_MAP.get(layout_name)
    if policy_pool_layout and policy_pool_layout not in candidates:
        candidates.append(policy_pool_layout)
    zsceval_layout = ZSCEVAL_LAYOUT_NAME_MAP.get(layout_name)
    if zsceval_layout and zsceval_layout not in candidates:
        candidates.append(zsceval_layout)
    return candidates


def get_recommended_zsceval_layout_name(layout_name):
    for candidate in get_zsceval_layout_candidates(layout_name):
        if candidate in DEFAULT_HSP_S2_LAYOUT_RECOMMENDATIONS:
            return candidate
    return get_zsceval_layout_candidates(layout_name)[-1]


def get_layout_candidate_paths(layout_name, subpath):
    candidate_paths = []
    for candidate_layout in get_policy_pool_layout_candidates(layout_name):
        for policy_pool_root in POLICY_POOL_ROOTS:
            candidate_paths.append(os.path.join(policy_pool_root, candidate_layout, subpath))
    return candidate_paths


def find_first_existing_path(candidate_paths):
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return os.path.abspath(candidate_path)
    return None


def normalize_hsp_s2_run_name(run_name):
    if str(run_name).isdigit():
        return f"hsp-S2-s{run_name}"
    return run_name


def get_default_hsp_s2_choice(layout_name, run_name=None):
    recommended_layout = get_recommended_zsceval_layout_name(layout_name)
    layout_recommendation = DEFAULT_HSP_S2_LAYOUT_RECOMMENDATIONS.get(recommended_layout, {})
    if run_name is None:
        return layout_recommendation.get("default", (DEFAULT_HSP_S2_RUN_NAME, 1))
    normalized_run_name = normalize_hsp_s2_run_name(run_name)
    return normalized_run_name, layout_recommendation.get(normalized_run_name, 1)


def get_default_policy_config_rel_path(layout_name, recurrent=False):
    config_filename = "rnn_policy_config.pkl" if recurrent else "mlp_policy_config.pkl"
    candidate_layouts = get_policy_pool_layout_candidates(layout_name)
    recommended_layout = get_recommended_zsceval_layout_name(layout_name)
    if recommended_layout not in candidate_layouts:
        candidate_layouts.append(recommended_layout)

    tried_paths = []
    for candidate_layout in candidate_layouts:
        policy_config_rel_path = os.path.join(candidate_layout, "policy_config", config_filename)
        tried_paths.append(policy_config_rel_path)
        for policy_pool_root in POLICY_POOL_ROOTS:
            policy_config_abs_path = os.path.join(policy_pool_root, policy_config_rel_path)
            if os.path.exists(policy_config_abs_path):
                return policy_config_rel_path

    raise FileNotFoundError(
        f"Policy config not found for layout '{layout_name}'. Tried: {tried_paths}"
    )


def resolve_hsp_s2_checkpoint(layout_name, run_name=DEFAULT_HSP_S2_RUN_NAME, model_index=1, model_path=None):
    if model_path:
        checkpoint_path = os.path.abspath(model_path)
    else:
        normalized_run_name = normalize_hsp_s2_run_name(run_name)
        candidate_paths = get_layout_candidate_paths(
            layout_name,
            os.path.join(DEFAULT_HSP_S2_SUBPATH, normalized_run_name, f"{model_index}.pt"),
        )
        checkpoint_path = find_first_existing_path(candidate_paths)

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"HSP S2 checkpoint not found for layout '{layout_name}'.")
    return checkpoint_path


def resolve_population_s2_checkpoint(layout_name, population_name, subpath, run_name, model_index=1, model_path=None):
    if model_path:
        checkpoint_path = os.path.abspath(model_path)
    else:
        candidate_paths = get_layout_candidate_paths(
            layout_name,
            os.path.join(subpath, run_name, f"{model_index}.pt"),
        )
        checkpoint_path = find_first_existing_path(candidate_paths)

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{population_name} checkpoint not found for layout '{layout_name}' "
            f"(run={run_name}, index={model_index})."
        )
    return checkpoint_path


def resolve_zsceval_policy_config_path(policy_config_rel_path):
    tried_paths = []
    for policy_pool_root in POLICY_POOL_ROOTS:
        config_path = os.path.abspath(os.path.join(policy_pool_root, policy_config_rel_path))
        tried_paths.append(config_path)
        if os.path.exists(config_path):
            return config_path
    raise FileNotFoundError(f"Policy config not found. Tried: {tried_paths}")


def get_obs_space_shape(obs_space):
    return tuple(getattr(obs_space, "shape", getattr(obs_space, "_shape", ())))


def resolve_torch_device(device):
    if isinstance(device, torch.device):
        resolved_device = device
    else:
        resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{resolved_device}' was requested, but CUDA is not available.")
    return resolved_device


def identify_policy_pool_root(path):
    abs_path = os.path.abspath(path)
    for policy_pool_root in POLICY_POOL_ROOTS:
        abs_root = os.path.abspath(policy_pool_root)
        try:
            if os.path.commonpath([abs_path, abs_root]) == abs_root:
                return abs_root
        except ValueError:
            continue
    return None


def natural_sort_key(path):
    parts = re.split(r"(\d+)", os.path.basename(str(path)))
    return [int(part) if part.isdigit() else part for part in parts]


def glob_policy_pool_paths(layout_name, relative_pattern):
    matched_paths = []
    seen_paths = set()
    for candidate_layout in get_policy_pool_layout_candidates(layout_name):
        for policy_pool_root in POLICY_POOL_ROOTS:
            pattern = os.path.join(policy_pool_root, candidate_layout, relative_pattern)
            for matched_path in sorted(glob.glob(pattern), key=natural_sort_key):
                abs_path = os.path.abspath(matched_path)
                if abs_path in seen_paths:
                    continue
                seen_paths.add(abs_path)
                matched_paths.append(abs_path)
    return matched_paths


def dedupe_variant_specs(variant_specs):
    deduped_specs = []
    seen_variant_names = set()
    for variant_spec in variant_specs:
        variant_name = variant_spec["variant_name"]
        if variant_name in seen_variant_names:
            continue
        seen_variant_names.add(variant_name)
        deduped_specs.append(variant_spec)
    return deduped_specs


def policy_pool_variant_name(layout_name, checkpoint_path):
    abs_path = os.path.abspath(checkpoint_path)
    for candidate_layout in get_policy_pool_layout_candidates(layout_name):
        for policy_pool_root in POLICY_POOL_ROOTS:
            layout_root = os.path.abspath(os.path.join(policy_pool_root, candidate_layout))
            try:
                if os.path.commonpath([abs_path, layout_root]) == layout_root:
                    relative_path = os.path.relpath(abs_path, layout_root)
                    stem = os.path.splitext(relative_path)[0]
                    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
            except ValueError:
                continue
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(abs_path))[0]).strip("_")


def checkpoint_variant_specs(layout_name, relative_patterns, algorithm_name_override=None, recurrent=None):
    variant_specs = []
    for relative_pattern in relative_patterns:
        checkpoint_paths = glob_policy_pool_paths(layout_name, relative_pattern)
        for checkpoint_path in checkpoint_paths:
            variant_specs.append(
                {
                    "variant_name": policy_pool_variant_name(layout_name, checkpoint_path),
                    "checkpoint_path": checkpoint_path,
                    "algorithm_name_override": algorithm_name_override,
                    "recurrent": recurrent,
                }
            )
    return dedupe_variant_specs(variant_specs)


def build_zsceval_policy(policy_args, obs_space, share_obs_space, act_space, device):
    resolved_device = resolve_torch_device(device)
    algorithm_name = policy_args.algorithm_name
    if algorithm_name in {"mappo", "rmappo"}:
        return R_MAPPOPolicy(policy_args, obs_space, share_obs_space, act_space, device=resolved_device)
    raise NotImplementedError(f"Unsupported zsceval policy algorithm: {algorithm_name}")


def convert_obs_to_hsp_channels(obs):
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.shape[-1] == 20:
        return obs_arr
    if obs_arr.shape[-1] != 26:
        raise ValueError(f"Unexpected observation shape for HSP: {obs_arr.shape}")

    obs_20 = np.zeros((*obs_arr.shape[:-1], 20), dtype=np.float32)
    obs_20[..., 0:10] = obs_arr[..., 0:10]
    obs_20[..., 10] = obs_arr[..., 10]
    obs_20[..., 11] = obs_arr[..., 11]
    obs_20[..., 12] = obs_arr[..., 12]
    obs_20[..., 13] = obs_arr[..., 14]
    obs_20[..., 14] = obs_arr[..., 15]
    obs_20[..., 15] = obs_arr[..., 16] + obs_arr[..., 18]
    obs_20[..., 16] = obs_arr[..., 20]
    obs_20[..., 17] = obs_arr[..., 21]
    obs_20[..., 18] = obs_arr[..., 22]
    obs_20[..., 19] = obs_arr[..., 23]
    return obs_20


def get_available_actions_for_state(mdp, state):
    num_agents = len(state.players)
    available_actions = np.ones((num_agents, ACTION_DIM), dtype=np.uint8)
    interact_index = Action.ACTION_TO_INDEX[Action.INTERACT]

    for agent_idx in range(num_agents):
        player = state.players[agent_idx]
        pos = player.position
        orientation = player.orientation

        for move_i, move in enumerate(Direction.ALL_DIRECTIONS):
            new_pos = Action.move_in_direction(pos, move)
            if new_pos not in mdp.get_valid_player_positions() and orientation == move:
                available_actions[agent_idx, move_i] = 0

        interact_pos = Action.move_in_direction(pos, orientation)
        terrain_type = mdp.get_terrain_type_at_pos(interact_pos)

        if (
            terrain_type == " "
            or (
                terrain_type == "X"
                and (
                    (not player.has_object() and not state.has_object(interact_pos))
                    or (player.has_object() and state.has_object(interact_pos))
                )
            )
            or (terrain_type in ["O", "T", "D"] and player.has_object())
            or (
                terrain_type == "P"
                and (not player.has_object() or player.get_object().name not in ["dish", "onion", "tomato"])
            )
            or (terrain_type == "S" and (not player.has_object() or player.get_object().name not in ["soup"]))
        ):
            available_actions[agent_idx, interact_index] = 0

    return available_actions


def featurize_state_for_agent(
    mdp,
    state,
    agent_index,
    expected_obs_shape=None,
    observation_mdp=None,
):
    encoding_mdp = observation_mdp or mdp
    obs = encoding_mdp.lossless_state_encoding(state)[agent_index]
    if expected_obs_shape and tuple(obs.shape) != expected_obs_shape:
        if obs.shape[-1] == 26 and expected_obs_shape[-1] == 20:
            obs = convert_obs_to_hsp_channels(obs)
        else:
            raise ValueError(
                f"Observation shape mismatch. expected={expected_obs_shape}, got={tuple(obs.shape)}"
            )
    available_actions = get_available_actions_for_state(mdp, state)[agent_index]
    return obs.astype(np.float32) * 255.0, available_actions


def normalize_action(action):
    if isinstance(action, np.ndarray):
        return normalize_action(action.reshape(-1)[0].item())
    if isinstance(action, (np.integer, int)):
        return Action.INDEX_TO_ACTION[int(action)]
    if isinstance(action, tuple) and len(action) == 2 and isinstance(action[1], dict):
        return normalize_action(action[0])
    return action


def action_to_index(action):
    return Action.ACTION_TO_INDEX[normalize_action(action)]


def action_to_string(action):
    normalized = normalize_action(action)
    if normalized == Action.INTERACT:
        return "INTERACT"
    return str(tuple(normalized))


def get_grid_shape(mdp):
    return len(mdp.terrain_mtx[0]), len(mdp.terrain_mtx)


def position_to_cell(position, grid_width):
    x, y = position
    return int(y) * int(grid_width) + int(x)


def cell_to_position(cell_index, grid_width):
    cell_index = int(cell_index)
    grid_width = int(grid_width)
    return cell_index % grid_width, cell_index // grid_width


def checkpoint_uses_recurrent_policy(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    return any(key.startswith("rnn.") for key in state_dict.keys())


def recipe_order_key(order):
    return tuple(sorted(order.get("ingredients", [])))


def get_layout_dict_path(layout_name):
    candidate_paths = [
        os.path.join(BASE_LAYOUTS_DIR, f"{layout_name}.layout"),
        os.path.join(TOMATO_LAYOUTS_DIR, f"{layout_name}.layout"),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError(
        f"Layout '{layout_name}' not found. Tried: {candidate_paths}"
    )


def layout_uses_tomatoes(layout_name):
    layout_dict = load_dict_from_file(get_layout_dict_path(layout_name))
    return "T" in layout_dict.get("grid", "")


def layout_uses_new_dynamics(layout_name):
    return layout_name not in OLD_LAYOUTS


def normalize_tomato_layout_params(layout_params):
    normalized_params = copy.deepcopy(layout_params)
    normalized_params["old_dynamics"] = True
    normalized_params["max_num_items_for_soup"] = 3
    normalized_params["start_all_orders"] = copy.deepcopy(list(CANONICAL_TOMATO_ORDERS))

    canonical_order_keys = [recipe_order_key(order) for order in CANONICAL_TOMATO_ORDERS]
    normalized_params["start_bonus_orders"] = [
        order
        for order in normalized_params.get("start_bonus_orders", [])
        if recipe_order_key(order) in canonical_order_keys
    ]

    original_orders = layout_params.get("start_all_orders", [])

    def remap_recipe_setting(setting_name):
        if setting_name not in layout_params:
            normalized_params.pop(setting_name, None)
            return

        original_values = layout_params[setting_name]
        value_by_order = {
            recipe_order_key(order): value
            for order, value in zip(original_orders, original_values)
        }
        fallback_value = next(
            (
                value
                for order, value in zip(original_orders, original_values)
                if len(order.get("ingredients", [])) == 3
            ),
            original_values[0] if original_values else None,
        )
        remapped_values = [
            value_by_order.get(order_key, fallback_value)
            for order_key in canonical_order_keys
        ]
        if any(value is None for value in remapped_values):
            normalized_params.pop(setting_name, None)
            return
        normalized_params[setting_name] = remapped_values

    remap_recipe_setting("recipe_values")
    remap_recipe_setting("recipe_times")
    return normalized_params


def build_tomato_layout_mdp(layout_name):
    layout_params = normalize_tomato_layout_params(
        load_dict_from_file(get_layout_dict_path(layout_name))
    )
    grid = [layout_row.strip() for layout_row in layout_params["grid"].split("\n")]
    del layout_params["grid"]
    layout_params["layout_name"] = layout_name
    if "start_state" in layout_params:
        layout_params["start_state"] = TomatoOvercookedState.from_dict(layout_params["start_state"])
    return TomatoOvercookedGridworld.from_grid(grid, layout_params, {})


def build_layout_mdp(layout_name):
    # if layout_uses_tomatoes(layout_name):
    #     return build_tomato_layout_mdp(layout_name)
    # if layout_uses_new_dynamics(layout_name):
    #     return TomatoOvercookedGridworld.from_layout_name(layout_name)
    return BaseOvercookedGridworld.from_layout_name(layout_name)


def OvercookedEnv(mdp, start_state_fn=None, horizon=1e10, debug=False):
    if isinstance(mdp, TomatoOvercookedGridworld):
        return TomatoOvercookedEnv.from_mdp(
            mdp,
            start_state_fn=start_state_fn,
            horizon=horizon,
            info_level=0,
        )
    return BaseOvercookedEnv(
        mdp,
        start_state_fn=start_state_fn,
        horizon=horizon,
        debug=debug,
    )


def build_env(layout_name, horizon):
    mdp = build_layout_mdp(layout_name)
    return mdp, OvercookedEnv(mdp, horizon=horizon)


class ZSCEvalPolicyAgent:
    def __init__(
        self,
        layout_name,
        mdp,
        agent_index,
        checkpoint_path,
        policy_config_rel_path,
        featurize_type="ppo",
        algorithm_name_override=None,
        deterministic=True,
        action_selection_epsilon=0.0,
        device="cpu",
        observation_mdp=None,
    ):
        self.layout_name = layout_name
        self.mdp = mdp
        self.observation_mdp = observation_mdp or mdp
        self.agent_index = agent_index
        self.featurize_type = featurize_type
        self.deterministic = deterministic
        self.action_selection_epsilon = action_selection_epsilon
        self.device = resolve_torch_device(device)

        policy_config_path = resolve_zsceval_policy_config_path(policy_config_rel_path)
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.checkpoint_root = identify_policy_pool_root(self.checkpoint_path)
        self.policy_config_path = os.path.abspath(policy_config_path)
        self.policy_config_root = identify_policy_pool_root(self.policy_config_path)
        with open(policy_config_path, "rb") as policy_config_file:
            unpickler = CompatUnpickler(policy_config_file)
            policy_args, obs_space, share_obs_space, act_space = copy.deepcopy(unpickler.load())

        if algorithm_name_override is not None:
            policy_args.algorithm_name = algorithm_name_override

        self.obs_shape = get_obs_space_shape(obs_space)
        self.policy = build_zsceval_policy(policy_args, obs_space, share_obs_space, act_space, self.device)
        self.policy.to(self.device)
        self.policy.load_checkpoint({"actor": self.checkpoint_path})
        self.eval_policy = EvalPolicy(policy_args, self.policy)
        self.reset()

    def reset(self):
        self.eval_policy.reset(1, 1)
        self.eval_policy.register_control_agent(0, 0)
        self.eval_policy.prep_rollout()

    def action_info(self, state, deterministic=None):
        if self.featurize_type != "ppo":
            raise NotImplementedError(f"Unsupported featurize_type: {self.featurize_type}")

        obs, available_actions = featurize_state_for_agent(
            self.mdp,
            state,
            self.agent_index,
            expected_obs_shape=self.obs_shape,
            observation_mdp=self.observation_mdp,
        )

        use_deterministic = self.deterministic if deterministic is None else deterministic
        if use_deterministic and self.action_selection_epsilon > 0 and np.random.random() < self.action_selection_epsilon:
            use_deterministic = False

        action_idx = self.eval_policy.step(
            np.array([obs], dtype=np.float32),
            [(0, 0)],
            deterministic=use_deterministic,
            masks=np.ones((1, 1), dtype=np.float32),
            available_actions=np.array([available_actions], dtype=np.uint8),
        )[0]
        action_idx = int(np.asarray(action_idx).reshape(-1)[0])

        return {
            "action": Action.INDEX_TO_ACTION[action_idx],
            "action_index": action_idx,
            "obs": obs,
            "available_actions": available_actions,
        }

    def action(self, state, deterministic=None):
        return self.action_info(state, deterministic=deterministic)["action"]


def build_checkpoint_policy_agent(
    layout_name,
    mdp,
    agent_index,
    checkpoint_path,
    algorithm_name_override=None,
    recurrent=None,
    action_selection_epsilon=0.0,
    device="cpu",
    observation_mdp=None,
):
    resolved_recurrent = checkpoint_uses_recurrent_policy(checkpoint_path) if recurrent is None else recurrent
    policy_config_rel_path = get_default_policy_config_rel_path(layout_name, recurrent=resolved_recurrent)
    return ZSCEvalPolicyAgent(
        layout_name,
        mdp,
        agent_index=agent_index,
        checkpoint_path=checkpoint_path,
        policy_config_rel_path=policy_config_rel_path,
        featurize_type="ppo",
        algorithm_name_override=algorithm_name_override,
        deterministic=True,
        action_selection_epsilon=action_selection_epsilon,
        device=device,
        observation_mdp=observation_mdp,
    )


def get_policy_pool_partner_variant_specs(layout_name, partner_name, agent_index):
    partner_name = partner_name.upper()
    if partner_name == "HSP":
        return checkpoint_variant_specs(
            layout_name,
            [
                os.path.join("hsp", "s1", "hsp", f"hsp*_w{int(agent_index)}_actor.pt"),
                os.path.join("hsp", "s2", "*", "*.pt"),
            ],
            algorithm_name_override=None,
            recurrent=None,
        )
    if partner_name == "MEP":
        return checkpoint_variant_specs(
            layout_name,
            [
                os.path.join("mep", "s1", "*", "*.pt"),
                os.path.join("mep", "s2", "*", "*.pt"),
            ],
            algorithm_name_override=None,
            recurrent=None,
        )
    return []


class HSPAgent(ZSCEvalPolicyAgent):
    def __init__(
        self,
        layout_name,
        mdp,
        agent_index=0,
        model_path=None,
        s2_run_name=None,
        s2_model_index=None,
        epsilon=None,
        device="cpu",
    ):
        if s2_run_name is None:
            resolved_run_name, resolved_model_index = get_default_hsp_s2_choice(layout_name)
        else:
            resolved_run_name, resolved_model_index = get_default_hsp_s2_choice(layout_name, run_name=s2_run_name)
        if s2_model_index is not None:
            resolved_model_index = s2_model_index

        checkpoint_path = resolve_hsp_s2_checkpoint(
            layout_name,
            run_name=resolved_run_name,
            model_index=resolved_model_index,
            model_path=model_path,
        )
        policy_config_rel_path = get_default_policy_config_rel_path(layout_name, recurrent=True)
        action_selection_epsilon = 0.5 if epsilon is None else epsilon

        super().__init__(
            layout_name,
            mdp,
            agent_index=agent_index,
            checkpoint_path=checkpoint_path,
            policy_config_rel_path=policy_config_rel_path,
            featurize_type="ppo",
            deterministic=True,
            action_selection_epsilon=action_selection_epsilon,
            device=device,
        )


class GreedyPartner:
    def __init__(
        self,
        mdp,
        agent_index,
        hl_boltzmann_rational=True,
        ll_boltzmann_rational=True,
        hl_temp=1,
        ll_temp=1,
    ):
        self.mdp = mdp
        self.agent_index = agent_index
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(
            mdp,
            NO_COUNTERS_PARAMS,
            force_compute=False,
        )
        self.model = GreedyHumanModel(
            mlp=self.mlp,
            hl_boltzmann_rational=hl_boltzmann_rational,
            ll_boltzmann_rational=ll_boltzmann_rational,
            hl_temp=hl_temp,
            ll_temp=ll_temp,
        )
        self.model.set_agent_index(agent_index)

    def reset(self):
        self.model.reset()

    def action_info(self, state, deterministic=None):
        action = normalize_action(self.model.action(state))
        obs, available_actions = featurize_state_for_agent(
            self.mdp,
            state,
            self.agent_index,
        )
        return {
            "action": action,
            "action_index": Action.ACTION_TO_INDEX[action],
            "obs": obs,
            "available_actions": available_actions,
        }

    def action(self, state, deterministic=None):
        return self.action_info(state, deterministic=deterministic)["action"]


class PopulationS2Partner(ZSCEvalPolicyAgent):
    PARTNER_NAME = None
    CHECKPOINT_SUBPATH = None
    DEFAULT_RUN_NAME = None
    DEFAULT_MODEL_INDEX = 1

    def __init__(self, layout_name, mdp, agent_index=0, model_path=None, device="cpu"):
        if self.PARTNER_NAME is None or self.CHECKPOINT_SUBPATH is None or self.DEFAULT_RUN_NAME is None:
            raise ValueError("PopulationS2Partner subclasses must define PARTNER_NAME, CHECKPOINT_SUBPATH, and DEFAULT_RUN_NAME")

        checkpoint_path = resolve_population_s2_checkpoint(
            layout_name,
            self.PARTNER_NAME,
            self.CHECKPOINT_SUBPATH,
            self.DEFAULT_RUN_NAME,
            model_index=self.DEFAULT_MODEL_INDEX,
            model_path=model_path,
        )
        policy_config_rel_path = get_default_policy_config_rel_path(
            layout_name,
            recurrent=checkpoint_uses_recurrent_policy(checkpoint_path),
        )

        super().__init__(
            layout_name,
            mdp,
            agent_index=agent_index,
            checkpoint_path=checkpoint_path,
            policy_config_rel_path=policy_config_rel_path,
            featurize_type="ppo",
            deterministic=True,
            device=device,
        )


class MEPPartner(PopulationS2Partner):
    PARTNER_NAME = "MEP"
    CHECKPOINT_SUBPATH = DEFAULT_MEP_S2_SUBPATH
    DEFAULT_RUN_NAME = DEFAULT_MEP_S2_RUN_NAME


def build_partner_agent(
    agent_name,
    layout_name,
    mdp,
    agent_index,
    hsp_epsilon=None,
    device="cpu",
):
    if agent_name == "GREEDY":
        return GreedyPartner(mdp, agent_index=agent_index)
    if agent_name == "MEP":
        return MEPPartner(layout_name, mdp, agent_index=agent_index, device=device)
    if agent_name == "HSP":
        return HSPAgent(layout_name, mdp, agent_index=agent_index, epsilon=hsp_epsilon, device=device)
    raise ValueError(f"Unsupported partner agent: {agent_name}")
