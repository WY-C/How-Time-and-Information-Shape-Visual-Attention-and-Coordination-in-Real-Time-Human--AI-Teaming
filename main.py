"""Interactive Overcooked heatmap experiment runner.

Example usage:
python main.py --layout counter_circuit --path-model-path predict_path/output/counter_circuit_1000000.pt
"""


import argparse
import copy
from collections import defaultdict, deque
import glob
import json
import os
import pickle
import re
import sys
import threading
import time
import pathlib

from experiment.counterbalancing import (
    INFORMATION_STEPS as CONDITION_INFORMATION_STEPS,
    LAYOUT_NAMES as CONDITION_LAYOUT_MAP,
    TIMES_MS as CONDITION_TIME_CONSTRAINTS_MS,
    TRIAL_COUNT as TOTAL_EXPERIMENT_TRIALS,
    format_participant_id,
    get_participant_schedule,
    parse_participant_id,
    print_schedule,
    save_schedule_csv,
)
from tutorial import (
    TUTORIAL_PATH_STEPS,
    TUTORIAL_TIME_CONSTRAINTS_MS,
    configure_tutorial_trial,
    run_tutorial,
)


def _parse_pid_arg(value):
    try:
        return format_participant_id(parse_participant_id(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_trial_arg(value):
    text = str(value).strip()
    if text.lower() == "completed":
        return "completed"
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--trial must be a 1-based trial number or 'completed'."
        ) from exc


def _parse_condition_arg(value):
    text = str(value).strip().upper()
    if not text:
        return ""
    if len(text) != 3 or text[1] != "-" or text[0] not in "ABC" or text[2] not in "012345678":
        raise argparse.ArgumentTypeError("--condition must be blank or one of A-0 through C-8.")
    return text


def resolve_condition_settings(condition):
    condition_text = _parse_condition_arg(condition)
    if not condition_text:
        return None

    layout_key = condition_text[0]
    condition_index = int(condition_text[2])
    time_constraint_ms = CONDITION_TIME_CONSTRAINTS_MS[condition_index // 3]
    info_index = condition_index % 3
    info_steps = CONDITION_INFORMATION_STEPS[info_index]

    if info_steps == 0:
        visualization_mode = PATH_VISUALIZATION_NONE
    else:
        visualization_mode = PATH_VISUALIZATION_FULL

    return {
        "condition": condition_text,
        "layout": CONDITION_LAYOUT_MAP[layout_key],
        "time_constraint_ms": time_constraint_ms,
        "path_step": info_steps,
        "visualization_mode": visualization_mode,
        "path_render_style": PATH_RENDER_STYLE_HIGHLIGHT,
    }

def _first_existing_path(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0]


def _dedupe_paths(candidates):
    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _ubuntu_project_root_candidates():
    repo_name = os.path.basename(PROJECT_ROOT)
    home_dir = os.path.expanduser("~")
    return _dedupe_paths(
        [
            PROJECT_ROOT,
            os.path.abspath(os.getcwd()),
            os.path.join(home_dir, repo_name),
        ]
    )


def _join_project_candidates(roots, *parts):
    return [os.path.join(root, *parts) for root in roots]


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UBUNTU_PROJECT_ROOT_CANDIDATES = _ubuntu_project_root_candidates()
LOCAL_OVERCOOKED_PATH = _first_existing_path(
    *_join_project_candidates(UBUNTU_PROJECT_ROOT_CANDIDATES, "lib", "overcooked_ai"),
)
if LOCAL_OVERCOOKED_PATH not in sys.path:
    sys.path.insert(0, LOCAL_OVERCOOKED_PATH)
LOCAL_ZSCEVAL_PATH = _first_existing_path(
    *_join_project_candidates(UBUNTU_PROJECT_ROOT_CANDIDATES, "lib", "ZSC-Eval"),
    *_join_project_candidates(UBUNTU_PROJECT_ROOT_CANDIDATES, "lib", "zsc_eval"),
)
LOCAL_POLICY_POOL_PATH = _first_existing_path(
    *_join_project_candidates(UBUNTU_PROJECT_ROOT_CANDIDATES, "policy_pool"),
)
LOCAL_ZSCEVAL_POLICY_POOL_PATH = _first_existing_path(
    os.path.join(LOCAL_ZSCEVAL_PATH, "policy_pool"),
    os.path.join(LOCAL_ZSCEVAL_PATH, "zsceval", "policy_pool"),
)
POLICY_POOL_ROOTS = [LOCAL_POLICY_POOL_PATH, LOCAL_ZSCEVAL_POLICY_POOL_PATH]
os.environ.setdefault("POLICY_POOL", LOCAL_POLICY_POOL_PATH)
if LOCAL_ZSCEVAL_PATH not in sys.path:
    sys.path.insert(0, LOCAL_ZSCEVAL_PATH)

import numpy as np
import pygame
import torch
from experiment_logging import (
    ExperimentLogger,
    build_visible_overlay_entries,
    resolve_response_input,
    select_ranked_normalized_entries as select_shared_ranked_normalized_entries,
    serialize_calibration_info,
)
from cooking_dynamics import (
    DEFAULT_ONION_COOK_TICKS_PER_ENV_STEP,
    configure_onion_cook_rate,
)
from preprocessing.model_dataset import build_model_dataset, get_completed_global_trial_indices
from ivt import IVTConfig
from tobii_gaze import (
    DEFAULT_GAZE_CALIBRATION_POINTS,
    DRIFT_CHECK_DURATION_SEC,
    GazeCalibration,
    TobiiGazeTracker,
    draw_gaze_calibration_overlay,
    draw_gaze_drift_overlay,
    draw_gaze_preview_overlay,
    evaluate_gaze_drift,
    evaluate_gaze_preview,
    get_display_size,
    get_surface_blit_rect,
    resolve_gaze_surface_point,
    window_point_to_display_normalized,
)

from predict_path.hsp_clone.runtime import (
    OvercookedEnv as build_runtime_env,
    build_checkpoint_policy_agent,
    build_layout_mdp as build_runtime_layout_mdp,
)
from predict_path.hsp_clone.train_hsp_policy import (
    build_hsp_policy_model_from_checkpoint,
    build_single_feature,
    checkpoint_prediction_target,
    run_action_model,
    unpack_model_outputs,
)
from overcooked_ai_py.mdp.overcooked_mdp import Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer as BaseStateVisualizer
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from zsceval.algorithms.population.utils import EvalPolicy
from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld as TomatoOvercookedGridworld,
)
from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import (
    StateVisualizer as TomatoStateVisualizer,
)

ROLLOUT_PARTNER_AGENT_NAMES = ("GREEDY", "HSP")
TARGET_GAME_FPS = 420


TIME_CONSTRAINT = 400
NUM_STEPS = 4
NUM_SAMPLES = 3
NUM_PATHS_TO_SHOW = 1
GAME_STATUS_BAR_HEIGHT = 56
MIN_EXPERIMENT_WINDOW_WIDTH = 760
MAP_RENDER_SCALE = 1.2
_CALIBRATION_WINDOW_SIZE = None
# SHOW_HEATMAP_TILES = NUM_STEPS
SHOW_HEATMAP_INTERACTIONS = NUM_STEPS
SHOW_INTERACTION_HEATMAP = True
STOP_PATH_AT_INTERACTION = False
PATH_VISUALIZATION_ENDPOINT = "endpoint"
PATH_VISUALIZATION_ENDPOINT_PLUS_FIRST = "endpoint_plus_first"
PATH_VISUALIZATION_FULL = "full"
PATH_VISUALIZATION_NONE = "none"
PATH_VISUALIZATION_MODE = PATH_VISUALIZATION_ENDPOINT_PLUS_FIRST
PATH_RENDER_STYLE_ARROW = "arrow"
PATH_RENDER_STYLE_HIGHLIGHT = "highlight"
PATH_RENDER_STYLE = PATH_RENDER_STYLE_HIGHLIGHT
SHOW_TOBII_GAZE = False

DEFAULT_HSP_S2_SUBPATH = os.path.join("hsp", "s2")
DEFAULT_HSP_S2_RUN_NAME = "hsp-S2-s36"
LAYOUT_HSP_MODEL_OVERRIDES = {
    "unident_s_hard_onion": os.path.join(
        "unident_s_hard_onion",
        "hsp",
        "s2",
        "hsp-S2-s36",
        "best_training_r.pt",
    ),
}
LAYOUT_PATH_MODEL_OVERRIDES = {
    "unident_s_hard_onion": os.path.join(
        "predict_path",
        "output",
        "asymmetric_coordination_agent0_onion83_left_compatible_best_training_r_1000000_h4_best.pt",
    ),
}

ZSCEVAL_LAYOUT_NAME_MAP = {
    "forced_coordination": "random0",
    "bothway_coordination": "random0_medium",
    "coordination_ring": "random1",
    "counter_circuit": "random3",
    "asymmetric_advantages": "unident_s",
    "unident_s_hard_legacy_obs": "unident_s_hard",
}
PATH_MODEL_LAYOUT_ALIASES = {
    "random0_medium": ("bothway_coordination",),
}
DEFAULT_HSP_S2_LAYOUT_RECOMMENDATIONS = {
    "random3": {
        "default": ("hsp-S2-s24", 3),
        "hsp-S2-s12": 5,
        "hsp-S2-s24": 3,
        "hsp-S2-s36": 2,
    },
    "double_rings": {
        "default": ("hsp-S2-s36", 5),
        "hsp-S2-s12": 1,
        "hsp-S2-s24": 1,
        "hsp-S2-s36": 1,
    },
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Overcooked heatmap experiment runner")
    parser.add_argument("--layout", default="counter_circuit", help="Overcooked layout name")
    parser.add_argument(
        "--tutorial",
        action="store_true",
        help="Run the staged tutorial with practice and score-gated tests.",
    )
    parser.add_argument(
        "--double-rings-right-pot-as-counter",
        dest="double_rings_right_pot_as_counter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In double_rings, keep game logic unchanged but render the right pot tile as a counter for the user.",
    )
    parser.add_argument(
        "--heatmap-source",
        default="model",
        choices=["rollout", "model"],
        help="Use sampled policy rollouts or a trained HSP path model for the heatmap overlay",
    )
    parser.add_argument(
        "--path-model-path",
        default='',
        help="Path to a trained HSP path model checkpoint (*.pt). Required when --heatmap-source=model.",
    )
    parser.add_argument(
        "--model-prediction",
        dest="enable_model_prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable path-model prediction in model heatmap mode. Use --no-model-prediction to disable it.",
    )
    parser.add_argument(
        "--ai-agent",
        default="HSP",
        choices=["GREEDY", "HSP"],
        help="AI agent used for player 0.",
    )
    parser.add_argument(
        "--hsp-model-path",
        default=None,
        help="Direct path to an HSP S2 checkpoint (*.pt). Overrides --hsp-s2-run-name/--hsp-s2-model-index.",
    )
    parser.add_argument(
        "--hsp-s2-run-name",
        default=None,
        help="S2 HSP run directory under policy_pool/{layout}/hsp/s2, for example hsp-S2-s24. Omit to use the layout-specific recommended run.",
    )
    parser.add_argument(
        "--hsp-s2-model-index",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Checkpoint index inside the selected HSP S2 run directory. Omit to use the recommended index for that layout/run.",
    )
    parser.add_argument(
        "--hsp-epsilon",
        type=float,
        default=0.0,
        help="Optional stochastic action epsilon for HSP. Defaults to 0.0 (deterministic).",
    )
    parser.add_argument(
        "--ai-player-index",
        type=int,
        choices=[0, 1],
        default=0,
        help="Which player slot in the environment is controlled by the AI. Default matches the current game loop: AI=0, human=1.",
    )
    parser.add_argument(
        "--ai-agent-index",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional policy observation index override. Defaults to --ai-player-index.",
    )
    parser.add_argument(
        "--agent-observation-layout",
        default=None,
        help=(
            "Optional layout used only to encode the AI policy observation. "
            "Game transitions, interactions, rendering, and action masks still use --layout."
        ),
    )
    parser.add_argument(
        "--time-constraint",
        type=int,
        default=TIME_CONSTRAINT,
        help="Action/update interval in milliseconds",
    )
    parser.add_argument(
        "--show-interaction-heatmap",
        type=str_to_bool,
        default=SHOW_INTERACTION_HEATMAP,
        help="Whether to render predicted interaction overlay",
    )
    parser.add_argument(
        "--stop-path-at-interaction",
        type=str_to_bool,
        default=STOP_PATH_AT_INTERACTION,
        help="Whether to truncate each visualized predicted path at its first interaction step.",
    )
    parser.add_argument(
        "--visualization-mode",
        choices=[
            PATH_VISUALIZATION_ENDPOINT,
            PATH_VISUALIZATION_ENDPOINT_PLUS_FIRST,
            PATH_VISUALIZATION_FULL,
            PATH_VISUALIZATION_NONE,
        ],
        default=PATH_VISUALIZATION_FULL,
        help="Ranked path visualization mode: endpoint / endpoint_plus_first / full.",
    )
    parser.add_argument(
        "--path-render-style",
        choices=[
            PATH_RENDER_STYLE_ARROW,
            PATH_RENDER_STYLE_HIGHLIGHT,
        ],
        default=PATH_RENDER_STYLE,
        help="How to draw predicted path segments when a path is shown: arrow or highlight.",
    )
    parser.add_argument(
        "--show-tobii-gaze",
        dest="show_tobii_gaze",
        type=str_to_bool,
        default=SHOW_TOBII_GAZE,
        help="Whether to render the latest Tobii gaze point on the game screen. Accepts true/false.",
    )
    parser.add_argument(
        "--eye-tracker",
        "--tobii-gaze",
        dest="show_tobii_gaze",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Enable or disable Tobii eye tracker gaze overlay. Use --eye-tracker or --no-eye-tracker.",
    )
    parser.add_argument(
        "--tobii-address",
        default=None,
        help="Optional Tobii address or serial number. Omit to use the first detected eye tracker.",
    )
    parser.add_argument(
        "--gaze-calibration-path",
        default=None,
        help="Optional JSON path for saved participant display calibration.",
    )
    parser.add_argument(
        "--gaze-calibration-points",
        type=int,
        default=DEFAULT_GAZE_CALIBRATION_POINTS,
        help="Number of display calibration targets. The dedicated calibration uses exactly 9 points.",
    )
    parser.add_argument(
        "--fixed-duration-ms",
        type=int,
        default=None,
        help="Optional wall-clock safety cap in milliseconds. Trials normally stop at --decision-horizon.",
    )
    parser.add_argument(
        "--decision-horizon",
        type=int,
        default=180,
        help="Number of decision steps per trial.",
    )
    parser.add_argument(
        "--path-step",
        default=NUM_STEPS,
        type=int,
    )
    parser.add_argument(
        "--path-num-samples",
        default=NUM_SAMPLES,
        type=int,
        help="Number of sampled future paths to visualize for rollout/model path prediction.",
    )
    parser.add_argument(
        "--experiment-logging",
        dest="enable_experiment_logging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable structured trial/step/raw-gaze/fixation logging.",
    )
    parser.add_argument(
        "--experiment-log-dir",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "experiment"),
        help="Directory where structured experiment logs are written.",
    )
    parser.add_argument(
        "--ivt-screen-diagonal-inches",
        type=float,
        default=27.0,
        help="Physical display diagonal used to convert gaze coordinates to visual angle.",
    )
    parser.add_argument(
        "--ivt-viewing-distance-mm",
        type=float,
        default=600.0,
        help="Estimated eye-to-screen distance used by the I-VT fixation classifier.",
    )
    parser.add_argument(
        "--pid",
        "--participant-id",
        dest="pid",
        type=_parse_pid_arg,
        default=None,
        help="Participant ID such as P01. Runs its deterministic 27-trial counterbalanced schedule.",
    )
    parser.add_argument(
        "--trial-id",
        default=None,
        help="Optional trial identifier stored in structured experiment logs.",
    )
    parser.add_argument(
        "--trial",
        type=_parse_trial_arg,
        default=1,
        help=(
            "1-based trial number to resume from, or 'completed' to scan from "
            "trial 1 and skip every trial already marked completed."
        ),
    )
    parser.add_argument(
        "--condition",
        type=_parse_condition_arg,
        default="",
        help="Run one explicit condition (A-0 through C-8), bypassing the automatic 27-condition order.",
    )
    parser.add_argument(
        "--condition-num",
        dest="condition",
        type=_parse_condition_arg,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--print-schedule",
        action="store_true",
        help="Print the PID's full schedule before running it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate, validate, save, and print the schedule without starting the game or eye tracker.",
    )
    parser.add_argument(
        "--schedule-output",
        default=None,
        help="Optional schedule CSV path. Defaults to <experiment-log-dir>/<PID>/schedule.csv.",
    )
    parser.add_argument(
        "--processed-output-dir",
        default=None,
        help="Derived dataset directory. Defaults to <project-root>/data/processed/<PID>.",
    )
    return parser.parse_args()


def build_render_terrain_mtx(layout_name, terrain_mtx, double_rings_right_pot_as_counter=False):
    if not double_rings_right_pot_as_counter or layout_name != "double_rings" or terrain_mtx is None:
        return terrain_mtx

    render_terrain_mtx = [list(row) for row in terrain_mtx]
    right_pot_y = 5
    right_pot_x = 4
    if (
        0 <= right_pot_y < len(render_terrain_mtx)
        and 0 <= right_pot_x < len(render_terrain_mtx[right_pot_y])
        and render_terrain_mtx[right_pot_y][right_pot_x] == "P"
    ):
        render_terrain_mtx[right_pot_y][right_pot_x] = "X"
    return render_terrain_mtx


def validate_agent_observation_mdp(game_mdp, observation_mdp):
    game_shape = (len(game_mdp.terrain_mtx), len(game_mdp.terrain_mtx[0]))
    observation_shape = (
        len(observation_mdp.terrain_mtx),
        len(observation_mdp.terrain_mtx[0]),
    )
    if observation_shape != game_shape:
        raise ValueError(
            "Agent observation layout must have the same grid shape as the game layout: "
            f"game={game_shape}, observation={observation_shape}."
        )

    game_positions = set(game_mdp.get_valid_player_positions())
    observation_positions = set(observation_mdp.get_valid_player_positions())
    if observation_positions != game_positions:
        raise ValueError(
            "Agent observation layout must have the same walkable tiles as the game layout."
        )


def should_block_double_rings_right_pot_interaction(
    layout_name,
    state,
    player_index,
    action,
    double_rings_right_pot_as_counter=False,
):
    if (
        not double_rings_right_pot_as_counter
        or layout_name != "double_rings"
        or action != Action.INTERACT
        or state is None
    ):
        return False

    player = state.players[player_index]
    interact_pos = Action.move_in_direction(player.position, player.orientation)
    return tuple(interact_pos) == (4, 5)


def get_layout_candidate_paths(layout_name, subpath):
    candidate_paths = []
    for candidate_layout in get_policy_pool_layout_candidates(layout_name):
        for policy_pool_root in POLICY_POOL_ROOTS:
            candidate_paths.append(os.path.abspath(os.path.join(policy_pool_root, candidate_layout, subpath)))
    return candidate_paths


def find_first_existing_path(candidate_paths):
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _checkpoint_step_sort_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    suffix = stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def normalize_hsp_s2_run_name(run_name):
    if str(run_name).isdigit():
        return f"hsp-S2-s{run_name}"
    return run_name


def get_agent_build_kwargs(args):
    return {
        "hsp_model_path": args.hsp_model_path,
        "hsp_s2_run_name": args.hsp_s2_run_name,
        "hsp_s2_model_index": args.hsp_s2_model_index,
        "hsp_epsilon": args.hsp_epsilon,
    }


def build_agent_from_args(
    args,
    layout_name,
    mdp,
    agent_index,
    agent_name=None,
    observation_mdp=None,
):
    tutorial_checkpoint = getattr(args, "_tutorial_ai_checkpoint", None)
    if tutorial_checkpoint:
        return build_checkpoint_policy_agent(
            layout_name,
            mdp,
            agent_index=agent_index,
            checkpoint_path=tutorial_checkpoint,
            algorithm_name_override=getattr(args, "_tutorial_ai_algorithm", None),
            action_selection_epsilon=0.0,
            device="cpu",
            observation_mdp=observation_mdp,
        )
    return build_ai_agent(
        agent_name or args.ai_agent,
        layout_name,
        mdp,
        agent_index=agent_index,
        observation_mdp=observation_mdp,
        **get_agent_build_kwargs(args),
    )


def get_zsceval_layout_candidates(layout_name):
    candidates = [layout_name]
    mapped_layout = ZSCEVAL_LAYOUT_NAME_MAP.get(layout_name)
    if mapped_layout and mapped_layout not in candidates:
        candidates.append(mapped_layout)
    return candidates


def get_policy_pool_layout_candidates(layout_name):
    candidates = [layout_name]
    if layout_name not in candidates:
        candidates.append(layout_name)
    mapped_zsceval_layout = ZSCEVAL_LAYOUT_NAME_MAP.get(layout_name)
    if mapped_zsceval_layout and mapped_zsceval_layout not in candidates:
        candidates.append(mapped_zsceval_layout)
    return candidates


def get_recommended_zsceval_layout_name(layout_name):
    for candidate in get_zsceval_layout_candidates(layout_name):
        if candidate in DEFAULT_HSP_S2_LAYOUT_RECOMMENDATIONS:
            return candidate
    return get_zsceval_layout_candidates(layout_name)[-1]


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
            policy_config_abs_path = os.path.abspath(
                os.path.join(policy_pool_root, policy_config_rel_path)
            )
            if os.path.exists(policy_config_abs_path):
                return policy_config_rel_path

    raise FileNotFoundError(
        f"Policy config not found for layout '{layout_name}'. Tried: {tried_paths}"
    )


def resolve_hsp_s2_checkpoint(layout_name, run_name=DEFAULT_HSP_S2_RUN_NAME, model_index=1, model_path=None):
    if model_path:
        checkpoint_path = os.path.abspath(model_path)
    elif layout_name in LAYOUT_HSP_MODEL_OVERRIDES:
        relative_path = LAYOUT_HSP_MODEL_OVERRIDES[layout_name]
        checkpoint_path = find_first_existing_path(
            [os.path.join(root, relative_path) for root in POLICY_POOL_ROOTS]
        )
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


def resolve_path_model_checkpoint(layout_name, model_path=None):
    if model_path:
        checkpoint_path = os.path.abspath(model_path)
    elif layout_name in LAYOUT_PATH_MODEL_OVERRIDES:
        checkpoint_path = os.path.join(
            PROJECT_ROOT,
            LAYOUT_PATH_MODEL_OVERRIDES[layout_name],
        )
    else:
        matching_paths = []
        candidate_layouts = get_policy_pool_layout_candidates(layout_name)
        for alias in PATH_MODEL_LAYOUT_ALIASES.get(layout_name, ()):
            if alias not in candidate_layouts:
                candidate_layouts.append(alias)
        for candidate_layout in candidate_layouts:
            pattern = os.path.join(
                PROJECT_ROOT,
                "predict_path",
                "output",
                f"{candidate_layout}_*.pt",
            )
            matching_paths.extend(glob.glob(pattern))
        checkpoint_path = max(matching_paths, key=_checkpoint_step_sort_key) if matching_paths else None

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Path model checkpoint not found for layout '{layout_name}'. "
            f"Expected --path-model-path or a file matching predict_path/output/{layout_name}_*.pt"
        )

    return checkpoint_path


def resolve_tutorial_ai_checkpoint(stage):
    if stage.checkpoint_layout is None:
        return None
    relative_path = os.path.join(stage.checkpoint_layout, stage.checkpoint_subpath)
    candidates = [os.path.join(root, relative_path) for root in POLICY_POOL_ROOTS]
    checkpoint_path = find_first_existing_path(candidates)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Tutorial beta checkpoint not found for layout '{stage.layout}'. Tried: {candidates}"
        )
    return checkpoint_path


def resolve_tutorial_path_model(stage):
    output_dir = os.path.join(PROJECT_ROOT, "predict_path", "output")
    preferred_path = os.path.join(output_dir, f"tutorial_beta_{stage.layout}.pt")
    if os.path.exists(preferred_path):
        return preferred_path

    matching_paths = glob.glob(os.path.join(output_dir, f"{stage.layout}_*.pt"))
    if matching_paths:
        return max(matching_paths, key=_checkpoint_step_sort_key)
    return preferred_path


def configure_tutorial(args, stage, phase):
    configure_tutorial_trial(
        args,
        stage,
        phase,
        PROJECT_ROOT,
        resolve_tutorial_ai_checkpoint,
    )
    if args.enable_model_prediction:
        args.path_model_path = resolve_tutorial_path_model(stage)

def resolve_zsceval_policy_config_path(policy_config_rel_path):
    tried_paths = []
    for policy_pool_root in POLICY_POOL_ROOTS:
        config_path = os.path.abspath(os.path.join(policy_pool_root, policy_config_rel_path))
        tried_paths.append(config_path)
        if os.path.exists(config_path):
            return config_path
    raise FileNotFoundError(f"Policy config not found. Tried: {tried_paths}")


def resolve_hsp_s2_policy_config_path(layout_name):
    inferred_path = get_default_policy_config_rel_path(layout_name, recurrent=True)
    print(f"[HSP S2] Using default policy config '{inferred_path}' for layout '{layout_name}'")
    return inferred_path


def get_obs_space_shape(obs_space):
    return tuple(getattr(obs_space, "shape", getattr(obs_space, "_shape", ())))


def build_zsceval_policy(policy_args, obs_space, share_obs_space, act_space):
    algorithm_name = policy_args.algorithm_name
    if algorithm_name in {"mappo", "rmappo"}:
        return R_MAPPOPolicy(policy_args, obs_space, share_obs_space, act_space, device=torch.device("cpu"))
    raise NotImplementedError(f"Unsupported zsceval policy algorithm: {algorithm_name}")


def get_available_actions_for_state(mdp, state):
    num_agents = len(state.players)
    available_actions = np.ones((num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
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
        observation_mdp=None,
    ):
        self.layout_name = layout_name
        self.mdp = mdp
        self.observation_mdp = observation_mdp or mdp
        self.agent_index = agent_index
        self.checkpoint_path = checkpoint_path
        self.policy_config_rel_path = policy_config_rel_path
        self.policy_config_path = resolve_zsceval_policy_config_path(policy_config_rel_path)
        self.featurize_type = featurize_type
        self.deterministic = deterministic
        self.action_selection_epsilon = action_selection_epsilon

        with open(self.policy_config_path, "rb") as f:
            if os.name == "nt":
                pathlib.PosixPath = pathlib.WindowsPath
            policy_args, obs_space, share_obs_space, act_space = copy.deepcopy(pickle.load(f))
        if algorithm_name_override is not None:
            policy_args.algorithm_name = algorithm_name_override

        self.policy_args = policy_args
        self.obs_space = obs_space
        self.obs_shape = get_obs_space_shape(obs_space)
        self.policy = build_zsceval_policy(policy_args, obs_space, share_obs_space, act_space)
        self.policy.to(torch.device("cpu"))
        self.policy.load_checkpoint({"actor": self.checkpoint_path})
        self.eval_policy = EvalPolicy(policy_args, self.policy)
        self.reset()

    def reset(self):
        self.eval_policy.reset(1, 1)
        self.eval_policy.register_control_agent(0, 0)
        self.eval_policy.prep_rollout()

    def clone(self):
        cloned = copy.copy(self)
        cloned.eval_policy = copy.copy(self.eval_policy)
        cloned.eval_policy._control_agents = list(self.eval_policy._control_agents)
        cloned.eval_policy._map_a2id = dict(self.eval_policy._map_a2id)
        cloned.eval_policy._rnn_states = {
            k: np.copy(v) for k, v in self.eval_policy._rnn_states.items()
        }
        return cloned

    def _featurize_state(self, state):
        if self.featurize_type != "ppo":
            raise NotImplementedError(f"Unsupported featurize_type: {self.featurize_type}")

        obs = self.observation_mdp.lossless_state_encoding(state)[self.agent_index]
        if self.obs_shape and tuple(obs.shape) != self.obs_shape:
            if obs.shape[-1] == 26 and self.obs_shape[-1] == 20:
                obs = convert_obs_to_hsp_channels(obs)
            else:
                raise ValueError(
                    f"Observation shape mismatch. expected={self.obs_shape}, got={tuple(obs.shape)}"
                )

        available_actions = get_available_actions_for_state(self.mdp, state)[self.agent_index]
        return obs.astype(np.float32) * 255.0, available_actions

    def action(self, state, deterministic=None):
        obs, available_actions = self._featurize_state(state)
        if deterministic is None:
            deterministic = self.deterministic
        if deterministic and self.action_selection_epsilon > 0 and np.random.random() < self.action_selection_epsilon:
            deterministic = False

        self.eval_policy.prep_rollout()

        action_idx = self.eval_policy.step(
            np.array([obs], dtype=np.float32),
            [(0, 0)],
            deterministic=deterministic,
            masks=np.ones((1, 1), dtype=np.float32),
            available_actions=np.array([available_actions], dtype=np.uint8),
        )[0]
        action_idx = int(np.asarray(action_idx).reshape(-1)[0])
        return Action.INDEX_TO_ACTION[action_idx]


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
        observation_mdp=None,
    ):
        self.s2_run_name = None
        self.s2_model_index = None

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
        policy_config_rel_path = resolve_hsp_s2_policy_config_path(layout_name)
        featurize_type = "ppo"
        action_selection_epsilon = 0.5 if epsilon is None else epsilon
        self.s2_run_name = resolved_run_name
        self.s2_model_index = resolved_model_index

        super().__init__(
            layout_name,
            mdp,
            agent_index=agent_index,
            checkpoint_path=checkpoint_path,
            policy_config_rel_path=policy_config_rel_path,
            featurize_type=featurize_type,
            deterministic=True,
            action_selection_epsilon=action_selection_epsilon,
            observation_mdp=observation_mdp,
        )


def build_ai_agent(
    agent_name,
    layout_name,
    mdp,
    agent_index=0,
    hsp_model_path=None,
    hsp_s2_run_name=None,
    hsp_s2_model_index=None,
    hsp_epsilon=None,
    observation_mdp=None,
):
    if agent_name == "GREEDY":
        if observation_mdp is not None and observation_mdp is not mdp:
            raise ValueError("--agent-observation-layout is not supported with --ai-agent GREEDY.")
        return GreedyModel(mdp, agent_index=agent_index)
    if agent_name == "HSP":
        return HSPAgent(
            layout_name,
            mdp,
            agent_index=agent_index,
            model_path=hsp_model_path,
            s2_run_name=hsp_s2_run_name,
            s2_model_index=hsp_s2_model_index,
            epsilon=hsp_epsilon,
            observation_mdp=observation_mdp,
        )
    raise ValueError(f"Unsupported ai agent: {agent_name}")


def build_rollout_partner_agents(args, layout_name, mdp, partner_agent_index):
    partner_agents = []
    partner_names = []

    for agent_name in ROLLOUT_PARTNER_AGENT_NAMES:
        try:
            if agent_name == "GREEDY":
                partner_agent = GreedyModel(mdp, agent_index=partner_agent_index)
            else:
                partner_agent = build_agent_from_args(
                    args,
                    layout_name,
                    mdp,
                    agent_index=partner_agent_index,
                    agent_name=agent_name,
                )
        except Exception as exc:
            print(f"[Partner Agents] Skipping {agent_name}: {type(exc).__name__}: {exc}")
            continue

        partner_agents.append(partner_agent)
        partner_names.append(agent_name)

    if not partner_agents:
        raise RuntimeError("No rollout partner agents could be created.")

    return partner_agents, partner_names


def snapshot_rollout_agent(agent):
    return agent.clone() if hasattr(agent, "clone") else agent


def background_rollout_worker(current_state, ai_agent, partner_list, mdp, result_container, ai_player_index):
    """Run rollout sampling in a background thread."""
    start_time = time.perf_counter()
    ai_paths = []
    partner_paths = []
    ai_start_player = current_state.players[ai_player_index]

    for _ in range(NUM_SAMPLES):
        ai_batch, partner_batch = run_rollout(current_state, ai_agent, partner_list, mdp, ai_player_index)
        ai_paths.extend(ai_batch)
        partner_paths.extend(partner_batch)

    end_time = time.perf_counter()
   
    result_container["ai_paths"] = ai_paths
    result_container["partner_paths"] = partner_paths
    result_container["ai_start_position"] = tuple(ai_start_player.position)
    result_container["ai_start_orientation"] = ai_start_player.orientation
    result_container["time_ms"] = (end_time - start_time) * 1000


class PathModelHeatmapPredictor:
    def __init__(self, model_path, mdp, agent_index):
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model_path = os.path.abspath(model_path)
        self.mdp = mdp
        self.agent_index = agent_index
        self.prediction_target = checkpoint_prediction_target(checkpoint)
        self.history_len = int(checkpoint["history_len"])
        self.target_horizon = int(checkpoint["target_horizon"])
        self.grid_width = int(checkpoint["grid_width"])
        self.grid_height = int(checkpoint["grid_height"])
        self.obs_shape = tuple(checkpoint["obs_shape"])
        self.valid_positions = np.asarray(checkpoint["valid_positions"], dtype=np.int64)
        self.num_positions = int(checkpoint["num_positions"])
        self.num_actions = int(checkpoint.get("num_actions", len(Action.ALL_ACTIONS)))
        self.predict_interact = bool(checkpoint.get("predict_interact", False))
        self.action_model_type = checkpoint.get("action_model_type", "parallel")
        self.model = build_hsp_policy_model_from_checkpoint(checkpoint)

    def _featurize_state(self, state):
        obs = self.mdp.lossless_state_encoding(state)[self.agent_index]
        if self.obs_shape and tuple(obs.shape) != self.obs_shape:
            if obs.shape[-1] == 26 and self.obs_shape[-1] == 20:
                obs = convert_obs_to_hsp_channels(obs)
            else:
                raise ValueError(
                    f"Observation shape mismatch. expected={self.obs_shape}, got={tuple(obs.shape)}"
                )
        available_actions = get_available_actions_for_state(self.mdp, state)[self.agent_index]
        return obs.astype(np.float32) * 255.0, available_actions

    def predict_outputs(self, state, self_position_history, partner_position_history):
        obs, available_actions = self._featurize_state(state)
        feature = build_single_feature(
            obs,
            available_actions,
            np.asarray(self_position_history, dtype=np.int64),
            np.asarray(partner_position_history, dtype=np.int64),
            self.grid_width,
            self.grid_height,
        )
        start_positions = torch.from_numpy(np.asarray(self_position_history, dtype=np.int64)[-1:]).reshape(1, 2)
        with torch.no_grad():
            feature_tensor = torch.from_numpy(feature[None, :])
            if self.prediction_target == "action":
                action_logits = run_action_model(
                    self.model,
                    feature_tensor,
                    start_positions,
                    target_actions=None,
                )
                action_probabilities = torch.softmax(action_logits[0], dim=-1).cpu().numpy()
                interact_action_index = Action.ACTION_TO_INDEX[Action.INTERACT]
                return {
                    "action_probabilities": action_probabilities,
                    "position_probabilities": None,
                    "interact_probabilities": action_probabilities[:, interact_action_index].astype(np.float32),
                }

            model_outputs = self.model(feature_tensor)
            position_logits, interact_logits = unpack_model_outputs(model_outputs)
            position_probabilities = torch.softmax(position_logits[0], dim=-1).cpu().numpy()
            if interact_logits is None:
                return {
                    "action_probabilities": None,
                    "position_probabilities": position_probabilities,
                    "interact_probabilities": None,
                }
            interact_probabilities = torch.softmax(interact_logits[0], dim=-1)[:, 1].cpu().numpy()
            return {
                "action_probabilities": None,
                "position_probabilities": position_probabilities,
                "interact_probabilities": interact_probabilities,
            }
def infer_orientation_from_positions(previous_position, current_position, fallback_orientation):
    dx = int(current_position[0]) - int(previous_position[0])
    dy = int(current_position[1]) - int(previous_position[1])
    if (dx, dy) == Direction.NORTH:
        return Direction.NORTH
    if (dx, dy) == Direction.SOUTH:
        return Direction.SOUTH
    if (dx, dy) == Direction.EAST:
        return Direction.EAST
    if (dx, dy) == Direction.WEST:
        return Direction.WEST
    return fallback_orientation


def sample_action_index_sequences(model, feature_tensor, start_positions, num_samples):
    if num_samples <= 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    with torch.no_grad():
        if getattr(model, "action_model_type", "parallel") != "autoregressive":
            action_logits = run_action_model(
                model,
                feature_tensor,
                start_positions,
                target_actions=None,
            )
            action_probabilities = torch.softmax(action_logits[0], dim=-1)
            sampled_sequences = []
            sampled_log_probs = []
            for _ in range(num_samples):
                sampled_actions = torch.multinomial(action_probabilities, 1).squeeze(-1)
                sampled_sequences.append(sampled_actions.cpu().numpy())
                sampled_action_probabilities = action_probabilities.gather(1, sampled_actions.unsqueeze(-1)).squeeze(-1)
                sampled_log_probs.append(torch.log(sampled_action_probabilities.clamp_min(1e-8)).sum().item())
            return np.stack(sampled_sequences, axis=0), np.asarray(sampled_log_probs, dtype=np.float32)

        context = model.backbone(feature_tensor)
        hidden_init = torch.tanh(model.decoder_init(context))
        sampled_sequences = []
        sampled_log_probs = []

        for _ in range(num_samples):
            hidden = hidden_init.clone()
            current_positions = start_positions.to(device=feature_tensor.device, dtype=torch.long).clone()
            prev_actions = torch.full(
                (feature_tensor.shape[0],),
                model.start_action_index,
                device=feature_tensor.device,
                dtype=torch.long,
            )
            sampled_actions = []
            sampled_log_prob = torch.zeros((feature_tensor.shape[0],), device=feature_tensor.device, dtype=torch.float32)

            for horizon_idx in range(model.target_horizon):
                step_indices = torch.full(
                    (feature_tensor.shape[0],),
                    horizon_idx,
                    device=feature_tensor.device,
                    dtype=torch.long,
                )
                decoder_input = torch.cat(
                    [
                        context,
                        model.action_embedding(prev_actions),
                        model.step_embedding(step_indices),
                        torch.relu(model.position_encoder(model._normalize_positions(current_positions))),
                    ],
                    dim=1,
                )
                hidden = model.decoder(decoder_input, hidden)
                logits = model.head(hidden)
                probabilities = torch.softmax(logits, dim=-1)
                next_actions = torch.multinomial(probabilities, 1).squeeze(-1)
                sampled_actions.append(next_actions)
                sampled_action_probabilities = probabilities.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                sampled_log_prob = sampled_log_prob + torch.log(sampled_action_probabilities.clamp_min(1e-8))
                current_positions = model._advance_positions(current_positions, next_actions)
                prev_actions = next_actions

            sampled_sequences.append(torch.stack(sampled_actions, dim=1)[0].cpu().numpy())
            sampled_log_probs.append(sampled_log_prob[0].item())

    return np.stack(sampled_sequences, axis=0), np.asarray(sampled_log_probs, dtype=np.float32)


def normalize_path_weights_from_log_probs(log_probabilities):
    log_probabilities = np.asarray(log_probabilities, dtype=np.float64)
    if log_probabilities.size == 0:
        return []
    shifted = log_probabilities - np.max(log_probabilities)
    weights = np.exp(shifted)
    total = weights.sum()
    if total <= 0:
        return [1.0 / log_probabilities.size] * int(log_probabilities.size)
    return (weights / total).tolist()


def build_predicted_action_path_from_indices(action_indices, start_position, start_orientation, valid_positions):
    if action_indices is None:
        return []

    valid_position_set = {
        tuple(int(component) for component in position)
        for position in np.asarray(valid_positions, dtype=np.int64)
    }
    current_position = tuple(int(component) for component in start_position)
    current_orientation = start_orientation
    predicted_path = []

    for action_index in np.asarray(action_indices, dtype=np.int64)[:NUM_STEPS]:
        action = Action.INDEX_TO_ACTION[int(action_index)]
        if action == Action.INTERACT:
            interact_position = tuple(Action.move_in_direction(current_position, current_orientation))
            predicted_path.append(
                {
                    "position": current_position,
                    "interact": True,
                    "interact_position": interact_position,
                }
            )
            continue

        if action in Direction.ALL_DIRECTIONS:
            current_orientation = action
            candidate = tuple(Action.move_in_direction(current_position, action))
            if candidate in valid_position_set:
                current_position = candidate

        predicted_path.append(
            {
                "position": current_position,
                "interact": False,
            }
        )

    return predicted_path


def build_predicted_action_path(action_probabilities, start_position, start_orientation, valid_positions):
    if action_probabilities is None:
        return []

    action_indices = np.asarray(action_probabilities, dtype=np.float32).argmax(axis=1)
    return build_predicted_action_path_from_indices(
        action_indices,
        start_position,
        start_orientation,
        valid_positions,
    )


def run_path_model_prediction(
    current_state,
    predictor,
    self_position_history,
    partner_position_history,
):
    start_time = time.perf_counter()
    prediction_outputs = predictor.predict_outputs(
        current_state,
        self_position_history,
        partner_position_history,
    )

    start_player = current_state.players[predictor.agent_index]
    action_probabilities = prediction_outputs["action_probabilities"]
    if action_probabilities is not None:
        action_probabilities = action_probabilities[:NUM_STEPS]
        interact_probabilities = prediction_outputs["interact_probabilities"]
        if interact_probabilities is not None:
            interact_probabilities = np.asarray(interact_probabilities, dtype=np.float32)[:NUM_STEPS]
        obs, available_actions = predictor._featurize_state(current_state)
        feature = build_single_feature(
            obs,
            available_actions,
            np.asarray(self_position_history, dtype=np.int64),
            np.asarray(partner_position_history, dtype=np.int64),
            predictor.grid_width,
            predictor.grid_height,
        )
        feature_tensor = torch.from_numpy(feature[None, :])
        start_positions = torch.from_numpy(np.asarray(self_position_history, dtype=np.int64)[-1:]).reshape(1, 2)
        greedy_action_indices = np.asarray(action_probabilities, dtype=np.float32).argmax(axis=1)
        greedy_action_probabilities = np.asarray(
            [
                action_probabilities[step_idx, greedy_action_indices[step_idx]]
                for step_idx in range(len(greedy_action_indices))
            ],
            dtype=np.float32,
        )
        greedy_log_probability = float(np.log(np.clip(greedy_action_probabilities, 1e-8, 1.0)).sum())
        sampled_action_sequences, sampled_log_probabilities = sample_action_index_sequences(
            predictor.model,
            feature_tensor,
            start_positions,
            max(0, NUM_SAMPLES - 1),
        )
        sampled_paths = [
            build_predicted_action_path(
                action_probabilities,
                tuple(start_player.position),
                start_player.orientation,
                predictor.valid_positions,
            )
        ]
        for sampled_actions in sampled_action_sequences:
            sampled_paths.append(
                build_predicted_action_path_from_indices(
                    sampled_actions,
                    tuple(start_player.position),
                    start_player.orientation,
                    predictor.valid_positions,
                )
            )
        path_weights = normalize_path_weights_from_log_probs(
            np.concatenate(
                [
                    np.asarray([greedy_log_probability], dtype=np.float32),
                    sampled_log_probabilities,
                ]
            )
        )
        return {
            "ai_paths": sampled_paths,
            "ai_path_weights": path_weights,
            "ai_greedy_path_count": 1,
            "ai_greedy_action_indices": greedy_action_indices.astype(np.int64).tolist(),
            "partner_paths": [],
            "ai_position_probabilities": None,
            "ai_valid_positions": predictor.valid_positions,
            "ai_interact_probabilities": interact_probabilities,
            "ai_start_position": tuple(start_player.position),
            "ai_start_orientation": start_player.orientation,
            "time_ms": (time.perf_counter() - start_time) * 1000,
        }

    position_probabilities = prediction_outputs["position_probabilities"][:NUM_STEPS]
    interact_probabilities = prediction_outputs["interact_probabilities"]
    if interact_probabilities is not None:
        interact_probabilities = interact_probabilities[:NUM_STEPS]

    return {
        "ai_paths": [],
        "ai_path_weights": [],
        "ai_greedy_path_count": 0,
        "ai_greedy_action_indices": [],
        "partner_paths": [],
        "ai_position_probabilities": position_probabilities,
        "ai_valid_positions": predictor.valid_positions,
        "ai_interact_probabilities": interact_probabilities,
        "ai_start_position": tuple(start_player.position),
        "ai_start_orientation": start_player.orientation,
        "time_ms": (time.perf_counter() - start_time) * 1000,
    }


def run_rollout(current_state, ai_agent, partner_list, mdp, ai_player_index):
    all_ai_positions = []
    all_partner_positions = []
    partner_player_index = 1 - ai_player_index
   
    for partner in partner_list:
        sim_state = copy.deepcopy(current_state)
        future_ai_positions = []
        future_partner_positions = []
        sim_ai_agent = snapshot_rollout_agent(ai_agent)
        sim_partner_agent = snapshot_rollout_agent(partner)
       
        for _ in range(NUM_STEPS):
            ai_action = sim_ai_agent.action(sim_state)
            partner_action = sim_partner_agent.action(sim_state)

            if isinstance(ai_action, (int, np.integer)):
                ai_action = Action.INDEX_TO_ACTION[ai_action]
            if isinstance(partner_action, (int, np.integer)):
                partner_action = Action.INDEX_TO_ACTION[partner_action]

            joint_action = [None, None]
            joint_action[ai_player_index] = ai_action
            joint_action[partner_player_index] = partner_action
            transition = mdp.get_state_transition(sim_state, tuple(joint_action))
            sim_state = transition[0] if isinstance(transition, tuple) else transition

            ai_pos = sim_state.players[ai_player_index].position
            ai_player = sim_state.players[ai_player_index]
            if ai_action == Action.INTERACT:
                facing_pos = Action.move_in_direction(ai_pos, ai_player.orientation)
                future_ai_positions.append((facing_pos[0], facing_pos[1], 'interact'))
            else:
                future_ai_positions.append((ai_pos[0], ai_pos[1]))
               
            partner_pos = sim_state.players[partner_player_index].position
            partner_player = sim_state.players[partner_player_index]
            if partner_action == Action.INTERACT:
                facing_pos = Action.move_in_direction(partner_pos, partner_player.orientation)
                future_partner_positions.append((facing_pos[0], facing_pos[1], 'interact'))
            else:
                future_partner_positions.append((partner_pos[0], partner_pos[1]))

        all_ai_positions.append(future_ai_positions)
        all_partner_positions.append(future_partner_positions)

    return all_ai_positions, all_partner_positions

class GreedyModel():
    def __init__(self, mdp, agent_index=1, hl_boltzmann_rational=True, ll_boltzmann_rational=True, hl_temp=1, ll_temp=1):
        self.mdp = mdp
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(
            mdp,
            NO_COUNTERS_PARAMS,
            force_compute=False
        )
        self.model = GreedyHumanModel(
            mlp=self.mlp,
            hl_boltzmann_rational=hl_boltzmann_rational,
            ll_boltzmann_rational=ll_boltzmann_rational,
            hl_temp=hl_temp,
            ll_temp=ll_temp
        )
        self.model.set_agent_index(agent_index)

    def action(self, state):
        return self.model.action(state)

def draw_heatmap_overlay(
    surface,
    terrain_mtx,
    ai_paths,
    partner_paths,
    num_heatmap_tiles,
    num_heatmap_interactions,
    show_interaction_heatmap,
    ai_position_probabilities=None,
    ai_valid_positions=None,
    ai_interact_probabilities=None,
    ai_start_position=None,
    ai_start_orientation=None,
    ai_path_weights=None,
    ai_greedy_path_count=0,
    path_animation_progress=1.0,
    previous_rollout_results=None,
):
    def truncate_path_after_first_interaction(path):
        if not STOP_PATH_AT_INTERACTION or not path:
            return path
        truncated_path = []
        for step in path:
            truncated_path.append(step)
            if isinstance(step, dict):
                if step.get("interact", False):
                    break
            elif len(step) >= 3 and step[2] == "interact":
                break
        return truncated_path

    ai_paths = [truncate_path_after_first_interaction(path) for path in ai_paths]
    partner_paths = [truncate_path_after_first_interaction(path) for path in partner_paths]

    if not ai_paths and ai_position_probabilities is None:
        return

    grid_width = len(terrain_mtx[0])
    grid_height = len(terrain_mtx)
    tile_w = surface.get_width() / grid_width
    tile_h = surface.get_height() / grid_height

    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    total_paths = len(ai_paths)
    min_visible_tile_alpha = 16
    badge_fonts = {}
    def in_bounds(tile):
        return 0 <= tile[0] < grid_width and 0 <= tile[1] < grid_height

    def get_tile_rect(tile):
        left = int(round(tile[0] * tile_w))
        top = int(round(tile[1] * tile_h))
        right = int(round((tile[0] + 1) * tile_w))
        bottom = int(round((tile[1] + 1) * tile_h))
        return pygame.Rect(left, top, max(1, right - left), max(1, bottom - top))

    def extract_position(step):
        if isinstance(step, dict):
            return tuple(step["position"])
        return (step[0], step[1])

    def extract_direction(current_position, next_position):
        dx = int(next_position[0]) - int(current_position[0])
        dy = int(next_position[1]) - int(current_position[1])
        direction = (dx, dy)
        if direction in {Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST}:
            return direction
        return None

    def vector_to_direction(dx, dy, fallback=None):
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return fallback
        if abs(dx) >= abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        return Direction.SOUTH if dy > 0 else Direction.NORTH

    def build_partial_polyline(points, progress):
        if not points:
            return []
        progress = max(0.0, min(1.0, float(progress)))
        if len(points) == 1 or progress >= 0.999:
            return list(points)
        if progress <= 0.0:
            return [points[0]]

        segment_lengths = []
        total_length = 0.0
        for start_point, end_point in zip(points[:-1], points[1:]):
            length = float(np.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1]))
            segment_lengths.append(length)
            total_length += length

        if total_length <= 1e-6:
            return [points[0]]

        target_length = total_length * progress
        traversed_length = 0.0
        partial_points = [points[0]]

        for index, (start_point, end_point) in enumerate(zip(points[:-1], points[1:])):
            segment_length = segment_lengths[index]
            if segment_length <= 1e-6:
                continue
            next_length = traversed_length + segment_length
            if next_length < target_length - 1e-6:
                partial_points.append(end_point)
                traversed_length = next_length
                continue

            segment_progress = max(0.0, min(1.0, (target_length - traversed_length) / segment_length))
            interpolated_point = (
                start_point[0] + ((end_point[0] - start_point[0]) * segment_progress),
                start_point[1] + ((end_point[1] - start_point[1]) * segment_progress),
            )
            partial_points.append(interpolated_point)
            break

        return partial_points

    def draw_direction_arrow(
        target_surface,
        tile,
        direction,
        alpha,
        center_offset=None,
        scale=1.0,
        fill_color=None,
        highlight_color=None,
        shadow_color=None,
    ):
        if alpha <= 0:
            return
        rect = get_tile_rect(tile)
        offset_x = 0 if center_offset is None else int(round(center_offset[0]))
        offset_y = 0 if center_offset is None else int(round(center_offset[1]))
        center_x = rect.centerx + offset_x
        center_y = rect.centery + offset_y
        arrow_scale = max(0.35, float(scale))
        arrow_size = max(9, int(min(rect.width, rect.height) * 0.78 * arrow_scale))
        half_len = arrow_size * 0.50
        head_len = arrow_size * 0.28
        head_half_width = max(5, int(arrow_size * 0.23))
        tail_half_width = max(4, int(arrow_size * 0.12))
        shaft_end = half_len - head_len
        tail_start = -half_len * 0.1

        local_points = [
            (tail_start, -tail_half_width),
            (shaft_end, -tail_half_width),
            (shaft_end, -head_half_width),
            (half_len, 0),
            (shaft_end, head_half_width),
            (shaft_end, tail_half_width),
            (tail_start, tail_half_width),
        ]

        def orient_point(point):
            px, py = point
            if direction == Direction.EAST:
                tx, ty = px, py
            elif direction == Direction.WEST:
                tx, ty = -px, -py
            elif direction == Direction.SOUTH:
                tx, ty = -py, px
            else:
                tx, ty = py, -px
            return (int(round(center_x + tx)), int(round(center_y + ty)))

        polygon_points = [orient_point(point) for point in local_points]
        shadow_offset = (2, 2)

        arrow_alpha = max(0, min(255, int(alpha)))
        shadow_rgb = shadow_color if shadow_color is not None else (20, 90, 30)
        fill_rgb = fill_color if fill_color is not None else (65, 230, 105)
        highlight_rgb = highlight_color if highlight_color is not None else (170, 255, 180)
        shadow_rgba = (*shadow_rgb, min(150, int(arrow_alpha * 0.55)))
        arrow_fill = (*fill_rgb, arrow_alpha)
        arrow_highlight = (*highlight_rgb, min(150, int(arrow_alpha * 0.75)))

        shadow_points = [
            (point[0] + shadow_offset[0], point[1] + shadow_offset[1])
            for point in polygon_points
        ]
        pygame.draw.polygon(target_surface, shadow_rgba, shadow_points)

        pygame.draw.polygon(target_surface, arrow_fill, polygon_points)

        inner_points = [orient_point((px * 0.78, py * 0.78)) for px, py in local_points]
        pygame.draw.polygon(target_surface, arrow_highlight, inner_points)

    def draw_endpoint_marker(
        target_surface,
        tile,
        color,
        alpha,
        center_offset=None,
        scale=1.0,
        show_highlight=True,
    ):
        if alpha <= 0:
            return
        rect = get_tile_rect(tile)
        offset_x = 0 if center_offset is None else int(round(center_offset[0]))
        offset_y = 0 if center_offset is None else int(round(center_offset[1]))
        center = (rect.centerx + offset_x, rect.centery + offset_y)
        marker_scale = max(0.35, float(scale))
        radius = max(8, int(min(rect.width, rect.height) * 0.27 * marker_scale))
        marker_alpha = max(0, min(255, int(alpha)))
        background_rgba = (255, 255, 255, min(245, max(180, marker_alpha)))
        background_outline_rgba = (40, 40, 40, min(170, max(110, int(marker_alpha * 0.6))))
        outer_rgba = (*color, marker_alpha)
        mid_rgba = (*lighten_color(color, 0.45), min(255, marker_alpha))
        center_rgba = (*color, marker_alpha)
        shadow_rgba = (*darken_color(color, 0.62), min(150, int(marker_alpha * 0.55)))
        shadow_center = (center[0] + 2, center[1] + 2)
        pygame.draw.circle(target_surface, shadow_rgba, shadow_center, radius)
        pygame.draw.circle(target_surface, background_rgba, center, radius)
        pygame.draw.circle(target_surface, background_outline_rgba, center, radius, width=max(1, radius // 7))
        pygame.draw.circle(target_surface, outer_rgba, center, radius, width=max(2, radius // 3))
        if show_highlight:
            pygame.draw.circle(target_surface, mid_rgba, center, max(3, int(radius * 0.55)), width=max(2, radius // 4))
        pygame.draw.circle(target_surface, center_rgba, center, max(2, int(radius * 0.22)))

    def draw_interaction_segment(
        target_surface,
        source_tile,
        target_tile,
        color,
        alpha,
        lateral_offset_px=0.0,
        width_scale=0.92,
        arrow_scale=0.44,
        progress=1.0,
        draw_terminal_arrow=True,
    ):
        if alpha <= 0:
            return
        if not in_bounds(source_tile) or not in_bounds(target_tile):
            return
        progress = max(0.0, min(1.0, float(progress)))
        if progress <= 0.0:
            return

        direction = extract_direction(source_tile, target_tile)
        if direction is None:
            direction = vector_to_direction(
                int(target_tile[0]) - int(source_tile[0]),
                int(target_tile[1]) - int(source_tile[1]),
            )

        source_rect = get_tile_rect(source_tile)
        target_rect = get_tile_rect(target_tile)
        center_offset = compute_lateral_center_offset(direction, lateral_offset_px)
        offset_x = int(round(center_offset[0]))
        offset_y = int(round(center_offset[1]))
        start = (source_rect.centerx + offset_x, source_rect.centery + offset_y)
        target_end = (target_rect.centerx + offset_x, target_rect.centery + offset_y)
        end = (
            int(round(start[0] + ((target_end[0] - start[0]) * progress))),
            int(round(start[1] + ((target_end[1] - start[1]) * progress))),
        )
        if start == end:
            return

        line_width = max(4, int(min(tile_w, tile_h) * 0.09 * max(0.5, float(width_scale))))
        trail_alpha = max(28, min(255, int(alpha)))
        trail_fill = (*color, min(235, trail_alpha))
        trail_highlight = (*lighten_color(color, 0.32), min(165, int(trail_alpha * 0.72)))
        pygame.draw.line(target_surface, trail_fill, start, end, line_width)
        pygame.draw.line(target_surface, trail_highlight, start, end, max(2, line_width // 3))

        if direction is not None and draw_terminal_arrow and progress >= 0.999:
            draw_direction_arrow(
                target_surface,
                target_tile,
                direction,
                alpha,
                center_offset=center_offset,
                scale=arrow_scale,
                fill_color=color,
                highlight_color=lighten_color(color, 0.45),
                shadow_color=darken_color(color, 0.62),
            )

    def draw_interaction_target_tile(target_surface, tile, color, alpha):
        if alpha <= 0 or not in_bounds(tile):
            return
        rect = get_tile_rect(tile)
        inset = max(3, int(min(rect.width, rect.height) * 0.18))
        draw_rect = rect.inflate(-inset, -inset)
        border_radius = max(6, int(min(draw_rect.width, draw_rect.height) * 0.18))
        fill_rgba = (*lighten_color(color, 0.45), min(190, max(24, int(round(alpha * 0.52)))))
        shadow_rgba = (*darken_color(color, 0.58), min(90, int(max(24, int(round(alpha * 0.52))) * 0.65)))
        shadow_rect = draw_rect.move(2, 2)
        pygame.draw.rect(target_surface, shadow_rgba, shadow_rect, border_radius=border_radius)
        pygame.draw.rect(target_surface, fill_rgba, draw_rect, border_radius=border_radius)

    def infer_path_step_direction(positions, step_idx):
        current_position = positions[step_idx]
        for next_idx in range(step_idx + 1, len(positions)):
            next_position = positions[next_idx]
            direction = extract_direction(current_position, next_position)
            if direction is not None:
                return direction
        for previous_idx in range(step_idx - 1, -1, -1):
            previous_position = positions[previous_idx]
            direction = extract_direction(previous_position, current_position)
            if direction is not None:
                return direction
        return None

    def lighten_color(color, ratio=0.35):
        return tuple(min(255, int(component + ((255 - component) * ratio))) for component in color)

    def darken_color(color, ratio=0.45):
        return tuple(max(0, int(component * (1.0 - ratio))) for component in color)

    def compute_lateral_center_offset(direction, lateral_offset_px):
        if direction is None or abs(lateral_offset_px) < 1e-6:
            return (0.0, 0.0)
        dx, dy = direction
        perp_x = -float(dy)
        perp_y = float(dx)
        length = float(np.hypot(perp_x, perp_y))
        if length < 1e-6:
            return (0.0, 0.0)
        return (
            (perp_x / length) * lateral_offset_px,
            (perp_y / length) * lateral_offset_px,
        )

    def build_shifted_path_geometry(path, lateral_offset_px=0.0, actual_start_position=None):
        """Build a connected polyline for one ranked path.

        The center-ranked path uses the tile centers. Side-ranked paths leave the
        AI's actual position, follow a lane perpendicular to each movement
        segment, and use a miter join at 90-degree turns. The final arrow is
        placed on the final segment's lane, so it stays aligned with the line.
        """
        predicted_positions = []
        for step in path[:num_heatmap_tiles]:
            position = extract_position(step)
            if not in_bounds(position):
                continue
            if not predicted_positions or position != predicted_positions[-1]:
                predicted_positions.append(position)

        if actual_start_position is not None:
            origin_tile = (
                int(actual_start_position[0]),
                int(actual_start_position[1]),
            )
        elif predicted_positions:
            origin_tile = predicted_positions[0]
        else:
            return [], None

        if not in_bounds(origin_tile):
            return [], None

        def tile_center(tile):
            rect = get_tile_rect(tile)
            return float(rect.centerx), float(rect.centery)

        def add_points_if_new(points, *new_points):
            for point in new_points:
                if point is None:
                    continue
                if points:
                    last = points[-1]
                    if abs(last[0] - point[0]) < 1e-6 and abs(last[1] - point[1]) < 1e-6:
                        continue
                points.append(point)

        # Keep only the contiguous prefix. The action-rollout model should
        # always produce adjacent moves, but this prevents a bad/stale path
        # from being drawn as a disconnected jump.
        movement_positions = [origin_tile]
        for position in predicted_positions:
            previous_position = movement_positions[-1]
            if position == previous_position:
                continue
            if extract_direction(previous_position, position) is None:
                break
            movement_positions.append(position)

        origin_center = tile_center(origin_tile)
        if len(movement_positions) < 2:
            return [origin_center], None

        # Each segment owns its own perpendicular lane offset. This is crucial:
        # when the path turns, the line and its endpoint arrow must rotate into
        # the new segment's lane instead of retaining the first segment's offset.
        segments = []
        for start_tile, end_tile in zip(movement_positions[:-1], movement_positions[1:]):
            direction = extract_direction(start_tile, end_tile)
            if direction is None:
                continue
            offset = compute_lateral_center_offset(direction, lateral_offset_px)
            start_center = tile_center(start_tile)
            end_center = tile_center(end_tile)
            segments.append(
                {
                    "start_tile": start_tile,
                    "end_tile": end_tile,
                    "direction": direction,
                    "offset": offset,
                    "start_point": (start_center[0] + offset[0], start_center[1] + offset[1]),
                    "end_point": (end_center[0] + offset[0], end_center[1] + offset[1]),
                }
            )

        if not segments:
            return [origin_center], None

        def directions_are_opposites(first_direction, second_direction):
            return (
                first_direction[0] == -second_direction[0]
                and first_direction[1] == -second_direction[1]
            )

        shifted_centers = [origin_center]
        add_points_if_new(shifted_centers, segments[0]["start_point"])

        for segment_index, segment in enumerate(segments):
            if segment_index == len(segments) - 1:
                add_points_if_new(shifted_centers, segment["end_point"])
                continue

            next_segment = segments[segment_index + 1]
            node_center = tile_center(segment["end_tile"])
            current_direction = segment["direction"]
            next_direction = next_segment["direction"]

            if current_direction == next_direction:
                # Same heading: both lane segments already meet exactly.
                add_points_if_new(shifted_centers, segment["end_point"])
                continue

            if directions_are_opposites(current_direction, next_direction):
                # A 180-degree reversal has no stable miter intersection. Route
                # through the turn tile to avoid a long crossing/cusp.
                add_points_if_new(
                    shifted_centers,
                    segment["end_point"],
                    node_center,
                    next_segment["start_point"],
                )
                continue

            # 90-degree turn: intersect the two offset segment centerlines.
            # For cardinal grid motion this is node_center + current_offset +
            # next_offset. It produces a compact, connected miter instead of
            # the old artificial elbow that could misalign the arrow.
            current_offset = segment["offset"]
            next_offset = next_segment["offset"]
            miter_point = (
                node_center[0] + current_offset[0] + next_offset[0],
                node_center[1] + current_offset[1] + next_offset[1],
            )
            add_points_if_new(shifted_centers, miter_point)

        final_segment = segments[-1]
        final_arrow = (
            final_segment["end_tile"],
            final_segment["direction"],
            final_segment["offset"],
        )
        return shifted_centers, final_arrow

    def build_ranked_path_tiles(path, actual_start_position=None, include_origin=False):
        predicted_positions = []
        for step in path[:num_heatmap_tiles]:
            position = extract_position(step)
            if not in_bounds(position):
                continue
            if not predicted_positions or position != predicted_positions[-1]:
                predicted_positions.append(position)

        if actual_start_position is not None:
            origin_tile = (
                int(actual_start_position[0]),
                int(actual_start_position[1]),
            )
        elif predicted_positions:
            origin_tile = predicted_positions[0]
        else:
            return []

        if not in_bounds(origin_tile):
            return []

        movement_tiles = [origin_tile]
        for position in predicted_positions:
            previous_position = movement_tiles[-1]
            if position == previous_position:
                continue
            if extract_direction(previous_position, position) is None:
                break
            movement_tiles.append(position)

        if include_origin:
            return movement_tiles
        return movement_tiles[1:]

    def draw_ranked_path_highlight(
        target_surface,
        path,
        color,
        alpha,
        actual_start_position=None,
        animation_progress=1.0,
        max_tiles=None,
        step_number_offset=0,
        badge_registry=None,
        use_raw_step_numbers=False,
        show_badges=False,
    ):
        if alpha <= 0 or not path:
            return

        progress = max(0.0, min(1.0, float(animation_progress)))
        if progress <= 0.0:
            return

        tile_inset = max(2, int(min(tile_w, tile_h) * 0.14))
        outline_width = max(1, int(min(tile_w, tile_h) * 0.05))
        step_numbers_by_tile = defaultdict(list)
        badge_alpha_by_tile = {}
        fill_alpha_by_tile = {}
        tile_order = []

        def is_interaction_step(step):
            if isinstance(step, dict):
                return bool(step.get("interact", False))
            return len(step) >= 3 and step[2] == "interact"

        def resolve_highlight_tile(step):
            position = extract_position(step)
            if not in_bounds(position):
                return None
            if isinstance(step, dict) and step.get("interact", False):
                interact_position = tuple(step.get("interact_position", position))
                if in_bounds(interact_position):
                    return interact_position
            return position

        if use_raw_step_numbers:
            raw_step_tiles = []
            raw_badge_tiles = []
            raw_step_is_interaction = []
            for step in path[:num_heatmap_tiles]:
                highlight_tile = resolve_highlight_tile(step)
                if highlight_tile is None:
                    continue
                raw_step_tiles.append(highlight_tile)
                interaction_step = is_interaction_step(step)
                badge_tile = highlight_tile
                raw_badge_tiles.append(badge_tile)
                raw_step_is_interaction.append(interaction_step)
            if max_tiles is not None:
                raw_step_tiles = raw_step_tiles[:max_tiles]
                raw_badge_tiles = raw_badge_tiles[:max_tiles]
                raw_step_is_interaction = raw_step_is_interaction[:max_tiles]
            if not raw_step_tiles:
                return
            if progress >= 0.999:
                visible_count = len(raw_step_tiles)
            else:
                visible_count = max(1, int(np.ceil(len(raw_step_tiles) * progress)))
            visible_step_tiles = raw_step_tiles[:visible_count]
            visible_badge_tiles = raw_badge_tiles[:visible_count]
            visible_step_is_interaction = raw_step_is_interaction[:visible_count]

            for step_idx, tile in enumerate(visible_step_tiles):
                step_number = step_number_offset + step_idx + 1
                fade_ratio = step_idx / max(1, len(visible_step_tiles) - 1)
                tile_alpha = max(26, min(255, int(round(alpha * (1.0 - (0.72 * fade_ratio))))))
                badge_tile = visible_badge_tiles[step_idx]
                badge_key = (badge_tile, step_number)
                if (
                    show_badges
                    and not visible_step_is_interaction[step_idx]
                    and (badge_registry is None or badge_key not in badge_registry)
                ):
                    if badge_tile == tile:
                        step_numbers_by_tile[badge_tile].append(step_number)
                    if badge_registry is not None:
                        badge_registry.add(badge_key)
                if tile not in fill_alpha_by_tile:
                    tile_order.append(tile)
                fill_alpha_by_tile[tile] = max(fill_alpha_by_tile.get(tile, 0), tile_alpha)
                if badge_tile == tile:
                    badge_alpha_by_tile[badge_tile] = max(badge_alpha_by_tile.get(badge_tile, 0), tile_alpha)
        else:
            highlight_tiles = build_ranked_path_tiles(
                path,
                actual_start_position=actual_start_position,
                include_origin=False,
            )
            if max_tiles is not None:
                highlight_tiles = highlight_tiles[:max_tiles]
            if not highlight_tiles:
                return

            if progress >= 0.999:
                visible_count = len(highlight_tiles)
            else:
                visible_count = max(1, int(np.ceil(len(highlight_tiles) * progress)))
            visible_tiles = highlight_tiles[:visible_count]

            for tile_idx, tile in enumerate(visible_tiles):
                step_number = step_number_offset + tile_idx + 1
                fade_ratio = tile_idx / max(1, len(visible_tiles) - 1)
                tile_alpha = max(26, min(255, int(round(alpha * (1.0 - (0.72 * fade_ratio))))))
                badge_key = (tile, step_number)
                if show_badges and (badge_registry is None or badge_key not in badge_registry):
                    step_numbers_by_tile[tile].append(step_number)
                    if badge_registry is not None:
                        badge_registry.add(badge_key)
                if tile not in fill_alpha_by_tile:
                    tile_order.append(tile)
                fill_alpha_by_tile[tile] = max(fill_alpha_by_tile.get(tile, 0), tile_alpha)
                badge_alpha_by_tile[tile] = max(badge_alpha_by_tile.get(tile, 0), tile_alpha)

        for tile in tile_order:
            tile_alpha = fill_alpha_by_tile.get(tile, alpha)
            fill_alpha = min(190, max(24, int(round(tile_alpha * 0.52))))
            rect = get_tile_rect(tile).inflate(-tile_inset, -tile_inset)
            border_radius = max(6, int(min(rect.width, rect.height) * 0.16))
            shadow_rect = rect.move(2, 2)
            pygame.draw.rect(
                target_surface,
                (*darken_color(color, 0.58), min(90, int(fill_alpha * 0.65))),
                shadow_rect,
                border_radius=border_radius,
            )
            pygame.draw.rect(
                target_surface,
                (*lighten_color(color, 0.45), fill_alpha),
                rect,
                border_radius=border_radius,
            )
            pygame.draw.rect(
                target_surface,
                (*color, tile_alpha),
                rect,
                width=outline_width,
                border_radius=border_radius,
            )

        for tile, step_numbers in step_numbers_by_tile.items():
            draw_step_badges(
                target_surface,
                tile,
                step_numbers,
                badge_alpha_by_tile.get(tile, alpha),
                combine_labels=True,
            )

    def draw_ranked_path(
        target_surface,
        path,
        color,
        alpha,
        actual_start_position=None,
        lateral_offset_px=0.0,
        width_scale=1.0,
        arrow_scale=0.58,
        suppress_terminal_interaction_arrow=False,
        animation_progress=1.0,
    ):
        if alpha <= 0 or not path:
            return

        shifted_centers, final_arrow = build_shifted_path_geometry(
            path,
            lateral_offset_px=lateral_offset_px,
            actual_start_position=actual_start_position,
        )
        if not shifted_centers:
            return

        visible_centers = build_partial_polyline(shifted_centers, animation_progress)
        if len(visible_centers) < 2:
            return

        line_width = max(4, int(min(tile_w, tile_h) * 0.10 * max(0.5, float(width_scale))))
        trail_alpha = max(28, min(255, int(alpha)))
        trail_fill = (*color, min(235, trail_alpha))
        trail_highlight = (*lighten_color(color, 0.32), min(165, int(trail_alpha * 0.72)))

        current_point = (
            int(round(visible_centers[0][0])),
            int(round(visible_centers[0][1])),
        )
        for next_point_raw in visible_centers[1:]:
            next_point = (
                int(round(next_point_raw[0])),
                int(round(next_point_raw[1])),
            )
            if current_point == next_point:
                continue
            pygame.draw.line(target_surface, trail_fill, current_point, next_point, line_width)
            pygame.draw.line(target_surface, trail_highlight, current_point, next_point, max(2, line_width // 3))
            current_point = next_point

        terminal_visible_step = None
        for step in path[:num_heatmap_tiles]:
            position = extract_position(step)
            if in_bounds(position):
                terminal_visible_step = step

        terminal_step_is_interaction = bool(
            isinstance(terminal_visible_step, dict)
            and terminal_visible_step.get("interact", False)
        )

        if final_arrow is not None and animation_progress >= 0.999:
            if suppress_terminal_interaction_arrow and terminal_step_is_interaction:
                return
            final_tile, final_direction, center_offset = final_arrow
            draw_direction_arrow(
                target_surface,
                final_tile,
                final_direction,
                alpha,
                center_offset=center_offset,
                scale=arrow_scale,
                fill_color=color,
                highlight_color=lighten_color(color, 0.45),
                shadow_color=darken_color(color, 0.62),
            )

    def resolve_ranked_path_endpoint(
        path,
        actual_start_position=None,
        actual_start_orientation=None,
        lateral_offset_px=0.0,
    ):
        visible_steps = []
        for step in path[:num_heatmap_tiles]:
            position = extract_position(step)
            if not in_bounds(position):
                continue
            visible_steps.append(step)

        if not visible_steps:
            return None

        last_step = visible_steps[-1]
        if isinstance(last_step, dict) and last_step.get("interact", False):
            source_tile = tuple(last_step["position"])
            target_tile = tuple(last_step.get("interact_position", source_tile))
            if not in_bounds(target_tile):
                return None
            direction = extract_direction(source_tile, target_tile)
            if direction is None:
                direction = actual_start_orientation
            center_offset = compute_lateral_center_offset(direction, lateral_offset_px)
            return target_tile, direction, center_offset

        shifted_centers, final_arrow = build_shifted_path_geometry(
            path,
            lateral_offset_px=lateral_offset_px,
            actual_start_position=actual_start_position,
        )
        if final_arrow is not None:
            return final_arrow

        endpoint_tile = extract_position(last_step)
        if not in_bounds(endpoint_tile):
            return None
        direction = actual_start_orientation
        center_offset = compute_lateral_center_offset(direction, lateral_offset_px)
        return endpoint_tile, direction, center_offset

    def draw_ranked_path_endpoint(
        target_surface,
        path,
        color,
        alpha,
        actual_start_position=None,
        actual_start_orientation=None,
        lateral_offset_px=0.0,
        arrow_scale=0.58,
        show_highlight=True,
    ):
        if alpha <= 0 or not path:
            return

        endpoint = resolve_ranked_path_endpoint(
            path,
            actual_start_position=actual_start_position,
            actual_start_orientation=actual_start_orientation,
            lateral_offset_px=lateral_offset_px,
        )
        if endpoint is None:
            return

        endpoint_tile, _endpoint_direction, center_offset = endpoint
        draw_endpoint_marker(
            target_surface,
            endpoint_tile,
            color,
            alpha,
            center_offset=center_offset,
            scale=arrow_scale,
            show_highlight=show_highlight,
        )

    def resolve_ranked_path_first_segment(
        path,
        actual_start_position=None,
        actual_start_orientation=None,
        lateral_offset_px=0.0,
    ):
        visible_steps = []
        for step in path[:num_heatmap_tiles]:
            position = extract_position(step)
            if not in_bounds(position):
                continue
            visible_steps.append(step)

        if not visible_steps:
            return None

        if actual_start_position is not None:
            current_tile = (
                int(actual_start_position[0]),
                int(actual_start_position[1]),
            )
        else:
            current_tile = extract_position(visible_steps[0])
        current_orientation = actual_start_orientation

        for step in visible_steps:
            step_tile = extract_position(step)

            if step_tile != current_tile:
                direction = extract_direction(current_tile, step_tile)
                if direction is None:
                    return None
                return current_tile, step_tile, direction

            if isinstance(step, dict) and step.get("interact", False):
                interact_tile = tuple(step.get("interact_position", step_tile))
                if not in_bounds(interact_tile):
                    return None
                direction = extract_direction(step_tile, interact_tile)
                if direction is None:
                    direction = current_orientation
                if direction is None:
                    return None
                return step_tile, interact_tile, direction

            current_tile = step_tile

        return None

    def draw_ranked_path_first_segment(
        target_surface,
        path,
        color,
        alpha,
        actual_start_position=None,
        actual_start_orientation=None,
        lateral_offset_px=0.0,
        width_scale=1.0,
        arrow_scale=0.58,
        animation_progress=1.0,
    ):
        if alpha <= 0 or not path:
            return

        first_segment = resolve_ranked_path_first_segment(
            path,
            actual_start_position=actual_start_position,
            actual_start_orientation=actual_start_orientation,
            lateral_offset_px=lateral_offset_px,
        )
        if first_segment is None:
            return

        source_tile, target_tile, _direction = first_segment
        draw_interaction_segment(
            target_surface,
            source_tile,
            target_tile,
            color,
            alpha,
            lateral_offset_px=lateral_offset_px,
            width_scale=width_scale,
            arrow_scale=arrow_scale,
            progress=animation_progress,
        )

    def draw_step_badges(target_surface, tile, step_numbers, alpha, combine_labels=False):
        if not step_numbers or alpha <= 0:
            return

        rect = get_tile_rect(tile)
        badge_alpha = max(180, min(255, int(alpha)))
        step_numbers = sorted(dict.fromkeys(step_numbers))
        if combine_labels and len(step_numbers) > 1:
            font_size = max(12, int(min(rect.width, rect.height) * 0.24))
            font = badge_fonts.get(("combined", font_size))
            if font is None:
                if not pygame.font.get_init():
                    pygame.font.init()
                font = pygame.font.Font(None, font_size)
                badge_fonts[("combined", font_size)] = font

            label_text = ",".join(str(step_number) for step_number in step_numbers)
            label_surface = font.render(label_text, True, (255, 255, 255))
            padding_x = max(6, int(rect.width * 0.06))
            padding_y = max(3, int(rect.height * 0.04))
            badge_width = min(rect.width - 6, label_surface.get_width() + padding_x * 2)
            badge_height = min(
                max(label_surface.get_height() + padding_y * 2, int(rect.height * 0.22)),
                rect.height - 6,
            )
            badge_surface = pygame.Surface((badge_width, badge_height), pygame.SRCALPHA)
            badge_rect = badge_surface.get_rect()
            pygame.draw.rect(
                badge_surface,
                (20, 20, 20, badge_alpha),
                badge_rect,
                border_radius=max(6, badge_height // 2),
            )
            pygame.draw.rect(
                badge_surface,
                (255, 255, 255, min(255, badge_alpha)),
                badge_rect,
                width=1,
                border_radius=max(6, badge_height // 2),
            )
            label_rect = label_surface.get_rect(center=badge_rect.center)
            badge_surface.blit(label_surface, label_rect)
            target_surface.blit(
                badge_surface,
                badge_surface.get_rect(topright=(rect.right - 3, rect.top + 3)),
            )
            return

        num_badges = len(step_numbers)
        columns = min(3, num_badges)
        rows = (num_badges + columns - 1) // columns
        spacing = 2
        badge_diameter = min(
            max(10, int((rect.width - 6 - spacing * (columns - 1)) / columns)),
            max(10, int((rect.height - 6 - spacing * (rows - 1)) / rows)),
            max(12, int(min(rect.width, rect.height) * 0.26)),
        )
        badge_radius = badge_diameter // 2
        font_size = max(10, int(badge_diameter * 0.9))
        font = badge_fonts.get(font_size)
        if font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.Font(None, font_size)
            badge_fonts[font_size] = font

        total_width = (columns * badge_diameter) + ((columns - 1) * spacing)
        origin_x = rect.right - total_width - 3
        origin_y = rect.top + 3

        for idx, step_number in enumerate(step_numbers):
            row = idx // columns
            col = idx % columns
            badge_surface = pygame.Surface((badge_diameter + 4, badge_diameter + 4), pygame.SRCALPHA)
            badge_rect = badge_surface.get_rect()
            local_center = badge_rect.center
            pygame.draw.circle(badge_surface, (20, 20, 20, badge_alpha), local_center, badge_radius)
            pygame.draw.circle(
                badge_surface,
                (255, 255, 255, min(255, badge_alpha)),
                local_center,
                badge_radius,
                width=1,
            )

            label_surface = font.render(str(step_number), True, (255, 255, 255))
            label_rect = label_surface.get_rect(center=local_center)
            badge_surface.blit(label_surface, label_rect)
            badge_center = (
                origin_x + (col * (badge_diameter + spacing)) + badge_diameter // 2,
                origin_y + (row * (badge_diameter + spacing)) + badge_diameter // 2,
            )
            target_surface.blit(badge_surface, badge_surface.get_rect(center=badge_center))

    def infer_representative_direction(representative_positions, step_idx, start_position, start_orientation):
        tile = (
            int(representative_positions[step_idx][0]),
            int(representative_positions[step_idx][1]),
        )
        direction = None
        if step_idx + 1 < len(representative_positions):
            next_position = representative_positions[step_idx + 1]
            direction = vector_to_direction(
                float(next_position[0]) - float(tile[0]),
                float(next_position[1]) - float(tile[1]),
            )
        if direction is None and step_idx > 0:
            previous_position = representative_positions[step_idx - 1]
            direction = vector_to_direction(
                float(tile[0]) - float(previous_position[0]),
                float(tile[1]) - float(previous_position[1]),
            )
        if direction is None and start_position is not None:
            direction = vector_to_direction(
                float(tile[0]) - float(start_position[0]),
                float(tile[1]) - float(start_position[1]),
                fallback=start_orientation,
            )
        if direction is None:
            direction = start_orientation
        return direction

    def draw_model_probabilities(
        position_probabilities,
        valid_positions,
        move_color,
        interact_color,
        interact_probabilities=None,
        start_position=None,
        start_orientation=None,
        fill_move_tiles=True,
        show_step_badges=True,
    ):
        if position_probabilities is None or valid_positions is None:
            return

        position_probabilities = np.asarray(position_probabilities, dtype=np.float32)
        valid_positions = np.asarray(valid_positions, dtype=np.int64)
        if position_probabilities.ndim != 2 or len(valid_positions) != position_probabilities.shape[1]:
            return

        representative_indices = np.argmax(position_probabilities, axis=1)
        representative_positions = valid_positions[representative_indices]
        representative_probabilities = position_probabilities[
            np.arange(len(representative_indices)),
            representative_indices,
        ]
        num_steps_to_draw = min(num_heatmap_tiles, len(representative_positions))

        if PATH_RENDER_STYLE == PATH_RENDER_STYLE_HIGHLIGHT:
            representative_path = []
            previous_position = (
                tuple(start_position)
                if start_position is not None
                else tuple(representative_positions[0])
            )
            previous_orientation = start_orientation or Direction.SOUTH
            for step_idx, position in enumerate(representative_positions[:num_steps_to_draw]):
                current_position = (int(position[0]), int(position[1]))
                current_orientation = infer_orientation_from_positions(
                    previous_position,
                    current_position,
                    previous_orientation,
                )
                interact = (
                    interact_probabilities is not None
                    and step_idx < len(interact_probabilities)
                    and float(interact_probabilities[step_idx]) >= 0.5
                )
                step = {"position": current_position, "interact": interact}
                if interact:
                    step["interact_position"] = tuple(
                        Action.move_in_direction(current_position, current_orientation)
                    )
                representative_path.append(step)
                previous_position = current_position
                previous_orientation = current_orientation

            draw_ranked_path_highlight(
                overlay,
                representative_path,
                color=move_color,
                alpha=250,
                actual_start_position=start_position,
                animation_progress=1.0,
                use_raw_step_numbers=True,
                show_badges=False,
            )
            return

        tile_summaries = {}

        for step_idx in range(num_steps_to_draw):
            tile = (int(representative_positions[step_idx][0]), int(representative_positions[step_idx][1]))
            if not in_bounds(tile):
                continue

            probability = float(representative_probabilities[step_idx])
            alpha = max(0, min(255, int(180 * probability)))
            direction = infer_representative_direction(
                representative_positions,
                step_idx,
                start_position,
                start_orientation,
            )

            summary = tile_summaries.get(tile)
            if summary is None:
                tile_summaries[tile] = {
                    "alpha": alpha,
                    "earliest_step": step_idx,
                    "direction": direction,
                    "step_numbers": [step_idx + 1],
                }
                continue

            summary["alpha"] = max(summary["alpha"], alpha)
            summary["step_numbers"].append(step_idx + 1)
            if step_idx < summary["earliest_step"]:
                summary["earliest_step"] = step_idx
                summary["direction"] = direction

        for tile, summary in sorted(tile_summaries.items(), key=lambda item: item[1]["earliest_step"]):
            alpha = summary["alpha"]
            if alpha <= 0:
                continue
            if fill_move_tiles:
                rect = get_tile_rect(tile)
                pygame.draw.rect(overlay, (*move_color, alpha), rect)
            if summary["direction"] is not None:
                draw_direction_arrow(overlay, tile, summary["direction"], alpha)
            if show_step_badges:
                draw_step_badges(overlay, tile, summary["step_numbers"], alpha)

        if not show_interaction_heatmap or interact_probabilities is None:
            return

        interact_probabilities = np.asarray(interact_probabilities, dtype=np.float32)
        representative_orientations = []
        previous_position = tuple(start_position) if start_position is not None else tuple(representative_positions[0])
        previous_orientation = start_orientation if start_orientation is not None else Direction.SOUTH
        for position in representative_positions:
            current_position = (int(position[0]), int(position[1]))
            current_orientation = infer_orientation_from_positions(
                previous_position,
                current_position,
                previous_orientation,
            )
            representative_orientations.append(current_orientation)
            previous_position = current_position
            previous_orientation = current_orientation

        num_interaction_steps = min(num_heatmap_interactions, len(representative_positions), len(interact_probabilities))
        previous_interaction_signature = None
        for horizon_idx in range(num_interaction_steps):
            source_tile = (
                int(representative_positions[horizon_idx][0]),
                int(representative_positions[horizon_idx][1]),
            )
            tile = tuple(
                Action.move_in_direction(
                    source_tile,
                    representative_orientations[horizon_idx],
                )
            )
            if not in_bounds(tile):
                continue
            probability = float(interact_probabilities[horizon_idx])
            alpha = int(255 * probability)
            alpha = max(0, min(255, alpha))
            if alpha <= 0:
                continue
            interaction_signature = (source_tile, tile)
            if interaction_signature == previous_interaction_signature:
                continue
            previous_interaction_signature = interaction_signature
            draw_interaction_segment(overlay, source_tile, tile, interact_color, alpha)

    def draw_paths(
        paths,
        move_color,
        interact_color,
        path_weights=None,
        interact_probabilities=None,
        fill_move_tiles=True,
        show_step_badges=True,
        greedy_path_count=0,
    ):
        tile_counts = defaultdict(float)
        interact_counts = defaultdict(float)
        earliest_step_by_tile = {}
        outgoing_direction_counts = {}
        incoming_direction_counts = {}
        step_numbers_by_tile = {}
        if path_weights is None or len(path_weights) != len(paths):
            normalized_path_weights = [1.0 / max(1, len(paths))] * len(paths)
        else:
            total_weight = float(sum(max(0.0, float(weight)) for weight in path_weights))
            if total_weight <= 0:
                normalized_path_weights = [1.0 / max(1, len(paths))] * len(paths)
            else:
                normalized_path_weights = [max(0.0, float(weight)) / total_weight for weight in path_weights]
        total_path_weight = max(1e-8, sum(normalized_path_weights))

        def normalized_path_to_steps(normalized_path):
            steps = []
            for position, interact_flag, interact_position in normalized_path:
                if interact_flag:
                    steps.append(
                        {
                            "position": tuple(position),
                            "interact": True,
                            "interact_position": tuple(interact_position if interact_position is not None else position),
                        }
                    )
                else:
                    steps.append(
                        {
                            "position": tuple(position),
                            "interact": False,
                        }
                    )
            return steps

        def common_prefix_length(first_path, second_path):
            max_len = min(len(first_path), len(second_path))
            prefix_len = 0
            while prefix_len < max_len and first_path[prefix_len] == second_path[prefix_len]:
                prefix_len += 1
            return prefix_len

        def select_ranked_normalized_entries(raw_paths, raw_weights, greedy_count):
            include_interaction_signatures = (
                show_interaction_heatmap
                or PATH_VISUALIZATION_MODE != PATH_VISUALIZATION_FULL
            )
            max_ranked_paths = max(0, min(2, int(NUM_PATHS_TO_SHOW)))
            return select_shared_ranked_normalized_entries(
                raw_paths,
                raw_weights,
                greedy_count,
                include_interactions=include_interaction_signatures,
                max_ranked_paths=max_ranked_paths,
                max_steps=num_heatmap_tiles,
                stop_at_interaction=STOP_PATH_AT_INTERACTION,
            )

        def find_best_previous_prefix(path_to_match, candidate_entries):
            best_prefix_len = 0
            best_entry = None
            for candidate_entry in candidate_entries:
                candidate_path = candidate_entry["normalized_path"]
                prefix_len = common_prefix_length(path_to_match, candidate_path)
                if prefix_len > best_prefix_len:
                    best_prefix_len = prefix_len
                    best_entry = candidate_entry
            return best_entry, best_prefix_len

        def draw_interactions_for_normalized_path(
            normalized_path,
            color,
            alpha,
            lateral_offset_px,
            width_scale,
            progress,
            badge_registry=None,
        ):
            if alpha <= 0 or not normalized_path:
                return
            if PATH_RENDER_STYLE == PATH_RENDER_STYLE_HIGHLIGHT:
                return
            previous_interaction_signature = None
            for step_idx, (position, interact_flag, interact_position) in enumerate(normalized_path[:num_heatmap_interactions]):
                if not interact_flag or interact_position is None:
                    continue
                source_tile = tuple(position)
                target_tile = tuple(interact_position)
                if not in_bounds(target_tile):
                    continue
                interaction_signature = (source_tile, target_tile)
                if interaction_signature == previous_interaction_signature:
                    continue
                previous_interaction_signature = interaction_signature
                draw_interaction_segment(
                    overlay,
                    source_tile,
                    target_tile,
                    color,
                    alpha,
                    lateral_offset_px=lateral_offset_px,
                    width_scale=width_scale,
                    progress=progress,
                )
                draw_interaction_target_tile(overlay, target_tile, color, alpha)

        def record_direction(tile, step_idx, outgoing_direction, incoming_direction, weight):
            previous_earliest = earliest_step_by_tile.get(tile)
            if previous_earliest is None or step_idx < previous_earliest:
                earliest_step_by_tile[tile] = step_idx
                outgoing_direction_counts[tile] = defaultdict(float)
                incoming_direction_counts[tile] = defaultdict(float)
                step_numbers_by_tile[tile] = []
            if earliest_step_by_tile[tile] != step_idx:
                step_numbers_by_tile.setdefault(tile, []).append(step_idx + 1)
                return
            step_numbers_by_tile.setdefault(tile, []).append(step_idx + 1)
            if outgoing_direction is not None:
                outgoing_direction_counts[tile][outgoing_direction] += weight
            if incoming_direction is not None:
                incoming_direction_counts[tile][incoming_direction] += weight

        for path_index, path in enumerate(paths):
            path_weight = normalized_path_weights[path_index]
            visited_tiles = set()
            interacted_tiles = set()
            positions = [extract_position(step) for step in path]

            for idx, step in enumerate(path):
                position = positions[idx]
                if isinstance(step, dict):
                    if in_bounds(position):
                        visited_tiles.add(position)
                    if step.get("interact", False):
                        interact_position = tuple(step.get("interact_position", position))
                        if in_bounds(interact_position):
                            interacted_tiles.add(interact_position)
                elif len(step) == 3 and step[2] == "interact":
                    interact_position = (step[0], step[1])
                    if in_bounds(interact_position):
                        interacted_tiles.add(interact_position)
                else:
                    if in_bounds(position):
                        visited_tiles.add(position)

                if idx + 1 >= len(positions):
                    next_position = None
                    outgoing_direction = None
                else:
                    next_position = positions[idx + 1]
                    if in_bounds(position) and in_bounds(next_position):
                        outgoing_direction = extract_direction(position, next_position)
                    else:
                        outgoing_direction = None

                if idx == 0:
                    incoming_direction = None
                else:
                    previous_position = positions[idx - 1]
                    if in_bounds(previous_position) and in_bounds(position):
                        incoming_direction = extract_direction(previous_position, position)
                    else:
                        incoming_direction = None

                if in_bounds(position):
                    record_direction(position, idx, outgoing_direction, incoming_direction, path_weight)
           
            for tile in visited_tiles:
                tile_counts[tile] += path_weight
            for tile in interacted_tiles:
                interact_counts[tile] += path_weight

        arrow_only_mode = not fill_move_tiles and not show_step_badges
        if arrow_only_mode:
            if not paths:
                return

            ranked_paths = select_ranked_normalized_entries(paths, path_weights, greedy_path_count)
            previous_ranked_paths = []
            previous_ai_start_position = None
            previous_ai_start_orientation = None
            if previous_rollout_results is not None:
                previous_ranked_paths = select_ranked_normalized_entries(
                    previous_rollout_results.get("ai_paths", []),
                    previous_rollout_results.get("ai_path_weights"),
                    int(previous_rollout_results.get("ai_greedy_path_count", 0)),
                )
                previous_ai_start_position = previous_rollout_results.get("ai_start_position")
                previous_ai_start_orientation = previous_rollout_results.get("ai_start_orientation")

            lane_unit = min(tile_w, tile_h)
            ranked_styles = [
                {
                    "color": (242, 51, 156),
                    "alpha": 250,
                    "width_scale": 1.12,
                    "lateral_offset_px": 0.0,
                    "arrow_scale": 0.64,
                },
                {
                    "color": (245, 128, 36),
                    "alpha": 250,
                    "width_scale": 1.00,
                    "lateral_offset_px": -0.18 * lane_unit,
                    "arrow_scale": 0.58,
                },
                {
                    "color": (248, 176, 46),
                    "alpha": 250,
                    "width_scale": 0.92,
                    "lateral_offset_px": 0.18 * lane_unit,
                    "arrow_scale": 0.54,
                },
                {
                    "color": (250, 225, 82),
                    "alpha": 202,
                    "width_scale": 0.86,
                    "lateral_offset_px": -0.34 * lane_unit,
                    "arrow_scale": 0.50,
                },
            ]

            if PATH_VISUALIZATION_MODE == PATH_VISUALIZATION_ENDPOINT:
                top_endpoint_tile = None
                previous_top_endpoint_tile = None
                drawn_endpoint_keys = set()
                current_endpoint_keys = set()
                previous_endpoint_keys = set()

                for rank_idx, entry in enumerate(previous_ranked_paths):
                    style = ranked_styles[min(rank_idx, len(ranked_styles) - 1)]
                    previous_path = normalized_path_to_steps(entry["normalized_path"])
                    endpoint = resolve_ranked_path_endpoint(
                        previous_path,
                        actual_start_position=previous_ai_start_position,
                        actual_start_orientation=previous_ai_start_orientation,
                        lateral_offset_px=style["lateral_offset_px"],
                    )
                    if endpoint is None:
                        continue
                    endpoint_tile, _endpoint_direction, center_offset = endpoint
                    if rank_idx == 0:
                        previous_top_endpoint_tile = endpoint_tile
                    elif rank_idx == 1 and previous_top_endpoint_tile is not None and endpoint_tile == previous_top_endpoint_tile:
                        continue
                    previous_endpoint_keys.add(
                        (
                            endpoint_tile,
                            int(round(center_offset[0])),
                            int(round(center_offset[1])),
                        )
                    )

                for rank_idx, entry in enumerate(ranked_paths):
                    style = ranked_styles[min(rank_idx, len(ranked_styles) - 1)]
                    path = normalized_path_to_steps(entry["normalized_path"])
                    endpoint = resolve_ranked_path_endpoint(
                        path,
                        actual_start_position=ai_start_position,
                        actual_start_orientation=ai_start_orientation,
                        lateral_offset_px=style["lateral_offset_px"],
                    )
                    if endpoint is None:
                        continue
                    endpoint_tile, _endpoint_direction, center_offset = endpoint
                    if rank_idx == 0:
                        top_endpoint_tile = endpoint_tile
                    elif rank_idx == 1 and top_endpoint_tile is not None and endpoint_tile == top_endpoint_tile:
                        continue
                    endpoint_key = (
                        endpoint_tile,
                        int(round(center_offset[0])),
                        int(round(center_offset[1])),
                    )
                    current_endpoint_keys.add(endpoint_key)
                    show_highlight = endpoint_key not in drawn_endpoint_keys
                    drawn_endpoint_keys.add(endpoint_key)
                    marker_alpha = style["alpha"] if endpoint_key in previous_endpoint_keys else int(round(style["alpha"] * path_animation_progress))
                    draw_endpoint_marker(
                        overlay,
                        endpoint_tile,
                        style["color"],
                        marker_alpha,
                        center_offset=center_offset,
                        scale=style["arrow_scale"],
                        show_highlight=show_highlight,
                    )

                previous_top_endpoint_tile = None
                for rank_idx, entry in enumerate(previous_ranked_paths):
                    style = ranked_styles[min(rank_idx, len(ranked_styles) - 1)]
                    previous_path = normalized_path_to_steps(entry["normalized_path"])
                    endpoint = resolve_ranked_path_endpoint(
                        previous_path,
                        actual_start_position=previous_ai_start_position,
                        actual_start_orientation=previous_ai_start_orientation,
                        lateral_offset_px=style["lateral_offset_px"],
                    )
                    if endpoint is None:
                        continue
                    endpoint_tile, _endpoint_direction, center_offset = endpoint
                    if rank_idx == 0:
                        previous_top_endpoint_tile = endpoint_tile
                    elif rank_idx == 1 and previous_top_endpoint_tile is not None and endpoint_tile == previous_top_endpoint_tile:
                        continue
                    endpoint_key = (
                        endpoint_tile,
                        int(round(center_offset[0])),
                        int(round(center_offset[1])),
                    )
                    if endpoint_key in current_endpoint_keys:
                        continue
                    draw_endpoint_marker(
                        overlay,
                        endpoint_tile,
                        style["color"],
                        int(round(style["alpha"] * (1.0 - path_animation_progress))),
                        center_offset=center_offset,
                        scale=style["arrow_scale"],
                        show_highlight=False,
                    )
                return

            if PATH_VISUALIZATION_MODE == PATH_VISUALIZATION_ENDPOINT_PLUS_FIRST:
                drawn_endpoint_keys = set()
                highlight_badge_registry = set()
                for rank_idx, entry in enumerate(ranked_paths):
                    style = ranked_styles[min(rank_idx, len(ranked_styles) - 1)]
                    path = normalized_path_to_steps(entry["normalized_path"])
                    endpoint = resolve_ranked_path_endpoint(
                        path,
                        actual_start_position=ai_start_position,
                        actual_start_orientation=ai_start_orientation,
                        lateral_offset_px=style["lateral_offset_px"],
                    )
                    show_highlight = True
                    if endpoint is not None:
                        endpoint_tile, _endpoint_direction, center_offset = endpoint
                        endpoint_key = (
                            endpoint_tile,
                            int(round(center_offset[0])),
                            int(round(center_offset[1])),
                        )
                        show_highlight = endpoint_key not in drawn_endpoint_keys
                        drawn_endpoint_keys.add(endpoint_key)
                    if PATH_RENDER_STYLE == PATH_RENDER_STYLE_HIGHLIGHT:
                        draw_ranked_path_highlight(
                            overlay,
                            path,
                            color=style["color"],
                            alpha=style["alpha"],
                            actual_start_position=ai_start_position,
                            animation_progress=path_animation_progress,
                            max_tiles=1,
                            badge_registry=highlight_badge_registry,
                            use_raw_step_numbers=True,
                        )
                    else:
                        draw_ranked_path_first_segment(
                            overlay,
                            path,
                            color=style["color"],
                            alpha=style["alpha"],
                            actual_start_position=ai_start_position,
                            actual_start_orientation=ai_start_orientation,
                            lateral_offset_px=style["lateral_offset_px"],
                            width_scale=style["width_scale"],
                            arrow_scale=style["arrow_scale"],
                            animation_progress=path_animation_progress,
                        )
                    draw_ranked_path_endpoint(
                        overlay,
                        path,
                        color=style["color"],
                        alpha=style["alpha"],
                        actual_start_position=ai_start_position,
                        actual_start_orientation=ai_start_orientation,
                        lateral_offset_px=style["lateral_offset_px"],
                        arrow_scale=style["arrow_scale"],
                        show_highlight=show_highlight,
                    )
                return
            if PATH_VISUALIZATION_MODE == PATH_VISUALIZATION_NONE:
                return

            highlight_badge_registry = set()
            for rank_idx, entry in enumerate(ranked_paths):
                style = ranked_styles[min(rank_idx, len(ranked_styles) - 1)]
                current_normalized_path = entry["normalized_path"]
                _best_previous_entry, common_len = find_best_previous_prefix(current_normalized_path, previous_ranked_paths)
                common_normalized_path = current_normalized_path[:common_len]
                added_normalized_path = current_normalized_path[common_len:]

                if common_normalized_path:
                    if PATH_RENDER_STYLE == PATH_RENDER_STYLE_HIGHLIGHT:
                        draw_ranked_path_highlight(
                            overlay,
                            normalized_path_to_steps(common_normalized_path),
                            color=style["color"],
                            alpha=style["alpha"],
                            actual_start_position=ai_start_position,
                            animation_progress=1.0,
                            step_number_offset=0,
                            badge_registry=highlight_badge_registry,
                            use_raw_step_numbers=True,
                        )
                    else:
                        draw_ranked_path(
                            overlay,
                            normalized_path_to_steps(common_normalized_path),
                            color=style["color"],
                            alpha=style["alpha"],
                            actual_start_position=ai_start_position,
                            lateral_offset_px=style["lateral_offset_px"],
                            width_scale=style["width_scale"],
                            arrow_scale=style["arrow_scale"],
                            suppress_terminal_interaction_arrow=(bool(added_normalized_path) or show_interaction_heatmap),
                            animation_progress=1.0,
                        )
                    if show_interaction_heatmap:
                        draw_interactions_for_normalized_path(
                            common_normalized_path,
                            style["color"],
                            style["alpha"],
                            style["lateral_offset_px"],
                            style["width_scale"],
                            1.0,
                            badge_registry=highlight_badge_registry,
                        )

                if added_normalized_path:
                    added_start_position = ai_start_position if common_len == 0 else common_normalized_path[-1][0]
                    if PATH_RENDER_STYLE == PATH_RENDER_STYLE_HIGHLIGHT:
                        draw_ranked_path_highlight(
                            overlay,
                            normalized_path_to_steps(added_normalized_path),
                            color=style["color"],
                            alpha=style["alpha"],
                            actual_start_position=added_start_position,
                            animation_progress=path_animation_progress,
                            step_number_offset=common_len,
                            badge_registry=highlight_badge_registry,
                            use_raw_step_numbers=True,
                        )
                    else:
                        draw_ranked_path(
                            overlay,
                            normalized_path_to_steps(added_normalized_path),
                            color=style["color"],
                            alpha=style["alpha"],
                            actual_start_position=added_start_position,
                            lateral_offset_px=style["lateral_offset_px"],
                            width_scale=style["width_scale"],
                            arrow_scale=style["arrow_scale"],
                            suppress_terminal_interaction_arrow=show_interaction_heatmap,
                            animation_progress=path_animation_progress,
                        )
                    if show_interaction_heatmap:
                        draw_interactions_for_normalized_path(
                            added_normalized_path,
                            style["color"],
                            style["alpha"],
                            style["lateral_offset_px"],
                            style["width_scale"],
                            path_animation_progress,
                            badge_registry=highlight_badge_registry,
                        )

            for rank_idx, entry in enumerate(previous_ranked_paths):
                style = ranked_styles[min(rank_idx, len(ranked_styles) - 1)]
                previous_normalized_path = entry["normalized_path"]
                _best_current_entry, common_len = find_best_previous_prefix(previous_normalized_path, ranked_paths)
                removed_normalized_path = previous_normalized_path[common_len:]
                if not removed_normalized_path:
                    continue
                removed_alpha = int(round(style["alpha"] * (1.0 - path_animation_progress)))
                if removed_alpha <= 0:
                    continue
                removed_start_position = previous_ai_start_position if common_len == 0 else previous_normalized_path[common_len - 1][0]
                if PATH_RENDER_STYLE == PATH_RENDER_STYLE_HIGHLIGHT:
                    draw_ranked_path_highlight(
                        overlay,
                        normalized_path_to_steps(removed_normalized_path),
                        color=style["color"],
                        alpha=removed_alpha,
                        actual_start_position=removed_start_position,
                        animation_progress=1.0,
                        step_number_offset=common_len,
                        badge_registry=highlight_badge_registry,
                        use_raw_step_numbers=True,
                    )
                else:
                    draw_ranked_path(
                        overlay,
                        normalized_path_to_steps(removed_normalized_path),
                        color=style["color"],
                        alpha=removed_alpha,
                        actual_start_position=removed_start_position,
                        lateral_offset_px=style["lateral_offset_px"],
                        width_scale=style["width_scale"],
                        arrow_scale=style["arrow_scale"],
                        suppress_terminal_interaction_arrow=show_interaction_heatmap,
                        animation_progress=1.0,
                    )
                if show_interaction_heatmap:
                    draw_interactions_for_normalized_path(
                        removed_normalized_path,
                        style["color"],
                        removed_alpha,
                        style["lateral_offset_px"],
                        style["width_scale"],
                        1.0,
                        badge_registry=highlight_badge_registry,
                    )
            return

        top_move_tiles = sorted(tile_counts.items(), key=lambda item: item[1], reverse=True)[:num_heatmap_tiles]
        top_interact_tiles = sorted(interact_counts.items(), key=lambda item: item[1], reverse=True)[:num_heatmap_interactions]

        for (x, y), count in top_move_tiles:
            prob = count / total_path_weight
            alpha = int(180 * prob)
            alpha = max(0, min(255, alpha))
            if alpha < min_visible_tile_alpha:
                continue
            if fill_move_tiles:
                rect = get_tile_rect((x, y))
                pygame.draw.rect(overlay, (*move_color, alpha), rect)
            direction_votes = outgoing_direction_counts.get((x, y), {})
            if not direction_votes:
                direction_votes = incoming_direction_counts.get((x, y), {})
            if direction_votes:
                direction = max(direction_votes.items(), key=lambda item: item[1])[0]
                draw_direction_arrow(overlay, (x, y), direction, alpha)
            if show_step_badges:
                draw_step_badges(overlay, (x, y), step_numbers_by_tile.get((x, y), []), alpha)

        if show_interaction_heatmap:
            for (x, y), count in top_interact_tiles:
                prob = count / total_paths
                alpha = int(255 * prob)
                alpha = max(0, min(255, alpha))
                rect = get_tile_rect((x, y))
                px = rect.centerx
                py = rect.centery
                pygame.draw.circle(overlay, (*interact_color, alpha), (px, py), 15, width=4)

    if ai_position_probabilities is not None and ai_valid_positions is not None:
        draw_model_probabilities(
            ai_position_probabilities,
            ai_valid_positions,
            move_color=(242, 51, 156),
            interact_color=(255, 200, 0),
            interact_probabilities=ai_interact_probabilities,
            start_position=ai_start_position,
            start_orientation=ai_start_orientation,
            fill_move_tiles=False,
            show_step_badges=False,
        )
    elif ai_paths:
        draw_paths(
            ai_paths,
            move_color=(50, 150, 255),
            interact_color=(255, 200, 0),
            path_weights=ai_path_weights,
            interact_probabilities=ai_interact_probabilities,
            fill_move_tiles=False,
            show_step_badges=False,
            greedy_path_count=ai_greedy_path_count,
        )

    surface.blit(overlay, (0, 0))


def draw_tobii_gaze_overlay(surface, surface_point):
    if surface_point is None:
        return

    gaze_x = int(round(surface_point[0]))
    gaze_y = int(round(surface_point[1]))

    if not (0 <= gaze_x < surface.get_width() and 0 <= gaze_y < surface.get_height()):
        return

    pygame.draw.circle(surface, (255, 255, 255), (gaze_x, gaze_y), 16, width=3)
    pygame.draw.circle(surface, (255, 60, 60), (gaze_x, gaze_y), 6)
    pygame.draw.line(surface, (255, 255, 255), (gaze_x - 10, gaze_y), (gaze_x + 10, gaze_y), width=2)
    pygame.draw.line(surface, (255, 255, 255), (gaze_x, gaze_y - 10), (gaze_x, gaze_y + 10), width=2)


def create_empty_rollout_results():
    return {
        "ai_paths": [],
        "ai_path_weights": [],
        "ai_greedy_path_count": 0,
        "ai_greedy_action_indices": [],
        "partner_paths": [],
        "ai_position_probabilities": None,
        "ai_valid_positions": None,
        "ai_interact_probabilities": None,
        "ai_start_position": None,
        "ai_start_orientation": None,
        "time_ms": 0,
    }


def initialize_position_histories(heatmap_predictor, state, ai_player_index, human_player_index):
    if heatmap_predictor is None:
        return None, None
        

    ai_start_position = tuple(state.players[ai_player_index].position)
    human_start_position = tuple(state.players[human_player_index].position)
    self_position_history = deque(
        [ai_start_position] * heatmap_predictor.history_len,
        maxlen=heatmap_predictor.history_len,
    )
    partner_position_history = deque(
        [human_start_position] * heatmap_predictor.history_len,
        maxlen=heatmap_predictor.history_len,
    )
    return self_position_history, partner_position_history


def snapshot_rollout_overlay_agent(heatmap_source, rollout_thread, ai_model):
    if heatmap_source != "rollout":
        return None
    if rollout_thread is not None and rollout_thread.is_alive():
        return None
    return snapshot_rollout_agent(ai_model)


def maybe_start_rollout_overlay_thread(
    rollout_thread,
    rollout_ai_model,
    env_state,
    partner_list,
    mdp,
    rollout_results,
    ai_player_index,
):
    if rollout_ai_model is None:
        return rollout_thread

    if rollout_results["time_ms"] > 0:
        rollout_results["time_ms"] = 0

    state_copy = copy.deepcopy(env_state)
    rollout_thread = threading.Thread(
        target=background_rollout_worker,
        args=(state_copy, rollout_ai_model, partner_list, mdp, rollout_results, ai_player_index),
    )
    rollout_thread.daemon = True
    rollout_thread.start()
    return rollout_thread


def update_model_heatmap_results(
    heatmap_predictor,
    state,
    self_position_history,
    partner_position_history,
    ai_player_index,
    human_player_index,
):
    if heatmap_predictor is None or self_position_history is None or partner_position_history is None:
        return None

    self_position_history.append(tuple(state.players[ai_player_index].position))
    partner_position_history.append(tuple(state.players[human_player_index].position))
    return run_path_model_prediction(
        copy.deepcopy(state),
        heatmap_predictor,
        np.asarray(self_position_history, dtype=np.int64),
        np.asarray(partner_position_history, dtype=np.int64),
    )


def normalize_action_value(action):
    if isinstance(action, (int, np.integer)):
        return Action.INDEX_TO_ACTION[int(action)]
    return action


def ensure_render_screen(
    screen,
    surface_size,
    layout_name,
    extra_vertical_space=0,
    minimum_width=0,
    map_scale=1.0,
):
    desired_size = get_render_window_size(
        surface_size,
        extra_vertical_space=extra_vertical_space,
        minimum_width=minimum_width,
        map_scale=map_scale,
    )
    if screen is not None and screen.get_size() == desired_size:
        return screen

    screen = pygame.display.set_mode(desired_size)
    pygame.display.set_caption(f"Pygame - User Study")
    return screen


def get_render_window_size(
    surface_size,
    extra_vertical_space=0,
    minimum_width=0,
    map_scale=1.0,
):
    scaled_surface_size = (
        int(round(surface_size[0] * map_scale)),
        int(round(surface_size[1] * map_scale)),
    )
    return (
        max(scaled_surface_size[0], int(minimum_width)),
        int(scaled_surface_size[1] + extra_vertical_space),
    )


def get_calibration_window_size():
    global _CALIBRATION_WINDOW_SIZE
    if _CALIBRATION_WINDOW_SIZE is not None:
        return _CALIBRATION_WINDOW_SIZE

    tile_size = int(BaseStateVisualizer.DEFAULT_VALUES["tile_size"])
    window_sizes = []
    for experiment_layout in CONDITION_LAYOUT_MAP.values():
        experiment_mdp = build_runtime_layout_mdp(experiment_layout)
        rows = len(experiment_mdp.terrain_mtx)
        columns = len(experiment_mdp.terrain_mtx[0])
        window_sizes.append(
            get_render_window_size(
                (columns * tile_size, rows * tile_size),
                extra_vertical_space=GAME_STATUS_BAR_HEIGHT * 2,
                minimum_width=MIN_EXPERIMENT_WINDOW_WIDTH,
                map_scale=MAP_RENDER_SCALE,
            )
        )
    _CALIBRATION_WINDOW_SIZE = (
        max(size[0] for size in window_sizes),
        max(size[1] for size in window_sizes),
    )
    return _CALIBRATION_WINDOW_SIZE


def draw_trial_wait_overlay(
    surface,
    *,
    condition,
    layout,
    status_pid,
    trial_progress,
    waiting_status,
    waiting_prompt,
    is_initial_wait,
):
    surface.fill((12, 15, 20))
    center_x = surface.get_width() // 2

    title_text = waiting_status or ("READY" if is_initial_wait else "TRIAL COMPLETE")
    title = render_fitted_text(title_text, (245, 247, 250), surface.get_width() - 80, 52)
    surface.blit(title, title.get_rect(center=(center_x, surface.get_height() // 2 - 90)))

    details = []
    if status_pid:
        details.append(f"PID  {str(status_pid).upper()}")
    if trial_progress:
        details.append(str(trial_progress).upper())
    if condition:
        details.append(f"CONDITION  {str(condition).upper()}")
    if layout:
        details.append(f"LAYOUT  {str(layout).upper()}")
    if details:
        detail = render_fitted_text(
            "    |    ".join(details),
            (185, 194, 207),
            surface.get_width() - 60,
            30,
        )
        surface.blit(detail, detail.get_rect(center=(center_x, surface.get_height() // 2 - 30)))

    prompt_text = waiting_prompt or (
        "PRESS Q TO START TRIAL"
        if is_initial_wait
        else "PRESS Q TO START NEXT TRIAL"
    )
    prompt = render_fitted_text(prompt_text, (76, 211, 130), surface.get_width() - 60, 32)
    surface.blit(prompt, prompt.get_rect(center=(center_x, surface.get_height() // 2 + 65)))


def render_fitted_text(
    text,
    color,
    max_width,
    preferred_size,
    minimum_size=14,
    measurement_text=None,
):
    fit_text = str(text if measurement_text is None else measurement_text)
    for font_size in range(preferred_size, minimum_size - 1, -1):
        font = pygame.font.Font(None, font_size)
        if font.render(fit_text, True, color).get_width() <= max_width:
            return font.render(str(text), True, color)
    return pygame.font.Font(None, minimum_size).render(str(text), True, color)


def get_layout_display_label(layout_name):
    for layout_code, configured_layout_name in CONDITION_LAYOUT_MAP.items():
        if layout_name == configured_layout_name:
            return layout_code
    return str(layout_name).upper().replace("_", " ")


def render_experiment_frame(
    screen,
    visualizer,
    env_state,
    terrain_mtx,
    layout_name,
    rollout_results,
    num_steps,
    show_heatmap_interactions,
    show_interaction_heatmap,
    show_tobii_gaze,
    tobii_gaze_tracker,
    gaze_calibration,
    decision_step,
    decision_horizon,
    score,
    condition,
    status_pid,
    trial_progress,
    waiting_for_block_start,
    waiting_status=None,
    waiting_prompt=None,
    eye_tracker_mode="game",
    drift_check_elapsed_sec=0.0,
    tutorial_test_target_score=None,
    tutorial_test_final_step=None,
    tutorial_notice=None,
):
    layout_text = get_layout_display_label(layout_name)
    if waiting_for_block_start:
        wait_window_size = get_calibration_window_size()
        if screen is None or screen.get_size() != wait_window_size:
            screen = pygame.display.set_mode(wait_window_size)
            pygame.display.set_caption("Pygame - User Study")
        draw_trial_wait_overlay(
            screen,
            condition=condition,
            layout=layout_text,
            status_pid=status_pid,
            trial_progress=trial_progress,
            waiting_status=waiting_status,
            waiting_prompt=waiting_prompt,
            is_initial_wait=decision_step == 0,
        )
        pygame.display.flip()
        return screen, None

    if gaze_calibration is not None and eye_tracker_mode in {
        "calibration",
        "gaze_drift",
        "gaze_preview",
    }:
        calibration_window_size = get_calibration_window_size()
        if screen is None or screen.get_size() != calibration_window_size:
            screen = pygame.display.set_mode(calibration_window_size)
            pygame.display.set_caption("Eye Tracker Calibration")
        if eye_tracker_mode == "calibration":
            draw_gaze_calibration_overlay(screen, gaze_calibration)
        elif eye_tracker_mode == "gaze_drift":
            draw_gaze_drift_overlay(screen, drift_check_elapsed_sec)
        else:
            preview_result = evaluate_gaze_preview(
                tobii_gaze_tracker,
                gaze_calibration,
                screen.get_size(),
            )
            draw_gaze_preview_overlay(
                screen,
                preview_result,
            )
        pygame.display.flip()
        return screen, None

    map_surface = visualizer.render_state(env_state, terrain_mtx)
    latest_surface_size = map_surface.get_size()
    overlay_animation_progress = 1.0
    previous_rollout_results = None

    heatmap_overlay_surface = None
    rollout_results["visible_overlay_entries"] = build_visible_overlay_entries(
        rollout_results,
        PATH_VISUALIZATION_MODE,
        num_steps,
        STOP_PATH_AT_INTERACTION,
        max_ranked_paths=NUM_PATHS_TO_SHOW,
        show_interaction_overlay=show_interaction_heatmap,
        path_render_style=PATH_RENDER_STYLE,
    )
    if (
        PATH_VISUALIZATION_MODE != PATH_VISUALIZATION_NONE
        and (rollout_results["ai_paths"] or rollout_results["ai_position_probabilities"] is not None)
    ):
        heatmap_overlay_surface = pygame.Surface(map_surface.get_size(), pygame.SRCALPHA)
        draw_heatmap_overlay(
            heatmap_overlay_surface,
            terrain_mtx,
            rollout_results["ai_paths"],
            rollout_results["partner_paths"],
            num_steps,
            show_heatmap_interactions,
            show_interaction_heatmap,
            ai_position_probabilities=rollout_results["ai_position_probabilities"],
            ai_valid_positions=rollout_results["ai_valid_positions"],
            ai_interact_probabilities=rollout_results["ai_interact_probabilities"],
            ai_start_position=rollout_results["ai_start_position"],
            ai_start_orientation=rollout_results["ai_start_orientation"],
            ai_path_weights=rollout_results["ai_path_weights"],
            ai_greedy_path_count=rollout_results["ai_greedy_path_count"],
            path_animation_progress=overlay_animation_progress,
            previous_rollout_results=previous_rollout_results,
        )
    if heatmap_overlay_surface is not None:
        map_surface.blit(heatmap_overlay_surface, (0, 0))

    score_text = f"{float(score):g}"
    condition_text = str(condition).upper()
    tutorial_status = re.fullmatch(r"TUTORIAL STAGE (\d+)", condition_text)
    is_tutorial_status = condition_text.startswith("TUT")
    tutorial_has_test_target = bool(tutorial_status and tutorial_test_target_score is not None)
    status_bar_height = GAME_STATUS_BAR_HEIGHT * (2 if tutorial_has_test_target else 1)

    top_reserved_space = status_bar_height
    screen = ensure_render_screen(
        screen,
        latest_surface_size,
        layout_name,
        extra_vertical_space=top_reserved_space * 2,
        minimum_width=MIN_EXPERIMENT_WINDOW_WIDTH,
        map_scale=MAP_RENDER_SCALE,
    )
    screen.fill((18, 18, 18))
    map_rect = get_surface_blit_rect(latest_surface_size, screen.get_size())

    panel_margin = 8
    status_rect = pygame.Rect(
        panel_margin,
        map_rect.top - status_bar_height,
        screen.get_width() - panel_margin * 2,
        status_bar_height,
    )
    pygame.draw.rect(screen, (28, 31, 36), status_rect, border_radius=8)
    pygame.draw.rect(screen, (75, 82, 92), status_rect, width=2, border_radius=8)
    if is_tutorial_status:
        progress_text = f"TOTAL STEP  {int(decision_step)}"
    else:
        progress_text = f"TOTAL STEP  {int(decision_step)} / {int(decision_horizon)}"
    if tutorial_status:
        status_label_rows = [
            [
                "TUTORIAL",
                f"STAGE {tutorial_status.group(1)}",
                f"LAYOUT  {layout_text}",
            ]
        ]
        if tutorial_notice:
            status_label_rows[0].append(str(tutorial_notice).upper())
        if tutorial_has_test_target:
            if waiting_status:
                status_label_rows.append([waiting_status])
            elif waiting_for_block_start:
                status_label_rows.append(["READY"])
            else:
                status_label_rows.append(
                    [
                        f"SCORE  {score_text} / {float(tutorial_test_target_score):g}",
                        f"STEP  {int(decision_step)} / {int(tutorial_test_final_step)}",
                        "R  RESET",
                    ]
                )
    else:
        if is_tutorial_status and tutorial_notice:
            condition_label = f"TUTORIAL  {str(tutorial_notice).upper()}"
        else:
            condition_label = (
                condition_text
                if is_tutorial_status
                else f"CONDITION  {condition_text or 'MANUAL'}"
            )
        status_labels = [condition_label, f"LAYOUT  {layout_text}"]
        if status_pid:
            status_labels.append(f"PID  {str(status_pid).upper()}")
        if trial_progress:
            status_labels.append(str(trial_progress).upper())
        status_labels.extend([progress_text, f"SCORE  {score_text}"])
        status_label_rows = [status_labels]

    for row_index, row_labels in enumerate(status_label_rows):
        preferred_font_size = 20 if len(row_labels) >= 5 else 24
        section_width = status_rect.width / len(row_labels)
        row_center_y = status_rect.top + GAME_STATUS_BAR_HEIGHT * (row_index + 0.5)
        for index, label in enumerate(row_labels):
            measurement_text = None
            if label == progress_text:
                max_step_digits = "8" * max(1, len(str(int(decision_horizon))))
                measurement_text = (
                    f"TOTAL STEP  {max_step_digits}"
                    if is_tutorial_status
                    else f"TOTAL STEP  {max_step_digits} / {max_step_digits}"
                )
            text_surface = render_fitted_text(
                label,
                (245, 247, 250),
                max(1, int(section_width) - 12),
                preferred_font_size,
                measurement_text=measurement_text,
            )
            text_rect = text_surface.get_rect(
                center=(
                    int(status_rect.left + section_width * (index + 0.5)),
                    int(row_center_y),
                )
            )
            screen.blit(text_surface, text_rect)

    if show_tobii_gaze and tobii_gaze_tracker is not None:
        gaze_surface_point, _ = resolve_gaze_surface_point(
            map_surface.get_size(),
            screen.get_size(),
            tobii_gaze_tracker.latest_gaze_point,
            gaze_calibration=gaze_calibration,
        )
        draw_tobii_gaze_overlay(map_surface, gaze_surface_point)

    rendered_map_surface = map_surface
    if map_rect.size != map_surface.get_size():
        rendered_map_surface = pygame.transform.scale(map_surface, map_rect.size)
    screen.blit(rendered_map_surface, map_rect)
    pygame.display.flip()
    return screen, latest_surface_size


def main(args):
    global NUM_STEPS, TIME_CONSTRAINT, NUM_SAMPLES, SHOW_HEATMAP_INTERACTIONS, SHOW_INTERACTION_HEATMAP, STOP_PATH_AT_INTERACTION, PATH_VISUALIZATION_MODE, PATH_RENDER_STYLE, SHOW_TOBII_GAZE
    resolved_condition = resolve_condition_settings(args.condition)
    if resolved_condition is not None:
        args.layout = resolved_condition["layout"]
        args.time_constraint = resolved_condition["time_constraint_ms"]
        args.path_step = resolved_condition["path_step"]
        args.visualization_mode = resolved_condition["visualization_mode"]
        args.path_render_style = resolved_condition["path_render_style"]
        # Experiment conditions must use the model registered for their own layout.
        args.hsp_model_path = None
        args.hsp_s2_run_name = None
        args.hsp_s2_model_index = None
        args.path_model_path = ""

    if args.path_step < 0:
        raise ValueError("--path-step must be non-negative.")
    if args.decision_horizon <= 0:
        raise ValueError("--decision-horizon must be positive.")
    NUM_STEPS = args.path_step
    TIME_CONSTRAINT = args.time_constraint
    if args.path_num_samples <= 0:
        raise ValueError("--path-num-samples must be positive.")
    NUM_SAMPLES = args.path_num_samples
    SHOW_HEATMAP_INTERACTIONS = NUM_STEPS
    SHOW_INTERACTION_HEATMAP = args.show_interaction_heatmap
    STOP_PATH_AT_INTERACTION = args.stop_path_at_interaction
    PATH_VISUALIZATION_MODE = args.visualization_mode
    PATH_RENDER_STYLE = args.path_render_style
    SHOW_TOBII_GAZE = args.show_tobii_gaze
    ai_player_index = args.ai_player_index
    human_player_index = 1 - ai_player_index
    ai_agent_index = args.ai_agent_index if args.ai_agent_index is not None else ai_player_index

    layout_name = args.layout
    mdp = build_runtime_layout_mdp(layout_name)
    configure_onion_cook_rate(
        mdp,
        ticks_per_env_step=DEFAULT_ONION_COOK_TICKS_PER_ENV_STEP,
    )
    env = build_runtime_env(mdp, horizon=100000000)
    agent_observation_layout = args.agent_observation_layout or layout_name
    agent_observation_mdp = mdp
    if agent_observation_layout != layout_name:
        agent_observation_mdp = build_runtime_layout_mdp(agent_observation_layout)
        validate_agent_observation_mdp(mdp, agent_observation_mdp)
        print(
            "[Agent Observation] "
            f"game_layout={layout_name}, observation_layout={agent_observation_layout}"
        )
    render_terrain_mtx = build_render_terrain_mtx(
        layout_name,
        mdp.terrain_mtx,
        double_rings_right_pot_as_counter=args.double_rings_right_pot_as_counter,
    )
    ai_model = build_agent_from_args(
        args,
        layout_name,
        mdp,
        agent_index=ai_agent_index,
        observation_mdp=agent_observation_mdp,
    )
    heatmap_predictor = None
    use_model_prediction = args.heatmap_source == "model" and args.enable_model_prediction

    if use_model_prediction:
        if (
            args.ai_agent != "HSP"
            and not getattr(args, "_tutorial_ai_checkpoint", None)
            and not args.tutorial
        ):
            raise ValueError("--heatmap-source=model currently supports --ai-agent HSP only")
        resolved_path_model_path = resolve_path_model_checkpoint(
            layout_name,
            model_path=args.path_model_path if args.path_model_path else None,
        )
        heatmap_predictor = PathModelHeatmapPredictor(
            resolved_path_model_path,
            mdp,
            agent_index=ai_player_index,
        )
    elif args.heatmap_source == "model":
        print("[Heatmap] model prediction is disabled. Running without model heatmap overlay.")

    # partner_ai_model = build_agent_from_args(args, layout_name, mdp, agent_index=human_player_index)
    partner_list = []
    partner_agent_names = []
    if args.heatmap_source == "rollout":
        partner_list, partner_agent_names = build_rollout_partner_agents(
            args,
            layout_name,
            mdp,
            partner_agent_index=human_player_index,
        )

    if ai_agent_index != ai_player_index:
        print(
            "[Warning] ai_agent_index does not match ai_player_index. "
            "The policy will observe one player slot but its action will be applied to the other slot."
        )

    config_parts = [
        "[Experiment Config]",
        f"layout={layout_name}, ai_agent={args.ai_agent}",
        f"agent_observation_layout={agent_observation_layout}",
        f"condition={args.condition or 'manual'}",
        f"ai_player_index={ai_player_index}",
        f"ai_agent_index={ai_agent_index}",
        f"human_player_index={human_player_index}",
        f"heatmap_source={args.heatmap_source}",
        f"model_prediction_enabled={use_model_prediction}",
        f"time_constraint={TIME_CONSTRAINT}",
        f"decision_horizon={args.decision_horizon}",
        f"show_interaction_heatmap={SHOW_INTERACTION_HEATMAP}",
        f"stop_path_at_interaction={STOP_PATH_AT_INTERACTION}",
        f"path_visualization_mode={PATH_VISUALIZATION_MODE}",
        f"path_render_style={PATH_RENDER_STYLE}",
        f"show_tobii_gaze={SHOW_TOBII_GAZE}",
        f"num_steps={NUM_STEPS}",
        f"double_rings_right_pot_as_counter={args.double_rings_right_pot_as_counter}",
        f"onion_cook_ticks_per_env_step={DEFAULT_ONION_COOK_TICKS_PER_ENV_STEP}",
    ]
    if args.heatmap_source == "rollout":
        config_parts.append(f"num_samples={NUM_SAMPLES}")
        config_parts.append(f"rollout_fixed_agent={args.ai_agent}")
        config_parts.append(f"rollout_partner_agents={','.join(partner_agent_names)}")
    elif heatmap_predictor is not None:
        if heatmap_predictor.prediction_target == "action":
            config_parts.append("model_visualization=action_rollout_greedy_plus_samples")
            config_parts.append(f"path_model_action_model_type={heatmap_predictor.action_model_type}")
            config_parts.append(f"path_model_num_samples={NUM_SAMPLES}")
        else:
            config_parts.append("model_visualization=raw_stepwise_argmax")
        config_parts.append(f"path_model={heatmap_predictor.model_path}")
        config_parts.append(f"path_model_prediction_target={heatmap_predictor.prediction_target}")
        config_parts.append(f"path_model_history_len={heatmap_predictor.history_len}")
        config_parts.append(f"path_model_target_horizon={heatmap_predictor.target_horizon}")
        config_parts.append(
            f"path_model_predict_interact={heatmap_predictor.predict_interact or heatmap_predictor.prediction_target == 'action'}"
        )
    else:
        config_parts.append("model_visualization=disabled")
    if getattr(args, "_tutorial_ai_checkpoint", None):
        config_parts.append(f"tutorial_ai_checkpoint={ai_model.checkpoint_path}")
        config_parts.append(f"tutorial_ai_policy_config={ai_model.policy_config_path}")
    elif args.ai_agent == "HSP":
        config_parts.append(f"hsp_checkpoint={ai_model.checkpoint_path}")
        if ai_model.s2_run_name is not None:
            config_parts.append(f"hsp_s2_run_name={ai_model.s2_run_name}")
            config_parts.append(f"hsp_s2_model_index={ai_model.s2_model_index}")
        config_parts.append(f"hsp_epsilon={ai_model.action_selection_epsilon}")
        config_parts.append(f"hsp_policy_config={ai_model.policy_config_rel_path}")
    print(
        "[Experiment Config] "
        + ", ".join(part for part in config_parts[1:])
    )

    pygame.init()
    clock = pygame.time.Clock()
    tobii_gaze_tracker = None
    gaze_calibration = None
    latest_surface_size = None
    eye_tracker_mode = "game"
    loaded_gaze_calibration = False
    if SHOW_TOBII_GAZE:
        tobii_gaze_tracker = TobiiGazeTracker(requested_address=args.tobii_address)
        gaze_calibration = GazeCalibration(
            layout_name,
            calibration_path=args.gaze_calibration_path,
            required_points=args.gaze_calibration_points,
            participant_id=args.pid,
        )
        try:
            loaded_gaze_calibration = gaze_calibration.load()
        except Exception as exc:
            gaze_calibration.status_message = f"Failed to load saved calibration: {exc}"
            print(f"[Tobii Calibration] {gaze_calibration.status_message}")

    visualizer_cls = TomatoStateVisualizer if isinstance(mdp, TomatoOvercookedGridworld) else BaseStateVisualizer
    visualizer = visualizer_cls(
        is_rendering_cooking_timer=True,
        show_timer_when_cooked=True,
    )
    screen = None
    env.reset()
    if hasattr(ai_model, "reset"):
        ai_model.reset()
    start_time = pygame.time.get_ticks()
    fixed_duration = args.fixed_duration_ms
    paused_since = None
    wait_for_block_start = bool(
        getattr(
            args,
            "_wait_for_block_start",
            not args.tutorial,
        )
    )
    wait_for_next_trial = bool(getattr(args, "_wait_for_next_trial", False))
    start_on_any_key = bool(getattr(args, "_start_on_any_key", False))
    start_wait_prompt = getattr(args, "_start_wait_prompt", None)
    current_state_onset_ms = start_time
    state_onset_initialized = False
    decision_step = 0
    pending_human_input_time_ms = None
    num_collisions = 0
    num_human_idle = 0
    gaze_time_origin_perf = time.perf_counter()
    gaze_time_origin_pygame_ms = start_time
    previous_gaze_context = None
    viewing_distance_locked = False
    drift_check_started_ms = None
    last_action_execution_ms = None

    trial_metadata = dict(getattr(args, "_trial_metadata", None) or {})
    trial_metadata["onion_cook_ticks_per_env_step"] = (
        DEFAULT_ONION_COOK_TICKS_PER_ENV_STEP
    )
    trial_metadata["agent_observation_layout"] = agent_observation_layout
    experiment_logger = ExperimentLogger(
        enabled=args.enable_experiment_logging,
        log_dir=args.experiment_log_dir,
        pid=args.pid,
        trial_id=args.trial_id,
        condition=args.condition,
        layout=layout_name,
        time_constraint_ms=TIME_CONSTRAINT,
        decision_horizon=args.decision_horizon,
        visualization_mode=PATH_VISUALIZATION_MODE,
        num_paths_shown=NUM_PATHS_TO_SHOW,  # NUM_SAMPLES 말고 이걸 사용
        num_steps=NUM_STEPS,
        stop_path_at_interaction=STOP_PATH_AT_INTERACTION,
        trial_metadata=trial_metadata,
        ivt_config=IVTConfig(
            screen_diagonal_inches=args.ivt_screen_diagonal_inches,
            viewing_distance_mm=args.ivt_viewing_distance_mm,
        ),
        ai_model=ai_model,
        path_predictor=heatmap_predictor,
        gaze_calibration=gaze_calibration,
        target_game_fps=TARGET_GAME_FPS,
    )
    if experiment_logger.enabled:
        print(f"[Experiment Logging] writing to {experiment_logger.session_dir}")

    pending_human_action = Action.STAY
    first_pending_human_action = None
    first_pending_human_input_time_ms = None
    done = False
    trial_completed = False
    tutorial_transition = None
    tutorial_mode = bool(args.tutorial)
    tutorial_phase = getattr(args, "_tutorial_phase", None)
    tutorial_test = tutorial_phase == "test"
    tutorial_test_passed = False
    tutorial_change_notice_until_ms = 0

    game_key_to_action = {
        pygame.K_UP: Direction.NORTH,
        pygame.K_DOWN: Direction.SOUTH,
        pygame.K_LEFT: Direction.WEST,
        pygame.K_RIGHT: Direction.EAST,
        pygame.K_SPACE: Action.INTERACT,
    }

    def enter_calibration_mode(current_time, start_collection=False):
        nonlocal eye_tracker_mode, paused_since
        if gaze_calibration is None:
            return
        eye_tracker_mode = "calibration"
        if paused_since is None:
            paused_since = current_time
        if start_collection:
            gaze_calibration.begin()
        elif gaze_calibration.has_transform:
            gaze_calibration.status_message = "Calibration ready. Press F5 to open the live gaze preview."
        else:
            gaze_calibration.status_message = "Calibration required. Press F6 to start the 9-point display calibration."

    def enter_game_mode(current_time):
        nonlocal eye_tracker_mode, paused_since, start_time
        nonlocal pending_human_action, pending_human_input_time_ms, current_state_onset_ms
        nonlocal first_pending_human_action, first_pending_human_input_time_ms
        nonlocal state_onset_initialized, viewing_distance_locked, drift_check_started_ms
        if gaze_calibration is not None and gaze_calibration.is_collecting:
            gaze_calibration.cancel()
        if paused_since is not None:
            paused_duration = current_time - paused_since
            start_time += paused_duration
            current_state_onset_ms += paused_duration
            if pending_human_input_time_ms is not None:
                pending_human_input_time_ms += paused_duration
            if first_pending_human_input_time_ms is not None:
                first_pending_human_input_time_ms += paused_duration
            paused_since = None
        eye_tracker_mode = "game"
        drift_check_started_ms = None
        state_onset_initialized = False
        if not viewing_distance_locked and tobii_gaze_tracker is not None:
            viewing_distance_info = tobii_gaze_tracker.estimate_viewing_distance(
                args.ivt_viewing_distance_mm
            )
            experiment_logger.set_viewing_distance(viewing_distance_info)
            viewing_distance_locked = True
            print(
                "[Tobii] Trial viewing distance "
                f"{viewing_distance_info['applied_distance_mm']:.1f}mm "
                f"source={viewing_distance_info['source']} "
                f"samples={viewing_distance_info['sample_count']}"
            )
        experiment_logger.mark_trial_started()
        pending_human_action = Action.STAY
        pending_human_input_time_ms = None
        first_pending_human_action = None
        first_pending_human_input_time_ms = None
        if tobii_gaze_tracker is not None:
            tobii_gaze_tracker.drain_gaze_samples()
        if gaze_calibration is not None:
            if gaze_calibration.has_transform:
                gaze_calibration.status_message = "Game mode. Press F6 to enter calibration mode."
            else:
                gaze_calibration.status_message = "Game mode without saved calibration. Press F6 to calibrate."

    def calibration_is_approved():
        if gaze_calibration is None:
            return True
        experiment_logger.set_calibration_quality(gaze_calibration)
        if gaze_calibration.can_start:
            return True
        gaze_calibration.status_message = gaze_calibration.quality_message() + " Press F6 to recalibrate."
        print(f"[Tobii Calibration] {gaze_calibration.status_message}")
        return False

    def start_gaze_preview(current_time):
        nonlocal eye_tracker_mode, paused_since
        if tobii_gaze_tracker is None or gaze_calibration is None:
            enter_game_mode(current_time)
            return
        if not calibration_is_approved():
            enter_calibration_mode(current_time, start_collection=False)
            gaze_calibration.status_message = gaze_calibration.quality_message() + " Press F6 to recalibrate."
            return
        eye_tracker_mode = "gaze_preview"
        if paused_since is None:
            paused_since = current_time
        tobii_gaze_tracker.drain_gaze_samples()
        gaze_calibration.status_message = "Live gaze preview. Press F5 to run the drift check."

    def start_gaze_drift_check(current_time):
        nonlocal eye_tracker_mode, paused_since, drift_check_started_ms
        if tobii_gaze_tracker is None or gaze_calibration is None:
            enter_game_mode(current_time)
            return
        eye_tracker_mode = "gaze_drift"
        drift_check_started_ms = current_time
        if paused_since is None:
            paused_since = current_time
        tobii_gaze_tracker.clear_recent_gaze_points()
        tobii_gaze_tracker.drain_gaze_samples()
        gaze_calibration.status_message = "Look at the center target for the drift check."

    def enter_block_wait_mode(current_time):
        nonlocal eye_tracker_mode, paused_since
        eye_tracker_mode = "block_wait"
        if paused_since is None:
            paused_since = current_time
        if tobii_gaze_tracker is not None:
            tobii_gaze_tracker.drain_gaze_samples()

    def drain_and_log_gaze_samples(max_timestamp_ms=None):
        if tobii_gaze_tracker is None:
            return
        map_display_size = None
        if latest_surface_size is not None and screen is not None:
            map_display_size = get_surface_blit_rect(
                latest_surface_size,
                screen.get_size(),
            ).size
        display_size = get_display_size()
        for gaze_sample in tobii_gaze_tracker.drain_gaze_samples():
            gaze_point = gaze_sample.get("point")
            gaze_surface_point = None
            if gaze_point is not None and latest_surface_size is not None and screen is not None:
                gaze_surface_point, _ = resolve_gaze_surface_point(
                    latest_surface_size,
                    screen.get_size(),
                    gaze_point,
                    gaze_calibration,
                )
            timestamp_ms = int(
                round(
                    gaze_time_origin_pygame_ms
                    + (float(gaze_sample["perf_counter"]) - gaze_time_origin_perf) * 1000.0
                )
            )
            if max_timestamp_ms is not None and timestamp_ms > max_timestamp_ms:
                continue
            gaze_context = {
                "step": decision_step,
                "state_onset_ms": current_state_onset_ms,
                "state": env.state,
                "rollout_results": rollout_results,
            }
            classify_gaze_aoi = True

            if not state_onset_initialized:
                gaze_context = None
                classify_gaze_aoi = False

            if gaze_context is not None and timestamp_ms < current_state_onset_ms:
                if (
                    previous_gaze_context is None
                    or timestamp_ms < previous_gaze_context["state_onset_ms"]
                ):
                    gaze_context = None
                    classify_gaze_aoi = False
                else:
                    gaze_context = previous_gaze_context
            experiment_logger.log_raw_gaze(
                step=None if gaze_context is None else gaze_context["step"],
                current_state_onset_ms=None if gaze_context is None else gaze_context["state_onset_ms"],
                timestamp_ms=timestamp_ms,
                gaze_x=None if gaze_point is None else float(gaze_point[0]),
                gaze_y=None if gaze_point is None else float(gaze_point[1]),
                gaze_validity=bool(gaze_sample.get("valid", False)),
                surface_point=gaze_surface_point,
                surface_size=latest_surface_size,
                map_display_size=map_display_size,
                display_size=display_size,
                terrain_mtx=render_terrain_mtx,
                state=None if gaze_context is None else gaze_context["state"],
                self_player_index=human_player_index,
                partner_player_index=ai_player_index,
                rollout_results=None if gaze_context is None else gaze_context["rollout_results"],
                classify_aoi=classify_gaze_aoi,
                callback_perf_counter=gaze_sample.get("callback_perf_counter"),
                tobii_device_timestamp_us=gaze_sample.get("device_timestamp_us"),
                tobii_system_timestamp_us=gaze_sample.get("system_timestamp_us"),
                gaze_point_validities=gaze_sample.get("gaze_point_validities"),
                gaze_origins=gaze_sample.get("gaze_origins"),
            )

    rollout_thread = None
    rollout_results = create_empty_rollout_results()
    self_position_history, partner_position_history = initialize_position_histories(
        heatmap_predictor,
        env.state,
        ai_player_index,
        human_player_index,
    )
    initial_rollout_results = None
    if heatmap_predictor is not None:
        initial_rollout_results = update_model_heatmap_results(
            heatmap_predictor,
            env.state,
            self_position_history,
            partner_position_history,
            ai_player_index,
            human_player_index,
        )

    if initial_rollout_results is not None:
        rollout_results = initial_rollout_results
        experiment_logger.record_path_prediction(
            prediction_step=0,
            rollout_results=rollout_results,
        )

    if gaze_calibration is not None:
        if loaded_gaze_calibration:
            start_gaze_preview(start_time)
        else:
            enter_calibration_mode(start_time, start_collection=True)
    elif wait_for_block_start:
        enter_block_wait_mode(start_time)
    else:
        experiment_logger.mark_trial_started()

    try:
        while not done:
            clock.tick(TARGET_GAME_FPS)
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if (
                        args.tutorial
                        and tutorial_phase == "test"
                        and eye_tracker_mode == "game"
                        and event.key == pygame.K_r
                    ):
                        tutorial_transition = {"type": "test_restart"}
                        break
                    if (
                        args.tutorial
                        and tutorial_phase == "practice"
                        and eye_tracker_mode == "game"
                        and event.key == pygame.K_q
                    ):
                        tutorial_transition = {
                            "type": "practice_complete",
                            "time_constraint": TIME_CONSTRAINT,
                            "path_steps": NUM_STEPS,
                        }
                        break
                    tutorial_controls_enabled = args.tutorial and tutorial_phase == "practice"
                    if tutorial_controls_enabled and eye_tracker_mode == "game" and event.key == pygame.K_F1:
                        tutorial_transition = {
                            "type": "layout",
                            "time_constraint": TIME_CONSTRAINT,
                            "path_steps": NUM_STEPS,
                        }
                        break
                    elif tutorial_controls_enabled and eye_tracker_mode == "game" and event.key == pygame.K_F2:
                        time_index = TUTORIAL_TIME_CONSTRAINTS_MS.index(TIME_CONSTRAINT)
                        TIME_CONSTRAINT = TUTORIAL_TIME_CONSTRAINTS_MS[
                            (time_index + 1) % len(TUTORIAL_TIME_CONSTRAINTS_MS)
                        ]
                        args.time_constraint = TIME_CONSTRAINT
                        current_state_onset_ms = current_time
                        state_onset_initialized = False
                        pending_human_action = Action.STAY
                        pending_human_input_time_ms = None
                        first_pending_human_action = None
                        first_pending_human_input_time_ms = None
                        tutorial_change_notice_until_ms = current_time + 1200
                        print("[Tutorial] changed")
                    elif tutorial_controls_enabled and eye_tracker_mode == "game" and event.key == pygame.K_F3:
                        path_index = TUTORIAL_PATH_STEPS.index(NUM_STEPS)
                        NUM_STEPS = TUTORIAL_PATH_STEPS[
                            (path_index + 1) % len(TUTORIAL_PATH_STEPS)
                        ]
                        args.path_step = NUM_STEPS
                        SHOW_HEATMAP_INTERACTIONS = NUM_STEPS
                        PATH_VISUALIZATION_MODE = (
                            PATH_VISUALIZATION_NONE
                            if NUM_STEPS == 0
                            else PATH_VISUALIZATION_FULL
                        )
                        args.visualization_mode = PATH_VISUALIZATION_MODE
                        tutorial_change_notice_until_ms = current_time + 1200
                        print("[Tutorial] changed")
                    elif eye_tracker_mode == "gaze_preview":
                        if event.key == pygame.K_F5:
                            preview_result = evaluate_gaze_preview(
                                tobii_gaze_tracker,
                                gaze_calibration,
                                (
                                    screen.get_size()
                                    if screen is not None
                                    else get_calibration_window_size()
                                ),
                            )
                            if preview_result["valid"]:
                                experiment_logger.set_gaze_preview(
                                    preview_result,
                                    accepted_ms=current_time,
                                )
                                start_gaze_drift_check(current_time)
                            else:
                                gaze_calibration.status_message = (
                                    "Cannot start until valid gaze is detected inside the preview window."
                                )
                        elif event.key == pygame.K_F6:
                            enter_calibration_mode(current_time, start_collection=True)
                        elif event.key == pygame.K_F7:
                            enter_calibration_mode(current_time, start_collection=False)
                            gaze_calibration.clear_saved()
                            gaze_calibration.status_message = (
                                "Saved calibration cleared. Press F6 to recalibrate."
                            )
                    elif eye_tracker_mode == "block_wait":
                        if start_on_any_key or event.key == pygame.K_q:
                            enter_game_mode(current_time)
                    elif gaze_calibration is not None and event.key == pygame.K_F5:
                        start_gaze_preview(current_time)
                    elif gaze_calibration is not None and event.key == pygame.K_F6:
                        enter_calibration_mode(current_time, start_collection=True)
                    elif gaze_calibration is not None and event.key == pygame.K_F7:
                        eye_tracker_mode = "calibration"
                        if paused_since is None:
                            paused_since = current_time
                        gaze_calibration.clear_saved()
                        gaze_calibration.status_message = "Saved calibration cleared. Press F6 to recalibrate."
                    elif (
                        gaze_calibration is not None
                        and eye_tracker_mode == "calibration"
                        and gaze_calibration.is_collecting
                        and event.key == pygame.K_ESCAPE
                    ):
                        gaze_calibration.cancel()
                        gaze_calibration.status_message = "Calibration canceled. Press F6 to retry."
                    elif (
                        gaze_calibration is not None
                        and eye_tracker_mode == "calibration"
                        and gaze_calibration.is_collecting
                        and event.key == pygame.K_BACKSPACE
                    ):
                        gaze_calibration.undo()
                    elif (
                        gaze_calibration is not None
                        and eye_tracker_mode == "calibration"
                        and gaze_calibration.is_collecting
                        and event.key in {pygame.K_RETURN, pygame.K_c}
                    ):
                        target_display_point = None
                        target_window_point = None
                        if screen is not None:
                            target_window_point = gaze_calibration.current_target_window_point(
                                screen.get_size()
                            )
                        if target_window_point is not None:
                            target_display_point = window_point_to_display_normalized(
                                target_window_point
                            )
                        if target_display_point is None:
                            target_display_point = gaze_calibration.current_target_normalized
                        raw_display_point = tobii_gaze_tracker.get_recent_average_gaze_point()
                        captured = gaze_calibration.capture_sample(raw_display_point, target_display_point)
                        if captured and not gaze_calibration.is_collecting and gaze_calibration.can_start:
                            gaze_calibration.status_message += " Press F5 to open the live gaze preview."
                    elif eye_tracker_mode == "game" and state_onset_initialized:
                        if event.key in game_key_to_action and not getattr(event, "repeat", False):
                            if first_pending_human_input_time_ms is None:
                                first_pending_human_action = game_key_to_action[event.key]
                                first_pending_human_input_time_ms = current_time
                            pending_human_action = game_key_to_action[event.key]
                            pending_human_input_time_ms = current_time
            if tutorial_transition is not None:
                break

            if (
                eye_tracker_mode == "gaze_drift"
                and drift_check_started_ms is not None
                and current_time - drift_check_started_ms
                >= int(round(DRIFT_CHECK_DURATION_SEC * 1000.0))
            ):
                viewing_distance_info = tobii_gaze_tracker.estimate_viewing_distance(
                    args.ivt_viewing_distance_mm
                )
                drift_target_display_point = window_point_to_display_normalized(
                    (screen.get_width() / 2.0, screen.get_height() / 2.0)
                )
                if drift_target_display_point is None:
                    drift_target_display_point = (0.5, 0.5)
                drift_result = evaluate_gaze_drift(
                    tobii_gaze_tracker.get_recent_valid_gaze_points(
                        DRIFT_CHECK_DURATION_SEC
                    ),
                    gaze_calibration,
                    viewing_distance_info["applied_distance_mm"],
                    screen_diagonal_inches=args.ivt_screen_diagonal_inches,
                    target_display_point=drift_target_display_point,
                )
                experiment_logger.set_drift_check(drift_result)
                if drift_result["passed"]:
                    print(
                        "[Tobii Drift] passed "
                        f"angle={drift_result['angle_deg']:.2f}deg "
                        f"samples={drift_result['sample_count']}"
                    )
                    if wait_for_block_start:
                        enter_block_wait_mode(current_time)
                    else:
                        enter_game_mode(current_time)
                else:
                    print(
                        "[Tobii Drift] failed "
                        f"status={drift_result['status']} "
                        f"angle={drift_result['angle_deg']} "
                        f"samples={drift_result['sample_count']}"
                    )
                    start_gaze_preview(current_time)
                    gaze_calibration.status_message = (
                        "Drift check failed. Verify gaze alignment, recalibrate with F6 if needed, "
                        "then press F5 to retry."
                    )

            if eye_tracker_mode == "game":
                experiment_logger.record_render_frame()
                duration_limit_reached = (
                    fixed_duration is not None
                    and current_time - start_time >= fixed_duration
                )
                tutorial_test_score_reached = (
                    tutorial_test
                    and float(env.cumulative_sparse_rewards)
                    >= float(getattr(args, "_tutorial_test_score_target", 80))
                )
                tutorial_test_step_limit_reached = (
                    tutorial_test
                    and decision_step >= args.decision_horizon
                )
                if tutorial_test and (
                    tutorial_test_score_reached
                    or tutorial_test_step_limit_reached
                    or duration_limit_reached
                ):
                    trial_completed = True
                    break
                if not tutorial_mode and (
                    decision_step >= args.decision_horizon or duration_limit_reached
                ):
                    trial_completed = True
                    break
            
            if eye_tracker_mode == "game":
                drain_and_log_gaze_samples()
            human_action = Action.STAY
            ai_action = Action.STAY
            human_action_due = False
            ai_action_due = False
            step_due = False
            action_execution_ms = None
            if eye_tracker_mode == "game" and state_onset_initialized:
                requested_human_action = Action.STAY
                human_input_ms = None
                first_human_action = None
                first_human_input_ms = None
                final_human_action = None
                final_human_input_ms = None
                human_input_changed = False
                response_input_source = None
                pre_step_state_onset_ms = current_state_onset_ms
                human_action_due = current_time - current_state_onset_ms >= TIME_CONSTRAINT
                if human_action_due:
                    requested_human_action = pending_human_action
                    first_human_action = first_pending_human_action
                    first_human_input_ms = first_pending_human_input_time_ms
                    final_human_action = (
                        pending_human_action
                        if pending_human_input_time_ms is not None
                        else None
                    )
                    final_human_input_ms = pending_human_input_time_ms
                    (
                        human_input_ms,
                        response_input_source,
                        human_input_changed,
                    ) = resolve_response_input(
                        first_human_action,
                        first_human_input_ms,
                        final_human_action,
                        final_human_input_ms,
                    )
                    human_action = requested_human_action
                    pending_human_action = Action.STAY
                    pending_human_input_time_ms = None
                    first_pending_human_action = None
                    first_pending_human_input_time_ms = None

                ai_action_due = current_time - current_state_onset_ms >= TIME_CONSTRAINT
                if ai_action_due:
                    rollout_ai_model = snapshot_rollout_overlay_agent(
                        args.heatmap_source,
                        rollout_thread,
                        ai_model,
                    )
                    ai_action = ai_model.action(env.state)
                    rollout_thread = maybe_start_rollout_overlay_thread(
                        rollout_thread,
                        rollout_ai_model,
                        env.state,
                        partner_list,
                        mdp,
                        rollout_results,
                        ai_player_index,
                    )

                ai_action = normalize_action_value(ai_action)
                human_action = normalize_action_value(human_action)
                if should_block_double_rings_right_pot_interaction(
                    layout_name,
                    env.state,
                    human_player_index,
                    human_action,
                    double_rings_right_pot_as_counter=args.double_rings_right_pot_as_counter,
                ):
                    human_action = Action.STAY

                step_due = ai_action_due or human_action_due

                joint_action = [Action.STAY, Action.STAY]
                joint_action[ai_player_index] = ai_action
                joint_action[human_player_index] = human_action

                logged_state = None
                logged_rollout_results = None
                pre_step_players = None

                if step_due:
                    # 사용자가 행동 전에 실제로 보고 있던 화면 상태/정보
                    logged_state = copy.deepcopy(env.state)
                    logged_rollout_results = copy.deepcopy(rollout_results)
                    pre_step_players = tuple(
                        player.deepcopy()
                        for player in env.state.players
                    )

                # 환경 시간은 렌더 프레임마다 진행해 조리 시간을 빠르게 흘린다.
                action_execution_ms = pygame.time.get_ticks() if step_due else None
                if action_execution_ms is not None:
                    last_action_execution_ms = action_execution_ms
                _, reward, done, _ = env.step(joint_action)
                experiment_logger.record_environment_step()
                if done and not tutorial_mode:
                    trial_completed = True

                if step_due:
                    experiment_logger.log_path_prediction_accuracy(
                        actual_step=decision_step,
                        actual_player=env.state.players[ai_player_index],
                        actual_action=ai_action,
                    )

                if heatmap_predictor is not None and step_due:
                    updated_rollout_results = update_model_heatmap_results(
                        heatmap_predictor,
                        env.state,
                        self_position_history,
                        partner_position_history,
                        ai_player_index,
                        human_player_index,
                    )
                    if updated_rollout_results is not None:
                        rollout_results = updated_rollout_results
                        experiment_logger.record_path_prediction(
                            prediction_step=decision_step + 1,
                            rollout_results=rollout_results,
                        )

                if step_due:
                    old_positions = tuple(player.position for player in pre_step_players)
                    candidate_positions = tuple(
                        mdp._move_if_direction(player.position, player.orientation, action)[0]
                        for player, action in zip(pre_step_players, joint_action)
                    )
                    is_collision = bool(mdp.is_transition_collision(old_positions, candidate_positions))
                    is_human_idle = bool(human_action_due and requested_human_action == Action.STAY)
                    response_time_ms = None
                    if human_input_ms is not None:
                        response_time_ms = max(0, int(human_input_ms - pre_step_state_onset_ms))

                    num_collisions += int(is_collision)
                    num_human_idle += int(is_human_idle)
                    logged_done = bool(
                        done
                        or (
                            not tutorial_mode
                            and decision_step + 1 >= args.decision_horizon
                        )
                    )

                    experiment_logger.log_step(
                        state=logged_state,
                        mdp=mdp,
                        step=decision_step,
                        state_onset_ms=pre_step_state_onset_ms,
                        human_action=human_action,
                        agent_action=ai_action,
                        joint_action=joint_action,
                        human_player_index=human_player_index,
                        agent_player_index=ai_player_index,
                        is_collision=is_collision,
                        human_action_due=human_action_due,
                        ai_action_due=ai_action_due,
                        is_human_idle=is_human_idle,
                        accumulated_score=env.cumulative_sparse_rewards,
                        reward=reward,
                        response_time_ms=response_time_ms,
                        human_input_ms=human_input_ms,
                        first_human_action=first_human_action,
                        first_human_input_ms=first_human_input_ms,
                        final_human_action=final_human_action,
                        final_human_input_ms=final_human_input_ms,
                        human_input_changed=human_input_changed,
                        response_input_source=response_input_source,
                        action_execution_ms=action_execution_ms,
                        done=logged_done,
                        rollout_results=logged_rollout_results,
                    )
                    previous_gaze_context = {
                        "step": decision_step,
                        "state_onset_ms": pre_step_state_onset_ms,
                        "state": logged_state,
                        "rollout_results": logged_rollout_results,
                    }
                    decision_step += 1

            screen, latest_surface_size = render_experiment_frame(
                screen,
                visualizer,
                env.state,
                render_terrain_mtx,
                layout_name,
                rollout_results,
                NUM_STEPS,
                SHOW_HEATMAP_INTERACTIONS,
                SHOW_INTERACTION_HEATMAP,
                SHOW_TOBII_GAZE and eye_tracker_mode == "calibration",
                tobii_gaze_tracker,
                gaze_calibration,
                decision_step,
                args.decision_horizon,
                env.cumulative_sparse_rewards,
                getattr(args, "_status_condition", args.condition),
                args.pid,
                getattr(args, "_status_trial_progress", None),
                eye_tracker_mode == "block_wait",
                waiting_prompt=start_wait_prompt,
                eye_tracker_mode=eye_tracker_mode,
                drift_check_elapsed_sec=(
                    0.0
                    if drift_check_started_ms is None
                    else max(0.0, (current_time - drift_check_started_ms) / 1000.0)
                ),
                tutorial_test_target_score=(
                    getattr(args, "_tutorial_test_score_target", None)
                    if tutorial_test
                    else None
                ),
                tutorial_test_final_step=(
                    args.decision_horizon if tutorial_test else None
                ),
                tutorial_notice=(
                    "CHANGED"
                    if tutorial_mode and current_time < tutorial_change_notice_until_ms
                    else None
                ),
            )
            if screen is not None:
                experiment_logger.set_screen_size(*screen.get_size())
            if eye_tracker_mode == "game" and (step_due or not state_onset_initialized):
                new_onset_ms = pygame.time.get_ticks()
                minimum_onset_ms = current_state_onset_ms + 1
                if action_execution_ms is not None:
                    minimum_onset_ms = max(minimum_onset_ms, action_execution_ms + 1)
                if new_onset_ms <= 0 or new_onset_ms < minimum_onset_ms:
                    new_onset_ms = minimum_onset_ms

                current_state_onset_ms = new_onset_ms
                state_onset_initialized = True
    finally:
        if tobii_gaze_tracker is not None:
            # Stop callbacks first so no delayed sample can arrive after the
            # final queue drain and disappear during logger shutdown.
            try:
                tobii_gaze_tracker.close()
            except Exception as exc:
                print(f"[Tobii] Failed to unsubscribe cleanly: {exc}")
            if last_action_execution_ms is not None:
                drain_and_log_gaze_samples(max_timestamp_ms=last_action_execution_ms)
        experiment_logger.write_metadata(
            ai_checkpoint=getattr(ai_model, "checkpoint_path", None)
            or getattr(ai_model, "model_path", None),
            path_predictor_checkpoint=(
                None if heatmap_predictor is None else heatmap_predictor.model_path
            ),
            calibration=serialize_calibration_info(gaze_calibration),
        )
        experiment_logger.log_trial_end(
            score_final=env.cumulative_sparse_rewards,
            num_collisions=num_collisions,
            num_human_idle=num_human_idle,
            decision_steps_completed=decision_step,
            completed=trial_completed,
            calibration_rmse=(
                None if gaze_calibration is None else gaze_calibration.fit_rmse
            ),
        )
        experiment_logger.close()
        if tutorial_test and trial_completed:
            tutorial_test_passed = (
                float(env.cumulative_sparse_rewards)
                >= float(getattr(args, "_tutorial_test_score_target", 80))
            )
        if (wait_for_next_trial or tutorial_test) and trial_completed:
            waiting_status = None
            waiting_prompt = None
            if tutorial_test:
                waiting_status = "TEST PASSED" if tutorial_test_passed else "TEST FAILED"
                waiting_prompt = (
                    "PRESS Q TO CONTINUE"
                    if tutorial_test_passed
                    else "PRESS Q TO RETRY"
                )
            screen, latest_surface_size = render_experiment_frame(
                screen,
                visualizer,
                env.state,
                render_terrain_mtx,
                layout_name,
                rollout_results,
                NUM_STEPS,
                SHOW_HEATMAP_INTERACTIONS,
                SHOW_INTERACTION_HEATMAP,
                False,
                None,
                gaze_calibration,
                decision_step,
                args.decision_horizon,
                env.cumulative_sparse_rewards,
                getattr(args, "_status_condition", args.condition),
                args.pid,
                getattr(args, "_status_trial_progress", None),
                True,
                waiting_status=waiting_status,
                waiting_prompt=waiting_prompt,
                tutorial_test_target_score=(
                    getattr(args, "_tutorial_test_score_target", None)
                    if tutorial_test
                    else None
                ),
                tutorial_test_final_step=(
                    args.decision_horizon if tutorial_test else None
                ),
            )
            waiting_for_q = True
            while waiting_for_q:
                clock.tick(60)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        waiting_for_q = False
        pygame.quit()
    if tutorial_test and trial_completed:
        return {
            "type": "test_result",
            "passed": tutorial_test_passed,
            "score": float(env.cumulative_sparse_rewards),
        }
    if tutorial_transition is None:
        print("🏁 Episode Finished!")
    return tutorial_transition


def run_experiment_args(args):
    if args.tutorial:
        run_tutorial(args, main, configure_tutorial)
        return
    if args.condition:
        if args.print_schedule:
            raise ValueError("--print-schedule requires --pid without an explicit --condition.")
        if args.dry_run:
            print(f"[Dry Run] explicit condition={args.condition}: {resolve_condition_settings(args.condition)}")
            return
        main(args)
        return
    if args.layout not in CONDITION_LAYOUT_MAP.values():
        if args.print_schedule or args.dry_run:
            raise ValueError("Schedule preview is only available for the A/B/C experiment layouts.")
        main(args)
        return
    if not args.pid:
        raise ValueError("Specify --pid to run all 27 balanced conditions, or specify --condition for one trial.")
    if args.trial != "completed" and not 1 <= args.trial <= TOTAL_EXPERIMENT_TRIALS:
        raise ValueError(f"--trial must be between 1 and {TOTAL_EXPERIMENT_TRIALS}.")

    schedule = get_participant_schedule(args.pid)
    schedule_output = args.schedule_output or os.path.join(
        args.experiment_log_dir,
        args.pid,
        "schedule.csv",
    )
    save_schedule_csv(schedule, schedule_output)
    print(f"[Schedule] pid={args.pid}, trials={len(schedule)}, csv={os.path.abspath(schedule_output)}")
    if args.print_schedule or args.dry_run:
        print_schedule(schedule)
    if args.dry_run:
        return

    completed_before_run = get_completed_global_trial_indices(
        pathlib.Path(args.experiment_log_dir) / args.pid / "trial.csv",
        args.pid,
    )
    previous_block = None
    first_trial_index = 1 if args.trial == "completed" else args.trial
    for trial_spec in schedule[first_trial_index - 1 :]:
        trial_index = trial_spec.global_trial_index
        if trial_index in completed_before_run:
            print(f"[Resume] skipping completed trial {trial_index:02d}/{len(schedule)}")
            continue
        condition = trial_spec.runtime_condition
        if trial_spec.block_index != previous_block:
            print(
                f"[Block {trial_spec.block_index}/3] "
                f"layout_assignment={trial_spec.layout_assignment_id}, "
                f"williams_sequence={trial_spec.williams_sequence_index}"
            )
        trial_args = copy.deepcopy(args)
        trial_args.condition = condition
        trial_args._trial_metadata = dict(trial_spec.__dict__)
        trial_args._status_condition = trial_spec.condition_id
        trial_args._status_trial_progress = f"TRIAL {trial_index} / {len(schedule)}"
        trial_args._wait_for_block_start = True
        trial_args._wait_for_next_trial = trial_index < len(schedule)
        if args.trial_id:
            trial_args.trial_id = (
                f"{args.trial_id}_{trial_index:02d}_{trial_spec.condition_id}_{trial_spec.layout_code}"
            )
        else:
            trial_args.trial_id = "_".join(
                [
                    time.strftime("%Y%m%d-%H%M%S"),
                    str(args.pid),
                    f"trial-{trial_index:02d}",
                    trial_spec.condition_id,
                    trial_spec.layout_code,
                ]
            )
        print(
            f"[Trial {trial_index:02d}/{len(schedule)}] "
            f"block={trial_spec.block_index}:{trial_spec.block_trial_index}, "
            f"condition={trial_spec.condition_id}, time={trial_spec.time_ms}ms, "
            f"information={trial_spec.information_steps}-step, "
            f"layout={trial_spec.layout_code}:{trial_spec.layout_name}"
        )
        main(trial_args)
        previous_block = trial_spec.block_index

    if not args.enable_experiment_logging:
        print("[Auto Preprocess] skipped because experiment logging is disabled.")
        return

    participant_log_dir = os.path.abspath(
        os.path.join(args.experiment_log_dir, args.pid)
    )
    completed_indices = get_completed_global_trial_indices(
        pathlib.Path(participant_log_dir) / "trial.csv",
        args.pid,
    )
    expected_indices = set(range(1, TOTAL_EXPERIMENT_TRIALS + 1))
    if completed_indices != expected_indices:
        missing_indices = sorted(expected_indices - completed_indices)
        print(
            "[Auto Preprocess] skipped because the 27-trial log is incomplete. "
            f"missing_trials={missing_indices}"
        )
        return

    processed_output_dir = os.path.abspath(
        args.processed_output_dir
        or os.path.join(PROJECT_ROOT, "data", "processed", args.pid)
    )
    try:
        result = build_model_dataset(
            pathlib.Path(participant_log_dir),
            pathlib.Path(processed_output_dir),
        )
    except Exception as exc:
        print(
            f"[Auto Preprocess] failed without changing source logs: {exc}",
            file=sys.stderr,
        )
        return
    print(
        "[Auto Preprocess] complete: "
        f"participants={result.participant_count}, trials={result.trial_count}, "
        f"decisions={result.decision_count}, output={processed_output_dir}"
    )


if __name__ == "__main__":
    run_experiment_args(parse_args())
