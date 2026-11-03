import argparse
import json
from collections import deque

import numpy as np
import torch

from predict_path.hsp_clone.runtime import (
    Action,
    DEFAULT_POLICY_POOL_PARTNER_AGENTS,
    Direction,
    HSPAgent,
    OvercookedEnv,
    build_env,
    build_partner_agent,
    ensure_parent_dir,
    featurize_state_for_agent,
)
from predict_path.hsp_clone.train_hsp_policy import (
    build_hsp_policy_model_from_checkpoint,
    build_single_feature,
    checkpoint_prediction_target,
    run_action_model,
    unpack_model_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample future HSP-clone paths for heatmap generation.")
    parser.add_argument("--layout", required=True, help="Overcooked layout name")
    parser.add_argument("--model-path", required=True, help="Trained clone checkpoint path")
    parser.add_argument(
        "--partner-agent",
        default=DEFAULT_POLICY_POOL_PARTNER_AGENTS[0],
        choices=list(DEFAULT_POLICY_POOL_PARTNER_AGENTS),
        help="Partner agent from the policy_pool set",
    )
    parser.add_argument("--hsp-slot", type=int, default=0, choices=[0, 1], help="Which slot the clone controls")
    parser.add_argument("--num-rollouts", type=int, default=100, help="Number of contexts to sample")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="How many real HSP/partner steps to run before forecasting. Defaults to history_len - 1.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax path instead of sampling")
    parser.add_argument("--hsp-epsilon", type=float, default=0.5, help="Sampling epsilon for HSP agents")
    parser.add_argument("--output-path", required=True, help="Where to save rollout JSON")
    return parser.parse_args()


class ClonePathPredictor:
    def __init__(self, checkpoint, mdp, agent_index):
        self.mdp = mdp
        self.agent_index = agent_index
        self.prediction_target = checkpoint_prediction_target(checkpoint)
        self.history_len = int(checkpoint["history_len"])
        self.obs_shape = tuple(checkpoint["obs_shape"])
        self.target_horizon = int(checkpoint["target_horizon"])
        self.grid_width = int(checkpoint["grid_width"])
        self.grid_height = int(checkpoint["grid_height"])
        self.valid_positions = np.asarray(checkpoint["valid_positions"], dtype=np.int64)
        self.num_positions = int(checkpoint["num_positions"])
        self.num_actions = int(checkpoint.get("num_actions", len(Action.ALL_ACTIONS)))
        self.predict_interact = bool(checkpoint.get("predict_interact", False))
        self.model = build_hsp_policy_model_from_checkpoint(checkpoint)

    def predict_outputs(self, state, self_position_history, partner_position_history):
        obs, available_actions = featurize_state_for_agent(
            self.mdp,
            state,
            self.agent_index,
            expected_obs_shape=self.obs_shape,
        )
        feature = build_single_feature(
            obs,
            available_actions,
            self_position_history,
            partner_position_history,
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
                return {
                    "action_probabilities": torch.softmax(action_logits[0], dim=-1).cpu().numpy(),
                    "position_probabilities": None,
                    "interact_probabilities": None,
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
            return {
                "action_probabilities": None,
                "position_probabilities": position_probabilities,
                "interact_probabilities": torch.softmax(interact_logits[0], dim=-1)[:, 1].cpu().numpy(),
            }


def reconstruct_action_path(action_probabilities, start_position, start_orientation, valid_positions):
    valid_position_set = {
        tuple(int(component) for component in position)
        for position in np.asarray(valid_positions, dtype=np.int64)
    }
    current_position = tuple(int(component) for component in start_position)
    current_orientation = start_orientation
    predicted_steps = []

    for horizon_idx, action_index in enumerate(np.asarray(action_probabilities).argmax(axis=1), start=1):
        action = Action.INDEX_TO_ACTION[int(action_index)]
        if action == Action.INTERACT:
            predicted_steps.append(
                {
                    "horizon": horizon_idx,
                    "action_index": int(action_index),
                    "action": "INTERACT",
                    "x": int(current_position[0]),
                    "y": int(current_position[1]),
                    "interact_x": int(Action.move_in_direction(current_position, current_orientation)[0]),
                    "interact_y": int(Action.move_in_direction(current_position, current_orientation)[1]),
                    "probability": float(action_probabilities[horizon_idx - 1, int(action_index)]),
                }
            )
            continue

        if action in Direction.ALL_DIRECTIONS:
            current_orientation = action
            candidate = tuple(Action.move_in_direction(current_position, action))
            if candidate in valid_position_set:
                current_position = candidate

        predicted_steps.append(
            {
                "horizon": horizon_idx,
                "action_index": int(action_index),
                "action": str(action),
                "x": int(current_position[0]),
                "y": int(current_position[1]),
                "probability": float(action_probabilities[horizon_idx - 1, int(action_index)]),
            }
        )

    return predicted_steps


def load_checkpoint(model_path):
    return torch.load(model_path, map_location="cpu")


def sparse_probability_map(probabilities, valid_positions):
    sparse_map = {}
    for position_index, probability in enumerate(probabilities):
        if probability <= 0:
            continue
        x, y = valid_positions[position_index]
        sparse_map[f"{x}:{y}"] = float(probability)
    return sparse_map


def sample_rollouts(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = load_checkpoint(args.model_path)
    mdp, _ = build_env(args.layout, horizon=checkpoint["target_horizon"])
    predictor = ClonePathPredictor(checkpoint, mdp, agent_index=args.hsp_slot)

    actual_hsp_agent = HSPAgent(
        args.layout,
        mdp,
        agent_index=args.hsp_slot,
        epsilon=args.hsp_epsilon,
    )
    partner_agent = build_partner_agent(
        args.partner_agent,
        args.layout,
        mdp,
        agent_index=1 - args.hsp_slot,
        hsp_epsilon=args.hsp_epsilon,
    )

    warmup_steps = predictor.history_len - 1 if args.warmup_steps is None else args.warmup_steps
    probability_sums = None
    if predictor.prediction_target != "action":
        probability_sums = np.zeros((predictor.target_horizon, predictor.num_positions), dtype=np.float64)
    rollouts = []

    for rollout_idx in range(args.num_rollouts):
        env = OvercookedEnv(mdp, horizon=warmup_steps + predictor.target_horizon)
        env.reset()
        actual_hsp_agent.reset()
        partner_agent.reset()

        self_position_history = deque(
            [tuple(env.state.players[args.hsp_slot].position)] * predictor.history_len,
            maxlen=predictor.history_len,
        )
        partner_position_history = deque(
            [tuple(env.state.players[1 - args.hsp_slot].position)] * predictor.history_len,
            maxlen=predictor.history_len,
        )

        for _ in range(warmup_steps):
            if env.is_done():
                break

            hsp_info = actual_hsp_agent.action_info(env.state)
            partner_info = partner_agent.action_info(env.state)

            if args.hsp_slot == 0:
                joint_action = (hsp_info["action"], partner_info["action"])
            else:
                joint_action = (partner_info["action"], hsp_info["action"])

            env.step(joint_action)
            self_position_history.append(tuple(env.state.players[args.hsp_slot].position))
            partner_position_history.append(tuple(env.state.players[1 - args.hsp_slot].position))

        prediction_outputs = predictor.predict_outputs(
            env.state,
            np.asarray(self_position_history, dtype=np.int64),
            np.asarray(partner_position_history, dtype=np.int64),
        )
        if predictor.prediction_target == "action":
            action_probabilities = prediction_outputs["action_probabilities"]
            predicted_steps = reconstruct_action_path(
                action_probabilities,
                env.state.players[args.hsp_slot].position,
                env.state.players[args.hsp_slot].orientation,
                predictor.valid_positions,
            )
        else:
            position_probabilities = prediction_outputs["position_probabilities"]
            interact_probabilities = prediction_outputs["interact_probabilities"]
            probability_sums += position_probabilities

            if args.deterministic:
                chosen_cells = position_probabilities.argmax(axis=1)
            else:
                chosen_cells = np.array(
                    [
                        np.random.choice(predictor.num_positions, p=position_probabilities[horizon_idx])
                        for horizon_idx in range(predictor.target_horizon)
                    ],
                    dtype=np.int64,
                )

            predicted_steps = []
            for horizon_idx, chosen_cell in enumerate(chosen_cells):
                x, y = predictor.valid_positions[chosen_cell]
                predicted_step = {
                    "horizon": horizon_idx + 1,
                    "x": int(x),
                    "y": int(y),
                    "probability": float(position_probabilities[horizon_idx, chosen_cell]),
                }
                if interact_probabilities is not None:
                    predicted_step["interact_probability"] = float(interact_probabilities[horizon_idx])
                predicted_steps.append(predicted_step)

        rollouts.append(
            {
                "rollout_id": rollout_idx,
                "self_position_history": [list(position) for position in self_position_history],
                "partner_position_history": [list(position) for position in partner_position_history],
                "predicted_steps": predicted_steps,
            }
        )

    aggregate = None
    if probability_sums is not None:
        average_probabilities = probability_sums / max(1, args.num_rollouts)
        aggregate = {
            f"h{horizon_idx + 1}": sparse_probability_map(
                average_probabilities[horizon_idx],
                predictor.valid_positions,
            )
            for horizon_idx in range(predictor.target_horizon)
        }

    result = {
        "layout": args.layout,
        "partner_agent": args.partner_agent,
        "hsp_slot": args.hsp_slot,
        "num_rollouts": args.num_rollouts,
        "warmup_steps": warmup_steps,
        "target_horizon": predictor.target_horizon,
        "prediction_target": predictor.prediction_target,
        "model_path": args.model_path,
        "aggregate": aggregate,
        "rollouts": rollouts,
    }

    ensure_parent_dir(args.output_path)
    with open(args.output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)

    print(f"[saved] rollout_json={args.output_path}")


def main():
    sample_rollouts(parse_args())


if __name__ == "__main__":
    main()
