"""
use example
python predict_path/hsp_clone/export_hsp_dataset.py --layout counter_circuit

python predict_path/hsp_clone/export_hsp_dataset.py --layout counter_circuit --target-samples 1000000



"""
import argparse
import gc
import json
import math
import os

import numpy as np
import torch

from predict_path.hsp_clone.runtime import (
    Action,
    DEFAULT_POLICY_POOL_PARTNER_AGENTS,
    GreedyPartner,
    HSPAgent,
    OvercookedEnv,
    build_checkpoint_policy_agent,
    build_env,
    build_partner_agent,
    ensure_parent_dir,
    get_grid_shape,
    get_policy_pool_partner_variant_specs,
    parse_int_list,
    parse_name_list,
    resolve_torch_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export layout-specific HSP path-forecasting data.")
    parser.add_argument("--layout", required=True, help="Overcooked layout name")
    parser.add_argument(
        "--partner-agents",
        default=None,
        help="Comma-separated partner agents. If omitted, scan policy_pool and use every supported partner type that has at least one checkpoint.",
    )
    parser.add_argument(
        "--hsp-slots",
        default="0",
        help="Comma-separated slots where the target HSP is placed",
    )
    parser.add_argument("--history-len", type=int, default=5, help="Number of past positions to include")
    parser.add_argument("--target-horizon", type=int, default=6, help="How many future HSP positions to predict")
    parser.add_argument("--target-samples", type=int, default=200000, help="Approximate number of samples to collect")
    parser.add_argument("--horizon", type=int, default=400, help="Episode horizon")
    parser.add_argument("--hsp-epsilon", type=float, default=0.5, help="Sampling epsilon used by HSP")
    parser.add_argument(
        "--target-checkpoint",
        default=None,
        help="Optional exact target-policy checkpoint. If omitted, use the layout's default HSP.",
    )
    parser.add_argument(
        "--target-greedy",
        action="store_true",
        help="Use the planner-based GreedyHumanModel as the path-prediction target.",
    )
    parser.add_argument(
        "--target-algorithm",
        default=None,
        help="Optional algorithm-name override for --target-checkpoint.",
    )
    parser.add_argument(
        "--target-epsilon",
        type=float,
        default=0.5,
        help="Stochastic action epsilon for the target checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Dataset output base path. The .npz suffix is added when omitted.",
    )
    parser.add_argument("--device", default="cuda:0", help="Policy inference device. Examples: cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--policy-pool-s1-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only policy_pool S1 partner checkpoints. If a partner has no S1 variants, skip it instead of falling back to an S2/default partner.",
    )
    # parser.add_argument("--output-path", required=True, help="Compressed dataset output path (*.npz)")
    return parser.parse_args()


def resolve_partner_agents(layout_name, partner_agents_arg):
    if partner_agents_arg is not None:
        return parse_name_list(partner_agents_arg)

    detected_partner_agents = []
    for partner_name in DEFAULT_POLICY_POOL_PARTNER_AGENTS:
        variant_specs = get_policy_pool_partner_variant_specs(layout_name, partner_name, agent_index=0)
        if variant_specs:
            detected_partner_agents.append(partner_name)
    print(f"[partner_scan] detected={','.join(detected_partner_agents) if detected_partner_agents else 'none'}")
    return detected_partner_agents


def summarize_policy_agent(agent):
    return {
        "agent_class": agent.__class__.__name__,
        "checkpoint_path": getattr(agent, "checkpoint_path", None),
        "checkpoint_root": getattr(agent, "checkpoint_root", None),
        "policy_config_path": getattr(agent, "policy_config_path", None),
        "policy_config_root": getattr(agent, "policy_config_root", None),
    }


def format_agent_source(label, agent):
    summary = summarize_policy_agent(agent)
    checkpoint_path = summary["checkpoint_path"]
    policy_config_path = summary["policy_config_path"]
    if checkpoint_path is None and policy_config_path is None:
        return f"{label}=n/a"
    checkpoint_root = summary["checkpoint_root"] or "external"
    policy_config_root = summary["policy_config_root"] or "external"
    checkpoint_rel = checkpoint_path
    policy_config_rel = policy_config_path
    if checkpoint_path and checkpoint_root != "external":
        checkpoint_rel = os.path.relpath(checkpoint_path, checkpoint_root)
    if policy_config_path and policy_config_root != "external":
        policy_config_rel = os.path.relpath(policy_config_path, policy_config_root)
    return (
        f"{label}.ckpt={checkpoint_rel} "
        f"({checkpoint_root}) "
        f"{label}.cfg={policy_config_rel} "
        f"({policy_config_root})"
    )


def build_matchups(args):
    mdp, _ = build_env(args.layout, horizon=args.horizon)
    device = resolve_torch_device(args.device)
    print(f"[device] {device}")
    matchups = []
    failures = []
    target_agents = {}
    failed_target_slots = set()
    partner_agent_names = resolve_partner_agents(args.layout, args.partner_agents)
    for partner_name in partner_agent_names:
        for hsp_slot in parse_int_list(args.hsp_slots):
            if hsp_slot not in (0, 1):
                raise ValueError(f"Unsupported HSP slot: {hsp_slot}")
            partner_slot = 1 - hsp_slot
            if hsp_slot in failed_target_slots:
                continue
            if hsp_slot not in target_agents:
                try:
                    target_checkpoint = getattr(args, "target_checkpoint", None)
                    target_greedy = bool(getattr(args, "target_greedy", False))
                    target_options = [
                        bool(target_checkpoint),
                        target_greedy,
                    ]
                    if sum(target_options) > 1:
                        raise ValueError(
                            "Use only one of --target-checkpoint and --target-greedy."
                        )
                    if target_greedy:
                        target_agent = GreedyPartner(mdp, agent_index=hsp_slot)
                    elif target_checkpoint:
                        target_agent = build_checkpoint_policy_agent(
                            args.layout,
                            mdp,
                            agent_index=hsp_slot,
                            checkpoint_path=os.path.abspath(target_checkpoint),
                            algorithm_name_override=getattr(args, "target_algorithm", None),
                            action_selection_epsilon=getattr(args, "target_epsilon", 0.5),
                            device=device,
                        )
                    else:
                        target_agent = HSPAgent(
                            args.layout,
                            mdp,
                            agent_index=hsp_slot,
                            epsilon=args.hsp_epsilon,
                            device=device,
                        )
                    target_agents[hsp_slot] = target_agent
                except Exception as exc:
                    print(
                        f"[skip] target_hsp_slot={hsp_slot}: {type(exc).__name__}: {exc}"
                    )
                    failures.append(
                        f"target_hsp_slot={hsp_slot}: {type(exc).__name__}: {exc}"
                    )
                    failed_target_slots.add(hsp_slot)
                    continue
            target_agent = target_agents[hsp_slot]
            partner_variant_specs = get_policy_pool_partner_variant_specs(
                args.layout,
                partner_name,
                partner_slot,
            )

            if not partner_variant_specs:
                if args.policy_pool_s1_only:
                    print(
                        f"[skip] partner={partner_name} hsp_slot={hsp_slot}: "
                        "no policy_pool S1 variants found"
                    )
                    failures.append(
                        f"partner={partner_name} hsp_slot={hsp_slot}: no policy_pool S1 variants found"
                    )
                    continue
                try:
                    partner_agent = build_partner_agent(
                        partner_name,
                        args.layout,
                        mdp,
                        agent_index=partner_slot,
                        hsp_epsilon=args.hsp_epsilon,
                        device=device,
                    )
                except Exception as exc:
                    print(
                        f"[skip] partner={partner_name} hsp_slot={hsp_slot}: {type(exc).__name__}: {exc}"
                    )
                    failures.append(
                        f"partner={partner_name} hsp_slot={hsp_slot}: {type(exc).__name__}: {exc}"
                    )
                    continue

                partner_variant_specs = [
                    {
                        "variant_name": partner_name.lower(),
                        "partner_agent": partner_agent,
                    }
                ]

            for partner_variant_spec in partner_variant_specs:
                matchups.append(
                    {
                        "partner_name": partner_name,
                        "partner_variant_name": partner_variant_spec.get("variant_name", partner_name.lower()),
                        "hsp_slot": hsp_slot,
                        "partner_slot": partner_slot,
                        "target_agent": target_agent,
                        "partner_agent": partner_variant_spec.get("partner_agent"),
                        "partner_variant_spec": partner_variant_spec,
                        "partner_agent_summary": None,
                    }
                )
                partner_source = (
                    partner_variant_spec.get("checkpoint_path")
                    or "built-in"
                )
                print(
                    "[matchup queued] "
                    f"partner={partner_name} "
                    f"variant={partner_variant_spec.get('variant_name', partner_name.lower())} "
                    f"hsp_slot={hsp_slot} "
                    f"{format_agent_source('target', target_agent)} "
                    f"partner.source={partner_source}"
                )
    if not matchups:
        failure_summary = "\n".join(failures) if failures else "No partner agents were attempted."
        raise RuntimeError(
            "No valid matchup could be built.\n"
            f"layout={args.layout}\n"
            f"partner_agents={args.partner_agents}\n"
            f"hsp_slots={args.hsp_slots}\n"
            f"failures:\n{failure_summary}"
        )
    return mdp, matchups


def load_matchup_partner(matchup, args, mdp, device):
    partner_agent = matchup.get("partner_agent")
    if partner_agent is not None:
        return partner_agent

    partner_variant_spec = matchup["partner_variant_spec"]
    partner_agent = build_checkpoint_policy_agent(
        args.layout,
        mdp,
        agent_index=matchup["partner_slot"],
        checkpoint_path=partner_variant_spec["checkpoint_path"],
        algorithm_name_override=partner_variant_spec.get("algorithm_name_override"),
        recurrent=partner_variant_spec.get("recurrent"),
        action_selection_epsilon=args.hsp_epsilon if matchup["partner_name"] == "HSP" else 0.0,
        device=device,
    )
    matchup["partner_agent"] = partner_agent
    return partner_agent


def release_matchup_partner(matchup):
    if matchup["partner_variant_spec"].get("partner_agent") is not None:
        return
    matchup["partner_agent"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_position_history(position_series, end_index, history_len):
    start_index = max(0, end_index - history_len + 1)
    history = list(position_series[start_index : end_index + 1])
    if len(history) < history_len:
        history = [position_series[0]] * (history_len - len(history)) + history
    return np.asarray(history, dtype=np.int64)


def collect_episode(matchup, mdp, episode_horizon):
    env = OvercookedEnv(mdp, horizon=episode_horizon)
    env.reset()

    target_agent = matchup["target_agent"]
    partner_agent = matchup["partner_agent"]
    target_agent.reset()
    partner_agent.reset()

    hsp_slot = matchup["hsp_slot"]
    partner_slot = matchup["partner_slot"]

    obs_series = []
    available_actions_series = []
    action_index_series = []
    self_position_series = [tuple(env.state.players[hsp_slot].position)]
    partner_position_series = [tuple(env.state.players[partner_slot].position)]

    done = False
    while not done:
        target_info = target_agent.action_info(env.state)
        partner_info = partner_agent.action_info(env.state)

        obs_series.append(target_info["obs"])
        available_actions_series.append(target_info["available_actions"])
        action_index_series.append(int(target_info["action_index"]))

        if hsp_slot == 0:
            joint_action = (target_info["action"], partner_info["action"])
        else:
            joint_action = (partner_info["action"], target_info["action"])

        env.step(joint_action)

        self_position_series.append(tuple(env.state.players[hsp_slot].position))
        partner_position_series.append(tuple(env.state.players[partner_slot].position))
        done = env.is_done()

    return {
        "obs_series": obs_series,
        "available_actions_series": available_actions_series,
        "action_index_series": action_index_series,
        "self_position_series": self_position_series,
        "partner_position_series": partner_position_series,
    }


def run_export(args):
    output_path = getattr(args, "output_path", None) or f"./data/model_training/predict_path/{args.layout}_{args.target_samples}"
    if output_path.endswith(".npz"):
        output_path = output_path[:-4]
    np.random.seed(args.seed)

    mdp, matchups = build_matchups(args)
    device = resolve_torch_device(args.device)
    grid_width, grid_height = get_grid_shape(mdp)
    num_cells = grid_width * grid_height
    valid_positions = [tuple(position) for position in mdp.get_valid_player_positions()]
    position_to_index = {position: index for index, position in enumerate(valid_positions)}
    num_positions = len(valid_positions)
    samples_per_matchup = max(1, math.ceil(args.target_samples / len(matchups)))

    obs_rows = []
    available_rows = []
    self_position_history_rows = []
    partner_position_history_rows = []
    future_position_rows = []
    future_position_index_rows = []
    future_action_index_rows = []
    future_interact_rows = []
    episode_rows = []
    timestep_rows = []
    matchup_rows = []
    hsp_slot_rows = []
    interact_index = Action.ACTION_TO_INDEX[Action.INTERACT]

    matchup_counts = [0 for _ in matchups]
    episode_id = 0

    for matchup_idx, matchup in enumerate(matchups):
        try:
            partner_agent = load_matchup_partner(matchup, args, mdp, device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load partner={matchup['partner_name']} "
                f"variant={matchup['partner_variant_name']} "
                f"hsp_slot={matchup['hsp_slot']}: {type(exc).__name__}: {exc}"
            ) from exc
        matchup["partner_agent_summary"] = summarize_policy_agent(partner_agent)
        print(
            f"[loaded] partner={matchup['partner_name']} "
            f"variant={matchup['partner_variant_name']} "
            f"hsp_slot={matchup['hsp_slot']} "
            f"{format_agent_source('partner', partner_agent)}"
        )
        print(
            f"[collect] partner={matchup['partner_name']} "
            f"variant={matchup.get('partner_variant_name', matchup['partner_name'].lower())} "
            f"hsp_slot={matchup['hsp_slot']} target={samples_per_matchup}"
        )
        while matchup_counts[matchup_idx] < samples_per_matchup:
            episode = collect_episode(matchup, mdp, args.horizon)
            total_steps = len(episode["obs_series"])
            max_timestep = total_steps - args.target_horizon

            if max_timestep >= 0:
                for timestep in range(max_timestep + 1):
                    if matchup_counts[matchup_idx] >= samples_per_matchup:
                        break

                    self_position_history = build_position_history(
                        episode["self_position_series"],
                        timestep,
                        args.history_len,
                    )
                    partner_position_history = build_position_history(
                        episode["partner_position_series"],
                        timestep,
                        args.history_len,
                    )
                    future_positions = np.asarray(
                        episode["self_position_series"][timestep + 1 : timestep + args.target_horizon + 1],
                        dtype=np.int64,
                    )
                    future_position_indices = np.asarray(
                        [position_to_index[tuple(position)] for position in future_positions],
                        dtype=np.int64,
                    )
                    future_action_indices = np.asarray(
                        episode["action_index_series"][timestep : timestep + args.target_horizon],
                        dtype=np.int64,
                    )
                    future_interact_flags = np.asarray(
                        [
                            1 if action_index == interact_index else 0
                            for action_index in episode["action_index_series"][timestep : timestep + args.target_horizon]
                        ],
                        dtype=np.int64,
                    )

                    obs_rows.append(episode["obs_series"][timestep])
                    available_rows.append(episode["available_actions_series"][timestep])
                    self_position_history_rows.append(self_position_history)
                    partner_position_history_rows.append(partner_position_history)
                    future_position_rows.append(future_positions)
                    future_position_index_rows.append(future_position_indices)
                    future_action_index_rows.append(future_action_indices)
                    future_interact_rows.append(future_interact_flags)
                    episode_rows.append(episode_id)
                    timestep_rows.append(timestep)
                    matchup_rows.append(matchup_idx)
                    hsp_slot_rows.append(matchup["hsp_slot"])
                    matchup_counts[matchup_idx] += 1

            episode_id += 1

        print(
            f"[done] partner={matchup['partner_name']} "
            f"variant={matchup.get('partner_variant_name', matchup['partner_name'].lower())} "
            f"hsp_slot={matchup['hsp_slot']} "
            f"samples={matchup_counts[matchup_idx]}"
        )
        release_matchup_partner(matchup)

    dataset_path = f"{output_path}.npz"
    ensure_parent_dir(dataset_path)
    np.savez_compressed(
        dataset_path,
        obs=np.stack(obs_rows).astype(np.float32),
        available_actions=np.stack(available_rows).astype(np.uint8),
        self_position_history=np.stack(self_position_history_rows).astype(np.int64),
        partner_position_history=np.stack(partner_position_history_rows).astype(np.int64),
        future_positions=np.stack(future_position_rows).astype(np.int64),
        future_position_indices=np.stack(future_position_index_rows).astype(np.int64),
        future_action_indices=np.stack(future_action_index_rows).astype(np.int64),
        future_interact_flags=np.stack(future_interact_rows).astype(np.int64),
        episode_id=np.asarray(episode_rows, dtype=np.int64),
        timestep=np.asarray(timestep_rows, dtype=np.int64),
        matchup_id=np.asarray(matchup_rows, dtype=np.int64),
        hsp_slot=np.asarray(hsp_slot_rows, dtype=np.int64),
        valid_positions=np.asarray(valid_positions, dtype=np.int64),
        num_positions=np.asarray(num_positions, dtype=np.int64),
        grid_width=np.asarray(grid_width, dtype=np.int64),
        grid_height=np.asarray(grid_height, dtype=np.int64),
        num_cells=np.asarray(num_cells, dtype=np.int64),
        target_horizon=np.asarray(args.target_horizon, dtype=np.int64),
    )

    metadata = {
        "layout": args.layout,
        "seed": args.seed,
        "history_len": args.history_len,
        "target_horizon": args.target_horizon,
        "target_samples": args.target_samples,
        "num_samples": int(len(obs_rows)),
        "episode_horizon": args.horizon,
        "hsp_epsilon": args.hsp_epsilon,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "num_cells": num_cells,
        "num_positions": num_positions,
        "has_future_interact_flags": True,
        "policy_source": "policy_pool",
        "matchups": [
            {
                "matchup_id": matchup_idx,
                "partner_name": matchup["partner_name"],
                "partner_variant_name": matchup.get("partner_variant_name", matchup["partner_name"].lower()),
                "hsp_slot": matchup["hsp_slot"],
                "partner_slot": matchup["partner_slot"],
                "sample_count": matchup_counts[matchup_idx],
                "target_agent": summarize_policy_agent(matchup["target_agent"]),
                "partner_agent": matchup["partner_agent_summary"],
            }
            for matchup_idx, matchup in enumerate(matchups)
        ],
        "obs_shape": list(obs_rows[0].shape),
        "output_path": dataset_path,
    }
    metadata_path = f"{output_path}.json"
    ensure_parent_dir(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"[saved] dataset={dataset_path}")
    print(f"[saved] metadata={metadata_path}")
    return {
        "output_path": dataset_path,
        "metadata_path": metadata_path,
        "num_samples": int(len(obs_rows)),
    }


def main():
    
    run_export(parse_args())


if __name__ == "__main__":
    main()
