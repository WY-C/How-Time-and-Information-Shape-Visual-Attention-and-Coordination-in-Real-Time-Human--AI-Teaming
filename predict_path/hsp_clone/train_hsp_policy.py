"""
run example)
python predict_path/hsp_clone/train_hsp_policy.py train --dataset-path data/model_training/predict_path/counter_circuit_200000.npz

python predict_path/hsp_clone/train_hsp_policy.py train --dataset-path data/model_training/predict_path/counter_circuit_200000.npz --hidden-dim 512

python predict_path/hsp_clone/train_hsp_policy.py train --dataset-path data/model_training/predict_path/counter_circuit_200000.npz --device cuda

python predict_path/hsp_clone/train_hsp_policy.py train_policy_pool --layout double_rings --target-samples 1000000
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from predict_path.hsp_clone.export_hsp_dataset import run_export as run_policy_pool_export
from predict_path.hsp_clone.runtime import Action, Direction, ensure_parent_dir
PREDICTION_TARGET_POSITION = "position"
PREDICTION_TARGET_ACTION = "action"
ACTION_DIM = len(Action.ALL_ACTIONS)
STAY_ACTION_INDEX = Action.ACTION_TO_INDEX[Action.STAY]
INTERACT_ACTION_INDEX = Action.ACTION_TO_INDEX[Action.INTERACT]
ACTION_MODEL_TYPE_PARALLEL = "parallel"
ACTION_MODEL_TYPE_AUTOREGRESSIVE = "autoregressive"
MOVE_DELTA_TO_ACTION_INDEX = {
    tuple(int(component) for component in direction): Action.ACTION_TO_INDEX[direction]
    for direction in Direction.ALL_DIRECTIONS
}
ACTION_INDEX_TO_DELTA = np.zeros((ACTION_DIM, 2), dtype=np.int64)
ACTION_INDEX_IS_DIRECTION = np.zeros((ACTION_DIM,), dtype=np.bool_)
for action, action_index in Action.ACTION_TO_INDEX.items():
    if action in Direction.ALL_DIRECTIONS:
        ACTION_INDEX_TO_DELTA[action_index] = np.asarray(action, dtype=np.int64)
        ACTION_INDEX_IS_DIRECTION[action_index] = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train or baseline-check a layout-specific HSP path forecaster.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    baseline_parser = subparsers.add_parser("baseline", help="Evaluate simple non-learned path baselines")
    add_common_args(baseline_parser)
    baseline_parser.add_argument("--output-path", required=True, help="Where to save baseline metrics JSON")

    train_parser = subparsers.add_parser("train", help="Train a multi-step HSP path forecaster")
    add_common_args(train_parser)
    # train_parser.add_argument("--output-model-path", required=True, help="Where to save the trained model")
    train_parser.add_argument("--output-metrics-path", default=None, help="Optional metrics JSON path")
    train_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size")
    train_parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    train_parser.add_argument(
        "--target-horizon",
        type=int,
        default=4,
        help="Optional training horizon override. Uses only the first H future steps from the dataset.",
    )
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    train_parser.add_argument(
        "--device",
        default="cuda:0",
        help="Training device. Examples: auto, cpu, cuda, cuda:0",
    )
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop if validation metric does not improve for this many epochs",
    )
    train_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement required to reset patience",
    )

    lopo_parser = subparsers.add_parser(
        "lopo",
        help="Run leave-one-partner-out training across all partners and optionally train a final mixed-partner model.",
    )
    add_common_args(lopo_parser)
    lopo_parser.add_argument("--output-metrics-path", default=None, help="Optional metrics JSON path for the final model")
    lopo_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size")
    lopo_parser.add_argument(
        "--hidden-dim-candidates",
        default="256,512,1024",
        help="Comma-separated hidden sizes to compare during LOPO model selection.",
    )
    lopo_parser.add_argument("--epochs", type=int, default=20000, help="Training epochs")
    lopo_parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    lopo_parser.add_argument(
        "--target-horizon",
        type=int,
        default=4,
        help="Optional training horizon override. Uses only the first H future steps from the dataset.",
    )
    lopo_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    lopo_parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    lopo_parser.add_argument(
        "--device",
        default="cuda:0",
        help="Training device. Examples: auto, cpu, cuda, cuda:0",
    )
    lopo_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop if validation metric does not improve for this many epochs",
    )
    lopo_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement required to reset patience",
    )
    lopo_parser.add_argument(
        "--output-summary-path",
        default=None,
        help="Optional JSON path for LOPO summary output",
    )
    lopo_parser.add_argument(
        "--skip-final-train",
        action="store_true",
        help="Run LOPO folds only and skip the final mixed-partner training pass.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Train multiple hidden sizes on the same dataset split and compare them.",
    )
    add_common_args(compare_parser)
    compare_parser.add_argument(
        "--hidden-dim-candidates",
        default="256,512,1024",
        help="Comma-separated hidden sizes to compare on the same train/val/test split.",
    )
    compare_parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    compare_parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    compare_parser.add_argument(
        "--target-horizon",
        type=int,
        default=None,
        help="Optional training horizon override. Uses only the first H future steps from the dataset.",
    )
    compare_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    compare_parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    compare_parser.add_argument(
        "--device",
        default="cuda:0",
        help="Training device. Examples: auto, cpu, cuda, cuda:0",
    )
    compare_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop if validation metric does not improve for this many epochs",
    )
    compare_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement required to reset patience",
    )
    compare_parser.add_argument(
        "--output-summary-path",
        default=None,
        help="Optional JSON path for model-comparison summary output",
    )
    compare_parser.add_argument(
        "--selected-model-path",
        default=None,
        help="Optional stable path where the best hidden-dimension checkpoint is copied.",
    )

    train_policy_pool_parser = subparsers.add_parser(
        "train_policy_pool",
        help="Build a dataset from policy_pool agents for a layout, then train immediately.",
    )
    train_policy_pool_parser.add_argument("--layout", required=True, help="Overcooked layout name")
    train_policy_pool_parser.add_argument(
        "--partner-agents",
        default=None,
        help="Comma-separated partner agents. If omitted, scan policy_pool and use every supported partner type that has at least one checkpoint.",
    )
    train_policy_pool_parser.add_argument(
        "--hsp-slots",
        default="0",
        help="Comma-separated slots where the target HSP is placed",
    )
    train_policy_pool_parser.add_argument("--history-len", type=int, default=5, help="Number of past positions to include")
    train_policy_pool_parser.add_argument(
        "--target-horizon",
        type=int,
        default=5,
        help="How many future HSP actions/positions to predict",
    )
    train_policy_pool_parser.add_argument(
        "--target-samples",
        type=int,
        default=200000,
        help="Approximate number of samples to collect before training",
    )
    train_policy_pool_parser.add_argument("--horizon", type=int, default=400, help="Episode horizon")
    train_policy_pool_parser.add_argument("--hsp-epsilon", type=float, default=0.5, help="Sampling epsilon used by HSP")
    train_policy_pool_parser.add_argument(
        "--device",
        default="cuda:0",
        help="Training and policy inference device. Examples: auto, cpu, cuda, cuda:0",
    )
    train_policy_pool_parser.add_argument("--seed", type=int, default=0, help="Split / training seed")
    train_policy_pool_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    train_policy_pool_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    train_policy_pool_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size")
    train_policy_pool_parser.add_argument(
        "--hidden-dim-candidates",
        default="256,512,1024",
        help="Comma-separated hidden sizes to compare. The best one is selected by acc@4.",
    )
    train_policy_pool_parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    train_policy_pool_parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    train_policy_pool_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_policy_pool_parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    train_policy_pool_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop if validation metric does not improve for this many epochs",
    )
    train_policy_pool_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement required to reset patience",
    )
    train_policy_pool_parser.add_argument(
        "--output-metrics-path",
        default=None,
        help="Optional metrics JSON path",
    )
    train_policy_pool_parser.add_argument(
        "--output-summary-path",
        default=None,
        help="Optional compare summary JSON path",
    )
    train_policy_pool_parser.add_argument(
        "--force-regenerate-dataset",
        action="store_true",
        help="Regenerate the policy-pool dataset even if the default output file already exists.",
    )
    train_policy_pool_parser.add_argument(
        "--policy-pool-s1-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only policy_pool S1 partner checkpoints when building the dataset.",
    )
    train_policy_pool_parser.add_argument(
        "--auto-select-hidden-dim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare hidden-dim candidates and select the best model automatically.",
    )
    return parser.parse_args()


def add_common_args(parser):
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Dataset path from predict_path/hsp_clone/export_hsp_dataset.py",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=0, help="Split / training seed")
    parser.add_argument(
        "--heldout-partner",
        default=None,
        help="Optional leave-one-partner-out split. Train/val use seen partners, test uses only this held-out partner.",
    )


def slice_dataset_target_horizon(dataset, target_horizon):
    if target_horizon is None:
        return dataset

    dataset_target_horizon = dataset_scalar(dataset, "target_horizon")
    if target_horizon <= 0:
        raise ValueError("--target-horizon must be positive.")
    if target_horizon > dataset_target_horizon:
        raise ValueError(
            f"--target-horizon={target_horizon} exceeds dataset target_horizon={dataset_target_horizon}."
        )
    if target_horizon == dataset_target_horizon:
        return dataset

    sliced_dataset = dict(dataset)
    for key in ("future_positions", "future_position_indices", "future_action_indices", "future_interact_flags"):
        if key in sliced_dataset:
            sliced_dataset[key] = np.asarray(sliced_dataset[key])[:, :target_horizon].copy()
    sliced_dataset["target_horizon"] = np.asarray(target_horizon, dtype=np.int64)
    return sliced_dataset


def parse_hidden_dim_candidates(value):
    if isinstance(value, (list, tuple)):
        candidates = [int(candidate) for candidate in value]
    else:
        candidates = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not candidates:
        raise ValueError("At least one hidden-dim candidate is required.")
    if any(candidate <= 0 for candidate in candidates):
        raise ValueError("Hidden-dim candidates must be positive integers.")
    return sorted(dict.fromkeys(candidates))


def load_dataset(dataset_path):
    with np.load(dataset_path, allow_pickle=False) as dataset:
        return {key: dataset[key] for key in dataset.files}


def dataset_scalar(dataset, key):
    return int(np.asarray(dataset[key]).item())


def validate_split_ratios(val_ratio, test_ratio):
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("Validation/test ratios must be non-negative.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("Validation/test ratios must sum to less than 1.0.")


def allocate_holdout_counts(total_count, val_ratio, test_ratio):
    val_count = int(round(total_count * val_ratio))
    test_count = int(round(total_count * test_ratio))

    if total_count >= 3:
        if val_ratio > 0:
            val_count = max(1, val_count)
        if test_ratio > 0:
            test_count = max(1, test_count)

    max_holdout = total_count - 1
    while val_count + test_count > max_holdout:
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    return val_count, test_count


def split_indices(episode_ids, val_ratio, test_ratio, seed):
    validate_split_ratios(val_ratio, test_ratio)

    rng = np.random.default_rng(seed)
    unique_episode_ids = np.unique(episode_ids)
    split_mode = "episode" if len(unique_episode_ids) >= 3 else "sample"

    if split_mode == "episode":
        shuffled_episode_ids = unique_episode_ids.copy()
        rng.shuffle(shuffled_episode_ids)
        val_count, test_count = allocate_holdout_counts(len(shuffled_episode_ids), val_ratio, test_ratio)

        val_episode_ids = set(shuffled_episode_ids[:val_count].tolist())
        test_episode_ids = set(shuffled_episode_ids[val_count : val_count + test_count].tolist())
        val_mask = np.array([episode_id in val_episode_ids for episode_id in episode_ids], dtype=bool)
        test_mask = np.array([episode_id in test_episode_ids for episode_id in episode_ids], dtype=bool)
    else:
        shuffled_indices = np.arange(len(episode_ids))
        rng.shuffle(shuffled_indices)
        val_count, test_count = allocate_holdout_counts(len(shuffled_indices), val_ratio, test_ratio)

        val_mask = np.zeros(len(shuffled_indices), dtype=bool)
        test_mask = np.zeros(len(shuffled_indices), dtype=bool)
        val_mask[shuffled_indices[:val_count]] = True
        test_mask[shuffled_indices[val_count : val_count + test_count]] = True

    train_mask = ~(val_mask | test_mask)
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            "Train/validation/test split failed. Increase dataset size or reduce val/test ratios."
        )

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mode": split_mode,
    }


def infer_metadata_path(dataset_path):
    if dataset_path.endswith(".npz"):
        return f"{dataset_path[:-4]}.json"
    return f"{dataset_path}.json"


def load_dataset_metadata(dataset_path):
    metadata_path = infer_metadata_path(dataset_path)
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file), metadata_path


def get_available_partners(dataset_path):
    metadata, _ = load_dataset_metadata(dataset_path)
    return sorted({str(entry["partner_name"]) for entry in metadata.get("matchups", [])})


def build_run_name(args, target_horizon):
    from pathlib import Path

    run_name = Path(args.dataset_path).stem
    if args.target_horizon is not None:
        run_name = f"{run_name}_h{target_horizon}"
    if getattr(args, "heldout_partner", None):
        run_name = f"{run_name}_heldout-{args.heldout_partner}"
    if getattr(args, "run_tag", None):
        run_name = f"{run_name}_{args.run_tag}"
    return run_name


def split_seen_indices(trainval_episode_ids, val_ratio, seed):
    if val_ratio < 0 or val_ratio >= 1.0:
        raise ValueError("For held-out partner splits, --val-ratio must satisfy 0 <= val_ratio < 1.")

    rng = np.random.default_rng(seed)
    unique_episode_ids = np.unique(trainval_episode_ids)
    split_mode = "episode" if len(unique_episode_ids) >= 2 else "sample"

    if split_mode == "episode":
        shuffled_episode_ids = unique_episode_ids.copy()
        rng.shuffle(shuffled_episode_ids)
        val_count = int(round(len(shuffled_episode_ids) * val_ratio))
        if len(shuffled_episode_ids) >= 2 and val_ratio > 0:
            val_count = max(1, val_count)
        val_count = min(val_count, max(0, len(shuffled_episode_ids) - 1))
        val_episode_ids = set(shuffled_episode_ids[:val_count].tolist())
        val_mask = np.array([episode_id in val_episode_ids for episode_id in trainval_episode_ids], dtype=bool)
    else:
        shuffled_indices = np.arange(len(trainval_episode_ids))
        rng.shuffle(shuffled_indices)
        val_count = int(round(len(shuffled_indices) * val_ratio))
        if len(shuffled_indices) >= 2 and val_ratio > 0:
            val_count = max(1, val_count)
        val_count = min(val_count, max(0, len(shuffled_indices) - 1))
        val_mask = np.zeros(len(shuffled_indices), dtype=bool)
        if val_count > 0:
            val_mask[shuffled_indices[:val_count]] = True

    train_mask = ~val_mask
    train_local_idx = np.flatnonzero(train_mask)
    val_local_idx = np.flatnonzero(val_mask)
    if len(train_local_idx) == 0 or len(val_local_idx) == 0:
        raise RuntimeError(
            "Held-out partner split failed to produce non-empty seen-partner train/val sets. "
            "Increase dataset size or reduce --val-ratio."
        )
    return train_local_idx, val_local_idx, split_mode


def split_indices_with_heldout_partner(dataset, dataset_path, heldout_partner, val_ratio, seed):
    metadata, metadata_path = load_dataset_metadata(dataset_path)
    matchup_entries = metadata.get("matchups", [])
    if not matchup_entries:
        raise RuntimeError(f"No matchup metadata found in {metadata_path}.")

    matchup_partner_lookup = {
        int(entry["matchup_id"]): str(entry["partner_name"])
        for entry in matchup_entries
    }
    available_partners = sorted(set(matchup_partner_lookup.values()))
    if heldout_partner not in available_partners:
        raise ValueError(
            f"Held-out partner '{heldout_partner}' was not found. Available partners: {available_partners}"
        )

    matchup_ids = np.asarray(dataset["matchup_id"], dtype=np.int64)
    episode_ids = np.asarray(dataset["episode_id"], dtype=np.int64)
    partner_names = np.asarray(
        [matchup_partner_lookup[int(matchup_id)] for matchup_id in matchup_ids],
        dtype=object,
    )
    heldout_mask = partner_names == heldout_partner
    seen_mask = ~heldout_mask

    if not np.any(heldout_mask):
        raise RuntimeError(f"No samples found for held-out partner '{heldout_partner}'.")
    if not np.any(seen_mask):
        raise RuntimeError(f"All samples belong to held-out partner '{heldout_partner}'.")

    seen_indices = np.flatnonzero(seen_mask)
    heldout_indices = np.flatnonzero(heldout_mask)
    train_local_idx, val_local_idx, seen_split_mode = split_seen_indices(
        episode_ids[seen_indices],
        val_ratio,
        seed,
    )

    return {
        "train_idx": seen_indices[train_local_idx],
        "val_idx": seen_indices[val_local_idx],
        "test_idx": heldout_indices,
        "mode": f"heldout_partner:{heldout_partner}",
        "heldout_partner": heldout_partner,
        "heldout_samples": int(len(heldout_indices)),
        "seen_split_mode": seen_split_mode,
        "available_partners": available_partners,
    }


def resolve_split(dataset, args):
    if args.heldout_partner:
        return split_indices_with_heldout_partner(
            dataset,
            args.dataset_path,
            args.heldout_partner,
            args.val_ratio,
            args.seed,
        )
    return split_indices(dataset["episode_id"], args.val_ratio, args.test_ratio, args.seed)


def normalize_position_history(position_history, grid_width, grid_height):
    normalized = np.asarray(position_history, dtype=np.float32).copy()
    normalized[..., 0] /= max(1, grid_width - 1)
    normalized[..., 1] /= max(1, grid_height - 1)
    return normalized


def build_feature_matrix_from_components(
    obs,
    available_actions,
    self_position_history,
    partner_position_history,
    grid_width,
    grid_height,
):
    obs_array = np.asarray(obs, dtype=np.float32)
    available_actions_array = np.asarray(available_actions, dtype=np.float32)
    self_position_history_array = np.asarray(self_position_history, dtype=np.float32)
    partner_position_history_array = np.asarray(partner_position_history, dtype=np.float32)

    if obs_array.ndim == 3:
        obs_array = obs_array[None, ...]
        available_actions_array = available_actions_array[None, ...]
        self_position_history_array = self_position_history_array[None, ...]
        partner_position_history_array = partner_position_history_array[None, ...]

    batch_size = len(obs_array)
    obs_features = obs_array.reshape(batch_size, -1) / 255.0
    available_actions_features = available_actions_array.reshape(batch_size, -1)
    self_history_features = normalize_position_history(
        self_position_history_array,
        grid_width,
        grid_height,
    ).reshape(batch_size, -1)
    partner_history_features = normalize_position_history(
        partner_position_history_array,
        grid_width,
        grid_height,
    ).reshape(batch_size, -1)

    return np.concatenate(
        [obs_features, available_actions_features, self_history_features, partner_history_features],
        axis=1,
    ).astype(np.float32)


def build_dataset_features(dataset):
    return build_feature_matrix_from_components(
        dataset["obs"],
        dataset["available_actions"],
        dataset["self_position_history"],
        dataset["partner_position_history"],
        dataset_scalar(dataset, "grid_width"),
        dataset_scalar(dataset, "grid_height"),
    )


def build_single_feature(
    obs,
    available_actions,
    self_position_history,
    partner_position_history,
    grid_width,
    grid_height,
):
    return build_feature_matrix_from_components(
        obs,
        available_actions,
        self_position_history,
        partner_position_history,
        grid_width,
        grid_height,
    )[0]


def get_valid_positions(dataset):
    return np.asarray(dataset["valid_positions"], dtype=np.int64)


def valid_index_to_position(valid_indices, valid_positions):
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    return valid_positions[valid_indices]


def build_position_to_index(valid_positions):
    return {tuple(position): index for index, position in enumerate(valid_positions)}


def evaluate_position_sequences(pred_positions, target_positions):
    pred_positions = np.asarray(pred_positions, dtype=np.int64)
    target_positions = np.asarray(target_positions, dtype=np.int64)
    target_horizon = target_positions.shape[1]
    metrics = {}
    accuracies = []
    distances = []

    for horizon_idx in range(target_horizon):
        pred_at_h = pred_positions[:, horizon_idx]
        target_at_h = target_positions[:, horizon_idx]
        accuracy = float(np.mean(np.all(pred_at_h == target_at_h, axis=1)))
        distance = float(np.mean(np.abs(pred_at_h[:, 0] - target_at_h[:, 0]) + np.abs(pred_at_h[:, 1] - target_at_h[:, 1])))

        metrics[f"acc@{horizon_idx + 1}"] = accuracy
        metrics[f"manhattan@{horizon_idx + 1}"] = distance
        accuracies.append(accuracy)
        distances.append(distance)

    metrics["avg_acc_1toH"] = float(np.mean(accuracies))
    metrics["avg_manhattan_1toH"] = float(np.mean(distances))
    if target_horizon >= 2:
        metrics["avg_acc_2toH"] = float(np.mean(accuracies[1:]))
        metrics["avg_manhattan_2toH"] = float(np.mean(distances[1:]))
    else:
        metrics["avg_acc_2toH"] = metrics["avg_acc_1toH"]
        metrics["avg_manhattan_2toH"] = metrics["avg_manhattan_1toH"]
    return metrics


def evaluate_position_predictions(pred_indices, target_indices, valid_positions):
    pred_positions = valid_index_to_position(pred_indices, valid_positions)
    target_positions = valid_index_to_position(target_indices, valid_positions)
    return evaluate_position_sequences(pred_positions, target_positions)


def infer_future_action_indices(self_position_history, future_positions, future_interact_flags=None):
    self_position_history = np.asarray(self_position_history, dtype=np.int64)
    future_positions = np.asarray(future_positions, dtype=np.int64)
    future_interact_flags = None if future_interact_flags is None else np.asarray(future_interact_flags, dtype=np.int64)

    batch_size, target_horizon = future_positions.shape[:2]
    action_indices = np.full((batch_size, target_horizon), STAY_ACTION_INDEX, dtype=np.int64)
    previous_positions = self_position_history[:, -1].copy()

    for horizon_idx in range(target_horizon):
        current_positions = future_positions[:, horizon_idx]
        deltas = current_positions - previous_positions
        for sample_idx, delta in enumerate(deltas):
            if future_interact_flags is not None and int(future_interact_flags[sample_idx, horizon_idx]) == 1:
                action_indices[sample_idx, horizon_idx] = INTERACT_ACTION_INDEX
                continue

            delta_key = (int(delta[0]), int(delta[1]))
            if delta_key == (0, 0):
                action_indices[sample_idx, horizon_idx] = STAY_ACTION_INDEX
            elif delta_key in MOVE_DELTA_TO_ACTION_INDEX:
                action_indices[sample_idx, horizon_idx] = MOVE_DELTA_TO_ACTION_INDEX[delta_key]
            else:
                raise ValueError(
                    f"Cannot infer one-step action from delta={delta_key} at horizon={horizon_idx + 1}. "
                    "Regenerate the dataset or inspect trajectory consistency."
                )
        previous_positions = current_positions

    return action_indices


def get_future_action_indices(dataset):
    if "future_action_indices" in dataset:
        return np.asarray(dataset["future_action_indices"], dtype=np.int64)
    return infer_future_action_indices(
        dataset["self_position_history"],
        dataset["future_positions"],
        future_interact_flags=dataset.get("future_interact_flags"),
    )


def rollout_action_indices_to_positions(start_positions, action_indices, valid_positions):
    start_positions = np.asarray(start_positions, dtype=np.int64)
    action_indices = np.asarray(action_indices, dtype=np.int64)
    valid_position_set = {
        tuple(int(component) for component in position)
        for position in np.asarray(valid_positions, dtype=np.int64)
    }

    batch_size, target_horizon = action_indices.shape
    predicted_positions = np.zeros((batch_size, target_horizon, 2), dtype=np.int64)
    current_positions = start_positions.copy()

    for horizon_idx in range(target_horizon):
        next_positions = current_positions.copy()
        for sample_idx, action_index in enumerate(action_indices[:, horizon_idx]):
            action = Action.INDEX_TO_ACTION[int(action_index)]
            if action in Direction.ALL_DIRECTIONS:
                candidate = tuple(
                    int(component)
                    for component in Action.move_in_direction(
                        (int(current_positions[sample_idx, 0]), int(current_positions[sample_idx, 1])),
                        action,
                    )
                )
                if candidate in valid_position_set:
                    next_positions[sample_idx] = np.asarray(candidate, dtype=np.int64)
        predicted_positions[:, horizon_idx] = next_positions
        current_positions = next_positions

    return predicted_positions


def evaluate_action_predictions(pred_action_indices, target_action_indices):
    target_horizon = target_action_indices.shape[1]
    metrics = {}
    accuracies = []

    for horizon_idx in range(target_horizon):
        accuracy = float(np.mean(pred_action_indices[:, horizon_idx] == target_action_indices[:, horizon_idx]))
        metrics[f"action_acc@{horizon_idx + 1}"] = accuracy
        accuracies.append(accuracy)

    metrics["avg_action_acc_1toH"] = float(np.mean(accuracies))
    if target_horizon >= 2:
        metrics["avg_action_acc_2toH"] = float(np.mean(accuracies[1:]))
    else:
        metrics["avg_action_acc_2toH"] = metrics["avg_action_acc_1toH"]
    return metrics


def format_horizon_metrics(metrics, metric_prefix, target_horizon):
    return " ".join(
        f"{metric_prefix}@{horizon}={metrics[f'{metric_prefix}@{horizon}']:.4f}"
        for horizon in range(1, target_horizon + 1)
    )


def make_position_tensors(features, future_position_indices, indices, future_interact_flags=None):
    feature_tensor = torch.from_numpy(features[indices])
    position_tensor = torch.from_numpy(future_position_indices[indices].astype(np.int64))
    interact_tensor = None
    if future_interact_flags is not None:
        interact_tensor = torch.from_numpy(future_interact_flags[indices].astype(np.int64))
    return feature_tensor, position_tensor, interact_tensor


def make_action_tensors(features, future_action_indices, indices):
    feature_tensor = torch.from_numpy(features[indices])
    action_tensor = torch.from_numpy(future_action_indices[indices].astype(np.int64))
    return feature_tensor, action_tensor


def make_action_training_tensors(features, future_action_indices, self_position_history, indices):
    feature_tensor, action_tensor = make_action_tensors(features, future_action_indices, indices)
    start_positions = torch.from_numpy(self_position_history[indices, -1].astype(np.int64))
    return feature_tensor, action_tensor, start_positions


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return device


def project_positions_to_valid_indices(predicted_positions, valid_positions, position_to_index):
    projected_indices = []
    for position in predicted_positions:
        position_key = (int(position[0]), int(position[1]))
        if position_key in position_to_index:
            projected_indices.append(position_to_index[position_key])
            continue
        distances = np.abs(valid_positions[:, 0] - position[0]) + np.abs(valid_positions[:, 1] - position[1])
        projected_indices.append(int(np.argmin(distances)))
    return np.asarray(projected_indices, dtype=np.int64)


def compute_stay_put_predictions(self_position_history, target_horizon, valid_positions, position_to_index):
    current_positions = self_position_history[:, -1]
    current_indices = np.asarray(
        [position_to_index[(int(position[0]), int(position[1]))] for position in current_positions],
        dtype=np.int64,
    )
    return np.repeat(current_indices[:, None], target_horizon, axis=1)


def compute_linear_extrapolation_predictions(
    self_position_history,
    target_horizon,
    grid_width,
    grid_height,
    valid_positions,
    position_to_index,
):
    current_positions = self_position_history[:, -1].astype(np.int64)
    if self_position_history.shape[1] > 1:
        previous_positions = self_position_history[:, -2].astype(np.int64)
    else:
        previous_positions = current_positions
    deltas = current_positions - previous_positions

    predictions = []
    for horizon in range(1, target_horizon + 1):
        predicted_positions = current_positions + horizon * deltas
        predicted_positions[:, 0] = np.clip(predicted_positions[:, 0], 0, grid_width - 1)
        predicted_positions[:, 1] = np.clip(predicted_positions[:, 1], 0, grid_height - 1)
        predicted_indices = project_positions_to_valid_indices(
            predicted_positions,
            valid_positions,
            position_to_index,
        )
        predictions.append(predicted_indices)
    return np.stack(predictions, axis=1)


def compute_baseline_metrics(dataset, eval_idx):
    grid_width = dataset_scalar(dataset, "grid_width")
    grid_height = dataset_scalar(dataset, "grid_height")
    target_horizon = dataset_scalar(dataset, "target_horizon")
    valid_positions = get_valid_positions(dataset)
    position_to_index = build_position_to_index(valid_positions)
    eval_targets = dataset["future_position_indices"][eval_idx]
    eval_self_position_history = dataset["self_position_history"][eval_idx]

    stay_put_predictions = compute_stay_put_predictions(
        eval_self_position_history,
        target_horizon,
        valid_positions,
        position_to_index,
    )
    linear_predictions = compute_linear_extrapolation_predictions(
        eval_self_position_history,
        target_horizon,
        grid_width,
        grid_height,
        valid_positions,
        position_to_index,
    )

    return {
        "stay_put": evaluate_position_predictions(stay_put_predictions, eval_targets, valid_positions),
        "linear_extrapolation": evaluate_position_predictions(linear_predictions, eval_targets, valid_positions),
    }


def compute_baselines(dataset, split):
    return {
        "val": compute_baseline_metrics(dataset, split["val_idx"]),
        "test": compute_baseline_metrics(dataset, split["test_idx"]),
        "num_train": int(len(split["train_idx"])),
        "num_val": int(len(split["val_idx"])),
        "num_test": int(len(split["test_idx"])),
        "split_mode": split["mode"],
    }


class HSPPathMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_horizon, num_positions):
        super().__init__()
        self.target_horizon = target_horizon
        self.num_positions = num_positions
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, target_horizon * num_positions)

    def forward(self, features):
        batch_size = features.shape[0]
        hidden = self.backbone(features)
        logits = self.head(hidden)
        return logits.reshape(batch_size, self.target_horizon, self.num_positions)


HSPPolicyMLP = HSPPathMLP


class HSPParallelActionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_horizon, num_actions):
        super().__init__()
        self.action_model_type = ACTION_MODEL_TYPE_PARALLEL
        self.target_horizon = target_horizon
        self.num_actions = num_actions
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, target_horizon * num_actions)

    def forward(self, features):
        batch_size = features.shape[0]
        hidden = self.backbone(features)
        logits = self.head(hidden)
        return logits.reshape(batch_size, self.target_horizon, self.num_actions)


class HSPAutoregressiveActionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_horizon, num_actions, grid_width, grid_height, valid_positions):
        super().__init__()
        self.action_model_type = ACTION_MODEL_TYPE_AUTOREGRESSIVE
        self.target_horizon = target_horizon
        self.num_actions = num_actions
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.start_action_index = num_actions
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder_init = nn.Linear(hidden_dim, hidden_dim)
        self.action_embedding = nn.Embedding(num_actions + 1, hidden_dim)
        self.step_embedding = nn.Embedding(target_horizon, hidden_dim)
        self.position_encoder = nn.Linear(2, hidden_dim)
        self.decoder = nn.GRUCell(hidden_dim * 4, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_actions)

        valid_position_mask = torch.zeros((grid_width, grid_height), dtype=torch.bool)
        for x, y in np.asarray(valid_positions, dtype=np.int64):
            valid_position_mask[int(x), int(y)] = True
        self.register_buffer(
            "action_deltas",
            torch.from_numpy(ACTION_INDEX_TO_DELTA.copy()),
            persistent=False,
        )
        self.register_buffer(
            "action_is_direction",
            torch.from_numpy(ACTION_INDEX_IS_DIRECTION.copy()),
            persistent=False,
        )
        self.register_buffer("valid_position_mask", valid_position_mask, persistent=False)

    def _normalize_positions(self, positions):
        normalized = positions.to(dtype=torch.float32)
        normalized = normalized.clone()
        normalized[:, 0] /= max(1, self.grid_width - 1)
        normalized[:, 1] /= max(1, self.grid_height - 1)
        return normalized

    def _advance_positions(self, current_positions, action_indices):
        candidate_positions = current_positions + self.action_deltas[action_indices]
        in_bounds = (
            (candidate_positions[:, 0] >= 0)
            & (candidate_positions[:, 0] < self.grid_width)
            & (candidate_positions[:, 1] >= 0)
            & (candidate_positions[:, 1] < self.grid_height)
        )
        candidate_x = candidate_positions[:, 0].clamp(0, self.grid_width - 1)
        candidate_y = candidate_positions[:, 1].clamp(0, self.grid_height - 1)
        valid_moves = (
            self.action_is_direction[action_indices]
            & in_bounds
            & self.valid_position_mask[candidate_x, candidate_y]
        )
        next_positions = current_positions.clone()
        next_positions[valid_moves] = candidate_positions[valid_moves]
        return next_positions

    def forward(self, features, start_positions, target_actions=None):
        batch_size = features.shape[0]
        if start_positions is None:
            raise ValueError("start_positions are required for autoregressive action decoding.")

        context = self.backbone(features)
        hidden = torch.tanh(self.decoder_init(context))
        current_positions = start_positions.to(device=features.device, dtype=torch.long)
        prev_actions = torch.full(
            (batch_size,),
            self.start_action_index,
            device=features.device,
            dtype=torch.long,
        )
        logits_per_step = []

        for horizon_idx in range(self.target_horizon):
            step_indices = torch.full(
                (batch_size,),
                horizon_idx,
                device=features.device,
                dtype=torch.long,
            )
            decoder_input = torch.cat(
                [
                    context,
                    self.action_embedding(prev_actions),
                    self.step_embedding(step_indices),
                    torch.relu(self.position_encoder(self._normalize_positions(current_positions))),
                ],
                dim=1,
            )
            hidden = self.decoder(decoder_input, hidden)
            logits = self.head(hidden)
            logits_per_step.append(logits)

            if target_actions is None:
                next_actions = logits.argmax(dim=-1)
            else:
                next_actions = target_actions[:, horizon_idx]
            current_positions = self._advance_positions(current_positions, next_actions)
            prev_actions = next_actions

        return torch.stack(logits_per_step, dim=1)


HSPActionMLP = HSPParallelActionMLP


class HSPPathMultiHeadMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_horizon, num_positions):
        super().__init__()
        self.target_horizon = target_horizon
        self.num_positions = num_positions
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.position_head = nn.Linear(hidden_dim, target_horizon * num_positions)
        self.interact_head = nn.Linear(hidden_dim, target_horizon * 2)

    def forward(self, features):
        batch_size = features.shape[0]
        hidden = self.backbone(features)
        position_logits = self.position_head(hidden).reshape(batch_size, self.target_horizon, self.num_positions)
        interact_logits = self.interact_head(hidden).reshape(batch_size, self.target_horizon, 2)
        return position_logits, interact_logits


def checkpoint_predicts_interact(checkpoint):
    return checkpoint_prediction_target(checkpoint) == PREDICTION_TARGET_POSITION and bool(checkpoint.get("predict_interact", False))


def checkpoint_prediction_target(checkpoint):
    return checkpoint.get("prediction_target", PREDICTION_TARGET_POSITION)


def checkpoint_action_model_type(checkpoint):
    return checkpoint.get("action_model_type", ACTION_MODEL_TYPE_PARALLEL)


def model_action_model_type(model):
    return getattr(model, "action_model_type", ACTION_MODEL_TYPE_PARALLEL)


def run_action_model(model, features, start_positions, target_actions=None):
    if model_action_model_type(model) == ACTION_MODEL_TYPE_AUTOREGRESSIVE:
        return model(features, start_positions, target_actions=target_actions)
    return model(features)


def build_hsp_policy_model_from_checkpoint(checkpoint):
    prediction_target = checkpoint_prediction_target(checkpoint)
    if prediction_target == PREDICTION_TARGET_ACTION:
        action_model_type = checkpoint_action_model_type(checkpoint)
        if action_model_type == ACTION_MODEL_TYPE_AUTOREGRESSIVE:
            model = HSPAutoregressiveActionMLP(
                input_dim=int(checkpoint["input_dim"]),
                hidden_dim=int(checkpoint["hidden_dim"]),
                target_horizon=int(checkpoint["target_horizon"]),
                num_actions=int(checkpoint.get("num_actions", ACTION_DIM)),
                grid_width=int(checkpoint["grid_width"]),
                grid_height=int(checkpoint["grid_height"]),
                valid_positions=np.asarray(checkpoint["valid_positions"], dtype=np.int64),
            )
        else:
            model = HSPParallelActionMLP(
                input_dim=int(checkpoint["input_dim"]),
                hidden_dim=int(checkpoint["hidden_dim"]),
                target_horizon=int(checkpoint["target_horizon"]),
                num_actions=int(checkpoint.get("num_actions", ACTION_DIM)),
            )
    else:
        model_cls = HSPPathMultiHeadMLP if checkpoint_predicts_interact(checkpoint) else HSPPathMLP
        model = model_cls(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            target_horizon=int(checkpoint["target_horizon"]),
            num_positions=int(checkpoint["num_positions"]),
        )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def unpack_model_outputs(model_outputs):
    if isinstance(model_outputs, tuple):
        return model_outputs
    return model_outputs, None


def multi_horizon_cross_entropy(logits, target_indices):
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_indices.reshape(-1),
    )


def interact_cross_entropy(interact_logits, future_interact_flags):
    return F.cross_entropy(
        interact_logits.reshape(-1, interact_logits.shape[-1]),
        future_interact_flags.reshape(-1),
    )


def compute_training_loss(logits, target_indices, interact_logits=None, future_interact_flags=None):
    loss = multi_horizon_cross_entropy(logits, target_indices)
    if interact_logits is not None and future_interact_flags is not None:
        loss = loss + interact_cross_entropy(interact_logits, future_interact_flags)
    return loss


def evaluate_interact_predictions(pred_interact_flags, target_interact_flags):
    target_horizon = target_interact_flags.shape[1]
    metrics = {}
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for horizon_idx in range(target_horizon):
        pred_at_h = pred_interact_flags[:, horizon_idx].astype(np.int64)
        target_at_h = target_interact_flags[:, horizon_idx].astype(np.int64)

        tp = int(np.sum((pred_at_h == 1) & (target_at_h == 1)))
        fp = int(np.sum((pred_at_h == 1) & (target_at_h == 0)))
        fn = int(np.sum((pred_at_h == 0) & (target_at_h == 1)))

        accuracy = float(np.mean(pred_at_h == target_at_h))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_score = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics[f"interact_acc@{horizon_idx + 1}"] = accuracy
        metrics[f"interact_precision@{horizon_idx + 1}"] = precision
        metrics[f"interact_recall@{horizon_idx + 1}"] = recall
        metrics[f"interact_f1@{horizon_idx + 1}"] = f1_score
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    metrics["avg_interact_acc_1toH"] = float(np.mean(accuracies))
    metrics["avg_interact_precision_1toH"] = float(np.mean(precisions))
    metrics["avg_interact_recall_1toH"] = float(np.mean(recalls))
    metrics["avg_interact_f1_1toH"] = float(np.mean(f1_scores))
    if target_horizon >= 2:
        metrics["avg_interact_acc_2toH"] = float(np.mean(accuracies[1:]))
        metrics["avg_interact_precision_2toH"] = float(np.mean(precisions[1:]))
        metrics["avg_interact_recall_2toH"] = float(np.mean(recalls[1:]))
        metrics["avg_interact_f1_2toH"] = float(np.mean(f1_scores[1:]))
    else:
        metrics["avg_interact_acc_2toH"] = metrics["avg_interact_acc_1toH"]
        metrics["avg_interact_precision_2toH"] = metrics["avg_interact_precision_1toH"]
        metrics["avg_interact_recall_2toH"] = metrics["avg_interact_recall_1toH"]
        metrics["avg_interact_f1_2toH"] = metrics["avg_interact_f1_1toH"]
    return metrics


@torch.no_grad()
def evaluate_model(
    model,
    features,
    target_indices,
    future_positions,
    self_position_history,
    valid_positions,
    device,
    batch_size,
    prediction_target,
    future_interact_flags=None,
):
    model.eval()
    primary_predictions = []
    interact_predictions = []
    total_loss = 0.0
    total_count = 0
    action_model_type = model_action_model_type(model) if prediction_target == PREDICTION_TARGET_ACTION else None

    for start_idx in range(0, len(features), batch_size):
        end_idx = start_idx + batch_size
        batch_features = features[start_idx:end_idx].to(device, non_blocking=device.type == "cuda")
        batch_targets = target_indices[start_idx:end_idx].to(device, non_blocking=device.type == "cuda")
        batch_start_positions = None
        if prediction_target == PREDICTION_TARGET_ACTION:
            batch_start_positions = self_position_history[start_idx:end_idx, -1].to(
                device,
                non_blocking=device.type == "cuda",
            )
        batch_interact_targets = None
        if future_interact_flags is not None:
            batch_interact_targets = future_interact_flags[start_idx:end_idx].to(
                device,
                non_blocking=device.type == "cuda",
            )

        if prediction_target == PREDICTION_TARGET_ACTION:
            logits = run_action_model(
                model,
                batch_features,
                batch_start_positions,
                target_actions=batch_targets if action_model_type == ACTION_MODEL_TYPE_AUTOREGRESSIVE else None,
            )
            interact_logits = None
            loss = compute_training_loss(logits, batch_targets)
            prediction_logits = logits
            if action_model_type == ACTION_MODEL_TYPE_AUTOREGRESSIVE:
                prediction_logits = run_action_model(
                    model,
                    batch_features,
                    batch_start_positions,
                    target_actions=None,
                )
        else:
            model_outputs = model(batch_features)
            logits, interact_logits = unpack_model_outputs(model_outputs)
            loss = compute_training_loss(
                logits,
                batch_targets,
                interact_logits=interact_logits,
                future_interact_flags=batch_interact_targets,
            )
            prediction_logits = logits

        batch_count = len(batch_features)
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
        primary_predictions.append(prediction_logits.argmax(dim=-1).cpu())
        if batch_interact_targets is not None and interact_logits is not None:
            interact_predictions.append(interact_logits.argmax(dim=-1).cpu())

    primary_predictions = torch.cat(primary_predictions, dim=0).numpy()
    if prediction_target == PREDICTION_TARGET_ACTION:
        metrics = evaluate_action_predictions(primary_predictions, target_indices.cpu().numpy())
        predicted_positions = rollout_action_indices_to_positions(
            self_position_history.cpu().numpy()[:, -1],
            primary_predictions,
            valid_positions,
        )
    else:
        metrics = {}
        predicted_positions = valid_index_to_position(primary_predictions, valid_positions)

    metrics.update(
        evaluate_position_sequences(
            predicted_positions,
            future_positions.cpu().numpy(),
        )
    )
    if interact_predictions:
        metrics.update(
            evaluate_interact_predictions(
                torch.cat(interact_predictions, dim=0).numpy(),
                future_interact_flags.cpu().numpy(),
            )
        )
    metrics["loss"] = total_loss / max(1, total_count)
    return metrics


def train_model(args, dataset, features, split):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    target_horizon = dataset_scalar(dataset, "target_horizon")
    valid_positions = get_valid_positions(dataset)
    future_positions = dataset["future_positions"]
    future_action_indices = get_future_action_indices(dataset)
    prediction_target = PREDICTION_TARGET_ACTION
    predict_interact = False
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    train_features, train_targets, train_start_positions = make_action_training_tensors(
        features,
        future_action_indices,
        dataset["self_position_history"],
        train_idx,
    )
    val_features, val_targets, _ = make_action_training_tensors(
        features,
        future_action_indices,
        dataset["self_position_history"],
        val_idx,
    )
    test_features, test_targets, _ = make_action_training_tensors(
        features,
        future_action_indices,
        dataset["self_position_history"],
        test_idx,
    )
    train_future_positions = torch.from_numpy(future_positions[train_idx].astype(np.int64))
    val_future_positions = torch.from_numpy(future_positions[val_idx].astype(np.int64))
    test_future_positions = torch.from_numpy(future_positions[test_idx].astype(np.int64))
    train_self_position_history = torch.from_numpy(dataset["self_position_history"][train_idx].astype(np.int64))
    val_self_position_history = torch.from_numpy(dataset["self_position_history"][val_idx].astype(np.int64))
    test_self_position_history = torch.from_numpy(dataset["self_position_history"][test_idx].astype(np.int64))

    train_dataset = TensorDataset(train_features, train_targets, train_start_positions)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    model = HSPAutoregressiveActionMLP(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        target_horizon=target_horizon,
        num_actions=ACTION_DIM,
        grid_width=int(dataset_scalar(dataset, "grid_width")),
        grid_height=int(dataset_scalar(dataset, "grid_height")),
        valid_positions=valid_positions,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if use_cuda:
        print(f"[device] {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"[device] {device}")

    best_state = None
    best_epoch = None
    best_val_metric = None
    selection_metric = resolve_selection_metric(target_horizon)
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch_features, batch_targets, batch_start_positions = batch
            batch_features = batch_features.to(device, non_blocking=use_cuda)
            batch_targets = batch_targets.to(device, non_blocking=use_cuda)
            batch_start_positions = batch_start_positions.to(device, non_blocking=use_cuda)
            optimizer.zero_grad()
            action_logits = run_action_model(
                model,
                batch_features,
                batch_start_positions,
                target_actions=batch_targets,
            )
            loss = compute_training_loss(action_logits, batch_targets)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_model(
            model,
            train_features,
            train_targets,
            train_future_positions,
            train_self_position_history,
            valid_positions,
            device,
            args.batch_size,
            prediction_target=prediction_target,
        )
        val_metrics = evaluate_model(
            model,
            val_features,
            val_targets,
            val_future_positions,
            val_self_position_history,
            valid_positions,
            device,
            args.batch_size,
            prediction_target=prediction_target,
        )

        current_val_metric = val_metrics[selection_metric]
        improved = best_val_metric is None or current_val_metric > best_val_metric + args.early_stopping_min_delta

        if improved:
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_val_metric = current_val_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[epoch {epoch:02d}] "
            f"train_acc2H={train_metrics['avg_acc_2toH']:.4f} "
            f"val_acc2H={val_metrics['avg_acc_2toH']:.4f} "
            f"val_acc4={val_metrics.get('acc@4', float('nan')):.4f} "
            f"val_dist2H={val_metrics['avg_manhattan_2toH']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        print(f"  train_accs {format_horizon_metrics(train_metrics, 'acc', target_horizon)}")
        print(f"  val_accs   {format_horizon_metrics(val_metrics, 'acc', target_horizon)}")
        print(f"  train_action_accs {format_horizon_metrics(train_metrics, 'action_acc', target_horizon)}")
        print(f"  val_action_accs   {format_horizon_metrics(val_metrics, 'action_acc', target_horizon)}")

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            print(
                f"[early-stopping] epoch={epoch} best_epoch={best_epoch} "
                f"best_{selection_metric}={best_val_metric:.4f}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    final_train_metrics = evaluate_model(
        model,
        train_features,
        train_targets,
        train_future_positions,
        train_self_position_history,
        valid_positions,
        device,
        args.batch_size,
        prediction_target=prediction_target,
    )
    final_val_metrics = evaluate_model(
        model,
        val_features,
        val_targets,
        val_future_positions,
        val_self_position_history,
        valid_positions,
        device,
        args.batch_size,
        prediction_target=prediction_target,
    )
    final_test_metrics = evaluate_model(
        model,
        test_features,
        test_targets,
        test_future_positions,
        test_self_position_history,
        valid_positions,
        device,
        args.batch_size,
        prediction_target=prediction_target,
    )
    best_metrics = {
        "epoch": int(best_epoch),
        "selection_metric": selection_metric,
        "prediction_target": prediction_target,
        "predict_interact": predict_interact,
        "stopped_early": stopped_early,
        "train": final_train_metrics,
        "val": final_val_metrics,
        "test": final_test_metrics,
        "split": {
            "mode": split["mode"],
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "heldout_partner": split.get("heldout_partner"),
            "seen_split_mode": split.get("seen_split_mode"),
        },
    }

    checkpoint = {
        "model_state": best_state,
        "input_dim": int(features.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "history_len": int(dataset["self_position_history"].shape[1]),
        "obs_shape": list(dataset["obs"].shape[1:]),
        "target_horizon": int(target_horizon),
        "grid_width": int(dataset_scalar(dataset, "grid_width")),
        "grid_height": int(dataset_scalar(dataset, "grid_height")),
        "valid_positions": valid_positions.tolist(),
        "num_positions": int(dataset_scalar(dataset, "num_positions")),
        "num_actions": int(ACTION_DIM),
        "prediction_target": prediction_target,
        "action_model_type": model_action_model_type(model),
        "predict_interact": predict_interact,
        "split": best_metrics["split"],
        "best_metrics": best_metrics,
    }
    run_name = build_run_name(args, target_horizon)
    output_model_path = f"./predict_path/output/{run_name}.pt"

    ensure_parent_dir(output_model_path)

    torch.save(checkpoint, output_model_path)

    metrics_path = args.output_metrics_path or f"{output_model_path}.json"
    ensure_parent_dir(metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(best_metrics, metrics_file, indent=2)

    print(f"[best] train_accs {format_horizon_metrics(final_train_metrics, 'acc', target_horizon)}")
    print(f"[best] val_accs   {format_horizon_metrics(final_val_metrics, 'acc', target_horizon)}")
    print(f"[best] test_accs  {format_horizon_metrics(final_test_metrics, 'acc', target_horizon)}")
    print(f"[best] train_action_accs {format_horizon_metrics(final_train_metrics, 'action_acc', target_horizon)}")
    print(f"[best] val_action_accs   {format_horizon_metrics(final_val_metrics, 'action_acc', target_horizon)}")
    print(f"[best] test_action_accs  {format_horizon_metrics(final_test_metrics, 'action_acc', target_horizon)}")
    print(f"[saved] model={output_model_path}")
    print(f"[saved] metrics={metrics_path}")
    return {
        "output_model_path": output_model_path,
        "metrics_path": metrics_path,
        "best_metrics": best_metrics,
        "checkpoint": checkpoint,
    }


def run_baseline(args):
    dataset = load_dataset(args.dataset_path)
    split = resolve_split(dataset, args)
    metrics = compute_baselines(dataset, split)
    metrics["dataset_path"] = args.dataset_path
    if args.heldout_partner:
        metrics["heldout_partner"] = args.heldout_partner

    ensure_parent_dir(args.output_path)
    with open(args.output_path, "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[saved] baseline={args.output_path}")


def run_train(args):
    dataset = load_dataset(args.dataset_path)
    dataset_target_horizon = dataset_scalar(dataset, "target_horizon")
    dataset = slice_dataset_target_horizon(dataset, args.target_horizon)
    model_target_horizon = dataset_scalar(dataset, "target_horizon")
    features = build_dataset_features(dataset)
    split = resolve_split(dataset, args)
    baselines = compute_baselines(dataset, split)
    print(
        f"[split] mode={split['mode']} train={len(split['train_idx'])} "
        f"val={len(split['val_idx'])} test={len(split['test_idx'])}"
    )
    if args.heldout_partner:
        print(
            f"[heldout_partner] partner={args.heldout_partner} "
            f"seen_split_mode={split.get('seen_split_mode', 'n/a')} "
            f"available={','.join(split.get('available_partners', []))}"
        )
    print(
        f"[target_horizon] dataset={dataset_target_horizon} model={model_target_horizon}"
    )
    print(f"[hidden_dim] model={args.hidden_dim}")
    print(f"[action_model_type] model={ACTION_MODEL_TYPE_AUTOREGRESSIVE}")
    print(
        f"[baseline:val] stay_put_acc2H={baselines['val']['stay_put']['avg_acc_2toH']:.4f} "
        f"linear_acc2H={baselines['val']['linear_extrapolation']['avg_acc_2toH']:.4f}"
    )
    print(
        f"[baseline:test] stay_put_acc2H={baselines['test']['stay_put']['avg_acc_2toH']:.4f} "
        f"linear_acc2H={baselines['test']['linear_extrapolation']['avg_acc_2toH']:.4f}"
    )
    return train_model(args, dataset, features, split)


def run_train_policy_pool(args):
    dataset_path = f"./data/model_training/predict_path/{args.layout}_{args.target_samples}.npz"
    metadata_path = infer_metadata_path(dataset_path)

    if args.force_regenerate_dataset or not os.path.exists(dataset_path):
        export_args = argparse.Namespace(
            layout=args.layout,
            partner_agents=args.partner_agents,
            hsp_slots=args.hsp_slots,
            history_len=args.history_len,
            target_horizon=args.target_horizon,
            target_samples=args.target_samples,
            horizon=args.horizon,
            hsp_epsilon=args.hsp_epsilon,
            device=args.device,
            seed=args.seed,
            policy_pool_s1_only=args.policy_pool_s1_only,
        )
        export_result = run_policy_pool_export(export_args)
        dataset_path = export_result["output_path"]
        metadata_path = export_result["metadata_path"]
    else:
        print(f"[reuse] dataset={dataset_path}")
        if os.path.exists(metadata_path):
            print(f"[reuse] metadata={metadata_path}")

    train_args = argparse.Namespace(**vars(args))
    train_args.dataset_path = dataset_path
    train_args.heldout_partner = None
    if args.auto_select_hidden_dim:
        train_args.command = "compare"
        return run_compare(train_args)

    train_args.command = "train"
    return run_train(train_args)


def run_compare(args):
    if args.heldout_partner:
        raise ValueError("compare does not support --heldout-partner. Use lopo for partner holdout experiments.")

    dataset = load_dataset(args.dataset_path)
    dataset_target_horizon = dataset_scalar(dataset, "target_horizon")
    dataset = slice_dataset_target_horizon(dataset, args.target_horizon)
    model_target_horizon = dataset_scalar(dataset, "target_horizon")
    features = build_dataset_features(dataset)
    split = resolve_split(dataset, args)
    baselines = compute_baselines(dataset, split)
    hidden_dim_candidates = parse_hidden_dim_candidates(args.hidden_dim_candidates)
    selection_metric = resolve_selection_metric(model_target_horizon)

    print(
        f"[split] mode={split['mode']} train={len(split['train_idx'])} "
        f"val={len(split['val_idx'])} test={len(split['test_idx'])}"
    )
    print(
        f"[target_horizon] dataset={dataset_target_horizon} model={model_target_horizon}"
    )
    print(f"[compare] hidden_dims={hidden_dim_candidates}")
    print(f"[action_model_type] model={ACTION_MODEL_TYPE_AUTOREGRESSIVE}")
    print(
        f"[baseline:val] stay_put_acc2H={baselines['val']['stay_put']['avg_acc_2toH']:.4f} "
        f"linear_acc2H={baselines['val']['linear_extrapolation']['avg_acc_2toH']:.4f}"
    )
    print(
        f"[baseline:test] stay_put_acc2H={baselines['test']['stay_put']['avg_acc_2toH']:.4f} "
        f"linear_acc2H={baselines['test']['linear_extrapolation']['avg_acc_2toH']:.4f}"
    )

    results = []
    for hidden_dim in hidden_dim_candidates:
        print(f"[compare] start hidden_dim={hidden_dim}")
        model_args = argparse.Namespace(**vars(args))
        model_args.command = "train"
        model_args.hidden_dim = hidden_dim
        model_args.run_tag = f"hd{hidden_dim}"
        model_args.output_metrics_path = None
        result = train_model(model_args, dataset, features, split)
        metrics = result["best_metrics"]
        results.append(
            {
                "hidden_dim": hidden_dim,
                "best_epoch": int(metrics["epoch"]),
                "selection_metric": selection_metric,
                "val_selection_score": float(metrics["val"][selection_metric]),
                "test_selection_score": float(metrics["test"][selection_metric]),
                "val_acc@4": float(metrics["val"].get("acc@4", metrics["val"][selection_metric])),
                "test_acc@4": float(metrics["test"].get("acc@4", metrics["test"][selection_metric])),
                "test_avg_acc_2toH": float(metrics["test"]["avg_acc_2toH"]),
                "test_avg_action_acc_2toH": float(metrics["test"]["avg_action_acc_2toH"]),
                "test_avg_manhattan_2toH": float(metrics["test"]["avg_manhattan_2toH"]),
                "model_path": result["output_model_path"],
                "metrics_path": result["metrics_path"],
            }
        )

    best_result = max(
        results,
        key=lambda result: (
            result["val_selection_score"],
            result["test_avg_acc_2toH"],
            -result["hidden_dim"],
        ),
    )
    selected_model_path = getattr(args, "selected_model_path", None)
    if selected_model_path:
        ensure_parent_dir(selected_model_path)
        shutil.copy2(best_result["model_path"], selected_model_path)
        best_result["selected_model_path"] = selected_model_path
        print(f"[saved] selected_model={selected_model_path}")
    summary = {
        "dataset_path": args.dataset_path,
        "dataset_target_horizon": int(dataset_target_horizon),
        "model_target_horizon": int(model_target_horizon),
        "selection_metric": f"val {selection_metric}",
        "hidden_dim_candidates": hidden_dim_candidates,
        "split": {
            "mode": split["mode"],
            "num_train": int(len(split["train_idx"])),
            "num_val": int(len(split["val_idx"])),
            "num_test": int(len(split["test_idx"])),
        },
        "results": results,
        "selected_hidden_dim": int(best_result["hidden_dim"]),
        "selected_result": best_result,
    }

    summary_path = args.output_summary_path or f"./predict_path/output/{build_run_name(args, model_target_horizon)}_compare_summary.json"
    ensure_parent_dir(summary_path)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(
        f"[compare] selected_hidden_dim={best_result['hidden_dim']} "
        f"val_{selection_metric}={best_result['val_selection_score']:.4f} "
        f"test_acc4={best_result['test_acc@4']:.4f}"
    )
    print(f"[saved] compare_summary={summary_path}")
    return summary


def summarize_scalar_list(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def resolve_selection_metric(target_horizon):
    if target_horizon >= 4:
        return "acc@4"
    if target_horizon >= 2:
        return "avg_acc_2toH"
    return "avg_acc_1toH"


def run_lopo(args):
    dataset = load_dataset(args.dataset_path)
    dataset_target_horizon = dataset_scalar(dataset, "target_horizon")
    dataset = slice_dataset_target_horizon(dataset, args.target_horizon)
    model_target_horizon = dataset_scalar(dataset, "target_horizon")
    if model_target_horizon < 4:
        raise ValueError("LOPO hidden-dim selection with acc@4 requires target_horizon >= 4.")
    partners = get_available_partners(args.dataset_path)
    if not partners:
        raise RuntimeError("No partner metadata was found for LOPO.")
    hidden_dim_candidates = parse_hidden_dim_candidates(args.hidden_dim_candidates)

    print(f"[lopo] partners={','.join(partners)}")
    print(f"[lopo] target_horizon dataset={dataset_target_horizon} model={model_target_horizon}")

    print(f"[lopo] hidden_dims={hidden_dim_candidates}")
    hidden_dim_results = []
    for hidden_dim in hidden_dim_candidates:
        print(f"[lopo] start hidden_dim={hidden_dim}")
        fold_results = []
        for partner in partners:
            print(f"[lopo] start heldout_partner={partner} hidden_dim={hidden_dim}")
            fold_args = argparse.Namespace(**vars(args))
            fold_args.command = "train"
            fold_args.hidden_dim = hidden_dim
            fold_args.heldout_partner = partner
            fold_args.run_tag = f"hd{hidden_dim}"
            fold_args.output_metrics_path = None
            result = run_train(fold_args)
            metrics = result["best_metrics"]
            fold_results.append(
                {
                    "heldout_partner": partner,
                    "hidden_dim": hidden_dim,
                    "best_epoch": int(metrics["epoch"]),
                    "val_acc@4": float(metrics["val"]["acc@4"]),
                    "test_acc@4": float(metrics["test"]["acc@4"]),
                    "test_avg_acc_2toH": float(metrics["test"]["avg_acc_2toH"]),
                    "test_avg_action_acc_2toH": float(metrics["test"]["avg_action_acc_2toH"]),
                    "test_avg_manhattan_2toH": float(metrics["test"]["avg_manhattan_2toH"]),
                    "model_path": result["output_model_path"],
                    "metrics_path": result["metrics_path"],
                }
            )

        best_epochs = [fold["best_epoch"] for fold in fold_results]
        test_acc4_values = [fold["test_acc@4"] for fold in fold_results]
        hidden_dim_results.append(
            {
                "hidden_dim": hidden_dim,
                "folds": fold_results,
                "best_epoch_stats": summarize_scalar_list(best_epochs),
                "test_acc@4_stats": summarize_scalar_list(test_acc4_values),
                "test_avg_acc_2toH_stats": summarize_scalar_list([fold["test_avg_acc_2toH"] for fold in fold_results]),
                "test_avg_action_acc_2toH_stats": summarize_scalar_list([fold["test_avg_action_acc_2toH"] for fold in fold_results]),
                "test_avg_manhattan_2toH_stats": summarize_scalar_list([fold["test_avg_manhattan_2toH"] for fold in fold_results]),
                "recommended_final_epoch": max(1, int(round(float(np.median(best_epochs))))),
                "selection_score_test_acc@4_mean": float(np.mean(test_acc4_values)),
            }
        )

    best_hidden_dim_result = max(
        hidden_dim_results,
        key=lambda result: (
            result["selection_score_test_acc@4_mean"],
            result["test_avg_acc_2toH_stats"]["mean"],
            -result["hidden_dim"],
        ),
    )
    best_hidden_dim = int(best_hidden_dim_result["hidden_dim"])
    recommended_final_epoch = int(best_hidden_dim_result["recommended_final_epoch"])
    summary = {
        "dataset_path": args.dataset_path,
        "dataset_target_horizon": int(dataset_target_horizon),
        "model_target_horizon": int(model_target_horizon),
        "partners": partners,
        "hidden_dim_candidates": hidden_dim_candidates,
        "selection_metric": "mean held-out test acc@4",
        "hidden_dim_results": hidden_dim_results,
        "selected_hidden_dim": best_hidden_dim,
        "selected_hidden_dim_result": best_hidden_dim_result,
        "recommended_final_epoch": recommended_final_epoch,
        "final_training": None,
    }

    if not args.skip_final_train:
        print(
            f"[lopo] final mixed-partner training with hidden_dim={best_hidden_dim} "
            f"epochs={recommended_final_epoch} (selected by mean held-out test acc@4)"
        )
        final_args = argparse.Namespace(**vars(args))
        final_args.command = "train"
        final_args.hidden_dim = best_hidden_dim
        final_args.heldout_partner = None
        final_args.run_tag = f"hd{best_hidden_dim}_final-after-lopo"
        final_args.epochs = recommended_final_epoch
        final_args.early_stopping_patience = 0
        final_result = run_train(final_args)
        summary["final_training"] = {
            "hidden_dim": best_hidden_dim,
            "epochs": recommended_final_epoch,
            "model_path": final_result["output_model_path"],
            "metrics_path": final_result["metrics_path"],
            "best_epoch": int(final_result["best_metrics"]["epoch"]),
            "val_acc@4": float(final_result["best_metrics"]["val"]["acc@4"]),
            "test_acc@4": float(final_result["best_metrics"]["test"]["acc@4"]),
            "test_avg_acc_2toH": float(final_result["best_metrics"]["test"]["avg_acc_2toH"]),
            "test_avg_action_acc_2toH": float(final_result["best_metrics"]["test"]["avg_action_acc_2toH"]),
        }

    summary_path = args.output_summary_path or f"./predict_path/output/{build_run_name(args, model_target_horizon)}_lopo_summary.json"
    ensure_parent_dir(summary_path)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"[saved] lopo_summary={summary_path}")
    return summary


def main():
    args = parse_args()
    if args.command == "train" or args.command is None:
        run_train(args)
        return
    if args.command == "train_policy_pool":
        run_train_policy_pool(args)
        return
    if args.command == "baseline":
        run_baseline(args)
        return
    if args.command == "compare":
        run_compare(args)
        return
    if args.command == "lopo":
        run_lopo(args)
        return
    
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
