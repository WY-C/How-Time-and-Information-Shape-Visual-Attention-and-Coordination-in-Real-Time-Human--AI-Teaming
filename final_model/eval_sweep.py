#!/usr/bin/env python3
"""Evaluate two/three horizon runs or one canonical HSP S2 baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm


PROJECT_PARENT = Path(__file__).resolve().parents[1]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from final_model.checkpoint import ACTION_SPACE, FORMAT_ID, load_action_checkpoint, load_gaze_checkpoint
from final_model.common.mappo import (
    ACTION_ARCHITECTURE_ID, FULL_OBSERVATION_CONTRACT, PARTIAL_OBSERVATION_CONTRACT,
    build_actionnet,
)
from final_model.common.overcooked_backend import (
    CANONICAL_ENCODER_ID,
    DirectOvercookedBackend,
    cooking_contract,
)
from final_model.common.partners import PARTNER_SET_ID, parameter_hash, partner_manifest
from final_model.common.zsceval_compat import install_zsceval, load_policy_config
from final_model.eval import EvaluationGazeController
from final_model.partial_ver.network import GAZE_ARCHITECTURE_ID, GazeNet
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.peripheral import PERIPHERAL_RNG_SEED_OFFSET
from final_model.partial_ver.preview_opacity import PREVIEW_OPACITY_RNG_SEED_OFFSET
from final_model.partial_ver.constraints import constraint_config_from_metadata
from final_model.partial_ver.gaze import HUMAN_DWELL_RNG_SEED_OFFSET
from final_model.partial_ver.memory import MEMORY_RNG_SEED_OFFSET
from final_model.partial_ver.rule_based import RULE_BASED_GAZE_POLICY_ID
from final_model.simple_ver.observation import FullObservationAdapter


MILESTONE_INTERVAL = 1_000_000
BASELINE_RUN_LABEL = "hsp_s2_baseline"
BASELINE_OBSERVATION_MODE = "canonical_20ch"
RUN_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")
HORIZON_LABEL = re.compile(r"^h([024])$")
EPISODE_FIELDS = (
    "run_label", "checkpoint_role", "milestone_target_step", "checkpoint_actual_step",
    "checkpoint_step", "cross_run_actual_step_status", "checkpoint_path",
    "observation_mode", "training_stage", "horizon", "decision_time",
    "fixation_shape", "partner", "partner_family", "episode_index",
    "evaluation_seed", "episode_length", "steps", "sparse_team_reward", "deliveries",
    "native_shaped_reward", "optimization_reward",
)
SUMMARY_FIELDS = (
    "run_label", "status", "checkpoint_role", "milestone_target_step",
    "checkpoint_actual_step", "checkpoint_step", "cross_run_actual_step_status", "checkpoint_path",
    "observation_mode", "training_stage", "horizon", "decision_time",
    "fixation_shape", "episode_count", "partner_count", "mean_sparse_team_reward",
    "std_sparse_team_reward", "mean_deliveries", "std_deliveries",
    "mep_mean_sparse_team_reward", "hsp_mean_sparse_team_reward",
    "mep_mean_deliveries", "hsp_mean_deliveries",
)


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    metadata: dict[str, Any]

    @property
    def step(self) -> int | None:
        value = self.metadata.get("step")
        return None if value is None else int(value)


@dataclass(frozen=True)
class EvaluationTarget:
    run_label: str
    checkpoint: CheckpointRecord
    role: str
    episodes_per_partner: int
    milestone_target_step: int | None = None


@dataclass
class RunPlan:
    label: str
    checkpoint_dir: Path
    horizon: int
    configured_steps: int | None
    rollout_length: int
    training_episode_length: int | None
    targets: list[EvaluationTarget]
    schedule_rows: list[dict[str, Any]]
    ignored_checkpoint_steps: list[int]
    final_status: str


class CanonicalHSPAdapter:
    """Evaluation-only adapter for an unmodified canonical HSP S2 learner."""

    def __init__(self, backend: DirectOvercookedBackend):
        self.backend = backend
        self.actionnet_online_forwards = 0
        self.task_decisions = 0

    def reset_episode(self) -> None:
        self.backend.reset()

    def current(self) -> tuple[np.ndarray, None, np.ndarray]:
        observation, available = self.backend.canonical_learner_input()
        if observation.shape != (8, 5, 20):
            raise RuntimeError("baseline learner must receive canonical 20-channel observation")
        if available.shape != (6,):
            raise RuntimeError("baseline learner legal-action mask contract changed")
        return observation, None, available

    def run_decision(self, policy):
        observation, critic, available = self.current()
        action = int(policy(observation, critic, available))
        self.actionnet_online_forwards += 1
        self.task_decisions += 1
        return observation, critic, available, action, self.backend.step(action)


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not RUN_LABEL.fullmatch(label) or not raw_path:
        raise argparse.ArgumentTypeError("--run must be LABEL=CHECKPOINT_DIRECTORY")
    return label, Path(raw_path).expanduser().resolve()


def validate_run_arguments(runs: list[tuple[str, Path]]) -> None:
    if len(runs) not in {2, 3}:
        raise ValueError("exactly two or three --run arguments are required")
    if len({label for label, _path in runs}) != len(runs):
        raise ValueError("--run labels must be unique")


def evaluation_seed_set(base_seed: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("evaluation seed count must be positive")
    return tuple(int(base_seed) + index for index in range(count))


def checkpoint_directory(path: Path) -> Path:
    if path.is_dir() and path.name == "checkpoints":
        return path
    candidate = path / "checkpoints"
    if candidate.is_dir():
        return candidate
    raise RuntimeError(f"run path has no checkpoint directory: {path}")


def load_hsp_s2_actor(
    checkpoint: Path, *, device: torch.device, construction_seed: int
) -> torch.nn.Module:
    path = checkpoint.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"baseline checkpoint does not exist: {path}")
    install_zsceval()
    from final_model.common.actor_state import load_actor_state_dict
    from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

    args, observation_space, _share_space, action_space = load_policy_config("rnn")
    if tuple(observation_space.shape) != (8, 5, 20) or int(action_space.n) != 6:
        raise RuntimeError("canonical HSP S2 policy config contract changed")
    torch.manual_seed(int(construction_seed))
    actor = R_Actor(args, observation_space, action_space, device=device)
    actor.load_state_dict(load_actor_state_dict(path), strict=True)
    if (
        actor.base.cnn[0].in_channels != 20
        or int(args.recurrent_N) != 1
        or int(args.hidden_size) != 64
    ):
        raise RuntimeError("baseline checkpoint is not a canonical 20ch GRU64 HSP actor")
    return actor.eval().requires_grad_(False)


def baseline_run_plan(checkpoint: Path) -> RunPlan:
    path = checkpoint.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"baseline checkpoint does not exist: {path}")
    record = CheckpointRecord(
        path,
        {
            "learner_kind": "hsp_s2_baseline",
            "observation_mode": BASELINE_OBSERVATION_MODE,
            "training_stage": "baseline",
            "horizon": 0,
            "decision_time": None,
            "fixation_shape": None,
            "step": None,
            "observation_contract_id": CANONICAL_ENCODER_ID,
            "partner_set_id": PARTNER_SET_ID,
            "action_space": list(ACTION_SPACE),
        },
    )
    target = EvaluationTarget(BASELINE_RUN_LABEL, record, "final", 10)
    row = schedule_row(BASELINE_RUN_LABEL, "BASELINE_FINAL", "final", None, record)
    row["cross_run_actual_step_status"] = "NOT_APPLICABLE"
    return RunPlan(
        label=BASELINE_RUN_LABEL,
        checkpoint_dir=path.parent,
        horizon=0,
        configured_steps=None,
        rollout_length=0,
        training_episode_length=None,
        targets=[target],
        schedule_rows=[row],
        ignored_checkpoint_steps=[],
        final_status="BASELINE_FINAL",
    )


def raw_checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"cannot read checkpoint {path}: {error}") from error
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise RuntimeError(f"checkpoint has no metadata mapping: {path}")
    return dict(metadata)


def validate_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    required = {
        "format_id", "training_stage", "observation_mode", "horizon", "step",
        "observation_contract_id", "partner_set_id",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise RuntimeError(f"checkpoint metadata missing {missing}: {path}")
    if metadata["training_stage"] not in {"action", "gaze"}:
        raise RuntimeError(f"invalid training_stage in {path}")
    if metadata["observation_mode"] not in {"full", "partial"}:
        raise RuntimeError(f"invalid observation_mode in {path}")
    if metadata["training_stage"] == "gaze" and metadata["observation_mode"] != "partial":
        raise RuntimeError(f"gaze checkpoint must use partial observation: {path}")
    if metadata["horizon"] not in {0, 2, 4}:
        raise RuntimeError(f"invalid horizon in {path}")
    if isinstance(metadata["step"], bool) or not isinstance(metadata["step"], int):
        raise RuntimeError(f"checkpoint step must be an integer: {path}")
    if metadata["step"] <= 0:
        raise RuntimeError(f"checkpoint step must be positive: {path}")
    expected_common = {
        "format_id": FORMAT_ID,
        "canonical_encoder_id": CANONICAL_ENCODER_ID,
        "partner_set_id": PARTNER_SET_ID,
        "action_space": list(ACTION_SPACE),
        "observation_contract_id": (
            FULL_OBSERVATION_CONTRACT
            if metadata["observation_mode"] == "full"
            else PARTIAL_OBSERVATION_CONTRACT
        ),
        **cooking_contract(),
    }
    for key, expected in expected_common.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"checkpoint contract mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected!r} ({path})"
            )
    if metadata["training_stage"] == "action":
        if metadata.get("architecture_id") != ACTION_ARCHITECTURE_ID:
            raise RuntimeError(f"Action architecture mismatch in {path}")
    elif (
        metadata.get("action_architecture_id") != ACTION_ARCHITECTURE_ID
        or metadata.get("gaze_architecture_id") != GAZE_ARCHITECTURE_ID
        or metadata.get("actionnet_frozen") is not True
        or metadata.get("action_optimizer_steps") != 0
    ):
        raise RuntimeError(f"Gaze checkpoint frozen architecture contract mismatch in {path}")
    if metadata["observation_mode"] == "partial":
        if metadata.get("decision_time") not in {200, 400, 600}:
            raise RuntimeError(f"invalid Partial decision_time in {path}")
        if metadata.get("fixation_shape") not in {"3x3", "2x2", "plus"}:
            raise RuntimeError(f"invalid Partial fixation_shape in {path}")
        if (
            metadata["training_stage"] == "action"
            and metadata.get("gaze_policy") != RULE_BASED_GAZE_POLICY_ID
        ):
            raise RuntimeError(f"Partial Action gaze policy mismatch in {path}")
    elif metadata.get("decision_time") is not None or metadata.get("fixation_shape") is not None:
        raise RuntimeError(f"Full checkpoint contains Partial condition metadata: {path}")


def condition_signature(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        metadata.get(key)
        for key in (
            "format_id", "observation_mode", "training_stage", "horizon",
            "decision_time", "fixation_shape", "observation_contract_id", "partner_set_id",
            "cooking_dynamics_id", "onion_cook_ticks_per_environment_step",
            "soup_cooking_time", "soup_items_required",
        )
    )


def load_run_config(checkpoint_dir: Path) -> dict[str, Any] | None:
    path = checkpoint_dir.parent / "config.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"invalid run config {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"run config must be a JSON object: {path}")
    return value


def validate_config(config: dict[str, Any], metadata: dict[str, Any], path: Path) -> tuple[int, int]:
    for key in ("observation_mode", "training_stage", "horizon", "decision_time", "fixation_shape"):
        if config.get(key) != metadata.get(key):
            raise RuntimeError(
                f"run config/checkpoint mismatch for {key}: "
                f"{config.get(key)!r} != {metadata.get(key)!r} ({path})"
            )
    steps = config.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise RuntimeError(f"run config has invalid total steps: {path}")
    rollout_length = config.get("rollout_length")
    if (
        isinstance(rollout_length, bool)
        or not isinstance(rollout_length, int)
        or rollout_length <= 0
    ):
        raise RuntimeError(f"run config has invalid rollout_length: {path}")
    return steps, rollout_length


def schedule_row(
    label: str,
    status: str,
    role: str,
    milestone_target_step: int | None,
    checkpoint: CheckpointRecord | None,
) -> dict[str, Any]:
    metadata = {} if checkpoint is None else checkpoint.metadata
    return {
        "run_label": label,
        "status": status,
        "checkpoint_role": role,
        "milestone_target_step": milestone_target_step,
        "checkpoint_actual_step": None if checkpoint is None else checkpoint.step,
        "checkpoint_step": None if checkpoint is None else checkpoint.step,
        "checkpoint_path": "" if checkpoint is None else str(checkpoint.path),
        "observation_mode": metadata.get("observation_mode", ""),
        "training_stage": metadata.get("training_stage", ""),
        "horizon": metadata.get("horizon", ""),
        "decision_time": metadata.get("decision_time", ""),
        "fixation_shape": metadata.get("fixation_shape", ""),
    }


def discover_run(
    label: str,
    raw_path: Path,
    *,
    milestone_episodes: int,
    final_episodes: int,
    allow_latest_as_final: bool,
) -> RunPlan:
    directory = checkpoint_directory(raw_path)
    paths = sorted(directory.glob("*.pt"))
    if not paths:
        raise RuntimeError(f"no .pt checkpoints found in {directory}")
    records = []
    for path in paths:
        metadata = raw_checkpoint_metadata(path)
        validate_checkpoint_metadata(path, metadata)
        records.append(CheckpointRecord(path.resolve(), metadata))
    signatures = {condition_signature(record.metadata) for record in records}
    if len(signatures) != 1:
        raise RuntimeError(f"run contains mixed checkpoint conditions: {directory}")
    by_step: dict[int, CheckpointRecord] = {}
    for record in records:
        if record.step in by_step:
            raise RuntimeError(f"duplicate checkpoint metadata step {record.step}: {directory}")
        by_step[record.step] = record

    config = load_run_config(directory)
    configured_steps = None
    rollout_length = None
    training_episode_length = None
    if config is not None:
        configured_steps, rollout_length = validate_config(
            config, records[0].metadata, directory.parent / "config.json"
        )
        configured_episode_length = config.get("episode_length")
        if configured_episode_length is not None:
            if (
                isinstance(configured_episode_length, bool)
                or not isinstance(configured_episode_length, int)
                or configured_episode_length <= 0
            ):
                raise RuntimeError(
                    f"run config has invalid episode_length: {directory.parent / 'config.json'}"
                )
            training_episode_length = configured_episode_length
        if any(step > configured_steps for step in by_step):
            raise RuntimeError(f"checkpoint step exceeds configured total steps: {directory}")
    elif not allow_latest_as_final:
        raise RuntimeError(
            f"missing config.json for canonical final validation: {directory.parent}"
        )
    if rollout_length is None:
        raise RuntimeError(
            f"missing config rollout_length for milestone selection: {directory.parent}"
        )

    final_record = by_step.get(configured_steps) if configured_steps is not None else None
    if final_record is not None:
        final_status = "CANONICAL_FINAL"
    elif allow_latest_as_final:
        final_record = by_step[max(by_step)]
        final_status = "LATEST_AS_FINAL_OVERRIDE"
    else:
        final_status = "FINAL_CHECKPOINT_INCOMPLETE"

    schedule_limit = configured_steps if configured_steps is not None else max(by_step)
    targets: list[EvaluationTarget] = []
    schedule_rows: list[dict[str, Any]] = []
    used_steps: set[int] = set()
    for step in range(MILESTONE_INTERVAL, schedule_limit, MILESTONE_INTERVAL):
        candidates = [
            record
            for actual_step, record in by_step.items()
            if step <= actual_step < step + rollout_length
        ]
        checkpoint = min(candidates, key=lambda record: record.step) if candidates else None
        if final_record is not None and checkpoint is not None and checkpoint.step == final_record.step:
            continue
        if checkpoint is None:
            schedule_rows.append(schedule_row(label, "MISSING_MILESTONE", "milestone", step, None))
            continue
        targets.append(EvaluationTarget(label, checkpoint, "milestone", milestone_episodes, step))
        schedule_rows.append(schedule_row(label, "SCHEDULED", "milestone", step, checkpoint))
        used_steps.add(checkpoint.step)

    if final_record is None:
        schedule_rows.append(
            schedule_row(label, "FINAL_CHECKPOINT_INCOMPLETE", "final", None, None)
        )
    else:
        targets.append(EvaluationTarget(label, final_record, "final", final_episodes))
        schedule_rows.append(
            schedule_row(label, final_status, "final", None, final_record)
        )
        used_steps.add(final_record.step)
    condition_metadata = records[0].metadata
    for row in schedule_rows:
        for key in (
            "observation_mode", "training_stage", "horizon", "decision_time", "fixation_shape"
        ):
            if row[key] == "":
                value = condition_metadata.get(key)
                row[key] = "" if value is None else value
    ignored = sorted(set(by_step) - used_steps)
    return RunPlan(
        label=label,
        checkpoint_dir=directory,
        horizon=int(records[0].metadata["horizon"]),
        configured_steps=configured_steps,
        rollout_length=rollout_length,
        training_episode_length=training_episode_length,
        targets=targets,
        schedule_rows=schedule_rows,
        ignored_checkpoint_steps=ignored,
        final_status=final_status,
    )


class CheckpointEvaluator:
    def __init__(self, checkpoint: CheckpointRecord, *, device: str, construction_seed: int):
        self.checkpoint = checkpoint
        self.metadata = dict(checkpoint.metadata)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.gaze_actor = None
        mode = self.metadata["observation_mode"]
        stage = self.metadata["training_stage"]
        if stage == "baseline":
            self.action_actor = load_hsp_s2_actor(
                checkpoint.path,
                device=self.device,
                construction_seed=construction_seed,
            )
            loaded = {}
        else:
            self.action_actor = build_actionnet(seed=construction_seed, device=self.device)
        if stage == "action":
            loaded = load_action_checkpoint(
                checkpoint.path,
                actor=self.action_actor,
                expected_mode=mode,
                expected_horizon=self.metadata["horizon"],
                expected_condition=(
                    {
                        "decision_time": self.metadata["decision_time"],
                        "fixation_shape": self.metadata["fixation_shape"],
                        "gaze_policy": RULE_BASED_GAZE_POLICY_ID,
                    }
                    if mode == "partial"
                    else None
                ),
            )
        elif stage == "gaze":
            self.gaze_actor = GazeNet().to(self.device)
            loaded = load_gaze_checkpoint(
                checkpoint.path,
                action_actor=self.action_actor,
                gaze_actor=self.gaze_actor,
                horizon=self.metadata["horizon"],
                decision_time=self.metadata["decision_time"],
                fixation_shape=self.metadata["fixation_shape"],
            )
            self.gaze_actor.eval().requires_grad_(False)
        for key, value in self.metadata.items():
            if key in loaded and loaded[key] != value:
                raise RuntimeError(f"checkpoint changed while loading: {checkpoint.path} ({key})")
        if mode == "partial" and stage != "baseline":
            self.metadata.update(
                constraint_config_from_metadata(loaded).metadata()
            )
        self.action_actor.eval().requires_grad_(False)
        self.action_hash = parameter_hash(self.action_actor)

    def initial_recurrent_state(self) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.zeros(
            (
                1,
                int(self.action_actor._recurrent_N),
                int(self.action_actor.hidden_size),
            ),
            dtype=np.float32,
        )
        return hidden, np.zeros((1, 1), dtype=np.float32)

    def deterministic_action(
        self,
        observation: np.ndarray,
        available: np.ndarray,
        hidden: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        with torch.no_grad():
            probabilities, next_hidden = self.action_actor.get_probs(
                observation[None], hidden, mask, available[None]
            )
        return int(probabilities.argmax(-1).item()), next_hidden.detach().cpu().numpy()

    def _adapter(
        self, backend: DirectOvercookedBackend, *, peripheral_seed: int,
        opacity_seed: int, dwell_seed: int,
    ):
        if self.metadata["training_stage"] == "baseline":
            return CanonicalHSPAdapter(backend)
        if self.metadata["observation_mode"] == "full":
            return FullObservationAdapter(backend)
        controller = None
        if self.gaze_actor is not None:
            controller = EvaluationGazeController(self.gaze_actor, None, self.device)
        constraint_config = constraint_config_from_metadata(self.metadata)
        return PartialObservationAdapter(
            backend,
            decision_time=int(self.metadata["decision_time"]),
            fixation_shape=str(self.metadata["fixation_shape"]),
            gaze_policy=controller,
            memory_lifetime_ms=self.metadata.get("memory_lifetime_ms", 2400.0),
            memory_decay_mode=self.metadata.get("memory_decay_mode", "linear"),
            memory_tau_ms=self.metadata.get("memory_tau_ms"),
            memory_seed=(
                peripheral_seed - PERIPHERAL_RNG_SEED_OFFSET
                + MEMORY_RNG_SEED_OFFSET
                if self.metadata.get("memory_decay_mode")
                == "stochastic_exponential"
                else None
            ),
            constraint_config=constraint_config,
            peripheral_seed=peripheral_seed,
            opacity_seed=opacity_seed,
            dwell_seed=(
                dwell_seed
                if constraint_config.emma.mode == "human_stochastic"
                else None
            ),
            ground_truth_policy_input=bool(
                self.metadata.get("ground_truth_policy_input", False)
            ),
        )

    @staticmethod
    def _reseed_partner(backend: DirectOvercookedBackend, evaluation_seed: int) -> None:
        backend.seed = int(evaluation_seed)
        backend.partners.rng = np.random.default_rng(int(evaluation_seed))
        backend.partners.torch_rng = torch.Generator(
            device=backend.partners.device.type
        ).manual_seed(int(evaluation_seed) + 91)

    def evaluate_partner(
        self,
        *,
        partner: str,
        evaluation_seeds: Iterable[int],
        episode_length: int,
    ) -> list[dict[str, Any]]:
        seeds = tuple(int(seed) for seed in evaluation_seeds)
        if not seeds:
            return []
        backend = DirectOvercookedBackend(
            seed=seeds[0],
            horizon=int(self.metadata["horizon"]),
            episode_length=int(episode_length),
            device=self.device,
            partner=partner,
        )
        adapter = self._adapter(
            backend,
            peripheral_seed=seeds[0] + PERIPHERAL_RNG_SEED_OFFSET,
            opacity_seed=seeds[0] + PREVIEW_OPACITY_RNG_SEED_OFFSET,
            dwell_seed=seeds[0] + HUMAN_DWELL_RNG_SEED_OFFSET,
        )
        rows = []
        for episode_index, evaluation_seed in enumerate(seeds):
            if episode_index:
                self._reseed_partner(backend, evaluation_seed)
                adapter.reset_episode()
            backend.verify_partner_parameters()
            hidden, mask = self.initial_recurrent_state()
            sparse = shaped = optimization = 0.0
            deliveries = steps = 0
            while True:
                def policy(observation, _critic, available):
                    nonlocal hidden, mask
                    action, hidden = self.deterministic_action(
                        observation, available, hidden, mask
                    )
                    mask.fill(1.0)
                    return action

                transition = adapter.run_decision(policy)
                result = (
                    transition[-1]
                    if self.metadata["observation_mode"] == "full"
                    or self.metadata["training_stage"] == "baseline"
                    else transition.result
                )
                sparse += float(result["sparse_team_reward"])
                shaped += float(result["native_shaped_reward"])
                optimization += float(result["optimization_reward"])
                deliveries += int(result["deliveries"])
                steps += 1
                if result["done"]:
                    break
            backend.verify_partner_parameters()
            if steps != episode_length:
                raise RuntimeError(f"episode length contract failed: {steps} != {episode_length}")
            rows.append(
                {
                    "episode_index": episode_index,
                    "evaluation_seed": evaluation_seed,
                    "steps": steps,
                    "sparse_team_reward": sparse,
                    "deliveries": deliveries,
                    "native_shaped_reward": shaped,
                    "optimization_reward": optimization,
                }
            )
        return rows

    def verify_unchanged(self) -> None:
        if parameter_hash(self.action_actor) != self.action_hash:
            raise RuntimeError(f"evaluation mutated ActionNet: {self.checkpoint.path}")


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_target(target: EvaluationTarget, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sparse = [float(row["sparse_team_reward"]) for row in rows]
    deliveries = [float(row["deliveries"]) for row in rows]
    families = {
        family: [row for row in rows if row["partner_family"] == family]
        for family in ("mep", "hsp")
    }
    partner_means = {}
    for partner in sorted({str(row["partner"]) for row in rows}):
        selected = [row for row in rows if row["partner"] == partner]
        partner_means[partner] = {
            "mean_sparse_team_reward": float(
                np.mean([row["sparse_team_reward"] for row in selected])
            ),
            "mean_deliveries": float(np.mean([row["deliveries"] for row in selected])),
            "episode_count": len(selected),
            "partner_family": selected[0]["partner_family"],
        }
    value = schedule_row(
        target.run_label, "EVALUATED", target.role, target.milestone_target_step,
        target.checkpoint,
    )
    value.update(
        {
            "episode_count": len(rows),
            "partner_count": len(partner_means),
            "mean_sparse_team_reward": float(np.mean(sparse)),
            "std_sparse_team_reward": sample_std(sparse),
            "mean_deliveries": float(np.mean(deliveries)),
            "std_deliveries": sample_std(deliveries),
            "mep_mean_sparse_team_reward": float(
                np.mean([row["sparse_team_reward"] for row in families["mep"]])
            ),
            "hsp_mean_sparse_team_reward": float(
                np.mean([row["sparse_team_reward"] for row in families["hsp"]])
            ),
            "mep_mean_deliveries": float(
                np.mean([row["deliveries"] for row in families["mep"]])
            ),
            "hsp_mean_deliveries": float(
                np.mean([row["deliveries"] for row in families["hsp"]])
            ),
            "partner_means": partner_means,
        }
    )
    return value


def write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cross_run_actual_step_validation(
    plans: list[RunPlan],
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    targets_by_run = [
        {
            target.milestone_target_step: target.checkpoint.step
            for target in plan.targets
            if target.role == "milestone"
        }
        for plan in plans
    ]
    milestone_targets = sorted(
        {
            row["milestone_target_step"]
            for plan in plans
            for row in plan.schedule_rows
            if row["checkpoint_role"] == "milestone"
        }
    )
    statuses = {}
    comparisons = []
    for target_step in milestone_targets:
        actual_steps = [mapping.get(target_step) for mapping in targets_by_run]
        if any(step is None for step in actual_steps):
            status = "UNAVAILABLE"
        elif len(set(actual_steps)) == 1:
            status = "MATCH"
        else:
            status = "MISMATCH"
        statuses[target_step] = status
        comparisons.append(
            {
                "milestone_target_step": target_step,
                "cross_run_actual_step_status": status,
                "actual_steps_by_run": {
                    plan.label: actual_step
                    for plan, actual_step in zip(plans, actual_steps)
                },
            }
        )
    for plan in plans:
        for row in plan.schedule_rows:
            target_step = row["milestone_target_step"]
            row["cross_run_actual_step_status"] = (
                statuses[target_step] if row["checkpoint_role"] == "milestone" else "NOT_APPLICABLE"
            )
    return statuses, comparisons


def validate_horizon_comparison(plans: list[RunPlan]) -> None:
    horizons = [plan.horizon for plan in plans]
    if len(set(horizons)) != len(horizons):
        raise RuntimeError("selected runs must have distinct checkpoint metadata horizons")
    for plan in plans:
        label_match = HORIZON_LABEL.fullmatch(plan.label)
        if label_match and int(label_match.group(1)) != plan.horizon:
            raise RuntimeError(
                f"run label {plan.label!r} conflicts with checkpoint metadata horizon {plan.horizon}"
            )


def run_sweep(args) -> dict[str, Any]:
    if args.milestone_episodes <= 0 or args.final_episodes < args.milestone_episodes:
        raise ValueError("final-episodes must be >= milestone-episodes > 0")
    if args.episode_length <= 0:
        raise ValueError("episode-length must be positive")
    baseline_mode = args.baseline_checkpoint is not None
    if baseline_mode:
        if args.run:
            raise ValueError("--baseline-checkpoint cannot be combined with --run")
        if args.final_episodes != 10 or args.episode_length != 400:
            raise ValueError("baseline evaluation requires 10 final episodes and episode-length 400")
    else:
        validate_run_arguments(args.run)
    episode_seeds = evaluation_seed_set(args.seed, args.final_episodes)
    if baseline_mode:
        plans = [baseline_run_plan(args.baseline_checkpoint)]
    else:
        plans = [
            discover_run(
                label,
                path,
                milestone_episodes=args.milestone_episodes,
                final_episodes=args.final_episodes,
                allow_latest_as_final=args.allow_latest_as_final,
            )
            for label, path in args.run
        ]
        validate_horizon_comparison(plans)
    cross_run_statuses, cross_run_comparisons = cross_run_actual_step_validation(plans)
    manifest = partner_manifest()
    if (
        len(manifest) != 24
        or len({str(record["name"]) for record in manifest}) != 24
        or sum(record["family"] == "mep" for record in manifest) != 18
        or sum(record["family"] == "hsp" for record in manifest) != 6
    ):
        raise RuntimeError(
            "evaluation requires the exact HSP-S2-s24 seed-3 population: 18 MEP and 6 HSP"
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output / name for name in ("eval_episodes.csv", "eval_summary.csv", "eval_summary.json"))
    if not args.overwrite and any(path.exists() for path in outputs):
        raise RuntimeError(f"evaluation output already exists; use --overwrite: {output}")

    total_episodes = sum(
        target.episodes_per_partner * len(manifest)
        for plan in plans
        for target in plan.targets
    )
    episode_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    progress = tqdm(
        total=total_episodes,
        unit="episode",
        dynamic_ncols=True,
        disable=args.no_progress,
        desc="cross-run evaluation",
    )
    try:
        for plan in plans:
            for target in sorted(
                plan.targets,
                key=lambda item: (
                    item.checkpoint.step is None,
                    0 if item.checkpoint.step is None else item.checkpoint.step,
                ),
            ):
                evaluator = CheckpointEvaluator(
                    target.checkpoint, device=args.device, construction_seed=args.seed
                )
                target_rows = []
                seeds = episode_seeds[: target.episodes_per_partner]
                for partner_record in manifest:
                    partner = str(partner_record["name"])
                    rows = evaluator.evaluate_partner(
                        partner=partner,
                        evaluation_seeds=seeds,
                        episode_length=args.episode_length,
                    )
                    for row in rows:
                        enriched = {
                            "run_label": target.run_label,
                            "checkpoint_role": target.role,
                            "milestone_target_step": target.milestone_target_step,
                            "checkpoint_actual_step": target.checkpoint.step,
                            "checkpoint_step": target.checkpoint.step,
                            "cross_run_actual_step_status": (
                                cross_run_statuses[target.milestone_target_step]
                                if target.role == "milestone"
                                else "NOT_APPLICABLE"
                            ),
                            "checkpoint_path": str(target.checkpoint.path),
                            "observation_mode": target.checkpoint.metadata["observation_mode"],
                            "training_stage": target.checkpoint.metadata["training_stage"],
                            "horizon": target.checkpoint.metadata["horizon"],
                            "decision_time": target.checkpoint.metadata.get("decision_time"),
                            "fixation_shape": target.checkpoint.metadata.get("fixation_shape"),
                            "partner": partner,
                            "partner_family": partner_record["family"],
                            "episode_length": args.episode_length,
                            **row,
                        }
                        episode_rows.append(enriched)
                        target_rows.append(enriched)
                        progress.update(1)
                    progress.set_postfix(
                        run=target.run_label,
                        step=target.checkpoint.step,
                        partner=partner,
                    )
                evaluator.verify_unchanged()
                summary = summarize_target(target, target_rows)
                summary["cross_run_actual_step_status"] = (
                    cross_run_statuses[target.milestone_target_step]
                    if target.role == "milestone"
                    else "NOT_APPLICABLE"
                )
                summaries.append(summary)
    finally:
        progress.close()

    schedule_rows = [row for plan in plans for row in plan.schedule_rows]
    summary_csv_rows = summaries + [
        row for row in schedule_rows if row["status"] in {"MISSING_MILESTONE", "FINAL_CHECKPOINT_INCOMPLETE"}
    ]
    write_csv(outputs[0], EPISODE_FIELDS, episode_rows)
    write_csv(outputs[1], SUMMARY_FIELDS, summary_csv_rows)
    result = {
        "evaluation_contract": {
            "comparison_run_count": len(plans),
            "evaluation_kind": "hsp_s2_baseline" if baseline_mode else "trained_run_comparison",
            "milestone_interval": MILESTONE_INTERVAL,
            "milestone_selection": "M <= checkpoint_actual_step < M + rollout_length; earliest valid",
            "milestone_episodes_per_partner": args.milestone_episodes,
            "final_episodes_per_partner": args.final_episodes,
            "episode_length": args.episode_length,
            "base_seed": args.seed,
            "evaluation_episode_seeds": list(episode_seeds),
            "partner_count": 24,
            "mep_count": 18,
            "hsp_count": 6,
            "learner_action_selection": "deterministic_argmax",
            "learner_observation": (
                "canonical_20ch" if baseline_mode else "metadata_condition_observation"
            ),
            "learner_legal_action_mask": (
                "original_agent1_legal_action_mask"
            ),
            "learner_recurrent_reset": (
                "zero_hidden_and_zero_mask_each_episode" if baseline_mode else "existing_evaluation_semantics"
            ),
            "learned_gaze_selection": "deterministic_argmax",
            "partner_action_selection": "original frozen policy torch.multinomial with per-episode seed",
            "primary_metrics": ["sparse_team_reward", "deliveries"],
            "diagnostic_metrics": ["native_shaped_reward", "optimization_reward"],
            "std_definition": "sample standard deviation over episode rows (ddof=1)",
            "seed_protocol": (
                "base_seed + episode_index; identical partner/episode seeds reused across "
                "all runs and checkpoints; final extends milestone seed prefix"
            ),
            "final_selection": (
                "checkpoint metadata step equals configured total steps; latest only with explicit override"
            ),
        },
        "runs": [
            {
                "run_label": plan.label,
                "horizon": plan.horizon,
                "checkpoint_dir": str(plan.checkpoint_dir),
                "configured_total_steps": plan.configured_steps,
                "rollout_length": plan.rollout_length,
                "training_episode_length": plan.training_episode_length,
                "evaluation_episode_length": args.episode_length,
                "episode_length_status": (
                    "UNRECORDED"
                    if plan.training_episode_length is None
                    else "MATCH"
                    if plan.training_episode_length == args.episode_length
                    else "EVALUATION_OVERRIDE_REPORTED"
                ),
                "final_status": plan.final_status,
                "ignored_checkpoint_steps": plan.ignored_checkpoint_steps,
                "schedule": plan.schedule_rows,
            }
            for plan in plans
        ],
        "cross_run_actual_step_validation": cross_run_comparisons,
        "checkpoint_summaries": summaries,
        "missing_milestones": [
            row for row in schedule_rows if row["status"] == "MISSING_MILESTONE"
        ],
        "incomplete_finals": [
            row for row in schedule_rows if row["status"] == "FINAL_CHECKPOINT_INCOMPLETE"
        ],
        "episode_row_count": len(episode_rows),
        "outputs": {
            "episodes_csv": str(outputs[0]),
            "summary_csv": str(outputs[1]),
            "summary_json": str(outputs[2]),
        },
    }
    outputs[2].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run", type=parse_run, action="append", default=[])
    value.add_argument("--baseline-checkpoint", type=Path)
    value.add_argument("--milestone-episodes", type=int, default=5)
    value.add_argument("--final-episodes", type=int, default=10)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--seed", type=int, default=20260821)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--output-dir", type=Path, default=Path("final_model/eval_results/comparison"))
    value.add_argument("--allow-latest-as-final", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--no-progress", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = run_sweep(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
