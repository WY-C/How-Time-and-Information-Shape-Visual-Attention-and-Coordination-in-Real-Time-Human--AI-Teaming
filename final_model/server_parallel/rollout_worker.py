"""Persistent CPU-only worker for environment, gaze, and Partial memory state."""

from __future__ import annotations

import copy
import os
import time
import traceback

import numpy as np
import torch

from final_model.checkpoint import state_dict_hash
from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.common.partners import FIXED_HSP_PARTNER_ID
from final_model.common.zsceval_compat import install_zsceval
from final_model.partial_ver.conditions import PartialCondition
from final_model.partial_ver.observation import PartialObservationAdapter
from final_model.partial_ver.observability import partial_observation_statistics
from final_model.partial_ver.network import GazeCritic, GazeNet
from final_model.partial_ver.training import LearnedGazeController
from final_model.parallel_gaze_contract import (
    PARALLEL_GAZE_CPU_INFERENCE_DTYPE,
    PARALLEL_GAZE_CPU_INFERENCE_PATH,
    PARALLEL_GAZE_CPU_INTER_OP_THREADS,
    PARALLEL_GAZE_CPU_INTRA_OP_THREADS,
    PARALLEL_GAZE_RNG_CONTRACT,
    logical_env_gaze_seed,
)
from final_model.server_parallel.collision_observability import (
    human_compatible_joint_transition_collision,
)
from final_model.server_parallel.footprint_audit_recorder import (
    FootprintAuditRecorder,
)
from final_model.server_parallel.protocol import (
    CODE_TO_FIXATION,
    MAX_ACQUISITIONS_PER_DECISION,
)


RULE_GAZE_EVENT_HISTORY_LIMIT = 128


class _NoNetworkActor(torch.nn.Module):
    """Marker actor: workers never execute or store fixed-partner weights."""

    def get_probs(self, *_args, **_kwargs):
        raise RuntimeError("fixed-partner inference is main-process-only")


class WorkerPartnerRepository:
    """Structural backend dependency without a partner neural network."""

    fixed_single_partner = True

    def __init__(self, *, parameter_sha256: str, checkpoint_sha256: str) -> None:
        self.device = torch.device("cpu")
        self.parameter_sha256 = str(parameter_sha256)
        self.manifest = [{
            "name": FIXED_HSP_PARTNER_ID,
            "family": "hsp_s2",
            "path": "main_process_only",
            "sha256": str(checkpoint_sha256),
        }]
        self._actor = _NoNetworkActor()
        self._guards = {FIXED_HSP_PARTNER_ID: ()}

    def actor(self, name: str):
        if name != FIXED_HSP_PARTNER_ID:
            raise ValueError(f"unknown fixed partner {name!r}")
        return self._actor

    @staticmethod
    def recurrent_shape() -> tuple[int, int]:
        return 1, 64

    def assert_frozen(self, name: str, *, full_hash: bool = False) -> str:
        if name != FIXED_HSP_PARTNER_ID:
            raise RuntimeError("worker fixed-partner identity changed")
        return self.parameter_sha256


def _trace_delta(adapter, event_count: int, rule_count: int) -> dict[str, object]:
    events = list(adapter.events)
    rule_events = list(getattr(adapter.gaze_policy, "events", ()))
    return {
        "events": copy.deepcopy(events[event_count:]),
        "rule_events": copy.deepcopy(rule_events[rule_count:]),
    }


def _trim_rule_gaze_event_history(adapter) -> int:
    """Retain the checkpoint-serialized tail after decision trace extraction."""
    events = getattr(adapter.gaze_policy, "events", None)
    if not isinstance(events, list):
        raise RuntimeError("rule-gaze event history must retain list semantics")
    if len(events) > RULE_GAZE_EVENT_HISTORY_LIMIT:
        del events[:-RULE_GAZE_EVENT_HISTORY_LIMIT]
    return len(events)


def rollout_worker_main(
    worker_id: int,
    env_ids: tuple[int, ...],
    command_queue,
    response_queue,
    shared_buffers,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    install_zsceval()
    views = shared_buffers.views()
    pending: dict[int, dict[str, object]] = {}
    adapters: dict[int, PartialObservationAdapter] = {}
    backends: dict[int, DirectOvercookedBackend] = {}
    controllers: dict[int, LearnedGazeController] = {}
    gaze_generators: dict[int, torch.Generator] = {}
    gaze_actor = gaze_critic = None
    gaze_policy_version = -1
    audit_recorders: dict[int, FootprintAuditRecorder] = {}
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError("rollout worker unexpectedly initialized CUDA")
        response_queue.put({
            "worker_id": int(worker_id),
            "phase": "boot",
            "pid": os.getpid(),
            "env_ids": tuple(int(value) for value in env_ids),
            "cuda_initialized": False,
        })
        initialize = command_queue.get()
        if str(initialize[0]) == "shutdown":
            response_queue.put({
                "worker_id": int(worker_id), "phase": "shutdown",
                "serial": int(initialize[1]),
            })
            return
        if str(initialize[0]) != "initialize":
            raise RuntimeError("first worker command must initialize runtime state")
        initialize_serial = int(initialize[1])
        config = initialize[2]
        gaze_type = str(config.get("gaze_type", "rule_based"))
        if gaze_type not in {"rule_based", "learned"}:
            raise RuntimeError(f"unsupported worker gaze type {gaze_type!r}")
        if gaze_type == "learned":
            requested_intra = int(config["worker_torch_intra_op_threads"])
            requested_inter = int(config["worker_torch_inter_op_threads"])
            if requested_intra != PARALLEL_GAZE_CPU_INTRA_OP_THREADS or (
                requested_inter != PARALLEL_GAZE_CPU_INTER_OP_THREADS
            ):
                raise RuntimeError("worker received an unsupported CPU thread contract")
            if config.get("gaze_cpu_inference_dtype") != (
                PARALLEL_GAZE_CPU_INFERENCE_DTYPE
            ) or config.get("gaze_cpu_inference_path") != (
                PARALLEL_GAZE_CPU_INFERENCE_PATH
            ):
                raise RuntimeError("worker received an unsupported inference contract")
            torch.set_num_threads(requested_intra)
            torch.set_num_interop_threads(
                requested_inter
            )
            if (
                torch.get_num_threads() != requested_intra
                or torch.get_num_interop_threads() != requested_inter
            ):
                raise RuntimeError("worker failed to apply the CPU thread contract")
            gaze_actor = GazeNet().cpu()
            gaze_critic = GazeCritic().cpu()
            gaze_actor.load_state_dict(config["gaze_actor_state"], strict=True)
            gaze_critic.load_state_dict(config["gaze_critic_state"], strict=True)
            gaze_actor.eval().requires_grad_(False)
            gaze_critic.eval().requires_grad_(False)
            if any(
                value.is_floating_point() and value.dtype != torch.float32
                for module in (gaze_actor, gaze_critic)
                for value in module.state_dict().values()
            ):
                raise RuntimeError("worker GazeNet dtype is not float32")
            if state_dict_hash(gaze_actor.state_dict()) != str(
                config["gaze_actor_sha256"]
            ):
                raise RuntimeError("worker initial GazeNet actor hash mismatch")
            if state_dict_hash(gaze_critic.state_dict()) != str(
                config["gaze_critic_sha256"]
            ):
                raise RuntimeError("worker initial GazeNet critic hash mismatch")
            gaze_policy_version = int(config["gaze_policy_version"])
        repository = WorkerPartnerRepository(
            parameter_sha256=str(config["partner_parameter_sha256"]),
            checkpoint_sha256=str(config["partner_checkpoint_sha256"]),
        )
        shared_oracle = None
        for env_id in env_ids:
            item = config["environments"][int(env_id)]
            condition = PartialCondition(**item["condition"])
            controller = None
            if gaze_type == "learned":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(logical_env_gaze_seed(
                    int(config["master_seed"]),
                    int(env_id),
                    int(item["gaze_episode_index"]),
                ))
                if item.get("gaze_generator_state") is not None:
                    generator.set_state(item["gaze_generator_state"])
                controller = LearnedGazeController(
                    gaze_actor,
                    gaze_critic,
                    torch.device("cpu"),
                    env_index=int(env_id),
                    action_generator=generator,
                )
                controllers[int(env_id)] = controller
                gaze_generators[int(env_id)] = generator
            backend = DirectOvercookedBackend(
                seed=int(item["seed"]),
                horizon=condition.horizon,
                episode_length=int(config["episode_length"]),
                device="cpu",
                partner=None,
                partner_repository=repository,
                prediction_oracle=shared_oracle,
                prediction_device="cpu",
                environment_index=int(env_id),
                enable_predictor=bool(config["enable_predictor"]),
            )
            adapter = PartialObservationAdapter(
                backend,
                decision_time=condition.decision_time,
                fixation_shape=condition.fixation_shape,
                gaze_policy=controller,
                rule_gaze_seed=int(item["rule_gaze_seed"]),
                rule_gaze_probabilities=config["rule_gaze_probabilities"],
                memory_lifetime_ms=config["memory_lifetime_ms"],
                memory_decay_mode=str(
                    config.get("memory_decay_mode", "linear")
                ),
                memory_tau_ms=config.get("memory_tau_ms"),
                memory_seed=item.get("memory_seed"),
                constraint_config=config["constraint_config"],
                peripheral_seed=int(item["peripheral_seed"]),
                opacity_seed=int(item["opacity_seed"]),
                dwell_seed=item.get("dwell_seed"),
                ground_truth_policy_input=bool(
                    config.get("ground_truth_policy_input", False)
                ),
            )
            backend.load_training_state(item["backend_state"])
            adapter.load_training_state(item["adapter_state"])
            if controller is not None:
                controller.set_policy_version(gaze_policy_version)
            if bool(config.get("memory_audit_enabled", False)):
                recorder = FootprintAuditRecorder(
                    env_id=int(env_id),
                    horizon=condition.horizon,
                    fixation_shape=condition.fixation_shape,
                )
                recorder.install(adapter)
                audit_recorders[int(env_id)] = recorder
            backends[int(env_id)] = backend
            adapters[int(env_id)] = adapter
            shared_oracle = backend.oracle
        response_queue.put({
            "worker_id": int(worker_id),
            "phase": "startup",
            "serial": initialize_serial,
            "pid": os.getpid(),
            "env_ids": tuple(int(value) for value in env_ids),
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "gaze_type": gaze_type,
            "gaze_policy_version": gaze_policy_version,
            "gaze_actor_sha256": (
                None if gaze_actor is None
                else state_dict_hash(gaze_actor.state_dict())
            ),
            "gaze_critic_sha256": (
                None if gaze_critic is None
                else state_dict_hash(gaze_critic.state_dict())
            ),
            "gaze_rng_contract": (
                PARALLEL_GAZE_RNG_CONTRACT if gaze_type == "learned" else None
            ),
            "torch_cpu_intra_op_threads": (
                torch.get_num_threads() if gaze_type == "learned" else None
            ),
            "torch_cpu_inter_op_threads": (
                torch.get_num_interop_threads()
                if gaze_type == "learned" else None
            ),
            "gaze_cpu_inference_dtype": (
                PARALLEL_GAZE_CPU_INFERENCE_DTYPE
                if gaze_type == "learned" else None
            ),
            "gaze_cpu_inference_path": (
                PARALLEL_GAZE_CPU_INFERENCE_PATH
                if gaze_type == "learned" else None
            ),
            "gaze_actor_eval": (
                not gaze_actor.training if gaze_actor is not None else None
            ),
            "gaze_actor_requires_grad": (
                any(parameter.requires_grad for parameter in gaze_actor.parameters())
                if gaze_actor is not None else None
            ),
        })

        while True:
            command = command_queue.get()
            phase = str(command[0])
            serial = int(command[1])
            started = time.perf_counter()
            trace_payload = []
            if phase == "shutdown":
                response_queue.put({
                    "worker_id": int(worker_id), "phase": phase, "serial": serial,
                })
                return
            if phase == "sync_gaze":
                if gaze_type != "learned" or gaze_actor is None or gaze_critic is None:
                    raise RuntimeError("sync_gaze requires learned gaze workers")
                actor_state, critic_state, actor_hash, critic_hash, version = command[2:]
                gaze_actor.load_state_dict(actor_state, strict=True)
                gaze_critic.load_state_dict(critic_state, strict=True)
                gaze_actor.eval().requires_grad_(False)
                gaze_critic.eval().requires_grad_(False)
                actual_actor = state_dict_hash(gaze_actor.state_dict())
                actual_critic = state_dict_hash(gaze_critic.state_dict())
                if actual_actor != str(actor_hash) or actual_critic != str(critic_hash):
                    raise RuntimeError("worker GazeNet synchronization hash mismatch")
                gaze_policy_version = int(version)
                for controller in controllers.values():
                    controller.set_policy_version(gaze_policy_version)
                trace_payload = [{
                    "gaze_policy_version": gaze_policy_version,
                    "gaze_actor_sha256": actual_actor,
                    "gaze_critic_sha256": actual_critic,
                }]
            elif phase == "collect":
                factor = float(command[2])
                pending.clear()
                for env_id in env_ids:
                    env = int(env_id)
                    adapter = adapters[env]
                    backend = backends[env]
                    backend.set_reward_shaping_factor(factor)
                    views["acquisition_counts"][env] = 0
                    gaze_requests_before = adapter.gaze_requests
                    same_targets_before = adapter.same_target_acquisitions
                    new_targets_before = adapter.new_target_acquisitions
                    distance_before = adapter.distance_sum_degrees
                    event_count = len(adapter.events)
                    rule_count = len(getattr(adapter.gaze_policy, "events", ()))
                    if gaze_type == "learned":
                        task_decision_id = int(adapter.decision_index)
                        controllers[env].begin_task_decision(task_decision_id)
                    else:
                        task_decision_id = None

                    def on_acquisition(observation, _critic, available, timestamp):
                        if env in audit_recorders:
                            audit_recorders[env].record_actionnet_query(
                                observation, available
                            )
                        slot = int(views["acquisition_counts"][env])
                        if slot >= MAX_ACQUISITIONS_PER_DECISION:
                            raise RuntimeError(
                                "acquisition bound exceeded; canonical gaze timing changed"
                            )
                        views["acquisition_observations"][env, slot] = observation
                        views["acquisition_timestamps"][env, slot] = float(timestamp)
                        views["acquisition_counts"][env] = slot + 1
                        return {"shared_acquisition_slot": slot,
                                "available_actions": available.copy()}

                    collected = adapter.collect_decision(on_acquisition)
                    if env in audit_recorders:
                        audit_recorders[env].record_deadline(
                            collected["observation"],
                            collected["available_actions"],
                        )
                    count = int(views["acquisition_counts"][env])
                    if count != len(collected["proposals"]):
                        raise RuntimeError(
                            "worker ActionNet-query/shared-buffer count mismatch"
                        )
                    observation_stats = partial_observation_statistics(
                        tuple(
                            views["acquisition_observations"][env, index]
                            for index in range(count)
                        )
                    )
                    views["gaze_request_counts"][env] = int(
                        adapter.gaze_requests - gaze_requests_before
                    )
                    views["same_target_acquisition_counts"][env] = int(
                        adapter.same_target_acquisitions - same_targets_before
                    )
                    views["new_target_acquisition_counts"][env] = int(
                        adapter.new_target_acquisitions - new_targets_before
                    )
                    views["valid_memory_tile_counts"][env] = int(
                        observation_stats["valid_memory_tiles"]
                    )
                    views["freshness_sample_counts"][env] = int(
                        observation_stats["freshness_sample_count"]
                    )
                    views["freshness_normalized_sums"][env] = float(
                        observation_stats["freshness_normalized_sum"]
                    )
                    views["saccade_distance_sums_degrees"][env] = float(
                        adapter.distance_sum_degrees - distance_before
                    )
                    pending[env] = collected
                    if task_decision_id is not None:
                        collected["parallel_gaze_task_decision_id"] = (
                            task_decision_id
                        )
                    views["final_observations"][env] = collected["observation"]
                    views["critic_observations"][env] = collected["critic"]
                    views["available_actions"][env] = collected["available_actions"]
                    partner_obs, partner_available = backend.partner_batch_input()
                    views["partner_observations"][env] = partner_obs
                    views["partner_available_actions"][env] = partner_available
                    trace_row = None
                    if bool(config["trace_enabled"]):
                        trace_row = {
                            "env_id": env,
                            **_trace_delta(adapter, event_count, rule_count),
                        }
                    retained_rule_events = (
                        _trim_rule_gaze_event_history(adapter)
                        if gaze_type == "rule_based" else 0
                    )
                    if trace_row is not None:
                        trace_row["retained_rule_event_count"] = retained_rule_events
                        trace_payload.append(trace_row)
            elif phase == "commit":
                for env_id in env_ids:
                    env = int(env_id)
                    collected = pending.get(env)
                    if collected is None:
                        raise RuntimeError("commit received without pending collection")
                    latest = collected["latest_proposal"]
                    if latest is not None:
                        latest["action"] = int(views["learner_actions"][env])
                    before = backends[env].begin_partner_step()
                    if before != str(config["partner_parameter_sha256"]):
                        raise RuntimeError("worker/main fixed-partner guard mismatch")
                    backend = backends[env]
                    learner_action = int(views["learner_actions"][env])
                    partner_action = int(views["partner_actions"][env])
                    views["joint_collisions"][env] = int(
                        human_compatible_joint_transition_collision(
                            backend.mdp,
                            backend.state,
                            (
                                backend.Action.INDEX_TO_ACTION[partner_action],
                                backend.Action.INDEX_TO_ACTION[learner_action],
                            ),
                        )
                    )
                    result = adapters[env].commit_collected(
                        collected,
                        partner_action=partner_action,
                        partner_hash_before=before,
                    )
                    views["rewards"][env] = (
                        float(result["optimization_reward"]),
                        float(result["sparse_team_reward"]),
                        float(result["native_shaped_reward"]),
                        float(result["weighted_shaped_reward"]),
                    )
                    views["deliveries"][env] = int(result["deliveries"])
                    views["dones"][env] = int(bool(result["done"]))
                    if gaze_type == "learned":
                        controller = controllers[env]
                        controller.accrue(
                            optimization_reward=result["optimization_reward"],
                            sparse_team_reward=result["sparse_team_reward"],
                            native_shaped_reward=result["native_shaped_reward"],
                            weighted_shaped_reward=result["weighted_shaped_reward"],
                            deliveries=result["deliveries"],
                            task_decision_id=int(
                                collected["parallel_gaze_task_decision_id"]
                            ),
                        )
                        if bool(result["done"]):
                            controller.finish_episode(adapter.current_time_ms)
            elif phase == "finalize":
                for env_id in env_ids:
                    env = int(env_id)
                    adapter = adapters[env]
                    if bool(views["reset_flags"][env]):
                        adapter.reset_episode(PartialCondition(
                            decision_time=int(views["reset_decision_times"][env]),
                            horizon=int(views["reset_horizons"][env]),
                            fixation_shape=CODE_TO_FIXATION[
                                int(views["reset_fixation_codes"][env])
                            ],
                        ))
                        if gaze_type == "learned":
                            gaze_generators[env].manual_seed(
                                logical_env_gaze_seed(
                                    int(config["master_seed"]),
                                    env,
                                    controllers[env].episode_id,
                                )
                            )
                    observation, critic, available = adapter.current()
                    views["next_observations"][env] = observation
                    views["next_critic_observations"][env] = critic
                    views["next_available_actions"][env] = available
                    if gaze_type == "learned":
                        trace_payload.append({
                            "env_id": env,
                            "gaze_records": controllers[env].drain(),
                            "gaze_policy_version": gaze_policy_version,
                        })
                pending.clear()
            elif phase == "metrics":
                trace_payload = [{
                    "env_id": int(env_id),
                    "adapter": adapters[int(env_id)].metrics(),
                    "partner_step_guard_checks": backends[int(env_id)].partner_step_guard_checks,
                    "partner_step_guard_failures": backends[int(env_id)].partner_step_guard_failures,
                    "partner_full_hash_checks": backends[int(env_id)].partner_full_hash_checks,
                    "partner_full_hash_failures": backends[int(env_id)].partner_full_hash_failures,
                    "environment_steps": backends[int(env_id)].environment_steps,
                    "active_partner": backends[int(env_id)].active_partner,
                    "memory_audit": (
                        audit_recorders[int(env_id)].payload()
                        if int(env_id) in audit_recorders else None
                    ),
                    "gaze_total_attributed_optimization_reward": (
                        controllers[int(env_id)].total_attributed_optimization_reward
                        if gaze_type == "learned" else 0.0
                    ),
                    "gaze_total_env_boundaries": (
                        controllers[int(env_id)].total_env_boundaries
                        if gaze_type == "learned" else 0
                    ),
                    "gaze_policy_calls_total": (
                        controllers[int(env_id)].next_record_sequence_id
                        if gaze_type == "learned" else 0
                    ),
                    "gaze_transition_count_total": (
                        controllers[int(env_id)].next_record_sequence_id
                        - int(controllers[int(env_id)].active is not None)
                        if gaze_type == "learned" else 0
                    ),
                    "bootstrap_unowned_reward_count_total": (
                        controllers[int(env_id)].total_bootstrap_unowned_reward_count
                        if gaze_type == "learned" else 0
                    ),
                    "bootstrap_unowned_reward_sum_total": (
                        controllers[int(env_id)].total_bootstrap_unowned_reward_sum
                        if gaze_type == "learned" else 0.0
                    ),
                } for env_id in env_ids]
            elif phase == "snapshot":
                if pending:
                    raise RuntimeError("worker snapshot requested mid-decision")
                trace_payload = [{
                    "env_id": int(env_id),
                    "backend": backends[int(env_id)].training_state(),
                    "adapter": adapters[int(env_id)].training_state(),
                    "gaze_generator_state": (
                        gaze_generators[int(env_id)].get_state()
                        if gaze_type == "learned" else None
                    ),
                    "gaze_policy_version": gaze_policy_version,
                } for env_id in env_ids]
            elif phase == "verify":
                for env_id in env_ids:
                    backends[int(env_id)].verify_partner_parameters()
            else:
                raise RuntimeError(f"unknown rollout worker phase {phase!r}")
            response_queue.put({
                "worker_id": int(worker_id),
                "phase": phase,
                "serial": serial,
                "elapsed_seconds": time.perf_counter() - started,
                "payload": trace_payload,
            })
    except BaseException as error:
        response_queue.put({
            "worker_id": int(worker_id),
            "phase": "error",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        })
