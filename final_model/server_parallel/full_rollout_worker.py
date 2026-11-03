"""Persistent CPU-only environment worker for canonical Full Action."""

from __future__ import annotations

import copy
import os
import time
import traceback

import torch

from final_model.common.overcooked_backend import DirectOvercookedBackend
from final_model.server_parallel.rollout_worker import WorkerPartnerRepository
from final_model.simple_ver.observation import FullObservationAdapter


class WorkerPredictionOracle:
    """Marker dependency; predictor inference is forbidden in workers."""

    checkpoint_path = "main_process_only"

    def predict(self, *_args, **_kwargs):
        raise RuntimeError("predictor inference is main-process-only")

    def predict_batch_with_logits(self, *_args, **_kwargs):
        raise RuntimeError("predictor inference is main-process-only")


def _write_inputs(env_id: int, backend, views) -> None:
    partner, learner = backend.canonical_pair()
    views["learner_observations"][env_id] = learner
    views["partner_observations"][env_id] = partner
    views["learner_available_actions"][env_id] = (
        backend.learner_available_actions()
    )
    views["partner_available_actions"][env_id] = backend.partner_batch_input()[1]


def _runtime_state(env_id: int, backend) -> dict[str, object]:
    return {
        "env_id": int(env_id),
        "horizon": int(backend.horizon),
        "state": copy.deepcopy(backend.state),
        "episode_step": int(backend.episode_step),
        "environment_steps": int(backend.environment_steps),
        "state_generation": int(backend.state_generation),
        "reward_shaping_factor": float(backend.reward_shaping_factor),
        "self_history": copy.deepcopy(backend.self_history),
        "partner_history": copy.deepcopy(backend.partner_history),
    }


def full_rollout_worker_main(
    worker_id: int,
    env_ids: tuple[int, ...],
    command_queue,
    response_queue,
    shared_buffers,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    views = shared_buffers.views()
    adapters: dict[int, FullObservationAdapter] = {}
    backends: dict[int, DirectOvercookedBackend] = {}
    last_response = time.perf_counter()
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError("a Full rollout worker unexpectedly initialized CUDA")
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
                "worker_id": int(worker_id),
                "phase": "shutdown",
                "serial": int(initialize[1]),
            })
            return
        if str(initialize[0]) != "initialize":
            raise RuntimeError("first Full worker command must initialize runtime state")
        initialize_serial = int(initialize[1])
        config = initialize[2]
        repository = WorkerPartnerRepository(
            parameter_sha256=str(config["partner_parameter_sha256"]),
            checkpoint_sha256=str(config["partner_checkpoint_sha256"]),
        )
        oracle = WorkerPredictionOracle() if bool(config["enable_predictor"]) else None
        for env_id in env_ids:
            env = int(env_id)
            item = config["environments"][env]
            backend = DirectOvercookedBackend(
                seed=int(item["seed"]),
                horizon=int(item["horizon"]),
                episode_length=int(config["episode_length"]),
                device="cpu",
                partner=None,
                partner_repository=repository,
                prediction_oracle=oracle,
                prediction_device="cpu",
                environment_index=env,
                enable_predictor=bool(config["enable_predictor"]),
            )
            adapter = FullObservationAdapter(backend)
            backend.load_training_state(item["backend_state"])
            adapter.load_training_state(item["adapter_state"])
            backends[env] = backend
            adapters[env] = adapter
            _write_inputs(env, backend, views)
        response_queue.put({
            "worker_id": int(worker_id),
            "phase": "startup",
            "serial": initialize_serial,
            "pid": os.getpid(),
            "env_ids": tuple(int(value) for value in env_ids),
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "payload": [
                _runtime_state(int(env_id), backends[int(env_id)])
                for env_id in env_ids
            ],
        })
        last_response = time.perf_counter()

        while True:
            command = command_queue.get()
            received = time.perf_counter()
            queue_wait = received - last_response
            phase = str(command[0])
            serial = int(command[1])
            started = time.perf_counter()
            payload = []
            if phase == "shutdown":
                response_queue.put({
                    "worker_id": int(worker_id),
                    "phase": phase,
                    "serial": serial,
                    "queue_wait_seconds": queue_wait,
                })
                return
            if phase == "commit":
                factor = float(command[2])
                for env_id in env_ids:
                    env = int(env_id)
                    backend = backends[env]
                    backend.set_reward_shaping_factor(factor)
                    before = backend.begin_partner_step()
                    result = adapters[env].commit_decision_with_partner(
                        int(views["learner_actions"][env]),
                        int(views["partner_actions"][env]),
                        before,
                    )
                    payload.append({
                        **_runtime_state(env, backend),
                        "result": result,
                    })
            elif phase == "prepare":
                for env_id in env_ids:
                    env = int(env_id)
                    backend = backends[env]
                    if bool(views["reset_flags"][env]):
                        adapters[env].reset_episode()
                        backend.set_horizon(int(views["reset_horizons"][env]))
                    elif backend.horizon != int(views["reset_horizons"][env]):
                        raise RuntimeError("Full worker horizon changed without reset")
                    _write_inputs(env, backend, views)
                    payload.append(_runtime_state(env, backend))
            elif phase == "metrics":
                payload = [{
                    "env_id": int(env_id),
                    "adapter": adapters[int(env_id)].training_state(),
                    "backend": backends[int(env_id)].training_state(),
                } for env_id in env_ids]
            elif phase == "snapshot":
                payload = [{
                    "env_id": int(env_id),
                    "adapter": adapters[int(env_id)].training_state(),
                    "backend": backends[int(env_id)].training_state(),
                } for env_id in env_ids]
            elif phase == "verify":
                for env_id in env_ids:
                    backends[int(env_id)].verify_partner_parameters()
            else:
                raise RuntimeError(f"unknown Full rollout worker phase {phase!r}")
            response_queue.put({
                "worker_id": int(worker_id),
                "phase": phase,
                "serial": serial,
                "elapsed_seconds": time.perf_counter() - started,
                "queue_wait_seconds": queue_wait,
                "payload": payload,
            })
            last_response = time.perf_counter()
    except BaseException as error:
        response_queue.put({
            "worker_id": int(worker_id),
            "phase": "error",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        })
