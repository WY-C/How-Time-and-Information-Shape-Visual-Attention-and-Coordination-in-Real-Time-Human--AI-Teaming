"""Persistent-worker execution backend for canonical Full Action PPO."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import queue
import time
from typing import Any

import numpy as np
import torch

from final_model.common.action_training import ActionTrainingSession
from final_model.common.hsp_initialization import HSP_SOURCE_CHECKPOINT_SHA256
from final_model.common.mappo import _actor_pipeline_snapshot
from final_model.common.overcooked_backend import cooking_contract
from final_model.common.partners import (
    FIXED_HSP_PARTNER_ID,
    FIXED_HSP_PARTNER_SHA256,
)
from final_model.common.paths import file_sha256
from final_model.server_parallel.full_rollout_worker import (
    full_rollout_worker_main,
)
from final_model.server_parallel.full_shared_buffers import FullSharedStepBuffers
from final_model.server_parallel.protocol import deterministic_assignments
from final_model.simple_ver.observation import build_full_observation


FULL_SERVER_PARALLEL_VERSION = "server_parallel_full_v1"
WORKER_RESPONSE_TIMEOUT_SECONDS = 300.0


def configure_torch_execution(
    intra_op_threads: int,
    inter_op_threads: int,
) -> None:
    """Apply the server main-process CPU thread contract before CUDA starts."""
    intra = int(intra_op_threads)
    inter = int(inter_op_threads)
    if intra <= 0:
        raise ValueError("torch intra-op threads must be positive")
    if inter <= 0:
        raise ValueError("torch inter-op threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(intra)
    os.environ["MKL_NUM_THREADS"] = str(intra)
    torch.set_num_threads(intra)
    if torch.get_num_interop_threads() != inter:
        try:
            torch.set_num_interop_threads(inter)
        except RuntimeError as error:
            if torch.get_num_interop_threads() != inter:
                raise RuntimeError(
                    "torch inter-op threads must be configured before parallel work"
                ) from error
    if torch.get_num_threads() != intra:
        raise RuntimeError("torch intra-op thread configuration did not take effect")
    if torch.get_num_interop_threads() != inter:
        raise RuntimeError("torch inter-op thread configuration did not take effect")


def _array_hash(value) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _object_hash(value) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


class MainFullObservationAdapter:
    """Canonical Full adapter whose 20-channel encoding comes from a worker."""

    def __init__(self, session, env_id: int, backend, state: dict[str, int]):
        self.session = session
        self.env_id = int(env_id)
        self.backend = backend
        self.actionnet_online_forwards = int(state["actionnet_online_forwards"])
        self.task_decisions = int(state["task_decisions"])

    def reset_episode(self) -> None:
        self.backend.reset()
        self.session._views["reset_flags"][self.env_id] = 1

    def prepare_decision(self):
        learner = self.session._views["learner_observations"][self.env_id]
        partner = self.session._views["partner_observations"][self.env_id]
        actor = build_full_observation(
            learner, self.backend.preview_planes(), self.backend.horizon
        )
        critic = np.ascontiguousarray(
            np.concatenate((learner, partner), axis=-1), dtype=np.float32
        )
        available = self.session._views["learner_available_actions"][
            self.env_id
        ].copy()
        return actor, critic, available

    current = prepare_decision

    def training_state(self) -> dict[str, int]:
        return {
            "actionnet_online_forwards": self.actionnet_online_forwards,
            "task_decisions": self.task_decisions,
        }

    def load_training_state(self, state: dict[str, int]) -> None:
        self.actionnet_online_forwards = int(state["actionnet_online_forwards"])
        self.task_decisions = int(state["task_decisions"])


class ServerParallelFullActionTrainingSession(ActionTrainingSession):
    """Canonical Full neural/PPO schedule with CPU environments in workers."""

    def __init__(
        self,
        *,
        rollout_workers: int = 1,
        torch_intra_op_threads: int | None = None,
        torch_inter_op_threads: int | None = None,
        fixed_partner_executor_workers: int | None = None,
        server_trace: bool = False,
        **kwargs,
    ) -> None:
        if kwargs.get("observation_mode") != "full":
            raise ValueError("Full server session requires observation_mode='full'")
        if kwargs.get("inference_batching") not in (None, "predictor"):
            raise ValueError(
                "Full server multiprocessing supports predictor-only batching"
            )
        if kwargs.get("partner") is not None:
            raise ValueError("Full server multiprocessing requires the fixed partner")
        environments = int(kwargs["n_rollout_threads"])
        workers = int(rollout_workers)
        if torch_intra_op_threads is not None or torch_inter_op_threads is not None:
            if torch_intra_op_threads is None or torch_inter_op_threads is None:
                raise ValueError(
                    "torch intra-op and inter-op threads must be configured together"
                )
            configure_torch_execution(
                torch_intra_op_threads,
                torch_inter_op_threads,
            )
        executor_workers = (
            None
            if fixed_partner_executor_workers is None
            else int(fixed_partner_executor_workers)
        )
        if executor_workers is not None and executor_workers <= 0:
            raise ValueError("fixed-partner executor workers must be positive")
        self.torch_intra_op_threads = (
            None
            if torch_intra_op_threads is None
            else int(torch_intra_op_threads)
        )
        self.torch_inter_op_threads = (
            None
            if torch_inter_op_threads is None
            else int(torch_inter_op_threads)
        )
        self.fixed_partner_executor_workers = executor_workers
        self.assignments = deterministic_assignments(environments, workers)
        self.rollout_workers = workers
        self.server_trace = bool(server_trace)
        fixed_partner = kwargs.get("fixed_partner_checkpoint")
        if (
            fixed_partner is None
            or file_sha256(fixed_partner) != FIXED_HSP_PARTNER_SHA256
        ):
            raise ValueError("Full server multiprocessing requires canonical partner")
        resume = kwargs.get("resume_checkpoint")
        initial = kwargs.get("hsp_initial_checkpoint")
        if resume is None:
            if (
                initial is None
                or file_sha256(initial) != HSP_SOURCE_CHECKPOINT_SHA256
            ):
                raise ValueError(
                    "Full server multiprocessing requires canonical HSP learner"
                )
        else:
            payload = torch.load(resume, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            prediction_enabled = (
                kwargs.get("shared_horizons") is not None
                or int(kwargs.get("horizon") or 0) > 0
            )
            expected = {
                "execution_backend": FULL_SERVER_PARALLEL_VERSION,
                "rollout_workers": workers,
                "learner_hsp_source_sha256": HSP_SOURCE_CHECKPOINT_SHA256,
                "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
                "inference_batching": "predictor",
                "partner_inference_schedule": "batch1_per_env",
                "predictor_inference_schedule": (
                    "one_shared_env_batch_per_environment_step"
                    if prediction_enabled else "none"
                ),
                "cooking_dynamics_id": cooking_contract()["cooking_dynamics_id"],
            }
            mismatch = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            optional_execution_schedule = {
                "torch_intra_op_threads": self.torch_intra_op_threads,
                "torch_inter_op_threads": self.torch_inter_op_threads,
                "fixed_partner_executor_workers": (
                    self.fixed_partner_executor_workers
                ),
            }
            mismatch.update({
                key: (metadata.get(key), value)
                for key, value in optional_execution_schedule.items()
                if key in metadata and metadata.get(key) != value
            })
            if mismatch:
                raise RuntimeError(
                    f"Full server resume execution contract mismatch: {mismatch}"
                )

        self._context = mp.get_context("spawn")
        self._shared = FullSharedStepBuffers.create(self._context, environments)
        self._views = self._shared.views()
        self._response_queue = self._context.Queue()
        self._command_queues = []
        self._workers = []
        self._serial = 0
        self._closed = False
        self._full_workers_ready = False
        for assignment in self.assignments:
            commands = self._context.Queue()
            process = self._context.Process(
                target=full_rollout_worker_main,
                args=(
                    assignment.worker_id,
                    assignment.env_ids,
                    commands,
                    self._response_queue,
                    self._shared,
                ),
                name=f"full-rollout-worker-{assignment.worker_id}",
                daemon=False,
            )
            process.start()
            self._command_queues.append(commands)
            self._workers.append(process)
        boot = self._wait_boot()
        self.worker_pids = [int(row["pid"]) for row in boot]
        if any(bool(row["cuda_initialized"]) for row in boot):
            self.close()
            raise RuntimeError("a Full rollout worker initialized CUDA")

        try:
            super().__init__(**kwargs)
            if self.fixed_partner_executor_workers is not None:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                self._executor = ThreadPoolExecutor(
                    max_workers=self.fixed_partner_executor_workers
                )
            partner_device = str(self.backends[0].partners.device)
            if str(self.device).startswith("cuda") and not partner_device.startswith(
                "cuda"
            ):
                raise RuntimeError("Full fixed partner must remain on CUDA")
            adapter_states = [adapter.training_state() for adapter in self.adapters]
            backend_states = [backend.training_state() for backend in self.backends]
            parameter_sha = self.backends[0].partners.assert_frozen(full_hash=True)
            if any(
                backend.partners.assert_frozen(full_hash=True) != parameter_sha
                for backend in self.backends
            ):
                raise RuntimeError("Full fixed-partner parameters differ across envs")
            environment_config = {}
            for env_id, backend_state in enumerate(backend_states):
                worker_state = copy.deepcopy(backend_state)
                worker_state["partners"]["torch_rng"] = (
                    torch.Generator(device="cpu")
                    .manual_seed(self.seed + env_id + 191)
                    .get_state()
                )
                environment_config[env_id] = {
                    "seed": self.seed + env_id,
                    "horizon": int(self.active_horizons[env_id]),
                    "backend_state": worker_state,
                    "adapter_state": adapter_states[env_id],
                }
            common = {
                "episode_length": self.episode_length,
                "enable_predictor": bool(
                    self.horizon_sampler is not None or int(self.horizon or 0) > 0
                ),
                "partner_parameter_sha256": parameter_sha,
                "partner_checkpoint_sha256": FIXED_HSP_PARTNER_SHA256,
            }
            configs = [{
                **common,
                "environments": {
                    env_id: environment_config[env_id]
                    for env_id in assignment.env_ids
                },
            } for assignment in self.assignments]
            startup = self._initialize_workers(configs)
            if any(bool(row["cuda_initialized"]) for row in startup):
                raise RuntimeError("a Full rollout worker initialized CUDA")
            rows = [item for row in startup for item in row.get("payload", ())]
            self._sync_runtime_rows(rows)
            self._populate_full_previews()
            self.adapters = [
                MainFullObservationAdapter(
                    self, env_id, backend, adapter_states[env_id]
                )
                for env_id, backend in enumerate(self.backends)
            ]
            self.backend = self.backends[0]
            self.adapter = self.adapters[0]
            self._views["reset_flags"].fill(0)
            self._views["reset_horizons"][:] = self.active_horizons
            initial_current = tuple(
                np.stack(values, axis=0)
                for values in zip(
                    *(adapter.prepare_decision() for adapter in self.adapters)
                )
            )
            if any(
                not np.array_equal(left, right)
                for left, right in zip(initial_current, self._prepared)
            ):
                raise RuntimeError("Full worker initial observation parity failed")
            self._full_workers_ready = True
            self._current_prepared_trace = self._capture_prepared_trace(
                self._prepared
            )
            self._install_policy_trace()
        except BaseException:
            self.close()
            raise
        self.last_rollout_trace: list[dict[str, object]] = []
        self.last_execution_timings: dict[str, float] = {}
        self._reset_execution_timings()

    def _get_response(self) -> dict[str, object]:
        try:
            response = self._response_queue.get(
                timeout=WORKER_RESPONSE_TIMEOUT_SECONDS
            )
        except queue.Empty as error:
            dead = [process.pid for process in self._workers if not process.is_alive()]
            raise RuntimeError(f"Full rollout worker timeout; dead={dead}") from error
        if response.get("phase") == "error":
            raise RuntimeError(
                f"Full rollout worker {response.get('worker_id')} failed: "
                f"{response.get('error')}\n{response.get('traceback')}"
            )
        return response

    def _wait_boot(self):
        responses = [self._get_response() for _ in self._workers]
        ordered = sorted(responses, key=lambda row: int(row["worker_id"]))
        if (
            {row.get("phase") for row in ordered} != {"boot"}
            or [tuple(row["env_ids"]) for row in ordered]
            != [assignment.env_ids for assignment in self.assignments]
        ):
            raise RuntimeError("invalid Full worker boot contract")
        return ordered

    def _initialize_workers(self, configs):
        self._serial += 1
        serial = self._serial
        for commands, config in zip(self._command_queues, configs):
            commands.put(("initialize", serial, config))
        responses = [self._get_response() for _ in self._workers]
        ordered = sorted(responses, key=lambda row: int(row["worker_id"]))
        if (
            any(row.get("phase") != "startup" for row in ordered)
            or any(int(row.get("serial", -1)) != serial for row in ordered)
        ):
            raise RuntimeError("invalid Full worker initialization response")
        return ordered

    def _phase(self, phase: str, *arguments):
        self._serial += 1
        serial = self._serial
        started = time.perf_counter()
        for commands in self._command_queues:
            commands.put((phase, serial, *arguments))
        responses = [self._get_response() for _ in self._workers]
        wall = time.perf_counter() - started
        expected = set(range(self.rollout_workers))
        if (
            {int(row.get("worker_id", -1)) for row in responses} != expected
            or any(row.get("phase") != phase for row in responses)
            or any(int(row.get("serial", -1)) != serial for row in responses)
        ):
            raise RuntimeError(f"invalid Full {phase} barrier response")
        ordered = sorted(responses, key=lambda row: int(row["worker_id"]))
        worker_elapsed = max(
            (float(row.get("elapsed_seconds", 0.0)) for row in ordered),
            default=0.0,
        )
        worker_wait = max(
            (float(row.get("queue_wait_seconds", 0.0)) for row in ordered),
            default=0.0,
        )
        if hasattr(self, "_timing_accumulator"):
            self._timing_accumulator["main_process_wait_seconds"] += wall
            self._timing_accumulator["ipc_barrier_overhead_seconds"] += max(
                0.0, wall - worker_elapsed
            )
            self._timing_accumulator["worker_wait_seconds"] += worker_wait
        return ordered

    def _reset_execution_timings(self) -> None:
        self._timing_accumulator = {
            "learner_inference_seconds": 0.0,
            "main_process_wait_seconds": 0.0,
            "worker_wait_seconds": 0.0,
            "ipc_barrier_overhead_seconds": 0.0,
        }

    def _install_policy_trace(self) -> None:
        original = self.policy.get_actions

        def traced_get_actions(*arguments, **keywords):
            started = time.perf_counter()
            output = original(*arguments, **keywords)
            self._timing_accumulator["learner_inference_seconds"] += (
                time.perf_counter() - started
            )
            if self.server_trace:
                share, observation, actor_rnn, critic_rnn, masks, available = (
                    arguments
                )
                values, actions, logs, next_actor, next_critic = output
                snapshot = _actor_pipeline_snapshot(
                    self.policy.actor,
                    observation,
                    actor_rnn,
                    masks,
                    available,
                    actions.detach().cpu().numpy(),
                )
                self._last_actor_trace = {
                    "actor_observation_sha256": _array_hash(observation),
                    "critic_observation_sha256": _array_hash(share),
                    "actor_pre_rnn_sha256": _array_hash(actor_rnn),
                    "critic_pre_rnn_sha256": _array_hash(critic_rnn),
                    "recurrent_mask_sha256": _array_hash(masks),
                    "available_actions_sha256": _array_hash(available),
                    "raw_logits_sha256": _array_hash(snapshot["raw_logits"]),
                    "probabilities_sha256": _array_hash(snapshot["probabilities"]),
                    "actions": actions.detach().cpu().numpy().reshape(-1).tolist(),
                    "action_log_probs": logs.detach().cpu().numpy().reshape(-1).tolist(),
                    "actor_candidate_rnn_sha256": _array_hash(next_actor),
                    "critic_candidate_rnn_sha256": _array_hash(next_critic),
                    "values_sha256": _array_hash(values),
                }
            return output

        self.policy.get_actions = traced_get_actions

    def _sync_runtime_rows(self, rows) -> None:
        ordered = sorted(rows, key=lambda row: int(row["env_id"]))
        if [int(row["env_id"]) for row in ordered] != list(
            range(self.n_rollout_threads)
        ):
            raise RuntimeError("Full worker runtime environment ordering changed")
        for row in ordered:
            env = int(row["env_id"])
            backend = self.backends[env]
            if int(row["horizon"]) != int(backend.horizon):
                raise RuntimeError("Full worker/main horizon mismatch")
            backend.state = copy.deepcopy(row["state"])
            backend.episode_step = int(row["episode_step"])
            backend.environment_steps = int(row["environment_steps"])
            backend.state_generation = int(row["state_generation"])
            backend.reward_shaping_factor = float(row["reward_shaping_factor"])
            backend.self_history = copy.deepcopy(row["self_history"])
            backend.partner_history = copy.deepcopy(row["partner_history"])
            backend._invalidate_observation_cache()

    def _capture_prepared_trace(self, prepared) -> dict[str, object]:
        if not self.server_trace:
            return {}
        observations, critics, available = prepared
        rows = []
        for env_id, backend in enumerate(self.backends):
            horizon = int(backend.horizon)
            hidden_channels = observations[env_id, ..., 21 + horizon :]
            if not np.array_equal(hidden_channels, np.zeros_like(hidden_channels)):
                raise RuntimeError("Full preview exposed channels beyond horizon")
            rows.append({
                "env_id": env_id,
                "horizon": horizon,
                "predictor_requested": bool(
                    self.horizon_sampler is not None or int(self.horizon or 0) > 0
                ),
                "predictor_input_sha256": _object_hash((
                    backend.state,
                    backend.self_history,
                    backend.partner_history,
                )),
                "predictor_output_sha256": _array_hash(backend.preview_planes()),
                "preview_channel_sha256": [
                    _array_hash(observations[env_id, ..., 21 + index])
                    for index in range(4)
                ],
                "actor_observation_sha256": _array_hash(observations[env_id]),
                "critic_observation_sha256": _array_hash(critics[env_id]),
                "available_actions_sha256": _array_hash(available[env_id]),
            })
        return {"environments": rows}

    def _prepare_decisions(self):
        if not getattr(self, "_full_workers_ready", False):
            return super()._prepare_decisions()
        self._views["reset_horizons"][:] = self.active_horizons
        responses = self._phase("prepare")
        rows = [item for response in responses for item in response["payload"]]
        self._sync_runtime_rows(rows)
        self._views["reset_flags"].fill(0)
        prepared = super()._prepare_decisions()
        self._current_prepared_trace = self._capture_prepared_trace(prepared)
        return prepared

    def _main_partner_actions(self, observations, legal_masks):
        rows = list(zip(self.backends, observations, legal_masks))
        if self._executor is None:
            return np.asarray([
                rows[0][0].partners.action(rows[0][1], rows[0][2])
            ], dtype=np.int64)
        return np.asarray(
            list(self._executor.map(
                lambda row: row[0].partners.action(row[1], row[2]), rows
            )),
            dtype=np.int64,
        )

    def _verify_partner_after_step(self, backend, before: str) -> None:
        try:
            after = backend.partners.assert_frozen()
        except RuntimeError:
            backend.partner_step_guard_failures += 1
            raise
        finally:
            backend.partner_step_guard_checks += 1
        if before != after:
            backend.partner_step_guard_failures += 1
            raise RuntimeError("Full fixed partner changed during remote env step")

    def _commit_decisions(self, actions: np.ndarray):
        if not getattr(self, "_full_workers_ready", False):
            return super()._commit_decisions(actions)
        selected = np.asarray(actions).reshape(-1).astype(np.int64)
        observations = self._views["partner_observations"].copy()
        legal_masks = self._views["partner_available_actions"].copy()
        before_hashes = [backend.begin_partner_step() for backend in self.backends]
        partner_pre_hidden = (
            [_array_hash(backend.partners.hidden) for backend in self.backends]
            if self.server_trace else []
        )
        partner_rng_before = (
            [
                _array_hash(backend.partners.torch_rng.get_state())
                for backend in self.backends
            ]
            if self.server_trace else []
        )
        self._synchronize_inference_device()
        partner_started = time.perf_counter()
        partner_actions = self._main_partner_actions(observations, legal_masks)
        self._synchronize_inference_device()
        self._inference_counters["partner_inference_seconds"] += (
            time.perf_counter() - partner_started
        )
        self._inference_counters["partner_forward_calls"] += self.n_rollout_threads
        for backend, before in zip(self.backends, before_hashes):
            self._verify_partner_after_step(backend, before)
        self._views["learner_actions"][:] = selected
        self._views["partner_actions"][:] = partner_actions
        env_started = time.perf_counter()
        responses = self._phase(
            "commit", float(self.backends[0].reward_shaping_factor)
        )
        self._inference_counters["env_step_seconds"] += (
            time.perf_counter() - env_started
        )
        rows = [item for response in responses for item in response["payload"]]
        self._sync_runtime_rows(rows)
        ordered = sorted(rows, key=lambda row: int(row["env_id"]))
        results = [row["result"] for row in ordered]
        for adapter in self.adapters:
            adapter.actionnet_online_forwards += 1
            adapter.task_decisions += 1
        if self.server_trace:
            actor_trace = getattr(self, "_last_actor_trace", {})
            self.last_rollout_trace.append({
                "rollout_step": int(self.buffer.step),
                "prepared": copy.deepcopy(self._current_prepared_trace),
                "actor": copy.deepcopy(actor_trace),
                "partner_observation_sha256": [
                    _array_hash(value) for value in observations
                ],
                "partner_available_actions_sha256": [
                    _array_hash(value) for value in legal_masks
                ],
                "partner_pre_rnn_sha256": partner_pre_hidden,
                "partner_rng_before_sha256": partner_rng_before,
                "partner_actions": partner_actions.tolist(),
                "learner_actions": selected.tolist(),
                "results": [{
                    "optimization_reward": float(result["optimization_reward"]),
                    "sparse_team_reward": float(result["sparse_team_reward"]),
                    "native_shaped_reward": float(result["native_shaped_reward"]),
                    "weighted_shaped_reward": float(result["weighted_shaped_reward"]),
                    "deliveries": int(result["deliveries"]),
                    "done": bool(result["done"]),
                } for result in results],
                "backend_state_sha256": [
                    _object_hash((
                        row["state"], row["self_history"], row["partner_history"],
                        row["episode_step"], row["environment_steps"],
                        row["state_generation"], row["horizon"],
                    ))
                    for row in ordered
                ],
            })
        return results

    def _training_state(self) -> dict[str, Any]:
        state = super()._training_state()
        state["server_parallel_state"] = {
            "execution_backend": FULL_SERVER_PARALLEL_VERSION,
            "rollout_workers": self.rollout_workers,
            "torch_intra_op_threads": self.torch_intra_op_threads,
            "torch_inter_op_threads": self.torch_inter_op_threads,
            "fixed_partner_executor_workers": self.fixed_partner_executor_workers,
            "assignments": {
                str(item.worker_id): list(item.env_ids)
                for item in self.assignments
            },
        }
        return state

    def save(self, path: Path, *, checkpoint_role: str | None = None):
        metadata = super().save(path, checkpoint_role=checkpoint_role)
        output = Path(path)
        payload = torch.load(output, map_location="cpu", weights_only=False)
        payload["metadata"].update(
            execution_backend=FULL_SERVER_PARALLEL_VERSION,
            rollout_workers=self.rollout_workers,
            torch_intra_op_threads=self.torch_intra_op_threads,
            torch_inter_op_threads=self.torch_inter_op_threads,
            fixed_partner_executor_workers=self.fixed_partner_executor_workers,
            worker_assignment={
                str(item.worker_id): list(item.env_ids)
                for item in self.assignments
            },
        )
        torch.save(payload, output)
        return {
            **payload["metadata"],
            "checkpoint_sha256": file_sha256(output),
        }

    def _runtime_reports(self):
        responses = self._phase("metrics")
        rows = [item for response in responses for item in response["payload"]]
        return sorted(rows, key=lambda row: int(row["env_id"]))

    def run_update(self) -> dict[str, Any]:
        self.last_rollout_trace = []
        self._reset_execution_timings()
        self._phase("verify")
        result = super().run_update()
        self._phase("verify")
        reports = self._runtime_reports()
        for env_id, row in enumerate(reports):
            adapter = row["adapter"]
            if adapter != self.adapters[env_id].training_state():
                raise RuntimeError("Full worker/main adapter counter mismatch")
            backend = row["backend"]
            if int(backend["environment_steps"]) != int(
                self.backends[env_id].environment_steps
            ):
                raise RuntimeError("Full worker/main environment-step mismatch")
            if int(backend["horizon"]) != int(self.backends[env_id].horizon):
                raise RuntimeError("Full worker/main final horizon mismatch")
            if int(backend["partner_step_guard_failures"]) != 0:
                raise RuntimeError("Full worker fixed-partner guard failed")
        self.last_execution_timings = dict(self._timing_accumulator)
        result.update({
            "execution_backend": FULL_SERVER_PARALLEL_VERSION,
            "rollout_workers": self.rollout_workers,
            "worker_pids": list(self.worker_pids),
            "worker_partner_guard_failures": sum(
                int(row["backend"]["partner_step_guard_failures"])
                for row in reports
            ),
            **self.last_execution_timings,
        })
        return result

    def startup_report(self) -> dict[str, object]:
        return {
            "server_parallel_version": FULL_SERVER_PARALLEL_VERSION,
            "observation_mode": "full",
            "rollout_workers": self.rollout_workers,
            "n_rollout_threads": self.n_rollout_threads,
            "torch_intra_op_threads": torch.get_num_threads(),
            "torch_inter_op_threads": torch.get_num_interop_threads(),
            "fixed_partner_executor_workers": (
                self.fixed_partner_executor_workers
                if self.fixed_partner_executor_workers is not None
                else self.n_rollout_threads
            ),
            "worker_assignments": {
                str(item.worker_id): list(item.env_ids)
                for item in self.assignments
            },
            "hsp_initial_checkpoint_sha256": HSP_SOURCE_CHECKPOINT_SHA256,
            "fixed_partner_sha256": FIXED_HSP_PARTNER_SHA256,
            "cooking_contract": cooking_contract(),
            "inference_batching": self.inference_batching,
            "partner_inference_schedule": "batch1_per_env",
            "fixed_partner_device": str(self.backends[0].partners.device),
            "predictor_inference_schedule": (
                "none"
                if self.horizon_sampler is None and int(self.horizon) == 0
                else "one_shared_env_batch_per_environment_step"
            ),
            "multiprocessing_start_method": self._context.get_start_method(),
            "device": str(self.device),
            "worker_pids": list(self.worker_pids),
        }

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._phase("shutdown")
        except Exception:
            pass
        for process in self._workers:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for commands in self._command_queues:
            commands.close()
        self._response_queue.close()
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
