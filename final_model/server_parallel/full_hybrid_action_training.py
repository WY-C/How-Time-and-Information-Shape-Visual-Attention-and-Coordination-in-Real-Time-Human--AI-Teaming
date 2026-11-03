"""Experimental Full backend with pipelined CPU fixed-partner inference."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import gc
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from final_model.common.action_training import ActionTrainingSession
from final_model.common.mappo import parameter_hash
from final_model.common.partners import (
    FIXED_HSP_PARTNER_SHA256,
    FixedHSPPartnerRepository,
    FrozenPartnerPool,
)
from final_model.common.paths import file_sha256
from final_model.server_parallel.full_action_training import (
    FULL_SERVER_PARALLEL_VERSION,
    ServerParallelFullActionTrainingSession,
    _array_hash,
)


FULL_HYBRID_VERSION = "server_parallel_full_hybrid_cpu_partner_v1"


def _partner_candidate(
    env_id: int,
    pool: FrozenPartnerPool,
    observation: np.ndarray,
    legal_mask: np.ndarray,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    rng_state: torch.Tensor,
) -> dict[str, object]:
    """Run one CPU batch-1 partner step without mutating canonical state."""
    started = time.perf_counter()
    generator = torch.Generator(device="cpu")
    generator.set_state(rng_state)
    with torch.no_grad():
        probabilities, next_hidden = pool.actor.get_probs(
            observation[None], hidden, mask, legal_mask[None]
        )
        action = torch.multinomial(probabilities[0], 1, generator=generator)
    finished = time.perf_counter()
    return {
        "env_id": int(env_id),
        "action": int(action.item()),
        "next_hidden": next_hidden.detach().clone(),
        "next_mask": torch.ones_like(mask),
        "next_rng_state": generator.get_state().clone(),
        "started": started,
        "finished": finished,
    }


class ServerParallelFullHybridActionTrainingSession(
    ServerParallelFullActionTrainingSession
):
    """CUDA learner/predictor plus pipelined CPU batch-1 fixed partner."""

    def __init__(
        self,
        *,
        partner_executor_workers: int = 4,
        runtime_executor_workers: int = 4,
        **kwargs,
    ) -> None:
        if str(kwargs.get("device")) != "cuda":
            raise ValueError("Full hybrid benchmark requires CUDA learner/predictor")
        if kwargs.get("resume_checkpoint") is not None:
            raise ValueError("experimental Full hybrid resume is not supported")
        partner_workers = int(partner_executor_workers)
        runtime_workers = int(runtime_executor_workers)
        if partner_workers <= 0 or runtime_workers <= 0:
            raise ValueError("hybrid executor worker counts must be positive")
        checkpoint = kwargs.get("fixed_partner_checkpoint")
        if (
            checkpoint is None
            or file_sha256(Path(checkpoint)) != FIXED_HSP_PARTNER_SHA256
        ):
            raise ValueError("Full hybrid requires canonical HSP S2-s24 seed3")

        self.partner_executor_workers = partner_workers
        self.runtime_executor_workers = runtime_workers
        self._partner_executor: ThreadPoolExecutor | None = None
        self._pending_partner: dict[str, object] | None = None
        self._update_in_progress = False
        super().__init__(**kwargs)
        try:
            old_executor = self._executor
            if old_executor is not None:
                old_executor.shutdown(wait=True)
            self._executor = ThreadPoolExecutor(max_workers=runtime_workers)
            self._partner_executor = ThreadPoolExecutor(
                max_workers=partner_workers,
                thread_name_prefix="full-cpu-partner",
            )
            self._install_cpu_partner_pools(Path(checkpoint))
            self._start_partner_inference()
        except BaseException:
            self.close()
            raise

    def _install_cpu_partner_pools(self, checkpoint: Path) -> None:
        repository = FixedHSPPartnerRepository(
            device="cpu", checkpoint=checkpoint
        )
        expected_parameter_hash = self.fixed_partner_provenance[
            "fixed_partner_parameter_sha256"
        ]
        actual_parameter_hash = repository.assert_frozen(
            self.backends[0].active_partner, full_hash=True
        )
        if actual_parameter_hash != expected_parameter_hash:
            raise RuntimeError("CPU/GPU fixed-partner parameter hash mismatch")
        for backend in self.backends:
            original = backend.partners
            if (
                torch.count_nonzero(original.hidden).item() != 0
                or torch.count_nonzero(original.mask).item() != 0
            ):
                raise RuntimeError("hybrid CPU partner replacement must be step-zero")
            replacement = FrozenPartnerPool(
                seed=backend.seed,
                device="cpu",
                requested=self.partner,
                repository=repository,
            )
            if dict(replacement.counts) != dict(original.counts):
                raise RuntimeError("hybrid partner identity/count initialization changed")
            backend.partners = replacement
            backend.last_partner_full_hash_baseline = actual_parameter_hash
            backend.last_partner_full_hash_actual = actual_parameter_hash
        self.backend = self.backends[0]
        self.partner_parity = None
        gc.collect()
        torch.cuda.empty_cache()

    def _reset_execution_timings(self) -> None:
        super()._reset_execution_timings()
        self._timing_accumulator.update({
            "cpu_partner_inference_seconds": 0.0,
            "cpu_partner_executor_task_seconds": 0.0,
            "partner_pipeline_elapsed_seconds": 0.0,
            "partner_commit_wait_seconds": 0.0,
        })

    def _partner_state_hashes(self) -> list[dict[str, str]]:
        return [{
            "hidden": _array_hash(backend.partners.hidden),
            "mask": _array_hash(backend.partners.mask),
            "rng": _array_hash(backend.partners.torch_rng.get_state()),
        } for backend in self.backends]

    def _start_partner_inference(self) -> None:
        if self._pending_partner is not None:
            raise RuntimeError("hybrid partner inference already pending")
        if self._partner_executor is None:
            raise RuntimeError("hybrid partner executor is not initialized")
        observations = self._views["partner_observations"].copy()
        legal_masks = self._views["partner_available_actions"].copy()
        pre_states = self._partner_state_hashes()
        submitted = time.perf_counter()
        futures: list[Future] = []
        for env_id, backend in enumerate(self.backends):
            pool = backend.partners
            if pool.device.type != "cpu":
                raise RuntimeError("hybrid fixed partner escaped CPU")
            futures.append(self._partner_executor.submit(
                _partner_candidate,
                env_id,
                pool,
                observations[env_id].copy(),
                legal_masks[env_id].copy(),
                pool.hidden.detach().clone(),
                pool.mask.detach().clone(),
                pool.torch_rng.get_state().clone(),
            ))
        self._pending_partner = {
            "observations": observations,
            "legal_masks": legal_masks,
            "pre_states": pre_states,
            "submitted": submitted,
            "futures": futures,
        }

    def _prepare_decisions(self):
        if not getattr(self, "_full_workers_ready", False):
            return super()._prepare_decisions()
        self._views["reset_horizons"][:] = self.active_horizons
        responses = self._phase("prepare")
        rows = [item for response in responses for item in response["payload"]]
        self._sync_runtime_rows(rows)
        self._views["reset_flags"].fill(0)
        final_rollout_observation = (
            self._update_in_progress
            and int(self.buffer.step) == self.rollout_length - 1
        )
        if not final_rollout_observation:
            self._start_partner_inference()
        prepared = ActionTrainingSession._prepare_decisions(self)
        self._current_prepared_trace = self._capture_prepared_trace(prepared)
        return prepared

    def _main_partner_actions(self, observations, legal_masks):
        pending = self._pending_partner
        if pending is None:
            raise RuntimeError("hybrid partner result was not started")
        if (
            not np.array_equal(observations, pending["observations"])
            or not np.array_equal(legal_masks, pending["legal_masks"])
        ):
            raise RuntimeError("hybrid partner input changed before commit")
        if self._partner_state_hashes() != pending["pre_states"]:
            raise RuntimeError("hybrid partner state advanced before commit")
        wait_started = time.perf_counter()
        candidates = [future.result() for future in pending["futures"]]
        wait_finished = time.perf_counter()
        if [row["env_id"] for row in candidates] != list(
            range(self.n_rollout_threads)
        ):
            raise RuntimeError("hybrid partner future/env association changed")
        actions = np.asarray(
            [row["action"] for row in candidates], dtype=np.int64
        )
        for env_id, row in enumerate(candidates):
            pool = self.backends[env_id].partners
            pool.hidden = row["next_hidden"].clone()
            pool.mask = row["next_mask"].clone()
            pool.torch_rng.set_state(row["next_rng_state"])
        task_started = min(float(row["started"]) for row in candidates)
        task_finished = max(float(row["finished"]) for row in candidates)
        self._timing_accumulator["cpu_partner_inference_seconds"] += (
            task_finished - task_started
        )
        self._timing_accumulator["cpu_partner_executor_task_seconds"] += sum(
            float(row["finished"]) - float(row["started"])
            for row in candidates
        )
        self._timing_accumulator["partner_pipeline_elapsed_seconds"] += (
            wait_finished - float(pending["submitted"])
        )
        self._timing_accumulator["partner_commit_wait_seconds"] += (
            wait_finished - wait_started
        )
        self._pending_partner = None
        return actions

    def _commit_decisions(self, actions: np.ndarray):
        results = super()._commit_decisions(actions)
        if self.server_trace:
            trace = self.last_rollout_trace[-1]
            trace["partner_post_rnn_sha256"] = [
                _array_hash(backend.partners.hidden) for backend in self.backends
            ]
            trace["partner_post_mask_sha256"] = [
                _array_hash(backend.partners.mask) for backend in self.backends
            ]
            trace["partner_rng_after_sha256"] = [
                _array_hash(backend.partners.torch_rng.get_state())
                for backend in self.backends
            ]
        return results

    def run_update(self) -> dict[str, Any]:
        if self._pending_partner is None:
            self._start_partner_inference()
        self._update_in_progress = True
        try:
            result = super().run_update()
        finally:
            self._update_in_progress = False
        if self._pending_partner is not None:
            raise RuntimeError("hybrid left speculative partner state after update")
        result.update({
            "execution_backend": FULL_HYBRID_VERSION,
            "base_execution_backend": FULL_SERVER_PARALLEL_VERSION,
            "learner_device": "cuda",
            "critic_ppo_device": "cuda",
            "predictor_device": "cuda",
            "fixed_partner_device": "cpu",
            "partner_executor_workers": self.partner_executor_workers,
            "runtime_executor_workers": self.runtime_executor_workers,
            "partner_inference_schedule": "cpu_batch1_per_env_pipelined",
            "partner_batch_size": 1,
        })
        return result

    def startup_report(self) -> dict[str, object]:
        return {
            **super().startup_report(),
            "server_parallel_version": FULL_HYBRID_VERSION,
            "base_execution_backend": FULL_SERVER_PARALLEL_VERSION,
            "learner_device": "cuda",
            "critic_ppo_device": "cuda",
            "predictor_device": "cuda",
            "fixed_partner_device": "cpu",
            "partner_executor_workers": self.partner_executor_workers,
            "runtime_executor_workers": self.runtime_executor_workers,
            "partner_inference_schedule": "cpu_batch1_per_env_pipelined",
        }

    def save(self, *_args, **_kwargs):
        raise RuntimeError("experimental Full hybrid checkpoints are disabled")

    def close(self) -> None:
        executor = getattr(self, "_partner_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
            self._partner_executor = None
        self._pending_partner = None
        super().close()
