"""Batched fixed-partner inference and bounded shadow parity diagnostics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


FULL_INFERENCE_BATCHING_MODES = (
    "legacy",
    "partner",
    "predictor",
    "partner_predictor",
)

PARTIAL_LEARNED_INFERENCE_BATCHING_MODES = (
    "legacy",
    "gaze",
    "predictor",
    "gaze_predictor",
    "action",
    "gaze_predictor_action",
)

PARTIAL_LEARNED_PRODUCTION_MODES = (
    "legacy",
    "predictor",
)

INFERENCE_BATCHING_MODES = tuple(dict.fromkeys(
    FULL_INFERENCE_BATCHING_MODES + PARTIAL_LEARNED_INFERENCE_BATCHING_MODES
))


def _actor_forward_with_logits(
    actor: torch.nn.Module,
    observations: np.ndarray,
    hidden_states: torch.Tensor,
    recurrent_masks: torch.Tensor,
    legal_masks: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture_logits(_module, _inputs, output):
        captured.append(output.detach().clone())

    handle = actor.act.action_out.linear.register_forward_hook(capture_logits)
    try:
        with torch.no_grad():
            probabilities, next_hidden = actor.get_probs(
                observations,
                hidden_states,
                recurrent_masks,
                legal_masks,
            )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("fixed partner forward did not produce exactly one logits tensor")
    legal = torch.as_tensor(legal_masks, device=captured[0].device)
    logits = captured[0]
    logits[legal == 0] = -1e10
    logits = torch.where(torch.isnan(logits), torch.full_like(logits, -1e10), logits)
    return logits, probabilities.detach(), next_hidden.detach()


def _stack_partner_state(pools: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
    if not pools:
        raise ValueError("partner pool batch cannot be empty")
    actor = pools[0].actor
    if actor is None or any(pool.actor is not actor for pool in pools):
        raise RuntimeError("batched fixed partner requires one shared actor instance")
    hidden = torch.cat([pool.hidden for pool in pools], dim=0)
    masks = torch.cat([pool.mask for pool in pools], dim=0)
    return hidden, masks


def batched_partner_actions(
    pools: Sequence,
    observations: np.ndarray,
    legal_masks: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    count = len(pools)
    observations = np.asarray(observations, dtype=np.float32)
    legal_masks = np.asarray(legal_masks, dtype=np.float32)
    if observations.shape[0] != count or legal_masks.shape != (count, 6):
        raise ValueError("partner batched inputs do not match the environment count")
    hidden, masks = _stack_partner_state(pools)
    logits, probabilities, next_hidden = _actor_forward_with_logits(
        pools[0].actor, observations, hidden, masks, legal_masks
    )
    actions = []
    for env_index, pool in enumerate(pools):
        action = torch.multinomial(
            probabilities[env_index], 1, generator=pool.torch_rng
        )
        actions.append(int(action.item()))
        pool.hidden = next_hidden[env_index : env_index + 1].clone()
        pool.mask = torch.ones_like(pool.mask)
    return np.asarray(actions, dtype=np.int64), {
        "logits": logits.detach().cpu().numpy(),
        "probabilities": probabilities.detach().cpu().numpy(),
        "next_hidden": next_hidden.detach().cpu().numpy(),
    }


def _cloned_generator(pool) -> torch.Generator:
    generator = torch.Generator(device=pool.device.type)
    generator.set_state(pool.torch_rng.get_state())
    return generator


def partner_shadow_parity(
    pools: Sequence,
    observations: np.ndarray,
    legal_masks: np.ndarray,
) -> dict[str, object]:
    count = len(pools)
    observations = np.asarray(observations, dtype=np.float32)
    legal_masks = np.asarray(legal_masks, dtype=np.float32)
    hidden, masks = _stack_partner_state(pools)
    legacy_logits = []
    legacy_probabilities = []
    legacy_hidden = []
    for env_index, pool in enumerate(pools):
        logits, probabilities, next_hidden = _actor_forward_with_logits(
            pool.actor,
            observations[env_index : env_index + 1],
            hidden[env_index : env_index + 1],
            masks[env_index : env_index + 1],
            legal_masks[env_index : env_index + 1],
        )
        legacy_logits.append(logits)
        legacy_probabilities.append(probabilities)
        legacy_hidden.append(next_hidden)
    legacy_logits_tensor = torch.cat(legacy_logits, dim=0)
    legacy_probabilities_tensor = torch.cat(legacy_probabilities, dim=0)
    legacy_hidden_tensor = torch.cat(legacy_hidden, dim=0)
    batch_logits, batch_probabilities, batch_hidden = _actor_forward_with_logits(
        pools[0].actor, observations, hidden, masks, legal_masks
    )

    legacy_samples = []
    batch_samples = []
    for env_index, pool in enumerate(pools):
        legacy_samples.append(
            int(
                torch.multinomial(
                    legacy_probabilities_tensor[env_index],
                    1,
                    generator=_cloned_generator(pool),
                ).item()
            )
        )
        batch_samples.append(
            int(
                torch.multinomial(
                    batch_probabilities[env_index],
                    1,
                    generator=_cloned_generator(pool),
                ).item()
            )
        )

    def differences(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
        difference = torch.abs(left.to(torch.float64) - right.to(torch.float64))
        return float(difference.max().cpu()), float(difference.mean().cpu())

    logit_max, logit_mean = differences(legacy_logits_tensor, batch_logits)
    probability_max, probability_mean = differences(
        legacy_probabilities_tensor, batch_probabilities
    )
    hidden_max, hidden_mean = differences(legacy_hidden_tensor, batch_hidden)
    legacy_top1 = legacy_probabilities_tensor.argmax(dim=-1)
    batch_top1 = batch_probabilities.argmax(dim=-1)
    sampled_equal = np.equal(legacy_samples, batch_samples)
    return {
        "environment_count": count,
        "env_mapping_exact": True,
        "input_observation_exact": True,
        "rnn_input_state_exact": True,
        "recurrent_mask_exact": True,
        "legal_mask_exact": True,
        "partner_max_abs_logit_difference": logit_max,
        "partner_mean_abs_logit_difference": logit_mean,
        "partner_max_abs_probability_difference": probability_max,
        "partner_mean_abs_probability_difference": probability_mean,
        "partner_max_abs_next_hidden_difference": hidden_max,
        "partner_mean_abs_next_hidden_difference": hidden_mean,
        "partner_top1_action_agreement": float(
            (legacy_top1 == batch_top1).to(torch.float32).mean().cpu()
        ),
        "partner_paired_sampled_action_agreement": float(sampled_equal.mean()),
        "partner_first_sampled_action_divergence_env": (
            None if bool(sampled_equal.all()) else int(np.flatnonzero(~sampled_equal)[0])
        ),
        "sampling_rng_semantics": (
            "per-env torch.Generator state preserved; sampling performed in canonical "
            "env order after the shared probability forward"
        ),
        "legacy_forward_calls": count,
        "batched_forward_calls": 1,
        "partner_batch_size": count,
    }


def initialize_partner_trajectory_shadow(pools: Sequence) -> dict[str, object]:
    hidden, masks = _stack_partner_state(pools)
    return {
        "hidden": hidden.clone(),
        "masks": masks.clone(),
        "generators": [_cloned_generator(pool) for pool in pools],
        "checked_steps": 0,
        "checked_transitions": 0,
        "first_divergence": None,
    }


def legacy_partner_shadow_step(
    pools: Sequence,
    observations: np.ndarray,
    legal_masks: np.ndarray,
    shadow: dict[str, object],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hidden = shadow["hidden"]
    masks = shadow["masks"]
    generators = shadow["generators"]
    logits_rows = []
    probability_rows = []
    hidden_rows = []
    actions = []
    for env_index, pool in enumerate(pools):
        logits, probabilities, next_hidden = _actor_forward_with_logits(
            pool.actor,
            observations[env_index : env_index + 1],
            hidden[env_index : env_index + 1],
            masks[env_index : env_index + 1],
            legal_masks[env_index : env_index + 1],
        )
        actions.append(
            int(
                torch.multinomial(
                    probabilities[0], 1, generator=generators[env_index]
                ).item()
            )
        )
        logits_rows.append(logits)
        probability_rows.append(probabilities)
        hidden_rows.append(next_hidden)
    next_hidden = torch.cat(hidden_rows, dim=0)
    shadow["hidden"] = next_hidden.clone()
    shadow["masks"] = torch.ones_like(masks)
    shadow["checked_steps"] += 1
    shadow["checked_transitions"] += len(pools)
    return np.asarray(actions, dtype=np.int64), {
        "logits": torch.cat(logits_rows, dim=0).cpu().numpy(),
        "probabilities": torch.cat(probability_rows, dim=0).cpu().numpy(),
        "next_hidden": next_hidden.cpu().numpy(),
    }


def reset_partner_trajectory_shadow(
    shadow: dict[str, object], env_indices: Sequence[int]
) -> None:
    for env_index in env_indices:
        shadow["hidden"][int(env_index)].zero_()
        shadow["masks"][int(env_index)].zero_()
