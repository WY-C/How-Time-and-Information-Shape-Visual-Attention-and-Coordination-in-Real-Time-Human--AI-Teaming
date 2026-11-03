"""Canonical preview-opacity perception derived from the human-study renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import math

from final_model.server_parallel.gaze_v1 import tile_to_action_index


PREVIEW_OPACITY_CONTRACT_ID = (
    "peripheral_preview_tile_shared_max_alpha_detection_v2"
)
PREVIEW_OPACITY_RNG_CONTRACT_ID = (
    "numpy_pcg64_preview_opacity_per_adapter_seed_offset_v1"
)
PREVIEW_OPACITY_RNG_SEED_OFFSET = 103_000_027

PREVIEW_ALPHA_PROBABILITY_CONTRACT_ID = "pygame_uint8_alpha_div255_v1"
PREVIEW_RENDERER_CONTRACT_ID = "human_study_rank1_highlight_full_stable_v1"
PREVIEW_RENDERER_SOURCE_PATH = "main.py"
PREVIEW_RENDERER_GIT_BLOB = "7b8935d6e0636c0e94774d0936f3d53186256eb1"
PREVIEW_RENDERER_SHA256 = (
    "8663771d26588c9a698d36d9c49c7c60fe186751d738f613e2eb9b3971b7e50a"
)
PREVIEW_RENDERER_ALPHA_SCALE = 255
PREVIEW_RENDERER_BASE_ALPHA = 250
PREVIEW_RENDERER_FADE_FRACTION = 0.72
PREVIEW_RENDERER_MIN_ALPHA = 26


def alpha_to_detection_probability(alpha: int | float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or value < 0.0 or value > 255.0:
        raise ValueError("renderer alpha must be finite and in [0, 255]")
    return float(value / PREVIEW_RENDERER_ALPHA_SCALE)


def rendered_preview_alphas(displayed_horizon: int) -> tuple[int, int, int, int]:
    horizon = int(displayed_horizon)
    if horizon not in {0, 1, 2, 3, 4}:
        raise ValueError("displayed preview horizon must be in [0, 4]")
    values = [0, 0, 0, 0]
    for step_index in range(horizon):
        fade_ratio = step_index / max(1, horizon - 1)
        values[step_index] = max(
            PREVIEW_RENDERER_MIN_ALPHA,
            min(255, int(round(
                PREVIEW_RENDERER_BASE_ALPHA
                * (1.0 - PREVIEW_RENDERER_FADE_FRACTION * fade_ratio)
            ))),
        )
    return tuple(values)


def rendered_preview_probabilities(
    displayed_horizon: int,
) -> tuple[float, float, float, float]:
    return tuple(
        alpha_to_detection_probability(alpha)
        for alpha in rendered_preview_alphas(displayed_horizon)
    )


def preview_opacity_metadata() -> dict[str, object]:
    def fill_alphas(horizon):
        return [
            0 if alpha == 0 else min(190, max(24, int(round(alpha * 0.52))))
            for alpha in rendered_preview_alphas(horizon)
        ]

    return {
        "preview_opacity_contract_id": PREVIEW_OPACITY_CONTRACT_ID,
        "preview_alpha_probability_contract_id": (
            PREVIEW_ALPHA_PROBABILITY_CONTRACT_ID
        ),
        "preview_opacity_rng_contract_id": PREVIEW_OPACITY_RNG_CONTRACT_ID,
        "preview_opacity_rng_seed_offset": PREVIEW_OPACITY_RNG_SEED_OFFSET,
        "preview_opacity_scope": (
            "peripheral_spatial_success_positive_preview_tile_shared_event"
        ),
        "preview_opacity_overlap_sampling": (
            "one_draw_per_positive_tile_at_max_positive_step_alpha"
        ),
        "preview_opacity_success_semantics": "preserve_all_positive_h_steps",
        "preview_opacity_miss_semantics": "clear_all_h_steps_on_observed_tile",
        "preview_renderer_contract_id": PREVIEW_RENDERER_CONTRACT_ID,
        "preview_renderer_source_path": PREVIEW_RENDERER_SOURCE_PATH,
        "preview_renderer_git_blob": PREVIEW_RENDERER_GIT_BLOB,
        "preview_renderer_sha256": PREVIEW_RENDERER_SHA256,
        "preview_renderer_alpha_scale": PREVIEW_RENDERER_ALPHA_SCALE,
        "preview_renderer_probability_alpha_component": (
            "primary_step_tile_alpha_used_by_highlight_outline"
        ),
        "preview_renderer_alpha_h2": list(rendered_preview_alphas(2)),
        "preview_renderer_alpha_h4": list(rendered_preview_alphas(4)),
        "preview_renderer_probability_h2": list(
            rendered_preview_probabilities(2)
        ),
        "preview_renderer_probability_h4": list(
            rendered_preview_probabilities(4)
        ),
        "preview_renderer_fill_alpha_h2": fill_alphas(2),
        "preview_renderer_fill_alpha_h4": fill_alphas(4),
        "preview_renderer_movement_interaction_alpha_shared": True,
        "preview_renderer_overlap_composition": "maximum_source_alpha_per_tile",
        "preview_renderer_animation_progress": 1.0,
        "preview_renderer_ranked_paths_shown": 1,
    }


@dataclass(frozen=True)
class PreviewOpacityDetection:
    perceived_preview4: np.ndarray
    opportunity_tile_indices: tuple[int, ...]
    opportunity_steps: tuple[int, ...]
    opportunity_probabilities: tuple[float, ...]
    opportunity_positive_steps: tuple[tuple[int, ...], ...]
    opportunity_max_alphas: tuple[int, ...]
    attempt_tile_indices: tuple[int, ...]
    attempt_steps: tuple[int, ...]
    attempt_probabilities: tuple[float, ...]
    success_mask: tuple[bool, ...]

    @property
    def success_tile_indices(self) -> tuple[int, ...]:
        return tuple(
            tile for tile, success in zip(
                self.attempt_tile_indices, self.success_mask
            ) if success
        )

    @property
    def success_steps(self) -> tuple[int, ...]:
        return tuple(
            step for step, success in zip(self.attempt_steps, self.success_mask)
            if success
        )

    @property
    def miss_tile_indices(self) -> tuple[int, ...]:
        return tuple(
            tile for tile, success in zip(
                self.attempt_tile_indices, self.success_mask
            ) if not success
        )

    @property
    def miss_steps(self) -> tuple[int, ...]:
        return tuple(
            step for step, success in zip(self.attempt_steps, self.success_mask)
            if not success
        )


def sample_preview_opacity(
    *, mode: str, displayed_horizon: int, displayed_preview4: np.ndarray,
    peripheral_success_tiles: Sequence[Sequence[int]],
    rng: np.random.Generator,
) -> PreviewOpacityDetection:
    displayed = np.asarray(displayed_preview4, dtype=np.float32)
    if displayed.shape != (8, 5, 4):
        raise ValueError("displayed preview must be (8,5,4)")
    perceived = displayed.copy()
    if mode not in {"off", "actual"}:
        raise ValueError("preview opacity mode must be off or actual")

    horizon = int(displayed_horizon)
    probabilities = rendered_preview_probabilities(horizon)
    if np.any(displayed[..., horizon:] != 0.0):
        raise ValueError("opacity input contains preview above displayed horizon")
    tiles = tuple(sorted(
        (tuple(map(int, tile)) for tile in peripheral_success_tiles),
        key=tile_to_action_index,
    ))
    attempt_tiles: list[int] = []
    attempt_steps: list[int] = []
    attempt_probabilities: list[float] = []
    success_mask: list[bool] = []
    opportunity_tiles: list[int] = []
    opportunity_steps: list[int] = []
    opportunity_probabilities: list[float] = []
    opportunity_positive_steps: list[tuple[int, ...]] = []
    opportunity_max_alphas: list[int] = []
    alphas = rendered_preview_alphas(horizon)
    for tile in tiles:
        tile_index = tile_to_action_index(tile)
        positive_steps = tuple(
            step_index + 1 for step_index in range(horizon)
            if displayed[tile][step_index] > 0.0
        )
        if not positive_steps:
            continue
        max_step = min(
            positive_steps, key=lambda step: (-alphas[step - 1], step)
        )
        max_alpha = alphas[max_step - 1]
        probability = probabilities[max_step - 1]
        opportunity_tiles.append(tile_index)
        opportunity_steps.append(max_step)
        opportunity_probabilities.append(probability)
        opportunity_positive_steps.append(positive_steps)
        opportunity_max_alphas.append(max_alpha)
        if mode == "off":
            continue
        success = bool(float(rng.random()) < probability)
        attempt_tiles.append(tile_index)
        attempt_steps.append(max_step)
        attempt_probabilities.append(probability)
        success_mask.append(success)
        if not success:
            perceived[tile] = 0.0
    return PreviewOpacityDetection(
        np.ascontiguousarray(perceived, dtype=np.float32),
        tuple(opportunity_tiles), tuple(opportunity_steps),
        tuple(opportunity_probabilities),
        tuple(opportunity_positive_steps), tuple(opportunity_max_alphas),
        tuple(attempt_tiles), tuple(attempt_steps),
        tuple(attempt_probabilities), tuple(success_mask),
    )
