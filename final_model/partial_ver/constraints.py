"""Immutable composition and provenance for Partial human-like constraints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from final_model.partial_ver.gaze import (
    DEFAULT_EMMA_TIMING,
    EmmaTimingConfig,
    emma_timing_from_metadata,
    resolve_emma_timing,
)
from final_model.partial_ver.peripheral import (
    LEGACY_PERIPHERAL_PERCEPTION_CONTRACT_ID,
    PERIPHERAL_PERCEPTION_CONTRACT_ID,
    PERIPHERAL_PROBABILITY_CONTRACT_ID,
    PERIPHERAL_RNG_CONTRACT_ID,
    PERIPHERAL_EXTENTS,
)
from final_model.partial_ver.preview_opacity import (
    PREVIEW_ALPHA_PROBABILITY_CONTRACT_ID,
    PREVIEW_OPACITY_CONTRACT_ID,
    PREVIEW_OPACITY_RNG_CONTRACT_ID,
    PREVIEW_RENDERER_CONTRACT_ID,
    preview_opacity_metadata,
)


CONSTRAINT_CONTRACT_ID = "partial_human_like_constraint_composition_v1"
PERIPHERAL_CONTRACT_ID = "peripheral_vision_plus_composition_v1"
LEGACY_PERIPHERAL_CONTRACT_ID = "peripheral_vision_reserved_not_implemented_v1"
LEGACY_PREVIEW_OPACITY_CONTRACT_ID = "preview_opacity_reserved_not_implemented_v1"
CANONICAL_PERIPHERAL_SIGMA_DEG = 3.0

CANONICAL_TUNED_EMMA_TIMING = EmmaTimingConfig(
    mode="tuned",
    prep_ms=135.0,
    exec_intercept_ms=44.8732080135,
    exec_slope_ms_per_degree=3.83832565515,
    dwell_ms=60.0,
)


@dataclass(frozen=True)
class PeripheralVisionConfig:
    mode: str
    sigma_deg: float | None = None
    contract_id: str = PERIPHERAL_CONTRACT_ID
    perception_contract_id: str = PERIPHERAL_PERCEPTION_CONTRACT_ID
    probability_contract_id: str = PERIPHERAL_PROBABILITY_CONTRACT_ID
    rng_contract_id: str = PERIPHERAL_RNG_CONTRACT_ID
    extent: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"none", "local", "global"}:
            raise ValueError("peripheral mode must be none, local, or global")
        if self.contract_id != PERIPHERAL_CONTRACT_ID:
            raise ValueError("unsupported peripheral vision contract")
        if self.perception_contract_id != PERIPHERAL_PERCEPTION_CONTRACT_ID:
            raise ValueError("unsupported peripheral perception contract")
        if self.probability_contract_id != PERIPHERAL_PROBABILITY_CONTRACT_ID:
            raise ValueError("unsupported peripheral probability contract")
        if self.rng_contract_id != PERIPHERAL_RNG_CONTRACT_ID:
            raise ValueError("unsupported peripheral RNG contract")
        if self.mode == "none":
            if self.sigma_deg is not None:
                raise ValueError("peripheral none mode requires inactive sigma")
            if self.extent is not None:
                raise ValueError("peripheral none mode requires inactive extent")
        elif (
            self.sigma_deg is None
            or not math.isfinite(float(self.sigma_deg))
            or float(self.sigma_deg) <= 0.0
        ):
            raise ValueError("active peripheral mode requires finite positive sigma")
        if self.extent is not None and self.extent not in PERIPHERAL_EXTENTS:
            raise ValueError(
                f"peripheral extent must be one of {PERIPHERAL_EXTENTS}"
            )


@dataclass(frozen=True)
class PreviewOpacityConfig:
    mode: str
    contract_id: str = PREVIEW_OPACITY_CONTRACT_ID
    alpha_probability_contract_id: str = PREVIEW_ALPHA_PROBABILITY_CONTRACT_ID
    rng_contract_id: str = PREVIEW_OPACITY_RNG_CONTRACT_ID
    renderer_contract_id: str = PREVIEW_RENDERER_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.mode not in {"off", "actual"}:
            raise ValueError("preview opacity mode must be off or actual")
        if self.contract_id != PREVIEW_OPACITY_CONTRACT_ID:
            raise ValueError("unsupported preview opacity contract")
        if self.alpha_probability_contract_id != PREVIEW_ALPHA_PROBABILITY_CONTRACT_ID:
            raise ValueError("unsupported preview alpha probability contract")
        if self.rng_contract_id != PREVIEW_OPACITY_RNG_CONTRACT_ID:
            raise ValueError("unsupported preview opacity RNG contract")
        if self.renderer_contract_id != PREVIEW_RENDERER_CONTRACT_ID:
            raise ValueError("unsupported preview renderer contract")


@dataclass(frozen=True)
class HumanLikeConstraintConfig:
    profile: str
    peripheral: PeripheralVisionConfig
    emma: EmmaTimingConfig
    preview_opacity: PreviewOpacityConfig
    contract_id: str = CONSTRAINT_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.profile not in {"baseline", "full", "custom"}:
            raise ValueError("constraint profile must be baseline, full, or custom")
        if self.contract_id != CONSTRAINT_CONTRACT_ID:
            raise ValueError("unsupported human-like constraint contract")
        components = (
            self.peripheral.mode, self.peripheral.sigma_deg,
            self.peripheral.extent, self.emma, self.preview_opacity.mode
        )
        if self.profile == "baseline" and components != (
            "none", None, None, DEFAULT_EMMA_TIMING, "off"
        ):
            raise ValueError("baseline profile must be exactly none/default/off")
        if self.profile == "full" and components != (
            "global", CANONICAL_PERIPHERAL_SIGMA_DEG, None,
            CANONICAL_TUNED_EMMA_TIMING, "actual"
        ):
            raise ValueError("full profile must be exactly global/canonical-tuned/actual")
        if self.preview_opacity.mode == "actual" and self.peripheral.mode == "none":
            raise ValueError(
                "preview opacity actual requires active peripheral vision; "
                "none+actual is scientifically redundant"
            )

    @property
    def variant_id(self) -> str:
        components = (
            self.peripheral.mode,
            self.peripheral.sigma_deg,
            self.peripheral.extent,
            self.emma,
            self.preview_opacity.mode,
        )
        variants = {
            ("none", None, None, DEFAULT_EMMA_TIMING, "off"): "b0",
            ("local", CANONICAL_PERIPHERAL_SIGMA_DEG, None,
             DEFAULT_EMMA_TIMING, "off"): "p1",
            ("global", CANONICAL_PERIPHERAL_SIGMA_DEG, None,
             DEFAULT_EMMA_TIMING, "off"): "p2",
            ("global", CANONICAL_PERIPHERAL_SIGMA_DEG, None,
             CANONICAL_TUNED_EMMA_TIMING, "off"): "p2",
            ("none", None, None, CANONICAL_TUNED_EMMA_TIMING, "off"): "e",
            ("global", CANONICAL_PERIPHERAL_SIGMA_DEG, None,
             DEFAULT_EMMA_TIMING, "actual"): "p2o",
            ("global", CANONICAL_PERIPHERAL_SIGMA_DEG, None,
             CANONICAL_TUNED_EMMA_TIMING, "actual"): "full",
        }
        return variants.get(components, "custom")

    @property
    def label(self) -> str:
        if self.variant_id != "custom":
            return self.variant_id
        payload = json.dumps(
            self.component_metadata(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"custom_{hashlib.sha256(payload).hexdigest()[:10]}"

    def component_metadata(self) -> dict[str, object]:
        result = {
            "peripheral_mode": self.peripheral.mode,
            "peripheral_sigma_deg": self.peripheral.sigma_deg,
            "peripheral_contract_id": self.peripheral.contract_id,
            "peripheral_perception_contract_id": (
                self.peripheral.perception_contract_id
            ),
            "peripheral_probability_contract_id": (
                self.peripheral.probability_contract_id
            ),
            "peripheral_rng_contract_id": self.peripheral.rng_contract_id,
            **self.emma.metadata(),
            "preview_opacity_mode": self.preview_opacity.mode,
            **preview_opacity_metadata(),
        }
        if self.peripheral.extent is not None:
            result.update({
                "peripheral_extent": self.peripheral.extent,
                "fixation_shape": "plus",
                "exact_core_definition": "manhattan_distance_le_1_plus_5",
            })
        return result

    def metadata(self) -> dict[str, object]:
        return {
            "constraint_contract_id": self.contract_id,
            "constraint_profile": self.profile,
            "constraint_variant_id": self.variant_id,
            "constraint_label": self.label,
            **self.component_metadata(),
        }


def canonical_constraint_config(profile: str) -> HumanLikeConstraintConfig:
    if profile == "baseline":
        return HumanLikeConstraintConfig(
            profile="baseline",
            peripheral=PeripheralVisionConfig("none"),
            emma=DEFAULT_EMMA_TIMING,
            preview_opacity=PreviewOpacityConfig("off"),
        )
    if profile == "full":
        return HumanLikeConstraintConfig(
            profile="full",
            peripheral=PeripheralVisionConfig(
                "global", CANONICAL_PERIPHERAL_SIGMA_DEG
            ),
            emma=CANONICAL_TUNED_EMMA_TIMING,
            preview_opacity=PreviewOpacityConfig("actual"),
        )
    raise ValueError("canonical constraint profile must be baseline or full")


BASELINE_CONSTRAINT_CONFIG = canonical_constraint_config("baseline")
FULL_CONSTRAINT_CONFIG = canonical_constraint_config("full")


def constraint_config_for_emma(
    emma: EmmaTimingConfig,
) -> HumanLikeConstraintConfig:
    return HumanLikeConstraintConfig(
        profile="baseline" if emma == DEFAULT_EMMA_TIMING else "custom",
        peripheral=PeripheralVisionConfig("none"),
        emma=emma,
        preview_opacity=PreviewOpacityConfig("off"),
    )


FITTED_EMMA_CONSTRAINT_CONFIG = constraint_config_for_emma(
    CANONICAL_TUNED_EMMA_TIMING
)


def validate_constraint_implementation(
    config: HumanLikeConstraintConfig,
) -> None:
    del config


def resolve_constraint_config(args: Any) -> HumanLikeConstraintConfig:
    profile = getattr(args, "constraint_profile", None)
    peripheral_mode = getattr(args, "peripheral_mode", None)
    peripheral_sigma = getattr(args, "peripheral_sigma_deg", None)
    peripheral_extent = getattr(args, "peripheral_extent", None)
    opacity_mode = getattr(args, "preview_opacity_mode", None)
    emma_mode = getattr(args, "emma_mode", None)
    emma_values = {
        "prep_ms": getattr(args, "emma_prep_ms", None),
        "exec_intercept_ms": getattr(args, "emma_exec_intercept_ms", None),
        "exec_slope_ms_per_degree": getattr(
            args, "emma_exec_slope_ms_per_degree", None
        ),
        "dwell_ms": getattr(args, "emma_dwell_ms", None),
    }
    any_emma_value = any(value is not None for value in emma_values.values())

    if profile in {"baseline", "full"}:
        if any(
            value is not None
            for value in (
                peripheral_mode, peripheral_sigma, opacity_mode, emma_mode,
                peripheral_extent,
            )
        ) or any_emma_value:
            raise ValueError(
                f"--constraint-profile {profile} defines the complete component "
                "configuration and rejects component overrides"
            )
        config = canonical_constraint_config(profile)
    elif profile == "custom":
        missing = [
            flag
            for flag, value in (
                ("--peripheral-mode", peripheral_mode),
                ("--emma-mode", emma_mode),
                ("--preview-opacity-mode", opacity_mode),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "--constraint-profile custom requires explicit " + ", ".join(missing)
            )
        config = HumanLikeConstraintConfig(
            profile="custom",
            peripheral=PeripheralVisionConfig(
                str(peripheral_mode),
                None if peripheral_sigma is None else float(peripheral_sigma),
                extent=(
                    None if peripheral_extent is None else str(peripheral_extent)
                ),
            ),
            emma=resolve_emma_timing(str(emma_mode), **emma_values),
            preview_opacity=PreviewOpacityConfig(str(opacity_mode)),
        )
    elif profile is None:
        if (
            peripheral_mode is not None
            or peripheral_sigma is not None
            or peripheral_extent is not None
            or opacity_mode is not None
        ):
            raise ValueError(
                "peripheral/opacity component arguments require "
                "--constraint-profile custom"
            )
        if emma_mode is None and not any_emma_value:
            config = (
                BASELINE_CONSTRAINT_CONFIG
                if getattr(args, "observation_mode", "partial") == "full"
                else FITTED_EMMA_CONSTRAINT_CONFIG
            )
        else:
            mode = "default" if emma_mode is None else str(emma_mode)
            config = constraint_config_for_emma(
                resolve_emma_timing(mode, **emma_values)
            )
    else:
        raise ValueError("constraint profile must be baseline, full, or custom")

    validate_constraint_implementation(config)
    return config


def constraint_config_from_metadata(
    metadata: Mapping[str, Any], *, require_implemented: bool = True
) -> HumanLikeConstraintConfig:
    has_contract = "constraint_contract_id" in metadata
    constraint_only_keys = {
        "constraint_profile", "constraint_variant_id", "constraint_label",
        "peripheral_mode", "peripheral_sigma_deg", "peripheral_contract_id",
        "peripheral_extent", "exact_core_definition",
        "peripheral_perception_contract_id",
        "peripheral_probability_contract_id", "peripheral_rng_contract_id",
        "preview_opacity_mode", "preview_opacity_contract_id",
        "preview_alpha_probability_contract_id",
        "preview_opacity_rng_contract_id", "preview_opacity_rng_seed_offset",
        "preview_opacity_scope", "preview_opacity_overlap_sampling",
        "preview_opacity_success_semantics", "preview_opacity_miss_semantics",
        "preview_renderer_contract_id",
        "preview_renderer_source_path", "preview_renderer_git_blob",
        "preview_renderer_sha256", "preview_renderer_alpha_scale",
        "preview_renderer_probability_alpha_component",
        "preview_renderer_alpha_h2", "preview_renderer_alpha_h4",
        "preview_renderer_probability_h2", "preview_renderer_probability_h4",
        "preview_renderer_fill_alpha_h2", "preview_renderer_fill_alpha_h4",
        "preview_renderer_movement_interaction_alpha_shared",
        "preview_renderer_overlap_composition",
        "preview_renderer_animation_progress",
        "preview_renderer_ranked_paths_shown",
    }
    if not has_contract:
        unexpected = sorted(key for key in constraint_only_keys if key in metadata)
        if unexpected:
            raise RuntimeError(
                f"checkpoint has partial constraint metadata without contract: {unexpected}"
            )
        config = constraint_config_for_emma(
            emma_timing_from_metadata(dict(metadata))
        )
    else:
        if metadata.get("constraint_contract_id") != CONSTRAINT_CONTRACT_ID:
            raise RuntimeError("checkpoint constraint contract mismatch")
        try:
            peripheral_mode = str(metadata.get("peripheral_mode"))
            legacy_peripheral = (
                peripheral_mode == "none"
                and metadata.get("peripheral_contract_id")
                == LEGACY_PERIPHERAL_CONTRACT_ID
                and "peripheral_sigma_deg" not in metadata
                and "peripheral_perception_contract_id" not in metadata
                and "peripheral_probability_contract_id" not in metadata
                and "peripheral_rng_contract_id" not in metadata
            )
            preview_opacity_mode = str(metadata.get("preview_opacity_mode"))
            inactive_old_opacity = (
                preview_opacity_mode == "off"
                and metadata.get("preview_opacity_contract_id")
                != PREVIEW_OPACITY_CONTRACT_ID
            )
            inactive_old_peripheral_perception = (
                peripheral_mode == "none"
                and metadata.get("peripheral_perception_contract_id")
                == LEGACY_PERIPHERAL_PERCEPTION_CONTRACT_ID
            )
            config = HumanLikeConstraintConfig(
                profile=str(metadata.get("constraint_profile")),
                peripheral=PeripheralVisionConfig(
                    peripheral_mode,
                    None if legacy_peripheral else metadata.get("peripheral_sigma_deg"),
                    PERIPHERAL_CONTRACT_ID if legacy_peripheral else str(
                        metadata.get("peripheral_contract_id")
                    ),
                    (
                        PERIPHERAL_PERCEPTION_CONTRACT_ID
                        if legacy_peripheral or inactive_old_peripheral_perception
                        else str(metadata.get("peripheral_perception_contract_id"))
                    ),
                    PERIPHERAL_PROBABILITY_CONTRACT_ID if legacy_peripheral else str(
                        metadata.get("peripheral_probability_contract_id")
                    ),
                    PERIPHERAL_RNG_CONTRACT_ID if legacy_peripheral else str(
                        metadata.get("peripheral_rng_contract_id")
                    ),
                    extent=(
                        None
                        if "peripheral_extent" not in metadata
                        else str(metadata.get("peripheral_extent"))
                    ),
                ),
                emma=emma_timing_from_metadata(dict(metadata)),
                preview_opacity=PreviewOpacityConfig(
                    preview_opacity_mode,
                    PREVIEW_OPACITY_CONTRACT_ID if inactive_old_opacity else str(
                        metadata.get("preview_opacity_contract_id")
                    ),
                    PREVIEW_ALPHA_PROBABILITY_CONTRACT_ID if inactive_old_opacity else str(
                        metadata.get("preview_alpha_probability_contract_id")
                    ),
                    PREVIEW_OPACITY_RNG_CONTRACT_ID if inactive_old_opacity else str(
                        metadata.get("preview_opacity_rng_contract_id")
                    ),
                    PREVIEW_RENDERER_CONTRACT_ID if inactive_old_opacity else str(
                        metadata.get("preview_renderer_contract_id")
                    ),
                ),
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"invalid checkpoint constraint config: {error}") from error
        expected = config.metadata()
        if legacy_peripheral:
            expected = {
                key: value for key, value in expected.items()
                if key not in {
                    "peripheral_sigma_deg", "peripheral_perception_contract_id",
                    "peripheral_probability_contract_id", "peripheral_rng_contract_id",
                }
            }
            expected["peripheral_contract_id"] = LEGACY_PERIPHERAL_CONTRACT_ID
        if inactive_old_opacity:
            expected = {
                key: value for key, value in expected.items()
                if key not in preview_opacity_metadata()
            }
            expected["preview_opacity_contract_id"] = metadata.get(
                "preview_opacity_contract_id"
            )
        if inactive_old_peripheral_perception:
            expected["peripheral_perception_contract_id"] = (
                LEGACY_PERIPHERAL_PERCEPTION_CONTRACT_ID
            )
        mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"checkpoint constraint metadata mismatch: {mismatch}")
    if require_implemented:
        try:
            validate_constraint_implementation(config)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
    return config
