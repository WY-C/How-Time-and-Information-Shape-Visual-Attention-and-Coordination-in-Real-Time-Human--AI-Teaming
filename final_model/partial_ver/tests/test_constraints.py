from argparse import Namespace
from dataclasses import FrozenInstanceError
import random

import numpy as np
import pytest
import torch

from final_model.partial_ver.constraints import (
    BASELINE_CONSTRAINT_CONFIG,
    CANONICAL_PERIPHERAL_SIGMA_DEG,
    CANONICAL_TUNED_EMMA_TIMING,
    FULL_CONSTRAINT_CONFIG,
    FITTED_EMMA_CONSTRAINT_CONFIG,
    HumanLikeConstraintConfig,
    LEGACY_PERIPHERAL_CONTRACT_ID,
    LEGACY_PREVIEW_OPACITY_CONTRACT_ID,
    PeripheralVisionConfig,
    PreviewOpacityConfig,
    constraint_config_from_metadata,
    resolve_constraint_config,
)
from final_model.partial_ver.gaze import DEFAULT_EMMA_TIMING, EmmaTimingConfig
from final_model.partial_ver.peripheral import (
    LEGACY_PERIPHERAL_PERCEPTION_CONTRACT_ID,
)


def args(**overrides):
    values = {
        "constraint_profile": None,
        "peripheral_mode": None,
        "peripheral_sigma_deg": None,
        "emma_mode": None,
        "emma_prep_ms": None,
        "emma_exec_intercept_ms": None,
        "emma_exec_slope_ms_per_degree": None,
        "emma_dwell_ms": None,
        "preview_opacity_mode": None,
        "observation_mode": "partial",
    }
    values.update(overrides)
    return Namespace(**values)


def test_no_arguments_use_fitted_emma_and_explicit_baseline_stays_available():
    implicit = resolve_constraint_config(args())
    explicit = resolve_constraint_config(args(constraint_profile="baseline"))
    assert implicit is FITTED_EMMA_CONSTRAINT_CONFIG
    assert implicit.variant_id == "e"
    assert implicit.emma == CANONICAL_TUNED_EMMA_TIMING
    assert explicit == BASELINE_CONSTRAINT_CONFIG
    assert explicit.variant_id == explicit.label == "b0"
    assert explicit.peripheral.mode == "none"
    assert explicit.emma is DEFAULT_EMMA_TIMING
    assert explicit.preview_opacity.mode == "off"


def test_full_observation_implicit_constraint_remains_non_gaze_baseline():
    assert resolve_constraint_config(args(observation_mode="full")) is (
        BASELINE_CONSTRAINT_CONFIG
    )


@pytest.mark.parametrize(
    "override",
    [
        {"peripheral_mode": "global"},
        {"peripheral_sigma_deg": 3.0},
        {"emma_mode": "default"},
        {"preview_opacity_mode": "off"},
    ],
)
def test_noncustom_profile_rejects_every_component_override(override):
    with pytest.raises(ValueError, match="rejects component overrides"):
        resolve_constraint_config(args(constraint_profile="baseline", **override))


def test_full_profile_override_fails_before_unavailable_component_check():
    with pytest.raises(ValueError, match="rejects component overrides"):
        resolve_constraint_config(args(
            constraint_profile="full", emma_mode="default"
        ))


def test_custom_is_explicit_and_preserves_emma_validation():
    baseline_components = resolve_constraint_config(args(
        constraint_profile="custom", peripheral_mode="none",
        emma_mode="default", preview_opacity_mode="off",
    ))
    assert baseline_components.profile == "custom"
    assert baseline_components.variant_id == "b0"
    with pytest.raises(ValueError, match="requires explicit"):
        resolve_constraint_config(args(
            constraint_profile="custom", peripheral_mode="none",
            emma_mode="default",
        ))
    with pytest.raises(ValueError, match="all four"):
        resolve_constraint_config(args(
            constraint_profile="custom", peripheral_mode="none",
            emma_mode="tuned", emma_prep_ms=135,
            preview_opacity_mode="off",
        ))


def test_canonical_tuned_maps_to_e_and_arbitrary_timing_maps_to_custom():
    tuned = resolve_constraint_config(args(
        constraint_profile="custom", peripheral_mode="none",
        emma_mode="tuned", emma_prep_ms=135,
        emma_exec_intercept_ms=44.8732080135,
        emma_exec_slope_ms_per_degree=3.83832565515,
        emma_dwell_ms=60, preview_opacity_mode="off",
    ))
    assert tuned.emma == CANONICAL_TUNED_EMMA_TIMING
    assert tuned.variant_id == tuned.label == "e"
    arbitrary = HumanLikeConstraintConfig(
        profile="custom", peripheral=PeripheralVisionConfig("none"),
        emma=EmmaTimingConfig("tuned", 1, 2, 3, 4),
        preview_opacity=PreviewOpacityConfig("off"),
    )
    assert arbitrary.variant_id == "custom"
    assert arbitrary.label.startswith("custom_")


def test_full_and_p2o_are_exact_compositions_and_none_actual_is_rejected():
    assert FULL_CONSTRAINT_CONFIG.variant_id == "full"
    assert FULL_CONSTRAINT_CONFIG.peripheral.mode == "global"
    assert FULL_CONSTRAINT_CONFIG.emma == CANONICAL_TUNED_EMMA_TIMING
    assert FULL_CONSTRAINT_CONFIG.preview_opacity.mode == "actual"
    assert resolve_constraint_config(args(constraint_profile="full")) == (
        FULL_CONSTRAINT_CONFIG
    )
    p1 = resolve_constraint_config(args(
        constraint_profile="custom", peripheral_mode="local",
        peripheral_sigma_deg=3.0,
        emma_mode="default", preview_opacity_mode="off",
    ))
    assert p1.variant_id == "p1"
    p2o = resolve_constraint_config(args(
        constraint_profile="custom", peripheral_mode="global",
        peripheral_sigma_deg=3.0,
        emma_mode="default", preview_opacity_mode="actual",
    ))
    assert p2o.variant_id == "p2o"
    with pytest.raises(ValueError, match="scientifically redundant"):
        resolve_constraint_config(args(
            constraint_profile="custom", peripheral_mode="none",
            emma_mode="default", preview_opacity_mode="actual",
        ))


@pytest.mark.parametrize(
    ("peripheral", "emma", "opacity", "variant"),
    [
        ("local", DEFAULT_EMMA_TIMING, "off", "p1"),
        ("global", DEFAULT_EMMA_TIMING, "off", "p2"),
        ("global", CANONICAL_TUNED_EMMA_TIMING, "off", "p2"),
        ("global", DEFAULT_EMMA_TIMING, "actual", "p2o"),
        ("global", CANONICAL_TUNED_EMMA_TIMING, "actual", "full"),
    ],
)
def test_variant_ids_depend_only_on_exact_resolved_components(
    peripheral, emma, opacity, variant
):
    config = HumanLikeConstraintConfig(
        profile="custom", peripheral=PeripheralVisionConfig(
            peripheral,
            CANONICAL_PERIPHERAL_SIGMA_DEG
            if peripheral != "none" else None,
        ),
        emma=emma, preview_opacity=PreviewOpacityConfig(opacity),
    )
    assert config.variant_id == config.label == variant


def test_peripheral_sigma_is_active_only_and_defines_canonical_variants():
    for mode, variant in (("local", "p1"), ("global", "p2")):
        config = resolve_constraint_config(args(
            constraint_profile="custom", peripheral_mode=mode,
            peripheral_sigma_deg=3.0, emma_mode="default",
            preview_opacity_mode="off",
        ))
        assert config.variant_id == variant
        assert config.peripheral.sigma_deg == CANONICAL_PERIPHERAL_SIGMA_DEG
        custom = resolve_constraint_config(args(
            constraint_profile="custom", peripheral_mode=mode,
            peripheral_sigma_deg=4.0, emma_mode="default",
            preview_opacity_mode="off",
        ))
        assert custom.variant_id == "custom"
    with pytest.raises(ValueError, match="requires finite positive sigma"):
        resolve_constraint_config(args(
            constraint_profile="custom", peripheral_mode="local",
            emma_mode="default", preview_opacity_mode="off",
        ))
    with pytest.raises(ValueError, match="inactive sigma"):
        resolve_constraint_config(args(
            constraint_profile="custom", peripheral_mode="none",
            peripheral_sigma_deg=3.0, emma_mode="default",
            preview_opacity_mode="off",
        ))


def test_legacy_and_current_metadata_use_one_reconstruction_helper():
    assert constraint_config_from_metadata({}) == BASELINE_CONSTRAINT_CONFIG
    legacy_tuned = constraint_config_from_metadata(
        CANONICAL_TUNED_EMMA_TIMING.metadata()
    )
    assert legacy_tuned.profile == "custom"
    assert legacy_tuned.variant_id == "e"
    assert constraint_config_from_metadata(legacy_tuned.metadata()) == legacy_tuned
    old_constraint_framework = dict(BASELINE_CONSTRAINT_CONFIG.metadata())
    for key in (
        "peripheral_sigma_deg", "peripheral_perception_contract_id",
        "peripheral_probability_contract_id", "peripheral_rng_contract_id",
    ):
        old_constraint_framework.pop(key)
    old_constraint_framework["peripheral_contract_id"] = (
        LEGACY_PERIPHERAL_CONTRACT_ID
    )
    for key in tuple(old_constraint_framework):
        if key.startswith("preview_") and key not in {
            "preview_opacity_mode", "preview_opacity_contract_id"
        }:
            old_constraint_framework.pop(key)
    old_constraint_framework["preview_opacity_contract_id"] = (
        LEGACY_PREVIEW_OPACITY_CONTRACT_ID
    )
    assert constraint_config_from_metadata(old_constraint_framework) == (
        BASELINE_CONSTRAINT_CONFIG
    )
    inactive_old_sampling = dict(BASELINE_CONSTRAINT_CONFIG.metadata())
    inactive_old_sampling["peripheral_perception_contract_id"] = (
        LEGACY_PERIPHERAL_PERCEPTION_CONTRACT_ID
    )
    assert constraint_config_from_metadata(inactive_old_sampling) == (
        BASELINE_CONSTRAINT_CONFIG
    )
    with pytest.raises(RuntimeError, match="partial constraint metadata"):
        constraint_config_from_metadata({"peripheral_mode": "none"})


def test_old_opacity_contract_is_ignored_only_when_opacity_is_off():
    old_contract = "peripheral_preview_step_after_spatial_success_bernoulli_v1"
    inactive = dict(BASELINE_CONSTRAINT_CONFIG.metadata())
    inactive["preview_opacity_contract_id"] = old_contract
    assert constraint_config_from_metadata(inactive) == BASELINE_CONSTRAINT_CONFIG

    active = dict(FULL_CONSTRAINT_CONFIG.metadata())
    active["preview_opacity_contract_id"] = old_contract
    with pytest.raises(RuntimeError, match="unsupported preview opacity contract"):
        constraint_config_from_metadata(active)


def test_config_is_immutable_and_resolution_consumes_no_rng():
    with pytest.raises(FrozenInstanceError):
        BASELINE_CONSTRAINT_CONFIG.profile = "custom"
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    resolve_constraint_config(args())
    assert random.getstate() == python_before
    assert all(
        np.array_equal(left, right)
        for left, right in zip(np.random.get_state(), numpy_before)
    )
    assert torch.equal(torch.random.get_rng_state(), torch_before)
