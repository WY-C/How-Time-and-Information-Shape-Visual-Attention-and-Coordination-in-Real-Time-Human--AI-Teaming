from pathlib import Path

import pytest

from final_model.common.paths import asset
from final_model.partial_ver.gaze import HUMAN_STOCHASTIC_EMMA_TIMING
from final_model.train import parser, validate_combination

PARTNER = asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt")

def parse(*extra):
    args = parser().parse_args([
        "--observation-mode", "partial", "--horizon", "0",
        "--decision-time", "200", "--fixation-shape", "3x3",
        "--steps", "2", "--rollout-length", "2",
        "--n-rollout-threads", "1",
        "--fixed-partner-checkpoint", str(PARTNER), *extra,
    ])
    validate_combination(args)
    return args


def test_new_valid_component_combinations():
    rule = parse("--gaze-type", "rule_based", "--train-components", "action")
    assert rule.train_action and not rule.train_gaze
    for components in (("gaze",), ("action",), ("action", "gaze")):
        args = parse(
            "--gaze-type", "learned", "--train-components", *components,
            "--source-checkpoint", str(Path("source.pt")),
        )
        assert args.train_action == ("action" in components)
        assert args.train_gaze == ("gaze" in components)


def test_gaze_only_uses_gaze_entropy_defaults_without_changing_action_or_explicit_schedule():
    gaze = parse(
        "--gaze-type", "learned", "--train-components", "gaze",
        "--source-checkpoint", "source.pt",
    )
    assert gaze.entropy_coefs == [0.05, 0.02, 0.01, 0.01]
    assert gaze.entropy_coef_horizons == [0, 5_000_000, 10_000_000, 20_000_000]

    action = parse("--gaze-type", "rule_based", "--train-components", "action")
    assert action.entropy_coefs == [0.2, 0.05, 0.01]
    assert action.entropy_coef_horizons == [0, 10_000_000, 20_000_000]

    explicit = parse(
        "--gaze-type", "learned", "--train-components", "gaze",
        "--source-checkpoint", "source.pt",
        "--entropy-coefs", "0.03", "0.01",
        "--entropy-coef-horizons", "0", "20000000",
    )
    assert explicit.entropy_coefs == [0.03, 0.01]
    assert explicit.entropy_coef_horizons == [0, 20_000_000]


def test_direct_joint_from_full_is_explicit_and_joint_only():
    direct = parse(
        "--gaze-type", "learned", "--train-components", "action", "gaze",
        "--source-checkpoint", "full.pt", "--direct-joint-from-full",
    )
    assert direct.direct_joint_from_full is True
    with pytest.raises(ValueError, match="requires Partial learned"):
        parse(
            "--gaze-type", "learned", "--train-components", "gaze",
            "--source-checkpoint", "full.pt", "--direct-joint-from-full",
        )


def test_full_gaze_cli_selects_frozen_ground_truth_runtime():
    args = parser().parse_args([
        "--observation-mode", "full",
        "--training-stage", "gaze",
        "--shared-horizons", "0", "2", "4",
        "--shared-decision-times", "200", "400", "600",
        "--fixation-shape", "3x3",
        "--source-checkpoint", "full_120m.pt",
        "--fixed-partner-checkpoint", str(PARTNER),
    ])
    validate_combination(args)
    assert args.gaze_type == "learned"
    assert not args.train_action and args.train_gaze
    assert args.ground_truth_policy_input is True
    assert args.inference_batching == "legacy"
    assert args.partial_decision_times == [200, 400, 600]
    assert args.partial_horizons == [0, 2, 4]


def test_partial_memory_lifetime_parser_default_custom_and_validation():
    assert parse(
        "--gaze-type", "rule_based", "--train-components", "action"
    ).memory_lifetime_ms == 2400
    custom = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--memory-lifetime-ms", "2501",
    )
    assert custom.memory_lifetime_ms == 2501
    with pytest.raises(ValueError, match="finite and positive"):
        parse(
            "--gaze-type", "rule_based", "--train-components", "action",
            "--memory-lifetime-ms", "0",
        )
    no_decay = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--memory-decay-mode", "none",
    )
    assert no_decay.memory_decay_mode == "none"
    assert no_decay.memory_lifetime_ms is None
    with pytest.raises(ValueError, match="cannot be combined"):
        parse(
            "--gaze-type", "rule_based", "--train-components", "action",
            "--memory-decay-mode", "none", "--memory-lifetime-ms", "2400",
        )


def test_emma_parser_requires_explicit_complete_tuned_configuration():
    default = parse(
        "--gaze-type", "rule_based", "--train-components", "action"
    )
    assert default.emma_mode == "tuned"
    assert default.emma_prep_ms == 135.0
    assert default.emma_exec_intercept_ms == pytest.approx(44.8732080135)
    assert default.emma_exec_slope_ms_per_degree == pytest.approx(3.83832565515)
    assert default.emma_dwell_ms == 60.0
    legacy = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--emma-mode", "default",
    )
    assert legacy.emma_mode == "default"
    stochastic = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--emma-mode", "human_stochastic",
    )
    assert stochastic.emma_mode == "human_stochastic"
    assert stochastic._resolved_constraint_config.emma is (
        HUMAN_STOCHASTIC_EMMA_TIMING
    )
    tuned = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--emma-mode", "tuned", "--emma-prep-ms", "135",
        "--emma-exec-intercept-ms", "44.8",
        "--emma-exec-slope-ms-per-degree", "3.8",
        "--emma-dwell-ms", "60",
    )
    assert tuned.emma_exec_intercept_ms == 44.8
    with pytest.raises(ValueError, match="all four"):
        parse(
            "--gaze-type", "rule_based", "--train-components", "action",
            "--emma-mode", "tuned", "--emma-prep-ms", "135",
        )


def test_constraint_profile_parser_rules_and_peripheral_components():
    baseline = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--constraint-profile", "baseline",
    )
    assert baseline.constraint_profile == "baseline"
    assert baseline.peripheral_mode == "none"
    assert baseline.peripheral_sigma_deg is None
    assert baseline.emma_mode == "default"
    assert baseline.preview_opacity_mode == "off"
    custom = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--constraint-profile", "custom", "--peripheral-mode", "none",
        "--emma-mode", "default", "--preview-opacity-mode", "off",
    )
    assert custom.constraint_profile == "custom"
    p1 = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--constraint-profile", "custom", "--peripheral-mode", "local",
        "--peripheral-sigma-deg", "3", "--emma-mode", "default",
        "--preview-opacity-mode", "off", "--fixation-shape", "plus",
    )
    assert p1._resolved_constraint_config.variant_id == "p1"
    assert p1.peripheral_sigma_deg == 3.0
    with pytest.raises(ValueError, match="rejects component overrides"):
        parse(
            "--gaze-type", "rule_based", "--train-components", "action",
            "--constraint-profile", "baseline", "--peripheral-mode", "global",
        )
    full = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--constraint-profile", "full", "--fixation-shape", "plus",
    )
    assert full._resolved_constraint_config.variant_id == "full"
    p2o = parse(
        "--gaze-type", "rule_based", "--train-components", "action",
        "--constraint-profile", "custom", "--peripheral-mode", "global",
        "--peripheral-sigma-deg", "3", "--emma-mode", "default",
        "--preview-opacity-mode", "actual", "--fixation-shape", "plus",
    )
    assert p2o._resolved_constraint_config.variant_id == "p2o"
    with pytest.raises(ValueError, match="rejects"):
        parse(
            "--gaze-type", "rule_based", "--train-components", "action",
            "--emma-prep-ms", "135",
        )


def test_partial_requires_canonical_fixed_partner_and_rejects_partner_selection():
    missing = parser().parse_args([
        "--observation-mode", "partial", "--horizon", "0",
        "--decision-time", "200", "--fixation-shape", "3x3",
        "--gaze-type", "rule_based", "--train-components", "action",
    ])
    with pytest.raises(ValueError, match="requires --fixed-partner-checkpoint"):
        validate_combination(missing)
    selected = parser().parse_args([
        "--observation-mode", "partial", "--horizon", "0",
        "--decision-time", "200", "--fixation-shape", "3x3",
        "--gaze-type", "rule_based", "--train-components", "action",
        "--fixed-partner-checkpoint", str(PARTNER), "--partner", "hsp5",
    ])
    with pytest.raises(ValueError, match="does not accept --partner"):
        validate_combination(selected)


@pytest.mark.parametrize("components", [("gaze",), ("action", "gaze")])
def test_rule_gaze_rejects_trainable_gaze(components):
    with pytest.raises(ValueError, match="rule_based"):
        parse("--gaze-type", "rule_based", "--train-components", *components)


def test_legacy_alias_and_conflict():
    action = parse("--training-stage", "action")
    assert action.gaze_type == "rule_based" and action.train_components == ["action"]
    gaze = parse(
        "--training-stage", "gaze", "--source-checkpoint", "source.pt"
    )
    assert gaze.gaze_type == "learned" and gaze.train_components == ["gaze"]
    with pytest.raises(ValueError, match="conflicts"):
        parse(
            "--training-stage", "action", "--gaze-type", "learned",
            "--train-components", "action", "--source-checkpoint", "source.pt",
        )


def test_generic_shared_axes_and_duplicate_rejection():
    args = parser().parse_args([
        "--observation-mode", "partial", "--gaze-type", "rule_based",
        "--train-components", "action", "--shared-horizons", "0", "2", "4",
        "--shared-decision-times", "200", "400", "600",
        "--shared-fixation-shapes", "3x3", "2x2", "plus",
        "--steps", "2", "--rollout-length", "2", "--n-rollout-threads", "1",
        "--fixed-partner-checkpoint", str(PARTNER),
    ])
    validate_combination(args)
    assert len(args.partial_decision_times) * len(args.partial_horizons) * len(args.partial_fixation_shapes) == 27
    duplicate = parser().parse_args([
        "--observation-mode", "partial", "--gaze-type", "rule_based",
        "--train-components", "action", "--horizon", "0",
        "--shared-decision-times", "200", "200",
        "--fixation-shape", "plus", "--steps", "2",
        "--rollout-length", "2", "--n-rollout-threads", "1",
        "--fixed-partner-checkpoint", str(PARTNER),
    ])
    with pytest.raises(ValueError, match="duplicates"):
        validate_combination(duplicate)
