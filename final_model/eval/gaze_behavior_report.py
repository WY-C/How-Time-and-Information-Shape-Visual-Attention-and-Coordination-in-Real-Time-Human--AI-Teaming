"""Markdown and cross-model outputs for gaze behavior evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from final_model.eval.gaze_behavior_metrics import read_csv, write_csv
from final_model.eval.gaze_behavior_stats import paired_cluster_bootstrap


def write_model_summary(
    model_dir: Path, manifest: Mapping[str, Any], analysis: Mapping[str, Any]
) -> None:
    task = analysis.get("task", [])
    spatial = analysis.get("spatial", [])
    enrichment = analysis.get("enrichment", [])
    response = analysis.get("response", [])
    def mean(rows, field):
        values = [float(row[field]) for row in rows if row.get(field) not in (None, "") and np.isfinite(float(row[field]))]
        return float(np.mean(values)) if values else float("nan")

    aoi = analysis.get("aoi", [])
    response_rows = analysis.get("response", [])
    revision = analysis.get("revision", [])
    stabilization = analysis.get("stabilization", [])
    content = analysis.get("content", [])
    proximity = analysis.get("proximity", [])
    economy = analysis.get("economy", [])
    peripheral = analysis.get("peripheral", [])
    opacity = analysis.get("opacity", [])
    cue = analysis.get("cue", [])
    ablation = analysis.get("ablation", [])
    dwell_description = (
        f"{manifest['emma_dwell_ms']} ms fixed"
        if manifest.get("emma_dwell_distribution", "fixed") == "fixed"
        else (
            f"{manifest['emma_dwell_distribution']} "
            f"(floor_p={manifest['emma_dwell_floor_probability']}, "
            f"floor={manifest['emma_dwell_floor_ms']} ms, "
            f"log_mu={manifest['emma_dwell_log_mean']}, "
            f"log_sigma={manifest['emma_dwell_log_sigma']}, "
            f"cap={manifest['emma_dwell_cap_ms']} ms)"
        )
    )
    lines = [
        f"# Gaze behavior: {manifest['model_name']}",
        "",
        "## 1. Model/checkpoint provenance",
        "",
        f"- Checkpoint: `{manifest['checkpoint_absolute_path']}`",
        f"- Completed step: {manifest['checkpoint_completed_step']}",
        f"- Checkpoint SHA-256: `{manifest['checkpoint_sha256']}`",
        f"- Actor/Gaze hashes: `{manifest['actor_sha256']}` / `{manifest['gaze_sha256']}`",
        f"- Memory lifetime: {manifest['memory_lifetime_ms']} ms",
        f"- Constraints: profile={manifest['constraint_profile']}, "
        f"variant={manifest['constraint_variant_id']}, "
        f"peripheral={manifest['peripheral_mode']}, "
        f"peripheral_sigma_deg={manifest.get('peripheral_sigma_deg')}, "
        f"opacity={manifest['preview_opacity_mode']}",
        f"- Preview renderer: {manifest.get('preview_renderer_source_path')} "
        f"SHA-256 `{manifest.get('preview_renderer_sha256')}`; "
        f"H2 alpha={manifest.get('preview_renderer_alpha_h2')}; "
        f"H4 alpha={manifest.get('preview_renderer_alpha_h4')}; "
        "probability uses alpha/255 after peripheral spatial success.",
        f"- EMMA timing: mode={manifest['emma_mode']}, prep={manifest['emma_prep_ms']} ms, "
        f"EXEC={manifest['emma_exec_intercept_ms']} + "
        f"{manifest['emma_exec_slope_ms_per_degree']}×degree ms, dwell={dwell_description}",
        f"- Modes: gaze={manifest['gaze_mode']}, ActionNet={manifest['action_mode']}",
        "",
        "## 2. Task performance",
        "",
    ]
    for row in task:
        lines.append(
            f"- T{row['T']}/H{row['H']}: deliveries {row['deliveries_mean']:.3f}; "
            f"acquisitions/decision {row['acquisitions_per_decision']:.3f}; "
            f"forced-STAY {row['forced_stay_rate']:.3f}."
        )
    lines.extend([
        "", "## 3. Gaze amount and temporal opportunity", "",
        f"Across saved conditions, the frozen policy produced {manifest['acquisition_count']} acquisitions from {manifest['request_count']} requests. Condition-level ActionNet queries/acquisitions per decision are reported in `task_performance.csv`.",
    ])
    if spatial:
        lines.extend(["", "## 4. Spatial gaze distribution", ""])
        for row in spatial:
            lines.append(
                f"- T{row['T']}/H{row['H']}: top-1 share {row['top1_target_share']:.3f}; "
                f"entropy {row['empirical_target_entropy']:.3f}; "
                f"mean saccade {row['mean_saccade_distance_degrees']:.3f}°."
            )
    partner_center = [row for row in aoi if row["aoi"] == "partner" and row["definition"] == "center"]
    preview_center = [row for row in aoi if row["aoi"] == "preview" and row["definition"] == "center"]
    lines.extend([
        "", "## 5. AOI", "",
        f"Center fixation shares averaged {mean(partner_center, 'acquisition_share'):.3f} for Partner and {mean(preview_center, 'acquisition_share'):.3f} for Preview. Center metrics are human-comparable; footprint metrics describe model information access. Raw overlaps remain in the dataset.",
        "", "## 6. Chance-normalized preference", "",
        "Availability-normalized policy mass and sampled-hit enrichment are in `enrichment.csv`; enrichment 1 denotes uniform-target chance. Raw viewing and preference are kept separate.",
        "", "## 7. Inferred action stabilization", "",
        f"Mean top1-stable RT across response-defined decisions was {mean(response_rows, 'top1_stable_rt_ms'):.3f} ms. This is inferred stabilization, not environment action execution.",
        "", "## 8. Action revision association", "",
        "Eligible adjacent-proposal revision risks, risk differences, risk ratios, and episode-seed bootstrap intervals are in `revision_association.csv`. These are associations, not causal effects.",
        "", "## 9. Post-stabilization monitoring", "",
        f"The mean decision-level postAny rate was {mean(response_rows, 'post_any'):.3f}; AT_RESPONSE versus first-POST paired results are in `stabilization_monitoring.csv`.",
        "", "## 10. Content versus freshness", "",
        "Content comparisons exclude channel 20. Frequencies and ActionNet effects for content, freshness-only, combined, and no-effective-change acquisitions are in `content_freshness.csv`.",
        "", "## 11. Proximity and coordination state", "",
        "Partner checking by canonical tile-Manhattan distance and exact low-level holding/pot/resource annotations are in `proximity.csv` and `coordination_states.csv`.",
        "", "## 12. Memory economy", "",
        "Pre-acquisition freshness/validity, newly valid tiles, content novelty, marginal validity, and per-saccade efficiency are in `memory_economy.csv`, split by Time, Horizon, response phase, and AOI.",
        "", "## 13. Peripheral perception", "",
        f"Geometry-derived peripheral attempts, successes, eccentricities, and foveal/peripheral/total AOI acquisition are in `peripheral_summary.csv`. Across ALL-phase condition rows, mean peripheral success rate was {mean([row for row in peripheral if row.get('response_phase') == 'ALL'], 'peripheral_success_rate'):.3f}. `peripheral_actionnet_impact.csv` is associative and does not claim causal effects.",
        "", "## 14. Peripheral preview opacity", "",
        f"Actual-renderer alpha/probability, spatial preview opportunities, conditional opacity successes, and effective H1–H4 detections are in `opacity_summary.csv`. Across ALL-phase rows, mean conditional opacity success rate was {mean([row for row in opacity if row.get('response_phase') == 'ALL'], 'opacity_success_rate'):.3f}. `opacity_actionnet_impact.csv` is associative and does not claim causal effects.",
        "", "## 15. Near-versus-far cue distance", "",
        "H4 first-preview choices use `ReferenceGeometry.visual_angle_degrees`; center and footprint odds ratios with episode-seed bootstrap intervals are in `cue_distance.csv`.",
        "", "## 16. Frozen gaze-input ablation", "",
        "Trajectory-fixed Time, Horizon, freshness, Preview, Partner, Self, and static-only ablations are in `input_ablation.csv`. They re-query the frozen GazeNet without rerunning the environment and are explicitly off-manifold diagnostics.",
        "", "## 17. Limitations", "",
        "Primitive directions do not uniquely identify task goals under a documented canonical planner, so semantic role revision and intent-dependent human-inspired cases are marked unsupported. Results characterize the selected frozen policy/checkpoint, not all trained models. No causal or automatically human-like claim is made.",
    ])
    (model_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_model(
    root: Path, models: Sequence[tuple[Mapping, Mapping]], *,
    bootstrap_iterations: int, bootstrap_seed: int, progress=None,
) -> None:
    cross = root / "cross_model"
    cross.mkdir(parents=True, exist_ok=True)
    metadata = []
    conditions = []
    trends = []
    raw: dict[str, dict[str, list[dict]]] = {}
    if progress is not None:
        progress.cross_step(1, 4, "metadata")
    for manifest, analysis in models:
        metadata.append({
            "model_name": manifest["model_name"],
            "checkpoint": manifest["checkpoint_absolute_path"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "completed_step": manifest["checkpoint_completed_step"],
            "actor_sha256": manifest["actor_sha256"],
            "gaze_sha256": manifest["gaze_sha256"],
            "memory_lifetime_ms": manifest["memory_lifetime_ms"],
            "observation_contract_id": manifest.get("observation_contract_id"),
            "freshness_contract_id": manifest.get("freshness_contract_id"),
            "gaze_action_contract_id": manifest.get("gaze_action_contract_id"),
            "gaze_input_contract_id": manifest.get("gaze_input_contract_id"),
            "gaze_credit_assignment_contract_id": manifest.get("gaze_credit_assignment_contract_id"),
            "constraint_contract_id": manifest.get("constraint_contract_id"),
            "constraint_profile": manifest.get("constraint_profile"),
            "constraint_variant_id": manifest.get("constraint_variant_id"),
            "constraint_label": manifest.get("constraint_label"),
            "peripheral_mode": manifest.get("peripheral_mode"),
            "peripheral_sigma_deg": manifest.get("peripheral_sigma_deg"),
            "peripheral_contract_id": manifest.get("peripheral_contract_id"),
            "peripheral_perception_contract_id": manifest.get("peripheral_perception_contract_id"),
            "peripheral_probability_contract_id": manifest.get("peripheral_probability_contract_id"),
            "peripheral_rng_contract_id": manifest.get("peripheral_rng_contract_id"),
            "preview_opacity_mode": manifest.get("preview_opacity_mode"),
            "preview_opacity_contract_id": manifest.get("preview_opacity_contract_id"),
            "preview_alpha_probability_contract_id": manifest.get("preview_alpha_probability_contract_id"),
            "preview_opacity_rng_contract_id": manifest.get("preview_opacity_rng_contract_id"),
            "preview_opacity_rng_seed_offset": manifest.get("preview_opacity_rng_seed_offset"),
            "preview_renderer_contract_id": manifest.get("preview_renderer_contract_id"),
            "preview_renderer_sha256": manifest.get("preview_renderer_sha256"),
            "preview_renderer_alpha_h2": json.dumps(manifest.get("preview_renderer_alpha_h2")),
            "preview_renderer_alpha_h4": json.dumps(manifest.get("preview_renderer_alpha_h4")),
            "emma_timing_contract_id": manifest.get("emma_timing_contract_id"),
            "emma_mode": manifest.get("emma_mode"),
            "emma_prep_ms": manifest.get("emma_prep_ms"),
            "emma_exec_intercept_ms": manifest.get("emma_exec_intercept_ms"),
            "emma_exec_slope_ms_per_degree": manifest.get("emma_exec_slope_ms_per_degree"),
            "emma_dwell_ms": manifest.get("emma_dwell_ms"),
            "emma_dwell_distribution": manifest.get(
                "emma_dwell_distribution", "fixed"
            ),
            "emma_dwell_floor_probability": manifest.get(
                "emma_dwell_floor_probability"
            ),
            "emma_dwell_floor_ms": manifest.get("emma_dwell_floor_ms"),
            "emma_dwell_log_mean": manifest.get("emma_dwell_log_mean"),
            "emma_dwell_log_sigma": manifest.get("emma_dwell_log_sigma"),
            "emma_dwell_cap_ms": manifest.get("emma_dwell_cap_ms"),
            "emma_dwell_calibration_id": manifest.get(
                "emma_dwell_calibration_id"
            ),
            "entropy_schedule": json.dumps(manifest.get("entropy_schedule")),
            "source_actionnet_provenance": json.dumps(manifest.get("source_full_checkpoint_provenance")),
        })
        conditions.extend({"model_name": manifest["model_name"], **row} for row in analysis.get("condition_summary", []))
        trends.extend({"model_name": manifest["model_name"], **row} for row in analysis.get("trends", []))
        model_dir = root / "models" / manifest["model_name"]
        raw[manifest["model_name"]] = {
            "episodes": read_csv(
                model_dir / "episodes.csv",
                columns=("T", "H", "seed", "deliveries"),
            ),
            "requests": read_csv(
                model_dir / "gaze_requests.csv.gz",
                columns=(
                    "T", "H", "seed",
                    "preview_footprint_mass_enrichment",
                    "partner_footprint_mass_enrichment",
                ),
            ),
        }
    write_csv(cross / "model_metadata.csv", metadata)
    if progress is not None:
        progress.cross_step(2, 4, "condition metrics")
    write_csv(cross / "condition_metrics.csv", conditions)
    if progress is not None:
        progress.cross_step(3, 4, "paired model contrasts")
    contrasts = []
    names = [manifest["model_name"] for manifest, _analysis in models]
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            for time_ms in sorted({int(float(row["T"])) for row in raw[left_name]["episodes"]}):
                for horizon in sorted({int(float(row["H"])) for row in raw[left_name]["episodes"]}):
                    for metric, source, field in (
                        ("deliveries", "episodes", "deliveries"),
                        ("preview_enrichment", "requests", "preview_footprint_mass_enrichment"),
                        ("partner_enrichment", "requests", "partner_footprint_mass_enrichment"),
                    ):
                        def seed_values(model_name):
                            grouped = {}
                            for row in raw[model_name][source]:
                                if int(float(row["T"])) != time_ms or int(float(row["H"])) != horizon:
                                    continue
                                grouped.setdefault(int(float(row["seed"])), []).append(float(row[field]))
                            return {key: float(np.mean(values)) for key, values in grouped.items()}
                        result = paired_cluster_bootstrap(
                            seed_values(left_name), seed_values(right_name),
                            iterations=bootstrap_iterations, seed=bootstrap_seed,
                        )
                        contrasts.append({
                            "left_model": left_name, "right_model": right_name,
                            "T": time_ms, "H": horizon, "metric": metric,
                            "contrast": f"{left_name} - {right_name}", **result,
                            "bootstrap_unit": "paired_evaluation_seed",
                        })
    write_csv(cross / "effect_contrasts.csv", contrasts)
    if progress is not None:
        progress.cross_step(4, 4, "trend matrix")
    write_csv(cross / "trend_matrix.csv", trends)
    lines = ["# Cross-model gaze behavior comparison", ""]
    for row in metadata:
        lines.append(
            f"- **{row['model_name']}**: step {row['completed_step']}, "
            f"memory lifetime {row['memory_lifetime_ms']} ms, checkpoint `{row['checkpoint_sha256']}`."
        )
    lines.extend([
        "", "Trend statuses are descriptive bootstrap-CI classifications. They are not scientific pass/fail labels and are not automatically described as human-like.",
    ])
    (cross / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if progress is not None:
        progress.cross_done()
