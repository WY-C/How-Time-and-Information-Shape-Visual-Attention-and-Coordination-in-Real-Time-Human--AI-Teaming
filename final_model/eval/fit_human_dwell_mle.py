#!/usr/bin/env python3
"""Fit and validate candidate human-residence samplers on the canonical cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from final_model.eval.audit_human_dwell_calibration import (
    CAP_MS,
    EXEC_INTERCEPT_MS,
    EXEC_SLOPE_MS_PER_DEGREE,
    EXPECTED_PAIR_COUNT,
    FLOOR_MS,
    PREP_MS,
    build_pairs,
)


ANALYSIS_ID = "emma_human_dwell_mle_validation_v1"
DEFAULT_MONTE_CARLO_N = 1_000_000
DEFAULT_MONTE_CARLO_SEED = 20260905
EMPIRICAL_QUANTILE_COUNT = 101
OLD_LOG_MEAN = 5.844786707880467
OLD_LOG_SIGMA = 1.0338073724375734
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
ONSET_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_key(value: float) -> str:
    return f"p{int(round(100 * value)):02d}"


def _summary(values: np.ndarray, quantiles: Sequence[float] = QUANTILES) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    result = {"mean": float(array.mean()), "std": float(array.std(ddof=0))}
    result.update({
        _quantile_key(q): float(value)
        for q, value in zip(quantiles, np.quantile(array, quantiles))
    })
    return result


def _residual_summary(raw: np.ndarray, residence: np.ndarray) -> dict[str, float]:
    result = _summary(raw)
    result.update({
        "p_le_0": float(np.mean(raw <= 0.0)),
        "p_le_60": float(np.mean(raw <= FLOOR_MS)),
        "effective_floor_mass": float(np.mean(residence == FLOOR_MS)),
        "cap_mass": float(np.mean(residence == CAP_MS)),
    })
    return result


def _residence_summary(residence: np.ndarray) -> dict[str, float]:
    result = _summary(residence)
    result.update({
        "effective_floor_mass": float(np.mean(residence == FLOOR_MS)),
        "cap_mass": float(np.mean(residence == CAP_MS)),
    })
    return result


def _distance_diagnostics(human: np.ndarray, sampled: np.ndarray) -> dict[str, float]:
    human = np.asarray(human, dtype=np.float64)
    sampled = np.asarray(sampled, dtype=np.float64)
    human_q = np.quantile(human, QUANTILES)
    sampled_q = np.quantile(sampled, QUANTILES)
    return {
        "ks_statistic": float(stats.ks_2samp(human, sampled).statistic),
        "wasserstein_ms": float(stats.wasserstein_distance(human, sampled)),
        "mean_absolute_quantile_error_ms": float(np.mean(np.abs(human_q - sampled_q))),
    }


def _assert_reference_pairs(pairs: pd.DataFrame, reference_path: Path) -> dict[str, Any]:
    if not reference_path.is_file():
        raise RuntimeError(f"canonical reference pair artifact is missing: {reference_path}")
    reference = pd.read_csv(reference_path)
    keys = ["pid", "trial_id", "source_fixation_id", "fixation_id"]
    numeric = ["onset_gap_ms", "visual_angle_deg", "residual_ms"]
    actual = pairs[keys + numeric].reset_index(drop=True)
    expected = reference[keys + numeric].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual[keys], expected[keys], check_dtype=False, check_exact=True
    )
    pd.testing.assert_frame_equal(
        actual[numeric], expected[numeric], check_dtype=False,
        check_exact=False, rtol=0.0, atol=1e-10,
    )
    return {
        "path": str(reference_path.resolve()),
        "sha256": _sha256(reference_path),
        "row_count": int(len(reference)),
        "identity_and_numeric_match": True,
        "numeric_absolute_tolerance": 1e-10,
    }


def _draw_lognormal(
    branch_uniform: np.ndarray,
    standard_normal: np.ndarray,
    *,
    non_positive_mass: float,
    log_mean: float,
    log_sigma: float,
) -> np.ndarray:
    positive = np.exp(log_mean + log_sigma * standard_normal)
    return np.where(branch_uniform < non_positive_mass, 0.0, positive)


def _comparison_rows(
    summaries: dict[str, dict[str, dict[str, float]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain in ("residual", "residence", "onset_interval"):
        metrics = list(summaries["human"][domain])
        for metric in metrics:
            human = summaries["human"][domain][metric]
            row: dict[str, object] = {"domain": domain, "metric": metric, "human": human}
            for model in ("old", "mle", "empirical"):
                value = summaries[model][domain][metric]
                row[model] = value
                row[f"{model}_absolute_error"] = abs(value - human)
            rows.append(row)
    return rows


def _plot_cdf(path: Path, datasets: dict[str, np.ndarray], xlabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probabilities = np.linspace(0.0, 1.0, 2001)
    fig, axis = plt.subplots(figsize=(8, 5))
    for label, values in datasets.items():
        axis.plot(np.quantile(values, probabilities), probabilities, label=label)
    lower = min(float(np.quantile(values, 0.001)) for values in datasets.values())
    upper = max(float(np.quantile(values, 0.995)) for values in datasets.values())
    axis.set(
        xlabel=xlabel, ylabel="Empirical CDF", ylim=(0.0, 1.0), xlim=(lower, upper)
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_qq(path: Path, human: np.ndarray, models: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probabilities = np.linspace(0.01, 0.99, 99)
    human_q = np.quantile(human, probabilities)
    fig, axis = plt.subplots(figsize=(7, 6))
    for label, values in models.items():
        axis.plot(human_q, np.quantile(values, probabilities), label=label)
    lower = min(float(human_q.min()), *(float(np.quantile(v, 0.01)) for v in models.values()))
    upper = max(float(human_q.max()), *(float(np.quantile(v, 0.99)) for v in models.values()))
    axis.plot((lower, upper), (lower, upper), linestyle="--", color="black", label="identity")
    axis.set(
        xlabel="Human onset-interval quantile (ms)",
        ylabel="Sampler onset-interval quantile (ms)",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(
    *,
    data_root: Path,
    reference_pairs: Path,
    output_dir: Path,
    monte_carlo_n: int,
    monte_carlo_seed: int,
) -> dict[str, Any]:
    if monte_carlo_n < 1_000_000:
        raise ValueError("Monte Carlo validation requires at least 1,000,000 samples")
    pairs, provenance = build_pairs(data_root)
    residual = pairs.residual_ms.to_numpy(dtype=np.float64)
    observed_onset = pairs.onset_gap_ms.to_numpy(dtype=np.float64)
    angles = pairs.visual_angle_deg.to_numpy(dtype=np.float64)

    assert len(residual) == EXPECTED_PAIR_COUNT
    assert int(np.count_nonzero(residual <= 0.0)) == 1_861
    assert int(np.count_nonzero((residual > 0.0) & (residual <= FLOOR_MS))) == 2_722
    assert int(np.count_nonzero(residual <= FLOOR_MS)) == 4_583
    reference_check = _assert_reference_pairs(pairs, reference_pairs)

    positive = residual[residual > 0.0]
    mle_log_mean = float(np.log(positive).mean())
    mle_log_sigma = float(np.log(positive).std(ddof=0))
    scipy_shape, scipy_loc, scipy_scale = stats.lognorm.fit(positive, floc=0.0)
    if not math.isclose(mle_log_sigma, scipy_shape, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("direct and SciPy MLE sigma disagree")
    if scipy_loc != 0.0 or not math.isclose(
        mle_log_mean, math.log(scipy_scale), rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("direct and SciPy MLE location/scale disagree")

    non_positive_mass = 1_861 / EXPECTED_PAIR_COUNT
    rng = np.random.default_rng(monte_carlo_seed)
    branch_uniform = rng.random(monte_carlo_n)
    standard_normal = rng.standard_normal(monte_carlo_n)
    empirical_uniform = rng.random(monte_carlo_n)
    old_raw = _draw_lognormal(
        branch_uniform, standard_normal,
        non_positive_mass=non_positive_mass,
        log_mean=OLD_LOG_MEAN, log_sigma=OLD_LOG_SIGMA,
    )
    mle_raw = _draw_lognormal(
        branch_uniform, standard_normal,
        non_positive_mass=non_positive_mass,
        log_mean=mle_log_mean, log_sigma=mle_log_sigma,
    )
    quantile_probabilities = np.linspace(0.0, 1.0, EMPIRICAL_QUANTILE_COUNT)
    quantile_values = np.quantile(residual, quantile_probabilities)
    empirical_raw = np.interp(
        empirical_uniform, quantile_probabilities, quantile_values
    )
    raw_samples = {"old": old_raw, "mle": mle_raw, "empirical": empirical_raw}
    residence_samples = {
        name: np.clip(values, FLOOR_MS, CAP_MS)
        for name, values in raw_samples.items()
    }

    repeated_angles = np.resize(angles, monte_carlo_n)
    repeated_overhead = (
        PREP_MS + EXEC_INTERCEPT_MS
        + EXEC_SLOPE_MS_PER_DEGREE * repeated_angles
    )
    onset_samples = {
        name: repeated_overhead + residence
        for name, residence in residence_samples.items()
    }
    human_residence = np.clip(residual, FLOOR_MS, CAP_MS)
    summaries: dict[str, dict[str, dict[str, float]]] = {
        "human": {
            "residual": _residual_summary(residual, human_residence),
            "residence": _residence_summary(human_residence),
            "onset_interval": _summary(observed_onset, ONSET_QUANTILES),
        }
    }
    for name in raw_samples:
        summaries[name] = {
            "residual": _residual_summary(raw_samples[name], residence_samples[name]),
            "residence": _residence_summary(residence_samples[name]),
            "onset_interval": _summary(onset_samples[name], ONSET_QUANTILES),
        }

    diagnostics: dict[str, dict[str, dict[str, float]]] = {}
    human_domains = {
        "residual": residual,
        "residence": human_residence,
        "onset_interval": observed_onset,
    }
    model_domains = {
        name: {
            "residual": raw_samples[name],
            "residence": residence_samples[name],
            "onset_interval": onset_samples[name],
        }
        for name in raw_samples
    }
    for name in raw_samples:
        diagnostics[name] = {
            domain: _distance_diagnostics(human_values, model_domains[name][domain])
            for domain, human_values in human_domains.items()
        }

    old_floor_error = abs(
        summaries["old"]["residence"]["effective_floor_mass"]
        - summaries["human"]["residence"]["effective_floor_mass"]
    )
    mle_floor_error = abs(
        summaries["mle"]["residence"]["effective_floor_mass"]
        - summaries["human"]["residence"]["effective_floor_mass"]
    )
    diagnostic_names = (
        "ks_statistic", "wasserstein_ms", "mean_absolute_quantile_error_ms"
    )
    mle_improves_old_onset = all(
        diagnostics["mle"]["onset_interval"][metric]
        < diagnostics["old"]["onset_interval"][metric]
        for metric in diagnostic_names
    )
    empirical_comparisons = [
        (
            diagnostics["empirical"][domain][metric],
            diagnostics["mle"][domain][metric],
        )
        for domain in ("residual", "onset_interval")
        for metric in diagnostic_names
    ]
    empirical_beats_mle = (
        all(empirical <= mle for empirical, mle in empirical_comparisons)
        and sum(empirical < mle for empirical, mle in empirical_comparisons) >= 4
    )
    if mle_floor_error < old_floor_error and mle_improves_old_onset and not empirical_beats_mle:
        verdict = "ADOPT MLE TIMING CONTRACT"
    else:
        verdict = "TIMING CONTRACT NOT YET SAFE TO FREEZE"

    theoretical = {
        "old_effective_floor_mass": float(
            non_positive_mass + (1.0 - non_positive_mass)
            * stats.norm.cdf((math.log(FLOOR_MS) - OLD_LOG_MEAN) / OLD_LOG_SIGMA)
        ),
        "mle_effective_floor_mass": float(
            non_positive_mass + (1.0 - non_positive_mass)
            * stats.norm.cdf((math.log(FLOOR_MS) - mle_log_mean) / mle_log_sigma)
        ),
    }
    fit_summary = {
        "analysis": ANALYSIS_ID,
        "canonical_positive_n": int(len(positive)),
        "non_positive_residual_mass": non_positive_mass,
        "old": {"log_mean": OLD_LOG_MEAN, "log_sigma": OLD_LOG_SIGMA},
        "mle": {
            "method": "two-parameter log-normal MLE with loc fixed at zero",
            "log_mean": mle_log_mean,
            "log_sigma": mle_log_sigma,
            "median_ms": float(math.exp(mle_log_mean)),
            "uncapped_mean_ms": float(math.exp(mle_log_mean + mle_log_sigma ** 2 / 2.0)),
        },
        "scipy_cross_check": {
            "shape": float(scipy_shape), "loc": float(scipy_loc),
            "scale": float(scipy_scale), "matches_direct_mle": True,
        },
        "theoretical_floor_mass": theoretical,
    }
    selection = {
        "verdict": verdict,
        "runtime_changed": False,
        "old_floor_absolute_error": old_floor_error,
        "mle_floor_absolute_error": mle_floor_error,
        "mle_improves_old_on_all_onset_diagnostics": mle_improves_old_onset,
        "empirical_beats_mle_on_all_residual_and_onset_diagnostics": empirical_beats_mle,
        "criteria": (
            "MLE must reduce floor-mass error, improve onset KS/Wasserstein/quantile-MAE, "
            "and not be uniformly dominated by the offline empirical sampler."
        ),
    }
    validation_summary = {
        "analysis": ANALYSIS_ID,
        "monte_carlo": {"n": monte_carlo_n, "seed": monte_carlo_seed},
        "summaries": summaries,
        "distribution_diagnostics": diagnostics,
        "selection": selection,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    pd.DataFrame(_comparison_rows(summaries)).to_csv(
        output_dir / "human_vs_old_vs_mle_vs_empirical.csv", index=False
    )
    (output_dir / "fit_summary.json").write_text(
        json.dumps(fit_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(validation_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    script_path = Path(__file__).resolve()
    manifest = {
        "analysis": ANALYSIS_ID,
        "canonical_sample_n": EXPECTED_PAIR_COUNT,
        "canonical_assertions": {
            "d_le_0": 1_861, "zero_lt_d_le_60": 2_722, "d_le_60": 4_583,
        },
        "filter_definition": provenance["filter_order"],
        "input_sha256": provenance["input_files"],
        "code_and_parameter_dependencies": provenance["code_and_parameter_dependencies"],
        "reference_pair_artifact": reference_check,
        "residual_equation": provenance["residual_definition"],
        "non_positive_residual_mass": non_positive_mass,
        "mle_parameters": fit_summary["mle"],
        "residence_floor_ms": FLOOR_MS,
        "residence_cap_ms": CAP_MS,
        "monte_carlo_seed": monte_carlo_seed,
        "monte_carlo_n": monte_carlo_n,
        "empirical_sampler": {
            "quantile_count": EMPIRICAL_QUANTILE_COUNT,
            "probabilities": quantile_probabilities.tolist(),
            "residual_values_ms": quantile_values.tolist(),
            "interpolation": "linear",
            "runtime_candidate": False,
        },
        "script": {
            "path": str(script_path.relative_to(script_path.parents[2])),
            "sha256": _sha256(script_path),
            "version": ANALYSIS_ID,
        },
    }
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_cdf(
        plots_dir / "residual_cdf.png",
        {"Human": residual, "Old": old_raw, "MLE": mle_raw, "Empirical": empirical_raw},
        "Raw residual (ms)",
    )
    _plot_cdf(
        plots_dir / "onset_interval_cdf.png",
        {"Human": observed_onset, **{name.title(): value for name, value in onset_samples.items()}},
        "Fixation onset interval (ms)",
    )
    _plot_qq(
        plots_dir / "onset_interval_qq.png", observed_onset,
        {name.title(): value for name, value in onset_samples.items()},
    )
    lines = [
        "# Canonical human dwell sampler comparison", "",
        f"- Canonical pairs: {EXPECTED_PAIR_COUNT:,}",
        "- D <= 0: 1,861; 0 < D <= 60: 2,722; D <= 60: 4,583",
        f"- Positive residuals: {len(positive):,}",
        f"- MLE log mean/sigma: {mle_log_mean:.15g} / {mle_log_sigma:.15g}",
        f"- Human empirical floor mass: {summaries['human']['residence']['effective_floor_mass']:.9%}",
        f"- Old theoretical floor mass: {theoretical['old_effective_floor_mass']:.9%}",
        f"- MLE theoretical floor mass: {theoretical['mle_effective_floor_mass']:.9%}",
        f"- Monte Carlo: N={monte_carlo_n:,}, seed={monte_carlo_seed}",
        f"- Verdict: **{verdict}**", "",
        "The empirical 101-quantile sampler is an offline diagnostic only. No training was started.",
        "See `human_vs_old_vs_mle_vs_empirical.csv` and `validation_summary.json` for all metrics.", "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "fit_summary": fit_summary,
        "validation_summary": validation_summary,
        "manifest": manifest,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    value.add_argument(
        "--data-root", type=Path, default=project_root / "data" / "raw" / "experiment"
    )
    value.add_argument(
        "--reference-pairs", type=Path,
        default=project_root / "final_model" / "eval" / "results"
        / "emma_human_dwell_30668_provenance" / "calibration_pairs.csv.gz",
    )
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--monte-carlo-n", type=int, default=DEFAULT_MONTE_CARLO_N)
    value.add_argument("--monte-carlo-seed", type=int, default=DEFAULT_MONTE_CARLO_SEED)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(
        data_root=args.data_root,
        reference_pairs=args.reference_pairs,
        output_dir=args.output_dir,
        monte_carlo_n=args.monte_carlo_n,
        monte_carlo_seed=args.monte_carlo_seed,
    )
    fit = result["fit_summary"]
    validation = result["validation_summary"]
    print(json.dumps({
        "positive_n": fit["canonical_positive_n"],
        "mle": fit["mle"],
        "theoretical_floor_mass": fit["theoretical_floor_mass"],
        "selection": validation["selection"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
