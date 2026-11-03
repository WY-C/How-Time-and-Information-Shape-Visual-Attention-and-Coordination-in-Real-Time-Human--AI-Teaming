"""Break down exclusive plus-footprint Other gaze by center-tile semantics."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from final_model.eval.analyze_opportunity_controlled_aoi import (
    load_model,
    new_fixations,
)
from final_model.eval.summarize_early_checkpoint import sha256
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
)


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "final_model/eval/results"
OUTPUT = (
    RESULTS
    / "benchmark_staged_pluslocal_10m_20m_v8_30seed_20260906"
    / "other_breakdown"
)
MODELS = (
    (
        "StagedPlusLocal10M_30seed",
        RESULTS / "humanfix_plus_local_gaze10m_v8_exponential_30ep_20260905",
    ),
    (
        "StagedPlusLocal20M_30seed",
        RESULTS / "benchmark_20m_staged_pluslocal_v8_30seed_20260906",
    ),
)
CATEGORIES = (
    "Pot",
    "Onion",
    "Dish",
    "Serving",
    "Counter",
    "Floor/Path",
    "Other object",
)
TERRAIN_CATEGORY = {
    "P": "Pot",
    "O": "Onion",
    "D": "Dish",
    "S": "Serving",
    "X": "Counter",
    " ": "Floor/Path",
}
TIMES = (200, 400, 600)
PAIRS = ((200, 400), (400, 600), (200, 600))
KEYS = ["T", "H", "seed", "episode", "decision"]


def _terrain_lookup(layout: str) -> dict[tuple[int, int], str]:
    mdp = OvercookedGridworld.from_layout_name(layout)
    return {
        (x, y): terrain
        for y, row in enumerate(mdp.terrain_mtx)
        for x, terrain in enumerate(row)
    }


def _state_objects(value: str) -> dict[tuple[int, int], str]:
    state = json.loads(value)
    return {
        tuple(int(v) for v in obj["position"]): str(obj.get("name", "unknown"))
        for obj in state.get("objects", [])
        if len(obj.get("position", [])) == 2
    }


def _annotate(
    frame: pd.DataFrame,
    transitions: pd.DataFrame,
    terrain_by_layout: dict[str, dict[tuple[int, int], str]],
) -> pd.DataFrame:
    merged = frame.merge(transitions, on=KEYS, how="left", validate="many_to_one")
    if merged.layout.isna().any() or merged.pre_state_json.isna().any():
        raise RuntimeError("gaze rows did not map one-to-one to pre-decision state")

    terrains = []
    objects = []
    categories = []
    for row in merged.itertuples(index=False):
        position = (int(row.target_x), int(row.target_y))
        terrain = terrain_by_layout[str(row.layout)].get(position, "OUTSIDE")
        obj = _state_objects(row.pre_state_json).get(position)
        terrains.append(terrain)
        objects.append(obj)
        categories.append(TERRAIN_CATEGORY.get(terrain, "Other object"))
    merged["center_terrain"] = terrains
    merged["center_object"] = objects
    merged["other_category"] = pd.Categorical(
        categories, categories=CATEGORIES, ordered=True
    )
    return merged


def _per_fixation(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, t), frame in events.groupby(["model", "T"], observed=True):
        other = frame[frame.other_hit.eq(1)]
        for category in CATEGORIES:
            count = int(other.other_category.eq(category).sum())
            rows.append(
                {
                    "model": model,
                    "T": int(t),
                    "category": category,
                    "all_new_fixations": len(frame),
                    "other_new_fixations": len(other),
                    "category_fixations": count,
                    "within_other_share": count / len(other),
                    "per_all_new_fixations": count / len(frame),
                }
            )
    return pd.DataFrame(rows)


def _first_new_fixation(events: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "decision_uid"]
    first = (
        events.sort_values(keys + ["fixation_start_ms", "segment_start_ms"])
        .drop_duplicates(keys)
    )
    rows = []
    for (model, t), frame in first.groupby(["model", "T"], observed=True):
        other = frame[frame.other_hit.eq(1)]
        for category in CATEGORIES:
            count = int(other.other_category.eq(category).sum())
            rows.append(
                {
                    "model": model,
                    "T": int(t),
                    "category": category,
                    "decisions_with_first_new_fixation": len(frame),
                    "first_new_fixation_other": len(other),
                    "category_first_fixations": count,
                    "within_first_other_share": count / len(other),
                    "per_first_new_fixation": count / len(frame),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_per_fixation(per_fixation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = per_fixation.set_index(["model", "category", "T"])
    for model in per_fixation.model.unique():
        for category in CATEGORIES:
            for low, high in PAIRS:
                lo = indexed.loc[(model, category, low)]
                hi = indexed.loc[(model, category, high)]
                for metric in ("within_other_share", "per_all_new_fixations"):
                    delta = float(hi[metric] - lo[metric])
                    rows.append(
                        {
                            "model": model,
                            "category": category,
                            "contrast": f"T{high}_vs_T{low}",
                            "metric": metric,
                            "low_probability": float(lo[metric]),
                            "high_probability": float(hi[metric]),
                            "delta_probability": delta,
                            "delta_pp": 100.0 * delta,
                            "slope_pp_per_100ms": 100.0 * delta / ((high - low) / 100.0),
                        }
                    )
    return pd.DataFrame(rows)


def _decision_outcomes(
    decisions: pd.DataFrame, segments: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    keys = ["model", "decision_uid"]
    counts = events.groupby(keys, observed=True).size().rename("n_new_fixations")
    other_segments = segments[segments.other_hit.eq(1)].copy()
    other_segments["category_hit"] = 1
    coverage = (
        other_segments.pivot_table(
            index=keys,
            columns="other_category",
            values="category_hit",
            aggfunc="max",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=CATEGORIES, fill_value=0)
        .rename(columns={category: f"hit_{index}" for index, category in enumerate(CATEGORIES)})
    )
    result = decisions.merge(counts, on=keys, how="left").merge(
        coverage, on=keys, how="left"
    )
    result["n_new_fixations"] = result.n_new_fixations.fillna(0).astype(int)
    hit_columns = [f"hit_{index}" for index in range(len(CATEGORIES))]
    result[hit_columns] = result[hit_columns].fillna(0).astype(np.int8)
    return result


def _fit_adjusted(decisions: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable

    rows = []
    for model, model_frame in decisions.groupby("model", observed=True):
        for category_index, category in enumerate(CATEGORIES):
            outcome = f"hit_{category_index}"
            for low, high in PAIRS:
                frame = model_frame[model_frame["T"].isin([low, high])].copy()
                formula = (
                    f"{outcome} ~ C(T, Treatment(reference={low}))"
                    " + C(H, Treatment(reference=0)) + n_new_fixations"
                )
                coefficient = f"C(T, Treatment(reference={low}))[T.{high}]"
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = smf.gee(
                            formula,
                            groups="cluster",
                            data=frame,
                            family=sm.families.Binomial(),
                            cov_struct=Exchangeable(),
                        ).fit()
                    estimate = float(result.params[coefficient])
                    se = float(result.bse[coefficient])
                    rows.append(
                        {
                            "model": model,
                            "category": category,
                            "contrast": f"T{high}_vs_T{low}",
                            "coefficient": estimate,
                            "standard_error": se,
                            "odds_ratio": float(np.exp(estimate)),
                            "ci_low": float(np.exp(estimate - 1.96 * se)),
                            "ci_high": float(np.exp(estimate + 1.96 * se)),
                            "p_value": float(result.pvalues[coefficient]),
                            "decisions": len(frame),
                            "clusters": frame.cluster.nunique(),
                            "status": "ok",
                        }
                    )
                except Exception as error:
                    rows.append(
                        {
                            "model": model,
                            "category": category,
                            "contrast": f"T{high}_vs_T{low}",
                            "decisions": len(frame),
                            "clusters": frame.cluster.nunique(),
                            "status": f"failed: {type(error).__name__}: {error}",
                        }
                    )
    return pd.DataFrame(rows)


def _counter_contents(events: pd.DataFrame) -> pd.DataFrame:
    counter = events[
        events.other_hit.eq(1) & events.other_category.eq("Counter")
    ].copy()
    counter["center_object"] = counter.center_object.fillna("empty")
    rows = (
        counter.groupby(["model", "T", "center_object"], observed=True)
        .size()
        .rename("fixations")
        .reset_index()
    )
    totals = counter.groupby(["model", "T"], observed=True).size().rename("total")
    rows = rows.merge(totals, on=["model", "T"], validate="many_to_one")
    rows["within_counter_share"] = rows.fixations / rows.total
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    decision_frames = []
    segment_frames = []
    event_frames = []

    for label, root in MODELS:
        decisions, segments = load_model(root, label=label, comparison="plus")
        transitions = pd.read_csv(
            root / "raw/task_transition_log.csv.gz",
            usecols=KEYS + ["layout", "pre_state_json"],
        )
        if transitions.duplicated(KEYS).any():
            raise RuntimeError(f"{label}: duplicate task transitions")
        terrain_by_layout = {
            layout: _terrain_lookup(layout) for layout in transitions.layout.unique()
        }
        annotated_segments = _annotate(segments, transitions, terrain_by_layout)
        annotated_events = new_fixations(annotated_segments)

        other = annotated_events[annotated_events.other_hit.eq(1)]
        if other.other_category.isna().any() or len(other) != int(
            other.other_category.value_counts().sum()
        ):
            raise RuntimeError(f"{label}: Other categories do not partition events")
        if decisions.seed.nunique() != 30:
            raise RuntimeError(f"{label}: expected 30 seeds")

        decision_frames.append(decisions)
        segment_frames.append(annotated_segments)
        event_frames.append(annotated_events)

    decisions = pd.concat(decision_frames, ignore_index=True)
    segments = pd.concat(segment_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)

    per_fixation = _per_fixation(events)
    first = _first_new_fixation(events)
    decision_outcomes = _decision_outcomes(decisions, segments, events)
    outputs = {
        "other_category_per_fixation.csv": per_fixation,
        "other_category_pairwise_per_fixation.csv": _pairwise_per_fixation(per_fixation),
        "other_category_first_new_fixation.csv": first,
        "other_category_adjusted_models.csv": _fit_adjusted(decision_outcomes),
        "other_counter_contents.csv": _counter_contents(events),
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / name, index=False)

    manifest = {
        "purpose": "Decompose exclusive plus-footprint Other gaze by center-tile semantics",
        "models": [label for label, _ in MODELS],
        "seeds_per_model": 30,
        "other_definition": (
            "Existing exclusive plus_other_hit: no Self, Teammate, or Preview tile "
            "inside the plus acquisition footprint"
        ),
        "category_unit": "fixation center tile",
        "category_mapping": {
            "P": "Pot",
            "O": "Onion",
            "D": "Dish",
            "S": "Serving",
            "X": "Counter",
            "space": "Floor/Path",
            "remaining terrain (counter_circuit: T tomato dispenser)": "Other object",
        },
        "counter_note": "Counter remains terrain-based; loose contents are reported separately",
        "adjusted_model": (
            "decision category coverage ~ categorical T + categorical H + new fixation count; "
            "binomial GEE clustered by seed"
        ),
        "outputs": {name: sha256(OUTPUT / name) for name in outputs},
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
