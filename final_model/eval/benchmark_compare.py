"""Tidy descriptive human--model comparison without a scalar score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compare_references(
    *, human_reference: Path, model_reference: Path, output: Path
) -> pd.DataFrame:
    human = pd.read_csv(human_reference)
    model = pd.read_csv(model_reference)
    matched = []
    for model_row in model.to_dict("records"):
        candidates = human[
            human.Time.eq(model_row["Time"])
            & human.metric.eq(model_row["metric"])
        ]
        horizon = model_row.get("Horizon")
        exact = candidates[
            candidates.Horizon.notna()
            & candidates.Horizon.eq(horizon)
        ]
        selected = exact if len(exact) else candidates[candidates.Horizon.isna()]
        if len(selected) > 1:
            raise RuntimeError(
                f"ambiguous human reference for {model_row['metric']} "
                f"T{model_row['Time']} H{horizon}"
            )
        human_row = {} if selected.empty else selected.iloc[0].to_dict()
        matched.append({
            **{f"{key}_model": value for key, value in model_row.items()},
            "Time": model_row["Time"], "metric": model_row["metric"],
            "aoi_contract_human": human_row.get("aoi_contract"),
            "value_human": human_row.get("value"),
            "numerator_human": human_row.get("numerator"),
            "denominator_human": human_row.get("denominator"),
        })
    rows = pd.DataFrame(matched)
    rows["aoi_contract_match"] = (
        rows.aoi_contract_model.fillna("") == rows.aoi_contract_human.fillna("")
    )
    rows["difference_model_minus_human"] = rows.value_model - rows.value_human
    rows["absolute_difference"] = rows.difference_model_minus_human.abs()
    rows["ratio_model_to_human"] = rows.value_model / rows.value_human.replace(0, np.nan)
    rows["comparison_status"] = np.where(
        rows.value_human.isna(), "human_reference_unavailable",
        np.where(rows.aoi_contract_match, "available", "aoi_contract_mismatch"),
    )
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(target, index=False)
    return rows
