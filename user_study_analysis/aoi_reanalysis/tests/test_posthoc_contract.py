import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


behavior = _load("behavior_reanalysis_posthoc", "02_behavior_reanalysis.py")
gaze = _load("gaze_reanalysis_posthoc", "01_gaze_reanalysis.py")


def test_behavior_posthoc_contains_marginal_and_simple_effect_families():
    rows = []
    for p in range(8):
        for t in [200, 400, 600]:
            for h in [0, 2, 4]:
                rows.append({"pid": f"P{p:02d}", "Time": t, "Horizon": h, "y": p + t / 1000 + h / 10})
    d = pd.DataFrame(rows)
    out = behavior._rm_posthoc_families(d, "y", "Y")
    assert {"marginal", "simple_effect"}.issubset(set(out["comparison_scope"]))
    assert (out["posthoc_family"] == "Y: Time within Horizon=0").any()
    assert (out["posthoc_family"] == "Y: Horizon within Time=200").any()
    assert out["p_holm"].notna().all()


def test_h1_h4_posthoc_contains_interaction_simple_effects():
    rng = np.random.default_rng(44)
    rows = []
    for p in range(8):
        up = rng.normal(scale=0.15)
        for t in [200, 400, 600]:
            for layout in ["A", "B", "C"]:
                tid = f"P{p:02d}_{t}_{layout}"
                ut = rng.normal(scale=0.08)
                for h in [1, 2, 3, 4]:
                    prop = 0.02 + 0.00002 * t + 0.003 * (5-h) + 0.000005 * t * (5-h)
                    prop = np.clip(prop + rng.normal(scale=0.0015), 0.001, 0.2)
                    rows.append({
                        "pid": f"P{p:02d}", "trial_id": tid, "Time": t, "Layout": layout,
                        "FutureStep": h, "dwell_prop_total": prop,
                        "dwell_prop_logit": np.log(prop / (1-prop)),
                    })
    d = pd.DataFrame(rows)
    stat, post = gaze.future_step_statistics(d, 0.05)
    assert stat["converged"].all()
    assert (post["posthoc_family"] == "H1-H4: Time within H1").any()
    assert (post["posthoc_family"] == "H1-H4: FutureStep within Time=200").any()
    assert {"marginal", "simple_effect"}.issubset(set(post["comparison_scope"]))
    assert post["p_holm"].notna().all()
