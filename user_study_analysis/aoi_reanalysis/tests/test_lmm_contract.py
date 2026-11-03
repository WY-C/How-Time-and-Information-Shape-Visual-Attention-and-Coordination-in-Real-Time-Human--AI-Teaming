import numpy as np
import pandas as pd

from lmm_stats import fit_lmm, lmm_emm_contrast, lmm_term_table


def _balanced_overall(n_pid=12):
    rng = np.random.default_rng(123)
    rows = []
    for p in range(n_pid):
        up = rng.normal(scale=0.4)
        for t in [200, 400, 600]:
            for h in [0, 2, 4]:
                for layout in ["A", "B", "C"]:
                    y = up + 0.0015 * t + 0.08 * h + rng.normal(scale=0.2)
                    rows.append({
                        "pid": f"P{p:02d}", "trial_id": f"P{p:02d}_{t}_{h}_{layout}",
                        "Time": t, "Horizon": h, "Layout": layout, "y": y,
                    })
    return pd.DataFrame(rows)


def test_overall_lmm_has_participant_random_intercept_and_wald_terms():
    d = _balanced_overall()
    fit = fit_lmm(
        d,
        "y ~ C(Time, Sum) * C(Horizon, Sum) + C(Layout, Sum)",
        required_cols=["y", "Time", "Horizon", "Layout"],
    )
    tab = lmm_term_table(
        fit, analysis="test", aoi="Self",
        target_terms=["Time", "Horizon", "Time×Horizon"],
    )
    assert fit.result.converged
    assert set(["Time", "Horizon", "Time×Horizon"]).issubset(set(tab["term"]))
    assert tab["random_effects"].eq("(1|Participant)").all()


def test_response_style_lmm_uses_participant_and_trial_random_intercepts():
    rng = np.random.default_rng(7)
    rows = []
    for p in range(12):
        up = rng.normal(scale=0.4)
        k = 0
        for t in [200, 400, 600]:
            for h in [0, 2, 4]:
                for layout in ["A", "B", "C"]:
                    k += 1
                    ut = rng.normal(scale=0.15)
                    for phase in ["Pre", "Post"]:
                        y = up + ut + (0.25 if phase == "Post" else 0) + 0.0005*t + rng.normal(scale=0.1)
                        rows.append({
                            "pid": f"P{p:02d}", "trial_id": f"P{p:02d}_T{k}",
                            "Time": t, "Horizon": h, "Layout": layout, "Phase": phase, "y": y,
                        })
    d = pd.DataFrame(rows)
    fit = fit_lmm(
        d,
        "y ~ C(Phase, Sum) * C(Time, Sum) + C(Horizon, Sum) + C(Layout, Sum)",
        trial_vc=True,
        required_cols=["y", "Phase", "Time", "Horizon", "Layout", "trial_id"],
    )
    tab = lmm_term_table(fit, analysis="test", aoi="Information", target_terms=["Phase", "Time", "Phase×Time"])
    assert fit.result.converged
    assert tab["random_effects"].eq("(1|Participant) + (1|Trial)").all()
    c = lmm_emm_contrast(
        fit, factor="Phase", level1="Pre", level2="Post",
        fixed={"Time": 200}, marginal_over=["Horizon", "Layout"],
    )
    assert np.isfinite(c["estimate_level2_minus_level1_logit"])
    assert np.isfinite(c["p_uncorrected"])
