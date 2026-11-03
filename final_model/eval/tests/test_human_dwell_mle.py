import math

import numpy as np
from scipy.stats import norm

from final_model.eval.fit_human_dwell_mle import _draw_lognormal


def test_candidate_mle_effective_floor_mass_matches_theory():
    count = 1_000_000
    seed = 20260905
    non_positive_mass = 1_861 / 30_668
    log_mean = 5.734037945196355
    log_sigma = 1.244271942784646
    rng = np.random.default_rng(seed)
    residual = _draw_lognormal(
        rng.random(count),
        rng.standard_normal(count),
        non_positive_mass=non_positive_mass,
        log_mean=log_mean,
        log_sigma=log_sigma,
    )
    residence = np.clip(residual, 60.0, 4000.0)
    expected = non_positive_mass + (1.0 - non_positive_mass) * norm.cdf(
        (math.log(60.0) - log_mean) / log_sigma
    )
    assert abs(np.mean(residence == 60.0) - expected) < 0.0015
    assert residence.min() == 60.0
    assert residence.max() <= 4000.0
