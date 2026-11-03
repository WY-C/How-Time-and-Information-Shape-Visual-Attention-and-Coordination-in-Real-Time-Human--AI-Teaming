"""Read-only same-state Partial freshness counterfactual."""

from __future__ import annotations

import numpy as np

from final_model.eval.counterfactual import actor_probabilities, distribution_metrics


SEMANTIC_FRESHNESS_TEST_VALUES = (1.0, 0.875, 0.75, 0.5, 0.1, 0.0)


def freshness_counterfactual(
    *, actor, observation, hidden, mask, available, native_probabilities,
) -> list[dict[str, object]]:
    """Change channel 20 only; never commit a returned recurrent state."""
    native_observation = np.asarray(observation, dtype=np.float32)
    hidden_before = np.asarray(hidden).copy()
    rows = []
    for semantic_freshness in SEMANTIC_FRESHNESS_TEST_VALUES:
        counterfactual = native_observation.copy()
        raw_freshness = np.float32(255.0 * semantic_freshness)
        counterfactual[..., 20] = raw_freshness
        probabilities = actor_probabilities(
            actor, counterfactual, hidden_before, mask, available
        )
        rows.append(
            {
                "counterfactual_semantic_freshness": float(semantic_freshness),
                "counterfactual_raw_freshness": float(raw_freshness),
                **distribution_metrics(native_probabilities, probabilities),
            }
        )
    if not np.array_equal(hidden_before, hidden):
        raise RuntimeError("freshness counterfactual mutated native recurrent state")
    return rows
