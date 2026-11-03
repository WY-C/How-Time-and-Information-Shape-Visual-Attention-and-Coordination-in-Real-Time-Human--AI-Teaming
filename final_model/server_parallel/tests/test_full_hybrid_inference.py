from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from final_model.server_parallel.full_hybrid_action_training import (
    ServerParallelFullHybridActionTrainingSession,
    _partner_candidate,
)


class _Actor:
    def get_probs(self, observation, hidden, mask, available):
        assert observation.shape[0] == 1
        assert available.shape[0] == 1
        probabilities = torch.as_tensor(
            [[0.05, 0.10, 0.15, 0.20, 0.20, 0.30]], dtype=torch.float32
        )
        return probabilities, hidden + 1.0 + mask[:, :, None]


class _Pool:
    actor = _Actor()


def test_partner_candidate_is_transactional_batch1_and_env_keyed():
    generator = torch.Generator(device="cpu").manual_seed(91)
    rng_before = generator.get_state().clone()
    hidden = torch.zeros((1, 1, 64), dtype=torch.float32)
    mask = torch.zeros((1, 1), dtype=torch.float32)
    observation = np.zeros((8, 5, 20), dtype=np.float32)
    available = np.ones((6,), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _partner_candidate,
                env_id,
                _Pool(),
                observation,
                available,
                hidden.clone(),
                mask.clone(),
                rng_before.clone(),
            )
            for env_id in (1, 0)
        ]
    rows = sorted((future.result() for future in futures), key=lambda row: row["env_id"])
    assert [row["env_id"] for row in rows] == [0, 1]
    assert rows[0]["action"] == rows[1]["action"]
    assert torch.equal(rows[0]["next_hidden"], torch.ones_like(hidden))
    assert torch.equal(hidden, torch.zeros_like(hidden))
    assert torch.equal(generator.get_state(), rng_before)
    assert not torch.equal(rows[0]["next_rng_state"], rng_before)


def test_hybrid_fails_closed_on_non_cuda_or_resume():
    with pytest.raises(ValueError, match="requires CUDA"):
        ServerParallelFullHybridActionTrainingSession(device="cpu")
    with pytest.raises(ValueError, match="resume is not supported"):
        ServerParallelFullHybridActionTrainingSession(
            device="cuda", resume_checkpoint="not-used.pt"
        )
