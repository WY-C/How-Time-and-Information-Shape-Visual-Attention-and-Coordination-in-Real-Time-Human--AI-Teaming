from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "final_model/checkpoint/action/legacy/action_full_action_network_120M.pt"
)
PARTNER = ROOT / "policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"
RULE = {"task": 0.60, "stay": 0.20, "stale": 0.15, "random": 0.05}


@pytest.fixture
def session_options():
    if not SOURCE.is_file():
        pytest.skip(
            "requires a legacy action source checkpoint not included in the final public set"
        )

    def build(**overrides):
        result = dict(
            gaze_type="rule_based",
            train_action=True,
            train_gaze=False,
            decision_times=[200],
            horizons=[0],
            fixation_shapes=["3x3"],
            seed=20260821,
            rollout_length=1,
            episode_length=1,
            n_rollout_threads=2,
            device="cpu",
            source_checkpoint=SOURCE,
            resume_checkpoint=None,
            fixed_partner_checkpoint=PARTNER,
            configured_total_steps=4,
            checkpoint_interval=2,
            reward_shaping_horizon_steps=20_000_000,
            rule_gaze_probabilities=RULE,
            partner=None,
            ppo_epochs=1,
            num_mini_batches=1,
        )
        result.update(overrides)
        return result
    return build
