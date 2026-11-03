#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET_SAMPLES="${1:-1000000}"
DEVICE="${2:-cuda:0}"

export_dataset() {
    local layout="$1"
    shift
    python predict_path/hsp_clone/export_hsp_dataset.py \
        --layout "${layout}" \
        --hsp-slots 0 \
        --history-len 5 \
        --target-horizon 4 \
        --target-samples "${TARGET_SAMPLES}" \
        --horizon 400 \
        --device "${DEVICE}" \
        --output-path "data/model_training/predict_path/tutorial_beta_${layout}_${TARGET_SAMPLES}" \
        "$@"
}

train_model() {
    local layout="$1"
    python predict_path/hsp_clone/train_hsp_policy.py compare \
        --dataset-path "data/model_training/predict_path/tutorial_beta_${layout}_${TARGET_SAMPLES}.npz" \
        --target-horizon 4 \
        --hidden-dim-candidates 256,512,1024 \
        --device "${DEVICE}" \
        --selected-model-path "predict_path/output/tutorial_beta_${layout}.pt"
}

export_dataset cramped_room \
    --target-greedy \
    --partner-agents GREEDY
train_model cramped_room

export_dataset bothway_coordination \
    --target-checkpoint policy_pool/random0_medium/hsp/s2/hsp-S2-s36/3.pt
train_model bothway_coordination

export_dataset coordination_ring \
    --target-checkpoint policy_pool/random1/hsp/s2/hsp-S2-s36/2.pt
train_model coordination_ring
