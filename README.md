# How Time and Information Shape Visual Attention and Coordination in Real-Time Human–AI Teaming

This repository contains the experiment software, model-training code, evaluation tools, and analysis pipeline for a study of real-time human–AI collaboration in Overcooked. The study varies the time available for each decision (200, 400, or 600 ms) and the amount of future-path information shown to the participant (0, 2, or 4 steps) across three task layouts.

## Repository contents

- `main.py`, `tutorial.py`: user-study runner, tutorial, and 9-point eye-tracker calibration
- `experiment/`: deterministic, participant-level counterbalancing for the 27 experimental conditions
- `predict_path/`: HSP rollout generation and human-like path-prediction training
- `final_model/`: Full and Partial observation models, gaze models, EMMA-based constraints, training, and evaluation
- `user_study_analysis/`: behavioral, subjective, and gaze-analysis pipeline
- `policy_pool/`: HSP policy pool provided as a Git submodule
- `lib/overcooked_ai/`: the Overcooked runtime used by the experiment

## Installation

The project was developed with Python 3.9. Git LFS is required for the released model checkpoints and other large artifacts.

```bash
git lfs install
git clone --recurse-submodules \
  https://github.com/WY-C/How-Time-and-Information-Shape-Visual-Attention-and-Coordination-in-Real-Time-Human--AI-Teaming.git
cd How-Time-and-Information-Shape-Visual-Attention-and-Coordination-in-Real-Time-Human--AI-Teaming
git lfs pull
python -m pip install -r requirements.txt
```

If the repository was cloned without submodules, initialize the policy pool separately:

```bash
git submodule update --init --recursive
```

For GPU training, install the PyTorch build appropriate for the target CUDA environment.

## Running the user study

Start a participant session with the tutorial and calibration:

```bash
python main.py --tutorial --pid P01
```

Resume an interrupted session at the first unfinished trial:

```bash
python main.py --pid P01 --trial completed
```

Generate and validate a participant's counterbalanced 27-trial schedule without launching the game or eye tracker:

```bash
python main.py --pid P01 --dry-run --print-schedule
```

The main experiment uses the `counter_circuit`, `unident_s_hard_onion`, and `double_rings` layouts. An HSP agent serves as the experimental partner; a greedy agent is available only for tutorial practice. Tobii support is installed only on Windows through the platform-specific dependency in `requirements.txt`.

Use `python main.py --help` for all runtime and logging options.

## Path-prediction models

The path-prediction package generates HSP/MEP rollouts and trains layout-specific multi-step path forecasters:

```bash
python -m predict_path.hsp_clone.train_hsp_policy --help
python -m predict_path.hsp_clone.export_hsp_dataset --help
python -m predict_path.hsp_clone.rollout_hsp_clone --help
```

Released runtime checkpoints and their checksums are documented in [`predict_path/output/README.md`](predict_path/output/README.md).

## Final models

The canonical trainer supports Full Action, Partial Action, and Partial Gaze configurations, including rule-based or learned gaze and EMMA-based visual constraints:

```bash
python -m final_model.train --help
```

The selected checkpoints and SHA-256 checksums are listed in [`final_model/checkpoint/final/README.md`](final_model/checkpoint/final/README.md). Evaluation tools and their protocols are documented in [`final_model/eval/README.md`](final_model/eval/README.md).

## Analysis

The user-study analysis pipeline covers trial eligibility, gaze reanalysis, behavioral outcomes, subjective measures, validation, and figure generation. After the dataset is released, run the complete pipeline from the repository root with:

```bash
python user_study_analysis/aoi_reanalysis/run_all.py \
  --roots data/raw/experiment \
  --subjective data/raw/questionnaires/subjective.xlsx \
  --out data/analysis/aoi_reanalysis/results
```

See [`user_study_analysis/aoi_reanalysis/README.md`](user_study_analysis/aoi_reanalysis/README.md) for the analysis contract, output structure, and validation criteria.

## Dataset availability

> **The human-participant dataset will be added to this repository after the associated paper has been published.**

Until then, the analysis code is provided for inspection, but the full empirical analysis cannot be reproduced from this repository alone. Dataset documentation will be updated when the data are released.

## Tests

Run the default test suite from the repository root:

```bash
python -m pytest -q
```

Some integration tests under `final_model/server_parallel/tests/` require CUDA and legacy source checkpoints that are not part of the standard release, so they are not included in the default test target.

## Citation

Citation information will be added after publication of the associated paper.

### Pretrained Policy Pool

The pretrained policy pool is maintained separately
on Hugging Face as a Git submodule.

- Repository: [ZSC-Eval-policy_pool](https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool)
- Pinned commit: `a39b45a326c6fb9c4aee79550903a7de702c6974`

To clone the repository with all submodules:

```bash

