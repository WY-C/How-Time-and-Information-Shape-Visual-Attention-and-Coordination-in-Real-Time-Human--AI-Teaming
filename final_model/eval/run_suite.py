"""Run a JSON-declared set of separate/shared evaluations without hard-coded steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_model.eval.common import (
    FINAL_ANALYSIS_SEED_START,
    create_output_directory,
    default_output_path,
    source_hashes,
    write_json,
)
from final_model.eval.rollout import EvaluationConfig, evaluate
from final_model.eval.partial_rollout import PartialEvaluationConfig, evaluate_partial


def run_suite(specification: dict, *, output_dir: Path, device: str) -> Path:
    root = create_output_directory(output_dir, prefix="suite")
    episodes = int(specification.get("episodes", 10))
    episode_length = int(specification.get("episode_length", 400))
    seed_start = int(specification.get("seed_start", FINAL_ANALYSIS_SEED_START))
    runs = []
    for target_index, target in enumerate(specification.get("targets", [])):
        checkpoint = Path(target["checkpoint"])
        if target.get("observation_mode", "full") == "partial":
            label = target.get("label", f"target_{target_index}")
            for condition in target["conditions"]:
                result = evaluate_partial(PartialEvaluationConfig(
                    checkpoint=checkpoint,
                    decision_time=int(condition["decision_time"]),
                    horizon=int(condition["horizon"]),
                    fixation_shape=str(condition["fixation_shape"]),
                    gaze_type=str(condition["gaze_type"]),
                    episodes=episodes, seed_start=seed_start,
                    episode_length=episode_length, device=device,
                    gaze_deterministic=bool(condition.get(
                        "gaze_deterministic", condition.get("deterministic", True)
                    )),
                    action_deterministic=bool(condition.get(
                        "action_deterministic", condition.get("deterministic", True)
                    )),
                    freshness_diagnostic=bool(
                        condition.get("freshness_diagnostic", False)
                    ),
                    output_dir=root / (
                        f"{label}_h{condition['horizon']}_t{condition['decision_time']}_"
                        f"{condition['fixation_shape']}_{condition['gaze_type']}"
                    ),
                ))
                runs.append({
                    "label": label, **result["summary"],
                    "output_dir": str(result["output_dir"]),
                })
            continue
        for horizon in target["evaluation_horizons"]:
            label = target.get("label", f"target_{target_index}")
            result = evaluate(EvaluationConfig(
                checkpoint=checkpoint, evaluation_horizon=int(horizon),
                episodes=episodes, seed_start=seed_start,
                episode_length=episode_length, device=device,
                behavior_events=bool(specification.get("behavior_events", False)),
                output_dir=root / f"{label}_h{int(horizon)}",
            ))
            runs.append({"label": label, **result.summary, "output_dir": str(result.output_dir)})
    write_json(root / "suite_summary.json", runs)
    write_json(root / "manifest.json", {
        "analysis": "evaluation_suite",
        "specification": specification,
        "runs": runs,
        "source_hashes": source_hashes(),
    })
    return root


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--output-dir", type=Path)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = run_suite(spec, output_dir=args.output_dir or default_output_path("suite"), device=args.device)
    print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
