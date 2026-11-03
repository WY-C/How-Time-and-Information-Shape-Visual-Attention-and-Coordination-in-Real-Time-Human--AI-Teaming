"""Canonical reproducible GazeNet behavioral evaluation and comparison suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from final_model.eval.checkpoint_loader import load_partial_checkpoint_for_evaluation
from final_model.eval.gaze_behavior_metrics import analyze_saved_model, read_csv
from final_model.eval.gaze_behavior_report import write_cross_model, write_model_summary
from final_model.eval.gaze_behavior_rollout import RolloutConfig, run_model_rollouts
from final_model.eval.gaze_behavior_progress import ProgressReporter
from final_model.eval.gaze_behavior_artifact import (
    INCOMPLETE_MARKER,
    ROLLOUT_DATA_FILES,
    ROLLOUT_MANIFEST,
    sha256_file,
    validate_rollout_artifact,
)
from final_model.eval.partial_rollout import PartialEvaluationConfig, _validate


DEFAULT_AOI_PRECEDENCE = ("self", "partner", "preview")
REQUIRED_ROLLOUT_FILES = (ROLLOUT_MANIFEST, *ROLLOUT_DATA_FILES)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path


@dataclass(frozen=True)
class EvaluationSpec:
    decision_times: tuple[int, ...]
    horizons: tuple[int, ...]
    fixation_shape: str
    episodes: int
    episode_length: int
    seed_start: int
    device: str
    gaze_mode: str
    action_mode: str
    bootstrap_iterations: int
    bootstrap_seed: int
    aoi_precedence: tuple[str, ...]


def _model_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(value)):
        raise ValueError("model name must be a filesystem-safe identifier")
    return str(value)


def _positive(value: Any, label: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def validate_manifest(payload: Mapping[str, Any], *, source: Path | None = None) -> tuple[list[ModelSpec], EvaluationSpec]:
    if not isinstance(payload, Mapping):
        raise ValueError("manifest root must be a JSON object")
    raw_models = payload.get("models")
    evaluation = payload.get("evaluation")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("manifest models must be a non-empty list")
    if not isinstance(evaluation, Mapping):
        raise ValueError("manifest evaluation must be an object")
    base = Path.cwd() if source is None else source.parent
    models = []
    seen = set()
    for raw in raw_models:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str) or not isinstance(raw.get("checkpoint"), str):
            raise ValueError("every model requires string name and checkpoint fields")
        name = _model_name(raw["name"])
        if name in seen:
            raise ValueError(f"duplicate model name: {name}")
        seen.add(name)
        checkpoint = Path(raw["checkpoint"]).expanduser()
        if not checkpoint.is_absolute():
            repository_relative = Path.cwd() / checkpoint
            checkpoint = repository_relative if repository_relative.exists() else base / checkpoint
        models.append(ModelSpec(name, checkpoint.resolve()))
    decision_times = tuple(map(int, evaluation.get("decision_times", (200, 400, 600))))
    horizons = tuple(map(int, evaluation.get("horizons", (0, 2, 4))))
    if not decision_times or any(value not in (200, 400, 600) for value in decision_times):
        raise ValueError("decision_times must contain only 200, 400, 600")
    if not horizons or any(value not in (0, 2, 4) for value in horizons):
        raise ValueError("horizons must contain only 0, 2, 4")
    precedence = tuple(evaluation.get("aoi_precedence", DEFAULT_AOI_PRECEDENCE))
    if sorted(precedence) != sorted(DEFAULT_AOI_PRECEDENCE):
        raise ValueError("aoi_precedence must order self, partner, preview exactly once")
    gaze_mode = str(evaluation.get("gaze_mode", "stochastic"))
    action_mode = str(evaluation.get("action_mode", "stochastic"))
    if gaze_mode not in {"stochastic", "argmax"} or action_mode not in {"stochastic", "argmax"}:
        raise ValueError("gaze_mode/action_mode must be stochastic or argmax")
    spec = EvaluationSpec(
        decision_times=decision_times, horizons=horizons,
        fixation_shape=str(evaluation.get("fixation_shape", "plus")),
        episodes=_positive(evaluation.get("episodes", 30), "episodes"),
        episode_length=_positive(evaluation.get("episode_length", 400), "episode_length"),
        seed_start=int(evaluation.get("seed_start", 20260826)),
        device=str(evaluation.get("device", "cpu")),
        gaze_mode=gaze_mode, action_mode=action_mode,
        bootstrap_iterations=_positive(evaluation.get("bootstrap_iterations", 20_000), "bootstrap_iterations"),
        bootstrap_seed=int(evaluation.get("bootstrap_seed", 20260826)),
        aoi_precedence=precedence,
    )
    if spec.fixation_shape not in {"plus", "2x2", "3x3"}:
        raise ValueError("fixation_shape must be plus, 2x2, or 3x3")
    if spec.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    return models, spec


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    source = value.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--rollout-artifact", type=Path)
    value.add_argument("--model-name")
    value.add_argument("--decision-times", type=int, nargs="+", default=(200, 400, 600))
    value.add_argument("--horizons", type=int, nargs="+", default=(0, 2, 4))
    value.add_argument("--fixation-shape", choices=("plus", "2x2", "3x3"), default="plus")
    value.add_argument("--episodes", type=int, default=30)
    value.add_argument("--episode-length", type=int, default=400)
    value.add_argument("--seed-start", type=int, default=20260826)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--gaze-mode", choices=("stochastic", "argmax"), default="stochastic")
    value.add_argument("--action-mode", choices=("stochastic", "argmax"), default="stochastic")
    value.add_argument("--bootstrap", type=int, default=20_000)
    value.add_argument("--bootstrap-seed", type=int, default=20260826)
    value.add_argument("--aoi-precedence", nargs=3, default=DEFAULT_AOI_PRECEDENCE)
    value.add_argument("--output-dir", type=Path, required=True)
    modes = value.add_mutually_exclusive_group()
    modes.add_argument("--rollout-only", action="store_true")
    modes.add_argument("--analysis-only", action="store_true")
    value.add_argument("--reuse-rollouts", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    progress = value.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="progress", action="store_true")
    progress.add_argument("--no-progress", dest="progress", action="store_false")
    value.set_defaults(progress=True)
    value.add_argument("--progress-every-episodes", type=int, default=1)
    value.add_argument("--bootstrap-progress-every", type=int, default=2000)
    return value


def _from_args(args) -> tuple[list[ModelSpec], EvaluationSpec]:
    if args.rollout_artifact is not None:
        if not args.analysis_only:
            raise ValueError("--rollout-artifact requires --analysis-only")
        artifact = args.rollout_artifact.expanduser().resolve()
        manifest = validate_rollout_artifact(artifact, allow_legacy=True)
        payload = {
            "models": [{
                "name": manifest["model_name"],
                "checkpoint": manifest["checkpoint_absolute_path"],
            }],
            "evaluation": {
                "decision_times": manifest["decision_times"],
                "horizons": manifest["horizons"],
                "fixation_shape": manifest["fixation_shape"],
                "episodes": manifest["episodes"],
                "episode_length": manifest["episode_length"],
                "seed_start": manifest["seed_start"],
                "device": args.device,
                "gaze_mode": manifest["gaze_mode"],
                "action_mode": manifest["action_mode"],
                "bootstrap_iterations": args.bootstrap,
                "bootstrap_seed": args.bootstrap_seed,
                "aoi_precedence": manifest["aoi_precedence"],
            },
        }
        return validate_manifest(payload)
    if args.manifest is not None:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        return validate_manifest(payload, source=args.manifest.resolve())
    if not args.model_name:
        raise ValueError("--model-name is required with --checkpoint")
    payload = {
        "models": [{"name": args.model_name, "checkpoint": str(args.checkpoint)}],
        "evaluation": {
            "decision_times": args.decision_times, "horizons": args.horizons,
            "fixation_shape": args.fixation_shape, "episodes": args.episodes,
            "episode_length": args.episode_length, "seed_start": args.seed_start,
            "device": args.device, "gaze_mode": args.gaze_mode,
            "action_mode": args.action_mode,
            "bootstrap_iterations": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "aoi_precedence": args.aoi_precedence,
        },
    }
    return validate_manifest(payload)


def _rollout_complete(model_dir: Path) -> bool:
    return (
        not (model_dir / INCOMPLETE_MARKER).exists()
        and all((model_dir / name).is_file() for name in REQUIRED_ROLLOUT_FILES)
    )


def _stage_rollout_artifact(source: Path, target: Path) -> None:
    source = source.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if source == target.resolve():
        return
    for name in REQUIRED_ROLLOUT_FILES:
        original = source / name
        destination = target / name
        if destination.exists():
            if destination.samefile(original) or (
                destination.stat().st_size == original.stat().st_size
                and sha256_file(destination) == sha256_file(original)
            ):
                continue
            raise RuntimeError(
                f"analysis output contains a different rollout file: {destination}"
            )
        try:
            os.link(original, destination)
        except OSError:
            shutil.copy2(original, destination)


def _validate_saved_contracts(manifest, evaluation, loaded) -> None:
    expected = {
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "decision_times": list(evaluation.decision_times),
        "horizons": list(evaluation.horizons),
        "fixation_shape": evaluation.fixation_shape,
        "episodes": evaluation.episodes,
        "episode_length": evaluation.episode_length,
        "seed_start": evaluation.seed_start,
        "gaze_mode": evaluation.gaze_mode,
        "action_mode": evaluation.action_mode,
        "aoi_precedence": list(evaluation.aoi_precedence),
        **loaded.constraint_config.metadata(),
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"saved rollout contract mismatch: {mismatch}")


def _release_rollout_heap() -> None:
    """Return coarse Phase-2 temporaries before projected Phase-3 parsing."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    models, evaluation = _from_args(args)
    root = args.output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not (
        args.overwrite or args.reuse_rollouts or args.analysis_only
    ):
        raise RuntimeError(
            "output directory is non-empty; use --reuse-rollouts, --analysis-only, or --overwrite"
        )
    root.mkdir(parents=True, exist_ok=True)
    condition_count = len(evaluation.decision_times) * len(evaluation.horizons)
    total_rollout_episodes = (
        len(models) * condition_count * evaluation.episodes
    )
    reporter = ProgressReporter(
        enabled=args.progress,
        total_rollout_episodes=total_rollout_episodes,
        progress_every_episodes=args.progress_every_episodes,
        bootstrap_progress_every=args.bootstrap_progress_every,
    )
    root_manifest = {
        "suite": "canonical_gaze_behavior_suite_v1",
        "models": [{"name": model.name, "checkpoint": str(model.checkpoint)} for model in models],
        "evaluation": asdict(evaluation),
        "analysis_only": bool(args.analysis_only),
        "rollout_only": bool(args.rollout_only),
        "progress": bool(args.progress),
        "progress_every_episodes": args.progress_every_episodes,
        "bootstrap_progress_every": args.bootstrap_progress_every,
    }
    (root / "manifest.json").write_text(
        json.dumps(root_manifest, indent=2, sort_keys=True, default=list) + "\n",
        encoding="utf-8",
    )
    reporter.banner(
        model_count=len(models), condition_count=condition_count,
        evaluation=evaluation, output=root,
    )
    try:
        reporter.phase(1, "PREFLIGHT")
        loaded_models = []
        for model_index, model in enumerate(models, 1):
            reporter.context["model"] = model.name
            model_dir = root / "models" / model.name
            if args.analysis_only:
                artifact_source = (
                    args.rollout_artifact.expanduser().resolve()
                    if args.rollout_artifact is not None else model_dir
                )
                validate_rollout_artifact(
                    artifact_source, allow_legacy=True, verify_hashes=True
                )
                _stage_rollout_artifact(artifact_source, model_dir)
            loaded = load_partial_checkpoint_for_evaluation(
                model.checkpoint, device=evaluation.device
            )
            for decision_time in evaluation.decision_times:
                for horizon in evaluation.horizons:
                    _validate(
                        PartialEvaluationConfig(
                            checkpoint=model.checkpoint,
                            decision_time=decision_time, horizon=horizon,
                            fixation_shape=evaluation.fixation_shape,
                            gaze_type="learned", episodes=evaluation.episodes,
                            episode_length=evaluation.episode_length,
                            seed_start=evaluation.seed_start,
                            device=evaluation.device,
                            gaze_deterministic=(evaluation.gaze_mode == "argmax"),
                            action_deterministic=(evaluation.action_mode == "argmax"),
                        ),
                        loaded,
                    )
            reporter.preflight_model(
                model_index, len(models), model.name, loaded
            )
            loaded_models.append((model, loaded, model_dir))

        reporter.phase(2, "ROLLOUT")
        if args.analysis_only:
            reporter.skip_rollout("analysis-only mode")
        manifests = []
        for model_index, (model, loaded, model_dir) in enumerate(
            loaded_models, 1
        ):
            if args.analysis_only:
                episode_count = len(
                    read_csv(model_dir / "episodes.csv")
                )
                reporter.account_existing(episode_count)
            elif _rollout_complete(model_dir) and args.reuse_rollouts:
                saved_manifest = validate_rollout_artifact(
                    model_dir, allow_legacy=True, verify_hashes=True
                )
                reporter.reuse(
                    model.name, model_dir,
                    episodes=(
                        int(saved_manifest["episodes"])
                        * len(saved_manifest["decision_times"])
                        * len(saved_manifest["horizons"])
                    ),
                    acquisitions=int(saved_manifest["acquisition_count"]),
                )
            else:
                run_model_rollouts(
                    loaded=loaded,
                    config=RolloutConfig(
                        model_name=model.name,
                        decision_times=evaluation.decision_times,
                        horizons=evaluation.horizons,
                        fixation_shape=evaluation.fixation_shape,
                        episodes=evaluation.episodes,
                        episode_length=evaluation.episode_length,
                        seed_start=evaluation.seed_start,
                        gaze_mode=evaluation.gaze_mode,
                        action_mode=evaluation.action_mode,
                        aoi_precedence=evaluation.aoi_precedence,
                    ),
                    model_dir=model_dir, reporter=reporter,
                    model_index=model_index, model_count=len(models),
                )
                _release_rollout_heap()
            manifest = json.loads(
                (model_dir / "rollout_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            validate_rollout_artifact(
                model_dir, allow_legacy=True, verify_hashes=True
            )
            _validate_saved_contracts(manifest, evaluation, loaded)
            manifests.append((model, loaded, model_dir, manifest))

        completed = []
        reporter.phase(3, "OFFLINE ANALYSIS")
        if args.rollout_only:
            reporter.skip_analysis("rollout-only mode")
        else:
            for model, loaded, model_dir, manifest in manifests:
                analysis = analyze_saved_model(
                    model_dir,
                    bootstrap_iterations=evaluation.bootstrap_iterations,
                    bootstrap_seed=evaluation.bootstrap_seed,
                    aoi_precedence=evaluation.aoi_precedence,
                    gaze_actor=loaded.gaze_actor,
                    gaze_device=loaded.device,
                    progress=reporter,
                    progress_model_name=model.name,
                )
                write_model_summary(model_dir, manifest, analysis)
                completed.append((manifest, analysis))
                print(
                    f"{model.name}: checkpoint={manifest['checkpoint_sha256']} "
                    f"actor={manifest['actor_sha256']} "
                    f"gaze={manifest['gaze_sha256']} "
                    f"requests={manifest['request_count']} "
                    f"acquisitions={manifest['acquisition_count']}",
                    flush=True,
                )

        reporter.phase(4, "CROSS-MODEL REPORT")
        if args.rollout_only:
            reporter.emit("[PHASE 4/4][CROSS-MODEL][SKIP] rollout-only mode")
            summary = root / "manifest.json"
        else:
            write_cross_model(
                root, completed,
                bootstrap_iterations=evaluation.bootstrap_iterations,
                bootstrap_seed=evaluation.bootstrap_seed,
                progress=reporter,
            )
            summary = root / "cross_model" / "summary.md"
        reporter.finish(
            output=root, summary=summary,
            bootstrap=evaluation.bootstrap_iterations,
            models_completed=len(models), model_count=len(models),
        )
        return 0
    except Exception as error:
        reporter.failure(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
