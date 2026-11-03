"""Command-line entry points for human reproduction and tiny model pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_model.eval.benchmark_rollout import (
    BenchmarkRolloutConfig,
    run_benchmark_rollout,
)
from final_model.eval.benchmark_contracts import BENCHMARK_CONTRACT_VERSIONS
from final_model.eval.human_benchmark import run_human_benchmark
from final_model.eval.benchmark_compare import compare_references


def _integers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    human = commands.add_parser("human", help="reproduce canonical human references")
    human.add_argument("--data-root", type=Path, required=True)
    human.add_argument("--output", type=Path, required=True)
    human.add_argument("--skip-task-gee", action="store_true")

    model = commands.add_parser("model", help="run a fixed-pair periodic threshold pilot")
    model.add_argument("--action-checkpoint", type=Path, required=True)
    model.add_argument("--gaze-checkpoint", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--times", type=_integers, default=(200, 400, 600))
    model.add_argument("--horizons", type=_integers, default=(0, 2, 4))
    model.add_argument("--episodes", type=int, default=1)
    model.add_argument("--episode-length", type=int, default=180)
    model.add_argument("--seed-start", type=int, default=810_000)
    model.add_argument(
        "--fixation-shape", choices=("plus", "2x2", "3x3"), default="plus"
    )
    model.add_argument("--tau", type=float, required=True)
    model.add_argument(
        "--action-mode",
        choices=(
            "periodic_20ms_threshold",
            "periodic_20ms_margin_threshold",
            "periodic_20ms_margin_motor_delay",
            "periodic_20ms_margin_motor_delay_stochastic_gaze",
            "periodic_20ms_margin_threshold_stochastic_gaze",
            "periodic_20ms_logit_commitment_stochastic_gaze",
            "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze",
            "periodic_20ms_no_commitment_stochastic_gaze",
        ),
        default="periodic_20ms_threshold",
    )
    model.add_argument("--motor-delay-ms", type=float, default=None)
    model.add_argument("--action-input-delay-ms", type=float, default=None)
    model.add_argument(
        "--contract-version", choices=BENCHMARK_CONTRACT_VERSIONS, default=None
    )
    model.add_argument("--device", default="cpu")
    model.add_argument("--model-name", default="partial_action_gaze_pair")
    model.add_argument(
        "--commitment-rule", choices=("fixed", "linear", "exponential", "none"),
        default="fixed",
    )
    model.add_argument("--tau-start", type=float, default=None)
    model.add_argument("--tau-min", type=float, default=0.0)
    model.add_argument("--tau-slope-per-ms", type=float, default=None)
    model.add_argument("--tau-kappa-ms", type=float, default=None)
    compare = commands.add_parser("compare", help="join tidy human/model references")
    compare.add_argument("--human-reference", type=Path, required=True)
    compare.add_argument("--model-reference", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "human":
        output = run_human_benchmark(
            data_root=args.data_root,
            output_dir=args.output,
            run_task_gee=not args.skip_task_gee,
        )
        summary = {
            "output_dir": str(output["output_dir"]),
            "regression_pass": output["regression_pass"],
            "process": output["process"],
            "task_events": output["task_events"],
        }
    elif args.command == "model":
        default_contracts = {
            "periodic_20ms_threshold": "contract_v2",
            "periodic_20ms_margin_threshold": "contract_v3",
            "periodic_20ms_margin_motor_delay": "contract_v4",
            "periodic_20ms_margin_motor_delay_stochastic_gaze": "contract_v5",
            "periodic_20ms_margin_threshold_stochastic_gaze": "contract_v6",
            "periodic_20ms_logit_commitment_stochastic_gaze": "contract_v7",
            "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze": "contract_v8",
            "periodic_20ms_no_commitment_stochastic_gaze": "contract_v9",
        }
        contract_version = args.contract_version or default_contracts[args.action_mode]
        if args.motor_delay_ms is not None and args.action_input_delay_ms is not None:
            result = parser()
            result.error(
                "use only one of --motor-delay-ms and --action-input-delay-ms"
            )
        motor_delay_ms = (
            args.action_input_delay_ms
            if args.action_input_delay_ms is not None else args.motor_delay_ms
        )
        if (
            args.action_mode
            == "periodic_20ms_logit_commitment_joint_delay_stochastic_gaze"
            and motor_delay_ms is None
        ):
            result = parser()
            result.error("joint-delay v8 requires --action-input-delay-ms")
        if motor_delay_ms is None:
            motor_delay_ms = (
                87.607603
                if args.action_mode in {
                    "periodic_20ms_margin_motor_delay",
                    "periodic_20ms_margin_motor_delay_stochastic_gaze",
                    "periodic_20ms_logit_commitment_stochastic_gaze",
                }
                else 0.0
            )
        output = run_benchmark_rollout(BenchmarkRolloutConfig(
            action_checkpoint=args.action_checkpoint,
            gaze_checkpoint=args.gaze_checkpoint,
            output_dir=args.output,
            decision_times=args.times,
            horizons=args.horizons,
            episodes=args.episodes,
            episode_length=args.episode_length,
            seed_start=args.seed_start,
            tau=args.tau,
            motor_delay_ms=motor_delay_ms,
            fixation_shape=args.fixation_shape,
            device=args.device,
            eval_action_mode=args.action_mode,
            benchmark_contract_version=contract_version,
            model_name=args.model_name,
            commitment_rule=args.commitment_rule,
            tau_start=args.tau_start,
            tau_min=args.tau_min,
            tau_slope_per_ms=args.tau_slope_per_ms,
            tau_kappa_ms=args.tau_kappa_ms,
        ))
        summary = {"output_dir": str(output["output_dir"]), **output["summary"]}
    else:
        compared = compare_references(
            human_reference=args.human_reference,
            model_reference=args.model_reference,
            output=args.output,
        )
        summary = {
            "output": str(args.output.resolve()),
            "rows": len(compared),
            "available": int(compared.comparison_status.eq("available").sum()),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
