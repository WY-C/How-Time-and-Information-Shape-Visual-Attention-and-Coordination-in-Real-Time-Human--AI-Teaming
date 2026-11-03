"""Summarize a gaze-behavior progression run with a matched collision pass."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


AOIS = ("self", "partner", "preview", "other")
FOOTPRINT_AOIS = ("self", "partner", "preview")


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(int(row[name]) for name in ("T", "H", "episode", "seed"))


def grouped(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def performance_row(model, rows, pre, *, t=None, h=None):
    decisions = sum(int(row["task_decisions"]) for row in rows)
    collisions = sum(int(row["collision_count"]) for row in rows)
    score = mean(float(row["sparse_team_reward"]) for row in rows)
    deliveries = mean(float(row["deliveries"]) for row in rows)
    rate = collisions / decisions
    baseline = pre[(t, h)] if h is not None else pre[(t,)] if t is not None else pre[()]
    return {
        "model": model,
        **({"T": t} if t is not None else {}),
        **({"H": h} if h is not None else {}),
        "episodes": len(rows),
        "decisions": decisions,
        "sparse_score_mean": score,
        "score_delta_vs_pre": score - baseline["score"],
        "deliveries_mean": deliveries,
        "collision_count": collisions,
        "collision_rate": rate,
        "collision_rate_pct": rate * 100.0,
        "collision_delta_vs_pre_pp": (rate - baseline["collision_rate"]) * 100.0,
    }


def exclusive_center_aoi(row, precedence):
    for label in precedence:
        if int(row.get(f"{label}_center", 0)):
            return label
    return "other"


def aoi_row(model, rows, pre, labels, *, t=None, h=None, footprint=False,
            precedence=AOIS[:-1]):
    total = len(rows)
    if footprint:
        shares = {
            label: sum(int(row[f"{label}_footprint"]) for row in rows) / total
            for label in labels
        }
    else:
        shares = {
            label: sum(
                exclusive_center_aoi(row, precedence) == label for row in rows
            ) / total
            for label in labels
        }
        if abs(sum(shares.values()) - 1.0) > 1e-12:
            raise RuntimeError(f"exclusive AOI shares do not sum to one: {model}")
    baseline = pre[(t, h)] if h is not None else pre[(t,)] if t is not None else pre[()]
    result = {
        "model": model,
        **({"T": t} if t is not None else {}),
        **({"H": h} if h is not None else {}),
        "events": total,
    }
    for label in labels:
        result[f"{label}_share"] = shares[label]
        result[f"{label}_pct"] = shares[label] * 100.0
        result[f"{label}_delta_vs_pre_pp"] = (
            shares[label] - baseline[label]
        ) * 100.0
    return result


def markdown_table(rows, columns, labels=None):
    labels = labels or columns
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-root", type=Path, required=True)
    parser.add_argument("--collision-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--aoi-precedence", nargs=3,
        choices=("self", "partner", "preview"),
        default=("self", "partner", "preview"),
        help="exclusive center AOI precedence; Other is implicit last",
    )
    args = parser.parse_args()
    precedence = tuple(args.aoi_precedence)
    labels = (*precedence, "other")
    behavior = args.behavior_root.resolve()
    collision = args.collision_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((behavior / "manifest.json").read_text())
    models = [item["name"] for item in manifest["models"]]
    if models[0] != "pre_joint_gaze16m":
        raise RuntimeError("first model must be the pre-joint baseline")

    episodes_by_model = {}
    acquisitions_by_model = {}
    for model in models:
        original = read_csv(behavior / "models" / model / "episodes.csv")
        measured = read_csv(collision / "models" / model / "episodes.csv")
        left = {key(row): row for row in original}
        right = {key(row): row for row in measured}
        if left.keys() != right.keys() or len(left) != 45:
            raise RuntimeError(f"episode pairing mismatch: {model}")
        for episode_key in left:
            for field in (
                "action_trajectory_sha256", "sparse_team_reward", "deliveries",
                "task_decisions",
            ):
                if left[episode_key][field] != right[episode_key][field]:
                    raise RuntimeError(
                        f"collision pass changed {field}: {model}/{episode_key}"
                    )
        episodes_by_model[model] = measured
        acquisitions_by_model[model] = read_csv(
            behavior / "models" / model / "acquisitions.csv.gz"
        )

    pre_performance = {}
    baseline_episodes = episodes_by_model[models[0]]
    pre_performance[()] = {
        "score": mean(float(row["sparse_team_reward"]) for row in baseline_episodes),
        "collision_rate": sum(int(row["collision_count"]) for row in baseline_episodes)
        / sum(int(row["task_decisions"]) for row in baseline_episodes),
    }
    for (t,), rows in grouped(baseline_episodes, ("T",)).items():
        pre_performance[(int(t),)] = {
            "score": mean(float(row["sparse_team_reward"]) for row in rows),
            "collision_rate": sum(int(row["collision_count"]) for row in rows)
            / sum(int(row["task_decisions"]) for row in rows),
        }
    for (t, h), rows in grouped(baseline_episodes, ("T", "H")).items():
        pre_performance[(int(t), int(h))] = {
            "score": mean(float(row["sparse_team_reward"]) for row in rows),
            "collision_rate": sum(int(row["collision_count"]) for row in rows)
            / sum(int(row["task_decisions"]) for row in rows),
        }

    performance_th, performance_t, performance_overall = [], [], []
    for model in models:
        episodes = episodes_by_model[model]
        performance_overall.append(
            performance_row(model, episodes, pre_performance)
        )
        for (t,), rows in sorted(grouped(episodes, ("T",)).items()):
            performance_t.append(
                performance_row(model, rows, pre_performance, t=int(t))
            )
        for (t, h), rows in sorted(grouped(episodes, ("T", "H")).items()):
            performance_th.append(
                performance_row(
                    model, rows, pre_performance, t=int(t), h=int(h)
                )
            )

    def baseline_aoi(rows, labels, footprint=False):
        result = {}
        for grouping_fields in ((), ("T",), ("T", "H")):
            groups = {(): rows} if not grouping_fields else grouped(rows, grouping_fields)
            for values, subset in groups.items():
                numeric = tuple(int(value) for value in values)
                result[numeric] = {
                    label: (
                        sum(int(row[f"{label}_footprint"]) for row in subset)
                        if footprint else
                        sum(
                            exclusive_center_aoi(row, precedence) == label
                            for row in subset
                        )
                    ) / len(subset)
                    for label in labels
                }
        return result

    baseline_acquisitions = acquisitions_by_model[models[0]]
    pre_exclusive = baseline_aoi(baseline_acquisitions, labels)
    pre_footprint = baseline_aoi(
        baseline_acquisitions, FOOTPRINT_AOIS, footprint=True
    )
    exclusive_th, exclusive_t, exclusive_overall = [], [], []
    footprint_th, footprint_t, footprint_overall = [], [], []
    for model in models:
        acquisitions = acquisitions_by_model[model]
        exclusive_overall.append(
            aoi_row(
                model, acquisitions, pre_exclusive, labels,
                precedence=precedence,
            )
        )
        footprint_overall.append(
            aoi_row(
                model, acquisitions, pre_footprint, FOOTPRINT_AOIS,
                footprint=True,
            )
        )
        for (t,), rows in sorted(grouped(acquisitions, ("T",)).items()):
            exclusive_t.append(
                aoi_row(
                    model, rows, pre_exclusive, labels, t=int(t),
                    precedence=precedence,
                )
            )
            footprint_t.append(
                aoi_row(
                    model, rows, pre_footprint, FOOTPRINT_AOIS,
                    t=int(t), footprint=True,
                )
            )
        for (t, h), rows in sorted(grouped(acquisitions, ("T", "H")).items()):
            exclusive_th.append(
                aoi_row(
                    model, rows, pre_exclusive, labels, t=int(t), h=int(h),
                    precedence=precedence,
                )
            )
            footprint_th.append(
                aoi_row(
                    model, rows, pre_footprint, FOOTPRINT_AOIS,
                    t=int(t), h=int(h), footprint=True,
                )
            )

    tables = {
        "performance_t_by_h.csv": performance_th,
        "performance_by_t.csv": performance_t,
        "performance_overall.csv": performance_overall,
        "aoi_exclusive_center_t_by_h.csv": exclusive_th,
        "aoi_exclusive_center_by_t.csv": exclusive_t,
        "aoi_exclusive_center_overall.csv": exclusive_overall,
        "aoi_footprint_nonexclusive_t_by_h.csv": footprint_th,
        "aoi_footprint_nonexclusive_by_t.csv": footprint_t,
        "aoi_footprint_nonexclusive_overall.csv": footprint_overall,
    }
    for name, rows in tables.items():
        write_csv(output / name, rows)

    report = [
        "# 3x3 peripheral / memory2400 joint progression\n",
        "All scores are sparse reward per 180-decision episode. Collision is the "
        "human-compatible joint-transition collision rate over learner decisions. "
        "Exclusive AOI is the ActionNet-query fixation center with precedence "
        f"{' > '.join(labels)}.\n",
        "## Overall\n",
        markdown_table(
            performance_overall,
            ("model", "sparse_score_mean", "score_delta_vs_pre", "collision_rate_pct", "collision_delta_vs_pre_pp"),
            ("model", "score", "Δscore", "collision %", "Δcollision pp"),
        ),
        "\n## By T\n",
        markdown_table(
            performance_t,
            ("model", "T", "sparse_score_mean", "score_delta_vs_pre", "collision_rate_pct", "collision_delta_vs_pre_pp"),
            ("model", "T", "score", "Δscore", "collision %", "Δcollision pp"),
        ),
        "\n## Exclusive center AOI overall (%)\n",
        markdown_table(
            exclusive_overall,
            ("model", *(f"{label}_pct" for label in labels)),
            ("model", *labels),
        ),
        "\n## T x H performance\n",
        markdown_table(
            performance_th,
            ("model", "T", "H", "sparse_score_mean", "score_delta_vs_pre", "collision_rate_pct", "collision_delta_vs_pre_pp"),
            ("model", "T", "H", "score", "Δscore", "collision %", "Δcollision pp"),
        ),
        "",
    ]
    (output / "summary.md").write_text("\n".join(report), encoding="utf-8")
    verification = {
        "models": models,
        "episode_pairs": len(models) * 45,
        "trajectory_hash_matches": len(models) * 45,
        "collision_contract_id": "human_compatible_joint_transition_collision_v1",
        "score_definition": "mean sparse_team_reward per 180-decision episode",
        "exclusive_aoi_definition": (
            "ActionNet-query fixation center; " + ">".join(labels)
        ),
    }
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
