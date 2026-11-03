import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


LAYOUT_FILES = {
    "counter_circuit": "counter_circuit.layout",
    "coordination_ring": "coordination_ring.layout",
    "double_rings": "double_rings.layout",
    "unident_s_hard": "unident_s_hard.layout",
    "unident_s_hard_onion": "unident_s_hard_onion.layout",
    "unident_s_onion": "unident_s_hard_onion.layout",
}
TILE_COLORS = {
    " ": "#D9C3A5",
    "X": "#9A6500",
    "P": "#565A60",
    "O": "#D9B44A",
    "D": "#E8E8E8",
    "S": "#333333",
    "1": "#0A9B78",
    "2": "#1976A3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw an IVT fixation scanpath over the recovered game surface."
    )
    parser.add_argument("--pid", required=True, help="Participant folder, for example P600.")
    trial_group = parser.add_mutually_exclusive_group()
    trial_group.add_argument("--trial-id", help="Exact trial_id to draw.")
    trial_group.add_argument(
        "--trial-index",
        type=int,
        default=1,
        help="Global trial index to draw when --trial-id is omitted.",
    )
    parser.add_argument(
        "--recovered-root",
        type=Path,
        default=Path("recovered"),
        help="Directory containing recovered participant gaze files.",
    )
    parser.add_argument(
        "--layout-root",
        type=Path,
        default=Path("lib/overcooked_ai/overcooked_ai_py/data/layouts"),
        help="Directory containing layout files.",
    )
    parser.add_argument(
        "--unident-version",
        choices=["legacy", "current"],
        default="legacy",
        help="Background layout version used for unident_s_hard.",
    )
    parser.add_argument("--start-step", type=int, help="First decision step to include.")
    parser.add_argument("--end-step", type=int, help="Last decision step to include.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. A participant/trial-specific name is used by default.",
    )
    return parser.parse_args()


def load_layout(layout_name, layout_root, unident_version):
    if layout_name == "unident_s_hard" and unident_version == "legacy":
        filename = "unident_s_hard_legacy_obs.layout"
    else:
        try:
            filename = LAYOUT_FILES[layout_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported scanpath layout: {layout_name}") from exc
    path = layout_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Layout file not found: {path}")
    config = ast.literal_eval(path.read_text(encoding="utf-8"))
    rows = [row.strip() for row in config["grid"].splitlines()]
    width = max(len(row) for row in rows)
    return [row.ljust(width) for row in rows]


def select_trial(manifest, trial_id, trial_index):
    if trial_id:
        selected = manifest[manifest["trial_id"].eq(trial_id)]
    else:
        selected = manifest[
            pd.to_numeric(
                manifest["global_trial_index"], errors="coerce"
            ).eq(trial_index)
        ]
    if len(selected) != 1:
        selector = trial_id if trial_id else f"global_trial_index={trial_index}"
        raise ValueError(f"Expected one trial for {selector}, found {len(selected)}")
    return selected.iloc[0]


def fixation_scanpath(raw_gaze, fixations, trial_id, start_step, end_step):
    raw = raw_gaze[raw_gaze["trial_id"].eq(trial_id)].copy()
    fixation_rows = fixations[
        fixations["trial_id"].eq(trial_id)
        & fixations["record_type"].eq("ivt_fixation")
    ].copy()
    if start_step is not None:
        fixation_rows = fixation_rows[fixation_rows["fixation_end_step"].ge(start_step)]
    if end_step is not None:
        fixation_rows = fixation_rows[fixation_rows["fixation_start_step"].le(end_step)]

    points = []
    for row in fixation_rows.sort_values("fixation_start_ms").itertuples():
        samples = raw[
            raw["timestamp_ms"].between(
                row.fixation_start_ms, row.fixation_end_ms, inclusive="both"
            )
        ]
        if start_step is not None:
            samples = samples[pd.to_numeric(samples["step"], errors="coerce").ge(start_step)]
        if end_step is not None:
            samples = samples[pd.to_numeric(samples["step"], errors="coerce").le(end_step)]
        x = pd.to_numeric(samples["recovered_surface_x"], errors="coerce")
        y = pd.to_numeric(samples["recovered_surface_y"], errors="coerce")
        valid = x.notna() & y.notna()
        if not valid.any():
            continue
        surface_width = pd.to_numeric(
            samples.loc[valid, "surface_width_px"], errors="coerce"
        ).median()
        surface_height = pd.to_numeric(
            samples.loc[valid, "surface_height_px"], errors="coerce"
        ).median()
        points.append(
            {
                "fixation_id": row.fixation_id,
                "start_ms": row.fixation_start_ms,
                "duration_ms": row.fixation_duration_ms,
                "surface_x": x[valid].median(),
                "surface_y": y[valid].median(),
                "surface_width": surface_width,
                "surface_height": surface_height,
                "base_aoi": row.base_aoi,
            }
        )
    return pd.DataFrame(points), len(fixation_rows)


def draw_layout(axis, rows):
    height = len(rows)
    width = len(rows[0])
    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            axis.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=TILE_COLORS.get(tile, "#D9C3A5"),
                    edgecolor="#F1E7D8",
                    linewidth=0.7,
                    zorder=0,
                )
            )
            if tile not in {" ", "X", "1", "2"}:
                axis.text(
                    x + 0.5,
                    y + 0.53,
                    tile,
                    ha="center",
                    va="center",
                    color="white" if tile in {"P", "S"} else "#333333",
                    fontsize=10,
                    fontweight="bold",
                    zorder=1,
                )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    return width, height


def draw_scanpath(points, rows, title, output_path):
    fig, axis = plt.subplots(figsize=(10, 7))
    width, height = draw_layout(axis, rows)
    if points.empty:
        raise ValueError("No recovered fixation points remain for the requested range.")

    x = points["surface_x"] / points["surface_width"] * width
    y = points["surface_y"] / points["surface_height"] * height
    inside = x.between(0, width) & y.between(0, height)
    plotted = points.loc[inside].copy()
    plotted["x"] = x[inside]
    plotted["y"] = y[inside]
    if plotted.empty:
        raise ValueError("All recovered fixation points are outside the game surface.")

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(plotted)))
    axis.plot(
        plotted["x"],
        plotted["y"],
        color="#252525",
        linewidth=1.5,
        alpha=0.65,
        zorder=2,
    )
    sizes = 45 + np.clip(plotted["duration_ms"].to_numpy(dtype=float), 0, 800) * 0.45
    axis.scatter(
        plotted["x"],
        plotted["y"],
        s=sizes,
        c=colors,
        edgecolors="white",
        linewidths=1.4,
        alpha=0.88,
        zorder=3,
    )
    for order, row in enumerate(plotted.itertuples(), start=1):
        axis.text(
            row.x,
            row.y,
            str(order),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    axis.set_title(title, fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Circle order = fixation order; circle size = fixation duration; color = elapsed order.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return len(plotted), int((~inside).sum())


def main():
    args = parse_args()
    participant_dir = args.recovered_root / args.pid
    manifest = pd.read_csv(participant_dir / "trial_recovery_manifest.csv")
    raw_gaze = pd.read_csv(participant_dir / "raw_gaze_recovered.csv")
    fixations = pd.read_csv(participant_dir / "fixation_recovered.csv")
    trial = select_trial(manifest, args.trial_id, args.trial_index)
    trial_id = trial["trial_id"]
    layout = trial["layout"]
    rows = load_layout(layout, args.layout_root, args.unident_version)
    points, selected_fixations = fixation_scanpath(
        raw_gaze,
        fixations,
        trial_id,
        args.start_step,
        args.end_step,
    )

    if args.output:
        output_path = args.output
    else:
        trial_index = int(trial["global_trial_index"])
        output_path = Path("analysis/scanpaths") / (
            f"{args.pid}_trial-{trial_index:02d}_{layout}.png"
        )
    step_range = ""
    if args.start_step is not None or args.end_step is not None:
        step_range = f" | steps {args.start_step or 0}-{args.end_step or 'end'}"
    title = (
        f"{args.pid} Trial {int(trial['global_trial_index'])}: {layout}{step_range}"
    )
    plotted, outside = draw_scanpath(points, rows, title, output_path)
    print(f"trial_id: {trial_id}")
    print(f"layout: {layout}")
    print(f"selected IVT fixations: {selected_fixations}")
    print(f"plotted fixations: {plotted}")
    print(f"outside/missing-surface fixations: {selected_fixations - plotted}")
    print(f"outside-surface centroids: {outside}")
    print(f"output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
