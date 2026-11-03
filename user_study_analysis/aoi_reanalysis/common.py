from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

TIME_LEVELS = [200, 400, 600]
HORIZON_LEVELS = [0, 2, 4]
FUTURE_STEPS = [1, 2, 3, 4]
AOIS = ["Self", "Partner", "Information", "Other"]
MAIN_PIDS = {f"P{i:02d}" for i in range(1, 37)}
EXPECTED_TRIALS_PER_PARTICIPANT = 27
EXPECTED_DECISION_STEPS_PER_TRIAL = 180
EXPECTED_TRIALS_PER_CONDITION = 3
EXPECTED_MAIN_TRIALS = 36 * EXPECTED_TRIALS_PER_PARTICIPANT
FINE_BIN_MS = 1000.0 / 60.0
COARSE_BIN_MS = 100.0


def ensure_output_tree(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    paths = {
        "root": root,
        "raw": root / "00_raw_intermediate",
        "inputs": root / "01_analysis_inputs",
        "stats": root / "02_statistics",
        "desc": root / "03_descriptives",
        "fig": root / "04_figures",
        "qc": root / "05_qc",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths




def group_mean_ci95(data: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """Participant-level mean and two-sided t-based 95% CI half-width.

    The caller must pass one row per participant for each plotted cell.
    This function deliberately does not treat trials/bins as independent samples.
    """
    g = (
        data.groupby(group_cols, dropna=False)[value_col]
        .agg(count="count", mean="mean", std="std", median="median")
        .reset_index()
    )
    g["sem"] = g["std"] / np.sqrt(g["count"])
    crit = np.where(
        g["count"].to_numpy() > 1,
        student_t.ppf(0.975, g["count"].to_numpy() - 1),
        np.nan,
    )
    g["ci95"] = g["sem"] * crit
    return g

def discover_files(roots: Iterable[str], filename: str) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        p = Path(root).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Root does not exist: {p}")
        found.extend(sorted(p.rglob(filename)))
    out, seen = [], set()
    for p in found:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def parse_segments(value) -> list[dict]:
    if not isinstance(value, str) or not value:
        return []
    try:
        obj = json.loads(value)
    except Exception:
        return []
    return obj if isinstance(obj, list) else []


def numeric_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().any():
        return x.fillna(0).ne(0)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def _read_trial_rows(roots: list[str]) -> pd.DataFrame:
    files = discover_files(roots, "trial.csv")
    if not files:
        raise FileNotFoundError("No trial.csv files found")
    parts = []
    required = {
        "trial_id", "pid", "completed", "time_constraint_ms",
        "information_steps", "layout_code", "decision_steps_completed",
    }
    for f in files:
        x = pd.read_csv(f)
        missing = sorted(required - set(x.columns))
        if missing:
            raise ValueError(f"{f}: missing trial columns {missing}")
        x["_source_file"] = str(f)
        parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    d["pid"] = d["pid"].astype(str).str.upper()
    return d


def build_trial_eligibility_audit(roots: list[str]) -> pd.DataFrame:
    """Return one row per main-study trial log with explicit inclusion/exclusion reasons.

    Canonical eligibility:
    - P01-P36 only
    - completed flag is true
    - Time in {200,400,600}
    - Horizon in {0,2,4}
    - decision_steps_completed == 180 exactly
    """
    d = _read_trial_rows(roots)
    d = d[d["pid"].isin(MAIN_PIDS)].copy()
    d["Time"] = pd.to_numeric(d["time_constraint_ms"], errors="coerce")
    d["Horizon"] = pd.to_numeric(d["information_steps"], errors="coerce")
    d["Layout"] = d["layout_code"].astype(str)
    d["decision_steps_completed_numeric"] = pd.to_numeric(d["decision_steps_completed"], errors="coerce")
    d["completed_ok"] = numeric_bool(d["completed"])
    d["time_ok"] = d["Time"].isin(TIME_LEVELS)
    d["horizon_ok"] = d["Horizon"].isin(HORIZON_LEVELS)
    d["decision_steps_ok"] = d["decision_steps_completed_numeric"].eq(EXPECTED_DECISION_STEPS_PER_TRIAL)

    reasons = []
    for r in d.itertuples(index=False):
        why = []
        if not bool(r.completed_ok):
            why.append("completed_flag_false")
        if not bool(r.time_ok):
            why.append("invalid_time")
        if not bool(r.horizon_ok):
            why.append("invalid_horizon")
        if not bool(r.decision_steps_ok):
            why.append(f"decision_steps_completed_not_{EXPECTED_DECISION_STEPS_PER_TRIAL}")
        reasons.append(";".join(why))
    d["exclusion_reason"] = reasons
    d["eligible"] = d["exclusion_reason"].eq("")
    return d.reset_index(drop=True)


def validate_trial_balance(eligible: pd.DataFrame, strict_main: bool = True) -> pd.DataFrame:
    """Fail closed unless every main participant has the complete 27-trial design.

    Besides 27 total valid trials, require exactly 3 trials in every Time x Horizon
    cell and 3 distinct layouts per cell. This catches condition duplication even
    when a participant still has 27 rows in total.
    """
    if eligible.duplicated("trial_id").any():
        dup = eligible.loc[eligible.duplicated("trial_id", keep=False), ["pid", "trial_id"]]
        raise RuntimeError(f"Duplicate eligible trial_id values found:\n{dup.head(30)}")

    summary_rows = []
    for pid in sorted(MAIN_PIDS):
        g = eligible[eligible["pid"].eq(pid)]
        row = {"pid": pid, "n_valid_trials": int(g["trial_id"].nunique())}
        row["exactly_27"] = row["n_valid_trials"] == EXPECTED_TRIALS_PER_PARTICIPANT
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    if strict_main:
        bad = summary[~summary["exactly_27"]]
        if len(bad):
            raise RuntimeError(
                "After excluding incomplete trials, every P01-P36 participant must have exactly "
                f"{EXPECTED_TRIALS_PER_PARTICIPANT} valid trials. Bad counts:\n{bad.to_string(index=False)}"
            )
        if len(eligible) != EXPECTED_MAIN_TRIALS:
            raise RuntimeError(f"Expected {EXPECTED_MAIN_TRIALS} eligible trials, found {len(eligible)}")

        cell = (
            eligible.groupby(["pid", "Time", "Horizon"], as_index=False)
            .agg(n_trials=("trial_id", "nunique"), n_layouts=("Layout", "nunique"))
        )
        full_index = pd.MultiIndex.from_product(
            [sorted(MAIN_PIDS), TIME_LEVELS, HORIZON_LEVELS],
            names=["pid", "Time", "Horizon"],
        )
        cell = cell.set_index(["pid", "Time", "Horizon"]).reindex(full_index, fill_value=0).reset_index()
        bad_cell = cell[(cell["n_trials"] != EXPECTED_TRIALS_PER_CONDITION) | (cell["n_layouts"] != EXPECTED_TRIALS_PER_CONDITION)]
        if len(bad_cell):
            raise RuntimeError(
                "Invalid within-participant condition balance: each Time x Horizon cell must contain "
                f"exactly {EXPECTED_TRIALS_PER_CONDITION} trials from {EXPECTED_TRIALS_PER_CONDITION} distinct layouts.\n"
                + bad_cell.head(50).to_string(index=False)
            )
    return summary


def load_trials(roots: list[str], strict_main: bool = True) -> pd.DataFrame:
    all_rows = _read_trial_rows(roots)
    audit = build_trial_eligibility_audit(roots)
    valid_ids = set(audit.loc[audit["eligible"], "trial_id"])
    d = all_rows[all_rows["trial_id"].isin(valid_ids) & all_rows["pid"].isin(MAIN_PIDS)].copy()
    d["Time"] = pd.to_numeric(d["time_constraint_ms"], errors="coerce").astype(int)
    d["Horizon"] = pd.to_numeric(d["information_steps"], errors="coerce").astype(int)
    d["Layout"] = d["layout_code"].astype(str)
    d["decision_steps_completed"] = pd.to_numeric(d["decision_steps_completed"], errors="raise").astype(int)
    validate_trial_balance(d, strict_main=strict_main)
    return d.reset_index(drop=True)


def load_steps(roots: list[str], trial: pd.DataFrame) -> pd.DataFrame:
    files = discover_files(roots, "step.csv")
    if not files:
        raise FileNotFoundError("No step.csv files found")
    valid = set(trial["trial_id"])
    parts = []
    wanted = [
        "pid", "trial_id", "step", "state_onset_ms", "response_time_ms",
        "is_human_idle", "human_action_due",
    ]
    for f in files:
        head = pd.read_csv(f, nrows=0)
        missing = [c for c in ["trial_id", "step", "state_onset_ms"] if c not in head.columns]
        if missing:
            continue
        use = [c for c in wanted if c in head.columns]
        x = pd.read_csv(f, usecols=use)
        x = x[x["trial_id"].isin(valid)].copy()
        if len(x):
            parts.append(x)
    if not parts:
        raise RuntimeError("No usable step rows found")
    d = pd.concat(parts, ignore_index=True)
    if "human_action_due" not in d:
        d["human_action_due"] = 1
    if "is_human_idle" not in d:
        d["is_human_idle"] = d.get("response_time_ms", pd.Series(index=d.index, dtype=float)).isna().astype(int)
    if "response_time_ms" not in d:
        d["response_time_ms"] = np.nan
    d["step"] = pd.to_numeric(d["step"], errors="coerce")
    d["state_onset_ms"] = pd.to_numeric(d["state_onset_ms"], errors="coerce")
    d["response_time_ms"] = pd.to_numeric(d["response_time_ms"], errors="coerce")
    d["human_action_due"] = numeric_bool(d["human_action_due"]).astype(int)
    d["is_human_idle"] = numeric_bool(d["is_human_idle"]).astype(int)
    d = d[d["step"].notna() & d["state_onset_ms"].notna()].copy()
    d["step"] = d["step"].astype(int)

    # Fail closed on conflicting duplicates, otherwise keep one copy.
    key = ["trial_id", "step"]
    rows = []
    for _, g in d.groupby(key, sort=False):
        for c in ["state_onset_ms", "human_action_due"]:
            vals = pd.Series(g[c]).dropna().unique()
            if len(vals) > 1:
                raise RuntimeError(f"Conflicting duplicate step rows for {g.iloc[0]['trial_id']} step={g.iloc[0]['step']}, column={c}")
        rt = pd.Series(g["response_time_ms"]).dropna().unique()
        if len(rt) > 1:
            raise RuntimeError(f"Conflicting response_time_ms for {g.iloc[0]['trial_id']} step={g.iloc[0]['step']}")
        r = g.iloc[0].copy()
        if len(rt) == 1:
            r["response_time_ms"] = rt[0]
        r["is_human_idle"] = int(pd.to_numeric(g["is_human_idle"], errors="coerce").fillna(0).max() > 0)
        rows.append(r)
    return pd.DataFrame(rows).sort_values(key).reset_index(drop=True)


def segment_has_information(seg: dict) -> bool:
    try:
        flagged = int(seg.get("overlay_flag") or 0) == 1
    except Exception:
        flagged = False
    if not flagged:
        return False
    overlay_type = seg.get("overlay_type")
    return overlay_type in (None, "", "path")


def matched_future_steps(seg: dict) -> list[int]:
    if not segment_has_information(seg):
        return []
    matches = seg.get("overlay_matches") or []
    hs = sorted({
        int(m.get("overlay_horizon"))
        for m in matches
        if m.get("overlay_type") in (None, "", "path")
        and m.get("overlay_horizon") in FUTURE_STEPS
    })
    if hs:
        return hs
    h = seg.get("overlay_horizon")
    return [int(h)] if h in FUTURE_STEPS else []


def allocate_top_level(seg: dict, duration_ms: float) -> dict[str, float]:
    """Allocate one segment across Self/Partner/Information/Other so sum == duration.

    Contract:
    - Self or Partner is a geometric AOI when base_aoi says so.
    - Information is a geometric AOI when the displayed preview is matched.
    - Other is residual, not a second geometric layer. It receives 100% only when
      neither Self, Partner, nor Information matches.
    - If Information overlaps Self or Partner, split equally among matched AOIs.
    """
    duration_ms = float(duration_ms)
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        return {a: 0.0 for a in AOIS}
    matched: list[str] = []
    base = seg.get("base_aoi")
    if base in ("Self", "Partner"):
        matched.append(base)
    if segment_has_information(seg):
        matched.append("Information")
    if not matched:
        matched = ["Other"]
    weight = duration_ms / len(matched)
    out = {a: 0.0 for a in AOIS}
    for a in matched:
        out[a] += weight
    if abs(sum(out.values()) - duration_ms) > 1e-9:
        raise RuntimeError("AOI allocation does not conserve dwell")
    return out


def allocate_future_steps(seg: dict, allocated_information_ms: float) -> dict[int, float]:
    out = {h: 0.0 for h in FUTURE_STEPS}
    hs = matched_future_steps(seg)
    if allocated_information_ms <= 0 or not hs:
        return out
    w = float(allocated_information_ms) / len(hs)
    for h in hs:
        out[h] += w
    return out


def adjusted_logit(values, n_reference: int = EXPECTED_MAIN_TRIALS) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    N = max(int(n_reference), 2)
    q = (np.clip(p[ok], 0.0, 1.0) * (N - 1) + 0.5) / N
    out[ok] = np.log(q / (1.0 - q))
    return out


def overlap_ms(start: float, end: float, left: float, right: float) -> float:
    return max(0.0, min(end, right) - max(start, left))


def audit_composition(df: pd.DataFrame, prefix: str, allow_empty: bool = False) -> None:
    x = df.copy()
    if allow_empty and "total_fixation_ms" in x:
        x = x[x["total_fixation_ms"] > 0]
    if x.empty:
        return
    prop_cols = [f"{a.lower()}_prop" for a in AOIS]
    ms_cols = [f"{a.lower()}_ms" for a in AOIS]
    ps = x[prop_cols].sum(axis=1)
    if np.nanmax(np.abs(ps - 1.0)) > 1e-9:
        raise RuntimeError(f"{prefix}: AOI proportions do not sum to 1")
    if "total_fixation_ms" in x:
        ms = x[ms_cols].sum(axis=1)
        if np.nanmax(np.abs(ms - x["total_fixation_ms"])) > 1e-6:
            raise RuntimeError(f"{prefix}: AOI dwell does not sum to total fixation dwell")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
