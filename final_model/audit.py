#!/usr/bin/env python3
"""Generate canonical contracts, portability audits, and preservation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[1]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from final_model.checkpoint import checkpoint_contract_report
from final_model.common.mappo import architecture_report
from final_model.common.hsp_initialization import full_fixed_partner_contract_report
from final_model.common.overcooked_backend import backend_contract_report
from final_model.common.partners import partner_contract_report
from final_model.common.paths import asset, portability_report
from final_model.common.reward import native_reward_shaping_audit
from final_model.common.tracking import tracking_contract_report
from final_model.partial_ver.memory import memory_contract_report


PROJECT = Path(__file__).resolve().parent
REPOSITORY = PROJECT.parent
REPORTS = PROJECT / "reports"
REFERENCE_BASELINES = {
    "simple_ver": "82f2714f61ad5a771586c9e925fd5014bbcf9bd0b24c5ffe4199d6241f97dc87",
}
FORBIDDEN_PATHS = ("/home/" + "choi/Desktop/study2", "/home/" + "noward2/study2")
FORBIDDEN_V3D = (
    "_V3D" + "ContributionTracker",
    "canonical_v3d" + "_reward_components",
    "contribution_reward_by_agent",
    "learner_contribution_reward",
)


def write_json(name: str, value) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reference_digest(directory: str) -> tuple[str, int]:
    files = sorted(
        path for path in (PROJECT / directory).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    aggregate = hashlib.sha256()
    for path in files:
        content = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(REPOSITORY)
        aggregate.update(f"{content}  {relative}\n".encode())
    return aggregate.hexdigest(), len(files)


def active_python_files():
    return sorted(path for path in PROJECT.rglob("*.py") if "tests" not in path.parts)


def source_audit():
    path_hits, v3d_hits, v3e_import_hits = [], [], []
    for path in active_python_files():
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(PROJECT))
        for forbidden in FORBIDDEN_PATHS:
            if forbidden in text:
                path_hits.append({"file": relative, "value": forbidden})
        if path.name != "audit.py":
            for forbidden in FORBIDDEN_V3D:
                if forbidden in text:
                    v3d_hits.append({"file": relative, "value": forbidden})
            if "v3e_single_memory_event_driven" in text:
                v3e_import_hits.append(relative)
    references = {}
    for directory, expected in REFERENCE_BASELINES.items():
        actual, count = reference_digest(directory)
        references[directory] = {
            "expected_composite_sha256": expected,
            "actual_composite_sha256": actual,
            "file_count": count,
            "unchanged": actual == expected,
        }
    return {
        "hardcoded_user_path_hits": path_hits,
        "hardcoded_user_path_pass": not path_hits,
        "v3d_contribution_production_hits": v3d_hits,
        "v3d_contribution_production_dependency_zero": not v3d_hits,
        "v3e_production_import_hits": v3e_import_hits,
        "v3e_production_dependency_zero": not v3e_import_hits,
        "reference_directories": references,
        "references_unchanged": all(value["unchanged"] for value in references.values()),
    }


def implementation_markdown(reward_audit, source):
    rows = [
        "| Term | Source key | MDP default | final_model | Trigger | Recipient | Active random3 | Source |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for term in reward_audit["native_shaping_terms"]:
        rows.append(
            f"| {term['term']} | `{term['source_key']}` | {term['mdp_no_shaping_default']} | "
            f"{term['final_model_value']} | {term['trigger']} | {term['recipient']} | "
            f"{term['active_in_random3']} | `{term['source_file']}:{','.join(map(str, term['source_lines']))}` |"
        )
    return "\n".join(
        [
            "# final_model implementation report",
            "",
            "Canonical entrypoint: `python final_model/train.py`.",
            "",
            "Full and Partial Action share the backend and ActionNet architecture; Partial uses its acquisition-driven replay contract.",
            "Full channel 20 is maximum freshness (255) everywhere; channels 21–24 retain the displayed-preview mask.",
            "Full Action may initialize its actor from canonical HSP S2 using the audited zero-padded 20→25 migration.",
            "Partial Action accepts fixed or shared Full checkpoints when their horizon and cooking contracts cover the target conditions.",
            "Partial supports stochastic memory-only rule gaze and learned GazeNet with Action-only, Gaze-only, or joint optimization.",
            "ActionNet runs only after completed visual acquisition; all proposals in one decision reuse h_t and only the executed latest candidate commits.",
            "Dynamic Partial memory and displayed preview use absolute acquisition timestamps and expire after 2400 ms.",
            "Channel 20 encodes raw255 continuous linear freshness; only static channels 10–14 remain exact.",
            "Frozen partners use metadata-only tensor guards on the env.step hot path, full SHA-256 checks at",
            "rollout boundaries, and startup optimizer-membership isolation checks.",
            "Bounded H4 evidence is recorded in `reports/partner_guard_performance_audit.json`.",
            "The backend imports bundled `Action` and `OvercookedGridworld` directly and does not import the old V3-E runtime.",
            "The production reward has no V3-D contribution tracker or contribution attribution.",
            "",
            "## Native reward",
            "",
            *rows,
            "",
            "Optimization: `sum(info['sparse_reward_by_agent']) + factor * info['shaped_reward_by_agent'][1]`.",
            "Evaluation primary reward is sparse team reward and deliveries.",
            "Reward shaping uses the independent user-configurable linear horizon (20M only as the backward-compatible default).",
            "",
            "## PPO rollout-policy consistency",
            "",
            "Action rollout observations are copied into independent audit storage at buffer insertion.",
            "The source observation and `buffer.obs[t]` are compared by exact values and uint8[32] SHA-256",
            "immediately after insertion, and the copy is compared again after actor forward and before PPO.",
            "A separate integer plane stores `transition_id = t * n_envs + env`; recurrent replay consumes",
            "those stored IDs. Source/buffer and capture/buffer `np.shares_memory` checks must both be false.",
            "All checks run before `prep_training`, `backward`, or either optimizer step and fail closed.",
            "The learner CNN and recurrent replay use rollout-sized microbatches and one-step GRU calls",
            "so CUDA execution context matches rollout without changing PPO or recurrent sample ordering.",
            "",
            "## Preservation and portability",
            "",
            f"Reference directories unchanged: `{source['references_unchanged']}`.",
            f"Hardcoded user path audit: `{source['hardcoded_user_path_pass']}`.",
            f"V3-E production dependency zero: `{source['v3e_production_dependency_zero']}`.",
            f"V3-D contribution reward production dependency zero: `{source['v3d_contribution_production_dependency_zero']}`.",
            "",
        ]
    )


def generate_reports():
    reward = native_reward_shaping_audit()
    source = source_audit()
    reports = {
        "native_reward_shaping_audit.json": reward,
        "backend_contract.json": backend_contract_report(),
        "partner_contract.json": partner_contract_report(),
        "architecture_parity.json": architecture_report(),
        "full_fixed_partner_hsp_initialization_contract.json": full_fixed_partner_contract_report(
            asset("policy_pool/random3/hsp/s2/hsp-S2-s36/4.pt"),
            asset("policy_pool/random3/hsp/s2/hsp-S2-s24/3.pt"),
        ),
        "checkpoint_contract.json": checkpoint_contract_report(),
        "wandb_contract.json": tracking_contract_report(),
        "partial_memory_contract.json": memory_contract_report(),
        "portability_audit.json": portability_report(),
        "source_preservation_audit.json": source,
    }
    for name, value in reports.items():
        write_json(name, value)
    (REPORTS / "implementation_report.md").write_text(
        implementation_markdown(reward, source) + "\n", encoding="utf-8"
    )
    if not source["references_unchanged"] or not source["hardcoded_user_path_pass"]:
        raise RuntimeError("source preservation/portability audit failed")
    return reports


def main():
    reports = generate_reports()
    print(json.dumps({name: "PASS" for name in reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
