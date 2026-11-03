"""Frozen HSP-S2-s24 seed-3 training-partner pool."""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from final_model.common.paths import asset, file_sha256
from final_model.common.actor_state import load_actor_state_dict
from final_model.common.zsceval_compat import install_zsceval, load_policy_config


MEP_IDS = (5, 6, 7, 8, 9, 10)
HSP_IDS = (5, 27, 60, 131, 138, 168)
PARTNER_SET_ID = "hsp_s2_s24_seed3_training_population_v1"
FIXED_HSP_PARTNER_ID = "hsp_s2_s24_seed3"
FIXED_HSP_PARTNER_SHA256 = (
    "0f13132b029f8bf2bc1da6c4ddd76d661182ed6e26a12c9ce1e29e9f1de8f02c"
)
FIXED_HSP_PARTNER_CONTRACT_ID = "fixed_hsp_s2_s24_seed3_recurrent_partner_v1"


def parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorGuard:
    name: str
    object_id: int
    data_ptr: int
    version: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


def tensor_guard_snapshot(module: torch.nn.Module) -> tuple[TensorGuard, ...]:
    """Capture metadata-only mutation guards without copying tensor values."""
    tensors = (
        [(f"parameter:{name}", value) for name, value in module.named_parameters()]
        + [(f"buffer:{name}", value) for name, value in module.named_buffers()]
    )
    return tuple(
        TensorGuard(
            name=name,
            object_id=id(value),
            data_ptr=value.data_ptr(),
            version=value._version,
            shape=tuple(value.shape),
            dtype=value.dtype,
            device=value.device,
        )
        for name, value in tensors
    )


def assert_optimizer_isolation(
    partner: torch.nn.Module, optimizers: Mapping[str, torch.optim.Optimizer]
) -> None:
    partner_ids = {id(parameter) for parameter in partner.parameters()}
    for name, optimizer in optimizers.items():
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if partner_ids & optimizer_ids:
            raise RuntimeError(f"frozen partner parameter found in {name}")


def partner_manifest() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    mep = asset("policy_pool/random3/mep/s1/mep-S1-s10")
    for seed in MEP_IDS:
        for phase in ("init", "mid", "final"):
            path = (mep / f"mep{seed}_{phase}_actor.pt").resolve()
            records.append(
                {"name": f"mep{seed}_{phase}", "family": "mep", "path": path, "sha256": file_sha256(path)}
            )
    hsp = asset("model/counter_circuit/hsp/s1/hsp")
    for policy_id in HSP_IDS:
        path = (hsp / f"hsp{policy_id}_final_w0_actor.pt").resolve()
        records.append(
            {"name": f"hsp{policy_id}", "family": "hsp", "path": path, "sha256": file_sha256(path)}
        )
    if (
        len(records) != 24
        or sum(item["family"] == "mep" for item in records) != 18
        or sum(item["family"] == "hsp" for item in records) != 6
    ):
        raise RuntimeError("partner manifest must contain exactly 18 MEP and 6 HSP actors")
    return records


class FrozenPartnerRepository:
    """Lazily load each frozen actor once and share it across rollout environments."""

    def __init__(self, *, device: str | torch.device):
        install_zsceval()
        self.device = torch.device(device)
        self.manifest = partner_manifest()
        self._records = {str(item["name"]): item for item in self.manifest}
        self._actors: dict[str, torch.nn.Module] = {}
        self._hashes: dict[str, str] = {}
        self._guards: dict[str, tuple[TensorGuard, ...]] = {}
        self._actor_args = None

    def actor(self, name: str):
        if name not in self._records:
            raise ValueError(f"unknown partner {name!r}")
        if name not in self._actors:
            from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

            args, obs_space, _share_space, action_space = load_policy_config("mlp")
            actor = R_Actor(args, obs_space, action_space, device=self.device)
            actor.load_state_dict(
                load_actor_state_dict(Path(self._records[name]["path"])), strict=True
            )
            actor.eval().requires_grad_(False)
            self._actors[name] = actor
            self._hashes[name] = parameter_hash(actor)
            self._guards[name] = tensor_guard_snapshot(actor)
            self._actor_args = args
        return self._actors[name]

    def recurrent_shape(self) -> tuple[int, int]:
        if self._actor_args is None:
            self.actor(next(iter(self._records)))
        return int(self._actor_args.recurrent_N), int(self._actor_args.hidden_size)

    def assert_frozen(self, name: str, *, full_hash: bool = False) -> str:
        actor = self.actor(name)
        if tensor_guard_snapshot(actor) != self._guards[name]:
            raise RuntimeError(f"frozen partner tensor guard changed: {name}")
        if full_hash:
            actual = parameter_hash(actor)
            if actual != self._hashes[name]:
                raise RuntimeError(f"frozen partner parameter hash changed: {name}")
            return actual
        return self._hashes[name]


class FixedHSPPartnerRepository:
    """One strict canonical recurrent HSP S2-s24 seed-3 training partner."""

    def __init__(self, *, device: str | torch.device, checkpoint: Path):
        from final_model.common.hsp_initialization import build_canonical_hsp_actor

        self.device = torch.device(device)
        path = Path(checkpoint).expanduser().resolve()
        checkpoint_sha256 = file_sha256(path)
        if checkpoint_sha256 != FIXED_HSP_PARTNER_SHA256:
            raise RuntimeError(
                "fixed partner is not canonical HSP S2-s24 seed-3: "
                f"sha256={checkpoint_sha256}"
            )
        actor, architecture, _state = build_canonical_hsp_actor(
            path,
            device=self.device,
            expected_sha256=FIXED_HSP_PARTNER_SHA256,
            expected_seed=3,
            expected_population=24,
        )
        actor.eval().requires_grad_(False)
        self.manifest = [
            {
                "name": FIXED_HSP_PARTNER_ID,
                "family": "hsp_s2",
                "path": path,
                "sha256": checkpoint_sha256,
            }
        ]
        self._records = {FIXED_HSP_PARTNER_ID: self.manifest[0]}
        self._actors = {FIXED_HSP_PARTNER_ID: actor}
        self._hashes = {FIXED_HSP_PARTNER_ID: parameter_hash(actor)}
        self._guards = {
            FIXED_HSP_PARTNER_ID: tensor_guard_snapshot(actor)
        }
        self._architecture = architecture
        self.fixed_single_partner = True

    def actor(self, name: str):
        if name != FIXED_HSP_PARTNER_ID:
            raise ValueError(f"unknown fixed partner {name!r}")
        return self._actors[name]

    def recurrent_shape(self) -> tuple[int, int]:
        return 1, 64

    def assert_frozen(self, name: str, *, full_hash: bool = False) -> str:
        actor = self.actor(name)
        if tensor_guard_snapshot(actor) != self._guards[name]:
            raise RuntimeError(f"frozen partner tensor guard changed: {name}")
        if full_hash:
            actual = parameter_hash(actor)
            if actual != self._hashes[name]:
                raise RuntimeError(f"frozen partner parameter hash changed: {name}")
            return actual
        return self._hashes[name]

    def provenance(self) -> dict[str, object]:
        record = self.manifest[0]
        return {
            "fixed_partner_checkpoint": str(record["path"]),
            "fixed_partner_sha256": str(record["sha256"]),
            "fixed_partner_seed": 3,
            "fixed_partner_population": 24,
            "fixed_partner_identity": FIXED_HSP_PARTNER_ID,
            "fixed_partner_contract_id": FIXED_HSP_PARTNER_CONTRACT_ID,
            "fixed_partner_parameter_sha256": self._hashes[FIXED_HSP_PARTNER_ID],
            "fixed_partner_frozen": True,
            "fixed_partner_observation_channels": 20,
            "fixed_partner_recurrent_n": 1,
            "fixed_partner_hidden_size": 64,
            "fixed_partner_action_semantics": "categorical_sampling_with_original_legal_mask",
            "fixed_partner_pool_size": 1,
            "fixed_partner_sampling_disabled": True,
        }


class FrozenPartnerPool:
    def __init__(
        self,
        *,
        seed: int,
        device: str | torch.device,
        requested: str | None = None,
        repository: FrozenPartnerRepository | None = None,
    ):
        self.repository = repository or FrozenPartnerRepository(device=device)
        self.device = self.repository.device
        self.rng = np.random.default_rng(int(seed))
        self.torch_rng = torch.Generator(device=self.device.type).manual_seed(int(seed) + 91)
        self.manifest = self.repository.manifest
        self.requested = requested
        if requested is not None and requested not in {str(item["name"]) for item in self.manifest}:
            raise ValueError(f"unknown partner {requested!r}")
        self.counts: Counter[str] = Counter()
        self.actor = None
        self.active: dict[str, object] | None = None
        self.hidden = None
        self.mask = None
        self.load_next()

    def load_next(self) -> None:
        choices = self.manifest
        if self.requested is not None:
            choices = [item for item in choices if item["name"] == self.requested]
        if getattr(self.repository, "fixed_single_partner", False):
            if len(choices) != 1:
                raise RuntimeError("fixed partner repository must expose exactly one actor")
            selected = choices[0]
        else:
            selected = choices[int(self.rng.integers(len(choices)))]
        actor = self.repository.actor(str(selected["name"]))
        recurrent_n, hidden_size = self.repository.recurrent_shape()
        self.actor = actor
        self.active = selected
        self.hidden = torch.zeros((1, recurrent_n, hidden_size), device=self.device)
        self.mask = torch.zeros((1, 1), device=self.device)
        self.frozen_parameter_hash = self.repository.assert_frozen(str(selected["name"]))
        self.frozen_tensor_guards = self.repository._guards[str(selected["name"])]
        self.counts[str(selected["name"])] += 1

    @property
    def name(self) -> str:
        assert self.active is not None
        return str(self.active["name"])

    def action(self, observation20: np.ndarray, available_actions: np.ndarray) -> int:
        assert self.actor is not None and self.hidden is not None and self.mask is not None
        with torch.no_grad():
            probabilities, hidden = self.actor.get_probs(
                observation20[None], self.hidden, self.mask, available_actions[None]
            )
            action = torch.multinomial(probabilities[0], 1, generator=self.torch_rng)
        self.hidden = hidden.detach()
        self.mask = torch.ones_like(self.mask)
        return int(action.item())

    def assert_frozen(self, *, full_hash: bool = False) -> str:
        if hasattr(self, "repository"):
            return self.repository.assert_frozen(self.name, full_hash=full_hash)
        if tensor_guard_snapshot(self.actor) != self.frozen_tensor_guards:
            raise RuntimeError("frozen partner tensor guard changed")
        if full_hash:
            actual = parameter_hash(self.actor)
            if actual != self.frozen_parameter_hash:
                raise RuntimeError("frozen partner parameter hash changed")
            return actual
        return self.frozen_parameter_hash

    def training_state(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "active_name": self.name,
            "counts": dict(self.counts),
            "numpy_rng": copy.deepcopy(self.rng.bit_generator.state),
            "torch_rng": self.torch_rng.get_state().cpu(),
            "hidden": self.hidden.detach().cpu(),
            "mask": self.mask.detach().cpu(),
        }

    def load_training_state(self, state: Mapping[str, object]) -> None:
        if state.get("requested") != self.requested:
            raise RuntimeError(
                "resume partner constraint mismatch: "
                f"{state.get('requested')!r} != {self.requested!r}"
            )
        name = str(state.get("active_name"))
        records = {str(item["name"]): item for item in self.manifest}
        if name not in records or (self.requested is not None and name != self.requested):
            raise RuntimeError(f"invalid active partner in resume state: {name!r}")
        self.actor = self.repository.actor(name)
        self.active = records[name]
        self.hidden = torch.as_tensor(state["hidden"], device=self.device).clone()
        self.mask = torch.as_tensor(state["mask"], device=self.device).clone()
        self.counts = Counter({str(key): int(value) for key, value in dict(state["counts"]).items()})
        self.rng.bit_generator.state = copy.deepcopy(state["numpy_rng"])
        self.torch_rng.set_state(torch.as_tensor(state["torch_rng"]).cpu())
        self.frozen_parameter_hash = self.repository.assert_frozen(name)
        self.frozen_tensor_guards = self.repository._guards[name]


def partner_contract_report() -> dict[str, object]:
    manifest = partner_manifest()
    return {
        "partner_set_id": PARTNER_SET_ID,
        "count": len(manifest),
        "mep_count": sum(item["family"] == "mep" for item in manifest),
        "hsp_count": sum(item["family"] == "hsp" for item in manifest),
        "mep_ids": list(MEP_IDS),
        "hsp_ids": list(HSP_IDS),
        "source_training_config": "random3/hsp/s2/train-s24-hsp_mep-S1-s10-3.yml",
        "observation_channels": 20,
        "frozen": True,
        "members": [
            {key: str(value) if key == "path" else value for key, value in item.items()}
            for item in manifest
        ],
    }
