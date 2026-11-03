"""Narrow compatibility layer for the bundled validated ZSC-Eval fork."""

from __future__ import annotations

import pickle
import sys
import types

import numpy as np

from final_model.common.paths import policy_config_directory, zsceval_source_root


def _namespace(name: str, path) -> None:
    existing = sys.modules.get(name)
    if existing is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
        return
    paths = [str(value) for value in getattr(existing, "__path__", ())]
    if paths and str(path) not in paths:
        raise RuntimeError(f"conflicting {name} package already loaded from {paths}")


def _install_optional_shims() -> None:
    try:
        __import__("loguru")
    except ModuleNotFoundError:
        class _NoOpLogger:
            def __getattr__(self, _name):
                return lambda *_args, **_kwargs: None

        module = types.ModuleType("loguru")
        module.logger = _NoOpLogger()
        sys.modules["loguru"] = module
    try:
        __import__("gym")
    except ModuleNotFoundError:
        class Box:
            __module__ = "gym.spaces.box"

            def __init__(self, low, high, dtype=np.float32, **_kwargs):
                self.low = np.asarray(low, dtype=dtype)
                self.high = np.asarray(high, dtype=dtype)
                self.dtype = np.dtype(dtype)
                self._shape = self.low.shape

            @property
            def shape(self):
                return self._shape

        class Discrete:
            __module__ = "gym.spaces.discrete"

            def __init__(self, n):
                self.n = int(n)
                self._shape = ()

            @property
            def shape(self):
                return self._shape

        gym = types.ModuleType("gym")
        spaces = types.ModuleType("gym.spaces")
        box_module = types.ModuleType("gym.spaces.box")
        discrete_module = types.ModuleType("gym.spaces.discrete")
        box_module.Box = Box
        discrete_module.Discrete = Discrete
        spaces.Box = Box
        spaces.Discrete = Discrete
        gym.spaces = spaces
        sys.modules.update(
            {
                "gym": gym,
                "gym.spaces": spaces,
                "gym.spaces.box": box_module,
                "gym.spaces.discrete": discrete_module,
            }
        )


def install_zsceval() -> None:
    root = zsceval_source_root()
    source = str(root)
    if source not in sys.path:
        sys.path.insert(0, source)
    _namespace("zsceval", root / "zsceval")
    _namespace("zsceval.utils", root / "zsceval/utils")
    _namespace("zsceval.envs", root / "zsceval/envs")
    _namespace("zsceval.envs.overcooked", root / "zsceval/envs/overcooked")
    _namespace(
        "zsceval.envs.overcooked.overcooked_ai_py",
        root / "zsceval/envs/overcooked/overcooked_ai_py",
    )
    _install_optional_shims()


def load_policy_config(architecture: str = "rnn"):
    install_zsceval()
    if architecture not in {"mlp", "rnn"}:
        raise ValueError("architecture must be mlp or rnn")
    with (policy_config_directory() / f"{architecture}_policy_config.pkl").open("rb") as handle:
        return pickle.load(handle)
