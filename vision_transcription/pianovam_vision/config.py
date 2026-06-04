"""Configuration loading.

A config is a nested dict loaded from YAML. We support dotted-key overrides
from the command line, e.g. ``train.batch_size=8`` or ``data.root=/x``.
Values are coerced from strings using YAML scalar parsing so that
``model.use_velocity=true`` becomes a real bool, ``labels.fps=25`` an int, etc.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce(value: str) -> Any:
    # Reuse YAML's scalar parser so "true"/"3"/"0.5"/"[a, b]" parse naturally.
    return yaml.safe_load(value)


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` strings onto a (copied) config dict."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        key, raw = ov.split("=", 1)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = _coerce(raw)
    return cfg


def load_config(config_path: str | Path, overrides: List[str] | None = None) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg
