"""Configuration loading.

A config is a nested dict loaded from YAML. We support dotted-key overrides
from the command line, e.g. ``train.batch_size=8`` or ``data.root=/x``.
Values are coerced from strings using YAML scalar parsing so that
``model.use_velocity=true`` becomes a real bool, ``labels.fps=25`` an int, etc.

Server profiles
---------------
The YAML may define a ``profiles:`` block with machine-specific values (paths,
device, ...). The active profile is chosen, highest priority first, by:
  1. a CLI override  ``profile=dib``
  2. the env var     ``PIANOVAM_SERVER=dib``
  3. ``profile:`` in the YAML (``auto`` -> matched against the hostname)
The chosen profile's ``overrides:`` are applied first; explicit CLI overrides
still win over them, so you can always tweak a single field by hand.
"""
from __future__ import annotations

import copy
import os
import socket
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce(value: str) -> Any:
    # Reuse YAML's scalar parser so "true"/"3"/"0.5"/"[a, b]" parse naturally.
    return yaml.safe_load(value)


def _set_dotted(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c`` to an already-typed value, creating dicts as needed."""
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` strings onto a (copied) config dict."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        key, raw = ov.split("=", 1)
        _set_dotted(cfg, key, _coerce(raw))
    return cfg


def select_profile_name(cfg: Dict[str, Any], overrides: List[str]) -> str | None:
    """Resolve which server profile to use (or None if profiles are unused)."""
    profiles = cfg.get("profiles") or {}
    if not profiles:
        return None
    # 1. env var, 2. CLI `profile=...`, 3. YAML `profile:` (default 'auto').
    sel = os.environ.get("PIANOVAM_SERVER")
    if not sel:
        for ov in overrides:
            if ov.startswith("profile="):
                sel = ov.split("=", 1)[1]
                break
    if not sel:
        sel = cfg.get("profile", "auto")
    if sel == "auto":
        host = socket.gethostname()
        for name, p in profiles.items():
            hp = (p or {}).get("hostname")
            if hp and hp in host:
                return name
        return None
    return sel if sel in profiles else None


def load_config(config_path: str | Path, overrides: List[str] | None = None) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    overrides = list(overrides or [])

    # 1. Apply the active server profile's defaults (paths / device / ...).
    sel = select_profile_name(cfg, overrides)
    if sel:
        for k, v in (cfg["profiles"][sel].get("overrides") or {}).items():
            _set_dotted(cfg, k, copy.deepcopy(v))
        cfg["_active_profile"] = sel

    # 2. Explicit CLI overrides win over the profile.
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg
