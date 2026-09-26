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

Config inheritance
------------------
A YAML may start with ``base: other.yaml`` (path relative to the YAML itself).
The base is loaded first and this file's keys are deep-merged on top, so an
experiment config only lists what differs from the pinned recipe.
"""
from __future__ import annotations

import copy
import os
import socket
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _deep_merge(base: Dict[str, Any], top: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``top`` onto a copy of ``base`` (``top`` wins)."""
    out = copy.deepcopy(base)
    for k, v in (top or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, resolving an optional ``base:`` parent config."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(load_yaml(path.parent / base), cfg)
    return cfg


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


def config_from_checkpoint(
    ckpt: Dict[str, Any],
    config_path: str | Path,
    overrides: List[str] | None = None,
    data_config: str | Path | None = None,
) -> Dict[str, Any]:
    """Config for evaluating/transcribing with a trained checkpoint.

    The model/keyboard/labels settings come from the checkpoint's OWN config so
    the architecture and warp always match the weights. ``data_config`` swaps in
    the ``data:`` block of another config (profile-resolved) -- this is how a
    model trained on one dataset is evaluated on the other (e.g. a PianoVAM
    checkpoint on PianoYT). CLI ``overrides`` are applied last.
    """
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg = copy.deepcopy(ckpt["cfg"])
    else:
        cfg = load_config(config_path)
    if data_config:
        cfg["data"] = copy.deepcopy(load_config(data_config)["data"])
        cfg["_data_config"] = str(data_config)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg
