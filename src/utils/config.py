"""Config loading with single-level `inherits:` merging and dotted access.

Configs are plain YAML. A child config may declare `inherits: base.yaml` (path
relative to the same directory) and override any subset of keys via a recursive
dict merge. Kept deliberately small — no external config framework.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


class Config(dict):
    """Dict with attribute access and recursive wrapping for nested dicts."""

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(val) if isinstance(val, dict) else val

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving a single `inherits:` parent if present."""
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    parent_name = raw.pop("inherits", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        merged = _deep_merge(parent, raw)
    else:
        merged = raw
    return Config(merged)
