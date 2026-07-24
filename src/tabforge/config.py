"""Configuration loading: default.yaml + optional preset overlay + tunings.

Config is a plain nested dict wrapped in a small accessor so stages can read
``cfg.get("articulation", "slide_max_ms")`` without KeyErrors surprising them.
Presets deep-merge over the defaults (rule: thresholds tunable without code).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Repo layout: this file is src/tabforge/config.py; config/ sits at repo root.
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent
CONFIG_DIR = _REPO_ROOT / "config"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """Read-only accessor over the merged configuration dict."""

    def __init__(self, data: dict[str, Any], preset: str | None = None):
        self._data = data
        self.preset = preset

    def section(self, name: str) -> dict[str, Any]:
        return self._data.get(name, {})

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def _config_dir(config_dir: str | Path | None) -> Path:
    return Path(config_dir) if config_dir else CONFIG_DIR


def load_config(preset: str | None = None,
                config_dir: str | Path | None = None) -> Config:
    """Load default.yaml, optionally overlaying config/presets/<preset>.yaml."""
    cdir = _config_dir(config_dir)
    with open(cdir / "default.yaml", "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if preset:
        preset_path = cdir / "presets" / f"{preset}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Unknown preset '{preset}': {preset_path}")
        with open(preset_path, "r", encoding="utf-8") as fh:
            overlay = yaml.safe_load(fh) or {}
        data = _deep_merge(data, overlay)
    return Config(data, preset=preset)


def load_tunings(config_dir: str | Path | None = None) -> dict[str, list[int]]:
    cdir = _config_dir(config_dir)
    with open(cdir / "tunings.yaml", "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("tunings", {})


def get_tuning(name: str, config_dir: str | Path | None = None) -> list[int]:
    tunings = load_tunings(config_dir)
    if name not in tunings:
        raise KeyError(
            f"Unknown tuning '{name}'. Available: {sorted(tunings)}"
        )
    return list(tunings[name])
