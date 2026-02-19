"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config into a plain dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

