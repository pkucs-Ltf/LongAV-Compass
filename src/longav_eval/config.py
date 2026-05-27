from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_yaml


def load_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> dict[str, Any]:
    return read_yaml(Path(path))


def load_api_keys(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    return read_yaml(candidate)

