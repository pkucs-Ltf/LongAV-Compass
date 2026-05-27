from __future__ import annotations

from pathlib import Path

from .io import read_yaml
from .schemas import SampleManifest


def load_manifest(path: str | Path) -> SampleManifest:
    manifest_path = Path(path).resolve()
    manifest = SampleManifest.from_dict(read_yaml(manifest_path))
    manifest.manifest_path = manifest_path.as_posix()
    return manifest
