from __future__ import annotations

from pathlib import Path

from .io import read_yaml
from .schemas import ConfigError, Profile


def load_profile(path: str | Path, known_metric_ids: set[str]) -> Profile:
    profile = Profile.from_dict(read_yaml(path))
    unknown_metrics = sorted(set(profile.metrics) - known_metric_ids)
    if unknown_metrics:
        joined = ", ".join(unknown_metrics)
        raise ConfigError(f"Profile '{profile.profile_name}' references unknown metrics: {joined}")
    return profile

