"""Machine-readable contract for trajectories handed to PPO/GRPO training."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def _load_schema() -> dict[str, Any]:
    resource = files("grasp_proxy_rollout").joinpath("trajectory_schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


TRAJECTORY_SCHEMA = _load_schema()

__all__ = ["TRAJECTORY_SCHEMA"]
