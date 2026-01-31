from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Step:
    """Single timestep in a streaming episode."""

    t: int
    observation: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Episode:
    """Container for a sequence of steps and optional labels."""

    steps: list[Step]
    labels: dict[str, Any] = field(default_factory=dict)
