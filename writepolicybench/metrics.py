from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    metadata: dict[str, Any] | None = None


def aggregate(results: list[MetricResult]) -> dict[str, float]:
    """Placeholder aggregation over results."""

    return {r.name: r.value for r in results}
