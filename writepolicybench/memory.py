from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Literal, Protocol

from .episode_schema import Step


@dataclass
class ByteBudget:
    """Tracks byte usage for memory writes."""

    max_bytes: int
    used_bytes: int = 0

    def remaining(self) -> int:
        return max(self.max_bytes - self.used_bytes, 0)

    def consume(self, count: int) -> bool:
        if count < 0:
            raise ValueError("Cannot consume negative bytes")
        if self.used_bytes + count > self.max_bytes:
            return False
        self.used_bytes += count
        return True

    def credit(self, count: int) -> None:
        if count < 0:
            raise ValueError("Cannot credit negative bytes")
        self.used_bytes = max(self.used_bytes - count, 0)


def estimate_bytes(step: Step) -> int:
    """Estimate bytes for storing a step.

    Spec v0 accounting:
    - payload bytes (json)
    - metadata bytes (json)
    - header overhead (32)
    - per-item index overhead (16)
    """

    payload = len(json.dumps(step.observation, sort_keys=True))
    metadata = len(json.dumps(step.metadata, sort_keys=True))
    header = 32
    index_overhead = 16
    return payload + metadata + header + index_overhead


@dataclass
class MemoryItem:
    step: Step
    written_at: int
    byte_cost: int
    metadata: dict[str, Any] = field(default_factory=dict)


ActionType = Literal["SKIP", "WRITE", "MERGE", "EXPIRE"]


@dataclass(frozen=True)
class MemoryAction:
    action: ActionType
    step: Step | None = None
    target_t: int | None = None
    delta: dict[str, Any] | None = None
    reason: str | None = None


class MemoryStore(Protocol):
    """Interface for storing chosen steps."""

    def apply(self, action: MemoryAction) -> bool: ...

    def items(self) -> Iterable[MemoryItem]: ...

    def clear(self) -> None: ...


@dataclass
class ByteMemoryStore:
    """In-memory store with byte-budget enforcement."""

    budget: ByteBudget
    _items: dict[int, MemoryItem] = field(default_factory=dict)
    _order: list[int] = field(default_factory=list)

    def apply(self, action: MemoryAction, *, current_t: int | None = None) -> bool:
        """Apply an action.

        `current_t` is used to enforce EXPIRE age constraints (target must be older).
        If not provided, step-based callers should pass the current step.t.
        """

        if action.action == "SKIP":
            return True
        if action.action == "WRITE":
            if action.step is None:
                raise ValueError("WRITE requires step")
            return self.write(action.step)
        if action.action == "MERGE":
            if action.step is None or action.target_t is None:
                raise ValueError("MERGE requires step and target_t")
            return self.merge(action.target_t, action.step, action.delta)
        if action.action == "EXPIRE":
            if action.target_t is None:
                raise ValueError("EXPIRE requires target_t")
            if current_t is not None and action.target_t >= current_t:
                return False
            return self.expire(action.target_t)
        raise ValueError(f"Unknown action {action.action}")

    def write(self, step: Step, written_at: int | None = None) -> bool:
        cost = estimate_bytes(step)
        if not self.budget.consume(cost):
            return False
        target_t = step.t if written_at is None else written_at
        item = MemoryItem(step=step, written_at=target_t, byte_cost=cost)
        self._items[step.t] = item
        self._order.append(step.t)
        return True

    def merge(self, target_t: int, step: Step, delta: dict[str, Any] | None) -> bool:
        """Append-only merge with reviewer-proof constraints.

        We model MERGE as storing a delta item that *references* an existing
        base item. A MERGE is valid only if:
        - target exists and is a base WRITE item (not itself a MERGE delta)
        - both base and incoming observations are dicts with the same "api"
        - delta is the exact, shallow field-diff (excluding "api")
        - delta is non-empty (prevents zero-byte "writes" that inflate |W|)

        Instead of mutating the base in-place, we append a delta step carrying
        fixed merge metadata.
        """

        base_item = self._items.get(target_t)
        if base_item is None:
            return False

        # Disallow MERGE chains: target must be a base WRITE item.
        if base_item.step.metadata.get("merge_parent_t") is not None:
            return False

        # Guardrail: MERGE only within the same endpoint (observation["api"]).
        base_obs = base_item.step.observation
        new_obs = step.observation
        if not (isinstance(base_obs, dict) and isinstance(new_obs, dict)):
            return False
        base_api = base_obs.get("api")
        new_api = new_obs.get("api")
        if base_api is None or new_api is None or base_api != new_api:
            return False

        expected = _compute_delta(base_obs, new_obs)
        if delta is None:
            delta = expected
        else:
            # If a policy supplies a delta, it must match the canonical diff.
            if delta != expected:
                return False

        # Hard constraint: delta may not redefine the endpoint key.
        if "api" in delta:
            return False

        # Prevent compression exploits: a no-op merge should be rejected.
        if not delta:
            return False

        # Store delta as its own memory item (append-only).
        # Spec v0 charges MERGE as bytes(delta) + 16.
        delta_step = Step(
            t=step.t,
            observation=delta,
            metadata={
                "merge_parent_t": target_t,
                "merge_parent_api": base_api,
            },
        )

        cost = len(json.dumps(delta, sort_keys=True)) + 16
        if not self.budget.consume(cost):
            return False

        # Use step.t as the key (MERGE represents retaining timestep t).
        if delta_step.t in self._items:
            return False

        item = MemoryItem(step=delta_step, written_at=delta_step.t, byte_cost=cost)
        self._items[delta_step.t] = item
        self._order.append(delta_step.t)
        return True

    def expire(self, target_t: int) -> bool:
        item = self._items.pop(target_t, None)
        if item is None:
            return False
        self.budget.credit(item.byte_cost)
        try:
            self._order.remove(target_t)
        except ValueError:
            pass
        return True

    def oldest_item(self) -> MemoryItem | None:
        if not self._order:
            return None
        return self._items.get(self._order[0])

    def items(self) -> Iterable[MemoryItem]:
        for key in self._order:
            item = self._items.get(key)
            if item is not None:
                yield item

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()
        self.budget.used_bytes = 0


def _compute_delta(old_obs: Any, new_obs: Any) -> dict[str, Any]:
    """Compute a shallow, fieldwise delta from old_obs -> new_obs.

    Delta semantics (spec v0): for dict observations we include only keys whose
    values changed, excluding the primary key field "api".
    """

    if not isinstance(old_obs, dict) or not isinstance(new_obs, dict):
        return {"value": new_obs}

    delta: dict[str, Any] = {}
    for key, value in new_obs.items():
        if key == "api":
            continue
        if old_obs.get(key) != value:
            delta[key] = value
    return delta
