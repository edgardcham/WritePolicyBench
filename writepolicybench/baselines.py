from __future__ import annotations

import json
from typing import Callable, Iterable

from .episode_schema import Step
from .memory import ByteMemoryStore, MemoryAction, MemoryItem, estimate_bytes


def utility_threshold_policy(step: Step, store: ByteMemoryStore, threshold: float = 0.5) -> list[MemoryAction]:
    """Write only if step priority is high enough; otherwise SKIP.

    Historically this baseline referenced `utility`, but utility is not policy-visible
    in WritePolicyBench. The policy-visible surrogate signal is `priority`.
    """

    _ = store
    pr = float(step.metadata.get("priority", 0.0))
    if pr >= threshold:
        return [MemoryAction(action="WRITE", step=step, reason=f"priority>={threshold}")]
    return [MemoryAction(action="SKIP", reason=f"priority<{threshold}")]


def no_mem_policy(step: Step, store: ByteMemoryStore) -> list[MemoryAction]:
    _ = store
    return [MemoryAction(action="SKIP")]


def fifo_store_all_policy(step: Step, store: ByteMemoryStore) -> list[MemoryAction]:
    cost = estimate_bytes(step)
    if cost <= store.budget.remaining():
        return [MemoryAction(action="WRITE", step=step)]
    return [MemoryAction(action="SKIP", reason="budget_exhausted")]


def last_kb_policy(step: Step, store: ByteMemoryStore) -> list[MemoryAction]:
    cost = estimate_bytes(step)
    remaining = store.budget.remaining()
    actions: list[MemoryAction] = []

    while cost > remaining:
        oldest = store.oldest_item()
        if oldest is None:
            return [MemoryAction(action="SKIP", reason="oversize_step")]
        actions.append(MemoryAction(action="EXPIRE", target_t=oldest.step.t))
        remaining += oldest.byte_cost

    actions.append(MemoryAction(action="WRITE", step=step))
    return actions


def uniform_sample_policy(step: Step, store: ByteMemoryStore, *, every_n: int = 10, start: int = 0) -> list[MemoryAction]:
    """Write every Nth step (deterministic).

    If the selected step does not fit in remaining budget, SKIP.
    """

    if every_n <= 0:
        raise ValueError("every_n must be > 0")

    if (step.t - start) % every_n != 0:
        return [MemoryAction(action="SKIP", reason=f"t%{every_n}!=0")]

    cost = estimate_bytes(step)
    if cost <= store.budget.remaining():
        return [MemoryAction(action="WRITE", step=step, reason=f"uniform_every_{every_n}")]
    return [MemoryAction(action="SKIP", reason="budget_exhausted")]


def priority_threshold_policy(step: Step, store: ByteMemoryStore, *, threshold: float = 0.5) -> list[MemoryAction]:
    """Write steps with priority above a threshold."""

    _ = store
    pr = float(step.metadata.get("priority", 0.0))
    if pr > threshold:
        return [MemoryAction(action="WRITE", step=step, reason=f"priority>{threshold}")]
    return [MemoryAction(action="SKIP", reason=f"priority<={threshold}")]


def _step_priority(item: MemoryItem) -> float:
    return float(item.step.metadata.get("priority", 0.0))


def utility_greedy_policy(step: Step, store: ByteMemoryStore) -> list[MemoryAction]:
    """Greedy priority policy.

    Historically this baseline referenced `utility`, but utility is not policy-visible
    in WritePolicyBench. The policy-visible surrogate signal is `priority`.

    Online approximation to "keep the highest-priority steps":
    - If the step fits, WRITE.
    - If it doesn't fit, evict the lowest-priority items until it fits, but only
      if the incoming step's priority is higher than the lowest priority present.
    """

    incoming_pr = float(step.metadata.get("priority", 0.0))
    cost = estimate_bytes(step)
    remaining = store.budget.remaining()

    if cost <= remaining:
        return [MemoryAction(action="WRITE", step=step, reason="fits")]

    items = list(store.items())
    if not items:
        return [MemoryAction(action="SKIP", reason="oversize_step")]

    lowest_pr = min(_step_priority(it) for it in items)
    if incoming_pr <= lowest_pr:
        return [MemoryAction(action="SKIP", reason="low_priority_vs_store")]

    # Expire lowest-priority items (tie-break by age) until enough room.
    actions: list[MemoryAction] = []
    evictables = sorted(items, key=lambda it: (_step_priority(it), it.step.t))
    freed = 0
    for it in evictables:
        actions.append(MemoryAction(action="EXPIRE", target_t=it.step.t))
        freed += it.byte_cost
        if cost <= remaining + freed:
            actions.append(MemoryAction(action="WRITE", step=step, reason="priority_greedy_replace"))
            return actions

    return [MemoryAction(action="SKIP", reason="cannot_free_enough")]


def _compute_delta(old_obs: object, new_obs: object) -> dict:
    if not isinstance(old_obs, dict) or not isinstance(new_obs, dict):
        return {"value": new_obs}
    delta: dict = {}
    for k, v in new_obs.items():
        if old_obs.get(k) != v:
            delta[k] = v
    return delta


def merge_aggressive_policy(step: Step, store: ByteMemoryStore) -> list[MemoryAction]:
    """Prefer MERGE into an existing item with the same API when possible.

    Deterministic heuristic:
    - If a prior item with the same observation["api"] exists, MERGE a delta into
      the most recent matching item.
    - Otherwise, fall back to a recency-style policy (expire oldest to make room
      and WRITE).
    """

    api = step.observation.get("api") if isinstance(step.observation, dict) else None
    if api is not None:
        prior_items = list(store.items())
        target: MemoryItem | None = None
        for it in reversed(prior_items):
            obs = it.step.observation
            if isinstance(obs, dict) and obs.get("api") == api:
                target = it
                break

        if target is not None:
            delta = _compute_delta(target.step.observation, step.observation)
            merge_cost = len(json.dumps(delta, sort_keys=True)) + 16
            remaining = store.budget.remaining()

            actions: list[MemoryAction] = []
            while merge_cost > remaining:
                oldest = store.oldest_item()
                if oldest is None:
                    return [MemoryAction(action="SKIP", reason="merge_oversize")]
                actions.append(MemoryAction(action="EXPIRE", target_t=oldest.step.t))
                remaining += oldest.byte_cost

            actions.append(
                MemoryAction(
                    action="MERGE",
                    step=step,
                    target_t=target.step.t,
                    delta=delta,
                    reason="merge_aggressive",
                )
            )
            return actions

    # No merge target available.
    return last_kb_policy(step, store)


def apply_policy_actions(store: ByteMemoryStore, actions: Iterable[MemoryAction], *, current_t: int | None = None) -> None:
    """Apply actions and enforce store constraints.

    Pass `current_t` (the timestep currently being processed) so EXPIRE can
    enforce the spec's age constraint.
    """

    for action in actions:
        store.apply(action, current_t=current_t)
