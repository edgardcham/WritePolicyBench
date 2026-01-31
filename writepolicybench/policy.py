from __future__ import annotations

"""Policy interface.

The spec is framed as policies emitting memory actions under a byte budget.
In code we standardize on:

  select(step, store) -> Iterable[MemoryAction]

so a policy can emit EXPIRE(s) then a WRITE.
"""

from typing import Iterable, Protocol

from .episode_schema import Step
from .memory import ByteMemoryStore, MemoryAction


class WritePolicy(Protocol):
    """Select one or more memory actions for a single incoming step."""

    def select(self, step: Step, store: ByteMemoryStore) -> Iterable[MemoryAction]: ...
