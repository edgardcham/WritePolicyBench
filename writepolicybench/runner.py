from __future__ import annotations

from dataclasses import dataclass

from .baselines import apply_policy_actions, fifo_store_all_policy
from .episode_schema import Episode, Step
from .memory import ByteBudget, ByteMemoryStore


@dataclass
class Runner:
    """Scaffold runner; replace with real loading/execution logic."""

    store: ByteMemoryStore | None = None
    budget_bytes: int = 10_000

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = ByteMemoryStore(ByteBudget(self.budget_bytes))

    def run(self, episodes: int, config_path: str | None) -> None:
        _ = config_path
        for idx in range(episodes):
            episode = self._fake_episode(idx)
            for step in episode.steps:
                actions = fifo_store_all_policy(step, self.store)
                apply_policy_actions(self.store, actions, current_t=step.t)

    def _fake_episode(self, seed: int) -> Episode:
        steps = [Step(t=i, observation={"seed": seed, "i": i}) for i in range(3)]
        return Episode(steps=steps, labels={"seed": seed})
