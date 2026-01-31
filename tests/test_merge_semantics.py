from __future__ import annotations

import unittest

from writepolicybench.episode_schema import Episode, Step
from writepolicybench.evaluator import run_policy_on_episode
from writepolicybench.memory import ByteBudget, ByteMemoryStore, MemoryAction


class MergeSemanticsTests(unittest.TestCase):
    def test_merge_requires_existing_base(self) -> None:
        store = ByteMemoryStore(ByteBudget(max_bytes=10_000))
        step0 = Step(t=0, observation={"api": "a", "params": ["x"], "version": 1}, metadata={})
        ok = store.apply(MemoryAction(action="MERGE", step=step0, target_t=0, delta={"params": ["x"]}))
        self.assertFalse(ok)

    def test_merge_requires_same_api(self) -> None:
        store = ByteMemoryStore(ByteBudget(max_bytes=10_000))
        base = Step(t=0, observation={"api": "a", "params": ["x"], "version": 1}, metadata={})
        store.apply(MemoryAction(action="WRITE", step=base))

        incoming = Step(t=1, observation={"api": "b", "params": ["y"], "version": 1}, metadata={})
        ok = store.apply(MemoryAction(action="MERGE", step=incoming, target_t=0, delta={"params": ["y"]}))
        self.assertFalse(ok)

    def test_merge_rejects_empty_delta(self) -> None:
        store = ByteMemoryStore(ByteBudget(max_bytes=10_000))
        base = Step(t=0, observation={"api": "a", "params": ["x"], "version": 1}, metadata={})
        store.apply(MemoryAction(action="WRITE", step=base))

        # Identical observation -> canonical delta is empty -> MERGE must be rejected.
        incoming = Step(t=1, observation={"api": "a", "params": ["x"], "version": 1}, metadata={})
        ok = store.apply(MemoryAction(action="MERGE", step=incoming, target_t=0, delta={}))
        self.assertFalse(ok)

    def test_merge_rejects_delta_mismatch(self) -> None:
        store = ByteMemoryStore(ByteBudget(max_bytes=10_000))
        base = Step(t=0, observation={"api": "a", "params": ["x"], "version": 1}, metadata={})
        store.apply(MemoryAction(action="WRITE", step=base))

        incoming = Step(t=1, observation={"api": "a", "params": ["x", "y"], "version": 1}, metadata={})
        # Canonical delta is {"params": ["x", "y"]}; supplying the wrong delta must fail.
        ok = store.apply(MemoryAction(action="MERGE", step=incoming, target_t=0, delta={"version": 2}))
        self.assertFalse(ok)

    def test_orphan_merge_does_not_count_toward_W(self) -> None:
        ep = Episode(
            steps=[
                Step(t=0, observation={"api": "a", "params": ["x"], "version": 1}, metadata={"drift": False}),
                Step(t=1, observation={"api": "a", "params": ["x", "y"], "version": 1}, metadata={"drift": True}),
            ],
            labels={
                "critical_steps": [1],
                "total_drift_events": 1,
                "utility_by_step": {0: 1.0, 1: 5.0},
            },
        )

        def policy(step: Step, store: ByteMemoryStore):
            _ = store
            if step.t == 0:
                return [MemoryAction(action="WRITE", step=step)]
            if step.t == 1:
                return [
                    MemoryAction(action="MERGE", step=step, target_t=0, delta=None),
                    MemoryAction(action="EXPIRE", target_t=0),
                ]
            return [MemoryAction(action="SKIP")]

        r = run_policy_on_episode(policy, ep, budget_bytes=10_000, track="unprivileged")
        # Base is expired, leaving only an orphan delta in memory => W should be empty.
        self.assertEqual(r["write_density"], 0.0)
        self.assertEqual(r["drift_coverage"], 0.0)
        self.assertEqual(r["recall"], 0.0)
        self.assertEqual(r["policy_utility"], 0.0)

    def test_merge_counts_when_base_present(self) -> None:
        ep = Episode(
            steps=[
                Step(t=0, observation={"api": "a", "params": ["x"], "version": 1}, metadata={"drift": False}),
                Step(t=1, observation={"api": "a", "params": ["x", "y"], "version": 1}, metadata={"drift": True}),
            ],
            labels={
                "critical_steps": [1],
                "total_drift_events": 1,
                "utility_by_step": {0: 1.0, 1: 5.0},
            },
        )

        def policy(step: Step, store: ByteMemoryStore):
            _ = store
            if step.t == 0:
                return [MemoryAction(action="WRITE", step=step)]
            if step.t == 1:
                return [MemoryAction(action="MERGE", step=step, target_t=0, delta=None)]
            return [MemoryAction(action="SKIP")]

        r = run_policy_on_episode(policy, ep, budget_bytes=10_000, track="unprivileged")
        # W should include both timesteps (0 from WRITE, 1 from MERGE).
        self.assertEqual(r["write_density"], 1.0)
        self.assertEqual(r["drift_coverage"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["policy_utility"], 6.0)


if __name__ == "__main__":
    unittest.main()
