from __future__ import annotations

import unittest

from writepolicybench.baselines import fifo_store_all_policy, no_mem_policy
from writepolicybench.evaluator import run_policy_on_episode
from writepolicybench.synthetic import DriftConfig, generate_synthetic_drift_episode


class MetricsSanityTests(unittest.TestCase):
    def test_no_mem_policy_has_zero_density_and_utilization(self) -> None:
        ep = generate_synthetic_drift_episode(
            0, DriftConfig(steps=5, api_pool=1, drift_prob=1.0, seed=123)
        )
        r = run_policy_on_episode(no_mem_policy, ep, budget_bytes=10_000, track="unprivileged")
        self.assertEqual(r["write_density"], 0.0)
        self.assertEqual(r["utilization"], 0.0)
        self.assertEqual(r["avg_staleness"], 0.0)
        self.assertEqual(r["drift_coverage"], 0.0)
        self.assertEqual(r["expire_rate"], 0.0)

    def test_fifo_store_all_writes_everything_and_covers_drift(self) -> None:
        ep = generate_synthetic_drift_episode(
            0, DriftConfig(steps=5, api_pool=1, drift_prob=1.0, seed=123)
        )
        r = run_policy_on_episode(fifo_store_all_policy, ep, budget_bytes=1_000_000, track="unprivileged")
        self.assertEqual(r["write_density"], 1.0)
        self.assertEqual(r["drift_coverage"], 1.0)
        # current_t=4, steps t=0..4 => mean staleness = (4+3+2+1+0)/5 = 2
        self.assertAlmostEqual(r["avg_staleness"], 2.0)
        self.assertEqual(r["expire_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
