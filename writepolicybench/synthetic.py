from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from .episode_schema import Episode, Step


@dataclass(frozen=True)
class DriftConfig:
    steps: int = 200
    api_pool: int = 8
    drift_prob: float = 0.08
    max_params: int = 6
    seed: int = 0
    # default|burst_drift|redundancy|burst_redundancy
    mode: str = "default"

    # burst mode params
    burst_interval: int = 50
    burst_len: int = 8
    burst_drift_prob: float = 0.6

    # redundancy mode params
    redundancy_prob: float = 0.7


def _build_observation(api_id: int, version: int, params: list[str], deprecated: bool) -> dict:
    return {
        "api": f"api.v{version}.endpoint_{api_id}",
        "params": list(params),
        "deprecated": deprecated,
        "version": version,
    }


def generate_synthetic_drift_episode(episode_id: int, config: DriftConfig) -> Episode:
    rng = random.Random(config.seed + episode_id)
    versions = [1 for _ in range(config.api_pool)]
    params = [
        [f"p{idx}_{j}" for j in range(rng.randint(2, config.max_params))]
        for idx in range(config.api_pool)
    ]
    steps: list[Step] = []
    critical_steps: list[int] = []
    utilities: dict[int, float] = {}

    def is_burst_window(t: int) -> bool:
        if config.burst_interval <= 0:
            return False
        start = (t // config.burst_interval) * config.burst_interval
        return (t - start) < config.burst_len

    for t in range(config.steps):
        bursty = config.mode in {"burst_drift", "burst_redundancy"}
        redundant = config.mode in {"redundancy", "burst_redundancy"}

        # redundancy: often repeat the same api_id across a streak
        if redundant and steps and rng.random() < config.redundancy_prob:
            prev_obs = steps[-1].observation
            api_id = int(prev_obs.get("api", "api.v1.endpoint_0").split("endpoint_")[-1]) if isinstance(prev_obs, dict) else 0
        else:
            api_id = rng.randrange(config.api_pool)

        # drift probability by regime
        if bursty and is_burst_window(t):
            drift_p = config.burst_drift_prob
        else:
            drift_p = config.drift_prob

        drift = rng.random() < drift_p
        if drift:
            versions[api_id] += 1
            if rng.random() < 0.5 and params[api_id]:
                params[api_id] = params[api_id][:-1]
            else:
                params[api_id].append(f"p{api_id}_{versions[api_id]}")

        deprecated = drift and rng.random() < 0.3
        observation = _build_observation(api_id, versions[api_id], params[api_id], deprecated)

        # utility regime: bursts/true drift are high utility; redundancy repeats are low
        if drift:
            utility = 6.0 if (bursty and is_burst_window(t)) else 5.0
        else:
            # In redundancy regimes, repeated observations are low-utility.
            utility = 0.5 if (redundant and steps and api_id == (steps[-1].observation.get("api_id", api_id) if isinstance(steps[-1].observation, dict) else api_id)) else 1.0

        # Policy-visible metadata.
        #
        # IMPORTANT: `utility` is a supervision/label signal and is *not* policy-visible.
        # For the privileged track we expose a bounded surrogate signal `priority`.
        priority = max(0.0, min(1.0, float(utility) / 6.0))
        metadata = {
            "mode": config.mode,
            "priority": priority,
        }

        if drift:
            critical_steps.append(t)

        utilities[t] = float(utility)
        steps.append(Step(t=t, observation=observation, metadata=metadata))

    labels = {
        "episode_id": episode_id,
        "mode": config.mode,
        "critical_steps": critical_steps,
        "total_drift_events": len(critical_steps),
        "utility_by_step": utilities,
        "max_utility": float(sum(utilities.values())),
    }
    return Episode(steps=steps, labels=labels)


def generate_synthetic_drift_episodes(count: int, config: DriftConfig) -> list[Episode]:
    return [generate_synthetic_drift_episode(idx, config) for idx in range(count)]


def score_written_steps(
    episode: Episode,
    written_steps: Iterable[Step],
    bytes_used: int,
    *,
    budget_bytes: int,
    write_actions: int,
    expire_actions: int,
    current_t: int | None = None,
    cost_steps: list[Step] | None = None,
) -> dict[str, float]:
    """Compute benchmark metrics.

    Definitions live in docs/BENCHMARK_SPEC.md.

    `write_actions`/`expire_actions` are counts of actions emitted by the policy
    over the episode.
    """

    written_steps = list(written_steps)

    # Interpret final memory as a set of *retained timesteps* W.
    # - WRITE retains its own timestep.
    # - MERGE retains timestep t only if it references a still-present base item
    #   of the same endpoint (prevents cross-endpoint and orphan-delta exploits).
    base_ts: set[int] = set()
    merge_ts: set[int] = set()
    merge_parent: dict[int, int] = {}

    for step in written_steps:
        parent_t = step.metadata.get("merge_parent_t")
        if parent_t is None:
            base_ts.add(int(step.t))
        else:
            try:
                t = int(step.t)
                p = int(parent_t)
            except Exception:
                continue
            merge_ts.add(t)
            merge_parent[t] = p

    # Episode endpoint identity (for the same-endpoint MERGE constraint).
    episode_api: dict[int, str | None] = {}
    for s in episode.steps:
        api = s.observation.get("api") if isinstance(s.observation, dict) else None
        episode_api[int(s.t)] = api if isinstance(api, str) else None

    retained_ts: set[int] = set(base_ts)
    for t in merge_ts:
        p = merge_parent.get(t)
        if p is None or p not in base_ts:
            continue
        if episode_api.get(t) is None or episode_api.get(p) is None:
            continue
        if episode_api[t] != episode_api[p]:
            continue
        retained_ts.add(t)

    # --- Success metrics ---
    critical_steps = set(episode.labels.get("critical_steps", []))
    critical_written = len(critical_steps.intersection(retained_ts))
    recall = critical_written / len(critical_steps) if critical_steps else 0.0
    precision = critical_written / len(retained_ts) if retained_ts else 0.0
    f1 = 0.0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)

    # --- Utility metrics ---
    from .memory import estimate_bytes

    utility_by_step = episode.labels.get("utility_by_step", {})

    def _u(t: int) -> float:
        # NOTE: when episodes are loaded from JSON, dict keys may become strings.
        return float(utility_by_step.get(t, utility_by_step.get(str(t), 0.0)) or 0.0)

    policy_utility = float(sum(_u(int(t)) for t in retained_ts))
    utility_per_kb = 0.0 if bytes_used <= 0 else policy_utility / (bytes_used / 1024.0)

    # Regret spec: regret(t) = optimal_utility(t) - policy_utility(t), where the
    # optimal reference is budget-constrained under the same byte model.
    #
    # We compute final-timestep regret via a 0/1 knapsack oracle:
    #   weight(step) = estimate_bytes(step)
    #   value(step)  = episode.labels["utility_by_step"][t] (hidden label)
    #
    # Exact DP is O(T * B) where T is number of steps and B is the budget (bytes).
    # For large B, we fall back to a fast ratio-greedy approximation.
    steps_for_cost = cost_steps if cost_steps is not None else episode.steps
    step_costs: list[int] = [estimate_bytes(s) for s in steps_for_cost]

    # Oracle values come from hidden per-step utility labels, not policy-visible metadata.
    step_utils: list[float] = [_u(int(s.t)) for s in steps_for_cost]
    total_cost = sum(step_costs)

    if budget_bytes >= total_cost:
        oracle_utility = float(sum(step_utils))
    else:
        # Exact knapsack DP is O(T * B) and becomes expensive when we sweep many budgets.
        # We therefore apply exact DP only for small budgets and use a greedy approximation otherwise.
        dp_budget_cap = 2_048
        max_ops = 25_000_000
        if budget_bytes <= dp_budget_cap and (len(step_costs) * budget_bytes) <= max_ops:
            dp = [0.0] * (budget_bytes + 1)
            for w, v in zip(step_costs, step_utils, strict=True):
                if w > budget_bytes:
                    continue
                for b in range(budget_bytes, w - 1, -1):
                    cand = dp[b - w] + v
                    if cand > dp[b]:
                        dp[b] = cand
            oracle_utility = float(max(dp))
        else:
            items = [(v / w if w > 0 else 0.0, w, v) for w, v in zip(step_costs, step_utils, strict=True)]
            items.sort(reverse=True)
            used = 0
            oracle_utility = 0.0
            picked: list[tuple[int, float]] = []
            for _, w, v in items:
                if used + w <= budget_bytes:
                    used += w
                    oracle_utility += v
                    picked.append((w, v))
            # Tiny local improvement: swap in a better single item if it fits.
            remaining = [(w, v) for _, w, v in items if (w, v) not in picked]
            for w_new, v_new in remaining:
                for i, (w_old, v_old) in enumerate(list(picked)):
                    if used - w_old + w_new <= budget_bytes and v_new > v_old:
                        used = used - w_old + w_new
                        oracle_utility = oracle_utility - v_old + v_new
                        picked[i] = (w_new, v_new)
                        break

    # Regret is defined against a WRITE-only oracle under the v0 byte estimator.
    # When MERGE is enabled, a policy can (legitimately) exceed this baseline;
    # we clamp to keep regret interpretable (non-negative).
    regret_write_only = float(max(0.0, oracle_utility - policy_utility))

    # --- Staleness metrics ---
    total_drift = float(episode.labels.get("total_drift_events", 0) or 0)
    # In v0 synthetic episodes, drift events are labeled via critical_steps.
    drift_written = len(set(episode.labels.get("critical_steps", [])).intersection(retained_ts))
    drift_coverage = (float(drift_written) / total_drift) if total_drift > 0 else 0.0

    if current_t is None:
        current_t = episode.steps[-1].t if episode.steps else 0
    stalenesses = [float(current_t - int(t)) for t in retained_ts]
    avg_staleness = (sum(stalenesses) / len(stalenesses)) if stalenesses else 0.0

    expire_rate = (float(expire_actions) / float(write_actions)) if write_actions > 0 else 0.0

    # --- Budget efficiency ---
    utilization = (float(bytes_used) / float(budget_bytes)) if budget_bytes > 0 else 0.0
    write_density = (float(len(retained_ts)) / float(len(episode.steps))) if episode.steps else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "utility_per_kb": utility_per_kb,
        "regret_write_only": regret_write_only,
        "regret": regret_write_only,  # backwards-compat alias (paper uses regret_write_only)
        "oracle_utility": float(oracle_utility),
        "policy_utility": float(policy_utility),
        "drift_coverage": drift_coverage,
        "avg_staleness": avg_staleness,
        "expire_rate": expire_rate,
        "utilization": utilization,
        "write_density": write_density,
    }
