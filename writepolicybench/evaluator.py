from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .baselines import (
    apply_policy_actions,
    fifo_store_all_policy,
    last_kb_policy,
    merge_aggressive_policy,
    no_mem_policy,
    priority_threshold_policy,
    uniform_sample_policy,
    utility_greedy_policy,
    utility_threshold_policy,
)
from .episode_schema import Episode, Step
from .episode_io import load_episodes_jsonl, write_episodes_jsonl
from .memory import ByteBudget, ByteMemoryStore
from .synthetic import DriftConfig, generate_synthetic_drift_episodes, score_written_steps


Track = Literal["unprivileged", "privileged"]
PolicyFn = Callable[[Step, ByteMemoryStore], list]


@dataclass(frozen=True)
class RunConfig:
    episodes: int = 10
    steps: int = 200
    seed: int = 0


_POLICY_VISIBLE_KEYS: dict[Track, tuple[str, ...]] = {
    # Unprivileged: the policy cannot access labels or any supervision-derived signals.
    # It receives only the observation and limited, explicitly-allowed metadata.
    "unprivileged": ("mode",),
    # Privileged: additionally receives a scalar `priority` hint.
    "privileged": ("mode", "priority"),
}


def policy_view_step(step: Step, *, track: Track) -> Step:
    allow = _POLICY_VISIBLE_KEYS[track]
    md = {k: step.metadata[k] for k in allow if k in step.metadata}
    return Step(t=step.t, observation=step.observation, metadata=md)


def run_policy_on_episode(policy: PolicyFn, episode: Episode, *, budget_bytes: int, track: Track) -> dict:
    store = ByteMemoryStore(ByteBudget(max_bytes=budget_bytes))

    write_actions = 0
    expire_actions = 0

    # Policy/cost view of steps (what could be written to memory under this track).
    view_steps: list[Step] = [policy_view_step(s, track=track) for s in episode.steps]

    for step in view_steps:
        actions = policy(step, store)
        write_actions += sum(1 for a in actions if getattr(a, "action", None) == "WRITE")
        expire_actions += sum(1 for a in actions if getattr(a, "action", None) == "EXPIRE")
        apply_policy_actions(store, actions, current_t=step.t)

    written_steps = [item.step for item in store.items()]
    metrics = score_written_steps(
        episode,
        written_steps,
        bytes_used=store.budget.used_bytes,
        budget_bytes=budget_bytes,
        write_actions=write_actions,
        expire_actions=expire_actions,
        current_t=(episode.steps[-1].t if episode.steps else 0),
        cost_steps=view_steps,
    )
    return {
        **metrics,
        "bytes_used": store.budget.used_bytes,
        "budget_bytes": budget_bytes,
        "episode_id": episode.labels.get("episode_id"),
        "write_actions": write_actions,
        "expire_actions": expire_actions,
        "track": track,
    }


def evaluate_baselines(out_csv: str | Path, budgets: list[int], cfg: RunConfig) -> None:
    # Evaluate across multiple regimes to expose policy tradeoffs.
    modes = ["default", "burst_drift", "redundancy", "burst_redundancy"]

    policies_by_track: dict[Track, list[tuple[str, PolicyFn]]] = {
        "unprivileged": [
            ("no_mem", no_mem_policy),
            ("fifo_store_all", fifo_store_all_policy),
            ("uniform_sample", lambda step, store: uniform_sample_policy(step, store, every_n=10)),
            ("last_kb", last_kb_policy),
            ("merge_aggressive", merge_aggressive_policy),
        ],
        "privileged": [
            ("no_mem", no_mem_policy),
            ("fifo_store_all", fifo_store_all_policy),
            ("uniform_sample", lambda step, store: uniform_sample_policy(step, store, every_n=10)),
            ("priority_threshold", lambda step, store: priority_threshold_policy(step, store, threshold=0.5)),
            # NOTE: priority_threshold_strict previously duplicated priority_threshold
            # under the current priority_v1 schema; removed to reduce plot/table clutter.
            ("priority_greedy", utility_greedy_policy),
            ("last_kb", last_kb_policy),
            ("merge_aggressive", merge_aggressive_policy),
        ],
    }

    rows: list[dict] = []
    for mode in modes:
        drift_cfg = DriftConfig(steps=cfg.steps, seed=cfg.seed, mode=mode)
        frozen = (
            Path("data/episodes")
            / f"episodes__schema=priority_v1__mode={mode}__seed={cfg.seed}__steps={cfg.steps}__n={cfg.episodes}.jsonl"
        )
        if frozen.exists():
            episodes = load_episodes_jsonl(frozen)
        else:
            episodes = generate_synthetic_drift_episodes(cfg.episodes, drift_cfg)
            frozen.parent.mkdir(parents=True, exist_ok=True)
            write_episodes_jsonl(frozen, episodes)

        for budget in budgets:
            for ep in episodes:
                ep.labels["budget_bytes"] = budget
                for track, policies in policies_by_track.items():
                    for name, pol in policies:
                        r = run_policy_on_episode(pol, ep, budget_bytes=budget, track=track)
                        r["policy"] = name
                        r["mode"] = mode
                        rows.append(r)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    budgets = [1024, 2048, 4096, 8192, 10_240, 16_384, 32_768, 65_536, 102_400, 262_144, 1_048_576]
    evaluate_baselines("artifacts/results.csv", budgets=budgets, cfg=RunConfig())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
