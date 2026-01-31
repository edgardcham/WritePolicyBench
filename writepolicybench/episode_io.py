from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .episode_schema import Episode, Step


def _step_from_dict(raw: dict) -> Step:
    t = raw.get("t")
    if t is None:
        raise ValueError("Step missing 't' field")
    observation = raw.get("observation")
    metadata = raw.get("metadata") or {}
    return Step(t=int(t), observation=observation, metadata=dict(metadata))


def _episode_from_dict(raw: dict) -> Episode:
    steps_raw = raw.get("steps")
    if steps_raw is None:
        raise ValueError("Episode missing 'steps' field")
    steps = [_step_from_dict(step) for step in steps_raw]
    labels = raw.get("labels") or {}
    return Episode(steps=steps, labels=dict(labels))


def iter_episodes_jsonl(path: str | Path) -> Iterable[Episode]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            yield _episode_from_dict(payload)


def load_episodes_jsonl(path: str | Path) -> list[Episode]:
    return list(iter_episodes_jsonl(path))


def write_episodes_jsonl(path: str | Path, episodes: Iterable[Episode]) -> None:
    file_path = Path(path)
    with file_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            payload = {
                "steps": [
                    {
                        "t": step.t,
                        "observation": step.observation,
                        "metadata": step.metadata,
                    }
                    for step in episode.steps
                ],
                "labels": episode.labels,
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
