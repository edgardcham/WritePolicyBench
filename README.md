# WritePolicyBench — Memory Write Policies under Byte Budgets

WritePolicyBench is a deterministic, streaming benchmark for **memory write policies**: deciding what to **WRITE / MERGE / EXPIRE / SKIP** under strict **byte budgets**.

The goal is to isolate “the write problem” (what enters memory, how it is updated, and what gets evicted) as a first-class evaluation target, with **byte-accurate accounting** and **reproducible grading**.

## What this repo contains

- `writepolicybench/` — benchmark implementation (episode schema, memory interface, evaluator)
- `data/episodes/` — frozen episode sets used for reproducible comparisons (see `MANIFEST.json`)
- `docs/` — benchmark specification and runbook
- `scripts/` — utilities (freeze episodes; sanity checks)
- `tests/` — unit tests for metric/budget invariants

## Quickstart

Create a virtualenv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Run the evaluator (writes `artifacts/results.csv`):

```bash
python3 -m writepolicybench.evaluator
```

(Optional) Run sanity checks:

```bash
python3 -m scripts.sanity_checks
```

(Optional) Freeze episodes (recommended for reproducible comparisons):

```bash
python3 -m scripts.freeze_episodes
```

## Tracks

The benchmark supports two evaluation tracks:

- **Unprivileged:** policies see only the observation stream (and explicitly allowed benign metadata).
- **Privileged:** policies additionally observe a bounded **priority** signal `p_t ∈ [0, 1]`.

## License

MIT (see `LICENSE`).
