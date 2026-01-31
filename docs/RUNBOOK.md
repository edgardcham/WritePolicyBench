# Runbook (v0)

## Setup (uv)

```bash
cd /home/edgard/projects/writepolicybench
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"

uv venv
uv pip install -e .
```

## Smoke run

```bash
python3 -m writepolicybench --episodes 2
```

## Notes
- The benchmark spec lives in `docs/BENCHMARK_SPEC.md`
- Smoke test plan: `docs/TEST_PLAN.md`
