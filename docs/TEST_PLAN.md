# Test Plan (Smoke) — WritePolicyBench (v0)

Goal: ensure the v0 harness runs end-to-end for the synthetic drift benchmark, and that budget enforcement + baselines behave sensibly.

## 1) Synthetic drift generator
- [ ] `python -m writepolicybench --help` works
- [ ] A generator script can produce a JSONL file with N episodes/steps deterministically given a seed.
- [ ] Regenerating with the same seed produces byte-identical output.

## 2) Episode JSONL IO
- [ ] `load_jsonl(path)` loads all episodes/steps without error.
- [ ] Round-trip: `write_jsonl(load_jsonl(x))` yields equivalent episodes.
- [ ] Malformed line handling: invalid JSON line is rejected with clear error.

## 3) Byte budget enforcement
- [ ] With budget=0 bytes: no writes occur.
- [ ] With tiny budget (< smallest item): policy actions are rejected and treated as SKIP.
- [ ] With moderate budget: writes succeed until budget exhausted; subsequent writes rejected.
- [ ] EXPIRE credits budget (bytes_used decreases, remaining increases).

## 4) Baselines
- [ ] No-memory baseline writes nothing.
- [ ] FIFO store-all writes until full, then evicts oldest to admit new.
- [ ] Last-KB keeps the most recent items within budget.

## 5) Runner end-to-end
- [ ] `python -m writepolicybench --episodes 2` runs without exceptions.
- [ ] Runner produces a small metrics summary (even if placeholder).
- [ ] Runner logs per-episode outcomes (success, bytes used, staleness flags) to an artifacts directory.

## 6) Reproducibility
- [ ] All runs record: seed, budget, policy name, code version (git commit hash).

## 7) Minimal evaluator sanity
- [ ] Success metric increases when budget increases (monotonic trend on synthetic tasks).
- [ ] Staleness rate decreases when time-aware policy is enabled (later milestone).
