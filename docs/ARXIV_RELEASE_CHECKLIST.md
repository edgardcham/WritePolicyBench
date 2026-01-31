# WritePolicyBench — arXiv / public release checklist

This is a practical pre-flight checklist to make the repo + paper “public-ready”.

## 0) Decide the public surface
- [ ] Choose the canonical repo URL (update paper to the real URL).
- [ ] Choose a release tag name (e.g., `v1.0-paper` or `arxiv-2026-01`).

## 1) Licensing (do not skip)
- [ ] Pick a license (MIT / Apache-2.0 / BSD-3 / etc.).
- [ ] Add `LICENSE` at repo root.
- [ ] Ensure all third-party code/data are compatible.

## 2) Reproducibility artifacts
- [ ] Confirm `data/episodes/MANIFEST.json` is included and referenced in docs.
- [ ] Confirm frozen episodes for all regimes used in the paper exist (including `burst_redundancy`).
- [ ] Confirm scripts can regenerate:
  - `artifacts/results.csv`
  - `paper/figures/*.png`
  - `paper/sections/results_table_*.tex`

## 3) One-command reproduction
- [ ] Add a single command sequence to `README.md` (or `docs/RUNBOOK.md`) that reproduces the paper:
  - create venv
  - install deps
  - freeze episodes
  - run evaluator
  - generate plots
  - generate tables
  - build PDF

## 4) Paper placeholders
- [x] Paper commit hash updated to `e37506c`.
- [ ] Replace `https://github.com/ORG/REPO` with real URL.

## 5) Sanity checks
- [ ] `python3 -m writepolicybench.evaluator` runs clean.
- [ ] `python3 -m scripts.plot_results` produces all expected figures.
- [ ] `python3 -m scripts.make_paper_tables` regenerates tables.
- [ ] `cd paper && latexmk -pdf main.tex` builds clean.

## 6) Tag + archive
- [ ] `git tag -a <tag> -m "paper release"` on the exact commit used by arXiv.
- [ ] Create a GitHub release (or equivalent) and attach a source zip.

## 7) Optional (nice-to-have)
- [ ] Add a minimal CI workflow to run unit tests + build paper.
- [ ] Add a `CITATION.cff`.
