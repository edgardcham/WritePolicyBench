from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import statistics


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def fmt(x: float) -> str:
    return f"{x:.3f}"


def make_table(rows: list[dict[str, str]], *, metric: str, mode: str) -> str:
    policies = sorted({r["policy"] for r in rows if r.get("mode", "default") == mode})
    budgets = sorted({int(r["budget_bytes"]) for r in rows if r.get("mode", "default") == mode})

    series: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("mode", "default") != mode:
            continue
        series[(r["policy"], int(r["budget_bytes"]))].append(float(r.get(metric, 0.0) or 0.0))

    out: list[str] = []
    out.append(f"## Mode: {mode} ({metric})")
    out.append("| policy | " + " | ".join(str(b) for b in budgets) + " |")
    out.append("|---|" + "|".join(["---"] * len(budgets)) + "|")

    for p in policies:
        vals = [fmt(mean(series[(p, b)])) for b in budgets]
        out.append("| " + p + " | " + " | ".join(vals) + " |")

    out.append("")
    return "\n".join(out)


def main() -> int:
    results_path = Path("artifacts/results.csv")
    if not results_path.exists():
        raise SystemExit("artifacts/results.csv missing; run python -m writepolicybench.evaluator")

    rows = load_rows(results_path)
    modes = sorted({r.get("mode", "default") for r in rows})

    metrics = [
        ("f1", "Mean F1"),
        ("regret_write_only", "Mean regret (WRITE-only oracle gap; lower is better)"),
    ]

    out_lines: list[str] = []
    out_lines.append("# Results summary")

    for metric, title in metrics:
        out_lines.append(f"\n## {title}")
        for mode in modes:
            out_lines.append(make_table(rows, metric=metric, mode=mode))

    out_path = Path("artifacts/RESULTS_SUMMARY.md")
    out_path.write_text("\n".join(out_lines).strip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
