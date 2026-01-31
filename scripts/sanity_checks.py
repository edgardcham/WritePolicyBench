from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import statistics


def mean(xs):
    return statistics.mean(xs) if xs else float('nan')


def main() -> int:
    path = Path('artifacts/results.csv')
    if not path.exists():
        raise SystemExit('Missing artifacts/results.csv; run evaluator first.')

    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
    budgets = sorted({int(r['budget_bytes']) for r in rows})
    modes = sorted({r.get('mode','default') for r in rows})

    # recall bounds for no_mem
    for mode in modes:
        vals = []
        for b in budgets:
            rs = [float(r['recall']) for r in rows if r['policy']=='no_mem' and r.get('mode','default')==mode and int(r['budget_bytes'])==b]
            vals.append(mean(rs))
        assert all(v == 0.0 for v in vals), f'no_mem recall not all-zero in mode={mode}: {vals}'

    # monotonicity-ish for last_kb recall
    for mode in modes:
        vals = []
        for b in budgets:
            rs = [float(r['recall']) for r in rows if r['policy']=='last_kb' and r.get('mode','default')==mode and int(r['budget_bytes'])==b]
            vals.append(mean(rs))
        for i in range(1, len(vals)):
            if vals[i] + 1e-9 < vals[i-1]:
                raise AssertionError(f'last_kb recall decreased with budget in mode={mode}: {list(zip(budgets, vals))}')

    print('sanity_ok')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
