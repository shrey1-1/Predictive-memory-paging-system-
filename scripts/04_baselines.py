"""Stage 4 — FIFO / LRU / Clock / OPT baselines + correctness verification."""
import _path  # noqa: F401
import os
import numpy as np, pandas as pd
from paging import config
from paging.traces import load_all
from paging.engine import POLICIES, run_policy

config.ensure_dirs()
TRACES = load_all(limit=config.TRACE_LIMIT)
RATIOS = [0.01, 0.02, 0.05, 0.10, 0.25]      # fraction of working set

rows = []
for name, t in TRACES.items():
    ws = len(np.unique(t.pages))
    for r in RATIOS:
        nf = max(8, int(ws * r))
        for pname, P in POLICIES.items():
            s = run_policy(t, nf, P)
            rows.append({'trace': name, 'ws': ws, 'ratio': r, 'frames': nf, 'policy': pname,
                         'faults': s['faults'], 'fault_rate': round(s['fault_rate'], 4)})
        print(f'{name:18s} ratio={r:.2f} frames={nf:5d} done')

base = pd.DataFrame(rows).drop_duplicates(['trace', 'frames', 'policy'])
base.to_csv(os.path.join(config.RESULTS_DIR, 'baseline_results.csv'), index=False)
print(base.pivot_table(index=['trace', 'frames'], columns='policy',
                       values='fault_rate')[['FIFO', 'LRU', 'Clock', 'OPT']])

# --- verification ---
piv = base.pivot_table(index=['trace', 'frames'], columns='policy', values='faults')
fails = []
for pol in ('FIFO', 'LRU', 'Clock'):
    n = int((piv[pol] < piv['OPT']).sum())
    if n: fails.append(f'{pol} beats OPT on {n} configs')
for name in TRACES:
    for pol in ('LRU', 'OPT'):
        sub = base[(base.trace == name) & (base.policy == pol)].sort_values('frames')
        if not sub.faults.is_monotonic_decreasing:
            fails.append(f'{pol} non-monotonic on {name}')
if ((base.fault_rate < 0) | (base.fault_rate > 1)).any():
    fails.append('fault_rate out of range')

print('\nVERIFICATION:', 'FAILED' if fails else 'all checks passed')
for f in fails: print(' -', f)
if fails: raise SystemExit(1)
