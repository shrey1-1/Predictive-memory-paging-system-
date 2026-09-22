"""Stage 5b — head-to-head: traditional algorithms vs the predictive system."""
import _path  # noqa: F401
import os
import pandas as pd
from paging import config
from paging.traces import load_all
from paging.engine import MemoryEngine, FIFO, LRU, Clock, OPTFast
from paging.predictors import make_predictor
from paging.deciders import CostAwareFaultDelta
from paging.simulate import run_prefetch_sim

config.ensure_dirs()
TRACES = load_all(limit=config.TRACE_LIMIT)

H2H_CONFIGS = [
    ('437.leslie3d-s0', 21),
    ('410.bwaves-s0',   30),
    ('clustered',        9),
    ('bfs-3',           20),
    ('pr-3',            15),
]
BASE_POLICIES = {'FIFO': FIFO, 'LRU': LRU, 'Clock': Clock}

rows = []
for tname, nf in H2H_CONFIGS:
    if tname not in TRACES:
        print(f'skipping {tname} — trace not available')
        continue
    t = TRACES[tname]
    r = {'trace': tname, 'frames': nf}

    # traditional algorithms, no prediction
    for pname, P in {**BASE_POLICIES, 'OPT': OPTFast}.items():
        e = MemoryEngine(nf, P(pages=t.pages))
        for i, p, _ in t.stream():
            e.access(p, i)
        r[pname] = e.n_fault

    # proposed system on top of each practical eviction policy
    for pname, P in BASE_POLICIES.items():
        s = run_prefetch_sim(t, nf, P, make_predictor('hybrid'), CostAwareFaultDelta())
        r[f'{pname}+pred'] = s['faults']

    rows.append(r)
    print(f'{tname:18s} frames={nf:3d} done')

if not rows:
    raise SystemExit('no traces available — run 01/02 first')

h2h = pd.DataFrame(rows).set_index(['trace', 'frames'])
h2h.to_csv(os.path.join(config.RESULTS_DIR, 'head_to_head.csv'))
pd.set_option('display.width', 200)

cols = ['FIFO', 'Clock', 'LRU', 'FIFO+pred', 'Clock+pred', 'LRU+pred', 'OPT']
print('\n=== page faults (lower = better) ===')
print(h2h[cols])

trad = h2h[['FIFO', 'LRU', 'Clock']]
red = pd.DataFrame(index=h2h.index)
for base in BASE_POLICIES:
    red[f'vs {base}'] = ((h2h[base] - h2h['LRU+pred']) / h2h[base]).round(4)
red['vs best traditional'] = ((trad.min(axis=1) - h2h['LRU+pred']) / trad.min(axis=1)).round(4)
print('\n=== fault reduction of LRU+pred vs each algorithm ===')
print(red)

gain = pd.DataFrame(index=h2h.index)
for base in BASE_POLICIES:
    gain[base] = ((h2h[base] - h2h[f'{base}+pred']) / h2h[base]).round(4)
gain.to_csv(os.path.join(config.RESULTS_DIR, 'prediction_gain_per_policy.csv'))
print('\n=== gain from adding prediction, per eviction policy ===')
print(gain)

print('\n=== rank per config (1 = fewest faults) ===')
print(h2h[cols].rank(axis=1, method='min').astype(int))
print(f'\nresults written to {config.RESULTS_DIR}')