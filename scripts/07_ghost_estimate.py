"""
Ghost-list estimate: fault-delta adaptation WITHOUT a shadow engine (deployable).
Compares the oracle decider with the ghost decider, and measures how well the
ghost estimate tracks the true fault delta per epoch.
"""
import _path  # noqa: F401
import os
import numpy as np, pandas as pd
from paging import config
from paging.traces import load_all
from paging.engine import MemoryEngine, LRU
from paging.predictors import make_predictor
from paging.deciders import CostAwareFaultDelta, CostAwareGhost
from paging.simulate import run_prefetch_sim_ghost

config.ensure_dirs()
TRACES = load_all(limit=config.TRACE_LIMIT)

rows, logs = [], []
for tname, nf in config.ABLATION_CONFIGS:
    if tname not in TRACES:
        print(f'skipping {tname}'); continue
    t = TRACES[tname]
    e = MemoryEngine(nf, LRU(pages=t.pages))
    for i, p, _ in t.stream():
        e.access(p, i)
    lru_ref = e.n_fault

    for label, D in [('oracle (shadow)', CostAwareFaultDelta), ('ghost estimate', CostAwareGhost)]:
        s = run_prefetch_sim_ghost(t, nf, LRU, make_predictor('hybrid'), D())
        log = np.array(s['epoch_log']) if s['epoch_log'] else np.zeros((0, 2))
        corr = (np.corrcoef(log[:, 0], log[:, 1])[0, 1]
                if len(log) > 2 and log[:, 0].std() > 0 and log[:, 1].std() > 0 else np.nan)
        if label == 'ghost estimate' and len(log):
            logs.append(log)
        rows.append({'trace': tname, 'frames': nf, 'decider': label, 'faults': s['faults'],
                     'vs_LRU': round((lru_ref - s['faults']) / lru_ref, 4),
                     'pf_used': s['pf_used'], 'pf_harm': s['pf_harm'],
                     'final_thr': s['final_thr'],
                     'est_true_corr': None if np.isnan(corr) else round(corr, 3)})
    print(f'{tname:18s} frames={nf:3d} done')

g = pd.DataFrame(rows)
g.to_csv(os.path.join(config.RESULTS_DIR, 'ghost_estimate_results.csv'), index=False)
pd.set_option('display.width', 200)

cmp = g.pivot_table(index=['trace', 'frames'], columns='decider', values='vs_LRU')
cmp['gap'] = (cmp['ghost estimate'] - cmp['oracle (shadow)']).round(4)
print('\n=== oracle vs deployable estimate (fault reduction vs LRU) ===')
print(cmp)
print('\n=== summary ===')
print(g.groupby('decider').agg(mean_gain=('vs_LRU', 'mean'), worst=('vs_LRU', 'min'),
                               best=('vs_LRU', 'max')).round(4))
print('\n=== estimate vs true delta, per epoch ===')
print(g[g.decider == 'ghost estimate'][['trace', 'frames', 'pf_used', 'pf_harm', 'est_true_corr']]
      .to_string(index=False))
if logs:
    pooled = np.vstack(logs)
    if pooled[:, 0].std() > 0 and pooled[:, 1].std() > 0:
        print(f'\npooled correlation: {np.corrcoef(pooled[:, 0], pooled[:, 1])[0, 1]:.3f} '
              f'({len(pooled)} epochs)')
