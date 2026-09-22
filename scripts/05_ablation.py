"""Stage 5 — decision-engine ablation (the main result)."""
import _path  # noqa: F401
import os
import numpy as np, pandas as pd
from paging import config
from paging.traces import load_all
from paging.engine import MemoryEngine, LRU, OPTFast
from paging.predictors import make_predictor
from paging.deciders import DECIDERS
from paging.simulate import run_prefetch_sim

config.ensure_dirs()
TRACES = load_all(limit=config.TRACE_LIMIT)
EXTRA = ('final_thr', 'thr_min', 'thr_max', 'thr_moves', 'mean_delta_rate',
         'epochs_winning', 'epochs_losing', 'approved', 'd_budget', 'd_cost',
         'd_thr', 'shadow_faults')

configs = [(t, f) for t, f in config.ABLATION_CONFIGS if t in TRACES]
skipped = [t for t, _ in config.ABLATION_CONFIGS if t not in TRACES]
if skipped:
    print(f'NOTE: skipping configs for missing traces: {sorted(set(skipped))}')

rows = []
for tname, nf in configs:
    t = TRACES[tname]
    e = MemoryEngine(nf, LRU(pages=t.pages))
    for i, p, _ in t.stream(): e.access(p, i)
    lru_ref = e.n_fault
    e = MemoryEngine(nf, OPTFast(pages=t.pages))
    for i, p, _ in t.stream(): e.access(p, i)
    opt_ref = e.n_fault
    headroom = (lru_ref - opt_ref) / max(1, lru_ref)

    for mk in DECIDERS:
        s = run_prefetch_sim(t, nf, LRU, make_predictor('hybrid'), mk())
        gain = (lru_ref - s['faults']) / max(1, lru_ref)
        row = {'trace': tname, 'frames': nf, 'decider': s['decider'], 'faults': s['faults'],
               'vs_LRU': round(gain, 4), 'headroom': round(headroom, 4),
               'prefetches': s['prefetches'], 'pf_used': s['pf_used'],
               'pf_wasted': s['pf_wasted'], 'pf_precision': round(s['pf_precision'], 3),
               'evictions': s['evictions'], 'LRU_ref': lru_ref, 'OPT_ref': opt_ref}
        for k in EXTRA: row[k] = s.get(k)
        rows.append(row)
    print(f'{tname:18s} frames={nf:3d} done')

abl = pd.DataFrame(rows)
base_ev = abl[abl.decider == 'no-prefetch'].set_index(['trace', 'frames'])['evictions']
abl['evict_vs_base'] = abl.apply(lambda r: round(r.evictions / base_ev.loc[(r.trace, r.frames)], 3), axis=1)
abl.to_csv(os.path.join(config.RESULTS_DIR, 'ablation_results.csv'), index=False)

pd.set_option('display.width', 200)
print('\n=== fault reduction vs LRU (higher = better) ===')
print(abl.pivot_table(index=['trace', 'frames'], columns='decider', values='vs_LRU'))

sub = abl[abl.decider != 'no-prefetch']
summary = sub.groupby('decider').agg(
    mean_gain=('vs_LRU', 'mean'), best_case=('vs_LRU', 'max'), worst_case=('vs_LRU', 'min'),
    total_harm=('vs_LRU', lambda x: x[x < 0].sum()),
    total_gain=('vs_LRU', lambda x: x[x > 0].sum())).round(4).sort_values('worst_case', ascending=False)
summary.to_csv(os.path.join(config.RESULTS_DIR, 'decider_summary.csv'))
print('\n=== decider summary (worst_case closer to 0 = safer) ===')
print(summary)

fd = abl[abl.decider == 'cost-aware-fd'][['trace', 'frames', 'final_thr', 'epochs_winning',
                                          'epochs_losing', 'vs_LRU']]
print('\n=== fault-delta controller ===')
print(fd.to_string(index=False))
c = sub[['pf_precision', 'vs_LRU']].corr().iloc[0, 1]
print(f'\ncorrelation(precision, fault reduction) = {c:.3f}')
print(f'\nresults written to {config.RESULTS_DIR}')
