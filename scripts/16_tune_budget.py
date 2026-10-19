"""
Phase 17 follow-up: the prefetch budget cap (prefetches <= budget_ratio x accesses) blocked the final system
on omnetpp and cc-5 while it was winning every epoch. Choose budget_ratio on the WARM-UP PORTION ONLY,
same rule as script 15 (worst case within TOL of the current setting, highest mean, keep current unless
beaten by > TOL). 1.01 = no cap. Writes results/tuning_budget.csv.   python scripts/16_tune_budget.py
"""
import _path  # noqa: F401
import os
import numpy as np
import pandas as pd
from paging import config
from paging.traces import load_trace, Trace
from paging.engine import LRU, run_policy
from paging.final import make_final_predictor
from paging.deciders import CostAwareDirectory
from paging.simulate import run_prefetch_sim_dir

NAMES = ['clustered', '410.bwaves-s0', '410.bwaves-s1', '410.bwaves-s2', '437.leslie3d-s0', '437.leslie3d-s1', '437.leslie3d-s2',
         '433.milc-s0', '462.libquantum-s0', '429.mcf-s0', 'bfs-3', 'bfs-10', 'pr-3', 'pr-10', '450.soplex-s0', '471.omnetpp-s0',
         '473.astar-s0', '459.GemsFDTD-s0', '470.lbm-s0', '482.sphinx3-s0', 'bc-3', 'cc-5', 'sssp-3']
BUDGETS, CURRENT, TOL = (0.35, 0.5, 0.75, 1.01), 0.35, 0.002
rows = []
for name in NAMES:
    if not os.path.exists(os.path.join(config.TRACE_DIR, f'{name}.csv')):
        print('skipping', name); continue
    full = load_trace(name, limit=config.TRACE_LIMIT)
    w = Trace(name, full.pages[:full.split])
    f = max(8, int(len(np.unique(full.pages)) * 0.01))
    base = run_policy(w, f, LRU)['faults']
    for b in BUDGETS:
        r = run_prefetch_sim_dir(w, f, LRU, make_final_predictor(f), CostAwareDirectory(budget_ratio=b, **config.CONTROLLER_FROZEN))
        rows.append(dict(trace=name, frames=f, budget=b, LRU=base, faults=r['faults'], prefetches=r['prefetches'],
                         vs_LRU=(base - r['faults']) / base))
    print(name, 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(config.RESULTS_DIR, 'tuning_budget.csv'), index=False)
print(df.pivot(index='trace', columns='budget', values='vs_LRU').round(4).to_string())
s = df.groupby('budget').vs_LRU.agg(mean='mean', worst='min')
cur = s.loc[CURRENT]
ok = s[s.worst >= cur.worst - TOL]
best = ok['mean'].idxmax()
chosen = best if ok.loc[best, 'mean'] > cur['mean'] + TOL else CURRENT
print(s.round(4).to_string()); print('CHOSEN budget_ratio =', chosen)
