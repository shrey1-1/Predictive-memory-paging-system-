"""
Phase C: break-even prefetch cost. Fault counts hide the I/O of prefetching, so ask:
how expensive may one prefetch be (as a fraction c of one fault's stall) before a system loses to plain LRU?
    stall(system) = faults + c * prefetches        stall(LRU) = faults_LRU
    break-even c* = (faults_LRU - faults) / prefetches      (no assumed latencies)
c = 0 : prefetch I/O fully hidden.   c = 1 : a prefetch costs as much as a fault.
c* >= 1 means it wins even if every prefetch stalls like a fault. Mean precision shown for context.
  python scripts/12_time_model.py
"""
import _path  # noqa: F401
import os
import numpy as np
import pandas as pd
from paging import config
from paging.traces import load_trace
from paging.engine import LRU, run_policy
from paging.predictors import make_predictor, make_readahead, StreamMarkovPredictor, TopN
from paging.deciders import AlwaysPrefetch, CostAwareGhost
from paging.simulate import run_prefetch_sim, run_prefetch_sim_ghost
from paging.final import run_final

NAMES = ['410.bwaves-s0', '410.bwaves-s1', '410.bwaves-s2', '437.leslie3d-s0', '437.leslie3d-s1', '437.leslie3d-s2',
         '433.milc-s0', '462.libquantum-s0', '429.mcf-s0', 'bfs-3', 'bfs-10', 'pr-3', 'pr-10',
         '450.soplex-s0', '471.omnetpp-s0', '473.astar-s0', '459.GemsFDTD-s0', '470.lbm-s0', '482.sphinx3-s0',
         'bc-3', 'cc-5', 'sssp-3']
RATIO = 0.01
config.ensure_dirs()
rows = []
for name in NAMES:
    if not os.path.exists(os.path.join(config.TRACE_DIR, f'{name}.csv')):
        print('skipping', name); continue
    t = load_trace(name, limit=config.TRACE_LIMIT)
    f = max(8, int(len(np.unique(t.pages)) * RATIO))
    base = run_policy(t, f, LRU)['faults']
    mk = lambda: make_predictor('delta-hybrid')
    systems = {
        'RA': lambda: run_prefetch_sim(t, f, LRU, make_readahead(f), AlwaysPrefetch()),
        'Markov': lambda: run_prefetch_sim_ghost(t, f, LRU, make_predictor('hybrid'), CostAwareGhost()),
        'Comb_d': lambda: run_prefetch_sim(t, f, LRU, StreamMarkovPredictor(f, markov=mk()), AlwaysPrefetch()),
        'Comb_d_t2': lambda: run_prefetch_sim(t, f, LRU, StreamMarkovPredictor(f, markov=TopN(mk(), 2)), AlwaysPrefetch()),
        'Final': lambda: run_final(t, f),                 # Phase 16/17 frozen final system
    }
    for s, run in systems.items():
        r = run()
        saved, pf = base - r['faults'], r['prefetches']
        rows.append(dict(trace=name, frames=f, system=s, LRU=base, faults=r['faults'], prefetches=pf,
                         precision=round(r['pf_precision'], 2),
                         breakeven_c=round(saved / pf, 2) if pf else float('nan')))
    print(name, 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(config.RESULTS_DIR, 'breakeven.csv'), index=False)
print(df.pivot(index='trace', columns='system', values='breakeven_c').to_string())
