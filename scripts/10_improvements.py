"""
Improvement experiments (Phase 9/10 items), full-trace evaluation of the adopted design.
Every design choice here was made on the 25% warm-up portion only (see docs/RESULTS.md section 9).

  python scripts/10_improvements.py
Columns (faults, lower is better):
  base        eviction policy alone          RA        readahead (+AlwaysPrefetch)
  Mk+gh       Markov hybrid + ghost decider  Mk+d+t2   delta-Markov hybrid, top-2, ghost decider
  Comb        readahead + Markov hybrid      Comb+d+t2 readahead + delta-Markov hybrid, top-2
Comb* variants use AlwaysPrefetch: readahead candidates carry confidence 1.0, so a threshold decider cannot gate them.
"""
import _path  # noqa: F401
import os
import pandas as pd
from paging import config
from paging.traces import load_trace
from paging.engine import LRU, ARC, OPTFast, run_policy
from paging.predictors import make_predictor, make_readahead, StreamMarkovPredictor, TopN
from paging.deciders import AlwaysPrefetch, CostAwareGhost
from paging.simulate import run_prefetch_sim, run_prefetch_sim_ghost

CONFIGS = [('clustered', 9), ('437.leslie3d-s0', 21), ('410.bwaves-s0', 30), ('bfs-3', 20), ('pr-3', 15)]
config.ensure_dirs()
rows = []
for name, f in CONFIGS:
    path = os.path.join(config.TRACE_DIR, f'{name}.csv')
    if not os.path.exists(path):
        print('skipping', name, '(trace missing)'); continue
    t = load_trace(name, limit=config.TRACE_LIMIT)
    opt = run_policy(t, f, OPTFast)['faults']
    for P in (LRU, ARC):
        d2 = lambda: TopN(make_predictor('delta-hybrid'), 2)
        rows.append(dict(
            trace=name, frames=f, policy=P.__name__,
            base=run_policy(t, f, P)['faults'],
            RA=run_prefetch_sim(t, f, P, make_readahead(f), AlwaysPrefetch())['faults'],
            Mk_gh=run_prefetch_sim_ghost(t, f, P, make_predictor('hybrid'), CostAwareGhost())['faults'],
            Mk_d_t2=run_prefetch_sim_ghost(t, f, P, d2(), CostAwareGhost())['faults'],
            Comb=run_prefetch_sim(t, f, P, StreamMarkovPredictor(f), AlwaysPrefetch())['faults'],
            Comb_d_t2=run_prefetch_sim(t, f, P, StreamMarkovPredictor(f, markov=d2()), AlwaysPrefetch())['faults'],
            OPT=opt))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df['best_vs_base'] = 1 - df[['RA', 'Mk_gh', 'Mk_d_t2', 'Comb', 'Comb_d_t2']].min(axis=1) / df['base']
df.to_csv(os.path.join(config.RESULTS_DIR, 'improvements.csv'), index=False)
print(df.to_string(index=False))
