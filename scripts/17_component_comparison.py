"""
Side-by-side comparison of every predictor and every decision engine, one component changed at a time.

  Predictors (6): stride, markov-o1, markov-o2, hybrid, delta-hybrid, readahead (+ the final combination)
      all run through the SAME decision layer (the final one: shadow directory + frozen controller)
  Deciders (9): no-prefetch, always, fixed-0.3/0.6/0.85, cost-aware (precision), fault-delta oracle,
      ghost list, shadow directory (+ the final directory decider with the frozen controller)
      all fed the SAME predictions (once from the original Markov hybrid, once from the final predictor);
      the three fault-delta variants share the default controller so only the feedback signal differs.

Traces: every real trace present + synthetic clustered, RAM = 1% of working set, LRU underneath.
Metrics: faults vs LRU, precision, prefetches per access, estimated speedup (src/paging/timing.py,
1 us compute per access, 0.2 us prediction cost). Resumable.

  python scripts/17_component_comparison.py
"""
import _path  # noqa: F401
import os
import numpy as np
import pandas as pd
from paging import config
from paging.traces import load_trace
from paging.engine import LRU
from paging.predictors import make_predictor, make_readahead
from paging.deciders import (NoPrefetch, AlwaysPrefetch, FixedThreshold, CostAware, CostAwareFaultDelta,
                             CostAwareGhost, CostAwareDirectory)
from paging.simulate import run_prefetch_sim, run_prefetch_sim_ghost, run_prefetch_sim_dir, eval_predictor
from paging.final import make_final_predictor, make_final_decider
from paging.timing import replay, DEVICES, T_PRED

NAMES = ['clustered', '410.bwaves-s0', '410.bwaves-s1', '410.bwaves-s2', '437.leslie3d-s0', '437.leslie3d-s1',
         '437.leslie3d-s2', '433.milc-s0', '462.libquantum-s0', '429.mcf-s0', 'bfs-3', 'bfs-10', 'pr-3', 'pr-10',
         '450.soplex-s0', '471.omnetpp-s0', '473.astar-s0', '459.GemsFDTD-s0', '470.lbm-s0', '482.sphinx3-s0',
         'bc-3', 'cc-5', 'sssp-3']
T_CPU = 1e-6

PREDICTORS = {
    'stride':        lambda f: make_predictor('stride'),
    'markov-o1':     lambda f: make_predictor('markov-o1'),
    'markov-o2':     lambda f: make_predictor('markov-o2'),
    'hybrid':        lambda f: make_predictor('hybrid'),
    'delta-hybrid':  lambda f: make_predictor('delta-hybrid'),
    'readahead':     lambda f: make_readahead(f, conf_k=1),
    'final (readahead + delta-hybrid)': lambda f: make_final_predictor(f),
}
DECIDERS = {   # name -> (simulator, decider factory)
    'no-prefetch':         (run_prefetch_sim, NoPrefetch),
    'always':              (run_prefetch_sim, AlwaysPrefetch),
    'fixed-0.3':           (run_prefetch_sim, lambda: FixedThreshold(0.3)),
    'fixed-0.6':           (run_prefetch_sim, lambda: FixedThreshold(0.6)),
    'fixed-0.85':          (run_prefetch_sim, lambda: FixedThreshold(0.85)),
    'precision (cost-aware)': (run_prefetch_sim, CostAware),
    'fault-delta oracle':  (run_prefetch_sim, CostAwareFaultDelta),
    'ghost list':          (run_prefetch_sim_ghost, CostAwareGhost),
    'shadow directory':    (run_prefetch_sim_dir, CostAwareDirectory),
    'directory + frozen controller (final)': (run_prefetch_sim_dir, make_final_decider),
}
DECIDER_PREDICTORS = {'Markov hybrid (original)': lambda f: make_predictor('hybrid'),
                      'final predictor': lambda f: make_final_predictor(f)}


def timed(sim, t, f, pred, dec):
    log = []
    r = sim(t, f, LRU, pred, dec, log=log)
    sp = {}
    for dev, d in DEVICES.items():
        sp[dev] = replay(log, r['accesses'], d['t_io'], d['channels'], T_CPU, T_PRED)['time']
    return r, sp


config.ensure_dirs()
out_p = os.path.join(config.RESULTS_DIR, 'component_predictors.csv')
out_d = os.path.join(config.RESULTS_DIR, 'component_deciders.csv')
rp = pd.read_csv(out_p).to_dict('records') if os.path.exists(out_p) else []
rd = pd.read_csv(out_d).to_dict('records') if os.path.exists(out_d) else []
done = {r['trace'] for r in rp} & {r['trace'] for r in rd}

for name in NAMES:
    if not os.path.exists(os.path.join(config.TRACE_DIR, f'{name}.csv')):
        print('skipping', name, '(trace missing)'); continue
    if name in done:
        continue
    t = load_trace(name, limit=config.TRACE_LIMIT)
    f = max(8, int(len(np.unique(t.pages)) * 0.01))
    base, bsp = timed(run_prefetch_sim, t, f, make_predictor('hybrid'), NoPrefetch())
    lru = base['faults']
    rp = [r for r in rp if r['trace'] != name]
    rd = [r for r in rd if r['trace'] != name]
    for pn, mk in PREDICTORS.items():
        q = eval_predictor(t, mk(f))                         # prediction quality, top-1 guess
        r, sp = timed(run_prefetch_sim_dir, t, f, mk(f), make_final_decider())
        rp.append(dict(trace=name, frames=f, predictor=pn, coverage=q['coverage'], accuracy=q['accuracy'],
                       useful=q['useful'], LRU=lru, faults=r['faults'], vs_LRU=(lru - r['faults']) / lru,
                       prefetches=r['prefetches'], precision=r['pf_precision'],
                       **{f'speedup_{d}': bsp[d] / sp[d] for d in DEVICES}))
    for pl, mkp in DECIDER_PREDICTORS.items():
        for dn, (sim, mkd) in DECIDERS.items():
            r, sp = timed(sim, t, f, mkp(f), mkd())
            rd.append(dict(trace=name, frames=f, predictions=pl, decider=dn, LRU=lru, faults=r['faults'],
                           vs_LRU=(lru - r['faults']) / lru, prefetches=r['prefetches'], precision=r['pf_precision'],
                           **{f'speedup_{d}': bsp[d] / sp[d] for d in DEVICES}))
    pd.DataFrame(rp).to_csv(out_p, index=False)
    pd.DataFrame(rd).to_csv(out_d, index=False)
    print(name, 'done', flush=True)


def summarise(df, key):
    real = df[df.trace != 'clustered']
    g = real.groupby(key, sort=False)
    s = pd.DataFrame({
        'mean_vs_LRU': g.vs_LRU.mean(), 'median_vs_LRU': g.vs_LRU.median(), 'worst_vs_LRU': g.vs_LRU.min(),
        'worse_than_LRU': g.vs_LRU.apply(lambda x: int((x < -0.0005).sum())),
        'prefetch_per_access': g.prefetches.mean() / 200_000,
        'precision': g.precision.mean(),
        'NVMe_speedup': g['speedup_NVMe SSD'].apply(lambda x: float(np.exp(np.log(x).mean()))),
        'HDD_speedup': g['speedup_HDD'].apply(lambda x: float(np.exp(np.log(x).mean()))),
        'clustered_vs_LRU': df[df.trace == 'clustered'].set_index(key).vs_LRU if 'clustered' in set(df.trace) else np.nan,
    })
    return s


P = pd.DataFrame(rp)
D = pd.DataFrame(rd)
sp = summarise(P, 'predictor')
q = P[P.trace != 'clustered'].groupby('predictor', sort=False)[['coverage', 'accuracy', 'useful']].mean()
sp = q.join(sp)
print(f'\nPREDICTORS (same final decider; {P.trace.nunique() - ("clustered" in set(P.trace))} real traces at 1%)')
print(sp.round(3).to_string())
for pl in DECIDER_PREDICTORS:
    print(f'\nDECIDERS fed by: {pl}')
    print(summarise(D[D.predictions == pl], 'decider').round(3).to_string())
sp.to_csv(os.path.join(config.RESULTS_DIR, 'component_predictors_summary.csv'))
pd.concat({pl: summarise(D[D.predictions == pl], 'decider') for pl in DECIDER_PREDICTORS}).to_csv(
    os.path.join(config.RESULTS_DIR, 'component_deciders_summary.csv'))
