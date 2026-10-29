"""
Phase B: does the adopted design hold on other segments / benchmarks / memory sizes?
NO retuning: the design (readahead + Markov, delta tier, top-2) was chosen on the warm-up of the
original four traces. Memory = ratio x working set (>= 8 frames). Resumable.

  python scripts/11_robustness.py [--ratios 0.01 0.05]
"""
import _path  # noqa: F401
import argparse, os
import pandas as pd
from paging import config
from paging.traces import load_trace
from paging.engine import LRU, ARC, OPTFast, run_policy
from paging.predictors import make_predictor, make_readahead, StreamMarkovPredictor, TopN
from paging.deciders import AlwaysPrefetch, CostAwareGhost, CostAwareDirectory
from paging.simulate import run_prefetch_sim, run_prefetch_sim_ghost, run_prefetch_sim_dir
from paging.final import run_final

NAMES = ['410.bwaves-s0', '410.bwaves-s1', '410.bwaves-s2', '437.leslie3d-s0', '437.leslie3d-s1', '437.leslie3d-s2',
         '433.milc-s0', '462.libquantum-s0', '429.mcf-s0', 'bfs-3', 'bfs-10', 'pr-3', 'pr-10',
         # Phase 17 additions
         '450.soplex-s0', '471.omnetpp-s0', '473.astar-s0', '459.GemsFDTD-s0', '470.lbm-s0', '482.sphinx3-s0',
         'bc-3', 'cc-5', 'sssp-3']
ap = argparse.ArgumentParser()
ap.add_argument('--ratios', nargs='*', type=float, default=[0.01, 0.05])
args = ap.parse_args()
config.ensure_dirs()


out = os.path.join(config.RESULTS_DIR, 'robustness.csv')
done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
rows = done.to_dict('records')
have = {(r['trace'], r['ratio']) for r in rows}
import numpy as np
for name in NAMES:
    path = os.path.join(config.TRACE_DIR, f'{name}.csv')
    if not os.path.exists(path):
        print('skipping', name, '(trace missing)'); continue
    t = load_trace(name, limit=config.TRACE_LIMIT)
    ws = len(np.unique(t.pages))
    for ratio in args.ratios:
        if (name, ratio) in have:
            continue
        f = max(8, int(ws * ratio))
        r = dict(trace=name, ratio=ratio, frames=f, ws=ws,
                 LRU=run_policy(t, f, LRU)['faults'], ARC=run_policy(t, f, ARC)['faults'],
                 OPT=run_policy(t, f, OPTFast)['faults'])
        r['RA'] = run_prefetch_sim(t, f, LRU, make_readahead(f), AlwaysPrefetch())['faults']
        r['Markov'] = run_prefetch_sim_ghost(t, f, LRU, make_predictor('hybrid'), CostAwareGhost())['faults']
        mk = lambda: make_predictor('delta-hybrid')
        r['Comb_d'] = run_prefetch_sim(t, f, LRU, StreamMarkovPredictor(f, markov=mk()), AlwaysPrefetch())['faults']
        r['Comb_d_t2'] = run_prefetch_sim(t, f, LRU, StreamMarkovPredictor(f, markov=TopN(mk(), 2)), AlwaysPrefetch())['faults']
        r['Comb_d_gated'] = run_prefetch_sim_ghost(
            t, f, LRU, StreamMarkovPredictor(f, markov=mk(), readahead=make_readahead(f, conf_k=1)), CostAwareGhost())['faults']
        r['Comb_d_dir'] = run_prefetch_sim_dir(       # Phase 14: shadow-directory feedback instead of the ghost list
            t, f, LRU, StreamMarkovPredictor(f, markov=mk(), readahead=make_readahead(f, conf_k=1)), CostAwareDirectory())['faults']
        r['Final'] = run_final(t, f)['faults']          # Phase 16: frozen controller (paging.final)
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print({k: v for k, v in r.items()}, flush=True)

df = pd.DataFrame(rows)
for c in ('ARC', 'RA', 'Markov', 'Comb_d', 'Comb_d_t2', 'Comb_d_gated', 'Comb_d_dir', 'Final'):
    df['g_' + c] = (df['LRU'] - df[c]) / df['LRU']          # vs_LRU: positive = better
print('\nvs_LRU (positive = fewer faults than LRU)')
print(df[['trace', 'ratio', 'frames', 'LRU', 'OPT'] + [c for c in df if c.startswith('g_')]].round(3).to_string(index=False))
for c in ('RA', 'Markov', 'Comb_d', 'Comb_d_t2', 'Comb_d_gated', 'Comb_d_dir', 'Final'):
    print(f"{c:10s} mean {df['g_'+c].mean():+.3f}  worst {df['g_'+c].min():+.3f}  best {df['g_'+c].max():+.3f}  worse-than-LRU in {(df['g_'+c] < 0).sum()}/{len(df)}")
