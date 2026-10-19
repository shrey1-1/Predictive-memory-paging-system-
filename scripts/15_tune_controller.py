"""
Phase 16: tune the fault-delta controller on the WARM-UP PORTION ONLY (first 25% of each trace).
System: RA + delta-Markov (conf_k=1) with shadow-directory feedback, LRU underneath, RAM = 1% of working set.
Grid: epoch x step x dead band (start threshold stays 0.5). Selection rule, fixed before the run:
  1. drop settings whose worst case is more than TOL below the default's worst case
  2. among the rest take the highest mean vs_LRU
  3. keep the default unless that setting beats it by more than TOL (prefer the simplest = unchanged)
Resumable. Writes results/tuning_controller.csv.   python scripts/15_tune_controller.py
"""
import _path  # noqa: F401
import itertools, os
import numpy as np
import pandas as pd
from paging import config
from paging.traces import load_trace, Trace
from paging.engine import LRU, run_policy
from paging.predictors import make_predictor, make_readahead, StreamMarkovPredictor
from paging.deciders import CostAwareDirectory
from paging.simulate import run_prefetch_sim_dir

NAMES = ['clustered', '410.bwaves-s0', '410.bwaves-s1', '410.bwaves-s2', '437.leslie3d-s0', '437.leslie3d-s1', '437.leslie3d-s2',
         '433.milc-s0', '462.libquantum-s0', '429.mcf-s0', 'bfs-3', 'bfs-10', 'pr-3', 'pr-10']
EPOCHS, STEPS, BANDS = (500, 1000, 2000, 4000), (0.05, 0.1, 0.2), (0.001, 0.002, 0.005)
DEFAULT = (2000, 0.05, 0.002)
TOL = 0.002
out = os.path.join(config.RESULTS_DIR, 'tuning_controller.csv')
rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
have = {(r['trace'], r['epoch'], r['step'], r['dead_band']) for r in rows}
for name in NAMES:
    if not os.path.exists(os.path.join(config.TRACE_DIR, f'{name}.csv')):
        print('skipping', name); continue
    full = load_trace(name, limit=config.TRACE_LIMIT)
    w = Trace(name, full.pages[:full.split])                  # warm-up portion only
    f = max(8, int(len(np.unique(full.pages)) * 0.01))
    base = run_policy(w, f, LRU)['faults']
    for ep, st, db in itertools.product(EPOCHS, STEPS, BANDS):
        if (name, ep, st, db) in have:
            continue
        p = StreamMarkovPredictor(f, markov=make_predictor('delta-hybrid'), readahead=make_readahead(f, conf_k=1))
        r = run_prefetch_sim_dir(w, f, LRU, p, CostAwareDirectory(epoch=ep, step=st, dead_band=db))
        rows.append(dict(trace=name, frames=f, epoch=ep, step=st, dead_band=db, LRU=base, faults=r['faults'],
                         vs_LRU=(base - r['faults']) / base))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(name, 'done', flush=True)

df = pd.DataFrame(rows)
s = df.groupby(['epoch', 'step', 'dead_band']).vs_LRU.agg(mean='mean', worst='min').reset_index()
d = s[(s.epoch == DEFAULT[0]) & (s.step == DEFAULT[1]) & (s.dead_band == DEFAULT[2])].iloc[0]
ok = s[s.worst >= d.worst - TOL]
best = ok.sort_values('mean', ascending=False).iloc[0]
chosen = best if best['mean'] > d['mean'] + TOL else d
print(s.sort_values('mean', ascending=False).round(4).to_string(index=False))
print(f"\ndefault  epoch={int(d.epoch)} step={d.step} band={d.dead_band}: mean {d['mean']:+.4f} worst {d.worst:+.4f}")
print(f"best ok  epoch={int(best.epoch)} step={best.step} band={best.dead_band}: mean {best['mean']:+.4f} worst {best.worst:+.4f}")
print(f"CHOSEN   epoch={int(chosen.epoch)} step={chosen.step} band={chosen.dead_band}")
