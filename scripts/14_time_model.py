"""
Phase 15: estimated run time vs LRU, from an I/O timeline (src/paging/timing.py).
Fault counts hide prefetch I/O, late prefetches and queueing; this charges all three.
Device latencies are stated typical values (not tuned); compute time per traced access is unknown
for LLC traces, so it is swept. Speedup = time(LRU) / time(system); > 1 means faster than LRU.

  python scripts/14_time_model.py [--ratio 0.01]
"""
import _path  # noqa: F401
import argparse, os
import numpy as np
import pandas as pd
from paging import config
from paging.traces import load_trace
from paging.engine import LRU
from paging.predictors import make_predictor, make_readahead, StreamMarkovPredictor
from paging.deciders import AlwaysPrefetch, CostAwareGhost, CostAwareDirectory, NoPrefetch
from paging.simulate import run_prefetch_sim, run_prefetch_sim_ghost, run_prefetch_sim_dir
from paging.timing import replay, DEVICES, T_PRED, T_CPU_SWEEP
from paging.final import run_final

NAMES = ['410.bwaves-s0', '410.bwaves-s1', '410.bwaves-s2', '437.leslie3d-s0', '437.leslie3d-s1', '437.leslie3d-s2',
         '433.milc-s0', '462.libquantum-s0', '429.mcf-s0', 'bfs-3', 'bfs-10', 'pr-3', 'pr-10',
         # Phase 17 additions
         '450.soplex-s0', '471.omnetpp-s0', '473.astar-s0', '459.GemsFDTD-s0', '470.lbm-s0', '482.sphinx3-s0',
         'bc-3', 'cc-5', 'sssp-3']
ap = argparse.ArgumentParser()
ap.add_argument('--ratio', type=float, default=0.01)
args = ap.parse_args()
config.ensure_dirs()

SYSTEMS = {   # name -> (runner, uses a predictor on every access)
    'LRU':        (lambda t, f, log: run_prefetch_sim(t, f, LRU, make_predictor('hybrid'), NoPrefetch(), log=log), False),
    'Readahead':  (lambda t, f, log: run_prefetch_sim(t, f, LRU, make_readahead(f), AlwaysPrefetch(), log=log), True),
    'Markov':     (lambda t, f, log: run_prefetch_sim_ghost(t, f, LRU, make_predictor('hybrid'), CostAwareGhost(), log=log), True),
    'RA+dMarkov': (lambda t, f, log: run_prefetch_sim_dir(t, f, LRU, StreamMarkovPredictor(
        f, markov=make_predictor('delta-hybrid'), readahead=make_readahead(f, conf_k=1)), CostAwareDirectory(), log=log), True),
    'Final':      (lambda t, f, log: run_final(t, f, log=log), True),     # Phase 16 frozen controller
}
rows = []
for name in NAMES:
    if not os.path.exists(os.path.join(config.TRACE_DIR, f'{name}.csv')):
        print('skipping', name); continue
    t = load_trace(name, limit=config.TRACE_LIMIT)
    f = max(8, int(len(np.unique(t.pages)) * args.ratio))
    logs = {}
    for s, (run, _) in SYSTEMS.items():
        logs[s] = []
        st = run(t, f, logs[s])
        logs[s + '_n'] = st['accesses']
    for dev, d in DEVICES.items():
        for tc in T_CPU_SWEEP:
            base = replay(logs['LRU'], logs['LRU_n'], d['t_io'], d['channels'], tc)['time']
            for tp in (0.0, T_PRED):                      # without and with the CPU cost of predicting
                for s, (_, pred) in SYSTEMS.items():
                    tm = replay(logs[s], logs[s + '_n'], d['t_io'], d['channels'], tc, tp if pred else 0.0)
                    rows.append(dict(trace=name, frames=f, device=dev, t_cpu_us=tc * 1e6, t_pred_us=tp * 1e6, system=s,
                                     time_s=tm['time'], speedup=base / tm['time'],
                                     late_share=tm['stall_late'] / tm['time'], faults=sum(e[0] == 'F' for e in logs[s]),
                                     prefetches=sum(e[0] == 'P' for e in logs[s])))
    print(name, 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(config.RESULTS_DIR, 'time_model.csv'), index=False)
df = df[df.system != 'LRU']
for dev in DEVICES:
    sub = df[(df.device == dev) & (df.t_cpu_us == 1.0) & (df.t_pred_us > 0)]
    print(f'\n{dev}, compute 1 us per access, prediction 0.2 us: speedup vs LRU (>1 = faster)')
    print(sub.pivot(index='trace', columns='system', values='speedup').round(2).to_string())
g = df.groupby(['device', 't_cpu_us', 't_pred_us', 'system']).speedup
summ = pd.DataFrame({'geomean': g.apply(lambda x: float(np.exp(np.log(x).mean()))), 'worst': g.min(),
                     'slower_than_LRU': g.apply(lambda x: int((x < 1).sum()))}).round(3)
print('\nsummary over traces'); print(summ.to_string())
