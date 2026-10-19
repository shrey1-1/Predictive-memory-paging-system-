"""
Phase 8 — stronger baselines: ARC, LRU-2 and Linux-style readahead.

  1. sanity checks on the new policies / predictor
  2. eviction policies alone (no prediction): does a smarter policy close the LRU->OPT gap?
  3. head-to-head including prediction on top of LRU and ARC, and readahead
  4. analysis tables and a figure

Usage:
  python scripts/09_phase8_baselines.py                    # everything available
  python scripts/09_phase8_baselines.py --with-lru2-pred   # also run Markov+LRU-2 (slower)
  python scripts/09_phase8_baselines.py --skip-sanity --skip-eviction-sweep

Head-to-head results are saved after every configuration and resumed on re-run; delete
results/phase8_head_to_head.csv if you change --with-lru2-pred.
Configurations whose trace is not available are skipped, so this also runs on synthetic only.
"""
import _path  # noqa: F401
import argparse, json, os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paging import config
from paging.traces import Trace, load_all
from paging.engine import (MemoryEngine, LRU, ARC, LRU2, POLICIES_EXTENDED, run_policy)
from paging.predictors import make_predictor, make_readahead
from paging.deciders import AlwaysPrefetch, CostAwareGhost
from paging.simulate import run_prefetch_sim, run_prefetch_sim_ghost

ap = argparse.ArgumentParser()
ap.add_argument('--skip-sanity', action='store_true')
ap.add_argument('--skip-eviction-sweep', action='store_true')
ap.add_argument('--with-lru2-pred', action='store_true')
args = ap.parse_args()

config.ensure_dirs()
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)
TRACES = load_all(limit=config.TRACE_LIMIT)
FIG = os.path.join(config.RESULTS_DIR, 'figures')
os.makedirs(FIG, exist_ok=True)
print(f'{len(TRACES)} traces loaded: {list(TRACES)}')


def pure_faults(trace, nf, P):
    return run_policy(trace, nf, P)['faults']


# ------------------------------------------------------------------ 1. sanity
if not args.skip_sanity:
    print('\n== 1. sanity checks ==')
    rng = np.random.default_rng(0)
    seq, scan = [], 1000
    for _ in range(20_000):                       # 12 reused pages + a one-time scan
        if rng.random() < 0.5:
            seq.append(int(rng.integers(0, 12)))
        else:
            seq.append(scan); scan += 1
    res = {k: pure_faults(Trace('scan-test', seq), 16, P) for k, P in POLICIES_EXTENDED.items()}
    print('scan resistance (16 frames):', res)
    assert res['LRU-2'] < res['LRU'] and res['ARC'] < res['LRU'], 'LRU-2 and ARC should beat LRU on a scan'
    assert min(res.values()) == res['OPT']

    bad = []
    for name, tr in TRACES.items():
        nf = max(8, int(len(np.unique(tr.pages)) * 0.02))
        r = {k: pure_faults(tr, nf, P) for k, P in POLICIES_EXTENDED.items()}
        bad += [(name, k) for k in r if r[k] < r['OPT']]
    print(f'OPT is the lower bound on {len(TRACES)} traces:', 'PASS' if not bad else f'FAIL {bad}')
    assert not bad

    if 'cyclic' in TRACES:
        uniq = len(np.unique(TRACES['cyclic'].pages))
        r = {k: pure_faults(TRACES['cyclic'], 4 * uniq, P) for k, P in POLICIES_EXTENDED.items()}
        assert all(v == uniq for v in r.values()), 'with enough memory faults must equal unique pages'
        print('plenty of memory: every policy faults only on first touches — PASS')

    for name, nf in [('sequential', 64), ('strided', 64), ('random', 64)]:
        if name not in TRACES:
            continue
        base = pure_faults(TRACES[name], nf, LRU)
        got = run_prefetch_sim(TRACES[name], nf, LRU, make_readahead(nf), AlwaysPrefetch())['faults']
        print(f'readahead on {name:10s}: LRU={base:6d}  readahead={got:6d}')
        if name == 'sequential':
            assert got < 0.05 * base
        else:
            assert abs(got - base) <= 0.01 * base
    print('all sanity checks passed')

# ------------------------------------------------------------------ 2. eviction only
if not args.skip_eviction_sweep:
    print('\n== 2. eviction policies alone ==')
    SKIP = {'sequential', 'strided', 'cyclic'}          # degenerate: all policies identical
    rows = []
    for name, t in TRACES.items():
        if name in SKIP:
            continue
        ws, seen = len(np.unique(t.pages)), set()
        for ratio in [0.01, 0.02, 0.05, 0.10]:
            nf = max(8, int(ws * ratio))
            if nf in seen:
                continue
            seen.add(nf)
            r = {'trace': name, 'ratio': ratio, 'frames': nf}
            for pname, P in POLICIES_EXTENDED.items():
                r[pname] = pure_faults(t, nf, P)
            rows.append(r)
        print(f'{name:18s} done')
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(config.RESULTS_DIR, 'phase8_eviction_only.csv'), index=False)
    red = ev.copy()
    for p in ['FIFO', 'Clock', 'LRU-2', 'ARC', 'OPT']:
        red[p + '_vs_LRU'] = ((red['LRU'] - red[p]) / red['LRU']).round(4)
    print('\npage faults:')
    print(ev.set_index(['trace', 'ratio', 'frames']).to_string())
    print('\nfault reduction vs LRU (positive = better than LRU):')
    print(red.set_index(['trace', 'ratio', 'frames'])[[c for c in red.columns if c.endswith('_vs_LRU')]].to_string())

# ------------------------------------------------------------------ 3. head-to-head
print('\n== 3. head-to-head ==')
CONFIGS = [('437.leslie3d-s0', 21), ('410.bwaves-s0', 30), ('clustered', 9), ('bfs-3', 20), ('pr-3', 15)]
CSV = os.path.join(config.RESULTS_DIR, 'phase8_head_to_head.csv')
if os.path.exists(CSV):
    rows = pd.read_csv(CSV).to_dict('records')
    done = {(r['trace'], int(r['frames'])) for r in rows}
    print(f'resuming: {len(done)} configuration(s) already saved')
else:
    rows, done = [], set()

t_all = time.time()
for tname, nf in CONFIGS:
    if tname not in TRACES:
        print(f'skipping {tname} — trace not available'); continue
    if (tname, nf) in done:
        continue
    t, t0 = TRACES[tname], time.time()
    r = {'trace': tname, 'frames': nf}
    for pname, P in POLICIES_EXTENDED.items():
        r[pname] = pure_faults(t, nf, P)
    r['RA+LRU'] = run_prefetch_sim(t, nf, LRU, make_readahead(nf), AlwaysPrefetch())['faults']
    r['RA+ARC'] = run_prefetch_sim(t, nf, ARC, make_readahead(nf), AlwaysPrefetch())['faults']
    r['Markov+LRU'] = run_prefetch_sim_ghost(t, nf, LRU, make_predictor('hybrid'), CostAwareGhost())['faults']
    r['Markov+ARC'] = run_prefetch_sim_ghost(t, nf, ARC, make_predictor('hybrid'), CostAwareGhost())['faults']
    if args.with_lru2_pred:
        r['Markov+LRU-2'] = run_prefetch_sim_ghost(t, nf, LRU2, make_predictor('hybrid'), CostAwareGhost())['faults']
    rows.append(r)
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f'{tname:18s} frames={nf:3d} done in {time.time() - t0:.0f}s')
print(f'total time: {(time.time() - t_all) / 60:.1f} min')
if not rows:
    raise SystemExit('no configurations available — generate/download traces first (scripts 01 / 02)')

h8 = pd.DataFrame(rows).set_index(['trace', 'frames'])
order = ['FIFO', 'Clock', 'LRU', 'LRU-2', 'ARC', 'RA+LRU', 'RA+ARC', 'Markov+LRU', 'Markov+ARC']
if 'Markov+LRU-2' in h8.columns:
    order.append('Markov+LRU-2')
order.append('OPT')
print('\npage faults (lower = better):')
print(h8[order].to_string())

# ------------------------------------------------------------------ 4. analysis
d = h8.copy()
trad = ['FIFO', 'Clock', 'LRU', 'LRU-2', 'ARC']
room = d['LRU'] - d['OPT']
q1 = pd.DataFrame({'LRU->OPT gap': (room / d['LRU']).round(3),
                   'LRU-2 closes': ((d['LRU'] - d['LRU-2']) / room).where(room > 0).round(3),
                   'ARC closes': ((d['LRU'] - d['ARC']) / room).where(room > 0).round(3)})
q2 = pd.DataFrame({'ARC alone': d['ARC'], 'Markov+ARC': d['Markov+ARC'],
                   'gain on ARC': ((d['ARC'] - d['Markov+ARC']) / d['ARC']).round(4),
                   'gain on LRU': ((d['LRU'] - d['Markov+LRU']) / d['LRU']).round(4)})
q3 = pd.DataFrame({'RA+LRU': d['RA+LRU'], 'Markov+LRU': d['Markov+LRU'],
                   'Markov vs readahead': ((d['RA+LRU'] - d['Markov+LRU']) / d['RA+LRU']).round(4),
                   'readahead vs LRU': ((d['LRU'] - d['RA+LRU']) / d['LRU']).round(4)})
best_trad = d[trad].min(axis=1)
best_ours = d[['Markov+LRU', 'Markov+ARC']].min(axis=1)
overall = pd.DataFrame({'best traditional': best_trad, 'best of ours': best_ours,
                        'reduction': ((best_trad - best_ours) / best_trad).round(4), 'OPT': d['OPT']})
for title, tbl in [('Q1. does a smarter eviction policy alone help? (share of LRU->OPT gap closed)', q1),
                   ('Q2. does prediction still add value on top of ARC?', q2),
                   ('Q3. Markov predictor vs Linux-style readahead (positive = Markov better)', q3),
                   ('Overall: best traditional (incl. ARC, LRU-2) vs our best', overall)]:
    print(f'\n{title}')
    print(tbl.to_string())

cols = [c for c in ['FIFO', 'LRU', 'LRU-2', 'ARC', 'RA+LRU', 'Markov+LRU', 'Markov+ARC', 'OPT'] if c in d.columns]
rel = d[cols].div(d['LRU'], axis=0)
colors = {'FIFO': '#d9d9d9', 'LRU': '#9a9a9a', 'LRU-2': '#7a7a7a', 'ARC': '#555555',
          'RA+LRU': '#b9b9b9', 'Markov+LRU': '#c2700a', 'Markov+ARC': '#e0a458', 'OPT': '#1a1a1a'}
ax = rel.plot(kind='bar', figsize=(10.5, 4.6), width=0.82, color=[colors[c] for c in cols])
ax.axhline(1.0, color='black', lw=0.7)
ax.set_ylabel('page faults relative to LRU (lower = better)'); ax.set_xlabel('')
ax.set_xticklabels([f'{a}\n({b} frames)' for a, b in rel.index], rotation=0, fontsize=8)
ax.set_title('Head-to-head including ARC, LRU-2 and Linux-style readahead')
ax.grid(axis='y', alpha=.3); ax.legend(ncol=4, fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_phase8_head_to_head.png'), dpi=200); plt.close()

with open(os.path.join(config.RESULTS_DIR, 'phase8_summary.json'), 'w') as f:
    json.dump({'faults': d.reset_index().to_dict('records'),
               'overall': overall.reset_index().to_dict('records')}, f, indent=2, default=str)
print(f'\nresults written to {config.RESULTS_DIR}')
