"""
Stage 6 — evaluation harness.
  6a  robustness across trace segments (-s0/-s1/-s2), mean ± std   [needs real traces]
  6b  memory-size sweep (does the gain survive as RAM grows?)
  6c  multiple random seeds for the stochastic synthetic traces
  6d  figures for the paper
Usage:
  python scripts/08_evaluation_harness.py            # everything available
  python scripts/08_evaluation_harness.py --quick    # small sweep, for a smoke test
"""
import _path  # noqa: F401
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paging import config
from paging.traces import Trace, load_trace, load_all, clustered, uniform_random, phase_changing
from paging.engine import MemoryEngine, LRU, OPTFast
from paging.predictors import make_predictor
from paging.deciders import AlwaysPrefetch, CostAwareGhost
from paging.simulate import run_prefetch_sim, run_prefetch_sim_ghost

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()

config.ensure_dirs()
FIG = os.path.join(config.RESULTS_DIR, 'figures')
os.makedirs(FIG, exist_ok=True)
pd.set_option('display.width', 200)
ACCENT, GREY, DARK = '#c2700a', '#6e6e6e', '#2a2a2a'


def faults_of(trace, nf, P):
    e = MemoryEngine(nf, P(pages=trace.pages))
    for i, p, _ in trace.stream():
        e.access(p, i)
    return e.n_fault


def evaluate_config(trace, nf):
    lru = faults_of(trace, nf, LRU)
    opt = faults_of(trace, nf, OPTFast)
    alw = run_prefetch_sim(trace, nf, LRU, make_predictor('hybrid'), AlwaysPrefetch())['faults']
    gho = run_prefetch_sim_ghost(trace, nf, LRU, make_predictor('hybrid'), CostAwareGhost())['faults']
    return {'frames': nf, 'LRU': lru, 'OPT': opt, 'always': alw, 'ghost': gho,
            'always_vs_LRU': round((lru - alw) / lru, 4),
            'ghost_vs_LRU': round((lru - gho) / lru, 4),
            'opt_vs_LRU': round((lru - opt) / lru, 4)}


def frames_for(trace, ratio, floor=4):
    return max(floor, int(len(np.unique(trace.pages)) * ratio))


summary = {}

# ---------------- 6a: trace segments ----------------
SEGMENTS = ['410.bwaves-s0', '410.bwaves-s1', '410.bwaves-s2',
            '437.leslie3d-s0', '437.leslie3d-s1', '437.leslie3d-s2']
present = [s for s in SEGMENTS if os.path.exists(os.path.join(config.TRACE_DIR, f'{s}.csv'))]
if present:
    print('\n== 6a: trace segments ==')
    rows = []
    for name in present:
        t = load_trace(name, limit=config.TRACE_LIMIT)
        r = evaluate_config(t, frames_for(t, 0.01, floor=8))
        r.update({'program': name.rsplit('-', 1)[0], 'segment': name.rsplit('-', 1)[1]})
        rows.append(r)
        print(f'{name:18s} ghost={r["ghost_vs_LRU"]:+.4f}  always={r["always_vs_LRU"]:+.4f}')
    seg = pd.DataFrame(rows)
    seg.to_csv(os.path.join(config.RESULTS_DIR, 'phase6_segments.csv'), index=False)
    ss = seg.groupby('program').agg(segments=('segment', 'count'),
                                    ghost_mean=('ghost_vs_LRU', 'mean'), ghost_std=('ghost_vs_LRU', 'std'),
                                    always_mean=('always_vs_LRU', 'mean'), always_std=('always_vs_LRU', 'std')).round(4)
    print(ss)
    summary['segments'] = ss.reset_index().to_dict('records')
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(ss))
    ax.bar(x - .2, ss.always_mean, .4, yerr=ss.always_std.fillna(0), capsize=4, color=GREY, label='always')
    ax.bar(x + .2, ss.ghost_mean, .4, yerr=ss.ghost_std.fillna(0), capsize=4, color=ACCENT, label='ghost-list decider')
    ax.set_xticks(x); ax.set_xticklabels(ss.index); ax.set_ylabel('fault reduction vs LRU')
    ax.set_title('Robustness across trace segments (mean ± std)'); ax.legend(); ax.grid(axis='y', alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_segments.png'), dpi=200); plt.close()
else:
    print('\n== 6a skipped: no -s0/-s1/-s2 segment traces found '
          '(download with: python scripts/02_download_real_traces.py --pick '
          + ' '.join(SEGMENTS) + ')')

# ---------------- 6b: memory sweep ----------------
print('\n== 6b: memory-size sweep ==')
TRACES = load_all(limit=config.TRACE_LIMIT)
sweep_traces = ['437.leslie3d-s0', '410.bwaves-s0', 'pr-3', 'bfs-3', 'clustered']
ratios = [0.01, 0.02] if args.quick else [0.005, 0.01, 0.02, 0.05]
rows = []
for name in [n for n in sweep_traces if n in TRACES]:
    t = TRACES[name]; seen = set()
    for r in ratios:
        nf = frames_for(t, r)
        if nf in seen:
            continue
        seen.add(nf)
        row = evaluate_config(t, nf); row.update({'trace': name, 'ratio': r}); rows.append(row)
        print(f'{name:18s} frames={nf:4d} ghost={row["ghost_vs_LRU"]:+.4f}  opt={row["opt_vs_LRU"]:+.4f}')
sweep = pd.DataFrame(rows)
sweep.to_csv(os.path.join(config.RESULTS_DIR, 'phase6_memory_sweep.csv'), index=False)
names = list(dict.fromkeys(sweep.trace))
fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.6), sharey=True)
for ax, name in zip(np.atleast_1d(axes), names):
    d = sweep[sweep.trace == name].sort_values('frames')
    ax.plot(d.frames, d.opt_vs_LRU, 's:', color='#bbbbbb', label='OPT (bound)')
    ax.plot(d.frames, d.always_vs_LRU, 'o--', color=GREY, label='always')
    ax.plot(d.frames, d.ghost_vs_LRU, 'o-', color=ACCENT, label='ghost')
    ax.axhline(0, color='black', lw=.6); ax.set_title(name, fontsize=10); ax.set_xlabel('frames'); ax.grid(alpha=.3)
np.atleast_1d(axes)[0].set_ylabel('fault reduction vs LRU'); np.atleast_1d(axes)[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_memory_sweep.png'), dpi=200); plt.close()
summary['sweep_points'] = len(sweep)

# ---------------- 6c: seeds ----------------
print('\n== 6c: random seeds ==')
gens = {'clustered': lambda s: clustered(seed=s), 'random': lambda s: uniform_random(seed=s),
        'phase_changing': lambda s: phase_changing(seed=s)}
seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
rows = []
for name, gen in gens.items():
    for s in seeds:
        t = Trace(f'{name}-seed{s}', gen(s))
        row = evaluate_config(t, frames_for(t, 0.01, floor=8)); row.update({'trace': name, 'seed': s}); rows.append(row)
    print(f'{name:15s} done')
sd = pd.DataFrame(rows)
sd.to_csv(os.path.join(config.RESULTS_DIR, 'phase6_seeds.csv'), index=False)
sds = sd.groupby('trace').agg(frames=('frames', 'first'),
                              ghost_mean=('ghost_vs_LRU', 'mean'), ghost_std=('ghost_vs_LRU', 'std'),
                              always_mean=('always_vs_LRU', 'mean'), always_std=('always_vs_LRU', 'std')).round(4)
print(sds)
summary['seeds'] = sds.reset_index().to_dict('records')

# ---------------- 6d: figures from earlier results ----------------
abl_path = os.path.join(config.RESULTS_DIR, 'ablation_results.csv')
if not os.path.exists(abl_path):
    abl_path = os.path.join(config.RESULTS_DIR, 'reference', 'ablation_results.csv')
if os.path.exists(abl_path):
    abl = pd.read_csv(abl_path)
    sm = abl[abl.decider != 'no-prefetch'].groupby('decider').agg(mean_gain=('vs_LRU', 'mean'), worst=('vs_LRU', 'min'))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for d, r in sm.iterrows():
        ax.scatter(r.worst, r.mean_gain, s=70, color=ACCENT if d == 'cost-aware-fd' else GREY)
        ax.annotate(d, (r.worst, r.mean_gain), textcoords='offset points', xytext=(6, 4), fontsize=9)
    ax.set_xlabel('worst-case fault reduction (closer to 0 = safer)'); ax.set_ylabel('mean fault reduction')
    ax.set_title('Decision engines: safety vs benefit'); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_decider_tradeoff.png'), dpi=200); plt.close()

h2h_path = os.path.join(config.RESULTS_DIR, 'head_to_head.csv')
if os.path.exists(h2h_path):
    h2h = pd.read_csv(h2h_path).set_index(['trace', 'frames'])
    cols = [c for c in ['FIFO', 'Clock', 'LRU', 'LRU+pred', 'OPT'] if c in h2h.columns]
    ax = h2h[cols].plot(kind='bar', figsize=(9, 4.2), width=.8,
                        color=['#d0d0d0', '#a8a8a8', GREY, ACCENT, DARK][:len(cols)])
    ax.set_ylabel('page faults'); ax.set_xlabel(''); ax.set_title('Page faults: traditional policies vs prediction')
    ax.set_xticklabels([f'{a}\n({b} frames)' for a, b in h2h.index], rotation=0, fontsize=8); ax.grid(axis='y', alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_head_to_head.png'), dpi=200); plt.close()

summary['figures'] = sorted(os.listdir(FIG))
with open(os.path.join(config.RESULTS_DIR, 'stage6_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print('\nfigures:', summary['figures'])
print(f'results written to {config.RESULTS_DIR}')
