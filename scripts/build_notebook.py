"""
Build notebooks/predictive_paging_full_pipeline.ipynb: one self-contained Colab notebook in the style of the
original project notebook (every class defined in a numbered cell, tables and plots after each step).

Class and function code is copied from src/paging with inspect, so the notebook runs exactly the tested code
(tests/test_core.py pins its numbers). Re-run this script after changing anything in src/paging:

    python scripts/build_notebook.py
"""
import inspect, json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from paging import traces, predictors, engine, deciders, simulate, timing, final  # noqa: E402
OUT = os.path.join(ROOT, 'notebooks', 'predictive_paging_full_pipeline.ipynb')


def src(*objs):
    """Source of classes/functions, without package-relative imports (everything lives in one namespace)."""
    parts = []
    for o in objs:
        s = inspect.getsource(o)
        s = '\n'.join(l for l in s.splitlines() if not re.match(r'\s*from \.\w* import ', l))
        parts.append(s.rstrip())
    return '\n\n\n'.join(parts)


CELLS = []


def md(text):
    CELLS.append(('markdown', text.strip('\n')))


def code(text):
    CELLS.append(('code', text.strip('\n')))


GUARD = "assert 'TRACE_DIR' in globals(), 'Run cell 0a (setup) first'"

# =====================================================================================================
md("""
# Predictive Memory Paging System with Workload Awareness

G.K Shreyas Bharadwaj (24BCE2256) · Vaastav Bajanthri (24BCE2223) · P. Giphy Bershan (24BCE2201)

A trace-driven paging simulator. A **predictor** guesses the next page, a **decision engine** decides whether
prefetching that guess is worth evicting a page for, and an **eviction policy** (LRU, ...) decides what leaves RAM.
We never replace the eviction policy; we add prediction on top of it.

**How to run.** Run cell 0a first, every time the runtime (re)connects; every later cell checks for it. Then run the cells
in order. Everything is saved to Google Drive (`MyDrive/paging_sim/`), and the long cells (21, 25, 26, 27, 29) save after
each trace and resume where they stopped, so a disconnect loses at most one trace.

| Cells | Stage | What it does |
|---|---|---|
| 0a-0e | 1-2 | setup, synthetic traces, real traces (22 LoadTraces), characterisation, the `Trace` object |
| 1-10 | 3 | predictors (stride, Markov, hybrid), quality, attribution, footprint, capacity, phase adaptation |
| 11-14b | 4 | memory engine, FIFO / LRU / Clock / OPT, baselines and their checks |
| 15-17 | 5 | decision engines and the ablation (the main Stage 5 result) |
| 18-19 | | head-to-head on FIFO/LRU/Clock; ghost-list estimate of the fault delta |
| 20-22 | 8-10 | ARC, LRU-2, Linux-style readahead; readahead + delta-Markov |
| 23-24 | 14-16 | shadow-directory feedback; **the final system** |
| 25-29 | 17-18 | 22 real traces, break-even cost, time model, (optional) tuning, component comparison |
| 30-31 | | results at a glance; consistency check of every result file |

**Honest framing.** Markov prefetching is Joseph & Grunwald (1997); fault-rate feedback resembles Page Fault Frequency;
prefetch-aware management resembles PACMan. Our contribution is the *evaluation* of which feedback signal works, a
deployable estimate of it, and the combination with readahead. Real traces are last-level-cache load traces viewed at
4 KB page granularity: an approximation of OS paging, not OS paging traces. Numbers from synthetic traces must be labelled as such.
""")

# ----------------------------------------------------------------------------------------------- 0a
md("""
## Stage 1-2: traces
### Cell 0a — setup
On Colab: mounts Drive; traces go to `paging_sim/traces/` (the same folder as before, so traces you already have
are reused), results to `paging_sim/results/`. Locally (VS Code / Jupyter): uses the repo's `data/traces/` and
`results/notebook/`. Also defines the constants and the house style (greys + one amber accent).
""")
code("""
#cell 0a — setup: mount Drive (Colab) or use the repo folders (local), constants, imports, plot style (run first, every time)
import os, csv, gzip, lzma, json, time, bisect, heapq, itertools, subprocess, urllib.request
from collections import OrderedDict, deque, defaultdict
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display

try:                                   # on Google Colab: everything lives on Drive
    from google.colab import drive
    drive.mount('/content/drive')
    ON_COLAB = True
except ImportError:                    # locally (VS Code / Jupyter): use the repo's data/ and results/ folders
    ON_COLAB = False

if ON_COLAB:
    BASE      = '/content/drive/MyDrive/paging_sim'
    TRACE_DIR = f'{BASE}/traces'         # converted traces (access_id,page_id), kept on Drive
    RESULTS   = f'{BASE}/results'        # every CSV / JSON this notebook writes
    RAW       = '/content/raw'           # compressed downloads: Colab disk, not Drive (large and temporary)
    CHAMPSIM  = '/content/ChampSim'      # cloned repo with the list of trace download links
else:
    REPO = os.path.abspath('..' if os.path.basename(os.getcwd()) == 'notebooks' else '.')
    TRACE_DIR = os.path.join(REPO, 'data', 'traces')
    RESULTS   = os.path.join(REPO, 'results', 'notebook')   # kept apart from the scripts' outputs
    RAW       = os.path.join(REPO, 'data', 'raw')
    CHAMPSIM  = os.path.join(REPO, 'ChampSim')
FIG = f'{RESULTS}/figures'
for d in (TRACE_DIR, RESULTS, FIG, RAW):
    os.makedirs(d, exist_ok=True)

PAGE_SHIFT   = 12        # 4 KB pages: page = address >> 12
MAX_ACCESSES = 500_000   # cap when converting a real trace
TRACE_LIMIT  = 200_000   # cap when loading a trace for the experiments
WARMUP_FRAC  = 0.25      # first 25% of each trace: the only part any setting may be tuned on

SYNTHETIC = ['sequential', 'strided', 'cyclic', 'clustered', 'random', 'phase_changing']
PICK      = ['410.bwaves-s0', '437.leslie3d-s0', 'bfs-3', 'pr-3']       # the original four (Stages 3-5)
REAL_ALL  = ['410.bwaves-s0', '410.bwaves-s1', '410.bwaves-s2', '437.leslie3d-s0', '437.leslie3d-s1', '437.leslie3d-s2',
             '433.milc-s0', '462.libquantum-s0', '429.mcf-s0', 'bfs-3', 'bfs-10', 'pr-3', 'pr-10',
             '450.soplex-s0', '471.omnetpp-s0', '473.astar-s0', '459.GemsFDTD-s0', '470.lbm-s0', '482.sphinx3-s0',
             'bc-3', 'cc-5', 'sssp-3']                                    # all 22 (Phase 17)

# house style: greys + one amber accent, no blue
GREY, AMBER = '#555555', '#c2700a'
GREYS = ['#222222', '#555555', '#888888', '#bbbbbb']
CMAP = LinearSegmentedColormap.from_list('grey_amber', ['#f4f4f4', '#d9a15c', AMBER])
plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=['#222222', AMBER, '#777777', '#aaaaaa', '#444444']),
                     'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)


def available(names):
    \"\"\"The traces from `names` that exist on Drive (experiments skip the others).\"\"\"
    return [n for n in names if os.path.exists(f'{TRACE_DIR}/{n}.csv')]


def saved_rows(name):
    \"\"\"Rows already saved by a long, resumable cell (empty list if none).\"\"\"
    p = f'{RESULTS}/{name}'
    return pd.read_csv(p).to_dict('records') if os.path.exists(p) else []


def frames_for(trace, ratio=0.01):
    \"\"\"RAM size = a fraction of the trace's working set (never a fixed frame count), at least 8 frames.\"\"\"
    return max(8, int(len(np.unique(trace.pages)) * ratio))


print('traces :', TRACE_DIR, f'({len(available(SYNTHETIC + REAL_ALL))} of {len(SYNTHETIC + REAL_ALL)} present)')
print('results:', RESULTS)
""")

# ----------------------------------------------------------------------------------------------- 0b
md("""
### Cell 0b — Stage 1a: six synthetic traces
Controlled workloads with a known pattern, so we can check that each predictor finds what it should, and that nothing
finds a pattern in *random*. `phase_changing` (sequential → cyclic → random) tests adaptation.
""")
code(f"""
#cell 0b — synthetic trace generators; writes any missing trace to Drive and plots the first accesses
{GUARD}

{src(traces.sequential, traces.strided, traces.cyclic, traces.clustered, traces.uniform_random, traces.phase_changing)}


GENERATORS = {{'sequential': sequential, 'strided': strided, 'cyclic': cyclic,
              'clustered': clustered, 'random': uniform_random, 'phase_changing': phase_changing}}


{src(traces.write_trace_csv)}


for name, fn in GENERATORS.items():
    path = f'{{TRACE_DIR}}/{{name}}.csv'
    if os.path.exists(path):
        print(f'{{name:15s}} already on Drive'); continue
    pages = fn()
    write_trace_csv(pages, path)
    print(f'{{name:15s}} {{len(pages):6d}} accesses written')

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, name in zip(axes.flat, SYNTHETIC):
    p = pd.read_csv(f'{{TRACE_DIR}}/{{name}}.csv')['page_id'].values[:2000]
    ax.plot(p, lw=0.6, color=AMBER if name == 'phase_changing' else '#333333')
    ax.set_title(name, fontsize=10); ax.set_xlabel('access'); ax.set_ylabel('page id')
plt.suptitle('Synthetic traces, first 2,000 accesses (synthetic)')
plt.tight_layout(); plt.savefig(f'{{FIG}}/fig_synthetic_traces.png', dpi=200); plt.show()
""")

# ----------------------------------------------------------------------------------------------- 0c
md("""
### Cell 0c — Stage 1b: real traces (22 ML-DPC LoadTraces)
SPEC CPU 2006 and GAP graph benchmarks from the ML-based Data Prefetching Competition. Use **LoadTraces** (`.txt.xz`,
plain text lines `instr_id, cycle, address, ip, llc_hit`), never ChampSimTraces (binary). Each is converted to 4 KB page
ids, capped at 500,000 accesses.

* Traces already on Drive are skipped. The download goes to the Colab disk first; only the converted CSV is copied to Drive.
* Several minutes per trace. If it disconnects, just re-run: finished traces are skipped.
* `DOWNLOAD_REAL = False` gives a synthetic-only run (every later cell skips what is missing).
""")
code(f"""
#cell 0c — download and convert the real traces (skips any already on Drive)
{GUARD}

DOWNLOAD_REAL = True


{src(traces._open, traces.convert, traces.parse_download_links)}


missing = [n for n in REAL_ALL if n not in available(REAL_ALL)]
print(f'{{len(REAL_ALL) - len(missing)}} of {{len(REAL_ALL)}} real traces already on Drive')
if DOWNLOAD_REAL and missing:
    if not os.path.isdir(CHAMPSIM):                      # the repo holds the list of download links
        subprocess.run(['git', 'clone', '-q', 'https://github.com/Quangmire/ChampSim.git', CHAMPSIM], check=True)
    URLS = parse_download_links(f'{{CHAMPSIM}}/download_links')
    print(len(URLS), 'LoadTraces listed')
    for name in missing:                                  # one at a time, so each finished trace is saved
        if name not in URLS:
            print(f'{{name:18s}} not in download_links'); continue
        raw = f'{{RAW}}/{{name}}.txt.xz'
        if not os.path.exists(raw):
            print(f'downloading {{name}} ...', flush=True)
            urllib.request.urlretrieve(URLS[name], raw)
        try:
            with lzma.open(raw, 'rt') as f:
                f.readline()
        except Exception as e:
            print(f'{{name}}: BAD DOWNLOAD ({{e}}): Box may have returned an HTML page; delete {{raw}} and retry')
            continue
        n, skipped = convert(raw, f'{{TRACE_DIR}}/{{name}}.csv')
        print(f'{{name:18s}} {{os.path.getsize(raw) / 1e6:6.0f}} MB -> {{n}} accesses, {{skipped}} skipped')
        os.remove(raw)

have = available(REAL_ALL)
print(f'\\nreal traces available: {{len(have)}} / {{len(REAL_ALL)}}')
if len(have) < len(REAL_ALL):
    print('missing (results for these are skipped):', [n for n in REAL_ALL if n not in have])
""")

# ----------------------------------------------------------------------------------------------- 0d
md("""
### Cell 0d — trace characterisation
How regular is each workload? `reuse` = accesses per distinct page; `top5_share` = share of steps covered by the five most
common page deltas (high = predictable, stride-like structure). Measured on the first 200,000 accesses, as in the experiments.
""")
code(f"""
#cell 0d — characterise every available trace, and plot the original four
{GUARD}

rows = []
for name in available(SYNTHETIC + REAL_ALL):
    p = pd.read_csv(f'{{TRACE_DIR}}/{{name}}.csv')['page_id'].values[:TRACE_LIMIT]
    d = np.diff(p); vc = pd.Series(d).value_counts()
    rows.append({{'trace': name, 'type': 'synthetic' if name in SYNTHETIC else 'real',
                 'accesses': len(p), 'unique_pages': len(np.unique(p)),
                 'reuse': round(len(p) / len(np.unique(p)), 2), 'unique_deltas': len(vc),
                 'top_delta': int(vc.index[0]), 'top_delta_share': round(vc.iloc[0] / len(d), 3),
                 'top5_share': round(vc.iloc[:5].sum() / len(d), 3)}})
chars = pd.DataFrame(rows)
chars.to_csv(f'{{RESULTS}}/trace_characteristics.csv', index=False)
display(chars)

real4 = available(PICK)
if real4:
    fig, axes = plt.subplots(2, len(real4), figsize=(4.5 * len(real4), 7), squeeze=False)
    for i, name in enumerate(real4):
        p = pd.read_csv(f'{{TRACE_DIR}}/{{name}}.csv')['page_id'].values
        axes[0, i].plot(p[:2000], lw=0.5, color='#333333')
        axes[0, i].set_title(name, fontsize=10); axes[0, i].set_ylabel('page id')
        d = np.diff(p[:50000]); d = d[np.abs(d) < 50]
        axes[1, i].hist(d, bins=101, color=AMBER)
        axes[1, i].set_xlabel('page delta'); axes[1, i].set_yscale('log')
    plt.tight_layout(); plt.savefig(f'{{FIG}}/fig_real_traces.png', dpi=200); plt.show()
""")

# ----------------------------------------------------------------------------------------------- 0e
md("""
### Cell 0e — Stage 2: the normalised `Trace` object
Every trace, synthetic or real, becomes the same object. `train` is the 25% warm-up portion (the only data any setting may
be tuned on); `test` is the rest. `stream()` yields accesses one at a time, with no lookahead.
""")
code(f"""
#cell 0e — Trace class, loader and checks
{GUARD}

{src(traces.Trace)}


def load_trace(name, trace_dir=None, limit=None, train_frac=WARMUP_FRAC):
    df = pd.read_csv(f'{{trace_dir or TRACE_DIR}}/{{name}}.csv')
    pages = df['page_id'].values
    if limit:
        pages = pages[:limit]
    return Trace(name, pages, train_frac)


t = load_trace(available(SYNTHETIC + REAL_ALL)[0], limit=50_000)
print(t)
assert len(t.pages) == len(t.deltas), 'delta/page length mismatch'
assert len(t.train) + len(t.test) == len(t), 'split loses accesses'
assert np.array_equal(np.concatenate([t.train, t.test]), t.pages), 'split reordered the sequence'
assert np.array_equal(np.cumsum(t.deltas) + t.pages[0] - t.deltas[0], t.pages), 'deltas inconsistent'
assert [p for _, p, _ in t.stream()] == t.pages.tolist(), 'stream != pages'
print(f'train: {{len(t.train)}}  test: {{len(t.test)}}')
print('first 10 pages :', t.pages[:10].tolist())
print('first 10 deltas:', t.deltas[:10].tolist())
print('all checks passed')
""")

# ======================================================================================== Stage 3
md("""
## Stage 3: prediction
### Cell 1 — the predictors
* **Markov**: a bounded table `context -> {next page: count}` (LRU-evicted contexts, top-K successors, counts halved periodically).
* **Stride**: next = last + stride when the last three steps agree.
* **Hybrid**: order-2 Markov first, then order-1, then stride. Every table learns on every access.

Every predictor implements `predict(history, top_n)`, `update(history, actual_next)`, `stats()`.
""")
code(f"""
#cell 1 — Markov, stride and hybrid predictors
{GUARD}

{src(predictors.MarkovPredictor, predictors.StridePredictor, predictors.HybridPredictor)}


print('predictors ready')
""")

md("""
### Cell 2 — how prediction quality is measured
Learns online over the whole trace, but measures only after the 25% warm-up. `coverage` = share of accesses with a
prediction; `accuracy` = share of predictions that were right; `useful = coverage × accuracy`.
""")
code(f"""
#cell 2 — eval_predictor
{GUARD}

{src(simulate.eval_predictor)}


print('eval_predictor ready')
""")

md("""
### Cell 3 — load the Stage 3-5 traces
The six synthetic traces plus the original four real ones, first 200,000 accesses each.
""")
code(f"""
#cell 3 — TRACES: the traces used in Stages 3-5, as Trace objects
{GUARD}

TRACES = {{n: load_trace(n, limit=TRACE_LIMIT) for n in available(SYNTHETIC + PICK)}}
REAL = [n for n in PICK if n in TRACES]
assert all(hasattr(t, 'pages') for t in TRACES.values()), 'TRACES holds raw lists'
print(f'{{len(TRACES)}} traces loaded ({{len(REAL)}} real):')
for t in TRACES.values():
    print('  ', t)
if not REAL:
    print('NOTE: no real traces: every result below is SYNTHETIC ONLY and must be labelled so')
""")

md("""
### Cell 4 — frozen hyperparameters
Chosen once from the sweeps below and then frozen: **never tuned on evaluation data**. `make_predictor` is the single
place that builds predictors (the `delta-hybrid` kind is defined in cell 22).
""")
code(f"""
#cell 4 — FROZEN hyperparameters and the predictor factory
{GUARD}

FROZEN = dict(capacity=256, max_succ=2, order=2, decay_every=25_000)
#   capacity=256  -> useful_real flat (0.783) from 256 to 4096 entries; 8.5 KB table
#   max_succ=2    -> 2 / 4 / 8 successors differ by < 0.0002; take the cheapest
#   order=2       -> beats order-1 by ~4.5 points
#   decay=25000   -> marginally best

KINDS = ['stride', 'markov-o1', 'markov-o2', 'hybrid']


{src(predictors.make_predictor)}


print('FROZEN =', FROZEN)
""")

md("""
### Cell 5 — prediction quality on every trace
What to look for: Markov is ~0 on `sequential`/`strided` (their working sets are larger than the 256-entry table; stride
covers them), nothing predicts `random`, and the hybrid is the best or tied everywhere.
""")
code(f"""
#cell 5 — coverage / accuracy / useful for every predictor and trace
{GUARD}
assert 'TRACES' in globals() and 'make_predictor' in globals(), 'run cells 1-4 first'

rows = []
for name, t in TRACES.items():
    for k in KINDS:
        s = eval_predictor(t, make_predictor(k))
        rows.append({{'trace': name, 'type': 'real' if name in REAL else 'synthetic', 'predictor': k,
                     'coverage': round(s['coverage'], 3), 'accuracy': round(s['accuracy'], 3),
                     'useful': round(s['useful'], 3)}})
results = pd.DataFrame(rows)
results.to_csv(f'{{RESULTS}}/predictor_results.csv', index=False)

for metric in ('coverage', 'accuracy'):
    print(f'\\n===== {{metric.upper()}} =====')
    display(results.pivot(index='trace', columns='predictor', values=metric)[KINDS])
print('\\n===== USEFUL = coverage x accuracy =====')
useful = results.pivot(index='trace', columns='predictor', values='useful')[KINDS]
try:                                     # coloured table (needs jinja2, which Colab has)
    display(useful.style.background_gradient(cmap=CMAP, axis=None, vmin=0, vmax=1).format('{{:.3f}}'))
except (AttributeError, ImportError):
    display(useful)
h = results[results.predictor == 'hybrid']
print(f"hybrid mean useful: real {{h[h.type == 'real'].useful.mean():.4f}}   all {{h.useful.mean():.4f}}")
""")

md("""
### Cell 6 — which part of the hybrid answered?
Share of predictions made by each tier (`none` = no tier had a prediction).
""")
code(f"""
#cell 6 — attribution of the hybrid's predictions
{GUARD}

attrib = []
for name, t in TRACES.items():
    h = make_predictor('hybrid')
    eval_predictor(t, h)
    src = h.stats()['sources']; tot = max(1, sum(src.values()))
    attrib.append({{'trace': name, **{{k: round(v / tot, 3) for k, v in src.items()}}}})
att = pd.DataFrame(attrib).set_index('trace')
att.to_csv(f'{{RESULTS}}/predictor_attribution.csv')
display(att)

att[['m2', 'm1', 'stride', 'none']].plot(kind='barh', stacked=True, figsize=(9, 5),
                                         color=['#222222', '#888888', AMBER, '#dddddd'])
plt.xlabel('fraction of predictions'); plt.title('Which sub-predictor answered')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)); plt.tight_layout()
plt.savefig(f'{{FIG}}/fig_attribution.png', dpi=200); plt.show()
""")

md("""
### Cell 7 — memory footprint of the order-2 table
Information-theoretic size (8-byte page numbers, 9 bytes per successor), not Python overhead. The full hybrid also has an
order-1 table (~6.5 KB more), so ~15 KB in total.
""")
code(f"""
#cell 7 — table footprint and overhead as a share of simulated RAM
{GUARD}

CTX_BYTES  = FROZEN['order'] * 8    # 64-bit page numbers
SUCC_BYTES = 9                      # 8 B page + 1 B saturating counter
MAX_BYTES  = FROZEN['capacity'] * CTX_BYTES + FROZEN['capacity'] * FROZEN['max_succ'] * SUCC_BYTES

foot = []
for name, t in TRACES.items():
    m = MarkovPredictor(**FROZEN)
    eval_predictor(t, m)
    n_ctx = len(m.table); n_succ = sum(len(s) for s in m.table.values())
    raw_b = n_ctx * CTX_BYTES + n_succ * SUCC_BYTES
    foot.append({{'trace': name, 'contexts': n_ctx, 'successors': n_succ, 'evictions': m.evictions,
                 'table_KB': round(raw_b / 1024, 1), 'pct_of_max': round(100 * raw_b / MAX_BYTES, 1)}})
fp = pd.DataFrame(foot)
fp.to_csv(f'{{RESULTS}}/memory_footprint.csv', index=False)
display(fp)
print(f"bounded ceiling: {{MAX_BYTES / 1024:.1f}} KB (~{{MAX_BYTES / 4096:.1f}} pages); observed max {{fp.table_KB.max():.1f}} KB")
for frames in (128, 256, 512, 1024):
    print(f'  overhead at {{frames:5d}} frames: {{100 * MAX_BYTES / 1024 / (frames * 4):.2f}}% of RAM')
""")

md("""
### Cell 8 — capacity sweep
Does a bigger table predict better? On the real traces `useful` is flat from 128 entries up, while the table grows 32×:
that is why capacity was frozen at 256.
""")
code(f"""
#cell 8 — table capacity vs usefulness vs size (order-2 Markov)
{GUARD}

rows = []
for cap in (128, 256, 512, 1024, 2048, 4096):
    cfg = {{**FROZEN, 'capacity': cap}}
    real_s, all_s = [], []
    for name, t in TRACES.items():
        u = eval_predictor(t, MarkovPredictor(**cfg))['useful']
        all_s.append(u)
        if name in REAL:
            real_s.append(u)
    rows.append({{'capacity': cap, 'table_KB': round((cap * CTX_BYTES + cap * cfg['max_succ'] * SUCC_BYTES) / 1024, 1),
                 'useful_real': round(np.mean(real_s), 3) if real_s else None, 'useful_all': round(np.mean(all_s), 3)}})
sw = pd.DataFrame(rows)
sw.to_csv(f'{{RESULTS}}/capacity_sweep.csv', index=False)
display(sw)

fig, ax1 = plt.subplots(figsize=(8, 4.5))
if sw.useful_real.notna().any():
    ax1.plot(sw.capacity, sw.useful_real, 'o-', color='#222222', label='useful (real)')
ax1.plot(sw.capacity, sw.useful_all, 's--', color='#888888', label='useful (all)')
ax1.set_xscale('log', base=2); ax1.set_xlabel('table capacity (entries)')
ax1.set_ylabel('useful prediction rate'); ax1.set_ylim(0, 1)
ax2 = ax1.twinx(); ax2.plot(sw.capacity, sw.table_KB, '^-', color=AMBER, label='table KB')
ax2.set_ylabel('table size (KB)')
ax1.legend(loc='lower right'); ax2.legend(loc='center right')
plt.title('Accuracy is flat; memory is not'); plt.tight_layout()
plt.savefig(f'{{FIG}}/fig_capacity_sweep.png', dpi=200); plt.show()
""")

md("""
### Cell 9 — adaptation across workload phases (synthetic)
Rolling accuracy on `phase_changing` (sequential → cyclic → random), and how many accesses each predictor needs to recover
after a phase change. The hybrid should hold the best tier in each phase.
""")
code(f"""
#cell 9 — rolling accuracy on phase_changing, and recovery times
{GUARD}

{src(simulate.rolling_accuracy)}


if 'phase_changing' in TRACES:
    t = TRACES['phase_changing']; n = len(t.pages); b1, b2 = n // 3, 2 * n // 3
    fig, axes = plt.subplots(len(KINDS), 1, figsize=(12, 2.2 * len(KINDS)), sharex=True, sharey=True)
    for ax, k in zip(axes, KINDS):
        xs, ys = rolling_accuracy(t, make_predictor(k))
        ax.plot(xs, ys, lw=1.0, color=AMBER if k == 'hybrid' else '#333333')
        for b in (b1, b2):
            ax.axvline(b, ls='--', c='grey', alpha=.6)
        ax.set_ylim(-.05, 1.1); ax.set_ylabel(k, fontsize=9); ax.grid(alpha=.3)
    axes[0].set_title('Adaptation across phases (synthetic)', pad=28)
    for x, label in ((b1 / 2, 'sequential'), ((b1 + b2) / 2, 'cyclic'), ((b2 + n) / 2, 'random')):
        axes[0].text(x, 1.22, label, ha='center', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{{FIG}}/fig_phase_adaptation.png', dpi=200, bbox_inches='tight'); plt.show()

    def recovery_time(trace, predictor, boundary, window=50, step=10, target=0.9, lookahead=5000):
        xs, ys = rolling_accuracy(trace, predictor, window=window, step=step)
        post = [(x, y) for x, y in zip(xs, ys) if boundary <= x <= boundary + lookahead]
        if not post:
            return None
        plateau = max(y for _, y in post)
        if plateau <= 0:
            return 0
        return next((x - boundary for x, y in post if y >= target * plateau), None)

    rec = pd.DataFrame([{{'predictor': k, 'seq -> cyclic': recovery_time(t, make_predictor(k), b1),
                         'cyclic -> random': recovery_time(t, make_predictor(k), b2)}} for k in KINDS])
    print('accesses needed to reach 90% of the post-transition plateau:')
    display(rec)
else:
    print('phase_changing trace missing: run cell 0b')
""")

md("""
### Cell 10 — Stage 3 summary
""")
code(f"""
#cell 10 — save the Stage 3 summary
{GUARD}

summary = {{'frozen_config': FROZEN, 'n_traces': len(TRACES), 'real_traces': REAL, 'synthetic': SYNTHETIC,
           'table_KB': round(MAX_BYTES / 1024, 1),
           'hybrid_mean_useful_real': float(results[(results.predictor == 'hybrid') & (results.type == 'real')].useful.mean())
                                      if REAL else None,
           'hybrid_mean_useful_all': float(results[results.predictor == 'hybrid'].useful.mean())}}
with open(f'{{RESULTS}}/stage3_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
""")

# ======================================================================================== Stage 4
md("""
## Stage 4: memory engine and eviction policies
### Cell 11 — the memory engine
Page table + frame table + fault handler; it knows nothing about policy. **Architectural rule: `_install()` is the only
path into RAM.** A prefetch goes through it exactly like a demand fault, so a broken predictor can make the system slow,
never wrong.
""")
code(f"""
#cell 11 — MemoryEngine
{GUARD}

{src(engine.MemoryEngine)}


print('MemoryEngine ready')
""")

md("""
### Cell 12 — FIFO, LRU and Clock
Every policy implements `on_insert`, `on_hit`, `on_evict`, `choose_victim`, and takes `pages=` in its constructor.
""")
code(f"""
#cell 12 — eviction policies
{GUARD}

{src(engine.FIFO, engine.LRU, engine.Clock)}


print('FIFO, LRU, Clock ready')
""")

md("""
### Cell 12b — Belady's OPT
The offline optimum: evicts the page whose next use is furthest away. It is allowed to see the future, so it is a lower
bound for every *eviction* policy (not for prefetching, which can also remove first-touch faults).
""")
code(f"""
#cell 12b — OPT and a helper that replays a trace through one policy
{GUARD}

{src(engine.OPTFast)}


POLICIES = {{'FIFO': FIFO, 'LRU': LRU, 'Clock': Clock, 'OPT': OPTFast}}


{src(engine.run_policy)}


print('OPTFast ready')
""")

md("""
### Cell 13 — baselines
FIFO, LRU, Clock and OPT with RAM = 1, 2, 5, 10, 25% of each trace's working set. Memory is always sized as a fraction of the
working set: with ample memory every policy is identical.
""")
code(f"""
#cell 13 — baseline fault counts
{GUARD}

RATIOS = [0.01, 0.02, 0.05, 0.10, 0.25]
rows = []
for name, t in TRACES.items():
    ws = len(np.unique(t.pages))
    for r in RATIOS:
        nf = max(8, int(ws * r))
        for pname, P in POLICIES.items():
            s = run_policy(t, nf, P)
            rows.append({{'trace': name, 'ws': ws, 'ratio': r, 'frames': nf, 'policy': pname,
                         'faults': s['faults'], 'fault_rate': round(s['fault_rate'], 4)}})
    print(f'{{name:18s}} done', flush=True)
base = pd.DataFrame(rows).drop_duplicates(['trace', 'frames', 'policy'])
base.to_csv(f'{{RESULTS}}/baseline_results.csv', index=False)
display(base.pivot_table(index=['trace', 'frames'], columns='policy', values='fault_rate')[['FIFO', 'LRU', 'Clock', 'OPT']])
""")

md("""
### Cell 14 — sanity checks on the baselines
OPT must never be beaten; LRU and OPT must never get worse with more memory; fault rates must lie in [0, 1].
""")
code(f"""
#cell 14 — baseline verification
{GUARD}
assert 'base' in globals(), 'run cell 13 first'

piv = base.pivot_table(index=['trace', 'frames'], columns='policy', values='faults')
fails = []
for pol in ('FIFO', 'LRU', 'Clock'):
    n = int((piv[pol] < piv['OPT']).sum())
    if n: fails.append(f'{{pol}} beats OPT on {{n}} configs')
for name in TRACES:
    for pol in ('LRU', 'OPT'):
        sub = base[(base.trace == name) & (base.policy == pol)].sort_values('frames')
        if not sub.faults.is_monotonic_decreasing:
            fails.append(f'{{pol}} non-monotonic on {{name}}')
if ((base.fault_rate < 0) | (base.fault_rate > 1)).any():
    fails.append('fault_rate out of range')
print('FAILED:' if fails else 'all checks passed')
for f in fails: print(' -', f)
assert not fails

cmp = piv.assign(LRU_vs_FIFO=lambda d: ((d.FIFO - d.LRU) / d.FIFO).round(4),
                 headroom=lambda d: ((d.LRU - d.OPT) / d.LRU).round(4))
print('\\nLRU vs FIFO (positive = LRU better) and headroom = (LRU - OPT) / LRU:')
display(cmp[['FIFO', 'LRU', 'Clock', 'OPT', 'LRU_vs_FIFO', 'headroom']])
""")

md("""
### Cell 14b — where do predictions actually go?
Before deciding *how* to gate prefetches: how many predictions name a page that is already in RAM, and how confident are
they? (If almost every prediction is already resident, there is little to prefetch.)
""")
code(f"""
#cell 14b — confidence diagnostic
{GUARD}


def diagnose(tname, nf, thr=0.4, hist_len=8):
    t = TRACES[tname]
    pred = make_predictor('hybrid'); eng = MemoryEngine(nf, LRU(pages=t.pages))
    hist, confs = [], []
    n_pred = n_resident = n_lowconf = 0
    for i, page, _ in t.stream():
        eng.access(page, i)
        hist.append(page)
        if len(hist) > hist_len: hist.pop(0)
        if len(hist) >= 2: pred.update(hist[:-1], page)
        for c, cf in pred.predict(hist, top_n=1):
            n_pred += 1; confs.append(cf)
            if c in eng.resident: n_resident += 1
            elif cf < thr: n_lowconf += 1
    q = np.percentile(confs, [10, 25, 50, 75, 90]) if confs else [0] * 5
    return {{'trace': tname, 'frames': nf, 'predictions': n_pred,
            'already_resident': round(n_resident / max(1, n_pred), 4), 'below_thr': n_lowconf,
            'actionable_per_access': round((n_pred - n_resident - n_lowconf) / len(t.pages), 4),
            **{{f'conf_p{{p}}': round(v, 3) for p, v in zip((10, 25, 50, 75, 90), q)}}}}


CONFIGS_14B = [('437.leslie3d-s0', 21), ('bfs-3', 58), ('bfs-3', 116), ('pr-3', 40), ('clustered', 9), ('410.bwaves-s0', 30)]
display(pd.DataFrame([diagnose(tn, nf) for tn, nf in CONFIGS_14B if tn in TRACES]))
""")

# ======================================================================================== Stage 5
md("""
## Stage 5: the decision engine
### Cell 15 — simple decision engines
On full RAM every prefetch evicts a page, so a wrong guess can *cause* a fault. A decider answers "prefetch this?".
Interface: `decide(page, conf, state)`, `feedback(page, used)`, `stats()`.
* `always` / `fixed-θ` ignore cost; `cost-aware` adapts a threshold from prefetch **precision** (kept as a comparison: it
  turns out to be a misleading signal under eviction pressure).
""")
code(f"""
#cell 15 — NoPrefetch, AlwaysPrefetch, FixedThreshold, CostAware (precision feedback)
{GUARD}

{src(deciders.NoPrefetch, deciders.AlwaysPrefetch, deciders.FixedThreshold, deciders.CostAware)}


print('deciders ready')
""")

md("""
### Cell 15b — fault-delta feedback
Uses the only signal that matters: **is prefetching reducing faults compared with not prefetching?** A shadow copy of the
memory runs the same trace without prefetching; every epoch, `delta = shadow faults − real faults`. Winning → lower the
threshold (prefetch more); losing → raise it. The shadow engine is an experimental oracle (cells 19 and 23 make it
deployable).
""")
code(f"""
#cell 15b — CostAwareFaultDelta (the oracle version)
{GUARD}

{src(deciders.CostAwareFaultDelta)}


print('CostAwareFaultDelta ready')
""")

md("""
### Cell 16 — the simulator
Predictor + decider + memory engine together. Each access: (1) demand access, (2) tell the decider which prefetches were
used or wasted, (3) fault-delta epoch, (4) the predictor learns, (5) predict and decide.
""")
code(f"""
#cell 16 — run_prefetch_sim
{GUARD}

{src(simulate.run_prefetch_sim)}


print('run_prefetch_sim ready')
""")

md("""
### Cell 17 — decision-engine ablation (the main Stage 5 result)
Same predictor (hybrid), LRU underneath, seven deciders, eight configurations (RAM ≈ 1% of working set; `clustered` is the
synthetic stress case). `vs_LRU = (LRU − faults) / LRU`: positive = fewer faults than LRU, negative = prefetching *caused*
faults. `headroom = (LRU − OPT) / LRU` is what a perfect eviction policy could save; prefetching can exceed it.
""")
code(f"""
#cell 17 — ablation
{GUARD}
assert 'run_prefetch_sim' in globals() and 'CostAwareFaultDelta' in globals(), 'run cells 15-16 first'

ABLATION_CONFIGS = [('437.leslie3d-s0', 21), ('410.bwaves-s0', 30), ('clustered', 6), ('clustered', 9),
                    ('bfs-3', 20), ('bfs-3', 58), ('pr-3', 15), ('pr-3', 40)]
DECIDERS = [lambda: NoPrefetch(), lambda: AlwaysPrefetch(), lambda: FixedThreshold(0.3), lambda: FixedThreshold(0.6),
            lambda: FixedThreshold(0.85), lambda: CostAware(), lambda: CostAwareFaultDelta()]
EXTRA = ('final_thr', 'thr_min', 'thr_max', 'thr_moves', 'mean_delta_rate', 'epochs_winning', 'epochs_losing',
         'approved', 'd_budget', 'd_cost', 'd_thr', 'shadow_faults')

rows = []
for tname, nf in ABLATION_CONFIGS:
    if tname not in TRACES:
        print(f'skipping {{tname}} (trace missing)'); continue
    t = TRACES[tname]
    lru_ref, opt_ref = run_policy(t, nf, LRU)['faults'], run_policy(t, nf, OPTFast)['faults']
    for mk in DECIDERS:
        s = run_prefetch_sim(t, nf, LRU, make_predictor('hybrid'), mk())
        row = {{'trace': tname, 'frames': nf, 'decider': s['decider'], 'faults': s['faults'],
               'vs_LRU': round((lru_ref - s['faults']) / max(1, lru_ref), 4),
               'headroom': round((lru_ref - opt_ref) / max(1, lru_ref), 4),
               'prefetches': s['prefetches'], 'pf_used': s['pf_used'], 'pf_wasted': s['pf_wasted'],
               'pf_precision': round(s['pf_precision'], 3), 'evictions': s['evictions'],
               'LRU_ref': lru_ref, 'OPT_ref': opt_ref}}
        for k in EXTRA:
            row[k] = s.get(k)
        rows.append(row)
    print(f'{{tname:18s}} frames={{nf:3d}} done', flush=True)

abl = pd.DataFrame(rows)
base_ev = abl[abl.decider == 'no-prefetch'].set_index(['trace', 'frames'])['evictions']
abl['evict_vs_base'] = abl.apply(lambda r: round(r.evictions / base_ev.loc[(r.trace, r.frames)], 3), axis=1)
abl.to_csv(f'{{RESULTS}}/ablation_results.csv', index=False)

print('=== fault reduction vs LRU (higher = better) ===')
display(abl.pivot_table(index=['trace', 'frames'], columns='decider', values='vs_LRU'))
print('\\n=== prefetch precision ===')
display(abl.pivot_table(index=['trace', 'frames'], columns='decider', values='pf_precision'))
print('\\n=== eviction inflation vs no-prefetch (lower = gentler) ===')
display(abl.pivot_table(index=['trace', 'frames'], columns='decider', values='evict_vs_base'))

sub = abl[abl.decider != 'no-prefetch']
summary = sub.groupby('decider').agg(
    mean_gain=('vs_LRU', 'mean'), best_case=('vs_LRU', 'max'), worst_case=('vs_LRU', 'min'),
    total_harm=('vs_LRU', lambda x: x[x < 0].sum()), total_gain=('vs_LRU', lambda x: x[x > 0].sum())
).round(4).sort_values('worst_case', ascending=False)
summary.to_csv(f'{{RESULTS}}/decider_summary.csv')
print('\\n=== decider summary (worst_case closer to 0 = safer) ===')
display(summary)

fd = abl[abl.decider == 'cost-aware-fd'][['trace', 'frames', 'final_thr', 'thr_min', 'thr_max', 'mean_delta_rate',
                                          'epochs_winning', 'epochs_losing', 'pf_precision', 'vs_LRU']].copy()
fd['verdict'] = np.where(fd.epochs_losing > fd.epochs_winning, 'shut down', 'went aggressive')
print('\\n=== fault-delta controller: did it detect harm? ===')
display(fd)

corr = sub[['pf_precision', 'vs_LRU']].corr().iloc[0, 1]
ex = sub[(sub.pf_precision > 0.75) & (sub.vs_LRU < 0)][['trace', 'frames', 'decider', 'pf_precision', 'vs_LRU', 'evict_vs_base']]
print(f'\\ncorrelation(precision, fault reduction) = {{corr:.3f}};  high precision (>0.75) but harmful: {{len(ex)}} runs')
display(ex)
""")

md("""
### Cell 18 — head-to-head: prediction added on top of FIFO, LRU and Clock
`+pred` = hybrid predictor + fault-delta decider on that eviction policy. Does the gain depend on the eviction policy?
""")
code(f"""
#cell 18 — head-to-head against the traditional algorithms
{GUARD}

H2H_CONFIGS = [('437.leslie3d-s0', 21), ('410.bwaves-s0', 30), ('clustered', 9), ('bfs-3', 20), ('pr-3', 15)]
BASE_POLICIES = {{'FIFO': FIFO, 'LRU': LRU, 'Clock': Clock}}

rows = []
for tname, nf in H2H_CONFIGS:
    if tname not in TRACES:
        print(f'skipping {{tname}}'); continue
    t = TRACES[tname]
    r = {{'trace': tname, 'frames': nf}}
    for pname, P in {{**BASE_POLICIES, 'OPT': OPTFast}}.items():
        r[pname] = run_policy(t, nf, P)['faults']
    for pname, P in BASE_POLICIES.items():
        r[f'{{pname}}+pred'] = run_prefetch_sim(t, nf, P, make_predictor('hybrid'), CostAwareFaultDelta())['faults']
    rows.append(r)
    print(f'{{tname:18s}} frames={{nf:3d}} done', flush=True)

h2h = pd.DataFrame(rows).set_index(['trace', 'frames'])
h2h.to_csv(f'{{RESULTS}}/head_to_head.csv')
cols = ['FIFO', 'Clock', 'LRU', 'FIFO+pred', 'Clock+pred', 'LRU+pred', 'OPT']
print('=== page faults (lower = better) ===')
display(h2h[cols])

trad = h2h[['FIFO', 'LRU', 'Clock']].min(axis=1)
print('\\n=== LRU+pred vs the best traditional algorithm ===')
display(((trad - h2h['LRU+pred']) / trad).round(4).rename('reduction'))
gain = pd.DataFrame({{b: ((h2h[b] - h2h[f'{{b}}+pred']) / h2h[b]).round(4) for b in BASE_POLICIES}})
gain.to_csv(f'{{RESULTS}}/prediction_gain_per_policy.csv')
print('\\n=== gain from adding prediction, per eviction policy ===')
display(gain)
""")

md("""
### Cell 19 — ghost-list estimate: fault-delta feedback without a shadow engine
A real OS cannot run a second copy of memory. The ghost list remembers pages that a *prefetch* pushed out; if one faults back
in soon, that prefetch probably caused a fault. Estimated delta per epoch = prefetches used − those refaults.
`est_true_corr` = how well the estimate tracks the true per-epoch delta (the shadow engine still runs, only to log the truth).
""")
code(f"""
#cell 19 — GhostMemoryEngine, CostAwareGhost, run_prefetch_sim_ghost, and the comparison with the oracle
{GUARD}

{src(engine.GhostMemoryEngine, deciders.CostAwareGhost, simulate.run_prefetch_sim_ghost)}


rows, all_logs = [], []
for tname, nf in ABLATION_CONFIGS:
    if tname not in TRACES:
        print(f'skipping {{tname}}'); continue
    t = TRACES[tname]
    lru_ref = run_policy(t, nf, LRU)['faults']
    for label, D in (('oracle (shadow)', CostAwareFaultDelta), ('ghost estimate', CostAwareGhost)):
        s = run_prefetch_sim_ghost(t, nf, LRU, make_predictor('hybrid'), D())
        log = np.array(s['epoch_log']) if s['epoch_log'] else np.zeros((0, 2))
        corr = (np.corrcoef(log[:, 0], log[:, 1])[0, 1]
                if len(log) > 2 and log[:, 0].std() > 0 and log[:, 1].std() > 0 else np.nan)
        if label == 'ghost estimate' and len(log):
            all_logs.append(log)
        rows.append({{'trace': tname, 'frames': nf, 'decider': label, 'faults': s['faults'],
                     'vs_LRU': round((lru_ref - s['faults']) / lru_ref, 4), 'pf_used': s['pf_used'],
                     'pf_harm': s['pf_harm'], 'final_thr': s['final_thr'],
                     'est_true_corr': None if np.isnan(corr) else round(corr, 3)}})
    print(f'{{tname:18s}} frames={{nf:3d}} done', flush=True)

ghost = pd.DataFrame(rows)
ghost.to_csv(f'{{RESULTS}}/ghost_estimate_results.csv', index=False)
cmpg = ghost.pivot_table(index=['trace', 'frames'], columns='decider', values='vs_LRU')
cmpg['gap'] = (cmpg['ghost estimate'] - cmpg['oracle (shadow)']).round(4)
print('=== fault reduction vs LRU: oracle vs deployable estimate ===')
display(cmpg)
display(ghost.groupby('decider').agg(mean_gain=('vs_LRU', 'mean'), worst_case=('vs_LRU', 'min'),
                                     best_case=('vs_LRU', 'max')).round(4))
print('=== does the estimate track the true delta? (per epoch) ===')
display(ghost[ghost.decider == 'ghost estimate'][['trace', 'frames', 'pf_used', 'pf_harm', 'final_thr', 'est_true_corr']])
if all_logs:
    pooled = np.vstack(all_logs)
    print(f'pooled correlation: {{np.corrcoef(pooled[:, 0], pooled[:, 1])[0, 1]:.3f}} ({{len(pooled)}} epochs)')
if 'abl' in globals():
    ref = abl[abl.decider == 'cost-aware-fd'].set_index(['trace', 'frames'])['faults']
    ora = ghost[ghost.decider == 'oracle (shadow)'].set_index(['trace', 'frames'])['faults']
    common = ref.index.intersection(ora.index)
    print('sanity check vs cell 17:', 'MATCH' if (ref.loc[common] == ora.loc[common]).all() else 'MISMATCH')
""")

# ======================================================================================== Phase 8
md("""
## Phase 8: stronger baselines
### Cell 20 — ARC and LRU-2
Two well-known smarter eviction policies. Simplifications, stated openly: ARC drops the textbook tie-break (the engine does
not say which page is about to be installed); LRU-2 has no correlated-reference period. Checked on a scan test, where both
must beat LRU, and against OPT on every trace.
""")
code(f"""
#cell 20 — ARC, LRU-2 and their sanity checks
{GUARD}

{src(engine.ARC, engine.LRU2)}


POLICIES_EXTENDED = {{'FIFO': FIFO, 'Clock': Clock, 'LRU': LRU, 'LRU-2': LRU2, 'ARC': ARC, 'OPT': OPTFast}}

rng = np.random.default_rng(0)                          # 12 reused pages + a one-time scan
seq, scan = [], 1000
for _ in range(20_000):
    if rng.random() < 0.5:
        seq.append(int(rng.integers(0, 12)))
    else:
        seq.append(scan); scan += 1
res = {{k: run_policy(Trace('scan-test', seq), 16, P)['faults'] for k, P in POLICIES_EXTENDED.items()}}
print('scan test, 16 frames (synthetic):', res)
assert res['LRU-2'] < res['LRU'] and res['ARC'] < res['LRU'] and min(res.values()) == res['OPT']

bad = []
for name, t in TRACES.items():
    nf = frames_for(t, 0.02)
    r = {{k: run_policy(t, nf, P)['faults'] for k, P in POLICIES_EXTENDED.items()}}
    bad += [(name, k) for k in r if r[k] < r['OPT']]
print('OPT is the lower bound on every trace:', 'PASS' if not bad else f'FAIL {{bad}}')
assert not bad
""")

md("""
### Cell 21 — Linux-style readahead, and the Phase 8 comparison
Readahead: two consecutive pages start a stream; fetch a window ahead and double it each time the program reaches its
middle, up to a quarter of memory (Linux caps at 32 pages, larger than our whole simulated RAM). Then: (a) do ARC / LRU-2
alone close the LRU→OPT gap? (b) does prediction still help on ARC? (c) Markov vs readahead. Resumable.
""")
code(f"""
#cell 21 — ReadaheadPredictor + eviction-only sweep + head-to-head (saves after each configuration)
{GUARD}

{src(predictors.ReadaheadPredictor, predictors.make_readahead)}


for name, nf in (('sequential', 64), ('strided', 64), ('random', 64)):
    if name in TRACES:
        b = run_policy(TRACES[name], nf, LRU)['faults']
        g = run_prefetch_sim(TRACES[name], nf, LRU, make_readahead(nf), AlwaysPrefetch())['faults']
        print(f'readahead on {{name:10s}} (synthetic): LRU={{b:6d}}  readahead={{g:6d}}')
        assert (g < 0.05 * b) if name == 'sequential' else (abs(g - b) <= 0.01 * b)

# (a) eviction policies alone
rows = []
for name, t in TRACES.items():
    if name in ('sequential', 'strided', 'cyclic'):      # degenerate: every policy identical
        continue
    ws, seen = len(np.unique(t.pages)), set()
    for ratio in (0.01, 0.02, 0.05, 0.10):
        nf = max(8, int(ws * ratio))
        if nf in seen:
            continue
        seen.add(nf)
        rows.append({{'trace': name, 'ratio': ratio, 'frames': nf,
                     **{{p: run_policy(t, nf, P)['faults'] for p, P in POLICIES_EXTENDED.items()}}}})
ev = pd.DataFrame(rows)
ev.to_csv(f'{{RESULTS}}/phase8_eviction_only.csv', index=False)
red = ev.set_index(['trace', 'ratio', 'frames'])
print('\\n(a) eviction only, fault reduction vs LRU (positive = better than LRU):')
display(pd.DataFrame({{p: ((red.LRU - red[p]) / red.LRU).round(4) for p in ('FIFO', 'Clock', 'LRU-2', 'ARC', 'OPT')}}))

# head-to-head, resumable
CSV8 = 'phase8_head_to_head.csv'
rows = saved_rows(CSV8)
done = {{(r['trace'], int(r['frames'])) for r in rows}}
for tname, nf in H2H_CONFIGS:
    if tname not in TRACES or (tname, nf) in done:
        continue
    t = TRACES[tname]
    r = {{'trace': tname, 'frames': nf, **{{p: run_policy(t, nf, P)['faults'] for p, P in POLICIES_EXTENDED.items()}}}}
    r['RA+LRU'] = run_prefetch_sim(t, nf, LRU, make_readahead(nf), AlwaysPrefetch())['faults']
    r['RA+ARC'] = run_prefetch_sim(t, nf, ARC, make_readahead(nf), AlwaysPrefetch())['faults']
    r['Markov+LRU'] = run_prefetch_sim_ghost(t, nf, LRU, make_predictor('hybrid'), CostAwareGhost())['faults']
    r['Markov+ARC'] = run_prefetch_sim_ghost(t, nf, ARC, make_predictor('hybrid'), CostAwareGhost())['faults']
    rows.append(r)
    pd.DataFrame(rows).to_csv(f'{{RESULTS}}/{{CSV8}}', index=False)
    print(f'{{tname:18s}} frames={{nf:3d}} done', flush=True)
d = pd.DataFrame(rows).set_index(['trace', 'frames'])
print('\\npage faults (lower = better):')
display(d[['FIFO', 'Clock', 'LRU', 'LRU-2', 'ARC', 'RA+LRU', 'RA+ARC', 'Markov+LRU', 'Markov+ARC', 'OPT']])
room = d.LRU - d.OPT
print('(a) share of the LRU->OPT gap closed by a smarter eviction policy:')
display(pd.DataFrame({{'gap': (room / d.LRU).round(3), 'LRU-2 closes': ((d.LRU - d['LRU-2']) / room).round(3),
                      'ARC closes': ((d.LRU - d.ARC) / room).round(3)}}))
print('(b) gain from prediction on ARC and on LRU; (c) Markov vs readahead (positive = Markov better):')
display(pd.DataFrame({{'gain on ARC': ((d.ARC - d['Markov+ARC']) / d.ARC).round(4),
                      'gain on LRU': ((d.LRU - d['Markov+LRU']) / d.LRU).round(4),
                      'readahead vs LRU': ((d.LRU - d['RA+LRU']) / d.LRU).round(4),
                      'Markov vs readahead': ((d['RA+LRU'] - d['Markov+LRU']) / d['RA+LRU']).round(4)}}))

cols = ['FIFO', 'LRU', 'LRU-2', 'ARC', 'RA+LRU', 'Markov+LRU', 'Markov+ARC', 'OPT']
colors = ['#d9d9d9', '#9a9a9a', '#7a7a7a', '#555555', '#b9b9b9', AMBER, '#e0a458', '#1a1a1a']
ax = d[cols].div(d.LRU, axis=0).plot(kind='bar', figsize=(10.5, 4.6), width=0.82, color=colors)
ax.axhline(1.0, color='black', lw=0.7); ax.set_ylabel('faults relative to LRU (lower = better)'); ax.set_xlabel('')
ax.set_xticklabels([f'{{a}}\\n({{b}} frames)' for a, b in d.index], rotation=0, fontsize=8)
ax.set_title('Including ARC, LRU-2 and Linux-style readahead'); ax.legend(ncol=4, fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(f'{{FIG}}/fig_phase8_head_to_head.png', dpi=200); plt.show()
""")

# ======================================================================================== Phase 9-10
md("""
## Phases 9-10: readahead + delta-Markov
### Cell 22 — combining readahead with a delta-Markov predictor
Readahead beat Markov on 3 of the original 4 traces, so readahead became a **baseline to beat**. Readahead covers *new*
pages in a run (a page-number Markov model cannot predict a page it has never seen); a **delta-Markov** tier learns
repeating gap patterns anywhere in the address space. `StreamMarkovPredictor` returns the readahead window first, then the
Markov guess. Every design choice here was made on the 25% warm-up portion only. `Comb` variants use `always`.
""")
code(f"""
#cell 22 — delta-Markov, TopN, StreamMarkovPredictor, and the improvement table
{GUARD}

{src(predictors.DeltaMarkovPredictor, predictors.DeltaHybridPredictor, predictors.TopN, predictors.StreamMarkovPredictor)}


rows = []
for tname, f in (('clustered', 9), ('437.leslie3d-s0', 21), ('410.bwaves-s0', 30), ('bfs-3', 20), ('pr-3', 15)):
    if tname not in TRACES:
        print('skipping', tname); continue
    t = TRACES[tname]
    opt = run_policy(t, f, OPTFast)['faults']
    for P in (LRU, ARC):
        d2 = lambda: TopN(make_predictor('delta-hybrid'), 2)
        rows.append(dict(
            trace=tname, frames=f, policy=P.__name__, base=run_policy(t, f, P)['faults'],
            RA=run_prefetch_sim(t, f, P, make_readahead(f), AlwaysPrefetch())['faults'],
            Mk_gh=run_prefetch_sim_ghost(t, f, P, make_predictor('hybrid'), CostAwareGhost())['faults'],
            Mk_d_t2=run_prefetch_sim_ghost(t, f, P, d2(), CostAwareGhost())['faults'],
            Comb=run_prefetch_sim(t, f, P, StreamMarkovPredictor(f), AlwaysPrefetch())['faults'],
            Comb_d_t2=run_prefetch_sim(t, f, P, StreamMarkovPredictor(f, markov=d2()), AlwaysPrefetch())['faults'],
            OPT=opt))
    print(f'{{tname:18s}} done', flush=True)
imp = pd.DataFrame(rows)
imp['best_vs_base'] = 1 - imp[['RA', 'Mk_gh', 'Mk_d_t2', 'Comb', 'Comb_d_t2']].min(axis=1) / imp['base']
imp.to_csv(f'{{RESULTS}}/improvements.csv', index=False)
print('page faults (lower = better); clustered is synthetic:')
display(imp)
""")

# ======================================================================================== Phase 14-16
md("""
## Phases 14-16: the final system
### Cell 23 — shadow-directory feedback
The ghost list was biased upwards once readahead was added (on `clustered@9` it reported *winning* while prefetching was
losing). The fix: a **shadow directory**, a no-prefetch copy of the eviction policy that stores page IDs only (8 B per
tracked frame, ~0.2% of RAM), optionally hash-sampled (SHARDS, Waldspurger et al., FAST 2015). With `sample = 1` its
estimate *equals* the true per-epoch delta, which this cell checks.
""")
code(f"""
#cell 23 — CostAwareDirectory, run_prefetch_sim_dir, and ghost vs directory on the synthetic stress case
{GUARD}

{src(deciders.CostAwareDirectory, simulate.sampled, simulate.run_prefetch_sim_dir)}


rows = []
for tname, f in (('clustered', 6), ('clustered', 9)):
    if tname not in TRACES:
        continue
    t = TRACES[tname]
    lru = run_policy(t, f, LRU)['faults']
    gated = lambda: StreamMarkovPredictor(f, markov=make_predictor('delta-hybrid'), readahead=make_readahead(f, conf_k=1))
    g = run_prefetch_sim_ghost(t, f, LRU, gated(), CostAwareGhost())
    s = run_prefetch_sim_dir(t, f, LRU, gated(), CostAwareDirectory())
    assert all(e == tr for e, tr in s['epoch_log']), 'with sample=1 the directory must give the exact delta'
    rows.append({{'trace': tname, 'frames': f, 'LRU': lru, 'ghost': g['faults'], 'directory': s['faults'],
                 'ghost_vs_LRU': round((lru - g['faults']) / lru, 4), 'directory_vs_LRU': round((lru - s['faults']) / lru, 4)}})
print('readahead + delta-Markov (gated), LRU underneath (synthetic):')
display(pd.DataFrame(rows))
print('directory estimate == true delta in every epoch: PASS')
""")

md("""
### Cell 24 — the final system (frozen)
**Readahead + delta-Markov predictor → fault-delta decider fed by a shadow directory → LRU.** The controller
(`epoch=1000, step=0.1, dead_band=0.002`) and the budget (no cap) were chosen on the **warm-up portion only** (cell 28) and
then frozen. Do not retune it on the evaluation traces.
""")
code(f"""
#cell 24 — the final system in one call
{GUARD}

CONTROLLER_FROZEN = dict(epoch=1000, step=0.1, dead_band=0.002, budget_ratio=1.01)   # chosen in cell 28, warm-up only


{src(final.make_final_predictor, final.make_final_decider, final.run_final)}


for tname, f in (('437.leslie3d-s0', 21), ('clustered', 9)):
    if tname in TRACES:
        t = TRACES[tname]
        lru, s = run_policy(t, f, LRU)['faults'], run_final(t, f)
        print(f"{{tname:16s}}@{{f}}: LRU {{lru:6d}}   final {{s['faults']:6d}}   vs_LRU {{(lru - s['faults']) / lru:+.4f}}   "
              f"prefetches {{s['prefetches']}}   precision {{s['pf_precision']:.3f}}")
""")

# ======================================================================================== Phase 17-18
md("""
## Phases 17-18: evaluation on 22 real traces
### Cell 25 — robustness (no retuning)
Every system on all 22 real traces at RAM = 1% and 5% of the working set (44 points), LRU underneath. Columns:
`RA` readahead · `Markov` original hybrid + ghost · `Comb_d` readahead + delta-Markov (always) · `Comb_d_t2` with top-2 ·
`Comb_d_gated` gated by the ghost list · `Comb_d_dir` gated by the shadow directory · `Final` the frozen final system.
**Long** (roughly 1-2 hours for all 22 traces); saves after every point and resumes.
""")
code(f"""
#cell 25 — robustness on 22 real traces x {{1%, 5%}} (resumable)
{GUARD}
assert 'run_final' in globals(), 'run cells 11-24 first'

CSV_R = 'robustness.csv'
rows = saved_rows(CSV_R)
have = {{(r['trace'], r['ratio']) for r in rows}}
for name in available(REAL_ALL):
    if all((name, ratio) in have for ratio in (0.01, 0.05)):
        continue
    t = load_trace(name, limit=TRACE_LIMIT)
    ws = len(np.unique(t.pages))
    for ratio in (0.01, 0.05):
        if (name, ratio) in have:
            continue
        f = max(8, int(ws * ratio))
        mk = lambda: make_predictor('delta-hybrid')
        r = dict(trace=name, ratio=ratio, frames=f, ws=ws, LRU=run_policy(t, f, LRU)['faults'],
                 ARC=run_policy(t, f, ARC)['faults'], OPT=run_policy(t, f, OPTFast)['faults'])
        r['RA'] = run_prefetch_sim(t, f, LRU, make_readahead(f), AlwaysPrefetch())['faults']
        r['Markov'] = run_prefetch_sim_ghost(t, f, LRU, make_predictor('hybrid'), CostAwareGhost())['faults']
        r['Comb_d'] = run_prefetch_sim(t, f, LRU, StreamMarkovPredictor(f, markov=mk()), AlwaysPrefetch())['faults']
        r['Comb_d_t2'] = run_prefetch_sim(t, f, LRU, StreamMarkovPredictor(f, markov=TopN(mk(), 2)), AlwaysPrefetch())['faults']
        r['Comb_d_gated'] = run_prefetch_sim_ghost(
            t, f, LRU, StreamMarkovPredictor(f, markov=mk(), readahead=make_readahead(f, conf_k=1)), CostAwareGhost())['faults']
        r['Comb_d_dir'] = run_prefetch_sim_dir(
            t, f, LRU, StreamMarkovPredictor(f, markov=mk(), readahead=make_readahead(f, conf_k=1)), CostAwareDirectory())['faults']
        r['Final'] = run_final(t, f)['faults']
        rows.append(r)
        pd.DataFrame(rows).to_csv(f'{{RESULTS}}/{{CSV_R}}', index=False)
        print(f"{{name:18s}} {{ratio:.0%}}  frames={{f:5d}}  LRU={{r['LRU']:7d}}  RA={{r['RA']:7d}}  Final={{r['Final']:7d}}", flush=True)

rb = pd.DataFrame(rows)
SYS = ['ARC', 'RA', 'Markov', 'Comb_d', 'Comb_d_t2', 'Comb_d_gated', 'Comb_d_dir', 'Final']
vs = rb[['trace', 'ratio', 'frames']].copy()
for c in SYS + ['OPT']:
    vs[c] = ((rb.LRU - rb[c]) / rb.LRU).round(3)
print('vs_LRU (positive = fewer faults than LRU):')
display(vs)
summ = pd.DataFrame({{'mean': vs[SYS].mean(), 'median': vs[SYS].median(), 'worst': vs[SYS].min(),
                     'worse_than_LRU': (vs[SYS] < 0).sum()}}).round(3)
print(f'over {{len(rb)}} points ({{rb.trace.nunique()}} traces x {{rb.ratio.nunique()}} RAM sizes):')
display(summ)
print(f"Final vs readahead: fewer faults in {{(rb.Final < rb.RA).sum()}}, equal {{(rb.Final == rb.RA).sum()}}, "
      f"more {{(rb.Final > rb.RA).sum()}};  below OPT in {{(rb.Final < rb.OPT).sum()}} of {{len(rb)}}")
""")

md("""
### Cell 26 — break-even prefetch cost
Fault counts hide the I/O that prefetching adds. `c* = (LRU faults − faults) / prefetches` is how expensive one prefetch may
be, as a fraction of one fault's stall, before the system loses to plain LRU. `c* ≥ 1` would mean it wins even if a prefetch
stalls like a fault. Saves after each trace.
""")
code(f"""
#cell 26 — break-even cost at RAM = 1% (resumable)
{GUARD}

CSV_B = 'breakeven.csv'
rows = saved_rows(CSV_B)
done = {{r['trace'] for r in rows}}
for name in available(REAL_ALL):
    if name in done:
        continue
    t = load_trace(name, limit=TRACE_LIMIT)
    f = frames_for(t, 0.01)
    lru = run_policy(t, f, LRU)['faults']
    mk = lambda: make_predictor('delta-hybrid')
    systems = {{
        'RA':        lambda: run_prefetch_sim(t, f, LRU, make_readahead(f), AlwaysPrefetch()),
        'Markov':    lambda: run_prefetch_sim_ghost(t, f, LRU, make_predictor('hybrid'), CostAwareGhost()),
        'Comb_d':    lambda: run_prefetch_sim(t, f, LRU, StreamMarkovPredictor(f, markov=mk()), AlwaysPrefetch()),
        'Comb_d_t2': lambda: run_prefetch_sim(t, f, LRU, StreamMarkovPredictor(f, markov=TopN(mk(), 2)), AlwaysPrefetch()),
        'Final':     lambda: run_final(t, f),
    }}
    for s, run in systems.items():
        r = run()
        saved, pf = lru - r['faults'], r['prefetches']
        rows.append(dict(trace=name, frames=f, system=s, LRU=lru, faults=r['faults'], prefetches=pf,
                         precision=round(r['pf_precision'], 2), breakeven_c=round(saved / pf, 2) if pf else float('nan')))
    pd.DataFrame(rows).to_csv(f'{{RESULTS}}/{{CSV_B}}', index=False)
    print(name, 'done', flush=True)
be = pd.DataFrame(rows)
display(be.pivot(index='trace', columns='system', values='breakeven_c'))
""")

md("""
### Cell 27 — time model: does fewer faults mean faster?
Replays each run's events on an I/O timeline: compute time per access (unknown for LLC traces, so swept 0.1 / 1 / 10 µs),
prediction cost 0.2 µs per access, a 4 KB read takes 80 µs on an NVMe SSD with 8 channels or 5 ms on a one-channel HDD;
demand faults can queue behind prefetches; a prefetched page used before its read finishes stalls for the rest.
**These are model estimates with stated latencies, not measurements.** `speedup > 1` = faster than LRU. Saves after each trace.
""")
code(f"""
#cell 27 — I/O timeline replay and estimated speed-up vs LRU (resumable)
{GUARD}

DEVICES = {{'NVMe SSD': dict(t_io=80e-6, channels=8), 'HDD': dict(t_io=5e-3, channels=1)}}   # stated, not tuned
T_PRED = 0.2e-6                         # one table lookup + counter update per access
T_CPU_SWEEP = (0.1e-6, 1e-6, 10e-6)     # compute between two traced accesses: unknown, so swept


{src(timing.replay)}


SYSTEMS = {{   # name -> (runner, asks a predictor on every access?)
    'LRU':        (lambda t, f, log: run_prefetch_sim(t, f, LRU, make_predictor('hybrid'), NoPrefetch(), log=log), False),
    'Readahead':  (lambda t, f, log: run_prefetch_sim(t, f, LRU, make_readahead(f), AlwaysPrefetch(), log=log), True),
    'Markov':     (lambda t, f, log: run_prefetch_sim_ghost(t, f, LRU, make_predictor('hybrid'), CostAwareGhost(), log=log), True),
    'RA+dMarkov': (lambda t, f, log: run_prefetch_sim_dir(t, f, LRU, StreamMarkovPredictor(
        f, markov=make_predictor('delta-hybrid'), readahead=make_readahead(f, conf_k=1)), CostAwareDirectory(), log=log), True),
    'Final':      (lambda t, f, log: run_final(t, f, log=log), True),
}}
CSV_T = 'time_model.csv'
rows = saved_rows(CSV_T)
done = {{r['trace'] for r in rows}}
for name in available(REAL_ALL):
    if name in done:
        continue
    t = load_trace(name, limit=TRACE_LIMIT)
    f = frames_for(t, 0.01)
    logs = {{}}
    for s, (run, _) in SYSTEMS.items():
        logs[s] = []
        logs[s + '_n'] = run(t, f, logs[s])['accesses']
    for dev, dv in DEVICES.items():
        for tc in T_CPU_SWEEP:
            base_t = replay(logs['LRU'], logs['LRU_n'], dv['t_io'], dv['channels'], tc)['time']
            for tp in (0.0, T_PRED):                      # without and with the CPU cost of predicting
                for s, (_, pred) in SYSTEMS.items():
                    tm = replay(logs[s], logs[s + '_n'], dv['t_io'], dv['channels'], tc, tp if pred else 0.0)
                    rows.append(dict(trace=name, frames=f, device=dev, t_cpu_us=tc * 1e6, t_pred_us=tp * 1e6, system=s,
                                     time_s=tm['time'], speedup=base_t / tm['time'], late_share=tm['stall_late'] / tm['time'],
                                     faults=sum(e[0] == 'F' for e in logs[s]), prefetches=sum(e[0] == 'P' for e in logs[s])))
    pd.DataFrame(rows).to_csv(f'{{RESULTS}}/{{CSV_T}}', index=False)
    print(name, 'done', flush=True)

tm = pd.DataFrame(rows)
tm = tm[tm.system != 'LRU']
for dev in DEVICES:
    sub = tm[(tm.device == dev) & (tm.t_cpu_us == 1.0) & (tm.t_pred_us > 0)]
    print(f'{{dev}}, 1 us compute per access, 0.2 us prediction: speed-up vs LRU (> 1 = faster)')
    display(sub.pivot(index='trace', columns='system', values='speedup').round(2))
g = tm[tm.t_pred_us > 0].groupby(['device', 't_cpu_us', 'system']).speedup
print('geometric mean over traces / number of traces slower than LRU:')
display(pd.DataFrame({{'geomean': g.apply(lambda x: float(np.exp(np.log(x).mean()))),
                      'slower_than_LRU': g.apply(lambda x: int((x < 1).sum()))}}).round(3).unstack('system'))
""")

md("""
### Cell 28 — *(optional)* how the frozen controller and budget were chosen
Re-derives `CONTROLLER_FROZEN` on the **first 25% of each trace only**. Rule fixed before the run: drop settings whose worst
case is more than 0.002 below the default's; take the highest mean; keep the default unless beaten by more than 0.002.
Slow (hundreds of runs), so it is off by default; with `RUN_TUNING = False` it only shows saved results.
""")
code(f"""
#cell 28 — (optional) controller and budget tuning on the warm-up portion
{GUARD}

RUN_TUNING = False
TUNE_NAMES = ['clustered'] + REAL_ALL[:13]               # the 13 traces available when the controller was tuned
TOL = 0.002

if RUN_TUNING:
    rows = saved_rows('tuning_controller.csv')
    have = {{(r['trace'], r['epoch'], r['step'], r['dead_band']) for r in rows}}
    for name in available(TUNE_NAMES):
        full = load_trace(name, limit=TRACE_LIMIT)
        w = Trace(name, full.pages[:full.split])           # warm-up portion only
        f = frames_for(full, 0.01)
        lru = run_policy(w, f, LRU)['faults']
        for ep, st, db in itertools.product((500, 1000, 2000, 4000), (0.05, 0.1, 0.2), (0.001, 0.002, 0.005)):
            if (name, ep, st, db) in have:
                continue
            p = StreamMarkovPredictor(f, markov=make_predictor('delta-hybrid'), readahead=make_readahead(f, conf_k=1))
            r = run_prefetch_sim_dir(w, f, LRU, p, CostAwareDirectory(epoch=ep, step=st, dead_band=db))
            rows.append(dict(trace=name, frames=f, epoch=ep, step=st, dead_band=db, LRU=lru, faults=r['faults'],
                             vs_LRU=(lru - r['faults']) / lru))
        pd.DataFrame(rows).to_csv(f'{{RESULTS}}/tuning_controller.csv', index=False)
        print(name, 'done', flush=True)
    rows = []
    for name in available(['clustered'] + REAL_ALL):
        full = load_trace(name, limit=TRACE_LIMIT)
        w = Trace(name, full.pages[:full.split]); f = frames_for(full, 0.01)
        lru = run_policy(w, f, LRU)['faults']
        for b in (0.35, 0.5, 0.75, 1.01):
            ctl = {{**CONTROLLER_FROZEN, 'budget_ratio': b}}
            r = run_prefetch_sim_dir(w, f, LRU, make_final_predictor(f), CostAwareDirectory(**ctl))
            rows.append(dict(trace=name, frames=f, budget=b, LRU=lru, faults=r['faults'], prefetches=r['prefetches'],
                             vs_LRU=(lru - r['faults']) / lru))
    pd.DataFrame(rows).to_csv(f'{{RESULTS}}/tuning_budget.csv', index=False)

for fname, keys, default in (('tuning_controller.csv', ['epoch', 'step', 'dead_band'], (2000, 0.05, 0.002)),
                             ('tuning_budget.csv', ['budget'], (0.35,))):
    rows = saved_rows(fname)
    if not rows:
        print(fname, 'not produced yet (set RUN_TUNING = True)'); continue
    s = pd.DataFrame(rows).groupby(keys).vs_LRU.agg(mean='mean', worst='min')
    dflt = s.loc[default if len(default) > 1 else default[0]]
    ok = s[s.worst >= dflt.worst - TOL]
    best = ok['mean'].idxmax()
    chosen = best if ok.loc[best, 'mean'] > dflt['mean'] + TOL else default
    print(f'{{fname}} (warm-up portion only): best 8 settings by mean vs_LRU')
    display(s.sort_values('mean', ascending=False).head(8).round(4))
    print('default:', default, ' chosen:', chosen)
""")

md("""
### Cell 29 — component comparison: one component changed at a time
**Predictors** (stride, markov-o1, markov-o2, hybrid, delta-hybrid, readahead, and the final combination), all through the
same final decision layer; **deciders** (9 + the final one), all fed the same predictions (once from the original Markov
hybrid, once from the final predictor). Every real trace at 1% plus synthetic `clustered` as the danger test; LRU underneath.
Summaries average over the **real traces only** and report `clustered` in its own column. Long; resumable per trace.
""")
code(f"""
#cell 29 — every predictor and every decider, side by side (resumable)
{GUARD}

PREDICTORS = {{
    'stride': lambda f: make_predictor('stride'), 'markov-o1': lambda f: make_predictor('markov-o1'),
    'markov-o2': lambda f: make_predictor('markov-o2'), 'hybrid': lambda f: make_predictor('hybrid'),
    'delta-hybrid': lambda f: make_predictor('delta-hybrid'), 'readahead': lambda f: make_readahead(f, conf_k=1),
    'final (readahead + delta-hybrid)': lambda f: make_final_predictor(f),
}}
DECIDERS_ALL = {{   # name -> (simulator, decider factory)
    'no-prefetch': (run_prefetch_sim, NoPrefetch), 'always': (run_prefetch_sim, AlwaysPrefetch),
    'fixed-0.3': (run_prefetch_sim, lambda: FixedThreshold(0.3)), 'fixed-0.6': (run_prefetch_sim, lambda: FixedThreshold(0.6)),
    'fixed-0.85': (run_prefetch_sim, lambda: FixedThreshold(0.85)), 'precision (cost-aware)': (run_prefetch_sim, CostAware),
    'fault-delta oracle': (run_prefetch_sim, CostAwareFaultDelta), 'ghost list': (run_prefetch_sim_ghost, CostAwareGhost),
    'shadow directory': (run_prefetch_sim_dir, CostAwareDirectory),
    'directory + frozen controller (final)': (run_prefetch_sim_dir, make_final_decider),
}}
DECIDER_PREDICTORS = {{'Markov hybrid (original)': lambda f: make_predictor('hybrid'),
                      'final predictor': lambda f: make_final_predictor(f)}}


def timed(sim, t, f, pred, dec):
    log = []
    r = sim(t, f, LRU, pred, dec, log=log)
    return r, {{dev: replay(log, r['accesses'], dv['t_io'], dv['channels'], 1e-6, T_PRED)['time'] for dev, dv in DEVICES.items()}}


rp, rd = saved_rows('component_predictors.csv'), saved_rows('component_deciders.csv')
done = {{r['trace'] for r in rp}} & {{r['trace'] for r in rd}}
for name in available(['clustered'] + REAL_ALL):
    if name in done:
        continue
    t = load_trace(name, limit=TRACE_LIMIT)
    f = frames_for(t, 0.01)
    b, bsp = timed(run_prefetch_sim, t, f, make_predictor('hybrid'), NoPrefetch())
    lru = b['faults']
    rp = [r for r in rp if r['trace'] != name]; rd = [r for r in rd if r['trace'] != name]
    for pn, mk in PREDICTORS.items():
        q = eval_predictor(t, mk(f))
        r, sp = timed(run_prefetch_sim_dir, t, f, mk(f), make_final_decider())
        rp.append(dict(trace=name, frames=f, predictor=pn, coverage=q['coverage'], accuracy=q['accuracy'], useful=q['useful'],
                       LRU=lru, faults=r['faults'], vs_LRU=(lru - r['faults']) / lru, prefetches=r['prefetches'],
                       precision=r['pf_precision'], **{{f'speedup_{{d}}': bsp[d] / sp[d] for d in DEVICES}}))
    for pl, mkp in DECIDER_PREDICTORS.items():
        for dn, (sim, mkd) in DECIDERS_ALL.items():
            r, sp = timed(sim, t, f, mkp(f), mkd())
            rd.append(dict(trace=name, frames=f, predictions=pl, decider=dn, LRU=lru, faults=r['faults'],
                           vs_LRU=(lru - r['faults']) / lru, prefetches=r['prefetches'], precision=r['pf_precision'],
                           **{{f'speedup_{{d}}': bsp[d] / sp[d] for d in DEVICES}}))
    pd.DataFrame(rp).to_csv(f'{{RESULTS}}/component_predictors.csv', index=False)
    pd.DataFrame(rd).to_csv(f'{{RESULTS}}/component_deciders.csv', index=False)
    print(name, 'done', flush=True)


def summarise(df, key):
    real = df[df.trace != 'clustered']
    g = real.groupby(key, sort=False)
    return pd.DataFrame({{
        'mean_vs_LRU': g.vs_LRU.mean(), 'median_vs_LRU': g.vs_LRU.median(), 'worst_vs_LRU': g.vs_LRU.min(),
        'worse_than_LRU': g.vs_LRU.apply(lambda x: int((x < -0.0005).sum())),
        'prefetch_per_access': g.prefetches.mean() / TRACE_LIMIT, 'precision': g.precision.mean(),
        'NVMe_speedup': g['speedup_NVMe SSD'].apply(lambda x: float(np.exp(np.log(x).mean()))),
        'HDD_speedup': g['speedup_HDD'].apply(lambda x: float(np.exp(np.log(x).mean()))),
        'clustered_vs_LRU': df[df.trace == 'clustered'].set_index(key).vs_LRU if 'clustered' in set(df.trace) else np.nan}})


P, D = pd.DataFrame(rp), pd.DataFrame(rd)
sp = P[P.trace != 'clustered'].groupby('predictor', sort=False)[['coverage', 'accuracy', 'useful']].mean().join(summarise(P, 'predictor'))
sp.to_csv(f'{{RESULTS}}/component_predictors_summary.csv')
print(f'PREDICTORS, same final decider ({{P.trace.nunique() - ("clustered" in set(P.trace))}} real traces at 1%):')
display(sp.round(3))
sd = pd.concat({{pl: summarise(D[D.predictions == pl], 'decider') for pl in DECIDER_PREDICTORS}}, names=['predictions'])
sd.to_csv(f'{{RESULTS}}/component_deciders_summary.csv')
for pl in DECIDER_PREDICTORS:
    print(f'\\nDECIDERS fed by: {{pl}}')
    display(sd.loc[pl].round(3))
""")

md("""
### Cell 30 — results at a glance
Built from the saved files only (no new simulation): the final system against readahead, the original Markov system and
OPT on every real trace, and the estimated time. What to look for: the final system has the most fault reduction, but in
estimated time it roughly ties readahead; on a one-channel HDD every prefetcher is slower than LRU.
""")
code(f"""
#cell 30 — headline table and figures from robustness.csv and time_model.csv
{GUARD}

rb = pd.DataFrame(saved_rows('robustness.csv'))
if len(rb):
    v = rb[['trace', 'ratio', 'frames']].copy()
    for c in ('RA', 'Markov', 'Final', 'OPT'):
        v[c] = (rb.LRU - rb[c]) / rb.LRU
    head = v.groupby('ratio')[['RA', 'Markov', 'Final']].agg(['mean', 'min']).round(3)
    print('fault reduction vs LRU: mean and worst over traces, per RAM size')
    display(head)
    one = v[v.ratio == 0.01].set_index('trace').sort_values('Final')
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(one)); w = 0.27
    ax.bar(x - w, one.Markov, w, color='#bbbbbb', label='Markov (original)')
    ax.bar(x, one.RA, w, color='#666666', label='Linux-style readahead')
    ax.bar(x + w, one.Final, w, color=AMBER, label='final system')
    ax.axhline(0, color='black', lw=0.8); ax.set_xticks(x); ax.set_xticklabels(one.index, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('fault reduction vs LRU'); ax.set_title('RAM = 1% of working set, 22 real traces')
    ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig(f'{{FIG}}/fig_final_vs_baselines.png', dpi=200); plt.show()
else:
    print('robustness.csv not produced yet: run cell 25')

tmr = pd.DataFrame(saved_rows('time_model.csv'))
if len(tmr):
    s = tmr[(tmr.t_pred_us > 0) & (tmr.system != 'LRU')]
    g = s.groupby(['device', 't_cpu_us', 'system']).speedup.apply(lambda x: float(np.exp(np.log(x).mean()))).unstack('system')
    print('estimated speed-up vs LRU (geometric mean over traces; model, not measurement):')
    display(g.round(2))
""")

md("""
### Cell 31 — consistency check: do the result files agree with each other?
Many cells recompute the same baseline (LRU, OPT, ...) or rerun the same system. If two files disagree, something changed
between runs (code, trace or RAM size) and those numbers must not be quoted side by side. This cell reads files only.
* `tuning_*.csv` are left out on purpose: they use the 25% warm-up portion, so their LRU counts are smaller.
* "readahead" in cell 29 runs *through the final decider*; the `RA` baseline in cell 25 does not, so they can differ.
* Prefetching systems may have fewer faults than OPT (OPT is the best *demand* policy); eviction-only policies may not.
""")
CHECK = open(os.path.join(ROOT, 'scripts', 'consistency_cell.py'), encoding='utf-8').read()
code(f"""
#cell 31 — consistency check across all result files (reads files only)
{GUARD}

{CHECK.strip()}
""")

md("""
## What to check before quoting any number
* Which real traces were present (cell 0c). Results from synthetic traces only must be labelled synthetic.
* The baseline checks (cell 14) and every consistency check (cell 31) passed.
* Nothing was tuned on evaluation data: `FROZEN` (cell 4) and `CONTROLLER_FROZEN` (cell 24) were chosen on the warm-up portion.
* Results are page-fault counts on LLC load traces at page granularity; the time model (cell 27) uses stated, untuned latencies.
* State the simplifications: ARC without the textbook tie-break, LRU-2 without a correlated-reference period, readahead window = RAM/4.
""")


# =====================================================================================================
def to_nb(cells):
    out = []
    for kind, text in cells:
        c = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {'cells': out,
            'metadata': {'colab': {'provenance': [], 'toc_visible': True},
                         'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                         'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 0}


if __name__ == '__main__':
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(to_nb(CELLS), f, indent=1, ensure_ascii=False)
    n_code = sum(k == 'code' for k, _ in CELLS)
    print(f'wrote {OUT}: {len(CELLS)} cells ({n_code} code)')
