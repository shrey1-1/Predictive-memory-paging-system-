"""
Append Phases 8-18 to the user's original Colab notebook (os-predictive-memory-paging-system.ipynb), after its cell 19.
The original cells and their outputs are kept unchanged; new cells 20-34 are appended in the same style.

    python scripts/extend_original_notebook.py <original.ipynb> <output.ipynb>

The appended part is self-contained (cell 20 re-mounts Drive and cell 21 defines the final versions of the core
classes), so it also runs after a Colab reconnect without re-running cells 1-19. The phase cells are taken from
scripts/build_notebook.py (code copied from src/paging), renumbered to follow cell 19.
"""
import copy, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_notebook as bn  # noqa: E402

from paging import traces, predictors, engine, deciders, simulate  # noqa: E402  (path set by build_notebook)

GEN = bn.CELLS
src = bn.src
GUARD = "assert 'TRACE_DIR' in globals(), 'Run cell 20 (setup for Phases 8-18) first'"

# generated-notebook cell number -> number in the extended original notebook
RENUMBER = {'0c': '22', '20': '23', '21': '24', '22': '25', '23': '26', '24': '27', '25': '28', '26': '29',
            '27': '30', '28': '31', '29': '32', '31': '34'}


def renumber(text):
    text = text.replace("Run cell 0a (setup) first", "Run cell 20 (setup for Phases 8-18) first")
    text = text.replace('run cells 11-24 first', 'run cells 20-27 first')
    text = text.replace('cells (21, 25, 26, 27, 29)', 'cells (24, 28, 29, 30, 32)')
    return re.sub(r'\b(cell|Cell) (0c|2[0-9]|31)\b', lambda m: f'{m.group(1)} {RENUMBER.get(m.group(2), m.group(2))}', text)


def gen_pair(num):
    """The (markdown, code) pair of generated cell `num`, renumbered."""
    for i, (kind, text) in enumerate(GEN):
        if kind == 'code' and text.startswith(f'#cell {num} —'):
            assert GEN[i - 1][0] == 'markdown'
            return [('markdown', renumber(GEN[i - 1][1])), ('code', renumber(text))]
    raise KeyError(num)


def gen_code(num):
    for kind, text in GEN:
        if kind == 'code' and text.startswith(f'#cell {num} —'):
            return text
    raise KeyError(num)


NEW = []
NEW.append(('markdown', """
---
# Phases 8-18: from the Markov prefetcher to the final system

Cells 1-19 above are the original Stages 3-5. The cells below add everything that came after, in the same style:

| Cell | Phase | What it does |
|---|---|---|
| 20 | | setup for this part (re-mounts Drive, folders, helpers, reloads `TRACES`) |
| 21 | | the core classes in their final form; checks that cell 17 is reproduced exactly |
| 22 | 17 | download the other 18 real traces (22 in total) |
| 23-24 | 8 | ARC, LRU-2, Linux-style readahead |
| 25 | 9-10 | readahead + delta-Markov |
| 26-27 | 14-16 | shadow-directory feedback; **the final system** |
| 28-32 | 17-18 | 22 real traces, break-even cost, time model, (optional) tuning, component comparison |
| 33 | | **comparison graphs of the final work** |
| 34 | | consistency check of every result file |

This part is self-contained: after a Colab reconnect, run cell 20 and cell 21, then any later cell. The long cells
(24, 28, 29, 30, 32) save after each trace and resume where they stopped.
""".strip('\n')))

# ---------------------------------------------------------------------------------------------- 20 setup
setup = gen_code('0a')
setup = setup.replace('#cell 0a — setup: mount Drive, folders, constants, imports, plot style (run first, every time)',
                      '#cell 20 — setup for Phases 8-18: mount Drive, folders, helpers, reload TRACES (run first after a reconnect)')
setup = setup.replace("RESULTS   = f'{BASE}/results'        # every CSV / JSON this notebook writes",
                      "RESULTS   = BASE                     # same place as cells 1-19 (ablation_results.csv, ...)")
setup = setup.replace("FIG       = f'{RESULTS}/figures'", "FIG       = f'{BASE}/figures'")
assert 'RESULTS   = BASE' in setup and "FIG       = f'{BASE}/figures'" in setup
trace_cls = src(traces.Trace)
NEW.append(('markdown', """
### Cell 20 — setup for Phases 8-18
Same Drive folder as cells 1-19: traces in `paging_sim/traces/`, result files next to the earlier ones in `paging_sim/`,
figures in `paging_sim/figures/`. Also reloads the Stage 3-5 traces as `TRACES` (six synthetic + the original four real).
""".strip('\n')))
NEW.append(('code', setup + f"""


{trace_cls}


def load_trace(name, trace_dir=None, limit=None, train_frac=WARMUP_FRAC):
    df = pd.read_csv(f'{{trace_dir or TRACE_DIR}}/{{name}}.csv')
    pages = df['page_id'].values
    if limit:
        pages = pages[:limit]
    return Trace(name, pages, train_frac)


TRACES = {{n: load_trace(n, limit=TRACE_LIMIT) for n in available(SYNTHETIC + PICK)}}
REAL = [n for n in PICK if n in TRACES]
ABLATION_CONFIGS = [('437.leslie3d-s0', 21), ('410.bwaves-s0', 30), ('clustered', 6), ('clustered', 9),
                    ('bfs-3', 20), ('bfs-3', 58), ('pr-3', 15), ('pr-3', 40)]
H2H_CONFIGS = [('437.leslie3d-s0', 21), ('410.bwaves-s0', 30), ('clustered', 9), ('bfs-3', 20), ('pr-3', 15)]
print(f'TRACES: {{len(TRACES)}} loaded ({{len(REAL)}} real)')
"""))

# ---------------------------------------------------------------------------------------------- 21 core
NEW.append(('markdown', """
### Cell 21 — the core classes in their final form
Phases 8-18 need a few additions to the classes of cells 1-19: an optional event log in the memory engine (for the time
model), a shared threshold controller in the fault-delta decider (reused by cells 19 and 26), and the `delta-hybrid`
predictor kind. With these additions switched off, behaviour is **identical** to cells 1-19; the check at the end of this
cell reruns two configurations of cell 17 and compares the fault counts.
""".strip('\n')))
NEW.append(('code', f"""
#cell 21 — final versions of the predictors, memory engine, policies, deciders and simulators
{GUARD}

{src(predictors.MarkovPredictor, predictors.StridePredictor, predictors.HybridPredictor)}


{src(simulate.eval_predictor)}


FROZEN = dict(capacity=256, max_succ=2, order=2, decay_every=25_000)      # frozen in cell 4; unchanged
KINDS = ['stride', 'markov-o1', 'markov-o2', 'hybrid']


{src(predictors.make_predictor)}


{src(engine.MemoryEngine, engine.FIFO, engine.LRU, engine.Clock, engine.OPTFast)}


POLICIES = {{'FIFO': FIFO, 'LRU': LRU, 'Clock': Clock, 'OPT': OPTFast}}


{src(engine.run_policy)}


{src(deciders.NoPrefetch, deciders.AlwaysPrefetch, deciders.FixedThreshold, deciders.CostAware,
     deciders.CostAwareFaultDelta, simulate.run_prefetch_sim)}


{src(engine.GhostMemoryEngine, deciders.CostAwareGhost, simulate.run_prefetch_sim_ghost)}


# Check: the final classes reproduce cell 17 exactly (fault-delta decider, hybrid predictor, LRU).
REFERENCE = {{('437.leslie3d-s0', 21): 9956, ('clustered', 9): 5831}}    # cell 17, cost-aware-fd (docs/RESULTS.md section 3)
for (tname, nf), expected in REFERENCE.items():
    if tname not in TRACES:
        continue
    got = run_prefetch_sim(TRACES[tname], nf, LRU, make_predictor('hybrid'), CostAwareFaultDelta())['faults']
    if 'abl' in globals():                     # this session also ran cell 17: compare with it directly
        expected = int(abl[(abl.trace == tname) & (abl.frames == nf) & (abl.decider == 'cost-aware-fd')].faults.iloc[0])
    assert got == expected, f'{{tname}}@{{nf}}: {{got}} faults, expected {{expected}}: the classes changed behaviour'
    print(f'{{tname}}@{{nf}}: {{got}} faults, same as cell 17')
print('core classes ready')
"""))

# ---------------------------------------------------------------------------------------------- 22 download
(_, md0c), (_, code0c) = gen_pair('0c')
code0c = code0c.replace('#cell 22 — download and convert the real traces (skips any already on Drive)',
                        '#cell 22 — download and convert all 22 real traces (skips the ones already on Drive)')
NEW += [('markdown', md0c.replace('### Cell 22 — Stage 1b: real traces (22 ML-DPC LoadTraces)',
                                  '### Cell 22 — the other real traces (Phase 17: 22 in total)')), ('code', code0c)]

for num in ('20', '21', '22', '23'):
    NEW += gen_pair(num)

# ---------------------------------------------------------------------------------------------- 27 final + classic configs
(_, md24), (_, code24) = gen_pair('24')
code24 += """


# The final system on the classic configurations of cells 18 and 24, for the comparison graphs (cell 33).
rows = []
for tname, f in H2H_CONFIGS:
    if tname in TRACES:
        s = run_final(TRACES[tname], f)
        rows.append({'trace': tname, 'frames': f, 'Final': s['faults'], 'prefetches': s['prefetches'],
                     'precision': round(s['pf_precision'], 3)})
pd.DataFrame(rows).to_csv(f'{RESULTS}/final_classic_configs.csv', index=False)
display(pd.DataFrame(rows))"""
NEW += [('markdown', md24), ('code', code24)]

for num in ('25', '26', '27', '28', '29'):
    NEW += gen_pair(num)

# ---------------------------------------------------------------------------------------------- 33 graphs
GRAPHS = open(os.path.join(HERE, 'comparison_graphs_cell.py'), encoding='utf-8').read().strip()
NEW.append(('markdown', """
### Cell 33 — comparison graphs of the final work
Built from the saved result files only (nothing is re-simulated). Colours: **amber = the final system**, dark grey =
Linux-style readahead (the baseline it has to beat), mid grey = the other baselines, black outline = OPT. Every figure is
saved to `paging_sim/figures/` and followed by its data table.

1. Classic configurations: every algorithm against the final system (faults relative to LRU)
2. The 22 real traces: final system vs readahead vs Markov (RAM = 1%)
3. How the system evolved: mean and worst-case fault reduction over 44 points
4. Final vs readahead at both memory sizes (one dot per trace)
5. Estimated time: speed-up vs LRU on an SSD and a disk (model, not measurement)
6. Break-even prefetch cost per trace
7. Accuracy is not fault reduction (predictors) · benefit vs danger (decision engines)
""".strip('\n')))
NEW.append(('code', '#cell 33 — comparison graphs of the final work (reads result files only)\n' + GUARD + '\n\n' + GRAPHS))

NEW += gen_pair('31')


# ---------------------------------------------------------------------------------------------- write
def to_cell(kind, text):
    c = {'cell_type': kind, 'metadata': {}, 'source': text.strip(chr(10)).splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def extend(original_path):
    nb = json.load(open(original_path, encoding='utf-8'))
    nb = copy.deepcopy(nb)
    nb['cells'] += [to_cell(k, t) for k, t in NEW]
    return nb


if __name__ == '__main__':
    orig, out = sys.argv[1], sys.argv[2]
    nb = extend(orig)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'{out}: {len(nb["cells"])} cells ({len(NEW)} appended)')
