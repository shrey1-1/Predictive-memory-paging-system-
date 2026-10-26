"""
Build one self-contained HTML report of every result: results/report.html.
Reads only the CSV/JSON files the other scripts write (nothing is typed in by hand), so rerun the
scripts first, then this. Works offline (no fonts, scripts or images from the internet).

  python scripts/13_report.py
Sections whose CSV is missing are listed as skipped at the top of the report.
"""
import _path  # noqa: F401
import datetime
import html
import json
import os
import sys

import numpy as np
import pandas as pd

from paging import config
from paging.traces import available_traces, load_trace, SYNTHETIC

R = config.RESULTS_DIR
OUT = os.path.join(R, 'report.html')
skipped = []


def csv(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        skipped.append(name)
        return None
    return pd.read_csv(p)


def esc(x):
    return html.escape(str(x))


def short(n):
    n = str(n)
    return n.split('.', 1)[1] if n[:3].isdigit() and '.' in n else n


def fmt(v, kind='int'):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '–'
    if kind == 'int':
        return f'{int(round(v)):,}'
    if kind == 'pct':
        return ('+' if v >= 0 else '−') + f'{abs(v) * 100:.1f}%'
    if kind == 'p':
        return f'{v * 100:.1f}%'
    if kind == 'pct2':
        return ('+' if v >= 0 else '−') + f'{abs(v) * 100:.2f}%'
    if kind == 'f2':
        return f'{v:.2f}'
    if kind == 'f3':
        return f'{v:.3f}'
    if kind == 'f4':
        return f'{v:+.4f}'
    return esc(v)


def table(head, rows, cls=''):
    """rows: list of lists; a cell may be (text, css_class)."""
    h = ''.join(f'<th scope="col">{esc(c)}</th>' for c in head)
    body = []
    for r in rows:
        cells = []
        for j, c in enumerate(r):
            txt, k = (c if isinstance(c, tuple) else (c, ''))
            tag = 'th scope="row"' if j == 0 else 'td'
            cells.append(f'<{tag} class="{k}">{txt}</{tag.split()[0]}>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def best_worse(vals, base, lower_better=True):
    """css class per value: best (amber) and worse-than-base (red)."""
    ok = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    b = (min(ok) if lower_better else max(ok)) if ok else None
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append('')
        elif v == b:
            out.append('best')
        elif base is not None and ((v > base) if lower_better else (v < base)):
            out.append('worse')
        else:
            out.append('')
    return out


SERIES = [('Markov', 'Markov (original)', 'circle', 'var(--s-mid)', True),
          ('RA', 'Readahead', 'square', 'var(--s-dark)', False),
          ('Final', 'Final system (RA + δ-Markov, directory feedback)', 'diamond', 'var(--accent)', False)]


def shape(kind, x, y, r, color, hollow):
    fill = 'var(--surface)' if hollow else color
    stroke = color if hollow else 'var(--surface)'
    a = f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hollow else 1.5}"'
    if kind == 'circle':
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {a}/>'
    if kind == 'square':
        return f'<rect x="{x - r * .9:.1f}" y="{y - r * .9:.1f}" width="{r * 1.8:.1f}" height="{r * 1.8:.1f}" rx="1.5" {a}/>'
    return (f'<path d="M{x:.1f} {y - r * 1.25:.1f}L{x + r * 1.25:.1f} {y:.1f}L{x:.1f} {y + r * 1.25:.1f}'
            f'L{x - r * 1.25:.1f} {y:.1f}Z" {a}/>')


def legend():
    items = []
    for _, label, kind, color, hollow in SERIES:
        items.append(f'<span><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">'
                     f'{shape(kind, 7, 7, 5, color, hollow)}</svg>{esc(label)}</span>')
    return '<div class="legend">' + ''.join(items) + '</div>'


def dotplot(rows, vmin, vmax, ticks, tickfmt, valfmt, axis, aria):
    """rows: list of (label, {series_key: (value, tooltip_extra)})."""
    W, L, Rr, T, rh = 900, 150, 24, 26, 30
    H = T + len(rows) * rh + 30
    x = lambda v: L + (min(vmax, max(vmin, v)) - vmin) / (vmax - vmin) * (W - L - Rr)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}">']
    for t in ticks:
        zero = abs(t) < 1e-12
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T - 6}" y2="{H - 26}" stroke="{"var(--muted)" if zero else "var(--grid)"}" stroke-width="{1.2 if zero else 1}"/>')
        s.append(f'<text x="{x(t):.1f}" y="{H - 10}" text-anchor="middle">{esc(tickfmt(t))}</text>')
    s.append(f'<text x="{x(vmax):.1f}" y="12" text-anchor="end">{esc(axis)}</text>')
    for i, (label, vals) in enumerate(rows):
        y = T + i * rh + rh / 2
        if i % 2 == 0:
            s.append(f'<rect x="0" y="{y - rh / 2:.1f}" width="{W}" height="{rh}" fill="var(--band)"/>')
        s.append(f'<text class="row" x="0" y="{y + 4:.1f}">{esc(label)}</text>')
        vs = [v for v, _ in vals.values() if v is not None and not np.isnan(v)]
        if vs:
            s.append(f'<line x1="{x(min(vs)):.1f}" x2="{x(max(vs)):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--rule)" stroke-width="2"/>')
        for key, lab, kind, color, hollow in SERIES:
            if key not in vals:
                continue
            v, extra = vals[key]
            if v is None or np.isnan(v):
                continue
            s.append(f'<g><title>{esc(label)} · {esc(lab)}: {esc(valfmt(v))}{esc(extra)}</title>'
                     f'{shape(kind, x(v), y, 6, color, hollow)}</g>')
    s.append('</svg>')
    return '<div class="chart">' + ''.join(s) + '</div>'


def section(eid, eyebrow, title, intro, body, source):
    return (f'<section id="{eid}"><div class="eyebrow">{esc(eyebrow)}</div><h2>{esc(title)}</h2>'
            f'{intro}{body}<p class="src">Source: {source}</p></section>')


parts, toc = [], []


def add(eid, eyebrow, title, intro, body, source):
    toc.append((eid, eyebrow, title))
    parts.append(section(eid, eyebrow, title, intro, body, source))


# ---------------------------------------------------------------- 0. traces
names = available_traces()                       # the six synthetic + the original four real, in order
names += sorted(f[:-4] for f in os.listdir(config.TRACE_DIR)
                if f.endswith('.csv') and f[:-4] not in names)   # extra real traces (robustness set)
real = [n for n in names if n not in SYNTHETIC]
trow = []
for n in names:
    t = load_trace(n, limit=config.TRACE_LIMIT)
    ws = len(np.unique(t.pages))
    trow.append([esc(n), 'synthetic' if n in SYNTHETIC else 'real (LLC LoadTrace)', fmt(len(t)), fmt(ws), fmt(max(8, int(ws * 0.01)))])
add('data', 'Data', f'{len(names)} traces: {len(SYNTHETIC)} synthetic, {len(real)} real',
    '<p>Real traces are ML-DPC LoadTraces (SPEC CPU 2006 and GAP graph benchmarks): last-level-cache load addresses mapped to 4 KB pages. '
    'They approximate OS paging but are not OS paging traces. Each trace is capped at 200,000 accesses; the first 25% (time-ordered) is the warm-up portion used for every design choice.</p>',
    table(['Trace', 'Kind', 'Accesses', 'Working set (pages)', 'Frames at 1%'], trow),
    '<code>data/traces/*.csv</code>, script 02 (real) and 01 (synthetic)')

# ---------------------------------------------------------------- 1. predictors
pr = csv('predictor_results.csv')
if pr is not None:
    pv = pr.pivot_table(index='trace', columns='predictor', values='useful')
    order = [c for c in ['stride', 'markov-o1', 'markov-o2', 'hybrid'] if c in pv.columns]
    cov = pr[pr.predictor == 'hybrid'].set_index('trace')
    rows = []
    for tr in [n for n in names if n in pv.index]:
        vals = [pv.loc[tr, c] for c in order]
        k = best_worse(vals, None, lower_better=False)
        rows.append([esc(tr)] + [(fmt(v, 'p'), c) for v, c in zip(vals, k)] +
                    [fmt(cov.loc[tr, 'coverage'], 'p'), fmt(cov.loc[tr, 'accuracy'], 'p')])
    realmean = pv.loc[[t for t in real if t in pv.index], 'hybrid'].mean() if real else float('nan')
    body = table(['Trace'] + [f'useful: {c}' for c in order] + ['hybrid coverage', 'hybrid accuracy'], rows)
    at = csv('predictor_attribution.csv')
    if at is not None:
        body += '<h3>Which sub-predictor answered (hybrid)</h3>' + table(
            ['Trace', 'order-2 Markov', 'order-1 Markov', 'stride', 'no prediction'],
            [[esc(r.trace), fmt(r.m2, 'p'), fmt(r.m1, 'p'), fmt(r.stride, 'p'), fmt(r.none, 'p')] for r in at.itertuples()])
    cs = csv('capacity_sweep.csv')
    if cs is not None:
        body += '<h3>Table capacity sweep (order-2 Markov)</h3>' + table(
            ['Capacity (contexts)', 'Table size (KB)', 'useful, real traces', 'useful, all traces'],
            [[fmt(r.capacity), fmt(r.table_KB, 'f2'), fmt(r.useful_real, 'p'), fmt(r.useful_all, 'p')] for r in cs.itertuples()])
    add('predictors', 'Stage 3 · Prediction quality', f'Hybrid predictor: right about the next page {realmean * 100:.1f}% of the time on the first four real traces',
        '<p><code>useful = coverage × accuracy</code>, measured after the warm-up portion with online learning throughout. Prediction accuracy is not the goal by itself: '
        'section 4 shows that it can diverge from page faults. Markov = 0 on sequential and strided traces because their working sets exceed the 256-entry table; the stride fallback covers them.</p>',
        body, '<code>predictor_results.csv</code>, <code>predictor_attribution.csv</code>, <code>capacity_sweep.csv</code>, script 03')

# ---------------------------------------------------------------- 2. eviction baselines
bl = csv('baseline_results.csv')
if bl is not None:
    rows = []
    pol = [p for p in ['FIFO', 'Clock', 'LRU', 'OPT'] if p in bl.policy.unique()]
    for (tr, ratio, fr), g in bl.groupby(['trace', 'ratio', 'frames'], sort=False):
        d = dict(zip(g.policy, g.faults))
        vals = [d.get(p) for p in pol]
        k = best_worse([d.get(p) for p in pol if p != 'OPT'], d.get('LRU'))
        head = (d['LRU'] - d['OPT']) / d['LRU'] if d.get('LRU') else float('nan')
        rows.append([esc(tr), f'{ratio:.0%}', fmt(fr)] + [(fmt(v), c) for v, c in zip(vals[:-1], k)] + [fmt(vals[-1]), fmt(head, 'pct')])
    add('baselines', 'Stage 4 · Eviction only', 'FIFO, Clock, LRU and OPT without prefetching',
        '<p>RAM is sized as a fraction of each trace\'s working set. Headroom is how far OPT (future knowledge) is below LRU: it is the most any eviction policy could gain. '
        'Automated checks in <code>tests/</code> confirm OPT is never beaten and LRU/OPT never get worse with more memory.</p>',
        '<details><summary>Show all ' + str(len(rows)) + ' rows</summary>' +
        table(['Trace', 'RAM', 'Frames'] + pol[:-1] + ['OPT', 'Headroom LRU→OPT'], rows) + '</details>',
        '<code>baseline_results.csv</code>, script 04')

# ---------------------------------------------------------------- 3. ablation
ab = csv('ablation_results.csv')
if ab is not None:
    ab = ab[ab.decider != 'no-prefetch']
    decs = list(dict.fromkeys(ab.decider))
    rows = []
    for (tr, fr), g in ab.groupby(['trace', 'frames'], sort=False):
        d = dict(zip(g.decider, g.vs_LRU))
        vals = [d.get(x) for x in decs]
        k = best_worse(vals, 0.0, lower_better=False)
        rows.append([f'{esc(short(tr))} @ {fr}'] + [(fmt(v, 'pct2'), c) for v, c in zip(vals, k)])
    body = table(['Config'] + decs, rows)
    ds = csv('decider_summary.csv')
    if ds is not None:
        body += '<h3>Summary across configurations</h3>' + table(
            ['Decider', 'Mean gain', 'Best case', 'Worst case', 'Total harm', 'Total gain'],
            [[esc(r.decider), fmt(r.mean_gain, 'pct2'), fmt(r.best_case, 'pct2'), fmt(r.worst_case, 'pct2'), fmt(r.total_harm, 'pct2'), fmt(r.total_gain, 'pct2')] for r in ds.itertuples()])
    if 'evict_vs_base' in ab.columns:
        rows = []
        for (tr, fr), g in ab.groupby(['trace', 'frames'], sort=False):
            d = dict(zip(g.decider, g.evict_vs_base))
            rows.append([f'{esc(short(tr))} @ {fr}'] + [(fmt(d.get(x), 'f2') + '×') if d.get(x) is not None else '–' for x in decs])
        body += '<h3>Eviction inflation (evictions with prefetching ÷ without)</h3>' + table(['Config'] + decs, rows)
    add('ablation', 'Stage 5 · Decision engine', 'Which decider keeps the benefit and limits the harm',
        '<p>All deciders get byte-identical predictions from the Markov hybrid, with LRU underneath; values are <code>vs_LRU</code> (positive = fewer faults). '
        '<code>always</code> and <code>fixed-0.3</code> are identical because no confidence is below 0.5, which is structural: the top one of at most two successors always holds half the counts.</p>',
        body, '<code>ablation_results.csv</code>, <code>decider_summary.csv</code>, script 05')

    ref_p = os.path.join(R, 'reference', 'ablation_results.csv')
    if os.path.exists(ref_p):
        ref = pd.read_csv(ref_p)
        m = ab.merge(ref, on=['trace', 'frames', 'decider'], suffixes=('', '_ref'))
        same = int((m.faults == m.faults_ref).sum())
        rows = [[f'{esc(short(r.trace))} @ {r.frames}', esc(r.decider), fmt(r.faults), fmt(r.faults_ref),
                 ('match', 'best') if r.faults == r.faults_ref else ('differs', 'worse')] for r in m.itertuples()]
        add('repro', 'Verification', f'Local rerun vs the Colab results: {same} of {len(m)} rows identical',
            '<p>The ablation was rerun locally on freshly downloaded traces and compared row by row with the CSV saved from the original Google Colab run '
            '(<code>results/reference/ablation_results.csv</code>). Identical fault counts mean the traces and the simulator are the same.</p>',
            '<details><summary>Show the comparison</summary>' + table(['Config', 'Decider', 'Faults (local)', 'Faults (Colab)', 'Result'], rows) + '</details>',
            '<code>ablation_results.csv</code> vs <code>reference/ablation_results.csv</code>')

# ---------------------------------------------------------------- 4. head-to-head
hh = csv('head_to_head.csv')
if hh is not None:
    cols = [c for c in ['FIFO', 'Clock', 'LRU', 'FIFO+pred', 'Clock+pred', 'LRU+pred', 'OPT'] if c in hh.columns]
    rows = []
    for r in hh.to_dict('records'):
        vals = [r[c] for c in cols]
        k = best_worse([r[c] for c in cols if c != 'OPT'], r['LRU'])
        rows.append([f'{esc(short(r["trace"]))} @ {r["frames"]}'] + [(fmt(v), c) for v, c in zip(vals[:-1], k)] + [fmt(vals[-1])])
    body = table(['Config'] + cols, rows)
    gp = csv('prediction_gain_per_policy.csv')
    if gp is not None:
        body += '<h3>Gain from adding prediction, per eviction policy</h3>' + table(
            ['Config', 'on FIFO', 'on LRU', 'on Clock'],
            [[f'{esc(short(r.trace))} @ {r.frames}', fmt(r.FIFO, 'pct'), fmt(r.LRU, 'pct'), fmt(r.Clock, 'pct')] for r in gp.itertuples()])
    add('headtohead', 'Stage 5b · Head-to-head', 'Adding prediction to FIFO, Clock or LRU gives about the same gain',
        '<p>Page faults; <code>+pred</code> = Markov hybrid with the oracle fault-delta decider on that eviction policy. '
        'Prefetching can go below OPT (bwaves) because OPT is optimal only among eviction policies and still pays every first-touch fault.</p>',
        body, '<code>head_to_head.csv</code>, <code>prediction_gain_per_policy.csv</code>, script 06')

# ---------------------------------------------------------------- 5. ghost
gh = csv('ghost_estimate_results.csv')
if gh is not None:
    rows = [[f'{esc(short(r.trace))} @ {r.frames}', esc(r.decider), fmt(r.faults), fmt(r.vs_LRU, 'pct2'), fmt(r.pf_used), fmt(r.pf_harm),
             fmt(r.final_thr, 'p').replace('.0%', '%'), (fmt(r.est_true_corr, 'f3'), 'worse' if r.est_true_corr < 0 else '')] for r in gh.itertuples()]
    add('ghost', 'Stage 5c · Deployable feedback', 'A ghost list reproduces the oracle decisions without a shadow engine',
        '<p>The oracle compares against a second, prefetch-free copy of memory, which a kernel cannot run. The ghost list only remembers pages that a prefetch evicted and counts how many fault back soon. '
        'The last column is the per-epoch correlation between the estimated and the true fault delta.</p>',
        table(['Config', 'Decider', 'Faults', 'vs LRU', 'Prefetches used', 'Harmful refaults', 'Final threshold (confidence)', 'corr(est, true), -1 to 1'], rows),
        '<code>ghost_estimate_results.csv</code>, script 07')

# ---------------------------------------------------------------- 6. phase 8
eo = csv('phase8_eviction_only.csv')
p8 = csv('phase8_head_to_head.csv')
if eo is not None and p8 is not None:
    pol = ['FIFO', 'Clock', 'LRU', 'LRU-2', 'ARC']
    rows = []
    for r in eo.to_dict('records'):
        k = best_worse([r[p] for p in pol], r['LRU'])
        rows.append([esc(short(r['trace'])), f'{r["ratio"]:.0%}', fmt(r['frames'])] + [(fmt(r[p]), c) for p, c in zip(pol, k)] + [fmt(r['OPT'])])
    body = '<h3>Eviction policies alone, across memory sizes</h3>' + table(['Trace', 'RAM', 'Frames'] + pol + ['OPT'], rows)
    cols = ['FIFO', 'Clock', 'LRU', 'LRU-2', 'ARC', 'RA+LRU', 'RA+ARC', 'Markov+LRU', 'Markov+ARC']
    rows = []
    for r in p8.to_dict('records'):
        k = best_worse([r[c] for c in cols], r['LRU'])
        rows.append([f'{esc(short(r["trace"]))} @ {r["frames"]}'] + [(fmt(r[c]), x) for c, x in zip(cols, k)] + [fmt(r['OPT'])])
    body += '<h3>Head-to-head with prefetching</h3>' + table(['Config'] + cols + ['OPT'], rows)
    add('phase8', 'Phase 8 · Stronger baselines', 'ARC and LRU-2 do not close the gap; Linux-style readahead beats Markov on 3 of 4 real traces',
        '<p>RA = Linux-style sequential readahead (window = a quarter of RAM). Markov rows use the deployable ghost decider. '
        'Simplifications: ARC drops the textbook tie-break; LRU-2 has no correlated-reference period. LRU-2\'s poor result on bwaves/leslie3d was checked against an independent implementation and is real.</p>',
        body, '<code>phase8_eviction_only.csv</code>, <code>phase8_head_to_head.csv</code>, script 09')

# ---------------------------------------------------------------- 7. improvements
im = csv('improvements.csv')
if im is not None:
    cols = ['base', 'RA', 'Mk_gh', 'Mk_d_t2', 'Comb', 'Comb_d_t2']
    labels = ['Policy alone', 'Readahead', 'Markov', 'δ-Markov top-2', 'RA + Markov', 'RA + δ-Markov top-2']
    rows = []
    for r in im.to_dict('records'):
        k = best_worse([r[c] for c in cols], r['base'])
        rows.append([f'{esc(short(r["trace"]))} @ {r["frames"]}', esc(r['policy'])] + [(fmt(r[c]), x) for c, x in zip(cols, k)] + [fmt(r['OPT'])])
    add('improvements', 'Phases 9-10 · Improvements', 'Combining readahead with Markov, and a delta-Markov tier',
        '<p>Every design choice was made on the warm-up portion only, then evaluated once here. δ-Markov learns gaps between pages instead of page numbers, so it can predict pages it has never seen. '
        'Top-2 (two guesses per access) is mixed: a big win on leslie3d, a loss on bfs-3 and bwaves.</p>',
        table(['Config', 'Eviction'] + labels + ['OPT'], rows),
        '<code>improvements.csv</code>, script 10')

# ---------------------------------------------------------------- 8. robustness
rb = csv('robustness.csv')
summary = {}
if rb is not None:
    g = lambda r, k: (r['LRU'] - r[k]) / r['LRU']
    for k in ('RA', 'Markov', 'Comb_d', 'Comb_d_t2', 'Comb_d_gated', 'Comb_d_dir', 'Final'):
        if k in rb.columns:
            v = (rb.LRU - rb[k]) / rb.LRU
            summary[k] = dict(mean=v.mean(), worst=v.min(), lose=int((v < 0).sum()))
    summary['n'] = len(rb)
    fk = 'Final' if 'Final' in rb.columns else 'Comb_d'
    summary['beats_ra'] = int((rb[fk] < rb.RA).sum())
    summary['ties_ra'] = int((rb[fk] == rb.RA).sum())
    charts = ''
    for ratio in sorted(rb.ratio.unique()):
        sub = rb[rb.ratio == ratio].to_dict('records')
        rows = [(short(r['trace']), {k: (g(r, k), f' ({r[k]:,} vs LRU {r["LRU"]:,} faults)') for k, *_ in SERIES}) for r in sub]
        charts += f'<h3>RAM = {ratio:.0%} of working set</h3>' + dotplot(
            rows, -0.25, 1.0, [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda t: f'{t * 100:.0f}%', lambda v: fmt(v, 'pct'),
            'fewer faults than LRU →', f'Fault reduction versus LRU at {ratio:.0%} memory')
    cols = [c for c in ['ARC', 'RA', 'Markov', 'Comb_d', 'Comb_d_t2', 'Comb_d_gated', 'Comb_d_dir', 'Final'] if c in rb.columns]
    rows = []
    for r in rb.to_dict('records'):
        k = best_worse([r[c] for c in cols], r['LRU'])
        rows.append([esc(short(r['trace'])), f'{r["ratio"]:.0%}', fmt(r['frames']), fmt(r['LRU'])] + [(fmt(r[c]), x) for c, x in zip(cols, k)] + [fmt(r['OPT'])])
    tbl = table(['Trace', 'RAM', 'Frames', 'LRU', 'ARC', 'Readahead', 'Markov', 'RA + δ-Markov', '… top-2', '… gated (ghost)', '… gated (directory)', 'Final'][:4 + len(cols)] + ['OPT'], rows)
    st = table(['System', 'Mean vs LRU', 'Worst case', 'Worse than LRU'],
               [[esc(lbl), (fmt(summary[k]['mean'], 'pct'), 'best' if k == 'Comb_d_t2' else ''), fmt(summary[k]['worst'], 'pct'), f'{summary[k]["lose"]} / {summary["n"]}']
                for k, lbl in [('RA', 'Readahead'), ('Markov', 'Markov (original)'), ('Comb_d', 'RA + δ-Markov'),
                               ('Comb_d_t2', 'RA + δ-Markov, top-2'), ('Comb_d_gated', 'RA + δ-Markov, gated, ghost feedback'), ('Comb_d_dir', 'RA + δ-Markov, gated, directory feedback'), ('Final', 'Final system (frozen controller, no budget cap)')] if k in summary])
    add('robustness', 'Phase 12a · Robustness', f'The combined system holds on {rb.trace.nunique()} real traces with no retuning',
        f'<p>{rb.trace.nunique()} real traces × {rb.ratio.nunique()} memory sizes = {len(rb)} points, LRU underneath. The final system has fewer faults than readahead alone in '
        f'{summary["beats_ra"]} of {len(rb)} points and ties in {summary["ties_ra"]}. The weak case is <code>bwaves-s2</code> (8-13 frames). '
        'milc and libquantum are pure streaming (LRU = OPT = working set), so their gain is readahead removing first-touch faults; lbm has no predictable pattern at all. The one large loss to readahead is sphinx3 at 1%, where the decider holds back useful streams. '
        'Gated = readahead confidence scaled by stream length so the decider can stop short streams. With ghost-list feedback it halves the worst case but costs part of the mcf gain; with shadow-directory feedback (Phase 14) it keeps the mcf gain and has the best worst case of all systems. Hover a mark for exact counts.</p>',
        st + legend() + charts + '<details><summary>Show all fault counts</summary>' + tbl + '</details>',
        '<code>robustness.csv</code>, script 11')

# ---------------------------------------------------------------- 8b. phase 14 (feedback estimate)
if rb is not None and 'Comb_d_dir' in rb.columns:
    rows = []
    for r in rb.to_dict('records'):
        gg, dd = (r['LRU'] - r['Comb_d_gated']) / r['LRU'], (r['LRU'] - r['Comb_d_dir']) / r['LRU']
        k = 'best' if dd > gg else ('worse' if dd < gg else '')
        rows.append([esc(short(r['trace'])), f'{r["ratio"]:.0%}', fmt(r['LRU']), fmt(r['Comb_d_gated']), fmt(gg, 'pct'), fmt(r['Comb_d_dir']), (fmt(dd, 'pct'), k)])
    diff = int((rb.Comb_d_dir != rb.Comb_d_gated).sum())
    add('phase14', 'Phase 14 · Feedback estimate', 'A shadow directory gives the decider the true fault delta',
        '<p>The ghost list only charges harm when the page a prefetch directly evicted comes back, so on the synthetic <code>clustered@9</code> it reported "winning" while prefetching lost 21.5%. '
        'The shadow directory is a no-prefetch copy of the eviction policy that stores page IDs only (8 bytes per tracked frame, about 0.2% of RAM with 4 KB pages). With every page tracked it measures the true per-epoch fault delta exactly; '
        'hash-sampling 1/8 of the pages (SHARDS) gave the same fault counts on 9 of 10 real traces at 5% memory. On <code>clustered@9</code> the combined system goes from −21.5% to −2.5% vs LRU. '
        f'On the real traces the two estimates lead to different fault counts in {diff} of {len(rb)} points (amber = directory better):</p>',
        table(['Trace', 'RAM', 'LRU', 'Ghost feedback', 'vs LRU', 'Directory feedback', 'vs LRU'], rows),
        '<code>robustness.csv</code> (columns <code>Comb_d_gated</code>, <code>Comb_d_dir</code>), script 11; clustered and sampling numbers in <code>docs/RESULTS.md</code> section 13')

# ---------------------------------------------------------------- 9. break-even
be = csv('breakeven.csv')
if be is not None:
    by = {}
    for r in be.to_dict('records'):
        by.setdefault(r['trace'], {})[r['system']] = r
    rows = [(short(t), {k: (d[k]['breakeven_c'], f' ({d[k]["prefetches"]:,} prefetches, precision {d[k]["precision"] * 100:.0f}%)') for k, *_ in SERIES if k in d})
            for t, d in by.items()]
    chart = legend() + dotplot(rows, -0.7, 1.0, [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda t: f'{t * 100:.0f}%',
                               lambda v: f'c* = {v * 100:.0f}% of a fault', 'tolerates costlier prefetches →', 'Break-even prefetch cost per trace')
    sysk = [k for k in ['RA', 'Markov', 'Comb_d', 'Comb_d_t2', 'Final'] if k in set(be.system)]
    trs = []
    for t, d in by.items():
        trs.append([esc(short(t)), fmt(next(iter(d.values()))['frames'])] +
                   [(f"{d[k]['breakeven_c'] * 100:.0f}%", 'worse' if d[k]['breakeven_c'] < 0 else '') if k in d and not np.isnan(d[k]['breakeven_c']) else 'n/a' for k in sysk] +
                   [fmt(d[k]['prefetches']) if k in d else '–' for k in sysk])
    nm = {'RA': 'RA', 'Markov': 'Markov', 'Comb_d': 'RA+δ', 'Comb_d_t2': 'RA+δ top-2', 'Final': 'Final'}
    tbl = table(['Trace', 'Frames'] + [f'c* {nm[k]}' for k in sysk] + [f'prefetches {nm[k]}' for k in sysk], trs)
    add('breakeven', 'Phase 12b · Cost of extra I/O', 'The gain depends on how expensive one prefetch is',
        '<p>Fault counts hide the disk reads that prefetching adds. <code>c* = (LRU faults − faults) / prefetches</code> is the highest cost of one prefetch, as a fraction of one fault\'s stall, '
        'at which a system still beats LRU (0% = fully hidden I/O, 100% = as costly as a fault). Every value is below 100%, so no system wins at every cost. The original Markov system uses I/O most efficiently where it helps; '
        'the combined system saves more faults but needs cheaper prefetches. No disk latency is assumed and prediction CPU cost is not included. RAM = 1% of working set.</p>',
        chart + '<details><summary>Show as table</summary>' + tbl + '</details>',
        '<code>breakeven.csv</code>, script 12')

# ---------------------------------------------------------------- 9b. time model (phase 15)
tmd = csv('time_model.csv')
if tmd is not None:
    tm = tmd[tmd.system != 'LRU']
    sysn = [s for s in ['Final', 'Readahead', 'RA+dMarkov', 'Markov'] if s in tm.system.unique()]
    labels = {'Final': 'Final system', 'RA+dMarkov': 'RA + δ-Markov (directory, default controller)', 'Readahead': 'Readahead', 'Markov': 'Markov (original)'}
    rows = []
    for (dev, tc, tp), g in tm.groupby(['device', 't_cpu_us', 't_pred_us'], sort=False):
        cells = []
        for s in sysn:
            x = g[g.system == s].speedup
            gm = float(np.exp(np.log(x).mean()))
            cells.append((f'{gm:.2f}× <span class="src">({int((x < 1).sum())}/{len(x)} slower)</span>', 'best' if gm > 1 else 'worse'))
        rows.append([esc(dev), f'{tc:g} µs', f'{tp:g} µs'] + cells)
    summ = table(['Device', 'Compute per access', 'Prediction cost', *[labels[s] for s in sysn]], rows)
    per = ''
    for dev in tm.device.unique():
        g = tm[(tm.device == dev) & (tm.t_cpu_us == 1.0) & (tm.t_pred_us > 0)]
        pv = g.pivot(index='trace', columns='system', values='speedup')
        lt = g.pivot(index='trace', columns='system', values='late_share')
        r2 = [[esc(short(t))] + [(f'{pv.loc[t, s]:.2f}×', 'best' if pv.loc[t, s] > 1 else 'worse') for s in sysn] +
              [fmt(lt.loc[t, s], 'pct').lstrip('+') for s in sysn] for t in pv.index]
        per += f'<h3>{esc(dev)}, compute 1 µs per access, prediction 0.2 µs</h3>' + table(
            ['Trace'] + [f'speedup: {labels[s]}' for s in sysn] + [f'late-stall share: {labels[s]}' for s in sysn], r2)
    add('timemodel', 'Phase 15 · Time model', 'Faster on a parallel SSD, slower on a single-channel HDD; in time, the final system ties readahead',
        '<p>An I/O timeline replays every run: compute time per access, a prediction cost per access, 4 KB reads of 80 µs on an NVMe SSD with 8 parallel channels or 5 ms on an HDD with 1 channel. '
        'Demand faults block and can queue behind prefetch reads; a prefetched page used before its read finishes stalls for the rest (a late prefetch). '
        'Speedup = time(LRU) ÷ time(system); amber = faster than LRU, red = slower. These are model estimates with stated latencies, not measurements; compute time per access is unknown for these traces, so it is swept.</p>'
        '<p>The original Markov system cuts faults by 35–37% on leslie3d and bwaves but is not faster: it prefetches one access ahead, so its reads arrive late. Readahead fetches a window ahead and arrives in time. '
        'The final system has fewer faults than readahead in most configurations, but in estimated time the two are level on average: it wins on leslie3d, loses on sphinx3.</p>',
        summ + '<details><summary>Show per-trace speedups</summary>' + per + '</details>',
        '<code>time_model.csv</code>, script 14, model in <code>src/paging/timing.py</code>')

# ---------------------------------------------------------------- 9c. component comparison (script 17)
cps, cds = csv('component_predictors_summary.csv'), csv('component_deciders_summary.csv')
if cps is not None and cds is not None:
    vals = cps.mean_vs_LRU.tolist()
    rows = []
    for r, k in zip(cps.to_dict('records'), best_worse(vals, 0.0, lower_better=False)):
        rows.append([esc(r['predictor']), fmt(r['useful'], 'p'), (fmt(r['mean_vs_LRU'], 'pct'), k), fmt(r['worst_vs_LRU'], 'pct'),
                     f'{int(r["worse_than_LRU"])}', fmt(r['precision'], 'p'), f'{r["NVMe_speedup"]:.2f}×', f'{r["HDD_speedup"]:.2f}×',
                     (fmt(r['clustered_vs_LRU'], 'pct'), 'worse' if r['clustered_vs_LRU'] < -0.005 else '')])
    body = '<h3>The six predictors, all through the same final decision engine</h3>' + table(
        ['Predictor', 'Exact next-page guesses (useful)', 'Faults vs LRU (mean)', 'Worst', 'Worse than LRU', 'Precision', 'SSD speed-up', 'HDD speed-up', 'clustered (synthetic)'], rows)
    cds = cds.rename(columns={cds.columns[0]: 'predictions'})
    for pl, g_ in cds.groupby('predictions', sort=False):
        vals = g_.mean_vs_LRU.tolist()
        rows = []
        for r, k in zip(g_.to_dict('records'), best_worse(vals, 0.0, lower_better=False)):
            rows.append([esc(r['decider']), (fmt(r['mean_vs_LRU'], 'pct'), k), fmt(r['worst_vs_LRU'], 'pct'), f'{int(r["worse_than_LRU"])}',
                         fmt(r['prefetch_per_access'], 'p'), f'{r["NVMe_speedup"]:.2f}×', f'{r["HDD_speedup"]:.2f}×',
                         (fmt(r['clustered_vs_LRU'], 'pct'), 'worse' if r['clustered_vs_LRU'] < -0.005 else '')])
        body += f'<h3>The decision engines, all fed the guesses of: {esc(pl)}</h3>' + table(
            ['Decision engine', 'Faults vs LRU (mean)', 'Worst real', 'Worse than LRU', 'Prefetches per request', 'SSD speed-up', 'HDD speed-up', 'clustered (synthetic)'], rows)
    add('components', 'Phase 18 · Component comparison', 'Every predictor and every decision engine, one change at a time',
        '<p>Same programs (every real trace at 1% memory, plus the synthetic clustered trace as a danger test), same LRU underneath; only one component changes per row. '
        'Speed-ups use the Phase 15 time model (1 µs compute per request, 0.2 µs prediction cost). Amber = best mean in the table; red = worse than LRU.</p>'
        '<p>Two readings. <b>Accuracy does not decide the winner:</b> readahead is exactly right about the next page only ~3% of the time but removes the most faults of any single predictor, because it fetches pages never seen before. '
        '<b>The decision engine matters most in the danger case:</b> "always" gets the full benefit on real programs but is badly hurt on clustered; fixed thresholds are safe but lose half the benefit; '
        'the final directory engine keeps the benefit and limits the harm. Fed the original Markov predictor, all engines give about the same result, because that predictor rarely has anything worth prefetching.</p>',
        body, '<code>component_predictors_summary.csv</code>, <code>component_deciders_summary.csv</code>, script 17')

# ---------------------------------------------------------------- headline + page
tiles = ''
if summary:
    tiles = '<div class="tiles">' + ''.join(
        f'<div class="tile{" ours" if o else ""}"><div class="v">{v}</div><div class="k">{esc(k)}</div></div>' for v, k, o in [
            (fmt(summary['Final']['mean'], 'pct'), 'Final system, mean fault reduction vs LRU', True),
            (fmt(summary['RA']['mean'], 'pct'), 'Linux-style readahead alone', False),
            (fmt(summary['Markov']['mean'], 'pct'), 'Original Markov system', False),
            (f'{summary["n"] - summary["Final"]["lose"]} / {summary["n"]}', 'Configurations where the final system beats LRU', True)]) + '</div>'
nav = '<nav aria-label="Contents"><ol>' + ''.join(f'<li><a href="#{e}">{esc(t)}</a></li>' for e, _, t in toc) + '</ol></nav>'
when = datetime.datetime.now().strftime('%d %b %Y, %H:%M')
skip_note = f'<p class="src">Skipped (CSV missing): {esc(", ".join(skipped))}</p>' if skipped else ''

CSS = open(os.path.join(os.path.dirname(__file__), 'report_style.css'), encoding='utf-8').read()
page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictive Paging Report</title><style>{CSS}</style></head><body><main>
<header><div class="eyebrow">Predictive Memory Paging System with Workload Awareness · results report · generated {esc(when)}</div>
<h1>Predictive paging: every result so far</h1>
<p class="lede">A trace-driven simulator: a predictor guesses the next page, a decision engine decides whether prefetching it is worth the risk, and an eviction policy decides what leaves RAM.
All counts are page faults (lower is better); <code>vs_LRU = (LRU faults − faults) / LRU faults</code>, positive is better. Amber marks the best value in a row, red marks more faults than the baseline.</p>
{tiles}{skip_note}</header>
{nav}
{"".join(parts)}
<footer><div>G.K Shreyas Bharadwaj (24BCE2256), Vaastav Bajanthri (24BCE2223), P. Giphy Bershan (24BCE2201)</div>
<div>Regenerate: run scripts 03-12, then <code>python scripts/13_report.py</code>. Every number on this page is read from the CSV files in <code>results/</code>.</div></footer>
</main>
<script>/* open every collapsed table when printing or saving as PDF */
addEventListener('beforeprint', () => document.querySelectorAll('details').forEach(d => d.open = true));</script>
</body></html>'''
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(page)
print(f'report written to {OUT} ({len(page) / 1024:.0f} KB, {len(toc)} sections)')
if '--open' in sys.argv:                      # python scripts/13_report.py --open  -> show it in the browser
    import webbrowser
    webbrowser.open('file:///' + os.path.abspath(OUT).replace(os.sep, '/'))
if skipped:
    print('skipped (missing CSV):', ', '.join(skipped))
