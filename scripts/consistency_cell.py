
def load(name):
    p = f'{RESULTS}/{name}'
    return pd.read_csv(p) if os.path.exists(p) else None


checks = []
def check(what, ok, detail=''):
    checks.append({'check': what, 'result': 'OK' if ok else 'MISMATCH', 'detail': detail})


def same(a, b, tol):
    # the CSVs store rounded values, so compare within the rounding step
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, equal_nan=True))


# 1. Same trace + RAM size + policy => same fault count in every file
recs = []
def collect(src, df, cols):
    if df is not None:
        for c in cols:
            if c in df.columns:
                recs.append(pd.DataFrame({'trace': df.trace, 'frames': df.frames, 'policy': c, 'faults': df[c], 'src': src}))

b = load('baseline_results.csv')
if b is not None:
    recs.append(b[['trace', 'frames', 'policy', 'faults']].assign(src='baseline_results'))
collect('phase8_eviction_only', load('phase8_eviction_only.csv'), ['FIFO', 'Clock', 'LRU', 'LRU-2', 'ARC', 'OPT'])
collect('phase8_head_to_head', load('phase8_head_to_head.csv'), ['FIFO', 'Clock', 'LRU', 'LRU-2', 'ARC', 'OPT'])
collect('head_to_head', load('head_to_head.csv'), ['FIFO', 'Clock', 'LRU', 'OPT'])
collect('robustness', load('robustness.csv'), ['LRU', 'ARC', 'OPT'])
collect('final_system', load('final_system.csv'), ['LRU', 'OPT'])
for f in ('breakeven', 'component_deciders', 'component_predictors'):
    collect(f, load(f + '.csv'), ['LRU'])
ab = load('ablation_results.csv')
if ab is not None:
    collect('ablation', ab.rename(columns={'LRU_ref': 'LRU', 'OPT_ref': 'OPT'}), ['LRU', 'OPT'])
im = load('improvements.csv')
if im is not None:
    collect('improvements', im[im.policy == 'LRU'].rename(columns={'base': 'LRU'}), ['LRU', 'OPT'])
for f in ('phase6_memory_sweep', 'phase6_segments'):
    d = load(f + '.csv')
    if d is not None and 'program' in d.columns:
        d = d.assign(trace=d.program + '-' + d.segment)
    collect(f, d, ['LRU', 'OPT'])
if recs:
    allr = pd.concat(recs, ignore_index=True).dropna(subset=['faults'])
    allr['faults'] = allr.faults.astype(int)
    n = allr.groupby(['trace', 'frames', 'policy']).faults.nunique()
    bad = n[n > 1]
    check('same faults for the same trace / RAM / policy in every file', bad.empty,
          f'{len(n)} combinations, {allr.src.nunique()} files' if bad.empty else f'{len(bad)} conflicts (shown below)')
    if len(bad):
        x = allr.set_index(['trace', 'frames', 'policy']).loc[bad.index].reset_index()
        display(x.pivot_table(index=['trace', 'frames', 'policy'], columns='src', values='faults'))

# 2. The final system gives the same faults wherever it was run (RAM = 1%)
rb = load('robustness.csv')
if rb is not None:
    ref = rb[rb.ratio == 0.01].set_index('trace')

    def same_final(what, s):
        s = s[s.index.isin(ref.index)]
        diff = int((s != ref.loc[s.index, 'Final']).sum())
        check(f'final system faults: {what} == robustness Final', diff == 0, f'{len(s)} traces, {diff} differ')

    be = load('breakeven.csv')
    if be is not None:
        same_final('breakeven', be[be.system == 'Final'].set_index('trace').faults)
    tm = load('time_model.csv')
    if tm is not None:
        same_final('time_model', tm[tm.system == 'Final'].drop_duplicates('trace').set_index('trace').faults)
    cd = load('component_deciders.csv')
    if cd is not None:
        sel = (cd.predictions == 'final predictor') & cd.decider.str.contains('final')
        same_final('component_deciders', cd[sel].set_index('trace').faults)
    fs = load('final_system.csv')
    if fs is not None:
        same_final('final_system (cell 19)', fs.set_index('trace').final)

# 3. Derived columns match their definitions
if ab is not None:
    check('ablation: vs_LRU = (LRU - faults) / LRU', same(ab.vs_LRU, (ab.LRU_ref - ab.faults) / ab.LRU_ref, 1e-4))
    check('ablation: headroom = (LRU - OPT) / LRU', same(ab.headroom, (ab.LRU_ref - ab.OPT_ref) / ab.LRU_ref, 1e-4))
    check('ablation: precision = used / prefetched', same(ab.pf_precision, ab.pf_used / ab.prefetches.clip(lower=1), 6e-4))
    ds = load('decider_summary.csv')
    if ds is not None:
        g = ab.groupby('decider').vs_LRU.agg(['mean', 'min']).loc[ds.decider]
        check('decider_summary matches ablation', same(ds.mean_gain, g['mean'], 1e-4) and same(ds.worst_case, g['min'], 1e-4))
hh, pg = load('head_to_head.csv'), load('prediction_gain_per_policy.csv')
if hh is not None and pg is not None:
    check('prediction_gain_per_policy matches head_to_head',
          all(same(pg[p], (hh[p] - hh[p + '+pred']) / hh[p], 1e-3) for p in ('FIFO', 'LRU', 'Clock')))
for f in ('component_deciders', 'component_predictors'):
    d = load(f + '.csv')
    if d is not None:
        check(f'{f}: vs_LRU = (LRU - faults) / LRU', same(d.vs_LRU, (d.LRU - d.faults) / d.LRU, 1e-4))
cp, cps = load('component_predictors.csv'), load('component_predictors_summary.csv')
if cp is not None and cps is not None:
    g = cp[cp.trace != 'clustered'].groupby('predictor').vs_LRU.mean().loc[cps.predictor]
    check('component_predictors_summary = mean over the real traces', same(cps.mean_vs_LRU, g, 1e-4))
for f in ('phase6_memory_sweep', 'phase6_segments', 'phase6_seeds'):
    d = load(f + '.csv')
    if d is not None:
        check(f'{f}: *_vs_LRU columns', all(same(d[c + '_vs_LRU'], (d.LRU - d[k]) / d.LRU, 1e-3)
                                            for c, k in (('always', 'always'), ('ghost', 'ghost'), ('opt', 'OPT'))))

# 4. Sanity: no eviction-only policy beats OPT; LRU and OPT never get worse with more RAM
if b is not None:
    pv = b.pivot_table(index=['trace', 'frames'], columns='policy', values='faults')
    check('baseline: no policy beats OPT', bool(pv.drop(columns='OPT').ge(pv.OPT, axis=0).all().all()))
    mono = all((np.diff(x.sort_values('frames')[p].values) <= 0).all()
               for _, x in pv.reset_index().groupby('trace') for p in ('LRU', 'OPT'))
    check('baseline: LRU and OPT never worsen with more RAM', mono)
ev = load('phase8_eviction_only.csv')
if ev is not None:
    check('phase8: FIFO / Clock / LRU / LRU-2 / ARC never beat OPT',
          bool(ev[['FIFO', 'Clock', 'LRU', 'LRU-2', 'ARC']].ge(ev.OPT, axis=0).all().all()))

res = pd.DataFrame(checks)
display(res)
n_bad = int((res.result != 'OK').sum()) if len(res) else 0
print(f'{len(res) - n_bad} of {len(res)} checks passed'
      + ('' if n_bad == 0 else '  <- do not quote the mismatching numbers until this is explained'))
