C_FINAL, C_RA, C_BASE = AMBER, '#333333', '#9a9a9a'      # validated: amber / dark grey / mid grey (+ OPT as black outline)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.frameon': False})


def load(name):
    p = f'{RESULTS}/{name}'
    return pd.read_csv(p) if os.path.exists(p) else None


def grid(ax, axis='x'):
    ax.grid(axis=axis, color='#e4e4e4', lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def finish(name):
    plt.tight_layout()
    plt.savefig(f'{FIG}/{name}.png', dpi=200, bbox_inches='tight')
    plt.show()
    print(f'saved {FIG}/{name}.png')


pct = lambda v: f'{v * 100:+.1f}%'
rb, h8, fc = load('robustness.csv'), load('phase8_head_to_head.csv'), load('final_classic_configs.csv')

# ---------------------------------------------------------------------------- 1. classic configurations
if h8 is not None and fc is not None:
    d = h8.merge(fc[['trace', 'frames', 'Final']], on=['trace', 'frames'])
    ROWS = [('OPT (offline bound)', 'OPT', None), ('FIFO', 'FIFO', C_BASE), ('Clock', 'Clock', C_BASE),
            ('LRU', 'LRU', C_BASE), ('LRU-2', 'LRU-2', C_BASE), ('ARC', 'ARC', C_BASE),
            ('Markov + LRU', 'Markov+LRU', C_RA), ('Readahead + LRU', 'RA+LRU', C_RA), ('Final system', 'Final', C_FINAL)]
    fig, axes = plt.subplots(1, len(d), figsize=(3.0 * len(d), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(ROWS))[::-1]
    for ax, (_, r) in zip(axes, d.iterrows()):
        vals = [r[c] / r['LRU'] for _, c, _ in ROWS]
        for yi, v, (label, c, col) in zip(y, vals, ROWS):
            if col is None:
                ax.barh(yi, v, height=0.62, facecolor='white', edgecolor='black', lw=1.0)
            else:
                ax.barh(yi, v, height=0.62, color=col)
        fv = r['Final'] / r['LRU']
        ax.text(fv + 0.03, y[-1], f'{fv:.2f}', va='center', fontsize=8, color='#222222')
        ax.axvline(1.0, color='#555555', lw=0.8, ls=':')
        tag = ' (synthetic)' if r['trace'] in SYNTHETIC else ''
        ax.set_title(f"{r['trace']}{tag}\n{int(r['frames'])} frames", fontsize=9)
        ax.set_xlim(0, max(1.15, max(vals) * 1.12)); grid(ax)
    axes[0].set_yticks(y); axes[0].set_yticklabels([lbl for lbl, _, _ in ROWS])
    fig.supxlabel('page faults relative to LRU (dotted line = LRU; lower is better)', fontsize=9)
    fig.suptitle('1. Every algorithm on the classic configurations (RAM about 1% of the working set)', fontsize=11, fontweight='bold')
    finish('cmp1_classic_configurations')
    show = d.set_index(['trace', 'frames'])[[c for _, c, _ in ROWS]]
    display(show.div(show.LRU, axis=0).round(3))
else:
    print('1. skipped: needs phase8_head_to_head.csv (cell 24) and final_classic_configs.csv (cell 27)')

# ---------------------------------------------------------------------------- 2. 22 real traces at 1%
if rb is not None:
    v = rb.copy()
    for c in ('ARC', 'RA', 'Markov', 'Comb_d', 'Comb_d_t2', 'Comb_d_gated', 'Comb_d_dir', 'Final', 'OPT'):
        v['g_' + c] = (v.LRU - v[c]) / v.LRU
    one = v[v.ratio == 0.01].sort_values('g_Final')
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(one) + 1.6))
    yy = np.arange(len(one))
    for yi, (_, r) in zip(yy, one.iterrows()):
        ax.plot([r.g_RA * 100, r.g_Final * 100], [yi, yi], color='#cfcfcf', lw=1.2, zorder=1)
    ax.scatter(one.g_Markov * 100, yy, s=34, marker='o', color=C_BASE, label='Markov prefetcher (Stage 5)', zorder=2)
    ax.scatter(one.g_RA * 100, yy, s=34, marker='s', color=C_RA, label='Linux-style readahead', zorder=3)
    ax.scatter(one.g_Final * 100, yy, s=48, marker='D', color=C_FINAL, label='final system', zorder=4,
               edgecolor='white', linewidth=0.6)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels(one.trace, fontsize=8)
    ax.set_xlabel('fault reduction vs LRU, % (right is better; left of 0 = more faults than LRU)')
    ax.set_title('2. Fault reduction on 22 real traces, RAM = 1% of working set', loc='left')
    ax.legend(loc='lower right', fontsize=8); grid(ax)
    finish('cmp2_real_traces_1pct')
    display((one.set_index('trace')[['frames', 'g_Markov', 'g_RA', 'g_Final', 'g_OPT']] * [1, 100, 100, 100, 100]).round(1)
            .rename(columns={'g_Markov': 'Markov %', 'g_RA': 'readahead %', 'g_Final': 'final %', 'g_OPT': 'OPT %'}))

    # ------------------------------------------------------------------------ 3. evolution of the system
    STEPS = [('ARC (eviction only)', 'ARC'), ('Markov + ghost (Stage 5)', 'Markov'), ('Linux-style readahead', 'RA'),
             ('readahead + delta-Markov', 'Comb_d'), ('  + top-2 guesses', 'Comb_d_t2'), ('  + ghost-list gating', 'Comb_d_gated'),
             ('  + shadow directory', 'Comb_d_dir'), ('final (frozen controller)', 'Final')]
    st = pd.DataFrame({'system': [s for s, _ in STEPS],
                       'mean': [v['g_' + c].mean() for _, c in STEPS], 'worst': [v['g_' + c].min() for _, c in STEPS],
                       'worse_than_LRU': [int((v['g_' + c] < 0).sum()) for _, c in STEPS]})
    cols = [C_FINAL if c == 'Final' else C_RA if c == 'RA' else C_BASE for _, c in STEPS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    yy = np.arange(len(st))[::-1]
    a1.barh(yy, st['mean'] * 100, color=cols, height=0.62)
    for yi, m in zip(yy, st['mean']):
        a1.text(m * 100 + (1 if m >= 0 else -1), yi, pct(m), va='center', ha='left' if m >= 0 else 'right', fontsize=8)
    a1.set_title('mean fault reduction vs LRU (higher is better)', loc='left'); a1.axvline(0, color='black', lw=0.8)
    a1.set_xlim(min(-5, st['mean'].min() * 100 - 8), st['mean'].max() * 100 + 12)
    a2.barh(yy, st['worst'] * 100, color=cols, height=0.62)
    for yi, w, k in zip(yy, st['worst'], st['worse_than_LRU']):
        a2.text(0.6, yi, f'{pct(w)}   worse than LRU in {k}/{len(v)}', va='center', fontsize=8)
    a2.set_title('worst case (closer to 0 is safer)', loc='left'); a2.axvline(0, color='black', lw=0.8)
    a2.set_xlim(st['worst'].min() * 100 - 2, 45)
    a1.set_yticks(yy); a1.set_yticklabels(st.system)
    for a in (a1, a2):
        grid(a); a.set_xlabel('fault reduction vs LRU, %')
    fig.suptitle(f'3. How the system evolved: {len(v)} points ({v.trace.nunique()} real traces x {v.ratio.nunique()} memory sizes)',
                 fontsize=11, fontweight='bold')
    finish('cmp3_evolution')
    display(st.assign(mean=(st['mean'] * 100).round(1), worst=(st['worst'] * 100).round(1)).rename(
        columns={'mean': 'mean %', 'worst': 'worst %'}))

    # ------------------------------------------------------------------------ 4. final vs readahead, both memory sizes
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    lo = min(v.g_RA.min(), v.g_Final.min()) * 100 - 5
    ax.plot([lo, 100], [lo, 100], color='#888888', lw=0.8, ls='--', label='same as readahead')
    for ratio, mk, col, label in ((0.01, 'o', C_FINAL, 'RAM = 1% of working set'), (0.05, '^', C_RA, 'RAM = 5% of working set')):
        s = v[v.ratio == ratio]
        ax.scatter(s.g_RA * 100, s.g_Final * 100, marker=mk, s=40, color=col, label=label, edgecolor='white', linewidth=0.6, zorder=3)
    far = v[(v.g_Final - v.g_RA).abs() > 0.10].copy()                       # label only the big differences
    far['program'] = far.trace.str.replace(r'-s\d$', '', regex=True)          # segments of one program share a label
    for (prog, ratio), grp in far.groupby(['program', 'ratio']):
        segs = [t.rsplit('-', 1)[1] for t in grp.trace if t != prog]
        name = f"{prog} {'/'.join(segs)}" if len(grp) > 1 else grp.trace.iloc[0]
        x, y = grp.g_RA.mean() * 100, grp.g_Final.mean() * 100
        left = y > x and x > 15                                 # near the left edge: always label to the right
        ax.annotate(f'{name} ({ratio:.0%})', (x, y), xytext=(-12, 6) if left else (10, -12),
                    ha='right' if left else 'left', textcoords='offset points', fontsize=7.5, color='#333333')
    ax.axhline(0, color='black', lw=0.6); ax.axvline(0, color='black', lw=0.6)
    ax.set_xlim(lo, 100); ax.set_ylim(lo, 100)
    ax.set_xlabel('readahead: fault reduction vs LRU, %'); ax.set_ylabel('final system: fault reduction vs LRU, %')
    ax.set_title('4. Final system vs readahead (above the line = final better)', loc='left')
    ax.legend(loc='lower right', fontsize=8); grid(ax, 'both')
    finish('cmp4_final_vs_readahead')
    print(f"final has fewer faults than readahead in {(rb.Final < rb.RA).sum()}, equal {(rb.Final == rb.RA).sum()}, "
          f"more {(rb.Final > rb.RA).sum()} of {len(rb)} points")
    display((far.set_index(['trace', 'ratio'])[['g_RA', 'g_Final']] * 100).round(1))
else:
    print('2-4. skipped: needs robustness.csv (cell 28)')

# ---------------------------------------------------------------------------- 5. estimated time
tm = load('time_model.csv')
if tm is not None:
    s = tm[(tm.t_pred_us > 0) & (tm.system.isin(['Readahead', 'Markov', 'Final']))]
    g = s.groupby(['device', 't_cpu_us', 'system']).speedup.apply(lambda x: float(np.exp(np.log(x).mean()))).reset_index()
    SYS = [('Markov', C_BASE, 'o', 'Markov prefetcher'), ('Readahead', C_RA, 's', 'readahead'), ('Final', C_FINAL, 'D', 'final system')]
    devs = [dv for dv in ('NVMe SSD', 'HDD') if dv in set(g.device)]
    fig, axes = plt.subplots(1, len(devs), figsize=(5.4 * len(devs), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dev in zip(axes, devs):
        ends = {}
        for sysn, col, mk, label in SYS:
            q = g[(g.device == dev) & (g.system == sysn)].sort_values('t_cpu_us')
            ax.plot(q.t_cpu_us, q.speedup, marker=mk, color=col, lw=2, ms=7, label=label)
            ends[sysn] = q.speedup.iloc[-1]
        ypos = dict(ends)                                   # push end labels apart when two lines end together
        if abs(ends['Final'] - ends['Readahead']) < 0.04:
            mid = (ends['Final'] + ends['Readahead']) / 2
            ypos['Readahead'], ypos['Final'] = (mid + 0.025, mid - 0.025) if ends['Readahead'] >= ends['Final'] else (mid - 0.025, mid + 0.025)
        for sysn, col, _, label in SYS:
            ax.text(13.5, ypos[sysn], f'{label.split()[0]} {ends[sysn]:.2f}x', va='center', fontsize=7.5, color='#222222')
        ax.axhline(1.0, color='black', lw=0.8, ls=':'); ax.text(0.105, 1.01, 'LRU (no prefetching)', fontsize=7.5, va='bottom')
        ax.set_xscale('log'); ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(['0.1', '1', '10'])
        ax.set_xlim(0.08, 40); ax.set_xlabel('compute time per access, µs (unknown for these traces, so swept)')
        chan = '8 channels, 80 µs per read' if dev == 'NVMe SSD' else '1 channel, 5 ms per read'
        ax.set_title(f'{dev} ({chan})', loc='left'); grid(ax, 'y')
    axes[0].set_ylabel('estimated speed-up vs LRU\n(geometric mean over traces; > 1 = faster)')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('5. Does fewer faults mean faster? Estimated run time (model with stated latencies, not a measurement)',
                 fontsize=11, fontweight='bold')
    finish('cmp5_time_model')
    display(g.pivot_table(index=['device', 't_cpu_us'], columns='system', values='speedup').round(3))
else:
    print('5. skipped: needs time_model.csv (cell 30)')

# ---------------------------------------------------------------------------- 6. break-even cost
be = load('breakeven.csv')
if be is not None:
    p = be.pivot(index='trace', columns='system', values='breakeven_c').sort_values('Final')
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(p) + 1.6))
    yy = np.arange(len(p))
    for sysn, col, mk, label, sz in (('Markov', C_BASE, 'o', 'Markov prefetcher', 34), ('RA', C_RA, 's', 'readahead', 34),
                                     ('Final', C_FINAL, 'D', 'final system', 48)):
        ax.scatter(p[sysn], yy, marker=mk, s=sz, color=col, label=label, zorder=3, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', lw=0.8)
    ax.axvline(1, color='#555555', lw=0.8, ls=':'); ax.text(0.99, len(p) - 0.4, 'prefetch costs\nas much as a fault', ha='right', fontsize=7.5)
    ax.set_yticks(yy); ax.set_yticklabels(p.index, fontsize=8)
    ax.set_xlabel('break-even cost c*: the most one prefetch may cost (as a share of one fault) before LRU wins')
    ax.set_title('6. Break-even prefetch cost per trace (RAM = 1%; right is more robust)', loc='left')
    ax.legend(loc='lower right', fontsize=8); grid(ax)
    finish('cmp6_breakeven')
    display(p[['Markov', 'RA', 'Final']].round(2))
else:
    print('6. skipped: needs breakeven.csv (cell 29)')

# ---------------------------------------------------------------------------- 7. components
cps, cds = load('component_predictors_summary.csv'), load('component_deciders_summary.csv')
if cps is not None and cds is not None:
    cds = cds.rename(columns={cds.columns[0]: 'predictions'})
    dec = cds[(cds.predictions == 'final predictor') & (cds.decider != 'no-prefetch')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for _, r in cps.iterrows():
        col = C_FINAL if r.predictor.startswith('final') else C_RA if r.predictor == 'readahead' else C_BASE
        a1.scatter(r.useful * 100, r.mean_vs_LRU * 100, s=55, color=col, zorder=3, edgecolor='white', linewidth=0.6)
        a1.annotate(r.predictor.replace(' (readahead + delta-hybrid)', ''), (r.useful * 100, r.mean_vs_LRU * 100),
                    xytext=(6, -3), textcoords='offset points', fontsize=8)
    a1.set_xlabel('exact next-page guesses ("useful" = coverage x accuracy), %')
    a1.set_ylabel('mean fault reduction vs LRU, %')
    a1.set_title('predictors, all through the final decision engine:\naccuracy does not decide fault reduction', loc='left')
    a1.set_xlim(-5, 75); grid(a1, 'both')
    for _, r in dec.iterrows():
        col = C_FINAL if 'final' in r.decider else C_BASE
        a2.scatter(r.clustered_vs_LRU * 100, r.mean_vs_LRU * 100, s=55, color=col, zorder=3, edgecolor='white', linewidth=0.6)
    OFFSET = {'directory + frozen controller (final)': (8, 6, 'left'), 'fault-delta oracle': (8, -10, 'left'),
              'shadow directory': (-8, -10, 'right')}                 # these three sit close together
    for (cx, my), names in dec.groupby([dec.clustered_vs_LRU.round(4), dec.mean_vs_LRU.round(4)]).decider:
        dx, dy, ha = OFFSET.get(names.iloc[0], (6, -3, 'left'))
        a2.annotate(' / '.join(names), (cx * 100, my * 100), xytext=(dx, dy), ha=ha, textcoords='offset points', fontsize=7.5)
    a2.axvline(0, color='black', lw=0.6)
    a2.set_xlabel('clustered trace (synthetic danger case): fault change vs LRU, %  (left = harmful)')
    a2.set_ylabel('mean fault reduction on real traces, %')
    a2.set_title('decision engines, all fed the final predictor:\nbenefit (up) vs harm in the danger case (left)', loc='left')
    a2.set_xlim(dec.clustered_vs_LRU.min() * 100 - 3, 12); grid(a2, 'both')
    fig.suptitle('7. One component changed at a time (22 real traces at 1% + synthetic clustered)', fontsize=11, fontweight='bold')
    finish('cmp7_components')
    display(cps.set_index('predictor')[['useful', 'mean_vs_LRU', 'worst_vs_LRU', 'NVMe_speedup']].round(3))
    display(dec.set_index('decider')[['mean_vs_LRU', 'worst_vs_LRU', 'clustered_vs_LRU', 'NVMe_speedup']].round(3))
else:
    print('7. skipped: needs the component_*_summary.csv files (cell 32)')

# ---------------------------------------------------------------------------- headline (computed, not typed)
if rb is not None:
    g = lambda c: (rb.LRU - rb[c]) / rb.LRU
    print(f"\nHEADLINE ({len(rb)} points): final system mean {pct(g('Final').mean())}, worst {pct(g('Final').min())}, "
          f"worse than LRU in {(g('Final') < 0).sum()};  readahead mean {pct(g('RA').mean())}, worst {pct(g('RA').min())};  "
          f"Markov (Stage 5) mean {pct(g('Markov').mean())}")
if tm is not None:
    q = tm[(tm.t_pred_us > 0) & (tm.device == 'NVMe SSD') & (tm.t_cpu_us == 1.0)].groupby('system').speedup.apply(
        lambda x: float(np.exp(np.log(x).mean())))
    print(f"estimated time on the SSD model (1 µs compute): final {q.get('Final', np.nan):.2f}x, "
          f"readahead {q.get('Readahead', np.nan):.2f}x vs LRU: fewer faults, but the time gain roughly ties readahead")
