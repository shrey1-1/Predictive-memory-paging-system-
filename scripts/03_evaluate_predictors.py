"""Stage 3 — predictor quality, attribution, memory footprint, capacity sweep."""
import _path  # noqa: F401
import os
import numpy as np, pandas as pd
from paging import config
from paging.traces import load_all, REAL
from paging.predictors import make_predictor, MarkovPredictor, KINDS
from paging.simulate import eval_predictor

config.ensure_dirs()
TRACES = load_all(limit=config.TRACE_LIMIT)
print(f'{len(TRACES)} traces loaded: {list(TRACES)}\n')
F = config.FROZEN

# --- main table ---
rows = []
for name, t in TRACES.items():
    for k in KINDS:
        s = eval_predictor(t, make_predictor(k))
        rows.append({'trace': name, 'type': 'real' if name in REAL else 'synthetic',
                     'predictor': k, 'coverage': round(s['coverage'], 3),
                     'accuracy': round(s['accuracy'], 3), 'useful': round(s['useful'], 3)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(config.RESULTS_DIR, 'predictor_results.csv'), index=False)
print('=== USEFUL (coverage x accuracy) ===')
print(res.pivot(index='trace', columns='predictor', values='useful')[KINDS], '\n')

# --- attribution ---
att = []
for name, t in TRACES.items():
    h = make_predictor('hybrid'); eval_predictor(t, h)
    src = h.stats()['sources']; tot = max(1, sum(src.values()))
    att.append({'trace': name, **{k: round(v / tot, 3) for k, v in src.items()}})
att = pd.DataFrame(att).set_index('trace')
att.to_csv(os.path.join(config.RESULTS_DIR, 'predictor_attribution.csv'))
print('=== which sub-predictor fired ===')
print(att, '\n')

# --- memory footprint (information-theoretic) ---
CTX, SUCC = F['order'] * 8, 9
MAX_B = F['capacity'] * CTX + F['capacity'] * F['max_succ'] * SUCC
print(f'bounded table ceiling: {MAX_B/1024:.1f} KB (~{MAX_B/4096:.1f} pages)')
for frames in (128, 256, 512, 1024):
    print(f'  overhead @ {frames:4d} frames: {100*MAX_B/1024/(frames*4):.2f}%')

# --- capacity sweep ---
sw = []
for cap in (128, 256, 512, 1024, 2048, 4096):
    cfg = {**F, 'capacity': cap}
    real_s, all_s = [], []
    for name, t in TRACES.items():
        u = eval_predictor(t, MarkovPredictor(**cfg))['useful']
        all_s.append(u)
        if name in REAL: real_s.append(u)
    sw.append({'capacity': cap, 'table_KB': round((cap*CTX + cap*F['max_succ']*SUCC)/1024, 1),
               'useful_real': round(np.mean(real_s), 3) if real_s else None,
               'useful_all': round(np.mean(all_s), 3)})
sw = pd.DataFrame(sw)
sw.to_csv(os.path.join(config.RESULTS_DIR, 'capacity_sweep.csv'), index=False)
print('\n=== capacity sweep ===')
print(sw.to_string(index=False))
print(f'\nresults written to {config.RESULTS_DIR}')
