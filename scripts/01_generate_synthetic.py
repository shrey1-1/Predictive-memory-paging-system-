"""Stage 1a — generate the six synthetic traces into data/traces/."""
import _path  # noqa: F401
import numpy as np
from paging import config
from paging.traces import generate_synthetic, load_trace, GENERATORS

config.ensure_dirs()
written = generate_synthetic(config.TRACE_DIR)
for name, n in written.items():
    t = load_trace(name)
    d = np.diff(t.pages[:5000])
    print(f'{name:16s} {n:7d} accesses  unique pages={len(np.unique(t.pages)):6d}  '
          f'unique deltas={len(np.unique(d)):5d}')

seq_deltas = np.unique(np.diff(load_trace('sequential').pages[:5000]))
assert len(seq_deltas) == 1, 'sequential generator broken: expected exactly 1 unique delta'
print(f'\nOK — traces written to {config.TRACE_DIR}')
