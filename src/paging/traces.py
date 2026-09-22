"""
Stages 1 & 2 — trace acquisition and normalisation.

Every trace, synthetic or real, becomes the same thing: an ordered sequence
of page IDs stored as CSV with columns `access_id,page_id`.
"""
import csv
import gzip
import lzma
import os

import numpy as np
import pandas as pd

from .config import PAGE_SHIFT, MAX_ACCESSES, TRACE_DIR

SYNTHETIC = ['sequential', 'strided', 'cyclic', 'clustered', 'random', 'phase_changing']
REAL = ['410.bwaves-s0', '437.leslie3d-s0', 'bfs-3', 'pr-3']


# --------------------------------------------------------------------------
# Synthetic generators (Stage 1a)
# --------------------------------------------------------------------------
def sequential(n=50000, start=0, stride=1, n_pages=10000):
    """1, 2, 3, 4 ... — the easiest possible pattern."""
    return [(start + i * stride) % n_pages for i in range(n)]


def strided(n=50000, stride=8, n_pages=10000):
    """0, 8, 16, 24 ... — regular but with gaps."""
    return sequential(n, 0, stride, n_pages)


def cyclic(n=50000, loop_len=20, start=100):
    """A fixed loop repeated forever."""
    loop = [start + i for i in range(loop_len)]
    return [loop[i % loop_len] for i in range(n)]


def clustered(n=50000, cluster_size=10, dwell=500, n_pages=10000, seed=0):
    """Bounces randomly inside a small neighbourhood, then jumps elsewhere."""
    rng = np.random.default_rng(seed)
    trace, i = [], 0
    while i < n:
        base = rng.integers(0, n_pages - cluster_size)
        for _ in range(min(dwell, n - i)):
            trace.append(int(base + rng.integers(0, cluster_size)))
            i += 1
    return trace


def uniform_random(n=50000, n_pages=10000, seed=0):
    """No structure at all — worst case for any predictor."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_pages, size=n).tolist()


def phase_changing(n=60000, seed=0):
    """Sequential -> cyclic -> random. Tests the adaptive layer."""
    third = n // 3
    return (sequential(third, start=0, stride=1)
            + cyclic(third, loop_len=25, start=5000)
            + uniform_random(n - 2 * third, seed=seed))


GENERATORS = {
    'sequential':     sequential,
    'strided':        strided,
    'cyclic':         cyclic,
    'clustered':      clustered,
    'random':         uniform_random,
    'phase_changing': phase_changing,
}


def write_trace_csv(pages, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['access_id', 'page_id'])
        w.writerows(enumerate(pages))


def generate_synthetic(out_dir=TRACE_DIR):
    """Generate all six synthetic traces into out_dir."""
    written = {}
    for name, fn in GENERATORS.items():
        pages = fn()
        path = os.path.join(out_dir, f'{name}.csv')
        write_trace_csv(pages, path)
        written[name] = len(pages)
    return written


# --------------------------------------------------------------------------
# Real trace conversion (Stage 1b)
# --------------------------------------------------------------------------
def _open(path):
    """Transparently handle .xz, .gz or plain text."""
    if path.endswith('.xz'):
        return lzma.open(path, 'rt')
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'rt')


def convert(in_path, out_path, addr_col=2, is_hex=True, delimiter=',',
            limit=MAX_ACCESSES, page_shift=PAGE_SHIFT):
    """
    Convert an ML-DPC LoadTrace
        (instr_id, cycle, load_addr, ip, llc_hit)
    into access_id,page_id. `addr >> page_shift` maps bytes to 4 KB pages.
    Returns (rows_written, rows_skipped).
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    n, skipped = 0, 0
    with _open(in_path) as fin, open(out_path, 'w', newline='') as fout:
        w = csv.writer(fout)
        w.writerow(['access_id', 'page_id'])
        for line in fin:
            parts = line.strip().split(delimiter)
            if len(parts) <= addr_col:
                skipped += 1
                continue
            try:
                addr = int(parts[addr_col].strip(), 16 if is_hex else 10)
            except ValueError:
                skipped += 1
                continue
            w.writerow([n, addr >> page_shift])
            n += 1
            if n >= limit:
                break
    return n, skipped


def parse_download_links(path):
    """Extract {benchmark_name: url} for LoadTraces from ChampSim/download_links."""
    urls = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2 and 'LoadTraces' in p[0]:
                urls[p[0].split('/')[-1].replace('.txt.xz', '')] = p[1]
    return urls


# --------------------------------------------------------------------------
# Normalised trace object (Stage 2)
# --------------------------------------------------------------------------
class Trace:
    """One workload, normalised. Knows nothing about where it came from."""

    def __init__(self, name, pages, train_frac=0.25):
        self.name = name
        self.pages = np.asarray(pages, dtype=np.int64)
        self.deltas = np.diff(self.pages, prepend=self.pages[0])
        self.split = int(len(self.pages) * train_frac)

    @property
    def train(self):
        """Warm-up portion. Model MAY learn from this."""
        return self.pages[:self.split]

    @property
    def test(self):
        """Evaluation portion. Metrics measured ONLY here."""
        return self.pages[self.split:]

    def stream(self, portion='all'):
        """Yield (index, page, delta) one at a time. No lookahead."""
        src = {'all':   (0, len(self.pages)),
               'train': (0, self.split),
               'test':  (self.split, len(self.pages))}[portion]
        for i in range(*src):
            yield i, int(self.pages[i]), int(self.deltas[i])

    def __len__(self):
        return len(self.pages)

    def __repr__(self):
        return (f'<Trace {self.name}: {len(self)} accesses, '
                f'{len(np.unique(self.pages))} unique pages, split@{self.split}>')


def load_trace(name, trace_dir=None, limit=None, train_frac=0.25):
    d = trace_dir or TRACE_DIR
    df = pd.read_csv(os.path.join(d, f'{name}.csv'))
    pages = df['page_id'].values
    if limit:
        pages = pages[:limit]
    return Trace(name, pages, train_frac)


def available_traces(trace_dir=None):
    d = trace_dir or TRACE_DIR
    return [n for n in SYNTHETIC + REAL if os.path.exists(os.path.join(d, f'{n}.csv'))]


def load_all(trace_dir=None, limit=200_000):
    traces = {n: load_trace(n, trace_dir, limit=limit) for n in available_traces(trace_dir)}
    assert all(hasattr(t, 'pages') for t in traces.values()), 'TRACES holds raw lists'
    return traces
