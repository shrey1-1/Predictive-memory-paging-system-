"""
Central configuration. Paths can be overridden with environment variables,
so the same code runs locally and on Google Colab (point PAGING_DATA_DIR at Drive).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_DIR    = os.environ.get('PAGING_DATA_DIR', os.path.join(ROOT, 'data'))
RAW_DIR     = os.path.join(DATA_DIR, 'raw')      # compressed originals (.txt.xz)
TRACE_DIR   = os.path.join(DATA_DIR, 'traces')   # converted CSVs
RESULTS_DIR = os.environ.get('PAGING_RESULTS_DIR', os.path.join(ROOT, 'results'))

PAGE_SHIFT   = 12          # 4 KB pages (2^12)
MAX_ACCESSES = 500_000     # cap when converting real traces
TRACE_LIMIT  = 200_000     # cap when loading for experiments
WARMUP_FRAC  = 0.25

# FROZEN predictor hyperparameters (Stage 3). Do not change after evaluation.
#   capacity=256  -> useful_real flat (0.783) from 256 to 4096; 8.5 KB table
#   max_succ=2    -> 2/4/8 differ by < 0.0002
#   order=2       -> beats order-1 by ~4.5 points
#   decay=25000   -> marginally best
FROZEN = dict(capacity=256, max_succ=2, order=2, decay_every=25_000)

# Real benchmarks used in the evaluation
PICK = ['410.bwaves-s0', '437.leslie3d-s0', 'bfs-3', 'pr-3']

# Stage 5 ablation configurations: (trace, frames)
ABLATION_CONFIGS = [
    ('437.leslie3d-s0', 21),
    ('410.bwaves-s0',   30),
    ('clustered',        6),
    ('clustered',        9),
    ('bfs-3',           20),
    ('bfs-3',           58),
    ('pr-3',            15),
    ('pr-3',            40),
]


def ensure_dirs():
    for d in (DATA_DIR, RAW_DIR, TRACE_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)
