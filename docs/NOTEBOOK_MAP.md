# Notebook to repo map

## The current notebook (2026-10-27)

`notebooks/predictive_paging_full_pipeline.ipynb` is one self-contained Colab notebook in the style of the original
(every class defined in a numbered `#cell N` cell, tables and plots after each step, everything saved to
`MyDrive/paging_sim/`). It needs no zip and no clone: only Drive. Cells 0a-0e are traces, cells 1-19 keep the original
numbering (so "cell 17/18/19" in the documents still means the same thing), cells 20-31 are Phases 8-18
(ARC/LRU-2/readahead, delta-Markov, shadow directory, final system, 22 traces, break-even, time model, tuning,
component comparison, results at a glance, consistency check).

It is **generated**: `python scripts/build_notebook.py` copies the class code from `src/paging` with `inspect`, so the
notebook runs exactly the tested code. `tests/test_notebook.py` fails if `src/paging` changed and the notebook was not rebuilt.
Do not edit the notebook by hand; edit `scripts/build_notebook.py` (or `scripts/consistency_cell.py` for cell 31) and rebuild.

**The user's original notebook, extended (2026-10-27).** `os-predictive-memory-paging-system.ipynb` (kept outside the repo,
in the user's Downloads / Drive) has its original cells 1-19 unchanged, plus cells 20-34 appended by
`python scripts/extend_original_notebook.py <original.ipynb> <output.ipynb>`: setup, final core classes (with a check that
cell 17 is reproduced), 22 traces, Phases 8-18, comparison graphs (cell 33, code in `scripts/comparison_graphs_cell.py`,
figures also in `results/figures/comparison/`) and the consistency check (cell 34). Verified locally: every number the
appended cells produce is identical to `results/` (4-trace subset), and all 18 consistency checks pass on the full results.

The rest of this file describes the *original* notebook.

The work started in a Google Colab notebook (cells numbered `#cell N`). The repo is the maintained version. Use this when a number in an old
document says "cell 17" or when you need the original plotting code.

The full code of the Stage 1-5 notebook cells, including the plots, is in `docs/implementation_guide.md`. The Phase 6-8 cells exist in the user's
`predictive_paging_clean.ipynb` (cells 20-30) and as scripts 08 and 09 here.

| Notebook | Repo | Ported? |
|---|---|---|
| SETUP (Drive, paths, constants) | `config.py`, env vars `PAGING_DATA_DIR` / `PAGING_RESULTS_DIR` | yes |
| Stage 1a: synthetic generators, save, plots | `traces.py`, `scripts/01` | yes, plots not |
| Stage 1b: download, convert (`convert`) | `traces.convert`, `parse_download_links`, `scripts/02` | yes (uses `wget`) |
| Trace characterisation (reuse, top-5 delta share) | none | **no** |
| Stage 2: `Trace`, `load_trace`, verification asserts | `traces.py`; asserts in `load_all` and `tests/` | yes |
| cell 1 predictors, cell 2 `eval_predictor` | `predictors.py`, `simulate.py` | yes |
| cell 3 rebuild `TRACES`, cell 4 `FROZEN` + `make_predictor` | `traces.load_all`, `config.FROZEN`, `predictors.make_predictor` | yes |
| cell 5 results tables and heatmap | `scripts/03` (CSV + printed pivot) | heatmap **no** |
| cell 6 attribution | `scripts/03` (CSV) | bar chart **no** |
| cell 7 memory footprint | `scripts/03` (printed) | yes, but counts the **order-2 table only** (see DECISIONS section 3) |
| cell 8 capacity sweep | `scripts/03` (CSV) | dual-axis chart **no** |
| cell 9 phase adaptation, recovery time | `simulate.rolling_accuracy` | stacked panels, zoom and recovery-time table **not ported** |
| cell 10 stage-3 summary JSON | none | no |
| cells 11, 12, 12b engine, FIFO/LRU/Clock, OPTFast | `engine.py` | yes |
| cells 13, 14 baselines and verification | `scripts/04`, `tests/test_core.py` | yes |
| cell 14b confidence diagnostic | none | **no** (source of the confidence percentiles in RESULTS) |
| cells 15, 15b deciders | `deciders.py` | yes |
| cell 16 simulator | `simulate.run_prefetch_sim` | yes |
| cell 17 ablation and summary views | `scripts/05` | yes; some harm/benefit views are CSV-only |
| cell 18 head-to-head | `scripts/06` | yes |
| cell 19 ghost-list estimate | `engine.GhostMemoryEngine`, `deciders.CostAwareGhost`, `simulate.run_prefetch_sim_ghost`, `scripts/07` | yes |
| cells 20-24 (Phase 6) | `scripts/08` | yes (not run on real traces) |
| cells 25-30 (Phase 8) | `engine.ARC/LRU2`, `predictors.ReadaheadPredictor`, `scripts/09`, `tests/` | yes |

## Not yet ported (good first tasks)

1. Figures for attribution, capacity sweep and phase adaptation (+ recovery times), plus the heatmap and the trace-characterisation table, as a script such as `scripts/10_figures.py` (monochrome + amber).
2. The cell-14b confidence diagnostic (how many predictions name resident pages; confidence percentiles).
3. Replace `wget` in script 02 with `urllib` (Windows).
4. Fix the footprint accounting in script 03 to report the full hybrid (and the ghost list).

## Colab / Drive details (when the user runs long jobs in Colab)

- `BASE = /content/drive/MyDrive/paging_sim`; traces in `traces/`, raw `.txt.xz` in `raw/`, figures in `figures/`; Drive files from earlier runs:
  `ablation_results.csv`, `baseline_results.csv`, `capacity_sweep.csv`, `decider_summary.csv`, `head_to_head.csv`, `memory_footprint.csv`, `predictor_attribution.csv`,
  `predictor_results.csv`, `trace_characteristics.csv`, `ghost_estimate_results.csv`, `stage3_summary.json`, `frozen_config.json`, `trace_urls.json`.
- To use the repo on Colab: clone it, `pip install -r requirements.txt`, set `PAGING_DATA_DIR` and `PAGING_RESULTS_DIR` to Drive folders.
- Real trace CSVs for this repo: copy `paging_sim/traces/{410.bwaves-s0,437.leslie3d-s0,bfs-3,pr-3}.csv` into `data/traces/`.
