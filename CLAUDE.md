# CLAUDE.md

**Predictive Memory Paging System with Workload Awareness** — a trace-driven OS paging simulator in Python.
College OS project by three students; the goal is a student / regional-conference research paper.
Team: G.K Shreyas Bharadwaj (24BCE2256), Vaastav Bajanthri (24BCE2223), P. Giphy Bershan (24BCE2201).

## What it does (read this first)

A bounded Markov-chain predictor watches the page-access stream and predicts the next page. A **decision
engine** decides whether each prediction is worth prefetching, because on full RAM every prefetch evicts a page,
so a wrong guess can *cause* a fault. A feedback loop moves the decision threshold using a **fault-delta**
signal (are we getting fewer faults than without prefetching?). Evaluated against FIFO, LRU, Clock, Belady's OPT
and (since Phase 8) ARC, LRU-2 and Linux-style readahead.

Prediction runs **on top of** an eviction policy: the predictor decides what arrives early, the eviction policy
decides what leaves. We never replace LRU, we add to it. Phase 9 (next big piece) makes eviction use the predictor.

Honest framing (keep it): Markov prefetching is Joseph & Grunwald 1997; fault-rate feedback resembles Page Fault
Frequency; prefetch-aware insertion resembles PACMan. Our contribution is the evaluation of *which feedback signal
works*, a *deployable ghost-list estimate* of it, and (planned) prediction-aware replacement. Do not claim the
components are new.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                                   # ~13 s. Must pass before AND after every change.
python scripts/run_all.py                   # whole pipeline, synthetic traces only (a few minutes)
python scripts/09_phase8_baselines.py       # Phase 8 (ARC, LRU-2, readahead). Resumable.
python scripts/02_download_real_traces.py --links ChampSim/download_links   # needs wget (see Pitfalls)
python scripts/build_notebook.py            # regenerate the Colab notebook after any change in src/paging
```
The Colab notebook (`notebooks/predictive_paging_full_pipeline.ipynb`) is generated from `src/paging`; never edit it by hand
(`tests/test_notebook.py` checks it is up to date).
Scripts skip any configuration whose trace is missing, so everything runs on synthetic traces alone.
Paths: `PAGING_DATA_DIR`, `PAGING_RESULTS_DIR` env vars override `data/` and `results/` (used on Colab/Drive).

## Layout

```
src/paging/   config.py (frozen hyperparameters, experiment configs)   traces.py (generators, converter, Trace)
              predictors.py (Markov, Stride, Hybrid, Readahead)        engine.py (MemoryEngine, FIFO/LRU/Clock/OPT/ARC/LRU2, GhostMemoryEngine)
              deciders.py (7 deciders incl. CostAwareFaultDelta, CostAwareGhost)   simulate.py (eval loops, run_prefetch_sim[_ghost])
scripts/      01 generate  02 download  03 predictors  04 baselines  05 ablation  06 head-to-head
              07 ghost estimate  08 evaluation harness (Phase 6)  09 Phase 8 baselines  run_all.py
tests/        test_core.py (invariants + pinned reference numbers)
docs/         PROJECT_CONTEXT, RESULTS, ROADMAP, DECISIONS_AND_LESSONS, REVIEW_PREP, NOTEBOOK_MAP,
              implementation_guide.{md,pdf} (stages 1-5 with full code), system_architecture.md
data/ results/   git-ignored except results/reference/ (the user's real ablation CSV)
```

## Status

| Done | Notes |
|---|---|
| Stages 1-5, head-to-head (script 06), ghost-list estimate (script 07) | validated on the user's real traces in Colab |
| Phase 7 (clean notebook) | code done; document updates pending (see REVIEW_PREP) |
| Phase 8 (ARC, LRU-2, readahead, script 09) | run on the real traces 2026-10-19; readahead beats Markov on 3 of 4 (RESULTS 9) |
| Phases 15-17 | time model (script 14), controller + budget frozen (`config.CONTROLLER_FROZEN`, `paging.final`), 22 real traces. Faults: best of all systems; time: ties readahead (RESULTS 14-16) |
| Phase 14 (shadow directory, `CostAwareDirectory` + `run_prefetch_sim_dir`) | adopted feedback; best worst case (RESULTS 13). Current best system: RA + delta-Markov, `conf_k=1`, directory feedback |
| Improvements (scripts 10-12) | readahead + delta-Markov (`StreamMarkovPredictor`, `DeltaHybridPredictor`, `TopN`, `conf_k` gating); 13 real traces; break-even cost (RESULTS 9-12) |
| Phase 6 harness (script 08) | built, deferred; the plan is to run it on the *final* system in Phase 12 |

**Real traces are not in the repo** (git-ignored; 13 are in `data/traces/` locally, fetched with script 02, which now uses urllib). The original four CSVs (`410.bwaves-s0`, `437.leslie3d-s0`, `bfs-3`, `pr-3`; columns
`access_id,page_id`) live on the user's Google Drive (`paging_sim/traces/`) or can be rebuilt with script 02.
Until they are in `data/traces/`, no result for the real benchmarks can be produced or claimed.

## Next actions, in order

1. Phase 18 write-up (`docs/ROADMAP.md`); stale documents listed in `docs/REVIEW_PREP.md`. Phases 14-17 are done.
   The final system is `paging.final.run_final`; do not retune it on the evaluation traces.
2. Phase 14 is done (shadow directory replaced the ghost list as the adopted feedback; RESULTS 13).
   Reframe: the strongest result is now readahead + delta-Markov, not Markov alone; readahead is a baseline it must beat.

## Hard rules

- **Never tune on evaluation data.** Hyperparameters are frozen in `config.FROZEN`; ghost-list and controller
  parameters are still untuned (Phase 11): sweep on the 25% warm-up portion only, then freeze and record why.
- **Prefetching enters RAM only through `MemoryEngine._install`.** A broken predictor may slow the system, never make it wrong.
- **Size memory as a fraction of each trace's working set** (usually 1%), never fixed frame counts: with ample memory every policy is identical.
- Policies take `pages=` in the constructor and implement `on_insert/on_hit/on_evict/choose_victim`. Deciders implement
  `decide(page, conf, state)`, `feedback(page, used)`, `stats()`. Predictors implement `predict(history, top_n)`, `update(history, actual_next)`, `stats()`.
- When extending something, add a check that the old behaviour is reproduced exactly with the new feature off.
- `tests/test_core.py` pins numbers from the real Colab runs. If one changes, behaviour changed: understand why before touching the number.
- **Never invent results.** Mark synthetic-only numbers as such. A negative result with a diagnosis is acceptable and valued.
- State simplifications openly (ARC drops the textbook tie-break; LRU-2 has no correlated-reference period; readahead window is RAM/4).
- Judge predictor changes by **page faults in the full system**, not prediction accuracy alone (they have diverged before).
- `vs_LRU = (LRU_faults - faults) / LRU_faults`: positive is better, negative means prefetching caused extra faults.
- Do not `git push` or touch any remote unless asked.

## Conventions

- Python 3.9+ compatible. Long runs must save per configuration and resume (pattern in script 09).
- Figures: monochrome greys + one amber accent `#c2700a`. **No blue** (the user ruled out blue gradients/themes for decks too).
- Results go to `results/` (git-ignored); only deliberate reference files go to `results/reference/`.
- Keep scripts runnable on synthetic-only data; print what was skipped.

## Working with the user

- They are students: explain in simple, plain terms; give complete runnable code and say exactly which file it goes in.
- They run long jobs on Google Colab (memory is wiped on reconnect) and develop locally on Windows (Git Bash / VS Code).
  If asked for Colab notebook cells: one complete cell at a time, header `#cell N — what it does`, say where it goes,
  guard assertions at the top, `display()` tables, save to Drive, and avoid modifying earlier cells.
- Ask for screenshots/data before building decks or write-ups; do not fabricate them. Decks: monochrome + amber.
- After producing results, interpret them (what to look for, what would be surprising) and flag caveats.

## Pitfalls

- `scripts/02_download_real_traces.py` uses `urllib` (worked against Box on 2026-10-19). Needs `git clone https://github.com/Quangmire/ChampSim.git` for the links (`ChampSim/` is git-ignored).
- Use **LoadTraces** (`.txt.xz`, plain CSV lines `instr_id, cycle, hex_addr, ip, llc_hit`), never ChampSimTraces (`.gz`, binary).
- Cache-level (LLC) traces treated at page granularity are an approximation of OS paging: always state it.
- `run_prefetch_sim` refuses `CostAwareGhost` (it would silently run as the oracle); use `run_prefetch_sim_ghost`.
- Readahead candidates have confidence 1.0 unless `conf_k` is set, so threshold deciders cannot gate them by default.
- `run_prefetch_sim_ghost` needs a decider with `.epoch` (`CostAwareFaultDelta`/`CostAwareGhost`); `AlwaysPrefetch` has none.
- The oracle `CostAwareFaultDelta` compares **running totals** since the trace start; `CostAwareGhost` is per-epoch.
- Shadow-engine runs roughly double runtime. Measured: ~4 s per 200k-access configuration on a laptop-class CPU (synthetic); real traces will be heavier.

## Read when relevant

`docs/PROJECT_CONTEXT.md` (concepts, architecture, data) · `docs/RESULTS.md` (every number we have, with provenance) ·
`docs/ROADMAP.md` (phases 6-13 in detail) · `docs/DECISIONS_AND_LESSONS.md` (why things are the way they are, corrected claims) ·
`docs/REVIEW_PREP.md` (framing, likely reviewer questions, documents that are now stale) · `docs/NOTEBOOK_MAP.md` (Colab cell to repo map, what is not yet ported).
