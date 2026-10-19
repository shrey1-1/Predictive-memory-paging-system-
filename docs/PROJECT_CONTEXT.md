# Project context

Everything needed to understand what this project is and why it is built the way it is.
Numbers live in `RESULTS.md`; the plan lives in `ROADMAP.md`.

## 1. The problem

A program uses more memory than fits in RAM, so the OS keeps hot **pages** (fixed 4 KB blocks) in RAM and the rest on
disk. Requesting a page that is not resident is a **page fault**: the program stalls while the page is read from disk
(~100,000x slower than RAM). Classical paging is *reactive*: it waits for the fault, and its only decision is which
page to evict when RAM is full (FIFO, LRU, Clock; OPT/Belady is the unattainable optimum).

## 2. The idea

Be *proactive*: predict the next page and load it before it is asked for. But prefetching is a bet. A right guess
removes a fault. A wrong guess wastes I/O and, if RAM is full, **evicts a page that may be needed sooner, creating a
fault**. So a second component, the **decision engine**, decides per prediction whether acting is worth the risk, and
adapts using measured outcomes.

Prediction sits **on top of** an eviction policy. The predictor decides what *arrives* early; the eviction policy
decides what *leaves*. (Phase 9 will let the predictor also inform eviction.)

## 3. Architecture (per memory access)

1. **Memory engine** checks residency: hit, or fault (load the page; the eviction policy picks a victim if RAM is full).
2. **Hybrid predictor** learns the transition just seen, then predicts the next page plus a confidence.
3. **Decision engine** decides prefetch yes/no from confidence, free frames, a budget cap and an adaptive threshold.
4. Approved prefetches go through the **same install path** as demand faults (`MemoryEngine._install`), so prediction
   changes speed, never correctness. Disabling prediction leaves a correct plain-LRU simulator.
5. **Feedback**: every 2,000 accesses (an *epoch*) the threshold is raised if prefetching is losing and lowered if it is winning.

Design principle: only `(page, confidence)` crosses between predictor and decision engine. The predictor never sees RAM
state; the decision engine never sees the model. That separation is what allows seven decision strategies to be compared
against byte-identical predictions, so any difference is caused by the decision policy alone.

## 4. Components

| Module | Contents |
|---|---|
| `config.py` | paths, `PAGE_SHIFT=12`, `TRACE_LIMIT=200_000`, `WARMUP_FRAC=0.25`, `FROZEN` predictor config, `ABLATION_CONFIGS` |
| `traces.py` | six synthetic generators, ML-DPC converter, `Trace` (pages, deltas, time-ordered 25% train split, `stream()`), `load_all` |
| `predictors.py` | `MarkovPredictor`, `StridePredictor`, `HybridPredictor`, `ReadaheadPredictor`, `make_predictor`, `make_readahead` |
| `engine.py` | `MemoryEngine`, `FIFO`, `LRU`, `Clock`, `OPTFast`, `ARC`, `LRU2`, `GhostMemoryEngine`, `POLICIES`, `POLICIES_EXTENDED`, `run_policy` |
| `deciders.py` | `NoPrefetch`, `AlwaysPrefetch`, `FixedThreshold`, `CostAware` (precision), `CostAwareFaultDelta` (oracle), `CostAwareGhost` (deployable) |
| `simulate.py` | `eval_predictor`, `rolling_accuracy`, `run_prefetch_sim`, `run_prefetch_sim_ghost` |

## 5. Data

**Synthetic (six, in `traces.py`)**: `sequential` (+1, 10,000-page loop), `strided` (+8, 1,250 pages), `cyclic` (20-page loop),
`clustered` (random within a 10-page neighbourhood, jump every 500 accesses), `random` (uniform), `phase_changing`
(20k sequential, 20k cyclic, 20k random). 50,000 accesses each (60,000 for phase_changing); stochastic ones take `seed`.
Known ground truth makes them tests of correctness; real traces test usefulness.

**Real (four)**: ML-Based Data Prefetching Competition **LoadTraces**, listed in `github.com/Quangmire/ChampSim`
(`download_links`, hosted on UT Austin Box). `410.bwaves-s0`, `437.leslie3d-s0` (SPEC CPU 2006, regular numerical code)
and `bfs-3`, `pr-3` (GAP graph benchmarks, irregular). Line format `instr_id, cycle, load_addr (hex), ip, llc_hit`;
page = `int(load_addr, 16) >> 12`. Capped at 200,000 accesses. Working sets: bwaves 3,051 pages, leslie3d 2,134,
bfs-3 5,846, pr-3 4,062. Other benchmarks available in the same list: milc, mcf, libquantum, soplex, GemsFDTD, bc, cc, sssp, and `-s1`/`-s2`
segments of each program.

**Caveat to always state**: these are *last-level-cache* accesses, already filtered by L1/L2, treated at page
granularity. That approximates, but is not, OS paging. (It is also why stride prediction does badly on them: the clean strides were absorbed upstream.)

**Method**: trace-driven simulation (reproducible, identical input for every policy, and OPT needs the recorded future).
Time-ordered split (never shuffled). Memory is sized as 1-25% of each trace's working set.

## 6. The predictor

**Markov chain**: a table of counts of "context -> next page". "Training" is counting; it keeps learning online during evaluation.
Order-2 context = last two pages, order-1 = last page. Order-2 is more specific (lower coverage, higher accuracy).
Kept lightweight on purpose because it runs on every access: bounded LRU table (256 contexts), top-2 successors per
context, integer counts, counts halved every 25,000 updates (so old behaviour fades). Footprint counted as a kernel would
store it (16 B per order-2 context + 9 B per successor): the **order-2 table is 8.5 KB** (8,704 B; ~2 pages; 1.66% of RAM at 128 frames).
**The hybrid also keeps an order-1 table** (256 x 8 B + 512 x 9 B = 6,656 B), so the full hybrid is **~15 KB (3.75 pages; 2.9% of RAM at 128 frames, 0.37% at 1,024)**;
adding the ghost list (below) gives ~19 KB. Older documents quote only the 8.5 KB order-2 figure.
(`sys.getsizeof` gives ~1.1 MB because of Python object overhead. Do not use it for this claim.)

**Frozen config**: `capacity=256, max_succ=2, order=2, decay_every=25_000`. Chosen because accuracy on real traces is
flat from 128 to 4,096 entries while memory grows 32x; `max_succ` 2/4/8 differ by <0.0002; order-2 beats order-1 by ~4.5 points.

**Stride**: last three gaps equal -> predict last + gap, confidence 1.0, no state.

**Hybrid**: tries order-2, then order-1, then stride, else no prediction. *Prediction is first-answer-wins; learning
happens in both Markov tables on every access* (so a table that is not currently answering can take over after a phase change).
A Markov model memorises raw page numbers, so sequential/strided traces whose working set exceeds 256 entries get zero
Markov coverage and are handled entirely by the stride fallback (a delta-based Markov model, Phase 10, would fix that structurally).

**Readahead** (Phase 8 baseline): Linux-style. Two consecutive pages start a stream; an initial window is fetched and doubles
each time the half-way marker is reached, up to a cap; any break resets it. Repeats of the same page are ignored.

## 7. Decision engines

`always` (naive control), `fixed-0.3/0.6/0.85`, `cost-aware` (precision-driven), `cost-aware-fd` (fault-delta), `cost-aware-ghost`
(fault-delta without a shadow engine). Common gates: a budget cap (prefetches <= 35% of accesses); free frames => accept low
confidence (a wrong prefetch evicts nothing); RAM full => require confidence above an adaptive threshold (0.15-0.95).

**Fault delta.** A *shadow engine* replays the same accesses with prefetching off. `delta = shadow_faults - real_faults`.
Every 2,000 accesses: delta rate above +0.002 -> lower the threshold by 0.05 (bolder); below -0.002 -> raise it (back off);
inside the dead band -> leave it. The oracle version uses running totals from trace start. It is an **oracle**: a real kernel cannot run a shadow copy.

**Ghost list (deployable replacement).** When a *prefetch* evicts a page, remember the page ID and the access index in a
bounded list (256 entries of page ID + access index = ~4 KB; ~2 KB if the timestamps are dropped). On each fault, if the faulting page is in the list and was evicted within `refault_window`
(2,000) accesses, count one unit of prefetch-caused harm. Estimated delta per epoch = prefetches used - harmful refaults.
Same idea as Linux `mm/workingset.c` shadow entries and ARC ghost lists.

**Why precision was rejected**: precision = fraction of prefetched pages later used. A used prefetch can still be net harmful
if installing it displaced a page needed sooner. On `clustered @ 9 frames` four deciders had precision 0.81-0.86 and still lost faults.

## 8. Metrics

| Metric | Definition |
|---|---|
| coverage | predictions made / accesses |
| accuracy | correct predictions / predictions made |
| useful | coverage x accuracy |
| `vs_LRU` | (LRU faults - faults) / LRU faults; positive = better |
| precision | prefetches used / prefetches issued |
| eviction inflation | evictions with prefetching / evictions without |
| headroom | (LRU faults - OPT faults) / LRU faults |

Compulsory (first-touch) faults are unavoidable for any eviction policy; only prefetching can avoid them. That is why prefetching can beat OPT (bwaves).

## 9. Environment and history

- Original development in **Google Colab** with data on Drive: `/content/drive/MyDrive/paging_sim/{traces,raw,figures}`. Colab wipes memory on reconnect.
- The user now develops locally on Windows (Git Bash, VS Code) and is moving to Claude Code.
- GitHub: `shrey1-1/Predictive-memory-paging-system-` (note the trailing hyphen). Pushing was deliberately on hold.
- Original design had six stages (data, preparation, training, runtime, decision engine, evaluation); work is now tracked as Phases 6-13 (`ROADMAP.md`).

## 10. Deliverables already produced (outside this repo)

| File | What it is |
|---|---|
| `predictive_paging_guide_stages_1_5.pdf` / `.md` | 58-page guide, every notebook cell's code and results (copy in `docs/implementation_guide.*`) |
| `Predictive_Memory_Paging_Concepts_and_Modules.pdf` | 27 pages: modules, concepts, Markov, fault delta, paper comparison, glossary |
| `Predictive_Memory_Paging_Project_Review.pptx` / `.pdf` | 13-slide review deck, monochrome + amber, with the user's screenshots (a first, blue deck also exists and is superseded) |
| `predictive_paging_clean.ipynb` | Phase 7 notebook: cleaned, ordered, Phase 6 cells appended (cells 20-24) |
| `COLAB_CONTINUATION_BRIEF.md`, `PROJECT_HANDOFF.md` | older handoffs; superseded by this repo's docs |

Several of these are now stale: see `REVIEW_PREP.md`.
