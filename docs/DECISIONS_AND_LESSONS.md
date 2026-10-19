# Decisions, corrections and lessons

Why things are the way they are, what we got wrong along the way, and what to watch for.

## 1. Design decisions and the reasoning

| Decision | Why |
|---|---|
| Trace-driven simulation, not a kernel hook | Reproducible; identical input for every policy; OPT needs the recorded future; far less engineering |
| Python | Fast iteration on predictor logic; the hot loop is dict lookups. A C++ port of the engine is only worth it for throughput claims |
| Markov chain, not an LSTM/CNN | It runs on every access, so its own cost counts against what it saves. One lookup + one counter increment; kilobytes of state; interpretable. A learned model is a fair future comparison |
| Hybrid chain (order-2 -> order-1 -> stride) | Order-2 is precise but narrow, order-1 broad, stride needs no memory and covers loops larger than the table |
| Bounded table, top-2 successors, integer counts, periodic halving | Keeps cost constant and the footprint a hard ceiling; old behaviour fades so the model tracks the current phase |
| Frozen config capacity 256 | Accuracy on real traces is identical from 128 to 4,096 entries; 16x memory buys nothing. The only traces that need more are synthetic loops, which stride covers |
| Memory sized as 1-25% of the working set | With fixed frame counts (64-512) bwaves gave identical results under every policy: nothing was evicted, so nothing was being tested |
| Time-ordered 25% warm-up, never shuffled | Shuffling destroys the sequence being modelled |
| Predictor and decider separated | Fair ablation; swap in another predictor without touching the decider |
| Prediction runs on top of the eviction policy | We never replace LRU; "adding prediction to LRU cut faults by 35%", not "our algorithm beat LRU" |
| Fault-delta feedback, not precision | Precision measures *used*, not *helpful* (see section 2) |
| Ghost list instead of a shadow engine | A kernel cannot run a second copy of memory; harm can be estimated from refaults of pages a prefetch evicted (cf. Linux `workingset.c`, ARC ghost lists) |
| `always` decider kept as a control | If it hurts somewhere, prediction alone is not enough: the empirical case for a decision engine |
| Information-theoretic memory accounting | `sys.getsizeof` measures CPython overhead (~1.1 MB here) not the algorithm |
| Monochrome + amber in figures and decks, no blue | User's explicit instruction |

## 2. Findings that shaped the design (all [real], see RESULTS.md)

- Confidence cannot discriminate: no prediction is below 0.5, so `always` and `fixed-0.3` are bitwise identical. **This is structural**: with at most 2 successors the top one always holds >= 50% of the counts (pinned by a test), so it is a consequence of the predictor design, not a measured property of the workloads. Calibration needs more successors or observation counts (tried smoothing: no gain, RESULTS section 9). The discriminating signal has to come from system state.
- Precision-driven adaptation first collapsed onto the fixed-threshold decider (adaptation never engaged), then pinned to the rails (updating on every event), then, once throttled, produced the *worst* result in the table on clustered@9 (-0.0122), worse than `always`.
- The first versions of the controller were fixed by: reporting wasted prefetches too, `min_samples` 20, `update_every=100` plus a target precision band. The fault-delta decider replaced all of it.
- Sequential/strided Markov = 0 is a **capacity artefact**, not a bug: the working set exceeds the table. The capacity sweep proved it (sequential needs ~2^14 entries).

## 3. Corrections: claims that were wrong or loose. Do not repeat them.

1. **Precision vs benefit**: an informal reading said they were uncorrelated. Measured correlation is **+0.73**. The defensible claim is narrower: precision is insufficient *under high eviction pressure*.
2. **Fault-delta uses running totals**, not per-epoch counts, in the oracle decider (cumulative real vs shadow faults since the trace start). Slower to react after a phase change. The ghost decider is per-epoch.
3. **Hybrid mean useful rate on real traces is 0.8115.** "0.78 / 0.783" is the order-2 Markov mean in the capacity sweep. Several documents quote 0.78 for the hybrid.
4. **Headline range**: "34.5-37.3% vs the best traditional algorithm" mixes definitions. LRU+pred vs best traditional is **34.2%** (leslie3d, best is Clock) and 37.3% (bwaves). 34.5% is Clock+pred vs Clock.
5. **Predictor memory**: the "8.5 KB / 1.66% overhead" figure is the **order-2 table alone** (256 x 16 B contexts + 512 x 9 B successors = 8,704 B). `HybridPredictor` also keeps an order-1 table of the same capacity
   (256 x 8 B + 512 x 9 B = 6,656 B), so the full hybrid is **~15 KB (3.75 pages), 2.9% of RAM at 128 frames and 0.37% at 1,024**. The deployable ghost list stores a page ID and an access index per entry:
   256 x 16 B = **~4 KB** (about 2 KB only if timestamps are dropped), so the whole deployable predictive state is **~19 KB (3.7% at 128 frames)**. Still lightweight; the claim just needs the right number.
6. **Confidence saturation**: "three quarters of predictions have confidence >= 0.93" holds for leslie3d, bfs-3, pr-3 but not bwaves (25th percentile 0.875) or clustered (25th 0.50, median 0.60, yet accuracy 0.10).
   The accurate, stronger statement is that no prediction is below 0.5.
7. **Timing**: early estimates of 10-40 minutes for several Colab cells were guesses. A 200k-access configuration measured ~4 s on synthetic traces. Real traces and Colab CPUs will be slower; measure before quoting.
8. **bwaves beats OPT** is correct and explained (compulsory faults), but any "percent of headroom" for it exceeds 100% and must be explained, not reported raw.

## 4. Bugs and gotchas already hit

- `'list' object has no attribute 'pages'`: `TRACES` was overwritten by a dict of raw lists from the synthetic-generation cell. Fixed by naming it `SYNTH_RAW`.
- Colab wipes memory on reconnect (`NameError` for `OUT`, `FROZEN`, `convert`...): rerun SETUP and cells in order.
- Truncated paste -> `_IncompleteInputError`: clear the cell and repaste.
- `AlwaysPrefetch` has no `.epoch`, so `run_prefetch_sim_ghost` cannot take it. Use `run_prefetch_sim` for it.
- Row-building loops that listed columns explicitly silently dropped decider-specific stats (`KeyError: 'epochs_losing'`). Loop over a key tuple with `.get`.
- "Share of the LRU->OPT gap closed" explodes when the gap is tiny (printed -30.6). Only compute it when OPT is >= 10% better than LRU; otherwise show `vs_LRU`.
- Box returns an HTML error page for a bad download: validate with `lzma.open` before converting.
- A trace length change makes the cross-check against old results report MISMATCH (that is the check working).
- Notebook cell numbering drifted from the design-doc stage numbering; this repo uses phases and script numbers (see NOTEBOOK_MAP.md).

## 5. Simplifications to state in any write-up

- ARC: textbook tie-break (`|T1| == p` and new page in B2) dropped; `p` adapted at install time (the engine installs pages itself).
- LRU-2: no correlated-reference period.
- Readahead window = RAM/4 (Linux's 32-page cap exceeds our whole simulated RAM); sequential-only (no stride detection).
- LLC traces at page granularity; counts faults, not time; one run per configuration so far.

## 6. Open questions

- Does the 35% survive other trace segments and larger memory? (Phase 6/12)
- Does ARC close part of the gap, and does prediction still add value on top of it? (Phase 8 on real traces)
- Does a smaller refault window fix the ghost estimate on pr-3?
- Can predictor-guided eviction (Phase 9) remove the eviction inflation on clustered?
- Does a delta-based Markov model cover scans in a handful of entries and beat the stride fallback? (Phase 10)
