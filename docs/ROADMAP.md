# Roadmap: Phases 6-13

Stages 1-5 (data, normalisation, predictor, memory engine, decision engine) plus the head-to-head and the ghost-list estimate are **done**.
Work continues as phases. Order was chosen so that each phase makes the next one's results meaningful.

| Phase | Focus | Effort | Depends on | Status |
|---|---|---|---|---|
| 6 | Evaluation harness (segments, memory sweep, seeds, figures) | ~1 h compute | none | code built (`scripts/08`), **deferred to Phase 12** |
| 7 | Consolidate (clean notebook, update documents, push repo) | half a day | 6 | notebook cleaned; **documents stale; push on hold** |
| 8 | Stronger baselines: ARC, LRU-2, Linux readahead | 1 day | 7 | **done; real-traces run 2026-10-19 (RESULTS section 9)** |
| 9 | Prediction-aware replacement (main new contribution) | 2-3 days | 8 | **tried: stale-prefetch eviction and predictor protection give no gain (RESULTS 9); readahead+Markov combination and gating adopted instead** |
| 10 | Better predictor | 2-3 days | 9 | **delta-Markov tier and top-2 done (RESULTS 9-10); smoothed confidence rejected** |
| 11 | Tune and freeze | 1 day | 9, 10 | not started |
| 12 | Final robustness on the finished system | 1-2 days | 11 | **13 real traces x 2 memory sizes (script 11) and break-even cost model (script 12) done; measured-latency time model and seeds still open** |
| 13 | Write-up (paper, deck, documents, README) | 2 days | all | not started |

Minimum worthwhile set if time is short: **6, 7, 8, 9** (robust numbers, a strong baseline, one genuinely new mechanism).

The user chose to **defer Phase 6 to the end** because Phase 12 reruns the same harness on the final system. A cheap early check is optional:
`437.leslie3d-s1` at 1% (one download, one point, ~5 minutes); if it lands near 35% the headline is safe, if not we find out early.

---

## Phase 6: evaluation harness (built, deferred)

Why: every claim so far is one measurement under one chosen condition. Each part removes one way the result could be luck.

- **6a segments**: `410.bwaves-s0/s1/s2`, `437.leslie3d-s0/s1/s2` at 1% of working set; mean +- std of LRU, OPT, always, ghost. Tests the headline 35%.
- **6b memory sweep**: 0.5/1/2/5% of working set on leslie3d, bwaves, pr-3, bfs-3, clustered. Shows where gains fade and whether the system ever turns harmful.
- **6c seeds**: clustered, random, phase_changing over 5 seeds.
- **6d figures**: head-to-head, decider safety-vs-benefit, eviction inflation, segments, memory sweep (200 dpi).

Run: `python scripts/08_evaluation_harness.py` (`--quick` for a smoke test). Done when the 35% has a spread across three segments and the sweep shows where it disappears.
Risk: if `-s1`/`-s2` are much weaker than `-s0`, the headline number changes. Better to learn that early.
Timing: an early estimate of 25-40 min for the sweep was a guess; ~4 s per 200k-access configuration was measured on synthetic traces, so expect minutes. Untimed on real traces.

## Phase 7: consolidate

- Notebook: done (`predictive_paging_clean.ipynb`: runs top to bottom, duplicates removed; the old cell that overwrote `TRACES` with raw lists now writes `SYNTH_RAW`).
- Repo: ghost-list code and scripts 06-09 are in the library/scripts. Pushing is the user's call.
- Documents still to update: see `REVIEW_PREP.md` section 4 (ghost estimate is now complete; correct "0.78" and "34.5%" claims; refresh the deck).

## Phase 8: stronger baselines (code done)

Added: `ARC`, `LRU2` (engine.py), `ReadaheadPredictor` / `make_readahead` (predictors.py), `POLICIES_EXTENDED`, `scripts/09_phase8_baselines.py`, tests.
Why: a reviewer will ask why only textbook baselines; ARC is also the bar Phase 9 must clear (if the new policy beats LRU but loses to ARC it is not worth claiming).

Script 09 does: sanity checks, eviction-only sweep (1/2/5/10% of working set), head-to-head on the five standard configs
(FIFO, Clock, LRU, LRU-2, ARC, RA+LRU, RA+ARC, Markov+LRU, Markov+ARC, OPT; Markov rows use the deployable ghost decider), analysis tables and a figure.

Questions it answers: (1) does ARC/LRU-2 alone close part of the LRU->OPT gap? (2) does prediction still help on top of ARC? (3) does Markov beat readahead on leslie3d/bwaves?

**Next step: run it on the real traces.** Expect possible bad news and report it: ARC may beat LRU+prediction on irregular workloads; readahead may beat Markov on sequential-heavy ones.
Implementation notes: `MemoryEngine` installs pages itself, so ARC drops the textbook tie-break and adapts `p` at install time; `LRU2` has no correlated-reference period; readahead window = RAM/4 because Linux's 32-page cap exceeds the simulated RAM.
Untested hypothesis worth a run (`--with-lru2-pred`): LRU-2 treats a freshly prefetched page as "seen once", so it evicts prefetches first, which is a natural low-priority insertion.

## Phase 9: prediction-aware replacement (main new contribution; no code yet)

Goal: today eviction is plain LRU and knows nothing about predictions, which is why wrong prefetches nearly doubled evictions on `clustered` (1.93x).

Mechanisms, built and measured **one at a time**:
1. **Stale-prefetch eviction**: a prefetched page still unused after a grace period was a wrong guess; evict it before any page known to be useful.
2. **Low-priority insertion**: prefetched pages enter at the LRU end and are promoted only when used (cf. PACMan, Wu et al., MICRO 2011; Linux's inactive list).
3. **Predictor protection**: never choose as victim a page the predictor currently expects next (top-k); approximates OPT's "keep what is needed soonest".
4. **Combined policy** with `CostAwareGhost`.

Design sketch: a new policy `PredictiveLRU` implementing the four-method interface plus `mark_prefetch(page, i)` and a `protected` set; a new simulator
(`run_prefetch_sim_v2`) that calls `mark_prefetch` after a successful `eng.prefetch(...)` and refreshes `policy.protected` from `predictor.predict(history, top_n=k)` each step
(still prefetch only the top-1 guess). Use `GhostMemoryEngine`; create a shadow engine only for `CostAwareFaultDelta`.
**Sanity check required**: with every mechanism off, the new simulator + `LRU` + `CostAwareGhost` must reproduce the existing ghost numbers exactly (tests pin clustered@9 = 5,798).

Ablation over the eight standard configs: LRU; LRU + ghost (current best); each mechanism alone; each mechanism with **no prefetching** (does predictor-guided eviction help by itself?);
the full combination; ARC from Phase 8. Measure `vs_LRU`, eviction inflation, precision and `pf_harm`.

Targets: eviction inflation on clustered well below 1.24x; harmful configurations reach >= 0; leslie3d keeps ~35%; beat ARC on regular workloads.
Risk: protection can backfire when the predictor is confidently wrong (clustered): keep a partial result if only some mechanisms help; that is still reportable.

## Phase 10: better predictor

- **Delta-based Markov**: learn gaps, not page numbers, so one entry covers a sequential scan anywhere (fixes the structural "Markov = 0 on large loops" and may shrink the table). Add as a hybrid tier; compare with/without. This is the only item with real novelty potential, and the riskiest.
- **Calibrated confidence**: currently confidence cannot discriminate (no prediction < 0.5) and is miscalibrated on clustered. Require a minimum observation count before reporting high confidence, or smooth counts. Check `fixed-0.6` stops equalling `always`.
- **Top-2 prefetching**: only after Phase 9, because low-priority insertion makes a wrong second guess cheap.
- Judge every change by page faults in the full system.

## Phase 11: tune and freeze

Sweep on the **25% warm-up portion only**, choose the simplest setting within noise of the best, freeze, record the justification in `config.py`.

| Component | Parameters | Known issue |
|---|---|---|
| Ghost list | refault window (250/500/1,000/2,000), size (64/128/256) | pr-3 correlation -0.975 |
| Controller | epoch, dead band, step, starting threshold | oscillates on bfs-3@58; too cautious on phase_changing (17.6% vs 50.3%) |
| Phase 9 policy | grace period, protection depth | not yet chosen |
| Phase 10 predictor | smoothing, delta context length | not yet chosen |

Risk: tuning on evaluation data by accident. Keep the split explicit in every sweep function.

## Phase 12: final robustness

Rerun the Phase 6 harness on the frozen final system; add `433.milc`, `462.libquantum` (regular) and `429.mcf` (stress); add a simple **time model**
(cost per demand fault, prefetch I/O overlapped with compute, CPU cost per prediction) so results can say "faster", not just "fewer faults".
Including prediction cost makes our own overhead count against us, which strengthens the cost-aware argument. Done when every headline number has error bars and covers >= 6 real benchmarks.

## Phase 13: write-up

Paper (intro, background, design, methodology, results, discussion, limitations, related work), plus updated deck (monochrome + amber), guide, concepts PDF, README.
Target: undergraduate research or regional IEEE conference, framed as an evaluation study plus a mechanism, not a new-algorithm claim.

## Other known gaps (not scheduled)

- Figures for attribution, capacity sweep and phase adaptation exist only in the notebook (see `NOTEBOOK_MAP.md`).
- Fault-delta oracle uses running totals; a per-epoch variant would react faster to phase changes (the ghost version already is per-epoch).
- The Markov predictor could be compared against an LSTM (a fair "future work" comparison, since our cost model counts predictor cost).
- (fixed) Script 02 now uses urllib.

---

## Phases 14-18 (added 2026-10-19, after the improvement work; RESULTS sections 9-12)

The adopted system is now readahead + delta-Markov (`StreamMarkovPredictor` + `DeltaHybridPredictor`) with the ghost decider. These phases replace the open parts of 11-13.

| Phase | Focus | Done when | Status |
|---|---|---|---|
| 14 | **Fix the feedback estimate's bias** (ghost list says "winning" on clustered@9 while losing) | clustered@9 no worse than LRU; real-trace gains kept; estimate vs true delta checked per epoch | **done**: shadow directory; clustered@9 -21.5% -> -2.5%, worst case over 26 real points -1.4% (RESULTS 13) |
| 15 | **Time model** with stated latencies (SSD, HDD), overlapped prefetch I/O, CPU cost per prediction | every headline shows estimated time vs LRU, not only faults | **done** (script 14): NVMe 1.47x geomean for RA + delta-Markov, HDD slower than LRU for every prefetcher (RESULTS 14) |
| 16 | **Tune and freeze** controller parameters (epoch, dead band, step, start threshold) on the warm-up portion only | values frozen in `config.py` with the reason | **done**: epoch 1000, step 0.1, dead band 0.002 (`config.CONTROLLER_FROZEN`, `paging.final`; RESULTS 15) |
| 17 | **Robustness** on more benchmarks (bc, cc, sssp, soplex, omnetpp, astar) and the Phase 6 harness on the final system | headline numbers hold or fail visibly on >= 20 real traces | **done**: 22 traces; faults +0.550 mean, but time ties readahead (RESULTS 16). Phase 6 seeds not rerun |
| 18 | **Write-up**: paper draft, refreshed documents, report | draft reviewed by the team | **paper draft 1 written** (`docs/paper_draft.md`); report in `results/report.html`; deck and stale PDFs need the team's input |
