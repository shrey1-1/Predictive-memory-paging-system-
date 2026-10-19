# Results

Every number we have, with where it came from. **Provenance tags:**

- **[real]**: run by the user in Colab on the real benchmark traces; transcribed from their screenshots / CSVs.
- **[synthetic]**: reproducible in this repo with no downloads (`pytest`, `scripts/`).
- **[sandbox]**: measured during the handoff session on synthetic traces only. Not evidence about real workloads.

Standard configurations (frames = ~1% of working set): leslie3d@21, bwaves@30, clustered@6 and @9, bfs-3@20 and @58, pr-3@15 and @40.
`vs_LRU` = (LRU faults - faults) / LRU faults; positive is better. Fixed-ratio frame counts were rejected (see DECISIONS).

## 1. Prediction quality (Stage 3) [real]

Frozen config (capacity 256, order 2, 2 successors, decay 25k).

| Trace | stride | markov-o1 | markov-o2 | **hybrid** | hybrid coverage | hybrid accuracy |
|---|---|---|---|---|---|---|
| 410.bwaves-s0 | 0.256 | 0.788 | 0.761 | **0.804** | 0.991 | 0.811 |
| 437.leslie3d-s0 | 0.002 | 0.843 | 0.816 | **0.844** | 0.984 | 0.857 |
| bfs-3 | 0.000 | 0.669 | 0.673 | **0.706** | 0.963 | 0.732 |
| pr-3 | 0.000 | 0.458 | 0.881 | **0.892** | 0.970 | 0.919 |
| clustered | 0.000 | 0.097 | 0.081 | 0.099 | 0.982 | 0.101 |
| cyclic | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| phase_changing | 0.484 | 0.444 | 0.444 | 0.555 | 0.567 | 0.980 |
| random | 0 | 0 | 0 | 0 | 0.025 | 0.000 |
| sequential | 1.000 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| strided | 0.997 | 0 | 0 | 0.997 | 0.998 | 0.999 |

Columns 2-5 are *useful* (coverage x accuracy). **Hybrid mean useful on the four real traces = 0.8115** (all ten traces 0.690).
(0.783 appears in older documents: that is the *order-2 Markov* mean over the real traces in the capacity sweep, not the hybrid.)
Markov = 0 on sequential/strided because their working sets (10,000 / 1,250 pages) exceed the 256-entry table; stride covers them.

**Which sub-predictor answered** (hybrid, capacity 256):

| Trace | order-2 | order-1 | stride | none |
|---|---|---|---|---|
| sequential | 0 | 0 | 1.000 | 0 |
| strided | 0 | 0 | 0.998 | 0.002 |
| cyclic | 1.000 | 0 | 0 | 0 |
| clustered | 0.799 | 0.182 | 0 | 0.018 |
| random | 0 | 0.025 | 0 | **0.975** |
| phase_changing | 0.444 | 0.011 | 0.112 | 0.433 |
| bwaves / leslie3d | 0.908 / 0.926 | 0.076 / 0.058 | 0.006 / 0.001 | 0.009 / 0.016 |
| bfs-3 / pr-3 | 0.850 / 0.927 | 0.113 / 0.042 | 0 / 0 | 0.037 / 0.030 |

**Capacity sweep**: `useful_real` (order-2 Markov) is 0.783 at every capacity from 128 to 4,096 while table size grows 4.2 KB -> 136 KB.
`useful_all` is 0.466 up to 1,024 entries and 0.566 from 2,048 (the `strided` trace then fits; `sequential` needs ~2^14 entries).

**Memory footprint** [real run + arithmetic]: the notebook measured the **order-2 table only**: ceiling 8.5 KB (~2.1 pages). Overhead 1.66% (128 frames), 0.83% (256), 0.42% (512), 0.21% (1,024). Observed max 8.4 KB (clustered).
The full hybrid also has an order-1 table (6.5 KB): **~15 KB total (3.75 pages; 2.9% at 128 frames, 0.37% at 1,024)**, and ~19 KB with the 256-entry ghost list. This is arithmetic from the code, not a notebook output; script 03 does not yet report it.

**Phase adaptation** (phase_changing): the hybrid holds rolling accuracy 1.0 through the sequential and cyclic phases; stride alone drops to ~0.84 on cyclic, Markov alone is 0 on sequential.
Accesses to reach 90% of the post-transition plateau after sequential -> cyclic: stride 9, markov-o1 69, markov-o2 79, hybrid 9.

**Confidence diagnostic** (cell 14b, six configs): no prediction ever below 0.4 (`below_thr` = 0). 10th percentile of confidence is >= 0.5 everywhere
(leslie3d 0.714, bfs-3 0.667, pr-3 1.000, clustered 0.500, bwaves 0.750). 25th percentile: leslie3d 0.927, bfs-3 1.000, pr-3 1.000, bwaves 0.875, clustered 0.500 (median 0.600).
So confidence is near-saturated on leslie3d/bfs/pr, only moderate on bwaves/clustered, and on clustered it is *miscalibrated* (median 0.6, accuracy 0.10).
Share of predictions that name an already-resident page: bfs-3 ~99.8%, pr-3 ~99.7%, leslie3d ~96.8%, bwaves ~99.1%, clustered ~90.6%. That is why graph workloads offer almost nothing to prefetch.

## 2. Eviction baselines (Stage 4) [real]

All checks passed: no policy beats OPT; LRU and OPT never worsen with more memory; sequential/strided fault 100% under FIFO/LRU/Clock.

| Config | FIFO | LRU | Clock | OPT | headroom (LRU->OPT) |
|---|---|---|---|---|---|
| leslie3d@21 | 15,608 | 15,313 | 15,126 | 5,881 | **62%** |
| bwaves@30 | 3,380 | 3,140 | 3,214 | 3,112 | ~1% |
| bfs-3@58 | 6,959 | 6,766 | | 6,179 | 9% |
| bfs-3@116 | 6,696 | 6,461 | | 6,121 | 5% |
| leslie3d@42 | 3,339 | **3,933** | | 3,223 | LRU is 17.8% *worse* than FIFO |

LRU is also worse than FIFO on clustered@9 (fault rates 0.1161 vs 0.1146): with a 10-page cluster and 9 frames LRU evicts the page about to be reused.

## 3. Decision-engine ablation (Stage 5) [real]

`vs_LRU`, LRU underneath, Markov hybrid predictor:

| Config | always | fixed-0.3 | fixed-0.6 | fixed-0.85 | cost-aware (precision) | **cost-aware-fd** |
|---|---|---|---|---|---|---|
| bwaves@30 | 0.3729 | 0.3729 | 0.3729 | 0.3729 | 0.3729 | **0.3729** |
| leslie3d@21 | 0.3500 | 0.3500 | 0.3359 | 0.2890 | 0.3413 | **0.3498** |
| bfs-3@20 | -0.0150 | -0.0150 | -0.0041 | -0.0005 | -0.0045 | **-0.0028** |
| bfs-3@58 | -0.0027 | -0.0027 | -0.0006 | +0.0001 | -0.0006 | **-0.0019** |
| clustered@6 | -0.0087 | -0.0087 | -0.0098 | -0.0081 | -0.0096 | **-0.0072** |
| clustered@9 | +0.0009 | +0.0009 | -0.0093 | -0.0019 | -0.0122 | **-0.0048** |
| pr-3@15 | 0.0051 | 0.0051 | 0.0063 | 0.0059 | 0.0063 | **0.0051** |
| pr-3@40 | 0.0074 | 0.0074 | 0.0069 | 0.0067 | 0.0069 | **0.0074** |

**Summary across the eight configurations**

| Decider | mean gain | worst case | total harm | total gain | benefit kept on leslie3d |
|---|---|---|---|---|---|
| **cost-aware-fd** | **0.0898** | **-0.0072** | -0.0167 | 0.7352 | **99.94%** |
| fixed-0.85 | 0.0830 | -0.0081 | -0.0105 | 0.6746 | 82.57% |
| fixed-0.6 | 0.0873 | -0.0098 | -0.0238 | 0.7220 | 95.97% |
| cost-aware (precision) | 0.0876 | -0.0122 | -0.0269 | 0.7274 | 97.51% |
| always | 0.0887 | -0.0150 | -0.0264 | 0.7363 | 100% |
| fixed-0.3 | 0.0887 | -0.0150 | -0.0264 | 0.7363 | 100% |

`always` and `fixed-0.3` are bitwise identical everywhere because no prediction has confidence below 0.5.

**Eviction inflation** (evictions with prefetching / without): clustered@6: always 1.933, fixed-0.6 1.456, fixed-0.85 1.215, cost-aware 1.448, fd 1.279;
clustered@9: 1.785, 1.383, 1.164, 1.396, 1.236; bfs-3@20: 1.149, 1.036, 1.013, 1.039, 1.029.

**Precision vs benefit**: correlation +0.730 over all runs, so precision is a decent global signal. But four clustered@9 runs have precision 0.81-0.86 with negative benefit
(fixed-0.6 0.822/-0.0093/1.383; fixed-0.85 0.863/-0.0019/1.164; cost-aware 0.812/-0.0122/1.396; fd 0.845/-0.0048/1.236). Within that configuration eviction inflation tracks harm; precision does not.

**Fault-delta controller behaviour** (epoch 2000, dead band 0.002, step 0.05, thresholds 0.15-0.95):

| Config | final thr | epochs winning | epochs losing | verdict |
|---|---|---|---|---|
| leslie3d@21 | 0.15 | 100 | 0 | went aggressive |
| bwaves@30 | 0.15 | 100 | 0 | went aggressive |
| clustered@6 | 0.95 | 1 | 24 | shut down |
| clustered@9 | 0.95 | 0 | 25 | shut down |
| bfs-3@20 | 0.95 | 5 | 29 | shut down |
| bfs-3@58 | 0.85 | 16 | 17 | **oscillates** (delta rate -0.00038, inside the dead band) |
| pr-3@15 / @40 | 0.45 / 0.40 | 2 / 2 | 0 / 0 | mildly aggressive |

## 4. Head-to-head against traditional algorithms (cell 18) [real]

Page faults; `+pred` = Markov hybrid with the **oracle** fault-delta decider on that eviction policy.

| Config | FIFO | Clock | LRU | FIFO+pred | Clock+pred | LRU+pred | OPT |
|---|---|---|---|---|---|---|---|
| leslie3d@21 | 15,608 | 15,126 | 15,313 | 9,992 | 9,913 | 9,956 | 5,881 |
| bwaves@30 | 3,380 | 3,214 | 3,140 | 2,075 | 1,993 | 1,969 | 3,112 |
| clustered@9 | 5,729 | 5,676 | 5,803 | 5,715 | 5,717 | 5,831 | 2,637 |
| bfs-3@20 | 7,614 | 7,390 | 7,394 | 7,658 | 7,417 | 7,415 | 6,349 |
| pr-3@15 | 5,945 | 5,933 | 5,905 | 5,907 | 5,901 | 5,875 | 5,591 |

**Reduction of LRU+pred vs the best traditional algorithm**: leslie3d **34.2%** (best traditional is Clock), bwaves **37.3%** (LRU), clustered -2.7%, bfs-3 -0.3%, pr-3 +0.5%.
(34.5% = Clock+pred vs Clock on leslie3d. Older documents quote "34.5-37.3%", mixing two definitions: see REVIEW_PREP.)

**Gain from adding prediction, per eviction policy**: leslie3d FIFO +36.0%, LRU +35.0%, Clock +34.5%; bwaves +38.6%, +37.3%, +38.0%;
clustered +0.2%, -0.5%, -0.7%; bfs-3 -0.6%, -0.3%, -0.4%; pr-3 +0.6%, +0.5%, +0.5%. The benefit is independent of the eviction policy.

bwaves LRU+pred (1,969) **beats OPT (3,112)**: OPT is optimal only among *eviction* policies and still pays every first-touch fault; prefetching avoids those.
On leslie3d prediction closes 57% of the LRU->OPT gap.

## 5. Ghost-list estimate (cell 19 / script 07) [real]

Deployable fault-delta feedback, no shadow engine, vs the oracle:

| | mean gain | worst case | best case |
|---|---|---|---|
| ghost estimate | **0.0902** | -0.0072 | 0.3729 |
| oracle (shadow) | 0.0898 | -0.0072 | 0.3729 |

Identical on four of eight configurations; on clustered@9 the ghost version is better (+0.0009 vs -0.0048). Sanity check vs cell 17: MATCH.
**Per-epoch correlation, estimated vs true delta** (the honest caveat): bwaves@30 0.999, clustered@9 0.689, leslie3d@21 0.662, bfs-3@58 0.596, bfs-3@20 0.516,
clustered@6 0.238, **pr-3@15 -0.975, pr-3@40 -0.976**; pooled 0.427.
Reading: accurate where the effect is large, noisy where it is tiny, never the wrong decision. The pr-3 anti-correlation is probably the refault window (2,000) blaming prefetches for evictions LRU would have made anyway. Untested hypothesis: try 250/500/1,000.

## 6. Phase 8 baselines [synthetic and sandbox only, NOT real traces]

Reproduce with `python scripts/09_phase8_baselines.py`. **No real-trace result exists yet.**

- **Scan test** (12 reused pages + one-time scan, 16 frames) [synthetic]: FIFO 15,261, Clock 14,892, LRU 14,520, **LRU-2 9,984, ARC 9,986**, OPT 9,980.
- **Readahead** [synthetic], 64 frames: sequential LRU 50,000 -> 11 faults; strided unchanged (50,000); random unchanged (49,671 -> 49,670). `cyclic`@8: 50,000 -> 5,001.
- **LRU-2 is slow to adapt when the working set jumps** [synthetic]: clustered@18 frames LRU 1,000 vs LRU-2 1,827. A known LRU-K weakness, not a bug.
- **Readahead can hurt** [synthetic]: clustered@9 LRU 5,803 -> 7,645, because random hops inside a cluster occasionally look like a stream.
- **clustered@9 full row** [synthetic, pinned in tests]: FIFO 5,729, Clock 5,676, LRU 5,803, LRU-2 6,512, ARC 6,207, RA+LRU 7,645, RA+ARC 8,025, Markov+LRU 5,798, Markov+ARC 6,029, OPT 2,637.
- **phase_changing@100** [sandbox]: LRU 39,796, Markov+LRU 32,790, RA+LRU 19,777. On sequential-heavy traces readahead can beat the (cautious) ghost-decider Markov system. Whether that happens on bwaves/leslie3d is the open question.
- Eviction-only sweep [sandbox, synthetic]: ARC and LRU-2 change fault counts by <1% on `random`/`phase_changing`, and are worse than LRU on `clustered`.

## 7. Phase 6 smoke test [sandbox]

`phase_changing` (synthetic): ghost decider +17.6% vs always-prefetch +50.3%, so the ghost/controller is too cautious on a workload with long predictable stretches (target of Phase 11 tuning).
The Phase 6 harness (`scripts/08_evaluation_harness.py`: segments -s0/-s1/-s2, memory sweep, 5 seeds, figures) has **not** been run on real traces.

## 8. The headline claims, and what supports them

1. Prefetching cuts faults 34.2-37.3% vs the best traditional algorithm on regular workloads (leslie3d, bwaves) [real, one segment each, no error bars].
2. The gain is independent of the eviction policy (FIFO/LRU/Clock all +34.5% to +38.6%) [real].
3. Confidence cannot gate prefetching (no prediction below 0.5) [real; but the 0.5 floor is structural for 2 successors, see DECISIONS section 2].
4. Precision is a misleading feedback signal under eviction pressure [real, four clustered@9 runs].
5. Fault-delta feedback halves worst-case harm (-0.0072 vs -0.0150) and keeps 99.94% of the benefit [real, 8 configs].
6. A ghost list reproduces it without a shadow engine [real; correlation caveat above].
7. Prefetching can beat OPT by avoiding compulsory faults [real, bwaves].

**Not yet supported**: robustness across trace segments, behaviour across memory sizes, anything about ARC/LRU-2/readahead on real traces, any execution-time claim (we count faults, not time).

## 9. Phase 8 on the real traces, and the improvement experiments [real traces, run locally 2026-10-19]

Traces re-downloaded with script 02 (now `urllib`); working sets and FIFO/Clock/LRU/OPT match section 2 exactly. Faults, ~1% of working set, one segment (`-s0`), no error bars.
Reproduce: `python scripts/09_phase8_baselines.py`, `python scripts/10_improvements.py` (-> `results/improvements.csv`).

**Phase 8 (LRU / ARC / LRU-2 / readahead+LRU / Markov+LRU(ghost) / OPT)**: leslie3d@21 15,313 / 16,553 / 26,188 / 11,624 / 9,956 / 5,881; bwaves@30 3,140 / 3,147 / 5,460 / **204** / 1,969 / 3,112;
bfs-3@20 7,394 / 7,368 / 7,969 / **2,924** / 7,427 / 6,349; pr-3@15 5,905 / 5,825 / 5,822 / **2,905** / 5,872 / 5,591.
Answers: ARC/LRU-2 do not close the LRU-OPT gap (ARC is worse than LRU on leslie3d; LRU-2 is much worse on leslie3d/bwaves, cause not yet diagnosed);
Markov still adds 35.6% / 34.3% on ARC for leslie3d / bwaves, nothing on graphs; **readahead beats Markov on bwaves, bfs-3, pr-3** and loses only on leslie3d (and clustered).

**Improvements** (every choice made on the first 25% of each trace, then evaluated once on the full trace; LRU underneath; `Comb` = readahead + Markov, always-prefetch):

| Trace | LRU | RA | Markov+ghost | Mk+delta+ghost | Comb | Comb+delta | Comb+delta+top2 | OPT |
|---|---|---|---|---|---|---|---|---|
| leslie3d@21 | 15,313 | 11,624 | 9,956 | 9,436 (top-2: 8,853) | 8,308 | 8,131 | **5,287** | 5,881 |
| bwaves@30 | 3,140 | 204 | 1,969 | 810 (766) | 205 | 205 | 234 | 3,112 |
| bfs-3@20 | 7,394 | 2,924 | 7,427 | 6,703 (6,706) | 3,122 | **2,570** | 3,084 | 6,349 |
| pr-3@15 | 5,905 | 2,905 | 5,872 | 5,790 (5,768) | 2,874 | 2,769 | **2,675** | 5,591 |
| clustered@9 [synthetic] | 5,803 | 7,645 | 5,798 | 5,703 (5,861) | 7,095 | 7,053 | 6,905 | 2,637 |

Adopted: readahead+Markov combination (`StreamMarkovPredictor`), delta-Markov tier (`DeltaHybridPredictor`), top-2 (`TopN`). Top-2 is **mixed**: big win on leslie3d, loss on bfs-3 (+20%) and bwaves (+14%); report per trace.
**Caveat (important): fault counts hide I/O.** Pages read from disk (faults + prefetches) on leslie3d: LRU 15,313; Comb+delta+top2 28,114; bfs-3: LRU 7,394, Comb+delta 13,638. Precision 0.29-0.47 on those. Fewer stalls, more I/O; a time model (Phase 12) is required before claiming "faster".
`clustered` is still worse than LRU with every readahead variant (random hops look like streams).

**Negative results (warm-up data, nothing adopted):**
- Stale-prefetch eviction (`PredictiveLRU`, grace 8-1000 accesses): never better than LRU; larger grace is much worse (protecting prefetches starves real pages).
- Predictor protection (never evict the top-1/2 predicted page): no change at all; predicted pages are almost always resident and recent, so LRU already keeps them.
- Minimum stream length 3-6 for readahead: fixes clustered only by losing the graph/leslie3d gains; kept at 2.
- Smoothed confidence (count/(total+1,2,4)): slightly worse; not adopted.
- Ghost refault window 250-2000: no effect on faults (graph warm-ups are uninformative: their first 25% is a sequential initialisation); left at 2000.
Also: the confidence floor of 0.5 is structural (top-1 of at most 2 successors), not an empirical property of the workloads.

## 10. Robustness (script 11, `results/robustness.csv`) [real traces, no retuning, 13 traces x {1%, 5%} of working set = 26 points, LRU underneath]

Mean `vs_LRU`: readahead +0.574, Markov hybrid + ghost (original) +0.101, **readahead + delta-Markov (Comb_d) +0.635**, + top-2 +0.653.
Comb_d has fewer faults than LRU in 25/26 points, than the original Markov in 24/26, than readahead alone in 19/26 (6 ties, 1 loss: bwaves-s0@1%, 205 vs 204).
Top-2 beats top-1 in 15/26, ties 6, loses 5 (so it is not a free win). With top-2, faults are below OPT in 21/26 points (prefetching removes compulsory faults, which OPT cannot).
Original Markov reproduces the Colab numbers on new data too (bwaves-s0 0.373, leslie3d-s0 0.350).
Segments: bwaves-s0/s1 +0.93 (readahead-dominated), leslie3d-s0/s1/s2 +0.42..0.47 (top-2: 0.59-0.66) at 1%, graphs bfs +0.80..0.89, pr +0.55..0.65, mcf +0.11..0.13, milc +0.88, libquantum +0.98.
**Weak/failing cases (report them):** bwaves-s2 (tiny working set, 8-13 frames): Comb_d -0.063 and top-2 -0.154 at 1%, +0.09/+0.04 at 5%; ARC is worse than LRU on leslie3d at 1% (-0.04..-0.08).
milc and libquantum have LRU = OPT = working set (pure streaming, every fault is compulsory), so the gain is purely readahead removing first-touch faults; Markov contributes nothing there.
Caveat unchanged: fault counts only (see section 9 on I/O), LLC traces approximate paging, one run per configuration (traces are deterministic).

## 11. Break-even prefetch cost (script 12, `results/breakeven.csv`) [real traces, 1% of working set, LRU underneath]

`c* = (LRU faults - faults) / prefetches`: the largest cost of one prefetch, as a fraction of one fault's stall, at which the system still beats LRU. c = 0 is fully hidden I/O; c = 1 is as costly as a fault.

| Trace | Readahead | Markov+ghost (original) | Comb_d | Comb_d+top2 |
|---|---|---|---|---|
| bwaves-s0 | 0.86 | 0.88 | 0.76 | 0.59 |
| leslie3d-s0 / s1 | 0.30 / 0.27 | **0.92 / 0.89** | 0.42 / 0.37 | 0.44 / 0.31 |
| milc / libquantum | 0.97 / 0.95 | 0.99 / n/a (no prefetches) | 0.94 / 0.95 | 0.93 / 0.95 |
| mcf | 0.37 | 0.03 | 0.47 | 0.35 |
| bfs-3 / bfs-10 | 0.78 / 0.79 | -0.05 / 0.14 | 0.59 / 0.65 | 0.48 / 0.57 |
| pr-3 / pr-10 | 0.73 / 0.72 | 0.10 / 0.21 | 0.61 / 0.63 | 0.56 / 0.57 |
| bwaves-s2 | -0.66 | 0.04 | -0.06 | -0.07 |

Reading: no system wins at every cost. Every c* is below 1, so if a prefetch stalls as much as a fault, none beats LRU. The original Markov+ghost system is the most I/O-efficient where it helps (leslie3d c* ~0.9);
the combined predictor saves far more faults but needs prefetches to cost less than ~0.4 of a fault there. Whether that is true is a hardware question (SSD vs HDD, overlap with compute): this model assumes no latencies,
and does not include the CPU cost of predicting. A real time model needs measured latencies (Phase 12).

## 12. Gating readahead with the ghost decider (Phase D) [real traces + synthetic clustered]

Readahead candidates now carry confidence `run/(run+k)` (`make_readahead(f, conf_k=k)`; default None = confidence 1.0, unchanged), so the ghost decider's threshold can stop short streams.
k chosen on warm-up only from {1, 2, 4, 8}: k=1 is the only value that keeps the graph/leslie3d gains (k >= 2 removes them). Column `Comb_d_gated` in `results/robustness.csv`.

Full traces (26 points): mean `vs_LRU` +0.632 (ungated +0.635); worst case **-0.032** (ungated -0.063, bwaves-s2@1%); 23/26 points within 0.003 of ungated; **cost: mcf** +0.107 -> +0.058 (1%), +0.113 -> +0.071 (5%).
Synthetic clustered: @6 LRU 20,344, ungated 21,089, gated **20,498** (most of the harm removed); @9 LRU 5,803, ungated 7,053, gated **7,053 (no change)**.

**Diagnosis of clustered@9** (why gating fails there): the ghost estimate says prefetching wins in 24/25 epochs (sum of estimated delta +988) while the true delta is **-1,250**; the threshold falls to 0.15.
Per-epoch correlation is high (0.81) but the estimate is biased upwards: it counts every *used* prefetch as a saved fault, and inside a 10-page cluster with 9 frames a used prefetch often displaced a page needed even sooner.
This is the same blind spot that made precision fail (section 3), now in the ghost estimate. A fix would need to charge indirect harm; untested.
Rerun check: all earlier columns of `robustness.csv` reproduced bit-for-bit.

## 13. Phase 14: shadow-directory feedback [real traces + synthetic clustered]

**Diagnosis.** Exact feedback alone did not fix clustered for the Markov-only system (-0.0117 with the true per-epoch delta vs +0.0009 ghost): the controller oscillates (10 winning / 14 losing epochs) and cannot switch off,
because 23.5% of predictions have confidence exactly 1.0 (contexts seen once) and pass the 0.95 ceiling. An "off" state with exponential-backoff probing (`allow_off=True`) was added and **never triggered** on any configuration (non-result; default off).
For the combined readahead system the estimate *was* the problem: the ghost list said "winning" on clustered@9 while losing (RESULTS 12).

**Mechanism.** `CostAwareDirectory` + `run_prefetch_sim_dir`: a no-prefetch copy of the eviction policy holding page IDs only (8 B per tracked frame, ~0.2% of RAM with 4 KB pages), optionally hash-sampled (SHARDS, Waldspurger et al., FAST 2015).
Estimated delta per epoch = (directory faults - real faults on sampled pages) / sample. With sample = 1 it equals the true per-epoch delta exactly (pinned by a test). No tuned parameter.
Honest framing: this is the old "shadow engine" minus the page data; the earlier claim that a kernel cannot run one was too strong for an ID-only directory.

**RA + delta-Markov (gated), LRU underneath, vs_LRU** (full traces; warm-up shows the same direction):

| Config | ghost | directory |
|---|---|---|
| clustered@9 [synthetic] | -0.2154 | **-0.0250** |
| clustered@6 [synthetic] | -0.0076 | -0.0111 |
| mcf@31 | +0.0577 | **+0.1017** |
| bwaves-s2@8 | -0.0317 | **-0.0143** |
| leslie3d@21 / bwaves@30 / bfs-3@20 / pr-3@15 | +0.4686 / +0.9344 / +0.6520 / +0.5311 | identical |

**26-point robustness (script 11, column `Comb_d_dir`)**: mean **+0.636**, worst **-0.014** (best worst case of every system; ghost-gated -0.032, ungated -0.063, top-2 -0.154), worse than LRU in 1/26 (bwaves-s2@1%).
Fewer faults than ghost-gated in 3/26, equal in 23, never more. It recovers the mcf gain that gating with the ghost list had cost (1%: 183,052 -> 174,506; ungated 173,439).
Against readahead alone: fewer faults in 19/26; the 7 losses are 1-2 faults each (<= 0.06% of LRU), from the lower confidence at the start of each stream. Rerun reproduced all earlier columns exactly.

**Sampling (RAM = 5% of working set, 10 real traces)**: at sample 1/2, 1/4 and 1/8 the fault counts are identical to the full directory on 9/10 traces; bwaves-s2 (13 frames -> 2 directory entries at 1/8) changes slightly (+0.0842 -> +0.0802).
Per-epoch correlation with the true delta falls with sampling (e.g. libquantum 1.00 / 0.71 / 0.43 / 0.23, bfs-3 1.00 / 1.00 / 0.99 / 0.99): decisions survive because the effects here are large. Not tested at kernel-scale memory.

## 14. Phase 15: time model (script 14, `results/time_model.csv`) [real traces, model estimates, NOT measurements]

I/O timeline replay (`src/paging/timing.py`): compute `t_cpu` per traced access (unknown for LLC traces, swept 0.1 / 1 / 10 us), prediction cost 0.2 us per access (or 0),
4 KB read 80 us on an NVMe SSD with 8 parallel channels, 5 ms on an HDD with 1 channel; demand faults block and can queue behind prefetch reads (no demand priority: pessimistic for prefetching);
a prefetched page used before its read completes stalls for the rest (late prefetch). 13 real traces, RAM = 1% of working set, LRU underneath. Speedup = time(LRU) / time(system).

| Device, compute per access | RA + delta-Markov (directory) | Readahead | Markov (original, ghost) |
|---|---|---|---|
| NVMe, 0.1 us | **1.72x** (0.79-3.96), slower on 1/13 | 1.70x | 0.95x, slower on 10/13 |
| NVMe, 1 us | **1.47x** (0.83-2.68), slower on 1/13 | 1.45x | 0.97x, slower on 10/13 |
| NVMe, 10 us | **1.13x**, slower on 1/13 | 1.11x | 1.00x |
| HDD, any | 0.72-0.76x, slower on 12-13/13 | 0.80-0.84x | 0.93-0.94x |

(Geometric means over traces, prediction cost 0.2 us included; without it NVMe 1 us gives 1.59x / 1.56x / 1.02x.) The one NVMe loss is bwaves-s2 (8 frames).
**Readings.** (1) Fault reductions become time savings only when the device can overlap reads (NVMe); on a single-channel HDD every prefetch costs a full access and all prefetching systems are slower than LRU, consistent with every break-even c* < 1 (section 11).
(2) The original Markov system cuts faults 35-37% on leslie3d/bwaves but is not faster: it prefetches one access ahead, so with 80 us reads its prefetches arrive late (late-stall share 16-22% of run time on bwaves/leslie3d). Readahead fetches a window ahead and arrives in time (late share ~3% on bwaves). Prefetch distance matters as much as accuracy.
(3) With prediction cost charged, the Markov system is slower than LRU on 10/13 traces where it barely prefetches (bfs, pr): 0.2 us per access is 20% of a 1 us compute gap. A real OS sees page faults, not every access, so this cost is an upper bound here.
Caveats: stated latencies, not measurements; compute per access unknown (hence the sweep); no demand-read priority, no write-back, no I/O merging; LLC traces.

## 15. Phase 16: controller tuned and frozen (script 15, `results/tuning_controller.csv`) [warm-up portion only]

Grid: epoch {500, 1000, 2000, 4000} x step {0.05, 0.1, 0.2} x dead band {0.001, 0.002, 0.005}, start threshold 0.5; system RA + delta-Markov (conf_k=1) + directory feedback, LRU, RAM 1% of working set;
first 25% of 13 real traces + clustered. Rule fixed before the run: keep settings whose worst case is within 0.002 of the default's, take the highest mean, keep the default unless beaten by > 0.002.
**Chosen: epoch 1000, step 0.1, dead band 0.002** (`config.CONTROLLER_FROZEN`, used by `paging.final`): mean +0.6647 vs +0.6613, worst -0.0382 vs -0.0800. Dead band 0.001 and 0.002 tie exactly.
Honest reading: the change comes **only** from clustered (synthetic; -0.0800 -> -0.0382) and bwaves-s2 (-0.0056 -> 0.0000); the other 12 traces are identical, because the controller only matters where prefetching can hurt.
Faster reaction (shorter epoch, bigger step) is what helps; consistent with the Phase 14 diagnosis that the controller was too slow to back off. Class defaults are unchanged, so every earlier number reproduces.

## 16. Phase 17: final system on 22 real traces [real traces; the final system is frozen in `paging.final`]

Added: soplex, omnetpp, astar, GemsFDTD, lbm, sphinx3 (SPEC 2006) and bc-3, cc-5, sssp-3 (GAP), now 22 real traces x {1%, 5%} = 44 points. Earlier 26 rows reproduced exactly.
**Budget cap found and removed** (scripts 16, warm-up only, same rule): the 35% prefetch budget from Stage 5 stopped the final system at exactly 70,000 prefetches on omnetpp while it was winning 200/200 epochs.
No cap was best and never worse on any warm-up trace (mean +0.5897 vs +0.5658); now `CONTROLLER_FROZEN['budget_ratio'] = 1.01`.

**Faults, 44 points, `vs_LRU`:** final system mean **+0.550**, median +0.593, worst -0.033 (bwaves-s2@1%), worse than LRU in 1/44; readahead +0.507 (worst -0.193); original Markov +0.077.
Final vs readahead: fewer faults in 34/44, equal 8, more 2 (sphinx3@1%: 11,263 vs 8,997, a 16.9%-of-LRU loss because the decider holds back useful streams after 41 losing epochs; bwaves-s0@1%: 1 fault). Below OPT in 30/44.
New traces: omnetpp +0.97 (readahead +0.97), GemsFDTD +0.74/+0.84, sssp +0.58, bc +0.41/+0.44, soplex +0.21/+0.34, cc +0.14, astar +0.07, lbm 0 (nothing is predictable; LRU ~= OPT).

**Time (script 14, 22 traces, prediction cost 0.2 us; geometric mean speedup vs LRU):**

| Compute per access | NVMe: final | NVMe: readahead | NVMe: original Markov | HDD: final | HDD: readahead |
|---|---|---|---|---|---|
| 0.1 us | 1.59x (slower on 3/22) | 1.60x (2/22) | 0.96x (19/22) | 0.70x | 0.80x |
| 1 us | **1.43x** (3/22) | **1.44x** (2/22) | 0.97x (18/22) | 0.70x | 0.80x |
| 10 us | 1.19x (3/22) | 1.19x (2/22) | 1.00x (17/22) | 0.73x | 0.83x |

**The honest headline:** the final system has the most fault reduction of every system, but in *estimated time* it ties Linux-style readahead (1.43x vs 1.44x on NVMe). It is faster on leslie3d (1.24x vs 1.14x) and slower on sphinx3 (0.93x vs 1.26x) and cc-5;
the extra faults it removes cost extra prefetch reads. On a single-channel HDD every prefetcher is slower than LRU, the final system most. lbm is 0.92x for everyone (prediction cost, nothing to gain).
Break-even prefetch cost (script 12, 22 traces): final system c* 0.00-0.95, below readahead's on most graph traces, above it on leslie3d (0.42 vs 0.30) and mcf (0.47 vs 0.37).

## 17. Component comparison (script 17, `results/component_*`) [22 real traces at 1% + synthetic clustered, LRU underneath]

One component changed at a time. **Predictors**, all through the final decider (directory + frozen controller); "useful" = share of accesses whose top-1 guess is exactly the next page:

| Predictor | useful | faults vs LRU (mean / worst) | SSD speed-up | HDD |
|---|---|---|---|---|
| stride | 0.027 | +5.8% / 0.0% | 1.01x | 1.00x |
| markov-o1 | 0.501 | +4.9% / -0.8% | 1.01x | 0.94x |
| markov-o2 | 0.533 | +2.4% / -1.1% | 1.00x | 0.97x |
| hybrid | 0.573 | +9.4% / -0.3% | 1.01x | 0.92x |
| delta-hybrid | **0.589** | +20.9% / 0.0% | 1.02x | 0.86x |
| readahead | 0.026 | +46.5% / -2.5% | **1.51x** | 0.83x |
| final (readahead + delta-hybrid) | 0.562 | **+51.2%** / -3.3% | 1.49x | 0.70x |

Accuracy and fault reduction rank predictors in nearly opposite orders: readahead is exactly right only 2.6% of the time but avoids the most faults among single predictors, because it fetches pages never seen before.
In time the combination is not faster than readahead alone on the SSD model (1.49x vs 1.51x).

**Deciders, fed the final predictor** (mean over real traces / worst real trace / clustered@9-equivalent at 1% / SSD speed-up):
always and fixed-0.3 +51.2% / -6.3% / **-21.5%** / 1.49x; fixed-0.6 +28.8% / 0.0% / -5.5% / 1.21x; fixed-0.85 +19.8% / 0.0% / -0.2% / 1.07x; precision +26.8% / 0.0% / -5.2% / 1.16x;
fault-delta oracle +49.1% / **0.0%** / -1.7% / 1.39x; ghost +47.9% / -3.2% / -21.5% / 1.39x; directory (default controller) +49.0% / -1.4% / -2.5% / 1.39x; **directory + frozen controller (final) +51.2% / -3.3% / -2.4% / 1.49x**.
Reading: the final decider keeps all of `always`'s benefit while cutting its clustered harm from -21.5% to -2.4%; fixed thresholds are safe but lose half or more of the benefit; the (non-deployable) oracle is the only one never worse than LRU.
Fed the original Markov hybrid, all deciders give +8.3% to +9.4%: that predictor rarely has anything worth prefetching, so the decision layer barely matters.
