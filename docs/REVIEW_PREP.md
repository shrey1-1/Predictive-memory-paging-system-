# Review and write-up preparation

## 0. Updated framing after Phases 9-17 (19 Oct 2026; supersedes parts of section 1)

The paper draft is `docs/paper_draft.md`. Headline is no longer "Markov prefetching cuts faults 35%":
- The final system (readahead + delta-Markov, shadow-directory feedback) removes 55.0% of LRU's faults on 22 real traces (readahead 50.7%, Markov alone 7.7%), worse than LRU in 1/44.
- In estimated time it **ties** Linux-style readahead on an NVMe model (1.43x vs 1.44x) and every prefetcher is slower than LRU on an HDD model. Say this first; it is the result a reviewer will check.
- Markov alone cuts faults 35-37% on leslie3d/bwaves but is not faster: late prefetches. Timeliness > accuracy.
- Feedback: precision and the ghost list share one blind spot; an ID-only shadow directory is exact and samplable. The earlier line "a kernel cannot run a shadow" is too strong for an ID-only directory.
- New likely questions: *Why not just use readahead?* (on time, it is as good; our gain is in faults and on leslie3d-like workloads). *Is the time model valid?* (stated latencies, swept compute, no demand priority: a model, not a measurement).

## 1. How to frame the contribution (keep this honest)

The components are not new and must not be presented as such:

- Markov-chain prefetching: Joseph & Grunwald, *Prefetching using Markov predictors*, ISCA 1997.
- Adapting behaviour to the measured fault rate: Page Fault Frequency (PFF), a classic OS technique.
- A 2026 ISMM paper (stride/delta Markov predictor with an address-Markov fallback for GPU unified-memory page prefetching) is close to our hybrid predictor.
- Prefetch-aware insertion/eviction: PACMan (Wu et al., MICRO 2011); Linux's inactive list.
- Ghost entries: ARC, and Linux `mm/workingset.c` shadow entries.

What *is* ours: an evaluation at OS page granularity of **which feedback signal a cost-aware prefetcher should use** (confidence cannot discriminate, precision misleads, measured fault delta works),
a **deployable ghost-list estimate** of that signal (~19 KB of predictive state in total), and (planned, Phase 9) **prediction-aware replacement**.
Frame it as an evaluation study plus a mechanism, aimed at an undergraduate research / regional IEEE conference. It would not clear a top systems venue as is.

## 2. Positioning against the ten reference papers

| Ref | Gap in that work | How we differ |
|---|---|---|
| [1] Workload-Aware Buffer Prefetching for Database Systems (2026) | database structures only | general OS paging + a cost-aware decision layer |
| [2] Learning-based Page Replacement for Efficient I/O (2025) | eviction only, no prefetching | adds prefetching; our result shows prediction helps on top of any eviction policy |
| [3] To Cross, or Not to Cross Pages for Prefetching? (2025) | spatial only | Markov captures non-spatial patterns |
| [4] Beyond VABlock: Aggressive Prefetching for Transformer Workloads (2025) | high overhead when workloads deviate | our `always` control reproduces that failure (up to 1.5% extra faults); the decider limits it |
| [5] Graph Analytics with Attention-Based Prefetcher (2024) | graph-specific, no cost-awareness | general-purpose and cost-aware, but we gain almost nothing on graph workloads |
| [6] ML-Based Runtime Prefetcher Selection (2024) | hardware L1/L2 | similar spirit to our hybrid chain, at OS page level |
| [7] Tyche: Prefetcher for Indirect Memory Accesses (2024) | no system-wide cost awareness | adds cost-awareness; Tyche handles pointer-chasing, we do not |
| [8] Prefetching Indexing Scheme for In-Memory Databases (2024) | bound to in-memory DBs | no application knowledge needed |
| [9] EXTMEM: Application-Aware Virtual Memory (2024) | requires app modification | transparent, works from the access stream |
| [10] RPG2: Profile-Guided Runtime Prefetch Generation (2024) | struggles with phase changes | learns online, adapts (shown on phase_changing) |

Where we are weaker: simpler predictor than attention/learned models; no pointer-chasing support; four real benchmarks; fault counts rather than time.
Do not claim "unique ML + adaptive + cost-aware + prefetching": the original comparison-matrix slide is easy to challenge. Say "integration and evaluation".

## 3. Likely reviewer questions (short answers)

- **What is novel?** The evaluation of the feedback signal, the deployable ghost-list estimate, and (Phase 9) predictor-informed eviction; not Markov prefetching.
- **Why not deep learning?** The predictor runs on every access, so its cost counts; ours is one lookup and kilobytes. A learned predictor is the right future comparison.
- **How is OPT beaten on bwaves?** OPT is optimal among eviction policies and still pays every first-touch fault; prefetching avoids those.
- **How do you know the simulator is correct?** Automated checks (nothing beats OPT; LRU/OPT monotonic in memory; sequential faults 100%; plenty of memory => faults = unique pages) and pinned reference numbers in `tests/`.
- **Why so little gain on graphs?** Measured: ~99.8% of predictions name an already-resident page, and LLC traces have already absorbed the easy locality.
- **Is "confidence cannot gate prefetching" a finding?** Partly an artefact: with 2 successors confidence is floored at 0.5 by construction. State it that way.
- **What does a negative `vs_LRU` mean?** Prefetching caused more faults than doing nothing: each prefetch into full RAM evicts a page; a wrong guess plus a needed evicted page is a created fault.
- **Why did precision fail?** On clustered@9 precision was 0.81-0.86 and faults still rose; a used prefetch can displace a page needed sooner.
- **Can it run in a kernel?** The oracle cannot; the ghost list can (about 4 KB plus the predictor tables), with the caveat that its per-epoch correlation with the true delta is uneven (pr-3 -0.975) though it never made a wrong decision.
- **How are hyperparameters chosen?** Predictor parameters by sweep on the warm-up portion, then frozen. Controller/ghost parameters are *not yet tuned* (Phase 11): say so.
- **Is 35% statistically meaningful?** Real traces are deterministic, so repeats are identical; robustness needs other segments and benchmarks (Phase 6/12). Cross-policy consistency (34.5-38.6%) is supporting evidence only.
- **Compared with Linux?** Readahead and ARC are now implemented (Phase 8); results on real traces pending.

Always volunteer the limitations first: gains concentrated in 2 of 5 real configurations; LLC traces approximate paging; counts not time; one run per configuration; small simulated memories (84-120 KB); untuned feedback parameters.

## 4. Documents that are now stale or wrong (fix before reuse)

| Document | What to change |
|---|---|
| Review deck `Predictive_Memory_Paging_Project_Review.pptx/.pdf` | slide 5: script 07 -> Completed (ghost-list validated); slides 8 and 13: replace "shadow run is an oracle a kernel cannot execute" with the ghost result and the correlation caveat; slide 7 chart mixes definitions (leslie3d "with prediction" 9,913 is Clock+pred; bwaves 1,969 and pr-3 5,875 are LRU+pred; LRU+pred on leslie3d is 9,956); the "34.5-37.3%" headline and the "8.5 KB / 1.66%" claim (order-2 table only; see DECISIONS section 3) |
| First deck `Predictive_Memory_Paging_Review.pptx` (blue) | superseded; also says hybrid useful 0.78 |
| `Predictive_Memory_Paging_Concepts_and_Modules.pdf` | hybrid useful **0.78 -> 0.81**; Part 9 and limitations: ghost list now validated, "future work" item done; "three-quarters of confidences >= 0.93" is not true for bwaves/clustered; 8.5 KB figure; 34.5% headline |
| `predictive_paging_guide_stages_1_5.{md,pdf}` and `docs/implementation_guide.*` | Appendix B quotes hybrid mean useful 0.783 (should be 0.8115); stages are 1-5 only (no ghost list, no Phase 8); 8.5 KB figure; limitation about the oracle |
| `COLAB_CONTINUATION_BRIEF.md`, `PROJECT_HANDOFF.md` | superseded by these docs |
| `README.md` (this repo) | already updated for scripts 06-09, tests and the corrected footprint |

## 5. Material the user may still supply

Screenshots of Phase 8 / Phase 6 outputs (ask before building any deck slide that needs them; never fabricate). The real-trace CSVs. Their `ghost_estimate_results.csv`, `head_to_head.csv`, `stage3_summary.json` etc. from Drive (`paging_sim/`).

## 6. Paper skeleton (Phase 13)

Introduction and motivation; background (paging, OPT, Markov prefetching, PFF); system design (predictor, decision engine, fault delta, ghost list); methodology (traces and LLC caveat, memory scaling, frozen configuration, ablation design);
results (head-to-head incl. ARC/LRU-2/readahead, decision-engine ablation, ghost validation, prediction-aware replacement, robustness); discussion (why precision fails, where prefetching helps); limitations; related work (section 1); conclusion.
