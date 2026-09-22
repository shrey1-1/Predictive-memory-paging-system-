# Predictive Memory Paging System — System Architecture

Reference document for the workload-aware adaptive predictive paging project.

---

## The big picture: two phases, one pipeline

The system has an **offline phase** (get data, prepare it, tune parameters) and an **online phase** (the simulator running access-by-access, predicting and deciding in real time). Most confusion in these projects comes from mixing them up.

The Markov chain is unusual in that it lives in both — it can be warm-started offline and still keeps learning online.

---

## Stage 1: Choosing and obtaining the dataset

Two tiers of data, serving different purposes.

### Tier 1 — Synthetic traces (generated)

These are the **controlled experiments**. Because the ground truth pattern is known, they prove the components work: if the classifier can't identify a pure sequential trace generated in-house, it's broken.

Patterns to generate:

- Pure sequential (fixed stride)
- Strided (larger gaps)
- Cyclic loops of varying length
- Clustered / locality-shifting
- Uniform random
- **Phase-changing traces** that switch pattern midway — this category is what actually justifies the word "adaptive" in the project title

### Tier 2 — Real traces

This is what makes it research rather than a class exercise. Options, roughly in order of effort:

1. Pre-recorded benchmark traces from standard suites (SPEC CPU, PARSEC) — easiest, and what reviewers expect to see.
2. Self-generated using a binary instrumentation tool (Intel PIN, DynamoRIO, or Valgrind's Lackey) attached to a real program. More work, but gives full control and a stronger methodology section.

### Note on granularity

Raw traces record **byte addresses**. The simulator wants **page numbers**. Converting is just dropping the low-order bits (dividing by page size). This is a preprocessing step, and it dramatically shrinks the space of distinct values being modeled — which is exactly what's wanted.

---

## Stage 2: Data preparation and feeding

This layer turns raw traces into something uniform the engine consumes.

**Normalize everything into one format.** Every trace — synthetic or real — becomes the same simple thing: an ordered sequence of page IDs. The engine should never know or care where a trace came from. This is what allows swapping datasets freely later.

**Derive deltas alongside the raw sequence.** Store the gap between consecutive accesses too. The classifier and the stride predictor both work on deltas, while the Markov chain can work on either. Computing this once during preparation avoids recomputing it per access.

**Split the trace in time order — never shuffle.** This is critical and easy to get wrong. Memory access traces are sequences; shuffling destroys the very thing being modeled. Use the first portion (say 20–30%) as warm-up/training and the remainder for evaluation. The model sees the training portion, then gets measured only on the unseen part.

**Feed it as a stream, not a batch.** The engine should pull one access at a time, exactly as a real system would. Even with the whole file available, the simulator must never peek ahead — except for OPT, which is *allowed* to cheat because it's defined as the theoretical bound.

---

## Stage 3: "Training" the model

The word "training" is slightly misleading here.

**A Markov chain doesn't train like a neural network.** There's no loss function, no gradient descent, no epochs. "Training" means *counting observed transitions*. Walk the training portion and, for each consecutive pair, increment a tally. That's it. The model is literally a table of counts.

This has a big practical consequence: **the model can keep learning during evaluation.** Unlike a neural net that gets frozen after training, the Markov table can continue updating as the test portion streams past. This is desirable — it's what lets the system adapt to phase changes. Be clear in the writeup that the model learns online, because it's a methodological point a reviewer will ask about.

### What the offline phase actually does

1. **Warm-start** the table so the system doesn't spend the first thousand accesses knowing nothing (the "cold start" problem).
2. **Tune the hyperparameters** — window size for the classifier, table capacity, number of successors kept, confidence thresholds, decay rate. These are fixed offline by sweeping values on the training portion, then held constant during evaluation.

> Tuning thresholds on test data is cheating and invalidates the results.

If a CNN/LSTM classifier gets added later, that's the one component with conventional training — labeled windows from synthetic traces (where the true pattern is known), standard train/validation split, frozen weights at evaluation time.

---

## Stage 4: The runtime architecture

### Components

| Component | Responsibility |
|---|---|
| **Memory Engine** | Owns the page table and frame table, detects hits/faults, performs eviction. Policy-agnostic. |
| **Access Monitor** | Maintains the sliding window of recent pages and deltas. |
| **Classifier** | Labels the current pattern from the window. |
| **Predictor Bank** | Stride predictor, Markov predictor, and a null/no-op predictor; the classifier selects which is active. |
| **Decision Engine** | Takes (candidate page, confidence) plus system state, outputs prefetch yes/no. |
| **Prefetch Queue** | Holds pending speculative fetches, subject to a budget cap. |
| **Feedback Tracker** | Records whether prefetched pages were actually used before eviction; adjusts thresholds. |
| **Metrics Logger** | Accumulates everything to be reported. |

### The flow on one access

1. Page ID arrives → Memory Engine checks residency
2. Hit or fault is recorded (if a fault, the page is loaded, evicting if necessary)
3. Access Monitor appends this page to its window and updates the Markov table with the transition that just completed
4. Classifier reads the window and labels the pattern
5. The corresponding predictor produces candidate next-pages with confidences
6. Decision Engine consults current RAM pressure, eviction-candidate value, I/O load, and the current adaptive threshold, then approves or rejects each candidate
7. Approved candidates enter the Prefetch Queue and get loaded speculatively, tagged as prefetched-not-yet-used
8. Later, when a page is accessed, the Feedback Tracker checks whether it was a prefetch that paid off, and nudges the threshold accordingly

### One architectural rule worth enforcing

**The prediction/decision path must never be able to alter correctness — only performance.**

A completely broken predictor should produce a slow system, never a wrong one. Structurally, that means prefetching only ever *adds* pages to RAM and evicts via the normal policy; it never bypasses the fault handler.

This makes debugging vastly easier, because the entire predictive layer can be disabled and the simulator still runs correctly as plain LRU.

---

## Stage 5: The decision engine

Architecturally it's a **scoring function with a dynamic threshold**, not a fixed rule.

### Inputs

- The candidate page
- Its confidence
- Current free-frame count
- The value of whatever would be evicted
- Current I/O queue depth
- Remaining prefetch budget for this interval

### Computation

Rough expected benefit (confidence × fault-cost-saved) weighed against expected cost (miss probability × wasted I/O, plus eviction damage), compared against the current adaptive threshold.

### Design for ablation

Build it as a **pluggable component with the same interface as the baselines**, so these four configurations can be run:

1. **No-prefetch** (pure LRU)
2. **Always-prefetch** (naive, ignores cost)
3. **Fixed-threshold**
4. **Adaptive** (the proposed system)

Those four configurations *are* the results section — they demonstrate that cost-awareness specifically contributes something, rather than just claiming it does.

---

## Stage 6: Evaluation harness

Run every configuration across every trace, sweeping RAM size.

### Metrics to log

- Page fault rate
- Raw prediction accuracy
- Useful-prefetch rate
- Wasted I/O
- Prefetch-induced evictions
- Simulated execution time

Multiple runs with different random seeds for the stochastic traces, and **report variance** — a single number with no error bars is the most common weakness in student systems papers.

---

## Build order

This order keeps a running system at every step, so the contribution of each addition is always visible:

1. Memory Engine + LRU — working and verified
2. Evaluation harness and metrics
3. Always-prefetch (deliberately naive)
4. Markov predictor
5. Classifier
6. Decision engine
