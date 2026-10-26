# Final review: likely viva questions and answers

Every number below comes from the project's result files (`results/*.csv`, `docs/RESULTS.md`).

## Problem and concept

**1. What is a page fault?**
A request for a page that is not in RAM. The program waits while the page is read from disk.

**2. What does your project do, in one sentence?**
It simulates operating-system paging and tests whether predicting the next pages and loading them early (prefetching) reduces page faults, using a decision engine that prefetches only when it measurably helps.

**3. Why is prefetching risky?**
When RAM is full, every prefetched page evicts another page. A wrong guess can evict a page that is needed soon and create a fault that would not otherwise have happened.

**4. How can your system beat OPT, the optimal policy?**
OPT is optimal only at choosing what to evict. It still pays for the first access to every page. Prefetching can avoid those first-access faults. The final system has fewer faults than OPT in 30 of 44 configurations.

**5. Why use a simulator instead of a real operating system?**
Every policy sees exactly the same input, results are reproducible, and OPT (which needs the future) can be computed. Modifying a kernel would also take far more engineering than a course project allows.

## Architecture and logic

**6. What are the main components?**
- Memory engine: RAM with an LRU eviction policy.
- Predictor: readahead plus delta-Markov.
- Decision engine: a confidence threshold.
- Feedback controller.
- Shadow directory.
- Time model.

**7. How does the decision engine decide?**
- With a free frame, it prefetches if confidence is at least 0.15, because a wrong prefetch then evicts nothing.
- With RAM full, it prefetches only if confidence is at least an adaptive threshold between 0.15 and 0.95.

**8. How does the threshold adapt?**
Every 1,000 requests it computes the delta rate: (faults without prefetching − faults with prefetching) / 1,000.
- Above +0.002: the threshold drops by 0.1 (prefetch more).
- Below −0.002: it rises by 0.1 (prefetch less).
- In between: no change.

**9. What is the shadow directory?**
A second copy of the eviction policy that runs without prefetching and stores page IDs only, never page data. It costs about 0.2% of RAM and gives the exact fault delta. Tracking a hash sample of 1 page in 8 gave identical fault counts on 9 of 10 traces tested.

**10. Why not use prefetch precision as the feedback?**
A used prefetch can still have evicted a page that was needed sooner. On the clustered test, precision was 81–86% while faults went up.

**11. Why not use the predictor's confidence?**
The predictor keeps 2 successors per context, so the top one always holds at least half the counts. Confidence is therefore never below 0.5, and a 0.3 threshold behaves exactly like always prefetching.

**12. What is delta-Markov?**
A Markov table over the gaps between pages instead of the pages themselves. A repeating gap pattern such as +1, +3 lets it predict pages it has never seen.

**13. How does readahead work?**
Two consecutive pages start a stream. The fetched window doubles each time the program reaches its middle (for example 4 → 8 → 16 pages), up to a quarter of RAM, and it resets when the sequence breaks.

**14. Why not deep learning?**
The predictor runs on every memory request, so it must be O(1) and tiny; ours uses 256-entry tables. A learned predictor is a fair comparison for future work.

## Implementation and testing

**15. What technologies did you use?**
- Python 3.14 (compatible with 3.9 and later), NumPy, pandas, matplotlib and pytest.
- Python standard-library modules for decompression, download and data structures.
- Git and GitHub for version control.
- Google Colab for the early stages.

**16. How do you know the simulator is correct?**
- 24 automated tests pass. They check invariants (nothing beats OPT; more memory never increases faults for LRU or OPT), equivalences (each new feature switched off reproduces the old behaviour exactly) and pinned numbers.
- Independent implementations of LRU, FIFO and OPT gave 0 mismatches on 40 traces.
- A rerun reproduced all 48 rows of the original Colab results exactly.

**17. How did you avoid tuning on test data?**
Every setting was chosen on the first 25% of each trace only, with the selection rule written down before each sweep, and then frozen.

**18. What data did you use?**
- 22 real traces from the ML-DPC LoadTraces (SPEC CPU 2006 and GAP graph benchmarks), 200,000 requests each, with addresses mapped to 4 KB pages.
- 6 synthetic traces used as correctness tests.

## Results

**19. What is the main result?**
The final system removes 55.0% of LRU's page faults on average over 44 configurations. That compares with 50.7% for Linux readahead and 7.7% for Markov prefetching alone. It is worse than LRU in only 1 of 44 configurations (−3.3%), against −19.3% for readahead's worst case.

**20. Is it faster?**
In the time model, about 1.43× faster than LRU on an NVMe SSD, level with readahead (1.44×). On a single-channel hard disk every prefetcher is slower than LRU. These are model estimates, not hardware measurements.

**21. Why is Markov alone not faster although it removes 35–37% of faults on leslie3d and bwaves?**
It prefetches only one request ahead, so its pages often arrive after they are needed (late prefetches).

**22. Why does prefetching lose on a hard disk?**
A hard disk does one read at a time, so each prefetch costs as much as the fault it tries to save.

**23. Did ARC or LRU-2 help?**
No. ARC averaged −0.5% against LRU on our traces, and LRU-2 was much worse on leslie3d and bwaves (confirmed with an independent implementation).

## Novelty, limitations and future work

**24. What is new in your project?**
The building blocks (Markov prefetching, readahead, fault-rate feedback, ghost lists) are known. Our contributions are:
- showing which feedback signal works, and why confidence and precision fail;
- an exact, cheap way to measure it (the shadow directory);
- the readahead and delta-Markov combination;
- a time-aware evaluation.

**25. Isn't the shadow directory just the impossible oracle again?**
It is the same idea, but it stores page IDs only, with no page data, and it can be sampled down to 1/8 of its size.

**26. What are the main limitations?**
- The traces are cache-level traces treated as pages, an approximation of OS paging.
- The time results come from a model.
- The predictor sees every request, which a kernel does not.
- Each configuration was run once (the traces are deterministic).

**27. What would you do next?**
Evaluate on OS page-fault traces or inside a kernel, add demand-read priority and prefetch-distance control, and compare against a learned predictor under the same time model.

**28. If readahead ties you in time, why not just use readahead?**
Our system removes more faults (fewer than readahead in 34 of 44 configurations), helps much more on workloads like leslie3d (46.9% against 24.1% at 1% memory), and has a far better worst case.

**29. Is this publishable or patentable?**
A paper draft exists for a student or regional conference, framed as an evaluation study. We do not claim acceptance or patentability.
