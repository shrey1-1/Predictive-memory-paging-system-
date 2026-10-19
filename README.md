# Predictive Memory Paging System

A trace-driven simulator for **workload-aware, cost-aware predictive paging**. It predicts future page accesses with a lightweight Markov chain, and uses a decision engine to choose which predictions are worth prefetching. The system is evaluated against FIFO, LRU, Clock and Belady's OPT on synthetic and real benchmark traces.

**Team:** G.K Shreyas Bharadwaj (24BCE2256) · Vaastav Bajanthri (24BCE2223) · P. Giphy Bershan (24BCE2201)

---

## Key results

| Result | Value |
|---|---|
| Fault reduction, `437.leslie3d` @ 21 frames | **35.0%** (15,313 → 9,956 faults) |
| Fault reduction, `410.bwaves` @ 30 frames | **37.3%** (3,140 → 1,969 faults) |
| Predictor memory (bounded ceiling) | order-2 table **8.5 KB** (1.66% of RAM at 128 frames); full hybrid ~15 KB (2.9%); ~19 KB with the ghost list |
| Readahead + delta-Markov, mean over 13 real traces x 2 memory sizes | **63.5%** fewer faults than LRU (worse than LRU in 1 of 26 points); fault counts only, see Limitations |
| Worst-case harm, always-prefetch → fault-delta decider | −1.50% → **−0.72%** (halved) |
| Benefit retained by fault-delta decider on leslie3d | **99.94%** |

**Main findings**

1. Prefetching cuts page faults by 35–37% on regular workloads under memory pressure.
2. Confidence thresholds alone do not separate good predictions from bad ones. No confidence is ever below 0.5, but that floor is structural (top-1 of at most 2 successors), not a property of the workloads.
3. Adapting the threshold based on prefetch **precision** makes things worse. Under eviction pressure, a prefetched page can be used and still cause net harm.
4. Adapting based on measured **fault delta** gives the best trade-off between safety and benefit.
6. Combining Linux-style readahead (new pages in a run) with a delta-Markov predictor (repeating patterns, including unseen pages) beats either alone in most configurations. It reads more pages from disk: it only pays off if a prefetch costs less than ~0.4-0.9 of a fault's stall (script 12).
5. A small **ghost list** (pages evicted by prefetches, checked on each fault) reproduces the fault-delta decisions without a shadow engine, making the mechanism deployable.

Full analysis is in [`docs/implementation_guide.pdf`](docs/implementation_guide.pdf).

---

## Repository structure

```
predictive-memory-paging/
├── README.md
├── requirements.txt
├── src/paging/               # the library
│   ├── config.py             # paths, frozen hyperparameters, experiment configs
│   ├── traces.py             # Stage 1-2: generators, converter, Trace class
│   ├── predictors.py         # Stage 3: Markov / stride / hybrid predictors
│   ├── engine.py             # Stage 4: MemoryEngine + FIFO/LRU/Clock/OPT
│   ├── deciders.py           # Stage 5: prefetch decision engines
│   └── simulate.py           # evaluation loops
├── scripts/                  # one runnable script per stage
│   ├── 01_generate_synthetic.py
│   ├── 02_download_real_traces.py
│   ├── 03_evaluate_predictors.py
│   ├── 04_baselines.py
│   ├── 05_ablation.py
│   ├── 06_head_to_head.py
│   ├── 07_ghost_estimate.py
│   ├── 08_evaluation_harness.py
│   ├── 09_phase8_baselines.py
│   ├── 10_improvements.py / 11_robustness.py / 12_time_model.py
│   └── run_all.py
├── tests/                    # pytest: invariants + pinned reference numbers
├── CLAUDE.md                 # context for Claude Code
├── notebooks/
│   └── colab_quickstart.ipynb
├── docs/
│   ├── PROJECT_CONTEXT.md / RESULTS.md / ROADMAP.md
│   ├── DECISIONS_AND_LESSONS.md / REVIEW_PREP.md / NOTEBOOK_MAP.md
│   ├── implementation_guide.pdf   # full write-up, stages 1-5 (stale in places: see REVIEW_PREP.md)
│   ├── implementation_guide.md
│   └── system_architecture.md
├── results/
│   └── reference/ablation_results.csv   # results reported above
└── data/                     # traces go here (git-ignored)
```

---

## How to run

### Option A — Google Colab (no setup)

1. Open `notebooks/colab_quickstart.ipynb` in Colab (File → Open notebook → GitHub tab → paste this repo's URL).
2. In the first cell, change `YOUR_USERNAME` to the GitHub account hosting this repo.
3. Run the cells top to bottom.

### Option B — Local machine

**Requirements:** Python 3.9+, `git`.

```bash
git clone https://github.com/YOUR_USERNAME/predictive-memory-paging.git
cd predictive-memory-paging
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Quick run (synthetic traces only, about 2 minutes)

```bash
python scripts/run_all.py
```

This generates the six synthetic traces and runs Stages 3, 4 and 5 on them.

#### Full run (with real benchmark traces)

```bash
# 1. synthetic traces
python scripts/01_generate_synthetic.py

# 2. real traces (ML-DPC LoadTraces from SPEC CPU 2006 / GAP)
git clone https://github.com/Quangmire/ChampSim.git
python scripts/02_download_real_traces.py --links ChampSim/download_links

# 3. predictor quality, attribution, memory footprint, capacity sweep
python scripts/03_evaluate_predictors.py

# 4. FIFO / LRU / Clock / OPT baselines, with correctness verification
python scripts/04_baselines.py

# 5. decision-engine ablation (main result, 10-20 minutes)
python scripts/05_ablation.py

# 6. head-to-head vs FIFO / LRU / Clock / OPT
python scripts/06_head_to_head.py

# 7. ghost-list estimate: fault-delta feedback without a shadow engine
python scripts/07_ghost_estimate.py

# 8. evaluation harness: segments, memory sweep, seeds, figures (--quick for a smoke test)
python scripts/08_evaluation_harness.py

# 9. Phase 8: ARC, LRU-2 and Linux-style readahead as baselines
python scripts/09_phase8_baselines.py

# 10-12. improvements, robustness on 13 traces, break-even prefetch cost
python scripts/02_download_real_traces.py --pick 410.bwaves-s1 410.bwaves-s2 437.leslie3d-s1 437.leslie3d-s2 bfs-10 pr-10 433.milc-s0 462.libquantum-s0 429.mcf-s0
python scripts/10_improvements.py
python scripts/11_robustness.py
python scripts/12_time_model.py

# 13. one self-contained HTML report of every result above -> results/report.html
python scripts/13_report.py
```

Run the tests with `pip install -r requirements-dev.txt && pytest -q`.

Each script writes CSV results into `results/`.

To download other benchmarks:

```bash
python scripts/02_download_real_traces.py --pick 429.mcf-s0 cc-5 sssp-3
```

### Custom data and results paths

Set these environment variables to keep data somewhere else, such as Google Drive:

```bash
export PAGING_DATA_DIR=/path/to/data
export PAGING_RESULTS_DIR=/path/to/results
```

---

## Pipeline

| Stage | Script | What it does | Output |
|---|---|---|---|
| 1a | `01_generate_synthetic.py` | Generates sequential, strided, cyclic, clustered, random and phase-changing traces | `data/traces/*.csv` |
| 1b | `02_download_real_traces.py` | Downloads ML-DPC LoadTraces and converts byte addresses to 4 KB page IDs | `data/traces/*.csv` |
| 2 | *(library)* | `Trace` class: time-ordered train/test split, streaming interface | — |
| 3 | `03_evaluate_predictors.py` | Coverage, accuracy and useful rate; sub-predictor attribution; memory footprint; capacity sweep | `predictor_results.csv`, `predictor_attribution.csv`, `capacity_sweep.csv` |
| 4 | `04_baselines.py` | FIFO, LRU, Clock and OPT across 1–25% of each working set; verifies OPT is the lower bound and LRU/OPT are monotonic | `baseline_results.csv` |
| 5 | `05_ablation.py` | Seven deciders on identical predictions; fault reduction, precision, eviction inflation, controller behaviour | `ablation_results.csv`, `decider_summary.csv` |
| 5b | `06_head_to_head.py` | FIFO, LRU, Clock, OPT vs each policy with prediction added | `head_to_head.csv`, `prediction_gain_per_policy.csv` |
| 5c | `07_ghost_estimate.py` | Fault-delta feedback without a shadow engine (ghost list), checked against the oracle | `ghost_estimate_results.csv` |
| 6 | `08_evaluation_harness.py` | Trace segments, memory-size sweep, seeds, figures (not yet run on real traces) | `phase6_*.csv`, `figures/` |
| 8 | `09_phase8_baselines.py` | ARC, LRU-2 and Linux-style readahead as stronger baselines | `phase8_*.csv`, `figures/` |
| 9-10 | `10_improvements.py` | Readahead + Markov, delta-Markov tier, top-2 on the original four traces | `improvements.csv` |
| 12a | `11_robustness.py` | Adopted design on 13 real traces x 1%/5% memory, no retuning | `robustness.csv` |
| 12b | `12_time_model.py` | Break-even cost per prefetch vs LRU | `breakeven.csv` |

### Deciders compared in Stage 5

| Decider | Behaviour |
|---|---|
| `no-prefetch` | Baseline: plain LRU |
| `always` | Acts on every prediction (naive control) |
| `fixed-0.3` / `0.6` / `0.85` | Static confidence thresholds |
| `cost-aware` | Adaptive threshold driven by prefetch precision |
| `cost-aware-fd` | Adaptive threshold driven by measured fault delta against a shadow engine |

---

## Using the library

```python
import sys; sys.path.insert(0, 'src')
from paging.traces import load_trace
from paging.predictors import make_predictor
from paging.engine import LRU
from paging.deciders import CostAwareFaultDelta
from paging.simulate import eval_predictor, run_prefetch_sim

t = load_trace('clustered')
print(eval_predictor(t, make_predictor('hybrid')))
print(run_prefetch_sim(t, 9, LRU, make_predictor('hybrid'), CostAwareFaultDelta()))
```

---

## Data source

Real traces come from the **ML-Based Data Prefetching Competition** (ML-DPC) LoadTraces, distributed through [Quangmire/ChampSim](https://github.com/Quangmire/ChampSim). Each line records `instr_id, cycle, load_address, ip, llc_hit`. Only the load address is used, shifted right by 12 bits to get a 4 KB page number.

Traces are **not** committed to this repository because of their size. The scripts download and convert them.

---

## Limitations

- **LLC-level traces at page granularity.** The real traces record last-level-cache accesses, not OS page faults. Treating them at page granularity is a common approximation, but it is still an approximation.
- **The fault-delta oracle needs a shadow engine**, which a real operating system cannot run. A ghost-list estimate (`CostAwareGhost`) reproduces its decisions without one, but its per-epoch correlation with the true delta is uneven (strong on bwaves, negative on pr-3), so treat it as a promising mechanism, not a settled one.
- **Gains are concentrated.** Large improvements appear on two regular workloads. On irregular ones, the decision engine's value is in limiting harm, not producing gains.
- **Fault-delta parameters are untuned.** `epoch`, `dead_band` and `step` were chosen once and not swept.
- **Single run per configuration.** Robustness across trace segments and seeds is future work (harness built, not yet run on real traces).
- **Baselines.** ARC, LRU-2 and readahead are implemented (Phase 8) but their real-trace results are pending.

---

## References

See `docs/implementation_guide.pdf` for the full literature review. Key sources:

- D. Joseph and D. Grunwald, "Prefetching using Markov predictors," ISCA 1997.
- ML-DPC / ChampSim — https://github.com/Quangmire/ChampSim
