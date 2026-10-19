"""
Regression and invariant tests. Fast (a few seconds); run with:  pytest -q

The pinned numbers come from the Colab notebook runs reported in the project documents
(cell 18 head-to-head and cell 19 ghost-list result, synthetic 'clustered' trace, seed 0).
If one of them changes, behaviour changed: investigate before updating the number.
"""
import numpy as np
import pytest

from paging.traces import Trace, sequential, strided, cyclic, clustered, uniform_random, phase_changing
from paging.engine import (MemoryEngine, GhostMemoryEngine, FIFO, LRU, Clock, OPTFast, ARC, LRU2,
                           POLICIES_EXTENDED, run_policy)
from paging.predictors import make_predictor, make_readahead, StridePredictor
from paging.deciders import AlwaysPrefetch, CostAwareGhost
from paging.simulate import run_prefetch_sim, run_prefetch_sim_ghost


def faults(trace, nf, P):
    return run_policy(trace, nf, P)['faults']


@pytest.fixture(scope='module')
def traces():
    return {'sequential': Trace('sequential', sequential()),
            'strided': Trace('strided', strided()),
            'cyclic': Trace('cyclic', cyclic()),
            'clustered': Trace('clustered', clustered()),
            'random': Trace('random', uniform_random()),
            'phase_changing': Trace('phase_changing', phase_changing())}


# ---------------------------------------------------------------- engine invariants
def test_opt_is_a_lower_bound_for_every_policy(traces):
    for name, t in traces.items():
        nf = max(8, int(len(np.unique(t.pages)) * 0.02))
        r = {k: faults(t, nf, P) for k, P in POLICIES_EXTENDED.items()}
        assert all(r[k] >= r['OPT'] for k in r), (name, r)


def test_lru_and_opt_never_get_worse_with_more_memory(traces):
    t = traces['phase_changing']
    for P in (LRU, OPTFast):
        f = [faults(t, nf, P) for nf in (50, 100, 200, 400)]
        assert f == sorted(f, reverse=True), (P.__name__, f)


def test_with_plenty_of_memory_faults_equal_unique_pages(traces):
    t = traces['cyclic']
    uniq = len(np.unique(t.pages))
    for k, P in POLICIES_EXTENDED.items():
        assert faults(t, 4 * uniq, P) == uniq, k


def test_sequential_trace_always_faults_without_prefetching(traces):
    t = traces['sequential']
    for P in (FIFO, LRU, Clock):
        assert faults(t, 64, P) == len(t.pages)


def test_ghost_engine_matches_plain_engine_without_prefetching(traces):
    t = traces['clustered']
    a = MemoryEngine(9, LRU(pages=t.pages))
    b = GhostMemoryEngine(9, LRU(pages=t.pages))
    for i, p, _ in t.stream():
        a.access(p, i); b.access(p, i)
    assert (a.n_fault, a.n_evict) == (b.n_fault, b.n_evict)


# ---------------------------------------------------------------- pinned reference numbers
def test_pinned_head_to_head_numbers_clustered_9(traces):
    """cell 18 / cell 19 results on the synthetic clustered trace."""
    t = traces['clustered']
    assert faults(t, 9, FIFO) == 5729
    assert faults(t, 9, Clock) == 5676
    assert faults(t, 9, LRU) == 5803
    assert faults(t, 9, OPTFast) == 2637
    got = run_prefetch_sim_ghost(t, 9, LRU, make_predictor('hybrid'), CostAwareGhost())['faults']
    assert got == 5798          # ghost-list decider, identical to the notebook's cell 19


def test_pinned_phase8_numbers_clustered_9(traces):
    """Phase 8 notebook cells 25-29, same configuration."""
    t = traces['clustered']
    assert faults(t, 9, LRU2) == 6512
    assert faults(t, 9, ARC) == 6207
    assert run_prefetch_sim(t, 9, LRU, make_readahead(9), AlwaysPrefetch())['faults'] == 7645


# ---------------------------------------------------------------- Phase 8 policies
def _scan_trace():
    rng = np.random.default_rng(0)
    seq, scan = [], 1000
    for _ in range(20_000):
        if rng.random() < 0.5:
            seq.append(int(rng.integers(0, 12)))     # reused pages
        else:
            seq.append(scan); scan += 1              # one-time scan
    return Trace('scan', seq)


def test_arc_and_lru2_resist_scans():
    t = _scan_trace()
    r = {k: faults(t, 16, P) for k, P in POLICIES_EXTENDED.items()}
    assert r['LRU-2'] < 0.75 * r['LRU'] and r['ARC'] < 0.75 * r['LRU'], r
    assert r['OPT'] == min(r.values())


def test_arc_keeps_ghost_lists_bounded():
    t = _scan_trace()
    arc = ARC()
    e = MemoryEngine(16, arc)
    for i, p, _ in t.stream():
        e.access(p, i)
    assert len(arc.T1) + len(arc.T2) == 16
    assert len(arc.T1) + len(arc.B1) <= 16
    assert len(arc.T1) + len(arc.T2) + len(arc.B1) + len(arc.B2) <= 32
    assert 0 <= arc.p <= 16


# ---------------------------------------------------------------- predictors
def test_readahead_behaves_like_sequential_readahead(traces):
    seq = traces['sequential']
    base = faults(seq, 64, LRU)
    got = run_prefetch_sim(seq, 64, LRU, make_readahead(64), AlwaysPrefetch())['faults']
    assert got < 0.05 * base                              # sequential stream: almost no faults
    for name in ('strided', 'random'):                    # not sequential: no effect
        t = traces[name]
        b = faults(t, 64, LRU)
        g = run_prefetch_sim(t, 64, LRU, make_readahead(64), AlwaysPrefetch())['faults']
        assert abs(g - b) <= 0.01 * b, name


def test_stride_predictor_needs_three_matching_gaps():
    s = StridePredictor()
    assert s.predict([10, 18, 26]) == []                  # only two gaps
    assert s.predict([2, 10, 18, 26]) == [(34, 1.0)]
    assert s.predict([2, 10, 18, 27]) == []               # gaps disagree


def test_hybrid_falls_back_to_stride_when_markov_table_is_too_small(traces):
    h = make_predictor('hybrid')
    t = traces['sequential']
    hist = []
    for i, p, _ in t.stream():
        hist.append(p)
        hist = hist[-8:]
        if len(hist) >= 2:
            h.update(hist[:-1], p)
        h.predict(hist)
        if i > 20000:
            break
    src = h.stats()['sources']
    assert src['stride'] > 0.95 * sum(src.values())      # 10,000-page loop >> 256-entry table


# ---------------------------------------------------------------- improvements (script 10)
def test_delta_markov_predicts_unseen_pages_of_a_repeating_gap_pattern():
    from paging.predictors import DeltaMarkovPredictor
    d = DeltaMarkovPredictor()
    pages, p = [], 1000
    for k in range(60):                      # gaps 1,3,1,3 ... : page numbers never repeat
        pages.append(p); p += 1 if k % 2 == 0 else 3
    hist, hits, tried = [], 0, 0
    for i, x in enumerate(pages):
        hist.append(x); hist = hist[-8:]
        if i + 1 < len(pages):
            r = d.predict(hist)
            if i > 20 and r:
                tried += 1; hits += r[0][0] == pages[i + 1]
            d.update(hist, pages[i + 1])
    assert tried > 30 and hits == tried


def test_predictive_lru_with_feature_off_is_exactly_lru(traces):
    from paging.engine import PredictiveLRU
    t = traces['clustered']
    mk = lambda pages=None, **k: PredictiveLRU(enabled=False)
    a = run_prefetch_sim(t, 9, LRU, make_readahead(9), AlwaysPrefetch())['faults']
    b = run_prefetch_sim(t, 9, mk, make_readahead(9), AlwaysPrefetch())['faults']
    assert a == b == 7645


def test_readahead_min_run_default_reproduces_original(traces):
    from paging.predictors import ReadaheadPredictor
    assert ReadaheadPredictor().min_run == 2 and make_readahead(9).min_run == 2


# ---------------------------------------------------------------- guards and structural facts
def test_ghost_decider_is_refused_by_the_plain_simulator(traces):
    with pytest.raises(ValueError):
        run_prefetch_sim(traces['clustered'], 9, LRU, make_predictor('hybrid'), CostAwareGhost())


def test_markov_confidence_is_never_below_half_with_two_successors():
    """Structural, not empirical: top-1 of at most 2 successors always holds >= 50% of the counts."""
    rng = np.random.default_rng(0)
    h = make_predictor('hybrid')
    hist, lo = [], 1.0
    for x in rng.integers(0, 40, 20000).tolist():
        hist = (hist + [x])[-8:]
        if len(hist) >= 2:
            h.update(hist[:-1], x)
        r = h.predict(hist)
        if r:
            lo = min(lo, r[0][1])
    assert lo >= 0.5


def test_pinned_improvement_numbers_clustered(traces):
    """script 10/11 designs on synthetic clustered (RESULTS sections 9 and 12)."""
    from paging.predictors import StreamMarkovPredictor
    t = traces['clustered']
    mk = lambda: make_predictor('delta-hybrid')
    assert run_prefetch_sim(t, 9, LRU, StreamMarkovPredictor(9, markov=mk()), AlwaysPrefetch())['faults'] == 7053
    gated = StreamMarkovPredictor(6, markov=mk(), readahead=make_readahead(6, conf_k=1))
    assert run_prefetch_sim_ghost(t, 6, LRU, gated, CostAwareGhost())['faults'] == 20498


# ---------------------------------------------------------------- Phase 14: shadow directory
def test_full_shadow_directory_measures_the_true_delta_every_epoch(traces):
    from paging.deciders import CostAwareDirectory
    from paging.simulate import run_prefetch_sim_dir
    r = run_prefetch_sim_dir(traces['clustered'], 9, LRU, make_predictor('hybrid'), CostAwareDirectory())
    assert r['epoch_log'] and all(est == true for est, true in r['epoch_log'])


def test_allow_off_disabled_reproduces_the_ghost_decider(traces):
    a = run_prefetch_sim_ghost(traces['clustered'], 9, LRU, make_predictor('hybrid'), CostAwareGhost())['faults']
    b = run_prefetch_sim_ghost(traces['clustered'], 9, LRU, make_predictor('hybrid'), CostAwareGhost(allow_off=False))['faults']
    assert a == b == 5798


def test_pinned_directory_number_clustered_9(traces):
    """RA + delta-Markov (gated) with directory feedback: -21.5% (ghost) becomes -2.5% vs LRU."""
    from paging.predictors import StreamMarkovPredictor
    from paging.deciders import CostAwareDirectory
    from paging.simulate import run_prefetch_sim_dir
    p = StreamMarkovPredictor(9, markov=make_predictor('delta-hybrid'), readahead=make_readahead(9, conf_k=1))
    got = run_prefetch_sim_dir(traces['clustered'], 9, LRU, p, CostAwareDirectory())['faults']
    assert got == 5948


# ---------------------------------------------------------------- Phase 15: time model
def test_time_model_without_prefetching_is_compute_plus_faults(traces):
    from paging.deciders import NoPrefetch
    from paging.timing import replay
    log = []
    r = run_prefetch_sim(traces['clustered'], 9, LRU, make_predictor('hybrid'), NoPrefetch(), log=log)
    assert r['faults'] == 5803 and len(log) == 5803 and all(e[0] == 'F' for e in log)
    tm = replay(log, r['accesses'], t_io=1.0, channels=4, t_cpu=0.01)
    assert abs(tm['time'] - (r['accesses'] * 0.01 + 5803 * 1.0)) < 1e-6


def test_time_model_charges_late_prefetches_and_queueing():
    from paging.timing import replay
    # access 0 prefetches page 7; access 1 uses it before the read (t_io=1) has finished
    late = replay([('P', 0, 7), ('U', 1, 7)], 2, t_io=1.0, channels=1, t_cpu=0.1)
    assert abs(late['stall_late'] - 0.9) < 1e-9 and abs(late['time'] - 1.1) < 1e-9
    # one channel: a demand fault queues behind an earlier prefetch read
    q = replay([('P', 0, 7), ('F', 1, 8)], 2, t_io=1.0, channels=1, t_cpu=0.1)
    assert abs(q['time'] - 2.1) < 1e-9


# ---------------------------------------------------------------- Phase 16: the frozen final system
def test_pinned_final_system_clustered(traces):
    """paging.final (RA + delta-Markov, directory feedback, frozen controller); LRU = 5803 / 20344."""
    from paging.final import run_final
    assert run_final(traces['clustered'], 9)['faults'] == 5942
    assert run_final(traces['clustered'], 6)['faults'] == 20522
