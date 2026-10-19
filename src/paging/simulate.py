"""
Evaluation loops.

eval_predictor   — Stage 3: prediction quality only (no memory engine)
rolling_accuracy — Stage 3: accuracy over time, for phase-change plots
run_prefetch_sim — Stage 5: predictor + decider + memory engine together
"""
import numpy as np

from .engine import MemoryEngine


def eval_predictor(trace, predictor, warmup_frac=0.25, hist_len=8):
    """Stream the trace; learn throughout, measure only past warm-up."""
    pages = trace.pages
    split = int(len(pages) * warmup_frac)
    history = []
    attempted = correct = 0
    conf_sum = 0.0

    for i in range(len(pages) - 1):
        history.append(int(pages[i]))
        if len(history) > hist_len:
            history.pop(0)
        actual = int(pages[i + 1])

        if i >= split:
            preds = predictor.predict(history, top_n=1)
            if preds:
                page, conf = preds[0]
                attempted += 1
                conf_sum += conf
                if page == actual:
                    correct += 1

        predictor.update(history, actual)     # online learning throughout

    n_eval = max(1, len(pages) - 1 - split)
    coverage = attempted / n_eval
    accuracy = correct / attempted if attempted else 0.0
    return {'coverage': coverage, 'accuracy': accuracy,
            'useful': coverage * accuracy,
            'mean_conf': conf_sum / attempted if attempted else 0.0,
            'extra': predictor.stats()}


def rolling_accuracy(trace, predictor, window=300, warmup_frac=0.0,
                     hist_len=8, step=50):
    pages = trace.pages
    split = int(len(pages) * warmup_frac)
    hist, hits, xs, ys = [], [], [], []
    for i in range(len(pages) - 1):
        hist.append(int(pages[i]))
        if len(hist) > hist_len:
            hist.pop(0)
        actual = int(pages[i + 1])
        if i >= split:
            p = predictor.predict(hist, top_n=1)
            hits.append(1 if (p and p[0][0] == actual) else 0)
            if len(hits) >= window and len(hits) % step == 0:
                xs.append(i)
                ys.append(float(np.mean(hits[-window:])))
        predictor.update(hist, actual)
    return xs, ys


def run_prefetch_sim(trace, n_frames, policy_cls, predictor, decider,
                     hist_len=8, top_n=1, log=None):
    if hasattr(decider, 'estimate_update'):
        raise ValueError(f'{decider.name} is a ghost-list decider: use run_prefetch_sim_ghost '
                         '(here it would silently run as the shadow-engine oracle)')
    eng = MemoryEngine(n_frames, policy_cls(pages=trace.pages))
    eng.log = log
    history = []
    prev_wasted = 0

    uses_shadow = hasattr(decider, 'epoch_update')
    shadow = MemoryEngine(n_frames, policy_cls(pages=trace.pages)) if uses_shadow else None

    for i, page, _ in trace.stream():
        # 1. demand access (correctness path)
        was_pf = eng.prefetched.get(page, False)
        eng.access(page, i)
        if shadow is not None:
            shadow.access(page, i)
        if was_pf:
            decider.feedback(page, True)

        # 2. report wasted prefetches
        if eng.n_pf_wasted > prev_wasted:
            for _ in range(eng.n_pf_wasted - prev_wasted):
                decider.feedback(None, False)
            prev_wasted = eng.n_pf_wasted

        # 3. fault-delta epoch comparison
        if uses_shadow and eng.n_access % decider.epoch == 0:
            decider.epoch_update(eng.n_fault, shadow.n_fault, eng.n_access)

        # 4. update model
        history.append(page)
        if len(history) > hist_len:
            history.pop(0)
        if len(history) >= 2:
            predictor.update(history[:-1], page)

        # 5. predict + decide
        for cand, conf in predictor.predict(history, top_n=top_n):
            if cand in eng.resident:
                continue
            state = {'free_frames': eng.n_frames - len(eng.resident),
                     'n_frames': eng.n_frames,
                     'n_prefetch': eng.n_prefetch,
                     'n_access': eng.n_access,
                     'n_fault': eng.n_fault}
            if decider.decide(cand, conf, state):
                eng.prefetch(cand, i)

    s = eng.stats()
    s['decider'] = decider.name
    s.update(decider.stats())
    if shadow is not None:
        s['shadow_faults'] = shadow.n_fault
    return s


def run_prefetch_sim_ghost(trace, n_frames, policy_cls, predictor, decider, hist_len=8, log=None):
    """
    Runs the oracle (CostAwareFaultDelta) or deployable (CostAwareGhost) decider.
    A shadow engine always runs, but only to LOG the true fault delta for comparison;
    the ghost decider never sees it. Returns stats plus 'pf_harm' and
    'epoch_log' = [(estimated_delta, true_delta), ...].
    """
    from .engine import GhostMemoryEngine
    eng = GhostMemoryEngine(n_frames, policy_cls(pages=trace.pages))
    eng.log = log
    shadow = MemoryEngine(n_frames, policy_cls(pages=trace.pages))
    history, prev_wasted = [], 0
    prev_real = prev_shadow = prev_used = prev_harm = 0
    log = []

    for i, page, _ in trace.stream():
        was_pf = eng.prefetched.get(page, False)
        eng.access(page, i)
        shadow.access(page, i)
        if was_pf:
            decider.feedback(page, True)
        if eng.n_pf_wasted > prev_wasted:
            for _ in range(eng.n_pf_wasted - prev_wasted):
                decider.feedback(None, False)
            prev_wasted = eng.n_pf_wasted

        if eng.n_access % decider.epoch == 0:
            est = (eng.n_pf_used - prev_used) - (eng.n_pf_harm - prev_harm)
            true = (shadow.n_fault - prev_shadow) - (eng.n_fault - prev_real)
            log.append((est, true))
            prev_used, prev_harm = eng.n_pf_used, eng.n_pf_harm
            prev_real, prev_shadow = eng.n_fault, shadow.n_fault
            if hasattr(decider, 'estimate_update'):
                decider.estimate_update(eng.n_pf_used, eng.n_pf_harm)
            else:
                decider.epoch_update(eng.n_fault, shadow.n_fault, eng.n_access)

        history.append(page)
        if len(history) > hist_len:
            history.pop(0)
        if len(history) >= 2:
            predictor.update(history[:-1], page)

        for cand, conf in predictor.predict(history, top_n=1):
            if cand in eng.resident:
                continue
            state = {'free_frames': eng.n_frames - len(eng.resident),
                     'n_frames': eng.n_frames, 'n_prefetch': eng.n_prefetch,
                     'n_access': eng.n_access, 'n_fault': eng.n_fault}
            if decider.decide(cand, conf, state):
                eng.prefetch(cand, i)

    s = eng.stats()
    s['decider'] = decider.name
    s.update(decider.stats())
    s['pf_harm'] = eng.n_pf_harm
    s['epoch_log'] = log
    return s


def sampled(page, rate):
    """Deterministic hash sampling of page numbers (SHARDS-style). rate=1.0 keeps every page."""
    if rate >= 1.0:
        return True
    return ((int(page) * 2654435761) & 0xFFFFFFFF) < rate * 4294967296


def run_prefetch_sim_dir(trace, n_frames, policy_cls, predictor, decider, hist_len=8, log=None):
    """
    Phase 14: feedback from a shadow directory (page IDs only, no-prefetch copy of the policy).
    The directory covers the sampled pages only, with sample x n_frames entries; estimated
    delta per epoch = (directory faults - real faults on sampled pages) / sample.
    A full shadow engine still runs, but only to LOG the true delta; the decider never sees it.
    Returns stats plus 'epoch_log' = [(estimated_delta, true_delta), ...] and 'dir_entries'.
    """
    s = decider.sample
    d_frames = max(1, round(n_frames * s))
    eng = MemoryEngine(n_frames, policy_cls(pages=trace.pages))
    eng.log = log
    shadow = MemoryEngine(n_frames, policy_cls(pages=trace.pages))
    direc = MemoryEngine(d_frames, policy_cls(pages=trace.pages))
    history, prev_wasted, log = [], 0, []
    real_s = dir_s = prev_real = prev_shadow = 0

    for i, page, _ in trace.stream():
        was_pf = eng.prefetched.get(page, False)
        fault = eng.access(page, i)
        shadow.access(page, i)
        if sampled(page, s):
            real_s += fault
            dir_s += direc.access(page, i)
        if was_pf:
            decider.feedback(page, True)
        if eng.n_pf_wasted > prev_wasted:
            for _ in range(eng.n_pf_wasted - prev_wasted):
                decider.feedback(None, False)
            prev_wasted = eng.n_pf_wasted

        if eng.n_access % decider.epoch == 0:
            est = (dir_s - real_s) / s
            true = (shadow.n_fault - prev_shadow) - (eng.n_fault - prev_real)
            log.append((est, true))
            real_s = dir_s = 0
            prev_real, prev_shadow = eng.n_fault, shadow.n_fault
            decider.directory_update(est)

        history.append(page)
        if len(history) > hist_len:
            history.pop(0)
        if len(history) >= 2:
            predictor.update(history[:-1], page)

        for cand, conf in predictor.predict(history, top_n=1):
            if cand in eng.resident:
                continue
            state = {'free_frames': eng.n_frames - len(eng.resident),
                     'n_frames': eng.n_frames, 'n_prefetch': eng.n_prefetch,
                     'n_access': eng.n_access, 'n_fault': eng.n_fault}
            if decider.decide(cand, conf, state):
                eng.prefetch(cand, i)

    st = eng.stats()
    st['decider'] = decider.name
    st.update(decider.stats())
    st['epoch_log'] = log
    st['dir_entries'] = d_frames
    return st
