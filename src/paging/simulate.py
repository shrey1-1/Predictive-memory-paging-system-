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
                     hist_len=8, top_n=1):
    eng = MemoryEngine(n_frames, policy_cls(pages=trace.pages))
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
