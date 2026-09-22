"""
Stage 5 — prefetch decision engines.

All deciders share one interface:
    decide(page, conf, state) -> bool
    feedback(page, used)      -> None
    stats()                   -> dict
CostAwareFaultDelta additionally exposes epoch_update(), which makes the
simulator create a shadow (no-prefetch) engine for it.
"""
import numpy as np


class NoPrefetch:
    name = 'no-prefetch'
    def decide(self, page, conf, state): return False
    def feedback(self, page, used): pass
    def stats(self): return {}


class AlwaysPrefetch:
    """Naive control: act on every prediction, ignore all cost."""
    name = 'always'
    def decide(self, page, conf, state): return True
    def feedback(self, page, used): pass
    def stats(self): return {}


class FixedThreshold:
    def __init__(self, thr=0.6):
        self.thr = thr
        self.name = f'fixed-{thr}'

    def decide(self, page, conf, state): return conf >= self.thr
    def feedback(self, page, used): pass
    def stats(self): return {'threshold': self.thr}


class CostAware:
    """
    Adaptive threshold driven by memory pressure and observed prefetch PRECISION.
    Kept as a comparison point — the experiments show precision is a poor
    feedback signal under high eviction pressure.
    """
    name = 'cost-aware'

    def __init__(self, base_thr=0.5, lo=0.2, hi=0.9, window=200, update_every=100,
                 step=0.02, target_lo=0.45, target_hi=0.75, budget_ratio=0.35):
        self.thr = base_thr
        self.lo, self.hi = lo, hi
        self.window = window
        self.update_every = update_every
        self.step = step
        self.target_lo, self.target_hi = target_lo, target_hi
        self.budget_ratio = budget_ratio
        self.recent, self.thr_history = [], []
        self.n_feedback = 0
        self.d_budget = self.d_cost = self.d_thr = self.n_yes = 0

    def decide(self, page, conf, state):
        if state['n_prefetch'] > self.budget_ratio * max(1, state['n_access']):
            self.d_budget += 1
            return False
        if state['free_frames'] > 0:
            ok = conf >= self.lo
            if ok: self.n_yes += 1
            else:  self.d_cost += 1
            return ok
        if conf <= (1.0 - conf):
            self.d_cost += 1
            return False
        if conf < self.thr:
            self.d_thr += 1
            return False
        self.n_yes += 1
        return True

    def feedback(self, page, used):
        self.recent.append(1 if used else 0)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        self.n_feedback += 1
        if self.n_feedback % self.update_every:
            return
        if len(self.recent) < 50:
            return
        prec = sum(self.recent) / len(self.recent)
        if prec < self.target_lo:
            self.thr = min(self.hi, self.thr + self.step)
        elif prec > self.target_hi:
            self.thr = max(self.lo, self.thr - self.step)
        self.thr_history.append(self.thr)

    def stats(self):
        th = self.thr_history
        return {'final_thr': round(self.thr, 3),
                'thr_min': round(min(th), 3) if th else None,
                'thr_max': round(max(th), 3) if th else None,
                'thr_moves': len(th), 'approved': self.n_yes,
                'd_budget': self.d_budget, 'd_cost': self.d_cost, 'd_thr': self.d_thr}


class CostAwareFaultDelta:
    """
    Adapts its threshold using measured fault delta against a shadow
    no-prefetch engine:  delta = shadow_faults - real_faults.
    Positive -> prefetching helps -> lower threshold. Negative -> raise it.

    NOTE: the shadow engine is an experimental oracle; a real OS cannot run one.
    """
    name = 'cost-aware-fd'

    def __init__(self, base_thr=0.5, lo=0.15, hi=0.95, epoch=2000, step=0.05,
                 budget_ratio=0.35, dead_band=0.002):
        self.thr = base_thr
        self.lo, self.hi = lo, hi
        self.epoch = epoch
        self.step = step
        self.budget_ratio = budget_ratio
        self.dead_band = dead_band
        self.thr_history, self.delta_history = [], []
        self.d_budget = self.d_cost = self.d_thr = self.n_yes = 0

    def decide(self, page, conf, state):
        if state['n_prefetch'] > self.budget_ratio * max(1, state['n_access']):
            self.d_budget += 1
            return False
        if state['free_frames'] > 0:
            ok = conf >= self.lo
            if ok: self.n_yes += 1
            else:  self.d_cost += 1
            return ok
        if conf < self.thr:
            self.d_thr += 1
            return False
        self.n_yes += 1
        return True

    def epoch_update(self, real_faults, shadow_faults, n_access):
        delta_rate = (shadow_faults - real_faults) / max(1, self.epoch)
        self.delta_history.append(round(delta_rate, 5))
        if delta_rate > self.dead_band:
            self.thr = max(self.lo, self.thr - self.step)
        elif delta_rate < -self.dead_band:
            self.thr = min(self.hi, self.thr + self.step)
        self.thr_history.append(round(self.thr, 3))

    def feedback(self, page, used):
        pass    # precision deliberately ignored

    def stats(self):
        th, dh = self.thr_history, self.delta_history
        return {'final_thr': round(self.thr, 3),
                'thr_min': min(th) if th else None,
                'thr_max': max(th) if th else None,
                'thr_moves': len(th),
                'mean_delta_rate': round(float(np.mean(dh)), 5) if dh else None,
                'epochs_winning': sum(1 for d in dh if d > 0),
                'epochs_losing': sum(1 for d in dh if d < 0),
                'approved': self.n_yes,
                'd_budget': self.d_budget, 'd_cost': self.d_cost, 'd_thr': self.d_thr}


DECIDERS = [
    lambda: NoPrefetch(),
    lambda: AlwaysPrefetch(),
    lambda: FixedThreshold(0.3),
    lambda: FixedThreshold(0.6),
    lambda: FixedThreshold(0.85),
    lambda: CostAware(),
    lambda: CostAwareFaultDelta(),
]
