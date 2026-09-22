"""
Stage 3 — prediction model.

MarkovPredictor : bounded LRU transition table, top-K successors, periodic decay
StridePredictor : next = last + stride when recent deltas agree
HybridPredictor : order-2 Markov -> order-1 Markov -> stride fallback chain
"""
from collections import OrderedDict

from .config import FROZEN


class MarkovPredictor:
    """Counts observed transitions. Bounded LRU table, top-K successors."""

    def __init__(self, capacity=256, max_succ=2, order=2,
                 decay_every=25_000, min_count=1):
        self.capacity = capacity
        self.max_succ = max_succ
        self.order = order
        self.decay_every = decay_every
        self.min_count = min_count
        self.table = OrderedDict()      # context -> {successor: count}
        self.n_updates = 0
        self.evictions = 0

    def _key(self, history):
        if len(history) < self.order:
            return None
        return tuple(history[-self.order:])

    def update(self, history, actual_next):
        k = self._key(history)
        if k is None:
            return
        if k in self.table:
            self.table.move_to_end(k)
        else:
            if len(self.table) >= self.capacity:
                self.table.popitem(last=False)
                self.evictions += 1
            self.table[k] = {}

        succ = self.table[k]
        succ[actual_next] = succ.get(actual_next, 0) + 1
        if len(succ) > self.max_succ:
            del succ[min(succ, key=succ.get)]

        self.n_updates += 1
        if self.n_updates % self.decay_every == 0:
            self._decay()

    def _decay(self):
        for succ in self.table.values():
            for s in list(succ):
                succ[s] >>= 1
                if succ[s] == 0:
                    del succ[s]

    def predict(self, history, top_n=1):
        k = self._key(history)
        if k is None or k not in self.table:
            return []
        succ = self.table[k]
        total = sum(succ.values())
        if total == 0:
            return []
        ranked = sorted(succ.items(), key=lambda kv: -kv[1])[:top_n]
        return [(p, c / total) for p, c in ranked if c >= self.min_count]

    def stats(self):
        return {'entries': len(self.table), 'evictions': self.evictions,
                'fill': round(len(self.table) / self.capacity, 3)}


class StridePredictor:
    """Predicts next = last + stride when recent deltas agree."""

    def __init__(self, confirm=3):
        self.confirm = confirm

    def predict(self, history, top_n=1):
        if len(history) < self.confirm + 1:
            return []
        d = [history[i] - history[i - 1] for i in range(-self.confirm, 0)]
        if len(set(d)) != 1 or d[0] == 0:
            return []
        return [(history[-1] + d[0], 1.0)]

    def update(self, history, actual_next):
        pass

    def stats(self):
        return {}


class HybridPredictor:
    """Order-2 Markov first (precise), order-1 fallback (broad), then stride."""

    def __init__(self, capacity=256, max_succ=2, decay_every=25_000):
        self.m2 = MarkovPredictor(capacity, max_succ, order=2, decay_every=decay_every)
        self.m1 = MarkovPredictor(capacity, max_succ, order=1, decay_every=decay_every)
        self.st = StridePredictor()
        self.source_counts = {'m2': 0, 'm1': 0, 'stride': 0, 'none': 0}

    def predict(self, history, top_n=1):
        for tag, p in (('m2', self.m2), ('m1', self.m1), ('stride', self.st)):
            r = p.predict(history, top_n)
            if r:
                self.source_counts[tag] += 1
                return r
        self.source_counts['none'] += 1
        return []

    def update(self, history, actual_next):
        self.m2.update(history, actual_next)
        self.m1.update(history, actual_next)

    def stats(self):
        return {'sources': dict(self.source_counts), **self.m2.stats()}


KINDS = ['stride', 'markov-o1', 'markov-o2', 'hybrid']


def make_predictor(kind, cfg=None):
    """Single place that constructs predictors, always from the frozen config."""
    cfg = cfg or FROZEN
    if kind == 'stride':
        return StridePredictor()
    if kind == 'markov-o1':
        return MarkovPredictor(**{**cfg, 'order': 1})
    if kind == 'markov-o2':
        return MarkovPredictor(**cfg)
    if kind == 'hybrid':
        return HybridPredictor(capacity=cfg['capacity'], max_succ=cfg['max_succ'],
                               decay_every=cfg['decay_every'])
    raise ValueError(kind)
