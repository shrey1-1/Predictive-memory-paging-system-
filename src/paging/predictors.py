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
                 decay_every=25_000, min_count=1, smooth=0.0):
        self.smooth = smooth             # confidence = count / (total + smooth); 0 = original
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
        return [(p, c / (total + self.smooth)) for p, c in ranked if c >= self.min_count]

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


class DeltaMarkovPredictor:
    """
    Markov chain over page DELTAS (gaps), not page numbers (Phase 10 idea).
    Context = last `order` deltas, successor = next delta; prediction = last page + delta.
    One entry covers a pattern anywhere in the address space, so it can predict pages it has
    never seen. A zero delta (same page again) is never predicted. Same bounded-table
    machinery (and cost per entry) as MarkovPredictor.
    """

    def __init__(self, capacity=256, max_succ=2, order=2, decay_every=25_000):
        self.order = order
        self.mk = MarkovPredictor(capacity, max_succ, order, decay_every)

    def _deltas(self, history):
        h = history[-(self.order + 1):]
        return [h[i] - h[i - 1] for i in range(1, len(h))]

    def update(self, history, actual_next):
        if len(history) < self.order + 1:
            return
        d = actual_next - history[-1]
        if d != 0:
            self.mk.update(self._deltas(history), d)

    def predict(self, history, top_n=1):
        if len(history) < self.order + 1:
            return []
        return [(history[-1] + d, c) for d, c in self.mk.predict(self._deltas(history), top_n) if d != 0]

    def stats(self):
        return self.mk.stats()


class DeltaHybridPredictor(HybridPredictor):
    """order-2 page Markov -> order-1 page Markov -> delta Markov -> stride."""

    def __init__(self, capacity=256, max_succ=2, decay_every=25_000):
        super().__init__(capacity, max_succ, decay_every)
        self.dm = DeltaMarkovPredictor(capacity, max_succ, 2, decay_every)
        self.source_counts['delta'] = 0

    def predict(self, history, top_n=1):
        for tag, p in (('m2', self.m2), ('m1', self.m1), ('delta', self.dm), ('stride', self.st)):
            r = p.predict(history, top_n)
            if r:
                self.source_counts[tag] += 1
                return r
        self.source_counts['none'] += 1
        return []

    def update(self, history, actual_next):
        super().update(history, actual_next)
        self.dm.update(history, actual_next)


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
    if kind == 'delta-hybrid':
        return DeltaHybridPredictor(capacity=cfg['capacity'], max_succ=cfg['max_succ'],
                                    decay_every=cfg['decay_every'])
    if kind == 'hybrid':
        return HybridPredictor(capacity=cfg['capacity'], max_succ=cfg['max_succ'],
                               decay_every=cfg['decay_every'])
    raise ValueError(kind)


class ReadaheadPredictor:
    """
    Sequential readahead, modelled on the Linux page cache:
      - two consecutive pages (n, n+1) start a stream: fetch an initial window ahead
      - while the stream continues, each time the program reaches the window's async marker
        (half way through) fetch the next window, DOUBLE the window, up to max_window
      - any non-sequential access resets the stream
    Repeats of the same page are ignored (LLC traces contain many).
    Unlike the Markov predictor it returns a whole window of pages, and run_prefetch_sim
    prefetches every candidate returned, so pair it with AlwaysPrefetch.
    """

    def __init__(self, init_window=4, max_window=32, min_run=2, conf_k=None):
        self.conf_k = conf_k             # None: confidence 1.0 (original); k: run/(run+k), so a decider can gate short streams
        self.min_run = min_run           # consecutive pages needed to start a stream (2 = original)
        self.run = 0
        self.init_window = init_window
        self.max_window = max_window
        self.last = None
        self.win = 0
        self.issued = None               # highest page already requested
        self.pending = []
        self.n_issued = 0
        self.n_resets = 0

    def _issue(self, base, n):
        self.pending = [base + k for k in range(1, n + 1)]
        self.issued = base + n
        self.n_issued += n

    def update(self, history, actual_next):
        page = int(actual_next)
        self.pending = []
        if self.last is None:
            self.last, self.issued = page, page
            return
        if page == self.last:            # same page again: no information
            return
        if page == self.last + 1:        # sequential continuation
            self.last = page
            self.run += 1
            if self.win == 0:
                if self.run < self.min_run - 1:
                    return
                self.win = min(self.init_window, self.max_window)
                self._issue(page, self.win)
            elif page >= self.issued - self.win // 2:
                self.win = min(self.win * 2, self.max_window)
                self._issue(self.issued, self.win)
        else:                            # stream broken
            self.last, self.win, self.issued = page, 0, page
            self.run = 0
            self.n_resets += 1

    def predict(self, history, top_n=1):
        c = 1.0 if not self.conf_k else self.run / (self.run + self.conf_k)
        return [(p, c) for p in self.pending]

    def stats(self):
        return {'issued': self.n_issued, 'resets': self.n_resets}


class TopN:
    """Wrap a predictor so it always returns its top-n guesses (n=2: prefetch the two likeliest)."""

    def __init__(self, inner, n=2):
        self.inner, self.n = inner, n

    def update(self, history, actual_next):
        self.inner.update(history, actual_next)

    def predict(self, history, top_n=1):
        return self.inner.predict(history, top_n=self.n)

    def stats(self):
        return self.inner.stats()


class StreamMarkovPredictor:
    """
    Readahead + Markov hybrid (Phase 9a, item 1 of the improvement plan).
    Readahead covers NEW pages in a sequential run (Markov cannot predict a page it has
    never seen); the Markov hybrid covers repeating patterns. Both learn on every access.
    predict() returns the readahead window first (confidence 1.0), then the Markov guess if
    it is not already in the list. No new tunable parameters: the readahead window comes
    from make_readahead(n_frames) and the Markov side from the frozen config.
    """

    def __init__(self, n_frames, markov=None, readahead=None):
        self.ra = readahead or make_readahead(n_frames)
        self.mk = markov or make_predictor('hybrid')
        self.n_ra = self.n_mk = 0

    def update(self, history, actual_next):
        self.ra.update(history, actual_next)
        self.mk.update(history, actual_next)

    def predict(self, history, top_n=1):
        out = list(self.ra.predict(history, top_n))
        have = {p for p, _ in out}
        self.n_ra += len(out)
        for p, c in self.mk.predict(history, top_n):
            if p not in have:
                out.append((p, c))
                self.n_mk += 1
        return out

    def stats(self):
        return {'ra_candidates': self.n_ra, 'markov_candidates': self.n_mk,
                **{'ra_' + k: v for k, v in self.ra.stats().items()}}


def make_readahead(n_frames, min_run=2, conf_k=None):
    """Linux caps readahead at 128 KB (32 pages), larger than our whole simulated RAM,
    so scale the window to a quarter of memory."""
    mx = max(2, n_frames // 4)
    return ReadaheadPredictor(init_window=min(4, mx), max_window=mx, min_run=min_run, conf_k=conf_k)
