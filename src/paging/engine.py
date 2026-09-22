"""
Stage 4 — memory engine and eviction policies.

Architectural rule: `_install()` is the ONLY path into RAM. Prefetching goes
through it exactly like a demand fault, so a broken predictor can make the
system slow but never incorrect.
"""
import bisect
from collections import OrderedDict, deque, defaultdict


class MemoryEngine:
    """Page table + frame table + fault handler. Knows nothing about policy."""

    def __init__(self, n_frames, policy):
        self.n_frames = n_frames
        self.policy = policy
        self.resident = set()
        self.reset_stats()

    def reset_stats(self):
        self.n_access = 0
        self.n_fault = 0
        self.n_evict = 0
        self.prefetched = {}       # page -> still-unused flag
        self.n_prefetch = 0
        self.n_pf_used = 0
        self.n_pf_wasted = 0

    def _install(self, page, i, is_prefetch=False):
        if page in self.resident:
            return
        if len(self.resident) >= self.n_frames:
            victim = self.policy.choose_victim(self.resident, i)
            self.resident.discard(victim)
            self.policy.on_evict(victim)
            self.n_evict += 1
            if self.prefetched.pop(victim, False):
                self.n_pf_wasted += 1
        self.resident.add(page)
        self.policy.on_insert(page, i)
        if is_prefetch:
            self.prefetched[page] = True
            self.n_prefetch += 1

    def access(self, page, i):
        """Demand access. Returns True if it was a fault."""
        self.n_access += 1
        hit = page in self.resident
        if self.prefetched.pop(page, False):
            self.n_pf_used += 1
        if hit:
            self.policy.on_hit(page, i)
            return False
        self.n_fault += 1
        self._install(page, i, is_prefetch=False)
        return True

    def prefetch(self, page, i):
        """Speculative load. Never bypasses the fault handler."""
        if page in self.resident:
            return False
        self._install(page, i, is_prefetch=True)
        return True

    def stats(self):
        return {
            'accesses':     self.n_access,
            'faults':       self.n_fault,
            'fault_rate':   self.n_fault / max(1, self.n_access),
            'evictions':    self.n_evict,
            'prefetches':   self.n_prefetch,
            'pf_used':      self.n_pf_used,
            'pf_wasted':    self.n_pf_wasted,
            'pf_precision': self.n_pf_used / max(1, self.n_prefetch),
        }


class FIFO:
    def __init__(self, **kw):
        self.q = deque()

    def on_insert(self, page, i): self.q.append(page)
    def on_hit(self, page, i): pass
    def on_evict(self, page): pass

    def choose_victim(self, resident, i):
        while self.q:
            p = self.q.popleft()
            if p in resident:
                return p
        return next(iter(resident))


class LRU:
    def __init__(self, **kw):
        self.od = OrderedDict()

    def on_insert(self, page, i):
        self.od[page] = i
        self.od.move_to_end(page)

    def on_hit(self, page, i):
        self.od[page] = i
        self.od.move_to_end(page)

    def on_evict(self, page):
        self.od.pop(page, None)

    def choose_victim(self, resident, i):
        for p in self.od:
            if p in resident:
                return p
        return next(iter(resident))


class Clock:
    """Second-chance approximation of LRU."""

    def __init__(self, **kw):
        self.ring, self.ref, self.hand = [], {}, 0

    def on_insert(self, page, i):
        self.ring.append(page)
        self.ref[page] = 1

    def on_hit(self, page, i):
        self.ref[page] = 1

    def on_evict(self, page):
        if page in self.ring:
            idx = self.ring.index(page)
            self.ring.pop(idx)
            self.ref.pop(page, None)
            if self.hand > idx:
                self.hand -= 1

    def choose_victim(self, resident, i):
        while True:
            if not self.ring:
                return next(iter(resident))
            self.hand %= len(self.ring)
            p = self.ring[self.hand]
            if p not in resident:
                self.ring.pop(self.hand)
                self.ref.pop(p, None)
                continue
            if self.ref.get(p, 0) == 0:
                return p
            self.ref[p] = 0
            self.hand += 1


class OPTFast:
    """
    Belady's optimal. Evicts the page whose next use is furthest away.
    ALLOWED to see the future — it is the theoretical lower bound.
    """

    def __init__(self, pages=None, **kw):
        self.occ = defaultdict(list)
        for i, p in enumerate(pages):
            self.occ[int(p)].append(i)

    def on_insert(self, page, i): pass
    def on_hit(self, page, i): pass
    def on_evict(self, page): pass

    def choose_victim(self, resident, i):
        best, best_d = None, -1
        for p in resident:
            lst = self.occ.get(p, [])
            k = bisect.bisect_right(lst, i)
            d = lst[k] if k < len(lst) else float('inf')
            if d > best_d:
                best, best_d = p, d
                if d == float('inf'):
                    break
        return best


POLICIES = {'FIFO': FIFO, 'LRU': LRU, 'Clock': Clock, 'OPT': OPTFast}


def run_policy(trace, n_frames, policy_cls):
    """Replay a trace through one policy with no prefetching."""
    eng = MemoryEngine(n_frames, policy_cls(pages=trace.pages))
    for i, page, _ in trace.stream():
        eng.access(page, i)
    return eng.stats()
