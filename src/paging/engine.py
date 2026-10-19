"""
Stage 4 — memory engine and eviction policies.

Architectural rule: `_install()` is the ONLY path into RAM. Prefetching goes
through it exactly like a demand fault, so a broken predictor can make the
system slow but never incorrect.
"""
import bisect
import heapq
from collections import OrderedDict, deque, defaultdict


class MemoryEngine:
    """Page table + frame table + fault handler. Knows nothing about policy."""

    def __init__(self, n_frames, policy):
        self.n_frames = n_frames
        self.policy = policy
        self.resident = set()
        self.log = None            # optional event list for the time model (Phase 15): ('F'|'P'|'U', i, page)
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
            mark = getattr(self.policy, 'mark_prefetch', None)    # optional hook (Phase 9)
            if mark is not None:
                mark(page, i)

    def access(self, page, i):
        """Demand access. Returns True if it was a fault."""
        self.n_access += 1
        hit = page in self.resident
        if self.prefetched.pop(page, False):
            self.n_pf_used += 1
            if self.log is not None:
                self.log.append(('U', i, page))
        if hit:
            self.policy.on_hit(page, i)
            return False
        self.n_fault += 1
        if self.log is not None:
            self.log.append(('F', i, page))
        self._install(page, i, is_prefetch=False)
        return True

    def prefetch(self, page, i):
        """Speculative load. Never bypasses the fault handler."""
        if page in self.resident:
            return False
        if self.log is not None:
            self.log.append(('P', i, page))
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


class GhostMemoryEngine(MemoryEngine):
    """
    Same behaviour as MemoryEngine; additionally remembers pages that were evicted
    to make room for a prefetch (a small 'ghost list') and counts how many fault back
    in soon afterwards. That count estimates the harm prefetching causes, with no
    shadow copy of the system.
    """

    def __init__(self, n_frames, policy, ghost_size=256, refault_window=2000):
        super().__init__(n_frames, policy)
        self.ghost = OrderedDict()           # page -> access index when a prefetch evicted it
        self.ghost_size = ghost_size
        self.refault_window = refault_window
        self.n_pf_harm = 0

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
            if is_prefetch:
                self.ghost[victim] = i
                self.ghost.move_to_end(victim)
                if len(self.ghost) > self.ghost_size:
                    self.ghost.popitem(last=False)
        self.resident.add(page)
        self.policy.on_insert(page, i)
        if is_prefetch:
            self.prefetched[page] = True
            self.n_prefetch += 1
            mark = getattr(self.policy, 'mark_prefetch', None)    # optional hook (Phase 9)
            if mark is not None:
                mark(page, i)

    def access(self, page, i):
        self.n_access += 1
        hit = page in self.resident
        if self.prefetched.pop(page, False):
            self.n_pf_used += 1
            if self.log is not None:
                self.log.append(('U', i, page))
        if hit:
            self.policy.on_hit(page, i)
            return False
        self.n_fault += 1
        if self.log is not None:
            self.log.append(('F', i, page))
        t = self.ghost.pop(page, None)
        if t is not None and i - t <= self.refault_window:
            self.n_pf_harm += 1
        self._install(page, i, is_prefetch=False)
        return True


class ARC:
    """
    Adaptive Replacement Cache (Megiddo & Modha, 2003), as an eviction policy.
      T1 = pages seen once recently      T2 = pages seen at least twice
      B1, B2 = 'ghost' lists: pages recently evicted from T1 / T2 (IDs only)
      p  = target size of T1; a hit in B1 grows p (recency matters), a hit in B2 shrinks it

    Two simplifications, because MemoryEngine installs pages itself and only asks the policy
    for a victim:
      1. the engine does not say which page is about to be installed, so the textbook
         tie-break (|T1| == p and the new page is in B2) is dropped
      2. p is adapted when the page is installed, i.e. just after the victim was chosen
    """

    def __init__(self, **kw):
        self.T1, self.T2 = OrderedDict(), OrderedDict()
        self.B1, self.B2 = OrderedDict(), OrderedDict()
        self.p = 0.0
        self.c = None                    # capacity, learned the first time RAM is full
        self.n_ghost_hits = 0

    def _trim(self):
        c = self.c
        if c is None:
            return
        while len(self.T1) + len(self.B1) > c and self.B1:
            self.B1.popitem(last=False)
        while (len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2)) > 2 * c:
            if self.B2:
                self.B2.popitem(last=False)
            elif self.B1:
                self.B1.popitem(last=False)
            else:
                break

    def on_insert(self, page, i):
        if page in self.B1:              # recently evicted from T1 -> favour recency
            self.p = min(float(self.c), self.p + max(1.0, len(self.B2) / len(self.B1)))
            del self.B1[page]
            self.T2[page] = i
            self.n_ghost_hits += 1
        elif page in self.B2:            # recently evicted from T2 -> favour frequency
            self.p = max(0.0, self.p - max(1.0, len(self.B1) / len(self.B2)))
            del self.B2[page]
            self.T2[page] = i
            self.n_ghost_hits += 1
        else:
            self.T1[page] = i
        self._trim()

    def on_hit(self, page, i):
        if page in self.T1:              # second touch: promote to the frequency list
            del self.T1[page]
            self.T2[page] = i
        elif page in self.T2:
            self.T2.move_to_end(page)

    def on_evict(self, page):
        if page in self.T1:
            del self.T1[page]
            self.B1[page] = 1
        elif page in self.T2:
            del self.T2[page]
            self.B2[page] = 1
        self._trim()

    def choose_victim(self, resident, i):
        if self.c is None:
            self.c = len(resident)
        if self.T1 and (len(self.T1) > self.p or not self.T2):
            return next(iter(self.T1))   # least recently used page of T1
        if self.T2:
            return next(iter(self.T2))   # least recently used page of T2
        return next(iter(resident))


class LRU2:
    """
    LRU-2 (O'Neil et al., 1993): evict the page whose SECOND most recent access is oldest.
    A page seen only once has no second access, so it is evicted first (oldest first among
    those), which makes it resistant to one-time scans. History is retained for recently
    evicted pages so a page that returns quickly counts as re-referenced.
    Simplification: no 'correlated reference period' (bursts of touches count as re-use).
    Heap with lazy deletion, so each eviction is O(log n).
    Known weakness (measured on the synthetic 'clustered' trace): slow to adapt when the
    working set jumps to brand-new pages.
    """

    def __init__(self, retain=1.0, **kw):
        self.hist = {}                   # page -> (second_last_access, last_access); -1 = none
        self.heap = []                   # entries (second_last, last, page)
        self.ghost = OrderedDict()       # evicted pages whose history is still retained
        self.retain = retain             # retained history = retain x capacity pages
        self.c = None

    def on_insert(self, page, i):
        old = self.hist.get(page)
        self.ghost.pop(page, None)
        prev = old[1] if old else -1
        self.hist[page] = (prev, i)
        heapq.heappush(self.heap, (prev, i, page))

    def on_hit(self, page, i):
        _, last = self.hist[page]
        self.hist[page] = (last, i)
        heapq.heappush(self.heap, (last, i, page))

    def on_evict(self, page):
        self.ghost[page] = 1
        limit = max(1, int(self.retain * (self.c or 1)))
        while len(self.ghost) > limit:
            old, _ = self.ghost.popitem(last=False)
            self.hist.pop(old, None)

    def choose_victim(self, resident, i):
        if self.c is None:
            self.c = len(resident)
        if len(self.heap) > 10 * len(resident) + 10_000:          # compact stale entries
            self.heap = [(self.hist[p][0], self.hist[p][1], p) for p in resident]
            heapq.heapify(self.heap)
        while self.heap:
            prev, last, p = heapq.heappop(self.heap)
            if p in resident and self.hist.get(p) == (prev, last):
                return p
        return next(iter(resident))


class PredictiveLRU:
    """
    Phase 9 prefetch-aware LRU. Prefetched pages that have not been used yet wait in a
    separate queue `pf` (oldest first); a hit promotes the page to the normal LRU list.
    Victim choice:
      stale prefetch  : the oldest unused prefetch, if it has waited longer than `grace`
                        accesses, goes first (it was a wrong guess)
      otherwise       : plain LRU among demand pages; if there are none, the oldest prefetch
    enabled=False makes mark_prefetch a no-op, so this class is then exactly LRU.
    Simplification: protection depth (never evict the predicted page) is not included yet.
    """

    def __init__(self, grace=64, enabled=True, **kw):
        self.od, self.pf = OrderedDict(), OrderedDict()
        self.grace, self.enabled = grace, enabled
        self.n_stale = 0
        self.protected = set()            # pages the predictor expects next; set by the simulator wrapper

    def on_insert(self, page, i):
        self.od[page] = i
        self.od.move_to_end(page)

    def mark_prefetch(self, page, i):
        if self.enabled and page in self.od:
            del self.od[page]
            self.pf[page] = i

    def on_hit(self, page, i):
        if page in self.pf:                      # a prefetch was used: promote it
            del self.pf[page]
        self.od[page] = i
        self.od.move_to_end(page)

    def on_evict(self, page):
        self.od.pop(page, None)
        self.pf.pop(page, None)

    def choose_victim(self, resident, i):
        if self.pf:
            page, t = next(iter(self.pf.items()))
            if i - t > self.grace or not self.od:
                if i - t > self.grace:
                    self.n_stale += 1
                return page
        for p in self.od:
            if p in resident and p not in self.protected:
                return p
        for p in self.od:
            if p in resident:
                return p
        return next(iter(resident))


# all six eviction policies used from Phase 8 onward (POLICIES above is the original four)
POLICIES_EXTENDED = {'FIFO': FIFO, 'Clock': Clock, 'LRU': LRU, 'LRU-2': LRU2,
                     'ARC': ARC, 'OPT': OPTFast}
