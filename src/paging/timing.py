"""
Phase 15 — time model. Replays the event log of one simulation run on an I/O timeline.

Events come from MemoryEngine.log: ('F', i, page) demand fault, ('P', i, page) prefetch issued,
('U', i, page) first use of a prefetched page. The model:
  - every access costs `t_cpu` of computation; every access that asks the predictor costs `t_pred`
  - the disk has `channels` identical servers; each read takes `t_io`; reads are served in issue
    order on the earliest free channel (no priority for demand reads: pessimistic for prefetching)
  - a demand fault blocks until its read completes (it may queue behind prefetch reads)
  - a prefetch does not block; if its page is used before the read completes, the program
    stalls for the rest (a late prefetch)
Simplifications to state: one outstanding demand fault at a time, no write-back of dirty pages,
no I/O merging, eviction costs nothing.
"""
import heapq

# Stated, not tuned. t_io: typical 4 KB random read; t_pred: one hash lookup + counter update.
DEVICES = {
    'NVMe SSD': dict(t_io=80e-6, channels=8),
    'HDD':      dict(t_io=5e-3, channels=1),
}
T_PRED = 0.2e-6
T_CPU_SWEEP = (0.1e-6, 1e-6, 10e-6)     # compute between two traced accesses: unknown, so swept


def replay(log, n_access, t_io, channels, t_cpu, t_pred=0.0):
    """Return a dict with total time and its parts (seconds)."""
    free = [0.0] * channels                 # channel free times (min-heap)
    heapq.heapify(free)
    ready = {}                              # page -> time its prefetch read completes
    t = 0.0
    stall_fault = stall_late = 0.0
    k, n = 0, len(log)
    for i in range(n_access):
        t += t_cpu + t_pred
        while k < n and log[k][1] == i:
            kind, _, page = log[k]
            k += 1
            if kind == 'U':                 # recorded at access time, before any fault of the same access
                r = ready.pop(page, 0.0)
                if r > t:
                    stall_late += r - t
                    t = r
            elif kind == 'F':
                start = max(t, heapq.heappop(free))
                done = start + t_io
                heapq.heappush(free, done)
                stall_fault += done - t
                t = done
            else:                           # 'P': issued after this access was served
                start = max(t, heapq.heappop(free))
                done = start + t_io
                heapq.heappush(free, done)
                ready[page] = done
    return {'time': t, 'stall_fault': stall_fault, 'stall_late': stall_late,
            'compute': n_access * t_cpu, 'predict': n_access * t_pred}
