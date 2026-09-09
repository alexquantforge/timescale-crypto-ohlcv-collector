"""
Heap-return helpers for long-running collector engines.

Why this module exists and what it fixes
----------------------------------------
Python's allocator ratchets RSS to the peak of a cycle and never returns it on
its own: the freed pymalloc arenas and the glibc top-of-heap pages sit there
after every cycle, and with TWO engines (1D + 15M) running for hours that is
exactly what slowly fills RAM and then swap until the machine thrashes.

The engines already called a one-shot `release_memory()` at the END of a cycle
(gc.collect + malloc_trim). That is not enough, because:

* a 15m cycle processes thousands of pairs over tens of minutes, and the freed
  pages from the first half are held until the cycle ends — the RSS peak is
  reached long before the trim runs;
* between cycles the engine *sleeps* (5 min for 15M, 1 h for 1D) but is still
  alive; nothing returns the working set during that window;
* malloc_trim only shrinks the top of the heap, so the longer between trims the
  more fragmentation there is and the less it can give back.

So `release_memory()` is now also driven by a periodic background task
(`memory_release_loop`) that hands freed pages back to the OS every
`MEMORY_RELEASE_INTERVAL_SEC` seconds, both during a long cycle and during the
idle sleep. The `M_TRIM_THRESHOLD` tweak makes glibc trim promptly as it frees
instead of waiting for a 128 KB top-of-heap byte count, which is what lets a
"many small objects" workload keep returning memory without ballooning.

Both engines use this one helper so the two never drift apart. The functions
are defensive no-ops on any OS that is not glibc (macOS, Windows, BSD).
"""

import asyncio
import gc
import logging

logger = logging.getLogger("memory")

# How often to hand freed pages back while an engine runs (seconds). 300 keeps a
# single cycle's RSS from ratcheting, without paying a full gc on every tick.
MEMORY_RELEASE_INTERVAL_SEC = 300.0

_MALLOC_TRIMMED = False


def _libc():
    """`ctypes.CDLL("libc.so.6")`, or None when not on glibc (safe no-op)."""
    try:
        import ctypes

        return ctypes.CDLL("libc.so.6")
    except Exception:
        return None


def setup_malloc_trim() -> None:
    """Lower glibc's M_TRIM_THRESHOLD so freed memory is returned promptly.

    The default threshold (128 KB) means a workload that frees many small
    objects never triggers a trim and keeps a fat heap long after its peak.
    Setting it once per process (idempotent) lets free()/malloc_trim hand the
    pages back as they are freed rather than only when enough accumulates at
    the top of the heap.
    """
    global _MALLOC_TRIMMED
    if _MALLOC_TRIMMED:
        return
    libc = _libc()
    if libc is None:
        return
    try:
        # M_TRIM_THRESHOLD = 2. A small value (~0) makes glibc trim eagerly.
        libc.mallopt(2, 0)
        _MALLOC_TRIMMED = True
    except Exception:
        pass


def release_memory() -> None:
    """Run a full gc and return freed heap pages to the OS. Safe no-op elsewhere.

    Not just the default `gc.collect()`: collecting all generations is what
    breaks reference cycles the pymalloc arenas are holding, and malloc_trim(0)
    is what hands the freed pages back rather than keeping them for reuse in an
    allocator that no longer needs them.
    """
    try:
        gc.collect()
    except BaseException:
        pass
    libc = _libc()
    if libc is None:
        return
    try:
        libc.malloc_trim(0)
    except BaseException:
        pass


async def memory_release_loop(interval: float = MEMORY_RELEASE_INTERVAL_SEC) -> None:
    """Run `release_memory()` periodically for the life of the process.

    Run next to the priority lane as a lightweight asyncio task: it sleeps for
    `interval`, hands freed pages back (in a worker thread so it never blocks the
    event loop mid-cycle), and repeats. It is what keeps a multi-hour run from
    climbing into swap — not the one-shot call at end-of-cycle.
    """
    setup_malloc_trim()
    try:
        while True:
            await asyncio.sleep(max(30.0, float(interval)))
            try:
                await asyncio.to_thread(release_memory)
            except Exception:
                pass
    except asyncio.CancelledError:
        pass
