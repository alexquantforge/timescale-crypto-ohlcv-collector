"""
Global progress counter and ETA calculation across all exchanges.
"""
import time
from typing import Tuple


def fmt_eta(seconds: float) -> str:
    """Formats seconds into a human ETA: '1h05m' / '22m13s'."""
    try:
        s = int(max(0, round(seconds)))
    except Exception:
        return "?"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    # Unit suffixes on purpose: the bare "22:13" was routinely misread as a
    # wall-clock time ("last updated at 22:13") instead of "22 min 13 s left".
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s"



class GlobalProgress:
    """
    Unified tracker across all concurrent exchange tasks for progress & ETA.
    """

    def __init__(self, total: int = 0):
        self.total = total
        self.done = 0
        self.start_ts = time.time()
        self.prefilled = total > 0

    def reset(self, total: int = 0) -> None:
        self.total = total
        self.done = 0
        self.start_ts = time.time()
        self.prefilled = total > 0

    def start_timing(self) -> None:
        """Resets start_ts right when actual symbol processing begins."""
        self.start_ts = time.time()

    def add_to_total(self, n: int) -> None:
        if not self.prefilled:
            self.total += int(n)

    def subtract_from_total(self, n: int) -> None:
        """Adjusts total if an exchange fails or is skipped."""
        self.total = max(self.done, self.total - int(n))

    def tick(self) -> Tuple[int, int, str, float]:
        """Increments processed count and returns (done, total, eta_str, percentage)."""
        self.done += 1
        done = self.done
        total = max(self.total, done)
        elapsed = time.time() - self.start_ts

        if done > 5 and total > done and elapsed > 0:
            rate = done / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            eta_str = fmt_eta(remaining)
        elif done >= total:
            eta_str = "0:00"
        else:
            eta_str = "?"

        pct = (done / total * 100.0) if total > 0 else 0.0
        return done, total, eta_str, pct
