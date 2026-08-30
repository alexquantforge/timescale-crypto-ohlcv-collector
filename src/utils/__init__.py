"""
Small dependency-free utilities shared across layers.

IMPORTANT: modules here must NOT import from any other src.* package.
Heavy package __init__ files (e.g. src.core.__init__ pulling the engine)
turn any cross-layer import into a circular-import trap — that is exactly
why hard_wait_for lives here and not in src/core/timeouts.py.
"""
from src.utils.timeouts import hard_wait_for

__all__ = ["hard_wait_for"]
