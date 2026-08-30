"""Circular-import regression tests.

Bug: `src/analytics/orderbook.py` imported `hard_wait_for` from
`src.core.timeouts`. Importing ANY submodule of a package runs the
package `__init__` first — and `src/core/__init__.py` eagerly imports
`src.core.updater`, which itself imports `src.analytics.orderbook`.
When the dashboard's import order hit `src.analytics` first, the chain
became: analytics/orderbook -> src.core.timeouts -> src.core.__init__
-> updater -> analytics/orderbook (partially initialized) -> ImportError.

Fix: `hard_wait_for` lives in `src/utils/timeouts.py` — a leaf package
that must never import from other src.* packages. These tests pin both
entry orders.
"""


def test_analytics_first_import_order():
    """The failing path from the dashboard: analytics must import standalone."""
    import src.analytics  # noqa: F401
    from src.analytics.orderbook import fetch_orderbook_snapshot  # noqa: F401
    from src.utils.timeouts import hard_wait_for  # noqa: F401

    assert callable(fetch_orderbook_snapshot)
    assert callable(hard_wait_for)


def test_gap_filler_first_import_order():
    from src.exchanges.gap_filler import (  # noqa: F401
        fetch_ohlcv_catch_up,
        fill_history_gaps,
    )

    assert callable(fill_history_gaps)
    assert callable(fetch_ohlcv_catch_up)


def test_core_engine_import_still_works():
    from src.core.updater import (  # noqa: F401
        MarketDataEngine,
        format_await_chain,
    )

    assert MarketDataEngine is not None
    assert callable(format_await_chain)
