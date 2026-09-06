"""Unit tests for the standalone `scan_growth.py` scanner (pure parts only).

The scanner reads the collector's stored tables and prints pairs that rose more
than X% from the lowest low of the last N days to the current close. These tests
pin the bits that do not need a live database: table-name -> ticker/exchange
inference, exchange-suffix resolution, and the SQL the per-table aggregate uses.
"""
import os

os.environ.setdefault("DASHBOARD_DEMO", "1")

import scan_growth as sg


def test_ticker_from_table_spot():
    """`btc_usdt_on_bybit` -> BTC/USDT @ bybit."""
    assert sg.ticker_from_table("btc_usdt_on_bybit") == ("BTC/USDT", "bybit", "bybit")


def test_ticker_from_table_perp():
    """`btc_usdt:usdt_on_gateio` -> BTC/USDT:USDT @ gateio (engine name -> gate ccxt)."""
    assert sg.ticker_from_table("btc_usdt:usdt_on_gateio") == ("BTC/USDT:USDT", "gateio", "gate")


def test_ticker_from_table_1d_ccxt_suffix():
    """1D tables write the ccxt id directly: `pixel_usdt_on_gate` -> ccxt 'gate',
    and it back-resolves to the engine label 'gateio' for link building."""
    assert sg.ticker_from_table("pixel_usdt:usdt_on_gate") == ("PIXEL/USDT:USDT", "gateio", "gate")


def test_ticker_from_table_long_base():
    """Bases with long numeric prefixes must not be truncated."""
    assert sg.ticker_from_table("1000000babydoge_usdt_on_bybit") == (
        "1000000BABYDOGE/USDT", "bybit", "bybit")


def test_ticker_from_table_garbage():
    """Names that do not look like `<pair>_on_<ex>` yield empty (never inject)."""
    assert sg.ticker_from_table("") == ("", "", "")
    assert sg.ticker_from_table("not_a_pair_table") == ("", "", "")     # no `_on_`
    assert sg.ticker_from_table("btc_usdt_on_") == ("", "", "")         # empty exchange
    assert sg.ticker_from_table('btc"on_bybit') == ("", "", "")         # bad chars
    assert sg.ticker_from_table("btc_usdt_on_bybit!") == ("", "", "")   # ex has !


def test_resolve_exchange_both_directions():
    """The 'gateio'/<ccxt 'gate'> gap: 1D uses `gate`, 15m uses `gateio`."""
    assert sg.resolve_exchange("gateio") == ("gateio", "gate")
    assert sg.resolve_exchange("gate") == ("gateio", "gate")
    assert sg.resolve_exchange("bybit") == ("bybit", "bybit")
    assert sg.resolve_exchange("okx") == ("okx", "okx")
    assert sg.resolve_exchange("unknown_ex") == ("unknown_ex", "unknown_ex")


def test_growth_sql_shape():
    """The aggregate query carries the seconds AND milliseconds cutoff, and a
    managed table name can never be injected (garbage -> '')."""
    sql = sg._growth_sql("btc_usdt_on_bybit", cutoff=1750000000)
    assert sql and '"close"' in sql and 'MIN("low")' in sql
    assert "100000000000" in sql                 # ms branch present
    assert f'"Timestamp" >= 1750000000' in sql   # seconds cutoff present
    # table names that fail the identifier check are refused outright
    assert sg._growth_sql("btc_usdt_on_bybit; DROP TABLE x", 1) == ""
    assert sg._growth_sql("btc/on_x", 1) == ""
    assert sg._growth_sql("", 1) == ""
