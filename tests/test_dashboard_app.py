"""
Integration tests for the dashboard UI using Streamlit's AppTest framework.
Runs the app in demo mode (no database) and simulates Prev/Next navigation.

Regression guard for the bug where chart-side buttons modified
st.session_state.sym_ticker after the selectbox was already instantiated
(StreamlitAPIException) — the click must switch the pair cleanly.
"""
import os

import pytest

os.environ.setdefault("DASHBOARD_DEMO", "1")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_FILE = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")


@pytest.fixture(scope="module")
def app_test():
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    assert not at.exception, f"Dashboard failed to render: {[e.value for e in at.exception]}"
    return at


def _pair(at):
    box = at.selectbox(key="sym_ticker")
    return box.value, list(box.options)


def test_dashboard_boots_in_demo_mode(app_test):
    value, options = _pair(app_test)
    assert value in options
    assert len(options) > 0


def test_chart_side_next_button_switches_pair(app_test):
    before, options = _pair(app_test)
    expected = options[(options.index(before) + 1) % len(options)]

    app_test.button(key="nav_next_pair").click()
    app_test.run()

    assert not app_test.exception, f"Next button raised: {[e.value for e in app_test.exception]}"
    after, _ = _pair(app_test)
    assert after == expected


def test_chart_side_prev_button_switches_pair(app_test):
    before, options = _pair(app_test)
    expected = options[(options.index(before) - 1) % len(options)]

    app_test.button(key="nav_prev_pair").click()
    app_test.run()

    assert not app_test.exception, f"Prev button raised: {[e.value for e in app_test.exception]}"
    after, _ = _pair(app_test)
    assert after == expected


def test_top_row_buttons_switch_pair(app_test):
    before, options = _pair(app_test)

    app_test.button(key="sel_next").click()
    app_test.run()

    assert not app_test.exception
    after, _ = _pair(app_test)
    assert after == options[(options.index(before) + 1) % len(options)]


def test_volume_toggle_renders_without_errors(app_test):
    """Volume bars are hidden by default; enabling the toggle must not break rendering."""
    app_test.checkbox(key="show_volume").check()
    app_test.run()
    assert not app_test.exception, f"Volume toggle raised: {[e.value for e in app_test.exception]}"

    app_test.checkbox(key="show_volume").uncheck()
    app_test.run()
    assert not app_test.exception


def test_stacked_layout_toggle_and_nav(app_test):
    """'Large stacked' toggle switches to 15m-top/1D-bottom with per-chart nav."""
    app_test.toggle(key="stacked_layout").set_value(True)
    app_test.run()
    assert not app_test.exception, f"Stacked layout raised: {[e.value for e in app_test.exception]}"

    before, options = _pair(app_test)
    expected = options[(options.index(before) + 1) % len(options)]
    app_test.button(key="nav_next_15m").click()
    app_test.run()
    assert not app_test.exception
    after, _ = _pair(app_test)
    assert after == expected


def test_only_with_15m_toggle_defaults_off_and_is_safe(app_test):
    """Chart options checkbox 'Only pairs with 15m data': OFF by default;
    toggling it must not break rendering (demo pairs exist on both TFs,
    so the pair list itself is unchanged here)."""
    box = app_test.checkbox(key="only_with_15m")
    assert box.value is False

    before, options_before = _pair(app_test)
    box.check()
    app_test.run()
    assert not app_test.exception, f"only_with_15m ON raised: {[e.value for e in app_test.exception]}"
    after, options_after = _pair(app_test)
    assert len(options_after) == len(options_before) > 0  # demo: both TFs exist
    assert after == before  # same pair stays selected

    app_test.checkbox(key="only_with_15m").uncheck()
    app_test.run()
    assert not app_test.exception


def test_summary_row_for_table_pads_missing_ob_columns():
    """Regression for 'No 15m table for PIXEL/USDT:USDT': the summary scan
    used a fixed column list including ob_* snapshot columns; tables whose
    snapshot never landed raised UndefinedColumn and were silently dropped.
    The scan now SELECTs * and pads missing keys with None."""
    import asyncio

    from dashboard.app import _EXPECTED_SUMMARY_KEYS, _summary_row_for_table

    class _Conn:
        async def fetchrow(self, query):
            assert "SELECT *" in query  # never a fixed ob_* column list again
            # Table WITHOUT any ob_* columns — only base candle data:
            return {
                "Timestamp": 1_787_700_000, "open": 1.0, "high": 2.0,
                "low": 0.5, "close": 1.5, "volume": 42.0,
                "ticker": "PIXEL/USDT:USDT", "exchange": "bybit",
                "asset_type": "swap",
            }

    class _Pool:
        def acquire(self):
            conn = _Conn()

            class _A:
                async def __aenter__(self):
                    return conn

                async def __aexit__(self, *exc):
                    return False

            return _A()

    errors = []
    row = asyncio.run(
        _summary_row_for_table(_Pool(), "pixel_usdt:usdt_on_bybit", asyncio.Semaphore(1), errors)
    )
    assert errors == []
    assert row is not None
    assert row["ticker"] == "PIXEL/USDT:USDT"
    assert row["max_ts"] == 1_787_700_000
    for key in _EXPECTED_SUMMARY_KEYS:
        assert key in row
    assert row["ob_vitality_score"] is None  # padded, not missing


def test_summary_row_for_table_reports_broken_table():
    """A genuinely broken table returns None but is REPORTED into errors
    (never silently skipped)."""
    import asyncio

    from dashboard.app import _summary_row_for_table

    class _Conn:
        async def fetchrow(self, query):
            raise RuntimeError("relation is corrupted")

    class _Pool:
        def acquire(self):
            conn = _Conn()

            class _A:
                async def __aenter__(self):
                    return conn

                async def __aexit__(self, *exc):
                    return False

            return _A()

    errors = []
    row = asyncio.run(_summary_row_for_table(_Pool(), "broken_tbl", asyncio.Semaphore(1), errors))
    assert row is None
    assert errors and errors[0][0] == "broken_tbl"


# ---------------------------------------------------------------------------
# /candles endpoint parameter guard — the perp-history regression.
# Table names are symbol.replace('/', '_')_on_<exchange>.lower(), so perp
# symbols carry ':' (PIXEL/USDT:USDT -> pixel_usdt:usdt_on_bybit). The old
# charset [A-Za-z0-9_] rejected every perp table, the endpoint answered
# {"c": []} and the chart's history loader read that as 'start of history':
# perp charts (both 15m and 1D, every exchange) silently stopped paging older
# chunks while spot scrolled infinitely.
# ---------------------------------------------------------------------------


def test_candles_guard_accepts_perp_and_spot_tables(monkeypatch):
    from dashboard import app as dapp

    monkeypatch.setattr(dapp, "_KNOWN_DBS", {"db1"}, raising=False)
    ts = 1756108800
    assert dapp._candles_query_ok("db1", "pixel_usdt:usdt_on_bybit", ts)
    assert dapp._candles_query_ok("db1", "btc_usdt_on_bybit", ts)
    assert dapp._candles_query_ok("db1", "1000sats_usdt:usdt_on_okx", ts)


def test_candles_guard_still_rejects_injection(monkeypatch):
    from dashboard import app as dapp

    monkeypatch.setattr(dapp, "_KNOWN_DBS", {"db1"}, raising=False)
    ts = 1756108800
    # '"' must stay banned — the name is interpolated as a quoted identifier
    assert not dapp._candles_query_ok("db1", 'x"evil', ts)
    assert not dapp._candles_query_ok("db1", 'btc_usdt_on_bybit" OR 1=1 --', ts)
    assert not dapp._candles_query_ok("db1", "tbl; DROP TABLE x", ts)
    # structural checks
    assert not dapp._candles_query_ok("db1", "no_exchange_suffix", ts)  # no _on_
    assert not dapp._candles_query_ok("unknown_db", "btc_usdt_on_bybit", ts)
    assert not dapp._candles_query_ok("db1", "btc_usdt_on_bybit", 0)
    assert not dapp._candles_query_ok("db1", "", ts)
    assert not dapp._candles_query_ok("db1", "a" * 97 + "_on_bybit", ts)  # too long


def test_history_loader_url_roundtrips_colon_table():
    """build_history_loader_js must produce a /candles URL whose table param
    decodes back to the exact perp table name (with ':')."""
    import json
    import re
    from urllib.parse import parse_qs, urlparse

    from dashboard.helpers import build_history_loader_js

    js = build_history_loader_js("db1", "pixel_usdt:usdt_on_bybit", 900, 8511, chunk=1200)
    m = re.search(r'const BASE = (".*?");', js)
    assert m, "history loader JS lost its BASE url"
    url = json.loads(m.group(1))
    qs = parse_qs(urlparse(url).query)
    assert qs["table"] == ["pixel_usdt:usdt_on_bybit"]
    assert qs["db"] == ["db1"]


def test_exchange_filter_defaults_include_gateio(app_test):
    """The sidebar 🌐 Exchanges filter must default to Bybit + Gate.io + OKX + MEXC
    (gateio added by user request — previously the default trio dropped it)."""
    ms = [m for m in app_test.multiselect if m.label == "🌐 Exchanges"]
    assert len(ms) == 1, "expected exactly one exchange multiselect in the sidebar"
    assert set(ms[0].value) >= {"bybit", "gateio", "okx", "mexc"}


def test_spread_history_panel_wired():
    """The under-charts metric section must include the orderbook spread line
    (ob_spread_pct) for every pair, next to the perp-only OI/funding panels."""
    with open(APP_FILE, encoding="utf-8") as fh:
        src = fh.read()
    assert '"ob_spread_pct"' in src
    assert "Spread History" in src
    assert "🟠 Spread %" in src
