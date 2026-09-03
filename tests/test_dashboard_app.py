"""
Integration tests for the dashboard UI using Streamlit's AppTest framework.
Runs the app in demo mode (no database) and simulates Prev/Next navigation.

Regression guard for the bug where chart-side buttons modified
st.session_state.sym_ticker after the selectbox was already instantiated
(StreamlitAPIException) — the click must switch the pair cleanly.
"""
import os
import time

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
        def acquire(self, *a, **kw):  # the scan passes acquire(timeout=...)
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
        def acquire(self, *a, **kw):  # the scan passes acquire(timeout=...)
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


# ---------------------------------------------------------------------------
# Pair flipping must never wait on the priority-pair publish
# ---------------------------------------------------------------------------

def _load_app_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("dashboard_app_mod", APP_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_rerun_does_not_forget_the_backoff(monkeypatch):
    """THE cause behind the six identical `[markets] ⚠️ gate: load_markets
    failed` lines in one minute.

    Streamlit creates a brand-new `__main__` module for every run, so a
    module-level `_X: dict = {}` is rebuilt on each rerun — which is harmless for
    a scratch variable and fatal for a rate limiter: every rerun "forgot" that
    gate had just failed and launched another market-catalog load, keeping the
    endpoint permanently timed out. Stores that exist to AVOID work therefore
    come from `st.cache_resource` (`_state`), which outlives the rerun.
    """
    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()
    app._MARKET_GATE["gate"] = {"fails": 3, "err": "RequestTimeout"}
    app._MARKET_LOG_AT[("load", "gate")] = 123.0

    again = _load_app_module()          # a fresh exec of the same file = a rerun
    assert again._MARKET_GATE.get("gate", {}).get("fails") == 3
    assert again._MARKET_LOG_AT.get(("load", "gate")) == 123.0
    assert again._MARKET_GATE is app._MARKET_GATE


def test_dashboard_exchanges_do_not_fetch_the_currency_table():
    """ccxt prepends `fetch_currencies()` to `load_markets()` when the exchange
    implements it — gate's `/spot/currencies` is the request that never finished
    in time here. Public candles/ticker/orderbook do not need it."""
    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()
    for ccxt_id in ("gate", "okx", "bybit"):
        ex = app._new_sync_exchange(ccxt_id)
        assert ex.has.get("fetchCurrencies") is False, ccxt_id
        assert ex.timeout == 8000


def test_publish_is_throttled_and_never_blocks(monkeypatch):
    """The publish is fire-and-forget and skipped when the set is unchanged:
    holding Prev/Next must not queue one database write per keystroke."""
    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()

    calls = []
    monkeypatch.setattr(
        app, "_live_infra_or_none", lambda: {"submit_publish": calls.append}
    )
    app._LAST_PUBLISH["ts"] = 0.0
    app._LAST_PUBLISH["key"] = None

    pairs = [{"db": "d", "ex": "bybit", "sym": "0G/USDT:USDT"}]
    app._publish_priority_pairs_async(pairs)
    app._publish_priority_pairs_async(pairs)        # same set, immediately -> skipped
    assert len(calls) == 1

    app._publish_priority_pairs_async(
        [{"db": "d", "ex": "bybit", "sym": "BTC/USDT:USDT"}]  # new pair -> published
    )
    assert len(calls) == 2


def test_a_failed_stitch_is_not_remembered_as_an_answer(monkeypatch):
    """The bug this guards: `st.cache_data(ttl=3600)` cached the `[]` that a
    failed fetch returned, so ONE hiccup left the chart gapped for an hour while
    the caption blamed the exchange for a dashboard-side failure. A failure must
    be visible to the caller and retried soon; a real answer stays cached."""
    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()

    class _Flaky:
        markets = {"X/USDT": {}}
        calls = 0

        def load_markets(self):
            return self.markets

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("connection reset")
            return [[since + i * 900_000, 1, 1, 1, 1, 1] for i in range(5)]

    monkeypatch.setattr(app, "_get_sync_exchange", lambda ccxt_id: _Flaky())
    monkeypatch.setattr(app, "_STITCH_CACHE", {}, raising=True)
    monkeypatch.setattr(app.settings, "dash_stitch_retry_sec", 0.0)

    errs = []
    assert app._fetch_missing_candles_cached(
        "bybit", "X/USDT", "15m", 10, 20, errors=errs) == []
    assert errs and "connection reset" in errs[0]     # the caller was told
    assert _Flaky.calls == 1

    # ...and the same range is asked again instead of staying empty for an hour.
    rows = app._fetch_missing_candles_cached("bybit", "X/USDT", "15m", 10, 20, errors=[])
    assert rows and _Flaky.calls > 1          # retried, and it filled this time

    # A real answer is cached: the same range is never asked again this hour.
    calls = _Flaky.calls
    again = app._fetch_missing_candles_cached("bybit", "X/USDT", "15m", 10, 20, errors=[])
    assert again == rows and _Flaky.calls == calls


def test_markets_are_loaded_once_per_exchange_and_then_reused(monkeypatch):
    """`if not ex.markets: ex.load_markets()` on a path that runs once per second
    per pair is a market-load storm (gate: 3 requests, slower than the 8 s
    instance timeout, so it ALWAYS failed and the chart could never fetch). One
    load per exchange, then nothing until the markets are gone."""
    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()
    calls = []

    class _Ex:
        markets = {}
        timeout = 8000

        def load_markets(self):
            calls.append(1)
            self.markets = {"A/B": {}}

    ex = _Ex()
    monkeypatch.setattr(app, "_MARKET_GATE", {}, raising=True)
    monkeypatch.setattr(app, "_MARKET_LOG_AT", {}, raising=True)
    monkeypatch.setattr(app, "_get_sync_exchange", lambda ccxt_id: ex)

    assert app.ensure_markets("gate", ex, wait_sec=5.0) == ""
    assert len(calls) == 1
    assert app.ensure_markets("gate", ex, wait_sec=0.0) == ""
    assert len(calls) == 1                      # and no lock dance at all now
    # a load is given more time than a fetch, and the tight timeout returns
    assert ex.timeout == 8000


def test_the_render_path_never_waits_for_a_market_load(monkeypatch):
    """The load may take 20 s; a pair flip may not. So the render path starts it
    in the background and is told, in words, that the data is not available yet."""
    import types as _types

    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()
    started = []

    class _T:
        def __init__(self, target=None, **kw):
            self._target = target

        def start(self):
            started.append(self._target)

    class _Ex:
        markets = {}
        timeout = 8000

        def load_markets(self):           # would block for 20 s in reality
            raise AssertionError("must not run inline on the render path")

    ex = _Ex()
    monkeypatch.setattr(app, "_MARKET_GATE", {}, raising=True)
    import threading as _real_threading
    monkeypatch.setattr(app, "threading", _types.SimpleNamespace(
        Thread=_T, Lock=_real_threading.Lock, Event=_real_threading.Event))
    monkeypatch.setattr(app, "_get_sync_exchange", lambda ccxt_id: ex)

    reason = app.ensure_markets("gate", ex, wait_sec=0.0)
    assert "background" in reason
    assert len(started) == 1                   # one loader, and the caller left

    # ...and a second caller while it is loading is not allowed to queue another.
    assert "loading elsewhere" in app.ensure_markets("gate", ex, wait_sec=0.0) or \
           "still loading" in app.ensure_markets("gate", ex, wait_sec=0.0)
    assert len(started) == 1


def test_a_failing_market_load_backs_off_instead_of_beating_the_exchange(monkeypatch):
    """The log the user pasted was 25 identical `RequestTimeout .../currencies`
    lines in a minute. A failure must cost ONE retry per backoff window."""
    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()

    class _Ex:
        markets = {}
        timeout = 8000
        loads = 0

        def load_markets(self):
            type(self).loads += 1
            raise RuntimeError("RequestTimeout: gate GET /spot/currencies")

    ex = _Ex()
    monkeypatch.setattr(app, "_MARKET_GATE", {}, raising=True)
    monkeypatch.setattr(app, "_MARKET_LOG_AT", {}, raising=True)
    monkeypatch.setattr(app, "_get_sync_exchange", lambda ccxt_id: ex)

    first = app.ensure_markets("gate", ex, wait_sec=5.0)
    assert "RequestTimeout" in first
    second = app.ensure_markets("gate", ex, wait_sec=5.0)
    assert "next try in" in second
    assert _Ex.loads == 1                      # the backoff ate the second attempt
    assert app._MARKET_GATE["gate"]["fails"] == 1


def test_a_stitch_without_markets_says_so_instead_of_claiming_no_candles(monkeypatch):
    """"The exchange returned nothing" and "we never got to ask" are different
    facts; the caption used to print the first one for both."""
    import types as _types

    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()

    class _T:
        def __init__(self, target=None, **kw):
            self._target = target

        def start(self):
            pass

    class _Ex:
        markets = {}
        timeout = 8000
        fetched = 0

        def load_markets(self):
            raise RuntimeError("markets unavailable")

        def fetch_ohlcv(self, *a, **k):
            type(self).fetched += 1
            return []

    ex = _Ex()
    monkeypatch.setattr(app, "_MARKET_GATE", {}, raising=True)
    monkeypatch.setattr(app, "_MARKET_LOG_AT", {}, raising=True)
    monkeypatch.setattr(app, "_STITCH_CACHE", {}, raising=True)
    import threading as _real_threading
    monkeypatch.setattr(app, "threading", _types.SimpleNamespace(
        Thread=_T, Lock=_real_threading.Lock, Event=_real_threading.Event))
    monkeypatch.setattr(app, "_get_sync_exchange", lambda ccxt_id: ex)

    errs = []
    got = app._fetch_missing_candles_cached(
        "gate", "0G/USDT:USDT", "15m", 10, 20, errors=errs)
    assert got == [] and errs and "background" in errs[0]
    assert _Ex.fetched == 0                    # not one OHLCV request burned
    # and the retry window for that exchange is wider than for a single range
    assert app._STITCH_CACHE[("gate", "0G/USDT:USDT", "15m", 10, 20)][3] >= 30.0


def test_the_chart_names_a_dead_collector_instead_of_a_stitch_bug(monkeypatch):
    """The missing half of "the dashboard wakes the updater": if no engine
    answers the wake-up, the chart has to SAY that the hole will not be written
    back — otherwise a stale collector and a broken stitch are indistinguishable."""
    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()

    monkeypatch.setattr(app, "_LANE_PULSE", {"at": 1.0, "watched": 11, "served": 0,
                                             "served_15m": 0, "idle_15m": 9000.0,
                                             "served_1d": 0, "idle_1d": 9000.0},
                        raising=True)
    txt = app._lane_pulse_note("15m")
    assert "no 15m engine" in txt and "2.5h" in txt and "--timeframe 15m" in txt

    monkeypatch.setattr(app, "_LANE_PULSE", {"at": 1.0, "watched": 11, "served": 3,
                                             "served_15m": 3, "idle_15m": 4.0,
                                             "served_1d": 0, "idle_1d": 9000.0},
                        raising=True)
    assert "15m lane alive" in app._lane_pulse_note("15m")
    # ...while the DAILY chart is still unanswered: `main.py run` starts 1D only,
    # so the missing engine has to be named per timeframe, not "the collector".
    assert "no 1d engine" in app._lane_pulse_note("1d")
    assert "--timeframe 1d" in app._lane_pulse_note("1d")

    monkeypatch.setattr(app, "_LANE_PULSE", {}, raising=True)
    assert app._lane_pulse_note() == ""        # never asked → nothing to claim


def test_an_empty_answer_stays_an_empty_answer(monkeypatch):
    """The other half of the rule: a pair the exchange genuinely has no candles
    for must NOT be re-fetched on every rerun — silence is an answer, only a
    refusal is not (otherwise this honesty fix would turn into hammering)."""
    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()

    class _Empty:
        markets = {"X/USDT": {}}
        calls = 0

        def load_markets(self):
            return self.markets

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
            type(self).calls += 1
            return []

    monkeypatch.setattr(app, "_get_sync_exchange", lambda ccxt_id: _Empty())
    monkeypatch.setattr(app, "_STITCH_CACHE", {}, raising=True)

    errs = []
    for _ in range(3):
        assert app._fetch_missing_candles_cached(
            "bybit", "X/USDT", "15m", 10, 20, errors=errs) == []
    assert errs == []          # nothing failed → nothing to warn about
    assert _Empty.calls == 1   # and one request for three renders


def test_stitch_errors_tell_a_failed_fetch_from_an_empty_one():
    """The helper must be able to report WHICH of the two it is: an empty
    answer is drawn flat, a refused one is a hole in the chart."""
    from dashboard.helpers import stitch_candle_gaps

    import pandas as pd

    base = 1_800_000_000 - (1_800_000_000 % 900)
    df = pd.DataFrame({"ts": [base, base + 2 * 900], "open": [1.0, 2.0],
                       "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0],
                       "volume": [1.0, 1.0]})

    asked, errs = [], []
    _, n = stitch_candle_gaps(df, lambda r0, r1: asked.append((r0, r1)) or [], 900,
                              errors=errs)
    assert n == 0 and len(asked) == 1 and errs == []      # empty answer: no complaint

    def boom(r0, r1):
        raise TimeoutError("stitch budget used up")

    errs2 = []
    _, n2 = stitch_candle_gaps(df, boom, 900, errors=errs2)
    assert n2 == 0 and len(errs2) == 1 and "budget" in errs2[0]


def test_stitch_fetch_honours_its_time_budget(monkeypatch):
    """A very stale table must not page an exchange forever on the chart's
    critical path."""
    import time as _time

    os.environ["DASHBOARD_DEMO"] = "1"
    app = _load_app_module()

    class _SlowExchange:
        markets = {"X/USDT": {}}
        calls = 0

        def load_markets(self):
            return self.markets

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
            type(self).calls += 1
            _time.sleep(0.05)
            return [[since + i * 900_000, 1, 1, 1, 1, 1] for i in range(1000)]

    monkeypatch.setattr(app, "_get_sync_exchange", lambda ccxt_id: _SlowExchange())
    monkeypatch.setattr(app.settings, "dash_stitch_budget_sec", 0.12)

    fetch = app._fetch_missing_candles
    started = _time.perf_counter()
    rows, partial = fetch("bybit", "X/USDT", "15m", 0, 40_000)
    elapsed = _time.perf_counter() - started

    assert elapsed < 2.0
    assert _SlowExchange.calls < 40  # stopped early instead of paging 40 times
    assert rows and partial          # drawn AND admitted to be incomplete


# ---------------------------------------------------------------------------
# Summary scan: batching must survive a heterogeneous database.
#
# Every chunk of the scan is ONE `UNION ALL` over ~120 per-table subqueries.
# PostgreSQL resolves the type of each projected column across all branches,
# and a legacy table whose ob_* column is TEXT (an old HIGH<->LOW move that
# TEXT-ified them) made the whole chunk die with
#   DatatypeMismatchError: UNION types double precision and text cannot be matched
# The scan then did 120 per-table reads FOR THAT CHUNK — the exact cost the
# batching was invented to remove, and the reason the dashboard's first paint
# took minutes on a database with a few thousand of those tables.
# ---------------------------------------------------------------------------


def _union_is_type_stable(sql: str, schema: dict) -> bool:
    """Re-implements PostgreSQL's UNION type check for the generated SQL.

    Returns True when every projected column exposes the same type family in
    every branch (which is what makes the chunk a single round trip).
    """
    from dashboard.helpers import pg_type_group

    families: dict = {}
    for branch in sql.split("\nUNION ALL\n"):
        inner = branch.split("SELECT", 1)[1].split(" FROM ", 1)[0]
        for idx, expr in enumerate(p.strip() for p in inner.split(", ")):
            if "::" in expr:
                typ = expr.split("::", 1)[1].split(" AS ")[0].strip()
                fam = pg_type_group(typ)
            else:
                col = expr.strip('"')
                tbl = branch.split('FROM "', 1)[1].split('"', 1)[0]
                fam = pg_type_group(schema[tbl].get(col, "double precision"))
            if idx not in families:
                families[idx] = fam
            elif families[idx] != fam:
                return False
    return True


class _FakeScanPool:
    """asyncpg pool stand-in that REFUSES an unstable UNION, like Postgres does."""

    def __init__(self, schema, *, break_unions=False, break_rows=False):
        # break_unions: False | "first" (only the native projection dies, the
        # all-TEXT retry works) | True (the chunk is simply unqueryable) |
        # "timeout" (the server is LOADED — the chunk is skipped, not retried).
        self.schema = schema
        self.break_unions = break_unions
        self.break_rows = break_rows
        self.union_queries: list = []
        self.catalog_queries: list = []
        self.per_table_reads = 0

    def acquire(self, *a, **kw):
        outer = self

        class _Conn:
            async def fetch(self, query, *args):
                # *args: the catalog lookup is a PARAMETERIZED query (the
                # column list is bound, not interpolated) — keep the fake
                # honest about it.
                if "pg_catalog" in query:
                    outer.catalog_queries.append(query)
                    # the column list is BOUND, never interpolated, and the
                    # candle-time column is matched case-identically ("Timestamp")
                    assert args[0] == "Timestamp" and "Timestamp" not in args[1]
                    return [
                        {"table_name": t, "column_name": c, "data_type": dt}
                        for t, cols in outer.schema.items()
                        for c, dt in cols.items()
                    ]
                outer.union_queries.append(query)
                if outer.break_unions == "timeout":
                    import asyncio as _a
                    raise _a.TimeoutError()
                if outer.break_unions is True or (
                    outer.break_unions == "first" and len(outer.union_queries) == 1
                ):
                    raise RuntimeError("relation disappeared mid-scan")
                if not _union_is_type_stable(query, outer.schema):
                    raise RuntimeError(
                        "UNION types double precision and text cannot be matched"
                    )
                tables = [t for t in outer.schema if f"'{t}'::text" in query]
                return [outer._row(t) for t in tables]

            async def fetchrow(self, query, *args):
                outer.per_table_reads += 1
                if outer.break_rows:
                    raise RuntimeError("relation disappeared mid-scan")
                tbl = query.split('FROM "', 1)[1].split('"', 1)[0]
                return outer._row(tbl, table_name=tbl)

        class _Acquire:
            async def __aenter__(self):
                return _Conn()

            async def __aexit__(self, *exc):
                return False

        return _Acquire()

    def _row(self, tbl, table_name=None):
        row = {"table_name": table_name or tbl, "max_ts": 1_787_700_000}
        for col in self.schema[tbl]:
            if col == "Timestamp":
                row["Timestamp"] = 1_787_700_000
            elif self.schema[tbl][col] == "text":
                row[col] = "42.5"          # TEXT-typed number, as legacy tables store it
            else:
                row[col] = 42.5
        return row

    async def close(self):
        pass


def _run_scan(monkeypatch, pool, **kwargs):
    import asyncio

    from dashboard import app as dapp

    async def fake_create_pool(**kw):
        return pool

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)
    # the inventory (and any sweep cursor) is keyed by database name and lives in
    # a process-wide store now; tests must not inherit it from each other
    monkeypatch.setattr(dapp, "_SCAN_INVENTORY", {}, raising=False)
    monkeypatch.setattr(dapp, "_SCAN_SWEEP_STATE", {}, raising=False)
    kwargs.setdefault("budget_sec", 10.0)
    return asyncio.run(
        dapp._scan_database("db_test", "HIGH", "h", 1, "u", "p",
                            pool_size=2, chunk_size=120, **kwargs)
    )


def test_scan_casts_mixed_columns_instead_of_falling_back(monkeypatch):
    """The chunk is served by ONE query even with TEXT-typed legacy columns."""
    schema = {
        "btc_usdt_on_bybit": {
            "Timestamp": "bigint", "ticker": "text", "close": "double precision",
            "ob_vitality_score": "double precision", "ob_is_barcode": "boolean",
        },
        "old_spot_usdt_on_gateio": {          # legacy TEXT-ified metrics
            "Timestamp": "bigint", "ticker": "text", "close": "double precision",
            "ob_vitality_score": "text", "ob_is_barcode": "text",
        },
        "fresh_usdt:usdt_on_okx": {           # perp: no ob_* columns at all
            "Timestamp": "bigint", "ticker": "text", "close": "double precision",
        },
    }
    pool = _FakeScanPool(schema)
    rows = _run_scan(monkeypatch, pool)

    assert len(rows) == 3
    assert len(pool.union_queries) == 1, "one round trip per chunk, no degradation"
    assert pool.per_table_reads == 0
    assert '"ob_vitality_score"::text' in pool.union_queries[0]
    assert all('"db_test"' not in r or r["db_name"] == "db_test" for r in rows)
    assert {r["volume_tier"] for r in rows} == {"HIGH"}


def test_scan_retries_a_broken_chunk_as_text_before_per_table_reads(monkeypatch):
    """A chunk that fails for any other reason gets a type-stable retry, not
    120 single-table queries."""
    schema = {
        f"p{i}_usdt_on_bybit": {"Timestamp": "bigint", "ticker": "text", "close": "numeric"}
        for i in range(3)
    }
    pool = _FakeScanPool(schema, break_unions="first")
    rows = _run_scan(monkeypatch, pool)

    assert len(pool.union_queries) == 2            # native attempt + all-TEXT retry
    assert '"close"::text' in pool.union_queries[1]
    assert pool.per_table_reads == 0               # never reached
    assert len(rows) == 3


def test_scan_per_table_recovery_is_bounded_and_reports(monkeypatch):
    """When even the all-TEXT chunk dies, recovery is per-table and the broken
    tables are REPORTED — and an exhausted budget stops the recovery loop."""
    import asyncio

    from dashboard import app as dapp

    schema = {
        f"p{i}_usdt_on_bybit": {"Timestamp": "bigint", "ticker": "text", "close": "double precision"}
        for i in range(3)
    }
    pool = _FakeScanPool(schema, break_unions=True, break_rows=True)

    async def fake_create_pool(**kw):
        return pool

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(dapp, "_SCAN_INVENTORY", {}, raising=False)
    rows = asyncio.run(
        dapp._scan_database("db_x", "LOW", "h", 1, "u", "p",
                            pool_size=2, chunk_size=120, budget_sec=10.0)
    )
    assert rows == []
    assert pool.per_table_reads == 3               # tried each table, each failed
    meta = dapp._SCAN_META["db_x"]
    # errors + missing rows => the frame is not the truth: never cache it
    assert meta["partial"] is True and meta["tables"] == 3 and meta["rows"] == 0

    # a zero budget must not issue a single chunk query at all
    pool2 = _FakeScanPool(schema)
    rows2 = asyncio.run(
        dapp._scan_database("db_y", "LOW", "h", 1, "u", "p",
                            pool_size=2, chunk_size=120, budget_sec=0.0)
    )
    assert rows2 == [] and pool2.union_queries == []
    assert dapp._SCAN_META["db_y"]["partial"] is True


def test_partial_scan_is_not_persisted_as_the_snapshot(monkeypatch):
    """A truncated scan renders, but never overwrites the last good snapshot."""
    import pandas as pd

    from dashboard import app as dapp

    saved: list = []
    monkeypatch.setattr(dapp, "snapshot_path", lambda directory, tf: f"/tmp/{tf}.pkl")
    monkeypatch.setattr(
        dapp, "save_summary_snapshot",
        lambda path, df: saved.append(path) or True,
    )
    monkeypatch.setattr(dapp.settings, "dash_snapshot_enabled", True, raising=False)

    df = pd.DataFrame([{"ticker": "BTC/USDT", "close": 1.0}])

    async def fake_load(*a, **kw):
        return df

    monkeypatch.setattr(dapp, "_load_summary", fake_load)

    dapp._SCAN_META["15m"] = {"partial": True, "rows": 10, "tables": 99, "seconds": 25.0}
    dapp._scan_summary_now("h", 1, "u", "p", "15m")
    assert saved == []

    dapp._SCAN_META["15m"] = {"partial": False, "rows": 99, "tables": 99, "seconds": 3.0}
    dapp._scan_summary_now("h", 1, "u", "p", "15m")
    assert saved == ["/tmp/15m.pkl"]


def test_html_component_prefers_st_iframe_without_pinning_height(monkeypatch):
    """Two regressions at once: `components.html` (deprecated, one console
    warning per call — hundreds while browsing pairs) must not be selected just
    because the new API dropped the `scrolling` kwarg, and the cached signature
    probe must not pin the FIRST chart's height for every later component."""
    import types

    from dashboard import app as dapp

    iframe_calls: list = []
    legacy_calls: list = []

    def fake_iframe(src, *, width="stretch", height="content"):
        iframe_calls.append(height)

    monkeypatch.setattr(dapp, "_IFRAME_SCROLLING", None, raising=False)
    monkeypatch.setattr(dapp, "st", types.SimpleNamespace(iframe=fake_iframe))
    monkeypatch.setattr(
        dapp.components, "html", lambda html, height=None: legacy_calls.append(height)
    )

    dapp._html_component("<a>", 470)
    dapp._html_component("<b>", 300)
    assert iframe_calls == [470, 300]      # per-call height, no keyword reuse
    assert legacy_calls == []              # st.iframe exists -> no legacy path

    # an old Streamlit without st.iframe still renders through components.html
    monkeypatch.setattr(dapp, "st", types.SimpleNamespace())
    dapp._html_component("<c>", 250)
    assert legacy_calls == [250]


def test_scan_aborts_chunks_that_never_answer(monkeypatch):
    """The budget must bound the QUERY, not only its scheduling.

    With a database under heavy write load, chunks time out — and the pool's
    own command_timeout (30 s) per chunk, repeated over dozens of chunks, is a
    minute of spinner for a page that then renders a partial list anyway. The
    scan cancels each chunk at the remaining budget instead.
    """
    import asyncio
    import time as _time

    from dashboard import app as dapp

    schema = {f"p{i}_usdt_on_bybit": {"Timestamp": "bigint", "ticker": "text"} for i in range(3)}

    class _Conn:
        async def fetch(self, query, *args):
            if "pg_catalog" in query:
                return [
                    {"table_name": t, "column_name": c, "data_type": dt}
                    for t, cols in schema.items() for c, dt in cols.items()
                ]
            await asyncio.sleep(10)  # a chunk PostgreSQL never answers in time
            return []

        async def fetchrow(self, query, *args):  # pragma: no cover - must not run
            raise AssertionError("per-table reads must be skipped once the budget is gone")

    class _Pool:
        def acquire(self, *a, **kw):
            class _A:
                async def __aenter__(self):
                    return _Conn()

                async def __aexit__(self, *exc):
                    return False

            return _A()

        async def close(self):
            pass

    async def fake_create_pool(**kw):
        return _Pool()

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(dapp, "_SCAN_INVENTORY", {}, raising=False)
    started = _time.time()
    rows = asyncio.run(
        dapp._scan_database("db_slow", "HIGH", "h", 1, "u", "p",
                            pool_size=2, chunk_size=120, budget_sec=0.2)
    )
    elapsed = _time.time() - started

    assert rows == []
    assert elapsed < 3.0, f"budget did not bound the query: {elapsed:.1f}s"
    assert dapp._SCAN_META["db_slow"]["partial"] is True


# ---------------------------------------------------------------------------
# Progressive paint: DB page first, exchange data patched in afterwards.
#
# Flipping to the next pair cost ~4 s because the render path built the chart
# page synchronously: candles from the DB, then the gap/tail stitch paging the
# exchange under DASH_STITCH_BUDGET_SEC (4 s by default), plus inline ticker /
# orderbook / trade-tape fetches for the live chips whenever the live DB row
# for the new pair did not exist yet. The numbers the user could already see
# were the last thing rendered.
# ---------------------------------------------------------------------------


def test_chart_page_args_identity_excludes_credentials():
    from types import SimpleNamespace

    from dashboard import app as dapp

    row = {"db_name": "db1", "table_name": "btc_usdt_on_bybit"}
    monkey_src = SimpleNamespace(db_host="h", db_port=1, db_user="u", db_pass="s3cret")
    saved = {k: getattr(dapp, k) for k in ("db_host", "db_port", "db_user", "db_pass")}
    for k, v in vars(monkey_src).items():
        setattr(dapp, k, v)
    try:
        page_kwargs, store_key = dapp._chart_page_args(
            row, "15m", 700, "", "", 0, "bybit", "BTC/USDT", "bybit",
            "Candlesticks", 430, False, "POLLER_JS", "HIST_JS", True,
        )
    finally:
        for k, v in saved.items():
            setattr(dapp, k, v)

    assert (page_kwargs["db_host"], page_kwargs["db_port"], page_kwargs["db_user"],
            page_kwargs["db_pass"]) == ("h", 1, "u", "s3cret")   # cache needs the DB
    assert "s3cret" not in store_key and "db_pass" not in store_key  # store must not
    assert "btc_usdt_on_bybit" in store_key and "POLLER_JS" in store_key
    # a different pair/exchange/style is a different page
    other = list(store_key)
    other[other.index("Candlesticks")] = "OHLCV Bars"
    assert tuple(other) != store_key


def test_chart_page_kwargs_bind_to_the_builder_signature():
    """`_chart_page_args` must produce kwargs the real builder accepts.

    Regression for the crash this helper caused: the page identity was passed
    as a POSITIONAL tuple while `stitch_enabled` sits in the middle of the
    builder's 22-argument signature, so `live_poller_js` landed on it and every
    chart died with

        TypeError: _render_chart_html_cached() got multiple values for
        argument 'stitch_enabled'

    No test caught it because demo mode (the only path AppTest can run without
    a database) takes the other branch. Binding against the live signature is
    the cheap way to keep that true for the next parameter added.
    """
    import inspect
    from types import SimpleNamespace

    from dashboard import app as dapp

    saved = {k: getattr(dapp, k) for k in ("db_host", "db_port", "db_user", "db_pass")}
    for k, v in {"db_host": "h", "db_port": 1, "db_user": "u", "db_pass": "p"}.items():
        setattr(dapp, k, v)
    try:
        page_kwargs, store_key = dapp._chart_page_args(
            {"db_name": "db1", "table_name": "btc_usdt_on_bybit"},
            "1D", 400, "db1", "btc_usdt_on_bybit", 200, "bybit", "BTC/USDT", "bybit",
            "OHLCV Bars", 430, True, "P", "H", False,
        )
        sig = inspect.signature(dapp._render_chart_html_cached.__wrapped__)
        sig.bind(**page_kwargs, stitch_enabled=False)     # render path
        sig.bind(**page_kwargs, stitch_enabled=True)       # background warm
        sig.bind_partial(**page_kwargs)                    # __wrapped__ call shape
    finally:
        for k, v in saved.items():
            setattr(dapp, k, v)

    # every kwarg name is derived from the signature: no silent renames
    assert set(page_kwargs) == set(sig.parameters) - {"stitch_enabled"}
    assert len(store_key) == 17 and all(isinstance(v, (str, int, bool)) for v in store_key)
    assert hash(store_key)                                   # dict key, must be hashable


def test_warm_stitched_page_stores_patch_and_counts_candles(monkeypatch):
    from types import SimpleNamespace

    from dashboard import app as dapp

    built = {"n": 0}

    def fake_build(*args, stitch_enabled=False, **kw):
        built["n"] += 1
        built["stitch"] = stitch_enabled
        built["as_kwargs"] = bool(kw) and not args
        return ("<html>patched</html>", "🩹 47 missing 15m candles stitched from exchange (in-memory)")

    monkeypatch.setattr(dapp, "_render_chart_html_cached",
                        SimpleNamespace(__wrapped__=fake_build), raising=False)
    key, args = ("db1.t1.15m",), {"table_name": "t1", "db_name": "db1"}

    assert dapp._warm_stitched_page(key, args) == 47
    # built UNCHACHED, with stitching on (the whole point: not on the UI path)
    assert built == {"n": 1, "stitch": True, "as_kwargs": True}   # keywords, not positional
    entry = dapp._STITCHED_PAGES[key]
    assert entry["html"] == "<html>patched</html>"
    assert entry["hash"] == hash("<html>patched</html>")
    # the store is bounded: browsing pairs must not pile up one HTML page each
    for i in range(200):
        dapp._remember(dapp._STITCHED_PAGES, f"k{i}", {"at": i})
    assert len(dapp._STITCHED_PAGES) == dapp._PAGE_STORE_LIMIT


def test_stitched_candle_count_parsing():
    from dashboard import app as dapp

    assert dapp._stitched_candle_count(
        "🩹 12 missing 1D candles stitched from exchange (in-memory) · ⏳ collector 5.0h behind"
    ) == 12
    assert dapp._stitched_candle_count("&nbsp;") == 0          # nothing stitched
    assert dapp._stitched_candle_count("") == 0
    assert dapp._stitched_candle_count(None) == 0


def test_the_spread_chips_do_not_need_the_orderbook_to_be_known(monkeypatch):
    """"spread 0.0001 (0.098%)" on the LIVE line, "↔ Spread % ATR n/a" in the
    chip above it — the user's question, and a real bug.

    The writer stores bid/ask from `fetch_ticker` for every pair, but
    `depth_usd` only when a full orderbook came back for the pair that is OPEN.
    The strip computed the spread-vs-ATR fields INSIDE the
    "depth_usd is not None" branch, so a pair with no book answered "n/a" from
    numbers it had in hand. Each field now takes the best source that has it."""
    from dashboard import app as dapp

    live = {"last": 0.1024, "bid": 0.1023, "ask": 0.1024,
            "depth_usd": None, "pct": None, "trades_per_min": 5.0}
    monkeypatch.setattr(dapp, "_db_live_read", lambda *a, **k: dict(live))
    monkeypatch.setattr(dapp, "_feed_value", lambda *a, **k: None)

    row = dapp._compute_live_health_row(
        "1000000MOG/USDT", "mexc", None, 0.0026, {}, db_name="vol_15m_low")

    assert row["ob_best_bid"] == 0.1023 and row["ob_best_ask"] == 0.1024
    assert row["ob_spread_abs"] == pytest.approx(0.0001)
    assert row["ob_spread_atr_pct"] == pytest.approx(0.0001 / 0.0026 * 100.0)
    # …and depth is NOT invented: it is missing, and the chip says so
    assert "ob_total_depth_usd" not in row
    assert row["ob_trades_per_min"] == 5.0

    # no daily ATR (a pair whose 1D table is too short) must not lose the spread
    # fields either — it only loses the ratio, which is what the chip means
    row2 = dapp._compute_live_health_row(
        "1000000MOG/USDT", "mexc", None, 0.0, {}, db_name="vol_15m_low")
    assert row2["ob_spread_abs"] == pytest.approx(0.0001)
    assert "ob_spread_atr_pct" not in row2

    # a book from the background feed fills the depth, still with no request here
    book = {"bids": [[0.1023, 4000.0], [0.1000, 9000.0]], "asks": [[0.1024, 2000.0]]}
    calls = []
    monkeypatch.setattr(dapp, "_feed_value",
                        lambda kind, *a, **k: calls.append(kind) or book)
    row3 = dapp._compute_live_health_row(
        "1000000MOG/USDT", "mexc", None, 0.0026, {}, db_name="vol_15m_low")
    # the chip is DEPTH WITHIN ±1%: 0.1000 is 2.3 % away from the mid, so it is
    # counted out — a book 10x deeper than the visible top would still read thin
    assert row3["ob_total_depth_usd"] == pytest.approx(0.1023 * 4000.0 + 0.1024 * 2000.0)
    assert calls and set(calls) <= {"orderbook", "tape"}


def test_the_live_line_omits_what_the_exchange_never_answered():
    """A missing `percentage` in the ticker printed a dangling "· 24h" with no
    value — the reader's conclusion being that the live feed is broken, while
    the writer simply stored NULL. Only finite, present fields are printed."""
    from dashboard import app as dapp

    html = dapp._live_line_html(
        {"last": 0.1024, "bid": 0.1023, "ask": 0.1024, "pct": None})
    assert "spread 0.0001" in html
    assert "(0.098% of mid)" in html          # the denominator is named
    assert "24h" not in html

    assert "24h" not in dapp._live_line_html(
        {"last": 1.0, "bid": 0.9, "ask": 1.1, "pct": float("nan")})
    assert "bid" not in dapp._live_line_html({"last": 1.0, "bid": None, "ask": float("nan")})
    assert dapp._live_line_html({"last": None, "bid": 1.0}) == ""
    assert dapp._live_line_html({}) == ""
    assert dapp._live_line_html({"last": 2.0, "pct": -0.27}) .count("24h") == 1


def test_the_strip_says_which_input_a_missing_chip_is_waiting_for():
    """"n/a" on its own reads as "the dashboard is broken"; the chip has to name
    the missing input, because for a live pair these are three different
    problems (no book for this pair, no daily ATR yet, or a NULL collector
    snapshot)."""
    from dashboard.helpers import build_health_strip_html

    html = build_health_strip_html({"ob_trades_per_min": 5.0,
                                    "ob_min_7d_volume_usd": 271_000.0})
    assert "no orderbook row" in html          # Depth chip explains itself
    assert "fewer than 3 bars" in html         # Spread chip explains itself
    assert "5/min" in html and "$271K" in html

    filled = build_health_strip_html({"ob_trades_per_min": 5.0, "ob_total_depth_usd": 1e3,
                                      "ob_spread_atr_pct": 3.9,
                                      "ob_min_7d_volume_usd": 4.95e5})
    assert "no orderbook row" not in filled and "daily ATR" not in filled


def test_every_atr_mention_names_the_timeframe_and_the_period(monkeypatch):
    """"ATR" alone is not a metric. The dashboard has THREE estimators behind
    similar-looking numbers — `1D_ATR(n)` filtered (strip + metric cards, n from
    the sidebar slider), `1D_ATR(ATR_PERIOD)` filtered (what the 1D engine wrote
    into the pair table) and `15m_ATR(ATR_PERIOD)` Gerchik-smoothed (what the 15m
    engine wrote) — so every label must be BUILT from the values that produced the
    number on screen, or a reader compares incomparable things."""
    from dashboard import app as dapp
    from dashboard.helpers import build_health_strip_html, format_atr_label

    assert format_atr_label("1D", 5) == "1D_ATR(5)"
    assert format_atr_label("15m", 10) == "15m_ATR(10)"
    assert format_atr_label("1d", 5) == "1D_ATR(5)"        # the slider/DB differ in case
    assert format_atr_label("1D", 0) == "1D_ATR(?)"         # never invent a period
    assert format_atr_label("15m", 5, style="Gerchik") == "15m_ATR(5)·Gerchik"

    monkeypatch.setattr(dapp.settings, "atr_period", 12)
    # a pair row read from the database is labelled by its OWN timeframe, because
    # the engine that wrote it is the only thing that decided the estimator
    assert dapp._db_atr_label(row_is_15m=False) == "1D_ATR(12)"
    assert dapp._db_atr_label(row_is_15m=True) == "15m_ATR(12)"

    live = {"last": 0.1022, "bid": 0.1021, "ask": 0.1022,
            "depth_usd": 45_000.0, "pct": -1.73, "trades_per_min": 4.0}
    monkeypatch.setattr(dapp, "_db_live_read", lambda *a, **k: dict(live))
    monkeypatch.setattr(dapp, "_feed_value", lambda *a, **k: None)
    row = dapp._compute_live_health_row(
        "1000000MOG/USDT", "bybit", None, 0.0036, {}, db_name="vol_15m_high",
        atr_label=format_atr_label("1D", 5))
    html = build_health_strip_html(row)
    assert "↔ Spread % 1D_ATR(5)" in html                  # chip label
    assert html.count("1D_ATR(5)") >= 2                    # label AND tooltip formula
    assert "Wilder" in html                                # says what it is NOT
    assert "$45K" in html                                  # depth came from the live row

    # the same strip from a stored 15m row must not claim to be daily
    row2 = dapp._compute_live_health_row(
        "1000000MOG/USDT", "bybit", None, 0.0, {"ob_spread_atr_pct": 2.8},
        db_name="vol_15m_high")
    assert "15m_ATR(12)" in build_health_strip_html(row2)
    assert dapp._live_line_html(live, "1D_ATR(5)").count("1D_ATR(5)") >= 1
    assert "0.098% of mid" in dapp._live_line_html(live, "1D_ATR(5)")


def test_no_bare_atr_label_is_left_in_the_ui():
    """A guard, not a style rule: the user-visible ATR strings are built through
    `format_atr_label`, so an edit that puts a bare "ATR" back into a chip, a
    metric heading or a table header has to fail something. (Docstrings and
    comments may still talk about ATR generically — they are not on screen.)"""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "dashboard"
    banned = [
        '"↔ Spread % ATR"',          # the chip, without a named estimator
        '"Spread % of ATR"',         # the metric card / column header
        '"ATR w/o Paranormal Bars',  # the metric card / column header
        '"🎯 ATR Period (bars)"',    # the slider, which did not say WHICH bars
        "### 1. ATR without Paranormal Bars",   # methodology section headings
        "### 2. Spread % of ATR",               # ditto
    ]
    offenders = []
    for name in ("app.py", "helpers.py"):
        text = (root / name).read_text()
        for lit in banned:
            if lit in text:
                line = text[: text.index(lit)].count("\n") + 1
                offenders.append(f"dashboard/{name}:{line}: {lit}")
    assert not offenders, offenders


def test_feed_value_serves_cache_and_never_fetches_inline(monkeypatch):
    from dashboard import app as dapp

    calls, scheduled = [], []
    monkeypatch.setattr(dapp, "_FEEDS", {}, raising=False)
    monkeypatch.setattr(
        dapp, "_fetch_ticker_cached",
        lambda ccxt_id, symbol: calls.append((ccxt_id, symbol)) or {"last": 7.5},
    )
    # _bg is recorded, not executed: the render path must only SCHEDULE work
    monkeypatch.setattr(dapp, "_bg", lambda key, fn, *a, **kw: scheduled.append((key, fn, a)))

    # first view of a pair: nothing cached, no REST call, a warm is scheduled
    assert dapp._feed_value("ticker", "bybit", "BTC/USDT") is None
    assert calls == []
    assert scheduled and scheduled[0][0] == ("feed", "ticker", "bybit", "BTC/USDT")

    # the background thread fills it; now the widget renders it for free
    scheduled[0][1](*scheduled[0][2])
    assert calls == [("bybit", "BTC/USDT")]
    assert dapp._feed_value("ticker", "bybit", "BTC/USDT") == {"last": 7.5}

    # a value older than the LIVE window is not shown as live — but refreshed
    dapp._FEEDS[("ticker", "bybit", "BTC/USDT")]["at"] -= 10_000
    assert dapp._feed_value("ticker", "bybit", "BTC/USDT") is None
    assert len(scheduled) == 2


def test_inventory_is_read_from_pg_catalog_and_cached(monkeypatch):
    """Two regressions in one line of code.

    * information_schema.columns is a VIEW that runs has_table_privilege() per
      column per table — on a 14k-table database it measured 30…250 s, which
      was the actual dashboard startup cost (a 75-table database "scanned" in
      8…24 s for ONE chunk query: the time was in the catalog, not the data).
    * and it was re-read on every rescan, i.e. on every rerun of the app.
    """
    import asyncio

    from dashboard import app as dapp

    queries = []

    class _Conn:
        async def fetch(self, query, *args):
            queries.append(query)
            return [{"table_name": "btc_usdt_on_bybit", "column_name": "close",
                     "data_type": "float8"}]

    class _Pool:
        def acquire(self, *a, **kw):
            class _A:
                async def __aenter__(self):
                    return _Conn()

                async def __aexit__(self, *exc):
                    return False

            return _A()

    monkeypatch.setattr(dapp, "_SCAN_INVENTORY", {}, raising=False)
    monkeypatch.setattr(dapp.settings, "dash_scan_inventory_ttl_sec", 600.0, raising=False)

    tables = asyncio.run(dapp._table_inventory(_Pool(), "db_a", "host1", 5432))
    assert tables == {"btc_usdt_on_bybit": {"close": "float8"}}
    assert len(queries) == 1
    q = queries[0]
    assert "information_schema" not in q and "pg_catalog.pg_attribute" in q
    # '_' is a LIKE wildcard: it must be escaped, or unrelated tables (a
    # "leone_tmp", any name containing "xony") join the pair list
    assert r"LIKE '%\_on\_%'" in q
    assert "a.attnum > 0" in q and "NOT a.attisdropped" in q        # real columns only
    assert "c.relkind IN ('r', 'p')" in q                          # hypertables are 'p'
    assert "t.typname" in q                                        # float8, not "double precision"

    # second call inside the TTL: zero queries (this is what stops a scan
    # storm on every rerun of the app)
    asyncio.run(dapp._table_inventory(_Pool(), "db_a", "host1", 5432))
    assert len(queries) == 1

    # …but a DIFFERENT server is a different inventory (the sidebar can point
    # the dashboard at another Postgres)
    asyncio.run(dapp._table_inventory(_Pool(), "db_a", "host2", 5433))
    assert len(queries) == 2
    queries.clear()

    # ttl=0 -> re-read every time (documented escape hatch for new listings)
    monkeypatch.setattr(dapp.settings, "dash_scan_inventory_ttl_sec", 0.0, raising=False)
    asyncio.run(dapp._table_inventory(_Pool(), "db_a", "host2", 5433))
    assert len(queries) == 1


def test_background_revalidation_is_throttled():
    """The scan is expensive; a snapshot means the list is USABLE, so the
    refresh must not re-fire on every rerun (pair click, 60 s reload, …)."""
    from dashboard.helpers import snapshot_refresh_due

    assert snapshot_refresh_due(0.0, 1000.0, 120.0) is True         # never scanned
    assert snapshot_refresh_due(900.0, 1000.0, 120.0) is False       # 100 s ago
    assert snapshot_refresh_due(880.0, 1000.0, 120.0) is True        # 120 s ago
    assert snapshot_refresh_due(None, 1000.0, 120.0) is True
    # interval 0 disables the throttle (legacy behaviour)
    assert snapshot_refresh_due(999.9, 1000.0, 0.0) is True


def test_pg_type_group_accepts_catalog_typnames():
    """The cast plan compares type FAMILIES, and pg_type speaks in typname
    aliases while information_schema speaks SQL names — both must land in the
    same family, or every chunk looks 'mixed' and gets flattened to TEXT."""
    from dashboard.helpers import pg_type_group, resolve_summary_union_casts

    assert pg_type_group("float8") == pg_type_group("double precision") == "number"
    assert pg_type_group("int8") == pg_type_group("bigint") == "number"
    assert pg_type_group("bool") == pg_type_group("boolean") == "bool"
    assert pg_type_group("varchar") == pg_type_group("text") == "text"

    # a whole database of float8 columns is uniform -> no casts at all
    tables = {
        f"p{i}_usdt_on_bybit": {
            "Timestamp": "int8", "close": "float8", "ob_spread_pct": "float8",
            "ob_is_barcode": "bool", "ticker": "text",
        }
        for i in range(3)
    }
    assert resolve_summary_union_casts(tables, ["close", "ob_spread_pct", "ticker"]) == {}


def test_chart_page_actually_builds_from_those_kwargs(tmp_path, monkeypatch):
    """End-to-end shape check: the real builder, fed by the real kwargs.

    Binding against the signature proves the names exist; this proves the
    values reach the right PARAMETERS — i.e. that the page that gets built is a
    chart, not a TypeError, for both variants (DB-only and stitched). It needs
    no database and no Streamlit app: the candle frame is the only input.
    """
    import time as _time

    import pandas as pd

    from dashboard import app as dapp

    now = int(_time.time()) - 3 * 900            # a healthy, recently written table
    frame = pd.DataFrame({
        "ts": [now + i * 900 for i in range(6)],
        "open": [1.0] * 6, "high": [1.1] * 6, "low": [0.9] * 6,
        "close": [1.05] * 6, "volume": [10.0] * 6,
    })
    monkeypatch.setattr(dapp, "load_candles_cached", lambda *a, **kw: frame)
    fetches = []
    monkeypatch.setattr(
        dapp, "_fetch_missing_candles_cached",
        lambda *a, **kw: fetches.append(a) or [],
    )
    for k, v in {"db_host": "h", "db_port": 1, "db_user": "u", "db_pass": "p"}.items():
        monkeypatch.setattr(dapp, k, v)

    row = {"db_name": "db1", "table_name": "btc_usdt_on_bybit", "max_ts": now + 5 * 900}
    page_kwargs, store_key = dapp._chart_page_args(
        row, "15m", 700, "db1", "btc_usdt_on_bybit", 200, "bybit",
        "BTC/USDT", "bybit", "Candlesticks", 430, True, "POLLER", "HIST", True,
    )

    html, txt = dapp._render_chart_html_cached.__wrapped__(**page_kwargs, stitch_enabled=False)
    assert "createChart" in html and len(html) > 2000
    assert txt == "&nbsp;"                        # nothing stitched, nothing to say
    assert fetches == []                          # the plain page touches no exchange

    # same kwargs, stitched variant — the ONLY difference is that the stitch runs
    html2, _ = dapp._render_chart_html_cached.__wrapped__(**page_kwargs, stitch_enabled=True)
    assert "createChart" in html2
    assert isinstance(store_key, tuple) and len(store_key) == 17

    # and the warning the progressive paint must NOT swallow: a stale table
    # with nothing to stitch says so, in red, on the plain page too
    stale = dict(row, max_ts=now - 200 * 3600)
    stale_kwargs, _ = dapp._chart_page_args(
        stale, "15m", 700, "", "", 0, "bybit", "BTC/USDT", "bybit",
        "Candlesticks", 430, True, "", "", True,
    )
    _, txt_stale = dapp._render_chart_html_cached.__wrapped__(**stale_kwargs, stitch_enabled=False)
    assert "collector" in txt_stale and "behind" in txt_stale


# ---------------------------------------------------------------------------
# The scan must never become the load it is scanning around.
#
# A production session showed the dashboard issuing a 4-database UNION sweep
# every ~30 s for minutes, chunks timing out, each timed-out chunk then being
# re-read table by table (250–313 s per database), and asyncpg printing
# "Fatal error on transport TCPTransport" for the connections cancelled at the
# deadline. All of it was the dashboard fighting the collector for the same
# Postgres — so the fixes below are about admission, retry policy and backoff.
# ---------------------------------------------------------------------------


def _summary_stores(monkeypatch, dapp):
    """Module stores + a fake `st`, so load_summary_cached can be driven without
    an app or a database. `types` is imported here because the tests below are
    the only ones that reach into the module's state."""
    import types as _types

    for name in ("_SUMMARY_STORE", "_SCAN_ATTEMPTS", "_LAST_SCAN_AT", "_SCAN_META",
                 "_SCAN_INVENTORY", "_SCAN_SWEEP_STATE", "_SCAN_DEFERRED_AT",
                 "_SCAN_DEFER_TRIES"):
        monkeypatch.setattr(dapp, name, {}, raising=False)
    monkeypatch.setattr(dapp, "st", _types.SimpleNamespace(
        session_state={}, error=lambda *a, **k: None))
    return dapp


def test_truncated_scan_is_served_from_memory_not_rescanned(monkeypatch):  # noqa: E301
    """THE rescan-storm regression.

    A truncated scan is deliberately never persisted to disk, so the snapshot
    branch (and the throttle inside it) never applies on a loaded database:
    the partial path used to (a) run a blocking scan on EVERY rerun and (b)
    clear the whole Streamlit cache after every background scan, so the next
    rerun scanned again. Serve the in-memory result and retry at most once per
    backoff window.
    """
    import pandas as pd

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_enabled", False)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_refresh_sec", 120.0)
    scans, launched = [], []

    async def fake_load(*a, **k):
        scans.append(1)
        dapp._SCAN_META["15m"] = {"partial": True, "rows": 5, "tables": 40, "seconds": 25.0}
        return pd.DataFrame({"ticker": [f"p{i}" for i in range(5)]})

    monkeypatch.setattr(dapp, "_load_summary", fake_load)
    monkeypatch.setattr(
        dapp, "_refresh_summary_in_background",
        lambda *a, **k: launched.append(1),
    )

    for _ in range(20):
        df = dapp.load_summary_cached.__wrapped__("h", 1, "u", "p", "15m")

    assert len(df) == 5
    assert len(scans) == 1, f"the render path scanned {len(scans)} times"
    assert len(launched) == 1, "a truncated scan must not re-arm a rescan per rerun"
    assert dapp._SCAN_ATTEMPTS["15m"] == 1          # the streak drives the delay
    assert dapp._rescan_delay_sec("15m") == 240.0   # … and it backs off
    assert dapp.st.session_state["_partial_scan_15m"][1]["tables"] == 40  # badge data


def test_a_complete_scan_resets_the_backoff(monkeypatch):
    import pandas as pd

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_enabled", False)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_refresh_sec", 120.0)
    dapp._SCAN_ATTEMPTS["15m"] = 4

    async def complete(*a, **k):
        dapp._SCAN_META["15m"] = {"partial": False, "rows": 40, "tables": 40, "seconds": 3.0}
        return pd.DataFrame({"ticker": ["a"]})

    monkeypatch.setattr(dapp, "_load_summary", complete)
    # the reset is about the BACKOFF streak, not the refresh interval: the
    # complete-scan floor (next test) is allowed to exceed it deliberately.
    monkeypatch.setattr(dapp.settings, "dash_scan_rescan_complete_sec", 0.0)
    dapp._scan_summary_now("h", 1, "u", "p", "15m")
    assert dapp._SCAN_ATTEMPTS["15m"] == 0
    assert dapp._rescan_delay_sec("15m") == 120.0
    assert dapp._SUMMARY_STORE["15m"]["meta"]["partial"] is False


def test_a_complete_tier_is_not_re_scanned_while_another_is_starving(monkeypatch):
    """One scan gate, two tiers: the tier that keeps FINISHING used to re-scan
    every refresh tick and the other one never got the database. Their 18:xx log
    is the picture — `1d: … 116 chunk(s) / 9094 tables in 20.5s` over and over,
    while `15m_low_vol` decayed from 1560 rendered tables to 0.

    A complete answer is therefore held for `dash_scan_rescan_complete_sec`;
    a truncated one still follows the retry streak, because converging fast is
    the whole point of carrying its rows."""
    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_refresh_sec", 30.0)
    monkeypatch.setattr(dapp.settings, "dash_scan_retry_max_sec", 1800.0)
    monkeypatch.setattr(dapp.settings, "dash_scan_rescan_complete_sec", 300.0)

    # a complete scan WITH an answer is the only thing worth holding quiet
    dapp._SCAN_META["15m"] = {"partial": False, "rows": 8312}
    dapp._SCAN_ATTEMPTS["15m"] = 0
    assert dapp._rescan_delay_sec("15m") == 300.0

    dapp._SCAN_META["15m"] = {"partial": True, "rows": 6514}
    assert dapp._rescan_delay_sec("15m") == 30.0

    # an "empty but complete" answer is NEVER worth 5 minutes of silence: that
    # is how a blank selector stayed blank
    dapp._SCAN_META["15m"] = {"partial": False, "rows": 0, "tables": 0}
    assert dapp._rescan_delay_sec("15m") == 30.0


def test_an_expired_catalog_is_not_reread_mid_sweep(monkeypatch, capsys):
    """`+catalog 14.2s`, `+catalog 17.0s` on every single 15m/LOW pass — that is
    a third of each pass spent re-listing 8 243 tables the sweep is still busy
    walking. While the sweep has not wrapped, the previous listing is reused:
    the table set does not change on that timescale, and the budget and pool
    connections go to the tables that have no answer yet."""
    import asyncio
    import time as _time

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_scan_inventory_ttl_sec", 600.0)
    monkeypatch.setattr(dapp, "_MARKET_LOG_AT", {})
    schema = {f"p{i}_usdt_on_bybit": {"Timestamp": "timestamptz"} for i in range(3)}
    pool = _FakeScanPool(schema)

    async def fake_create_pool(**kw):
        return pool

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)

    # a sweep in progress with a stale-but-present cache: no catalog query
    dapp._SCAN_INVENTORY[("h", 1, "db_grace")] = {
        "at": _time.time() - 5000.0, "tables": dict(schema)}
    rows = asyncio.run(dapp._scan_database(
        "db_grace", "HIGH", "h", 1, "u", "p", pool_size=1, chunk_size=120,
        budget_sec=10.0, sweep={"start": 3}))
    assert len(rows) == 3
    assert pool.catalog_queries == [], "the catalog must not be re-read mid-sweep"
    assert "reusing it instead of paying another 15s read" in capsys.readouterr().out

    # a fresh sweep (no cursor) pays the read, so a new listing is seen promptly
    pool2 = _FakeScanPool(schema)

    async def fake_create_pool2(**kw):
        return pool2

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool2)
    asyncio.run(dapp._scan_database(
        "db_grace", "HIGH", "h", 1, "u", "p", pool_size=1, chunk_size=120,
        budget_sec=10.0, sweep={}))
    assert pool2.catalog_queries, "a wrapped sweep must re-read the catalog"


def test_an_empty_pair_list_never_replaces_a_real_one(monkeypatch):
    """The remaining hole after the last round, and the difference between
    "the list is short" and "the tier is gone".

    The monotonic merge used to be conditional on the scan calling itself
    truncated. A scan that answers NOTHING because the catalog came back empty
    does not call itself truncated (there is nothing to be missing, as far as it
    knows) — so an empty frame was allowed to overwrite a 8 312-pair list, and it
    was even persisted to disk as the startup snapshot, so a restart did not
    help either."""
    import pandas as pd

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_enabled", True)
    saved = []
    monkeypatch.setattr(dapp, "save_summary_snapshot",
                        lambda path, df: saved.append(len(df)))
    dapp._SUMMARY_STORE["15m"] = {
        "df": pd.DataFrame({"db_name": ["a", "b", "c"],
                            "table_name": ["t1", "t2", "t3"],
                            "ticker": ["A/USDT", "B/USDT", "C/USDT"]}),
        "meta": {"partial": True}, "at": 0.0}

    async def blank(*a, **k):
        # the worst case: the scan says "complete", and the answer is nothing
        dapp._SCAN_META["15m"] = {"partial": False, "rows": 0, "tables": 0}
        return pd.DataFrame()

    monkeypatch.setattr(dapp, "_load_summary", blank)
    df = dapp._scan_summary_now("h", 1, "u", "p", "15m")

    assert len(df) == 3                       # the old list survives
    assert dapp._SUMMARY_STORE["15m"]["meta"]["kept_rows"] == 3
    assert saved == [], "an empty frame must never be persisted as the snapshot"


def test_an_empty_catalog_is_reported_as_a_failed_read(monkeypatch, capsys):
    """`_scan_database` used to `return []` silently when the inventory came back
    empty, leaving `_SCAN_META[db]` at whatever the PREVIOUS pass wrote — so a
    tier could be declared complete and empty by a catalog read that answered
    nothing. A database that had tables a pass ago and now has none is a failed
    read, and it has to be labelled as one."""
    import asyncio

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    pool = _FakeScanPool({})          # catalog answers: no pair tables at all

    async def fake_create_pool(**kw):
        return pool

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)

    # first pass: this database had never listed anything -> a real "no pairs"
    rows = asyncio.run(dapp._scan_database(
        "db_empty", "LOW", "h", 1, "u", "p", pool_size=1, chunk_size=120,
        budget_sec=10.0, sweep={}))
    assert rows == []
    assert dapp._SCAN_META["db_empty"]["partial"] is False
    assert "0 pair tables" not in capsys.readouterr().out

    # second pass, after a pass that DID list tables -> suspected failed read
    dapp._SCAN_META["db_empty"] = {"tables": 4173, "rows": 4173, "partial": False}
    asyncio.run(dapp._scan_database(
        "db_empty", "LOW", "h", 1, "u", "p", pool_size=1, chunk_size=120,
        budget_sec=10.0, sweep={}))
    meta = dapp._SCAN_META["db_empty"]
    assert meta["partial"] is True
    assert meta["missing_tables"] == 4173 and meta["tables"] == 0
    out = capsys.readouterr().out
    assert "the catalog answered 0 pair tables (previous pass: 4173)" in out
    assert "NOT caching that as 'no pairs'" in out


def test_the_pair_list_funnel_says_which_filter_ate_the_tier(capsys):
    """"No 15m charts" with a database full of them was twice a mystery because
    the page only warns when BOTH tiers are empty. The funnel has to name the
    step, not the symptom."""
    import pandas as pd

    from dashboard import app as dapp

    class _St:
        def __init__(self):
            self.warnings, self.captions = [], []

        def warning(self, *a, **k):
            self.warnings.append(a[0])

        def caption(self, *a, **k):
            self.captions.append(a[0])

    st = _St()
    orig = dapp.st
    dapp.st = st
    try:
        assert dapp._pair_list_funnel("15m", 8312, pd.DataFrame()) == 8312
        assert st.warnings and "8312" in st.warnings[0]
        assert "Hide dead spot duplicates" in st.warnings[0]
        assert "filter_sane_summary_rows" in st.warnings[0]

        # a modest, legitimate shrink says nothing
        st.warnings.clear(); st.captions.clear()
        df = pd.DataFrame({"ticker": [f"P{i}/USDT" for i in range(8000)]})
        assert dapp._pair_list_funnel("15m", 8312, df) == 312
        assert not st.warnings and not st.captions

        # most of the tier gone -> a caption, and the console line
        st.warnings.clear(); st.captions.clear()
        df2 = pd.DataFrame({"ticker": [f"P{i}/USDT" for i in range(40)]})
        assert dapp._pair_list_funnel("1D", 800, df2) == 760
        assert st.captions and "40 in the list" in st.captions[0]
        assert not st.warnings
        assert "[pairs] 1D: the scan answered 800 table(s)" in capsys.readouterr().out
    finally:
        dapp.st = orig


def test_a_sweep_that_is_still_building_the_list_is_not_backed_off(monkeypatch):
    """A truncated scan has two very different causes, and the schedule must
    tell them apart: a sweep that ANSWERS chunks but runs out of budget is
    converging, while a database that answers nothing needs the doubling backoff.

    The converging case must not be paced at all beyond letting clicks cut in
    front: their log shows `rendering 720/8243 … chunk 6/69`, then 1080 at
    chunk 9, then 1560 at chunk 14 — real progress, ~6 chunks per pass — with
    `retry in ~173 s (backoff)` between the passes. 69 chunks at that pace is
    ~40 minutes of half-built selector, which reads as "15m не загружается"
    even though the sweep re-reads nothing the pass before it answered.
    Conflating the two cases is also what left an 8 235-table tier rendering
    1 560 pairs, then 600, then 120 — an hour of "the charts are missing"."""
    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_refresh_sec", 120.0)
    monkeypatch.setattr(dapp.settings, "dash_scan_retry_max_sec", 1800.0)
    monkeypatch.setattr(dapp.settings, "dash_scan_rescan_complete_sec", 300.0)
    monkeypatch.setattr(dapp.settings, "dash_scan_defer_retry_sec", 8.0)
    dapp._SCAN_ATTEMPTS["15m"] = 6

    # converging: partial, cursor unwrapped, chunks answered → keep going soon,
    # whatever the failure streak says (the pause is for clicks, not for pacing)
    dapp._SCAN_META["15m"] = {"partial": True, "sweep_incomplete": True,
                              "answered_chunks": 4}
    assert dapp._rescan_delay_sec("15m") == 8.0

    # stuck: same streak, but the last pass answered nothing → back off, the
    # database is the problem and hammering it makes everyone slower
    dapp._SCAN_META["15m"] = {"partial": True, "sweep_incomplete": True,
                              "answered_chunks": 0}
    assert dapp._rescan_delay_sec("15m") > 240.0

    # a sweep that wrapped and stayed truncated (errors in the tail): backoff too
    dapp._SCAN_META["15m"] = {"partial": True, "sweep_incomplete": False,
                              "answered_chunks": 69}
    assert dapp._rescan_delay_sec("15m") > 240.0


def test_a_partial_scan_never_shrinks_the_pair_list(monkeypatch):
    """The bug the user reported as "пропали все 15ти минутные графики".

    A resumed sweep read two chunks, the database was busy, and the pass
    answered nothing; that empty answer became `_SUMMARY_STORE`, so the selector
    had no 15m rows to render even though every table was still in Postgres.
    Serving is now monotonic: while a sweep is incomplete it may only ADD."""
    import pandas as pd

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_enabled", False)
    old = pd.DataFrame({
        "db_name": ["vol_15m_high", "vol_15m_high", "vol_15m_low"],
        "table_name": ["btc_usdt_on_okx", "eth_usdt_on_okx", "sol_usdt_on_bybit"],
        "ticker": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    })
    dapp._SUMMARY_STORE["15m"] = {"df": old, "meta": {"partial": True}, "at": 0.0}

    async def one_row(*a, **k):
        dapp._SCAN_META["15m"] = {"partial": True, "rows": 1, "tables": 8235}
        return pd.DataFrame({
            "db_name": ["vol_15m_low"],
            "table_name": ["sol_usdt_on_bybit"],
            "ticker": ["SOL/USDT"],
            # the row the sweep DID read is newer, so it must win per table
            "last_price": [11.5],
        })

    monkeypatch.setattr(dapp, "_load_summary", one_row)
    dapp._scan_summary_now("h", 1, "u", "p", "15m")

    served = dapp._SUMMARY_STORE["15m"]["df"]
    assert len(served) == 3
    assert sorted(served["table_name"]) == [
        "btc_usdt_on_okx", "eth_usdt_on_okx", "sol_usdt_on_bybit"]
    assert served[served["table_name"] == "sol_usdt_on_bybit"]["last_price"].iloc[0] == 11.5
    meta = dapp._SUMMARY_STORE["15m"]["meta"]
    assert meta["kept_rows"] == 2 and meta["rows"] == 3


def test_a_scan_that_answered_nothing_but_lists_tables_does_not_blank_it(monkeypatch):
    """`rendering 0/8235 tables` without a `partial` flag: the catalog cache
    names 8235 pair tables and the sweep read none of them. That is a scan that
    got nothing, not a database with nothing in it, so the previous list stays
    on screen — the user must not lose their charts to a busy minute."""
    import pandas as pd

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_enabled", False)
    dapp._SUMMARY_STORE["15m"] = {
        "df": pd.DataFrame({"db_name": ["a", "a"], "table_name": ["t1", "t2"],
                            "ticker": ["A/USDT", "B/USDT"]}),
        "meta": {"partial": True}, "at": 0.0}

    async def nothing(*a, **k):
        dapp._SCAN_META["15m"] = {"partial": False, "rows": 0, "tables": 8235}
        return pd.DataFrame()

    monkeypatch.setattr(dapp, "_load_summary", nothing)
    df = dapp._scan_summary_now("h", 1, "u", "p", "15m")
    assert len(df) == 2


def test_a_complete_scan_may_shrink_the_pair_list(monkeypatch):
    """The other half: monotonic serving is about TRUNCATED passes. A scan that
    read the whole database and found fewer tables is the truth — deleted pairs
    must leave the selector, or the dashboard would list ghosts forever."""
    import pandas as pd

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_enabled", False)
    dapp._SUMMARY_STORE["15m"] = {
        "df": pd.DataFrame({"db_name": ["a", "a"],
                            "table_name": ["t1", "t2"], "ticker": ["A/USDT", "B/USDT"]}),
        "meta": {"partial": True}, "at": 0.0}

    async def complete(*a, **k):
        dapp._SCAN_META["15m"] = {"partial": False, "rows": 1, "tables": 1}
        return pd.DataFrame({"db_name": ["a"], "table_name": ["t1"], "ticker": ["A/USDT"]})

    monkeypatch.setattr(dapp, "_load_summary", complete)
    df = dapp._scan_summary_now("h", 1, "u", "p", "15m")
    assert len(df) == 1
    assert "kept_rows" not in dapp._SUMMARY_STORE["15m"]["meta"]


def test_a_sweep_in_progress_does_not_expire_its_own_rows(monkeypatch):
    """`dash_scan_carryover_ttl_sec` retires rows a COMPLETE pass failed to
    confirm; it must never be what deletes rows a pass has not finished reading.

    With the sweep at chunk 23 of 69 and the 15m tier 15 minutes from finishing,
    the TTL aged out everything chunks 0..22 had answered, `all_rows` came back
    empty, and `_load_summary` served an empty frame as the pair list."""
    from dashboard import app as dapp

    monkeypatch.setattr(dapp.settings, "dash_scan_carryover_ttl_sec", 900.0)
    assert dapp._carry_ttl_sec(None) == 0.0          # one-shot scan: no carrying
    assert dapp._carry_ttl_sec({}) == 900.0          # last sweep wrapped: TTL rules
    assert dapp._carry_ttl_sec({"start": 23}) == float("inf")   # mid-sweep: never

    stale = {"t1": (time.time() - 100000.0, {"table_name": "t1", "ticker": "A/USDT"})}
    out = []
    n = dapp._merge_carried_rows(out, {"start": 23, "rows": stale}, {"t1"},
                                 now=time.time(), ttl=dapp._carry_ttl_sec({"start": 23}))
    assert n == 1 and out[0]["table_name"] == "t1"
    out = []
    n = dapp._merge_carried_rows(out, {"rows": stale}, {"t1"},
                                 now=time.time(), ttl=dapp._carry_ttl_sec({}))
    assert n == 0 and out == []
    # a table the catalog no longer has is dropped even mid-sweep
    out = []
    n = dapp._merge_carried_rows(out, {"start": 1, "rows": stale}, set(),
                                 now=time.time(), ttl=float("inf"))
    assert n == 0


def test_a_gone_table_leaves_the_cached_inventory(monkeypatch, capsys):
    """`relation "wbtc_usdt_on_bitget" does not exist` (1d/LOW, 3 chunks) is not
    a slow chunk: the pair was pruned or moved tier and the cached catalog is
    naming a table that is gone. Retrying it spends budget every pass; the entry
    has to leave the inventory until the catalog is re-read."""
    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp, "_MARKET_LOG_AT", {})
    tables = {"wbtc_usdt_on_bitget": {}, "btc_usdt_on_okx": {}}
    inventory = {"at": time.time(), "tables": tables}
    dapp._SCAN_INVENTORY[("h", 1, "liq_1d_low")] = inventory

    err = RuntimeError('relation "wbtc_usdt_on_bitget" does not exist')
    assert dapp.forget_missing_relations("liq_1d_low", "h", 1, tables, err) == [
        "wbtc_usdt_on_bitget"]
    assert "wbtc_usdt_on_bitget" not in tables and "btc_usdt_on_okx" in tables
    assert "wbtc_usdt_on_bitget" not in dapp._SCAN_INVENTORY[("h", 1, "liq_1d_low")]["tables"]
    assert "[scan] liq_1d_low: 1 table(s)" in capsys.readouterr().out

    # a TimeoutError mentions nothing: nothing is forgotten, the chunk retries
    tables2 = {"a": {}, "b": {}}
    assert dapp.forget_missing_relations("x", "h", 1, tables2, TimeoutError()) == []
    assert set(tables2) == {"a", "b"}


def test_a_dropped_cache_entry_does_not_blank_the_strip(monkeypatch, capsys):
    """`KeyError: '903f84a38…'` from `streamlit/runtime/caching/ttl_cache.py:125`
    was their console's red box: the ~1s live cache evicted an entry between
    Streamlit's lookup and its read, and the exception escaped into the
    `run_every` fragment, taking the whole health strip with it."""
    from dashboard import app as dapp

    monkeypatch.setattr(dapp, "_MARKET_LOG_AT", {})

    def race(*a, **k):
        raise KeyError("903f84a381ce6b5c396e5bfb00875bf1")

    def broken(*a, **k):
        raise ValueError("pool is closed")

    assert dapp._cached_read(race, "db", "ex", "BTC/USDT") is None
    out = capsys.readouterr().out
    assert "[live] race: the ~1s cache dropped its entry" in out
    with pytest.raises(ValueError):
        dapp._cached_read(broken)          # a real failure is still a real failure


def test_a_symbol_the_exchange_dropped_is_an_empty_answer(monkeypatch):
    """`BadSymbol: gate does not have market symbol 1000000BABYDOGE/USDT:USDT`
    printed one [stitch] line per chart page and cached a retry every
    `DASH_STITCH_RETRY_SEC`. An exchange that does not list the symbol is not
    failing to answer, it is answering "no candles" — so it is stored as the
    (empty) answer and the caption stops blaming the feed for it."""
    from dashboard import app as dapp

    monkeypatch.setattr(dapp, "_STITCH_CACHE", {})
    monkeypatch.setattr(dapp, "_MARKET_LOG_AT", {})
    monkeypatch.setattr(dapp, "_MARKET_GATE", {})

    class BadSymbol(Exception):
        pass

    def no_market(ccxt_id, symbol, timeframe, r0, r1):
        raise BadSymbol("gate does not have market symbol 1000000BABYDOGE/USDT:USDT")

    monkeypatch.setattr(dapp, "_fetch_missing_candles", no_market)
    errors: list = []
    out = dapp._fetch_missing_candles_cached(
        "gate", "1000000BABYDOGE/USDT:USDT", "15m", 1987067, 1987070, errors)
    assert out == []
    assert errors == []      # no "retry in Ns": there is nothing to retry
    ent = dapp._STITCH_CACHE[("gate", "1000000BABYDOGE/USDT:USDT", "15m", 1987067, 1987070)]
    assert ent[1] == "" and ent[3] == dapp._STITCH_OK_TTL

    # a timeout is still a failure, and is reported as one
    dapp._STITCH_CACHE.clear()

    def timed_out(ccxt_id, symbol, timeframe, r0, r1):
        raise TimeoutError("stitch budget (4.0s) used up")

    monkeypatch.setattr(dapp, "_fetch_missing_candles", timed_out)
    errors = []
    assert dapp._fetch_missing_candles_cached(
        "gate", "BTC/USDT", "15m", 1, 2, errors) == []
    assert errors and "used up" in errors[0]


def test_two_scans_never_hold_the_database_at_once(monkeypatch, capsys):
    """One scan per process — the two timeframes used to fan out over both of
    their databases, i.e. up to 8 pools of heavy UNION queries at once."""
    import pandas as pd
    import threading

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    held = threading.BoundedSemaphore(1)
    held.acquire()
    monkeypatch.setattr(dapp, "_SCAN_GATE", held)
    called = []

    async def fake_load(*a, **k):
        called.append(1)
        return pd.DataFrame()

    monkeypatch.setattr(dapp, "_load_summary", fake_load)
    monkeypatch.setattr(dapp, "_SUMMARY_STORE", {
        "15m": {"df": pd.DataFrame({"ticker": ["OLD"]}), "meta": {}, "at": 1.0}
    })

    df = dapp._scan_summary_now("h", 1, "u", "p", "15m")     # background path: no waiting
    assert list(df["ticker"]) == ["OLD"]
    assert called == [], "the second scan must not start while the first holds the gate"
    assert "holds the database" in capsys.readouterr().out


def test_databases_are_swept_one_at_a_time_by_default(monkeypatch):
    import asyncio

    from dashboard import app as dapp

    seen = []

    async def fake_scan(db, tier, *a, **k):
        seen.append(("start", db))
        await asyncio.sleep(0)
        seen.append(("end", db))
        return []

    monkeypatch.setattr(dapp, "_scan_database", fake_scan)
    monkeypatch.setattr(dapp, "_SCAN_META", {}, raising=False)
    monkeypatch.setattr(dapp.settings, "dash_scan_max_parallel_dbs", 1)
    monkeypatch.setattr(dapp.settings, "db_high_15m", "hi")
    monkeypatch.setattr(dapp.settings, "db_low_15m", "lo")
    asyncio.run(dapp._load_summary("h", 1, "u", "p", "15m"))
    assert seen == [("start", "hi"), ("end", "hi"), ("start", "lo"), ("end", "lo")]

    seen.clear()
    monkeypatch.setattr(dapp.settings, "dash_scan_max_parallel_dbs", 2)
    asyncio.run(dapp._load_summary("h", 1, "u", "p", "15m"))
    assert seen[:2] == [("start", "hi"), ("start", "lo")]        # opt-in concurrency


def test_a_busy_database_does_not_get_a_query_per_table(monkeypatch):
    """Timeout ⇒ skip the chunk. Type/schema error ⇒ recover, but bounded."""
    from dashboard import app as dapp

    schema = {
        f"p{i}_usdt_on_bybit": {"Timestamp": "bigint", "ticker": "text", "close": "numeric"}
        for i in range(30)
    }

    # (1) the server is loaded: no all-TEXT retry, no per-table recovery
    pool = _FakeScanPool(schema, break_unions="timeout")
    rows = _run_scan(monkeypatch, pool)
    assert rows == []
    assert len(pool.union_queries) == 1, "a timed-out chunk must not be re-queried"
    assert pool.per_table_reads == 0

    # (2) a broken chunk: recovered, but at most dash_scan_recovery_max_tables
    monkeypatch.setattr(dapp.settings, "dash_scan_recovery_max_tables", 7)
    pool2 = _FakeScanPool(schema, break_unions=True, break_rows=False)
    rows2 = _run_scan(monkeypatch, pool2)
    assert len(pool2.union_queries) == 2                 # native + all-TEXT retry
    assert pool2.per_table_reads == 7                    # capped, not 30
    assert len(rows2) == 7
    assert dapp._SCAN_META["db_test"]["partial"] is True  # 23 tables are unknown


def test_scan_budget_excludes_the_catalog_read(monkeypatch):
    """`in 76.0s` next to `25s budget exhausted` is not a contradiction, and
    pretending otherwise cost two rounds of tuning: the sweep and the catalog
    read are now timed and reported separately."""
    from dashboard import app as dapp

    schema = {
        f"p{i}_usdt_on_bybit": {"Timestamp": "bigint", "ticker": "text"} for i in range(2)
    }
    pool = _FakeScanPool(schema)
    _run_scan(monkeypatch, pool)
    meta = dapp._SCAN_META["db_test"]
    assert "sweep_seconds" in meta and "catalog_seconds" in meta
    assert meta["skipped_chunks"] == 0 and meta["partial"] is False


def test_a_database_that_refuses_connection_is_reported_not_empty(monkeypatch, capsys):
    """`return []` on a failed connect is the silence this app keeps having to
    unlearn: an unreachable database and a database with no pairs both render
    as an empty pair list, and only one of them is true."""
    import asyncio

    from dashboard import app as dapp

    async def boom(**kw):
        raise OSError("could not connect to server: Connection timed out")

    monkeypatch.setattr(dapp.asyncpg, "create_pool", boom)
    monkeypatch.setattr(dapp, "_SCAN_INVENTORY", {}, raising=False)
    rows = asyncio.run(dapp._scan_database("db_dead", "HIGH", "h", 1, "u", "p",
                                           pool_size=2, chunk_size=120, budget_sec=5.0))
    assert rows == []
    assert "connect failed" in capsys.readouterr().out
    meta = dapp._SCAN_META["db_dead"]
    assert meta["partial"] is True and "Connection timed out" in meta["error"]


def test_the_cold_path_waits_for_an_in_flight_scan_instead_of_joining_it(monkeypatch):  # noqa: E302
    """Queueing on the gate is allowed to pay off: if a scan lands while the
    render path waits, that result is the answer — starting a second sweep of
    the same database would be the exact behaviour this is here to remove."""
    import threading
    import time as _time

    import pandas as pd

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    gate = threading.BoundedSemaphore(1)
    gate.acquire()
    monkeypatch.setattr(dapp, "_SCAN_GATE", gate)
    scans = []

    async def fake_load(*a, **k):
        scans.append(1)
        return pd.DataFrame({"ticker": ["X"]})

    monkeypatch.setattr(dapp, "_load_summary", fake_load)

    def finish_someone_elses_scan():
        dapp._SUMMARY_STORE["15m"] = {
            "df": pd.DataFrame({"ticker": ["FROM_BG"]}), "meta": {},
            "at": _time.time(),
        }
        gate.release()

    timer = threading.Timer(0.05, finish_someone_elses_scan)
    timer.start()
    try:
        df = dapp._scan_summary_now("h", 1, "u", "p", "15m", gate_sec=2.0)
    finally:
        timer.cancel()
    assert list(df["ticker"]) == ["FROM_BG"]
    assert scans == [], "the queued caller must not launch a second scan"
    assert gate.acquire(blocking=False), "the gate was not returned"
    gate.release()


def test_budget_is_measured_after_the_semaphore(monkeypatch):
    """A queued chunk must not run a query authorized a minute earlier.

    69 chunks behind 6 connections: the sweep checked the budget when the chunk
    STARTED and then blocked on the semaphore, so every chunk that got a slot
    late still issued its query with the allowance it had been given at queueing
    time. That is how a 25 s sweep ran for 142 s while chart queries waited on
    the same database.
    """
    import asyncio
    import time as _time

    from dashboard import app as dapp

    schema = {
        f"p{i}_usdt_on_bybit": {"Timestamp": "bigint", "ticker": "text"} for i in range(4)
    }

    class _SlowPool(_FakeScanPool):
        def acquire(self, *a, **kw):
            outer = self

            class _Conn:
                async def fetch(self, query, *args):
                    if "pg_catalog" in query:
                        return [
                            {"table_name": t, "column_name": c, "data_type": dt}
                            for t, cols in outer.schema.items() for c, dt in cols.items()
                        ]
                    outer.union_queries.append(query)
                    await asyncio.sleep(1.5)      # the database is busy
                    return [outer._row(t) for t in outer.schema if f"'{t}'::text" in query]

                async def fetchrow(self, query, *args):  # pragma: no cover
                    raise AssertionError("a chunk skipped on budget must not recover")

            class _A:
                async def __aenter__(self):
                    return _Conn()

                async def __aexit__(self, *exc):
                    return False

            return _A()

    pool = _SlowPool(schema)

    async def fake_create_pool(**kw):
        return pool

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(dapp, "_SCAN_INVENTORY", {}, raising=False)
    monkeypatch.setattr(dapp.settings, "dash_scan_yield_gap_sec", 0.0)
    started = _time.time()
    rows = asyncio.run(
        dapp._scan_database("db_q", "HIGH", "h", 1, "u", "p",
                            pool_size=1, chunk_size=2, budget_sec=2.0)
    )
    elapsed = _time.time() - started

    assert len(rows) == 2                             # one chunk in, one not
    assert len(pool.union_queries) == 1, "a chunk whose budget evaporated while " \
                                         "queueing must never reach the database"
    meta = dapp._SCAN_META["db_q"]
    assert meta["skipped_chunks"] == 1 and meta["partial"] is True
    assert elapsed < 3.5, f"the sweep outlived its budget: {elapsed:.1f}s"


def test_neighbour_warm_primes_the_plain_page_too(monkeypatch):
    """The store key and the cache key are two different lookups on the render
    path: with stitching on, the warm used to fill only the first, so a flip
    made before the stitch landed paid two queries + a full HTML build."""
    from dashboard import app as dapp

    plain, stitched = [], []
    monkeypatch.setattr(dapp, "_render_chart_html_cached",
                        lambda **kw: plain.append(kw))
    monkeypatch.setattr(dapp, "_warm_stitched_page",
                        lambda key, page_kwargs: stitched.append((key, page_kwargs)) or 0)
    for k, v in {"db_host": "h", "db_port": 1, "db_user": "u", "db_pass": "p"}.items():
        monkeypatch.setattr(dapp, k, v)

    rows = {
        "15m": {"db_name": "db15", "table_name": "btc_usdt_on_bybit", "max_ts": 1},
        "1D": {"db_name": "db1d", "table_name": "btc_usdt_on_bybit", "max_ts": 2},
    }
    ctx = {"interval_ms": 0, "tick_port": None, "style": "Candlesticks", "height": 430,
           "volume": False, "stitch": True, "flat_fill": True}
    keys = dapp._warm_chart_pages(ctx, "BTC/USDT", "bybit", "bybit",
                                  rows["15m"], rows["1D"], 700, 200)

    assert len(plain) == 2 and len(stitched) == 2      # both charts, both pages
    assert keys == [k for k, _ in stitched]            # and the same keys
    assert all(p["stitch_enabled"] is False for p in plain)
    assert all("db_pass" not in p or p.get("db_pass") for p in plain)

    # the plain page must be keyed EXACTLY like the render path computes it,
    # otherwise the warm is dead weight — so compare against the helper itself
    page_kwargs, store_key = dapp._chart_page_args(
        rows["15m"], "15m", 700, "", "", 0, "bybit", "BTC/USDT", "bybit",
        "Candlesticks", 430, False, "", "", True,
    )
    assert store_key == keys[0]
    assert {k: v for k, v in plain[0].items() if k != "stitch_enabled"} == page_kwargs

    # with stitching off, only the page that will actually be shown is built
    plain.clear(); stitched.clear()
    dapp._warm_chart_pages({**ctx, "stitch": False}, "BTC/USDT", "bybit", "bybit",
                           rows["15m"], rows["1D"], 700, 200)
    assert len(plain) == 2 and stitched == []


def test_candles_use_the_live_pool_and_fall_back_on_error(monkeypatch, capsys):
    """First paint of a pair needs rows, not a fresh connection per row-set."""
    import asyncio
    import time as _time

    import pandas as pd

    from dashboard import app as dapp

    now = int(_time.time())
    rows = [{"ts": now - i * 900, "open": 1.0, "high": 1.1, "low": 0.9,
             "close": 1.05, "volume": 10.0} for i in range(4)][::-1]

    used = []
    monkeypatch.setattr(dapp, "_live_infra_or_none", lambda: {
        "submit_recent": lambda db, tbl, lim, timeout=8.0: used.append((db, tbl, lim)) or {"rows": rows}
    })

    async def never(*a, **k):  # pragma: no cover
        raise AssertionError("the direct path must not run when the pool answered")

    monkeypatch.setattr(dapp, "_load_candles", never)
    df = dapp.load_candles_cached.__wrapped__("h", 1, "u", "p", "db1", "t1", 4)
    assert len(df) == 4 and used == [("db1", "t1", 4)]
    assert list(df["ts"]) == sorted(df["ts"])              # oldest first, as the chart wants

    # a pool problem must never look like an empty table: fall back, and say so
    monkeypatch.setattr(dapp, "_live_infra_or_none", lambda: {
        "submit_recent": lambda *a, **k: {"err": "InterfaceError: connection is closed"}
    })
    sentinel = pd.DataFrame({"ts": [now], "open": [1.0], "high": [1.0], "low": [1.0],
                             "close": [1.0], "volume": [1.0]})

    async def direct(db_name, table_name, limit, *a):
        return sentinel

    monkeypatch.setattr(dapp, "_load_candles", direct)
    df2 = dapp.load_candles_cached.__wrapped__("h", 1, "u", "p", "db1", "t1", 1)
    assert df2 is sentinel
    out = capsys.readouterr().out
    assert "pool path failed" in out and "retrying directly" in out

    # and an infra that is not up at all (demo mode, startup race) is quiet
    monkeypatch.setattr(dapp, "_live_infra_or_none", lambda: None)
    assert dapp.load_candles_cached.__wrapped__("h", 1, "u", "p", "db1", "t1", 1) is sentinel


# ---------------------------------------------------------------------------
# Warming: a head start for the next click, never a competitor with this one.
#
# The prefetch was the load: it was re-armed by every rerun, and it fetched
# hundreds of missing candles for the dead spot tables left by the spot→perp
# migration. See _warm_pair_due / _pair_is_frozen.
# ---------------------------------------------------------------------------


def test_a_pair_is_warmed_once_per_page_lifetime(monkeypatch):
    import time as _time

    from dashboard import app as dapp

    monkeypatch.setattr(dapp, "_WARMED_AT", {}, raising=False)
    monkeypatch.setattr(dapp, "CHART_PAGE_TTL_SEC", 0.15)

    assert dapp._warm_pair_due("BTC/USDT", "bybit", _time.time()) is True
    assert dapp._warm_pair_due("BTC/USDT", "bybit", _time.time()) is False   # just done
    assert dapp._warm_pair_due("ETH/USDT", "bybit", _time.time()) is True      # another pair
    _time.sleep(0.2)
    assert dapp._warm_pair_due("BTC/USDT", "bybit", _time.time()) is True      # expired


def test_frozen_spot_leftovers_are_not_fetched_from_the_exchange(monkeypatch):
    """Pairs the collector stopped writing get their page primed from the DB and
    no exchange traffic: their missing history is pages of candles nobody is
    about to look at, and every fetch steals a connection from a live click."""
    import time as _time

    from dashboard import app as dapp

    now = int(_time.time())      # real clock: _pair_is_frozen compares against it
    assert dapp._pair_is_frozen({"max_ts": now - 60}, None, now) is False
    assert dapp._pair_is_frozen({"max_ts": now - 3 * 86400}, None, now) is True
    assert dapp._pair_is_frozen(None, {"max_ts": (now - 60) * 1000}, now) is False   # ms table
    assert dapp._pair_is_frozen(None, None, now) is True
    assert dapp._pair_is_frozen({"max_ts": None}, {"max_ts": "junk"}, now) is True

    calls = []
    monkeypatch.setattr(dapp.settings, "dash_warm_delay_sec", 0.0)
    monkeypatch.setattr(dapp, "_warm_yield_to_clicks", lambda *a, **k: True)
    monkeypatch.setattr(dapp, "load_candles_cached", lambda *a, **k: __import__("pandas").DataFrame())
    monkeypatch.setattr(dapp, "_fetch_missing_candles_cached", lambda *a, **k: calls.append(("range", a)))
    for fn in ("_fetch_ticker_cached", "_fetch_orderbook_top", "_fetch_trade_tape"):
        monkeypatch.setattr(dapp, fn, lambda *a, _n=fn: calls.append((_n, a)))
    monkeypatch.setattr(dapp, "_warm_live_snapshot", lambda *a, **k: calls.append(("live", a)))
    stitched_pages = []
    monkeypatch.setattr(dapp, "_warm_stitched_page",
                        lambda *a, **k: stitched_pages.append(a))
    monkeypatch.setattr(
        dapp, "_warm_chart_pages",
        lambda *a, **k: calls.append(("pages", k.get("skip_stitch"))),
    )
    for k, v in {"db_host": "h", "db_port": 1, "db_user": "u", "db_pass": "p"}.items():
        monkeypatch.setattr(dapp, k, v)

    frozen = {"db_name": "db1", "table_name": "old_spot_usdt_on_gateio", "max_ts": now - 9 * 86400}
    dapp._warm_pair_caches("OLD/USDT", "gateio", "gateio", frozen, None, 700, 700, 14, True)
    assert calls == [("pages", True)]               # primed, with the stitch skipped
    assert stitched_pages == []                     # no exchange page-walk from the warm

    # the same pair while it is still being written: ranges are pre-fetched, but
    # only a few per timeframe — a head start on the stitch, not the stitch
    calls.clear()
    import pandas as pd

    step = 900
    holes = pd.DataFrame({"ts": [now - 40 * step, now - 30 * step, now - 20 * step,
                                 now - 10 * step]})
    monkeypatch.setattr(dapp, "load_candles_cached", lambda *a, **k: holes)
    live = {"db_name": "db1", "table_name": "btc_usdt_on_bybit", "max_ts": now - 60}
    dapp._warm_pair_caches("BTC/USDT", "bybit", "bybit", live, None, 700, 700, 14, False)
    kinds = [c[0] for c in calls]
    assert kinds.count("range") <= 3, "the warm must not become the whole stitch"
    assert "live" not in kinds                    # full_live=False for far neighbours
    assert ("pages", False) in calls              # and here the stitch IS prepared


def test_warm_thread_reports_a_cancelled_exchange_call_and_survives(monkeypatch, capsys):
    """`asyncio.run` re-raises the inner CancelledError, which is a
    BaseException — so an ordinary `except Exception` let it kill the thread and
    print a traceback instead of warming the page it was asked to warm."""
    import asyncio
    import time as _time

    from dashboard import app as dapp

    monkeypatch.setattr(dapp, "_BG_RUNNING", set(), raising=False)
    done = []

    def cancelled():
        raise asyncio.CancelledError()

    dapp._bg(("t", 1), cancelled)
    for _ in range(100):
        if ("t", 1) not in dapp._BG_RUNNING:
            break
        _time.sleep(0.01)
    out = capsys.readouterr().out
    assert "[bg]" in out and "CancelledError" in out
    assert ("t", 1) not in dapp._BG_RUNNING          # the key was released, not leaked

    dapp._bg(("t", 2), lambda: done.append(1))
    for _ in range(100):
        if done:
            break
        _time.sleep(0.01)
    assert done == [1]                                # and _bg still works after


def test_a_slow_switch_says_where_the_time_went(capsys, monkeypatch):
    """The diagnostic that should have existed two rounds ago.

    Three opposite causes hide behind "switching is slow": a warm that never
    ran (Streamlit), a slow candle query (database), a slow HTML/JSON build
    (CPU). The line names which one it was, so the next report can be fixed
    instead of guessed at.
    """
    import time as _time

    from dashboard import app as dapp

    monkeypatch.setattr(dapp.settings, "dash_switch_report_ms", 20)
    dapp._report_switch("BTC/USDT", "15m", "cached page",
                        _time.perf_counter() - 0.5, _time.perf_counter() - 0.1)
    out = capsys.readouterr().out
    assert "[switch] BTC/USDT 15m" in out and "cached page" in out and "ms" in out

    # fast renders stay silent — this is a diagnostic, not a log of every frame
    dapp._report_switch("BTC/USDT", "15m", "warmed page", _time.perf_counter(), _time.perf_counter())
    assert capsys.readouterr().out == ""

    # 0 logs everything, which is what DASH_SWITCH_REPORT_MS=0 is for
    monkeypatch.setattr(dapp.settings, "dash_switch_report_ms", 0)
    dapp._report_switch("ETH/USDT", "1D", "warmed page", _time.perf_counter(), _time.perf_counter())
    assert "[switch] ETH/USDT 1D" in capsys.readouterr().out


def test_the_build_counter_tells_a_cache_hit_from_a_query(monkeypatch):
    """"cached page" vs "built from DB" is only knowable from inside the cached
    function, so the function counts its own runs."""
    import pandas as pd

    from dashboard import app as dapp

    monkeypatch.setattr(dapp, "load_candles_cached", lambda *a, **k: pd.DataFrame())
    for k, v in {"db_host": "h", "db_port": 1, "db_user": "u", "db_pass": "p"}.items():
        monkeypatch.setattr(dapp, k, v)

    before = dapp._CHART_BUILDS
    # __wrapped__ bypasses st.cache_data, which is exactly the "we did query" case
    out = dapp._render_chart_html_cached.__wrapped__(
        db_host="h", db_port=1, db_user="u", db_pass="p", db_name="db1",
        table_name="t1", max_ts=1, tf_label="15m", limit=10, merge_db="", merge_table="",
        merge_limit=0, ccxt_id="bybit", sym_ticker="BTC/USDT", sym_ex="bybit",
        chart_style="Candlesticks", chart_height=430, show_volume=False,
        stitch_enabled=False, live_poller_js="", history_loader_js="", flat_fill=True,
    )
    assert out is None                      # no candles → the caller's empty-state path
    assert dapp._CHART_BUILDS == before + 1


# ---------------------------------------------------------------------------
# a budget-limited sweep has to make progress, not repeat itself
# ---------------------------------------------------------------------------
class _BusyPool:
    """Answers only the chunks whose tables are in `allow`.

    The rest hit the wall the way the rest of a sweep does when the budget runs
    out on a loaded database: a timeout, i.e. skipped, not retried.
    """

    def __init__(self, schema, allow):
        self.schema = schema
        self.allow = set(allow)
        self.union_tables: list = []

    def acquire(self, *a, **kw):
        outer = self

        class _Conn:
            async def fetch(self, query, *args):
                if "pg_catalog" in query:
                    return [
                        {"table_name": t, "column_name": c, "data_type": dt}
                        for t, cols in outer.schema.items() for c, dt in cols.items()
                    ]
                hit = [t for t in outer.schema if f"'{t}'::text" in query]
                outer.union_tables.append(tuple(hit))
                if not hit or not set(hit) <= outer.allow:
                    import asyncio as _a
                    raise _a.TimeoutError()
                return [{"table_name": t, "max_ts": 1_787_700_000,
                         "Timestamp": 1_787_700_000} for t in hit]

        class _Acq:
            async def __aenter__(self):
                return _Conn()

            async def __aexit__(self, *exc):
                return False

        return _Acq()

    async def close(self):
        pass


def test_a_partial_sweep_resumes_and_carries_what_it_already_read(monkeypatch):
    import asyncio

    from dashboard import app as dapp

    names = [f"t{i}_usdt_on_bybit" for i in range(6)]
    schema = {n: {"Timestamp": "timestamptz"} for n in names}
    state: dict = {}
    monkeypatch.setattr(dapp, "_SCAN_INVENTORY", {}, raising=False)
    monkeypatch.setattr(dapp.settings, "dash_scan_carryover_ttl_sec", 3600.0)

    def sweep(allow):
        """One budget-limited sweep: only `allow`'s tables get answered."""
        pool = _BusyPool(schema, allow)

        async def fake_create_pool(**kw):
            return pool

        monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)
        rows = asyncio.run(dapp._scan_database(
            "db_rot", "HIGH", "h", 1, "u", "p",
            pool_size=1, chunk_size=2, budget_sec=10.0, sweep=state,
        ))
        return {r["table_name"] for r in rows}, rows, pool

    # 6 tables, 3 chunks, one chunk answerable per sweep
    seen, rows, pool = sweep(names[0:2])
    assert seen == set(names[0:2]) and state["start"] == 1
    assert pool.union_tables[0] == tuple(names[0:2])
    # a pass that cannot vouch for 4 of the 6 tables says so, and names them
    assert dapp._SCAN_META["db_rot"]["partial"] is True
    assert dapp._SCAN_META["db_rot"]["missing_tables"] == 4

    seen, rows, pool = sweep(names[2:4])
    # resumed at chunk 1 instead of restarting, and the first chunk's rows are
    # still in the list — the pair list only ever grows
    assert pool.union_tables[0] == tuple(names[2:4]), "the sweep must continue, not restart"
    assert seen == set(names[0:4]), seen
    assert state["start"] == 2

    seen, rows, pool = sweep(names[4:6])
    assert seen == set(names)
    # ...and THIS pass is complete, so the cursor is cleared: every table is
    # vouched for (read now, or carried from a pass of the same sweep that is
    # inside the carry window). `partial` used to be `bool(skipped)` — measured
    # per pass — so a tier whose budget never covers the whole database stayed
    # "incomplete" forever, kept the badge up, re-read 6 394 of 8 312 tables on
    # every retry and never got the cheap rescan throttle.
    assert "start" not in state, state
    assert dapp._SCAN_META["db_rot"]["partial"] is False
    assert dapp._SCAN_META["db_rot"]["missing_tables"] == 0
    # carried rows are complete rows: they know their database and tier, so the
    # table they are painted into needs no special case
    assert all(r["db_name"] == "db_rot" and r["volume_tier"] == "HIGH" for r in rows)

    meta = dapp._SCAN_META["db_rot"]
    assert meta["answered_chunks"] == 1 and meta["carried_rows"] == 4


def test_a_stale_carry_may_not_claim_the_list_is_complete(monkeypatch):
    """The other half of the same rule, and the reason `partial` is not simply
    "the union is covered".

    Rows read by a sweep in progress are kept however old, so the pair list can
    only grow. But being KEPT is not being BELIEVED: a row may vouch for the
    list only while it is younger than `DASH_SCAN_CARRYOVER_TTL_SEC`. Without
    that bound a database busy for a day would call a frozen list complete,
    drop to the 5-minute rescan floor and stop looking at the tables it has not
    read — a quietly stale dashboard is worse than an honest "incomplete"."""
    import asyncio

    from dashboard import app as dapp

    names = [f"t{i}_usdt_on_bybit" for i in range(4)]
    schema = {n: {"Timestamp": "timestamptz"} for n in names}
    state: dict = {}
    monkeypatch.setattr(dapp, "_SCAN_INVENTORY", {}, raising=False)
    # carrying stays ON mid-sweep (that is `_carry_ttl_sec`), but nothing counts
    # as a fresh answer any more
    monkeypatch.setattr(dapp.settings, "dash_scan_carryover_ttl_sec", 0.0)

    def sweep(allow):
        pool = _BusyPool(schema, allow)

        async def fake_create_pool(**kw):
            return pool

        monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)
        rows = asyncio.run(dapp._scan_database(
            "db_stale", "LOW", "h", 1, "u", "p",
            pool_size=1, chunk_size=2, budget_sec=10.0, sweep=state,
        ))
        return {r["table_name"] for r in rows}

    assert sweep(names[0:2]) == set(names[0:2])
    # the list grows (rows are carried) while the CLAIM stays partial
    assert sweep(names[2:4]) == set(names)
    meta = dapp._SCAN_META["db_stale"]
    assert meta["rows"] == 4 and meta["carried_rows"] == 2
    assert meta["partial"] is True and meta["missing_tables"] == 2
    assert "start" in state             # and the sweep keeps its place (unlike a
    assert state["start"] == 0          # complete pass, which clears the cursor)

    assert dapp._carry_ttl_sec(state) == float("inf")     # still not expiring them
    assert dapp._carry_ttl_sec({}) == 0.0                   # wrapped: the TTL rules


def test_unanswered_tables_counts_an_empty_table_as_an_answer():
    from dashboard.app import _unanswered_tables

    tables = {"a": {}, "b": {}, "c": {}}
    # a chunk query that came back with no rows IS an answer (a pair table with
    # no candles is normal, and it must not keep the tier "incomplete" forever)
    assert set(_unanswered_tables(tables, {"a", "b", "c"}, set())) == set()
    assert set(_unanswered_tables(tables, {"a"}, {"b"})) == {"c"}
    assert set(_unanswered_tables(tables, set(), set())) == {"a", "b", "c"}


def test_carried_rows_expire_and_follow_the_inventory():
    from dashboard.app import _merge_carried_rows

    now = 1000.0
    sweep = {"rows": {"a": (990.0, {"table_name": "a"}),      # read this sweep
                      "g": (960.0, {"table_name": "g"}),      # 40 s old, within ttl
                      "b": (50.0, {"table_name": "b"}),       # older than the ttl
                      "c": (999.0, {"table_name": "c"})}}     # dropped from the db
    out = [{"table_name": "a"}, {"table_name": "d"}]
    added = _merge_carried_rows(out, sweep, ["a", "b", "d", "g"], now=now, ttl=60.0)
    # 'a' is not duplicated, 'b' is too old to show, 'c' no longer exists
    assert [r["table_name"] for r in out] == ["a", "d", "g"]
    assert added == 1
    assert set(sweep["rows"]) == {"a", "d", "g"}
    assert all(at == now for at, _ in sweep["rows"].values())

    # ttl=0 means "carry nothing": the knob in .env.example
    out2 = [{"table_name": "e"}]
    assert _merge_carried_rows(out2, {"rows": {"f": (now, {"table_name": "f"})}},
                              ["e", "f"], now=now, ttl=0.0) == 0
    assert [r["table_name"] for r in out2] == ["e"]


def test_a_scan_pushed_aside_by_another_tier_retries_soon_then_gives_up(monkeypatch):
    import threading

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_snapshot_refresh_sec", 120.0)
    monkeypatch.setattr(dapp.settings, "dash_scan_defer_retry_sec", 8.0)
    held = threading.BoundedSemaphore(1)
    held.acquire()                      # another timeframe's sweep, for the whole thing
    monkeypatch.setattr(dapp, "_SCAN_GATE", held)

    assert dapp._scan_summary_now("h", 1, "u", "p", "1d").empty
    assert dapp._SCAN_DEFER_TRIES["1d"] == 1 and dapp._SCAN_DEFERRED_AT["1d"]
    # the truncated-scan backoff would say 240 s; a skipped scan queries nothing
    assert dapp._rescan_delay_sec("1d") == 8.0

    dapp._SCAN_ATTEMPTS["1d"] = 2        # a real streak of truncated scans: 480 s
    assert dapp._rescan_delay_sec("1d") == 8.0

    for _ in range(3):
        dapp._scan_summary_now("h", 1, "u", "p", "1d")
    assert dapp._SCAN_DEFER_TRIES["1d"] == 4
    # after a bounded number of quick tries, patience resumes: the other tier is
    # not momentarily busy, it IS busy, and retrying is how that becomes load
    assert dapp._rescan_delay_sec("1d") == 480.0

    # once a scan gets through, both markers are gone
    held.release()
    dapp._scan_summary_now("h", 1, "u", "p", "1d")
    assert "1d" not in dapp._SCAN_DEFERRED_AT and "1d" not in dapp._SCAN_DEFER_TRIES


def test_the_collector_skips_the_currency_table_too(monkeypatch):
    """Same defect on the collector side, bigger consequence: the engine wraps
    `load_markets()` in a 30 s hard wait, and gate's extra currency round trip is
    what pushed it over — `load_markets for gateio failed (attempt 1/3):
    TimeoutError()` means that exchange collects NOTHING for the cycle while the
    log looks healthy. `CCXT_FETCH_CURRENCIES=true` brings the request back."""
    import sys
    from config.settings import settings

    sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.exchanges.client import create_exchange

    monkeypatch.setattr(settings, "ccxt_fetch_currencies", False, raising=False)
    assert create_exchange("gate").has.get("fetchCurrencies") is False

    monkeypatch.setattr(settings, "ccxt_fetch_currencies", True, raising=False)
    assert create_exchange("gate").has.get("fetchCurrencies") is not False


def test_a_later_pass_of_a_building_sweep_may_run_longer_while_the_page_is_idle(monkeypatch, capsys):
    """Their log, first passes after a restart:

        [scan] 15m_low_vol…: 25s budget exhausted — rendering 0/8245 tables
               (69 chunk(s) skipped on budget/busy db)

    69 chunks of 120 tables at ~6s a chunk is ~6 min of ONE-OFF reading, which
    25s slices cover at ~4 chunks per pass — 17 passes of "the list is
    incomplete". The budget is therefore two-valued: the first paint of the
    process keeps the short one (the page must still open fast), while a later
    pass of a sweep that is still building runs longer ONLY while nobody has
    touched the page. No extra query is asked by this: chunks rotate and the
    catalog is reused mid-sweep, so a longer pass is fewer passes."""
    import asyncio
    import time as _time

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_scan_budget_sec", 1.0)
    monkeypatch.setattr(dapp.settings, "dash_scan_budget_idle_sec", 9.0)
    schema = {f"p{i}_usdt_on_bybit": {"Timestamp": "timestamptz"} for i in range(3)}
    pool = _FakeScanPool(schema)

    async def fake_create_pool(**kw):
        return pool

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(dapp, "_LAST_INTERACTION_AT", _time.time() - 100.0)
    st = {}
    # an already-known listing: the pass that pays for it is the first paint
    dapp._SCAN_INVENTORY[("h", 1, "db_idle")] = {"at": _time.time(), "tables": dict(schema)}

    asyncio.run(dapp._scan_database("db_idle", "LOW", "h", 1, "u", "p",
                                    pool_size=1, chunk_size=1, sweep=st))
    first = dict(dapp._SCAN_META["db_idle"])
    asyncio.run(dapp._scan_database("db_idle", "LOW", "h", 1, "u", "p",
                                    pool_size=1, chunk_size=1, sweep=st))
    later = dict(dapp._SCAN_META["db_idle"])
    assert first["budget"] == 1.0, "the first paint must not be slowed by this"
    assert later["budget"] == 9.0, later

    # …and a click brings the short budget back
    monkeypatch.setattr(dapp, "_LAST_INTERACTION_AT", _time.time())
    asyncio.run(dapp._scan_database("db_idle", "LOW", "h", 1, "u", "p",
                                    pool_size=1, chunk_size=1, sweep=st))
    assert dapp._SCAN_META["db_idle"]["budget"] == 1.0
    capsys.readouterr()


def test_a_chunk_skipped_for_being_busy_is_counted_as_its_own_chunk(monkeypatch, capsys):
    """The retry counters were deduplicated by the error TAG, and every chunk of
    120 tables was tagged `chunk[120]` — so a pass in which 68 chunks were
    skipped on a loaded server printed `2 chunk(s) skipped (db busy)` next to
    `69 chunk(s) skipped on budget/busy db` in the very next line. One of the
    two must be wrong, and it was the one the user read first: chunks are tagged
    by position now."""
    import asyncio

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)
    monkeypatch.setattr(dapp.settings, "dash_scan_yield_gap_sec", 0.0)
    schema = {f"p{i}_usdt_on_bybit": {"Timestamp": "timestamptz"} for i in range(3)}
    pool = _FakeScanPool(schema, break_unions="timeout")
    pool.schema = schema

    async def fake_create_pool(**kw):
        return pool

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)
    dapp._SCAN_INVENTORY[("h", 1, "db_tags")] = {
        "at": __import__("time").time(), "tables": dict(schema)}
    rows = asyncio.run(dapp._scan_database("db_tags", "LOW", "h", 1, "u", "p",
                                           pool_size=1, chunk_size=1, budget_sec=10.0,
                                           sweep={}))
    out = capsys.readouterr().out
    assert rows == []
    # the FIRST number is the one the user reads; both must agree now
    assert "\u2014 3 chunk(s) skipped (db busy)" in out, out
    tags = {ln for ln in out.splitlines() if "chunk[1]" in ln or "chunk[2]" in ln}
    assert tags, "the chunks must be named by position: " + out


def test_a_listing_that_fails_is_a_failed_pass_and_not_a_crash(monkeypatch, capsys):
    """Without a listing there is no chunk to run, so this is the one read a
    sweep cannot work around — but it must not escape the cached load: an
    exception there loses the whole page, and returning `[]` loses the PAIR
    LIST, which is what "15минутка пропала" is. So: partial, cursor untouched,
    the last listing still counted, and a line saying so."""
    import asyncio

    from dashboard import app as dapp

    _summary_stores(monkeypatch, dapp)

    async def fake_create_pool(**kw):
        return _FakeScanPool({})

    async def bad_inventory(*a, **kw):
        raise RuntimeError("catalog statement timeout")

    monkeypatch.setattr(dapp.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(dapp, "_table_inventory", bad_inventory)
    st = {"start": 14, "rows": {}}
    rows = asyncio.run(dapp._scan_database("db_cat", "LOW", "h", 1, "u", "p",
                                           pool_size=1, chunk_size=1, budget_sec=5.0,
                                           sweep=st))
    assert rows == []
    meta = dapp._SCAN_META["db_cat"]
    assert meta["partial"] is True and "listing" in meta["error"], meta
    assert st["start"] == 14, "a failed read must not move the sweep cursor"
    out = capsys.readouterr().out
    assert "the catalog listing failed" in out and "not 'no pairs'" in out


def test_the_badge_quotes_the_database_that_is_still_being_read(monkeypatch):
    """15m/HIGH finished in one chunk, 15m/LOW is at 0/69 — and the badge said
    "chunk 0/70" because the tier SUMMED both chunk counts. A diagnostic that
    miscounts its own progress is how a converging sweep gets read as a stalled
    one, so the position comes from the database that is actually starving."""
    import asyncio

    import pandas as pd

    from dashboard import app as dapp

    monkeypatch.setattr(dapp.settings, "db_high_15m", "db_hi")
    monkeypatch.setattr(dapp.settings, "db_low_15m", "db_lo")
    dapp._SCAN_SWEEP_STATE.clear()
    dapp._SCAN_SWEEP_STATE["db_lo"] = {"start": 14}

    async def fake_scan(db_name, tier, *a, **kw):
        # The rows do not matter here: the tier meta is written before the frame
        # is assembled, and it is the meta the badge reads.
        if tier == "HIGH":
            dapp._SCAN_META[db_name] = {
                "tables": 75, "rows": 75, "chunks": 1, "chunk_start": 0,
                "missing_tables": 0, "answered_chunks": 1, "partial": False,
                "carried_rows": 0, "budget": 25.0, "carry_ttl": 900.0,
                "seconds": 3.0, "sweep_seconds": 1.0, "catalog_seconds": 2.0,
            }
            return []
        dapp._SCAN_META[db_name] = {
            "tables": 8245, "rows": 0, "chunks": 69, "chunk_start": 14,
            "missing_tables": 8245, "answered_chunks": 1, "partial": True,
            "carried_rows": 0, "budget": 90.0, "carry_ttl": float("inf"),
            "seconds": 120.0, "sweep_seconds": 92.0, "catalog_seconds": 27.3,
        }
        return []

    monkeypatch.setattr(dapp, "_scan_database", fake_scan)
    df = asyncio.run(dapp._load_summary("h", 1, "u", "p", "15m"))
    m = dapp._SCAN_META["15m"]
    assert m["starving_chunks"] == 69 and m["starving_chunk_start"] == 14, m
    assert m["budget"] == 90.0, "the badge must quote the budget the pass ran on"
    assert m["sweep_incomplete"] is True
