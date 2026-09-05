"""
Memory fix: the engines used to create a fresh ccxt instance per exchange on
EVERY cycle (plus one more for the pre-count), re-download tens of MB of
market JSON and throw it all away — allocator churn ratcheted RSS up until
the machine swapped. Exchanges are now long-lived instances: markets are
reloaded at most every MARKETS_TTL_SECONDS, instances rotated by age.
"""
import asyncio
import time

import pytest

from src.core import updater_15m, updater


class _FakeEx:
    def __init__(self):
        self.markets = {}
        self.closed = False

    async def load_markets(self, reload=False):
        self.markets = {"BTC/USDT:USDT": {}}
        return self.markets

    async def close(self):
        self.closed = True


def _patch_15m(monkeypatch, made, loads=None):
    monkeypatch.setattr(
        updater_15m, "_make_exchange", lambda cid: (made.append(cid), _FakeEx())[1]
    )

    async def fake_loader(exchange, ccxt_name, attempts=3, timeout=30.0, reload=False):
        if loads is not None:
            loads.append(reload)
        await exchange.load_markets(reload)
        return True

    monkeypatch.setattr(updater_15m, "load_markets_retries", fake_loader)


def test_15m_exchange_reused_across_cycles(monkeypatch):
    made = []
    _patch_15m(monkeypatch, made)
    updater_15m._EXCHANGES.clear()
    ex1 = asyncio.run(updater_15m.get_persistent_exchange("bybit"))
    ex2 = asyncio.run(updater_15m.get_persistent_exchange("bybit"))
    assert ex1 is ex2
    assert len(made) == 1  # second cycle must NOT build a new instance


def test_15m_markets_reload_forced_when_empty(monkeypatch):
    made, loads = [], []
    _patch_15m(monkeypatch, made, loads)
    updater_15m._EXCHANGES.clear()
    ex = asyncio.run(updater_15m.get_persistent_exchange("bybit"))
    ex.markets = {}  # markets lost -> reload forced on the SAME instance
    ex2 = asyncio.run(updater_15m.get_persistent_exchange("bybit"))
    assert ex2 is ex and ex2.markets
    assert loads == [True, True]


def test_15m_instance_rotated_by_age(monkeypatch):
    made = []
    _patch_15m(monkeypatch, made)
    updater_15m._EXCHANGES.clear()
    ex1 = asyncio.run(updater_15m.get_persistent_exchange("bybit"))
    updater_15m._EXCHANGES["bybit"]["born_at"] = (
        time.time() - updater_15m.EXCHANGE_MAX_AGE_SECONDS - 1
    )
    ex2 = asyncio.run(updater_15m.get_persistent_exchange("bybit"))
    assert ex2 is not ex1
    assert ex1.closed is True  # the aged instance is properly closed
    assert len(made) == 2


def test_15m_hard_failure_returns_none_and_resets(monkeypatch):
    made = []
    _patch_15m(monkeypatch, made)

    async def bad_loader(*args, **kwargs):
        return False

    monkeypatch.setattr(updater_15m, "load_markets_retries", bad_loader)
    updater_15m._EXCHANGES.clear()
    assert asyncio.run(updater_15m.get_persistent_exchange("bybit")) is None
    assert updater_15m._EXCHANGES == {}  # dropped — next cycle starts clean


def test_1d_exchange_reused(monkeypatch):
    made = []
    monkeypatch.setattr(
        updater, "create_exchange", lambda cid: (made.append(cid), _FakeEx())[1]
    )

    async def fake_loader(exchange, name, attempts=3, timeout=30.0, reload=False):
        await exchange.load_markets(reload)
        return True

    monkeypatch.setattr(updater, "load_markets_with_retry", fake_loader)
    updater._EXCHANGES.clear()
    ex1 = asyncio.run(updater.get_persistent_exchange("bybit", "bybit"))
    ex2 = asyncio.run(updater.get_persistent_exchange("bybit", "bybit"))
    assert ex1 is ex2
    assert len(made) == 1


def test_release_memory_is_safe_noop():
    updater.release_memory()      # must never raise (any OS)
    updater_15m.release_memory()


def test_option_markets_are_never_asked_for():
    """ccxt's market load asks for every category an exchange declares and —
    synchronously — one after another inside a single timeout, so ONE slow
    category means no markets at all. gate's list is spot, swap, future, option:
    an exchange this project never trades options on, and their
    `BLUR/USDT:USDT` perp sat in "waiting for the first tick" while the *spot*
    leg timed out. Dropping a category drops a request; that is the only kind of
    speed-up allowed here.

    The trim is a DENY list because the names are not universal — bybit calls its
    linear perpetuals `linear`, so an allow-list of ["spot","swap"] would have
    deleted every bybit perp from the market cache and produced a `BadSymbol`
    that reads like a delisting.
    """
    import ccxt

    from src.exchanges.client import apply_market_type_trim

    gate = ccxt.gate()
    assert gate.options["fetchMarkets"]["types"] == ["spot", "swap", "future", "option"]
    assert apply_market_type_trim(gate) == ["spot", "swap", "future"]
    assert gate.options["fetchMarkets"]["types"] == ["spot", "swap", "future"]

    bybit = ccxt.bybit()                      # perps are "linear" here, and must stay
    kept = apply_market_type_trim(bybit)
    assert kept == ["spot", "linear", "inverse"], kept
    assert bybit.options["fetchMarkets"]["types"] == kept
    # the sibling keys of that dict are ccxt's own settings: preserved, not
    # replaced by a dict we invented
    assert "usePrivateInstrumentsInfo" in bybit.options["fetchMarkets"]

    mexc = ccxt.mexc()                        # flags, not a list -> untouched
    before = mexc.options["fetchMarkets"]["types"]
    assert apply_market_type_trim(mexc) == []
    assert mexc.options["fetchMarkets"]["types"] == before

    okx = ccxt.okx()
    assert apply_market_type_trim(okx, skip="") == [], "the knob off = ccxt default"
    assert okx.options["fetchMarkets"]["types"] == ["spot", "future", "swap", "option"]


def test_the_async_client_says_which_route_it_took(monkeypatch, caplog):
    """`SOCKS5_PROXY` has a non-empty default, so "is gate loaded through the VPN?"
    was a question about a line of code nobody had read. It is now a line in the
    log — and the dashboard can ask for the other route explicitly, because the
    sync half of the dashboard CANNOT use the tunnel (ccxt hands a `socks5://` URL
    to requests, which needs PySocks; only aiohttp_socks is installed)."""
    import logging

    from config.settings import settings
    from src.exchanges.client import create_exchange

    monkeypatch.setattr(settings, "socks5_proxy", "socks5://127.0.0.1:10808",
                        raising=False)
    with caplog.at_level(logging.INFO, logger="exchange_client"):
        via = create_exchange("gate")
        direct = create_exchange("gate", use_proxy=False)
    assert getattr(via, "socks_proxy", None) == "socks5://127.0.0.1:10808"
    assert getattr(direct, "socks_proxy", None) in (None, ""), "use_proxy=False is real"
    text = caplog.text
    assert "gate: route=socks5://127.0.0.1:10808" in text, text
    assert "gate: route=direct" in text, text


def test_an_empty_socks5_proxy_means_direct_everywhere(monkeypatch):
    from config.settings import settings
    from src.exchanges.client import create_exchange

    monkeypatch.setattr(settings, "socks5_proxy", "", raising=False)
    assert getattr(create_exchange("gate"), "socks_proxy", None) in (None, "")


# --- the market-load budget: sized by a measurement, not by a guess ---------

class _SlowCatalogue:
    """A ccxt-shaped client whose `load_markets` needs `delay` seconds and which
    records the per-request timeout in force at the moment it was asked.

    The recording is the whole point: "the load got a bigger timeout" and "the
    bigger timeout was left on the client for the rest of the cycle" are
    indistinguishable from the return value, and the second one is the bug the
    first one would introduce.
    """

    def __init__(self, delay: float = 0.05, timeout: int = 20_000):
        self.timeout = timeout
        self._delay = delay
        self.asked = 0
        self.timeouts_seen = []

    async def load_markets(self, reload: bool = False):
        self.asked += 1
        self.timeouts_seen.append(self.timeout)
        if self._delay:
            await asyncio.sleep(self._delay)
        return {"BTC/USDT": {"id": "BTC_USDT", "spot": True}}


_ENGINES = [
    pytest.param(updater, "load_markets_with_retry", id="1d-engine"),
    pytest.param(updater_15m, "load_markets_retries", id="15m-engine"),
]


@pytest.mark.parametrize("mod,loader_name", _ENGINES)
def test_a_market_list_that_needs_minutes_is_not_judged_by_a_30s_stopwatch(
        mod, loader_name, monkeypatch):
    """Measured on this box on 2026-09-05: 92,784 B in 12.2 s through the SOCKS5
    tunnel (curl, `Accept-Encoding: gzip`), and the identical DIRECT request
    delivered 17,226 B in 45 s and never finished. gate's whole `fetchMarkets` is
    ~1.4 MB, so the load needs minutes; the engines had a 20 s (`_make_exchange`)
    or 40 s (`exchanges.client`) per-request timeout inside a 30 s total wait, so
    gate could never load there — permanently, on any network.

    Both ceilings have to lift together: widening only the outer wait leaves the
    load dying inside ccxt, and that failure is indistinguishable in the log from
    "the exchange is down".
    """
    loader = getattr(mod, loader_name)
    monkeypatch.setattr(mod.settings, "exchange_markets_load_sec", 240.0)
    ex = _SlowCatalogue(delay=0.05, timeout=20_000)

    assert asyncio.run(loader(ex, "gateio", attempts=1)) is True
    assert ex.timeouts_seen == [240_000], "asked with the widened per-request timeout"
    assert ex.timeout == 20_000, "…and the client is handed back as it was found"
    assert ex.asked == 1, "a load that succeeded must not be repeated"


@pytest.mark.parametrize("mod,loader_name", _ENGINES)
def test_a_request_timeout_already_bigger_than_the_budget_is_never_lowered(
        mod, loader_name, monkeypatch):
    """`exchange_markets_load_sec` is a floor for the load, not a leash on the
    operator: someone who set 300 s in ccxt itself wants 300 s."""
    loader = getattr(mod, loader_name)
    monkeypatch.setattr(mod.settings, "exchange_markets_load_sec", 5.0)
    ex = _SlowCatalogue(delay=0.0, timeout=300_000)
    assert asyncio.run(loader(ex, "mexc", attempts=1)) is True
    assert ex.timeouts_seen == [300_000]


@pytest.mark.parametrize("mod,loader_name", _ENGINES)
def test_a_load_killed_by_its_budget_does_not_leave_that_budget_on_the_client(
        mod, loader_name, monkeypatch):
    """The failure path is where a `finally` earns its keep: an engine whose
    market load was abandoned at 50 ms must go on fetching candles with its own
    20 s, or one slow exchange turns every candle request of the cycle into a
    50 ms one and the outage spreads instead of stopping."""
    loader = getattr(mod, loader_name)
    monkeypatch.setattr(mod.settings, "exchange_markets_load_sec", 0.05)
    # A client configured below the budget, so the load IS widened — and dies
    # anyway. That is the case a `finally` is written for.
    ex = _SlowCatalogue(delay=1.0, timeout=1)

    assert asyncio.run(loader(ex, "gateio", attempts=1)) is False
    assert ex.timeouts_seen == [50], "widened to the budget before being given up on"
    assert ex.timeout == 1, "a cancelled load must restore the client"
    assert mod._markets_load_budget() == 0.05, "the budget came from settings, per call"


def test_the_default_budget_is_the_measured_size_of_the_slowest_list():
    """A default is a claim about the world: 240 s is gate's ~1.4 MB of market
    lists at the measured 7.6 KB/s with room, and it must not be quietly rolled
    back to a number that no exchange could satisfy on this link. The engine
    refresh policy stays what it was (30 min) until the operator changes it."""
    from config.settings import settings
    assert settings.exchange_markets_load_sec >= 240.0
    assert settings.exchange_markets_ttl_sec == 1800.0
    for mod in (updater, updater_15m):
        assert mod.MARKETS_TTL_SECONDS == 1800.0
        assert mod._markets_load_budget() == settings.exchange_markets_load_sec


def test_the_market_list_refresh_interval_is_a_setting_both_engines_read(monkeypatch):
    """Same knob, both engines — a half-raised TTL leaves one engine re-burning
    the link every 30 minutes, which is the throughput bug this exists to fix.
    0 is honoured ("re-download every cycle", the honest way to watch a new
    listing appear), and a negative value falls back to the default instead of
    becoming "reload forever"."""
    for mod in (updater, updater_15m):
        monkeypatch.setattr(mod.settings, "exchange_markets_ttl_sec", 21600.0)
        assert mod._markets_ttl_sec() == 21600.0
        monkeypatch.setattr(mod.settings, "exchange_markets_ttl_sec", 0.0)
        assert mod._markets_ttl_sec() == 0.0
        monkeypatch.setattr(mod.settings, "exchange_markets_ttl_sec", -1.0)
        assert mod._markets_ttl_sec() == 1800.0
