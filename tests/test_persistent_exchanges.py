"""
Memory fix: the engines used to create a fresh ccxt instance per exchange on
EVERY cycle (plus one more for the pre-count), re-download tens of MB of
market JSON and throw it all away — allocator churn ratcheted RSS up until
the machine swapped. Exchanges are now long-lived instances: markets are
reloaded at most every MARKETS_TTL_SECONDS, instances rotated by age.
"""
import asyncio
import time

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
