"""
Unit tests for engine helpers: load_markets retry logic.
"""
import asyncio


class FlakyExchange:
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0
        self.reloads = []

    async def load_markets(self, reload=False):
        self.calls += 1
        self.reloads.append(reload)
        if self.calls <= self.fail_times:
            raise asyncio.TimeoutError()  # empty message, like in production logs
        return {"BTC/USDT": {}}


def test_load_markets_with_retry_forwards_reload_flag():
    """The persistent-instance registry reloads markets with reload=True —
    make sure the retry helper forwards the flag to the exchange."""
    from src.core.updater import load_markets_with_retry

    ex = FlakyExchange(fail_times=0)
    ok = asyncio.run(load_markets_with_retry(ex, "gateio", attempts=1, timeout=2.0, reload=True))
    assert ok is True
    assert ex.reloads == [True]


def test_load_markets_with_retry_recovers_after_failures():
    from src.core.updater import load_markets_with_retry

    ex = FlakyExchange(fail_times=2)
    ok = asyncio.run(load_markets_with_retry(ex, "gateio", attempts=3, timeout=2.0))
    assert ok is True
    assert ex.calls == 3


def test_load_markets_with_retry_gives_up_cleanly():
    from src.core.updater import load_markets_with_retry

    ex = FlakyExchange(fail_times=99)
    ok = asyncio.run(load_markets_with_retry(ex, "gateio", attempts=2, timeout=2.0))
    assert ok is False
    assert ex.calls == 2


def test_allowed_15m_exchanges_follows_env_not_a_hardcoded_list():
    """
    The 15m startup table cleanup must compare against the .env allow/deny
    lists. Empty lists = "everything supported is kept" (so a default .env can
    never DROP tables); 1D-only names in ALLOWED_EXCHANGES are ignored instead
    of being subtracted from the keep-set.
    """
    from src.core.updater_15m import _compute_allowed_15m_exchanges

    supported = ["bybit", "gateio", "mexc", "okx", "bingx"]
    assert _compute_allowed_15m_exchanges(supported, "", "") == set(supported)
    # bitget is not servable by the 15m engine -> ignored, the other 4 survive
    assert _compute_allowed_15m_exchanges(
        supported, "bybit,bitget,mexc", ""
    ) == {"bybit", "mexc"}
    assert _compute_allowed_15m_exchanges(
        supported, "", "mexc,htx"
    ) == {"bybit", "gateio", "okx", "bingx"}
    assert _compute_allowed_15m_exchanges(supported, "okx", "okx") == set()


def test_allowed_15m_exchanges_accepts_json_list():
    from src.core.updater_15m import _compute_allowed_15m_exchanges

    assert _compute_allowed_15m_exchanges(
        ["bybit", "gateio", "mexc", "okx", "bingx"], '["bybit","gateio"]', ""
    ) == {"bybit", "gateio"}
