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


# ---------------------------------------------------------------------------
# Per-pair log line & ETA formatting
# ---------------------------------------------------------------------------

def test_format_pair_result_cooldown_is_not_reported_as_empty_table():
    """total_bars == 0 means the gap scan hit its 6h cooldown, NOT an empty
    table — the old '0.0d stored' text read like data loss."""
    from src.core.updater_15m import format_pair_result

    assert format_pair_result(4, 0, 0, 0) == "+4 candles, gap scan on cooldown, no gaps. OK"
    assert format_pair_result(4, 17280, 0, 0) == "+4 candles, 180.0d stored, no gaps. OK"
    assert "⚠️ gaps: 2, filled: 1" in format_pair_result(0, 96, 2, 1)


def test_fmt_eta_is_not_mistakable_for_a_wall_clock_time():
    from src.core.progress import fmt_eta

    assert fmt_eta(22 * 60 + 13) == "22m13s"       # was "22:13" -> read as 22:13 o'clock
    assert fmt_eta(22 * 3600 + 13 * 60) == "22h13m"
    assert fmt_eta(-5) == "0m00s"
    assert fmt_eta("nope") == "?"
