"""
Settings parsing tests: ALLOWED_EXCHANGES accepts comma-separated and JSON formats.
"""
from config.settings import Settings


def test_allowed_exchanges_comma_separated():
    st = Settings(ALLOWED_EXCHANGES="bybit, okx , bitget")
    assert st.allowed_exchanges == ["bybit", "okx", "bitget"]


def test_allowed_exchanges_json():
    st = Settings(ALLOWED_EXCHANGES='["bybit","okx"]')
    assert st.allowed_exchanges == ["bybit", "okx"]


def test_allowed_exchanges_empty():
    st = Settings(ALLOWED_EXCHANGES="")
    assert st.allowed_exchanges == []


def test_dash_warm_neighbors_default_and_alias():
    st = Settings()
    # 2, not 5: prefetching 11 pairs (2 timeframes of candles, up to 20 exchange
    # range fetches, 3 feed calls and a chart build each) cost more database and
    # network than it saved in clicks — the app felt slower to use, which is the
    # one thing prefetching must never do.
    assert st.dash_warm_neighbors == 2
    assert Settings().dash_warm_delay_sec == 1.5
    assert Settings().dash_warm_stale_skip_sec == 172800.0
    st2 = Settings(DASH_WARM_NEIGHBORS="0")
    assert st2.dash_warm_neighbors == 0


# --- EXCLUDED_EXCHANGES + filter_exchange_ids -------------------------------
# The 1D map order matters for per-exchange tuning, so filtering preserves it.
_MAP_1D = ["bybit", "gateio", "mexc", "okx", "bingx", "bitget", "kucoin", "htx", "coinex"]


def test_excluded_exchanges_comma_and_json():
    st = Settings(EXCLUDED_EXCHANGES="kucoin, HTX ")
    assert st.excluded_exchanges == ["kucoin", "htx"]  # trimmed + lower-cased
    st2 = Settings(EXCLUDED_EXCHANGES='["kucoin"]')
    assert st2.excluded_exchanges == ["kucoin"]
    assert Settings(EXCLUDED_EXCHANGES="").excluded_exchanges == []
    # malformed JSON list -> treated as "no filter", never as a crash
    assert Settings(EXCLUDED_EXCHANGES="[not, json").excluded_exchanges == []
    # NOTE: a comma-list is taken literally, so a typo silently excludes nothing
    assert Settings(EXCLUDED_EXCHANGES="{bitget}").excluded_exchanges == ["{bitget}"]


def test_filter_preserves_input_order_with_no_filters():
    st = Settings()
    assert st.filter_exchange_ids(_MAP_1D) == _MAP_1D


def test_filter_excluded_wins_over_allowed():
    st = Settings(ALLOWED_EXCHANGES="bybit,okx,bitget", EXCLUDED_EXCHANGES="bitget")
    assert st.filter_exchange_ids(_MAP_1D) == ["bybit", "okx"]


def test_filter_excluding_everything_yields_empty_list():
    st = Settings(EXCLUDED_EXCHANGES=",".join(_MAP_1D))
    assert st.filter_exchange_ids(_MAP_1D) == []


def test_engine_configured_exchanges_deny_only_does_not_imply_allow_list(monkeypatch):
    """
    Regression: with only EXCLUDED_EXCHANGES set, the engine must serve every
    OTHER exchange. A naive `if settings.allowed_exchanges: filter` + an
    allow-list-derived table cleanup would have treated "no allow-list" as
    "allow nothing" and dropped the kucoin/htx tables from the 15m databases.
    """
    from src.core.updater import MarketDataEngine
    import src.core.updater as upd_mod

    monkeypatch.setattr(
        upd_mod, "settings", Settings(EXCLUDED_EXCHANGES="bitget"), raising=True
    )
    eng = MarketDataEngine.__new__(MarketDataEngine)  # no DB, no __init__
    eng.timeframe = "1d"
    assert eng.get_configured_exchanges() == [
        "bybit", "gateio", "mexc", "okx", "bingx", "kucoin", "htx", "coinex",
    ]


def test_engine_configured_exchanges_empty_list_when_nothing_allowed(monkeypatch):
    from src.core.updater import MarketDataEngine
    import src.core.updater as upd_mod

    monkeypatch.setattr(
        upd_mod,
        "settings",
        Settings(ALLOWED_EXCHANGES="kucoin", EXCLUDED_EXCHANGES="kucoin"),
        raising=True,
    )
    eng = MarketDataEngine.__new__(MarketDataEngine)
    eng.timeframe = "1d"
    assert eng.get_configured_exchanges() == []


def test_engine_configured_exchanges_honours_allow_list(monkeypatch):
    from src.core.updater import MarketDataEngine
    import src.core.updater as upd_mod

    monkeypatch.setattr(
        upd_mod, "settings", Settings(ALLOWED_EXCHANGES='["bybit","okx"]'), raising=True
    )
    eng = MarketDataEngine.__new__(MarketDataEngine)
    eng.timeframe = "1d"
    assert eng.get_configured_exchanges() == ["bybit", "okx"]
