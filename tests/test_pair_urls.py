"""Pair URLs for the links the engines print and the dashboard renders.

These strings are the only part of the pipeline that a human clicks, and every
exchange spells its own path differently — no amount of consistency would make
them all right. So each format below is pinned to a URL that was opened in a
browser, not to what the previous line of code did.

All of it comes from `get_exchange_url` / `get_swap_url`: the engines' `🔗` log
lines, the `url_of_trading_pair` column written into TimescaleDB, and the
dashboard's `Spot ↗ / Swap ↗` row. A wrong entry here is therefore wrong in the
terminal, in the table and on the page at once — and right in all three after a
fix, which is why this file tests the builders and then the two surfaces that
consume them.
"""
from src.exchanges.symbol_selector import get_exchange_url, get_swap_url


def test_bingx_spot_pairs_are_named_glued_under_en():
    """`https://bingx.com/en/spot/RAINPROTOCOLUSDT` — the operator's working
    example: base and quote glued, locale `en`, nothing after the pair name."""
    assert get_exchange_url("bingx", "RAINPROTOCOL/USDT") == (
        "https://bingx.com/en/spot/RAINPROTOCOLUSDT")
    assert get_exchange_url("bingx", "BTC/USDT") == "https://bingx.com/en/spot/BTCUSDT"


def test_bingx_perps_are_named_dashed_under_en():
    """`https://bingx.com/en/perpetual/PROLOGUE-USDT` — same exchange, different
    spelling: a dash, no `-SWAP`, no trailing slash. The pair arrives as a ccxt
    perp symbol (`BASE/QUOTE:QUOTE`), so the settle leg must not leak into the URL."""
    assert get_swap_url("bingx", "PROLOGUE/USDT:USDT") == (
        "https://bingx.com/en/perpetual/PROLOGUE-USDT")
    assert get_swap_url("bingx", "BTC/USDT:USDT") == "https://bingx.com/en/perpetual/BTC-USDT"


def test_no_bingx_url_carries_the_shapes_that_do_not_open():
    """`en-us`, a trailing `/`, and the ccxt `-SWAP` suffix were all in the old
    output; each produced a BingX 404. Asserted on every pair kind so a future
    edit cannot reintroduce one of them in just one of the two dicts."""
    for sym in ("RAINPROTOCOL/USDT", "PROLOGUE/USDT:USDT", "BTC/USDT", "BTC/USDT:USDT"):
        for url in (get_exchange_url("bingx", sym), get_swap_url("bingx", sym)):
            assert "en-us" not in url, url
            assert not url.endswith("/"), url
            assert "SWAP" not in url.upper(), url


def test_the_dashboard_link_row_and_the_stored_column_use_the_same_urls():
    """The two consumers, so a builder cannot be right while a renderer keeps
    assembling its own string (that is how the terminal and the page disagreed)."""
    from dashboard.helpers import build_pair_links_html
    html = build_pair_links_html("RAINPROTOCOL/USDT", "bingx", "RAINPROTOCOL/USDT:USDT")
    assert "https://bingx.com/en/spot/RAINPROTOCOLUSDT" in html, html
    assert "https://bingx.com/en/perpetual/RAINPROTOCOL-USDT" in html, html

    from src.core.updater import _candles_df_from_rows
    df = _candles_df_from_rows([[1700000000000, 1, 1, 1, 1, 1]],
                               "PROLOGUE/USDT:USDT", "bingx")
    assert df["url_of_trading_pair"].iloc[0] == (
        "https://bingx.com/en/perpetual/PROLOGUE-USDT"
    ), df["url_of_trading_pair"].iloc[0]
    # a perp row has no separate spot link by design
    assert df["url_of_swap_contract_if_it_exists"].iloc[0] is None


def test_the_other_exchanges_keep_their_own_grammar():
    """Not a tautology: these are the shapes the URLs had already been checked
    against, and the bingx edit must not be "generalised" into them (gate wants an
    underscore, okx a lowercase dashed path, mexc a `exchange/` page …)."""
    assert get_exchange_url("gateio", "BTC/USDT") == "https://www.gate.io/trade/BTC_USDT"
    assert get_exchange_url("mexc", "BTC/USDT") == "https://www.mexc.com/exchange/BTC_USDT"
    assert get_swap_url("okx", "BTC/USDT:USDT") == "https://www.okx.com/en/trade-swap/btc-usdt-swap"
    assert get_swap_url("bybit", "BTC/USDT:USDT") == "https://www.bybit.com/trade/usdt/BTCUSDT"
    assert get_exchange_url("unknownx", "BTC/USDT") == ""
