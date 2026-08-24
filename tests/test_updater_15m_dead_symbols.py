"""
BingX (and others) answer some spot symbols with a permanent
'symbol is not found' (code 100204 — delisted tokens still present in
load_markets). Retrying them on every 5-minute cycle is pure noise and
rate-limit burn: the 15m engine marks such pairs dead until restart.

Also: synthetic *STOCK* tokens (MEXC stock perps like CXMTSTOCK) carry
garbage kline timestamps (their tables showed ~28 YEARS of 'history') —
both pair selectors must skip them.
"""
import ccxt

from src.core import updater_15m
from src.exchanges import symbol_selector


def test_symbol_not_found_classification():
    assert updater_15m._is_symbol_not_found_error(
        ccxt.BadSymbol("bingx does not have market symbol ROSS/USDT")
    )
    assert updater_15m._is_symbol_not_found_error(
        ccxt.BadRequest('bingx {"code":100204,"msg":"symbol is not found.","timestamp":1}')
    )
    assert not updater_15m._is_symbol_not_found_error(
        ccxt.RequestTimeout("timed out")
    )
    assert not updater_15m._is_symbol_not_found_error(
        ccxt.BadRequest('gate {"label":"INVALID_PARAM_VALUE","message":"Candlestick too long ago"}')
    )


def test_mark_dead_symbol_populates_graveyard():
    updater_15m._DEAD_SYMBOLS.clear()
    err = ccxt.BadRequest('bingx {"code":100204,"msg":"symbol is not found.","timestamp":1}')
    updater_15m._mark_dead_symbol_if_gone(err, "bingx", "ROSS/USDT")
    assert ("bingx", "ROSS/USDT") in updater_15m._DEAD_SYMBOLS
    updater_15m._mark_dead_symbol_if_gone(ccxt.RequestTimeout("x"), "bingx", "ROUTE/USDT")
    assert ("bingx", "ROUTE/USDT") not in updater_15m._DEAD_SYMBOLS
    updater_15m._DEAD_SYMBOLS.clear()


def test_stock_tokens_skipped_in_both_selectors():
    for mod in (updater_15m, symbol_selector):
        for pair in ("CXMTSTOCK/USDT", "AAOISTOCK/USDT:USDT", "DXCMSTOCK/USDT:USDT"):
            assert mod.should_skip_pair(pair, "mexc"), f"{mod.__name__}: {pair}"
        # legit crypto untouched, incl. R-starting bases outside bitget
        assert not mod.should_skip_pair("BTC/USDT:USDT", "mexc")
        assert not mod.should_skip_pair("ROUTE/USDT", "bingx")
        assert not mod.should_skip_pair("ROY/USDT", "bingx")


def test_junk_dollar_and_old_tickers_skipped():
    for mod in (updater_15m, symbol_selector):
        for pair in ("$1/USDT", "$BAR_OLD/USDT", "$TIME/USDT", "$NAP/USDT", "BAR_OLD/USDT"):
            assert mod.should_skip_pair(pair, "bingx"), f"{mod.__name__}: {pair}"
        # digit-leading tickers are NOT pattern-skipped (1INCH is legit) —
        # the dead-symbol graveyard handles the truly broken ones
        assert not mod.should_skip_pair("1INCH/USDT", "bingx")
        assert not mod.should_skip_pair("1CAT/USDT", "bingx")
        assert not mod.should_skip_pair("BTC/USDT:USDT", "bingx")
