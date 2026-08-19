"""
Unit tests for Perp-First symbol selection and token filtering.
"""
from src.exchanges.symbol_selector import (
    select_symbols_perp_first,
    should_skip_pair,
)


def test_should_skip_leveraged_tokens():
    assert should_skip_pair("BTC3L/USDT") is True
    assert should_skip_pair("ETH3S/USDT") is True
    assert should_skip_pair("BULL/USDT") is True
    assert should_skip_pair("BTC/USDT") is False
    assert should_skip_pair("BTC/USDT:USDT") is False


def test_select_symbols_perp_first_prioritizes_swaps():
    symbols = [
        "BTC/USDT",          # Spot
        "BTC/USDT:USDT",     # Perpetual Swap
        "ETH/USDT",          # Spot only
        "SOL/USDT:USDT",     # Perpetual Swap only
        "BTC3L/USDT",        # Leveraged token to skip
    ]
    markets = {
        "BTC/USDT": {"base": "BTC", "spot": True, "swap": False},
        "BTC/USDT:USDT": {"base": "BTC", "spot": False, "swap": True},
        "ETH/USDT": {"base": "ETH", "spot": True, "swap": False},
        "SOL/USDT:USDT": {"base": "SOL", "spot": False, "swap": True},
        "BTC3L/USDT": {"base": "BTC3L", "spot": True, "swap": False},
    }

    selected = select_symbols_perp_first(symbols, markets)

    # BTC should select perp (BTC/USDT:USDT), ETH spot, SOL perp
    assert "BTC/USDT:USDT" in selected
    assert "BTC/USDT" not in selected
    assert "ETH/USDT" in selected
    assert "SOL/USDT:USDT" in selected
    assert "BTC3L/USDT" not in selected
