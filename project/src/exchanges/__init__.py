"""
Exchange interaction module: CCXT Async client, symbol selection, and gap filler.
"""
from src.exchanges.symbol_selector import select_symbols_perp_first, should_skip_pair
from src.exchanges.client import create_exchange, close_exchange_safely
from src.exchanges.gap_filler import fill_history_gaps

__all__ = [
    "select_symbols_perp_first",
    "should_skip_pair",
    "create_exchange",
    "close_exchange_safely",
    "fill_history_gaps",
]
