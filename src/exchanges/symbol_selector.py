"""
Perp-First symbol selection and token filtering logic.
"""
import re
from typing import Dict, List, Tuple

SKIP_PATTERNS = re.compile(
    r"(3L|3S|5L|5S|2L|2S|4L|4S|UP|DOWN|BULL|BEAR)/USDT$", re.IGNORECASE
)


def should_skip_pair(symbol: str, exchange: str = "") -> bool:
    """
    Returns True if the symbol is a leveraged token, tokenized stock, or invalid pair.
    """
    if not symbol:
        return True

    spot_like = symbol.split(":")[0]
    parts = spot_like.split("/")
    base = parts[0].upper() if len(parts) > 0 else ""

    # Skip stablecoin pairs or non-USDT quote bases
    if base and "USD" in base and base != "USDT":
        return True

    # Skip leveraged tokens (3L, 3S, BULL, BEAR, etc.)
    if SKIP_PATTERNS.search(spot_like):
        return True

    # Skip synthetic *STOCK* tokens (MEXC stock perps like CXMTSTOCK): their
    # klines carry garbage timestamps that poison tables and charts.
    if base.endswith("STOCK"):
        return True

    # Skip withdrawn/delisted listings still present in some exchanges'
    # markets (BingX): '$'-prefixed and '*_OLD'-suffixed tickers never trade.
    if base.startswith("$") or base.endswith("_OLD"):
        return True

    # Bitget: Skip R* tokenized stock symbols (e.g. RAAPL, RGOOGL, RSAM, RSNOW, etc.)
    # These tokenized equity contracts fail parameter validation on Bitget API.
    if exchange.lower() in ("bitget", "bg") and base.startswith("R"):
        # Allow rare legitimate crypto exceptions if any, otherwise skip R-prefixed Bitget stocks
        crypto_exceptions = {"RARE", "RAY", "RAMP", "RAU", "RAVE", "RNDR", "RSR", "RUNE", "RVN", "ROSE", "REQ"}
        if base not in crypto_exceptions:
            return True

    return False


def split_symbol(symbol: str) -> Tuple[str, str]:
    """
    Splits symbol into base and quote assets (e.g. BTC/USDT:USDT -> BTC, USDT).
    """
    parts = (symbol or "").split("/")
    if len(parts) >= 2:
        return parts[0], parts[1].split(":")[0]
    return symbol or "", "USDT"


def get_exchange_url(exchange_id: str, symbol: str) -> str:
    """Returns direct web trading URL for spot pair."""
    base, quote = split_symbol(symbol)
    b_up, q_up = base.upper(), quote.upper()
    b_lo, q_lo = base.lower(), quote.lower()
    urls = {
        "bybit": f"https://www.bybit.com/trade/spot/{b_up}/{q_up}",
        "bitget": f"https://www.bitget.com/en/spot/{b_up}{q_up}_SPBL?type=spot",
        "mexc": f"https://www.mexc.com/exchange/{b_up}_{q_up}",
        "kucoin": f"https://trade.kucoin.com/{b_up}-{q_up}",
        "gateio": f"https://www.gate.io/trade/{b_up}_{q_up}",
        "bingx": f"https://bingx.com/en-us/spot/{b_up}{q_up}/",
        "htx": f"https://www.htx.com/trade/{b_lo}_{q_lo}/",
        "coinex": f"https://www.coinex.com/exchange/{b_lo}-{q_lo}",
        "okx": f"https://www.okx.com/en/trade-spot/{b_lo}-{q_lo}",
    }
    return urls.get((exchange_id or "").lower(), "")


def get_swap_url(exchange_id: str, symbol: str) -> str:
    """Returns direct web trading URL for perpetual swap contract."""
    base, quote = split_symbol(symbol)
    b_up, q_up = base.upper(), quote.upper()
    b_lo, q_lo = base.lower(), quote.lower()
    urls = {
        "bybit": f"https://www.bybit.com/trade/usdt/{b_up}{q_up}",
        "bitget": f"https://www.bitget.com/en/futures/usdt/{b_up}{q_up}",
        "mexc": f"https://futures.mexc.com/exchange/{b_up}_{q_up}",
        "kucoin": f"https://www.kucoin.com/futures/trade/{b_up}{q_up}M",
        "gateio": f"https://www.gate.com/futures/USDT/{b_up}_{q_up}",
        "bingx": f"https://bingx.com/en-us/perpetual/{b_up}{q_up}/",
        "htx": f"https://www.htx.com/futures/linear_swap/exchange/{b_lo}-{q_lo}/",
        "coinex": f"https://www.coinex.com/futures/{b_lo}-{q_lo}",
        "okx": f"https://www.okx.com/en/trade-swap/{b_lo}-{q_lo}-swap",
    }
    return urls.get((exchange_id or "").lower(), "")


def select_symbols_perp_first(
    symbols: List[str], markets: Dict[str, dict], exchange_name: str = ""
) -> List[str]:
    """
    Selects instruments per base asset using PERP-FIRST strategy:
    For each base asset (e.g. BTC), prefers linear perpetual swap (BTC/USDT:USDT).
    Falls back to spot (BTC/USDT) only if no perpetual contract exists.
    """
    spots: Dict[str, str] = {}
    swaps: Dict[str, str] = {}

    for symbol in symbols:
        if symbol not in markets:
            continue
        market = markets[symbol]

        if should_skip_pair(symbol, exchange_name):
            continue

        base = market.get("base")
        if not base:
            continue
        base = base.upper()

        # Spot market check
        if market.get("spot") and symbol.endswith("/USDT"):
            spots[base] = symbol

        # Perpetual linear swap check
        elif market.get("swap") and symbol.endswith("/USDT:USDT"):
            swaps[base] = symbol

    selected_symbols: List[str] = []
    all_bases = sorted(set(spots.keys()) | set(swaps.keys()))

    for base in all_bases:
        if base in swaps:
            selected_symbols.append(swaps[base])
        elif base in spots:
            selected_symbols.append(spots[base])

    return selected_symbols
