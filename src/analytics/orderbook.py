"""
Orderbook snapshot and trade tape analysis module with strict network timeouts.
"""
import asyncio
import time
from typing import Any, Dict, Optional
from src.analytics.vitality import compute_vitality_score
from src.utils.timeouts import hard_wait_for


async def fetch_orderbook_snapshot(
    exchange,
    symbol: str,
    atr_no_paranormal: float,
    fetch_limit: int = 50,
    trades_limit: int = 100,
    trades_window_sec: int = 300,
    depth_pct: float = 1.0,
    fallback_limits: Optional[list] = None,
    timeout_sec: float = 8.0,
) -> Optional[Dict[str, Any]]:
    """
    Fetches current orderbook snapshot and recent trades from exchange with strict timeouts,
    computing spread relative to ATR without paranormal bars, depth, CVD, TPM,
    and vitality score.
    """
    if fallback_limits is None:
        fallback_limits = [20, 10, 5]

    # --- Fetch Orderbook with timeout ---
    ob = None
    limits_to_try = [fetch_limit] + [fb for fb in fallback_limits if fb != fetch_limit]
    for l in limits_to_try:
        try:
            ob = await hard_wait_for(
                exchange.fetch_order_book(symbol, limit=l),
                timeout_sec,
                label=f"{symbol} orderbook(limit={l})",
            )
            if ob and ob.get("bids") and ob.get("asks"):
                break
        except Exception:
            continue

    if not ob:
        return None

    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks:
        return None

    try:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
    except Exception:
        return None

    if best_bid <= 0 or best_ask <= 0:
        return None

    mid = (best_bid + best_ask) / 2.0
    spread_abs = best_ask - best_bid
    spread_pct = (spread_abs / best_bid) * 100.0 if best_bid > 0 else 0.0

    # Spread as % of ATR without paranormal bars
    spread_atr_pct = (
        (spread_abs / atr_no_paranormal) * 100.0
        if atr_no_paranormal and atr_no_paranormal > 0
        else 0.0
    )

    # Calculate Orderbook Depth within ±depth_pct
    lo = mid * (1.0 - depth_pct / 100.0)
    hi = mid * (1.0 + depth_pct / 100.0)

    bid_depth_usd = 0.0
    for entry in bids:
        if len(entry) < 2:
            continue
        p, a = float(entry[0]), float(entry[1])
        if p >= lo:
            bid_depth_usd += p * a

    ask_depth_usd = 0.0
    for entry in asks:
        if len(entry) < 2:
            continue
        p, a = float(entry[0]), float(entry[1])
        if p <= hi:
            ask_depth_usd += p * a

    total_depth_usd = bid_depth_usd + ask_depth_usd
    imbalance = round(bid_depth_usd / ask_depth_usd, 4) if ask_depth_usd > 0 else 99.0

    # --- Fetch Recent Trade Tape with timeout ---
    tpm = 0.0
    last_sec = 999.0
    buy_pct = 50.0
    cvd = 0.0
    cvd_5m = 0.0
    trades = []

    try:
        trades = await hard_wait_for(
            exchange.fetch_trades(symbol, limit=trades_limit),
            timeout_sec,
            label=f"{symbol} trades",
        ) or []
    except Exception:
        trades = []

    if trades:
        now_ms = time.time() * 1000.0
        for t in trades:
            price = float(t.get("price", 0) or 0)
            amt = float(t.get("amount", 0) or 0)
            usd = price * amt
            side = t.get("side")
            signed = usd if side == "buy" else (-usd if side == "sell" else 0.0)
            cvd += signed

            ts = t.get("timestamp") or 0
            if ts and ts >= now_ms - trades_window_sec * 1000.0:
                cvd_5m += signed

        recent = [
            t for t in trades
            if t.get("timestamp") and t["timestamp"] >= now_ms - trades_window_sec * 1000.0
        ]
        if recent:
            tpm = len(recent) / (trades_window_sec / 60.0)
            buys = sum(1 for t in recent if t.get("side") == "buy")
            buy_pct = (buys / len(recent)) * 100.0

        valid_ts = [t.get("timestamp", 0) or 0 for t in trades if t.get("timestamp")]
        if valid_ts:
            last_sec = (now_ms - max(valid_ts)) / 1000.0

    # --- Barcode / Dead Market Detection ---
    is_barcode = False
    try:
        prices = [float(t["price"]) for t in trades if t.get("price")]
        if len(trades) >= 30 and len(set(prices)) <= 4:
            is_barcode = True
    except Exception:
        is_barcode = False

    score, grade = compute_vitality_score(
        tpm=tpm,
        total_depth_usd=total_depth_usd,
        spread_pct=spread_pct,
        is_barcode=is_barcode,
    )

    return {
        "ob_last_trade_sec": round(last_sec, 1),
        "ob_trades_per_min": round(tpm, 2),
        "ob_buy_pressure_pct": round(buy_pct, 1),
        "ob_cvd": round(cvd, 2),
        "ob_cvd_5m": round(cvd_5m, 2),
        "ob_spread_abs": spread_abs,
        "ob_spread_pct": round(spread_pct, 4),
        "ob_spread_atr_pct": round(spread_atr_pct, 4),
        "ob_atr_no_paranormal": round(float(atr_no_paranormal or 0.0), 10),
        "ob_best_bid": best_bid,
        "ob_best_ask": best_ask,
        "ob_bid_depth_usd": round(bid_depth_usd, 2),
        "ob_ask_depth_usd": round(ask_depth_usd, 2),
        "ob_total_depth_usd": round(total_depth_usd, 2),
        "ob_imbalance": imbalance,
        "ob_vitality_score": float(score),
        "ob_vitality_grade": grade,
        "ob_is_barcode": bool(is_barcode),
    }
