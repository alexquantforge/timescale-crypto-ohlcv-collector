"""The LIVE line must never leave a blank "waiting for the first tick" when the
pair HAS a stored price, and it must never pretend that stored price is live.

Backstory: on 1000000BABYDOGE/USDT:USDT @bybit the operator saw
"Live: waiting for the first tick" for ~20 seconds. The render path was fine —
it just had nothing live yet: the first exchange round trip was queued behind a
busy 15m/LOW summary scan (same database) and a stale but reloading bybit market
catalog, and neither the live DB row nor the feed cache had answered. The honest
fix that does not add a request on the render path is to paint the pair's last
STORED CLOSE and say plainly it is not live.

The guard is that a stored value may never wear the LIVE badge: the LIVE line is
the last trade, and a stored CLOSE is a different fact. `_live_line_html` gets an
`is_live` flag so the badge can switch, and `_render_live_panel` supplies the
stored fallback from whatever pair row the chart already loaded.
"""
import os

os.environ.setdefault("DASHBOARD_DEMO", "1")

from dashboard import app as dapp


def test_stored_fallback_badge_is_not_live():
    """`is_live=False` renders an amber "last saved (not live)" chip, never the
    red LIVE badge, and still carries the ATR tooltip when asked."""
    html = dapp._live_line_html({"last": 0.0012, "bid": 0.0011, "ask": 0.0013},
                                "1D_ATR(5)", is_live=False)
    assert "last saved" in html
    assert "(not live)" in html
    assert "🔴 LIVE" not in html
    assert "1D_ATR(5)" in html          # the ATR note is about the chip above, still true


def test_live_default_keeps_the_live_badge():
    """Backward compat: the default `is_live=True` keeps the red LIVE badge, so
    existing callers need no change."""
    html = dapp._live_line_html({"last": 0.0012})
    assert "🔴 LIVE" in html
    assert "last saved" not in html


def test_live_panel_paints_stored_close_and_labels_it_not_live(monkeypatch):
    """With neither the live DB row nor the feed cache answering, the panel shows
    the pair's stored CLOSE, labelled as not live, instead of 'waiting'."""
    calls = {}
    monkeypatch.setattr(dapp, "_db_live_read", lambda *a, **k: None)
    monkeypatch.setattr(dapp, "_feed_value", lambda *a, **k: None)
    monkeypatch.setattr(dapp.st, "markdown",
                        lambda html, **k: calls.setdefault("md", html))
    monkeypatch.setattr(dapp.st, "caption",
                        lambda txt, **k: calls.setdefault("cap", txt))

    dapp._render_live_panel(
        "1000000BABYDOGE/USDT:USDT", "bybit", False,
        db_name="ohlcv_15m_low_vol_usdt_pairs_using_ccxt_and_direct_api1",
        atr_label="1D_ATR(5)",
        db_row={"close": 0.0012, "ob_best_bid": 0.0011, "ob_best_ask": 0.0013},
    )
    assert "md" in calls
    assert "last saved" in calls["md"]
    assert "(not live)" in calls["md"]
    assert "🔴 LIVE" not in calls["md"]
    assert "cap" not in calls            # it did NOT fall through to 'waiting'


def test_live_panel_still_waits_when_there_is_no_stored_price(monkeypatch):
    """The 'waiting for the first tick' caption is preserved for the one honest
    case: nothing live AND nothing stored (a genuinely new/empty table)."""
    calls = {}
    monkeypatch.setattr(dapp, "_db_live_read", lambda *a, **k: None)
    monkeypatch.setattr(dapp, "_feed_value", lambda *a, **k: None)
    monkeypatch.setattr(dapp, "_live_wait_text", lambda c: "WAIT")
    monkeypatch.setattr(dapp.st, "caption",
                        lambda txt, **k: calls.setdefault("cap", txt))

    dapp._render_live_panel("X/USDT", "bybit", False,
                            db_name="low", db_row={"close": None})
    assert "cap" in calls and calls["cap"] == "WAIT"
