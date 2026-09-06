"""Growth-filter regression tests.

The dashboard gains an optional pair filter (off by default): keep only the
pairs whose price is up more than X% from the lowest LOW of the last N days to
the current CLOSE. The metric is computed per table in a background thread and
cached; these tests pin the pure parts — the SQL shape, the percent math, and
the frame filtering — without a database.
"""
import os

os.environ.setdefault("DASHBOARD_DEMO", "1")

import pandas as pd

from dashboard import app as dapp


def _ok_sql(days=7):
    return dapp._growth_sql("btc_usdt_on_bybit", cutoff=1750000000)


def test_growth_sql_rejects_bad_table_names():
    """A table name with anything outside the identifier charset must not be
    interpolated into SQL (the table names come from the catalog, but an injected
    name would otherwise turn a diagnostic into an injection surface)."""
    assert dapp._growth_sql("btc_usdt_on_bybit", 1)
    assert dapp._growth_sql("1000000babydoge_usdt:usdt_on_bybit", 1)
    assert dapp._growth_sql("btc/usdt", 1) == ""       # '/' is not valid in a table name here
    assert dapp._growth_sql("btc\" ON ...", 1) == ""   # quote injection refuses
    assert dapp._growth_sql("", 1) == ""


def test_growth_sql_handles_both_epoch_units():
    """Timestamp can be seconds OR milliseconds; both the cutoff and the ms-cutoff
    are in the WHERE, so a legacy ms table is matched in its own unit."""
    sql = _ok_sql()
    assert sql.count('"Timestamp" >= {cutoff}') == 0  # non-literal phrasing
    assert "MIN(\"low\")" in sql
    assert '"close"' in sql
    assert "100000000000" in sql            # the ms-vs-seconds branch exists


def test_growth_from_row_math():
    """(current close - lowest low) / lowest low * 100 is the bottom→now rise."""
    assert abs(dapp._growth_from_row({"cur_close": 150.0, "min_low": 100.0}) - 50.0) < 1e-9
    assert abs(dapp._growth_from_row({"cur_close": 100.0, "min_low": 200.0}) + 50.0) < 1e-9
    assert dapp._growth_from_row(None) is None
    assert dapp._growth_from_row({}) is None
    assert dapp._growth_from_row({"cur_close": 0, "min_low": 100}) is None
    assert dapp._growth_from_row({"cur_close": "x", "min_low": 100}) is None


def test_filter_frames_by_growth_keeps_only_matching(monkeypatch):
    """Frame filtering reads the cached metrics and drops rows below the bar."""
    metrics = {
        ("db1", "t1"): (80.0, 0.0),   # over the bar
        ("db1", "t2"): (20.0, 0.0),   # under the bar
        ("db1", "t3"): (None, 0.0),   # uncomputable / errored
    }
    monkeypatch.setattr(dapp, "_GROWTH_STATE", {"metrics": metrics})

    df = pd.DataFrame([
        {"ticker": "AAA/USDT", "db_name": "db1", "table_name": "t1"},
        {"ticker": "BBB/USDT", "db_name": "db1", "table_name": "t2"},
        {"ticker": "CCC/USDT", "db_name": "db1", "table_name": "t3"},
    ])
    out15, out1d = dapp._filter_frames_by_growth(df, pd.DataFrame(), 50.0)
    assert list(out15["ticker"]) == ["AAA/USDT"]       # only the 80% row survives
    assert out1d.empty


def test_filter_frames_by_growth_empty_when_off():
    """Without any cached metric, the filter keeps nothing (the pair list falls
    back to the unfiltered one in the caller while the background fills)."""
    monkeypatch = None
    original = dapp._GROWTH_STATE.get("metrics")
    dapp._GROWTH_STATE["metrics"] = {}
    try:
        df = pd.DataFrame([{"ticker": "AAA/USDT", "db_name": "db1", "table_name": "t1"}])
        out15, _ = dapp._filter_frames_by_growth(df, pd.DataFrame(), 50.0)
        assert out15.empty
    finally:
        if original is None:
            dapp._GROWTH_STATE.pop("metrics", None)
        else:
            dapp._GROWTH_STATE["metrics"] = original
