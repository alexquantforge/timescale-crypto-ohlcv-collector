"""Tests for the zombie-spot pruner (src/utils/zombie_prune).

The deletion rule is small, but it is the only irreversible thing in this
repository, so the tests pin the CONSERVATISM of it: what must never be touched
(perps, unpaired tables, unmeasurable counts, live tables, mass prunes) at least
as hard as what must.
"""
import asyncio
import datetime as dt
import time

import pytest

# Real clock: `normalize_last_ts` compares against it, and a fixture date that
# lands in the future is exactly the garbage-row case that guard exists for.
NOW = int(time.time())

from src.utils.zombie_prune import (
    apply_pruning,
    candidate_tables,
    count_and_last_sql,
    is_perp_symbol,
    measure,
    normalize_last_ts,
    over_prune_guard,
    pair_tables_sql,
    perp_to_spot_name,
    plan_pruning,
    read_pair_tables,
    spot_perp_pairs,
    split_pair_table,
    statements_for,
    summarize,
    zombie_verdict,
)


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------
def test_pair_table_names_are_read_from_the_right_side():
    assert split_pair_table("1000rats_usdt:usdt_on_bybit") == ("1000rats_usdt:usdt", "bybit")
    assert split_pair_table("1.0g_usdt_on_gateio") == ("1.0g_usdt", "gateio")
    # rpartition: a base that itself contains `_on_` must not truncate the name
    assert split_pair_table("lion_on_me_usdt_on_bybit") == ("lion_on_me_usdt", "bybit")
    assert split_pair_table("dashboard_live_ticks") is None      # not a pair table
    assert split_pair_table("_on_bybit") is None                 # no symbol
    assert split_pair_table("btc_usdt_on_") is None               # no exchange


def test_a_candidate_pair_comes_only_from_the_perp_side():
    assert perp_to_spot_name("1000rats_usdt:usdt_on_bybit") == "1000rats_usdt_on_bybit"
    # older migrations wrote the perp without the colon
    assert perp_to_spot_name("btc_usdt_usdt_on_bybit") == "btc_usdt_on_bybit"
    # a spot is never a perp, so it can never invent a counterpart to delete
    assert perp_to_spot_name("btc_usdt_on_bybit") is None
    # mixed quotes are not guessed at: doge_usdc_usdt could be a real spot
    assert perp_to_spot_name("doge_usdc_usdt_on_bybit") is None
    assert perp_to_spot_name("random_table") is None
    assert perp_to_spot_name("dashboard_live_ticks") is None

    assert is_perp_symbol("1_000cats_usdt:usdt") is True
    assert is_perp_symbol("1_000cats_usdt") is False


def test_dots_and_underscores_survive_the_round_trip():
    for perp in ("1.0g_usdt:usdt_on_gateio", "fartcoin_usdt:usdt_on_mexc",
                 "1000shib_inu_usdt:usdt_on_okx"):
        spot = perp_to_spot_name(perp)
        assert spot is not None
        assert spot.rsplit("_on_", 1)[1] == perp.rsplit("_on_", 1)[1]   # same exchange
        assert ":" not in spot and spot.count("_on_") == 1               # one table, one suffix


def test_only_tables_with_both_sides_are_candidates():
    tables = {
        "a_usdt:usdt_on_bybit": 100,     # perp with a spot → pair
        "a_usdt_on_bybit": 10,          # the zombie
        "b_usdt:usdt_on_mexc": 50,      # perp alone → nothing to prune
        "c_usdt_on_okx": 7,             # spot alone → nothing to prune
        "dashboard_live_ticks": 1,
    }
    assert spot_perp_pairs(tables) == {"a_usdt_on_bybit": "a_usdt:usdt_on_bybit"}
    to_count, n_tables = candidate_tables(tables)
    assert to_count == sorted(["a_usdt_on_bybit", "a_usdt:usdt_on_bybit"])  # both sides
    assert n_tables == 5


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------
def test_smaller_spot_that_stopped_being_written_is_the_only_prune_case():
    verdict, why = zombie_verdict(10, 50, 40 * 3600)
    assert verdict == "prune" and "10 bars" in why and "40.0h" in why

    # still collected → smaller is not "dead", and deleting live data is not on
    verdict, why = zombie_verdict(10, 50, 60)
    assert verdict == "keep" and "still collected" in why

    # An EMPTY spot table is prunable even with the freshness rule on: there is
    # no data to lose and no last-write time to look at, and it still costs a
    # line in every scan, summary and duplicate check.
    assert zombie_verdict(0, 50, None)[0] == "prune"
    assert zombie_verdict(0, 50, 5)[0] == "keep"          # empty but still collected
    assert zombie_verdict(0, 0)[0] == "keep"               # perp empty too: no premise
    assert zombie_verdict(0, 50, 5, stale_sec=0)[0] == "prune"  # freshness off

    # equal or bigger: the perp-first premise does not hold
    assert zombie_verdict(50, 50, 40 * 3600)[0] == "keep"
    assert zombie_verdict(51, 50, 40 * 3600)[0] == "keep"


def test_an_unmeasurable_table_is_never_pruned():
    assert zombie_verdict(None, 50, 4000)[0] == "unknown"
    assert zombie_verdict(10, None, 4000)[0] == "unknown"
    # freshness unknown while the freshness rule is on → also unknown, because
    # "we could not check whether it is live" is not a licence to delete
    assert zombie_verdict(10, 50, None)[0] == "unknown"
    assert zombie_verdict(10, 50, float("nan"))[0] == "unknown"
    # and `unknown` is never `prune`, whatever the numbers say
    for spot in (0, 1, 1000):
        assert zombie_verdict(spot, None, 4000)[0] == "unknown"


def test_the_operator_can_waive_freshness_but_not_the_count():
    assert zombie_verdict(10, 50, 5, stale_sec=0)[0] == "prune"      # waived
    assert zombie_verdict(10, 50, 5, stale_sec=24 * 3600)[0] == "keep"
    assert zombie_verdict(None, 50, 5, stale_sec=0)[0] == "unknown"  # not waivable


def test_last_timestamp_normalisation():
    now = 1_800_000_000
    assert normalize_last_ts(now - 3600, now) == pytest.approx(3600.0)
    assert normalize_last_ts((now - 7200) * 1000, now) == pytest.approx(7200.0)   # ms table
    ts = dt.datetime.fromtimestamp(now - 60, tz=dt.timezone.utc)
    assert normalize_last_ts(ts, now) == pytest.approx(60.0, abs=2)
    assert normalize_last_ts(None, now) is None
    assert normalize_last_ts("junk", now) is None
    # garbage future rows (the 2031 bug) must not look "recently written"
    assert normalize_last_ts(now + 10 * 365 * 86400, now) is None
    assert normalize_last_ts(0, now) is None


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def test_plan_reports_every_pair_with_its_reason():
    tables = {"a_usdt_on_bybit": 10, "a_usdt:usdt_on_bybit": 50,
              "b_usdt_on_bybit": 90, "b_usdt:usdt_on_bybit": 40,
              "c_usdt_on_bybit": 5}
    counts = {
        "a_usdt_on_bybit": {"bars": 10, "last_ts": NOW - 40 * 3600},
        "a_usdt:usdt_on_bybit": {"bars": 50, "last_ts": NOW},
        "b_usdt_on_bybit": {"bars": 90, "last_ts": NOW - 40 * 3600},
        "b_usdt:usdt_on_bybit": {"bars": 40, "last_ts": NOW},
    }
    plan = plan_pruning(tables, counts, stale_sec=24 * 3600)
    assert len(plan) == 2                                  # c has no perp → not discussed
    by_table = {r["spot"]: r for r in plan}
    assert by_table["a_usdt_on_bybit"]["verdict"] == "prune"
    assert by_table["b_usdt_on_bybit"]["verdict"] == "keep"
    assert by_table["a_usdt_on_bybit"]["perp"] == "a_usdt:usdt_on_bybit"
    assert summarize(plan) == {"prune": 1, "keep": 1, "unknown": 0, "candidates": 2}

    # a count that failed on one table makes that pair unknown, not prune
    counts["a_usdt_on_bybit"] = {"bars": None, "last_ts": None, "error": "TimeoutError: "}
    plan2 = plan_pruning(tables, counts, stale_sec=24 * 3600)
    assert {r["spot"]: r["verdict"] for r in plan2}["a_usdt_on_bybit"] == "unknown"


def test_statements_move_rather_than_drop_unless_told_otherwise():
    actions = [{"spot": "a_usdt_on_bybit"}, {"spot": "z_usdt_on_okx"}]
    trash = statements_for(actions, "trash")
    assert trash[0] == 'CREATE SCHEMA IF NOT EXISTS "zombie_pruned"'
    assert all(s.startswith('ALTER TABLE "public"."') and 'SET SCHEMA "zombie_pruned"' in s
               for s in trash[1:])
    assert all("DROP" not in s for s in trash)

    dropped = statements_for(actions, "drop")
    assert dropped[0] == 'DROP TABLE IF EXISTS "public"."a_usdt_on_bybit" CASCADE'
    assert len(dropped) == 2 and not any("CREATE SCHEMA" in s for s in dropped)

    with pytest.raises(ValueError):
        statements_for(actions, "yolo")

    # nothing that looks like a perp is ever a target
    assert not any(":usdt" in s for s in trash + dropped)


def test_identifier_quoting_cannot_be_injected():
    assert count_and_last_sql('we"ird').startswith('SELECT COUNT(*)::bigint AS bars')
    assert '"we""ird"' in count_and_last_sql('we"ird')
    assert "MAX(\"Timestamp\")" in count_and_last_sql("x")


def test_catalog_query_avoids_information_schema_and_the_underscore_wildcard():
    sql = pair_tables_sql()
    assert "information_schema" not in sql
    assert "LIKE '%\\_on\\_%'" in sql
    assert "pg_catalog.pg_class" in sql and "relkind IN ('r', 'p')" in sql


def test_mass_prune_guard_needs_a_second_consent():
    assert over_prune_guard(10, 1000, 0.35) is None
    assert over_prune_guard(0, 0, 0.35) is None
    msg = over_prune_guard(400, 1000, 0.35)
    assert msg and "refusing to prune 400 of 1000" in msg
    assert over_prune_guard(400, 1000, 0.0) is None       # guard waived by the operator


# ---------------------------------------------------------------------------
# the async part, against a fake connection (no database needed)
# ---------------------------------------------------------------------------
class _FakeConn:
    """Answers the catalog query and per-table counts, with knobs for failure."""

    def __init__(self, rows, bars, *, fail=(), missing_time=False):
        self.rows = rows
        self.bars = bars
        self.fail = set(fail)
        self.missing_time = missing_time
        self.executed: list = []
        self.counted: list = []

    async def fetch(self, sql, *args):
        assert "pg_catalog.pg_class" in sql
        # answer the way the real WHERE clause would: pair tables only
        return [
            dict(r) for r in self.rows
            if "_on_" in r["table_name"] and r["table_name"] != "dashboard_live_ticks"
        ]

    async def fetchrow(self, sql, *args):
        tbl = sql.split("FROM ", 1)[1].strip().strip('"')
        self.counted.append(tbl)
        if tbl in self.fail:
            raise RuntimeError("statement timeout")
        return {"bars": self.bars.get(tbl),
                "last_ts": None if self.missing_time else int(time.time()) - 90 * 3600}

    async def execute(self, sql, *args):
        self.executed.append(sql)
        return "OK"

    def transaction(self):
        conn = self

        class _T:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *exc):
                return False

        return _T()


def _catalog(table_names):
    return [{"table_name": n, "est_bars": 7, "has_time_col": "Timestamp"} for n in table_names]


def test_measure_counts_both_sides_and_keeps_failures_visible():
    conn = _FakeConn([], {"a_usdt_on_bybit": 10, "a_usdt:usdt_on_bybit": 50}, fail={"a_usdt_on_bybit"})
    out = asyncio.run(measure(conn, ["a_usdt_on_bybit", "a_usdt:usdt_on_bybit"], concurrency=2))
    assert out["a_usdt:usdt_on_bybit"]["bars"] == 50
    assert out["a_usdt_on_bybit"]["bars"] is None and "statement timeout" in out["a_usdt_on_bybit"]["error"]


def test_a_pool_is_what_makes_the_counts_concurrent():
    """Regression found by running the tool against a real PostgreSQL.

    `measure` used to open a transaction block per table on ONE connection, and
    asyncpg refuses concurrent `transaction()` blocks on the same connection:
    every COUNT(*) but the first failed, so the whole run reported `unknown` and
    proposed nothing. The fix is a pool; the test keeps the two paths distinct —
    a bare connection is serialized (no SET LOCAL, which needs a transaction),
    a pool gets one bounded transaction per table.
    """
    bars = {f"t{i}_usdt_on_bybit": i for i in range(12)}
    conn = _FakeConn([], bars)

    class _Pool:
        def __init__(self):
            self.borrows = 0

        def acquire(self):
            pool = self

            class _A:
                async def __aenter__(self):
                    pool.borrows += 1
                    return conn

                async def __aexit__(self, *exc):
                    return False

            return _A()

    pool = _Pool()
    out = asyncio.run(measure(pool, sorted(bars), concurrency=4))
    assert {tb: out[tb]["bars"] for tb in bars} == bars          # every table counted
    assert pool.borrows == 12
    assert sum("SET LOCAL statement_timeout" in sql for sql in conn.executed) == 12

    # and on a single connection nothing is asked that would need a transaction
    conn2 = _FakeConn([], bars)
    out2 = asyncio.run(measure(conn2, sorted(bars), concurrency=4))
    assert {tb: out2[tb]["bars"] for tb in bars} == bars and conn2.executed == []


def test_read_pair_tables_then_measure_then_plan_is_the_whole_pipeline():
    tables = ["a_usdt_on_bybit", "a_usdt:usdt_on_bybit", "solo_usdt_on_bybit", "dashboard_live_ticks"]
    bars = {"a_usdt_on_bybit": 10, "a_usdt:usdt_on_bybit": 50, "solo_usdt_on_bybit": 3}
    conn = _FakeConn(_catalog(tables), bars)

    async def run():
        cat = await read_pair_tables(conn)
        to_count, n = candidate_tables(cat)
        counts = await measure(conn, to_count, concurrency=4)
        return cat, to_count, n, plan_pruning({t: cat[t] for t in to_count}, counts, stale_sec=24 * 3600)

    cat, to_count, n, plan = asyncio.run(run())
    assert "dashboard_live_ticks" not in cat                      # filtered in SQL
    assert n == 3 and len(cat) == 3        # the guard's denominator is pair tables only
    assert to_count == sorted(["a_usdt:usdt_on_bybit", "a_usdt_on_bybit"])
    assert [r["verdict"] for r in plan] == ["prune"]
    assert sorted(conn.counted) == to_count                        # nothing else was queried


def test_missing_timestamp_column_is_not_silently_treated_as_stale():
    """`last_ts` unknown + a freshness rule that is on ⇒ unknown ⇒ kept."""
    conn = _FakeConn(_catalog(["a_usdt_on_bybit", "a_usdt:usdt_on_bybit"]),
                     {"a_usdt_on_bybit": 10, "a_usdt:usdt_on_bybit": 50}, missing_time=True)
    counts = asyncio.run(measure(conn, ["a_usdt_on_bybit", "a_usdt:usdt_on_bybit"]))
    plan = plan_pruning({"a_usdt_on_bybit": 10, "a_usdt:usdt_on_bybit": 50}, counts, stale_sec=24 * 3600)
    assert plan[0]["verdict"] == "unknown"

    # the same data with the freshness rule waived does prune (that is the knob)
    assert plan_pruning({"a_usdt_on_bybit": 10, "a_usdt:usdt_on_bybit": 50}, counts,
                        stale_sec=0)[0]["verdict"] == "prune"


# ---------------------------------------------------------------------------
# running the plan
# ---------------------------------------------------------------------------
def test_apply_pruning_separates_the_schema_preamble_from_the_tables():
    class _Boom(_FakeConn):
        def __init__(self, boom_fragment):
            super().__init__([], {})
            self.boom_fragment = boom_fragment

        async def execute(self, sql, *args):
            self.executed.append(sql)
            if self.boom_fragment in sql:
                raise RuntimeError("permission denied for schema public")
            return "OK"

    actions = [{"spot": "a_usdt_on_bybit"}, {"spot": "b_usdt_on_okx"}]
    res = asyncio.run(apply_pruning(_Boom("ALTER TABLE"), actions, "trash"))
    assert [r["spot"] for r in res] == [None, "a_usdt_on_bybit", "b_usdt_on_okx"]
    assert res[0]["ok"] and not res[1]["ok"] and not res[2]["ok"]
    # a preamble failure is reported per table, not swallowed into a nice "1/2"
    assert all("permission denied" in r["error"] for r in res[1:])
    assert sum(1 for r in res if r["spot"] and r["ok"]) == 0


def test_a_parked_name_collision_says_how_to_undo_the_earlier_run():
    class _Taken(_FakeConn):
        async def execute(self, sql, *args):
            self.executed.append(sql)
            if sql.startswith("ALTER TABLE"):
                raise RuntimeError('relation "a_usdt_on_bybit" already exists')
            return "OK"

    res = asyncio.run(apply_pruning(_Taken([], {}), [{"spot": "a_usdt_on_bybit"}], "trash"))
    assert not res[1]["ok"]
    assert "already exists" in res[1]["error"] and "SET SCHEMA public" in res[1]["error"]


def test_apply_pruning_records_every_table_it_moved():
    conn = _FakeConn([], {})
    res = asyncio.run(apply_pruning(conn, [{"spot": "a_usdt_on_bybit"}], "drop"))
    assert [r["spot"] for r in res] == ["a_usdt_on_bybit"]
    assert all(r["ok"] for r in res)
    assert conn.executed == ['DROP TABLE IF EXISTS "public"."a_usdt_on_bybit" CASCADE']
