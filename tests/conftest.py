"""Keep every test run off the real cache directory.

`DASH_SNAPSHOT_DIR` (default `~/.cache/timescale-ohlcv-dashboard/`) holds the
startup snapshot and, since the inventory cache landed, the last `pg_catalog`
listing per database. Both are read by the code under test BEFORE it decides to
query anything — which is exactly the behaviour the tests exist to check — so a
shared directory would let one test's writes decide another's assertions, and let
the suite read (and write) the developer's real cache while it runs.
"""
import os
import tempfile

os.environ.setdefault("DASH_SNAPSHOT_DIR",
                      tempfile.mkdtemp(prefix="ohlcv-dashboard-tests-"))
