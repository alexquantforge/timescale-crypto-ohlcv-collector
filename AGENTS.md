# Notes for AI agents working in this repository

Read this before touching `dashboard/`. Most of it is not style advice: every rule
below is the residue of a change that was made, shipped, looked like an improvement,
and broke something for the person running this collector at home. The commit
messages on this branch carry the per-round reasoning (`git log --oneline
dashboard/app.py`); this file carries the rules that outlive any single round.

## Getting the history (a shallow clone hides it)

The sandbox copy of this repo is usually `--depth 1` with
`+refs/heads/master:refs/remotes/origin/master`, which means **one** commit is
visible locally and the previous versions are not. Fetch the working branch
explicitly and read it before "fixing" something that was already fixed:

```bash
git fetch origin <branch>
git log --oneline FETCH_HEAD
git log -p  FETCH_HEAD -- dashboard/app.py | head -400   # per-round reasoning
```

The history is also partly *in the files*, on purpose: `README.md`'s dashboard
bullet and the comments in `dashboard/app.py` quote the operator's own log lines
and state why a knob exists. Tests carry docstrings that name the failure they
guard. If you remove code, you are deleting the explanation too — move the
explanation with it.

## The operator's standing contract

These were stated as instructions, not preferences. Breaking one is a regression
even when the change is measurably faster in isolation.

* **The dashboard is a guest.** Any dashboard change must do *less* DB/exchange
  work, not more. A retry policy that multiplies requests is a regression.
  "It did not get faster" / "it got slower" = revert.
* **Never wait for an exchange or the DB on the render path.** Paint what is in
  TimescaleDB first; stitching and live feeds land in background threads and are
  swapped in. Do not "improve" the first paint by making it synchronous.
* **Do not propose prefetching** as a fix for slow pair switching. Prefetch is
  deliberately small, lazy and skipped for stale tables; growing it was tried and
  rejected.
* **Never swallow a DB/API error into an empty result.** No bare `return 0`
  after a failed fetch, no `{}` from `/candles` for a rejected or guarded query.
  Report it: a console line (`[scan]`, `[stitch]`, `[markets]`, `[lane]`,
  `[candles]`) plus, when the user can act on it, a UI badge. Rate-limit the
  line, never the fact.
* **A scan result must never render FEWER tables than the last usable one.**
  Partial frames merge into the store; a truncated pass is never persisted as the
  startup snapshot; an empty frame never replaces a non-empty pair list.
* **The UI may wake the collector** for the pair being opened (delete-then-refetch
  on the engine's own timeframe), because painting a stale chart is not enough —
  the user expects the visible date gap to close. Engines stamp a per-timeframe
  heartbeat and the UI reports when nothing answers.
* **Displayed price is `last`** (last trade). Mid price appears only as the
  denominator of the spread; keep that distinction in the wording. Never present
  mid as "the price".
* **A label that disagrees with the method is worse than no label.** UI text that
  quotes a parameter (timeframe, bar count, estimator) must be generated from the
  same variables the calculation received — see `format_atr_label()`, and the
  grep test `test_no_bare_atr_label_is_left_in_the_ui` that bans unlabelled `ATR`
  strings in the UI.
* **Every knob gets documented in the same round it lands**: `config/settings.py`
  comment, `.env.example` entry, and the README bullet if it changes behaviour.
* The operator writes **Russian**; answer in Russian, quote their own log lines
  back at them, and say which number in their log produced the diagnosis.

## Invariants that each cost a whole round to learn

Touch any of these and re-check the others; they were broken by "improvements".

1. **An empty answer is not a fact about the data.** An empty catalog, an empty
   frame, or a scan that published `0 tables` must be a *failed read*
   (`partial`, `missing_tables`, a `[scan]` line) — never an authoritative empty
   pair list, never the thing that gets persisted. Tests:
   `test_an_empty_pair_list_never_replaces_a_real_one`,
   `test_an_empty_catalog_is_reported_as_a_failed_read`,
   `test_the_pair_list_funnel_says_which_filter_ate_the_tier`.
2. **Completeness is measured on the covered set** (chunks answered now ∪ carried
   rows younger than `DASH_SCAN_CARRYOVER_TTL_SEC`), not on
   `len(out) < len(tables)`. Unbounded carry would freeze the list, so the TTL
   bounds only the *claim*, never the merge.
3. **Never pace progress.** `_rescan_delay_sec` keeps three cases distinct:
   a converging sweep (chunks answered, cursor advanced) retries at
   `DASH_SCAN_DEFER_RETRY_SEC`; a stuck database (nothing answered) uses the
   doubling backoff; a complete tier rests for `DASH_SCAN_RESCAN_COMPLETE_SEC`.
   Applying the doubling rule to a converging sweep is what turned ~5 minutes of
   one-time work into ~40 minutes of "15m не загружается". Tests:
   `test_a_sweep_that_is_still_building_the_list_is_not_backed_off`,
   `test_a_later_pass_of_a_building_sweep_may_run_longer_while_the_page_is_idle`,
   `test_a_complete_tier_is_not_re_scanned_while_another_is_starving`.
4. **A diagnostic that miscounts is a bug of the same severity as a slow query.**
   Chunks are tagged by position (`chunk[14]`), the badge quotes the starving
   database's own `chunk 14/69` and the budget the pass actually ran on, and a
   `(backoff)` note must never be printed for work that is progressing.
5. **Anything that exists to avoid work must survive a rerun.** Backoffs,
   cooldowns, the stitch cache, the scan throttle and its resume cursor live in
   `st.cache_resource` via `_state()` — a module-level `dict` is rebuilt on every
   Streamlit rerun, several times a minute, which silently disables them all.

## Dead ends — already investigated, do not re-open

* ccxt's extra `/spot/currencies` round trip: already handled by setting
  `has["fetchCurrencies"] = False` in `create_exchange` / `_new_sync_exchange`.
  A surviving `…/api/v4/spot/currency_pairs` request is ccxt's own
  `fetch_markets()`; that is a timeout-length problem (`DASH_MARKET_LOAD_TIMEOUT_SEC`),
  not a duplicate-call problem.
* Depth is ±1 % of mid, by contract. A neighbouring pair showing
  `Depth ±1% n/a` is correct behaviour, not a bug; do not add REST calls on the
  render path to fill it.
* Do not "fix" the sweep's convergence by setting `DASH_SCAN_CARRYOVER_TTL_SEC=0`.
* Do not freeze or skip the stitch for stale tables on the click path. The
  bounded stitch (≤ `DASH_STITCH_BUDGET_SEC`, page cap) is accepted as it is.
* `asyncio.CancelledError` must keep propagating out of scan code: catching
  `Exception` around a bounded read is fine, catching `BaseException` is not.
* There is no `st.query_params` support in the dashboard: a pair that is not in
  the selector cannot be deep-linked. Treat that as a real gap, not as a bug in
  the scan.

## Working in this tree

```bash
python3 -m venv tools-env && ./tools-env/bin/pip install -r requirements.txt aiohttp-socks pytz pyflakes
./tools-env/bin/python -m pytest tests/ -q      # the whole point of every rule above
./tools-env/bin/python -m pyflakes dashboard/app.py config/settings.py
echo 'tools-env/' >> .git/info/exclude          # keep the venv out of snapshots
```

Known, deliberately unfixed pyflakes noise: `_ccxt` (dashboard/app.py),
`close_all_db_pools` / `BadSymbol as e` (updater), `ex_id`/`ccxt_id`/`e`
(updater_15m), `SimpleNamespace`/`asyncio` in tests.

Patch discipline that has burned this repo:

* One `write_text` per script, and never anchor on non-ASCII (em dash, `⚠️`) —
  splice with `s.index("def _x") … s.index("def _y")` instead.
* If your replacement rewrites a `def` line, re-grep the whole signature
  afterwards: an `async def _table_inventory` once lost its `async`, parsed
  fine, and only the suite caught it.
* When your own test rejects an intended semantic change, fix the fixture.
  Relaxing the code is how these bugs were introduced.
* A change to state persistence must be re-validated against *list convergence*,
  not only against unit tests: an optimisation that makes a degraded path
  reachable can blank the UI for a day.
* Any pipeline hides its own exit code: `pytest -q | tail && git push` green-lights
  a red suite, because `tail` succeeds. Read `${PIPESTATUS[0]}`, or the tail text.
* A cached artifact may only be written by the real thing it stands in for. A
  disk-seeded catalog that re-saves itself, or a listing persisted from a
  previous file, silently converts "the exchange is broken" into "the cache is
  the truth" — see `save_markets_snapshot` / `save_inventory_snapshot`, which are
  called from the success path of a *network* read and a *pg_catalog* read only.
* A disk cache read must be paired with a test that no request happened
  (`test_a_cold_process_applies_the_disk_catalog_without_a_request`,
  `test_a_cold_process_does_not_read_the_catalog_before_the_first_chunk`) and
  with the age printed in the log line, never hidden.

Market loading, the two facts that took four rounds:

* `CCXT_MARKET_TYPES_SKIP=option` (default) removes whole market lists — trimming
  is legitimate because it *removes requests*; gate loads spot+swap+future, bybit
  dropped from 3429 to 1423 markets. It only touches `options['fetchMarkets']`
  when that value's `types` is a LIST (bybit names perps `linear`; mexc's `types`
  is a dict — leaving it alone is the point of the shape check).
* `DASH_MARKET_LOAD_DEBUG=true` times each category after a failed load and prints
  `[markets] gate: probe, 3 categories, 60s per request — spot ok …`. It is the
  only tool that separates "the big list is slow" from "this connection is being
  blackholed"; a `RequestTimeout` on one URL names the leg in flight, not the
  costly one, because the legs are sequential under one per-request timeout. The
  probe must leave the instance with NO markets and the original `fetchMarkets`
  dict, or every later `market()` lookup reports a delisting.

Read a market-load failure by SIZE before reading it by latency. `DASH_MARKET_-
LOAD_DEBUG=true` on 2026-09-05 produced: 94 KB and 1.3 MB lists hung for the
whole 90s per request, a 30-market list answered in 2.8s — same host, same
instance, seconds apart. That rules out the host, the budget and the throttle
queue in one line, and the only remaining difference from a `curl` that finishes
in 3s is `Accept-Encoding`. So the loader re-asks a hanging leg once bare and, if
that answers, retries the load that way in the same cycle
(`DASH_MARKET_LOAD_ACCEPT_ENCODING=identity` to keep it): one header, zero extra
requests, feeds untouched. A load that returns an EMPTY market list is a failure,
not a success — an empty dict reads as "loaded" and comes back as BadSymbol for
every pair on that exchange.

Two routes, one host: `SOCKS5_PROXY` (default `socks5://127.0.0.1:10808`) is used by
the ASYNC clients only (engines, collector, and the dashboard's order-book card via
`create_exchange`); the dashboard's SYNC clients cannot use a `socks5://` URL at all
(`requests` needs `pysocks`, which is not a dependency), so they are direct whatever
`.env` says. "gate is slow" and "gate works" were both true on one machine because
they were measured on different paths — so any network number the dashboard prints
must name its route (`_route_note`, `create_exchange`'s `route=` log), and any proxy
setting that cannot be honoured is reported as ignored, never obeyed silently.

A timeout longer than the operator's own measurement of the same endpoint is not
a timeout problem — but check which measurement they actually made. The
falsified version of this round's theory, recorded so it is not re-derived:
"the gzip stream is broken, because curl (no `Accept-Encoding`) was fast" — the
operator then timed BOTH shapes of the same URL and identity was *slower in wall
time and 12x bigger in bytes* (92,784 B / 8.2s gz vs 1,137,626 B / 22.2s bare,
i.e. a 11-50 KB/s pipe). A narrow link and a corrupted stream both look like
`RequestTimeout`, so the rule in code is: never answer a timeout with more bytes
(`_is_timeout_like`), only with fewer; a decode/framing error is the one case
worth one bare retry (`DASH_MARKET_LOAD_ACCEPT_ENCODING`). `curl` answered gate's `/spot/currency_pairs` in 2.7-5.8s while
ccxt hit the 60s per-request wall: the suspect is state inside the process (a
reused session left half-read by an abandoned request), and the fix is to measure
and to make the page survivable without the endpoint — not to add retries.
