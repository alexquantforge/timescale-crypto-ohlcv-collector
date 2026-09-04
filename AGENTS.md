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
