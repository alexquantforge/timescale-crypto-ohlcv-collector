#!/usr/bin/env python3
"""
swap_watch.py — per-process RSS + swap sampler (stdlib only).

PURPOSE
-------
`free -h` shows ONLY the system total. To answer "what exactly is eating my
swap" you have to look at /proc/<pid>/status, which carries per-process:

    VmRSS   : resident set size in kB
    VmSwap  : how many kB of THIS process live in swap

Run this over a few hours (or days) while the dashboard + updaters + browser
are up and it will tell you which process *family* is climbing — the collector
(main.py), the streamlit dashboard, or the browser's Bybit tabs.

USAGE
-----
    python swap_watch.py                            # sample every 60 s, until Ctrl-C
    python swap_watch.py --interval 5               # sample every 5 s
    python swap_watch.py --interval 60 --max 120    # 120 samples (~2 h) then stop
    python swap_watch.py --out ./swap_watch.csv     # write CSV here (default)
    python swap_watch.py --detail 0                 # print EVERY process to the terminal
    python swap_watch.py --detail 30                # print top 30 by swap to the terminal
    python swap_watch.py --top 25                   # log only top-N processes by swap (CSV)
    python swap_watch.py --group                    # 1 row per family instead of 1 per pid (CSV)
    python swap_watch.py --summary 5                # print a delta summary every 5 samples

The terminal gets BOTH a detailed per-process table (pid, comm, RSS, swap, % of
system swap, command) and the family totals, refreshed every tick; the CSV holds
the full long-run history for trend analysis.

EXIT AND WATCH
--------------
Leave it running in a terminal (or `nohup python swap_watch.py --interval 60 &
`). After a few hours open the CSV and look at `swap_kb` trending up under one
family. The collector's own memory-release task is the fix for the updaters;
if the climbing family is the dashboard/browser, the next step is outside the
collectors (restart streamlit, close tabs, etc.).

NOTES
-----
* `comm` lines that are threads (worker threads show a pid == tgid later) are
  de-duplicated by (familY, pid).
* Reading every /proc/<pid>/status is cheap (~ms) even with thousands of
  processes; the sampler is not a load on the machine.
* Needs no pip deps — pure standard library.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

# How to map a process to a family the operator actually cares about. The
# remaining processes are labelled by their own command/comm name.
_COLLECTOR_TOKENS = ("main.py", "updater", "updater_15m")
_DASHBOARD_TOKENS = ("dashboard/app.py", "streamlit", "streamlit_app")
_BROWSER_COMMS = {"chrome", "chromium", "chromium-browser", "firefox", "brave",
                  "msedge", "opera", "vivaldi"}


def _pid_is_threading_leader(pid: int) -> bool:
    """A thread shares its memory with the process that owns it; VmSwap is per
    process, not per thread, so reporting one row per thread double-counts.
    The canonical owner is the one whose pid == its own tgid."""
    try:
        with open(f"/proc/{pid}/status", "r") as fh:
            for line in fh:
                if line.startswith("Tgid:"):
                    return int(line.split()[1]) == pid
    except OSError:
        return False  # process vanished mid-scan — caller will skip it
    return True


def _read_status(pid: int) -> dict | None:
    """Return smoothed {comm, rss_kb, swap_kb} for a pid, or None if it is
    gone / we have no permission. Only the leader of each thread group is
    reported, so a browser's ~40 render threads collapse to a few rows."""
    try:
        with open(f"/proc/{pid}/status", "r") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    comm = ""
    rss_kb = swap_kb = 0
    tgid = pid
    for line in lines:
        if line.startswith("Name:"):
            comm = line.split(":", 1)[1].strip()
        elif line.startswith("VmRSS:"):
            rss_kb = int(line.split()[1])
        elif line.startswith("VmSwap:"):
            swap_kb = int(line.split()[1])
        elif line.startswith("Tgid:"):
            tgid = int(line.split()[1])
    if tgid != pid:
        return None  # not the leader; skip (avoid double counting)
    if not comm:
        return None
    return {"comm": comm, "rss_kb": rss_kb, "swap_kb": swap_kb}


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        return raw.strip()
    except OSError:
        return ""


def _family(comm: str, cmdline: str) -> str:
    cl = cmdline or comm
    low = cl.lower()
    if any(t in low for t in _COLLECTOR_TOKENS):
        return "collector"
    if any(t in low for t in _DASHBOARD_TOKENS):
        return "dashboard"
    base = os.path.basename(comm).lower()
    if base in _BROWSER_COMMS or "chrome" in low or "chromium" in low:
        return "browser"
    return comm[:32] or "unknown"


def _scan() -> list[dict]:
    """Snapshot every interesting process: return rows with family/pid/comm/rss/
    swap/cmd. Pids that die between reads are simply dropped."""
    rows: list[dict] = []
    seen: set[int] = set()
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        if pid in seen:
            continue
        if not _pid_is_threading_leader(pid):
            continue
        status = _read_status(pid)
        if status is None:
            continue
        seen.add(pid)
        cmdline = _cmdline(pid)
        rows.append({
            "family": _family(status["comm"], cmdline),
            "pid": pid,
            "comm": status["comm"],
            "rss_kb": status["rss_kb"],
            "swap_kb": status["swap_kb"],
            "cmd": cmdline[:120],
        })
    return rows


def _group(rows: list[dict]) -> dict:
    """Sum rss/swap per family."""
    groups: dict[str, dict] = {}
    for r in rows:
        g = groups.setdefault(r["family"], {"rss_kb": 0, "swap_kb": 0, "pids": 0})
        g["rss_kb"] += r["rss_kb"]
        g["swap_kb"] += r["swap_kb"]
        g["pids"] += 1
    return groups


def _fmt_kb(kb: int) -> str:
    if kb >= 1024 * 1024:
        return f"{kb / (1024 * 1024):.2f} GB"
    if kb >= 1024:
        return f"{kb / 1024:.1f} MB"
    return f"{kb} KB"


def _pct(part: int, whole: int) -> str:
    """Percent of `whole` that `part` is; '0%' for negligible but non-zero."""
    if whole <= 0:
        return "--"
    p = part / whole * 100.0
    if p >= 99.9:
        return "100%"
    if p < 0.05:
        return "<0.1%"
    return f"{p:.1f}%"


def _short_cmd(cmd: str, width: int = 60) -> str:
    cmd = (cmd or "").strip()
    if len(cmd) <= width:
        return cmd
    return cmd[: width - 3] + "..."


def _print_detail(rows: list[dict], limit: int) -> None:
    """Print the top-by-swap processes as an aligned, readable table.

    `limit=0` prints everything; otherwise only the `limit` rows that use the
    most swap (ties broken by RSS). Zero-rss kernel threads are dropped — they
    are noise on a memory hunt.
    """
    shown = [r for r in rows if r["swap_kb"] > 0 or r["rss_kb"] > 0]
    if not shown:
        print("    (no process has a non-zero footprint)")
        return
    shown.sort(key=lambda r: (r["swap_kb"], r["rss_kb"]), reverse=True)
    if limit and len(shown) > limit:
        shown = shown[:limit]

    total_swap = sum(r["swap_kb"] for r in rows)
    total_rss = sum(r["rss_kb"] for r in rows)

    fams = [r["family"] for r in shown]
    pids = [str(r["pid"]) for r in shown]
    comms = [r["comm"] for r in shown]
    rss = [_fmt_kb(r["rss_kb"] * 1024) for r in shown]
    swap = [_fmt_kb(r["swap_kb"] * 1024) for r in shown]
    pcts = [_pct(r["swap_kb"], total_swap) for r in shown]
    cmds = [_short_cmd(r["cmd"]) for r in shown]

    w_fam = max(10, max(len(x) for x in fams))
    w_pid = max(5, max(len(x) for x in pids))
    w_com = max(8, max(len(x) for x in comms))
    w_rss = max(9, max(len(x) for x in rss))
    w_sw = max(9, max(len(x) for x in swap))
    w_pct = max(5, max(len(x) for x in pcts))

    hdr = (f"  {'FAMILY':<{w_fam}}  {'PID':>{w_pid}}  {'COMM':>{w_com}}  "
           f"{'RSS':>{w_rss}}  {'SWAP':>{w_sw}}  {'%SWAP':>{w_pct}}  CMD")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i in range(len(shown)):
        print(f"  {fams[i]:<{w_fam}}  {pids[i]:>{w_pid}}  {comms[i]:>{w_com}}  "
              f"{rss[i]:>{w_rss}}  {swap[i]:>{w_sw}}  {pcts[i]:>{w_pct}}  {cmds[i]}")
    if limit and len(shown) >= limit:
        more = sum(1 for r in rows if r["swap_kb"] > 0 or r["rss_kb"] > 0) - len(shown)
        if more > 0:
            print(f"  ... and {more} more process(es) (use --detail 0 to show all)")


def _print_groups(rows: list[dict]) -> None:
    """Family totals, largest swap first."""
    groups = _group(rows)
    if not groups:
        print("    (no groups)")
        return
    print("  ---- family totals ----")
    for fam, g in sorted(groups.items(), key=lambda kv: -kv[1]["swap_kb"]):
        if g["swap_kb"] <= 0 and g["rss_kb"] <= 0:
            continue
        print(f"    {fam:<24} {g['pids']:>3} proc   "
              f"rss={_fmt_kb(g['rss_kb'] * 1024):>10}   "
              f"swap={_fmt_kb(g['swap_kb'] * 1024):>10}")


def _human_time(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sample per-process RSS + swap to find what eats your swap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--interval", type=float, default=60.0,
                   help="seconds between samples")
    p.add_argument("--max", type=int, default=0,
                   help="max samples to take (0 = forever / until Ctrl-C)")
    p.add_argument("--out", default="swap_watch.csv",
                   help="CSV output path (append mode)")
    p.add_argument("--top", type=int, default=0,
                   help="log only the top-N processes by swap in the CSV (0 = all)")
    p.add_argument("--detail", type=int, default=15,
                   help="print this many top-by-swap processes to the terminal (0 = all)")
    p.add_argument("--group", action="store_true",
                   help="write one row per family to the CSV, not one per pid")
    p.add_argument("--summary", type=int, default=0,
                   help="print a delta summary every N samples")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    new = not os.path.exists(args.out)
    fieldnames = ["ts_iso", "uptime_s", "swap_used_kb", "family", "pid",
                  "comm", "rss_kb", "swap_kb", "cmd"]
    out_fh = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
    if new:
        writer.writeheader()
        out_fh.flush()

    print(f"swap_watch: writing to {args.out}  (interval={args.interval:g}s, "
          f"{'forever' if not args.max else args.max} sample(s), "
          f"{'grouped' if args.group else 'per-pid'} CSV; "
          f"terminal detail=top{args.detail})", file=sys.stderr)
    print("Swap will be reported per process (VmSwap). Watch for a family whose "
          "swap_kb trends UP.", file=sys.stderr)

    # baseline for the delta summary
    prev_groups: dict[str, dict] = {}
    sample_no = 0
    total_rss = total_swap = 0

    def _free_swap() -> int:
        try:
            with open("/proc/meminfo", "r") as mh:
                for line in mh:
                    if line.startswith("SwapTotal:"):
                        tot = int(line.split()[1])
                    elif line.startswith("SwapFree:"):
                        free = int(line.split()[1])
                return (tot - free) * 1024
        except Exception:
            return 0

    try:
        while args.max == 0 or sample_no < args.max:
            sample_no += 1
            rows = _scan()
            if args.top:
                rows = sorted(rows, key=lambda r: r["swap_kb"], reverse=True)[: args.top]
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            uptime = time.monotonic()
            total_rss = sum(r["rss_kb"] for r in rows)
            total_swap = sum(r["swap_kb"] for r in rows)
            used_kb = _free_swap() // 1024  # system-wide swap used, kB

            if args.group:
                for fam, g in sorted(_group(rows).items(),
                                     key=lambda kv: -kv[1]["swap_kb"]):
                    writer.writerow({
                        "ts_iso": ts, "uptime_s": round(uptime),
                        "swap_used_kb": used_kb, "family": fam,
                        "pid": g["pids"], "comm": "", "rss_kb": g["rss_kb"],
                        "swap_kb": g["swap_kb"], "cmd": "",
                    })
            else:
                for r in rows:
                    writer.writerow({
                        "ts_iso": ts, "uptime_s": round(uptime),
                        "swap_used_kb": used_kb, "family": r["family"],
                        "pid": r["pid"], "comm": r["comm"],
                        "rss_kb": r["rss_kb"], "swap_kb": r["swap_kb"],
                        "cmd": r["cmd"],
                    })
            out_fh.flush()

            # --- Terminal output (detailed) ---------------------------------
            used = used_kb * 1024
            ts_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n[#{sample_no}] {ts_local}  "
                f"swap used={_fmt_kb(used)}  "
                f"(sampled of {len(rows)} proc: RSS={_fmt_kb(total_rss * 1024)}, "
                f"swap={_fmt_kb(total_swap * 1024)})"
            )
            _print_detail(rows, args.detail)
            _print_groups(rows)

            # Delta summary every N samples
            if args.summary and sample_no % args.summary == 0 and sample_no > 1:
                print("  --- deltas since last summary ---")
                cur_groups = _group(rows)
                for fam, g in sorted(cur_groups.items(),
                                     key=lambda kv: -kv[1]["swap_kb"]):
                    dg = prev_groups.get(fam)
                    if dg is None:
                        continue
                    drss = g["rss_kb"] - dg["rss_kb"]
                    dsw = g["swap_kb"] - dg["swap_kb"]
                    if drss or dsw:
                        print(f"    {fam:<12} dRSS={_fmt_kb(drss*1024):>10} "
                              f"dSwap={_fmt_kb(dsw*1024):>10}")
                print("  ---------------------------")
                prev_groups = cur_groups
            elif args.summary and sample_no <= 1:
                prev_groups = _group(rows)

            if args.max and sample_no >= args.max:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nswap_watch: stopped by user.", file=sys.stderr)
    finally:
        out_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
