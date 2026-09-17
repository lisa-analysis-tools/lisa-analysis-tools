"""Digest globalfit_run.log + gpu_util CSVs for the 3-mo production run."""
import os
import re
import sys
from datetime import datetime

import numpy as np


def discover_run_logs(run_dir):
    """Every rank's run log under ``run_dir``: the head's ``globalfit_run.log``
    first, then ``globalfit_run.rank<k>.log`` files sorted by rank number.

    Each rank writes its own log file under the walker-block layout
    (``run.py::_rank_log_filenames``); concatenating them (head first) is
    what lets the parsing below see every rank's lines, not just the
    head's (Plan 5 Task 4 of the multi-rank port).

    The walk is RECURSIVE and deterministic (directories and file names
    sorted), identical to ``gf_monitor_gen.py``'s copy of this helper: a
    snapshot may nest the artifacts directory one level down. A second file
    for a rank already seen (the same run unpacked twice under ``run_dir``)
    is NOT silently dropped -- the first one found wins and the duplicate is
    named on stderr, so a half-merged snapshot is visible rather than
    invisible.
    """
    found = {}
    for root, dirs, fns in os.walk(run_dir):
        dirs.sort()
        for fn in sorted(fns):
            if fn == "globalfit_run.log":
                key = -1
            else:
                m = re.match(r"^globalfit_run\.rank(\d+)\.log$", fn)
                if m is None:
                    continue
                key = int(m.group(1))
            path = os.path.join(root, fn)
            if key in found:
                label = "head" if key < 0 else f"rank {key}"
                print(
                    f"# WARNING: duplicate {label} run log {path}; keeping {found[key]}",
                    file=sys.stderr,
                )
                continue
            found[key] = path
    return [found[k] for k in sorted(found)]


#: ``communication/fanout.py``'s head-only load-balance line, DEBUG level:
#: "[FANOUT] op=%s move=%s head_s=%.3f max_rank_s=%.3f wait_s=%.3f"
FANOUT_RE = re.compile(
    r"\[FANOUT\] op=(?P<op>\S+) move=(?P<move>\S+) "
    r"head_s=(?P<head_s>[\d.]+) max_rank_s=(?P<max_rank_s>[\d.]+) "
    r"wait_s=(?P<wait_s>[\d.]+)"
)


def summarize_fanout(lines):
    """Per-(op, move) ``[FANOUT]`` stats over ``lines`` (any iterable of str).

    Returns ``{(op, move): {"count", "mean_head_s", "max_rank_s", "mean_wait_s"}}``,
    empty if no ``[FANOUT]`` line is present.
    """
    groups = {}
    for line in lines:
        m = FANOUT_RE.search(line)
        if not m:
            continue
        key = (m.group("op"), m.group("move"))
        bucket = groups.setdefault(
            key, {"count": 0, "head_s_sum": 0.0, "max_rank_s_max": 0.0, "wait_s_sum": 0.0}
        )
        bucket["count"] += 1
        bucket["head_s_sum"] += float(m.group("head_s"))
        bucket["max_rank_s_max"] = max(bucket["max_rank_s_max"], float(m.group("max_rank_s")))
        bucket["wait_s_sum"] += float(m.group("wait_s"))
    return {
        key: {
            "count": b["count"],
            "mean_head_s": b["head_s_sum"] / b["count"],
            "max_rank_s": b["max_rank_s_max"],
            "mean_wait_s": b["wait_s_sum"] / b["count"],
        }
        for key, b in groups.items()
    }


#: ``communication/fanout.py``'s ``fanout_digest_line()`` replica suffix
#: (one-walker replica mode only): "... residual=r0:11,r1:22 replicas_agree=False".
#: ``re.search`` (not ``match``) so this also finds the suffix inside a full
#: log line ("<date> <time>,<ms> - <module> - <LEVEL> - <msg>"). A
#: multi-walker-mode ``[FANOUT_DIGEST]`` line carries no ``residual=`` suffix
#: at all and simply fails to match, which is what "ignored" means below.
DIGEST_REPLICA_RE = re.compile(
    r"\[FANOUT_DIGEST\] it=(?P<it>\d+) .*?residual=(?P<res>\S+) "
    r"replicas_agree=(?P<agree>True|False)"
)


def summarize_replica_digest(lines):
    """Per-iteration replica agreement from ``[FANOUT_DIGEST] ... residual=...
    replicas_agree=...`` lines (one-walker replica mode).

    Returns ``{"iterations": n, "agree": n_true, "disagree": [it, ...]}``,
    empty if no line carries the ``residual=``/``replicas_agree=`` suffix
    (e.g. every line comes from multi-walker mode).
    """
    iterations = 0
    agree = 0
    disagree = []
    for line in lines:
        m = DIGEST_REPLICA_RE.search(line)
        if not m:
            continue
        iterations += 1
        if m.group("agree") == "True":
            agree += 1
        else:
            disagree.append(int(m.group("it")))
    if iterations == 0:
        return {}
    return {"iterations": iterations, "agree": agree, "disagree": disagree}


#: ``gbspecialstretch.py``'s one-walker replica-mode ``[GB_REPLICA]`` lines --
#: two ``logger.info`` shapes (no move name inside the brackets; drift always
#: present), one ``logger.warning`` shape for the hash bonus check (move name
#: inside the brackets, no drift, now demoted to informational text but kept
#: here under its historical key), and one ``logger.warning`` shape for the
#: REAL divergence guard -- ``_replica_apply_sync``'s per-rank
#: ``log_like_final`` tolerance check (move name inside the brackets, no
#: drift): grep ``GB_REPLICA`` in that file for the exact emitters.
GB_REPLICA_RE = re.compile(
    r"\[GB_REPLICA(?: [^\]]*)?\] "
    r"(?P<msg>residual hashes disagree after sync|"
    r"log_like_final disagrees after sync|"
    r"residual authoritative; rebuild deferred to gb_sync)"
    r"(?:.*?drift (?P<drift>[0-9.eE+-]+))?"
)


def summarize_gb_replica(lines):
    """Counts/worst-case over ``[GB_REPLICA]`` info/warning lines.

    Returns ``{"disagree_warnings": n, "loglike_disagree_warnings": n,
    "deferred_rebuilds": n, "max_drift": float|None}``, empty if no
    ``[GB_REPLICA]`` line is present.

    ``disagree_warnings`` counts the residual-hash line (an INFO-level bonus
    check -- expected to fire routinely on GPU, see
    ``_replica_apply_sync``). ``loglike_disagree_warnings`` counts the real
    divergence guard: the per-rank ``log_like_final`` tolerance check, which
    is the one that should stay at zero.
    """
    disagree_warnings = 0
    loglike_disagree_warnings = 0
    deferred_rebuilds = 0
    max_drift = None
    for line in lines:
        m = GB_REPLICA_RE.search(line)
        if not m:
            continue
        msg = m.group("msg")
        if msg == "residual hashes disagree after sync":
            disagree_warnings += 1
        elif msg == "log_like_final disagrees after sync":
            loglike_disagree_warnings += 1
        else:
            deferred_rebuilds += 1
            drift = m.group("drift")
            if drift is not None:
                drift = float(drift)
                if max_drift is None or drift > max_drift:
                    max_drift = drift
    if not disagree_warnings and not loglike_disagree_warnings and not deferred_rebuilds:
        return {}
    return {
        "disagree_warnings": disagree_warnings,
        "loglike_disagree_warnings": loglike_disagree_warnings,
        "deferred_rebuilds": deferred_rebuilds,
        "max_drift": max_drift,
    }


if __name__ != "__main__":
    # Standalone report script, not a library; guard the rest of the file
    # (which reads sys.argv / real log files unconditionally) so tests can
    # import discover_run_logs()/summarize_fanout() above without running it.
    raise SystemExit(0)

RUN = sys.argv[1] if len(sys.argv) > 1 else None
LOG_PATHS = discover_run_logs(f"{RUN}/gf_prod_3mo_artifacts")

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+) - (\S+) - (\w+) - (.*)$")

def parse_ts(s, ms):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp() + int(ms) / 1e3

events = []
for LOG in LOG_PATHS:
    with open(LOG, errors="replace") as fh:
        for line in fh:
            m = TS.match(line)
            if m:
                events.append((parse_ts(m.group(1), m.group(2)), m.group(3),
                               m.group(4), m.group(5)))
# Multiple files are read one after another above, not merged chronologically;
# re-sort so downstream slicing (``ev[-1]`` = latest event) is still correct.
events.sort(key=lambda e: e[0])

if not events:
    print(f"no timestamped log lines under {LOG_PATHS} -- nothing to digest")
    raise SystemExit(0)

# Attempt boundaries. Primary anchor: the per-rank startup layout line
# (``run.py`` logs ``layout.describe()`` once per rank right after
# ``prepare_rank``), which every launch emits. The old anchor -- the
# 'Multiple GPUs detected' warning -- fires only when ONE rank owns more
# than one device, so it is absent from every walker-block run (one device
# per rank); keep it as the fallback for pre-port logs, and fall back again
# to the first event so an unanchored log still gets a timing summary
# instead of an IndexError.
starts = [t for t, mod, lvl, msg in events if "walker-block layout: size=" in msg]
anchor = "walker-block layout"
if not starts:
    starts = [t for t, mod, lvl, msg in events
              if "Multiple GPUs detected" in msg and "analysiscontainer" in mod]
    anchor = "Multiple GPUs detected"
if not starts:
    starts = [events[0][0]]
    anchor = "first log line (no layout / multi-GPU anchor found)"
# Every rank logs the layout line, so one attempt contributes several
# near-simultaneous stamps once the per-rank logs are merged: collapse a
# cluster to its first stamp so "attempt starts" stays one entry per attempt
# and ``t_last`` is the START of the last attempt, not its slowest rank.
# Scoped to the walker-block anchor ONLY: the legacy 'Multiple GPUs detected'
# anchor fires once per ``AnalysisContainer`` construction, not once per
# attempt, so collapsing it too would re-base the LAST ATTEMPT window of
# already-reported pre-port snapshots. That anchor keeps its pre-collapse
# behaviour: ``t_last = starts[-1]`` over every stamp, uncollapsed.
if anchor == "walker-block layout":
    starts = [t for i, t in enumerate(starts) if i == 0 or t - starts[i - 1] > 120.0]
print(f"attempt starts [{anchor}]:",
      [datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S") for t in starts])

# last attempt slice
t_last = starts[-1]
ev = [e for e in events if e[0] >= t_last - 60]
print(f"\n=== LAST ATTEMPT ({datetime.fromtimestamp(t_last)}) "
      f"-> {datetime.fromtimestamp(ev[-1][0])} "
      f"({(ev[-1][0]-t_last)/3600:.2f} h of log) ===")

# ---- per-move propose boundaries: 'buffer lifecycle' = propose exit --------
lif = [(t, msg) for t, _, _, msg in ev if "buffer lifecycle" in msg]
moves = {}
for t, msg in lif:
    mv = msg.split(":")[0]
    moves.setdefault(mv, []).append(t)
for mv, ts in moves.items():
    dt = np.diff(ts)
    print(f"propose cadence {mv}: n={len(ts)}, median gap "
          f"{np.median(dt) if len(dt) else float('nan'):.1f} s")

# ---- rj band units: 'band unit complete after N pick rounds (M cells)' -----
unit_re = re.compile(r"(\w+): band unit complete after (\d+) pick rounds \((\d+) cells\)")
units = []
for t, _, _, msg in ev:
    m = unit_re.search(msg)
    if m:
        units.append((t, m.group(1), int(m.group(2)), int(m.group(3))))
rj_units = [(t, r, c) for t, mv, r, c in units if "rj" in mv]
if rj_units:
    ts_u = np.array([u[0] for u in rj_units])
    walls = np.diff(np.concatenate(([t_last], ts_u)))
    print(f"\nrj units: {len(rj_units)}; cells/unit "
          f"{[u[2] for u in rj_units[:8]]}...; pick rounds "
          f"{[u[1] for u in rj_units[:8]]}...")
    print(f"rj unit walls [s]: {np.round(walls[:10], 1)}")
    print(f"  median {np.median(walls):.1f}, total {walls.sum()/3600:.2f} h")

# ---- at-cap skip lines -----------------------------------------------------
cap = [msg for t, _, _, msg in ev if "at-cap skip" in msg]
if cap:
    print(f"\nat-cap skip lines: {len(cap)}; last: ...{cap[-1][-130:]}")

# ---- in-model flush / grouped lines ---------------------------------------
for key in ("flush", "grouped", "pool", "polish"):
    hits = [msg for t, _, _, msg in ev
            if key in msg.lower() and "GPU pool" not in msg]
    if hits:
        print(f"\n'{key}' lines: {len(hits)}; e.g. {hits[len(hits)//2][:150]}")

# ---- leaves growth (gb move: after-proposal cold-chain leaves) -------------
leaves = []
for t, _, _, msg in ev:
    if "active leaves in cold chain after proposal" in msg:
        arr = re.findall(r"\d+", msg.split("[")[-1])
        if arr:
            leaves.append((t, int(np.mean([int(a) for a in arr]))))
if leaves:
    lv = [(datetime.fromtimestamp(t).strftime("%H:%M"), n) for t, n in leaves]
    print(f"\nleaves (mean over walkers) samples: {lv[::max(1,len(lv)//10)]}")

# ---- memory series from buffer-build lines ---------------------------------
mem_re = re.compile(
    r"GPU pool used ([\d.]+) / total ([\d.]+) GB; device used/total GB: "
    r"dev0 ([\d.]+)/[\d.]+, dev1 ([\d.]+)")
mems = []
for t, _, _, msg in ev:
    m = mem_re.search(msg)
    if m:
        mems.append((t, *[float(g) for g in m.groups()]))
if mems:
    a = np.array(mems)
    print(f"\nmemory series ({len(a)} pts): pool used "
          f"{a[:, 1].min():.1f}->{a[:, 1].max():.1f} GB; dev0 "
          f"{a[:, 3].min():.1f}->{a[:, 3].max():.1f} GB; dev1 "
          f"{a[:, 4].min():.1f}->{a[:, 4].max():.1f} GB")
    np.save(f"{RUN}/../mem_series_last.npy", a)

# ---- host RSS from SubBandBuffer lines --------------------------------------
rss = [(t, float(m.group(1))) for t, _, _, msg in ev
       if (m := re.search(r"host maxRSS ([\d.]+) GB", msg))]
if rss:
    r = np.array(rss)
    print(f"host maxRSS: {r[:,1].min():.1f} -> {r[:,1].max():.1f} GB")

# ---- fstat fit + [SAVE] + warnings ------------------------------------------
for t, _, lvl, msg in ev:
    if "grid fit epoch" in msg or "[SAVE]" in msg or "multi-device scorer" in msg:
        print(f"  {datetime.fromtimestamp(t).strftime('%H:%M:%S')} {msg[:150]}")
warns = {}
for t, _, lvl, msg in ev:
    if lvl in ("WARNING", "ERROR"):
        key = msg[:60]
        warns[key] = warns.get(key, 0) + 1
print("\nwarnings/errors (last attempt):")
for k, v in sorted(warns.items(), key=lambda x: -x[1])[:8]:
    print(f"  {v:4d}x {k}")

# ---- [FANOUT] load-balance summary (Plan 5 Task 4) --------------------------
fanout_summary = summarize_fanout(msg for _, _, _, msg in ev)
if fanout_summary:
    print("\n=== [FANOUT] summary (last attempt, per op/move) ===")
    print(f"{'op':<12}{'move':<22}{'n':>6}{'mean_head_s':>13}"
          f"{'max_rank_s':>12}{'mean_wait_s':>13}")
    for (op, move), stats in sorted(fanout_summary.items()):
        print(f"{op:<12}{move:<22}{stats['count']:>6}{stats['mean_head_s']:>13.3f}"
              f"{stats['max_rank_s']:>12.3f}{stats['mean_wait_s']:>13.3f}")

# ---- [FANOUT_DIGEST] replica agreement + [GB_REPLICA] summaries (Plan 3 ----
# Task 3, one-walker replica mode) -------------------------------------------
replica_digest_summary = summarize_replica_digest(msg for _, _, _, msg in ev)
if replica_digest_summary:
    print(
        f"\n[FANOUT_DIGEST] replicas: {replica_digest_summary['iterations']} "
        f"iterations, {replica_digest_summary['agree']} agree, "
        f"disagree at it={replica_digest_summary['disagree']}"
    )

gb_replica_summary = summarize_gb_replica(msg for _, _, _, msg in ev)
if gb_replica_summary:
    print(
        f"[GB_REPLICA]: {gb_replica_summary['disagree_warnings']} hash "
        f"disagree warnings (bonus check, INFO), "
        f"{gb_replica_summary['loglike_disagree_warnings']} log_like_final "
        f"disagree warnings (THE GUARD), "
        f"{gb_replica_summary['deferred_rebuilds']} deferred rebuilds, "
        f"max drift {gb_replica_summary['max_drift']}"
    )
