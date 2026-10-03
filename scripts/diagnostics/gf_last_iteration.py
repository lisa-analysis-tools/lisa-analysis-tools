"""A one-screen dashboard of where a global-fit run is RIGHT NOW: last saved row,
stage, the live pace (log age, time since the last save, row cadence), the
ratchet's latest gate action and verdict, head-node GPU utilization and
memory, cold-chain lnL and leaf counts, the last rows, and the stage's
proposal cycle with the last saved leg marked and the next proposal named.

    python scripts/diagnostics/gf_last_iteration.py /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu

    ================================================================================================
     gf_prod_6mo_testing.h5                 /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu
    ------------------------------------------------------------------------------------------------
     last saved row   85             saved after   in_model_removal
     stage            gb_search_3    step 4 of 6 . started row 47 . 38 rows in
     done             gb_search_seed, gb_search_1, gb_search_2
     galfor ratchet   running (no stop stamp)
    ------------------------------------------------------------------------------------------------
     live        log age 2 min . last save 17 min ago . row cadence: last 29 min, median of 5 = 31 min
     ratchet     iteration 1: NUDGE (cycle 1 of 20)     after RELEASE 0: reference recorded
     gpu (head)  GPU0 util 28% (60 s 41%)  mem 29.6/95.8 GB  99 W 50C  .  GPU1 ...   sample 1 min ago
    ------------------------------------------------------------------------------------------------
     cold lnL   row 85   max 104,343,050.4   mean 104,342,594.2   min 104,342,228.4   spread 822.0
                vs 84    max -815.2   mean -842.6   min -360.6
     cold leaves        gb  min 2371  mean 2411.5  max 2491    emri 8 . mbh 4 . sobbh 6 . vgb 55
    ------------------------------------------------------------------------------------------------
     last rows   row  saved after            lnL max          lnL min        gb leaves
                  81  noise_ratchet_search   104,332,775.9    104,330,927.8  2355-2485
                  ...
    ------------------------------------------------------------------------------------------------
     stage cycle (gb_search_3.move_order)   * = a row lands after this leg   <<< last saved   <-- next
       1  * noise_ratchet_search
       ...
       8  * in_model_removal        <<< LAST SAVED (row 85)
       9    gb_ridge_gibbs          <-- NEXT
    ================================================================================================

``--line`` prints the old one-line form (``85  stage=gb_search_3 (step 4 of 6;
...)  saved_after=in_model_removal  store=...``); ``-v`` is accepted and means
the dashboard. ``--rows N`` sets the recent-rows table length (default 5);
``--no-color`` / ``--color`` override the tty detection.

Sources, all read-only and lock-free (safe while the saver writes): the live
store (newest ``*.h5`` in the folder that is not a running backup, a
pre-migration copy or a quarantined file; an ``.h5`` path works too); the
head's run log ``<run_dir>/*_artifacts/globalfit_run.log`` (or the tar's
``globalfit_run_tail.log``), last 4 MB; the launcher's nvidia-smi sampler
``<run_dir>/gpu_util_<job>.csv`` (newest; the HEAD node's GPUs only -- a
2-node layout's second node is not sampled). A dataset read that races the
saver is reported as unreadable on its own line, never as a crash; a missing
log or CSV prints n/a.
"""

import glob
import json
import os
import re
import sys
from datetime import datetime

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SKIP = ("backup", "CORRUPT", "pre_rerung", "pre_repair", "midit", ".stale")
WIDTH = 96
TAIL_BYTES = 48 * 1024 * 1024      # ~half a day of the head's log: enough saves for a cadence
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


# --------------------------------------------------------------------------
# store access
# --------------------------------------------------------------------------
def find_store(path):
    if os.path.isfile(path):
        return path
    cands = [p for p in glob.glob(os.path.join(path, "*.h5"))
             if not any(s in os.path.basename(p) for s in SKIP)]
    if not cands:
        raise FileNotFoundError(f"no run store (*.h5) in {path}")
    return max(cands, key=os.path.getmtime)


def _attr(a, key, default=None):
    v = a.get(key, default)
    if v is None:
        return None
    try:
        return v.item() if hasattr(v, "item") else v
    except Exception:  # noqa: BLE001
        return v


def _str(raw):
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def current_stage(g):
    """The recipe step a resume would run: ``dict(name, order, total, start, done,
    ratchet_done, move_order)``.

    ``None`` when the store carries no recipe group. ``name`` is None when
    every step is complete (the recipe has finished).
    """
    if "recipe" not in g:
        return None
    steps = []
    for name in g["recipe"]:
        a = g["recipe"][name].attrs
        mo = _attr(a, "move_order")
        try:
            mo = list(json.loads(_str(mo))) if mo is not None else None
        except Exception:  # noqa: BLE001
            mo = None
        steps.append(dict(name=name, order=int(_attr(a, "order num", 0) or 0),
                          status=bool(_attr(a, "status", False)),
                          start=_attr(a, "start_iteration"),
                          completed=_attr(a, "completed_iteration"),
                          ratchet_done=_attr(a, "galfor_ratchet_done"),
                          move_order=mo))
    steps.sort(key=lambda s: s["order"])
    done = [s["name"] for s in steps if s["status"]]
    cur = next((s for s in steps if not s["status"]), None)
    return dict(name=None if cur is None else cur["name"],
                order=None if cur is None else cur["order"], total=len(steps),
                start=None if cur is None else cur["start"],
                ratchet_done=None if cur is None else cur["ratchet_done"],
                move_order=None if cur is None else cur["move_order"], done=done)


def last_iteration(path, group="global_fit"):
    """``(iteration, store_path, saved_after, stage)``; ``saved_after`` / ``stage`` None when absent."""
    import h5py

    store = find_store(path)
    with h5py.File(store, "r") as f:
        g = f[group] if group in f else f
        it = int(g.attrs["iteration"])
        after = None
        if it > 0 and "saved_after" in g and g["saved_after"].shape[0] >= it:
            after = _str(g["saved_after"][it - 1])
        stage = current_stage(g)
    return it, store, after, stage


def _cold(arr):
    """Collapse eryn's leading (nsamplers, ntemps) axes of one stored row to the
    cold walkers: ``(nw,)`` for log_like, ``(nw, nleaves)`` for inds."""
    import numpy as np

    a = np.asarray(arr)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim > 2:                      # ntemps > 1 in the main group: cold rung
        a = a[0]
    return a


def read_rows(path, it, n_rows, group="global_fit"):
    """Per-row cold-chain summary for rows ``it-n_rows .. it-1``: a list of
    ``dict(row, after, ll (nw,), leaves {branch: (nw,)})``; a row that cannot
    be read carries ``error``."""
    import h5py
    import numpy as np

    out = []
    lo = max(0, it - n_rows)
    with h5py.File(path, "r") as f:
        g = f[group] if group in f else f
        branches = list(g["inds"].keys()) if "inds" in g else []
        for r in range(lo, it):
            rec = dict(row=r + 1, after=None, ll=None, leaves={})
            try:
                if "saved_after" in g and g["saved_after"].shape[0] > r:
                    rec["after"] = _str(g["saved_after"][r])
                rec["ll"] = np.asarray(_cold(g["log_like"][r]), dtype=float).ravel()
                for b in branches:
                    rec["leaves"][b] = _cold(g["inds"][b][r]).sum(axis=-1).ravel()
            except Exception as exc:  # noqa: BLE001 -- a read racing the saver
                rec["error"] = f"{type(exc).__name__}: {exc}"
            out.append(rec)
    return out


# --------------------------------------------------------------------------
# live sidecars: the head's run log and the nvidia-smi CSV
# --------------------------------------------------------------------------
def _tail_lines(path, nbytes=TAIL_BYTES):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if size > nbytes:
            fh.seek(size - nbytes)
            fh.readline()                               # drop the torn first line
        data = fh.read()
    return data.decode("utf-8", "replace").splitlines()


def find_run_log(run_dir):
    for name in ("globalfit_run.log", "globalfit_run_tail.log"):
        hits = glob.glob(os.path.join(run_dir, "*_artifacts", name))
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


def _ts(line):
    m = _TS.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _mins(a, b):
    return (b - a).total_seconds() / 60.0


def read_live(run_dir, now=None):
    """What the head's run log says about the pace and the gate:
    ``dict(log, log_age_min, last_save_age_min, saves (times), cadence (mins),
    gate, verdict, replica, stage_line)``, or None without a log."""
    path = find_run_log(run_dir)
    if path is None:
        return None
    now = now or datetime.now()
    lines = _tail_lines(path)
    out = dict(log=path, log_age_min=None, last_save_age_min=None, saves=[], cadence=[],
               gate=None, verdict=None, replica=None, stage_line=None)
    last_ts = None
    for line in lines:
        t = _ts(line)
        if t is not None:
            last_ts = t
        if "[SAVE] save_step" in line and t is not None:
            out["saves"].append(t)
        elif "[GALFOR_RATCHET" in line:
            body = line.split("] ", 1)[-1] if "] " in line else line
            if re.search(r"\] iteration \d+: (NUDGE|RELEASE|HOLD)", line):
                out["gate"] = body.split(" -- ")[0].strip()
            elif "] after RELEASE" in line:
                out["verdict"] = body
            elif "RATCHET DONE" in line or "already FINISHED" in line:
                out["verdict"] = body
        elif "[REPLICA_PE" in line and "stored row" in line:
            out["replica"] = line.split("] ", 1)[-1]
        elif "[V9-STAGE" in line and ("entering" in line or "STAGE COMPLETE" in line):
            out["stage_line"] = line.split(" - INFO - ", 1)[-1]
    if last_ts is not None:
        out["log_age_min"] = _mins(last_ts, now)
    if out["saves"]:
        out["last_save_age_min"] = _mins(out["saves"][-1], now)
        s = out["saves"][-7:]
        out["cadence"] = [_mins(a, b) for a, b in zip(s[:-1], s[1:])]
    return out


def read_gpu(run_dir, now=None, window_s=60.0):
    """The newest ``gpu_util_<job>.csv``: per GPU index the last sample
    ``dict(util, mem_used_gb, mem_total_gb, power_w, temp_c, util_avg)`` with
    ``util_avg`` over the trailing ``window_s`` seconds; plus ``age_min`` and
    the CSV's name. None without a CSV."""
    cands = glob.glob(os.path.join(run_dir, "gpu_util_*.csv"))
    if not cands:
        return None
    path = max(cands, key=os.path.getmtime)
    now = now or datetime.now()
    rows = []
    for line in _tail_lines(path, 256 * 1024)[-400:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        try:
            t = datetime.strptime(parts[0].split(".")[0], "%Y/%m/%d %H:%M:%S")
            rows.append(dict(t=t, idx=int(parts[1]), util=float(parts[3]),
                             used=float(parts[5]) / 1024.0, total=float(parts[6]) / 1024.0,
                             power=float(parts[7]), temp=float(parts[8])))
        except ValueError:
            continue
    if not rows:
        return None
    t_last = max(r["t"] for r in rows)
    out = dict(csv=path, age_min=_mins(t_last, now), gpus={})
    for r in rows:
        if (t_last - r["t"]).total_seconds() <= window_s:
            out["gpus"].setdefault(r["idx"], []).append(r)
    gpus = {}
    for idx, rs in out["gpus"].items():
        last = max(rs, key=lambda r: r["t"])
        gpus[idx] = dict(util=last["util"], util_avg=sum(r["util"] for r in rs) / len(rs),
                         mem_used_gb=last["used"], mem_total_gb=last["total"],
                         power_w=last["power"], temp_c=last["temp"])
    out["gpus"] = gpus
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
class Style:
    def __init__(self, color):
        self.color = color

    def _w(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def bold(self, s):
        return self._w("1", s)

    def dim(self, s):
        return self._w("2", s)

    def hi(self, s):          # the last saved leg
        return self._w("1;33", s)

    def nxt(self, s):         # the next proposal
        return self._w("1;36", s)

    def warn(self, s):
        return self._w("1;31", s)


def _fmt(x):
    return f"{x:,.1f}"


def _age(m):
    if m is None:
        return "n/a"
    if m < 1.0:
        return f"{m * 60:.0f} s ago"
    if m < 120:
        return f"{m:.0f} min ago"
    return f"{m / 60:.1f} h ago"


def _rule(ch="-"):
    return ch * WIDTH


def describe_stage(it, stage, verbose):
    if stage is None:
        return "stage=none (no recipe group)"
    if stage["name"] is None:
        return f"stage=FINISHED (all {stage['total']} steps complete)"
    s = f"stage={stage['name']}"
    if not verbose:
        return s
    bits = [f"step {stage['order']} of {stage['total']}"]
    if stage["start"] is not None:
        bits.append(f"started at row {int(stage['start'])}, {it - int(stage['start'])} rows in")
    if stage["ratchet_done"]:
        bits.append("galfor ratchet DONE (stamped)")
    if stage["done"]:
        bits.append("done: " + ", ".join(stage["done"]))
    return s + " (" + "; ".join(bits) + ")"


def _short_verdict(v):
    """One phrase from a ratchet verdict line."""
    if not v:
        return None
    head = v.split(":", 1)[0].strip()                     # "after RELEASE 4"
    if "RATCHET DONE" in v or "already FINISHED" in v:
        return head + ": RATCHET DONE"
    if "recorded as the reference" in v or "baseline" in v:
        return head + ": reference recorded"
    m = re.search(r"drop ([+-][\d.]+%) \(threshold ([\d.]+%)", v)
    if m:
        tail = "continues" if "continues" in v else ("STOP" if "stop" in v.lower() else "")
        return f"{head}: foreground drop {m.group(1)} vs {m.group(2)} -> {tail}".rstrip(" ->")
    m = re.search(r"([+-][\d.,]+) over the previous release \(threshold (\d+)\)", v)
    if m:
        tail = "continues" if "continues" in v else ""
        return f"{head}: lnL gain {m.group(1)} vs {m.group(2)} {tail}".strip()
    return (head + ": " + v.split(":", 1)[-1].strip())[:70]


def render(it, store, after, stage, rows, live, gpu, st):
    L = []
    run_dir = os.path.dirname(os.path.abspath(store))
    L.append(_rule("="))
    L.append(f" {st.bold(os.path.basename(store))}".ljust(40) + f"  {run_dir}")
    L.append(_rule())
    # ---- where ----------------------------------------------------------
    L.append(f" {'last saved row':<16} {st.bold(str(it)):<14}  saved after   "
             f"{st.hi(after) if after else st.dim('(none recorded)')}")
    if stage is None:
        L.append(f" {'stage':<16} none (no recipe group)")
    elif stage["name"] is None:
        L.append(f" {'stage':<16} {st.bold('FINISHED')}  all {stage['total']} steps complete")
    else:
        bits = [f"step {stage['order']} of {stage['total']}"]
        if stage["start"] is not None:
            s0 = int(stage["start"])
            bits.append(f"started row {s0}")
            bits.append(f"{it - s0} rows in")
        L.append(f" {'stage':<16} {st.bold(stage['name']):<14} " + " . ".join(bits))
        if stage["done"]:
            L.append(f" {'done':<16} {', '.join(stage['done'])}")
        if stage["name"].startswith("gb_search"):
            L.append(f" {'galfor ratchet':<16} "
                     + (st.bold("DONE (stop stamped in the store)") if stage["ratchet_done"]
                        else "running (no stop stamp)"))
    L.append(_rule())
    # ---- live: pace, gate, gpu -----------------------------------------
    if live is None:
        L.append(f" {'live':<11} " + st.dim("n/a (no *_artifacts/globalfit_run.log in the run dir)"))
    else:
        bits = [f"log age {_age(live['log_age_min'])}"]
        if live["last_save_age_min"] is not None:
            bits.append(f"last save {_age(live['last_save_age_min'])}")
        if live["cadence"]:
            c = live["cadence"]
            med = sorted(c)[len(c) // 2]
            bits.append(f"row cadence: last {c[-1]:.0f} min, median of {len(c)} = {med:.0f} min")
        L.append(f" {'live':<11} " + " . ".join(bits))
        if live["gate"] or live["verdict"]:
            g = live["gate"] or ""
            v = _short_verdict(live["verdict"])
            L.append(f" {'ratchet':<11} {st.bold(g):<44} {v or ''}")
        if live["replica"]:
            L.append(f" {'replica pe':<11} {live['replica'][:WIDTH - 14]}")
        if live["stage_line"]:
            L.append(f" {'stage log':<11} {live['stage_line'][:WIDTH - 14]}")
    if gpu is None:
        L.append(f" {'gpu':<11} " + st.dim("n/a (no gpu_util_*.csv in the run dir)"))
    else:
        parts = []
        for idx in sorted(gpu["gpus"]):
            d = gpu["gpus"][idx]
            parts.append(f"GPU{idx} util {d['util']:3.0f}% (60 s {d['util_avg']:3.0f}%)  "
                         f"mem {d['mem_used_gb']:4.1f}/{d['mem_total_gb']:.1f} GiB  "
                         f"{d['power_w']:3.0f} W {d['temp_c']:.0f}C")
        stale = gpu["age_min"] is not None and gpu["age_min"] > 5
        tag = f"sample {_age(gpu['age_min'])}"
        L.append(f" {'gpu (head)':<11} " + "  .  ".join(parts) + "   "
                 + (st.warn(tag + " -- STALE") if stale else st.dim(tag)))
    L.append(_rule())
    # ---- cold chain -----------------------------------------------------
    last = rows[-1] if rows else None
    prev = rows[-2] if len(rows) > 1 else None
    if last is None or last.get("error") or last["ll"] is None:
        L.append(f" {'cold lnL':<10} " + st.warn(
            f"row {it} unreadable" + (f" ({last['error']})" if last and last.get('error') else "")))
    else:
        ll = last["ll"]
        L.append(f" {'cold lnL':<10} row {it:<5} max {_fmt(ll.max()):>16}   mean {_fmt(ll.mean()):>16}   "
                 f"min {_fmt(ll.min()):>16}   spread {_fmt(ll.max() - ll.min())}")
        if prev is not None and not prev.get("error") and prev["ll"] is not None:
            p = prev["ll"]
            L.append(f" {'':<10} vs {prev['row']:<6} "
                     f"max {ll.max() - p.max():>+16,.1f}   mean {ll.mean() - p.mean():>+16,.1f}   "
                     f"min {ll.min() - p.min():>+16,.1f}")
        lv = last["leaves"]
        if lv:
            main = "gb" if "gb" in lv else sorted(lv)[0]
            v = lv[main]
            s = (f" {'cold leaves':<10} {'':<9} {main}  min {int(v.min()):<6} mean {v.mean():<8.1f} "
                 f"max {int(v.max()):<6}")
            others = []
            for b in sorted(lv):
                if b == main or b in ("psd", "galfor"):
                    continue
                w = lv[b]
                others.append(f"{b} {int(w.min())}" if w.min() == w.max()
                              else f"{b} {int(w.min())}-{int(w.max())}")
            if others:
                s += "   " + " . ".join(others)
            L.append(s)
    L.append(_rule())
    # ---- recent rows ----------------------------------------------------
    if rows:
        L.append(f" {'last rows':<10} {'row':>5}  {'saved after':<22} {'lnL max':>16} {'lnL min':>16}"
                 f"  {'gb leaves':<11}")
        for rec in rows:
            if rec.get("error") or rec["ll"] is None:
                L.append(f" {'':<10} {rec['row']:>5}  " + st.warn(f"unreadable ({rec.get('error', '?')})"))
                continue
            gb = rec["leaves"].get("gb")
            gbs = f"{int(gb.min())}-{int(gb.max())}" if gb is not None else ""
            line = (f" {'':<10} {rec['row']:>5}  {(rec['after'] or '-'):<22} "
                    f"{_fmt(rec['ll'].max()):>16} {_fmt(rec['ll'].min()):>16}  {gbs:<11}")
            L.append(st.bold(line) if rec is last else line)
        L.append(_rule())
    # ---- the stage's proposal cycle -------------------------------------
    if stage is not None and stage["name"] is not None:
        mo = stage["move_order"]
        if not mo:
            L.append(f" {'stage cycle':<10} " + st.dim("(no move_order recorded for this stage)"))
        else:
            legs = {rec["after"] for rec in rows if rec.get("after")}
            if after:
                legs.add(after)
            L.append(f" stage cycle ({stage['name']}.move_order)   "
                     f"* = a row lands after this leg   <<< last saved   <-- next proposal")
            i_last = mo.index(after) if after in mo else None
            i_next = (i_last + 1) % len(mo) if i_last is not None else None
            for i, m in enumerate(mo):
                mark = "*" if m in legs else " "
                line = f"   {i + 1:>2}  {mark} {m:<24}"
                if i == i_last:
                    line = st.hi(line + f"  <<< LAST SAVED (row {it})")
                elif i == i_next:
                    line = st.nxt(line + "  <-- NEXT")
                L.append(line)
            if i_last is not None and i_next == 0:
                L.append(st.dim("       (the cycle wraps: the next proposal is its first move)"))
            elif after and i_last is None:
                L.append(st.warn(f"       saved_after={after!r} is not in this stage's move_order "
                                 "(the row was saved by the previous stage)"))
    L.append(_rule("="))
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    one_line = "--line" in argv
    color = None
    if "--no-color" in argv:
        color = False
    if "--color" in argv:
        color = True
    n_rows = 5
    if "--rows" in argv:
        i = argv.index("--rows")
        try:
            n_rows = max(1, int(argv[i + 1]))
        except (IndexError, ValueError):
            print("usage: --rows N")
            return 2
        del argv[i:i + 2]
    args = [a for a in argv if a not in ("-v", "--line", "--no-color", "--color")]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[0])
        print("usage: gf_last_iteration.py <run_dir | store.h5> [--line] [--rows N] "
              "[--no-color | --color]")
        return 2
    it, store, after, stage = last_iteration(args[0])
    if one_line:
        print(f"{it}  {describe_stage(it, stage, True)}  saved_after={after}  "
              f"store={os.path.basename(store)}")
        return 0
    if color is None:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    run_dir = os.path.dirname(os.path.abspath(store))
    rows = read_rows(store, it, n_rows) if it > 0 else []
    try:
        live = read_live(run_dir)
    except Exception as exc:  # noqa: BLE001 -- a sidecar never takes the page down
        live = None
        print(f"(run log unreadable: {exc})", file=sys.stderr)
    try:
        gpu = read_gpu(run_dir)
    except Exception as exc:  # noqa: BLE001
        gpu = None
        print(f"(gpu csv unreadable: {exc})", file=sys.stderr)
    print(render(it, store, after, stage, rows, live, gpu, Style(color)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
