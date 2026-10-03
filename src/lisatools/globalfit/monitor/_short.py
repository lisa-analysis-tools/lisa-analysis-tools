"""The SHORT monitor page: the topline and the handful of panels people open
the full page for, built in well under a minute.

USER REQUEST 2026-10-03: "let's make a short version of the html as well.
That picks out the most important plots and topline info. phase maximized
overlap. current psd measurements. nleaves over time, likelihood over time.
Add what you think should be there but we want to keep it contained and easy
to generate."

    python -m lisatools.globalfit.monitor --short-page RUN_DIR [OUT.html]
    python -m lisatools.globalfit.monitor.from_tar --short-page SNAP.tar.gz
    GF_MONITOR_AFTER_SAVE=1 GF_MONITOR_PAGE_SHORT=1      (in-run, saver rank)

The page lands BESIDE the run folder as ``RUN_DIR_monitor_short.html`` (the
run directory's path plus ``_monitor_short.html``) -- a SEPARATE file. The
full page keeps its own name (``..._monitor.html``) and this module never
writes it.

CONTENT, in this order
----------------------
1. TOPLINE. Store, last saved row, stage (step k of N, rows in it), the leg
   the row was saved after and the next one, the galfor ratchet (the store's
   stop stamp, the latest gate line and verdict from the head log), the row
   cadence from the ``[SAVE]`` lines, head-node GPU utilisation and memory
   from the newest ``gpu_util_JOB.csv`` (the head node only -- a second node
   is not sampled), cold lnL max / mean / min at the last row and the change
   against the row before, cold leaf counts per branch, the overlap headline
   and the instrument-noise bias.
2. Cold log-likelihood against stored row: min / mean / max over walkers, the
   whole run and the recent window, with the recipe stage boundaries (start
   and completed iterations) and the ratchet gate rows marked.
3. GB leaf count against stored row, same layout. The other branches are
   fixed-dimension and are listed in the topline.
4. The phase-maximised, noise-weighted overlap of every proxy-matched
   (recovered, injected) pair at the last row for the max-lnL walker -- the
   full page's headline match criterion, computed the same way.
5. The CURRENT noise measurement: every cold walker's sensitivity at the last
   row against the injected instrument and the injected instrument + FittedHT
   foreground (and the ratio to the latter), plus the instrument-noise
   posteriors against their injected values.
6. Two additions: the seven noise parameters (and the foreground at 3 mHz)
   against stored row, and the ratchet gate rows read off the store.

WHY IT DOES NOT CALL THE FULL PAGE'S PANELS
-------------------------------------------
There are none to call. ``_generator.py`` is a script: it reads ``sys.argv``
and raises ``SystemExit(0)`` when imported, and every panel is a block of
MODULE-LEVEL statements closing over ~200 globals, not a function. Running it
means running all of it -- every per-rank log (about 2 GB on the 6-month
run), the mojito data bricks for the residual spectrum, the VGB catalogue.

The smallest adapter that reuses its code anyway: :func:`_generator_functions`
compiles NAMED top-level ``def`` statements out of the generator's syntax tree
into a namespace this module supplies. The generator's own torn-read ladder
(``_row`` / ``_backup_group``), pairing (``_match_pairs``), survival count
(``_survival``), zoom window (``_zoom_its``), sub-backend trim
(``_last_written``) and stage marker (``mark_stages``) run here unchanged, and
nothing else of the script executes. The physics goes through the same
importable routes the full page uses: ``build_truth.l1_orbits`` /
``build_truth.sens_grids`` (byte-for-byte the generator's ``SA_G``/``SE_G``),
GBGPU ``run_wave``, ``make_gb_transform_container``, ``get_sensitivity``. The
one inline block it cannot reach -- the per-pair overlap loop -- is mirrored in
:func:`phase_max_overlaps`, and ``tests/test_monitor_short_page.py`` executes
the generator's own loop against it so the two cannot drift.

WHAT IT DOES NOT READ: the per-rank logs, the mojito data bricks, the F-stat
caches, the VGB catalogue. The head log (``*_artifacts/globalfit_run.log``,
its ``_tail`` copy in a snapshot, else the newest ``slurm_stdout_JOB.log``)
is read from its last 48 MB only, as ``scripts/diagnostics/gf_last_iteration.py``
does. Its readers are mirrored here rather than imported because that script
lives outside the package and does not ship in the wheel.

EVERY SOURCE IS OPTIONAL except the store. A missing head log, GPU CSV, truth
set, mojito tree or GBGPU costs its own row or panel and a line under "Notes",
never the page.

Knobs (2026-10-03): ``GF_MONITOR_PAGE_SHORT=1`` makes the in-run hook build
this page INSTEAD of the full one (``hooks.py``); ``GF_MONITOR_MATCH_STATS=0``
drops the overlap section, as it does on the full page;
``GF_MONITOR_MATCH_MM`` sets the overlap threshold for "matched" (default
0.8, as on the full page); ``MOJITO_INFO_PATH`` locates the L1 orbits and the
noise brick, resolved exactly as for the full page.
"""

from __future__ import annotations

import base64
import glob
import html as _html
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["build_short_monitor", "default_short_out_path", "render_short_page",
           "phase_max_overlaps"]

#: The generator's own top-level helpers this page runs unchanged.
GEN_HELPERS = ("mark_stages", "_zoom_its", "_last_written", "_backup_group",
               "_row", "_match_pairs", "_survival")

#: Stores that are never the live one (same list as gf_last_iteration.py).
SKIP = ("backup", "CORRUPT", "pre_rerung", "pre_repair", "midit", ".stale")
#: How much of the head log is read: about half a day, enough saves for a cadence.
TAIL_BYTES = 48 * 1024 * 1024
#: The generator's f0 pairing tolerance, in FD bins, and its overlap waveform length.
TOL_BINS = 2.0
NW_OVERLAP = 1024
#: The generator's fallback GB phase epoch when run_settings.log is absent.
T_REF_DEFAULT = 97729089.327664
#: Rows of the ratchet gate table.
GATE_ROWS_SHOWN = 8

# The full page's palette, one meaning per hue (see _generator.py's docstring).
BG, PANEL, LINE, FG, DIM = "#0A0E14", "#10161F", "#223041", "#B8C6D4", "#67788A"
CYAN, AMBER, GREEN, RED, VIOLET = "#4FD8EB", "#F5A623", "#58C48A", "#E5484D", "#9B7BFF"

#: The full page's figure style, applied through ``rc_context`` so building
#: this page never changes the CALLER's rcParams (it runs on the saver rank).
#: Set in full, font size included: importing lisatools reaches eryn, which
#: restyles matplotlib as an import side effect (font.size 10 -> 16).
STYLE = {
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110, "text.usetex": False,
}

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def default_short_out_path(run_dir: str) -> str:
    """``RUN_DIR`` + ``_monitor_short.html``: a sibling of the run folder, like
    the full page, under a name of its own so neither overwrites the other."""
    return os.path.abspath(str(run_dir).rstrip("/")) + "_monitor_short.html"


# --------------------------------------------------------------------------
# the adapter onto the generator's own helpers
# --------------------------------------------------------------------------
def _generator_functions(ns: dict, names=GEN_HELPERS) -> dict:
    """Compile the named top-level ``def``s of ``_generator.py`` into ``ns``.

    Only those function definitions are executed -- defining a function runs
    none of its body -- so nothing of the script's module-level flow happens.
    Each function resolves its globals in ``ns``: ``_row`` wants ``g``,
    ``NIT``, ``RUN_DIR``, ``_BACKUP_G``, ``_TORN_ROWS``; ``mark_stages`` wants
    ``STAGE_BOUNDS``; the rest want ``np`` at most. Raises if a name is no
    longer a top-level ``def`` there, rather than silently running without it.
    """
    import ast

    from . import generator_path

    path = generator_path()
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    defs = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = sorted(set(names) - {n.name for n in defs})
    if missing:
        raise RuntimeError(
            f"_generator.py no longer defines {missing} at top level; the "
            "short page reuses them by name")
    exec(compile(ast.Module(body=defs, type_ignores=[]), path, "exec"), ns)  # noqa: S102
    return ns


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
def find_store(run_dir: str) -> str:
    """The newest ``*.h5`` in ``run_dir`` that is not a backup or quarantined
    copy. A reduced ``*_extract.h5`` counts: a snapshot ships nothing else."""
    cands = [p for p in glob.glob(os.path.join(run_dir, "*.h5"))
             if not any(s in os.path.basename(p) for s in SKIP)]
    if not cands:
        raise FileNotFoundError(f"no run store (*.h5) in {run_dir}")
    return max(cands, key=os.path.getmtime)


def _s(raw) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _attr(a, key, default=None):
    v = a.get(key, default)
    if v is None:
        return None
    try:
        return v.item() if hasattr(v, "item") else v
    except Exception:  # noqa: BLE001
        return v


def _recipe_steps(g) -> list:
    """Every recipe step as a dict, in order. Empty without a recipe group."""
    rg = g.get("recipe")
    if rg is None:
        return []
    steps = []
    for name in rg:
        a = rg[name].attrs
        mo = _attr(a, "move_order")
        try:
            mo = list(json.loads(_s(mo))) if mo is not None else None
        except Exception:  # noqa: BLE001
            mo = None
        steps.append(dict(
            name=str(name), order=int(_attr(a, "order num", 0) or 0),
            status=bool(_attr(a, "status", False)),
            start=_attr(a, "start_iteration"),
            completed=_attr(a, "completed_iteration"),
            ratchet_done=_attr(a, "galfor_ratchet_done"), move_order=mo))
    steps.sort(key=lambda s: s["order"])
    return steps


def stage_bounds(steps: list) -> list:
    """``[(iteration, label), ...]`` for ``mark_stages``.

    A step's ``start_iteration`` is drawn when the step is done or current,
    its ``completed_iteration`` only when the step is done -- a step reset by
    a stage migration keeps stale stamps from the abandoned trajectory (the
    6-month store carries ``full_pe start_iteration=55`` while it is still in
    gb_search_3), and drawing those would mark a boundary that never happened.
    Boundaries at the same row share one label.
    """
    cur = next((s for s in steps if not s["status"]), None)
    lim = cur["order"] if cur is not None else float("inf")
    at = {}
    for s in steps:
        if s["order"] > lim:
            continue
        if s["status"] and s["completed"] is not None:
            at.setdefault(int(s["completed"]), [[], []])[0].append(s["name"])
        if s["start"] is not None:
            at.setdefault(int(s["start"]), [[], []])[1].append(s["name"])
    out = []
    for it, (done, start) in sorted(at.items()):
        lab = " / ".join(f"{n} done" for n in done)
        if start:
            lab = (lab + " -> " if lab else "-> ") + " / ".join(start)
        out.append((it, lab))
    return out


def read_store(store: str, run_dir: str, ns: dict, notes: list) -> dict:
    """Everything the page needs from the store, as plain arrays.

    Row-count rules are the full page's: the filled extent of ``log_like``,
    lowered to the ``iteration`` attr on a rewound store, then stepped back
    past a torn last row whose ``inds/gb`` is still empty. Every dataset read
    goes through the generator's ``_row``, so a chunk torn by a racing saver
    falls back to the run's backup copy and is named under Notes.
    """
    import h5py

    S = dict(store=store, extract="_extract" in os.path.basename(store))
    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        ns.update(g=g, NIT=0, RUN_DIR=run_dir, _BACKUP_G=None, _TORN_ROWS=[],
                  h5py=h5py, os=os, np=np)
        try:
            _row = ns["_row"]
            ll_all = _row("log_like", (slice(None), 0, 0))
            if ll_all is None:
                raise RuntimeError("log_like is unreadable in the store and its backup")
            ll_all = np.asarray(ll_all, float)
            filled = np.where(np.any(ll_all != 0.0, axis=1))[0]
            if not filled.size:
                raise RuntimeError(f"{os.path.basename(store)}: no filled rows yet")
            nit = int(filled.max()) + 1
            it_attr = g.attrs.get("iteration")
            S["it_attr"] = int(it_attr) if it_attr is not None else None
            if it_attr is not None and 0 < int(it_attr) < nit:
                notes.append(f"store rewound: iteration attr {int(it_attr)} &lt; "
                             f"filled rows {nit}; plotting the live {int(it_attr)} rows.")
                nit = int(it_attr)
            ns["NIT"] = nit

            branches = sorted(g["inds"].keys()) if "inds" in g else []
            inds_gb = (_row("inds/gb", (slice(0, nit), 0, 0))
                       if "gb" in branches else None)
            if inds_gb is not None and nit > 0 and not inds_gb[nit - 1].any():
                back = 0
                for j in range(nit - 2, max(nit - 40, -1), -1):
                    if inds_gb[j].any():
                        back = nit - 1 - j
                        break
                if back:
                    notes.append(f"stored row {nit - 1} carries no GB leaves (a torn "
                                 f"save); plotting through row {nit - 1 - back}.")
                    nit -= back
                    ns["NIT"] = nit
                else:
                    notes.append(f"stored row {nit - 1} and the 40 before it carry no "
                                 "GB leaves; check the store before reading the GB panels.")
            S["nit"] = nit
            S["ll"] = ll_all[:nit]
            S["nwalk"] = int(S["ll"].shape[1])

            leaves = {}
            for b in branches:
                ind = ((inds_gb[:nit] if inds_gb is not None else None) if b == "gb"
                       else _row(f"inds/{b}", (slice(0, nit), 0, 0)))
                if ind is not None:
                    leaves[b] = np.asarray(ind).sum(axis=-1).astype(int)
            S["leaves"] = leaves

            S["saved_after"] = []
            if "saved_after" in g:
                sa = _row("saved_after", (slice(0, nit),))
                if sa is not None:
                    S["saved_after"] = [_s(x) for x in sa]

            S["steps"] = _recipe_steps(g)

            # noise chains: the sub-backend's cold rung, trimmed to the last
            # row every noise branch has actually written (_last_written)
            def _noise(b):
                if f"sub_backend/{b}/chain" in g:
                    a = _row(f"sub_backend/{b}/chain", (slice(0, nit),))
                    return None if a is None else np.asarray(a)[:, 0, :, 0, :]
                if f"chain/{b}" in g:
                    a = _row(f"chain/{b}", (slice(0, nit), 0, 0))
                    return None if a is None else np.asarray(a)[:, :, 0, :]
                return None

            S["psd"], S["gal"] = _noise("psd"), _noise("galfor")
            if S["psd"] is not None and S["gal"] is not None:
                sub_nit = min(ns["_last_written"](S["psd"]), ns["_last_written"](S["gal"]))
                if sub_nit < nit:
                    notes.append(f"noise chains written through row {sub_nit - 1} while "
                                 f"the main backend reached {nit - 1} (caught mid-flush); "
                                 "the noise panels use the last complete row.")
                S["psd"], S["gal"] = S["psd"][:sub_nit], S["gal"][:sub_nit]

            try:
                a = dict(g["domain_settings/args"].attrs)
                S["tobs"] = float(a["0"]) * float(a["1"]) * float(a["2"])
            except Exception:  # noqa: BLE001
                S["tobs"] = None
                notes.append("no domain_settings in the store; Tobs unknown, so "
                             "no overlap panel.")

            # the max-lnL walker's GB leaves at the last row, for the overlap
            S["wb"] = int(np.argmax(S["ll"][nit - 1]))
            S["rec9"] = None
            if inds_gb is not None:
                row = _row("chain/gb", (nit - 1, 0, 0, S["wb"]))
                if row is not None:
                    S["rec9"] = np.asarray(row, float)[np.asarray(inds_gb[nit - 1, S["wb"]], bool)]
            S["torn_rows"] = list(ns["_TORN_ROWS"])
        finally:
            bg = ns.get("_BACKUP_G")
            if bg:
                try:
                    bg.file.close()
                except Exception:  # noqa: BLE001
                    pass
            ns.update(g=None, _BACKUP_G=None)

    from lisatools.globalfit.stock.erebor.noise import (
        galfor_params_to_physical, read_noise_model_identity)

    S["galfor_log"] = bool(read_noise_model_identity(store).get("galfor_log_sampling", False))
    S["gal_phys"] = (galfor_params_to_physical(S["gal"], S["galfor_log"])
                     if S["gal"] is not None else None)
    if S["torn_rows"]:
        notes.append("rows read around a racing saver: " + "; ".join(S["torn_rows"]))
    return S


def current_stage(steps: list, nit: int, saved_after: list) -> Optional[dict]:
    """The step a resume would run, with rows-in and the next leg."""
    if not steps:
        return None
    cur = next((s for s in steps if not s["status"]), None)
    out = dict(total=len(steps), done=[s["name"] for s in steps if s["status"]],
               name=None, order=None, start=None, rows_in=None,
               ratchet_done=None, next_leg=None)
    if cur is None:
        return out
    out.update(name=cur["name"], order=cur["order"], start=cur["start"],
               ratchet_done=cur["ratchet_done"])
    if cur["start"] is not None:
        out["rows_in"] = nit - int(cur["start"])
    mo, last = cur["move_order"], (saved_after[-1] if saved_after else None)
    if mo and last in mo:
        out["next_leg"] = mo[(mo.index(last) + 1) % len(mo)]
    return out


def gate_rows(saved_after: list, gal_phys, ll, leaves_gb) -> list:
    """Rows saved after the gated noise step, classified off the galfor chain.

    ``nudge``: amplitude moved with alpha and f_1 held -- the forced step (the
    full page's rule, ``fg_ratchet_timeline``). ``moved``: the noise was
    sampled (a release, or noise PE after one). ``held``: nothing moved. Each
    entry also carries what the search legs did until the next gate row.
    """
    if gal_phys is None or not saved_after:
        return []
    n = min(len(saved_after), gal_phys.shape[0], ll.shape[0])
    rows = [r for r in range(1, n) if saved_after[r] == "noise_ratchet_search"]
    out = []
    for j, r in enumerate(rows):
        a, b = gal_phys[r], gal_phys[r - 1]
        nudge = (np.allclose(a[:, 2], b[:, 2], rtol=0.0, atol=1e-9)
                 and np.allclose(a[:, 3], b[:, 3], rtol=1e-9, atol=0.0)
                 and not np.allclose(a[:, 0], b[:, 0], rtol=1e-9, atol=0.0))
        kind = ("nudge" if nudge else
                "held" if np.allclose(a, b, rtol=1e-12, atol=0.0) else "moved")
        end = (rows[j + 1] - 1) if j + 1 < len(rows) else (ll.shape[0] - 1)
        rec = dict(row=r, kind=kind, dll=float(ll[r].mean() - ll[r - 1].mean()),
                   end=end, dll_after=float(ll[end].mean() - ll[r].mean()))
        if leaves_gb is not None and leaves_gb.shape[0] > end:
            rec.update(leaves=float(leaves_gb[r].mean()),
                       dleaves_after=float(leaves_gb[end].mean() - leaves_gb[r].mean()))
        out.append(rec)
    return out


def run_settings_t0(run_dir: str) -> float:
    """The GB branch's phase epoch from run_settings.log (the generator's regex)."""
    for p in glob.glob(os.path.join(run_dir, "*_artifacts", "run_settings.log")):
        try:
            with open(p, errors="replace") as fh:
                m = re.search(r"\[gb\].*?\n\s+t0:\s*([\d.]+)", fh.read(), re.S)
            if m:
                return float(m.group(1))
        except OSError:
            continue
    return T_REF_DEFAULT


# --------------------------------------------------------------------------
# the head log and the GPU sampler (mirrors gf_last_iteration.py)
# --------------------------------------------------------------------------
def _tail_lines(path, nbytes=TAIL_BYTES):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if size > nbytes:
            fh.seek(size - nbytes)
            fh.readline()                               # drop the torn first line
        data = fh.read()
    return data.decode("utf-8", "replace").splitlines()


def find_head_log(run_dir: str) -> Optional[str]:
    """The head's run log, its snapshot ``_tail`` copy, or the newest job stdout
    (a superset of it, and the only log a short tar carries)."""
    for name in ("globalfit_run.log", "globalfit_run_tail.log"):
        hits = glob.glob(os.path.join(run_dir, "*_artifacts", name))
        if hits:
            return max(hits, key=os.path.getmtime)
    hits = [p for p in glob.glob(os.path.join(run_dir, "slurm_stdout_*.log"))
            if not p.endswith(("_filtered.log", "_tail.log"))]
    return max(hits, key=os.path.getmtime) if hits else None


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


def read_live(run_dir: str, now=None) -> Optional[dict]:
    """Pace and gate from the head log's tail, or ``None`` without a log."""
    path = find_head_log(run_dir)
    if path is None:
        return None
    now = now or datetime.now()
    out = dict(log=path, last_ts=None, log_age_min=None, saves=[], cadence=[],
               gate=None, gate_ts=None, verdict=None, replica=None, stage_line=None)
    for line in _tail_lines(path):
        t = _ts(line)
        if t is not None:
            out["last_ts"] = t
        if "[SAVE] save_step" in line and t is not None:
            out["saves"].append(t)
        elif "[GALFOR_RATCHET" in line:
            body = line.split("] ", 1)[-1] if "] " in line else line
            if re.search(r"\] iteration \d+: (NUDGE|RELEASE|HOLD)", line):
                out["gate"], out["gate_ts"] = body.split(" -- ")[0].strip(), t
            elif (re.search(r"\] after (RELEASE|iteration)", line)
                  or "RATCHET DONE" in line or "already FINISHED" in line):
                out["verdict"] = body
        elif "[REPLICA_PE" in line and "stored row" in line:
            out["replica"] = line.split("] ", 1)[-1]
        elif "[V9-STAGE" in line and ("entering" in line or "STAGE COMPLETE" in line):
            out["stage_line"] = line.split(" - INFO - ", 1)[-1]
    if out["last_ts"] is not None:
        out["log_age_min"] = _mins(out["last_ts"], now)
    s = out["saves"][-7:]
    out["cadence"] = [_mins(a, b) for a, b in zip(s[:-1], s[1:])]
    return out


def verdict_phrase(v: Optional[str]) -> Optional[str]:
    """One phrase from a ratchet verdict line (gf_last_iteration's reading,
    plus the release-before-reference case)."""
    if not v:
        return None
    head = v.split(":", 1)[0].strip()
    if "RATCHET DONE" in v or "already FINISHED" in v:
        return head + ": RATCHET DONE"
    if "recorded as the reference" in v or "baseline" in v:
        return head + ": reference recorded"
    if "reference captured yet" in v:
        return head + ": no pre-nudge reference yet"
    m = re.search(r"drop ([+-][\d.]+%) \(threshold ([\d.]+%)", v)
    if m:
        tail = "continues" if "continues" in v else ("STOP" if "stop" in v.lower() else "")
        return f"{head}: foreground drop {m.group(1)} vs {m.group(2)} -> {tail}".rstrip(" ->")
    m = re.search(r"([+-][\d.,]+) over the previous release \(threshold (\d+)\)", v)
    if m:
        tail = "continues" if "continues" in v else ""
        return f"{head}: lnL gain {m.group(1)} vs {m.group(2)} {tail}".strip()
    return (head + ": " + v.split(":", 1)[-1].strip())[:90]


def read_gpu(run_dir: str, now=None, window_s: float = 60.0) -> Optional[dict]:
    """The newest ``gpu_util_JOB.csv``: per GPU its last sample and the mean
    utilisation over the trailing ``window_s``. The launcher's sampler runs on
    the HEAD node only."""
    cands = glob.glob(os.path.join(run_dir, "gpu_util_*.csv"))
    if not cands:
        return None
    path = max(cands, key=os.path.getmtime)
    now = now or datetime.now()
    rows = []
    for line in _tail_lines(path, 256 * 1024)[-400:]:
        p = [x.strip() for x in line.split(",")]
        if len(p) < 9:
            continue
        try:
            rows.append(dict(t=datetime.strptime(p[0].split(".")[0], "%Y/%m/%d %H:%M:%S"),
                             idx=int(p[1]), util=float(p[3]), used=float(p[5]) / 1024.0,
                             total=float(p[6]) / 1024.0, power=float(p[7]), temp=float(p[8])))
        except ValueError:
            continue
    if not rows:
        return None
    t_last = max(r["t"] for r in rows)
    per = {}
    for r in rows:
        if (t_last - r["t"]).total_seconds() <= window_s:
            per.setdefault(r["idx"], []).append(r)
    gpus = {}
    for idx, rs in sorted(per.items()):
        last = max(rs, key=lambda r: r["t"])
        gpus[idx] = dict(util=last["util"], util_avg=sum(r["util"] for r in rs) / len(rs),
                         mem_used_gb=last["used"], mem_total_gb=last["total"],
                         power_w=last["power"], temp_c=last["temp"])
    return dict(csv=path, t_last=t_last, age_min=_mins(t_last, now), gpus=gpus)


# --------------------------------------------------------------------------
# the phase-maximised overlap
# --------------------------------------------------------------------------
def phase_max_overlaps(Ar, Er, sr, At, Et, st, SA, SE, nw):
    """``|<a|b>| / sqrt(<a|a><b|b>)`` over the A and E channels, per pair.

    The modulus IS the maximum over an overall phase, so no phase grid is
    needed. ``Ar/Er`` (``At/Et``) are the recovered (injected) waveforms,
    ``nw`` FD points each starting at FD bin ``sr`` (``st``); ``SA/SE`` the
    noise on the full FD grid. A pair whose window leaves the grid scores 0.

    THIS MIRRORS THE GENERATOR'S INLINE LOOP (the ``MM = np.zeros(MI.size)``
    block of ``_generator.py``'s recovery section) statement for statement;
    ``tests/test_monitor_short_page.py`` executes that block against this
    function on the same inputs.
    """
    n = len(sr)
    MM = np.zeros(n)
    for i in range(n):
        o = min(sr[i], st[i])
        sp = max(sr[i], st[i]) + nw - o
        if o < 0 or o + sp > SA.size:
            continue
        sa = SA[o:o + sp]
        se = SE[o:o + sp]
        a1 = np.zeros(sp, complex)
        e1 = np.zeros(sp, complex)
        a2 = np.zeros(sp, complex)
        e2 = np.zeros(sp, complex)
        k = sr[i] - o
        a1[k:k + nw] = Ar[i]
        e1[k:k + nw] = Er[i]
        k = st[i] - o
        a2[k:k + nw] = At[i]
        e2[k:k + nw] = Et[i]
        num = np.sum(np.conj(a1) * a2 / sa) + np.sum(np.conj(e1) * e2 / se)
        n1 = np.sum(np.abs(a1) ** 2 / sa) + np.sum(np.abs(e1) ** 2 / se)
        n2 = np.sum(np.abs(a2) ** 2 / sa) + np.sum(np.abs(e2) ** 2 / se)
        MM[i] = float(np.abs(num) / np.sqrt(max(n1 * n2, 1e-300)))
    return np.clip(MM, 0.0, 1.0)


def _l1_orbits_from(mojito_tree):
    """``(orbits, label)``: the injected mojito L1 orbits via build_truth, the
    tree handed over through ``MOJITO_INFO_PATH`` for the duration of the call."""
    from .build_truth import l1_orbits

    saved = os.environ.get("MOJITO_INFO_PATH")
    try:
        if mojito_tree:
            os.environ["MOJITO_INFO_PATH"] = mojito_tree
        orb, where = l1_orbits()
    finally:
        if saved is None:
            os.environ.pop("MOJITO_INFO_PATH", None)
        else:
            os.environ["MOJITO_INFO_PATH"] = saved
    return orb, where


def compute_overlap(S: dict, truth_path: str, ns: dict, mojito_tree, t_ref: float,
                    notes: list, thresh: float) -> Optional[dict]:
    """The full page's headline match at the last row, for the max-lnL walker.

    Same route as ``_generator.py``'s recovery section: the frozen truth set's
    detectable sources in its stamped band, a globally-greedy one-to-one f0
    pairing within 2 FD bins (``_match_pairs``), then the phase-maximised,
    noise-weighted overlap of each pair under the walker-median noise of the
    last row, with GBGPU waveforms on the injected L1 orbits. ``matched`` =
    overlap at or above ``thresh`` (``GF_MONITOR_MATCH_MM``).
    """
    T = np.load(truth_path)
    tobs = S["tobs"]
    df = 1.0 / tobs
    flo, fhi = 3e-3, 21.94e-3                     # the generator's pre-stamp fallback
    if "band" in T.files:
        b = np.asarray(T["band"], float).reshape(-1)
        if b.size >= 2 and 0 < b[0] < b[1]:
            flo, fhi = float(b[0]), float(b[1])
    sel = np.asarray(T["det"], bool) & (T["f0"] >= flo) & (T["f0"] <= fhi)
    t_f0 = np.asarray(T["f0"], float)[sel]
    ndet = int(sel.sum())
    out = dict(ndet=ndet, band=(flo, fhi), walker=S["wb"], row=S["nit"] - 1,
               thresh=thresh, chance=ndet * (2 * TOL_BINS * df) / (fhi - flo))
    rec9 = S["rec9"]
    if rec9 is None or not rec9.size or not np.any(rec9[:, 1] != 0.0):
        notes.append("overlap: the max-lnL walker carries no GB coordinates at the "
                     "last row (unreadable, or outside an extract's keep window).")
        return None
    f_rec = rec9[:, 1] * 1e-3
    inb = (f_rec >= flo) & (f_rec <= fhi)
    rec9, f_rec = rec9[inb], f_rec[inb]
    MI, TI, _ = ns["_match_pairs"](f_rec, t_f0, TOL_BINS * df)
    out.update(n_band=int(rec9.shape[0]), n_proxy=int(MI.size),
               completeness_proxy=MI.size / max(ndet, 1),
               purity_proxy=MI.size / max(rec9.shape[0], 1), mm=None)
    try:
        from gbgpu.gbgpu import GBGPU
        from lisatools import detector as lisa_models
        from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

        from .build_truth import sens_grids

        SA, SE = sens_grids(np.median(S["psd"][-1], axis=0),
                            np.median(S["gal_phys"][-1], axis=0), df)
        orb, where = _l1_orbits_from(mojito_tree)
        if orb is None:
            notes.append("overlap: no mojito L1 brick reachable, so the waveforms use "
                         "the analytic DefaultOrbits ephemeris -- overlaps above ~5 mHz "
                         "carry an annual-Doppler phase error.")
            orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
            out["orbits"] = "analytic DefaultOrbits"
        else:
            out["orbits"] = "mojito L1 (" + os.path.basename(str(where)) + ")"
        gbw = GBGPU(force_backend="cpu", orbits=orb, t0=float(t_ref))
        tc = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                         use_distance=True, mc_lims=(0.001, 1.0))
        rphys = tc.both_transforms(np.asarray(rec9[MI], float).copy())
        tphys = np.asarray(T["phys"], float)[sel][TI]

        def _ae(phys):
            gbw.run_wave(*[np.ascontiguousarray(phys[:, k]) for k in range(9)],
                         N=NW_OVERLAP, T=tobs, dt=2.5, tdi2=True, tdi_channel_setup="AE")
            return (np.array(gbw.A), np.array(gbw.E),
                    np.asarray(gbw.start_inds).astype(int).copy())

        Ar, Er, sr = _ae(rphys)
        At, Et, st = _ae(tphys)
        mm = phase_max_overlaps(Ar, Er, sr, At, Et, st, SA, SE, NW_OVERLAP)
    except Exception as e:  # noqa: BLE001
        notes.append(f"overlap: waveform-level statistics unavailable "
                     f"({type(e).__name__}: {_html.escape(str(e))}); the headline falls "
                     "back to the 2-bin f0 proxy.")
        return out
    kept = mm >= thresh
    out.update(mm=mm, t_f0=t_f0[TI], n_match=int(kept.sum()),
               completeness=float(kept.sum()) / max(ndet, 1),
               purity=float(kept.sum()) / max(rec9.shape[0], 1),
               mm_med=float(np.median(mm)) if mm.size else float("nan"),
               mm_hi=float(np.mean(mm > 0.9)) if mm.size else 0.0)
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _png(fig, dpi=None) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    return base64.b64encode(buf.getvalue()).decode()


def _gate_marks(ax, gates, lo, hi):
    for gr in gates:
        if lo <= gr["row"] <= hi:
            ax.axvline(gr["row"], color=AMBER if gr["kind"] == "nudge" else GREEN,
                       lw=0.8, ls=(0, (2, 2)), alpha=0.7, zorder=1)


def _band_panel(ax, x, y, color, ylabel):
    ax.fill_between(x, y.min(axis=1), y.max(axis=1), color=color, alpha=0.22, lw=0,
                    label="min-max over walkers")
    ax.plot(x, y.max(axis=1), color=color, lw=1.0)
    ax.plot(x, y.min(axis=1), color=color, lw=1.0)
    ax.plot(x, y.mean(axis=1), color=FG, lw=1.5, label="mean")
    ax.set_xlabel("stored row")
    ax.set_ylabel(ylabel)


def fig_history(ns, y, gates, color, ylabel, title):
    """Whole run (left) and the recent window (right): min / mean / max over
    walkers, stage boundaries via the generator's ``mark_stages``."""
    from matplotlib.figure import Figure

    nit = y.shape[0]
    x = np.arange(nit)
    nz = max(int(ns["_zoom_its"](nit)), 1)
    fig = Figure(figsize=(11.6, 3.5))
    ax = fig.subplots(1, 2)
    every = ns["STAGE_BOUNDS"]
    for a, sl, ttl in ((ax[0], slice(0, nit), f"{title}, all {nit} rows"),
                       (ax[1], slice(nit - nz, nit), f"last {nz} rows")):
        _band_panel(a, x[sl], y[sl], color, ylabel)
        lo, hi = int(x[sl][0]), int(x[sl][-1])
        # mark_stages has a max_it but no min_it, and its labels use a blended
        # (data, axes) transform that is NOT clipped -- a boundary left of the
        # zoom window would print its label beside the panel. Hand it only the
        # boundaries inside this window.
        ns["STAGE_BOUNDS"] = [b for b in every if lo <= b[0] <= hi]
        try:
            ns["mark_stages"](a, label=True, max_it=hi)
        finally:
            ns["STAGE_BOUNDS"] = every
        _gate_marks(a, gates, lo, hi)
        a.set_xlim(lo - 0.5, hi + 0.5)
        a.set_title(ttl)
    ax[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _png(fig)


def fig_overlap(ns, ov):
    from matplotlib.figure import Figure

    mm = np.asarray(ov["mm"], float)
    fig = Figure(figsize=(11.6, 3.8))
    ax = fig.subplots(1, 2)
    s, n = ns["_survival"](mm)
    ax[0].plot(s, n, color=GREEN, lw=1.8, label=f"{mm.size} f0-paired sources")
    ax[0].axhline(ov["ndet"], color=FG, ls="--", lw=1.0)
    ax[0].text(0.02, ov["ndet"], f" {ov['ndet']} detectable injections", color=FG,
               fontsize=8, va="bottom")
    ax[0].axvline(ov["thresh"], color=RED, ls=":", lw=1.2)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1, max(2500, 1.5 * ov["ndet"]))
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("phase-maximised overlap with the paired injection")
    ax[0].set_ylabel("sources at or above")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[1].scatter(ov["t_f0"] * 1e3, mm, s=5, color=CYAN, alpha=0.6, lw=0)
    ax[1].axhline(ov["thresh"], color=RED, ls=":", lw=1.2)
    ax[1].set_xscale("log")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("injected f0 [mHz]")
    ax[1].set_ylabel("overlap")
    ax[1].set_title(f"row {ov['row']}, max-lnL walker {ov['walker']}")
    fig.tight_layout()
    return _png(fig)


def noise_figures(S, soms_inj, sa_inj, gates, ns):
    """``(psd_now, psd_params, noise_traces)`` PNGs: the current sensitivity
    per cold walker, the instrument posteriors, and the parameter traces."""
    from matplotlib.figure import Figure

    from lisatools import detector as lisa_models
    from lisatools.globalfit.stock.erebor.noise import GALFOR_BASIS, GALFOR_LOG_PARAMS
    from lisatools.sensitivity import LISASens, get_sensitivity
    from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground as FHT
    from lisatools.stochastic import HyperbolicTangentGalacticForeground as HTGF

    psd, gph, gal = S["psd"], S["gal_phys"], S["gal"]
    nw, nsub = psd.shape[1], psd.shape[0]
    orb0 = lisa_models.DefaultOrbits()
    fr = np.logspace(np.log10(2e-4), np.log10(2.6e-2), 500)

    def sens(soms, sa, galp=None):
        model = lisa_models.LISAModel(soms ** 2, sa ** 2, orb0, "mon")
        if galp is None:
            return np.asarray(get_sensitivity(fr, sens_fn=LISASens, model=model,
                                              stochastic_params=()), float)
        return np.asarray(get_sensitivity(fr, sens_fn=LISASens, model=model,
                                          stochastic_params=tuple(galp),
                                          stochastic_function=HTGF), float)

    pm = np.median(psd[-1], axis=0)
    inj = sens(soms_inj, sa_inj)
    ref = inj + np.asarray(FHT.specific_Sh_function(fr, S["tobs"] or 7776000.0), float)
    fig = Figure(figsize=(11.6, 4.0))
    ax = fig.subplots(1, 2)
    ax[0].plot(fr, sens(*pm), color=CYAN, lw=1.6, label="instrument (walker median)")
    for w in range(nw):
        cw = sens(psd[-1, w, 0], psd[-1, w, 1], gph[-1, w])
        ax[0].plot(fr, cw, color=AMBER, lw=1.2, alpha=0.75,
                   label="instrument + foreground, per walker" if w == 0 else None)
        ax[1].plot(fr, cw / ref, color=AMBER, lw=1.2, alpha=0.75)
    ax[0].plot(fr, inj, color=RED, ls=":", lw=1.3, label="injected instrument")
    ax[0].plot(fr, ref, color=RED, ls="-.", lw=1.3, label="injected + FittedHT")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("f [Hz]")
    ax[0].set_ylabel("Sn(f) [LISASens]")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"sensitivity at row {nsub - 1}, all {nw} cold walkers")
    ax[1].axhline(1.0, color=FG, lw=1.0, ls=":")
    ax[1].set_xscale("log")
    ax[1].set_xlim(2e-4, 2.6e-2)
    ax[1].set_xlabel("f [Hz]")
    ax[1].set_ylabel("fitted / (injected + FittedHT)")
    ax[1].set_title("each walker over the injected reference")
    fig.tight_layout()
    psd_now = _png(fig)

    nsh = min(3, nsub)
    fig = Figure(figsize=(11.6, 2.9))
    ax = fig.subplots(1, 2)
    for j, (name, injv, unit) in enumerate((("Soms_d", soms_inj, "m"),
                                            ("Sa_a", sa_inj, "m/s$^2$"))):
        v = psd[-nsh:, :, j].ravel()
        bias = float(np.median(v) / injv - 1.0)
        ax[j].hist(v, bins=max(8, min(26, v.size)), color=VIOLET, alpha=0.85)
        ax[j].axvline(injv, color=CYAN, lw=1.6, ls="--", label="injected")
        ax[j].set_title(f"{name} [{unit}]: median {100 * bias:+.2f}% from injected",
                        fontsize=10)
        ax[j].legend(fontsize=8)
        ax[j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle(f"instrument-noise posteriors, last {nsh} rows x {nw} cold walkers",
                 fontsize=10, color=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    psd_params = _png(fig)

    x = np.arange(nsub)
    names = (["Soms_d", "Sa_a"]
             + [("log10 " + n) if (S["galfor_log"] and n in GALFOR_LOG_PARAMS) else n
                for n in GALFOR_BASIS] + ["S_gal(3 mHz)"])
    s3 = np.array([[float(np.asarray(HTGF.specific_Sh_function(np.array([3e-3]),
                                                               *gph[i, w])).ravel()[0])
                    for w in range(nw)] for i in range(nsub)])
    series = [psd[:, :, 0], psd[:, :, 1]] + [gal[:, :, k] for k in range(5)] + [s3]
    fig = Figure(figsize=(13.6, 5.2))
    ax = fig.subplots(2, 4).ravel()
    for k, (nm, ys) in enumerate(zip(names, series)):
        col = CYAN if k < 2 else AMBER
        for w in range(nw):
            ax[k].plot(x, ys[:, w], color=col, lw=0.9, alpha=0.6)
        if k == 0:
            ax[k].axhline(soms_inj, color=RED, lw=1.1, ls=":")
        elif k == 1:
            ax[k].axhline(sa_inj, color=RED, lw=1.1, ls=":")
        if k == 7:
            ax[k].set_yscale("log")
        ns["mark_stages"](ax[k], label=False)
        _gate_marks(ax[k], gates, 0, nsub - 1)
        ax[k].set_xlim(-0.5, nsub - 0.5)
        ax[k].set_title(nm, fontsize=9)
        ax[k].tick_params(labelsize=7)
    fig.tight_layout()
    traces = _png(fig)
    return psd_now, psd_params, traces


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
CSS = """
:root {
  --bg:#0A0E14; --panel:#10161F; --line:#223041; --fg:#B8C6D4; --dim:#67788A;
  --cyan:#4FD8EB; --amber:#F5A623; --green:#58C48A; --red:#E5484D; --violet:#9B7BFF;
}
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; margin:0; }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
  padding:10px 16px; z-index:5; display:flex; flex-wrap:wrap; gap:8px 18px; align-items:baseline; }
header h1 { font-size:15px; margin:0; letter-spacing:.06em; color:var(--cyan); text-transform:uppercase; }
header .stamp { color:var(--dim); font-size:12px; }
nav { display:flex; flex-wrap:wrap; gap:6px; padding:8px 16px; border-bottom:1px solid var(--line); }
nav a { color:var(--dim); text-decoration:none; font-size:12px; padding:2px 8px; border:1px solid var(--line); border-radius:3px; }
nav a:hover, nav a:focus { color:var(--cyan); border-color:var(--cyan); outline:none; }
main { max-width:1240px; margin:0 auto; padding:16px 16px 60px; }
section { margin-top:26px; }
h2 { font-size:13px; letter-spacing:.1em; text-transform:uppercase; color:var(--fg);
  border-bottom:1px solid var(--line); padding-bottom:6px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:12px; margin-top:12px; overflow-x:auto; }
.panel img { max-width:100%; display:block; margin:0 auto; }
.caption { color:var(--dim); font-size:12px; margin-top:6px; }
.chip { display:inline-block; border:1px solid var(--line); border-radius:3px; padding:2px 9px; font-size:12px; color:var(--dim); }
.chip.done { color:var(--green); border-color:var(--green); }
.chip.now { color:var(--amber); border-color:var(--amber); }
.kpi { display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; }
.kpi div { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:8px 14px; }
.kpi b { display:block; font-size:18px; color:var(--cyan); font-variant-numeric:tabular-nums; }
.kpi span { font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:.05em; }
.missing { color:var(--amber); border:1px dashed var(--amber); border-radius:4px; padding:10px; font-size:12px; margin-top:12px; }
table.top, table.gates { border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; width:100%; }
table.top td { padding:4px 10px 4px 0; vertical-align:top; border-bottom:1px solid var(--line); }
table.top td.k { color:var(--dim); white-space:nowrap; width:1%; text-transform:uppercase; font-size:11px; letter-spacing:.05em; }
table.gates th, table.gates td { padding:3px 14px 3px 0; text-align:right; border-bottom:1px solid var(--line); }
table.gates th:first-child, table.gates td:first-child, table.gates td.l, table.gates th.l { text-align:left; }
.warn { color:var(--red); }
.dim { color:var(--dim); }
code { color:var(--cyan); }
ul { color:var(--dim); font-size:13px; }
"""


def _e(x) -> str:
    return _html.escape(str(x))


def _f(x) -> str:
    return f"{x:,.1f}"


def _pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


def _age(m) -> str:
    if m is None:
        return "n/a"
    if m < -1.0:
        return f"{-m:.0f} min in the future (clock skew between the run's host and this one)"
    if m < 1.0:
        return f"{max(m, 0.0) * 60:.0f} s ago"
    if m < 120:
        return f"{m:.0f} min ago"
    return f"{m / 60:.1f} h ago"


def _img(b64: Optional[str], alt: str) -> str:
    if not b64:
        return f'<div class="missing">plot unavailable: {_e(alt)}</div>'
    return f'<img src="data:image/png;base64,{b64}" alt="{_e(alt)}">'


def _topline(S, stage, live, gpu, ov, noise_bias, run_dir) -> str:
    nit, ll = S["nit"], S["ll"]
    R = []

    def row(k, v):
        R.append(f'<tr><td class="k">{k}</td><td>{v}</td></tr>')

    store = os.path.basename(S["store"])
    row("store", f"<code>{_e(store)}</code> <span class='dim'>in {_e(run_dir)}"
        + (" &middot; reduced extract" if S["extract"] else "") + "</span>")
    last_after = S["saved_after"][-1] if S["saved_after"] else None
    row("last saved row", f"<b>{nit - 1}</b> <span class='dim'>(rows are numbered from 0 "
        f"on this page, as on the full page: {nit} rows stored, iteration attr "
        f"{S.get('it_attr')})</span> &middot; saved after "
        f"<code>{_e(last_after or 'n/a')}</code>"
        + (f" &middot; next leg <code>{_e(stage['next_leg'])}</code>"
           if stage and stage.get("next_leg") else ""))
    if stage is None:
        row("stage", "none (no recipe group in the store)")
    elif stage["name"] is None:
        row("stage", f"<b>FINISHED</b>: all {stage['total']} steps complete")
    else:
        bits = [f"step {stage['order']} of {stage['total']}"]
        if stage["start"] is not None:
            bits.append(f"started row {int(stage['start'])}, {stage['rows_in']} rows in")
        row("stage", f"<b>{_e(stage['name'])}</b> &middot; " + " &middot; ".join(bits)
            + (f" <span class='dim'>(done: {_e(', '.join(stage['done']))})</span>"
               if stage["done"] else ""))
    rat = []
    if stage and stage.get("name"):
        rat.append("store: DONE (stop stamped)" if stage.get("ratchet_done")
                   else "store: no stop stamp")
    if live and live.get("gate"):
        rat.append(f"latest gate <b>{_e(live['gate'])}</b>"
                   + (f" <span class='dim'>({live['gate_ts']:%Y-%m-%d %H:%M})</span>"
                      if live.get("gate_ts") else ""))
    if live and live.get("verdict"):
        rat.append("verdict: " + _e(verdict_phrase(live["verdict"])))
    row("galfor ratchet", " &middot; ".join(rat) if rat else "<span class='dim'>n/a</span>")
    if live is None:
        row("pace", "<span class='dim'>n/a (no head log or job stdout in the run dir)</span>")
    else:
        bits = []
        if live["saves"]:
            bits.append(f"last [SAVE] {live['saves'][-1]:%Y-%m-%d %H:%M:%S} (log clock)")
        if live["cadence"]:
            c = live["cadence"]
            bits.append(f"row cadence: last {c[-1]:.0f} min, median of {len(c)} = "
                        f"{sorted(c)[len(c) // 2]:.0f} min")
        elif live["saves"]:
            bits.append("one [SAVE] in the log tail, no cadence yet")
        else:
            bits.append("no [SAVE] lines in the log tail")
        bits.append(f"log age {_age(live['log_age_min'])}")
        row("pace", " &middot; ".join(bits)
            + f" <span class='dim'>({_e(os.path.basename(live['log']))})</span>")
        if live.get("replica"):
            row("replica pe", _e(live["replica"][:160]))
        if live.get("stage_line"):
            row("stage log", _e(live["stage_line"][:160]))
    if gpu is None:
        row("gpu (head node)", "<span class='dim'>n/a (no gpu_util_*.csv in the run dir)</span>")
    else:
        parts = [f"GPU{i} util {d['util']:.0f}% (60 s {d['util_avg']:.0f}%), mem "
                 f"{d['mem_used_gb']:.1f}/{d['mem_total_gb']:.1f} GiB, "
                 f"{d['power_w']:.0f} W, {d['temp_c']:.0f} C"
                 for i, d in gpu["gpus"].items()]
        stale = gpu["age_min"] is not None and gpu["age_min"] > 5
        row("gpu (head node)", " &middot; ".join(parts)
            + f" <span class='{'warn' if stale else 'dim'}'>sample "
            f"{gpu['t_last']:%Y-%m-%d %H:%M:%S}, {_e(_age(gpu['age_min']))}"
            f"{' -- STALE' if stale else ''}; head node only, a second node is not sampled "
            f"({_e(os.path.basename(gpu['csv']))})</span>")
    L = ll[nit - 1]
    txt = (f"max <b>{_f(L.max())}</b> &middot; mean {_f(L.mean())} &middot; min "
           f"{_f(L.min())} &middot; spread {_f(L.max() - L.min())}")
    if nit > 1:
        P = ll[nit - 2]
        txt += (f"<br><span class='dim'>change vs row {nit - 2}:</span> max "
                f"{L.max() - P.max():+,.1f} &middot; mean {L.mean() - P.mean():+,.1f} "
                f"&middot; min {L.min() - P.min():+,.1f}")
    row(f"cold lnL, row {nit - 1}", txt)
    lv = S["leaves"]
    if lv:
        main = "gb" if "gb" in lv else sorted(lv)[0]
        v = lv[main][nit - 1]
        txt = (f"{main} min <b>{int(v.min()):,}</b> &middot; mean {v.mean():,.1f} "
               f"&middot; max {int(v.max()):,}")
        if nit > 1:
            pv = lv[main][nit - 2]
            txt += f" <span class='dim'>(mean {v.mean() - pv.mean():+.1f} vs row {nit - 2})</span>"
        others = []
        for b in sorted(lv):
            if b == main:
                continue
            w = lv[b][nit - 1]
            others.append(f"{b} {int(w.min())}" if w.min() == w.max()
                          else f"{b} {int(w.min())}-{int(w.max())}")
        if others:
            txt += "<br><span class='dim'>fixed-size branches:</span> " + " &middot; ".join(others)
        row("cold leaves", txt)
    if ov is not None:
        if ov.get("mm") is not None:
            row("overlap match", f"<b>{ov['n_match']:,}</b> of {ov['ndet']:,} detectable "
                f"matched at overlap &ge; {ov['thresh']:.2f} &middot; completeness "
                f"{_pct(ov['completeness'])} &middot; purity {_pct(ov['purity'])} &middot; "
                f"median overlap {ov['mm_med']:.3f} <span class='dim'>(row {ov['row']}, "
                f"max-lnL walker {ov['walker']})</span>")
        else:
            row("overlap match", f"2-bin f0 proxy only: {ov['n_proxy']:,} of "
                f"{ov['ndet']:,} &middot; completeness {_pct(ov['completeness_proxy'])} "
                f"&middot; purity {_pct(ov['purity_proxy'])}")
    if noise_bias:
        row("instrument noise", " &middot; ".join(
            f"{k} median {100 * b:+.2f}% from injected" for k, b in noise_bias.items()))
    return '<table class="top">' + "".join(R) + "</table>"


def _gate_table(gates) -> str:
    if not gates:
        return ""
    show = gates[-GATE_ROWS_SHOWN:]
    has_lv = all("leaves" in g_ for g_ in show)
    head = ("<tr><th>row</th><th class='l'>kind</th><th>cold lnL change at the row</th>"
            "<th>then, through row</th><th>cold lnL gained</th>"
            + ("<th>GB leaves at the row</th><th>leaves gained</th>" if has_lv else "")
            + "</tr>")
    body = []
    for g_ in show:
        col = "var(--amber)" if g_["kind"] == "nudge" else (
            "var(--green)" if g_["kind"] == "moved" else "var(--dim)")
        body.append(
            f"<tr><td>{g_['row']}</td><td class='l' style='color:{col}'>{g_['kind']}</td>"
            f"<td>{g_['dll']:+,.1f}</td><td>{g_['end']}</td><td>{g_['dll_after']:+,.1f}</td>"
            + (f"<td>{g_['leaves']:,.1f}</td><td>{g_['dleaves_after']:+,.1f}</td>"
               if has_lv else "") + "</tr>")
    return '<table class="gates">' + head + "".join(body) + "</table>"


def render_short_page(run_dir: str, *, mojito: Optional[str] = None, now=None) -> str:
    """The short page for ``run_dir`` as one self-contained HTML string."""
    import matplotlib

    from . import check_truth, resolve_mojito_path

    t0 = time.perf_counter()
    phases, mark = {}, [t0]

    def _lap(name):
        """Charge the time since the previous lap to ``name`` (footer + log)."""
        now_ = time.perf_counter()
        phases[name] = phases.get(name, 0.0) + now_ - mark[0]
        mark[0] = now_

    run_dir = os.path.abspath(str(run_dir))
    notes = []
    ns = _generator_functions({"np": np})
    store = find_store(run_dir)
    S = read_store(store, run_dir, ns, notes)
    _lap("store")
    nit, nw = S["nit"], S["nwalk"]
    ns["STAGE_BOUNDS"] = stage_bounds(S["steps"])
    stage = current_stage(S["steps"], nit, S["saved_after"])
    gb_leaves = S["leaves"].get("gb")
    gates = gate_rows(S["saved_after"], S["gal_phys"], S["ll"], gb_leaves)

    live = gpu = None
    try:
        live = read_live(run_dir, now=now)
    except Exception as e:  # noqa: BLE001
        notes.append(f"head log unreadable ({type(e).__name__}: {_e(e)})")
    if live is None:
        notes.append("no head log (*_artifacts/globalfit_run.log or _tail) and no "
                     "slurm_stdout_*.log: no pace or ratchet log lines.")
    try:
        gpu = read_gpu(run_dir, now=now)
    except Exception as e:  # noqa: BLE001
        notes.append(f"gpu_util csv unreadable ({type(e).__name__}: {_e(e)})")
    if gpu is None:
        notes.append("no gpu_util_*.csv in the run dir: no GPU row.")
    _lap("log+gpu")

    moj, moj_src = resolve_mojito_path(run_dir=run_dir, explicit=mojito)
    from lisatools.globalfit.stock.erebor.noise import psd_truth_levels

    soms_inj, sa_inj = psd_truth_levels(mojito_data_path=moj)
    if not moj:
        notes.append(f"no mojito tree ({_e(moj_src)}): the injected Soms_d / Sa_a are "
                     "the round analytic values, not the brick fit.")
    _lap("noise truth")

    show_match = os.environ.get("GF_MONITOR_MATCH_STATS", "1") != "0"
    thresh = float(os.environ.get("GF_MONITOR_MATCH_MM", "0.8"))
    ov = None
    ov_note = ""
    if not show_match:
        ov_note = ("Overlap section SUPPRESSED (GF_MONITOR_MATCH_STATS=0). Nothing is "
                   "wrong with the run; unset the knob to include it.")
    elif S["tobs"] is None:
        ov_note = "No Tobs in the store, so no truth set can be matched."
    else:
        truth, tnote = check_truth(run_dir)
        if truth is None:
            ov_note = (f"No usable truth set: {_e(tnote)}. Build one with "
                       "<code>python -m lisatools.globalfit.monitor --build-truth "
                       "--short-page RUN_DIR</code> (tens of minutes, CPU-only).")
        else:
            try:
                ov = compute_overlap(S, truth, ns, moj, run_settings_t0(run_dir),
                                     notes, thresh)
            except Exception as e:  # noqa: BLE001
                ov_note = f"overlap failed: {type(e).__name__}: {_e(e)}"
            if ov is None and not ov_note:
                ov_note = "No GB coordinates to match at the last row (see Notes)."
    _lap("overlap")

    psd_bias = {}
    imgs = {}
    with matplotlib.rc_context(STYLE):
        imgs["lnl"] = fig_history(ns, S["ll"] - S["ll"][nit - 1].max(), gates, CYAN,
                                  f"cold lnL - {S['ll'][nit - 1].max():,.1f}",
                                  "cold log-likelihood")
        if gb_leaves is not None:
            imgs["leaves"] = fig_history(ns, gb_leaves, gates, GREEN, "cold GB leaves",
                                         "GB leaf count")
        if ov is not None and ov.get("mm") is not None and ov["mm"].size:
            imgs["overlap"] = fig_overlap(ns, ov)
        _lap("history+overlap figures")
        if S["psd"] is not None and S["gal_phys"] is not None and S["psd"].shape[0]:
            try:
                imgs["psd_now"], imgs["psd_params"], imgs["noise_traces"] = noise_figures(
                    S, soms_inj, sa_inj, gates, ns)
            except Exception as e:  # noqa: BLE001
                notes.append(f"noise panels failed: {type(e).__name__}: {_e(e)}")
            _lap("noise figures")
            nsh = min(3, S["psd"].shape[0])
            for j, (k, injv) in enumerate((("Soms_d", soms_inj), ("Sa_a", sa_inj))):
                psd_bias[k] = float(np.median(S["psd"][-nsh:, :, j]) / injv - 1.0)
        else:
            notes.append("no psd/galfor chains in the store: no noise panels.")

    chips = ""
    for s in S["steps"]:
        cls = "done" if s["status"] else ("now" if stage and s["name"] == stage["name"] else "")
        chips += (f'<span class="chip {cls}">{s["order"]}. {_e(s["name"])}'
                  f'{" &#10003;" if s["status"] else ""}</span> ')
    label = os.path.basename(run_dir)
    tobs_txt = f"{S['tobs'] / 86400.0:.0f}-day" if S["tobs"] else "unknown-Tobs"

    if ov is not None and ov.get("mm") is not None:
        kpi = (f'<div class="kpi"><div><b>{ov["n_match"]:,}</b><span>matched, overlap '
               f'&ge; {ov["thresh"]:.2f}</span></div>'
               f'<div><b>{_pct(ov["completeness"])}</b><span>completeness of '
               f'{ov["ndet"]:,}</span></div>'
               f'<div><b>{_pct(ov["purity"])}</b><span>purity of {ov["n_band"]:,} '
               f'in band</span></div>'
               f'<div><b>{ov["mm_med"]:.3f}</b><span>median overlap</span></div>'
               f'<div><b>{_pct(ov["mm_hi"])}</b><span>pairs above 0.9</span></div></div>')
        ov_html = (kpi + '<div class="panel">'
                   f'{_img(imgs.get("overlap"), "phase-maximised overlap")}'
                   f'<div class="caption">Every recovered source of the max-lnL cold walker '
                   f'(walker {ov["walker"]}) at row {ov["row"]} in '
                   f'{ov["band"][0] * 1e3:.4g}&ndash;{ov["band"][1] * 1e3:.4g} mHz is paired '
                   f'one-to-one with a detectable injection within {TOL_BINS:.0f} FD bins '
                   f'({ov["n_proxy"]:,} pairs; the windows cover {_pct(ov["chance"])} of the '
                   f'band, the chance rate), then scored by the noise-weighted overlap over A '
                   f'and E maximised over an overall phase, under the walker-median noise of '
                   f'that row, waveforms on {_e(ov.get("orbits", "?"))}. Left: how many pairs '
                   f'reach each overlap (the red line is the threshold); right: where the poor '
                   f'ones sit in frequency. Same criterion and numbers as the full page&rsquo;s '
                   f'headline.</div></div>')
    elif ov is not None:
        ov_html = (f'<div class="missing">Waveform overlaps unavailable (see Notes). '
                   f'2-bin f0 proxy: {ov["n_proxy"]:,} of {ov["ndet"]:,} detectable paired, '
                   f'completeness {_pct(ov["completeness_proxy"])}, purity '
                   f'{_pct(ov["purity_proxy"])}.</div>')
    else:
        ov_html = f'<div class="missing">{ov_note}</div>'

    nudges = [g_["row"] for g_ in gates if g_["kind"] == "nudge"]
    gate_txt = (f"Dashed verticals mark the ratchet gate rows (amber: nudge rows "
                f"{nudges}; green: rows where the noise moved)." if gates else "")
    ratchet_html = (
        f'<div class="panel">{_gate_table(gates)}<div class="caption">Rows saved right '
        'after the gated noise step (<code>saved_after = noise_ratchet_search</code>), '
        'classified off the stored galfor chain: <b>nudge</b> = the amplitude stepped '
        'with alpha and f_1 held (the forced step), <b>moved</b> = the noise was sampled '
        '(a release, or noise PE after one), <b>held</b> = nothing moved. The next '
        'columns are what the search legs did from that row to the row before the next '
        f'gate: cold lnL and GB leaves, walker means. The last {GATE_ROWS_SHOWN} gate rows '
        'are listed.</div></div>' if gates else
        '<div class="missing">No row in this store was saved after the gated noise '
        'step (ratchet or search legs off, or not reached yet).</div>')
    if live and (live.get("gate") or live.get("verdict")):
        ratchet_html += (f'<div class="caption">Head log: {_e(live.get("gate") or "no gate line")}'
                         + (f' &middot; {_e(verdict_phrase(live.get("verdict")))}'
                            if live.get("verdict") else "") + "</div>")

    elapsed = time.perf_counter() - t0
    phase_txt = ", ".join(f"{k} {v:.1f}" for k, v in phases.items())
    logger.info("short page phases (s): %s", phase_txt)
    notes_html = "".join(f"<li>{n}</li>" for n in notes) or "<li>none</li>"
    built = (now or datetime.now())
    full_page = os.path.basename(run_dir) + "_monitor.html"
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GF short &middot; {_e(label)} &middot; row {nit - 1}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>LISA Global Fit &middot; short status</h1>
  <span class="stamp">{_e(label)} &middot; {tobs_txt} &middot; row {nit - 1} &middot; built {built:%Y-%m-%d %H:%M}</span>
  <span>{chips}</span>
</header>
<nav>
  <a href="#topline">topline</a><a href="#lnl">likelihood</a><a href="#leaves">leaves</a>
  <a href="#overlap">overlap</a><a href="#noise">noise</a><a href="#ratchet">ratchet gates</a>
  <a href="#notes">notes</a>
</nav>
<main>
<section id="topline"><h2>Topline</h2>
{_topline(S, stage, live, gpu, ov, psd_bias, run_dir)}
</section>

<section id="lnl"><h2>Log-likelihood</h2>
<div class="panel">{_img(imgs.get("lnl"), "cold log-likelihood")}
<div class="caption">Cold-chain total log-likelihood per stored row, relative to the best
walker at the last row: the band spans min to max over the {nw} walkers, the white line is
the mean. Left the whole run, right the recent window. Grey dashed verticals are recipe stage
boundaries (completed and start iterations from the store's recipe group). {gate_txt}</div></div>
</section>

<section id="leaves"><h2>GB Leaf Count</h2>
<div class="panel">{_img(imgs.get("leaves"), "GB leaf count")}
<div class="caption">Galactic-binary leaves per cold walker per stored row, min / mean /
max over walkers, same layout and marks. The fixed-size branches are listed in the
topline.</div></div>
</section>

<section id="overlap"><h2>Phase-Maximised Overlap</h2>
{ov_html}
</section>

<section id="noise"><h2>Noise: Current Measurement</h2>
<div class="panel">{_img(imgs.get("psd_now"), "current sensitivity per walker")}
<div class="caption">The fitted noise at the last stored row as the sky-averaged
sensitivity: every cold walker's instrument + galactic foreground (amber) against the
injected instrument (red dotted) and the injected instrument + the FittedHT foreground
estimate (red dash-dot); right, each walker divided by that reference. The run multiplies
the foreground by a tabulated time modulation these static curves do not carry.</div></div>
<div class="panel">{_img(imgs.get("psd_params"), "instrument-noise posteriors")}
<div class="caption">The two instrument-noise parameters, the only noise parameters with a
truth to compare against.</div></div>
<div class="panel">{_img(imgs.get("noise_traces"), "noise parameters vs row")}
<div class="caption">All seven noise parameters per cold walker against stored row (galfor
in its sampling basis), and the fitted foreground at 3 mHz. Dotted red: injected. Gate rows
marked as above.</div></div>
</section>

<section id="ratchet"><h2>Ratchet Gates</h2>
{ratchet_html}
</section>

<section id="notes"><h2>Notes</h2>
<ul>{notes_html}</ul>
<p class="caption">This is the SHORT page. The residual spectrum, recovered population,
search gates, F-statistic, verification binaries and detectability table are on the full
page, <code>{_e(full_page)}</code> beside the run folder
(<code>python -m lisatools.globalfit.monitor RUN_DIR</code>). Built in {elapsed:.1f} s
({_e(phase_txt)}) from <code>{_e(os.path.basename(store))}</code>.</p>
</section>
</main>
</body>
</html>
"""
    return page


def build_short_monitor(run_dir: str, out_path: Optional[str] = None, *,
                        mojito: Optional[str] = None, check: bool = True,
                        now=None) -> Optional[str]:
    """Write the short page; returns its path, or ``None`` when ``check=False``
    and the build failed.

    Published atomically (temp beside the target, then rename), so a browser
    refreshing the page never catches half a document, and a failed build
    leaves any previous page in place.
    """
    run_dir = os.path.abspath(str(run_dir))
    out_path = out_path or default_short_out_path(run_dir)
    tmp = out_path + ".tmp"
    st = time.perf_counter()
    try:
        page = render_short_page(run_dir, mojito=mojito, now=now)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(page)
        os.replace(tmp, out_path)
    except Exception as e:  # noqa: BLE001
        import traceback

        logger.warning("short monitor page NOT refreshed (%s: %s). Any previous page "
                       "is untouched.\n%s", type(e).__name__, e,
                       traceback.format_exc().strip()[-4000:])
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        if check:
            raise
        return None
    logger.info("short monitor page built in %.1f s -> %s",
                time.perf_counter() - st, out_path)
    return out_path
