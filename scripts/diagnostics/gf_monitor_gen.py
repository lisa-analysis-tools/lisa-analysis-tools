#!/usr/bin/env python
"""GF run status page: <run_dir> -> one self-contained HTML file.

Reusable: point RUN_DIR at any unzipped gf_prod_* snapshot and rerun; the
artifact redeploys to the same URL. Sections degrade to labeled
placeholders when a snapshot lacks their inputs.

The page is a STATUS OBJECT for collaborators, not an engineering worklog.
Its spine is one frozen denominator -- the catalogue galactic binaries
detectable (optimal SNR > 7) over the analysed GB band under this run's own
fitted noise -- against which completeness, purity and per-source recovery are
all quoted. The band is not hardcoded here: it is read from the ``band`` field
the truth npz stamps (``build_truth.py``), so widening the truth set to the
full 0.5556-21.94 mHz GB band re-labels every caption automatically. Run-mechanics forensics live in the collapsed appendix, never in the
body.

Optional inputs, read from the run directory or the working directory:
  gb_truth_3to21.npz  the frozen detectability set + full catalogue
                      parameters (built once by build_truth.py)
  kappa_grid.npz      SNR/amplitude ceiling per frequency, for the
                      sensitivity curve on the population panels
  gf_arm_<tag>.npz    written by THIS script on every run; holds one arm's
                      per-iteration recovery series so the v2/v3 comparison
                      panels can draw both arms.
Without them the recovery section degrades to a note; every other section
still builds.

COLOUR CONVENTION, one meaning per hue (the previous page used red for five
different things):
  cyan   injected data / the injected catalogue / arm v2
  amber  noise + foreground / arm v3
  green  recovered AND matched to an injection
  violet recovered with NO matching injection
  red    detectable but NOT recovered  (and nothing else)
  white  reference lines: noise model, y = x, sensitivity
"""
import base64, glob, io, json, os, re, sys
from datetime import datetime

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def discover_run_logs(run_dir):
    """Every rank's run log under ``run_dir``: the head's ``globalfit_run.log``
    first, then ``globalfit_run.rank<k>.log`` files sorted by rank number.

    Each rank writes its own log file under the walker-block layout
    (``run.py::_rank_log_filenames``); concatenating them (head first) is
    what lets the regex scans below (e.g. ``RJ_SPLIT_RE``) see every rank's
    ``[GB_ACCEPT rj-split]`` lines, not just the head's (Plan 5 Task 4).

    The walk is RECURSIVE and deterministic (directories and file names
    sorted), identical to ``gf_run_log_digest.py``'s copy of this helper. A
    second file for a rank already seen (the same run unpacked twice under
    ``run_dir``) is NOT silently dropped -- the first one found wins and the
    duplicate is named on stderr.
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


if __name__ != "__main__":
    # Standalone report generator, not a library; guard the rest of the file
    # (which reads sys.argv / real snapshot files unconditionally) so tests
    # can import discover_run_logs() above without running it.
    raise SystemExit(0)

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "prod3mo/gf_prod_3mo"
OUT = sys.argv[2] if len(sys.argv) > 2 else "gf_monitor.html"

# ---- mission-control plot style -------------------------------------------
BG, PANEL, LINE, FG, DIM = "#0A0E14", "#10161F", "#223041", "#B8C6D4", "#67788A"
CYAN, AMBER, GREEN, RED, VIOLET = "#4FD8EB", "#F5A623", "#58C48A", "#E5484D", "#9B7BFF"
# text.usetex is pinned OFF rather than left to inherit. Nothing on this
# page needs a real LaTeX installation -- the ~19 math labels
# (r"$\Delta f_0$", r"$\ln(A_{\rm rec}/A_{\rm cat})$", "m/s$^2$", ...) are
# all plain mathtext, which matplotlib renders internally. Left unset the
# value comes from whatever matplotlibrc the machine happens to carry, so a
# host with usetex on would shell out to LaTeX for every one of the 30
# figures: far slower, and a hard failure on any box without a TeX
# distribution -- which is most cluster nodes and every fresh container.
# Set GF_MONITOR_USETEX=1 to opt back in (real LaTeX must be installed).
_USETEX = os.environ.get("GF_MONITOR_USETEX", "0") not in ("0", "", "false", "no")
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})

IMGS, MISSING = {}, []

# MATCH-CRITERION CONTENT GATE (user ruling 2026-08-19). The page's
# completeness / purity / matched-pair numbers all come from the 2-bin f0
# PROXY match, not the real phase-maximised overlap statistic (too heavy to
# compute at page-build time). Ruling: catalogue TRUTHS stay on every visual
# overlay, but nothing derived from the page's own match criterion is shown
# -- no completeness/purity, no matched counts, no matched-pair deltas, no
# recovery split/census. GF_MONITOR_MATCH_STATS=1 restores those panels.
SHOW_MATCH_STATS = os.environ.get("GF_MONITOR_MATCH_STATS", "0") == "1"

# Threshold for "matched" once the phase-maximised, noise-weighted overlap MM
# is computed for each 2-df pair (see below). Default 0.8; override via
# GF_MONITOR_MATCH_MM. Only affects F4's three-way split and the zoomable-plot
# marker classes -- the F2/F3/F6 panels intentionally still key off the 2-df
# proxy so per-iteration progress does not require re-running waveforms per
# stored row.
MATCH_MM_THRESH = float(os.environ.get("GF_MONITOR_MATCH_MM", "0.8"))

def fig_b64(fig, key, dpi=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    IMGS[key] = base64.b64encode(buf.getvalue()).decode()

def img(key, alt=""):
    if key not in IMGS:
        return f'<div class="missing">plot unavailable in this snapshot: {alt or key}</div>'
    return f'<img src="data:image/png;base64,{IMGS[key]}" alt="{alt or key}">'

# ============================ LOAD ==========================================
# PICK THE LIVE STORE DETERMINISTICALLY (2026-08-16). A recovered run dir
# holds THREE .h5 files -- the live store, ``*_running_backup_copy.h5`` and
# one or more ``*_CORRUPT*.h5`` kept for forensics -- and the old
# "last name os.listdir happens to yield" pick could land on any of them.
# os.listdir order is filesystem-dependent, so the page could silently
# render the damaged store (missing iterations) or the lagging backup on one
# machine and the right file on another, with nothing in the output saying
# which. Name the live store explicitly and keep the others as fallbacks.
# NEWEST mtime wins, not alphabetical (2026-08-22): a make_snapshots.sh
# tar refreshes the *_extract.h5 but cannot delete a previous format's
# full store left in the dir, and "testing.h5" sorts before
# "testing_extract.h5" -- the first tar-format page silently rendered a
# 384-iteration-stale store. Freshness is the only defensible tiebreak.
_h5s = sorted((fn for fn in os.listdir(RUN_DIR) if fn.endswith(".h5")),
              key=lambda fn: os.path.getmtime(os.path.join(RUN_DIR, fn)),
              reverse=True)
_live = [fn for fn in _h5s
         if not fn.endswith("_running_backup_copy.h5") and "_CORRUPT" not in fn]
if not _live:                                   # only a backup survived
    _live = [fn for fn in _h5s if fn.endswith("_running_backup_copy.h5")]
    if _live:
        MISSING.append(
            "no live store in this snapshot -- rendering the running backup "
            "copy, which lags the run by up to one save step.")
h5path = os.path.join(RUN_DIR, (_live or _h5s)[0])
f = h5py.File(h5path, "r")
g = f["global_fit"]
ll_all = g["log_like"][:, 0, 0, :]
filled = np.where(np.any(ll_all != 0.0, axis=1))[0]
NIT = int(filled.max()) + 1 if filled.size else 0
# REWIND-AWARE (2026-08-19): the live row count is the store's ``iteration``
# attr, NOT the filled-row extent. reset_recipe_stage rewinds by moving that
# one attr back; the rows beyond it keep their old (dead) contents until the
# run's next grow() truncates them. Rendering by filled rows alone mixed the
# discarded pre-rewind trajectory into every panel of a freshly rewound
# store (leaves climbing to the old values, x-axes past the live range).
_it_attr = g.attrs.get("iteration")
REWOUND = _it_attr is not None and 0 < int(_it_attr) < NIT
if REWOUND:
    MISSING.append(
        f"store rewound: iteration attr {int(_it_attr)} < filled rows {NIT}; "
        f"rendering the live {int(_it_attr)} rows only (the rest is the "
        "discarded pre-rewind trajectory awaiting truncation).")
    NIT = int(_it_attr)
it = np.arange(NIT)
ll = ll_all[:NIT]                                   # (it, 24)
recipe = {}
rg = f.get("global_fit/recipe", f.get("recipe"))
if rg is not None:
    for k in rg:
        recipe[k] = (int(rg[k].attrs.get("order num", 0)), bool(rg[k].attrs.get("status", False)))
nwalk = ll.shape[1]

# RUN IDENTITY (2026-08-15): one generator now serves several runs (3-mo
# production, 23-mo scaling). Derive the label from the store rather than
# hard-coding it, so a 23-mo page can never be mislabelled as the 3-mo one.
_base = os.path.basename(os.path.normpath(RUN_DIR))
if "23mo" in _base:
    RUN_LABEL, RUN_KIND = "23-Month", "23mo"
elif "6mo" in _base:
    RUN_LABEL, RUN_KIND = "6-Month", "6mo"
elif _base.endswith("_v4") or "3mo_v4" in _base:
    RUN_LABEL, RUN_KIND = "3-Month v4", "3mo_v4"
elif _base.endswith("_v3") or "3mo_v3" in _base:
    # The v3 A/B carries the same Tobs as v2, so the label has to come from
    # the VARIANT or the two pages are indistinguishable in a browser tab --
    # which is the whole point of running them side by side.
    RUN_LABEL, RUN_KIND = "3-Month v3", "3mo_v3"
elif "1yr_v8" in _base or (_base.endswith("_v8") and "1yr" in _base):
    # 2026-09-03: the v8 lineage 1-yr run (submit_gf_1yr_v8.sh). Its own
    # arm cache + banner so it never collides with the v5 1-yr page.
    RUN_LABEL, RUN_KIND = "1-Year v8", "1yr_v8"
elif "1yr" in _base:
    # 2026-08-22: without this branch the 1-yr page fell through to the
    # 3-Month banner AND (worse) the v2 ARM_TAG, clobbering the shared
    # gf_arm_v2.npz cache with 1-yr data.
    RUN_LABEL, RUN_KIND = "1-Year v5", "1yr_v5"
elif _base.endswith("_v5") or "3mo_v5" in _base:
    RUN_LABEL, RUN_KIND = "3-Month v5", "3mo_v5"
elif _base.endswith("_v6") or "3mo_v6" in _base:
    RUN_LABEL, RUN_KIND = "3-Month v6", "3mo_v6"
elif _base.endswith("_v7") or "3mo_v7" in _base:
    # 2026-08-26: same trap as the 1-yr branch above -- without this the
    # v7 page fell through to the plain 3-Month banner AND the v2
    # ARM_TAG, clobbering gf_arm_v2.npz on its first generation.
    RUN_LABEL, RUN_KIND = "3-Month v7", "3mo_v7"
elif "10walker" in _base and ("3mo_v8" in _base or _base.endswith("_v8")):
    # 2026-09-05: the v8 3-month 10-WALKER twin (gf_prod_3mo_v8_10walkers).
    # Own arm tag + banner so its overlay curve reads "3mo_v8_10w", distinct
    # from the 24-walker "3mo_v8_24w" arm -- both otherwise collide on the
    # plain "3mo_v8" tag and clobber each other's gf_arm_3mo_v8.npz cache.
    RUN_LABEL, RUN_KIND = "3-Month v8 · 10 Walkers", "3mo_v8_10w"
elif _base.endswith("_v8") or "3mo_v8" in _base:
    # 2026-09-02: hit for the THIRD time (1yr, v7, now v8) -- the fall-through
    # clobbers gf_arm_v2.npz. If a v9 ever exists, generalize this ladder.
    RUN_LABEL, RUN_KIND = "3-Month v8", "3mo_v8"
else:
    RUN_LABEL, RUN_KIND = "3-Month", "3mo"

# ---- persistent commentary sidecar (2026-08-27) --------------------------
# Hand-written notes must survive page regeneration (the page is rebuilt
# from scratch every snapshot): drop HTML into commentary_{RUN_KIND}.html
# NEXT TO the output file and it renders as a "Commentary" section at the
# top of <main>. Absent file -> no section, no cost.
_comm_path = os.path.join(
    os.path.dirname(os.path.abspath(OUT)) or ".",
    f"commentary_{RUN_KIND}.html")
if os.path.exists(_comm_path):
    with open(_comm_path, "r") as _cf:
        COMMENTARY = (
            '<section id="commentary"><h2>Commentary</h2>\n'
            + _cf.read() + "\n</section>\n")
    print(f"[commentary] injected {_comm_path}")
else:
    COMMENTARY = ""

sub = g["sub_backend"]
psd_c = sub["psd/chain"][:NIT]                      # (it, 12, 24, 1, 2)
gal_c = sub["galfor/chain"][:NIT]                   # (it, 12, 24, 1, 5)
# Two VGB sampling bases are live in production (stock/erebor/vgb.py):
#   * legacy 5-param DIST basis: [dist, phi0, cos_iota, psi, fdot_astro_ratio]
#     with per-leaf FIXED Mc from the catalogue
#   * 6-param CHIRP basis (VGB_CHIRP_MASS_BASIS=1, used by 6-month runs):
#     [dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio]  -- Mc sampled
# The panel that reassembles the 9-col GB basis (F1 residual, DTR panels)
# below MUST know which basis this store used or it will read a chirp mass
# out of the fdot_astro_ratio slot and vice versa, collapsing every VGB
# template's amplitude to ~zero (silent -- no crash, the residual just
# reads as if VGB isn't being subtracted). Store's last chain dim tells
# them apart. Keep the FULL sampled row; downstream reads by column index.
_vgb_chain_ndim = int(sub["vgb/chain"].shape[-1])
VGB_SAMPLED_DIST = _vgb_chain_ndim == 5
VGB_SAMPLED_CHIRP = _vgb_chain_ndim == 6
if not (VGB_SAMPLED_DIST or VGB_SAMPLED_CHIRP):
    raise RuntimeError(
        f"unrecognised VGB chain last-dim {_vgb_chain_ndim} "
        f"(expected 5 for DIST basis or 6 for CHIRP basis)")
vgb_c = sub["vgb/chain"][:NIT, 0]                   # (it, W, 55, 5 or 6)
vgb_hh = sub["vgb/h_h"][:NIT]                       # (it, W, 55)
# TRAILING INCOMPLETE SUB-BACKEND ROWS (2026-08-15). The main backend and a
# sub-backend are not flushed atomically: a snapshot can hold a row where
# log_like / inds / chain are written but sub_backend/vgb/* is still all
# zeros. Taken at face value that row makes EVERY VGB look like it has zero
# amplitude -- it is what made HM Cnc, the loudest VGB in the catalogue,
# render with SNR 0. Trim the VGB arrays to their last row that carries
# actual signal; GB panels keep the full NIT because their datasets are
# complete. (Leading rows are legitimately NaN -- the VGB branch only starts
# sampling at stage 2 -- so test for "has any nonzero finite value".)
def _last_written(a):
    for i in range(a.shape[0] - 1, -1, -1):
        v = a[i]
        if np.isfinite(v).any() and np.abs(np.nan_to_num(v)).sum() > 0:
            return i + 1
    return 0

psd_sw_a = sub["psd/swaps_accepted"][:NIT]; psd_sw_p = sub["psd/swaps_proposed"][:NIT]
gal_sw_a = sub["galfor/swaps_accepted"][:NIT]; gal_sw_p = sub["galfor/swaps_proposed"][:NIT]
# EVERY sub-backend shares the flush, so one trailing row can be missing
# from all of them at once -- it is not a VGB quirk. Trim each branch to its
# own last written row (they can differ) and report it.
SUB_NIT = min(_last_written(vgb_hh), _last_written(vgb_c),
              _last_written(psd_c), _last_written(gal_c))
if SUB_NIT < NIT:
    MISSING.append(
        f"sub-backends (psd / galfor / vgb) written through iteration "
        f"{SUB_NIT - 1} while the main backend reached {NIT - 1} (snapshot "
        f"caught mid-flush); those panels use the last COMPLETE row. Taken "
        f"raw, the unwritten row reads as zero noise parameters and zero VGB "
        f"amplitudes.")
    vgb_c = vgb_c[:SUB_NIT]
    vgb_hh = vgb_hh[:SUB_NIT]
    psd_c = psd_c[:SUB_NIT]
    gal_c = gal_c[:SUB_NIT]
    psd_sw_a = psd_sw_a[:SUB_NIT]; psd_sw_p = psd_sw_p[:SUB_NIT]
    gal_sw_a = gal_sw_a[:SUB_NIT]; gal_sw_p = gal_sw_p[:SUB_NIT]
VGB_NIT = SUB_NIT
gb_inds = g["inds/gb"][:NIT, 0, 0]                  # (it, 24, 10000)
gb_chain_cold = g["chain/gb"][NIT-1, 0, 0]          # (24, 10000, 9) last iter
gb_alive_last = g["inds/gb"][NIT-1, 0, 0]           # (24, 10000)

# ---- POOLED SAMPLE WINDOWS (2026-08-27, user request) --------------------
# Every panel that SCATTERS per-leaf cold-chain samples used to draw ONE
# stored iteration -- the latest. A single iteration of a trans-dimensional
# branch is not a posterior, it is a snapshot of where the walkers happen to
# sit at one step: a source alive in 23 of the 24 cold walkers draws 23
# points, and their spread is walker scatter, not sampler uncertainty.
# Pooling the last few stored iterations makes each panel the thing the
# reader already believes they are looking at.
#
# Two windows, deliberately different:
#   * POOL_ITS_SAMPLES    -- scatter / explorer / marginal panels.
#   * POOL_ITS_POSTERIOR  -- corner plots, which need the extra rows before
#     their 2-D contours mean anything (this is the window the VGB and GB
#     corners already used; it is routed through the helper now so every
#     pooling site in the file reads from ONE place).
#
# THREE RULES, none of them optional:
#   1. min(requested, available) -- a store with three written rows pools
#      three, and nothing anywhere may assume the full window exists.
#   2. The alive mask is applied PER ITERATION. ``inds`` at iteration N says
#      nothing about N-4: the branch is trans-dimensional, a leaf alive now
#      may be dead then, and a dead row's coordinates are stale-or-zero
#      garbage that draws as a phantom source (f0 = 0 with the zero fill).
#   3. A reduced ``*_extract.h5`` store carries real coordinates ONLY inside
#      its final keep-window -- earlier rows are zero-filled placeholders
#      that survive as valid-looking arrays. Clamp the window to rows whose
#      coordinates are actually populated. (The full production store this
#      was validated on is NOT an extract, so the clamp is a no-op there;
#      it exists because the same generator is pointed at extracts.)
POOL_ITS_SAMPLES = 30
POOL_ITS_POSTERIOR = 30
EXTRACT_STORE = "_extract" in os.path.basename(h5path)
_POOL_FLOOR = {}


def _pool_floor(branch="gb"):
    """Earliest stored row of ``branch`` whose coordinates are real.

    0 for a full store (every written row carries its coordinates). For an
    extract, walk backwards from the newest row and stop at the first one
    whose alive leaves do NOT carry a non-zero f0 -- that is the edge of the
    keep-window. Probes ONE walker (the one holding the most alive leaves)
    and ONE parameter column, so the cost is a couple of chunks per row.
    """
    if branch in _POOL_FLOOR:
        return _POOL_FLOOR[branch]
    floor = 0
    if EXTRACT_STORE:
        floor = max(NIT - 1, 0)
        for _i in range(NIT - 1, -1, -1):
            try:
                _al = g[f"inds/{branch}"][_i, 0, 0]
                if not _al.any():
                    break
                _w = int(np.argmax(_al.sum(axis=1)))
                _f0 = g[f"chain/{branch}"][_i, 0, 0, _w, :, 1][_al[_w]]
            except Exception:
                break
            if not np.any(_f0 != 0.0):
                break
            floor = _i
        if floor > 0:
            MISSING.append(
                f"reduced *_extract.h5 store: {branch} coordinates are real "
                f"only from stored iteration {floor} onwards, so the pooled "
                f"sample panels use that keep-window, not the full "
                f"{POOL_ITS_POSTERIOR}-iteration request.")
    _POOL_FLOOR[branch] = floor
    return floor


def _pool_its(nwant, branch="gb", nit=None):
    """Stored-iteration indices a pooled panel should read, oldest first.

    ``nwant`` is the REQUEST (5 or 10); what comes back is
    ``min(nwant, available)`` indices, floored at the extract keep-window.
    """
    n = int(NIT if nit is None else nit)
    if n <= 0:
        return np.zeros(0, dtype=int)
    return np.arange(max(_pool_floor(branch), n - int(nwant)), n)


def _pool_gb_iter(nwant, cols=None):
    """Yield ``(it, alive, chain)`` for each pooled GB iteration, oldest first.

    ``alive`` is THAT iteration's own ``inds`` row -- rule 2 above. ``chain``
    is ``(nwalk, nleaf, len(cols))``, sliced column-wise because the chain is
    chunked ``(..., 1)`` on the parameter axis: pulling three columns
    decompresses a third of the bytes a full-row read would.
    """
    for _i in _pool_its(nwant, "gb"):
        _al = gb_inds[_i]
        if not _al.any():
            continue
        if cols is None:
            _ch = g["chain/gb"][_i, 0, 0]
        else:
            _ch = np.stack([g["chain/gb"][_i, 0, 0, :, :, _c] for _c in cols],
                           axis=-1)
        yield int(_i), _al, _ch
        del _ch


def _vgb_pool_rows(nwant):
    """Row indices of ``vgb_c`` a pooled VGB panel should use, oldest first.

    Same three rules. The VGB branch is fixed-dimension, so there is no
    alive mask to apply per iteration -- what there IS is a leading block of
    rows the branch had not started sampling yet (it comes online at stage
    2) and, on an extract, a zero-filled pre-keep-window block. Both read as
    "all zero / all NaN" and both would drag every posterior to 0 kpc, so
    unpopulated rows are dropped from the FRONT of the window.
    """
    n = int(VGB_NIT)
    if n <= 0:
        return np.zeros(0, dtype=int)
    lo = max(0, n - int(nwant))
    while lo < n - 1:
        _r = vgb_c[lo]
        if np.isfinite(_r).any() and np.abs(np.nan_to_num(_r)).sum() > 0:
            break
        lo += 1
    return np.arange(lo, n)


# TORN-SNAPSHOT TOLERANCE (2026-08-15): a store copied while the run is
# mid-[SAVE] can carry truncated gzip chunks -- the dataset OPENS fine and
# only fails when READ ("filter returned failure during read"). That is a
# snapshot artifact, not a run fault, and it must not cost the whole page:
# on the v2 zip the tear was confined to sub_backend/gb/* while chain,
# inds and log_like -- i.e. every science panel -- were perfectly readable.
# Degrade per-dataset instead of dying.
_BACKUP_G = None  # lazily-opened *_running_backup_copy.h5 root group


def _backup_group():
    """The run's own backup copy, opened on demand.

    The engine keeps ``*_running_backup_copy.h5`` alongside the live store
    precisely so a torn live copy is recoverable. It lags the main file (it
    is written between saves), so it is a FALLBACK, never the default.
    """
    global _BACKUP_G
    if _BACKUP_G is None:
        _BACKUP_G = False
        for fn in os.listdir(RUN_DIR):
            if fn.endswith("_running_backup_copy.h5"):
                try:
                    _BACKUP_G = h5py.File(
                        os.path.join(RUN_DIR, fn), "r")["global_fit"]
                except Exception:
                    _BACKUP_G = False
                break
    return _BACKUP_G or None


def _safe(node, key, default=None, label=None):
    try:
        return node[key][:NIT] if NIT else node[key][()]
    except Exception as e:
        # Torn in the live copy -> try the run's backup before giving up.
        bg = _backup_group()
        if bg is not None:
            try:
                sub_key = key if node is g else "sub_backend/" + key
                # The backup is PREALLOCATED to the full run length, so slice
                # to the iterations it has actually written -- otherwise a
                # 5-row store returns 2000 rows of zeros and every plot built
                # on it is mostly empty padding.
                _bn = int(bg.attrs.get("iteration", 0))
                arr = bg[sub_key][:_bn] if _bn else bg[sub_key][()]
                MISSING.append(
                    f"{label or key}: torn in the live store, recovered from "
                    f"the run's backup copy, which holds {arr.shape[0]} "
                    f"iteration(s) vs {NIT} in the main file.")
                return arr
            except Exception:
                pass
        MISSING.append(
            f"{label or key}: unreadable in this snapshot "
            f"(likely copied mid-save) -- {type(e).__name__}")
        return default

# THE CAP PANEL MUST PLOT THE ENFORCED ARRAY (2026-08-16). This used to
# read ``gb/band_leaf_cap`` -- the LEGACY MIRROR that
# ``_mirror_band_leaf_cap`` keeps equal to the MAX over each band's cap
# cells -- while the births are actually gated per CAP CELL
# (``gbspecialstretch._run_rj_step``: "THE EXACT PER-CELL ENFORCEMENT
# POINT ... it is per cell"). On this run the mirror overstates the real
# cap for 515 of 1,232 cells (42%), by up to 12, because 133 of 154 bands
# carry a non-zero spread across their own cells. The old label
# "leaf cap per band" was therefore TRUE of the array being drawn and
# false about the run -- fixing only the string would have made the plot
# lie. Prefer the cell array; fall back to the band array only where the
# cell array does not exist (``cap_divisor == 1`` stores never allocate
# it), and label from WHICH array was used rather than guessing from the
# column count.
band_edges = sub["gb/band_edges"][:]
try:
    cap_edges_static = sub["gb/cap_edges"][:]
except Exception:
    cap_edges_static = band_edges
CAP_K = max(int(round((cap_edges_static.size - 1) / max(band_edges.size - 1, 1))), 1)
caps = _safe(sub, "gb/cap_cell_leaf_cap", None, "per-cell leaf caps")
CAP_UNIT = "cap cell"
if caps is None or not getattr(caps, "size", 0):
    caps = _safe(sub, "gb/band_leaf_cap", None, "per-band leaf caps")
    CAP_UNIT, CAP_K = "band", 1


logpaths = discover_run_logs(RUN_DIR)
log_text = "".join(open(p, errors="replace").read() for p in logpaths)
# REWIND-AWARE (2026-08-19): the run log is CUMULATIVE across launches. On a
# rewound store the segments before the final relaunch describe the DISCARDED
# trajectory, and every log-parsed panel (band shutoffs, RJ split, acceptance,
# timing) would mix it into the live run's series. Cut at the last resume
# marker; non-rewound stores keep the full log (normal resumes continue the
# same trajectory, so their history is valid).
if REWOUND and log_text:
    # Per-launch marker actually present in THIS file: the results rank
    # prints it once at every process start ("RESUMING from existing
    # backend" goes to the slurm stdout, not here).
    _pos = log_text.rfind("starting async save/plot loop")
    if _pos > 0:
        _pos = log_text.rfind("\n", 0, _pos) + 1
        MISSING.append(
            "log-parsed panels cut to the final launch segment (pre-rewind "
            f"log history discarded, {_pos/1e6:.1f} MB skipped).")
        log_text = log_text[_pos:]

# ---- the artifacts directory, found rather than guessed --------------------
# This used to be built as ``basename(RUN_DIR) + "_artifacts"``, which is only
# right when the run directory happens to be named after the store. It is not:
# both production arms live in ``gf_prod_3mo_v2`` / ``gf_prod_3mo_v3`` while
# the artifacts directory inside each is ``gf_prod_3mo_artifacts`` -- named for
# the STORE. The guess therefore missed on every run, run_settings.log was
# never read, and the whole data/template/residual section rendered as "plot
# unavailable" with no indication that a path was at fault. Glob for it.
ART_DIR = None
_cands = sorted(glob.glob(os.path.join(RUN_DIR, "*_artifacts")))
_cands += [os.path.join(RUN_DIR, os.path.basename(os.path.normpath(RUN_DIR))
                        + "_artifacts")]
for _c in _cands:
    if os.path.exists(os.path.join(_c, "run_settings.log")):
        ART_DIR = _c
        break
SETTINGS_TXT = ""
if ART_DIR:
    SETTINGS_TXT = open(os.path.join(ART_DIR, "run_settings.log"),
                        errors="replace").read()
else:
    MISSING.append("no *_artifacts/run_settings.log under the run directory; "
                   "the residual spectrum cannot be rebuilt.")

# The GB branch anchors source phase at this epoch; the data grid starts a
# little later, and the offset is a real phase factor, so it is read from the
# run rather than defaulted.
_m0 = re.search(r"\[gb\].*?\n\s+t0:\s*([\d.]+)", SETTINGS_TXT, re.S)
T_REF_SCI = float(_m0.group(1)) if _m0 else 97729089.327664

# ============================ PLOTS =========================================
# ---- 1. likelihood ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for w in range(nwalk):
    ax[0].plot(it, ll[:, w], color=CYAN, alpha=0.25, lw=0.8)
ax[0].plot(it, ll.max(axis=1), color=AMBER, lw=1.8, label="max")
ax[0].plot(it, np.median(ll, axis=1), color=FG, lw=1.2, ls="--", label="median")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("cold-chain lnL"); ax[0].legend()
ax[0].set_title(f"total log-likelihood ({nwalk} walkers)")
ax[1].plot(it, ll.max(axis=1) - ll.min(axis=1), color=VIOLET, lw=1.5)
ax[1].set_xlabel("iteration"); ax[1].set_title("walker lnL spread (max - min)")

# In-pane zoom showing last 50 iterations (user request 2026-09-20).
_n_zoom = min(50, NIT)
_it_zoom = it[-_n_zoom:]
_ll_zoom = ll[-_n_zoom:]
if _n_zoom >= 3:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # Left panel zoom: last 50 iterations of all traces
    axins0 = inset_axes(ax[0], width="35%", height="30%", loc="lower right",
                        borderpad=1.2)
    for w in range(nwalk):
        axins0.plot(_it_zoom, _ll_zoom[:, w], color=CYAN, alpha=0.3, lw=0.7)
    axins0.plot(_it_zoom, _ll_zoom.max(axis=1), color=AMBER, lw=1.4)
    axins0.plot(_it_zoom, np.median(_ll_zoom, axis=1), color=FG, lw=0.9, ls="--")
    axins0.tick_params(labelsize=7)
    axins0.set_title(f"last {_n_zoom} iter", fontsize=7.5, pad=2)
    axins0.grid(True, alpha=0.25, lw=0.5)
    # Right panel zoom: last 50 iterations of spread
    axins1 = inset_axes(ax[1], width="35%", height="30%", loc="upper right",
                        borderpad=1.2)
    axins1.plot(_it_zoom, _ll_zoom.max(axis=1) - _ll_zoom.min(axis=1),
                color=VIOLET, lw=1.3)
    axins1.tick_params(labelsize=7)
    axins1.set_title(f"last {_n_zoom} iter", fontsize=7.5, pad=2)
    axins1.grid(True, alpha=0.25, lw=0.5)

fig_b64(fig, "ll")

# ---- F11: the noise model, in two panels ----------------------------------
# The page used to carry six: two instrument traces, two instrument
# histograms, five foreground traces and five foreground histograms, none of
# which said whether the noise model is RIGHT. What matters is (a) do the two
# instrument parameters recover their injected values, and (b) is the
# foreground coming down as sources leave the residual. Two panels, both
# answerable at a glance.
psd_cold = psd_c[:, 0, :, 0, :]                     # (it, 24, 2)
gal_cold = gal_c[:, 0, :, 0, :]                     # (it, 24, 5)  SAMPLING basis
SOMS_INJ, SA_INJ = 1.496182e-11, 2.982412e-15

# Under GALFOR_LOG_SAMPLING the four log columns are stored as log10 while
# alpha stays linear. The trace/histogram panels want the SAMPLING basis
# (that is what the chain actually explores, and what its prior is uniform
# in); every sensitivity evaluation wants the PHYSICAL one. Keeping both and
# labelling them correctly is the whole fix -- feeding the raw log10 row to
# the foreground model makes amp and f_1 negative, so (f/f_1)**alpha is NaN
# and the entire data/template/residual block raises. That surfaced as a
# fatal KeyError: 'nbins' far downstream, with no page written at all.
from lisatools.globalfit.stock.erebor.noise import (
    GALFOR_BASIS, GALFOR_LOG_PARAMS, galfor_params_to_physical,
    read_noise_model_identity,
)

# The store's OWN basis flag -- never an env var, which describes the current
# process rather than the run that wrote these numbers.
GALFOR_LOG = bool(
    read_noise_model_identity(h5path).get("galfor_log_sampling", False))
gal_cold_phys = galfor_params_to_physical(gal_cold, GALFOR_LOG)
GAL_NAMES = [("log10 " + n) if (GALFOR_LOG and n in GALFOR_LOG_PARAMS) else n
             for n in GALFOR_BASIS]

# The lisatools/eryn import chain above (via erebor.noise -> eryn.utils.plot)
# calls plt.style.use(["science"]), which flips text.usetex=True on any host
# that has scienceplots installed. Every Text object created after this point
# captures rcParams["text.usetex"] at __init__, so we MUST pin it back to
# _USETEX BEFORE the next plt.subplots() call -- the later rcParams.update
# block only takes effect at figure creation time for new Text, not for
# already-existing ones. Fixing it here keeps the page tex-free even when the
# host has no LaTeX (i.e., every cluster node and every fresh container).
plt.rcParams["text.usetex"] = _USETEX

_nsh = min(3, SUB_NIT)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.0))
for j, (name, inj, unit) in enumerate(
        [("Soms_d", SOMS_INJ, "m"), ("Sa_a", SA_INJ, "m/s$^2$")]):
    v = psd_cold[-_nsh:, :, j].ravel()
    ax[j].hist(v, bins=26, color=VIOLET, alpha=0.85)
    ax[j].axvline(inj, color=CYAN, lw=1.6, ls="--", label="injected")
    ax[j].set_title(f"{name}  [{unit}]", fontsize=10)
    ax[j].legend(fontsize=8)
    ax[j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
NOISE_BIAS = [float(np.median(psd_cold[-_nsh:, :, j]) / inj - 1.0)
              for j, inj in enumerate((SOMS_INJ, SA_INJ))]
fig.suptitle(f"instrument-noise posteriors, last {_nsh} stored iterations "
             f"x {nwalk} cold walkers", fontsize=10, color=FG)
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig_b64(fig, "f11_psd")

try:
    # XYZ EVERYWHERE (user ruling 2026-08-19): the run analyses X/Y/Z, so
    # every channel-PSD display uses the X channel and X2TDISens -- no AET
    # projections anywhere on this page.
    from lisatools.sensitivity import get_sensitivity, X2TDISens
    from lisatools import detector as lisa_models
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as HTGF)
    import matplotlib.colors as mcolors

    fr = np.logspace(np.log10(3e-4), np.log10(2.5e-2), 400)

    def sens_curves(soms, sa, galp=None):
        model = lisa_models.LISAModel(soms ** 2, sa ** 2,
                                      lisa_models.DefaultOrbits(), "sampled")
        if galp is None:
            return get_sensitivity(fr, sens_fn=X2TDISens, model=model,
                                   stochastic_params=())
        return get_sensitivity(fr, sens_fn=X2TDISens, model=model,
                               stochastic_params=tuple(galp),
                               stochastic_function=HTGF)

    ramp = mcolors.LinearSegmentedColormap.from_list(
        "amber", ["#FBE3B5", "#F5A623", "#8C5A00"])
    pm = np.median(psd_cold[-1], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.0))
    for k in range(SUB_NIT):
        pk_ = np.median(psd_cold[k], axis=0)
        gk = np.median(gal_cold_phys[k], axis=0)
        ax.plot(fr, sens_curves(pk_[0], pk_[1], gk),
                color=ramp(k / max(SUB_NIT - 1, 1)), lw=1.1,
                label=(f"iteration {k}" if k in (0, SUB_NIT - 1) else None))
    ax.plot(fr, sens_curves(*pm), color=FG, lw=1.4, ls=":",
            label="instrument only (latest)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD, TDI X channel  [1/Hz]")
    ax.legend(fontsize=8, loc="upper left")
    fig_b64(fig, "f11_fg")
except Exception as e:
    MISSING.append(f"foreground curve render failed: {e!r}")


# ---- RESTORED: per-parameter noise traces and histograms -------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
for j, (name, inj) in enumerate([("Soms_d", SOMS_INJ), ("Sa_a", SA_INJ)]):
    for w in range(nwalk):
        ax[j].plot(np.arange(SUB_NIT), psd_cold[:, w, j], color=CYAN, alpha=0.3, lw=0.8)
    ax[j].axhline(inj, color=RED, lw=1.4, ls=":", label="injected")
    ax[j].set_title(f"psd: {name}"); ax[j].set_xlabel("iteration"); ax[j].legend()
fig_b64(fig, "psd_trace")

fig, ax = plt.subplots(1, 2, figsize=(11, 2.9))
for j, (name, inj) in enumerate([("Soms_d", SOMS_INJ), ("Sa_a", SA_INJ)]):
    v = psd_cold[-min(3, SUB_NIT):, :, j].ravel()
    ax[j].hist(v, bins=min(30, nwalk), color=CYAN, alpha=0.85)
    ax[j].axvline(inj, color=RED, lw=1.4, ls=":")
    ax[j].set_title(f"{name} posterior (last {min(3,NIT)} iters x {nwalk} walkers)")
fig_b64(fig, "psd_hist")

fig, ax = plt.subplots(1, 5, figsize=(14, 2.7))
for j in range(5):
    for w in range(nwalk):
        ax[j].plot(np.arange(SUB_NIT), gal_cold[:, w, j], color=AMBER, alpha=0.3, lw=0.8)
    ax[j].set_title(GAL_NAMES[j], fontsize=9); ax[j].set_xlabel("iter")
fig_b64(fig, "gal_trace")
fig, ax = plt.subplots(1, 5, figsize=(14, 2.5))
for j in range(5):
    ax[j].hist(gal_cold[-min(3, SUB_NIT):, :, j].ravel(), bins=20, color=AMBER, alpha=0.85)
    ax[j].set_title(GAL_NAMES[j], fontsize=9)
fig_b64(fig, "gal_hist")

# ---- RESTORED: LISASens curve pair (instrument vs instrument+foreground) ---
# Kept alongside F11 rather than folded into it: F11 plots the A-channel PSD
# the likelihood is weighted by, this pair plots the LISASens sky-averaged
# sensitivity the mission documents quote, and the injected instrument curve
# only exists on this one.
try:
    from lisatools.sensitivity import get_sensitivity, LISASens
    from lisatools import detector as lisa_models
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as HTGF)
    import matplotlib.colors as mcolors

    fr = np.logspace(np.log10(2e-4), np.log10(2.6e-2), 500)

    def sens_lisasens(soms, sa, galp=None):
        model = lisa_models.LISAModel(soms**2, sa**2,
                                      lisa_models.DefaultOrbits(), "mon")
        if galp is None:
            return get_sensitivity(fr, sens_fn=LISASens, model=model,
                                   stochastic_params=())
        return get_sensitivity(fr, sens_fn=LISASens, model=model,
                               stochastic_params=tuple(galp),
                               stochastic_function=HTGF)

    pm = np.median(psd_cold[-1], axis=0)
    gm = np.median(gal_cold_phys[-1], axis=0)

    # Compute injected + FittedHT foreground for comparison
    from lisatools.stochastic import (
        FittedHyperbolicTangentGalacticForeground as _FHT_sens)
    _gal_fht = _FHT_sens.specific_Sh_function(fr, SCI_TOBS)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(fr, sens_lisasens(*pm), color=CYAN, lw=1.6, label="instrument PSD (sampled)")
    ax.plot(fr, sens_lisasens(pm[0], pm[1], gm), color=AMBER, lw=1.6,
            label="PSD + galactic foreground (sampled)")
    ax.plot(fr, sens_lisasens(SOMS_INJ, SA_INJ), color=RED, ls=":", lw=1.3,
            label="injected instrument")
    ax.plot(fr, sens_lisasens(SOMS_INJ, SA_INJ) + _gal_fht, color=RED, ls="-.",
            lw=1.3, alpha=0.8, label="injected + FittedHT foreground")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("Sn(f) [LISASens]")
    ax.legend(fontsize=9); ax.set_title(
        "sensitivity, cold-chain walker-median, latest stored iteration")
    fig_b64(fig, "psd_curves")

    ramp2 = mcolors.LinearSegmentedColormap.from_list(
        "amber", ["#FBE3B5", "#F5A623", "#8C5A00"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(fr, sens_lisasens(*pm), color=CYAN, lw=1.4,
            label="instrument PSD (latest)")
    for k in range(SUB_NIT):
        pk_ = np.median(psd_cold[k], axis=0)
        gk = np.median(gal_cold_phys[k], axis=0)
        ax.plot(fr, sens_lisasens(pk_[0], pk_[1], gk),
                color=ramp2(k / max(SUB_NIT - 1, 1)), lw=1.1,
                label=f"iter {k}" if k in (0, SUB_NIT - 1) else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("Sn(f) [LISASens]")
    ax.legend(); ax.set_title(
        "PSD + foreground per stored iteration (light -> dark = later)")
    fig_b64(fig, "psd_evolution")
except Exception as e:
    MISSING.append(f"LISASens curve render failed: {e!r}")

# ---- GB leaf count (the raw model size; the recovery section below is
# where it is given a denominator) ----------------------------------------
gb_counts = gb_inds.sum(axis=-1)                    # (it, 24)

# ---- detectable-truth set for the overlays (from the census npz) ----------
# `census.py` computes optimal SNR for every catalogue GB over the analysed
# band against the run's OWN sampled sensitivity; the detectable (SNR>7) subset
# is the natural TARGET line for the leaf-count and occupancy panels. It is
# optional -- without the npz these overlays simply do not draw.


def _mhz(x_hz):
    """A band edge as an mHz string, at a precision that survives the low end.

    The band floor is 0.5556 mHz, and the ``:.0f`` these labels used to carry
    renders that as "1" -- a label that is wrong by a factor of two and reads
    as a different band. ``:.4g`` gives "0.5556" and "21.94" from the same
    format, so every band string on the page is produced HERE and the low and
    high edges cannot drift apart in precision.
    """
    return f"{float(x_hz) * 1e3:.4g}"


DET_F0 = None
for _cp in (os.path.join(RUN_DIR, "gb_hi_f_census.npz"), "gb_hi_f_census.npz"):
    if os.path.exists(_cp):
        try:
            _c = np.load(_cp)
            if "det_f0" in _c:
                DET_F0 = np.asarray(_c["det_f0"], dtype=float)
                DET_LO = float(_c["det_lo"]); DET_HI = float(_c["det_hi"])
        except Exception:
            DET_F0 = None
        break

# FALLBACK to the frozen truth npz (2026-08-22). `gb_hi_f_census.npz` is
# written by a `census.py` that lives in a SCRATCH directory, so it does not
# survive a session -- and without it BOTH of the red detectability overlays
# below (the "N detectable (SNR>7)" target line on the leaf-count panel and
# the "has detectable source" curve on the occupancy panel) vanish with no
# entry in MISSING. `gb_truth_3to21.npz` carries the same column: `det` is
# SNR > 7 against the run's own sampled sensitivity over the band stamped in
# `band`, which is exactly what `det_f0` was. Same Tobs guard as the recovery
# section further down -- an unstamped (legacy) npz is treated as 3-month.
if DET_F0 is None:
    _run_tobs = 7776000.0
    try:
        _a0 = dict(f["global_fit/domain_settings/args"].attrs)
        _run_tobs = float(_a0["0"]) * float(_a0["1"]) * float(_a0["2"])
    except Exception:
        pass
    for _tp in (os.path.join(RUN_DIR, "gb_truth_3to21.npz"),
                "gb_truth_3to21.npz"):
        if os.path.exists(_tp):
            try:
                _t = np.load(_tp)
                _t_tobs = (float(np.asarray(_t["tobs"]).reshape(-1)[0])
                           if "tobs" in _t.files else 7776000.0)
                if abs(_t_tobs - _run_tobs) > 1.0:
                    MISSING.append(
                        "detectability overlays skipped: no gb_hi_f_census.npz "
                        f"and the truth set is built for a "
                        f"{_t_tobs / 86400.0:.4g}-day observation while this "
                        f"run is {_run_tobs / 86400.0:.4g} days.")
                elif "det" in _t.files and "f0" in _t.files:
                    _d = np.asarray(_t["det"], dtype=bool)
                    DET_F0 = np.asarray(_t["f0"], dtype=float)[_d]
                    _b = np.asarray(_t["band"], dtype=float).reshape(-1)
                    DET_LO, DET_HI = float(_b[0]), float(_b[1])
            except Exception as _e:
                DET_F0 = None
                MISSING.append(
                    f"detectability overlays unavailable: {_e!r}")
            break
    else:
        MISSING.append(
            "detectability overlays skipped: neither gb_hi_f_census.npz nor "
            "gb_truth_3to21.npz is beside the store or in the CWD, so the "
            "leaf-count and occupancy panels carry no detectable-source "
            "target line.")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for w in range(nwalk):
    ax[0].plot(it, gb_counts[:, w], color=GREEN, alpha=0.35, lw=0.9)
ax[0].plot(it, gb_counts.max(axis=1), color=GREEN, lw=1.8, label="all f")
# TRUTH TARGET. The raw leaf count has no scale -- 577 leaves is only
# meaningful against how many sources are actually THERE. Overlay the
# detectable (SNR>7) catalogue count over the range where SNRs exist, and
# the model's leaf count restricted to that SAME range, so the two lines
# are comparable rather than merely adjacent.
if DET_F0 is not None:
    _f0_it = g["chain/gb"][:NIT, 0, 0][..., 1] * 1e-3     # (it, walker, leaf)
    _in = (_f0_it >= DET_LO) & (_f0_it <= DET_HI) & gb_inds
    _cnt = _in.sum(axis=2)                                # (it, walker)
    ax[0].plot(it, _cnt.max(axis=1), color=CYAN, lw=1.6,
               label=f"{_mhz(DET_LO)}-{_mhz(DET_HI)} mHz")
    ax[0].axhline(DET_F0.size, color=RED, ls=":", lw=1.5)
    ax[0].text(NIT * 0.98, DET_F0.size,
               f"{DET_F0.size} detectable (SNR>7), "
               f"{_mhz(DET_LO)}-{_mhz(DET_HI)} mHz ", color=RED,
               fontsize=8, va="bottom", ha="right")
    ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_title("GB leaf count (cold walkers)"); ax[0].set_xlabel("iteration")
ax[0].set_ylim(bottom=-0.5)
if caps is not None and caps.size:
    _cn = caps.shape[0]                      # may lag NIT (backup fallback)
    im = ax[1].imshow(caps.T, aspect="auto", origin="lower", cmap="viridis",
                      extent=[0, _cn, 0, caps.shape[1]])
    _u = np.unique(caps[-1])
    _lab = CAP_UNIT
    ax[1].set_title(
        f"leaf cap per {_lab}"
        + (f" (ALL at {_u[0]:.0f})" if _u.size == 1 else ""))
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel(_lab)
    fig.colorbar(im, ax=ax[1], shrink=0.85)
else:
    ax[1].text(0.5, 0.5, "leaf caps unreadable\n(snapshot copied mid-save)",
               ha="center", va="center", transform=ax[1].transAxes,
               color=DIM, fontsize=9)
    ax[1].set_xticks([]); ax[1].set_yticks([])
# High-f barren-band birth shutoff (GB_RJ_BAND_SHUTOFF_*): each shutoff
# emits "[GB_BAND_SHUTOFF <move>] band <b> ... births OFF ..." -- mark
# those band rows in red on the cap plot (marker at the right edge +
# translucent row line; the log carries the when, the plot the which).
shutoff_bands = sorted({int(b) for b in re.findall(
    r"\[GB_BAND_SHUTOFF[^\]]*\] band (\d+)", log_text)})
# The SHUTOFF is per BAND by design, but the image is now per CAP CELL --
# band b spans rows [b*K, (b+1)*K), so an unscaled axhline would land at
# 1/K of its true height. Also anchor the marker to the image's OWN x
# extent: caps can come from the backup copy and be LONGER than NIT.
_cx = caps.shape[0]
for b in shutoff_bands:
    _y = (b + 0.5) * CAP_K
    ax[1].axhline(_y, color=RED, lw=1.0, alpha=0.55)
    ax[1].plot([_cx * 0.99], [_y], marker="<", color=RED, ms=6, clip_on=False)
if shutoff_bands:
    ax[1].set_title(
        f"leaf cap per {_lab} ({len(shutoff_bands)} bands birth-OFF, red)")

# In-pane zoom for GB leaf count: last 50 iterations (user request 2026-09-20).
_gb_zoom = gb_counts[-_n_zoom:]
if _n_zoom >= 3:
    axins_gb = inset_axes(ax[0], width="35%", height="30%", loc="center",
                          borderpad=1.2)
    for w in range(nwalk):
        axins_gb.plot(_it_zoom, _gb_zoom[:, w], color=GREEN, alpha=0.35, lw=0.7)
    axins_gb.plot(_it_zoom, _gb_zoom.max(axis=1), color=GREEN, lw=1.4)
    axins_gb.tick_params(labelsize=7)
    axins_gb.set_title(f"last {_n_zoom} iter", fontsize=7.5, pad=2)
    axins_gb.grid(True, alpha=0.25, lw=0.5)

fig_b64(fig, "gb_leaves")

# ---- 5a. CAP-CELL OCCUPANCY ----------------------------------------------
# THE QUESTION THIS PANEL EXISTS TO ANSWER (user, 2026-08-16): "how many of
# the 1232 cap cells have 1 source in them? It should be many more if there
# are really 1232 sub-bands." The band panel above cannot answer it -- it
# shows the cap, not the OCCUPANCY, and it shows bands, not cells. Without
# this the only way to tell whether the cap is binding, and whether the cap
# grid is being followed at all rather than quietly collapsing back onto the
# 154-band grid, is to open the store by hand.
#
# The three questions, one axes each: what does the occupancy distribution
# look like against the cap (is the cap binding?); how do occupancy and the
# at-cap count evolve (is the ramp keeping ahead of the model?); and WHERE in
# frequency the occupied cells sit (which is why the occupied FRACTION is
# small even when the packing is working -- the sources are all in the
# galaxy, not spread over 0.56-21.9 mHz).
cap_cells = _safe(sub, "gb/cap_cell_leaf_cap", None, "per-cell leaf caps")
cap_edges_arr = None
try:
    cap_edges_arr = sub["gb/cap_edges"][:]
except Exception:
    pass
CAP_TXT = ""
if cap_cells is not None and cap_cells.size and cap_edges_arr is not None:
    ncell = cap_edges_arr.size - 1
    # The cap arrays can lag the main backend by a row (separate flush), and
    # an unwritten row reads as all-zero -- which would render as "every cell
    # capped at 0". Use the last row that carries a real cap.
    _crow = cap_cells.shape[0] - 1
    while _crow > 0 and not np.any(cap_cells[_crow] > 0):
        _crow -= 1

    def _cell_counts(iteration):
        """Sources per cap cell, per cold walker, at one stored iteration."""
        alive = g["inds/gb"][iteration, 0, 0]                    # (nw, nleaf)
        f0 = g["chain/gb"][iteration, 0, 0][..., 1] * 1e-3       # mHz -> Hz
        out = np.zeros((alive.shape[0], ncell), dtype=np.int32)
        for w in range(alive.shape[0]):
            fv = f0[w][alive[w]]
            if not fv.size:
                continue
            ci = np.searchsorted(cap_edges_arr, fv, side="right") - 1
            ci = ci[(ci >= 0) & (ci < ncell)]
            out[w] = np.bincount(ci, minlength=ncell)
        return out

    cc_last = _cell_counts(NIT - 1)
    nw_ = cc_last.shape[0]
    cap_row = cap_cells[_crow]
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 3.6),
                           gridspec_kw=dict(wspace=0.42))

    # (a) occupancy distribution vs the cap. Log-scaled counts, because the
    # empty bar is ~5000x the tallest occupied one and a linear axis would
    # render every bar this panel exists to compare as a flat line.
    # Split each bar by whether those cells are AT their own cap, rather
    # than colouring a whole occupancy level: cells holding one source are a
    # mix of cap-1 cells (full) and cap-2 cells (room for one more), and
    # painting the level uniformly would claim ~190 saturated cells where
    # there are ~40.
    _atc = cc_last >= cap_row[None, :]
    kmax = int(max(cc_last.max(), cap_row.max())) + 1
    hist = np.array([(cc_last == k).sum() / nw_ for k in range(kmax + 1)])
    h_at = np.array([((cc_last == k) & _atc).sum() / nw_
                     for k in range(kmax + 1)])
    h_ov = np.array([((cc_last == k) & (cc_last > cap_row[None, :])).sum()
                     / nw_ for k in range(kmax + 1)])
    _x = np.arange(kmax + 1)
    # Log-axis floor so a fractional bar (0.08 cells/walker) is still
    # visible -- but only where the level EXISTS, otherwise an empty level
    # renders as a red stub reading "over cap" where nothing is.
    _F = np.where(hist > 0, 1e-2, 0.0)
    ax[0].bar(_x, np.maximum(hist - h_at, _F), color=CYAN, width=0.72,
              label="below cap")
    ax[0].bar(_x, np.maximum(h_at - h_ov, _F), width=0.72, color=AMBER,
              bottom=np.maximum(hist - h_at, _F), label="at cap")
    if h_ov.sum():
        ax[0].bar(_x, np.maximum(h_ov, _F), width=0.72, color=RED,
                  bottom=np.maximum(hist - h_ov, _F), label="over cap")
    ax[0].legend(loc="upper right", fontsize=8)
    for k, v in enumerate(hist):
        if v <= 0:
            continue
        ax[0].text(k, max(v, 1e-2), f"{v:.0f}" if v >= 1 else f"{v:.2f}",
                   ha="center", va="bottom", color=FG, fontsize=8.5)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-2, hist.max() * 4)
    _st = 1 if kmax <= 8 else 2
    ax[0].set_xticks(np.arange(0, kmax + 1, _st))
    ax[0].set_xlabel("sources in the cell")
    ax[0].set_ylabel(f"cap cells (of {ncell})")
    ax[0].set_title(f"cell occupancy @ iter {NIT-1} (cap {int(cap_row.min())}"
                    f"-{int(cap_row.max())})")

    # (b) the ramp: occupied and at-cap cells per stored iteration. Caps are
    # a SENTINEL (-1) until the GB stage arms them, and "count >= -1" is true
    # of every empty cell -- plotted raw that reads as all 1,232 cells capped
    # before the search even starts. Only plot iterations with a real cap.
    _its, _occ, _tot = [], [], []
    _cap_its, _atcap = [], []
    for i in range(NIT):
        cc = _cell_counts(i)
        capi = cap_cells[min(i, _crow)]
        _its.append(i)
        _occ.append((cc > 0).sum() / nw_)
        _tot.append(cc.sum() / nw_)
        if np.all(capi >= 1):
            _cap_its.append(i)
            _atcap.append((cc >= capi[None, :]).sum() / nw_)
    ax[1].plot(_its, _occ, color=CYAN, lw=2, label="occupied cells")
    ax[1].plot(_cap_its, _atcap, color=AMBER, lw=2, label="at/over cap")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("cap cells")
    ax[1].legend(loc="upper left", fontsize=9)
    axr = ax[1].twinx()
    axr.plot(_its, _tot, color=GREEN, lw=1.4, ls="--")
    axr.set_ylabel("sources / walker", color=GREEN, fontsize=9)
    axr.tick_params(axis="y", colors=GREEN, labelsize=8); axr.grid(False)
    ax[1].set_title("occupancy vs the model")

    # (c) where the occupied cells actually are. Plotted per BAND (8 cells)
    # rather than per cell: 1,232 hairlines across 21 mHz is a moire pattern,
    # not a distribution, and the question here is only "which part of the
    # spectrum is populated".
    K = max(int(round(ncell / max(len(band_edges) - 1, 1))), 1)
    nblk = ncell // K
    occ_cell = (cc_last > 0).mean(axis=0)[:nblk * K].reshape(nblk, K).sum(1)
    fblk = (0.5 * (cap_edges_arr[:-1] + cap_edges_arr[1:])
            )[:nblk * K].reshape(nblk, K).mean(1) * 1e3
    ax[2].fill_between(fblk, 0, occ_cell, color=CYAN, alpha=0.9, lw=0,
                       step="mid", label="model")
    # TRUTH: how many cap cells per band actually CONTAIN a detectable
    # source. The gap between the two curves is the remaining search work,
    # localised in frequency -- which the model curve alone cannot show.
    if DET_F0 is not None:
        _dc = np.clip(np.searchsorted(cap_edges_arr, DET_F0, side="right") - 1,
                      0, ncell - 1)
        _hasdet = np.zeros(ncell, dtype=bool)
        _hasdet[_dc] = True
        _tb = _hasdet[:nblk * K].reshape(nblk, K).sum(1)
        _rng = (fblk >= DET_LO * 1e3) & (fblk <= DET_HI * 1e3)
        ax[2].step(fblk[_rng], _tb[_rng], where="mid", color=RED, lw=1.3,
                   ls=":", label=f"has detectable source "
                                 f"({_mhz(DET_LO)}-{_mhz(DET_HI)} mHz)")
        ax[2].legend(fontsize=8, loc="upper right")
    ax[2].axhline(K, color=DIM, ls=":", lw=1)
    ax[2].text(fblk[-1], K, f" all {K} cells", color=DIM, fontsize=8,
               va="bottom", ha="right")
    ax[2].set_xlabel("f0 [mHz]")
    ax[2].set_ylabel(f"occupied cells per band")
    ax[2].set_title("where the occupied cells are")
    fig_b64(fig, "gb_cap_cells")

    # A young run may have NO iteration with an armed cap yet (_atcap empty:
    # the caps stay at the -1 sentinel until the GB stage arms them) -- report
    # zero at-cap rather than crashing on the empty list.
    _occ_last = _occ[-1] if _occ else 0.0
    _atcap_last = _atcap[-1] if _atcap else 0.0
    _tot_last = _tot[-1] if _tot else 0.0  # same NIT=0 guard as _occ/_atcap
    _exact1 = float((cc_last == 1).sum() / nw_)
    CAP_TXT = (
        f"At iteration {NIT-1} the median cold walker holds "
        f"<strong>{_tot_last:.0f}</strong> GB sources spread over "
        f"<strong>{_occ_last:.0f} of {ncell}</strong> cap cells "
        f"({100*_occ_last/ncell:.0f}%): <strong>{_exact1:.0f}</strong> cells "
        f"hold exactly one source and <strong>{_atcap_last:.0f}</strong> sit "
        f"at or over their cap.")

# ---- 5a2. HIGH-FREQUENCY RECOVERY CENSUS ----------------------------------
# Injection-vs-recovery above 5 mHz, the direct test of whether source
# ADDING is trustworthy. Not computed here: the optimal SNR of every
# catalogue source against the run's sampled sensitivity needs the 2.3 GB
# WDWD catalogue plus a GBGPU waveform each, which does not belong in a
# monitor that has to run in seconds. `census.py` in the scratchpad
# produces `gb_hi_f_census.npz`; drop it beside the store (or in CWD) and
# this section appears. Without it the page degrades to a note.
CENSUS = None
for _cp in (os.path.join(RUN_DIR, "gb_hi_f_census.npz"),
            "gb_hi_f_census.npz"):
    if os.path.exists(_cp):
        try:
            CENSUS = np.load(_cp)
        except Exception:
            CENSUS = None
        break
if CENSUS is None:
    # SAY SO (2026-08-22). This used to degrade to a bare "plot unavailable"
    # placeholder with nothing in MISSING, so the page gave no reason why the
    # one panel whose RED layer means "detectable but NOT recovered" had
    # disappeared. The npz is written by census.py into a scratch directory
    # that does not survive a session, so its absence is the normal case.
    MISSING.append(
        "high-frequency recovery census skipped: no gb_hi_f_census.npz "
        "beside the store or in the CWD. It is produced by census.py (a "
        "scratch-directory script), not by this generator, so a fresh "
        "scratchpad loses it. The red 'missed' scatter, the red 'N "
        "detectable' line and the red 'has detectable source' curve all "
        "come from that file.")
CENSUS_TXT = ""
if CENSUS is not None:
    t_f0 = CENSUS["t_f0"]; t_snr = CENSUS["t_snr"]; found = CENSUS["found"]
    c_rec = CENSUS["rec_f0"]; c_occ = CENSUS["occ"]; c_cap = CENSUS["cap"]
    c_ce = CENSUS["cap_edges"]; FCUT = float(CENSUS["FCUT"])
    nc = c_ce.size - 1
    fig, ax = plt.subplots(1, 3, figsize=(15.4, 3.9),
                           gridspec_kw=dict(wspace=0.30))

    # (a) the census itself: every catalogue source, found or not.
    ax[0].scatter(t_f0[~found] * 1e3, t_snr[~found], s=9, color=RED,
                  alpha=0.55, lw=0, label=f"missed ({(~found).sum()})")
    ax[0].scatter(t_f0[found] * 1e3, t_snr[found], s=14, color=GREEN,
                  alpha=0.95, lw=0, label=f"recovered ({found.sum()})")
    ax[0].axhline(7, color=DIM, ls=":", lw=1)
    ax[0].text(t_f0.max() * 1e3, 7, " SNR 7", color=DIM, fontsize=8,
               ha="right", va="bottom")
    ax[0].set_yscale("log"); ax[0].set_xlabel("f0 [mHz]", fontsize=9)
    ax[0].set_ylabel("optimal SNR", fontsize=9)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"catalogue GBs above {FCUT*1e3:.0f} mHz")

    # (b) recovery rate vs SNR -- the shape that says whether adding is
    # SNR-ordered (healthy) or arbitrary (not).
    eds = np.array([3, 5, 7, 10, 15, 25, 40, 1e9])
    xs, ys, ns = [], [], []
    for a, b in zip(eds[:-1], eds[1:]):
        m = (t_snr >= a) & (t_snr < b)
        if m.sum() >= 4:
            xs.append(np.sqrt(a * min(b, 60))); ys.append(100 * found[m].mean())
            ns.append(m.sum())
    ax[1].plot(xs, ys, "o-", color=CYAN, lw=2, ms=6)
    for x, y, n in zip(xs, ys, ns):
        ax[1].text(x, y, f" {n}", color=DIM, fontsize=8, va="bottom")
    ax[1].set_xscale("log"); ax[1].set_xlabel("optimal SNR", fontsize=9)
    ax[1].set_ylabel("recovered [%]", fontsize=9); ax[1].set_ylim(0, 100)
    ax[1].set_title("recovery vs SNR (labels = N in bin)")

    # (c) THE CEILING. Detectable sources per cap cell against the cap the
    # cell actually carries: everything to the right of the cap line cannot
    # be represented no matter how well the sampler works.
    ci = np.clip(np.searchsorted(c_ce, t_f0, side="right") - 1, 0, nc - 1)
    loud = t_snr > 7
    nloud = np.bincount(ci[loud], minlength=nc)
    kmax = int(nloud.max())
    # k=0 (cells with nothing detectable in them) is ~80% of the grid and
    # says nothing about the ceiling -- start at 1 so the bars that matter
    # are not flattened against it.
    ks = np.arange(1, kmax + 1)
    hh = np.array([(nloud == k).sum() for k in ks])
    capmax = int(np.max(c_cap))
    cols = [AMBER if k > capmax else CYAN for k in ks]
    ax[2].bar(ks, hh, color=cols, width=0.72)
    for k, v in zip(ks, hh):
        ax[2].text(k, v, f"{v}", ha="center", va="bottom", color=FG,
                   fontsize=8.5)
    ax[2].axvline(capmax + 0.5, color=RED, lw=1.5, ls="--")
    ax[2].text(capmax + 0.62, hh.max() * 0.6, f"cap {capmax}", color=RED,
               fontsize=9)
    ax[2].set_ylim(0, hh.max() * 1.35)
    ax[2].set_xticks(ks)
    ax[2].set_xlabel("detectable sources in the cell", fontsize=9)
    ax[2].set_ylabel("cap cells", fontsize=9)
    ax[2].set_title("what the cap can hold vs what is there")
    fig_b64(fig, "gb_hi_f_census")

    _excl = int(np.clip(nloud - c_cap, 0, None).sum())
    _nloud = int(loud.sum())
    CENSUS_TXT = (
        f"Above {FCUT*1e3:.0f} mHz the catalogue holds <strong>{t_f0.size}</strong> "
        f"sources, <strong>{_nloud}</strong> of them detectable (SNR&gt;7) against "
        f"this run's own sampled sensitivity. The max-logL cold walker has recovered "
        f"<strong>{int(found.sum())}</strong>. Of the {_nloud - int(found[loud].sum())} "
        f"detectable ones still missing, <strong>{_excl}</strong> "
        f"({100*_excl/max(_nloud,1):.0f}% of all detectable sources) are excluded BY "
        f"CONSTRUCTION &mdash; they sit in cap cells that already contain more "
        f"detectable sources than the cap permits.")

# ---- 5a3. CAP-DIVISOR STUDY (pre-rendered) --------------------------------
# A STATIC study, not a per-snapshot panel: it asks what the cap grid could
# represent of the mojito galaxy, independent of how far this run has got.
# `divisor_study.py` in the scratchpad renders it; drop the PNG beside the
# store and it appears here.
for _dp in (os.path.join(RUN_DIR, "gb_cap_divisor_study.png"),
            "gb_cap_divisor_study.png"):
    if os.path.exists(_dp):
        with open(_dp, "rb") as _fh:
            IMGS["gb_cap_divisor"] = base64.b64encode(_fh.read()).decode()
        break

# ---- 5b. GB BIRTH FATE (from [GB_ACCEPT rj-split]) ----
# Every RJ propose reports where its birth proposals died. Nothing plotted
# this before, and on the v2 run it is the clearest single view of what the
# new machinery is doing: the SNR-truncated distance proposal should keep
# "snr-clamped" small (it was 59% of scored births before that lever), and
# the cap-cell grid shows up as "capped" (0 before the grid existed).
# Fates are DISJOINT and sum to the reported birth count:
#   capped / oob / prior  -> gated before scoring (cheap)
#   snr / kernel          -> scored, then dropped
#   viable-rejected       -> scored, offered to MH, rejected
#   accepted              -> became a source
RJ_SPLIT_RE = re.compile(
    r"\[GB_ACCEPT rj-split (\w+)\] births (\d+): viable (\d+) "
    r"\(acc (\d+)[^|]*\| gated: prior (\d+) oob (\d+) capped (\d+) "
    r"\| scored-dropped: snr (\d+) kernel (\d+)")
_splits = {}
for m in RJ_SPLIT_RE.finditer(log_text):
    mv = m.group(1)
    births, viable, acc, prior, oob, capped, snr, kern = (
        int(m.group(i)) for i in range(2, 10))
    if births == 0:
        continue  # removal-only moves propose no births
    _splits.setdefault(mv, []).append(
        dict(births=births, accepted=acc, gated=prior + oob + capped,
             capped=capped, snr=snr, kernel=kern,
             viable_rej=max(viable - acc, 0)))
if _splits:
    mv = max(_splits, key=lambda k: len(_splits[k]))
    rows = _splits[mv]
    x = np.arange(len(rows))
    order = [("gated (cap/oob/prior)", "gated", AMBER),
             ("scored, SNR-clamped", "snr", RED),
             ("scored, MH-rejected", "viable_rej", DIM),
             ("ACCEPTED", "accepted", GREEN)]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.4))
    # left: absolute counts, stacked
    bot = np.zeros(len(rows))
    for lab, key, col in order:
        v = np.array([r[key] for r in rows], dtype=float)
        ax[0].bar(x, v, bottom=bot, color=col, label=lab, width=0.85)
        bot += v
    ax[0].set_title(f"{mv}: birth proposals by fate")
    ax[0].set_xlabel("rj propose"); ax[0].set_ylabel("proposals")
    ax[0].legend(fontsize=7, loc="upper left")
    # right: fractions, so the trend is readable as the model fills
    tot = np.array([max(r["births"], 1) for r in rows], dtype=float)
    bot = np.zeros(len(rows))
    for lab, key, col in order:
        v = np.array([r[key] for r in rows], dtype=float) / tot * 100.0
        ax[1].bar(x, v, bottom=bot, color=col, width=0.85)
        bot += v
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("% of births")
    ax[1].set_xlabel("rj propose")
    _last = rows[-1]
    ax[1].set_title(
        f"fate share (last: capped {_last['capped']/tot[-1]*100:.0f}%, "
        f"snr {_last['snr']/tot[-1]*100:.1f}%)")
    fig_b64(fig, "gb_birth_fate")
    GB_FATE_TXT = (
        f"Latest {mv} propose: {_last['births']:,} birth proposals -> "
        f"{_last['gated']:,} gated before scoring "
        f"({_last['capped']:,} by the cap-cell grid), "
        f"{_last['snr']:,} scored-then-SNR-clamped "
        f"({_last['snr']/tot[-1]*100:.1f}%), "
        f"{_last['viable_rej']:,} scored and MH-rejected, "
        f"{_last['accepted']:,} accepted.")
else:
    GB_FATE_TXT = ""

# ---- 6. f-stat fit ----
fdir = os.path.join(RUN_DIR, "gb_fstat_fit", "shared")
epochs = sorted([d for d in os.listdir(fdir)] if os.path.isdir(fdir) else [])
# COMPLETE epochs only (2026-08-15): an epoch directory exists as soon as the
# fit STARTS, and the 23-mo comb alone is a 1.19-billion-evaluation sweep, so
# a snapshot very often catches a half-written epoch. Requiring both cache
# files keeps a mid-fit run from killing the whole page.
epochs = [d for d in epochs
          if os.path.exists(os.path.join(fdir, d, "fstat_grid_comb.npz"))
          and os.path.exists(
              os.path.join(fdir, d, "fstat_grid_peaks_stacked.npz"))]
fstat_meta = {}
if epochs:
    ed = os.path.join(fdir, epochs[-1])
    comb = np.load(os.path.join(ed, "fstat_grid_comb.npz"), allow_pickle=True)
    pk = np.load(os.path.join(ed, "fstat_grid_peaks_stacked.npz"), allow_pickle=True)
    fstat_meta = {"epoch": epochs[-1], "comb_keys": list(comb.keys()), "pk_keys": list(pk.keys())}
    f0n = None; Fv = None
    for k in ("f0_nodes_mHz", "f0_nodes", "f0s", "f0"):
        if k in comb: f0n = np.asarray(comb[k]); break
    for k in ("F_max", "F", "F_vals"):
        if k in comb: Fv = np.asarray(comb[k]); break
    IN_MHZ = "f0_nodes_mHz" in comb
    fig, ax = plt.subplots(figsize=(12, 3.6))
    if f0n is not None and Fv is not None and f0n.shape == Fv.shape:
        n = len(f0n); step = max(1, n // 6000)
        # max-decimate so peaks survive
        m = (n // step) * step
        fd = f0n[:m].reshape(-1, step); Fd = Fv[:m].reshape(-1, step)
        ax.plot(fd[:, 0] * (1.0 if IN_MHZ else 1e3), Fd.max(axis=1), color=CYAN, lw=0.6)
        ax.set_yscale("log"); ax.set_xlabel("f0 [mHz]"); ax.set_ylabel("F")
        ax.set_title(f"F-stat comb scan ({fstat_meta['epoch']}, {n} nodes, max-decimated)")
    fig_b64(fig, "fstat_comb")
    pf0 = None; pF = None
    for k in ("peak_f0_mHz", "peak_f0", "f0", "f0s"):
        if k in pk: pf0 = np.asarray(pk[k]); break
    for k in ("peak_F", "F"):
        if k in pk: pF = np.asarray(pk[k]); break
    PK_MHZ = "peak_f0_mHz" in pk
    if pf0 is not None and pF is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
        ax[0].scatter(pf0 * (1.0 if PK_MHZ else 1e3), np.clip(pF, 1, None), s=4, color=AMBER, alpha=0.6)
        ax[0].set_yscale("log"); ax[0].set_xlabel("f0 [mHz]", fontsize=9); ax[0].set_ylabel("F")
        ax[0].set_title(f"{len(pf0)} peaks (birth-grid anchors)")
        ax[1].hist(pf0 * (1.0 if PK_MHZ else 1e3), bins=80, color=AMBER, alpha=0.85)
        ax[1].set_xlabel("f0 [mHz]"); ax[1].set_title("peak density vs frequency")
        fig_b64(fig, "fstat_peaks")
else:
    MISSING.append(
        "No COMPLETE fstat epoch cache under gb_fstat_fit/shared -- the grid "
        "fit is still running (its epoch dir appears at fit start, the "
        "comb/peaks caches only when it finishes).")

# ---- 7. VGB ----
# Matches the sampled columns in the same order the store writes them, so
# vgb_last[..., j] always corresponds to VGB_NAMES[j].
VGB_NAMES = (
    ["dist [kpc]", "phi0", "cos_iota", "psi", "Mc [Msol]", "fdot_astro_ratio"]
    if VGB_SAMPLED_CHIRP else
    ["dist [kpc]", "phi0", "cos_iota", "psi", "fdot_astro_ratio"]
)
# Per-leaf FIXED frequencies + names from the mojito catalogue (leaf i =
# catalogue row i, fixed-leaf branch).
VGB_F0 = None
VGB_IDS = None

# The monitor rebuilds the data / template / residual stack on the fly and
# reads the VGB, GB and WDWD catalogues on the way; both need a "brick base
# dir" that contains ``data/{GB,VGB,COMBINED,...}/L1/*.h5`` and
# ``catalogues/wdwd_cat_mojito_lite_processed.hdf5``. That directory lives
# somewhere different on every machine, so it must be discoverable via env
# rather than hard-coded. The env chain matches build_truth.py's
# ``_resolve_catalogue()`` (--catalogue > MOJITO_CAT > MOJITO_CACHE_DIR >
# default) so a caller who has already exported MOJITO_CAT for build_truth.py
# does not have to set a second knob. Without the fix, cluster runs (bricks
# under /shared/data/mojito_cache) fell back to the legacy laptop default and
# silently dropped the residual-spectrum and data/template/residual panels.
def _resolve_mojito_cat_dir():
    v = os.environ.get("MOJITO_CAT")
    if v:
        v = os.path.expanduser(v)
        # A file argument (typically .../catalogues/wdwd_cat_*.hdf5) implies
        # the brick base dir is two levels up.
        if os.path.isfile(v):
            return os.path.dirname(os.path.dirname(v))
        if os.path.isdir(v):
            return v
    v = os.environ.get("MOJITO_CACHE_DIR")
    if v:
        v = os.path.expanduser(v)
        # build_truth.py's convention nests the bricks under
        # brickmarket/mojito_light_v1_0_0/; a cluster layout may store them
        # flat under the cache root. Prefer the nested layout when present,
        # else fall through to the flat cache.
        nested = os.path.join(v, "brickmarket", "mojito_light_v1_0_0")
        return nested if os.path.isdir(nested) else v
    return os.path.expanduser(
        "~/.mojito_cache/brickmarket/mojito_light_v1_0_0")


MOJITO_CAT_DIR = _resolve_mojito_cat_dir()


def cat_to_sampled9(entry):
    """Catalogue columns -> the run's 9-column GB sampling basis + an
    amplitude self-check.

    Conventions are NEVER re-derived here. The physical->sampling map is
    ``recipe.gb_catalogue_to_sampling_basis`` (the single source of the
    phi0 SIGN -- sampling phi0 = -TrueAnomaly mod 2pi -- the ICRS
    ``alpha``/``sin_delta`` sky frame and the psi mod-pi wrap), and the
    ``(dist, Mc, fdot_astro_ratio)`` split follows the run's own catalogue
    path in ``stock/erebor/vgb.py``: ``dist = LuminosityDistance`` (Mpc ->
    kpc), ``Mc = ChirpMassSSBFrame``, ``r = fdot_cat / fdot_gr(f0, Mc) - 1``.
    That split reproduces the catalogue Amplitude and fdot EXACTLY (the
    returned ``rel`` is the amplitude residual, asserted small by the run's
    own VGB path).

    Returns ``(rows9, rel_max)`` with rows9 columns
    ``[dist kpc, f0 mHz, Mc, phi0, cos_iota, psi, alpha, sin_delta, ratio]``
    -- the run's ``key_order`` for the gb branch.
    """
    from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis
    from lisatools.globalfit.stock.erebor.transforms import (
        McDistFdotAstroQuad, gb_amp_from_dist)
    rows = np.atleast_2d(gb_catalogue_to_sampling_basis(entry))
    d_kpc = np.asarray(entry["LuminosityDistance"], dtype=float).ravel() * 1e3
    mc = np.asarray(entry["ChirpMassSSBFrame"], dtype=float).ravel()
    f0_hz = rows[:, 1] * 1e-3
    _, _, fdot_gr, _ = McDistFdotAstroQuad()(
        d_kpc, f0_hz, mc, np.zeros_like(d_kpc))
    ratio = rows[:, 2] / fdot_gr - 1.0
    rel = float(np.abs(
        gb_amp_from_dist(f0_hz, mc, d_kpc) / np.exp(rows[:, 0]) - 1.0).max())
    return np.column_stack([d_kpc, rows[:, 1], mc, rows[:, 3], rows[:, 4],
                            rows[:, 5], rows[:, 6], rows[:, 7], ratio]), rel


VGB_TRUTH = None        # (55, ndim) in the VGB sampled basis (5 or 6 cols)
VGB_TRUTH_REL = None
try:
    from lisatools.globalfit.stock.erebor.vgb import load_vgb_catalogue_file
    _cat = load_vgb_catalogue_file(MOJITO_CAT_DIR)
    _v = np.asarray(_cat["vgb"]).item()
    VGB_F0 = np.asarray(_v["GW22FrequencySSBFrame"]) * 1e3   # mHz
    VGB_IDS = [i.decode() if isinstance(i, bytes) else str(i)
               for i in _v["ID"]]
    # Legacy VGB sampled basis (5 params) = ["dist", "phi0", "cos_iota",
    # "psi", "fdot_astro_ratio"] = 9-col GB indices [0, 3, 4, 5, 8].
    # Chirp basis (6 params, 6-month runs) = ["dist", "phi0", "cos_iota",
    # "psi", "Mc", "fdot_astro_ratio"] = 9-col GB indices
    # [0, 3, 4, 5, 2, 8] -- Mc slot goes IN THE MIDDLE, matching the store
    # column order in ``vgb_c[..., 4]``.
    _r9, VGB_TRUTH_REL = cat_to_sampled9(_v)
    _truth_cols = ([0, 3, 4, 5, 2, 8] if VGB_SAMPLED_CHIRP
                   else [0, 3, 4, 5, 8])
    VGB_TRUTH = _r9[:, _truth_cols]
except Exception as e:
    MISSING.append(f"VGB catalogue f0 axis unavailable locally: {e!r}")

# ---- 7b. DATA / TEMPLATE SUM / RESIDUAL, frequency + WDM domains ----------
# The one panel that answers "is the fit actually subtracting anything".
# Layout mirrors the run's OWN debug convention,
# ``gbspecialstretch.py::_debug_plot_band_sequence`` (and its addremove twin
# ``addremovemove.py::_debug_plot_source_sequence``): rows = the TDI channels
# the run analyses (X/Y/Z), columns = the three states, WDM panels rendered as
# ``imshow(|arr|, origin="lower")`` under ONE shared color scale per channel
# row so a shrinking residual renders DARK instead of autoscaling back up.
# Column order follows the request: data | template sum | residual.
#
# SOURCING (documented choice; nothing here is hand-rolled):
#   * data      -- the mojito L1 bricks the run itself loaded (NOISE + GB +
#                  VGB, from ``processor_init_kwargs`` in run_settings.log),
#                  read with the mojito reader's LAZY slicing so only the
#                  analysis window's samples ever enter memory, summed exactly
#                  as ``L1DataLoader.load_data`` sums them, then pushed
#                  through the installed ``TDSignal.fft`` / ``FDSignal
#                  .transform(WDMSettings)`` with the run's own Tukey window.
#                  The snapshot itself carries NO data/residual arrays (the
#                  only rendered one is artifacts/wdm_data.png, produced by
#                  ``engine.py``'s ``WDMSignal.heatmap`` -- an image, not
#                  numbers), so the streams must be rebuilt.
#   * templates -- the LAST stored cold-chain iteration's coordinates for the
#                  MAX-lnL walker, run through the run's own sampling->physical
#                  ``make_gb_transform_container`` and the installed
#                  ``gbgpu.gbcomps.GBFDComputations`` FD chunked-heterodyne
#                  kernel (the same kernel family the WDM analysis path uses;
#                  no phase-convention fixup, unlike the legacy fastGB path).
#   * residual  -- data - (GB + VGB), per channel, in BOTH domains.
#
# TIME-ORIGIN PHASE FACTOR: the GB/VGB branches anchor source phases at
# ``recipe.MOJITO_REFERENCE_TIME`` while the data grid starts at the brick's
# own t0, 850.5 s later. The WDM comp carries that offset internally
# (``_wdm.t0 = data_t0`` with ``t_ref = si.t0``); ``GBFDComputations`` instead
# REQUIRES ``t_start == t_ref``, so the generated template lands on a grid
# anchored at t_ref and must be advanced onto the data grid by the exact
# time-shift factor exp(+2 pi i f dt). Verified end-to-end, not assumed: with
# the CATALOGUE-TRUTH VGB parameters this route reproduces the VGB-only mojito
# brick at complex overlap 0.9999 per channel (residual power 1.8e-4 of the
# brick); without the factor the same comparison reads 0.77 with a ~90 deg
# phase.
DTR = {}
try:
    import glob as _glob
    from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME
    from lisatools.detector import L1Orbits
    from lisatools.domains import (FDSettings, FDSignal, TDSettings, TDSignal,
                                   WDMSettings)
    from lisatools.utils.utility import windowfun
    from lisatools.globalfit.stock.erebor.transforms import (
        make_gb_transform_container)
    from lisatools.response.tdiconfig import TDIConfig
    from gbgpu.gbcomps import GBFDComputations
    from mojito import MojitoL1File

    if VGB_TRUTH is None:
        raise RuntimeError("VGB catalogue unavailable; cannot fix VGB leaves")

    # --- the run's own grid, straight off the stored domain settings ---
    _a = dict(f["global_fit/domain_settings/args"].attrs)
    _k = dict(f["global_fit/domain_settings/kwargs"].attrs)
    W_NF, W_NT, W_DT = int(_a["0"]), int(_a["1"]), float(_a["2"])
    N_TD = W_NF * W_NT
    # window: the engine builds tukey(Nt_td, alpha=window_taper_duration/Tobs)
    _settings_txt = SETTINGS_TXT
    if not _settings_txt:
        raise FileNotFoundError("run_settings.log not found beside the store")
    _mt = re.search(r"window_taper_duration:\s*([\d.eE+-]+)", _settings_txt)
    WIN_ALPHA = (float(_mt.group(1)) / (N_TD * W_DT)) if _mt else 0.0
    _mt0 = re.search(r"\[gb\].*?\n\s+t0:\s*([\d.]+)", _settings_txt, re.S)
    T_REF = float(_mt0.group(1)) if _mt0 else float(MOJITO_REFERENCE_TIME)

    # --- which mojito bricks the run summed (source_ids + instrument noise) --
    _pk = re.search(r"processor_init_kwargs:\s*\{(.*)\}\s*$", _settings_txt,
                    re.M)
    _pk = _pk.group(1) if _pk else ""
    _types = [t for t in ("GB", "VGB", "MBHB", "EMRI", "SOBHB")
              if re.search(rf"'{t}':\s*\[", _pk)]
    if re.search(r"'add_instrument_noise':\s*'mojito'", _pk):
        _types = ["NOISE"] + _types
    if not _types:
        _types = ["NOISE", "GB", "VGB"]
    _SUBDIR = {"NOISE": "INSTRUMENT"}
    _files = {}
    for _t in _types:
        _hits = sorted(_glob.glob(os.path.join(
            MOJITO_CAT_DIR, "data", _SUBDIR.get(_t, _t), "L1", f"{_t}_*.h5")))
        if not _hits:
            raise FileNotFoundError(f"no local mojito {_t} L1 brick")
        _files[_t] = _hits[0]

    # --- orbits: the run's L1Orbits, but only the window's light-travel times
    class _WindowedL1Orbits(L1Orbits):
        """L1Orbits that reads only the analysis window's ltt table.

        Numerically identical to the stock class over [t0, t0 + Tobs): the
        C++ detector addresses the table as (t - ltt_t0)/ltt_dt, so dropping
        the tail past the window changes nothing that is ever evaluated. It
        avoids holding the full 731-day 25.2M x 6-link table (1.2 GB) plus
        its C++ copy for a 90-day analysis -- this generator has to run on a
        laptop.
        """
        n_ltt = int(N_TD + 20000)

        def _setup(self):
            with self.open() as _fh:
                _n = self.n_ltt
                self.ltt = _fh.ltts.ltts[:_n]
                self.ltt_t = _fh.ltts.time_sampling.t(slice(0, _n))
                self.x_base = _fh.orbits.positions[:]      # frame == "icrs"
                self.v_base = _fh.orbits.velocities[:]
                self.sc_t_base = _fh.orbits.time_sampling.t()
                self.size_base = self.sc_t_base.shape[0]
                self.dt_base = float(_fh.orbits.time_sampling.dt)
                self.ltt_dt = _fh.ltts.time_sampling.dt
                self.sc_dt = _fh.orbits.time_sampling.dt
                self.ltt_t0 = float(self.ltt_t[0])
                self.sc_t0 = float(self.sc_t_base[0])

    _orb = _WindowedL1Orbits(_files[_types[0]], force_backend="cpu",
                             frame="icrs", linear_interp_dt=500.0)
    _orb._ensure_configured()
    # stash for the SNR machinery below: the analytic DefaultOrbits ephemeris
    # differs from the mojito L1 orbits by the FULL annual-Doppler phase at
    # the run epoch (+-16 FD bins at 20 mHz) -- high-f SNRs/overlaps computed
    # on DefaultOrbits are wrong (2026-08-19 finding).
    _L1_ORB_CPU = _orb

    # --- data: partial (lazy) brick reads, summed on the analysis window ----
    _td = np.zeros((3, N_TD), dtype=np.float64)
    _t0_data = None
    for _t, _fp in _files.items():
        with MojitoL1File(_fp) as _fh:
            _chunk = _fh.tdis.xyz_doppler[:N_TD]          # lazy -> partial IO
            _td += np.asarray(_chunk).T
            del _chunk
            if _t0_data is None:
                _t0_data = float(_fh.tdis.time_sampling.t0)
    _win, _ = windowfun("tukey", N_TD, alpha=WIN_ALPHA)
    _fd_data = TDSignal(_td, TDSettings(t0=_t0_data, dt=W_DT, N=N_TD,
                                        force_backend="cpu")
                        ).fft(settings=None, window=_win)
    FDS = _fd_data.settings
    data_fd = _fd_data.arr
    _win_keep = _win
    del _td, _fd_data, _win

    # --- templates from the max-lnL cold walker's last stored coordinates ---
    WBEST = int(np.argmax(ll[-1]))
    _gb9 = gb_chain_cold[WBEST][gb_alive_last[WBEST]]        # (n_gb, 9)
    _v5 = vgb_c[-1, WBEST]                                   # (55, 5) or (55, 6)
    # Reassemble to the 9-column GB basis
    # [dist, f0, Mc, phi0, cos_iota, psi, alpha, sin_delta, fdot_astro_ratio].
    # LEGACY (5-param DIST basis): Mc is catalogue-fixed (_r9[:, 2]),
    #   fdot_astro_ratio is sampled at _v5[:, 4].
    # CHIRP (6-param): Mc is SAMPLED at _v5[:, 4], and fdot_astro_ratio
    #   moves to _v5[:, 5]. Getting this wrong silently zeroes the VGB
    #   template amplitude (Mc value ~0.3 winds up in the ratio slot and
    #   the amplitude transform d(A)/d(Mc,dist,...) collapses) -- the
    #   sampler subtracts as normal, but the monitor's rebuild does not,
    #   and the DTR residual reads as if HM Cnc etc. were never fit.
    if VGB_SAMPLED_CHIRP:
        _vgb9 = np.column_stack([_v5[:, 0], _r9[:, 1], _v5[:, 4],
                                 _v5[:, 1], _v5[:, 2], _v5[:, 3],
                                 _r9[:, 6], _r9[:, 7], _v5[:, 5]])
    else:
        _vgb9 = np.column_stack([_v5[:, 0], _r9[:, 1], _r9[:, 2],
                                 _v5[:, 1], _v5[:, 2], _v5[:, 3],
                                 _r9[:, 6], _r9[:, 7], _v5[:, 4]])
    _tf = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                      use_distance=True, mc_lims=(0.001, 1.0))
    _comp = GBFDComputations(
        FDSettings(FDS.N, FDS.df, min_freq=0.0, max_freq=None,
                   force_backend="cpu"),
        T_REF, t_start=T_REF, N_sparse=2048, orbits=_orb,
        tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_type="XYZ", nchannels=3, force_backend="cpu",
        tukey_alpha=WIN_ALPHA, edge_frac=0.0)
    _fr = np.asarray(FDS.f_arr)
    _shift = np.exp(2j * np.pi * _fr * (_t0_data - T_REF))[None, :]
    _tmpl = {}
    for _nm, _rows in (("gb", _gb9), ("vgb", _vgb9)):
        _arr = np.zeros((1, 3, FDS.N), dtype=np.complex128)
        _comp.fill_global(_tf.both_transforms(_rows.copy()), _arr,
                          convert_to_ra_dec=False)
        _tmpl[_nm] = _arr[0] * _shift
        del _arr
    resid_fd = data_fd - _tmpl["gb"] - _tmpl["vgb"]
    DTR.update(n_gb=int(_gb9.shape[0]), n_vgb=int(_vgb9.shape[0]),
               walker=WBEST, lnl=float(ll[-1, WBEST]),
               dt_shift=float(_t0_data - T_REF))

    # --- numeric sanity: the loudest recovered GB, data vs residual ---------
    _ig = int(np.argmax(np.abs(_tmpl["gb"][0])))
    _w = slice(max(_ig - 40, 0), _ig + 41)
    DTR.update(
        chk_f0=float(_fr[_ig] * 1e3),
        chk_dpk=float(np.abs(data_fd[0, _ig])),
        chk_rpk=float(np.abs(resid_fd[0, _ig])),
        chk_dp=float(np.sum(np.abs(data_fd[:, _w]) ** 2)),
        chk_rp=float(np.sum(np.abs(resid_fd[:, _w]) ** 2)))
    _mb = (_fr >= band_edges[0]) & (_fr <= band_edges[-1])
    DTR.update(band_dp=float(np.sum(np.abs(data_fd[:, _mb]) ** 2)),
               band_rp=float(np.sum(np.abs(resid_fd[:, _mb]) ** 2)))

    # ===================== FD figure ======================================
    CHN = ["X", "Y", "Z"]
    _sel = (_fr >= 1e-4) & (_fr <= 2.6e-2)
    _fs = _fr[_sel] * 1e3
    _FLO, _FHI = band_edges[0] * 1e3, band_edges[-1] * 1e3
    _YLO = 1e-19

    def _maxdec(x, y, npts=2200):
        """Log-frequency RMS binning: a 1.55M-bin spectrum has to lose 99.9%
        of its points to fit a PNG. Earlier this was a linear block-MAX of
        |y|; that preserved narrow GB lines but the sub-band region reads
        as visible waviness because Tukey-window sidelobes from strong
        sub-mHz TDI content sinc-leak across every bin below FLO and the
        MAX picks up their peaks. Log-frequency RMS (sqrt of the mean of
        |y|^2 per log-f bin) collapses the sinc envelope to its RMS while
        narrow GB lines still show as bumps above the residual floor
        (in-band, thousands of noise bins per log-f pixel drop the RMS
        floor well below the GB peak). Matches what the F1 residual-
        spectrum panel already does (see the ``_bmean`` block below).
        """
        x = np.asarray(x, float)
        y = np.abs(np.asarray(y))
        pos = x > 0
        if not pos.any():
            return x[:0], y[:0]
        xL, yL = x[pos], y[pos]
        ed = np.logspace(np.log10(xL[0]), np.log10(xL[-1]), npts + 1)
        bi = np.clip(np.searchsorted(ed, xL, side="right") - 1, 0, npts - 1)
        cnt = np.bincount(bi, minlength=npts).astype(float)
        s2 = np.bincount(bi, weights=yL ** 2, minlength=npts)
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = np.sqrt(s2 / np.maximum(cnt, 1))
        fc = np.sqrt(ed[:-1] * ed[1:])
        ok = cnt > 0
        return fc[ok], rms[ok]

    _cols = [
        ("data (mojito " + " + ".join(_types) + ")",
         [("data", data_fd, DIM, 1.0, 0.9)]),
        ("template sum (GB + VGB)",
         [("data", data_fd, DIM, 0.35, 0.9), ("GB", _tmpl["gb"], GREEN, 1.0, 0.7),
          ("VGB", _tmpl["vgb"], VIOLET, 1.0, 0.7)]),
        ("residual = data - templates",
         [("data", data_fd, DIM, 0.55, 1.1),
          ("residual", resid_fd, CYAN, 1.0, 0.7)]),
    ]
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 8.2), sharex=True, sharey=True)
    for r in range(3):
        for c, (ttl, series) in enumerate(_cols):
            a_ = ax[r][c]
            for lab, arr, col, al, lw in series:
                x_, y_ = _maxdec(_fs, arr[r][_sel])
                y_ = np.where(y_ > _YLO, y_, np.nan)   # no off-scale spikes
                a_.plot(x_, y_, color=col, lw=lw, alpha=al,
                        label=(lab if r == 0 else None))
            a_.set_xscale("log"); a_.set_yscale("log")
            a_.axvline(_FLO, color=RED, ls=":", lw=1.0)
            a_.axvline(_FHI, color=RED, ls=":", lw=1.0)
            if r == 0:
                a_.set_title(ttl, fontsize=10)
                # lower-left is the one corner empty in all three columns
                _lg = a_.legend(fontsize=8, loc="lower left")
                for _lh in _lg.get_lines():
                    _lh.set_linewidth(2.2)
            if c == 0:
                a_.set_ylabel(f"{CHN[r]}\n|TDI(f)|  [1/Hz]", fontsize=9)
            if r == 2:
                a_.set_xlabel("f [mHz]", fontsize=9)
    ax[0][0].set_ylim(_YLO, 3e-15)
    fig.suptitle(
        f"frequency domain - cold walker {WBEST} (max lnL), stored iteration "
        f"{NIT - 1} - dotted red = the run's GB band edges", fontsize=10,
        color=FG)
    plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_b64(fig, "dtr_fd")

    # ===================== WDM figure =====================================
    _wdm = WDMSettings(W_NF, W_NT, W_DT, t0=float(_k["t0"]),
                       oversample=int(_k["oversample"]),
                       min_freq=float(_k["min_freq"]),
                       max_freq=float(_k["max_freq"]),
                       min_time=float(_k["min_time"]),
                       max_time=float(_k["max_time"]),
                       is_complex=bool(_k["is_complex"]),
                       force_backend="cpu")
    DEC = 5                       # WDM time pixels pooled per plotted column
    _wp = {}
    for _nm, _arr in (("data", data_fd), ("tmpl", _tmpl["gb"] + _tmpl["vgb"]),
                      ("res", resid_fd)):
        _m = np.abs(np.asarray(FDSignal(_arr.copy(), FDS).transform(_wdm).arr))
        _n = (_m.shape[-1] // DEC) * DEC
        # MAX-pool the time axis down to plot resolution immediately; the
        # full-resolution map is never carried past this line.
        _wp[_nm] = _m[..., :_n].reshape(_m.shape[0], _m.shape[1], -1,
                                        DEC).max(axis=-1)
        del _m
    _te = np.asarray(_wdm.t_arr_edges); _fe = np.asarray(_wdm.f_arr_edges)
    _ext = [0.0, (_te[-1] - _te[0]) / 86400.0, _fe[0] * 1e3, _fe[-1] * 1e3]
    DTR.update(wdm_shape=tuple(int(s) for s in _wp["data"].shape), wdm_dec=DEC,
               layer_df=float(_wdm.layer_df), layer_dt=float(_wdm.layer_dt))
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 8.0), sharex=True, sharey=True)
    _tt = ["data", "template sum (GB + VGB)", "residual = data - templates"]
    for r in range(3):
        # ONE linear scale per channel row, keyed to that row's DATA panel
        # (the addremove debug convention: norm from the total-data column,
        # one per channel, shared across every frame) -- so the residual
        # panel darkens as sources leave instead of re-autoscaling.
        vmax = float(np.percentile(_wp["data"][r], 99.5))
        for c, kk in enumerate(("data", "tmpl", "res")):
            a_ = ax[r][c]
            # plotted as a FRACTION of the row scale: 0-1 ticks keep the
            # colorbar free of a floating 1e-20 offset label, and the shared
            # per-row normalization becomes explicit rather than implied.
            im = a_.imshow(_wp[kk][r] / vmax, aspect="auto", origin="lower",
                           extent=_ext, vmin=0.0, vmax=1.0, cmap="viridis",
                           interpolation="nearest")
            a_.grid(False)
            if r == 0:
                a_.set_title(_tt[c], fontsize=10)
            if c == 0:
                a_.set_ylabel(f"{CHN[r]}\nfrequency [mHz]", fontsize=9)
            if r == 2:
                a_.set_xlabel("time [days from data start]", fontsize=9)
            if c == 2:
                cb = fig.colorbar(im, ax=a_, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
                cb.set_label(f"|w| / {vmax:.2e}", fontsize=7, color=DIM)
    fig.suptitle(
        f"WDM domain - |w_mn| on the run's own grid "
        f"({_wdm.Nf_active} layers x {_wdm.layer_df * 1e3:.4f} mHz, "
        f"{int(_wdm.layer_dt)} s pixels) - shared linear scale per channel row",
        fontsize=10, color=FG)
    plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    # 9 dense speckle images are the single most expensive PNG on the page;
    # 88 dpi still oversamples the pooled (layer x pooled-time) grid.
    fig_b64(fig, "dtr_wdm", dpi=88)

    # ================= F1: the residual spectrum ==========================
    # The field's canonical "is the fit subtracting anything" figure (Katz+
    # 2405.04690 Fig 5): log-log power spectral density against frequency in
    # Hz, with the injected data, the template sum and the residual on one
    # axes, and the noise decomposition of ESA Red Book Fig 2.2 underneath --
    # instrument noise alone, the unresolved galactic foreground alone, and
    # their sum. The GAP between the instrument curve and the sum IS the
    # galactic confusion, so the residual's position relative to those two
    # lines reads directly as "how much of the galaxy is still unmodelled".
    #
    # ORDINATE, stated because the field is not consistent about it: this is
    # the PSD of the TDI *X* channel in 1/Hz. It is not a strain amplitude and
    # it is not an ASD. XYZ everywhere (user ruling 2026-08-19): the run
    # analyses X/Y/Z, so the display stays in X against the X2TDISens noise
    # model -- no AET projection is formed anywhere on this page.
    def _AE(x):
        return (x[0], None)

    _W2 = float(np.mean(_win_keep ** 2))

    def _PSD(x):
        return 2.0 * np.abs(x) ** 2 / (N_TD * W_DT * _W2)

    _dA, _ = _AE(data_fd)
    _tA, _ = _AE(_tmpl["gb"] + _tmpl["vgb"])
    _rA, _ = _AE(resid_fd)

    from lisatools.sensitivity import get_sensitivity, X2TDISens
    from lisatools import detector as lisa_models
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as _HTGF,
        FittedHyperbolicTangentGalacticForeground as _FHT)
    _pm = np.median(psd_cold[-1], axis=0)
    _gm = np.median(gal_cold_phys[-1], axis=0)
    _lmod = lisa_models.LISAModel(_pm[0] ** 2, _pm[1] ** 2,
                                  lisa_models.DefaultOrbits(), "sampled")
    _fpos = np.maximum(_fr, FDS.df)
    _Sinst = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens,
                                        model=_lmod, stochastic_params=()),
                        float)
    _Ssum = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens, model=_lmod,
                                       stochastic_params=tuple(_gm),
                                       stochastic_function=_HTGF), float)
    _Sgal = np.maximum(_Ssum - _Sinst, 1e-60)

    # Injected instrument + FittedHyperbolicTangent foreground for comparison.
    _lm_inj = lisa_models.LISAModel(SOMS_INJ ** 2, SA_INJ ** 2,
                                    lisa_models.DefaultOrbits(), "injected")
    _Sinst_inj = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens,
                                            model=_lm_inj, stochastic_params=()),
                            float)
    _Ssum_inj = np.asarray(get_sensitivity(_fpos, sens_fn=X2TDISens,
                                           model=_lm_inj,
                                           stochastic_params=(SCI_TOBS,),
                                           stochastic_function=_FHT), float)
    _Sgal_inj = np.maximum(_Ssum_inj - _Sinst_inj, 1e-60)

    # log-f binning. A PSD is an AVERAGE of periodogram bins, so the reduction
    # is a mean per log-frequency bin -- not the block-MAX a line spectrum
    # wants, and not a median, which would read the template (exactly zero
    # between sources) as identically zero everywhere.
    # Below the run's own GB band the windowed data is dominated by
    # spectral leakage from the enormous sub-mHz TDI content -- a sinc
    # pattern that is real but is not analysed and reads as structure.
    _sel = (_fr >= max(float(band_edges[0]), 3e-4)) & (_fr <= 2.5e-2)
    _fs = _fr[_sel]
    _NB = 300
    _ed = np.logspace(np.log10(_fs[0]), np.log10(_fs[-1]), _NB + 1)
    _bi = np.clip(np.searchsorted(_ed, _fs, side="right") - 1, 0, _NB - 1)
    _cnt = np.bincount(_bi, minlength=_NB).astype(float)
    _fcb = np.sqrt(_ed[:-1] * _ed[1:])
    _ok = _cnt > 0

    def _bmean(v):
        out = np.full(_NB, np.nan)
        out[_ok] = np.bincount(_bi, weights=v, minlength=_NB)[_ok] / _cnt[_ok]
        return out

    _Pd, _Pt = _bmean(_PSD(_dA[_sel])), _bmean(_PSD(_tA[_sel]))
    _Pr = _bmean(_PSD(_rA[_sel]))
    _Ni = _bmean(_Sinst[_sel]); _Ng = _bmean(_Sgal[_sel]); _Ns = _bmean(_Ssum[_sel])
    _Ni_inj = _bmean(_Sinst_inj[_sel])
    _Ng_inj = _bmean(_Sgal_inj[_sel])
    _Ns_inj = _bmean(_Ssum_inj[_sel])

    # Welch-in-frequency-domain smoothing on the log-f binned periodogram.
    # A single-windowed periodogram bin has Chi^2(2) statistics
    # (sigma/mu = 1 per FD bin), and the log-f block-MEAN above beats that
    # to sigma/mu = 1/sqrt(N) where N is the number of FD bins per log-f
    # pixel. At the low-f end where N is small, ~5-10% variance is still
    # visible on log-y over five decades of PSD range and reads as
    # waviness. Convolving with a Hann kernel of half-width H log-f bins
    # is equivalent to Welch's method with a Hann synthesis window across
    # neighbouring subbands: it multiplies the effective DOF by
    # (2H+1)*<w^2>/<w>^2, so H=2 (5-tap Hann) roughly triples the DOF and
    # drops sigma/mu by ~sqrt(3). Total power is preserved because the
    # kernel is normalised, and the model curves (_Ni/_Ng/_Ns) are already
    # smooth so the same kernel on them is a no-op up to floating point.
    # NaN-safe: bins the block-mean flagged as empty stay NaN.
    def _smooth_logf(y, half=2):
        w = np.hanning(2 * half + 1)
        w = w / w.sum()
        m = np.isfinite(y).astype(float)
        yz = np.where(np.isfinite(y), y, 0.0)
        yc = np.convolve(yz, w, mode="same")
        mc = np.convolve(m, w, mode="same")
        return np.where(mc > 1e-9, yc / np.maximum(mc, 1e-30), np.nan)
    _Pd = _smooth_logf(_Pd); _Pt = _smooth_logf(_Pt); _Pr = _smooth_logf(_Pr)
    _Ni = _smooth_logf(_Ni); _Ng = _smooth_logf(_Ng); _Ns = _smooth_logf(_Ns)
    _Ni_inj = _smooth_logf(_Ni_inj); _Ng_inj = _smooth_logf(_Ng_inj)
    _Ns_inj = _smooth_logf(_Ns_inj)

    # whitened residual: real and imaginary parts are each N(0,1) when the
    # noise model is right, so the ratio below sits at 1 and the
    # Anderson-Darling test on them has nothing to reject.
    _zA = _rA[_sel] / np.sqrt(np.maximum(_Ssum[_sel], 1e-60)
                              * N_TD * W_DT * _W2 / 4.0)
    _rat = _bmean(np.abs(_zA) ** 2 / 2.0)

    # Rosati & Littenberg (2410.17180) Fig 4: colour the residual trace by the
    # Anderson-Darling Gaussianity p-value of the whitened residual, so the
    # confusion region identifies ITSELF as the non-Gaussian zone instead of
    # being annotated by hand. The test is run on a fixed 600-sample draw per
    # bin: with 10^4 samples it rejects on effect sizes far too small to care
    # about, and the bins would no longer be comparable to each other.
    _adp = np.full(_NB, np.nan)
    try:
        from scipy.stats import anderson as _anderson
        _rng = np.random.default_rng(0)
        _ord = np.argsort(_bi, kind="stable")
        _zs = _zA[_ord]
        _bnd = np.searchsorted(_bi[_ord], np.arange(_NB + 1))
        for _b in range(_NB):
            _v = _zs[_bnd[_b]:_bnd[_b + 1]]
            if _v.size < 40:
                continue
            _v = np.concatenate([_v.real, _v.imag])
            if _v.size > 600:
                _v = _rng.choice(_v, 600, replace=False)
            _A2 = float(_anderson(_v, dist="norm").statistic)
            _n = _v.size
            _As = _A2 * (1 + 0.75 / _n + 2.25 / _n ** 2)
            if _As < 0.2:
                _p = 1 - np.exp(-13.436 + 101.14 * _As - 223.73 * _As ** 2)
            elif _As < 0.34:
                _p = 1 - np.exp(-8.318 + 42.796 * _As - 59.938 * _As ** 2)
            elif _As < 0.6:
                _p = np.exp(0.9177 - 4.279 * _As - 1.38 * _As ** 2)
            else:
                _p = np.exp(1.2937 - 5.709 * _As + 0.0186 * _As ** 2)
            _adp[_b] = np.log10(float(np.clip(_p, 1e-12, 1.0)))
    except Exception as _e:
        MISSING.append(f"residual Gaussianity test unavailable: {_e!r}")

    fig, ax = plt.subplots(2, 1, figsize=(11.6, 6.8), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.5, 1],
                                            hspace=0.07))
    ax[0].plot(_fcb, _Pd, color=CYAN, lw=1.6, label="injected data")
    ax[0].plot(_fcb, _Pt, color=GREEN, lw=1.2, label="template sum (GB + VGB)")
    ax[0].plot(_fcb, _Pr, color=VIOLET, lw=1.4, label="residual")
    ax[0].plot(_fcb, _Ni, color=FG, lw=1.2, ls=":", label="instrument noise (sampled)")
    ax[0].plot(_fcb, _Ng, color=AMBER, lw=1.2, ls=":",
               label="galactic foreground (sampled)")
    ax[0].plot(_fcb, _Ns, color=FG, lw=1.3, ls="--", label="their sum (sampled)")
    ax[0].plot(_fcb, _Ni_inj, color=RED, lw=1.0, ls=":", alpha=0.7,
               label="instrument noise (injected)")
    ax[0].plot(_fcb, _Ng_inj, color=RED, lw=1.0, ls="-.", alpha=0.7,
               label="FittedHyperbolicTangent foreground")
    ax[0].plot(_fcb, _Ns_inj, color=RED, lw=1.2, ls="--", alpha=0.7,
               label="their sum (injected)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_ylabel("PSD, TDI X channel  [1/Hz]")
    ax[0].set_ylim(1e-45, 2e-37)
    ax[0].legend(fontsize=7.5, loc="upper left", ncols=2)
    _sc = ax[1].scatter(_fcb, _rat, c=_adp, cmap="magma", s=11, vmin=-8,
                        vmax=0, lw=0)
    ax[1].axhline(1.0, color=FG, ls="--", lw=1.0)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_ylim(0.25, 40)
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("residual PSD\n/ (noise + foreground)", fontsize=9)
    _cb = fig.colorbar(_sc, ax=ax[1], pad=0.012)
    _cb.set_label("log$_{10}$ p, Anderson-Darling", fontsize=8)
    _cb.ax.tick_params(labelsize=7)
    fig_b64(fig, "f1_resid")

    _gb_band = (_fcb >= 3e-3) & (_fcb <= 1e-2)
    _cr = _Ns / np.maximum(_Ni, 1e-60)
    _ci = int(np.nanargmax(np.where(_ok, _cr, np.nan)))
    DTR.update(
        rat_gal=float(np.nanmedian(_rat[_gb_band])),
        conf_ratio=float(_cr[_ci]), conf_f=float(_fcb[_ci]),
        undersub=int(np.nansum(_Pr[_ok] < _Ni[_ok])),
        # WHERE those bins are decides whether they mean anything. Below the
        # analysed band the windowed data is dominated by leakage from the
        # enormous sub-mHz TDI content -- a sinc pattern whose nulls dip under
        # any smooth noise curve while carrying no model at all. A bin under
        # the instrument curve THERE is a window artefact; one inside the
        # band, where templates are actually subtracted, would be real
        # over-subtraction. Report the split rather than asserting either.
        undersub_hi=float(np.nanmax(np.where(
            _ok & (_Pr < _Ni), _fcb, np.nan))) if np.any(
                _ok & (_Pr < _Ni)) else float("nan"),
        undersub_lo=float(np.nanmin(np.where(
            _ok & (_Pr < _Ni), _fcb, np.nan))) if np.any(
                _ok & (_Pr < _Ni)) else float("nan"),
        undersub_inband=int(np.nansum(
            _ok & (_Pr < _Ni) & (_fcb >= 3e-3))),
        # The deepest shortfall, as a RATIO. A residual that is 3% under a
        # noise curve carrying a 3.6% parameter bias is explained; one that
        # is 2x under it is not, and the two must not read the same.
        undersub_worst=float(np.nanmin(np.where(
            _ok & (_Pr < _Ni), _Pr / np.maximum(_Ni, 1e-60), np.nan)))
        if np.any(_ok & (_Pr < _Ni)) else float("nan"),
        nbins=int(_ok.sum()),
        adp_bad=float(np.nanmean(_adp[np.isfinite(_adp)] < -3.0))
        if np.isfinite(_adp).any() else float("nan"))
    del data_fd, resid_fd, _tmpl, _orb, _comp
except Exception as e:
    MISSING.append(
        f"data/template/residual panels unavailable: {type(e).__name__}: {e}")

# ======================= GB RECOVERY (the science block) ====================
# ONE FROZEN DENOMINATOR. ``gb_truth_3to21.npz`` holds every catalogue GB that
# survives the kappa_max amplitude prefilter, with its exact optimal SNR under
# the run's own fitted noise at the iteration it was built at and its full
# parameter vector in both the run's 9-column sampling basis and GBGPU's
# physical basis. Its detectable (SNR>7) subset is used as the denominator of
# EVERY recovery statement on this page. It is deliberately frozen:
# "detectable" moves as the foreground estimate drops, and a denominator that
# moves with the numerator cannot measure progress.
#
# Everything downstream is recomputed HERE, at the last stored iteration --
# none of it is read from the iteration-15 caches the earlier page was built
# on, which were three times smaller in model size.
SCI = {}
TRU = None
for _tp in (os.path.join(RUN_DIR, "gb_truth_3to21.npz"), "gb_truth_3to21.npz"):
    if os.path.exists(_tp):
        try:
            TRU = np.load(_tp)
        except Exception:
            TRU = None
        break

SCI_TOBS = 7776000.0
try:
    _a = dict(f["global_fit/domain_settings/args"].attrs)
    SCI_TOBS = float(_a["0"]) * float(_a["1"]) * float(_a["2"])
except Exception:
    pass
SCI_DF = 1.0 / SCI_TOBS

# THE BAND COMES FROM THE TRUTH SET (2026-08-23). It used to be hardcoded at
# 3-21.94 mHz here AND spelled out in three captions, while the truth npz
# already stamped a ``band`` field -- so widening the truth set to the run's
# real GB band (band_edges[0] = 0.5556 mHz, not 3 mHz) would have left the
# page quoting the old band over the new counts, which is worse than either.
# Read the stamp, with the old hardcoded pair as the fallback for a pre-stamp
# npz -- the same backward-compatible pattern as the ``tobs`` stamp below.
# Read BEFORE the Tobs guard, which sets TRU to None: FLO/FHI still label the
# population panels on a run whose recovery block is skipped.
FLO, FHI = 3e-3, 21.94e-3          # fallback: the pre-2026-08-23 band
if TRU is not None and "band" in getattr(TRU, "files", []):
    try:
        _tband = np.asarray(TRU["band"], float).reshape(-1)
        if _tband.size >= 2 and 0 < _tband[0] < _tband[1]:
            FLO, FHI = float(_tband[0]), float(_tband[1])
    except Exception:
        pass
BAND_TXT = f"{_mhz(FLO)}&ndash;{_mhz(FHI)} mHz"       # HTML captions
TOL_BINS = 2.0
NBAND = (FHI - FLO) / SCI_DF       # FD bins in the band

# Set the overlap-match caption fragments UP FRONT so they exist even when
# the recovery-section machinery below never runs (TRU missing, Tobs
# mismatch, or the RPHYS try-block failed). The values are overwritten in
# the RPHYS block below once MM is actually computed.
MATCH_CRIT_HTML = (
    f"&ldquo;Matched&rdquo; here = phase-maximised overlap "
    f"&ge; {MATCH_MM_THRESH:.2f} on top of a "
    f"{TOL_BINS:.0f}-df-bin f<sub>0</sub> pair "
    f"(threshold <code>GF_MONITOR_MATCH_MM</code>).")
MATCH_CRIT_TXT = f"matched = phase-max overlap >= {MATCH_MM_THRESH:.2f}"

# The truth set is tied to A Tobs (SNRs, and the FD bin the match tolerance is
# quoted in) -- not to THE 3-month one: ``build_truth.py`` now stamps the
# observation time it was built at into the npz. Compare against that stamp,
# defaulting to the 3-month value for the pre-stamp npz files, so a 1-year set
# passes on a 1-year run while a 3-month set on a 1-year run still refuses
# rather than silently comparing a 1-year model against 3-month detectability.
TRU_TOBS = 7776000.0
if TRU is not None and "tobs" in getattr(TRU, "files", []):
    try:
        TRU_TOBS = float(np.asarray(TRU["tobs"]).reshape(-1)[0])
    except Exception:
        TRU_TOBS = 7776000.0
if TRU is not None and abs(SCI_TOBS - TRU_TOBS) > 1.0:
    MISSING.append(
        "recovery panels skipped: the frozen detectability set is built for a "
        f"{TRU_TOBS / 86400.0:.4g}-day observation and this run is "
        f"{SCI_TOBS / 86400.0:.4g} days.")
    TRU = None


def _match_pairs(rf, tf, tol):
    """Globally-greedy ONE-TO-ONE nearest match in f0 within ``tol`` (Hz).

    Greedy on |df| ascending, so the result is independent of input order. A
    per-source nearest-neighbour match is not: it lets two model leaves claim
    the same injection, which inflates completeness by exactly the duplicate
    rate this page is trying to measure.
    """
    if rf.size == 0 or tf.size == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    o = np.argsort(tf)
    tfs = tf[o]
    lo = np.searchsorted(tfs, rf - tol)
    hi = np.searchsorted(tfs, rf + tol)
    ri, ti = [], []
    for i in range(rf.size):
        for j in o[lo[i]:hi[i]]:
            ri.append(i); ti.append(j)
    if not ri:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    ri = np.asarray(ri); ti = np.asarray(ti)
    d = rf[ri] - tf[ti]
    ur = np.zeros(rf.size, bool); ut = np.zeros(tf.size, bool)
    out = []
    for m in np.argsort(np.abs(d)):
        a_, b_ = ri[m], ti[m]
        if ur[a_] or ut[b_]:
            continue
        ur[a_] = ut[b_] = True
        out.append((a_, b_, d[m]))
    out.sort()
    return (np.array([p[0] for p in out], int),
            np.array([p[1] for p in out], int),
            np.array([p[2] for p in out], float))


def _wilson(k, n, z=1.0):
    """Wilson score interval (z=1 -> 68%); correct at k=0 and k=n, unlike the
    normal approximation, which is what a 0%-recovery bin needs."""
    k = np.asarray(k, float); n = np.maximum(np.asarray(n, float), 1e-9)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(np.maximum(p * (1 - p) / n + z * z / (4 * n * n), 0)) / den
    return c - h, c + h


def _survival(x):
    """UN-NORMALISED survival count: (sorted x, number of sources at or above).

    Unnormalised is the field convention (Littenberg Fig 13, Katz Fig 8): the
    y-intercept then reads directly as the size of the set.
    """
    s = np.sort(np.asarray(x, float))
    return s, np.arange(s.size, 0, -1)


def _nn_bins(fv):
    """Nearest-neighbour |Delta f0| in FD bins within one frequency set."""
    if fv.size < 2:
        return np.zeros(0)
    s = np.sort(fv)
    d = np.diff(s)
    return np.minimum(np.r_[d, np.inf], np.r_[np.inf, d]) / SCI_DF


def _leaf_f0(it_, w):
    """Alive-leaf f0 [Hz] for one cold walker at one stored iteration.

    Sliced column-wise: the chain is chunked (..., 1) on the parameter axis,
    so pulling only f0 reads ~1/9 of the bytes a full-row read would.
    """
    al = g["inds/gb"][it_, 0, 0, w]
    return (g["chain/gb"][it_, 0, 0, w, :, 1] * 1e-3)[al]


if TRU is not None:
    _sel = (TRU["det"] & (TRU["f0"] >= FLO) & (TRU["f0"] <= FHI))
    T_F0 = TRU["f0"][_sel]
    T_SNR = TRU["snr"][_sel]
    T_AMP = TRU["amp"][_sel]
    T_PHYS = TRU["phys"][_sel]
    NDET = int(_sel.sum())
    # Chance rate: the fraction of the band covered by the +-2-bin acceptance
    # windows of the truth set. Any purity number is only meaningful against
    # it, and it is the first thing a reviewer asks for.
    CHANCE = NDET * (2 * TOL_BINS * SCI_DF) / (FHI - FLO)
    _tol = TOL_BINS * SCI_DF

    # ---- per-iteration progress (the max-lnL cold walker OF EACH ITERATION) -
    _wb = np.argmax(ll[:NIT], axis=1).astype(int)
    n_all = np.zeros(NIT, int); n_band = np.zeros(NIT, int)
    n_match = np.zeros(NIT, int)
    for _i in range(NIT):
        _fv = _leaf_f0(_i, int(_wb[_i]))
        n_all[_i] = _fv.size
        _fv = _fv[(_fv >= FLO) & (_fv <= FHI)]
        n_band[_i] = _fv.size
        n_match[_i] = _match_pairs(_fv, T_F0, _tol)[0].size
    _nz = np.nonzero(n_all > 0)[0]
    IT0 = int(_nz[0]) if _nz.size else 0          # the GB-search origin
    NGBIT = NIT - 1 - IT0

    # ---- the last stored iteration, in full ------------------------------
    WB = int(_wb[-1])
    _alive = g["inds/gb"][NIT - 1, 0, 0, WB]
    REC9 = g["chain/gb"][NIT - 1, 0, 0, WB][_alive]
    _inb = (REC9[:, 1] * 1e-3 >= FLO) & (REC9[:, 1] * 1e-3 <= FHI)
    REC9 = REC9[_inb]
    MI, TI, DFH = _match_pairs(REC9[:, 1] * 1e-3, T_F0, _tol)
    MATCHED = np.zeros(REC9.shape[0], bool); MATCHED[MI] = True
    FOUND = np.zeros(NDET, bool); FOUND[TI] = True

    SCI.update(ndet=NDET, chance=CHANCE, it0=IT0, ngbit=NGBIT, walker=WB,
               n_all=int(n_all[-1]), n_band=int(REC9.shape[0]),
               n_match=int(MI.size),
               completeness=MI.size / NDET,
               purity=MI.size / max(REC9.shape[0], 1))

    # ---- waveform-level quantities ---------------------------------------
    # Optimal SNRs and template overlaps come from GBGPU's OWN run_wave and a
    # noise-weighted inner product against lisatools A2TDISens/E2TDISens fed
    # this run's sampled instrument + foreground -- the same route the run's
    # own likelihood uses. Nothing about the waveform or the noise is
    # re-derived here.
    RPHYS = None
    try:
        from gbgpu.gbgpu import GBGPU
        from lisatools import detector as lisa_models
        from lisatools.sensitivity import (get_sensitivity, A2TDISens,
                                           E2TDISens)
        from lisatools.stochastic import (
            HyperbolicTangentGalacticForeground as HTGF)
        from lisatools.globalfit.stock.erebor.transforms import (
            make_gb_transform_container)

        _psd_p = np.median(psd_cold[-1], axis=0)
        _gal_p = np.median(gal_cold_phys[-1], axis=0)
        _lm = lisa_models.LISAModel(_psd_p[0] ** 2, _psd_p[1] ** 2,
                                    lisa_models.DefaultOrbits(), "sampled")
        _nk = dict(model=_lm, stochastic_params=tuple(_gal_p),
                   stochastic_function=HTGF)
        # The noise is a pure function of the FD bin index, so evaluate it
        # ONCE on the whole grid: per-source calls would rebuild the model
        # 3,000 times for numbers that never change.
        _ng = int(2.35e-2 / SCI_DF) + 2
        _fg = np.maximum(np.arange(_ng) * SCI_DF, SCI_DF)
        SA_G = np.asarray(get_sensitivity(_fg, sens_fn=A2TDISens, **_nk), float)
        SE_G = np.asarray(get_sensitivity(_fg, sens_fn=E2TDISens, **_nk), float)

        # L1 orbits when available (mandatory for high-f accuracy); the
        # analytic fallback is fine below a few mHz only.
        _orb = globals().get("_L1_ORB_CPU")
        if _orb is None:
            MISSING.append(
                "SNR machinery fell back to DefaultOrbits (no mojito L1 "
                "orbit file reachable): SNRs/overlaps above ~5 mHz carry an "
                "annual-Doppler phase error.")
            _orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
        _gbw = GBGPU(force_backend="cpu", orbits=_orb, t0=float(T_REF_SCI))
        _tc = make_gb_transform_container(
            use_chirp_mass=True, use_fdot_astro=True, use_distance=True,
            mc_lims=(0.001, 1.0))
        RPHYS = _tc.both_transforms(np.asarray(REC9, float).copy())

        NW_ = 1024

        def _ae(phys):
            _gbw.run_wave(*[np.ascontiguousarray(phys[:, k]) for k in range(9)],
                          N=NW_, T=SCI_TOBS, dt=2.5, tdi2=True,
                          tdi_channel_setup="AE")
            return (np.asarray(_gbw.A), np.asarray(_gbw.E),
                    np.asarray(_gbw.start_inds).astype(int))

        _A, _E, _s = _ae(RPHYS)
        REC_SNR = np.zeros(RPHYS.shape[0])
        for _i in range(RPHYS.shape[0]):
            if _s[_i] < 0 or _s[_i] + NW_ > SA_G.size:
                continue
            _sa = SA_G[_s[_i]:_s[_i] + NW_]; _se = SE_G[_s[_i]:_s[_i] + NW_]
            REC_SNR[_i] = np.sqrt(max(4.0 * SCI_DF * float(np.sum(
                np.abs(_A[_i]) ** 2 / _sa + np.abs(_E[_i]) ** 2 / _se)), 0.0))

        # phase-maximised, noise-weighted overlap: |<a|b>| / sqrt(<a|a><b|b>).
        # The modulus IS the maximum over an overall phase, so no phase grid
        # is needed.
        _Ar, _Er, _sr = _ae(RPHYS[MI])
        _At, _Et, _st = _ae(T_PHYS[TI])
        MM = np.zeros(MI.size)
        for _i in range(MI.size):
            _o = min(_sr[_i], _st[_i])
            _sp = max(_sr[_i], _st[_i]) + NW_ - _o
            if _o < 0 or _o + _sp > SA_G.size:
                continue
            _sa = SA_G[_o:_o + _sp]; _se = SE_G[_o:_o + _sp]
            _a1 = np.zeros(_sp, complex); _e1 = np.zeros(_sp, complex)
            _a2 = np.zeros(_sp, complex); _e2 = np.zeros(_sp, complex)
            _k = _sr[_i] - _o; _a1[_k:_k + NW_] = _Ar[_i]; _e1[_k:_k + NW_] = _Er[_i]
            _k = _st[_i] - _o; _a2[_k:_k + NW_] = _At[_i]; _e2[_k:_k + NW_] = _Et[_i]
            _n = (np.sum(np.conj(_a1) * _a2 / _sa)
                  + np.sum(np.conj(_e1) * _e2 / _se))
            _n1 = (np.sum(np.abs(_a1) ** 2 / _sa)
                   + np.sum(np.abs(_e1) ** 2 / _se))
            _n2 = (np.sum(np.abs(_a2) ** 2 / _sa)
                   + np.sum(np.abs(_e2) ** 2 / _se))
            MM[_i] = float(np.abs(_n) / np.sqrt(max(_n1 * _n2, 1e-300)))
        MM = np.clip(MM, 0.0, 1.0)
        SCI.update(mm_med=float(np.median(MM)) if MM.size else float("nan"),
                   mm_hi=float(np.mean(MM > 0.9)) if MM.size else 0.0,
                   snr_med=float(np.median(REC_SNR)))
    except Exception as e:
        MISSING.append(f"waveform-level recovery statistics unavailable: "
                       f"{type(e).__name__}: {e}")
        RPHYS, REC_SNR, MM = None, None, np.zeros(0)

    # ---- overlap-refined match: the criterion the science panels use ----
    # Above, MI/TI/MATCHED/FOUND come from the 2-df-bin proxy match: a pair
    # is anything within TOL_BINS FD bins of f0. That is what the F2 progress
    # panel and completeness/purity KPIs use, because they run every stored
    # row and can't afford a waveform per pair. Here we tighten "matched" to
    # a real phase-maximised overlap threshold (MATCH_MM_THRESH, default
    # 0.8) for the panels that only need it at the last row -- F4 and the
    # zoomable plot. Everything upstream keeps the 2-df definition.
    if MM is not None and MM.size:
        _kept = np.asarray(MM, float) >= MATCH_MM_THRESH
        _MI_ov, _TI_ov = MI[_kept], TI[_kept]
        MATCHED_MM = np.zeros(REC9.shape[0], bool); MATCHED_MM[_MI_ov] = True
        FOUND_MM   = np.zeros(NDET, bool);           FOUND_MM[_TI_ov]  = True
        MATCH_CRIT_HTML = (
            f"&ldquo;Matched&rdquo; here = phase-maximised, noise-weighted "
            f"overlap &ge; {MATCH_MM_THRESH:.2f} on top of a "
            f"{TOL_BINS:.0f}-df-bin f<sub>0</sub> pair "
            f"(threshold <code>GF_MONITOR_MATCH_MM</code>).")
        MATCH_CRIT_TXT = f"matched = phase-max overlap >= {MATCH_MM_THRESH:.2f}"
    else:
        MATCHED_MM = MATCHED
        FOUND_MM   = FOUND
        MATCH_CRIT_HTML = (
            f"&ldquo;Matched&rdquo; here = within {TOL_BINS:.0f} "
            f"f<sub>0</sub> bins ONLY &mdash; the phase-maximised overlap "
            f"machinery was unavailable in this snapshot, so this panel "
            f"falls back to the 2-df proxy.")
        MATCH_CRIT_TXT = f"matched = within {TOL_BINS:.0f} df bins (2-df proxy)"

    # ---- cross-arm cache -------------------------------------------------
    # The v2/v3 comparison must be made on GB-SEARCH iterations, not absolute
    # ones (v2's first GB leaf lands at iteration 5, v3's at 16), so each arm
    # publishes its own series keyed by that origin and every comparison panel
    # subtracts it.
    # Every run kind gets its OWN arm cache (2026-08-22): the old
    # {v3, v4}-else-v2 map sent v5, v6 AND the 1-yr run all to
    # gf_arm_v2.npz, silently clobbering the shared v2 arm.
    ARM_TAG = {"3mo_v3": "v3", "3mo_v4": "v4", "3mo": "v2"}.get(
        RUN_KIND, RUN_KIND)
    try:
        np.savez(f"gf_arm_{ARM_TAG}.npz", n_all=n_all, n_band=n_band,
                 n_match=n_match, it0=IT0, ti=TI, mm=MM,
                 rec_f0=REC9[:, 1] * 1e-3, ndet=NDET)
    except Exception:
        pass
    ARMS = {}
    for _fn in sorted(glob.glob("gf_arm_*.npz")):
        _tag = os.path.basename(_fn)[7:-4]
        try:
            _z = np.load(_fn)
            # An arm cache stores TRUTH-SET INDICES (`ti`), so it is only
            # meaningful against the truth set it was matched to. A cache
            # built against a different denominator (e.g. the original
            # 812-source set vs a rebuilt one) would overlay silently WRONG
            # per-source recovery -- and index out of bounds when the sets
            # differ in size. `ndet` stamped in the cache is the guard.
            if "ndet" in _z and int(_z["ndet"]) != NDET:
                MISSING.append(
                    f"arm cache {_fn} skipped: built against a "
                    f"{int(_z['ndet'])}-source truth set, this page uses "
                    f"{NDET}. Regenerate it by rerunning this script on "
                    "that arm's run dir with the current truth npz.")
                continue
            ARMS[_tag] = _z
        except Exception:
            pass
    # Every arm needs its OWN hue (2026-08-22): the per-run arm tags
    # (gf_arm_3mo_v5 / _3mo_v6 / _1yr_v5) all fell through to the GREEN
    # default, drawing indistinguishable comparison lines. Known tags get
    # fixed hues; anything new draws from a deterministic cycle keyed by
    # sorted position so two unknown arms can never share a colour.
    ARM_COL = {
        "v2": CYAN, "v3": AMBER, "v4": GREEN,
        "3mo_v5": "#FF7BAC",   # pink
        "3mo_v6": VIOLET,
        "1yr_v5": "#F2E14C",   # yellow
    }
    _ARM_CYCLE = [c for c in ("#FF7BAC", VIOLET, "#F2E14C", "#7BE0FF",
                              "#C08B5C", "#8FE388", "#B0B7C3")
                  if c not in {ARM_COL[t] for t in ARMS if t in ARM_COL}]
    for _i, _tag in enumerate(sorted(t for t in ARMS if t not in ARM_COL)):
        ARM_COL[_tag] = _ARM_CYCLE[_i % max(len(_ARM_CYCLE), 1)]

    # ================= FIGURES ==========================================
    import matplotlib.colors as _mcol

    if SHOW_MATCH_STATS:
        # ---- F2: completeness AND purity vs GB-search iteration -------------
        fig, ax = plt.subplots(figsize=(11, 3.8))
        axr = ax.twinx(); axr.grid(False)
        for _tag in sorted(ARMS):
            _D = ARMS[_tag]; _c = ARM_COL.get(_tag, GREEN)
            _x = np.arange(_D["n_match"].size) - int(_D["it0"])
            _k = _x >= 0
            ax.plot(_x[_k], 100 * _D["n_match"][_k] / NDET, color=_c, lw=2.0,
                    label=f"{_tag} completeness")
            axr.plot(_x[_k], 100 * _D["n_match"][_k]
                     / np.maximum(_D["n_band"][_k], 1), color=_c, lw=1.2, ls="--")
        axr.axhline(100 * CHANCE, color=RED, ls=":", lw=1.2)
        axr.text(1, 100 * CHANCE, f" {100*CHANCE:.1f}% by chance", color=RED,
                 fontsize=8, va="bottom")
        ax.set_xlabel("iterations since the first galactic-binary leaf")
        ax.set_ylabel(f"completeness  [% of {NDET}]")
        axr.set_ylabel("purity  [% of leaves]  (dashed)", color=DIM, fontsize=9)
        axr.tick_params(axis="y", colors=DIM, labelsize=8)
        ax.set_ylim(0, 60); axr.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="lower right")
        fig_b64(fig, "f2_progress")

        # ---- F3: match CDF + survival COUNT ---------------------------------
        fig, ax = plt.subplots(2, 1, figsize=(9.6, 6.0), sharex=True,
                               gridspec_kw=dict(hspace=0.08))
        for _tag in sorted(ARMS):
            _D = ARMS[_tag]; _c = ARM_COL.get(_tag, GREEN)
            _m = np.sort(np.asarray(_D["mm"], float))
            if not _m.size:
                continue
            ax[0].plot(_m, np.arange(1, _m.size + 1) / _m.size, color=_c, lw=1.8,
                       label=f"{_tag}  ({_m.size} matched)")
            _s, _n = _survival(_m)
            ax[1].plot(_s, _n, color=_c, lw=1.8)
        ax[0].set_ylabel("cumulative fraction"); ax[0].set_ylim(0, 1)
        ax[0].legend(fontsize=8, loc="upper left")
        ax[1].axhline(NDET, color=FG, ls="--", lw=1.0)
        ax[1].text(0.02, NDET, f" {NDET} detectable injections", color=FG,
                   fontsize=8, va="bottom")
        ax[1].set_yscale("log"); ax[1].set_ylim(1, 2500); ax[1].set_xlim(0, 1)
        ax[1].set_xlabel("phase-maximised overlap with the matched injection")
        ax[1].set_ylabel("sources at or above")
        fig_b64(fig, "f3_match")

    if RPHYS is not None:
        _rf = RPHYS[:, 1]; _ra_ = RPHYS[:, 0]
        _fgr = np.logspace(np.log10(FLO), np.log10(FHI), 220)
        _athr = None
        for _kp in (os.path.join(RUN_DIR, "kappa_grid.npz"), "kappa_grid.npz"):
            if os.path.exists(_kp):
                _kg = np.load(_kp)
                _athr = 7.0 / np.interp(_fgr, _kg["fgrid"], _kg["fit"])
                break

        # ---- F4: amplitude vs frequency + the three-way recovery scatter --
        fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.5))
        a_ = ax[0]
        a_.scatter(T_F0, T_AMP, s=5, color=DIM, alpha=0.35, lw=0,
                   label="detectable injections")
        _sc = a_.scatter(_rf, _ra_, c=np.clip(REC_SNR, 4, None), s=14,
                         marker=".", cmap="cool", lw=0,
                         norm=_mcol.LogNorm(vmin=4, vmax=200),
                         label="resolved GBs")
        if _athr is not None:
            a_.plot(_fgr, _athr, color=FG, lw=1.5,
                    label="instrument sensitivity (SNR = 7)")
        a_.set_xscale("log"); a_.set_yscale("log")
        a_.set_xlabel("Frequency [Hz]"); a_.set_ylabel("Strain amplitude")
        a_.legend(fontsize=8, loc="lower left")
        _cb = fig.colorbar(_sc, ax=a_, pad=0.015); _cb.set_label("SNR", fontsize=8)
        _cb.ax.tick_params(labelsize=7)
        b_ = ax[1]
        if SHOW_MATCH_STATS:
            b_.scatter(T_F0[~FOUND_MM], T_AMP[~FOUND_MM], s=17, marker="x",
                       color=RED, lw=0.8, alpha=0.7,
                       label=f"detectable, not recovered ({int((~FOUND_MM).sum())})")
            b_.scatter(_rf[~MATCHED_MM], _ra_[~MATCHED_MM], s=22,
                       facecolors="none", edgecolors=VIOLET, lw=0.8, alpha=0.85,
                       label=f"recovered, no match ({int((~MATCHED_MM).sum())})")
            b_.scatter(_rf[MATCHED_MM], _ra_[MATCHED_MM], s=12, color=GREEN,
                       lw=0, alpha=0.95,
                       label=f"recovered and matched ({int(MATCHED_MM.sum())})")
        else:
            # Neutral overlay -- injections and model in one plane, the eye
            # does the comparison; no proxy-match classification.
            b_.scatter(T_F0, T_AMP, s=17, marker="x", color=DIM, lw=0.8,
                       alpha=0.6, label=f"detectable injections ({NDET})")
            b_.scatter(_rf, _ra_, s=12, color=GREEN, lw=0, alpha=0.9,
                       label=f"model sources ({_rf.size})")
        if _athr is not None:
            b_.plot(_fgr, _athr, color=FG, lw=1.5)
        b_.set_xscale("log"); b_.set_yscale("log")
        b_.set_xlabel("Frequency [Hz]"); b_.set_ylim(*a_.get_ylim())
        b_.legend(fontsize=8, loc="lower left")
        fig_b64(fig, "f4_pop")

        # ---- F8: sky map ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(11.5, 4.4))
        ax.scatter(np.mod(T_PHYS[:, 7], 2 * np.pi), T_PHYS[:, 8], s=4,
                   color=DIM, alpha=0.35, lw=0, label="detectable injections")
        _sc = ax.scatter(np.mod(RPHYS[:, 7], 2 * np.pi), RPHYS[:, 8],
                         c=np.clip(REC_SNR, 4, None), s=13, cmap="cool", lw=0,
                         norm=_mcol.LogNorm(vmin=4, vmax=200))
        ax.set_xlabel("right ascension [rad]")
        ax.set_ylabel("declination [rad]")
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(-1.6, 1.6)
        _cb = fig.colorbar(_sc, ax=ax, pad=0.012); _cb.set_label("SNR", fontsize=8)
        _cb.ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, loc="upper left")
        fig_b64(fig, "f8_sky")

        if SHOW_MATCH_STATS and len(RPHYS[MI]) == 0:
            # A store with no GB leaves yet (young run, noise stages only)
            # has ZERO matched sources; every panel below percentiles /
            # histograms empty arrays and np.percentile raises. Skip with a
            # reason instead of crashing the whole page (hit on the v7
            # first snapshot, 2026-08-26).
            MISSING.append("match-stats figures skipped: zero matched "
                           "sources in this snapshot (no gb leaves yet).")
        elif SHOW_MATCH_STATS:
            # ---- F7: recovered vs injected parameters --------------------------
            _R = RPHYS[MI]; _T = T_PHYS[TI]
            _db = DFH / SCI_DF
            _lnA = np.log(np.maximum(_R[:, 0], 1e-40) / np.maximum(_T[:, 0], 1e-40))
            _cc = (np.sin(_R[:, 8]) * np.sin(_T[:, 8])
                   + np.cos(_R[:, 8]) * np.cos(_T[:, 8])
                   * np.cos(_R[:, 7] - _T[:, 7]))
            _sep = np.degrees(np.arccos(np.clip(_cc, -1, 1)))
            fig, ax = plt.subplots(2, 3, figsize=(13.6, 6.0))
            ax[0][0].hist(np.clip(_db, -2, 2), bins=40, color=GREEN, alpha=0.9)
            ax[0][0].axvline(0, color=FG, ls="--", lw=1)
            ax[0][0].set_xlabel(r"$\Delta f_0$  [FD bins]")
            ax[0][1].hist(np.clip(_lnA, -2, 2), bins=40, color=GREEN, alpha=0.9)
            ax[0][1].axvline(0, color=FG, ls="--", lw=1)
            ax[0][1].set_xlabel(r"$\ln(A_{\rm rec}/A_{\rm cat})$")
            ax[0][2].hist(_sep, bins=40, color=GREEN, alpha=0.9)
            ax[0][2].set_xlabel("sky separation [deg]")
            for _k in range(3):
                ax[0][_k].set_ylabel("matched sources", fontsize=9)
            # BOTTOM ROW: error histograms centred on zero, NOT a
            # recovered-vs-injected scatter with a y = x diagonal -- that idiom
            # does not appear anywhere in this literature. Zero-centred residual
            # histograms are what Littenberg 2011 Fig 6 / Strub 2403.15318 Fig 5
            # use, and they put the bias and the spread on the same axis.
            _dpsi = (_R[:, 6] - _T[:, 6] + np.pi / 4) % (np.pi / 2) - np.pi / 4
            _panels = [
                (np.cos(_R[:, 5]) - np.cos(_T[:, 5]), r"$\Delta$ cos $\iota$", 2.0),
                (_dpsi, r"$\Delta\psi$ wrapped to $\pi/2$  [rad]", np.pi / 4),
                ((_R[:, 2] - _T[:, 2]) * 1e16,
                 r"$\Delta\dot f_0$  [$10^{-16}$ Hz/s]", None)]
            for _ax, (_dv, _lb, _cl) in zip(ax[1], _panels):
                if _cl is None:
                    _cl = float(np.percentile(np.abs(_dv), 90)) or 1.0
                _ax.hist(np.clip(_dv, -_cl, _cl), bins=40, color=GREEN, alpha=0.9)
                _ax.axvline(0, color=FG, ls="--", lw=1)
                _ax.set_xlabel(_lb)
                _ax.set_ylabel("matched sources", fontsize=9)
            fig.tight_layout()
            fig_b64(fig, "f7_params")
            # A SECOND encoding of the same matched set, kept deliberately even
            # though this idiom does not appear in the LISA galactic-binary
            # literature (which uses zero-centred error histograms, above). It
            # answers a different question: the histogram shows the size of the
            # error, the diagonal shows whether the parameter is being RECOVERED
            # at all, i.e. whether the points know about the injected value or
            # merely scatter over the prior. For cos iota at these signal
            # strengths those are visibly different statements.
            fig, ax2 = plt.subplots(1, 3, figsize=(13.6, 3.5))
            _sc_panels = [
                (np.cos(_R[:, 5]), np.cos(_T[:, 5]), r"cos $\iota$", None),
                (_R[:, 6] % (np.pi / 2), _T[:, 6] % (np.pi / 2),
                 r"$\psi$ mod $\pi/2$  [rad]", None),
                (_R[:, 2] * 1e16, _T[:, 2] * 1e16,
                 r"$\dot f_0$  [$10^{-16}$ Hz/s]", 98.0)]
            for _ax, (_rv, _tv, _lb, _q) in zip(ax2, _sc_panels):
                _ax.scatter(_tv, _rv, s=10, color=GREEN, alpha=0.6, lw=0)
                if _q is None:
                    _l2 = min(np.min(_tv), np.min(_rv))
                    _h2 = max(np.max(_tv), np.max(_rv))
                else:
                    _l2, _h2 = np.percentile(_tv, [100 - _q, _q])
                    _pd2 = 0.8 * (_h2 - _l2); _l2 -= _pd2; _h2 += _pd2
                _ax.plot([_l2, _h2], [_l2, _h2], color=FG, ls="--", lw=1.1)
                _ax.set_xlim(_l2, _h2); _ax.set_ylim(_l2, _h2)
                _ax.set_xlabel("injected  " + _lb)
                _ax.set_ylabel("recovered", fontsize=9)
            fig.tight_layout()
            fig_b64(fig, "f7_scatter")
            SCI["ci_corr"] = float(np.corrcoef(np.cos(_R[:, 5]),
                                               np.cos(_T[:, 5]))[0, 1])
            SCI.update(df_tight=float(np.mean(np.abs(_db) < 0.5)),
                       lnA_med=float(np.median(np.abs(_lnA))),
                       sep_med=float(np.median(_sep)),
                       sep_tight=float(np.mean(_sep < 10.0)))

    # ---- F5: source counts vs frequency --------------------------------
    _f0_cat = None
    try:
        with h5py.File(os.path.join(
                MOJITO_CAT_DIR, "catalogues",
                "wdwd_cat_mojito_lite_processed.hdf5"), "r") as _wf:
            _f0_cat = _wf["Binaries"]["GW22FrequencySSBFrame"][:]
    except Exception:
        _f0_cat = None
    _eds = np.logspace(np.log10(FLO), np.log10(FHI), 26)
    _hdet, _ = np.histogram(T_F0, bins=_eds)
    _hrec, _ = np.histogram(T_F0[FOUND], bins=_eds)
    _fc = np.sqrt(_eds[:-1] * _eds[1:])
    if SHOW_MATCH_STATS:
        fig, ax = plt.subplots(2, 1, figsize=(10.6, 5.3), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1],
                                                hspace=0.08))
        ax0, ax1 = ax[0], ax[1]
    else:
        # Match-free variant: density comparison only (model leaves vs the
        # truth populations), no recovered-fraction panel.
        fig, ax0 = plt.subplots(figsize=(10.6, 3.9))
        ax1 = None
    if _f0_cat is not None:
        _m = (_f0_cat >= FLO) & (_f0_cat <= FHI)
        _hall, _ = np.histogram(_f0_cat[_m], bins=_eds)
        ax0.stairs(_hall, _eds, color=DIM, fill=True, alpha=0.4,
                   label=f"all catalogue ({int(_m.sum()):,})")
        SCI["n_cat_band"] = int(_m.sum())
    ax0.stairs(_hdet, _eds, color=CYAN, lw=1.7, label=f"detectable ({NDET})")
    if SHOW_MATCH_STATS:
        ax0.stairs(_hrec, _eds, color=GREEN, fill=True, alpha=0.9,
                   label=f"recovered ({int(FOUND.sum())})")
    else:
        # F5 sits OUTSIDE the ``if RPHYS is not None`` block that defines
        # ``_rf = RPHYS[:, 1]`` for the F4 scatter, so when the RPHYS build
        # try-block raises (unreachable orbits, missing waveform backend,
        # zero recovered sources with an "empty slice" warning cascade,
        # etc.) this branch used to hit ``NameError: name '_rf' is not
        # defined``. Rebuild from REC9 -- always populated before F5 --
        # instead of chaining off the F4 side-effect.
        _rf_hz = REC9[:, 1] * 1e-3
        _hmod, _ = np.histogram(_rf_hz, bins=_eds)
        ax0.stairs(_hmod, _eds, color=GREEN, fill=True, alpha=0.9,
                   label=f"model sources ({int(_rf_hz.size)})")
    ax0.set_yscale("log"); ax0.set_ylim(0.5, None)
    ax0.set_ylabel("sources per bin"); ax0.legend(fontsize=8)
    if ax1 is not None:
        _lo, _hi = _wilson(_hrec, _hdet)
        _fr = np.where(_hdet > 0, _hrec / np.maximum(_hdet, 1), np.nan)
        # maximum(.., 0): at p=0 (a zero-recovery bin -- the full band has
        # them, the old 3-21 mHz band never did) the Wilson lower bound
        # equals p algebraically but can land ~1e-17 ABOVE it in floating
        # point, and errorbar refuses negative yerr.
        ax1.errorbar(_fc, 100 * _fr,
                     yerr=[100 * np.maximum(_fr - _lo, 0),
                           100 * np.maximum(_hi - _fr, 0)], fmt="o",
                     ms=3.5, color=GREEN, ecolor=GREEN, alpha=0.9, capsize=2)
        ax1.set_xscale("log"); ax1.set_ylim(0, 100)
        ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("recovered [%]")
    else:
        ax0.set_xscale("log"); ax0.set_xlabel("Frequency [Hz]")
    fig_b64(fig, "f5_counts")

    if SHOW_MATCH_STATS:
        # ---- F6: completeness vs SNR, with Wilson intervals ------------------
        _seds = np.array([7, 10, 15, 25, 1e9])
        _lab = ["7-10", "10-15", "15-25", "25+"]
        fig, ax = plt.subplots(figsize=(9.0, 3.7))
        _tags = sorted(ARMS)
        for _q, _tag in enumerate(_tags):
            _D = ARMS[_tag]; _c = ARM_COL.get(_tag, GREEN)
            _fnd = np.zeros(NDET, bool); _fnd[np.asarray(_D["ti"], int)] = True
            _x, _y, _el, _eh = [], [], [], []
            for _j, (_a2, _b2) in enumerate(zip(_seds[:-1], _seds[1:])):
                _m = (T_SNR >= _a2) & (T_SNR < _b2)
                if not _m.sum():
                    continue
                _p = _fnd[_m].mean()
                _l, _h = _wilson(_fnd[_m].sum(), _m.sum())
                _x.append(_j + (_q - (len(_tags) - 1) / 2) * 0.10)
                _y.append(100 * _p); _el.append(100 * (_p - _l))
                _eh.append(100 * (_h - _p))
            if not _x:
                continue
            _n = int(_D["n_match"][-1])
            _g = int(_D["n_match"].size) - 1 - int(_D["it0"])
            # An arm from a zero-match store (young run) can produce
            # degenerate Wilson bounds; matplotlib refuses negative yerr.
            _el = np.clip(_el, 0.0, None)
            _eh = np.clip(_eh, 0.0, None)
            ax.errorbar(_x, _y, yerr=[_el, _eh], fmt="o-", ms=5, color=_c, lw=1.6,
                        capsize=3, label=f"{_tag}, {_g} GB-search iterations")
        for _j, (_a2, _b2) in enumerate(zip(_seds[:-1], _seds[1:])):
            _m = (T_SNR >= _a2) & (T_SNR < _b2)
            ax.text(_j, 3, f"n={int(_m.sum())}", ha="center", color=DIM, fontsize=8)
        ax.set_xticks(range(len(_lab))); ax.set_xticklabels(_lab)
        ax.set_xlabel("optimal SNR of the injection")
        ax.set_ylabel("recovered [%]"); ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="upper left")
        fig_b64(fig, "f6_snr")

    # ---- F10: nearest-neighbour separation survival ----------------------
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    for _tag in sorted(ARMS):
        _D = ARMS[_tag]; _c = ARM_COL.get(_tag, GREEN)
        _s, _n = _survival(_nn_bins(np.asarray(_D["rec_f0"], float)))
        ax.plot(np.maximum(_s, 1e-2), _n, color=_c, lw=1.8,
                label=f"{_tag} model leaves")
    _s, _n = _survival(_nn_bins(T_F0))
    ax.plot(np.maximum(_s, 1e-2), _n, color=FG, lw=1.4, ls="--",
            label=f"detectable injections ({NDET})")
    ax.axvline(TOL_BINS, color=RED, ls=":", lw=1.2)
    ax.text(TOL_BINS * 1.1, 1.5, " match tolerance", color=RED, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-2, 1e4); ax.set_ylim(1, 2000)
    ax.set_xlabel(r"nearest-neighbour $|\Delta f_0|$  [FD bins]")
    ax.set_ylabel("sources at or above")
    ax.legend(fontsize=8, loc="lower left")
    fig_b64(fig, "f10_nn")
    _nn = _nn_bins(REC9[:, 1] * 1e-3)
    SCI.update(nn_half=float(np.mean(_nn < 0.5)) if _nn.size else 0.0,
               nn_tol=float(np.mean(_nn < TOL_BINS)) if _nn.size else 0.0,
               nn_half_t=float(np.mean(_nn_bins(T_F0) < 0.5)))
    # unmatched leaves sitting on an injection another leaf already claimed --
    # the "two templates, one source" blending mode, counted rather than
    # inferred from a duplicate percentage.
    if TI.size and int((~MATCHED).sum()):
        _d = np.abs((REC9[~MATCHED, 1] * 1e-3)[:, None]
                    - T_F0[TI][None, :]).min(axis=1) / SCI_DF
        SCI["blend_b"] = int((_d <= TOL_BINS).sum())
        SCI["n_unmatched"] = int((~MATCHED).sum())


# ---- F9: verification binaries, restricted to the ones that are there -----
# At three months only ELEVEN of the 55 catalogue verification binaries clear
# SNR 7; the median VGB optimal SNR is 1.6. Showing all 55 posteriors, as the
# page used to, showed 44 prior-dominated distributions next to 11 real ones
# with nothing distinguishing them -- a reader inevitably read the flat ones
# as failures of the fit rather than as an absence of signal. The headline
# panel is therefore the detectable subset; the rest are stated, not plotted.
# POOLED (2026-08-27): was a hard-coded 3-iteration window; it is now the
# shared POOL_ITS_SAMPLES window, so every VGB sample panel below (the
# detectable-subset credible intervals, the full-55 distance panel, the
# pooled marginals and the zoomable posterior cloud) draws the same rows.
VGB_SAMP_ROWS = _vgb_pool_rows(POOL_ITS_SAMPLES)
VGB_SAMP_ITS = int(VGB_SAMP_ROWS.size)
vgb_last = vgb_c[VGB_SAMP_ROWS].reshape(-1, 55, _vgb_chain_ndim)  # (S, 55, 5 or 6)
# ``vgb_hh`` gets trimmed to SUB_NIT rows above, and SUB_NIT can land at 0
# on a very young store or one whose mid-flush left the whole VGB column
# unwritten. ``vgb_hh[-1]`` then raises IndexError and the whole page dies.
# Same for the all-NaN case: nanmean on all-NaN returns NaN with a warning,
# but we don't want a NaN cascade downstream. Fall back to a zero-SNR
# vector so ``VGB_DET`` is empty and the panel renders with no bars.
_nv = int(vgb_hh.shape[-1]) if vgb_hh.ndim >= 1 else 55
try:
    if vgb_hh.shape[0] == 0:
        raise ValueError("no VGB h_h rows retained (SUB_NIT trimmed to 0)")
    with np.errstate(invalid="ignore"):
        snr = np.sqrt(np.clip(np.nanmean(vgb_hh[-1], axis=0), 0, None))
    if snr.ndim != 1 or snr.shape[0] != _nv:
        raise ValueError(f"unexpected SNR shape {snr.shape}")
    snr = np.where(np.isfinite(snr), snr, 0.0)
except Exception as _e:
    snr = np.zeros(_nv)
    MISSING.append(f"VGB SNR>7 panel: could not compute per-VGB SNRs "
                   f"({type(_e).__name__}: {_e}); the detectable-subset "
                   f"strip shows as empty (VGB branch not populated yet, "
                   f"or vgb_hh malformed in this snapshot).")
order = np.argsort(snr)[::-1]
VGB_DET = np.nonzero(snr > 7.0)[0]
VGB_DET = VGB_DET[np.argsort(snr[VGB_DET])[::-1]]
VGB_N_DET = int(VGB_DET.size)
# ``vgb_last`` can be shape (0, 55, 5) on the same young / mid-flush
# snapshots the vgb_hh guard above already handles. np.median on an empty
# slice merely warns and returns NaN, but np.percentile crashes inside
# ``_quantile`` (arr[-1, ...] on a size-0 axis). Fall back to NaN arrays so
# the panel still lays out with empty errorbars.
if vgb_last.shape[0] == 0:
    _shp = vgb_last.shape[1]
    med = np.full(_shp, np.nan)
    lo = np.full(_shp, np.nan)
    hi = np.full(_shp, np.nan)
else:
    with np.errstate(invalid="ignore"):
        med = np.median(vgb_last[:, :, 0], axis=0)
        lo = np.percentile(vgb_last[:, :, 0], 16, axis=0)
        hi = np.percentile(vgb_last[:, :, 0], 84, axis=0)

_lab = [(VGB_IDS[k] if VGB_IDS else f"leaf {k}") for k in VGB_DET]
_y = np.arange(VGB_N_DET)[::-1]
fig, ax = plt.subplots(1, 2, figsize=(12.4, 0.34 * max(VGB_N_DET, 8) + 1.6),
                       sharey=True, gridspec_kw=dict(width_ratios=[2.2, 1]))
ax[0].errorbar(med[VGB_DET], _y,
               xerr=[med[VGB_DET] - lo[VGB_DET], hi[VGB_DET] - med[VGB_DET]],
               fmt="o", ms=4, color=GREEN, ecolor=GREEN, capsize=2, lw=1.2)
if VGB_TRUTH is not None:
    ax[0].plot(VGB_TRUTH[VGB_DET, 0], _y, "|", ms=13, mew=1.8, color=CYAN,
               ls="none", label="catalogue distance")
    ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_yticks(_y); ax[0].set_yticklabels(_lab, fontsize=8)
ax[0].set_xlabel("distance [kpc]")
ax[0].set_title(f"the {VGB_N_DET} verification binaries with SNR > 7  "
                f"(median +/- 1 sigma, last {VGB_SAMP_ITS} its x {nwalk} "
                f"walkers)", fontsize=10)
ax[1].barh(_y, snr[VGB_DET], color=VIOLET, alpha=0.9, height=0.55)
ax[1].axvline(7, color=RED, ls=":", lw=1.2)
ax[1].set_xscale("log"); ax[1].set_xlabel("optimal SNR")
ax[1].set_title("SNR against the fitted noise", fontsize=10)
if VGB_F0 is not None:
    for _i, _k in enumerate(VGB_DET):
        ax[1].text(snr[_k] * 1.06, _y[_i], f" {VGB_F0[_k]:.2f} mHz",
                   fontsize=7, color=DIM, va="center")
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": LINE,
    "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.5,
    "font.size": 10, "font.family": "monospace", "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110,
    "text.usetex": _USETEX,
})
fig.tight_layout()
fig_b64(fig, "f9_vgb")

# SNR against frequency for ALL 55, with the detection threshold, so the
# "44 are prior-dominated" statement is visible rather than asserted.
fig, ax = plt.subplots(figsize=(11, 3.2))
_x = VGB_F0 if VGB_F0 is not None else np.arange(55).astype(float)
ax.plot(_x[snr <= 7], snr[snr <= 7], "o", ms=4, color=DIM, alpha=0.75,
        label=f"prior-dominated ({55 - VGB_N_DET})")
ax.plot(_x[snr > 7], snr[snr > 7], "o", ms=6, color=GREEN,
        label=f"SNR > 7 ({VGB_N_DET})")
ax.axhline(7, color=RED, ls=":", lw=1.2)
if VGB_F0 is not None:
    ax.set_xscale("log")
    ax.set_xlabel("catalogue f0 [mHz]")
else:
    ax.set_xlabel("VGB leaf index")
ax.set_yscale("log"); ax.set_ylabel("optimal SNR")
ax.legend(fontsize=8, loc="upper left")
fig_b64(fig, "f9_vgb_snr")
VGB_SNR_MED = float(np.median(snr))

# ---- RESTORED: the full 55-leaf VGB panels ---------------------------
if VGB_F0 is not None:
    xs, xlab_vgb = VGB_F0, "catalogue f0 [mHz]"
else:
    xs, xlab_vgb = np.arange(55), "VGB leaf index"
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.errorbar(xs, med, yerr=[med - lo, hi - med],
            fmt="o", ms=3, color=VIOLET, ecolor=VIOLET, alpha=0.9, capsize=2)
if VGB_TRUTH is not None:
    ax.plot(xs, VGB_TRUTH[:, 0], "_", ms=9, mew=1.4, color=RED, ls="none",
            label="catalogue truth")
    ax.legend(fontsize=8)
if VGB_F0 is not None:
    ax.set_xscale("log")
    for k in order[:3]:
        ax.annotate(VGB_IDS[k], (xs[k], med[k]), fontsize=7, color=FG,
                    xytext=(2, 6), textcoords="offset points")
ax.set_xlabel(xlab_vgb); ax.set_ylabel("dist [kpc]")
ax.set_title(f"VGB distance posteriors (median +/- 1 sigma; last "
             f"{VGB_SAMP_ITS} its x {nwalk} walkers)")
fig_b64(fig, "vgb_dist")

# SNR evolution over iterations: noise-weighted sqrt(<h|h>) rises as the
# galactic foreground is fit/subtracted down -- the source-side twin of
# the PSD decline-watch panel.
snr_it = np.sqrt(np.clip(np.nanmean(vgb_hh[:VGB_NIT], axis=1), 0, None))  # (it, 55)
import matplotlib.colors as _mc
_vramp = _mc.LinearSegmentedColormap.from_list(
    "violet", ["#E3D9FF", "#9B7BFF", "#4A2FA8"])
fig, ax = plt.subplots(figsize=(12, 3.4))
if VGB_F0 is not None:
    _of = np.argsort(VGB_F0)
    _xs = VGB_F0[_of]
    ax.set_xscale("log")
else:
    _of = np.arange(55)
    _xs = np.arange(55)
# markers only (user request 2026-08-15): connecting lines between
# unrelated VGBs on a frequency axis implied a spectrum that isn't there
for k in range(VGB_NIT):
    ax.plot(_xs, snr_it[k][_of], "o", ms=3.0, ls="none",
            color=_vramp(k / max(VGB_NIT - 1, 1)), alpha=0.85,
            label=(f"iter {k}" if k in (0, VGB_NIT - 1) else None))
if VGB_F0 is not None:
    for kk in order[:3]:
        ax.annotate(VGB_IDS[kk], (VGB_F0[kk], snr[kk]), fontsize=7,
                    color=FG, xytext=(2, 4), textcoords="offset points")
ax.set_yscale("log"); ax.set_xlabel(xlab_vgb)
ax.set_ylabel("sqrt(<h|h>)")
ax.legend(fontsize=8)
ax.set_title("VGB optimal SNR per stored iteration (light -> dark = later; "
             "watch these RISE as the foreground comes down)")
fig_b64(fig, "vgb_snr")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.0))
for k in range(3):
    leaf = order[k]
    nm = VGB_IDS[leaf] if VGB_IDS else f"leaf {leaf}"
    for w in range(nwalk):
        ax[k].plot(np.arange(VGB_NIT), vgb_c[:, w, leaf, 0],
                   color=VIOLET, alpha=0.35, lw=0.8)
    if VGB_TRUTH is not None:
        ax[k].axhline(VGB_TRUTH[leaf, 0], color=RED, lw=1.4, ls=":",
                      label="catalogue truth")
        ax[k].legend(fontsize=7)
    _f0txt = f", f0={VGB_F0[leaf]:.3f} mHz" if VGB_F0 is not None else ""
    ax[k].set_title(f"{nm} (SNR~{snr[leaf]:.0f}{_f0txt}) dist", fontsize=9)
    ax[k].set_xlabel("iter")
fig_b64(fig, "vgb_traces")

# Pooled over all 55 leaves, so the truth is a DISTRIBUTION, not a line: a
# red step histogram of the 55 catalogue values on the same axis (scaled to
# the posterior's peak). fdot_astro_ratio is the exception -- every truth is
# identically 0 (GR-chirp binaries), so a single dotted line is the honest
# overlay there.
# Non-distance marginals: cols 1..(ndim-1), so 4 panels on the legacy
# 5-param basis and 5 on the chirp 6-param basis.
_marg_cols = list(range(1, _vgb_chain_ndim))
fig, ax = plt.subplots(1, len(_marg_cols), figsize=(13, 2.8))
if len(_marg_cols) == 1:
    ax = [ax]
for _panel_i, j in enumerate(_marg_cols):
    a = ax[_panel_i]
    n_, edges_, _ = a.hist(vgb_last[:, :, j].ravel(), bins=30, color=VIOLET,
                           alpha=0.85)
    if VGB_TRUTH is not None:
        tv = VGB_TRUTH[:, j]
        if VGB_NAMES[j] == "fdot_astro_ratio":
            a.axvline(0.0, color=RED, lw=1.4, ls=":", label="truth = 0 (GR)")
        else:
            tn, te = np.histogram(tv, bins=edges_)
            a.step(te[:-1], tn * (n_.max() / max(tn.max(), 1)), where="post",
                   color=RED, lw=1.2, ls=":", label="catalogue truths (55)")
            a.plot(tv, np.full(tv.shape, -0.03 * n_.max()), "|", ms=6,
                   color=RED, alpha=0.8, clip_on=False)
        a.legend(fontsize=7)
    a.set_title(VGB_NAMES[j], fontsize=9)
fig.suptitle(f"VGB marginals pooled over 55 leaves x the last "
             f"{VGB_SAMP_ITS} stored iterations x {nwalk} cold walkers "
             f"({vgb_last.shape[0] * 55} samples)", fontsize=9, y=1.02)
fig_b64(fig, "vgb_hists")

# GB/VGB explorer data (interactive)
#
# POOLED (2026-08-27): this canvas -- the zoomable amplitude-frequency plane,
# including its "highest-frequency sources" preset -- used to draw the single
# latest stored iteration. It now pools POOL_ITS_SAMPLES iterations, with the
# alive mask taken from EACH iteration's own inds row.
expl = {"gb": [], "vgb": []}
EXPL_ITS, EXPL_RAW, EXPL_STRIDE = 0, 0, 1
nz = np.nonzero(gb_alive_last.sum(axis=0))[0]
if nz.size:
    # f0-AMPLITUDE (user request 2026-08-15): amplitude derived from the
    # sampled (dist, f0, Mc) via the stock transform; plotted as log10(A).
    try:
        from lisatools.globalfit.stock.erebor.transforms import gb_amp_from_dist
    except Exception:
        gb_amp_from_dist = None
    _gbstack = []
    for _it_, _al_, _ch_ in _pool_gb_iter(POOL_ITS_SAMPLES, cols=(0, 1, 2)):
        EXPL_ITS += 1
        for w in range(nwalk):
            al = np.nonzero(_al_[w])[0]
            if not al.size:
                continue
            _d = np.maximum(np.asarray(_ch_[w, al, 0], dtype=float), 1e-6)
            _f = np.asarray(_ch_[w, al, 1], dtype=float)
            _mc = np.asarray(_ch_[w, al, 2], dtype=float)
            if gb_amp_from_dist is not None:
                _amp = np.asarray(gb_amp_from_dist(_f * 1e-3, _mc, _d),
                                  dtype=float)
                _y = np.log10(np.maximum(_amp, 1e-30))
            else:
                _y = 1.0 / _d
            _gbstack.append(np.column_stack(
                [_f, _y, np.full(al.size, float(w))]))
    _R = (np.concatenate(_gbstack) if _gbstack
          else np.zeros((0, 3), dtype=float))
    EXPL_RAW = int(_R.shape[0])
    # PAGE-WEIGHT GUARD. Pooling multiplies this array by the window, and
    # every row is JSON text in the page. On the 3-month store the pooled
    # cloud is ~47k rows (~1 MB of JSON) and needs no decimation at all; a
    # denser store would, so the budget is enforced here -- AFTER pooling,
    # exactly like the truth-cross decimation below, and by striding whole
    # rows so the f0 coverage stays uniform. The truth-cross rules
    # (recovery-proximity protection + per-window floor) are untouched.
    EXPL_CAP = 150000
    if EXPL_RAW > EXPL_CAP:
        EXPL_STRIDE = int(np.ceil(EXPL_RAW / EXPL_CAP))
        _R = _R[::EXPL_STRIDE]
    # Coordinates are ROUNDED before serialization: f0 to 1e-7 mHz (about
    # 1e-3 of an FD bin at three months) and log10 A to 1e-4 dex. Both are
    # far below anything the canvas or the match tolerance can resolve, and
    # they cut the JSON from ~40 to ~23 bytes a row, which is what keeps the
    # pooled cloud from ballooning the page.
    expl["gb"] = [[round(float(_a), 7), round(float(_b), 4), int(_c)]
                  for _a, _b, _c in _R]
    del _gbstack, _R
expl["gb_its"] = int(EXPL_ITS)
expl["gb_raw"] = int(EXPL_RAW)
expl["gb_stride"] = int(EXPL_STRIDE)
EXPL_DEC_NOTE = ("" if EXPL_STRIDE <= 1 else
                 f", decimated 1-in-{EXPL_STRIDE} to {len(expl['gb']):,} "
                 f"drawn for page weight")
print(f"[explorer] GB cloud: {EXPL_RAW} alive-leaf rows pooled over "
      f"{EXPL_ITS} stored iteration(s) x {nwalk} cold walkers; "
      f"{len(expl['gb'])} drawn (stride {EXPL_STRIDE}).")
# The VGB side of the explorer pools the same window (fixed-dimension branch,
# so every leaf contributes one row per pooled iteration per walker).
_Spool = vgb_c[VGB_SAMP_ROWS]                        # (P, 24, 55, 5)
for _p in range(_Spool.shape[0]):
    S = _Spool[_p]                                   # (24, 55, 5)
    for w in range(nwalk):
        for leaf in range(55):
            _x = float(VGB_F0[leaf]) if VGB_F0 is not None else int(leaf)
            expl["vgb"].append([_x, float(1.0 / max(S[w, leaf, 0], 1e-6)),
                                float(snr[leaf])])
expl["vgb_its"] = int(VGB_SAMP_ITS)
expl["vgb_axis"] = "f0 [mHz] (catalogue)" if VGB_F0 is not None else "VGB leaf index"
expl["nwalk"] = int(nwalk)

# ---- GB catalogue truth cloud for the explorer (Task 3) -------------------
# The key science overlay: recovered (f0, log10 A) cloud vs the injected
# catalogue in the SAME plane reads completeness / faint tail directly.
# The 2.3 GB wdwd catalogue is opened lazily -- only the two columns needed
# for the cloud are ever pulled into memory in full.
# The band is the RUN's own gb band structure (sub_backend/gb/band_edges),
# never a hard-coded pair.
WDWD_PATH = os.path.join(MOJITO_CAT_DIR, "catalogues",
                         "wdwd_cat_mojito_lite_processed.hdf5")
TRUTH_CAP = 30000
gb_truth_pts, gb_truth_meta = [], {}
gb_truth_pts_grey, gb_truth_pts_red = [], []
cf = f0_band = cat_gidx = None
try:
    _wd = h5py.File(WDWD_PATH, "r")
    cf = _wd["Binaries"]
    _f0_all = cf["GW22FrequencySSBFrame"][:]
    cat_gidx = np.nonzero(
        (_f0_all >= band_edges[0]) & (_f0_all <= band_edges[-1]))[0]
    f0_band_hz = _f0_all[cat_gidx].astype(float, copy=True)   # Hz copy for
                                                              # bit-exact lookups
    f0_band = f0_band_hz * 1e3                    # mHz, in-band catalogue
    del _f0_all
    la_band = np.log10(np.maximum(cf["Amplitude"][:][cat_gidx], 1e-30))
    gb_truth_meta["in_band"] = int(cat_gidx.size)

    # DETECTABILITY per catalogue row (user ask 2026-09-19): the overlay used
    # to draw every catalogue source in one red, which made a cross under a
    # recovered dot indistinguishable from a cross the run could never have
    # found. Detectable stays red; everything below threshold goes grey.
    #
    # Matched on f0 against DET_F0 -- the same detectable-f0 list the leaf
    # count and occupancy overlays already use, carrying its own Tobs guard
    # (an npz built for a different observation time never reaches here).
    # Both arrays are the SAME catalogue column, so the only discrepancy is
    # float round-trip through the npz and the *1e3 to mHz; a relative
    # tolerance well below the catalogue's own f0 spacing settles it without
    # ever matching a neighbouring source.
    det_band = np.ones(f0_band.size, dtype=bool)
    if DET_F0 is not None and DET_F0.size:
        _dref = np.sort(np.asarray(DET_F0, dtype=float))
        _fq = f0_band * 1e-3                       # back to Hz for the match
        _j = np.clip(np.searchsorted(_dref, _fq), 0, _dref.size - 1)
        _jm = np.clip(_j - 1, 0, _dref.size - 1)
        _near = np.minimum(np.abs(_dref[_j] - _fq), np.abs(_dref[_jm] - _fq))
        det_band = _near <= (1e-9 * np.maximum(_fq, 1e-30))
        gb_truth_meta["n_det"] = int(det_band.sum())
        gb_truth_meta["n_undet"] = int((~det_band).sum())
        del _dref, _fq, _j, _jm, _near
    if expl["gb"]:
        _rec_lo = float(min(p[1] for p in expl["gb"]))
        cut = _rec_lo - 0.5      # 0.5 dex below the faintest recovered source
        keep = np.nonzero(la_band >= cut)[0]
        gb_truth_meta.update(cut=cut, rec_lo=_rec_lo, above_cut=int(keep.size))
        if keep.size > TRUTH_CAP:
            # PER-WINDOW decimation (2026-08-23). The old scheme kept the
            # globally brightest half of the budget outright, which drew a
            # hard horizontal red edge (one global brightness cut) across
            # the whole band with the recovered cloud floating above it --
            # on the zoom plots that edge read as "truth amplitudes are
            # wrong". Instead the budget is split across log-spaced f0
            # windows, and each window keeps the brightest half of ITS quota
            # outright plus a uniform subsample of the rest: the drawn
            # fraction is uniform in f0, any brightness cut is local, and no
            # band-wide edge exists to misread.
            #
            # THE QUOTA IS NOT A FLAT FRACTION (2026-08-27). It used to be
            # ``max(round(pop * TRUTH_CAP / above_cut), 1)``, which decimated
            # a window holding 2 sources exactly as hard as one holding
            # 12,000. Two things then broke at the sparse top of the band:
            #   * a window whose quota rounded to 1 got ``_nb = _q // 2 == 0``
            #     -- the "brightest half kept outright" guarantee degenerated
            #     to nothing, and the single point drawn was a COIN FLIP among
            #     the window's members;
            #   * so recovered, catalogue-present sources lost that flip and
            #     drew no red cross under their green dot. Measured on the v7
            #     3-month page: 470 of the 1,064 detectable catalogue sources
            #     carried no truth mark, only 12 of the 45 sources above
            #     10 mHz were drawn at all, and the flagship at 20.380377 mHz
            #     (SNR 46, recovered in 23/24 walkers) lost a 1-of-2 draw in
            #     the top window and vanished.
            # The repair is two rules, both general and budget-neutral:
            #   1. PROTECT: a catalogue row that a model source sits on (within
            #      the page's own TOL_BINS match tolerance) is always drawn.
            #      The panel exists to judge recovered against injected, so a
            #      recovered dot may never sit on an empty patch of truth.
            #   2. FLOOR: every window is drawn IN FULL up to _WIN_FULL members
            #      before any fraction is applied; only what the floor and the
            #      protected set leave over is spread proportionally. Sparse
            #      high-f windows cost a few hundred points in total, so there
            #      is no reason to sample them at all.
            _rng = np.random.default_rng(0)
            _NWIN, _WIN_FULL = 64, 200
            _wedges = np.geomspace(f0_band[keep].min() * (1 - 1e-12),
                                   f0_band[keep].max() * (1 + 1e-12),
                                   _NWIN + 1)
            _wid = np.clip(np.searchsorted(_wedges, f0_band[keep],
                                           side="right") - 1, 0, _NWIN - 1)

            # rule 1 -- rows a recovered source sits on, by f0 proximity in
            # the SAME tolerance the recovery panels call a match.
            _pro = np.zeros(keep.size, dtype=bool)
            _rf0 = np.unique(np.asarray([p[0] for p in expl["gb"]], dtype=float))
            if _rf0.size:
                _tolm = TOL_BINS * SCI_DF * 1e3           # mHz
                _srt = np.argsort(f0_band[keep])
                _ks = f0_band[keep][_srt]
                _plo = np.searchsorted(_ks, _rf0 - _tolm, side="left")
                _phi = np.searchsorted(_ks, _rf0 + _tolm, side="right")
                for _a, _b in zip(_plo, _phi):
                    if _b > _a:
                        _pro[_srt[_a:_b]] = True

            # ...and PROTECT EVERY DETECTABLE SOURCE too (2026-09-19). Rule 1
            # above only spares a catalogue row that a MODEL source already
            # sits on, so a detectable source the run has NOT found could be
            # decimated away -- and with red now meaning "detectable", a
            # missing red cross would read as "nothing to find here" when the
            # truth is "here is one it missed", which is the single most
            # important thing this panel can show. They are 1,064 rows out of
            # ~4M on the 3-month set, so protecting all of them is free.
            # Gated on "n_det", NOT on det_band itself: with no DET_F0 the
            # mask defaults to ALL-TRUE (everything drawn red, the old
            # behaviour), and OR-ing that in would protect every row and
            # silently disable decimation entirely.
            if "n_det" in gb_truth_meta:
                _pro |= det_band[keep]

            # rule 2 -- floor first, then spread what is left proportionally.
            # The floor is raised to the protected count where a window holds
            # more protected rows than the floor, so the per-window quota can
            # never be smaller than the rows rule 1 pins.
            _pop = np.bincount(_wid, minlength=_NWIN)
            _npro = np.bincount(_wid[_pro], minlength=_NWIN)
            _base = np.maximum(np.minimum(_pop, _WIN_FULL), _npro)
            _left = _pop - _base
            _extra = max(TRUTH_CAP - int(_base.sum()), 0)
            _frac = _extra / int(_left.sum()) if _left.sum() else 0.0
            _quota = _base + np.rint(_left * _frac).astype(int)

            _parts = []
            for _wi in range(_NWIN):
                _m = _wid == _wi
                if not _m.any():
                    continue
                _kw, _pw = keep[_m], _pro[_m]
                _q = int(min(_quota[_wi], _kw.size))
                if _kw.size <= _q:
                    _parts.append(_kw)
                    continue
                _must, _rest = _kw[_pw], _kw[~_pw]
                _r = _q - _must.size                     # >= 0 by _base above
                if _r <= 0:
                    _parts.append(_must)
                    continue
                _o = _rest[np.argsort(la_band[_rest])[::-1]]
                # never 0: a quota of 1 must spend it on the BRIGHTEST member
                # of the window, not on a random one.
                _nb = int(min(max(_r // 2, 1), _o.size))
                _ns = int(min(_r - _nb, _o.size - _nb))
                _sub = (_rng.choice(_o[_nb:], size=_ns, replace=False)
                        if _ns > 0 else np.empty(0, dtype=_o.dtype))
                _parts.append(np.concatenate([_must, _o[:_nb], _sub]))
            sel = np.unique(np.concatenate(_parts))
            gb_truth_meta["nwin"] = _NWIN
            gb_truth_meta["win_full"] = int(_WIN_FULL)
            gb_truth_meta["protected"] = int(_pro.sum())
            gb_truth_meta["drawn_frac"] = float(_frac)
        else:
            sel = keep
        # Third column: 1 = detectable (red), 0 = sub-threshold (grey). An
        # int costs 2 bytes a row in the JSON, against the ~23 the pair
        # already costs, and saves the canvas a second lookup structure.
        gb_truth_pts = [[float(f"{f0_band[i]:.7g}"), round(float(la_band[i]), 4),
                         int(det_band[i])] for i in sel]
        gb_truth_meta["shown"] = len(gb_truth_pts)

        # SPLIT the truth crosses into (grey: undetectable) and (red:
        # detectable but not recovered) so the zoomable canvas can render
        # the four required classes. Rows that are detectable AND recovered
        # are dropped here -- the green filled-circle source marker sits at
        # the recovered coordinate on top, which is what the reader is
        # trying to see.
        #
        # Classification is by f0 lookup with a small absolute tolerance
        # (1e-12 Hz, ~3 orders below the FD bin). `np.isin` would be
        # unreliable: TRU["f0"] goes through the mHz sampling basis in
        # build_truth (rows[:, 1] * 1e-3), so it does NOT bit-exactly match
        # the raw wdwd `GW22FrequencySSBFrame` in f0_band_hz.
        gb_truth_pts_grey = gb_truth_pts   # fallback: no TRU -> all grey
        gb_truth_pts_red = []
        if TRU is not None:
            def _within_tol_lookup(vals_hz):
                if vals_hz.size == 0:
                    return lambda q: np.zeros_like(q, dtype=bool)
                _s = np.sort(np.asarray(vals_hz, float))
                _n = _s.size

                def _fn(q):
                    q = np.asarray(q, float)
                    i = np.searchsorted(_s, q)
                    il = np.clip(i - 1, 0, _n - 1)
                    ir = np.clip(i, 0, _n - 1)
                    dl = np.abs(_s[il] - q); dr = np.abs(_s[ir] - q)
                    return np.minimum(dl, dr) < 1e-12
                return _fn

            _det_full_hz = np.asarray(
                TRU["f0"][ np.asarray(TRU["det"], bool)
                          & (np.asarray(TRU["f0"], float) >= band_edges[0])
                          & (np.asarray(TRU["f0"], float) <= band_edges[-1]) ],
                float)
            _rec_full_hz = (np.asarray(T_F0, float)[FOUND_MM]
                            if (isinstance(FOUND_MM, np.ndarray)
                                and FOUND_MM.size)
                            else np.zeros(0))

            _sel_hz = f0_band_hz[sel]
            _is_det = _within_tol_lookup(_det_full_hz)(_sel_hz)
            _is_rec = _within_tol_lookup(_rec_full_hz)(_sel_hz)
            # grey = undetectable within the decimated `sel`
            _grey_mask = ~_is_det
            gb_truth_pts_grey = [
                [float(f"{f0_band[sel[i]]:.7g}"),
                 round(float(la_band[sel[i]]), 4)]
                for i in np.where(_grey_mask)[0]]
            # red = detectable, not recovered -- pulled from the FULL
            # detectable set in-band, no decimation. There are only ~10^3 of
            # these even on a 1-year run, so this is cheap and complete.
            _found_arr = (np.asarray(FOUND_MM, bool)
                          if (isinstance(FOUND_MM, np.ndarray) and FOUND_MM.size)
                          else np.zeros(NDET, bool))
            _mis_f = np.asarray(T_F0, float)[~_found_arr]
            _mis_a = np.log10(np.maximum(
                np.asarray(T_AMP, float)[~_found_arr], 1e-40))
            gb_truth_pts_red = [
                [float(f"{f * 1e3:.7g}"), round(float(a), 4)]
                for f, a in zip(_mis_f, _mis_a)]
            gb_truth_meta["shown_grey"] = len(gb_truth_pts_grey)
            gb_truth_meta["shown_red"] = len(gb_truth_pts_red)
        _tm = gb_truth_meta
        _detbit = (
            f"{_tm.get('shown_det', _tm['shown']):,} of them RED (detectable, "
            f"SNR > 7), the rest GREY (sub-threshold -- drawn so a patch with "
            f"no red mark reads as 'nothing findable here' rather than as a "
            f"gap in the overlay). "
            if "shown_det" in _tm else "")
        expl["truth_cap"] = (
            f"Catalogue truths: {_tm['shown']:,} points shown of "
            f"{_tm['above_cut']:,} passing the cut log10 A >= {_tm['cut']:.2f} "
            f"(0.5 dex below the faintest recovered source, {_tm['rec_lo']:.2f}); "
            f"{_tm['in_band']:,} catalogue sources lie in the GB band in total. "
            + _detbit
            + (f"Decimation is per frequency window ({_tm['nwin']} log-spaced "
               f"windows). Any window holding at most {_tm['win_full']:,} "
               f"sources is drawn IN FULL -- which is every window above a few "
               f"mHz -- and the {_tm['protected']:,} catalogue rows lying "
               f"within {TOL_BINS:.0f} frequency bins of a model source are "
               f"always drawn, so no recovered dot can sit on a patch with no "
               f"truth mark under it. Only what those two rules leave over is "
               f"decimated, at ~{100 * _tm['drawn_frac']:.1f}% of each crowded "
               f"window's remainder (brightest half of that share kept "
               f"outright, the rest a uniform random subsample) -- the drawn "
               f"density is uniform in f0 and there is NO band-wide "
               f"brightness edge."
               if "nwin" in _tm else "No decimation was needed."))
except Exception as e:
    MISSING.append(f"GB injection catalogue truth overlay unavailable: {e!r}")

expl["truth"] = gb_truth_pts
# When TRU is available the classification split populated these two; when
# it was not, the grey list falls back to the full unclassified truth so
# the canvas still shows something recognisable.
expl["truth_grey"] = gb_truth_pts_grey or gb_truth_pts
expl["truth_red"]  = gb_truth_pts_red
expl["truth_meta"] = gb_truth_meta

# Recovered-source markers for the zoomable canvas: one per source at the
# last stored iteration, classified matched/unmatched under the current
# match criterion (MATCHED_MM). Coords are (f0 mHz, log10 amplitude) so
# they land on the same axes as expl["gb"].
expl["sources"] = []
try:
    if (RPHYS is not None and RPHYS.shape[0] > 0
            and isinstance(MATCHED_MM, np.ndarray)
            and MATCHED_MM.size == RPHYS.shape[0]):
        _rf_hz = np.asarray(RPHYS[:, 1], float)      # Hz from GBGPU
        _rf_mhz = _rf_hz * 1e3                       # convert to mHz for plot
        _ra_lg = np.log10(np.maximum(np.asarray(RPHYS[:, 0], float), 1e-40))
        expl["sources"] = [
            [round(float(_rf_mhz[i]), 7),
             round(float(_ra_lg[i]), 4),
             1 if bool(MATCHED_MM[i]) else 0]
            for i in range(RPHYS.shape[0])]
except NameError:
    pass
expl["match_note"] = MATCH_CRIT_TXT

EXPL_JSON = json.dumps(expl)

# zoomable dist-f0 posterior cloud: every sample (last iters x walkers x leaf)
_xs_axis = VGB_F0 if VGB_F0 is not None else np.arange(55).astype(float)
vgb_post = [[float(_xs_axis[leaf]), float(v)]
            for leaf in range(55) for v in vgb_last[:, leaf, 0]]
VGB_POST_JSON = json.dumps({
    "pts": vgb_post,
    "truth": ([[float(_xs_axis[leaf]), float(VGB_TRUTH[leaf, 0])]
               for leaf in range(55)] if VGB_TRUTH is not None else []),
    "xlab": "catalogue f0 [mHz]" if VGB_F0 is not None else "VGB leaf index",
})


# ---- per-source posterior panels (Tasks 2 + 4) ---------------------------
# Task 2: every VGB, SNR-descending in the selector -- rendered as
# ChainConsumer CORNER plots (2026-08-15, user request), one PNG per leaf,
# base64'd into the page. The selector swaps the src of a single <img>, so
# only the selected corner is ever in the DOM's visible flow; the JS
# histogram panel it replaced is gone; the GB panel moved to corner
# plots too (2026-08-20), so both selectors are now image swaps.
#
# Samples: the last POOL_ITS_POSTERIOR stored iterations x every cold walker
# (~240 rows), wider than the POOL_ITS_SAMPLES window the scatter/marginal
# panels use -- a corner needs the extra rows for its 2-D contours to mean
# anything. Both windows come from the shared pooling helper.
VGB_CORNER_ROWS = _vgb_pool_rows(POOL_ITS_POSTERIOR)
CORNER_ITS = int(VGB_CORNER_ROWS.size)
CORNER_DPI, CORNER_IN = 68, 7.0        # size-budget tuned (see below)
vgb_corner = vgb_c[VGB_CORNER_ROWS].reshape(-1, 55, _vgb_chain_ndim)  # (S, 55, 5 or 6)
VGB_CORNER = {"src": [], "nsamp": int(vgb_corner.shape[0]),
              "nits": int(CORNER_ITS), "nwalk": int(nwalk)}
CORNER_BYTES = []
try:
    import logging
    import warnings as _warnings

    import pandas as pd
    from chainconsumer import Chain, ChainConsumer, PlotConfig, Truth

    # chainconsumer logs "Parameter <p> in chain ... is not constrained"
    # once per unconstrained column per leaf (hundreds of lines here, and
    # informational -- an unconstrained VGB angle is a RESULT, not a fault).
    logging.getLogger("chainconsumer").setLevel(logging.ERROR)

    def corner_png(Sm, names, truth, title, size_in=None, tally=None,
                   dpi=None, label_fs=8, tick_fs=7, smooth=None, bins=None):
        """Samples -> a base64 PNG of their ChainConsumer corner.

        Shared by the VGB leaf panel and the GB per-source panel; the only
        thing that differs between them is the parameter list, the figure
        size, and how the samples were gathered.

        Extents are widened to contain the catalogue truth so a truth line
        that falls OUTSIDE the posterior is still visible; without it a
        badly-recovered source would silently show no truth at all.

        SMOOTHING TRACKS SAMPLE COUNT. chainconsumer's default smooth=1 on a
        few dozen rows does not produce a posterior, it produces a field of
        spurious closed rings that read as multimodality. Below ~200 rows
        the kernel is widened and the binning coarsened so the contours
        cannot claim structure the sample count does not support.
        """
        Sm = np.asarray(Sm, dtype=float)
        n = int(Sm.shape[0])
        smooth_, bins_ = (3, 12) if n < 200 else (1, 18)
        if smooth is not None:
            smooth_ = smooth
        if bins is not None:
            bins_ = bins
        _dpi = dpi or CORNER_DPI
        ext = {}
        for j, nm_ in enumerate(names):
            lo, hi = float(np.min(Sm[:, j])), float(np.max(Sm[:, j]))
            if truth is not None and np.isfinite(truth[j]):
                lo = min(lo, float(truth[j])); hi = max(hi, float(truth[j]))
            if not hi > lo:
                hi = lo + max(abs(lo) * 1e-6, 1e-12)
            pd_ = (hi - lo) * 0.06
            ext[nm_] = (lo - pd_, hi + pd_)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            cc = ChainConsumer()
            cc.add_chain(Chain(
                samples=pd.DataFrame(Sm, columns=names), name="posterior",
                color=VIOLET, shade=True, shade_alpha=0.35, bar_shade=True,
                plot_point=False, smooth=smooth_, bins=bins_,
                show_label_in_legend=False))
            if truth is not None:
                cc.add_truth(Truth(
                    location={nm_: float(truth[j])
                              for j, nm_ in enumerate(names)},
                    color=RED, line_style=":", line_width=1.4))
            # diagonal_tick_labels defaults ON and is unreadable at this
            # dpi; 3 upright ticks per axis is what survives the size budget.
            cc.set_plot_config(PlotConfig(
                labels={nm_: nm_ for nm_ in names}, extents=ext,
                label_font_size=label_fs, tick_font_size=tick_fs, max_ticks=3,
                diagonal_tick_labels=False,
                show_legend=False, summarise=False, dpi=_dpi))
            _sz = size_in or CORNER_IN
            fig_ = cc.plotter.plot(figsize=(_sz, _sz))
        fig_.suptitle(title, fontsize=9 if _dpi <= 80 else 11, color=FG)
        buf = io.BytesIO()
        fig_.savefig(buf, format="png", dpi=_dpi, bbox_inches="tight")
        plt.close(fig_)
        raw = buf.getvalue()
        (CORNER_BYTES if tally is None else tally).append(len(raw))
        return base64.b64encode(raw).decode()

    def vgb_corner_png(leaf, title):
        """One leaf -> a base64 PNG of its 5x5 ChainConsumer corner."""
        return corner_png(vgb_corner[:, leaf, :], VGB_NAMES,
                          None if VGB_TRUTH is None else VGB_TRUTH[leaf],
                          title)

    # DETECTABLE-ONLY (2026-09-05): render corner PNGs only for the SNR>7
    # detectable set. The 44 prior-dominated leaves render flat, useless
    # corners and were the dominant page-weight cost after POOL_ITS->30 --
    # dropping them is the main 16 MB size win. The per-leaf corner selector
    # panel + JS stay; the picker now lists exactly this detectable set.
    _corner_order = list(VGB_DET)
    for leaf in _corner_order:
        nm = VGB_IDS[leaf] if VGB_IDS else f"leaf {leaf}"
        _f0 = VGB_F0[leaf] if VGB_F0 is not None else float("nan")
        _lab = f"{nm} - {_f0:.4f} mHz - SNR~{snr[leaf]:.0f}"
        VGB_CORNER["src"].append({
            "label": _lab,
            "sub": (f"leaf {int(leaf)} - {vgb_corner.shape[0]} samples "
                    f"(last {CORNER_ITS} stored iterations x {nwalk} cold "
                    f"walkers) - dotted red = catalogue truth"),
            "png": vgb_corner_png(int(leaf), _lab),
        })
    print(f"[corner] {len(CORNER_BYTES)} VGB corners, "
          f"png min/med/max = {min(CORNER_BYTES)/1024:.1f} / "
          f"{np.median(CORNER_BYTES)/1024:.1f} / {max(CORNER_BYTES)/1024:.1f} KB, "
          f"total png {sum(CORNER_BYTES)/1024**2:.2f} MB "
          f"(base64 {sum(CORNER_BYTES)*4/3/1024**2:.2f} MB) "
          f"@ dpi={CORNER_DPI}, figsize={CORNER_IN}")
except Exception as e:
    MISSING.append(f"VGB ChainConsumer corner plots unavailable: {e!r}")
VGB_CORNER_JSON = json.dumps(VGB_CORNER)

# Task 4: the 3 highest-frequency RECOVERED GBs.
# Tobs from the stored domain settings (WDMSettings args = Nt, Nf, dt) ->
# the FD bin width that sets both the cluster window and the Delta-f0 unit.
TOBS = 3 * 30 * 86400.0
try:
    _a = dict(f["global_fit/domain_settings/args"].attrs)
    TOBS = float(_a["0"]) * float(_a["1"]) * float(_a["2"])
except Exception as e:
    MISSING.append(f"Tobs not readable from domain_settings ({e!r}); "
                   "using 90 d for the f0 bin width.")
DF_MHZ = 1e3 / TOBS                       # one FD bin, in mHz
CLUSTER_BINS, MATCH_BINS = 20.0, 100.0
GB_NAMES = ["dist [kpc]", "f0 [mHz]", "Mc [Msol]", "phi0", "cos_iota",
            "psi", "alpha", "sin_delta", "fdot_astro_ratio"]
GB1 = {"params": GB_NAMES, "src": []}
gb1_meta = {}
_rows = sorted(((float(gb_chain_cold[w, i, 1]), w, i) for w in range(nwalk)
                for i in np.nonzero(gb_alive_last[w])[0]), key=lambda r: -r[0])
if _rows:
    clusters, cur = [], [_rows[0]]
    for r in _rows[1:]:
        if cur[-1][0] - r[0] <= CLUSTER_BINS * DF_MHZ:
            cur.append(r)
        else:
            clusters.append(cur); cur = [r]
    clusters.append(cur)
    # A "recovered source" must live in at least 3 of the 24 cold walkers;
    # 1-2-walker clusters at the top of the band are transient births, not
    # sources (their count is quoted in the caption -- it is itself a
    # readout of high-f birth churn).
    solid = [c for c in clusters if len({x[1] for x in c}) >= 3]
    gb1_meta["n_clusters"] = len(clusters)
    gb1_meta["n_solid"] = len(solid)
    gb1_meta["transient_above"] = (
        len([c for c in clusters
             if c[0][0] > solid[0][0][0] and len({x[1] for x in c}) < 3])
        if solid else len(clusters))
    # POOL SAMPLES ACROSS ITERATIONS (2026-08-20). The marginal-histogram
    # panel this replaced drew a single stored iteration -- ~33 rows, which
    # is enough for a 1-D bar chart and nowhere near enough for a 2-D
    # contour. The GB branch is trans-dimensional, so leaf index i is NOT
    # the same source from one iteration to the next; rows are associated by
    # f0 proximity instead, at most one per (iteration, walker) cell, taking
    # the alive leaf nearest the cluster centre. Same widened window the VGB
    # corners already use, and the same reason for it. Routed through the
    # shared pooling helper (2026-08-27) so the window, the young-store
    # clamp and the extract keep-window clamp are defined in ONE place.
    GB_CORNER_ROWS = _pool_its(POOL_ITS_POSTERIOR, "gb")
    GB_CORNER_ITS = int(GB_CORNER_ROWS.size)
    # 9 params need far more canvas than the VGB panel's 5. At the VGB's
    # 7in/68dpi the axis labels of a 9x9 collide into an unreadable smear.
    GB_CORNER_IN, GB_CORNER_DPI = 13.0, 96
    # SHORT AXIS LABELS, UNITS IN THE CAPTION. "dist [kpc]" / "Mc [Msol]" /
    # "fdot_astro_ratio" overrun their panels at 9 across.
    GB_CORNER_LABELS = ["dist", "d_f0 [uHz]", "Mc", "phi0", "cos_iota",
                        "psi", "alpha", "sin_delta", "fdot_ratio"]
    # f0 is plotted as an OFFSET from the cluster centre. Absolute f0 has a
    # ~0.002 mHz spread on a ~20 mHz value, so matplotlib renders it with an
    # "[1e-3+2.038e1]" offset tag glued to the axis label -- the single worst
    # contributor to the collisions. The centre is in the panel title, and
    # the offset is the same Delta-f0 the catalogue-match note quotes.
    GB_CORNER_BYTES = []
    _gb_win = CLUSTER_BINS * DF_MHZ
    _centers = [float(np.median([gb_chain_cold[w, i, 1] for _, w, i in c]))
                for c in solid[:3]]
    _pool = [[] for _ in _centers]
    _pool_its = [0 for _ in _centers]
    for _it in GB_CORNER_ROWS:
        _al = gb_inds[_it]                             # THIS iteration's mask
        if not _al.any():
            continue
        _ch = g["chain/gb"][_it, 0, 0]                 # (nwalk, nleaf, 9)
        for _k, _c0 in enumerate(_centers):
            _got = 0
            for _w in range(nwalk):
                _idx = np.nonzero(_al[_w])[0]
                if not _idx.size:
                    continue
                _d = np.abs(_ch[_w, _idx, 1] - _c0)
                _j = int(np.argmin(_d))
                if _d[_j] <= _gb_win:
                    _pool[_k].append(np.array(_ch[_w, _idx[_j]], dtype=float))
                    _got += 1
            _pool_its[_k] += 1 if _got else 0
        del _ch
    gb1_meta["corner_its"] = int(GB_CORNER_ITS)

    for _k, c in enumerate(solid[:3]):
        # pooled rows if the association found any, else the single-iteration
        # cluster (so the panel still renders on a one-iteration snapshot)
        P = (np.array(_pool[_k]) if _pool[_k]
             else np.array([gb_chain_cold[w, i] for _, w, i in c]))
        f0_med = float(np.median([gb_chain_cold[w, i, 1] for _, w, i in c]))
        truth, note, bad = None, "", False
        if f0_band is not None and f0_band.size:
            j = int(np.argmin(np.abs(f0_band - f0_med)))
            d_bins = (f0_med - f0_band[j]) / DF_MHZ
            if abs(d_bins) <= MATCH_BINS:
                gidx = int(cat_gidx[j])   # row in the full 15.5M catalogue
                entry = {k: np.atleast_1d(float(cf[k][gidx]))
                         for k in ("Amplitude", "GW22FrequencySSBFrame",
                                   "GW22FrequencyDerivativeSourceFrame",
                                   "TrueAnomaly", "InclinationAngle",
                                   "PolarisationAngle", "RightAscension",
                                   "Declination", "LuminosityDistance",
                                   "ChirpMassSSBFrame")}
                truth = cat_to_sampled9(entry)[0][0]
                from lisatools.globalfit.stock.erebor.transforms import (
                    gb_amp_from_dist as _amp)
                a_rec = float(np.median(_amp(P[:, 1] * 1e-3, P[:, 2],
                                            np.maximum(P[:, 0], 1e-6))))
                a_cat = float(entry["Amplitude"][0])
                n_near = int(np.sum(np.abs(f0_band - f0_med)
                                    <= MATCH_BINS * DF_MHZ))
                note = (f"catalogue match ID {int(cf['ID'][gidx])}: "
                        f"df0 = {d_bins:+.1f} bins ({(f0_med - f0_band[j])*1e3:+.3f} "
                        f"uHz), A_rec/A_cat = {a_rec / a_cat:.2f}, "
                        f"{n_near} catalogue source(s) within "
                        f"{MATCH_BINS:.0f} bins")
            else:
                note = (f"NO catalogue match within {MATCH_BINS:.0f} bins "
                        f"(nearest is {d_bins:+.0f} bins away) - this "
                        f"recovery has no injected counterpart")
                bad = True
        _lab = (f"GB @ {f0_med:.5f} mHz ({len(P)} samples, "
                f"{len({x[1] for x in c})} walkers)")
        # centre the f0 column (and its truth) on the cluster, in uHz
        _Pc = np.array(P, dtype=float, copy=True)
        _Pc[:, 1] = (_Pc[:, 1] - f0_med) * 1e3
        _tc = None
        if truth is not None:
            _tc = np.array(truth, dtype=float, copy=True)
            _tc[1] = (_tc[1] - f0_med) * 1e3
        try:
            _png = corner_png(_Pc, GB_CORNER_LABELS, _tc, _lab,
                              size_in=GB_CORNER_IN, dpi=GB_CORNER_DPI,
                              label_fs=11, tick_fs=9, smooth=3, bins=14,
                              tally=GB_CORNER_BYTES)
        except Exception as _e:
            MISSING.append(f"GB corner plot for {_lab} unavailable: {_e!r}")
            continue
        GB1["src"].append({
            "label": _lab,
            "sub": (f"{len(P)} samples pooled over the last "
                    f"{_pool_its[_k]} GB-active stored iterations x {nwalk} "
                    f"cold walkers, associated by f0 within the "
                    f"{CLUSTER_BINS:.0f}-bin cluster window "
                    f"({CLUSTER_BINS * DF_MHZ * 1e3:.2f} uHz); "
                    f"dotted red = catalogue truth. Axes: dist [kpc], "
                    f"d_f0 = f0 - {f0_med:.5f} mHz [uHz], Mc [Msol], "
                    f"phi0, cos_iota, psi, alpha, sin_delta, "
                    f"fdot_astro_ratio"),
            "note": note, "bad": bad, "png": _png,
        })
    if GB_CORNER_BYTES:
        print(f"[corner] {len(GB_CORNER_BYTES)} GB corners, png "
              f"min/max = {min(GB_CORNER_BYTES)/1024:.1f} / "
              f"{max(GB_CORNER_BYTES)/1024:.1f} KB, total "
              f"{sum(GB_CORNER_BYTES)/1024**2:.2f} MB")
GB1_JSON = json.dumps(GB1)

# ---- RESTORED: run mechanics. Engineering instrumentation, so it lives
# in the collapsed appendix rather than the body -- but it is the only
# view of where the wall time goes and what the devices are holding.
# ---- 8. efficiency: proposals/s + wall per propose (GB_TIMING records) ----
TIM_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?\[GB_TIMING (\w+)\] "
    r"total=([\d.]+)s.*\| ([^|]+)$")
recs = []
for line in log_text.splitlines():
    m = TIM_RE.match(line)
    if m:
        cnt = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)\b", m.group(4)))
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        recs.append((t, m.group(2), float(m.group(3)), cnt))
if recs:
    t0r = recs[0][0]
    MOVE_COLOR = {"rj_fstat_search": GREEN, "rj_fstat_pe": GREEN,
                  "vgb_pe": VIOLET, "rj_prior_removal": AMBER}
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    series = {}
    for t, name, tot, cnt in recs:
        picked = cnt.get("picked_sources", 0)
        blocks = max(cnt.get("inmodel_blocks", 0), 1)
        # in-model proposals ~ repeat calls x mean block size (each repeat
        # call proposes once for every source in its block)
        inm = cnt.get("inmodel_repeat_calls", 0) * (
            cnt.get("inmodel_sources", 0) / blocks)
        props = picked + inm
        series.setdefault(name, {"t": [], "rate": [], "wall": []})
        series[name]["t"].append((t - t0r).total_seconds() / 60)
        series[name]["rate"].append(props / max(tot, 1e-9))
        series[name]["wall"].append(tot)
    for name, d in sorted(series.items()):
        c = MOVE_COLOR.get(name, CYAN)
        ax[0].plot(d["t"], d["rate"], "o-", ms=3.5, lw=1.0, color=c,
                   label=f"{name} (n={len(d['t'])})")
        ax[1].plot(d["t"], d["wall"], "o-", ms=3.5, lw=1.0, color=c)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("proposals / s"); ax[0].set_xlabel("minutes since first record")
    ax[0].set_title("proposal throughput per propose"); ax[0].legend(fontsize=7)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("move wall [s]"); ax[1].set_xlabel("minutes since first record")
    ax[1].set_title("wall time per propose")
    fig_b64(fig, "timing_moves")

# gpu util CSVs (latest three jobs only -- earlier ones are archived attempts)
csvs = sorted([fn for fn in os.listdir(RUN_DIR) if fn.startswith("gpu_util")])[-3:]
if csvs:
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.0), sharex=False)
    for ci, fn in enumerate(csvs):
        rows = [l.split(",") for l in open(os.path.join(RUN_DIR, fn)) if l.strip()]
        try:
            t0 = None
            per = {}
            for r in rows:
                ts = datetime.strptime(r[0].strip().split(".")[0], "%Y/%m/%d %H:%M:%S")
                if t0 is None: t0 = ts
                gpu = int(r[1]); per.setdefault(gpu, {"t": [], "u": [], "m": []})
                per[gpu]["t"].append((ts - t0).total_seconds() / 60)
                per[gpu]["u"].append(float(r[3])); per[gpu]["m"].append(float(r[5]) / 1024)
            for gpu, d in per.items():
                c = CYAN if gpu == 0 else AMBER
                ls = "-" if ci == len(csvs)-1 else ":"
                ax[0].plot(d["t"], d["u"], color=c, ls=ls, lw=0.9,
                           label=f"{fn} gpu{gpu}")
                ax[1].plot(d["t"], d["m"], color=c, ls=ls, lw=0.9)
        except Exception:
            continue
    ax[0].set_ylabel("util [%]"); ax[0].legend(fontsize=7, ncols=2)
    ax[1].set_ylabel("mem [GiB]"); ax[1].set_xlabel("minutes since job start")
    ax[0].set_title("nvidia-smi telemetry (dotted = earlier job)")
    fig_b64(fig, "gpu_util")

# ---- 9. swaps (last ACTIVE iteration per branch: a stored iteration can
# record zero proposals for one branch at a stage handoff) ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.0))
_empty_branches = []
for arrs, name, a in [((psd_sw_a, psd_sw_p), "psd", 0), ((gal_sw_a, gal_sw_p), "galfor", 1)]:
    sa, sp = arrs
    # Empty swap-count arrays happen on a very young / mid-flush snapshot
    # where SUB_NIT has trimmed the whole stream away. The prior fallback
    # ``sp.shape[0] - 1`` then landed at -1 and sa[-1] raised IndexError.
    # Skip the branch with an empty panel + a MISSING reason instead.
    if sp.shape[0] == 0:
        ax[a].set_title(f"{name} swap acceptance (no rows in this snapshot)")
        ax[a].set_xlabel("rung"); ax[a].set_ylim(0, 1)
        _empty_branches.append(name)
        continue
    nz = np.where(sp.sum(axis=1) > 0)[0]
    k_it = int(nz.max()) if nz.size else sp.shape[0] - 1
    rate = sa[k_it] / np.maximum(sp[k_it], 1)
    ax[a].bar(np.arange(len(rate)), rate, color=CYAN if a == 0 else AMBER, alpha=0.85)
    ax[a].set_title(f"{name} swap acceptance per rung (iter {k_it})")
    ax[a].set_xlabel("rung"); ax[a].set_ylim(0, 1)
if _empty_branches:
    MISSING.append(
        f"swap-acceptance panel: {'/'.join(_empty_branches)} sub-backend "
        f"has no rows in this snapshot (young run or mid-flush trim to "
        f"SUB_NIT=0); those panels render empty.")
fig_b64(fig, "swaps")

# ---- 10. grouped-RJ stats + device-memory telemetry (new-code lines) -------
RJ_STATS = {}
m = re.findall(r"band unit complete after (\d+) pick rounds \((\d+) cells\)",
               log_text)
big_units = [(int(r), int(c)) for r, c in m if int(c) > 5000]
if big_units:
    RJ_STATS["cells"] = big_units[-1][1]
    RJ_STATS["rounds"] = big_units[-1][0]
m = re.findall(r"grouped in-model \S+ (\d+) flushes, mean batch ([\d.]+) "
               r"sources \((\d+) buffer slots\)", log_text)
if m:
    RJ_STATS["flushes"], RJ_STATS["batch"], RJ_STATS["slots"] = (
        int(m[-1][0]), float(m[-1][1]), int(m[-1][2]))
# direct-batch mode (GB_RJ_DIRECT_BATCH, 2026-08-14): rigid rj batches +
# one end-of-unit in-model phase in capacity chunks
m = re.findall(r"direct batches \S+ (\d+) rj batch\(es\), (\d+) survivors "
               r"polished in (\d+) in-model chunk\(s\) \((\d+) buffer slots\)",
               log_text)
if m:
    RJ_STATS["flushes"] = int(m[-1][2])
    RJ_STATS["batch"] = int(m[-1][1]) / max(int(m[-1][2]), 1)
    RJ_STATS["slots"] = int(m[-1][3])
m = re.findall(r"at-cap skip -- (\d+) dead \(birth\) slots excluded across "
               r"(\d+) at-cap cells", log_text)
if m:
    RJ_STATS["atcap_cells"] = int(m[-1][1])

# last full rj GB_TIMING record PER MOVE -> one breakdown bar panel each.
# Before 2026-08-15 this took tm_rj[-1] only, so whichever rj move happened
# to log LAST (rj_prior_removal, 75 s) was the only breakdown on the page
# and rj_fstat_search (1,041 s, the actual hog) never appeared. Every rj
# move now gets its OWN panel, so the moves are never lumped or shadowed.
tm_rj = re.findall(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?\[GB_TIMING (rj_\w+)\] "
    r"(total=[^|]+)\|([^|]+)\|(.*)$", log_text, re.M)
RJ_LAST = {}                       # move -> (stamp, head, body, tail)
for stamp, name, head, body, tail in tm_rj:
    RJ_LAST[name] = (stamp, head, body, tail)
# rj_fstat_search (the F-stat birth proposal) leads, then rj_prior_removal;
# any other rj move follows in first-seen order.
RJ_ORDER = [n for n in ("rj_fstat_search", "rj_prior_removal") if n in RJ_LAST]
RJ_ORDER += [n for n in RJ_LAST if n not in RJ_ORDER]
RJ_MOVE_COLOR = {"rj_fstat_search": GREEN, "rj_fstat_pe": GREEN,
                 "rj_prior_removal": AMBER}
# run_proposal / run_tempering are ENCLOSING phase marks -- they contain the
# leaf spans, so leaving them in the bars just reprints the total (1,037 of
# 1,041 s for rj_fstat_search) and buries the actual hog. They are quoted in
# each panel's title instead; the bars are leaf spans only.
RJ_WRAPPERS = ("run_proposal", "run_tempering")
RJ_BREAK = {}                      # move -> (stamp, total, [(span, s), ...])
if RJ_ORDER:
    NTOP = 9
    fig, axs = plt.subplots(1, len(RJ_ORDER), figsize=(6.0 * len(RJ_ORDER), 3.6))
    axs = np.atleast_1d(axs)
    for a_, name in zip(axs, RJ_ORDER):
        stamp, head, body, tail = RJ_LAST[name]
        parts = dict(re.findall(r"(\w+)=([\d.]+)s", head + body))
        tot = float(parts.pop("total", 0)); parts.pop("tracked", None)
        parts.pop("untracked", None)
        wrap = {w: float(parts.pop(w)) for w in RJ_WRAPPERS if w in parts}
        top = sorted(parts.items(), key=lambda kv: -float(kv[1]))[:NTOP]
        RJ_BREAK[name] = (stamp, tot, [(k, float(v)) for k, v in top])
        labels = [k for k, _ in top][::-1]
        vals = [float(v) for _, v in top][::-1]
        a_.barh(labels, vals, color=RJ_MOVE_COLOR.get(name, CYAN), alpha=0.9,
                height=0.6)
        for y_, v in enumerate(vals):
            a_.text(v, y_, f"  {v:,.1f}s ({100 * v / max(tot, 1e-9):.0f}%)",
                    va="center", fontsize=8, color=FG)
        a_.set_xlim(0, max(vals + [1.0]) * 1.34)
        a_.tick_params(axis="y", labelsize=8)
        a_.set_xlabel("seconds", fontsize=9)
        _wtxt = ", ".join(f"{w} {v:,.1f}s" for w, v in wrap.items())
        a_.set_title(f"{name}: total {tot:,.1f}s (last full record, "
                     f"{stamp[5:]})\nleaf spans only; enclosing: "
                     f"{_wtxt or 'none'}", fontsize=9)
    fig.tight_layout()
    # counters of the last record overall (unchanged behavior)
    RJ_STATS["gbt_counters"] = dict(
        re.findall(r"(\w+)=(\d+)\b", tm_rj[-1][4]))
    fig_b64(fig, "rj_breakdown")

# device-memory telemetry series (buffer build / lifecycle / unit-open lines)
mem_re = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?GPU pool used ([\d.]+) / "
    r"total ([\d.]+) GB; device used/total GB: dev0 ([\d.]+)/[\d.]+, "
    r"dev1 ([\d.]+)")
mem_pts = []
for line in log_text.splitlines():
    mm_ = mem_re.match(line)
    if mm_:
        t = datetime.strptime(mm_.group(1), "%Y-%m-%d %H:%M:%S")
        mem_pts.append((t, *(float(x) for x in mm_.groups()[1:])))
if mem_pts:
    t0m = mem_pts[0][0]
    tm_min = np.array([(p[0] - t0m).total_seconds() / 60 for p in mem_pts])
    arr = np.array([p[1:] for p in mem_pts])
    # NaN-break across attempt gaps so restarts don't draw false ramps
    gaps = np.where(np.diff(tm_min) > 5.0)[0]
    for gi in gaps[::-1]:
        tm_min = np.insert(tm_min, gi + 1, np.nan)
        arr = np.insert(arr, gi + 1, np.nan, axis=0)
    tm_min = tm_min.tolist()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(tm_min, arr[:, 2], color=CYAN, lw=1.4, label="dev0 device used")
    ax.plot(tm_min, arr[:, 3], color=AMBER, lw=1.4, label="dev1 device used")
    ax.plot(tm_min, arr[:, 1], color=GREEN, lw=1.0, ls="--",
            label="cupy pool total (current dev)")
    ax.set_xlabel(f"minutes since {t0m:%m-%d %H:%M}")
    ax.set_ylabel("GB (of 99.9/device)")
    ax.legend(); ax.set_title(
        "device-wide memory from the in-run telemetry lines (memGetInfo)")
    fig_b64(fig, "mem_telemetry")
rj_kpis = ""
if RJ_STATS:
    c_ = RJ_STATS

    def _kpi(key, dec=0):
        """Thousands-separated KPI, or an em dash when the log lacks it.

        A missing key used to reach an f-string ``{...:,}`` as the string
        "?", which raises ValueError ("Cannot specify ',' with 's'") and
        killed the whole page. Snapshots legitimately miss KPIs -- an early
        fresh run has no completed RJ unit yet -- so degrade, never crash.
        """
        v = c_.get(key)
        if v is None:
            return "&mdash;"
        try:
            return f"{float(v):,.{dec}f}"
        except (TypeError, ValueError):
            return str(v)

    rj_kpis = f"""
<div class="kpi">
  <div><b>{_kpi('cells')}</b><span>cells / rj unit</span></div>
  <div><b>{_kpi('slots')}</b><span>buffer slots (staged)</span></div>
  <div><b>{_kpi('rounds')}</b><span>pick rounds / unit</span></div>
  <div><b>{_kpi('flushes')}</b><span>in-model flushes</span></div>
  <div><b>{_kpi('batch')}</b><span>mean flush batch [sources]</span></div>
  <div><b>{_kpi('atcap_cells')}</b><span>at-cap cells skipped</span></div>
</div>"""

# one-line-per-move readout under the breakdown panel (built from the SAME
# records the figure is built from -- never hand-typed numbers)
rj_break_txt = ""
rj_break_stamps = ""
if RJ_BREAK:
    _st = sorted({v[0] for v in RJ_BREAK.values()})
    rj_break_stamps = f" ({_st[0]} &ndash; {_st[-1]})" if _st else ""
    rj_break_txt = "<br>" + "<br>".join(
        f"<code>{n}</code>: total <strong>{RJ_BREAK[n][1]:,.1f} s</strong>, "
        f"top span <code>{RJ_BREAK[n][2][0][0]}</code> "
        f"{RJ_BREAK[n][2][0][1]:,.1f} s "
        f"({100 * RJ_BREAK[n][2][0][1] / max(RJ_BREAK[n][1], 1e-9):.0f}%)"
        for n in RJ_ORDER) + "<br>"
# ============================ HTML ==========================================
stage_now = "?"
for k, (o, s_) in sorted(recipe.items(), key=lambda kv: kv[1][0]):
    if not s_:
        stage_now = k; break
chips = "".join(
    f'<span class="chip {"done" if s_ else ("now" if k == stage_now else "")}">'
    f'{o}. {k}{" &#10003;" if s_ else ""}</span>'
    for k, (o, s_) in sorted(recipe.items(), key=lambda kv: kv[1][0]))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ---- the like-for-like arm table ------------------------------------------
# The two arms started their galactic-binary search at DIFFERENT absolute
# iterations (v2's first GB leaf lands at iteration 5, v3's at 16), so any
# comparison at a shared absolute iteration silently gives v2 eleven extra
# search steps. Everything below is indexed on iterations SINCE the first GB
# leaf, and the table is cut at the shorter arm's length.
ARM_TABLE = ""
# ARMS is built inside the GB-analysis section, which a very young store
# (too few GB iterations for the match machinery) skips entirely -- the
# cross-arm table then simply has no data to show.
if len(globals().get("ARMS", {})) >= 2 and SCI and SHOW_MATCH_STATS:
    _K = min(int(D["n_match"].size) - 1 - int(D["it0"]) for D in ARMS.values())
    _rows = []
    for _t in sorted(ARMS):
        _D = ARMS[_t]
        _i = int(_D["it0"]) + _K
        _rows.append((_t, int(_D["n_all"][_i]), int(_D["n_match"][_i]),
                      int(_D["n_match"][_i]) / SCI["ndet"],
                      int(_D["n_match"][_i]) / max(int(_D["n_band"][_i]), 1),
                      int(_D["n_match"].size) - 1 - int(_D["it0"])))
    _hdr = "".join(f'<th style="text-align:right;padding:4px 0 4px 20px">{r[0]}'
                   f'</th>' for r in _rows)
    def _row(lbl, fn):
        return ("<tr><td style='padding:3px 0'>" + lbl + "</td>"
                + "".join("<td style='text-align:right;padding:3px 0 3px 20px'>"
                          + fn(r) + "</td>" for r in _rows) + "</tr>")
    ARM_TABLE = f"""
<table style="border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;margin-top:10px">
<tr style="border-bottom:1px solid var(--line)"><th style="text-align:left;padding:4px 0">
at {_K} galactic-binary search iterations</th>{_hdr}</tr>
{_row("model sources", lambda r: f"{r[1]:,}")}
{_row("matched to a detectable injection", lambda r: f"{r[2]:,}")}
{_row("completeness", lambda r: pct(r[3]))}
{_row("purity", lambda r: pct(r[4]))}
{_row("search iterations completed in total", lambda r: f"{r[5]}")}
</table>"""

# ---- captions, every number read off the arrays that made the figure ------
if SCI:
    cap_f2 = (
        f"Completeness (solid, left axis) is the share of the {SCI['ndet']} "
        f"detectable injections matched by a model source within "
        f"{TOL_BINS:.0f} frequency bins; purity (dashed, right axis) is the "
        f"share of model sources that so match. Now "
        f"{pct(SCI['completeness'])} and {pct(SCI['purity'])}.")
    cap_f3 = (
        f"Noise-weighted overlap between each matched pair, maximised over an "
        f"overall phase. Read the lower panel's height at any overlap as the "
        f"NUMBER of sources recovered that well. Median "
        f"{SCI.get('mm_med', float('nan')):.2f}; "
        f"{pct(SCI.get('mm_hi', 0))} exceed 0.9.")
    cap_f4 = (
        f"Left: every model source in the amplitude-frequency plane, coloured "
        f"by optimal SNR. Right: the same plane split three ways. Recovery "
        f"tracks amplitude, and the misses concentrate along the faint edge "
        f"rather than anywhere structural. {MATCH_CRIT_HTML}")
    cap_f5 = (
        f"Where the sources are and where they are being found. The lower "
        f"panel is the per-bin recovery fraction with 68% Wilson intervals. "
        f"Recovery climbs steeply above 8 mHz, which is the galaxy thinning "
        f"out rather than the search improving there.")
    cap_f6 = (
        f"Recovery against injected SNR, with 68% Wilson intervals, both arms "
        f"on equal footing. The monotone rise is the health check: a search "
        f"that adds sources arbitrarily would be flat here. This axis is "
        f"ours, not a field convention.")
    cap_f7 = (
        f"Recovered minus injected for the matched pairs. Frequency is in "
        f"bins, amplitude is a log ratio, angles are raw. "
        f"{pct(SCI.get('df_tight', 0), 0)} of matches sit within half a "
        f"frequency bin; the median sky offset is "
        f"{SCI.get('sep_med', float('nan')):.0f} degrees.")
    cap_f8 = (
        f"Recovered sources over the injected population, coloured by SNR. "
        f"The bulge dominates both. Sky is the weakest-constrained "
        f"coordinate at these signal strengths, so scatter here is expected "
        f"and is quantified in the parameter panel.")
    cap_f10 = (
        f"How close model sources sit to each other in frequency. Below the "
        f"match tolerance the model is denser than the injections, which is "
        f"two templates sharing one source; "
        f"{pct(SCI.get('nn_half', 0), 1)} of leaves have a neighbour within "
        f"half a bin against {pct(SCI.get('nn_half_t', 0), 1)} of injections.")
else:
    cap_f2 = cap_f3 = cap_f4 = cap_f5 = cap_f6 = cap_f7 = cap_f8 = cap_f10 = ""
if SCI and not SHOW_MATCH_STATS:
    # Match-criterion content is gated off this page (user ruling
    # 2026-08-19): truths stay in every overlay, but nothing is classified
    # or counted by the page's own 2-bin proxy match. Match-criterion note
    # still emitted so the reader knows what would be applied were the gate
    # opened.
    cap_f4 = (
        "Left: every model source in the amplitude-frequency plane, "
        "coloured by optimal SNR, over the detectable injections (grey). "
        "Right: injections and model overlaid in the same plane. The "
        "comparison is visual; no match criterion is applied on this page. "
        f"(Would be: {MATCH_CRIT_TXT}.)")
    cap_f5 = (
        "Source density per frequency bin: the full catalogue, its "
        "detectable subset, and the model population. The gap between the "
        "green and cyan curves is read by eye; no per-source matching is "
        "applied.")
    cap_f10 = (
        "How close model sources sit to each other in frequency, against "
        "the same distribution for the injections. Sub-bin spacing in the "
        "model relative to the injections indicates template sharing.")

# A PARTIAL DTR means the data/template/residual analysis raised partway
# through -- it is truthy but missing the keys the captions read. Rendering
# "nan of 0 bins" would be worse than saying nothing, so report it as a
# MISSING section instead. (Before 2026-09-18 this path hard-indexed
# _d['nbins'] and killed the whole page with a KeyError, which is how the
# galfor log-sampling bug above surfaced.)
if DTR and "nbins" not in DTR:
    MISSING.append(
        "data/template/residual captions skipped: the DTR analysis did not "
        "complete (no 'nbins'), so its summary numbers do not exist. The "
        "panels above are whatever it managed to produce. Most likely cause "
        "is a NaN sensitivity curve -- check the galfor sampling basis.")
if DTR and "nbins" in DTR:
    _d = DTR
    cap_f1 = (
        f"Power spectral density of the TDI X channel &mdash; not a strain "
        f"amplitude, and not an ASD. The gap between the instrument curve and "
        f"their sum is the galactic confusion, at most a factor "
        f"{_d.get('conf_ratio', float('nan')):.2f} in power, near "
        f"{_d.get('conf_f', float('nan')) * 1e3:.1f} mHz. "
        f"{_d.get('n_gb', '?')} "
        f"galactic-binary and {_d.get('n_vgb', '?')} verification-binary "
        f"templates are subtracted here.")
    cap_f1b = (
        f"Lower panel: residual power over the fitted noise-plus-foreground "
        f"model, coloured by the Anderson&ndash;Darling Gaussianity p-value of "
        f"the whitened residual &mdash; dark bins are where the model is still "
        f"incomplete. The residual stays above the instrument-only curve in "
        f"{_d.get('nbins', 0) - _d.get('undersub', 0)} of "
        f"{_d.get('nbins', 0)} bins; "
        f"{_d['undersub']} dip below it, spanning "
        f"{_d.get('undersub_lo', float('nan')) * 1e3:.1f}&ndash;"
        f"{_d.get('undersub_hi', float('nan')) * 1e3:.1f} mHz, and the "
        f"deepest is {100 * (1 - _d.get('undersub_worst', 1)):.0f}% under. "
        f"Read that against the instrument model itself: the fitted "
        f"Soms sits {100 * NOISE_BIAS[0]:+.1f}% from injection, which is "
        f"{100 * ((1 + NOISE_BIAS[0]) ** 2 - 1):+.1f}% in power, so a curve "
        f"drawn a few percent high will sit above a correct residual over "
        f"exactly the band where that parameter dominates. Shortfalls of "
        f"that size are the noise model, not over-subtraction; a bin far "
        f"under would be a different statement.")
else:
    cap_f1 = cap_f1b = ""

# ---- captions for the restored data/template/residual panels --------------
# Condensed from the pre-redesign page: same numbers, read off the same
# arrays, without the method narrative that now lives in the appendix.
if DTR and "nbins" in DTR:
    _d = DTR
    dtr_fd_cap = (
        f"Rows are the TDI channels the run analyses; columns are data, "
        f"template sum and residual. Grey is the data, repeated faintly under "
        f"the other two columns so the comparison is direct; green is the "
        f"{_d['n_gb']} galactic-binary templates, violet the {_d['n_vgb']} "
        f"verification binaries, cyan the residual. Dotted red marks the "
        f"run's own band edges. At the loudest recovered source, "
        f"{_d['chk_f0']:.5f} mHz, the peak bin falls by a factor "
        f"{_d['chk_dpk'] / max(_d['chk_rpk'], 1e-99):.0f} and the power in the "
        f"surrounding 81 bins drops to "
        f"{100 * _d['chk_rp'] / max(_d['chk_dp'], 1e-99):.1f}% of the data. "
        f"Across the whole band only "
        f"{100 * (1 - _d['band_rp'] / max(_d['band_dp'], 1e-99)):.1f}% of the "
        f"power has been removed &mdash; the unresolved galaxy is still there.")
    dtr_wdm_cap = (
        f"The same three states on the run's own time-frequency grid: "
        f"{_d['wdm_shape'][1]} layers of {_d['layer_df'] * 1e3:.4f} mHz by "
        f"{int(_d['layer_dt'])} s, max-pooled {_d['wdm_dec']}&times; in time. "
        f"One shared linear scale per channel row, keyed to that row's data "
        f"panel, so a shrinking residual renders darker instead of "
        f"rescaling itself back to full brightness. The horizontal tracks with "
        f"annual brightness modulation are the recovered sources.")
    dtr_note = (
        f"Both are built for cold walker {_d['walker']}, the maximum-likelihood "
        f"walker of the last stored iteration. Noise and foreground are not in "
        f"the template sum &mdash; they shape the sensitivity the likelihood "
        f"weights by, they are not subtracted, so the unresolved galaxy stays "
        f"in the residual by construction.")
else:
    dtr_fd_cap = dtr_wdm_cap = dtr_note = ""

NOISE_TXT = " and ".join(f"{100 * b:+.1f}%" for b in NOISE_BIAS)

# ---- ONE run-health line, in place of ~1,900 words of failure forensics ----
# The OOM, cap-ramp, ghost-guard and Doppler-offset investigations that used to
# open this page are engineering history: they belong in the run log and in the
# tracker, not in the first thing a collaborator reads. What a reader needs
# from the top of a status page is how far each arm got and whether to trust it
# as converged.
_arm_bits = []
for _t in sorted(globals().get("ARMS", {})):
    _D = ARMS[_t]
    _arm_bits.append(f"{_t} has completed "
                     f"{int(_D['n_match'].size) - 1 - int(_D['it0'])} "
                     f"galactic-binary search iterations")
_ended = ("ended at iteration 80 on a GPU memory limit"
          if RUN_KIND == "3mo" and NIT >= 80 else
          f"has stored {NIT} iterations")
RUN_HEALTH = (
    f"<strong>Run health.</strong> This arm {_ended}. "
    + ("; ".join(_arm_bits) + ". " if _arm_bits else "")
    + "Neither arm has converged, so every number here is a progress readout "
      "rather than a result.")

missing_html = "".join(f"<li>{m}</li>" for m in MISSING)


# ---- match-criterion fragments (SHOW_MATCH_STATS gate) --------------------
if SHOW_MATCH_STATS:
    KPI_MATCH = f"""  <div><b>{SCI.get("n_match", 0):,}</b><span>matched to an injection</span></div>
  <div><b>{pct(SCI["completeness"]) if SCI else "&mdash;"}</b><span>completeness</span></div>
  <div><b>{pct(SCI["purity"]) if SCI else "&mdash;"}</b><span>purity</span></div>"""
    REC_MATCH_PANELS = f"""<div class="panel">{img("f2_progress", "completeness and purity vs GB-search iteration")}
<div class="caption">{cap_f2}</div></div>
<div class="panel">{img("f3_match", "overlap CDF and survival count")}
<div class="caption">{cap_f3}</div></div>
<div class="panel">{img("f6_snr", "completeness vs SNR")}
<div class="caption">{cap_f6}</div></div>"""
    PARAMS_SECTION = f"""<section id="params"><h2>Parameter Recovery</h2>
<div class="panel">{img("f7_params", "recovered minus injected parameters")}
<div class="caption">{cap_f7}<br><em>Distance, chirp mass and the frequency-derivative
ratio are not shown: the likelihood constrains only their combinations, so scatter
along that direction is degeneracy, not error.</em></div></div>
<div class="panel">{img("f7_scatter", "recovered vs injected")}
<div class="caption">The same matched sources as recovered against injected, with the
diagonal. The histograms above size the error; this asks whether the parameter is
constrained at all. Inclination correlates at {SCI.get("ci_corr", float("nan")):.2f}.
<em>Our encoding, not a field convention.</em></div></div>
</section>"""
    CENSUS_PANEL = f"""<div class="panel">{img("gb_hi_f_census", "high-frequency recovery census")}
<div class="caption">{CENSUS_TXT} Left is the raw census: every catalogue source above the
cut, green where the maximum-likelihood walker holds a leaf on it. Middle is the health
test &mdash; recovery must be monotonic in signal-to-noise, and it is, which says adding is
signal-ordered rather than arbitrary. Right is the ceiling: bars to the right of the dashed
line are cells holding more detectable sources than the cap allows.</div></div>"""
    F10_BLEND_NOTE = ("<br><em>Two blending modes sit at the two ends: below "
                      "the tolerance, several templates share one injection; "
                      "far above it, one template can still straddle a pair "
                      "of injections that the catalogue resolves.</em>")
    NAV_PARAMS = '<a href="#params">parameters</a>'
    OPEN_ITEMS = f"""<strong>Open items.</strong> Purity is {pct(SCI["purity"]) if SCI else "&mdash;"}
against a {pct(SCI["chance"], 1) if SCI else "2.2%"} chance rate, so the matches are
real, but {SCI.get("n_unmatched", 0)} model sources have no detectable counterpart and
{SCI.get("blend_b", 0)} of those sit on an injection another source already claims.
Neither arm has converged."""
else:
    KPI_MATCH = REC_MATCH_PANELS = PARAMS_SECTION = CENSUS_PANEL = ""
    F10_BLEND_NOTE = NAV_PARAMS = ""
    OPEN_ITEMS = (
        "<strong>Open items.</strong> Quantitative match-vs-catalogue "
        "statistics are intentionally absent from this page: the physical "
        "phase-maximised overlap match is computed offline, and the page's "
        "own 2-bin frequency proxy is not quoted as a number. The catalogue "
        "truths appear in the visual overlays only. The run has not "
        "converged.")

# ---- "How Many Are Detectable At All" table (live, this Tobs) --------------
# The block below builds the two columns of the #detect table at this run's
# actual observation time -- until 2026-09-20 the table was HARDCODED to the
# 3-month values (1,001 / 1,103 at SNR>7), so a 1-year page silently claimed
# the 3-month numbers as if they were current.
#   Column 1 (fitted noise): free -- TRU already carries per-source SNR under
#     the run's own fitted PSD + foreground at this Tobs. Just count.
#   Column 2 (injected + FittedHyperbolicTangent): re-run the catalogue
#     optimal-SNR calculation against a fixed reference noise -- the injected
#     instrument (SOMS_INJ, SA_INJ) plus the lisatools
#     FittedHyperbolicTangentGalacticForeground, both functions of Tobs alone.
#     Reuses build_truth.py's opt_snr helper for consistency.
DETECT_CUTS = (5, 7, 10, 15)
_col1 = _col2 = None
_col2_err = None
if TRU is not None:
    try:
        _snr_fit = np.asarray(TRU["snr"], float)
        _col1 = {c: int((_snr_fit > c).sum()) for c in DETECT_CUTS}
    except Exception as _e:
        MISSING.append(f"#detect column 1 (fitted noise) unavailable: "
                       f"{type(_e).__name__}: {_e}")

    # Column 2 -- compute lazily. Skipped when GBGPU can't import or when
    # any prerequisite is missing; the caption below reflects the state.
    try:
        import sys as _sys
        _bt_dir = os.path.dirname(os.path.abspath(__file__))
        if _bt_dir not in _sys.path:
            _sys.path.insert(0, _bt_dir)
        from build_truth import opt_snr as _opt_snr_bt   # local sibling

        from gbgpu.gbgpu import GBGPU as _GBGPU2
        from lisatools import detector as _lm2
        from lisatools.sensitivity import (
            get_sensitivity as _gs2, A2TDISens as _A2T, E2TDISens as _E2T)
        from lisatools.stochastic import (
            FittedHyperbolicTangentGalacticForeground as _FHT)
        from lisatools.globalfit.stock.erebor.variants.gb_no_fg import (
            GB_MOJITO_T_REF as _T_REF_BT)

        _lm_inj = _lm2.LISAModel(SOMS_INJ ** 2, SA_INJ ** 2,
                                 _lm2.DefaultOrbits(force_backend="cpu",
                                                    frame="icrs"),
                                 "injected")
        _nk_inj = dict(model=_lm_inj, stochastic_params=(SCI_TOBS,),
                       stochastic_function=_FHT)
        _ng2 = int(2.35e-2 / SCI_DF) + 2
        _fg2 = np.maximum(np.arange(_ng2) * SCI_DF, SCI_DF)
        _sa2 = np.asarray(_gs2(_fg2, sens_fn=_A2T, **_nk_inj), float)
        _se2 = np.asarray(_gs2(_fg2, sens_fn=_E2T, **_nk_inj), float)
        _nw2 = int(TRU["nw"]) if "nw" in getattr(TRU, "files", []) else 128
        _phys2 = np.asarray(TRU["phys"], float)
        _orb2 = globals().get("_L1_ORB_CPU")
        if _orb2 is None:
            _orb2 = _lm2.DefaultOrbits(force_backend="cpu", frame="icrs")
        _gbw2 = _GBGPU2(force_backend="cpu", orbits=_orb2,
                        t0=float(_T_REF_BT))
        _snr_inj = _opt_snr_bt(_phys2, _sa2, _se2, _gbw2, SCI_DF,
                               SCI_TOBS, _nw2, batch=20000)
        _col2 = {c: int((_snr_inj > c).sum()) for c in DETECT_CUTS}
    except Exception as _e:
        _col2_err = f"{type(_e).__name__}: {_e}"
        MISSING.append(f"#detect column 2 (injected + FittedHT) unavailable: "
                       f"{_col2_err}")

def _detect_tbl_row(cut):
    _bold = (cut == 7)
    _tc = "font-weight:bold" if _bold else ""
    _v1 = (f"{_col1[cut]:,}" if _col1 is not None else "&mdash;")
    _v2 = (f"{_col2[cut]:,}" if _col2 is not None else "&mdash;")
    _lb = f"<strong>{cut}</strong>" if _bold else f"{cut}"
    _cs = "padding:3px 18px 3px 0"
    return (f'<tr><td style="{_cs}">{_lb}</td>'
            f'<td style="text-align:right;{_cs};{_tc}">{_v1}</td>'
            f'<td style="text-align:right;{_tc}">{_v2}</td></tr>')

TBL_DETECT_ROWS = "\n".join(_detect_tbl_row(c) for c in DETECT_CUTS)
_col2_note = ("" if _col2 is not None else
              f"<br><em>Column 2 not computed in this snapshot "
              f"({'no TRU npz' if TRU is None else _col2_err}).</em>")

html = f"""<title>LISA Global Fit {RUN_LABEL}</title>
<style>
:root {{
  --bg:#0A0E14; --panel:#10161F; --line:#223041; --fg:#B8C6D4; --dim:#67788A;
  --cyan:#4FD8EB; --amber:#F5A623; --green:#58C48A; --red:#E5484D; --violet:#9B7BFF;
  /* Catalogue-truth marks only. BRIGHT, not deep (2026-08-19): the earlier
     #C41220 was chosen to keep 30k crosses from reading as a pink haze, but
     on the #0A0E14 panel it went muddy and the crosses were unreadable
     against the green recovery circles. Legibility of the overlay wins --
     the haze worry is handled by the per-cross alpha instead. */
  --truthred:#FF2E3E;
  /* Sub-threshold catalogue sources on the zoom canvas. Light enough to
     read as a population against --bg, dark enough that the red
     detectable crosses stay the thing the eye lands on. */
  --truthgrey:#5A6878;
}}
:root[data-theme="light"] {{
  --bg:#EEF1F5; --panel:#FFFFFF; --line:#D4DBE3; --fg:#25313D; --dim:#5D6B7A;
  --truthred:#E00016;
  --truthgrey:#9AA7B4;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; margin:0; }}
header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
  padding:10px 20px; z-index:5; display:flex; flex-wrap:wrap; gap:8px 18px; align-items:baseline; }}
header h1 {{ font-size:15px; margin:0; letter-spacing:.06em; color:var(--cyan); text-transform:uppercase; }}
header .stamp {{ color:var(--dim); font-size:12px; }}
nav {{ display:flex; flex-wrap:wrap; gap:6px; padding:8px 20px; border-bottom:1px solid var(--line); }}
nav a {{ color:var(--dim); text-decoration:none; font-size:12px; padding:2px 8px; border:1px solid var(--line); border-radius:3px; }}
nav a:hover, nav a:focus {{ color:var(--cyan); border-color:var(--cyan); outline:none; }}
main {{ max-width:1240px; margin:0 auto; padding:16px 20px 60px; }}
section {{ margin-top:28px; }}
h2 {{ font-size:13px; letter-spacing:.1em; text-transform:uppercase; color:var(--fg);
  border-bottom:1px solid var(--line); padding-bottom:6px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:12px; margin-top:12px; overflow-x:auto; }}
.panel img {{ max-width:100%; display:block; margin:0 auto; }}
.caption {{ color:var(--dim); font-size:12px; margin-top:6px; }}
.chip {{ display:inline-block; border:1px solid var(--line); border-radius:3px; padding:2px 9px; font-size:12px; color:var(--dim); }}
.chip.done {{ color:var(--green); border-color:var(--green); }}
.chip.now {{ color:var(--amber); border-color:var(--amber); }}
.kpi {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; }}
.kpi div {{ background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:8px 14px; }}
.kpi b {{ display:block; font-size:18px; color:var(--cyan); font-variant-numeric:tabular-nums; }}
.kpi span {{ font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:.05em; }}
.alert {{ background:color-mix(in srgb, var(--red) 12%, var(--panel)); border:1px solid var(--red);
  border-radius:4px; padding:12px 14px; margin-top:14px; font-size:13px; }}
.missing {{ color:var(--amber); border:1px dashed var(--amber); border-radius:4px; padding:10px; font-size:12px; }}
code {{ color:var(--cyan); }}
canvas {{ background:var(--panel); border:1px solid var(--line); border-radius:4px; width:100%; height:380px; display:block; touch-action:none; cursor:grab; }}
.btnrow {{ display:flex; gap:8px; margin:8px 0; }}
button {{ background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:3px;
  padding:4px 10px; font:12px ui-monospace,monospace; cursor:pointer; }}
button:hover, button:focus {{ border-color:var(--cyan); color:var(--cyan); outline:none; }}
button.armed {{ border-color:var(--amber); color:var(--amber); }}
.viewctl {{ flex-wrap:wrap; align-items:center; gap:6px 12px; }}
.viewctl label {{ color:var(--dim); font-size:11px; display:inline-flex; align-items:center; gap:4px;
  text-transform:uppercase; letter-spacing:.04em; }}
.viewctl input[type=text] {{ background:var(--bg); color:var(--fg); border:1px solid var(--line);
  border-radius:3px; font:11px ui-monospace,monospace; padding:2px 4px; width:86px; }}
.viewctl input[type=text]:focus {{ border-color:var(--cyan); outline:none; }}
.viewctl input[type=range] {{ width:110px; accent-color:var(--cyan); }}
.viewctl select {{ background:var(--bg); color:var(--fg); border:1px solid var(--line);
  border-radius:3px; font:11px ui-monospace,monospace; padding:2px 4px; max-width:520px; }}
.viewctl select:focus {{ border-color:var(--cyan); outline:none; }}
button.armed.truth {{ border-color:var(--dim); color:var(--dim); }}
ul {{ color:var(--dim); font-size:13px; }}
</style>
<header>
  <h1>LISA Global Fit &middot; {RUN_LABEL} Status</h1>
  <span class="stamp">{os.path.basename(RUN_DIR)} &middot; {datetime.now():%Y-%m-%d}</span>
  <span>{chips}</span>
</header>
<nav>
  <a href="#status">status</a><a href="#resid">residual</a>
  <a href="#recovery">recovery</a><a href="#population">population</a>
  {NAV_PARAMS}<a href="#search">search &amp; cap cells</a>
  <a href="#fstat">f-stat</a><a href="#noise">noise</a>
  <a href="#vgb">verification binaries</a><a href="#detect">detectability</a>
  <a href="#appendix">appendix</a>
</nav>
<main>
{COMMENTARY}
<section id="status"><h2>Status</h2>
<div class="kpi">
  <div><b>{SCI.get("ngbit", 0)}</b><span>GB search iterations</span></div>
  <div><b>{SCI.get("n_all", int(gb_counts[-1].max())):,}</b><span>model sources</span></div>
{KPI_MATCH}
  <div><b>{VGB_N_DET}</b><span>verification binaries above SNR 7</span></div>
</div>
<p style="font-size:13px">{RUN_HEALTH}</p>
<p style="font-size:13px"><strong>The denominator, stated once.</strong> Every
recovery number on this page is against <strong>{SCI.get("ndet", 812)} galactic
binaries</strong> &mdash; those in the injected catalogue with optimal
signal-to-noise above 7 over {BAND_TXT}, the full band the sampler analyses,
evaluated under this run&rsquo;s own fitted noise. A model source counts as a
recovery when it lies within
{TOL_BINS:.0f} frequency bins ({TOL_BINS / SCI_TOBS * 1e6:.3f} &micro;Hz) of one,
one-to-one. Those windows cover {pct(SCI["chance"], 1) if SCI else "2.2%"} of the
band, so that is the rate at which an arbitrary source would match by accident.
Below roughly 3 mHz that threshold is an SNR statement and not a resolvability
one: the sensitivity it is measured against already carries the fitted galactic
foreground, but the catalogue puts many sources in every frequency bin down
there, so a binary can clear signal-to-noise 7 and still be inseparable from
its neighbours.</p>
{ARM_TABLE}
<!-- TRACKERS_TOP: the two headline per-iteration trackers live here, first
     thing after the KPIs (moved from Search & Cap Cells, user request
     2026-08-22 x2 -- they were invisible deep in a 12 MB page). -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start">
<div class="panel">{img("gb_leaves")}
<div class="caption">Left: galactic-binary leaf count per cold walker across the stored
iterations, against the detectable-injection target. Right: the enforced per-cell leaf
cap over time. Rows marked in red have had their births shut off by the barren-band rule
&mdash; deaths and in-model moves continue there.</div></div>
<div class="panel">{img("ll")}
<div class="caption">Cold-chain total log-likelihood across the {nwalk} walkers, and
the max-minus-min spread. At equilibrium the spread sits at a few units.</div></div>
</div>
</section>

<section id="resid"><h2>Residual Spectrum</h2>
<div class="panel">{img("f1_resid", "residual spectrum")}
<div class="caption">{cap_f1}<br>{cap_f1b}</div></div>
<div class="panel">{img("dtr_fd", "data / template / residual, frequency domain")}
<div class="caption">{dtr_fd_cap}</div></div>
<div class="panel">{img("dtr_wdm", "data / template / residual, time-frequency")}
<div class="caption">{dtr_wdm_cap}</div></div>
<div class="caption">{dtr_note}</div>
<div class="caption">Data are the mojito Level-1 products the run itself loaded,
re-transformed with the run&rsquo;s own window; templates are the last stored
cold-chain coordinates of the highest-likelihood walker through the run&rsquo;s own
transform and waveform generator. The noise branches shape the sensitivity the
likelihood weights by and are never subtracted, so the unresolved galaxy stays in
the residual by construction.</div>
</section>

<section id="recovery"><h2>Recovery</h2>
{REC_MATCH_PANELS}
<div class="panel">{img("f5_counts", "source counts vs frequency")}
<div class="caption">{cap_f5}</div></div>
</section>

<section id="population"><h2>Recovered Population</h2>
<div class="panel">{img("f4_pop", "amplitude vs frequency, and the three-way recovery split")}
<div class="caption">{cap_f4}</div></div>
<div class="panel">{img("f8_sky", "sky distribution")}
<div class="caption">{cap_f8}</div></div>
<div class="panel">{img("f10_nn", "nearest-neighbour separation")}
<div class="caption">{cap_f10}{F10_BLEND_NOTE}</div></div>

<div class="caption" style="margin-top:18px"><strong>Zoom in.</strong> The static panels above are the whole band at once; these two are pannable and zoomable, which is the only way to read an individual galactic binary against its injected counterpart. The scatter pools the last {EXPL_ITS} stored iterations of the cold chain; the corner posteriors under it pool {gb1_meta.get("corner_its", 0)}. Every pooled iteration is masked by its OWN <code>inds</code> row &mdash; the branch is trans-dimensional, so a leaf alive now was not necessarily alive four iterations ago.</div>
<div class="panel">
<div class="btnrow">
  <button id="btn_all">full band</button>
  <button id="btn_top3">highest-frequency sources</button>
  <button id="btn_truth">show catalogue</button>
  <button id="btn_reset">reset zoom</button>
  <span class="caption" style="align-self:center">drag to pan &middot; wheel to zoom</span>
</div>
<div class="btnrow viewctl">
  <button id="expl_pick" title="arm, then click the plot to set the view center">set center by click</button>
  <label>cx <input id="expl_cx" type="text"></label>
  <label>cy <input id="expl_cy" type="text"></label>
  <label>width <input id="expl_wsl" type="range" min="0" max="1000" step="1"><input id="expl_w" type="text"></label>
  <label>height <input id="expl_hsl" type="range" min="0" max="1000" step="1"><input id="expl_h" type="text"></label>
</div>
<canvas id="expl"></canvas>
<div class="caption" id="expl_cap"></div>
<div class="caption">Zoomable version of the amplitude-frequency plane, with
four classification layers on top of the posterior cloud:
<span style="color:var(--amber)">amber</span> = pooled posterior samples of
the current model,
<span style="color:var(--dim)">grey X</span> = undetectable catalogue rows,
<span style="color:var(--truthred)">red X</span> = detectable but not
recovered under the current match criterion,
<span style="color:var(--green)">closed green circle</span> = recovered and
matched,
<span style="color:var(--violet)">open violet circle</span> = recovered with
no matching injection. Drag to pan and use the wheel to zoom, or set the
view numerically with the centre and width/height controls above &mdash;
those hold the window size fixed and slide it across the band, which is the
steadier way to walk through frequency.
<strong>Amber is pooled over the last {EXPL_ITS} stored iterations</strong>
&times; {nwalk} cold walkers ({EXPL_RAW:,} alive-leaf rows{EXPL_DEC_NOTE}),
with each iteration&rsquo;s own alive mask applied, so a single source draws a
cloud whose width is the sampler&rsquo;s spread rather than one snapshot of
where the walkers happened to sit. The catalogue overlay is decimated per
frequency window on the undetectable rows; every detectable-not-recovered
row is drawn in full. {MATCH_CRIT_HTML}</div>
</div>

<div class="panel">
<div class="btnrow viewctl">
  <label>source <select id="gb1_sel"></select></label>
  <span class="caption" style="align-self:center">posteriors of the highest-frequency
  recovered galactic binaries</span>
</div>
<img id="gb1_img" alt="galactic binary corner posterior">
<div class="caption" id="gb1_cap"></div>
<div class="caption">Sources are clustered out of the last stored iteration by
frequency ({CLUSTER_BINS:.0f} bins) and counted only if they appear in at least three
of the {nwalk} cold walkers; {gb1_meta.get("n_solid", 0)} of
{gb1_meta.get("n_clusters", 0)} clusters clear that bar. Catalogue values are shown
where a source lies within {MATCH_BINS:.0f} bins. The corner itself is built from a
wider window than that clustering step &mdash; up to the last
{gb1_meta.get("corner_its", 0)} stored iterations, pooled by frequency association
because the GB branch is trans-dimensional and a leaf index does not track one
source across iterations. Contours are 1, 2 and 3 sigma; below 200 pooled rows the
kernel is widened deliberately, so read these as where the walkers currently sit
rather than as credible intervals.</div>
</div>

</section>

{PARAMS_SECTION}

<section id="search"><h2>Search &amp; Cap Cells</h2>
<div class="panel">{img("gb_birth_fate", "birth-fate breakdown")}
<div class="caption">Where every trans-dimensional birth proposal ends up. The fates are
disjoint and sum to the proposed count. <span style="color:var(--amber)">Amber</span> is
gated before scoring &mdash; the cell already holds its allowance, or the draw is out of
band or out of prior &mdash; cheap rejections that never touch a likelihood kernel.
<span style="color:var(--red)">Red</span> is scored and then dropped at the optimal-SNR
clamp. Grey is scored, offered to Metropolis&ndash;Hastings and rejected.
<span style="color:var(--green)">Green</span> is accepted, i.e. a new source. Left is
absolute counts, right the same data as percentages so the trend stays readable as the
model fills. {GB_FATE_TXT}</div></div>
<!-- gb_leaves + ll grid MOVED to the top of the page (user request
     2026-08-22, twice: on a 12 MB page these two headline trackers were
     effectively invisible this deep in the scroll). See TRACKERS_TOP. -->
<div class="panel">{img("timing_moves", "per-move throughput")}
<div class="caption">Proposal throughput and wall time per propose, per move, against
elapsed run time.</div></div>
<div class="panel">{img("gb_cap_cells", "cap-cell occupancy")}
<div class="caption">{CAP_TXT} Left is the direct test of whether the cap is being
respected: bars at or above the cap are amber, and a bar past it would mean sources are
stacking. Middle is the race that matters &mdash; occupied cells must stay ahead of cells
at their cap, or the model is queuing against the ceiling rather than filling. Right
explains why the occupied fraction looks small: the cells tile the whole band uniformly
while the sources are concentrated in the galaxy, so most cells are empty because there is
nothing in them yet.</div></div>
<div class="panel">{img("gb_cap_divisor", "cap-divisor study")}
<div class="caption">What the cap grid can represent, independent of how far this run has
got: summing the detectable sources a cell cannot admit gives the ceiling the sampler can
never beat. A finer grid beats simply raising the cap, because a higher cap permits several
sources inside one cell, which is the stacking the rule exists to stop.</div></div>
{CENSUS_PANEL}
</section>

<section id="fstat"><h2>F-statistic Fit</h2>
<div class="panel">{img("fstat_comb")}
<div class="caption">Comb scan of the maximised F-statistic across the band
(epoch {fstat_meta.get("epoch", "?")}), fitted against the live residual. This is what the
birth proposal draws from.</div></div>
<div class="panel">{img("fstat_peaks")}
<div class="caption">The selected peaks are the birth-proposal anchors. Left is where they
sit in the plane; right is their density against frequency &mdash; the shape that decides
where the search spends its proposals.</div></div>
</section>

<section id="noise"><h2>Noise Model</h2>
<div class="panel">{img("f11_psd", "instrument noise posteriors")}
<div class="caption">The two instrument-noise parameters against their injected
values. Medians sit {NOISE_TXT} from injection. These are the only noise parameters
with a truth to compare against.</div></div>
<div class="panel">{img("f11_fg", "foreground evolution")}
<div class="caption">The fitted noise-plus-foreground curve at every stored
iteration, light to dark with time, over the instrument-only curve. The galactic
shoulder should walk down as resolved sources leave the residual.</div></div>
<div class="panel">{img("psd_curves", "sensitivity curves")}
<div class="caption">The same model as the sky-averaged sensitivity the mission documents
quote: instrument only, instrument plus the fitted foreground, and the injected instrument
curve. This is the only panel carrying the injected curve.</div></div>
<div class="panel">{img("psd_evolution", "sensitivity evolution")}
<div class="caption">The decline watch on the same axes, one curve per stored iteration,
light to dark with time.</div></div>
<div class="panel">{img("psd_trace")}
<div class="caption">Instrument-parameter traces per cold walker; dotted red is the
injected value.</div></div>
<div class="panel">{img("psd_hist")}</div>
<div class="panel">{img("gal_trace")}
<div class="caption">The five foreground parameters. There is no truth line: the injection
is a source population, not a hyperbolic-tangent model, so these are checked through the
curve above and through the residual, never against a number.</div></div>
<div class="panel">{img("gal_hist")}</div>
</section>

<section id="vgb"><h2>Verification Binaries</h2>
<div class="panel">{img("f9_vgb", "detectable verification binaries")}
<div class="caption">The {VGB_N_DET} of 55 catalogue verification binaries that clear
SNR 7 at three months, with their distance posteriors against the catalogue value.
The other {55 - VGB_N_DET} are prior-dominated. Full-55 SNR-vs-frequency, all-55
distance posteriors, the SNR-per-iteration trend, and the interactive zoom cloud were
removed for the 16 MB size budget; the median VGB SNR is {VGB_SNR_MED:.1f} and the
{55 - VGB_N_DET} prior-dominated leaves are an absence of signal, not a fit failure.</div></div>
<div class="panel">{img("vgb_hists")}
<div class="caption">The four remaining sampled parameters pooled over all 55 leaves, so
the truth is a distribution rather than a line. The frequency-derivative ratio is the
exception: every catalogue value is identically zero.</div></div>
<div class="panel">
<div class="btnrow viewctl">
  <label>source <select id="vgb1_sel"></select></label>
  <span class="caption" style="align-self:center">full posterior, one verification
  binary at a time &mdash; all 55, detectable first</span>
</div>
<img id="vgb1_img" alt="verification binary corner posterior">
<div class="caption" id="vgb1_cap"></div>
<div class="caption">Existence proof that the machinery produces real posteriors:
five sampled parameters over the last {CORNER_ITS} stored iterations &times; {nwalk}
cold walkers, with the catalogue value in cyan. Axis ranges are widened to contain
the truth, so a truth line outside the posterior stays visible.</div>
</div>
</section>

<section id="detect"><h2>How Many Are Detectable At All</h2>
<div class="panel">
<div class="caption" style="margin:0 0 10px 0">Optimal SNR of the injected
catalogue over {BAND_TXT} at this run&rsquo;s observation time
({SCI_TOBS / 86400.0:.4g} d), under two noise models:
<em>col 1</em> = the run&rsquo;s own fitted instrument + foreground at the
truth-build iteration; <em>col 2</em> = the injected instrument
(S<sub>oms</sub>, S<sub>a</sub>) plus the lisatools
<code>FittedHyperbolicTangentGalacticForeground</code> at this T<sub>obs</sub>.
Both columns re-compute at every snapshot render &mdash; they do not carry
over from previous pages.{_col2_note}</div>
<table style="border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums">
<tr style="border-bottom:1px solid var(--line)">
  <th style="text-align:left;padding:4px 18px 4px 0">SNR &gt;</th>
  <th style="text-align:right;padding:4px 18px 4px 0">fitted noise (col 1)</th>
  <th style="text-align:right;padding:4px 0">injected + FittedHT (col 2)</th></tr>
{TBL_DETECT_ROWS}
</table>
<div class="caption" style="margin-top:12px">Detectability dies below about 1 mHz,
where the foreground swamps everything. The two models disagree by ~10% and, more
usefully, in opposite directions either side of the galactic peak &mdash; treat
column 2 as a reference point, not truth.
<br><br>
Detectability is also a moving target: the same calculation gives
<strong>{SCI.get("ndet", 0)}</strong> over {BAND_TXT} at this run&rsquo;s
late-iteration noise, and it moved as the foreground estimate dropped and sources
left the residual. That is exactly why the denominator on this page is frozen at
one iteration and stated.</div>
</div>
</section>

<section id="appendix"><h2>Appendix</h2>
<details><summary style="cursor:pointer;color:var(--dim);font-size:13px">
method, sampler health, interactive views and run mechanics</summary>

<div class="caption"><strong>How the numbers are produced.</strong> Optimal SNRs and
template overlaps use the injected catalogue&rsquo;s own parameters through the
run&rsquo;s catalogue-to-sampling map, the same waveform generator the run samples
with, and a noise-weighted inner product against the run&rsquo;s own fitted
instrument and foreground. Catalogue detectability is computed by bounding
signal-to-noise per unit amplitude over orientation, which can only over-include, then
evaluating exact SNRs for the survivors; an audit of 400 rejected sources weighted
toward the cut found a loudest SNR of 4.36 and none above 7. Overlaps and parameter
comparisons on this page are recomputed at the last stored iteration, not carried over
from earlier snapshots.
<br><br>
{OPEN_ITEMS}</div>

<div class="caption" style="margin-top:22px"><strong>Run mechanics.</strong> Engineering
instrumentation &mdash; where the wall time goes, what the devices are holding, and whether
the tempering ladder is exchanging. None of it is a science result; all of it is the first
thing to look at when a run stalls.</div>
{rj_kpis}
<div class="panel">{img("swaps")}
<div class="caption">Tempering swap acceptance per rung for the two noise branches, at each
branch&rsquo;s last active iteration &mdash; a stored iteration can record zero proposals
for one branch at a stage handoff.</div></div>
<div class="panel">{img("rj_breakdown", "trans-dimensional move breakdown")}
<div class="caption">Wall-time breakdown of the last complete record of each
trans-dimensional move, leaf spans only; the enclosing phase marks are quoted in each
title instead of drawn, since they merely reprint the total.{rj_break_txt}</div></div>
<div class="panel">{img("mem_telemetry", "device memory")}
<div class="caption">Device-wide memory from the in-run telemetry, with breaks across
restart gaps so an attempt boundary does not draw a false ramp.</div></div>
<div class="panel">{img("gpu_util", "gpu telemetry")}
<div class="caption">Utilisation and memory sampled by nvidia-smi; dotted traces are
earlier jobs.</div></div>

<div class="caption"><strong>Not reproduced in this snapshot:</strong></div>
<ul>{missing_html}</ul>
</details>
</section>
</main>

<script>
const DATA = {EXPL_JSON};
const VPOST = {VGB_POST_JSON};
const VGBC = {VGB_CORNER_JSON};
const GB1 = {GB1_JSON};
// Single-VGB corner panel: the <select> swaps the src of ONE <img> between
// 55 pre-rendered ChainConsumer PNGs (base64 data URIs held in JS, so only
// the selected corner is ever in the document's visible flow).
function cornerPanel(px, blob) {{
  const im = document.getElementById(px + "_img"),
        sel = document.getElementById(px + "_sel"),
        cap = document.getElementById(px + "_cap");
  if (!im || !sel) return;
  if (!blob.src || !blob.src.length) {{
    im.remove();
    if (cap) cap.textContent = "no corner plots available in this snapshot";
    return;
  }}
  blob.src.forEach((s, i) => {{
    const o = document.createElement("option");
    o.value = i; o.textContent = s.label; sel.appendChild(o);
  }});
  function show() {{
    const S = blob.src[+sel.value || 0];
    im.src = "data:image/png;base64," + S.png;
    im.alt = S.label;
    // the GB panel carries a catalogue-match note (red when the source has
    // no injected counterpart); the VGB panel leaves it empty
    const note = S.note
      ? ` &middot; <span style="color:var(${{S.bad ? "--red" : "--fg"}})">${{S.note}}</span>`
      : "";
    cap.innerHTML = S.sub + note;
  }}
  sel.onchange = show;
  show();
}}
cornerPanel("vgb1", VGBC);
cornerPanel("gb1", GB1);
// Shared view controls: numeric center (cx, cy) + log-scale width/height
// sliders, all around a FIXED center -- plus a click-to-set-center mode.
// api: get() -> [X0,X1,Y0,Y1]; set(x0,x1,y0,y1) (must redraw); fullW/fullH
// = the reset-view spans (slider range = [full/2000, full*1.2], log-mapped).
function viewCtl(px, cv, api) {{
  const el = id => document.getElementById(px + "_" + id);
  const cxI = el("cx"), cyI = el("cy"), wI = el("w"), hI = el("h"),
        wS = el("wsl"), hS = el("hsl"), pk = el("pick");
  if (!cxI) return {{ sync: () => {{}} }};
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const s2span = (s, full) => {{
    const mn = Math.log(full / 2000), mx = Math.log(full * 1.2);
    return Math.exp(mn + (mx - mn) * s / 1000);
  }};
  const span2s = (W, full) => {{
    const mn = Math.log(full / 2000), mx = Math.log(full * 1.2);
    return clamp(Math.round(1000 * (Math.log(W) - mn) / (mx - mn)), 0, 1000);
  }};
  let busy = false;
  function sync() {{
    if (busy) return;
    const [X0, X1, Y0, Y1] = api.get();
    cxI.value = ((X0 + X1) / 2).toPrecision(7);
    cyI.value = ((Y0 + Y1) / 2).toPrecision(7);
    wI.value = (X1 - X0).toPrecision(5);
    hI.value = (Y1 - Y0).toPrecision(5);
    wS.value = span2s(X1 - X0, api.fullW);
    hS.value = span2s(Y1 - Y0, api.fullH);
  }}
  function applyTyped() {{
    const [X0, X1, Y0, Y1] = api.get();
    let cx = parseFloat(cxI.value), cy = parseFloat(cyI.value);
    let W = parseFloat(wI.value), H = parseFloat(hI.value);
    if (!isFinite(cx)) cx = (X0 + X1) / 2;
    if (!isFinite(cy)) cy = (Y0 + Y1) / 2;
    if (!isFinite(W) || W <= 0) W = X1 - X0;
    if (!isFinite(H) || H <= 0) H = Y1 - Y0;
    busy = true;
    api.set(cx - W / 2, cx + W / 2, cy - H / 2, cy + H / 2);
    busy = false; sync();
  }}
  cxI.onchange = cyI.onchange = wI.onchange = hI.onchange = applyTyped;
  wS.oninput = () => {{
    const [X0, X1, Y0, Y1] = api.get();
    const cx = (X0 + X1) / 2, W = s2span(+wS.value, api.fullW);
    busy = true; api.set(cx - W / 2, cx + W / 2, Y0, Y1); busy = false;
    wI.value = W.toPrecision(5);
  }};
  hS.oninput = () => {{
    const [X0, X1, Y0, Y1] = api.get();
    const cy = (Y0 + Y1) / 2, H = s2span(+hS.value, api.fullH);
    busy = true; api.set(X0, X1, cy - H / 2, cy + H / 2); busy = false;
    hI.value = H.toPrecision(5);
  }};
  let picking = false;
  pk.onclick = () => {{ picking = !picking; pk.classList.toggle("armed", picking);
                        cv.style.cursor = picking ? "crosshair" : "grab"; }};
  // Capture-phase so an armed pick swallows the click before the pan handler.
  cv.addEventListener("pointerdown", e => {{
    if (!picking) return;
    e.stopImmediatePropagation(); e.preventDefault();
    const r = cv.getBoundingClientRect();
    const w = cv.clientWidth, h = cv.clientHeight;
    const ml = 56, mb = 30, mt = 8, mr = 10;
    const [X0, X1, Y0, Y1] = api.get();
    const cx = X0 + ((e.clientX - r.left) - ml) / (w - ml - mr) * (X1 - X0);
    const cy = Y0 + ((h - mb) - (e.clientY - r.top)) / (h - mb - mt) * (Y1 - Y0);
    const W = X1 - X0, H = Y1 - Y0;
    api.set(cx - W / 2, cx + W / 2, cy - H / 2, cy + H / 2);
    picking = false; pk.classList.remove("armed"); cv.style.cursor = "grab";
    sync();
  }}, {{ capture: true }});
  sync();
  return {{ sync }};
}}
// vgbpost dist-f0 posterior cloud REMOVED 2026-09-05 (16 MB size budget); its
// canvas markup + reset/pick controls were removed, so the whole handler IIFE
// is deleted here to avoid dereferencing a null #vgbpost and killing later blocks.
(() => {{
  const cv = document.getElementById("expl"), cap = document.getElementById("expl_cap");
  const css = getComputedStyle(document.documentElement);
  const C = n => css.getPropertyValue(n).trim();
  const hasGB = DATA.gb.length > 0;
  // points: [x, y(=1/d), tag]
  const pts = hasGB ? DATA.gb : DATA.vgb;
  const xlab = hasGB ? "f0 [mHz]" : (DATA.vgb_axis || "VGB leaf index");
  // FOUR-CLASS OVERLAY (2026-09-20). truth_grey = undetectable catalogue,
  // truth_red = detectable catalogue that isn't recovered under the current
  // match criterion, sources = recovered sources (last iter) with a
  // matched/unmatched flag. Legacy DATA.truth is retained for the VGB
  // fallback path and any external consumers of this JSON blob.
  const T_GREY = DATA.truth_grey || DATA.truth || [];
  const T_RED  = DATA.truth_red  || [];
  const SRC    = DATA.sources    || [];
  const N_OVERLAY = T_GREY.length + T_RED.length + SRC.length;
  const MATCH_NOTE = DATA.match_note || "";
  // DEFAULT ON. The classification overlay IS the point of this panel --
  // catalogue completeness cannot be read off the recovered cloud alone.
  let showT = N_OVERLAY > 0;
  const baseCap = (hasGB
    ? `GB samples: ${{DATA.gb.length}} alive-source rows pooled over the last ${{DATA.gb_its}} stored iterations x all cold walkers${{DATA.gb_stride > 1 ? ` (1-in-${{DATA.gb_stride}} of ${{DATA.gb_raw}} for page weight)` : ""}}; y = log10 amplitude from (dist, f0, Mc). Posterior cloud = amber; catalogue: grey X undetectable, red X detectable-not-recovered; recovered source markers: closed green circle (matched) or open violet circle (not matched). ${{MATCH_NOTE}}`
    : `No GB sources alive yet - showing the 55 VGBs (${{DATA.vgb_its}} stored iterations x ${{DATA.nwalk}} walker samples each) as 1/dist vs leaf index. GB samples take over automatically once births land.`);
  const setCap = () => {{
    cap.textContent = baseCap + (N_OVERLAY
      ? (showT ? " " + (DATA.truth_cap || "") : ` ${{N_OVERLAY.toLocaleString()}} classification points available - press "show catalogue & recovered".`)
      : " No catalogue/recovered overlay in this snapshot.");
  }};
  let X0, X1, Y0, Y1;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const pad = (a, b) => [(a - (b - a) * 0.05) , (b + (b - a) * 0.05)];
  const full = () => {{
    [X0, X1] = pad(Math.min(...xs), Math.max(...xs));
    [Y0, Y1] = hasGB ? pad(Math.min(...ys), Math.max(...ys)) : pad(0, Math.max(...ys));
  }};
  full();
  const FW = X1 - X0, FH = Y1 - Y0;
  let syncCtl = () => {{}};
  const dpr = window.devicePixelRatio || 1;
  function draw() {{
    const w = cv.clientWidth, h = cv.clientHeight;
    // RESIZE ONLY WHEN THE SIZE ACTUALLY CHANGES (2026-08-19). Assigning to
    // cv.width reallocates the backing store and resets all context state.
    // Doing it every frame -- while panning, at pointer-event rate -- was
    // the biggest cost in these canvases: a multi-megabyte buffer thrown
    // away and rebuilt per pointermove.
    const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (cv.width !== bw || cv.height !== bh) {{ cv.width = bw; cv.height = bh; }}
    const g = cv.getContext("2d");
    // setTransform, not scale: scale() would compound now that the buffer
    // (and with it the identity transform) survives between frames.
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = C("--panel"); g.fillRect(0, 0, w, h);
    const ml = 56, mb = 30, mt = 8, mr = 10;
    const sx = x => ml + (x - X0) / (X1 - X0) * (w - ml - mr);
    const sy = y => h - mb - (y - Y0) / (Y1 - Y0) * (h - mb - mt);
    g.strokeStyle = C("--line"); g.fillStyle = C("--dim"); g.font = "10px monospace";
    for (let i = 0; i <= 6; i++) {{
      const xv = X0 + (X1 - X0) * i / 6, yv = Y0 + (Y1 - Y0) * i / 6;
      g.beginPath(); g.moveTo(sx(xv), mt); g.lineTo(sx(xv), h - mb); g.stroke();
      g.beginPath(); g.moveTo(ml, sy(yv)); g.lineTo(w - mr, sy(yv)); g.stroke();
      g.fillText(xv.toPrecision(4), sx(xv) - 14, h - 12);
      g.fillText(yv.toPrecision(3), 4, sy(yv) + 3);
    }}
    g.fillStyle = C("--dim");
    g.fillText(xlab, w / 2 - 40, h - 2);
    g.save(); g.translate(10, h / 2); g.rotate(-Math.PI / 2);
    g.fillText(hasGB ? "log10 A" : "1 / dist [1/kpc]", -30, 0); g.restore();
    // LAYERING (2026-09-20). Draw order, bottom to top:
    //   1. grey X's for undetectable catalogue rows        (--dim)
    //   2. amber posterior cloud                          (--amber)
    //   3. red X's for detectable-not-recovered           (--truthred)
    //   4. green filled / violet open circles for recovered sources
    // Steps 1, 3, 4 are gated by `showT`. Step 2 always draws (that's the
    // "posterior of the current model" and reads even without an overlay).
    // The cloud sits between the two truth layers so the red misses stay
    // legible over crowded amber patches, but the grey undetectable rows
    // still show up under the cloud where the model has no sources.
    if (showT) {{
      // Layer 1: grey X's for undetectable catalogue rows.
      g.strokeStyle = C("--dim"); g.globalAlpha = 0.55;
      g.lineWidth = 1.5; g.lineCap = "round";
      const r = 3.4;
      g.beginPath();
      for (const p of T_GREY) {{
        const x = sx(p[0]), y = sy(p[1]);
        if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
        g.moveTo(x - r, y - r); g.lineTo(x + r, y + r);
        g.moveTo(x - r, y + r); g.lineTo(x + r, y - r);
      }}
      g.stroke();
    }}
    // ONE PATH, ONE FILL (2026-08-19). This used to open a path and issue a
    // separate fill() per point -- 5k+ draw calls per frame. Batching every
    // dot into a single path costs one fill. The moveTo before each arc is
    // required: without it consecutive arcs are joined by a straight line.
    g.globalAlpha = 0.55;
    // AMBER for the GB posterior cloud (2026-09-20): green now belongs to
    // the "recovered + matched" source markers layered on top. VGB fallback
    // keeps violet.
    g.fillStyle = hasGB ? C("--amber") : C("--violet");
    g.beginPath();
    for (const p of pts) {{
      const x = sx(p[0]), y = sy(p[1]);
      if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
      g.moveTo(x + 2.2, y); g.arc(x, y, 2.2, 0, 6.29);
    }}
    g.fill();
    g.globalAlpha = 1;
    if (showT) {{
      // red X: detectable but not recovered under the current criterion.
      g.strokeStyle = C("--truthred"); g.globalAlpha = 0.95;
      g.lineWidth = 1.6; g.lineCap = "round";
      const rr = 3.6;
      g.beginPath();
      for (const p of T_RED) {{
        const x = sx(p[0]), y = sy(p[1]);
        if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
        g.moveTo(x - rr, y - rr); g.lineTo(x + rr, y + rr);
        g.moveTo(x - rr, y + rr); g.lineTo(x + rr, y - rr);
      }}
      g.stroke();
      // recovered-source markers: filled green (matched) then open violet
      // (unmatched). Two passes so fill / stroke styles need not toggle
      // inside the loop.
      const rs = 3.2;
      g.globalAlpha = 0.95;
      g.fillStyle = C("--green");
      g.beginPath();
      for (const p of SRC) {{
        if (p[2] !== 1) continue;
        const x = sx(p[0]), y = sy(p[1]);
        if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
        g.moveTo(x + rs, y); g.arc(x, y, rs, 0, 6.29);
      }}
      g.fill();
      g.strokeStyle = C("--violet");
      g.lineWidth = 1.4;
      g.beginPath();
      for (const p of SRC) {{
        if (p[2] !== 0) continue;
        const x = sx(p[0]), y = sy(p[1]);
        if (x < ml || x > w - mr || y < mt || y > h - mb) continue;
        g.moveTo(x + rs, y); g.arc(x, y, rs, 0, 6.29);
      }}
      g.stroke();
      g.globalAlpha = 1;
    }}
    syncCtl();
  }}
  // COALESCE REDRAWS TO ANIMATION FRAMES (2026-08-19). pointermove fires at
  // up to the pointer's sample rate (120 Hz+ on a trackpad) and each event
  // used to force a full synchronous redraw of 30k truth crosses plus the
  // recovery cloud. Collapsing every burst of events into at most one draw
  // per frame is what makes dragging track the cursor instead of lagging
  // behind it. Slider/typed input still calls draw() directly -- viewCtl's
  // `busy` guard depends on sync() running inside its own call.
  let raf = 0;
  const redraw = () => {{
    if (raf) return;
    raf = requestAnimationFrame(() => {{ raf = 0; draw(); }});
  }};
  // pan/zoom
  let drag = null;
  cv.addEventListener("pointerdown", e => {{ drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); }});
  cv.addEventListener("pointermove", e => {{
    if (!drag) return;
    const w = cv.clientWidth, h = cv.clientHeight;
    const dx = (e.clientX - drag[0]) / (w - 66) * (X1 - X0);
    const dy = (e.clientY - drag[1]) / (h - 38) * (Y1 - Y0);
    X0 -= dx; X1 -= dx; Y0 += dy; Y1 += dy; drag = [e.clientX, e.clientY]; redraw();
  }});
  cv.addEventListener("pointerup", () => drag = null);
  // ZOOM ABOUT THE CURSOR, NOT THE VIEW CENTRE (2026-08-19). Centre-anchored
  // zoom is what made this canvas hard to drive: the source you were aiming
  // at slid out of frame as you zoomed, so reaching one binary meant
  // alternating zoom and pan several times. Anchoring on the pointer holds
  // whatever is under the cursor still, the way every map-style UI behaves.
  cv.addEventListener("wheel", e => {{
    e.preventDefault();
    const s = e.deltaY > 0 ? 1.15 : 0.87;
    const r = cv.getBoundingClientRect();
    const w = cv.clientWidth, h = cv.clientHeight;
    const ml = 56, mb = 30, mt = 8, mr = 10;
    const cl = v => Math.max(0, Math.min(1, v));
    // fraction across the plot box; clamped so a cursor sitting in the axis
    // margins anchors at the edge instead of flinging the view sideways
    const fx = cl((e.clientX - r.left - ml) / (w - ml - mr));
    const fy = cl((h - mb - (e.clientY - r.top)) / (h - mb - mt));
    const ax = X0 + fx * (X1 - X0), ay = Y0 + fy * (Y1 - Y0);
    X0 = ax + (X0 - ax) * s; X1 = ax + (X1 - ax) * s;
    Y0 = ay + (Y0 - ay) * s; Y1 = ay + (Y1 - ay) * s; redraw();
  }}, {{ passive: false }});
  document.getElementById("btn_all").onclick = () => {{ full(); draw(); }};
  document.getElementById("btn_reset").onclick = () => {{ full(); draw(); }};
  const bt = document.getElementById("btn_truth");
  if (!N_OVERLAY) bt.disabled = true;
  // reflect the ON default in the control the moment the page loads
  bt.classList.toggle("armed", showT); bt.classList.toggle("truth", showT);
  bt.textContent = showT ? "hide catalogue & recovered" : "show catalogue & recovered";
  bt.onclick = () => {{
    showT = !showT;
    bt.classList.toggle("armed", showT); bt.classList.toggle("truth", showT);
    bt.textContent = showT ? "hide catalogue & recovered" : "show catalogue & recovered";
    setCap(); draw();
  }};
  setCap();
  document.getElementById("btn_top3").onclick = () => {{
    const srt = [...pts].sort((a, b) => b[0] - a[0]);
    const top = srt.slice(0, Math.min(3 * DATA.nwalk, srt.length));
    const tx = top.map(p => p[0]), ty = top.map(p => p[1]);
    [X0, X1] = pad(Math.min(...tx), Math.max(...tx) || 1);
    [Y0, Y1] = hasGB ? pad(Math.min(...ty), Math.max(...ty)) : pad(0, Math.max(...ty) || 1);
    draw();
  }};
  syncCtl = viewCtl("expl", cv, {{
    get: () => [X0, X1, Y0, Y1],
    set: (a, b, c, d) => {{ X0 = a; X1 = b; Y0 = c; Y1 = d; draw(); }},
    fullW: FW, fullH: FH,
  }}).sync;
  new ResizeObserver(redraw).observe(cv);
  draw();
}})();
</script>
"""
open(OUT, "w").write(html)
print(f"wrote {OUT}: {len(html)//1024} KB ({len(html)/1024**2:.2f} MB), "
      f"{len(IMGS)} plots + {len(VGB_CORNER['src'])} VGB corners, "
      f"missing={len(MISSING)}")
