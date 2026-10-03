#!/usr/bin/env python
"""CD1L MBHB campaign: logL / mismatch / SNR of the MBH templates against the mojito data.

For ONE CD1L MBHB (catalogue row ``--src``, 0-19) and ONE analysis window (``--window``),
build up to four templates on the data grid and score each in the WDM domain against the
per-source mojito L1 stream (noise-free, source only; no time, phase or amplitude
maximisation). Noise weighting (mbh_harness.py: the EMRI harness's ``RunBox`` noise,
imported from scripts/emri/emri_batch_speed.py; recorded as ``noise`` / ``foreground`` /
``tobs_s`` in every row): SciRD v1 XYZ (TDI-2) PLUS the
FittedHyperbolicTangentGalacticForeground at Tobs = Nf*Nt*dt of the window's grid
(``--foreground on``, the default since 2026-10-02); ``--foreground off`` is the SciRD
v1 instrument alone, as every row before that date. Templates are scored through the
container slicing (``AnalysisContainer._slice_to_template``: the batched one on its
sub-box), not ``RunBox.score``:

* ``prod``     the production MBH_LIKELIHOOD=full path today: the stock
               ``PhenomTHMTDIWaveform`` from ``get_mbh_phenom_wave_gen`` with the stock
               ``SourceMBHSettings`` knobs (phentax T = MBH_WAVEFORM_DURATION, default
               1/12 yr = 30.4 d), the UNSNAPPED reference epoch (MOJITO_REFERENCE_TIME),
               one row through ``get_signals_for_residuals`` (the per-leaf stock call);
* ``prod90``   the stock generator as MBH_LIKELIHOOD=batched installs it for the engine's
               residual rebuilds and the move's cross-check: T = MBH_WINDOW_BEFORE_DAYS
               (90 d), lattice-snapped epoch through ``SnappedEpochMBHGen``;
* ``batched``  the windowed grid-aligned path (``WindowedGridAlignedMBHWaveform`` inside
               ``MBHWindowedWDMSignalGen``): kept box [merger - 90 d - 1 d, merger + 10 d +
               1 d] snapped outward to layers and CLAMPED to the data's active box, 4 d of
               discarded segment pad (``mbh_window_layers``), scored on its sub-box through
               the container's sub-box slicing, exactly as MBHBatchedLikeMove does;
* ``tof``      OPTIONAL, not in the defaults, NOT smoke-tested: ``MBHTDIonFly`` through the
               stock getter (``get_mbh_tdionfly_gen`` + ``MBHTDIonFlyWaveWrap``, t_start =
               MOJITO_REFERENCE_TIME, T = window + MBH_TDIONFLY_MARGIN). Its phentax span
               starts T before the merger, so the orbit ltt slice is widened to cover it.
All templates use the response order ``--order`` (default: SourceMBHSettings.response_order,
env MBH_RESPONSE_ORDER, 8 = the production default) and ``tukey_alpha = 0`` (data and
templates unwindowed; the edge crop removes the edges).

Windows (dt 2.5 s, Nf = ``--nf`` = 1440 -> 3600 s layers, edge crop ``--edge-crop`` = 60
layers per side as the production EDGE_CROP_WAVELETS, ``--min-freq``/``--max-freq`` 4e-4 /
2.5e-2 Hz by default -- the v8 launcher's; the v9 6-month launcher and
mbh_speed_durations.sh use 2.5e-4):

* ``6mo``      Nt 4320 (Tobs = TOBS_TARGET = 15552000 s). PLACED AS PRODUCTION: the stock
               all_sources mojito loader (L1ProcessingStepWithSyntheticNoise(Tobs=Nf*Nt*dt),
               ``window_start_offset`` never set, so 0; default_preprocess_kwargs skip the
               highpass / trim) keeps file samples [0, Nf*Nt): the window starts AT THE FILE
               START, data_t0 = the file's tdis t0.
* ``24mo``     likewise, Nt 17280 (4 x 15552000 s = 720 d of the 731-d files).
* ``<N>d``     likewise, N days from the file start: Nt = N * 24 one-hour layers (``180d`` is
               ``6mo``, ``720d`` is ``24mo``) -- the DAYS sweep ("180 360 720") of
               scripts/mbh/mbh_speed_durations.sh's data step. The bricks' data and ltts end
               REF + 730.5 d and the EMRI harness's check_window (run on every real window)
               wants the window end + 4e4 s inside the ltt table.
* ``centered`` ``--window-days`` (default 120) with the catalogue merger ``--merger-at-days``
               (default 100) in: the mbh_batched_mojito_check.py placement and THE MBH
               harness window (mbh_speed_durations.sh's data step) -- the batched template's
               90 d before / 10 d after the merger + margins + pads + edge crop fit inside.
               A merger within 100 d of the file start (or 20 d + 1 d of its end) SHIFTS
               the window into the file (the merger moves off 100 d; the batched window
               clamps to the active box as production clamps at the data edges); only a
               file shorter than the window is ``skipped_window_outside_file``.
A source whose merger violates the production admission rule
``window_start <= t_merge < window_end + MBH_MERGER_TIME_BUFFER`` (7 d) is recorded as
``skipped_outside_window`` (not an error; production drops it from the MBH branch).

Per template (keys ``<tag>_...``): logL = -1/2 <d-h|d-h> (0 is perfect: the data is
source-only), snr_opt = sqrt<h|h>, snr_det = <d|h>/sqrt<h|h>, snr_ratio = snr_opt/snr_data,
mm_data = 1 - <d|h>/sqrt(<d|d><h|h>), flat per-channel mm + amplitude ratio vs the data,
per-band residual SNR sqrt<d-h|d-h> restricted to the WDM frequency layers whose centre
m * layer_df lies in [<1, 1-5, 5-15, 15-25] mHz (is a ~20 mHz response burst noise-
significant?) with ``band_sum_check`` = sum of the band <r|r> / total <r|r> (1 by
construction), wall s and peak RSS. Per window: logL0 = -1/2 <d|d>, snr_data, the data's
band SNRs, the batched kept-box geometry (merger layer, active box, clamps, outside flags).
The EMRI harness's names ride along (mbh_speed_durations.sh's data step): ``data_snr``,
``brick`` (file name), ``tobs_s``, ``foreground`` (bool), ``noise`` (RunBox.describe()), per
template ``snr_<name>`` and ``data_<name>`` = {mm, logL, snr_ratio, snr} (name: production
for prod, prod90, batched, tof), ``mm_vs_production`` = the batched vs production
noise-weighted mismatch, and ``differs_from_emri`` (mbh_harness.DIFFERS_FROM_EMRI).

Bricks: the source's ``MBHB_*_L1_source<id>_*.h5``, searched as the EMRI harness does
(``--l1-dir`` alone when given; else MOJITO_LIGHT_PATH/data/MBHB/L1, then MOJITO_DATA_PATH,
MOJITO_INFO_PATH, the catalogue's root -- each directly, then recursively); its orbits are
read through ``WindowedL1Orbits`` (the ltt slice only) and pass the EMRI harness's
``check_window``. ``--start-offset-s`` starts the file-start windows that many seconds
into the brick (default 0, production; the harness passes START_OFFSET_S = 5e4).
Pairwise (batched-prod90, batched-prod, prod90-prod; tof-prod / tof-prod90): noise-weighted
mm, dlogL, ||a-b|| (noise weighted), flat per-channel mm + amplitude ratio; plus the
fraction of prod90's flat power outside the batched kept box. One JSON line per
(src, window) is APPENDED to ``--out``.

Usage (the other computer; MOJITO_LIGHT_PATH holds catalogues/ and data/MBHB/L1/):

    export MOJITO_LIGHT_PATH=/path/to/mojito_light_v1_0_0
    # placement / admission / batched geometry for every source, no data, no waveforms
    # (one '[admitted] <window> l1=<ids> no_l1=<ids>' line per window closes the table):
    python scripts/mbh/mbh_cd1l_campaign.py --src all --window 6mo,24mo,centered --dry-run
    python scripts/mbh/mbh_cd1l_campaign.py --src all --window 180d,360d,720d --dry-run
    # one (source, window):
    python scripts/mbh/mbh_cd1l_campaign.py --src 16 --window centered --out mbh_cd1l.jsonl
    python scripts/mbh/mbh_cd1l_campaign.py --src 16 --window 6mo --backend cuda12x --out mbh_cd1l.jsonl
    # every source x (6mo, centered), serially, resumable, summary at the end:
    BACKEND=cuda12x OUT_DIR=mbh_cd1l_out bash scripts/mbh/mbh_cd1l_campaign.sh
    python scripts/mbh/mbh_cd1l_campaign_summary.py mbh_cd1l_out/results.jsonl

Memory (peak RSS, CPU backend; the window's TD data, the ltt slice, one 90-d phentax +
order-8 response and one full-window WDM transform dominate):
    centered 120 d, id 16, prod+prod90+batched: 4.19 GB peak (ru_maxrss), 208 s wall on
        the 8-GB laptop (i5-8257U, one thread; 2026-09-30). Cumulative peak after the data
        load 1.30 GB, after prod (T 30.4 d) 1.93, after prod90 (T 90 d) 3.77, after
        batched 4.19;
    6mo (180 d) and 24mo (720 d): NOT measured (too big for the laptop). The data, ltt
        slice and full-grid templates scale with the window (6mo ~1.5x, 24mo ~6x the
        centered sizes), the 90-d waveform does not: expect ~5-7 GB for 6mo and ~12-20 GB
        for 24mo on CPU. On a GPU the host still holds the data read and the ltt slice; set
        RSS_LIMIT_GB accordingly (0 = no watchdog).

Knobs (env): MOJITO_LIGHT_PATH (default ~/.mojito_cache/brickmarket/mojito_light_v1_0_0);
RSS_LIMIT_GB (hard exit 42 above this peak RSS; 0 = off, default); the stock MBH knobs
MBH_RESPONSE_ORDER, MBH_WAVEFORM_DURATION, MBH_WINDOW_{BEFORE,AFTER,PAD,MARGIN}_DAYS,
MBH_MERGER_TIME_BUFFER are read through ``SourceMBHSettings`` exactly as a production build
reads them. ``--backend cpu`` sets JAX_PLATFORMS=cpu (unless set) and never imports cupy;
a CUDA backend sets XLA_PYTHON_CLIENT_PREALLOCATE=false (unless set) so JAX (phentax)
and cupy share the device.
"""
import argparse
import functools
import gc
import glob
import json
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 2.5
TOBS_6MO = 15552000.0                  # production TOBS_TARGET of the 6mo campaign
WINDOWS = {"6mo": TOBS_6MO, "24mo": 4 * TOBS_6MO}
ALL_WINDOWS = ("6mo", "24mo", "centered")
#: '<N>d': N days from the file start, placed as production (like 6mo / 24mo)
DAYS_WINDOW = re.compile(r"^([0-9]+(?:\.[0-9]+)?)d$")
#: a centered window shifted against the file end stops this far before it (ltt table cover)
CENTERED_END_GUARD_S = 86400.0


def file_start_tobs(window):
    """Tobs [s] of a window placed at the FILE START (6mo, 24mo, '<N>d'); None otherwise."""
    if window in WINDOWS:
        return WINDOWS[window]
    m = DAYS_WINDOW.match(window)
    return None if m is None else float(m.group(1)) * 86400.0


def valid_window(window):
    return window in ALL_WINDOWS or DAYS_WINDOW.match(window) is not None
ALL_TEMPLATES = ("prod", "prod90", "batched", "tof")
PAIRS = (("batched", "prod90"), ("batched", "prod"), ("prod90", "prod"),
         ("tof", "prod"), ("tof", "prod90"))
# WDM frequency-layer bands by layer centre m * layer_df [Hz]; the active box ends at
# --max-freq (25 mHz by default), so the last band is 15-25 mHz.
BANDS = (("<1", 0.0, 1e-3), ("1-5", 1e-3, 5e-3), ("5-15", 5e-3, 15e-3), ("15-25", 15e-3, np.inf))
LTT_PAD = 1.0e5   # s of ltt kept beyond the span the response reads (as the check script)
CAT_NAME = "mbhb_cat_mojito_lite_processed_MT_rounding_fixed.hdf5"   # L1DataLoader's MBHB map
T_MERGE_KEY = "TimeCoalescencePhenomTPHMSSBFrame"   # seconds after MOJITO_REFERENCE_TIME


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if platform.system() == "Darwin" else r / 1e6   # bytes on macOS, kB on Linux


def watchdog(limit_gb):
    def _run():
        while True:
            if peak_rss_gb() > limit_gb:
                print(f"[campaign] RSS {peak_rss_gb():.2f} GB > RSS_LIMIT_GB {limit_gb} -> exit 42",
                      flush=True)
                os._exit(42)
            time.sleep(0.3)
    threading.Thread(target=_run, daemon=True).start()


def fval(x):
    """Host float of a (possibly device, possibly complex) scalar."""
    from lisatools.utils.utility import asnumpy
    return float(np.real(asnumpy(x)))


def jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def flat_stats(a, b):
    """Per channel flat 1 - Re(O) and |a|/|b| (host arrays on the same box)."""
    mm, amp = [], []
    for c in range(3):
        x, y = a[c], b[c]
        den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        mm.append(1.0 - float(np.sum(x * y)) / den if den > 0 else float("nan"))
        ny = float(np.linalg.norm(y))
        amp.append(float(np.linalg.norm(x)) / ny if ny > 0 else float("nan"))
    return mm, amp


def mojito_root():
    return os.environ.get("MOJITO_LIGHT_PATH",
                          os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"))


def load_catalogue(path):
    """Every row of the MBHB catalogue as ``L1DataLoader.load_single_binary`` reads one
    (``Binaries/<key>[row]``, numeric fields only; row index = source id)."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["Binaries"]
        cols = {k: np.asarray(g[k][:]) for k in g.keys()}
    cols = {k: v for k, v in cols.items() if v.dtype.kind in "fiu" and v.ndim == 1}
    n = len(next(iter(cols.values())))
    return [{k: float(v[i]) for k, v in cols.items()} for i in range(n)]


def brick_search_roots(l1_dir=None, catalogue=None):
    """Directories searched for a source's brick, in the EMRI harness's order
    (scripts/emri/emri_batch_speed.find_emri_brick, origin/dev ddaad46f): an explicit
    ``l1_dir`` ALONE; otherwise MOJITO_LIGHT_PATH/data/MBHB/L1, MOJITO_DATA_PATH
    (/shared/data/mojito_cache on the cluster), MOJITO_INFO_PATH and the catalogue's root
    (two levels above the catalogue file)."""
    if l1_dir:
        return [os.path.expanduser(l1_dir)]
    roots = [os.path.join(mojito_root(), "data", "MBHB", "L1")]
    roots += [os.path.expanduser(os.environ[k]) for k in ("MOJITO_DATA_PATH", "MOJITO_INFO_PATH")
              if os.environ.get(k)]
    if catalogue and os.path.isfile(catalogue):
        roots.append(os.path.dirname(os.path.dirname(os.path.abspath(catalogue))))
    return roots


@functools.lru_cache(maxsize=None)
def _bricks_under(root):
    """``(direct, recursive)`` sorted MBHB L1 brick paths under ``root`` (one walk per root
    and process: the dry-run asks for 20 sources)."""
    if not os.path.isdir(root):
        return (), ()
    direct = tuple(sorted(glob.glob(os.path.join(root, "MBHB_*_L1_source*.h5"))))
    deep = tuple(sorted(glob.glob(os.path.join(root, "**", "MBHB_*_L1_source*.h5"), recursive=True)))
    return direct, deep


def find_mbhb_brick(src, l1_dir=None, catalogue=None):
    """Path of source ``src``'s MBHB L1 brick (``MBHB_*_L1_source<src>_*.h5``; ``source1_`` never
    matches ``source11_``), or None: per root of :func:`brick_search_roots`, the first sorted
    direct match, else the first recursive one -- find_emri_brick's rule."""
    tag = f"_L1_source{int(src)}_"
    for root in brick_search_roots(l1_dir, catalogue):
        for hits in _bricks_under(root):
            hits = [p for p in hits if tag in os.path.basename(p)]
            if hits:
                return hits[0]
    return None


def l1_file(src):
    """Path of source ``src``'s MBHB L1 file, or None (:func:`find_mbhb_brick`, no ``l1_dir``)."""
    return find_mbhb_brick(src)


def file_sampling(path):
    """``(t0, dt, size)`` of the file's tdis stream (metadata only)."""
    from mojito import MojitoL1File

    with MojitoL1File(path) as f:
        ts = f.tdis.time_sampling
        return float(ts.t0), float(ts.dt), int(ts.size)


def production_knobs(order):
    """The stock MBH branch values a production build reads (env-backed dataclass)."""
    from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

    mbh = SourceMBHSettings()
    return dict(
        order=int(mbh.response_order if order is None else order),
        prod_waveform_duration=None if mbh.waveform_duration is None else float(mbh.waveform_duration),
        window_before=float(mbh.window_before_days) * 86400.0,
        window_after=float(mbh.window_after_days) * 86400.0,
        window_pad=float(mbh.window_pad_days) * 86400.0,
        window_margin=float(mbh.window_margin_days) * 86400.0,
        window_decimate=int(getattr(mbh, "window_decimate", 1)),   # MBH_WINDOW_DECIMATE
        buffer_time=float(mbh.buffer_time),
        higher_modes=[int(m) for m in mbh.higher_modes],
        phenom_tol=float(mbh.phenom_tol),
        start_freq=float(mbh.start_freq),
        merger_time_buffer=float(mbh.mbh_merger_time_buffer),
        tdionfly_margin=float(mbh.tdionfly_margin),
    )


def placement(window, args, ref, file_t0, file_size, t_merge_rel, merger_buffer):
    """Window start / size and the production admission verdict, from metadata only."""
    layer = args.nf * DT
    tobs_target = file_start_tobs(window)
    if tobs_target is not None:
        nt = int(round(tobs_target / layer))
        # production: file samples [0, Nf*Nt) (window_start_offset = 0); --start-offset-s
        # moves the start (mbh_speed_durations.sh: START_OFFSET_S = 5e4, the EMRI run box)
        start = int(round(float(getattr(args, "start_offset_s", 0.0)) / DT))
        how = (f"file start + {start * DT:.0f} s" + (
            " (all_sources mojito loader, window_start_offset=0)" if start == 0 else " (--start-offset-s)"))
    else:
        nt = int(round(args.window_days * 86400.0 / layer))
        nt -= nt % 2
        start = int(round((ref + t_merge_rel - args.merger_at_days * 86400.0 - file_t0) / DT))
        how = f"catalogue merger {args.merger_at_days:g} d in (mbh_batched_mojito_check placement)"
        # a merger too near either end of the file: SHIFT the window into the file (the
        # merger stays inside it; the batched window clamps to the active box, as production
        # clamps at the data edges) rather than dropping the source. The end keeps
        # CENTERED_END_GUARD_S clear so the ltt table (REF + 730.5 d on the 731-d bricks)
        # covers the window end + the production wrapper's 4e4 s (check_window).
        hi = file_size - args.nf * nt - int(round(CENTERED_END_GUARD_S / DT))
        if 0 <= hi and not 0 <= start <= hi:
            start = min(max(start, 0), hi)
            how += (f"; SHIFTED into the file: merger "
                    f"{(ref + t_merge_rel - file_t0 - start * DT) / 86400.0:.2f} d in")
    nt -= nt % 2   # WDMSettings needs an even layer count
    n = args.nf * nt
    window_t0 = file_t0 + start * DT
    tobs = n * DT
    obs_start = window_t0 - ref                     # production's frame: relative to the epoch
    obs_end = obs_start + tobs
    in_file = start >= 0 and start + n <= file_size
    admitted = obs_start <= t_merge_rel < obs_end + merger_buffer
    status = ("skipped_window_outside_file" if not in_file
              else "ok" if admitted else "skipped_outside_window")
    return dict(
        window=window, status=status, placement=how, nf=int(args.nf), nt=int(nt), n=int(n),
        tobs_s=tobs, window_days=tobs / 86400.0, start_index=int(start), window_t0=window_t0,
        file_t0=file_t0, file_size=int(file_size),
        window_start_days_after_file_start=start * DT / 86400.0,
        window_start_after_ref_s=obs_start,
        t_merge_rel=t_merge_rel, t_merge_abs=ref + t_merge_rel,
        merger_days_in_window=(ref + t_merge_rel - window_t0) / 86400.0,
        merger_days_after_window_end=(ref + t_merge_rel - window_t0 - tobs) / 86400.0,
        merger_time_buffer_s=merger_buffer, admitted=bool(admitted), in_file=bool(in_file),
    )


def make_grid(p, args, backend):
    from lisatools.domains import TDSettings, WDMSettings

    layer = args.nf * DT
    crop = int(args.edge_crop)
    wdm = WDMSettings(args.nf, p["nt"], DT, t0=p["window_t0"], min_freq=args.min_freq,
                      max_freq=args.max_freq, min_time=crop * layer if crop else None,
                      max_time=(p["nt"] - crop) * layer if crop else None, force_backend=backend)
    tds = TDSettings(p["n"], DT, t0=p["window_t0"], force_backend=backend)
    return wdm, tds


def batched_geometry(wdm, t_merge_abs, k):
    """``mbh_window_layers`` + the check script's edge diagnostics (dict, never raises)."""
    from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

    try:
        geom = mbh_window_layers(wdm, t_merge_abs, k["window_before"], k["window_after"],
                                 k["window_pad"], k["window_margin"])
    except ValueError as exc:
        return dict(error=f"{type(exc).__name__}: {exc}")
    layer_s = float(wdm.layer_dt)
    t0 = float(wdm.t0)
    act_lo = t0 + int(wdm.ind_min_t) * layer_s
    act_hi = t0 + (int(wdm.ind_max_t) + 1) * layer_s
    box_lo = t0 + geom["n_start"] * layer_s
    box_hi = box_lo + geom["Nt_keep"] * layer_s
    geom.update(
        merger_layer=(t_merge_abs - t0) / layer_s,
        active_t_layers=[int(wdm.ind_min_t), int(wdm.ind_max_t)],
        active_box_days_in_window=[(act_lo - t0) / 86400.0, (act_hi - t0) / 86400.0],
        kept_box_days_in_window=[(box_lo - t0) / 86400.0, (box_hi - t0) / 86400.0],
        box_clamped_lo=bool(geom["n_start"] == int(wdm.ind_min_t)),
        box_clamped_hi=bool(geom["n_start"] + geom["Nt_keep"] == int(wdm.ind_max_t) + 1),
        merger_in_active_box=bool(act_lo <= t_merge_abs < act_hi),
        # the move's [MBH_BATCH] counters
        outside_box=bool(max(t_merge_abs, act_lo) < box_lo or min(t_merge_abs, act_hi) > box_hi),
        outside_data=bool(t_merge_abs < act_lo or t_merge_abs >= act_hi),
    )
    return geom


def versions(backend):
    from importlib.metadata import version

    out = {}
    for pkg in ("lisaanalysistools", "bbhx", "phentax", "jax", "jaxlib", "numpy", "mojito",
                "gpubackendtools", "cupy-cuda12x", "cupy-cuda13x"):
        try:
            out[pkg] = version(pkg)
        except Exception:
            pass
    import lisatools

    out["lisatools_file"] = lisatools.__file__
    try:
        out["lisatools_git"] = subprocess.run(
            ["git", "-C", os.path.dirname(lisatools.__file__), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return out


def device_pool_gb(backend):
    if backend == "cpu":
        return None
    import cupy as cp   # CUDA backends only

    return cp.get_default_memory_pool().total_bytes() / 1e9


def windowed_orbits_class(out_dir):
    """``WindowedL1Orbits`` from mbh_batched_mojito_check.py (ltt read over a slice only).

    Importing that script starts its own RSS watchdog (MBH_MEM_CAP_GB, default 5 GB) and
    creates MBH_CHECK_OUT; both are neutralised here so RSS_LIMIT_GB alone governs and no
    directory appears next to the scripts (MBH_CHECK_OUT -> the --out directory).
    """
    os.environ["MBH_MEM_CAP_GB"] = "1e9"
    os.environ["MBH_CHECK_OUT"] = out_dir
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from mbh_batched_mojito_check import WindowedL1Orbits

    return WindowedL1Orbits


def dry_run(args, srcs, windows):
    """Placement, admission and batched kept-box geometry per (src, window); no data read."""
    from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME as REF

    k = production_knobs(args.order)
    cat_path = os.path.join(mojito_root(), "catalogues", CAT_NAME)
    cat = load_catalogue(cat_path)
    have = {s: find_mbhb_brick(s, args.l1_dir, cat_path) for s in srcs}
    sampling = {s: file_sampling(p) for s, p in have.items() if p}
    if not sampling:
        # no L1 file at all: every brick is a 731-d, 2.5-s stream; borrow nothing
        raise SystemExit("dry-run: no MBHB L1 file under MOJITO_LIGHT_PATH to read t0/size from")
    lender = min(sampling)
    t0s = {round(v[0], 6) for v in sampling.values()}
    print(f"# MOJITO_REFERENCE_TIME {REF:.6f}; L1 files present for {sorted(sampling)}; "
          f"distinct file t0 {sorted(t0s)} (file t0 - REF = "
          f"{', '.join(f'{t - REF:+.3f}' for t in sorted(t0s))} s)")
    print(f"# missing-file sources borrow t0/size from source {lender} (flag 'borrowed'); "
          f"merger buffer {k['merger_time_buffer'] / 86400:g} d; batched window "
          f"{k['window_before'] / 86400:g} d before / {k['window_after'] / 86400:g} d after, "
          f"pad {k['window_pad'] / 86400:g} d, margin {k['window_margin'] / 86400:g} d")
    print("| src | window | status | L1 | merger d in file | window [d in file] | merger d in window "
          "| merger layer | active t layers | kept layers (Nt_keep) | kept box d | clamp lo/hi | outside box/data |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    admitted = {w: [] for w in windows}
    for s in srcs:
        t0, dt, size = sampling.get(s, sampling[lender])
        assert abs(dt - DT) < 1e-9, dt
        t_rel = cat[s][T_MERGE_KEY]
        for w in windows:
            p = placement(w, args, REF, t0, size, t_rel, k["merger_time_buffer"])
            if p["status"] == "ok":
                admitted[w].append(s)
            g = {}
            if p["in_file"]:
                wdm, _ = make_grid(p, args, "cpu")
                g = batched_geometry(wdm, p["t_merge_abs"], k)
            lo = p["window_start_days_after_file_start"]
            geo = ("| - | - | - | - | - | - |" if not g else f"| {g['error'][:50]} | | | | | |" if "error" in g else
                   f"| {g['merger_layer']:.1f} | [{g['active_t_layers'][0]}, {g['active_t_layers'][1]}] "
                   f"| {g['n_start']}..{g['n_start'] + g['Nt_keep']} ({g['Nt_keep']}) "
                   f"| {g['kept_box_days_in_window'][0]:.2f}..{g['kept_box_days_in_window'][1]:.2f} "
                   f"| {int(g['box_clamped_lo'])}/{int(g['box_clamped_hi'])} "
                   f"| {int(g['outside_box'])}/{int(g['outside_data'])} |")
            print(f"| {s} | {w} | {p['status']} | {'yes' if have[s] else 'borrowed'} "
                  f"| {(REF + t_rel - t0) / 86400:.2f} | {lo:.2f}..{lo + p['window_days']:.2f} "
                  f"| {p['merger_days_in_window']:.2f} " + geo, flush=True)
    # machine-readable admission per window (mbh_speed_durations.sh's data step runs l1=):
    # admitted AND an L1 file present / admitted on borrowed file timing, no L1 file
    for w in windows:
        with_l1 = ",".join(str(s) for s in admitted[w] if have[s]) or "-"
        without = ",".join(str(s) for s in admitted[w] if not have[s]) or "-"
        print(f"[admitted] {w} l1={with_l1} no_l1={without}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="CD1L MBHB catalogue row (0-19); with --dry-run also a comma list or 'all'")
    ap.add_argument("--window", required=True,
                    help="6mo, 24mo, '<N>d' (N days from the file start) or centered; with --dry-run "
                         "also a comma list or 'all' (= 6mo,24mo,centered)")
    ap.add_argument("--templates", default="prod,prod90,batched",
                    help="comma list of prod,prod90,batched,tof (default prod,prod90,batched)")
    ap.add_argument("--order", type=int, default=None,
                    help="response Lagrange order (default SourceMBHSettings.response_order = 8)")
    ap.add_argument("--backend", default="cpu", choices=("cpu", "cuda12x", "cuda13x"))
    ap.add_argument("--window-days", type=float, default=120.0, help="centered window length [d]")
    ap.add_argument("--merger-at-days", type=float, default=100.0,
                    help="centered window: catalogue merger this far in [d]")
    ap.add_argument("--nf", type=int, default=1440, help="WDM Nf (1440 -> 3600 s layers at dt 2.5 s)")
    ap.add_argument("--edge-crop", type=int, default=60, help="WDM time-edge crop per side [layers]")
    ap.add_argument("--min-freq", type=float, default=4e-4)
    ap.add_argument("--max-freq", type=float, default=2.5e-2)
    ap.add_argument("--l1-dir", default=None,
                    help="first place to look for the source's MBHB L1 brick (then MOJITO_LIGHT_PATH/data/"
                         "MBHB/L1, then recursively MOJITO_DATA_PATH, MOJITO_INFO_PATH, the catalogue root)")
    ap.add_argument("--start-offset-s", type=float, default=0.0,
                    help="file-start windows (6mo, 24mo, <N>d): start this many seconds after the brick's "
                         "tdis t0 (default 0 = production; mbh_speed_durations.sh passes START_OFFSET_S)")
    ap.add_argument("--foreground", default="on", choices=("on", "off"),
                    help="noise weighting (mbh_harness.py / RunBox): XYZ2 scirdv1 + the fitted tanh galactic "
                         "foreground at Tobs = Nf*Nt*dt of the window (on, default) or the instrument "
                         "alone (off: the campaign's numbers before 2026-10-02)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print placement / admission / batched geometry only (no data, no waveforms)")
    ap.add_argument("--out", default="mbh_cd1l_campaign.jsonl")
    args = ap.parse_args()

    if args.backend == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    bad = set(templates) - set(ALL_TEMPLATES)
    if bad:
        ap.error(f"unknown templates {sorted(bad)}")
    windows = list(ALL_WINDOWS) if args.window == "all" else [w.strip() for w in args.window.split(",")]
    bad_w = [w for w in windows if not valid_window(w)]
    if bad_w:
        ap.error(f"--window {bad_w}: must be among {ALL_WINDOWS} or '<N>d'")

    if args.dry_run:
        n_cat = len(load_catalogue(os.path.join(mojito_root(), "catalogues", CAT_NAME)))
        srcs = list(range(n_cat)) if args.src == "all" else [int(s) for s in args.src.split(",")]
        dry_run(args, srcs, windows)
        return 0
    if len(windows) != 1 or not args.src.lstrip("-").isdigit():
        ap.error("a real run takes ONE --src and ONE --window (lists are for --dry-run)")
    src, window = int(args.src), windows[0]

    rss_limit = float(os.environ.get("RSS_LIMIT_GB", "0"))
    if rss_limit > 0:
        watchdog(rss_limit)
    t_start = time.perf_counter()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from mbh_harness import DIFFERS_FROM_EMRI, check_window, harness_box, noise_record

    from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME as REF

    k = production_knobs(args.order)
    cat_path = os.path.join(mojito_root(), "catalogues", CAT_NAME)
    cat = load_catalogue(cat_path)
    if not 0 <= src < len(cat):
        ap.error(f"--src {src} outside the catalogue's {len(cat)} rows")
    path = find_mbhb_brick(src, args.l1_dir, cat_path)
    if path is None:
        print(f"[campaign] no MBHB L1 file for source {src} (searched "
              f"{brick_search_roots(args.l1_dir, cat_path)})", flush=True)
        return 3
    file_t0, file_dt, file_size = file_sampling(path)
    assert abs(file_dt - DT) < 1e-9, file_dt
    p = placement(window, args, REF, file_t0, file_size, cat[src][T_MERGE_KEY], k["merger_time_buffer"])
    row = dict(src=src, window=window, status=p["status"], placement=p, knobs=k,
               templates=templates, backend=args.backend, host=socket.gethostname(),
               platform=platform.platform(), python=sys.version.split()[0], argv=sys.argv,
               l1_file=os.path.basename(path), brick=os.path.basename(path), edge_crop=int(args.edge_crop),
               min_freq=args.min_freq, max_freq=args.max_freq, dt=DT, nf=p["nf"], nt=p["nt"],
               tobs_s=p["tobs_s"], versions=versions(args.backend),
               # the EMRI harness's record fields (noise / data_snr filled once the grid exists)
               foreground=args.foreground == "on", data_snr=None, differs_from_emri=list(DIFFERS_FROM_EMRI))
    print(f"[campaign] src={src} {window}: {p['status']}; window {p['window_days']:.2f} d from "
          f"file day {p['window_start_days_after_file_start']:.3f}; merger "
          f"{p['merger_days_in_window']:.3f} d in", flush=True)
    if p["status"] != "ok":
        with open(args.out, "a") as f:
            f.write(json.dumps(row, default=jsonable) + "\n")
        return 0

    from mojito import MojitoL1File

    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import TDSettings, TDSignal, WDMSignal
    from lisatools.globalfit.recipe import mbh_catalogue_to_sampling_basis
    from lisatools.globalfit.stock.erebor import make_mbh_transform_container
    from lisatools.globalfit.stock.erebor.source_runtime import (
        SnappedEpochMBHGen, snap_waveform_t0_to_lattice)
    from lisatools.globalfit.stock.erebor import wrappers
    from lisatools.utils.utility import asnumpy

    WindowedL1Orbits = windowed_orbits_class(os.path.dirname(os.path.abspath(args.out)))
    backend = args.backend
    window_t0, n, tobs = p["window_t0"], p["n"], p["tobs_s"]
    t_merge_abs = p["t_merge_abs"]

    # ---- orbits FIRST (refuse before reading the data): the ltt slice -- the stock response
    # reads from data_t0 - tdi_buffer to the merger + ringdown + buffer_time (a merger up to
    # 7 d after the window end is admitted); the TOF reads its whole phentax span, T = window +
    # margin before the merger -- and the EMRI harness's rule: the window end + the production
    # wrapper's 4e4 s inside the ltt table
    ltt_lo = window_t0 - LTT_PAD
    if "tof" in templates:
        ltt_lo = min(ltt_lo, t_merge_abs - (tobs + k["tdionfly_margin"]) - LTT_PAD)
    ltt_hi = max(window_t0 + tobs, t_merge_abs) + LTT_PAD
    orb = WindowedL1Orbits(path, ltt_lo, ltt_hi, force_backend=backend, frame="icrs")
    orb._ensure_configured()
    check_window(orb, window_t0, tobs, "L1 brick")

    # ---- data: the same dataset L1DataLoader reads (X2,Y2,Z2 / laser_frequency), window only
    with MojitoL1File(path) as f:
        data_td = np.ascontiguousarray(np.asarray(f.tdis.xyz_doppler[p["start_index"]:p["start_index"] + n]).T)
    assert data_td.shape == (3, n), data_td.shape
    wdm, tds = make_grid(p, args, backend)
    xp = wdm.xp
    d_sig = TDSignal(xp.asarray(data_td), tds).transform(wdm)
    del data_td
    gc.collect()
    # mbh_harness.py: the EMRI harness's RunBox noise on this grid -- XYZ2 scirdv1 + (default)
    # the fitted tanh galactic foreground at Tobs = Nf*Nt*dt; --foreground off = the instrument
    box = harness_box(wdm, args.edge_crop, args.foreground)
    sens = box.sens
    row.update(noise_record(box))
    ac_data = AnalysisContainer(d_sig, sens)
    d_arr = d_sig.arr
    box_shape = (3, int(wdm.Nf_active), int(wdm.Nt_active))
    assert tuple(d_arr.shape) == box_shape, (d_arr.shape, box_shape)
    dd = fval(ac_data.inner_product())

    # frequency bands in active-box-relative layer slices
    m_abs = np.arange(int(wdm.ind_min_f), int(wdm.ind_max_f) + 1)
    f_mid = m_abs * float(wdm.layer_df)
    bands = []
    for label, lo, hi in BANDS:
        idx = np.flatnonzero((f_mid >= lo) & (f_mid < hi))
        if idx.size:
            fsl = slice(int(idx[0]), int(idx[-1]) + 1)
            bands.append((label, fsl, wdm.get_slice((fsl, slice(0, box_shape[2])))))

    def band_snrs(arr):
        return {label: fval(ac_data.template_snr(WDMSignal(xp.ascontiguousarray(arr[:, fsl, :]), sub))[0])
                for label, fsl, sub in bands}

    row.update(
        window_t0=window_t0, layer_dt=float(wdm.layer_dt), layer_df=float(wdm.layer_df),
        active_f_layers=[int(wdm.ind_min_f), int(wdm.ind_max_f)],
        active_t_layers=[int(wdm.ind_min_t), int(wdm.ind_max_t)],
        bands={label: [int(m_abs[fsl.start]), int(m_abs[fsl.stop - 1])] for label, fsl, _ in bands},
        snr_data=float(np.sqrt(max(dd, 0.0))), logL0=-0.5 * dd, band_snr_data=band_snrs(d_arr),
    )
    row["data_snr"] = row["snr_data"]    # the EMRI / SOBBH harness's name for sqrt<d|d>
    print(f"[campaign] grid Nf={p['nf']} Nt={p['nt']} active f {row['active_f_layers']} t "
          f"{row['active_t_layers']}; snr_data={row['snr_data']:.3f}; band SNR "
          f"{ {b: round(v, 3) for b, v in row['band_snr_data'].items()} }", flush=True)

    truth = np.asarray(make_mbh_transform_container().both_transforms(
        np.asarray(mbh_catalogue_to_sampling_basis(cat[src]), float)), float)
    row["truth_waveform_basis"] = truth.tolist()
    assert abs(truth[10] - p["t_merge_rel"]) < 1e-6, (truth[10], p["t_merge_rel"])

    # the run's rule: with MBH_WINDOW_DECIMATE = q both prod90 and batched snap onto q*DT
    q = int(k["window_decimate"])
    t0s, snap = snap_waveform_t0_to_lattice(REF, window_t0, DT * q)
    geom = batched_geometry(wdm, t_merge_abs, k)
    row.update(snap=snap, window_decimate=q, ltt_slice=[ltt_lo, ltt_hi], batched_geometry=geom,
               load_wall_s=time.perf_counter() - t_start, load_peak_rss_gb=peak_rss_gb())
    print(f"[campaign] snap {snap:+.6f} s; batched geometry {json.dumps(geom, default=jsonable)}; "
          f"load {row['load_wall_s']:.0f} s, rss {row['load_peak_rss_gb']:.2f} GB", flush=True)

    def stock_gen(waveform_t0, duration):
        return wrappers.get_mbh_phenom_wave_gen(
            data_td_settings=tds, waveform_t0=waveform_t0, dt=DT, orbits=orb,
            output_domain_settings=wdm, tukey_alpha=0.0, force_backend=backend,
            waveform_duration=duration, data_span=tobs, higher_modes=k["higher_modes"],
            phenom_tol=k["phenom_tol"], start_freq=k["start_freq"], response_order=k["order"],
            buffer_time=k["buffer_time"], tdi_gen_str="2nd generation", tdi_chan="XYZ",
            min_freq=args.min_freq, max_freq=args.max_freq)

    arrs, host = {}, {}
    d_host = np.asarray(asnumpy(d_arr))
    for tag in templates:
        t0 = time.perf_counter()
        try:
            if tag == "prod":
                sig = stock_gen(REF, k["prod_waveform_duration"]).get_signals_for_residuals(*truth)
            elif tag == "prod90":
                gen = SnappedEpochMBHGen(stock_gen(t0s, k["window_before"]), snap)
                sig = gen.get_signals_for_residuals(*truth)
            elif tag == "batched":
                from lisatools.sources.batching import MBHWindowedWDMSignalGen
                from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform

                if "error" in geom:
                    raise ValueError(geom["error"])
                wgen = WindowedGridAlignedMBHWaveform(   # get_mbh_windowed_gen's kwargs
                    waveform_kwargs=dict(higher_modes=list(k["higher_modes"]), include_negative_modes=True,
                                         t_low_fit=True, coarse_grain=False, atol=k["phenom_tol"],
                                         rtol=k["phenom_tol"]),
                    Tobs=k["window_before"], start_freq=k["start_freq"], use_reference_time=True,
                    waveform_t0=t0s, data_td_settings=(tds if q == 1 else TDSettings(
                        int(tds.N) // q, DT * q, t0=float(tds.t0), force_backend=backend)),
                    tdi_generation="2nd generation",
                    tdi_channels="XYZ", sampling_frequency=1.0 / (DT * q), orbits=orb, order=k["order"],
                    tukey_alpha=0.0, stft_dt=None, freq_min=args.min_freq, freq_max=args.max_freq,
                    fft_batch_size=1, buffer_time=k["buffer_time"], output_domain_settings=wdm,
                    force_backend=backend)
                adapter = MBHWindowedWDMSignalGen(wgen, wdm, nchannels=3, tukey_alpha=0.0,
                                                  **({"decimate": q} if q != 1 else {}))
                adapter.set_window(geom["n_start"], geom["Nt_keep"], geom["n_pad_lo"], geom["n_pad_hi"])
                prow = truth.copy()
                prow[10] -= snap                  # the move's t_plunge_snap
                out = adapter(*prow[:, None])     # one-row batch, as the move's chunks
                sig = WDMSignal(out.arr[0], out.settings)
                row["batched_segment"] = dict(adapter.geometry)
                del out, adapter, wgen
            else:
                from lisatools.response.tdiconfig import TDIConfig

                dur_s = tobs + k["tdionfly_margin"]   # get_mbh_tdionfly_wave_wrap's recipe
                gen = wrappers.get_mbh_tdionfly_gen(
                    dt=DT, t_start=REF, dur_s=dur_s,
                    tdi_config=TDIConfig("2nd generation", force_backend=backend), orbits=orb,
                    waveform_duration=dur_s, force_backend=backend)
                wrap = wrappers.MBHTDIonFlyWaveWrap(gen, window_t0 + np.arange(n) * DT, tds, wdm, nchannels=3)
                sig = wrap(*truth)
                del wrap
            row[f"{tag}_wall_s"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            opt, det = ac_data.template_snr(sig)   # sub-box (batched) sliced by the container
            opt, det = fval(opt), fval(det)
            ll = fval(ac_data.template_likelihood(sig))
            hh, dh = opt * opt, det * opt
            # the same template on the data's full active box (zeros outside a sub-box)
            fsl, tsl = wdm.sub_box_slices(sig.settings)
            h = xp.zeros(box_shape)
            h[:, fsl, tsl] = sig.arr
            del sig
            r = d_arr - h
            bres = band_snrs(r)
            del r
            row.update({
                f"{tag}_logL": ll, f"{tag}_snr_opt": opt, f"{tag}_snr_det": det,
                f"{tag}_snr_ratio": opt / row["snr_data"] if row["snr_data"] > 0 else float("nan"),
                f"{tag}_dh": dh, f"{tag}_hh": hh,
                f"{tag}_mm_data": 1.0 - dh / np.sqrt(dd * hh) if dd > 0 and hh > 0 else float("nan"),
                f"{tag}_resid_snr": float(np.sqrt(max(-2.0 * ll, 0.0))),
                f"{tag}_band_resid_snr": bres,
                f"{tag}_band_sum_check": (sum(v * v for v in bres.values()) / (-2.0 * ll)) if ll < 0 else float("nan"),
            })
            # the EMRI harness's names: snr_<name> = sqrt<h|h> and data_<name> = RunBox.score's
            # {mm = 1 - <d|h>/sqrt(<d|d><h|h>), logL = -1/2<d-h|d-h>, snr_ratio = sqrt(<h|h>/<d|d>),
            # snr}, the batched one through the container's sub-box slicing
            name = "production" if tag == "prod" else tag
            row[f"snr_{name}"] = opt
            row[f"data_{name}"] = dict(mm=row[f"{tag}_mm_data"], logL=ll, snr_ratio=row[f"{tag}_snr_ratio"],
                                       snr=opt)
            host[tag] = np.asarray(asnumpy(h))
            row[f"{tag}_flat_mm_data"], row[f"{tag}_flat_amp_data"] = flat_stats(host[tag], d_host)
            arrs[tag] = h
            row[f"{tag}_score_s"] = time.perf_counter() - t1
            row[f"{tag}_peak_rss_gb"] = peak_rss_gb()
            if backend != "cpu":
                row[f"{tag}_device_pool_gb"] = device_pool_gb(backend)
            print(f"[campaign] {tag}: logL={ll:+.4f} mm_data={row[f'{tag}_mm_data']:.3e} "
                  f"snr_opt/data={row[f'{tag}_snr_ratio']:.6f} resid band SNR "
                  f"{ {b: float(f'{v:.3g}') for b, v in bres.items()} } wall={row[f'{tag}_wall_s']:.0f}s "
                  f"rss={row[f'{tag}_peak_rss_gb']:.2f}GB", flush=True)
        except Exception as exc:   # record and keep going: this is a debugging campaign
            row[f"{tag}_error"] = f"{type(exc).__name__}: {exc}"
            row[f"{tag}_traceback"] = traceback.format_exc()[-4000:]
            print(f"[campaign] {tag} FAILED: {row[f'{tag}_error']}", flush=True)
        # drop the generators (module caches hold their response buffers)
        wrappers._MBH_PHENOM_GEN_CACHE.clear()
        wrappers._MBH_TDIONFLY_GEN_CACHE.clear()
        gen = None
        gc.collect()

    for a, b in PAIRS:
        if a in arrs and b in arrs:
            ac_b = AnalysisContainer(WDMSignal(arrs[b], wdm), sens)
            opt_a, det_a = ac_b.template_snr(WDMSignal(arrs[a], wdm))
            opt_a, det_a = fval(opt_a), fval(det_a)
            aa, ab, bb = opt_a * opt_a, det_a * opt_a, fval(ac_b.inner_product())
            row[f"mm_{a}_{b}"] = 1.0 - ab / np.sqrt(aa * bb) if aa > 0 and bb > 0 else float("nan")
            row[f"dlogL_{a}_{b}"] = row[f"{a}_logL"] - row[f"{b}_logL"]
            row[f"delta_norm_{a}_{b}"] = fval(ac_data.template_snr(WDMSignal(arrs[a] - arrs[b], wdm))[0])
            row[f"flat_mm_{a}_{b}"], row[f"flat_amp_{a}_{b}"] = flat_stats(host[a], host[b])
            if (a, b) == ("batched", "prod"):   # the EMRI harness's fast-vs-production key
                row["mm_vs_production"] = row[f"mm_{a}_{b}"]
            print(f"[campaign] {a} vs {b}: mm={row[f'mm_{a}_{b}']:.3e} dlogL={row[f'dlogL_{a}_{b}']:+.4e} "
                  f"||a-b||={row[f'delta_norm_{a}_{b}']:.3e}", flush=True)
            del ac_b
    if "prod90" in host and "n_start" in geom:
        pw = host["prod90"] ** 2
        rel0 = geom["n_start"] - int(wdm.ind_min_t)
        tot = float(pw.sum())
        row["prod90_power_outside_box"] = (
            float(tot - pw[..., rel0:rel0 + geom["Nt_keep"]].sum()) / tot if tot > 0 else float("nan"))
    row["total_wall_s"] = time.perf_counter() - t_start
    row["peak_rss_gb"] = peak_rss_gb()
    with open(args.out, "a") as f:
        f.write(json.dumps(row, default=jsonable) + "\n")
    print(f"[campaign] done in {row['total_wall_s']:.0f} s, peak RSS {row['peak_rss_gb']:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
