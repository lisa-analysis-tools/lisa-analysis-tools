#!/usr/bin/env python
"""GPU speed + batching benchmark for the batched, windowed MBH likelihood.

Times the REAL production scoring paths (no re-implementation of either):

* STOCK (``MBH_LIKELIHOOD=full``), one row per call, exactly as the per-row
  container path runs in the global fit:
  ``ResidualAddOneRemoveOneMove.compute_acs_like`` -> ``AnalysisContainer.
  calculate_signal_likelihood`` -> ``PhenomTHMTDIWaveform.get_signals_for_residuals``
  (built by the stock ``get_mbh_phenom_wave_gen``) -> ``template_likelihood``.
  Configurations: response order x phentax generation window T.
* BATCHED (``MBH_LIKELIHOOD=batched``): ``MBHBatchedLikeMove.compute_like`` at
  ``batch_max_size`` B over N = max(2B, 32) rows spread round-robin over the
  walker containers, plus the per-walker expose/fold
  (``remove_cold_chain_sources`` / ``add_back_in_cold_chain_sources``). The
  windowed generator + sub-transform adapter are built exactly as
  ``source_runtime.get_mbh_windowed_gen`` builds them.

Per configuration it reports s/row (an UN-instrumented pass), a device-synced
stage split (a second pass with ``lisatools.utils.device.synchronize`` at every
stage boundary, accumulated through ``lisatools.utils.stagetimer.record``), the
first-call (JIT) time kept OUT of s/row, memory (CuPy pool, JAX
``memory_stats``, device-level peak from a ``cudaMemGetInfo`` sampler thread),
OOM (recorded, and B stops increasing), and an ACCURACY guard: max |logL
batched - logL stock| on the first rows, the stock side being the move's own
cross-check generator (T = window_before, lattice-SNAPPED epoch, the batched
response order), so a speed number never comes from a broken path.

Defaults = the production 6-month run (scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh):
dt 2.5 s, Nf 1440 x Nt 4320 (1-h layers, 180 d), MIN_FREQ 2.5e-4 (v9; job 677 ran the
v8 launcher's 4e-4), MAX_FREQ 2.5e-2, EDGE_CROP_WAVELETS 60 (the active time box),
data-window alpha = the stock fit's default taper (WINDOW_TAPER_WAVELETS = 2 -> alpha =
2 * 2 / Nt, erebor fit.py). Noise weighting (``--foreground``; mbh_harness.py: the
EMRI harness's ``RunBox`` noise, imported from scripts/emri/emri_batch_speed.py):
XYZ2 scirdv1 + the FittedHyperbolicTangentGalacticForeground at Tobs = Nf*Nt*dt (on,
default) or the instrument alone (off; job 677's weighting). Every grid passes the EMRI
harness's ``check_window`` (window end + 4e4 s inside the orbits' ltt table; packaged
equal-arm ends REF + 697.9 d, so 720 d needs a brick or a placement that fits).

    # cluster (one GPU; see submit_mbh_batched_gpu_benchmark.sh)
    python scripts/mbh/mbh_batched_gpu_benchmark.py --backend cuda13x --out-dir OUT
    # scripts/mbh/mbh_speed_durations.sh's speed step: the merger-centred 120-d grid
    python scripts/mbh/mbh_batched_gpu_benchmark.py --backend cuda13x --nt 2880 --merger-day 100 \
        --batch-sizes 1,2,4,8,16,32 --stock-orders 8 --stock-T-days default,window \
        --orbits auto --out-dir OUT --tag src16 --jsonl OUT/speed.jsonl --strict
    # laptop smoke (tiny CPU grid, a few minutes, < 3 GB RSS)
    python scripts/mbh/mbh_batched_gpu_benchmark.py --smoke --backend cpu

``--jsonl`` APPENDS one JSON line per configuration (plus one ``kind=summary``
line) with the grid, source, its optimal SNR, the orbits and the noise model
next to the record; ``--strict`` exits 3 when a configuration errors (OOM
excepted: it is a result), the accuracy guard fails, or no stock / batched
configuration completes.

ORBITS AND PLACEMENT. ``--orbits equal-arm`` (default): EqualArmlengthOrbits, the
merger mid-grid (``--merger-day`` moves it). ``--orbits auto`` / ``l1`` /
``--orbits-file``: the source's own mojito MBHB L1 brick (searched as the EMRI
harness does: ``--l1-dir``, MOJITO_LIGHT_PATH/data/MBHB/L1, then recursively
MOJITO_DATA_PATH, MOJITO_INFO_PATH, the catalogue's root) supplies the orbits --
its light-travel times read over the grid's span only (``WindowedL1Orbits`` of
mbh_batched_mojito_check.py: exact inside the slice, the full-mission 25 M x 6
ltt table never enters memory) -- AND the time frame. With ``--merger-day``
(mbh_speed_durations.sh: 100 d into a 120-d grid) the grid is merger-centred:
it starts that long before the catalogue merger, shifted into the brick near
its ends. Without it the window starts ``--start-offset-s`` (START_OFFSET_S,
5e4 s) after the brick's start, the catalogue places the merger, and a merger
outside ``[start, end + 7 d)`` (the production admission rule) exits 4 (NOT
ADMITTED). ``auto`` without a brick falls back to equal-arm.

The residual content does not affect cost: every container holds the injected
stock template (the same source), so near-truth rows score near logL = 0,
which is also where the accuracy guard is meaningful (far from the posterior
any 1e-8 template difference is amplified by <r|delta>).

Importable (``import mbh_batched_gpu_benchmark``, as mbh_batched_accuracy.py
does for its grid / generator builders): the command line is parsed, and the
process environment and RSS watchdog set up, only when run as a script.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict

DAY = 86400.0
REF_EPOCH = 97729089.327664  # MOJITO_REFERENCE_TIME (checked against recipe.py at import)
HIGHER_MODES = (21, 33, 44)
NTEMPS = 2  # temperature controls only (the move's test pattern); rows carry no rung

# Mojito light v1.0.0 MBHB catalogue (mbhb_cat_mojito_lite_processed_MT_rounding_fixed.hdf5,
# Binaries/<key>[id]); only the fields mbh_catalogue_to_sampling_basis reads.
CATALOGUE = {
    16: dict(
        PrimaryMassSSBFrame=6353521.768, SecondaryMassSSBFrame=3319445.8880000003,
        PrimarySpinCompZ=0.0003555296777408, SecondarySpinCompZ=-0.0280261424292837,
        LuminosityDistance=106935.41534217342, PhaseReferenceSourceFrame=1.2966745697150712,
        InclinationAngle=2.0740226223532474, RightAscension=4.681322677930459,
        Declination=-1.2794628331819926, PolarisationAngle=1.1825694997845275,
        TimeCoalescencePhenomTPHMSSBFrame=9623901.030910421,
    ),
    17: dict(
        PrimaryMassSSBFrame=10458801.18, SecondaryMassSSBFrame=1136216.72,
        PrimarySpinCompZ=0.7266097066250652, SecondarySpinCompZ=0.6075579778343494,
        LuminosityDistance=19711.808522203224, PhaseReferenceSourceFrame=-0.9410828143915088,
        InclinationAngle=1.2292883576129234, RightAscension=3.9808371911125615,
        Declination=0.12866767841950486, PolarisationAngle=1.364812645877738,
        TimeCoalescencePhenomTPHMSSBFrame=50588600.32974778,
    ),
}

# Presets: every flag below defaults to None and resolves CLI > preset.
# ``stock_T_days`` tokens: "default" = the stock MBH_WAVEFORM_DURATION default
# (YRSID/12 ~ 30.4 d), "window" = window_before (the batched T); else days.
PRESETS = {
    "full": dict(
        # min_freq: the v9 6-month launcher's MIN_FREQ (layer 2 of the 1-h grid)
        dt=2.5, nf=1440, nt=4320, min_freq=2.5e-4, max_freq=2.5e-2, edge_crop=60,
        taper_wavelets=2, window_before_days=90.0, window_after_days=10.0,
        window_pad_days=4.0, window_margin_days=1.0,
        batch_sizes="1,2,4,8,16,24,32", min_rows=32,
        stock_orders="8,30", stock_T_days="default,window", stock_rows=5,
        n_containers=4, n_accuracy=3, mem_cap_gb=64.0, git_snapshot=True,
        orbits_dt=0.0, clear_jax_caches=False, warm_expose_fold=True,
    ),
    # WindowedGridAlignedPhentaxTest geometry (tests/test_mbh_windowed_signal_gen.py):
    # 2 days at dt = 10 s (960-s layers), T = 12 h; windows scaled to match.
    # min_freq 1.1e-3 = layer 3 (layer_df 5.2e-4 Hz): layer 1 borders DC, where
    # the instrument PSD diverges (fit.py's min_freq >= 2 layer_df rule; the
    # noise term refuses the NaN layer). Orbits on a 600-s grid: the default
    # 50-s linear-interp grid over the 5-yr equal-arm file is ~2 GB of host RAM.
    # JAX caches are cleared after every generation and between configs (an
    # un-jitted phentax grows the footprint ~0.3 GB per CALL until cleared; see
    # cache_relief), and the warm expose/fold pair is skipped (it only absorbs
    # JIT compiles) -- smoke only.
    "smoke": dict(
        dt=10.0, nf=96, nt=180, min_freq=1.1e-3, max_freq=2.5e-2, edge_crop=4, orbits_dt=600.0,
        taper_wavelets=2, window_before_days=0.5, window_after_days=4.0 / 24,
        window_pad_days=2.0 / 24, window_margin_days=1.0 / 24,
        batch_sizes="1,2", min_rows=4,
        stock_orders="8", stock_T_days="window", stock_rows=2,
        n_containers=2, n_accuracy=2, mem_cap_gb=3.0, git_snapshot=False,
        clear_jax_caches=True, warm_expose_fold=False,
    ),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="cuda13x", choices=["cpu", "cuda12x", "cuda13x"])
    p.add_argument("--smoke", action="store_true", help="tiny CPU config (laptop proof run)")
    g = p.add_argument_group("grid (defaults: production 6mo run)")
    g.add_argument("--dt", type=float)
    g.add_argument("--nf", type=int)
    g.add_argument("--nt", type=int)
    g.add_argument("--min-freq", type=float)
    g.add_argument("--max-freq", type=float)
    g.add_argument("--edge-crop", type=int, help="EDGE_CROP_WAVELETS (active time box)")
    g.add_argument("--taper-wavelets", type=int, help="WINDOW_TAPER_WAVELETS (alpha = 2K/Nt)")
    g.add_argument("--window-alpha", type=float, help="explicit data-window Tukey alpha (overrides the taper)")
    g.add_argument("--foreground", default="on", choices=("on", "off"),
                   help="noise weighting (mbh_harness.py / RunBox): XYZ2 scirdv1 + the fitted tanh galactic "
                        "foreground at Tobs = Nf*Nt*dt (on, default) or the instrument alone (off)")
    w = p.add_argument_group("MBH window / generator")
    w.add_argument("--window-before-days", type=float)
    w.add_argument("--window-after-days", type=float)
    w.add_argument("--window-pad-days", type=float)
    w.add_argument("--window-margin-days", type=float)
    w.add_argument("--buffer-time", type=float, default=15000.0)
    w.add_argument("--order", type=int, default=8, help="batched response order (MBH_RESPONSE_ORDER)")
    w.add_argument("--window-decimate", type=int, default=int(os.environ.get("MBH_WINDOW_DECIMATE", "1")),
                   help="batched window lattice decimation factor (MBH_WINDOW_DECIMATE): generator, response "
                        "and segment transform at decimate*dt on Nf/decimate layers; the epoch snaps onto that lattice "
                        "(the stock references too, as in the run)")
    s = p.add_argument_group("sweep")
    s.add_argument("--batch-sizes", help="comma list of batch_max_size B (run ascending)")
    s.add_argument("--min-rows", type=int, help="N = max(2B, min_rows), rounded up to a multiple of B")
    s.add_argument("--stock-orders", help="comma list of stock response orders")
    s.add_argument("--stock-T-days", help="comma list of stock phentax T: days | default | window")
    s.add_argument("--stock-rows", type=int, help="timed stock rows (after one warm-up row)")
    s.add_argument("--n-containers", type=int, help="walker containers (rows assigned round-robin)")
    s.add_argument("--n-accuracy", type=int, help="rows in the accuracy guard")
    s.add_argument("--acc-tol", type=float, default=0.5, help="flag |dlogL| above this (MBH_CHECK_LL_TOL)")
    s.add_argument("--skip-stock", action="store_true")
    s.add_argument("--skip-batched", action="store_true")
    s.add_argument("--warm-expose-fold", dest="warm_expose_fold", action="store_true", default=None,
                   help="one untimed expose/fold pair before the timed one (default on; off in --smoke)")
    s.add_argument("--no-warm-expose-fold", dest="warm_expose_fold", action="store_false", default=None)
    s.add_argument("--clear-jax-caches", dest="clear_jax_caches", action="store_true", default=None,
                   help="jax.clear_caches() between configs (host-RAM relief; default only in --smoke)")
    s.add_argument("--no-clear-jax-caches", dest="clear_jax_caches", action="store_false", default=None)
    src = p.add_argument_group("source / orbits")
    src.add_argument("--source-id", type=int, default=16, help="mojito MBHB id (16, 17 hardcoded)")
    src.add_argument("--catalogue", help="mojito MBHB catalogue hdf5 (reads Binaries/<key>[source-id])")
    src.add_argument("--epoch", type=float, default=REF_EPOCH, help="waveform_t0 (t_plunge epoch)")
    src.add_argument("--merger-day", type=float,
                     help="merger position in the grid (default: mid-grid; with a brick: centre the grid on "
                          "the catalogue merger this far in, shifted into the file near its ends)")
    src.add_argument("--snap-frac", type=float, default=0.2,
                     help="data_t0 offset from the epoch lattice, in dt (0.2 -> +0.5 s at 2.5 s, as mojito)")
    src.add_argument("--orbits", default="equal-arm", choices=("equal-arm", "auto", "l1"),
                     help="equal-arm (default; synthetic placement), auto (the source's own mojito MBHB L1 "
                          "brick when found, else equal-arm) or l1 (the brick, required): with a brick the "
                          "orbits are its WindowedL1Orbits(frame='icrs') and the window starts "
                          "--start-offset-s after its start (the EMRI harness's run box)")
    src.add_argument("--orbits-file", help="this mojito MBHB L1 brick (overrides --orbits' search)")
    src.add_argument("--l1-dir", help="first place --orbits auto/l1 look for the source's brick")
    src.add_argument("--start-offset-s", type=float, default=float(os.environ.get("START_OFFSET_S", "5e4")),
                     help="with a brick: window start after the brick's tdis t0 [s] (env START_OFFSET_S, "
                          "default 5e4 as the EMRI harness; production MBH starts at the file start, 0)")
    src.add_argument("--orbits-dt", type=float,
                     help="EqualArmlengthOrbits grid step in s (0 = library default 50-s linear-interp grid)")
    src.add_argument("--jitter-scale", type=float, default=1.0)
    src.add_argument("--seed", type=int, default=0)
    o = p.add_argument_group("output / safety")
    o.add_argument("--out-dir", default=".")
    o.add_argument("--tag", default="")
    o.add_argument("--mem-cap-gb", type=float, help="host RSS watchdog cap (0 = off)")
    o.add_argument("--sample-ms", type=float, default=20.0, help="device memory sampler period")
    o.add_argument("--git-snapshot", dest="git_snapshot", action="store_true", default=None,
                   help="record `git stash create` (default on in full mode, off in --smoke)")
    o.add_argument("--no-git-snapshot", dest="git_snapshot", action="store_false", default=None)
    o.add_argument("--jsonl", help="APPEND one JSON line per configuration (+ a summary line) here")
    o.add_argument("--strict", action="store_true",
                   help="exit 3 on a configuration error (OOM excepted), a failed accuracy guard, or no "
                        "completed stock / batched configuration")
    args = p.parse_args(argv)
    preset = PRESETS["smoke" if args.smoke else "full"]
    for k, v in preset.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


#: None when imported as a module (mbh_batched_accuracy.py): the importer parses
#: its own command line with :func:`parse_args` and sets the process environment
#: itself, before its own numpy/jax import.
ARGS = parse_args() if __name__ == "__main__" else None

# ---- process environment BEFORE numpy/jax/lisatools import -----------------
# (mirrored by mbh_batched_accuracy.py's _configure_process: keep them equal)
if ARGS is not None:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    # JAX on demand (lisatools.detector sets the same at import): without it JAX
    # preallocates 75 % of the GPU and every device-level number is that constant.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if ARGS.backend == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        # One XLA CPU thread for execution AND codegen: the shared-laptop CPU
        # budget, and a bounded compile transient (parallel LLVM codegen splits
        # made the smoke's host-RSS peak vary by ~0.8 GB run to run).
        os.environ.setdefault(
            "XLA_FLAGS",
            "--xla_cpu_multi_thread_eigen=false --xla_cpu_parallel_codegen_split_count=1",
        )

import numpy as np  # noqa: E402

_IS_MAC = sys.platform == "darwin"


def rss_gb():
    """PEAK resident set size of this process (ru_maxrss), GB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6


def _watchdog(cap):
    while True:
        if rss_gb() > cap:
            print(f"[watchdog] RSS {rss_gb():.2f} GB > cap {cap} GB -> exit", flush=True)
            os._exit(42)
        time.sleep(0.3)


def start_watchdog(cap_gb):
    """Daemon thread: hard exit 42 once the PEAK RSS passes ``cap_gb`` (0 / None = off)."""
    if cap_gb and float(cap_gb) > 0:
        threading.Thread(target=_watchdog, args=(float(cap_gb),), daemon=True).start()


if ARGS is not None:
    start_watchdog(ARGS.mem_cap_gb)

T_START = time.perf_counter()

try:  # optional: CURRENT rss next to the peak in the progress marks
    import psutil as _psutil

    _PROC = _psutil.Process()
except Exception:  # noqa: BLE001
    _PROC = None


def rss_now_gb():
    return None if _PROC is None else _PROC.memory_info().rss / 1e9


def mark(msg):
    now = rss_now_gb()
    now_s = "" if now is None else f" now {now:5.2f}"
    print(f"[{time.perf_counter() - T_START:7.1f} s | RSS peak {rss_gb():5.2f}{now_s} GB] {msg}", flush=True)


logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("lisatools.globalfit.moves.mbhbatchedmove").setLevel(logging.INFO)

import jax  # noqa: E402

from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray  # noqa: E402
from lisatools.domains import TDSettings, WDMSettings, WDMSignal  # noqa: E402
from lisatools.globalfit.moves.mbhbatchedmove import MBHBatchedLikeMove, mbh_window_layers  # noqa: E402
from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME, mbh_catalogue_to_sampling_basis  # noqa: E402
from lisatools.globalfit.stock.erebor import make_mbh_transform_container  # noqa: E402
from lisatools.globalfit.stock.erebor import wrappers as _erebor_wrappers  # noqa: E402
from lisatools.globalfit.stock.erebor.source_runtime import (  # noqa: E402
    MBH_DEFAULT_WAVEFORM_DURATION,
    SnappedEpochMBHGen,
    snap_waveform_t0_to_lattice,
)
from lisatools.sources.batching import MBHWindowedWDMSignalGen  # noqa: E402
from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform  # noqa: E402
from lisatools.utils import stagetimer  # noqa: E402
from lisatools.utils.device import synchronize  # noqa: E402
from lisatools.utils.utility import asnumpy  # noqa: E402

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbh_harness import DIFFERS_FROM_EMRI, check_window, harness_box, noise_record  # noqa: E402

assert abs(MOJITO_REFERENCE_TIME - REF_EPOCH) < 1e-6, MOJITO_REFERENCE_TIME

ORBITS_NOTE = (
    "orbits: the response cost is insensitive to the orbit model (the same "
    "pyResponseTDI kernel over the same sample count; only the tabulated "
    "positions/light-travel times differ), so EqualArmlengthOrbits timings "
    "stand for the mojito L1 orbits; --orbits-file switches to L1Orbits(frame='icrs')."
)


# ============================================================================
# timing / memory helpers
# ============================================================================
_MISSING = object()


class StageClock:
    """Device-synced per-stage accumulators installed as instance-method wrappers.

    ``wrap(obj, method, key)`` shadows ``obj.method`` with an instance attribute
    that synchronizes the device before and after the call, so asynchronous
    CUDA/XLA work is charged to the stage that launched it; ``unwrap_all``
    deletes the instance attributes (the class methods show through again).
    Every stage span is also recorded into ``lisatools.utils.stagetimer``.
    """

    def __init__(self, sync, prefix):
        self._sync = sync
        self.prefix = prefix
        self.acc = defaultdict(float)
        self.cnt = defaultdict(int)
        self._installed = []

    def wrap(self, obj, name, key):
        orig = getattr(obj, name)
        prev = obj.__dict__.get(name, _MISSING)  # an instance attribute already there
        clock = self

        def timed(*a, **k):
            clock._sync()
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                clock._sync()
                dt = time.perf_counter() - t0
                clock.acc[key] += dt
                clock.cnt[key] += 1
                stagetimer.record(f"{clock.prefix}/{key}", dt)

        setattr(obj, name, timed)
        self._installed.append((obj, name, prev))

    def unwrap_all(self):
        for obj, name, prev in reversed(self._installed):
            try:
                if prev is _MISSING:
                    delattr(obj, name)
                else:
                    setattr(obj, name, prev)
            except AttributeError:
                pass
        self._installed = []


class MemProbe:
    """CuPy pool, JAX memory_stats and a device-level (cudaMemGetInfo) peak sampler.

    ``begin()`` frees the CuPy pools, records the device baseline and resets
    the sampler peak; ``end(start)`` reads everything back. CPU backend: only
    the host peak RSS (process lifetime) is recorded.
    """

    def __init__(self, backend, period_s):
        self.gpu = backend != "cpu"
        self.period = float(period_s)
        self._lock = threading.Lock()
        self._peak = 0
        self._stop = False
        self.total_bytes = None
        if self.gpu:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import cupy as cp  # GPU only

        while not self._stop:
            try:
                free, total = cp.cuda.runtime.memGetInfo()
            except Exception:  # pragma: no cover - transient driver hiccup
                time.sleep(self.period)
                continue
            self.total_bytes = int(total)
            used = int(total) - int(free)
            with self._lock:
                if used > self._peak:
                    self._peak = used
            time.sleep(self.period)

    @staticmethod
    def jax_stats():
        try:
            st = jax.devices()[0].memory_stats()
        except Exception:
            return None
        if not st:
            return None
        return dict(
            in_use_gb=st.get("bytes_in_use", 0) / 1e9,
            peak_gb=st.get("peak_bytes_in_use", 0) / 1e9,
            limit_gb=(st.get("bytes_limit") or 0) / 1e9,
        )

    def _device_used(self):
        import cupy as cp  # GPU only

        free, total = cp.cuda.runtime.memGetInfo()
        return int(total) - int(free)

    def begin(self, sync):
        gc.collect()
        start = dict(jax=self.jax_stats(), rss_gb=rss_gb())
        if self.gpu:
            import cupy as cp  # GPU only

            sync()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            base = self._device_used()
            with self._lock:
                self._peak = base
            start["device_base"] = base
        return start

    def end(self, start, sync):
        out = dict(rss_peak_gb=rss_gb())
        jax1 = self.jax_stats()
        jax0 = start.get("jax")
        out["jax_in_use_gb"] = None if jax1 is None else jax1["in_use_gb"]
        out["jax_peak_gb"] = None if jax1 is None else jax1["peak_gb"]
        out["jax_peak_increase_gb"] = (
            None if (jax1 is None or jax0 is None) else jax1["peak_gb"] - jax0["peak_gb"]
        )
        if self.gpu:
            import cupy as cp  # GPU only

            sync()
            pool = cp.get_default_memory_pool()
            out["cupy_pool_total_gb"] = pool.total_bytes() / 1e9
            out["cupy_pool_used_gb"] = pool.used_bytes() / 1e9
            time.sleep(3 * self.period)  # let the sampler see the tail
            with self._lock:
                peak = self._peak
            base = start["device_base"]
            out["device_base_gb"] = base / 1e9
            out["device_peak_gb"] = peak / 1e9
            out["device_peak_delta_gb"] = (peak - base) / 1e9
            out["device_total_gb"] = None if self.total_bytes is None else self.total_bytes / 1e9
        return out


def cache_relief(fn, args):
    """``fn`` followed by ``jax.clear_caches()`` -- ONLY with --clear-jax-caches (smoke).

    Measured on the laptop (whose phentax runs without ``jax.jit``): every
    eager phentax call grows the process footprint by ~0.3 GB at the SAME
    shapes (the ``fori_loop`` body is a fresh closure per call, compiled and
    retained) until the caches are cleared (1.44 -> 0.51 GB). Never used on a
    GPU run: it would throw away the compiled phentax every call."""
    if not args.clear_jax_caches:
        return fn

    def relieved(*a, **k):
        try:
            return fn(*a, **k)
        finally:
            jax.clear_caches()

    return relieved


def config_start(args, mem, sync, clear=False):
    """Per-config reset: optional JAX cache clear (smoke, batched configs), then the memory baseline."""
    if clear and args.clear_jax_caches:
        try:
            jax.clear_caches()
        except Exception:  # noqa: BLE001 - older jax: nothing to clear
            pass
    return mem.begin(sync)


def safe_mem_end(mem, start, sync):
    """``mem.end`` that cannot raise (a failed config may leave the device unhappy)."""
    try:
        return mem.end(start, sync)
    except Exception as exc:  # noqa: BLE001
        return dict(error=f"{type(exc).__name__}: {exc}", rss_peak_gb=rss_gb())


def is_oom(exc):
    name = type(exc).__name__
    msg = str(exc)
    return (
        isinstance(exc, MemoryError)
        or "OutOfMemory" in name
        or "RESOURCE_EXHAUSTED" in msg
        or "out of memory" in msg.lower()
        or "cudaErrorMemoryAllocation" in msg
        or "CUDA_ERROR_OUT_OF_MEMORY" in msg
    )


# ============================================================================
# environment
# ============================================================================
def _git(*cmd):
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run(["git", *cmd], cwd=here, capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def env_info(args):
    import lisatools

    info = dict(
        python=sys.version.split()[0], platform=platform.platform(), hostname=platform.node(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        xla_preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        xla_flags=os.environ.get("XLA_FLAGS"),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        backend=args.backend, argv=sys.argv,
        numpy=np.__version__, jax=jax.__version__, jax_devices=[str(d) for d in jax.devices()],
        lisatools_file=lisatools.__file__,
        lisatools_version=getattr(lisatools, "__version__", None),
    )
    try:
        import jaxlib

        info["jaxlib"] = jaxlib.__version__
    except Exception:
        pass
    for mod in ("phentax", "eryn"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            info[mod] = None
    if args.backend != "cpu":
        try:
            import cupy as cp  # GPU only

            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            info.update(
                cupy=cp.__version__,
                gpu_name=name.decode() if isinstance(name, bytes) else str(name),
                gpu_total_mem_gb=props["totalGlobalMem"] / 1e9,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                cuda_driver=cp.cuda.runtime.driverGetVersion(),
            )
        except Exception as exc:
            info["cupy_error"] = repr(exc)
    git = dict(head=_git("rev-parse", "HEAD"), branch=_git("rev-parse", "--abbrev-ref", "HEAD"))
    status = _git("status", "--porcelain", "--untracked-files=no")
    git["n_modified_tracked"] = None if status is None else len([s for s in status.splitlines() if s.strip()])
    # ``git stash create`` writes a commit object for the working tree WITHOUT
    # touching the stash list (no ref is updated); empty output = clean tree.
    git["stash_create"] = (_git("stash", "create") or None) if args.git_snapshot else "skipped"
    git["snapshot"] = git["stash_create"] if git["stash_create"] not in (None, "skipped") else git["head"]
    info["git"] = git
    return info


# ============================================================================
# source, grid, generators
# ============================================================================
def load_catalogue(args):
    if args.catalogue:
        import h5py

        with h5py.File(args.catalogue, "r") as f:
            g = f["Binaries"]
            cat = {}
            for k in g.keys():
                v = np.asarray(g[k][args.source_id])
                if v.dtype.kind in "fiu":
                    cat[k] = float(v)
        return cat, f"{args.catalogue}[{args.source_id}]"
    if args.source_id not in CATALOGUE:
        raise SystemExit(f"--source-id {args.source_id} is not hardcoded (16, 17); pass --catalogue")
    return dict(CATALOGUE[args.source_id]), f"hardcoded mojito MBHB id {args.source_id}"


def jitter_rows(truth, n, scale, seed):
    """Row 0 = truth; rows 1.. = NEAR-truth jitters (logL within O(1-10) of the
    truth at SNR ~ 500, the mojito check's near rows): the region a sampler
    scores and where the accuracy guard means something."""
    rng = np.random.default_rng(seed)
    rows = np.tile(np.asarray(truth, dtype=np.float64), (n, 1))
    # waveform basis: m1 m2 s1z s2z dist[Mpc] phi_ref inc psi ra dec t_plunge
    rel = np.array([1e-6, 1e-6, 0, 0, 5e-4, 0, 0, 0, 0, 0, 0]) * scale
    add = np.array([0, 0, 1e-5, 1e-5, 0, 2e-3, 1e-4, 1e-4, 1e-5, 1e-5, 0.3]) * scale
    u = rng.uniform(-1.0, 1.0, size=(n - 1, rows.shape[1]))
    rows[1:] *= 1.0 + rel * u
    rows[1:] += add * u
    return rows


class Ctx:
    """Plain container for the run objects (script-local; never pickled)."""


#: ltt kept beyond the grid's span and the merger (mbh_cd1l_campaign.LTT_PAD)
LTT_PAD = 1.0e5


class NotAdmitted(Exception):
    """The source's merger falls outside the brick-placed window under the production
    admission rule (``window_start <= t_merge < window_end + MBH_MERGER_TIME_BUFFER``)."""


def _campaign():
    """mbh_cd1l_campaign.py (the MBH data loader: brick search, file sampling, the
    windowed ltt reader, the production knobs); side-effect free at import."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import mbh_cd1l_campaign

    return mbh_cd1l_campaign


def windowed_l1_orbits_class(out_dir):
    """``WindowedL1Orbits`` (mbh_batched_mojito_check.py) through the campaign's
    side-effect-neutralising importer (its watchdog off, MBH_CHECK_OUT -> ``out_dir``)."""
    os.makedirs(out_dir, exist_ok=True)
    return _campaign().windowed_orbits_class(os.path.abspath(out_dir))


def resolve_brick(args):
    """The source's mojito MBHB L1 brick for the orbits (and the window start), or None.

    ``--orbits-file`` names it; else ``--orbits auto`` / ``l1`` search for the source's
    own brick (mbh_cd1l_campaign.find_mbhb_brick: ``--l1-dir``, MOJITO_LIGHT_PATH/data/
    MBHB/L1, then recursively MOJITO_DATA_PATH, MOJITO_INFO_PATH, the catalogue's root --
    the EMRI harness's search order); ``l1`` refuses to run without one, ``equal-arm``
    never looks."""
    if args.orbits_file:
        if not os.path.isfile(args.orbits_file):
            raise SystemExit(f"--orbits-file {args.orbits_file}: no such file")
        return args.orbits_file
    if args.orbits == "equal-arm":
        return None
    path = _campaign().find_mbhb_brick(args.source_id, l1_dir=args.l1_dir, catalogue=args.catalogue)
    if path is None and args.orbits == "l1":
        raise SystemExit(f"--orbits l1: no MBHB L1 brick for source {args.source_id} "
                         f"({_campaign().brick_search_roots(args.l1_dir, args.catalogue)})")
    return path


def _place_on_brick(ctx, args, path):
    """EMRI's run box on the brick: data_t0 = the brick's tdis t0 + START_OFFSET_S
    (``--start-offset-s``, rounded to the sample lattice); the merger stays where the
    catalogue puts it and must pass the production admission rule (else NotAdmitted)."""
    camp = _campaign()
    t0_file, dt_file, size = camp.file_sampling(path)
    t_m = ctx.epoch + float(ctx.truth[10])
    if args.merger_day is not None:
        # merger-centred (the campaign's ``centered`` placement): the grid starts
        # --merger-day before the merger, shifted into the file near its ends
        i0 = int(round((t_m - float(args.merger_day) * DAY - t0_file) / ctx.dt))
        hi = int(size * dt_file / ctx.dt) - ctx.N - int(round(camp.CENTERED_END_GUARD_S / ctx.dt))
        if hi < 0:
            raise SystemExit(f"{os.path.basename(path)} holds {size * dt_file / DAY:.2f} d: a "
                             f"{ctx.Tobs / DAY:.2f}-d grid does not fit")
        i0 = min(max(i0, 0), hi)
        ctx.brick_t0, ctx.start_offset_s = t0_file, i0 * ctx.dt
        ctx.placement_note = (f"brick, merger-centred: merger {(t_m - t0_file - i0 * ctx.dt) / DAY:.2f} d in "
                              f"(asked {float(args.merger_day):g})")
        return t0_file + i0 * ctx.dt
    i0 = int(round(float(args.start_offset_s) / ctx.dt))
    if i0 < 0 or (i0 + ctx.N) * ctx.dt > size * dt_file + 1e-6:
        raise SystemExit(
            f"{os.path.basename(path)} holds {size * dt_file / DAY:.2f} d: a {ctx.Tobs / DAY:.2f}-d window "
            f"starting {i0 * ctx.dt:.0f} s in does not fit")
    data_t0 = t0_file + i0 * ctx.dt
    buffer = camp.production_knobs(None)["merger_time_buffer"]
    if not data_t0 <= t_m < data_t0 + ctx.Tobs + buffer:
        raise NotAdmitted(
            f"source {args.source_id} merges {(t_m - data_t0) / DAY:.2f} d into the {ctx.Tobs / DAY:g}-d window "
            f"starting {i0 * ctx.dt:.0f} s after the start of {os.path.basename(path)} (admitted: "
            f"[0, {(ctx.Tobs + buffer) / DAY:g}) d, MBH_MERGER_TIME_BUFFER {buffer / DAY:g} d)")
    ctx.brick_t0, ctx.start_offset_s = t0_file, i0 * ctx.dt
    ctx.placement_note = (f"brick start + {i0 * ctx.dt:.0f} s (START_OFFSET_S); merger "
                          f"{(t_m - data_t0) / DAY:.2f} d in")
    return data_t0


def build_context(args):
    ctx = Ctx()
    ctx.args = args
    ctx.backend = args.backend
    ctx.dt = float(args.dt)
    Nf, Nt = int(args.nf), int(args.nt)
    ctx.Nf, ctx.Nt = Nf, Nt
    ctx.N = Nf * Nt
    ctx.Tobs = ctx.N * ctx.dt
    ctx.layer = Nf * ctx.dt
    ctx.window_alpha = (
        float(args.window_alpha) if args.window_alpha is not None
        else min(1.0, 2.0 * int(args.taper_wavelets) / float(Nt))
    )
    ctx.min_freq, ctx.max_freq = float(args.min_freq), float(args.max_freq)
    ctx.buffer_time = float(args.buffer_time)
    ctx.W_before = float(args.window_before_days) * DAY
    ctx.W_after = float(args.window_after_days) * DAY
    ctx.W_pad = float(args.window_pad_days) * DAY
    ctx.W_margin = float(args.window_margin_days) * DAY
    ctx.epoch = float(args.epoch)

    cat, cat_src = load_catalogue(args)
    ctx.cat, ctx.cat_src = cat, cat_src
    transform = make_mbh_transform_container()
    ctx.truth = np.asarray(
        transform.both_transforms(np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float
    )

    ctx.brick = resolve_brick(args)
    ctx.brick_t0 = ctx.start_offset_s = None
    if ctx.brick is not None:
        # a mojito brick: its orbits AND its time frame; --merger-day centres the grid on
        # the catalogue merger, else it starts --start-offset-s after the brick start
        ctx.data_t0 = _place_on_brick(ctx, args, ctx.brick)
    else:
        # data_t0: merger at ``merger_day`` into the grid; data_t0 sits snap_frac*dt
        # off the epoch's lattice so the snap (and the stock path's sub-sample
        # t0_shift_to_data) is exercised exactly as on mojito (+0.5 s at dt 2.5 s).
        merger_s = (float(args.merger_day) * DAY) if args.merger_day is not None else 0.5 * ctx.Tobs
        ctx.placement_note = "merger at --merger-day" if args.merger_day is not None else "merger mid-grid"
        k = int(np.rint((ctx.truth[10] - merger_s) / ctx.dt))
        ctx.data_t0 = ctx.epoch + k * ctx.dt + float(args.snap_frac) * ctx.dt
    ctx.t_merge_abs = ctx.epoch + ctx.truth[10]
    # the run's rule (get_mbh_phenom_gen / get_mbh_windowed_gen): snap onto the decimate*dt lattice
    ctx.decimate = int(args.window_decimate)
    ctx.t0_snapped, ctx.snap = snap_waveform_t0_to_lattice(ctx.epoch, ctx.data_t0, ctx.dt * ctx.decimate)

    crop = int(args.edge_crop)
    ctx.td = TDSettings(ctx.N, ctx.dt, t0=ctx.data_t0, force_backend=ctx.backend)
    # t0 = data_t0 explicitly: the batched geometry (mbh_window_layers, the
    # adapter's segment start) is expressed in ABSOLUTE time through wdm.t0.
    ctx.wdm = WDMSettings(
        Nf, Nt, ctx.dt, t0=ctx.data_t0, min_freq=ctx.min_freq, max_freq=ctx.max_freq,
        min_time=crop * ctx.layer if crop else None,
        max_time=(Nt - crop) * ctx.layer if crop else None,
        force_backend=ctx.backend,
    )
    if int(ctx.wdm.ind_min_f) < 2:
        raise SystemExit(
            f"min_freq={ctx.min_freq:g} Hz resolves to WDM layer {ctx.wdm.ind_min_f} "
            f"(layer_df={ctx.wdm.layer_df:.4g} Hz): layer 1 borders DC, the instrument PSD "
            "diverges there and the noise term refuses the NaN layer. Use --min-freq >= "
            f"{2 * ctx.wdm.layer_df:.4g} Hz (erebor fit.py rule: min_freq >= 2 layer_df)."
        )
    ctx.xp = ctx.wdm.xp
    ctx.sync = lambda: synchronize(ctx.xp)
    # what weights every inner product (the containers; mbh_batched_accuracy.py too)
    # (mbh_harness.py: the EMRI harness's RunBox on this grid -- scirdv1 XYZ + the fitted tanh
    # galactic foreground at Tobs = Nf*Nt*dt unless --foreground off)
    ctx.box = harness_box(ctx.wdm, crop, args.foreground)
    ctx.noise = noise_record(ctx.box)
    ctx.noise_label = ctx.box.describe()

    if ctx.brick is not None:
        # the ltt table read over the grid's span only (mbh_cd1l_campaign.py's
        # slice: LTT_PAD beyond the data and the merger), positions whole
        wl1 = windowed_l1_orbits_class(args.out_dir)
        lo = ctx.data_t0 - LTT_PAD
        hi = max(ctx.data_t0 + ctx.Tobs, ctx.t_merge_abs) + LTT_PAD
        ctx.orbits = wl1(ctx.brick, lo, hi, force_backend=ctx.backend, frame="icrs")
        ctx.orbits_desc = (f"WindowedL1Orbits({os.path.basename(ctx.brick)}, frame='icrs', "
                           f"ltt slice [{lo:.0f}, {hi:.0f}] s)")
    else:
        from lisatools.detector import EqualArmlengthOrbits

        if args.orbits_dt:
            ctx.orbits = EqualArmlengthOrbits(
                force_backend=ctx.backend, linear_interp_setup=False, dt=float(args.orbits_dt)
            )
            ctx.orbits_desc = f"EqualArmlengthOrbits(dt={float(args.orbits_dt):g} s grid)"
        else:
            ctx.orbits = EqualArmlengthOrbits(force_backend=ctx.backend)
            ctx.orbits_desc = "EqualArmlengthOrbits() (default 50-s linear-interp grid)"
    ensure = getattr(ctx.orbits, "_ensure_configured", None)
    if callable(ensure):
        ensure()
    # the EMRI harness's rule: the window (+ the production wrapper's 4e4 s) must end inside
    # the orbits' light-travel-time table -- a loud refusal, never an extrapolated response
    check_window(ctx.orbits, ctx.data_t0, ctx.Tobs, "L1 brick" if ctx.brick is not None else "packaged equal-arm")
    return ctx


def build_stock_gen(ctx, waveform_t0, T, order):
    """The stock ``PhenomTHMTDIWaveform`` through the stock builder itself."""
    return _erebor_wrappers.get_mbh_phenom_wave_gen(
        data_td_settings=ctx.td, waveform_t0=float(waveform_t0), dt=ctx.dt, orbits=ctx.orbits,
        output_domain_settings=ctx.wdm, tukey_alpha=ctx.window_alpha, force_backend=ctx.backend,
        waveform_duration=float(T), data_span=ctx.Tobs, higher_modes=HIGHER_MODES,
        phenom_tol=1e-12, start_freq=7e-5, response_order=int(order), buffer_time=ctx.buffer_time,
        tdi_gen_str="2nd generation", tdi_chan="XYZ", min_freq=ctx.min_freq, max_freq=ctx.max_freq,
    )


def evict_stock_gen(gen):
    cache = _erebor_wrappers._MBH_PHENOM_GEN_CACHE
    for k, v in list(cache.items()):
        if v is gen:
            del cache[k]


def build_windowed_adapter(ctx):
    """Mirror of ``source_runtime.get_mbh_windowed_gen`` (same generator kwargs)."""
    decimation = int(getattr(ctx, "decimate", 1))
    gen_td = ctx.td if decimation == 1 else TDSettings(
        ctx.N // decimation, ctx.dt * decimation, t0=ctx.data_t0, force_backend=ctx.backend)
    gen = WindowedGridAlignedMBHWaveform(
        waveform_kwargs=dict(
            higher_modes=list(HIGHER_MODES), include_negative_modes=True,
            t_low_fit=True, coarse_grain=False, atol=1e-12, rtol=1e-12,
        ),
        Tobs=ctx.W_before, start_freq=7e-5, use_reference_time=True,
        waveform_t0=ctx.t0_snapped, data_td_settings=gen_td,
        tdi_generation="2nd generation", tdi_channels="XYZ", sampling_frequency=1.0 / gen_td.dt,
        orbits=ctx.orbits, order=int(ctx.args.order), tukey_alpha=ctx.window_alpha, stft_dt=None,
        freq_min=ctx.min_freq, freq_max=ctx.max_freq, fft_batch_size=1,
        buffer_time=ctx.buffer_time, output_domain_settings=ctx.wdm, force_backend=ctx.backend,
    )
    # the wiring passes the absolute layer-0 time explicitly (``t0_abs`` =
    # data_t0) where the adapter takes it; here wdm.t0 IS data_t0 either way
    import inspect

    extra = {}
    if "t0_abs" in inspect.signature(MBHWindowedWDMSignalGen.__init__).parameters:
        extra["t0_abs"] = float(ctx.data_t0)
    if decimation != 1:
        extra["decimate"] = decimation
    adapter = MBHWindowedWDMSignalGen(
        gen, ctx.wdm, nchannels=3, tukey_alpha=ctx.window_alpha, **extra
    )
    adapter.waveform_t0 = ctx.t0_snapped
    adapter.t_plunge_snap = ctx.snap
    return gen, adapter


def build_containers(ctx, h_inj):
    acs_list = []
    for _ in range(int(ctx.args.n_containers)):
        data = WDMSignal(ctx.xp.array(h_inj.arr, copy=True), h_inj.settings)
        # one RunBox noise matrix per walker container (ctx.box's construction, fresh arrays)
        acs_list.append(AnalysisContainer(data, harness_box(ctx.wdm, ctx.args.edge_crop, ctx.args.foreground).sens))
    return AnalysisContainerArray(acs_list, gpus=None if ctx.backend == "cpu" else [0])


def build_move(ctx, acs, adapter, ref_gen, B):
    """Built as tests/test_mbh_batched_move.py builds it; ``waveform_gen`` is
    the production cross-check generator (snapped stock, T = window_before)."""
    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, uniform_dist

    nw = int(ctx.args.n_containers)
    betas = 1 / 1.2 ** np.arange(NTEMPS)
    priors = {"mbh": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(11)})}
    move = MBHBatchedLikeMove(
        "mbh", (NTEMPS, nw, 1, 11), cache_relief(ref_gen.get_signals_for_residuals, ctx.args),
        {}, {}, acs, 1, None,
        priors, [(StretchMove(), 1.0)], betas_all=np.tile(betas, (1, 1)),
        batched_gen=adapter, batch_max_size=int(B), name=f"mbh batched bench B={B}",
        window_before=ctx.W_before, window_after=ctx.W_after,
        window_pad=ctx.W_pad, window_margin=ctx.W_margin,
    )
    move._current_leaf = 0
    if ctx.args.clear_jax_caches:  # smoke only: every batched generation (scoring AND fill)
        move._generate = cache_relief(move._generate, ctx.args)
    return move


def rows_for(B, min_rows):
    n = max(2 * int(B), int(min_rows))
    return int(-(-n // int(B)) * int(B))


def parse_T_list(tokens, ctx):
    out = []
    for tok in str(tokens).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "default":
            out.append(float(MBH_DEFAULT_WAVEFORM_DURATION))
        elif tok == "window":
            out.append(ctx.W_before)
        else:
            out.append(float(tok) * DAY)
    return out


# ============================================================================
# benchmark sections
# ============================================================================
def timed(ctx, fn):
    ctx.sync()
    t0 = time.perf_counter()
    out = fn()
    ctx.sync()
    return out, time.perf_counter() - t0


def run_stock(ctx, move, rows, idx, mem):
    args = ctx.args
    orders = [int(o) for o in str(args.stock_orders).split(",") if o.strip()]
    Ts = parse_T_list(args.stock_T_days, ctx)
    n = int(args.stock_rows)
    out = []
    for order in orders:
        for T in Ts:
            label = f"stock o{order} T{T / DAY:.4g}d"
            rec = dict(kind="stock", label=label, order=order, T_days=T / DAY, rows=n,
                       epoch="unsnapped (production full path)", oom=False, error=None)
            mark(f"=== {label}: building")
            start = config_start(args, mem, ctx.sync)
            gen = None
            sg = None
            clock = StageClock(ctx.sync, label)

            def score(i):
                return move.compute_acs_like(rows[i:i + 1], idx[i:i + 1], signal_gen=sg)[0]

            try:
                gen = build_stock_gen(ctx, ctx.epoch, T, order)
                sg = cache_relief(gen.get_signals_for_residuals, args)

                _, rec["first_call_s"] = timed(ctx, lambda: score(0))
                sel = list(range(1, 1 + n))
                ll, el = timed(ctx, lambda: [score(i) for i in sel])
                rec["s_per_row"] = el / n
                rec["logL"] = [float(v) for v in ll]
                clock.wrap(gen, "wave_gen", "pol")
                clock.wrap(gen, "_apply_response", "resp")
                clock.wrap(gen, "_td_to_output_domain", "xform")
                for ac in move.acs.acs.flatten():
                    clock.wrap(ac, "template_likelihood", "like")
                ll_i, el_i = timed(ctx, lambda: [score(i) for i in sel])
                clock.unwrap_all()
                rec["s_per_row_instrumented"] = el_i / n
                rec["stages_s_per_row"] = {k: v / n for k, v in clock.acc.items()}
                rec["stage_calls"] = dict(clock.cnt)
                rec["instrumented_vs_clean_max_abs_dlogL"] = float(np.max(np.abs(np.asarray(ll_i) - np.asarray(ll))))
                rec["jit_compile_est_s"] = rec["first_call_s"] - rec["s_per_row"]
                rec["mem"] = safe_mem_end(mem, start, ctx.sync)
                mark(f"{label}: {rec['s_per_row']:.4g} s/row (first call {rec['first_call_s']:.3g} s); "
                     f"stages {fmt_stages(rec['stages_s_per_row'])}")
            except Exception as exc:  # noqa: BLE001 - recorded, sweep continues
                clock.unwrap_all()
                rec["oom"] = is_oom(exc)
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["mem"] = safe_mem_end(mem, start, ctx.sync)
                mark(f"{label}: {'OOM' if rec['oom'] else 'ERROR'} -- {rec['error'][:300]}")
                if not rec["oom"]:
                    traceback.print_exc()
            finally:
                if gen is not None:
                    evict_stock_gen(gen)
                gen = sg = None  # the closure cell too: nothing keeps the generator alive
                gc.collect()
            rec["stagetimer"] = stagetimer.report_timing(reset=True)
            out.append(rec)
    return out


def accuracy_reference(ctx, move, rows, idx, state):
    """Stock logL of the first rows through the move's cross-check path
    (``compute_check_like``: the container path with the move's ``waveform_gen``
    = snapped stock, T = window_before), on the UNEXPOSED residual every
    batched config also scores against. Run once, outside any timed config."""
    k = int(ctx.args.n_accuracy)
    mark(f"accuracy reference: {k} rows through the stock cross-check generator")
    ll_ref, t_ref = timed(ctx, lambda: move.compute_check_like(rows[:k], idx[:k]))
    state["ll_ref"] = np.asarray(ll_ref, dtype=float).reshape(-1)
    state["ll_ref_s_per_row"] = t_ref / k


def run_batched(ctx, acs, windowed, adapter, ref_gen, rows, idx, cold, mem, state):
    args = ctx.args
    Bs = sorted({int(b) for b in str(args.batch_sizes).split(",") if b.strip()})
    k_acc = int(args.n_accuracy)
    out = []
    stop = False
    ll_first = None
    for B in Bs:
        N = rows_for(B, args.min_rows)
        label = f"batched o{args.order} T{ctx.W_before / DAY:.4g}d" + (
            f" decim{ctx.decimate}" if ctx.decimate != 1 else "")
        rec = dict(kind="batched", label=label, B=B, rows=N, order=int(args.order),
                   T_days=ctx.W_before / DAY, decimate=int(ctx.decimate), oom=False, error=None)
        if stop:
            rec["skipped"] = "a smaller B ran out of memory"
            out.append(rec)
            continue
        mark(f"=== {label} B={B}: N={N} rows over {args.n_containers} containers")
        start = config_start(args, mem, ctx.sync, clear=True)
        mark(f"B={B}: config baseline (after cache clear / pool free)")
        clock = StageClock(ctx.sync, f"{label} B{B}")
        move = None
        exposed = False
        try:
            move = build_move(ctx, acs, adapter, ref_gen, B)
            move.setup_likelihood_here(cold)  # leaf window + per-walker offset
            # first call of this batch shape: phentax (vmap over B) compiles here
            _, rec["first_call_s"] = timed(ctx, lambda: move.compute_like(rows[:B], idx[:B]))
            mark(f"B={B}: first call {rec['first_call_s']:.3g} s")
            move._stats = move._new_stats()
            ll, el = timed(ctx, lambda: move.compute_like(rows[:N], idx[:N]))
            ll = np.asarray(ll, dtype=float)
            st = dict(move._stats)
            rec["s_per_row"] = el / N
            rec["move_telemetry"] = dict(  # the move's own [MBH_BATCH] split (clean pass)
                gen_s_per_row=st["gen_s"] / N, score_s_per_row=st["score_s"] / N,
                chunks=st["chunks"], fallbacks=st["fallbacks"],
                outside_box=st["outside_box"], outside_data=st["outside_data"],
            )
            rec["n_batch_fallbacks"] = int(move.n_batch_fallbacks)
            clock.wrap(windowed, "_aligned_polarizations", "pol")
            clock.wrap(windowed, "_apply_response", "resp")
            clock.wrap(windowed, "_zero_onset_warmup", "onset")
            clock.wrap(adapter, "_to_domain", "xform")
            clock.wrap(move, "_score_templates", "score")
            ll_i, el_i = timed(ctx, lambda: move.compute_like(rows[:N], idx[:N]))
            clock.unwrap_all()
            rec["s_per_row_instrumented"] = el_i / N
            rec["stages_s_per_row"] = {k: v / N for k, v in clock.acc.items()}
            rec["stage_calls"] = dict(clock.cnt)
            rec["instrumented_vs_clean_max_abs_dlogL"] = float(np.max(np.abs(np.asarray(ll_i) - ll)))
            rec["jit_compile_est_s"] = rec["first_call_s"] - B * rec["s_per_row"]
            # expose/fold: one warm pair (compiles any new chunk shape), one timed pair
            exposed = True
            if args.warm_expose_fold:
                move.remove_cold_chain_sources(cold)
                move.add_back_in_cold_chain_sources(cold)
            _, t_exp = timed(ctx, lambda: move.remove_cold_chain_sources(cold))
            _, t_fold = timed(ctx, lambda: move.add_back_in_cold_chain_sources(cold))
            exposed = False
            nw = int(args.n_containers)
            rec["expose_s_per_walker"] = t_exp / nw
            rec["fold_s_per_walker"] = t_fold / nw
            rec["mem"] = safe_mem_end(mem, start, ctx.sync)
            # ---- accuracy guard (after the memory read-out) -----------------
            if state.get("ll_ref") is None:  # normally computed after the stock section
                accuracy_reference(ctx, move, rows, idx, state)
            d = ll[:k_acc] - state["ll_ref"]
            rec["accuracy"] = dict(
                max_abs_dlogL=float(np.max(np.abs(d))), dlogL=[float(x) for x in d],
                logL_batched=[float(x) for x in ll[:k_acc]],
                logL_stock=[float(x) for x in state["ll_ref"]],
                ok=bool(np.max(np.abs(d)) <= float(args.acc_tol)),
            )
            if ll_first is None:
                ll_first = (B, ll)
            m = min(ll.size, ll_first[1].size)
            rec["max_abs_dlogL_vs_B%d" % ll_first[0]] = float(np.max(np.abs(ll[:m] - ll_first[1][:m])))
            rec["logL"] = [float(x) for x in ll]
            mark(f"B={B}: {rec['s_per_row']:.4g} s/row (first call {rec['first_call_s']:.3g} s); "
                 f"stages {fmt_stages(rec['stages_s_per_row'])}; expose/fold "
                 f"{rec['expose_s_per_walker']:.3g}/{rec['fold_s_per_walker']:.3g} s/walker; "
                 f"acc max|dlogL| {rec['accuracy']['max_abs_dlogL']:.3g}")
        except Exception as exc:  # noqa: BLE001 - recorded; OOM stops the sweep
            clock.unwrap_all()
            rec["oom"] = is_oom(exc)
            rec["error"] = f"{type(exc).__name__}: {exc}"
            if "mem" not in rec:
                rec["mem"] = safe_mem_end(mem, start, ctx.sync)
            mark(f"B={B}: {'OOM' if rec['oom'] else 'ERROR'} -- {rec['error'][:300]}")
            if rec["oom"]:
                stop = True
            else:
                traceback.print_exc()
            if exposed and move is not None:
                try:  # leave the residual as the next config expects it
                    move.add_back_in_cold_chain_sources(cold)
                except Exception:  # noqa: BLE001
                    stop = True
                    rec["error"] += " | residual restore failed; sweep stopped"
        finally:
            move = None
            gc.collect()
        rec["stagetimer"] = stagetimer.report_timing(reset=True)
        out.append(rec)
    return out


# ============================================================================
# reporting
# ============================================================================
def fmt_stages(st):
    return ", ".join(f"{k} {v:.3g}" for k, v in st.items())


def _f(v, spec=".3g"):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return format(v, spec)


def make_table(stock, batched, ref_label, ref_s):
    head = ("| config | B | rows | s/row | x vs " + ref_label + " | pol | resp(+onset) | xform | like/score "
            "| other | expose+fold s/walker | 1st call s | CuPy pool GB | JAX peak GB (+inc) "
            "| dev peak GB (+delta) | OOM | acc max abs dlogL |")
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = [head, sep]
    for r in list(stock) + list(batched):
        st = r.get("stages_s_per_row") or {}
        like = st.get("like", st.get("score"))
        resp = None if "resp" not in st else st["resp"] + st.get("onset", 0.0)
        inst = r.get("s_per_row_instrumented")
        other = None if (inst is None or not st) else inst - sum(st.values())
        s = r.get("s_per_row")
        speed = None if (s is None or not ref_s) else ref_s / s
        memr = r.get("mem") or {}
        jax_s = None if memr.get("jax_peak_gb") is None else (
            f"{memr['jax_peak_gb']:.3g} (+{_f(memr.get('jax_peak_increase_gb'))})")
        dev_s = None if memr.get("device_peak_gb") is None else (
            f"{memr['device_peak_gb']:.3g} (+{_f(memr.get('device_peak_delta_gb'))})")
        ef = None
        if r.get("expose_s_per_walker") is not None:
            ef = f"{r['expose_s_per_walker']:.3g}+{r['fold_s_per_walker']:.3g}"
        acc = r.get("accuracy")
        acc_s = "-" if acc is None else (_f(acc["max_abs_dlogL"]) + ("" if acc["ok"] else " FAIL"))
        oom = "OOM" if r.get("oom") else ("skipped" if r.get("skipped") else ("ERR" if r.get("error") else "-"))
        lines.append(
            f"| {r['label']} | {r.get('B', '-')} | {r.get('rows')} | {_f(s, '.4g')} | {_f(speed)} "
            f"| {_f(st.get('pol'))} | {_f(resp)} | {_f(st.get('xform'))} | {_f(like)} | {_f(other)} "
            f"| {_f(ef)} | {_f(r.get('first_call_s'))} | {_f(memr.get('cupy_pool_total_gb'))} "
            f"| {_f(jax_s)} | {_f(dev_s)} | {oom} | {acc_s} |"
        )
    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return repr(o)


# ============================================================================
# main
# ============================================================================
def main():
    args = ARGS
    mode = "smoke" if args.smoke else "full"
    mark(f"mode={mode} backend={args.backend} jax devices={jax.devices()}")
    try:
        ctx = build_context(args)
    except NotAdmitted as exc:
        print(f"[bench] NOT ADMITTED: {exc}; pick a source merging inside the window", flush=True)
        sys.exit(4)
    layer_h = ctx.layer / 3600.0
    mark(f"grid Nf={ctx.Nf} Nt={ctx.Nt} dt={ctx.dt} ({ctx.Tobs / DAY:.2f} d, {layer_h:.3g}-h layers), "
         f"active f [{ctx.wdm.ind_min_f}, {ctx.wdm.ind_max_f}] t [{ctx.wdm.ind_min_t}, {ctx.wdm.ind_max_t}], "
         f"window alpha {ctx.window_alpha:.4g}; data_t0 {ctx.data_t0:.6f}; merger at "
         f"{(ctx.t_merge_abs - ctx.data_t0) / DAY:.3f} d ({ctx.placement_note}); snap {ctx.snap:+.6f} s")
    print(ORBITS_NOTE, f"[{ctx.orbits_desc}]", flush=True)
    print(f"noise weighting: {ctx.noise_label}", flush=True)
    print("truth (waveform basis m1 m2 s1z s2z dist[Mpc] phi_ref inc psi ra dec t_plunge):",
          np.array2string(ctx.truth, precision=8), flush=True)

    n_pool = max([rows_for(int(b), args.min_rows) for b in str(args.batch_sizes).split(",") if b.strip()]
                 + [int(args.stock_rows) + 1, int(args.n_accuracy), int(args.n_containers)])
    rows = jitter_rows(ctx.truth, n_pool, args.jitter_scale, args.seed)
    idx = np.arange(n_pool) % int(args.n_containers)
    cold = rows[: int(args.n_containers)].copy()

    # the production cross-check generator (batched mode): snapped epoch, T = window_before
    ref_raw = build_stock_gen(ctx, ctx.t0_snapped, ctx.W_before, args.order)
    ref_gen = SnappedEpochMBHGen(ref_raw, ctx.snap)
    mark("injecting the truth template (stock, T = window_before, snapped epoch)")
    h_inj = cache_relief(ref_gen.get_signals_for_residuals, args)(*ctx.truth)
    acs = build_containers(ctx, h_inj)
    del h_inj
    # the source's optimal SNR at this duration (sqrt<h|h> of the injected truth over the
    # active box, the harness noise): container 0 holds exactly h_inj before any expose
    ctx.snr = float(np.sqrt(max(float(np.real(asnumpy(acs.acs.flatten()[0].inner_product()))), 0.0)))
    mark(f"[speed] source {args.source_id} optimal SNR at {ctx.Tobs / DAY:g} d: {ctx.snr:.2f} "
         f"(stock 90-d snapped truth template, {ctx.noise_label})")
    windowed, adapter = build_windowed_adapter(ctx)
    geom = mbh_window_layers(ctx.wdm, ctx.t_merge_abs, ctx.W_before, ctx.W_after, ctx.W_pad, ctx.W_margin)
    adapter.set_window(geom["n_start"], geom["Nt_keep"], geom["n_pad_lo"], geom["n_pad_hi"])
    geometry = dict(geom, **adapter.geometry)
    geometry.update(
        merger_layer=(ctx.t_merge_abs - ctx.data_t0) / ctx.layer,
        box_clamped_lo=bool(geom["n_start"] == int(ctx.wdm.ind_min_t)),
        box_clamped_hi=bool(geom["n_start"] + geom["Nt_keep"] == int(ctx.wdm.ind_max_t) + 1),
        segment_samples=int(ctx.Nf * adapter.geometry["Nt_seg"]),
        lattice_samples=int(windowed.window_spec[1]),
    )
    mark(f"geometry: kept layers [{geom['n_start']}, {geom['n_start'] + geom['Nt_keep']}) "
         f"segment [{adapter.geometry['s0']}, {adapter.geometry['s0'] + adapter.geometry['Nt_seg']}) "
         f"pads lo/hi {adapter.geometry['n_pad_lo']}/{adapter.geometry['n_pad_hi']}; "
         f"{geometry['lattice_samples']} lattice samples per row (full grid {ctx.N}); "
         f"clamped lo/hi {geometry['box_clamped_lo']}/{geometry['box_clamped_hi']}")

    mem = MemProbe(args.backend, args.sample_ms / 1e3)
    stock_move = build_move(ctx, acs, adapter, ref_gen, 1)
    stock_move.setup_likelihood_here(cold)
    state = {}
    stock = [] if args.skip_stock else run_stock(ctx, stock_move, rows, idx, mem)
    if not args.skip_batched:
        # stock shapes are still JIT-cached here (and the residual is the one
        # every batched config scores against); on failure the first batched
        # config retries it inside its own error handling
        try:
            accuracy_reference(ctx, stock_move, rows, idx, state)
        except Exception as exc:  # noqa: BLE001
            mark(f"accuracy reference failed ({type(exc).__name__}: {str(exc)[:200]}); retried in the sweep")
            state.pop("ll_ref", None)
    del stock_move
    batched = [] if args.skip_batched else run_batched(
        ctx, acs, windowed, adapter, ref_gen, rows, idx, cold, mem, state)

    ref = None
    for r in stock:
        if r.get("s_per_row") and r["order"] == 8 and abs(r["T_days"] * DAY - ctx.W_before) < 1.0:
            ref = r
    if ref is None:
        ref = next((r for r in stock if r.get("s_per_row")), None)
    ref_label = "n/a" if ref is None else ref["label"]
    ref_s = None if ref is None else ref["s_per_row"]
    table = make_table(stock, batched, ref_label, ref_s)
    notes = [
        ORBITS_NOTE,
        "s/row: un-instrumented pass after one warm-up call (JIT) of the same shape; the "
        "first call is reported separately (1st call s) and excluded from s/row.",
        "stage columns: s/row of a SECOND pass with a device synchronize at every stage "
        "boundary; other = instrumented s/row - sum(stages) (host overhead, stacking, placement).",
        "batched stages: pol = _aligned_polarizations (phentax), resp(+onset) = _apply_response "
        "+ _zero_onset_warmup, xform = MBHWindowedWDMSignalGen._to_domain (segment transform), "
        "score = MBHBatchedLikeMove._score_templates. stock: pol = wave_gen, resp = "
        "_apply_response, xform = _td_to_output_domain (full-grid WDM), like = template_likelihood.",
        "memory: CuPy pool total_bytes at config end (pool freed before each config); JAX peak "
        "is process-lifetime (B ascending, +inc = increase in this config); device peak from "
        "cudaMemGetInfo every --sample-ms (+delta over the config's starting usage).",
        "accuracy: max |logL batched - logL stock| on the first rows; stock = the move's "
        "cross-check generator (T = window_before, snapped epoch, batched order).",
        f"noise weighting (every container): {ctx.noise_label}.",
    ]
    if args.clear_jax_caches:
        notes.append(
            "--clear-jax-caches: jax.clear_caches() after EVERY generation and between configs "
            "(host-RAM relief for an un-jitted phentax; its cost is inside the timings)."
        )
    if args.backend == "cpu":
        notes.append(
            "CPU backend: proves the paths end to end; NOT a performance measurement (if the "
            "installed phentax runs without jax.jit, pol is eager op-by-op dispatch and "
            "dominates every row, and '1st call' contains no compile)."
        )
    print("\n" + table + "\n", flush=True)
    for n_ in notes:
        print("  * " + n_, flush=True)
    if state.get("ll_ref") is not None:
        print(f"  * accuracy reference cost: {state['ll_ref_s_per_row']:.3g} s/row", flush=True)

    result = dict(
        mode=mode, env=env_info(args), args=vars(args),
        grid=dict(Nf=ctx.Nf, Nt=ctx.Nt, dt=ctx.dt, Tobs_days=ctx.Tobs / DAY, layer_s=ctx.layer,
                  ind_min_f=int(ctx.wdm.ind_min_f), ind_max_f=int(ctx.wdm.ind_max_f),
                  ind_min_t=int(ctx.wdm.ind_min_t), ind_max_t=int(ctx.wdm.ind_max_t),
                  window_alpha=ctx.window_alpha, edge_crop=int(args.edge_crop),
                  data_t0=ctx.data_t0, epoch=ctx.epoch, snap=ctx.snap, t0_snapped=ctx.t0_snapped,
                  placement=ctx.placement_note, brick=ctx.brick, brick_t0=ctx.brick_t0,
                  start_offset_s=ctx.start_offset_s),
        window=dict(before_s=ctx.W_before, after_s=ctx.W_after, pad_s=ctx.W_pad, margin_s=ctx.W_margin,
                    buffer_time_s=ctx.buffer_time, geometry=geometry),
        source=dict(id=args.source_id, catalogue_source=ctx.cat_src, catalogue=ctx.cat,
                    truth_waveform_basis=ctx.truth, t_merge_abs=ctx.t_merge_abs, snr_stock90=ctx.snr),
        orbits=ctx.orbits_desc, noise=ctx.noise, rows=rows, data_index=idx,
        stock=stock, batched=batched, reference=dict(label=ref_label, s_per_row=ref_s),
        accuracy_reference=dict(ll=state.get("ll_ref"), s_per_row=state.get("ll_ref_s_per_row")),
        accuracy_ok=all(r["accuracy"]["ok"] for r in batched if r.get("accuracy")),
        table_md=table, notes=notes, wall_s=time.perf_counter() - T_START, peak_rss_gb=rss_gb(),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.out_dir, f"mbh_batched_gpu_benchmark_{mode}_{args.backend}{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    mark(f"wrote {path}")
    failures = strict_failures(args, stock, batched, state)
    if args.jsonl:
        lines = jsonl_lines(ctx, args, result, stock, batched, geometry, failures)
        with open(args.jsonl, "a") as f:
            for line in lines:
                f.write(json.dumps(line, default=_json_default) + "\n")
        mark(f"appended {len(lines)} lines to {args.jsonl}")
    print(f"[bench] {'FAIL' if failures else 'PASS'}: " + ("; ".join(failures) if failures else
          "every configuration completed or ran out of memory, accuracy guard held"), flush=True)
    print("DONE", flush=True)
    if failures and args.strict:
        sys.exit(3)


def strict_failures(args, stock, batched, state):
    """What ``--strict`` fails on (an OOM is a RESULT -- B stops growing -- not a failure)."""
    out = []
    for r in list(stock) + list(batched):
        name = r["label"] + (f" B={r['B']}" if "B" in r else "")
        if r.get("error") and not r.get("oom"):
            out.append(f"{name}: {r['error'][:200]}")
        acc = r.get("accuracy")
        if acc is not None and not acc["ok"]:
            out.append(f"{name}: accuracy guard max|dlogL| {acc['max_abs_dlogL']:.3g} > {args.acc_tol:g}")
    if not args.skip_stock and not any(r.get("s_per_row") for r in stock):
        out.append("no stock configuration completed")
    if not args.skip_batched:
        if not any(r.get("s_per_row") for r in batched):
            out.append("no batched configuration completed")
        if state.get("ll_ref") is None:
            out.append("no accuracy reference (stock cross-check) was computed")
    return out


def jsonl_lines(ctx, args, result, stock, batched, geometry, failures):
    """One flat-ish JSON line per configuration + one summary line (``--jsonl``)."""
    env = result["env"]
    base = dict(
        script="mbh_batched_gpu_benchmark", mode=result["mode"], backend=args.backend, tag=args.tag,
        days=ctx.Tobs / DAY, Nf=ctx.Nf, Nt=ctx.Nt, dt=ctx.dt, edge_crop=int(args.edge_crop),
        active_t_layers=[int(ctx.wdm.ind_min_t), int(ctx.wdm.ind_max_t)],
        source_id=args.source_id, catalogue_source=ctx.cat_src, orbits=ctx.orbits_desc,
        # the EMRI harness's names: brick (file name), tobs_s, foreground (bool), noise, data_snr
        # (None: the speed step reads no mojito stream; the data step scores it)
        brick=None if ctx.brick is None else os.path.basename(ctx.brick), start_offset_s=ctx.start_offset_s,
        **ctx.noise, data_snr=None,
        snr_stock90=ctx.snr,   # the injected truth template's optimal SNR (the source at this duration)
        differs_from_emri=list(DIFFERS_FROM_EMRI),
        placement=ctx.placement_note, merger_day=(ctx.t_merge_abs - ctx.data_t0) / DAY,
        window_days=dict(before=ctx.W_before / DAY, after=ctx.W_after / DAY, pad=ctx.W_pad / DAY,
                         margin=ctx.W_margin / DAY),
        Nt_keep=int(geometry["Nt_keep"]), Nt_seg=int(geometry["Nt_seg"]),
        lattice_samples=int(geometry["lattice_samples"]), grid_samples=int(ctx.N),
        gpu_name=env.get("gpu_name"), hostname=env.get("hostname"), slurm_job_id=env.get("slurm_job_id"),
        git=env["git"]["snapshot"],
    )
    out = []
    for r in list(stock) + list(batched):
        out.append(dict(base, **{k: v for k, v in r.items() if k not in ("logL", "stagetimer")}))
    ok_b = [r for r in batched if r.get("s_per_row")]
    best = min(ok_b, key=lambda r: r["s_per_row"]) if ok_b else None
    out.append(dict(
        base, kind="summary", reference=result["reference"], accuracy_ok=result["accuracy_ok"],
        accuracy_reference_s_per_row=result["accuracy_reference"]["s_per_row"],
        stock_s_per_row={r["label"]: r.get("s_per_row") for r in stock},
        batched_s_per_row={str(r["B"]): r.get("s_per_row") for r in batched},
        best_batched=None if best is None else dict(B=best["B"], s_per_row=best["s_per_row"]),
        failures=failures, wall_s=result["wall_s"], peak_rss_gb=result["peak_rss_gb"],
    ))
    return out


if __name__ == "__main__":
    main()
