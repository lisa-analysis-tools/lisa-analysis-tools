"""Batched EMRI template speed test: GPU (or CPU) production templates + CPU trajectory fan-out.

For ONE CD1L EMRI, build a batch of template parameter rows the way the sampler uses them
(information-matrix style: +/- steps in each of the 14 parameters, plus jittered walker
rows), and time, per mode threshold:

* ``serial``: each template integrates its own trajectory in the parent process;
* ``pooled``: ``few.trajectory.pool``: every row's inspiral call is captured, the unique
  trajectories are integrated on a pool of CPU worker processes, and the templates are
  then built with every trajectory served from the cache.

Template = the production EMRI response (get_emri_response_wrapper: SPECIAL frame, ICRS L1
orbits, Lagrange order 40, REF-anchored) sliced onto the data grid and transformed to WDM,
on ``--backend`` (cuda12x on the cluster). Reports wall per template, the trajectory-only
share, the precompute cost, the speedup, and a bitwise check that pooled templates equal
serial ones. Appends one JSON line per threshold to ``--out``.

Inputs: only source PARAMETERS and orbits are needed (no data samples are read). Params come
from ``--catalog`` (or ``--catalog fixed``: a built-in source, no files); orbits from the
source's L1 brick when found (``--l1-dir``), else the packaged equal-arm file
(``--orbits equal-arm`` forces it). Env: MOJITO_LIGHT_PATH (default root for both),
START_OFFSET_S (default 5e4).
No files at all:
    python scripts/emri/emri_batch_speed.py --catalog fixed --orbits equal-arm --backend cuda12x --workers 8
Example (cluster GPU, 6mo production grid):
    python scripts/emri/emri_batch_speed.py --src 1 --backend cuda12x --workers 8 --rows 64
Direct-to-WDM template too (table grid Nf=180, dt=20 s; production on the same grid):
    python scripts/emri/emri_batch_speed.py --src 1 --backend cuda12x --dt 20 --workers 8 --rows 64 \
        --direct-table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5
Laptop smoke:
    python scripts/emri/emri_batch_speed.py --src 1 --backend cpu --days 4 --dt 20 --rows 8 --workers 2
"""
import argparse
import json
import os
import socket
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_tof_xyz_threeway as W  # noqa: E402  (catalogue reader constants: PATH, REF)

LAYER_DT = 3600.0


# a plausible plunging EMRI (FEW order, special frame) for runs with no catalogue at hand
FIXED_PARAMS = [1e6, 10.0, 0.9, 10.0, 0.3, 1.0, 1.0, 1.0, 2.0, 0.8, 0.5, 0.1, 0.0, 0.2]


#: the mojito L1 brick load_source read the orbits (and window start) from, or None
LAST_BRICK = None


def find_emri_brick(src, l1_dir=None, catalog=None):
    """The source's mojito L1 brick (``EMRI_*_L1_source{src}_*.h5``: its source-only TDI stream
    and the injection orbits), or ``None``. An explicit ``l1_dir`` is the only place searched;
    otherwise MOJITO_LIGHT_PATH/data/EMRI/L1, then recursively MOJITO_DATA_PATH, MOJITO_INFO_PATH
    and the catalogue's root (two levels above the catalogue file: the mojito cache layout
    catalogues/ + data/)."""
    import glob

    if l1_dir:
        roots = [l1_dir]
    else:
        roots = [os.path.join(os.environ.get("MOJITO_LIGHT_PATH", W.PATH), "data", "EMRI", "L1")]
        roots += [os.environ[k] for k in ("MOJITO_DATA_PATH", "MOJITO_INFO_PATH") if os.environ.get(k)]
        if catalog and catalog != "fixed" and os.path.isfile(catalog):
            roots.append(os.path.dirname(os.path.dirname(os.path.abspath(catalog))))
    pat = f"EMRI_*_L1_source{int(src)}_*.h5"
    for r in roots:
        if not os.path.isdir(r):
            continue
        hits = sorted(glob.glob(os.path.join(r, pat))) or sorted(glob.glob(os.path.join(r, "**", pat), recursive=True))
        if hits:
            return hits[0]
    return None


def _window_start(fp):
    """(brick t0, sample spacing, first window sample): the window starts START_OFFSET_S after
    the brick's first sample (the production wrapper zeroes a buffer at its ends), on a sample."""
    from mojito import MojitoL1File

    ts = MojitoL1File(fp).tdis.time_sampling
    i0 = int(round(float(os.environ.get("START_OFFSET_S", "5e4")) / float(ts.dt)))
    return float(ts.t0), float(ts.dt), i0


#: the production wrapper evaluates the response this far past the window end [s]
PRODUCTION_BUFFER_S = 4e4


def check_window(orb, data_t0, span, kind):
    """Refuse (with the numbers) a window whose end plus the production wrapper's buffer runs past
    the orbits' light-travel-time table: the response is undefined there. The packaged equal-arm
    file ends REF + 697.9 d, the 731-day mojito L1 bricks' tables REF + 730.5 d."""
    t_hi = float(np.asarray(orb.ltt_t)[-1])
    need = data_t0 + span + PRODUCTION_BUFFER_S
    if need > t_hi:
        raise ValueError(
            f"[speed] the {span / 86400.0:g} d window from REF + {(data_t0 - W.REF) / 86400.0:.2f} d (+ the "
            f"{PRODUCTION_BUFFER_S:g} s production buffer) ends {(need - t_hi) / 86400.0:.2f} d after the {kind} "
            f"orbits' light-travel-time table (REF + {(t_hi - W.REF) / 86400.0:.2f} d). Shorten the window "
            f"(production convention: 180 / 360 / 720 d) or, for equal-arm, use the source's L1 brick (--orbits auto/l1).")


def load_source(src, backend, span, catalog=None, l1_dir=None, orbits="auto"):
    """Params (FEW order, special frame), window start, and ICRS orbits.

    ``catalog``: the mojito EMRI catalogue h5 (default under MOJITO_LIGHT_PATH); ``"fixed"``
    uses FIXED_PARAMS. ``orbits``: ``"l1"`` reads the source's L1 brick (orbits + its start
    time; :func:`find_emri_brick`), ``"equal-arm"`` uses the packaged equal-arm file (no data at
    all), ``"auto"`` takes the brick when it is found. Sets ``LAST_BRICK``; the data samples are
    read separately (:func:`load_l1_data`).
    """
    global LAST_BRICK
    from lisatools.detector import EqualArmlengthOrbits, L1Orbits
    from lisatools.sources.utils import icrs_to_ecliptic

    root = os.environ.get("MOJITO_LIGHT_PATH", W.PATH)
    catalog = catalog or os.path.join(root, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    if catalog == "fixed":
        params = list(FIXED_PARAMS)
    else:
        with h5py.File(catalog, "r") as f:
            b = f["Binaries"]
            g = lambda k: float(b[k][src])  # noqa: E731
            lam, beta = icrs_to_ecliptic(g("RightAscension") % (2 * np.pi), g("Declination"))
            params = [g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter"),
                      g("SemiLatusRectum"), g("Eccentricity"), 1.0, g("LuminosityDistance") / 1e3,
                      float(np.pi / 2 - beta), float(lam) % (2 * np.pi),
                      g("PolarAnglePrimarySpin"), g("AzimuthalAnglePrimarySpin"),
                      g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase")]
    fp = None
    if orbits in ("auto", "l1"):
        fp = find_emri_brick(src, l1_dir=l1_dir, catalog=catalog)
        if fp is None and orbits == "l1":
            raise FileNotFoundError(f"no mojito L1 brick for EMRI source {src} (l1_dir={l1_dir}, catalog={catalog})")
    LAST_BRICK = fp
    if fp is None:
        print("[speed] orbits: packaged equal-arm (no L1 brick used)", flush=True)
        data_t0 = W.REF + float(os.environ.get("START_OFFSET_S", "5e4"))
        orb = EqualArmlengthOrbits(force_backend=backend, frame="icrs")
        check_window(orb, data_t0, span, "packaged equal-arm")
        return params, data_t0, orb
    print(f"[speed] orbits: L1 brick {fp}", flush=True)
    t0, dt0, i0 = _window_start(fp)
    data_t0 = t0 + i0 * dt0
    orb = L1Orbits(fp, force_backend=backend, frame="icrs")
    check_window(orb, data_t0, span, "L1 brick")
    if backend != "cpu":
        return params, data_t0, orb          # GPU nodes: keep the full table (no host-side trim)
    # keep the light-travel-time table to [REF - pad, window end + pad] (laptop memory), as the campaign does
    pad = 1e5
    lo = max(W.REF - pad, float(orb.sc_t0))
    hi = min(data_t0 + span + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t)
    m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy()
    orb.ltt_t = lt[m].copy()
    orb.ltt_t0 = float(orb.ltt_t[0])
    orb.configure(linear_interp_setup=True)
    return params, data_t0, orb


def load_l1_data(fp, dt, n):
    """The brick's source-only, noise-free TDI stream on the window of :func:`load_source` (same
    first sample), decimated to ``dt``: a host ``(3, n)`` array, or ``None`` (with a message)
    when the brick ends before the window does. Read through ``MojitoL1File.tdis.xyz_doppler``
    (X2, Y2, Z2 / laser frequency), the accessor the fit's ``L1DataLoader`` uses."""
    from mojito import MojitoL1File

    t0, dt0, i0 = _window_start(fp)
    deci = int(round(dt / dt0))
    if deci < 1 or abs(deci * dt0 - dt) > 1e-9 * dt:
        raise ValueError(f"grid dt={dt} is not a multiple of the brick's {dt0} s sampling")
    xyz = np.asarray(MojitoL1File(fp).tdis.xyz_doppler[i0:i0 + n * deci])[::deci][:n]      # (n, 3)
    if xyz.shape[0] < n:
        print(f"[data] {os.path.basename(fp)} holds {xyz.shape[0]} of the window's {n} samples "
              f"(dt {dt} s): no data comparison", flush=True)
        return None
    return np.ascontiguousarray(xyz.T)


class RunBox:
    """The fit's run box -- layers min_freq..max_freq, ``edge`` pixels cropped at each grid end
    (the launcher's EDGE_CROP_WAVELETS) -- with the scirdv1 XYZ sensitivity plus, with
    ``foreground=True`` (default), the fitted hyperbolic-tangent galactic confusion foreground at
    this grid's observation time (``FittedHyperbolicTangentGalacticForeground``, Tobs = Nf Nt dt;
    the analytic fit, NOT a run's sampled galfor/PSD). Scores full-grid WDM templates
    ``(3, Nf, Nt)``: optimal SNR, and, after :meth:`set_data` with the source's mojito L1 stream
    (source only, noise-free), the match to it."""

    def __init__(self, nf, nt, dt, backend, edge=60, min_freq=2.5e-4, max_freq=2.5e-2, foreground=True):
        from lisatools.domains import WDMSettings
        from lisatools.sensitivity import XYZ2SensitivityMatrix

        layer = nf * dt
        self.nf, self.nt, self.dt, self.backend, self.edge = nf, nt, dt, backend, int(edge)
        self.f_lo, self.f_hi = float(min_freq), float(max_freq)
        self.tobs = float(nf * nt * dt)
        self.foreground = bool(foreground)
        self.dom = WDMSettings(nf, nt, dt, min_freq=min_freq, max_freq=max_freq, min_time=edge * layer,
                               max_time=(nt - edge) * layer, force_backend=backend)
        # the fitted tanh foreground's only parameter is Tobs [s] (stochastic_function default)
        fg = dict(stochastic_params=(self.tobs,)) if self.foreground else {}
        self.sens = XYZ2SensitivityMatrix(self.dom, model="scirdv1", **fg)
        self.d = None
        self.dd = None
        self.brick = None

    def crop(self, arr):
        xp = self.dom.xp
        return xp.ascontiguousarray(xp.asarray(arr)[:, self.dom.active_slice_f, self.dom.active_slice_t])

    def ip(self, box):
        """<a|a> of a box-cropped array."""
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.domains import WDMSignal

        return float(np.real(AnalysisContainer(WDMSignal(box, self.dom), self.sens).inner_product()))

    def snr(self, full):
        return float(np.sqrt(self.ip(self.crop(full))))

    def set_data(self, data_td, brick):
        """The mojito stream ``(3, Nf * Nt)`` (time domain, window samples) -> WDM, cropped."""
        from lisatools.domains import TDSettings, TDSignal, WDMSettings

        xp = self.dom.xp
        n = self.nf * self.nt
        tds = TDSettings(n, self.dt, t0=0.0, force_backend=self.backend)
        full = TDSignal(xp.asarray(data_td), tds).transform(WDMSettings(self.nf, self.nt, self.dt,
                                                                        force_backend=self.backend)).arr
        self.d = self.crop(full)
        self.dd = self.ip(self.d)
        self.brick = os.path.basename(brick)

    @property
    def data_snr(self):
        return float(np.sqrt(self.dd))

    def score(self, full):
        """Template vs the mojito stream: ``mm = 1 - <d|h> / sqrt(<d|d><h|h>)`` (noise weighted, no
        time/phase maximisation), ``logL = -1/2 <d-h|d-h>`` (0 for a perfect template),
        ``snr_ratio = sqrt(<h|h> / <d|d>)`` (the amplitude safeguard)."""
        h = self.crop(full)
        hh = self.ip(h)
        rr = self.ip(self.d - h)
        dh = 0.5 * (self.dd + hh - rr)
        return dict(mm=1.0 - dh / np.sqrt(self.dd * hh), logL=-0.5 * rr, snr_ratio=float(np.sqrt(hh / self.dd)),
                    snr=float(np.sqrt(hh)))

    def match(self, full_a, full_b):
        """Noise-weighted mismatch of two templates on the box, ``1 - <a|b> / sqrt(<a|a><b|b>)``
        (no time/phase maximisation): the fast-vs-production ``mm_vs_production``."""
        a, b = self.crop(full_a), self.crop(full_b)
        aa, bb, rr = self.ip(a), self.ip(b), self.ip(a - b)
        return float(1.0 - 0.5 * (aa + bb - rr) / np.sqrt(aa * bb))

    def describe(self):
        fg = (f"+ fitted tanh galactic foreground at Tobs {self.tobs / 86400.0:g} d" if self.foreground
              else "no galactic foreground")
        return f"scirdv1 XYZ {fg}, {self.f_lo * 1e3:g}-{self.f_hi * 1e3:g} mHz, {self.edge} px edges cropped"


_BOXES = {}


def opt_snr(arr, dt, backend, edge=60, min_freq=2.5e-4, max_freq=2.5e-2, foreground=True):
    """Optimal SNR of a full-grid WDM template on the fit's run box (:class:`RunBox`)."""
    nch, nf, nt = (int(v) for v in arr.shape)
    key = (nf, nt, float(dt), backend, int(edge), float(min_freq), float(max_freq), bool(foreground))
    if key not in _BOXES:
        _BOXES[key] = RunBox(nf, nt, dt, backend, edge=edge, min_freq=min_freq, max_freq=max_freq,
                             foreground=foreground)
    return _BOXES[key].snr(arr)


def batch_rows(params, n_rows, seed=11):
    """Information-matrix rows (+/- a step per parameter; x0 held) then jittered walker rows."""
    p = np.asarray(params, dtype=float)
    step = np.array([1e-6 * p[0], 1e-6 * p[1], 1e-6, 1e-6, 1e-6, 0.0, 1e-4 * p[6],
                     1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    rows = []
    for i in range(len(p)):
        if step[i] == 0.0:
            continue
        for s in (+1, -1):
            q = p.copy()
            q[i] += s * step[i]
            rows.append(q)
    rng = np.random.default_rng(seed)
    while len(rows) < n_rows:
        rows.append(p + rng.normal(size=p.size) * step * 10)
    return [list(map(float, r)) for r in rows[:n_rows]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=int, default=1)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--dt", type=float, default=2.5, help="sample spacing [s] (production 2.5)")
    ap.add_argument("--days", type=float, default=180.0, help="observation span (production 180)")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--thresh", default="1e-3,1e-4,1e-5")
    ap.add_argument("--out", default="emri_batch_speed.jsonl")
    ap.add_argument("--catalog", default=None,
                    help="EMRI catalogue h5 (default: $MOJITO_LIGHT_PATH/catalogues/emri_cat_mojito_lite_processed_MT.hdf5); "
                         "'fixed' = a built-in source, no files")
    ap.add_argument("--l1-dir", default=None, help="dir of EMRI L1 bricks (default: $MOJITO_LIGHT_PATH/data/EMRI/L1)")
    ap.add_argument("--orbits", choices=("auto", "l1", "equal-arm"), default="auto")
    ap.add_argument("--direct-table", default=None,
                    help="WDM lookup table h5: also time the direct-to-WDM template (EMRIDirectWDM) and score it "
                         "against the production template. The table's Nf/dt must match the grid "
                         "(laptop table: --dt 20)")
    ap.add_argument("--direct-mode-batch", type=int, default=0, help="0: all modes in one batch")
    ap.add_argument("--direct-response", choices=("spline", "dense"), default="dense")
    args = ap.parse_args()

    from few.trajectory.pool import TrajectoryCache, TrajectoryPool, inspiral_init_kwargs_from, reset_stepper

    from lisatools.domains import TDSettings, TDSignal, WDMSettings
    from lisatools.globalfit.stock import erebor
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri.response import get_emri_response_wrapper

    nf = int(round(LAYER_DT / args.dt))
    nt = int(round(args.days * 86400.0 / LAYER_DT))
    n = nf * nt
    params, data_t0, orb = load_source(args.src, args.backend, n * args.dt, catalog=args.catalog,
                                       l1_dir=args.l1_dir, orbits=args.orbits)
    fit = erebor.get_stock("all_sources")
    off = data_t0 - W.REF
    offset_int = int(round(off / args.dt))
    wg = get_emri_response_wrapper(
        Tobs=(n + offset_int) * args.dt + 4e4, dt=args.dt, t_start=W.REF, t0_shift_to_data=off - offset_int * args.dt,
        tdi_config=TDIConfig(fit.general.tdi_gen_str, force_backend=args.backend), tdi_chan=fit.general.tdi_chan,
        order=fit.emri.response_order, force_backend=args.backend, orbits=orb)
    gen = wg.waveform_gen.waveform_generator
    tds = TDSettings(n, args.dt, t0=0.0, force_backend=args.backend)
    wdm = WDMSettings(nf, nt, args.dt, force_backend=args.backend)
    gpu = args.backend not in ("cpu",)
    if gpu:
        import cupy as cp

    def sync():
        if gpu:
            cp.cuda.Device().synchronize()

    thr_now = {"v": None}

    def template(*p):
        h = wg(*p, mode_selection_threshold=thr_now["v"])
        xp = cp if gpu else np
        h = xp.stack([xp.asarray(c) for c in h]) if isinstance(h, (list, tuple)) else xp.atleast_2d(h)
        h = h[:3, offset_int:offset_int + n]
        return TDSignal(h, tds).transform(wdm).arr

    direct = None
    if args.direct_table:
        from lisatools.domains import WDMLookupTable
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM
        table = WDMLookupTable.from_file(args.direct_table, force_backend=args.backend)
        if int(table.Nf) != nf or abs(float(table.data_dt) - args.dt) > 1e-9:
            raise SystemExit(f"--direct-table is built for Nf={table.Nf} dt={table.data_dt}; this grid is "
                             f"Nf={nf} dt={args.dt} (pass --dt {table.data_dt})")
        direct = EMRIDirectWDM(gen, table, wdm, orbits=orb,
                               tdi_config=TDIConfig(fit.general.tdi_gen_str, force_backend=args.backend),
                               t_start=W.REF, data_t0=data_t0, mode_batch=args.direct_mode_batch or None,
                               force_backend=args.backend, response=args.direct_response)

    def host(x):
        return x.get() if hasattr(x, "get") else np.asarray(x)

    rows = batch_rows(params, args.rows)
    t0 = time.perf_counter()
    pool = TrajectoryPool(n_workers=args.workers, inspiral_init_kwargs=inspiral_init_kwargs_from(gen))
    cache = TrajectoryCache.install(gen)
    thr_now["v"] = 1e-3
    cache.precompute(template, rows[:args.workers], pool)      # spawn + warm every worker
    pool_start = time.perf_counter() - t0
    cache.clear()
    print(f"[speed] src={args.src} backend={args.backend} grid Nf={nf} Nt={nt} dt={args.dt} rows={len(rows)} "
          f"workers={args.workers} pool start-up {pool_start:.1f}s", flush=True)

    for thr in [float(x) for x in args.thresh.split(",")]:
        thr_now["v"] = thr
        rec = dict(src=args.src, backend=args.backend, host=socket.gethostname(), nf=nf, nt=nt, dt=args.dt,
                   rows=len(rows), workers=args.workers, thresh=thr, pool_startup_s=pool_start)
        TrajectoryCache.uninstall(gen)
        reset_stepper(gen.inspiral_generator)
        template(*rows[0])                                       # warm-up (FEW/GPU kernels)
        sync()

        # serial: every template integrates its own trajectory
        t_ser = time.perf_counter()
        for r in rows:
            reset_stepper(gen.inspiral_generator)
            template(*r)
        sync()
        t_ser = time.perf_counter() - t_ser

        # the trajectory-only share of the serial path
        cache = TrajectoryCache.install(gen)
        calls = cache.capture(template, rows)
        TrajectoryCache.uninstall(gen)
        uniq = {}
        from few.trajectory.pool import call_key
        for c in calls:
            if c is not None:
                uniq.setdefault(call_key(*c), c)
        t_traj = time.perf_counter()
        for c in calls:
            if c is not None:
                reset_stepper(gen.inspiral_generator)
                gen.inspiral_generator(*c[0], **c[1])
        t_traj = time.perf_counter() - t_traj

        # pooled: capture + fan-out + collect, then templates from the cache
        cache = TrajectoryCache.install(gen)
        cache.clear()
        t_pre = time.perf_counter()
        info = cache.precompute(template, rows, pool)
        t_pre = time.perf_counter() - t_pre
        t_pool = time.perf_counter()
        for r in rows:
            template(*r)
        sync()
        t_pool = time.perf_counter() - t_pool
        hits, misses = cache.hits, cache.misses

        # bitwise check on a few rows: pooled (cache hit) vs serial (fresh stepper)
        maxdiff = 0.0
        for r in rows[:3]:
            a = template(*r)
            TrajectoryCache.uninstall(gen)
            reset_stepper(gen.inspiral_generator)
            b = template(*r)
            cache = TrajectoryCache.install(gen)
            a, b = (x.get() if hasattr(x, "get") else np.asarray(x) for x in (a, b))
            maxdiff = max(maxdiff, float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)))
        cache.clear()

        if direct is not None:
            TrajectoryCache.uninstall(gen)
            reset_stepper(gen.inspiral_generator)
            h_dir = host(direct(*rows[0], mode_selection_threshold=thr).arr)      # warm-up + accuracy row
            reset_stepper(gen.inspiral_generator)
            h_prod = host(template(*rows[0]))
            act = slice(20, nt - 20)                                             # production edge crop
            dmm, damp = [], []
            for c in range(3):
                a_, b_ = h_dir[c][:, act], h_prod[c][:, act]
                dmm.append(float(1 - np.sum(a_ * b_) / np.sqrt(np.sum(a_ * a_) * np.sum(b_ * b_))))
                damp.append(float(np.linalg.norm(a_) / np.linalg.norm(b_)))
            sync()
            t_dir = time.perf_counter()
            for r in rows:
                reset_stepper(gen.inspiral_generator)
                direct(*r, mode_selection_threshold=thr)
            sync()
            t_dir = time.perf_counter() - t_dir
            rec.update(direct_s=t_dir, direct_per_template_s=t_dir / len(rows), direct_mm_vs_prod=dmm,
                       direct_amp_vs_prod=damp, direct_stats=direct.last_stats)
            print(f"[speed] thr={thr:g} DIRECT {t_dir / len(rows) * 1e3:.0f} ms/tmpl (production serial "
                  f"{t_ser / len(rows) * 1e3:.0f}) mm vs prod {'/'.join(f'{x:.1e}' for x in dmm)} "
                  f"amp {'/'.join(f'{x:.6f}' for x in damp)} {direct.last_stats}", flush=True)
            cache = TrajectoryCache.install(gen)

        rec.update(nmodes=int(getattr(gen, "num_modes_kept", -1)), unique_trajectories=len(uniq),
                   serial_s=t_ser, serial_per_template_s=t_ser / len(rows), traj_only_s=t_traj,
                   traj_share_serial=t_traj / t_ser, precompute_s=t_pre, pooled_templates_s=t_pool,
                   pooled_total_s=t_pre + t_pool, speedup=t_ser / (t_pre + t_pool),
                   cache_hits=hits, cache_misses=misses, precompute_info=info,
                   max_rel_diff_pooled_vs_serial=maxdiff)
        print(f"[speed] thr={thr:g} modes={rec['nmodes']} serial {t_ser:.2f}s ({t_ser / len(rows) * 1e3:.0f} ms/tmpl, "
              f"traj share {t_traj / t_ser:.0%}) | pooled precompute {t_pre:.2f}s + templates {t_pool:.2f}s = "
              f"{t_pre + t_pool:.2f}s  speedup {rec['speedup']:.2f}x  hits {hits}/{len(rows)}  "
              f"unique traj {len(uniq)}  maxdiff {maxdiff:.1e}", flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    pool.close()


if __name__ == "__main__":
    main()
