#!/usr/bin/env python
"""Batched windowed MBH likelihood vs the stock path on REAL mojito MBHB data.

Laptop (CPU, one id) or cluster GPU (batch sweep + memory). For the catalogue
truth, two near-truth rows (gating) and one far-jitter row (information
only) it reports mismatch between the two templates,
<d|h>, <h|h>, logL of each against the mojito stream, delta logL, the stock
template's power outside the kept box, the kept-layer relative error, and
s/row. User ruling 2026-09-29: "make sure our match/logL will be okay".

    MBHB_ID=16 MBH_BACKEND=cpu python scripts/mbh/mbh_batched_mojito_check.py
    MBHB_ID=16 MBH_BACKEND=cuda12x MBH_BATCH_SIZES=1,4,8,16,24 python scripts/mbh/mbh_batched_mojito_check.py

Edge coverage (merger near / after a data edge, production WDM grid: 1-h
layers, 20-layer edge crop; the window must fit inside the 731-day file, so
id 16 -- merging 111.4 d into its file -- cannot sit >= 114 d into a window):

    MBHB_ID=17 MBH_CHECK_MERGER_AT_DAYS=126 MBH_CHECK_WAVELET_S=3600,4400 \
    MBH_CHECK_EDGE_CROP=20 MBH_CHECK_N_ROWS=2 MBH_BATCH_SIZES=1 python scripts/mbh/mbh_batched_mojito_check.py

MEMORY (8 GB laptop): the stock loader (``L1ProcessingStep``) reads the whole
731-day stream (25.2 M samples x 3) and ``L1Orbits`` the whole 6-link ltt
table (25.2 M x 6). This script reproduces exactly what they produce for the
analysis window but reads ONLY that window:

* data  = ``MojitoL1File(...).tdis.xyz_doppler[start:stop]`` -- the same lazy
  dataset ``L1DataLoader.load_data`` reads with ``[:]`` (X2/Y2/Z2 stacked and
  divided by the file's ``laser_frequency``), time axis
  ``tdis.time_sampling.t0 + i * dt`` (``time_sampling.t()``);
* catalogue = ``Binaries/<key>[MBHB_ID]`` for every key, as
  ``L1DataLoader.load_single_binary`` does (row index = source id);
* orbits = :class:`WindowedL1Orbits`, an ``L1Orbits`` whose ``_setup`` reads
  the ltt table only over ``[lo, hi]`` (the C++ indexes the ltts purely by
  ``(ltt_t0, ltt_dt, n)``, so a contiguous slice is exact inside it). Positions
  and velocities (274 rows) are read whole, as the stock class does.

The window npz is cached in ``MBH_CHECK_OUT``. An RSS watchdog kills the
process above ``MBH_MEM_CAP_GB`` (default 5.0).
"""
import gc
import json
import os
import resource
import sys
import threading
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

MEM_CAP_GB = float(os.environ.get("MBH_MEM_CAP_GB", "5.0"))
_IS_MAC = sys.platform == "darwin"


def rss_gb():
    """PEAK resident set size of this process (ru_maxrss), GB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6


def _watchdog():
    while True:
        if rss_gb() > MEM_CAP_GB:
            print(f"[watchdog] RSS {rss_gb():.2f} GB > cap {MEM_CAP_GB} GB -> exit", flush=True)
            os._exit(42)
        time.sleep(0.3)


def mark(m):
    print(f"[RSS {rss_gb():5.2f} GB] {m}", flush=True)


threading.Thread(target=_watchdog, daemon=True).start()

import h5py  # noqa: E402

from lisatools.detector import L1Orbits  # noqa: E402
from lisatools.globalfit.preprocessing import find_file  # noqa: E402
from lisatools.globalfit.recipe import mbh_catalogue_to_sampling_basis  # noqa: E402
from lisatools.globalfit.stock.erebor import make_mbh_transform_container  # noqa: E402
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal  # noqa: E402
from lisatools.analysiscontainer import AnalysisContainer  # noqa: E402
from lisatools.sensitivity import XYZ2SensitivityMatrix  # noqa: E402
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform  # noqa: E402
from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform  # noqa: E402
from lisatools.sources.batching import MBHWindowedWDMSignalGen  # noqa: E402
from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers  # noqa: E402
from lisatools.globalfit.stock.erebor.source_runtime import snap_waveform_t0_to_lattice  # noqa: E402
from lisatools.utils.utility import asnumpy  # noqa: E402

REF = 97729089.327664                       # MOJITO_REFERENCE_TIME = waveform_t0
PATH = os.environ.get("MOJITO_ROOT", os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"))
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_CAT = os.path.join(PATH, "catalogues", "mbhb_cat_mojito_lite_processed_MT_rounding_fixed.hdf5")
MBHB_ID = int(os.environ.get("MBHB_ID", "16"))
BACKEND = os.environ.get("MBH_BACKEND", "cpu")
DT = 2.5
WINDOW_DAYS = float(os.environ.get("MBH_CHECK_WINDOW_DAYS", "120"))
MERGER_AT_DAYS = float(os.environ.get("MBH_CHECK_MERGER_AT_DAYS", "100"))
# WDM grid: wavelet (layer) duration bounds in s, and the edge crop in layers.
# Defaults keep the original check (12-h layers, no crop). Production is
# "3600,4400" with EDGE_CROP_WAVELETS auto = 20 (erebor fit.py) -- use that
# pair for the edge-coverage cases, whose active box then excludes the crop.
WAVELET_S = tuple(float(x) for x in os.environ.get("MBH_CHECK_WAVELET_S", "43200,64800").split(","))
EDGE_CROP = int(os.environ.get("MBH_CHECK_EDGE_CROP", "0"))
BEFORE, AFTER, MARGIN = 90 * 86400.0, 10 * 86400.0, 86400.0
PAD = float(os.environ.get("MBH_CHECK_PAD_DAYS", "4")) * 86400.0   # brief: raise to 8 if kept err > 1e-4
ORDER = int(os.environ.get("MBH_RESPONSE_ORDER", "8"))
BATCH_SIZES = [int(b) for b in os.environ.get("MBH_BATCH_SIZES", "1,4").split(",")]
N_JITTER = int(os.environ.get("MBH_CHECK_N_ROWS", "4"))   # truth, near+, near-, far (truncates)
NEAR_DT = float(os.environ.get("MBH_CHECK_NEAR_DT", "0.3"))        # s
NEAR_DIST = float(os.environ.get("MBH_CHECK_NEAR_DIST", "5e-4"))   # fractional
NEAR_PHI = float(os.environ.get("MBH_CHECK_NEAR_PHI", "0.002"))    # rad
OUT_DIR = os.environ.get("MBH_CHECK_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbh_batched_check_out"))
os.makedirs(OUT_DIR, exist_ok=True)
# the cache key carries the merger placement: a window cached for another
# MERGER_AT_DAYS is a different slice of the file and must not be reused
CASE_TAG = f"id{MBHB_ID}_{WINDOW_DAYS:g}d_m{MERGER_AT_DAYS:g}d"
# grid tag is for the results/log name only (the cached TD window does not depend on it)
GRID_TAG = "" if (WAVELET_S == (43200.0, 64800.0) and EDGE_CROP == 0) else f"_w{WAVELET_S[0]:g}_crop{EDGE_CROP}"
CACHE = os.path.join(OUT_DIR, f"mbh_mojito_window_{CASE_TAG}.npz")
TRANSFORM = make_mbh_transform_container()
LTT_PAD = 1.0e5    # s of ltt kept beyond the window: covers buffer_time 1.5e4 + light-travel delays


class WindowedL1Orbits(L1Orbits):
    """``L1Orbits`` reading the ltt table only over ``[ltt_lo, ltt_hi]``.

    Identical to the stock ``_setup`` except that the six ltt datasets are
    sliced on read (same ``(t0, dt)`` lattice, contiguous indices), so the
    full-mission 25 M x 6 table (1.2 GB) never enters memory. ``_configure``
    still builds the position grid over the whole mission; outside the ltt
    slice its ``np.interp`` clamps the ltt (only the unit vectors there are
    approximate, far from the analysis window).
    """

    def __init__(self, filename, ltt_lo, ltt_hi, **kwargs):
        # _setup runs inside Orbits.__init__, so the bounds must exist first
        self._ltt_lo, self._ltt_hi = float(ltt_lo), float(ltt_hi)
        super().__init__(filename, **kwargs)

    def _setup(self):
        from lisatools.detector import icrs_to_ecliptic

        with self.open() as f:
            ts = f.ltts.time_sampling
            dt, t0, size = float(ts.dt), float(ts.t0), int(ts.size)
            i0 = max(0, int(np.floor((self._ltt_lo - t0) / dt)))
            i1 = min(size, int(np.ceil((self._ltt_hi - t0) / dt)) + 1)
            self.ltt = np.asarray(f.ltts.ltts[i0:i1])
            self.ltt_t = t0 + np.arange(i0, i1) * dt
            pos_icrs = f.orbits.positions[:]
            vel_icrs = f.orbits.velocities[:]
            if self.frame == "ecliptic":
                self.x_base = icrs_to_ecliptic(pos_icrs)
                self.v_base = icrs_to_ecliptic(vel_icrs)
            else:
                self.x_base = pos_icrs
                self.v_base = vel_icrs
            self.sc_t_base = f.orbits.time_sampling.t()
            self.size_base = self.sc_t_base.shape[0]
            self.dt_base = float(f.orbits.time_sampling.dt)
            self.ltt_dt = ts.dt
            self.sc_dt = f.orbits.time_sampling.dt
            self.ltt_t0 = float(self.ltt_t[0])
            self.sc_t0 = float(self.sc_t_base[0])


def load_catalogue_entry():
    """``L1DataLoader.load_single_binary(Binaries, MBHB_ID, 'MBHB')``, numeric fields."""
    with h5py.File(MBHB_CAT, "r") as f:
        g = f["Binaries"]
        cat = {}
        for k in g.keys():
            v = np.asarray(g[k][MBHB_ID])
            if v.dtype.kind in "fiu":
                cat[k] = float(v)
    return cat


def load_window():
    """(data_td (3, N_WIN), window_t0, cat) -- cached; merger MERGER_AT_DAYS in."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        data_td, window_t0, cat = z["data_td"], float(z["window_t0"]), z["cat"].item()
        # the key carries the placement, but check it: a stale cache must fail
        # loudly, never test a different geometry
        at = (REF + cat["TimeCoalescencePhenomTPHMSSBFrame"] - window_t0) / 86400.0
        if abs(at - MERGER_AT_DAYS) > DT / 86400.0 or data_td.shape[1] != int(round(WINDOW_DAYS * 86400.0 / DT)):
            raise ValueError(f"cache {CACHE} holds merger at {at:.4f} d, {data_td.shape[1]} samples; "
                             f"expected {MERGER_AT_DAYS} d, {WINDOW_DAYS} d -- delete it")
        return data_td, window_t0, cat
    from mojito import MojitoL1File

    cat = load_catalogue_entry()
    abs_merger = REF + cat["TimeCoalescencePhenomTPHMSSBFrame"]
    n_win = int(round(WINDOW_DAYS * 86400.0 / DT))
    path = find_file(MBHB_L1, "MBHB", MBHB_ID)
    mark(f"reading the {WINDOW_DAYS:.0f}-day window of {os.path.basename(path)}")
    with MojitoL1File(path) as f:
        ts = f.tdis.time_sampling
        dt_native, data_t0, size = float(ts.dt), float(ts.t0), int(ts.size)
        assert abs(dt_native - DT) < 1e-9, dt_native
        start = int(round((abs_merger - MERGER_AT_DAYS * 86400.0 - data_t0) / DT))
        if start < 0 or start + n_win > size:
            # never clamp silently: a clamped start moves the merger away from
            # MERGER_AT_DAYS and the case would test a different geometry
            raise ValueError(
                f"MBHB {MBHB_ID}: a {WINDOW_DAYS:g}-day window with the merger "
                f"{MERGER_AT_DAYS:g} d in needs file samples [{start}, {start + n_win}) "
                f"but the file holds [0, {size}) (merger at "
                f"{(abs_merger - data_t0) / 86400:.2f} d into the file); pick another id."
            )
        # same dataset L1DataLoader reads with [:] (X2,Y2,Z2 / laser_frequency)
        data_td = np.ascontiguousarray(np.asarray(f.tdis.xyz_doppler[start:start + n_win]).T)
    window_t0 = data_t0 + start * DT
    mark(f"data_t0 {data_t0:.6f}  window start index {start}  window_t0 {window_t0:.6f}  "
         f"merger at {(abs_merger - window_t0) / 86400:.3f} d into the window")
    np.savez(CACHE, data_td=data_td, window_t0=window_t0, cat=cat)
    gc.collect()
    return data_td, window_t0, cat


def orbits_for(window_t0, tobs):
    orb = WindowedL1Orbits(
        find_file(MBHB_L1, "MBHB", MBHB_ID), window_t0 - LTT_PAD, window_t0 + tobs + LTT_PAD,
        force_backend=BACKEND, frame="icrs",
    )
    orb._ensure_configured()
    return orb


def gen_kwargs(window_t0, n_win, orb, wdm, waveform_t0):
    return dict(
        waveform_kwargs=dict(higher_modes=[21, 33, 44], include_negative_modes=True,
                             t_low_fit=True, coarse_grain=False, atol=1e-12, rtol=1e-12),
        Tobs=BEFORE, start_freq=7e-5, use_reference_time=True, waveform_t0=waveform_t0,
        data_td_settings=TDSettings(n_win, DT, t0=window_t0, force_backend=BACKEND),
        tdi_generation="2nd generation", tdi_channels="XYZ", sampling_frequency=1.0 / DT,
        orbits=orb, order=ORDER, tukey_alpha=0.0, stft_dt=None, freq_min=1e-4, freq_max=2.5e-2,
        fft_batch_size=1, buffer_time=15000.0, output_domain_settings=wdm, force_backend=BACKEND,
    )


def device_mem():
    if BACKEND == "cpu":
        return rss_gb()
    import cupy as cp
    return cp.get_default_memory_pool().used_bytes() / 1e9


def main():
    t_start = time.perf_counter()
    data_td, window_t0, cat = load_window()
    n_win = data_td.shape[1]
    NF, NT, _ = WDMSettings.adjust_to_even_bins(WAVELET_S[0], WAVELET_S[1], DT, n_win * DT)
    n_win = NF * NT
    data_td = data_td[:, :n_win]
    tobs = n_win * DT
    crop = dict(min_time=EDGE_CROP * NF * DT, max_time=(NT - EDGE_CROP) * NF * DT) if EDGE_CROP else {}
    wdm = WDMSettings(NF, NT, DT, t0=window_t0, min_freq=1e-4, max_freq=2.5e-2,
                      force_backend=BACKEND, **crop)
    mark(f"window {tobs / 86400:.1f} d, Nf={NF} Nt={NT} layer={NF * DT / 3600:.2f} h, "
         f"active f layers [{wdm.ind_min_f}, {wdm.ind_max_f}] t layers [{wdm.ind_min_t}, {wdm.ind_max_t}]")
    xp = wdm.xp
    # the data is the mojito MBHB stream itself (noiseless), transformed on the run grid
    d_wdm = TDSignal(
        xp.asarray(data_td), TDSettings(n_win, DT, t0=window_t0, force_backend=BACKEND)
    ).transform(wdm)
    del data_td
    gc.collect()
    ac = AnalysisContainer(d_wdm, XYZ2SensitivityMatrix(wdm, model="scirdv1"))
    # <d|d> over the ACTIVE box: how much MBH signal the (noiseless) data holds there
    dd = float(np.real(asnumpy(ac.inner_product())))
    mark(f"container ready; data <d|d> in the active box {dd:.6e} (SNR {np.sqrt(max(dd, 0.0)):.2f})")

    truth = np.asarray(TRANSFORM.both_transforms(np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float)
    print("truth (waveform basis m1 m2 s1z s2z dist[Mpc] phi_ref inc psi ra dec t_plunge):",
          np.array2string(truth, precision=6), flush=True)
    # Rows (controller ruling 2026-09-30): truth + two NEAR-truth rows gate the
    # acceptance (|logL - logL_truth| = O(1-50), where a sampler actually
    # lives); one FAR-jitter row is reported as non-gating information.
    rng = np.random.default_rng(0)
    rows, kinds = [truth.copy()], ["truth"]
    for sgn in (+1.0, -1.0):
        r = truth.copy()
        r[10] += sgn * NEAR_DT            # t_plunge, s
        r[4] *= 1.0 + sgn * NEAR_DIST     # dist, fractional
        r[5] += sgn * NEAR_PHI            # phi_ref, rad
        rows.append(r)
        kinds.append("near")
    far = truth.copy()
    far[4] *= rng.uniform(0.97, 1.03)
    far[10] += rng.uniform(-20.0, 20.0)
    far[5] += rng.uniform(-0.2, 0.2)
    rows.append(far)
    kinds.append("far")
    rows, kinds = np.asarray(rows)[:N_JITTER], kinds[:N_JITTER]

    orb = orbits_for(window_t0, tobs)
    mark(f"orbits ready (ltt sliced to {len(orb.ltt_t)} pts from {orb.ltt_t0:.1f})")
    t0s, snap = snap_waveform_t0_to_lattice(REF, window_t0, DT)
    # Reference (controller ruling 2026-09-30): the STOCK generator gets the
    # same lattice-snapped epoch (and t_plunge - snap) as the windowed one --
    # the same absolute merger, without the stock path's own sub-sample
    # placement of an off-lattice epoch. MBH_CHECK_STOCK_T0=ref keeps the
    # unsnapped REF variant (the stock run as configured) as an option.
    STOCK_T0 = os.environ.get("MBH_CHECK_STOCK_T0", "snapped")
    stock_t0, stock_shift = (t0s, snap) if STOCK_T0 == "snapped" else (REF, 0.0)
    # Diagnostic (MBH_CHECK_STOCK_CALL=batched): call the stock generator with a
    # one-row BATCH, whose path applies the leading onset ramp like the windowed
    # generator does; the default single-row call ("single", the stock
    # per-leaf path) starts the template abruptly.
    STOCK_CALL = os.environ.get("MBH_CHECK_STOCK_CALL", "single")
    stock = PhenomTHMTDIWaveform(**gen_kwargs(window_t0, n_win, orb, wdm, stock_t0))
    windowed = WindowedGridAlignedMBHWaveform(**gen_kwargs(window_t0, n_win, orb, wdm, t0s))
    adapter = MBHWindowedWDMSignalGen(windowed, wdm, nchannels=3, tukey_alpha=0.0)
    geom = mbh_window_layers(wdm, REF + truth[10], BEFORE, AFTER, PAD, MARGIN)
    adapter.set_window(geom["n_start"], geom["Nt_keep"], geom["n_pad_lo"], geom["n_pad_hi"])
    rel0 = geom["n_start"] - int(wdm.ind_min_t)
    box = slice(rel0, rel0 + geom["Nt_keep"])
    geom_line = (f"snap {snap:+.6f} s; kept layers {geom['n_start']}..{geom['n_start'] + geom['Nt_keep']} "
                 f"of {NT} (Nt_keep={geom['Nt_keep']}, n_pad={geom['n_pad']}, segment "
                 f"{adapter.geometry['s0']}..{adapter.geometry['s0'] + adapter.geometry['Nt_seg']}); "
                 f"merger layer {(REF + truth[10] - window_t0) / (NF * DT):.2f}; "
                 f"active t layers [{wdm.ind_min_t}, {wdm.ind_max_t}]; "
                 f"segment pads lo/hi {geom['n_pad_lo']}/{geom['n_pad_hi']}")
    # edge coverage (2026-09-30): where the merger sits relative to the data's
    # ACTIVE box and the kept box.
    layer_s = NF * DT
    t_m = REF + truth[10]
    act_lo_t = window_t0 + int(wdm.ind_min_t) * layer_s
    act_hi_t = window_t0 + (int(wdm.ind_max_t) + 1) * layer_s
    box_lo_t = window_t0 + geom["n_start"] * layer_s
    box_hi_t = box_lo_t + geom["Nt_keep"] * layer_s
    edge = dict(
        merger_minus_active_lo_days=(t_m - act_lo_t) / 86400.0,
        merger_minus_active_hi_days=(t_m - act_hi_t) / 86400.0,
        merger_in_active_box=bool(act_lo_t <= t_m < act_hi_t),
        box_clamped_lo=bool(geom["n_start"] == int(wdm.ind_min_t)),
        box_clamped_hi=bool(geom["n_start"] + geom["Nt_keep"] == int(wdm.ind_max_t) + 1),
        kept_box_days_in_window=((box_lo_t - window_t0) / 86400.0, (box_hi_t - window_t0) / 86400.0),
        active_box_days_in_window=((act_lo_t - window_t0) / 86400.0, (act_hi_t - window_t0) / 86400.0),
        # the move's [MBH_BATCH] counters: outside_box = the in-data merger
        # (clamped to the active box) falls outside the kept box -> in-data
        # signal cut by the box; outside_data = merger outside the active box
        outside_box_move=bool(max(t_m, act_lo_t) < box_lo_t or min(t_m, act_hi_t) > box_hi_t),
        outside_data_move=bool(t_m < act_lo_t or t_m >= act_hi_t),
    )
    print("edge:", json.dumps(edge), flush=True)
    mark(f"generators ready; {geom_line}")

    results = dict(id=MBHB_ID, backend=BACKEND, order=ORDER, window_days=tobs / 86400, Nf=NF, Nt=NT,
                   window_t0=window_t0, snap=snap, data_dd_active=dd, geometry=dict(geom, **adapter.geometry),
                   geometry_line=geom_line, edge=edge, merger_at_days=MERGER_AT_DAYS,
                   wavelet_s=list(WAVELET_S), edge_crop=EDGE_CROP, pad_days=PAD / 86400.0, stock_t0=STOCK_T0, stock_call=STOCK_CALL, rows=[])
    # ---- stock, one row at a time ------------------------------------------
    stock_tmpl, stock_t = [], []
    for i, r in enumerate(rows):
        t = time.perf_counter()
        rs = r.copy()
        rs[10] -= stock_shift
        if STOCK_CALL == "batched":
            # stock BATCH entry (wave_gen_batch): same waveform, but with the
            # 3000 s leading onset ramp the single-row entry (wave_gen) omits
            # (a B=1 batch returns times (1, N) but channels squeezed to (3, N),
            # which get_signals_for_residuals mis-indexes -- squeeze both here)
            tt, ch = stock.compute_tdi_channels(*rs[:, None])
            tt = tt[0] if tt.ndim == 2 else tt
            ch = ch[0] if ch.ndim == 3 else ch
            h = stock._td_to_output_domain(times_in=tt, signal_in=ch)
        else:
            h = stock.get_signals_for_residuals(*rs)
        stock_t.append(time.perf_counter() - t)
        stock_tmpl.append(h)
        mark(f"stock row {i}: {stock_t[-1]:.1f} s")
        gc.collect()
    mark(f"stock: {np.mean(stock_t):.2f} s/row")
    # ---- batched windowed, batch sweep --------------------------------------
    batched_tmpl = None
    for B in BATCH_SIZES:
        mem0 = device_mem()
        t = time.perf_counter()
        outs = []
        for lo in range(0, N_JITTER, B):
            p = rows[lo:lo + B].copy()
            p[:, 10] -= snap
            outs.append(adapter(*p.T))
        el = time.perf_counter() - t
        peak = device_mem()
        print(f"[batched B={B}] {el / N_JITTER:.3f} s/row, device mem {mem0:.2f} -> {peak:.2f} GB", flush=True)
        results[f"batched_B{B}_s_per_row"] = el / N_JITTER
        results[f"batched_B{B}_mem_gb"] = peak
        if batched_tmpl is None:
            batched_tmpl = [WDMSignal(o.arr[i], o.settings) for o in outs for i in range(o.arr.shape[0])]
        del outs
        gc.collect()
    results["stock_s_per_row"] = float(np.mean(stock_t))
    assert len(batched_tmpl) == N_JITTER, (len(batched_tmpl), N_JITTER)
    # ---- accuracy -----------------------------------------------------------
    for i, (hs, hb) in enumerate(zip(stock_tmpl, batched_tmpl)):
        hs_arr = np.asarray(asnumpy(hs.arr))
        hb_arr = np.asarray(asnumpy(hb.arr))
        scale = float(np.abs(hs_arr).max())
        diff = hb_arr - hs_arr[..., box]
        kept_err = float(np.abs(diff).max() / scale)
        # where the worst kept-layer error sits: (channel, f layer, t layer ABSOLUTE)
        c_e, f_e, t_e = np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape)
        err_where = (int(c_e), int(f_e) + int(wdm.ind_min_f), int(t_e) + geom["n_start"])
        # per-layer (time) error envelope: max over (channel, f) per kept layer
        err_t = np.abs(diff).max(axis=(0, 1)) / scale
        # noise-weighted norm of the difference, ||hb - hs_box||; with residual
        # r = d - hs, dlogL = <r|delta> - ||delta||^2 / 2 exactly
        delta = WDMSignal(diff, hb.settings)
        dnorm = float(ac.template_snr(delta)[0])
        del delta
        p_tot = float((hs_arr ** 2).sum())
        outside = float((np.delete(hs_arr, np.s_[box], axis=-1) ** 2).sum() / p_tot) if p_tot > 0 else float("nan")
        hh_s = float(ac.template_snr(hs)[0]) ** 2       # optimal SNR^2 = <h|h>
        hh_b = float(ac.template_snr(hb)[0]) ** 2
        dh_s = float(np.real(asnumpy(ac.template_inner_product(hs))))
        dh_b = float(np.real(asnumpy(ac.template_inner_product(hb))))
        ll_s = float(np.real(asnumpy(ac.template_likelihood(hs))))
        ll_b = float(np.real(asnumpy(ac.template_likelihood(hb))))
        # mismatch between the two templates: the stock template (as "data" on
        # the run grid) is sliced to the batched template's box by the container,
        # normalized with the noise-weighted norms of both inside that box
        ac_s = AnalysisContainer(hs, XYZ2SensitivityMatrix(wdm, model="scirdv1"))
        O = float(np.real(asnumpy(ac_s.template_inner_product(hb, normalize=True))))
        # <hs|hb>: the part of the stock template's active-box power that the
        # batched template reproduces (1.0 = all of it, 0 = none)
        hs_hb = float(np.real(asnumpy(ac_s.template_inner_product(hb))))
        del ac_s
        # ||delta|| restricted to the first / last EDGE_N kept layers: where an
        # edge-clamped box meets the data edge (a truncation or wrap defect would
        # concentrate there)
        edge_n = min(10, geom["Nt_keep"] // 2)
        dnorm_edges = []
        for sl in (slice(0, edge_n), slice(geom["Nt_keep"] - edge_n, geom["Nt_keep"])):
            sub = wdm.get_slice((slice(0, int(wdm.Nf_active)),
                                 slice(rel0 + sl.start, rel0 + sl.stop)))
            dnorm_edges.append(float(ac.template_snr(WDMSignal(np.ascontiguousarray(diff[..., sl]), sub))[0]))
        row = dict(row=i, kind=kinds[i], gating=kinds[i] != "far", dh_stock=dh_s, dh_batched=dh_b, hh_stock=hh_s, hh_batched=hh_b,
                   logL_stock=ll_s, logL_batched=ll_b, dlogL=ll_b - ll_s,
                   mismatch=1.0 - O, captured_snr2_frac=hh_b / hh_s if hh_s > 0 else float("nan"),
                   captured_overlap_frac=hs_hb / hh_s if hh_s > 0 else float("nan"),
                   stock_resid_frac_of_data=(-2.0 * ll_s / dd) if dd > 0 else float("nan"),
                   kept_layer_rel_err=kept_err, stock_power_outside_box=outside,
                   kept_err_at_c_f_t=err_where, delta_norm=dnorm,
                   delta_norm_first_last_kept_layers=dnorm_edges, edge_layers=edge_n,
                   kept_err_first3_last3_layers=[float(x) for x in np.r_[err_t[:3], err_t[-3:]]],
                   kept_err_median_layer=float(np.median(err_t)))
        results["rows"].append(row)
        print(f"row {i} [{kinds[i]}]: logL stock {ll_s:.3f} batched {ll_b:.3f} dlogL {ll_b - ll_s:+.4f} | "
              f"mm {1 - O:.3e} | kept err {kept_err:.2e} at (c,f,t)={err_where} | "
              f"||delta|| {dnorm:.3e} (first/last {edge_n} kept layers {dnorm_edges[0]:.2e}/{dnorm_edges[1]:.2e}) | "
              f"power outside box {outside:.2e} | "
              f"SNRopt {np.sqrt(hh_s):.2f}/{np.sqrt(hh_b):.2f} (captured SNR^2 frac {hh_b / hh_s:.9f}, "
              f"<hs|hb>/<hs|hs> {hs_hb / hh_s:.9f}) | <d|h> {dh_s:.3f}/{dh_b:.3f} | "
              f"stock ||d-hs||^2/<d|d> {-2.0 * ll_s / dd:.2e}", flush=True)
        # edge diagnostics: per-layer (sum over channel, f) power of the stock
        # template, the batched template and the data at both ends of the
        # ACTIVE box (stock/data coords) -- shows whether the in-data signal
        # at an edge is present in each
        d_arr = np.asarray(asnumpy(d_wdm.arr))
        p_s = (hs_arr ** 2).sum(axis=(0, 1))
        p_b = np.zeros_like(p_s)
        p_b[box] = (hb_arr ** 2).sum(axis=(0, 1))
        row["layer_power_first5_last5"] = dict(
            stock=[float(x) for x in np.r_[p_s[:5], p_s[-5:]]],
            batched=[float(x) for x in np.r_[p_b[:5], p_b[-5:]]],
        )
        if d_arr is not None and d_arr.shape[-1] == p_s.shape[0]:
            p_d = (d_arr ** 2).sum(axis=tuple(range(d_arr.ndim - 1)))
            row["layer_power_first5_last5"]["data"] = [float(x) for x in np.r_[p_d[:5], p_d[-5:]]]
        print("        layer power first5|last5 (active box):",
              {k: np.array2string(np.asarray(v), precision=2) for k, v in row["layer_power_first5_last5"].items()},
              flush=True)
        print(f"        kept err per layer: first3 {np.array2string(err_t[:3], precision=2)} "
              f"last3 {np.array2string(err_t[-3:], precision=2)} median {np.median(err_t):.2e}", flush=True)
    ll_truth = results["rows"][0]["logL_stock"]
    for r in results["rows"]:
        r["logL_stock_minus_truth"] = r["logL_stock"] - ll_truth
    print("logL_stock - logL_truth per row:",
          ", ".join(f"{r['kind']} {r['logL_stock_minus_truth']:+.2f}" for r in results["rows"]), flush=True)
    # mm is RELATIVE: for a source merging after the data end only a sliver of
    # its SNR is in the data (SNR ~ 1), and mm there measures the absolute
    # template-difference floor against a tiny signal. The likelihood only
    # sees the absolute, noise-weighted difference (|dlogL| <= ||d-h||
    # ||delta|| + ||delta||^2 / 2), so a row also passes the template clause
    # when ||delta|| <= DELTA_ABS: the interior truth rows (SNR 536 / 1420,
    # accepted above at mm <= 1.7e-10) carry ||delta|| = 0.010 / 0.016.
    DELTA_ABS = 0.05
    ok = all(abs(r["dlogL"]) <= 0.5 and (r["mismatch"] < 1e-6 or r["delta_norm"] <= DELTA_ABS)
             and r["stock_power_outside_box"] < 1e-4
             for r in results["rows"] if r["gating"])
    results["acceptance"] = bool(ok)
    results["peak_rss_gb"] = rss_gb()
    results["wall_s"] = time.perf_counter() - t_start
    with open(os.path.join(OUT_DIR, f"results_{CASE_TAG}{GRID_TAG}_{BACKEND}_pad{PAD / 86400:g}_{STOCK_T0}_{STOCK_CALL}.json"), "w") as f:
        json.dump(results, f, indent=2)
    mark(f"acceptance on truth+near rows (|dlogL|<=0.5, mm<1e-6 or ||delta||<={DELTA_ABS}, "
         f"outside<1e-4): {'HELD' if ok else 'FAILED'}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
