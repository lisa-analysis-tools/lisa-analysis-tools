"""Three-way X, Y, Z comparison: mojito L1 data / legacy response / TDI-on-the-fly.

Task A0.5 of the EMRI direct-to-WDM plan. On a SHORT window of one CD1L EMRI it
reports, per channel, with NO time or phase maximisation:

* ``1 - Re(O)`` for TOF-vs-data, legacy-vs-data and TOF-vs-legacy (flat
  weighting, Tukey 0.1, full band), and the same restricted to frequency bands
  (the localisation method of scripts/mbh/mbh_tdionfly_mm_vs_freq.py: an
  f-growing residual is a time/phase/reference bug, a flat low-f floor is
  under-resolution);
* SciRD ``dlogL = logL(h) - logL(d)`` and opt/data SNR for both templates.

The legacy template is the production recipe (``get_emri_response_wrapper``:
SPECIAL frame, ICRS orbits, Lagrange order 40, REF-anchored, sliced onto the
data grid), i.e. the base signal of the plan.

Env: SRC (catalogue row, default 1), N_WIN (default 4096 at DT=20 s; 65536 = 15.2 d),
START_OFFSET_S (window start after the data start, default 5e4 s: the legacy
wrapper zeroes t_buffer=3e4 s of edge garbage at both ends of its span),
THRESH (comma list of mode thresholds, default "1e-5,1e-7"), N_FINE (default
max(1024, N_WIN//4), ~80 s spacing), EVAL_CHUNK (4096 samples per TOF eval).
"""

import gc
import os
import resource
import threading
import time

import h5py
import numpy as np
from scipy.signal.windows import tukey

PATH = os.environ.get("MOJITO_LIGHT_PATH", "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/")  # dir with catalogues/ and data/EMRI/L1/
REF = 97729089.327664
SRC = int(os.environ.get("SRC", "1"))
DT = 20.0
N_WIN = int(os.environ.get("N_WIN", "4096"))
N_FINE = int(os.environ.get("N_FINE", str(max(1024, N_WIN // 4))))   # ~80 s trajectory spacing
EVAL_CHUNK = int(os.environ.get("EVAL_CHUNK", "4096"))   # TOF eval in time chunks: bounded memory
MODE_BATCH = int(os.environ.get("MODE_BATCH", "64"))     # modes per TOF call (linear sum)
START_OFFSET_S = float(os.environ.get("START_OFFSET_S", "5e4"))  # clear of the legacy t_buffer garbage
LEG_TAIL_S = 4e4   # legacy span past the window end (t_buffer=3e4 zeroed there)
THRESHES = [float(x) for x in os.environ.get("THRESH", "1e-5,1e-7").split(",")]
BANDS = [(1e-4, 1e-3), (1e-3, 3e-3), (3e-3, 1e-2), (1e-2, 2.5e-2)]


def watchdog(limit_gb=6.5):
    def _run():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > limit_gb:
                os._exit(42)
            time.sleep(0.3)
    threading.Thread(target=_run, daemon=True).start()


def load(src):
    from mojito import MojitoL1File

    from lisatools.detector import L1Orbits
    from lisatools.globalfit.preprocessing import find_file
    from lisatools.sources.utils import icrs_to_ecliptic

    cat = os.path.join(PATH, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]
        g = lambda k: float(b[k][src])
        lam, beta = icrs_to_ecliptic(g("RightAscension") % (2 * np.pi), g("Declination"))
        params = [
            g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter"),
            g("SemiLatusRectum"), g("Eccentricity"), 1.0, g("LuminosityDistance") / 1e3,
            float(np.pi / 2 - beta), float(lam) % (2 * np.pi),
            g("PolarAnglePrimarySpin"), g("AzimuthalAnglePrimarySpin"),
            g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase"),
        ]
    fp = find_file(os.path.join(PATH, "data", "EMRI", "L1"), "EMRI", src)
    ts = MojitoL1File(fp).tdis.time_sampling
    deci = int(round(DT / ts.dt))
    i0 = int(round(START_OFFSET_S / ts.dt))
    data_t0 = float(ts.t0) + i0 * float(ts.dt)
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        data = np.stack([np.asarray(f["tdis"][c][i0: i0 + N_WIN * deci])[::deci][:N_WIN] / lf
                         for c in ("X2", "Y2", "Z2")])
    orb = L1Orbits(fp, force_backend="cpu", frame="icrs")
    pad = 1e5
    lo = max(REF - pad, float(orb.sc_t0))
    hi = min(data_t0 + N_WIN * DT + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t)
    m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy()
    orb.ltt_t = lt[m].copy()
    orb.ltt_t0 = float(orb.ltt_t[0])
    gc.collect()
    orb.configure(linear_interp_setup=True)
    return params, data, data_t0, orb


def _mode_list(params, gen, thr, data_t0):
    """(l, m, k, n) of the modes FEW keeps at ``thr`` for this source."""
    from few.utils.utility import get_viewing_angles

    M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = params
    th, ph = get_viewing_angles(qS, phiS, qK, phiK)
    saved = dict(gen.inspiral_kwargs)
    try:
        lo = data_t0 - REF
        H = gen(M, mu, a, p0, e0, x0, th, ph, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt, Phi_r0=Pr,
                T=(lo + N_WIN * DT + 2000.0) / 3.15581497635456e7, dt=DT, return_sparse_holder=True,
                include_minus_mkn=True, mode_selection_threshold=thr,
                inspiral_kwargs={"upsample": True, "fix_t": True,
                                 "new_t": np.linspace(lo, lo + N_WIN * DT, 256)})
    finally:
        gen.inspiral_kwargs.clear()
        gen.inspiral_kwargs.update(saved)
    return [(int(l), int(m), int(k), int(n)) for l, m, k, n in zip(H.ls, H.ms, H.ks, H.ns)]


def tof_td(params, orb, data_t0, thr, gen):
    """TOF template on the window; ``gen`` is the FEW generator shared with the
    legacy wrapper (one FEW construction per process: each reads the 5.1 GB
    amplitude file whole, a ~6 GB transient footprint).

    The modes are fed in batches of MODE_BATCH (the mode sum is linear: a split
    call reproduces the full call to 1e-15), bounding the response's memory.
    """
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri import EMRITDIonFly

    Tn = data_t0 - REF + N_WIN * DT + 2000.0
    tdi = TDIConfig("2nd generation", force_backend="cpu")
    tg = data_t0 + np.arange(N_WIN) * DT
    modes = _mode_list(params, gen, thr, data_t0)
    td = np.zeros((3, N_WIN))
    nsub_total = 0
    n_in_min = N_WIN
    for j in range(0, len(modes), MODE_BATCH):
        fly = EMRITDIonFly(gen, orb, tdi, DT, Tn, REF, frame="icrs_special", n_fine=N_FINE,
                           t_fine_window=(data_t0, data_t0 + N_WIN * DT))
        out = fly(*params, mode_selection=modes[j:j + MODE_BATCH])
        x = np.asarray(out.x)
        inside = (tg > float(np.max(x[:, 0]))) & (tg < float(np.min(x[:, -1])))
        idx = np.flatnonzero(inside)
        for i in range(0, idx.size, EVAL_CHUNK):   # (num_sub, 3, chunk) per step, summed over subs
            sl = idx[i:i + EVAL_CHUNK]
            td[:, sl] += np.real(np.sum(np.asarray(out.eval_tdi(tg[sl])), axis=0))
        nsub_total += x.shape[0]
        n_in_min = min(n_in_min, int(idx.size))
        del out, fly
        gc.collect()
    return td, nsub_total, n_in_min


def legacy_wrapper(orb, data_t0):
    """Production EMRI response (get_emri_response_wrapper), built ONCE per process.

    Returns ``(wrapper, offset_int, few_generator)``; the FEW generator inside is
    reused by the TOF so the process holds a single FEW construction.
    """
    from lisatools.globalfit.stock import erebor
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri.response import get_emri_response_wrapper

    # the production knobs, read from the stock fit the 6mo run is built from (construction
    # is validation-only; env overrides set by the launchers are honoured)
    fit = erebor.get_stock("all_sources")
    off = data_t0 - REF
    offset_int = int(round(off / DT))
    wg = get_emri_response_wrapper(
        Tobs=(N_WIN + offset_int) * DT + LEG_TAIL_S, dt=DT, t_start=REF,
        t0_shift_to_data=off - offset_int * DT,
        tdi_config=TDIConfig(fit.general.tdi_gen_str, force_backend="cpu"), tdi_chan=fit.general.tdi_chan,
        order=fit.emri.response_order, force_backend="cpu", orbits=orb)
    return wg, offset_int, wg.waveform_gen.waveform_generator


def legacy_td(params, wg, offset_int, thr):
    h = np.atleast_2d(np.asarray(wg(*params, mode_selection_threshold=thr)))[:3]
    h = h[:, offset_int:offset_int + N_WIN]
    if h.shape[-1] < N_WIN:
        h = np.pad(h, ((0, 0), (0, N_WIN - h.shape[-1])))
    return h


def mismatch(a, b, win, ff, band=None):
    A = np.fft.rfft(a * win)
    B = np.fft.rfft(b * win)
    if band is not None:
        k = (ff >= band[0]) & (ff < band[1])
        A, B = A[k], B[k]
    den = np.sqrt(np.sum(np.abs(A) ** 2) * np.sum(np.abs(B) ** 2))
    return 1.0 - np.real(np.sum(np.conj(B) * A)) / den if den > 0 else np.nan


def likelihood(td, data, win):
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, TDSettings, TDSignal
    from lisatools.sensitivity import XYZ2SensitivityMatrix

    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=1e-4, max_freq=2.5e-2,
                        force_backend="cpu")
    ac = AnalysisContainer(TDSignal(data, td_set).transform(fd_set, window=win),
                           XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    tmpl = TDSignal(td, td_set).transform(fd_set, window=win)
    dlogl = float(np.real(ac.template_likelihood(tmpl)))
    opt, _ = ac.template_snr(tmpl)
    return dlogl, float(np.real(opt)) / float(np.sqrt(np.real(ac.inner_product())))


def main():
    watchdog()
    params, data, data_t0, orb = load(SRC)
    win = tukey(N_WIN, 0.1)
    ff = np.fft.rfftfreq(N_WIN, d=DT)
    print(f"SRC={SRC} N_WIN={N_WIN} DT={DT} window starts REF+{data_t0 - REF:.1f}s  N_FINE={N_FINE}", flush=True)
    wg, offset_int, gen = legacy_wrapper(orb, data_t0)
    for thr in THRESHES:
        tof, nsub, n_in = tof_td(params, orb, data_t0, thr, gen)
        leg = legacy_td(params, wg, offset_int, thr)
        print(f"\n== mode threshold {thr:g}: TOF subs={nsub}, TOF inside={n_in}/{N_WIN}", flush=True)
        for tag, s in (("tof", tof), ("leg", leg)):
            dl, r = likelihood(s, data, win)
            print(f"  {tag}: dlogL={dl:+.5f}  opt/data SNR={r:.5f}", flush=True)
        hdr = "  pair        ch   full        " + "  ".join(f"[{lo*1e3:g},{hi*1e3:g}) mHz" for lo, hi in BANDS)
        print(hdr, flush=True)
        for tag, a, b in (("tof-data", tof, data), ("leg-data", leg, data), ("tof-leg", tof, leg)):
            for c, ch in enumerate("XYZ"):
                row = [mismatch(a[c], b[c], win, ff)] + [mismatch(a[c], b[c], win, ff, bd) for bd in BANDS]
                print(f"  {tag:10s}  {ch}  " + "  ".join(f"{v:10.3e}" for v in row), flush=True)


if __name__ == "__main__":
    main()
