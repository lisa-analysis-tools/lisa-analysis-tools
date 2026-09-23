"""Build ``gb_truth_3to21.npz`` -- the frozen recovery denominator.

The status page (``gf_monitor_gen.py``) quotes every completeness / purity /
per-source recovery number against ONE frozen set: the catalogue galactic
binaries detectable (optimal SNR > 7) over the analysed GB band under the
run's own fitted noise at a chosen iteration. The original build script lived
in a session scratchpad and was lost -- which is why this one is IN the repo.
Committed 2026-08-19; validated by reproducing the original count (812
detectable over the then-hardcoded 3-21.94 mHz).

THE BAND IS SETTABLE, AND THE DEFAULT IS A JUDGEMENT CALL. It used to be
hardcoded at 3-21.94 mHz, which threw away most of the GB band: the run's
own ``sub_backend/gb/band_edges`` starts at 0.5556 mHz. ``--flo``/``--fhi``
now set the band and the value used is stamped into BOTH output npz files,
so the monitor quotes it rather than assuming.

The DEFAULT floor is **0.8 mHz** (2026-09-18), above the sampler's own
0.5556 mHz. That is deliberate: see the resolvability warning below. Pass
``--flo 5.5555555556e-4`` for the full analysed band.

A WARNING ABOUT THE LOW-FREQUENCY END -- the reason for the 0.8 mHz floor.
``det`` is and remains an SNR statement: optimal SNR > 7 against a
sensitivity that already includes the fitted galactic foreground, so the
confusion background is accounted for in the DENOMINATOR of the SNR. It is
NOT a resolvability statement. Below ~0.8 mHz the catalogue puts many sources
in every frequency bin, and a source can clear SNR 7 while being hopelessly
blended with its neighbours; charging the run for those in the completeness
denominator measures the catalogue, not the sampler. Even between 0.8 and
3 mHz, read this set as "carries enough signal power to matter" rather than
"is individually recoverable".

DETECTABILITY IS PER-TOBS. The observation time sets the FD bin width, the
waveform duration and hence the optimal SNR itself, so a truth set is only
valid for the run it was built against: at 1 year the same catalogue source
integrates 4x longer and the detectable count grows. ``Tobs`` is therefore
read from the STORE's own domain settings (``global_fit/domain_settings/args``
-> ``attrs["0"] * attrs["1"] * attrs["2"]``, the same expression the monitor
uses), with the 3-month value as the fallback and ``--tobs`` as an explicit
override, and the value used is stamped into both output npz files. The
monitor refuses a truth set whose stamped ``tobs`` does not match the run.

Nothing here is re-derived: the noise, waveform and SNR route are copied from
the monitor's own recovery section (GBGPU ``run_wave`` on CPU, lisatools
``A2TDISens``/``E2TDISens`` fed the run's sampled instrument + foreground) --
the same path the run's likelihood uses.

Shape of the output (the monitor's ``TRU`` contract):
    f0    (N,)   Hz
    amp   (N,)   GW amplitude
    snr   (N,)   optimal SNR under the frozen noise
    det   (N,)   bool, snr > 7
    phys  (N,9)  GBGPU physical basis [amp, f0, fdot, fddot, phi0, iota,
                 psi, lam(icrs), beta(icrs)]
plus provenance attrs (store path, iteration, band, psd/galfor params).

The amplitude PREFILTER: running waveforms for every one of the 15.5M
catalogue rows is pointless when SNR/amp at fixed f is bounded. A unit-SNR
kappa curve (best-case orientation, coarse f grid, buffered 2x) cuts the
waveform set to the plausibly-detectable tail; the same curve is written to
``kappa_grid.npz`` for the monitor's population panels.

Usage::

    OMP_NUM_THREADS=1 python scripts/diagnostics/build_truth.py \
        STORE.h5 [--iteration 78] [--out gb_truth_3to21.npz] [--tobs SEC] \
        [--flo 0.8e-3] [--fhi 2.1944444e-2]

Runs on CPU. THE WAVEFORM COUNT IS SET BY THE PREFILTER, NOT THE BAND --
measured on the 3-month store at iteration 208:

    floor       in-band rows   waveforms run   detectable
    0.5556 mHz     3 959 524           9 086         1048
    0.8    mHz     1 008 443           9 011         1048
    1      mHz       396 653           8 885         1042

Widening the band multiplies the catalogue rows by 10 and the waveforms by
2%, because `amp * kappa(f)` bounds the SNR and everything below the cut is
discarded before any waveform runs. The prefilter is a vectorised interp
over the catalogue array, so the extra rows cost seconds. Budget tens of
minutes for the ~9k waveforms. Keep the thread pins: laptop policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import h5py

# The band the truth set is built over. The ceiling is the runs' own
# ``sub_backend/gb/band_edges[-1]`` (21.9444 mHz); the FLOOR is a DELIBERATE
# 0.8 mHz, set above the sampler's band_edges[0] of 0.5556 mHz.
#
# Why not the full analysed band: ``det`` is an SNR statement, not a
# resolvability one. Below ~0.8 mHz the catalogue puts many sources in every
# frequency bin, so a binary clears SNR 7 while being inseparable from its
# neighbours. Counting those in the DENOMINATOR charges the run for sources
# nothing could individually recover, which depresses completeness by an
# amount that says nothing about the sampler. 0.8 mHz is the floor; use
# ``--flo 5.5555555556e-4`` to reinstate the full analysed band.
FLO, FHI = 0.8e-3, 2.1944444444e-2
SNR_DET = 7.0
TOBS_3MO = 7776000.0   # the original 3-month set; also the fallback Tobs
DT = 2.5
NW_3MO = 128      # FD points per waveform at 3 months (the monitor's own NW_)

# The GB catalogue this truth set is built against. The default is the
# mojito brick cache as it lands on a laptop; a cluster, a container or a
# second checkout will all put it somewhere else, so the path is
# overridable by --catalogue or MOJITO_CAT and the default is a SEARCH
# rather than a single guess.
MOJITO_CAT_DEFAULT = os.path.expanduser(
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/catalogues/"
    "wdwd_cat_mojito_lite_processed.hdf5")
MOJITO_CAT_NAME = "wdwd_cat_mojito_lite_processed.hdf5"


def resolve_catalogue(explicit=None):
    """Locate the GB catalogue, most specific source first.

    ``--catalogue`` beats ``MOJITO_CAT`` beats ``MOJITO_CACHE_DIR`` beats
    the packaged default. Raises with every path it tried rather than
    failing later inside h5py with a bare "unable to open file", which is
    what made this hard to diagnose the first time.
    """
    tried = []
    for cand in (
        explicit,
        os.environ.get("MOJITO_CAT"),
        (os.path.join(os.environ["MOJITO_CACHE_DIR"], "brickmarket",
                      "mojito_light_v1_0_0", "catalogues", MOJITO_CAT_NAME)
         if os.environ.get("MOJITO_CACHE_DIR") else None),
        MOJITO_CAT_DEFAULT,
    ):
        if not cand:
            continue
        cand = os.path.expanduser(cand)
        # a directory is accepted and the catalogue looked up inside it
        if os.path.isdir(cand):
            cand = os.path.join(cand, MOJITO_CAT_NAME)
        tried.append(cand)
        if os.path.isfile(cand):
            return cand
    raise SystemExit(
        "could not find the GB catalogue "
        f"({MOJITO_CAT_NAME}).\nTried, in order:\n  "
        + "\n  ".join(tried)
        + "\n\nSet one of:\n"
        "  --catalogue /path/to/wdwd_cat_mojito_lite_processed.hdf5\n"
        "  MOJITO_CAT=/path/to/wdwd_cat_mojito_lite_processed.hdf5\n"
        "  MOJITO_CACHE_DIR=/path/to/.mojito_cache\n"
        "(--catalogue and MOJITO_CAT also accept the containing directory.)"
    )


def store_tobs(store, fallback=TOBS_3MO):
    """Observation time of the run, from the store's own domain settings.

    ``global_fit/domain_settings/args`` carries the WDM/FD grid arguments as
    attrs "0"/"1"/"2" = (Nt, Nf, dt) -- their product is Tobs. This is exactly
    how ``gf_monitor_gen.py`` derives ``SCI_TOBS``, so a truth set built here
    and a page rendered there agree by construction. Falls back to the 3-month
    value when the store predates the group.
    """
    try:
        with h5py.File(store, "r") as f:
            a = dict(f["global_fit/domain_settings/args"].attrs)
        return float(a["0"]) * float(a["1"]) * float(a["2"])
    except Exception as e:
        print(f"WARNING: no domain settings in {store} ({type(e).__name__}); "
              f"falling back to Tobs = {fallback}")
        return float(fallback)


def nw_for(tobs):
    """FD points per waveform at this Tobs.

    ``N`` has to span the Doppler-modulated bandwidth of the source, which is
    a fixed number of HERTZ (a few tens of 1/yr sidebands); the FD bin is
    1/Tobs, so the bandwidth in BINS grows in proportion to Tobs. Hold the
    3-month value at exactly 128 (reproducing the frozen 812-source set) and
    scale up in powers of two from there -- 512 at one year. Over-sizing N is
    safe (the extra bins are ~zero and the slow part is merely sampled more
    finely); under-sizing truncates the waveform and silently loses SNR.
    """
    n = NW_3MO * max(float(tobs) / TOBS_3MO, 1.0)
    return int(min(2 ** int(np.ceil(np.log2(n))), 2048))


def galfor_log_sampling_of(store) -> bool:
    """``noise_model_identity["galfor_log_sampling"]`` for this STORE.

    Read it off the store, never off the environment: the env var describes
    the current process, while the stored numbers were written in whatever
    basis that run used. A store predating the key is linear.
    """
    from lisatools.globalfit.stock.erebor.noise import read_noise_model_identity

    return bool(read_noise_model_identity(store).get("galfor_log_sampling", False))


def fitted_noise(store, it):
    """Cold-chain median psd + galfor params at ``it``, in PHYSICAL units.

    The galfor row is de-log10'd when the store says so -- see
    :func:`~lisatools.globalfit.stock.erebor.noise.galfor_params_to_physical`.
    Handing the raw log10 row to the foreground model gives a negative
    ``amp``/``f_1`` and NaN sensitivity, which showed up here not as a crash
    but as an ALL-ZERO ``snr`` and an all-False ``det`` -- a silently empty
    truth set that still passes a "psd/galfor are nonzero" sanity check.

    The median is taken in the SAMPLING basis and converted afterwards,
    matching what the sampler and the monitor page both do; for the log
    columns that is a geometric median-of-logs, which is the meaningful
    centre for a quantity spanning decades.
    """
    from lisatools.globalfit.stock.erebor.noise import galfor_params_to_physical

    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        psd = np.median(np.asarray(g["chain"]["psd"][it, 0, 0, :, 0, :]),
                        axis=0)
        gal = np.median(np.asarray(g["chain"]["galfor"][it, 0, 0, :, 0, :]),
                        axis=0)
    return psd, galfor_params_to_physical(gal, galfor_log_sampling_of(store))


def sens_grids(psd_p, gal_p, df):
    from lisatools import detector as lisa_models
    from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as HTGF)
    lm = lisa_models.LISAModel(psd_p[0] ** 2, psd_p[1] ** 2,
                               lisa_models.DefaultOrbits(), "sampled")
    nk = dict(model=lm, stochastic_params=tuple(gal_p),
              stochastic_function=HTGF)
    ng = int(2.35e-2 / df) + 2
    fg = np.maximum(np.arange(ng) * df, df)
    sa = np.asarray(get_sensitivity(fg, sens_fn=A2TDISens, **nk), float)
    se = np.asarray(get_sensitivity(fg, sens_fn=E2TDISens, **nk), float)
    return sa, se


def find_l1_brick(path=None):
    """Any mojito L1 file -- they all carry the same orbits/ltt tables."""
    import glob
    if path:
        return path
    for root in (os.environ.get("MOJITO_DATA_PATH"),
                 os.path.expanduser("~/.mojito_cache")):
        if not root or not os.path.isdir(root):
            continue
        hits = sorted(glob.glob(os.path.join(root, "**", "*_L1_*.h5"),
                                recursive=True))
        if hits:
            return hits[0]
    return None


def l1_orbits(stride=200, path=None, frame="icrs"):
    """The INJECTED (mojito L1) orbits, with the light-travel-time table strided.

    THIS IS NOT A REFINEMENT, IT IS THE DIFFERENCE BETWEEN RIGHT AND WRONG.
    Measured at 7.44-7.60 mHz on the 6mo store (2026-09-23), swapping the
    analytic ``DefaultOrbits`` ephemeris for these:

        template-vs-injection overlap   0.052 -> 0.580,  0.368 -> 0.982,
                                        0.262 -> 0.962
        optimal SNR                     7.96 -> 8.69,   32.81 -> 42.67

    The analytic ephemeris differs from the mojito orbits by the full
    annual-Doppler phase at the run epoch, so anything above a few mHz --
    SNRs, overlaps, the detectable denominator -- is simply a different
    number. Phase-MAXIMISED overlaps do not rescue it.

    Why striding is exact: the C++ side interpolates ltt linearly in
    ``(t - ltt_t0) / ltt_dt`` (``Detector.cu`` ``get_window`` ->
    ``interpolate``) and light travel times vary on ORBITAL timescales -- an
    8.3 s travel time changing by ~1% over a year. Sampling them every
    ``stride * 2.5`` s instead of every 2.5 s leaves a linear-interpolation
    error of ~1e-15 s, and turns a 1.2 GB table (plus its C++ copy) into
    1.5 MB. That is what lets this run on a laptop.

    Returns ``(orbits, path)``, or ``(None, reason)`` when no brick is found.
    """
    from lisatools import detector as lisa_models
    fp = find_l1_brick(path)
    if fp is None:
        return None, ("no mojito L1 brick under MOJITO_DATA_PATH or "
                      "~/.mojito_cache")

    class _StridedL1(lisa_models.L1Orbits):
        def _setup(self):
            with self.open() as f:
                self.ltt = np.ascontiguousarray(f.ltts.ltts[::stride])
                try:
                    self.ltt_t = f.ltts.time_sampling.t(slice(0, None, stride))
                except TypeError:
                    self.ltt_t = f.ltts.time_sampling.t()[::stride]
                self.x_base = f.orbits.positions[:]
                self.v_base = f.orbits.velocities[:]
                self.sc_t_base = f.orbits.time_sampling.t()
                self.size_base = self.sc_t_base.shape[0]
                self.dt_base = float(f.orbits.time_sampling.dt)
                self.ltt_dt = float(f.ltts.time_sampling.dt) * stride
                self.sc_dt = f.orbits.time_sampling.dt
                self.ltt_t0 = float(self.ltt_t[0])
                self.sc_t0 = float(self.sc_t_base[0])

    orb = _StridedL1(fp, force_backend="cpu", frame=frame,
                     linear_interp_dt=500.0)
    orb._ensure_configured()
    return orb, fp


def catalogue_phys(t_ref, flo=FLO, fhi=FHI, catalogue=None):
    """(N,9) GBGPU physical rows for every catalogue GB in [flo, fhi].

    Column conventions copied from the run's own catalogue path
    (``gb_catalogue_to_sampling_basis`` handles the epoch shift and ICRS
    frame; ``both_transforms`` of the 8-col basis is what the sampler feeds
    GBGPU) -- here we only need the physical 9-vector, which
    ``gb_catalogue_to_sampling_basis`` exposes directly through the run's
    transform container."""
    from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis
    cat = resolve_catalogue(catalogue)
    print(f"catalogue: {cat}")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]
        f0 = np.asarray(b["GW22FrequencySSBFrame"][:], float)
        sel = (f0 >= flo) & (f0 <= fhi)
        entry = {k: np.asarray(b[k][sel]) for k in (
            "Amplitude", "GW22FrequencySSBFrame",
            "GW22FrequencyDerivativeSourceFrame", "InclinationAngle",
            "PolarisationAngle", "RightAscension", "Declination",
            "TrueAnomaly", "TimeReferenceSSBFrame", "ChirpMassSSBFrame",
            "LuminosityDistance", "Eccentricity")}
    # 8-col sampling basis [lnA, f0 mHz, fdot, phi0, cos_i, psi, lam, sinb]
    rows = np.atleast_2d(gb_catalogue_to_sampling_basis(entry))
    phys = np.column_stack([
        np.exp(rows[:, 0]), rows[:, 1] * 1e-3, rows[:, 2],
        np.zeros(len(rows)), rows[:, 3], np.arccos(rows[:, 4]),
        rows[:, 5], rows[:, 6], np.arcsin(np.clip(rows[:, 7], -1, 1)),
    ])
    return phys


def opt_snr(phys, sa, se, gbw, df, tobs, nw, batch=20000):
    """Optimal SNR of each row: 4 df sum(|A|^2/SA + |E|^2/SE), sqrt.

    ``tobs`` is the run's observation time -- it enters BOTH the waveform
    (``T=`` in ``run_wave``, which sets how long the source integrates) and
    the inner-product measure ``df = 1/tobs``; ``sa``/``se`` must already be
    sampled on that same ``df`` grid.
    """
    out = np.zeros(len(phys))
    # One batch holds ``batch x nw`` complex A and E. That was harmless at
    # 4.7k rows and nw=128; over the full band the kept set is tens of
    # thousands and a 1-year build carries nw=512, so the knob is exposed --
    # drop it on a memory-constrained host rather than editing this file.
    B = int(batch)
    for lo in range(0, len(phys), B):
        p = phys[lo:lo + B]
        gbw.run_wave(*[np.ascontiguousarray(p[:, k]) for k in range(9)],
                     N=nw, T=tobs, dt=DT, tdi2=True, tdi_channel_setup="AE")
        A = np.asarray(gbw.A); E = np.asarray(gbw.E)
        s = np.asarray(gbw.start_inds).astype(int)
        for i in range(len(p)):
            if s[i] < 0 or s[i] + nw > sa.size:
                continue
            _sa = sa[s[i]:s[i] + nw]; _se = se[s[i]:s[i] + nw]
            out[lo + i] = np.sqrt(
                4 * df * (np.abs(A[i]) ** 2 / _sa
                          + np.abs(E[i]) ** 2 / _se).real.sum())
        print(f"  snr: {min(lo + B, len(phys))}/{len(phys)}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("store")
    ap.add_argument("--iteration", type=int, default=78)
    ap.add_argument("--out", default="gb_truth_3to21.npz")
    ap.add_argument("--kappa-out", default="kappa_grid.npz")
    ap.add_argument("--tobs", type=float, default=None,
                    help="observation time [s]; default: read from the "
                         "store's global_fit/domain_settings/args")
    ap.add_argument("--nw", type=int, default=None,
                    help="FD points per waveform; default: 128 scaled by "
                         "Tobs/3 months, rounded up to a power of two")
    ap.add_argument("--flo", type=float, default=FLO,
                    help=f"band floor [Hz]; default {FLO:.10g} = the GB "
                         "band_edges[0] of the erebor stores")
    ap.add_argument("--fhi", type=float, default=FHI,
                    help=f"band ceiling [Hz]; default {FHI:.10g} = the GB "
                         "band_edges[-1] of the erebor stores")
    ap.add_argument("--catalogue", default=None,
                    help="GB catalogue hdf5, or the directory holding it. "
                         "Overrides MOJITO_CAT and MOJITO_CACHE_DIR; "
                         f"default {MOJITO_CAT_DEFAULT}")
    ap.add_argument("--analytic-orbits", action="store_true",
                    help="build on DefaultOrbits instead of the injected "
                         "mojito L1 orbits. Reproduces truth sets made before "
                         "2026-09-23; wrong above a few mHz (see l1_orbits).")
    ap.add_argument("--batch", type=int, default=20000,
                    help="waveform rows per run_wave call (default 20000); "
                         "lower it if the build runs out of memory")
    a = ap.parse_args(argv)

    flo, fhi = float(a.flo), float(a.fhi)
    if not (0 < flo < fhi):
        ap.error(f"need 0 < --flo < --fhi (got {flo}, {fhi})")
    tobs = float(a.tobs) if a.tobs else store_tobs(a.store)
    nw = int(a.nw) if a.nw else nw_for(tobs)
    df = 1.0 / tobs
    print(f"Tobs = {tobs:.1f} s ({tobs / 86400.0:.1f} d), df = {df:.4g} Hz, "
          f"N per waveform = {nw}")
    print(f"band = [{flo:.10g}, {fhi:.10g}] Hz "
          f"({flo * 1e3:.4g}-{fhi * 1e3:.4g} mHz)")
    psd_p, gal_p = fitted_noise(a.store, a.iteration)
    print(f"noise @ it {a.iteration}: psd={psd_p} galfor={gal_p}")
    sa, se = sens_grids(psd_p, gal_p, df)

    from gbgpu.gbgpu import GBGPU
    from lisatools import detector as lisa_models
    from lisatools.globalfit.stock.erebor.variants.gb_no_fg import (
        GB_MOJITO_T_REF)
    # THE INJECTED ORBITS BY DEFAULT (2026-09-23). This file used to build
    # every truth set on the analytic ephemeris, which put the monitor in a
    # mixed state: the page's own overlap/SNR block uses the mojito orbits
    # when it can reach them, while the detectable DENOMINATOR, the SNR axis
    # of the recovery curve and column 1 of the #detect table came from here.
    # The two disagree by tens of percent above a few mHz (see l1_orbits).
    orb, orb_src = (None, "--analytic-orbits requested")
    if not a.analytic_orbits:
        orb, orb_src = l1_orbits()
    orbits_tag = "mojito_l1"
    if orb is None:
        orbits_tag = "analytic"
        print(f"WARNING: building on the ANALYTIC ephemeris ({orb_src}). "
              "SNRs and the detectable set above a few mHz will not match "
              "the monitor's own overlap block.")
        orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
    else:
        print(f"orbits: mojito L1 {orb_src}")
    gbw = GBGPU(force_backend="cpu", orbits=orb, t0=float(GB_MOJITO_T_REF))

    phys = catalogue_phys(GB_MOJITO_T_REF, flo, fhi, a.catalogue)
    print(f"catalogue rows in [{flo:.10g}, {fhi:.10g}] Hz: {len(phys)}")

    # ---- kappa prefilter -------------------------------------------------
    # Best-case-orientation unit-SNR curve on a coarse grid: face-on,
    # psi=phi0=0, one sky draw per node maximised over 8 sky positions.
    # Node count scales with the LOG SPAN so the grid keeps the density the
    # 48-node 3-21.94 mHz grid had (48 nodes / 0.864 decades). A wider band on
    # a fixed node count would interpolate the kappa curve across the galactic
    # knee, and the prefilter reads that curve as a ceiling -- a sagging
    # interpolant there silently drops real sources before any waveform runs.
    _NK_REF, _SPAN_REF = 48, np.log10(21.94e-3 / 3e-3)
    _nk = max(int(round(_NK_REF * np.log10(fhi / flo) / _SPAN_REF)), _NK_REF)
    fgrid = np.geomspace(flo, fhi, _nk)
    kap = np.zeros(fgrid.size)
    A0 = 1e-22
    for isky in range(8):
        rng = np.random.default_rng(isky)
        lam = rng.uniform(0, 2 * np.pi); beta = np.arcsin(rng.uniform(-1, 1))
        proto = np.column_stack([
            np.full(fgrid.size, A0), fgrid, np.zeros(fgrid.size),
            np.zeros(fgrid.size), np.zeros(fgrid.size),
            np.zeros(fgrid.size), np.zeros(fgrid.size),
            np.full(fgrid.size, lam), np.full(fgrid.size, beta)])
        kap = np.maximum(kap, opt_snr(proto, sa, se, gbw, df, tobs, nw) / A0)
    keep = phys[:, 0] * np.interp(phys[:, 1], fgrid, kap) > 0.5 * SNR_DET
    print(f"kappa prefilter keeps {keep.sum()} / {len(phys)}")
    # keys are the monitor's contract: kappa_grid.npz carries fgrid/fit. The
    # curve is a unit-SNR ceiling under THIS Tobs over THIS band, so stamp
    # both -- the monitor draws the curve only over the band it was built on.
    np.savez(a.kappa_out, fgrid=fgrid, fit=kap, tobs=np.array(tobs),
             band=np.array([flo, fhi]))

    snr = np.zeros(len(phys))
    snr[keep] = opt_snr(phys[keep], sa, se, gbw, df, tobs, nw, a.batch)
    det = snr > SNR_DET
    print(f"DETECTABLE (SNR > {SNR_DET}): {det.sum()}")

    np.savez_compressed(
        a.out, f0=phys[:, 1], amp=phys[:, 0], snr=snr, det=det, phys=phys,
        store=np.array(a.store), iteration=np.array(a.iteration),
        psd_params=psd_p, galfor_params=gal_p,
        band=np.array([flo, fhi]), tobs=np.array(tobs), nw=np.array(nw),
        # STAMPED so the monitor can refuse to mix ephemerides silently.
        orbits=np.array(orbits_tag))
    print(f"wrote {a.out}  (orbits: {orbits_tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
