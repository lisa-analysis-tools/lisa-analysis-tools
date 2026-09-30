"""EMRITDIonFly vs the mojito L1 injection, per channel, on a SHORT window.

Gate for Task A0 of the EMRI direct-to-WDM plan: the TDI-on-the-fly X, Y, Z must
reproduce the injection before anything is built on top of it.

* ``frame="icrs_special"`` + a fine trajectory (``n_fine``) must equal the
  PRODUCTION response (``get_emri_response_wrapper``, the 6mo base signal) to
  ``1 - Re(O) < 1e-10`` per channel, and match the data at mode threshold
  1e-7 to ``1 - Re(O) < 1e-4`` with NO time or phase maximisation, with the
  SciRD log-likelihood within 0.05 of the injection's (``logL(h=d) = 0``).
* Paired negative control: the historical all-ecliptic frame must NOT pass.
* The call must not leak ``new_t``/``upsample`` into the shared FEW generator
  (FEW merges call-time ``inspiral_kwargs`` permanently, waveform/base.py:236).

Needs the local mojito cache and ``mojito``/``few``; skips otherwise.
"""

import gc
import os
import unittest

import numpy as np

from lisatools.utils.constants import YRSID_SI
PATH = os.environ.get("MOJITO_LIGHT_PATH", "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/")  # dir with catalogues/ and data/EMRI/L1/
REF = 97729089.327664          # catalogue reference epoch (MOJITO_REFERENCE_TIME)
SRC = 1                        # catalogue ROW 1 (the L1 file index; ID field is 1-based)
DT = 20.0
N_WIN = 4096                   # 0.95 d: small window
N_FINE = 1024                  # ~80 s trajectory spacing, as the validated special check
START_OFFSET_S = 5e4       # clear of the legacy wrapper's t_buffer=3e4 s edge zeroing


def _load():
    import h5py
    from mojito import MojitoL1File

    from lisatools.detector import L1Orbits
    from lisatools.globalfit.preprocessing import find_file
    from lisatools.sources.utils import icrs_to_ecliptic

    cat = os.path.join(PATH, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5")
    with h5py.File(cat, "r") as f:
        b = f["Binaries"]
        g = lambda k: float(b[k][SRC])
        lam, beta = icrs_to_ecliptic(g("RightAscension") % (2 * np.pi), g("Declination"))
        params = [
            g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter"),
            g("SemiLatusRectum"), g("Eccentricity"), 1.0, g("LuminosityDistance") / 1e3,
            float(np.pi / 2 - beta), float(lam) % (2 * np.pi),
            g("PolarAnglePrimarySpin"), g("AzimuthalAnglePrimarySpin"),
            g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase"),
        ]
    fp = find_file(os.path.join(PATH, "data", "EMRI", "L1"), "EMRI", SRC)
    ts = MojitoL1File(fp).tdis.time_sampling
    deci = int(round(DT / ts.dt))
    i0 = int(round(START_OFFSET_S / ts.dt))
    data_t0 = float(ts.t0) + i0 * float(ts.dt)
    with h5py.File(fp, "r") as f:
        lf = float(f.attrs["laser_frequency"])
        data = np.stack([
            np.asarray(f["tdis"][c][i0: i0 + N_WIN * deci])[::deci][:N_WIN] / lf
            for c in ("X2", "Y2", "Z2")
        ])

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


def _td_of(out, t_grid):
    """Channel TD on ``t_grid``; zero outside the TOF spline support."""
    tg = np.asarray(out.x)
    inside = (t_grid > float(np.max(tg[:, 0]))) & (t_grid < float(np.min(tg[:, -1])))
    td = np.zeros((3, t_grid.size))
    td[:, inside] = np.real(np.sum(np.asarray(out.eval_tdi(t_grid[inside])), axis=0))
    return td, int(inside.sum())


def _metrics(td, data):
    """Per-channel 1-Re(O) (flat, Tukey 0.1, no maximisation) and SciRD dlogL."""
    from scipy.signal.windows import tukey

    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, TDSettings, TDSignal
    from lisatools.sensitivity import XYZ2SensitivityMatrix

    win = tukey(N_WIN, 0.1)
    mm = []
    for c in range(3):
        A = np.fft.rfft(td[c] * win)
        B = np.fft.rfft(data[c] * win)
        mm.append(1.0 - np.real(np.sum(np.conj(B) * A)) / np.sqrt(np.sum(np.abs(A) ** 2) * np.sum(np.abs(B) ** 2)))
    td_set = TDSettings(N_WIN, DT, t0=0.0, force_backend="cpu")
    fd_set = FDSettings(N=N_WIN // 2 + 1, df=1.0 / (N_WIN * DT), min_freq=1e-4, max_freq=1e-2,
                        force_backend="cpu")
    ac = AnalysisContainer(TDSignal(data, td_set).transform(fd_set, window=win),
                           XYZ2SensitivityMatrix(fd_set, model="scirdv1"))
    tmpl = TDSignal(td, td_set).transform(fd_set, window=win)
    dlogl = float(np.real(ac.template_likelihood(tmpl)))      # logL(h=d) = 0
    opt, det = ac.template_snr(tmpl)
    return np.array(mm), dlogl, float(np.real(opt)), float(np.real(det)), float(np.sqrt(np.real(ac.inner_product())))


class EMRIModeConventionTest(unittest.TestCase):
    """The per-mode (amp, phase) feed must reproduce FEW's own mode sum exactly.

    FEW (summation/directmodesum.py):
        w1 = sum_{m>=0} Y_lm A e^{-i Phi},  w2 = sum_{m>0} (-1)^l Y_{l,-m} conj(A) e^{+i Phi}.
    The -m partner may NOT be built as the +m mode with a negated phase: the real
    prefactors of Y_{l,m}(theta) and Y_{l,-m}(theta) carry independent signs, so that
    shortcut is off by pi for some (mostly higher-l) modes. Low thresholds (1e-7)
    keep those modes; the historical check only ran at 1e-3 (29 modes).
    """

    @classmethod
    def setUpClass(cls):
        try:
            from few.waveform import FastKerrEccentricEquatorialFlux
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"needs few: {exc}")
        gen = FastKerrEccentricEquatorialFlux(
            force_backend="cpu", inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4)},
            sum_kwargs={"pad_output": True})
        cls.K = gen(1e6, 20.0, 0.9, 10.0, 0.4, 1.0, 1.1, 0.4, dist=1.0, Phi_phi0=0.3, Phi_theta0=0.0,
                    Phi_r0=1.2, T=86400.0 / 3.15581497635456e7, dt=20.0, return_sparse_holder=True,
                    include_minus_mkn=True, mode_selection_threshold=1e-7)

    @staticmethod
    def _few_sum(K):
        nm = K.ms.shape[0]
        phi = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
               + K.ns[None, :] * K.phases[:, 2][:, None])
        w1 = (K.ylms[None, :nm] * K.teuk_modes * np.exp(-1j * phi)).sum(1)
        keep = K.ms > 0
        sgn = ((-1.0) ** np.asarray(K.ls))[None, keep]
        w2 = (sgn * K.ylms[None, nm:][:, keep] * np.conj(K.teuk_modes[:, keep])
              * np.exp(1j * phi[:, keep])).sum(1)
        return w1 + w2

    def test_mode_feed_reproduces_few_sum_at_low_threshold(self):
        from lisatools.sources.emri import EMRITDIonFly

        K = self.K
        self.assertGreater(K.ms.shape[0], 150)            # really exercises the high-l modes
        amp, phase = EMRITDIonFly.mode_amp_phase(K, include_minus_mkn=True, amp_factor=1.0)
        recon = (amp * np.exp(-1j * phase)).sum(0)
        ref = self._few_sum(K)
        rel = np.linalg.norm(recon - ref) / np.linalg.norm(ref)
        self.assertLess(rel, 1e-10, f"rel L2 err {rel:.3e} over {K.ms.shape[0]} modes")

    def test_mode_feed_plus_m_only(self):
        from lisatools.sources.emri import EMRITDIonFly

        K = self.K
        amp, phase = EMRITDIonFly.mode_amp_phase(K, include_minus_mkn=False, amp_factor=1.0)
        nm = K.ms.shape[0]
        phi = (K.ms[None, :] * K.phases[:, 0][:, None] + K.ks[None, :] * K.phases[:, 1][:, None]
               + K.ns[None, :] * K.phases[:, 2][:, None])
        ref = (K.ylms[None, :nm] * K.teuk_modes * np.exp(-1j * phi)).sum(1)
        recon = (amp * np.exp(-1j * phase)).sum(0)
        self.assertLess(np.linalg.norm(recon - ref) / np.linalg.norm(ref), 1e-10)


class EMRITofVsInjectionTest(unittest.TestCase):
    """Short window of CD1L EMRI 1 (row 1), START_OFFSET_S into the data so the
    legacy wrapper's t_buffer=3e4 s edge-garbage zeroing is outside it."""

    @classmethod
    def setUpClass(cls):
        try:
            import mojito  # noqa: F401
            from few.waveform import FastKerrEccentricEquatorialFlux  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"needs mojito + few: {exc}")
        if not os.path.isdir(PATH):
            raise unittest.SkipTest("local mojito cache missing")
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.sources.emri import EMRITDIonFly

        cls.EMRITDIonFly = EMRITDIonFly
        cls.params, cls.data, cls.data_t0, cls.orb = _load()
        cls.tdi = TDIConfig("2nd generation", force_backend="cpu")
        cls.t_grid = cls.data_t0 + np.arange(N_WIN) * DT
        cls.T_needed = cls.data_t0 - REF + N_WIN * DT + 2000.0
        cls.window = (cls.data_t0, cls.data_t0 + N_WIN * DT)

    _LEG = None

    @classmethod
    def _legacy_wrapper(cls):
        # The production wrapper, built ONCE; its FEW generator is reused for the TOF.
        # Each FEW construction reads the 5.1 GB amplitude file whole (~6 GB transient
        # footprint); one generator per test grew the run to an 18.5 GB footprint and
        # macOS killed it (exit 137). The mode threshold is passed per call.
        if cls._LEG is None:
            from lisatools.sources.emri.response import get_emri_response_wrapper

            off = cls.data_t0 - REF
            k = int(round(off / DT))
            wg = get_emri_response_wrapper(
                Tobs=(N_WIN + k) * DT + 4e4, dt=DT, t_start=REF, t0_shift_to_data=off - k * DT,
                tdi_config=cls.tdi, tdi_chan="XYZ", force_backend="cpu", orbits=cls.orb)
            cls._LEG = (wg, k)
        return cls._LEG

    @classmethod
    def _gen(cls, thr=None):
        return cls._legacy_wrapper()[0].waveform_gen.waveform_generator

    @classmethod
    def tearDownClass(cls):
        cls._LEG = None
        gc.collect()

    def tearDown(self):
        gc.collect()

    def _run(self, thr, gen=None, **kw):
        gen = gen if gen is not None else self._gen(thr)
        fly = self.EMRITDIonFly(gen, self.orb, self.tdi, DT, self.T_needed, REF, **kw)
        out = fly(*self.params, mode_selection_threshold=thr)
        return _td_of(out, self.t_grid)

    def _legacy(self, thr):
        wg, k = self._legacy_wrapper()
        return np.atleast_2d(np.asarray(wg(*self.params, mode_selection_threshold=thr)))[:3, k:k + N_WIN]

    def test_fine_icrs_special_matches_production_response(self):
        td, n_in = self._run(1e-5, frame="icrs_special", n_fine=N_FINE, t_fine_window=self.window)
        self.assertEqual(n_in, N_WIN)
        leg = self._legacy(1e-5)
        mm, _, opt, _, _ = _metrics(td, leg)
        msg = f"TOF vs legacy mm XYZ={mm}, opt SNR={opt:.8f}"
        print("\n[tof vs production response]", msg)
        self.assertTrue(np.all(mm < 1e-10), msg)

    def test_fine_icrs_special_matches_injection(self):
        td, n_in = self._run(1e-7, frame="icrs_special", n_fine=N_FINE, t_fine_window=self.window)
        mm, dlogl, opt, det, dsnr = _metrics(td, self.data)
        msg = f"mm XYZ={mm}, dlogL={dlogl:.5f}, opt={opt:.4f} det={det:.4f} data={dsnr:.4f}"
        print("\n[tof vs injection @1e-7]", msg)
        self.assertTrue(np.all(mm < 1e-4), msg)
        self.assertLess(abs(dlogl), 0.05, msg)
        self.assertAlmostEqual(opt / dsnr, 1.0, delta=2e-3, msg=msg)

    def test_ecliptic_frame_does_not_match_injection(self):
        # paired negative control: same fine feed, but the historical all-ecliptic sky
        # handed to the ICRS orbits the injection needs -> the wrong sky frame
        td, _ = self._run(1e-7, frame="ecliptic", n_fine=N_FINE, t_fine_window=self.window)
        mm, dlogl, *_ = _metrics(td, self.data)
        print("\n[ecliptic frame]", f"mm XYZ={mm}, dlogL={dlogl:.4f}")
        self.assertTrue(np.all(mm > 1e-3), f"mm XYZ={mm}")

    def test_few_receives_tobs_in_years(self):
        # FEW's T is in YEARS (few/waveform/base.py:173); Tobs is in seconds. Passing
        # seconds integrated this source to plunge (REF + 1.42 yr) for a 1-day request.
        gen = self._gen(1e-5)
        seen = {}

        class Spy:
            def __init__(self, g):
                self._g = g

            def __getattr__(self, name):
                return getattr(self._g, name)

            def __call__(self, *a, **kw):
                seen["T"] = kw["T"]
                return self._g(*a, **kw)

        fly = self.EMRITDIonFly(Spy(gen), self.orb, self.tdi, DT, self.T_needed, REF,
                                frame="icrs_special", n_fine=N_FINE, t_fine_window=self.window)
        fly(*self.params, mode_selection_threshold=1e-5)
        self.assertAlmostEqual(seen["T"] * YRSID_SI, self.T_needed, delta=1.0)

    def test_sparse_feed_too_coarse_raises_clearly(self):
        # a 1-day span has only a few adaptive knots; the delay trim empties the grid
        fly = self.EMRITDIonFly(self._gen(1e-5), self.orb, self.tdi, DT, 86400.0, REF)
        with self.assertRaisesRegex(ValueError, "n_fine"):
            fly(*self.params, mode_selection_threshold=1e-5)

    def test_call_does_not_leak_inspiral_kwargs(self):
        gen = self._gen(1e-5)
        before = dict(gen.inspiral_kwargs)
        self._run(1e-5, gen=gen, frame="icrs_special", n_fine=64, t_fine_window=self.window)
        self.assertEqual(set(gen.inspiral_kwargs), set(before))
        for k, v in before.items():
            self.assertIs(gen.inspiral_kwargs[k], v)


class EMRITofPlungeEndTest(unittest.TestCase):
    """At a plunge the fine feed ends with the trajectory; the response's delay trim then
    dropped the last ~720 s (and the response's ~470 s tail after the stop). The feed must
    run past the end with zero amplitude, as the production waveform's zero padding does."""

    NF, DT, NT = 180, 20.0, 1024

    @classmethod
    def setUpClass(cls):
        import sys
        try:
            import mojito  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"needs mojito: {exc}")
        if not os.path.isdir(PATH):
            raise unittest.SkipTest("local mojito cache missing")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "emri"))
        import emri_tof_xyz_threeway as W

        W.N_WIN = cls.NF * cls.NT
        _, _, _, cls.orb = W.load(1)
        cls.data_t0 = REF + 5e4
        cls.span = cls.NF * cls.NT * cls.DT

    def test_plunge_end_matches_production(self):
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.sources.emri import EMRITDIonFly
        from lisatools.sources.emri.response import get_emri_response_wrapper

        NF, DT = self.NF, self.DT
        params = [1e6, 1e2, 0.9, 7.0, 0.4, 1.0, 1.0, 1.1, 2.3, 0.7, 4.0, 0.2, 0.0, 0.7]   # plunges ~34 d
        off = self.data_t0 - REF
        k = int(round(off / DT))
        tdi = TDIConfig("2nd generation", force_backend="cpu")
        wg = get_emri_response_wrapper(Tobs=(NF * self.NT + k) * DT + 4e4, dt=DT, t_start=REF,
                                       t0_shift_to_data=off - k * DT, tdi_config=tdi, tdi_chan="XYZ",
                                       force_backend="cpu", orbits=self.orb)
        modes = [(2, 2, 0, 0)]
        prod = np.atleast_2d(np.asarray(wg(*params, mode_selection=modes,
                                            mode_selection_threshold=1e-5)))[:3, k:k + NF * self.NT]
        fly = EMRITDIonFly(wg.waveform_gen.waveform_generator, self.orb, tdi, DT, off + self.span + 2000.0, REF,
                           frame="icrs_special", n_fine=int(self.span / 80),
                           t_fine_window=(self.data_t0, self.data_t0 + self.span))
        out = fly(*params, mode_selection=modes)
        t_end = REF + float(np.asarray(fly.last_holder.t_arr)[-1])
        tg = self.data_t0 + np.arange(NF * self.NT) * DT
        x = np.asarray(out.x)
        ins = (tg > x[:, 0].max()) & (tg < x[:, -1].min())
        tof = np.zeros_like(prod)
        tof[:, ins] = np.real(np.sum(np.asarray(out.eval_tdi(tg[ins])), axis=0))
        sl = (tg > t_end - 3000.0) & (tg < t_end + 600.0)
        rel = [np.linalg.norm(tof[c, sl] - prod[c, sl]) / np.linalg.norm(prod[c, sl]) for c in range(3)]
        amp = [np.linalg.norm(tof[c, sl]) / np.linalg.norm(prod[c, sl]) for c in range(3)]
        print(f"\n[plunge end] rel L2 XYZ {np.round(rel, 4)}, amplitude ratio {np.round(amp, 4)}, "
              f"TOF grid ends {x[:, -1].min() - t_end:+.0f} s rel. to the trajectory end")
        self.assertGreaterEqual(x[:, -1].min(), t_end + 300.0)          # grid runs past the stop
        self.assertTrue(np.all(np.array(rel) < 0.05), rel)


if __name__ == "__main__":
    unittest.main()
