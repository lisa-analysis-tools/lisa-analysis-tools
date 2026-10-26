"""EMRI direct-to-WDM building blocks (Tasks A5, A7, A8).

A5: harmonic tracks (A, Phi, f, fdot, fddot) at pixel centres from a FEW sparse
holder, exact on a synthetic holder/integrator, with FEW's conventions pinned:
retrograde sign(xI0) on Phi_phi, the backwards-integration knot offset, and the
exact -m partner (-1)^l Y_{l,-m} conj(A) e^{+i Phi}.
"""

import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np


class _FakeIntegrator:
    """Phi_phi = w t + wd t^2 / 2 + wdd t^3 / 6, Phi_theta = 0, Phi_r = 0.3 w t."""

    def __init__(self, w=1e-3, wd=2e-9, wdd=3e-15):
        self.w, self.wd, self.wdd = w, wd, wdd
        self.generating_trajectory = False
        self.massratio = 1.0

    def eval_integrator_spline(self, t):
        out = np.zeros((t.size, 6))
        out[:, 3] = self.w * t + 0.5 * self.wd * t ** 2 + self.wdd * t ** 3 / 6
        out[:, 5] = 0.3 * self.w * t
        return out

    def eval_integrator_derivative_spline(self, t, order=1):
        out = np.zeros((t.size, 6))
        if order == 1:
            out[:, 3] = self.w + self.wd * t + 0.5 * self.wdd * t ** 2
            out[:, 5] = 0.3 * self.w
        elif order == 2:
            out[:, 3] = self.wd + self.wdd * t
        elif order == 3:
            out[:, 3] = self.wdd
        return out


def _fake_holder(t_knots, minus_m=False):
    teuk = (np.linspace(1.0, 2.0, t_knots.size) * np.exp(1j * 0.1 * t_knots / t_knots[-1]))[:, None]
    ylms = np.array([0.5 + 0.2j, -0.3 + 0.1j]) if minus_m else np.array([0.5 + 0.2j])
    return SimpleNamespace(t_arr=t_knots, teuk_modes=teuk, ylms=ylms,
                           ls=np.array([3]), ms=np.array([2]), ks=np.array([0]), ns=np.array([1]),
                           phases=None, freqs=None, integrate_backwards=False)


class HarmonicTrackTest(unittest.TestCase):
    def setUp(self):
        from lisatools.sources.emri.wdm_direct import harmonic_tracks_from_holder

        self.fn = harmonic_tracks_from_holder
        self.t_knots = np.linspace(0.0, 1e6, 50)
        self.t_pix = np.arange(10, 250) * 3600.0

    def _phi(self, integ, t):
        return integ.eval_integrator_spline(t)[:, 3]

    def test_phase_and_derivatives_are_exact(self):
        integ = _FakeIntegrator()
        tr = self.fn(_fake_holder(self.t_knots), integ, self.t_pix, a=0.9, xI0=1.0)[0]
        t, w = self.t_pix, integ.w
        np.testing.assert_allclose(tr.phase, 2 * self._phi(integ, t) + 0.3 * w * t, rtol=1e-12)
        np.testing.assert_allclose(tr.f, (2 * (w + integ.wd * t + 0.5 * integ.wdd * t ** 2) + 0.3 * w) / (2 * np.pi), rtol=1e-12)
        np.testing.assert_allclose(tr.fdot, 2 * (integ.wd + integ.wdd * t) / (2 * np.pi), rtol=1e-12)
        np.testing.assert_allclose(tr.fddot, 2 * integ.wdd / (2 * np.pi), rtol=1e-12)

    def test_amplitude_is_spline_of_teuk_times_ylm(self):
        from scipy.interpolate import CubicSpline

        h = _fake_holder(self.t_knots)
        tr = self.fn(h, _FakeIntegrator(), self.t_pix, a=0.9, xI0=1.0)[0]
        re = CubicSpline(h.t_arr, h.teuk_modes[:, 0].real)(self.t_pix)
        im = CubicSpline(h.t_arr, h.teuk_modes[:, 0].imag)(self.t_pix)
        np.testing.assert_allclose(tr.amp, (re + 1j * im) * h.ylms[0], rtol=1e-12)

    def test_retrograde_user_form_is_canonicalised_like_few(self):
        # FEW first maps xI0 < 0 -> (a, xI0) = (-a, +1) (few/waveform/base.py:208-212), so its
        # "if a > 0: Phi_phi *= sign(xI0)" never flips for this model: retrograde lives in the
        # trajectory's a < 0. The user form (a=0.9, xI0=-1) must NOT flip the holder phase.
        integ = _FakeIntegrator()
        retro = self.fn(_fake_holder(self.t_knots), integ, self.t_pix, a=0.9, xI0=-1.0)[0]
        canon = self.fn(_fake_holder(self.t_knots), integ, self.t_pix, a=-0.9, xI0=1.0)[0]
        pro = self.fn(_fake_holder(self.t_knots), integ, self.t_pix, a=0.9, xI0=1.0)[0]
        np.testing.assert_allclose(retro.phase, canon.phase, rtol=1e-14)
        np.testing.assert_allclose(retro.phase, pro.phase, rtol=1e-14)
        np.testing.assert_allclose(retro.f, pro.f, rtol=1e-14)

    def test_backwards_offset_applied(self):
        integ = _FakeIntegrator()
        h = _fake_holder(self.t_knots)
        h.integrate_backwards = True
        tr = self.fn(h, integ, self.t_pix, a=0.9, xI0=1.0)[0]
        ph = integ.eval_integrator_spline(h.t_arr)
        offset = 2 * (ph[-1, 3] + ph[0, 3]) + 1 * (ph[-1, 5] + ph[0, 5])
        t = self.t_pix
        np.testing.assert_allclose(tr.phase, 2 * self._phi(integ, t) + 0.3 * integ.w * t + offset, rtol=1e-12)

    def test_minus_m_partner_is_few_exact(self):
        integ = _FakeIntegrator()
        h = _fake_holder(self.t_knots, minus_m=True)
        plus, minus = self.fn(h, integ, self.t_pix, a=0.9, xI0=1.0)
        self.assertEqual(minus.lmkn, (3, -2, 0, -1))
        np.testing.assert_allclose(minus.phase, -plus.phase, rtol=1e-12)
        np.testing.assert_allclose(minus.f, -plus.f, rtol=1e-12)
        np.testing.assert_allclose(minus.fdot, -plus.fdot, rtol=1e-12)
        a_plus = plus.amp / h.ylms[0]                               # the splined teuk amplitude
        np.testing.assert_allclose(minus.amp, (-1.0) ** 3 * h.ylms[1] * np.conj(a_plus), rtol=1e-12)

    def test_track_sum_equals_few_mode_sum(self):
        # sum_tracks amp e^{-i phase} == FEW's w1 + w2 at the pixel times
        integ = _FakeIntegrator()
        h = _fake_holder(self.t_knots, minus_m=True)
        tracks = self.fn(h, integ, self.t_pix, a=0.9, xI0=1.0)
        h_sum = sum(tr.amp * np.exp(-1j * tr.phase) for tr in tracks)
        A = tracks[0].amp / h.ylms[0]
        Phi = tracks[0].phase
        ref = h.ylms[0] * A * np.exp(-1j * Phi) + (-1.0) ** 3 * h.ylms[1] * np.conj(A) * np.exp(1j * Phi)
        np.testing.assert_allclose(h_sum, ref, rtol=1e-12)


class HandoffTest(unittest.TestCase):
    """A7: handoff on the cubic (curvature) term OR the fdot range, whichever fires first.
    (No intra-chunk sweep guard: the SOBBH kappa limit is for HETERODYNED chunks; the plunge
    chunk is a raw TD->WDM transform, exact through the plunge — A8 gate, ~1e-14.)"""

    def setUp(self):
        from lisatools.sources.emri import wdm_direct as wd

        self.wd = wd
        self.layer_dt, self.layer_df = 3600.0, 1.0 / 7200.0

    def _track(self, fdot, fddot, P=100):
        t = np.arange(P) * self.layer_dt
        z = np.zeros(P)
        return self.wd.HarmonicTrack((2, 2, 0, 0), t, np.ones(P), z, z + 1e-3,
                                     np.broadcast_to(fdot, (P,)).astype(float),
                                     np.broadcast_to(fddot, (P,)).astype(float))

    def test_curvature_trips_before_range(self):
        tau = self.wd.WDM_HALF_SUPPORT_LAYERS * self.layer_dt
        fddot = 0.2 / ((np.pi / 3) * tau ** 3)
        tr = self._track(1e-8, fddot)
        self.assertEqual(self.wd.handoff_pixel(tr, self.layer_dt, self.layer_df, 8 * self.layer_df / self.layer_dt), 0)

    def test_range_trips_when_fdot_leaves_axis(self):
        tr = self._track(1e-6, 0.0)
        self.assertEqual(self.wd.handoff_pixel(tr, self.layer_dt, self.layer_df, 3.086e-7), 0)

    def test_handoff_is_first_trip_along_the_track(self):
        fdot = np.linspace(0.0, 1e-6, 100)          # leaves a 3.086e-7 axis at pixel 31
        tr = self._track(fdot, 0.0)
        n = self.wd.handoff_pixel(tr, self.layer_dt, self.layer_df, 3.086e-7)
        self.assertEqual(n, int(np.argmax(fdot > 3.086e-7)))

    def test_no_trip_for_slow_source(self):
        tr = self._track(1e-11, 1e-19)
        self.assertEqual(self.wd.handoff_pixel(tr, self.layer_dt, self.layer_df, 3.086e-7), 100)


class HandoffBoundTest(unittest.TestCase):
    """The fast handoff path (EMRIDirectWDM._handoff_may_trip): a bound over the three
    fundamentals skips the per-harmonic tracks when no harmonic can hand off, and never hides
    a handoff the exact tracks find (the plunge-chunk start is the same with and without it)."""

    FD_AX = 3.086e-7

    def _direct(self, integ):
        from lisatools.domains import WDMSettings
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

        table = SimpleNamespace(fdot_vals=np.array([-self.FD_AX, self.FD_AX]), INTERP_METHOD="spline")
        gen = SimpleNamespace(inspiral_generator=SimpleNamespace(inspiral_generator=integ))
        return EMRIDirectWDM(gen, table, WDMSettings(32, 256, 112.5, force_backend="cpu"), orbits=None,
                             tdi_config=None, t_start=0.0, data_t0=0.0)

    def _case(self, wd, wdd):
        import os
        from unittest import mock

        d = self._direct(_FakeIntegrator(w=1e-3, wd=wd, wdd=wdd))
        H = _fake_holder(np.linspace(0.0, 1.2e6, 80))          # ends after the 256 h window
        args = (1e6, 10.0, 0.9, 10.0, 0.3, 1.0)
        n_in, t_rel = d._pixel_times(H)
        fast = d._maybe_tracks(H, args, t_rel)
        start_fast = d._chunk_start(H, args, tracks=fast, t_rel=t_rel)
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_HANDOFF_BOUND": "0"}):
            exact = d._maybe_tracks(H, args, t_rel)
        start_exact = d._chunk_start(H, args, tracks=exact, t_rel=t_rel)
        trips = [self.__class__._k(d, tr) < t_rel.size for tr in exact]
        return fast, start_fast, start_exact, any(trips)

    @staticmethod
    def _k(d, tr):
        from lisatools.sources.emri.wdm_direct import handoff_pixel

        return handoff_pixel(tr, d.wdm.layer_dt, d.wdm.layer_df, d.fdot_axis_max)

    def test_slow_source_skips_the_tracks(self):
        fast, s_fast, s_exact, trips = self._case(wd=2e-9, wdd=3e-15)
        self.assertIsNone(fast)
        self.assertFalse(trips)
        self.assertIsNone(s_fast)
        self.assertIsNone(s_exact)

    def test_fdot_range_trip_is_kept(self):
        fast, s_fast, s_exact, trips = self._case(wd=1e-6, wdd=0.0)
        self.assertTrue(trips)
        self.assertIsNotNone(fast)
        self.assertEqual(s_fast, s_exact)

    def test_curvature_trip_is_kept(self):
        fast, s_fast, s_exact, trips = self._case(wd=0.0, wdd=5e-13)
        self.assertTrue(trips)
        self.assertIsNotNone(fast)
        self.assertEqual(s_fast, s_exact)

    def test_bound_never_hides_a_handoff(self):
        for wd in (0.0, 1e-8, 9e-7, 1e-6):
            for wdd in (0.0, 1e-14, 9e-14, 3e-13):
                fast, s_fast, s_exact, trips = self._case(wd, wdd)
                self.assertEqual(s_fast, s_exact, (wd, wdd))
                if fast is None:
                    self.assertFalse(trips, (wd, wdd))


class NtSubKnobTest(unittest.TestCase):
    """The plunge-chunk length: explicit ``Nt_sub`` > env ``EMRI_DIRECT_NT_SUB`` > 128, refused
    unless it tiles (even, >= 8 so ``n_pad = Nt_sub // 4`` leaves an interior, <= Nt)."""

    def _make_gen(self, **kw):
        # the cheap construction of HandoffBoundTest._direct (grid Nt = 256)
        from lisatools.domains import WDMSettings
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

        table = SimpleNamespace(fdot_vals=np.array([-3.086e-7, 3.086e-7]), INTERP_METHOD="spline")
        gen = SimpleNamespace(inspiral_generator=SimpleNamespace(inspiral_generator=_FakeIntegrator()))
        return EMRIDirectWDM(gen, table, WDMSettings(32, 256, 112.5, force_backend="cpu"), orbits=None,
                             tdi_config=None, t_start=0.0, data_t0=0.0, **kw)

    def test_nt_sub_knob_reads_the_env(self):
        import os

        with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": "64"}):
            self.assertEqual(self._make_gen().Nt_sub, 64)
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": "64"}):
            self.assertEqual(self._make_gen(Nt_sub=96).Nt_sub, 96)     # explicit wins
        self.assertEqual(self._make_gen().Nt_sub, 128)

    def test_nt_sub_knob_refuses_untileable_values(self):
        import os

        for bad in ("7", "6", "0", "-128", "100000"):
            with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": bad}):
                with self.assertRaisesRegex(ValueError, "EMRI_DIRECT_NT_SUB"):
                    self._make_gen()

    def test_nt_sub_knob_auto_reads_its_floor(self):
        """``auto`` / ``auto:<floor>``: one window length per call from the tails (``Nt_sub ==
        "auto"``), never below ``nt_sub_floor`` (128 unless given); the floor tiles like a fixed
        length (even, >= 8, <= Nt). A fixed length is its own floor."""
        import os

        with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": "auto"}):
            g = self._make_gen()
            self.assertEqual((g.Nt_sub, g.nt_sub_floor), ("auto", 128))
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": "auto:96"}):
            g = self._make_gen()
            self.assertEqual((g.Nt_sub, g.nt_sub_floor), ("auto", 96))
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": "64"}):
            self.assertEqual(self._make_gen(Nt_sub="auto:48").nt_sub_floor, 48)   # explicit wins
        g = self._make_gen(Nt_sub=96)
        self.assertEqual((g.Nt_sub, g.nt_sub_floor), (96, 96))
        g = self._make_gen()
        self.assertEqual((g.Nt_sub, g.nt_sub_floor), (128, 128))
        for bad in ("auto:7", "auto:x", "auto:6", "auto:100000", "auto:", "autox", "auto:96:2", "x"):
            with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": bad}):
                with self.assertRaisesRegex(ValueError, "EMRI_DIRECT_NT_SUB"):
                    self._make_gen()

    def test_tail_skip_knob(self):
        """``tail_skip_px``: explicit > env ``EMRI_DIRECT_TAIL_SKIP_PX`` > 0 (never skip); an
        integer >= 0, else ``ValueError`` naming the knob."""
        import os

        self.assertEqual(self._make_gen().tail_skip_px, 0)
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_TAIL_SKIP_PX": "100"}):
            self.assertEqual(self._make_gen().tail_skip_px, 100)
            self.assertEqual(self._make_gen(tail_skip_px=16).tail_skip_px, 16)      # explicit wins
            self.assertEqual(self._make_gen(tail_skip_px=0).tail_skip_px, 0)
        for bad in ("-1", "x", "1.5", ""):
            with mock.patch.dict(os.environ, {"EMRI_DIRECT_TAIL_SKIP_PX": bad}):
                with self.assertRaisesRegex(ValueError, "EMRI_DIRECT_TAIL_SKIP_PX"):
                    self._make_gen()
        with self.assertRaisesRegex(ValueError, "EMRI_DIRECT_TAIL_SKIP_PX"):
            self._make_gen(tail_skip_px=-3)
        for bad in (1.5, 2.0, True):                    # an explicit value is checked as the env one is
            with self.assertRaisesRegex(ValueError, "EMRI_DIRECT_TAIL_SKIP_PX"):
                self._make_gen(tail_skip_px=bad)
        self.assertEqual(self._make_gen(tail_skip_px=np.int64(4)).tail_skip_px, 4)

    def test_walls_knob(self):
        """``walls``: explicit > env ``EMRI_DIRECT_WALLS`` (0 / 1, false / true) > False (off);
        anything else raises naming the knob."""
        import os

        with mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "EMRI_DIRECT_WALLS"},
                             clear=True):
            self.assertFalse(self._make_gen().walls)
        for val, want in (("1", True), ("TRUE", True), ("0", False), ("false", False)):
            with mock.patch.dict(os.environ, {"EMRI_DIRECT_WALLS": val}):
                self.assertIs(self._make_gen().walls, want, val)
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_WALLS": "1"}):
            self.assertFalse(self._make_gen(walls=False).walls)                # explicit wins
        for bad in ("no", "2", ""):
            with mock.patch.dict(os.environ, {"EMRI_DIRECT_WALLS": bad}):
                with self.assertRaisesRegex(ValueError, "EMRI_DIRECT_WALLS"):
                    self._make_gen()


class AssemblyTest(unittest.TestCase):
    """A8: accumulate_harmonic_batch places the lookup before the handoff and the chunk
    after it, and the sum reproduces the TD->WDM truth of a linear chirp across both."""

    NF, NT, DT = 64, 128, 56.25

    @classmethod
    def setUpClass(cls):
        import os
        import tempfile

        from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings

        cls.wdm = WDMSettings(Nf=cls.NF, Nt=cls.NT, dt=cls.DT, force_backend="cpu")
        nf, md, mr = WDMLookupTable.apply_eps_frequency(0.005, cls.wdm, m_ref=20, num_layers_diff=2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.table = WDMLookupTable(cls.wdm, 1, m_ref=mr, norm_freq_single_layer=nf, m_diffs=md,
                                   fdot_vals=WDMLookupTable.apply_eps_fdot(0.05, cls.wdm, fdot_max_factor=1.0),
                                   store_path=os.path.join(cls.tmp.name, "t.h5"), batch_size_gen=32,
                                   build_kind="n_ref_complex", time_layers=64)
        df, ldt = cls.wdm.layer_df, cls.wdm.layer_dt
        # a slow chirp (0.05 layer per pixel, on a table node): 5 layers hold its power;
        # faster chirps need more neighbour layers (open item in docs/emri-direct-wdm.md)
        cls.f0, cls.fdot, cls.phi0 = 18.2 * df, 0.05 * df / ldt, 0.4
        N = cls.NF * cls.NT
        cls.t = np.arange(N) * cls.DT
        cls.y = np.cos(2 * np.pi * (cls.f0 * cls.t + 0.5 * cls.fdot * cls.t ** 2) + cls.phi0)[None, :]
        cls.truth = np.asarray(TDSignal(cls.y, TDSettings(N, cls.DT, force_backend="cpu")).transform(cls.wdm).arr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, n_h):
        from lisatools.sources.emri import wdm_direct as wd

        ldt, ldf = self.wdm.layer_dt, self.wdm.layer_df
        n_ok = np.arange(40, self.NT - 8)
        tn = n_ok * ldt
        f = self.f0 + self.fdot * tn
        phase = 2 * np.pi * (self.f0 * tn + 0.5 * self.fdot * tn ** 2) + self.phi0
        fddot = np.where(n_ok >= n_h, 1.0, 0.0)          # trips the curvature trigger at n_h
        track = wd.HarmonicTrack((2, 2, 0, 0), tn, np.ones_like(tn), phase, f, np.full_like(tn, self.fdot), fddot)
        tracer = (np.ones((1, 1, tn.size)), phase[None, None], f[None, None], np.full((1, 1, tn.size), self.fdot))
        tail_td = lambda ts: np.cos(2 * np.pi * (self.f0 * ts + 0.5 * self.fdot * ts ** 2) + self.phi0)[None, None, :]
        acc = np.zeros((1, self.NF, self.NT))
        stats = wd.accumulate_harmonic_batch(
            acc, self.table, [track], tracer, n_ok, tail_td, Nf=self.NF, Nt=self.NT, dt=self.DT,
            layer_dt=ldt, layer_df=ldf, t0=0.0, Nt_sub=128, num_m_layers=2,
            fdot_axis_max=float(np.max(np.abs(self.table.fdot_vals))), pixel_edge=8)
        return acc, stats, n_ok

    def test_lookup_then_chunk_reproduces_truth(self):
        n_h = 90
        acc, stats, n_ok = self._run(n_h)
        self.assertEqual(stats["lookup_pixels"], int(np.sum(n_ok < n_h)))
        self.assertGreater(stats["chunk_pixels"], 0)
        sl = slice(40, self.NT - 32)   # the (non-stopping) truth carries grid-end contamination
        rel = np.linalg.norm(acc[..., sl] - self.truth[..., sl]) / np.linalg.norm(self.truth[..., sl])
        self.assertLess(rel, 2e-3, f"rel L2 {rel:.2e}")
        # nothing written before the first tracked pixel
        self.assertTrue(np.all(acc[..., :32] == 0.0))

    def test_subs_sharing_a_chunk_range_are_transformed_together(self):
        """Two half-amplitude copies of the harmonic (same handoff, so one keep range per chunk:
        their tails are summed and transformed once) == the single harmonic."""
        from lisatools.sources.emri import wdm_direct as wd

        n_h = 90
        one, st1, n_ok = self._run(n_h)
        ldt, ldf = self.wdm.layer_dt, self.wdm.layer_df
        tn = n_ok * ldt
        f = self.f0 + self.fdot * tn
        phase = 2 * np.pi * (self.f0 * tn + 0.5 * self.fdot * tn ** 2) + self.phi0
        fddot = np.where(n_ok >= n_h, 1.0, 0.0)
        track = wd.HarmonicTrack((2, 2, 0, 0), tn, np.ones_like(tn), phase, f, np.full_like(tn, self.fdot), fddot)
        tracer = (np.full((2, 1, tn.size), 0.5), np.stack([phase[None]] * 2), np.stack([f[None]] * 2),
                  np.full((2, 1, tn.size), self.fdot))
        tail_td = lambda ts: 0.5 * np.repeat(np.cos(2 * np.pi * (self.f0 * ts + 0.5 * self.fdot * ts ** 2)
                                                    + self.phi0)[None, None, :], 2, axis=0)
        two = np.zeros((1, self.NF, self.NT))
        st2 = wd.accumulate_harmonic_batch(
            two, self.table, [track, track], tracer, n_ok, tail_td, Nf=self.NF, Nt=self.NT, dt=self.DT,
            layer_dt=ldt, layer_df=ldf, t0=0.0, Nt_sub=128, num_m_layers=2,
            fdot_axis_max=float(np.max(np.abs(self.table.fdot_vals))), pixel_edge=8)
        self.assertEqual(st2["chunk_pixels"], 2 * st1["chunk_pixels"])
        # the pixels the chunk filled: once per keep range, not per sub
        self.assertEqual(st1["tail_layers"], st1["chunk_pixels"])
        self.assertEqual((st2["windows"], st2["tail_layers"]), (st1["windows"], st1["tail_layers"]))
        np.testing.assert_allclose(two, one, rtol=0, atol=1e-12 * np.abs(one).max())

    def test_common_handoff_with_a_summed_tail(self):
        """Two harmonics handing off at different pixels: with common_handoff both go at the
        earlier one, and a summed tail (tail_td.summed, which implies the common handoff) == the
        per-sub tail at that handoff."""
        from lisatools.sources.emri import wdm_direct as wd

        ldt, ldf = self.wdm.layer_dt, self.wdm.layer_df
        n_ok = np.arange(40, self.NT - 8)
        tn = n_ok * ldt
        f = self.f0 + self.fdot * tn
        phase = 2 * np.pi * (self.f0 * tn + 0.5 * self.fdot * tn ** 2) + self.phi0

        def track(n_h):
            return wd.HarmonicTrack((2, 2, 0, 0), tn, np.ones_like(tn), phase, f, np.full_like(tn, self.fdot),
                                    np.where(n_ok >= n_h, 1.0, 0.0))

        tracer = (np.full((2, 1, tn.size), 0.5), np.stack([phase[None]] * 2), np.stack([f[None]] * 2),
                  np.full((2, 1, tn.size), self.fdot))
        y = lambda ts: 0.5 * np.cos(2 * np.pi * (self.f0 * ts + 0.5 * self.fdot * ts ** 2) + self.phi0)
        per_sub = lambda ts: np.repeat(y(ts)[None, None, :], 2, axis=0)

        def summed(ts):
            return 2 * y(ts)[None, :]
        summed.summed = True
        kw = dict(Nf=self.NF, Nt=self.NT, dt=self.DT, layer_dt=ldt, layer_df=ldf, t0=0.0, Nt_sub=128,
                  num_m_layers=2, fdot_axis_max=float(np.max(np.abs(self.table.fdot_vals))), pixel_edge=8)
        want = np.zeros((1, self.NF, self.NT))
        wd.accumulate_harmonic_batch(want, self.table, [track(90), track(90)], tracer, n_ok, per_sub, **kw)
        got = np.zeros_like(want)
        st = wd.accumulate_harmonic_batch(got, self.table, [track(90), track(100)], tracer, n_ok, summed,
                                          common_handoff=True, **kw)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())
        self.assertGreater(st["chunk_pixels"], 0)
        implied = np.zeros_like(want)                     # summed tail, no explicit common_handoff
        wd.accumulate_harmonic_batch(implied, self.table, [track(90), track(100)], tracer, n_ok, summed, **kw)
        np.testing.assert_allclose(implied, want, rtol=0, atol=1e-12 * np.abs(want).max())
        apart = np.zeros_like(want)                       # per-sub tail: each harmonic at its own pixel
        wd.accumulate_harmonic_batch(apart, self.table, [track(90), track(100)], tracer, n_ok, per_sub, **kw)
        self.assertGreater(np.abs(apart - want).max(), 1e-6 * np.abs(want).max())

    # -- many templates with plunge chunks in ONE call (sub_row): two rows of two half-amplitude
    # copies of the chirp each (row r's carrier offset by 0.7 r rad); a row's subs hand off at
    # k_row[r] and k_row[r] + 15, so its common handoff is k_row[r]; Nt_sub = 64 gives each row two
    # tail windows
    k_row0, k_row1 = 80, 70
    NT_SUB = 64

    def _row_signal(self, ts, r):
        """Row ``r``'s channel time series (its two half-amplitude subs summed)."""
        return np.cos(2 * np.pi * (self.f0 * ts + 0.5 * self.fdot * ts ** 2) + self.phi0 + 0.7 * r)

    def _row_tracks(self, r):
        """Row ``r``'s two tracks, its tracer ``(2, 1, P)`` and the tracked pixels."""
        from lisatools.sources.emri import wdm_direct as wd

        n_ok = np.arange(40, self.NT - 8)
        tn = n_ok * self.wdm.layer_dt
        f = self.f0 + self.fdot * tn
        phase = 2 * np.pi * (self.f0 * tn + 0.5 * self.fdot * tn ** 2) + self.phi0 + 0.7 * r
        k = (self.k_row0, self.k_row1)[r]
        tracks = [wd.HarmonicTrack((2, 2, 0, 0), tn, np.ones_like(tn), phase, f, np.full_like(tn, self.fdot),
                                   np.where(n_ok >= n_h, 1.0, 0.0)) for n_h in (k, k + 15)]
        tracer = (np.full((2, 1, tn.size), 0.5), np.stack([phase[None]] * 2), np.stack([f[None]] * 2),
                  np.full((2, 1, tn.size), self.fdot))
        return tracks, tracer, n_ok

    def _kw(self):
        return dict(Nf=self.NF, Nt=self.NT, dt=self.DT, layer_dt=self.wdm.layer_dt, layer_df=self.wdm.layer_df,
                    t0=0.0, Nt_sub=self.NT_SUB, num_m_layers=2,
                    fdot_axis_max=float(np.max(np.abs(self.table.fdot_vals))), pixel_edge=8)

    def _tail_two_rows(self):
        """The pairs form of the plunge tail: ``(row, n0, klo, khi)`` -> row's summed series on the
        window grid of n0, ``(len(pairs), nch, Nf * Nt_sub)``; counts its calls."""
        N = self.NF * self.NT_SUB

        def tail(pairs):
            tail.calls += 1
            return np.stack([self._row_signal((n0 * self.NF + np.arange(N)) * self.DT, r)[None, :]
                             for r, n0, _, _ in pairs])
        tail.calls, tail.summed, tail.by_pairs = 0, True, True
        return tail

    def _accumulate_two_rows(self, sub_row, tail_td):
        from lisatools.sources.emri import wdm_direct as wd

        tr0, c0, n_ok = self._row_tracks(0)
        tr1, c1, _ = self._row_tracks(1)
        tracer = tuple(np.concatenate([a, b]) for a, b in zip(c0, c1))
        acc = np.zeros((2, 1, self.NF, self.NT))
        stats = wd.accumulate_harmonic_batch(acc, self.table, tr0 + tr1, tracer, n_ok, tail_td, sub_row=sub_row,
                                             **self._kw())
        return acc, stats

    def _accumulate_one_row(self, row):
        """Row ``row`` alone (``sub_row=None``), its tail in the single-template ``tail_td(ts)``
        form (summed)."""
        from lisatools.sources.emri import wdm_direct as wd

        tracks, tracer, n_ok = self._row_tracks(row)

        def tail(ts):
            return self._row_signal(ts, row)[None, :]
        tail.summed = True
        acc = np.zeros((1, self.NF, self.NT))
        stats = wd.accumulate_harmonic_batch(acc, self.table, tracks, tracer, n_ok, tail, **self._kw())
        return acc, stats

    def test_common_handoff_is_per_row(self):
        # rows 0 and 1 each with 2 subs; row 1 hands off 10 pixels earlier
        acc, stats = self._accumulate_two_rows(sub_row=np.array([0, 0, 1, 1]), tail_td=self._tail_two_rows())
        self.assertEqual(stats["handoff_pixel_per_row"], [self.k_row0, self.k_row1])   # expose in stats
        # row 0's lookup ran up to ITS handoff: pixels between k_row1 and k_row0 are non-zero on row 0
        self.assertTrue(np.any(acc[0, :, :, self.k_row1:self.k_row0] != 0))
        # row 1's first tail window starts before row 0's (its handoff is earlier)
        self.assertLess(min(n0 for (r, n0, _, _) in stats["pairs"] if r == 1),
                        min(n0 for (r, n0, _, _) in stats["pairs"] if r == 0))
        self.assertEqual(stats["windows"], len(stats["pairs"]))

    def test_batched_tail_matches_single_row_tails(self):
        acc_b, _ = self._accumulate_two_rows(sub_row=np.array([0, 0, 1, 1]), tail_td=self._tail_two_rows())
        acc_0, _ = self._accumulate_one_row(row=0)
        acc_1, _ = self._accumulate_one_row(row=1)
        np.testing.assert_allclose(acc_b[0], acc_0, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(acc_b[1], acc_1, rtol=1e-12, atol=1e-14)

    def test_tail_is_one_dense_call_and_one_transform(self):
        from lisatools import wdm_het

        # two rows, two windows each -> 4 pairs; the tail function and the transform run ONCE
        tail = self._tail_two_rows(); tail.calls = 0
        with mock.patch("lisatools.wdm_het.wdm_chunk_of_td", wraps=wdm_het.wdm_chunk_of_td) as tr:
            _, stats = self._accumulate_two_rows(sub_row=np.array([0, 0, 1, 1]), tail_td=tail)
        self.assertEqual(tail.calls, 1)
        self.assertEqual(tr.call_count, 1)
        self.assertEqual(tr.call_args[0][0].shape[0], stats["windows"])     # all pairs in one batch
        self.assertEqual(len(stats["pairs"]), 4)

    def test_all_lookup_matches_truth_too(self):
        acc, stats, n_ok = self._run(10 ** 6)          # never hands off
        self.assertEqual(stats["chunk_pixels"], 0)
        sl = slice(40, self.NT - 32)   # the (non-stopping) truth carries grid-end contamination
        rel = np.linalg.norm(acc[..., sl] - self.truth[..., sl]) / np.linalg.norm(self.truth[..., sl])
        self.assertLess(rel, 2e-3, f"rel L2 {rel:.2e}")

class TailLeverTest(unittest.TestCase):
    """The window length per call (EMRI_DIRECT_NT_SUB=auto), the conditional tail
    (EMRI_DIRECT_TAIL_SKIP_PX) and the phase walls of accumulate_harmonic_batch, on rows of
    AssemblyTest's two-row fixture with chosen handoff-to-end spans (the pixels from a row's common
    handoff to its last tracked pixel, inclusive) on grids long enough for 164- to 392-layer
    windows (the table is grid-length agnostic). AssemblyTest's fixture is reused, but this class
    does not derive from it: VectorisedAccumulateTest(AssemblyTest) re-runs every AssemblyTest
    test."""

    NF, NT, DT = AssemblyTest.NF, AssemblyTest.NT, AssemblyTest.DT
    setUpClass = classmethod(AssemblyTest.setUpClass.__func__)
    tearDownClass = classmethod(AssemblyTest.tearDownClass.__func__)
    _row_signal = AssemblyTest._row_signal


    def _span_rows(self, spans, n_last):
        """One row per span: two half-amplitude subs of the chirp (row r's carrier offset by 0.7 r
        rad, as :meth:`_row_signal`) handing off ``spans[r] - 1`` pixels before the last tracked
        pixel ``n_last`` and 3 pixels later (the row's common handoff is the first). Returns
        ``(tracks, tracer, n_ok, sub_row)``."""
        from lisatools.sources.emri import wdm_direct as wd

        n_ok = np.arange(40, n_last + 1)
        tn = n_ok * self.wdm.layer_dt
        f = self.f0 + self.fdot * tn
        tracks, parts = [], []
        for r, s in enumerate(spans):
            n_h = n_last - s + 1
            phase = 2 * np.pi * (self.f0 * tn + 0.5 * self.fdot * tn ** 2) + self.phi0 + 0.7 * r
            tracks += [wd.HarmonicTrack((2, 2, 0, 0), tn, np.ones_like(tn), phase, f, np.full_like(tn, self.fdot),
                                        np.where(n_ok >= k, 1.0, 0.0)) for k in (n_h, n_h + 3)]
            parts.append((np.full((2, 1, tn.size), 0.5), np.stack([phase[None]] * 2), np.stack([f[None]] * 2),
                          np.full((2, 1, tn.size), self.fdot)))
        tracer = tuple(np.concatenate(x) for x in zip(*parts))
        return tracks, tracer, n_ok, np.repeat(np.arange(len(spans)), 2)

    def _tail_rows(self, nt_sub):
        """The pairs-form tail of :meth:`_span_rows`' rows: windows of ``nt_sub`` layers unless the
        call passes its own ``Nt_sub`` (the auto window length); records its calls and lengths."""
        def tail(pairs, Nt_sub=None):
            tail.calls += 1
            tail.lengths.append(Nt_sub)
            N = self.NF * (nt_sub if Nt_sub is None else Nt_sub)
            return np.stack([self._row_signal((n0 * self.NF + np.arange(N)) * self.DT, r)[None, :]
                             for r, n0, _, _ in pairs])
        tail.calls, tail.lengths, tail.summed, tail.by_pairs = 0, [], True, True
        return tail

    def _accumulate_spans(self, spans, n_last, Nt, pixel_edge, Nt_sub=128, **kw):
        """:func:`accumulate_harmonic_batch` over :meth:`_span_rows`' rows on an ``Nt``-layer grid;
        ``self.transform_calls`` / ``self.tail_calls`` / ``self.tail_lengths`` count the
        ``wdm_chunk_of_td`` and tail calls. Returns ``(acc, stats)``."""
        from lisatools import wdm_het
        from lisatools.sources.emri import wdm_direct as wd

        tracks, tracer, n_ok, sub_row = self._span_rows(spans, n_last)
        tail = self._tail_rows(Nt_sub)
        acc = np.zeros((len(spans), 1, self.NF, Nt))
        with mock.patch("lisatools.wdm_het.wdm_chunk_of_td", wraps=wdm_het.wdm_chunk_of_td) as tr:
            stats = wd.accumulate_harmonic_batch(
                acc, self.table, tracks, tracer, n_ok, tail, Nf=self.NF, Nt=Nt, dt=self.DT,
                layer_dt=self.wdm.layer_dt, layer_df=self.wdm.layer_df, t0=0.0, Nt_sub=Nt_sub, num_m_layers=2,
                fdot_axis_max=float(np.max(np.abs(self.table.fdot_vals))), pixel_edge=pixel_edge,
                sub_row=sub_row, **kw)
        self.transform_calls, self.tail_calls, self.tail_lengths = tr.call_count, tail.calls, tail.lengths
        return acc, stats

    # both modes clip the tail end at Nt - pixel_edge here (pixel_edge 64 on a 320-layer grid, last
    # tracked pixel 223: every row's tail ends at pixel 256), so a FIXED run at the auto length
    # plans exactly the auto windows -- otherwise its past-end extent, Nt_sub // 4, would differ
    # from auto's nt_sub_floor // 4 -- and the windows stay clear of the grid-end chunk
    AUTO_GEOM = dict(n_last=223, Nt=320, pixel_edge=64)

    def test_auto_window_fits_every_row_in_one_window(self):
        acc, stats = self._accumulate_spans((37, 48), nt_sub_auto=True, nt_sub_floor=128, **self.AUTO_GEOM)
        self.assertEqual(stats["nt_sub_eff"], 164)          # 2 * (48 + 32) + 4 = 164, a multiple of 4, >= 128
        self.assertEqual(stats["windows"], 2)               # one per row
        self.assertEqual(sorted(r for (r, *_) in stats["pairs"]), [0, 1])
        self.assertEqual((self.transform_calls, self.tail_calls), (1, 1))
        self.assertEqual(self.tail_lengths, [164])          # the pairs form is told the call's length
        acc_fixed, stats_fixed = self._accumulate_spans((37, 48), Nt_sub=164, **self.AUTO_GEOM)
        self.assertEqual(stats_fixed["pairs"], stats["pairs"])
        self.assertEqual(self.tail_lengths, [None])         # fixed mode: the call is unchanged
        np.testing.assert_allclose(acc, acc_fixed, rtol=1e-12, atol=1e-14)   # auto == the same fixed length
        # ... and the template of the default 128-layer windows (two per row here) up to their
        # edge contamination (measured 3.1e-6 of the peak; 128's worst kept pixel ~2e-6)
        acc_128, st_128 = self._accumulate_spans((37, 48), Nt_sub=128, **self.AUTO_GEOM)
        self.assertEqual((st_128["windows"], st_128["nt_sub_eff"]), (4, 128))
        rel = np.abs(acc - acc_128).max() / np.abs(acc_128).max()
        self.assertLess(rel, 1e-5, f"auto vs fixed 128: max rel {rel:.2e}")

    def test_auto_window_respects_floor_and_grid(self):
        _, stats = self._accumulate_spans((5, 7), nt_sub_auto=True, nt_sub_floor=128, **self.AUTO_GEOM)
        self.assertEqual((stats["nt_sub_eff"], stats["windows"]), (128, 2))   # tiny tails: the floor
        _, stats = self._accumulate_spans((5, 7), nt_sub_auto=True, nt_sub_floor=96, **self.AUTO_GEOM)
        self.assertEqual(stats["nt_sub_eff"], 96)           # 2 * (7 + 24) + 4 = 66 -> 68 < 96
        # tails too long for one window on the grid: capped at the largest even length <= Nt -
        # pixel_edge, tiled (still one transform), and the fixed result at that length
        geom = dict(n_last=359, Nt=400, pixel_edge=8)
        acc, stats = self._accumulate_spans((300, 310), nt_sub_auto=True, nt_sub_floor=128, **geom)
        self.assertEqual(stats["nt_sub_eff"], 392)          # 2 * (310 + 32) + 4 = 688 > 392
        self.assertLessEqual(stats["nt_sub_eff"], 392)
        self.assertEqual(stats["nt_sub_eff"] % 2, 0)
        self.assertGreaterEqual(stats["windows"], 2)        # a tail that cannot fit is tiled
        self.assertEqual(self.transform_calls, 1)
        acc_fixed, _ = self._accumulate_spans((300, 310), Nt_sub=392, **geom)
        np.testing.assert_allclose(acc, acc_fixed, rtol=1e-12, atol=1e-14)
        _, stats = self._accumulate_spans((300, 310), nt_sub_auto=True, nt_sub_floor=128,
                                          **dict(geom, pixel_edge=9))
        self.assertEqual(stats["nt_sub_eff"], 390)          # Nt - pixel_edge = 391: the even 390

    def test_tail_skip_is_per_row(self):
        """``tail_skip_px``: a row whose span is at most the knob keeps the lookup through its last
        pixel -- exactly the row that never trips -- and no pairs; the other row of the same call
        keeps its tail."""
        acc, stats = self._accumulate_spans((10, 48), tail_skip_px=16, **self.AUTO_GEOM)
        self.assertEqual((self.transform_calls, self.tail_calls), (1, 1))
        self.assertEqual((stats["tail_skipped"], stats["tail_skipped_rows"]), (1, [0]))
        self.assertEqual(sorted({r for r, *_ in stats["pairs"]}), [1])        # row 0: no pairs
        self.assertEqual(stats["windows"], len({(r, n0) for r, n0, _, _ in stats["pairs"]}))   # row 1 only
        self.assertEqual(stats["tail_layers"][0], 0)
        self.assertIsNone(stats["handoff_pixel_per_row"][0])                  # no handoff, as no trip
        # the lookup ran through row 0's last pixel: 2 subs x all 184 pixels, row 1's 2 subs up to
        # its handoff (pixel 176 = index 136)
        self.assertEqual((stats["lookup_pixels"], stats["dropped_pixels"]), (2 * 184 + 2 * 136, 0))
        never, st_never = self._accumulate_spans((0, 48), **self.AUTO_GEOM)   # span 0: row 0 never trips
        np.testing.assert_array_equal(acc[0], never[0])
        self.assertEqual(stats["lookup_pixels"], st_never["lookup_pixels"])
        full, st_full = self._accumulate_spans((10, 48), **self.AUTO_GEOM)   # no skip: both tails
        self.assertEqual(st_full["tail_skipped"], 0)
        self.assertEqual(stats["pairs"], [p for p in st_full["pairs"] if p[0] == 1])
        self.assertEqual(stats["handoff_pixel_per_row"][1], st_full["handoff_pixel_per_row"][1])
        self.assertEqual(stats["tail_layers"][1], st_full["tail_layers"][1])
        np.testing.assert_allclose(acc[1], full[1], rtol=1e-12, atol=1e-14)
        self.assertGreater(np.abs(acc[0] - full[0]).max(), 1e-6 * np.abs(full[0]).max())   # paired control

    def test_tail_skip_zero_is_todays_behaviour(self):
        for kw in (dict(), dict(nt_sub_auto=True, nt_sub_floor=128)):
            a0, s0 = self._accumulate_spans((10, 48), **kw, **self.AUTO_GEOM)
            a1, s1 = self._accumulate_spans((10, 48), tail_skip_px=0, **kw, **self.AUTO_GEOM)
            np.testing.assert_array_equal(a1, a0)
            self.assertEqual(*({k: v for k, v in st.items() if k != "walls"} for st in (s1, s0)))   # walls: timing
            self.assertEqual((s1["tail_skipped"], s1["tail_skipped_rows"]), (0, []))

    def test_accumulate_records_its_phases(self):
        """``walls``: the call's phases go into the caller's clock and come back as ``stats["walls"]``
        (this call only)."""
        from lisatools.sources.emri.wdm_direct import _PhaseWalls

        clock = _PhaseWalls()
        with clock.phase("outer"):
            _, st = self._accumulate_spans((10, 48), walls=clock, **self.AUTO_GEOM)
        self.assertEqual(set(st["walls"]), {"plan", "lookup", "tail_dense", "transform", "splice"})
        self.assertTrue(all(v >= 0 for v in st["walls"].values()))
        for k, v in st["walls"].items():
            self.assertAlmostEqual(clock.walls[k], v, places=9)    # one call: the clock holds exactly these
        self.assertIn("outer", clock.walls)                         # the enclosing phase keeps the rest

    def test_auto_window_excludes_skipped_rows(self):
        """Ruling R1: the auto length comes from the rows that KEEP a tail (the skip acts first)."""
        _, st = self._accumulate_spans((5, 7), nt_sub_auto=True, nt_sub_floor=8, **self.AUTO_GEOM)
        self.assertEqual(st["nt_sub_eff"], 24)                # 2 * (7 + 2) + 4 = 22 -> 24
        _, st = self._accumulate_spans((5, 7), nt_sub_auto=True, nt_sub_floor=8, tail_skip_px=16, **self.AUTO_GEOM)
        self.assertEqual((st["nt_sub_eff"], st["windows"], st["tail_skipped"]), (8, 0, 2))   # no tail: the floor
        _, st = self._accumulate_spans((10, 48), nt_sub_auto=True, nt_sub_floor=8, tail_skip_px=16,
                                       **self.AUTO_GEOM)
        self.assertEqual((st["nt_sub_eff"], st["windows"]), (104, 1))   # row 1 alone: 2 * (48 + 2) + 4


    def test_non_batched_return_reports_window_and_skip(self):
        """One template (``sub_row=None``, a summed single-template tail): the stats carry
        ``nt_sub_eff`` (the fixed length here) and ``tail_skipped``, as the batched ones do; no
        clock, no ``walls``."""
        from lisatools.sources.emri import wdm_direct as wd

        def one(span, **kw):
            tracks, tracer, n_ok, _ = self._span_rows((span,), 223)
            tail = lambda ts: self._row_signal(ts, 0)[None, :]                   # noqa: E731
            tail.summed = True
            acc = np.zeros((1, self.NF, 320))
            return wd.accumulate_harmonic_batch(
                acc, self.table, tracks[:1], tuple(a[:1] for a in tracer), n_ok, tail, Nf=self.NF, Nt=320,
                dt=self.DT, layer_dt=self.wdm.layer_dt, layer_df=self.wdm.layer_df, t0=0.0, Nt_sub=128,
                num_m_layers=2, fdot_axis_max=float(np.max(np.abs(self.table.fdot_vals))), pixel_edge=64, **kw)
        st = one(48)
        self.assertEqual((st["nt_sub_eff"], st["tail_skipped"]), (128, 0))
        self.assertGreater(st["windows"], 0)
        self.assertNotIn("walls", st)
        st = one(10, tail_skip_px=16)
        self.assertEqual((st["tail_skipped"], st["windows"], st["chunk_pixels"]), (1, 0, 0))


class PhaseWallsTest(unittest.TestCase):
    """_PhaseWalls: a nested phase pauses the enclosing one (the phases partition the covered time),
    a detail timer does not."""

    def test_nested_phases_are_exclusive(self):
        import time

        from lisatools.sources.emri.wdm_direct import _PhaseWalls

        c = _PhaseWalls()
        with c.phase("a") as ra:
            time.sleep(0.02)
            with c.phase("b") as rb:
                time.sleep(0.03)
            with c.detail("d"):
                time.sleep(0.01)
        self.assertAlmostEqual(c.walls["a"] + c.walls["b"], ra["s"], places=9)
        self.assertAlmostEqual(c.walls["b"], rb["s"], places=9)
        self.assertGreater(c.walls["b"], 0.025)
        self.assertGreater(c.walls["a"], 0.025)                    # 0.02 + the 0.01 detail
        self.assertGreater(c.detail_walls["d"], 0.005)
        self.assertEqual(set(c.walls), {"a", "b"})

    def test_the_clock_holds_no_array_module_and_pickles(self):
        """LAT rule: never an array module as an attribute -- only whether to synchronise."""
        import copy
        import pickle
        import types

        from lisatools.sources.emri.wdm_direct import _PhaseWalls

        c = _PhaseWalls(np)
        with c.phase("a"):
            pass
        self.assertFalse(any(isinstance(v, types.ModuleType) for v in vars(c).values()))
        self.assertFalse(c._sync_device)
        back = pickle.loads(pickle.dumps(copy.deepcopy(c)))
        self.assertEqual(set(back.walls), {"a"})


class HolderSliceTest(unittest.TestCase):
    """slice_holder + track_rows: the tracks of a mode subset taken from the full holder's
    tracks equal the tracks built from the sliced holder (same rows, same order), and the TOF
    feed of the sliced holder equals the matching rows of the full feed."""

    def test_subset_rows_match_sliced_holder(self):
        from lisatools.sources.emri.emritdionfly import EMRITDIonFly
        from lisatools.sources.emri.wdm_direct import harmonic_tracks_from_holder, slice_holder, track_rows

        t_k = np.linspace(0.0, 1e6, 60)
        rng = np.random.default_rng(4)
        teuk = rng.normal(size=(t_k.size, 4)) + 1j * rng.normal(size=(t_k.size, 4))
        ylms = rng.normal(size=8) + 1j * rng.normal(size=8)
        H = SimpleNamespace(t_arr=t_k, teuk_modes=teuk, ylms=ylms, ls=np.array([2, 2, 3, 4]),
                            ms=np.array([2, 0, 1, 3]), ks=np.zeros(4, int), ns=np.array([0, 1, -1, 2]),
                            phases=np.stack([1e-3 * t_k, 0 * t_k, 3e-4 * t_k], -1), freqs=None,
                            integrate_backwards=False)
        integ = _FakeIntegrator()
        t_pix = np.arange(5, 200) * 3600.0
        full = harmonic_tracks_from_holder(H, integ, t_pix, a=0.9, xI0=1.0)
        amp_f, ph_f = EMRITDIonFly.mode_amp_phase(H, include_minus_mkn=True)
        for idx in (np.array([1, 2]), np.array([0, 3]), np.array([1])):
            Hs = slice_holder(H, idx)
            sub = harmonic_tracks_from_holder(Hs, integ, t_pix, a=0.9, xI0=1.0)
            rows = track_rows(H, idx)
            self.assertEqual(len(rows), len(sub))
            for r, tr in zip(rows, sub):
                self.assertEqual(full[r].lmkn, tr.lmkn)
                np.testing.assert_array_equal(full[r].amp, tr.amp)
                np.testing.assert_array_equal(full[r].phase, tr.phase)
            amp_s, ph_s = EMRITDIonFly.mode_amp_phase(Hs, include_minus_mkn=True)
            np.testing.assert_allclose(amp_s, amp_f[rows], rtol=1e-14, atol=0)
            np.testing.assert_allclose(ph_s, ph_f[rows], rtol=1e-12, atol=1e-12)


class VectorisedAccumulateTest(AssemblyTest):
    """accumulate_harmonic_batch (one table call + one scatter-add for every sub, channel
    and pixel) == the per-(sub, channel) loop it replaced, stats included."""

    def test_vectorised_equals_loop(self):
        from lisatools.sources.emri import wdm_direct as wd

        ldt, ldf = self.wdm.layer_dt, self.wdm.layer_df
        n_ok = np.arange(40, self.NT - 8)
        tn = n_ok * ldt
        tracks, amps, phs, fs, fds = [], [], [], [], []
        for s, (f0, n_h) in enumerate(((18.2, 90), (18.6, 10 ** 6), (22.9, 70))):   # subs 0,1 share layers: duplicate scatter targets
            f0 = f0 * ldf
            f = f0 + self.fdot * tn
            ph = 2 * np.pi * (f0 * tn + 0.5 * self.fdot * tn ** 2) + 0.3 * s
            fdd = np.where(n_ok >= n_h, 1.0, 0.0)
            tracks.append(wd.HarmonicTrack((2, 2, s, 0), tn, np.ones_like(tn), ph, f, np.full_like(tn, self.fdot), fdd))
            amps.append(np.stack([np.ones_like(tn), 0.5 + 0.1 * np.sin(tn / 1e5)]))
            phs.append(np.stack([ph, ph + 0.7]))
            fs.append(np.stack([f, f]))
            fd = np.full((2, tn.size), self.fdot)
            fd[1, :5] = 1e3                                   # off the table axis: dropped
            fds.append(fd)
        tracer = tuple(np.stack(x) for x in (amps, phs, fs, fds))
        tail_td = lambda ts: np.stack([np.stack([np.cos(2 * np.pi * 1e-4 * ts + k), np.sin(2 * np.pi * 1e-4 * ts)])
                                       for k in range(3)])
        kw = dict(Nf=self.NF, Nt=self.NT, dt=self.DT, layer_dt=ldt, layer_df=ldf, t0=0.0, Nt_sub=128,
                  num_m_layers=2, fdot_axis_max=float(np.max(np.abs(self.table.fdot_vals))), pixel_edge=8)
        a1, a2 = np.zeros((2, self.NF, self.NT)), np.zeros((2, self.NF, self.NT))
        s1 = wd.accumulate_harmonic_batch(a1, self.table, tracks, tracer, n_ok, tail_td, **kw)
        a3 = np.zeros_like(a1)                                   # tiny lookup chunks: same result
        s3 = wd.accumulate_harmonic_batch(a3, self.table, tracks, tracer, n_ok, tail_td, lookup_chunk=37, **kw)
        self.assertEqual(s1, s3)
        np.testing.assert_allclose(a3, a1, rtol=0, atol=1e-14 * np.max(np.abs(a1)))
        s2 = wd._accumulate_harmonic_batch_loop(a2, self.table, tracks, tracer, n_ok, tail_td, **kw)
        self.assertEqual(s1, s2)
        self.assertGreater(s1["dropped_pixels"], 0)
        self.assertGreater(s1["chunk_pixels"], 0)
        np.testing.assert_allclose(a1, a2, rtol=0, atol=1e-13 * np.max(np.abs(a2)))


class TracerTest(unittest.TestCase):
    """tracer_from_tof_output: amplitude, phase and analytic f/fdot of a known channel
    phase; negative-frequency subs mirrored to positive frequency."""

    def test_known_quadratic_phase(self):
        from lisatools.sources.emri.wdm_direct import tracer_from_tof_output

        f0, fd = 3e-3, 2e-9

        class Out:
            def eval_spline_vals(self, t):
                t = np.asarray(t, float)
                ph = 2 * np.pi * (f0 * t + 0.5 * fd * t ** 2)
                amp = np.stack([np.full((3, t.size), 2.0), np.full((3, t.size), 1.0)])
                tdi_phase = np.stack([np.zeros((3, t.size)), np.zeros((3, t.size))])
                phase_ref = np.stack([ph, -ph])                    # sub 1: negative frequency
                return amp, tdi_phase, phase_ref

        t = np.linspace(1e5, 2e5, 11)
        amp, phase, f, fdot = tracer_from_tof_output(Out(), t)
        ph = 2 * np.pi * (f0 * t + 0.5 * fd * t ** 2)
        for s_ in (0, 1):
            np.testing.assert_allclose(f[s_], np.broadcast_to(f0 + fd * t, (3, t.size)), rtol=1e-8)
            np.testing.assert_allclose(fdot[s_], fd, rtol=1e-6)
            np.testing.assert_allclose(phase[s_], np.broadcast_to(ph, (3, t.size)), rtol=1e-12)
        np.testing.assert_allclose(amp[0], 2.0)


class DenseTailBatchTest(unittest.TestCase):
    """EMRIDirectWDM._dense_td over a LIST of items, one time grid each: one kernel per block of
    whole items of one grid length (EMRI_DIRECT_TAIL_BLOCK_ELEMS bounds the block's output entries
    -- items x channels x samples -- PLUS its items' dense amplitude inputs, 8 x harmonics x the
    block's largest knot count; an item's samples are never split), the channels split per item on
    the kernel's template axis; and _dense_tail_fn mapping the (row, n0, klo, khi) pairs in ONE
    _dense_td call to ONE item per ROW, its windows' grids concatenated (a row's inputs are never
    repeated per window), cut back into the pairs' series. The kernel is a stand-in here (the
    compiled one: tests/test_tdi_dense.py)."""

    NCH = 3

    def _direct(self):
        from lisatools.domains import WDMSettings
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

        table = SimpleNamespace(fdot_vals=np.array([-1.0, 1.0]), INTERP_METHOD="spline")
        return EMRIDirectWDM(None, table, WDMSettings(16, 64, 5.0, force_backend="cpu"), orbits=None,
                             tdi_config=SimpleNamespace(nchannels=self.NCH), t_start=0.0, data_t0=100.0)

    @staticmethod
    def _item(tag, n_sub, n_knots=3):
        """A dense-kernel item carrying ``tag`` (as psi), ``n_sub`` harmonics and ``n_knots`` knots."""
        return ((np.zeros(n_knots), None, np.zeros((n_sub, 3), dtype=np.int32), None, None),
                (float(tag), 0.0, 0.0))

    @staticmethod
    def _value(tag, t, sub=None, taper=0.0):
        """The stand-in kernel's channel ``(nch, len(t))``: template tag, channel, sample time,
        harmonic and taper all visible in the value."""
        v = tag * 1e6 + np.arange(DenseTailBatchTest.NCH)[:, None] * 1e3 + np.asarray(t)[None, :] + 1e-3 * taper
        return v if sub is None else v + 0.1 * sub

    def _kernel(self, calls, tags=None):
        def kernel(items, t_abs):
            t_abs = np.asarray(t_abs)
            calls.append(t_abs.shape)
            if tags is not None:
                tags.append([it[1][0] for it in items])

            class _Dense:
                def channels(inner, par, sum_subs=False, stop_taper=0.0):
                    rows = []
                    for b, it in enumerate(items):
                        tag, ns = it[1][0], it[0][2].shape[0]
                        if sum_subs:
                            rows.append(self._value(tag, t_abs[b], taper=stop_taper))
                        else:
                            rows += [self._value(tag, t_abs[b], s, stop_taper) for s in range(ns)]
                    return np.stack(rows) + 7j                  # the real part is the channel
            return _Dense(), np.zeros((len(items), 4)), None, None, None
        return kernel

    def test_items_with_their_own_grids_split_per_item_in_blocks_of_items(self):
        import os

        d = self._direct()
        N = 40
        items = [self._item(k + 1, 2 + k) for k in range(5)]          # 2..6 harmonics, 3 knots each
        t_abs = np.stack([1000.0 * k + np.arange(N) for k in range(5)])
        # the largest two-item block: items 3 and 4, outputs 2 x nch x N plus 8 x (5 + 6) x 3 inputs;
        # no three consecutive items fit under it
        two = 2 * self.NCH * N + 8 * (5 + 6) * 3
        for budget, n_calls in ((two, 3), (2e7, 1), (1, 5)):          # 2 items, all, 1 (at least one)
            calls = []
            with mock.patch.dict(os.environ, {"EMRI_DIRECT_TAIL_BLOCK_ELEMS": str(budget)}), \
                    mock.patch.object(d, "_dense_kernel", side_effect=self._kernel(calls)):
                out = np.asarray(d._dense_td(items, t_abs, sum_subs=True, stop_taper=120.0))
            self.assertEqual(out.shape, (5, self.NCH, N))
            self.assertEqual(len(calls), n_calls, budget)
            self.assertTrue(all(c[1] == N for c in calls))            # whole grids per item
            for k in range(5):
                np.testing.assert_array_equal(out[k], self._value(k + 1, t_abs[k], taper=120.0))

    def test_a_bare_item_and_a_shared_grid_keep_the_single_template_form(self):
        d = self._direct()
        ts = 500.0 + np.arange(30)
        calls = []
        with mock.patch.object(d, "_dense_kernel", side_effect=self._kernel(calls)):
            per = np.asarray(d._dense_td(self._item(4, 3), ts))
            summed = np.asarray(d._dense_td(self._item(4, 3), ts, sum_subs=True))
        self.assertEqual(per.shape, (3, self.NCH, 30))              # (num_sub, nch, n)
        self.assertEqual(summed.shape, (self.NCH, 30))              # (nch, n)
        for s in range(3):
            np.testing.assert_array_equal(per[s], self._value(4, ts, s))
        np.testing.assert_array_equal(summed, self._value(4, ts))

    def test_dense_inputs_count_in_the_block(self):
        """The block budget counts each item's dense amplitude inputs (8 x harmonics x the BLOCK's
        largest knot count) next to the outputs: with large S x K fewer items share a block."""
        import os

        d = self._direct()
        N = 40
        t_abs = np.stack([1000.0 * k + np.arange(N) for k in range(4)])
        out_all = 4 * self.NCH * N
        cases = (
            ([3, 3, 3, 3], out_all + 8 * 8 * 3, [4]),         # small inputs: all four in one block
            ([100, 100, 100, 100], out_all + 8 * 8 * 3, [1, 1, 1, 1]),   # 1600 inputs each: one per block
            # item 1's 50 knots pad its neighbours too (K of the block), so 0 and 1 cannot pair up;
            # 2 and 3 (3 knots) still do
            ([3, 50, 3, 3], 1000, [1, 1, 2]),
        )
        for knots, budget, per_call in cases:
            items = [self._item(k + 1, 2, n_knots=nk) for k, nk in enumerate(knots)]
            calls = []
            with mock.patch.dict(os.environ, {"EMRI_DIRECT_TAIL_BLOCK_ELEMS": str(budget)}), \
                    mock.patch.object(d, "_dense_kernel", side_effect=self._kernel(calls)):
                out = np.asarray(d._dense_td(items, t_abs, sum_subs=True))
            self.assertEqual([c[0] for c in calls], per_call, knots)
            for k in range(4):
                np.testing.assert_array_equal(out[k], self._value(k + 1, t_abs[k]))

    def test_grids_of_their_own_lengths_go_to_blocks_of_one_length(self):
        """A list of grids of different lengths (the tail's per-row items): blocks hold items of
        one length, each item's channels come back on its own grid (a list, or through emit)."""
        d = self._direct()
        items = [self._item(k + 1, 2) for k in range(4)]
        grids = [100.0 * k + np.arange(n) for k, n in enumerate((20, 40, 20, 40))]
        calls, tags = [], []
        with mock.patch.object(d, "_dense_kernel", side_effect=self._kernel(calls, tags)):
            out = d._dense_td(items, grids, sum_subs=True, stop_taper=120.0)
            got = {}
            res = d._dense_td(items, grids, sum_subs=True, stop_taper=120.0,
                              emit=lambda b, arr: got.__setitem__(b, np.array(arr)))
        self.assertIsNone(res)
        self.assertEqual(calls[:2], [(2, 20), (2, 40)])               # one kernel per length
        self.assertEqual(tags[:2], [[1.0, 3.0], [2.0, 4.0]])
        for k in range(4):
            np.testing.assert_array_equal(np.asarray(out[k]), self._value(k + 1, grids[k], taper=120.0))
            np.testing.assert_array_equal(got[k], self._value(k + 1, grids[k], taper=120.0))

    def _tail_values(self, d, pairs, tags):
        """What the tail must return: each pair's row value on that pair's own window grid."""
        return np.stack([self._value(tags[r], 100.0 + (n0 * 16 + np.arange(16 * 8)) * 5.0, taper=d.plunge_taper_s)
                         for r, n0, _, _ in pairs])

    def test_tail_fn_gives_each_row_one_item_with_its_windows_concatenated(self):
        d = self._direct()
        chunk_items = {3: self._item(30, 2), 5: self._item(50, 4), 6: self._item(60, 1)}
        tail = d._dense_tail_fn(chunk_items, 100.0, 16, 8, 5.0)
        self.assertTrue(tail.summed)
        self.assertEqual(tail.taper, d.plunge_taper_s)
        # row 3: one window; rows 5 and 6: two windows each
        pairs = [(3, 10, 2, 6), (5, 10, 2, 6), (5, 16, 2, 8), (6, 12, 2, 6), (6, 18, 2, 8)]
        calls, tags = [], []
        with mock.patch.object(d, "_dense_td", wraps=d._dense_td) as spy, \
                mock.patch.object(d, "_dense_kernel", side_effect=self._kernel(calls, tags)):
            out = np.asarray(tail(pairs))
        self.assertEqual((spy.call_count, tail.calls), (1, 1))       # ONE dense call for every pair
        items, grids = spy.call_args[0][:2]
        self.assertEqual([it[1][0] for it in items], [30.0, 50.0, 60.0])   # one item per ROW
        want_grid = {r: np.concatenate([100.0 + (n0 * 16 + np.arange(128)) * 5.0 for rr, n0, _, _ in pairs if rr == r])
                     for r in (3, 5, 6)}
        for it, g in zip(items, grids):
            np.testing.assert_array_equal(np.asarray(g), want_grid[int(it[1][0]) // 10])
        # blocks by grid length: row 3 (128 samples) alone, rows 5 and 6 (256 each) together; no
        # row's item is ever repeated per window
        self.assertEqual(sorted(calls), [(1, 128), (2, 256)])
        self.assertEqual(sorted(t for blk in tags for t in blk), [30.0, 50.0, 60.0])
        self.assertEqual(out.shape, (5, self.NCH, 128))
        np.testing.assert_array_equal(out, self._tail_values(d, pairs, {3: 30, 5: 50, 6: 60}))

    def test_tail_fn_cuts_a_row_into_runs_only_when_its_windows_exceed_the_budget(self):
        import os

        d = self._direct()
        chunk_items = {2: self._item(20, 4), 4: self._item(40, 1)}
        tail = d._dense_tail_fn(chunk_items, 100.0, 16, 8, 5.0)
        pairs = [(2, 10, 2, 6), (2, 16, 2, 6), (2, 22, 2, 8), (4, 30, 2, 8)]
        # two windows of row 2 (4 harmonics, 3 knots) fit, three do not
        budget = 2 * self.NCH * 128 + 8 * 4 * 3
        calls, tags = [], []
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_TAIL_BLOCK_ELEMS": str(budget)}), \
                mock.patch.object(d, "_dense_kernel", side_effect=self._kernel(calls, tags)):
            out = np.asarray(tail(pairs))
        self.assertEqual(sorted(calls), [(1, 128), (1, 128), (1, 256)])   # row 2: runs of 2 + 1 windows
        self.assertEqual(sorted(t for blk in tags for t in blk), [20.0, 20.0, 40.0])
        np.testing.assert_array_equal(out, self._tail_values(d, pairs, {2: 20, 4: 40}))


class DenseChunkRowsTest(unittest.TestCase):
    """The production branch of EMRIDirectWDM._chunk_rows_batch (response="dense") with the
    compiled pieces stood in (response, fused lookup, dense kernel): ONE response for every chunk
    row (each on its own grid), ONE lookup over the common pixels with each harmonic stopping at
    its ROW's handoff, ONE dense evaluation for every (row, window) pair (one kernel item per row,
    its windows' grids end to end) and ONE transform over the pairs; the response is released once
    the lookup has run, before the tail; each row's tail runs from its own handoff to a quarter
    chunk past its own last pixel; the other rows of the batch array are untouched. The stand-ins
    are plain functions (a Mock would keep the response alive in its call records)."""

    NF, NT, DT, T_START, T0_OFF = 16, 128, 10.0, 1.0e6, 5000.0     # layer_dt 160 s

    def test_one_response_one_lookup_one_tail_per_row_handoffs(self):
        from lisatools import wdm_het
        from lisatools.domains import WDMSettings
        from lisatools.sources.emri import wdm_direct as wd

        table = SimpleNamespace(fdot_vals=np.array([-1.0, 1.0]), INTERP_METHOD="spline")
        d = wd.EMRIDirectWDM(None, table, WDMSettings(self.NF, self.NT, self.DT, force_backend="cpu"), orbits=None,
                             tdi_config=SimpleNamespace(nchannels=3), t_start=self.T_START,
                             data_t0=self.T_START + self.T0_OFF, Nt_sub=32, response="dense")
        ldt = d.wdm.layer_dt
        # rows 1 and 3 of a 4-row batch plunge: (trajectory end [FEW clock], sub handoff pixels)
        spec = {1: (self.T0_OFF + 100 * ldt, (80, 90)), 3: (self.T0_OFF + 112.5 * ldt, (95, 100, None))}
        chunk_items, grids = {}, {}
        for r, (t_end, n_hs) in spec.items():
            n_tr = np.arange(8, int((t_end - self.T0_OFF) // ldt) + 1)        # pixels inside the trajectory
            tn = self.T0_OFF + n_tr * ldt
            z = np.zeros_like(tn)
            tracks = [wd.HarmonicTrack((2, 2, 0, 0), tn, z + 1, z, z + 1e-3, z,
                                       z if n_h is None else np.where(n_tr >= n_h, 1.0, 0.0)) for n_h in n_hs]
            grids[r] = np.linspace(1000.0, t_end - 600.0, 30)                 # its own response grid
            chunk_items[r] = dict(t_end=t_end, tracks=tracks, track_n=n_tr, sky=(0.1 * r, 1.0, 0.2),
                                  din=(None, None, np.zeros((len(n_hs), 3)), None, None), grid=grids[r])
        x_all = np.concatenate([np.tile(self.T_START + grids[r], (len(spec[r][1]), 1)) for r in (1, 3)])
        seen = dict(resp=[], lookup=[], kernel=[], resp_alive=[])

        class _Resp:                                         # weakref-able, like the real output
            def __init__(self, x):
                self.x = x

        def response(items, t_grid):
            seen["resp"].append((items, t_grid))
            resp = _Resp(x_all)
            seen["resp_ref"] = weakref.ref(resp)
            return resp

        def lookup_fn(be, out, acc4, sub_row, n_ok, m_lo):
            def run(n_stop):                                 # holds the response, as the real closure
                seen["lookup"].append((np.asarray(sub_row), np.asarray(n_ok), np.asarray(n_stop), out.x.shape))
                return dict(lookup_pixels=1, dropped_pixels=0)
            return run

        def kernel(items, t_abs):
            seen["kernel"].append(([it[1][0] for it in items], np.asarray(t_abs)))
            seen["resp_alive"].append(seen["resp_ref"]() is not None)
            vals = np.stack([np.cos(1e-3 * (k + 1) * np.asarray(t_abs[k]))[None, :].repeat(3, 0)
                             for k in range(len(items))])

            class _Dense:
                def channels(self, par, sum_subs=False, stop_taper=0.0):
                    return vals + 0j
            return _Dense(), np.zeros((len(items), 4)), None, None, None

        out_arr = np.zeros((4, 3, self.NF, self.NT))
        with mock.patch.object(d, "_dense_response", new=response), \
                mock.patch.object(d, "_kernel_backend", new=lambda out: "be"), \
                mock.patch.object(d, "_lookup_fn", new=lookup_fn), \
                mock.patch.object(d, "_dense_kernel", new=kernel), \
                mock.patch("lisatools.wdm_het.wdm_chunk_of_td", wraps=wdm_het.wdm_chunk_of_td) as tr:
            st = d._chunk_rows_batch(out_arr, chunk_items, 0, self.NF, {})
        # ONE response: every row's item and own grid, in row order
        self.assertEqual(len(seen["resp"]), 1)
        items, t_grid = seen["resp"][0]
        self.assertEqual([it[1] for it in items], [chunk_items[1]["sky"], chunk_items[3]["sky"]])
        self.assertEqual(len(t_grid), 2)
        np.testing.assert_array_equal(t_grid[1], grids[3])
        # ONE lookup: the common pixels (to the later row's own end), each harmonic at its row's handoff
        self.assertEqual(len(seen["lookup"]), 1)
        sub_row, n_ok, n_stop, _ = seen["lookup"][0]
        np.testing.assert_array_equal(sub_row, [1, 1, 3, 3, 3])
        last = {1: 96, 3: 108}                               # last pixel before each row's response end
        np.testing.assert_array_equal(n_ok, np.arange(8, last[3] + 1))
        np.testing.assert_array_equal(n_stop, [80, 80, 95, 95, 95])
        # ONE dense evaluation for every (row, window) pair -- one kernel item per ROW, its two
        # windows' grids end to end -- and ONE transform over the four pairs
        self.assertEqual(len(seen["kernel"]), 1)
        psis, t_abs = seen["kernel"][0]
        self.assertEqual(psis, [0.1, 0.1 * 3])                        # one item per row, row order
        self.assertEqual(tr.call_count, 1)
        self.assertEqual(tr.call_args[0][0].shape[0], 4)
        self.assertEqual(st["windows"], 4)
        N = self.NF * 32
        self.assertEqual(t_abs.shape, (2, 2 * N))
        win = t_abs.reshape(2, 2, N)                                  # (row, window, sample)
        n0s = np.rint((win[:, :, 0] - d.data_t0) / (self.NF * self.DT)).astype(int)
        np.testing.assert_array_equal(win, d.data_t0 + (n0s[:, :, None] * self.NF + np.arange(N)) * self.DT)
        self.assertTrue(np.all(np.diff(n0s, axis=1) > 0))
        # the response was released before the tail ran (its only reader, the lookup, had run)
        self.assertEqual(seen["resp_alive"], [False])
        # each row's tail: from its own handoff to a quarter chunk past its own last pixel
        self.assertEqual(st["tail_layers"], [last[1] + 1 + 8 - 80, last[3] + 1 + 8 - 95])
        self.assertEqual((st["rows"], st["lookup"], st["n_response"]), (2, "kernel", 30))
        self.assertTrue(np.all(out_arr[[0, 2]] == 0.0))
        self.assertTrue(np.any(out_arr[1] != 0.0) and np.any(out_arr[3] != 0.0))


if __name__ == "__main__":
    unittest.main()
