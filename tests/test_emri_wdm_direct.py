"""EMRI direct-to-WDM building blocks (Tasks A5, A7, A8).

A5: harmonic tracks (A, Phi, f, fdot, fddot) at pixel centres from a FEW sparse
holder, exact on a synthetic holder/integrator, with FEW's conventions pinned:
retrograde sign(xI0) on Phi_phi, the backwards-integration knot offset, and the
exact -m partner (-1)^l Y_{l,-m} conj(A) e^{+i Phi}.
"""

import unittest
from types import SimpleNamespace

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

    def test_all_lookup_matches_truth_too(self):
        acc, stats, n_ok = self._run(10 ** 6)          # never hands off
        self.assertEqual(stats["chunk_pixels"], 0)
        sl = slice(40, self.NT - 32)   # the (non-stopping) truth carries grid-end contamination
        rel = np.linalg.norm(acc[..., sl] - self.truth[..., sl]) / np.linalg.norm(self.truth[..., sl])
        self.assertLess(rel, 2e-3, f"rel L2 {rel:.2e}")


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


if __name__ == "__main__":
    unittest.main()
