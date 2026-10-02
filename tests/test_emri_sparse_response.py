"""The dense TDI response on a SPARSE grid (12 h knots) + the exact dense-output carrier
(ExactPhaseTDIOutput) == the same response on a 60 s grid with the whole phase splined, at the
pixel-centre quantities the lookup reads (amp, phase, f, fdot) and on the channel signal.

Paired negative control: splining the whole channel phase over the sparse grid (the original
construction on a coarse grid) misses by orders of magnitude -- the carrier is what lets the
response leave the pixel grid."""
import unittest

import numpy as np

DAY = 86400.0


def _quadratic_coeffs(t_k, phi0, f0, fdot):
    """DOPR853 nested coefficients (K-1, 3, 8) of Phi(t) = phi0 + 2 pi (f0 tau + fdot tau^2 / 2)
    per fundamental (EXACT: the nested basis spans degree 7), tau = t - t_k[0]."""
    from lisatools.sources.emri.wdm_direct import _dense_basis

    x = 0.5 * (1 - np.cos(np.pi * (np.arange(16) + 0.5) / 16))
    B = _dense_basis(x)
    C = np.zeros((t_k.size - 1, 3, 8))
    for q in range(t_k.size - 1):
        tau = t_k[q] - t_k[0] + x * (t_k[q + 1] - t_k[q])
        vals = phi0[None, :] + 2 * np.pi * (f0[None, :] * tau[:, None] + 0.5 * fdot[None, :] * tau[:, None] ** 2)
        C[q] = np.linalg.lstsq(B, vals, rcond=None)[0].T
    return C


def _chirp_coeffs(t_k, f0, tau_c):
    """Nested coefficients of a power-law chirp f = f0 (1 - tau/tau_c)^(-3/8) on Phi_phi (all
    polynomial orders: a cubic spline of the whole phase is NOT exact), least-squares per segment."""
    from lisatools.sources.emri.wdm_direct import _dense_basis

    x = 0.5 * (1 - np.cos(np.pi * (np.arange(16) + 0.5) / 16))
    B = _dense_basis(x)
    C = np.zeros((t_k.size - 1, 3, 8))
    for q in range(t_k.size - 1):
        tau = t_k[q] - t_k[0] + x * (t_k[q + 1] - t_k[q])
        phi = 2 * np.pi * f0 * tau_c * 1.6 * (1 - (1 - tau / tau_c) ** 0.625)
        vals = np.stack([phi, np.zeros_like(tau), 0.3 * phi], axis=1)
        C[q] = np.linalg.lstsq(B, vals, rcond=None)[0].T
    return C


class SparseResponseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.response.tdionfly import TDDenseTDIonTheFly
        from lisatools.sources.emri.wdm_direct import ExactPhaseTDIOutput

        orbits = EqualArmlengthOrbits(force_backend="cpu")
        tdi = TDIConfig("2nd generation", force_backend="cpu")
        t0 = 2.0e7
        cls.t_k = t0 + np.arange(13) * 12 * 3600.0                     # 6 days, 12 h knots
        cls.phi0 = np.array([0.3, 0.0, 1.1])
        f0 = np.array([3.1e-3, 0.0, 0.47e-3])                          # fundamental frequencies [Hz]
        fdot = np.array([2e-10, 0.0, 3e-11])
        cls.C = _quadratic_coeffs(cls.t_k, cls.phi0, f0, fdot)
        cls.mkn = np.array([[2, 0, 0], [3, 0, 1], [-2, 0, -1]])
        K, S = cls.t_k.size, cls.mkn.shape[0]
        are = np.zeros((S, K - 1, 4))
        aim = np.zeros((S, K - 1, 4))
        # a slowly varying, CONTINUOUS complex amplitude (monomials in t - t_knot per segment,
        # as FEW's cubic amplitude splines are): c + slope (t - t_k[0])
        slope = 1e-7 * np.array([1.0, 2.0, -1.0])
        are[:, :, 0] = np.array([1.0, -0.3, 0.4])[:, None] + slope[:, None] * (cls.t_k[:-1] - t0)[None]
        aim[:, :, 0] = np.array([0.5, 0.2, -0.7])[:, None]
        are[:, :, 1] = slope[:, None]
        params = np.array([[0.7, 1.9, 4.0, -0.8]])
        args = (np.array([0, S]), cls.mkn, cls.t_k[None], np.array([K]), cls.C[None], are, aim)
        kw = dict(amp_factor=0.5, tdi_config=tdi, orbits=orbits, force_backend="cpu")

        t_fine = np.arange(cls.t_k[0], cls.t_k[-1] + 1.0, 60.0)
        cls.fine = TDDenseTDIonTheFly(t_fine[None], *args, **kw)(params)          # splined output
        from lisatools.sources.emri.wdm_direct import sparse_response_grid

        # the knots (12 h) kept where the response is complete: 600 s inside both ends
        t_sparse = sparse_response_grid(cls.t_k, cls.t_k[0], cls.t_k[-1], 43200.0, 600.0)
        cls.n_sparse = t_sparse.size
        dense_s = TDDenseTDIonTheFly(t_sparse[None], *args, **kw)
        raw = dense_s(params, return_spline=False)
        cls.sparse = ExactPhaseTDIOutput(raw, [cls.t_k], [cls.C], cls.mkn, dense_s.sub_temp_host)
        # the same, with the truncated trajectory-end point kept (the ringing control)
        dense_e = TDDenseTDIonTheFly(np.append(t_sparse, cls.t_k[-1])[None], *args, **kw)
        cls.sparse_endpoint = ExactPhaseTDIOutput(dense_e(params, return_spline=False), [cls.t_k],
                                                  [cls.C], cls.mkn, dense_e.sub_temp_host)
        cls.t_pix = np.linspace(cls.t_k[0] + DAY, cls.t_k[-1] - DAY, 97)

    def _tracer(self, out):
        from lisatools.sources.emri.wdm_direct import tracer_from_tof_output

        return [np.asarray(a) for a in tracer_from_tof_output(out, self.t_pix)]

    def test_pixel_quantities_match_the_fine_grid(self):
        amp_f, ph_f, f_f, fd_f = self._tracer(self.fine)
        amp_s, ph_s, f_s, fd_s = self._tracer(self.sparse)
        d_ph = np.abs(np.angle(np.exp(1j * (ph_s - ph_f)))).max()
        print(f"[sparse response] |dphase| {d_ph:.2e} rad, |df| {np.abs(f_s - f_f).max():.2e} Hz, "
              f"|dfdot| {np.abs(fd_s - fd_f).max():.2e} Hz/s, amp rel "
              f"{np.abs(amp_s / amp_f - 1).max():.2e}")
        self.assertLess(d_ph, 1e-6)
        self.assertLess(np.abs(f_s - f_f).max(), 1e-10)          # layer_df ~1.4e-4 Hz
        self.assertLess(np.abs(fd_s - fd_f).max(), 1e-13)        # fdot axis step ~4e-10 Hz/s
        np.testing.assert_allclose(amp_s, amp_f, rtol=1e-6, atol=0)

    def test_channel_signal_matches(self):
        t = np.sort(np.random.default_rng(4).uniform(self.t_pix[0], self.t_pix[-1], 400))
        a = np.asarray(self.sparse.eval_tdi(t))
        b = np.asarray(self.fine.eval_tdi(t))
        self.assertLess(np.abs(a - b).max() / np.abs(b).max(), 1e-6)

    def test_a_point_at_the_trajectory_end_rings_into_the_interior(self):
        """Why the grid stops a delay margin inside the trajectory (paired with the test above)."""
        amp_f, ph_f, _, _ = self._tracer(self.fine)
        amp_e, ph_e, _, _ = self._tracer(self.sparse_endpoint)
        self.assertGreater(np.abs(amp_e / amp_f - 1).max(), 1e-5)       # vs 1e-8 without the point

    def test_grid_helpers(self):
        from lisatools.sources.emri.wdm_direct import pad_grid, sparse_response_grid

        g = sparse_response_grid(np.array([0.0, 1e4, 5e4, 2e5]), -1e5, 3e5, 43200.0, 600.0)
        self.assertEqual((g[0], g[-1]), (600.0, 2e5 - 600.0))
        self.assertLessEqual(np.diff(g).max(), 43200.0)
        self.assertTrue({1e4, 5e4} <= set(g.tolist()))
        p = pad_grid(g, g.size + 5)
        self.assertEqual(p.size, g.size + 5)
        self.assertEqual((p[0], p[-1]), (g[0], g[-1]))          # stays inside the span
        self.assertTrue(set(g.tolist()) <= set(p.tolist()) and np.all(np.diff(p) > 0))

    def test_tracer_in_pixel_blocks_equals_one_pass(self):
        """The memory-bounded tracer (many small pixel blocks) == one pass, on both outputs."""
        from lisatools.sources.emri.wdm_direct import tracer_from_tof_output

        for out in (self.sparse, self.fine):
            one = tracer_from_tof_output(out, self.t_pix)
            blocks = tracer_from_tof_output(out, self.t_pix, max_elems=3 * 3 * 7)   # 7 pixels per block
            for a, b in zip(one, blocks):
                np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-14, atol=0)

    def test_carrier_equals_the_reference_evaluator(self):
        from lisatools.sources.emri.wdm_direct import dense_phase_eval

        car = np.asarray(self.sparse.carrier(self.t_pix))
        want = self.mkn @ dense_phase_eval(self.t_k, self.C, self.t_pix).T
        np.testing.assert_allclose(car, want, rtol=1e-13, atol=0)

    def test_carrier_is_held_outside_the_trajectory(self):
        t = np.array([self.t_k[-1], self.t_k[-1] + 5000.0, self.t_k[0] - 5000.0, self.t_k[0]])
        car = np.asarray(self.sparse.carrier(t))
        np.testing.assert_allclose(car[:, 1], car[:, 0], rtol=0, atol=0)
        np.testing.assert_allclose(car[:, 2], car[:, 3], rtol=0, atol=0)



class ChirpSparseResponseTest(unittest.TestCase):
    """A chirping source (3 -> 4 mHz in 6 days, all polynomial orders in the phase): on 6 h
    knots the exact carrier keeps the pixel phase at the fine grid's to ~1e-6 rad, while
    splining the whole phase over the same nodes misses by > 100x that. The carrier is what
    makes the response grid independent of the phase evolution."""

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.response.tdionfly import TDDenseTDIonTheFly
        from lisatools.sources.emri.wdm_direct import ExactPhaseTDIOutput, sparse_response_grid

        orbits = EqualArmlengthOrbits(force_backend="cpu")
        tdi = TDIConfig("2nd generation", force_backend="cpu")
        t_k = 2.0e7 + np.arange(25) * 6 * 3600.0
        C = _chirp_coeffs(t_k, 3e-3, 10 * DAY)
        mkn = np.array([[2, 0, 0], [3, 0, 1]])
        K, S = t_k.size, 2
        are = np.zeros((S, K - 1, 4))
        are[:, :, 0] = 1.0
        params = np.array([[0.4, 0.9, 2.0, 0.5]])
        args = (np.array([0, S]), mkn, t_k[None], np.array([K]), C[None], are, np.zeros_like(are))
        kw = dict(amp_factor=0.5, tdi_config=tdi, orbits=orbits, force_backend="cpu")
        fine = TDDenseTDIonTheFly(np.arange(t_k[0], t_k[-1] + 1.0, 60.0)[None], *args, **kw)(params)
        g = sparse_response_grid(t_k, t_k[0], t_k[-1], 43200.0, 600.0)
        dense_s = TDDenseTDIonTheFly(g[None], *args, **kw)
        naive = dense_s(params)
        exact = ExactPhaseTDIOutput(dense_s(params, return_spline=False), [t_k], [C], mkn, dense_s.sub_temp_host)
        from lisatools.sources.emri.wdm_direct import tracer_from_tof_output

        t_pix = np.linspace(t_k[0] + DAY, t_k[-1] - DAY, 97)
        cls.ph = {k: np.asarray(tracer_from_tof_output(o, t_pix)[1]) for k, o in
                  (("fine", fine), ("naive", naive), ("exact", exact))}

    def test_exact_carrier_beats_the_whole_phase_spline(self):
        err = {k: np.abs(np.angle(np.exp(1j * (self.ph[k] - self.ph["fine"])))).max() for k in ("naive", "exact")}
        print(f"[chirp sparse response] whole-phase spline {err['naive']:.2e} rad, exact carrier {err['exact']:.2e} rad")
        self.assertLess(err["exact"], 1e-5)
        self.assertGreater(err["naive"], 100 * err["exact"])



class DirectWDMResponseGridTest(unittest.TestCase):
    """EMRIDirectWDM._dense_response: per-template sparse grids (different lengths, padded
    inside their spans) -> ExactPhaseTDIOutput; response_grid="pixels" -> the splined output."""

    def _direct(self, grid):
        from types import SimpleNamespace

        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.domains import WDMSettings
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

        table = SimpleNamespace(fdot_vals=np.array([-1.0, 1.0]), INTERP_METHOD="spline")
        return EMRIDirectWDM(None, table, WDMSettings(32, 64, 10.0, force_backend="cpu"),
                             orbits=EqualArmlengthOrbits(force_backend="cpu"),
                             tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
                             t_start=1.0e7, data_t0=1.0e7, response="dense", response_grid=grid)

    def _items(self):
        t_k = np.arange(9) * 6 * 3600.0                                    # FEW clock
        C = _quadratic_coeffs(t_k, np.array([0.1, 0.0, 0.2]), np.array([2e-3, 0.0, 4e-4]),
                              np.array([1e-10, 0.0, 0.0]))
        are = np.zeros((2, t_k.size - 1, 4))
        are[:, :, 0] = 1.0
        mkn = np.array([[2, 0, 0], [3, 0, 1]])
        return [((t_k, C, mkn, are, np.zeros_like(are)), (0.3, 1.0, 0.2)),
                ((t_k, C, mkn, are, np.zeros_like(are)), (0.6, 2.0, -0.4))], t_k

    def test_sparse_per_template_grids(self):
        from lisatools.sources.emri.wdm_direct import ExactPhaseTDIOutput

        items, t_k = self._items()
        g1 = np.linspace(t_k[0] + 600, t_k[-1] - 600, 11)
        g2 = np.linspace(t_k[0] + 600, t_k[-1] - 600, 7)
        out = self._direct("sparse")._dense_response(items, [g1, g2])
        self.assertIsInstance(out, ExactPhaseTDIOutput)
        x = np.asarray(out.x)
        self.assertEqual(x.shape, (4, 11))                                  # 2 templates x 2 harmonics
        np.testing.assert_allclose(x[0], 1.0e7 + g1)
        self.assertEqual((x[2, 0], x[2, -1]), (1.0e7 + g2[0], 1.0e7 + g2[-1]))   # padded INSIDE
        self.assertTrue(set((1.0e7 + g2).tolist()) <= set(x[2].tolist()))

    def test_pixels_keeps_the_whole_phase_spline(self):
        from lisatools.sources.emri.wdm_direct import ExactPhaseTDIOutput, SplinedTDIOutput

        items, t_k = self._items()
        out = self._direct("pixels")._dense_response(items, np.linspace(t_k[0], t_k[-1], 50))
        self.assertIsInstance(out, SplinedTDIOutput)
        self.assertNotIsInstance(out, ExactPhaseTDIOutput)

    def test_mode_blocks_equal_one_pass(self):
        """The tracer over blocks of harmonics (the batch path's mode blocks) == all at once,
        for both output kinds, across a template boundary."""
        from lisatools.sources.emri.wdm_direct import tracer_from_tof_output

        items, t_k = self._items()
        t_pix = np.linspace(1.0e7 + t_k[0] + 3 * 3600, 1.0e7 + t_k[-1] - 3 * 3600, 23)
        for grid, g in (("sparse", [np.linspace(t_k[0] + 600, t_k[-1] - 600, 11)] * 2),
                        ("pixels", np.linspace(t_k[0], t_k[-1], 200))):
            out = self._direct(grid)._dense_response(items, g)
            full = tracer_from_tof_output(out, t_pix)
            for blocks in ([[0, 1], [2, 3]], [[0], [1, 2], [3]]):
                parts = [tracer_from_tof_output(out, t_pix, subs=np.array(b)) for b in blocks]
                for k in range(4):
                    got = np.concatenate([np.asarray(p[k]) for p in parts], axis=0)
                    np.testing.assert_allclose(got, np.asarray(full[k]), rtol=1e-14, atol=0, err_msg=grid)

    def test_sparse_grid_stays_inside_the_orbit_tables(self):
        """Outside the orbit tables the kernel zeroes the channel; a half-day spline across that
        edge rang 4e-4 rad into the first pixels of a laptop-trimmed L1 run."""
        from types import SimpleNamespace

        d = self._direct("sparse")
        t_s = d.t_start                                     # FEW clock origin (absolute)
        # tables: positions from t_s + 2 h, light travel times until t_s + 5 d
        d.orbits = SimpleNamespace(pycppdetector_args=[t_s + 7200.0, 500.0, 2000, t_s - 1e5, 2.5, 233000])
        d._orbit_span_cache = None
        H = SimpleNamespace(t_arr=np.array([0.0, 3 * DAY, 9 * DAY]))
        g = d._sparse_grid(H, None)
        o_hi = min(t_s + 7200.0 + 1999 * 500.0, t_s - 1e5 + 232999 * 2.5)
        self.assertGreaterEqual(g[0], 7200.0 + 600.0)
        self.assertLessEqual(g[-1], o_hi - t_s - 600.0)
        self.assertLessEqual(np.diff(g).max(), d.sparse_dt)

    def test_default_and_env(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMRI_DIRECT_RESPONSE_GRID", None)
            self.assertEqual(self._direct(None).response_grid, "sparse")
            os.environ["EMRI_DIRECT_RESPONSE_GRID"] = "pixels"
            self.assertEqual(self._direct(None).response_grid, "pixels")
            self.assertEqual(self._direct("sparse").response_grid, "sparse")   # explicit wins


if __name__ == "__main__":
    unittest.main()
