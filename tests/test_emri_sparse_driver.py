"""EMRISparseResponse: the domain-independent driver of the sparse EMRI front-end.

Synthetic dense inputs (exactly representable quadratic phases), so neither FEW nor data files
are needed. Mirrors the EMRIDirectWDM response-grid / exact-channel tests of ``dev``
(tests/test_emri_sparse_response.py there), for the re-packaged driver.
"""
import unittest
from types import SimpleNamespace

import numpy as np

DAY = 86400.0
T_START = 1.0e7


def _quadratic_coeffs(t_k, phi0, f0, fdot):
    """DOPR853 nested coefficients (K-1, 3, 8) of Phi(t) = phi0 + 2 pi (f0 tau + fdot tau^2 / 2)
    per fundamental (EXACT: the nested basis spans degree 7), tau = t - t_k[0]."""
    from lisatools.sources.emri.sparse_tracks import _dense_basis

    x = 0.5 * (1 - np.cos(np.pi * (np.arange(16) + 0.5) / 16))
    B = _dense_basis(x)
    C = np.zeros((t_k.size - 1, 3, 8))
    for q in range(t_k.size - 1):
        tau = t_k[q] - t_k[0] + x * (t_k[q + 1] - t_k[q])
        vals = phi0[None, :] + 2 * np.pi * (f0[None, :] * tau[:, None] + 0.5 * fdot[None, :] * tau[:, None] ** 2)
        C[q] = np.linalg.lstsq(B, vals, rcond=None)[0].T
    return C


def _driver(**kw):
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri.sparse_tracks import EMRISparseResponse

    args = dict(orbits=EqualArmlengthOrbits(force_backend="cpu"),
                tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
                t_start=T_START, t_window=(T_START, T_START + 20480.0), data_dt=10.0)
    args.update(kw)
    return EMRISparseResponse(None, **args)


def _item(t_end, sky=(0.3, 1.0, 0.2)):
    t_k = np.linspace(0.0, t_end, 9)                                       # FEW clock
    C = _quadratic_coeffs(t_k, np.array([0.1, 0.0, 0.2]), np.array([2e-3, 0.0, 4e-4]),
                          np.array([1e-10, 0.0, 0.0]))
    are = np.zeros((2, t_k.size - 1, 4))
    are[:, :, 0] = 1.0
    return (t_k, C, np.array([[2, 0, 0], [3, 0, 1]]), are, np.zeros_like(are)), sky


class SparseDriverGridTest(unittest.TestCase):
    def test_per_template_grids_are_padded_inside_their_span(self):
        from lisatools.sources.emri.sparse_tracks import ExactPhaseTDIOutput

        items = [_item(8 * 6 * 3600.0), _item(8 * 6 * 3600.0, sky=(0.6, 2.0, -0.4))]
        t_k = items[0][0][0]
        g1 = np.linspace(t_k[0] + 600, t_k[-1] - 600, 11)
        g2 = np.linspace(t_k[0] + 600, t_k[-1] - 600, 7)
        out = _driver().tracks(items, [g1, g2])
        self.assertIsInstance(out, ExactPhaseTDIOutput)
        x = np.asarray(out.x)
        self.assertEqual(x.shape, (4, 11))                                 # 2 templates x 2 harmonics
        np.testing.assert_allclose(x[0], T_START + g1)
        self.assertEqual((x[2, 0], x[2, -1]), (T_START + g2[0], T_START + g2[-1]))
        self.assertTrue(set((T_START + g2).tolist()) <= set(x[2].tolist()))

    def test_tracer_shapes_and_positive_frequency(self):
        item = _item(8 * 6 * 3600.0)
        t_k = item[0][0]
        out = _driver().tracks([item], np.linspace(t_k[0] + 600, 30000.0, 400))
        amp, phase, f, fdot = (np.asarray(v) for v in out.eval_tracer(T_START + np.linspace(2000.0, 20000.0, 50)))
        for v in (amp, phase, f, fdot):
            self.assertEqual(v.shape, (2, 3, 50))                          # (harmonic, channel, time)
        # harmonic (2,0,0): f = 2 * 2e-3 Hz up to the Doppler shift; (3,0,1): 3 * 2e-3 + 4e-4
        np.testing.assert_allclose(f[0].mean(), 4e-3, rtol=1e-3)
        np.testing.assert_allclose(f[1].mean(), 6.4e-3, rtol=1e-3)

    def test_sparse_grid_stays_inside_the_orbit_tables(self):
        d = _driver(t_window=(T_START, T_START + 8 * DAY))
        d.orbits = SimpleNamespace(pycppdetector_args=[T_START + 7200.0, 500.0, 2000, T_START - 1e5, 2.5, 233000])
        d._orbit_span_cache = None
        H = SimpleNamespace(t_arr=np.array([0.0, 3 * DAY, 9 * DAY]))
        g = d.sparse_grid(H)
        o_hi = min(T_START + 7200.0 + 1999 * 500.0, T_START - 1e5 + 232999 * 2.5)
        self.assertGreaterEqual(g[0], 7200.0 + 600.0)
        self.assertLessEqual(g[-1], o_hi - T_START - 600.0)
        self.assertLessEqual(np.diff(g).max(), d.sparse_dt + 1e-9)

    def test_too_short_a_grid_raises_the_domain_error(self):
        from lisatools.utils.exceptions import WaveformDomainError

        with self.assertRaises(WaveformDomainError):
            _driver().tracks([_item(8 * 6 * 3600.0)], np.array([600.0, 601.0]))

    def test_ends_before_window(self):
        d = _driver(t_window=(T_START + 5 * DAY, T_START + 6 * DAY))
        self.assertTrue(d.ends_before_window(SimpleNamespace(t_arr=np.array([0.0, DAY]))))
        self.assertFalse(d.ends_before_window(SimpleNamespace(t_arr=np.array([0.0, 5.5 * DAY]))))


class SparseDriverExactChannelsTest(unittest.TestCase):
    def test_exact_channels_equal_the_tracks_away_from_stops(self):
        d = _driver()
        item = _item(8 * 6 * 3600.0)                                       # stops long after the window
        t_k = item[0][0]
        out = d.tracks([item], np.linspace(t_k[0] + 600, 30000.0, 400))
        ts = T_START + np.linspace(2000.0, 20000.0, 300)
        exact = np.asarray(d.td_channels(item, ts))
        spl = np.asarray(out.eval_tdi(ts))
        self.assertLess(np.abs(exact - spl).max() / np.abs(spl).max(), 1e-5)

    def test_summed_channels_equal_the_per_harmonic_sum(self):
        d = _driver()
        item = _item(8 * 6 * 3600.0)
        ts = T_START + np.arange(0.0, 20480.0, 10.0)
        per = np.asarray(d.td_channels(item, ts))
        summed = np.asarray(d.td_channels(item, ts, sum_subs=True))
        self.assertEqual(summed.shape, per.shape[1:])
        np.testing.assert_allclose(summed, per.sum(axis=0), rtol=0, atol=1e-13 * np.abs(per).max())

    def test_stop_taper_only_changes_the_end(self):
        d = _driver()
        t_end = 15000.0
        item = _item(t_end)                                                # stops inside the window
        ts = T_START + np.arange(0.0, 20480.0, 10.0)
        raw = np.asarray(d.td_channels(item, ts, sum_subs=True))
        tap = np.asarray(d.td_channels(item, ts, sum_subs=True, stop_taper=120.0))
        early = ts < T_START + t_end - 120.0 - 700.0   # emission time: SSB (<= ~500 s) + TDI delays
        np.testing.assert_allclose(tap[:, early], raw[:, early], rtol=0, atol=1e-13 * np.abs(raw).max())
        self.assertGreater(np.abs(tap - raw).max(), 0.0)
        late = ts > T_START + t_end + 700.0                                # every delayed term has stopped
        self.assertEqual(np.abs(tap[:, late]).max(), 0.0)


def _mismatch(a, b):
    return 1 - np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)


def _stft_reference(d, item, seg, T_seg, dt, alpha, stop_taper=120.0):
    from lisatools.stft_tracks import periodic_tukey

    nper = int(round(T_seg / dt))
    ts = (seg[:, None] + np.arange(nper) * dt).ravel()
    x = np.asarray(d.td_channels(item, ts, sum_subs=True, stop_taper=stop_taper)).reshape(3, seg.size, nper)
    return dt * np.fft.rfft(x * periodic_tukey(nper, alpha), axis=-1)


class TrackSourceTest(unittest.TestCase):
    """EMRITrackSource + the source-agnostic STFT driver on synthetic templates: one that
    covers the window, one that stops inside it, one that ends before it."""

    def test_batch_tracks_dense_and_zero_rows(self):
        from lisatools.sources.emri.sparse_tracks import EMRITrackSource
        from lisatools.stft_tracks import SparseTrackSource, STFTTrackAccumulator, stft_templates_from_tracks

        T_seg, dt, alpha = 14400.0, 10.0, 0.3
        w_lo = T_START + 1 * DAY
        d = _driver(t_window=(w_lo, w_lo + 2 * DAY), data_dt=dt, sparse_dt=7200.0)
        ends = (4 * DAY, 2.3 * DAY, 0.5 * DAY)
        items = [_item(e) for e in ends]
        holders = [SimpleNamespace(t_arr=it[0][0]) for it in items]
        src = EMRITrackSource(d, items, holders)
        self.assertIsInstance(src, SparseTrackSource)
        self.assertEqual(src.num_rows, 3)
        np.testing.assert_array_equal(np.unique(src.component_row), [0, 1])     # row 2 has no track
        seg = w_lo + np.arange(12) * T_seg
        nper = int(T_seg / dt)
        acc = STFTTrackAccumulator(T_seg, 0.0, nper // 2 + 1, alpha)
        out, info = stft_templates_from_tracks(src, acc, seg, dt, n_side=40, return_info=True)
        mode = info["mode"]
        self.assertTrue(np.all(mode[0] == 1))
        self.assertTrue(np.all(mode[2] == 0) and np.all(out[2] == 0))
        stop = T_START + ends[1]
        self.assertTrue(np.all(mode[1][seg + T_seg <= stop - 2 * d.DELAY_MARGIN] == 1))
        self.assertTrue(np.all(mode[1][(seg < stop) & (seg + T_seg > stop)] == 2))
        self.assertTrue(np.all(mode[1][seg >= stop + d.DELAY_MARGIN] == 0))
        for r in (0, 1):
            ref = _stft_reference(d, items[r], seg, T_seg, dt, alpha)
            self.assertLess(_mismatch(out[r], ref), 1e-6, f"row {r}")
            dense = mode[r] == 2
            np.testing.assert_allclose(out[r][:, dense], ref[:, dense], rtol=1e-12, atol=1e-12 * np.abs(ref).max())
            # what the driver calls zero really is zero in the exact series
            self.assertLess(np.abs(ref[:, mode[r] == 0]).max(initial=0.0), 1e-12 * np.abs(ref).max())

    def test_too_short_a_trajectory_is_dense(self):
        from lisatools.sources.emri.sparse_tracks import EMRITrackSource
        from lisatools.stft_tracks import STFTTrackAccumulator, stft_templates_from_tracks

        T_seg, dt = 2560.0, 10.0
        d = _driver()                                              # window: 20480 s from T_START
        item = _item(1300.0)                                       # shorter than two delay margins
        src = EMRITrackSource(d, [item], [SimpleNamespace(t_arr=item[0][0])])
        self.assertEqual(src.component_row.size, 0)
        seg = T_START + np.arange(8) * T_seg
        nper = int(T_seg / dt)
        out, info = stft_templates_from_tracks(src, STFTTrackAccumulator(T_seg, 0.0, nper // 2 + 1, 0.3),
                                               seg, dt, return_info=True)
        np.testing.assert_array_equal(info["mode"][0], [2] + [0] * 7)
        ref = _stft_reference(d, item, seg, T_seg, dt, 0.3)
        np.testing.assert_allclose(out[0], ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


class SparseDriverRealFEWTest(unittest.TestCase):
    """End to end on a real FEW Kerr trajectory: FEW sparse holder -> dense inputs -> sparse
    response -> tracks; the tracks summed over harmonics reproduce the exact channels.
    Skips without FEW (``gpu_backend``: ``return_sparse_holder``) or its data files."""

    @classmethod
    def setUpClass(cls):
        try:
            from few.waveform import FastKerrEccentricEquatorialFlux

            cls.gen = FastKerrEccentricEquatorialFlux(
                inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4)},
                sum_kwargs={"pad_output": True}, force_backend="cpu")
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"FEW unavailable: {exc}")

    def test_tracks_reproduce_the_exact_channels(self):
        T0 = 2.0e7
        d = _driver(t_start=T0, t_window=(T0, T0 + 30 * DAY), data_dt=5.0)
        d.few_gen = self.gen
        saved = dict(self.gen.inspiral_kwargs)
        #           m1   m2    a    p0   e0   x0   dist qS   phiS qK   phiK Pp   Pt   Pr
        few_args = (1e6, 10.0, 0.9, 9.5, 0.3, 1.0, 1.0, 0.8, 1.1, 0.5, 2.0, 0.3, 0.0, 1.2)
        item, H = d.template_inputs(few_args, lam=1.1, beta=np.pi / 2 - 0.8, mode_selection_threshold=1e-2)
        self.assertEqual(dict(self.gen.inspiral_kwargs), saved)           # no leaked call-time kwargs
        self.assertFalse(d.ends_before_window(H))
        S = item[0][2].shape[0]
        self.assertEqual(S, 2 * np.asarray(H.ms).size - int(np.sum(np.asarray(H.ms) == 0)))
        g = d.sparse_grid(H)
        self.assertLess(g.size, 100)                                        # sparse: ~2 points per day
        out = d.tracks([item], g)
        ts = T0 + 5 * DAY + np.arange(2048) * 5.0
        exact = np.asarray(d.td_channels(item, ts, sum_subs=True))
        spl = np.real(np.asarray(out.eval_tdi(ts))).sum(axis=0)
        self.assertLess(np.abs(exact - spl).max() / np.abs(exact).max(), 1e-7)
        amp, phase, f, fdot = (np.asarray(v) for v in out.eval_tracer(T0 + (np.arange(40, 80) + 0.5) * 21600.0))
        self.assertEqual(amp.shape, (S, 3, 40))
        self.assertTrue(np.all(f > 0))                                      # -m partners mirrored

    def test_stft_template_of_a_plunging_source(self):
        """FEW -> sparse response -> EMRITrackSource -> STFT template, through the plunge: track
        segments, dense segments and zero segments against the dense STFT of the exact channels."""
        from lisatools.sources.emri.sparse_tracks import EMRITrackSource
        from lisatools.stft_tracks import STFTTrackAccumulator, stft_templates_from_tracks

        T0, T_seg, dt, alpha = 2.0e7, 21600.0, 10.0, 0.5
        w_lo = T0 + 26 * DAY
        d = _driver(t_start=T0 - 2 * DAY, t_window=(w_lo, w_lo + 8 * DAY), data_dt=dt)
        d.few_gen = self.gen
        few_args = (1e6, 100.0, 0.9, 7.0, 0.4, 1.0, 1.0, 0.8, 1.1, 0.5, 2.0, 0.3, 0.0, 1.2)   # ends ~day 32.4
        item, H = d.template_inputs(few_args, lam=1.1, beta=np.pi / 2 - 0.8, mode_selection_threshold=1e-2)
        src = EMRITrackSource(d, [item], [H])
        seg = w_lo + np.arange(32) * T_seg
        nper = int(T_seg / dt)
        out, info = stft_templates_from_tracks(src, STFTTrackAccumulator(T_seg, 0.0, nper // 2 + 1, alpha),
                                               seg, dt, cubic_tol=0.03, return_info=True)
        mode = info["mode"][0]
        self.assertTrue(set(mode.tolist()) == {0, 1, 2})
        self.assertTrue(np.all(np.diff(np.where(mode == 1, 0, np.where(mode == 2, 1, 2))) >= 0))
        ref = _stft_reference(d, item, seg, T_seg, dt, alpha)
        self.assertLess(_mismatch(out[0], ref), 1e-5)
        tr = mode == 1
        self.assertLess(_mismatch(out[0][:, tr], ref[:, tr]), 5e-5)
        self.assertLess(np.abs(ref[:, mode == 0]).max(), 1e-9 * np.abs(ref).max())


if __name__ == "__main__":
    unittest.main()
