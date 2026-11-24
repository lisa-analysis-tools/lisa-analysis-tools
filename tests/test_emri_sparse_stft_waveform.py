"""EMRISparseSTFTWaveform against the production dense EMRITDIWaveform (real FEW, CPU).

Same constructor arguments and call parameters; the STFT templates agree where the dense path
is complete (it zeroes its first ``buffer_time`` after ``waveform_t0``, so the waveform starts a
day before the data here). Skips without FEW ``gpu_backend`` or its data files.

Every FEW ``GenerateEMRIWaveform`` holds about 5.5 GB while alive, and each waveform here builds
one: the classes release theirs when done (:func:`_release`), or one test run holds several and
is killed on a 24 GB machine.
"""
import gc
import unittest

import numpy as np

DAY = 86400.0
T0 = 2.0e7
DT = 10.0
STFT_DT = 21600.0
#      m1   m2    a    p0   e0   x0   dist qK   phiK Pp   Pt   Pr   ra   dec
PAR = (1e6, 10.0, 0.9, 9.5, 0.3, 1.0, 1.0, 0.5, 2.0, 0.3, 0.0, 1.2, 1.1, 0.4)
KW = dict(mode_selection_threshold=1e-2)


def _release(obj, *names):
    """Drop the waveforms held in ``obj.<names>`` and collect (FEW's amplitude data is large)."""
    for name in names:
        if hasattr(obj, name):
            delattr(obj, name)
    gc.collect()


def _mismatch(a, b):
    return 1 - np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)


def _stft_of_td(wf, signal):
    """The data-side STFT (window, rfft, band) of a ``TDSignal`` placed on the data grid."""
    from lisatools.utils.utility import tukey

    nper = wf.nperseg
    nseg = wf.domain_settings.N // nper
    j0 = int(round((signal.t0 - wf.data_t0) / wf.dt))
    x = np.zeros((signal.arr.shape[0], wf.domain_settings.N))
    x[:, j0:j0 + signal.arr.shape[-1]] = np.asarray(signal.arr)
    spec = wf.dt * np.fft.rfft(
        x[:, :nseg * nper].reshape(-1, nseg, nper) * tukey(nper, alpha=wf.tukey_alpha, xp=np), axis=-1)
    freqs = np.fft.rfftfreq(nper, d=wf.dt)
    return spec[..., np.searchsorted(freqs, wf.freq_min):np.searchsorted(freqs, wf.freq_max, side="right")]


class SparseSTFTWaveformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from few.waveform import FastKerrEccentricEquatorialFlux  # noqa: F401
            from lisatools.detector import EqualArmlengthOrbits
            from lisatools.domains import TDSettings
            from lisatools.sources.emri.response import EMRI_INSPIRAL_KWARGS, EMRI_SUM_KWARGS
            from lisatools.sources.emri.waveform import EMRISparseSTFTWaveform, EMRITDIWaveform

            cls.common = dict(
                waveform_class="FastKerrEccentricEquatorialFlux",
                waveform_kwargs=dict(inspiral_kwargs=dict(EMRI_INSPIRAL_KWARGS), sum_kwargs=dict(EMRI_SUM_KWARGS)),
                waveform_t0=T0 - DAY, data_td_settings=TDSettings(N=int(6 * DAY / DT), dt=DT, t0=T0),
                sampling_frequency=1 / DT, tukey_alpha=0.5, stft_dt=STFT_DT, freq_min=1e-4, freq_max=3e-2,
                force_backend="cpu")
            cls.dense = EMRITDIWaveform(orbits=EqualArmlengthOrbits(force_backend="cpu"), **cls.common)
            cls.sparse = EMRISparseSTFTWaveform(orbits=EqualArmlengthOrbits(force_backend="cpu"), **cls.common)
            cls.sparse(*PAR, **KW)
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"FEW unavailable: {exc}")

    @classmethod
    def tearDownClass(cls):
        _release(cls, "dense", "sparse")

    def test_matches_the_dense_waveform(self):
        ref, ref_f, ref_t = self.dense(*PAR, **KW)
        out, out_f, out_t = self.sparse(*PAR, **KW)
        self.assertEqual(out.shape, ref.shape)
        np.testing.assert_allclose(out_f, ref_f, rtol=1e-13)
        np.testing.assert_allclose(out_t, ref_t, rtol=0, atol=1e-6)
        self.assertTrue(np.all(self.sparse.last_segment_mode == 1))
        self.assertLess(_mismatch(out, ref), 5e-6)
        worst = max(_mismatch(out[0][:, j], ref[0][:, j]) for j in range(ref.shape[2]))
        self.assertLess(worst, 1e-5)
        # the tracks model the symmetric Tukey window of the data (window_samples): no
        # amplitude offset of order 1 / samples per segment (4.6e-4 here) is left
        self.assertLess(abs(np.sqrt(np.vdot(out, out).real / np.vdot(ref, ref).real) - 1), 2e-5)

    def test_batch_and_keyword_sky(self):
        single = self.sparse(*PAR, **KW)[0]
        other = list(PAR)
        other[3] = 10.0
        batch = [np.array([a, b]) for a, b in zip(PAR, other)]
        out, start_f, start_t = self.sparse(*batch[:12], ra=batch[12], dec=batch[13], **KW)
        self.assertEqual(out.shape, (2,) + single.shape[1:])
        self.assertEqual(start_f.shape, (2,))
        np.testing.assert_allclose(out[0], single[0], rtol=1e-12, atol=1e-12 * np.abs(single).max())
        np.testing.assert_allclose(out[1], self.sparse(*other, **KW)[0][0], rtol=1e-12, atol=1e-12 * np.abs(single).max())
        self.assertGreater(_mismatch(out[0], out[1]), 0.1)

    def test_residual_signals_are_the_scored_templates(self):
        from lisatools.domains import DomainBaseArray, STFTSignal

        template = self.sparse(*PAR, **KW)[0]
        signal = self.sparse.get_signals_for_residuals(*PAR, **KW)
        self.assertIsInstance(signal, STFTSignal)
        np.testing.assert_array_equal(np.asarray(signal.arr), template[0])
        # on the STFT grid of the data, as the parent's transform of a full-length series
        nseg = self.sparse.domain_settings.N // self.sparse.nperseg
        grid = T0 + np.arange(nseg * self.sparse.nperseg) * DT
        self.assertEqual(signal.settings, self.sparse.get_output_settings(grid))
        # subtracting it from data equal to the scored template leaves nothing
        data = STFTSignal(template[0].copy(), signal.settings)
        self.assertFalse(np.any(np.asarray(data.subtract_signal(signal, copy=True).arr)))

        other = list(PAR)
        other[3] = 10.0
        batch = [np.array([a, b]) for a, b in zip(PAR, other)]
        signals = self.sparse.get_signals_for_residuals(*batch[:12], ra=batch[12], dec=batch[13], **KW)
        self.assertIsInstance(signals, DomainBaseArray)
        templates = self.sparse(*batch[:12], ra=batch[12], dec=batch[13], **KW)[0]
        for got, want in zip(signals, templates):
            np.testing.assert_array_equal(np.asarray(got.arr), want)

    def test_time_series_is_the_templates_signal(self):
        from lisatools.domains import DomainBaseArray, FDSignal, TDSettings, TDSignal

        template = self.sparse(*PAR, **KW)[0][0]
        td = self.sparse.get_td_signals(*PAR, **KW)
        self.assertIsInstance(td, TDSignal)
        self.assertEqual(td.t0, T0)                       # no sample outside the data
        self.assertEqual(td.arr.shape, (3, self.sparse.domain_settings.N))
        # all segments are tracks here: the template is their linear-chirp model of this series
        spec = _stft_of_td(self.sparse, td)
        self.assertLess(_mismatch(spec, template), 5e-6)
        # ... and not production's, which zeroes its first buffer_time; elsewhere they agree
        times, prod = self.dense.compute_tdi_channels(*PAR, **KW)
        j = np.rint((np.asarray(times) - T0) / DT).astype(int)
        ok = (j >= 0) & (j < td.arr.shape[-1])
        self.assertLess(np.abs(np.asarray(td.arr)[:, j[ok]] - np.asarray(prod)[:, ok]).max()
                        / np.abs(prod).max(), 1e-3)

        sub = TDSettings(N=1000, dt=DT, t0=T0 + 5000 * DT)
        part = self.sparse.get_td_signals(*PAR, td_settings=sub, **KW)
        np.testing.assert_array_equal(np.asarray(part.arr), np.asarray(td.arr)[:, 5000:6000])
        N = self.sparse.domain_settings.N
        fd = self.sparse.get_signals_for_residuals(
            *PAR, output_domain="FD", domain_kwargs=dict(N=N // 2 + 1, df=1 / (N * DT)), **KW)
        self.assertIsInstance(fd, FDSignal)
        self.assertTrue(np.any(np.asarray(fd.arr)))

        batch = [np.array([a, a]) for a in PAR]
        both = self.sparse.get_td_signals(*batch[:12], ra=batch[12], dec=batch[13], **KW)
        self.assertIsInstance(both, DomainBaseArray)
        for one in both:
            np.testing.assert_allclose(np.asarray(one.arr), np.asarray(td.arr), rtol=0,
                                       atol=1e-12 * np.abs(td.arr).max())

    def test_kwargs_rebuild_and_inherited_paths(self):
        from lisatools.sources.emri.waveform import EMRISparseSTFTWaveform

        kw = self.sparse.kwargs
        for key in ("n_side", "cubic_tol", "max_half_width", "sparse_dt", "stop_taper",
                    "fallback_via_polarizations", "waveform_class", "stft_dt"):
            self.assertIn(key, kw)
        clone = EMRISparseSTFTWaveform(**kw)
        np.testing.assert_allclose(clone(*PAR, **KW)[0], self.sparse(*PAR, **KW)[0], rtol=1e-12, atol=0)
        self.assertIs(self.sparse.xp, np)
        # the time domain goes through the parent's get_td_signals, on this class's channels
        self.assertIs(type(self.sparse).get_td_signals, type(self.dense).get_td_signals)
        self.assertIsNot(type(self.sparse).compute_tdi_channels, type(self.dense).compute_tdi_channels)
        self.assertIsNot(type(self.sparse).get_signals_for_residuals, type(self.dense).get_signals_for_residuals)

    def test_needs_stft(self):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.sources.emri.waveform import EMRISparseSTFTWaveform

        kw = dict(self.common, stft_dt=None)
        with self.assertRaises(ValueError):
            EMRISparseSTFTWaveform(orbits=EqualArmlengthOrbits(force_backend="cpu"), **kw)

    def test_polarization_fallback_where_production_cuts_at_T(self):
        # no plunge: production's FEW trajectory ends at its T, the holder's runs past it
        ref = _production_on_data_grid(self.dense, PAR, KW)
        item = _item(self.sparse, PAR, KW)
        samples = _assert_production_samples(self, self.dense, self.sparse, PAR, item)
        self.assertLess((samples[0] - 1) * DT, float(item[0][0][-1]))
        _assert_fallback_matches(self, self.sparse, item, samples, ref)


# plunges 4.6 d after the waveform start (3.6 d into the data); cubic_tol 0.3 puts the first
# 7 segments on tracks, the next 8 on the fallback, the last 5 after the plunge are zero
PLUNGE_PAR = (1e6, 400.0, 0.9, 6.0, 0.4, 1.0, 1.0, 0.5, 2.0, 0.3, 0.0, 1.2, 1.1, 0.4)


def _build(sum_kwargs, n_data, waveform_t0, **sparse_kwargs):
    """Production ``EMRITDIWaveform`` and ``EMRISparseSTFTWaveform`` on the same settings."""
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.domains import TDSettings
    from lisatools.sources.emri.response import EMRI_INSPIRAL_KWARGS
    from lisatools.sources.emri.waveform import EMRISparseSTFTWaveform, EMRITDIWaveform

    common = dict(
        waveform_class="FastKerrEccentricEquatorialFlux",
        waveform_kwargs=dict(inspiral_kwargs=dict(EMRI_INSPIRAL_KWARGS), sum_kwargs=dict(sum_kwargs)),
        waveform_t0=waveform_t0, data_td_settings=TDSettings(N=n_data, dt=DT, t0=T0),
        sampling_frequency=1 / DT, tukey_alpha=0.5, stft_dt=STFT_DT, freq_min=1e-4, freq_max=3e-2,
        force_backend="cpu")
    dense = EMRITDIWaveform(orbits=EqualArmlengthOrbits(force_backend="cpu"), **common)
    sparse = EMRISparseSTFTWaveform(orbits=EqualArmlengthOrbits(force_backend="cpu"),
                                    fallback_via_polarizations=True, **sparse_kwargs, **common)
    return dense, sparse


def _production_on_data_grid(dense, par, kw):
    """Production ``compute_tdi_channels`` on the data samples (zero where it has none)."""
    times, prod = dense.compute_tdi_channels(*par, **kw)
    N = dense.domain_settings.N
    j = np.rint((np.asarray(times) - dense.data_t0) / dense.dt).astype(int)
    ok = (j >= 0) & (j < N)
    out = np.zeros((prod.shape[0], N))
    out[:, j[ok]] = np.asarray(prod)[:, ok]
    return out


def _item(sparse, par, kw):
    resp, _ = sparse._sparse_setup()
    qS, phiS = sparse.convert_sky_coords(par[12], par[13])
    return resp.template_inputs((*par[:7], qS, phiS, *par[7:12]), lam=par[12], beta=par[13], **kw)[0]


def _assert_production_samples(test, dense, sparse, par, item):
    """``_production_samples`` is the h array production's ``wave_gen`` returns: its length and
    FEW's summed / padded split. Returns ``(n_signal, n_prod)``."""
    t_k = item[0][0]
    samples = sparse._production_samples(par[0], par[1], t_k[0], t_k[-1])
    merger_time = float(dense.domain_settings.t_end - dense.waveform_t0)
    _t, hp, _hc = dense.wave_gen(*par, merger_time, **KW)
    summation = dense.waveform.waveform_generator.create_waveform
    test.assertEqual(samples, (summation.num_pts, summation.num_pts + summation.num_pts_pad))
    test.assertEqual(hp.size, samples[1])
    test.assertEqual(int(np.flatnonzero(hp)[-1]), samples[0] - 1)
    return samples


def _assert_fallback_matches(test, sparse, item, samples, ref, tol=1e-9):
    """``_fallback_channels`` against production on contiguous, gappy and shuffled samples."""
    nper, N = sparse.nperseg, sparse.domain_settings.N
    nseg = N // nper
    scale = np.abs(ref).max()

    def seg(*idx):
        return np.concatenate([np.arange(s * nper, (s + 1) * nper) for s in idx])

    cases = dict(first=seg(0), middle=seg(nseg // 2), last=seg(nseg - 1), data_end=np.arange(N - nper, N),
                 gappy=seg(0, 2, 3, nseg - 1), everything=np.arange(N),
                 shuffled=np.random.default_rng(3).permutation(N)[:3 * nper])
    for name, j in cases.items():
        with test.subTest(samples=name):
            got = sparse._fallback_channels(item, sparse.data_t0 + j * sparse.dt, *samples)
            test.assertEqual(got.shape, (3, j.size))
            test.assertLess(np.abs(got - ref[:, j]).max() / scale, tol)
            np.testing.assert_array_equal(got == 0, ref[:, j] == 0)


class PolarizationFallbackTest(unittest.TestCase):
    """``fallback_via_polarizations``: the segments the tracks cannot represent are production's."""

    @classmethod
    def setUpClass(cls):
        try:
            from few.waveform import FastKerrEccentricEquatorialFlux  # noqa: F401
            from lisatools.sources.emri.response import EMRI_SUM_KWARGS

            cls.dense, cls.sparse = _build(EMRI_SUM_KWARGS, int(5 * DAY / DT), T0 - DAY, cubic_tol=0.3)
            cls.ref = _production_on_data_grid(cls.dense, PLUNGE_PAR, KW)
            cls.item = _item(cls.sparse, PLUNGE_PAR, KW)
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"FEW unavailable: {exc}")

    @classmethod
    def tearDownClass(cls):
        _release(cls, "dense", "sparse", "item", "ref")

    def test_polarizations_are_production_wave_gen(self):
        from lisatools.sources.emri.sparse_tracks import polarizations_from_item

        n_signal, n_prod = _assert_production_samples(self, self.dense, self.sparse, PLUNGE_PAR, self.item)
        # the plunge comes before production's T (padded): the item's own end is production's
        self.assertLess(n_signal, n_prod)
        merger_time = float(self.dense.domain_settings.t_end - self.dense.waveform_t0)
        t, hp, hc = self.dense.wave_gen(*PLUNGE_PAR, merger_time, **KW)
        mine = polarizations_from_item(self.item, t, t_end=(n_signal - 1) * DT)
        scale = max(np.abs(hp).max(), np.abs(hc).max())
        for got, want in zip(mine, (hp, hc)):
            self.assertLess(np.abs(got - want).max() / scale, 1e-9)
            np.testing.assert_array_equal(got == 0, want == 0)
        with self.assertRaises(ValueError):          # far past the trajectory
            polarizations_from_item(self.item, t, t_end=float(self.item[0][0][-1]) + DT)

    def test_fallback_channels_are_production(self):
        samples = _assert_production_samples(self, self.dense, self.sparse, PLUNGE_PAR, self.item)
        _assert_fallback_matches(self, self.sparse, self.item, samples, self.ref)

    def test_dense_segments_are_production(self):
        ref = self.dense(*PLUNGE_PAR, **KW)[0][0]
        out = self.sparse(*PLUNGE_PAR, **KW)[0][0]
        mode = self.sparse.last_segment_mode[0]
        self.assertEqual(set(np.unique(mode).tolist()), {0, 1, 2})
        scale = np.abs(ref).max()
        for j in np.flatnonzero(mode == 2):
            self.assertLess(np.abs(out[:, j] - ref[:, j]).max() / scale, 1e-9)
        for j in np.flatnonzero(mode == 0):
            self.assertFalse(np.any(out[:, j]))
            self.assertFalse(np.any(ref[:, j]))

    def test_time_series_matches_the_template_per_segment(self):
        scale_ref = np.abs(self.ref).max()
        for via_polarizations in (True, False):
            with self.subTest(fallback_via_polarizations=via_polarizations):
                self.sparse.fallback_via_polarizations = via_polarizations
                try:
                    template = self.sparse(*PLUNGE_PAR, **KW)[0][0]
                    mode = self.sparse.last_segment_mode[0].copy()
                    td = self.sparse.get_td_signals(*PLUNGE_PAR, **KW)
                    residual = self.sparse.get_signals_for_residuals(*PLUNGE_PAR, **KW)
                finally:
                    self.sparse.fallback_via_polarizations = True
                np.testing.assert_array_equal(np.asarray(residual.arr), template)
                self.assertEqual(set(np.unique(mode).tolist()), {0, 1, 2})
                spec = _stft_of_td(self.sparse, td)
                scale = np.abs(template).max()
                for j in np.flatnonzero(mode == 2):       # the same exact series, windowed
                    self.assertLess(np.abs(spec[:, j] - template[:, j]).max() / scale, 1e-10)
                for j in np.flatnonzero(mode == 1):       # its linear-chirp model (cubic_tol 0.3)
                    self.assertLess(_mismatch(spec[:, j], template[:, j]), 2e-3)
                for j in np.flatnonzero(mode == 0):
                    self.assertFalse(np.any(spec[:, j]))
                # the span stops with the signal's last segment
                nper = self.sparse.nperseg
                self.assertEqual(td.arr.shape[-1], (np.flatnonzero(mode)[-1] + 1) * nper)
                got = np.asarray(td.arr)
                diff = np.abs(got - self.ref[:, :got.shape[-1]]).max() / scale_ref
                if via_polarizations:                     # production's channels on every sample
                    self.assertLess(diff, 1e-9)
                    self.assertFalse(np.any(self.ref[:, got.shape[-1]:]))
                else:                                     # another response, tapered at the plunge
                    self.assertGreater(diff, 1e-9)

    def test_window_margins_are_enforced(self):
        from lisatools.sources.windowed_response import response_margins, response_on_window

        left, _back, fwd = response_margins(self.sparse)
        span = (float(self.sparse.waveform_t0), 51840)
        k_out = 30000                                    # an output sample well inside the data
        t_out = self.sparse.waveform_t0 + k_out * DT + np.arange(10) * DT
        h = np.zeros(left + 10 + fwd + 2)
        with self.assertRaises(ValueError):              # one sample short on the left
            response_on_window(self.sparse, span, k_out - left + 1, h, h, 1.1, 0.4, t_out)
        with self.assertRaises(ValueError):              # too short on the right
            response_on_window(self.sparse, span, k_out - left - 1, h[:left + 12], h[:left + 12], 1.1, 0.4, t_out)

    def test_spline_projection_is_refused(self):
        import types

        from lisatools.sources.windowed_response import response_margins

        with self.assertRaises(ValueError):
            response_margins(types.SimpleNamespace(response=types.SimpleNamespace(use_spline=True)))


class ProductionEdgeCasesTest(unittest.TestCase):
    """Where production's polarization array is not ``int(merger_time / dt) + 1`` samples of a
    padded mode sum, or the data grid is not on the waveform's sample grid."""

    def setUp(self):
        try:
            from few.waveform import FastKerrEccentricEquatorialFlux  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"FEW unavailable: {exc}")

    def tearDown(self):
        _release(self, "built")

    def _build(self, *args, **kwargs):
        self.built = _build(*args, **kwargs)        # released in tearDown, even on a failure
        return self.built

    def test_unpadded_production_stops_at_the_plunge(self):
        # FEW's default pad_output=False: production's h, hence its channels, stop at the plunge
        dense, sparse = self._build(dict(pad_output=False), int(5 * DAY / DT), T0 - DAY, cubic_tol=0.3)
        item = _item(sparse, PLUNGE_PAR, KW)
        samples = _assert_production_samples(self, dense, sparse, PLUNGE_PAR, item)
        self.assertEqual(samples[0], samples[1])
        _assert_fallback_matches(self, sparse, item, samples, _production_on_data_grid(dense, PLUNGE_PAR, KW))
        ref = dense(*PLUNGE_PAR, **KW)[0][0]
        out = sparse(*PLUNGE_PAR, **KW)[0][0]
        dense_j = np.flatnonzero(sparse.last_segment_mode[0] == 2)
        self.assertGreater(dense_j.size, 0)
        self.assertLess(np.abs(out[:, dense_j] - ref[:, dense_j]).max() / np.abs(ref).max(), 1e-9)

    def test_sample_lost_in_the_years_round_trip(self):
        # 25289 dt -> years -> FEW's int(T YRSID_SI / dt) = 25288: production is one sample short
        dense, sparse = self._build(dict(pad_output=True), int(2 * DAY / DT), T0 - 8010 * DT)
        merger_time = float(dense.domain_settings.t_end - dense.waveform_t0)
        item = _item(sparse, PAR, KW)
        samples = _assert_production_samples(self, dense, sparse, PAR, item)
        self.assertEqual(samples[1], int(merger_time / DT))
        _assert_fallback_matches(self, sparse, item, samples, _production_on_data_grid(dense, PAR, KW))

    def test_subsample_shift_and_window_at_the_waveform_start(self):
        # data_t0 - waveform_t0 = 1996.3 s: production Lagrange-shifts h by -3.7 s, and a window
        # at the data start begins at production's first sample (k0 = 0)
        from lisatools.sources.windowed_response import response_on_window, window_input_range

        dense, sparse = self._build(dict(pad_output=True), int(2 * DAY / DT), T0 - 2000.0 + 3.7)
        ref = _production_on_data_grid(dense, PAR, KW)
        item = _item(sparse, PAR, KW)
        n_signal, n_prod = _assert_production_samples(self, dense, sparse, PAR, item)
        merger_time = float(dense.domain_settings.t_end - dense.waveform_t0)
        _t, hp, hc = dense.wave_gen(*PAR, merger_time, **KW)
        span = (float(dense.waveform_t0), n_prod)
        N, nper = dense.domain_settings.N, dense.nperseg
        scale = np.abs(ref).max()
        # the windowed response fed with production's own h: exact
        for name, j in (("start", np.arange(nper)), ("end", np.arange(N - nper, N))):
            with self.subTest(window=name):
                t_out = T0 + j * DT
                k0, n_in, n_zero = window_input_range(sparse, span, t_out[0], t_out[-1])
                if name == "start":
                    self.assertEqual(k0, 0)
                hpw, hcw = np.zeros(n_in), np.zeros(n_in)
                hpw[n_zero:], hcw[n_zero:] = hp[k0 + n_zero:k0 + n_in], hc[k0 + n_zero:k0 + n_in]
                got = response_on_window(sparse, span, k0, hpw, hcw, PAR[12], PAR[13], t_out)
                self.assertLess(np.abs(got - ref[:, j]).max() / scale, 1e-11)
                np.testing.assert_array_equal(got == 0, ref[:, j] == 0)
        # end to end: production's 2-d trajectory is cut at T after 4 knots, the holder's runs
        # on, so their last DOPR853 steps differ (h differs by ~2.5e-8, integrator tolerance)
        _assert_fallback_matches(self, sparse, item, (n_signal, n_prod), ref, tol=1e-6)


if __name__ == "__main__":
    unittest.main()
