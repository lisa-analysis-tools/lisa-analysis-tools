"""EMRISparseSTFTWaveform against the production dense EMRITDIWaveform (real FEW, CPU).

Same constructor arguments and call parameters; the STFT templates agree where the dense path
is complete (it zeroes its first ``buffer_time`` after ``waveform_t0``, so the waveform starts a
day before the data here). Skips without FEW ``gpu_backend`` or its data files.
"""
import unittest

import numpy as np

DAY = 86400.0
T0 = 2.0e7
DT = 10.0
STFT_DT = 21600.0
#      m1   m2    a    p0   e0   x0   dist qK   phiK Pp   Pt   Pr   ra   dec
PAR = (1e6, 10.0, 0.9, 9.5, 0.3, 1.0, 1.0, 0.5, 2.0, 0.3, 0.0, 1.2, 1.1, 0.4)
KW = dict(mode_selection_threshold=1e-2)


def _mismatch(a, b):
    return 1 - np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)


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
        # ? the dense path samples a symmetric Tukey window, the tracks model the periodic one:
        # ? an amplitude difference of order 1 / samples per segment
        self.assertLess(abs(np.sqrt(np.vdot(out, out).real / np.vdot(ref, ref).real) - 1), 2 / (STFT_DT / DT))

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

    def test_kwargs_rebuild_and_inherited_paths(self):
        from lisatools.sources.emri.waveform import EMRISparseSTFTWaveform

        kw = self.sparse.kwargs
        for key in ("n_side", "cubic_tol", "max_half_width", "sparse_dt", "stop_taper", "waveform_class", "stft_dt"):
            self.assertIn(key, kw)
        clone = EMRISparseSTFTWaveform(**kw)
        np.testing.assert_allclose(clone(*PAR, **KW)[0], self.sparse(*PAR, **KW)[0], rtol=1e-12, atol=0)
        self.assertIs(self.sparse.xp, np)
        # the time-domain and residual paths are the dense ones, untouched
        self.assertIs(type(self.sparse).get_td_signals, type(self.dense).get_td_signals)
        self.assertIs(type(self.sparse).get_signals_for_residuals, type(self.dense).get_signals_for_residuals)

    def test_needs_stft(self):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.sources.emri.waveform import EMRISparseSTFTWaveform

        kw = dict(self.common, stft_dt=None)
        with self.assertRaises(ValueError):
            EMRISparseSTFTWaveform(orbits=EqualArmlengthOrbits(force_backend="cpu"), **kw)


if __name__ == "__main__":
    unittest.main()
