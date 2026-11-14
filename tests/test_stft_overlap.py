"""Overlapping STFT grids: settings, the numerical STFT, its normalisation, the noise term, guards.

A contiguous grid (overlap = 1) must be unchanged bit for bit. An overlapping grid puts a Hann
window on segments of overlap hops, whose squares sum to window_power at every sample, so that
differential_component weights the pixels to the time-domain inner product exactly (white noise).
"""

import unittest

import numpy as np

from lisatools.diagnostic import noise_likelihood_term
from lisatools.domains import (
    STFTSettings,
    STFTSignal,
    TDSettings,
    TDSignal,
    _apply_stft_add,
    get_stft_settings,
)
from lisatools.utils.utility import tukey

T0 = 100.0
DT = 1.0
NPERSEG = 96
OVERLAP = 3
NHOP = NPERSEG // OVERLAP
# * Not a whole number of segments, so the contiguous grid truncates and the last overlapping
# * segment has to stop inside the record.
N = NPERSEG * 40 + 17


def record_times(n=N):
    return T0 + DT * np.arange(n)


def td_signal(x):
    return TDSignal(x, TDSettings(N=x.shape[-1], dt=DT, t0=T0, force_backend="cpu"))


def one_sided_weights(num_freqs, nperseg):
    """Parseval weights of an rfft: 1 at DC and at an even-length Nyquist bin, 2 elsewhere."""
    weights = np.full(num_freqs, 2.0)
    weights[0] = 1.0
    if nperseg % 2 == 0:
        weights[-1] = 1.0
    return weights


def white_record(rng, settings):
    """White noise on the samples every segment window covers fully, zero elsewhere."""
    x = np.zeros(N)
    # * Full coverage runs from the (overlap - 1)-th hop to the start of the last hop's successor.
    start = (settings.overlap - 1) * NHOP
    stop = settings.NT * NHOP
    x[start:stop] = rng.standard_normal(stop - start)
    return x


class _PSDStub:
    """What noise_likelihood_term reads from a sensitivity matrix."""

    def __init__(self, basis_settings, det_c):
        self.basis_settings = basis_settings
        self.detC = det_c
        self.shape = det_c.shape

    def __getitem__(self, index):
        return self.detC[index]


class ContiguousGridTest(unittest.TestCase):
    """overlap = 1 keeps today's grid, transform and normalisation."""

    def setUp(self):
        self.settings = get_stft_settings(record_times(), NPERSEG * DT, force_backend="cpu")

    def test_settings_unchanged(self):
        s = self.settings
        self.assertEqual(
            (s.t0, s.dt, s.df, s.NT, s.NF, s.overlap),
            (T0, NPERSEG * DT, 1.0 / (NPERSEG * DT), N // NPERSEG, NPERSEG // 2 + 1, 1),
        )
        self.assertEqual(s.seg_duration, s.dt)
        self.assertEqual(s.differential_component, s.df)
        self.assertEqual(s.redundancy, 1.0)
        self.assertIsNone(s.segment_window(DT))
        self.assertNotIn("overlap", s.kwargs)

    def test_stft_bit_identical(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(N)
        window = tukey(NPERSEG, 0.2)
        out = td_signal(x).stft(self.settings, window=window).arr
        segments = x[: self.settings.NT * NPERSEG].reshape(self.settings.NT, NPERSEG)
        reference = DT * np.fft.rfft(window[None, :] * segments, axis=-1)
        np.testing.assert_array_equal(out, reference[None, :, self.settings.active_slice])

    def test_rectangular_parseval(self):
        """The Parseval identity the overlapping test relies on, on the contiguous grid."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(N)
        x[self.settings.NT * NPERSEG :] = 0.0
        pixels = td_signal(x).stft(self.settings).arr[0]
        weights = one_sided_weights(self.settings.NF, NPERSEG)
        stft_norm = self.settings.differential_component * np.sum(weights * np.abs(pixels) ** 2)
        self.assertAlmostEqual(stft_norm / (DT * np.sum(x**2)), 1.0, delta=1e-12)


class OverlappingGridTest(unittest.TestCase):
    """overlap = 3: Hann segments of NPERSEG samples every NHOP samples."""

    def setUp(self):
        self.settings = get_stft_settings(
            record_times(), NPERSEG * DT, overlap=OVERLAP, force_backend="cpu"
        )

    def test_grid(self):
        s = self.settings
        self.assertEqual((s.dt, s.seg_duration, s.overlap), (NHOP * DT, NPERSEG * DT, OVERLAP))
        self.assertAlmostEqual(s.df * s.seg_duration, 1.0, delta=1e-15)
        self.assertEqual(s.NF, NPERSEG // 2 + 1)
        self.assertEqual(s.get_nperseg(DT), NPERSEG)
        self.assertEqual(s.get_nhop(DT), NHOP)
        # * The last segment ends inside the record and one more would not.
        self.assertLessEqual((s.NT - 1) * NHOP + NPERSEG, N)
        self.assertGreater(s.NT * NHOP + NPERSEG, N)
        self.assertEqual(s.window_power, 9.0 / 8.0)
        self.assertEqual(s.redundancy, 1.0 / 3.0)
        self.assertEqual(s.differential_component, s.df / (9.0 / 8.0))

    def test_copies_keep_overlap(self):
        s = self.settings
        rebuilt = STFTSettings(*s.args, **s.kwargs)
        self.assertEqual(rebuilt, s)
        self.assertEqual(rebuilt.overlap, OVERLAP)
        self.assertNotEqual(get_stft_settings(record_times(), NPERSEG * DT), s)
        self.assertEqual(s.get_slice((slice(2, 10), slice(3, 20))).overlap, OVERLAP)
        signal = STFTSignal(np.zeros((1,) + s.basis_shape_active, dtype=complex), s)
        self.assertEqual(signal.settings.overlap, OVERLAP)

    def test_window_squares_sum_to_window_power(self):
        window = self.settings.segment_window(DT)
        self.settings.check_segment_window(window, DT)
        coverage = (window**2).reshape(OVERLAP, NHOP).sum(axis=0)
        np.testing.assert_allclose(coverage, 9.0 / 8.0, rtol=1e-14)

    def test_frames(self):
        """Segment k holds samples k NHOP ... k NHOP + NPERSEG - 1 under the Hann window."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(N)
        out = td_signal(x).stft(self.settings).arr[0]
        window = self.settings.segment_window(DT)
        for k in (0, 1, self.settings.NT - 1):
            segment = x[k * NHOP : k * NHOP + NPERSEG]
            np.testing.assert_array_equal(out[k], DT * np.fft.rfft(window * segment))

    def test_white_noise_parseval(self):
        """Pixels weighted by differential_component give the time-domain inner product."""
        rng = np.random.default_rng(4)
        x = white_record(rng, self.settings)
        y = white_record(rng, self.settings)
        pixels_x = td_signal(x).stft(self.settings).arr[0]
        pixels_y = td_signal(y).stft(self.settings).arr[0]
        weights = one_sided_weights(self.settings.NF, NPERSEG)
        diff = self.settings.differential_component
        norm = diff * np.sum(weights * np.abs(pixels_x) ** 2)
        cross = diff * np.sum(weights * (pixels_x.conj() * pixels_y).real)
        self.assertAlmostEqual(norm / (DT * np.sum(x**2)), 1.0, delta=1e-12)
        self.assertAlmostEqual(cross / (DT * np.sum(x * y)), 1.0, delta=1e-10)

    def test_noise_term_weighted_by_redundancy(self):
        det_c = np.random.default_rng(5).uniform(0.5, 2.0, size=(4, 7))
        contiguous = get_stft_settings(record_times(), NPERSEG * DT, force_backend="cpu")
        full = noise_likelihood_term(_PSDStub(contiguous, det_c))
        redundant = noise_likelihood_term(_PSDStub(self.settings, det_c))
        self.assertAlmostEqual(redundant / full, 1.0 / OVERLAP, delta=1e-15)


def _gpu_available():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@unittest.skipUnless(_gpu_available(), "requires a CUDA device")
class OverlappingGridGPUTest(unittest.TestCase):
    """The overlapping transform on CuPy arrays matches the NumPy one."""

    def test_gpu_matches_cpu(self):
        import cupy

        x = np.random.default_rng(6).standard_normal(N)
        cpu_settings = get_stft_settings(
            record_times(), NPERSEG * DT, overlap=OVERLAP, force_backend="cpu"
        )
        gpu_settings = get_stft_settings(
            record_times(), NPERSEG * DT, overlap=OVERLAP, force_backend="gpu"
        )
        gpu_signal = TDSignal(
            cupy.asarray(x), TDSettings(N=N, dt=DT, t0=T0, force_backend="gpu")
        )
        out_gpu = cupy.asnumpy(gpu_signal.stft(gpu_settings).arr)
        out_cpu = td_signal(x).stft(cpu_settings).arr
        np.testing.assert_allclose(out_gpu, out_cpu, rtol=0.0, atol=1e-12 * np.abs(out_cpu).max())


class InvalidInputTest(unittest.TestCase):
    def test_overlap_two_is_refused(self):
        with self.assertRaises(ValueError):
            STFTSettings(T0, 48.0, 1.0 / 96.0, 10, 49, overlap=2)

    def test_df_must_match_segment(self):
        with self.assertRaises(ValueError):
            STFTSettings(T0, 32.0, 1.0 / 32.0, 10, 49, overlap=3)

    def test_hop_must_be_whole_samples(self):
        with self.assertRaises(ValueError):
            get_stft_settings(record_times(), 95 * DT, overlap=3)

    def test_window_must_fit(self):
        settings = get_stft_settings(record_times(), NPERSEG * DT, overlap=OVERLAP)
        signal = td_signal(np.zeros(N))
        with self.assertRaises(ValueError):
            signal.stft(settings, window=np.ones(NPERSEG))  # squares sum to 3, not constant 9/8
        with self.assertRaises(ValueError):
            signal.stft(settings, window=np.ones(NPERSEG - 1))


class GuardTest(unittest.TestCase):
    """Consumers that still assume contiguous segments refuse an overlapping grid."""

    def setUp(self):
        self.settings = get_stft_settings(
            record_times(), NPERSEG * DT, overlap=OVERLAP, min_freq=0.05, max_freq=0.3,
            force_backend="cpu",
        )

    def test_template_on_another_grid(self):
        contiguous = get_stft_settings(record_times(), NPERSEG * DT, force_backend="cpu")
        target = STFTSignal(
            np.zeros((1,) + self.settings.basis_shape_active, dtype=complex), self.settings
        )
        template = np.zeros((1,) + contiguous.basis_shape_active, dtype=complex)
        with self.assertRaises(ValueError):
            _apply_stft_add(target, 1.0, template, contiguous)

    def test_computation_group(self):
        from types import SimpleNamespace

        from lisatools.domaincomputation import STFTComputationGroup

        with self.assertRaises(NotImplementedError):
            STFTComputationGroup(SimpleNamespace(settings=self.settings))

    def test_batched_psd_likelihood(self):
        from types import SimpleNamespace

        from lisatools.sensitivity import XYZSensitivityBackend

        stub = SimpleNamespace(basis_settings=self.settings)
        with self.assertRaises(NotImplementedError):
            XYZSensitivityBackend._log_like_batch(stub, *([None] * 14))

    def test_band_structure(self):
        from lisatools.globalfit.moves.gbbandstructure import stft_band_structure

        with self.assertRaises(NotImplementedError):
            stft_band_structure(self.settings, 0.1)

    def test_gwtf_generator(self):
        try:
            from lisatools.sources.sobbh.waveform_gwtf import GWTF_generator
        except ImportError as error:
            self.skipTest(f"pygwtf unavailable: {error}")
        with self.assertRaises(NotImplementedError):
            GWTF_generator(settings=self.settings, orbits=None, T_obs=1.0)


if __name__ == "__main__":
    unittest.main()
