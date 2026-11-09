"""Validate the batched (num_psds) noise-covariance kernel.

``get_noise_covariance_wrap`` evaluates the XYZ covariance for ``num_psds``
noise-parameter sets in a single launch (3rd grid dimension). Each batched slice
must be bitwise-identical to a ``num_psds=1`` call with that slice's parameters and
per-PSD spline weights. On the spline path each PSD (walker) carries its own knots;
shared knots are rejected. CPU backend only; no data files required.
"""

# Pre-existing env shims (see tests/test_quintic_response.py): scienceplots is
# incompatible with the installed matplotlib, and eryn.utils.plot uses a bare
# ``typing.Union`` without importing typing.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU backend

import unittest

import numpy as np

import lisatools
from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import FDSettings
from lisatools.sensitivity import XYZSensitivityBackend

N_TIMES, N_FREQS, N_PSDS = 7, 300, 5


def _alloc(n):
    return [np.empty(n, dtype=np.float64 if real else np.complex128)
            for real in (True, False, False, True, False, True)]


def _random_params(rng, k):
    return np.column_stack([
        15e-12 * (1 + 0.1 * rng.random(k)),  # Soms_d
        3e-15 * (1 + 0.1 * rng.random(k)),   # Sa_a
        1e-43 * rng.random(k),               # Amp
        1.5 * rng.random(k),                 # alpha
        1e-3 * rng.random(k) + 1e-4,         # f_1
        2e-3 * rng.random(k) + 1e-3,         # f_knee
        1e-3 * rng.random(k) + 5e-4,         # f_2
    ])


class BatchedNoiseCovarianceWrapTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.backend = lisatools.get_backend("cpu")
        # the wrap stores the LTT / normalization pointers NON-OWNED -> keep them alive on self
        self.avg = np.ascontiguousarray(8.33 + 1e-3 * rng.standard_normal((N_TIMES, 6))).flatten()
        self.delta = np.ascontiguousarray(1e-4 * rng.standard_normal((N_TIMES, 6))).flatten()
        self.noise_norm = np.full(N_FREQS, 0.9)
        self.wrap = self.backend.SensitivityMatrixWrap(
            self.avg, self.delta, N_TIMES, 2.5e9, 2, True, self.noise_norm)
        self.f = np.logspace(-4, -1, N_FREQS)
        self.ti = np.arange(N_TIMES, dtype=np.int32)
        self.params = _random_params(rng, N_PSDS)
        self.spl_oms = 0.1 * rng.standard_normal((N_PSDS, N_FREQS))
        self.spl_tm = 0.1 * rng.standard_normal((N_PSDS, N_FREQS))

    def _call(self, params, spl_oms, spl_tm, num_psds, **kwargs):
        c = _alloc(num_psds * N_TIMES * N_FREQS)
        self.wrap.get_noise_covariance_wrap(
            self.f, self.ti,
            *[np.ascontiguousarray(params[:, j]) for j in range(7)],
            np.ascontiguousarray(spl_oms).ravel(), np.ascontiguousarray(spl_tm).ravel(),
            *c, N_FREQS, N_TIMES, num_psds, **kwargs)
        return np.stack([x.astype(np.complex128) for x in c])

    def test_batched_matches_looped(self):
        batched = self._call(self.params, self.spl_oms, self.spl_tm, N_PSDS)
        batched = batched.reshape(6, N_PSDS, N_TIMES * N_FREQS)
        for k in range(N_PSDS):
            single = self._call(self.params[k:k + 1], self.spl_oms[k], self.spl_tm[k], 1)
            np.testing.assert_array_equal(batched[:, k], single)

    def test_run_async_flag_accepted_on_cpu(self):
        sync = self._call(self.params, self.spl_oms, self.spl_tm, N_PSDS)
        async_ = self._call(self.params, self.spl_oms, self.spl_tm, N_PSDS, run_async=True)
        np.testing.assert_array_equal(sync, async_)

    def test_length_check_rejects_short_param_array(self):
        with self.assertRaises(Exception):
            self._call(self.params[:2], self.spl_oms, self.spl_tm, N_PSDS)


class BatchedComputeMatrixElementsTest(unittest.TestCase):
    """``XYZSensitivityBackend._compute_matrix_elements`` with scalar vs array params."""

    def setUp(self):
        settings = FDSettings(N=4096, df=5e-6, min_freq=1e-4, max_freq=2e-2, force_backend="cpu")
        orbits = EqualArmlengthOrbits(force_backend="cpu")
        self.sm = XYZSensitivityBackend(orbits=orbits, settings=settings, force_backend="cpu")
        self.params = _random_params(np.random.default_rng(1), N_PSDS)

    def test_scalar_shape_unchanged_and_array_matches_scalar(self):
        total_terms = self.sm.basis_settings.total_terms
        batched = self.sm._compute_matrix_elements(self.sm.f_arr, *self.params.T)
        for c in batched:
            self.assertEqual(c.shape, (N_PSDS, total_terms))
        for k in range(N_PSDS):
            single = self.sm._compute_matrix_elements(self.sm.f_arr, *[float(p) for p in self.params[k]])
            for c_b, c_s in zip(batched, single):
                self.assertEqual(c_s.shape, (total_terms,))
                np.testing.assert_array_equal(c_b[k], c_s)

    def test_scalars_broadcast_against_arrays(self):
        p = self.params
        mixed = self.sm._compute_matrix_elements(self.sm.f_arr, p[:, 0], float(p[0, 1]), *p[:, 2:].T)
        full = self.sm._compute_matrix_elements(
            self.sm.f_arr, p[:, 0], np.full(N_PSDS, p[0, 1]), *p[:, 2:].T)
        for a, b in zip(mixed, full):
            np.testing.assert_array_equal(a, b)


class BatchedSplineComputeMatrixElementsTest(unittest.TestCase):
    """Spline path: every walker carries its own knots, shape (2, num_psds, n_knots)
    as built by ``globalfit.moves.psdmove``; shared knots are rejected."""

    N_KNOTS = 6

    def setUp(self):
        settings = FDSettings(N=4096, df=5e-6, min_freq=1e-4, max_freq=2e-2, force_backend="cpu")
        orbits = EqualArmlengthOrbits(force_backend="cpu")
        self.sm = XYZSensitivityBackend(
            orbits=orbits, settings=settings, force_backend="cpu", use_splines=True)
        rng = np.random.default_rng(2)
        self.params = _random_params(rng, N_PSDS)
        lo, hi = np.log10(self.sm.f_arr[0]) - 0.1, np.log10(self.sm.f_arr[-1]) + 0.1
        inner = np.sort(rng.uniform(lo, hi, (2, N_PSDS, self.N_KNOTS - 2)), axis=-1)
        self.knots_pos = np.concatenate(
            [np.full((2, N_PSDS, 1), lo), inner, np.full((2, N_PSDS, 1), hi)], axis=-1)
        self.knots_amp = 0.5 * rng.standard_normal((2, N_PSDS, self.N_KNOTS))

    def test_per_walker_knots_match_looped(self):
        batched = self.sm._compute_matrix_elements(
            self.sm.f_arr, *self.params.T, self.knots_pos, self.knots_amp)
        flat = self.sm._compute_matrix_elements(
            self.sm.f_arr, *self.params.T, self.knots_pos, np.zeros_like(self.knots_amp))
        # the splines must actually change the covariance (non-vacuous check)
        # (atol=0: covariance entries are ~1e-40, below allclose's default atol)
        self.assertFalse(np.allclose(batched[0], flat[0], rtol=1e-3, atol=0.0))
        for k in range(N_PSDS):
            single = self.sm._compute_matrix_elements(
                self.sm.f_arr, *[float(p) for p in self.params[k]],
                np.ascontiguousarray(self.knots_pos[:, k]), np.ascontiguousarray(self.knots_amp[:, k]))
            for c_b, c_s in zip(batched, single):
                np.testing.assert_array_equal(c_b[k], c_s)

    def test_shared_knots_rejected(self):
        shared_pos = np.ascontiguousarray(self.knots_pos[:, 0])
        shared_amp = np.ascontiguousarray(self.knots_amp[:, 0])
        with self.assertRaises(ValueError):
            self.sm._compute_matrix_elements(self.sm.f_arr, *self.params.T, shared_pos, shared_amp)

    def test_shared_knots_rejected_by_log_like(self):
        # raised before the kernel call, so the data contents are irrelevant
        data = np.zeros((N_PSDS, 3 * self.sm.basis_settings.total_terms), dtype=np.complex128)
        with self.assertRaises(ValueError):
            self.sm.compute_log_like(
                data, np.arange(N_PSDS, dtype=np.int32), *self.params.T,
                knots_position_all=np.ascontiguousarray(self.knots_pos[:, 0]),
                knots_amplitude_all=np.ascontiguousarray(self.knots_amp[:, 0]))


class ExternalMatrixLogLikeTest(unittest.TestCase):
    """compute_log_like: the convolve path (external matrix) must reproduce the in-kernel
    path when the window is rectangular. |FFT(1)|^2 / N^2 is a delta at zero lag, so the
    convolution is the identity and both paths get noise_normalization = filters_response.
    Checks that every noise parameter, the per-walker splines and noise_normalization play
    the same role on both paths."""

    N_KNOTS = 6
    N_WALKERS = 4

    def _backends(self, use_splines, filters_response=None):
        settings = FDSettings(N=4096, df=5e-6, min_freq=1e-4, max_freq=2e-2, force_backend="cpu")
        orbits = EqualArmlengthOrbits(force_backend="cpu")
        # one-sided grid of the window's DFT must be the FD grid: len//2 + 1 == N
        window = np.ones(2 * (settings.N - 1))
        return [
            XYZSensitivityBackend(
                orbits=orbits, settings=settings, force_backend="cpu", use_splines=use_splines,
                window_values=window, convolve_window=convolve, filters_response=filters_response)
            for convolve in (False, True)
        ]

    def setUp(self):
        rng = np.random.default_rng(3)
        self.rng = rng
        self.params = _random_params(rng, self.N_WALKERS)

    def _knots(self, sm):
        rng, k = self.rng, self.N_WALKERS
        lo, hi = np.log10(sm.f_arr[0]) - 0.1, np.log10(sm.f_arr[-1]) + 0.1
        inner = np.sort(rng.uniform(lo, hi, (2, k, self.N_KNOTS - 2)), axis=-1)
        pos = np.concatenate([np.full((2, k, 1), lo), inner, np.full((2, k, 1), hi)], axis=-1)
        return pos, 0.5 * rng.standard_normal((2, k, self.N_KNOTS))

    def _data(self, sm, knots_position_all=None, knots_amplitude_all=None):
        """Noise realisation drawn from each walker's own covariance, scaled so that
        2 df d^H C^-1 d ~ 3 per bin (realistic likelihood regime)."""
        c = sm._compute_matrix_elements(
            sm.f_arr, *self.params.T, knots_position_all, knots_amplitude_all, convolve_window=False)
        c00, c11, c22, c01, c02, c12 = [np.asarray(x) for x in c]
        C = np.empty(c00.shape + (3, 3), dtype=np.complex128)
        C[..., 0, 0] = c00; C[..., 1, 1] = c11; C[..., 2, 2] = c22
        C[..., 0, 1] = c01; C[..., 1, 0] = c01.conj()
        C[..., 0, 2] = c02; C[..., 2, 0] = c02.conj()
        C[..., 1, 2] = c12; C[..., 2, 1] = c12.conj()
        z = (self.rng.standard_normal(c00.shape + (3,))
             + 1j * self.rng.standard_normal(c00.shape + (3,))) / np.sqrt(2)
        d = np.einsum("...ij,...j->...i", np.linalg.cholesky(C), z)
        d /= np.sqrt(2 * sm.basis_settings.differential_component)
        return np.ascontiguousarray(np.swapaxes(d, -1, -2)).reshape(self.N_WALKERS, -1)

    def _compare(self, internal, external, **knots):
        data = self._data(internal, **knots)
        idx = np.arange(self.N_WALKERS, dtype=np.int32)
        ll_int = internal.compute_log_like(data, idx, *self.params.T, **knots)
        ll_ext = external.compute_log_like(data, idx, *self.params.T, **knots)
        self.assertTrue(np.all(np.isfinite(ll_int)) and np.all(ll_int != 0.0))
        # The identity FFT convolution is exact only to round-off relative to the band's
        # largest |C| (~6e-16 of it). At low f the XYZ covariance is near-singular (cond ~1e6
        # at 1e-4 Hz), so this reaches ~1e-3 in log-likelihood for noise-like data, and
        # ~1e-2 with a jagged filter response. Grows as min_freq drops.
        np.testing.assert_allclose(ll_ext, ll_int, rtol=0.0, atol=5e-2)
        return ll_int

    def test_external_unconvolved_matches_internal(self):
        """Binding + kernel external path, fed the unconvolved covariance: ~exact."""
        internal, external = self._backends(use_splines=True)
        pos, amp = self._knots(internal)
        data = self._data(internal, pos, amp)
        idx = np.arange(self.N_WALKERS, dtype=np.int32)
        ll_int = internal.compute_log_like(
            data, idx, *self.params.T, knots_position_all=pos, knots_amplitude_all=amp)
        raw = external._compute_matrix_elements(
            external.f_arr, *self.params.T, pos, amp, convolve_window=False)
        out = np.zeros(self.N_WALKERS)
        empty = np.empty(0)
        external.pycpp_sensitivity_matrix.psd_likelihood_wrap(
            out, external.f_arr, data.flatten(), idx, external.time_indices,
            *[np.ascontiguousarray(p) for p in self.params.T], empty, empty,
            external.basis_settings.differential_component, external.num_freqs,
            external.num_times, external.dips_mask, self.N_WALKERS,
            *[np.ascontiguousarray(c).ravel() for c in raw], True, False)
        np.testing.assert_allclose(out, ll_int, rtol=1e-13, atol=0.0)

    def test_strided_params_are_sanitized(self):
        """Regression: column views params[:, j] are strided; walker i>0 must still get
        its own parameters (nanobind reads .data() as contiguous)."""
        internal, _ = self._backends(use_splines=False)
        data = self._data(internal)
        idx = np.arange(self.N_WALKERS, dtype=np.int32)
        ll_strided = internal.compute_log_like(data, idx, *self.params.T)
        ll_contig = internal.compute_log_like(
            data, idx, *[np.ascontiguousarray(p) for p in self.params.T])
        np.testing.assert_array_equal(ll_strided, ll_contig)

    def test_no_splines(self):
        self._compare(*self._backends(use_splines=False))

    def test_splines(self):
        internal, external = self._backends(use_splines=True)
        pos, amp = self._knots(internal)
        knots = dict(knots_position_all=pos, knots_amplitude_all=amp)
        data = self._data(internal, **knots)
        idx = np.arange(self.N_WALKERS, dtype=np.int32)
        # the knots must actually move the likelihood (non-vacuous)
        ll = internal.compute_log_like(data, idx, *self.params.T, **knots)
        ll_flat = internal.compute_log_like(
            data, idx, *self.params.T, knots_position_all=pos, knots_amplitude_all=np.zeros_like(amp))
        self.assertTrue(np.all(np.abs(ll - ll_flat) > 1.0))
        self._compare(internal, external, **knots)

    def test_filters_response(self):
        nf = len(FDSettings(N=4096, df=5e-6, min_freq=1e-4, max_freq=2e-2, force_backend="cpu").f_arr)
        filt = 0.5 + self.rng.random(nf)
        internal, external = self._backends(use_splines=True, filters_response=filt)
        pos, amp = self._knots(internal)
        self._compare(internal, external, knots_position_all=pos, knots_amplitude_all=amp)


if __name__ == "__main__":
    unittest.main()
