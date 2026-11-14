"""Information matrix (:mod:`lisatools.diagnostic`) for waveforms that return ``DomainBase`` objects."""

import unittest

import numpy as np

import lisatools
from lisatools import domains
from lisatools.diagnostic import covariance, dh_dlambda, estimate_eps, info_matrix, inner_product
from lisatools.sensitivity import SensitivityMatrix

N_FREQ = 2049
DF = 1.0 / (4096 * 5.0)
NCHANNELS = 2


def make_setup(force_backend="cpu"):
    """Full-grid settings, a non-flat PSD on it and two fixed basis signals (host arrays)."""
    settings = domains.FDSettings(N_FREQ, DF, force_backend=force_backend)
    xp = settings.xp
    rng = np.random.default_rng(42)
    freqs = np.arange(N_FREQ) * DF
    psd_values = xp.asarray(1.0 + (freqs / 0.02) ** 2)
    psd = SensitivityMatrix(settings, [psd_values] * NCHANNELS)
    basis = [
        xp.asarray(rng.normal(size=(NCHANNELS, N_FREQ)) + 1j * rng.normal(size=(NCHANNELS, N_FREQ)))
        for _ in range(2)
    ]
    return settings, psd, basis


def wave_packet(amp, f0, phi, tc, freqs):
    """One channel of a Gaussian packet in frequency with a phase and a time shift."""
    return amp * np.exp(-0.5 * ((freqs - f0) / 2e-3) ** 2) * np.exp(1j * (phi - 2 * np.pi * freqs * tc))


class LinearModelTest(unittest.TestCase):
    """h = a u + b v: the stencil is exact, so F_ij = <u_i|u_j>."""

    def _check(self, force_backend):
        settings, psd, (u, v) = make_setup(force_backend)

        def waveform(a, b):
            return domains.FDSignal(a * u + b * v, settings)

        info = info_matrix(1e-2, waveform, np.array([0.7, -1.3]), inner_product_kwargs=dict(psd=psd))

        signals = [domains.FDSignal(u, settings), domains.FDSignal(v, settings)]
        expected = np.array([[inner_product(si, sj, psd=psd) for sj in signals] for si in signals])
        np.testing.assert_allclose(info, expected, rtol=1e-10)
        np.testing.assert_allclose(info, info.T)

    def _check_array_waveform(self, force_backend):
        """Array waveforms reach inner_product as a list of channel arrays, on either array module."""
        settings, psd, (u, v) = make_setup(force_backend)
        kwargs = dict(inner_product_kwargs=dict(psd=psd, basis_settings=settings))
        from_array = info_matrix(1e-2, lambda a, b: a * u + b * v, np.array([0.7, -1.3]), **kwargs)
        from_domain = info_matrix(
            1e-2, lambda a, b: domains.FDSignal(a * u + b * v, settings), np.array([0.7, -1.3]), **kwargs
        )
        np.testing.assert_allclose(from_array, from_domain, rtol=1e-12)

    def test_cpu(self):
        self._check("cpu")
        self._check_array_waveform("cpu")

    @unittest.skipUnless(lisatools.has_backend("cuda12x"), "no cuda12x backend")
    def test_gpu(self):
        self._check("cuda12x")
        self._check_array_waveform("cuda12x")


class DomainInfoMatrixTest(unittest.TestCase):
    def setUp(self):
        self.settings, self.psd, _ = make_setup()
        self.freqs = np.arange(N_FREQ) * DF
        self.params = np.array([1.5, 0.03, 0.4, 20.0])  # amp, f0, phi, tc
        self.eps = np.array([1e-3, 1e-6, 1e-3, 1e-2])
        self.ip_kwargs = dict(psd=self.psd)

    def _packet(self, amp, f0, phi, tc):
        return np.stack([wave_packet(amp, f0, phi, tc, self.freqs), 0.5j * wave_packet(amp, f0, phi, tc, self.freqs)])

    def full_grid_waveform(self, *params):
        return domains.FDSignal(self._packet(*params), self.settings)

    def array_waveform(self, *params):
        return self._packet(*params)

    def moving_band_waveform(self, amp, f0, phi, tc):
        """The packet on its own band, which follows ``f0`` (as per-source templates do)."""
        band = domains.FDSettings(N_FREQ, DF, min_freq=f0 - 0.012, max_freq=f0 + 0.012, force_backend="cpu")
        return domains.FDSignal(self._packet(amp, f0, phi, tc)[:, band.active_slice], band)

    def test_matches_analytic_derivatives(self):
        amp, f0, phi, tc = self.params
        h = self._packet(*self.params)
        derivs = [
            h / amp,
            h * (self.freqs - f0) / 2e-3**2,
            1j * h,
            -2j * np.pi * self.freqs * h,
        ]
        derivs = [domains.FDSignal(d, self.settings) for d in derivs]
        expected = np.array([[inner_product(di, dj, psd=self.psd) for dj in derivs] for di in derivs])

        info = info_matrix(self.eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs)

        np.testing.assert_allclose(info, expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max())
        h_h = inner_product(self.full_grid_waveform(*self.params), self.full_grid_waveform(*self.params), psd=self.psd)
        np.testing.assert_allclose(info[0, 0], h_h / amp**2, rtol=1e-8)

    def test_two_point_stencil(self):
        accurate = info_matrix(self.eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs)
        simple = info_matrix(
            self.eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs, more_accurate=False
        )
        np.testing.assert_allclose(simple, accurate, rtol=1e-4, atol=1e-4 * np.abs(accurate).max())

    def test_array_and_domain_waveforms_agree(self):
        from_domain = info_matrix(self.eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs)
        from_array = info_matrix(
            self.eps,
            self.array_waveform,
            self.params,
            inner_product_kwargs=dict(psd=self.psd, basis_settings=self.settings),
        )
        np.testing.assert_allclose(from_domain, from_array, rtol=1e-12, atol=1e-12 * np.abs(from_domain).max())

    def test_moving_support_needs_basis_settings(self):
        # a step in f0 of several bins moves the template's band
        with self.assertRaisesRegex(ValueError, "basis_settings"):
            dh_dlambda(4 * DF, self.moving_band_waveform, self.params, 1)

    def test_moving_support_on_common_grid(self):
        eps = self.eps.copy()
        eps[1] = 4 * DF  # the band moves between the stencil points
        placed = info_matrix(
            eps,
            self.moving_band_waveform,
            self.params,
            inner_product_kwargs=self.ip_kwargs,
            basis_settings=self.settings,
        )
        full = info_matrix(eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs)
        # same stencil on both sides; the packet is ~1e-8 of its peak at the band edges
        np.testing.assert_allclose(placed, full, rtol=1e-6, atol=1e-6 * np.abs(full).max())

    def test_fixed_sub_band_without_basis_settings(self):
        band = domains.FDSettings(N_FREQ, DF, min_freq=0.01, max_freq=0.05, force_backend="cpu")
        band_psd = SensitivityMatrix(band, [1.0 + (band.f_arr / 0.02) ** 2] * NCHANNELS)

        def waveform(*params):
            return domains.FDSignal(self._packet(*params)[:, band.active_slice], band)

        info, derivs = info_matrix(
            self.eps, waveform, self.params, inner_product_kwargs=dict(psd=band_psd), return_derivs=True
        )
        self.assertTrue(all(d.settings == band for d in derivs))
        full = info_matrix(self.eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs)
        np.testing.assert_allclose(info, full, rtol=1e-6, atol=1e-6 * np.abs(full).max())

    def test_deriv_inds_subset(self):
        inds = np.array([0, 2])
        sub = info_matrix(
            self.eps[inds], self.full_grid_waveform, self.params, deriv_inds=inds, inner_product_kwargs=self.ip_kwargs
        )
        full = info_matrix(self.eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs)
        np.testing.assert_allclose(sub, full[np.ix_(inds, inds)], rtol=1e-12, atol=1e-12 * np.abs(full).max())

    def test_derivatives_are_domain_objects_on_the_common_grid(self):
        info, derivs = info_matrix(
            self.eps,
            self.moving_band_waveform,
            self.params,
            inner_product_kwargs=self.ip_kwargs,
            basis_settings=self.settings,
            return_derivs=True,
        )
        self.assertEqual(len(derivs), len(self.params))
        for deriv in derivs:
            self.assertIsInstance(deriv, domains.FDSignal)
            self.assertEqual(deriv.settings, self.settings)
            self.assertEqual(deriv.shape, (NCHANNELS, N_FREQ))

    def test_covariance_inverts_the_matrix(self):
        cov, info = covariance(
            self.eps,
            self.full_grid_waveform,
            self.params,
            inner_product_kwargs=self.ip_kwargs,
            return_info_mat=True,
        )
        np.testing.assert_allclose(cov, np.linalg.pinv(info))
        np.testing.assert_allclose(cov @ info, np.eye(len(self.params)), atol=1e-6)

    def test_parameter_transforms_fill_fixed_parameters(self):
        """Derivatives over (amp, phi) with f0 and tc held in the transform's fill_dict."""
        from eryn.utils import TransformContainer

        transform = TransformContainer(
            input_basis=["amp", "phi"],
            output_basis=["amp", "f0", "phi", "tc"],
            parameter_transforms={"amp": np.exp},
            fill_dict={"f0": self.params[1], "tc": self.params[3]},
        )
        point = np.array([np.log(self.params[0]), self.params[2]])
        info = info_matrix(
            self.eps[[0, 2]],
            self.full_grid_waveform,
            point,
            parameter_transforms=transform,
            inner_product_kwargs=self.ip_kwargs,
        )
        full = info_matrix(self.eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs)
        expected = full[np.ix_([0, 2], [0, 2])]
        expected[0, :] *= self.params[0]  # d/d(log amp) = amp d/d(amp)
        expected[:, 0] *= self.params[0]
        np.testing.assert_allclose(info, expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max())

    def _analytic_info(self):
        amp, f0, phi, tc = self.params
        h = self._packet(*self.params)
        derivs = [h / amp, h * (self.freqs - f0) / 2e-3**2, 1j * h, -2j * np.pi * self.freqs * h]
        derivs = [domains.FDSignal(d, self.settings) for d in derivs]
        return np.array([[inner_product(di, dj, psd=self.psd) for dj in derivs] for di in derivs])

    def test_estimated_eps_sits_between_noise_and_truncation(self):
        eps, ladders, changes = estimate_eps(
            self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs, return_changes=True
        )
        self.assertEqual(eps.shape, (4,))
        for index, (step, ladder, change) in enumerate(zip(eps, ladders, changes)):
            best = int(np.nanargmin(change))
            self.assertEqual(step, ladder[best])
            self.assertLess(change[best], 1e-12)
            if index > 0:  # the waveform is linear in amp: no truncation error, the largest step wins
                # an interior minimum: the ladder covers both regimes
                self.assertTrue(0 < best < len(change) - 1, f"minimum at the ladder's edge: {change}")

    def test_info_matrix_with_estimated_eps(self):
        info = info_matrix(estimate_eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs)
        expected = self._analytic_info()
        np.testing.assert_allclose(info, expected, rtol=1e-7, atol=1e-7 * np.abs(expected).max())

    def test_estimate_eps_for_array_waveforms_and_a_subset(self):
        kwargs = dict(inner_product_kwargs=dict(psd=self.psd, basis_settings=self.settings))
        inds = np.array([1, 3])
        from_array = estimate_eps(self.array_waveform, self.params, deriv_inds=inds, **kwargs)
        from_domain = estimate_eps(self.full_grid_waveform, self.params, deriv_inds=inds, **kwargs)
        np.testing.assert_array_equal(from_array, from_domain)
        ladders = [np.logspace(-8, -4, 5), np.logspace(-3, 1, 5)]
        custom = estimate_eps(self.full_grid_waveform, self.params, deriv_inds=inds, steps=ladders, **kwargs)
        self.assertIn(custom[0], ladders[0])
        self.assertIn(custom[1], ladders[1])

    def test_eps_hook_receives_the_call(self):
        seen = {}

        def hook(waveform_model, params, deriv_inds=None, inner_product_kwargs=None, **kwargs):
            seen.update(deriv_inds=deriv_inds, psd=inner_product_kwargs["psd"], kwargs=kwargs)
            return self.eps[deriv_inds]

        inds = np.array([0, 2])
        sub = info_matrix(
            hook, self.full_grid_waveform, self.params, deriv_inds=inds, inner_product_kwargs=self.ip_kwargs,
            more_accurate=True,
        )
        self.assertIs(seen["psd"], self.psd)
        self.assertEqual(seen["kwargs"], dict(more_accurate=True))
        full = info_matrix(self.eps, self.full_grid_waveform, self.params, inner_product_kwargs=self.ip_kwargs)
        np.testing.assert_allclose(sub, full[np.ix_(inds, inds)], rtol=1e-12, atol=1e-12 * np.abs(full).max())

    def test_estimate_eps_without_a_finite_step_raises(self):
        def waveform(*params):
            return domains.FDSignal(np.full((NCHANNELS, N_FREQ), np.nan + 0j), self.settings)

        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_eps(waveform, self.params, deriv_inds=[0], inner_product_kwargs=self.ip_kwargs)

    def test_batched_signal_is_refused(self):
        def waveform(*params):
            return domains.FDSignal(np.stack([self._packet(*params)] * 3), self.settings)

        with self.assertRaisesRegex(ValueError, "batch"):
            dh_dlambda(1e-3, waveform, self.params, 0)

    def test_mixed_domain_classes_are_refused(self):
        td = domains.TDSettings(64, 5.0, force_backend="cpu")

        def waveform(*params):
            return domains.TDSignal(np.zeros((NCHANNELS, 64)), td)

        with self.assertRaises(ValueError):
            dh_dlambda(1e-3, waveform, self.params, 0, basis_settings=self.settings)


if __name__ == "__main__":
    unittest.main()
