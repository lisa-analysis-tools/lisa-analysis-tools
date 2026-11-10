"""Periodic columns of the per-band GMM fit and of ``FullGaussianMixtureModel``.

A periodic column (an angle) is fitted inside a window of one period around its circular mean. The model takes its
draws of that column modulo the period, and its density, the density of those draws, sums each component over the
period images of the point. The tests compare the density with a brute-force sum of the full Gaussian over a wide
block of images, the draws with the density's own marginals, and the fit of a mode across the ends of the period with
the mode itself. Without periodic columns the model keeps its compiled path. On the CPU, and on a GPU when present.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from scipy import integrate, stats

TWO_PI = 2.0 * np.pi
# * The mixture columns of the F-statistic move's RJ proposal: ln A, f0, fdot, theta+, cos iota, theta-, lam, sin beta,
# * with theta+, theta- and lam periodic.
NDIM = 8
PERIODIC = {3: TWO_PI, 5: TWO_PI, 6: TWO_PI}
F0_COLUMN = 1
# FullGaussianMixtureModel's culling limit in the mapped basis.
CULL_LIMIT = 10.0
# Images per side of the brute-force sum: far more than the model's ring of 12 conditional standard deviations needs.
BRUTE_FORCE_BLOCK = 7


def gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def random_domain(rng, n_components, f0_window, periodic_width, wide_periodic):
    """One sub-domain: production's seven component entries (mapped-basis covariances) and its box.

    The periodic columns get a window of ``periodic_width`` (at most one period) at a random place; with
    ``wide_periodic`` their components are broad against the period, so many images count.
    """
    mins = rng.uniform(-1.0, 0.0, NDIM)
    maxs = mins + rng.uniform(0.5, 2.0, NDIM)
    mins[F0_COLUMN], maxs[F0_COLUMN] = f0_window
    for column in PERIODIC:
        start = rng.uniform(0.0, TWO_PI)
        mins[column], maxs[column] = start, start + periodic_width
    weights = rng.dirichlet(np.ones(n_components))
    means = rng.uniform(-0.5, 0.5, (n_components, NDIM))
    covs = np.empty((n_components, NDIM, NDIM))
    for component in range(n_components):
        lower = np.tril(rng.normal(0.0, 0.08, (NDIM, NDIM)), -1)
        diagonal = rng.uniform(0.05, 0.3, NDIM)
        if wide_periodic:
            diagonal[list(PERIODIC)] = rng.uniform(0.4, 0.8, len(PERIODIC))
        lower += np.diag(diagonal)
        covs[component] = lower @ lower.T
    return [weights, means, covs, np.linalg.inv(covs), np.linalg.det(covs), mins, maxs]


def model_from_domains(domains, use_cupy=False, periodic=PERIODIC):
    from lisatools.sampling.prior import FullGaussianMixtureModel

    components = [[domain[entry] for domain in domains] for entry in range(7)]
    return FullGaussianMixtureModel(*components, limit=CULL_LIMIT, use_cupy=use_cupy, periodic=periodic)


def brute_force_logpdf(points, domains, periodic):
    """ln q by brute force: every component's full Gaussian at every image of the point in a block of
    BRUTE_FORCE_BLOCK periods per side and periodic column, with the model's f0 culling and equal domain weights."""
    columns = sorted(periodic)
    periods = np.array([periodic[column] for column in columns])
    offsets = np.arange(-BRUTE_FORCE_BLOCK, BRUTE_FORCE_BLOCK + 1)
    shifts = np.stack(np.meshgrid(*[offsets] * len(columns), indexing="ij"), axis=-1).reshape(-1, len(columns))
    out = np.full(len(points), -np.inf)
    for weights, means, covs, _, _, mins, maxs in domains:
        ndim = len(mins)
        scale = 2.0 / (maxs - mins)
        log_jacobian = ndim * np.log(2.0) - np.sum(np.log(maxs - mins))
        f0_low = mins[F0_COLUMN] + 0.5 * (1.0 - CULL_LIMIT) * (maxs[F0_COLUMN] - mins[F0_COLUMN])
        f0_high = mins[F0_COLUMN] + 0.5 * (1.0 + CULL_LIMIT) * (maxs[F0_COLUMN] - mins[F0_COLUMN])
        inside = (points[:, F0_COLUMN] >= f0_low) & (points[:, F0_COLUMN] <= f0_high)
        images = np.repeat(points[:, None, :], len(shifts), axis=1)
        images[:, :, columns] += shifts * periods
        mapped = (images - mins) * scale - 1.0
        terms = []
        for weight, mean, cov in zip(weights, means, covs):
            delta = mapped - mean
            mahalanobis = np.einsum("psi,ij,psj->ps", delta, np.linalg.inv(cov), delta)
            log_norm = -0.5 * ndim * np.log(TWO_PI) - 0.5 * np.linalg.slogdet(cov)[1]
            terms.append(np.log(weight / len(domains)) + log_norm + log_jacobian - 0.5 * mahalanobis)
        terms = np.concatenate(terms, axis=1)
        peak = terms.max(axis=1)
        domain_value = peak + np.log(np.exp(terms - peak[:, None]).sum(axis=1))
        out = np.logaddexp(out, np.where(inside, domain_value, -np.inf))
    return out


def assert_densities_agree(test, values, reference, label):
    """C1's limits: 1e-10 absolute where ln q > -50, 1e-10 relative to |ln q| below, the same -inf pattern."""
    test.assertTrue(np.array_equal(np.isneginf(values), np.isneginf(reference)), f"{label}: -inf pattern differs")
    finite = np.isfinite(reference)
    difference = np.abs(values[finite] - reference[finite])
    limit = 1e-10 * np.maximum(1.0, np.abs(reference[finite]))
    worst = int(np.argmax(difference / limit)) if difference.size else 0
    test.assertTrue(np.all(difference <= limit),
                    f"{label}: largest difference {difference.max():.3e} at ln q = {reference[finite][worst]:.3e}")


class PeriodicMixtureDensityTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        # * two neighbouring bands: one with narrow phase windows, one whose windows span the whole period
        self.domains = [
            random_domain(rng, 3, (2.8300, 2.8340), periodic_width=1.0, wide_periodic=False),
            random_domain(rng, 4, (2.8340, 2.8380), periodic_width=TWO_PI, wide_periodic=True),
        ]
        self.model = model_from_domains(self.domains)
        np.random.seed(12)
        draws = np.asarray(self.model.rvs(size=(150,)))
        # the same draws at other images, uniform points of each box (tails) and points beyond the f0 culling
        shifted = draws.copy()
        for column in PERIODIC:
            shifted[:, column] += TWO_PI * rng.integers(-2, 3, len(shifted))
        uniform = np.concatenate([rng.uniform(domain[5], domain[6], (40, NDIM)) for domain in self.domains])
        beyond = uniform[:5].copy()
        beyond[:, F0_COLUMN] = 3.5
        self.points = np.concatenate([draws, shifted, uniform, beyond])

    def test_draws_lie_in_one_period(self):
        np.random.seed(13)
        draws = np.asarray(self.model.rvs(size=(2000,)))
        for column, period in PERIODIC.items():
            # ! a value a hair below zero can round to the period itself after the modulo
            self.assertTrue(np.all((draws[:, column] >= 0.0) & (draws[:, column] <= period)))

    def test_density_matches_brute_force_image_sum(self):
        values = np.asarray(self.model.logpdf(self.points.copy()))
        reference = brute_force_logpdf(self.points, self.domains, PERIODIC)
        assert_densities_agree(self, values, reference, "CPU")

    def test_density_is_periodic_in_each_periodic_column(self):
        values = np.asarray(self.model.logpdf(self.points.copy()))
        for column, period in PERIODIC.items():
            moved = self.points.copy()
            moved[:, column] += period
            assert_densities_agree(self, np.asarray(self.model.logpdf(moved)), values, f"column {column} + period")

    @unittest.skipUnless(gpu_available(), "needs a GPU")
    def test_gpu_density_matches_cpu(self):
        import cupy

        gpu_model = model_from_domains(self.domains, use_cupy=True)
        values = cupy.asnumpy(gpu_model.logpdf(cupy.asarray(self.points)))
        assert_densities_agree(self, values, np.asarray(self.model.logpdf(self.points.copy())), "GPU against CPU")


class PeriodicMixtureDrawsTest(unittest.TestCase):
    """Two columns, an angle and f0: the draws follow the density's own marginals, integrated on a grid."""

    def test_draws_follow_the_density(self):
        weights = np.array([0.6, 0.4])
        # * mapped basis; the second component is broad against the period (one period = 2 in the mapped basis here)
        means = np.array([[0.3, -0.2], [-0.4, 0.3]])
        covs = np.array([[[0.05, 0.01], [0.01, 0.04]], [[0.8, -0.05], [-0.05, 0.09]]])
        mins, maxs = np.array([1.0, 2.830]), np.array([1.0 + TWO_PI, 2.834])
        domain = [weights, means, covs, np.linalg.inv(covs), np.linalg.det(covs), mins, maxs]
        model = model_from_domains([domain], periodic={0: TWO_PI})
        np.random.seed(21)
        draws = np.asarray(model.rvs(size=(100000,)))

        angles = np.linspace(0.0, TWO_PI, 721)
        f0_values = np.linspace(2.8265, 2.8375, 1401)
        grid = np.stack(np.meshgrid(angles, f0_values, indexing="ij"), axis=-1).reshape(-1, 2)
        density = np.exp(np.asarray(model.logpdf(grid))).reshape(len(angles), len(f0_values))
        total = integrate.trapezoid(integrate.trapezoid(density, f0_values, axis=1), angles)
        self.assertLess(abs(total - 1.0), 1e-4)
        for column, axis_values, marginal in ((0, angles, integrate.trapezoid(density, f0_values, axis=1)),
                                              (1, f0_values, integrate.trapezoid(density, angles, axis=0))):
            cdf = integrate.cumulative_trapezoid(marginal, axis_values, initial=0.0)
            cdf /= cdf[-1]
            result = stats.kstest(draws[:, column], lambda value: np.interp(value, axis_values, cdf))
            self.assertGreater(result.pvalue, 0.01, f"column {column}: KS p = {result.pvalue:.3g}")


class PeriodicFitTest(unittest.TestCase):
    """The periodic fit is the plain fit of the samples moved into one window around each group's circular mean.

    The component count is not checked against the number of modes: production's choice of the count (BIC on draws of
    each fit itself) picks one to five components for one Gaussian mode far from the ends of the period as well.
    """

    def samples(self, rng):
        # * two groups of one Gaussian mode each, with the angle's mode 0.02 rad from zero, so its draws wrap
        angle = np.mod(rng.normal(0.02, 0.15, (2, 3000)), TWO_PI)
        f0 = rng.normal(2.832, 1e-4, (2, 3000))
        other = rng.normal(0.0, 1.0, (2, 3000))
        return np.stack([angle, f0, other], axis=-1)

    def test_unwrap_puts_a_mode_across_the_ends_into_one_window(self):
        from lisatools.sampling.gmm import unwrap_periodic_columns

        wrapped = self.samples(np.random.default_rng(32))
        unwrapped = unwrap_periodic_columns(wrapped, {0: TWO_PI})
        self.assertGreater(np.ptp(wrapped[:, :, 0], axis=1).min(), 6.0)
        self.assertLess(np.ptp(unwrapped[:, :, 0], axis=1).max(), 2.0)
        # the same angles on the circle, the other columns untouched, the window centred on the mode
        self.assertLess(np.abs(np.angle(np.exp(1j * (unwrapped[:, :, 0] - wrapped[:, :, 0])))).max(), 1e-12)
        self.assertTrue(np.array_equal(unwrapped[:, :, 1:], wrapped[:, :, 1:]))
        standard_error = 0.15 / np.sqrt(wrapped.shape[1])
        self.assertLess(np.abs(unwrapped[:, :, 0].mean(axis=1) - 0.02).max(), 4.0 * standard_error)

    def check_fit(self, gpu):
        from lisatools.sampling.gmm import unwrap_periodic_columns, vec_fit_gmm_min_bic
        from lisatools.utils.utility import asnumpy

        xp = np
        if gpu is not None:
            import cupy

            xp = cupy
        samples = xp.asarray(self.samples(np.random.default_rng(33)))
        fits = []
        for data, periodic in ((samples, {0: TWO_PI}), (unwrap_periodic_columns(samples, {0: TWO_PI}, xp), None)):
            np.random.seed(34)
            if gpu is not None:
                xp.random.seed(34)
            fits.append(vec_fit_gmm_min_bic(data, max_comp=6, gpu=gpu, return_components=True, periodic=periodic))
        for entry_periodic, entry_by_hand in zip(*fits):
            for group in range(2):
                self.assertTrue(np.array_equal(asnumpy(entry_periodic[group]), asnumpy(entry_by_hand[group])))

    def test_fit_is_the_fit_of_the_unwrapped_samples(self):
        self.check_fit(gpu=None)

    @unittest.skipUnless(gpu_available(), "needs a GPU")
    def test_fit_on_the_gpu(self):
        import cupy

        self.check_fit(gpu=int(cupy.cuda.runtime.getDevice()))


class NonPeriodicPathTest(unittest.TestCase):
    """Without periodic columns the model never enters the periodic density."""

    def test_six_columns_keep_the_compiled_path(self):
        from lisatools.sampling.prior import FullGaussianMixtureModel

        rng = np.random.default_rng(41)
        covs = np.array([np.eye(6) * 0.05 + 0.01 for _ in range(2)])
        domain = [np.array([0.5, 0.5]), rng.uniform(-0.3, 0.3, (2, 6)), covs, np.linalg.inv(covs),
                  np.linalg.det(covs), np.zeros(6), np.ones(6)]
        model = FullGaussianMixtureModel(*[[entry] for entry in domain])
        self.assertIsNone(model.periodic)
        points = rng.uniform(0.0, 1.0, (200, 6))
        with mock.patch.object(FullGaussianMixtureModel, "_logpdf_periodic", side_effect=AssertionError):
            values = np.asarray(model.logpdf(points))
        assert_densities_agree(self, values, np.asarray(model._logpdf_xp(points)), "compiled against xp mirror")


if __name__ == "__main__":
    unittest.main()
