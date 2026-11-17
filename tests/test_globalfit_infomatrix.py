"""``InfoMatrixGenerator`` (:mod:`lisatools.globalfit.infomatrix`): cached steps, prior-regularised covariance."""

import copy
import unittest

import numpy as np
from eryn.priors import ProbDistContainer, UniformDistribution

from lisatools import domains
from lisatools.diagnostic import covariance, info_matrix, inner_product
from lisatools.globalfit.infomatrix import (
    PROPOSAL_BASES, InfoMatrixGenerator, ProposalBasis, _PriorSupport, chirp_basis,
)
from tests.test_diagnostic_info_matrix import DF, N_FREQ, make_setup, wave_packet

NDIM = 4
LADDER = 10  # default ladder length of estimate_eps
STENCIL = 4  # waveform calls per derivative with more_accurate
# f0 = 0.03 in a prior up to 0.04: the stencil of the 1e-2 step leaves the prior, and that step is skipped
ESTIMATE_CALLS = (NDIM * LADDER - 1) * STENCIL
INFO_CALLS = NDIM * STENCIL
BOUNDS = [(0.5, 3.0), (0.02, 0.04), (0.0, 2 * np.pi), (0.0, 40.0)]  # amp, f0, phi, tc


class CountingWaveform:
    """The toy packet as an ``FDSignal``, counting its calls."""

    def __init__(self, settings, nan=False):
        self.settings = settings
        self.freqs = np.arange(N_FREQ) * DF
        self.calls = 0
        self.nan = nan

    def __call__(self, amp, f0, phi, tc):
        self.calls += 1
        packet = wave_packet(amp, f0, phi, tc, self.freqs)
        arr = np.stack([packet, 0.5j * packet])
        if self.nan:
            arr = np.full_like(arr, np.nan)
        return domains.FDSignal(arr, self.settings)


def make_prior():
    return ProbDistContainer({i: UniformDistribution(lo, hi) for i, (lo, hi) in enumerate(BOUNDS)})


class InfoMatrixGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.settings, self.psd, _ = make_setup()
        self.point = np.array([1.5, 0.03, 0.4, 20.0])
        self.waveform = CountingWaveform(self.settings)
        self.gen = InfoMatrixGenerator(make_prior())

    def call(self, gen=None, point=None, waveform=None, **kwargs):
        gen = self.gen if gen is None else gen
        point = self.point if point is None else point
        waveform = self.waveform if waveform is None else waveform
        return gen(point, waveform, self.psd, self.settings, **kwargs)

    def calls_of(self, **kwargs):
        before = self.waveform.calls
        self.call(**kwargs)
        return self.waveform.calls - before

    def test_result_is_the_regularised_covariance_at_the_cached_steps(self):
        cov = self.call()
        self.assertEqual(cov.shape, (NDIM, NDIM))
        eps = self.gen.last["eps"]
        info = info_matrix(
            eps, self.waveform, self.point, inner_product_kwargs=dict(psd=self.psd), basis_settings=self.settings
        )
        np.testing.assert_array_equal(self.gen.last["info_matrix"], info)
        np.testing.assert_array_equal(cov, covariance(info_mat=info, prior_precision=self.gen.prior_precision))
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0.0)

    def test_steps_are_estimated_once_per_leaf(self):
        self.assertEqual(self.calls_of(leaf=0), ESTIMATE_CALLS + INFO_CALLS)
        self.assertTrue(self.gen.last["estimated"])
        first = self.gen.last["eps"].copy()
        self.assertEqual(self.calls_of(leaf=0), INFO_CALLS)
        self.assertFalse(self.gen.last["estimated"])
        np.testing.assert_array_equal(self.gen.last["eps"], first)
        self.assertEqual(self.calls_of(leaf=0), INFO_CALLS)
        # a new leaf (None is one too) estimates again
        self.assertEqual(self.calls_of(leaf=1), ESTIMATE_CALLS + INFO_CALLS)
        self.assertEqual(self.calls_of(), ESTIMATE_CALLS + INFO_CALLS)
        self.assertEqual(self.calls_of(), INFO_CALLS)
        self.assertEqual(self.calls_of(leaf=1), INFO_CALLS)

    def test_eps_every_re_estimates_on_the_right_call(self):
        self.gen = InfoMatrixGenerator(make_prior(), eps_every=2)
        costs = [self.calls_of(leaf=3) for _ in range(5)]
        estimate, reuse = ESTIMATE_CALLS + INFO_CALLS, INFO_CALLS
        self.assertEqual(costs, [estimate, reuse, estimate, reuse, estimate])

    def test_clear_forces_re_estimation(self):
        for leaf in (0, 1):
            self.call(leaf=leaf)
        self.gen.clear(0)
        self.assertEqual(self.calls_of(leaf=1), INFO_CALLS)
        self.assertEqual(self.calls_of(leaf=0), ESTIMATE_CALLS + INFO_CALLS)
        self.gen.clear()
        self.assertEqual(self.calls_of(leaf=0), ESTIMATE_CALLS + INFO_CALLS)
        self.assertEqual(self.calls_of(leaf=1), ESTIMATE_CALLS + INFO_CALLS)
        self.gen.clear(17)  # an unknown leaf is not an error

    def test_steps_and_more_accurate_are_forwarded(self):
        ladder = np.logspace(-7, -4, 4)
        self.gen = InfoMatrixGenerator(make_prior(), more_accurate=False, steps=ladder)
        self.assertEqual(self.calls_of(), NDIM * len(ladder) * 2 + NDIM * 2)
        self.assertTrue(np.all(np.isin(self.gen.last["eps"], ladder)))

    def test_betas_give_one_matrix_each(self):
        betas = [1.0, 0.5, 0.1]
        covs = self.call(betas=betas)
        self.assertEqual(covs.shape, (3, NDIM, NDIM))
        info, prior_precision = self.gen.last["info_matrix"], self.gen.prior_precision
        for beta, cov in zip(betas, covs):
            np.testing.assert_array_equal(cov, covariance(info_mat=beta * info, prior_precision=prior_precision))
        variances = np.array([np.diag(cov) for cov in covs])
        self.assertTrue(np.all(variances <= (1.0 / prior_precision) * (1 + 1e-9)))
        self.assertTrue(np.all(np.diff(variances, axis=0) > 0.0))
        np.testing.assert_array_equal(self.call(betas=np.array([1.0]))[0], self.call())

    def test_prior_precision_of_a_uniform_prior(self):
        np.random.seed(123)
        state = np.random.get_state()
        gen = InfoMatrixGenerator(make_prior())
        prior_precision = gen.prior_precision  # available before the first call
        self.call(gen=gen)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

        expected = np.array([12.0 / (hi - lo) ** 2 for lo, hi in BOUNDS])
        self.assertEqual(prior_precision.shape, (NDIM,))
        np.testing.assert_allclose(prior_precision, expected, rtol=0.05)
        # reproducible, whatever the global state
        np.random.seed(7)
        np.testing.assert_array_equal(InfoMatrixGenerator(make_prior()).prior_precision, prior_precision)
        self.assertFalse(
            np.array_equal(InfoMatrixGenerator(make_prior(), seed=1).prior_precision, prior_precision)
        )

    def test_nan_waveform_raises_naming_the_leaf(self):
        bad = CountingWaveform(self.settings, nan=True)
        # a new leaf: the step estimate fails
        with self.assertRaisesRegex(ValueError, "leaf 5") as caught:
            self.call(waveform=bad, leaf=5)
        self.assertIn(str(self.point[3]), str(caught.exception))
        # cached steps: the matrix is not finite
        self.call(leaf=5)
        with self.assertRaisesRegex(ValueError, "leaf 5.*not finite"):
            self.call(waveform=bad, leaf=5)
        # the cache survives the failure
        self.assertEqual(self.calls_of(leaf=5), INFO_CALLS)

    def test_deepcopy_keeps_the_cache(self):
        cov = self.call(leaf=2)
        clone = copy.deepcopy(self.gen)
        np.testing.assert_array_equal(clone.prior_precision, self.gen.prior_precision)
        before = self.waveform.calls
        np.testing.assert_array_equal(self.call(gen=clone, leaf=2), cov)
        self.assertEqual(self.waveform.calls - before, INFO_CALLS)
        # the two caches are independent
        clone.clear()
        self.assertEqual(self.calls_of(leaf=2), INFO_CALLS)

    def test_last_holds_the_diagnostics(self):
        self.assertEqual(self.gen.last, {})
        self.call(leaf=4)
        last = self.gen.last
        self.assertEqual(
            set(last),
            {
                "leaf", "basis", "point", "eps", "estimated", "info_matrix", "condition", "time",
                "extents", "capped", "cap_waveforms",
            },
        )
        np.testing.assert_array_equal(last["extents"], np.ones(NDIM))
        self.assertEqual((last["capped"], last["cap_waveforms"]), (0, 0))
        self.assertIsNone(last["basis"])
        self.assertEqual(last["leaf"], 4)
        np.testing.assert_array_equal(last["point"], self.point)
        self.assertEqual(last["eps"].shape, (NDIM,))
        self.assertIs(last["estimated"], True)
        self.assertEqual(last["info_matrix"].shape, (NDIM, NDIM))
        self.assertGreater(last["time"], 0.0)

        regularised = last["info_matrix"] + np.diag(self.gen.prior_precision)
        d = np.sqrt(np.diag(regularised))
        eigenvalues = np.linalg.eigvalsh(regularised / np.outer(d, d))
        np.testing.assert_allclose(last["condition"], eigenvalues[-1] / eigenvalues[0], rtol=1e-10)
        self.assertGreaterEqual(last["condition"], 1.0)

    def test_point_as_a_list_and_parameter_transforms(self):
        from eryn.utils import TransformContainer

        np.testing.assert_array_equal(self.call(point=list(self.point)), self.call())

        transform = TransformContainer(
            input_basis=["amp", "phi"],
            output_basis=["amp", "f0", "phi", "tc"],
            parameter_transforms={"amp": np.exp},
            fill_dict={"f0": self.point[1], "tc": self.point[3]},
        )
        prior = ProbDistContainer({0: UniformDistribution(-1.0, 1.0), 1: UniformDistribution(0.0, 2 * np.pi)})
        gen = InfoMatrixGenerator(prior)
        point = np.array([np.log(self.point[0]), self.point[2]])
        cov = self.call(gen=gen, point=point, parameter_transforms=transform)
        info = info_matrix(
            gen.last["eps"],
            self.waveform,
            point,
            inner_product_kwargs=dict(psd=self.psd),
            basis_settings=self.settings,
            parameter_transforms=transform,
        )
        np.testing.assert_array_equal(cov, covariance(info_mat=info, prior_precision=gen.prior_precision))


def _numerical_jacobian(basis, x, step=1e-6):
    """``d(proposal basis) / d(sampling basis)`` at the sampling point ``x`` (central differences)."""
    steps = step * np.maximum(np.abs(x), 1.0) * np.eye(len(x))  # relative: t_c is ~1e7
    return np.array(
        [(basis.from_sampling(x + h) - basis.from_sampling(x - h)) / (2 * h[i]) for i, h in enumerate(steps)]
    ).T


def toy_basis():
    """(amp, f0, phi, tc) -> (log amp, f0 * amp^2, phi, tc): mixes two columns, log|det| = log amp."""
    from eryn.utils import TransformContainer

    names = ["amp", "f0", "phi", "tc"]
    transform = TransformContainer(
        input_basis=names,
        output_basis=names,
        parameter_transforms={("amp", "f0"): lambda amp, f0: (np.log(amp), f0 * amp**2)},
        inverse_parameter_transforms={("amp", "f0"): lambda la, g: (np.exp(la), g * np.exp(-2 * la))},
    )
    return ProposalBasis("toy", transform, lambda x: np.log(np.asarray(x)[:, 0]))


class PriorSupportTest(InfoMatrixGeneratorTest):
    """The prior the generator hands to ``estimate_eps``: steps whose stencil leaves it are skipped."""

    def test_step_outside_the_prior_is_not_evaluated(self):
        seen = []

        def waveform(amp, f0, phi, tc):
            seen.append(f0)
            return self.waveform(amp, f0, phi, tc)

        with self.assertWarnsRegex(UserWarning, r"parameter 1: skipping the steps 0\.01 \(outside the prior\)"):
            self.call(waveform=waveform)
        low, high = BOUNDS[1]
        self.assertTrue(all(low <= f0 <= high for f0 in seen))
        self.assertLessEqual(self.gen.last["eps"][1], 1e-3)

    def test_point_too_close_to_an_edge_names_the_parameter(self):
        point = self.point.copy()
        point[1] = BOUNDS[1][1] - 1e-13  # no step of the ladder fits between the point and the edge
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "leaf None.*Parameter 1"):
            self.call(point=point)

    def test_periodic_columns_are_wrapped(self):
        class Periodic:  # the two methods of the erebor priors that the support reads
            def __init__(self, prior):
                self.prior = prior

            def get_input_basis(self):
                return ["amp", "f0", "phi", "tc"]

            def get_periodic(self):
                return {"phi": 2 * np.pi}

            def logpdf(self, points):
                return self.prior.logpdf(points)

        points = np.array([self.point, self.point, self.point])
        points[:, 2] = [-0.01, 2 * np.pi + 0.01, 0.3]
        plain = _PriorSupport(make_prior())
        self.assertEqual(plain.periodic, {})
        self.assertEqual(list(np.isfinite(plain.logpdf(points))), [False, False, True])
        wrapped = _PriorSupport(Periodic(make_prior()))
        self.assertEqual(wrapped.periodic, {2: 2 * np.pi})
        self.assertTrue(np.all(np.isfinite(wrapped.logpdf(points))))
        # the other columns are not wrapped
        points[:, 1] = BOUNDS[1][1] + 1e-3
        self.assertFalse(np.any(np.isfinite(wrapped.logpdf(points))))

    def test_support_in_a_proposal_basis_is_the_sampling_one(self):
        basis = toy_basis()
        support = _PriorSupport(make_prior(), basis)
        inside = basis.from_sampling(self.point)
        outside = basis.from_sampling(np.array([1.5, 0.05, 0.4, 20.0]))  # f0 above its prior
        self.assertEqual(list(np.isfinite(support.logpdf(np.array([inside, outside])))), [True, False])


class ChirpBasisTest(unittest.TestCase):
    NAMES = ["logM", "q", "s1z", "s2z", "d_L", "phi_ref", "cos_iota", "psi", "alpha", "sin_delta", "t_c"]

    def setUp(self):
        self.basis = chirp_basis(self.NAMES)
        rng = np.random.default_rng(3)
        low = [np.log(1e5), 1.0, -0.99, -0.99, 1.0, 0.0, -0.99, 0.0, 0.0, -0.99, 0.0]
        high = [np.log(1e8), 10.0, 0.99, 0.99, 150.0, 2 * np.pi, 0.99, 2 * np.pi, 2 * np.pi, 0.99, 6e7]
        self.x = rng.uniform(low, high, size=(50, 11))

    def test_registered_under_its_name(self):
        self.assertIs(PROPOSAL_BASES["chirp"], chirp_basis)
        self.assertEqual(self.basis.name, "chirp")

    def test_columns_are_chirp_mass_and_effective_spin(self):
        y = self.basis.from_sampling(self.x)
        total, q, s1, s2 = np.exp(self.x[:, 0]), self.x[:, 1], self.x[:, 2], self.x[:, 3]
        m1, m2 = total * q / (1 + q), total / (1 + q)  # q = m1 / m2 >= 1
        np.testing.assert_allclose(y[:, 0], np.log((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2), rtol=1e-13)
        np.testing.assert_allclose(y[:, 1], np.log(m1 / m2), rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(y[:, 2], (m1 * s1 + m2 * s2) / (m1 + m2), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(y[:, 3], 0.5 * (s1 - s2), rtol=1e-13, atol=1e-15)
        np.testing.assert_array_equal(y[:, 4:], self.x[:, 4:])

    def test_round_trip_rows_and_single_points(self):
        np.testing.assert_allclose(self.basis.to_sampling(self.basis.from_sampling(self.x)), self.x, rtol=1e-12)
        one = self.basis.from_sampling(self.x[0])
        self.assertEqual(one.shape, (11,))
        np.testing.assert_allclose(self.basis.to_sampling(one), self.x[0], rtol=1e-12)

    def test_log_jacobian_matches_the_numerical_determinant(self):
        analytic = self.basis.log_jacobian(self.x)
        self.assertEqual(analytic.shape, (50,))
        for row, value in zip(self.x[:10], analytic[:10]):
            _, logdet = np.linalg.slogdet(_numerical_jacobian(self.basis, row))
            np.testing.assert_allclose(value, logdet, atol=1e-6)

    def test_columns_are_found_by_name(self):
        names = ["d_L", "s2z", "q", "t_c", "s1z", "logM"]
        basis = chirp_basis(names)
        x = np.array([[30.0, -0.2, 3.0, 1e6, 0.5, 14.0]])
        y = basis.from_sampling(x)
        np.testing.assert_array_equal(y[:, [0, 3]], x[:, [0, 3]])
        np.testing.assert_allclose(y[0, 2], np.log(3.0))
        np.testing.assert_allclose(y[0, 4], (3.0 * 0.5 - 0.2) / 4.0)
        np.testing.assert_allclose(basis.log_jacobian(x), -np.log(6.0))

    def test_move_keeps_the_wrapped_density_with_this_basis(self):
        # the three periodic MBH columns pass through the basis: the move must not fall back to the
        # approximate (plain) density, which it would do silently
        from eryn.moves import InfoMatrixMove
        from eryn.utils import PeriodicContainer

        move = InfoMatrixMove(
            "mbh", transform=self.basis.transform, log_jacobian_fn=self.basis.log_jacobian,
            periodic=PeriodicContainer({"mbh": {5: 2 * np.pi, 7: 2 * np.pi, 8: 2 * np.pi}}),
        )
        centre = self.basis.from_sampling(self.x[0])
        move.update_proposal({0: (centre, np.diag((1e-3 * np.maximum(np.abs(centre), 1.0)) ** 2))})
        move.get_proposal({"mbh": self.x[None, :, None, :]}, np.random.RandomState(0))
        self.assertIs(move._periodic_passthrough, True)

    def test_missing_columns_are_named(self):
        with self.assertRaisesRegex(ValueError, r"\['q', 's2z'\]"):
            chirp_basis(["logM", "s1z", "d_L"])


class ProposalBasisGeneratorTest(InfoMatrixGeneratorTest):
    """The generator with a proposal basis; the inherited tests check that the sampling-basis path is unchanged."""

    def setUp(self):
        super().setUp()
        self.basis = toy_basis()
        self.plain = self.gen
        self.gen = InfoMatrixGenerator(make_prior(), proposal_basis=self.basis)

    def test_sampling_path_is_the_one_without_a_basis(self):
        np.testing.assert_array_equal(self.call(), self.call(gen=self.plain))
        np.testing.assert_array_equal(self.gen.prior_precision, self.plain.prior_precision)

    def test_matrix_in_the_proposal_basis_is_the_congruent_one(self):
        self.call()
        sampling = self.gen.last["info_matrix"]
        cov = self.call(in_proposal_basis=True)
        last = self.gen.last
        self.assertEqual(last["basis"], "toy")
        np.testing.assert_allclose(last["point"], self.basis.from_sampling(self.point), rtol=1e-14)

        inverse_jacobian = np.linalg.inv(_numerical_jacobian(self.basis, self.point))
        expected = inverse_jacobian.T @ sampling @ inverse_jacobian
        scale = np.sqrt(np.diag(expected))
        np.testing.assert_allclose(last["info_matrix"] / np.outer(scale, scale), expected / np.outer(scale, scale), atol=2e-4)
        np.testing.assert_array_equal(
            cov, covariance(info_mat=last["info_matrix"], prior_precision=self.gen.prior_precision_proposal)
        )

    def test_prior_precision_is_that_of_the_mapped_draws(self):
        state = np.random.get_state()
        np.random.seed(0)
        draws = make_prior().rvs(size=(10000,))
        np.random.set_state(state)
        np.testing.assert_allclose(
            self.gen.prior_precision_proposal, 1.0 / self.basis.from_sampling(draws).var(axis=0), rtol=1e-12
        )
        # the two columns the basis leaves alone keep the sampling-basis precision
        np.testing.assert_array_equal(self.gen.prior_precision_proposal[2:], self.gen.prior_precision[2:])
        self.assertFalse(np.allclose(self.gen.prior_precision_proposal[:2], self.gen.prior_precision[:2]))

    def test_each_basis_has_its_own_steps(self):
        # in the toy basis f0 * amp^2 = 0.0675 +- 0.02 maps back inside the f0 prior: no step is skipped there
        in_basis = NDIM * LADDER * STENCIL
        self.assertEqual(self.calls_of(leaf=0), ESTIMATE_CALLS + INFO_CALLS)
        sampling_steps = self.gen.last["eps"].copy()
        # the steps estimated in the sampling basis are not used for the proposal basis
        self.assertEqual(self.calls_of(leaf=0, in_proposal_basis=True), in_basis + INFO_CALLS)
        self.assertTrue(self.gen.last["estimated"])
        self.assertEqual(self.calls_of(leaf=0, in_proposal_basis=True), INFO_CALLS)
        self.assertEqual(self.calls_of(leaf=0), INFO_CALLS)
        np.testing.assert_array_equal(self.gen.last["eps"], sampling_steps)
        self.gen.clear(0)
        self.assertEqual(self.calls_of(leaf=0, in_proposal_basis=True), in_basis + INFO_CALLS)
        self.assertEqual(self.calls_of(leaf=0), ESTIMATE_CALLS + INFO_CALLS)

    def test_betas_in_the_proposal_basis(self):
        covs = self.call(in_proposal_basis=True, betas=[1.0, 0.25])
        info, precision = self.gen.last["info_matrix"], self.gen.prior_precision_proposal
        np.testing.assert_array_equal(covs[1], covariance(info_mat=0.25 * info, prior_precision=precision))

    def test_parameter_transforms_follow_the_basis(self):
        from eryn.utils import TransformContainer

        # sampling basis (log amp, f0, phi, tc); the waveform takes amp
        to_waveform = TransformContainer(
            input_basis=["amp", "f0", "phi", "tc"], output_basis=["amp", "f0", "phi", "tc"],
            parameter_transforms={"amp": np.exp},
        )
        basis = ProposalBasis(
            "double",
            TransformContainer(
                input_basis=["amp", "f0", "phi", "tc"], output_basis=["amp", "f0", "phi", "tc"],
                parameter_transforms={"amp": lambda v: 2.0 * v}, inverse_parameter_transforms={"amp": lambda v: 0.5 * v},
            ),
            lambda x: np.full(len(x), np.log(2.0)),
        )
        prior = ProbDistContainer(
            {i: UniformDistribution(lo, hi) for i, (lo, hi) in enumerate([(-1.0, 1.5)] + BOUNDS[1:])}
        )
        gen = InfoMatrixGenerator(prior, proposal_basis=basis)
        point = np.array([np.log(self.point[0]), *self.point[1:]])
        self.call(gen=gen, point=point, parameter_transforms=to_waveform)
        sampling = gen.last["info_matrix"]
        self.call(gen=gen, point=point, parameter_transforms=to_waveform, in_proposal_basis=True)
        factor = np.array([0.5, 1.0, 1.0, 1.0])  # d(sampling) / d(proposal)
        scale = np.sqrt(np.diag(sampling)) * factor
        np.testing.assert_allclose(
            gen.last["info_matrix"] / np.outer(scale, scale),
            sampling * np.outer(factor, factor) / np.outer(scale, scale),
            atol=2e-4,
        )

    def test_proposal_basis_call_without_a_basis_raises(self):
        with self.assertRaisesRegex(ValueError, "proposal_basis"):
            self.call(gen=self.plain, in_proposal_basis=True)
        with self.assertRaisesRegex(ValueError, "proposal_basis"):
            self.plain.prior_precision_proposal

    def test_deep_copy_keeps_the_basis_and_both_caches(self):
        cov = self.call(leaf=1, in_proposal_basis=True)
        clone = copy.deepcopy(self.gen)
        self.assertEqual(clone.proposal_basis.name, "toy")
        before = self.waveform.calls
        np.testing.assert_array_equal(self.call(gen=clone, leaf=1, in_proposal_basis=True), cov)
        self.assertEqual(self.waveform.calls - before, INFO_CALLS)


class RecordingWaveform:
    """``fn(*params, freqs, basis signals) -> (channels, N_FREQ) array`` as an ``FDSignal``, recording its calls."""

    def __init__(self, settings, fn):
        self.settings, self.fn = settings, fn
        self.freqs = np.arange(N_FREQ) * DF
        self.seen = []

    @property
    def calls(self):
        return len(self.seen)

    def __call__(self, *params):
        self.seen.append(np.array(params, dtype=float))
        return domains.FDSignal(self.fn(*params), self.settings)


def uniform_prior(bounds):
    return ProbDistContainer({i: UniformDistribution(lo, hi) for i, (lo, hi) in enumerate(bounds)})


class DistanceCapTest(unittest.TestCase):
    """The waveform-distance cap: ``distance_cap`` on the generator and ``cap=True`` on the call."""

    TARGET = 1.0
    TOLERANCE = 1.5
    STEPS = np.logspace(-7, -3, 5)

    def setUp(self):
        self.settings, self.psd, (self.u, self.v) = make_setup()
        self.freqs = np.arange(N_FREQ) * DF

    # ---- toy waveforms

    def product_waveform(self):
        """Depends on (a, b) through a * b only: the direction a b = const is null, and curved."""

        def fn(a, b, phi):
            packet = wave_packet(4e3 * a * b, 0.03, phi, 20.0, self.freqs)
            return np.stack([packet, 0.5j * packet])

        bounds = [(0.5, 3.0), (0.5, 3.0), (0.0, 2 * np.pi)]
        return RecordingWaveform(self.settings, fn), bounds, np.array([1.5, 1.2, 0.4])

    def linear_waveform(self, scales=(100.0, 100.0), point=(0.7, -1.3)):
        fn = lambda a, b: scales[0] * a * self.u + scales[1] * b * self.v  # noqa: E731
        return RecordingWaveform(self.settings, fn), [(-5.0, 5.0), (-5.0, 5.0)], np.array(point)

    # ---- helpers

    def generator(self, bounds, **kwargs):
        return InfoMatrixGenerator(uniform_prior(bounds), steps=self.STEPS, **kwargs)

    def distance(self, waveform, point, offset):
        diff = domains.FDSignal(waveform(*(point + offset)).arr - waveform(*point).arr, self.settings)
        return inner_product(diff, diff, psd=self.psd)

    @staticmethod
    def axes(info, precision):
        """``(s, eigenvalues, V, one-sigma steps as rows)`` of ``info + diag(precision)`` scaled to unit diagonal."""
        regularised = info + np.diag(precision)
        scale = np.sqrt(np.diag(regularised))
        eigenvalues, vectors = np.linalg.eigh(regularised / np.outer(scale, scale))
        return scale, eigenvalues, vectors, (vectors / scale[:, None] / np.sqrt(eigenvalues)).T

    def capped_info(self, info, precision, extents):
        scale, eigenvalues, vectors, _ = self.axes(info, precision)
        capped = (vectors * (eigenvalues / extents**2)) @ vectors.T * np.outer(scale, scale) - np.diag(precision)
        return 0.5 * (capped + capped.T)

    # ---- tests

    def test_curved_null_direction_is_shortened_to_the_target(self):
        waveform, bounds, point = self.product_waveform()
        plain, gen = self.generator(bounds), self.generator(bounds, distance_cap=self.TARGET)
        uncapped = plain(point, waveform, self.psd, self.settings)
        precision = plain.prior_precision
        scale, eigenvalues, vectors, steps = self.axes(plain.last["info_matrix"], precision)
        # uncapped: one sigma along the weak axis is far beyond the target
        self.assertGreater(max(self.distance(waveform, point, sign * steps[0]) for sign in (1, -1)), 100.0)

        cov = gen(point, waveform, self.psd, self.settings, cap=True)
        last = gen.last
        np.testing.assert_array_equal(last["info_matrix"], plain.last["info_matrix"])
        self.assertEqual(last["condition"], plain.last["condition"])
        extents = last["extents"]
        self.assertEqual(extents.shape, (3,))
        self.assertLess(extents[0], 0.5)
        self.assertGreaterEqual(extents[0], 1e-4)
        self.assertEqual(last["capped"], int(np.sum(extents < 1.0)))
        self.assertGreater(last["capped"], 0)
        self.assertLess(last["capped"], 3)
        self.assertGreater(last["cap_waveforms"], 1 + 2 * 3)

        # every principal axis of the new covariance (in the scaling of the regularised matrix) meets the target
        # at both ends of one sigma, up to the tolerance of the axes that were left alone
        variances, directions = np.linalg.eigh(cov * np.outer(scale, scale))
        self.assertGreater(variances.min(), 0.0)
        for variance, direction in zip(variances, directions.T):
            step = direction * np.sqrt(variance) / scale
            for sign in (1, -1):
                self.assertLessEqual(self.distance(waveform, point, sign * step), self.TOLERANCE * self.TARGET + 1e-9)
        # the capped axes meet the target itself, and are shorter than before by their extent
        for k in range(3):
            along = scale * vectors[:, k]
            np.testing.assert_allclose(along @ cov @ along, extents[k] ** 2 / eigenvalues[k], rtol=1e-8)
            if extents[k] < 1.0:
                for sign in (1, -1):
                    self.assertLessEqual(
                        self.distance(waveform, point, sign * extents[k] * steps[k]), self.TARGET + 1e-9
                    )
            else:  # a well-measured axis keeps its variance
                np.testing.assert_allclose(along @ cov @ along, along @ uncapped @ along, rtol=1e-8)
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12)
        self.assertTrue(np.all(np.diag(cov) <= np.diag(uncapped) * (1 + 1e-9)))

    def test_linear_waveform_is_not_capped(self):
        waveform, bounds, point = self.linear_waveform()
        plain, gen = self.generator(bounds), self.generator(bounds, distance_cap=self.TARGET)
        uncapped = plain(point, waveform, self.psd, self.settings)
        cost = waveform.calls
        waveform.seen.clear()
        cov = gen(point, waveform, self.psd, self.settings, cap=True)
        np.testing.assert_array_equal(cov, uncapped)
        np.testing.assert_array_equal(gen.last["extents"], np.ones(2))
        self.assertEqual(gen.last["capped"], 0)
        # the waveform at the point, then two per axis
        self.assertEqual(gen.last["cap_waveforms"], 1 + 2 * 2)
        self.assertEqual(waveform.calls - cost, 1 + 2 * 2)

    def test_prior_edge_does_not_shorten_an_axis(self):
        # a is hardly measured (sigma: that of the prior, 2.9) and 0.5 from its upper edge; b is well measured.
        # Both axes carry a, so one end of each leaves the prior at one sigma. The seeds that land outside are
        # redrawn; the axes keep their length because the waveform is within the target up to the edge.
        waveform, bounds, point = self.linear_waveform(scales=(1e-2, 100.0), point=(4.5, -1.3))
        gen = self.generator(bounds, distance_cap=self.TARGET)
        uncapped = gen(point, waveform, self.psd, self.settings)
        _, _, _, steps = self.axes(gen.last["info_matrix"], gen.prior_precision)
        low, high = np.array(bounds).T
        outside = [not np.all((low <= point + sign * step) & (point + sign * step <= high)) for step in steps for sign in (1, -1)]
        self.assertTrue(any(outside))

        waveform.seen.clear()
        cov = gen(point, waveform, self.psd, self.settings, cap=True)
        self.assertEqual(gen.last["capped"], 0)
        np.testing.assert_array_equal(gen.last["extents"], np.ones(2))
        np.testing.assert_array_equal(cov, uncapped)
        # the ends outside the prior were measured at the edge: the waveform never saw a point outside it
        seen = np.array(waveform.seen)
        self.assertEqual(len(seen), gen.last["cap_waveforms"] + 2 * 2 * 2)  # the cap, and the stencils of the matrix
        self.assertEqual(gen.last["cap_waveforms"], 1 + 2 * 2)
        self.assertTrue(np.all((low <= seen) & (seen <= high)))
        self.assertTrue(np.any(np.isclose(seen[:, 0], high[0], rtol=1e-6)))  # one of them at the edge of a

    def test_waveform_shortens_an_axis_that_also_leaves_the_prior(self):
        # the curved null direction a b = const, from a point 0.05 from the upper edge of a: the axis is
        # shortened by the waveform (measured where the prior allows), not to the distance to the edge
        waveform, bounds, point = self.product_waveform()
        point = np.array([2.95, 0.61, 0.4])
        gen = self.generator(bounds, distance_cap=self.TARGET)
        gen(point, waveform, self.psd, self.settings)
        _, _, _, steps = self.axes(gen.last["info_matrix"], gen.prior_precision)
        low, high = np.array(bounds).T
        self.assertGreater(abs(steps[0][0]), 0.05)  # one sigma of the weak axis crosses the edge of a

        waveform.seen.clear()
        gen(point, waveform, self.psd, self.settings, cap=True)
        extent = gen.last["extents"][0]
        self.assertLess(extent, 1.0)
        self.assertGreater(extent * abs(steps[0][0]), 2 * 0.05)  # longer than the distance to the edge
        seen = np.array(waveform.seen)
        self.assertTrue(np.all((low <= seen) & (seen <= high)))
        for sign in (1, -1):  # the target is met wherever the shortened axis is inside the prior
            end = point + sign * extent * steps[0]
            if np.all((low <= end) & (end <= high)):
                self.assertLessEqual(self.distance(waveform, point, sign * extent * steps[0]), self.TARGET * (1 + 1e-9))

    def test_side_on_which_the_prior_ends_at_the_point_is_not_tested(self):
        waveform, bounds, point = self.linear_waveform(scales=(100.0, 100.0), point=(5.0 - 1e-13, -1.3))
        gen = self.generator(bounds, distance_cap=self.TARGET)
        # the waveform is linear: its matrix is the same everywhere, and exact at any step
        info = info_matrix(1e-3, waveform, np.array([0.7, -1.3]), inner_product_kwargs=dict(psd=self.psd))
        waveform.seen.clear()
        capped, extents, calls = gen._cap(
            info, np.zeros(2), point, waveform, lambda x: x, self.psd, self.settings, None, None
        )
        low, high = np.array(bounds).T
        seen = np.array(waveform.seen)
        self.assertTrue(np.all((low <= seen) & (seen <= high)))
        # both axes carry a (two parameters scaled to unit diagonal have the axes (1, +-1) / sqrt(2)), so each
        # is measured on its inner side only: the reference and one call per axis. Neither is shortened (the
        # waveform is linear and the matrix is its own)
        self.assertEqual(calls, 1 + 1 + 1)
        np.testing.assert_array_equal(extents, np.ones(2))
        self.assertIs(capped, info)

    def test_one_prior_per_leaf(self):
        waveform, bounds, point = self.linear_waveform(scales=(1e-2, 100.0), point=(0.7, -1.3))
        narrow = [(0.0, 1.0), (-5.0, 5.0)]
        gen = InfoMatrixGenerator([uniform_prior(bounds), uniform_prior(narrow)], steps=self.STEPS)
        self.assertTrue(gen.per_leaf_priors)
        wide_cov = gen(point, waveform, self.psd, self.settings, leaf=0)
        narrow_cov = gen(point, waveform, self.psd, self.settings, leaf=1)
        # a is unmeasured: its sigma is the one of the leaf's own prior
        np.testing.assert_allclose(np.sqrt(wide_cov[0, 0]), 10.0 / np.sqrt(12.0), rtol=0.05)
        np.testing.assert_allclose(np.sqrt(narrow_cov[0, 0]), 1.0 / np.sqrt(12.0), rtol=0.05)
        np.testing.assert_allclose(gen.prior_precisions(1)[0][0] / gen.prior_precisions(0)[0][0], 100.0, rtol=0.1)
        self.assertIs(gen.prior_of(1), gen.prior[1])
        # each leaf equals a generator holding that prior alone
        single = InfoMatrixGenerator(uniform_prior(narrow), steps=self.STEPS)
        np.testing.assert_array_equal(narrow_cov, single(point, waveform, self.psd, self.settings, leaf=1))
        # the derivative steps stay inside the leaf's own prior: a point outside it has none
        # (a fresh generator: the steps of a leaf are estimated, and checked, once)
        fresh = InfoMatrixGenerator([uniform_prior(bounds), uniform_prior(narrow)], steps=self.STEPS)
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "leaf 1"):
            fresh(np.array([2.0, -1.3]), waveform, self.psd, self.settings, leaf=1)
        fresh(np.array([2.0, -1.3]), waveform, self.psd, self.settings, leaf=0)
        for leaf in (None, 5, "a"):
            with self.assertRaisesRegex(ValueError, "one prior per leaf"):
                gen(point, waveform, self.psd, self.settings, leaf=leaf)
        clone = copy.deepcopy(gen)
        np.testing.assert_array_equal(clone(point, waveform, self.psd, self.settings, leaf=1), narrow_cov)

    def test_betas_use_the_capped_matrix(self):
        waveform, bounds, point = self.product_waveform()
        gen = self.generator(bounds, distance_cap=self.TARGET)
        betas = [1.0, 0.5, 0.05]
        covs = gen(point, waveform, self.psd, self.settings, betas=betas, cap=True)
        self.assertEqual(covs.shape, (3, 3, 3))
        self.assertGreater(gen.last["capped"], 0)
        precision = gen.prior_precision
        capped = self.capped_info(gen.last["info_matrix"], precision, gen.last["extents"])
        uncapped = covariance(info_mat=gen.last["info_matrix"], prior_precision=precision)
        self.assertLess(covs[0][0, 0], 0.1 * uncapped[0, 0])
        for beta, cov in zip(betas, covs):
            np.testing.assert_allclose(
                cov, covariance(info_mat=beta * capped, prior_precision=precision), rtol=1e-9, atol=0.0
            )
        np.testing.assert_array_equal(covs[0], gen(point, waveform, self.psd, self.settings, cap=True))
        # the capped matrix is never smaller than the matrix
        self.assertGreater(np.linalg.eigvalsh(capped - gen.last["info_matrix"]).min(), -1e-9 * np.abs(capped).max())

    def test_without_both_switches_nothing_changes(self):
        waveform, bounds, point = self.product_waveform()
        plain = self.generator(bounds)
        expected = plain(point, waveform, self.psd, self.settings, leaf=0, betas=[1.0, 0.3])
        info_calls = 3 * STENCIL
        cases = [
            (self.generator(bounds, distance_cap=None), True),
            (self.generator(bounds, distance_cap=self.TARGET), False),
        ]
        for gen, cap in cases:
            gen(point, waveform, self.psd, self.settings, leaf=0)  # the steps of the leaf
            before = waveform.calls
            covs = gen(point, waveform, self.psd, self.settings, leaf=0, betas=[1.0, 0.3], cap=cap)
            self.assertEqual(waveform.calls - before, info_calls)
            np.testing.assert_array_equal(covs, expected)
            np.testing.assert_array_equal(gen.last["extents"], np.ones(3))
            self.assertEqual((gen.last["capped"], gen.last["cap_waveforms"]), (0, 0))

    def test_distance_cap_must_be_positive(self):
        for bad in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "distance_cap"):
                InfoMatrixGenerator(make_prior(), distance_cap=bad)

    def test_deep_copy_keeps_the_cap(self):
        waveform, bounds, point = self.product_waveform()
        gen = self.generator(bounds, distance_cap=0.7)
        cov = gen(point, waveform, self.psd, self.settings, cap=True)
        clone = copy.deepcopy(gen)
        self.assertEqual(clone.distance_cap, 0.7)
        np.testing.assert_array_equal(clone(point, waveform, self.psd, self.settings, cap=True), cov)
        np.testing.assert_array_equal(clone.last["extents"], gen.last["extents"])

    def test_cap_in_the_proposal_basis(self):
        basis = toy_basis()
        waveform = CountingWaveform(self.settings)
        point = np.array([1.5, 0.03, 0.4, 20.0])
        target = 1e-4  # the packet is quiet: <h|h> is about 0.01
        gen = InfoMatrixGenerator(make_prior(), proposal_basis=basis, distance_cap=target)
        uncapped = gen(point, waveform, self.psd, self.settings, in_proposal_basis=True)
        cov = gen(point, waveform, self.psd, self.settings, in_proposal_basis=True, cap=True)
        last = gen.last
        self.assertEqual(last["basis"], "toy")
        self.assertEqual(last["extents"].shape, (NDIM,))
        self.assertGreaterEqual(last["cap_waveforms"], 1 + 2 * last["capped"])
        precision = gen.prior_precision_proposal
        np.testing.assert_allclose(
            cov,
            covariance(info_mat=self.capped_info(last["info_matrix"], precision, last["extents"]), prior_precision=precision),
            rtol=1e-9,
        )
        self.assertTrue(np.all(np.diag(cov) <= np.diag(uncapped) * (1 + 1e-9)))

        # where a shortened axis ends inside the prior, the target is met there
        centre = basis.from_sampling(point)
        _, _, _, steps = self.axes(last["info_matrix"], precision)
        low, high = np.array(BOUNDS).T
        reference = waveform(*point).arr
        self.assertGreater(last["capped"], 0)
        checked = 0
        for extent, step in zip(last["extents"], steps):
            for sign in (1, -1):
                end = basis.to_sampling(centre + sign * extent * step)
                if np.all((low <= end) & (end <= high)):
                    diff = domains.FDSignal(waveform(*end).arr - reference, self.settings)
                    limit = target if extent < 1.0 else self.TOLERANCE * target
                    self.assertLessEqual(inner_product(diff, diff, psd=self.psd), limit * (1 + 1e-9))
                    checked += 1
        self.assertGreater(checked, 0)

    def test_cap_with_parameter_transforms(self):
        from eryn.utils import TransformContainer

        fn = lambda a, b, phi: self.product_waveform()[0].fn(a, b, phi)  # noqa: E731
        waveform = RecordingWaveform(self.settings, fn)
        transform = TransformContainer(
            input_basis=["a", "b", "phi"], output_basis=["a", "b", "phi"], parameter_transforms={"a": np.exp}
        )
        bounds = [(np.log(0.5), np.log(3.0)), (0.5, 3.0), (0.0, 2 * np.pi)]
        point = np.array([np.log(1.5), 1.2, 0.4])
        gen = self.generator(bounds, distance_cap=self.TARGET)
        gen(point, waveform, self.psd, self.settings, parameter_transforms=transform, cap=True)
        self.assertGreater(gen.last["capped"], 0)
        _, _, _, steps = self.axes(gen.last["info_matrix"], gen.prior_precision)
        for extent, step in zip(gen.last["extents"], steps):
            for sign in (1, -1):
                end = point + sign * extent * step
                offset = np.array([np.exp(end[0]), *end[1:]]) - np.array([1.5, 1.2, 0.4])
                limit = self.TARGET if extent < 1.0 else self.TOLERANCE * self.TARGET
                self.assertLessEqual(self.distance(waveform, np.array([1.5, 1.2, 0.4]), offset), limit + 1e-9)

    def test_failing_waveform_counts_as_too_far_and_a_failing_point_raises(self):
        waveform, bounds, point = self.linear_waveform()
        gen = self.generator(bounds, distance_cap=self.TARGET)
        gen(point, waveform, self.psd, self.settings, leaf=8)

        def one_sided(a, b):  # fails above the point in a
            if a > point[0] + 5e-3:  # beyond the stencils of the matrix, within one sigma
                raise RuntimeError("no waveform here")
            return waveform(a, b)

        gen(point, one_sided, self.psd, self.settings, leaf=8, cap=True)
        self.assertGreater(gen.last["capped"], 0)
        self.assertTrue(np.all(gen.last["extents"] > 0.0))

        def only_near(a, b):  # the stencils move one parameter, the axes both: no extent passes, the floor
            if a != point[0] and b != point[1]:
                raise RuntimeError("no waveform here")
            return waveform(a, b)

        gen(point, only_near, self.psd, self.settings, leaf=8, cap=True)
        np.testing.assert_allclose(gen.last["extents"], 1e-4, rtol=1e-12)

        def nan_at_the_point(a, b):
            h = waveform(a, b)
            return domains.FDSignal(np.full_like(h.arr, np.nan), self.settings) if (a, b) == tuple(point) else h

        with self.assertRaisesRegex(ValueError, "distance cap of leaf 8"):
            gen(point, nan_at_the_point, self.psd, self.settings, leaf=8, cap=True)


if __name__ == "__main__":
    unittest.main()
