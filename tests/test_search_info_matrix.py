"""``spread="info_matrix"`` of the search interfaces: walkers scattered with the information-matrix covariance.

A toy two-parameter packet (amplitude, phase) on a CPU frequency grid stands in for a source; ``curr`` is a
stand-in holding only what the search interface reads.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from eryn.priors import ProbDistContainer, UniformDistribution

from lisatools import domains
from lisatools.globalfit.infomatrix import InfoMatrixGenerator
from lisatools.globalfit import searches
from lisatools.globalfit.searches import BaseSearchInterface, seed_from_search_interfaces
from tests.test_diagnostic_info_matrix import DF, N_FREQ, make_setup, wave_packet

BRANCH = "toy"
BASIS = ["amp", "phi"]
BOUNDS = [(0.5, 3.0), (0.0, 2 * np.pi)]
NDIM = len(BASIS)
# * Loud enough for the information matrix to be far narrower than the prior (sigma ~ 5e-3 and 3e-3).
LOUDNESS = 3.0e3
# ? The second source sits next to the lower edge of the periodic phase.
POINTS = np.array([[1.5, 0.4], [2.2, 0.004]])
PSD_KWARGS = dict(psd_params=np.array([1.0, 2.0]))
SEARCH_LOGGER = searches.logger.name


class IdentityTransform:
    """What the interface reads of a ``TransformContainer``, for a sampling basis equal to the physical one."""

    input_basis = BASIS
    ndim = NDIM
    ndim_full = NDIM

    @staticmethod
    def both_inverse_transforms(params):
        return params


class ToyInterface(BaseSearchInterface):
    @property
    def branch_name(self):
        return BRANCH

    @property
    def catalog_source_name(self):
        return BRANCH

    def unpack_catalog_entries(self, catalog_entry, curr):
        return np.array([catalog_entry[name] for name in BASIS])


class ToyWaveform:
    """The packet as an ``FDSignal`` of the sampling parameters; raises above ``fail_above`` in amplitude."""

    def __init__(self, settings, fail_above=None):
        self.settings = settings
        self.freqs = np.arange(N_FREQ) * DF
        self.fail_above = fail_above

    def __call__(self, amp, phi):
        if self.fail_above is not None and amp > self.fail_above:
            raise RuntimeError("the toy waveform broke")
        packet = LOUDNESS * wave_packet(amp, 0.03, phi, 20.0, self.freqs)
        return domains.FDSignal(np.stack([packet, 0.5j * packet]), self.settings)


class RecordingGenerator(InfoMatrixGenerator):
    """Keeps the ``leaf`` and the ``cap`` of every call.

    ``cap`` is recorded and not handed on: this generator has no ``distance_cap``, so no cap would run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leaves = []
        self.caps = []

    def __call__(self, point, waveform, psd, basis_settings, leaf=None, cap=False, **kwargs):
        self.leaves.append(leaf)
        self.caps.append(cap)
        return super().__call__(point, waveform, psd, basis_settings, leaf=leaf, **kwargs)


class CapReportGenerator(RecordingGenerator):
    """Sets the cap entries of ``last`` after every call: ``report`` replaces them, ``None`` removes them."""

    CAP_KEYS = ("extents", "capped", "cap_waveforms")

    def __init__(self, *args, report=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.report = report

    def __call__(self, *args, **kwargs):
        cov = super().__call__(*args, **kwargs)
        self.last = {key: value for key, value in self.last.items() if key not in self.CAP_KEYS}
        if self.report is not None:
            self.last.update(self.report)
        return cov


class RecordingBackend:
    """A sensitivity backend with the call signature of the real ones, returning one fixed matrix."""

    def __init__(self, psd):
        self.psd = psd
        self.calls = []

    def __call__(self, name, psd_params, galfor_params=None):
        self.calls.append((name, psd_params, galfor_params))
        return self.psd


def make_prior():
    return ProbDistContainer({i: UniformDistribution(lo, hi) for i, (lo, hi) in enumerate(BOUNDS)})


class SearchInfoMatrixCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.points_file = os.path.join(cls.folder.name, "points.npy")
        np.save(cls.points_file, POINTS)

    def setUp(self):
        self.settings, self.psd, _ = make_setup()
        self.prior = make_prior()
        self.priors = {BRANCH: self.prior}
        self.waveform = ToyWaveform(self.settings)
        self.gen = RecordingGenerator(self.prior)
        self.backend = RecordingBackend(self.psd)
        self.curr = self.make_curr()

    def make_curr(self, **source_overrides):
        source = dict(signal_gen=self.waveform, info_matrix_gen=self.gen, betas=np.array([1.0, 0.25]))
        source.update(source_overrides)
        general_info = SimpleNamespace(
            input_data_residual_array=SimpleNamespace(settings=self.settings),
            gpus=None,
            fixed_psd_kwargs=PSD_KWARGS,
            sensitivity_backend=self.backend,
            catalogue={BRANCH: {index: dict(zip(BASIS, point)) for index, point in enumerate(POINTS)}},
        )
        return SimpleNamespace(general_info=general_info, source_info={BRANCH: SimpleNamespace(**source)})

    def make_interface(self, search_type="point_estimate", spread="info_matrix", **kwargs):
        if search_type not in ("catalog", "prior"):
            kwargs.setdefault("search_path", self.points_file)
        kwargs.setdefault("periodic", {BRANCH: {"phi": 2 * np.pi}})
        return ToyInterface(search_type=search_type, spread=spread, transform=IdentityTransform(), **kwargs)

    def inside_prior(self, draws):
        return np.isfinite(self.prior.logpdf(draws.reshape(-1, NDIM))).all()


class ValidationTest(SearchInfoMatrixCase):
    def test_accepted_for_point_estimate_and_catalog(self):
        for search_type in ("point_estimate", "catalog"):
            self.assertEqual(self.make_interface(search_type).spread, "info_matrix")

    def test_another_string_is_refused(self):
        with self.assertRaisesRegex(ValueError, "spread.*'info_matrix'.*got the string 'covariance'"):
            self.make_interface(spread="covariance")

    def test_refused_for_other_search_types(self):
        for search_type in ("prior", "samples"):
            with self.assertRaisesRegex(ValueError, "info_matrix.*point_estimate.*catalog"):
                self.make_interface(search_type)

    def test_refused_for_a_subset_of_the_parameters(self):
        with self.assertRaisesRegex(ValueError, "info_matrix.*every parameter"):
            self.make_interface(parameters_searched=["amp"])

    def test_refused_for_reordered_parameters(self):
        with self.assertRaisesRegex(ValueError, "info_matrix.*every parameter"):
            self.make_interface(parameters_searched=["phi", "amp"])

    def test_all_parameters_by_name_are_accepted(self):
        self.make_interface(parameters_searched=list(BASIS))

    def test_numeric_spread_is_untouched(self):
        interface = self.make_interface(spread=[1e-3, 1e-3], parameters_searched=["amp"])
        self.assertEqual(interface.spread, [1e-3, 1e-3])
        covs = self.make_interface(spread=[1e-3, 2e-3]).get_covariance_matrices_from_spread(2, NDIM)
        np.testing.assert_allclose(covs, np.tile(np.diag([1e-6, 4e-6]), (2, 1, 1)))

    def test_covariance_from_spread_refuses_the_string(self):
        with self.assertRaisesRegex(ValueError, "info_matrix_covariances"):
            self.make_interface().get_covariance_matrices_from_spread(2, NDIM)


class CovariancesTest(SearchInfoMatrixCase):
    def test_shape_and_values(self):
        betas = np.array([1.0, 0.25, 0.0])
        covs = self.make_interface().info_matrix_covariances(self.curr, POINTS, betas)
        self.assertEqual(covs.shape, (len(betas), len(POINTS), NDIM, NDIM))

        direct = InfoMatrixGenerator(make_prior())
        for leaf, point in enumerate(POINTS):
            expected = direct(point, self.waveform, self.psd, self.settings, leaf=leaf, betas=betas)
            np.testing.assert_allclose(covs[:, leaf], expected, rtol=1e-12)
        # no temperature is wider than the prior, and beta = 0 is the prior
        np.testing.assert_allclose(covs[2], np.tile(np.diag(1.0 / self.gen.prior_precision), (2, 1, 1)))
        self.assertTrue((np.diagonal(covs[1], axis1=1, axis2=2) > np.diagonal(covs[0], axis1=1, axis2=2)).all())

    def test_no_ladder_is_one_temperature_at_beta_one(self):
        interface = self.make_interface()
        covs = interface.info_matrix_covariances(self.curr, POINTS, None)
        self.assertEqual(covs.shape, (1, len(POINTS), NDIM, NDIM))
        np.testing.assert_allclose(covs, interface.info_matrix_covariances(self.curr, POINTS, np.array([1.0])))

    def test_one_backend_call_and_one_generator_call_per_source(self):
        self.make_interface().info_matrix_covariances(self.curr, POINTS, np.array([1.0, 0.5]))
        self.assertEqual(len(self.backend.calls), 1)
        name, psd_params, galfor_params = self.backend.calls[0]
        self.assertEqual(name, "info_matrix")
        self.assertIs(psd_params, PSD_KWARGS["psd_params"])
        self.assertIsNone(galfor_params)
        self.assertEqual(self.gen.leaves, [0, 1])

    def test_every_source_asks_for_the_cap(self):
        self.make_interface().info_matrix_covariances(self.curr, POINTS, np.array([1.0, 0.5]))
        self.assertEqual(self.gen.leaves, [0, 1])
        self.assertEqual(self.gen.caps, [True, True])

    def source_log_lines(self, gen):
        with self.assertLogs(SEARCH_LOGGER, level="INFO") as logs:
            self.make_interface().info_matrix_covariances(self.make_curr(info_matrix_gen=gen), POINTS, None)
        lines = [line for line in logs.output if "Information-matrix covariance" in line]
        self.assertEqual(len(lines), len(POINTS))
        for leaf, line in enumerate(lines):
            self.assertIn(f"{BRANCH} source {leaf}", line)
            self.assertIn("condition number", line)
        return lines

    def test_log_line_reports_the_cap(self):
        report = dict(extents=np.array([1.0, 0.0625]), capped=1, cap_waveforms=4)
        for line in self.source_log_lines(CapReportGenerator(self.prior, report=report)):
            self.assertIn("1 directions capped", line)
            self.assertIn("smallest extent 0.0625", line)

    def test_log_line_without_a_cap_report(self):
        for line in self.source_log_lines(CapReportGenerator(self.prior)):
            self.assertNotIn("capped", line)
            self.assertNotIn("extent", line)

    def test_log_line_with_an_empty_cap_report(self):
        for line in self.source_log_lines(CapReportGenerator(self.prior, report=dict(extents=None, capped=None))):
            self.assertNotIn("capped", line)
            self.assertNotIn("extent", line)
        for line in self.source_log_lines(CapReportGenerator(self.prior, report=dict(extents=np.empty(0), capped=0))):
            self.assertIn("0 directions capped", line)
            self.assertNotIn("extent", line)

    def test_log_line_with_an_empty_last(self):
        covs = np.tile(np.eye(NDIM), (1, 1, 1))
        gen = mock.Mock(return_value=covs, last={})
        lines = self.source_log_lines(gen)
        self.assertNotIn("capped", lines[0])
        for call in gen.call_args_list:
            self.assertIs(call.kwargs["cap"], True)

    def test_missing_signal_gen(self):
        for curr in (self.make_curr(signal_gen=None), self.make_curr()):
            if curr.source_info[BRANCH].signal_gen is not None:
                del curr.source_info[BRANCH].signal_gen
            with self.assertRaisesRegex(ValueError, f"signal_gen.*{BRANCH}"):
                self.make_interface().info_matrix_covariances(curr, POINTS, None)
        self.assertEqual(self.backend.calls, [])

    def test_missing_info_matrix_gen(self):
        with self.assertRaisesRegex(ValueError, f"info_matrix_gen.*{BRANCH}"):
            self.make_interface().info_matrix_covariances(self.make_curr(info_matrix_gen=None), POINTS, None)

    def test_missing_fixed_psd_kwargs(self):
        self.curr.general_info.fixed_psd_kwargs = None
        with self.assertRaisesRegex(ValueError, "fixed_psd_kwargs"):
            self.make_interface().info_matrix_covariances(self.curr, POINTS, None)

    def test_a_waveform_failure_names_the_branch_and_the_source(self):
        curr = self.make_curr(signal_gen=ToyWaveform(self.settings, fail_above=2.0))
        with self.assertRaisesRegex(RuntimeError, rf"{BRANCH}.*source 1.*2\.2") as caught:
            self.make_interface().info_matrix_covariances(curr, POINTS, None)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn("the toy waveform broke", str(caught.exception.__cause__))

    def test_a_generator_failure_stays_a_value_error(self):
        nan_waveform = lambda amp, phi: domains.FDSignal(np.full((2, N_FREQ), np.nan + 0j), self.settings)  # noqa: E731
        with self.assertRaisesRegex(ValueError, rf"{BRANCH}.*source 0") as caught:
            self.make_interface().info_matrix_covariances(self.make_curr(signal_gen=nan_waveform), POINTS, None)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_a_failing_source_takes_the_fallback_spread(self):
        curr = self.make_curr(signal_gen=ToyWaveform(self.settings, fail_above=2.0))  # source 1 (amp 2.2) fails
        fallback = np.array([0.02, 0.3])
        interface = self.make_interface(fallback_spread=fallback)
        betas = np.array([1.0, 0.25])
        with self.assertLogs(SEARCH_LOGGER, level="WARNING") as logs:
            covs = interface.info_matrix_covariances(curr, POINTS, betas)
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertRegex(message, rf"{BRANCH}.*source 1.*2\.2")
        self.assertIn("the toy waveform broke", message)
        self.assertIn("fallback_spread", message)
        self.assertEqual(interface.fallback_leaves, [1])
        # the failing source: the numeric spread, widened by 1 / beta like any numeric spread
        np.testing.assert_allclose(covs[0, 1], np.diag(fallback**2))
        np.testing.assert_allclose(covs[1, 1], np.diag(fallback**2) / 0.25)
        # the other source keeps its information matrix
        plain = self.make_interface().info_matrix_covariances(self.make_curr(), POINTS[:1], betas)
        np.testing.assert_allclose(covs[:, 0], plain[:, 0], rtol=1e-12)
        # the next call starts a new list
        interface.info_matrix_covariances(self.make_curr(), POINTS, betas)
        self.assertEqual(interface.fallback_leaves, [])

    def test_fallback_spread_forms(self):
        curr = self.make_curr(signal_gen=ToyWaveform(self.settings, fail_above=2.0))
        full = np.array([[4e-4, 1e-4], [1e-4, 9e-2]])
        with self.assertLogs(SEARCH_LOGGER, level="WARNING"):
            scalar = self.make_interface(fallback_spread=0.1).info_matrix_covariances(curr, POINTS, None)
            matrix = self.make_interface(fallback_spread=full).info_matrix_covariances(curr, POINTS, None)
        np.testing.assert_allclose(scalar[0, 1], 0.01 * np.eye(NDIM))
        np.testing.assert_allclose(matrix[0, 1], full)

    def test_seeds_of_a_fallback_source_are_drawn(self):
        curr = self.make_curr(signal_gen=ToyWaveform(self.settings, fail_above=2.0))
        interface = self.make_interface(fallback_spread=np.array([0.02, 0.3]))
        np.random.seed(1)
        with self.assertLogs(SEARCH_LOGGER, level="WARNING"):
            draws = interface.generate_initial_draws(curr, 200, np.array([1.0, 0.25]), priors=self.priors)
        self.assertEqual(draws.shape, (2, 200, 2, NDIM))
        self.assertTrue(self.inside_prior(draws))
        np.testing.assert_allclose(draws[0, :, 1, 0].std(), 0.02, rtol=0.2)

    def test_first_device_is_selected_before_the_noise_is_built(self):
        try:
            import cupy  # noqa: F401
        except ImportError:
            self.skipTest("no cupy")
        self.curr.general_info.gpus = [7, 4]
        with mock.patch("cupy.cuda.runtime.setDevice") as set_device:
            set_device.side_effect = lambda device: self.assertEqual(self.backend.calls, [])
            self.make_interface().info_matrix_covariances(self.curr, POINTS, None)
        set_device.assert_called_once_with(7)


class DrawFromPointValueTest(SearchInfoMatrixCase):
    NUM_WALKERS = 4000

    def setUp(self):
        super().setUp()
        self.interface = self.make_interface()
        self.betas = np.array([1.0, 0.25, 0.0])
        self.covs = self.interface.info_matrix_covariances(self.curr, POINTS, self.betas)

    def draw(self, covariances, betas, num_walkers=None, seed=3):
        np.random.seed(seed)
        return self.interface.draw_from_point_value(
            starting_points=POINTS,
            covariances=covariances,
            num_walkers=self.NUM_WALKERS if num_walkers is None else num_walkers,
            priors=self.priors,
            betas=betas,
        )

    def test_temperature_axis_is_used_as_it_is(self):
        draws = self.draw(self.covs, self.betas)
        self.assertEqual(draws.shape, (len(self.betas), self.NUM_WALKERS, len(POINTS), NDIM))
        self.assertTrue(self.inside_prior(draws))

        leaf = 0  # far from every prior edge: no draw is resampled
        for temp in (0, 1):
            sample_cov = np.cov(draws[temp, :, leaf], rowvar=False)
            np.testing.assert_allclose(np.diag(sample_cov), np.diag(self.covs[temp, leaf]), rtol=0.15)
            np.testing.assert_allclose(draws[temp, :, leaf].mean(axis=0), POINTS[leaf], atol=1e-3)
        # no further 1 / beta: the spread at beta = 0.25 is covs[1], a factor 4 below covs[1] / beta
        hot = np.cov(draws[1, :, leaf], rowvar=False)
        self.assertTrue((np.diag(hot) < 2.0 * np.diag(self.covs[1, leaf])).all())
        # hotter is wider, up to the prior (beta = 0)
        variances = draws[:, :, leaf].var(axis=1)
        self.assertTrue((variances[1] > 2.0 * variances[0]).all())
        self.assertTrue((variances[2] > 100.0 * variances[1]).all())

    def test_no_ladder_takes_one_temperature(self):
        draws = self.draw(self.covs[:1], None, num_walkers=8)
        self.assertEqual(draws.shape, (1, 8, len(POINTS), NDIM))

    def test_mismatched_temperature_axis(self):
        with self.assertRaisesRegex(ValueError, "3 temperatures.*2"):
            self.draw(self.covs, self.betas[:2], num_walkers=8)
        with self.assertRaisesRegex(ValueError, "3 temperatures.*1"):
            self.draw(self.covs, None, num_walkers=8)

    def test_mismatched_leaves_and_dimensions(self):
        with self.assertRaisesRegex(ValueError, "covariances"):
            self.draw(self.covs[:, :1], self.betas, num_walkers=8)
        with self.assertRaisesRegex(ValueError, "covariances"):
            self.draw(self.covs[0, 0], self.betas, num_walkers=8)

    def test_per_leaf_covariances_are_divided_by_beta_as_before(self):
        """The 3-D form: the draws of the formula ``cov / max(beta, 1e-10)``, in the same random order."""
        num_walkers = 16
        covariances = np.array([np.diag([1e-6, 4e-6]), np.diag([9e-6, 1e-6])])
        centers = np.array([[1.5, 0.4], [2.2, 3.0]])  # no draw leaves the prior: nothing is resampled
        betas = np.array([1.0, 0.25])

        np.random.seed(11)
        draws = self.interface.draw_from_point_value(
            starting_points=centers, covariances=covariances, num_walkers=num_walkers, priors=self.priors, betas=betas
        )

        np.random.seed(11)
        expected = self.prior.rvs(size=(len(betas), num_walkers, len(centers)))
        for leaf in range(len(centers)):
            for temp, beta in enumerate(betas):
                scatter = np.random.multivariate_normal(centers[leaf], covariances[leaf] / max(beta, 1e-10), size=num_walkers)
                self.prior.rvs(size=(num_walkers,))  # the filler of the columns that are not searched
                expected[temp, :, leaf] = scatter
        np.testing.assert_array_equal(draws, expected)

        # the temperature axis with the same matrices gives the same draws
        np.random.seed(11)
        with_axis = self.interface.draw_from_point_value(
            starting_points=centers,
            covariances=covariances[None] / betas[:, None, None, None],
            num_walkers=num_walkers,
            priors=self.priors,
            betas=betas,
        )
        np.testing.assert_array_equal(with_axis, expected)


class LeafPriorsTest(SearchInfoMatrixCase):
    """``leaf_priors``: the walkers of a leaf are redrawn into the leaf's own prior."""

    TILES = [[(1.0, 2.0), (0.0, 2 * np.pi)], [(1.9, 2.5), (0.0, 2 * np.pi)]]

    def tiles(self):
        from eryn.priors import ProbDistContainer, UniformDistribution

        return [ProbDistContainer({i: UniformDistribution(lo, hi) for i, (lo, hi) in enumerate(b)}) for b in self.TILES]

    def draw(self, interface, spread=(0.3, 0.05), seed=4):
        np.random.seed(seed)
        covs = np.tile(np.diag(np.array(spread) ** 2), (2, 1, 1))
        return interface.draw_from_point_value(
            starting_points=POINTS, covariances=covs, num_walkers=500, priors=self.priors, betas=np.array([1.0, 0.5])
        )

    def test_each_leaf_stays_inside_its_own_prior(self):
        interface = self.make_interface(spread=np.array([0.3, 0.05]), leaf_priors=self.tiles())
        draws = self.draw(interface)
        for leaf, bounds in enumerate(self.TILES):
            amp = draws[:, :, leaf, 0]
            self.assertTrue(np.all((amp > bounds[0][0]) & (amp < bounds[0][1])), leaf)
        # the scatter is wider than the tiles: the redraws were needed, and are recorded
        self.assertEqual(interface.resample_passes.shape, (2, 2))
        self.assertTrue(np.all(interface.resample_passes > 0))
        self.assertTrue(np.all(interface.resample_passes < interface.max_resample_tries))

    def test_without_leaf_priors_the_branch_prior_bounds_the_draws(self):
        interface = self.make_interface(spread=np.array([0.3, 0.05]))
        draws = self.draw(interface)
        self.assertTrue(self.inside_prior(draws))
        amp = draws[:, :, 1, 0]  # the second point is at 2.2: part of the draws are outside its tile
        self.assertTrue(np.any(amp < 1.9) and np.any(amp > 2.5))

    def test_number_of_leaf_priors_must_match(self):
        interface = self.make_interface(spread=np.array([0.3, 0.05]), leaf_priors=self.tiles()[:1])
        with self.assertRaisesRegex(ValueError, "leaf_priors holds 1 priors for the 2 leaves"):
            self.draw(interface)

    def test_point_outside_its_tile_exhausts_the_redraws(self):
        tiles = self.tiles()
        interface = self.make_interface(spread=np.array([1e-3, 0.05]), leaf_priors=[tiles[1], tiles[1]])
        with self.assertRaisesRegex(RuntimeError, "leaf=0.*outside prior support"):
            self.draw(interface, spread=(1e-3, 0.05))  # the first point (amp 1.5) is not in the second tile


class EndToEndTest(SearchInfoMatrixCase):
    NUM_WALKERS = 500

    def check_draws(self, draws, betas):
        self.assertEqual(draws.shape, (len(betas), self.NUM_WALKERS, len(POINTS), NDIM))
        self.assertTrue(self.inside_prior(draws))
        cold = draws[0]
        # centred on the starting point (the second source sits on a prior edge: its phase is one-sided)
        np.testing.assert_allclose(cold[:, 0].mean(axis=0), POINTS[0], atol=2e-3)
        self.assertTrue((np.abs(cold - POINTS[None]) < 0.05).all())
        covs = InfoMatrixGenerator(make_prior())(POINTS[0], self.waveform, self.psd, self.settings)
        np.testing.assert_allclose(cold[:, 0].var(axis=0), np.diag(covs), rtol=0.3)

    def test_generate_initial_draws(self):
        betas = np.array([1.0, 0.25])
        for search_type in ("point_estimate", "catalog"):
            np.random.seed(5)
            draws = self.make_interface(search_type).generate_initial_draws(
                self.curr, self.NUM_WALKERS, betas, priors=self.priors
            )
            self.check_draws(draws, betas)
        self.assertEqual(len(self.backend.calls), 2)  # once per seeding

    def test_seeding_fills_the_state(self):
        ntemps, nleaves_max = 3, 4
        state = SimpleNamespace(
            branches_coords={BRANCH: np.zeros((ntemps, self.NUM_WALKERS, nleaves_max, NDIM))},
            branches_inds={BRANCH: np.zeros((ntemps, self.NUM_WALKERS, nleaves_max), dtype=bool)},
        )
        np.random.seed(5)
        seed_from_search_interfaces(self.curr, state, {BRANCH: self.make_interface()}, self.priors, ntemps)

        inds = state.branches_inds[BRANCH]
        self.assertTrue(inds[:, :, : len(POINTS)].all())
        self.assertFalse(inds[:, :, len(POINTS):].any())
        # the branch ladder (1, 0.25) is padded with its last rung
        self.check_draws(state.branches_coords[BRANCH][:, :, : len(POINTS)], np.array([1.0, 0.25, 0.25]))
        self.assertEqual(self.gen.leaves, [0, 1])
        self.assertEqual(self.gen.caps, [True, True])

    def test_numeric_spread_does_not_touch_the_generator(self):
        np.random.seed(5)
        draws = self.make_interface(spread=[1e-3, 1e-3]).generate_initial_draws(
            self.curr, 8, np.array([1.0, 0.5]), priors=self.priors
        )
        self.assertEqual(draws.shape, (2, 8, len(POINTS), NDIM))
        self.assertEqual(self.gen.leaves, [])
        self.assertEqual(self.backend.calls, [])


class GBWithoutSignalGenTest(unittest.TestCase):
    """The GB branch has no ``signal_gen``: the string ends in the error that says so."""

    def test_missing_signal_gen(self):
        from ereborpriors.stock.mojitolight import GBPrior

        from lisatools.globalfit.searches import GBSearchInterface
        from tests.test_gb_search_interface import CANDIDATE_COLUMNS, write_table

        prior = GBPrior.prior_factory()
        with tempfile.TemporaryDirectory() as folder:
            interface = GBSearchInterface(
                search_type="point_estimate",
                search_path=write_table(os.path.join(folder, "gb_candidates.hdf5"), CANDIDATE_COLUMNS),
                spread="info_matrix",
                transform=prior.get_transforms(),
                periodic={"gb": prior.get_periodic()},
                evolve_to_data_start=False,
                seed_phase_copies=True,
            )
            curr = SimpleNamespace(
                general_info=SimpleNamespace(),
                source_info={"gb": SimpleNamespace(signal_gen=None, info_matrix_gen=object())},
            )
            with self.assertRaisesRegex(ValueError, "signal_gen.*gb"):
                interface.generate_initial_draws(curr, 4, np.array([1.0, 0.5]), priors={"gb": prior})


if __name__ == "__main__":
    unittest.main()
