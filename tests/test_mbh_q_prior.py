# -*- coding: utf-8 -*-
"""The MBH mass-ratio column ``Q`` is a LINEAR-column log-uniform on [1, 10].

Regression cover for the 2026-07-05 eryn prior rewrite (eryn ``20ff728``),
which redefined ``log_uniform`` as a SAMPLING-space (``u = ln x``)
distribution. ``ProbDistContainer`` calls the plain ``rvs``/``logpdf``, so
the stock MBH ``"Q"`` column silently became log-uniform in ``ln Q`` with
support ``[0, ln 10] = [0, 2.30]``: 43% of prior draws (``Q < 1``) raised
``AssertionError`` inside ``mT_Q`` ("m1 should be the larger mass"), and
every ``Q`` in ``(2.30, 10]`` scored ``-inf``.

The fix is :class:`lisatools.sampling.prior.LogUniformLinear`, used at both
stock call sites (``stock/erebor/mbh.py`` and
``stock/erebor/source_runtime.py``). The negative control at the bottom
pins the defect: the same dict built with eryn's ``log_uniform`` still
draws ``Q < 1`` and scores ``Q = 5`` as ``-inf``.
"""

import inspect
import shutil
import tempfile
import unittest
import warnings

import numpy as np

from lisatools.globalfit.moves.eigen_refresh import prior_box_widths
from lisatools.globalfit.stock.erebor.transforms import make_mbh_transform_container
from lisatools.sampling.prior import LogUniformLinear

# numpy >= 2 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

# stock MBH settings used to build the prior (Tobs enters only the
# ``t_plunge`` box)
TOBS = 15552000.0
DT = 5.0
NDIM_MBH = 11
Q_COL = 1


def _valid_mbh_row(q):
    """An otherwise in-prior MBH sampling-basis row with ``Q = q``."""
    return np.array(
        [
            np.log(1e6),  # logM
            float(q),  # Q
            0.1,  # s1z
            0.1,  # s2z
            50.0,  # dist (Gpc)
            1.0,  # phi_ref
            0.0,  # cos_iota
            1.0,  # psi
            1.0,  # alpha
            0.0,  # sin_delta
            1.0e6,  # t_plunge
        ]
    )


class LogUniformLinearTest(unittest.TestCase):
    """Unit behavior of the distribution itself."""

    def test_rvs_stays_in_the_linear_box_and_is_log_uniform(self):
        dist = LogUniformLinear(1.0, 10.0)
        np.random.seed(20260916)
        n = 20000
        x = np.asarray(dist.rvs(n))

        self.assertEqual(x.shape, (n,))
        self.assertTrue(np.all(x >= 1.0), f"min draw {x.min()} < 1")
        self.assertTrue(np.all(x <= 10.0), f"max draw {x.max()} > 10")

        # 1/x density <=> uniform in ln x on [ln 1, ln 10]
        lo, hi = np.log(1.0), np.log(10.0)
        expected = 0.5 * (lo + hi)
        sigma_mean = (hi - lo) / np.sqrt(12.0 * n)
        self.assertLess(abs(np.mean(np.log(x)) - expected), 3.0 * sigma_mean)

    def test_logpdf_outside_the_box_is_minus_inf(self):
        dist = LogUniformLinear(1.0, 10.0)
        self.assertEqual(float(dist.logpdf(0.5)), -np.inf)
        self.assertEqual(float(dist.logpdf(10.5)), -np.inf)
        # a non-positive input must be masked, not logged
        self.assertEqual(float(dist.logpdf(0.0)), -np.inf)
        self.assertEqual(float(dist.logpdf(-3.0)), -np.inf)

    def test_logpdf_ratio_is_the_one_over_x_density(self):
        dist = LogUniformLinear(1.0, 10.0)
        diff = float(dist.logpdf(1.5)) - float(dist.logpdf(6.0))
        np.testing.assert_allclose(diff, np.log(6.0 / 1.5), rtol=1e-12)

    def test_pdf_is_exp_logpdf(self):
        dist = LogUniformLinear(1.0, 10.0)
        x = np.array([0.5, 1.0, 2.0, 9.9, 10.0, 11.0])
        np.testing.assert_allclose(
            np.asarray(dist.pdf(x)), np.exp(np.asarray(dist.logpdf(x))), rtol=1e-12
        )

    def test_density_normalizes_to_one(self):
        dist = LogUniformLinear(1.0, 10.0)
        grid = np.linspace(1.0, 10.0, 200001)
        integral = _trapz(np.exp(np.asarray(dist.logpdf(grid))), grid)
        np.testing.assert_allclose(integral, 1.0, atol=1e-6)

    def test_bounds_are_the_physical_ones(self):
        """The eigen tables read ``minimum``/``maximum``/``width``; they must
        be the LINEAR box (9.0 wide), not eryn's ln-space 2.30."""
        dist = LogUniformLinear(1.0, 10.0)
        self.assertEqual(dist.minimum, 1.0)
        self.assertEqual(dist.maximum, 10.0)
        self.assertEqual(dist.width, 9.0)

    def test_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            LogUniformLinear(0.0, 10.0)
        with self.assertRaises(ValueError):
            LogUniformLinear(-1.0, 10.0)
        with self.assertRaises(ValueError):
            LogUniformLinear(10.0, 1.0)


class _StockMBHPriorMixin:
    def _mbh_setup(self):
        """The stock MBH setup (cheap: no data load, no waveform build)."""
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings, MBHSetup

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return MBHSetup(MBHSettings(Tobs=TOBS, dt=DT, log_dir=tmp))


class StockMBHPriorTest(_StockMBHPriorMixin, unittest.TestCase):
    """The prior the stock MBH branch actually builds."""

    def test_q_entry_is_the_linear_log_uniform(self):
        container = self._mbh_setup().priors["mbh"]
        q_dist = container.priors_in["Q"]
        self.assertIsInstance(q_dist, LogUniformLinear)
        self.assertEqual((q_dist.minimum, q_dist.maximum), (1.0, 10.0))

    def test_prior_draws_pass_the_mbh_transform(self):
        container = self._mbh_setup().priors["mbh"]
        np.random.seed(5)
        draws = np.asarray(container.rvs(10000))
        self.assertEqual(draws.shape, (10000, NDIM_MBH))

        q = draws[:, Q_COL]
        self.assertTrue(np.all(q >= 1.0), f"{(q < 1.0).mean():.3f} of draws have Q < 1")
        self.assertTrue(np.all(q <= 10.0))

        # the transform asserts m1 >= m2; before the fix 43% of draws raised
        out = make_mbh_transform_container().both_transforms(draws.copy())
        m1, m2 = out[:, 0], out[:, 1]
        self.assertTrue(np.all(m1 >= m2))
        np.testing.assert_allclose(m1 / m2, q, rtol=1e-10)

    def test_logpdf_covers_the_whole_declared_range(self):
        container = self._mbh_setup().priors["mbh"]
        for q in (1.0, 2.4, 5.0, 9.9):
            self.assertTrue(
                np.isfinite(container.logpdf(_valid_mbh_row(q))),
                f"Q = {q} scored -inf inside the declared [1, 10] prior",
            )
        for q in (0.5, 10.5):
            self.assertEqual(container.logpdf(_valid_mbh_row(q)), -np.inf)

    def test_prior_box_width_for_q_is_nine(self):
        container = self._mbh_setup().priors["mbh"]
        widths = prior_box_widths(container, NDIM_MBH)
        self.assertEqual(widths[Q_COL], 9.0)

    def test_tuple_sized_draws_as_the_mbh_move_makes_them(self):
        # mbhspecialmove draws ``rvs(size=(ntemps, nwalkers, 1))``
        container = self._mbh_setup().priors["mbh"]
        np.random.seed(7)
        draws = np.asarray(container.rvs(size=(2, 5, 1)))
        self.assertEqual(draws.shape, (2, 5, 1, NDIM_MBH))
        q = draws[..., Q_COL]
        self.assertTrue(np.all((q >= 1.0) & (q <= 10.0)))

    def test_setup_and_distribution_survive_deepcopy_and_pickle(self):
        # sprint-wide rule: the pre-build fit must pickle/deepcopy
        # (mbhsearch.py deepcopies priors_in)
        import copy
        import pickle

        dist = pickle.loads(pickle.dumps(copy.deepcopy(LogUniformLinear(1.0, 10.0))))
        self.assertEqual((dist.minimum, dist.maximum, dist.width), (1.0, 10.0, 9.0))
        self.assertEqual(float(dist.logpdf(2.0)), float(LogUniformLinear(1.0, 10.0).logpdf(2.0)))
        setup = pickle.loads(pickle.dumps(copy.deepcopy(self._mbh_setup())))
        self.assertIsInstance(setup.priors["mbh"].priors_in["Q"], LogUniformLinear)

    def test_source_runtime_builder_uses_the_linear_class(self):
        """``prepare_mbh_branch`` needs a built ``GeneralSetup``, so cover the
        second call site by text -- of that function only, so a legitimate
        ln-space ``log_uniform`` on some other branch's column cannot trip
        an MBH test."""
        from lisatools.globalfit.stock.erebor import source_runtime

        text = inspect.getsource(source_runtime.prepare_mbh_branch)
        self.assertNotIn("log_uniform(", text)
        self.assertIn('"Q": LogUniformLinear(1.0, 10.0)', text)


class EryLogUniformNegativeControlTest(_StockMBHPriorMixin, unittest.TestCase):
    """Negative control: the SAME dict with eryn's ``log_uniform`` is broken.

    This is why :class:`LogUniformLinear` exists -- without it these
    assertions are what the stock MBH prior does.
    """

    def _defective_container(self):
        from eryn.prior import ProbDistContainer, log_uniform

        priors_in = dict(self._mbh_setup().priors["mbh"].priors_in)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            priors_in["Q"] = log_uniform(1.0, 10.0)
            return ProbDistContainer(priors_in)

    def test_eryn_log_uniform_draws_below_one_and_rejects_q_five(self):
        container = self._defective_container()
        np.random.seed(11)
        q = np.asarray(container.rvs(2000))[:, Q_COL]

        # ln-space draws: support [0, ln 10], so ~43% land below Q = 1
        self.assertTrue(np.any(q < 1.0))
        self.assertGreater((q < 1.0).mean(), 0.3)
        self.assertLess(q.max(), np.log(10.0) + 1e-9)

        # and everything above ln(10) is outside the prior
        self.assertEqual(container.logpdf(_valid_mbh_row(5.0)), -np.inf)

    def test_eryn_log_uniform_draws_crash_the_mbh_transform(self):
        container = self._defective_container()
        np.random.seed(12)
        draws = np.asarray(container.rvs(2000))
        with self.assertRaises(AssertionError):
            make_mbh_transform_container().both_transforms(draws.copy())


if __name__ == "__main__":
    unittest.main()
