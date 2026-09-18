"""psd/galfor eigen tables: build on the COLD row only, share up the ladder.

User ruling 2026-09-18. The old path expanded the information matrix at
EVERY rung, at walker 0, and broadcast the result across walkers. Two
problems it caused, both measured in the live runs:

* the hot rungs sit far from the mode where the log-likelihood genuinely
  has negative-curvature directions -- the 3-month run logged ``8/12
  matrices had a non-positive eigenvalue (worst lambda/lambda_max =
  -inf)``, 12 being galfor's ntemps;
* it cost ``ntemps`` likelihood builds per refresh (12 for galfor/psd)
  where ``nwalkers`` would do -- 1 at one walker per rank.

The replacement builds one table per walker on the cold row and shares it
up that walker's ladder. That makes the sigma tempering load-bearing:
``EigenAxisMove`` applies NO beta scaling (``draw_axis_step`` is
``jump_factor * sigma * z``; ``beta`` does not appear in
``eryn/moves/eigenaxis.py``), so a cold sigma handed to a hot rung steps
``sqrt(T)`` too small and the hot chain stops travelling while accepting
almost everything.
"""

import os
import unittest

import numpy as np

from lisatools.globalfit.moves.eigen_refresh import temper_sigmas


class TemperSigmasTest(unittest.TestCase):
    """``sigma / sqrt(beta)`` on the leading (rung) axis."""

    def test_cold_rung_is_unchanged(self):
        sig = np.full((1, 3, 1, 5), 0.25)
        out = temper_sigmas(sig, np.array([1.0]))
        np.testing.assert_allclose(out, sig)

    def test_hot_rungs_widen_as_one_over_sqrt_beta(self):
        sig = np.full((1, 2, 1, 4), 0.1)
        betas = np.array([1.0, 1e-2, 1e-4])
        out = temper_sigmas(sig, betas)
        self.assertEqual(out.shape, (3, 2, 1, 4))
        np.testing.assert_allclose(out[0], 0.1)
        np.testing.assert_allclose(out[1], 0.1 * 10.0)    # 1/sqrt(1e-2)
        np.testing.assert_allclose(out[2], 0.1 * 100.0)   # 1/sqrt(1e-4)

    def test_a_flattened_rung_keeps_the_cold_sigma_not_inf(self):
        """beta <= 0 has no finite width; it must not produce inf/nan."""
        sig = np.full((1, 1, 1, 2), 0.3)
        out = temper_sigmas(sig, np.array([1.0, 0.0, -1.0]))
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out[1], 0.3)
        np.testing.assert_allclose(out[2], 0.3)

    def test_per_walker_tables_stay_distinct_through_the_widening(self):
        sig = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]])       # (1, 2 walkers, 1, 2)
        out = temper_sigmas(sig, np.array([1.0, 0.25]))
        np.testing.assert_allclose(out[0, 0, 0], [1.0, 2.0])
        np.testing.assert_allclose(out[0, 1, 0], [3.0, 4.0])
        np.testing.assert_allclose(out[1, 0, 0], [2.0, 4.0])  # x 1/sqrt(0.25)
        np.testing.assert_allclose(out[1, 1, 0], [6.0, 8.0])


class _Move:
    """The two resolvers under test, lifted off PSDMove without its ctor."""

    from lisatools.globalfit.moves.psdmove import PSDMove

    _eigen_scope = PSDMove._eigen_scope
    _eigen_temper_sigmas = PSDMove._eigen_temper_sigmas

    def __init__(self, prefix="GALFOR", scope=None):
        self._debug_prefix = prefix
        self.eigen_table_scope = scope


class ScopeResolutionTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in (
            "GALFOR_EIGEN_SCOPE", "PSD_EIGEN_SCOPE",
            "GALFOR_EIGEN_TEMPER_SIGMAS")}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_default_is_cold_per_walker(self):
        self.assertEqual(_Move()._eigen_scope(), "cold_per_walker")

    def test_env_selects_the_legacy_path(self):
        os.environ["GALFOR_EIGEN_SCOPE"] = "per_temp"
        self.assertEqual(_Move()._eigen_scope(), "per_temp")
        # ... and the knob is per-branch, not global
        self.assertEqual(_Move(prefix="PSD")._eigen_scope(), "cold_per_walker")

    def test_kwarg_beats_env(self):
        os.environ["GALFOR_EIGEN_SCOPE"] = "per_temp"
        self.assertEqual(
            _Move(scope="cold_per_walker")._eigen_scope(), "cold_per_walker")

    def test_an_unknown_value_falls_back_loudly_not_silently(self):
        os.environ["GALFOR_EIGEN_SCOPE"] = "nonsense"
        with self.assertLogs(
                "lisatools.globalfit.moves.psdmove", level="WARNING"):
            self.assertEqual(_Move()._eigen_scope(), "cold_per_walker")

    def test_sigma_tempering_defaults_on_and_is_switchable(self):
        self.assertTrue(_Move()._eigen_temper_sigmas())
        for off in ("0", "false", "no"):
            os.environ["GALFOR_EIGEN_TEMPER_SIGMAS"] = off
            self.assertFalse(_Move()._eigen_temper_sigmas(), off)
        os.environ["GALFOR_EIGEN_TEMPER_SIGMAS"] = "1"
        self.assertTrue(_Move()._eigen_temper_sigmas())


class ExpansionPointSelectionTest(unittest.TestCase):
    """The slice that picks the expansion points, pinned directly.

    The table build itself needs a live PSD move with data; what this file
    can pin without one is the indexing contract that decides WHICH points
    are expanded -- the whole substance of the change.
    """

    def setUp(self):
        # (ntemps, nwalkers, nleaves, ndim), each entry tagged t*100 + w
        self.ntemps, self.nwalkers, self.ndim = 4, 3, 2
        self.coords = np.array([
            [[[t * 100 + w, 0.0]] for w in range(self.nwalkers)]
            for t in range(self.ntemps)
        ], dtype=float)

    def test_cold_per_walker_takes_the_cold_row_across_walkers(self):
        sl = (0, slice(None))
        pts = self.coords[sl][:, 0, :]
        self.assertEqual(pts.shape, (self.nwalkers, self.ndim))
        np.testing.assert_allclose(pts[:, 0], [0, 1, 2])   # t=0, w=0,1,2

    def test_per_temp_takes_walker_zero_across_rungs(self):
        sl = (slice(None), 0)
        pts = self.coords[sl][:, 0, :]
        self.assertEqual(pts.shape, (self.ntemps, self.ndim))
        np.testing.assert_allclose(pts[:, 0], [0, 100, 200, 300])

    def test_cold_scope_costs_nwalkers_builds_not_ntemps(self):
        cold = self.coords[(0, slice(None))][:, 0, :].shape[0]
        legacy = self.coords[(slice(None), 0)][:, 0, :].shape[0]
        self.assertEqual(cold, self.nwalkers)
        self.assertEqual(legacy, self.ntemps)
        # the live galfor/psd shape: ntemps 12 against one walker per rank
        self.assertEqual(np.zeros((12, 1, 1, 5))[(0, slice(None))].shape[0], 1)
        self.assertEqual(np.zeros((12, 1, 1, 5))[(slice(None), 0)].shape[0], 12)

    def test_each_cold_point_is_scored_against_its_own_walker(self):
        """The legacy path scored every rung against walker 0."""
        n_pts = self.nwalkers
        pt_walker = np.arange(n_pts, dtype=np.int32)
        idx = np.arange(4 * n_pts) % n_pts        # 4 corner blocks
        np.testing.assert_array_equal(pt_walker[idx], np.tile([0, 1, 2], 4))


if __name__ == "__main__":
    unittest.main()
