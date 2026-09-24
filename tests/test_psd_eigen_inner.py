"""PSD eigen-axis inner proposal (one-walker regime): kind resolution, tables, one MH step."""

import os
import unittest
from unittest import mock

import numpy as np
from eryn.model import Model
from eryn.moves import TemperatureControl
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import BranchSupplemental

from lisatools.globalfit.moves.psdmove import PSDMove
from lisatools.globalfit.state import GFState

H = np.array([[4.0, 0.0], [0.0, 1.0]])  # curvature -> sigmas 0.5 and 1.0 along the axes
NT = 3


def _quad(p):
    p = np.atleast_2d(np.asarray(p, dtype=float))
    return -0.5 * np.einsum("ni,ij,nj->n", p, H, p)


def _like_fn(coords, inds=None, logp=None, supps=None, branch_supps=None):
    x = coords["psd"]
    return _quad(x.reshape(-1, 2)).reshape(x.shape[:2]), None


def _prior_fn(coords, *args, **kwargs):
    x = coords["psd"]
    return np.zeros(x.shape[:2])


def _move(kind=None, nwalkers=1):
    tc = TemperatureControl(2, nwalkers, ntemps=NT, permute=False)
    m = PSDMove(
        None, {"psd": ProbDistContainer({0: uniform_dist(-10, 10), 1: uniform_dist(-10, 10)})},
        sampled_branches=["psd"], temperature_control=tc, live_dangerously=True,
        inner_move_kind=kind, name="eigen test",
    )
    m.compute_log_like = _like_fn
    m.compute_log_prior = _prior_fn
    m._score_rows = lambda w, p, g, s: _quad(p)
    m._fixed_noise_coords = {}
    m.periodic = None
    m.accepted = np.zeros((NT, nwalkers))
    return m, tc


class InnerKindTest(unittest.TestCase):
    def test_default_is_eigen_at_one_walker_and_stretch_otherwise(self):
        m, _ = _move()
        self.assertEqual(m._resolve_inner_kind(1), "eigen")
        self.assertEqual(m._resolve_inner_kind(4), "stretch")

    def test_stretch_at_one_walker_raises_and_names_the_knob(self):
        m, _ = _move(kind="stretch")
        with self.assertRaisesRegex(ValueError, "INNER_MOVE_KIND"):
            m._resolve_inner_kind(1)
        self.assertEqual(m._resolve_inner_kind(2), "stretch")
        with self.assertRaises(ValueError):
            _move(kind="bogus")[0]._resolve_inner_kind(1)

    def test_default_keys_off_the_block_width_not_the_run(self):
        """2026-09-17 ruling: the stretch complement is the rank's LOCAL
        block (cross-rank pooling is WP8), so a 1- or 2-walker BLOCK defaults
        to eigen whatever the RUN's walker count -- a 4-GPU, 4-walker
        campaign run (one walker per rank) must not hard-error at its first
        PSD propose. An explicit ``stretch`` at a 1-walker block still
        raises and names the knob, block and run counts."""
        m, _ = _move()
        self.assertEqual(m._resolve_inner_kind(1, nwalkers_run=10), "eigen")
        self.assertEqual(m._resolve_inner_kind(2, nwalkers_run=10), "eigen")
        self.assertEqual(m._resolve_inner_kind(1, nwalkers_run=1), "eigen")
        self.assertEqual(m._resolve_inner_kind(3, nwalkers_run=12), "stretch")
        self.assertEqual(m._resolve_inner_kind(4), "stretch")
        m2, _ = _move(kind="stretch")
        with self.assertRaisesRegex(ValueError, "INNER_MOVE_KIND") as ctx:
            m2._resolve_inner_kind(1, nwalkers_run=10)
        self.assertIn("block", str(ctx.exception).lower())
        self.assertIn("1", str(ctx.exception))
        self.assertIn("10", str(ctx.exception))

    def test_resolved_kind_logged_once_per_move(self):
        m, _ = _move()
        with mock.patch("lisatools.globalfit.moves.psdmove.logger") as mock_logger:
            m._resolve_inner_kind(1)
            m._resolve_inner_kind(1)
        self.assertEqual(mock_logger.info.call_count, 1)


class EigenTablesTest(unittest.TestCase):
    def test_tables_recover_the_quadratic_axes_per_rung(self):
        m, _ = _move()
        coords = {"psd": np.zeros((NT, 1, 1, 2))}
        m._refresh_eigen_tables(coords)
        axes, sigmas = m._eigen_inner._tables["psd"]
        self.assertEqual(axes.shape, (NT, 1, 1, 2, 2))
        self.assertEqual(sigmas.shape, (NT, 1, 1, 2))
        # Since the 2026-09-18 cold-scope ruling the table is built ONCE on
        # the cold row and shared up the ladder, with the sigmas widened by
        # 1/sqrt(beta) per rung -- EigenAxisMove applies no beta scaling of
        # its own, so a cold sigma handed to a hot rung would step sqrt(T)
        # too small. The COLD rung still recovers the quadratic exactly.
        betas = np.asarray(m.temperature_control.betas, dtype=float)
        for t in range(NT):
            widen = 1.0 / np.sqrt(betas[t]) if betas[t] > 0 else 1.0
            np.testing.assert_allclose(
                np.sort(sigmas[t, 0, 0]), np.array([0.5, 1.0]) * widen, rtol=1e-3
            )
            A = np.abs(axes[t, 0, 0])  # columns are +-e_i in some order
            np.testing.assert_allclose(A @ A.T, np.eye(2), atol=1e-3)
            np.testing.assert_allclose(np.sort(A.ravel()), [0.0, 0.0, 1.0, 1.0], atol=1e-3)

    def test_tables_tile_over_walkers(self):
        m, _ = _move(kind="eigen", nwalkers=3)
        m._refresh_eigen_tables({"psd": np.zeros((NT, 3, 1, 2))})
        axes, _ = m._eigen_inner._tables["psd"]
        self.assertEqual(axes.shape, (NT, 3, 1, 2, 2))


class EigenStepTest(unittest.TestCase):
    def test_run_move_takes_an_eigen_mh_step_with_one_walker(self):
        m, tc = _move()
        m._inner_kind = "eigen"
        m._refresh_eigen_tables({"psd": np.zeros((NT, 1, 1, 2))})
        m._tally_in_model_proposed = np.zeros(NT, dtype=int)
        m._tally_in_model_accepted = np.zeros(NT, dtype=int)
        m._tally_swaps_proposed = np.zeros(NT - 1, dtype=int)
        m._tally_swaps_accepted = np.zeros(NT - 1, dtype=int)
        rng = np.random.RandomState(3)
        coords = {"psd": rng.normal(size=(NT, 1, 1, 2))}
        supps = BranchSupplemental({"walker_inds": np.zeros((NT, 1), dtype=int)}, base_shape=(NT, 1), copy=True)
        state = GFState(coords, copy=True, supplemental=supps)
        state.log_prior = _prior_fn(coords)
        state.log_like = _like_fn(coords)[0]
        model = Model(None, _like_fn, _prior_fn, tc, map, rng)
        np.random.seed(5)
        new_state, accepted = m.run_move(0, model, state)
        self.assertEqual(np.asarray(accepted).shape, (NT, 1))
        np.testing.assert_allclose(new_state.log_like, _like_fn(new_state.branches_coords)[0])
        moved = np.any(new_state.branches_coords["psd"] != coords["psd"], axis=(2, 3))
        # rejected rows keep their coords, accepted rows moved (swaps may shuffle rungs, so compare sets)
        self.assertEqual(int(moved.sum()) >= int(np.asarray(accepted).sum()), True)
        # the OUTER move's own accepted/num_proposals must advance too (the
        # eigen branch used to leave them at zero, poisoning
        # acceptance_fraction with a zeros/0 division)
        self.assertEqual(m.num_proposals, 1)
        self.assertEqual(m.accepted.shape, (NT, 1))
        self.assertEqual(m.accepted.sum(), np.asarray(accepted).sum())


class RealNoisePriorWidthsTest(unittest.TestCase):
    """The eigen step/cap scale must come off the REAL branch priors.

    Regression for the 2026-09-16 one-walker defect: ``prior_box_widths``
    resolved columns off ``priors_in`` KEYS, and ``psd_prior_dict`` spells
    its keys as LaTeX labels (``r"$S_{\\rm oms}$"``), so every psd column
    silently fell back to width 1.0 — 5e9x / 5e12x the true box. The
    finite-difference corners and the ``axis_prior_bounds`` cap are both
    ``width``-scaled, so the move proposed ~1e5 prior widths per step and
    accepted nothing. ``galfor`` (integer keys) is the paired control: it
    was already correct and must stay bit-identical.
    """

    def _check(self, dct, ndim):
        from eryn.prior import ProbDistContainer

        from lisatools.globalfit.moves.eigen_refresh import prior_box_widths

        container = ProbDistContainer(dct)
        true = np.array(
            [float(d.maximum) - float(d.minimum) for d in dct.values()], dtype=float
        )
        self.assertEqual(true.size, ndim)
        np.testing.assert_allclose(prior_box_widths(container, ndim), true, rtol=1e-12)
        return true

    def test_psd_widths_match_the_prior_box_in_both_bases(self):
        from lisatools.globalfit.stock.erebor.noise import (
            PSD_PRIOR_RANGE, psd_prior_dict,
        )

        lin = self._check(psd_prior_dict(False), 2)
        # the physically meaningful numbers, not just self-consistency
        np.testing.assert_allclose(
            lin, [hi - lo for (lo, hi) in PSD_PRIOR_RANGE], rtol=1e-12
        )
        # the defect's signature: these are nowhere near 1.0
        self.assertLess(lin.max(), 1e-9)
        self._check(psd_prior_dict(True), 2)

    def test_galfor_widths_unchanged_the_integer_keyed_control(self):
        from lisatools.globalfit.stock.erebor.noise import (
            GALFOR_PRIOR_RANGE, galfor_prior_dict,
        )

        lin = self._check(galfor_prior_dict(False), 5)
        np.testing.assert_allclose(
            lin, [hi - lo for (lo, hi) in GALFOR_PRIOR_RANGE], rtol=1e-12
        )
        self._check(galfor_prior_dict(True), 5)

    def test_string_keyed_source_branches_resolve_too(self):
        """Same root cause, same blast radius: mbh / emri / sobbh also spell
        their prior dicts with parameter NAMES, so the addremove eigen
        tables were built on unit widths as well."""
        from eryn.prior import ProbDistContainer, log_uniform, uniform_dist

        from lisatools.globalfit.moves.eigen_refresh import prior_box_widths

        dct = {
            "logM": uniform_dist(np.log(1e5), np.log(1e8)),
            "Q": log_uniform(1.0, 10.0),
            "dist": uniform_dist(1.0, 150.0),
            "t_plunge": uniform_dist(0.0, 7.9e6),
        }
        w = prior_box_widths(ProbDistContainer(dct), 4)
        # LogUniform's minimum/maximum are the SAMPLER-space (ln) bounds,
        # which is the basis the eigen table works in. NOTE: this dict is a
        # generic string-keyed stand-in -- the stock MBH "Q" column is
        # ``LogUniformLinear(1.0, 10.0)`` (physical bounds, width 9.0) since
        # 2026-09-16; see tests/test_mbh_q_prior.py.
        np.testing.assert_allclose(
            w, [np.log(1e8) - np.log(1e5), np.log(10.0), 149.0, 7.9e6], rtol=1e-12
        )

    def test_psd_eigen_sigmas_stay_inside_the_prior_box(self):
        """End of the causal chain: with the right widths every axis step is
        at most one prior width, so a draw can land inside the prior."""
        from eryn.moves.eigenaxis import axis_prior_bounds
        from eryn.prior import ProbDistContainer

        from lisatools.globalfit.moves.eigen_refresh import (
            eigen_tables_from_ll_batch, prior_box_widths,
        )
        from lisatools.globalfit.stock.erebor.noise import psd_prior_dict

        container = ProbDistContainer(psd_prior_dict(False))
        widths = prior_box_widths(container, 2)
        x0 = np.array([[4.9e-11, 8.9e-14], [1.7e-10, 3.3e-14]])

        def call_ll(x):
            # a slope plus mild curvature, in the branch's own units
            y = (np.atleast_2d(x) - x0[0][None, :]) / widths[None, :]
            return -(y ** 2).sum(axis=1) - 3.0 * y[:, 0]

        axes, sigmas = eigen_tables_from_ll_batch(call_ll, x0, widths)
        np.testing.assert_array_less(
            sigmas, axis_prior_bounds(axes, widths) * (1.0 + 1e-9)
        )
        # ... and a draw off the table stays in the box for a central point
        rng = np.random.RandomState(0)
        z = rng.standard_normal((200, 2))
        step = np.einsum("ik,nk->ni", axes[0] * sigmas[0][None, :], z)
        inside = np.isfinite(container.logpdf(x0[0][None, :] + step))
        self.assertGreater(inside.mean(), 0.2)


class SettingsKnobsTest(unittest.TestCase):
    def test_noise_settings_carry_the_eigen_fields(self):
        from lisatools.globalfit.stock.erebor.noise import GalForSettings, PSDSettings
        from lisatools.globalfit.stock.erebor.stochastic import SGWBSettings

        for cls in (PSDSettings, GalForSettings, SGWBSettings):
            s = cls()
            self.assertIsNone(s.inner_move_kind)
            self.assertEqual(s.eigen_refresh_every, 10)
            self.assertAlmostEqual(s.eigen_eps_rel, 1e-4)
        with mock.patch.dict(os.environ, {"GALFOR_INNER_MOVE_KIND": "eigen", "GALFOR_EIGEN_REFRESH": "3"}):
            s = GalForSettings()
            self.assertEqual(s.inner_move_kind, "eigen")
            self.assertEqual(s.eigen_refresh_every, 3)


if __name__ == "__main__":
    unittest.main()
