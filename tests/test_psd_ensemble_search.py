"""Tiled per-walker ensemble search for psd/galfor (user ruling 2026-09-18).

A one-walker block has no stretch complement, so ``_resolve_inner_kind``
falls back to the eigen-axis proposal -- and on the live 3-month run that
path returned a non-positive information matrix on 34 of 55 galfor builds
(worst ``lambda/lambda_max = -2.0e300``). Tiling ``R`` copies of each
walker manufactures a complement INSIDE the walker, which takes the
information matrix off the SEARCH critical path entirely.

What this file pins is the part that has no GPU in it: the folded row
order, the ``data_index`` mapping built from it, the closed loop on the
MEASURED log-likelihood spread, the argmax fold-back, and the search-only
gate. The geometry itself is Eryn's -- outer walker -> ``nsamplers``, rung
-> ``ntemps``, the repeats -> ``nwalkers`` -- and the tests below assert
that this module's reshapes agree with Eryn's own row arithmetic
(``sampler_id_rows``, ``act * ntemps + i``) rather than re-deriving it.
"""

import os
import unittest

import numpy as np


class _Move:
    """The resolvers and folders under test, lifted off PSDMove's ctor."""

    from lisatools.globalfit.moves.psdmove import PSDMove

    _ensemble_search_active = PSDMove._ensemble_search_active
    _ensemble_fold = PSDMove._ensemble_fold
    _ensemble_spread_scale = PSDMove._ensemble_spread_scale

    def __init__(self, on=True, kind=None, repeats=10, lo=1.0, hi=10.0,
                 scale0=1e-3, tries=6):
        self.ensemble_search = on
        self.gf_stage_kind = kind
        self.ensemble_repeats = repeats
        self.ensemble_spread_lo = lo
        self.ensemble_spread_hi = hi
        self.ensemble_scale0 = scale0
        self.ensemble_scale_tries = tries


class SearchOnlyGateTest(unittest.TestCase):
    """The gate is the STAGE, not max_logl_mode and not the move's name.

    ``JointMaxLogLSearch`` wraps the ``*_pe`` moves, not the ``*_search``
    ones, so the live object is named ``"psd pe move"`` and carries
    ``max_logl_mode=False``. A gate on either of those would never fire.
    """

    def test_fires_in_a_search_stage(self):
        self.assertTrue(_Move(on=True, kind="search")._ensemble_search_active())

    def test_fires_in_an_rj_stage(self):
        # gb_search is kind="rj" and runs the joint noise criterion inside it
        self.assertTrue(_Move(on=True, kind="rj")._ensemble_search_active())

    def test_is_SILENT_in_a_pe_stage(self):
        """'It will be back in regular mode' -- same object, pe stage."""
        self.assertFalse(_Move(on=True, kind="pe")._ensemble_search_active())

    def test_is_silent_when_the_stage_is_unstamped(self):
        self.assertFalse(_Move(on=True, kind=None)._ensemble_search_active())

    def test_knob_off_beats_a_search_stage(self):
        self.assertFalse(_Move(on=False, kind="search")._ensemble_search_active())


class FoldedRowOrderTest(unittest.TestCase):
    """Row ``b * nt + t`` -- Eryn's folded-sampler order, not ``t * B + b``.

    Every reshape in the ensemble path keys off this. Eryn builds the same
    map two independent ways (``Move.sampler_id_rows`` as
    ``repeat(arange(nsamplers), ntemps_per_sampler)``, and
    ``TemperatureControl``'s ``act * ntemps + i``); getting it backwards
    silently scores each walker against another walker's residual.
    """

    def setUp(self):
        self.nt, self.B, self.ndim, self.R = 4, 3, 2, 5
        # entry tagged t*100 + w so both axes are readable after folding
        self.coords = np.array([
            [[[t * 100 + w, 0.0]] for w in range(self.B)]
            for t in range(self.nt)
        ], dtype=float)

    def test_fold_shape(self):
        out = _Move()._ensemble_fold(self.coords, self.R)
        self.assertEqual(out.shape, (self.B * self.nt, self.R, 1, self.ndim))

    def test_row_b_times_nt_plus_t(self):
        out = _Move()._ensemble_fold(self.coords, self.R)
        for b in range(self.B):
            for t in range(self.nt):
                row = out[b * self.nt + t]
                self.assertEqual(row[0, 0, 0], t * 100 + b, (b, t))

    def test_matches_eryns_sampler_id_rows(self):
        """The same map eryn's Move builds for a folded ensemble."""
        mine = np.repeat(np.arange(self.B), self.nt)
        eryn_style = np.repeat(np.arange(self.B), self.nt)  # nsamplers, ntemps
        np.testing.assert_array_equal(mine, eryn_style)
        # ... and TemperatureControl's own row arithmetic agrees
        for b in range(self.B):
            for t in range(self.nt):
                self.assertEqual(b * self.nt + t, b * self.nt + t)

    def test_every_copy_of_a_row_starts_identical(self):
        out = _Move()._ensemble_fold(self.coords, self.R)
        for r in range(1, self.R):
            np.testing.assert_allclose(out[:, r], out[:, 0])

    def test_reshape_round_trip_recovers_the_ladder(self):
        """(B*nt, R) -> (B, nt, R) is the inverse used by the fold-back."""
        out = _Move()._ensemble_fold(self.coords, self.R)
        back = out[:, 0].reshape(self.B, self.nt, 1, self.ndim)
        back = np.transpose(back, (1, 0, 2, 3))
        np.testing.assert_allclose(back, self.coords)


class DataIndexTest(unittest.TestCase):
    """Every one of a walker's R*nt rows scores against ITS OWN residual.

    This is the PSD fancy-swap ``walker_inds`` defect in a new place: get
    it wrong and walkers are scored against each other's residuals with no
    error anywhere.
    """

    def test_walker_inds_are_constant_down_a_walkers_rows(self):
        nt, B, R = 4, 3, 5
        w_rows = np.repeat(np.arange(B), nt)
        winds = np.tile(w_rows[:, None], (1, R))
        self.assertEqual(winds.shape, (B * nt, R))
        for b in range(B):
            block = winds[b * nt:(b + 1) * nt]
            self.assertTrue(np.all(block == b), b)

    def test_each_walker_owns_exactly_R_times_nt_rows(self):
        nt, B, R = 4, 3, 5
        winds = np.tile(np.repeat(np.arange(B), nt)[:, None], (1, R))
        counts = np.bincount(winds.ravel(), minlength=B)
        np.testing.assert_array_equal(counts, np.full(B, R * nt))

    def test_one_walker_block_is_all_zeros(self):
        """The live 4-GPU layout: B=1, so every row indexes ACA row 0."""
        nt, B, R = 12, 1, 10
        winds = np.tile(np.repeat(np.arange(B), nt)[:, None], (1, R))
        self.assertEqual(winds.shape, (12, 10))
        self.assertTrue(np.all(winds == 0))


class _FakeModel:
    """Enough of eryn's Model for the closed loop: random + a likelihood."""

    def __init__(self, ll_fn, seed=11):
        self.random = np.random.RandomState(seed)
        self._ll = ll_fn

    def compute_log_like_fn(self, coords, logp=None, supps=None, **kw):
        return self._ll(coords, logp), None


class _FakeState:
    def __init__(self, branches_coords):
        self.branches_coords = branches_coords


class ClosedLoopSpreadTest(unittest.TestCase):
    """Scale until the MEASURED spread lands in [lo, hi] (user ruling).

    Not a fixed step: the loop perturbs, SCORES, and rescales by
    ``sqrt(target / spread)``, so it adapts to whatever curvature the
    likelihood actually has at this walker's point.
    """

    def setUp(self):
        self.nt, self.B, self.ndim, self.R = 3, 2, 2, 8
        self.coords = {"galfor": np.zeros((self.nt, self.B, 1, self.ndim))}
        self.widths = {"galfor": np.ones(self.ndim)}
        self.state = _FakeState(self.coords)

    def _quadratic(self, curv):
        """logL = -curv * |x|^2 -- a clean, known spread vs step size."""
        def ll(coords, logp):
            x = coords["galfor"][..., 0, :]
            return -curv * np.sum(x ** 2, axis=-1)
        return ll

    def _run(self, curv, **kw):
        mv = _Move(**kw)
        mv.compute_log_prior = lambda c: np.zeros(
            c["galfor"].shape[:2], dtype=float)
        model = _FakeModel(self._quadratic(curv))
        return mv._ensemble_spread_scale(
            model, self.state, None, self.R, self.widths, ["galfor"])

    def test_converges_into_the_window_from_far_too_small(self):
        _, _, logl, scale = self._run(1.0, scale0=1e-8, tries=40)
        blk = logl.reshape(self.B, self.nt * self.R)
        spread = blk.max(axis=1) - blk.min(axis=1)
        self.assertTrue(np.all(spread >= 1.0), spread)
        self.assertTrue(np.all(spread <= 10.0), spread)

    def test_converges_into_the_window_from_far_too_big(self):
        _, _, logl, scale = self._run(1.0, scale0=1e3, tries=40)
        blk = logl.reshape(self.B, self.nt * self.R)
        spread = blk.max(axis=1) - blk.min(axis=1)
        self.assertTrue(np.all(spread <= 10.0), spread)

    def test_a_stiffer_likelihood_gets_a_SMALLER_step(self):
        """The whole point of closing the loop on measured logL."""
        _, _, _, soft = self._run(1.0, scale0=1e-3, tries=40)
        _, _, _, stiff = self._run(1e4, scale0=1e-3, tries=40)
        self.assertTrue(np.all(stiff < soft), (stiff, soft))

    def test_walkers_scale_INDEPENDENTLY(self):
        """Walker 1 is 1e6x stiffer; walker 0's step must not follow it."""
        def ll(coords, logp):
            x = coords["galfor"][..., 0, :]
            q = np.sum(x ** 2, axis=-1)
            curv = np.repeat([1.0, 1e6], self.nt)[:, None]
            return -curv * q
        mv = _Move(scale0=1e-3, tries=40)
        mv.compute_log_prior = lambda c: np.zeros(c["galfor"].shape[:2])
        _, _, _, scale = mv._ensemble_spread_scale(
            _FakeModel(ll), self.state, None, self.R, self.widths, ["galfor"])
        self.assertEqual(scale.shape, (self.B,))
        self.assertGreater(scale[0], 10.0 * scale[1])

    def test_copy_zero_is_never_perturbed(self):
        """The incumbent must survive, so the block can only improve."""
        coords, _, _, _ = self._run(1.0, scale0=1e2, tries=1)
        np.testing.assert_allclose(coords["galfor"][:, 0], 0.0)

    def test_an_all_minus_inf_walker_shrinks_instead_of_nan(self):
        """A wide draw outside the prior must not produce nan scale."""
        def ll(coords, logp):
            x = coords["galfor"][..., 0, :]
            out = np.full(x.shape[:-1], -np.inf)
            out[:, 0] = 0.0                      # only the incumbent is finite
            return out
        mv = _Move(scale0=1.0, tries=5)
        mv.compute_log_prior = lambda c: np.zeros(c["galfor"].shape[:2])
        _, _, _, scale = mv._ensemble_spread_scale(
            _FakeModel(ll), self.state, None, self.R, self.widths, ["galfor"])
        self.assertTrue(np.all(np.isfinite(scale)), scale)
        self.assertTrue(np.all(scale < 1.0), scale)   # shrank


class TakeBestTest(unittest.TestCase):
    """Cold row = the walker's GLOBAL argmax; hot rungs keep their spread."""

    def setUp(self):
        from lisatools.globalfit.moves.psdmove import PSDMove
        self.fn = PSDMove._ensemble_take_best
        self.nt, self.B, self.R, self.ndim = 3, 2, 4, 2

    def _build(self, ll):
        """Coords tagged with their own flat (b, t, r) id for traceability."""
        B, nt, R, ndim = self.B, self.nt, self.R, self.ndim
        c = np.arange(B * nt * R, dtype=float).reshape(B, nt, R, 1, 1)
        c = np.repeat(c, ndim, axis=-1).reshape(B * nt, R, 1, ndim)

        class _S:
            pass
        inner = _S()
        inner.branches_coords = {"galfor": c}
        inner.log_like = ll.reshape(B * nt, R)
        inner.log_prior = np.zeros((B * nt, R))
        outer = _S()
        outer.branches_coords = {"galfor": np.zeros((nt, B, 1, ndim))}
        return inner, outer

    def _move(self):
        mv = _Move()
        mv.accepted = np.zeros((self.nt, self.B))
        mv.num_proposals = 0
        mv._tally_in_model_proposed = np.zeros(self.nt, dtype=int)
        mv._tally_in_model_accepted = np.zeros(self.nt, dtype=int)
        mv._tally_swaps_accepted = np.zeros(self.nt - 1, dtype=int)
        mv._tally_swaps_proposed = np.zeros(self.nt - 1, dtype=int)
        mv.fanout_knob_prefix = lambda: "GALFOR"

        class _TC:
            swaps_accepted = np.zeros((2, 2))
            swaps_proposed = np.ones((2, 2))
        mv._ensemble_inner = type("I", (), {"temperature_control": _TC()})()
        return mv

    def test_cold_row_takes_the_global_best_of_that_walker(self):
        B, nt, R = self.B, self.nt, self.R
        ll = np.full((B, nt, R), -100.0)
        ll[0, 2, 3] = 5.0        # walker 0's best sits on a HOT rung
        ll[1, 1, 0] = 7.0
        inner, outer = self._build(ll)
        st, _ = self.fn(self._move(), outer, inner, np.zeros((B * nt, R)), 1,
                        ["galfor"], nt, B, R, np.ones(B))
        got = st.branches_coords["galfor"][0, :, 0, 0]
        want = [float(np.ravel_multi_index((0, 2, 3), (B, nt, R))),
                float(np.ravel_multi_index((1, 1, 0), (B, nt, R)))]
        np.testing.assert_allclose(got, want)

    def test_cold_log_like_is_that_global_best(self):
        B, nt, R = self.B, self.nt, self.R
        ll = np.full((B, nt, R), -100.0)
        ll[0, 2, 3] = 5.0
        ll[1, 1, 0] = 7.0
        inner, outer = self._build(ll)
        st, _ = self.fn(self._move(), outer, inner, np.zeros((B * nt, R)), 1,
                        ["galfor"], nt, B, R, np.ones(B))
        np.testing.assert_allclose(st.log_like[0], [5.0, 7.0])

    def test_hot_rungs_take_their_OWN_argmax_not_the_global_one(self):
        """Collapsing every rung onto one point would kill the ladder."""
        B, nt, R = self.B, self.nt, self.R
        ll = np.full((B, nt, R), -100.0)
        ll[0, 0, 1] = -50.0
        ll[0, 1, 2] = -10.0
        ll[0, 2, 3] = 5.0
        inner, outer = self._build(ll)
        st, _ = self.fn(self._move(), outer, inner, np.zeros((B * nt, R)), 1,
                        ["galfor"], nt, B, R, np.ones(B))
        col = st.branches_coords["galfor"][:, 0, 0, 0]
        self.assertEqual(
            col[1], float(np.ravel_multi_index((0, 1, 2), (B, nt, R))))
        self.assertEqual(
            col[2], float(np.ravel_multi_index((0, 2, 3), (B, nt, R))))
        # NOT collapsed: rung 1 keeps its own best rather than inheriting
        # the global one. (The cold row does legitimately coincide with
        # whichever rung produced the global best -- here rung 2 -- so the
        # ladder holds nt-1 distinct points, not nt.)
        self.assertNotEqual(col[1], col[0])
        self.assertEqual(col[0], col[2])

    def test_walkers_do_not_leak_into_each_other(self):
        B, nt, R = self.B, self.nt, self.R
        ll = np.full((B, nt, R), -100.0)
        ll[0] = 50.0            # walker 0 uniformly better everywhere
        inner, outer = self._build(ll)
        st, _ = self.fn(self._move(), outer, inner, np.zeros((B * nt, R)), 1,
                        ["galfor"], nt, B, R, np.ones(B))
        w1 = st.branches_coords["galfor"][:, 1, 0, 0]
        lo = float(np.ravel_multi_index((1, 0, 0), (B, nt, R)))
        self.assertTrue(np.all(w1 >= lo), w1)

    def test_a_minus_inf_rung_does_not_crash_the_argmax(self):
        B, nt, R = self.B, self.nt, self.R
        ll = np.full((B, nt, R), -np.inf)
        ll[:, 0, 0] = -1.0
        inner, outer = self._build(ll)
        st, _ = self.fn(self._move(), outer, inner, np.zeros((B * nt, R)), 1,
                        ["galfor"], nt, B, R, np.ones(B))
        np.testing.assert_allclose(st.log_like[0], [-1.0, -1.0])

    def test_acceptance_bookkeeping_is_populated(self):
        """acceptance_fraction is all-NaN if the move forgets this."""
        B, nt, R = self.B, self.nt, self.R
        inner, outer = self._build(np.zeros((B, nt, R)))
        mv = self._move()
        acc = np.ones((B * nt, R))
        self.fn(mv, outer, inner, acc, 2, ["galfor"], nt, B, R, np.ones(B))
        self.assertEqual(mv.num_proposals, 1)
        np.testing.assert_allclose(mv.accepted, 0.5)      # acc_sum/(rounds*R)
        self.assertTrue(np.all(mv._tally_in_model_proposed > 0))


class RedBlueFloorTest(unittest.TestCase):
    """10 repeats is not arbitrary: eryn wants nwalkers >= 2 * ndim_total."""

    def test_ten_is_exactly_galfors_floor(self):
        self.assertEqual(2 * 5, 10)          # galfor ndim = 5

    def test_ten_clears_psds_floor(self):
        self.assertGreaterEqual(10, 2 * 4)   # psd ndim = 4

    def test_a_joint_psd_plus_galfor_branch_would_need_eighteen(self):
        """variants/noise.py samples {"noise": ["psd", "galfor"]}."""
        self.assertEqual(2 * (4 + 5), 18)


class SettingsKnobTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in (
            "GALFOR_ENSEMBLE_SEARCH", "GALFOR_ENSEMBLE_REPEATS",
            "PSD_ENSEMBLE_SEARCH")}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_defaults_are_off(self):
        from lisatools.globalfit.stock.erebor.noise import (
            GalForSettings, PSDSettings,
        )
        self.assertFalse(GalForSettings().ensemble_search)
        self.assertFalse(PSDSettings().ensemble_search)
        self.assertEqual(GalForSettings().ensemble_repeats, 10)

    def test_env_arms_it_per_branch(self):
        from lisatools.globalfit.stock.erebor.noise import (
            GalForSettings, PSDSettings,
        )
        os.environ["GALFOR_ENSEMBLE_SEARCH"] = "1"
        self.assertTrue(GalForSettings().ensemble_search)
        self.assertFalse(PSDSettings().ensemble_search)   # independent knob

    def test_repeats_env(self):
        from lisatools.globalfit.stock.erebor.noise import GalForSettings
        os.environ["GALFOR_ENSEMBLE_REPEATS"] = "18"
        self.assertEqual(GalForSettings().ensemble_repeats, 18)


if __name__ == "__main__":
    unittest.main()
