"""GB combined proposal: OBSERVABLE basis + information-matrix EIGENBASIS.

User ruling 2026-09-14: "GB in model should always be observed basis. We
should combine that with the eigenbasis." The in-model step stays a
symmetric draw in the internal observable coordinates z (factors = the
same log-Jacobian as today), but instead of independent per-coordinate
steps it can jump along the eigenvectors of the information matrix
CONGRUENCED INTO z, whitened by the existing analytic step scales — so a
diagonal Gamma_z reduces the combined proposal exactly to the current
one, and correlations (the extrinsic block) are what the eigenbasis adds.

Knob: GB_INMODEL_OBSERVABLE_EIGEN = 0 (default, bit-identical) | 1/axis
(one eigen-axis per repeat) | full (joint correlated step).
"""

import functools
import os
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _observable_eigen_mode,
)

IN_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
            "sin_delta", "fdot_astro_ratio"]
DIST, F0, MC, R = 0, 1, 2, 8
NDIM = 9
TOBS = 7864320.0
DF = 1.0 / TOBS
FIBER = 8  # Mc in the internal basis

FLAGSHIP = np.array([9.05215813, 20.3803767, 0.465777687, -3.41840873,
                     -0.883028, 2.09992313, 5.51968548, -0.2445, 0.0])


def _coords(n=5, seed=3):
    rng = np.random.default_rng(seed)
    c = np.repeat(FLAGSHIP[None, :], n, axis=0)
    c[:, DIST] *= np.exp(rng.normal(0.0, 0.15, n))
    c[:, F0] += rng.normal(0.0, 1e-4, n)
    c[:, MC] *= np.exp(rng.normal(0.0, 0.05, n))
    c[:, R] += rng.normal(0.0, 0.05, n)
    return c


def _stub(**over):
    from eryn.prior import ProbDistContainer, uniform_dist

    s = types.SimpleNamespace(
        xp=np,
        name="rj_fstat_search",
        branch_name="gb",
        _dist_col=DIST, _mc_col=MC, _fdot_astro_col=R, _f0_col=F0,
        # per-branch observable hooks (the VGB reduction parameterizes
        # these; GB takes the class defaults)
        _observable_map_class=GBSpecialStretchMove._observable_map_class,
        _observable_required_cols=(
            GBSpecialStretchMove._observable_required_cols),
        # Step-scale knob NAME (split GB/VGB 2026-09-19). Bound to the real
        # class method rather than the literal so the stub cannot drift
        # from it; the method ignores self, hence the None.
        _obs_jump_knob=functools.partial(
            GBSpecialStretchMove._obs_jump_knob, None),
        _eigen_axis_min_dim=NDIM,
        _eigen_axis_widths_cache=None,
        _observable_map_cache=None,
        _obs_rho=None,
        _obs_gamma_z=None,
        _obs_eigen_table=None,
        _last_im_kind=None,
        jump_factor=1.2,
        stretch_probability=0.0,
        time=0,
        df=DF,
        transform_fn=types.SimpleNamespace(input_basis=list(IN_BASIS)),
        _proposal_param_scales=np.ones(NDIM),
        gpu_priors={"gb": ProbDistContainer({
            0: uniform_dist(0.01, 18.0),
            1: uniform_dist(0.1, 26.0),
            2: uniform_dist(0.1, 1.0),
            3: uniform_dist(-2 * np.pi, 2 * np.pi),
            4: uniform_dist(-1.0, 1.0),
            5: uniform_dist(0.0, np.pi),
            6: uniform_dist(0.0, 2 * np.pi),
            7: uniform_dist(-1.0, 1.0),
            8: uniform_dist(-5.0, 5.0),
        })},
    )
    for k, v in over.items():
        setattr(s, k, v)
    for meth in ("_observable_basis_ready", "_observable_map",
                 "_observable_step_scales", "_observable_proposal",
                 "_eigen_axis_widths", "_observable_stash_gamma_z",
                 "_observable_eigen_prepare", "_inmodel_kind",
                 "_obs_eigen_mode"):
        setattr(s, meth, getattr(GBSpecialStretchMove, meth).__get__(s))
    return s


class KnobTest(unittest.TestCase):
    def test_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_OBSERVABLE_EIGEN", None)
            self.assertEqual(_observable_eigen_mode(), "off")

    def test_axis_and_full(self):
        for raw, want in (("1", "axis"), ("axis", "axis"),
                          ("full", "full"), ("0", "off"),
                          ("off", "off")):
            with mock.patch.dict(
                os.environ, {"GB_INMODEL_OBSERVABLE_EIGEN": raw}
            ):
                self.assertEqual(_observable_eigen_mode(), want, raw)

    def test_junk_warns_and_stays_off(self):
        with mock.patch.dict(
            os.environ, {"GB_INMODEL_OBSERVABLE_EIGEN": "banana"}
        ):
            with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch",
                level="WARNING",
            ):
                self.assertEqual(_observable_eigen_mode(), "off")


class GammaZChainRuleTest(unittest.TestCase):
    """Gamma_z must be the exact pullback of Gamma_x through from_internal:
    dz^T Gamma_z dz == dx^T Gamma_x dx for dx = x(z+dz) - x(z)."""

    def test_quadratic_form_invariance(self):
        rng = np.random.default_rng(11)
        n = 4
        coords = _coords(n)
        s = _stub()
        a = rng.standard_normal((n, NDIM, NDIM))
        info_y = a @ np.swapaxes(a, -1, -2) + 0.5 * np.eye(NDIM)
        # s = ones => Gamma_x == info_y
        s._observable_stash_gamma_z(
            info_y, coords, np.ones(NDIM), np.arange(n), n
        )
        gz = s._obs_gamma_z
        self.assertEqual(gz.shape, (n, NDIM, NDIM))
        self.assertTrue(np.all(np.isfinite(gz)))

        m = s._observable_map()
        z = np.asarray(m.to_internal(coords))
        x0 = np.asarray(m.from_internal(z, template=coords))
        # small dz scaled per column magnitude
        col = np.maximum(np.abs(z).max(axis=0), 1e-3)
        col[2] = max(abs(float(z[:, 2].mean())), 1e-18)  # fdot
        for _ in range(3):
            dz = 1e-6 * col * rng.standard_normal(NDIM)
            x1 = np.asarray(m.from_internal(z + dz, template=coords))
            dx = x1 - x0
            qz = np.einsum("i,nij,j->n", dz, gz, dz)
            qx = np.einsum("ni,nij,nj->n", dx, info_y, dx)
            np.testing.assert_allclose(qz, qx, rtol=2e-3)

    def test_scatter_by_source_id(self):
        n_src = 10
        ids = np.array([2, 7])
        coords = _coords(2)
        s = _stub()
        info_y = np.repeat(np.eye(NDIM)[None], 2, axis=0)
        s._observable_stash_gamma_z(
            info_y, coords, np.ones(NDIM), ids, n_src
        )
        gz = s._obs_gamma_z
        self.assertEqual(gz.shape, (n_src, NDIM, NDIM))
        self.assertTrue(np.all(np.isfinite(gz[ids])))
        others = np.setdiff1d(np.arange(n_src), ids)
        self.assertTrue(np.all(np.isnan(gz[others])))


class PrepareReductionTest(unittest.TestCase):
    """A DIAGONAL Gamma_z must reduce the eigen table to per-coordinate
    steps with exactly the diagonal-path widths (the 'combine' contract:
    the eigenbasis only ADDS the rotations)."""

    def _prepped(self):
        n = 3
        coords = _coords(n)
        s = _stub()
        s._obs_rho = np.full(20, 40.0)
        ids = np.arange(n)
        w = np.asarray(s._observable_step_scales(None, ids, NDIM))
        rng = np.random.default_rng(23)
        c = rng.uniform(0.5, 4.0, (n, NDIM))  # distinct curvatures
        gz = np.zeros((20, NDIM, NDIM)) * np.nan
        for k in range(n):
            gz[k] = np.diag(c[k] / w[k] ** 2)
        s._obs_gamma_z = gz
        s._observable_eigen_prepare(None, ids, 20)
        return s, ids, w, c

    def test_diagonal_gamma_gives_coordinate_axes_with_scaled_widths(self):
        s, ids, w, c = self._prepped()
        tab = s._obs_eigen_table
        self.assertEqual(tab.shape[1:], (NDIM, NDIM))
        for row, k in enumerate(ids):
            T = np.asarray(tab[k])
            # every non-fiber coordinate direction appears as exactly one
            # column sigma_j * e_j with sigma_j = w_j / sqrt(c_j)
            for j in range(NDIM):
                if j == FIBER:
                    continue
                cols = np.nonzero(
                    np.abs(np.abs(T[j, :]) -
                           np.linalg.norm(T, axis=0)) < 1e-10
                )[0]
                hits = [cc for cc in cols
                        if abs(np.linalg.norm(T[:, cc])
                               - w[row, j] / np.sqrt(c[row, j])) < 1e-8]
                self.assertTrue(hits, f"row {row} coord {j}")

    def test_unset_rows_stay_nan(self):
        s, ids, _, _ = self._prepped()
        others = np.setdiff1d(np.arange(20), ids)
        self.assertTrue(np.all(np.isnan(s._obs_eigen_table[others])))


class DrawTest(unittest.TestCase):
    def _stub_with_table(self, n, mode, seed=31):
        # fiber-structured table, as _observable_eigen_prepare produces:
        # 8 orthonormal WHITENED axes with ZERO Mc component + the pure-
        # fiber axis last (dropped from picks at fiber weight 0), scaled
        # per z-COORDINATE by physical widths (the fdot column lives at
        # ~1e-16 -- uniform column steps are unphysical and NaN the map)
        s = _stub()
        s._obs_rho = np.full(50, 40.0)
        rng = np.random.default_rng(seed)
        q8, _ = np.linalg.qr(rng.standard_normal((NDIM - 1, NDIM - 1)))
        q = np.zeros((NDIM, NDIM))
        rows = [i for i in range(NDIM) if i != FIBER]
        for a, ra in enumerate(rows):
            q[ra, :NDIM - 1] = q8[a]
        q[FIBER, NDIM - 1] = 1.0
        colscale = np.array([1e-2, 1e-5, 1e-20, 1e-2, 1e-2, 1e-2, 1e-2,
                             1e-2, 1e-2])
        sig = 0.1 * (1.0 + np.arange(NDIM))
        table = np.full((50, NDIM, NDIM), np.nan)
        ids = np.arange(n)
        for k in ids:
            table[k] = (colscale[:, None] * q) * sig[None, :]
        s._obs_eigen_table = table
        return s, ids, table[0], sig

    def test_axis_mode_steps_along_one_table_axis(self):
        n = 6
        coords = _coords(n)
        s, ids, T0, sig = self._stub_with_table(n, "axis")
        with mock.patch.dict(
            os.environ, {"GB_INMODEL_OBSERVABLE_EIGEN": "axis",
                         "GB_INMODEL_OBSERVABLE_FIBER_WEIGHT": "0.0"}
        ):
            np.random.seed(7)
            new, factors = s._observable_proposal(coords, None, ids)
        m = s._observable_map()
        dz = np.asarray(m.to_internal(new)) - np.asarray(
            m.to_internal(coords))
        for i in range(n):
            # recover the per-axis coefficients against the table columns
            # (solve, NOT lstsq: the fdot row makes cond(T0) ~ 1e18 and
            # lstsq's rcond cutoff would truncate that direction, smearing
            # the projection across every component)
            comps = np.linalg.solve(T0, dz[i])
            big = np.abs(comps) > 1e-6 * np.abs(comps).max()
            self.assertEqual(int(big.sum()), 1, f"row {i}: {comps}")
            # the pure-fiber weighting still zeroes any Mc motion
            self.assertLess(abs(dz[i, FIBER]), 1e-24)
        # factors are STILL the observable log-Jacobian difference
        want = np.asarray(m.factors(coords, new)).ravel()
        np.testing.assert_allclose(np.asarray(factors), want, rtol=1e-12)

    def test_off_mode_never_touches_the_table(self):
        n = 3
        coords = _coords(n)
        s = _stub()
        s._obs_rho = np.full(50, 40.0)
        poison = mock.MagicMock()
        poison.__getitem__ = mock.Mock(
            side_effect=AssertionError("table touched with knob off"))
        s._obs_eigen_table = poison
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_OBSERVABLE_EIGEN", None)
            new, factors = s._observable_proposal(
                coords, None, np.arange(n))
        self.assertEqual(new.shape, coords.shape)

    def test_nan_rows_fall_back_to_the_diagonal_draw(self):
        n = 4
        coords = _coords(n)
        s = _stub()
        s._obs_rho = np.full(50, 40.0)
        s._obs_eigen_table = np.full((50, NDIM, NDIM), np.nan)
        with mock.patch.dict(
            os.environ, {"GB_INMODEL_OBSERVABLE_EIGEN": "axis"}
        ):
            np.random.seed(9)
            new, _ = s._observable_proposal(coords, None, np.arange(n))
        m = s._observable_map()
        dz = np.asarray(m.to_internal(new)) - np.asarray(
            m.to_internal(coords))
        # diagonal draw moves every non-fiber observable, not one axis
        nz = (np.abs(dz[:, :FIBER]) > 0).sum(axis=1)
        self.assertTrue(np.all(nz > 1), nz)


if __name__ == "__main__":
    unittest.main()


class PerAxisCensusTest(unittest.TestCase):
    """One axis is drawn per row, so an axis's accepted/proposed ratio
    IS its own acceptance -- and a healthy POOLED rate hides a timid
    axis completely.

    Job 663: cold acceptance 0.55-0.79 while mean |df_mid| accepted /
    proposed was 0.01-0.34 and |dln_fdot| 0.04-0.10, i.e. f_mid and
    fdot were accepting a small fraction of their own draws while the
    other seven axes ran at ~80% to make up the pooled number. The
    two-axis line could not show that; this one can.
    """

    NA = 9

    def _move(self, pick, absz, ok=None, na=None):
        from types import SimpleNamespace
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        m._last_obs_axis_pick = np.asarray(pick)
        m._last_obs_axis_absz = np.asarray(absz, dtype=float)
        m._last_obs_axis_ok = (np.ones(len(pick), bool) if ok is None
                               else np.asarray(ok, bool))
        m._eigen_axis_min_dim = self.NA if na is None else na
        m._obs_axis_acc = None
        m.name = "rj_warm_search"
        return m

    def _accum(self, m, accept, d_fmid=None, d_lnfd=None, ok=None):
        import lisatools.globalfit.moves.gbspecialstretch as g
        n = len(accept)
        a = np.asarray(accept, dtype=float)
        g.GBSpecialBase._obs_axis_accum(
            m, a,
            np.ones(n) if d_fmid is None else np.asarray(d_fmid, float),
            np.ones(n) if d_lnfd is None else np.asarray(d_lnfd, float),
            np.ones(n, bool) if ok is None else np.asarray(ok, bool),
            np)
        return m._obs_axis_acc

    # (a) the counts reconcile with the pooled totals
    def test_per_axis_draws_and_accepts_SUM_to_the_pooled_totals(self):
        pick = [0, 1, 1, 2, 8, 8, 8]
        acc = [1, 0, 1, 1, 0, 0, 1]
        t = self._accum(self._move(pick, np.ones(len(pick))), acc)
        self.assertEqual(int(t[0].sum()), len(pick))
        self.assertEqual(int(t[1].sum()), sum(acc))

    def test_each_axis_gets_its_OWN_draws_and_accepts(self):
        pick = [1, 1, 1, 2]
        acc = [1, 0, 0, 1]
        t = self._accum(self._move(pick, np.ones(4)), acc)
        self.assertEqual((int(t[0, 1]), int(t[1, 1])), (3, 1))
        self.assertEqual((int(t[0, 2]), int(t[1, 2])), (1, 1))
        self.assertEqual(int(t[0, 0]), 0)

    def test_the_df_mid_means_are_attributed_to_the_DRAWN_axis(self):
        pick = [1, 2]
        t = self._accum(self._move(pick, np.ones(2)), [1, 1],
                        d_fmid=[0.5, 0.1])
        self.assertAlmostEqual(float(t[4, 1]), 0.5)
        self.assertAlmostEqual(float(t[4, 2]), 0.1)

    def test_the_step_size_is_recorded_in_the_axis_OWN_unit(self):
        """|zz| is the whitened step -- what a per-axis multiplier
        scales -- not the observable-space delta."""
        t = self._accum(self._move([3, 3], [2.0, 4.0]), [1, 0])
        self.assertAlmostEqual(float(t[2, 3]), 6.0)      # proposed
        self.assertAlmostEqual(float(t[3, 3]), 2.0)      # accepted only

    # the guards
    def test_a_DIAGONAL_fallback_row_is_attributed_to_NO_axis(self):
        """Rows with no eigen table take the diagonal draw; counting
        them under an axis would corrupt exactly the number the
        adaptation will read."""
        t = self._accum(self._move([0, 1], np.ones(2), ok=[True, False]),
                        [1, 1])
        self.assertEqual(int(t[0].sum()), 1)
        self.assertEqual(int(t[0, 0]), 1)
        self.assertEqual(int(t[0, 1]), 0)

    def test_a_LENGTH_MISMATCH_skips_the_tally_entirely(self):
        """If the gate compacted rows the pick no longer aligns, and a
        wrong attribution is worse than none."""
        m = self._move([0, 1, 2], np.ones(3))
        t = self._accum(m, [1, 0])          # 2 accepts vs 3 picks
        self.assertIsNone(t)

    def test_no_pick_means_no_tally(self):
        """full / diagonal modes draw no single axis and clear it."""
        m = self._move([0], [1.0])
        m._last_obs_axis_pick = None
        self.assertIsNone(self._accum(m, [1]))

    def test_the_pick_is_CONSUMED_so_a_repeat_cannot_double_count(self):
        m = self._move([0, 0], np.ones(2))
        self._accum(m, [1, 1])
        self.assertIsNone(m._last_obs_axis_pick)
        self._accum(m, [1, 1])              # second call, stale pick
        self.assertEqual(int(m._obs_axis_acc[0].sum()), 2)

    # wiring
    def test_the_draw_site_records_the_pick(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._observable_proposal)
        self.assertIn("self._last_obs_axis_pick = pick", src)
        self.assertIn("self._last_obs_axis_absz", src)
        self.assertIn("self._last_obs_axis_pick = None", src,
                      "full/diagonal must clear it")

    def test_the_two_axis_line_is_still_emitted(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._report_obs_motion)
        self.assertIn("in-model motion -- draws %d accepted %d", src)
        self.assertIn("_report_obs_axis", src)


class AxisMultSeamTest(unittest.TestCase):
    """``axis_mult`` may enter the step scales; ``coords`` may not."""

    def _args(self, n=3):
        from lisatools.sampling.gb_observable_basis import GB_INTERNAL_BASIS
        return dict(
            snr=np.full(n, 20.0), tobs=1.0e7,
            extrinsic_scales=np.ones((n, 5)), mc_step=1e-3,
            internal_basis=GB_INTERNAL_BASIS)

    def test_the_signature_STILL_cannot_see_coords(self):
        """The property the function exists to guarantee."""
        import inspect
        from lisatools.sampling import gb_observable_basis as m
        sig = inspect.signature(m.gb_observable_step_scales)
        self.assertNotIn("coords", sig.parameters)
        self.assertIn("axis_mult", sig.parameters)

    def test_None_is_byte_identical_to_before(self):
        from lisatools.sampling.gb_observable_basis import (
            gb_observable_step_scales as f)
        a = f(jump=2.0, **self._args())
        b = f(jump=2.0, axis_mult=None, **self._args())
        np.testing.assert_array_equal(a, b)

    def test_it_multiplies_ON_TOP_of_jump(self):
        from lisatools.sampling.gb_observable_basis import (
            gb_observable_step_scales as f)
        base = f(jump=2.0, **self._args())
        mult = np.full(base.shape, 3.0)
        got = f(jump=2.0, axis_mult=mult, **self._args())
        np.testing.assert_allclose(got, base * 3.0)

    def test_a_MISSHAPEN_multiplier_raises_rather_than_broadcasting(self):
        """A silently broadcast multiplier scales the wrong axis for
        every row."""
        from lisatools.sampling.gb_observable_basis import (
            gb_observable_step_scales as f)
        with self.assertRaises(ValueError):
            f(jump=1.0, axis_mult=np.ones(3), **self._args())

    def test_the_seam_defaults_to_None(self):
        """A+B is telemetry only: the step must be unchanged."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        self.assertIsNone(g._obs_axis_mult_for(object(), None, 9))

    def test_the_seam_is_MODULE_LEVEL_so_stubs_do_not_crash(self):
        """``_observable_proposal`` is driven by duck-typed stubs in
        these very tests and by rank processes; a bound method would
        make every such object grow one or crash the draw path. Same
        rationale as fstat_band_min_F_for, and the first version of
        this WAS a method and broke five existing DrawTest cases."""
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        self.assertTrue(inspect.isfunction(g._obs_axis_mult_for))
        self.assertFalse(hasattr(g.GBSpecialBase, "_obs_axis_mult"))

    def test_a_move_that_DOES_define_one_is_used(self):
        from types import SimpleNamespace
        import lisatools.globalfit.moves.gbspecialstretch as g
        want = np.full((2, 9), 4.0)
        mv = SimpleNamespace(_obs_axis_mult=lambda ids, n: want)
        np.testing.assert_array_equal(
            g._obs_axis_mult_for(mv, None, 9), want)


class AxisIdentityIsStableTest(unittest.TestCase):
    """The eigen table's COLUMN INDEX has to mean something.

    ``eigen_axis_set`` orders columns by ``|overlap with the fiber|``.
    The fiber is projected out EXACTLY, so the other eight eigenvectors
    are orthogonal to it to machine precision and that sort key is
    rounding noise -- a permutation redrawn every block. Three things
    depend on the index being an identity: the per-axis census labels,
    a per-axis multiplier learned between blocks, and the per-band mean
    table a newborn is initialised from.
    """

    D = 9
    LOGGER = "lisatools.globalfit.moves.gbspecialstretch"

    def _batch(self, n=400, seed=5, spread=1.0):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(n, self.D, self.D))
        g = np.einsum("nij,nkj->nik", A, A)
        _, evec = np.linalg.eigh(g)
        lam = 10.0 ** rng.uniform(-spread, spread, size=(n, self.D))
        gw = np.einsum("nij,nj,nkj->nik", evec, lam, evec)
        pert = gw * (1.0 + 0.01 * rng.normal(size=gw.shape))
        return gw, 0.5 * (pert + np.swapaxes(pert, -1, -2))

    def _axes(self, gw, reorder, fiber=True):
        from eryn.moves.eigenaxis import eigen_axis_set
        import lisatools.globalfit.moves.gbspecialstretch as g
        n = int(gw.shape[0])
        tf = None
        if fiber:
            tf = np.zeros((n, self.D))
            tf[:, self.D - 1] = 1.0
        a, s = eigen_axis_set(gw, t_fiber=tf, sigma_max=10.0)
        if not reorder:
            return a, s, None
        return g.obs_axis_reorder(a, s, gw, fiber, np)

    @staticmethod
    def _held(a1, a2, ncol):
        return float((np.abs(np.einsum("nik,nik->nk", a1, a2))[:, :ncol]
                      > 0.9).mean())

    # --- the pair that makes the reorder non-deletable -------------------
    def test_column_k_SURVIVES_a_1pct_change_to_the_matrix(self):
        gw, pert = self._batch()
        a1, _, _ = self._axes(gw, True)
        a2, _, _ = self._axes(pert, True)
        held = self._held(a1, a2, self.D - 1)
        self.assertGreater(
            held, 0.9,
            f"only {held:.3f} of non-fiber columns kept their direction; "
            "a multiplier learned last block would land on a different axis")

    def test_NEGATIVE_CONTROL_the_shipped_order_does_NOT_survive_it(self):
        """Delete the reorder and the test above becomes this number."""
        gw, pert = self._batch()
        a1, _, _ = self._axes(gw, False)
        a2, _, _ = self._axes(pert, False)
        held = self._held(a1, a2, self.D - 1)
        self.assertLess(
            held, 0.3,
            f"{held:.3f} -- the |fiber overlap| order was expected to be "
            "noise; if this passes the premise of obs_axis_reorder is gone")

    def test_the_sort_key_is_ROUNDING_NOISE_without_the_reorder(self):
        from eryn.moves.eigenaxis import project_out_direction
        gw, _ = self._batch(n=200)
        tf = np.zeros((200, self.D))
        tf[:, self.D - 1] = 1.0
        _, evecs = np.linalg.eigh(project_out_direction(gw, tf))
        ov = np.sort(np.abs(np.einsum("ni,nij->nj", tf, evecs)), axis=-1)
        self.assertLess(float(np.median(ov[:, self.D - 2])), 1e-10)
        self.assertGreater(float(np.median(ov[:, self.D - 1])), 0.99)

    # --- what the new order MEANS ----------------------------------------
    def test_axis_0_is_the_WIDEST_and_the_last_non_fiber_the_tightest(self):
        gw, _ = self._batch()
        a, _, _ = self._axes(gw, True)
        quad = np.einsum("nik,nij,njk->nk", a, gw, a)[:, :self.D - 1]
        self.assertTrue(np.all(np.diff(quad, axis=-1) >= -1e-9),
                        "curvature must ascend => width must descend")

    def test_the_sort_key_is_the_UNCAPPED_width(self):
        """``sig`` is capped at ``sigma_max``, so every railed axis
        carries the same value; sorting on it would leave exactly the
        near-null axes in an arbitrary order again."""
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.obs_axis_reorder)
        self.assertIn('xp.einsum("nik,nij,njk->nk", axes, gw, axes)', src)
        gw, _ = self._batch()
        gw = gw * 1e-12                          # everything rails
        a, s, _ = self._axes(gw, True)
        self.assertTrue(np.allclose(s, 10.0), "precondition: all railed")
        quad = np.einsum("nik,nij,njk->nk", a, gw, a)[:, :self.D - 1]
        self.assertTrue(np.all(np.diff(quad, axis=-1) >= -1e-30))

    def test_the_fiber_STAYS_in_the_last_column(self):
        """``_observable_proposal`` drops the last column from the pick
        set and weights it separately; moving it would silently start
        proposing along the flat direction."""
        gw, _ = self._batch(n=200)
        a0, _, _ = self._axes(gw, False)
        a1, _, _ = self._axes(gw, True)
        np.testing.assert_allclose(a1[:, :, -1], a0[:, :, -1], atol=1e-12)

    def test_it_is_a_PERMUTATION_so_the_draw_DISTRIBUTION_is_unchanged(self):
        """axis mode picks uniformly and full mode contracts with iid
        normals -- both are invariant under a column permutation."""
        gw, _ = self._batch(n=150)
        a0, s0, _ = self._axes(gw, False)
        a1, s1, _ = self._axes(gw, True)
        np.testing.assert_allclose(np.sort(s0, axis=-1),
                                   np.sort(s1, axis=-1), atol=1e-12)
        for src, out in ((a0, a1),):
            k0 = np.sort(np.abs(src).sum(axis=1), axis=-1)
            k1 = np.sort(np.abs(out).sum(axis=1), axis=-1)
            np.testing.assert_allclose(k0, k1, atol=1e-10)

    def test_a_FIBERLESS_layout_sorts_EVERY_column(self):
        """The VGB restriction pins ``Mc``: there is no fiber column to
        hold back, and leaving the last one unsorted would exempt it."""
        gw, _ = self._batch(n=150)
        a, _, _ = self._axes(gw, True, fiber=False)
        quad = np.einsum("nik,nij,njk->nk", a, gw, a)
        self.assertTrue(np.all(np.diff(quad, axis=-1) >= -1e-9))

    def test_dom_names_the_coordinate_that_dominates_each_axis(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        axes = np.zeros((2, 3, 3))
        axes[0, 2, 0] = axes[0, 0, 1] = axes[0, 1, 2] = 1.0
        axes[1] = axes[0]
        gw = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
        _, _, dom = g.obs_axis_reorder(axes, np.ones((2, 3)), gw, False, np)
        # identity gw => quad all 1 => stable sort keeps the order
        np.testing.assert_array_equal(dom[0], [2, 0, 1])


class CensusLabelsAreRanksNotCoordinatesTest(unittest.TestCase):
    """The census line used to label bucket ``k`` ``GB_INTERNAL_BASIS[k]``.

    That reads as "the f_mid axis accepted 7%" and is what the launcher's
    2.0 -> 1.5 decision was argued from. Bucket ``k`` is an axis RANK.
    """

    LOGGER = "lisatools.globalfit.moves.gbspecialstretch"

    def _move(self, nax=9):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        t = np.zeros((6, nax))
        t[0] = 10.0                       # draws
        t[1] = np.arange(nax)             # accepts
        m._obs_axis_acc = t
        m._obs_axis_dom = None
        m.name = "in_model"
        return m

    def _line(self, m):
        import lisatools.globalfit.moves.gbspecialstretch as g
        with self.assertLogs(self.LOGGER, "INFO") as cm:
            g.GBSpecialBase._report_obs_axis(m)
        return "\n".join(cm.output)

    def test_the_buckets_are_labelled_by_RANK(self):
        out = self._line(self._move())
        self.assertIn("s0 ", out)
        self.assertIn("s7 ", out)
        self.assertIn("fib ", out)
        self.assertIn("s0=widest", out)

    def test_a_bucket_is_NOT_labelled_with_the_coordinate_at_its_index(self):
        out = self._line(self._move())
        for bad in ("lnA d=", "f_mid d=", "fdot d=", "phi0 d="):
            self.assertNotIn(bad, out, f"{bad!r} names a coordinate, not "
                                       "the axis that bucket holds")

    def test_dom_names_the_coordinate_the_rank_is_USUALLY_made_of(self):
        m = self._move()
        h = np.zeros((9, 9))
        h[0, 5] = 30.0                   # rank 0 dominated by psi
        h[0, 1] = 10.0
        m._obs_axis_dom = h
        out = self._line(m)
        self.assertIn("s0 dom=psi(0.75)", out)

    def test_the_histogram_counts_the_dominant_coordinate_per_rank(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        m._obs_axis_dom = None
        # (n_src, n_axes); the eigen table is square, so a dominant
        # coordinate index and an axis index share the same range.
        dom = np.array([[2, 0, 1], [2, 1, 0]])
        g.GBSpecialBase._obs_axis_dom_accum(m, dom, None, np)
        self.assertEqual(m._obs_axis_dom.shape, (3, 3))
        self.assertEqual(float(m._obs_axis_dom[0, 2]), 2.0)
        self.assertEqual(float(m._obs_axis_dom[1, 0]), 1.0)
        self.assertEqual(float(m._obs_axis_dom[1, 1]), 1.0)

    def test_rows_with_NO_finite_table_are_left_out_of_the_histogram(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        m._obs_axis_dom = None
        g.GBSpecialBase._obs_axis_dom_accum(
            m, np.array([[0, 1], [1, 0]]), np.array([True, False]), np)
        self.assertEqual(float(m._obs_axis_dom.sum()), 2.0)

    def test_the_table_build_FEEDS_the_histogram(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._observable_eigen_prepare)
        self.assertIn("obs_axis_reorder(", src)
        self.assertIn("_obs_axis_dom_accum_for(", src)

    def test_the_hook_is_MODULE_LEVEL_so_a_stub_without_it_is_fine(self):
        from types import SimpleNamespace
        import lisatools.globalfit.moves.gbspecialstretch as g
        g._obs_axis_dom_accum_for(SimpleNamespace(), None, None, np)


class PerAxisScaleReachesTheStepTest(unittest.TestCase):
    """A per-axis multiplier has exactly one place it can go.

    The step scales ``w`` are consumed as a WHITENING METRIC:
    ``gw = gz * w (x) w`` and ``sigma_w = 1/sqrt(a^T gw a)``, so
    ``w -> c w`` gives ``sigma_w -> sigma_w / c`` and the table product
    ``w * a_w * sigma_w`` comes out unchanged. ``axis_mult`` and
    ``GB_INMODEL_OBSERVABLE_JUMP`` both enter there, so on the eigen path
    they move only the axes railed at ``SMAX``. ``obs_axis_scale_for``
    enters at ``sigma_w``, where nothing cancels it.
    """

    N, POOL = 4, 20

    def _prep(self, curv=None, scale=None, jump=None, axis_mult=None):
        s = _stub()
        s._obs_rho = np.full(self.POOL, 40.0)
        ids = np.arange(self.N)
        w = np.asarray(s._observable_step_scales(None, ids, NDIM))
        c = (np.full((self.N, NDIM), 1.0) if curv is None
             else np.asarray(curv, float))
        gz = np.full((self.POOL, NDIM, NDIM), np.nan)
        for k in range(self.N):
            gz[k] = np.diag(c[k] / w[k] ** 2)
        s._obs_gamma_z = gz
        if scale is not None:
            s._obs_axis_scale = lambda _ids, na, _xp, _g=scale: _g
        if axis_mult is not None:
            s._obs_axis_mult = lambda _ids, nz, _m=axis_mult: _m
        env = {} if jump is None else {
            "GB_INMODEL_OBSERVABLE_JUMP": str(jump)}
        with mock.patch.dict(os.environ, env):
            s._observable_eigen_prepare(None, ids, self.POOL)
        return np.linalg.norm(np.asarray(s._obs_eigen_table)[:self.N],
                              axis=1)

    # --- the new seam ----------------------------------------------------
    def test_the_multiplier_scales_axis_k_by_EXACTLY_g_k(self):
        rng = np.random.default_rng(4)
        curv = rng.uniform(0.5, 4.0, (self.N, NDIM))
        g = rng.uniform(0.3, 3.0, (self.N, NDIM))
        base = self._prep(curv=curv)
        got = self._prep(curv=curv, scale=g)
        np.testing.assert_allclose(got / base, g, rtol=1e-12)

    def test_None_leaves_the_table_byte_identical(self):
        curv = np.random.default_rng(6).uniform(0.5, 4.0, (self.N, NDIM))
        np.testing.assert_array_equal(self._prep(curv=curv),
                                      self._prep(curv=curv, scale=None))

    def test_a_MISSHAPEN_multiplier_raises_instead_of_broadcasting(self):
        with self.assertRaises(ValueError) as cm:
            self._prep(scale=np.ones((self.N, NDIM - 1)))
        self.assertIn("wrong axis", str(cm.exception))

    def test_the_hook_is_MODULE_LEVEL_so_a_stub_without_it_is_fine(self):
        from types import SimpleNamespace
        import lisatools.globalfit.moves.gbspecialstretch as g
        self.assertIsNone(
            g.obs_axis_scale_for(SimpleNamespace(), np.arange(3), 9, np))

    # --- why the OLD seam could not be used ------------------------------
    def test_NEGATIVE_CONTROL_the_same_factor_via_axis_mult_does_NOTHING(
            self):
        """``axis_mult`` is the seam that was built for this in
        `_obs_axis_mult_for`. On the eigen path it cancels exactly."""
        curv = np.random.default_rng(8).uniform(0.5, 4.0, (self.N, NDIM))
        base = self._prep(curv=curv)
        via_mult = self._prep(curv=curv,
                              axis_mult=np.full((self.N, NDIM), 2.5))
        np.testing.assert_allclose(np.sort(via_mult, axis=-1),
                                   np.sort(base, axis=-1), rtol=1e-10)

    def test_NEGATIVE_CONTROL_the_JUMP_knob_is_cancelled_too(self):
        """``GB_INMODEL_OBSERVABLE_JUMP`` enters at the same place. The
        launcher's 1.0 -> 2.0 -> 1.5 history was argued as a step-size
        change under ``*_OBSERVABLE_EIGEN=axis``; on every axis the
        information matrix could measure, it is not one."""
        curv = np.random.default_rng(10).uniform(0.5, 4.0, (self.N, NDIM))
        base = self._prep(curv=curv, jump=1.0)
        for j in (1.5, 2.0, 7.3):
            got = self._prep(curv=curv, jump=j)
            np.testing.assert_allclose(
                np.sort(got, axis=-1), np.sort(base, axis=-1), rtol=1e-10,
                err_msg=f"jump={j} moved a non-railed axis")

    def test_the_jump_knob_DOES_reach_an_axis_railed_at_SMAX(self):
        """Where it survives: the directions the matrix could not
        measure -- i.e. exactly the ones already overshooting."""
        curv = np.full((self.N, NDIM), 1e-6)     # sigma ~ 1e3 >> smax 10
        base = self._prep(curv=curv, jump=1.0)
        got = self._prep(curv=curv, jump=2.0)
        np.testing.assert_allclose(np.sort(got, axis=-1),
                                   2.0 * np.sort(base, axis=-1), rtol=1e-10)

    def test_the_multiplier_still_works_on_a_RAILED_axis(self):
        """It is applied AFTER the cap on purpose: the cap bounds what
        the information matrix may claim, not what the acceptance
        measured."""
        curv = np.full((self.N, NDIM), 1e-6)
        base = self._prep(curv=curv)
        got = self._prep(curv=curv, scale=np.full((self.N, NDIM), 0.25))
        np.testing.assert_allclose(got / base, 0.25, rtol=1e-12)


class AxisAdaptUpdateTest(unittest.TestCase):
    """The update rule itself, as a pure function."""

    def _u(self, g, nd, na, **kw):
        import lisatools.globalfit.moves.gbspecialstretch as m
        return m.obs_axis_adapt_update(
            np.asarray(g, float), np.asarray(nd, float),
            np.asarray(na, float), np, **kw)

    def test_accepting_ABOVE_target_widens_the_step(self):
        g = self._u([[1.0]], [[10]], [[10]], target=0.44, gain=0.2)
        self.assertGreater(float(g[0, 0]), 1.0)

    def test_accepting_BELOW_target_narrows_it(self):
        g = self._u([[1.0]], [[10]], [[0]], target=0.44, gain=0.2)
        self.assertLess(float(g[0, 0]), 1.0)

    def test_AT_target_is_a_fixed_point(self):
        g = self._u([[2.0]], [[100]], [[44]], target=0.44, gain=0.2)
        self.assertAlmostEqual(float(g[0, 0]), 2.0, places=12)

    def test_a_cell_with_NO_draws_is_returned_unchanged(self):
        """Most cells see no draw in a given propose; drifting them
        toward the initial value would be adaptation from no data."""
        g = self._u([[0.3, 5.0]], [[0, 0]], [[0, 0]])
        np.testing.assert_allclose(g, [[0.3, 5.0]], rtol=1e-12)

    def test_the_step_is_gain_times_the_rate_error_in_LOG_space(self):
        g = self._u([[1.0]], [[4]], [[3]], target=0.5, gain=0.4)
        self.assertAlmostEqual(float(np.log(g[0, 0])), 0.4 * (0.75 - 0.5))

    def test_the_multiplier_is_CLAMPED_both_ways(self):
        hi = self._u([[100.0]], [[10]], [[10]], bound=8.0)
        lo = self._u([[0.001]], [[10]], [[0]], bound=8.0)
        self.assertAlmostEqual(float(hi[0, 0]), 8.0)
        self.assertAlmostEqual(float(lo[0, 0]), 0.125)

    def test_each_cell_moves_on_its_OWN_evidence(self):
        g = self._u([[1.0, 1.0]], [[10, 10]], [[10, 0]], target=0.44,
                    gain=0.2)
        self.assertGreater(float(g[0, 0]), 1.0)
        self.assertLess(float(g[0, 1]), 1.0)


class AxisAdaptArmingTest(unittest.TestCase):
    """Adaptation is SEARCH-ONLY: a learned step scale is not a fixed
    proposal, and the PE stage has to keep detailed balance."""

    def _mv(self, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(**kw)

    def _on(self, mv, adapt="1"):
        import lisatools.globalfit.moves.gbspecialstretch as g
        with mock.patch.dict(
            os.environ, {"GB_INMODEL_OBSERVABLE_AXIS_ADAPT": adapt}
        ):
            return g.obs_axis_adapt_on(mv)

    def test_OFF_by_default(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_OBSERVABLE_AXIS_ADAPT", None)
            self.assertFalse(g.obs_axis_adapt_on(
                self._mv(_is_search_move=lambda: True)))

    def test_armed_for_a_SEARCH_move(self):
        self.assertTrue(self._on(self._mv(_is_search_move=lambda: True)))

    def test_NOT_armed_for_a_PE_move_even_with_the_knob_on(self):
        self.assertFalse(self._on(self._mv(_is_search_move=lambda: False)))

    def test_the_stamp_or_the_name_both_say_search(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        f = g.GBSpecialBase._is_search_move
        self.assertTrue(f(self._mv(name="rj_warm_search")))
        self.assertTrue(f(self._mv(name="in_model", gb_search_stage=True)))
        self.assertTrue(f(self._mv(name="rj_replace",
                                   replace_search_stage=True)))
        self.assertFalse(f(self._mv(name="in_model")))
        self.assertFalse(f(self._mv(name="rj_prior_removal")))

    def test_the_recipe_stamps_search_EXCLUSIVE_moves_only(self):
        """``gb_ridge_gibbs`` is ONE object in both lists; a single
        object cannot be in two stages, so it is left unstamped."""
        import inspect
        from lisatools.globalfit import recipe
        src = inspect.getsource(recipe.build_gb_moves)
        self.assertIn("_pe_ids = {id(_m) for _m in gb_pe_moves}", src)
        self.assertIn("_m.gb_search_stage = True", src)


class AxisAdaptCellKeyTest(unittest.TestCase):
    """The key is (temp, walker, band, axis)."""

    NT, NW, NB, NA = 3, 2, 4, 5

    def _mv(self, bound=True):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        # ``xp`` is a read-only property; every hook takes it explicitly.
        m.name = "rj_warm_search"
        m.gb_search_stage = True
        m._obs_axis_g = None
        m._obs_axis_nd = m._obs_axis_na_acc = None
        m._last_obs_axis_ids = None
        m._obs_axis_keys = None
        if bound:
            sorter = types.SimpleNamespace(
                temp_inds=np.array([0, 0, 2, 2]),
                walker_inds=np.array([0, 1, 0, 1]),
                band_inds=np.array([3, 3, 1, 1]),
                ntemps=self.NT, nwalkers=self.NW, num_bands=self.NB)
            g.GBSpecialBase._obs_axis_bind_keys(m, sorter)
        return m

    def test_the_dims_come_from_the_SORTER_not_from_the_rows(self):
        """A propose that populates no row on the top rung must not
        resize the table -- that would discard everything learned."""
        m = self._mv()
        self.assertEqual(m._obs_axis_keys[3], (self.NT, self.NW, self.NB))

    def test_no_sorter_means_no_key_and_no_multiplier(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = self._mv(bound=False)
        g.GBSpecialBase._obs_axis_bind_keys(m, None)
        self.assertIsNone(m._obs_axis_keys)
        self.assertIsNone(
            g.GBSpecialBase._obs_axis_scale(m, np.arange(4), self.NA, np))

    def test_rows_in_the_SAME_cell_share_a_multiplier(self):
        """Which is what gives a newborn its band's learned value for
        free -- there is no separate newborn table."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = self._mv()
        m._obs_axis_keys = (np.array([0, 0]), np.array([1, 1]),
                            np.array([2, 2]), (self.NT, self.NW, self.NB))
        gtab = np.ones((self.NT, self.NW, self.NB, self.NA))
        gtab[0, 1, 2] = np.arange(1, self.NA + 1)
        m._obs_axis_g = gtab
        out = g.GBSpecialBase._obs_axis_scale(m, np.array([0, 1]), self.NA, np)
        np.testing.assert_allclose(out[0], np.arange(1, self.NA + 1))
        np.testing.assert_allclose(out[1], out[0])

    def test_different_TEMPS_get_different_multipliers(self):
        """The optimal width goes as 1/sqrt(beta); keying on the rung is
        what lets the adaptation discover that."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = self._mv()
        gtab = np.ones((self.NT, self.NW, self.NB, self.NA))
        gtab[0, 0, 3] = 2.0
        gtab[2, 0, 1] = 0.5
        m._obs_axis_g = gtab
        out = g.GBSpecialBase._obs_axis_scale(m, np.arange(4), self.NA, np)
        np.testing.assert_allclose(out[0], 2.0)      # t0 w0 b3
        np.testing.assert_allclose(out[2], 0.5)      # t2 w0 b1

    def test_a_LAYOUT_change_raises_rather_than_scaling_the_wrong_axis(
            self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = self._mv()
        m._obs_axis_g = np.ones((self.NT, self.NW, self.NB, self.NA - 1))
        with self.assertRaises(ValueError):
            g.GBSpecialBase._obs_axis_scale(m, np.arange(4), self.NA, np)

    def test_no_table_yet_means_no_multiplier(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        self.assertIsNone(
            g.GBSpecialBase._obs_axis_scale(self._mv(), np.arange(4),
                                            self.NA, np))


class AxisAdaptEndToEndTest(unittest.TestCase):
    """Draws in, multipliers out."""

    NT, NW, NB, NA = 2, 1, 2, 3

    def _mv(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        # ``xp`` is a read-only property; every hook takes it explicitly.
        m.name = "rj_warm_search"
        m.gb_search_stage = True
        m._obs_axis_g = None
        m._obs_axis_nd = m._obs_axis_na_acc = None
        g.GBSpecialBase._obs_axis_bind_keys(m, types.SimpleNamespace(
            temp_inds=np.array([0, 0, 1, 1]),
            walker_inds=np.zeros(4, int),
            band_inds=np.array([0, 0, 1, 1]),
            ntemps=self.NT, nwalkers=self.NW, num_bands=self.NB))
        return m

    def _feed(self, m, pick, acc, ids=None):
        import lisatools.globalfit.moves.gbspecialstretch as g
        n = len(pick)
        m._last_obs_axis_ids = (np.arange(n) if ids is None
                                else np.asarray(ids))
        g.GBSpecialBase._obs_axis_adapt_accum(
            m, np.asarray(pick), np.ones(n),
            np.asarray(acc, float), self.NA, np)

    def _step(self, m, **env):
        import lisatools.globalfit.moves.gbspecialstretch as g
        e = {"GB_INMODEL_OBSERVABLE_AXIS_ADAPT": "1"}
        e.update({k: str(v) for k, v in env.items()})
        with mock.patch.dict(os.environ, e):
            g.GBSpecialBase._obs_axis_adapt_step(m, np)
        return m._obs_axis_g

    def test_an_always_accepting_cell_axis_WIDENS(self):
        m = self._mv()
        self._feed(m, [0, 0, 1, 1], [1, 1, 1, 1])
        g = self._step(m)
        self.assertGreater(float(g[0, 0, 0, 0]), 1.0)   # t0 b0 axis0
        self.assertGreater(float(g[1, 0, 1, 1]), 1.0)   # t1 b1 axis1
        self.assertAlmostEqual(float(g[0, 0, 0, 2]), 1.0)  # untouched

    def test_a_never_accepting_cell_axis_NARROWS(self):
        m = self._mv()
        self._feed(m, [2, 2, 2, 2], [0, 0, 0, 0])
        g = self._step(m)
        self.assertLess(float(g[0, 0, 0, 2]), 1.0)

    def test_the_table_PERSISTS_and_compounds_across_proposes(self):
        """The whole point: within one propose the source ids are
        stable, across proposes only the cell key is."""
        m = self._mv()
        seen = []
        for _ in range(3):
            self._feed(m, [0, 0, 0, 0], [1, 1, 1, 1])
            seen.append(float(self._step(m)[0, 0, 0, 0]))
        self.assertLess(seen[0], seen[1])
        self.assertLess(seen[1], seen[2])

    def test_the_counters_RESET_each_propose(self):
        m = self._mv()
        self._feed(m, [0, 0, 0, 0], [1, 1, 1, 1])
        self._step(m)
        self.assertIsNone(m._obs_axis_nd)
        first = float(m._obs_axis_g[0, 0, 0, 0])
        self._step(m)                       # no new evidence
        self.assertAlmostEqual(float(m._obs_axis_g[0, 0, 0, 0]), first)

    def test_a_PE_move_accumulates_NOTHING_and_keeps_no_table(self):
        m = self._mv()
        m.gb_search_stage = False
        m.name = "in_model"
        self._feed(m, [0, 0, 0, 0], [1, 1, 1, 1])
        self.assertIsNone(self._step(m))

    def test_a_length_MISMATCH_drops_the_tally_rather_than_misattribute(
            self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = self._mv()
        m._last_obs_axis_ids = np.arange(3)
        g.GBSpecialBase._obs_axis_adapt_accum(
            m, np.zeros(4, int), np.ones(4), np.ones(4), self.NA, np)
        self.assertIsNone(m._obs_axis_nd)

    def test_the_census_guards_carry_over_to_the_adaptation(self):
        """``w`` is the census's use-weight: a row that took the
        diagonal fallback is zero-weighted and cannot vote."""
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._obs_axis_accum)
        self.assertIn("_obs_axis_adapt_accum_for(self, pick, w, wa, na, xp)",
                      src)

    def test_the_draw_site_records_the_SOURCE_IDS(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._observable_proposal)
        self.assertIn("self._last_obs_axis_ids", src)

    def test_the_propose_boundary_runs_the_step_BEFORE_the_census(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase.run_proposal)
        i = src.index("self._obs_axis_adapt_step(self.xp)")
        j = src.index("self._report_obs_motion()")
        self.assertLess(i, j)

    def test_the_block_open_binds_the_keys(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertIn("self._obs_axis_bind_keys(band_sorter)", src)
