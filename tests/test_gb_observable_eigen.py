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
