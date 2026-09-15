"""VGB in-model proposal = the GB OBSERVABLE composite, reduced.

User ruling 2026-09-15: "we should be sampling the VGBs just like the GBs
now (in terms of the basis/proposal type not RJ, still fixed model. f0
should be filled. sky location should be filled. Everything else is just
like the GB setup. We should be sampling still in the observed basis for
VGBs (just without f0, sky coords)" -- plus the hard directive to REUSE the
GB machinery rather than write a parallel VGB one.

So the reduced map is a subclass of the GB map with per-leaf pinned
constants, and the proposal is the SAME composite code path
(``_observable_proposal`` / ``_observable_stash_gamma_z`` /
``_observable_eigen_prepare`` / ``_observable_step_scales``) with the map
swapped.

THE DEAD COLUMN THIS ALSO FIXES (commit 2bb484e2's KNOWN LIMIT, confirmed
here on the REAL stock container): the VGB distance basis pins ``Mc``, so
``Mc`` occupies the physical ``fdot`` output slot through the container's
``key_map`` and ``fdot`` sits OUTSIDE ``test_inds``. The sampled
``fdot_astro_ratio``'s only scored target is then the ``fddot`` slot, which
``McDistFdotAstroQuad`` emits as exactly ``f0 * 0.0``. Result:
``J[:, :, r] == 0`` EXACTLY, ``info_y`` rank-deficient, and the eigen step
along ``r`` set by the prior box rather than by curvature -- a blind jump in
the one coordinate a known-f0/known-sky branch exists to measure.
``_infomat_phys_inds`` substitutes the live ``fdot`` slot for the dead
``fddot`` one; the tests below pin the before and the after.
"""

import os
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbbands as _gbbands
from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialBase,
    VGBSpecialStretchMove,
    _vgb_inmodel_defaults,
    _vgb_inmodel_proposal_kind,
    _vgb_observable_eigen_mode,
)
from lisatools.globalfit.stock.erebor.transforms import (
    make_gb_transform_container,
)
from lisatools.globalfit.stock.erebor.vgb import (
    VGB_FIXED_BASIS_CHIRP,
    VGB_FIXED_BASIS_DIST,
    VGB_SAMPLED_BASIS_CHIRP,
    VGB_SAMPLED_BASIS_DIST,
)
from lisatools.sampling.gb_observable_basis import (
    GB_INTERNAL_BASIS,
    OBSERVABLE_CD_FLOORS,
    OBSERVABLE_CD_FLOOR_DEFAULT,
    STEP_C_FDOT,
    STEP_C_LNA,
    GBObservableFiberBasis,
    VGBObservableBasis,
    fdot_gr,
    gb_observable_step_scales,
)

TOBS = 15768000.0                      # 6 months
DF = 1.0 / TOBS
NDIM = 5                               # the default VGB sampled width

#: the stock default VGB layout
IN_BASIS = list(VGB_SAMPLED_BASIS_DIST)     # dist phi0 cos_iota psi r
DIST, PHI0, CI, PSI, R = range(5)
#: internal (observable) slots of the reduced map
Z_LNA, Z_FDOT, Z_PHI0, Z_CI, Z_PSI = range(5)

#: two DIFFERENT physical sources; values in SAMPLING units (f0 in mHz)
LEAF_FILLS = [
    {"f0": 3.2001, "alpha": 0.7, "sin_delta": 0.2, "Mc": 0.30},
    {"f0": 9.4402, "alpha": 2.4, "sin_delta": -0.6, "Mc": 0.55},
]

#: rows 0/1 and 2/3 are the SAME coordinates visited by DIFFERENT leaves --
#: the only thing that can separate them is the per-leaf pinned constants
COORDS = np.array([
    [4.0, 1.10, 0.30, 0.80, 0.02],
    [4.0, 1.10, 0.30, 0.80, 0.02],
    [6.5, 2.00, -0.40, 1.90, -0.05],
    [6.5, 2.00, -0.40, 1.90, -0.05],
])
LEAVES = np.array([0, 1, 0, 1])


def _container(chirp=False):
    """The REAL stock VGB transform container (per-leaf fill LIST)."""
    if chirp:
        sampled, fixed = VGB_SAMPLED_BASIS_CHIRP, VGB_FIXED_BASIS_CHIRP
    else:
        sampled, fixed = VGB_SAMPLED_BASIS_DIST, VGB_FIXED_BASIS_DIST
    return make_gb_transform_container(
        use_chirp_mass=True, use_fdot_astro=True, use_distance=True,
        input_basis=list(sampled),
        fill_dict=[{k: d[k] for k in fixed} for d in LEAF_FILLS],
        mc_lims=(0.001, 1.0),
    )


def _gb_container():
    """The REAL stock 9-column GB container (scalar fills)."""
    return make_gb_transform_container(
        use_chirp_mass=True, use_fdot_astro=True, use_distance=True)


def _map(chirp=False):
    return VGBObservableBasis(_container(chirp), Tobs=TOBS)


def _pin(key, leaves=LEAVES):
    return np.array([LEAF_FILLS[i][key] for i in leaves], dtype=float)


# ===========================================================================
# 1. the reduced map itself
# ===========================================================================

class ReducedLayoutTest(unittest.TestCase):
    def test_layout_drops_f_mid_sky_and_mc(self):
        m = _map()
        self.assertEqual(m.INTERNAL_BASIS,
                         ("lnA", "fdot", "phi0", "cos_iota", "psi"))
        self.assertIsNone(m.FIBER_INDEX)
        self.assertIsNone(m.f0_index)
        self.assertIsNone(m.mc_index)
        self.assertEqual(m.dist_index, DIST)
        self.assertEqual(m.ratio_index, R)
        self.assertEqual(m.n_leaves, len(LEAF_FILLS))

    def test_the_gb_layout_is_untouched(self):
        """The base class must keep its documented 9-column layout."""
        b = GBObservableFiberBasis(_gb_container(), Tobs=TOBS)
        self.assertEqual(b.INTERNAL_BASIS, GB_INTERNAL_BASIS)
        self.assertEqual(b.FIBER_INDEX, 8)
        self.assertEqual(GBObservableFiberBasis.INTERNAL_BASIS,
                         GB_INTERNAL_BASIS)

    def test_chirp_mass_basis_grows_the_mc_column_and_fiber_back(self):
        """VGB_CHIRP_MASS_BASIS=1 samples Mc, so (dist, Mc, r) -> (A, fdot)
        is 3->2 again and the fiber returns -- through the SAME class."""
        m = _map(chirp=True)
        self.assertEqual(m.INTERNAL_BASIS,
                         ("lnA", "fdot", "phi0", "cos_iota", "psi", "Mc"))
        self.assertEqual(m.FIBER_INDEX, 5)
        self.assertIsNotNone(m.mc_index)

    def test_a_basis_without_dist_or_ratio_is_refused_loudly(self):
        for drop in ("dist", "fdot_astro_ratio"):
            bad = [c for c in IN_BASIS if c != drop]
            tc = types.SimpleNamespace(input_basis=bad)
            with self.assertRaises(ValueError) as cm:
                VGBObservableBasis(tc, Tobs=TOBS)
            self.assertIn(drop, str(cm.exception))

    def test_a_container_without_the_pinned_fills_is_refused_loudly(self):
        tc = types.SimpleNamespace(input_basis=list(IN_BASIS))
        with self.assertRaises(ValueError) as cm:
            VGBObservableBasis(tc, Tobs=TOBS)
        self.assertIn("fill", str(cm.exception))


class ReducedBijectionTest(unittest.TestCase):
    def test_round_trip_is_exact_across_leaves(self):
        m = _map()
        z = m.to_internal(COORDS, leaf_inds=LEAVES)
        back = np.asarray(m.from_internal(z, leaf_inds=LEAVES))
        np.testing.assert_allclose(back, COORDS, rtol=1e-11, atol=1e-14)

    def test_fdot_column_is_fdot_total_from_the_pinned_constants(self):
        m = _map()
        z = m.to_internal(COORDS, leaf_inds=LEAVES)
        want = (fdot_gr(_pin("f0") * 1e-3, _pin("Mc"))
                * (1.0 + COORDS[:, R]))
        np.testing.assert_allclose(z[:, Z_FDOT], want, rtol=1e-13)

    def test_extrinsic_columns_pass_through_untouched(self):
        m = _map()
        z = m.to_internal(COORDS, leaf_inds=LEAVES)
        for zc, yc in ((Z_PHI0, PHI0), (Z_CI, CI), (Z_PSI, PSI)):
            np.testing.assert_allclose(z[:, zc], COORDS[:, yc], rtol=0,
                                       atol=0)

    def test_identical_coords_on_different_leaves_map_differently(self):
        """The cross-leaf discrimination the pinned constants exist for."""
        m = _map()
        z = m.to_internal(COORDS, leaf_inds=LEAVES)
        for a, b in ((0, 1), (2, 3)):
            self.assertNotAlmostEqual(float(z[a, Z_LNA]),
                                      float(z[b, Z_LNA]), places=6)
            self.assertNotEqual(float(z[a, Z_FDOT]), float(z[b, Z_FDOT]))
        # and each row matches a single-leaf build of its OWN leaf
        z0 = m.to_internal(COORDS, leaf_inds=np.zeros(4, dtype=int))
        z1 = m.to_internal(COORDS, leaf_inds=np.ones(4, dtype=int))
        np.testing.assert_allclose(z[0], z0[0], rtol=0, atol=0)
        np.testing.assert_allclose(z[1], z1[1], rtol=0, atol=0)
        np.testing.assert_allclose(z[2], z0[2], rtol=0, atol=0)
        np.testing.assert_allclose(z[3], z1[3], rtol=0, atol=0)

    def test_missing_leaf_inds_raises_with_the_reason(self):
        m = _map()
        with self.assertRaises(ValueError) as cm:
            m.to_internal(COORDS)
        self.assertIn("leaf_inds", str(cm.exception))

    def test_agrees_with_the_container_on_the_physical_values(self):
        """The map and the transform container must not drift: both derive
        the physical (A, fdot) from the same pinned constants."""
        tc = _container()
        m = VGBObservableBasis(tc, Tobs=TOBS)
        phys = tc.both_transforms(COORDS.copy(), xp=np, leaf_inds=LEAVES)
        ob = list(tc.output_basis)
        z = m.to_internal(COORDS, leaf_inds=LEAVES)
        np.testing.assert_allclose(np.exp(z[:, Z_LNA]),
                                   phys[:, ob.index("A")], rtol=1e-12)
        np.testing.assert_allclose(z[:, Z_FDOT], phys[:, ob.index("fdot")],
                                   rtol=1e-12)


class ReducedMeasureTest(unittest.TestCase):
    def test_log_jacobian_is_the_shared_expression(self):
        m = _map()
        lj = m.log_jacobian(COORDS, leaf_inds=LEAVES)
        want = (np.log(COORDS[:, DIST])
                - np.log(fdot_gr(_pin("f0") * 1e-3, _pin("Mc"))))
        np.testing.assert_allclose(lj, want, rtol=1e-13)

    def test_log_jacobian_matches_the_numeric_determinant(self):
        """|dy/dz| computed by central differences through the REDUCED map."""
        m = _map()
        z = m.to_internal(COORDS, leaf_inds=LEAVES)
        floors = np.array([
            OBSERVABLE_CD_FLOORS.get(nm, OBSERVABLE_CD_FLOOR_DEFAULT)
            for nm in m.INTERNAL_BASIS])
        M = np.zeros((4, NDIM, NDIM))
        for i in range(NDIM):
            h = 1e-6 * np.maximum(np.abs(z[:, i]), floors[i])
            up, dn = z.copy(), z.copy()
            up[:, i] += h
            dn[:, i] -= h
            dx = (np.asarray(m.from_internal(up, template=COORDS,
                                             leaf_inds=LEAVES))
                  - np.asarray(m.from_internal(dn, template=COORDS,
                                               leaf_inds=LEAVES)))
            M[:, :, i] = dx / (2.0 * h)[:, None]
        det = np.abs(np.linalg.det(M))
        np.testing.assert_allclose(
            np.log(det), m.log_jacobian(COORDS, leaf_inds=LEAVES),
            rtol=1e-5)

    def test_factors_are_new_minus_old_and_the_constant_cancels(self):
        m = _map()
        new = COORDS.copy()
        new[:, DIST] *= 1.3
        new[:, R] += 0.01
        f = m.factors(COORDS, new, leaf_inds=LEAVES)
        np.testing.assert_allclose(
            f,
            m.log_jacobian(new, leaf_inds=LEAVES)
            - m.log_jacobian(COORDS, leaf_inds=LEAVES), rtol=0, atol=0)
        # with f0/Mc pinned, fdot_gr is a per-leaf CONSTANT and drops out
        np.testing.assert_allclose(f, np.log(1.3) * np.ones(4), rtol=1e-12)

    def test_non_finite_rows_clamp_to_a_finite_negative(self):
        m = _map()
        new = COORDS.copy()
        new[0, DIST] = -1.0
        f = m.factors(COORDS, new, leaf_inds=LEAVES)
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertLess(f[0], -1e200)


class ReducedStepScaleTest(unittest.TestCase):
    def test_reduced_layout_gets_each_coordinate_its_own_rule(self):
        m = _map()
        s = gb_observable_step_scales(
            np.array([40.0]), TOBS, extrinsic_scales=np.ones(3),
            mc_step=0.0, internal_basis=m.INTERNAL_BASIS)
        self.assertEqual(s.shape, (1, NDIM))
        self.assertAlmostEqual(float(s[0, Z_LNA]), STEP_C_LNA / 40.0,
                               places=12)
        self.assertAlmostEqual(
            float(s[0, Z_FDOT]),
            (STEP_C_FDOT / 40.0) / TOBS / TOBS, places=24)
        np.testing.assert_allclose(s[0, Z_PHI0:], 1.0)

    def test_the_gb_default_layout_is_value_for_value_unchanged(self):
        """The generalization must not move the 9-column array."""
        rho = np.array([7.0, 46.0])
        ex = np.arange(10.0).reshape(2, 5) + 1.0
        got = gb_observable_step_scales(rho, TOBS, extrinsic_scales=ex,
                                        mc_step=0.07, jump=1.3)
        want = np.zeros((2, 9))
        want[:, 0] = 1.0 / rho
        want[:, 1] = (0.5513 / rho) / TOBS
        want[:, 2] = (4.2705 / rho) / TOBS / TOBS
        want[:, 3:8] = ex
        want[:, 8] = 0.07
        np.testing.assert_allclose(got, want * 1.3, rtol=1e-15)

    def test_a_mismatched_extrinsic_width_raises(self):
        m = _map()
        with self.assertRaises(ValueError):
            gb_observable_step_scales(
                np.array([40.0]), TOBS, extrinsic_scales=np.ones(5),
                mc_step=0.0, internal_basis=m.INTERNAL_BASIS)


# ===========================================================================
# 2. the dead fdot_astro_ratio column, and its repair
# ===========================================================================

def _stub(container, **over):
    """A ``self`` carrying the REAL info-matrix / observable chain."""
    from eryn.prior import ProbDistContainer, uniform_dist

    s = types.SimpleNamespace(
        xp=np,
        name="vgb_pe",
        branch_name="vgb",
        use_gpu=False,
        df=DF,
        transform_fn=container,
        parameter_transforms=container,
        _prop_timer=None,
        _basis_settings=None,             # not FDSettings -> wdm comp
        gb_wdm_comp=object(),             # no ``chunked`` -> no slots
        gb_fd_comp=object(),
        _f0_col=None,
        _mc_col=None,
        _dist_col=DIST,
        _fdot_astro_col=R,
        _fdot_col=None,
        _fdot_scale=1e-16,
        _per_leaf_fill=True,
        mempool=None,
        jump_factor=1.0,
        use_info_mat_proposal=True,
        stretch_probability=0.0,
        eigen_axis_generic_ok=True,
        infomat_per_block=True,
        _eigen_axis_min_dim=9,
        _eigen_axis_widths_cache=None,
        _observable_map_cache=None,
        _obs_rho=None,
        _obs_gamma_z=None,
        _obs_eigen_table=None,
        _last_im_kind=None,
        _last_axis_sigmas=None,
        _last_axis_pick=None,
        _proposal_param_scales=np.ones(NDIM),
        gpu_priors={"vgb": ProbDistContainer({
            0: uniform_dist(0.1, 20.0),         # dist kpc
            1: uniform_dist(0.0, 2 * np.pi),    # phi0
            2: uniform_dist(-1.0, 1.0),         # cos_iota
            3: uniform_dist(0.0, np.pi),        # psi
            4: uniform_dist(-5.0, 5.0),         # r
        })},
    )
    s._eigen_axis_ready = lambda: False
    s._infomat_route_check = lambda *a, **k: None
    for name in ("_compute_proposal_cholesky", "_infomat_jacobian",
                 "_infomat_phys_inds", "_eigen_axes_from_info",
                 "_eigen_axis_widths", "_observable_map",
                 "_observable_basis_ready", "_observable_step_scales",
                 "_observable_stash_gamma_z", "_observable_eigen_prepare",
                 "_observable_proposal", "_observable_leaf_kw",
                 "_maybe_observable_proposal", "_observable_rho_snapshot"):
        setattr(s, name, types.MethodType(getattr(GBSpecialBase, name), s))
    # the per-branch hooks the VGB move overrides
    for name in ("_inmodel_kind", "_obs_eigen_mode"):
        setattr(s, name,
                types.MethodType(getattr(VGBSpecialStretchMove, name), s))
    s._observable_map_class = VGBSpecialStretchMove._observable_map_class
    s._observable_required_cols = (
        VGBSpecialStretchMove._observable_required_cols)
    for k, v in over.items():
        setattr(s, k, v)
    return s


def _sorter(coords=COORDS, leaves=LEAVES):
    n = coords.shape[0]
    return types.SimpleNamespace(
        coords=coords.copy(),
        leaf_inds=leaves.copy(),
        walker_inds=np.arange(n),
        inds=np.ones(n, dtype=bool),
        infomat_take_inds=None,              # -> direct per-block branch
    )


def _info_phys(n, ndim=NDIM, seed=5):
    """One SPD physical matrix, IDENTICAL for every source."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((ndim, ndim))
    m = a @ a.T + ndim * np.eye(ndim)
    return np.broadcast_to(m, (n, ndim, ndim)).copy()


class PhysIndsTest(unittest.TestCase):
    """The scored PHYSICAL slots -- the dead-column fix."""

    def test_the_container_itself_leaves_fdot_outside_test_inds(self):
        """GROUND TRUTH, on the real stock container."""
        tc = _container()
        ob = list(tc.output_basis)
        ti = [int(i) for i in tc.fill_dict["test_inds"]]
        self.assertNotIn(ob.index("fdot"), ti,
                         "the physical fdot slot is scored after all -- the "
                         "premise of this fix is gone, re-derive it")
        self.assertIn(ob.index("fddot"), ti)
        # ... and the fddot the transform emits there is EXACTLY zero
        phys = tc.both_transforms(COORDS.copy(), xp=np, leaf_inds=LEAVES)
        np.testing.assert_array_equal(phys[:, ob.index("fddot")],
                                      np.zeros(4))

    def test_phys_inds_substitutes_the_live_fdot_slot(self):
        tc = _container()
        ob = list(tc.output_basis)
        s = _stub(tc)
        got = [int(i) for i in s._infomat_phys_inds()]
        ti = [int(i) for i in tc.fill_dict["test_inds"]]
        self.assertEqual(len(got), len(ti))
        self.assertIn(ob.index("fdot"), got)
        self.assertNotIn(ob.index("fddot"), got)
        # every OTHER slot is untouched, in place
        for k, (a, b) in enumerate(zip(ti, got)):
            if a == ob.index("fddot"):
                self.assertEqual(b, ob.index("fdot"))
            else:
                self.assertEqual(a, b, f"slot {k} moved")

    def test_the_gb_container_is_a_no_op(self):
        """GB samples Mc, so fdot is already scored: byte-identical inds."""
        tc = _gb_container()
        s = _stub(tc, _f0_col=1, _mc_col=2, _dist_col=0, _fdot_astro_col=8,
                  _per_leaf_fill=False)
        np.testing.assert_array_equal(
            s._infomat_phys_inds(),
            np.asarray(tc.fill_dict["test_inds"]))


class DeadRatioColumnTest(unittest.TestCase):
    """The curvature the r column had, and the curvature it has now."""

    def test_on_the_container_test_inds_the_r_column_is_exactly_dead(self):
        """CONFIRMS commit 2bb484e2's KNOWN LIMIT on the real container."""
        tc = _container()
        s = _stub(tc)
        ti = np.asarray(tc.fill_dict["test_inds"])
        j = s._infomat_jacobian(COORDS.copy(), ti, np.ones(NDIM),
                                leaf_inds=LEAVES)
        np.testing.assert_array_equal(
            j[:, :, R], np.zeros((4, NDIM)),
            "the r column is NOT dead on the container test_inds -- this "
            "test documents the defect the fix removes; re-derive it")
        info_y = np.einsum("nai,nab,nbj->nij", j, _info_phys(4), j)
        # rank-deficient: one exactly-zero eigenvalue
        self.assertAlmostEqual(
            float(np.abs(np.linalg.eigvalsh(info_y)).min()), 0.0, places=30)

    def test_on_the_substituted_inds_the_r_column_carries_real_curvature(self):
        tc = _container()
        s = _stub(tc)
        j = s._infomat_jacobian(COORDS.copy(), s._infomat_phys_inds(),
                                np.ones(NDIM), leaf_inds=LEAVES)
        self.assertTrue(np.all(np.abs(j[:, :, R]).max(axis=-1) > 0.0))
        # d(fdot)/dr is analytically fdot_gr(f0, Mc) of THAT leaf's fills
        ob = list(tc.output_basis)
        row = list(int(i) for i in s._infomat_phys_inds()).index(
            ob.index("fdot"))
        want = fdot_gr(_pin("f0") * 1e-3, _pin("Mc"))
        np.testing.assert_allclose(j[:, row, R], want, rtol=1e-6)
        info_y = np.einsum("nai,nab,nbj->nij", j, _info_phys(4), j)
        self.assertGreater(float(np.linalg.eigvalsh(info_y).min()), 0.0)

    def test_the_jacobian_still_discriminates_the_two_leaves(self):
        """Rows with identical coords must differ by their OWN leaf fills.

        Compared RELATIVELY: the physical amplitude is ~1e-22 and fdot
        ~1e-17, so ``np.allclose``'s 1e-8 absolute tolerance calls every
        row identical and would pass this test with the fills ignored.
        """
        tc = _container()
        s = _stub(tc)
        ob = list(tc.output_basis)
        inds = [int(i) for i in s._infomat_phys_inds()]
        j = s._infomat_jacobian(COORDS.copy(), np.asarray(inds),
                                np.ones(NDIM), leaf_inds=LEAVES)
        a_row = inds.index(ob.index("A"))
        fd_row = inds.index(ob.index("fdot"))
        for a, b in ((0, 1), (2, 3)):
            for row, col, what in ((a_row, DIST, "dA/d(dist)"),
                                   (fd_row, R, "d(fdot)/dr")):
                ja, jb = float(j[a, row, col]), float(j[b, row, col])
                self.assertGreater(
                    abs(ja - jb) / max(abs(ja), abs(jb)), 1e-3,
                    f"rows {a}/{b} agree on {what} -- the per-leaf fills "
                    "are not reaching the central differences")


class ReducedGammaZTest(unittest.TestCase):
    """Gamma_z through the REDUCED map: exact pullback, live fdot."""

    def _stash(self, seed=11):
        tc = _container()
        s = _stub(tc)
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((4, NDIM, NDIM))
        info_y = a @ np.swapaxes(a, -1, -2) + 0.5 * np.eye(NDIM)
        s._observable_stash_gamma_z(info_y, COORDS, np.ones(NDIM),
                                    np.arange(4), 4, leaf_inds=LEAVES)
        return s, info_y

    def test_quadratic_form_invariance_across_two_different_leaves(self):
        s, info_y = self._stash()
        gz = s._obs_gamma_z
        self.assertEqual(gz.shape, (4, NDIM, NDIM))
        self.assertTrue(np.all(np.isfinite(gz)))
        m = s._observable_map()
        z = np.asarray(m.to_internal(COORDS, leaf_inds=LEAVES))
        x0 = np.asarray(m.from_internal(z, template=COORDS,
                                        leaf_inds=LEAVES))
        col = np.maximum(np.abs(z).max(axis=0), 1e-3)
        col[Z_FDOT] = max(abs(float(z[:, Z_FDOT].mean())), 1e-22)
        rng = np.random.default_rng(3)
        for _ in range(3):
            dz = 1e-6 * col * rng.standard_normal(NDIM)
            x1 = np.asarray(m.from_internal(z + dz, template=COORDS,
                                            leaf_inds=LEAVES))
            dx = x1 - x0
            qz = np.einsum("i,nij,j->n", dz, gz, dz)
            qx = np.einsum("ni,nij,nj->n", dx, info_y, dx)
            np.testing.assert_allclose(qz, qx, rtol=2e-3)

    def test_rows_with_identical_coords_but_different_leaves_differ(self):
        s, _ = self._stash()
        gz = s._obs_gamma_z
        for a, b in ((0, 1), (2, 3)):
            self.assertFalse(np.allclose(gz[a], gz[b]))

    def test_the_fdot_direction_of_the_final_table_is_nonzero(self):
        """The whole point: a real curvature-set fdot step, not a prior box.

        Drives the FULL chain -- info matrix -> y congruence -> Gamma_z ->
        whitened eigen table -- exactly as a block does.
        """
        tc = _container()
        s = _stub(tc)
        sorter = _sorter()
        ids = np.arange(4)
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "observable",
                         "VGB_INMODEL_OBSERVABLE_EIGEN": "full"}
        ):
            with mock.patch.object(
                _gbbands._RoutedBandEngine, "route_information_matrix",
                side_effect=lambda comp, holder, params_phys, **kw:
                    _info_phys(int(np.shape(params_phys)[0])),
            ):
                chol = s._compute_proposal_cholesky(
                    types.SimpleNamespace(analysis_container_arr=None),
                    sorter, ids)
            self.assertIsNotNone(chol)
            gz = s._obs_gamma_z
            self.assertIsNotNone(
                gz, "Gamma_z was never stashed -- the observable eigen "
                    "path did not run for the reduced basis")
            self.assertGreater(float(np.abs(gz[:, Z_FDOT, Z_FDOT]).min()),
                               0.0)
            s._obs_rho = np.full(4, 30.0)
            s._observable_eigen_prepare(chol, ids, 4)
        tab = s._obs_eigen_table
        self.assertEqual(tab.shape, (4, NDIM, NDIM))
        self.assertTrue(np.all(np.isfinite(tab)))
        # some axis actually moves fdot
        self.assertTrue(np.all(np.abs(tab[:, Z_FDOT, :]).max(axis=-1) > 0.0))

    def test_engine_is_asked_for_the_substituted_slots(self):
        tc = _container()
        ob = list(tc.output_basis)
        s = _stub(tc)
        seen = {}

        def _cap(comp, holder, params_phys, **kw):
            seen["inds"] = np.asarray(kw["inds"])
            return _info_phys(int(np.shape(params_phys)[0]))

        with mock.patch.object(
            _gbbands._RoutedBandEngine, "route_information_matrix",
            side_effect=_cap,
        ):
            s._compute_proposal_cholesky(
                types.SimpleNamespace(analysis_container_arr=None),
                _sorter(), np.arange(4))
        self.assertIn(ob.index("fdot"), [int(i) for i in seen["inds"]])
        self.assertNotIn(ob.index("fddot"), [int(i) for i in seen["inds"]])


# ===========================================================================
# 3. knobs + draw wiring
# ===========================================================================

class VGBKnobTest(unittest.TestCase):
    def test_default_proposal_is_observable(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VGB_INMODEL_PROPOSAL", None)
            self.assertEqual(_vgb_inmodel_proposal_kind(), "observable")

    def test_escapes_remain_reachable(self):
        for raw, want in ((" Eigen ", "eigen"), ("stretch", "stretch"),
                          ("observable", "observable")):
            with mock.patch.dict(
                os.environ, {"VGB_INMODEL_PROPOSAL": raw}
            ):
                self.assertEqual(_vgb_inmodel_proposal_kind(), want, raw)

    def test_junk_warns_and_falls_to_observable(self):
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "obserable"}
        ):
            with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "WARNING"
            ):
                self.assertEqual(_vgb_inmodel_proposal_kind(), "observable")

    def test_eigen_mode_mirrors_the_gb_knob_semantics(self):
        for raw, want in (("", "off"), ("0", "off"), ("off", "off"),
                          ("1", "axis"), ("axis", "axis"),
                          ("full", "full"), ("FULL", "full")):
            with mock.patch.dict(
                os.environ, {"VGB_INMODEL_OBSERVABLE_EIGEN": raw}
            ):
                self.assertEqual(_vgb_observable_eigen_mode(), want, raw)

    def test_eigen_mode_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VGB_INMODEL_OBSERVABLE_EIGEN", None)
            self.assertEqual(_vgb_observable_eigen_mode(), "off")

    def test_eigen_mode_junk_warns_and_stays_off(self):
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_OBSERVABLE_EIGEN": "banana"}
        ):
            with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "WARNING"
            ):
                self.assertEqual(_vgb_observable_eigen_mode(), "off")

    def test_observable_arms_the_info_matrix_machinery(self):
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "observable"}
        ):
            kw = {}
            _vgb_inmodel_defaults(kw)
        self.assertTrue(kw["use_info_mat_proposal"])
        self.assertEqual(kw["stretch_probability"], 0.0)

    def test_the_move_class_wires_the_reduced_map(self):
        self.assertIs(VGBSpecialStretchMove._observable_map_class,
                      VGBObservableBasis)
        self.assertIs(GBSpecialBase._observable_map_class,
                      GBObservableFiberBasis)
        self.assertEqual(VGBSpecialStretchMove._observable_required_cols,
                         ("_dist_col", "_fdot_astro_col"))


class ReadyGateTest(unittest.TestCase):
    def test_ready_on_the_reduced_basis_with_f0_and_mc_pinned(self):
        s = _stub(_container())
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "observable"}
        ):
            self.assertTrue(s._observable_basis_ready())

    def test_not_ready_under_the_escapes(self):
        for kind in ("eigen", "stretch"):
            s = _stub(_container())
            with mock.patch.dict(
                os.environ, {"VGB_INMODEL_PROPOSAL": kind}
            ):
                self.assertFalse(s._observable_basis_ready(), kind)

    def test_gb_still_refuses_the_reduced_basis(self):
        """The GB column list must keep rejecting a basis with no f0/Mc."""
        s = _stub(_container())
        s._observable_required_cols = GBSpecialBase._observable_required_cols
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "observable"}
        ):
            self.assertFalse(s._observable_basis_ready())


class ReducedDrawTest(unittest.TestCase):
    def _armed(self, seed=31):
        """A stub with rho + a fiberless eigen table, as prepare builds it."""
        s = _stub(_container())
        s._obs_rho = np.full(4, 30.0)
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((NDIM, NDIM)))
        colscale = np.array([1e-2, 1e-20, 1e-2, 1e-2, 1e-2])
        sig = 0.1 * (1.0 + np.arange(NDIM))
        table = np.full((4, NDIM, NDIM), np.nan)
        for k in range(4):
            table[k] = (colscale[:, None] * q) * sig[None, :]
        s._obs_eigen_table = table
        return s, table[0]

    def test_axis_mode_steps_along_exactly_one_table_axis(self):
        s, T0 = self._armed()
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_OBSERVABLE_EIGEN": "axis"}
        ):
            np.random.seed(7)
            new, factors = s._observable_proposal(
                COORDS.copy(), None, np.arange(4), leaf_inds=LEAVES)
        m = s._observable_map()
        dz = (np.asarray(m.to_internal(new, leaf_inds=LEAVES))
              - np.asarray(m.to_internal(COORDS, leaf_inds=LEAVES)))
        for i in range(4):
            # solve, NOT lstsq: the fdot row makes cond(T0) ~ 1e18 and
            # lstsq's rcond cutoff would truncate that direction
            comps = np.linalg.solve(T0, dz[i])
            big = np.abs(comps) > 1e-6 * np.abs(comps).max()
            self.assertEqual(int(big.sum()), 1, f"row {i}: {comps}")

    def test_axis_mode_keeps_every_axis_in_the_pick_set(self):
        """A FIBERLESS layout has no pure-fiber last column to drop."""
        s, T0 = self._armed()
        picked = set()
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_OBSERVABLE_EIGEN": "axis"}
        ):
            m = s._observable_map()
            for seed in range(60):
                np.random.seed(seed)
                new, _ = s._observable_proposal(
                    COORDS.copy(), None, np.arange(4), leaf_inds=LEAVES)
                dz = (np.asarray(m.to_internal(new, leaf_inds=LEAVES))
                      - np.asarray(m.to_internal(COORDS,
                                                 leaf_inds=LEAVES)))
                for i in range(4):
                    c = np.linalg.solve(T0, dz[i])
                    picked.add(int(np.argmax(np.abs(c))))
        self.assertEqual(picked, set(range(NDIM)),
                         "some eigen axis is unreachable -- the GB "
                         "fiber-column drop leaked into the fiberless map")

    def test_full_mode_moves_every_observable_coordinate(self):
        s, _ = self._armed()
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_OBSERVABLE_EIGEN": "full"}
        ):
            np.random.seed(5)
            new, _ = s._observable_proposal(
                COORDS.copy(), None, np.arange(4), leaf_inds=LEAVES)
        m = s._observable_map()
        dz = (np.asarray(m.to_internal(new, leaf_inds=LEAVES))
              - np.asarray(m.to_internal(COORDS, leaf_inds=LEAVES)))
        self.assertTrue(np.all((np.abs(dz) > 0).sum(axis=1) == NDIM))

    def test_off_mode_never_touches_the_table(self):
        s = _stub(_container())
        s._obs_rho = np.full(4, 30.0)
        poison = mock.MagicMock()
        poison.__getitem__ = mock.Mock(
            side_effect=AssertionError("table touched with knob off"))
        s._obs_eigen_table = poison
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VGB_INMODEL_OBSERVABLE_EIGEN", None)
            new, _ = s._observable_proposal(
                COORDS.copy(), None, np.arange(4), leaf_inds=LEAVES)
        self.assertEqual(new.shape, COORDS.shape)

    def test_nan_rows_fall_back_to_the_diagonal_draw(self):
        s = _stub(_container())
        s._obs_rho = np.full(4, 30.0)
        s._obs_eigen_table = np.full((4, NDIM, NDIM), np.nan)
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_OBSERVABLE_EIGEN": "axis"}
        ):
            np.random.seed(9)
            new, _ = s._observable_proposal(
                COORDS.copy(), None, np.arange(4), leaf_inds=LEAVES)
        m = s._observable_map()
        dz = (np.asarray(m.to_internal(new, leaf_inds=LEAVES))
              - np.asarray(m.to_internal(COORDS, leaf_inds=LEAVES)))
        self.assertTrue(np.all((np.abs(dz) > 0).sum(axis=1) > 1))

    def test_factors_are_the_maps_log_jacobian_and_layout_clean(self):
        s = _stub(_container())
        s._obs_rho = np.full(4, 30.0)
        np.random.seed(4)
        new, factors = s._observable_proposal(
            COORDS.copy(), None, np.arange(4), leaf_inds=LEAVES)
        m = s._observable_map()
        want = np.asarray(m.factors(COORDS, new, leaf_inds=LEAVES)).ravel()
        np.testing.assert_allclose(np.asarray(factors), want, rtol=1e-12)
        f = np.asarray(factors)
        self.assertEqual(f.dtype, np.float64)
        self.assertEqual(f.ndim, 1)
        self.assertTrue(f.flags["C_CONTIGUOUS"])

    def test_a_zero_spread_ensemble_still_moves(self):
        """VGB_START_FACTOR=0 gives bit-identical walkers; the observable
        step comes from the map + step scales, never the spread."""
        coords = np.tile(COORDS[0], (4, 1))
        s = _stub(_container())
        s._obs_rho = np.full(4, 30.0)
        np.random.seed(2)
        new, factors = s._observable_proposal(
            coords.copy(), None, np.arange(4), leaf_inds=LEAVES)
        dy = np.asarray(new) - coords
        self.assertTrue(np.all(np.abs(dy).max(axis=-1) > 0.0))
        self.assertTrue(np.all(np.isfinite(np.asarray(factors))))

    def test_the_move_routes_the_observable_step_and_labels_it(self):
        s = _stub(_container())
        s._obs_rho = np.full(4, 30.0)
        sorter = _sorter()
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "observable"}
        ):
            np.random.seed(1)
            out = VGBSpecialStretchMove.in_model_proposal(
                s, COORDS.copy(), None, sorter, np.arange(4), None)
        self.assertIsNotNone(out)
        self.assertEqual(s._last_im_kind, "obs_basis")
        new, factors = out
        self.assertEqual(np.asarray(new).shape, COORDS.shape)
        # the pinned columns are untouched: f0/sky are not sampled at all
        self.assertEqual(np.asarray(new).shape[1], NDIM)

    def test_the_leaf_kw_is_derived_from_the_sorter(self):
        s = _stub(_container())
        sorter = _sorter()
        kw = s._observable_leaf_kw(sorter, np.array([2, 3]))
        np.testing.assert_array_equal(kw["leaf_inds"], LEAVES[[2, 3]])
        # GB (scalar fills) passes NO kwarg at all
        s2 = _stub(_gb_container(), _per_leaf_fill=False)
        self.assertEqual(s2._observable_leaf_kw(sorter, np.arange(2)), {})


if __name__ == "__main__":
    unittest.main()
