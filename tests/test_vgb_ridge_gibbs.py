"""VGB = GB parity: the ridge-Gibbs fiber move on the VGB branch.

User ruling 2026-09-16: "The VGBs should now (with stretch removed) have the
exact same mechanics as the GBs except f0 and sky fixed" -- Mc AND
fdot_astro_ratio BOTH sampled -- plus "really mirror the GBs as much as
possible ... VGBs get the ridge-gibbs fiber move too."

Three construction-level gates, no data and no waveform machinery:

(a) the chirp-mass VGB basis is 6 columns, the per-leaf fills are exactly
    ``f0 / alpha / sin_delta``, and the sampled columns carry BOTH ``Mc``
    and ``fdot_astro_ratio`` -- i.e. the (Mc, r, dist) fiber the ridge move
    resamples actually exists on the branch;
(b) ridge eligibility is decided BY COLUMN NAME, never by branch name: the
    6-column chirp basis registers the move, the 5-column legacy distance
    basis (Mc a per-leaf fill) does not -- which is what keeps every other
    script's 5-dim VGB store behaving exactly as before -- and
    :class:`McRatioDistFiber` resolves dist/Mc/r by name on the VGB layout
    (slots 0/4/5, NOT the GB 9-column slots 0/2/8);
(c) the store-vs-config ndim guard fires with a message naming
    ``VGB_CHIRP_MASS_BASIS`` and the fresh-store / migration requirement --
    the vgb chain goes 5 -> 6 columns, so this is NOT resume-compatible.
"""

import os
import unittest

from lisatools.globalfit.recipe import ridge_gibbs_eligible
from lisatools.globalfit.run import check_store_branch_ndims
from lisatools.globalfit.stock.erebor.vgb import (
    VGB_FIXED_BASIS_CHIRP,
    VGB_SAMPLED_BASIS_CHIRP,
    VGB_SAMPLED_BASIS_DIST,
    VGBSettings,
    vgb_fixed_basis,
    vgb_sampled_basis,
)
from lisatools.sampling.ridge_fiber import McRatioDistFiber

# The production 9-column GB sampling basis, for the parity assertions.
GB_BASIS_9 = [
    "dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha", "sin_delta",
    "fdot_astro_ratio",
]

MC_LIMS = (0.001, 1.0)
DIST_LIMS = (0.01, 100.0)
RATIO_MAX = 5.0


class _Basis:
    """Minimal duck-type of a TransformContainer (only ``input_basis``)."""

    def __init__(self, input_basis):
        self.input_basis = list(input_basis)


class _Info:
    """Minimal duck-type of a branch ``source_info`` entry."""

    def __init__(self, input_basis, ratio_max=RATIO_MAX):
        self.transform = _Basis(input_basis)
        self.fdot_astro_ratio_max = ratio_max


class _EnvGuard:
    """Set/clear env vars for the duration of a block (None = unset)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.saved = {}

    def __enter__(self):
        for key, value in self.kwargs.items():
            self.saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


# ----------------------------------------------------------------------
# (a) the chirp basis actually carries the fiber
# ----------------------------------------------------------------------
class VGBChirpBasisCarriesFiberTest(unittest.TestCase):
    def test_chirp_settings_are_six_dim_with_three_fills(self):
        with _EnvGuard(VGB_CHIRP_MASS_BASIS="1", VGB_SAMPLE_DISTANCE=None):
            settings = VGBSettings()
        self.assertTrue(settings.chirp_mass_basis)
        self.assertEqual(settings.ndim, 6)
        self.assertEqual(vgb_sampled_basis(settings), VGB_SAMPLED_BASIS_CHIRP)
        # f0 and sky ONLY -- the ruling's "except f0 and sky fixed"
        self.assertEqual(
            vgb_fixed_basis(settings), ["f0", "alpha", "sin_delta"]
        )
        self.assertEqual(VGB_FIXED_BASIS_CHIRP, ["f0", "alpha", "sin_delta"])

    def test_sampled_columns_include_mc_and_ratio(self):
        for name in ("Mc", "fdot_astro_ratio", "dist"):
            self.assertIn(name, VGB_SAMPLED_BASIS_CHIRP, name)
        # the 5-column legacy basis pins Mc -> no fiber on that branch
        self.assertNotIn("Mc", VGB_SAMPLED_BASIS_DIST)

    def test_every_fiber_column_the_gb_branch_has_is_present(self):
        """Parity: the VGB chirp basis is the GB basis minus f0 + sky."""
        self.assertEqual(
            set(GB_BASIS_9) - set(VGB_SAMPLED_BASIS_CHIRP),
            {"f0", "alpha", "sin_delta"},
        )


# ----------------------------------------------------------------------
# (b) eligibility is by COLUMN NAME, and the fiber resolves the VGB layout
# ----------------------------------------------------------------------
class RidgeGibbsEligibilityTest(unittest.TestCase):
    def test_vgb_chirp_basis_is_eligible(self):
        with _EnvGuard(GB_RIDGE_GIBBS=None):
            self.assertTrue(ridge_gibbs_eligible(_Info(VGB_SAMPLED_BASIS_CHIRP)))

    def test_vgb_five_column_basis_is_not_eligible(self):
        """Back-compat: every script still on the 5-column basis is untouched."""
        with _EnvGuard(GB_RIDGE_GIBBS=None):
            self.assertFalse(ridge_gibbs_eligible(_Info(VGB_SAMPLED_BASIS_DIST)))

    def test_gb_nine_column_basis_still_eligible(self):
        with _EnvGuard(GB_RIDGE_GIBBS=None):
            self.assertTrue(ridge_gibbs_eligible(_Info(GB_BASIS_9)))

    def test_gb_eight_column_basis_not_eligible(self):
        eight = [c for c in GB_BASIS_9 if c not in ("Mc", "dist")]
        with _EnvGuard(GB_RIDGE_GIBBS=None):
            self.assertFalse(ridge_gibbs_eligible(_Info(eight)))

    def test_env_knob_off_disables_both_branches(self):
        with _EnvGuard(GB_RIDGE_GIBBS="0"):
            self.assertFalse(ridge_gibbs_eligible(_Info(VGB_SAMPLED_BASIS_CHIRP)))
            self.assertFalse(ridge_gibbs_eligible(_Info(GB_BASIS_9)))

    def test_missing_ratio_max_disables(self):
        with _EnvGuard(GB_RIDGE_GIBBS=None):
            self.assertFalse(
                ridge_gibbs_eligible(_Info(VGB_SAMPLED_BASIS_CHIRP, ratio_max=None))
            )

    def test_missing_transform_disables(self):
        info = _Info(VGB_SAMPLED_BASIS_CHIRP)
        info.transform = None
        with _EnvGuard(GB_RIDGE_GIBBS=None):
            self.assertFalse(ridge_gibbs_eligible(info))


class VGBFiberColumnResolutionTest(unittest.TestCase):
    """McRatioDistFiber works VERBATIM on the VGB layout (by-name lookup)."""

    def test_resolves_vgb_chirp_columns_by_name(self):
        fiber = McRatioDistFiber(
            _Basis(VGB_SAMPLED_BASIS_CHIRP), MC_LIMS, DIST_LIMS, RATIO_MAX
        )
        self.assertEqual(fiber.dist_index, 0)
        self.assertEqual(fiber.mc_index, 4)
        self.assertEqual(fiber.ratio_index, 5)
        # explicitly NOT the GB 9-column slots: proof the lookup is by name
        gb_fiber = McRatioDistFiber(
            _Basis(GB_BASIS_9), MC_LIMS, DIST_LIMS, RATIO_MAX
        )
        self.assertEqual(
            (gb_fiber.dist_index, gb_fiber.mc_index, gb_fiber.ratio_index),
            (0, 2, 8),
        )

    def test_rejects_the_five_column_basis_loudly(self):
        with self.assertRaises(ValueError) as ctx:
            McRatioDistFiber(
                _Basis(VGB_SAMPLED_BASIS_DIST), MC_LIMS, DIST_LIMS, RATIO_MAX
            )
        self.assertIn("Mc", str(ctx.exception))

    def test_fiber_is_exactly_likelihood_invariant_on_the_vgb_layout(self):
        """u -> u' carries (Mc, r, d) so Mc^{5/3}(1+r) and Mc^{5/3}/d hold."""
        import numpy as np

        fiber = McRatioDistFiber(
            _Basis(VGB_SAMPLED_BASIS_CHIRP), MC_LIMS, DIST_LIMS, RATIO_MAX
        )
        rows = np.zeros((3, len(VGB_SAMPLED_BASIS_CHIRP)))
        rows[:, 0] = [1.0, 5.0, 20.0]        # dist (kpc)
        rows[:, 1] = [0.3, 1.1, 2.0]         # phi0  (untouched)
        rows[:, 2] = [-0.5, 0.0, 0.7]        # cos_iota (untouched)
        rows[:, 3] = [0.1, 0.9, 1.4]         # psi   (untouched)
        rows[:, 4] = [0.2, 0.35, 0.5]        # Mc
        rows[:, 5] = [0.0, 0.1, -0.2]        # fdot_astro_ratio

        u, inv = fiber.to_fiber(rows)
        new = fiber.from_fiber(1.7 * u, inv, rows)
        u2, inv2 = fiber.to_fiber(new)

        np.testing.assert_allclose(inv2["Kf"], inv["Kf"], rtol=1e-12)
        np.testing.assert_allclose(inv2["KA"], inv["KA"], rtol=1e-12)
        # untouched columns carry through bit-for-bit
        for col in (1, 2, 3):
            np.testing.assert_array_equal(new[:, col], rows[:, col])


# ----------------------------------------------------------------------
# (c) the resume ndim guard
# ----------------------------------------------------------------------
class StoreNdimGuardTest(unittest.TestCase):
    PATH = "/tmp/does_not_matter.h5"

    def test_matching_ndims_pass(self):
        check_store_branch_ndims(
            {"vgb": 6, "gb": 9}, {"vgb": 6, "gb": 9}, self.PATH
        )

    def test_branches_absent_from_the_store_are_ignored(self):
        """REMOVE_BRANCHES / added branches must not trip the guard."""
        check_store_branch_ndims({"gb": 9}, {"gb": 9, "vgb": 6}, self.PATH)
        check_store_branch_ndims({"gb": 9, "vgb": 5}, {"gb": 9}, self.PATH)

    def test_vgb_mismatch_names_the_knob_and_the_fresh_store_requirement(self):
        with self.assertRaises(ValueError) as ctx:
            check_store_branch_ndims({"vgb": 5}, {"vgb": 6}, self.PATH)
        msg = str(ctx.exception)
        self.assertIn("VGB_CHIRP_MASS_BASIS", msg)
        self.assertIn("vgb", msg)
        self.assertIn("5", msg)
        self.assertIn("6", msg)
        # the operator's two actual options, both named
        self.assertIn("fresh", msg.lower())
        self.assertIn("migrate_vgb_chirp_basis.py", msg)
        self.assertIn(self.PATH, msg)

    def test_generic_branch_mismatch_still_fires(self):
        with self.assertRaises(ValueError) as ctx:
            check_store_branch_ndims({"gb": 8}, {"gb": 9}, self.PATH)
        msg = str(ctx.exception)
        self.assertIn("gb", msg)
        self.assertIn("GB_USE_CHIRP_MASS", msg)

    def test_none_stored_ndims_is_a_no_op(self):
        check_store_branch_ndims(None, {"vgb": 6}, self.PATH)
        check_store_branch_ndims({}, {"vgb": 6}, self.PATH)


# ----------------------------------------------------------------------
# wiring: the move the recipe registers is named + branch-scoped correctly
# ----------------------------------------------------------------------
class VGBRidgeMoveWiringTest(unittest.TestCase):
    def test_make_move_binds_the_vgb_branch(self):
        from eryn.prior import ProbDistContainer, uniform_dist

        from lisatools.sampling.ridge_fiber import make_gb_ridge_gibbs_move

        priors = ProbDistContainer({
            0: uniform_dist(*DIST_LIMS),
            1: uniform_dist(0.0, 2 * 3.141592653589793),
            2: uniform_dist(-1.0, 1.0),
            3: uniform_dist(0.0, 3.141592653589793),
            4: uniform_dist(*MC_LIMS),
            5: uniform_dist(-RATIO_MAX, RATIO_MAX),
        })
        move = make_gb_ridge_gibbs_move(
            priors,
            _Basis(VGB_SAMPLED_BASIS_CHIRP),
            mc_lims=MC_LIMS,
            dist_lims=DIST_LIMS,
            ratio_max=RATIO_MAX,
            branch_name="vgb",
        )
        self.assertEqual(move.branch_name, "vgb")
        self.assertEqual(move.fiber_map.mc_index, 4)


if __name__ == "__main__":
    unittest.main()
