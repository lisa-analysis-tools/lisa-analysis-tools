"""Intake converts the leaf table to the observable basis."""
import unittest

import numpy as np

from lisatools.globalfit.warmstart import basis as wb
from lisatools.globalfit.warmstart import fit_from_store as ffs

SAMPLING_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                  "sin_delta", "fdot_astro_ratio"]


class _Container:
    def __init__(self):
        self.input_basis = list(SAMPLING_BASIS)


def _rows(n=64, seed=3):
    rng = np.random.default_rng(seed)
    x = np.empty((n, 9))
    x[:, 0] = rng.uniform(0.5, 30.0, n)        # dist
    x[:, 1] = rng.uniform(3.0, 4.0, n)         # f0 mHz
    x[:, 2] = rng.uniform(0.2, 0.8, n)         # Mc
    x[:, 3] = rng.uniform(0, 2 * np.pi, n)
    x[:, 4] = rng.uniform(-1, 1, n)
    x[:, 5] = rng.uniform(0, np.pi, n)
    x[:, 6] = rng.uniform(0, 2 * np.pi, n)
    x[:, 7] = rng.uniform(-1, 1, n)
    x[:, 8] = rng.uniform(-0.5, 0.5, n)        # ratio
    return x


class IntakeConversionTest(unittest.TestCase):
    def test_to_observable_matches_the_map(self):
        x = _rows()
        m = wb.build_map(_Container(), Tobs=7.776e6)
        got = ffs.to_observable(x, m)
        np.testing.assert_allclose(got, m.to_internal(x), rtol=0, atol=0)
        self.assertEqual(got.shape, x.shape)

    def test_observable_columns_are_the_expected_physics(self):
        x = _rows(4)
        m = wb.build_map(_Container(), Tobs=7.776e6)
        z = ffs.to_observable(x, m)
        # fdot is column 2 and must be positive for positive ratio > -1
        self.assertTrue(np.all(z[:, 2] > 0))
        # the five extrinsic columns pass straight through
        np.testing.assert_allclose(z[:, 3:8], x[:, 3:8], rtol=0, atol=0)

    def test_circular_and_cos_iota_constants_are_basis_independent(self):
        self.assertEqual(ffs.COS_IOTA_COL, 4)
        self.assertEqual(sorted(ffs.CIRCULAR_COLS), [3, 5, 6])
        self.assertEqual(
            [wb.OBSERVABLE_COLUMN_NAMES[i] for i in (3, 5, 6)],
            ["phi0", "psi", "alpha"])

    def test_observable_bounded_cols_drop_the_ratio_rail(self):
        # the +/- ratio_max rail is GONE in the observable basis: fdot is
        # unbounded there. cos_iota keeps its physical bound.
        self.assertEqual(ffs.OBSERVABLE_BOUNDED_COLS, {4: (-1.0, 1.0)})


class ClusterFeatureTest(unittest.TestCase):
    def test_observable_features_are_the_measured_ones(self):
        self.assertEqual(ffs.OBSERVABLE_FEAT_NAMES,
                         ["f_mid", "fdot", "lnA", "alpha", "sin_delta"])

    def test_observable_features_read_the_right_columns(self):
        z = np.zeros((3, 9))
        z[:, 0] = [1.0, 2.0, 3.0]        # lnA
        z[:, 1] = [3.0, 3.1, 3.2]        # f_mid
        z[:, 2] = [1e-16, 2e-16, 3e-16]  # fdot
        z[:, 6] = [0.1, 0.2, 0.3]        # alpha
        z[:, 7] = [-0.5, 0.0, 0.5]       # sin_delta
        f = ffs.make_cluster_features(z, basis="observable")
        self.assertEqual(f.shape, (3, 5))
        np.testing.assert_allclose(f[:, 0], z[:, 1])   # f_mid
        np.testing.assert_allclose(f[:, 1], z[:, 2])   # fdot
        np.testing.assert_allclose(f[:, 2], z[:, 0])   # lnA, already logged
        np.testing.assert_allclose(f[:, 4], z[:, 7])   # sin_delta

    def test_fdot_separates_what_f0_alone_cannot(self):
        """Two sources at one frequency, differing only in fdot."""
        z = np.zeros((40, 9))
        z[:, 1] = 3.0                      # identical f_mid
        z[:20, 2] = 1e-16
        z[20:, 2] = 9e-16                  # 9x apart in fdot
        f_obs = ffs.make_cluster_features(z, basis="observable")
        self.assertGreater(np.ptp(f_obs[:, 1]), 0.0)
        # the SAMPLING metric has no fdot column at all, which is the defect
        self.assertNotIn("fdot", ffs.FEAT_NAMES)

    def test_sampling_basis_features_are_unchanged(self):
        x = _rows(8)
        np.testing.assert_allclose(
            ffs.make_cluster_features(x, basis="sampling"),
            ffs.make_cluster_features(x))   # default must stay back-compatible


class WhiteningScaleTest(unittest.TestCase):
    """The whitening floor must not flatten the axis Task 3 exists to add.

    The sampling-basis floor vector is ``[0.05*df, 1e-4, 1e-3, 1e-3, 1e-3]``
    where entry 1 belongs to ``Mc`` (order 0.5). In the observable basis
    entry 1 is ``fdot``, which runs ~1e-17 at 1 mHz to ~1e-12 at 30 mHz --
    reusing 1e-4 there sets the scale 1e12x above the data and whitens every
    fdot difference to zero, silently defeating the new metric.
    """

    def _island(self, fdot_lo, fdot_hi, n=40):
        z = np.zeros((n, 9))
        z[:, 1] = 3.0e-3                          # one f_mid [Hz]
        z[:n // 2, 2] = fdot_lo
        z[n // 2:, 2] = fdot_hi
        z[:, 0] = -50.0                           # lnA
        return z

    def test_floor_is_relative_to_the_island_chirp_scale(self):
        z = self._island(1.0e-16, 9.0e-16)
        feats = ffs.make_cluster_features(z, basis="observable")
        floor = ffs.cluster_scale_floor(feats, 1.2861e-7, "observable")
        self.assertLess(floor[1], 1.0e-17,
                        "the fdot floor must sit far below the island's own "
                        "chirp scale, not 1e12x above it")
        self.assertAlmostEqual(floor[0], 0.05 * 1.2861e-7)

    def test_sampling_floor_is_the_historical_vector(self):
        x = _rows(8)
        feats = ffs.make_cluster_features(x, basis="sampling")
        np.testing.assert_allclose(
            ffs.cluster_scale_floor(feats, 1.2861e-4, "sampling"),
            np.array([0.05 * 1.2861e-4, 1e-4, 1e-3, 1e-3, 1e-3]))

    def _two_chirp_island(self, seed=4):
        """One island, one f_mid, two chirps.

        Deliberately UNEQUAL populations (100 vs 20). Two equal-size modes
        on a single MAD-whitened axis always land ~1.35 whitened units
        apart whatever their separation -- the MAD is set by the split
        itself -- so single linkage at T_CUT 2.0 could never cut them in
        EITHER basis. An unequal pair is the real satellite-fragment
        geometry: the MAD tracks the dominant mode and the minority sits
        many whitened units out.
        """
        rng = np.random.default_rng(seed)
        n_big, n_small = 100, 20
        n = n_big + n_small
        z = np.zeros((n, 9))
        z[:, 1] = 3.0e-3 + 1e-12 * rng.standard_normal(n)   # one f_mid
        z[:n_big, 2] = rng.normal(1.0e-16, 2e-18, n_big)
        z[n_big:, 2] = rng.normal(1.4e-16, 2e-18, n_small)  # +4e-17 chirp
        z[:, 0] = rng.normal(-50.0, 1e-4, n)                # lnA
        for c in (3, 4, 5, 6, 7, 8):
            z[:, c] = rng.normal(0.3, 1e-4, n)
        return z, rng

    def test_fdot_axis_is_live_under_the_relative_floor(self):
        """Paired control: the historical fixed floor vs the new one.

        Same island, same separation -- only the floor differs. This is
        the defect and the fix in one measurement.
        """
        z, _ = self._two_chirp_island()
        feats = ffs.make_cluster_features(z, basis="observable")
        fdot = feats[:, 1]
        mad = 1.4826 * np.median(np.abs(fdot - np.median(fdot)))
        sep = 4.0e-17
        historical = sep / max(mad, 1e-4)     # entry 1 was Mc's floor
        relative = sep / max(
            mad, ffs.cluster_scale_floor(feats, 1.2861e-7, "observable")[1])
        self.assertLess(historical, 1e-6,
                        "the historical 1e-4 floor crushes the chirp axis "
                        "to nothing -- the new feature would be inert")
        self.assertGreater(relative, ffs.T_CUT,
                           "the relative floor must leave the chirp axis "
                           "separating at more than the linkage cut")

    def test_unequal_chirp_groups_split_into_two_clusters(self):
        z, rng = self._two_chirp_island()
        stats = dict(satellite_merges=0, basis="observable")
        labels = ffs.split_single_linkage(z, rng, 1.2861e-7, stats)
        self.assertEqual(
            len(np.unique(labels[labels >= 0])), 2,
            "two chirps at one frequency must separate in the observable "
            "metric -- f0 alone cannot tell them apart")


if __name__ == "__main__":
    unittest.main()
