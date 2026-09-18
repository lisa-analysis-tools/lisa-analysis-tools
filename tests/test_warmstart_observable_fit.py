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


class ClusterGMMTest(unittest.TestCase):
    def _two_mode_cluster(self, n=400, seed=11):
        """One cluster whose fdot marginal is genuinely bimodal."""
        rng = np.random.default_rng(seed)
        z = np.zeros((n, 9))
        z[:, 0] = rng.normal(1.0, 0.05, n)
        z[:, 1] = rng.normal(3.0, 1e-6, n)
        half = n // 2
        z[:half, 2] = rng.normal(1.0e-16, 2e-18, half)
        z[half:, 2] = rng.normal(9.0e-16, 2e-18, n - half)
        for c, sc in ((3, 0.05), (4, 0.05), (5, 0.05), (6, 0.05), (7, 0.05)):
            z[:, c] = rng.normal(0.3, sc, n)
        z[:, 8] = rng.normal(0.45, 0.01, n)
        return z

    def test_bimodal_cluster_gets_more_than_one_component(self):
        comps = ffs.fit_cluster_gmms([self._two_mode_cluster()],
                                     n_samples=2048, max_comp=6, seed=5)
        weights = comps[0]
        self.assertGreaterEqual(len(weights[0]), 2,
                                "BIC should prefer >1 component for a "
                                "genuinely bimodal cluster")

    def test_component_count_is_set_by_the_min_members_cap(self):
        """MEASURED 2026-09-18: the existing fitter's BIC is monotone in K.

        ``vec_fit_gmm_min_bic`` scores BIC on the model's OWN synthetic
        draws (``gmm.bic(gmm.rvs(n))``), which is an entropy estimate: more
        components always fit tighter, so BIC falls monotonically and the
        "risen twice past the running minimum" retirement never fires. On a
        clean unimodal 9-D Gaussian, measured BIC ran 30556 (K=1) down to
        22255 (K=7), with AND without resampling.

        So the sweep returns ``max_comp_effective`` every time, and
        ``min_members`` -- the spec's guard against BIC believing resampled
        evidence -- is the ACTUAL selector. This test pins that, because it
        is what anyone tuning ``--gmm-max-comp`` needs to know.

        The underlying EM also runs with ``random_state=None``, so the
        selected K is not reproducible run to run; the assertions below are
        the guarantees that DO hold.
        """
        rng = np.random.default_rng(2)
        z = np.zeros((400, 9))
        for c in range(9):
            z[:, c] = rng.normal(1.0, 0.05, 400)
        # cap = min(6, 400 // 25 = 16) = 6
        k6 = len(ffs.fit_cluster_gmms([z], n_samples=2048, max_comp=6,
                                      seed=5)[0][0])
        self.assertLessEqual(k6, 6)
        self.assertGreater(k6, 1,
                           "BIC-on-own-draws does not select 1 even for a "
                           "clean unimodal cloud -- the cap is the selector")
        # min_members binds when it is the tighter of the two: 400 // 200
        k_guard = len(ffs.fit_cluster_gmms([z], n_samples=2048, max_comp=12,
                                           min_members=200, seed=5)[0][0])
        self.assertLessEqual(k_guard, 2)

    def test_small_cluster_is_capped_at_one_component(self):
        """min_members guards against BIC believing resampled evidence."""
        z = self._two_mode_cluster(n=20)
        comps = ffs.fit_cluster_gmms([z], n_samples=2048, max_comp=6,
                                     min_members=25, seed=5)
        self.assertEqual(len(comps[0][0]), 1)

    def test_weights_sum_to_one_per_cluster(self):
        comps = ffs.fit_cluster_gmms(
            [self._two_mode_cluster(), self._two_mode_cluster(seed=99)],
            n_samples=1024, max_comp=4, seed=5)
        for w in comps[0]:
            self.assertAlmostEqual(float(np.sum(w)), 1.0, places=6)

    def test_output_is_the_seven_ragged_lists(self):
        comps = ffs.fit_cluster_gmms([self._two_mode_cluster()],
                                     n_samples=1024, max_comp=4, seed=5)
        self.assertEqual(len(comps), 7)
        weights, means, covs, invcovs, dets, mins, maxs = comps
        k = len(weights[0])
        self.assertEqual(np.shape(means[0]), (k, 9))
        self.assertEqual(np.shape(covs[0]), (k, 9, 9))

    def test_components_are_in_PHYSICAL_not_unit_cube_coordinates(self):
        """The fitter returns means on [-1, 1]; we must store physical ones.

        ``GMMFit`` maps each group onto the unit cube before fitting, so
        ``vec_fit_gmm_min_bic``'s means/covs are cube coordinates and its
        mins/maxs are the affine map back. ``WarmStartComponents`` draws
        ``mean + L z`` directly, so the packed arrays have to be unscaled
        here or every draw lands in the wrong place entirely.
        """
        z = self._two_mode_cluster()
        comps = ffs.fit_cluster_gmms([z], n_samples=2048, max_comp=4, seed=5)
        weights, means, covs, invcovs, dets, mins, maxs = comps
        w = np.asarray(weights[0])
        mu = np.asarray(means[0])
        # the mixture mean of each column must sit inside the data range
        mix_mean = w @ mu
        for c in range(9):
            self.assertGreaterEqual(mix_mean[c], z[:, c].min() - 1e-9)
            self.assertLessEqual(mix_mean[c], z[:, c].max() + 1e-9)
        # f_mid is ~3.0 physically and would be ~0 on the unit cube
        self.assertAlmostEqual(mix_mean[1], float(z[:, 1].mean()), places=4)

        # The decisive scale check: the mixture's own per-column variance
        # must reproduce the members'. On the unit cube every column's
        # variance is O(0.1); physically they run from ~1e-32 (fdot) to
        # ~1e-3 (lnA), so a cube-coordinate slip fails this by decades.
        cv = np.asarray(covs[0])
        for c in range(9):
            mix_var = float(w @ (cv[:, c, c] + (mu[:, c] - mix_mean[c]) ** 2))
            self.assertAlmostEqual(
                np.log10(mix_var), np.log10(float(z[:, c].var())), places=1,
                msg=f"column {c} variance is off by orders of magnitude")

    def test_stored_dets_match_the_stored_covariances(self):
        """invcovs/dets ride along for the deferred FullGaussianMixtureModel
        swap, so they must stay consistent with the unscaled covariances.

        Checked via slogdet: the physical covariance spans ~18 orders of
        magnitude between the lnA and fdot columns, so a plain
        ``C @ invC == I`` comparison is meaningless in floating point even
        though the identity holds exactly in exact arithmetic.
        """
        z = self._two_mode_cluster()
        comps = ffs.fit_cluster_gmms([z], n_samples=2048, max_comp=4, seed=5)
        covs, dets = np.asarray(comps[2][0]), np.asarray(comps[4][0])
        for k in range(covs.shape[0]):
            sign, logabsdet = np.linalg.slogdet(covs[k])
            self.assertEqual(int(sign), 1)
            self.assertAlmostEqual(logabsdet, float(np.log(dets[k])),
                                   places=6)

    def test_bounds_are_the_physical_member_range(self):
        z = self._two_mode_cluster()
        comps = ffs.fit_cluster_gmms([z], n_samples=2048, max_comp=4, seed=5)
        mins, maxs = np.asarray(comps[5][0]), np.asarray(comps[6][0])
        self.assertEqual(mins.shape, (9,))
        for c in range(9):
            self.assertGreaterEqual(mins[c], z[:, c].min() - 1e-9)
            self.assertLessEqual(maxs[c], z[:, c].max() + 1e-9)


if __name__ == "__main__":
    unittest.main()
