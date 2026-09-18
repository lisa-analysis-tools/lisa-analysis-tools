"""WarmStartComponents: both formats, and the Jacobian on the draw path."""
import json
import os
import tempfile
import unittest

import numpy as np

from lisatools.globalfit.warmstart import basis as wb
from lisatools.globalfit.warmstart.proposal import WarmStartComponents

SAMPLING_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                  "sin_delta", "fdot_astro_ratio"]


class _Container:
    def __init__(self):
        self.input_basis = list(SAMPLING_BASIS)


class LegacyFormatTest(unittest.TestCase):
    def test_a_file_without_a_basis_key_is_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.npz")
            means = np.array([[1.5, 3.0, 0.45, 1.0, 0.2, 0.5, 2.0, -0.3, 0.01]])
            covs = np.eye(9)[None] * 1e-4
            np.savez(path, means=means, covs=covs, p=np.array([1.0]),
                     mult=np.array([1.0]), n_members=np.array([50]),
                     island_id=np.array([0]),
                     f0_window_edges=np.array([[2.9, 3.1]]),
                     meta=json.dumps({"tobs": 7.776e6,
                                      "column_names": SAMPLING_BASIS}))
            c = WarmStartComponents.from_npz(path, new_tobs=1.5552e7)
            self.assertEqual(c.basis, "sampling")
            self.assertIsNone(c.obs_map)
            x = c.rvs(16)
            self.assertEqual(x.shape, (16, 9))

    def test_legacy_draws_are_byte_identical_under_a_fixed_seed(self):
        """The shipped refereed npz must keep drawing exactly as today."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.npz")
            rng = np.random.default_rng(0)
            means = np.column_stack([
                rng.uniform(1.0, 9.0, 8), rng.uniform(2.9, 3.1, 8),
                rng.uniform(0.3, 0.6, 8), rng.uniform(0, 6.2, 8),
                rng.uniform(-0.9, 0.9, 8), rng.uniform(0, 3.1, 8),
                rng.uniform(0, 6.2, 8), rng.uniform(-0.9, 0.9, 8),
                rng.uniform(-0.5, 0.5, 8)])
            covs = np.tile(np.eye(9) * 1e-4, (8, 1, 1))
            np.savez(path, means=means, covs=covs,
                     p=np.linspace(0.2, 1.0, 8), mult=np.ones(8),
                     n_members=np.full(8, 50), island_id=np.arange(8),
                     f0_window_edges=np.tile([[2.9, 3.1]], (8, 1)),
                     meta=json.dumps({"tobs": 7.776e6,
                                      "column_names": SAMPLING_BASIS}))
            a = WarmStartComponents.from_npz(path, new_tobs=1.5552e7,
                                             seed=1234).rvs(64)
            b = WarmStartComponents.from_npz(path, new_tobs=1.5552e7,
                                             seed=1234).rvs(64)
            np.testing.assert_array_equal(a, b)
            lp = WarmStartComponents.from_npz(
                path, new_tobs=1.5552e7, seed=1234).logpdf(a)
            self.assertTrue(np.all(np.isfinite(lp)))


def _obs_npz(tmp, mean_z, cov_z, name="obs.npz", p=1.0, tobs=7.776e6):
    """One-cluster, one-component observable file in the packed layout."""
    from lisatools.sampling.fstat_proposal import pack_gmm_components

    path = os.path.join(tmp, name)
    comps = [[np.array([1.0])], [mean_z[None]], [cov_z[None]],
             [np.linalg.inv(cov_z)[None]],
             [np.array([np.linalg.det(cov_z)])],
             [mean_z - 10.0], [mean_z + 10.0]]
    packed = pack_gmm_components(comps)
    m = wb.build_map(_Container(), Tobs=tobs)
    np.savez(path, p=np.array([p]), mult=np.array([1.0]),
             n_members=np.array([90]), island_id=np.array([0]),
             f0_window_edges=np.array([[2.9, 3.1]]),
             meta=json.dumps({
                 "tobs": tobs, "basis": "observable", "f0_units": "Hz",
                 "column_names": wb.OBSERVABLE_COLUMN_NAMES,
                 "map_params": wb.map_params_from_map(m)}),
             **packed)
    return path, m


X0 = np.array([[9.69, 20.380376, 0.4678, 6.17, -0.9, 1.40,
                4.085, -0.7795, -0.0018]])


def _flagship_component(tmp, **kw):
    m = wb.build_map(_Container(), Tobs=7.776e6)
    z0 = np.asarray(m.to_internal(X0))[0]
    cov = np.diag(np.maximum(np.abs(z0) * 1e-5, 1e-30) ** 2)
    path, _ = _obs_npz(tmp, z0, cov, **kw)
    c = WarmStartComponents.from_npz(path, new_tobs=1.5552e7)
    c.attach_transform(_Container())
    return c, m, z0, cov


class ObservableFormatTest(unittest.TestCase):
    def test_observable_file_loads_with_its_map_and_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _, _, _ = _flagship_component(tmp)
            self.assertEqual(c.basis, "observable")
            self.assertIsNotNone(c.obs_map)
            np.testing.assert_array_equal(c.gmm_ncomp, np.array([1]))
            self.assertEqual(c.n_components, 1)

    def test_a_missing_packed_key_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _obs_npz(tmp, np.zeros(9), np.eye(9))
            d = {k: v for k, v in np.load(path).items()
                 if k != "gmm_covs"}
            bad = os.path.join(tmp, "bad.npz")
            np.savez(bad, **d)
            with self.assertRaises(ValueError):
                WarmStartComponents.from_npz(bad, new_tobs=1.5552e7)

    def test_observable_column_names_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = _obs_npz(tmp, np.zeros(9), np.eye(9))
            d = dict(np.load(path).items())
            meta = json.loads(str(d["meta"]))
            meta["column_names"] = SAMPLING_BASIS      # wrong for observable
            d["meta"] = json.dumps(meta)
            bad = os.path.join(tmp, "bad.npz")
            np.savez(bad, **d)
            with self.assertRaises(ValueError):
                WarmStartComponents.from_npz(bad, new_tobs=1.5552e7)


class ObservableDrawTest(unittest.TestCase):
    def test_rvs_returns_sampling_columns_near_the_mapped_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _, _, _ = _flagship_component(tmp)
            x = c.rvs(512)
            self.assertEqual(x.shape, (512, 9))
            # f0 (sampling col 1) lands on the source, not scattered
            self.assertLess(abs(np.median(x[:, 1]) - 20.380376), 1e-4)

    def test_logpdf_carries_the_log_jacobian_with_the_right_sign(self):
        """A density transforms with the determinant of the INVERSE map.

        ``log_jacobian`` is ``ln|dy/dz|`` at a sampling point y, and
        ``q_y(y) = q_z(z(y)) * |dz/dy|``, so the correction SUBTRACTS.
        Getting it backwards leaves rvs and logpdf inconsistent by
        ``2 * log_jacobian``, which varies across a component and so biases
        the RJ birth/death factors instead of cancelling.
        """
        with tempfile.TemporaryDirectory() as tmp:
            c, m, z0, cov = _flagship_component(tmp)
            lp = c.logpdf(X0)
            z = np.asarray(m.to_internal(X0))
            d = (z - z0).ravel()
            gauss = (-0.5 * float(d @ np.linalg.inv(cov) @ d)
                     - 0.5 * np.log(np.linalg.det(2 * np.pi * cov)))
            jac = float(np.asarray(m.log_jacobian(X0))[0])
            self.assertAlmostEqual(float(lp[0]), gauss - jac, places=6)
            # and the wrong sign is far away, so this is a real test
            self.assertGreater(abs(2 * jac), 10.0)

    def test_sampling_density_integrates_to_one(self):
        """Spec test 2, by importance sampling with an INDEPENDENT proposal.

        Draw z from a deliberately broadened Gaussian h, map to y, and
        estimate ``int q_y dy = E_h[q_y(y(z)) |dy/dz| / h(z)]``. The
        estimator is 1 only if the Jacobian correction is applied with the
        right sign and magnitude.
        """
        with tempfile.TemporaryDirectory() as tmp:
            c, m, z0, cov = _flagship_component(tmp)
            rng = np.random.default_rng(11)
            sd = np.sqrt(np.diag(cov)) * 1.3
            z = z0 + sd * rng.standard_normal((50000, 9))
            y = np.asarray(m.from_internal(z))
            log_h = (-0.5 * np.sum(((z - z0) / sd) ** 2, axis=1)
                     - np.sum(np.log(sd)) - 4.5 * np.log(2 * np.pi))
            log_w = (c.logpdf(y) + np.asarray(m.log_jacobian(y)) - log_h)
            integral = float(np.mean(np.exp(log_w)))
            # Monte Carlo, so a tolerance rather than an equality -- but a
            # flipped Jacobian sign lands this near e^64, not near 1.02.
            self.assertAlmostEqual(integral, 1.0, delta=0.02)

    def test_draws_are_self_consistent_between_rvs_and_logpdf(self):
        """logpdf must be finite at every point rvs produces."""
        with tempfile.TemporaryDirectory() as tmp:
            c, _, _, _ = _flagship_component(tmp)
            x = c.rvs(64)
            self.assertTrue(np.all(np.isfinite(c.logpdf(x))))

    def test_draws_follow_the_fitted_law_in_the_observable_basis(self):
        """rvs maps back correctly: the draws, mapped FORWARD again, must
        be chi-square(9) in Mahalanobis distance about the fitted mean.

        This checks the draw path end to end -- Cholesky draw, circular
        wrap, ``from_internal`` -- independently of logpdf, which the
        normalisation test above covers.
        """
        from scipy.stats import chi2

        with tempfile.TemporaryDirectory() as tmp:
            c, m, z0, cov = _flagship_component(tmp)
            x = c.rvs(20000)
            z = np.asarray(m.to_internal(x))
            d = z - z0
            maha = np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)
            self.assertAlmostEqual(float(np.mean(maha)), 9.0, delta=0.4)
            for q in (0.25, 0.5, 0.75):
                frac = float(np.mean(maha < chi2.ppf(q, 9)))
                self.assertAlmostEqual(frac, q, delta=0.02)


class MergeScopeTest(unittest.TestCase):
    def test_merge_candidate_pairs_never_cross_within_a_cluster(self):
        from lisatools.globalfit.warmstart.referee_apply import (
            merge_candidate_pairs)
        # three clusters: 2, 1 and 3 components -> flat indices
        ncomp = np.array([2, 1, 3])
        pairs = merge_candidate_pairs(ncomp, island_id=np.array([0, 0, 0]))
        # (0,1) and (3,4),(3,5),(4,5) are WITHIN clusters and must be absent
        for bad in [(0, 1), (3, 4), (3, 5), (4, 5)]:
            self.assertNotIn(bad, pairs)
        # across-cluster pairs in the same island survive
        self.assertIn((0, 2), pairs)
        self.assertIn((2, 3), pairs)

    def test_single_component_clusters_behave_as_before(self):
        from lisatools.globalfit.warmstart.referee_apply import (
            merge_candidate_pairs)
        ncomp = np.array([1, 1, 1])
        pairs = merge_candidate_pairs(ncomp, island_id=np.array([0, 0, 1]))
        self.assertIn((0, 1), pairs)      # same island
        self.assertNotIn((0, 2), pairs)   # different island

    def test_filtering_an_existing_pair_list_is_the_same_rule(self):
        """Production filters the referee's OWN pairs -- O(pairs), not the
        O(n^2) enumeration, which at 5573 components is 15M iterations."""
        from lisatools.globalfit.warmstart.referee_apply import (
            merge_candidate_pairs)
        ncomp = np.array([2, 1, 3])
        isl = np.array([0, 0, 0])
        proposed = [(0, 1), (0, 2), (3, 4), (2, 3), (1, 5)]
        kept = merge_candidate_pairs(ncomp, isl, pairs=proposed)
        self.assertEqual(kept, [(0, 2), (2, 3), (1, 5)])


class RefereeApplyObservableTest(unittest.TestCase):
    """Mixture siblings must survive the auto-merge that legacy pairs get."""

    def _fit_and_referee(self, tmp, ncomp, cross):
        from lisatools.sampling.fstat_proposal import pack_gmm_components

        n = int(np.sum(ncomp))
        m = wb.build_map(_Container(), Tobs=7.776e6)
        z0 = np.asarray(m.to_internal(X0))[0]
        means = np.repeat(z0[None], n, axis=0)
        means[:, 1] += np.arange(n) * 1e-9        # distinct f_mid
        cov = np.diag(np.maximum(np.abs(z0) * 1e-5, 1e-30) ** 2)
        splits = np.cumsum(ncomp)[:-1]
        comps = [
            [np.full(k, 1.0 / k) for k in ncomp],
            np.split(means, splits, axis=0),
            np.split(np.repeat(cov[None], n, axis=0), splits, axis=0),
            np.split(np.repeat(np.linalg.inv(cov)[None], n, axis=0), splits,
                     axis=0),
            np.split(np.full(n, np.linalg.det(cov)), splits),
            [z0 - 10.0 for _ in ncomp], [z0 + 10.0 for _ in ncomp],
        ]
        fit = os.path.join(tmp, "fit.npz")
        np.savez(fit, p=np.full(len(ncomp), 0.9),
                 mult=np.ones(len(ncomp)),
                 n_members=np.full(len(ncomp), 90),
                 island_id=np.zeros(len(ncomp), dtype=int),
                 f0_window_edges=np.array([[2.0e-2, 2.1e-2]]),
                 meta=json.dumps({
                     "tobs": 7.776e6, "basis": "observable", "f0_units": "Hz",
                     "column_names": wb.OBSERVABLE_COLUMN_NAMES,
                     "map_params": wb.map_params_from_map(m)}),
                 **pack_gmm_components(comps))
        ref = os.path.join(tmp, "ref.npz")
        pairs = np.array([[i, j] for i in range(n) for j in range(i + 1, n)])
        np.savez(ref, pairs=pairs,
                 cross_match=np.full(len(pairs), cross),
                 med_ratio=np.full(n, 0.9), med_match=np.full(n, 0.9))
        return fit, ref

    def test_mixture_siblings_are_not_merged_back_together(self):
        from lisatools.globalfit.warmstart.referee_apply import apply

        with tempfile.TemporaryDirectory() as tmp:
            # ONE cluster of 3 components, all cross-matching at 0.99
            fit, ref = self._fit_and_referee(tmp, np.array([3]), 0.99)
            out = os.path.join(tmp, "refereed.npz")
            apply(fit, ref, out)
            with np.load(out, allow_pickle=False) as d:
                self.assertEqual(int(np.sum(d["gmm_ncomp"])), 3)
                self.assertEqual(len(d["gmm_means"]), 3)

    def test_across_cluster_pairs_still_merge(self):
        """Three split artifacts become ONE cluster -- but they stay three
        mixture components of it rather than being moment-matched into one
        Gaussian, which would put the merged mass between the lobes."""
        from lisatools.globalfit.warmstart.referee_apply import apply

        with tempfile.TemporaryDirectory() as tmp:
            # THREE single-component clusters, all cross-matching at 0.99
            fit, ref = self._fit_and_referee(tmp, np.array([1, 1, 1]), 0.99)
            out = os.path.join(tmp, "refereed.npz")
            apply(fit, ref, out)
            with np.load(out, allow_pickle=False) as d:
                self.assertEqual(len(d["p"]), 1)             # one cluster
                np.testing.assert_array_equal(d["gmm_ncomp"], [3])
                self.assertEqual(len(d["gmm_means"]), 3)
                self.assertAlmostEqual(float(d["gmm_weights"].sum()), 1.0)
                self.assertAlmostEqual(float(d["p"][0]), 1.0)  # min(1, 2.7)

    def test_a_merged_observable_set_still_loads_as_a_proposal(self):
        from lisatools.globalfit.warmstart.referee_apply import apply

        with tempfile.TemporaryDirectory() as tmp:
            fit, ref = self._fit_and_referee(tmp, np.array([2, 2]), 0.99)
            out = os.path.join(tmp, "refereed.npz")
            apply(fit, ref, out)
            c = WarmStartComponents.from_npz(out, new_tobs=1.5552e7)
            c.attach_transform(_Container())
            x = c.rvs(64)
            self.assertEqual(x.shape, (64, 9))
            self.assertTrue(np.all(np.isfinite(c.logpdf(x))))


if __name__ == "__main__":
    unittest.main()
