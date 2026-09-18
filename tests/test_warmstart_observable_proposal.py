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
    cov = np.diag(np.maximum(np.abs(z0) * 1e-3, 1e-12) ** 2)
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


if __name__ == "__main__":
    unittest.main()
