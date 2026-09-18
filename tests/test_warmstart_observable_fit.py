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


if __name__ == "__main__":
    unittest.main()
