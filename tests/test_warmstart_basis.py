"""warmstart.basis: one place to build and round-trip the observable map."""
import unittest

import numpy as np

from lisatools.globalfit.warmstart import basis as wb
from lisatools.sampling.gb_observable_basis import GB_INTERNAL_BASIS

SAMPLING_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                  "sin_delta", "fdot_astro_ratio"]


class _Container:
    """Minimal stand-in for a TransformContainer: only input_basis is read."""
    def __init__(self, basis=None):
        self.input_basis = list(basis if basis is not None else SAMPLING_BASIS)


class BasisAdapterTest(unittest.TestCase):
    def test_observable_column_names_match_the_map(self):
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES, list(GB_INTERNAL_BASIS))
        # the four that change meaning, pinned so a reorder is caught here
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[0], "lnA")
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[1], "f_mid")
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[2], "fdot")
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[8], "Mc")
        # and the five that do NOT
        self.assertEqual(wb.OBSERVABLE_COLUMN_NAMES[3:8],
                         ["phi0", "cos_iota", "psi", "alpha", "sin_delta"])

    def test_params_round_trip_rebuilds_an_equivalent_map(self):
        c = _Container()
        m = wb.build_map(c, Tobs=7.776e6, shear=0.5, fiber_coord="Mc")
        params = wb.map_params_from_map(m)
        self.assertEqual(params["Tobs"], 7.776e6)
        self.assertEqual(params["shear"], 0.5)
        self.assertEqual(params["fiber_coord"], "Mc")
        self.assertEqual(params["input_basis"], SAMPLING_BASIS)
        m2 = wb.build_map_from_params(c, params)
        x = np.array([[1.5, 3.0, 0.45, 1.0, 0.2, 0.5, 2.0, -0.3, 0.01]])
        np.testing.assert_allclose(m.to_internal(x), m2.to_internal(x),
                                   rtol=0, atol=0)

    def test_map_round_trip_is_exact(self):
        c = _Container()
        m = wb.build_map(c, Tobs=7.776e6)
        x = np.array([
            [1.5, 3.0, 0.45, 1.0, 0.2, 0.5, 2.0, -0.3, 0.01],
            [9.7, 20.380376, 0.4678, 6.17, -0.9, 1.40, 4.085, -0.7795, -0.0018],
        ])
        back = m.from_internal(m.to_internal(x))
        np.testing.assert_allclose(back, x, rtol=1e-12, atol=1e-12)

    def test_params_reject_a_basis_mismatch(self):
        c = _Container()
        m = wb.build_map(c, Tobs=7.776e6)
        params = wb.map_params_from_map(m)
        params["input_basis"] = SAMPLING_BASIS[:-1] + ["something_else"]
        with self.assertRaises(ValueError):
            wb.build_map_from_params(c, params)


if __name__ == "__main__":
    unittest.main()
