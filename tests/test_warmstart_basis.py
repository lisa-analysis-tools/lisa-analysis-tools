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


class _DeviceArray:
    """A stand-in for ``cupy.ndarray`` that refuses host conversion.

    cupy raises ``TypeError: Implicit conversion to a NumPy array is not
    allowed`` from ``__array__``; this reproduces that exactly so the
    device path is testable on a CPU-only machine.
    """

    def __init__(self, a):
        self._a = np.asarray(a, dtype=float)

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not "
                        "allowed. Please use `.get()` to construct a NumPy "
                        "array explicitly.")

    def get(self):
        return self._a

    def __sub__(self, other):
        rhs = other.get() if isinstance(other, _DeviceArray) else other
        return _DeviceArray(self._a - rhs)


class _FakeXP:
    """The array-module face ``log_density_to_sampling`` uses."""
    float64 = float

    @staticmethod
    def asarray(a, dtype=None):
        if isinstance(a, _DeviceArray):
            return _DeviceArray(a.get())
        return _DeviceArray(a)


def _fake_module_for(a):
    if isinstance(a, _DeviceArray):
        return _FakeXP
    return np


class _StubMap:
    def __init__(self, lj, device=False):
        self._lj = np.asarray(lj, dtype=float)
        self._device = device
        self.seen = None

    def log_jacobian(self, coords, leaf_inds=None):
        self.seen = leaf_inds
        return _DeviceArray(self._lj) if self._device else self._lj


class LogDensityTransportTest(unittest.TestCase):
    """``log_density_to_sampling``: the sign, and the array module.

    Both halves have bitten. The SIGN was caught in review: ``log_jacobian``
    is ``ln|dy/dz|`` and a density carried to sampling coordinates
    SUBTRACTS it, so getting it backwards leaves ``rvs`` and ``logpdf``
    inconsistent by ``2 * log_jacobian`` -- a quantity that varies across a
    component, so it biases every RJ birth/death factor instead of
    cancelling. The ARRAY MODULE was caught in production: the helper
    forced both terms through ``np.asarray``, which raises the moment
    either side is on device -- and on the real path both are, because
    ``BandSorter`` scores ``rj_prop.logpdf`` on device-resident band
    coordinates and ``log_jacobian`` is itself ``xp``-generic.
    """

    def test_the_jacobian_is_SUBTRACTED(self):
        log_q_z = np.array([-1.0, 2.0, 0.5])
        lj = np.array([0.25, -1.5, 3.0])
        x = np.zeros((3, 9))
        got = wb.log_density_to_sampling(log_q_z, x, _StubMap(lj))
        np.testing.assert_allclose(got, log_q_z - lj, rtol=0, atol=0)

    def test_leaf_inds_reach_the_map(self):
        leaf = np.arange(3)
        m = _StubMap(np.zeros(3))
        wb.log_density_to_sampling(np.zeros(3), np.zeros((3, 9)), m,
                                   leaf_inds=leaf)
        np.testing.assert_array_equal(m.seen, leaf)

    def test_device_inputs_do_not_go_through_numpy(self):
        """The production traceback, reproduced and pinned.

        Before the fix this raised ``TypeError: Implicit conversion to a
        NumPy array is not allowed`` out of ``np.asarray(log_q_z)`` and
        took the whole MPI job down with ``MPI_Abort``.
        """
        import unittest.mock as mock

        log_q_z = _DeviceArray([-1.0, 2.0, 0.5])
        lj = np.array([0.25, -1.5, 3.0])
        x = _DeviceArray(np.zeros((3, 9)))
        m = _StubMap(lj, device=True)
        with mock.patch("lisatools.utils.utility.get_array_module",
                        _fake_module_for):
            got = wb.log_density_to_sampling(log_q_z, x, m)
        self.assertIsInstance(
            got, _DeviceArray,
            "the result left the device -- the helper is not dispatching")
        np.testing.assert_allclose(got.get(),
                                   np.array([-1.0, 2.0, 0.5]) - lj)

    def test_a_host_only_call_still_returns_numpy(self):
        got = wb.log_density_to_sampling(
            np.zeros(2), np.zeros((2, 9)), _StubMap(np.ones(2)))
        self.assertIsInstance(got, np.ndarray)


if __name__ == "__main__":
    unittest.main()
