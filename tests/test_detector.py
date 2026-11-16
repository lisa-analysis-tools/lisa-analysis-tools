"""Tests for :mod:`lisatools.detector` orbit and geometry queries."""

import numpy as np
import time

import unittest
from copy import deepcopy

try:
    import cupy as cp
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from lisatools.utils.constants import *
from lisatools import detector as lisa

import sys

class DetectorTest(unittest.TestCase):
    """Sanity checks for :class:`lisatools.detector.DefaultOrbits`."""

    def test_orbits(self):
        """Verify normal vectors, light travel times, and positions are finite over a year."""
        xp = cp if gpu_available else np
        force_backend = "cpu" if not gpu_available else "cuda12x"
        orbits = lisa.DefaultOrbits(force_backend=force_backend)
        # configuration is lazy (first use); no explicit configure() needed
        dt = 100.0
        _t = xp.arange(0.0, YRSID_SI, dt)
        t_arr_links = xp.tile(_t, (len(orbits.LINKS), 1)).flatten()
        links = xp.repeat(xp.asarray(orbits.LINKS), len(_t))
        normal_vec = orbits.get_normal_unit_vec(t_arr_links, links)
        ltt = orbits.get_light_travel_times(t_arr_links, links)
        import lisatools
        _backend = lisatools.get_backend(force_backend)
        t_arr_sc = xp.tile(_t, (3, 1)).flatten()
        sc = xp.repeat(xp.array([1, 2, 3]), len(_t))
        pos = orbits.get_pos(t_arr_sc, sc)
        vel = orbits.get_vel(t_arr_sc, sc)

        self.assertFalse(xp.any(xp.isnan(normal_vec)))
        self.assertFalse(xp.any(xp.isnan(ltt)))
        self.assertFalse(xp.any(xp.isnan(pos)))
        self.assertFalse(xp.any(xp.isnan(vel)))

    def test_get_vel(self):
        """get_vel interpolates the stored velocity grid the way get_pos does for positions."""
        xp = cp if gpu_available else np
        force_backend = "cpu" if not gpu_available else "cuda12x"
        orbits = lisa.DefaultOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)

        sc_t = xp.asarray(orbits.sc_t)
        v_grid = xp.asarray(orbits.v)  # (N, 3_sc, 3_xyz)
        i = len(sc_t) // 2

        for sc in [1, 2, 3]:
            # node exactness: interpolation at a grid node returns the node value
            t_node = float(sc_t[i])
            vel_node = orbits.get_vel(t_node, sc)
            self.assertEqual(vel_node.shape, (3,))
            xp.testing.assert_allclose(vel_node, v_grid[i, sc - 1, :], rtol=1e-12, atol=0.0)

            # linearity: a midpoint between two nodes == average of the node velocities
            t_mid = 0.5 * (float(sc_t[i]) + float(sc_t[i + 1]))
            vel_mid = orbits.get_vel(t_mid, sc)
            expected_mid = 0.5 * (v_grid[i, sc - 1, :] + v_grid[i + 1, sc - 1, :])
            xp.testing.assert_allclose(vel_mid, expected_mid, rtol=1e-9, atol=0.0)

        # input-mode shapes mirror get_pos
        t_arr = xp.asarray([float(sc_t[i]), float(sc_t[i + 1]), float(sc_t[i + 2])])
        self.assertEqual(orbits.get_vel(t_arr, 2).shape, (3, 3))  # (ndarray, int)
        sc_arr = xp.asarray([1, 2, 3], dtype=np.int32)
        out_arr = orbits.get_vel(t_arr, sc_arr)  # (ndarray, ndarray)
        self.assertEqual(out_arr.shape, (3, 3))
        self.assertFalse(xp.any(xp.isnan(out_arr)))

        # out-of-range spacecraft ids raise instead of silently returning zeros
        with self.assertRaises(ValueError):
            orbits.get_vel(t_arr, xp.asarray([1, 2, 9], dtype=np.int32))


class OrbitsDeepCopyTest(unittest.TestCase):
    """A deep copy of a configured orbit shares its read-only arrays and reads the same values.

    Settings trees and each computation's private copy are deep copied. A Mojito L1 table is 1.26 GiB
    on the device, so a copy that duplicated it would fill the card.
    """

    SHARED_TABLES = ("x", "v", "n", "sc_t")
    # The default 50 s grid takes several GiB per configure; sharing does not depend on the grid.
    GRID_STEP_S = 1000.0

    def check_each_orbit(self, check, configured=True):
        """Run ``check(orbits)`` for every orbit class with packaged tables, on every backend."""
        import lisatools

        # Named backends: has_backend does not know the "cuda" and "gpu" aliases.
        backends = [name for name in ("cpu", "cuda13x", "cuda12x", "cuda11x") if lisatools.has_backend(name)]
        for backend in backends:
            for orbit_class in (lisa.EqualArmlengthOrbits, lisa.ESAOrbits):
                with self.subTest(backend=backend, orbit=orbit_class.__name__):
                    orbits = orbit_class(force_backend=backend, linear_interp_setup=False, dt=self.GRID_STEP_S)
                    if configured:
                        orbits._ensure_configured()
                    check(orbits)

    def test_configured_copy_shares_its_tables(self):
        def check(orbits):
            duplicate = deepcopy(orbits)
            self.assertIsNot(duplicate, orbits)
            self.assertTrue(duplicate.configured)
            for name in self.SHARED_TABLES:
                self.assertIs(getattr(duplicate, name), getattr(orbits, name), msg=name)
            # The C++ orbit reads these arrays, and the light travel times are its largest one.
            for position, argument in enumerate(orbits.pycppdetector_args):
                if hasattr(argument, "shape"):
                    self.assertIs(duplicate.pycppdetector_args[position], argument, msg=str(position))

        self.check_each_orbit(check)

    def test_unconfigured_copy_owns_its_tables(self):
        def check(orbits):
            duplicate = deepcopy(orbits)
            self.assertIsNot(duplicate, orbits)
            self.assertFalse(duplicate.configured)
            self.assertIsNot(duplicate.x_base, orbits.x_base)
            duplicate._ensure_configured()
            self.assertTrue(duplicate.configured)
            self.assertFalse(orbits.configured)

        self.check_each_orbit(check, configured=False)

    def test_one_orbit_reached_twice_stays_one_copy(self):
        def check(orbits):
            settings_tree = deepcopy({"first": orbits, "nested": {"second": orbits}})
            self.assertIs(settings_tree["first"], settings_tree["nested"]["second"])
            self.assertIsNot(settings_tree["first"], orbits)

        self.check_each_orbit(check)

    def test_copy_reads_the_same_values(self):
        def check(orbits):
            duplicate = deepcopy(orbits)
            xp = orbits.xp
            times = xp.linspace(1e4, 1e6, 50)
            for link in orbits.LINKS:
                self.assertTrue(xp.array_equal(duplicate.get_light_travel_times(times, int(link)),
                                               orbits.get_light_travel_times(times, int(link))))
                self.assertTrue(xp.array_equal(duplicate.get_normal_unit_vec(times, int(link)),
                                               orbits.get_normal_unit_vec(times, int(link))))
            for spacecraft in (1, 2, 3):
                self.assertTrue(xp.array_equal(duplicate.get_pos(times, spacecraft),
                                               orbits.get_pos(times, spacecraft)))

        self.check_each_orbit(check)

    def test_new_configure_rebinds_the_copy_only(self):
        def check(orbits):
            duplicate = deepcopy(orbits)
            source_table = orbits.x
            duplicate._configure(**duplicate._configure_kwargs)
            self.assertIs(orbits.x, source_table)
            self.assertIsNot(duplicate.x, source_table)

        self.check_each_orbit(check)
