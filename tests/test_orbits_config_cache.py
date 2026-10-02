"""The TDI-on-the-fly configured-orbits cache must never hand one backend's (or one
device's) configured orbit tables to another.

Found 2026-10-02 on the cluster: ``TDDenseGPUParityTest`` builds the same
``EqualArmlengthOrbits`` on "cpu" then "gpu" in one process; the cache key ignored the
backend, so the GPU response received the CPU-configured (numpy) tables and
``OrbitsWrapGPU`` raised ``incompatible function arguments``. Multi-GPU has the same
shape: a table configured on device 0 served to device 1."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


class _FakeCudaXp:
    """An ``xp`` with a cupy-like current device."""

    def __init__(self, dev):
        self.cuda = SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: dev))


def _stub(backend_name, xp=np):
    return SimpleNamespace(frame="icrs", armlength=2.5e9, t0=0.0, filename="orbits.h5",
                           _configure_kwargs={"dt": 1e4},
                           backend=SimpleNamespace(backend_name=backend_name, xp=xp))


class OrbitsCacheKeyTest(unittest.TestCase):
    def test_backend_is_part_of_the_key(self):
        from lisatools.response.tdionfly import _orbits_cache_key

        self.assertNotEqual(_orbits_cache_key(_stub("lisatools_cpu")),
                            _orbits_cache_key(_stub("lisatools_cuda13x")))

    def test_device_is_part_of_the_key(self):
        from lisatools.response.tdionfly import _orbits_cache_key

        a = _orbits_cache_key(_stub("lisatools_cuda12x", _FakeCudaXp(0)))
        b = _orbits_cache_key(_stub("lisatools_cuda12x", _FakeCudaXp(1)))
        self.assertNotEqual(a, b)

    def test_identical_orbits_still_share_one_entry(self):
        from lisatools.response.tdionfly import _orbits_cache_key

        self.assertEqual(_orbits_cache_key(_stub("lisatools_cpu")),
                         _orbits_cache_key(_stub("lisatools_cpu")))

    def test_real_cpu_orbits_are_reused(self):
        """The perf cache still works: two identical CPU orbits -> one configured object."""
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response import tdionfly

        with mock.patch.dict(tdionfly._ORBITS_CONFIGURED_CACHE, clear=True):
            a = tdionfly._get_configured_orbits(EqualArmlengthOrbits(force_backend="cpu"))
            b = tdionfly._get_configured_orbits(EqualArmlengthOrbits(force_backend="cpu"))
            self.assertIs(a, b)
            self.assertTrue(a.configured)

    def test_a_cached_entry_of_another_backend_is_not_served(self):
        """Pre-populate the cache as if a GPU run configured these orbits first: a CPU
        request must configure its own copy (numpy tables), not take the GPU one."""
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response import tdionfly

        orb = EqualArmlengthOrbits(force_backend="cpu")
        poisoned = SimpleNamespace(configured=True, tag="other-backend")
        with mock.patch.dict(tdionfly._ORBITS_CONFIGURED_CACHE, clear=True):
            key = list(tdionfly._orbits_cache_key(orb))
            key[key.index(orb.backend.backend_name)] = "lisatools_cuda13x"
            tdionfly._ORBITS_CONFIGURED_CACHE[tuple(key)] = poisoned
            got = tdionfly._get_configured_orbits(orb)
        self.assertIsNot(got, poisoned)
        self.assertIsInstance(got.pycppdetector_args[7], np.ndarray)


if __name__ == "__main__":
    unittest.main()
