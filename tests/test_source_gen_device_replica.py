"""Per-device source-generator replicas (multi-GPU MBH parallelism).

Structural CPU coverage for the device-keyed replica mechanism that makes the
MBH source move generate each walker's template on the walker's OWNING device
(no peer access off the primary device):

- ``jax_device_context`` is a safe no-op on CPU / when JAX can't see the device
  (so single-GPU/CPU behaviour is unchanged and importing never hard-fails).
- ``source_runtime._device_local_orbits`` reuses the shared orbits on the run's
  primary device / CPU (zero extra memory, byte-identical to the pre-multi-GPU
  path) and builds ONE cached ``orbits.__class__(*args, **kwargs)`` replica per
  non-primary device (the DCGA build pattern), so the ``id(orbits)``-keyed
  phentax generator cache yields one generator per device.

The actual on-GPU device placement (phentax JAX tables, cupy response, orbit
grids) can only be validated on the cluster; these tests pin the routing.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

try:
    from tests._multishard import RecordingXp
except ImportError:
    from _multishard import RecordingXp


class JaxDeviceContextTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.utils.device import jax_device_context
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"device helpers not available: {exc}")
        self.jax_device_context = jax_device_context

    def test_none_is_nullcontext(self):
        # CPU / single-device / primary device -> plain no-op, no jax needed.
        with self.jax_device_context(None):
            pass

    def test_missing_device_is_nullcontext(self):
        # A device index JAX cannot possibly have must degrade to a no-op
        # (never raise), so a GPU run with fewer JAX devices than cupy sees,
        # or JAX-on-CPU, keeps working.
        with self.jax_device_context(9999):
            pass


class _FakeOrbits:
    """Minimal orbits stand-in: reconstructable via
    ``__class__(*args, **kwargs)`` (mirrors ``EqualArmlengthOrbits``)."""

    def __init__(self, tag="base"):
        self.tag = tag

    @property
    def args(self):
        return ()

    @property
    def kwargs(self):
        return {"tag": self.tag}


class DeviceLocalOrbitsTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.globalfit.stock.erebor import source_runtime as sr
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"source_runtime not available: {exc}")
        self.sr = sr
        sr._DEVICE_ORBITS_REPLICAS.clear()
        self.addCleanup(sr._DEVICE_ORBITS_REPLICAS.clear)
        self.xp = RecordingXp()
        self.orbits = _FakeOrbits(tag="shared")

    def test_primary_device_reuses_shared(self):
        # current device (RecordingXp default 0) == primary 0 -> shared orbits,
        # no replica built.
        rep = self.sr._device_local_orbits(
            self.orbits, self.xp, primary_device=0)
        self.assertIs(rep, self.orbits)
        self.assertEqual(len(self.sr._DEVICE_ORBITS_REPLICAS), 0)

    def test_cpu_none_reuses_shared(self):
        # numpy xp -> current_device is None -> shared orbits, no replica.
        rep = self.sr._device_local_orbits(
            self.orbits, np, primary_device=None)
        self.assertIs(rep, self.orbits)
        self.assertEqual(len(self.sr._DEVICE_ORBITS_REPLICAS), 0)

    def test_nonprimary_builds_and_caches_replica(self):
        with self.xp.cuda.Device(1):  # current device 1 != primary 0
            rep1 = self.sr._device_local_orbits(
                self.orbits, self.xp, primary_device=0)
            rep2 = self.sr._device_local_orbits(
                self.orbits, self.xp, primary_device=0)
        self.assertIsNot(rep1, self.orbits)              # distinct replica
        self.assertIsInstance(rep1, _FakeOrbits)
        self.assertEqual(rep1.kwargs["tag"], "shared")   # rebuilt from kwargs
        self.assertIs(rep1, rep2)                        # built once, cached
        self.assertEqual(len(self.sr._DEVICE_ORBITS_REPLICAS), 1)
        # replica construction was routed through the device-1 context
        self.assertIn(1, self.xp.device_log)

    def test_distinct_replica_per_device(self):
        with self.xp.cuda.Device(1):
            rep1 = self.sr._device_local_orbits(
                self.orbits, self.xp, primary_device=0)
        with self.xp.cuda.Device(2):
            rep2 = self.sr._device_local_orbits(
                self.orbits, self.xp, primary_device=0)
        self.assertIsNot(rep1, rep2)
        self.assertEqual(len(self.sr._DEVICE_ORBITS_REPLICAS), 2)


class _FakeDomainSettings:
    """WDMSettings-like stand-in: ``__class__(*args, **kwargs)`` rebuildable,
    with a device-resident ``window`` in kwargs that the replica drops so it
    would be regenerated on the target device."""

    def __init__(self, Nf=8, Nt=4, dt=5.0, window="win", omega="om", tag="shared"):
        self.Nf, self.Nt, self.dt = Nf, Nt, dt
        self.window, self.omega, self.tag = window, omega, tag

    @property
    def args(self):
        return (self.Nf, self.Nt, self.dt)

    @property
    def kwargs(self):
        return {"window": self.window, "omega": self.omega, "tag": self.tag}


class DeviceLocalDomainSettingsTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.globalfit.stock.erebor import source_runtime as sr
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"source_runtime not available: {exc}")
        self.sr = sr
        sr._DEVICE_DOMAIN_REPLICAS.clear()
        self.addCleanup(sr._DEVICE_DOMAIN_REPLICAS.clear)
        self.xp = RecordingXp()
        self.settings = _FakeDomainSettings()

    def test_primary_and_cpu_reuse_shared(self):
        rep = self.sr._device_local_domain_settings(self.settings, self.xp, 0)
        self.assertIs(rep, self.settings)                 # primary reuses
        rep = self.sr._device_local_domain_settings(self.settings, np, None)
        self.assertIs(rep, self.settings)                 # CPU reuses
        self.assertEqual(len(self.sr._DEVICE_DOMAIN_REPLICAS), 0)

    def test_nonprimary_rebuilds_dropping_window(self):
        with self.xp.cuda.Device(1):
            rep1 = self.sr._device_local_domain_settings(self.settings, self.xp, 0)
            rep2 = self.sr._device_local_domain_settings(self.settings, self.xp, 0)
        self.assertIsNot(rep1, self.settings)             # distinct replica
        self.assertIsInstance(rep1, _FakeDomainSettings)
        # window/omega were dropped -> regenerated by __init__ (default "win"),
        # not the primary-device array that was passed in.
        self.assertIs(rep1, rep2)                         # built once, cached
        self.assertEqual(len(self.sr._DEVICE_DOMAIN_REPLICAS), 1)
        self.assertIn(1, self.xp.device_log)              # built under device 1

    def test_unknown_settings_type_left_shared(self):
        plain = object()  # no args/kwargs
        with self.xp.cuda.Device(1):
            rep = self.sr._device_local_domain_settings(plain, self.xp, 0)
        self.assertIs(rep, plain)


class WrapDeviceAndOrbitsTest(unittest.TestCase):
    """The EMRI/SOBBH resolver: run cupy device + per-device orbits + per-device
    domain settings, keeping the shared objects on the primary device (so the
    wave-wrap cache and the inner id(orbits) generator caches fan out per
    device)."""

    def setUp(self):
        try:
            from lisatools.globalfit.stock.erebor import source_runtime as sr
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"source_runtime not available: {exc}")
        self.sr = sr
        sr._DEVICE_ORBITS_REPLICAS.clear()
        sr._DEVICE_DOMAIN_REPLICAS.clear()
        self.addCleanup(sr._DEVICE_ORBITS_REPLICAS.clear)
        self.addCleanup(sr._DEVICE_DOMAIN_REPLICAS.clear)
        self.xp = RecordingXp()
        self.orbits = _FakeOrbits(tag="shared")
        self.settings = _FakeDomainSettings()
        # gpu_orbits only needs an ``xp`` handle (the run's cupy module).
        self.gi = SimpleNamespace(
            orbits=self.orbits,
            gpu_orbits=SimpleNamespace(xp=self.xp),
            domain_settings=self.settings,
            gpus=[0, 1],
        )

    def test_primary_device_shares(self):
        xp, dev, orb, ds = self.sr._wrap_device_and_orbits(self.gi)
        self.assertIs(xp, self.xp)
        self.assertEqual(dev, 0)                 # RecordingXp default device
        self.assertIs(orb, self.orbits)          # primary reuses shared orbits
        self.assertIs(ds, self.settings)         # primary reuses shared domain
        self.assertEqual(len(self.sr._DEVICE_ORBITS_REPLICAS), 0)
        self.assertEqual(len(self.sr._DEVICE_DOMAIN_REPLICAS), 0)

    def test_nonprimary_device_gets_replicas(self):
        with self.xp.cuda.Device(1):
            xp, dev, orb, ds = self.sr._wrap_device_and_orbits(self.gi)
        self.assertEqual(dev, 1)
        self.assertIsNot(orb, self.orbits)
        self.assertIsInstance(orb, _FakeOrbits)
        self.assertIsNot(ds, self.settings)
        self.assertIsInstance(ds, _FakeDomainSettings)


class _FakeReplicaGen:
    """Stand-in generator tagged with the device it was resolved on."""

    def __init__(self, device):
        self.device = device
        self.kwargs = {"device": device}  # the DCGA duck-type marker
        self.calls = []

    def get_signals_for_residuals(self, *params, **kwargs):
        self.calls.append(params)
        return ("signals", self.device)

    def __call__(self, *params, **kwargs):
        # the EMRI/SOBBH shape: the resolved wrap is itself callable
        self.calls.append(params)
        return ("signals", self.device)


class DeviceLocalWaveGenLateBindingTest(unittest.TestCase):
    """``DeviceLocalWaveGen`` attributes must resolve at CALL time.

    The 2026-09-16 nogb-null ``mbh_pe`` crash: ``recipe.py``'s
    ``build_mbh_moves_phenom`` captures
    ``wave_gen.get_signals_for_residuals`` ONCE, at recipe-build time on the
    main thread (hence on ``gpus[0]``), and the move keeps it for the life of
    the run. With eager ``__getattr__`` that is permanently the PRIMARY
    device's generator, so a walker-shard worker thread running on ``gpus[1]``
    (``AnalysisContainerArray._vectorized_dispatch``) generated its template
    through gpu-0's cached device arrays -- the WDM window and the
    ``fold_shift_map`` gather indices read in
    ``domains.py::FDSignal.wdmtransform`` -- against a gpu-1 residual.
    Cross-device gather + multiply: an asynchronous
    ``cudaErrorIllegalAddress`` where P2P is unavailable, surfacing at the
    next allocation.

    EMRI and SOBBH pass the ``DeviceLocalWaveGen`` OBJECT
    (``source_runtime.py`` build_emri/build_sobbh), so they re-resolved on
    every call and were unaffected -- which is exactly why only the MBH block
    crashed.
    """

    def setUp(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        self.sr = sr
        self.current = {"device": 0}
        self.resolved = []

        def getter(general_info, cfg):
            dev = self.current["device"]
            self.resolved.append(dev)
            return self.gens.setdefault(dev, _FakeReplicaGen(dev))

        self.gens = {}
        self.wave_gen = sr.DeviceLocalWaveGen(getter, SimpleNamespace(), {})

    def test_captured_method_follows_the_device_at_call_time(self):
        # captured on the primary device, the way recipe.py does it
        fn = self.wave_gen.get_signals_for_residuals
        # ... then called from a shard worker running on device 1
        self.current["device"] = 1
        self.assertEqual(fn(1.0, 2.0), ("signals", 1))
        # and back on device 0 the same captured handle uses gpu 0 again
        self.current["device"] = 0
        self.assertEqual(fn(3.0), ("signals", 0))

        self.assertEqual(self.gens[1].calls, [(1.0, 2.0)])
        self.assertEqual(self.gens[0].calls, [(3.0,)])

    def test_direct_call_still_resolves_per_call(self):
        """The ``__call__`` path EMRI/SOBBH use is unchanged."""
        self.current["device"] = 1
        self.assertEqual(self.wave_gen(0.0), ("signals", 1))

    def test_dcga_duck_type_is_preserved(self):
        """``MoveBuilder`` reads ``__self__``/``__name__`` off the handle.

        ``recipe.py::MoveBuilder.build`` recovers the generator object via
        ``getattr(wave_gen, "__self__")`` and needs it to carry ``.kwargs``
        for the per-device DCGA replica path; the method name rides along as
        ``waveform_gen_method``.
        """
        fn = self.wave_gen.get_signals_for_residuals
        self.assertEqual(fn.__name__, "get_signals_for_residuals")
        owner = getattr(fn, "__self__", None)
        self.assertIsNotNone(owner)
        self.assertTrue(hasattr(owner, "kwargs"))
        self.assertIsInstance(owner, _FakeReplicaGen)

    def test_non_callable_attributes_still_pass_through(self):
        self.assertEqual(self.wave_gen.kwargs, {"device": 0})

    def test_underscore_attributes_still_raise(self):
        with self.assertRaises(AttributeError):
            self.wave_gen.__deepcopy__


if __name__ == "__main__":
    unittest.main()
