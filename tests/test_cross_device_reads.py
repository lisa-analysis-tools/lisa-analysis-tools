"""No cross-device reads in the WDM transform or the SOBBH Gram build.

The 6mo relaunch (2026-10-09) died with an illegal memory access that
``CUDA_LAUNCH_BLOCKING=1`` made disappear: a race. The log carried cupy's
"The device where the array resides (0) is different from the current device
(1)" warnings from two places: ``FDSignal.wdmtransform`` reading the WDM
settings' window and fold map (cached on GPU 0) for a GPU-1 signal, and the
SOBBH Gram eigen table multiplying templates on the comp's device by the
walker container's ``invC`` on the walker's device. Peer reads are unordered
against the owning device's stream.

There is no second GPU here, so a fake two-device ``xp`` (numpy arrays
tagged with a device id; ``asnumpy`` counts host hops) checks the helpers,
and the WDM transform's wiring is checked by patching them.
"""
from __future__ import annotations

import copy
import pickle
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.sensitivity import SensitivityMatrixBase
from lisatools.utils import device as D


class _Arr(np.ndarray):
    """A numpy array that says which (fake) GPU it lives on."""

    def __array_finalize__(self, obj):
        self._dev = getattr(obj, "_dev", 0)

    @property
    def device(self):
        return SimpleNamespace(id=self._dev)


def _on(a, dev):
    out = np.array(a).view(_Arr)
    out._dev = int(dev)
    return out


class _TwoGPU:
    """Just the cupy surface ``lisatools.utils.device`` touches."""

    def __init__(self):
        self.current = 0
        self.hops = 0
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(getDevice=lambda: self.current),
            Device=self._device)

    @contextmanager
    def _device(self, i):
        prev, self.current = self.current, int(i)
        try:
            yield
        finally:
            self.current = prev

    def asnumpy(self, a):
        self.hops += 1
        return np.array(a).view(np.ndarray)

    def asarray(self, a):
        return _on(a, self.current)


class ToCurrentDeviceCachedTest(unittest.TestCase):
    def setUp(self):
        self.xp = _TwoGPU()
        self.owner = SimpleNamespace()
        self.src = (_on(np.arange(4.0), 0), _on(np.ones(3), 0))

    def test_a_foreign_array_is_copied_once_per_device(self):
        self.xp.current = 1
        a, b = D.to_current_device_cached(self.xp, self.owner, "t", self.src)
        self.assertEqual((a.device.id, b.device.id), (1, 1))
        np.testing.assert_array_equal(a, self.src[0])
        self.assertEqual(self.xp.hops, 2)
        again = D.to_current_device_cached(self.xp, self.owner, "t", self.src)
        self.assertIs(again[0], a)
        self.assertEqual(self.xp.hops, 2)          # cached: no new host hop

    def test_a_rebuilt_source_is_copied_again(self):
        self.xp.current = 1
        a, _ = D.to_current_device_cached(self.xp, self.owner, "t", self.src)
        rebuilt = (_on(np.arange(4.0) + 1, 0), self.src[1])
        a2, _ = D.to_current_device_cached(self.xp, self.owner, "t", rebuilt)
        self.assertIsNot(a2, a)
        np.testing.assert_array_equal(a2, rebuilt[0])

    def test_home_device_reads_are_the_source_objects(self):
        out = D.to_current_device_cached(self.xp, self.owner, "t", self.src)
        self.assertIs(out[0], self.src[0])
        self.assertEqual(self.xp.hops, 0)

    def test_numpy_is_a_strict_noop(self):
        src = (np.arange(3.0),)
        out = D.to_current_device_cached(np, self.owner, "t", src)
        self.assertIs(out[0], src[0])
        self.assertFalse(hasattr(self.owner, "_device_copy_cache"))


def _sens(xp_dev, dirty=False):
    """A real ``SensitivityMatrixBase`` (no basis needed) on fake GPU ``xp_dev``."""
    sm = object.__new__(SensitivityMatrixBase)
    sm._sens_mat = _on(np.full((2, 2, 5), 2.0), xp_dev)
    sm._invC = _on(np.full((2, 2, 5), 0.5), xp_dev)
    sm._detC = _on(np.full(5, 4.0), xp_dev)
    sm.do_inv_det = True
    sm._inv_det_dirty = dirty
    sm.tag = "shared"
    return sm


class SensitivityToCurrentDeviceTest(unittest.TestCase):
    def setUp(self):
        self.xp = _TwoGPU()

    def test_foreign_psd_becomes_a_same_class_clone_on_this_device(self):
        sm = _sens(1)
        out = D.sensitivity_to_current_device(self.xp, sm)
        # inner_product type-checks its psd: a proxy would be rebuilt as a
        # brand-new SensitivityMatrix around the proxy
        self.assertIs(type(out), SensitivityMatrixBase)
        self.assertEqual((out.invC.device.id, out.detC.device.id), (0, 0))
        np.testing.assert_array_equal(out.invC, sm.invC)
        self.assertIs(out.sens_mat, sm.sens_mat)        # shared, shape only
        self.assertEqual(out.tag, "shared")
        self.assertEqual(sm.invC.device.id, 1)          # the original is untouched

    def test_a_dirty_psd_is_inverted_on_its_own_device(self):
        sm = _sens(1, dirty=True)
        seen = []

        def setup():
            seen.append(self.xp.current)
            sm._invC = _on(np.full((2, 2, 5), 0.25), 1)
            sm._detC = _on(np.full(5, 16.0), 1)
            sm._inv_det_dirty = False

        sm._setup_det_and_inv = setup
        out = D.sensitivity_to_current_device(self.xp, sm)
        self.assertEqual(seen, [1])
        np.testing.assert_array_equal(out.invC, 0.25)
        self.assertFalse(out._inv_det_dirty)

    def test_same_device_and_numpy_return_the_object_itself(self):
        sm = _sens(0)
        self.assertIs(D.sensitivity_to_current_device(self.xp, sm), sm)
        self.assertIs(D.sensitivity_to_current_device(np, sm), sm)
        self.assertEqual(self.xp.hops, 0)


class WDMTransformWiringTest(unittest.TestCase):
    """``wdmtransform`` runs in the signal's device context and reads the
    settings' window / fold map through the per-device cache."""

    def _signal(self):
        from lisatools.domains import FDSettings, FDSignal, WDMSettings

        wdm = WDMSettings(Nf=128, Nt=64, dt=5.0, min_freq=3e-4, max_freq=8e-3,
                          force_backend="cpu")
        f = np.fft.rfftfreq(wdm.N, wdm.data_dt)
        fd = FDSettings(f.shape[0], float(f[1] - f[0]), force_backend="cpu")
        rng = np.random.default_rng(3)
        arr = rng.normal(size=(3, fd.N)) + 1j * rng.normal(size=(3, fd.N))
        return FDSignal(arr, fd), wdm

    def test_settings_arrays_go_through_the_per_device_cache(self):
        sig, wdm = self._signal()
        ref = sig.wdmtransform(settings=wdm)
        tags = []
        real = D.to_current_device_cached

        def spy(xp, owner, tag, arrays):
            self.assertIs(owner, wdm)
            tags.append(tag)
            return real(xp, owner, tag, arrays)

        with mock.patch.object(D, "to_current_device_cached", side_effect=spy), \
                mock.patch.object(D, "device_context",
                                  side_effect=D.device_context) as ctx:
            out = sig.wdmtransform(settings=wdm)
        self.assertEqual(sorted(tags), ["fold_shift_map", "window"])
        ctx.assert_called_once()
        np.testing.assert_array_equal(np.asarray(out.arr), np.asarray(ref.arr))

    def test_the_transform_uses_what_the_cache_returns(self):
        # a cache that hands back a doubled window must change the output:
        # the fold map / window are not read from the settings behind its back
        sig, wdm = self._signal()
        ref = np.asarray(sig.wdmtransform(settings=wdm).arr)

        def doubled(xp, owner, tag, arrays):
            arrays = tuple(arrays)
            return (2.0 * arrays[0],) if tag == "window" else arrays

        with mock.patch.object(D, "to_current_device_cached", side_effect=doubled):
            out = np.asarray(sig.wdmtransform(settings=wdm).arr)
        np.testing.assert_allclose(out, 2.0 * ref, rtol=1e-12, atol=1e-30)

    def test_copies_and_pickles_drop_the_device_copies(self):
        _, wdm = self._signal()
        wdm._device_copy_cache = {("window", 1): ((), ())}
        for clone in (copy.deepcopy(wdm), pickle.loads(pickle.dumps(wdm))):
            self.assertFalse(hasattr(clone, "_device_copy_cache"))


if __name__ == "__main__":
    unittest.main()
