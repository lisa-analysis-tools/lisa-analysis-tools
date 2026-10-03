# tests/test_sobbh_lookup_router.py
"""Per-device replicas of the SOBBH lookup comp (multi-GPU walker shards).

The SOBBH move drives ONE comp object under each walker shard's device context; the lookup comp
holds device arrays (table coefficients, orbits) and refuses calls off its build device. The
router builds one replica per device on first use there and dispatches every call to the replica
of the CURRENT device, so the lookup can be the default on multi-GPU layouts too.
"""

import copy
import unittest
from types import SimpleNamespace
from unittest import mock


class _FakeComp:
    def __init__(self, device):
        self.device = device
        self.calls = []
        self.d_h_out = None
        self.h_h_out = None
        self.wdm_settings = f"wdm@{device}"
        self.d_d = 0.0

    def get_ll_wdm(self, params, holder, **kw):
        self.calls.append(("ll", params))
        self.d_h_out = f"dh@{self.device}"
        self.h_h_out = f"hh@{self.device}"
        return f"ll@{self.device}"

    def fill_global_wdm(self, params, templates, **kw):
        self.calls.append(("fill", params))


class RouterTest(unittest.TestCase):
    def _router(self, devices):
        from lisatools.sources.sobbh import wdm_direct as wd

        built = []

        def build(dev):
            built.append(dev)
            return _FakeComp(dev)

        seq = iter(devices)
        patcher = mock.patch.object(wd, "current_device", side_effect=lambda xp: next(seq))
        patcher.start()
        self.addCleanup(patcher.stop)
        return wd.SOBBHLookupRouter(build, "cpu"), built

    def test_one_replica_per_device_and_calls_go_to_the_current_one(self):
        # construction (primary on device 0), then calls on 1, 0, 1
        router, built = self._router([0, 1, 0, 1])
        self.assertEqual(built, [0])
        self.assertEqual(router.get_ll_wdm("p1", None), "ll@1")
        self.assertEqual(router.d_h_out, "dh@1")  # the stashes follow the last call
        self.assertEqual(router.get_ll_wdm("p0", None), "ll@0")
        self.assertEqual(router.h_h_out, "hh@0")
        router.fill_global_wdm("f1", None)
        self.assertEqual(built, [0, 1])  # device 1 built once, reused
        self.assertEqual(router.replicas[1].calls, [("ll", "p1"), ("fill", "f1")])
        self.assertEqual(router.replicas[0].calls, [("ll", "p0")])

    def test_attributes_delegate_and_dunders_do_not(self):
        router, _ = self._router([None])
        self.assertEqual(router.wdm_settings, "wdm@None")
        self.assertEqual(router.d_d, 0.0)
        with self.assertRaises(AttributeError):
            router.__deepcopy_probe__
        copy.copy(router)  # dunder probing on a shallow copy must not recurse

    def test_single_device_is_one_replica(self):
        router, built = self._router([None, None, None])
        router.get_ll_wdm("a", None)
        router.fill_global_wdm("b", None)
        self.assertEqual(built, [None])


class StockRouterTest(unittest.TestCase):
    def test_stock_fast_comp_is_the_router(self):
        import os
        import sys
        import tempfile

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _wdm_lookup_toy import build_tiny_table
        from test_sobbh_lookup_stock import _cfg, _general_info

        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations, SOBBHLookupRouter

        with tempfile.TemporaryDirectory() as tmp:
            wdm, table = build_tiny_table(tmp)
            gi = _general_info(wdm)
            gi.gpus = [0, 1]  # multi-GPU walker shards: no longer refused
            gi.gpu_orbits = gi.orbits  # (CPU stand-in; the run's device-0 orbits)
            comp = sr.get_sobbh_fast_comp(gi, _cfg(table.store_path))
            self.assertIsInstance(comp, SOBBHLookupRouter)
            self.assertIsInstance(comp.primary, SOBBHLookupComputations)
            self.assertIs(sr.get_sobbh_fast_comp(gi, _cfg(table.store_path)), comp)


if __name__ == "__main__":
    unittest.main()
