"""Per-device caching of the chunked-het static geometry kernel args.

``get_ll_wdm`` re-asserted the chunk geometry + WDM window through
``xp.asarray`` on EVERY call so that a call routed into another shard's
device context (the SOBBH/GB multi-GPU walker-shard routers) handed the
kernel device-local pointers. Those five arrays are STATIC -- built once in
``__init__`` and never mutated -- so on the home device the re-assert is a
wasted no-op and on a shard device it is a full re-upload of the window per
scoring call: row-independent cost paid once per call, and job 508's
[SOBBH_LL_TIMING] put essentially the whole 3.48 s/call inside that call.

These tests pin the cache: asserted ONCE per CUDA device, reused forever
after (the arrays are geometry; there is nothing to invalidate).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.chunked_het import WDMComputationsBase


class _CountingXp:
    """NumPy stand-in that counts ``asarray`` and fakes a CUDA device id."""

    def __init__(self, device=None):
        self.asarray_calls = 0
        self.device = device
        if device is not None:
            self.cuda = SimpleNamespace(
                runtime=SimpleNamespace(getDevice=lambda: self.device)
            )

    def asarray(self, a, dtype=None):
        self.asarray_calls += 1
        return np.asarray(a, dtype=dtype)

    def __getattr__(self, name):  # delegate everything else to numpy
        return getattr(np, name)


class _GeometryOnlyComp(WDMComputationsBase):
    """Static geometry + an ``xp`` stand-in; no backend resolution.

    ``WDMComputationsBase.xp`` is a read-only property deriving from the
    resolved backend, so the fake is installed by overriding it rather than
    by assignment.
    """

    def __init__(self, xp):  # noqa: D107 - deliberately skips super()
        self._fake_xp = xp
        self.chunk_t_starts = np.arange(4, dtype=float)
        self.chunk_keep_lo = np.zeros(4, dtype=np.int32)
        self.chunk_keep_hi = np.full(4, 3, dtype=np.int32)
        self.chunk_n_global_offset = np.arange(4, dtype=np.int32)
        self.wdm_window = np.ones(16, dtype=float)

    @property
    def xp(self):
        return self._fake_xp


def _bare_comp(xp):
    """Instance carrying only the static geometry the helper reads."""
    return _GeometryOnlyComp(xp)


class GeometryArgsShapeTest(unittest.TestCase):
    def test_returns_the_five_geometry_args_in_kernel_order(self):
        xp = _CountingXp()
        comp = _bare_comp(xp)
        args = comp._geometry_kernel_args()
        self.assertEqual(len(args), 5)
        np.testing.assert_array_equal(args[0], comp.chunk_t_starts)
        np.testing.assert_array_equal(args[1], comp.chunk_keep_lo)
        np.testing.assert_array_equal(args[2], comp.chunk_keep_hi)
        np.testing.assert_array_equal(args[3], comp.chunk_n_global_offset)
        np.testing.assert_array_equal(args[4], comp.wdm_window)


class GeometryArgsCacheTest(unittest.TestCase):
    def test_first_call_asserts_all_five(self):
        xp = _CountingXp()
        comp = _bare_comp(xp)
        comp._geometry_kernel_args()
        self.assertEqual(xp.asarray_calls, 5)

    def test_second_call_on_the_same_device_skips_the_upload(self):
        xp = _CountingXp()
        comp = _bare_comp(xp)
        comp._geometry_kernel_args()
        comp._geometry_kernel_args()
        self.assertEqual(
            xp.asarray_calls, 5,
            "the second call must re-use the cached device arrays",
        )

    def test_cached_args_are_the_identical_objects(self):
        comp = _bare_comp(_CountingXp())
        first = comp._geometry_kernel_args()
        second = comp._geometry_kernel_args()
        for a, b in zip(first, second):
            self.assertIs(a, b)


class GeometryArgsPerDeviceTest(unittest.TestCase):
    def test_a_second_device_gets_its_own_upload(self):
        xp = _CountingXp(device=0)
        comp = _bare_comp(xp)
        comp._geometry_kernel_args()
        self.assertEqual(xp.asarray_calls, 5)
        xp.device = 1
        comp._geometry_kernel_args()
        self.assertEqual(
            xp.asarray_calls, 10,
            "device 1 needs its own device-local copies",
        )

    def test_returning_to_the_first_device_reuses_its_cache(self):
        xp = _CountingXp(device=0)
        comp = _bare_comp(xp)
        dev0 = comp._geometry_kernel_args()
        xp.device = 1
        comp._geometry_kernel_args()
        xp.device = 0
        again = comp._geometry_kernel_args()
        self.assertEqual(xp.asarray_calls, 10)
        for a, b in zip(dev0, again):
            self.assertIs(a, b)

    def test_shard_round_trip_uploads_once_per_device_not_per_call(self):
        """The routed-scoring pattern: alternating shards, many calls."""
        xp = _CountingXp(device=0)
        comp = _bare_comp(xp)
        for _ in range(25):
            for dev in (0, 1):
                xp.device = dev
                comp._geometry_kernel_args()
        self.assertEqual(xp.asarray_calls, 10)


class _FakeHolder:
    def __init__(self):
        self.linear_data_arr = [np.zeros(8)]
        self.linear_psd_arr = [np.ones(8)]

    def __len__(self):
        return 1


class _KernelRecordingComp(_GeometryOnlyComp):
    """``get_ll_wdm`` with every collaborator stubbed, recording kernel args.

    Everything but the geometry path is faked so the test can drive the REAL
    ``get_ll_wdm`` body cheaply and inspect exactly which arrays reach the
    kernel.
    """

    def __init__(self, xp):
        super().__init__(xp)
        self.kernel_args = []
        self.d_d = 0.0
        self.n_chunks = 4
        self.Nt_sub, self.log2_Nt_sub = 32, 5
        self.N_sparse, self.log2_N_sparse = 64, 6
        self.nchannels, self.n_rfft_chunk = 3, 17
        self.T_chunk, self.dt, self.T, self.t_ref = 1.0, 10.0, 100.0, 0.0
        self.tdi_type = "XYZ"
        self.resolved_tukey_alpha = 0.0
        self.N_cp_sig, self.N_cp_orbit = 0, 0
        self.layer_df = 1e-4
        self.cpp_orbits = object()
        self.cpp_tdi_config = object()
        self.cpp_wdm_settings = object()

    @property
    def backend(self):
        return SimpleNamespace(name="cpu", TDITypeDict={"XYZ": 1})

    def _as_wdm_holder(self, holder):
        return holder

    def _prep_indices(self, num_bin, num_data, num_noise, d_idx, n_idx):
        z = np.zeros(num_bin, dtype=np.int32)
        return z, z

    def _layer_groups(self, params_2d, *args, **kwargs):
        return self._empty_groups(len(params_2d))

    def _slab_kernel_args(self, holder):
        return ()

    def _psd_kernel_args(self, holder):
        return ()

    def _kernel(self, name):
        def _record(*args):
            self.kernel_args.append(args)

        return _record


#: Positions of the five geometry args in the ``get_ll`` kernel signature
#: (d_h_out, h_h_out, orbits, tdi_config, wdm_settings, params, data_index,
#: noise_index, THEN the geometry).
_GEOM_SLICE = slice(8, 13)


class GetLlWdmUsesTheCacheTest(unittest.TestCase):
    """The scoring entry point must go THROUGH the cache, not around it."""

    def _params(self, n=3):
        p = np.zeros((n, WDMComputationsBase._NPARAMS))
        p[:, WDMComputationsBase._F0_PARAM_INDEX] = 1e-3
        return p

    def test_kernel_receives_the_geometry_args(self):
        comp = _KernelRecordingComp(_CountingXp())
        comp.get_ll_wdm(self._params(), _FakeHolder())
        geom = comp.kernel_args[0][_GEOM_SLICE]
        self.assertEqual(len(geom), 5)
        np.testing.assert_array_equal(geom[4], comp.wdm_window)

    def test_two_calls_pass_the_identical_geometry_objects(self):
        comp = _KernelRecordingComp(_CountingXp())
        holder = _FakeHolder()
        comp.get_ll_wdm(self._params(), holder)
        comp.get_ll_wdm(self._params(4), holder)
        first = comp.kernel_args[0][_GEOM_SLICE]
        second = comp.kernel_args[1][_GEOM_SLICE]
        for a, b in zip(first, second):
            self.assertIs(
                a, b, "geometry was re-asserted on the second scoring call"
            )

    def test_one_cache_entry_after_many_calls(self):
        comp = _KernelRecordingComp(_CountingXp())
        holder = _FakeHolder()
        for _ in range(5):
            comp.get_ll_wdm(self._params(), holder)
        self.assertEqual(len(comp._geom_args_by_device), 1)

    def test_call_records_its_span_breakdown(self):
        comp = _KernelRecordingComp(_CountingXp())
        comp.get_ll_wdm(self._params(), _FakeHolder())
        spans = comp.last_call_spans
        for key in ("stage", "geom", "wrap", "launch", "total"):
            self.assertIn(key, spans)
            self.assertGreaterEqual(spans[key], 0.0)
        self.assertEqual(spans["num_bin"], 3)
        self.assertIn("n_groups", spans)
        self.assertGreaterEqual(
            spans["total"],
            spans["stage"] + spans["geom"] + spans["wrap"] - 1e-9,
        )


if __name__ == "__main__":
    unittest.main()
