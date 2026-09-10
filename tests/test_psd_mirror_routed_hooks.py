"""Shared-psd MIRROR parity gate: hooked where PRODUCTION calls land.

COVERAGE GAP (measured on the production runs, jobs 469/470): only the
``"get_ll"`` and ``"sighet_setup"`` entries of the
``GB_PSD_MIRROR_PARITY_PROPOSES`` gate ever fired. The gate was hooked on
``SubBandBuffer``'s OWN methods, but the RJ path scores through
``_RoutedBandEngine.get_swap_ll`` (the buffer method is not on that path at
all), ``_RoutedBandEngine.get_ll`` carried no hook of its own, and
``route_fstat_ll``'s bare ``getattr(holder, "_psd_mirror_parity_check")``
finds nothing whenever the holder is a VIEW (:class:`_ShardHolderView`,
:class:`_FStatRefRowHolder`) -- so it silently skipped.

Pinned here with FAKES only (no GB comps, no orbit files, no real buffer):

* ``get_ll`` / ``get_swap_ll`` / ``route_fstat_ll`` each fire the gate
  EXACTLY ONCE per production call, with their own entry name, the physical
  params, and row-aligned remove/add params for the swap entry;
* rows reach the check as the buffer's GLOBAL slot ids -- a
  ``_ShardHolderView`` holder's INTRA-shard indices are translated through
  its ``rows`` map (the check indexes ``psd_buffer`` /
  ``_get_fill_buffer_ind_map`` with global slots);
* nothing fires when the gate is disarmed, when no gated buffer owns the
  holder, or when a hop in the ``_parent`` chain re-indexes rows without
  publishing a map (never compared against the wrong slots);
* ``SubBandBuffer.get_ll`` does NOT double-fire behind a routed engine, and
  still fires its own fallback behind a non-routed one.

CPU-only, microseconds, a few kB of arrays.
"""

from __future__ import annotations

import unittest

import numpy as np

from lisatools.globalfit.moves.gbbands import (
    SubBandBuffer,
    _FStatRefRowHolder,
    _RoutedBandEngine,
    _ShardHolderView,
    _parity_target,
)


class _RecordingBuffer:
    """Stub :class:`SubBandBuffer` exposing exactly what the gate consumes.

    ``_psd_mirror_parity_check`` records instead of shadow-scoring, but keeps
    the real method's counter contract (every ARMED invocation bumps
    ``_psd_mirror_parity_ncheck``) so the double-fire guard is under test.
    """

    def __init__(self, armed=True, n_shards=1):
        self.xp = np
        self.linear_data_arr = [np.zeros(4) for _ in range(n_shards)]
        self.linear_psd_arr = [np.zeros(4) for _ in range(n_shards)]
        self._psd_mirror_parity = bool(armed)
        self._psd_mirror_parity_ncheck = 0
        self.calls = []

    def _psd_mirror_parity_check(self, entry, params_phys, data_index,
                                 noise_index=None, params_remove_phys=None,
                                 fstat_fold=None):
        if not self._psd_mirror_parity:
            return
        self._psd_mirror_parity_ncheck += 1
        self.calls.append(dict(
            entry=entry,
            params=np.asarray(params_phys),
            data_index=np.asarray(data_index),
            noise_index=None if noise_index is None else np.asarray(noise_index),
            params_remove=(None if params_remove_phys is None
                           else np.asarray(params_remove_phys)),
            fstat_fold=fstat_fold,
        ))


class _PlainHolder:
    """A holder with no gate anywhere in its chain (the parent residual ACA)."""

    def __init__(self):
        self.xp = np
        self.linear_data_arr = [np.zeros(4)]
        self.linear_psd_arr = [np.zeros(4)]


class _StubEngine:
    """Single-shard band engine stub: records, returns zeros."""

    def __init__(self):
        self.calls = []
        self.ncheck_at_call = []
        self.d_h_out = np.zeros(2)
        self.h_h_out = np.zeros(2)
        self.phase_angle = None
        self.kept_out = np.ones(2, dtype=bool)

    def _seen(self, holder):
        self.ncheck_at_call.append(
            int(getattr(holder, "_psd_mirror_parity_ncheck", -1)))

    def get_ll(self, holder, params_phys, **kwargs):
        self._seen(holder)
        self.calls.append(("get_ll", kwargs))
        return np.zeros(int(np.asarray(params_phys).shape[0]))

    def get_swap_ll(self, holder, params_remove_phys, params_add_phys,
                    **kwargs):
        self._seen(holder)
        self.calls.append(("get_swap_ll", kwargs))
        return "swap-result"


class _StubComp:
    """Raw F-stat comp stub (``route_fstat_ll`` takes the OBJECT + name)."""

    def __init__(self):
        self.calls = []

    def get_fstat_ll_wdm(self, params_phys, holder, *, data_index,
                         noise_index=None, **kwargs):
        self.calls.append(dict(data_index=data_index, noise_index=noise_index,
                               kwargs=kwargs))
        n = int(np.atleast_2d(params_phys).shape[0])
        return np.zeros((n, 4)), np.zeros((n, 10))


def _params(n, ndim=8, base=0.0):
    return base + np.arange(n * ndim, dtype=float).reshape(n, ndim)


class RoutedParityHookTest(unittest.TestCase):
    """The three routed entry points fire the gate once, with global rows."""

    def setUp(self):
        self.buf = _RecordingBuffer()
        self.engine = _StubEngine()
        self.routed = _RoutedBandEngine(self.engine)
        self.di = np.array([3, 1, 4], dtype=np.int32)
        self.ni = np.array([3, 1, 4], dtype=np.int32)

    def test_get_ll_fires_once_with_global_rows(self):
        p = _params(3)
        out = self.routed.get_ll(self.buf, p, data_index=self.di,
                                 noise_index=self.ni, N_vals=None,
                                 waveform_kwargs={})
        self.assertEqual(len(out), 3)
        self.assertEqual(len(self.buf.calls), 1)
        c = self.buf.calls[0]
        self.assertEqual(c["entry"], "get_ll")
        np.testing.assert_array_equal(c["data_index"], self.di)
        np.testing.assert_array_equal(c["noise_index"], self.ni)
        np.testing.assert_array_equal(c["params"], p)
        self.assertIsNone(c["params_remove"])
        # fired BEFORE the production kernel (so the shadow scoring can never
        # clobber the accumulators the caller reads afterwards)
        self.assertEqual(self.engine.ncheck_at_call, [1])

    def test_get_swap_ll_fires_once_row_aligned(self):
        add = _params(3, base=100.0)
        rem = _params(3, base=200.0)
        res = self.routed.get_swap_ll(self.buf, rem, add, data_index=self.di,
                                      noise_index=self.ni, N_vals=None,
                                      waveform_kwargs={})
        self.assertEqual(res, "swap-result")
        self.assertEqual(len(self.buf.calls), 1)
        c = self.buf.calls[0]
        self.assertEqual(c["entry"], "swap_ll")
        np.testing.assert_array_equal(c["params"], add)
        np.testing.assert_array_equal(c["params_remove"], rem)
        self.assertEqual(c["params"].shape, c["params_remove"].shape)
        np.testing.assert_array_equal(c["data_index"], self.di)
        self.assertEqual(self.engine.ncheck_at_call, [1])

    def test_route_fstat_fires_once(self):
        comp = _StubComp()
        p = _params(3)
        N, M = _RoutedBandEngine.route_fstat_ll(
            comp, "get_fstat_ll_wdm", self.buf, p, data_index=self.di,
            noise_index=self.ni, fstat_fold=7)
        self.assertEqual(N.shape, (3, 4))
        self.assertEqual(M.shape, (3, 10))
        self.assertEqual(len(self.buf.calls), 1)
        c = self.buf.calls[0]
        self.assertEqual(c["entry"], "fstat")
        self.assertEqual(c["fstat_fold"], 7)
        np.testing.assert_array_equal(c["data_index"], self.di)
        self.assertEqual(len(comp.calls), 1)

    def test_fstat_noise_index_defaults_to_data_index(self):
        comp = _StubComp()
        _RoutedBandEngine.route_fstat_ll(
            comp, "get_fstat_ll_wdm", self.buf, _params(3),
            data_index=self.di)
        c = self.buf.calls[0]
        np.testing.assert_array_equal(c["noise_index"], self.di)

    def test_one_dim_params_reach_the_check_as_a_row(self):
        comp = _StubComp()
        _RoutedBandEngine.route_fstat_ll(
            comp, "get_fstat_ll_wdm", self.buf, _params(1)[0],
            data_index=np.array([2], dtype=np.int32))
        self.assertEqual(self.buf.calls[0]["params"].shape, (1, 8))


class RoutedParityNoFireTest(unittest.TestCase):
    """Every case that must stay silent."""

    def setUp(self):
        self.engine = _StubEngine()
        self.routed = _RoutedBandEngine(self.engine)
        self.di = np.array([0, 1], dtype=np.int32)

    def _call_all(self, holder):
        self.routed.get_ll(holder, _params(2), data_index=self.di,
                           noise_index=self.di, N_vals=None,
                           waveform_kwargs={})
        self.routed.get_swap_ll(holder, _params(2), _params(2),
                                data_index=self.di, noise_index=self.di,
                                N_vals=None, waveform_kwargs={})
        _RoutedBandEngine.route_fstat_ll(
            _StubComp(), "get_fstat_ll_wdm", holder, _params(2),
            data_index=self.di, noise_index=self.di)

    def test_disarmed_buffer_never_fires(self):
        buf = _RecordingBuffer(armed=False)
        self._call_all(buf)
        self.assertEqual(buf.calls, [])
        self.assertEqual(buf._psd_mirror_parity_ncheck, 0)

    def test_holder_with_no_gate_anywhere(self):
        holder = _PlainHolder()
        self._call_all(holder)  # must not raise
        self.assertEqual(_parity_target(holder), (None, None))

    def test_unmapped_view_chain_is_refused(self):
        """A hop that re-indexes rows without publishing a map (the F-stat
        reference-row holder, whose single row is a device-local COPY) must
        SKIP -- never hand intra-view indices to a global-slot check."""
        buf = _RecordingBuffer()
        view = _FStatRefRowHolder(buf, None, np.zeros(4), np.zeros(4))
        self.assertEqual(_parity_target(view), (None, None))
        self.routed.get_ll(view, _params(1), data_index=np.array([0]),
                           noise_index=np.array([0]), N_vals=None,
                           waveform_kwargs={})
        self.assertEqual(buf.calls, [])

    def test_no_data_index_skips(self):
        buf = _RecordingBuffer()
        _RoutedBandEngine.route_fstat_ll(
            _StubComp(), "get_fstat_ll_wdm", buf, _params(2),
            data_index=None)
        self.assertEqual(buf.calls, [])


class _ShardParent(_RecordingBuffer):
    """Two-shard gated buffer a real ``_ShardHolderView`` can wrap."""

    def __init__(self, splits):
        super().__init__(n_shards=len(splits))
        self.gpus = None
        self.gpu_splits = [np.asarray(s, dtype=int) for s in splits]
        self.acs_total_entries = int(sum(len(s) for s in self.gpu_splits))


class ShardViewIndexSpaceTest(unittest.TestCase):
    """A view's INTRA-shard rows must reach the check as GLOBAL slot ids."""

    def setUp(self):
        self.parent = _ShardParent([[2, 5, 7], [0, 1, 3, 6]])
        self.engine = _StubEngine()
        self.routed = _RoutedBandEngine(self.engine)

    def test_parity_target_resolves_parent_and_rows(self):
        view = _ShardHolderView(self.parent, 1)
        buf, rows = _parity_target(view)
        self.assertIs(buf, self.parent)
        np.testing.assert_array_equal(rows, [0, 1, 3, 6])

    def test_get_ll_translates_intra_shard_rows(self):
        view = _ShardHolderView(self.parent, 0)
        intra = np.array([2, 0], dtype=np.int32)
        self.routed.get_ll(view, _params(2), data_index=intra,
                           noise_index=intra, N_vals=None,
                           waveform_kwargs={})
        self.assertEqual(len(self.parent.calls), 1)
        c = self.parent.calls[0]
        self.assertEqual(c["entry"], "get_ll")
        np.testing.assert_array_equal(c["data_index"], [7, 2])
        np.testing.assert_array_equal(c["noise_index"], [7, 2])

    def test_get_swap_ll_translates_intra_shard_rows(self):
        view = _ShardHolderView(self.parent, 1)
        intra = np.array([0, 3], dtype=np.int32)
        self.routed.get_swap_ll(view, _params(2), _params(2),
                                data_index=intra, noise_index=intra,
                                N_vals=None, waveform_kwargs={})
        c = self.parent.calls[0]
        self.assertEqual(c["entry"], "swap_ll")
        np.testing.assert_array_equal(c["data_index"], [0, 6])

    def test_buffer_holder_needs_no_translation(self):
        buf, rows = _parity_target(self.parent)
        self.assertIs(buf, self.parent)
        self.assertIsNone(rows)


class _MethodBuffer(_RecordingBuffer):
    """Stub with the REAL ``SubBandBuffer.get_ll`` bound on (the fallback +
    double-fire guard live in that method)."""

    get_ll = SubBandBuffer.get_ll

    def __init__(self, engine, armed=True):
        super().__init__(armed=armed)
        self._likelihood_engine = engine
        self.waveform_kwargs = {}

    def _to_phys(self, params, leaf_inds=None):
        return params


class BufferMethodNoDoubleFireTest(unittest.TestCase):
    """One production call == one check, whichever engine is underneath."""

    def _run(self, engine):
        buf = _MethodBuffer(engine)
        di = np.array([1, 2], dtype=np.int32)
        buf.get_ll(_params(2), di, di, None)
        return buf

    def test_routed_engine_fires_once_from_the_router(self):
        stub = _StubEngine()
        buf = self._run(_RoutedBandEngine(stub))
        self.assertEqual(len(buf.calls), 1, "gate must fire exactly once")
        self.assertEqual(buf.calls[0]["entry"], "get_ll")
        # the router fired it: the counter was already up when the wrapped
        # engine ran, and the method's own fallback then stood down
        self.assertEqual(stub.ncheck_at_call, [1])

    def test_non_routed_engine_still_fires_the_fallback(self):
        stub = _StubEngine()
        buf = self._run(stub)
        self.assertEqual(len(buf.calls), 1)
        self.assertEqual(buf.calls[0]["entry"], "get_ll")
        # nothing fired before the engine call -> the fallback did the work
        self.assertEqual(stub.ncheck_at_call, [0])

    def test_disarmed_fires_nothing_either_way(self):
        for engine in (_StubEngine(), _RoutedBandEngine(_StubEngine())):
            buf = _MethodBuffer(engine, armed=False)
            di = np.array([0, 1], dtype=np.int32)
            buf.get_ll(_params(2), di, di, None)
            self.assertEqual(buf.calls, [])


if __name__ == "__main__":
    unittest.main()
