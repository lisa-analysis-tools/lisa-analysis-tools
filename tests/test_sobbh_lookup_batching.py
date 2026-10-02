# tests/test_sobbh_lookup_batching.py
"""One batched response per comp call; the lookup runs in row blocks of ``row_batch``.

On the sparse 12-h grid the response outputs are tiny (~361 points per row over 6 months), so
the response is built ONCE for all rows of a call (its fixed cost -- the PN at the node grid,
the generator, the input/output spline fits -- is paid once), and ``row_batch`` only bounds the
tracer / lookup temporaries (~17 MB per row at 6 months). Cluster, one H100, before this change:
the 288-row call paid the ~35 ms response fixed cost nine times (0.366 of its 0.569 s).
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table  # noqa: E402
from test_sobbh_wdm_direct import DT, NF, NT, REF, ROWS, T_START  # noqa: E402


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.domains import WDMSettings, WDMSignal
        from lisatools.sensitivity import XYZ2SensitivityMatrix

        cls.tmp = tempfile.TemporaryDirectory()
        _, cls.table = build_tiny_table(cls.tmp.name)
        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")
        cls.wdm = WDMSettings(
            NF, NT, DT, t0=T_START, min_freq=2e-3, max_freq=2e-2, force_backend="cpu"
        )
        cls.rows = np.vstack([ROWS, ROWS[:1] * [1, 1, 1, 1, 1, 1.0004, 1, 1, 1, 1, 1]])
        comp = cls._comp(row_batch=64)
        nch, nfa, nta = 3, int(cls.wdm.Nf_active), int(cls.wdm.Nt_active)
        data = np.zeros((2, nch, nfa, nta))
        comp.fill_global_wdm(
            ROWS[:2], data, data_index=np.array([0, 1]), factors=np.array([1.0, -0.5])
        )
        cls.aca = AnalysisContainerArray(
            [
                AnalysisContainer(
                    WDMSignal(np.ascontiguousarray(data[k]), cls.wdm),
                    XYZ2SensitivityMatrix(cls.wdm, model="scirdv1"),
                )
                for k in range(2)
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def _comp(cls, row_batch):
        from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

        return SOBBHLookupComputations(
            cls.wdm,
            REF,
            cls.table,
            orbits=cls.orbits,
            tdi_config="2nd generation",
            tdi_type="XYZ",
            n_grid=1024,
            row_batch=row_batch,
            force_backend="cpu",
        )


def _count_builds(comp):
    calls = []
    orig = comp.direct.tof.build

    def build(params, t_lo, t_hi):
        calls.append(np.atleast_2d(params).shape[0])
        return orig(params, t_lo, t_hi)

    comp.direct.tof.build = build
    return calls


class TracerSubsetTest(_Fixture):
    def test_tracer_on_a_row_range_equals_the_full_tracer_slice(self):
        from lisatools.sources.sobbh.wdm_direct import sobbh_tracer

        direct = self._comp(row_batch=64).direct
        out = direct.response(self.rows)
        full = [np.asarray(a) for a in sobbh_tracer(out, direct.t_pixels)]
        part = [np.asarray(a) for a in sobbh_tracer(out, direct.t_pixels, rows=(1, 3))]
        for a, b in zip(full, part):
            self.assertEqual(b.shape[0], 2)
            np.testing.assert_array_equal(b, a[1:3])


class OneResponsePerCallTest(_Fixture):
    def test_get_ll_builds_the_response_once_for_all_rows(self):
        comp = self._comp(row_batch=2)
        calls = _count_builds(comp)
        idx = np.arange(self.rows.shape[0]) % 2
        comp.get_ll_wdm(self.rows, self.aca, data_index=idx, noise_index=idx)
        self.assertEqual(calls, [self.rows.shape[0]])

    def test_fill_builds_the_response_once_for_all_rows(self):
        comp = self._comp(row_batch=2)
        calls = _count_builds(comp)
        nch, nfa, nta = 3, int(self.wdm.Nf_active), int(self.wdm.Nt_active)
        buf = np.zeros((2, nch, nfa, nta))
        comp.fill_global_wdm(self.rows, buf, data_index=np.arange(self.rows.shape[0]) % 2)
        self.assertEqual(calls, [self.rows.shape[0]])

    def test_row_batch_only_blocks_the_lookup_results_unchanged(self):
        idx = np.arange(self.rows.shape[0]) % 2
        small, big = self._comp(row_batch=2), self._comp(row_batch=64)
        ll_s = np.asarray(small.get_ll_wdm(self.rows, self.aca, data_index=idx, noise_index=idx))
        ll_b = np.asarray(big.get_ll_wdm(self.rows, self.aca, data_index=idx, noise_index=idx))
        np.testing.assert_allclose(ll_s, ll_b, rtol=1e-12, atol=1e-12 * np.abs(ll_b).max())
        self.assertGreater(np.abs(ll_b).max(), 0.0)
        nch, nfa, nta = 3, int(self.wdm.Nf_active), int(self.wdm.Nt_active)
        bufs = []
        for comp in (small, big):
            buf = np.zeros((2, nch, nfa, nta))
            comp.fill_global_wdm(
                self.rows, buf, data_index=idx, factors=np.arange(1.0, 1.0 + self.rows.shape[0])
            )
            bufs.append(buf)
        np.testing.assert_allclose(bufs[0], bufs[1], rtol=1e-12, atol=1e-14 * np.abs(bufs[1]).max())

    def test_spans_split_the_pn_from_the_response(self):
        comp = self._comp(row_batch=2)
        idx = np.zeros(self.rows.shape[0], dtype=int)
        comp.get_ll_wdm(self.rows, self.aca, data_index=idx, noise_index=idx)
        sp = comp.last_call_spans
        for key in ("pn", "response", "tracer", "lookup", "inner", "total"):
            self.assertIn(key, sp)
        self.assertGreater(sp["pn"], 0.0)
        self.assertLessEqual(sp["pn"], sp["response"])
        self.assertLessEqual(sp["response"] + sp["tracer"] + sp["lookup"], sp["total"])


if __name__ == "__main__":
    unittest.main()
