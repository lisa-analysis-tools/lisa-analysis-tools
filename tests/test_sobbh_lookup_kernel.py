# tests/test_sobbh_lookup_kernel.py
"""The fused SOBBH lookup kernel (sobbh_lookup_kernel.cu) == the Python lookup path.

Reference: ``SOBBHDirectWDM.sparse_from`` + ``sparse_inner_products`` (scoring) and
``scatter_add`` (fill) with the table's B-spline (``interp="spline"``), which
``tests/test_wdm_lookup_eval.py`` pins to ``scipy.ndimage.map_coordinates``. The kernel
evaluates the same response splines at the pixel centres, the same channel-mean layer window,
the same B-spline (``wdm_lookup_kernels.hh``, shared with the EMRI ``wdm_lookup_sum``) and the
same quarter turn, and sums in a different order: agreement to roundoff. Paired control: the
Python path without the quarter turn differs at O(1).
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table  # noqa: E402
from test_sobbh_wdm_direct import DT, NF, NT, REF, ROW_MERGE, ROWS, T_START  # noqa: E402

#: a chirpier in-band row (heavy pair at 15 mHz) on the toy grid's 2-20 mHz band
ROW_CHIRPY = np.array([60.0, 55.0, 0.1, 0.2, 0.8e9, 1.5e-2, 1.1, 1.2, 0.7, 3.1, 0.2])


def _kernel_available(backend="cpu"):
    import lisatools

    return getattr(lisatools.get_backend(backend), "sobbh_lookup", None) is not None


def _gpu_backend():
    import lisatools

    try:
        lisatools.get_backend("gpu")
    except Exception:
        return None
    return "gpu"


@unittest.skipUnless(_kernel_available(), "the CPU backend module has no sobbh_lookup (rebuild)")
class KernelParityTest(unittest.TestCase):
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
        cls.rows = np.vstack([ROWS, ROW_CHIRPY[None, :], ROW_MERGE[None, :]])
        cls.nr = cls.rows.shape[0]
        comp = cls._comp("python")
        nch, nfa, nta = 3, int(cls.wdm.Nf_active), int(cls.wdm.Nt_active)
        data = np.zeros((2, nch, nfa, nta))
        comp.fill_global_wdm(
            ROWS[:2], data, data_index=np.array([0, 1]), factors=np.array([1.0, -0.5])
        )
        cls.aca = AnalysisContainerArray(
            [
                AnalysisContainer(
                    WDMSignal(np.ascontiguousarray(data[k]), cls.wdm),
                    XYZ2SensitivityMatrix(cls.wdm, model="scirdv1" if k == 0 else "sangria"),
                )
                for k in range(2)
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def _comp(cls, kernel, interp="spline", force_backend="cpu", row_batch=3):
        from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

        return SOBBHLookupComputations(
            cls.wdm,
            REF,
            cls.table,
            orbits=cls.orbits,
            tdi_config="2nd generation",
            tdi_type="XYZ",
            n_grid=1024,
            interp=interp,
            row_batch=row_batch,
            kernel=kernel,
            force_backend=force_backend,
        )

    def _ll(self, comp, di, ni):
        ll = np.asarray(comp.get_ll_wdm(self.rows, self.aca, data_index=di, noise_index=ni))
        return ll, np.asarray(comp.d_h_out), np.asarray(comp.h_h_out)

    def test_scoring_matches_python_xyz(self):
        di = np.arange(self.nr) % 2
        ni = (np.arange(self.nr) + 1) % 2  # swapped noise routing: data slab != noise slab
        k = self._comp("kernel")
        self.assertTrue(k.uses_kernel)
        ll_k, dh_k, hh_k = self._ll(k, di, ni)
        ll_p, dh_p, hh_p = self._ll(self._comp("python"), di, ni)
        # each quantity against its OWN scale (the chirpy row's <h|h> dwarfs every <d|h>)
        for got, ref in ((hh_k, hh_p), (dh_k, dh_p), (ll_k, ll_p)):
            scale = np.abs(ref).max()
            self.assertGreater(scale, 0.0)
            np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12 * scale)

    def test_routing_is_per_row(self):
        k = self._comp("kernel")
        a = self._ll(k, np.zeros(self.nr, int), np.zeros(self.nr, int))[0]
        b = self._ll(k, np.ones(self.nr, int), np.ones(self.nr, int))[0]
        self.assertGreater(np.abs(a - b).max(), 1e-6 * np.abs(a).max())

    def test_stats_match_python(self):
        k, p = self._comp("kernel"), self._comp("python")
        di = np.zeros(self.nr, int)
        k.get_ll_wdm(self.rows, self.aca, data_index=di, noise_index=di)
        p.get_ll_wdm(self.rows, self.aca, data_index=di, noise_index=di)
        self.assertEqual(k.last_stats, p.last_stats)
        self.assertGreaterEqual(p.last_stats["merged_rows"], 1)  # ROW_MERGE merges in the window

    def test_diagonal_noise_matches_python(self):
        # AET / AE: the diagonal invC layout (nch, Nfa, Nta) per slab; random slabs so every
        # term is exercised
        from lisatools.sources.sobbh.wdm_direct import sparse_inner_products

        direct = self._comp("kernel").direct
        out = direct.response(self.rows)
        tpl = direct.sparse_from(out)
        nch, nfa, nta = 3, int(self.wdm.Nf_active), int(self.wdm.Nt_active)
        rng = np.random.default_rng(7)
        data = rng.normal(size=2 * nch * nfa * nta) * 1e-20
        inv = rng.uniform(0.5, 2.0, size=2 * nch * nfa * nta) * 1e40
        di = np.arange(self.nr) % 2
        ni = 1 - di
        ref = sparse_inner_products(
            tpl, data, inv, di, ni, nchannels=nch, Nf_active=nfa, Nt_active=nta, tdi_type="AET"
        )
        got = direct.kernel_inner_products(out, data, inv, di, ni, tdi_type="AET")
        for a, b in zip(got[:2], ref):
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12 * np.abs(ref[1]).max()
            )

    def test_fill_matches_python(self):
        nch, nfa, nta = 3, int(self.wdm.Nf_active), int(self.wdm.Nt_active)
        di = np.arange(self.nr) % 2
        fac = np.linspace(-1.5, 2.0, self.nr)
        bufs = []
        for kind in ("kernel", "python"):
            buf = np.zeros((2, nch, nfa, nta))
            self._comp(kind).fill_global_wdm(self.rows, buf, data_index=di, factors=fac)
            bufs.append(buf)
        scale = np.abs(bufs[1]).max()
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(bufs[0], bufs[1], rtol=1e-10, atol=1e-12 * scale)

    def test_control_without_the_quarter_turn_differs(self):
        di = np.zeros(self.nr, int)
        hh_k = self._ll(self._comp("kernel"), di, di)[2]
        p = self._comp("python")
        p.direct.ev.basis_cycle = "no_parity_turn"
        hh_c = self._ll(p, di, di)[2]
        self.assertGreater(np.abs(hh_c - hh_k).max(), 1e-2 * np.abs(hh_k).max())

    def test_kernel_needs_the_bspline_table(self):
        with self.assertRaises(ValueError):
            self._comp("kernel", interp="cubic")
        self.assertFalse(self._comp("auto", interp="cubic").uses_kernel)
        self.assertTrue(self._comp("auto").uses_kernel)
        self.assertFalse(self._comp("python").uses_kernel)
        with self.assertRaises(ValueError):
            self._comp("fused")


class OrbitCoverageTest(unittest.TestCase):
    """A pixel window that runs past the orbit tables (the mojito L1 bricks carry orbits for
    ~449 days from the data start): the response exists only inside the tables, so pixels
    outside are a ZERO template on both lookup paths (the C++ response zeroes them too), never
    an extrapolated spline or an out-of-bounds error."""

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits

        cls.tmp = tempfile.TemporaryDirectory()
        _, cls.table = build_tiny_table(cls.tmp.name)
        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")
        cls.t_end = float(cls.orbits.t_base[-1])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _direct(self, kernel):
        from lisatools.domains import WDMSettings
        from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

        # the window starts 4 days before the orbit tables end and lasts 10.7 days
        t0 = self.t_end - 4 * 86400.0
        wdm = WDMSettings(NF, NT, DT, t0=t0, min_freq=2e-3, max_freq=2e-2, force_backend="cpu")
        comp = SOBBHLookupComputations(
            wdm,
            t0,
            self.table,
            orbits=self.orbits,
            tdi_config="2nd generation",
            tdi_type="XYZ",
            n_grid=1024,
            kernel=kernel,
            force_backend="cpu",
        )
        return comp

    def test_pixels_past_the_orbits_are_zero_on_both_paths(self):
        nch = 3
        bufs = {}
        for kind in ("python",) + (("kernel",) if _kernel_available() else ()):
            comp = self._direct(kind)
            ws = comp.wdm_settings
            buf = np.zeros(nch * int(ws.Nf_active) * int(ws.Nt_active))
            comp.fill_global_wdm(ROWS[:2], buf, data_index=np.zeros(2, dtype=int))
            bufs[kind] = buf.reshape(nch, int(ws.Nf_active), int(ws.Nt_active))
        ref = bufs["python"]
        t_pix = comp.direct.t_pixels
        past = t_pix > self.t_end
        self.assertTrue(past.any() and (~past).any())
        self.assertGreater(np.abs(ref[..., ~past]).max(), 0.0)  # signal while the orbits last
        self.assertEqual(np.abs(ref[..., past]).max(), 0.0)  # nothing past them
        if "kernel" in bufs:
            # atol 1e-11 of the peak: a few pixels sit 1e-2 below the peak with 1e-10 relative
            # roundoff (summation order)
            np.testing.assert_allclose(
                bufs["kernel"], ref, rtol=1e-10, atol=1e-11 * np.abs(ref).max()
            )


@unittest.skipUnless(_kernel_available(), "the CPU backend module has no sobbh_lookup (rebuild)")
class GPUKernelParityTest(unittest.TestCase):
    """The CUDA build of the fused kernel == its CPU build (same source, sobbh_lookup_kernel.cu;
    only the summation order differs). The 6-month launcher's SOBBH preflight runs this on the
    node's GPU before mpiexec when the lookup scores through the kernel."""

    def _direct(self, backend):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.domains import WDMSettings
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.sources.sobbh.wdm_direct import SOBBHDirectWDM

        wdm = WDMSettings(NF, NT, DT, t0=T_START, min_freq=2e-3, max_freq=2e-2, force_backend="cpu")
        return SOBBHDirectWDM(
            wdm,
            self.table,
            orbits=EqualArmlengthOrbits(force_backend=backend),
            tdi_config=TDIConfig("2nd generation", force_backend=backend),
            reference_time=REF,
            n_grid=1024,
            force_backend=backend,
        )

    def test_gpu_equals_cpu(self):
        gpu = _gpu_backend()
        if gpu is None:
            self.skipTest("no GPU backend")
        if not _kernel_available(gpu):
            self.fail("the GPU backend module has no sobbh_lookup: rebuild lisatools")
        import cupy as cp

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        _, self.table = build_tiny_table(self.tmp.name)
        rows = np.vstack([ROWS, ROW_CHIRPY[None, :], ROW_MERGE[None, :]])
        dc, dg = self._direct("cpu"), self._direct(gpu)
        oc, og = dc.response(rows), dg.response(rows)
        nch, nfa, nta = 3, int(dc.wdm.Nf_active), int(dc.wdm.Nt_active)
        rng = np.random.default_rng(11)
        data = rng.normal(size=2 * nch * nfa * nta) * 1e-20
        inv = rng.uniform(0.5, 2.0, size=2 * nch * nch * nfa * nta) * 1e40
        di = np.arange(rows.shape[0]) % 2
        ni = 1 - di
        hc = dc.kernel_inner_products(oc, data, inv, di, ni, tdi_type="XYZ")
        hg = dg.kernel_inner_products(og, cp.asarray(data), cp.asarray(inv), di, ni, tdi_type="XYZ")
        for a, b in zip(hg[:2], hc[:2]):
            b = np.asarray(b)
            np.testing.assert_allclose(cp.asnumpy(a), b, rtol=1e-9, atol=1e-12 * np.abs(b).max())
        self.assertEqual(hg[2], hc[2])  # the stats (lookup pixels, dropped, merged rows)
        fac = np.linspace(-1.0, 2.0, rows.shape[0])
        bc = np.zeros(2 * nch * nfa * nta)
        bg = cp.zeros(2 * nch * nfa * nta)
        dc.kernel_fill(oc, bc, di, fac)
        dg.kernel_fill(og, bg, di, fac)
        np.testing.assert_allclose(cp.asnumpy(bg), bc, rtol=1e-9, atol=1e-12 * np.abs(bc).max())


if __name__ == "__main__":
    unittest.main()
