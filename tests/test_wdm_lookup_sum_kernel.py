"""The fused C++/CUDA lookup-sum kernel (``wdm_lookup_sum``) == the Python reference: the
analytic tracer (``tracer_from_tof_output``) + ``WDMLookupTable.get_wdm_coeffs`` + scatter-add
(``accumulate_harmonic_batch``), on

* a synthetic response whose (f, fdot) sweep the table's edges (fdot axis ends, f_norm support
  ends with 3 neighbour layers, the top of the grid, the f > 2 layer_df and |fdot| filters,
  negative-frequency subs);
* the dense TDI response on a sparse grid with the exact carrier (ExactPhaseTDIOutput, two
  templates in one call) and on a pixel grid with the carrier spline folded into the phase
  spline (SplinedTDIOutput);

plus the output band, the per-harmonic handoff pixel and GPU == CPU (skips without a GPU)."""
import tempfile
import unittest

import numpy as np

from tests._wdm_lookup_toy import build_tiny_table

DAY = 86400.0


def _backend(name="cpu"):
    import lisatools

    return lisatools.get_backend(name)


def _has_kernel():
    try:
        return _backend().wdm_lookup_sum is not None
    except Exception:
        return False


class _FakeResponse:
    """One cubic segment per (sub, channel) spanning the pixels: amp = a0, phase r(t) = y + c1 dx
    + c2 dx^2 + c3 dx^3 (dx = t - x0). Exposes ``kernel_inputs`` (carrier 0) and the reference
    ``tracer(t)``."""

    def __init__(self, x0, x1, amp, y, c1, c2, c3, xp=np):
        self.xp = xp
        self.S, self.nch = amp.shape
        self.x0, self.x1 = x0, x1
        self.amp, self.coef = amp, (y, c1, c2, c3)

    def kernel_inputs(self):
        S, nch, N = self.S, self.nch, 2
        x = np.tile(np.array([self.x0, self.x1]), S * nch)

        def flat(v):          # value at knot 0; knot 1 is never a segment start (seg clamps to 0)
            out = np.zeros((S * nch, N))
            out[:, 0] = v.reshape(-1)
            return out.reshape(-1)

        z = np.zeros((S, nch))
        xp = self.xp
        return dict(carrier=0, N=N, nch=nch, num_sub=S, x=xp.asarray(x),
                    amp=[xp.asarray(flat(v)) for v in (self.amp, z, z, z)],
                    res=[xp.asarray(flat(c)) for c in self.coef])

    def tracer(self, t):
        y, c1, c2, c3 = (c[:, :, None] for c in self.coef)
        dx = (t - self.x0)[None, None, :]
        ph = y + dx * (c1 + dx * (c2 + dx * c3))
        f = (c1 + dx * (2 * c2 + 3 * dx * c3)) / (2 * np.pi)
        fd = (2 * c2 + 6 * dx * c3) / (2 * np.pi)
        neg = f < 0
        amp = np.broadcast_to(self.amp[:, :, None], ph.shape).copy()
        return amp, np.where(neg, -ph, ph), np.where(neg, -f, f), np.where(neg, -fd, fd)


@unittest.skipUnless(_has_kernel(), "backend module built without wdm_lookup_sum")
class SyntheticEdgesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wdm, cls.table = build_tiny_table(cls.tmp.name, nt=96)
        cls.table.set_interp_method("spline")
        cls.ldt, cls.ldf = cls.wdm.layer_dt, cls.wdm.layer_df
        cls.fd_max = float(np.max(np.abs(np.asarray(cls.table.fdot_vals))))
        cls.n_ok = np.arange(6, cls.wdm.Nt - 6)
        cls.t0 = 1.0e6
        t = cls.t0 + cls.n_ok * cls.ldt
        # fdot per sub: both axis ends, just inside, inside, just outside (dropped), zero; f
        # sweeping from below 2 layers (dropped) through the grid to past its top; one sub
        # with negative frequency (mirrored)
        fdot = np.array([cls.fd_max, -cls.fd_max, 0.999 * cls.fd_max, 0.37 * cls.fd_max,
                         1.01 * cls.fd_max, 0.0, -0.2 * cls.fd_max])
        f_start = np.array([30.3, 50.7, 3.01, 10.5, 25.0, 61.2, 40.4]) * cls.ldf
        sign = np.array([1, 1, 1, 1, 1, 1, -1.0])
        S, nch = fdot.size, 2
        rng = np.random.default_rng(3)
        x0 = t[0] - 1800.0
        c1 = (2 * np.pi * sign * (f_start - fdot * 1800.0))[:, None] * np.ones((1, nch))
        c1 += 2 * np.pi * 1e-7 * rng.normal(size=(S, nch))          # channel Doppler offsets
        c2 = (np.pi * sign * fdot)[:, None] * np.ones((1, nch))
        c3 = 1e-20 * rng.normal(size=(S, nch))
        y = rng.uniform(0, 2 * np.pi, size=(S, nch))
        amp = rng.uniform(0.5, 1.5, size=(S, nch))
        cls.resp = _FakeResponse(x0, t[-1] + 1800.0, amp, y, c1, c2, c3)
        cls.t = t

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _python(self, num_m_layers, sub_row, n_rows, fdot_max=None, m_lo=0, n_m=None, amp_mask=None):
        from lisatools.sources.emri.wdm_direct import accumulate_harmonic_batch

        tracer = list(self.resp.tracer(self.t))
        if amp_mask is not None:
            tracer[0] = tracer[0] * amp_mask
        n_m = self.wdm.Nf - m_lo if n_m is None else n_m
        acc = np.zeros((n_rows, self.resp.nch, n_m, self.wdm.Nt))
        st = accumulate_harmonic_batch(
            acc, self.table, None, tuple(tracer), self.n_ok, None, Nf=self.wdm.Nf, Nt=self.wdm.Nt,
            dt=self.wdm.data_dt, layer_dt=self.ldt, layer_df=self.ldf, t0=self.t0, num_m_layers=num_m_layers,
            fdot_axis_max=self.fd_max if fdot_max is None else fdot_max, sub_row=sub_row, m_lo=m_lo)
        return acc, st

    def _kernel(self, num_m_layers, sub_row, n_rows, fdot_max=None, m_lo=0, n_m=None, n_stop=None,
                backend=None, resp=None, table=None):
        from lisatools.sources.emri.wdm_direct import lookup_sum_kernel

        backend = _backend() if backend is None else backend
        n_m = self.wdm.Nf - m_lo if n_m is None else n_m
        acc = backend.xp.zeros((n_rows, self.resp.nch, n_m, self.wdm.Nt))
        n_hi = int(self.n_ok[-1]) + 1
        st = lookup_sum_kernel(
            backend, self.resp if resp is None else resp, acc, self.table if table is None else table,
            sub_row=sub_row,
            n_stop=np.full(self.resp.S, n_hi) if n_stop is None else n_stop, n_lo=int(self.n_ok[0]), n_hi=n_hi,
            t0=self.t0, layer_dt=self.ldt, Nf=self.wdm.Nf, m_lo=m_lo, num_m_layers=num_m_layers,
            fdot_axis_max=self.fd_max if fdot_max is None else fdot_max, f_min=2 * self.ldf)
        return acc, st

    def _check(self, a, b, tol=1e-11):
        # phases reach ~2e4 rad here: cos/sin of the same phase summed in a different order
        # differ by ~1e-12 rad; the B-spline weights by a few ulp
        scale = np.abs(b).max()
        self.assertGreater(scale, 0)
        err = np.abs(a - b).max() / scale
        self.assertLess(err, tol, f"kernel vs python max rel {err:.2e}")
        return err

    def test_kernel_equals_python_with_table_edges(self):
        sub_row = np.array([0, 1, 0, 2, 1, 2, 0])
        for L in (2, 3):                      # 3: f_norm reaches both ends of the table support
            ref, st_ref = self._python(L, sub_row, 3)
            got, st = self._kernel(L, sub_row, 3)
            err = self._check(got, ref)
            print(f"[lookup kernel] synthetic, {2 * L + 1} layers: max rel {err:.1e}, "
                  f"lookup {st['lookup_pixels']} dropped {st['dropped_pixels']}")
            self.assertEqual((st["lookup_pixels"], st["dropped_pixels"]),
                             (st_ref["lookup_pixels"], st_ref["dropped_pixels"]))
            self.assertGreater(st["dropped_pixels"], 0)

    def test_both_reference_pixel_parities(self):
        """A table built at odd (m_ref + n_ref) bakes another rotation (get_wdm_coeffs maps
        (c, s) -> (-c, s) first); the EMRI recipe's m_ref is 21."""
        import os

        from lisatools.sources.emri.wdm_direct import table_kernel_view

        d = os.path.join(self.tmp.name, "odd")
        os.makedirs(d, exist_ok=True)
        _, odd = build_tiny_table(d, nt=96, m_ref=21)
        odd.set_interp_method("spline")
        parities = {table_kernel_view(t)["ref_odd"] for t in (self.table, odd)}
        self.assertEqual(parities, {0, 1})
        sub_row = np.array([0, 1, 0, 2, 1, 2, 0])
        saved = self.table
        try:
            type(self).table = odd
            ref, _ = self._python(2, sub_row, 3)
            got, _ = self._kernel(2, sub_row, 3)
        finally:
            type(self).table = saved
        self._check(got, ref)

    def test_fdot_filter(self):
        sub_row = np.zeros(7, int)
        ref, st_ref = self._python(2, sub_row, 1, fdot_max=0.5 * self.fd_max)
        got, st = self._kernel(2, sub_row, 1, fdot_max=0.5 * self.fd_max)
        self._check(got, ref)
        self.assertEqual(st["dropped_pixels"], st_ref["dropped_pixels"])

    def test_band(self):
        sub_row = np.array([0, 1, 0, 2, 1, 2, 0])
        full, _ = self._kernel(2, sub_row, 3)
        band, _ = self._kernel(2, sub_row, 3, m_lo=20, n_m=25)
        np.testing.assert_array_equal(band, full[:, :, 20:45])
        ref, _ = self._python(2, sub_row, 3, m_lo=20, n_m=25)
        self._check(band, ref)

    def test_handoff_pixel_stops_each_sub(self):
        sub_row = np.array([0, 1, 0, 2, 1, 2, 0])
        n_stop = np.array([40, 100, 20, 60, 6, 50, 33])
        got, _ = self._kernel(2, sub_row, 3, n_stop=n_stop)
        mask = (self.n_ok[None, None, :] < n_stop[:, None, None]).astype(float)
        ref, _ = self._python(2, sub_row, 3, amp_mask=mask)
        self._check(got, ref)

    def test_gpu_equals_cpu(self):
        """CUDA leads, the CPU build mirrors it: same sums to round-off (atomic order)."""
        from lisatools.domains import WDMLookupTable

        try:
            gpu = _backend("gpu")
        except Exception:
            self.skipTest("no GPU backend")
        if gpu.wdm_lookup_sum is None:
            self.skipTest("GPU module built without wdm_lookup_sum")
        import os

        table_g = WDMLookupTable.from_file(os.path.join(self.tmp.name, "lookup_nf64_dt56.25.h5"),
                                           force_backend=gpu.name.split("_")[-1])
        table_g.set_interp_method("spline")
        r = self.resp
        resp_g = _FakeResponse(r.x0, r.x1, r.amp, *r.coef, xp=gpu.xp)
        sub_row = np.array([0, 1, 0, 2, 1, 2, 0])
        for L in (2, 3):
            cpu, st_c = self._kernel(L, sub_row, 3)
            got, st_g = self._kernel(L, sub_row, 3, backend=gpu, resp=resp_g, table=table_g)
            err = self._check(got.get(), cpu, tol=1e-13)
            print(f"[lookup kernel] GPU vs CPU, {2 * L + 1} layers: max rel {err:.1e}")
            self.assertEqual(st_c, st_g)


@unittest.skipUnless(_has_kernel(), "backend module built without wdm_lookup_sum")
class DenseResponseKernelTest(unittest.TestCase):
    """Two templates (different sky, harmonics) through TDDenseTDIonTheFly on a sparse 12 h
    grid (exact carrier) and a 30 min grid (carrier folded into the phase spline)."""

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.response.tdionfly import TDDenseTDIonTheFly
        from lisatools.sources.emri.wdm_direct import (ExactPhaseTDIOutput, SplinedTDIOutput,
                                                       sparse_response_grid)
        from tests.test_emri_sparse_response import _chirp_coeffs, _quadratic_coeffs

        cls.tmp = tempfile.TemporaryDirectory()
        cls.wdm, cls.table = build_tiny_table(cls.tmp.name, nt=160)
        cls.table.set_interp_method("spline")
        orbits = EqualArmlengthOrbits(force_backend="cpu")
        tdi = TDIConfig("2nd generation", force_backend="cpu")
        t0 = 2.0e7
        t_k = [t0 + np.arange(13) * 12 * 3600.0, t0 + np.arange(25) * 6 * 3600.0]   # 6 days each
        C = [_quadratic_coeffs(t_k[0], np.array([0.3, 0.0, 1.1]), np.array([2.1e-3, 0.0, 0.47e-3]),
                               np.array([2e-10, 0.0, 3e-11])),
             _chirp_coeffs(t_k[1], 1.4e-3, 9 * DAY)]
        mkn = [np.array([[2, 0, 0], [3, 0, 1], [-2, 0, -1], [4, 0, 1]]), np.array([[2, 0, 0], [3, 0, 0], [-2, 0, 0]])]
        K = max(tk.size for tk in t_k)
        S = sum(m.shape[0] for m in mkn)
        tkp = np.stack([np.append(tk, np.full(K - tk.size, tk[-1])) for tk in t_k])
        Cp = np.zeros((2, K - 1, 3, 8))
        for b in range(2):
            Cp[b, :t_k[b].size - 1] = C[b]
        are = np.zeros((S, K - 1, 4))
        aim = np.zeros((S, K - 1, 4))
        are[:, :, 0] = np.linspace(0.6, 1.4, S)[:, None]
        aim[:, :, 0] = np.linspace(-0.3, 0.5, S)[:, None]
        are[:, :, 1] = 1e-7
        params = np.array([[0.7, 1.9, 4.0, -0.8], [0.2, 0.4, 1.0, 0.6]])
        args = (np.array([0, 4, S]), np.concatenate(mkn), tkp, np.array([tk.size for tk in t_k]), Cp, are, aim)
        kw = dict(amp_factor=0.5, tdi_config=tdi, orbits=orbits, force_backend="cpu")
        g = [sparse_response_grid(tk, tk[0], tk[-1], 43200.0, 600.0) for tk in t_k]
        N = max(x.size for x in g)
        from lisatools.sources.emri.wdm_direct import pad_grid

        dense = TDDenseTDIonTheFly(np.stack([pad_grid(x, N) for x in g]), *args, **kw)
        cls.exact = ExactPhaseTDIOutput(dense(params, return_spline=False), [tk for tk in t_k], C,
                                        np.concatenate(mkn), dense.sub_temp_host)
        tp = np.arange(t0 + 600.0, t0 + 6 * DAY - 599.0, 1800.0)
        dense_p = TDDenseTDIonTheFly(np.stack([tp, tp]), *args, **kw)
        cls.pixels = SplinedTDIOutput(dense_p(params, return_spline=False), np.concatenate(mkn),
                                      dense_p.sub_temp_host)
        cls.sub_row = dense.sub_temp_host.copy()
        cls.data_t0 = t0 - 3 * 3600.0
        ldt = cls.wdm.layer_dt
        n = np.arange(cls.wdm.Nt)
        tt = cls.data_t0 + n * ldt
        cls.n_ok = n[(tt > tp[0] + 1800.0) & (tt < tp[-1] - 1800.0) & (n >= 8) & (n < cls.wdm.Nt - 8)]
        cls.fd_max = float(np.max(np.abs(np.asarray(cls.table.fdot_vals))))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _both(self, resp):
        from lisatools.sources.emri.wdm_direct import (accumulate_harmonic_batch, lookup_sum_kernel,
                                                       tracer_from_tof_output)

        wdm = self.wdm
        t = self.data_t0 + self.n_ok * wdm.layer_dt
        tracer = tracer_from_tof_output(resp, t, method="analytic")
        shape = (2, 3, wdm.Nf, wdm.Nt)
        ref = np.zeros(shape)
        st_ref = accumulate_harmonic_batch(
            ref, self.table, None, tracer, self.n_ok, None, Nf=wdm.Nf, Nt=wdm.Nt, dt=wdm.data_dt,
            layer_dt=wdm.layer_dt, layer_df=wdm.layer_df, t0=self.data_t0, fdot_axis_max=self.fd_max,
            sub_row=self.sub_row)
        got = np.zeros(shape)
        st = lookup_sum_kernel(
            _backend(), resp, got, self.table, sub_row=self.sub_row, n_stop=np.full(self.sub_row.size, 10 ** 6),
            n_lo=int(self.n_ok[0]), n_hi=int(self.n_ok[-1]) + 1, t0=self.data_t0, layer_dt=wdm.layer_dt,
            Nf=wdm.Nf, m_lo=0, num_m_layers=2, fdot_axis_max=self.fd_max, f_min=2 * wdm.layer_df)
        return got, ref, st, st_ref

    def test_exact_carrier_two_templates(self):
        got, ref, st, st_ref = self._both(self.exact)
        err = np.abs(got - ref).max() / np.abs(ref).max()
        print(f"[lookup kernel] sparse grid + exact carrier, 2 templates: max rel {err:.1e}, "
              f"lookup {st['lookup_pixels']}")
        self.assertLess(err, 1e-11)
        self.assertEqual(st["lookup_pixels"], st_ref["lookup_pixels"])
        self.assertTrue(np.all(np.abs(ref).reshape(2, -1).max(axis=1) > 0))      # both rows lit

    def test_pixel_grid_folded_carrier(self):
        got, ref, st, st_ref = self._both(self.pixels)
        err = np.abs(got - ref).max() / np.abs(ref).max()
        print(f"[lookup kernel] pixel grid (carrier folded): max rel {err:.1e}")
        self.assertLess(err, 1e-11)
        self.assertEqual(st["lookup_pixels"], st_ref["lookup_pixels"])


@unittest.skipUnless(_has_kernel(), "backend module built without wdm_lookup_sum")
class KernelSelectionTest(unittest.TestCase):
    """EMRIDirectWDM runs the fused kernel whenever it can (lookup="kernel", the default) and
    falls back to the Python lookup, warned once, when the table or the response cannot."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wdm, cls.table = build_tiny_table(cls.tmp.name, nt=64)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _direct(self, interp="spline", **kw):
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

        return EMRIDirectWDM(None, self.table, self.wdm, orbits=None, tdi_config=None, t_start=0.0,
                             data_t0=0.0, interp=interp, **kw)

    def test_default_is_the_kernel(self):
        import os
        from unittest import mock

        resp = _FakeResponse(0.0, 1.0, np.ones((1, 1)), *([np.zeros((1, 1))] * 4))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMRI_DIRECT_LOOKUP", None)
            d = self._direct()
        self.assertEqual(d.lookup, "kernel")
        self.assertIsNotNone(d._kernel_backend(resp))
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_LOOKUP": "python"}):
            self.assertIsNone(self._direct()._kernel_backend(resp))
        with self.assertRaises(ValueError):
            self._direct(lookup="fortran")

    def test_falls_back_once_when_it_cannot_run(self):
        import warnings

        resp = _FakeResponse(0.0, 1.0, np.ones((1, 1)), *([np.zeros((1, 1))] * 4))
        d = self._direct()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertIsNone(d._kernel_backend(object()))              # no kernel inputs
            self.assertIsNone(d._kernel_backend(object()))
        self.assertEqual(len(w), 1)
        self.assertIn("response output", str(w[0].message))
        try:
            d = self._direct(interp="linear")                            # table not splined
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.assertIsNone(d._kernel_backend(resp))
            self.assertIn("spline-interpolated", str(w[0].message))
        finally:
            self.table.set_interp_method("spline")
