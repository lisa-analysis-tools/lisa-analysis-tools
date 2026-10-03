# tests/test_mbh_windowed_signal_gen.py
"""MBH windowed sub-transform adapter: kept layers equal the full-grid
transform to a pinned tolerance, the pad is load-bearing, rows stack.
Stub generator (no phentax) so the transform is what is under test; the
real-phentax classes are named ``...Phentax...``.

The MBH twin of the EMRI pair (``python -m unittest tests.test_wdm_lookup_sum_kernel
tests.test_tdi_dense -v``): the generator / sub-transform half. Run on a GPU node::

    python -m unittest tests.test_mbh_windowed_signal_gen tests.test_mbh_batched_move \\
        tests.test_mbh_harness_noise -v

GPU == CPU (skip without a GPU backend): ``WindowedSignalGenGPUParityTest`` (stub
generator: the segment placement + WDM transform, cupy vs numpy) and
``WindowedGridAlignedPhentaxGPUParityTest`` (real phentax: the
WindowedGridAlignedMBHWaveform TDI channels and the adapter's kept layers on the
CUDA vs the CPU build of the response). Time on a GPU node is the phentax JIT
(~25 s per batch shape on an H100, job 677): the ``...Phentax...`` classes;
everything else is stub-based and runs in seconds. Laptop: never run the
``...Phentax...`` classes casually (WindowedGridAlignedPhentaxTest is ~6.5 min of
un-jitted CPU phentax); name the stub classes instead."""
from __future__ import annotations

import unittest

import numpy as np

from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.utils.utility import asnumpy
from lisatools.utils.utility import tukey as lat_tukey

NF, NT, DT = 32, 128, 10.0
N = NF * NT


def _chirp(lo_layer, hi_layer):
    from scipy.signal.windows import tukey

    t = np.arange(N) * DT
    x = np.zeros(N)
    lo, hi = lo_layer * NF, hi_layer * NF
    tt = t[lo:hi]
    x[lo:hi] = tukey(hi - lo, alpha=0.3) * np.sin(
        2 * np.pi * (4e-3 * tt + 0.5 * 1.6e-2 * (tt - tt[0]) ** 2 / (tt[-1] - tt[0]))
    )
    return np.stack([x, 0.5 * x, 0.25 * x])


class _SegmentGen:
    """Stub generator returning (times, channels) of a TD signal on the data
    lattice: scalar ``amp`` -> ``(N,)`` / ``(3, N)``; array -> ``(B, N)`` /
    ``(B, 3, N)``, the ``compute_tdi_channels`` contract."""

    supports_batch = True

    def __init__(self, td_full):
        self.td_full = td_full
        self.n_calls = 0

    def compute_tdi_channels(self, amp, **kwargs):
        self.n_calls += 1
        t = np.arange(N) * DT
        if np.ndim(amp) == 0:
            return t, float(amp) * self.td_full
        a = np.asarray(amp, dtype=float)
        return np.broadcast_to(t, (a.size, N)).copy(), a[:, None, None] * self.td_full[None]


def _wdm():
    return WDMSettings(NF, NT, DT, force_backend="cpu")


def _gpu_backend_name():
    """The concrete GPU backend (``"cuda12x"`` / ``"cuda13x"``), or None without one."""
    import lisatools

    try:
        return lisatools.get_backend("gpu").name.split("_")[-1]
    except Exception:
        return None


class WindowedSignalGenTest(unittest.TestCase):
    def setUp(self):
        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        self.full = _wdm()
        self.h_td = _chirp(45, 75)
        self.gen = _SegmentGen(self.h_td)
        self.sg = MBHWindowedWDMSignalGen(self.gen, self.full, nchannels=3, tukey_alpha=0.0)
        self.ref = TDSignal(self.h_td, TDSettings(N, DT, force_backend="cpu")).transform(self.full).arr
        self.scale = float(np.abs(self.ref).max())

    def _err(self, out):
        return float(np.abs(np.asarray(out.arr) - self.ref[..., 40:80]).max() / self.scale)

    def test_kept_layers_match_full_transform(self):
        self.sg.set_window(n_start=40, Nt_keep=40, n_pad=8)
        out = self.sg(1.0)
        self.assertEqual(out.arr.shape, (3, NF, 40))
        self.assertEqual((out.ind_min_t, out.ind_max_t), (40, 79))
        self.assertEqual(self.sg.window_key, (40, 40, 8, 8))
        # measured 2026-09-29 spike: 1.2e-5 at 8 pad layers on this grid
        self.assertLess(self._err(out), 5e-5)

    def test_pad_is_load_bearing(self):
        self.sg.set_window(n_start=40, Nt_keep=40, n_pad=0)
        # measured: 2e-2 with no pad -- the discarded edge layers carry the error
        self.assertGreater(self._err(self.sg(1.0)), 1e-3)

    def test_batched_rows_stack_and_scale(self):
        self.sg.set_window(n_start=40, Nt_keep=40, n_pad=8)
        out = self.sg(np.array([1.0, 2.0, 0.5]))
        self.assertTrue(out.is_batched)
        self.assertEqual(out.arr.shape, (3, 3, NF, 40))
        np.testing.assert_allclose(out.arr[1], 2.0 * out.arr[0], rtol=1e-12, atol=0)
        self.assertEqual(self.gen.n_calls, 1)

    def test_single_row_batch_keeps_leading_axis(self):
        self.sg.set_window(n_start=40, Nt_keep=40, n_pad=8)
        out = self.sg(np.array([1.0]))
        self.assertTrue(out.is_batched)
        self.assertEqual(out.arr.shape, (1, 3, NF, 40))

    def test_window_must_be_set(self):
        with self.assertRaises(RuntimeError):
            self.sg(1.0)

    def test_window_outside_grid_raises(self):
        with self.assertRaises(ValueError):
            self.sg.set_window(n_start=2, Nt_keep=40, n_pad=8)
        with self.assertRaises(ValueError):
            self.sg.set_window(n_start=100, Nt_keep=40, n_pad=8)

    def test_odd_segment_is_made_even(self):
        self.sg.set_window(n_start=40, Nt_keep=41, n_pad=8)
        g = self.sg.geometry
        self.assertEqual(g["Nt_seg"] % 2, 0)
        self.assertEqual(g["Nt_keep"], 41)
        self.assertEqual(self.sg(1.0).arr.shape, (3, NF, 41))

    def test_odd_segment_start_layer_matches_full_transform(self):
        """The real WDM basis alternates with the parity of the ABSOLUTE layer
        index, so a segment starting on an odd layer transforms in the wrong
        phase convention unless the segment start is moved to an even layer.
        Measured 2026-09-30 before the fix: err 1.36 at s0 = 33 (n_start=41,
        n_pad=8) and at s0 = 31 (n_start=40, n_pad=9), vs 2.9e-5 at s0 = 32."""
        for n_start, n_pad in ((41, 8), (40, 9)):
            self.sg.set_window(n_start=n_start, Nt_keep=34, n_pad=n_pad)
            self.assertEqual(self.sg.geometry["s0"] % 2, 0)
            out = self.sg(1.0)
            self.assertEqual(out.arr.shape, (3, NF, 34))
            err = float(
                np.abs(np.asarray(out.arr) - self.ref[..., n_start:n_start + 34]).max() / self.scale
            )
            self.assertLess(err, 5e-5)

    def test_forwards_set_window_to_generator(self):
        calls = []

        class _Gen(_SegmentGen):
            def set_window(self, t_seg_abs, n_seg):
                calls.append((t_seg_abs, n_seg))

        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        sg = MBHWindowedWDMSignalGen(_Gen(self.h_td), self.full, nchannels=3)
        sg.set_window(n_start=40, Nt_keep=40, n_pad=8)
        self.assertEqual(calls, [(32 * NF * DT, NF * 56)])
        self.assertEqual(sg.t0_abs, 0.0)   # default: the settings' t0

    def test_segment_start_uses_t0_abs_not_settings_t0(self):
        """Stock erebor build: the settings carry t0 = 0 (or the data start,
        build-order dependent) while the generator's times are absolute. The
        segment must start at ``t0_abs + s0 * layer_dt``."""
        calls = []

        class _Gen(_SegmentGen):
            def set_window(self, t_seg_abs, n_seg):
                calls.append((t_seg_abs, n_seg))

        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        epoch = 9.7e7
        sg = MBHWindowedWDMSignalGen(_Gen(self.h_td), self.full, nchannels=3, t0_abs=epoch)
        sg.set_window(n_start=40, Nt_keep=40, n_pad=8)
        self.assertEqual(self.full.t0, 0.0)
        self.assertEqual(calls, [(epoch + 32 * NF * DT, NF * 56)])
        self.assertEqual(sg.geometry["t_seg"], epoch + 32 * NF * DT)
        self.assertEqual(float(sg._seg_td.t0), epoch + 32 * NF * DT)

    def test_tukey_window_kept_layers_match_full_transform(self):
        """tukey_alpha > 0: kept layers must equal the full-grid transform of
        the SAME signal with the full-length data window passed as
        ``transform(..., window=...)`` -- the convention
        ``TDWaveformBase._td_to_output_domain`` uses (waveformbase.py
        ~L550-557: ``window = tukey(full_grid.N, alpha=self.tukey_alpha,
        xp=self.xp); padded_td_signal.transform(self.output_domain_settings,
        window=window)``), i.e. the window multiplies inside the transform
        rather than being pre-multiplied into the TD signal.

        The chirp (layers 3..40) is placed to overlap the tukey_alpha=0.3
        onset ramp (layers 0..~19 of this NT=128 grid), and the kept box
        (layers 8..48) is placed to overlap that ramp too, with the segment
        (n_pad=8) still fully containing the chirp so the pad's low-error
        guarantee from test_kept_layers_match_full_transform still applies.
        This also means the alpha=0.3 and alpha=0 outputs must diverge well
        past the 5e-5 tolerance over that same kept box -- pinning that the
        ``tukey_alpha > 0`` branch in set_window is load-bearing (deleting it
        would leave both outputs identical)."""
        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        h_edge = _chirp(3, 40)
        n_start, Nt_keep, n_pad = 8, 40, 8

        full_window = lat_tukey(N, 0.3, xp=np)
        ref = (
            TDSignal(h_edge, TDSettings(N, DT, force_backend="cpu"))
            .transform(self.full, window=full_window)
            .arr
        )
        kept_ref = ref[..., n_start : n_start + Nt_keep]
        scale = float(np.abs(kept_ref).max())

        sg_windowed = MBHWindowedWDMSignalGen(
            _SegmentGen(h_edge), self.full, nchannels=3, tukey_alpha=0.3
        )
        sg_windowed.set_window(n_start=n_start, Nt_keep=Nt_keep, n_pad=n_pad)
        out_windowed = sg_windowed(1.0)
        err = float(np.abs(np.asarray(out_windowed.arr) - kept_ref).max() / scale)
        # measured 2026-09-29: 2.1e-5 (same order as the alpha=0 case).
        self.assertLess(err, 5e-5)

        sg_flat = MBHWindowedWDMSignalGen(
            _SegmentGen(h_edge), self.full, nchannels=3, tukey_alpha=0.0
        )
        sg_flat.set_window(n_start=n_start, Nt_keep=Nt_keep, n_pad=n_pad)
        out_flat = sg_flat(1.0)
        diff = float(
            np.abs(np.asarray(out_windowed.arr) - np.asarray(out_flat.arr)).max() / scale
        )
        # measured 2026-09-29: 0.45 -- the taper is ~63% attenuation at the
        # kept box's first layer, nowhere near the 5e-5 floor.
        self.assertGreater(diff, 5e-5)


class _XpSegmentGen(_SegmentGen):
    """:class:`_SegmentGen` whose output lives on the GPU when ``gpu``: the adapter's
    array module follows its input, as with the real generator's cupy output."""

    def __init__(self, td_full, gpu):
        super().__init__(td_full)
        self.gpu = bool(gpu)

    def compute_tdi_channels(self, amp, **kwargs):
        t, ch = super().compute_tdi_channels(amp, **kwargs)
        if not self.gpu:
            return t, ch
        import cupy as cp

        return cp.asarray(t), cp.asarray(ch)


class WindowedSignalGenGPUParityTest(unittest.TestCase):
    """GPU build of the windowed sub-transform == its CPU build (skips without a GPU).

    The same stub signal and window, three rows in one call, on WDMSettings of each
    backend: the segment placement, the data-window slice and the segment WDM
    transform (cupy + cuFFT vs numpy + pocketfft) are the only differences.

    TOL = 1e-11 of the kept layers' maximum. Both are float64 FFT pipelines whose
    rounding is ~eps * log2(n) ~ 1e-15 of the transform's norm (n = 1792-sample
    segment), <~1e-13 of the largest coefficient here: the bound leaves two orders over
    that floor and sits six orders below the pad-truncation error the CPU tests pin
    (1.2e-5), while the defects it exists to catch -- an odd-parity segment start
    (measured 1.36), a mis-sliced data window, a mis-placed segment -- are O(1e-2..1).
    """

    TOL = 1e-11
    CASES = (  # name, chirp layers, tukey alpha, n_start, Nt_keep, n_pad
        ("interior", (45, 75), 0.0, 40, 40, 8),
        ("odd start layer", (45, 75), 0.0, 41, 34, 8),
        ("tukey 0.3 over the onset ramp", (3, 40), 0.3, 8, 40, 8),
    )

    @classmethod
    def setUpClass(cls):
        cls.gpu = _gpu_backend_name()
        if cls.gpu is None:
            raise unittest.SkipTest("no GPU backend")

    def _kept(self, backend, h_td, alpha, n_start, Nt_keep, n_pad):
        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        wdm = WDMSettings(NF, NT, DT, force_backend=backend)
        sg = MBHWindowedWDMSignalGen(_XpSegmentGen(h_td, backend != "cpu"), wdm, nchannels=3,
                                     tukey_alpha=alpha)
        sg.set_window(n_start=n_start, Nt_keep=Nt_keep, n_pad=n_pad)
        out = sg(np.array([1.0, 2.0, 0.5]))
        return np.asarray(asnumpy(out.arr)), dict(sg.geometry), (int(out.ind_min_t), int(out.ind_max_t))

    def test_gpu_equals_cpu(self):
        for name, layers, alpha, n_start, Nt_keep, n_pad in self.CASES:
            with self.subTest(name):
                h = _chirp(*layers)
                c, geo_c, box_c = self._kept("cpu", h, alpha, n_start, Nt_keep, n_pad)
                g, geo_g, box_g = self._kept(self.gpu, h, alpha, n_start, Nt_keep, n_pad)
                self.assertEqual(g.shape, (3, 3, NF, Nt_keep))
                self.assertEqual((geo_g, box_g), (geo_c, box_c))
                scale = float(np.abs(c).max())
                self.assertGreater(scale, 0.0)
                err = float(np.abs(g - c).max()) / scale
                print(f"[windowed GPU vs CPU, {name}] kept-layer max rel diff {err:.2e}")
                self.assertLess(err, self.TOL)


class WindowedEdgeClampTest(unittest.TestCase):
    """Edge-clamped windows (``mbh_window_layers`` near the first/last layers):
    the segment stops AT the grid edge (0 pad on that side, 2 x pad on the
    other) and the kept box reaches the edge. The kept layers must still
    match the full-grid transform of the same signal. Both transforms are
    periodic -- the full grid wraps its last layer onto its first, the
    segment wraps onto the segment's far end -- and in both cases the wrap
    lands on zero signal (the far end is pad, the template's onset is inside
    the box), so the edge layers agree to the interior tolerance."""

    WINDOW = dict(window_before=10 * NF * DT, window_after=2 * NF * DT,
                  window_pad=8 * NF * DT, window_margin=1 * NF * DT)

    def _check(self, h_td, t_merge):
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers
        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        full = _wdm()
        g = mbh_window_layers(full, t_merge, **self.WINDOW)
        sg = MBHWindowedWDMSignalGen(_SegmentGen(h_td), full, nchannels=3, tukey_alpha=0.0)
        sg.set_window(g["n_start"], g["Nt_keep"], g["n_pad_lo"], g["n_pad_hi"])
        ref = TDSignal(h_td, TDSettings(N, DT, force_backend="cpu")).transform(full).arr
        kept = ref[..., g["n_start"]: g["n_start"] + g["Nt_keep"]]
        err_t = np.abs(np.asarray(sg(1.0).arr) - kept).max(axis=(0, 1)) / float(np.abs(ref).max())
        return g, sg.geometry, err_t

    def test_high_edge_kept_layers_match_full_transform(self):
        # chirp over layers 112..128: its end IS the grid end
        g, a, err_t = self._check(_chirp(112, 128), 126 * NF * DT)
        self.assertEqual(g["n_start"] + g["Nt_keep"], NT)
        self.assertEqual((a["n_pad_hi"], a["s0"] % 2), (0, 0))
        print(f"[edge clamp hi] kept-layer err: last layer {err_t[-1]:.2e}, max {err_t.max():.2e}")
        # measured 2026-09-30: last layer 3.3e-5, max 3.4e-5 (interior 8-pad
        # window: 1.2e-5); bound 1e-4 = 3x margin
        self.assertLess(err_t.max(), 1e-4)

    def test_low_edge_kept_layers_match_full_transform(self):
        g, a, err_t = self._check(_chirp(0, 12), 10 * NF * DT)
        self.assertEqual((g["n_start"], a["s0"], a["n_pad_lo"]), (0, 0, 0))
        print(f"[edge clamp lo] kept-layer err: first layer {err_t[0]:.2e}, max {err_t.max():.2e}")
        # measured 2026-09-30: first layer 2.0e-5, max 3.0e-5
        self.assertLess(err_t.max(), 1e-4)


class WindowedWholeGridClampTest(unittest.TestCase):
    """A data span SHORTER than the window (2026-09-30: the batched path is the
    default, so 3-month runs and the lite smokes use it): ``mbh_window_layers``
    clamps the kept box to the active box and the segment to the whole grid
    with zero pads. That segment transform IS the stock full-grid transform,
    so the kept layers must equal it at float rounding -- not merely at the
    ~1e-5 windowed (pad-truncation) tolerance above."""

    HUGE = dict(window_before=200 * NF * DT, window_after=10 * NF * DT,
                window_pad=8 * NF * DT, window_margin=1 * NF * DT)

    def setUp(self):
        import lisatools.globalfit.moves.mbhbatchedmove as mod

        mod._CLAMP_LOGGED.clear()
        self.addCleanup(mod._CLAMP_LOGGED.clear)

    def _run(self, run_wdm, h_td, alpha):
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers
        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        g = mbh_window_layers(run_wdm, 64 * NF * DT, **self.HUGE)
        sg = MBHWindowedWDMSignalGen(_SegmentGen(h_td), run_wdm, nchannels=3, tukey_alpha=alpha)
        sg.set_window(g["n_start"], g["Nt_keep"], g["n_pad_lo"], g["n_pad_hi"])
        full = _wdm()   # the uncropped grid: kept layers are ABSOLUTE layers
        window = lat_tukey(N, alpha, xp=np) if alpha > 0 else None
        ref = TDSignal(h_td, TDSettings(N, DT, force_backend="cpu")).transform(full, window=window).arr
        kept = ref[..., g["n_start"]: g["n_start"] + g["Nt_keep"]]
        out = np.asarray(sg(1.0).arr)
        return g, sg.geometry, float(np.abs(out - kept).max() / np.abs(kept).max())

    def test_whole_grid_segment_equals_the_full_grid_transform(self):
        for alpha in (0.0, 0.3):
            # a chirp reaching both grid edges: no pad absorbs anything here
            g, a, err = self._run(_wdm(), _chirp(0, 128), alpha)
            self.assertEqual((g["n_start"], g["Nt_keep"], g["n_pad_lo"], g["n_pad_hi"]), (0, NT, 0, 0))
            self.assertEqual((a["s0"], a["Nt_seg"]), (0, NT))
            print(f"[whole-grid clamp, alpha={alpha}] kept-layer max rel err {err:.3e}")
            self.assertLess(err, 1e-12)

    def test_cropped_active_box_keeps_the_box_and_the_whole_grid(self):
        """Active box [4, 124) of the 128-layer grid: kept box = the active
        box, segment = the whole grid (pads 4 / 4) -- again the stock
        full-grid transform restricted to the active box."""
        run = WDMSettings(NF, NT, DT, min_time=4 * NF * DT, max_time=123.5 * NF * DT, force_backend="cpu")
        self.assertEqual((int(run.ind_min_t), int(run.ind_max_t) + 1), (4, 124))
        g, a, err = self._run(run, _chirp(2, 126), 0.0)
        self.assertEqual((g["n_start"], g["Nt_keep"], g["n_pad_lo"], g["n_pad_hi"]), (4, 120, 4, 4))
        self.assertEqual((a["s0"], a["Nt_seg"]), (0, NT))
        print(f"[whole-grid clamp, cropped active box] kept-layer max rel err {err:.3e}")
        self.assertLess(err, 1e-12)


class WindowedLeadCoversZeroedHeadTest(unittest.TestCase):
    """No phentax: ``_apply_response`` zeros the first ``buffer_time / dt``
    output samples of the lattice (15000 s by default) -- far more than the
    ``tdi_buffer_time`` (600 s) retarded-read margin. ``set_window``'s lead
    must cover BOTH so the zeroed head ends at or before the segment start."""

    def _gen(self, dt, tdi_buffer_time, buffer_time, waveform_t0=0.0):
        from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform

        class _Stub(WindowedGridAlignedMBHWaveform):
            # dt / tdi_buffer_time are read-only properties on the base class
            dt = None
            tdi_buffer_time = None

        gen = _Stub.__new__(_Stub)
        gen.dt, gen.tdi_buffer_time = dt, tdi_buffer_time
        gen.buffer_time, gen.waveform_t0 = buffer_time, waveform_t0
        return gen

    def test_lead_covers_buffer_time_and_tdi_buffer_time(self):
        for dt, tdi_buf, buf in ((10.0, 600.0, 15000.0), (5.0, 600.0, 15000.0),
                                 (10.0, 600.0, 0.0), (2.5, 600.0, 100.0)):
            gen = self._gen(dt, tdi_buf, buf)
            t_seg, n_seg = 35520.0, 7680
            gen.set_window(t_seg, n_seg)
            k0, n_grid = gen.window_spec
            k_seg = int(round(t_seg / dt))
            lead_time = (k_seg - k0) * dt
            self.assertGreaterEqual(lead_time, buf, (dt, tdi_buf, buf))
            self.assertGreaterEqual(lead_time, tdi_buf, (dt, tdi_buf, buf))
            # the lattice also runs PAST the segment end by >= tdi_buffer_time:
            # the response reads the strain up to ~500 s AHEAD (SSB ->
            # spacecraft delay), so a lattice stopping at the segment end
            # zero-pads a still-inspiralling source there (a merger after the
            # data end) and the segment's last samples carry a step response
            tail_time = (k0 + n_grid - (k_seg + n_seg)) * dt
            self.assertGreaterEqual(tail_time, tdi_buf, (dt, tdi_buf, buf))
            self.assertLess(tail_time, tdi_buf + 2 * dt, (dt, tdi_buf, buf))


class GridAlignedOnsetWarmupTest(unittest.TestCase):
    """No phentax: the stock path zeros the first ``buffer_time`` of TDI output
    counted from the waveform's OWN first sample (its response array starts
    there). On the shared lattice ``_apply_response`` can only zero the
    lattice head, so a row whose onset lies inside the lattice kept its
    turn-on transient: mojito MBHB id 17 carried a 7e-24 TDI spike (200x the
    local signal) through 5000 s after onset, costing 2.4 nats at truth
    (2026-09-30). The grid-aligned dispatch must zero each row's
    ``[onset, onset + buffer_time)`` like the stock path does."""

    DT, BUF, T0 = 10.0, 100.0, 5.0     # 10 zeroed samples per row
    N = 60

    def _gen(self, onsets):
        from lisatools.sources.bbh.gridaligned import GridAlignedPhenomTHMTDIWaveform

        dt, n, t0 = self.DT, self.N, self.T0

        class _Stub(GridAlignedPhenomTHMTDIWaveform):
            dt = None

            def _output_tail_samples(self):
                return 0   # this stubbed lattice carries no tail to crop

            def _aligned_polarizations(self, *args, merger_time, **kwargs):
                b = len(onsets)
                t = np.broadcast_to(np.arange(n) * dt, (b, n))
                h = np.ones((b, n))
                return t, h, h, np.zeros(b), t0 + np.asarray(onsets, float)

            def _apply_response(self, t, hp, hc, ra, dec, m_grid):
                tt = np.atleast_2d(t) + t0
                ch = np.ones((tt.shape[0], 3, n))
                if isinstance(ra, float):
                    return tt[0], ch[0]
                return tt, ch

        gen = _Stub.__new__(_Stub)
        gen.dt, gen.buffer_time, gen.waveform_t0 = dt, self.BUF, t0
        return gen

    def test_batched_rows_zero_their_own_onset_warmup(self):
        onsets = [200.0, 0.0]           # row 0 starts inside the lattice
        gen = self._gen(onsets)
        t, ch = gen._call_batched(np.ones(2), ra=np.ones(2), dec=np.ones(2), merger_time=np.zeros(2))
        for b, on in enumerate(onsets):
            k_on = int(on / self.DT)
            np.testing.assert_array_equal(ch[b, :, :k_on + 10], 0.0)
            np.testing.assert_array_equal(ch[b, :, k_on + 10:], 1.0)

    def test_onset_zeroing_is_in_place(self):
        """No second (B, C, N) array: the mask multiplies ``channels`` in place."""
        gen = self._gen([200.0, 0.0])
        t = np.broadcast_to(np.arange(self.N) * self.DT + self.T0, (2, self.N))
        ch = np.ones((2, 3, self.N))
        _, out = gen._zero_onset_warmup(t, ch, self.T0 + np.array([200.0, 0.0]))
        self.assertIs(out, ch)
        np.testing.assert_array_equal(ch[0, :, :30], 0.0)

    def test_single_row_zeros_its_onset_warmup(self):
        gen = self._gen([200.0])
        t, ch = gen._call_single(1.0, ra=1.0, dec=1.0, merger_time=0.0)
        self.assertEqual(ch.shape, (3, self.N))
        np.testing.assert_array_equal(ch[:, :30], 0.0)
        np.testing.assert_array_equal(ch[:, 30:], 1.0)


class GridAlignedLatticeTailTest(unittest.TestCase):
    """No phentax: the PARENT (full analysis window) lattice runs
    ``tdi_buffer_time`` past the data end, like the windowed subclass's. The
    response reads the strain ~500 s AHEAD of each output sample, so a
    lattice stopping at the data end is a strain step for a source merging
    after it (a truncated epoch, ``scripts/mbh/cd1l_pe.py``). The dispatch
    crops the tail off again after the response, so the output stays
    precisely the data grid (FD consumers require exactly N samples)."""

    DT, N, DATA_T0 = 10.0, 50, 1000.0

    def _gen(self):
        from lisatools.sources.bbh.gridaligned import GridAlignedPhenomTHMTDIWaveform

        seen = {}

        class _Stub(GridAlignedPhenomTHMTDIWaveform):
            def _aligned_polarizations(self, *args, merger_time, **kwargs):
                k0, n_grid = self._common_grid_spec(None)
                b = np.size(merger_time)
                t = np.broadcast_to(np.arange(n_grid) * self.dt + k0 * self.dt, (b, n_grid))
                h = np.ones((b, n_grid))
                return t, h, h, np.zeros(b), np.full(b, -np.inf)

            def _apply_response(self, t, hp, hc, ra, dec, m_grid):
                # like the real one: absolute labels, the lead before data_t0
                # cropped, the lattice END kept
                tt = np.atleast_2d(t) + self.waveform_t0
                tt = tt[:, int(np.rint((self.data_t0 - tt[0, 0]) / self.dt)):]
                seen["response_end"] = float(tt[0, -1])
                ch = np.ones((tt.shape[0], 3, tt.shape[1]))
                if isinstance(ra, float):
                    return tt[0], ch[0]
                return tt, ch

        gen = _Stub.__new__(_Stub)
        gen.domain_settings = TDSettings(self.N, self.DT, t0=self.DATA_T0, force_backend="cpu")
        gen.waveform_t0, gen.buffer_time = self.DATA_T0 - 30 * self.DT, 0.0
        return gen, seen

    def test_lattice_runs_past_the_data_end(self):
        gen, _ = self._gen()
        k0, n_grid = gen._common_grid_spec(None)
        k_data = int(np.rint((self.DATA_T0 - gen.waveform_t0) / self.DT))
        tail = (k0 + n_grid - (k_data + self.N)) * self.DT
        self.assertGreaterEqual(tail, gen.tdi_buffer_time)
        self.assertLess(tail, gen.tdi_buffer_time + 2 * self.DT)

    def test_dispatch_crops_the_tail_back_to_the_data_grid(self):
        gen, seen = self._gen()
        t_end = self.DATA_T0 + (self.N - 1) * self.DT
        t, ch = gen._call_batched(np.ones(2), ra=np.ones(2), dec=np.ones(2), merger_time=np.zeros(2))
        self.assertGreaterEqual(seen["response_end"], t_end + gen.tdi_buffer_time)   # response saw it
        self.assertEqual(ch.shape, (2, 3, self.N))
        self.assertEqual((float(t[0, 0]), float(t[0, -1])), (self.DATA_T0, t_end))
        t1, ch1 = gen._call_single(1.0, ra=1.0, dec=1.0, merger_time=0.0)
        self.assertEqual(ch1.shape, (3, self.N))
        self.assertEqual((float(t1[0]), float(t1[-1])), (self.DATA_T0, t_end))


def _phentax_available():
    try:
        import jax  # noqa: F401
        import phentax  # noqa: F401
        from lisatools.sources.bbh import waveform as _w
        return _w.jax is not None
    except Exception:
        return False


@unittest.skipUnless(_phentax_available(), "needs jax + phentax")
class WindowedGridAlignedPhentaxTest(unittest.TestCase):
    """Real phentax on a tiny CPU grid: 2 days at dt = 10 s, a 6e6 Msun binary
    merging at day 1, generation window T = 12 h, kept box ~19 h."""

    DT = 10.0
    NF, NT = 96, 180          # N = 17280 samples = 2 days
    T_GEN = 43200.0           # phentax T: 12 h before merger
    # waveform basis: m1 m2 s1z s2z dist[Mpc] phi_ref iota psi alpha delta t_plunge
    ROW = np.array([4.0e6, 2.0e6, 0.3, 0.3, 5000.0, 0.3, 0.9, 0.4, 1.0, 0.2, 86400.0])

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform
        from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        cls.wdm = WDMSettings(cls.NF, cls.NT, cls.DT, t0=0.0, min_freq=1e-4, max_freq=2.5e-2, force_backend="cpu")
        common = dict(
            waveform_kwargs=dict(higher_modes=[21, 33, 44], include_negative_modes=True,
                                 t_low_fit=True, coarse_grain=False, atol=1e-12, rtol=1e-12),
            Tobs=cls.T_GEN, start_freq=7e-5, use_reference_time=True,
            waveform_t0=0.0,
            data_td_settings=TDSettings(cls.NF * cls.NT, cls.DT, t0=0.0, force_backend="cpu"),
            tdi_generation="2nd generation", tdi_channels="XYZ",
            sampling_frequency=1.0 / cls.DT,
            orbits=EqualArmlengthOrbits(force_backend="cpu"),
            order=8, tukey_alpha=0.0, stft_dt=None, freq_min=1e-4, freq_max=2.5e-2,
            fft_batch_size=1, buffer_time=15000.0,
            output_domain_settings=cls.wdm, force_backend="cpu",
        )
        cls.common = common
        cls.windowed = WindowedGridAlignedMBHWaveform(**common)
        cls.stock = PhenomTHMTDIWaveform(**common)
        cls.adapter = MBHWindowedWDMSignalGen(cls.windowed, cls.wdm, nchannels=3, tukey_alpha=0.0)
        # kept box: 12 h before + 4 h after merger + 1 h margins; pad 1 h (4 layers of 960 s)
        layer = cls.NF * cls.DT
        n_start = int(np.floor((86400.0 - cls.T_GEN - 3600.0) / layer))
        Nt_keep = int(np.ceil((cls.T_GEN + 4 * 3600.0 + 2 * 3600.0) / layer)) + 1
        if (Nt_keep + 8) % 2:
            Nt_keep += 1
        cls.adapter.set_window(n_start=n_start, Nt_keep=Nt_keep, n_pad=4)
        cls.n_start, cls.Nt_keep = n_start, Nt_keep

    def test_refuses_without_window(self):
        from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform

        gen = WindowedGridAlignedMBHWaveform.__new__(WindowedGridAlignedMBHWaveform)
        with self.assertRaises(RuntimeError):
            WindowedGridAlignedMBHWaveform._common_grid_spec(gen, self.T_GEN)

    def test_batched_rows_share_the_lattice_and_scale_with_distance(self):
        rows = np.stack([self.ROW, self.ROW])
        rows[1, 4] *= 2.0
        times, ch = self.windowed.compute_tdi_channels(*rows.T)
        self.assertEqual(times.ndim, 2)
        np.testing.assert_array_equal(times[0], times[1])
        np.testing.assert_allclose(np.diff(times[0]), self.DT, rtol=0, atol=1e-9)
        self.assertEqual(ch.shape[:2], (2, 3))
        m = np.abs(ch[0]).max()
        np.testing.assert_allclose(ch[1], 0.5 * ch[0], rtol=0, atol=1e-6 * m)

    def test_single_equals_batched_row(self):
        t_b, ch_b = self.windowed.compute_tdi_channels(*np.stack([self.ROW, self.ROW]).T)
        t_s, ch_s = self.windowed.compute_tdi_channels(*self.ROW)
        np.testing.assert_array_equal(t_s, t_b[0])
        # Tolerance is relative to the PEAK, not elementwise: XLA's vmap rounds
        # a B=2 batch differently from B=1 (polarizations differ at 1e-14),
        # and TDI-2 differencing amplifies that to 6e-13 of peak -- 1.3e-7
        # elementwise on samples at 1e-6 of peak (measured 2026-09-30). A B=1
        # batch is bit-identical to the single call.
        m = float(np.abs(ch_b[0]).max())
        np.testing.assert_allclose(ch_s, ch_b[0], rtol=0, atol=1e-10 * m)
        t_1, ch_1 = self.windowed.compute_tdi_channels(*np.stack([self.ROW]).T)
        t_1, ch_1 = np.asarray(t_1), np.asarray(ch_1)
        # a B=1 batch returns times (1, N) but channels already squeezed to
        # (3, N) (measured 2026-09-30); squeeze each independently
        if t_1.ndim == 2:
            t_1 = t_1[0]
        if ch_1.ndim == 3:
            ch_1 = ch_1[0]
        np.testing.assert_array_equal(t_s, t_1)
        np.testing.assert_array_equal(ch_s, ch_1)

    def test_onset_warmup_zeroing_matches_the_stock(self):
        # onset (merger - T_GEN = 43200 s) is INSIDE the windowed lattice: the
        # first nonzero TDI sample must be the stock's (onset + buffer_time),
        # not the ramp's first sample
        t_s, ch_s = self.stock.compute_tdi_channels(*self.ROW)
        t_w, ch_w = self.windowed.compute_tdi_channels(*self.ROW)
        t_s, ch_s, t_w, ch_w = (np.asarray(a) for a in (t_s, ch_s, t_w, ch_w))
        first_s = float(t_s[np.nonzero(ch_s[0])[0][0]])
        first_w = float(t_w[np.nonzero(ch_w[0])[0][0]])
        self.assertAlmostEqual(first_w, first_s, delta=0.5 * self.DT)

    def test_segment_ending_before_the_merger_has_no_end_step(self):
        """A merger AFTER the segment end (edge-clamped box, merger after the
        data end): the source is still inspiralling at the segment end and the
        stock waveform continues past it. The response reads the strain up to
        ~500 s AHEAD of each output sample (SSB -> spacecraft delay), so a
        lattice stopping at the segment end zero-pads the strain there and the
        segment's last few hundred seconds carry a step response (mojito id 16,
        merger 1 d after the grid end: 2.6e5 x the local TDI signal; 3.4e-3 of
        noise-weighted ||delta|| leaking 20 WDM layers into the active box,
        2026-09-30). The sign of the delay depends on the sky position, so the
        row is evaluated at alpha and alpha + pi (one of them reads ahead)."""
        from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform

        gen = WindowedGridAlignedMBHWaveform(**self.common)
        # segment ends 2 h before the merger, where the stock TDI output is
        # nonzero (its first nonzero sample is ~3.7 h before the merger: onset
        # + the 15000 s buffer_time zeroing)
        t_end = 86400.0 - 2 * 3600.0
        n_seg = 4 * self.NF
        gen.set_window(t_end - n_seg * self.DT, n_seg)
        rows = np.stack([self.ROW, self.ROW])
        rows[1, 8] += np.pi
        t_w, ch_w = gen.compute_tdi_channels(*rows.T)
        t_w, ch_w = np.asarray(t_w), np.asarray(ch_w)
        for b in range(2):
            t_s, ch_s = self.stock.compute_tdi_channels(*rows[b])
            t_s, ch_s = np.asarray(t_s), np.asarray(ch_s)
            sel = np.nonzero((t_w[b] >= t_end - 1000.0) & (t_w[b] < t_end))[0]
            i0 = int(np.rint((t_w[b, sel[0]] - t_s[0]) / self.DT))
            np.testing.assert_allclose(t_s[i0:i0 + sel.size], t_w[b, sel], rtol=0, atol=1e-6)
            hs = ch_s[:, i0:i0 + sel.size]
            rel = float(np.abs(ch_w[b][:, sel] - hs).max() / np.abs(hs).max())
            print(f"[segment end before merger, alpha +{b}pi] last 1000 s max rel diff {rel:.3e}")
            # measured 2026-09-30: without the lattice tail, alpha reads ahead
            # and peaks at 2.1e3 x the local stock maximum 300-500 s before the
            # segment end (alpha + pi reads behind: no step); the toy's
            # windowed-vs-stock floor away from the end is ~1e-3
            self.assertLess(rel, 1e-2)

    def test_parent_lattice_ending_before_the_merger_has_no_end_step(self):
        """The PARENT (full analysis window) on a truncated epoch -- data end
        2.1 h before the merger, as ``scripts/mbh/cd1l_pe.py`` truncates --
        has the same forward-read step at the lattice end as the windowed
        lattice had (test above). Paired negative control in the same run:
        the identical generator with the tail switched off."""
        from lisatools.sources.bbh.gridaligned import GridAlignedPhenomTHMTDIWaveform

        class _NoTail(GridAlignedPhenomTHMTDIWaveform):
            def _tail_samples(self):
                return 0

        nt = 82                                  # 82 x 960 s = 21.9 h of data
        n = self.NF * nt
        kw = dict(self.common)
        kw["data_td_settings"] = TDSettings(n, self.DT, t0=0.0, force_backend="cpu")
        kw["output_domain_settings"] = WDMSettings(
            self.NF, nt, self.DT, t0=0.0, min_freq=1e-4, max_freq=2.5e-2, force_backend="cpu")
        t_end = n * self.DT
        rows = np.stack([self.ROW, self.ROW])
        rows[1, 8] += np.pi
        ref = [tuple(np.asarray(a) for a in self.stock.compute_tdi_channels(*rows[b])) for b in range(2)]
        worst = {}
        for name, cls in (("tail", GridAlignedPhenomTHMTDIWaveform), ("no tail", _NoTail)):
            t_p, ch_p = (np.asarray(a) for a in cls(**kw).compute_tdi_channels(*rows.T))
            self.assertEqual(ch_p.shape[-1], n)                   # output = the data grid
            self.assertAlmostEqual(float(t_p[0, -1]), t_end - self.DT, delta=1e-6)
            rel = []
            for b, (t_s, ch_s) in enumerate(ref):
                sel = np.nonzero((t_p[b] >= t_end - 1000.0) & (t_p[b] < t_end))[0]
                i0 = int(np.rint((t_p[b, sel[0]] - t_s[0]) / self.DT))
                np.testing.assert_allclose(t_s[i0:i0 + sel.size], t_p[b, sel], rtol=0, atol=1e-6)
                hs = ch_s[:, i0:i0 + sel.size]
                rel.append(float(np.abs(ch_p[b][:, sel] - hs).max() / np.abs(hs).max()))
            worst[name] = max(rel)
            print(f"[parent, data end 2.1 h before merger, {name}] last 1000 s max rel diff "
                  f"(alpha, alpha + pi) = {rel[0]:.3e}, {rel[1]:.3e}")
        self.assertLess(worst["tail"], 1e-2)
        self.assertGreater(worst["no tail"], 1.0)

    def test_windowed_template_tracks_the_stock_template(self):
        out = self.adapter(*np.stack([self.ROW]).T)
        self.assertEqual(out.arr.shape, (1, 3, self.wdm.Nf_active, self.Nt_keep))
        ref = self.stock.get_signals_for_residuals(*self.ROW)
        ref_box = np.asarray(ref.arr)[..., self.n_start: self.n_start + self.Nt_keep]
        scale = float(np.abs(np.asarray(ref.arr)).max())
        rel = float(np.abs(np.asarray(out.arr[0]) - ref_box).max() / scale)
        outside = float(np.abs(np.delete(np.asarray(ref.arr), np.s_[self.n_start: self.n_start + self.Nt_keep], axis=-1)).max() / scale)
        print(f"[windowed vs stock] kept-box max rel diff {rel:.3e}; stock power outside box {outside:.3e}")
        self.assertLess(rel, 1e-2)
        self.assertLess(outside, 1e-2)


@unittest.skipUnless(_phentax_available(), "needs jax + phentax")
class WindowedGridAlignedPhentaxGPUParityTest(unittest.TestCase):
    """CUDA build of the grid-aligned windowed MBH template == its CPU build (real
    phentax; skips without a GPU backend).

    WindowedGridAlignedPhentaxTest's geometry (2 d at dt = 10 s, T = 12 h, its kept box
    and 4-layer pad -- the phentax JIT shapes are shared), built once per backend.
    phentax runs in JAX on JAX's DEFAULT device for both builds (the generator's
    backend selects the response and transform build, not JAX's device), so the
    polarizations are one computation and the comparison isolates LAT's CUDA vs CPU
    build of the response kernel (LISAResponse.cu) plus the cupy vs numpy placement
    and segment transform.

    Tolerances. TDI channels: 1e-9 of each channel's peak. The two builds compile one
    source (CUDA leads, the CPU build mirrors it) and differ by FMA contraction and
    libm ulps, ~1e-16 relative per operation in the projections; the TDI-2 combination
    differences eight delayed copies and amplifies relative rounding (measured in
    WindowedGridAlignedPhentaxTest.test_single_equals_batched_row: a 1e-14 polarization
    difference becomes 6e-13 of peak), so the floor is <~1e-12 and the bound sits three
    orders above it. Kept WDM layers: 1e-8 of their maximum (one order looser: a
    wavelet's support spans several layers' samples). Paired negative control in the
    same call: row 1 is row 0 with phi_ref + 1e-6 rad, a ~1e-6-of-peak template change;
    GPU row 1 vs CPU row 0 must exceed 1e-7, i.e. the bounds catch a defect two to
    three orders smaller than that control."""

    TDI_TOL, WDM_TOL, CONTROL_MIN = 1e-9, 1e-8, 1e-7

    @classmethod
    def setUpClass(cls):
        cls.gpu = _gpu_backend_name()
        if cls.gpu is None:
            raise unittest.SkipTest("no GPU backend")
        G = WindowedGridAlignedPhentaxTest
        cls.rows = np.stack([G.ROW, G.ROW])
        cls.rows[1, 5] += 1e-6               # the paired control (phi_ref)
        cls.out = {be: cls._run(be) for be in ("cpu", cls.gpu)}

    @classmethod
    def _run(cls, backend):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.sources.batching import MBHWindowedWDMSignalGen
        from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform

        G = WindowedGridAlignedPhentaxTest
        layer = G.NF * G.DT
        wdm = WDMSettings(G.NF, G.NT, G.DT, t0=0.0, min_freq=1e-4, max_freq=2.5e-2, force_backend=backend)
        gen = WindowedGridAlignedMBHWaveform(   # WindowedGridAlignedPhentaxTest's generator kwargs
            waveform_kwargs=dict(higher_modes=[21, 33, 44], include_negative_modes=True,
                                 t_low_fit=True, coarse_grain=False, atol=1e-12, rtol=1e-12),
            Tobs=G.T_GEN, start_freq=7e-5, use_reference_time=True, waveform_t0=0.0,
            data_td_settings=TDSettings(G.NF * G.NT, G.DT, t0=0.0, force_backend=backend),
            tdi_generation="2nd generation", tdi_channels="XYZ", sampling_frequency=1.0 / G.DT,
            # a 600-s orbit grid (the benchmark smoke's): the default 50-s grid over the 5-yr
            # file costs ~2 GB per build, and both builds read identical tables either way
            orbits=EqualArmlengthOrbits(force_backend=backend, linear_interp_setup=False, dt=600.0),
            order=8, tukey_alpha=0.0, stft_dt=None, freq_min=1e-4, freq_max=2.5e-2,
            fft_batch_size=1, buffer_time=15000.0, output_domain_settings=wdm, force_backend=backend,
        )
        adapter = MBHWindowedWDMSignalGen(gen, wdm, nchannels=3, tukey_alpha=0.0)
        n_start = int(np.floor((86400.0 - G.T_GEN - 3600.0) / layer))
        Nt_keep = int(np.ceil((G.T_GEN + 4 * 3600.0 + 2 * 3600.0) / layer)) + 1
        if (Nt_keep + 8) % 2:
            Nt_keep += 1
        adapter.set_window(n_start=n_start, Nt_keep=Nt_keep, n_pad=4)   # also sets gen's window
        t, ch = gen.compute_tdi_channels(*cls.rows.T)
        kept = adapter(*cls.rows.T).arr
        return tuple(np.asarray(asnumpy(a)) for a in (t, ch, kept))

    def _compare(self, idx, tol, what):
        c = self.out["cpu"][idx]
        g = self.out[self.gpu][idx]
        self.assertEqual(g.shape, c.shape)
        for b in range(c.shape[0]):
            for ch in range(3):
                peak = float(np.abs(c[b, ch]).max())
                self.assertGreater(peak, 0.0, (what, b, ch))
                err = float(np.abs(g[b, ch] - c[b, ch]).max()) / peak
                print(f"[phentax GPU vs CPU, {what}] row {b} channel {ch}: max rel diff {err:.2e}")
                self.assertLess(err, tol, (what, b, ch))
        control = float(np.abs(g[1] - c[0]).max() / np.abs(c[0]).max())
        print(f"[phentax GPU vs CPU, {what}] control (phi_ref + 1e-6 rad): {control:.2e}")
        self.assertGreater(control, self.CONTROL_MIN)

    def test_tdi_channels_gpu_equals_cpu(self):
        np.testing.assert_allclose(self.out[self.gpu][0], self.out["cpu"][0], rtol=0, atol=1e-6)
        self._compare(1, self.TDI_TOL, "TDI channels")

    def test_kept_layers_gpu_equals_cpu(self):
        self._compare(2, self.WDM_TOL, "kept WDM layers")


if __name__ == "__main__":
    unittest.main()
