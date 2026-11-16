"""Multi-component STFT track accumulator (cutils/lat_stft_track_sum, lisatools.stft_tracks).

* One analytic linear chirp against dt * rfft(w * x), boxcar and Tukey.
* A sum of chirps in several rows, channels and segments against the dense transform of the sum.
* The band values equal STFTFresnelWrap.compute_fourier_values (same evaluator).
* Input normalisation: signed amplitude, negative frequency, chirp-rate floor, non-finite input.
* half_width: skipping, widening, clipping at the edges of the output, an output starting at a
  nonzero f_min.
* Linear envelope: the amplitude slope improves a chirp with a ramping amplitude.
* The same on the CUDA backend against the CPU backend (skipped without a usable GPU).
"""
import unittest

import numpy as np
from scipy.signal.windows import tukey

import lisatools
from lisatools.stft_tracks import (SparseTrackSource, STFTTrackAccumulator, cubic_phase_error, periodic_tukey,
                                   stft_templates_from_tracks, track_half_width)

T_SEG = 21600.0
DT = 5.0
NPER = int(T_SEG / DT)
NF = NPER // 2 + 1
DF = 1.0 / T_SEG
TAU = np.arange(NPER) * DT - 0.5 * T_SEG       # time from the segment midpoint


def chirp(amp, phase, f, fdot, slope=0.0):
    return amp * (1.0 + slope * TAU) * np.cos(phase + 2 * np.pi * (f * TAU + 0.5 * fdot * TAU**2))


def dense(x, alpha):
    w = tukey(NPER, alpha, sym=False) if alpha > 0 else np.ones(NPER)
    return DT * np.fft.rfft(w * x, axis=-1)


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


class TrackSumTest(unittest.TestCase):
    backend = "cpu"

    def acc(self, alpha, **kw):
        kw.setdefault("f_min", 0.0)
        kw.setdefault("num_freqs", NF)
        return STFTTrackAccumulator(T_SEG, window_alpha=alpha, force_backend=self.backend, **kw)

    def get(self, a):
        return a.get() if hasattr(a, "get") else np.asarray(a)

    def one(self, alpha, amp, phase, f, fdot, **kw):
        z = np.array([[[1.0]]])
        return self.get(self.acc(alpha)(amp * z, phase * z, f * z, fdot * z, **kw))[0, 0, 0]

    def test_single_chirp(self):
        par = (1.3, 0.7, 3.0e-3 + 0.37 * DF, 4e-10)
        x = chirp(*par)
        # ? the reference is a sum over samples, the kernel an integral: the boxcar's edges cost
        # ? about dt / (2 * peak), the Tukey window removes that
        for alpha, bound in ((0.0, 4e-3), (0.1, 5e-4), (0.5, 5e-5)):
            ref = dense(x, alpha)
            val = self.one(alpha, *par, n_side=NF)
            self.assertLess(rel(val, ref), bound, f"alpha {alpha}")

    def test_sum_rows_channels_segments(self):
        rng = np.random.default_rng(3)
        C, nch, NT = 5, 3, 4
        rows = np.array([0, 0, 1, 1, 1])
        amp = rng.uniform(0.5, 2.0, (C, nch, NT)) * rng.choice([-1.0, 1.0], (C, nch, NT))
        phase = rng.uniform(-50, 50, (C, nch, NT))
        f = (rng.uniform(1e-3, 8e-3, (C, 1, 1)) + rng.uniform(-2, 2, (C, nch, NT)) * DF)
        fdot = rng.uniform(-3e-9, 3e-9, (C, nch, NT))
        ref = np.zeros((2, nch, NT, NF), dtype=complex)
        for c in range(C):
            for ch in range(nch):
                for j in range(NT):
                    ref[rows[c], ch, j] += dense(chirp(amp[c, ch, j], phase[c, ch, j], f[c, ch, j], fdot[c, ch, j]), 0.2)
        out = self.get(self.acc(0.2)(amp, phase, f, fdot, component_row=rows, n_side=NF))
        self.assertEqual(out.shape, (2, nch, NT, NF))
        for r in range(2):
            self.assertLess(rel(out[r], ref[r]), 3e-4)
        # accumulation into a given output
        acc = self.acc(0.2)
        again = acc(amp, phase, f, fdot, component_row=rows, n_side=NF,
                    out=acc.xp.asarray(out.copy()))
        np.testing.assert_allclose(self.get(again), 2 * out, rtol=1e-13, atol=0)

    def test_equals_array_entry_point(self):
        be = lisatools.get_backend("cpu")
        fr = be.STFTFresnelWrap(1, 1, 3, 0.0, 0.0, 1.0, T_SEG, DF, 0.1, True, False)
        amp, phase, f, fdot = 0.9, -2.2, 5e-3 + 0.2 * DF, -7e-10
        k0 = int(np.floor(f / DF + 0.5))
        ks = np.arange(k0 - 12, k0 + 13)
        ref = np.zeros(ks.size, dtype=complex)
        fr.compute_fourier_values(ref, np.array([amp]), np.array([phase]), np.array([f]), np.array([fdot]),
                                  np.array([0.0]), ks * DF, 1.0, 1, ks.size)
        val = self.one(0.1, amp, phase, f, fdot, half_width=np.array([[12]]))
        np.testing.assert_allclose(val[ks], 0.5 * ref, rtol=1e-12, atol=0)
        self.assertTrue(np.all(val[:ks[0]] == 0) and np.all(val[ks[-1] + 1:] == 0))

    def test_normalisation(self):
        amp, phase, f, fdot = 1.1, 0.4, 4e-3, 6e-10
        base = self.one(0.1, amp, phase, f, fdot)
        self.assertTrue(np.isfinite(base).all() and np.abs(base).max() > 0)
        np.testing.assert_allclose(self.one(0.1, -amp, phase - np.pi, f, fdot), base, rtol=1e-10, atol=1e-12)
        # amp cos(phase) with a negative frequency is the same signal
        np.testing.assert_allclose(self.one(0.1, amp, -phase, -f, -fdot), base, rtol=1e-12, atol=0)
        # chirp rates below the floor behave like a monochromatic line of either sign
        lo = self.one(0.1, amp, phase, f, 0.0, n_side=NF)
        hi = self.one(0.1, amp, phase, f, -1e-30, n_side=NF)
        self.assertTrue(np.isfinite(lo).all())
        self.assertLess(rel(hi, lo), 1e-12)
        self.assertLess(rel(lo, dense(chirp(amp, phase, f, 0.0), 0.1)), 1e-4)
        for bad in (np.nan, np.inf):
            self.assertTrue(np.all(self.one(0.1, bad, phase, f, fdot) == 0))
            self.assertTrue(np.all(self.one(0.1, amp, phase, bad, fdot) == 0))
        self.assertTrue(np.all(self.one(0.1, 0.0, phase, f, fdot) == 0))

    def test_half_width(self):
        amp, phase, f, fdot = 1.0, 0.3, 4e-3, 2e-9
        z = np.ones((1, 1, 3))
        acc = self.acc(0.1)
        full = self.get(acc(amp * z, phase * z, f * z, fdot * z, n_side=NF))[0, 0]
        hw = np.array([[5, -1, 40]])
        out = self.get(acc(amp * z, phase * z, f * z, fdot * z, half_width=hw))[0, 0]
        k0 = int(np.floor(f / DF + 0.5))
        self.assertTrue(np.all(out[1] == 0))
        for j, h in ((0, 5), (2, 40)):
            np.testing.assert_array_equal(out[j, k0 - h:k0 + h + 1], full[j, k0 - h:k0 + h + 1])
            self.assertEqual(np.count_nonzero(out[j]), 2 * h + 1)
        # default width covers the sweep: |fdot| T^2 / 2 bins plus n_side
        self.assertEqual(int(track_half_width(fdot * z, T_SEG, 7)[0, 0]), int(np.ceil(0.5 * fdot * T_SEG**2)) + 7)
        # an output band starting at f_min > 0, clipped on both sides
        lo, n = k0 - 3, 9
        sub = self.get(self.acc(0.1, f_min=lo * DF, num_freqs=n)(amp * z, phase * z, f * z, fdot * z, n_side=NF))[0, 0]
        np.testing.assert_allclose(sub, full[:, lo:lo + n], rtol=1e-9, atol=0)
        # a track entirely outside the output adds nothing
        far = self.get(self.acc(0.1, f_min=lo * DF, num_freqs=n)(amp * z, phase * z, 10 * f * z, fdot * z, n_side=4))
        self.assertTrue(np.all(far == 0))

    def test_linear_envelope(self):
        par = (1.2, 0.9, 3.5e-3 + 0.1 * DF, 5e-10)
        slope = 0.3 / T_SEG
        ref = dense(chirp(*par, slope=slope), 0.3)
        z = np.array([[[1.0]]])
        args = [p * z for p in par]
        flat = self.get(self.acc(0.3)(*args, n_side=NF))[0, 0, 0]
        with_slope = self.get(self.acc(0.3, linear_envelope=True)(*args, n_side=NF, slope=slope * z))[0, 0, 0]
        self.assertGreater(rel(flat, ref), 1e-2)
        self.assertLess(rel(with_slope, ref), 2e-4)
        with self.assertRaises(ValueError):
            self.acc(0.3)(*args, slope=slope * z)

    def test_cubic_phase_error(self):
        t = (np.arange(6) + 0.5) * T_SEG
        fddot = 3e-15
        fdot = np.broadcast_to(1e-9 + fddot * t, (2, 3, 6))
        np.testing.assert_allclose(cubic_phase_error(fdot, t, T_SEG),
                                   (np.pi / 3) * fddot * (T_SEG / 2) ** 3, rtol=1e-9)

    def test_cubic_phase_error_validity(self):
        t = (np.arange(5) + 0.5) * T_SEG
        fdot = np.broadcast_to(np.array([0.0, 1.0, 3.0, 6.0, 10.0]) * 1e-12, (1, 3, 5))
        unit = (np.pi / 3) * (T_SEG / 2) ** 3 * 1e-12 / T_SEG
        # the larger of the two neighbouring differences
        np.testing.assert_allclose(cubic_phase_error(fdot, t, T_SEG)[0] / unit, [1, 2, 3, 4, 4], rtol=1e-9)
        valid = np.array([[True, True, False, True, False]])
        got = cubic_phase_error(fdot, t, T_SEG, valid=valid)[0] / unit
        np.testing.assert_allclose(got[:2], [1, 1], rtol=1e-9)
        self.assertTrue(np.isinf(got[3]))            # no valid neighbour

    def test_periodic_tukey(self):
        for alpha in (0.0, 0.1, 0.5, 1.0):
            ref = tukey(NPER, alpha, sym=False) if alpha > 0 else np.ones(NPER)
            np.testing.assert_allclose(periodic_tukey(NPER, alpha), ref, atol=1e-13)

    def test_shape_errors(self):
        z = np.ones((2, 3, 4))
        acc = self.acc(0.1)
        with self.assertRaises(ValueError):
            acc(z, z, z, z[:, :, :3])
        with self.assertRaises(ValueError):
            acc(z, z, z, z, component_row=np.array([0, 2]), num_rows=2)
        with self.assertRaises(ValueError):
            acc(z, z, z, z, half_width=np.ones((2, 3), dtype=np.int32))


class ToySource:
    """Two rows of chirps with cubic and quartic phase terms, no relation to any astrophysical source.

    Row 0: two components that chirp faster and faster and stop at ``t_end`` (tapered over
    ``TAPER``); their tracks are defined up to ``t_end - MARGIN``. Row 1: one slow component
    that lasts forever. Channel ch scales the amplitude and shifts the phase.
    """

    TAPER = 600.0
    MARGIN = 900.0

    def __init__(self, t_end, xp=np):
        self.xp = xp
        self.num_rows = 2
        self.component_row = np.array([0, 0, 1])
        self.t_end = float(t_end)
        #           amp   phi0  f0      fdot0   fddot0   amp drift [1/s]  fdddot
        self.par = np.array([[1.0, 0.3, 3.0e-3, 2e-10, 1.5e-15, 2e-7, 1.2e-19],
                             [0.6, 1.9, 4.5e-3, -1e-10, -2.5e-15, -1e-7, -0.8e-19],
                             [0.8, -0.7, 6.0e-3, 3e-11, 0.0, 0.0, 0.0]])
        self.ch_amp = np.array([1.0, -0.7, 0.4])     # a signed amplitude on channel 1
        self.ch_phase = np.array([0.0, 0.9, -2.1])

    def _parts(self, t):
        xp = self.xp
        t = xp.asarray(t, dtype=float)[None, None, :]
        a0, p0, f0, fd, fdd, drift, f3 = (xp.asarray(self.par[:, i])[:, None, None] for i in range(7))
        amp = a0 * (1.0 + drift * t) * xp.asarray(self.ch_amp)[None, :, None]
        phase = p0 + xp.asarray(self.ch_phase)[None, :, None] + 2 * np.pi * (
            f0 * t + fd * t**2 / 2 + fdd * t**3 / 6 + f3 * t**4 / 24)
        f = f0 + fd * t + fdd * t**2 / 2 + f3 * t**3 / 6 + 0 * amp
        fdot = fd + fdd * t + f3 * t**2 / 2 + 0 * amp
        return amp, phase, f, fdot

    def eval_tracks(self, t_anchor):
        return self._parts(t_anchor)

    def track_span(self):
        return np.array([-np.inf] * 3), np.array([self.t_end - self.MARGIN] * 2 + [np.inf])

    def signal_span(self):
        return np.array([-np.inf, -np.inf]), np.array([self.t_end, np.inf])

    def td_channels(self, row, t):
        xp = self.xp
        t = xp.asarray(t, dtype=float)
        amp, phase, _, _ = self._parts(t)
        x = (amp * xp.cos(phase))[xp.asarray(np.flatnonzero(self.component_row == row))].sum(axis=0)
        if row == 0:
            u = xp.clip((self.t_end - t) / self.TAPER, 0.0, 1.0)
            x = x * 0.5 * (1.0 - xp.cos(np.pi * u))
        return x


class TemplateDriverTest(unittest.TestCase):
    """stft_templates_from_tracks on a toy source: tracks, dense fallback and zero segments."""

    backend = "cpu"
    ALPHA = 0.3

    def run_case(self, hop):
        acc = STFTTrackAccumulator(T_SEG, 0.0, NF, self.ALPHA, force_backend=self.backend)
        xp = acc.xp
        src = ToySource(t_end=13.4 * T_SEG, xp=xp)
        self.assertIsInstance(src, SparseTrackSource)
        seg = np.arange(0.0, 16 * T_SEG - 1e-6, hop)
        out, info = stft_templates_from_tracks(src, acc, seg, DT, n_side=40, cubic_tol=0.03, return_info=True)
        out = out.get() if hasattr(out, "get") else out
        w = tukey(NPER, self.ALPHA, sym=False)
        t = seg[:, None] + np.arange(NPER) * DT
        ref = np.stack([DT * np.fft.rfft(np.asarray(src.td_channels(r, xp.asarray(t.ravel())).get()
                                                    if hasattr(xp, "asnumpy") else src.td_channels(r, t.ravel())
                                                    ).reshape(3, seg.size, NPER) * w, axis=-1) for r in range(2)])
        return seg, out, ref, info

    def check(self, hop):
        seg, out, ref, info = self.run_case(hop)
        mode = info["mode"]
        # row 1 is slow everywhere; row 0 goes tracks -> dense -> zero
        self.assertTrue(np.all(mode[1] == 1))
        m0 = mode[0]
        self.assertTrue({0, 1, 2} <= set(m0.tolist()))
        self.assertTrue(np.all(np.diff(np.where(m0 == 1, 0, np.where(m0 == 2, 1, 2))) >= 0))
        self.assertTrue(np.all(m0[seg >= 13.4 * T_SEG] == 0))
        self.assertTrue(np.all(m0[(seg < 13.4 * T_SEG) & (seg + T_SEG > 13.4 * T_SEG - ToySource.MARGIN)] == 2))
        # dense segments are the reference itself; zero segments are zero
        np.testing.assert_allclose(out[0][:, m0 == 2], ref[0][:, m0 == 2], rtol=1e-12, atol=1e-12 * np.abs(ref).max())
        self.assertTrue(np.all(out[0][:, m0 == 0] == 0))
        # track segments: every one within the tolerance's reach, and the whole template
        for r in range(2):
            tr = mode[r] == 1
            a, b = out[r][:, tr], ref[r][:, tr]
            worst = max(1 - np.vdot(a[:, j], b[:, j]).real / np.sqrt(np.vdot(a[:, j], a[:, j]).real * np.vdot(b[:, j], b[:, j]).real)
                        for j in range(a.shape[1]))
            self.assertLess(worst, 3e-5, f"row {r}")
            self.assertLess(np.abs(a - b).max() / np.abs(b).max(), 5e-3)
        mm = 1 - np.vdot(out, ref).real / np.sqrt(np.vdot(out, out).real * np.vdot(ref, ref).real)
        self.assertLess(mm, 1e-5)
        return info

    def test_contiguous_segments(self):
        info = self.check(T_SEG)
        # the cubic criterion, not only the track end, sends segments to the dense path
        self.assertGreater(int(np.sum(info["mode"][0] == 2)), 2)

    def test_overlapping_segments(self):
        self.check(0.5 * T_SEG)

    def test_tolerance_controls_fallback(self):
        acc = STFTTrackAccumulator(T_SEG, 0.0, NF, self.ALPHA, force_backend=self.backend)
        src = ToySource(t_end=13.4 * T_SEG, xp=acc.xp)
        seg = np.arange(16) * T_SEG
        n_dense = [int(np.sum(stft_templates_from_tracks(src, acc, seg, DT, cubic_tol=tol, return_info=True)[1]["mode"] == 2))
                   for tol in (1e-3, 0.03, 10.0)]
        self.assertTrue(n_dense[0] > n_dense[1] > n_dense[2] >= 1)
        capped = stft_templates_from_tracks(src, acc, seg, DT, cubic_tol=10.0, max_half_width=31, n_side=30,
                                            return_info=True)[1]["mode"]
        self.assertGreater(int(np.sum(capped == 2)), n_dense[2])

    def test_grid_errors(self):
        acc = STFTTrackAccumulator(T_SEG, 0.0, NF, self.ALPHA, force_backend=self.backend)
        src = ToySource(t_end=5 * T_SEG, xp=acc.xp)
        with self.assertRaises(ValueError):
            stft_templates_from_tracks(src, acc, np.arange(3) * T_SEG, 7.0)
        with self.assertRaises(ValueError):
            stft_templates_from_tracks(src, acc, np.arange(3) * T_SEG, 4 * DT)   # band above Nyquist


def _gpu_backend():
    for name in ("cuda12x", "cuda11x"):
        try:
            be = lisatools.get_backend(name)
            be.xp.zeros(1)
            return name
        except Exception:
            continue
    return None


@unittest.skipIf(_gpu_backend() is None, "no usable GPU backend")
class TrackSumGPUTest(TrackSumTest):
    backend = _gpu_backend()

    def test_matches_cpu(self):
        rng = np.random.default_rng(11)
        C, nch, NT = 40, 3, 16
        rows = rng.integers(0, 3, C)
        amp = rng.normal(size=(C, nch, NT))
        phase = rng.uniform(-1e4, 1e4, (C, nch, NT))
        f = rng.uniform(5e-4, 2e-2, (C, nch, NT))
        fdot = rng.normal(scale=2e-9, size=(C, nch, NT))
        kw = dict(component_row=rows, num_rows=3, n_side=20)
        cpu = STFTTrackAccumulator(T_SEG, 0.0, NF, 0.1, force_backend="cpu")(amp, phase, f, fdot, **kw)
        gpu = self.get(self.acc(0.1)(amp, phase, f, fdot, **kw))
        np.testing.assert_allclose(gpu, cpu, rtol=1e-9, atol=1e-9 * np.abs(cpu).max())


@unittest.skipIf(_gpu_backend() is None, "no usable GPU backend")
class TemplateDriverGPUTest(TemplateDriverTest):
    backend = _gpu_backend()


if __name__ == "__main__":
    unittest.main()

