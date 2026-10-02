# tests/test_sobbh_sparse_response.py
"""Sparse response grid for the SOBBH lookup: one response point per 12 h, not per 600 s.

The batched response's evaluation grid must end exactly at the window, stay inside the orbit
tables' coverage (the C++ kernel zeroes the channel outside them, and a half-day cubic spline
across that edge rings back into the window), and at 12-h nodes it must still reproduce the
production dense response and the fine-grid tracer at the pixel centres: the whole channel
phase of an in-band SOBBH is the slow PN carrier plus the orbital Doppler term, both with tiny
fourth derivatives (cubic-spline error ~ h^4 phi''''/384, ~3e-7 rad at 12 h for the chirpiest
in-band case; the 6-month laptop gate reproduced every digit of the 600-s result up to 1-day
nodes).
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import td_mismatch  # noqa: E402
from test_sobbh_wdm_direct import DT, NOBS, REF, ROW_MERGE, ROWS, T_START  # noqa: E402

N_GRID, BUFFER = 2048, 5000.0
FINE_DT, SPARSE_DT = 600.0, 43200.0
#: the chirpiest in-band case: a heavy pair just under the toy grid's 25 mHz Nyquist
ROW_CHIRPY = np.array([60.0, 55.0, 0.1, 0.2, 0.8e9, 2.45e-2, 1.1, 1.2, 0.7, 3.1, 0.2])


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig

        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")
        cls.tdi = TDIConfig("2nd generation", force_backend="cpu")
        cls.grid_t = np.arange(NOBS) * DT + T_START
        cls.t_lo, cls.t_hi = float(cls.grid_t[0]), float(cls.grid_t[-1])

    def _tof(self, eval_dt=SPARSE_DT, buffer_time=BUFFER, reference_time=REF):
        from lisatools.sources.sobbh.wdm_direct import SOBBHBatchedTOF

        return SOBBHBatchedTOF(
            self.orbits,
            self.tdi,
            reference_time,
            n_grid=N_GRID,
            buffer_time=buffer_time,
            eval_dt=eval_dt,
            force_backend="cpu",
        )


class SparseGridTest(_Fixture):
    def test_eval_grid_ends_exactly_at_the_window(self):
        out = self._tof().build(ROWS[:2], self.t_lo, self.t_hi)
        x = np.asarray(out.x)
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(float(x[0, 0]), self.t_lo)
        self.assertEqual(float(x[0, -1]), self.t_hi)
        steps = np.diff(x[0])
        self.assertLessEqual(float(steps.max()), SPARSE_DT + 1e-6)
        # uniform, ceil((hi - lo) / eval_dt) + 1 points: no step shorter than half a step
        self.assertGreater(float(steps.min()), 0.5 * SPARSE_DT)
        self.assertGreaterEqual(x.shape[1], 4)

    def test_eval_grid_stays_inside_the_orbit_span(self):
        from lisatools.sources.sobbh.wdm_direct import SOBBHBatchedTOF

        t0 = float(self.orbits.t_base[0])
        lo = t0 + 100.0  # 100 s into the orbit tables: inside the TDI delay margin
        hi = lo + 10 * SPARSE_DT
        out = self._tof().build(ROWS[:1], lo, hi)
        x = np.asarray(out.x)[0]
        self.assertEqual(float(x[0]), t0 + SOBBHBatchedTOF.DELAY_MARGIN)
        self.assertEqual(float(x[-1]), hi)
        self.assertGreaterEqual(x.size, 4)

    def test_window_outside_the_orbit_span_is_refused(self):
        t_end = float(self.orbits.t_base[-1])
        tof = self._tof()  # constructed OUTSIDE the assertRaises: only build() may raise
        with self.assertRaises(ValueError):
            tof.build(ROWS[:1], t_end + 1.0, t_end + 10 * SPARSE_DT)

    def test_buffer_guard_is_the_delay_margin_not_eval_dt(self):
        from lisatools.sources.sobbh.wdm_direct import SOBBHBatchedTOF

        self.assertEqual(SOBBHBatchedTOF.DELAY_MARGIN, 600.0)
        self._tof(eval_dt=SPARSE_DT, buffer_time=BUFFER)  # 43200 >> 5000 / 2: allowed now
        self._tof(eval_dt=86400.0, buffer_time=BUFFER)
        with self.assertRaises(ValueError):
            self._tof(buffer_time=SOBBHBatchedTOF.DELAY_MARGIN - 1.0)
        with self.assertRaises(ValueError):
            self._tof(eval_dt=0.0)

    def test_default_eval_dt_is_twelve_hours(self):
        from lisatools.sources.sobbh.wdm_direct import SOBBHBatchedTOF, SOBBHDirectWDM

        tof = SOBBHBatchedTOF(self.orbits, self.tdi, REF, force_backend="cpu")
        self.assertEqual(tof.eval_dt, SPARSE_DT)
        self.assertEqual(SOBBHDirectWDM.__init__.__kwdefaults__["eval_dt"], SPARSE_DT)


class SparseGridAccuracyTest(_Fixture):
    def _production(self, row):
        from bbhx.sobbhtdionfly import SOBBHTDIonFly

        from lisatools.sources.sobbh.waveform import SOBBHWaveform

        fly = SOBBHTDIonFly(
            SOBBHWaveform(Tobs=NOBS * DT, dt=DT, t0=T_START, reference_time=REF),
            self.orbits,
            self.tdi,
            DT,
            NOBS * DT,
            t0=T_START,
            n_grid=N_GRID,
            buffer_time=BUFFER,
            force_backend="cpu",
        )
        m1, m2, s1, s2, dist_pc, f_low, phi_c, inc, psi, lam, beta = row
        return np.asarray(
            fly(
                m1,
                m2,
                s1,
                s2,
                dist_pc * 1e-9,
                f_low,
                phi_c,
                inc,
                lam,
                beta,
                psi,
                upsample_t_arr=self.grid_t,
                combine=True,
            )
        )

    def _td(self, rows, eval_dt, reference_time=REF):
        out = self._tof(eval_dt=eval_dt, reference_time=reference_time).build(
            rows, self.t_lo, self.t_hi
        )
        return np.asarray(out.eval_tdi(self.grid_t)), out

    def test_sparse_grid_matches_production(self):
        rows = np.vstack([ROWS, ROW_CHIRPY[None, :]])
        td, out = self._td(rows, SPARSE_DT)
        self.assertTrue(np.all(out.tc > self.t_hi))  # none of these rows merges in the window
        for i, row in enumerate(rows):
            ref = self._production(row)
            for ch in range(3):
                mm, ratio = td_mismatch(td[i, ch], ref[ch])
                self.assertLess(mm, 1e-8, f"row {i} ch {ch}: mm {mm:.3e}")
                self.assertAlmostEqual(ratio, 1.0, delta=1e-6, msg=f"row {i} ch {ch}")

    def test_sparse_grid_control_has_power(self):
        td, _ = self._td(ROW_CHIRPY[None, :], SPARSE_DT, reference_time=REF + 86400.0)
        mm, _ = td_mismatch(td[0, 0], self._production(ROW_CHIRPY)[0])
        self.assertGreater(mm, 1e-1)

    def test_tracer_matches_the_fine_grid_at_the_pixels(self):
        from lisatools.sources.sobbh.wdm_direct import sobbh_tracer

        rows = np.vstack([ROWS, ROW_CHIRPY[None, :]])
        t_pix = T_START + 3600.0 * np.arange(256)  # the toy grid's pixel centres
        fine = sobbh_tracer(self._td(rows, FINE_DT)[1], t_pix)
        sparse = sobbh_tracer(self._td(rows, SPARSE_DT)[1], t_pix)
        amp_f, ph_f, f_f, fd_f = (np.asarray(a) for a in fine)
        amp_s, ph_s, f_s, fd_s = (np.asarray(a) for a in sparse)
        # Measured on this grid (10.7 days, 23 nodes at 12 h): the worst pixel is the chirpy
        # row at the last pixels (the spline's end), 2.9e-6 rad; interior 1.1e-6; catalogue
        # rows < 1e-7. The error scales as h^4 (6 h: 2.0e-7, 24 h: 4.5e-5 -- see the control
        # below). Bounds: phase 1e-5 rad (mismatch contribution ~5e-11), f 1e-10 Hz (< 1e-6 of
        # a layer), fdot 1e-12 Hz/s (1/77 of the production table's fdot step).
        self.assertLess(np.abs(amp_s - amp_f).max() / np.abs(amp_f).max(), 1e-7)
        self.assertLess(np.abs(ph_s - ph_f).max(), 1e-5)
        self.assertLess(np.abs(f_s - f_f).max(), 1e-10)
        self.assertLess(np.abs(fd_s - fd_f).max(), 1e-12)
        # the chirpy row is what it claims to be: the largest fdot of the batch (5.2e-10 Hz/s
        # vs 1.8e-10 for the heaviest catalogue row)
        self.assertGreater(np.abs(fd_f[-1]).max(), 2 * np.abs(fd_f[:-1]).max())

    def test_one_day_grid_exceeds_the_phase_bound(self):
        # the bound above has power: doubling the step multiplies the spline error by ~16
        # (h^4), and a 1-day grid lands at 4.5e-5 rad on the chirpy row
        from lisatools.sources.sobbh.wdm_direct import sobbh_tracer

        t_pix = T_START + 3600.0 * np.arange(256)
        ph_f = np.asarray(sobbh_tracer(self._td(ROW_CHIRPY[None, :], FINE_DT)[1], t_pix)[1])
        ph_d = np.asarray(sobbh_tracer(self._td(ROW_CHIRPY[None, :], 86400.0)[1], t_pix)[1])
        self.assertGreater(np.abs(ph_d - ph_f).max(), 1e-5)

    def test_merging_row_is_masked_past_merger_on_the_sparse_grid(self):
        from lisatools.sources.sobbh.wdm_direct import sobbh_tracer

        _, out = self._td(ROW_MERGE[None, :], SPARSE_DT)
        tc = float(out.tc[0])
        self.assertTrue(self.t_lo < tc < self.t_hi)
        t_pix = T_START + 3600.0 * np.arange(256)
        amp, ph, f, fdot = (np.asarray(a) for a in sobbh_tracer(out, t_pix))
        self.assertTrue(np.all(amp[..., t_pix >= tc] == 0.0))
        self.assertTrue(np.all(np.isfinite(ph)) and np.all(np.isfinite(f)))
        self.assertTrue(np.all(np.isfinite(fdot)))


if __name__ == "__main__":
    unittest.main()
