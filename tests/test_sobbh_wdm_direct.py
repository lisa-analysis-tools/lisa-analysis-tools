# tests/test_sobbh_wdm_direct.py
"""SOBBH direct-to-WDM building blocks: batched PN, batched response, tracer, template, fill."""

import os
import sys
import tempfile
import unittest

import numpy as np

from lisatools.utils.constants import YRSID_SI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table, td_mismatch  # noqa: E402

# chunked basis rows: (m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)
ROWS = np.array(
    [
        [60.0, 55.0, 0.1, 0.2, 0.4e9, 6.0e-3, 1.1, np.arccos(0.3), 0.7, 3.1, np.arcsin(0.2)],
        [35.0, 30.0, -0.3, 0.5, 2.0e9, 1.2e-2, 2.5, 0.4, 1.9, 1.0, -0.6],
        [90.0, 70.0, 0.0, 0.0, 1.0e9, 1.6e-2, 0.3, 2.2, 0.2, 5.5, 0.9],
    ]
)
# merges inside a few days: masses far above the prior on purpose (chirp-rate stress)
ROW_MERGE = np.array([2000.0, 2000.0, 0.0, 0.0, 1.0e9, 1.5e-2, 0.3, 0.5, 0.7, 1.0, 0.2])

DT = 20.0
NF, NT = 180, 256  # layer_dt = 3600 s, Nyquist 25 mHz, 5.9 days
NOBS = NF * NT
T_START = int(0.5 * YRSID_SI / DT) * DT
REF = float(T_START)
N_GRID, BUFFER, EVAL_DT = 1024, 5000.0, 600.0
EDGE = 24  # grid-end pixels of the dense transform dropped


class AmpPhaseBatchTest(unittest.TestCase):
    def test_rows_match_sobbhwaveform(self):
        from lisatools.sources.sobbh.waveform import SOBBHWaveform
        from lisatools.sources.sobbh.wdm_direct import sobbh_amp_phase_batch

        N, dt = 16384, 20.0
        gen = SOBBHWaveform(Tobs=N * dt, dt=dt, t0=T_START + 3600.0, reference_time=REF)
        rows = np.vstack([ROWS, ROW_MERGE[None, :]])
        amp, ph, tc = sobbh_amp_phase_batch(rows, gen.times, REF)
        self.assertEqual(amp.shape, (rows.shape[0], N))
        n_dead_total = 0
        for i, r in enumerate(rows):
            t_live, a_ref, p_ref = gen.compute_amp_phase(
                r[0], r[1], r[2], r[3], r[4] * 1e-9, r[5], r[6]
            )
            k = t_live.size
            np.testing.assert_allclose(amp[i, :k], a_ref, rtol=1e-12)
            np.testing.assert_allclose(ph[i, :k], p_ref, rtol=1e-12)
            self.assertTrue(np.all(amp[i, k:] == 0.0))
            n_dead_total += N - k
            if k < N:
                self.assertLess(gen.times[k - 1], tc[i])
                self.assertGreaterEqual(gen.times[k], tc[i])
                # the frozen phase is finite and constant past merger
                self.assertTrue(np.all(np.isfinite(ph[i])))
                self.assertEqual(float(np.ptp(ph[i, k:])), 0.0)
                # the freeze actually carries the last live value forward
                # (not merely a tau=0-at-pn_t=0 coincidence)
                self.assertEqual(ph[i, k], ph[i, k - 1])
        self.assertGreater(n_dead_total, 0, "the merging row must merge inside the window")

    def test_t_shift_matches_reference(self):
        from lisatools.sources.sobbh.waveform import SOBBHWaveform
        from lisatools.sources.sobbh.wdm_direct import sobbh_amp_phase_batch

        N, dt = 16384, 20.0
        gen = SOBBHWaveform(Tobs=N * dt, dt=dt, t0=T_START + 3600.0, reference_time=REF)
        rows = np.vstack([ROWS, ROW_MERGE[None, :]])
        ts = 5.0e4
        amp, ph, tc = sobbh_amp_phase_batch(rows, gen.times, REF, t_shift=ts)
        for i, r in enumerate(rows):
            t_live, a_ref, p_ref = gen.compute_amp_phase(
                r[0], r[1], r[2], r[3], r[4] * 1e-9, r[5], r[6], t_shift=ts
            )
            k = t_live.size
            np.testing.assert_allclose(amp[i, :k], a_ref, rtol=1e-12)
            np.testing.assert_allclose(ph[i, :k], p_ref, rtol=1e-12)
            self.assertTrue(np.all(amp[i, k:] == 0.0))
            if k < N:
                self.assertLess(gen.times[k - 1], tc[i])
                self.assertGreaterEqual(gen.times[k], tc[i])

    def test_rejects_short_rows(self):
        from lisatools.sources.sobbh.wdm_direct import sobbh_amp_phase_batch

        with self.assertRaises(ValueError):
            sobbh_amp_phase_batch(np.ones((2, 5)), np.arange(10.0), 0.0)


class BatchedTOFTest(unittest.TestCase):
    """One TDTDIonTheFly for the batch, coarse eval grid, equals the production SOBBHTDIonFly."""

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig

        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")
        cls.tdi = TDIConfig("2nd generation", force_backend="cpu")
        cls.grid_t = np.arange(NOBS) * DT + T_START

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

    def _batched(self, rows, reference_time=REF, eval_dt=EVAL_DT):
        from lisatools.sources.sobbh.wdm_direct import SOBBHBatchedTOF

        tof = SOBBHBatchedTOF(
            self.orbits,
            self.tdi,
            reference_time,
            n_grid=N_GRID,
            buffer_time=BUFFER,
            eval_dt=eval_dt,
            force_backend="cpu",
        )
        out = tof.build(rows, float(self.grid_t[0]), float(self.grid_t[-1]))
        return np.asarray(out.eval_tdi(self.grid_t)), out

    def test_batch_matches_production_per_row(self):
        td, out = self._batched(ROWS)
        self.assertEqual(td.shape, (ROWS.shape[0], 3, NOBS))
        self.assertEqual(out.tc.shape, (ROWS.shape[0],))
        self.assertTrue(np.all(out.tc > self.grid_t[-1]))  # none of ROWS merges here
        for i, row in enumerate(ROWS):
            ref = self._production(row)
            for ch in range(3):
                mm, ratio = td_mismatch(td[i, ch], ref[ch])
                self.assertLess(mm, 1e-8, f"row {i} ch {ch}: mm {mm:.3e}")
                self.assertAlmostEqual(ratio, 1.0, delta=1e-6, msg=f"row {i} ch {ch}")

    def test_reference_epoch_control(self):
        # the parity test has power: a one-day reference-epoch error is caught
        td, _ = self._batched(ROWS[:1], reference_time=REF + 86400.0)
        mm, _ = td_mismatch(td[0, 0], self._production(ROWS[0])[0])
        self.assertGreater(mm, 1e-1)

    def test_tracer_zero_past_merger(self):
        # TDTDIonTheFly fits one global cubic spline through the amplitude nodes, which does
        # not itself land on exactly zero past a merging row's tc -- it RINGS (alternating-sign
        # lobes that decay per node interval, confirmed against production: the worst leak
        # measured for ROW_MERGE at t=tc+2000 was ~9.07e-23). sobbh_tracer must mask using
        # out.tc explicitly rather than relying on the spline.
        from lisatools.sources.sobbh.wdm_direct import sobbh_tracer

        out = self._batched(ROW_MERGE[None, :])[1]
        tc = float(out.tc[0])
        self.assertTrue(self.grid_t[0] < tc < self.grid_t[-1])
        t = np.array([tc - 20000.0, tc - 2000.0, tc + 2000.0, tc + 20000.0])
        amp, _phase, _f, _fdot = sobbh_tracer(out, t)
        # ROW_MERGE's sky/polarization angles happen to project onto a NEGATIVE channel
        # amplitude here (confirmed against the production SOBBHTDIonFly reference, which
        # agrees in sign at t=tc-20000), so the live/dead distinction is on magnitude, not
        # sign: the live samples are ~1e-20..1e-21, 4+ orders of magnitude above the worst
        # pre-fix leak.
        self.assertTrue(np.all(np.abs(amp[0, :, :2]) > 1e-21))
        self.assertTrue(np.all(amp[0, :, 2:] == 0.0))


class TracerTest(unittest.TestCase):
    """sobbh_tracer: amplitude, phase and spline-derivative f / fdot of a known channel phase;
    negative-frequency rows mirrored to positive frequency."""

    def test_known_quadratic_phase(self):
        from lisatools.sources.sobbh.wdm_direct import sobbh_tracer

        f0, fd = 3e-3, 2e-9

        class _Spl:
            def __init__(self, fns):
                self.fns = fns

            def __call__(self, x, derivative=0, **kw):
                return self.fns[derivative](np.asarray(x, dtype=float))

        def ph(t):
            return 2 * np.pi * (f0 * t + 0.5 * fd * t**2)

        class Out:
            xp = np
            num_bin = 2
            tdi_amp = np.zeros((2, 3, 1))
            tdi_amp_spl = _Spl(
                [lambda x: np.where(np.arange(2)[:, None, None] == 0, 2.0, 1.0) + 0 * x]
            )
            tdi_phase_spl = _Spl([lambda x: 0 * x, lambda x: 0 * x, lambda x: 0 * x])
            phase_ref_spl = _Spl(
                [
                    lambda x: np.where(np.arange(2)[:, None] == 0, ph(x), -ph(x)),
                    lambda x: np.where(np.arange(2)[:, None] == 0, 1, -1)
                    * 2
                    * np.pi
                    * (f0 + fd * x),
                    lambda x: np.where(np.arange(2)[:, None] == 0, 1, -1) * 2 * np.pi * fd + 0 * x,
                ]
            )

        t = np.linspace(1e5, 2e5, 11)
        amp, phase, f, fdot = sobbh_tracer(Out(), t)
        self.assertEqual(amp.shape, (2, 3, 11))
        for s_ in (0, 1):
            np.testing.assert_allclose(f[s_], np.broadcast_to(f0 + fd * t, (3, 11)), rtol=1e-12)
            np.testing.assert_allclose(fdot[s_], fd, rtol=1e-12)
            np.testing.assert_allclose(phase[s_], np.broadcast_to(ph(t), (3, 11)), rtol=1e-12)
        np.testing.assert_allclose(amp[0], 2.0)
        np.testing.assert_allclose(amp[1], 1.0)


class DirectWDMTest(unittest.TestCase):
    """The lookup template vs the batched response's own dense TD->WDM; inner products vs the
    container; the fill; guards."""

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.domains import WDMSettings
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.sources.sobbh.wdm_direct import SOBBHDirectWDM

        cls.tmp = tempfile.TemporaryDirectory()
        _, cls.table = build_tiny_table(cls.tmp.name)  # layer 3600 s at Nf=64, dt=56.25
        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")
        cls.tdi = TDIConfig("2nd generation", force_backend="cpu")
        cls.wdm = WDMSettings(
            NF,
            NT,
            DT,
            t0=T_START,
            min_freq=2e-3,
            max_freq=2e-2,
            is_complex=False,
            force_backend="cpu",
        )
        cls.grid_t = np.arange(NOBS) * DT + T_START
        cls.direct = SOBBHDirectWDM(
            cls.wdm,
            cls.table,
            orbits=cls.orbits,
            tdi_config=cls.tdi,
            reference_time=REF,
            n_grid=N_GRID,
            buffer_time=BUFFER,
            eval_dt=EVAL_DT,
            num_m_layers=2,
            interp="cubic",
            force_backend="cpu",
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _truth(self, rows):
        """(N, 3, Nf_active, Nt) dense TD->WDM of the batched response itself."""
        from lisatools.domains import TDSettings, TDSignal

        out = self.direct.tof.build(rows, float(self.grid_t[0]), float(self.grid_t[-1]))
        td = np.asarray(out.eval_tdi(self.grid_t))
        tds = TDSettings(NOBS, DT, force_backend="cpu")
        return np.stack(
            [np.asarray(TDSignal(td[i], tds).transform(self.wdm).arr) for i in range(td.shape[0])]
        )

    def test_dense_matches_tof_own_transform(self):
        sigs = self.direct.dense(ROWS)
        truth = self._truth(ROWS)
        sl = slice(EDGE, NT - EDGE)
        for i in range(ROWS.shape[0]):
            got = np.asarray(sigs[i].arr)
            self.assertEqual(got.shape, truth[i].shape)
            for ch in range(3):
                mm, ratio = td_mismatch(got[ch, :, sl], truth[i, ch, :, sl])
                self.assertLess(mm, 1e-3, f"row {i} ch {ch}: mm {mm:.3e}")
                self.assertAlmostEqual(ratio, 1.0, delta=2e-3, msg=f"row {i} ch {ch}")
        st = self.direct.last_stats
        self.assertEqual(st["dropped_pixels"], 0)
        self.assertEqual(st["merged_rows"], 0)

    def test_dense_matches_truth_on_cropped_grid(self):
        """A cropped active time band whose first pixel is ODD: the lookup must key its
        basis parity / phase on the ABSOLUTE pixel index, not the offset into the band."""
        from lisatools.domains import TDSettings, TDSignal, WDMSettings
        from lisatools.sources.sobbh.wdm_direct import SOBBHDirectWDM

        lt = NF * DT
        wdm_c = WDMSettings(
            NF,
            NT,
            DT,
            t0=T_START,
            min_freq=2e-3,
            max_freq=2e-2,
            min_time=21 * lt,
            max_time=(NT - 25) * lt,
            is_complex=False,
            force_backend="cpu",
        )
        self.assertEqual(wdm_c.ind_min_t % 2, 1)
        self.assertEqual(int(wdm_c.ind_min_t), 21)
        direct_c = SOBBHDirectWDM(
            wdm_c,
            self.table,
            orbits=self.orbits,
            tdi_config=self.tdi,
            reference_time=REF,
            n_grid=N_GRID,
            buffer_time=BUFFER,
            eval_dt=EVAL_DT,
            num_m_layers=2,
            interp="cubic",
            force_backend="cpu",
        )
        rows = ROWS[:2]
        sigs = direct_c.dense(rows)
        out = direct_c.tof.build(rows, float(self.grid_t[0]), float(self.grid_t[-1]))
        td = np.asarray(out.eval_tdi(self.grid_t))
        tds = TDSettings(NOBS, DT, force_backend="cpu")
        nta = int(wdm_c.Nt_active)
        sl = slice(EDGE, nta - EDGE)
        for i in range(rows.shape[0]):
            truth = np.asarray(TDSignal(td[i], tds).transform(wdm_c).arr)
            got = np.asarray(sigs[i].arr)
            self.assertEqual(got.shape, truth.shape)
            self.assertEqual(got.shape[-1], nta)
            for ch in range(3):
                mm, ratio = td_mismatch(got[ch, :, sl], truth[ch, :, sl])
                self.assertLess(mm, 1e-3, f"row {i} ch {ch}: mm {mm:.3e}")
                self.assertAlmostEqual(ratio, 1.0, delta=2e-3, msg=f"row {i} ch {ch}")

    def test_no_parity_turn_control_fails(self):
        self.direct.ev.basis_cycle = "no_parity_turn"
        try:
            got = np.asarray(self.direct.dense(ROWS[:1])[0].arr)
        finally:
            self.direct.ev.basis_cycle = "quarter_turn"
        truth = self._truth(ROWS[:1])[0]
        mm, _ = td_mismatch(got[0, :, EDGE : NT - EDGE], truth[0, :, EDGE : NT - EDGE])
        self.assertGreater(mm, 1e-1)

    def _containers(self):
        from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
        from lisatools.domains import WDMSignal
        from lisatools.sensitivity import XYZ2SensitivityMatrix

        h = [np.asarray(s.arr) for s in self.direct.dense(ROWS)]
        data = [0.7 * h[0] + h[1] + 0.3 * h[2], 0.5 * h[0] - h[1]]
        # a DIFFERENT noise model per walker, so per-walker noise routing has power
        acs = [
            AnalysisContainer(
                WDMSignal(d.copy(), self.wdm), XYZ2SensitivityMatrix(self.wdm, model=model)
            )
            for d, model in zip(data, ("scirdv1", "sangria"))
        ]
        aca = AnalysisContainerArray(acs)
        per_c = 9 * int(self.wdm.Nf_active) * int(self.wdm.Nt_active)
        psd = aca.linear_psd_arr[0]
        self.assertFalse(np.allclose(psd[:per_c], psd[per_c : 2 * per_c]))
        return h, acs, aca

    def test_sparse_inner_products_match_container(self):
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.diagnostic import inner_product
        from lisatools.domains import WDMSignal
        from lisatools.sources.sobbh.wdm_direct import sparse_inner_products

        h, acs, aca = self._containers()
        tpl = self.direct.sparse(np.vstack([ROWS[0], ROWS[0]]))  # row 0 vs both walkers
        kw = dict(
            nchannels=3,
            Nf_active=int(self.wdm.Nf_active),
            Nt_active=int(self.wdm.Nt_active),
            tdi_type="XYZ",
        )
        h0 = WDMSignal(h[0], self.wdm)
        # aligned (noise follows data) and SWAPPED (row r scores walker data_index[r]'s data
        # against walker noise_index[r]'s noise): the swap is what gives the noise routing power
        for data_index, noise_index in (([0, 1], [0, 1]), ([0, 1], [1, 0])):
            d_h, h_h = sparse_inner_products(
                tpl,
                aca.linear_data_arr[0],
                aca.linear_psd_arr[0],
                np.array(data_index),
                np.array(noise_index),
                **kw,
            )
            for r in range(2):
                ac = AnalysisContainer(acs[data_index[r]].data, acs[noise_index[r]].sens_mat)
                ref_dh = float(np.real(ac.template_inner_product(h0)))
                ref_hh = float(np.real(inner_product(h0, h0, psd=ac.sens_mat)))
                msg = f"data {data_index[r]} noise {noise_index[r]}"
                self.assertAlmostEqual(float(d_h[r]) / ref_dh, 1.0, delta=1e-9, msg=msg)
                self.assertAlmostEqual(float(h_h[r]) / ref_hh, 1.0, delta=1e-9, msg=msg)
            # same template, different noise -> different <h|h>
            self.assertNotAlmostEqual(float(h_h[0]) / float(h_h[1]), 1.0, delta=1e-3)
            self.assertNotAlmostEqual(float(d_h[0]), float(d_h[1]))

    def test_fill_round_trip_and_matches_dense(self):
        from lisatools.sources.sobbh.wdm_direct import scatter_add

        h, acs, aca = self._containers()
        buf = aca.linear_data_arr[0]
        before = np.array(buf, copy=True)
        tpl = self.direct.sparse(ROWS[:2])
        kw = dict(nchannels=3, Nf_active=int(self.wdm.Nf_active), Nt_active=int(self.wdm.Nt_active))
        per = 3 * int(self.wdm.Nf_active) * int(self.wdm.Nt_active)
        # data_index deliberately != arange(N) and factors deliberately non-uniform, so the test
        # has power over both the per-row slab routing and the per-row factor (a `scatter_add`
        # that ignored `data_index` in favor of `arange(N)`, or that reused `factors[0]` for
        # every row, would still pass a same-order/uniform-factor version of this check).
        data_index = np.array([1, 0])
        scatter_add(tpl, buf, data_index, np.array([0.5, -2.0]), **kw)
        np.testing.assert_allclose(
            buf[per : 2 * per] - before[per : 2 * per],
            0.5 * h[0].ravel(),
            rtol=0,
            atol=1e-12 * np.abs(h[0]).max(),
        )
        np.testing.assert_allclose(
            buf[:per] - before[:per],
            -2.0 * h[1].ravel(),
            rtol=0,
            atol=1e-12 * np.abs(h[1]).max(),
        )
        scatter_add(tpl, buf, data_index, np.array([-0.5, 2.0]), **kw)
        np.testing.assert_allclose(buf, before, rtol=0, atol=1e-12 * np.abs(before).max())

    def test_merging_row_is_truncated_and_counted(self):
        with self.assertLogs("lisatools.sources.sobbh.wdm_direct", level="WARNING") as cm:
            sigs = self.direct.dense(ROW_MERGE[None, :])
        self.assertEqual(len(cm.records), 1)
        st = self.direct.last_stats
        self.assertEqual(st["merged_rows"], 1)
        self.assertGreater(st["dropped_pixels"], 0)
        tc = float(
            self.direct.tof.build(
                ROW_MERGE[None, :], float(self.grid_t[0]), float(self.grid_t[-1])
            ).tc[0]
        )
        after = self.direct.t_pixels > tc
        self.assertTrue(after.any())
        arr = np.asarray(sigs[0].arr)
        self.assertEqual(float(np.abs(arr[:, :, after]).max()), 0.0)

    def test_layer_dt_mismatch_raises(self):
        from lisatools.domains import WDMSettings
        from lisatools.sources.sobbh.wdm_direct import SOBBHDirectWDM

        other = WDMSettings(90, 64, DT, force_backend="cpu")  # layer_dt 1800 s
        with self.assertRaises(ValueError):
            SOBBHDirectWDM(
                other, self.table, orbits=self.orbits, tdi_config=self.tdi, reference_time=REF
            )

    def test_fill_rejects_noncontiguous(self):
        from lisatools.sources.sobbh.wdm_direct import scatter_add

        tpl = self.direct.sparse(ROWS[:1])
        kw = dict(
            nchannels=3,
            Nf_active=int(self.wdm.Nf_active),
            Nt_active=int(self.wdm.Nt_active),
        )
        buf = np.zeros((int(self.wdm.Nt_active), int(self.wdm.Nf_active), 3)).T  # a view
        with self.assertRaises(ValueError):
            scatter_add(
                tpl,
                buf.reshape(-1) if buf.flags.c_contiguous else buf,
                np.array([0]),
                np.array([1.0]),
                **kw,
            )
        # ndim == 1 but still non-contiguous (a strided slice) must also be rejected -- the
        # ndim-only half of the guard above would let this through.
        per = 3 * int(self.wdm.Nf_active) * int(self.wdm.Nt_active)
        strided = np.zeros(2 * per)[::2]
        self.assertEqual(strided.ndim, 1)
        self.assertFalse(strided.flags.c_contiguous)
        with self.assertRaises(ValueError):
            scatter_add(tpl, strided, np.array([0]), np.array([1.0]), **kw)

    def test_numpy_path_does_not_import_cupyx(self):
        from lisatools.sources.sobbh.wdm_direct import scatter_add

        tpl = self.direct.sparse(ROWS[:1])
        buf = np.zeros(3 * int(self.wdm.Nf_active) * int(self.wdm.Nt_active))
        scatter_add(
            tpl,
            buf,
            np.array([0]),
            np.array([1.0]),
            nchannels=3,
            Nf_active=int(self.wdm.Nf_active),
            Nt_active=int(self.wdm.Nt_active),
        )
        self.assertNotIn("cupyx", sys.modules)
        self.assertGreater(float(np.abs(buf).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
