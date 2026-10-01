"""dense_inputs_from_holder: the dense-kernel inputs reproduce harmonic_tracks_from_holder
(phase to ~1e-9 rad of a ~1e5 rad phase, complex amplitude to 1e-12) at arbitrary times, for
prograde and retrograde input, including the -m partners."""
import unittest
from types import SimpleNamespace

import numpy as np


class DenseInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from few.trajectory.inspiral import EMRIInspiral
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"FEW unavailable: {exc}")
        cls.insp = EMRIInspiral(func="KerrEccEqFlux")

    def _run(self, a, xI0, backwards=False):
        from lisatools.sources.emri.wdm_direct import (dense_inputs_from_holder, dense_phase_eval,
                                                       harmonic_tracks_from_holder)

        # one trajectory for both cases: the test checks the two code paths agree on it
        if backwards:   # integrate back from a final orbit; FEW adds the backwards phase offsets
            self.insp(1e6, 10.0, 0.9, 8.0, 0.3, 1.0, T=0.2, dt=10.0, err=1e-11, integrate_backwards=True)
        else:
            self.insp(1e6, 10.0, 0.9, 10.0, 0.3, 1.0, T=0.2, dt=10.0, err=1e-11)
        integ = self.insp.inspiral_generator
        # FEW's holder knots: the integrator's knots with the LAST cut inside its final step (at T)
        t_k = np.array(integ.integrator_t_cache)
        t_k[-1] = t_k[-2] + 0.63 * (t_k[-1] - t_k[-2])
        rng = np.random.default_rng(5)
        nm = 3
        teuk = (rng.normal(size=(t_k.size, nm)) + 1j * rng.normal(size=(t_k.size, nm))) * np.linspace(1, 2, t_k.size)[:, None]
        H = SimpleNamespace(t_arr=t_k, teuk_modes=teuk, ylms=rng.normal(size=2 * nm) + 1j * rng.normal(size=2 * nm),
                            ls=np.array([2, 3, 2]), ms=np.array([2, 1, 0]), ks=np.zeros(nm, int),
                            ns=np.array([0, -1, 2]), phases=None, freqs=None, integrate_backwards=backwards)
        # random times, plus the knots themselves and points inside the (refit) LAST segment
        t = np.sort(np.concatenate([rng.uniform(t_k[0], t_k[-1], 400), t_k,
                                    t_k[-2] + rng.uniform(0, 1, 20) * (t_k[-1] - t_k[-2])]))
        tracks = harmonic_tracks_from_holder(H, integ, t, a=a, xI0=xI0, phase_only=True)
        tk2, C, mkn, are, aim = dense_inputs_from_holder(H, integ, a=a, xI0=xI0)
        np.testing.assert_array_equal(tk2, t_k)
        self.assertEqual(len(tracks), mkn.shape[0])
        Phi = dense_phase_eval(t_k, C, t)                                   # (T, 3)
        seg = np.clip(np.searchsorted(t_k, t, side="right") - 1, 0, t_k.size - 2)
        dx = t - t_k[seg]
        for s, tr in enumerate(tracks):
            ph = Phi @ mkn[s].astype(float)
            np.testing.assert_allclose(ph, tr.phase, rtol=0, atol=1e-9 * max(1.0, np.max(np.abs(tr.phase))))
            cr = are[s, seg].T
            ci = aim[s, seg].T
            amp = (cr[0] + dx * (cr[1] + dx * (cr[2] + dx * cr[3]))) + 1j * (ci[0] + dx * (ci[1] + dx * (ci[2] + dx * ci[3])))
            np.testing.assert_allclose(amp, tr.amp, rtol=1e-12, atol=1e-12 * np.max(np.abs(tr.amp)))

    def test_prograde(self):
        self._run(0.9, 1.0)

    def test_retrograde_user_form(self):
        self._run(0.9, -1.0)

    def test_backwards_integrated(self):
        self._run(0.9, 1.0, backwards=True)


class DenseKernelRealTrajectoryTest(unittest.TestCase):
    """TDDenseTDIonTheFly on a REAL FEW trajectory == the spline-fed TDTDIonTheFly fed the same
    harmonics (exact dense-output phases and the same amplitude splines) on a 10 s grid."""

    def test_dense_equals_fine_spline_feed(self):
        try:
            from few.trajectory.inspiral import EMRIInspiral
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"FEW unavailable: {exc}")
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.response.tdionfly import TDDenseTDIonTheFly, TDTDIonTheFly
        from lisatools.sources.emri.wdm_direct import (dense_inputs_from_holder, feed_from_tracks,
                                                       harmonic_tracks_from_holder)

        insp = EMRIInspiral(func="KerrEccEqFlux")
        insp(1e6, 10.0, 0.9, 10.0, 0.3, 1.0, T=0.05, dt=10.0, err=1e-11)
        integ = insp.inspiral_generator
        t_k = np.array(integ.integrator_t_cache)
        rng = np.random.default_rng(9)
        nm = 2
        teuk = (1.0 + 0.3 * np.sin(t_k[:, None] / 2e5 + np.arange(nm))) * np.exp(1j * (t_k[:, None] / 3e5 + np.arange(nm)))
        H = SimpleNamespace(t_arr=t_k, teuk_modes=teuk, ylms=rng.normal(size=2 * nm) + 1j * rng.normal(size=2 * nm),
                            ls=np.array([2, 3]), ms=np.array([2, 3]), ks=np.zeros(nm, int), ns=np.array([0, 1]),
                            phases=None, freqs=None, integrate_backwards=False)
        T0 = 2.0e7
        orb = EqualArmlengthOrbits(force_backend="cpu")
        tdi = TDIConfig("2nd generation", force_backend="cpu")
        par = np.array([0.0, 0.7, 1.3, 0.2])
        t_eval = T0 + np.linspace(t_k[0] + 3000.0, t_k[-1] - 3000.0, 200)

        tk, C, mkn, are, aim = dense_inputs_from_holder(H, integ, a=0.9, xI0=1.0)
        S = mkn.shape[0]
        dense = TDDenseTDIonTheFly(t_eval[None], np.array([0, S]), mkn, (T0 + tk)[None], np.array([tk.size]),
                                   C[None], are, aim, amp_factor=0.5, tdi_config=tdi, orbits=orb, force_backend="cpu")
        od = dense(par[None], return_spline=False)
        sig_d = np.asarray(od.tdi_amp) * np.exp(-1j * (np.asarray(od.tdi_phase) + np.asarray(od.phase_ref)[:, None, :]))

        t_in = np.arange(t_k[0], t_k[-1], 10.0)
        amp_in, ph_in = feed_from_tracks(harmonic_tracks_from_holder(H, integ, t_in, a=0.9, xI0=1.0, phase_only=True), 0.5)
        old = TDTDIonTheFly(np.tile(t_eval, (S, 1)), amp_in, ph_in, 1.0, S, t_input=np.tile(T0 + t_in, (S, 1)),
                            tdi_config=tdi, orbits=orb, force_backend="cpu")
        oo = old(np.full(S, par[0]), np.full(S, par[1]), np.full(S, par[2]), np.full(S, par[3]), return_spline=False)
        sig_o = np.asarray(oo.tdi_amp) * np.exp(-1j * (np.asarray(oo.tdi_phase) + np.asarray(oo.phase_ref)[:, None, :]))
        for s in range(S):
            for ch in range(3):
                rel = np.max(np.abs(sig_d[s, ch] - sig_o[s, ch])) / np.max(np.abs(sig_o[s, ch]))
                self.assertLess(rel, 1e-6, f"sub {s} ch {ch}: rel {rel:.2e}")


if __name__ == "__main__":
    unittest.main()
