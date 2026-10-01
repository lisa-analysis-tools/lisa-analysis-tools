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

    def _run(self, a, xI0):
        from lisatools.sources.emri.wdm_direct import (dense_inputs_from_holder, dense_phase_eval,
                                                       harmonic_tracks_from_holder)

        # one trajectory for both cases: the test checks the two code paths agree on it
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
                            ns=np.array([0, -1, 2]), phases=None, freqs=None, integrate_backwards=False)
        t = np.sort(rng.uniform(t_k[0], t_k[-1], 400))
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


if __name__ == "__main__":
    unittest.main()
