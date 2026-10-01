"""TDDenseTDIonTheFly (template-batched, dense-output phases, geometry shared across
harmonics) == TDTDIonTheFly fed the same signal, on inputs both represent EXACTLY:
linear fundamental phases (the dense polynomial reduces to linear interpolation, which a
cubic spline on the old feed reproduces) and constant complex amplitudes.

Checked per harmonic and channel on the analytic signal amp * exp(-i (tdi_phase + phase_ref))
(the two classes split the phase differently between tdi_phase and phase_ref), with several
templates of different sky/polarisation/inclination in ONE call, and the -m-like harmonics
(negative integers).
"""
import unittest

import numpy as np


def _dense_linear(t_k, Phi_k):
    """DOPR853 coefficients (K-1, 3, 8) whose polynomial is linear interpolation of Phi_k."""
    C = np.zeros((t_k.size - 1, 3, 8))
    C[:, :, 0] = Phi_k[:-1]
    C[:, :, 1] = np.diff(Phi_k, axis=0)
    return C


class TDDenseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig

        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")
        cls.tdi = TDIConfig("2nd generation", force_backend="cpu")
        t0, T = 2.0e7, 3 * 86400.0
        cls.t_k = t0 + np.linspace(0.0, T, 41)
        cls.t_in = t0 + np.linspace(0.0, T, 4001)
        cls.t_eval = t0 + np.linspace(2000.0, T - 2000.0, 300)
        cls.omegas = np.array([2 * np.pi * 2.1e-3, 0.0, 2 * np.pi * 0.37e-3])      # d Phi / dt
        cls.phi0 = np.array([0.3, 0.0, 1.1])
        # templates: (inc, psi, lam, beta), harmonics (m, k, n) with complex amplitude
        cls.params = np.array([[0.0, 0.4, 1.2, 0.3], [0.7, 1.9, 4.0, -0.8]])
        cls.harm = [[((2, 0, 0), 1.0 + 0.5j), ((3, 0, 1), -0.3 + 0.2j), ((-2, 0, -1), 0.4 - 0.7j)],
                    [((2, 0, 1), 0.8 - 0.1j), ((1, 0, -2), 0.2 + 0.9j)]]

    def _fund(self, t):
        return self.phi0[None, :] + (t - self.t_k[0])[:, None] * self.omegas[None, :]

    def test_dense_equals_spline_feed(self):
        from lisatools.response.tdionfly import TDDenseTDIonTheFly, TDTDIonTheFly

        AF = 0.5
        K = self.t_k.size
        Phi_k = self._fund(self.t_k)
        offsets, mkn, are, aim, temps = [0], [], [], [], []
        for b, hs in enumerate(self.harm):
            for (m, k, n), c in hs:
                mkn.append((m, k, n))
                a_r = np.zeros((K - 1, 4))
                a_i = np.zeros((K - 1, 4))
                a_r[:, 0], a_i[:, 0] = c.real, c.imag
                are.append(a_r)
                aim.append(a_i)
                temps.append(b)
            offsets.append(len(mkn))
        n_temp = len(self.harm)
        dense = TDDenseTDIonTheFly(
            np.tile(self.t_eval, (n_temp, 1)), np.array(offsets), np.array(mkn),
            np.tile(self.t_k, (n_temp, 1)), np.full(n_temp, K), np.tile(_dense_linear(self.t_k, Phi_k)[None], (n_temp, 1, 1, 1)),
            np.array(are), np.array(aim), amp_factor=AF, tdi_config=self.tdi, orbits=self.orbits, force_backend="cpu")
        out_d = dense(self.params, return_spline=False)
        sig_d = np.asarray(out_d.tdi_amp) * np.exp(-1j * (np.asarray(out_d.tdi_phase) + np.asarray(out_d.phase_ref)[:, None, :]))

        # the old spline feed, one sub per harmonic, same signal: amp |AF c|, phase Phi_s - arg c
        F = self._fund(self.t_in)
        amp_in, ph_in, par = [], [], []
        for b, hs in enumerate(self.harm):
            for (m, k, n), c in hs:
                amp_in.append(np.full(self.t_in.size, AF * abs(c)))
                ph_in.append(F @ np.array([m, k, n], float) - np.angle(c))
                par.append(self.params[b])
        S = len(amp_in)
        old = TDTDIonTheFly(np.tile(self.t_eval, (S, 1)), np.array(amp_in), np.array(ph_in), 1.0, S,
                            t_input=np.tile(self.t_in, (S, 1)), tdi_config=self.tdi, orbits=self.orbits,
                            force_backend="cpu")
        par = np.array(par)
        out_o = old(par[:, 0], par[:, 1], par[:, 2], par[:, 3], return_spline=False)
        sig_o = np.asarray(out_o.tdi_amp) * np.exp(-1j * (np.asarray(out_o.tdi_phase) + np.asarray(out_o.phase_ref)[:, None, :]))

        self.assertEqual(sig_d.shape, sig_o.shape)
        for s in range(S):
            for ch in range(3):
                rel = np.max(np.abs(sig_d[s, ch] - sig_o[s, ch])) / np.max(np.abs(sig_o[s, ch]))
                self.assertLess(rel, 1e-9, f"sub {s} ch {ch}: rel {rel:.2e}")

    def test_zero_outside_trajectory(self):
        """Delayed times past the last knot contribute nothing (plunge handled without a feed extension)."""
        from lisatools.response.tdionfly import TDDenseTDIonTheFly

        K = self.t_k.size
        Phi_k = self._fund(self.t_k)
        a_r = np.zeros((1, K - 1, 4))
        a_r[0, :, 0] = 1.0
        t_late = self.t_k[-1] + np.linspace(2000.0, 4000.0, 20)
        dense = TDDenseTDIonTheFly(t_late[None], np.array([0, 1]), np.array([[2, 0, 0]]), self.t_k[None],
                                   np.array([K]), _dense_linear(self.t_k, Phi_k)[None], a_r, np.zeros_like(a_r),
                                   tdi_config=self.tdi, orbits=self.orbits, force_backend="cpu")
        out = dense(self.params[:1], return_spline=False)
        np.testing.assert_array_equal(np.asarray(out.tdi_amp), 0.0)


class TDDenseGPUParityTest(TDDenseTest):
    """GPU build of the dense kernel == its CPU build (skips without a GPU backend)."""

    def test_gpu_equals_cpu(self):
        import lisatools
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.response.tdionfly import TDDenseTDIonTheFly

        if not lisatools.has_backend("gpu"):
            self.skipTest("no GPU backend")
        K = self.t_k.size
        Phi_k = self._fund(self.t_k)
        rng = np.random.default_rng(3)
        S = 5
        are = rng.normal(size=(S, K - 1, 4)) * np.array([1.0, 1e-6, 1e-12, 1e-18])
        aim = rng.normal(size=(S, K - 1, 4)) * np.array([1.0, 1e-6, 1e-12, 1e-18])
        mkn = np.array([[2, 0, 0], [3, 0, 1], [-2, 0, -1], [2, 0, 1], [1, 0, -2]])
        # N spans the three scratch tiers of kernel 2 (21 N bytes per block): default shared
        # (<= 48 KB), opt-in shared, and the per-block global fallback
        for N in (2000, 4000, 20000):
            t_eval = np.linspace(self.t_eval[0], self.t_eval[-1], N)
            args = (np.tile(t_eval, (2, 1)), np.array([0, 3, 5]), mkn, np.tile(self.t_k, (2, 1)), np.full(2, K),
                    np.tile(_dense_linear(self.t_k, Phi_k)[None], (2, 1, 1, 1)), are, aim)
            outs = []
            for be in ("cpu", "gpu"):
                orb = EqualArmlengthOrbits(force_backend=be)
                tdi = TDIConfig("2nd generation", force_backend=be)
                o = TDDenseTDIonTheFly(*args, amp_factor=0.5, tdi_config=tdi, orbits=orb, force_backend=be)(self.params, return_spline=False)
                outs.append([np.asarray(x.get() if hasattr(x, "get") else x) for x in (o.tdi_amp, o.tdi_phase, o.phase_ref)])
            for a, b in zip(*outs):
                np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-10 * np.max(np.abs(a)), err_msg=f"N={N}")


if __name__ == "__main__":
    unittest.main()
