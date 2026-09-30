"""The JAX bilinear lookup must be exact on a table that is itself bilinear in (f, fdot).

With the two historical bugs (the fdot axis y1/y2 built from ``df_interp`` with no
``min_fdot`` offset, and ``f_x_y2`` reading ``z21`` where ``z12`` belongs) it is not.
Both were inert while every table had ``num_fdot == 1``.
"""

import unittest

import numpy as np


class BilinearExactnessTest(unittest.TestCase):
    def setUp(self):
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            from lisatools.jax.wdm.wavelet_lookup import LOOKUP_N_REF_ONLY, WaveletLookupTableWrapJAX
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"jax lookup unavailable: {exc}")
        self.cls = WaveletLookupTableWrapJAX
        self.kind = LOOKUP_N_REF_ONLY

    def _make(self, z_fn):
        num_f, num_fdot = 9, 7
        df, dfdot = 0.01, 2e-9
        min_f, min_fdot = 0.0, -3 * dfdot
        f = min_f + df * np.arange(num_f)
        fd = min_fdot + dfdot * np.arange(num_fdot)
        F, FD = np.meshgrid(f, fd)                     # (num_fdot, num_f)
        z = z_fn(F, FD)
        tab = self.cls(
            c_nm_all=z, s_nm_all=2 * z, num_f=num_f, num_fdot=num_fdot,
            df_interp=df, dfdot_interp=dfdot, min_f_scaled=min_f, min_fdot=min_fdot,
            layer_df=1.0, layer_dt=1.0, Nf=4, Nt=4, num_channel=1,
            ind_min_t=0, ind_max_t=4, ind_min_f=0, ind_max_f=4, m_ref=0, n_ref=0, kind=self.kind)
        return tab, f, fd

    def test_linear_in_fdot_is_exact(self):
        tab, f, fd = self._make(lambda F, FD: 1.5 * F - 4.0e8 * FD + 0.25)
        fq, fdq = 0.5 * (f[2] + f[3]) + 0.001, 0.5 * (fd[1] + fd[2]) + 1e-10
        got = float(tab._linear_interp(fq, fdq, tab.c_nm_all, 0))
        self.assertAlmostEqual(got, 1.5 * fq - 4.0e8 * fdq + 0.25, places=12)

    def test_cross_term_is_exact(self):
        # f * fdot is bilinear: exercises the z12 / z21 corners (the second bug)
        tab, f, fd = self._make(lambda F, FD: 3.0e8 * F * FD + F)
        fq, fdq = 0.3 * f[4] + 0.7 * f[5], 0.6 * fd[3] + 0.4 * fd[4]
        got = float(tab._linear_interp(fq, fdq, tab.c_nm_all, 0))
        self.assertAlmostEqual(got, 3.0e8 * fq * fdq + fq, places=12)


if __name__ == "__main__":
    unittest.main()
