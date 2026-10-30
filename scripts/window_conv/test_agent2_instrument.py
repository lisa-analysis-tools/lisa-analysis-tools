"""Agent 2: exact instrument bases + fused likelihood (``lisatools.convolution_helpers``).

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m unittest -v test_agent2_instrument

Checks (cpu and cuda12x, golden inputs):
  (a) ``basis_coeffs @ bases`` vs ``compute_sensitivity_matrix(convolve_window=True)``
      (band-normalised, exact to round-off);
  (b) ``fused_log_like`` vs legacy ``compute_log_like(convolve_window=True)`` on the same
      instance and vs the golden (within the legacy CPU-vs-GPU FFT round-off spread);
  (c) ``fused_log_like`` on UNconvolved bases vs ``compute_log_like(convolve_window=False)``
      -- isolates the kernel port, exact to round-off. With splines this is exact too
      (weights multiply unconvolved bases), so it also checks the spline-weight indexing.
Plus: galaxy-free bases (gal1 backend == gal0 twin), the f[0] guard (``full_*``), a
spline smoke test of the factorized combination, a gal1 sanity check, and -- when
agent 3's ``galaxy_convolved_rows`` exists -- bases + galaxy end to end vs golden gal1.
"""

from __future__ import annotations

import unittest

import numpy as np

import _use_worktree  # noqa: F401
from compare import band_reldiff, golden_inputs, load_golden
from fixtures import CONFIGS, make_backend
from lisatools import convolution_helpers as sw

try:
    import cupy as cp

    HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    cp, HAVE_GPU = None, False

BACKENDS = ("cpu", "cuda12x") if HAVE_GPU else ("cpu",)

SENS_TOL = 1e-14  # band-normalised
NOCONV_RTOL = 1e-12  # (c) relative
# (b): legacy convolved ll is reproducible only to |dll| ~ 0.02-0.17 (CPU vs GPU, ll ~ 4e6),
# from FFT round-off amplified near TDI dips; accept a few times that.
CONV_ATOL = 0.5

_BE = {}


def backend(cfg, be):
    if (cfg, be) not in _BE:
        _BE[(cfg, be)] = make_backend(cfg, be)
    return _BE[(cfg, be)]


def host(x):
    return cp.asnumpy(x) if cp is not None and isinstance(x, cp.ndarray) else np.asarray(x)


def spline_weights(sm, llk):
    """(num_psds, 2, nf) multiplicative weights 10**spline, a no-copy transposed view."""
    xp = sm.xp
    s = sm.spline_interpolant(xp.log10(sm.f_arr), llk["knots_position_all"], llk["knots_amplitude_all"])
    return xp.moveaxis(10.0 ** xp.asarray(s), 0, 1)


def run_fused(sm, llk, **kw):
    co = sw.basis_coeffs(sm, llk["Soms_in_all"], llk["Sa_in_all"])
    return sw.fused_log_like(sm, llk["data_in_all"], llk["data_index_all"], co, **kw)


class TestInstrumentBases(unittest.TestCase):
    def test_a_sens_mat(self):
        for cfg in ("gal0_spl0_sym", "gal0_spl0_asym"):
            for be in BACKENDS:
                with self.subTest(cfg=cfg, be=be):
                    sm = backend(cfg, be)
                    smk = golden_inputs(cfg, sm.xp)["sens_mat_kwargs"]
                    B = sw.get_instrument_bases(sm)
                    n_basis = 2 if sm.n_noise_par == 1 else 12
                    self.assertEqual(B[0].shape, (n_basis, sm.basis_settings.total_terms))
                    self.assertEqual(B[0].dtype, np.float64)
                    self.assertEqual(B[3].dtype, np.complex128)
                    self.assertTrue(all(b.flags.c_contiguous for b in B))
                    self.assertTrue(all(bool(sm.xp.isfinite(b).all()) for b in B))
                    self.assertIs(sw.get_instrument_bases(sm), B)  # cached
                    co = sw.basis_coeffs(sm, smk["Soms_d_in"], smk["Sa_a_in"])
                    S_new = sm._fill_matrix(*[co @ b for b in B], num_psds=co.shape[0])
                    S_ref = sm.compute_sensitivity_matrix(**smk, convolve_window=True, smooth=False)
                    r = band_reldiff(S_new, S_ref)[0]
                    r_gold = band_reldiff(S_new, load_golden(cfg, be)["sens_mat_conv"])[0]
                    print(f"\n(a) {cfg}/{be}: band reldiff vs legacy {r:.2e}, vs golden {r_gold:.2e}")
                    self.assertLessEqual(r, SENS_TOL)
                    self.assertLessEqual(r_gold, SENS_TOL)

    def test_basis_coeffs_shapes(self):
        sm = backend("gal0_spl0_sym", "cpu")
        np.testing.assert_array_equal(sw.basis_coeffs(sm, 2.0, 3.0), [[4.0, 9.0]])
        np.testing.assert_array_equal(sw.basis_coeffs(sm, [1.0, 2.0], [3.0, 4.0]), [[1, 9], [4, 16]])
        sm = backend("gal0_spl0_asym", "cpu")
        a, b = np.arange(1.0, 7.0), np.arange(7.0, 13.0)
        np.testing.assert_array_equal(sw.basis_coeffs(sm, a, b), np.concatenate([a, b])[None] ** 2)
        self.assertEqual(sw.basis_coeffs(sm, np.ones((5, 6)), np.ones((5, 6))).shape, (5, 12))

    def test_bases_galaxy_free(self):
        """gal1 backend bases (Amp = 0) are identical to the gal0 twin's."""
        for be in BACKENDS:
            with self.subTest(be=be):
                B0 = sw.get_instrument_bases(backend("gal0_spl0_sym", be))
                B1 = sw.get_instrument_bases(backend("gal1_spl0_sym", be))
                d = max(band_reldiff(x, y)[0] for x, y in zip(B1, B0))
                print(f"\nbases gal1 vs gal0 twin ({be}): {d:.2e}")
                self.assertLessEqual(d, SENS_TOL)

    def test_f0_guard_full_band(self):
        """full_* grid (f[0] = 0): bases finite and equal to legacy with Amp = 0."""
        cfg = "full_gal1_spl0_sym"
        for be in BACKENDS:
            with self.subTest(be=be):
                sm = backend(cfg, be)
                self.assertEqual(float(sm.f_arr[0]), 0.0)
                B = sw.get_instrument_bases(sm)
                self.assertTrue(all(bool(sm.xp.isfinite(b).all()) for b in B))
                smk = dict(golden_inputs(cfg, sm.xp)["sens_mat_kwargs"])
                for k in ("Amp", "alpha", "f_1", "kn", "f_2"):
                    smk[k] = sm.xp.zeros_like(smk[k])
                co = sw.basis_coeffs(sm, smk["Soms_d_in"], smk["Sa_a_in"])
                S_new = sm._fill_matrix(*[co @ b for b in B], num_psds=co.shape[0])
                S_ref = sm.compute_sensitivity_matrix(**smk, convolve_window=True)
                r = band_reldiff(S_new, S_ref)[0]
                print(f"\nf0 guard {cfg}/{be}: band reldiff {r:.2e}")
                self.assertLessEqual(r, SENS_TOL)

    def test_spline_zero_knots(self):
        sm = backend("gal0_spl1_sym", "cpu")
        kp, ka = sw._inst_zero_knots(sm)
        z = np.asarray(sm.spline_interpolant(np.log10(sm.f_arr), kp, ka))
        self.assertEqual(np.abs(z).max(), 0.0)


class TestFusedLogLike(unittest.TestCase):
    def test_b_conv_vs_legacy_and_golden(self):
        for cfg in ("gal0_spl0_sym", "gal0_spl0_asym"):
            for be in BACKENDS:
                with self.subTest(cfg=cfg, be=be):
                    sm = backend(cfg, be)
                    llk = golden_inputs(cfg, sm.xp)["log_like_kwargs"]
                    ll_new = host(run_fused(sm, llk))
                    ll_leg = host(sm.compute_log_like(**llk, convolve_window=True))
                    ll_gold = load_golden(cfg, be)["ll_conv"]
                    d_leg = np.abs(ll_new - ll_leg).max()
                    d_gold = np.abs(ll_new - ll_gold).max()
                    print(f"\n(b) {cfg}/{be}: |dll| vs legacy {d_leg:.3e} vs golden {d_gold:.3e} "
                          f"(rel {d_leg / np.abs(ll_leg).max():.1e}; ll ~ {ll_leg[0]:.6e})")
                    self.assertTrue(np.isfinite(ll_new).all())
                    self.assertLessEqual(d_leg, CONV_ATOL)
                    self.assertLessEqual(d_gold, CONV_ATOL)

    def test_c_noconv_kernel_port(self):
        cases = [("gal0_spl0_sym", False), ("gal0_spl0_asym", False),
                 ("gal0_spl1_sym", True), ("gal0_spl1_asym", True)]
        for cfg, spl in cases:
            for be in BACKENDS:
                with self.subTest(cfg=cfg, be=be):
                    sm = backend(cfg, be)
                    llk = golden_inputs(cfg, sm.xp)["log_like_kwargs"]
                    Braw = sw.get_instrument_bases(sm, convolve_window=False)
                    w = spline_weights(sm, llk) if spl else None
                    ll_new = host(run_fused(sm, llk, bases=Braw, spline_weights=w))
                    ll_ref = host(sm.compute_log_like(**llk, convolve_window=False))
                    r = (np.abs(ll_new - ll_ref) / np.abs(ll_ref)).max()
                    r_gold = (np.abs(ll_new - load_golden(cfg, be)["ll_noconv"]) / np.abs(ll_ref)).max()
                    print(f"\n(c) {cfg}/{be}: rel diff vs legacy noconv {r:.2e} (golden {r_gold:.2e}),"
                          f" |dll| {np.abs(ll_new - ll_ref).max():.2e}")
                    self.assertLessEqual(r, NOCONV_RTOL)
                    self.assertLessEqual(r_gold, NOCONV_RTOL)

    def test_cpu_vs_gpu_fused(self):
        if not HAVE_GPU:
            self.skipTest("no GPU")
        for cfg in ("gal0_spl0_sym", "gal0_spl0_asym"):
            with self.subTest(cfg=cfg):
                out = {}
                for be in BACKENDS:
                    sm = backend(cfg, be)
                    out[be] = host(run_fused(sm, golden_inputs(cfg, sm.xp)["log_like_kwargs"]))
                d = np.abs(out["cpu"] - out["cuda12x"]).max()
                g = load_golden(cfg, "cpu")["ll_conv"] - load_golden(cfg, "cuda12x")["ll_conv"]
                print(f"\nfused cpu vs gpu {cfg}: |dll| {d:.3e} (legacy golden cpu vs gpu {np.abs(g).max():.3e})")
                self.assertLessEqual(d, CONV_ATOL)

    def test_spline_smoke(self):
        """Factorized spline weights on convolved bases: approximate, not asserted."""
        for cfg in ("gal0_spl1_sym", "gal0_spl1_asym"):
            for be in BACKENDS:
                with self.subTest(cfg=cfg, be=be):
                    sm = backend(cfg, be)
                    llk = golden_inputs(cfg, sm.xp)["log_like_kwargs"]
                    ll_new = host(run_fused(sm, llk, spline_weights=spline_weights(sm, llk)))
                    ll_leg = host(sm.compute_log_like(**llk, convolve_window=True))
                    print(f"\nspline smoke {cfg}/{be}: |dll| factorized vs exact {np.abs(ll_new - ll_leg).max():.3e}")
                    self.assertTrue(np.isfinite(ll_new).all())

    def test_gal1_without_galaxy_differs(self):
        for be in BACKENDS:
            with self.subTest(be=be):
                cfg = "gal1_spl0_sym"
                sm = backend(cfg, be)
                llk = golden_inputs(cfg, sm.xp)["log_like_kwargs"]
                d = np.abs(host(run_fused(sm, llk)) - load_golden(cfg, be)["ll_conv"]).min()
                print(f"\ngal1 sanity ({be}): min |dll| without galaxy {d:.3e}")
                self.assertGreater(d, 100 * CONV_ATOL)

    def test_gal1_end_to_end(self):
        """Bases + agent 3's exact galaxy rows vs golden gal1 (if implemented)."""
        cfgs = ("gal1_spl0_sym", "gal1_spl0_asym", "full_gal1_spl0_sym", "narrow_gal1_spl0_sym")
        for cfg in cfgs:
            for be in BACKENDS:
                with self.subTest(cfg=cfg, be=be):
                    sm = backend(cfg, be)
                    llk = golden_inputs(cfg, sm.xp)["log_like_kwargs"]
                    try:
                        gal, ratios = sw.galaxy_convolved_rows(
                            sm, llk["Amp_in_all"], llk["alpha_in_all"], llk["f_1_in_all"],
                            llk["kn_in_all"], llk["f_2_in_all"],
                        )
                    except NotImplementedError:
                        self.skipTest("galaxy_convolved_rows not implemented yet")
                    ll_new = host(run_fused(sm, llk, gal_conv=gal, gal_ratios=ratios))
                    ll_leg = host(sm.compute_log_like(**llk, convolve_window=True))
                    ll_gold = load_golden(cfg, be)["ll_conv"]
                    d_leg, d_gold = np.abs(ll_new - ll_leg).max(), np.abs(ll_new - ll_gold).max()
                    print(f"\ne2e {cfg}/{be}: |dll| vs legacy {d_leg:.3e}, vs golden {d_gold:.3e}")
                    # the legacy ll itself is round-off limited: allow 2x its CPU-vs-GPU spread
                    # (full_*: 1.6 on ll ~ 4.3e6, from the near-DC bins)
                    spread = np.abs(load_golden(cfg, "cpu")["ll_conv"] - load_golden(cfg, "cuda12x")["ll_conv"]).max()
                    tol = max(CONV_ATOL, 2.0 * spread)
                    print(f"     (legacy cpu-vs-gpu spread {spread:.3e}, tol {tol:.3e})")
                    self.assertLessEqual(d_leg, tol)
                    self.assertLessEqual(d_gold, tol)
                    # covariance: bases + galaxy rows vs legacy, exact to round-off
                    smk = golden_inputs(cfg, sm.xp)["sens_mat_kwargs"]
                    g2, R2 = sw.galaxy_convolved_rows(sm, smk["Amp"], smk["alpha"], smk["f_1"], smk["kn"], smk["f_2"])
                    co = sw.basis_coeffs(sm, smk["Soms_d_in"], smk["Sa_a_in"])
                    els = [co @ b for b in sw.get_instrument_bases(sm)]
                    ng = g2.shape[-1]
                    for e, k in enumerate(sw._FUSED_GAL_K):
                        els[e][:, :ng] += R2[0, k] * g2[:, 0, :]
                    S_new = sm._fill_matrix(*els, num_psds=co.shape[0])
                    r = band_reldiff(S_new, sm.compute_sensitivity_matrix(**smk, convolve_window=True))[0]
                    print(f"     sens_mat bases+galaxy band reldiff {r:.2e}")
                    self.assertLessEqual(r, SENS_TOL)


if __name__ == "__main__":
    unittest.main()
