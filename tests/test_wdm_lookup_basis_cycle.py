"""n_ref lookup evaluation reproduces the TD->WDM transform of a linear chirp (Task A3).

The evaluation rule (derived numerically by scripts/wdm/derive_basis_cycle.py; worst
1.4e-5 of the peak over 2688 (pixel, offset, fdot) samples in 6 offset blocks):

    (c, s) = table (cos, sin) at offset delta = f - m*df and fdot
    if (m_ref + n_ref) is odd:  (c, s) -> (-c, s)
    s *= (-1)**floor(delta/df)                 (undone at the table nodes)
    if (m + n) is odd:  (c, s) -> (s, -c)
    w[m, n] = A * (c cos(phi) - s sin(phi)),   phi = carrier phase at t_n = n * layer_dt
    ((c, s) are the pixel responses to the cos and sin carriers).

A linear chirp has exactly quadratic phase, the model the table stores, so the lookup
must match the truth up to table-interpolation error. Paired negative control: the
historical 2-way rule (``BASIS_CYCLE = "legacy_2way"``) must NOT.
"""

import os
import tempfile
import unittest

import numpy as np

NF, NT, DT = 64, 128, 56.25            # layer_dt = 3600 s
EDGE = 10                              # skip WDM time-edge pixels


class BasisCycleChirpTest(unittest.TestCase):
    M_REF = 20          # even m_ref + n_ref

    @classmethod
    def setUpClass(cls):
        from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings

        cls.WDMLookupTable = WDMLookupTable
        cls.wdm = WDMSettings(Nf=NF, Nt=NT, dt=DT, force_backend="cpu")
        norm_f, m_diffs, m_ref = WDMLookupTable.apply_eps_frequency(0.005, cls.wdm, m_ref=cls.M_REF, num_layers_diff=2)
        fdot_vals = WDMLookupTable.apply_eps_fdot(0.05, cls.wdm, fdot_max_factor=1.0)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.table = WDMLookupTable(
            cls.wdm, 1, m_ref=m_ref, norm_freq_single_layer=norm_f, m_diffs=m_diffs,
            fdot_vals=fdot_vals, store_path=os.path.join(cls.tmp.name, "t.h5"),
            batch_size_gen=32, build_kind="n_ref_complex", time_layers=64)
        cls.TDSettings, cls.TDSignal = TDSettings, TDSignal

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _truth(self, f0, fdot, phi0):
        N = NF * NT
        t = np.arange(N) * DT
        y = np.cos(2 * np.pi * (f0 * t + 0.5 * fdot * t ** 2) + phi0)[None, :]
        return np.asarray(self.TDSignal(y, self.TDSettings(N, DT, force_backend="cpu")).transform(self.wdm).arr)[0]

    def _rel_err(self, f0, fdot, phi0):
        truth = self._truth(f0, fdot, phi0)
        n = np.arange(EDGE, NT - EDGE)
        tn = n * self.wdm.layer_dt
        f = f0 + fdot * tn
        phi = 2 * np.pi * (f0 * tn + 0.5 * fdot * tn ** 2) + phi0
        coeffs, m_map = self.table.get_wdm_coeffs(np.ones_like(tn), phi, f, np.full_like(tn, fdot), n)
        coeffs, m_map = np.asarray(coeffs), np.asarray(m_map)
        num = den = 0.0
        for j in range(coeffs.shape[1]):
            ok = m_map[:, j] >= 0
            tv = truth[m_map[ok, j] - int(self.wdm.ind_min_f), n[ok]]
            num += np.sum((coeffs[ok, j] - tv) ** 2)
            den += np.sum(tv ** 2)
        return np.sqrt(num / den)

    # (f0 in layers, fdot in df/layer_dt ON the table's 0.05 fdot nodes, phi0); every sweep
    # stays inside (3, 60) layers over the tested pixels (a sweep past Nyquist or below 0
    # aliases the truth). f0 + fdot*t_n hits exact layer boundaries for fdot=0.2 (the
    # block-seam case). Off-node fdot is table-resolution (Task A4 sizing), not the rule.
    CASES = [
        (20.3, 0.0, 0.7), (20.5, 0.0, 2.1), (20.93, 0.0, -1.0),
        (18.2, 0.25, 0.4), (20.3, 0.2, 1.1), (15.3, 0.35, -2.2), (50.4, -0.35, 1.3),
    ]

    def _cases(self):
        df, ldt = self.wdm.layer_df, self.wdm.layer_dt
        return [(f * df, fd * df / ldt, p) for f, fd, p in self.CASES]

    def test_quarter_turn_rule_matches_linear_chirp(self):
        self.assertEqual(self.WDMLookupTable.BASIS_CYCLE, "quarter_turn")
        for f0, fdot, phi0 in self._cases():
            e = self._rel_err(f0, fdot, phi0)
            self.assertLess(e, 1e-3, f"f0={f0:.4e} fdot={fdot:.3e} phi0={phi0}: rel L2 err {e:.3e}")

    def test_legacy_two_way_rule_fails(self):
        self.table.BASIS_CYCLE = "legacy_2way"
        try:
            errs = [self._rel_err(*c) for c in self._cases()]
        finally:
            del self.table.BASIS_CYCLE     # back to the class attribute
        self.assertGreater(max(errs), 1e-1, f"legacy errs {errs}")


class InterpMethodTest(BasisCycleChirpTest):
    """set_interp_method rebuilds the interpolators (built once at load); cubic beats linear
    off the table nodes (A9: EMRI 1 amplitude deficit 2.5e-4 -> 5e-6)."""

    def test_cubic_beats_linear_off_node(self):
        # the frequency-direction bias of linear interpolation across the peaked response
        # (what dominated A9); the fdot direction has fine structure cubic cannot fix (A3 scan)
        df = self.wdm.layer_df
        case = (20.31270 * df, 0.0, 0.4)                     # f between nodes (step 0.005 df)
        try:
            self.table.set_interp_method("linear")
            e_lin = self._rel_err(*case)
            self.table.set_interp_method("cubic")
            self.assertEqual(self.table.INTERP_METHOD, "cubic")
            e_cub = self._rel_err(*case)
        finally:
            self.table.set_interp_method("linear")
        self.assertLess(e_cub, 0.5 * e_lin, f"linear {e_lin:.2e} cubic {e_cub:.2e}")

    def test_spline_matches_cubic_accuracy_off_node(self):
        """The GPU-capable uniform spline (prefiltered B-spline + map_coordinates) is as
        accurate as scipy's cubic RegularGridInterpolator off the nodes, and beats linear."""
        df = self.wdm.layer_df
        case = (20.31270 * df, 0.0, 0.4)
        try:
            self.table.set_interp_method("linear")
            e_lin = self._rel_err(*case)
            self.table.set_interp_method("cubic")
            e_cub = self._rel_err(*case)
            self.table.set_interp_method("spline")
            e_spl = self._rel_err(*case)
        finally:
            self.table.set_interp_method("linear")
        self.assertLess(e_spl, 0.5 * e_lin, f"linear {e_lin:.2e} spline {e_spl:.2e}")
        self.assertLess(e_spl, 1.5 * e_cub + 1e-12, f"cubic {e_cub:.2e} spline {e_spl:.2e}")

    def test_spline_reproduces_nodes_and_zero_outside(self):
        from lisatools.domains import _UniformCubicSpline

        rng = np.random.default_rng(2)
        fd, fv = np.linspace(-1, 1, 9), np.linspace(-3, 3, 31)
        tab = rng.normal(size=(9, 31))
        sp = _UniformCubicSpline(tab, fd, fv, np, False)
        FD, FV = np.meshgrid(fd, fv, indexing="ij")
        np.testing.assert_allclose(sp(np.stack([FD, FV], -1)), tab, rtol=0, atol=1e-12)
        self.assertEqual(float(sp(np.array([[0.0, 3.5]]))[0]), 0.0)

    def test_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            self.table.set_interp_method("quintic")


class BasisCycleChirpOddRefTest(BasisCycleChirpTest):
    """Odd m_ref + n_ref: the build bakes a different rotation (is_m_ref_n_ref_even);
    the evaluation must first map the table's (c, s) -> (-c, s)."""

    M_REF = 21


if __name__ == "__main__":
    unittest.main()
