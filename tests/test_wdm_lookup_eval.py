# tests/test_wdm_lookup_eval.py
"""WDMLookupEvaluator: the vectorized n_ref lookup matches the reference evaluation rule.

Reference = ``WDMLookupTable.get_wdm_coeffs`` (the EMRI path, per-layer loop + scipy) and the
TD->WDM truth of a linear chirp; paired negative control = the quarter turn switched off.
Also pins the table's invariance under (Nf, dt) at fixed layer duration (the reason the laptop
EMRI table serves the production grid).
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table, chirp_truth  # noqa: E402

EDGE = 10


class EvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wdm, cls.table = build_tiny_table(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _chirp(self, f_layers, fdot_units, phi0):
        df, ldt = self.wdm.layer_df, self.wdm.layer_dt
        f0, fdot = f_layers * df, fdot_units * df / ldt
        n = np.arange(EDGE, self.wdm.Nt - EDGE)
        tn = n * ldt
        f = f0 + fdot * tn
        phi = 2 * np.pi * (f0 * tn + 0.5 * fdot * tn**2) + phi0
        return f0, fdot, n, f, phi

    def _evaluator(self, interp):
        from lisatools.wdm_lookup_eval import WDMLookupEvaluator

        return WDMLookupEvaluator(self.table, interp=interp)

    def _rel_err(self, ev, f_layers, fdot_units, phi0):
        f0, fdot, n, f, phi = self._chirp(f_layers, fdot_units, phi0)
        truth = chirp_truth(self.wdm, f0, fdot, phi0)
        m = ev.layers_for(f, 2)
        w, ok = ev.coeffs(1.0, phi[:, None], f[:, None], fdot, n[:, None], m)
        w, m = np.asarray(w), np.asarray(m)
        tv = truth[m - int(self.wdm.ind_min_f), n[:, None]]
        return float(np.sqrt(np.sum((w - tv) ** 2) / np.sum(tv**2)))

    def test_matches_get_wdm_coeffs_linear(self):
        self.table.set_interp_method("linear")
        ev = self._evaluator("linear")
        f0, fdot, n, f, phi = self._chirp(20.37, 0.23, 0.4)  # off the f AND fdot nodes
        amp = np.linspace(0.5, 1.5, n.size)
        ref, m_map = self.table.get_wdm_coeffs(
            amp, phi, f, np.full_like(f, fdot), n, num_m_layers=2, out_of_support="zero"
        )
        ref, m_map = np.asarray(ref), np.asarray(m_map)
        m = ev.layers_for(f, 2)
        w, ok = ev.coeffs(amp[:, None], phi[:, None], f[:, None], fdot, n[:, None], m)
        w = np.asarray(w)
        keep = m_map >= 0
        np.testing.assert_array_equal(np.asarray(m)[keep], m_map[keep])
        np.testing.assert_allclose(w[keep], ref[keep], rtol=1e-9, atol=1e-12 * np.abs(ref).max())
        self.assertTrue(np.all(w[~keep] == 0.0))

    # (f0 in layers, fdot in layer units ON the table's 0.1 nodes, phi0)
    CASES = [(20.3, 0.0, 0.7), (18.2, 0.2, 0.4), (15.3, 0.3, -2.2), (50.4, -0.3, 1.3)]

    def test_quarter_turn_matches_chirp_truth(self):
        for interp, tol in (("linear", 2e-3), ("cubic", 1e-3)):
            ev = self._evaluator(interp)
            for case in self.CASES:
                e = self._rel_err(ev, *case)
                self.assertLess(e, tol, f"{interp} {case}: rel L2 {e:.3e}")

    def test_no_parity_turn_control_fails(self):
        ev = self._evaluator("cubic")
        ev.basis_cycle = "no_parity_turn"
        errs = [self._rel_err(ev, *case) for case in self.CASES]
        self.assertGreater(max(errs), 1e-1, f"control errs {errs}")

    def test_cubic_beats_linear_off_node(self):
        case = (20.3137, 0.0, 0.4)  # between f nodes (step 0.01 df)
        e_lin = self._rel_err(self._evaluator("linear"), *case)
        e_cub = self._rel_err(self._evaluator("cubic"), *case)
        self.assertLess(e_cub, 0.5 * e_lin, f"linear {e_lin:.2e} cubic {e_cub:.2e}")

    def test_cubic_off_fdot_node(self):
        # fdot midway between the 0.2 / 0.3 nodes; f stays on its 0.01-df nodes throughout
        # (f_layers=18.2, fdot_units=0.25 are both multiples of the 0.01 f-node spacing), so
        # this isolates fdot interpolation the way test_cubic_beats_linear_off_node isolates f.
        case = (18.2, 0.25, 0.4)
        ev_lin = self._evaluator("linear")
        ev_cub = self._evaluator("cubic")
        e_lin = self._rel_err(ev_lin, *case)
        e_cub = self._rel_err(ev_cub, *case)
        f0, fdot, n, f, phi = self._chirp(*case)
        m = ev_lin.layers_for(f, 2)
        w_lin, _ = ev_lin.coeffs(1.0, phi[:, None], f[:, None], fdot, n[:, None], m)
        w_cub, _ = ev_cub.coeffs(1.0, phi[:, None], f[:, None], fdot, n[:, None], m)
        w_lin, w_cub = np.asarray(w_lin), np.asarray(w_cub)
        self.assertLessEqual(e_cub, e_lin, f"linear {e_lin:.3e} cubic {e_cub:.3e}")
        max_diff = np.abs(w_cub - w_lin).max()
        self.assertGreater(
            max_diff,
            1e-6 * np.abs(w_lin).max(),
            f"cubic and linear must differ off the fdot node: max|diff|={max_diff:.3e}",
        )
        # Sanity bound: this toy table's fdot axis is deliberately coarse (eps_fdot=0.1,
        # fdot_max_factor=1.0 -> 19 nodes over +-0.9 layer units), so even one node-step off
        # fdot=0.2/0.3 carries real interpolation error -- cross-checked against the
        # WDMLookupTable.get_wdm_coeffs reference (scipy RegularGridInterpolator, cubic),
        # which independently reproduces ~0.12 rel L2 at a nearby off-node fdot on this same
        # table (measured 2026-10-01). 0.2 is a loose bound: it is comfortably above the
        # ~0.11 measured here but far below the >=0.23 a real defect produces (the m=0
        # aliasing case, the quarter-turn-sign mutation).
        self.assertLess(e_cub, 0.2, f"cubic rel L2 {e_cub:.3e}")

    def test_cubic_interpolates_the_nodes(self):
        # fdot = 0 keeps every pixel on an f node: both schemes return the node values
        f0, fdot, n, f, phi = self._chirp(20.3, 0.0, 0.4)
        m = self._evaluator("linear").layers_for(f, 2)
        w_lin, _ = self._evaluator("linear").coeffs(
            1.0, phi[:, None], f[:, None], 0.0, n[:, None], m
        )
        w_cub, _ = self._evaluator("cubic").coeffs(
            1.0, phi[:, None], f[:, None], 0.0, n[:, None], m
        )
        np.testing.assert_allclose(np.asarray(w_cub), np.asarray(w_lin), rtol=1e-12, atol=1e-14)

    def test_out_of_support_is_zero_and_flagged(self):
        ev = self._evaluator("cubic")
        f0, fdot, n, f, phi = self._chirp(20.3, 0.0, 0.4)
        m = np.full((n.size, 1), 40)  # delta = -19.7 layers: outside
        w, ok = ev.coeffs(1.0, phi[:, None], f[:, None], 0.0, n[:, None], m)
        self.assertFalse(bool(np.asarray(ok).any()))
        self.assertTrue(bool((np.asarray(w) == 0.0).all()))
        w2, ok2 = ev.coeffs(
            1.0, phi[:, None], f[:, None], 5.0 * ev.fdot_max, n[:, None], ev.layers_for(f, 2)
        )
        self.assertFalse(bool(np.asarray(ok2).any()))

    def test_rejects_bad_options(self):
        from lisatools.wdm_lookup_eval import WDMLookupEvaluator

        with self.assertRaises(ValueError):
            WDMLookupEvaluator(self.table, interp="quintic")


class TablePortabilityTest(unittest.TestCase):
    """The n_ref table depends on the layer duration only (measured 2026-09-30: 4e-9)."""

    def test_same_layer_dt_tables_agree_across_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, ta = build_tiny_table(tmp, nf=64, dt=56.25, eps_freq=0.02, eps_fdot=0.25)
            _, tb = build_tiny_table(tmp, nf=128, dt=28.125, eps_freq=0.02, eps_fdot=0.25)
            A, B = np.asarray(ta.table_cx), np.asarray(tb.table_cx)
            self.assertEqual(A.shape, B.shape)
            np.testing.assert_allclose(B, A, rtol=0, atol=1e-8 * np.abs(A).max())
            # control: a different layer duration (1800 s) is a different table
            _, tc = build_tiny_table(tmp, nf=64, dt=28.125, eps_freq=0.02, eps_fdot=0.25)
            self.assertGreater(np.abs(np.asarray(tc.table_cx) - A).max(), 1e-2 * np.abs(A).max())


if __name__ == "__main__":
    unittest.main()
