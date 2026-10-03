"""The ratchet readout compares the MEAN OF THE WALKERS' CURVES, not the curve
of the mean parameters.

6mo job 695, row 81 (after the second release): alpha per walker 2.3 / 3.0 /
8.6 / 9.8 (the alpha-f1 near-degeneracy splits the walkers). The log's
"curve / pre-nudge" from the cold-MEAN vector read +2..5 % at 3.5-5 mHz; the
mean of the four walkers' curves read -6..7 % there. The readout now uses
per-walker curves (``galfor_curves``) and prints the walker spread.
"""

import unittest
from types import SimpleNamespace

import numpy as np


class GalforCurvesTest(unittest.TestCase):

    def test_rows_and_split_alpha_shows_the_difference(self):
        from lisatools.globalfit.noise_ratchet import galfor_curve_ratio, galfor_curves

        f = np.array([3.0, 3.5, 4.5, 5.5]) * 1e-3
        v = np.array([[-43.82, -2.63, 2.3, -2.2, -2.78],      # alpha 2.3
                      [-43.82, -2.63, 9.8, -2.2, -2.78]])     # alpha 9.8
        c = galfor_curves(v, f)
        self.assertEqual(c.shape, (2, 4))
        self.assertTrue(np.all(np.isfinite(c)) and np.all(c > 0))
        np.testing.assert_allclose(galfor_curves(v[0], f)[0], c[0])       # 1-D input = one row
        np.testing.assert_allclose(c[0] / c[1], galfor_curve_ratio(v[0], v[1], f))
        mean_of_curves = c.mean(axis=0)
        curve_of_mean = galfor_curves(v.mean(axis=0), f)[0]
        self.assertGreater(float(np.max(np.abs(mean_of_curves / curve_of_mean - 1.0))), 0.05)


class ReadoutTest(unittest.TestCase):

    def _step(self):
        from lisatools.globalfit.noise_ratchet import RatchetSchedule
        from lisatools.globalfit.recipe import SearchStageProfileStep

        gate = SimpleNamespace(is_noise_ratchet_gate=True, moves=[], mode="release",
                               release_to_convergence=True, ratchet_finished=False,
                               set_mode=lambda m: None, finish_ratchet=lambda: None)
        tree = [SimpleNamespace(moves=[gate])]
        return SearchStageProfileStep(
            moves=tree, convergence_iter=2, plateau_branch="gb", profile={},
            stage_name="gb_search_3",
            ratchet=RatchetSchedule(hold=1, release=1, cycles=2, release_first=True),
            ratchet_delta=np.zeros(5))

    @staticmethod
    def _sample(rows):
        rows = np.asarray(rows, dtype=float)                 # (nw, 5)
        bc = np.zeros((2, rows.shape[0], 1, 5))              # (ntemps, nw, nleaves, ndim)
        bc[0, :, 0, :] = rows
        return SimpleNamespace(log_like=np.zeros((2, rows.shape[0])),
                               branches_coords={"galfor": bc})

    def test_mean_of_curves_and_spread_when_the_per_walker_reference_exists(self):
        from lisatools.globalfit.noise_ratchet import galfor_curves

        st = self._step()
        ref = np.array([[-43.82, -2.63, 2.3, -2.2, -2.78],
                        [-43.82, -2.63, 9.8, -2.2, -2.78]])
        cur = ref.copy()
        cur[:, 0] -= 0.05                                    # every walker's amplitude x 10^-0.05
        st._ratchet_capture_reference(1, self._sample(ref))  # k=1 is the nudge of a release-first schedule
        np.testing.assert_allclose(st._ratchet_pre_nudge, ref.mean(axis=0))
        np.testing.assert_allclose(st._ratchet_pre_nudge_w, ref)
        with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
            st._ratchet_readout(1, self._sample(cur))
        line = [l for l in cm.output if "curve (mean over walkers) / pre-nudge" in l]
        self.assertEqual(len(line), 1, cm.output)
        self.assertIn("walker max/min of the curve", line[0])
        # a pure amplitude shift is exactly 10^-0.05 in the mean-of-curves ratio
        f = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0]) * 1e-3
        expect = galfor_curves(cur, f).mean(0) / galfor_curves(ref, f).mean(0)
        np.testing.assert_allclose(expect, 10 ** -0.05)
        self.assertIn("0.891", line[0])

    def test_falls_back_to_the_mean_vector_without_a_per_walker_reference(self):
        st = self._step()
        ref = np.array([[-43.8, -2.6, 5.0, -2.0, -2.85]] * 2)
        st._ratchet_pre_nudge = ref.mean(axis=0)
        st._ratchet_pre_nudge_w = None
        with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
            st._ratchet_readout(1, self._sample(ref))
        self.assertTrue(any("curve / pre-nudge at" in l and "mean over walkers" not in l
                            for l in cm.output), cm.output)


if __name__ == "__main__":
    unittest.main()
