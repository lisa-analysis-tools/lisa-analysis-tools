"""The RJ shutoff valve's frequency floor while the galfor ratchet is active.

User design 2026-10-02: "maybe don't do RJ shutoff during the ratchet
cycles ... you can shut off bands above 7 mHz if their likelihoods converge
as usual." Below the floor the valve never shuts a pair, reopens any shut
pair with a fresh streak; above it the ordinary rule runs; the floor is
lifted at the ratchet's stop, the cycle ceiling and stage end so the stage
can end on the full valve.
"""

import contextlib
import os
import unittest
from types import SimpleNamespace

import numpy as np


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ApplyShutoffFloorTest(unittest.TestCase):

    def test_exempt_bands_never_shut_and_shut_ones_reopen_with_a_fresh_streak(self):
        from lisatools.globalfit.moves.gbspecialstretch import apply_shutoff_floor

        nw, nb = 2, 6
        converged = np.array([[1, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1]], dtype=bool)
        shut = np.array([[1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0]], dtype=bool)
        streak = np.array([[3, 2, 5, 1, 0, 4], [3, 3, 1, 0, 2, 2]])
        exempt = np.array([1, 1, 1, 0, 0, 0], dtype=bool)   # bands 0-2 below the floor
        reopened = apply_shutoff_floor(converged, shut, streak, exempt)
        self.assertEqual(reopened, 4)                                   # (0,0) (0,2) (1,0) (1,1)
        self.assertFalse(converged[:, :3].any())
        self.assertFalse(shut[:, :3].any())
        self.assertTrue((streak[:, :3] == 0).all())
        # above the floor nothing moved
        np.testing.assert_array_equal(converged[:, 3:], [[1, 0, 1], [0, 1, 1]])
        np.testing.assert_array_equal(shut[:, 3:], [[0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(streak[:, 3:], [[1, 0, 4], [0, 2, 2]])
        self.assertEqual(apply_shutoff_floor(converged, shut, streak, np.zeros(nb, bool)), 0)

    def test_exempt_mask_comes_from_the_moves_band_centres(self):
        from lisatools.globalfit.moves import gbspecialstretch as M

        mv = M.GBSpecialStretchMove.__new__(M.GBSpecialStretchMove)
        mv.name = "rj_fstat_search"
        mv.band_edges = np.array([5e-3, 6e-3, 7e-3, 8e-3, 9e-3])     # centres 5.5 6.5 7.5 8.5 mHz
        np.testing.assert_array_equal(mv._shutoff_exempt_bands(7e-3, 4), [True, True, False, False])
        # a grid that does not match the valve's width exempts nothing
        self.assertIsNone(mv._shutoff_exempt_bands(7e-3, 7))


class StepSetsTheFloorTest(unittest.TestCase):

    def _tree(self):
        gate = SimpleNamespace(is_noise_ratchet_gate=True, moves=[], mode="release",
                               release_to_convergence=True, ratchet_finished=False,
                               set_mode=lambda m: None, finish_ratchet=lambda: None)
        grid = SimpleNamespace(name="rj_fstat_search", branch_name="gb", is_rj_prop=True,
                               opt_snr_rej_samp_limit=5.0, phase_maximize=False,
                               _snr_lim_table=None, _shutoff_w_pending=None,
                               rj_shutoff_min_freq=0.0, armed=[])
        grid.arm_fstat_refit = lambda s, r="", ignore_age=False: grid.armed.append(s)
        removal = SimpleNamespace(name="rj_prior_removal", branch_name="gb", is_rj_prop=True,
                                  opt_snr_rej_samp_limit=5.0, phase_maximize=False,
                                  _snr_lim_table=None, rj_removal_only=True, rj_replace=False)
        vgb = SimpleNamespace(name="vgb_pe", branch_name="vgb", opt_snr_rej_samp_limit=0.0)
        return [SimpleNamespace(moves=[gate, grid, removal, vgb])], grid, removal, vgb

    def _step(self, tree, **kw):
        from lisatools.globalfit.noise_ratchet import RatchetSchedule
        from lisatools.globalfit.recipe import SearchStageProfileStep

        return SearchStageProfileStep(
            moves=tree, convergence_iter=2, plateau_branch="gb", profile={},
            stage_name="gb_search_3",
            ratchet=RatchetSchedule(hold=1, release=1, cycles=1, release_first=True),
            ratchet_delta=np.zeros(5), **kw)

    @staticmethod
    def _sampler(it, tree):
        be = SimpleNamespace(iteration=it,
                             get_nleaves=lambda branch_names=None, temp_index=0: {
                                 branch_names[0]: np.arange(it)[:, None]},
                             stage_start_iteration=lambda name: None)
        return SimpleNamespace(backend=be, moves=tree, periodic=None,
                               temperature_control=None, weights=None)

    @staticmethod
    def _smp():
        return SimpleNamespace(log_like=np.array([[1.0, 2.0], [0.0, 0.0]]),
                               branches_coords={"galfor": np.zeros((2, 2, 1, 5))})

    def test_floor_on_every_gb_move_while_active_lifted_at_the_ceiling(self):
        tree, grid, removal, vgb = self._tree()
        st = self._step(tree)
        with env(GALFOR_RATCHET_SHUTOFF_MIN_FREQ=None, GALFOR_RATCHET_HOLD_STAGE=None):
            st.setup_run(47, self._smp(), self._sampler(47, tree))
            st.note_recipe_step(3)
            self.assertEqual(grid.rj_shutoff_min_freq, 7e-3)          # the default
            self.assertEqual(removal.rj_shutoff_min_freq, 7e-3)       # every GB band move
            self.assertFalse(hasattr(vgb, "rj_shutoff_min_freq"))     # vgb untouched
            st.stopping_function(48, self._smp(), self._sampler(48, tree))   # -> k=1 nudge
            self.assertEqual(grid.rj_shutoff_min_freq, 7e-3)
            st.stopping_function(49, self._smp(), self._sampler(49, tree))   # -> k=2 release
            st.stopping_function(50, self._smp(), self._sampler(50, tree))   # -> k=3: no nudge can come
            self.assertEqual(grid.rj_shutoff_min_freq, 0.0)             # lifted
            self.assertEqual(removal.rj_shutoff_min_freq, 0.0)
        with env(GALFOR_RATCHET_SHUTOFF_MIN_FREQ="0"):
            tree, grid, _, _ = self._tree()
            st = self._step(tree)
            st.setup_run(47, self._smp(), self._sampler(47, tree))
            st.note_recipe_step(3)
            self.assertEqual(grid.rj_shutoff_min_freq, 0.0)             # knob 0 = no floor

    def test_floor_lifted_when_the_min_gain_stop_ends_the_stage(self):
        tree, grid, removal, _ = self._tree()
        grid._shutoff_w_pending = 0                                   # valve says done
        st = self._step(tree, ratchet_min_gain=200.0)
        with env(GALFOR_RATCHET_SHUTOFF_MIN_FREQ=None, GALFOR_RATCHET_HOLD_STAGE=None,
                 GB_SEARCH_STAGE_END_ON_SHUTOFF=None):
            def smp(mx):
                return SimpleNamespace(log_like=np.array([[mx, mx - 1.0], [0.0, 0.0]]),
                                       branches_coords={"galfor": np.zeros((2, 2, 1, 5))})
            st.setup_run(47, smp(1000.0), self._sampler(47, tree))
            st.note_recipe_step(3)
            self.assertEqual(grid.rj_shutoff_min_freq, 7e-3)
            st.stopping_function(48, smp(1000.0), self._sampler(48, tree))   # baseline, -> nudge
            st.stopping_function(49, smp(-9000.0), self._sampler(49, tree))  # -> release
            self.assertTrue(st.stopping_function(50, smp(1050.0), self._sampler(50, tree)))  # stop + end
            self.assertEqual(grid.rj_shutoff_min_freq, 0.0)
            self.assertEqual(removal.rj_shutoff_min_freq, 0.0)


if __name__ == "__main__":
    unittest.main()
