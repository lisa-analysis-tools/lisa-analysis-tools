"""gb_search_3 re-learns the shutoff valve's per-band cold-lnL max.

User ruling 2026-09-30 ("instead of starting from the stored value, let's
reset that -- for gb search 3"). The max carried in from the fixed-noise
stages was earned under another noise curve; once the foreground moves,
every band sits below it, the valve reads "no improvement" and shuts pairs
that never had a window (6mo job 672: stage 3 complete after three
iterations). The stage profile resets it at entry; the galfor ratchet resets
it again at every nudge.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.noise_ratchet import RatchetSchedule
from lisatools.globalfit.recipe import SearchStageProfileStep, reset_band_logl_max


def _sample(nw=4, nb=10):
    bi = {"band_cold_logl_max_w": np.full((nw, nb), 12.5),
          "band_cold_logl_peak_w": np.full((nw, nb), -np.inf)}
    return SimpleNamespace(sub_states={"gb": SimpleNamespace(band_info=bi)},
                           branches_coords={})


class _FakeGate:
    is_noise_ratchet_gate = True

    def __init__(self):
        self.moves = []
        self.modes = []

    def set_mode(self, m):
        self.modes.append(m)


class _FakeBackend:
    def __init__(self, iteration):
        self.iteration = iteration

    def get_nleaves(self, branch_names=None, temp_index=0):
        return {branch_names[0]: np.arange(self.iteration)[:, None]}

    def stage_start_iteration(self, name):
        return None


class _FakeSampler:
    def __init__(self, iteration, moves):
        self.backend = _FakeBackend(iteration)
        self.moves = moves
        self.periodic = None
        self.temperature_control = None
        self.weights = None


class ResetHelperTest(unittest.TestCase):

    def test_resets_the_max_in_place_and_counts(self):
        s = _sample()
        arr = s.sub_states["gb"].band_info["band_cold_logl_max_w"]
        self.assertEqual(reset_band_logl_max(s), 40)
        self.assertTrue(np.all(np.isneginf(arr)))
        # the peak (the within-cycle window) is left to the judge
        self.assertTrue(np.all(np.isneginf(s.sub_states["gb"].band_info["band_cold_logl_peak_w"])))

    def test_tolerates_states_without_the_array(self):
        self.assertEqual(reset_band_logl_max(SimpleNamespace(sub_states={})), 0)
        self.assertEqual(reset_band_logl_max(None), 0)
        self.assertEqual(reset_band_logl_max(
            SimpleNamespace(sub_states={"gb": SimpleNamespace(band_info={})})), 0)


class ProfileResetTest(unittest.TestCase):

    def _step(self, profile, ratchet=None):
        gate = _FakeGate()
        tree = [SimpleNamespace(moves=[gate])]
        st = SearchStageProfileStep(
            moves=tree, convergence_iter=2, plateau_branch="gb",
            profile=profile, stage_name="gb_search_3", ratchet=ratchet,
            ratchet_delta=None if ratchet is None else np.zeros(5))
        return st, gate, tree

    def test_stage_3_profile_resets_at_entry(self):
        st, _, tree = self._step({"reset_band_max": True})
        s = _sample()
        st.setup_run(47, s, _FakeSampler(47, tree))
        st.note_recipe_step(3)
        self.assertTrue(np.all(np.isneginf(s.sub_states["gb"].band_info["band_cold_logl_max_w"])))

    def test_other_stages_keep_the_stored_max(self):
        st, _, tree = self._step({})
        s = _sample()
        st.setup_run(47, s, _FakeSampler(47, tree))
        st.note_recipe_step(3)
        np.testing.assert_allclose(s.sub_states["gb"].band_info["band_cold_logl_max_w"], 12.5)

    def test_unknown_profile_key_still_refused(self):
        with self.assertRaises(ValueError):
            self._step({"reset_band_maxx": True})

    def test_every_ratchet_nudge_resets_the_max(self):
        st, gate, tree = self._step({}, ratchet=RatchetSchedule(hold=2, release=1, cycles=2))
        s = _sample()
        st.setup_run(47, s, _FakeSampler(47, tree))
        st.note_recipe_step(3)                       # cycle 0: nudge at entry
        self.assertEqual(gate.modes, ["nudge"])
        arr = s.sub_states["gb"].band_info["band_cold_logl_max_w"]
        self.assertTrue(np.all(np.isneginf(arr)))
        arr[...] = 40.0                              # the judge re-learned a max
        st.stopping_function(48, s, _FakeSampler(48, tree))    # -> hold
        st.stopping_function(49, s, _FakeSampler(49, tree))    # -> release
        np.testing.assert_allclose(arr, 40.0)        # holds/releases keep it
        st.stopping_function(50, s, _FakeSampler(50, tree))    # -> nudge (cycle 2)
        self.assertEqual(gate.modes[-1], "nudge")
        self.assertTrue(np.all(np.isneginf(arr)))


if __name__ == "__main__":
    unittest.main()
