"""The level-3 valve is released on the STATE at recipe-step entry.

6mo job 675 (2026-10-01, the first ratcheted + legged relaunch): the gated
noise head's in-model follow-up -- the first GB move of gb_search_3 -- ran
ZERO proposals ("[GB_ACCEPT in_model] in-model cold 0/0", one 1 s pass with
every occupied sub-band "static"). The pure in-model move binds the valve's
shut table READ-ONLY for its pick mask (no stamp check, no release), and the
release on a step change lived only in the RJ moves' bind path, so the
first non-RJ move of a new step read the PREVIOUS step's table: row 46 ended
gb_search_2 with every occupied (walker, band) shut, hence nothing to pick.
Job 672 did not show it only because its state came from a stage-3
checkpoint whose stamp already matched.

The fix releases the window on the live state when the stored step stamp
names another step, at ``note_recipe_step`` (before the step's first move),
with the same rule the RJ bind path applies (stamp differs -> release and
re-stamp; same stamp -> honour). The all-time max is untouched, as ruled.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.recipe import (SearchStageProfileStep,
                                        release_band_shutoff_window)


def _sample(nw=4, nb=10, stamp=2):
    bi = {
        "band_rj_shutoff_w": np.ones((nw, nb), dtype=bool),
        "band_shutoff_w_step": np.array([stamp], dtype=np.int64),
        "band_shutoff_streak_w": np.full((nw, nb), 3, dtype=np.int64),
        "band_shutoff_reset_w": np.full((nw, nb), 1, dtype=np.int64),
        "band_cold_logl_peak_w": np.full((nw, nb), 7.5),
        "band_cold_logl_w": np.full((nw, nb), 7.0),
        "band_cold_logl_max_w": np.full((nw, nb), 12.5),
    }
    return SimpleNamespace(sub_states={"gb": SimpleNamespace(band_info=bi)},
                           branches_coords={})


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


class HelperTest(unittest.TestCase):

    def test_a_stale_stamp_reopens_every_pair_and_clears_the_window(self):
        s = _sample(stamp=2)
        bi = s.sub_states["gb"].band_info
        n, released = release_band_shutoff_window(s, 3)
        self.assertTrue(released)
        self.assertEqual(n, 40)
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self.assertEqual(int(bi["band_shutoff_w_step"][0]), 3)
        self.assertTrue((bi["band_shutoff_streak_w"] == 0).all())
        self.assertTrue((bi["band_shutoff_reset_w"] == 0).all())
        self.assertTrue(np.all(np.isneginf(bi["band_cold_logl_peak_w"])))
        # the all-time max outlives steps (ruling 2026-09-27)
        np.testing.assert_allclose(bi["band_cold_logl_max_w"], 12.5)

    def test_the_arrays_are_the_same_objects_the_saver_holds(self):
        s = _sample(stamp=2)
        bi = s.sub_states["gb"].band_info
        shut, streak = bi["band_rj_shutoff_w"], bi["band_shutoff_streak_w"]
        release_band_shutoff_window(s, 3)
        self.assertIs(bi["band_rj_shutoff_w"], shut)
        self.assertIs(bi["band_shutoff_streak_w"], streak)

    def test_the_same_step_is_honoured(self):
        s = _sample(stamp=3)
        bi = s.sub_states["gb"].band_info
        self.assertEqual(release_band_shutoff_window(s, 3), (0, False))
        self.assertTrue(bi["band_rj_shutoff_w"].all())
        self.assertTrue((bi["band_shutoff_streak_w"] == 3).all())

    def test_the_unset_sentinel_counts_as_another_step(self):
        s = _sample(stamp=-1)
        n, released = release_band_shutoff_window(s, 0)
        self.assertTrue(released)
        self.assertEqual(int(s.sub_states["gb"].band_info["band_shutoff_w_step"][0]), 0)

    def test_tolerates_states_without_the_valve(self):
        self.assertEqual(release_band_shutoff_window(None, 3), (0, False))
        self.assertEqual(release_band_shutoff_window(
            SimpleNamespace(sub_states={}), 3), (0, False))
        self.assertEqual(release_band_shutoff_window(
            SimpleNamespace(sub_states={"gb": SimpleNamespace(band_info={})}), 3),
            (0, False))
        s = _sample(stamp=2)
        self.assertEqual(release_band_shutoff_window(s, None), (0, False))
        self.assertTrue(s.sub_states["gb"].band_info["band_rj_shutoff_w"].all())


class StepEntryTest(unittest.TestCase):

    def _step(self, profile=None):
        tree = [SimpleNamespace(moves=[])]
        st = SearchStageProfileStep(
            moves=tree, convergence_iter=2, plateau_branch="gb",
            profile=profile or {}, stage_name="gb_search_3")
        return st, tree

    def test_step_entry_releases_a_valve_earned_under_another_step(self):
        st, tree = self._step()
        s = _sample(stamp=2)
        st.setup_run(47, s, _FakeSampler(47, tree))
        st.note_recipe_step(3)
        bi = s.sub_states["gb"].band_info
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self.assertEqual(int(bi["band_shutoff_w_step"][0]), 3)

    def test_a_resume_inside_the_same_step_keeps_the_valve(self):
        st, tree = self._step()
        s = _sample(stamp=3)
        st.setup_run(47, s, _FakeSampler(47, tree))
        st.note_recipe_step(3)
        bi = s.sub_states["gb"].band_info
        self.assertTrue(bi["band_rj_shutoff_w"].all())
        self.assertTrue((bi["band_shutoff_streak_w"] == 3).all())

    def test_release_and_the_stage_3_max_reset_compose(self):
        st, tree = self._step({"reset_band_max": True})
        s = _sample(stamp=2)
        st.setup_run(47, s, _FakeSampler(47, tree))
        st.note_recipe_step(3)
        bi = s.sub_states["gb"].band_info
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self.assertTrue(np.all(np.isneginf(bi["band_cold_logl_max_w"])))


if __name__ == "__main__":
    unittest.main()
