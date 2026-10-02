"""Per-stage prior births for the shared ``rj_prior_removal`` move.

User ruling 2026-10-02: "set it specifically for gb search 3 ... prior removal
only for gb search 1." The prior RJ move is ONE object shared by every search
stage, built deaths-only in a v9 launch (GB_SEARCH_PRIOR_REMOVAL_ONLY=1), so
the stage profile now carries a MOVE-SCOPED ``prior_births`` key that the
stage step applies to that move alone on entry -- never broadcast, because
``rj_removal_only`` on rj_fstat_search means no births at all (the 09-29
hazard note). Table: gb_search_1 deaths only, gb_search_2 and 3 births and
deaths; ``GB_SEARCH_{N}_PRIOR_BIRTHS`` overrides a row.
"""

import contextlib
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "scripts", "fstat_proposal"))


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


def _gb(name, **kw):
    base = dict(name=name, branch_name="gb", opt_snr_rej_samp_limit=8.0,
                phase_maximize=False, _snr_lim_table=None, is_rj_prop=True,
                rj_removal_only=False, rj_replace=False)
    base.update(kw)
    return SimpleNamespace(**base)


class _FakeBackend:
    def __init__(self, iteration):
        self.iteration = iteration

    def get_nleaves(self, branch_names=None, temp_index=0):
        return {branch_names[0]: np.arange(self.iteration)[:, None]}

    def stage_start_iteration(self, name):
        return None


def _step(profile, tree):
    from lisatools.globalfit.recipe import SearchStageProfileStep

    st = SearchStageProfileStep(moves=tree, convergence_iter=2, plateau_branch="gb",
                                profile=profile, stage_name="gb_search_x")
    sampler = SimpleNamespace(backend=_FakeBackend(10), moves=tree, periodic=None,
                              temperature_control=None, weights=None)
    st.setup_run(10, None, sampler)
    st.note_recipe_step(1)
    return st


class ApplyProfilePriorBirthsTest(unittest.TestCase):

    def test_only_the_prior_move_changes_and_the_fstat_move_never_does(self):
        prior = _gb("rj_prior_removal", rj_removal_only=True)      # built deaths-only
        fstat = _gb("rj_fstat_search")
        warm = _gb("rj_warm_search")
        tree = [SimpleNamespace(moves=[warm, fstat, prior])]
        _step(dict(opt_snr=5.0, prior_births=True), tree)
        self.assertFalse(prior.rj_removal_only)                    # births AND deaths
        self.assertFalse(fstat.rj_removal_only)                    # untouched: births live
        self.assertFalse(warm.rj_removal_only)
        self.assertEqual(fstat.opt_snr_rej_samp_limit, 5.0)        # the broadcast keys still broadcast
        # the next stage's profile puts it back (the object is shared)
        _step(dict(opt_snr=8.0, prior_births=False), tree)
        self.assertTrue(prior.rj_removal_only)
        self.assertFalse(fstat.rj_removal_only)

    def test_profile_without_the_key_leaves_the_move_alone(self):
        prior = _gb("rj_prior_removal", rj_removal_only=True)
        tree = [SimpleNamespace(moves=[_gb("rj_fstat_search"), prior])]
        _step(dict(opt_snr=5.0), tree)
        self.assertTrue(prior.rj_removal_only)

    def test_a_replace_move_cannot_be_made_deaths_only(self):
        prior = _gb("rj_prior_removal", rj_removal_only=False, rj_replace=True)
        tree = [SimpleNamespace(moves=[prior])]
        with self.assertRaises(ValueError):
            _step(dict(prior_births=False), tree)

    def test_missing_prior_move_is_a_warning_not_an_error(self):
        tree = [SimpleNamespace(moves=[_gb("rj_fstat_search")])]
        with self.assertLogs("lisatools.globalfit.recipe", level="WARNING") as cm:
            _step(dict(prior_births=True), tree)
        self.assertTrue(any("rj_prior_removal" in line for line in cm.output))


class ProfileTableTest(unittest.TestCase):

    def test_table_stage_1_deaths_only_stages_2_and_3_births(self):
        import run_combined_staged as R

        with env(GB_SEARCH_1_PRIOR_BIRTHS=None, GB_SEARCH_2_PRIOR_BIRTHS=None,
                 GB_SEARCH_3_PRIOR_BIRTHS=None):
            rows = {n: p for n, p, _ in R.search_stage_profiles()}
        self.assertFalse(rows["gb_search_1"]["prior_births"])
        self.assertTrue(rows["gb_search_2"]["prior_births"])
        self.assertTrue(rows["gb_search_3"]["prior_births"])

    def test_per_stage_env_override(self):
        import run_combined_staged as R

        with env(GB_SEARCH_3_PRIOR_BIRTHS="0", GB_SEARCH_1_PRIOR_BIRTHS="1"):
            rows = {n: p for n, p, _ in R.search_stage_profiles()}
        self.assertFalse(rows["gb_search_3"]["prior_births"])
        self.assertTrue(rows["gb_search_1"]["prior_births"])
        with env(GB_SEARCH_3_PRIOR_BIRTHS="maybe"):
            with self.assertRaises(ValueError):
                R.search_stage_profiles()

    def test_the_built_stages_carry_the_key(self):
        import run_combined_staged as R

        base = dict(GB_SEARCH_IN_MODEL="1", GB_SEARCH_RJ_REPLACE="0",
                    GB_SEARCH_IN_MODEL_REPLACE="0",
                    GB_WARM_START_COMPONENTS="/nonexistent/warm.npz", STAGE_V9_SEARCH="1",
                    GB_SEARCH_3_WARM_EVERY="5", MBHB_IDS="2,5", STAGE_SKIP_SOURCE_SEARCH="1",
                    VGB_CHIRP_MASS_BASIS="1", PSD_START_PARAMS=None, GALFOR_START_PARAMS=None,
                    GB_SEARCH_1_PRIOR_BIRTHS=None, GB_SEARCH_2_PRIOR_BIRTHS=None,
                    GB_SEARCH_3_PRIOR_BIRTHS=None)
        with env(**base):
            fit = R.build_fit()
        by = {s.name: s for s in fit.recipe.stages}
        self.assertFalse(by["gb_search_1"].step_kwargs["profile"]["prior_births"])
        self.assertTrue(by["gb_search_2"].step_kwargs["profile"]["prior_births"])
        self.assertTrue(by["gb_search_3"].step_kwargs["profile"]["prior_births"])
        self.assertNotIn("prior_births", by["full_pe"].step_kwargs)


if __name__ == "__main__":
    unittest.main()
