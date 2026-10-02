"""full_pe declares its moves' in-model repeat counts, per branch, on entry.

User ruling 2026-10-02: "let's export in-model repeats of 25 for all sources
except emris for right now for PE. This includes VGBs and GBs." The built
moves are shared with the search stages, so this is a per-STAGE declaration
applied by the PE step on entry, not a process-global knob: sobbh / mbh keep
their search-stage counts until full_pe starts, emri and the noise moves are
never touched, and a move of a listed branch that carries neither repeat
attribute is reported rather than silently skipped.
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


def _tree():
    """A PE move tree: one move per family, with each family's own attribute."""
    sobbh = SimpleNamespace(name="sobbh_pe", branch_name="sobbh", num_repeats=10)
    mbh = SimpleNamespace(name="mbh_pe", branch_name="mbh", num_repeats=2)
    emri = SimpleNamespace(name="emri_pe", branch_name="emri", num_repeats=2)
    gb = SimpleNamespace(name="rj_fstat_pe", branch_name="gb", num_repeat_proposals=25,
                         opt_snr_rej_samp_limit=5.0)
    vgb = SimpleNamespace(name="vgb_pe", branch_name="vgb", num_repeat_proposals=1,
                          opt_snr_rej_samp_limit=0.0)
    psd = SimpleNamespace(name="psd_pe", branch_name="psd", num_repeats=10)
    combine = SimpleNamespace(moves=[psd, sobbh, mbh, emri, gb, vgb])
    return [combine], dict(sobbh=sobbh, mbh=mbh, emri=emri, gb=gb, vgb=vgb, psd=psd)


class ApplyInModelRepeatsTest(unittest.TestCase):

    def test_sets_each_familys_attribute_for_listed_branches_only(self):
        from lisatools.globalfit.recipe import apply_inmodel_repeats

        tree, m = _tree()
        # psd (a PSDMove-style num_repeats) included when listed, left when not
        changed = apply_inmodel_repeats(tree, {"psd": 25})
        self.assertEqual((m["psd"].num_repeats, changed), (25, {"psd_pe": (10, 25)}))
        m["psd"].num_repeats = 10
        changed = apply_inmodel_repeats(tree, {"gb": 25, "vgb": 25, "sobbh": 25, "mbh": 25})
        self.assertEqual(m["sobbh"].num_repeats, 25)
        self.assertEqual(m["mbh"].num_repeats, 25)
        self.assertEqual(m["vgb"].num_repeat_proposals, 25)
        self.assertEqual(m["gb"].num_repeat_proposals, 25)        # already 25: unchanged
        self.assertEqual(m["emri"].num_repeats, 2)                 # not listed
        self.assertEqual(m["psd"].num_repeats, 10)                 # not listed
        self.assertEqual(changed, {"sobbh_pe": (10, 25), "mbh_pe": (2, 25), "vgb_pe": (1, 25)})
        # idempotent
        self.assertEqual(apply_inmodel_repeats(tree, {"gb": 25, "vgb": 25, "sobbh": 25, "mbh": 25}), {})

    def test_a_listed_move_without_a_repeat_attribute_is_reported_not_skipped_silently(self):
        from lisatools.globalfit.recipe import apply_inmodel_repeats

        odd = SimpleNamespace(name="odd_pe", branch_name="sobbh")
        with self.assertLogs("lisatools.globalfit.recipe", level="WARNING") as cm:
            changed = apply_inmodel_repeats([SimpleNamespace(moves=[odd])], {"sobbh": 25})
        self.assertEqual(changed, {})
        self.assertTrue(any("odd_pe" in line for line in cm.output))

    def test_refuses_a_zero_or_negative_count(self):
        from lisatools.globalfit.recipe import apply_inmodel_repeats

        tree, _ = _tree()
        with self.assertRaises(ValueError):
            apply_inmodel_repeats(tree, {"gb": 0})


class PEStepAppliesRepeatsTest(unittest.TestCase):

    def test_pe_step_applies_its_declaration_on_entry_once_per_step(self):
        from lisatools.globalfit.recipe import PERecipeStep

        tree, m = _tree()
        st = PERecipeStep(moves=tree, peak_min_snr=6.25, stage_name="full_pe",
                          pe_repeats={"gb": 25, "vgb": 25, "sobbh": 25, "mbh": 25})
        st.note_recipe_step(7)
        self.assertEqual((m["sobbh"].num_repeats, m["mbh"].num_repeats,
                          m["vgb"].num_repeat_proposals, m["emri"].num_repeats), (25, 25, 25, 2))
        # a later change by hand survives a repeat announce of the SAME step
        m["sobbh"].num_repeats = 3
        st.note_recipe_step(7)
        self.assertEqual(m["sobbh"].num_repeats, 3)
        # a new step re-applies
        st.note_recipe_step(8)
        self.assertEqual(m["sobbh"].num_repeats, 25)

    def test_pe_step_without_a_declaration_touches_nothing(self):
        from lisatools.globalfit.recipe import PERecipeStep

        tree, m = _tree()
        st = PERecipeStep(moves=tree, peak_min_snr=6.25, stage_name="full_pe")
        st.note_recipe_step(7)
        self.assertEqual((m["sobbh"].num_repeats, m["vgb"].num_repeat_proposals), (10, 1))


class CompositionTest(unittest.TestCase):

    _BASE = dict(
        GB_SEARCH_IN_MODEL="1", GB_SEARCH_RJ_REPLACE="0", GB_SEARCH_IN_MODEL_REPLACE="0",
        GB_WARM_START_COMPONENTS="/nonexistent/warm.npz",
        STAGE_V9_SEARCH="1", GB_SEARCH_3_WARM_EVERY="5",
        MBHB_IDS="2,5", EMRI_IDS="0,1", SOBHB_IDS="0,1", STAGE_SKIP_SOURCE_SEARCH="1",
        VGB_CHIRP_MASS_BASIS="1", PSD_START_PARAMS=None, GALFOR_START_PARAMS=None,
    )

    def _full_pe(self, **ov):
        import run_combined_staged as R

        with env(**{**self._BASE, **ov}):
            fit = R.build_fit()
        return {s.name: s for s in fit.recipe.stages}["full_pe"]

    def test_knob_reaches_full_pe_and_only_full_pe(self):
        pe = self._full_pe(PE_INMODEL_REPEATS="25", PE_INMODEL_REPEATS_BRANCHES=None)
        # the general thinning factor: every branch full_pe samples except emri
        self.assertEqual(pe.step_kwargs["pe_repeats"],
                         {"gb": 25, "vgb": 25, "sobbh": 25, "mbh": 25, "psd": 25, "galfor": 25})
        self.assertEqual(pe.step_kwargs["peak_min_snr"], 6.25)
        pe = self._full_pe(PE_INMODEL_REPEATS="25", PE_INMODEL_REPEATS_BRANCHES="vgb, sobbh")
        self.assertEqual(pe.step_kwargs["pe_repeats"], {"vgb": 25, "sobbh": 25})
        pe = self._full_pe(PE_INMODEL_REPEATS=None)
        self.assertIsNone(pe.step_kwargs["pe_repeats"])
        with self.assertRaises(ValueError):
            self._full_pe(PE_INMODEL_REPEATS="0")

    def test_search_stages_carry_no_declaration(self):
        import run_combined_staged as R

        with env(**{**self._BASE, "PE_INMODEL_REPEATS": "25"}):
            fit = R.build_fit()
        for s in fit.recipe.stages:
            if s.name != "full_pe":
                self.assertNotIn("pe_repeats", s.step_kwargs or {})


class RJFlipFractionTest(unittest.TestCase):
    """User ruling 2026-10-02: "make sure for GBs, during full PE all RJ moves
    sample 0.1 of the available slots. This is the FRAC env variable." In a
    search-mode run the recipe BUILDS the PE RJ moves with the search fraction
    (1.0), so full_pe declares the PE value and applies it on entry."""

    @staticmethod
    def _tree():
        warm = SimpleNamespace(name="rj_warm_pe", branch_name="gb", is_rj_prop=True,
                               rj_flip_fraction=1.0, opt_snr_rej_samp_limit=5.0)
        fstat = SimpleNamespace(name="rj_fstat_pe", branch_name="gb", is_rj_prop=True,
                                rj_flip_fraction=1.0, opt_snr_rej_samp_limit=5.0)
        prior = SimpleNamespace(name="rj_prior_pe", branch_name="gb", is_rj_prop=True,
                                rj_flip_fraction=1.0, opt_snr_rej_samp_limit=5.0)
        inmodel = SimpleNamespace(name="gb_ridge_gibbs", branch_name="gb", is_rj_prop=False,
                                  rj_flip_fraction=1.0, opt_snr_rej_samp_limit=5.0)
        vgb = SimpleNamespace(name="vgb_pe", branch_name="vgb", is_rj_prop=False,
                              rj_flip_fraction=1.0, opt_snr_rej_samp_limit=0.0)
        return [SimpleNamespace(moves=[warm, fstat, prior, inmodel, vgb])], dict(
            warm=warm, fstat=fstat, prior=prior, inmodel=inmodel, vgb=vgb)

    def test_every_gb_rj_move_gets_the_fraction_and_nothing_else_does(self):
        from lisatools.globalfit.recipe import apply_rj_flip_fraction

        tree, m = self._tree()
        changed = apply_rj_flip_fraction(tree, 0.1)
        self.assertEqual((m["warm"].rj_flip_fraction, m["fstat"].rj_flip_fraction,
                          m["prior"].rj_flip_fraction), (0.1, 0.1, 0.1))
        self.assertEqual(m["inmodel"].rj_flip_fraction, 1.0)       # not an RJ move
        self.assertEqual(m["vgb"].rj_flip_fraction, 1.0)           # fixed-leaf branch
        self.assertEqual(set(changed), {"rj_warm_pe", "rj_fstat_pe", "rj_prior_pe"})
        self.assertEqual(apply_rj_flip_fraction(tree, 0.1), {})    # idempotent
        with self.assertRaises(ValueError):
            apply_rj_flip_fraction(tree, 0.0)

    def test_pe_step_applies_it_on_entry(self):
        from lisatools.globalfit.recipe import PERecipeStep

        tree, m = self._tree()
        st = PERecipeStep(moves=tree, peak_min_snr=6.25, stage_name="full_pe",
                          pe_rj_flip_fraction=0.1)
        st.note_recipe_step(3)
        self.assertEqual(m["fstat"].rj_flip_fraction, 0.1)
        with self.assertRaises(ValueError):
            PERecipeStep(moves=tree, stage_name="full_pe", pe_rj_flip_fraction=1.5)

    def test_the_launcher_knob_reaches_full_pe(self):
        import run_combined_staged as R

        base = dict(CompositionTest._BASE)
        with env(**{**base, "GB_PE_RJ_FLIP_FRACTION": "0.1", "GB_SEARCH_RJ_FLIP_FRACTION": "1.0"}):
            fit = R.build_fit()
        pe = {s.name: s for s in fit.recipe.stages}["full_pe"]
        self.assertEqual(pe.step_kwargs["pe_rj_flip_fraction"], 0.1)
        with env(**{**base, "GB_PE_RJ_FLIP_FRACTION": None}):
            fit = R.build_fit()
        pe = {s.name: s for s in fit.recipe.stages}["full_pe"]
        self.assertEqual(pe.step_kwargs["pe_rj_flip_fraction"], 0.2)     # the code default


if __name__ == "__main__":
    unittest.main()
