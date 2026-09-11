"""Wiring tests for MBHB/EMRI/SOBHB in the staged combined runner (6mo_v8).

Campaign gate S6 groundwork (docs/6mo-campaign.md: "Fold mbh/emri/sobbh back
into the staged recipe ... stage order: MBH search first, full_year
pattern"). All assertions run at the construction level via
``run_combined_staged.build_fit`` -- no data built, no materialization
(the WarmStartWiringTest pattern).

The env contract (user ruling 2026-09-02, "we are adding MBHB EMRI SOBHB"):
``MBHB_IDS`` / ``EMRI_IDS`` / ``SOBHB_IDS`` arm each source branch with its
mojito catalogue ids; an absent/empty env drops that branch (today's
behavior). When any branch is armed:

* a ``source_search`` stage (kind="search", joint max-lnL over the armed
  source PE moves) runs FIRST -- sources converge and subtract before the
  noise stages fit the PSD;
* the armed source PE moves join gb_search and full_pe (sobbh -> mbh ->
  emri, the full_year banking order);
* ``fit.general.mojito_source_ids`` carries the id lists (GB/VGB keys
  preserved) and ``source_types`` gains the armed classes' data streams.
"""
import importlib.util
import os
import unittest
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "fstat_proposal" / "run_combined_staged.py")

ALL_IDS = {"MBHB_IDS": "2,5,16,18",
           "EMRI_IDS": "0,1,2,3,4,5,6,7",
           "SOBHB_IDS": "0,1,2,3,4,5"}
SRC_ENVS = tuple(ALL_IDS) + ("SOURCE_TYPES",)
STAGE_ENVS = ("GB_ONLY", "STAGE_SKIP_NOISE", "STAGE_NOISE_ONLY",
              "STAGE_NOISE_VGB_PE", "COMBINED_SMOKE", "TOBS_TARGET",
              "GB_WARM_START_COMPONENTS", "REMOVE_BRANCHES")


def _build_fit():
    spec = importlib.util.spec_from_file_location(
        "run_combined_staged_for_sources_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_fit()


class StagedSourcesWiringTest(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.copy()
        for k in SRC_ENVS + STAGE_ENVS:
            os.environ.pop(k, None)
        os.environ["NWALKERS"] = "4"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _stages(self, fit):
        return {st.name: [m.name for m in st.moves]
                for st in fit.recipe.stages}

    def test_baseline_unchanged_without_id_envs(self):
        fit = _build_fit()
        self.assertEqual(
            [st.name for st in fit.recipe.stages],
            ["noise_search", "noise_vgb_search", "gb_search", "full_pe"])
        for b in ("mbh", "emri", "sobbh"):
            self.assertNotIn(b, fit.branches)
        for names in self._stages(fit).values():
            for mv in ("sobbh_pe", "mbh_pe", "emri_pe",
                       "source_joint_search"):
                self.assertNotIn(mv, names)
        self.assertEqual(fit.general.source_types, ("NOISE", "GB", "VGB"))

    def test_all_sources_armed(self):
        os.environ.update(ALL_IDS)
        fit = _build_fit()
        for b in ("mbh", "emri", "sobbh"):
            self.assertIn(b, fit.branches)
        # id lists land on the general settings; GB/VGB keys preserved
        ids = fit.general.mojito_source_ids
        self.assertEqual(ids["MBHB"], [2, 5, 16, 18])
        self.assertEqual(ids["EMRI"], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(ids["SOBHB"], [0, 1, 2, 3, 4, 5])
        self.assertIn("GB", ids)
        self.assertIn("VGB", ids)
        # data streams follow the armed classes
        for cls in ("MBHB", "EMRI", "SOBHB"):
            self.assertIn(cls, fit.general.source_types)
        # stage order: source_search FIRST (MBH search first ruling), then
        # the unchanged noise -> gb -> pe ladder
        self.assertEqual(
            [st.name for st in fit.recipe.stages],
            ["source_search", "noise_search", "noise_vgb_search",
             "gb_search", "full_pe"])
        st0 = fit.recipe.stages[0]
        self.assertEqual(st0.kind, "search")
        self.assertEqual([m.name for m in st0.moves],
                         ["source_joint_search"])
        # armed source PE moves ride gb_search and full_pe in the
        # full_year banking order (sobbh -> mbh -> emri)
        stages = self._stages(fit)
        for stage in ("gb_search", "full_pe"):
            names = stages[stage]
            sub = [n for n in names
                   if n in ("sobbh_pe", "mbh_pe", "emri_pe")]
            self.assertEqual(sub, ["sobbh_pe", "mbh_pe", "emri_pe"],
                             f"{stage}: {names}")
            self.assertIn("rj_fstat_search" if stage == "gb_search"
                          else "rj_fstat_pe", names)

    def test_partial_arming_mbh_only(self):
        os.environ["MBHB_IDS"] = "2,5"
        fit = _build_fit()
        self.assertIn("mbh", fit.branches)
        self.assertNotIn("emri", fit.branches)
        self.assertNotIn("sobbh", fit.branches)
        self.assertEqual(fit.general.mojito_source_ids["MBHB"], [2, 5])
        self.assertIn("MBHB", fit.general.source_types)
        self.assertNotIn("EMRI", fit.general.source_types)
        stages = self._stages(fit)
        self.assertEqual(
            [n for n in stages["full_pe"]
             if n in ("sobbh_pe", "mbh_pe", "emri_pe")],
            ["mbh_pe"])

    def test_explicit_source_types_wins(self):
        os.environ["MBHB_IDS"] = "2"
        os.environ["SOURCE_TYPES"] = "NOISE,GB,VGB,MBHB"
        fit = _build_fit()
        self.assertEqual(fit.general.source_types,
                         ("NOISE", "GB", "VGB", "MBHB"))

    def test_gb_only_rejects_sources(self):
        os.environ["GB_ONLY"] = "1"
        os.environ["MBHB_IDS"] = "2"
        with self.assertRaises(ValueError):
            _build_fit()


class RemoveBranchesWiringTest(unittest.TestCase):
    """REMOVE_BRANCHES (user ask 2026-09-11: '6mo for everything except the
    GBs and galfor -- we will not inject them'). Removed branches drop from
    the fit, their moves drop from every stage, the gb stages collapse to a
    single full_pe over what remains, and the DEFAULT injection stream list
    loses the removed classes ('not inject them'); explicit SOURCE_TYPES
    still wins."""

    def setUp(self):
        self._env = os.environ.copy()
        for k in SRC_ENVS + STAGE_ENVS:
            os.environ.pop(k, None)
        os.environ["NWALKERS"] = "4"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _stages(self, fit):
        return {st.name: [m.name for m in st.moves]
                for st in fit.recipe.stages}

    def test_no_gb_no_galfor_composition(self):
        os.environ.update(ALL_IDS)
        os.environ["REMOVE_BRANCHES"] = "gb,galfor"
        fit = _build_fit()
        for b in ("gb", "galfor"):
            self.assertNotIn(b, fit.branches)
        for b in ("psd", "vgb", "sobbh", "mbh", "emri"):
            self.assertIn(b, fit.branches)
        # no gb stages; everything that remains PE-samples in full_pe
        self.assertEqual(
            [st.name for st in fit.recipe.stages],
            ["source_search", "noise_search", "noise_vgb_search",
             "full_pe"])
        stages = self._stages(fit)
        self.assertEqual(
            stages["full_pe"],
            ["psd_pe", "sobbh_pe", "mbh_pe", "emri_pe", "vgb_pe"])
        # no galfor / gb move anywhere
        for name, names in stages.items():
            for mv in names:
                self.assertNotIn("galfor", mv, f"{name}: {names}")
                self.assertFalse(
                    mv.startswith("rj_") or mv.startswith("gb_"),
                    f"{name}: {names}")
        # the joint noise criteria shrink to the present branches
        st = {s.name: s for s in fit.recipe.stages}
        self.assertEqual(st["noise_search"].moves[0].inner_names,
                         ["psd_pe"])
        self.assertEqual(st["noise_vgb_search"].moves[0].inner_names,
                         ["psd_pe", "vgb_pe"])
        # 'we will not inject them': GB stream gone from the DEFAULT
        self.assertEqual(
            fit.general.source_types,
            ("NOISE", "VGB", "SOBHB", "MBHB", "EMRI"))

    def test_explicit_source_types_still_wins(self):
        os.environ["REMOVE_BRANCHES"] = "gb,galfor"
        os.environ["SOURCE_TYPES"] = "NOISE,GB,VGB"
        fit = _build_fit()
        self.assertEqual(fit.general.source_types, ("NOISE", "GB", "VGB"))

    def test_unknown_or_anchor_branch_rejected(self):
        for bad in ("psd", "nonsense"):
            os.environ["REMOVE_BRANCHES"] = bad
            with self.assertRaises(ValueError, msg=bad):
                _build_fit()

    def test_incompatible_with_gb_only(self):
        os.environ["GB_ONLY"] = "1"
        os.environ["REMOVE_BRANCHES"] = "galfor"
        with self.assertRaises(ValueError):
            _build_fit()


class WarmPhaseMaxSeedingTest(unittest.TestCase):
    """rj_warm_search phase-max follows GB_RJ_PHASE_MAXIMIZE (user ruling
    2026-09-02: 'built exactly like fstat_search, with the distribution
    switched'). The cycle invariants stay fixed: run_swaps False (one
    tempering move per iteration), leaf_cap_update False (rj_fstat_search
    is the designated cap updater)."""

    def setUp(self):
        self._env = os.environ.copy()
        os.environ.pop("GB_RJ_PHASE_MAXIMIZE", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_overrides_follow_env(self):
        from lisatools.globalfit.recipe import warm_search_move_overrides

        off = warm_search_move_overrides()
        self.assertFalse(off["phase_maximize"])
        os.environ["GB_RJ_PHASE_MAXIMIZE"] = "1"
        on = warm_search_move_overrides()
        self.assertTrue(on["phase_maximize"])
        for d in (off, on):
            self.assertFalse(d["run_swaps"])
            self.assertFalse(d["leaf_cap_update"])


class WarmPeOverridesTest(unittest.TestCase):
    """rj_warm_pe mirrors the fstat search<->pe deltas (user ruling
    2026-09-07): NO phase maximization in PE regardless of
    GB_RJ_PHASE_MAXIMIZE (the no-PE-maximization policy), swap-ENABLED
    under the PE tempering budget (GB_TEMPER_EVERY_PROPOSES, like
    rj_fstat_pe), and never a cap updater."""

    def setUp(self):
        self._env = os.environ.copy()
        for k in ("GB_RJ_PHASE_MAXIMIZE", "GB_TEMPER_EVERY_PROPOSES"):
            os.environ.pop(k, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_pe_overrides(self):
        from lisatools.globalfit.recipe import warm_pe_move_overrides

        os.environ["GB_RJ_PHASE_MAXIMIZE"] = "1"  # must NOT leak into PE
        d = warm_pe_move_overrides()
        self.assertFalse(d["phase_maximize"])
        self.assertTrue(d["run_swaps"])
        self.assertFalse(d["leaf_cap_update"])
        self.assertEqual(d["temper_every_proposes"], 3)  # PE budget default
        os.environ["GB_TEMPER_EVERY_PROPOSES"] = "5"
        self.assertEqual(warm_pe_move_overrides()["temper_every_proposes"], 5)


if __name__ == "__main__":
    unittest.main()
