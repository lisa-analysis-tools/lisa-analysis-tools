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
STAGE_ENVS = ("GB_ONLY", "STAGE_SKIP_NOISE", "STAGE_SKIP_SOURCE_SEARCH",
              "GB_SEARCH_SOURCE_EVERY",
              "STAGE_NOISE_ONLY",
              "STAGE_NOISE_VGB_PE", "COMBINED_SMOKE", "TOBS_TARGET",
              "GB_WARM_START_COMPONENTS", "REMOVE_BRANCHES",
              # 2026-09-16: =1 adds vgb_ridge_gibbs to the gb_search / PE
              # stage move lists, so every exact stage-list assertion below
              # needs it CLEARED rather than inherited from the ambient env
              # (VGBChirpRidgeWiringTest covers the =1 composition).
              "VGB_CHIRP_MASS_BASIS",
              # 2026-09-24: shapes the stage LIST itself, so an ambient
              # value would silently change every exact-list assertion here.
              "STAGE_V9_SEARCH", "GB_SEARCH_3_WARM_EVERY",
              "PSD_START_PARAMS", "GALFOR_START_PARAMS",
              "STAGE_FORCE_NOISE_SEARCH")


def _build_fit():
    spec = importlib.util.spec_from_file_location(
        "run_combined_staged_for_sources_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_fit()


#: The GB search stage names under the v9 three-stage restructure
#: (2026-09-24). ``STAGE_V9_SEARCH=0`` collapses them back to one
#: ``gb_search``; ``LegacySingleSearchStageTest`` below covers that path, and
#: everything else here runs against the DEFAULT composition, which is v9.
V9_SEARCH_STAGES = ("gb_search_1", "gb_search_2", "gb_search_3")

#: The FULL ordered search list, including ``gb_search_seed`` (prepended
#: 2026-09-26). Kept separate from ``V9_SEARCH_STAGES`` on purpose: the
#: tests that ITERATE stages are asserting per-stage move composition, and
#: the seed stage deliberately carries a reduced list (warm RJ + in-model
#: only, no F-stat, no sources), so it belongs in the ORDER assertions and
#: not in those.
V9_ALL_SEARCH_STAGES = ("gb_search_seed", *V9_SEARCH_STAGES)


def _everies(fit, stage_name):
    st = next(s for s in fit.recipe.stages if s.name == stage_name)
    return {m.name: getattr(m, "every", 1) for m in st.moves
            if m.name in ("sobbh_pe", "mbh_pe", "emri_pe")}


class LegacySingleSearchStageTest(unittest.TestCase):
    """``STAGE_V9_SEARCH=0`` restores the pre-2026-09-24 composition.

    The escape hatch has to keep working: it is what a bisect against the
    restructure would use, and what a run that only wants the v9 knobs
    (in-model convergence, caps off, fancy tempering off) WITHOUT the
    three-stage cycle would set.
    """

    def setUp(self):
        self._env = os.environ.copy()
        for k in SRC_ENVS + STAGE_ENVS:
            os.environ.pop(k, None)
        os.environ["STAGE_V9_SEARCH"] = "0"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_one_gb_search_stage_of_kind_rj(self):
        fit = _build_fit()
        self.assertEqual([st.name for st in fit.recipe.stages],
                         ["noise_search", "noise_vgb_search", "gb_search",
                          "full_pe"])
        st = next(s for s in fit.recipe.stages if s.name == "gb_search")
        self.assertEqual(st.kind, "rj")
        self.assertNotIn("profile", st.step_kwargs)

    def test_the_source_cadence_still_applies(self):
        os.environ.update(ALL_IDS)
        os.environ["STAGE_SKIP_SOURCE_SEARCH"] = "1"
        fit = _build_fit()
        self.assertEqual(_everies(fit, "gb_search"),
                         {"sobbh_pe": 5, "mbh_pe": 5, "emri_pe": 5})


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
            ["noise_search", "noise_vgb_search", *V9_ALL_SEARCH_STAGES,
             "full_pe"])
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
             *V9_ALL_SEARCH_STAGES, "full_pe"])
        st0 = fit.recipe.stages[0]
        self.assertEqual(st0.kind, "search")
        self.assertEqual([m.name for m in st0.moves],
                         ["source_joint_search"])
        # armed source PE moves ride gb_search and full_pe in the
        # full_year banking order (sobbh -> mbh -> emri). In gb_search
        # the mbh/emri moves carry a 1-in-N iteration CADENCE (user
        # ruling 2026-09-15: "run them in gb_search as before, but make
        # them run every 10 iterations" -- their dense rows cost minutes
        # per pass at near-zero GPU); sobbh's cheap chunked-het rows run
        # every iteration, and full_pe runs everything uncadenced.
        stages = self._stages(fit)
        for stage in (*V9_SEARCH_STAGES, "full_pe"):
            names = stages[stage]
            sub = [n for n in names
                   if n in ("sobbh_pe", "mbh_pe", "emri_pe")]
            self.assertEqual(sub, ["sobbh_pe", "mbh_pe", "emri_pe"],
                             f"{stage}: {names}")
        for stage in V9_SEARCH_STAGES:
            self.assertIn("rj_fstat_search", stages[stage])
        self.assertIn("rj_fstat_pe", stages["full_pe"])
        # User ruling 2026-09-18: ALL THREE armed source branches ride the
        # gb_search cadence, default every 5. sobbh joined because the
        # 4-GPU run measured it at 250 s/propose -- 43% of an iteration and
        # the largest single cost in the stage -- against the 09-15
        # assumption that its chunked-het rows were cheap. full_pe is never
        # cadenced.
        for stage in V9_SEARCH_STAGES:
            self.assertEqual(_everies(fit, stage),
                             {"sobbh_pe": 5, "mbh_pe": 5, "emri_pe": 5},
                             stage)
        self.assertEqual(_everies(fit, "full_pe"),
                         {"sobbh_pe": 1, "mbh_pe": 1, "emri_pe": 1})

    def test_gb_search_source_every_env_override(self):
        os.environ.update(ALL_IDS)
        os.environ["GB_SEARCH_SOURCE_EVERY"] = "7"
        fit = _build_fit()
        for stage in V9_SEARCH_STAGES:
            self.assertEqual(_everies(fit, stage),
                             {"sobbh_pe": 7, "mbh_pe": 7, "emri_pe": 7},
                             stage)
        # the override never leaks into full_pe
        self.assertEqual(_everies(fit, "full_pe"),
                         {"sobbh_pe": 1, "mbh_pe": 1, "emri_pe": 1})

    def test_gb_search_source_every_one_is_uncadenced(self):
        os.environ.update(ALL_IDS)
        os.environ["GB_SEARCH_SOURCE_EVERY"] = "1"
        fit = _build_fit()
        for stage in V9_SEARCH_STAGES:
            self.assertEqual(_everies(fit, stage),
                             {"sobbh_pe": 1, "mbh_pe": 1, "emri_pe": 1},
                             stage)

    def test_skip_source_search_keeps_moves_in_gb_stages(self):
        # User ruling 2026-09-14 late: with exact-truth starts
        # (*_START_FACTOR=0) there is nothing for the joint source search
        # to converge -- sources stay SUBTRACTED throughout (templates
        # ride the residual from setup) and their proposals run only in
        # gb_search + full_pe.
        os.environ.update(ALL_IDS)
        os.environ["STAGE_SKIP_SOURCE_SEARCH"] = "1"
        fit = _build_fit()
        stages = self._stages(fit)
        self.assertEqual(list(stages),
                         ["noise_search", "noise_vgb_search",
                          *V9_ALL_SEARCH_STAGES, "full_pe"])
        # user ruling 2026-09-18 (superseding 09-15's mbh/emri-only
        # 1-in-10): all three ride the GB search stages at a 1-in-5
        # cadence; all three in full_pe uncadenced.
        for mv in ("sobbh_pe", "mbh_pe", "emri_pe"):
            for stage in V9_SEARCH_STAGES:
                self.assertIn(mv, stages[stage], stage)
            self.assertIn(mv, stages["full_pe"])
        for stage in V9_SEARCH_STAGES:
            self.assertEqual(_everies(fit, stage),
                             {"sobbh_pe": 5, "mbh_pe": 5, "emri_pe": 5},
                             stage)

    def test_skip_source_search_without_sources_is_refused(self):
        # the flag has no stage to skip when nothing is armed -- refuse
        # rather than silently no-op (the STAGE_* flag convention).
        os.environ["STAGE_SKIP_SOURCE_SEARCH"] = "1"
        with self.assertRaises(ValueError):
            _build_fit()

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

    def test_synthetic_interim_no_vgb_composition(self):
        # the 6mo_v8_nogb SYNTHETIC interim (user 2026-09-14): vgb sits out
        # too (all_sources synthetic has no VGB stream), and the redundant
        # noise_vgb_search stage drops with it
        os.environ.update(ALL_IDS)
        os.environ["REMOVE_BRANCHES"] = "gb,galfor,vgb"
        os.environ["DATA_MODE"] = "synthetic"
        try:
            fit = _build_fit()
        finally:
            os.environ.pop("DATA_MODE", None)
        for b in ("gb", "galfor", "vgb"):
            self.assertNotIn(b, fit.branches)
        self.assertEqual(
            [st.name for st in fit.recipe.stages],
            ["source_search", "noise_search", "full_pe"])
        self.assertEqual(
            self._stages(fit)["full_pe"],
            ["psd_pe", "sobbh_pe", "mbh_pe", "emri_pe"])
        self.assertEqual(
            fit.general.source_types,
            ("NOISE", "SOBHB", "MBHB", "EMRI"))

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


class VGBChirpRidgeWiringTest(unittest.TestCase):
    """VGB_CHIRP_MASS_BASIS=1 adds ``vgb_ridge_gibbs`` to the stage lists.

    User ruling 2026-09-16 ("VGBs get the ridge-gibbs fiber move too"). The
    move is registered by ``recipe.build_vgb_moves`` ONLY when the vgb basis
    carries dist/Mc/fdot_astro_ratio, so the stage must request it only
    under the 6-column chirp basis -- requesting an unregistered stock name
    raises at ``Move.setup``. These assertions pin BOTH directions, since
    the 5-column default is what every other production script still runs.
    """

    def setUp(self):
        self._env = os.environ.copy()
        for k in SRC_ENVS + STAGE_ENVS:
            os.environ.pop(k, None)
        os.environ["NWALKERS"] = "4"
        os.environ.update(ALL_IDS)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _stages(self, fit):
        return {st.name: [m.name for m in st.moves]
                for st in fit.recipe.stages}

    def test_absent_on_the_five_column_default(self):
        stages = self._stages(_build_fit())
        for name, names in stages.items():
            self.assertNotIn("vgb_ridge_gibbs", names, f"{name}: {names}")

    def test_present_in_gb_search_and_full_pe_under_the_chirp_basis(self):
        os.environ["VGB_CHIRP_MASS_BASIS"] = "1"
        stages = self._stages(_build_fit())
        # exactly where gb_ridge_gibbs rides
        for stage_name in (*V9_SEARCH_STAGES, "full_pe"):
            self.assertIn("vgb_ridge_gibbs", stages[stage_name],
                          f"{stage_name}: {stages[stage_name]}")
            self.assertIn("gb_ridge_gibbs", stages[stage_name])
        # ... and nowhere else (search stages carry the joint criterion only)
        for name, names in stages.items():
            if name in (*V9_SEARCH_STAGES, "full_pe"):
                continue
            self.assertNotIn("vgb_ridge_gibbs", names, f"{name}: {names}")

    def test_never_inside_the_joint_noise_criterion(self):
        """It is a zero-likelihood move: a max-logL criterion must not own it."""
        os.environ["VGB_CHIRP_MASS_BASIS"] = "1"
        fit = _build_fit()
        for st in fit.recipe.stages:
            for mv in st.moves:
                inner = getattr(mv, "inner_names", None)
                if inner:
                    self.assertNotIn("vgb_ridge_gibbs", inner,
                                     f"{st.name}: {inner}")

    def test_gb_ridge_gibbs_kill_switch_takes_the_vgb_twin_with_it(self):
        os.environ["VGB_CHIRP_MASS_BASIS"] = "1"
        os.environ["GB_RIDGE_GIBBS"] = "0"
        try:
            stages = self._stages(_build_fit())
        finally:
            os.environ.pop("GB_RIDGE_GIBBS", None)
        for name, names in stages.items():
            self.assertNotIn("vgb_ridge_gibbs", names, f"{name}: {names}")
            self.assertNotIn("gb_ridge_gibbs", names, f"{name}: {names}")

    def test_absent_when_the_vgb_branch_is_removed(self):
        """No vgb branch -> no vgb MOVE, chirp basis or not.

        (The joint-search move keeps its ``noise_vgb_joint_search`` NAME
        either way; what shrinks is its ``inner_names``. So assert on the
        move names that actually belong to the branch.)
        """
        os.environ["VGB_CHIRP_MASS_BASIS"] = "1"
        os.environ["REMOVE_BRANCHES"] = "vgb"
        fit = _build_fit()
        self.assertNotIn("vgb", fit.branches)
        stages = self._stages(fit)
        for name, names in stages.items():
            for dead in ("vgb_ridge_gibbs", "vgb_pe"):
                self.assertNotIn(dead, names, f"{name}: {names}")
        for st in fit.recipe.stages:
            for mv in st.moves:
                inner = getattr(mv, "inner_names", None) or []
                self.assertNotIn("vgb_pe", inner, f"{st.name}: {inner}")


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


class CadenceFilterTest(unittest.TestCase):
    """GFCombineMove's stage-local 1-in-N sub-move cadence (Move(every=N),
    user ruling 2026-09-15: mbh/emri ride gb_search every 10 iterations)."""

    def _combine(self, everies):
        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        obj = GFCombineMove.__new__(GFCombineMove)
        obj.moves = [object() for _ in everies]
        obj.random_choice = False
        obj.weighted_cycle = False
        obj.gf_move_every = obj._validate_move_every(everies)
        obj._gf_cadence_idx = 0
        return obj

    def test_skip_pattern_over_a_cycle(self):
        c = self._combine([1, 3, 3])
        ran = []
        for _ in range(6):
            due, skipped = c._gf_cadence_plan(c.moves)
            ran.append([c.moves.index(m) for m in due])
        # move 0 every iteration; moves 1/2 on iterations 0 and 3 only
        self.assertEqual(ran, [[0, 1, 2], [0], [0],
                               [0, 1, 2], [0], [0]])

    def test_all_ones_is_a_passthrough(self):
        c = self._combine([1, 1])
        for _ in range(3):
            due, skipped = c._gf_cadence_plan(c.moves)
            self.assertIs(due, c.moves)
            self.assertIsNone(skipped)
        self.assertEqual(c._gf_cadence_idx, 0)  # counter untouched

    def test_validation(self):
        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        c = self._combine([1, 1])
        with self.assertRaises(ValueError):
            c._validate_move_every([1])          # length mismatch
        with self.assertRaises(ValueError):
            c._validate_move_every([0, 1])       # < 1
        c.weighted_cycle = True
        with self.assertRaises(ValueError):
            c._validate_move_every([1, 5])       # cadence + drawn cycle

    def test_move_descriptor_validates_every(self):
        from lisatools.globalfit.moves.globalfitmove import Move

        self.assertEqual(Move("x_pe", every=10).every, 10)
        with self.assertRaises(ValueError):
            Move("x_pe", every=0)
