"""The v9 three-stage GB search: composition, per-stage profile, gate 6.

Covers the 2026-09-24 restructure end to end at the level that does not need
data or a GPU:

* ``run_combined_staged.build_fit`` composes three ``gb_search_*`` stages
  running the sanctioned 7-slot cycle, plus the unchanged ``full_pe``;
* the per-stage PROFILE (phase max / opt SNR / F-stat peak floor) reaches the
  right moves and only those;
* GATE 6 -- a stage ends on the nleaves plateau AND "every OCCUPIED
  (walker, band) pair has shut off", composed, never replaced;
* the F-stat peak floor's in-code override and the forced fresh-epoch refit
  that has to accompany it;
* the noise START PIN and the conditional noise stages it gates.

Every test here is construction-level: no data load, no backend, no kernel.
"""

from __future__ import annotations

import contextlib
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scripts", "fstat_proposal"))

from lisatools.globalfit.recipe import (  # noqa: E402
    SearchStageProfileStep, Stage, band_shutoff_w_armed,
    band_shutoff_w_pending_total, force_fstat_refit, gb_moves_in_tree,
    iter_move_tree,
)
from lisatools.sampling.fstat_proposal import (  # noqa: E402
    fstat_peak_min_F, peak_min_F_override, set_peak_min_F_override,
)


@contextlib.contextmanager
def env(**kw):
    """Set/unset env vars for the duration of the block (``None`` unsets)."""
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


#: The env every composition test needs: a GB-only search recipe with the
#: in-model slots, replace and a (fake) warm-start npz all armed.
_BASE_ENV = dict(
    GB_ONLY="1",
    GB_SEARCH_IN_MODEL="1",
    GB_SEARCH_RJ_REPLACE="1",
    GB_WARM_START_COMPONENTS="/nonexistent/warm.npz",
    STAGE_V9_SEARCH="1",
    GB_SEARCH_3_WARM_EVERY="5",
)


def _build(**overrides):
    import run_combined_staged as R

    with env(**{**_BASE_ENV, **overrides}):
        return R.build_fit()


def _names(stage):
    return [m.name for m in stage.moves]


# ======================================================================
# 1. COMPOSITION
# ======================================================================

class StageCompositionTest(unittest.TestCase):
    """Three search stages, the sanctioned cycle, then full_pe."""

    def test_three_search_stages_then_full_pe(self):
        fit = _build()
        self.assertEqual(
            [s.name for s in fit.recipe.stages],
            ["gb_search_1", "gb_search_2", "gb_search_3", "full_pe"])
        self.assertEqual([s.kind for s in fit.recipe.stages],
                         ["gb_search", "gb_search", "gb_search", "pe"])

    def test_the_seven_slot_cycle_in_order(self):
        """The user's ordering, 2026-09-24, verbatim:

        warm -> in-model -> fstat -> in-model -> replace -> in-model
        -> prior-removal.

        Order is the whole point, and the cycle ENDS on the removal judge
        (user amendment 2026-09-24). Every birth and every swap gets a full
        in-model refinement pass BEFORE ``rj_prior_removal`` sees it -- a
        source still sitting at its birth or swap coordinates looks far more
        deletable than the same source after it has walked onto its peak.
        """
        fit = _build()
        for st in fit.recipe.stages[:3]:
            gb = [n for n in _names(st) if n.startswith(("rj_", "in_model"))]
            self.assertEqual(
                gb,
                ["rj_warm_search", "in_model", "rj_fstat_search",
                 "in_model_fstat", "rj_replace", "in_model_replace",
                 "rj_prior_removal"],
                f"{st.name} cycle is wrong",
            )

    def test_three_distinct_in_model_slots_per_stage(self):
        """Distinct NAMES, because a stage's move names must be unique and
        because each slot's timing/acceptance must be attributable."""
        fit = _build()
        for st in fit.recipe.stages[:3]:
            slots = [n for n in _names(st) if n.startswith("in_model")]
            self.assertEqual(len(slots), 3)
            self.assertEqual(len(set(slots)), 3)

    def test_recipe_accepts_the_duplicated_move_names_across_stages(self):
        """``_check_unique`` is per-stage; the same stock move legitimately
        recurs in all three. If this ever became global the whole restructure
        would fail at composition."""
        fit = _build()  # Recipe() runs _check_unique in its ctor
        self.assertEqual(len(fit.recipe.stages), 4)

    def test_warm_start_cadence_is_stage_3_only(self):
        fit = _build(GB_SEARCH_3_WARM_EVERY="7")
        got = [
            next(m.every for m in st.moves if m.name == "rj_warm_search")
            for st in fit.recipe.stages[:3]
        ]
        self.assertEqual(got, [1, 1, 7])

    def test_warm_cadence_must_be_positive(self):
        with self.assertRaises(ValueError):
            _build(GB_SEARCH_3_WARM_EVERY="0")

    def test_in_model_slots_absent_when_the_move_is_not_built(self):
        """``build_gb_moves`` registers the slots only under
        GB_SEARCH_IN_MODEL=1, and a listed-but-unbuilt move fails recipe
        materialization -- so the descriptors must be knob-conditional."""
        fit = _build(GB_SEARCH_IN_MODEL="0")
        for st in fit.recipe.stages[:3]:
            self.assertEqual(
                [n for n in _names(st) if n.startswith("in_model")], [])

    def test_replace_absent_when_disabled(self):
        fit = _build(GB_SEARCH_RJ_REPLACE="0")
        for st in fit.recipe.stages[:3]:
            self.assertNotIn("rj_replace", _names(st))

    def test_stage_v9_search_0_restores_the_legacy_single_stage(self):
        fit = _build(STAGE_V9_SEARCH="0")
        self.assertEqual([s.name for s in fit.recipe.stages],
                         ["gb_search", "full_pe"])
        self.assertEqual(fit.recipe.stages[0].kind, "rj")

    def test_full_pe_is_unchanged_by_the_restructure(self):
        """The user's spec: ``full_pe`` unchanged. Compare the v9 and legacy
        compositions directly rather than asserting a hardcoded list."""
        v9 = _build().recipe.stages[-1]
        legacy = _build(STAGE_V9_SEARCH="0").recipe.stages[-1]
        self.assertEqual(v9.name, legacy.name)
        self.assertEqual(v9.kind, legacy.kind)
        self.assertEqual(_names(v9), _names(legacy))


class FullCompositionTest(unittest.TestCase):
    """The all_sources composition (noise + vgb + source branches)."""

    def _full(self, **ov):
        import run_combined_staged as R

        base = dict(_BASE_ENV)
        base.pop("GB_ONLY")
        base.update(MBHB_IDS="2,5", STAGE_SKIP_SOURCE_SEARCH="1",
                    VGB_CHIRP_MASS_BASIS="1")
        with env(**{**base, **ov}):
            return R.build_fit()

    def test_noise_moves_only_in_stage_3(self):
        """'Fixed noise' IS the absence of the psd/galfor moves: an unsampled
        branch does not move, and setup_acs rebuilds each walker's
        sensitivity from the state coords every pass."""
        fit = self._full(PSD_START_PARAMS=None, GALFOR_START_PARAMS=None)
        by = {s.name: _names(s) for s in fit.recipe.stages}
        for s in ("gb_search_1", "gb_search_2"):
            self.assertNotIn("noise_vgb_joint_search", by[s], s)
            self.assertNotIn("noise_joint_search_1", by[s], s)
            self.assertNotIn("psd_pe", by[s], s)
            self.assertNotIn("galfor_pe", by[s], s)
        self.assertIn("noise_vgb_joint_search", by["gb_search_3"])
        self.assertIn("noise_joint_search_1", by["gb_search_3"])
        self.assertIn("noise_joint_search_2", by["gb_search_3"])

    def test_vgb_keeps_sampling_in_the_fixed_noise_stages(self):
        """The VGBs are 55 KNOWN sources and have nothing to do with the
        noise model -- freezing them alongside it would leave their power in
        the residual for two whole stages."""
        fit = self._full(PSD_START_PARAMS=None, GALFOR_START_PARAMS=None)
        by = {s.name: _names(s) for s in fit.recipe.stages}
        self.assertIn("vgb_pe", by["gb_search_1"])
        self.assertIn("vgb_pe", by["gb_search_2"])

    def test_noise_stages_are_SKIPPED_when_the_noise_is_pinned(self):
        fit = self._full(PSD_START_PARAMS="1.5e-11,3e-15",
                         GALFOR_START_PARAMS="1e-44,1e-3,1.5,5e-4,5e-4")
        self.assertEqual([s.name for s in fit.recipe.stages],
                         ["gb_search_1", "gb_search_2", "gb_search_3",
                          "full_pe"])

    def test_noise_stages_are_KEPT_when_no_pin_is_supplied(self):
        fit = self._full(PSD_START_PARAMS=None, GALFOR_START_PARAMS=None)
        self.assertIn("noise_search", [s.name for s in fit.recipe.stages])

    def test_a_HALF_pin_keeps_the_noise_stages(self):
        """⚠ The dangerous case. psd seeded, galfor prior-drawn, noise stages
        skipped => gb_search_1, which does not sample noise at all, would run
        for a whole stage against a RANDOM foreground."""
        fit = self._full(PSD_START_PARAMS="1.5e-11,3e-15",
                         GALFOR_START_PARAMS=None)
        self.assertIn("noise_search", [s.name for s in fit.recipe.stages])

    def test_force_override_runs_the_noise_stages_anyway(self):
        fit = self._full(PSD_START_PARAMS="1.5e-11,3e-15",
                         GALFOR_START_PARAMS="1e-44,1e-3,1.5,5e-4,5e-4",
                         STAGE_FORCE_NOISE_SEARCH="1")
        self.assertIn("noise_search", [s.name for s in fit.recipe.stages])


# ======================================================================
# 2. THE PER-STAGE PROFILE
# ======================================================================

class _FakeGBMove:
    """Minimal stand-in for a GB band move, as the profile sees it."""

    branch_name = "gb"

    def __init__(self, name, *, is_rj_prop=True, snr=8.0, pm=False,
                 fstat=False):
        self.name = name
        self.is_rj_prop = is_rj_prop
        self.opt_snr_rej_samp_limit = snr
        self.phase_maximize = pm
        self._snr_lim_table = None
        self._shutoff_w_pending = None
        self.armed = []          # (serial, reason) for each forced refit
        if fstat:
            self.arm_fstat_refit = lambda s, r="": self.armed.append((s, r))


class _FakeVGBMove(_FakeGBMove):
    branch_name = "vgb"


class _FakeCombine:
    def __init__(self, moves):
        self.moves = list(moves)


class ProfileDeclarationTest(unittest.TestCase):

    def test_profile_values_per_stage(self):
        fit = _build()
        got = {s.name: s.step_kwargs["profile"] for s in fit.recipe.stages[:3]}
        self.assertEqual(got["gb_search_1"], dict(
            phase_maximize=True, opt_snr=8.0, peak_min_snr=8.0))
        self.assertEqual(got["gb_search_2"], dict(
            phase_maximize=False, opt_snr=5.0, peak_min_snr=6.25))
        self.assertEqual(got["gb_search_3"], dict(
            phase_maximize=False, opt_snr=5.0, peak_min_snr=6.25))

    def test_stage_name_is_declared_for_the_log_lines(self):
        fit = _build()
        for s in fit.recipe.stages[:3]:
            self.assertEqual(s.step_kwargs["stage_name"], s.name)

    def test_gb_only_and_full_declare_the_SAME_profiles(self):
        """Both compositions read V9_SEARCH_STAGE_PROFILES, so the GB-only
        probe scripts genuinely test the production cycle."""
        import run_combined_staged as R

        gb_only = {s.name: s.step_kwargs["profile"]
                   for s in _build().recipe.stages[:3]}
        table = {n: p for n, p, _ in R.V9_SEARCH_STAGE_PROFILES}
        self.assertEqual(gb_only, table)


class ReviewFixesTest(unittest.TestCase):
    """Regressions for the CONFIRMED findings of the 2026-09-24 code review.

    Every one of these was silent: the run would have produced plausible
    output with the feature quietly wrong.
    """

    def tearDown(self):
        set_peak_min_F_override(None)

    # -- the profile must not re-arm rj_replace's phase max ---------------
    def test_the_profile_LEAVES_rj_replace_phase_max_alone(self):
        """⚠ rj_replace is built ``phase_maximize=False`` because that
        acceptance WAS the root-caused lnL drift that retired it. The stage-1
        profile setting it True would re-arm the one thing the move is on
        probation for. Its own scoring switch (GB_REPLACE_PHASE_MAX) has
        rotation-on-accept behind it -- a different mechanism."""
        rep = _FakeGBMove("rj_replace", pm=False)
        rep.rj_replace = True
        other = _FakeGBMove("rj_fstat_search", pm=False)
        st = SearchStageProfileStep(
            moves=[_FakeCombine([rep, other])],
            profile=dict(phase_maximize=True), stage_name="gb_search_1")
        st.note_recipe_step(1)
        self.assertFalse(rep.phase_maximize, "rj_replace was re-armed")
        self.assertTrue(other.phase_maximize)

    def test_rj_prior_removal_DOES_follow_the_profile(self):
        """It is not an exception: user ruling 2026-09-02, "phase
        maximization on for the prior removal just like fstat"."""
        rm = _FakeGBMove("rj_prior_removal", pm=False)
        st = SearchStageProfileStep(
            moves=[_FakeCombine([rm])],
            profile=dict(phase_maximize=True), stage_name="gb_search_1")
        st.note_recipe_step(1)
        self.assertTrue(rm.phase_maximize)

    # -- the pending count is per VALVE, not per move ---------------------
    def test_pending_counts_each_shared_valve_ONCE(self):
        """⚠ Four armed RJ moves all publish against the SAME shared
        band_info table. Summing naively reported 4x the pending pairs. The
        `== 0` gate survived that (4x0 is 0), which is why it went unnoticed
        -- the operator-facing number and the monitor trace did not."""
        table = np.zeros((2, 3), dtype=bool)
        moves = []
        for n in ("rj_warm_search", "rj_fstat_search", "rj_replace",
                  "rj_prior_removal"):
            m = _FakeGBMove(n)
            m._rj_band_shutoff_w = table       # one shared valve
            m._shutoff_w_pending = 7
            moves.append(m)
        self.assertEqual(
            band_shutoff_w_pending_total([_FakeCombine(moves)]), 7)

    def test_genuinely_independent_valves_still_add(self):
        a, b = _FakeGBMove("a"), _FakeGBMove("b")
        a._rj_band_shutoff_w = np.zeros((2, 3), dtype=bool)
        b._rj_band_shutoff_w = np.zeros((2, 3), dtype=bool)
        a._shutoff_w_pending, b._shutoff_w_pending = 7, 3
        self.assertEqual(
            band_shutoff_w_pending_total([_FakeCombine([a, b])]), 10)

    def test_zero_pending_still_reads_as_converged(self):
        table = np.zeros((2, 3), dtype=bool)
        moves = []
        for n in ("a", "b", "c"):
            m = _FakeGBMove(n)
            m._rj_band_shutoff_w = table
            m._shutoff_w_pending = 0
            moves.append(m)
        self.assertEqual(
            band_shutoff_w_pending_total([_FakeCombine(moves)]), 0)
        self.assertTrue(band_shutoff_w_armed([_FakeCombine(moves)]))

    # -- full_pe declares its own peak floor ------------------------------
    def test_full_pe_DECLARES_its_peak_floor(self):
        """⚠ The search stages install a PROCESS-GLOBAL override of the
        F-stat peak floor and nothing cleared it, so full_pe silently
        inherited gb_search_3's 6.25 while the script exported 8.0. 6.25 is
        what the design wants here -- but inheriting it by accident is not
        choosing it, and the next person to retune a search stage would have
        moved this one too."""
        import run_combined_staged as R

        fit = _build()
        pe = fit.recipe.stages[-1]
        self.assertEqual(pe.name, "full_pe")
        self.assertEqual(pe.step_kwargs["peak_min_snr"], R._PE_PEAK_MIN_SNR)
        self.assertEqual(pe.step_kwargs["stage_name"], "full_pe")

    def test_the_pe_step_applies_and_logs_the_floor(self):
        from lisatools.globalfit.recipe import PERecipeStep

        set_peak_min_F_override(6.25)             # as if stage 3 had run
        st = PERecipeStep(moves=[_FakeCombine([])], peak_min_snr=6.25,
                          stage_name="full_pe")
        st.note_recipe_step(9)
        self.assertAlmostEqual(peak_min_F_override(), 0.5 * 6.25 ** 2)

    def test_a_None_pe_floor_CLEARS_a_search_stage_override(self):
        from lisatools.globalfit.recipe import PERecipeStep

        set_peak_min_F_override(6.25)
        PERecipeStep(moves=[_FakeCombine([])], peak_min_snr=None,
                     stage_name="full_pe").note_recipe_step(9)
        self.assertIsNone(peak_min_F_override())

    # -- the pass identity must cross the wire ----------------------------
    def test_the_replace_pass_is_shipped_to_the_compute_ranks(self):
        """⚠ THE WORST ONE. ``_replace_pass_source`` is set by the pass loop
        in the HEAD process, but every compute rank holds its OWN move
        instance. Without the payload key, ranks 1..N-1 drew from the F-stat
        container on BOTH passes: the warm pass was a no-op on 3 of 4 walker
        blocks while the log said it ran, and the first block was sampled
        under a different proposal from the rest. Pinned at the source level
        because exercising it needs a real MPI fan-out."""
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch as g

        src = inspect.getsource(g)
        self.assertIn('"replace_pass_source": getattr(', src,
                      "the pass source is not put INTO the rank payload")
        self.assertIn('self._replace_pass_source = (payload or {}).get(\n'
                      '                "replace_pass_source", None)', src,
                      "the rank body does not READ the pass source back")

    def test_the_stage_name_is_restamped_per_propose(self):
        """⚠ Stage.setup stamps gf_stage_name ONCE at materialization, but
        every stage is materialized up front and the GB moves are SHARED --
        so the static stamp was whichever stage was built last, and every
        stage-labelled line during gb_search_1 read 'gb_search_3'."""
        import inspect

        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        src = inspect.getsource(GFCombineMove._gf_precondition)
        self.assertIn("gf_stage_name", src,
                      "the per-propose preamble does not re-stamp the "
                      "stage NAME, only the kind")


class RuntimeStageKindTest(unittest.TestCase):
    """⚠ A NEW STAGE KIND MUST NOT FALL THROUGH THE LITERAL KIND TESTS.

    ``gf_stage_kind`` is compared against literal tuples in several places --
    ``psdmove._coarse_mode`` resolves "auto" to ``search_approx`` only for
    ``("search", "rj")``, and the delayed-acceptance PE path keys on
    ``== "pe"``. Adding ``"gb_search"`` as a fourth kind did not extend those
    tests, it fell THROUGH them: the v9 search stages would silently have
    resolved the coarse noise likelihood to the PE behaviour in the stages
    where the wall time is. ``Stage.runtime_kind`` maps the specialization
    back to the kind it behaves as, fixing every such site at once.

    v9 exports ``COARSE_GPU_MODE=off``, so that particular resolution was
    inert -- which is exactly why it would not have been noticed.
    """

    def test_gb_search_behaves_as_rj(self):
        self.assertEqual(Stage("s", kind="gb_search").runtime_kind, "rj")

    def test_every_other_kind_is_its_own_runtime_kind(self):
        for k in ("search", "pe", "rj"):
            self.assertEqual(Stage("s", kind=k).runtime_kind, k)

    def test_the_declared_kind_is_preserved(self):
        """``Stage.kind`` still picks the step class and is what the recipe
        prints -- only the stamp changes."""
        st = Stage("s", kind="gb_search")
        self.assertEqual(st.kind, "gb_search")

    def test_the_coarse_resolver_accepts_the_runtime_kind(self):
        """Pin the actual downstream test rather than trusting the mapping:
        whatever tuple ``psdmove`` compares against must contain it."""
        src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "lisatools", "globalfit", "moves", "psdmove.py")).read()
        self.assertIn('kind in ("search", "rj")', src)
        self.assertEqual(Stage("s", kind="gb_search").runtime_kind, "rj")


class ProfileApplicationTest(unittest.TestCase):

    def setUp(self):
        self._prev = set_peak_min_F_override(None)

    def tearDown(self):
        set_peak_min_F_override(None)

    def _step(self, profile, moves, name="gb_search_1"):
        st = SearchStageProfileStep(moves=[_FakeCombine(moves)],
                                    profile=profile, stage_name=name)
        return st

    def test_opt_snr_reaches_every_gb_move_including_in_model(self):
        """It is the SNR PRIOR BOUNDARY, not a per-move heuristic: a birth
        move and an in-model move disagreeing about where the prior ends
        would let a source be walked into a region the move that birthed it
        treats as zero-prior."""
        rj = _FakeGBMove("rj_fstat_search", snr=8.0)
        inm = _FakeGBMove("in_model", is_rj_prop=False, snr=8.0)
        st = self._step(dict(opt_snr=5.0), [rj, inm])
        st.note_recipe_step(1)
        self.assertEqual(rj.opt_snr_rej_samp_limit, 5.0)
        self.assertEqual(inm.opt_snr_rej_samp_limit, 5.0)

    def test_phase_max_reaches_RJ_moves_ONLY(self):
        """The pure in-model move is constructed phase_maximize=False on
        purpose -- in-model scoring is at the ACTUAL phase, and phase
        maximization is a BIRTH heuristic, not a stage choice."""
        rj = _FakeGBMove("rj_fstat_search", pm=False)
        inm = _FakeGBMove("in_model", is_rj_prop=False, pm=False)
        st = self._step(dict(phase_maximize=True), [rj, inm])
        st.note_recipe_step(1)
        self.assertTrue(rj.phase_maximize)
        self.assertFalse(inm.phase_maximize)

    def test_vgb_moves_are_NEVER_touched(self):
        """⚠ VGB carries the same attribute name with its own value
        (VGB_OPT_SNR_LIMIT, default 0.0 = off). Writing the GB floor onto it
        would silently arm an SNR gate on 55 KNOWN sources."""
        gb = _FakeGBMove("rj_fstat_search", snr=8.0)
        vgb = _FakeVGBMove("vgb_pe", snr=0.0)
        st = self._step(dict(opt_snr=5.0, phase_maximize=True), [gb, vgb])
        st.note_recipe_step(1)
        self.assertEqual(vgb.opt_snr_rej_samp_limit, 0.0)
        self.assertFalse(vgb.phase_maximize)

    def test_a_None_profile_entry_leaves_the_knob_alone(self):
        rj = _FakeGBMove("rj_fstat_search", snr=8.0, pm=True)
        st = self._step(dict(opt_snr=None, phase_maximize=None), [rj])
        st.note_recipe_step(1)
        self.assertEqual(rj.opt_snr_rej_samp_limit, 8.0)
        self.assertTrue(rj.phase_maximize)

    def test_unknown_profile_key_is_REFUSED(self):
        """A misspelled key would otherwise be silently ignored -- which is
        exactly the failure mode this whole feature has to avoid."""
        with self.assertRaises(ValueError):
            SearchStageProfileStep(moves=[_FakeCombine([])],
                                   profile=dict(phase_max=True),
                                   stage_name="x")

    def test_peak_floor_installs_the_in_code_override(self):
        rj = _FakeGBMove("rj_fstat_search", fstat=True)
        st = self._step(dict(peak_min_snr=6.25), [rj])
        st.note_recipe_step(2)
        self.assertAlmostEqual(peak_min_F_override(), 0.5 * 6.25 ** 2)
        self.assertAlmostEqual(fstat_peak_min_F(), 0.5 * 6.25 ** 2)

    def test_the_override_BEATS_the_environment(self):
        """The env value is the RUN's floor, set once in the submit script;
        the override is the STAGE's, and the stage is the later, narrower
        statement."""
        with env(FSTAT_PEAK_MIN_SNR="8.0"):
            self.assertAlmostEqual(fstat_peak_min_F(), 0.5 * 8.0 ** 2)
            set_peak_min_F_override(6.25)
            self.assertAlmostEqual(fstat_peak_min_F(), 0.5 * 6.25 ** 2)

    def test_clearing_the_override_restores_the_environment(self):
        with env(FSTAT_PEAK_MIN_SNR="8.0"):
            set_peak_min_F_override(6.25)
            set_peak_min_F_override(None)
            self.assertIsNone(peak_min_F_override())
            self.assertAlmostEqual(fstat_peak_min_F(), 0.5 * 8.0 ** 2)

    def test_a_peak_floor_CHANGE_forces_a_refit(self):
        rj = _FakeGBMove("rj_fstat_search", fstat=True)
        st = self._step(dict(peak_min_snr=6.25), [rj])
        st.note_recipe_step(2)
        self.assertEqual(len(rj.armed), 1)
        self.assertEqual(rj.armed[0][0], 2)

    def test_an_UNCHANGED_peak_floor_forces_nothing(self):
        """gb_search_3 keeps gb_search_2's floor; paying for a second full
        comb scan there would be pure waste."""
        rj = _FakeGBMove("rj_fstat_search", fstat=True)
        self._step(dict(peak_min_snr=6.25), [rj]).note_recipe_step(2)
        self.assertEqual(len(rj.armed), 1)
        self._step(dict(peak_min_snr=6.25), [rj], "gb_search_3"
                   ).note_recipe_step(3)
        self.assertEqual(len(rj.armed), 1)

    def test_the_profile_is_idempotent_per_recipe_step(self):
        """A mid-step resume re-announces the ACTIVE step. Re-applying must
        not buy a second forced refit."""
        rj = _FakeGBMove("rj_fstat_search", fstat=True)
        st = self._step(dict(peak_min_snr=6.25), [rj])
        st.note_recipe_step(2)
        st.note_recipe_step(2)
        st.note_recipe_step(2)
        self.assertEqual(len(rj.armed), 1)


class ForceRefitHelperTest(unittest.TestCase):

    def test_force_fstat_refit_walks_the_whole_tree(self):
        a = _FakeGBMove("rj_fstat_search", fstat=True)
        b = _FakeGBMove("rj_replace", fstat=True)
        plain = _FakeGBMove("in_model", is_rj_prop=False)
        n = force_fstat_refit([_FakeCombine([a, _FakeCombine([b, plain])])],
                              7, "why")
        self.assertEqual(n, 2)
        self.assertEqual(a.armed, [(7, "why")])
        self.assertEqual(b.armed, [(7, "why")])

    def test_returns_zero_with_no_grid_move(self):
        self.assertEqual(
            force_fstat_refit([_FakeCombine([_FakeGBMove("in_model")])], 1), 0)

    def test_iter_move_tree_unwraps_weighted_entries(self):
        """A weighted entry is ``(move, weight)``; a combine nests under
        ``moves``. The walker yields the combine itself too -- it is a move --
        so filter by the attribute the callers key on."""
        a, b = _FakeGBMove("a"), _FakeGBMove("b")
        got = [getattr(m, "name", "<combine>")
               for m in iter_move_tree([(a, 0.5), _FakeCombine([b])])]
        self.assertEqual(got, ["a", "<combine>", "b"])

    def test_gb_moves_in_tree_excludes_vgb(self):
        got = [m.name for m in gb_moves_in_tree(
            [_FakeCombine([_FakeGBMove("g"), _FakeVGBMove("v")])])]
        self.assertEqual(got, ["g"])


# ======================================================================
# 3. GATE 6 -- the stage stopping rule
# ======================================================================

class _FakeBackend:
    def __init__(self, nleaves, iteration):
        self._n = np.asarray(nleaves)
        self.iteration = iteration

    def get_nleaves(self, branch_names=None, temp_index=0):
        return {branch_names[0]: self._n}


class _FakeSampler:
    def __init__(self, nleaves, iteration, moves):
        self.backend = _FakeBackend(nleaves, iteration)
        self.moves = moves


class _ValveMove(_FakeGBMove):
    def __init__(self, name, pending):
        super().__init__(name)
        self._shutoff_w_pending = pending


class Gate6Test(unittest.TestCase):
    """Compose, never replace.

    ``pending_total`` counts OCCUPIED pairs only -- an EMPTY (walker, band)
    can never shut off, because a band that has found nothing has not
    plateaued, it has not started. So the literal "all pairs shut" is
    unreachable in any run with an empty band, i.e. every run, and the
    nleaves plateau stays composed in to cover them.
    """

    def _step(self, valve_pending):
        moves = ([_ValveMove("rj_fstat_search", valve_pending)]
                 if valve_pending is not None else [_FakeGBMove("x")])
        st = SearchStageProfileStep(
            moves=[_FakeCombine(moves)], convergence_iter=2,
            plateau_branch="gb", profile={}, stage_name="gb_search_1")
        st._stage_start_iter = 0
        return st, _FakeSampler(np.full((20, 1), 7), 20, [_FakeCombine(moves)])

    def _plateaued(self, st, sampler):
        """A plateau: the leaf count has not grown in the last window."""
        return st.stopping_function(20, None, sampler)

    def test_plateau_alone_stops_when_the_valve_is_NOT_armed(self):
        st, sampler = self._step(None)
        self.assertFalse(band_shutoff_w_armed(sampler.moves))
        self.assertTrue(self._plateaued(st, sampler))

    def test_plateau_is_HELD_while_occupied_pairs_are_still_open(self):
        st, sampler = self._step(4)
        self.assertTrue(band_shutoff_w_armed(sampler.moves))
        self.assertEqual(band_shutoff_w_pending_total(sampler.moves), 4)
        self.assertFalse(self._plateaued(st, sampler))

    def test_plateau_PLUS_zero_pending_advances(self):
        st, sampler = self._step(0)
        self.assertTrue(band_shutoff_w_armed(sampler.moves))
        self.assertTrue(self._plateaued(st, sampler))

    def test_zero_pending_does_NOT_advance_without_the_plateau(self):
        """The valve is a VETO on the plateau, not a replacement for it. A
        growing leaf count means the search is still finding sources."""
        moves = [_ValveMove("rj_fstat_search", 0)]
        st = SearchStageProfileStep(
            moves=[_FakeCombine(moves)], convergence_iter=2,
            plateau_branch="gb", profile={}, stage_name="s")
        st._stage_start_iter = 0
        growing = np.arange(1, 21).reshape(20, 1)
        sampler = _FakeSampler(growing, 20, [_FakeCombine(moves)])
        self.assertFalse(st.stopping_function(20, None, sampler))

    def test_the_armed_guard_is_load_bearing(self):
        """⚠ ``pending_total`` returns 0 BOTH when everything converged and
        when the feature is off. Without the armed guard the second reading
        would end every stage at its first check -- and it would look exactly
        like convergence in the log."""
        unarmed = [_FakeCombine([_FakeGBMove("rj_fstat_search")])]
        self.assertEqual(band_shutoff_w_pending_total(unarmed), 0)
        self.assertFalse(band_shutoff_w_armed(unarmed))

    def test_the_stage_never_ends_at_zero_leaves(self):
        """A search that has not found its FIRST source has not plateaued --
        it has not started. Inherited from RJRecipeStep and pinned here
        because gate 6 must not weaken it."""
        moves = [_ValveMove("rj_fstat_search", 0)]
        st = SearchStageProfileStep(
            moves=[_FakeCombine(moves)], convergence_iter=2,
            plateau_branch="gb", profile={}, stage_name="s")
        st._stage_start_iter = 0
        sampler = _FakeSampler(np.zeros((20, 1), dtype=int), 20,
                               [_FakeCombine(moves)])
        self.assertFalse(st.stopping_function(20, None, sampler))


# ======================================================================
# 4. THE NOISE START PIN
# ======================================================================

class NoisePinBasisTest(unittest.TestCase):
    """Physical in, sampling basis out -- in both directions."""

    def test_extractor_detects_a_LINEAR_source(self):
        from lisatools.globalfit.warmstart.noise_pin import _to_physical

        vals, basis = _to_physical([1.5e-11, 3e-15], "psd", "auto")
        self.assertEqual(basis, "linear")
        np.testing.assert_allclose(vals, [1.5e-11, 3e-15])

    def test_extractor_detects_an_LN_source_for_psd(self):
        from lisatools.globalfit.warmstart.noise_pin import _to_physical

        vals, basis = _to_physical(np.log([1.5e-11, 3e-15]), "psd", "auto")
        self.assertEqual(basis, "log")
        np.testing.assert_allclose(vals, [1.5e-11, 3e-15], rtol=1e-12)

    def test_extractor_detects_a_LOG10_source_for_galfor_and_keeps_alpha(self):
        from lisatools.globalfit.warmstart.noise_pin import _to_physical

        phys = np.array([1e-44, 1e-3, 1.7, 5e-4, 5e-4])
        stored = phys.copy()
        stored[[0, 1, 3, 4]] = np.log10(phys[[0, 1, 3, 4]])
        vals, basis = _to_physical(stored, "galfor", "auto")
        self.assertEqual(basis, "log")
        np.testing.assert_allclose(vals, phys, rtol=1e-12)
        self.assertAlmostEqual(vals[2], 1.7)  # alpha untouched

    def test_out_of_window_values_are_REFUSED(self):
        from lisatools.globalfit.warmstart.noise_pin import (
            _PSD_WINDOW, _check)

        with self.assertRaises(SystemExit):
            _check([1.0, 1.0], _PSD_WINDOW, "PSD_START_PARAMS")


class NoisePinEndToEndTest(unittest.TestCase):
    """The CLI's whole path against a real (tiny) store.

    The unit tests above cover the basis conversion in isolation; this covers
    what they cannot -- that ``best_logl_noise`` picks the right ROW and
    WALKER, and that a LINEAR source store and a LOG one produce the SAME
    physical pin. That equality is the property the interchange format
    exists for: the seed run and the new run are allowed to sample in
    different bases.
    """

    PSD = [1.4669894e-11, 4.5655124e-15]
    GAL = [9.4438957e-45, 3.4227289e-03, 1.63, 5.1e-4, 7.2e-4]

    def _store(self, tmp, basis):
        import h5py

        nit, nw = 6, 4
        psd = np.asarray(self.PSD)
        gal = np.asarray(self.GAL)
        if basis == "log":
            psd = np.log(psd)
            gal = gal.copy()
            gal[[0, 1, 3, 4]] = np.log10(np.asarray(self.GAL)[[0, 1, 3, 4]])
        path = os.path.join(tmp, f"seed_{basis}.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("global_fit")
            ll = np.zeros((nit, 1, 1, nw))
            ll[:4] = -1000.0
            ll[3, 0, 0, 2] = -10.0        # THE maxlogL cold walker
            g.create_dataset("log_like", data=ll)
            c = g.create_group("chain")
            gb = np.zeros((nit, 1, 1, nw, 2, 9))
            gb[:4] = 1.0                  # nonzero gb slab == a valid row
            c.create_dataset("gb", data=gb)
            c.create_dataset("psd", data=np.broadcast_to(
                psd, (nit, 1, 1, nw, 1, 2)).copy())
            c.create_dataset("galfor", data=np.broadcast_to(
                gal, (nit, 1, 1, nw, 1, 5)).copy())
        return path

    def test_maxlogl_row_and_walker_are_picked(self):
        import tempfile

        from lisatools.globalfit.warmstart.noise_pin import (
            noise_pin_from_store)

        with tempfile.TemporaryDirectory() as tmp:
            pin = noise_pin_from_store(self._store(tmp, "linear"))
        self.assertEqual(pin["walker"], 2)
        self.assertEqual(pin["iteration"], 3)
        self.assertAlmostEqual(pin["log_like"], -10.0)

    def test_linear_and_log_sources_give_the_SAME_physical_pin(self):
        import tempfile

        from lisatools.globalfit.warmstart.noise_pin import (
            noise_pin_from_store)

        with tempfile.TemporaryDirectory() as tmp:
            lin = noise_pin_from_store(self._store(tmp, "linear"))
            log = noise_pin_from_store(self._store(tmp, "log"))
        self.assertEqual(lin["psd_basis"], "linear")
        self.assertEqual(log["psd_basis"], "log")
        np.testing.assert_allclose(lin["psd"], log["psd"], rtol=1e-9)
        np.testing.assert_allclose(lin["galfor"], log["galfor"], rtol=1e-9)
        np.testing.assert_allclose(lin["psd"], self.PSD, rtol=1e-9)
        np.testing.assert_allclose(lin["galfor"], self.GAL, rtol=1e-9)

    def test_an_OLD_PARAMETERIZATION_store_is_refused(self):
        """⚠ MEASURED 2026-09-24 against a real v7 store: its galfor came
        back f_1 ~ 2e5, f_2 ~ 4e3 against the current prior range 1e-5..1e-2.
        That is a different MODEL, not a different basis, and no conversion
        rescues it -- a pin from it would start the run at a foreground
        nothing in the current model can represent. The window check must
        refuse rather than emit it; the submit script then falls back to
        running the noise stages.
        """
        import tempfile

        from lisatools.globalfit.warmstart.noise_pin import (
            noise_pin_from_store)

        with tempfile.TemporaryDirectory() as tmp:
            path = self._store(tmp, "linear")
            import h5py

            with h5py.File(path, "a") as f:
                f["global_fit/chain/galfor"][..., 3] = 2.0e5
            with self.assertRaises(SystemExit):
                noise_pin_from_store(path)


class NoisePinSeedingTest(unittest.TestCase):
    """``run.py``'s side: physical env value -> sampling-basis start coords."""

    def _seeder(self, log_sampling, branch="psd"):
        from lisatools.globalfit.run import GlobalFit

        obj = GlobalFit.__new__(GlobalFit)
        obj.logger = types.SimpleNamespace(info=lambda *a, **k: None)
        obj.curr = types.SimpleNamespace(source_info={
            branch: types.SimpleNamespace(log_sampling=log_sampling)})
        return obj

    def test_linear_branch_takes_the_values_verbatim(self):
        obj = self._seeder(False)
        out = obj._seed_noise_start_coords(
            "psd", "1.5e-11,3e-15", np.zeros((2, 3, 1, 2)))
        self.assertEqual(out.shape, (2, 3, 1, 2))
        np.testing.assert_allclose(out[..., 0], 1.5e-11)
        np.testing.assert_allclose(out[..., 1], 3e-15)

    def test_log_sampled_psd_is_ln(self):
        obj = self._seeder(True)
        out = obj._seed_noise_start_coords(
            "psd", "1.5e-11,3e-15", np.zeros((1, 1, 1, 2)))
        np.testing.assert_allclose(out[0, 0, 0], np.log([1.5e-11, 3e-15]))

    def test_log_sampled_galfor_is_log10_except_alpha(self):
        obj = self._seeder(True, "galfor")
        out = obj._seed_noise_start_coords(
            "galfor", "1e-44,1e-3,1.7,5e-4,5e-4", np.zeros((1, 1, 1, 5)))
        got = out[0, 0, 0]
        np.testing.assert_allclose(
            got, [np.log10(1e-44), np.log10(1e-3), 1.7,
                  np.log10(5e-4), np.log10(5e-4)])

    def test_every_walker_and_rung_starts_at_the_SAME_point(self):
        """A pin is meant to BE a pin: START_FACTOR scatter would defeat the
        purpose of pinning a converged run's maxlogL point."""
        obj = self._seeder(False)
        out = obj._seed_noise_start_coords(
            "psd", "1.5e-11,3e-15", np.zeros((4, 6, 1, 2)))
        self.assertEqual(len(np.unique(out[..., 0])), 1)

    def test_wrong_length_is_refused(self):
        obj = self._seeder(False)
        with self.assertRaises(ValueError):
            obj._seed_noise_start_coords(
                "psd", "1.5e-11", np.zeros((1, 1, 1, 2)))

    def test_a_LOG_value_pasted_in_is_refused(self):
        """The knob's whole failure mode: a raw chain row from a log-sampled
        run pasted straight in. Silent, and costs a whole allocation."""
        obj = self._seeder(False)
        with self.assertRaises(ValueError) as cm:
            obj._seed_noise_start_coords(
                "psd", "-25.0,-33.4", np.zeros((1, 1, 1, 2)))
        self.assertIn("LINEAR", str(cm.exception))

    def test_non_numeric_is_refused(self):
        obj = self._seeder(False)
        with self.assertRaises(ValueError):
            obj._seed_noise_start_coords(
                "psd", "1.5e-11,abc", np.zeros((1, 1, 1, 2)))


# ======================================================================
# 5. THE GROUP-KNOB ENV NAMES (regression)
# ======================================================================

class GroupKnobEnvNameTest(unittest.TestCase):
    """⚠ REGRESSION GUARD, 2026-09-24 audit.

    The in-GROUP knobs were briefly resolved as ``GB_INMODEL_CONVERGE_GROUP*``
    while every docstring, every ``[GB_IMGROUP]`` log line and the v9 submit
    script said ``GB_INMODEL_GROUP*``. An unrecognized env var is SILENTLY
    IGNORED, so nothing failed -- the whole group rule was simply off in a run
    whose script, log header and drift guard all claimed it was on. No test
    covered the env NAME because the unit tests set the attributes directly,
    which is exactly how a name mismatch survives a green suite.
    """

    def _r(self, knob, default, cast):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _resolve_converge_knob)

        return _resolve_converge_knob("gb", knob, None, default, cast,
                                      family="INMODEL_GROUP")

    def test_the_documented_names_resolve(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _converge_cast_refill, _converge_cast_scale,
            _converge_cast_window)

        with env(GB_INMODEL_GROUP="1", GB_INMODEL_GROUP_ITERS="3",
                 GB_INMODEL_GROUP_DLL="4.0", GB_INMODEL_GROUP_SCALE="flat",
                 GB_INMODEL_GROUP_MAX_PASSES="20"):
            self.assertIs(self._r("", False, _converge_cast_refill), True)
            self.assertEqual(self._r("iters", 999, _converge_cast_window), 3)
            self.assertEqual(self._r("dll", -1.0, float), 4.0)
            self.assertEqual(self._r("scale", "per_source",
                                     _converge_cast_scale), "flat")
            self.assertEqual(self._r("max_passes", 999, int), 20)

    def test_the_group_family_does_NOT_shadow_the_converge_family(self):
        """``GB_INMODEL_CONVERGE`` (scope 1) and ``GB_INMODEL_GROUP``
        (scope 2) are siblings, not nested. Setting one must not move the
        other."""
        from lisatools.globalfit.moves.gbspecialstretch import (
            _converge_cast_mode, _converge_cast_refill, _resolve_converge_knob)

        with env(GB_INMODEL_GROUP="1", GB_INMODEL_CONVERGE=None):
            self.assertIs(self._r("", False, _converge_cast_refill), True)
            self.assertEqual(
                _resolve_converge_knob("gb", "", None, "off",
                                       _converge_cast_mode), "off")

    def test_the_old_spelling_is_still_honoured(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _converge_cast_window)

        with env(GB_INMODEL_GROUP_ITERS=None,
                 GB_INMODEL_CONVERGE_GROUP_ITERS="11"):
            self.assertEqual(self._r("iters", 999, _converge_cast_window), 11)

    def test_the_documented_spelling_WINS_over_the_old_one(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _converge_cast_window)

        with env(GB_INMODEL_GROUP_ITERS="3",
                 GB_INMODEL_CONVERGE_GROUP_ITERS="11"):
            self.assertEqual(self._r("iters", 999, _converge_cast_window), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
