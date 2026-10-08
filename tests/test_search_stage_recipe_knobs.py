"""Per-stage recipe knobs: the 9mo / 1yr v9 recipe (user ruling 2026-10-07).

"I want to start with 3 iterations of gb search seed (same as before but 3
iterations not 5, no other source moves). Then I want to have gb search 1 as is
currently but with no other source moves (snr limit 8, phase maximization,
prior removal only). Then GB search 2 with valves reset (snr limit default of
5, no phase maximization, prior can add and remove). Gb search 2 should include
the other sources every iteration. And it should allow the PSD and galfor to
vary normally ... After GB search 2 will be full pe. No replica pe and no
ratcheting."  Refined the same day: "during gb search 2 the galfor and psd
sampling should be required to converge before moving on to the other
proposals. There should only be one noise/galfor proposal though. Not multiple
times during the cycle. Just have the galfor/psd convergence proposal at the
end of the cycle."

The knobs (every one unset = today's composition, byte-identical; the running
6mo store's recipe cannot change, so the new recipe reaches the 1yr / 9mo
launchers only through their derive tables):

  GB_SEARCH_STAGES               numbered search stages composed (default 1,2,3)
  GB_SEARCH_{N}_SOURCE_EVERY     sobbh/mbh/emri cadence in stage N; 0 = none;
                                 unset = the global GB_SEARCH_SOURCE_EVERY
  GB_SEARCH_{N}_NOISE_MODE       fixed | interleaved | cycle_end; unset = the
                                 table (stage 3 interleaved, 1/2 fixed)
  GB_SEARCH_{N}_NOISE_MAX_ROUNDS the cycle-end convergence's round ceiling
  GB_SEARCH_{N}_RESET_VALVES     1 = reset the RJ valves at a FRESH stage entry

Construction level (build_fit: no data, no backend) except the step tests,
which drive SearchStageProfileStep with fake states.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import os
import pickle
import sys
import types
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "fstat_proposal"))

SRC = ("sobbh_pe", "mbh_pe", "emri_pe")
NOISE = {"psd_pe", "galfor_pe"}


@contextlib.contextmanager
def env(**kw):
    """Set/unset env vars for the block (``None`` unsets)."""
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


#: Every knob this module is about, cleared unless a test sets it.
_KNOBS = {k: None for k in (
    "GB_SEARCH_STAGES", "GB_SEARCH_SAMPLE_NOISE_ALL_STAGES", "GALFOR_RATCHET",
    "STAGE_REPLICA_PE", "GB_SEARCH_SOURCE_EVERY", "GB_SEARCH_SEED_ITERS",
    "NOISE_SEARCH_CHECKS", "GB_SEARCH_LEGS",
    *(f"GB_SEARCH_{n}_{s}" for n in (1, 2, 3)
      for s in ("SOURCE_EVERY", "NOISE_MODE", "NOISE_MAX_ROUNDS", "RESET_VALVES")))}

_GB_ONLY = dict(_KNOBS, GB_ONLY="1", GB_SEARCH_IN_MODEL="1", GB_SEARCH_RJ_REPLACE="0",
                GB_SEARCH_IN_MODEL_REPLACE="0", GB_WARM_START_COMPONENTS="/nonexistent/warm.npz",
                STAGE_V9_SEARCH="1", GB_SEARCH_3_WARM_EVERY="5")

#: the FULL composition with all three source branches armed and the noise pinned
#: (so the standalone noise stages are skipped, as in production)
_FULL = dict(_GB_ONLY, GB_ONLY=None, MBHB_IDS="2,5", EMRI_IDS="0", SOBHB_IDS="0",
             STAGE_SKIP_SOURCE_SEARCH="1", VGB_CHIRP_MASS_BASIS="1",
             PSD_START_PARAMS="1.5e-11,3e-15",
             GALFOR_START_PARAMS="1e-44,1e-3,1.5,5e-4,5e-4")

#: the 9mo / 1yr recipe, as the derive tables export it
_RECIPE_9MO = dict(GB_SEARCH_SEED_ITERS="3", GB_SEARCH_STAGES="1,2",
                   GB_SEARCH_1_SOURCE_EVERY="0", GB_SEARCH_2_SOURCE_EVERY="1",
                   GB_SEARCH_2_NOISE_MODE="cycle_end", GB_SEARCH_2_RESET_VALVES="1",
                   STAGE_REPLICA_PE="0")


def _R():
    import run_combined_staged as R
    return R


def _build(base, **over):
    with env(**{**base, **over}):
        return _R().build_fit()


def _stage(fit, name):
    return next(s for s in fit.recipe.stages if s.name == name)


def _names(stage):
    return [m.name for m in stage.moves]


def _noise_moves(stage):
    """Every move in the stage that proposes psd or galfor (directly or wrapped)."""
    return [m for m in stage.moves
            if m.name in NOISE or NOISE & set(getattr(m, "inner_names", ()) or ())]


def _clear_peak_floor():
    from lisatools.sampling.fstat_proposal import set_peak_min_F_override
    set_peak_min_F_override(None)


# ======================================================================
# 1. which numbered stages compose
# ======================================================================
class StageSelectionTest(unittest.TestCase):

    def test_unset_composes_all_three(self):
        names = [s.name for s in _build(_GB_ONLY).recipe.stages]
        self.assertEqual(names, ["gb_search_seed", "gb_search_1", "gb_search_2",
                                 "gb_search_3", "replica_pe", "full_pe"])

    def test_1_2_drops_stage_3_in_BOTH_assemblies(self):
        """Two v9 assemblies (GB_ONLY and the full composition): a knob wired into
        one only is the "knob resolves, consuming path never runs" shape."""
        for base in (_GB_ONLY, _FULL):
            names = [s.name for s in _build(base, GB_SEARCH_STAGES="1,2",
                                            STAGE_REPLICA_PE="0").recipe.stages]
            self.assertEqual(names, ["gb_search_seed", "gb_search_1", "gb_search_2",
                                     "full_pe"], base.get("GB_ONLY"))

    def test_the_kept_stages_keep_their_names_and_table_profiles(self):
        fit = _build(_GB_ONLY, GB_SEARCH_STAGES="1,2")
        table = {n: p for n, p, _ in _R().V9_SEARCH_STAGE_PROFILES}
        for n in ("gb_search_1", "gb_search_2"):
            self.assertEqual(_stage(fit, n).step_kwargs["profile"], table[n])

    def test_the_resolver_filters(self):
        with env(**{**_KNOBS, "GB_SEARCH_STAGES": "1,2"}):
            self.assertEqual([n for n, _, _ in _R().search_stage_profiles()],
                             ["gb_search_1", "gb_search_2"])
        with env(**{**_KNOBS, "GB_SEARCH_STAGES": ""}):
            self.assertEqual(len(_R().search_stage_profiles()), 3)   # empty = unset

    def test_bad_values_are_refused(self):
        for bad in ("4", "1,1", "2,1", "one", "0", "1,,2"):
            with env(**{**_KNOBS, "GB_SEARCH_STAGES": bad}), \
                    self.assertRaises(ValueError, msg=bad):
                _R().search_stage_profiles()

    def test_the_seed_needs_stage_1s_profile(self):
        """The seed takes gb_search_1's profile verbatim; with stage 1 dropped there
        is nothing to take it from, so the composition refuses."""
        with self.assertRaises(ValueError):
            _build(_GB_ONLY, GB_SEARCH_STAGES="2,3")
        names = [s.name for s in _build(_GB_ONLY, GB_SEARCH_STAGES="2,3",
                                        GB_SEARCH_SEED_ITERS="0").recipe.stages]
        self.assertEqual(names, ["gb_search_2", "gb_search_3", "replica_pe", "full_pe"])


# ======================================================================
# 2. per-stage source cadence
# ======================================================================
class SourceCadenceTest(unittest.TestCase):

    def tearDown(self):
        _clear_peak_floor()

    def test_unset_is_the_global_cadence(self):
        fit = _build(_FULL, GB_SEARCH_SOURCE_EVERY="7")
        for n in ("gb_search_1", "gb_search_2", "gb_search_3"):
            self.assertEqual({m.name: m.every for m in _stage(fit, n).moves if m.name in SRC},
                             dict.fromkeys(SRC, 7), n)

    def test_zero_removes_ONLY_the_source_moves(self):
        """Stage 1 with no other sources keeps vgb, the warm start and both ridge
        moves: only sobbh/mbh/emri leave, and the GB cycle is untouched."""
        base = _build(_FULL)
        fit = _build(_FULL, GB_SEARCH_1_SOURCE_EVERY="0")
        got, want = _names(_stage(fit, "gb_search_1")), _names(_stage(base, "gb_search_1"))
        self.assertEqual([n for n in got if n in SRC], [])
        self.assertEqual(got, [n for n in want if n not in SRC])
        for keep in ("vgb_pe", "vgb_ridge_gibbs", "rj_warm_search", "gb_ridge_gibbs",
                     "rj_fstat_search", "rj_prior_removal", "in_model", "in_model_fstat",
                     "in_model_removal"):
            self.assertIn(keep, got, keep)
        # the other stages keep the global cadence
        self.assertEqual({m.name: m.every for m in _stage(fit, "gb_search_2").moves
                          if m.name in SRC}, dict.fromkeys(SRC, 5))

    def test_one_runs_them_EVERY_iteration(self):
        """every=1 on every move = the combine installs no cadence at all, so the
        sources run on every propose (Stage.setup only passes move_every when some
        entry differs from 1)."""
        st = _stage(_build(_FULL, GB_SEARCH_2_SOURCE_EVERY="1"), "gb_search_2")
        self.assertEqual({m.name: m.every for m in st.moves if m.name in SRC},
                         dict.fromkeys(SRC, 1))
        self.assertEqual({getattr(m, "every", 1) for m in st.moves}, {1})
        # and they stay the LAST GB-cycle entries (before any cycle-end noise)
        names = _names(st)
        self.assertEqual(names[-3:], list(SRC))

    def test_bad_values_are_refused(self):
        for bad in ("-1", "x", "1.5"):
            with self.assertRaises(ValueError, msg=bad):
                _build(_FULL, GB_SEARCH_2_SOURCE_EVERY=bad)

    def test_the_seed_never_carries_sources(self):
        seed = _stage(_build(_FULL, GB_SEARCH_1_SOURCE_EVERY="1"), "gb_search_seed")
        self.assertEqual([n for n in _names(seed) if n in SRC], [])


# ======================================================================
# 3. per-stage noise mode: the cycle-end convergence
# ======================================================================
class NoiseModeTest(unittest.TestCase):

    def tearDown(self):
        _clear_peak_floor()

    def test_cycle_end_is_ONE_psd_galfor_proposal_LAST_and_to_convergence(self):
        fit = _build(_FULL, GB_SEARCH_2_NOISE_MODE="cycle_end", GB_SEARCH_2_SOURCE_EVERY="1")
        st = _stage(fit, "gb_search_2")
        noise = _noise_moves(st)
        self.assertEqual(len(noise), 1, _names(st))
        slot = noise[0]
        self.assertIs(st.moves[-1], slot, "the convergence slot must END the cycle")
        self.assertEqual(slot.name, "noise_cycle_end_search")
        self.assertEqual(list(slot.inner_names), ["psd_pe", "galfor_pe"])  # no vgb_pe
        self.assertIsInstance(slot, _R().JointMaxLogLSearch)
        # the convergence rule: armed, afresh every cycle, the standalone
        # noise stage's plateau count, and a ceiling that is not the stop
        self.assertTrue(slot.restart_each_propose)
        self.assertEqual(slot.num_checks, 5)                       # NOISE_SEARCH_CHECKS default
        self.assertEqual(slot.iters_per_step, 5000)
        # no rider, no interleaved slots, no ratchet gate
        names = _names(st)
        for absent in ("noise_vgb_joint_search", "noise_ratchet_search",
                       "noise_joint_search_1", "noise_joint_search_2",
                       "noise_joint_search_3", "noise_joint_search_4"):
            self.assertNotIn(absent, names)
        # vgb_pe still leads the cycle (55 KNOWN sources, not noise)
        self.assertEqual(names[0], "vgb_pe")
        # the stage before keeps the noise FIXED, and so does the seed
        for n in ("gb_search_1", "gb_search_seed"):
            self.assertEqual(_noise_moves(_stage(fit, n)), [], n)

    def test_the_convergence_rule_knobs(self):
        fit = _build(_FULL, GB_SEARCH_2_NOISE_MODE="cycle_end", NOISE_SEARCH_CHECKS="7",
                     GB_SEARCH_2_NOISE_MAX_ROUNDS="300")
        slot = _noise_moves(_stage(fit, "gb_search_2"))[0]
        self.assertEqual((slot.num_checks, slot.iters_per_step), (7, 300))
        for bad in ("0", "-3", "x"):
            with self.assertRaises(ValueError, msg=bad):
                _build(_FULL, GB_SEARCH_2_NOISE_MODE="cycle_end", GB_SEARCH_2_NOISE_MAX_ROUNDS=bad)

    def test_the_slot_builds_a_restarting_max_logl_search(self):
        """setup(ctx) hands the flag to the runtime MaxLogLCombineMove, which is
        what makes every cycle converge afresh."""
        from lisatools.globalfit.moves.globalfitmove import MaxLogLCombineMove

        slot = _noise_moves(_stage(_build(_FULL, GB_SEARCH_2_NOISE_MODE="cycle_end"),
                                   "gb_search_2"))[0]
        built = []

        class _Fake(MaxLogLCombineMove):
            def __init__(self, moves, **kw):        # record, do not build eryn state
                built.append(kw)
                self.moves = moves
        with mock.patch("lisatools.globalfit.moves.globalfitmove.MaxLogLCombineMove", _Fake):
            mv = slot.setup(types.SimpleNamespace(stock_moves={"psd_pe": object(),
                                                               "galfor_pe": object()}))
        self.assertIsInstance(mv, _Fake)
        self.assertTrue(built[0]["restart_each_propose"])
        self.assertEqual((built[0]["num_checks"], built[0]["iters_per_step"]), (5, 5000))
        self.assertEqual(mv.gf_move_name, "noise_cycle_end_search")
        # the descriptor still pickles (pre-build fit rule)
        self.assertTrue(pickle.loads(pickle.dumps(slot)).restart_each_propose)

    def test_the_ratchet_and_the_cycle_end_are_refused_together(self):
        with self.assertRaises(ValueError):
            _build(_FULL, GB_SEARCH_2_NOISE_MODE="cycle_end", GALFOR_RATCHET="1")

    def test_interleaved_and_fixed_follow_the_table_composition(self):
        base = _build(_FULL)
        fit = _build(_FULL, GB_SEARCH_2_NOISE_MODE="interleaved", GB_SEARCH_3_NOISE_MODE="fixed")
        n2, n3 = _names(_stage(fit, "gb_search_2")), _names(_stage(fit, "gb_search_3"))
        self.assertIn("noise_vgb_joint_search", n2)
        self.assertEqual([n for n in n2 if n.startswith("noise_joint_search")],
                         [f"noise_joint_search_{k}" for k in (1, 2, 3, 4)])
        self.assertEqual(_noise_moves(_stage(fit, "gb_search_3")), [])
        # fixed stage 3 = stage 2's default shape
        self.assertEqual(n3, [n for n in _names(_stage(base, "gb_search_2"))])
        # the warm cadence stays a stage-3 property, whatever the noise does
        self.assertEqual(next(m.every for m in _stage(fit, "gb_search_2").moves
                              if m.name == "rj_warm_search"), 1)

    def test_unset_is_the_table(self):
        fit = _build(_FULL)
        self.assertEqual(_noise_moves(_stage(fit, "gb_search_1")), [])
        self.assertEqual(_noise_moves(_stage(fit, "gb_search_2")), [])
        self.assertEqual(len(_noise_moves(_stage(fit, "gb_search_3"))), 5)  # rider + 4 slots

    def test_bad_mode_is_refused(self):
        with self.assertRaises(ValueError):
            _build(_FULL, GB_SEARCH_2_NOISE_MODE="end")

    def test_gb_only_refuses_a_noise_mode_it_cannot_honour(self):
        """GB_ONLY has no psd/galfor branch: asking for noise there must fail loud."""
        with self.assertRaises(ValueError):
            _build(_GB_ONLY, GB_SEARCH_2_NOISE_MODE="cycle_end")
        _build(_GB_ONLY, GB_SEARCH_2_NOISE_MODE="fixed")          # harmless


class ConvergeEachProposeTest(unittest.TestCase):
    """MaxLogLCombineMove(restart_each_propose=True): every propose runs the plateau
    rule afresh. Without it a plateaued instance takes ONE round per call (the
    rider behaviour, 6mo job 675's 2-s second release)."""

    @staticmethod
    def _move(restart, num_checks=3):
        from lisatools.globalfit.moves.globalfitmove import MaxLogLCombineMove

        mv = MaxLogLCombineMove.__new__(MaxLogLCombineMove)
        mv.num_checks, mv.tol, mv.max_iter, mv.iters_per_step = num_checks, 1.0, 0, 1000
        mv.restart_each_propose = restart
        mv.gf_stage_name, mv.gf_stage_kind = "gb_search_2", "rj"
        mv.rounds = 0
        # a climb, then the sub-tol twitch of an ensemble at its mode (a real
        # stretch round always moves lnL a little; the plateau latch arms on it)
        ll = itertools.chain(np.arange(1, 6) * 10.0,
                             (50.0 + 0.01 * (k % 2) for k in itertools.count()))

        def once(model, state):
            mv.rounds += 1
            state.log_like = np.array([[next(ll), next(ll) - 1.0]])
            return state, np.ones((1, 2), dtype=bool)
        mv._propose_moves_once = once
        return mv

    def _rounds(self, restart):
        mv, state = self._move(restart), types.SimpleNamespace(log_like=np.zeros((1, 2)))
        out = []
        with env(MAXLOGL_PER_WALKER="1", MAXLOGL_LOG_EVERY="0"), \
                contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):
                before = mv.rounds
                mv._propose_moves(None, state)
                out.append(mv.rounds - before)
        return out

    def test_restart_converges_every_call(self):
        got = self._rounds(True)
        self.assertTrue(all(r >= 3 + 1 for r in got), got)   # >= num_checks flat rounds + 1

    def test_without_it_a_plateaued_search_takes_one_round(self):
        got = self._rounds(False)
        self.assertGreater(got[0], 1)
        self.assertEqual(got[1:], [1, 1])

    def test_the_default_is_off(self):
        import inspect
        from lisatools.globalfit.moves.globalfitmove import MaxLogLCombineMove

        p = inspect.signature(MaxLogLCombineMove.__init__).parameters["restart_each_propose"]
        self.assertIs(p.default, False)


# ======================================================================
# 4. per-stage valve reset
# ======================================================================
class _ValveMove:
    """A GB band move carrying the REAL barren-valve revive method."""

    branch_name = "gb"

    def __init__(self, name, shut=None):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

        self._revive = GBSpecialBase._band_shutoff_revive
        self.name = name
        self.opt_snr_rej_samp_limit = 5.0
        self.phase_maximize = False
        self._snr_lim_table = None
        self.is_rj_prop = True
        self.calls = []
        if shut is not None:
            self._rj_band_shutoff = np.asarray(shut, dtype=bool).copy()
            self._band_occ_streak = np.full(len(shut), 2, dtype=np.int64)
            self._band_occ_last = np.zeros(len(shut), dtype=np.int64)

    def _band_shutoff_revive(self, reason):
        self.calls.append(reason)
        return self._revive(self, reason)


class _Backend:
    def __init__(self, rows, start):
        self.iteration = rows
        self._start = start

    def stage_start_iteration(self, name):
        return self._start


def _state(nb=4, nw=2, stamp=0):
    bi = dict(
        band_rj_shutoff=np.array([True, False, True, False]),
        band_occ_streak=np.array([3, 1, 4, 0], dtype=np.int64),
        band_occ_last=np.zeros(nb, dtype=np.int64),
        band_shutoff_since_revive=np.array([7], dtype=np.int64),
        band_shutoff_epoch=np.array([12], dtype=np.int64),
        band_cold_logl_max_w=np.full((nw, nb), 123.0),
        band_rj_shutoff_w=np.ones((nw, nb), dtype=bool),
        band_shutoff_w_step=np.array([stamp], dtype=np.int64),
    )
    return types.SimpleNamespace(sub_states={"gb": types.SimpleNamespace(band_info=bi)})


class ResetValvesTest(unittest.TestCase):

    def tearDown(self):
        _clear_peak_floor()

    def test_the_knob_puts_the_key_on_that_stage_only(self):
        fit = _build(_FULL, GB_SEARCH_2_RESET_VALVES="1")
        self.assertTrue(_stage(fit, "gb_search_2").step_kwargs["profile"]["reset_valves"])
        for n in ("gb_search_seed", "gb_search_1", "gb_search_3"):
            self.assertNotIn("reset_valves", _stage(fit, n).step_kwargs["profile"], n)
        fit = _build(_FULL, GB_SEARCH_2_RESET_VALVES="0")
        self.assertNotIn("reset_valves", _stage(fit, "gb_search_2").step_kwargs["profile"])
        with self.assertRaises(ValueError):
            _build(_FULL, GB_SEARCH_2_RESET_VALVES="maybe")

    def test_both_assemblies_carry_it(self):
        fit = _build(_GB_ONLY, GB_SEARCH_2_RESET_VALVES="1")
        self.assertTrue(_stage(fit, "gb_search_2").step_kwargs["profile"]["reset_valves"])

    def _enter(self, profile, rows, start, stamp=0, serial=3):
        from lisatools.globalfit.recipe import SearchStageProfileStep

        designated = _ValveMove("rj_fstat_search", shut=[True, True, False, False])
        bare = _ValveMove("in_model")
        tree = [types.SimpleNamespace(moves=[designated, bare])]
        st = SearchStageProfileStep(moves=tree, convergence_iter=2, plateau_branch="gb",
                                    profile=profile, stage_name="gb_search_2")
        state = _state(stamp=stamp)
        st._ratchet_last_sample = state
        st._ratchet_backend = _Backend(rows, start)
        st._stage_start_iter = start
        with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
            st.note_recipe_step(serial)
        return state.sub_states["gb"].band_info, designated, bare, "\n".join(cm.output)

    def test_a_FRESH_entry_resets_every_valve(self):
        bi, designated, bare, log = self._enter(dict(reset_valves=True), rows=40, start=40)
        # the barren valve: the move's own revive ran, on the in-memory arrays ...
        self.assertEqual(len(designated.calls), 1)
        self.assertFalse(designated._rj_band_shutoff.any())
        self.assertFalse(designated._band_occ_streak.any())
        self.assertTrue((designated._band_occ_last == -1).all())
        self.assertEqual(len(bare.calls), 1)                  # no state there: a no-op
        # ... AND on the persisted record a fresh process restores from
        self.assertFalse(np.asarray(bi["band_rj_shutoff"]).any())
        self.assertFalse(np.asarray(bi["band_occ_streak"]).any())
        self.assertTrue((np.asarray(bi["band_occ_last"]) == -1).all())
        self.assertEqual(int(bi["band_shutoff_since_revive"][0]), 0)
        self.assertEqual(int(bi["band_shutoff_epoch"][0]), 12)     # the epoch is not evidence
        # the per-(walker, band) valve: max re-learned, window released
        self.assertTrue(np.isneginf(bi["band_cold_logl_max_w"]).all())
        self.assertFalse(np.asarray(bi["band_rj_shutoff_w"]).any())
        self.assertEqual(int(bi["band_shutoff_w_step"][0]), 3)
        self.assertIn("reset_valves", log)

    def test_a_RESUME_inside_the_stage_resets_nothing(self):
        """A relaunch inside gb_search_2 must not restart the valves' clocks: the
        stage ends on them, and a run bounced every few hours would never end."""
        bi, designated, bare, log = self._enter(dict(reset_valves=True), rows=47, start=40,
                                                stamp=3)
        self.assertEqual(designated.calls, [])
        self.assertTrue(designated._rj_band_shutoff[:2].all())
        self.assertTrue(np.asarray(bi["band_rj_shutoff"])[0])
        self.assertEqual(float(bi["band_cold_logl_max_w"][0, 0]), 123.0)
        self.assertTrue(np.asarray(bi["band_rj_shutoff_w"]).all())    # same step: honoured
        self.assertIn("RESUME", log)

    def test_without_the_key_the_barren_valve_is_untouched(self):
        bi, designated, _, _ = self._enter(dict(opt_snr=5.0), rows=40, start=40)
        self.assertEqual(designated.calls, [])
        self.assertTrue(np.asarray(bi["band_rj_shutoff"])[0])
        self.assertEqual(float(bi["band_cold_logl_max_w"][0, 0]), 123.0)
        # the per-walker release is the step-serial rule, independent of the key
        self.assertFalse(np.asarray(bi["band_rj_shutoff_w"]).any())

    def test_a_misspelled_key_is_still_refused(self):
        from lisatools.globalfit.recipe import SearchStageProfileStep

        with self.assertRaises(ValueError):
            SearchStageProfileStep(moves=[], profile=dict(reset_valve=True), stage_name="x")


# ======================================================================
# 5. the 1yr / 9mo launchers compose the ruled recipe; the 6mo is unchanged
# ======================================================================
class LauncherRecipeTest(unittest.TestCase):
    """Build the fit under each launcher's RESOLVED exports (+ the seed-store noise
    pin the launcher fills at launch time; a placeholder -- only its presence
    changes the composition)."""

    @classmethod
    def setUpClass(cls):
        from tests.test_submit_scripts_layout import NINE_MO_V9, ONE_YR_V9, SIX_MO_V9, _exports

        # Import (and so initialize MPI) BEFORE the environment is replaced:
        # MPI_Init inside a cleared environment ABORTS the whole process (no
        # test output at all), and erebor snapshots env-backed defaults at
        # import time. Order-independent this way (the class run alone, or
        # first, behaves as inside the module run).
        _R()
        from lisatools.globalfit.stock import erebor  # noqa: F401
        try:
            from mpi4py import MPI  # noqa: F401
        except ImportError:
            pass
        system = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "USER")
                  if k in os.environ}
        cls.fits = {}
        for tag, path in (("6mo", SIX_MO_V9), ("1yr", ONE_YR_V9), ("9mo", NINE_MO_V9)):
            ex = {k: v for k, v in _exports(path).items()
                  if k not in ("_", "SHLVL", "PWD", "OLDPWD", "PATH", "HOME")}
            ex["PSD_START_PARAMS"] = "1.5e-11,3e-15"
            ex.update(system)
            with mock.patch.dict(os.environ, ex, clear=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                cls.fits[tag] = _R().build_fit()
            _clear_peak_floor()

    def test_the_9mo_recipe(self):
        fit = self.fits["9mo"]
        self.assertEqual([s.name for s in fit.recipe.stages],
                         ["gb_search_seed", "gb_search_1", "gb_search_2", "full_pe"])
        seed, s1, s2 = (_stage(fit, n) for n in ("gb_search_seed", "gb_search_1", "gb_search_2"))
        self.assertEqual(seed.step_kwargs["convergence_fn"].n, 3)
        self.assertEqual([n for n in _names(seed) if n in SRC], [])
        self.assertEqual(s1.step_kwargs["profile"], dict(
            phase_maximize=True, opt_snr=8.0, peak_min_snr=8.0, prior_births=False))
        self.assertEqual([n for n in _names(s1) if n in SRC], [])
        self.assertEqual(_noise_moves(s1), [])
        self.assertEqual(s2.step_kwargs["profile"], dict(
            phase_maximize=False, opt_snr=5.0, peak_min_snr=6.25, prior_births=True,
            reset_valves=True))
        self.assertEqual({m.name: m.every for m in s2.moves if m.name in SRC},
                         dict.fromkeys(SRC, 1))
        noise = _noise_moves(s2)
        self.assertEqual(len(noise), 1)
        self.assertIs(s2.moves[-1], noise[0])
        self.assertTrue(noise[0].restart_each_propose)
        # each RJ move with its in-model associate, as in every search stage
        for a, b in (("rj_warm_search", "in_model"), ("rj_fstat_search", "in_model_fstat"),
                     ("rj_prior_removal", "in_model_removal")):
            self.assertEqual(_names(s2).index(b), _names(s2).index(a) + 1, (a, b))
        self.assertIsNone(s2.step_kwargs.get("ratchet"))

    def test_the_1yr_composes_the_SAME_recipe(self):
        def sig(fit):
            return [(s.name, s.kind, [(m.name, getattr(m, "every", 1)) for m in s.moves],
                     repr(s.step_kwargs.get("profile")),
                     getattr(s.step_kwargs.get("convergence_fn"), "n", None))
                    for s in fit.recipe.stages]
        self.assertEqual(sig(self.fits["1yr"]), sig(self.fits["9mo"]))

    def test_full_pe_declarations_are_todays(self):
        new, old = _stage(self.fits["9mo"], "full_pe"), _stage(self.fits["6mo"], "full_pe")
        self.assertEqual(_names(new), _names(old))
        self.assertEqual([m.every for m in new.moves], [m.every for m in old.moves])
        for k in ("peak_min_snr", "pe_repeats", "pe_rj_flip_fraction", "stage_name"):
            self.assertEqual(new.step_kwargs[k], old.step_kwargs[k], k)
        self.assertEqual(new.combine_kwargs, old.combine_kwargs)

    def test_the_6mo_recipe_is_untouched(self):
        fit = self.fits["6mo"]
        self.assertEqual([s.name for s in fit.recipe.stages],
                         ["gb_search_seed", "gb_search_1", "gb_search_2", "gb_search_3",
                          "replica_pe", "full_pe"])
        self.assertEqual(_stage(fit, "gb_search_seed").step_kwargs["convergence_fn"].n, 5)
        for s in fit.recipe.stages:
            self.assertNotIn("reset_valves", (s.step_kwargs.get("profile") or {}), s.name)
            self.assertNotIn("noise_cycle_end_search", _names(s), s.name)
        for n in ("gb_search_1", "gb_search_2", "gb_search_3"):
            self.assertEqual({m.name: m.every for m in _stage(fit, n).moves if m.name in SRC},
                             dict.fromkeys(SRC, 5), n)


class DefaultsAreTodayTest(unittest.TestCase):
    """Unset == the knobs at their documented defaults (the 6mo composition)."""

    def tearDown(self):
        _clear_peak_floor()

    @staticmethod
    def _sig(fit):
        return [(s.name, s.kind, [(m.name, getattr(m, "every", 1), type(m).__name__,
                                   getattr(m, "num_checks", None),
                                   getattr(m, "iters_per_step", None))
                                  for m in s.moves],
                 repr(sorted((s.step_kwargs.get("profile") or {}).items())),
                 repr(sorted(s.combine_kwargs.items())))
                for s in fit.recipe.stages]

    def test_explicit_defaults_compose_the_same_recipe(self):
        explicit = dict(GB_SEARCH_STAGES="1,2,3", GB_SEARCH_1_NOISE_MODE="fixed",
                        GB_SEARCH_2_NOISE_MODE="fixed", GB_SEARCH_3_NOISE_MODE="interleaved",
                        GB_SEARCH_1_SOURCE_EVERY="5", GB_SEARCH_2_SOURCE_EVERY="5",
                        GB_SEARCH_3_SOURCE_EVERY="5", GB_SEARCH_1_RESET_VALVES="0",
                        GB_SEARCH_2_RESET_VALVES="0", GB_SEARCH_3_RESET_VALVES="0",
                        GB_SEARCH_2_NOISE_MAX_ROUNDS="77")         # inert outside cycle_end
        self.assertEqual(self._sig(_build(_FULL)), self._sig(_build(_FULL, **explicit)))


# ======================================================================
# 6. the recipe printout
# ======================================================================
class DescribeRecipeTest(unittest.TestCase):

    def tearDown(self):
        _clear_peak_floor()

    def test_the_description_and_the_builder_agree_on_names_and_order(self):
        for over in ({}, _RECIPE_9MO):
            fit = _build(_FULL, **over)
            text = _R().describe_recipe(fit)
            got = [ln.split()[2] for ln in text.splitlines() if ln.startswith("stage ")]
            self.assertEqual(got, [s.name for s in fit.recipe.stages], text)

    def test_the_cycle_end_slot_is_shown_last_and_to_convergence(self):
        text = _R().describe_recipe(_build(_FULL, **_RECIPE_9MO))
        block = text.split("stage 3/4 ")[1].split("stage 4/4 ")[0]
        moves = [ln for ln in block.splitlines() if ln.startswith("      ")]
        self.assertIn("noise_cycle_end_search", moves[-1])
        self.assertIn("TO CONVERGENCE", moves[-1])
        self.assertIn("reset_valves=True", block)
        self.assertIn("every iteration", block.lower())

    def test_print_recipe_needs_no_mpi(self):
        out = io.StringIO()
        with env(**_GB_ONLY), contextlib.redirect_stdout(out):
            rc = _R().print_recipe()
        self.assertEqual(rc, 0)
        self.assertIn("RESOLVED RECIPE", out.getvalue())
        import inspect
        src = inspect.getsource(sys.modules[_R().__name__])
        main = src[src.index('if __name__ == "__main__":'):]
        self.assertLess(main.index("--print-recipe"), main.index("from mpi4py import MPI"))


if __name__ == "__main__":
    unittest.main()
