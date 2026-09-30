"""The galfor RATCHET (user design 2026-09-30).

gb_search_3 carries ONE gated noise proposal at the head of the iteration
instead of the leading rider plus four interleaved slots. A stage-local
schedule drives the gate: iteration 0 of a cycle FORCES the foreground down
(a deterministic shift of the galfor coordinates on every rung and walker,
published through the noise move's own accept path, followed by one GB
in-model pass so the sources settle to the new noise before any RJ move sees
the residual), the next ``hold - 1`` iterations HOLD (the noise moves do not
run), then ``release`` iterations RELEASE (the ordinary joint max-logL noise
search runs). ``cycles`` nudges, then permanent release.

Every nudge arms a HARD forced F-stat refit -- one that ignores the epoch's
tick age -- because a grid fitted under the old noise curve selects the wrong
peaks for the lowered one.

Construction-level tests: no data, no GPU.
"""

from __future__ import annotations

import contextlib
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scripts", "fstat_proposal"))

from lisatools.globalfit.noise_ratchet import (  # noqa: E402
    GALFOR_RATCHET_DEFAULT_DELTA, NoiseRatchetGate, RatchetSchedule,
    galfor_curve_ratio, is_noise_ratchet_gate, nudge_delta_from_env,
    ratchet_from_env,
)


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


_RATCHET_KNOBS = dict(
    GALFOR_RATCHET=None, GALFOR_RATCHET_HOLD=None, GALFOR_RATCHET_RELEASE=None,
    GALFOR_RATCHET_CYCLES=None, GALFOR_RATCHET_DLOG10_AMP=None,
    GALFOR_RATCHET_DLOG10_FK=None, GALFOR_RATCHET_DLOG10_F2=None,
)


# ======================================================================
# 1. the schedule
# ======================================================================

class ScheduleTest(unittest.TestCase):

    def test_sequence_hold3_release2_cycles2(self):
        s = RatchetSchedule(hold=3, release=2, cycles=2)
        got = [s.action(k) for k in range(12)]
        self.assertEqual(got, [
            "nudge", "hold", "hold", "release", "release",
            "nudge", "hold", "hold", "release", "release",
            "release", "release"])

    def test_hold_of_one_is_nudge_then_release(self):
        s = RatchetSchedule(hold=1, release=2, cycles=1)
        self.assertEqual([s.action(k) for k in range(4)],
                         ["nudge", "release", "release", "release"])

    def test_validation(self):
        for bad in (dict(hold=0, release=2, cycles=1),
                    dict(hold=3, release=0, cycles=1),
                    dict(hold=3, release=2, cycles=0)):
            with self.assertRaises(ValueError, msg=str(bad)):
                RatchetSchedule(**bad)

    def test_env_unset_is_off(self):
        with env(**_RATCHET_KNOBS):
            self.assertIsNone(ratchet_from_env())

    def test_env_armed_defaults(self):
        """hold 3 (nudge + 2 held), release 2, TWO cycles (user ruling
        2026-09-30: "do at least 2 cycles for now")."""
        with env(**{**_RATCHET_KNOBS, "GALFOR_RATCHET": "1"}):
            self.assertEqual(ratchet_from_env(),
                             RatchetSchedule(hold=3, release=2, cycles=2))

    def test_env_overrides(self):
        with env(**{**_RATCHET_KNOBS, "GALFOR_RATCHET": "1",
                    "GALFOR_RATCHET_HOLD": "4", "GALFOR_RATCHET_RELEASE": "1",
                    "GALFOR_RATCHET_CYCLES": "3"}):
            self.assertEqual(ratchet_from_env(),
                             RatchetSchedule(hold=4, release=1, cycles=3))

    def test_env_bad_value_raises_not_silently_disables(self):
        with env(**{**_RATCHET_KNOBS, "GALFOR_RATCHET": "1",
                    "GALFOR_RATCHET_HOLD": "0"}):
            with self.assertRaises(ValueError):
                ratchet_from_env()


# ======================================================================
# 2. the nudge vector and its readout
# ======================================================================

class DeltaTest(unittest.TestCase):

    def test_default_delta_is_the_broad_and_SUFFICIENT_nudge(self):
        """amp -0.05, fk -0.10, f_2 -0.15 dex; alpha and f_1 untouched.
        Broad (user ruling 2026-09-30: nudge the WHOLE curve and let the
        release show which side comes back) and sized so ONE nudge brings the
        3-5 mHz total noise to ~0.75-1.5x the add-back estimate, where the
        F-stat peak floor admits true SNR-8 sources across the band (user
        instruction, same day: "make sure your step size in the forcing
        downward is sufficient"). The smaller fk -0.05 / f_2 -0.10 step left
        SNR 8 at 5.3 at 4 mHz, under the 6.25 peak floor."""
        with env(**_RATCHET_KNOBS):
            d = nudge_delta_from_env()
        np.testing.assert_allclose(d, [-0.05, -0.10, 0.0, 0.0, -0.15])
        np.testing.assert_allclose(d, GALFOR_RATCHET_DEFAULT_DELTA)

    def test_delta_overrides(self):
        with env(**{**_RATCHET_KNOBS, "GALFOR_RATCHET_DLOG10_AMP": "0",
                    "GALFOR_RATCHET_DLOG10_FK": "-0.1",
                    "GALFOR_RATCHET_DLOG10_F2": "-0.15"}):
            d = nudge_delta_from_env()
        np.testing.assert_allclose(d, [0.0, -0.1, 0.0, 0.0, -0.15])

    def test_curve_ratio_invariants(self):
        v = np.array([-43.84, -2.60, 5.0, -2.0, -2.85])   # log10 basis
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-3
        np.testing.assert_allclose(galfor_curve_ratio(v, v, f), 1.0)
        amp = v + np.array([-0.05, 0, 0, 0, 0])
        np.testing.assert_allclose(galfor_curve_ratio(amp, v, f), 10 ** -0.05)
        fk = v + np.array([0, -0.05, 0, 0, 0])
        r = galfor_curve_ratio(fk, v, f)
        self.assertTrue(np.all(r < 1.0), r)
        # a lower knee bites hardest above it
        self.assertLess(r[3], r[0])


# ======================================================================
# 3. the gate
# ======================================================================

class _Log:
    def __init__(self):
        self.calls = []


class _FakeInner:
    """Stands in for the built joint max-logL noise search."""

    def __init__(self, log):
        self.log = log
        self.moves = []
        self.accepted = None

    def propose(self, model, state):
        self.log.calls.append("inner")
        return state, np.ones(np.shape(state.log_like), dtype=bool)


class _FakeGalfor:
    def __init__(self, log):
        self.log = log

    def forced_noise_step(self, model, state, deltas):
        self.log.calls.append(("forced", {k: tuple(v) for k, v in deltas.items()}))
        return state, np.ones(np.shape(state.log_like), dtype=bool)


class _FakeInModel:
    def __init__(self, log):
        self.log = log

    def propose(self, model, state):
        self.log.calls.append("in_model")
        return state, np.zeros(np.shape(state.log_like), dtype=bool)


def _state(nt=2, nw=4):
    return SimpleNamespace(log_like=np.zeros((nt, nw)), sub_states=None)


class GateTest(unittest.TestCase):

    def _gate(self, with_in_model=True):
        log = _Log()
        inner, gal = _FakeInner(log), _FakeGalfor(log)
        im = _FakeInModel(log) if with_in_model else None
        gate = NoiseRatchetGate(inner, gal, np.array([-0.05, -0.05, 0, 0, -0.1]),
                                in_model_move=im)
        return gate, log

    def test_is_a_walkable_combine_holding_the_inner(self):
        gate, _ = self._gate()
        self.assertTrue(is_noise_ratchet_gate(gate))
        self.assertEqual(gate.moves, [gate.moves[0]])
        self.assertIsInstance(gate.moves[0], _FakeInner)
        self.assertEqual(gate.mode, "release")   # inert until a step drives it

    def test_hold_runs_nothing_and_returns_the_state(self):
        gate, log = self._gate()
        gate.set_mode("hold")
        st = _state()
        out, acc = gate.propose(None, st)
        self.assertIs(out, st)
        self.assertEqual(log.calls, [])
        self.assertEqual(np.shape(acc), (2, 4))
        self.assertFalse(np.any(acc))

    def test_release_runs_the_inner_search_then_in_model(self):
        """User ruling 2026-09-30: the in-model follow-up runs whenever the
        noise CHANGED this iteration, forced or free -- never on a hold."""
        gate, log = self._gate()
        gate.set_mode("release")
        out, acc = gate.propose(None, _state())
        self.assertEqual(log.calls, ["inner", "in_model"])
        self.assertTrue(np.all(acc))
        self.assertEqual(gate.mode, "release")   # a release does not latch

    def test_nudge_forces_then_in_model_then_holds(self):
        gate, log = self._gate()
        gate.set_mode("nudge")
        out, acc = gate.propose(None, _state())
        self.assertEqual(log.calls, [
            ("forced", {"galfor": (-0.05, -0.05, 0.0, 0.0, -0.1)}),
            "in_model"])
        self.assertEqual(gate.mode, "hold")
        self.assertTrue(np.all(acc))

    def test_nudge_without_an_in_model_move_still_forces(self):
        gate, log = self._gate(with_in_model=False)
        gate.set_mode("nudge")
        gate.propose(None, _state())
        self.assertEqual(log.calls, [
            ("forced", {"galfor": (-0.05, -0.05, 0.0, 0.0, -0.1)})])

    def test_unknown_mode_refused(self):
        gate, _ = self._gate()
        with self.assertRaises(ValueError):
            gate.set_mode("wobble")


# ======================================================================
# 4. the HARD forced refit (ignores the epoch's tick age)
# ======================================================================

class HardRefitTest(unittest.TestCase):

    def _fake(self, root):
        from lisatools.sampling.fstat_proposal import fstat_peak_min_F

        return SimpleNamespace(
            name="rj_fstat_search", fstat_refit_every=2, _fstat_root=root,
            _force_refit_serial=None, _force_refit_done=None,
            _force_refit_reason="", _force_refit_hard=False,
            _latest_epoch=lambda: 28, _epoch_dir=lambda k: f"{root}/epoch_{k:04d}",
            _epoch_peak_min_F=lambda d: float(fstat_peak_min_F()),
            _fstat_clock=lambda: 5, _epoch_fit_clock=lambda k: 4,   # 1 tick old
        )

    def test_soft_arm_still_declines_a_fresh_epoch(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove as M)

        fake = self._fake(f"/nonexistent/ratchet-soft-{id(self)}")
        M.arm_fstat_refit(fake, ("stage", 3), "profile")
        self.assertIsNone(M._consume_forced_refit(fake))

    def test_hard_arm_opens_a_new_epoch_regardless_of_age(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove as M)

        fake = self._fake(f"/nonexistent/ratchet-hard-{id(self)}")
        M.arm_fstat_refit(fake, ("galfor_ratchet", 0), "nudge", ignore_age=True)
        self.assertEqual(M._consume_forced_refit(fake), 29)
        # consumed: the second setup() of the same serial takes the cadence path
        self.assertIsNone(M._consume_forced_refit(fake))
        # and the hard flag does not leak into the NEXT soft arm
        M.arm_fstat_refit(fake, ("stage", 4), "profile")
        self.assertIsNone(M._consume_forced_refit(fake))

    def test_force_helper_passes_hard_through(self):
        from lisatools.globalfit.recipe import force_fstat_refit

        armed = []
        grid = SimpleNamespace(
            arm_fstat_refit=lambda s, r="", ignore_age=False: armed.append((s, r, ignore_age)))
        n = force_fstat_refit([SimpleNamespace(moves=[grid])], ("x", 1), "why", hard=True)
        self.assertEqual(n, 1)
        self.assertEqual(armed, [(("x", 1), "why", True)])


# ======================================================================
# 5. the recipe step drives the gate from the stage-local iteration
# ======================================================================

class _FakeGate:
    is_noise_ratchet_gate = True

    def __init__(self):
        self.moves = []
        self.modes = []
        self.mode = "release"

    def set_mode(self, mode):
        self.mode = mode
        self.modes.append(mode)


class _FakeGrid:
    branch_name = "gb"

    def __init__(self):
        self.name = "rj_fstat_search"
        self.is_rj_prop = True
        self.opt_snr_rej_samp_limit = 5.0
        self.phase_maximize = False
        self._snr_lim_table = None
        self._shutoff_w_pending = None
        self.armed = []

    def arm_fstat_refit(self, serial, reason="", ignore_age=False):
        self.armed.append((serial, reason, ignore_age))


class _FakeBackend:
    def __init__(self, iteration, stage_start=None):
        self.iteration = iteration
        self._stage_start = stage_start

    def get_nleaves(self, branch_names=None, temp_index=0):
        # strictly growing: the nleaves plateau never trips in these tests
        return {branch_names[0]: np.arange(self.iteration)[:, None]}

    def stage_start_iteration(self, name):
        return self._stage_start


class _FakeSampler:
    def __init__(self, iteration, moves, stage_start=None):
        self.backend = _FakeBackend(iteration, stage_start)
        self.moves = moves
        self.periodic = None
        self.temperature_control = None
        self.weights = None


class StepDrivesRatchetTest(unittest.TestCase):
    """The step drives the gate from the STAGE-LOCAL iteration.

    Entry goes through the head-only announce path (``setup_run`` then
    ``note_recipe_step``), never through ``setup_run`` alone -- compute
    ranks call ``setup_run(0, ...)`` at build and must not touch the gate.
    """

    def _step(self):
        from lisatools.globalfit.recipe import SearchStageProfileStep

        gate, grid = _FakeGate(), _FakeGrid()
        tree = [SimpleNamespace(moves=[gate, grid])]
        st = SearchStageProfileStep(
            moves=tree, convergence_iter=2, plateau_branch="gb", profile={},
            stage_name="gb_search_3",
            ratchet=RatchetSchedule(hold=3, release=2, cycles=2),
            ratchet_delta=np.array([-0.05, -0.05, 0, 0, -0.1]))
        return st, gate, grid, tree

    def _enter(self, st, tree, iteration, stage_start=None, serial=3):
        st.setup_run(iteration, None, _FakeSampler(iteration, tree, stage_start))
        st.note_recipe_step(serial)

    def test_setup_run_alone_touches_nothing(self):
        """The compute-rank path: setup_run without an announce."""
        st, gate, grid, tree = self._step()
        st.setup_run(0, None, _FakeSampler(0, tree))
        self.assertEqual(gate.modes, [])
        self.assertEqual(grid.armed, [])

    def test_stage_entry_is_a_nudge_with_a_hard_refit(self):
        st, gate, grid, tree = self._step()
        self._enter(st, tree, 47)
        self.assertEqual(gate.modes, ["nudge"])
        self.assertEqual(len(grid.armed), 1)
        serial, reason, hard = grid.armed[0]
        self.assertTrue(hard)
        self.assertIn(0, serial)          # stage-local iteration 0 in the serial
        self.assertIn("ratchet", reason)

    def test_schedule_follows_the_stored_iteration(self):
        st, gate, grid, tree = self._step()
        self._enter(st, tree, 47)
        # end of iteration 0 -> the backend now points at row 48 -> k = 1
        for it in (48, 49, 50, 51, 52):
            st.stopping_function(it, None, _FakeSampler(it, tree))
        self.assertEqual(gate.modes, ["nudge", "hold", "hold", "release",
                                      "release", "nudge"])
        # two nudges, two DISTINCT hard serials
        self.assertEqual(len(grid.armed), 2)
        self.assertNotEqual(grid.armed[0][0], grid.armed[1][0])
        self.assertTrue(all(h for _, _, h in grid.armed))

    def test_resume_mid_hold_does_not_re_nudge(self):
        """A relaunch at stored iteration 49 whose recipe group says the
        stage started at 47 is stage-local k = 2: a HOLD, not a fresh nudge."""
        st, gate, grid, tree = self._step()
        self._enter(st, tree, 49, stage_start=47)
        self.assertEqual(gate.modes, ["hold"])
        self.assertEqual(grid.armed, [])
        # and the clock keeps the stored origin afterwards: 50 -> k = 3
        st.stopping_function(50, None, _FakeSampler(50, tree, stage_start=47))
        self.assertEqual(gate.modes[-1], "release")

    def test_stale_stamp_ahead_of_the_live_iteration_is_ignored(self):
        st, gate, grid, tree = self._step()
        self._enter(st, tree, 47, stage_start=60)
        self.assertEqual(gate.modes, ["nudge"])

    def test_no_ratchet_means_no_gate_traffic(self):
        from lisatools.globalfit.recipe import SearchStageProfileStep

        gate, grid = _FakeGate(), _FakeGrid()
        tree = [SimpleNamespace(moves=[gate, grid])]
        st = SearchStageProfileStep(
            moves=tree, convergence_iter=2, plateau_branch="gb", profile={},
            stage_name="gb_search_3")
        st.setup_run(47, None, _FakeSampler(47, tree))
        st.note_recipe_step(3)
        st.stopping_function(48, None, _FakeSampler(48, tree))
        self.assertEqual(gate.modes, [])
        self.assertEqual(grid.armed, [])


class NextStepStartStampTest(unittest.TestCase):
    """The transition stamps the NEXT step's start_iteration in the same
    write as this step's completion, so the ratchet clock survives a resume."""

    def _store(self, tmp):
        import h5py

        path = os.path.join(tmp, "s.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("global_fit")
            g.attrs["iteration"] = 47
            r = g.create_group("recipe")
            for i, n in enumerate(("gb_search_2", "gb_search_3", "full_pe")):
                s = r.create_group(n)
                s.attrs["status"] = False
                s.attrs["order num"] = i
        return path

    def test_backend_stamps_next_start_once(self):
        import tempfile

        import h5py

        from lisatools.globalfit.hdfbackend import GFHDFBackend

        with tempfile.TemporaryDirectory() as tmp:
            path = self._store(tmp)
            be = GFHDFBackend(path)
            be.completed_recipe_step("gb_search_2", next_step_name="gb_search_3")
            with h5py.File(path, "r") as f:
                r = f["global_fit/recipe"]
                self.assertTrue(bool(r["gb_search_2"].attrs["status"]))
                self.assertEqual(int(r["gb_search_2"].attrs["completed_iteration"]), 47)
                self.assertEqual(int(r["gb_search_3"].attrs["start_iteration"]), 47)
                self.assertIsNone(r["full_pe"].attrs.get("start_iteration"))
            self.assertEqual(be.stage_start_iteration("gb_search_3"), 47)
            # an existing stamp is KEPT (a resume never rewrites the origin)
            with h5py.File(path, "a") as f:
                f["global_fit"].attrs["iteration"] = 60
            be.completed_recipe_step("gb_search_2", next_step_name="gb_search_3")
            self.assertEqual(be.stage_start_iteration("gb_search_3"), 47)

    def test_recipe_passes_the_next_incomplete_step(self):
        from lisatools.globalfit.recipe import Recipe

        calls = []

        class _Step:
            def __init__(self):
                self.moves = []
                self.weights = None

            def stopping_function(self, i, s, sampler):
                return True

            def setup_run(self, *a, **k):
                pass

        class _Backend:
            def completed_recipe_step(self, name, next_step_name=None):
                calls.append((name, next_step_name))

        recipe = Recipe()
        recipe._init_runtime()
        recipe.recipe = [
            {"name": "a", "adjust": _Step(), "status": False},
            {"name": "b", "adjust": _Step(), "status": True},   # already done
            {"name": "c", "adjust": _Step(), "status": False},
        ]
        recipe.backend = _Backend()
        recipe._current_iter = 0
        recipe._current_recipe_step = recipe.recipe[0]
        sampler = SimpleNamespace(moves=None, weights=None, periodic=None,
                                  temperature_control=None,
                                  backend=SimpleNamespace(iteration=5))
        recipe(5, None, sampler)
        self.assertEqual(calls, [("a", "c")])


# ======================================================================
# 6. the forced step on a REAL PSDMove (like/prior/sensitivity mocked)
# ======================================================================

class _FakeACS:
    def __init__(self, nw):
        self._acs = [SimpleNamespace(sens_mat="original") for _ in range(nw)]
        self.gpus = None
        self.xp = np
        self.reset_calls = 0

    def __len__(self):
        return len(self._acs)

    def __getitem__(self, i):
        return self._acs[i]

    def reset_linear_psd_arr(self):
        self.reset_calls += 1

    def likelihood(self):
        return 10.0 * np.arange(len(self._acs), dtype=float)

    def flatten(self):
        return list(self._acs)


class ForcedStepTest(unittest.TestCase):
    NT, NW = 2, 4

    def setUp(self):
        try:
            from eryn.moves.tempering import TemperatureControl  # noqa: F401
            from lisatools.globalfit.moves.psdmove import PSDMove  # noqa: F401
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"psdmove not available: {exc}")

    def _like(self, coords, inds=None, logp=None, supps=None, branch_supps=None):
        g = np.asarray(coords["galfor"])[:, :, 0, :]
        return -0.5 * np.sum(g ** 2, axis=-1), None

    @staticmethod
    def _prior(coords, inds=None, supps=None, branch_supps=None):
        g = np.asarray(coords["galfor"])[:, :, 0, :]
        inside = np.all(np.abs(g) < 50.0, axis=-1)
        return np.where(inside, 0.0, -np.inf)

    def _fixture(self):
        from eryn.model import Model
        from eryn.moves.tempering import TemperatureControl
        from eryn.state import BranchSupplemental

        from lisatools.globalfit.moves.psdmove import PSDMove
        from lisatools.globalfit.state import GFState

        acs = _FakeACS(self.NW)
        tc = TemperatureControl(5, self.NW, ntemps=self.NT, permute=False)

        class _FlatPrior:
            """logpdf(x: (n, ndim)) -> zeros(n): the cold-row prior sum the
            finish half writes into the engine log_prior."""

            def logpdf(self, x):
                return np.zeros(np.shape(x)[0])

        move = PSDMove(acs, {"psd": _FlatPrior(), "galfor": _FlatPrior()},
                       sampled_branches=["galfor"],
                       temperature_control=tc, live_dangerously=True,
                       name="ratchet forced step")
        move.compute_log_like = self._like
        move.compute_log_prior = self._prior
        move.periodic = None
        move.accepted = np.zeros((self.NT, self.NW))
        move._replay_noise_begin = lambda: None
        built = []
        move._build_sensitivity_for_walker = (
            lambda w, p, g, s: built.append((int(w), np.array(g, copy=True))) or f"sens{w}")
        rng = np.random.default_rng(3)
        coords = {"psd": rng.uniform(1, 2, size=(self.NT, self.NW, 1, 2)),
                  "galfor": rng.uniform(-3, -1, size=(self.NT, self.NW, 1, 5))}
        supps = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(self.NW), (self.NT, 1))},
            base_shape=(self.NT, self.NW), copy=True)
        # sub_state_bases as the engine builds it (None per branch = no
        # tempered sub-state); GFState(state, copy=True) reads it back
        state = GFState(coords, copy=True, supplemental=supps,
                        sub_state_bases={"psd": None, "galfor": None})
        state.log_prior = self._prior(coords)
        state.log_like = self._like(coords)[0]
        # the global-fit model object as the moves see it: the engine's
        # container array rides on it (eryn's bare Model has no such field)
        model = SimpleNamespace(analysis_container_arr=acs, map_fn=map,
                                random=np.random.RandomState(1),
                                temperature_control=tc)
        del Model  # noqa: F821 -- imported for parity with the DA harness only
        return move, model, state, acs, built

    def test_shift_lands_on_every_rung_and_walker_and_is_published(self):
        move, model, state, acs, built = self._fixture()
        before_g = np.array(state.branches["galfor"].coords, copy=True)
        before_p = np.array(state.branches["psd"].coords, copy=True)
        delta = np.array([-0.05, -0.05, 0.0, 0.0, -0.10])

        new_state, accepted = move.forced_noise_step(model, state, {"galfor": delta})

        np.testing.assert_allclose(new_state.branches["galfor"].coords, before_g + delta)
        np.testing.assert_array_equal(new_state.branches["psd"].coords, before_p)
        # engine cold row: the published containers' likelihood
        np.testing.assert_allclose(new_state.log_like[0], acs.likelihood())
        # every walker's sensitivity rebuilt with ITS nudged cold-row galfor
        self.assertEqual(sorted(w for w, _ in built), list(range(self.NW)))
        for w, g in built:
            np.testing.assert_allclose(g, before_g[0, w, 0] + delta)
        for w in range(self.NW):
            self.assertEqual(acs[w].sens_mat, f"sens{w}")
        self.assertEqual(np.shape(accepted), (self.NT, self.NW))
        self.assertTrue(np.all(accepted))
        # the input state is left alone (eryn contract: a NEW state)
        np.testing.assert_array_equal(state.branches["galfor"].coords, before_g)

    def test_shift_outside_the_prior_is_refused_and_nothing_moves(self):
        move, model, state, acs, built = self._fixture()
        before_g = np.array(state.branches["galfor"].coords, copy=True)
        with self.assertRaises(ValueError):
            move.forced_noise_step(model, state, {"galfor": np.full(5, 100.0)})
        np.testing.assert_array_equal(state.branches["galfor"].coords, before_g)
        self.assertEqual(built, [])

    def test_unsampled_branch_is_refused(self):
        move, model, state, acs, built = self._fixture()
        with self.assertRaises(ValueError):
            move.forced_noise_step(model, state, {"psd": np.zeros(2)})


# ======================================================================
# 7. composition: gb_search_3 under the ratchet
# ======================================================================

class CompositionTest(unittest.TestCase):

    _BASE = dict(
        GB_SEARCH_IN_MODEL="1", GB_SEARCH_RJ_REPLACE="1",
        GB_SEARCH_IN_MODEL_REPLACE="1",
        GB_WARM_START_COMPONENTS="/nonexistent/warm.npz",
        STAGE_V9_SEARCH="1", GB_SEARCH_3_WARM_EVERY="5",
        MBHB_IDS="2,5", STAGE_SKIP_SOURCE_SEARCH="1", VGB_CHIRP_MASS_BASIS="1",
        PSD_START_PARAMS=None, GALFOR_START_PARAMS=None,
    )

    def _full(self, **ov):
        import run_combined_staged as R

        with env(**{**_RATCHET_KNOBS, **self._BASE, **ov}):
            return R.build_fit()

    @staticmethod
    def _names(stage):
        return [m.name for m in stage.moves]

    def test_armed_stage_3_has_one_gated_noise_head_and_no_slots(self):
        fit = self._full(GALFOR_RATCHET="1")
        by = {s.name: s for s in fit.recipe.stages}
        names = self._names(by["gb_search_3"])
        self.assertEqual(names[:2], ["vgb_pe", "noise_ratchet_search"])
        self.assertNotIn("noise_vgb_joint_search", names)
        self.assertEqual([n for n in names if n.startswith("noise_joint_search")], [])
        # the GB cycle itself is untouched
        for n in ("in_model", "rj_fstat_search", "in_model_fstat",
                  "rj_prior_removal", "in_model_removal"):
            self.assertIn(n, names)
        gate = by["gb_search_3"].moves[1]
        self.assertEqual(sorted(gate.inner_names), ["galfor_pe", "psd_pe"])
        # the in-model follow-up is a declared dependency so it gets BUILT
        self.assertIn("in_model", gate.stock_dependencies())
        kw = by["gb_search_3"].step_kwargs
        self.assertEqual(kw["ratchet"], RatchetSchedule(hold=3, release=2, cycles=2))
        np.testing.assert_allclose(kw["ratchet_delta"], GALFOR_RATCHET_DEFAULT_DELTA)
        for s in ("gb_search_1", "gb_search_2"):
            self.assertIsNone(by[s].step_kwargs.get("ratchet"))
            self.assertNotIn("noise_ratchet_search", self._names(by[s]))

    def test_unarmed_composition_is_unchanged(self):
        fit = self._full()
        by = {s.name: self._names(s) for s in fit.recipe.stages}
        self.assertIn("noise_vgb_joint_search", by["gb_search_3"])
        self.assertIn("noise_joint_search_1", by["gb_search_3"])
        self.assertNotIn("noise_ratchet_search", by["gb_search_3"])
        st3 = [s for s in fit.recipe.stages if s.name == "gb_search_3"][0]
        self.assertIsNone(st3.step_kwargs.get("ratchet"))


if __name__ == "__main__":
    unittest.main()
