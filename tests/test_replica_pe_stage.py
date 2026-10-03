"""The replica-PE stage and the staging changes around it (user design 2026-10-03).

"after the ratchets stop (which should be a criteria focused entirely on the
foreground convergence), then we should run in an exact replica of pe mode (and
I mean exact) until the mean leaf count converges over say 10 replica pe
iterations. Then you switch to full PE and mark that as the start of sample
taking. ... can we make gb search 3 only the ratcheting and foreground
convergence? And then gb search 4 will actually be called 'replica pe' ...
The only difference between replica pe and full pe is that the other sources
(emris, mbhs, sobhbs) are still only run every five iterations."

Four pieces, each tested here against the real code path:
1. GFHDFBackend.add_recipe accepts a stage INSERTED after the active one on a
   store that already carries a recipe (the 6mo store is in gb_search_3).
2. ReplicaPERecipeStep ends on flat trends of min/mean/max leaves and lnL.
3. gb_search_3 ends at the ratchet's stop (GALFOR_RATCHET_END_STAGE_ON_STOP).
4. run_combined_staged composes replica_pe = full_pe + the source cadence.
"""

import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "scripts", "fstat_proposal"))

from tests.test_noise_ratchet import _FakeGate, _FakeGrid, _FakeSampler, env  # noqa: E402
from tests.test_ratchet_stop_galfor import _sample, _walkers  # noqa: E402

STORED = [("gb_search_seed", True), ("gb_search_1", True), ("gb_search_2", True),
          ("gb_search_3", False), ("full_pe", False)]


# ----------------------------------------------------------------------------
# 1. the store takes an inserted stage
# ----------------------------------------------------------------------------
class _Store:
    """The slice of GFHDFBackend that add_recipe touches, over a real h5 file."""

    name = "global_fit"

    def __init__(self, path):
        self.filename = path

    def open(self, mode="r"):
        return h5py.File(self.filename, mode)

    @property
    def has_recipe(self):
        with h5py.File(self.filename, "r") as f:
            return bool(f[self.name].attrs["has_recipe"])


def _make_store(path, steps=STORED, stamps=None):
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["has_recipe"] = True
        r = g.create_group("recipe")
        for i, (name, status) in enumerate(steps):
            s = r.create_group(name)
            s.attrs["status"] = bool(status)
            s.attrs["order num"] = i + 1
        for name, kv in (stamps or {}).items():
            for k, v in kv.items():
                r[name].attrs[k] = v


def _recipe(names):
    return SimpleNamespace(recipe=[{"name": n, "adjust": None, "status": False} for n in names])


def _read(path):
    with h5py.File(path, "r") as f:
        r = f["global_fit/recipe"]
        return {k: dict(r[k].attrs) for k in r}


class AddRecipeMigrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="recipe_mig_")
        self.path = os.path.join(self.tmp, "run_testing.h5")
        # the 6mo store as it stands: in gb_search_3, full_pe carrying a stale start stamp
        _make_store(self.path, stamps={"gb_search_3": {"start_iteration": 47},
                                       "full_pe": {"start_iteration": 55},
                                       "gb_search_2": {"start_iteration": 36,
                                                       "completed_iteration": 47}})
        from lisatools.globalfit.hdfbackend import GFHDFBackend

        self.add = GFHDFBackend.add_recipe
        self.store = _Store(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inserting_replica_pe_after_the_active_stage(self):
        new = ["gb_search_seed", "gb_search_1", "gb_search_2", "gb_search_3", "replica_pe", "full_pe"]
        rec = _recipe(new)
        with self.assertLogs("lisatools.globalfit.hdfbackend", level="INFO") as cm:
            self.add(self.store, rec)
        self.assertTrue(any("EXTENDED" in l and "replica_pe" in l for l in cm.output), cm.output)
        a = _read(self.path)
        self.assertEqual(sorted(a), sorted(new))
        self.assertEqual([a[n]["order num"] for n in new], [1, 2, 3, 4, 5, 6])
        self.assertFalse(bool(a["replica_pe"]["status"]))
        self.assertEqual([s["status"] for s in rec.recipe], [True, True, True, False, False, False])
        self.assertEqual(int(a["gb_search_3"]["start_iteration"]), 47)      # the active stage keeps its stamp
        self.assertNotIn("start_iteration", a["full_pe"])                   # the stale 55 is gone
        self.assertEqual(int(a["gb_search_2"]["completed_iteration"]), 47)  # completed stamps untouched
        # a second launch with the same recipe: nothing to add, statuses read, no renumbering
        rec2 = _recipe(new)
        self.add(self.store, rec2)
        self.assertEqual([s["status"] for s in rec2.recipe], [True, True, True, False, False, False])
        self.assertEqual([_read(self.path)[n]["order num"] for n in new], [1, 2, 3, 4, 5, 6])

    def test_unchanged_recipe_reads_statuses_as_before(self):
        rec = _recipe([n for n, _ in STORED])
        self.add(self.store, rec)
        self.assertEqual([s["status"] for s in rec.recipe], [True, True, True, False, False])
        self.assertIn("start_iteration", _read(self.path)["full_pe"])      # untouched when nothing is added

    def test_refusals(self):
        with self.assertRaises(ValueError):          # a stored step dropped
            self.add(self.store, _recipe(["gb_search_seed", "gb_search_1", "gb_search_2", "gb_search_3"]))
        with self.assertRaises(ValueError):          # stored steps reordered
            self.add(self.store, _recipe(["gb_search_seed", "gb_search_2", "gb_search_1", "gb_search_3", "full_pe"]))
        with self.assertRaises(ValueError):          # inserted BEFORE the active stage
            self.add(self.store, _recipe(["gb_search_seed", "gb_search_1", "gb_search_2", "new_stage", "gb_search_3", "full_pe"]))
        with self.assertRaises(ValueError):          # inserted among the completed ones
            self.add(self.store, _recipe(["gb_search_seed", "new_stage", "gb_search_1", "gb_search_2", "gb_search_3", "full_pe"]))
        self.assertEqual(sorted(_read(self.path)), sorted(n for n, _ in STORED))   # nothing written

    def test_a_finished_recipe_can_be_extended_at_the_end(self):
        p = os.path.join(self.tmp, "done_testing.h5")
        _make_store(p, steps=[("a", True), ("b", True)])
        rec = _recipe(["a", "b", "c"])
        self.add(_Store(p), rec)
        self.assertEqual([s["status"] for s in rec.recipe], [True, True, False])
        self.assertEqual(_read(p)["c"]["order num"], 3)


# ----------------------------------------------------------------------------
# 2. the replica step's stop
# ----------------------------------------------------------------------------
class _Backend:
    def __init__(self, leaves, lnl):
        self.leaves, self.lnl = np.asarray(leaves, float), np.asarray(lnl, float)
        self.iteration = self.leaves.shape[0]

    def get_nleaves(self, branch_names=None, temp_index=0):
        return {branch_names[0]: self.leaves}

    def get_log_like(self):
        return self.lnl[:, None, :]                    # (rows, ntemps=1, nwalkers): the store's extra axis

    def stage_start_iteration(self, name):
        return None


def _sampler(be):
    return SimpleNamespace(backend=be, moves=None, weights=None, periodic=None,
                           temperature_control=None)


def _series(n, nw=4, leaf_drift=0.0, lnl_drift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    base_l = 2400 + np.array([60, 0, 20, 10])[:nw]
    leaves = base_l[None, :] + leaf_drift * np.arange(n)[:, None] / max(n - 1, 1) \
        + rng.normal(0, 3.0, size=(n, nw))
    lnl = 1.0e8 + np.array([0, -300, -120, -500])[:nw][None, :] \
        + lnl_drift * np.arange(n)[:, None] / max(n - 1, 1) + rng.normal(0, 20.0, size=(n, nw))
    return leaves, lnl


class ReplicaStepTest(unittest.TestCase):
    def _step(self, **kw):
        from lisatools.globalfit.recipe import ReplicaPERecipeStep

        kwargs = dict(moves=[SimpleNamespace(periodic=None)], stage_name="replica_pe",
                      window=10, leaf_tol=10.0, lnl_tol=100.0)
        kwargs.update(kw)
        return ReplicaPERecipeStep(**kwargs)

    def test_trend_is_the_lines_total_change(self):
        from lisatools.globalfit.recipe import ReplicaPERecipeStep as S

        self.assertAlmostEqual(S.trend(np.arange(10) * 2.0), 18.0)
        self.assertAlmostEqual(S.trend(np.full(7, 5.0)), 0.0)
        self.assertEqual(S.trend([1.0]), 0.0)

    def test_waits_for_the_window_then_stops_only_when_every_series_is_flat(self):
        st = self._step()
        leaves, lnl = _series(30)
        st.setup_run(100, None, _sampler(_Backend(leaves[:0], lnl[:0])))     # stage starts at row 100
        self.assertEqual(st._stage_start_iter, 100)
        # fewer rows in-stage than the window: hold
        pre = np.zeros((100, 4))
        be = _Backend(np.vstack([pre, leaves[:6]]), np.vstack([pre, lnl[:6]]))
        self.assertFalse(st.stopping_function(105, None, _sampler(be)))
        # flat series over the window: stop, with the sample-taking line
        be = _Backend(np.vstack([pre, leaves[:12]]), np.vstack([pre, lnl[:12]]))
        with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
            self.assertTrue(st.stopping_function(111, None, _sampler(be)))
        self.assertTrue(any("THE START OF SAMPLE TAKING" in l and "row 112" in l for l in cm.output), cm.output)
        self.assertTrue(any("converged 6 of 6" in l for l in cm.output))

    def test_a_drifting_leaf_count_or_lnl_holds_the_stage(self):
        st = self._step()
        pre = np.zeros((50, 4))
        for leaf_drift, lnl_drift in ((40.0, 0.0), (0.0, -900.0), (-30.0, 400.0)):
            leaves, lnl = _series(12, leaf_drift=leaf_drift, lnl_drift=lnl_drift)
            st.setup_run(50, None, _sampler(_Backend(pre, pre)))
            be = _Backend(np.vstack([pre, leaves]), np.vstack([pre, lnl]))
            with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
                self.assertFalse(st.stopping_function(61, None, _sampler(be)), (leaf_drift, lnl_drift))
            self.assertFalse(any("CONVERGED" in l for l in cm.output))
        # only the mean drifts (walker 0 up, walker 1 down by the same amount keeps min/max flat?) --
        # a single series out of tolerance is enough to hold
        leaves, lnl = _series(12)
        lnl[:, 3] -= 300.0 * np.arange(12) / 11.0                           # the min walker slides down
        st.setup_run(50, None, _sampler(_Backend(pre, pre)))
        be = _Backend(np.vstack([pre, leaves]), np.vstack([pre, lnl]))
        self.assertFalse(st.stopping_function(61, None, _sampler(be)))

    def test_min_iters_and_validation(self):
        from lisatools.globalfit.recipe import ReplicaPERecipeStep

        st = self._step(min_iters=20)
        self.assertEqual(st.min_iters, 20)
        leaves, lnl = _series(15)
        st.setup_run(0, None, _sampler(_Backend(leaves[:0], lnl[:0])))
        self.assertFalse(st.stopping_function(14, None, _sampler(_Backend(leaves, lnl))))
        for bad in (dict(window=2), dict(leaf_tol=0.0), dict(lnl_tol=-1.0)):
            with self.assertRaises(ValueError, msg=str(bad)):
                ReplicaPERecipeStep(moves=[SimpleNamespace(periodic=None)], **bad)

    def test_unreadable_store_holds_the_stage_with_a_warning(self):
        st = self._step()
        st.setup_run(0, None, _sampler(_Backend(np.zeros((0, 4)), np.zeros((0, 4)))))
        bad = SimpleNamespace(backend=SimpleNamespace(get_nleaves=lambda **k: (_ for _ in ()).throw(OSError("torn"))),
                              moves=None, weights=None, periodic=None, temperature_control=None)
        with self.assertLogs("lisatools.globalfit.recipe", level="WARNING"):
            self.assertFalse(st.stopping_function(5, None, bad))


# ----------------------------------------------------------------------------
# 3. gb_search_3 ends at the ratchet's stop
# ----------------------------------------------------------------------------
class EndStageOnStopTest(unittest.TestCase):
    ENV = dict(GALFOR_RATCHET_HOLD_STAGE=None, GB_SEARCH_STAGE_END_ON_SHUTOFF=None,
               GALFOR_RATCHET_REFIT_ONLY_ON_NUDGE=None, GALFOR_RATCHET_SHUTOFF_MIN_FREQ=None,
               GALFOR_RATCHET_CLOCK_RESET=None)

    def _mk(self, **kw):
        from lisatools.globalfit.noise_ratchet import RatchetSchedule
        from lisatools.globalfit.recipe import SearchStageProfileStep

        gate, grid = _FakeGate(), _FakeGrid()
        grid._shutoff_w_pending = 5                    # the valve says NOT done
        tree = [SimpleNamespace(moves=[gate, grid])]
        kwargs = dict(moves=tree, convergence_iter=2, plateau_branch="gb", profile={},
                      stage_name="gb_search_3",
                      ratchet=RatchetSchedule(hold=1, release=1, cycles=20, release_first=True),
                      ratchet_delta=np.zeros(5), ratchet_stop_rule="galfor", ratchet_min_drop=0.01)
        kwargs.update(kw)
        return SearchStageProfileStep(**kwargs), gate, grid, tree

    def _drive_to_stop(self, st, tree):
        st.setup_run(47, _sample(1000.0, _walkers(0.0)), _FakeSampler(47, tree))
        st.note_recipe_step(3)
        st.stopping_function(48, _sample(1000.0, _walkers(0.0)), _FakeSampler(48, tree))       # reference release
        st.stopping_function(49, _sample(-9000.0, _walkers(-0.1)), _FakeSampler(49, tree))     # nudge
        return st.stopping_function(50, _sample(1000.0, _walkers(0.0)), _FakeSampler(50, tree))  # flat release -> stop

    def test_the_stage_ends_with_the_ratchet_when_the_flag_is_on(self):
        st, gate, grid, tree = self._mk(ratchet_end_stage_on_stop=True)
        with env(**self.ENV):
            with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
                ended = self._drive_to_stop(st, tree)
        self.assertTrue(st._ratchet_stopped)
        self.assertTrue(ended)                                               # despite the valve pending
        self.assertTrue(any("STAGE COMPLETE at the ratchet's stop" in l for l in cm.output), cm.output)
        self.assertTrue(gate.ratchet_finished)
        self.assertFalse(getattr(grid, "fstat_refit_only_forced", False))   # cadence handed back at stage end
        self.assertEqual(float(getattr(grid, "rj_shutoff_min_freq", 0.0) or 0.0), 0.0)

    def test_without_the_flag_the_valve_still_decides(self):
        st, gate, grid, tree = self._mk(ratchet_end_stage_on_stop=False)
        with env(**self.ENV):
            ended = self._drive_to_stop(st, tree)
        self.assertTrue(st._ratchet_stopped)
        self.assertFalse(ended)                                              # held by the valve, as before

    def test_a_stamped_stop_ends_the_stage_at_the_first_check_of_a_relaunch(self):
        st, gate, grid, tree = self._mk(ratchet_end_stage_on_stop=True)
        with env(**self.ENV):
            st.setup_run(60, _sample(1000.0, _walkers(0.0)), _FakeSampler(60, tree))
            st._ratchet_backend = SimpleNamespace(stage_flag=lambda s, k: 1,
                                                  stage_start_iteration=lambda n: 47,
                                                  stamp_stage_flag=lambda *a: True)
            st.note_recipe_step(3)
            self.assertTrue(st._ratchet_stopped)
            self.assertTrue(st.stopping_function(61, _sample(1000.0, _walkers(0.0)),
                                                 _FakeSampler(61, tree)))


# ----------------------------------------------------------------------------
# 4. composition
# ----------------------------------------------------------------------------
class CompositionTest(unittest.TestCase):
    BASE = dict(GB_SEARCH_IN_MODEL="1", GB_SEARCH_RJ_REPLACE="0", GB_SEARCH_IN_MODEL_REPLACE="0",
                GB_WARM_START_COMPONENTS="/nonexistent/warm.npz", STAGE_V9_SEARCH="1",
                GB_SEARCH_3_WARM_EVERY="5", MBHB_IDS="2,5", STAGE_SKIP_SOURCE_SEARCH="1",
                VGB_CHIRP_MASS_BASIS="1", PSD_START_PARAMS=None, GALFOR_START_PARAMS=None,
                GALFOR_RATCHET="1", GALFOR_RATCHET_RELEASE_FIRST="1", GALFOR_RATCHET_HOLD="1",
                GALFOR_RATCHET_RELEASE="1", GALFOR_RATCHET_CYCLES="20",
                STAGE_REPLICA_PE=None, REPLICA_PE_WINDOW=None, REPLICA_PE_LEAF_TOL=None,
                REPLICA_PE_LNL_TOL=None, REPLICA_PE_MIN_ITERS=None,
                GALFOR_RATCHET_END_STAGE_ON_STOP=None, GB_SEARCH_SOURCE_EVERY=None)

    def tearDown(self):
        from lisatools.sampling.fstat_proposal import set_peak_min_F_override

        set_peak_min_F_override(None)

    def _fit(self, **over):
        import run_combined_staged as R

        with env(**{**self.BASE, **over}):
            return R.build_fit()

    def test_replica_pe_is_full_pe_with_the_source_cadence(self):
        fit = self._fit()
        by = {s.name: s for s in fit.recipe.stages}
        self.assertEqual([s.name for s in fit.recipe.stages][-3:], ["gb_search_3", "replica_pe", "full_pe"])
        rep, full = by["replica_pe"], by["full_pe"]
        self.assertEqual(rep.kind, "replica_pe")
        self.assertEqual(rep.runtime_kind, "pe")
        self.assertEqual(rep.move_names(), full.move_names())                # same moves, same order
        src = {"sobbh_pe", "mbh_pe", "emri_pe"}
        for mr, mf in zip(rep.moves, full.moves):
            self.assertEqual(mr.branch, mf.branch)
            if mr.name in src:
                self.assertEqual((mr.every, mf.every), (5, 1), mr.name)    # THE one difference
            else:
                self.assertEqual((mr.every, mf.every), (1, 1), mr.name)
        self.assertIn("mbh_pe", rep.move_names())
        for k in ("peak_min_snr", "pe_repeats", "pe_rj_flip_fraction"):
            self.assertEqual(rep.step_kwargs[k], full.step_kwargs[k], k)      # same declarations
        self.assertEqual(rep.step_kwargs["window"], 10)
        self.assertEqual(rep.step_kwargs["leaf_tol"], 10.0)
        self.assertEqual(rep.step_kwargs["lnl_tol"], 100.0)
        self.assertEqual(rep.step_kwargs["min_iters"], 10)
        self.assertFalse(rep.combine_kwargs["weighted_cycle"])
        self.assertEqual(rep.combine_kwargs["random_choice"], full.combine_kwargs["random_choice"])
        self.assertTrue(by["gb_search_3"].step_kwargs["ratchet_end_stage_on_stop"])
        self.assertNotIn("ratchet_end_stage_on_stop", full.step_kwargs)

    def test_knobs_and_opt_out(self):
        fit = self._fit(STAGE_REPLICA_PE="0", GALFOR_RATCHET_END_STAGE_ON_STOP="0")
        names = [s.name for s in fit.recipe.stages]
        self.assertNotIn("replica_pe", names)
        self.assertEqual(names[-2:], ["gb_search_3", "full_pe"])
        self.assertFalse({s.name: s for s in fit.recipe.stages}["gb_search_3"]
                         .step_kwargs["ratchet_end_stage_on_stop"])
        fit = self._fit(REPLICA_PE_WINDOW="20", REPLICA_PE_LEAF_TOL="5", REPLICA_PE_LNL_TOL="50",
                        REPLICA_PE_MIN_ITERS="30", GB_SEARCH_SOURCE_EVERY="3")
        rep = {s.name: s for s in fit.recipe.stages}["replica_pe"]
        self.assertEqual((rep.step_kwargs["window"], rep.step_kwargs["leaf_tol"],
                          rep.step_kwargs["lnl_tol"], rep.step_kwargs["min_iters"]), (20, 5.0, 50.0, 30))
        self.assertEqual({m.name: m.every for m in rep.moves}["mbh_pe"], 3)
        import run_combined_staged as R

        for bad in (dict(REPLICA_PE_WINDOW="2"), dict(REPLICA_PE_LEAF_TOL="0"),
                    dict(REPLICA_PE_MIN_ITERS="4")):
            with env(**{**self.BASE, **bad}), self.assertRaises(ValueError, msg=str(bad)):
                R.build_fit()

    def test_the_replica_stage_materializes_to_the_replica_step(self):
        from lisatools.globalfit.recipe import ReplicaPERecipeStep, Stage, _STEP_CLASSES

        self.assertIs(_STEP_CLASSES["replica_pe"], ReplicaPERecipeStep)
        self.assertEqual(Stage("x", kind="replica_pe").runtime_kind, "pe")


if __name__ == "__main__":
    unittest.main()
