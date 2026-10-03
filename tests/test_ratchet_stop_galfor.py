"""The ratchet's stop decides on the released FOREGROUND, not the raw lnL gain.

User ruling 2026-10-03 ("lets adjust away from the raw lnL gain ... We are
talking about gating"): with the prior RJ move in births+deaths mode the cold
chain relaxes toward the typical set (job 706: -360..-920 lnL per walker in one
prior-removal leg), so the max cold lnL at successive release wraps can fall
for reasons unrelated to the foreground. The stop now compares each walker's
released galfor curve over a band with the same walker's curve at the previous
release; the median walker must still show a drop of ``ratchet_min_drop`` for
the ratchet to continue. The lnL gain stays in the log as information.

Every test below runs the real :class:`SearchStageProfileStep` machinery
through ``stopping_function`` with the fake gate / grid / sampler of
tests/test_noise_ratchet.py, so the stop's side effects (gate finished,
cadence and floor restored, stamp attempted, no further refit) are exercised,
not just the arithmetic.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "scripts", "fstat_proposal"))

from tests.test_noise_ratchet import _FakeGate, _FakeGrid, _FakeSampler, env  # noqa: E402

BASE = np.array([-43.83, -2.63, 5.0, -2.2, -2.78])          # the 6mo released cold mean


def _coords(rows):
    """branches_coords['galfor'] of shape (ntemps, nwalkers, nleaves, ndim) from (nw, 5)."""
    rows = np.asarray(rows, dtype=float)
    out = np.zeros((2, rows.shape[0], 1, 5))
    out[0, :, 0, :] = rows
    return out


def _sample(mx, rows):
    """A wrap sample: cold lnL (max over walkers = mx) + per-walker galfor rows."""
    nw = np.shape(rows)[0]
    ll = np.full(nw, mx - 100.0)
    ll[0] = mx
    return SimpleNamespace(log_like=np.vstack([ll, np.zeros(nw)]),
                           branches_coords={"galfor": _coords(rows)})


def _walkers(d_amp, nw=4, spread=0.0):
    """Four walkers at BASE with a common amplitude shift d_amp (dex) and an optional
    per-walker spread of the knee (which the band statistic must not react to)."""
    rows = np.tile(BASE, (nw, 1))
    rows[:, 0] += d_amp
    rows[:, 1] += spread * np.linspace(-1, 1, nw)
    return rows


class EnvTest(unittest.TestCase):

    def test_defaults_and_overrides(self):
        from lisatools.globalfit.noise_ratchet import (
            drop_band_from_env, min_drop_from_env, stop_rule_from_env)

        with env(GALFOR_RATCHET_STOP_RULE=None, GALFOR_RATCHET_MIN_DROP=None,
                 GALFOR_RATCHET_DROP_BAND=None):
            self.assertEqual(stop_rule_from_env(), "galfor")
            self.assertEqual(min_drop_from_env(), 0.01)
            self.assertEqual(drop_band_from_env(), (3e-3, 5e-3))
        with env(GALFOR_RATCHET_STOP_RULE="GAIN", GALFOR_RATCHET_MIN_DROP="0.025",
                 GALFOR_RATCHET_DROP_BAND="2.5e-3, 6e-3"):
            self.assertEqual(stop_rule_from_env(), "gain")
            self.assertEqual(min_drop_from_env(), 0.025)
            self.assertEqual(drop_band_from_env(), (2.5e-3, 6e-3))
        with env(GALFOR_RATCHET_STOP_RULE="off"):
            self.assertEqual(stop_rule_from_env(), "off")
        for bad in (dict(GALFOR_RATCHET_STOP_RULE="lnl"),
                    dict(GALFOR_RATCHET_MIN_DROP="0"), dict(GALFOR_RATCHET_MIN_DROP="1.5"),
                    dict(GALFOR_RATCHET_DROP_BAND="3e-3"), dict(GALFOR_RATCHET_DROP_BAND="5e-3,3e-3"),
                    dict(GALFOR_RATCHET_DROP_BAND="a,b")):
            with env(**bad), self.assertRaises(ValueError, msg=str(bad)):
                stop_rule_from_env(); min_drop_from_env(); drop_band_from_env()


class BandDropTest(unittest.TestCase):

    def test_amplitude_shift_is_exact_and_walkers_are_compared_with_themselves(self):
        from lisatools.globalfit.noise_ratchet import galfor_band_drop

        prev = _walkers(0.0, spread=0.05)             # walkers differ by their knee (posterior spread)
        new = prev.copy()
        new[:, 0] -= 0.05                               # every walker's amplitude x 10^-0.05
        s = galfor_band_drop(new, prev, 3e-3, 5e-3)
        np.testing.assert_allclose(s["ratio_walkers"], 10 ** -0.05, rtol=1e-12)
        self.assertAlmostEqual(s["ratio_median"], 10 ** -0.05, places=12)
        self.assertAlmostEqual(s["drop"], 1 - 10 ** -0.05, places=12)      # 10.9 %
        # identical releases: no drop, whatever the spread between walkers
        z = galfor_band_drop(prev, prev, 3e-3, 5e-3)
        self.assertEqual(z["drop"], 0.0)
        # one wandering walker does not move the median
        odd = new.copy()
        odd[3, 0] += 0.3
        s2 = galfor_band_drop(odd, prev, 3e-3, 5e-3)
        self.assertAlmostEqual(s2["ratio_median"], 10 ** -0.05, places=6)
        # a rise reads as a negative drop
        up = prev.copy()
        up[:, 0] += 0.02
        self.assertLess(galfor_band_drop(up, prev, 3e-3, 5e-3)["drop"], 0.0)
        with self.assertRaises(ValueError):
            galfor_band_drop(new[:3], prev, 3e-3, 5e-3)


class _StepBase(unittest.TestCase):
    ENV = dict(GALFOR_RATCHET_HOLD_STAGE=None, GB_SEARCH_STAGE_END_ON_SHUTOFF=None,
               GALFOR_RATCHET_REFIT_ONLY_ON_NUDGE=None, GALFOR_RATCHET_SHUTOFF_MIN_FREQ=None,
               GALFOR_RATCHET_CLOCK_RESET=None)

    def _mk(self, **kw):
        from lisatools.globalfit.noise_ratchet import RatchetSchedule
        from lisatools.globalfit.recipe import SearchStageProfileStep

        gate, grid = _FakeGate(), _FakeGrid()
        grid._shutoff_w_pending = 0                  # the valve says "done": a stop ends the stage
        tree = [SimpleNamespace(moves=[gate, grid])]
        kwargs = dict(moves=tree, convergence_iter=2, plateau_branch="gb", profile={},
                      stage_name="gb_search_3",
                      ratchet=RatchetSchedule(hold=1, release=1, cycles=20, release_first=True),
                      ratchet_delta=np.zeros(5))
        kwargs.update(kw)
        return SearchStageProfileStep(**kwargs), gate, grid, tree


class GalforRuleTest(_StepBase):

    def test_the_foreground_decides_and_the_lnl_gain_does_not(self):
        """Release 1 drops the foreground 10.9 % while lnL gains only +50 (the
        gain rule at 200 would have STOPPED): continues. Release 2 drops it
        0.7 % while lnL gains +5000 (the gain rule would have CONTINUED): stops,
        with every side effect of a stop."""
        st, gate, grid, tree = self._mk(ratchet_stop_rule="galfor", ratchet_min_drop=0.01,
                                        ratchet_min_gain=200.0)
        with env(**self.ENV):
            st.setup_run(47, _sample(1000.0, _walkers(0.0)), _FakeSampler(47, tree))
            st.note_recipe_step(3)
            self.assertEqual(gate.modes, ["release"])
            # k=0 wrap: the reference release
            self.assertFalse(st.stopping_function(48, _sample(1000.0, _walkers(0.0, spread=0.05)),
                                                  _FakeSampler(48, tree)))
            self.assertEqual(gate.modes[-1], "nudge")
            self.assertIsNotNone(st._ratchet_last_release_galfor)
            # k=1 nudge wrap: foreground held down, lnL way down -- not a release, no verdict
            self.assertFalse(st.stopping_function(49, _sample(-9000.0, _walkers(-0.1)),
                                                  _FakeSampler(49, tree)))
            self.assertEqual(gate.modes[-1], "release")
            # k=2 release wrap: amplitude -0.05 dex on every walker (drop 10.9 %), lnL only +50
            self.assertFalse(st.stopping_function(50, _sample(1050.0, _walkers(-0.05, spread=0.05)),
                                                  _FakeSampler(50, tree)))
            self.assertFalse(st._ratchet_stopped)
            self.assertEqual(gate.modes[-1], "nudge")                 # ratcheting continues
            self.assertEqual(len(grid.armed), 2)
            self.assertAlmostEqual(st._ratchet_release_drops[-1][1], 1 - 10 ** -0.05, places=9)
            self.assertFalse(st.stopping_function(51, _sample(-8000.0, _walkers(-0.15)),
                                                  _FakeSampler(51, tree)))
            # k=4 release wrap: amplitude -0.003 dex more (drop 0.69 % < 1 %), lnL +5000
            with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
                ended = st.stopping_function(52, _sample(6050.0, _walkers(-0.053, spread=0.05)),
                                             _FakeSampler(52, tree))
            self.assertTrue(ended)
            self.assertTrue(st._ratchet_stopped)
            self.assertTrue(gate.ratchet_finished)                      # rider mode
            self.assertFalse(gate.release_to_convergence)
            self.assertNotIn("nudge", gate.modes[-1:])
            self.assertEqual(len(grid.armed), 2)                        # no third refit
            self.assertFalse(getattr(grid, "fstat_refit_only_forced", False))   # cadence back
            self.assertEqual(float(getattr(grid, "rj_shutoff_min_freq", 0.0) or 0.0), 0.0)  # floor lifted
            self.assertEqual([k for k, _ in st._ratchet_release_maxes], [0, 2, 4])
            self.assertEqual([k for k, _ in st._ratchet_release_drops], [2, 4])
            done = [l for l in cm.output if "RATCHET DONE" in l]
            self.assertEqual(len(done), 1, cm.output)
            self.assertIn("released foreground at 3-5 mHz", done[0])
            self.assertIn("+5000.0 over the previous release (informational)", done[0])

    def test_a_rise_stops_too(self):
        st, gate, grid, tree = self._mk(ratchet_stop_rule="galfor", ratchet_min_drop=0.01)
        with env(**self.ENV):
            st.setup_run(47, _sample(1000.0, _walkers(0.0)), _FakeSampler(47, tree))
            st.note_recipe_step(3)
            st.stopping_function(48, _sample(1000.0, _walkers(0.0)), _FakeSampler(48, tree))
            st.stopping_function(49, _sample(-9000.0, _walkers(-0.1)), _FakeSampler(49, tree))
            # the release came back ABOVE the previous one: no step bought
            self.assertTrue(st.stopping_function(50, _sample(3000.0, _walkers(+0.03)),
                                                 _FakeSampler(50, tree)))
            self.assertTrue(st._ratchet_stopped)

    def test_min_nudges_holds_the_galfor_stop(self):
        st, gate, grid, tree = self._mk(ratchet_stop_rule="galfor", ratchet_min_drop=0.01,
                                        ratchet_min_nudges=2)
        with env(**self.ENV):
            st.setup_run(47, _sample(1000.0, _walkers(0.0)), _FakeSampler(47, tree))
            st.note_recipe_step(3)
            st.stopping_function(48, _sample(1000.0, _walkers(0.0)), _FakeSampler(48, tree))
            st.stopping_function(49, _sample(-9000.0, _walkers(-0.1)), _FakeSampler(49, tree))
            # flat release after ONE nudge: below threshold but the floor says continue
            with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
                self.assertFalse(st.stopping_function(50, _sample(1000.0, _walkers(0.0)),
                                                      _FakeSampler(50, tree)))
            self.assertFalse(st._ratchet_stopped)
            self.assertEqual(gate.modes[-1], "nudge")
            self.assertTrue(any("only 1 of the required 2 nudge(s)" in l for l in cm.output))
            st.stopping_function(51, _sample(-9000.0, _walkers(-0.1)), _FakeSampler(51, tree))
            # flat again after TWO nudges: stops
            self.assertTrue(st.stopping_function(52, _sample(1000.0, _walkers(0.0)),
                                                 _FakeSampler(52, tree)))
            self.assertTrue(st._ratchet_stopped)

    def test_missing_galfor_falls_back_to_the_gain_rule_with_a_warning(self):
        st, gate, grid, tree = self._mk(ratchet_stop_rule="galfor", ratchet_min_drop=0.01,
                                        ratchet_min_gain=200.0)

        def bare(mx):
            return SimpleNamespace(log_like=np.array([[mx, mx - 100.0, mx - 50.0, mx - 20.0],
                                                      np.zeros(4)]))

        with env(**self.ENV):
            st.setup_run(47, bare(1000.0), _FakeSampler(47, tree))
            st.note_recipe_step(3)
            st.stopping_function(48, bare(1000.0), _FakeSampler(48, tree))
            st.stopping_function(49, bare(-9000.0), _FakeSampler(49, tree))
            with self.assertLogs("lisatools.globalfit.recipe", level="WARNING") as cm:
                self.assertFalse(st.stopping_function(50, bare(1500.0), _FakeSampler(50, tree)))
            self.assertTrue(any("deciding this release on the lnL gain rule" in l for l in cm.output))
            self.assertFalse(st._ratchet_stopped)                       # +500 >= 200
            st.stopping_function(51, bare(-9000.0), _FakeSampler(51, tree))
            self.assertTrue(st.stopping_function(52, bare(1550.0), _FakeSampler(52, tree)))  # +50 < 200
            self.assertTrue(st._ratchet_stopped)

    def test_gain_rule_is_unchanged_and_off_never_stops(self):
        st, gate, grid, tree = self._mk(ratchet_stop_rule="gain", ratchet_min_gain=200.0)
        with env(**self.ENV):
            st.setup_run(47, _sample(1000.0, _walkers(0.0)), _FakeSampler(47, tree))
            st.note_recipe_step(3)
            st.stopping_function(48, _sample(1000.0, _walkers(0.0)), _FakeSampler(48, tree))
            st.stopping_function(49, _sample(-9000.0, _walkers(-0.1)), _FakeSampler(49, tree))
            # the foreground fell 10.9 % but lnL gained only +50: the GAIN rule stops
            self.assertTrue(st.stopping_function(50, _sample(1050.0, _walkers(-0.05)),
                                                 _FakeSampler(50, tree)))
            self.assertTrue(st._ratchet_stopped)
        st, gate, grid, tree = self._mk(ratchet_stop_rule="off", ratchet_min_gain=200.0)
        with env(**self.ENV):
            st.setup_run(47, _sample(1000.0, _walkers(0.0)), _FakeSampler(47, tree))
            st.note_recipe_step(3)
            for i, (mx, d) in enumerate([(1000.0, 0.0), (-9000.0, -0.1), (1000.0, 0.0),
                                         (-9000.0, -0.1), (1000.0, 0.0)]):
                st.stopping_function(48 + i, _sample(mx, _walkers(d)), _FakeSampler(48 + i, tree))
            self.assertFalse(st._ratchet_stopped)
            # wraps k=0..4 drive k=1..5: nudges at 1, 3, 5 -- nothing stopped them
            self.assertEqual(gate.modes.count("nudge"), 3)
            self.assertFalse(gate.ratchet_finished)

    def test_constructor_refuses_a_bad_rule_or_a_galfor_rule_without_a_threshold(self):
        with self.assertRaises(ValueError):
            self._mk(ratchet_stop_rule="lnl")
        with self.assertRaises(ValueError):
            self._mk(ratchet_stop_rule="galfor", ratchet_min_drop=0.0)
        with self.assertRaises(ValueError):
            self._mk(ratchet_stop_rule="galfor", ratchet_min_drop=0.01, ratchet_drop_band=(5e-3, 3e-3))
        st, *_ = self._mk(ratchet_stop_rule="galfor", ratchet_min_drop=0.01,
                          ratchet_drop_band=(2e-3, 6e-3))
        self.assertEqual(st.ratchet_drop_band, (2e-3, 6e-3))


class CompositionTest(unittest.TestCase):
    """The launcher knobs reach the stage-3 step through run_combined_staged."""

    BASE = dict(GB_SEARCH_IN_MODEL="1", GB_SEARCH_RJ_REPLACE="0", GB_SEARCH_IN_MODEL_REPLACE="0",
                GB_WARM_START_COMPONENTS="/nonexistent/warm.npz", STAGE_V9_SEARCH="1",
                GB_SEARCH_3_WARM_EVERY="5", MBHB_IDS="2,5", STAGE_SKIP_SOURCE_SEARCH="1",
                VGB_CHIRP_MASS_BASIS="1", PSD_START_PARAMS=None, GALFOR_START_PARAMS=None,
                GALFOR_RATCHET="1", GALFOR_RATCHET_RELEASE_FIRST="1", GALFOR_RATCHET_HOLD="1",
                GALFOR_RATCHET_RELEASE="1", GALFOR_RATCHET_CYCLES="20",
                GALFOR_RATCHET_MIN_GAIN="200", GALFOR_RATCHET_STOP_RULE=None,
                GALFOR_RATCHET_MIN_DROP=None, GALFOR_RATCHET_DROP_BAND=None)

    def tearDown(self):
        from lisatools.sampling.fstat_proposal import set_peak_min_F_override

        set_peak_min_F_override(None)

    def _kw(self, **over):
        import run_combined_staged as R

        with env(**{**self.BASE, **over}):
            fit = R.build_fit()
        return {s.name: s for s in fit.recipe.stages}["gb_search_3"].step_kwargs

    def test_defaults_are_the_galfor_rule_and_overrides_reach_the_step(self):
        kw = self._kw()
        self.assertEqual(kw["ratchet_stop_rule"], "galfor")
        self.assertEqual(kw["ratchet_min_drop"], 0.01)
        self.assertEqual(kw["ratchet_drop_band"], (3e-3, 5e-3))
        self.assertEqual(kw["ratchet_min_gain"], 200.0)             # still carried, informational
        kw = self._kw(GALFOR_RATCHET_STOP_RULE="gain", GALFOR_RATCHET_MIN_DROP="0.02",
                      GALFOR_RATCHET_DROP_BAND="2e-3,6e-3")
        self.assertEqual(kw["ratchet_stop_rule"], "gain")
        self.assertEqual(kw["ratchet_min_drop"], 0.02)
        self.assertEqual(kw["ratchet_drop_band"], (2e-3, 6e-3))

    def test_a_bad_knob_refuses_the_launch(self):
        import run_combined_staged as R

        with env(**{**self.BASE, "GALFOR_RATCHET_STOP_RULE": "lnl"}):
            with self.assertRaises(ValueError):
                R.build_fit()

    def test_an_unratcheted_launch_reads_no_new_knobs(self):
        kw = self._kw(GALFOR_RATCHET="0", GALFOR_RATCHET_STOP_RULE="lnl")   # would raise if read
        self.assertIsNone(kw.get("ratchet"))
        self.assertEqual(kw["ratchet_stop_rule"], "gain")


if __name__ == "__main__":
    unittest.main()
