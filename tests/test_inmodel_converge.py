"""Convergence-driven in-model polish for freshly-born GB sources.

Opt-in mode (``{BRANCH}_INMODEL_CONVERGE``, default ``off``) that turns the
fixed per-class in-model repeat budget into a CEILING and lets each row stop
when its own likelihood stops climbing: a converged row is frozen (dropped
from the per-half row sets, so it costs nothing) but stays resident, and the
block ends when every row is frozen. Steps become unique per source and may
exceed the budget.

Fake-based (the style of ``test_inmodel_repeats`` / ``test_vertical_swap``):
no waveform, no buffer, no GPU. Covered here:

* knob resolution (kwarg > env > default, per-branch prefix, validation,
  the tri-state off/observe/on switch);
* the rule -- flat, above-rate, below-rate, late spike, latching, row
  independence -- plus a regression guard pinning it AGAINST the cap gate's
  literal form, which on a per-repeat clock retires a row mid-climb;
* state persistence across blocks, the ring buffer above all, and the
  column-AND retirement rule ("all temperatures travel together");
* the LADDER GATE -- the hot half of the ladder gets no vote on when the
  block may stop, and a climbing hot rung must not hold it open while a
  climbing cold one must;
* the COLUMN REFILL loop -- a finished band leaves, a queued one takes its
  slots, carried columns keep their state, and a generation that retires
  nothing caps rather than spins;
* the search-vs-PE stage gate (optional stopping is search-only);
* END TO END through the real ``_run_in_model_repeats``: off is
  bit-identical, the freeze shrinks the per-repeat work, frozen rows stop
  proposing within one poll, ``observe`` changes nothing while still filling
  in the bookkeeping, the ceiling bounds a never-converging block, and the
  whole thing survives ``GB_TEMPER_VERTICAL=1``.

Design: docs/superpowers/specs/2026-09-24-gb-inmodel-convergence-design.md
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    _InModelConvergeState,
    _converge_cast_classes,
    _converge_cast_mode,
    _converge_cast_swap_frac,
    _converge_cast_window,
    _converge_stage_allows,
    _resolve_converge_knob,
)


# --------------------------------------------------------------------------
# knob resolution
# --------------------------------------------------------------------------
class ConvergeKnobTest(unittest.TestCase):
    def test_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_CONVERGE", None)
            self.assertEqual(
                _resolve_converge_knob(
                    "gb", "", None, "off", _converge_cast_mode),
                "off",
            )

    def test_kwarg_wins_over_env(self):
        with mock.patch.dict(os.environ, {"GB_INMODEL_CONVERGE_ITERS": "7"}):
            self.assertEqual(
                _resolve_converge_knob(
                    "gb", "iters", 13, 50, _converge_cast_window),
                13,
            )

    def test_env_wins_over_default(self):
        with mock.patch.dict(os.environ, {"GB_INMODEL_CONVERGE_ITERS": "7"}):
            self.assertEqual(
                _resolve_converge_knob(
                    "gb", "iters", None, 50, _converge_cast_window),
                7,
            )

    def test_branch_prefix_does_not_leak(self):
        with mock.patch.dict(os.environ, {"GB_INMODEL_CONVERGE": "1"}):
            self.assertEqual(
                _resolve_converge_knob(
                    "gb", "", None, "off", _converge_cast_mode),
                "on",
            )
            self.assertEqual(
                _resolve_converge_knob(
                    "vgb", "", None, "off", _converge_cast_mode),
                "off",
            )

    def test_mode_is_tri_state(self):
        self.assertEqual(_converge_cast_mode("observe"), "observe")
        self.assertEqual(_converge_cast_mode("1"), "on")
        self.assertEqual(_converge_cast_mode(False), "off")
        with self.assertRaises(ValueError):
            _converge_cast_mode("yes please")

    def test_window_must_be_positive(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                _converge_cast_window(bad)

    def test_swap_frac_range(self):
        self.assertAlmostEqual(_converge_cast_swap_frac("0.5"), 0.5)
        self.assertAlmostEqual(_converge_cast_swap_frac(1.0), 1.0)
        for bad in (0.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                _converge_cast_swap_frac(bad)

    def test_classes_parsing(self):
        self.assertEqual(_converge_cast_classes("newborn"), {"newborn"})
        self.assertEqual(
            _converge_cast_classes(" newborn , MATURE "),
            {"newborn", "mature"},
        )
        for bad in ("", "survivor", "newborn,bogus"):
            with self.assertRaises(ValueError):
                _converge_cast_classes(bad)


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------
def _run_rule(trace, window, thresh):
    """Drive ``_InModelConvergeState.step`` over a single column's trace.

    Returns the 1-based repeat at which the column would be retired, or
    ``len(trace)`` if it never is.
    """
    best = np.full(1, -np.inf)
    ring = np.full((window, 1), -np.inf)
    seen = np.zeros(1, dtype=np.int64)
    done = np.zeros(1, dtype=bool)
    cols = np.zeros(1, dtype=np.int64)
    for i, v in enumerate(trace):
        best, done = _InModelConvergeState.step(
            np, np.array([v]), best, ring, seen, cols, window, thresh, done
        )
        if bool(done[0]):
            return i + 1
    return len(trace)


class ConvergeRuleTest(unittest.TestCase):
    def test_flat_trace_fires_exactly_one_repeat_after_the_window(self):
        # best is stored at repeat 0; the first comparison is at seen == W,
        # i.e. the (W+1)-th repeat.
        for window in (3, 10, 50):
            with self.subTest(window=window):
                self.assertEqual(
                    _run_rule(np.zeros(4 * window), window, 4.0), window + 1
                )

    def test_steady_climb_above_the_rate_never_fires(self):
        # thresh/window = 0.08 lnL/repeat at (4.0, 50); climb twice that.
        trace = np.arange(500) * 0.16
        self.assertEqual(_run_rule(trace, 50, 4.0), 500)

    def test_steady_climb_below_the_rate_fires(self):
        trace = np.arange(500) * 0.04
        self.assertEqual(_run_rule(trace, 50, 4.0), 51)

    def test_the_cap_gate_form_would_retire_mid_climb(self):
        """Regression guard for the rule this design deliberately rejects.

        ``_update_band_leaf_caps`` resets patience whenever a single tick
        beats ``best + D/2``. Its tick is a whole sampler iteration; a tick
        here is one unthinned MH step, so that form degenerates into a
        per-step test. This asserts the two rules DISAGREE on a climbing
        trace -- if a future refactor makes them agree, the ring buffer has
        been lost.
        """
        trace = np.linspace(-30.0, 0.0, 60)          # 0.5 lnL/repeat
        # the shipped rule keeps going: the climb beats 4.0 per 50 repeats
        self.assertEqual(_run_rule(trace, 50, 4.0), 60)
        # the cap-gate form retires it after 50 sub-threshold single steps
        best, patience, fired = -np.inf, 0, None
        for i, v in enumerate(trace):
            improved = v > best + 4.0
            best = max(best, v)
            patience = 0 if improved else patience + 1
            if patience >= 50:
                fired = i + 1
                break
        self.assertEqual(fired, 51)

    def test_a_late_spike_resets_the_clock(self):
        trace = np.concatenate([np.zeros(40), [99.0], np.zeros(200)])
        # best jumps at repeat 41 (index 40), so the first comparison that
        # can pass is the one whose window OPENS there: index 40 + 50 = 90,
        # i.e. the 91st repeat. A full window after the spike, exactly.
        self.assertEqual(_run_rule(trace, 50, 4.0), 91)
        # without the spike the same trace retires at repeat 51
        self.assertEqual(_run_rule(np.zeros(241), 50, 4.0), 51)

    def test_done_latches(self):
        best = np.full(1, -np.inf)
        ring = np.full((3, 1), -np.inf)
        seen = np.zeros(1, dtype=np.int64)
        done = np.zeros(1, dtype=bool)
        cols = np.zeros(1, dtype=np.int64)
        for v in (0.0, 0.0, 0.0, 0.0, 1e6):
            best, done = _InModelConvergeState.step(
                np, np.array([v]), best, ring, seen, cols, 3, 4.0, done
            )
        self.assertTrue(bool(done[0]))

    def test_rows_are_independent(self):
        window, thresh = 5, 4.0
        best = np.full(2, -np.inf)
        ring = np.full((window, 2), -np.inf)
        seen = np.zeros(2, dtype=np.int64)
        done = np.zeros(2, dtype=bool)
        cols = np.arange(2)
        # column 0 flat, column 1 climbing hard
        for i in range(20):
            best, done = _InModelConvergeState.step(
                np, np.array([0.0, 10.0 * i]), best, ring, seen, cols,
                window, thresh, done,
            )
        self.assertTrue(bool(done[0]))
        self.assertFalse(bool(done[1]))


# --------------------------------------------------------------------------
# state persistence across generations
# --------------------------------------------------------------------------
class ConvergeStateTest(unittest.TestCase):
    def _drive(self, st, rows, trace_by_row, n_rep):
        """Run ``n_rep`` repeats of one block over ``rows`` (source ids).

        ``trace_by_row`` gives each row's cumulative GAIN per repeat, i.e.
        what the block's ``_cv_gain`` accumulator would hold.
        """
        best, ring, seen, gain, done, at = st.gather(rows)
        rows_ar = np.arange(len(rows))
        base = [st.clock(r) for r in rows]
        for i in range(n_rep):
            gain = np.array([trace_by_row[r][base[j] + i]
                             for j, r in enumerate(rows)])
            best, done, at = _InModelConvergeState.step(
                np, gain, best, ring, seen, rows_ar, st.window, st.thresh,
                done, at,
            )
        st.absorb(rows, best, ring, seen, gain, done, at)
        return done

    def test_carryover_does_not_re_earn_the_warmup(self):
        """A row carried into a second block must be retirable on its
        FIRST repeat there -- the ring buffer persists."""
        st = _InModelConvergeState(window=10, thresh=4.0, max_repeats=1000)
        flat = {7: np.zeros(200)}
        # block 1: 10 repeats -- one short of the first comparison
        done = self._drive(st, [7], flat, 10)
        self.assertFalse(bool(done[0]))
        self.assertEqual(st.clock(7), 10)
        # block 2: the very first repeat completes the window
        done = self._drive(st, [7], flat, 1)
        self.assertTrue(bool(done[0]))
        self.assertIn(7, st.converged)

    def test_best_persists_so_a_rebind_is_not_an_improvement(self):
        st = _InModelConvergeState(window=5, thresh=4.0, max_repeats=1000)
        tr = {3: np.concatenate([np.linspace(0, 100, 20), np.zeros(200)])}
        self._drive(st, [3], tr, 20)
        best_g1 = st.gather([3])[0]
        self._drive(st, [3], tr, 1)
        best_g2 = st.gather([3])[0]
        # the second block's value (0.0) is far below the carried best
        self.assertAlmostEqual(float(best_g1[0]), float(best_g2[0]))
        self.assertAlmostEqual(float(best_g2[0]), 100.0)

    def test_done_is_carried_back_in_so_a_frozen_row_stays_frozen(self):
        st = _InModelConvergeState(window=3, thresh=4.0, max_repeats=1000)
        flat = {5: np.zeros(100)}
        self._drive(st, [5], flat, 10)
        self.assertIn(5, st.converged)
        # a later block that re-gathers the row must see it already done
        done = st.gather([5])[4]
        self.assertTrue(bool(done[0]))

    def test_max_repeats_caps_a_never_converging_row(self):
        st = _InModelConvergeState(window=10, thresh=4.0, max_repeats=30)
        climb = {1: np.arange(400) * 10.0}
        for _ in range(3):
            self._drive(st, [1], climb, 10)
        self.assertNotIn(1, st.converged)
        self.assertIn(1, st.capped)
        self.assertTrue(st.row_retired(1))

    def test_unseen_row_starts_clean(self):
        st = _InModelConvergeState(window=4, thresh=4.0, max_repeats=100)
        best, ring, seen, gain, done, _at = st.gather([42])
        self.assertEqual(float(best[0]), -np.inf)
        self.assertEqual(int(seen[0]), 0)
        self.assertEqual(float(gain[0]), 0.0)
        self.assertTrue(np.all(np.isneginf(ring)))
        self.assertFalse(bool(done[0]))
        self.assertFalse(st.row_retired(42))

    def test_column_leaves_only_when_every_rung_is_done(self):
        """USER SPEC: temperatures travel together. A column with one
        unconverged rung stays in the block even if the rest are frozen."""
        st = _InModelConvergeState(window=5, thresh=4.0, max_repeats=1000)
        rows = [10, 11, 12, 13]          # four rungs of one column
        traces = {
            10: np.zeros(400),                      # converges at once
            11: np.zeros(400),
            12: np.zeros(400),
            13: np.arange(400) * 10.0,              # climbs forever
        }
        self._drive(st, rows, traces, 20)
        self.assertTrue(st.row_retired(10))
        self.assertTrue(st.row_retired(11))
        self.assertTrue(st.row_retired(12))
        self.assertFalse(st.row_retired(13))
        self.assertFalse(st.column_retired(rows))
        # once the straggler hits the ceiling the column may leave
        st.max_repeats = 20
        self._drive(st, rows, traces, 1)
        self.assertTrue(st.row_retired(13))
        self.assertTrue(st.column_retired(rows))


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# END TO END through the REAL ``_run_in_model_repeats`` (numpy harness)
# --------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from lisatools.globalfit.moves.gbspecialstretch import (  # noqa: E402
    _CONVERGE_POLL_EVERY,
)
from tests.test_inmodel_repeats import (  # noqa: E402
    _FakeBuffer,
    _make_move,
)

NTEMPS, NWALKERS, NBANDS = 4, 2, 3


def _column_problem(seed=5):
    """One row per (temp, walker, band): 4x2x3 = 24 rows in 6 columns."""
    rng = np.random.RandomState(seed)
    t, w, b = np.meshgrid(
        np.arange(NTEMPS), np.arange(NWALKERS), np.arange(NBANDS),
        indexing="ij",
    )
    # column order: (band, walker, temp) with temp innermost
    order = np.lexsort((t.ravel(), w.ravel(), b.ravel()))
    t, w, b = t.ravel()[order], w.ravel()[order], b.ravel()[order]
    n = t.size
    n_src = 2 * n
    coords = np.zeros((n_src, 4))
    coords[:, 0] = rng.uniform(-0.5, 0.5, n_src)
    coords[:, 1] = rng.uniform(2.95, 3.05, n_src)
    coords[:, 2] = rng.uniform(-1, 1, n_src)
    coords[:, 3] = rng.uniform(-1, 1, n_src)
    ids = np.arange(0, 2 * n, 2)
    picked = {
        "ids": ids,
        "specials": (t * NWALKERS * NBANDS + w * NBANDS + b),
        "slot_index": np.arange(n, dtype=np.int32),
        "temp_inds": t,
        "walker_inds": w,
        "band_inds": b,
        "N_vals": np.full(n, 64),
    }
    sorter = SimpleNamespace(
        inds=np.ones(n_src, dtype=bool),
        coords=coords.copy(),
        leaf_inds=np.arange(n_src),
    )
    band_temps = np.tile(
        np.array([1.0, 0.5, 0.25, 0.1])[None, :NTEMPS], (NBANDS, 1)
    )
    return picked, sorter, band_temps, n


class _CountingBuffer(_FakeBuffer):
    """Counts the rows handed to ``get_add_ll`` -- the per-repeat work."""

    def __init__(self, n_slots):
        super().__init__(n_slots)
        self.scored_rows = []

    def get_add_ll(self, params, slots_in, slots_out, N,
                   phase_maximize=False, leaf_inds=None):
        self.scored_rows.append(int(np.shape(params)[0]))
        return super().get_add_ll(
            params, slots_in, slots_out, N,
            phase_maximize=phase_maximize, leaf_inds=leaf_inds,
        )


def _run_block(n_rep, converge, seed=2026, vertical=False):
    picked, sorter, band_temps, n = _column_problem()
    move = _make_move(n_rep)
    move.ntemps = NTEMPS
    move.nwalkers = NWALKERS
    move.num_bands = NBANDS
    move.sequential_parity_repeats = False
    move.temper_vertical = vertical
    buf = _CountingBuffer(n)
    ll_change = np.zeros((NTEMPS, NWALKERS, NBANDS))
    prop = np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int)
    acc = np.zeros_like(prop)
    np.random.seed(seed)
    move._run_in_model_repeats(
        None, sorter, buf, band_temps, picked, ll_change, prop, acc,
        num_repeats=n_rep, converge=converge,
    )
    return move, sorter, picked, buf, prop, acc


class ConvergeNoOpTest(unittest.TestCase):
    """``converge=None`` must be byte-for-byte the historical block."""

    def test_off_is_bit_identical(self):
        a = _run_block(30, None, seed=7)
        b = _run_block(30, None, seed=7)
        np.testing.assert_array_equal(a[1].coords, b[1].coords)
        self.assertEqual(a[3].scored_rows, b[3].scored_rows)
        # every row proposed on every repeat: nothing froze. (The SCORED
        # count varies repeat to repeat even here -- the f0-window / prior
        # gate compacts rows before the kernel -- so the proposal counters,
        # not the scored counts, are what proves no freezing happened.)
        self.assertEqual(sum(a[4][1].ravel()), 24 * 30)

    def test_off_leaves_no_state_on_the_move(self):
        move = _run_block(10, None)[0]
        for attr in ("_converge_armed_logged", "_converge_pe_warned"):
            self.assertFalse(hasattr(move, attr))


class ConvergeBlockTest(unittest.TestCase):
    """The freeze and the early exit, through the real repeat loop."""

    def _state(self, window=5, thresh=4.0, max_repeats=500, **kw):
        return _InModelConvergeState(
            window=window, thresh=thresh, max_repeats=max_repeats, **kw)

    def test_block_stops_once_every_row_has_converged(self):
        # the fake likelihood is a fixed quadratic: the chain equilibrates
        # fast, so with a 5-repeat window every row retires well inside 300.
        st = self._state(window=5)
        move, sorter, picked, buf, prop, acc = _run_block(300, st)
        rows = [int(i) for i in picked["ids"]]
        self.assertTrue(all(st.row_retired(r) for r in rows))
        # ... and the block stopped early rather than burning the ceiling
        self.assertLess(len(buf.scored_rows), 300)
        self.assertGreater(len(buf.scored_rows), 5)

    def test_converged_rows_stop_being_scored(self):
        """The freeze must actually shrink the per-repeat work.

        Compared in AGGREGATE, not repeat by repeat: the f0-window / prior
        gate already compacts rows before the scoring kernel, so the
        per-repeat scored count fluctuates for reasons that have nothing
        to do with freezing. The signal is that the last poll window
        scores strictly fewer rows than the first.
        """
        st = self._state(window=5)
        move, sorter, picked, buf, prop, acc = _run_block(300, st)
        rows = buf.scored_rows
        self.assertGreater(len(rows), 2 * _CONVERGE_POLL_EVERY)
        head = float(np.mean(rows[:_CONVERGE_POLL_EVERY]))
        tail = float(np.mean(rows[-_CONVERGE_POLL_EVERY:]))
        self.assertLess(tail, head)

    def test_frozen_rows_stop_proposing_entirely(self):
        """A frozen row must post no further proposals in the counters."""
        st = self._state(window=5)
        move, sorter, picked, buf, prop, acc = _run_block(300, st)
        t, w, b = (picked["temp_inds"], picked["walker_inds"],
                   picked["band_inds"])
        reps = np.array([st.repeats(int(i)) for i in picked["ids"]])
        proposed = prop[1][t, w, b]
        # a row proposes until the poll that notices it froze, so its
        # proposal count sits between its freeze repeat and one poll later
        self.assertTrue(np.all(proposed >= reps), f"{proposed} < {reps}")
        self.assertTrue(
            np.all(proposed <= reps + _CONVERGE_POLL_EVERY),
            f"{proposed} > {reps} + {_CONVERGE_POLL_EVERY}",
        )
        # and no row kept proposing for the whole ceiling
        self.assertLess(int(proposed.max()), 300)

    def test_observe_mode_changes_nothing(self):
        """OBSERVE must leave the chain bit-identical to the fixed budget
        while still filling in the convergence bookkeeping."""
        st = self._state(window=5, observe=True)
        obs = _run_block(40, st, seed=99)
        ref = _run_block(40, None, seed=99)
        np.testing.assert_array_equal(obs[1].coords, ref[1].coords)
        np.testing.assert_array_equal(obs[4], ref[4])
        np.testing.assert_array_equal(obs[5], ref[5])
        self.assertEqual(obs[3].scored_rows, ref[3].scored_rows)
        # bookkeeping still ran
        rows = [int(i) for i in obs[2]["ids"]]
        self.assertTrue(any(st.row_retired(r) for r in rows))
        # the chain really ran the full fixed budget ...
        self.assertTrue(all(st.clock(r) == 40 for r in rows))
        # ... while the report says where the rule WOULD have stopped it
        self.assertTrue(all(st.repeats(r) <= 40 for r in rows))
        self.assertTrue(any(st.repeats(r) < 40 for r in rows))

    def test_ceiling_bounds_a_block_that_never_converges(self):
        # an unreachable threshold: no row can ever satisfy the rule
        st = self._state(window=5, thresh=-1e9, max_repeats=17)
        move, sorter, picked, buf, prop, acc = _run_block(300, st)
        rows = [int(i) for i in picked["ids"]]
        self.assertTrue(all(r in st.capped for r in rows))
        self.assertFalse(any(r in st.converged for r in rows))
        # the ceiling stopped it, at the poll granularity
        self.assertLessEqual(len(buf.scored_rows), 17 + _CONVERGE_POLL_EVERY)

    def test_stop_frac_ends_the_block_on_the_tail(self):
        st_all = self._state(window=5, stop_frac=1.0)
        st_half = self._state(window=5, stop_frac=0.5)
        n_all = len(_run_block(300, st_all)[3].scored_rows)
        n_half = len(_run_block(300, st_half)[3].scored_rows)
        self.assertLessEqual(n_half, n_all)

    def test_runs_with_vertical_swaps_armed(self):
        """The rule must survive the sweep relabelling ``t_i`` mid-block.

        Uses ``test_vertical_swap``'s sorter fake, which carries the REAL
        special-index semantics the sweep relabels through -- the point
        being that the per-row statistic is untouched by a relabel.
        """
        from tests.test_vertical_swap import _FakeSorter, _ladder
        from lisatools.globalfit.moves.gbbands import pack_special_index

        nt, nw, nb = 4, 2, 3
        t = np.arange(nt)
        w = np.zeros(nt, dtype=int)
        b = np.ones(nt, dtype=int)
        n_src = 8
        rng = np.random.RandomState(3)
        coords = np.zeros((n_src, 4))
        coords[:, 0] = rng.uniform(-0.5, 0.5, n_src)
        coords[:, 1] = rng.uniform(2.95, 3.05, n_src)
        coords[:, 2] = rng.uniform(-1, 1, n_src)
        coords[:, 3] = rng.uniform(-1, 1, n_src)
        picked = {
            "ids": np.arange(nt),
            "specials": pack_special_index(t, w, b, nw),
            "slot_index": np.arange(nt, dtype=np.int32),
            "temp_inds": t.copy(), "walker_inds": w.copy(),
            "band_inds": b.copy(), "N_vals": np.full(nt, 64),
        }
        sorter = _FakeSorter(t, w, b, nw)
        sorter.inds = np.ones(n_src, dtype=bool)
        sorter.coords = coords.copy()
        sorter.leaf_inds = np.arange(n_src)

        mv = _make_move(120)
        mv.ntemps, mv.nwalkers, mv.num_bands = nt, nw, nb
        mv.temper_vertical = True
        mv._temper_rng = np.random.default_rng(5)
        mv.sequential_parity_repeats = False
        ll_change = np.zeros((nt, nw, nb))
        prop = np.zeros((2, nt, nw, nb), dtype=int)
        acc = np.zeros_like(prop)
        cell_ll = {
            "spec": sorter.special_band_inds.copy(),
            "ll0": np.zeros(nt), "led0": np.zeros(nt),
            "rep0": np.zeros(nt, dtype=int),
        }
        st = self._state(window=5)
        np.random.seed(11)
        mv._run_in_model_repeats(
            None, sorter, _FakeBuffer(nt), _ladder(), picked,
            ll_change, prop, acc, num_repeats=120, cell_ll_state=cell_ll,
            converge=st,
        )
        rows = [int(i) for i in picked["ids"]]
        self.assertTrue(all(st.row_retired(r) for r in rows))

    def test_report_is_safe_on_an_empty_row_set(self):
        self._state().report("gb", "newborn", {}, [], 100)


class ConvergeStageGateTest(unittest.TestCase):
    def test_pe_named_moves_are_refused(self):
        for name in ("rj_fstat_pe", "rj_prior_pe", "rj_replace_pe"):
            self.assertFalse(
                _converge_stage_allows(SimpleNamespace(name=name)), name)

    def test_search_cycle_moves_are_allowed(self):
        for name in ("rj_fstat_search", "rj_prior_removal", "rj_replace",
                     "in_model"):
            self.assertTrue(
                _converge_stage_allows(SimpleNamespace(name=name)), name)

    def test_state_factory_refuses_on_a_pe_move(self):
        move = _make_move(100)
        move.name = "rj_prior_pe"
        move.inmodel_converge = "on"
        move.inmodel_converge_classes = frozenset({"newborn"})
        self.assertIsNone(move._converge_state_for("newborn"))
        self.assertTrue(move._converge_pe_warned)

    def test_state_factory_off_and_class_filter(self):
        move = _make_move(100)
        move.name = "rj_fstat_search"
        move.inmodel_converge = "off"
        move.inmodel_converge_classes = frozenset({"newborn"})
        self.assertIsNone(move._converge_state_for("newborn"))
        move.inmodel_converge = "on"
        move.inmodel_converge_iters = 50
        move.inmodel_converge_dll = 4.0
        move.inmodel_converge_max = 0
        move.inmodel_converge_stop_frac = 1.0
        move.inmodel_converge_gate_frac = 0.5
        move.inmodel_converge_refill = True
        move.ntemps = 24
        move.inmodel_repeats_newborn = 100
        move.inmodel_repeats_survivor = 50
        self.assertIsNone(move._converge_state_for("mature"))
        st = move._converge_state_for("newborn")
        self.assertIsNotNone(st)
        self.assertEqual(st.max_repeats, 400)      # 0 -> 4x the budget
        self.assertEqual(st.window, 50)
        self.assertEqual(st.n_gate, 12)            # coldest half of 24
        self.assertTrue(st.refill)

    def test_the_shipped_window_default_is_100(self):
        """User ruling 2026-09-24. Pinned because the FLOOR it implies
        (W + 1 = 101) sits above the stock newborn budget of 100, i.e. the
        mode becomes a deliberate spend rather than a saving -- a silent
        change back to 50 would quietly reverse that trade."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_CONVERGE_ITERS", None)
            self.assertEqual(
                _resolve_converge_knob(
                    "gb", "iters", None, 100, _converge_cast_window),
                100,
            )

    def test_gate_frac_rounds_up_so_it_can_never_disarm(self):
        move = _make_move(100)
        move.name = "rj_fstat_search"
        move.inmodel_converge = "on"
        move.inmodel_converge_classes = frozenset({"newborn"})
        move.inmodel_converge_iters = 50
        move.inmodel_converge_dll = 4.0
        move.inmodel_converge_max = 0
        move.inmodel_converge_stop_frac = 1.0
        move.inmodel_converge_refill = False
        move.inmodel_repeats_newborn = 100
        move.inmodel_repeats_survivor = 50
        move.ntemps = 4
        for frac, want in ((0.5, 2), (0.01, 1), (1.0, 4)):
            move.inmodel_converge_gate_frac = frac
            self.assertEqual(
                move._converge_state_for("newborn").n_gate, want, frac)


# --------------------------------------------------------------------------
# LADDER GATE: the hot half must not hold the block open
# --------------------------------------------------------------------------
from lisatools.globalfit.moves.gbspecialstretch import (  # noqa: E402
    _converge_column_spans,
    _converge_gate_mask,
    _converge_refill,
    _converge_take,
)


class ConvergeGateMaskTest(unittest.TestCase):
    def test_gate_picks_the_cold_rungs(self):
        t = np.array([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(
            _converge_gate_mask(t, 3, np),
            np.array([True, True, True, False, False, False]),
        )

    def test_none_gates_everything(self):
        t = np.array([0, 3, 7])
        self.assertTrue(np.all(_converge_gate_mask(t, None, np)))

    def test_a_hot_only_block_falls_back_to_every_row(self):
        """A pool whose columns kept no cold rung must NOT exit instantly."""
        t = np.array([5, 6, 7])
        self.assertTrue(np.all(_converge_gate_mask(t, 3, np)))

    def test_the_mask_follows_the_rung_not_the_row(self):
        """A vertical swap relabels t_i; the gate must move with it."""
        t = np.array([0, 1, 2, 3])
        before = _converge_gate_mask(t, 2, np)
        t[[0, 3]] = t[[3, 0]]                   # rows 0 and 3 trade rungs
        after = _converge_gate_mask(t, 2, np)
        np.testing.assert_array_equal(before, [True, True, False, False])
        np.testing.assert_array_equal(after, [False, True, False, True])


class ConvergeGateBlockTest(unittest.TestCase):
    """Through the real repeat loop: a never-converging HOT rung must not
    keep the block running, but a never-converging COLD one must."""

    def _run(self, gate_n, hot_blocks):
        """``hot_blocks``: put the unconvergeable row on a hot rung (True)
        or on the cold rung (False)."""
        picked, sorter, band_temps, n = _column_problem()
        move = _make_move(400)
        move.ntemps, move.nwalkers, move.num_bands = NTEMPS, NWALKERS, NBANDS
        move.sequential_parity_repeats = False
        move.temper_vertical = False
        buf = _CountingBuffer(n)
        ll_change = np.zeros((NTEMPS, NWALKERS, NBANDS))
        prop = np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int)
        acc = np.zeros_like(prop)
        st = _InModelConvergeState(
            window=5, thresh=4.0, max_repeats=100000, n_gate=gate_n)
        # Make ONE row unconvergeable by giving it a permanently climbing
        # gain: patch step so that row's gain always beats the window.
        target_t = NTEMPS - 1 if hot_blocks else 0
        target = int(np.where(picked["temp_inds"] == target_t)[0][0])
        real_step = _InModelConvergeState.step
        bump = {"n": 0}

        def fake_step(xp, gain, best, ring, seen, rows_ar, window, thresh,
                      done, at=None):
            gain = np.array(gain, dtype=float, copy=True)
            bump["n"] += 1
            gain[target] = 1e3 * bump["n"]
            return real_step(xp, gain, best, ring, seen, rows_ar, window,
                             thresh, done, at)

        with mock.patch.object(
                _InModelConvergeState, "step", staticmethod(fake_step)):
            np.random.seed(4)
            move._run_in_model_repeats(
                None, sorter, buf, band_temps, picked, ll_change, prop, acc,
                num_repeats=400, converge=st,
            )
        # The block's length in REPEATS. Not ``len(buf.scored_rows)``: the
        # scoring kernel is skipped entirely on a repeat where every live
        # row failed the f0-window / prior gate, so that list undercounts
        # exactly when the block has shrunk to a handful of rows -- which
        # is the regime these tests are about. The clock advances once per
        # repeat for every row present.
        return max(st.clock(int(i)) for i in picked["ids"])

    def test_a_climbing_hot_rung_does_not_hold_the_block(self):
        # gate = coldest 2 of 4 rungs; the runaway sits at t=3
        n_hot = self._run(gate_n=2, hot_blocks=True)
        # gate = the whole ladder: the same runaway now holds it open
        n_all = self._run(gate_n=NTEMPS, hot_blocks=True)
        self.assertLess(n_hot, n_all)
        self.assertEqual(n_all, 400)

    def test_a_climbing_cold_rung_still_holds_the_block(self):
        self.assertEqual(self._run(gate_n=2, hot_blocks=False), 400)

    def test_ungated_rows_are_released_not_converged(self):
        picked, sorter, band_temps, n = _column_problem()
        move = _make_move(200)
        move.ntemps, move.nwalkers, move.num_bands = NTEMPS, NWALKERS, NBANDS
        move.sequential_parity_repeats = False
        move.temper_vertical = False
        st = _InModelConvergeState(
            window=5, thresh=4.0, max_repeats=100000, n_gate=2)
        np.random.seed(6)
        move._run_in_model_repeats(
            None, sorter, _CountingBuffer(n), band_temps, picked,
            np.zeros((NTEMPS, NWALKERS, NBANDS)),
            np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int),
            np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int),
            num_repeats=200, converge=st,
        )
        ids = picked["ids"]
        hot = [int(i) for i, t in zip(ids, picked["temp_inds"]) if t >= 2]
        cold = [int(i) for i, t in zip(ids, picked["temp_inds"]) if t < 2]
        self.assertTrue(all(r in st.released for r in hot))
        self.assertTrue(all(not st.row_retired(r) for r in hot))
        self.assertTrue(all(st.row_retired(r) for r in cold))
        # a column is retired on its GATED rungs alone
        for w in range(NWALKERS):
            for b in range(NBANDS):
                rows = [int(i) for i, ww, bb in zip(
                    ids, picked["walker_inds"], picked["band_inds"])
                    if ww == w and bb == b]
                self.assertTrue(st.column_retired(rows))


# --------------------------------------------------------------------------
# COLUMN REFILL: a finished band leaves, a new one takes its slots
# --------------------------------------------------------------------------
class ConvergeRefillHelperTest(unittest.TestCase):
    def test_spans_cover_the_pool_exactly(self):
        p = {
            "ids": np.arange(6),
            "walker_inds": np.array([0, 0, 0, 0, 0, 1]),
            "band_inds": np.array([0, 0, 0, 1, 1, 0]),
            "temp_inds": np.array([0, 1, 2, 0, 1, 0]),
        }
        spans = _converge_column_spans(p, 2)
        self.assertEqual([(s, e) for _, s, e in spans],
                         [(0, 3), (3, 5), (5, 6)])

    def test_non_contiguous_column_raises(self):
        p = {
            "ids": np.arange(3),
            "walker_inds": np.array([0, 1, 0]),
            "band_inds": np.array([0, 0, 0]),
            "temp_inds": np.array([0, 0, 1]),
        }
        with self.assertRaises(RuntimeError):
            _converge_column_spans(p, 2)

    def test_refill_never_splits_a_column(self):
        queue = [(0, 0, 4), (1, 4, 8), (2, 8, 12)]
        active, queue = _converge_refill([], queue, width=10)
        self.assertEqual([c for c, _, _ in active], [0, 1])
        self.assertEqual([c for c, _, _ in queue], [2])

    def test_oversized_column_never_evicts_carried_work(self):
        active, queue = _converge_refill([(9, 0, 6)], [(0, 6, 26)], width=8)
        self.assertEqual([c for c, _, _ in active], [9])
        self.assertEqual([c for c, _, _ in queue], [0])

    def test_oversized_column_taken_whole_into_an_empty_set(self):
        active, queue = _converge_refill([], [(0, 0, 20)], width=8)
        self.assertEqual([c for c, _, _ in active], [0])
        self.assertEqual(queue, [])

    def test_take_gathers_span_rows_in_order(self):
        pool = {"ids": np.arange(10), "x": np.arange(10) * 2}
        sub = _converge_take(pool, [(0, 6, 9), (1, 1, 3)], np)
        np.testing.assert_array_equal(sub["ids"], [6, 7, 8, 1, 2])
        np.testing.assert_array_equal(sub["x"], [12, 14, 16, 2, 4])

    def test_take_of_nothing_is_none(self):
        self.assertIsNone(_converge_take({"ids": np.arange(3)}, [], np))

    def test_refill_drains_and_terminates(self):
        queue = [(i, 3 * i, 3 * i + 3) for i in range(7)]
        seen, active, guard = [], [], 0
        while queue or active:
            guard += 1
            self.assertLess(guard, 50, "refill loop did not terminate")
            active, queue = _converge_refill(active, queue, width=6)
            seen.extend(c for c, _, _ in active)
            active = []               # pretend every column retired
        self.assertEqual(sorted(seen), list(range(7)))


class ConvergeRefillLoopTest(unittest.TestCase):
    """``_converge_refill_loop``: whole columns retire and are replaced.

    Driven with a fake ``polish`` so the loop is exercised without a
    buffer: the fake marks whichever columns the scenario says finished.
    """

    NB = 4

    def _pool(self, n_cols, rungs=3):
        w, b, t, ids = [], [], [], []
        for c in range(n_cols):
            for r in range(rungs):
                w.append(c // self.NB)
                b.append(c % self.NB)
                t.append(r)
                ids.append(len(ids))
        return {
            "ids": np.array(ids),
            "specials": np.array(ids) * 10,
            "walker_inds": np.array(w),
            "band_inds": np.array(b),
            "temp_inds": np.array(t),
        }

    def _move(self):
        mv = _make_move(100)
        mv.name = "rj_fstat_search"
        mv.num_bands = self.NB
        return mv

    def _sorter(self, pool):
        return SimpleNamespace(
            special_band_inds=pool["specials"].copy(),
            temp_inds=pool["temp_inds"].copy(),
        )

    def test_finished_columns_are_replaced_not_waited_on(self):
        """A column that finishes in generation 1 must free its slots for a
        queued column while a slow neighbour keeps working."""
        pool = self._pool(4)                     # 4 columns x 3 rungs
        mv, st = self._move(), _InModelConvergeState(
            window=2, thresh=4.0, max_repeats=99, refill=True)
        slow = set(range(0, 3))                  # column 0's rows: never done
        seen = []

        def polish(chunk):
            rows = [int(i) for i in chunk["ids"]]
            seen.append(sorted(rows))
            for r in rows:
                st._gated[r] = True
                if r not in slow:
                    st.converged.add(r)

        n = mv._converge_refill_loop(
            pool, st, polish, 6, self._sorter(pool), np)
        self.assertEqual(n, len(seen))
        # generation 1 holds columns 0+1; column 1 finishes, 0 carries
        self.assertEqual(seen[0], list(range(6)))
        # the slow column is still present in generation 2, alongside a NEW
        # one -- it was not left to finish alone, and the queue advanced
        self.assertTrue(set(range(0, 3)).issubset(set(seen[1])))
        self.assertTrue(set(range(6, 9)).issubset(set(seen[1])))
        # every column was polished at least once
        polished = set().union(*[set(s) for s in seen])
        self.assertEqual(polished, set(range(12)))

    def test_all_columns_finishing_gives_one_generation_per_full_width(self):
        pool = self._pool(4)
        mv, st = self._move(), _InModelConvergeState(
            window=2, thresh=4.0, max_repeats=99, refill=True)

        def polish(chunk):
            for r in [int(i) for i in chunk["ids"]]:
                st._gated[r] = True
                st.converged.add(r)

        n = mv._converge_refill_loop(
            pool, st, polish, 6, self._sorter(pool), np)
        self.assertEqual(n, 2)          # 12 rows / width 6, nothing carried
        self.assertEqual(st.generations, 2)

    def test_a_stalled_generation_does_not_spin(self):
        """Nothing retires and the queue is empty -> stop, do not re-pay the
        setup forever."""
        pool = self._pool(2)
        mv, st = self._move(), _InModelConvergeState(
            window=2, thresh=4.0, max_repeats=99, refill=True)
        calls = []

        def polish(chunk):
            calls.append(len(chunk["ids"]))
            for r in [int(i) for i in chunk["ids"]]:
                st._gated[r] = True     # gated, but never converged

        n = mv._converge_refill_loop(
            pool, st, polish, 99, self._sorter(pool), np)
        self.assertEqual(n, 1)
        self.assertEqual(calls, [6])

    def test_a_generation_that_retires_nothing_caps_and_moves_on(self):
        """The queue must never be starved by a column that will not
        finish: if a generation retires none, they are capped so the
        queued columns get the slots."""
        pool = self._pool(3)
        mv, st = self._move(), _InModelConvergeState(
            window=2, thresh=4.0, max_repeats=99, refill=True)
        calls = []

        def polish(chunk):
            rows = [int(i) for i in chunk["ids"]]
            calls.append(sorted(rows))
            for r in rows:          # gated, but NEVER converged
                st._gated[r] = True

        n = mv._converge_refill_loop(
            pool, st, polish, 3, self._sorter(pool), np)
        # one generation per column, no spinning, and every column ran
        self.assertEqual(n, 3)
        self.assertEqual(calls, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(st.capped, set(range(9)))
        self.assertEqual(st.converged, set())

    def test_the_guard_is_never_reached_in_practice(self):
        """Progress is mandatory, so the warning path stays dead."""
        pool = self._pool(5)
        mv, st = self._move(), _InModelConvergeState(
            window=2, thresh=4.0, max_repeats=99, refill=True)

        def polish(chunk):
            for r in [int(i) for i in chunk["ids"]]:
                st._gated[r] = True

        with self.assertNoLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "WARNING"):
            n = mv._converge_refill_loop(
                pool, st, polish, 3, self._sorter(pool), np)
        self.assertEqual(n, 5)

    def test_labels_are_re_read_from_the_sorter_each_generation(self):
        """A vertical swap relabels cells; a stale ``specials`` would bind a
        row to another cell's slab."""
        pool = self._pool(2)
        sorter = self._sorter(pool)
        sorter.special_band_inds = pool["specials"] + 777   # "a swap happened"
        mv, st = self._move(), _InModelConvergeState(
            window=2, thresh=4.0, max_repeats=99, refill=True)
        got = []

        def polish(chunk):
            got.append(np.asarray(chunk["specials"]).copy())
            for r in [int(i) for i in chunk["ids"]]:
                st._gated[r] = True
                st.converged.add(r)

        mv._converge_refill_loop(pool, st, polish, 99, sorter, np)
        np.testing.assert_array_equal(got[0], pool["ids"] * 10 + 777)

    def test_refill_off_is_not_this_loop(self):
        st = _InModelConvergeState(
            window=2, thresh=4.0, max_repeats=99, refill=False)
        self.assertFalse(st.refill)


class ConvergeStopFracTest(unittest.TestCase):
    """``stop_frac`` counts finished COLUMNS, and 1.0 disables the refill."""

    def test_stop_frac_one_retires_every_column_together(self):
        """Pins the degeneracy the default avoids: at 1.0 the block only
        ends when all columns are done, so nothing ever carries and the
        refill loop has nothing to swap."""
        picked, sorter, band_temps, n = _column_problem()
        move = _make_move(400)
        move.ntemps, move.nwalkers, move.num_bands = NTEMPS, NWALKERS, NBANDS
        move.sequential_parity_repeats = False
        move.temper_vertical = False
        st = _InModelConvergeState(
            window=5, thresh=4.0, max_repeats=100000, n_gate=2,
            stop_frac=1.0, refill=True)
        np.random.seed(8)
        move._run_in_model_repeats(
            None, sorter, _CountingBuffer(n), band_temps, picked,
            np.zeros((NTEMPS, NWALKERS, NBANDS)),
            np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int),
            np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int),
            num_repeats=400, converge=st,
        )
        ids = picked["ids"]
        cols = {}
        for i, w, b in zip(ids, picked["walker_inds"], picked["band_inds"]):
            cols.setdefault((int(w), int(b)), []).append(int(i))
        # EVERY column finished -> a refill loop would carry nothing
        self.assertTrue(all(st.column_retired(r) for r in cols.values()))

    def test_stop_frac_half_leaves_columns_to_carry(self):
        """At 0.5 the block stops with some columns done and some not --
        which is exactly what gives the refill something to swap."""
        picked, sorter, band_temps, n = _column_problem()
        move = _make_move(400)
        move.ntemps, move.nwalkers, move.num_bands = NTEMPS, NWALKERS, NBANDS
        move.sequential_parity_repeats = False
        move.temper_vertical = False
        st = _InModelConvergeState(
            window=5, thresh=4.0, max_repeats=100000, n_gate=2,
            stop_frac=0.5, refill=True)
        # one column's cold rung never settles
        real_step = _InModelConvergeState.step
        target = int(np.where(
            (picked["temp_inds"] == 0) & (picked["walker_inds"] == 0)
            & (picked["band_inds"] == 0))[0][0])
        bump = {"n": 0}

        def fake_step(xp, gain, best, ring, seen, rows_ar, window, thresh,
                      done, at=None):
            gain = np.array(gain, dtype=float, copy=True)
            bump["n"] += 1
            gain[target] = 1e3 * bump["n"]
            return real_step(xp, gain, best, ring, seen, rows_ar, window,
                             thresh, done, at)

        with mock.patch.object(
                _InModelConvergeState, "step", staticmethod(fake_step)):
            np.random.seed(8)
            move._run_in_model_repeats(
                None, sorter, _CountingBuffer(n), band_temps, picked,
                np.zeros((NTEMPS, NWALKERS, NBANDS)),
                np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int),
                np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int),
                num_repeats=400, converge=st,
            )
        cols = {}
        for i, w, b in zip(picked["ids"], picked["walker_inds"],
                           picked["band_inds"]):
            cols.setdefault((int(w), int(b)), []).append(int(i))
        done = [k for k, r in cols.items() if st.column_retired(r)]
        left = [k for k, r in cols.items() if not st.column_retired(r)]
        self.assertTrue(done, "no column finished -- nothing to swap out")
        self.assertEqual(left, [(0, 0)], "the stalled band must carry")
        # and it stopped well before the ceiling rather than waiting it out
        self.assertLess(max(st.clock(int(i)) for i in picked["ids"]), 400)


# --------------------------------------------------------------------------
# SCOPE 2: per-(walker, band) convergence within ONE in-model proposal group
# --------------------------------------------------------------------------
from lisatools.globalfit.moves.gbspecialstretch import (  # noqa: E402
    _InModelGroupState,
    _converge_cast_scale,
)


class GroupScaleCastTest(unittest.TestCase):
    def test_accepts_the_two_modes(self):
        self.assertEqual(_converge_cast_scale("flat"), "flat")
        self.assertEqual(_converge_cast_scale(" PER_SOURCE "), "per_source")

    def test_rejects_anything_else(self):
        with self.assertRaises(ValueError):
            _converge_cast_scale("per_band")


class GroupStateTest(unittest.TestCase):
    """The pass-clocked, per-(walker, band) rule."""

    NW, NB = 2, 3

    def _st(self, window=2, thresh=4.0, max_passes=50, scale="flat"):
        return _InModelGroupState(window=window, thresh=thresh,
                                  max_passes=max_passes, scale=scale)

    def _occ(self, n=1):
        return np.full((self.NW, self.NB), n)

    def test_a_flat_sub_band_shuts_one_pass_after_the_window(self):
        st = self._st(window=2)
        zero = np.zeros((self.NW, self.NB))
        for _ in range(2):
            st.update(np, zero, self._occ())
            self.assertFalse(st.all_shut(np))
        st.update(np, zero, self._occ())
        self.assertTrue(st.all_shut(np))
        self.assertEqual(st.passes, 3)

    def test_a_climbing_sub_band_stays_open(self):
        st = self._st(window=2, thresh=4.0)
        gain = np.zeros((self.NW, self.NB))
        gain[0, 0] = 100.0                       # this one keeps paying
        for _ in range(10):
            st.update(np, gain, self._occ())
        self.assertFalse(bool(st.shut[0, 0]))
        self.assertTrue(bool(st.shut[1, 1]))     # the flat ones shut
        self.assertFalse(st.all_shut(np))

    def test_an_unoccupied_pair_shuts_immediately(self):
        """An empty sub-band has nothing to converge. Unlike the
        STAGE-scoped valve (which leaves empty pairs open forever on
        purpose), this group must terminate inside one propose."""
        st = self._st(window=5)
        occ = self._occ()
        occ[0, 0] = 0
        st.update(np, np.zeros((self.NW, self.NB)), occ)
        self.assertTrue(bool(st.shut[0, 0]))
        self.assertFalse(bool(st.shut[1, 1]))

    def test_shut_latches_and_records_the_pass(self):
        st = self._st(window=1)
        zero = np.zeros((self.NW, self.NB))
        st.update(np, zero, self._occ())
        st.update(np, zero, self._occ())
        self.assertTrue(st.all_shut(np))
        at = st.shut_at.copy()
        st.update(np, np.full((self.NW, self.NB), 1e6), self._occ())
        self.assertTrue(st.all_shut(np))         # a late gain cannot reopen
        np.testing.assert_array_equal(st.shut_at, at)

    def test_flat_threshold_gives_dense_bands_more_passes(self):
        """The user's stated intent: flat D/2 focuses resources on the
        sub-bands with more sources. A band gaining 0.1/source with 50
        sources clears 4.0; the same rate with 2 sources does not."""
        st = self._st(window=2, thresh=4.0, scale="flat")
        gain = np.zeros((self.NW, self.NB))
        gain[0, 0] = 50 * 0.1                    # dense band: 5.0 > 4.0
        gain[0, 1] = 2 * 0.1                     # sparse band: 0.2 < 4.0
        occ = self._occ()
        occ[0, 0], occ[0, 1] = 50, 2
        for _ in range(6):
            st.update(np, gain, occ)
        self.assertFalse(bool(st.shut[0, 0]), "dense band should keep going")
        self.assertTrue(bool(st.shut[0, 1]), "sparse band should have shut")

    def test_per_source_scaling_removes_that_asymmetry(self):
        st = self._st(window=2, thresh=4.0, scale="per_source")
        gain = np.zeros((self.NW, self.NB))
        gain[0, 0] = 50 * 0.1
        gain[0, 1] = 2 * 0.1
        occ = self._occ()
        occ[0, 0], occ[0, 1] = 50, 2
        for _ in range(6):
            st.update(np, gain, occ)
        # both now measured as 0.1/source/pass, both below the threshold
        self.assertTrue(bool(st.shut[0, 0]))
        self.assertTrue(bool(st.shut[0, 1]))

    def test_census_counts_open_occupied_pairs(self):
        st = self._st(window=5)
        occ = self._occ()
        occ[0, 0] = 0
        st.update(np, np.zeros((self.NW, self.NB)), occ)
        n_shut, n_open, n_occ_open = st.census(np)
        self.assertEqual(n_shut, 1)                  # the empty one
        self.assertEqual(n_open, self.NW * self.NB - 1)
        self.assertEqual(n_occ_open, self.NW * self.NB - 1)

    def test_update_reports_newly_shut_only(self):
        st = self._st(window=1)
        zero = np.zeros((self.NW, self.NB))
        self.assertEqual(st.update(np, zero, self._occ()), 0)
        self.assertEqual(st.update(np, zero, self._occ()),
                         self.NW * self.NB)
        self.assertEqual(st.update(np, zero, self._occ()), 0)


class GroupWiringTest(unittest.TestCase):
    """``_group_state_or_none`` gating."""

    def _move(self, **kw):
        mv = _make_move(25)
        mv.name = kw.pop("name", "in_model")
        mv.is_rj_prop = kw.pop("is_rj_prop", False)
        mv.inmodel_group = kw.pop("inmodel_group", True)
        mv.inmodel_group_iters = 3
        mv.inmodel_group_dll = 4.0
        mv.inmodel_group_max_passes = 20
        mv.inmodel_group_scale = "flat"
        for k, v in kw.items():
            setattr(mv, k, v)
        return mv

    def test_off_is_none(self):
        self.assertIsNone(
            self._move(inmodel_group=False)._group_state_or_none())

    def test_pure_in_model_move_gets_a_state(self):
        st = self._move()._group_state_or_none()
        self.assertIsNotNone(st)
        self.assertEqual(st.window, 3)
        self.assertEqual(st.max_passes, 20)
        self.assertEqual(st.scale, "flat")

    def test_an_rj_move_is_refused_with_a_warning(self):
        """On an RJ pass a flat sub-band logL means the BIRTHS stopped
        paying -- that is the stage valve's call, not this one's."""
        mv = self._move(name="rj_fstat_search", is_rj_prop=True)
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "WARNING"):
            self.assertIsNone(mv._group_state_or_none())
        self.assertTrue(mv._group_rj_warned)

    def test_each_call_is_a_fresh_state(self):
        """The group is scoped to ONE proposal; nothing may carry."""
        mv = self._move()
        a, b = mv._group_state_or_none(), mv._group_state_or_none()
        self.assertIsNot(a, b)
        self.assertEqual(a.passes, 0)
        self.assertEqual(b.passes, 0)


class GroupLoopTerminationTest(unittest.TestCase):
    """``_run_group_passes`` must ALWAYS terminate.

    This is a stopping-gate audit, not a feature test: the group loop is
    unbounded by construction (it repeats a full sweep until the data says
    stop), so every exit has to be reachable and the ceiling has to be
    enforced even when the data never cooperates.
    """

    NW, NB = 2, 3

    def _move(self, cold_delta, max_passes=6, window=2, occ=1):
        """A move whose run_proposal always reports ``cold_delta``."""
        mv = _make_move(25)
        mv.name = "in_model"
        mv.is_rj_prop = False
        mv.nwalkers, mv.num_bands, mv.ntemps = self.NW, self.NB, 2
        mv.branch_name = "gb"
        calls = []

        def fake_run_proposal(model, state, sorter, temps):
            calls.append(1)
            log = np.zeros((2, self.NW, self.NB))
            log[0] = cold_delta
            return log, None, None

        mv.run_proposal = fake_run_proposal
        mv._group_cold_occupancy = lambda sorter: np.full(
            (self.NW, self.NB), occ)
        st = _InModelGroupState(window=window, thresh=4.0,
                                max_passes=max_passes)
        sorter = SimpleNamespace(has_run_rj=np.zeros(4, dtype=bool))
        state = SimpleNamespace(log_like=np.zeros((1, self.NW)))
        first = np.zeros((2, self.NW, self.NB))
        first[0] = cold_delta
        return mv, st, sorter, state, first, calls

    def test_it_stops_when_every_sub_band_converges(self):
        mv, st, sorter, state, first, calls = self._move(
            np.zeros((self.NW, self.NB)), max_passes=50, window=2)
        mv._run_group_passes(st, None, state, sorter, None, first, None, None)
        self.assertTrue(st.all_shut(np))
        self.assertLess(st.passes, 50, "should stop on convergence, not the cap")

    def test_the_ceiling_stops_a_group_that_never_converges(self):
        """Every sub-band climbing forever: only the ceiling can end this."""
        mv, st, sorter, state, first, calls = self._move(
            np.full((self.NW, self.NB), 1e3), max_passes=6, window=2)
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "WARNING") as cm:
            mv._run_group_passes(st, None, state, sorter, None, first, None, None)
        self.assertEqual(st.passes, 6)
        self.assertFalse(st.all_shut(np))
        self.assertTrue(any("CEILING" in m for m in cm.output),
                        "hitting the cap must WARN, not pass silently")

    def test_an_empty_pool_terminates_immediately(self):
        """No occupied pair anywhere: every pair shuts on pass 1 rather than
        holding the group open forever."""
        mv, st, sorter, state, first, calls = self._move(
            np.zeros((self.NW, self.NB)), max_passes=50, window=5, occ=0)
        mv._run_group_passes(st, None, state, sorter, None, first, None, None)
        self.assertTrue(st.all_shut(np))
        self.assertEqual(st.passes, 1)

    def test_has_run_rj_is_reset_every_pass(self):
        """Without this the second pass picks NOTHING and the group looks
        instantly converged -- has_run_rj is allocated once per SORTER."""
        mv, st, sorter, state, first, calls = self._move(
            np.full((self.NW, self.NB), 1e3), max_passes=3, window=2)
        sorter.has_run_rj[:] = True
        mv._run_group_passes(st, None, state, sorter, None, first, None, None)
        self.assertFalse(bool(sorter.has_run_rj.any()),
                         "has_run_rj must be cleared for the next pass")
        self.assertEqual(len(calls), 2, "ceiling 3 => 2 further passes run")

    def test_each_pass_banks_its_cold_delta_into_log_like(self):
        """Every pass but the last banks its own delta; the last is banked by
        the caller. Dropping one silently loses lnL from the chain."""
        mv, st, sorter, state, first, calls = self._move(
            np.full((self.NW, self.NB), 1e3), max_passes=4, window=2)
        mv._run_group_passes(st, None, state, sorter, None, first, None, None)
        # passes 1..3 bank; pass 4's delta is returned for the caller
        self.assertAlmostEqual(
            float(state.log_like[0].sum()),
            3 * self.NW * self.NB * 1e3, places=3)

    def test_an_IMMEDIATE_convergence_returns_the_CALLERS_counts(self):
        """⚠ The no-extra-pass case used to return ``(ll, None, None)``.

        Both callers unpack straight into their own ``prop_counts`` /
        ``acc_counts`` and both then index ``prop_counts[0]`` (the
        per-propose census, and on the fan-out path the reply the head
        sums across blocks). So a group that shut every sub-band on the
        caller's FIRST pass -- the cheapest, most ordinary outcome -- was a
        TypeError waiting on a path nobody had run, because the group rule
        itself had never executed in production.
        """
        mv, st, sorter, state, first, calls = self._move(
            np.zeros((self.NW, self.NB)), max_passes=50, window=2)
        # Force the "everything shut on the caller's own pass" branch. The
        # state machine needs two updates to reach it on its own, so
        # stubbing is what isolates the RETURN contract from the shut-off
        # arithmetic, which its own tests already cover.
        st.all_shut = lambda xp: True
        pc = np.ones((2, 2, self.NB), dtype=int)
        ac = np.full((2, 2, self.NB), 7, dtype=int)
        ll, out_pc, out_ac = mv._run_group_passes(
            st, None, state, sorter, None, first, pc, ac)
        self.assertEqual(len(calls), 0, "no further pass should have run")
        self.assertIsNotNone(out_pc, "returned None: the old TypeError trap")
        self.assertIsNotNone(out_ac, "returned None: the old TypeError trap")
        np.testing.assert_array_equal(out_pc, pc)
        np.testing.assert_array_equal(out_ac, ac)
        np.testing.assert_array_equal(ll, first)
        # the shape both callers then index -- prop_counts[0]
        self.assertEqual(out_pc[0].shape, pc[0].shape)


class GroupRuleReachesTheFanoutPathTest(unittest.TestCase):
    """REGRESSION, job 620: ``GB_INMODEL_GROUP`` was inert in production.

    ``_group_state_or_none`` had exactly ONE caller, inside
    ``_propose_legacy``. A 4-rank run dispatches to ``_propose_orchestrated``
    and the GB work is served on the ranks by ``_gb_serve_run_proposal``,
    so the group rule never executed: zero ``[GB_IMGROUP]`` lines in 63
    minutes with ``in_model`` having run for 97 s.

    Everything else about the knob was healthy -- it resolved to True, the
    preflight passed, the launcher printed ``[V9-IMGROUP] group=1`` -- which
    is exactly why a source-level check is worth having. A behavioural test
    of the served body needs MPI and GPUs; this pins the one fact that
    failed.
    """

    def _src(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        return g

    def test_BOTH_propose_paths_build_the_group_state(self):
        import inspect

        g = self._src()
        cls = g.GBSpecialBase
        for name in ("_propose_legacy", "_gb_serve_run_proposal"):
            body = inspect.getsource(getattr(cls, name))
            self.assertIn(
                "_group_state_or_none()", body,
                f"{name} does not build the in-model group state -- "
                f"GB_INMODEL_GROUP is inert on that path")
            self.assertIn(
                "_run_group_passes", body,
                f"{name} builds the group state but never runs its passes")

    def test_both_paths_clear_the_shutoff_in_a_finally(self):
        """``_group_shutoff_wb`` is the NEXT pass's eligibility filter. Left
        set, it would silently suppress sub-bands in a later propose -- and
        the group is scoped to ONE propose by design."""
        import inspect

        g = self._src()
        cls = g.GBSpecialBase
        for name in ("_propose_legacy", "_gb_serve_run_proposal"):
            body = inspect.getsource(getattr(cls, name))
            self.assertIn("self._group_shutoff_wb = None", body, name)
            fin = body.index("finally:")
            self.assertGreater(
                body.index("self._group_shutoff_wb = None", fin), fin,
                f"{name}: the reset must be in the finally, not the happy path")


class RunTemperingOffTest(unittest.TestCase):
    """``{BRANCH}_RUN_FANCY_TEMPERING=0`` kills the permuted swaps outright."""

    def _mk(self, env, run_swaps=True, branch="gb"):
        import lisatools.globalfit.moves.gbspecialstretch as g
        # the ctor line under test, exercised in isolation: the real ctor
        # needs a full fit to build.
        with mock.patch.dict(os.environ, env, clear=False):
            for k in ("GB_RUN_FANCY_TEMPERING", "VGB_RUN_FANCY_TEMPERING"):
                if k not in env:
                    os.environ.pop(k, None)
            _rt = os.environ.get(f"{branch.upper()}_RUN_FANCY_TEMPERING")
            if _rt is not None and _rt.strip().lower() in ("0", "false"):
                return False
            return run_swaps

    def test_unset_leaves_the_callers_value(self):
        self.assertTrue(self._mk({}, run_swaps=True))
        self.assertFalse(self._mk({}, run_swaps=False))

    def test_zero_forces_off(self):
        self.assertFalse(self._mk({"GB_RUN_FANCY_TEMPERING": "0"}, run_swaps=True))

    def test_one_leaves_it_alone(self):
        self.assertTrue(self._mk({"GB_RUN_FANCY_TEMPERING": "1"}, run_swaps=True))

    def test_it_is_branch_scoped(self):
        """A vgb knob must not silence gb's swaps."""
        self.assertTrue(
            self._mk({"VGB_RUN_FANCY_TEMPERING": "0"}, run_swaps=True, branch="gb"))

    def test_the_cadence_knob_cannot_do_this(self):
        """Regression guard for the trap this knob exists to avoid:
        GB_TEMPER_EVERY_PROPOSES=0 means 'fire ALWAYS', not 'never'."""
        mv = _make_move(25)
        mv.temper_every_proposes = 0
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase
        self.assertTrue(
            GBSpecialBase._temper_cadence_fire(mv),
            "n <= 1 must still mean 'always fire' -- if this ever becomes "
            "False, GB_TEMPER_EVERY_PROPOSES=0 has silently become an off "
            "switch and the docs here are wrong",
        )


class VerticalLadderAdaptTest(unittest.TestCase):
    """Adapting the band temperature ladder from the VERTICAL swaps.

    The hole this closes: ``_adapt_band_temps`` is called from exactly one
    place, ``run_tempering``. So GB_RUN_FANCY_TEMPERING=0 does not merely
    stop the permuted swaps -- it FREEZES THE LADDER for the whole run,
    silently. These pin that the vertical census can drive it instead.
    """

    NB, NT = 3, 4

    def _mv(self):
        mv = _make_move(25)
        mv.name = "in_model"
        mv.num_bands, mv.ntemps = self.NB, self.NT
        mv.branch_name = "gb"
        mv._vertical_ladder_reset()
        return mv

    def _census(self, prop, acc):
        return {"prop_by_bandrung_dev": prop, "acc_by_bandrung_dev": acc}

    def test_bank_accumulates_across_blocks(self):
        """Per PROPOSE, not per block -- one block's counts are far too
        sparse to steer a ladder with."""
        mv = self._mv()
        p = np.ones((self.NB, self.NT - 1), dtype=np.int64)
        a = np.full((self.NB, self.NT - 1), 2, dtype=np.int64)
        for _ in range(3):
            mv._vertical_ladder_bank(self._census(p, a))
        np.testing.assert_array_equal(mv._vert_ladder_prop, 3 * p)
        np.testing.assert_array_equal(mv._vert_ladder_acc, 3 * a)

    def test_bank_is_a_copy_not_an_alias(self):
        """The census array is reused by the next block; banking must not
        alias it or later blocks would mutate the banked totals."""
        mv = self._mv()
        p = np.ones((self.NB, self.NT - 1), dtype=np.int64)
        mv._vertical_ladder_bank(self._census(p, p))
        p += 99
        np.testing.assert_array_equal(
            mv._vert_ladder_prop, np.ones((self.NB, self.NT - 1)))

    def test_reset_clears_between_proposes(self):
        mv = self._mv()
        p = np.ones((self.NB, self.NT - 1), dtype=np.int64)
        mv._vertical_ladder_bank(self._census(p, p))
        mv._vertical_ladder_reset()
        self.assertIsNone(mv._vert_ladder_prop)
        self.assertIsNone(mv._vert_ladder_acc)

    def test_it_adapts_and_reports(self):
        mv = self._mv()
        # Set the BACKING field, not the property: eryn's
        # Move.temperature_control setter also rebinds compute_log_posterior
        # and re-derives ntemps/nsamplers from the object, none of which this
        # test needs and all of which a fake would have to impersonate.
        # _adapt_band_temps only reads .adaptation_lag / .adaptation_time.
        mv._temperature_control = SimpleNamespace(
            adaptation_lag=100.0, adaptation_time=10.0)
        mv.time = 5
        p = np.full((self.NB, self.NT - 1), 10, dtype=np.int64)
        a = np.full((self.NB, self.NT - 1), 5, dtype=np.int64)
        mv._vertical_ladder_bank(self._census(p, a))
        bt = np.tile(np.array([1.0, 0.5, 0.25, 0.1]), (self.NB, 1))
        before = bt.copy()
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "INFO") as cm:
            ok = mv._vertical_adapt_ladder(bt)
        self.assertTrue(ok)
        self.assertTrue(any("VERTICAL swaps" in m for m in cm.output))
        # cold and hot rungs are pinned; the interior must be free to move
        np.testing.assert_allclose(bt[:, 0], before[:, 0])
        np.testing.assert_allclose(bt[:, -1], before[:, -1])

    def test_no_proposals_skips_rather_than_collapsing_the_ladder(self):
        """An all-zero ratio column is an ABSENCE of data, not a
        measurement -- adapting on it would drag every rung together."""
        mv = self._mv()
        # Set the BACKING field, not the property: eryn's
        # Move.temperature_control setter also rebinds compute_log_posterior
        # and re-derives ntemps/nsamplers from the object, none of which this
        # test needs and all of which a fake would have to impersonate.
        # _adapt_band_temps only reads .adaptation_lag / .adaptation_time.
        mv._temperature_control = SimpleNamespace(
            adaptation_lag=100.0, adaptation_time=10.0)
        mv.time = 5
        z = np.zeros((self.NB, self.NT - 1), dtype=np.int64)
        mv._vertical_ladder_bank(self._census(z, z))
        bt = np.tile(np.array([1.0, 0.5, 0.25, 0.1]), (self.NB, 1))
        before = bt.copy()
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "INFO") as cm:
            ok = mv._vertical_adapt_ladder(bt)
        self.assertFalse(ok)
        self.assertTrue(any("SKIPPED" in m for m in cm.output))
        np.testing.assert_array_equal(bt, before)

    def test_nothing_banked_is_a_no_op(self):
        mv = self._mv()
        # Set the BACKING field, not the property: eryn's
        # Move.temperature_control setter also rebinds compute_log_posterior
        # and re-derives ntemps/nsamplers from the object, none of which this
        # test needs and all of which a fake would have to impersonate.
        # _adapt_band_temps only reads .adaptation_lag / .adaptation_time.
        mv._temperature_control = SimpleNamespace(
            adaptation_lag=100.0, adaptation_time=10.0)
        self.assertFalse(mv._vertical_adapt_ladder(
            np.tile(np.array([1.0, 0.5, 0.25, 0.1]), (self.NB, 1))))

    def test_the_counts_are_pooled_across_walkers_per_band(self):
        """User ruling: "the tuning should be across walkers per band".
        The banked shape carries NO walker axis -- it is (band, rung pair),
        which is exactly what _adapt_band_temps consumes."""
        mv = self._mv()
        p = np.ones((self.NB, self.NT - 1), dtype=np.int64)
        mv._vertical_ladder_bank(self._census(p, p))
        self.assertEqual(mv._vert_ladder_prop.shape, (self.NB, self.NT - 1))


class GroupPEGuardTest(unittest.TestCase):
    """The group rule is SEARCH-ONLY.

    Non-uniform per-source effort is sanctioned in search (user ruling
    2026-09-24: "this is okay in search") and forbidden in PE, where the
    number of sweeps a source receives would depend on the state in a way
    the posterior never sanctioned -- on top of the optional-stopping
    problem the plateau exit already carries.
    """

    def _mv(self, name):
        mv = _make_move(25)
        mv.name = name
        mv.is_rj_prop = False
        mv.inmodel_group = True
        mv.inmodel_group_iters = 3
        mv.inmodel_group_dll = 4.0
        mv.inmodel_group_max_passes = 20
        mv.inmodel_group_scale = "flat"
        return mv

    def test_a_pe_stage_move_is_refused_with_a_warning(self):
        mv = self._mv("in_model_pe")
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "WARNING") as cm:
            self.assertIsNone(mv._group_state_or_none())
        self.assertTrue(any("search-only" in m for m in cm.output))
        self.assertTrue(mv._group_pe_warned)

    def test_a_search_stage_move_still_gets_its_state(self):
        self.assertIsNotNone(self._mv("in_model")._group_state_or_none())


class SerialWithinBandFreezeTest(unittest.TestCase):
    """REGRESSION: the DIRECT-BATCH RJ path lost the cell freeze.

    ``_pick_sources`` takes ``blocked_specials`` precisely so that "a cell
    already holding a pending alive source is frozen until the accumulated
    in-model flush runs, so the pool can never collect two same-cell
    sources (serial-within-band rule)". The staged-scheduler call site
    passed it. The DIRECT-BATCH call site -- GB_RJ_DIRECT_BATCH=1, the
    production default -- did not.

    What that cost: ``_pooled_host`` dedups the POOL, and only AFTER
    ``_run_rj_step`` has proposed and accepted. ``has_run_rj`` retires each
    SOURCE, not each cell, so a later round could pick a different dead
    slot in a cell that had just birthed, accept a second birth into the
    state, and then have it dropped by the pool dedup -- never polished,
    left at its birth coordinates for the rest of the run.

    Measured on the job-621 checkpoint: 33 of 96 (rung, walker) cells held
    more than one leaf on the SINGLE injected source at 19.66822 mHz,
    summing to 1.5-3.4x its true amplitude, the extras at 0.76-0.99x each.
    """

    def _direct_batch_src(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._run_rj_and_inmodel) \
            if hasattr(g.GBSpecialBase, "_run_rj_and_inmodel") else None
        if src is None:
            # the loop lives in the big propose helper; fall back to module
            src = inspect.getsource(g)
        return src

    def test_only_the_NON_POOLING_path_may_omit_blocked_specials(self):
        """Exactly one call site is allowed to skip the freeze.

        The two POOLING paths -- direct-batch and the staged scheduler --
        accumulate survivors and polish them later, so a second same-cell
        birth in between is the bug. The third call site is the
        ``GB_RJ_GROUPED_INMODEL=0`` per-round interleave, which polishes
        each pick immediately: there is no pool to protect, so it needs no
        freeze. Pinning the COUNT rather than deleting the check keeps a
        newly added pooling call site from silently joining the exception.
        """
        import inspect
        import re

        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g)
        without = []
        for m in re.finditer(r"self\._pick_sources\(", src):
            tail, depth, arg = src[m.end():m.end() + 400], 1, []
            for ch in tail:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                arg.append(ch)
            if "blocked_specials" not in "".join(arg):
                without.append("".join(arg).strip()[:70])
        self.assertEqual(
            len(without), 1,
            f"expected exactly one non-pooling call site, found {without}")
        # ...and it must be the plain interleave one, not a pooling path
        self.assertNotIn("bview", without[0])

    def test_the_direct_batch_loop_refreshes_the_frozen_set(self):
        """Passing the gate is not enough -- it has to be REBUILT as cells
        pool, or only the cells frozen before round 1 are ever honoured."""
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g)
        self.assertIn("_pooled_dev = xp.asarray(", src)
        self.assertIn("blocked_specials=_pooled_dev", src)


class VerticalLadderFanoutTest(unittest.TestCase):
    """REGRESSION: the ladder froze on the fan-out path.

    ``_adapt_band_temps`` is reachable from ONE place, the permuted-swap
    block, so GB_RUN_FANCY_TEMPERING=0 does not merely stop those swaps --
    it freezes the ladder for the whole run, silently.
    ``_vertical_adapt_ladder`` exists for exactly that and was wired into
    ``_propose_legacy`` only.

    Measured on the job-621 checkpoint: all 1232 bands still carried ONE
    identical 1.2-geometric ladder, and every band_swaps_* counter was 0.
    """

    def test_the_rank_ships_its_vertical_census(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._gb_serve_run_proposal)
        for key in ("vert_ladder_prop", "vert_ladder_acc"):
            self.assertIn(key, body,
                          f"the rank reply does not carry {key}")

    def test_the_head_pools_across_blocks_and_adapts(self):
        """⚠ Pooling must happen on the HEAD: band_temps has a band axis and
        no walker axis, so four ranks adapting from one walker each would
        write four ladders for the same band and the merge keeps one."""
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._propose_orchestrated)
        self.assertIn("_vert_prop += np.asarray", body)
        self.assertIn("_vert_acc += np.asarray", body)
        self.assertIn("_vertical_adapt_ladder", body)

    def test_the_head_never_adapts_twice_in_one_propose(self):
        """Both routes call the same _adapt_band_temps on the same array;
        adapting twice would take two ladder steps for one measurement."""
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._propose_orchestrated)
        self.assertIn("_ladder_adapted", body)
        self.assertIn("if not _ladder_adapted", body)


class VerticalLadderPerBlockStepTest(unittest.TestCase):
    """The LOCAL per-block ladder step (user: "every 25 iterations").

    One in-model block IS 25 sweeps, so "every block" is that cadence.
    This step is a within-propose refinement only -- the head still
    overwrites every rank's ladder from the walker-POOLED counts at the end
    of the propose, because band_temps has no walker axis.
    """

    def test_the_cadence_knob_exists_and_defaults_to_every_block(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertIn("GB_TEMPER_VERT_ADAPT_EVERY", body)
        self.assertIn('"GB_TEMPER_VERT_ADAPT_EVERY", "1"', body)
        self.assertIn("self._adapt_band_temps(band_temps, _ba, _bp)", body)

    def test_it_still_banks_for_the_pooled_step(self):
        """The local step must not REPLACE the banking -- the head's pooled
        adaptation is the authoritative one and needs the totals."""
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        i_bank = body.index("_vertical_ladder_bank(_cn)")
        i_adapt = body.index("GB_TEMPER_VERT_ADAPT_EVERY")
        self.assertLess(i_bank, i_adapt, "banking must precede the local step")

    def test_the_block_counter_resets_per_propose(self):
        """Left running, the cadence would count blocks across the whole RUN
        and which block takes the step would drift with the propose index."""
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._vertical_ladder_reset)
        self.assertIn("self._vert_block_i = 0", body)

    def test_zero_disables_the_local_step(self):
        """0 must leave ONLY the pooled head-side adaptation, which is the
        fallback if the local step ever looks noisy in a live run."""
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertIn("if _adapt_every > 0", body)


class StagedFlushConvergesTest(unittest.TestCase):
    """The staged scheduler's flush must POLISH TO CONVERGENCE.

    User ruling 2026-09-25: "RJ 1 round -> in model converge -> RJ 1 round
    -> in model converge (PE should be the same except not the in-model
    converge)."

    Before this, the two paths each had half the answer: direct-batch had
    the convergence rule and the wrong schedule (ALL RJ rounds, then one
    in-model phase -- ~12 rounds per propose, so a cell could accept a
    dozen births unpolished); the staged scheduler had the right schedule
    and a flat ``inmodel_repeats_survivor`` block with no convergence and
    no newborn/mature split.
    """

    def _staged_src(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        return inspect.getsource(g)

    def test_the_flush_splits_by_class_and_passes_converge(self):
        src = self._staged_src()
        self.assertIn("for _cls_name, _cls in _split_by_newborn(merged, self.xp)",
                      src)
        self.assertIn("converge=_cv,", src)

    def test_the_pool_carries_pick_time_provenance(self):
        """_split_by_newborn needs a 'newborn' key. Without it every pooled
        row would take the mature budget -- which is how this path came to
        run a flat 50 repeats for newborns too."""
        src = self._staged_src()
        # ``alive_at_pick`` is captured before the RJ step and feeds the
        # flag; an accepted REPLACE is OR-ed in on top (see
        # ReplaceGetsConvergencePolishTest).
        self.assertIn("alive_at_pick = band_sorter.inds[picked[\"ids\"]].copy()",
                      src)
        self.assertIn("_nb = ~alive_at_pick", src)
        self.assertIn('held["newborn"] = _nb[alive_now]', src)

    def test_PE_needs_no_branch(self):
        """PE gets the same schedule WITHOUT convergence for free: the
        state factory refuses a PE-stage move, so _cv is None and the class
        budget applies. A separate PE code path would be a second thing to
        keep in sync."""
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._converge_state_for)
        self.assertIn("_converge_stage_allows", body)


class ReplaceGetsConvergencePolishTest(unittest.TestCase):
    """An accepted REPLACE must be polished like a birth.

    User ruling 2026-09-25: "replace should get convergence polish."

    The provenance flag was "dead at pick -> newborn". A replace writes
    brand-new parameters onto a row that was ALIVE at pick, so every
    swapped source was classified MATURE and took the fixed survivor
    budget instead of the convergence rule -- despite being exactly as
    unpolished as a birth. (``rj_prior_removal`` survivors stay mature,
    correctly: nothing about them changed.)
    """

    def test_the_replace_step_stashes_a_picked_aligned_accept_mask(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._run_replace_step)
        self.assertIn("self._last_replace_accept = None", body,
                      "stale mask from a previous round would leak")
        self.assertIn("_acc_picked[sel[accept]] = True", body)
        self.assertIn("self._last_replace_accept = _acc_picked", body)

    def test_BOTH_pooling_sites_fold_it_into_newborn(self):
        """Direct-batch and staged both build the pool; a fix on one only
        would make the polish depend on GB_RJ_DIRECT_BATCH."""
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g)
        self.assertEqual(src.count('held["newborn"] = _nb[alive_now]'), 2)
        self.assertEqual(src.count("_nb = _nb | _ra"), 2)

    def test_a_rejected_replace_stays_mature(self):
        """The mask is the ACCEPT mask, not the picked set -- a rejected
        swap leaves the source at its old parameters, which are already
        polished."""
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._run_replace_step)
        self.assertIn("_acc_picked[sel[accept]] = True", body)
        self.assertNotIn("_acc_picked[:] = True", body)

    def _gate(self, name, classes):
        """Does the convergence driver arm for this move / class?"""
        from types import SimpleNamespace

        import lisatools.globalfit.moves.gbspecialstretch as g
        f = SimpleNamespace(
            name=name, branch_name="gb", inmodel_converge="on",
            inmodel_converge_classes=frozenset(classes),
            inmodel_repeats_newborn=100, inmodel_repeats_survivor=50,
            inmodel_converge_iters=250, inmodel_converge_max=20000,
            inmodel_converge_dll=4.0, inmodel_converge_gate_frac=0.5,
            inmodel_converge_stop_frac=0.5, inmodel_converge_refill=True,
            ntemps=24, _converge_armed_logged=True, _converge_pe_warned=True,
        )
        return {
            c: g.GBSpecialBase._converge_state_for(f, c) is not None
            for c in ("newborn", "mature")
        }

    def test_BOTH_classes_converge_for_replace_under_the_shipped_classes(self):
        """User ruling 2026-09-25: replace runs "both successful changes
        and survivors" to convergence.

        Accepted swap -> newborn, rejected swap -> mature, and the shipped
        GB_INMODEL_CONVERGE_CLASSES=newborn,mature arms BOTH -- so neither
        falls back to a flat budget. This is the behavioural half of the
        source assertions above; the tests either side of it would all
        still pass if the classes knob silently dropped one class.
        """
        self.assertEqual(self._gate("rj_replace", ("newborn", "mature")),
                         {"newborn": True, "mature": True})

    def test_the_default_classes_would_leave_survivors_on_a_flat_budget(self):
        """The control: the knob is what carries the survivor half, so the
        stock default really does behave differently. Without this, the
        test above passes for the wrong reason."""
        self.assertEqual(self._gate("rj_replace", ("newborn",)),
                         {"newborn": True, "mature": False})

    def test_replace_PE_still_refuses_both(self):
        """A convergence-plateau stop is a search-only licence."""
        self.assertEqual(self._gate("rj_replace_pe", ("newborn", "mature")),
                         {"newborn": False, "mature": False})


class PerClassConvergeWindowTest(unittest.TestCase):
    """Survivors get their own convergence WINDOW, at the SAME rate.

    User ruling 2026-09-25, off job 628. The window is a FLOOR on block
    length -- a row cannot converge before ``seen >= window`` -- and
    survivors are 96.3% of all converging in-model work against births'
    0.9%. Their measured mean block was 276-306 repeats against a floor of
    255, i.e. ~90% of the bill was the floor, not convergence. Births do
    use the depth: they tail to 1200-1865 where survivors cap at 420-480.

    ⚠ THE PAIR IS A RATE. ``thresh/window`` is lnL per repeat, so
    shortening the window at a fixed thresh does not let survivors stop
    sooner at the same bar -- it LOWERS the bar (250 -> 100 at dll 4.0 is
    2.5x looser, 250 -> 50 is 5x). The survivor threshold is therefore
    DERIVED from its window, and these tests pin that the enforced rate is
    identical for both classes. Without that, this "speedup" would be a
    silent weakening of the convergence criterion.
    """

    def _move(self, window=250, survivor=0, dll=4.0):
        return SimpleNamespace(
            name="rj_warm_search", branch_name="gb", inmodel_converge="on",
            inmodel_converge_classes=frozenset({"newborn", "mature"}),
            inmodel_repeats_newborn=100, inmodel_repeats_survivor=50,
            inmodel_converge_iters=window,
            inmodel_converge_iters_survivor=survivor,
            inmodel_converge_max=20000, inmodel_converge_dll=dll,
            inmodel_converge_gate_frac=0.5, inmodel_converge_stop_frac=0.5,
            inmodel_converge_refill=True, ntemps=24,
            _converge_armed_logged=True, _converge_pe_warned=True,
        )

    def _states(self, **kw):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = self._move(**kw)
        return {c: g.GBSpecialBase._converge_state_for(m, c)
                for c in ("newborn", "mature")}

    def test_off_by_default_both_classes_identical(self):
        """0 = historical behaviour; no run changes meaning silently."""
        st = self._states(survivor=0)
        self.assertEqual(st["newborn"].window, st["mature"].window)
        self.assertEqual(st["newborn"].thresh, st["mature"].thresh)

    def test_survivor_window_shortens_only_the_survivor_class(self):
        st = self._states(window=250, survivor=100)
        self.assertEqual(st["newborn"].window, 250)
        self.assertEqual(st["mature"].window, 100)

    def test_the_enforced_RATE_is_identical_across_classes(self):
        """The whole point. If this fails the change is a weakening of the
        criterion dressed up as a speedup."""
        for surv in (50, 100, 150, 250):
            st = self._states(window=250, survivor=surv)
            rn = st["newborn"].thresh / st["newborn"].window
            rm = st["mature"].thresh / st["mature"].window
            self.assertAlmostEqual(rn, rm, places=12, msg=f"survivor={surv}")
            self.assertAlmostEqual(rn, 4.0 / 250, places=12)

    def test_the_control_a_fixed_thresh_would_have_loosened_the_bar(self):
        """Shows the trap this design avoids: had thresh been left at 4.0,
        a 100-repeat window would enforce 2.5x the permitted drift."""
        st = self._states(window=250, survivor=100)
        naive_rate = 4.0 / 100                      # thresh NOT scaled
        actual_rate = st["mature"].thresh / st["mature"].window
        self.assertAlmostEqual(actual_rate, 4.0 / 250, places=12)
        self.assertAlmostEqual(naive_rate / actual_rate, 2.5, places=6)

    def test_survivor_threshold_scales_down_not_up(self):
        st = self._states(window=250, survivor=100)
        self.assertLess(st["mature"].thresh, st["newborn"].thresh)
        self.assertAlmostEqual(st["mature"].thresh, 1.6, places=9)

    def test_the_armed_line_reports_both_windows(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        body = inspect.getsource(g.GBSpecialBase._converge_state_for)
        self.assertIn("(newborn) / %d (survivor", body)


class UngroupedPathHasNoConvergenceAndSaysSoTest(unittest.TestCase):
    """``GB_RJ_GROUPED_INMODEL=0`` silently disables the whole rule.

    The per-round interleave keeps no survivor pool, so it never calls
    ``_split_by_newborn`` and never consults ``_converge_state_for``: it
    runs a flat ``inmodel_repeats_survivor`` for every row -- births,
    accepted replaces and survivors alike. That is the intended legacy
    behaviour of a baseline path, but an operator who also exported
    ``GB_INMODEL_CONVERGE=on`` got no convergence and no indication of
    it, which is the exact "knob resolves, consuming path never runs"
    shape that produced four separate defects in this run.
    """

    def _src(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        return inspect.getsource(g.GBSpecialBase._run_band_unit)

    def test_the_ungrouped_path_still_takes_the_flat_survivor_budget(self):
        """Pin the premise: if this ever gains a pool, the warning is
        wrong and must go."""
        src = self._src()
        self.assertIn("self.inmodel_repeats_survivor\n", src)
        # the provenance split belongs to the pooled paths only
        self.assertEqual(src.count("_split_by_newborn"), 2)

    def test_it_warns_once_when_convergence_is_armed(self):
        src = self._src()
        self.assertIn("_converge_ungrouped_warned", src)
        self.assertIn("GB_RJ_GROUPED_INMODEL=0", src)
        self.assertIn("Nothing runs to convergence on this path.", src)

    def test_the_warning_is_search_scoped_and_rj_scoped(self):
        """A pure in-model move converges through the GROUP rule instead,
        and PE has no convergence by design -- neither should be nagged."""
        src = self._src()
        blk = src[src.index("grouped = ("):src.index("round_i = 0")]
        self.assertIn("_converge_ungrouped_warned", blk)
        self.assertIn("self.is_rj_prop", blk)
        self.assertIn("_converge_stage_allows(self)", blk)


class PerWalkerValveStatisticTest(unittest.TestCase):
    """REGRESSION: the per-walker RJ valve got a ONE-BLOCK lnL with caps off.

        RuntimeError: rj_fstat_search: the per-walker RJ valve received a
        (4, 1232) cold census and a (1, 1232) lnL statistic

    ``cap_stats`` is the only place a (nwalkers, nbands) cold lnL is
    assembled on the head -- every block ships its rows and they are
    concatenated on the walker axis. Requesting it was gated on
    ``_band_leaf_cap is not None``, so with LEAF CAPS OFF the ranks were
    never asked, nothing stamped the valve's stash, and
    ``_shutoff_band_lls`` fell through to ``_cap_stats_local`` on the HEAD
    -- a rank-local routine that returns ONE block.

    Only reachable once rj_fstat_search (the single designated
    ``leaf_cap_update`` move) completes a propose, which is why several
    runs that died earlier in the cycle never saw it.
    """

    def _orch(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        return inspect.getsource(g.GBSpecialBase._propose_orchestrated)

    def test_the_valve_alone_is_enough_to_request_cap_stats(self):
        body = self._orch()
        self.assertIn("or self._search_shutoff_per_walker)", body)
        self.assertNotIn(
            "want_cap_stats = bool(\n            self._band_leaf_cap is not None "
            "and self.leaf_cap_update)", body,
            "the request is still gated on caps existing")

    def test_caps_off_still_stashes_the_N_walker_statistic(self):
        """The cap gate is the usual writer of the stash; with caps off it
        does not run, so the assembled array must be stamped here."""
        body = self._orch()
        self.assertIn("elif cap_stats is not None and self.leaf_cap_update:",
                      body)
        self.assertIn('self._stage_band_lls = np.array(cap_stats["band_lls"]',
                      body)

    def test_the_cap_GATE_stays_off_when_caps_are_off(self):
        """⚠ The fix must hand over a number, not switch the cap gate on --
        this run has GB_LEAF_CAP_START empty on purpose."""
        body = self._orch()
        self.assertIn(
            "if self._band_leaf_cap is not None and self.leaf_cap_update:",
            body, "the cap gate's own condition must be unchanged")
        i_gate = body.index(
            "if self._band_leaf_cap is not None and self.leaf_cap_update:")
        i_elif = body.index("elif cap_stats is not None")
        self.assertLess(i_gate, i_elif, "the stash must be the ELSE branch")


class GroupStaticOnePassShutoffTest(unittest.TestCase):
    """A (walker, band) that moved NOWHERE in the ladder retires at pass 1.

    User ruling 2026-09-26: "track the max logL of a band-walker set over
    all temperatures. If it is the same at the beginning of the
    in-model-only move and the end of 1 iteration, then that one gets shut
    off. Otherwise we use 2 to shut off."

    WHY IT MATTERS: the group window is a FLOOR -- passes 1..W shut
    nothing, because a pair needs W passes of history before the ring test
    can fire. On job 634 those warm-up passes were 62.6% of ALL pure
    in-model work across the three slots (each slot runs its own group
    state, so the warm-up is paid once PER slot). This is the only exit
    that does not pay the floor.
    """

    NW, NB, NT = 2, 4, 3

    def _state(self, window=2, thresh=4.0):
        import lisatools.globalfit.moves.gbspecialstretch as g
        return g._InModelGroupState(
            window=window, thresh=thresh, max_passes=100, scale="flat")

    def _occ(self, n=1):
        return np.full((self.NW, self.NB), float(n))

    def test_a_pair_that_moved_nowhere_shuts_at_pass_1(self):
        st = self._state()
        allt = np.zeros((self.NT, self.NW, self.NB))     # nothing accepted
        st.update(np, allt[0], self._occ(), all_temp_delta=allt)
        self.assertEqual(st.passes, 1)
        self.assertTrue(bool(np.asarray(st.shut).all()),
                        "a fully static ladder must retire on pass 1")
        self.assertEqual(st.static_shut, self.NW * self.NB)

    def test_movement_on_ANY_rung_keeps_it_open(self):
        """Cold flat but a HOT rung moving is exactly the case the
        cold-only statistic would have missed."""
        st = self._state()
        allt = np.zeros((self.NT, self.NW, self.NB))
        allt[2, 0, 0] = 1e-9                              # hottest rung only
        st.update(np, allt[0], self._occ(), all_temp_delta=allt)
        self.assertFalse(bool(np.asarray(st.shut)[0, 0]),
                         "movement anywhere in the ladder must hold it open")
        self.assertEqual(st.static_shut, self.NW * self.NB - 1)

    def test_a_BUSY_pair_whose_MAX_did_not_rise_also_retires(self):
        """The case my first draft missed. User ruling 2026-09-26: "It can
        have delta_ll, but over the course of the 1 iteration beginning to
        end, the max logL did not adjust at all" -- accepted moves that
        shuffle a cell without lifting its best rung are exactly the churn
        this is meant to stop paying for. A ``delta_ll == 0 everywhere``
        test only catches the completely dead pairs."""
        st = self._state()
        allt = np.zeros((self.NT, self.NW, self.NB))
        allt[0, 0, 0] = -5.0        # busy, but DOWN
        allt[1, 0, 0] = -2.0
        allt[2, 0, 0] = -0.1
        self.assertNotEqual(float(np.abs(allt).sum()), 0.0, "must be busy")
        st.update(np, allt[0], self._occ(), all_temp_delta=allt)
        self.assertTrue(bool(np.asarray(st.shut)[0, 0]),
                        "max never rose, so it must retire at pass 1")

    def test_a_pair_whose_max_ROSE_stays_open_even_if_others_fell(self):
        """The discriminator: one rung up is enough, however much the rest
        moved down. The statistic is the MAX over the ladder."""
        st = self._state()
        allt = np.zeros((self.NT, self.NW, self.NB))
        allt[0, 0, 0] = -50.0
        allt[2, 0, 0] = +0.5        # one rung reached a new high
        st.update(np, allt[0], self._occ(), all_temp_delta=allt)
        self.assertFalse(bool(np.asarray(st.shut)[0, 0]))

    def test_the_control_without_the_ladder_nothing_shuts_at_pass_1(self):
        """THE CONTROL. Same static input, fast path not armed: the window
        is 2, so pass 1 can retire nothing. If this ever passes, the test
        above is not measuring the fast path."""
        st = self._state()
        allt = np.zeros((self.NT, self.NW, self.NB))
        st.update(np, allt[0], self._occ())               # no all_temp_delta
        self.assertFalse(bool(np.asarray(st.shut).any()))
        self.assertEqual(st.static_shut, 0)

    def test_it_only_fires_on_pass_1(self):
        """A pair that goes quiet LATER must still be judged by the
        windowed rule -- the fast path is about never paying the floor,
        not about a zero-tolerance criterion."""
        st = self._state()
        moving = np.zeros((self.NT, self.NW, self.NB)); moving[0] = 50.0
        st.update(np, moving[0], self._occ(), all_temp_delta=moving)
        self.assertFalse(bool(np.asarray(st.shut).any()))
        quiet = np.zeros((self.NT, self.NW, self.NB))
        st.update(np, quiet[0], self._occ(), all_temp_delta=quiet)
        self.assertEqual(st.static_shut, 0, "fast path must not re-fire")

    def test_unoccupied_pairs_are_unaffected(self):
        """They already shut immediately; the fast path must not double
        count them into static_shut."""
        st = self._state()
        allt = np.zeros((self.NT, self.NW, self.NB))
        st.update(np, allt[0], np.zeros((self.NW, self.NB)),
                  all_temp_delta=allt)
        self.assertTrue(bool(np.asarray(st.shut).all()))
        self.assertEqual(st.static_shut, 0)

    def test_the_call_site_passes_the_WHOLE_ladder(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._run_group_passes)
        self.assertIn("all_temp_delta=ll_change_log", src)
        self.assertNotIn("all_temp_delta=ll_change_log[0]", src)


class ConvergeReportReachesProductionTest(unittest.TestCase):
    """The ``[GB_IMCONV]`` report must fire on the path production takes.

    It existed only on the DIRECT-BATCH path, and production runs
    GB_RJ_DIRECT_BATCH=0 (the staged scheduler). Job 634 emitted twelve
    [GB_IMCONV] lines and every one was "armed" -- the richest diagnostic
    the convergence rule has (rows converged vs ceiling vs released,
    columns fully retired, work as a multiple of the fixed budget, and the
    window/dll/floor actually in force for THIS class) was unreachable in
    every production run to date.

    That is the same "knob resolves, consuming path never runs" shape as
    the defects it is supposed to help find -- in its diagnostic form,
    which is worse, because a missing diagnostic is invisible by
    definition.

    It is also the only AFFIRMATIVE evidence that the per-class survivor
    window does anything: the armed line reports what was CONFIGURED, the
    report reports what the rows actually did.
    """

    def _src(self):
        import inspect

        import lisatools.globalfit.moves.gbspecialstretch as g
        return inspect.getsource(g.GBSpecialBase._run_band_unit)

    def test_BOTH_flush_paths_emit_it(self):
        src = self._src()
        self.assertEqual(
            src.count("_cv.report("), 2,
            "one call is the direct-batch path only; production takes the "
            "staged one and would log nothing")

    def test_each_call_sits_under_a_None_guard(self):
        """``_converge_state_for`` returns None whenever the rule is off
        (mode off, class not selected, PE stage), and the report must not
        fire then."""
        src = self._src()
        self.assertEqual(src.count("if _cv is not None:\n"
                                   "                        _cv.report("), 1)
        self.assertEqual(src.count("if _cv is not None:\n"
                                   "                            _cv.report("), 1)

    def test_it_is_keyed_by_CLASS(self):
        """newborn and mature are separate states with separate windows;
        one line per class or the survivor window cannot be read off."""
        src = self._src()
        self.assertEqual(src.count("self.name, _cls_name,"), 2)
