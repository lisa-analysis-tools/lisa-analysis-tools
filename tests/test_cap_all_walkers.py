"""The leaf cap may not open until EVERY engaged walker has plateaued.

The shipped gate runs one patience clock per cell on ``lls.max(axis=0)``.
That series is level-invariant (it compares a running best over TIME), so a
constant offset between walkers does not blind it -- but it follows whichever
walker is on top, so a LAGGARD still climbing is invisible to it, and the cap
it opens applies to every walker including the ones that were not ready.

``GB_LEAF_CAP_ALL_WALKERS=1`` adds a per-walker clock and requires all engaged
walkers to be inside a plateau. The properties under test are the ones that
make it safe to switch on mid-campaign:

* it can only ever TIGHTEN the gate (a cell it blocks, the aggregate gate
  would have allowed; never the reverse);
* a walker that never engaged in a cell cannot freeze that cell forever --
  the failure mode that a naive "all walkers" reading produces, because an
  unengaged walker's clock sits at 0 and never reaches min_iters;
* a single laggard DOES block, which is the whole point.
"""

import unittest

import numpy as np


class _Gate:
    """The helper under test, with only the attributes it touches."""

    leaf_cap_min_iters = 3

    def __init__(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialBase,
        )

        self._fn = GBSpecialBase._cap_all_walkers_converged

    def step(self, lls, occ, thresh=4.5, tol=0.1):
        return self._fn(self, np.asarray(lls, float),
                        np.asarray(occ, float), thresh, tol)


def _run(gate, series, occ=None, n=None):
    """Feed rows of (nw, ncells); return the last mask."""
    out = None
    for row in series:
        row = np.asarray(row, float)
        o = np.ones_like(row) if occ is None else np.asarray(occ, float)
        out = gate.step(row, o)
    return out


class AllWalkersConvergedTest(unittest.TestCase):
    def setUp(self):
        self.g = _Gate()

    def test_all_flat_walkers_converge(self):
        """Two walkers, both plateaued past the window -> cell may ramp."""
        rows = [[[10.0], [3.0]]] * 8          # engage on row 0, flat after
        self.assertTrue(bool(_run(self.g, rows)[0]))

    def test_one_climbing_walker_blocks(self):
        """Walker 1 keeps improving by > D/2 -> the cell must NOT ramp."""
        rows = []
        for i in range(8):
            rows.append([[10.0], [3.0 + 10.0 * i]])   # w1 climbs hard
        self.assertFalse(bool(_run(self.g, rows)[0]))

    def test_the_laggard_is_invisible_to_the_max_series(self):
        """The motivating case: leader flat, laggard climbing far below it.

        ``max(axis=0)`` is constant here, so the aggregate clock runs out and
        the aggregate gate would ramp. The per-walker gate must not.
        """
        rows = [[[1000.0], [1.0 + 20.0 * i]] for i in range(8)]
        mx = np.array([np.max(r, axis=0)[0] for r in rows])
        self.assertTrue(np.all(mx == mx[0]), "max series must be flat")
        self.assertFalse(bool(_run(self.g, rows)[0]))

    def test_unengaged_walker_does_not_freeze_the_cell(self):
        """A walker with no source here must not block forever."""
        # w1 is never occupied and its statistic never moves.
        rows = [[[10.0], [-np.inf]]] * 8
        occ = [[1.0], [0.0]]
        self.assertTrue(bool(_run(self.g, rows, occ=occ)[0]))

    def test_no_walker_engaged_means_no_ramp(self):
        """An empty cell stays frozen -- the ghost-increment guard."""
        rows = [[[-np.inf], [-np.inf]]] * 8
        occ = [[0.0], [0.0]]
        self.assertFalse(bool(_run(self.g, rows, occ=occ)[0]))

    def test_patience_window_is_respected_per_walker(self):
        """Exactly min_iters flat rows are needed after the last improvement."""
        g = self.g
        base = [[10.0], [3.0]]
        g.step(np.array(base), np.ones((2, 1)))          # engage
        for _ in range(g.leaf_cap_min_iters - 1):
            out = g.step(np.array(base), np.ones((2, 1)))
        self.assertFalse(bool(out[0]), "ramped before the window closed")
        out = g.step(np.array(base), np.ones((2, 1)))
        self.assertTrue(bool(out[0]), "did not ramp once the window closed")

    def test_occupancy_stops_the_clock(self):
        """A walker whose source died holds its clock instead of ratcheting."""
        g = self.g
        g.step(np.array([[10.0], [3.0]]), np.ones((2, 1)))   # both engage
        # w1 loses its source: occupied=0. Its clock must not advance, so it
        # stays blocking at iters < min_iters.
        for _ in range(10):
            out = g.step(np.array([[10.0], [3.0]]),
                         np.array([[1.0], [0.0]]))
        self.assertFalse(bool(out[0]))

    def test_shape_change_resets_cleanly(self):
        """A walker rescale or resume reshapes the state without raising."""
        g = self.g
        g.step(np.array([[10.0], [3.0]]), np.ones((2, 1)))
        out = g.step(np.zeros((4, 5)), np.ones((4, 5)))
        self.assertEqual(out.shape, (5,))

    def test_it_only_ever_tightens(self):
        """Random trial: never True where a max-series clock would be False."""
        rng = np.random.default_rng(0)
        g = self.g
        nw, nc, mi = 4, 40, g.leaf_cap_min_iters
        best = np.full(nc, -np.inf)
        iters = np.zeros(nc, int)
        seen = np.zeros(nc, bool)
        prev = None
        for _ in range(30):
            lls = rng.normal(size=(nw, nc)) * 5.0
            occ = np.ones((nw, nc))
            mine = g.step(lls, occ)
            cur = lls.max(axis=0)
            chg = (np.ones(nc, bool) if prev is None
                   else np.abs(cur - prev) > 0.1)
            prev = cur.copy()
            seen |= chg
            imp = cur > (best + 4.5)
            best = np.maximum(best, cur)
            iters[imp] = 0
            iters[~imp & seen] += 1
            agg = iters >= mi
            self.assertFalse(bool(np.any(mine & ~agg)),
                             "per-walker gate opened where the aggregate did not")


if __name__ == "__main__":
    unittest.main()
