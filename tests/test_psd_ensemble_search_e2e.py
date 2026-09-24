"""End-to-end run of the tiled per-walker ensemble search.

``test_psd_ensemble_search`` pins the arithmetic -- the folded row order,
the ``data_index`` map, the closed loop on the measured spread, the argmax
fold-back -- by lifting those helpers onto a stub. None of it runs
:meth:`PSDMove.run_move_ensemble_search`, so the seam that actually carries
risk was never exercised: a REAL ``eryn.moves.StretchMove`` proposing on a
folded ``(B*nt, R, 1, ndim)`` :class:`GFState`, with a REAL
``TemperatureControl(nsamplers=B)`` swapping inside each walker's own
ladder, and the supplemental surviving eryn's red/blue slicing.

This file runs that. It stops short of the ACA: the likelihood is a stub
that reads ``walker_inds`` off the supplemental it is handed, which is what
makes the walker-isolation assertion below meaningful -- a walker scored
against another walker's residual is the
``project_psd_fancy_swap_walker_inds_defect_0916`` failure in a new place,
and it is invisible to every arithmetic-only test.
"""

import unittest

import numpy as np
from eryn.moves.tempering import TemperatureControl
from eryn.prior import ProbDistContainer, uniform_dist

from lisatools.globalfit.moves.psdmove import PSDMove


NT, B, R, NDIM = 3, 2, 10, 5
LO, HI = -6.0, 6.0

# each walker's own likelihood optimum, well inside the prior box; the
# stub scores walker w against ITS centre, so a mis-mapped walker_inds
# shows up as a walker that cannot improve.
CENTRES = np.array([[2.0] * NDIM, [-3.0] * NDIM])


class _Ensemble(PSDMove):
    """The ensemble-search path with the ACA and the ctor cut away.

    Only the methods under test are inherited live; everything the real
    ctor would have wired (acs, dcga, transforms, sensitivity backend) is
    irrelevant to this seam and is not built.
    """

    def __init__(self, num_repeats=3, repeats=R, ntemps=NT, nwalkers=B):
        self.sampled_branches = ["galfor"]
        self.priors = {
            "galfor": ProbDistContainer(
                {i: uniform_dist(LO, HI) for i in range(NDIM)}
            )
        }
        self.periodic = None
        self.num_repeats = num_repeats
        self.ensemble_search = True
        self.ensemble_repeats = repeats
        self.ensemble_spread_lo = 1.0
        self.ensemble_spread_hi = 10.0
        self.ensemble_scale0 = 1e-2
        self.ensemble_scale_tries = 6
        self._ensemble_inner = None
        self.gf_stage_kind = "search"
        self.accepted = np.zeros((ntemps, nwalkers))
        self.num_proposals = 0
        self._tally_in_model_proposed = np.zeros(ntemps, dtype=int)
        self._tally_in_model_accepted = np.zeros(ntemps, dtype=int)
        self._tally_swaps_proposed = np.zeros(ntemps - 1, dtype=int)
        self._tally_swaps_accepted = np.zeros(ntemps - 1, dtype=int)
        self.temperature_control = TemperatureControl(
            NDIM, nwalkers, ntemps=ntemps, Tmax=1e6
        )


class _LikeStub:
    """Scores each row against the centre of the walker it CLAIMS to be.

    Records every ``walker_inds`` block it is handed so the test can assert
    that eryn's red/blue slicing kept each walker's rows on that walker.
    """

    def __init__(self):
        self.seen = []

    def __call__(self, coords, inds=None, logp=None, supps=None, **kw):
        x = np.asarray(coords["galfor"])[..., 0, :]
        rows, cols = x.shape[0], x.shape[1]
        if supps is not None:
            # BranchSupplemental.__getitem__ indexes, it does not look up a
            # name -- reach into .holder, exactly as
            # PSDMove.compute_log_like does. reshape() to the coord block is
            # also what the real reader does (`.reshape(logp.shape)`), so
            # this test fails if eryn's red/blue slicing ever hands back a
            # supplemental whose flat order disagrees with the coords.
            w = np.asarray(supps.holder["walker_inds"]).reshape(rows, cols)
            self.seen.append(w.copy())
        else:  # pragma: no cover - every call site passes supps
            w = np.zeros((rows, cols), dtype=int)
        d = x - CENTRES[w]
        # (logl, blobs), the eryn signature PSDMove.compute_log_like returns
        return -0.5 * np.sum(d ** 2, axis=-1), None


def _model(move, tc, like, seed=3):
    from eryn.model import Model

    return Model(None, like, move.compute_log_prior, tc,
                 map, np.random.RandomState(seed))


def _state(move, rng):
    from lisatools.globalfit.state import GFState
    from eryn.state import BranchSupplemental

    coords = {"galfor": rng.uniform(-1.0, 1.0, size=(NT, B, 1, NDIM))}
    supps = BranchSupplemental(
        {"walker_inds": np.tile(np.arange(B), (NT, 1))},
        base_shape=(NT, B), copy=True,
    )
    st = GFState(coords, copy=True, supplemental=supps)
    st.log_prior = move.compute_log_prior(coords)
    st.log_like = np.zeros((NT, B))
    return st


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.move = _Ensemble()
        self.like = _LikeStub()
        rng = np.random.RandomState(7)
        self.state = _state(self.move, rng)
        self.before = {
            k: v.copy() for k, v in self.state.branches_coords.items()
        }
        self.model = _model(self.move, self.move.temperature_control, self.like)
        self.out, self.acc = self.move.run_move_ensemble_search(
            self.model, self.state
        )

    def test_it_runs_and_keeps_the_module_ladder_shape(self):
        """The whole point: a real StretchMove on the folded state returns."""
        self.assertEqual(
            self.out.branches_coords["galfor"].shape, (NT, B, 1, NDIM)
        )
        self.assertEqual(self.out.log_like.shape, (NT, B))
        self.assertEqual(self.out.log_prior.shape, (NT, B))
        self.assertEqual(np.shape(self.acc), (NT, B))

    def test_the_inner_move_is_eryns_stretch(self):
        from eryn.moves import StretchMove

        self.assertIsInstance(self.move._ensemble_inner, StretchMove)

    def test_every_walkers_rows_stay_on_that_walker(self):
        """Walker isolation through eryn's red/blue slicing.

        Each block the likelihood saw must be constant along its row: row
        ``b*nt + t`` is walker ``b``, whichever half of the split it landed
        in. A row carrying two different walker ids means a walker was
        scored against another walker's residual.
        """
        self.assertTrue(self.like.seen, "the likelihood was never called")
        for blk in self.like.seen:
            self.assertTrue(
                np.all(blk == blk[:, :1]),
                "a folded row mixed walkers inside one likelihood block",
            )
            # and the row's id is the one the fold promises
            self.assertTrue(
                np.array_equal(
                    blk[:, 0], np.repeat(np.arange(B), blk.shape[0] // B)
                )
            )

    def test_the_cold_row_never_gets_worse(self):
        """Copy 0 is unperturbed, so the block can only improve.

        This is what makes the deterministic argmax admissible: the
        incumbent is always a candidate, so the enclosing plateau criterion
        sees a monotone cold-row log-likelihood.
        """
        start = self.like(self.before, supps=self.state.supplemental)[0]
        self.assertTrue(
            np.all(self.out.log_like[0] >= start[0] - 1e-9),
            f"cold row degraded: {self.out.log_like[0]} < {start[0]}",
        )

    def test_the_cold_row_is_that_walkers_best_over_the_whole_set(self):
        """Cold row >= every hot rung of the SAME walker (global argmax)."""
        for b in range(B):
            self.assertGreaterEqual(
                self.out.log_like[0, b] + 1e-9,
                float(np.max(self.out.log_like[:, b])),
            )

    def test_the_search_actually_moves_toward_each_walkers_own_centre(self):
        """Walker 0 climbs toward +2 and walker 1 toward -3, not each other."""
        cold = self.out.branches_coords["galfor"][0, :, 0, :]
        start = self.before["galfor"][0, :, 0, :]
        for b in range(B):
            d_new = np.linalg.norm(cold[b] - CENTRES[b])
            d_old = np.linalg.norm(start[b] - CENTRES[b])
            self.assertLess(d_new, d_old, f"walker {b} did not improve")

    def test_log_like_matches_the_returned_coordinates(self):
        """The fold-back must not mix a coordinate row with another's lnL."""
        recomputed = self.like(
            self.out.branches_coords, supps=self.state.supplemental
        )[0]
        np.testing.assert_allclose(recomputed, self.out.log_like, rtol=1e-9)

    def test_tallies_are_shaped_for_the_sub_state_write(self):
        """run_move writes these straight into sub.in_model_* / sub.swaps_*."""
        self.assertEqual(self.move._tally_in_model_proposed.shape, (NT,))
        self.assertEqual(self.move._tally_in_model_accepted.shape, (NT,))
        self.assertEqual(self.move._tally_swaps_proposed.shape, (NT - 1,))
        self.assertEqual(self.move._tally_swaps_accepted.shape, (NT - 1,))
        self.assertTrue(np.all(self.move._tally_in_model_proposed > 0))
        self.assertTrue(np.all(self.move._tally_swaps_proposed > 0))
        self.assertTrue(
            np.all(
                self.move._tally_swaps_accepted
                <= self.move._tally_swaps_proposed
            )
        )

    def test_the_prior_is_respected(self):
        self.assertTrue(np.all(np.isfinite(self.out.log_prior)))


class OneWalkerBlockTest(unittest.TestCase):
    """B = 1 is the configuration this move exists for.

    A one-walker block has no stretch complement at all, which is what
    sends ``_resolve_inner_kind`` to the failing eigen path. It also sends
    ``TemperatureControl`` down its ``nsamplers == 1`` branch, where
    ``swaps_accepted`` is ``(ntemps-1,)`` rather than
    ``(nsamplers, ntemps-1)`` -- the reshape in ``_ensemble_take_best`` has
    to hold in both.
    """

    def test_one_walker_runs_and_improves(self):
        global B
        old_b = B
        try:
            B = 1
            move = _Ensemble(nwalkers=1)
            like = _LikeStub()
            state = _state(move, np.random.RandomState(5))
            out, acc = move.run_move_ensemble_search(
                _model(move, move.temperature_control, like), state
            )
            self.assertEqual(out.branches_coords["galfor"].shape,
                             (NT, 1, 1, NDIM))
            self.assertEqual(np.shape(acc), (NT, 1))
            self.assertEqual(move._tally_swaps_accepted.shape, (NT - 1,))
            cold = out.branches_coords["galfor"][0, 0, 0, :]
            self.assertLess(
                np.linalg.norm(cold - CENTRES[0]), np.linalg.norm(CENTRES[0])
            )
        finally:
            B = old_b


if __name__ == "__main__":
    unittest.main()
