"""GFCombineMove PE-only exclusive RJ draw (user ruling 2026-09-10).

In a stage of kind ``"pe"`` that wraps BOTH ``rj_fstat_pe`` and
``rj_prior_pe``, ``GB_PE_RJ_DRAW_ONE=1`` makes exactly ONE of the two run
per iteration, drawn from the sampler RNG with
P(rj_fstat_pe) = ``GB_PE_RJ_FSTAT_FRACTION`` (default 0.8); every other
wrapped move runs once per iteration in its fixed order. Search / rj
stages, PE stages without the pair, and runs with the knob unset are
untouched (they keep the stage's configured weighted-cycle / sequential
mode).

Stub moves, no sampler, no GPU, no data (same style as
``test_gf_combine_weighted.py``).
"""

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.globalfitmove import GFCombineMove


class _StubMove:
    temperature_control = None
    periodic = None

    def __init__(self, name, log):
        self.gf_move_name = name
        self.log = log
        self.accepted = np.zeros((2, 4))

    def propose(self, model, state):
        self.log.append(self.gf_move_name)
        return state, np.ones((2, 4))


class _StubModel:
    def __init__(self, seed=1234):
        self.random = np.random.RandomState(seed)


class _StubState:
    sub_states = None


PE_NAMES = ("rj_warm_pe", "rj_fstat_pe", "rj_prior_pe", "gb_ridge_gibbs")
RJ = ("rj_fstat_pe", "rj_prior_pe")


def _make(names, log, kind="pe", **kwargs):
    moves = [_StubMove(n, log) for n in names]
    kwargs.setdefault("share_temperature_control", False)
    comb = GFCombineMove(moves=moves, **kwargs)
    comb.gf_stage_kind = kind
    comb.gf_stage_name = "full_pe" if kind == "pe" else "gb_search"
    return moves, comb


def _env(**kw):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kw.items()}, clear=False)


class PeRjDrawOneTest(unittest.TestCase):
    def test_exactly_one_rj_per_iteration_others_in_order(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        model = _StubModel(seed=3)
        with _env(GB_PE_RJ_DRAW_ONE=1):
            for _ in range(200):
                log.clear()
                comb.propose(model, _StubState())
                rj = [n for n in log if n in RJ]
                self.assertEqual(len(rj), 1, log)
                # the drawn RJ move sits where the pair sat: after warm,
                # before ridge; everything else exactly once, in order
                self.assertEqual(log, ["rj_warm_pe", rj[0], "gb_ridge_gibbs"])

    def test_fraction_is_honoured_and_reproducible(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        picks = []
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.8):
            model = _StubModel(seed=11)
            for _ in range(2000):
                log.clear()
                comb.propose(model, _StubState())
                picks.append([n for n in log if n in RJ][0])
            frac = np.mean([p == "rj_fstat_pe" for p in picks])
            # 2000 Bernoulli(0.8) draws: sigma = 0.009 -> a 5-sigma band
            self.assertGreater(frac, 0.755)
            self.assertLess(frac, 0.845)
            # same seed -> identical sequence (the draw comes from model.random)
            log2 = []
            _, comb2 = _make(PE_NAMES, log2, weighted_cycle=True)
            model2 = _StubModel(seed=11)
            picks2 = []
            for _ in range(2000):
                log2.clear()
                comb2.propose(model2, _StubState())
                picks2.append([n for n in log2 if n in RJ][0])
            self.assertEqual(picks, picks2)

    def test_fraction_endpoints(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        model = _StubModel()
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=1.0):
            for _ in range(20):
                log.clear()
                comb.propose(model, _StubState())
                self.assertIn("rj_fstat_pe", log)
                self.assertNotIn("rj_prior_pe", log)
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.0):
            for _ in range(20):
                log.clear()
                comb.propose(model, _StubState())
                self.assertIn("rj_prior_pe", log)
                self.assertNotIn("rj_fstat_pe", log)

    def test_bad_fraction_raises(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=1.5):
            with self.assertRaises(ValueError):
                comb.propose(_StubModel(), _StubState())

    def test_accepted_sums_over_executed_moves(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        with _env(GB_PE_RJ_DRAW_ONE=1):
            _, accepted = comb.propose(_StubModel(), _StubState())
        # warm + one RJ + ridge = 3 executed moves, each accepting ones
        np.testing.assert_array_equal(accepted, 3.0 * np.ones((2, 4)))

    # ---- the mode must NOT leak ------------------------------------------------
    def test_search_stage_untouched(self):
        log = []
        names = ("rj_fstat_search", "rj_prior_removal", "rj_fstat_pe", "rj_prior_pe")
        _, comb = _make(names, log, kind="search")
        with _env(GB_PE_RJ_DRAW_ONE=1):
            comb.propose(_StubModel(), _StubState())
        self.assertEqual(log, list(names))  # sequential, all of them

    def test_knob_unset_keeps_weighted_cycle(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        with _env(GB_PE_RJ_DRAW_ONE=0):
            model = _StubModel(seed=99)
            comb.propose(model, _StubState())
        mirror = np.random.RandomState(99)
        expected = list(mirror.choice(4, size=4, replace=True, p=np.full(4, 0.25)))
        self.assertEqual(log, [PE_NAMES[k] for k in expected])

    def test_pe_stage_without_the_pair_falls_through(self):
        log = []
        names = ("rj_warm_pe", "rj_fstat_pe", "gb_ridge_gibbs")
        _, comb = _make(names, log, weighted_cycle=True)
        with _env(GB_PE_RJ_DRAW_ONE=1):
            model = _StubModel(seed=5)
            comb.propose(model, _StubState())
        mirror = np.random.RandomState(5)
        expected = list(mirror.choice(3, size=3, replace=True, p=np.full(3, 1 / 3)))
        self.assertEqual(log, [names[k] for k in expected])


if __name__ == "__main__":
    unittest.main()
