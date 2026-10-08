"""The per-band barren valve is live on the FIRST propose after a (re)launch.

``_rj_band_shutoff`` (one bool per band, True = the band is frozen for RJ of
any kind at every temperature) is persisted in the GB sub-state's
``band_info`` (``band_rj_shutoff`` + the streak / revive / epoch record). It
used to be ADOPTED only inside ``_update_band_shutoff``, which runs at
propose END of the designated move -- so the first propose of every process
read no valve (``None`` / all-False): the RJ subset filter froze nothing and
the orchestrator shipped that empty table to every compute rank. One
full-width RJ propose per relaunch in the search stages.

Fixed by adopting the record at propose START on the designated move
(``_band_shutoff_adopt_early``, called by both propose bodies before
``setup()`` and before anything reads the valve), with the same once-per-
process guard the propose-end tick keeps.
"""

import inspect
import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as gbs
from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase
from tests.test_band_shutoff_revival import BARREN, _MoveStub, _PersistState, _env
import tests.test_gb_orchestrator_merge as OM
from tests.test_gf_substate_roundtrip import NUM_BANDS

LOGGER = "lisatools.globalfit.moves.gbspecialstretch"


def _stored_record(num_bands, shut_bands):
    """A complete persisted valve record with ``shut_bands`` already off."""
    shut = np.zeros(num_bands, dtype=bool)
    shut[list(shut_bands)] = True
    return {
        "band_occ_streak": np.where(shut, 5, 0).astype(np.int64),
        "band_occ_last": np.zeros(num_bands, dtype=np.int64),
        "band_rj_shutoff": shut,
        "band_shutoff_since_revive": np.array([2], dtype=np.int64),
        "band_shutoff_epoch": np.array([0], dtype=np.int64),
    }


class _ValveMove(_MoveStub):
    """The revival tests' move stub, plus the early adopt and a switch for
    the designation rule (``_band_shutoff_enabled``)."""

    branch_name = "gb"
    designated = True

    def _band_shutoff_enabled(self):
        return self.designated

    _band_shutoff_adopt = getattr(GBSpecialBase, "_band_shutoff_adopt", None)
    _band_shutoff_adopt_early = getattr(
        GBSpecialBase, "_band_shutoff_adopt_early", None)


def _persist_state(shut_bands=(3,)):
    bi = {"initialized": True, "num_bands": 4}
    bi.update(_stored_record(4, shut_bands))
    return _PersistState(bi), bi


class EarlyAdoptTest(unittest.TestCase):
    def test_the_stored_valve_is_installed_before_any_tick(self):
        st, bi = _persist_state()
        m = _ValveMove()
        with _env(), self.assertLogs(LOGGER, "INFO") as logs:
            self.assertTrue(m._band_shutoff_adopt_early(st))
        np.testing.assert_array_equal(m._rj_band_shutoff, bi["band_rj_shutoff"])
        np.testing.assert_array_equal(m._band_occ_streak, bi["band_occ_streak"])
        self.assertTrue(m._band_shutoff_loaded)
        self.assertIn("valve state restored from the store: 1 band(s) already off",
                      "\n".join(logs.output))

    def test_the_propose_end_tick_does_not_re_adopt(self):
        """Exactly once per process: after the early adopt, the tick runs on
        the move's own record -- a store that changed underneath is not read
        again, and the "restored" line appears once."""
        st, bi = _persist_state()
        m = _ValveMove()
        with _env(), self.assertLogs(LOGGER, "INFO") as logs:
            m._band_shutoff_adopt_early(st)
            bi["band_rj_shutoff"][:] = False          # the store moved
            m._update_band_shutoff(BARREN, st)
            self.assertFalse(m._band_shutoff_adopt_early(st))
        self.assertTrue(m._rj_band_shutoff[3])        # still the adopted valve
        restored = [l for l in logs.output if "valve state restored" in l]
        self.assertEqual(len(restored), 1)

    def test_a_new_process_adopts_again(self):
        st, bi = _persist_state()
        with _env():
            _ValveMove()._band_shutoff_adopt_early(st)
            m2 = _ValveMove()
            self.assertTrue(m2._band_shutoff_adopt_early(st))
        self.assertTrue(m2._rj_band_shutoff[3])

    def test_a_non_designated_move_does_not_adopt(self):
        st, _bi = _persist_state()
        m = _ValveMove()
        m.designated = False
        with _env():
            self.assertFalse(m._band_shutoff_adopt_early(st))
        self.assertNotIn("_rj_band_shutoff", vars(m))
        self.assertNotIn("_band_shutoff_loaded", vars(m))

    def test_a_state_without_band_info_keeps_todays_behaviour(self):
        m = _ValveMove()
        with _env():
            for st in (None, _PersistState(None)):
                self.assertFalse(m._band_shutoff_adopt_early(st))
            self.assertNotIn("_rj_band_shutoff", vars(m))
            self.assertNotIn("_band_shutoff_loaded", vars(m))
            m._update_band_shutoff(BARREN, None)          # in-memory clock
        self.assertFalse(m._rj_band_shutoff.any())
        self.assertEqual(m._band_shutoff_origin, "memory")

    def test_a_disabled_valve_is_not_adopted_early(self):
        """``GB_RJ_BAND_SHUTOFF_ITERS <= 0``: unchanged -- the first propose
        freezes nothing, and the propose-end tick adopts the record and
        persists its release."""
        st, bi = _persist_state()
        m = _ValveMove()
        with _env(GB_RJ_BAND_SHUTOFF_ITERS="0"):
            self.assertFalse(m._band_shutoff_adopt_early(st))
            self.assertNotIn("_rj_band_shutoff", vars(m))
            m._update_band_shutoff(BARREN, st)
        self.assertFalse(m._rj_band_shutoff.any())
        self.assertFalse(bi["band_rj_shutoff"].any())


class FirstProposeShipsTheStoredValveTest(unittest.TestCase):
    """Orchestrator level: what the head SHIPS to every compute rank on the
    first propose of a process (command 1, long before the propose-end
    tick), with a state whose store already has bands off."""

    def setUp(self):
        if gbs.cp is not np:  # pragma: no cover - GPU box
            self.skipTest("the orchestrator harness runs on the host array module")
        self.state = OM.make_state(np.random.default_rng(7))
        self.state.sub_states["gb"].band_info.update(
            _stored_record(NUM_BANDS, (1, 4)))
        self.stored = np.array(
            self.state.sub_states["gb"].band_info["band_rj_shutoff"], copy=True)
        self._census = dict(gbs.GBSpecialBase._branch_propose_counts)
        self.addCleanup(self._restore_census)
        p = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        p.start()
        self.addCleanup(p.stop)

    def _restore_census(self):
        gbs.GBSpecialBase._branch_propose_counts.clear()
        gbs.GBSpecialBase._branch_propose_counts.update(self._census)

    def _run(self, designated=True):
        real = OM.make_move

        def fresh_process_move(*a, **k):
            move = real(*a, **k)
            del move._rj_band_shutoff                 # nothing in memory yet
            move._band_shutoff_enabled = lambda: designated
            return move

        with mock.patch.object(OM, "make_move", fresh_process_move), \
                mock.patch.dict(os.environ, {"GB_RJ_BAND_SHUTOFF_ITERS": "2"}):
            return OM.run_propose(self.state)

    def test_the_first_propose_ships_the_stored_valve(self):
        (_new, _acc), moves = self._run()
        for rank, move in moves.items():
            shipped = move.payloads[0]["tables"]["rj_band_shutoff"]
            self.assertIsNotNone(shipped, f"rank {rank} got no valve")
            np.testing.assert_array_equal(shipped, self.stored)
        # shipped BEFORE the propose-end tick ran (it is stubbed to record)
        self.assertEqual(len(moves[0].shutoff_calls), 1)

    def test_a_non_designated_move_ships_no_valve(self):
        (_new, _acc), moves = self._run(designated=False)
        for move in moves.values():
            self.assertIsNone(move.payloads[0]["tables"]["rj_band_shutoff"])
        self.assertNotIn("_band_shutoff_loaded", vars(moves[0]))


class EarlyAdoptWiringTest(unittest.TestCase):
    """Both propose bodies adopt before ``setup()`` (an epoch install there
    syncs against the ADOPTED epoch) and before anything reads the valve."""

    def _order(self, fn, *needles):
        src = inspect.getsource(fn)
        pos = [src.index(n) for n in needles]
        self.assertEqual(pos, sorted(pos), needles)

    def test_the_legacy_body(self):
        self._order(GBSpecialBase._propose_legacy,
                    "self._band_shutoff_adopt_early(state)",
                    "self.setup(model, state.branches)",
                    "self.run_proposal(")

    def test_the_orchestrator_head(self):
        self._order(GBSpecialBase._propose_orchestrated,
                    "self._band_shutoff_adopt_early(state)",
                    "self.setup(model, state.branches)",
                    '"rj_band_shutoff": self._ro_table(')

    def test_the_propose_end_tick_uses_the_same_adopt(self):
        src = inspect.getsource(GBSpecialBase._update_band_shutoff)
        self.assertIn("self._band_shutoff_adopt(state)", src)


if __name__ == "__main__":
    unittest.main()
