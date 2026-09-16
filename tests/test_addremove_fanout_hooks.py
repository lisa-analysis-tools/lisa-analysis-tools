"""addremove fan-out hooks: swap tallies per leaf, pooled once-per-propose ladder step."""

import types
import unittest

import numpy as np
from eryn.moves.tempering import TemperatureControl

from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin

NT, B = 3, 2


def _skeleton(nleaves=2):
    """A move object with only the attributes the hooks touch (no waveform build)."""
    move = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name = "mbh"
    move.gf_move_name = "mbh_pe"
    move.nwalkers = B
    move.ntemps = NT
    move.temperature_controls = [
        TemperatureControl(2, B, ntemps=NT, permute=False) for _ in range(nleaves)
    ]
    for tc in move.temperature_controls:
        tc.gf_configured_adaptive = True
        tc.adaptive = False
    move._fanout_swap_tally = {}
    move._fanout_body = False
    move._fancy_swap_clock = 4
    move._dbg_step = 9
    return move


def _state(betas_all):
    sub = types.SimpleNamespace(betas_all=[np.array(b, dtype=float) for b in betas_all])
    return types.SimpleNamespace(sub_states={"mbh": sub})


class AddRemoveFanoutHooksTest(unittest.TestCase):
    def test_mro_and_body_rename(self):
        self.assertIs(ResidualAddOneRemoveOneMove.__mro__[1], WalkerFanoutMixin)
        self.assertTrue(callable(ResidualAddOneRemoveOneMove.propose_local))
        self.assertIs(
            ResidualAddOneRemoveOneMove.propose, WalkerFanoutMixin.propose
        )

    def test_note_swaps_accumulates_over_repeats(self):
        move = _skeleton()
        tc = move.temperature_controls[1]
        tc.swaps_accepted = np.array([1, 0])
        move._fanout_note_swaps(1, tc)
        tc.swaps_accepted = np.array([0, 2])
        move._fanout_note_swaps(1, tc)
        acc, prop = move._fanout_swap_tally[1]
        np.testing.assert_array_equal(acc, [1.0, 2.0])
        # Eryn holds swaps_proposed = nwalkers per rung on every call
        np.testing.assert_array_equal(prop, [2.0 * B, 2.0 * B])
        self.assertEqual(move.fanout_reply_extra(None)["swap_tally"].keys(), {1})

    def test_clock_round_trip(self):
        move = _skeleton()
        extra = move.fanout_payload_extra()
        self.assertEqual(extra, {"fancy_swap_clock": 4, "dbg_step": 9})
        other = _skeleton()
        other.fanout_apply_extra(extra)
        self.assertEqual((other._fancy_swap_clock, other._dbg_step), (4, 9))

    def test_merge_pools_tallies_and_adapts_each_visited_leaf_once(self):
        move = _skeleton()
        betas0 = [np.array([1.0, 0.5, 0.25]), np.array([1.0, 0.4, 0.1])]
        state = _state(betas0)
        r0 = {"swap_tally": {0: (np.array([2.0, 1.0]), np.array([4.0, 4.0]))}}
        r1 = {"swap_tally": {0: (np.array([1.0, 1.0]), np.array([4.0, 4.0]))}}
        move.fanout_merge_extra({0: r0, 1: r1}, state)
        tc0 = move.temperature_controls[0]
        ref = TemperatureControl(2, B, ntemps=NT, permute=False)
        expect = betas0[0] + ref._get_ladder_adjustment(
            0, betas0[0].copy(), np.array([3.0, 2.0]) / 8.0
        )
        np.testing.assert_allclose(state.sub_states["mbh"].betas_all[0], expect)
        np.testing.assert_allclose(tc0.betas, expect)
        self.assertEqual(tc0.time, 1)
        # leaf 1 was visited on no rank: ladder and clock untouched
        np.testing.assert_array_equal(state.sub_states["mbh"].betas_all[1], betas0[1])
        self.assertEqual(move.temperature_controls[1].time, 0)

    def test_hooks_lists(self):
        move = _skeleton(3)
        self.assertEqual(len(move.fanout_temperature_controls()), 3)


if __name__ == "__main__":
    unittest.main()
