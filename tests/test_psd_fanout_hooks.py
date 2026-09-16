"""PSD fan-out hooks: shipped ladder, pooled swap tallies, once-per-propose adaptation."""

import types
import unittest

import numpy as np
from eryn.moves.tempering import TemperatureControl

from lisatools.globalfit.moves.psdmove import PSDMove
from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin

NT, B = 3, 2


def _skeleton():
    move = PSDMove.__new__(PSDMove)
    move.gf_move_name = "psd_pe"
    move.sampled_branches = ["psd", "galfor"]
    move.temperature_control = TemperatureControl(4, B, ntemps=NT, permute=False)
    move.temperature_control.gf_configured_adaptive = True
    move.temperature_control.adaptive = False
    move._tally_swaps_accepted = np.array([1, 2])
    move._tally_swaps_proposed = np.array([2 * B, 2 * B])
    move._fanout_body = False
    return move


def _state():
    subs = {k: types.SimpleNamespace(betas=np.zeros(NT)) for k in ("psd", "galfor")}
    subs["sgwb"] = None
    return types.SimpleNamespace(sub_states=subs)


class PSDFanoutHooksTest(unittest.TestCase):
    def test_mro_flags_and_body_rename(self):
        self.assertIs(PSDMove.__mro__[1], WalkerFanoutMixin)
        self.assertTrue(PSDMove.fanout_assigns_counters)
        self.assertTrue(callable(PSDMove.propose_local))
        self.assertIs(PSDMove.propose, WalkerFanoutMixin.propose)

    def test_ladder_ships_and_applies(self):
        move = _skeleton()
        move.temperature_control.betas[:] = [1.0, 0.3, 0.05]
        extra = move.fanout_payload_extra()
        other = _skeleton()
        other.fanout_apply_extra(extra)
        np.testing.assert_array_equal(other.temperature_control.betas, [1.0, 0.3, 0.05])
        self.assertEqual(move.fanout_temperature_controls(), [move.temperature_control])

    def test_reply_carries_tallies_and_none_is_zero(self):
        move = _skeleton()
        r = move.fanout_reply_extra(None)
        np.testing.assert_array_equal(r["swaps_accepted"], [1.0, 2.0])
        np.testing.assert_array_equal(r["swaps_proposed"], [4.0, 4.0])
        move._tally_swaps_accepted = None
        move._tally_swaps_proposed = None
        r = move.fanout_reply_extra(None)
        self.assertEqual(r["swaps_accepted"].size, 0)

    def test_merge_pools_adapts_once_and_publishes_the_ladder(self):
        move = _skeleton()
        tc = move.temperature_control
        betas0 = np.array(tc.betas, copy=True)
        replies = {
            0: {"swaps_accepted": np.array([1.0, 2.0]), "swaps_proposed": np.array([4.0, 4.0])},
            1: {"swaps_accepted": np.array([2.0, 0.0]), "swaps_proposed": np.array([4.0, 4.0])},
        }
        state = _state()
        move.fanout_merge_extra(replies, state)
        ref = TemperatureControl(4, B, ntemps=NT, permute=False)
        expect = betas0 + ref._get_ladder_adjustment(0, betas0.copy(), np.array([3.0, 2.0]) / 8.0)
        np.testing.assert_allclose(tc.betas, expect)
        self.assertEqual(tc.time, 1)
        for key in ("psd", "galfor"):
            np.testing.assert_allclose(state.sub_states[key].betas, expect)


if __name__ == "__main__":
    unittest.main()
