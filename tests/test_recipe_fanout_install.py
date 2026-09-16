"""The recipe installs the fan-out on every addremove/PSD move it builds."""

import unittest

from lisatools.globalfit import recipe as recipe_mod
from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
from lisatools.globalfit.moves.psdmove import PSDMove
from lisatools.globalfit.run import _fanout_unready_moves


class FanoutInstallTest(unittest.TestCase):
    def test_ported_families_are_not_unready(self):
        a = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
        a.gf_move_name = "mbh_pe"
        p = PSDMove.__new__(PSDMove)
        p.gf_move_name = "psd_pe"
        unready, head_only = _fanout_unready_moves([a, p])
        self.assertEqual((unready, head_only), ([], []))

    def test_builders_call_install(self):
        # the builders are exercised end to end by the gated smokes; here we pin
        # the call sites so a refactor cannot silently drop them
        src = open(recipe_mod.__file__).read()
        self.assertEqual(src.count("install_walker_fanout(curr)"), 3)


if __name__ == "__main__":
    unittest.main()
