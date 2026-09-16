"""Recipe builders size moves at the LOCAL walker block and read the block from the layout."""

import unittest

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import build_layout
from lisatools.globalfit.recipe import _local_nwalkers, _local_walker_block


class _Acs:
    def __init__(self, n):
        self.acs_total_entries = n


class _Curr:
    def __init__(self, layout=None, rank=None):
        self.rank_layout = layout
        self.rank = rank


class LocalBlockTest(unittest.TestCase):
    def test_no_layout_means_whole_range(self):
        self.assertEqual(_local_walker_block(_Curr(), _Acs(4)), (0, 4))
        self.assertEqual(_local_nwalkers(_Acs(4)), 4)

    def test_layout_gives_this_ranks_block(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(_local_walker_block(_Curr(lay, 0), _Acs(2)), (0, 2))
        self.assertEqual(_local_walker_block(_Curr(lay, 1), _Acs(2)), (2, 4))

    def test_block_must_match_the_aca_size(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        with self.assertRaises(ValueError):
            _local_walker_block(_Curr(lay, 1), _Acs(4))


if __name__ == "__main__":
    unittest.main()
