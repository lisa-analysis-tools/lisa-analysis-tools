"""GB replica identity: band ranges, rank-block attributes."""

import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase, replica_band_ranges


class BandRangesTest(unittest.TestCase):
    def test_count_split_is_contiguous_and_covers(self):
        r = replica_band_ranges(10, 3)
        self.assertEqual(r, [(0, 4), (4, 7), (7, 10)])
        self.assertEqual(replica_band_ranges(10, 1), [(0, 10)])
        self.assertEqual(replica_band_ranges(2, 3), [(0, 1), (1, 2), (2, 2)])  # empty tail range allowed

    def test_all_zero_weights_fall_back_to_the_count_split(self):
        # A branch whose weight source is empty this propose (no catalogue
        # source anywhere) must not collapse every range but the last to
        # width zero -- it degrades to the plain count split.
        self.assertEqual(
            replica_band_ranges(6, 2, weights=np.zeros(6)),
            replica_band_ranges(6, 2),
        )
        self.assertEqual(
            replica_band_ranges(7, 3, weights=np.zeros(7)),
            replica_band_ranges(7, 3),
        )

    def test_weight_split_balances_cumulative_weight(self):
        w = np.array([0, 0, 5, 5, 0, 0, 5, 5, 0, 0])
        r = replica_band_ranges(10, 2, weights=w)
        self.assertEqual(r[0][0], 0)
        self.assertEqual(r[-1][1], 10)
        self.assertEqual(r[0][1], r[1][0])
        self.assertEqual(int(w[r[0][0]:r[0][1]].sum()), 10)  # half the weight on each side

    def test_rank_block_carries_replica_identity(self):
        m = GBSpecialBase.__new__(GBSpecialBase)
        self.assertIsNone(m._owned_band_range)
        self.assertEqual((m._replica_index, m._n_replicas), (0, 1))
        m._apply_replica_payload({"band_range": (3, 7), "replica": (1, 2)})
        self.assertEqual(m._owned_band_range, (3, 7))
        self.assertEqual((m._replica_index, m._n_replicas), (1, 2))
        m._apply_replica_payload({})
        self.assertIsNone(m._owned_band_range)
        self.assertEqual((m._replica_index, m._n_replicas), (0, 1))


class OwnedRowsMaskTest(unittest.TestCase):
    def _sorter(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            band_inds=np.array([0, 1, 2, 3, 4, 5, 2, 3]),
            leaf_inds=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            inds=np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
            xp=np,
        )

    def test_none_outside_replica_mode(self):
        m = GBSpecialBase.__new__(GBSpecialBase)
        self.assertIsNone(m._owned_rows_mask(self._sorter()))
        self.assertIsNone(m._owned_band_row_mask(np.arange(6)))

    def test_owned_alive_rows_and_partitioned_dead_rows(self):
        m = GBSpecialBase.__new__(GBSpecialBase)
        m._apply_replica_payload({"band_range": (2, 4), "replica": (1, 2)})
        mask = m._owned_rows_mask(self._sorter())
        # bands 2,3: alive rows (leaf 2,3) kept; dead rows leaf 6 (band 2, 6%2==0 -> rank 0) dropped,
        # leaf 7 (band 3, 7%2==1 -> rank 1) kept; everything outside bands 2,3 dropped
        np.testing.assert_array_equal(mask, [0, 0, 1, 1, 0, 0, 0, 1])
        np.testing.assert_array_equal(m._owned_band_row_mask(np.arange(6)), [0, 0, 1, 1, 0, 0])


class TemperingOpenCloseMaskTest(unittest.TestCase):
    """``run_tempering``'s full-width cold-chain OPEN/CLOSE ``extra_bool``.

    In replica mode this rank's sorter is authoritative only for its OWN
    bands -- the ledger reconciles the RESIDUAL, never the sorter -- so a
    full-width open would inject another rank's stale templates (a source
    that died there gets added a second time). The owned-band factor is
    ``None`` outside replica mode, where the composed mask must reduce to
    the historical residue expression exactly.
    """

    def _sorter(self):
        from types import SimpleNamespace
        return SimpleNamespace(band_inds=np.arange(6), xp=np)

    def test_no_range_is_the_plain_residue_mask(self):
        m = GBSpecialBase.__new__(GBSpecialBase)
        s = self._sorter()
        for units, rem in ((2, 0), (2, 1), (3, 2)):
            mask = m._tempering_open_close_mask(s, units, rem)
            np.testing.assert_array_equal(mask, s.band_inds % units == rem)
            self.assertEqual(mask.dtype, np.bool_)

    def test_owned_range_excludes_foreign_bands(self):
        m = GBSpecialBase.__new__(GBSpecialBase)
        m._apply_replica_payload({"band_range": (2, 4), "replica": (0, 2)})
        s = self._sorter()
        # bands 0, 1, 4, 5 are another rank's: never opened, whatever their
        # residue class; inside [2, 4) the residue rule still applies
        np.testing.assert_array_equal(
            m._tempering_open_close_mask(s, 2, 1), [0, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(
            m._tempering_open_close_mask(s, 2, 0), [0, 0, 1, 0, 0, 0])
        # the union over the residue classes is exactly the owned range
        both = (m._tempering_open_close_mask(s, 2, 0)
                | m._tempering_open_close_mask(s, 2, 1))
        np.testing.assert_array_equal(both, [0, 0, 1, 1, 0, 0])


class VGBWeightsTest(unittest.TestCase):
    def test_weights_count_cold_sources_per_band(self):
        from types import SimpleNamespace
        from lisatools.globalfit.moves.gbspecialstretch import VGBSpecialStretchMove
        m = VGBSpecialStretchMove.__new__(VGBSpecialStretchMove)
        m.band_edges = np.array([0.0, 1.0, 2.0, 3.0])
        m.num_bands = 3
        m._cold_source_freqs_hz = lambda work: np.array([0.5, 1.5, 1.6, 2.5, 2.6, 2.7])
        w = m._replica_band_weights(SimpleNamespace())
        np.testing.assert_array_equal(w, [1, 2, 3])
