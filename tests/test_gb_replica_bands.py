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


class GBBandWeightsTest(unittest.TestCase):
    """GB's own source-weighted band split.

    Until 2026-09-23 ``GBSpecialBase._replica_band_weights`` returned
    ``None``, so ``replica_band_ranges`` split the band grid by COUNT. The
    galaxy is not uniform in frequency, so an equal-count split hands the
    low-frequency replica most of the work; VGB already overrode this
    correctly and GB now shares the same implementation.
    """

    @staticmethod
    def _move(nwalkers=1):
        m = GBSpecialBase.__new__(GBSpecialBase)
        m.band_edges = np.array([0.0, 1.0, 2.0, 3.0])
        m.num_bands = 3
        m._f0_col = 1  # sampling-basis f0 column (mHz)
        return m

    @staticmethod
    def _work(f0_mhz_per_walker, ntemps=2):
        """``work`` with f0 at sampling column 1, alive rows only where given."""
        from types import SimpleNamespace
        nw = len(f0_mhz_per_walker)
        nleaves = max(len(f) for f in f0_mhz_per_walker)
        coords = np.zeros((ntemps, nw, nleaves, 4))
        inds = np.zeros((ntemps, nw, nleaves), dtype=bool)
        for w, f0 in enumerate(f0_mhz_per_walker):
            coords[:, w, : len(f0), 1] = np.asarray(f0, dtype=float)
            inds[:, w, : len(f0)] = True
        return SimpleNamespace(coords=coords, inds=inds)

    def test_base_no_longer_returns_none(self):
        # The regression guard: a stub here silently halves dispersal.
        m = self._move()
        w = m._replica_band_weights(self._work([[500.0, 1500.0]]))
        self.assertIsNotNone(w)

    def test_counts_alive_cold_sources_per_band(self):
        # mHz -> Hz: 500 mHz lands in band 0 of edges [0, 1, 2, 3] Hz.
        m = self._move()
        w = m._replica_band_weights(
            self._work([[500.0, 1500.0, 1600.0, 2500.0, 2600.0, 2700.0]]))
        np.testing.assert_array_equal(w, [1, 2, 3])

    def test_sums_over_every_walker_of_the_block(self):
        # A block wider than one walker must weight the whole block, not
        # walker 0 alone -- otherwise B>1 splits balance the wrong load.
        m = self._move()
        w = m._replica_band_weights(self._work([[500.0], [1500.0, 2500.0]]))
        np.testing.assert_array_equal(w, [1, 1, 1])

    def test_ignores_dead_rows_and_hot_rungs(self):
        from types import SimpleNamespace
        m = self._move()
        coords = np.zeros((2, 1, 3, 4))
        coords[0, 0, :, 1] = [500.0, 1500.0, 2500.0]   # cold rung
        coords[1, 0, :, 1] = [2500.0, 2500.0, 2500.0]  # hot rung, must not count
        inds = np.zeros((2, 1, 3), dtype=bool)
        inds[0, 0, :2] = True   # only the first two cold rows are alive
        inds[1, 0, :] = True
        w = m._replica_band_weights(SimpleNamespace(coords=coords, inds=inds))
        np.testing.assert_array_equal(w, [1, 1, 0])

    def test_empty_weights_degrade_to_the_count_split(self):
        m = self._move()
        w = m._replica_band_weights(self._work([[]], ntemps=1))
        self.assertEqual(
            replica_band_ranges(3, 2, weights=w), replica_band_ranges(3, 2))

    def test_weighted_split_beats_the_count_split_on_a_skewed_galaxy(self):
        # The real shape: most sources in the low-frequency bands, spread
        # over many of them. The count split gives replica 0 nine times
        # replica 1's load; the weighted split nearly equalizes it.
        #
        # (A skew concentrated in ONE band is deliberately not the fixture:
        # a band is atomic, so no contiguous split can balance it and the
        # weighting correctly makes no difference.)
        m = GBSpecialBase.__new__(GBSpecialBase)
        m.band_edges = np.arange(31, dtype=float)
        m.num_bands = 30
        m._f0_col = 1
        f0_mhz = np.concatenate([
            np.repeat(np.arange(0, 10) + 0.5, 9) * 1e3,    # 90 over bands 0-9
            np.repeat(np.arange(20, 30) + 0.5, 1) * 1e3,   # 10 over bands 20-29
        ])
        w = m._replica_band_weights(self._work([f0_mhz]))
        self.assertEqual(int(w.sum()), 100)

        def load(ranges):
            return [float(w[a:b].sum()) for a, b in ranges]

        by_count = load(replica_band_ranges(30, 2))
        by_weight = load(replica_band_ranges(30, 2, weights=w))
        self.assertEqual(by_count, [90.0, 10.0])          # 9:1 today
        self.assertLessEqual(max(by_weight) - min(by_weight), 10.0)
        # every replica keeps a contiguous, covering range
        r = replica_band_ranges(30, 2, weights=w)
        self.assertEqual(r[0][0], 0)
        self.assertEqual(r[-1][1], 30)
        self.assertEqual(r[0][1], r[1][0])

    def test_weighted_split_scales_to_many_replicas(self):
        m = GBSpecialBase.__new__(GBSpecialBase)
        m.band_edges = np.arange(31, dtype=float)
        m.num_bands = 30
        m._f0_col = 1
        f0_mhz = np.repeat(np.arange(0, 10) + 0.5, 8) * 1e3  # 80, bands 0-9
        w = m._replica_band_weights(self._work([f0_mhz]))
        for n in (2, 4, 8):
            loads = [float(w[a:b].sum())
                     for a, b in replica_band_ranges(30, n, weights=w)]
            self.assertAlmostEqual(sum(loads), 80.0)
            # never worse than the count split it replaces
            count_loads = [float(w[a:b].sum())
                           for a, b in replica_band_ranges(30, n)]
            self.assertLessEqual(max(loads), max(count_loads))
