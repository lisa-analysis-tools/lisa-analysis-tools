"""The public F-stat reference row holder + the host row snapshot.

CPU-only, no ACA: a stand-in parent exposing exactly the attributes the
extraction reads (``linear_data_arr`` / ``linear_psd_arr`` flat buffers,
``acs_total_entries``, ``xp``, ``nchannels``, ``shape_sens``). The point is
the ROW ARITHMETIC -- one walker's residual row and inverse-PSD row pulled
out of the flat per-shard buffers -- which is what the fan-out ships.
"""

import unittest

import numpy as np

from lisatools.globalfit.moves import gbbands


class _FakeParent:
    """Minimal single-shard holder: B rows in one flat buffer per kind."""

    def __init__(self, n_rows, data_row_size, psd_row_size):
        self.acs_total_entries = int(n_rows)
        self.nchannels = 3
        self.shape_sens = (3, 3)
        self.device = None
        self.gpus = None
        rng = np.random.default_rng(3)
        self.linear_data_arr = [
            rng.normal(size=int(n_rows) * int(data_row_size))]
        self.linear_psd_arr = [
            rng.normal(size=int(n_rows) * int(psd_row_size))]
        self.psd_row_index = None

    @property
    def xp(self):
        return np


class FStatRefRowHolderTest(unittest.TestCase):
    def test_is_public(self):
        self.assertTrue(hasattr(gbbands, "FStatRefRowHolder"))
        self.assertIs(gbbands._FStatRefRowHolder, gbbands.FStatRefRowHolder)

    def test_single_slab_shape_and_delegation(self):
        parent = _FakeParent(4, 12, 36)
        holder = gbbands.FStatRefRowHolder(
            parent, None, np.zeros(12), np.zeros(36))
        self.assertEqual(len(holder), 1)
        self.assertEqual(holder.acs_total_entries, 1)
        self.assertEqual(len(holder.linear_data_arr), 1)
        self.assertIsNone(holder.gpus)
        self.assertIs(holder.xp, np)
        self.assertEqual(holder.nchannels, 3)  # delegated to the parent

    def test_device_sets_the_single_entry_gpu_list(self):
        parent = _FakeParent(2, 4, 4)
        holder = gbbands.FStatRefRowHolder(
            parent, 1, np.zeros(4), np.zeros(4))
        self.assertEqual(holder.gpus, [1])
        self.assertEqual(holder.device, 1)

    def test_underscore_attributes_are_not_delegated(self):
        parent = _FakeParent(2, 4, 4)
        holder = gbbands.FStatRefRowHolder(
            parent, None, np.zeros(4), np.zeros(4))
        with self.assertRaises(AttributeError):
            holder._not_a_real_attribute  # noqa: B018


class SnapshotRefRowsTest(unittest.TestCase):
    def test_pulls_exactly_one_walkers_rows(self):
        n_rows, drow, prow = 4, 12, 36
        parent = _FakeParent(n_rows, drow, prow)
        for row in range(n_rows):
            d, p = gbbands.snapshot_ref_rows(
                parent, parent, row, row, xp=np, device=None)
            np.testing.assert_array_equal(
                d, np.asarray(parent.linear_data_arr[0]).reshape(
                    n_rows, -1)[row])
            np.testing.assert_array_equal(
                p, np.asarray(parent.linear_psd_arr[0]).reshape(
                    n_rows, -1)[row])
            self.assertTrue(d.flags["C_CONTIGUOUS"])
            self.assertTrue(p.flags["C_CONTIGUOUS"])

    def test_the_rows_are_copies_not_views_of_the_live_buffer(self):
        """A "snapshot" that aliases the live residual is not a snapshot.

        On the CPU backend ``asnumpy`` is the identity and a row slice of a
        C-contiguous reshape is already contiguous, so the old
        ``np.ascontiguousarray`` returned THAT VERY VIEW. The GB-free window
        closes immediately after this call by design, so its restore would
        have undone the snapshot -- the owner's holder would then disagree
        with every worker's (they receive a genuine ``Bcast`` copy) and the
        epoch would be fitted against a residual nobody chose.
        """
        parent = _FakeParent(3, 8, 8)
        parent.linear_data_arr[0].reshape(3, -1)[1] += 1.0
        parent.linear_psd_arr[0].reshape(3, -1)[1] += 2.0
        d, p = gbbands.snapshot_ref_rows(parent, parent, 1, 1, xp=np)
        d_seen, p_seen = np.array(d, copy=True), np.array(p, copy=True)
        # the window closing: the live rows go back to what they were
        parent.linear_data_arr[0].reshape(3, -1)[1] -= 1.0
        parent.linear_psd_arr[0].reshape(3, -1)[1] -= 2.0
        np.testing.assert_array_equal(d, d_seen)
        np.testing.assert_array_equal(p, p_seen)

    def test_the_snapshot_feeds_a_holder_that_scores_row_zero(self):
        parent = _FakeParent(3, 8, 8)
        d, p = gbbands.snapshot_ref_rows(parent, parent, 2, 2, xp=np)
        holder = gbbands.FStatRefRowHolder(parent, None, d, p)
        np.testing.assert_array_equal(
            np.asarray(holder.linear_data_arr[0]).reshape(1, -1)[0], d)
        np.testing.assert_array_equal(
            np.asarray(holder.linear_psd_arr[0]).reshape(1, -1)[0], p)


if __name__ == "__main__":
    unittest.main()
