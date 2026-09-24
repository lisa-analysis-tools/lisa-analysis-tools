"""``migrate_gb_cap_grid`` and the PER-WALKER cap family (2026-09-22).

The script rebuilds the cap-cell state when ``GB_CAP_DIVISOR`` changes,
splitting each band's stored cap into its ``k`` children. The per-walker
twins (``cap_cell_leaf_cap_w`` and friends) live on the same cell grid, so
a migration that left them at the OLD cell count would hand the resume
guard a cap table that does not cover the grid -- caught, but only as a
refusal the operator then has to diagnose.

Migrated ONLY when already present: a store that never ran with
``GB_LEAF_CAP_PER_WALKER`` must not acquire the arrays (and the datasets
they imply) from a cap-grid migration.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "fstat_proposal"))

import migrate_gb_cap_grid as mig  # noqa: E402


class SeedPerWalkerCellsTest(unittest.TestCase):
    """The pure step: a per-cell array broadcast onto the walker axis."""

    def test_every_walker_inherits_the_cell_row(self):
        cells = np.array([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0]])          # (nsteps, nc)
        out = mig.seed_per_walker_cells(cells, 3)
        self.assertEqual(out.shape, (2, 3, 4))
        for w in range(3):
            np.testing.assert_array_equal(out[:, w, :], cells)

    def test_one_walker_is_a_bare_axis_insert(self):
        cells = np.array([[1.0, 2.0]])
        out = mig.seed_per_walker_cells(cells, 1)
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_array_equal(out[:, 0, :], cells)

    def test_the_result_does_not_alias_the_source(self):
        """A broadcast VIEW would make all walkers move together."""
        cells = np.zeros((1, 3))
        out = mig.seed_per_walker_cells(cells, 2)
        out[0, 0, 0] = 7.0
        self.assertEqual(float(out[0, 1, 0]), 0.0,
                         "walkers share memory: this is a broadcast view")
        self.assertEqual(float(cells[0, 0]), 0.0,
                         "the seed aliases the source array")


class PerWalkerSpecTest(unittest.TestCase):
    """The names the migration knows about."""

    def test_the_per_walker_family_is_enumerated(self):
        for name in ("cap_cell_leaf_cap_w", "cap_cell_iters_w",
                     "cap_cell_best_ll_w"):
            self.assertIn(name, mig.CAP_CELL_PER_WALKER_SPEC, name)

    def test_each_maps_to_its_shared_twin(self):
        self.assertEqual(
            mig.CAP_CELL_PER_WALKER_SPEC["cap_cell_leaf_cap_w"],
            "cap_cell_leaf_cap")
        self.assertEqual(
            mig.CAP_CELL_PER_WALKER_SPEC["cap_cell_iters_w"],
            "cap_cell_iters")
        self.assertEqual(
            mig.CAP_CELL_PER_WALKER_SPEC["cap_cell_best_ll_w"],
            "cap_cell_best_ll")

    def test_band_best_ll_w_is_not_migrated(self):
        """It is per BAND, so a cap-grid change does not touch it."""
        self.assertNotIn("band_best_ll_w", mig.CAP_CELL_PER_WALKER_SPEC)


if __name__ == "__main__":
    unittest.main()
