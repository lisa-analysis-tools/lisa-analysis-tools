"""A rewind must set the mid-iteration checkpoint aside.

6mo jobs 671/672 (2026-09-30): the store was rewound to 47 but job 670's
checkpoint said 48, and the resume rule "the checkpoint wins iff it is at or
after the store's iteration" -- correct for a preempted save -- adopted the
discarded future: both relaunches started from 670's mid-iteration state,
not from the rewound row. The rewind tool now moves the checkpoint aside.
"""

import os
import sys
import tempfile
import unittest

import h5py

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scripts", "fstat_proposal"))

import reset_recipe_stage as R  # noqa: E402


def _store(tmp, iteration=5):
    path = os.path.join(tmp, "s_testing.h5")
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = iteration
        r = g.create_group("recipe")
        for i, n in enumerate(("a", "b")):
            s = r.create_group(n)
            s.attrs["status"] = True
            s.attrs["order num"] = i
    return path


class RewindMovesTheCheckpointAsideTest(unittest.TestCase):

    def test_apply_with_a_rewind_renames_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            ckpt = os.path.join(tmp, "s_testing_midit_checkpoint.pkl")
            with open(ckpt, "wb") as fh:
                fh.write(b"x")
            rc = R.main([store, "b", "--iteration", "3", "--apply"])
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(ckpt))
            moved = [n for n in os.listdir(tmp) if n.startswith("s_testing_midit_checkpoint.pkl.rewound")]
            self.assertEqual(len(moved), 1, os.listdir(tmp))
            with h5py.File(store, "r") as f:
                self.assertEqual(int(f["global_fit"].attrs["iteration"]), 3)

    def test_dry_run_leaves_the_checkpoint_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            ckpt = os.path.join(tmp, "s_testing_midit_checkpoint.pkl")
            with open(ckpt, "wb") as fh:
                fh.write(b"x")
            R.main([store, "b", "--iteration", "3"])
            self.assertTrue(os.path.exists(ckpt))

    def test_reopen_without_a_rewind_keeps_the_checkpoint(self):
        """Re-opening a stage without moving the counter keeps the chain where
        it is, so a checkpoint at the store's iteration is still valid."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            ckpt = os.path.join(tmp, "s_testing_midit_checkpoint.pkl")
            with open(ckpt, "wb") as fh:
                fh.write(b"x")
            R.main([store, "b", "--apply"])
            self.assertTrue(os.path.exists(ckpt))


if __name__ == "__main__":
    unittest.main()
