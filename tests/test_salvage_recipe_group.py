"""salvage_recipe_group.py on a store with a REALLY torn step header.

The fixture overwrites the first message header of
``global_fit/recipe/replica_pe``'s object header in the file bytes, the same
class of damage the 6mo store took on 2026-10-04 (HDF5: "message not
aligned"). Every other object must come across bit-identical in layout,
attrs and last row; the torn object must never be opened; the original
must not be modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "salvage_recipe_group",
    os.path.join(ROOT, "scripts", "fstat_proposal", "salvage_recipe_group.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

ORDER = ["gb_search_seed", "gb_search_1", "gb_search_2", "gb_search_3",
         "replica_pe", "full_pe"]


def _md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _make_store(path, it=7):
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        f.attrs["file_attr"] = "x"
        g = f.create_group("global_fit")
        g.attrs["iteration"] = it
        g.attrs["has_recipe"] = True
        g.attrs["branch_names"] = np.array([b"gb", b"psd"])
        ll = g.create_dataset("log_like", shape=(10, 1, 1, 4), maxshape=(None, 1, 1, 4),
                              dtype="f8", chunks=(1, 1, 1, 4), compression="gzip",
                              compression_opts=4)
        ll[:it] = rng.normal(size=(it, 1, 1, 4))
        ch = g.create_group("chain")
        c = ch.create_dataset("gb", shape=(10, 1, 1, 4, 5, 9), maxshape=(None, 1, 1, 4, 5, 9),
                              dtype="f8", chunks=(1, 1, 1, 4, 5, 9), compression="gzip")
        c[:it] = rng.normal(size=(it, 1, 1, 4, 5, 9))
        c.attrs["units"] = "phys"
        sb = g.create_group("sub_backend").create_group("gb")
        sb.create_dataset("band_edges", data=np.linspace(0, 1, 9))
        rec = g.create_group("recipe")
        for i, name in enumerate(ORDER, start=1):
            s = rec.create_group(name)
            s.attrs["order num"] = i
            s.attrs["status"] = i <= 4
        rec["gb_search_3"].attrs["completed_iteration"] = 4
        rec["gb_search_3"].attrs["move_order"] = '["a", "b"]'
        rec["full_pe"].attrs["start_iteration"] = 6
        rec["replica_pe"].attrs["completed_iteration"] = 6
        addr = h5py.h5o.get_info(rec["replica_pe"].id).addr
    # TEAR the replica_pe header: garbage over its first message header.
    with open(path, "r+b") as fh:
        fh.seek(addr + 16)
        fh.write(b"\x0c\x00\x05\x00\xff\x00\x00\x00")
    return path


class SalvageTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self.store = _make_store(os.path.join(self.d, "gf_prod_6mo_testing.h5"))

    def test_the_fixture_is_really_torn(self):
        with h5py.File(self.store, "r") as f:
            with self.assertRaises(Exception):
                dict(f["global_fit/recipe/replica_pe"].attrs)
            dict(f["global_fit/recipe/full_pe"].attrs)          # neighbours fine

    def test_NEGATIVE_CONTROL_a_wholesale_copy_hits_the_torn_header(self):
        """Why the tool copies around the recipe group instead."""
        with h5py.File(self.store, "r") as src, \
                h5py.File(os.path.join(self.d, "whole.h5"), "w") as dst:
            with self.assertRaises(Exception):
                src.copy(src["global_fit"], dst, name="global_fit")

    def test_salvage_rebuilds_the_recipe_and_keeps_everything_else(self):
        before = _md5(self.store)
        rc = S.main([self.store, "--order", ",".join(ORDER),
                     "--set", "replica_pe:status=true,completed_iteration=6",
                     "--set", "full_pe:status=false,start_iteration=6", "--apply"])
        self.assertEqual(rc, 0)
        self.assertEqual(_md5(self.store), before, "the original was modified")
        out = self.store[:-3] + "_salvaged.h5"
        with h5py.File(self.store, "r") as a, h5py.File(out, "r") as b:
            np.testing.assert_array_equal(a["global_fit/log_like"][:7],
                                          b["global_fit/log_like"][:7])
            np.testing.assert_array_equal(a["global_fit/chain/gb"][:7],
                                          b["global_fit/chain/gb"][:7])
            self.assertEqual(b["global_fit/log_like"].maxshape[0], None)
            self.assertEqual(b["global_fit/chain/gb"].compression, "gzip")
            self.assertEqual(b.attrs["file_attr"], "x")
            rec = b["global_fit/recipe"]
            self.assertEqual(sorted(rec.keys()), sorted(ORDER))
            for i, name in enumerate(ORDER, start=1):
                self.assertEqual(int(rec[name].attrs["order num"]), i)
            self.assertTrue(bool(rec["replica_pe"].attrs["status"]))
            self.assertEqual(int(rec["replica_pe"].attrs["completed_iteration"]), 6)
            self.assertFalse(bool(rec["full_pe"].attrs["status"]))
            self.assertEqual(int(rec["full_pe"].attrs["start_iteration"]), 6)
            self.assertEqual(rec["gb_search_3"].attrs["move_order"], '["a", "b"]')
            seen = []
            b.visititems(lambda n, o: seen.append(n))      # the snapshot's walk
            self.assertIn("global_fit/recipe/replica_pe", seen)

    def test_dry_run_writes_nothing(self):
        rc = S.main([self.store, "--order", ",".join(ORDER),
                     "--set", "replica_pe:status=true"])
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(self.store[:-3] + "_salvaged.h5"))

    def test_a_torn_step_without_a_status_patch_is_refused(self):
        with self.assertRaises(ValueError):
            S.main([self.store, "--order", ",".join(ORDER)])

    def test_steps_missing_from_order_are_refused(self):
        rc = S.main([self.store, "--order", ",".join(ORDER[:-1]),
                     "--set", "replica_pe:status=true"])
        self.assertEqual(rc, 2)

    def test_verify_catches_a_changed_row(self):
        S.main([self.store, "--order", ",".join(ORDER),
                "--set", "replica_pe:status=true", "--apply"])
        out = self.store[:-3] + "_salvaged.h5"
        with h5py.File(out, "a") as b:
            b["global_fit/log_like"][6] = 0.0
        plan = S.plan_recipe(S.read_steps(self.store, ORDER)[0], ORDER,
                             S.parse_sets(["replica_pe:status=true"]))
        probs = S.verify(self.store, out, plan)
        self.assertTrue(any("log_like[6]" in p for p in probs), probs)


if __name__ == "__main__":
    unittest.main()
