"""``scripts/fstat_proposal/repair_recipe_group.py`` rebuilds a torn recipe.

3-month job 666 died at resume on a malformed object header in
``global_fit/recipe``, and the running backup copy was a BYTE copy of the
same damage, so there was nothing to restore from. The group is five tiny
groups holding two attrs each and its correct contents are known, so the
recovery is to REBUILD it -- which costs zero stored iterations.

These pin the two properties that make the script safe to hand someone at
03:00: it never modifies the input, and its verify step reads the group
back exactly the way ``GFHDFBackend.add_recipe`` will.
"""
import importlib.util
import os
import tempfile
import unittest

import h5py
import numpy as np

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "fstat_proposal", "repair_recipe_group.py")


def _load():
    spec = importlib.util.spec_from_file_location("_repair", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RepairRecipeGroupTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.p = os.path.join(self.d.name, "gf_prod_3mo_testing.h5")
        with h5py.File(self.p, "w") as f:
            g = f.create_group("global_fit")
            g.attrs["iteration"] = 116
            g.attrs["has_recipe"] = True
            g.create_dataset("log_like", data=np.full((200, 1, 1, 4), 5.2679e7))
            r = g.create_group("recipe")
            # a WRONG recipe, so a no-op would be visible
            for i, n in enumerate(["noise_search", "gb_search_1"], start=1):
                s = r.create_group(n)
                s.attrs["status"] = False
                s.attrs["order num"] = i

    def test_rebuild_writes_the_known_good_recipe(self):
        self.mod.rebuild(self.p)
        it, steps, ll = self.mod.verify(self.p)
        self.assertEqual(it, 116)
        self.assertEqual(
            [(n, s) for n, s, _ in steps],
            [("noise_search", True), ("gb_search_1", False),
             ("gb_search_2", False), ("gb_search_3", False),
             ("full_pe", False)])
        # noise_search is the only step this run ever completed, at 24
        self.assertEqual(dict((n, c) for n, _, c in steps)["noise_search"], 24)
        self.assertTrue(all(c is None for n, _, c in steps
                            if n != "noise_search"))

    def test_order_num_matches_what_add_recipe_asserts(self):
        """``add_recipe`` asserts order num == index + 1; a mismatch here
        would reproduce the original crash in a new form."""
        self.mod.rebuild(self.p)
        with h5py.File(self.p, "r") as f:
            grp = f["global_fit"]["recipe"]
            for i, (name, _) in enumerate(self.mod.RECIPE, start=1):
                self.assertEqual(int(grp[name].attrs["order num"]), i)
            self.assertTrue(bool(f["global_fit"].attrs["has_recipe"]))

    def test_verify_rejects_a_recipe_that_was_not_rebuilt(self):
        """The verify step must actually be able to fail, or it is theatre."""
        with self.assertRaises(AssertionError):
            self.mod.verify(self.p)          # still the wrong 2-step recipe

    def test_the_rest_of_the_store_survives(self):
        self.mod.rebuild(self.p)
        with h5py.File(self.p, "r") as f:
            self.assertEqual(f["global_fit"]["log_like"].shape, (200, 1, 1, 4))
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 116)


if __name__ == "__main__":
    unittest.main()
