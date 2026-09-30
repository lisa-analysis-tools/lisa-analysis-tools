"""The snapshot extract must carry variable-length STRING datasets.

Search legs (2026-09-30) add ``global_fit/saved_after``, a resizable utf-8
string dataset (the leg-ender each row was saved after). The extract copies
every dataset generically with ``create_dataset(data=...)``; an object
(vlen) array has no native HDF5 type unless the dtype is passed through, so
without care the FIRST legged snapshot would fail with "Object dtype has no
native HDF5 equivalent" and the tar would not be built.
"""

import os
import tempfile
import unittest

import h5py
import numpy as np

from lisatools.globalfit.monitor._store_extract import extract


class ExtractCarriesStringDatasetsTest(unittest.TestCase):

    def _src(self, tmp):
        path = os.path.join(tmp, "s_testing.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("global_fit")
            g.attrs["iteration"] = 3
            g.create_dataset("log_like", data=np.arange(12.0).reshape(3, 1, 1, 4))
            ds = g.create_dataset("saved_after", shape=(3,), maxshape=(None,),
                                  dtype=h5py.string_dtype(encoding="utf-8"))
            ds[:] = ["", "in_model", "in_model_fstat"]
            r = g.create_group("recipe")
            s = r.create_group("gb_search_3")
            s.attrs["move_order"] = '["a", "in_model"]'
        return path

    def test_vlen_strings_survive_the_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._src(tmp)
            dst = os.path.join(tmp, "s_testing_extract.h5")
            extract(src, dst, keep=2)
            with h5py.File(dst, "r") as f:
                g = f["global_fit"]
                got = [v.decode() if isinstance(v, bytes) else str(v)
                       for v in g["saved_after"][...]]
                self.assertEqual(got, ["", "in_model", "in_model_fstat"])
                self.assertEqual(g["recipe/gb_search_3"].attrs["move_order"],
                                 '["a", "in_model"]')
                np.testing.assert_array_equal(g["log_like"][...],
                                              np.arange(12.0).reshape(3, 1, 1, 4))


if __name__ == "__main__":
    unittest.main()
