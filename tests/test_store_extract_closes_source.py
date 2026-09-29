"""``extract()`` must not leave the LIVE store open.

2026-09-28, from a dead production run. ``_store_extract.extract`` opened
the source with ``h5py.File(src_path, "r")`` and never closed it -- only
``dst`` and ``fb`` were closed. As a CLI that is invisible: the process
exits and the handle dies with it, which is why it survived from the day
the tool was written.

``b33f8bcd`` armed ``GF_MONITOR_AFTER_SAVE=1`` + ``GF_MONITOR_SNAPSHOT=1``
in both v9 launchers, and the after-save hook calls ``build_snapshot`` ->
``extract`` **in-process on the saver rank -- the run's only writer**. The
leaked read-only handle then blocked the next ``save_step``:

    File ".../eryn/backends/hdfbackend.py", line 852, in save_step
        with self.open("a") as f:
    OSError: Unable to synchronously open file
             (file is already open for read-only)

HDF5 refuses RDWR while the SAME PROCESS holds the file RDONLY, so this is
reproducible in one interpreter with no MPI and no run. The hook's blanket
``except Exception`` is why nothing pointed at the hook: it succeeded, and
the damage landed one iteration later in ``save_step_main``, outside its
try.
"""
import os
import tempfile
import unittest

import h5py
import numpy as np

from lisatools.globalfit.monitor._store_extract import extract


def _tiny_store(path, nrows=6, iteration=5):
    """The minimum ``extract`` reads: a ``global_fit`` group with an
    ``iteration`` attr, plus one row-axis dataset to copy."""
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = iteration
        g.create_dataset("log_like", data=np.arange(nrows, dtype=float))
        sub = f.create_group("global_fit/sub_backend/gb")
        sub.create_dataset("band_rj_shutoff",
                           data=np.zeros((nrows, 4), dtype=bool))


class ExtractReleasesTheSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "gf_prod_testing.h5")
        self.dst = os.path.join(self.tmp, "gf_prod_testing_extract.h5")
        _tiny_store(self.src)

    def test_the_source_can_be_reopened_for_WRITING_afterwards(self):
        """THE REGRESSION. This is the saver rank's next ``save_step``."""
        extract(self.src, self.dst, keep=3)
        # Before the fix this raises OSError: "file is already open for
        # read-only" -- the exact production failure.
        with h5py.File(self.src, "a") as f:
            f["global_fit"].attrs["iteration"] = 6
        with h5py.File(self.src, "r") as f:
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 6)

    def test_h5py_reports_no_open_handle_on_the_source(self):
        """Direct statement of the leak, independent of what reopening does."""
        extract(self.src, self.dst, keep=3)
        self.assertEqual(
            h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE), 0,
            "extract() left an HDF5 file handle open in this process")

    def test_the_fallback_copy_is_released_too(self):
        """``fb`` IS closed today; pin it so a refactor cannot regress it
        while fixing ``src``. The fallback is auto-discovered from the
        source name, so it exercises the ``fb is not None`` branch."""
        fb_path = self.src[:-3] + "_running_backup_copy.h5"
        _tiny_store(fb_path)
        extract(self.src, self.dst, keep=3)
        for p in (self.src, fb_path):
            with h5py.File(p, "a") as f:
                self.assertIn("global_fit", f)

    def test_the_destination_is_still_complete(self):
        """Closing the source must not cost the extract its output."""
        extract(self.src, self.dst, keep=3)
        with h5py.File(self.dst, "r") as f:
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 5)
            self.assertIn("global_fit/sub_backend/gb/band_rj_shutoff", f)


if __name__ == "__main__":
    unittest.main()
