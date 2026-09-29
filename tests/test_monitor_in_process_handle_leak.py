"""A FAILING page must not leave the store open in the saver process.

2026-09-28, from the dead 3mo production run. ``build_monitor`` defaults to
``in_process=True``, so on the saver rank -- the run's ONLY writer -- the
generator is executed by ``runpy.run_path`` inside that process. The
generator is a script: it opens the store at module level
(``f = h5py.File(h5path, "r")``) and never closes it, because as a child it
exits and takes the handle with it.

``build_monitor_in_process`` knows this and closes the handles in a
``finally`` -- but it reaches them through ``ns``, and **runpy returns the
module globals only on SUCCESS**. Let the generator raise anything but
``SystemExit`` and ``ns`` is still ``None``, ``(ns or {})`` is ``{}``, and
nothing is closed. Refcounting does not save it either: a module globals
dict is a reference cycle (every function's ``__globals__`` points back at
it), so it survives until the cyclic GC runs.

The store then stays open READ-ONLY inside the writer, and the next save is:

    File ".../eryn/backends/hdfbackend.py", line 852, in save_step
        with self.open("a") as f:
    OSError: Unable to synchronously open file
             (file is already open for read-only)

which is exactly how the 3mo run died. It explains both symptoms Mike saw:
the 3mo page raises (his standalone monitor error), so the handles leak and
the following save dies; the 6mo page succeeds, so ``ns`` comes back, the
handles close, and that run is fine.

The ``finally`` block's own comment already documents a sibling failure of
this class ("unable to truncate a file which is already open", which broke
``--snapshot``). It simply never covered the raise path.
"""
import gc
import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from lisatools.globalfit import monitor


# ⚠ THE STUB MUST DEFINE A FUNCTION. That is not decoration: a module
# globals dict only becomes a REFERENCE CYCLE because the functions defined
# in it hold ``__globals__`` back-references. Without one, refcounting
# reaps the namespace as soon as the traceback is released and the handle
# closes on its own -- the leak does not reproduce and the test passes
# against the broken code. ``_generator.py`` defines dozens.
_STUB_RAISES = '''
import glob, os, sys
import h5py
run_dir = sys.argv[1]
store = glob.glob(os.path.join(run_dir, "*testing*.h5"))[0]
# module level, never closed -- exactly like _generator.py
f = h5py.File(store, "r")
g = f["global_fit"]
d = g["log_like"]

def _panel():                 # the back-reference that makes the cycle
    return f, g, d

raise RuntimeError("the page failed, as the 3mo page does")
'''

_STUB_SUCCEEDS = '''
import glob, os, sys
import h5py
run_dir, out_path = sys.argv[1], sys.argv[2]
store = glob.glob(os.path.join(run_dir, "*testing*.h5"))[0]
f = h5py.File(store, "r")
g = f["global_fit"]
d = g["log_like"]
with open(out_path, "w") as fh:
    fh.write("<html><body>page</body></html>")
'''


class InProcessPageLeakTest(unittest.TestCase):
    def setUp(self):
        self.run_dir = tempfile.mkdtemp()
        self.store = os.path.join(self.run_dir, "gf_prod_testing.h5")
        with h5py.File(self.store, "w") as f:
            g = f.create_group("global_fit")
            g.attrs["iteration"] = 5
            g.create_dataset("log_like", data=np.zeros((6, 1, 1, 4)))
        self.out = os.path.join(self.run_dir, "gf_monitor.html")
        gc.collect()

    def _stub(self, body):
        p = os.path.join(self.run_dir, "_stub_generator.py")
        with open(p, "w") as fh:
            fh.write(body)
        return mock.patch.object(monitor, "generator_path", lambda: p)

    def _open_handles(self):
        return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE)

    def test_a_RAISING_page_releases_the_store(self):
        """THE REGRESSION -- this is the production failure."""
        with self._stub(_STUB_RAISES):
            self.assertIsNone(
                monitor.build_monitor(self.run_dir, self.out, check=False))
        self.assertEqual(self._open_handles(), 0,
                         "the failed page left an HDF5 handle open in this "
                         "process; the next save_step will fail")
        # the saver rank's very next action
        with h5py.File(self.store, "a") as f:
            f["global_fit"].attrs["iteration"] = 6

    def test_a_SUCCEEDING_page_still_releases_the_store(self):
        """Guard the path that already worked, so a fix cannot regress it."""
        with self._stub(_STUB_SUCCEEDS):
            self.assertEqual(
                monitor.build_monitor(self.run_dir, self.out, check=False),
                self.out)
        self.assertEqual(self._open_handles(), 0)
        with h5py.File(self.store, "a") as f:
            f["global_fit"].attrs["iteration"] = 7

    def test_the_generators_traceback_reaches_the_log(self):
        """``check=False`` reduced the failure to one line with no
        traceback (the in-process path has no ``.stderr``), which is why
        nobody could see WHY the 3mo page failed."""
        with self._stub(_STUB_RAISES):
            with self.assertLogs("lisatools.globalfit.monitor",
                                 level="WARNING") as cm:
                monitor.build_monitor(self.run_dir, self.out, check=False)
        blob = "\n".join(cm.output)
        self.assertIn("the page failed, as the 3mo page does", blob)
        self.assertIn("Traceback", blob)


if __name__ == "__main__":
    unittest.main()
