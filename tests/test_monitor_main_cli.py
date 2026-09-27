"""``python -m lisatools.globalfit.monitor [--snapshot] RUN_DIR``

User ask 2026-09-26: "I want the tar file build together with the html
generation in one command."

That command is ``--snapshot``, and the thing worth pinning is the ORDER.
The page is written INTO the run directory and the tarball is built
AFTERWARDS, so the tar contains the report. Reverse them and you get a
snapshot of a run with no report in it -- which defeats the point of
doing both in one go, and would do so silently: both artifacts still
appear, both look right, and only opening the tar shows the page missing.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import lisatools.globalfit.monitor.__main__ as m


class SnapshotFlagTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_prod_run")
        os.makedirs(self.run)

    def test_without_the_flag_only_the_page_is_built(self):
        with mock.patch.object(m, "build_monitor") as bm, \
                mock.patch.object(m, "build_snapshot") as bs:
            rc = m.main([self.run])
        self.assertEqual(rc, 0)
        bm.assert_called_once()
        bs.assert_not_called()

    def test_the_flag_builds_BOTH(self):
        with mock.patch.object(m, "build_monitor") as bm, \
                mock.patch.object(m, "build_snapshot",
                                  return_value="/t.tar.gz") as bs:
            rc = m.main(["--snapshot", self.run])
        self.assertEqual(rc, 0)
        bm.assert_called_once()
        bs.assert_called_once_with(self.run)

    def test_the_page_is_still_built_first(self):
        """No longer a containment guarantee -- the page is a SIBLING of
        the run dir now, so the tar does not carry it. Kept because a
        failed tar must not cost you the page as well: build the cheap
        certain artifact before the expensive fallible one."""
        order = []
        with mock.patch.object(m, "build_monitor",
                               side_effect=lambda *a, **k: order.append("page")), \
                mock.patch.object(m, "build_snapshot",
                                  side_effect=lambda *a, **k: (
                                      order.append("tar") or "/t.tar.gz")):
            m.main(["--snapshot", self.run])
        self.assertEqual(order, ["page", "tar"])

    def test_the_page_defaults_BESIDE_the_run_dir_not_inside_it(self):
        """User ruling 2026-09-26. Both artifacts sit next to the folder so
        they are found together; nothing is written into the run."""
        with mock.patch.object(m, "build_monitor") as bm, \
                mock.patch.object(m, "build_snapshot",
                                  return_value="/t.tar.gz"):
            m.main(["--snapshot", self.run])
        out = bm.call_args[0][1]
        self.assertEqual(out, os.path.abspath(self.run) + "_monitor.html")
        self.assertEqual(os.path.dirname(out),
                         os.path.dirname(os.path.abspath(self.run)))
        self.assertFalse(out.startswith(os.path.abspath(self.run) + os.sep))

    def test_an_explicit_out_path_is_honoured(self):
        out = os.path.join(self.d, "elsewhere.html")
        with mock.patch.object(m, "build_monitor") as bm:
            m.main([self.run, out])
        self.assertEqual(bm.call_args[0][1], out)

    def test_a_failed_snapshot_is_a_nonzero_exit_but_keeps_the_page(self):
        """Silently returning 0 would let a cron job think it shipped a
        tarball it never wrote."""
        with mock.patch.object(m, "build_monitor") as bm, \
                mock.patch.object(m, "build_snapshot", return_value=None):
            rc = m.main(["--snapshot", self.run])
        self.assertEqual(rc, 1)
        bm.assert_called_once()

    def test_usage_names_BOTH_directions(self):
        """The two entry points are easy to confuse: this one takes a RUN
        DIR and can make a tar; from_tar takes a TAR and cannot."""
        self.assertIn("--snapshot", m.USAGE)
        self.assertIn("from_tar", m.USAGE)

    def test_no_args_is_usage_and_a_nonzero_exit(self):
        self.assertEqual(m.main([]), 2)

    def test_help_exits_zero(self):
        self.assertEqual(m.main(["--help"]), 0)

    def test_the_sibling_layout_is_written_down(self):
        """Including the guarantee it reverses."""
        src = Path(m.__file__).read_text()
        self.assertIn("BESIDE THE RUN FOLDER", src)
        self.assertIn("It no\nlonger does", src)


if __name__ == "__main__":
    unittest.main()


class InProcessHandleLeakTest(unittest.TestCase):
    """The generator leaves its HDF5 store open; in-process we must close it.

    It is a SCRIPT: it opens the store at module level and never closes
    it, because a child process exits and takes the handle with it. Once
    the render stopped being a subprocess the handle survived into the
    snapshot step and h5py refused to create the extract:

        OSError: Unable to synchronously create file (unable to truncate
                 a file which is already open)

    The page was written and the tar silently was not -- exit 0 in the
    first version, which is why --snapshot now also returns nonzero.
    """

    def test_build_monitor_in_process_closes_what_the_script_left_open(self):
        import lisatools.globalfit.monitor as mon
        src = Path(mon.__file__).read_text()
        i = src.index("def build_monitor_in_process")
        blk = src[i:i + 4000]
        self.assertIn("ns = runpy.run_path", blk,
                      "the module globals must be captured to reach the "
                      "handles")
        self.assertIn("isinstance(_v, h5py.File)", blk)
        self.assertIn(".close()", blk)

    def test_the_reason_is_recorded(self):
        """Without it this reads as defensive tidying and gets removed."""
        import lisatools.globalfit.monitor as mon
        src = Path(mon.__file__).read_text()
        self.assertIn("truncate a file which is already open", src)
