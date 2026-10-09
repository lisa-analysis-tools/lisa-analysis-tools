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

import io
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
        bs.assert_called_once_with(self.run, short=False, include_fstat=False,
                                   keep=None, cold_keep=None)

    def test_the_fstat_caches_are_OUT_unless_add_fstat(self):
        # User ruling 2026-10-03: "make it default to leaving them out. If
        # you want them, you add --add-fstat". The CLI passes the choice
        # through as build_snapshot(include_fstat=...).
        with mock.patch.object(m, "build_monitor"), \
                mock.patch.object(m, "build_snapshot",
                                  return_value="/x/y_snapshot.tar.gz") as bs:
            self.assertEqual(m.main(["--snapshot", self.run]), 0)
            self.assertIs(bs.call_args.kwargs.get("include_fstat"), False)
        with mock.patch.object(m, "build_monitor"), \
                mock.patch.object(m, "build_snapshot",
                                  return_value="/x/y_snapshot.tar.gz") as bs:
            self.assertEqual(m.main(["--snapshot", "--add-fstat", self.run]), 0)
            self.assertIs(bs.call_args.kwargs.get("include_fstat"), True)

    def test_snapshot_only_builds_the_TAR_and_NOT_the_page(self):
        with mock.patch.object(m, "build_monitor") as bm, \
                mock.patch.object(m, "build_snapshot",
                                  return_value="/t.tar.gz") as bs:
            rc = m.main(["--snapshot-only", self.run])
        self.assertEqual(rc, 0)
        bm.assert_not_called()
        bs.assert_called_once_with(self.run, short=False, include_fstat=False,
                                   keep=None, cold_keep=None)

    def test_snapshot_only_failure_does_not_claim_a_page_was_written(self):
        """The old message reassured the operator about a page that, in
        this mode, was never asked for."""
        with mock.patch.object(m, "build_monitor"), \
                mock.patch.object(m, "build_snapshot", return_value=None), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            rc = m.main(["--snapshot-only", self.run])
        self.assertEqual(rc, 1)
        self.assertNotIn("page was still written", err.getvalue())

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

    def test_no_args_is_an_error(self):
        """argparse, not hand-rolled exit codes."""
        with self.assertRaises(SystemExit) as cm:
            m.main([])
        self.assertNotEqual(cm.exception.code, 0)

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            m.main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_an_UNKNOWN_FLAG_is_rejected_not_used_as_the_out_path(self):
        """The bug this parser exists for. `RUN_DIR --catalogue X` used to
        set out="--catalogue" silently -- the command ran, wrote the page
        to a file called "--catalogue", and never told anyone."""
        with self.assertRaises(SystemExit) as cm:
            m.main([self.run, "--no-such-flag"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_build_truth_args_pass_through_after_a_double_dash(self):
        """And --build-truth itself must SURVIVE the split. With
        argparse.REMAINDER it did not: everything from the first
        unrecognised token on was absorbed, so --build-truth parsed as
        False and was handed to build_truth as an argument."""
        with mock.patch.object(m, "build_monitor"), \
                mock.patch.object(m, "check_truth",
                                  return_value=(None, "absent")), \
                mock.patch.object(m, "build_truth_set") as bt:
            m.main(["--snapshot", self.run, "--build-truth",
                    "--", "--catalogue", "/p/wdwd.hdf5"])
        self.assertEqual(bt.call_args.kwargs["extra_argv"],
                         ["--catalogue", "/p/wdwd.hdf5"])

    def test_passthru_args_without_build_truth_are_called_out(self):
        """Otherwise they vanish without a word."""
        with mock.patch.object(m, "build_monitor"), \
                mock.patch.object(m, "check_truth",
                                  return_value=(None, "absent")), \
                mock.patch.object(m.sys, "stderr") as err:
            m.main([self.run, "--", "--catalogue", "/p/x.hdf5"])
        said = "".join(str(c) for c in err.write.call_args_list)
        self.assertIn("--catalogue", said)

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
        blk = src[i:i + 6000]
        # MECHANISM CHANGED 2026-09-28: runpy.run_path -> exec into a dict
        # we own. run_path hands the globals back only on SUCCESS, so a
        # page that RAISED closed nothing, the store stayed open read-only
        # in the saver rank, and the next save_step died with
        # "file is already open for read-only" -- how the 3mo run died.
        # Pin the INVARIANT (the namespace is ours before the code runs),
        # not whichever call happens to provide it.
        self.assertIn('ns = {"__name__": "__main__"', blk,
                      "the module globals must be captured BEFORE the "
                      "generator runs, so they are reachable even when it "
                      "raises")
        # Match the IMPORT, not the name: the comment above the fix cites
        # ``runpy.run_path`` precisely to stop anyone reinstating it, and
        # an assertNotIn on the bare name would fire on that warning.
        self.assertNotIn("import runpy", blk,
                         "runpy.run_path cannot reach the handles of a page "
                         "that FAILED -- it returns the globals only on "
                         "success. See test_monitor_in_process_handle_leak.")
        self.assertIn("isinstance(_v, h5py.File)", blk)
        # a live Dataset holds its File open just as firmly as a Group
        self.assertIn("h5py.Dataset", blk)
        self.assertIn(".close()", blk)

    def test_the_reason_is_recorded(self):
        """Without it this reads as defensive tidying and gets removed."""
        import lisatools.globalfit.monitor as mon
        src = Path(mon.__file__).read_text()
        self.assertIn("truncate a file which is already open", src)


class CatalogueFlagTest(unittest.TestCase):
    """--catalogue / --l1-brick are first-class, not pass-through only.

    The natural spelling -- `--build-truth RUN --catalogue /path/x.hdf5` --
    produced "unrecognized arguments" on a real cluster run, because the
    only way through was `-- --catalogue /path/x.hdf5`. Requiring that is
    a trap for the two overrides anyone actually reaches for.
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_prod_run")
        os.makedirs(self.run)

    def _call(self, argv):
        with mock.patch.object(m, "build_monitor"), \
                mock.patch.object(m, "check_truth",
                                  return_value=(None, "absent")), \
                mock.patch.object(m, "build_truth_set") as bt:
            m.main(argv)
        return bt.call_args.kwargs["extra_argv"]

    def test_catalogue_is_accepted_in_the_natural_position(self):
        got = self._call([self.run, "--build-truth",
                          "--catalogue", "/p/wdwd.hdf5"])
        self.assertEqual(got, ["--catalogue", "/p/wdwd.hdf5"])

    def test_l1_brick_too(self):
        got = self._call([self.run, "--build-truth",
                          "--l1-brick", "/p/any_L1_x.h5"])
        self.assertEqual(got, ["--l1-brick", "/p/any_L1_x.h5"])

    def test_both_plus_double_dash_passthru_compose(self):
        got = self._call([self.run, "--build-truth",
                          "--catalogue", "/p/w.hdf5",
                          "--", "--flo", "5e-4"])
        self.assertEqual(got, ["--flo", "5e-4", "--catalogue", "/p/w.hdf5"])

    def test_neither_given_means_no_extra_args(self):
        self.assertEqual(self._call([self.run, "--build-truth"]), [])


class CatalogueDirectoryLevelTest(unittest.TestCase):
    """A directory candidate may be EITHER level.

    MOJITO_CAT historically named the directory holding the file
    (.../catalogues/); MOJITO_INFO_PATH names the TREE that holds
    catalogues/ and data/. Trying only <dir>/<name> meant a correct
    MOJITO_INFO_PATH resolved one level short and the build died having
    "tried" a path that never existed.
    """

    def setUp(self):
        from lisatools.globalfit.monitor import build_truth as bt
        self.bt = bt
        self.d = tempfile.mkdtemp()
        self.cat = os.path.join(self.d, "catalogues", bt.MOJITO_CAT_NAME)
        os.makedirs(os.path.dirname(self.cat))
        open(self.cat, "wb").close()
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("MOJITO_INFO_PATH", "MOJITO_CAT", "MOJITO_CACHE_DIR"):
            os.environ.pop(k, None)
        # Neutralise the ~/.mojito_cache default, or a real laptop cache
        # resolves and the failure test never sees a failure.
        os.environ["HOME"] = os.path.join(self.d, "nohome")
        self.addCleanup(self.env.stop)

    def test_the_TREE_level_resolves(self):
        os.environ["MOJITO_INFO_PATH"] = self.d
        self.assertEqual(self.bt.resolve_catalogue(), self.cat)

    def test_the_catalogues_level_still_resolves(self):
        os.environ["MOJITO_CAT"] = os.path.dirname(self.cat)
        self.assertEqual(self.bt.resolve_catalogue(), self.cat)

    def test_an_explicit_file_still_wins(self):
        self.assertEqual(self.bt.resolve_catalogue(self.cat), self.cat)

    def test_both_attempted_paths_are_named_when_it_fails(self):
        """The error has to show what it looked for, or the next person
        cannot tell a wrong variable from a wrong level."""
        bad = os.path.join(self.d, "nope")
        os.makedirs(bad)
        os.environ["MOJITO_INFO_PATH"] = bad
        # MOJITO_CAT_DEFAULT is computed at import from the ORIGINAL HOME,
        # so patch it too -- otherwise the real cache satisfies the lookup.
        with mock.patch.object(self.bt, "MOJITO_CAT_DEFAULT",
                               os.path.join(bad, "none.hdf5")):
            # SystemExit, not Exception -- build_truth is a CLI. That is
            # precisely why build_truth_set has to convert it.
            with self.assertRaises(SystemExit) as cm:
                self.bt.resolve_catalogue()
        msg = str(cm.exception.code)
        self.assertIn("catalogues", msg,
                      "the error must show BOTH levels it tried")


class TruthFailureMustNotCostThePageTest(unittest.TestCase):
    """A failed truth build costs overlays, never the report.

    build_truth is a CLI: resolve_catalogue raises SystemExit, which
    `except Exception` does not catch. On the cluster (2026-09-27) the
    catalogue was not found, the truth build aborted, and THE PAGE WAS
    NEVER WRITTEN -- the command simply ended.
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_prod_run")
        os.makedirs(self.run)

    def test_a_SystemExit_from_build_truth_still_renders_the_page(self):
        import lisatools.globalfit.monitor as mon
        with mock.patch.object(mon, "resolve_mojito_path",
                               return_value=(None, "x")), \
                mock.patch("lisatools.globalfit.monitor.build_truth.main",
                           side_effect=SystemExit("no catalogue")), \
                mock.patch.object(m, "build_monitor") as bm, \
                mock.patch.object(m, "check_truth",
                                  return_value=(None, "absent")), \
                mock.patch.object(m.sys, "stderr"):
            rc = m.main([self.run, "--build-truth"])
        self.assertEqual(rc, 0)
        bm.assert_called_once()

    def test_build_truth_set_converts_SystemExit_to_RuntimeError(self):
        import lisatools.globalfit.monitor as mon
        import h5py
        with h5py.File(os.path.join(self.run, "gf_prod_testing.h5"), "w") as f:
            f.create_group("global_fit")
        with mock.patch.object(mon, "resolve_mojito_path",
                               return_value=(None, "x")), \
                mock.patch("lisatools.globalfit.monitor.build_truth.main",
                           side_effect=SystemExit("boom")):
            with self.assertRaises(RuntimeError) as cm:
                mon.build_truth_set(self.run)
        self.assertIn("boom", str(cm.exception))


class ShortFlagTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_prod_run")
        os.makedirs(self.run)

    def test_short_implies_building_a_tar(self):
        """--short alone is a request for the short tar; it should not
        need --snapshot as well."""
        with mock.patch.object(m, "build_monitor"), \
                mock.patch.object(m, "build_snapshot",
                                  return_value="/t_short.tar.gz") as bs:
            rc = m.main(["--short", self.run])
        self.assertEqual(rc, 0)
        bs.assert_called_once_with(self.run, short=True, include_fstat=False,
                                   keep=None, cold_keep=None)

    def test_short_with_snapshot_only_skips_the_page(self):
        with mock.patch.object(m, "build_monitor") as bm, \
                mock.patch.object(m, "build_snapshot",
                                  return_value="/t_short.tar.gz") as bs:
            rc = m.main(["--snapshot-only", "--short", self.run])
        self.assertEqual(rc, 0)
        bm.assert_not_called()
        bs.assert_called_once_with(self.run, short=True, include_fstat=False,
                                   keep=None, cold_keep=None)

    def test_keep_and_cold_keep_reach_the_snapshot(self):
        # 189f7207 (10-08): --keep / --cold-keep size the extract's warm and
        # cold row windows; None leaves build_snapshot's own defaults
        with mock.patch.object(m, "build_monitor"), \
                mock.patch.object(m, "build_snapshot",
                                  return_value="/t.tar.gz") as bs:
            m.main(["--snapshot", "--keep", "3", "--cold-keep", "7", self.run])
        bs.assert_called_once_with(self.run, short=False, include_fstat=False,
                                   keep=3, cold_keep=7)

    def test_plain_snapshot_is_still_the_FULL_one(self):
        with mock.patch.object(m, "build_monitor"), \
                mock.patch.object(m, "build_snapshot",
                                  return_value="/t.tar.gz") as bs:
            m.main(["--snapshot", self.run])
        bs.assert_called_once_with(self.run, short=False, include_fstat=False,
                                   keep=None, cold_keep=None)
