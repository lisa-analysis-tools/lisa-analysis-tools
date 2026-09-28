"""``snapshot.tar.gz`` -> monitor page in one command.

User ask 2026-09-26: "add a script that allows the user to run the full
tar --> html pipeline from the new code setup."

The manual loop was: mkdir, untar, hunt for the run directory somewhere
under ``shared/data/global_fit_output/<run>/``, call the generator with
it. The failure mode worth testing is the quiet one -- pointing the
generator at the tar's ROOT instead of the run directory renders a page
with every panel missing and no error -- so discovery gets the most
tests here.
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lisatools.globalfit.monitor import from_tar as ft


def _mk(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


class FindRunDirTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_it_finds_the_run_dir_under_the_cluster_layout(self):
        run = os.path.join(self.d, "shared", "data", "global_fit_output",
                           "gf_prod_6mo_v9_4gpu")
        _mk(os.path.join(run, "gf_prod_6mo_testing_extract.h5"))
        _mk(os.path.join(self.d, "preflight.log"))
        self.assertEqual(ft.find_run_dir(self.d), run)

    def test_an_extract_only_snapshot_still_counts(self):
        """A snapshot tar normally ships ONLY the *_extract.h5, so
        requiring a full store would find nothing."""
        run = os.path.join(self.d, "r")
        _mk(os.path.join(run, "gf_prod_3mo_testing_extract.h5"))
        self.assertEqual(ft.find_run_dir(self.d), run)

    def test_it_returns_None_when_there_is_no_store(self):
        _mk(os.path.join(self.d, "a", "notes.txt"))
        self.assertIsNone(ft.find_run_dir(self.d))

    def test_a_non_testing_h5_is_not_a_run_dir(self):
        _mk(os.path.join(self.d, "a", "gb_truth_3to21.h5"))
        self.assertIsNone(ft.find_run_dir(self.d))

    def test_the_DEEPEST_match_wins(self):
        """Tars have nested junk; the run dir is the deep one."""
        shallow = os.path.join(self.d, "x")
        deep = os.path.join(self.d, "x", "y", "z")
        _mk(os.path.join(shallow, "old_testing.h5"))
        _mk(os.path.join(deep, "gf_testing_extract.h5"))
        self.assertEqual(ft.find_run_dir(self.d), deep)


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.src = os.path.join(self.d, "run", "gf_prod_testing_extract.h5")
        _mk(self.src, b"y" * 64)
        self.tar = os.path.join(self.d, "snap.tar.gz")
        with tarfile.open(self.tar, "w:gz") as tf:
            tf.add(os.path.join(self.d, "run"), arcname="run")
        self.scratch = os.path.join(self.d, "scratch")

    def test_it_extracts_and_the_run_dir_is_discoverable(self):
        dest = ft.extract(self.tar, self.scratch)
        self.assertTrue(os.path.isdir(dest))
        self.assertIsNotNone(ft.find_run_dir(dest))

    def test_a_second_call_REUSES_the_extraction(self):
        """130 MB is slow to unpack and usually rendered more than once."""
        d1 = ft.extract(self.tar, self.scratch)
        marker = os.path.join(d1, "run", "TOUCHED")
        _mk(marker)
        d2 = ft.extract(self.tar, self.scratch)
        self.assertEqual(d1, d2)
        self.assertTrue(os.path.exists(marker), "it re-extracted")

    def test_fresh_forces_a_re_extract(self):
        d1 = ft.extract(self.tar, self.scratch)
        marker = os.path.join(d1, "run", "TOUCHED")
        _mk(marker)
        ft.extract(self.tar, self.scratch, fresh=True)
        self.assertFalse(os.path.exists(marker))

    def test_a_REDOWNLOADED_tar_gets_a_different_key(self):
        """The browser names them all '... (3).gz'; reusing a stale
        extraction would silently render the wrong iteration."""
        k1 = ft._tar_key(self.tar)
        os.utime(self.tar, (0, 0))
        self.assertNotEqual(k1, ft._tar_key(self.tar))


class CliTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        _mk(os.path.join(self.d, "run", "gf_prod_testing_extract.h5"))
        self.tar = os.path.join(self.d, "snap.tar.gz")
        with tarfile.open(self.tar, "w:gz") as tf:
            tf.add(os.path.join(self.d, "run"), arcname="run")

    def test_it_calls_build_monitor_with_the_discovered_run_dir(self):
        import lisatools.globalfit.monitor as mon
        with mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(ft.os.path, "getsize", return_value=1):
            rc = ft.main([self.tar, os.path.join(self.d, "p.html"),
                          "--scratch", os.path.join(self.d, "s")])
        self.assertEqual(rc, 0)
        self.assertTrue(bm.call_args[0][0].endswith("run"))

    def test_a_missing_run_dir_is_an_error_not_a_blank_page(self):
        empty = os.path.join(self.d, "empty.tar.gz")
        _mk(os.path.join(self.d, "junk", "notes.txt"))
        with tarfile.open(empty, "w:gz") as tf:
            tf.add(os.path.join(self.d, "junk"), arcname="junk")
        rc = ft.main([empty, "--scratch", os.path.join(self.d, "s2")])
        self.assertEqual(rc, 2)

    def test_it_RESOLVES_the_mojito_tree_and_says_where_from(self):
        """One command, not two (user 2026-09-26). The env var is no
        longer a prerequisite -- it is one source among several, and the
        chosen one is named so a wrong pick is visible."""
        import contextlib
        import io
        import lisatools.globalfit.monitor as mon
        buf = io.StringIO()
        with mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(ft.os.path, "getsize", return_value=1), \
                mock.patch.object(
                    mon, "resolve_mojito_path",
                    return_value=("/moj/tree", "the default search path")), \
                contextlib.redirect_stdout(buf):
            ft.main([self.tar, os.path.join(self.d, "q.html"),
                     "--scratch", os.path.join(self.d, "s3")])
        self.assertIn("/moj/tree", buf.getvalue())
        self.assertIn("the default search path", buf.getvalue())
        # and it is handed to the builder, not left to the child's env
        self.assertEqual(bm.call_args.kwargs.get("mojito"), "/moj/tree")

    def test_it_warns_LOUDLY_when_no_mojito_tree_can_be_found(self):
        """Both things it feeds degrade SILENTLY, so the page cannot tell
        you; this is the only place it gets said."""
        import lisatools.globalfit.monitor as mon
        with mock.patch.object(mon, "build_monitor"), \
                mock.patch.object(ft.os.path, "getsize", return_value=1), \
                mock.patch.object(mon, "resolve_mojito_path",
                                  return_value=(None, "nothing found")), \
                mock.patch.object(ft.sys, "stderr") as err:
            ft.main([self.tar, os.path.join(self.d, "q2.html"),
                     "--scratch", os.path.join(self.d, "s4")])
        said = "".join(str(c) for c in err.write.call_args_list)
        self.assertIn("nothing found", said)
        self.assertIn("--mojito", said)

    def test_run_dir_override_skips_extraction_entirely(self):
        import lisatools.globalfit.monitor as mon
        run = os.path.join(self.d, "run")
        with mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(ft, "extract") as ex, \
                mock.patch.object(ft.os.path, "getsize", return_value=1):
            ft.main([self.tar, os.path.join(self.d, "r.html"),
                     "--run-dir", run])
        ex.assert_not_called()
        self.assertEqual(bm.call_args[0][0], run)

    def test_it_uses_the_packaged_builder_not_a_reimplementation(self):
        """Same fresh-interpreter path as the saver hook, so the page is
        byte-identical rather than merely similar."""
        src = Path(ft.__file__).read_text()
        self.assertIn("from . import build_monitor", src)
        # It must not spawn or shell out ITSELF. (--subprocess is a flag
        # name, not a use, so ban the actual calls rather than the word --
        # the earlier substring ban failed the moment the flag was added.)
        for bad in ("import subprocess", "subprocess.run", "os.system",
                    "shell=True"):
            self.assertNotIn(bad, src, bad)


if __name__ == "__main__":
    unittest.main()


class SingleProcessTest(unittest.TestCase):
    """One python, both outputs (user 2026-09-26).

    The page used to be rendered in a CHILD interpreter because the output
    depended on the caller's import history. Root cause, measured: the
    erebor.noise import chain changes exactly ONE rcParam, font.size
    10 -> 16, and a figure is drawn immediately after it. Fresh process:
    set 10, import raises to 16, panel at 16. Already imported: 16 first,
    the generator's own block sets 10, panel at 10. One panel, 25806
    pixels, everything else identical.

    Pinning font.size at that point made both paths byte-identical -- AND
    identical to every page built before the change -- so the CLI no
    longer needs a child at all.
    """

    def test_from_tar_renders_in_this_process_by_default(self):
        import lisatools.globalfit.monitor as mon
        d = tempfile.mkdtemp()
        _mk(os.path.join(d, "run", "gf_prod_testing_extract.h5"))
        tar = os.path.join(d, "s.tar.gz")
        with tarfile.open(tar, "w:gz") as tf:
            tf.add(os.path.join(d, "run"), arcname="run")
        with mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(ft.os.path, "getsize", return_value=1):
            ft.main([tar, os.path.join(d, "p.html"),
                     "--scratch", os.path.join(d, "s")])
        self.assertIs(bm.call_args.kwargs.get("in_process"), True)

    def test_the_subprocess_escape_hatch_exists(self):
        """Keeping the ~2.5 GB peak and any matplotlib fault out of the
        calling process is still sometimes what you want."""
        import lisatools.globalfit.monitor as mon
        d = tempfile.mkdtemp()
        _mk(os.path.join(d, "run", "gf_prod_testing_extract.h5"))
        tar = os.path.join(d, "s.tar.gz")
        with tarfile.open(tar, "w:gz") as tf:
            tf.add(os.path.join(d, "run"), arcname="run")
        with mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(ft.os.path, "getsize", return_value=1):
            ft.main([tar, os.path.join(d, "p.html"), "--subprocess",
                     "--scratch", os.path.join(d, "s2")])
        self.assertIs(bm.call_args.kwargs.get("in_process"), False)

    def test_the_font_pin_is_present_and_explained(self):
        """It looks like an arbitrary literal; it is the inherited value
        every page to date rendered at, and removing it silently changes
        one panel on every page."""
        import lisatools.globalfit.monitor as mon
        src = Path(mon.generator_path()).read_text()
        self.assertIn("_NOISE_PANEL_FONT = 16.0", src)
        self.assertIn('plt.rcParams["font.size"] = _NOISE_PANEL_FONT', src)
        i = src.index("_NOISE_PANEL_FONT = 16.0")
        self.assertIn("font.size 10.0 -> 16.0", src[max(0, i - 1800):i])


class ArgvDefaultTest(unittest.TestCase):
    """``main()`` must work with NO argument, the way __main__ calls it.

    REGRESSION (2026-09-27). ``main(argv=None)`` fell straight into
    ``if "--" in argv`` without resolving the default, so
    ``sys.exit(main())`` -- the only way the module is ever actually run --
    raised ``TypeError: argument of type 'NoneType' is not iterable`` on
    every real command line. Every test in this file passed throughout,
    because they all call ``main([...])`` with an explicit list: the one
    path nobody exercised was the one everybody uses.
    """

    def test_main_with_no_argv_reads_sys_argv(self):
        import lisatools.globalfit.monitor.from_tar as ft
        with tempfile.TemporaryDirectory() as d:
            _mk(os.path.join(d, "run", "gf_prod_testing_extract.h5"))
            tar = os.path.join(d, "s.tar.gz")
            with tarfile.open(tar, "w:gz") as tf:
                tf.add(os.path.join(d, "run"), arcname="run")
            import lisatools.globalfit.monitor as mon
            argv = ["from_tar", tar, os.path.join(d, "p.html"),
                    "--scratch", os.path.join(d, "s2")]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(mon, "build_monitor") as bm, \
                    mock.patch.object(ft.os.path, "getsize", return_value=1):
                rc = ft.main()          # <-- NO argument. The broken call.
            self.assertEqual(rc, 0)
            self.assertTrue(bm.called)

    def test_no_argv_and_no_tar_exits_cleanly_not_with_a_TypeError(self):
        """argparse's own "required argument" error, not a crash."""
        import lisatools.globalfit.monitor.from_tar as ft
        with mock.patch.object(sys, "argv", ["from_tar"]):
            with self.assertRaises(SystemExit) as cm:
                ft.main()
        self.assertNotEqual(cm.exception.code, 0)


def _load_generator():
    """Load the PACKAGED generator without running its main body.

    ``_generator.py`` guards everything after its helpers with
    ``if __name__ != "__main__": raise SystemExit(0)`` -- the 2026-09-26
    rule that it must never be imported (importing lisatools pulls
    eryn's ``plt.style.use(['science'])`` before the generator's own
    rcParams and silently restyles a panel). Loading it by spec and
    swallowing the SystemExit keeps that rule while still letting the
    helpers above the guard be tested.
    """
    import importlib.util
    import os as _os
    path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "src", "lisatools", "globalfit", "monitor", "_generator.py")
    spec = importlib.util.spec_from_file_location("_gen_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


class SlurmStdoutIsTheSupersetTest(unittest.TestCase):
    """The page was blind on the cluster.

    Measured on the 6mo v9 tars (2026-09-28): ``slurm_stdout_<job>.log``
    is a verbatim SUPERSET of globalfit_run.log plus the rank logs
    (3000/3000 and 2960/3000 sampled lines found in it), and it ALONE
    carries [GF_TIMING], [V9-SEED], [r4/saver] and the stage table. So
    GFT_RE could never match on a cluster-built page -- the "search
    efficiency" panel has been dead there -- and a page built from a
    snapshot whose head log shipped as the filtered + tail pair saw no
    head text at all.
    """

    def setUp(self):
        self.g = _load_generator()
        self.d = tempfile.mkdtemp()

    def _touch(self, rel, text="x\n"):
        p = os.path.join(self.d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def test_the_newest_stdout_is_appended_when_run_logs_exist(self):
        head = self._touch("globalfit_run.log")
        r1 = self._touch("globalfit_run.rank1.log")
        old = self._touch("slurm_stdout_650.log")
        new = self._touch("slurm_stdout_659.log")
        os.utime(old, (1, 1))
        os.utime(new, (10**9, 10**9))
        got = self.g.discover_run_logs(self.d)
        self.assertEqual(got[:2], [head, r1], "head must keep priority")
        self.assertEqual(got[-1], new)
        self.assertNotIn(old, got, "a dead job's stdout must not be replayed")

    def test_a_snapshot_with_only_filtered_and_tail_still_gets_text(self):
        """The names the short/full tar actually ships for a big log."""
        self._touch("globalfit_run_filtered.log")
        self._touch("globalfit_run_tail.log")
        so = self._touch("slurm_stdout_659.log")
        self.assertEqual(self.g.discover_run_logs(self.d), [so])

    def test_stdout_alone_is_enough(self):
        so = self._touch("slurm_stdout_1.log")
        self.assertEqual(self.g.discover_run_logs(self.d), [so])

    def test_nothing_at_all_warns_rather_than_rendering_empty_panels(self):
        import contextlib
        import io as _io
        err = _io.StringIO()
        with contextlib.redirect_stderr(err):
            got = self.g.discover_run_logs(self.d)
        self.assertEqual(got, [])
        self.assertIn("no run log and no slurm_stdout", err.getvalue())

    def test_run_logs_only_is_unchanged(self):
        head = self._touch("globalfit_run.log")
        r1 = self._touch("globalfit_run.rank1.log")
        self.assertEqual(self.g.discover_run_logs(self.d), [head, r1])


class FstatPoolIsNotAlwaysCalledSharedTest(unittest.TestCase):
    """Why the comb / peaks panels are missing on every cluster page.

    The generator hard-coded ``gb_fstat_fit/shared``; 6mo v9 writes
    ``gb_fstat_fit/shared_search`` (confirmed in the snapshot listing:
    ``gb_fstat_fit/shared_search/epoch_0000/DONE.json``), so the path
    simply did not exist and the panels degraded silently.
    """

    def test_the_generator_no_longer_hardcodes_the_pool_name(self):
        import inspect
        src = inspect.getsource(_load_generator())
        self.assertNotIn('"gb_fstat_fit", "shared")', src)
        self.assertIn('_froot = os.path.join(RUN_DIR, "gb_fstat_fit")', src)

    def test_the_legacy_pool_is_still_preferred_when_present(self):
        import inspect
        src = inspect.getsource(_load_generator())
        self.assertIn("_pools.insert(0, _legacy)", src)

    def test_the_degrade_message_names_the_snapshot_cause_too(self):
        """A snapshot drops everything under gb_fstat_fit but DONE.json
        unless include_fstat, so 'the fit is still running' was a
        misleading explanation for half the cases."""
        import inspect
        src = inspect.getsource(_load_generator())
        self.assertIn("include_fstat", src)
