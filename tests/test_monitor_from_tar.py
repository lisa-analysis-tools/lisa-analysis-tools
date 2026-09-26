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

    def test_it_warns_up_front_when_MOJITO_INFO_PATH_is_unset(self):
        """Both things it feeds degrade SILENTLY; finding out after a
        two-minute render is the annoying way to learn it."""
        import lisatools.globalfit.monitor as mon
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOJITO_INFO_PATH", None)
            with mock.patch.object(mon, "build_monitor"), \
                    mock.patch.object(ft.os.path, "getsize", return_value=1), \
                    mock.patch.object(ft.sys, "stderr") as err:
                ft.main([self.tar, os.path.join(self.d, "q.html"),
                         "--scratch", os.path.join(self.d, "s3")])
        said = "".join(str(c) for c in err.write.call_args_list)
        self.assertIn("MOJITO_INFO_PATH", said)

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
        self.assertNotIn("subprocess", src)


if __name__ == "__main__":
    unittest.main()
