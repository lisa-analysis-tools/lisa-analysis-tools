"""Snapshot tar production from the installed package.

``lisatools.globalfit.monitor.snapshot`` is ``make_snapshots.sh``
expressed in Python so the saver rank can build a snapshot without a
shell, a ``cd`` to the repo root, or a source checkout.

THE SHELL SCRIPT IS DELIBERATELY STILL THERE (user instruction
2026-09-26: "try not to touch any other code"), so two implementations
of one recipe exist. ``SnapshotMatchesTheShellRecipeTest`` compares the
parts that can silently diverge -- the kept log-line families and the
exclusion list -- so the pair cannot drift without a test saying so.
"""

from __future__ import annotations

import os
import re
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lisatools.globalfit.monitor import snapshot as snap

REPO = Path(__file__).resolve().parents[1]
SHELL = REPO / "scripts" / "fstat_proposal" / "make_snapshots.sh"


def _touch(p, size=16):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(b"x" * size)
    return p


class LiveStoreTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_it_ignores_backup_corrupt_and_extract(self):
        import time
        for name in ("gf_prod_testing.h5",
                     "gf_prod_testing_running_backup_copy.h5",
                     "gf_prod_testing_CORRUPT.h5",
                     "gf_prod_testing_extract.h5"):
            _touch(os.path.join(self.d, name))
            time.sleep(0.01)
        self.assertEqual(os.path.basename(snap._live_store(self.d)),
                         "gf_prod_testing.h5")

    def test_no_store_is_None_not_an_exception(self):
        self.assertIsNone(snap._live_store(self.d))


class MembersTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_run")
        for rel in (
            "gf_prod_testing.h5",                  # full store: OUT
            "gf_prod_testing_extract.h5",          # reduced store: IN
            "gf_prod_testing.h5.bak",              # backup copy: OUT
            "gf_prod_testing_midit_checkpoint.pkl",  # OUT
            "old_snapshot.tar.gz",                 # OUT
            "gf_artifacts/globalfit_run.log",      # IN
            "gf_artifacts/diagnostics/plot.png",   # OUT
            "dissect/dump.npz",                    # OUT
            "gb_fstat_fit/DONE.json",              # IN
            "gb_fstat_fit/peaks_stacked.npz",      # OUT (payload)
            "fstat_grid_parts/part0.npz",          # OUT
            "run_settings.log",                    # IN
        ):
            _touch(os.path.join(self.run, rel))

    def _names(self, include_fstat=False):
        got = snap._members(self.run, include_fstat, None)
        return sorted(os.path.relpath(p, self.run) for p in got)

    def test_the_exclusions_hold(self):
        self.assertEqual(self._names(), [
            "gb_fstat_fit/DONE.json",
            "gf_artifacts/globalfit_run.log",
            "gf_prod_testing_extract.h5",
            "run_settings.log",
        ])

    def test_the_full_store_is_out_and_the_extract_is_in(self):
        names = self._names()
        self.assertIn("gf_prod_testing_extract.h5", names)
        self.assertNotIn("gf_prod_testing.h5", names)

    def test_INCLUDE_FSTAT_keeps_the_payloads(self):
        self.assertIn("gb_fstat_fit/peaks_stacked.npz",
                      self._names(include_fstat=True))

    def test_the_grid_parts_stay_out_even_with_INCLUDE_FSTAT(self):
        """The one family that is excluded unconditionally -- it is the
        heaviest thing in the directory."""
        self.assertNotIn("fstat_grid_parts/part0.npz",
                         self._names(include_fstat=True))

    def test_an_explicitly_skipped_raw_log_is_dropped(self):
        raw = os.path.join(self.run, "gf_artifacts", "globalfit_run.log")
        got = snap._members(self.run, False, raw)
        self.assertNotIn(raw, got)


class BuildSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_run")
        _touch(os.path.join(self.run, "gf_prod_testing.h5"))
        _touch(os.path.join(self.run, "run_settings.log"))

    def _fake_extract(self, src, dst, keep, cold_keep=None):
        _touch(dst, 32)

    def test_it_writes_a_tar_containing_the_extract(self):
        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=self._fake_extract):
            out = snap.build_snapshot(self.run)
        self.assertIsNotNone(out)
        self.assertTrue(out.endswith("gf_run_snapshot.tar.gz"))
        with tarfile.open(out) as tf:
            names = tf.getnames()
        self.assertIn("gf_run/gf_prod_testing_extract.h5", names)
        self.assertIn("gf_run/run_settings.log", names)
        self.assertNotIn("gf_run/gf_prod_testing.h5", names)

    def test_it_publishes_atomically(self):
        """A consumer polling for the tar must never see a partial one."""
        seen = {}
        real = os.replace

        def spy(a, b):
            seen["from"], seen["to"] = a, b
            return real(a, b)

        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=self._fake_extract), \
                mock.patch.object(snap.os, "replace", side_effect=spy):
            out = snap.build_snapshot(self.run)
        self.assertEqual(seen["from"], out + ".tmp")
        self.assertEqual(seen["to"], out)

    def test_a_failed_extract_returns_None_and_does_not_raise(self):
        """This runs on the run's only writer. A failed diagnostic must
        never become a failed run."""
        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=RuntimeError("boom")):
            self.assertIsNone(snap.build_snapshot(self.run))

    def test_no_live_store_returns_None_and_does_not_raise(self):
        empty = os.path.join(self.d, "empty")
        os.makedirs(empty)
        self.assertIsNone(snap.build_snapshot(empty))

    def test_no_tmp_file_is_left_behind_on_failure(self):
        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=RuntimeError("boom")):
            snap.build_snapshot(self.run)
        self.assertFalse(os.path.exists(self.run + "_snapshot.tar.gz.tmp"))


class LogFilterTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_run")
        self.log = os.path.join(self.run, "gf_artifacts", "globalfit_run.log")

    def _write(self, mb):
        body = (b"[SAVE] keep me\nnoise line drop me\nWARNING keep me too\n")
        _touch(self.log, 1)
        with open(self.log, "wb") as fh:
            fh.write(body)
            fh.write(b"filler drop\n" * (mb * 1048576 // 12))

    def test_below_the_cap_the_raw_log_ships_whole(self):
        self._write(0)
        self.assertIsNone(snap._filter_run_log(self.run, log_cap_mb=200))

    def test_above_the_cap_it_writes_filtered_plus_tail_and_excludes_raw(self):
        self._write(3)
        skip = snap._filter_run_log(self.run, log_cap_mb=1)
        self.assertEqual(skip, self.log)
        base = self.log[:-4]
        self.assertTrue(os.path.exists(base + "_filtered.log"))
        self.assertTrue(os.path.exists(base + "_tail.log"))
        kept = open(base + "_filtered.log", "rb").read()
        self.assertIn(b"[SAVE] keep me", kept)
        self.assertIn(b"WARNING keep me too", kept)
        self.assertNotIn(b"filler drop", kept)

    def test_a_filtering_failure_ships_the_log_whole_rather_than_nothing(self):
        self._write(3)
        with mock.patch("builtins.open", side_effect=OSError("nope")):
            self.assertIsNone(snap._filter_run_log(self.run, log_cap_mb=1))


class SnapshotMatchesTheShellRecipeTest(unittest.TestCase):
    """Anti-drift between the Python port and make_snapshots.sh."""

    @classmethod
    def setUpClass(cls):
        cls.sh = SHELL.read_text() if SHELL.exists() else ""

    def test_the_shell_script_is_still_there_to_compare_against(self):
        """If it is ever deleted, delete this class too -- do not let it
        pass vacuously."""
        self.assertTrue(self.sh, f"{SHELL} missing")

    def test_the_kept_log_line_families_match(self):
        """The grep -aE alternation and LOG_KEEP_PATTERN must select the
        same line families, or a filtered snapshot silently loses a
        diagnostic the page reads."""
        m = re.search(r'grep -aE "([^"]+)"', self.sh, re.S)
        self.assertIsNotNone(m, "the grep -aE alternation moved")
        shell_alts = {a for a in m.group(1).replace("\\\n", "").split("|") if a}
        py_alts = {a for a in snap.LOG_KEEP_PATTERN.split("|") if a}
        self.assertEqual(shell_alts, py_alts)

    def test_the_heavy_exclusions_match(self):
        for frag in ("fstat_grid_parts", "dissect", "_midit_checkpoint.pkl"):
            self.assertIn(frag, self.sh)
            self.assertTrue(
                any(frag in x for x in
                    snap._EXCLUDE_PATH_PARTS + snap._EXCLUDE_SUFFIXES),
                f"{frag} excluded by the shell but not by the port")

    def test_the_default_keep_windows_match(self):
        import inspect
        sig = inspect.signature(snap.build_snapshot)
        self.assertIn("KEEP=${KEEP:-5}", self.sh)
        self.assertIn("COLD_KEEP=${COLD_KEEP:-12}", self.sh)
        self.assertEqual(sig.parameters["keep"].default, 5)
        self.assertEqual(sig.parameters["cold_keep"].default, 12)

    def test_the_log_cap_matches(self):
        self.assertIn("LOG_CAP_MB=${LOG_CAP_MB:-200}", self.sh)
        import inspect
        self.assertEqual(
            inspect.signature(snap.build_snapshot)
            .parameters["log_cap_mb"].default, 200)


if __name__ == "__main__":
    unittest.main()
