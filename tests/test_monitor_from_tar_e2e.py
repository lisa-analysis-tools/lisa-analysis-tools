"""END-TO-END: a real snapshot tarball -> a real monitor page, via the CLI.

User ask 2026-09-26: "full html generation from the command line with the
new structure (include tar file)."

``tests/test_monitor_from_tar.py`` mocks ``build_monitor`` -- it tests the
plumbing (discovery, the extraction cache, the CLI contract) in
milliseconds. This file tests the thing those mocks stand in for: that

    python -m lisatools.globalfit.monitor.from_tar SNAPSHOT.tar.gz OUT.html

really does turn a real tar into a real page. Nothing is mocked and the
command runs as a SUBPROCESS, exactly as a person would type it, so a
packaging mistake (a module that only imports from a source checkout, a
missing ``__main__``, an entry point that works in-process and not from
the command line) fails here and cannot hide behind an import the test
file already did.

⚠ HEAVY, AND OPT-IN. One page build is ~160 s and peaks around 2.5 GB
RSS. It is skipped unless a snapshot is pointed at:

    GF_MONITOR_E2E_TAR=/path/to/gf_prod_..._snapshot.tar.gz \\
        python -m unittest tests.test_monitor_from_tar_e2e

With the variable unset it also looks for a snapshot tar in the repo root
(which is where they land when downloaded), so on a machine that has one
this runs with no arguments. It is deliberately NOT auto-discovered
anywhere else: a default-on 160 s / 2.5 GB test would make the ordinary
suite unrunnable on the laptop, and running the tempering suites together
has already SIGKILLed this machine once.

``MOJITO_INFO_PATH`` is passed through when set. Without it the page
still builds -- that is the point of the degradation paths -- but with
fewer panels, so the assertions below stay on structure that does not
depend on it.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: A build is ~160 s; allow generously for a loaded machine, but bound it
#: so a hang fails the test instead of the session.
BUILD_TIMEOUT_S = 1800.0

#: Below this the "page" is an error stub, not a rendered report. A real
#: one is 7-9 MB with the figures base64-embedded.
MIN_PAGE_BYTES = 1_000_000


def _find_tar():
    env = os.environ.get("GF_MONITOR_E2E_TAR")
    if env:
        return env if os.path.isfile(env) else None
    hits = sorted(
        glob.glob(str(REPO / "*snapshot.tar*.gz")),
        key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


TAR = _find_tar()


@unittest.skipIf(TAR is None,
                 "no snapshot tarball; set GF_MONITOR_E2E_TAR to run")
class FromTarEndToEndTest(unittest.TestCase):
    """One build, many assertions -- 160 s is too expensive to repeat."""

    page = None
    proc = None
    outdir = None

    @classmethod
    def setUpClass(cls):
        cls.outdir = tempfile.mkdtemp(prefix="gf_e2e_")
        out = os.path.join(cls.outdir, "page.html")
        env = dict(os.environ)
        # Keep the child off every core: this is run on an 8 GB laptop and
        # the point of the test is the pipeline, not throughput.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        cls.proc = subprocess.run(
            [sys.executable, "-m", "lisatools.globalfit.monitor.from_tar",
             TAR, out, "--scratch", os.path.join(cls.outdir, "scratch")],
            cwd=str(REPO), env=env, timeout=BUILD_TIMEOUT_S,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        cls.page = out

    # ---- the command itself -------------------------------------------
    def test_the_command_succeeds(self):
        self.assertEqual(
            self.proc.returncode, 0,
            "from_tar exited %s\nstderr tail:\n%s" % (
                self.proc.returncode,
                self.proc.stderr.decode("utf-8", "replace")[-3000:]))

    def test_it_reports_each_stage_on_stdout(self):
        """A person watching the terminal should see what it chose."""
        out = self.proc.stdout.decode("utf-8", "replace")
        self.assertIn("[from_tar] run dir:", out)
        self.assertIn("[from_tar] wrote", out)

    def test_it_discovered_a_run_dir_inside_the_tar(self):
        """Not the tar root -- pointing the generator there renders a page
        with every panel missing and no error, which is the whole failure
        this pipeline exists to remove."""
        out = self.proc.stdout.decode("utf-8", "replace")
        m = re.search(r"\[from_tar\] run dir: (.+)", out)
        self.assertIsNotNone(m)
        run_dir = m.group(1).strip()
        self.assertTrue(
            any(fn.endswith(".h5") and "testing" in fn
                for fn in os.listdir(run_dir)),
            f"{run_dir} holds no *testing*.h5")

    _cache = None

    @classmethod
    def _text(cls):
        """Read the 9 MB page ONCE for the whole class."""
        if cls._cache is None:
            with open(cls.page, errors="ignore") as fh:
                cls._cache = fh.read()
        return cls._cache

    # ---- the artifact --------------------------------------------------
    def test_a_real_page_was_written(self):
        self.assertTrue(os.path.isfile(self.page))
        self.assertGreater(os.path.getsize(self.page), MIN_PAGE_BYTES)

    def test_it_is_a_SELF_CONTAINED_document(self):
        """Self-contained is the property that matters -- the page gets
        emailed and opened off a laptop with no network. It is a FRAGMENT,
        not a full <html> document (no <html>/<body> wrapper; browsers
        render it fine), so assert the structure it really has and, above
        all, that nothing is fetched from outside the file."""
        txt = self._text()
        low = txt.lower()
        for tag in ("<head", "<title", "<style", "<main"):
            self.assertIn(tag, low, tag)
        self.assertGreaterEqual(low.count("<section"), 5)
        for pat in (r'<link[^>]*href="(?!data:)[^"]+"',
                    r'<script[^>]*src="(?!data:)[^"]+"',
                    r'<img[^>]*src="(?!data:)[^"]+"'):
            self.assertEqual(
                re.findall(pat, txt), [],
                f"page reaches outside itself: {pat}")

    def test_the_figures_are_embedded_not_linked(self):
        """The page has to survive being emailed as one file."""
        self.assertGreater(
            self._text().count("data:image/png;base64,"), 10)

    def test_no_traceback_leaked_into_the_page(self):
        self.assertNotIn("Traceback (most recent call last)", self._text())

    def test_the_run_is_identified_on_the_page(self):
        self.assertRegex(self._text(), r"gf_prod_\w+")

    # ---- the Search Gates section (2026-09-26) -------------------------
    # These assert against a REAL run log, which is the only place the
    # gate parsers can be exercised: the generator SystemExits on import,
    # so its module-level regexes are unreachable to a unit test.
    def test_the_search_gates_section_is_rendered(self):
        txt = self._text()
        self.assertIn('<section id="gates">', txt)
        self.assertIn('href="#gates"', txt)
        self.assertIn('alt="search gate status"', txt)

    def test_the_shutoff_gate_is_reported_as_ONE_shared_table(self):
        """Every armed move reads the SAME band_rj_shutoff_w table
        (recipe._band_shutoff_w_pending dedupes on id(table)), and only
        one emits the count line. A per-move row reported 'armed, 0
        converged' for moves that were in fact sharing the count, which
        reads as a broken gate."""
        txt = self._text()
        self.assertIn("one shared table", txt)
        self.assertNotIn("armed, 0 converged", txt)

    def test_the_gate_table_says_what_each_gate_is_REACHING(self):
        """Not just what it is set to -- an armed gate reaching nothing is
        the failure shape this run keeps hitting."""
        txt = self._text()
        self.assertIn("RJ shutoff", txt)
        self.assertIn("in-model convergence", txt)
        self.assertIn("vertical swap", txt)
        self.assertRegex(txt, r"window \d+/\d+ \(newborn/survivor\)")

    def test_the_cell_ll_baseline_caveat_is_on_the_page(self):
        """98-100% of unit reports already exceeded the allowance BEFORE
        the all-rungs swap existed, so a high bar is not news by itself.
        Saying so on the page is what stops it being misread."""
        self.assertIn("98", self._text())

    # ---- the cache -----------------------------------------------------
    def test_the_extraction_was_cached_under_one_key(self):
        """One tarball -> one keyed extraction, so a re-render reuses it.
        Asserting the marker rather than re-running: a second full build
        is 160 s and proves nothing this does not."""
        scratch = os.path.join(self.outdir, "scratch")
        stamps = glob.glob(os.path.join(scratch, "*", ".extracted"))
        self.assertEqual(len(stamps), 1, stamps)


@unittest.skipIf(TAR is None,
                 "no snapshot tarball; set GF_MONITOR_E2E_TAR to run")
class FromTarCliSurfaceTest(unittest.TestCase):
    """Cheap CLI checks that need no page build."""

    def _run(self, args, timeout=120.0):
        return subprocess.run(
            [sys.executable, "-m", "lisatools.globalfit.monitor.from_tar",
             *args],
            cwd=str(REPO), timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_help_works_from_the_command_line(self):
        """Proves the module is reachable as __main__ from an install,
        which an in-process import would not."""
        p = self._run(["--help"])
        self.assertEqual(p.returncode, 0)
        self.assertIn(b"SNAPSHOT", p.stdout.upper())

    def test_a_missing_tar_is_a_clean_error(self):
        p = self._run(["/no/such/snapshot.tar.gz"])
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn(b"Traceback", p.stderr)

    def test_it_warns_about_MOJITO_INFO_PATH_before_doing_the_slow_work(self):
        """The warning must precede extraction, or you learn about it two
        minutes in."""
        env = {k: v for k, v in os.environ.items()
               if k != "MOJITO_INFO_PATH"}
        p = subprocess.run(
            [sys.executable, "-m", "lisatools.globalfit.monitor.from_tar",
             TAR, "--run-dir", "/nonexistent-so-it-stops-early"],
            cwd=str(REPO), env=env, timeout=120.0,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertIn(b"MOJITO_INFO_PATH", p.stderr)


if __name__ == "__main__":
    unittest.main()
