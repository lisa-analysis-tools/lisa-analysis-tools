"""The page + snapshot hook on the saver rank, and its watchdog.

The saver rank is the run's ONLY writer, and the sampler's ``save_step``
is a BLOCKING send to it. Anything bolted on there has to satisfy four
things, and each gets tests here:

  1. it is OFF unless the launcher asks for it;
  2. a queued save always wins -- the page can never delay the science
     by going first;
  3. it cannot raise into the saver loop, whatever fails;
  4. when it DOES hold things up, it says so loudly -- and if it ever
     costs stored iterations it turns itself off.

Plus the containerization contract: the packaged entry point must
reproduce the page the old script produced. That is verified end to end
elsewhere (measured byte-identical, 2026-09-26); what is pinned here is
the mechanism that makes it reproducible -- a FRESH interpreter -- since
an in-process run demonstrably restyles one panel.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from unittest import mock

from lisatools.globalfit import hdfbackend as hb
from lisatools.globalfit import monitor as mon
from lisatools.globalfit.monitor import hooks as mh


class _Comm:
    """Minimal MPI stand-in with a scripted iprobe.

    ``iprobe`` is consulted several times per pass and the calls mean
    different things -- "is there more to drain", then "is a save
    waiting, so skip the page", then "did one arrive while we built".
    A single flag cannot express that, so answers are scripted and
    anything past the end is False.
    """

    def __init__(self, payloads, probes=()):
        self._payloads = list(payloads)
        self._probes = list(probes)

    def recv(self, source=None):
        return self._payloads.pop(0)

    def iprobe(self, source=None):
        return self._probes.pop(0) if self._probes else False


class _Reader:
    def __init__(self, filename="/run/dir/gf_prod.h5"):
        self.filename = filename
        self.calls = 0

    def save_step_main(self, *a, **k):
        self.calls += 1


def _payloads(n=1):
    return ([{"save_args": (), "save_kwargs": {}}] * n) + [{"finish_run": True}]


def _run_loop(comm, reader, **kwargs):
    with mock.patch.object(hb, "_atomic_backup_copy"):
        hb.save_to_backend_asynchronously_and_plot(
            reader, comm, main_rank=0, plot_container=None, **kwargs)


class HookGateTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("GF_MONITOR_AFTER_SAVE", "GF_MONITOR_ITER",
                  "GF_MONITOR_SNAPSHOT"):
            os.environ.pop(k, None)
        self.addCleanup(self.env.stop)

    def test_it_is_OFF_by_default(self):
        """An upgrade must not silently put a ~140 s / 2.5 GB subprocess
        on the run's only writer."""
        reader = _Reader()
        with mock.patch.object(mon, "build_monitor") as bm:
            _run_loop(_Comm(_payloads()), reader)
        bm.assert_not_called()
        self.assertEqual(reader.calls, 1)      # the save still happened

    def test_it_fires_when_the_launcher_asks(self):
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        os.environ["GF_MONITOR_SNAPSHOT"] = "0"
        with mock.patch.object(mon, "build_monitor") as bm:
            _run_loop(_Comm(_payloads()), _Reader())
        bm.assert_called_once()
        self.assertEqual(bm.call_args[0][0], "/run/dir")

    def test_a_queued_save_wins(self):
        """probe #1 drains, probe #2 is the hook's own check."""
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        with mock.patch.object(mon, "build_monitor") as bm:
            _run_loop(_Comm(_payloads(), probes=[False, True]), _Reader())
        bm.assert_not_called()

    def test_the_cadence_knob_is_honoured(self):
        """GF_MONITOR_ITER=3 over seven saves in ONE loop -> builds after
        the 3rd and 6th."""
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        os.environ["GF_MONITOR_SNAPSHOT"] = "0"
        os.environ["GF_MONITOR_ITER"] = "3"
        reader = _Reader()
        seen = []
        with mock.patch.object(mon, "build_monitor",
                               side_effect=lambda *a, **k: seen.append(
                                   reader.calls)):
            _run_loop(_Comm(_payloads(7)), reader)
        self.assertEqual(reader.calls, 7)
        self.assertEqual(seen, [3, 6])

    def test_cadence_zero_does_not_divide_by_zero(self):
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        os.environ["GF_MONITOR_SNAPSHOT"] = "0"
        os.environ["GF_MONITOR_ITER"] = "0"
        with mock.patch.object(mon, "build_monitor") as bm:
            _run_loop(_Comm(_payloads()), _Reader())
        bm.assert_called_once()

    def test_the_snapshot_is_built_too_and_by_default(self):
        """User spec: 'production of tar file and html'. The tar is ON
        once the hook is on -- it is half of what was asked for."""
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        from lisatools.globalfit.monitor import snapshot as snap
        with mock.patch.object(mon, "build_monitor"), \
                mock.patch.object(snap, "build_snapshot") as bs:
            _run_loop(_Comm(_payloads()), _Reader())
        bs.assert_called_once_with("/run/dir")

    def test_a_failing_build_never_reaches_the_saver_loop(self):
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        reader = _Reader()
        with mock.patch.object(mon, "build_monitor",
                               side_effect=RuntimeError("boom")):
            _run_loop(_Comm(_payloads()), reader)   # must not raise
        self.assertEqual(reader.calls, 1)


class WatchdogTest(unittest.TestCase):
    """Signal 1/2/3 of 'is it holding anything up'."""

    def _wd(self, interval=None):
        w = mh.MonitorWatchdog()
        w.save_interval = interval
        return w

    def test_a_cheap_build_with_nothing_queued_is_quiet(self):
        w = self._wd(interval=5000.0)
        with self.assertNoLogs("lisatools.globalfit.monitor.hooks", "WARNING"):
            w.report(140.0, queued_after=False, dropped_delta=0)
        self.assertEqual(w.delayed, 0)

    def test_a_save_waiting_on_us_warns(self):
        w = self._wd(interval=5000.0)
        with self.assertLogs("lisatools.globalfit.monitor.hooks",
                             "WARNING") as cm:
            w.report(140.0, queued_after=True, dropped_delta=0)
        self.assertIn("HOLDING THE SAVER", "\n".join(cm.output))
        self.assertEqual(w.delayed, 1)

    def test_eating_the_save_interval_warns_even_with_nothing_queued(self):
        """The next one probably WILL wait. 140 s against a 200 s
        interval is 70%, well over the 25% default."""
        w = self._wd(interval=200.0)
        with self.assertLogs("lisatools.globalfit.monitor.hooks",
                             "WARNING") as cm:
            w.report(140.0, queued_after=False, dropped_delta=0)
        self.assertIn("70% of the 200 s save interval", "\n".join(cm.output))

    def test_dropping_a_state_is_an_escalation_AND_a_shutdown(self):
        """Dropped states are stored iterations that no longer exist.
        Never a trade worth making twice."""
        w = self._wd(interval=200.0)
        with self.assertLogs("lisatools.globalfit.monitor.hooks",
                             "WARNING") as cm:
            w.report(300.0, queued_after=True, dropped_delta=2)
        self.assertIn("STOPPING", "\n".join(cm.output))
        self.assertIsNotNone(w.disabled_reason)

    def test_a_disabled_watchdog_never_builds_again(self):
        w = self._wd()
        w.report(300.0, queued_after=True, dropped_delta=1)
        with mock.patch.dict(os.environ, {"GF_MONITOR_AFTER_SAVE": "1"}):
            self.assertFalse(w.should_build(1))
            self.assertFalse(w.should_build(999))

    def test_the_interval_is_measured_not_guessed(self):
        w = mh.MonitorWatchdog()
        self.assertIsNone(w.save_interval)
        w.note_save()
        self.assertIsNone(w.save_interval)      # one save is not a gap
        w.note_save()
        self.assertIsNotNone(w.save_interval)

    def test_the_interval_is_recorded_even_when_the_hook_is_OFF(self):
        """So turning it on mid-campaign has a real denominator."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GF_MONITOR_AFTER_SAVE", None)
            reader = _Reader()
            with mock.patch.object(mh.MonitorWatchdog, "note_save",
                                   autospec=True) as ns:
                _run_loop(_Comm(_payloads(3)), reader)
            self.assertEqual(ns.call_count, 3)


class FreshInterpreterTest(unittest.TestCase):
    """The containerization contract.

    An in-process run measurably restyles the 'instrument noise
    posteriors' panel, because importing lisatools pulls eryn's
    plt.style.use(["science"]) BEFORE the generator's own rcParams block
    instead of after. A child interpreter restores the original order.
    """

    def test_build_monitor_spawns_a_child_running_the_generator_FILE(self):
        with mock.patch.object(mon.subprocess, "run") as run, \
                mock.patch.object(mon.os, "replace"):
            mon.build_monitor("/run/dir", "/tmp/p.html", check=False)
        argv = run.call_args[0][0]
        self.assertEqual(argv[0], mon.sys.executable)
        self.assertTrue(argv[1].endswith("_generator.py"), argv)
        self.assertEqual(argv[2], "/run/dir")

    def test_it_never_imports_the_generator(self):
        """Importing it raises the SystemExit(0) its guard exists to
        raise; the path is resolved from this package's __file__."""
        import sys as _sys
        self.assertNotIn("lisatools.globalfit.monitor._generator",
                         _sys.modules)
        self.assertTrue(os.path.exists(mon.generator_path()))

    def test_the_page_is_published_atomically(self):
        with mock.patch.object(mon.subprocess, "run") as run, \
                mock.patch.object(mon.os, "replace") as rep:
            mon.build_monitor("/run/dir", "/tmp/p.html", check=False)
        self.assertTrue(run.call_args[0][0][3].endswith(".tmp"))
        rep.assert_called_once_with("/tmp/p.html.tmp", "/tmp/p.html")

    def test_a_failed_child_leaves_the_previous_page_alone(self):
        with mock.patch.object(
                mon.subprocess, "run",
                side_effect=subprocess.CalledProcessError(1, "c", stderr=b"x")), \
                mock.patch.object(mon.os, "replace") as rep:
            self.assertIsNone(
                mon.build_monitor("/run/dir", "/tmp/p.html", check=False))
        rep.assert_not_called()

    def test_a_hung_child_is_killed_and_does_not_raise(self):
        with mock.patch.object(
                mon.subprocess, "run",
                side_effect=subprocess.TimeoutExpired("c", 1)), \
                mock.patch.object(mon.os, "replace") as rep:
            self.assertIsNone(
                mon.build_monitor("/run/dir", "/tmp/p.html", check=False))
        rep.assert_not_called()


class MonitorLogsCopyIsInSyncTest(unittest.TestCase):
    """``_logs.discover_run_logs`` is a deliberate duplicate.

    The generator cannot import it -- that import pulls lisatools, whose
    chain restyles matplotlib before the generator's own rcParams block
    and changes the page. So the copy exists and this pins it.
    """

    @staticmethod
    def _fn_source(path):
        import ast
        with open(path) as fh:
            tree = ast.parse(fh.read())
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == "discover_run_logs":
                return ast.dump(ast.parse(ast.unparse(n)))
        return None

    def test_the_two_copies_are_identical(self):
        from lisatools.globalfit.monitor import _logs
        a = self._fn_source(_logs.__file__)
        b = self._fn_source(mon.generator_path())
        self.assertIsNotNone(a)
        self.assertIsNotNone(b, "the generator must keep its own copy")
        self.assertEqual(a, b)

    def test_the_generator_does_not_import_the_package_before_its_rcParams(self):
        """The regression this whole arrangement exists to prevent.

        AST, not a substring search: the comment ABOVE the function
        explains the trap in prose and contains the words "from
        lisatools", so a text match passes on the comment and proves
        nothing. (It did, on the first draft of this test.)
        """
        import ast
        with open(mon.generator_path()) as fh:
            src = fh.read()
        tree = ast.parse(src)
        rc_line = min(
            (n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and n.attr == "rcParams"),
            default=None)
        self.assertIsNotNone(rc_line, "the rcParams block moved or vanished")
        early = [
            n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            and n.lineno < rc_line
        ]
        names = []
        for n in early:
            if isinstance(n, ast.ImportFrom):
                names.append(n.module or "")
            else:
                names.extend(a.name for a in n.names)
        offenders = [m for m in names if m.split(".")[0] == "lisatools"]
        self.assertEqual(
            offenders, [],
            "importing lisatools before the generator's rcParams block "
            "lets eryn's plt.style.use(['science']) land FIRST, which "
            "changes the rendered page: %r" % (offenders,))


if __name__ == "__main__":
    unittest.main()
