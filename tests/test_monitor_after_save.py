"""The HTML monitor page regenerated on the saver rank.

The saver rank is the run's ONLY writer. Anything bolted onto it has to
satisfy three things before it is allowed near production, and each one
gets a test here:

  1. it is OFF unless the launcher asks for it,
  2. a queued save always wins -- the page can never delay the science,
  3. it cannot raise, time out, or exit non-zero in a way that reaches
     the saver loop.

Plus the publish itself: the page is written to a temp and ``os.replace``d
into position, so a browser refreshing it never sees half a document.

Measured cost the design is sized against (2026-09-26, 6mo v9 snapshot,
job 638): 141 s wall, 2.46 GB peak RSS, 7.3 MB page.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from unittest import mock

from lisatools.globalfit import hdfbackend as hb


class _Comm:
    """Minimal MPI stand-in: a scripted recv queue and a scripted iprobe.

    ``iprobe`` is consulted twice per loop pass and the two calls mean
    different things -- first "is there more to drain", then "is a save
    already waiting, so skip the page". A single flag cannot express
    that, so the answers are scripted; anything past the end is False.
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


def _run_loop(comm, reader, **kwargs):
    """One save payload then finish, with the backup copy stubbed out."""
    with mock.patch.object(hb, "_atomic_backup_copy"):
        hb.save_to_backend_asynchronously_and_plot(
            reader, comm, main_rank=0, plot_container=None, **kwargs)


def _payloads(n=1):
    return ([{"save_args": (), "save_kwargs": {}}] * n) + [{"finish_run": True}]


class MonitorGateTest(unittest.TestCase):
    """Whether the hook fires at all."""

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("GF_MONITOR_AFTER_SAVE", "GF_MONITOR_ITER"):
            os.environ.pop(k, None)
        self.addCleanup(self.env.stop)

    def test_it_is_OFF_by_default(self):
        """The whole point of the default: an upgrade must not silently
        put a 141 s / 2.5 GB subprocess on the run's only writer."""
        reader = _Reader()
        with mock.patch.object(hb, "_regenerate_monitor_page") as gen:
            _run_loop(_Comm(_payloads()), reader)
        gen.assert_not_called()
        self.assertEqual(reader.calls, 1)   # the save still happened

    def test_it_fires_when_the_launcher_asks(self):
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        reader = _Reader()
        with mock.patch.object(hb, "_regenerate_monitor_page") as gen:
            _run_loop(_Comm(_payloads()), reader)
        gen.assert_called_once_with(reader.filename)

    def test_a_queued_save_wins(self):
        """iprobe True means another state is already waiting. Writing it
        comes first; the page waits for the next quiet gap."""
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        # probe #1 = the drain check (False, nothing queued yet);
        # probe #2 = the monitor's own check (True, a save arrived while
        # this pass was working). The page must yield to it.
        with mock.patch.object(hb, "_regenerate_monitor_page") as gen:
            _run_loop(_Comm(_payloads(), probes=[False, True]), _Reader())
        gen.assert_not_called()

    def test_the_cadence_knob_is_honoured(self):
        """GF_MONITOR_ITER=3 over SEVEN saves in ONE loop -> pages after
        the 3rd and 6th, not the others. Seven saves in one invocation,
        because the loop's counter restarts on every call."""
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        os.environ["GF_MONITOR_ITER"] = "3"
        reader = _Reader()
        seen = []
        with mock.patch.object(hb, "_regenerate_monitor_page",
                               lambda _p: seen.append(reader.calls)):
            _run_loop(_Comm(_payloads(7)), reader)
        self.assertEqual(reader.calls, 7)
        self.assertEqual(seen, [3, 6])

    def test_cadence_zero_does_not_divide_by_zero(self):
        """A launcher typo must not kill the saver."""
        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        os.environ["GF_MONITOR_ITER"] = "0"
        with mock.patch.object(hb, "_regenerate_monitor_page") as gen:
            _run_loop(_Comm(_payloads()), _Reader())
        gen.assert_called_once()


class MonitorPublishTest(unittest.TestCase):
    """:func:`_regenerate_monitor_page` itself."""

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("GF_MONITOR_OUT", "GF_MONITOR_SCRIPT", "GF_MONITOR_TIMEOUT"):
            os.environ.pop(k, None)
        self.addCleanup(self.env.stop)

    def test_it_writes_a_temp_and_replaces_it(self):
        """Never a half-written page: the generator's target is the temp,
        and only a clean exit promotes it."""
        os.environ["GF_MONITOR_SCRIPT"] = __file__      # any existing file
        with mock.patch.object(hb.subprocess, "run") as run, \
                mock.patch.object(hb.os, "replace") as rep:
            hb._regenerate_monitor_page("/run/dir/gf_prod.h5")
        argv = run.call_args[0][0]
        self.assertTrue(argv[-1].endswith(".tmp"), argv)
        self.assertEqual(argv[-2], "/run/dir")          # the run directory
        rep.assert_called_once_with(argv[-1], "/run/dir/gf_monitor.html")

    def test_a_failed_build_leaves_the_previous_page_alone(self):
        os.environ["GF_MONITOR_SCRIPT"] = __file__
        with mock.patch.object(
                hb.subprocess, "run",
                side_effect=subprocess.CalledProcessError(
                    1, "cmd", stderr=b"boom")), \
                mock.patch.object(hb.os, "replace") as rep:
            hb._regenerate_monitor_page("/run/dir/gf_prod.h5")
        rep.assert_not_called()

    def test_a_hung_build_is_killed_and_does_not_raise(self):
        os.environ["GF_MONITOR_SCRIPT"] = __file__
        with mock.patch.object(
                hb.subprocess, "run",
                side_effect=subprocess.TimeoutExpired("cmd", 1)), \
                mock.patch.object(hb.os, "replace") as rep:
            hb._regenerate_monitor_page("/run/dir/gf_prod.h5")
        rep.assert_not_called()

    def test_a_missing_generator_warns_and_returns(self):
        """A wheel install has no scripts/ directory. That must degrade to
        'no page', not to an exception on the writer rank."""
        os.environ["GF_MONITOR_SCRIPT"] = "/nope/does/not/exist.py"
        with mock.patch.object(hb.subprocess, "run") as run:
            hb._regenerate_monitor_page("/run/dir/gf_prod.h5")
        run.assert_not_called()

    def test_GF_MONITOR_OUT_overrides_the_destination(self):
        os.environ["GF_MONITOR_SCRIPT"] = __file__
        os.environ["GF_MONITOR_OUT"] = "/elsewhere/page.html"
        with mock.patch.object(hb.subprocess, "run"), \
                mock.patch.object(hb.os, "replace") as rep:
            hb._regenerate_monitor_page("/run/dir/gf_prod.h5")
        self.assertEqual(rep.call_args[0][1], "/elsewhere/page.html")

    def test_nothing_in_this_module_raises_out(self):
        """The catch-all arm. Any unexpected failure is a warning."""
        os.environ["GF_MONITOR_SCRIPT"] = __file__
        with mock.patch.object(
                hb.subprocess, "run", side_effect=RuntimeError("weird")):
            hb._regenerate_monitor_page("/run/dir/gf_prod.h5")


class MonitorScriptLookupTest(unittest.TestCase):
    def test_it_finds_the_real_generator_in_this_checkout(self):
        """The walk-up default has to actually work, or every source-tree
        run silently falls back to the not-found warning."""
        os.environ.pop("GF_MONITOR_SCRIPT", None)
        path = hb._monitor_script_path()
        self.assertIsNotNone(path)
        self.assertTrue(path.endswith("gf_monitor_gen.py"), path)
        self.assertTrue(os.path.exists(path))

    def test_an_env_override_that_does_not_exist_is_None_not_a_path(self):
        with mock.patch.dict(os.environ,
                             {"GF_MONITOR_SCRIPT": "/no/such/file.py"}):
            self.assertIsNone(hb._monitor_script_path())


if __name__ == "__main__":
    unittest.main()
