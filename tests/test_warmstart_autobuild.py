"""Automatic warm-start component building (user ruling 2026-09-14).

"Can we build those into the 'def setup()' part of the refit proposal?
Check if it is done, if not run it? I would like it to be automatic."
-- the 6mo campaign launch FAILED on the missing refereed npz, so the
fit -> referee -> apply pipeline now runs automatically at recipe build
when GB_WARM_START_COMPONENTS is missing and GB_WARM_START_SOURCE_STORE
names the previous run's store. MPI-safe: builds happen on every rank,
so one rank wins a lock and the rest wait for the npz to appear.
"""

import os
import threading
import time
import unittest
from unittest import mock

from lisatools.globalfit.warmstart_build import ensure_warm_start_components


class _Runner:
    """Fake step runner: records commands, fabricates each step's output."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on  # substring of the script name to fail on

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        script = next(a for a in cmd if a.endswith(".py"))
        if self.fail_on and self.fail_on in os.path.basename(script):
            raise RuntimeError(f"fake failure in {script}")
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)
                if str(cmd[i]).startswith("--")}
        if "warmstart_fit_from_store" in script:
            with open(args["--out"], "w") as fh:
                fh.write("fit")
        elif "warmstart_match_referee" in script:
            ref = os.path.splitext(args["--npz"])[0] + "_referee.npz"
            with open(ref, "w") as fh:
                fh.write("referee")
        elif "warmstart_referee_apply" in script:
            with open(args["--out"], "w") as fh:
                fh.write("refereed")


class ExistingPathTest(unittest.TestCase):
    def test_existing_npz_short_circuits(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "refereed.npz")
            with open(path, "w") as fh:
                fh.write("x")
            runner = _Runner()
            got = ensure_warm_start_components(path, runner=runner)
            self.assertEqual(got, path)
            self.assertEqual(runner.calls, [])


class MissingStoreTest(unittest.TestCase):
    def test_missing_npz_without_store_raises_with_the_recipe(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "refereed.npz")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GB_WARM_START_SOURCE_STORE", None)
                with self.assertRaises(FileNotFoundError) as cm:
                    ensure_warm_start_components(path, runner=_Runner())
            msg = str(cm.exception)
            # the manual recipe must travel with the refusal
            for script in ("warmstart_fit_from_store.py",
                           "warmstart_match_referee.py",
                           "warmstart_referee_apply.py"):
                self.assertIn(script, msg)
            self.assertIn("GB_WARM_START_SOURCE_STORE", msg)

    def test_nonexistent_store_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "refereed.npz")
            with self.assertRaises(FileNotFoundError):
                ensure_warm_start_components(
                    path, store=os.path.join(d, "no_such_store.h5"),
                    runner=_Runner())


class BuildTest(unittest.TestCase):
    def _build(self, d, **kw):
        path = os.path.join(d, "warmstart", "gf_refereed.npz")
        store = os.path.join(d, "gf_prod_3mo_testing.h5")
        with open(store, "w") as fh:
            fh.write("store")
        runner = _Runner(**kw)
        return path, store, runner

    def test_builds_fit_referee_apply_in_order(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path, store, runner = self._build(d)
            got = ensure_warm_start_components(
                path, store=store, last_k=10, tobs=7776000.0, runner=runner)
            self.assertEqual(got, path)
            self.assertTrue(os.path.exists(path))
            # step commands are [script_path, *args] -- the scripts run
            # IN-PROCESS via import (user ruling 2026-09-14: "python
            # imports right? not like calling bash"), so no interpreter
            # element leads the command.
            for c in runner.calls:
                self.assertTrue(str(c[0]).endswith(".py"), c[0])
            names = [os.path.basename(next(a for a in c if a.endswith(".py")))
                     for c in runner.calls]
            self.assertEqual(names, ["warmstart_fit_from_store.py",
                                     "warmstart_match_referee.py",
                                     "warmstart_referee_apply.py"])
            fit_cmd, ref_cmd, apply_cmd = runner.calls
            self.assertIn("--store", fit_cmd)
            self.assertEqual(fit_cmd[fit_cmd.index("--store") + 1], store)
            self.assertEqual(fit_cmd[fit_cmd.index("--last-k") + 1], "10")
            self.assertEqual(
                float(fit_cmd[fit_cmd.index("--tobs") + 1]), 7776000.0)
            fit_out = fit_cmd[fit_cmd.index("--out") + 1]
            self.assertEqual(ref_cmd[ref_cmd.index("--npz") + 1], fit_out)
            self.assertEqual(ref_cmd[ref_cmd.index("--store") + 1], store)
            self.assertEqual(apply_cmd[apply_cmd.index("--fit") + 1], fit_out)
            self.assertEqual(
                apply_cmd[apply_cmd.index("--referee") + 1],
                os.path.splitext(fit_out)[0] + "_referee.npz")
            # the apply step wrote a TMP file that was renamed into place
            self.assertNotEqual(
                apply_cmd[apply_cmd.index("--out") + 1], path)
            # lock released
            self.assertFalse(os.path.exists(path + ".build.lock"))

    def test_step_failure_cleans_up_and_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path, store, runner = self._build(
                d, fail_on="warmstart_match_referee")
            with self.assertRaises(RuntimeError):
                ensure_warm_start_components(
                    path, store=store, runner=runner)
            self.assertFalse(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".build.lock"))

    def test_store_from_env(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path, store, runner = self._build(d)
            with mock.patch.dict(
                os.environ, {"GB_WARM_START_SOURCE_STORE": store}
            ):
                got = ensure_warm_start_components(path, runner=runner)
            self.assertEqual(got, path)
            self.assertEqual(len(runner.calls), 3)


class LockTest(unittest.TestCase):
    def test_waiter_returns_when_another_rank_finishes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "refereed.npz")
            store = os.path.join(d, "s.h5")
            with open(store, "w") as fh:
                fh.write("store")
            lock = path + ".build.lock"
            os.mkdir(lock)

            def winner():
                time.sleep(0.3)
                with open(path, "w") as fh:
                    fh.write("refereed")
                os.rmdir(lock)

            t = threading.Thread(target=winner)
            t.start()
            try:
                runner = _Runner()
                got = ensure_warm_start_components(
                    path, store=store, runner=runner, timeout=10.0,
                    poll=0.05)
            finally:
                t.join()
            self.assertEqual(got, path)
            self.assertEqual(runner.calls, [])  # the waiter never builds

    def test_waiter_times_out_loudly(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "refereed.npz")
            store = os.path.join(d, "s.h5")
            with open(store, "w") as fh:
                fh.write("store")
            lock = path + ".build.lock"
            os.mkdir(lock)
            with self.assertRaises(RuntimeError) as cm:
                ensure_warm_start_components(
                    path, store=store, runner=_Runner(), timeout=0.2,
                    poll=0.05)
            self.assertIn(lock, str(cm.exception))


class InProcessExecutionTest(unittest.TestCase):
    """The default step executor IMPORTS the script module and calls its
    ``main()`` inside this python process (sys.argv swapped) -- it never
    spawns a subprocess. Proven by pid identity."""

    def test_run_script_executes_in_this_process(self):
        import json
        import tempfile

        from lisatools.globalfit.warmstart_build import _run_script

        with tempfile.TemporaryDirectory() as d:
            script = os.path.join(d, "fake_step.py")
            out = os.path.join(d, "out.json")
            with open(script, "w") as fh:
                fh.write(
                    "import json, os, sys\n"
                    "def main():\n"
                    "    args = sys.argv[1:]\n"
                    "    out = args[args.index('--out') + 1]\n"
                    "    with open(out, 'w') as fh:\n"
                    "        json.dump({'pid': os.getpid(),"
                    " 'argv': args}, fh)\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                )
            import sys as _sys

            argv_before = list(_sys.argv)
            _run_script(script, ["--out", out, "--flag", "7"])
            with open(out) as fh:
                got = json.load(fh)
            self.assertEqual(got["pid"], os.getpid())  # in-process, no fork
            self.assertEqual(got["argv"], ["--out", out, "--flag", "7"])
            self.assertEqual(_sys.argv, argv_before)  # argv restored

    def test_run_script_raises_on_nonzero_exit(self):
        import tempfile

        from lisatools.globalfit.warmstart_build import _run_script

        with tempfile.TemporaryDirectory() as d:
            script = os.path.join(d, "fail_step.py")
            with open(script, "w") as fh:
                fh.write(
                    "import sys\n"
                    "def main():\n"
                    "    sys.exit(3)\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                )
            with self.assertRaises(RuntimeError):
                _run_script(script, [])


class RecipeWiringTest(unittest.TestCase):
    """The auto-build must sit at the shared _warm_path choke point in
    recipe.py -- BEFORE either twin's ``WarmStartComponents.from_npz`` --
    so both rj_warm_search and rj_warm_pe see the built npz."""

    def test_recipe_calls_ensure_before_from_npz(self):
        import inspect

        import lisatools.globalfit.recipe as recipe

        src = inspect.getsource(recipe)
        i_ensure = src.find("ensure_warm_start_components(")
        i_load = src.find("WarmStartComponents.from_npz(")
        self.assertGreater(i_ensure, 0, "recipe.py never calls the builder")
        self.assertGreater(i_load, 0)
        self.assertLess(i_ensure, i_load,
                        "builder must run before the npz is loaded")


if __name__ == "__main__":
    unittest.main()
