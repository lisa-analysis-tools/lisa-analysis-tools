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

from lisatools.globalfit.warmstart.build import ensure_warm_start_components


class _Runner:
    """Fake step runner: records (step, args), fabricates each output.

    Steps are INSTALLED-MODULE names (user ruling 2026-09-14: all the
    warm-start code lives in lisatools.globalfit.warmstart; the runner
    dispatches to those modules' main(argv), never to script files)."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on  # step name to fail on

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        step = cmd[0]
        if self.fail_on and self.fail_on == step:
            raise RuntimeError(f"fake failure in {step}")
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)
                if str(cmd[i]).startswith("--")}
        if step == "fit_from_store":
            with open(args["--out"], "w") as fh:
                fh.write("fit")
        elif step == "match_referee":
            ref = os.path.splitext(args["--npz"])[0] + "_referee.npz"
            with open(ref, "w") as fh:
                fh.write("referee")
        elif step == "referee_apply":
            with open(args["--out"], "w") as fh:
                fh.write("refereed")
        else:
            raise AssertionError(f"unknown step {step!r}")


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
            # the manual recipe must travel with the refusal -- the
            # installed-module commands (-m lisatools.globalfit.warmstart.*)
            for step in ("fit_from_store", "match_referee",
                         "referee_apply"):
                self.assertIn(
                    f"-m lisatools.globalfit.warmstart.{step}", msg)
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
            # step commands are [installed_module_step, *args] -- pure
            # python calls into lisatools.globalfit.warmstart (user
            # rulings 2026-09-14: in-process, and all warm-start code in
            # the package), no interpreter, no script paths.
            names = [c[0] for c in runner.calls]
            self.assertEqual(names, ["fit_from_store",
                                     "match_referee",
                                     "referee_apply"])
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
            path, store, runner = self._build(d, fail_on="match_referee")
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


class DefaultRunnerTest(unittest.TestCase):
    """The default step runner calls the INSTALLED modules' main(argv)
    directly -- a plain python function call inside this process."""

    def test_dispatches_to_installed_module_main(self):
        from lisatools.globalfit.warmstart import build as wb

        with mock.patch(
            "lisatools.globalfit.warmstart.fit_from_store.main",
            return_value=0,
        ) as m:
            wb._default_runner(["fit_from_store", "--store", "s.h5",
                                "--last-k", 10])
        m.assert_called_once_with(["--store", "s.h5", "--last-k", "10"])

    def test_nonzero_systemexit_becomes_runtimeerror(self):
        from lisatools.globalfit.warmstart import build as wb

        with mock.patch(
            "lisatools.globalfit.warmstart.referee_apply.main",
            side_effect=SystemExit(3),
        ):
            with self.assertRaises(RuntimeError):
                wb._default_runner(["referee_apply", "--fit", "f"])

    def test_zero_systemexit_is_success(self):
        from lisatools.globalfit.warmstart import build as wb

        with mock.patch(
            "lisatools.globalfit.warmstart.match_referee.main",
            side_effect=SystemExit(0),
        ):
            wb._default_runner(["match_referee", "--npz", "n"])


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
