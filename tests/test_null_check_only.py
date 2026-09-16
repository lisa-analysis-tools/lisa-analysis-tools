"""NULL_CHECK_ONLY: stop the run at the initial-lnL print, cleanly.

The PER-SOURCE truth-injection null test (user ask 2026-09-15, decomposing
the combined -325.62 over 18 sources into one number per source) launches one
job per (branch, id) pair whose ENTIRE output is run.py's

    initial log likelihood (after recipe setup): <value>

There is nothing to sample afterwards, so the job should end there rather
than burn spot allocation on iterations nobody reads.

WHY THIS CANNOT BE A ``sys.exit``. The production layout is ``mpiexec -n 3``:
main + a dedicated saver rank (blocked in ``save_to_backend_asynchronously
_and_plot``'s ``comm.recv``) + a spare (blocked in ``comm.recv`` waiting for
its startup ``"stop"``). A bare exit on the main rank leaves both of them
parked in ``recv`` until walltime -- the opposite of the point. So the knob
routes through the SAME shutdown path a normal finish uses:

* ``prepare_main`` releases the legacy spares with the ordinary ``"stop"``
  sends (factored out as ``_stop_spare_ranks`` so the early return still does
  it) AND, under the walker-block layout, the compute ranks parked in
  ``ComputeService.serve()`` via the guarded ``fanout.stop()``;
* ``run_global_fit`` skips ``run_mcmc`` + the submission writer and falls
  through to the ordinary ``{"finish_run": True}`` send to the saver.

Every assertion here is construction-level -- a fake ``self`` and a stub
backend, no data load, no engine build (the RunValidateGateTest pattern from
test_midit_checkpoint.py).

The id restriction the launcher rides on (``SOBHB_IDS`` / ``MBHB_IDS`` /
``EMRI_IDS`` -> ONE id) is asserted at the parsing level here; its landing on
``general.mojito_source_ids`` -- the single list that drives BOTH the
injected streams and the sampled branch -- is covered by
test_staged_sources_wiring.py.
"""

import importlib.util
import os
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "fstat_proposal" / "run_combined_staged.py"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "run_combined_staged_for_null_check_test", str(DRIVER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _EnvClean(unittest.TestCase):
    """Every test runs with NULL_CHECK_ONLY and the id envs restored."""

    ENVS = ("NULL_CHECK_ONLY", "SOBHB_IDS", "MBHB_IDS", "EMRI_IDS")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.ENVS}
        for k in self.ENVS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class NullCheckOnlyEnvTest(_EnvClean):
    """The knob's spelling and truthiness (rule 0: capitalized attr name)."""

    def _flag(self):
        from lisatools.globalfit.run import null_check_only

        return null_check_only()

    def test_unset_is_off(self):
        self.assertFalse(self._flag())

    def test_empty_is_off(self):
        # A present-but-empty knob counts as unset everywhere else in the
        # settings tree (stock/base._env_lookup), so it must here too --
        # `NULL_CHECK_ONLY= sbatch ...` should not silently skip the run.
        os.environ["NULL_CHECK_ONLY"] = ""
        self.assertFalse(self._flag())

    def test_false_spellings_are_off(self):
        for raw in ("0", "false", "False", "no", "off"):
            with self.subTest(raw=raw):
                os.environ["NULL_CHECK_ONLY"] = raw
                self.assertFalse(self._flag())

    def test_true_spellings_are_on(self):
        for raw in ("1", "true", "True", "yes", "on"):
            with self.subTest(raw=raw):
                os.environ["NULL_CHECK_ONLY"] = raw
                self.assertTrue(self._flag())


class _FakeComm:
    def __init__(self):
        self.sent = []

    def send(self, obj, dest=None):
        self.sent.append((dest, obj))


class StopSpareRanksTest(unittest.TestCase):
    """The factored-out spare release must keep the ORIGINAL semantics.

    Under the walker-block layout the set it iterates is
    ``GlobalFit.ranks_to_give`` -- the layout's ``RankRole.SPARE`` ranks,
    which only the legacy layout (``GF_LEGACY_RANK_LAYOUT=1``) populates.
    A COMPUTE rank must never appear here: it is parked in
    ``ComputeService.serve()`` on the fan-out communicator and a bare
    ``"stop"`` string on COMM_WORLD would never reach it.
    """

    def _fake_self(self, spares=(2,)):
        return types.SimpleNamespace(
            ranks_to_give=list(spares),
            comm=_FakeComm(),
        )

    def test_stops_only_the_roleless_ranks(self):
        from lisatools.globalfit.run import GlobalFit

        me = self._fake_self(spares=(2,))
        GlobalFit._stop_spare_ranks(me)
        self.assertEqual(me.comm.sent, [(2, "stop")])

    def test_no_spares_sends_nothing(self):
        from lisatools.globalfit.run import GlobalFit

        me = self._fake_self(spares=())
        GlobalFit._stop_spare_ranks(me)
        self.assertEqual(me.comm.sent, [])

    def test_it_reads_ranks_to_give_and_not_the_compute_ranks(self):
        """Non-vacuous: a fake carrying BOTH sets must use the spare one.

        ``used_ranks``/``all_ranks`` (the pre-merge dev spelling) would have
        made every non-used rank a target, which under the walker-block
        layout means sending a bare COMM_WORLD ``"stop"`` to a compute rank
        that is listening on the FAN-OUT communicator instead -- a hang.
        """
        from lisatools.globalfit.run import GlobalFit

        me = types.SimpleNamespace(
            ranks_to_give=[],            # walker-block layout: no spares
            compute_ranks=(2, 3),
            all_ranks=[0, 1, 2, 3],
            used_ranks=[0, 1],
            comm=_FakeComm(),
        )
        GlobalFit._stop_spare_ranks(me)
        self.assertEqual(me.comm.sent, [])

    def test_null_check_exit_also_releases_the_fanout(self):
        """prepare_main's NULL_CHECK_ONLY exit must stop the compute ranks.

        They are parked in ``ComputeService.serve()``, which only the
        fan-out STOP command releases; ``_stop_spare_ranks`` cannot reach
        them. Source-level pin (the exit is mid-way through a build-heavy
        method), asserting both releases sit in that block.
        """
        import inspect

        from lisatools.globalfit.run import GlobalFit

        src = inspect.getsource(GlobalFit.prepare_main)
        head = src.index('if getattr(self, "_null_check_only", False):')
        block = src[head:src.index("return", head)]
        self.assertIn("self._stop_spare_ranks()", block)
        self.assertIn("self.fanout.stop()", block)
        self.assertIn('getattr(self, "fanout", None)', block)

    def test_the_shared_setup_path_only_raises_the_flag(self):
        """``_build_acs_and_recipe`` must NOT act on the knob.

        Regression guard for the dev->branch merge: dev carried the early
        return inside ``prepare_main``, but the branch factored that block
        into the setup path SHARED with ``prepare_compute``. A ``return``
        there breaks the ``(acs, like_mix)`` contract for both callers AND
        makes every compute rank skip ``ComputeService.serve()``, so the
        head's fan-out STOP would find nobody listening.
        """
        import inspect

        from lisatools.globalfit.run import GlobalFit

        src = inspect.getsource(GlobalFit._build_acs_and_recipe)
        head = src.index("if null_check_only():")
        # the block runs to the next top-level (8-space) statement
        tail = src.index("\n        # [layer-chi2 diag", head)
        block = src[head:tail]
        self.assertIn("self._null_check_only = True", block)
        self.assertNotIn("return", block)
        self.assertNotIn("_stop_spare_ranks", block)
        self.assertNotIn("fanout.stop", block)
        # and the method still ends by handing both callers the pair
        self.assertTrue(src.rstrip().endswith("return acs, like_mix"))


class RunGlobalFitNullCheckTest(_EnvClean):
    """The main-rank path: skip sampling, still finish the helper ranks.

    Paired negative control (feedback_paired_negative_controls): the SAME
    fake self with the flag unset must still call ``run_mcmc`` -- otherwise
    "no sampling happened" would prove nothing about the knob.
    """

    MAIN, SAVER = 0, 1

    def _fake_self(self, null_check):
        from lisatools.globalfit.communication.ranks import RankRole

        calls = []

        def _prepare_main():
            calls.append("prepare_main")
            # The real prepare_main sets this right after the initial-lnL
            # print (and has already released the spares AND the fan-out
            # compute ranks by then).
            if null_check:
                me._null_check_only = True

        def _run_mcmc(*args, **kwargs):
            calls.append("run_mcmc")

        def _write_submission():
            calls.append("write_submission")

        def _fanout_stop():
            calls.append("fanout_stop")

        me = types.SimpleNamespace(
            rank=self.MAIN,
            role=RankRole.HEAD,
            main_rank=self.MAIN,
            results_rank=self.SAVER,
            comm=_FakeComm(),
            progress=False,
            state=object(),
            prepare_main=_prepare_main,
            sampler=types.SimpleNamespace(run_mcmc=_run_mcmc),
            _write_submission=_write_submission,
            fanout=types.SimpleNamespace(stop=_fanout_stop),
            engine_info=types.SimpleNamespace(
                branch_backends=None, branch_states=None),
            curr=types.SimpleNamespace(
                general_info=types.SimpleNamespace(
                    main_file_path="/dev/null/unused.h5",
                    num_iterations=10,
                    submission_parent_folder=None,
                ),
                settings_dict=types.SimpleNamespace(
                    rank_info=types.SimpleNamespace(main_rank=self.MAIN)),
            ),
        )
        me._calls = calls
        return me

    def _run(self, null_check):
        import lisatools.globalfit.run as run_mod

        me = self._fake_self(null_check)
        real_backend = run_mod.GFHDFBackend
        run_mod.GFHDFBackend = lambda *a, **k: object()
        try:
            run_mod.GlobalFit.run_global_fit(me)
        finally:
            run_mod.GFHDFBackend = real_backend
        return me

    def test_knob_set_skips_sampling(self):
        os.environ["NULL_CHECK_ONLY"] = "1"
        me = self._run(null_check=True)
        self.assertIn("prepare_main", me._calls)
        self.assertNotIn("run_mcmc", me._calls)

    def test_knob_set_still_finishes_the_saver_rank(self):
        # THE hang guard: the saver blocks in recv until it is told the run
        # is over, so the early exit must still send it finish_run.
        os.environ["NULL_CHECK_ONLY"] = "1"
        me = self._run(null_check=True)
        self.assertEqual(me.comm.sent, [(self.SAVER, {"finish_run": True})])

    def test_knob_set_does_not_re_stop_the_fanout_here(self):
        # prepare_main already released it on the early return; the skip
        # branch must not reach the sampling path's finally at all.
        os.environ["NULL_CHECK_ONLY"] = "1"
        me = self._run(null_check=True)
        self.assertEqual(me._calls, ["prepare_main"])

    def test_control_unset_still_samples(self):
        me = self._run(null_check=False)
        self.assertEqual(
            me._calls,
            ["prepare_main", "run_mcmc", "write_submission", "fanout_stop"],
        )
        self.assertEqual(me.comm.sent, [(self.SAVER, {"finish_run": True})])

    def test_control_stops_the_fanout_even_when_run_mcmc_raises(self):
        # The `finally` is what lets the compute ranks exit instead of
        # blocking forever on a head exception (FakeWorld / in-process runs
        # have no MPI abort to end the job for them).
        import lisatools.globalfit.run as run_mod

        me = self._fake_self(null_check=False)

        def _boom(*args, **kwargs):
            me._calls.append("run_mcmc")
            raise RuntimeError("head blew up")

        me.sampler.run_mcmc = _boom
        with self.assertRaises(RuntimeError):
            run_mod.GlobalFit.run_global_fit(me)
        self.assertEqual(me._calls, ["prepare_main", "run_mcmc", "fanout_stop"])


class SingleSourceIdEnvTest(_EnvClean):
    """One source id at a time -- what the per-source launcher exports.

    ``_source_ids_from_env`` is the ONLY reader of the id envs, and its
    result both arms the branch and seeds ``general.mojito_source_ids``
    (run_combined_staged.py:548-556) -- the one list the injection loader
    and the sampled branch share. A single id in, a single id out, and the
    other two branches unarmed (= removed, so their streams never enter the
    data).
    """

    def test_single_branch_single_id(self):
        mod = _driver()
        os.environ["MBHB_IDS"] = "16"
        self.assertEqual(mod._source_ids_from_env(), {"mbh": [16]})

    def test_id_zero_is_armed_not_falsy(self):
        # `if ids:` on a LIST, not on the id -- sobbh 0 and emri 0 are both
        # real sources in the production arming.
        mod = _driver()
        os.environ["SOBHB_IDS"] = "0"
        self.assertEqual(mod._source_ids_from_env(), {"sobbh": [0]})

    def test_whitespace_and_trailing_comma_tolerated(self):
        mod = _driver()
        os.environ["EMRI_IDS"] = " 7 ,"
        self.assertEqual(mod._source_ids_from_env(), {"emri": [7]})

    def test_empty_env_arms_nothing(self):
        mod = _driver()
        os.environ["MBHB_IDS"] = ""
        self.assertEqual(mod._source_ids_from_env(), {})


if __name__ == "__main__":
    unittest.main()
