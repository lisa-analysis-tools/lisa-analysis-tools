"""Multi-rank control plane end to end on the blank synthetic fit (fake communicator)."""

import os
import shutil
import tempfile
import unittest

import numpy as np

RUN_GF_SMOKE = os.environ.get("RUN_GF_SMOKE", "") not in ("", "0")


def _count_move(model, state):
    _count_move.calls += 1
    return state, None


_count_move.calls = 0


@unittest.skipUnless(RUN_GF_SMOKE, "set RUN_GF_SMOKE=1 to run the multi-rank blank smoke")
class MultiRankBlankSmokeTest(unittest.TestCase):
    def setUp(self):
        _count_move.calls = 0
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_smoke_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fit(self, subdir):
        from eryn.prior import uniform_dist

        from lisatools.globalfit.stock import erebor

        fit = erebor.blank(
            nwalkers=4, ntemps=2, file_store_dir=os.path.join(self.tmpdir, subdir),
            make_diagnostic_plots=False,
        )
        fit.general.num_iterations = 3
        fit.add_branch(
            "line", ndim=2,
            priors={0: uniform_dist(0.0, 1.0), 1: uniform_dist(0.0, 1.0)},
            moves=[_count_move],
        )
        return fit

    def _run_world(self, size, subdir=None, env=None):
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import prepare_rank
        from lisatools.globalfit.run import GlobalFit

        store = subdir if subdir is not None else f"n{size}"

        def fn(rank, comm):
            fit = self._make_fit(store)
            layout = prepare_rank(fit, comm)
            fit.build()
            gf = GlobalFit(fit, comm)
            gf.run_global_fit()
            out = {
                "role": layout.role_of(rank).value,
                "block": layout.block_of(rank),
                # ``prepare_main`` publishes these LAST, so a head that
                # returned early (NULL_CHECK_ONLY) has neither
                "acs_rows": (int(gf.acs.acs_total_entries)
                             if hasattr(gf, "acs") else None),
                "nwalkers_state": (int(gf.state.branches["line"].nwalkers)
                                   if hasattr(gf, "state") else None),
                "null_check": bool(getattr(gf, "_null_check_only", False)),
                "has_sampler": hasattr(gf, "sampler"),
            }
            if hasattr(gf, "compute_service"):
                out["served"] = gf.compute_service_served
            return out

        # FakeWorld ranks are THREADS of this process and share os.environ, so
        # a per-rank write would race: set it outside and restore after.
        saved = {k: os.environ.get(k) for k in (env or {})}
        os.environ.update(env or {})
        try:
            return FakeWorld(size, timeout=600.0).run(fn)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_single_rank_path(self):
        out = self._run_world(1)
        self.assertEqual(out[0]["role"], "head")
        self.assertEqual(out[0]["acs_rows"], 4)
        self.assertGreaterEqual(_count_move.calls, 3)

    def test_two_compute_ranks_share_the_walkers(self):
        from lisatools.globalfit.hdfbackend import GFHDFBackend

        out = self._run_world(2)
        self.assertEqual(out[0]["role"], "head")
        self.assertEqual(out[1]["role"], "compute")
        self.assertEqual(out[0]["block"], (0, 2))
        self.assertEqual(out[1]["block"], (2, 4))
        self.assertEqual(out[0]["acs_rows"], 2)
        self.assertEqual(out[1]["acs_rows"], 2)
        self.assertEqual(out[0]["nwalkers_state"], 4)
        self.assertGreaterEqual(out[1]["served"], 1)  # the startup ping
        store = os.path.join(self.tmpdir, "n2")
        h5 = [f for f in os.listdir(store) if f.endswith(".h5")]
        self.assertTrue(h5)
        reader = GFHDFBackend(os.path.join(store, h5[0]))
        self.assertGreaterEqual(reader.iteration, 1)

    def test_null_check_only_releases_every_rank(self):
        """``NULL_CHECK_ONLY=1`` at size 3: measure the lnL, then everyone exits.

        The dev merge added ``fanout.stop()`` to that early path, and it is
        load-bearing: without it the compute rank stays parked in
        ``ComputeService.serve()`` on the fan-out communicator (the bare
        COMM_WORLD ``"stop"`` that releases legacy spares never reaches it)
        and a real job hangs until its wall clock runs out. Nothing else in
        the suite runs a MULTI-RANK null check end to end -- the merge report
        flagged exactly this gap (concern 2). Size 3 = head + one compute rank
        + saver; a hang fails on ``FakeWorld``'s own timeout.
        """
        out = self._run_world(3, subdir="null3", env={"NULL_CHECK_ONLY": "1"})
        self.assertEqual(
            (out[0]["role"], out[1]["role"], out[2]["role"]),
            ("head", "compute", "saver"))
        # the head stopped after the initial-lnL print: prepare_main returned
        # before it built (and published) the sampler
        self.assertTrue(out[0]["null_check"])
        self.assertFalse(out[0]["has_sampler"])
        # the compute rank's serve() RETURNED -- i.e. it was sent the fan-out
        # STOP. It answered nothing: the head's ``ping`` comes after the early
        # return, so this is 0 commands, not "the ping and then stop".
        self.assertIn("served", out[1])
        self.assertEqual(out[1]["served"], 0)
        # and the saver finished off its {"finish_run": True}
        self.assertEqual(out[2]["role"], "saver")


if __name__ == "__main__":
    unittest.main()
