"""PSD fan-out end to end: the noise_only synthetic fit on 2 compute ranks (fake comm)."""

import os
import shutil
import tempfile
import unittest

import numpy as np

RUN_GF_SMOKE = os.environ.get("RUN_GF_SMOKE", "") not in ("", "0")

#: the noise_only stage runs psd_pe AND galfor_pe, so every stored iteration
#: sends the worker two proposes (plus the one start-up ping)
MOVES_PER_ITERATION = 2


@unittest.skipUnless(RUN_GF_SMOKE, "set RUN_GF_SMOKE=1 to run the multi-rank noise smoke")
class MultiRankNoiseSmokeTest(unittest.TestCase):
    WORLD_SIZE = 2  # head + one compute rank
    ITERATIONS = 2

    def setUp(self):
        os.environ.setdefault("USE_GPU", "0")
        os.environ.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_noise_")
        # the per-branch setup logs ignore file_store_dir and land in the cwd;
        # remember whether that directory was already there so tearDown only
        # removes one THIS test created
        self._stray = os.path.join(os.getcwd(), "gf_output_noise")
        self._stray_existed = os.path.isdir(self._stray)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        if not self._stray_existed:
            shutil.rmtree(self._stray, ignore_errors=True)

    def _run_world(self):
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import prepare_rank
        from lisatools.globalfit.run import GlobalFit
        from lisatools.globalfit.stock import erebor

        size = self.WORLD_SIZE

        def fn(rank, comm):
            fit = erebor.noise_only(
                nwalkers=4, ntemps=2, data_mode="synthetic",
                file_store_dir=os.path.join(self.tmpdir, f"n{size}"),
                make_diagnostic_plots=False,
            )
            fit.general.num_iterations = self.ITERATIONS
            layout = prepare_rank(fit, comm)
            fit.build()
            gf = GlobalFit(fit, comm)
            gf.run_global_fit()
            out = {"role": layout.role_of(rank).value, "acs_rows": int(gf.acs.acs_total_entries)}
            if hasattr(gf, "compute_service"):
                out["served"] = gf.compute_service_served
            else:
                out["log_like"] = np.array(gf.state.log_like[0], copy=True)
                out["betas"] = {
                    k: np.array(s.betas, copy=True)
                    for k, s in gf.state.sub_states.items()
                    if s is not None and getattr(s, "betas", None) is not None
                }
            return out

        return FakeWorld(size, timeout=1800.0).run(fn)

    def test_two_compute_ranks_run_the_noise_moves(self):
        out = self._run_world()
        self.assertEqual(out[0]["role"], "head")
        self.assertEqual(out[1]["role"], "compute")
        self.assertEqual((out[0]["acs_rows"], out[1]["acs_rows"]), (2, 2))
        # ping + one propose per noise move per stored iteration
        self.assertGreaterEqual(
            out[1]["served"], 1 + MOVES_PER_ITERATION * self.ITERATIONS
        )
        self.assertTrue(np.all(np.isfinite(out[0]["log_like"])))
        self.assertEqual(out[0]["log_like"].shape, (4,))
        for betas in out[0]["betas"].values():
            self.assertTrue(np.all(np.diff(betas) <= 0))  # a valid ladder after the head's step


if __name__ == "__main__":
    unittest.main()
