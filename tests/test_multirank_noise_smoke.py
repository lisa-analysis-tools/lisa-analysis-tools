"""PSD fan-out end to end: noise_only synthetic fit, 1 vs 2 compute ranks (fake comm)."""

import os
import shutil
import tempfile
import unittest

import numpy as np

RUN_GF_SMOKE = os.environ.get("RUN_GF_SMOKE", "") not in ("", "0")


@unittest.skipUnless(RUN_GF_SMOKE, "set RUN_GF_SMOKE=1 to run the multi-rank noise smoke")
class MultiRankNoiseSmokeTest(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("USE_GPU", "0")
        os.environ.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_noise_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_world(self, size, iterations=2):
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import prepare_rank
        from lisatools.globalfit.run import GlobalFit
        from lisatools.globalfit.stock import erebor

        def fn(rank, comm):
            fit = erebor.noise_only(
                nwalkers=4, ntemps=2, data_mode="synthetic",
                file_store_dir=os.path.join(self.tmpdir, f"n{size}"),
                make_diagnostic_plots=False,
            )
            fit.general.num_iterations = iterations
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
        out = self._run_world(2)
        self.assertEqual(out[0]["role"], "head")
        self.assertEqual(out[1]["role"], "compute")
        self.assertEqual((out[0]["acs_rows"], out[1]["acs_rows"]), (2, 2))
        # ping + at least one propose per stored iteration per noise move
        self.assertGreaterEqual(out[1]["served"], 1 + 2)
        self.assertTrue(np.all(np.isfinite(out[0]["log_like"])))
        self.assertEqual(out[0]["log_like"].shape, (4,))
        for betas in out[0]["betas"].values():
            self.assertTrue(np.all(np.diff(betas) <= 0))  # a valid ladder after the head's step


if __name__ == "__main__":
    unittest.main()
