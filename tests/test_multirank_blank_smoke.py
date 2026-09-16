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

    def _run_world(self, size):
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import prepare_rank
        from lisatools.globalfit.run import GlobalFit

        def fn(rank, comm):
            fit = self._make_fit(f"n{size}")
            layout = prepare_rank(fit, comm)
            fit.build()
            gf = GlobalFit(fit, comm)
            gf.run_global_fit()
            out = {
                "role": layout.role_of(rank).value,
                "block": layout.block_of(rank),
                "acs_rows": int(gf.acs.acs_total_entries),
                "nwalkers_state": int(gf.state.branches["line"].nwalkers),
            }
            if hasattr(gf, "compute_service"):
                out["served"] = gf.compute_service_served
            return out

        return FakeWorld(size, timeout=600.0).run(fn)

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


if __name__ == "__main__":
    unittest.main()
