"""``GF_RESIDUAL_SNAPSHOT_EVERY``: the run writes the max-lnL walker's own
residual (data minus EVERY fitted signal) beside the store, so the monitor's
data / template / residual panels stop subtracting GB + VGB only
(Mike 2026-10-07: an MBH merger at 111 d showed as "residual" on the page
while the run had fitted it)."""
import inspect
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit import residual_snapshot as rs

SHAPE = (3, 4, 6)


class _Settings:
    Nf, Nt, data_dt = 16, 12, 2.5
    f_arr_edges = np.arange(5) * 0.1
    t_arr_edges = np.arange(7) * 10.0


class _AC:
    def __init__(self, arr):
        self.data = SimpleNamespace(arr=arr, settings=_Settings())


class _ACA:
    """Stand-in AnalysisContainerArray: one residual per walker."""

    def __init__(self, arrs, lls):
        self.acs = [_AC(a) for a in arrs]
        self.lls = np.asarray(lls, float)

    def __getitem__(self, i):
        return self.acs[i]

    def likelihood(self, complex=False):
        return self.lls


def _arrs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=SHAPE) + 10 * k for k in range(n)]


class _Layout:
    """4 walkers on 2 compute ranks (head = rank 0, worker = rank 1)."""
    worker_ranks = (1,)

    def owner_of(self, w):
        return (0, w) if w < 2 else (1, w - 2)


class _Fanout:
    """Runs the worker side through the real builtin, like ComputeService."""

    def __init__(self, head_acs, worker_acs, lls):
        self.layout = _Layout()
        self.head_acs, self.worker_acs, self.lls = head_acs, worker_acs, lls
        self.payloads = {}

    def gather_likelihood(self, acs):
        return np.asarray(self.lls, float)

    def run(self, op, *, move, per_rank_payload, local_body, merge):
        assert op == rs.RESIDUAL_SNAPSHOT_OP
        self.payloads = {0: per_rank_payload(0, 0, 2), 1: per_rank_payload(1, 2, 4)}
        worker_model = SimpleNamespace(analysis_container_arr=self.worker_acs)
        return merge({0: local_body(self.payloads[0], None),
                      1: rs.walker_residual(worker_model, self.payloads[1])})


class TakeSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "run_testing.h5")
        self.data = np.full(SHAPE, 7.0)
        self.holder = SimpleNamespace(data_res_arr=SimpleNamespace(arr=self.data))

    def test_single_process_writes_the_max_lnl_walkers_residual(self):
        arrs = _arrs(4)
        acs = _ACA(arrs, [3.0, 9.0, 1.0, np.nan])
        path = rs.take_snapshot(acs=acs, data_holder=self.holder,
                                store_path=self.store, iteration=41,
                                store_iteration0=707)
        self.assertEqual(path, os.path.join(self.tmp, "run_testing_residual_snapshot.npz"))
        with np.load(path) as z:
            self.assertEqual(int(z["walker"]), 1)
            self.assertEqual(int(z["iteration"]), 41)
            # the store row, not the job's own count (which restarts at 0
            # on every resume: a page read "iteration 161" at row ~870)
            self.assertEqual(int(z["store_row"]), 748)
            np.testing.assert_allclose(z["residual"], arrs[1].astype(np.float32))
            np.testing.assert_array_equal(z["data"], self.data.astype(np.float32))
            self.assertEqual(z["residual"].dtype, np.float32)
            np.testing.assert_array_equal(z["t_edges"], _Settings.t_arr_edges)
        self.assertFalse(any(f.endswith(".tmp.npz") for f in os.listdir(self.tmp)))

    def test_fanout_asks_only_the_owning_rank(self):
        arrs = _arrs(4, seed=1)
        head, worker = _ACA(arrs[:2], [0, 0]), _ACA(arrs[2:], [0, 0])
        fan = _Fanout(head, worker, lls=[1.0, 2.0, 5.0, 3.0])   # max = walker 2
        path = rs.take_snapshot(acs=head, data_holder=self.holder,
                                store_path=self.store, iteration=5, fanout=fan)
        self.assertIsNone(fan.payloads[0])
        self.assertEqual(fan.payloads[1], {"local": 0})
        with np.load(path) as z:
            self.assertEqual(int(z["walker"]), 2)
            np.testing.assert_allclose(z["residual"], arrs[2].astype(np.float32))

    def test_a_shape_mismatch_refuses(self):
        acs = _ACA(_arrs(1), [0.0])
        bad = SimpleNamespace(data_res_arr=SimpleNamespace(arr=np.zeros((3, 4, 5))))
        with self.assertRaises(RuntimeError):
            rs.take_snapshot(acs=acs, data_holder=bad, store_path=self.store,
                             iteration=0)


class CadenceTest(unittest.TestCase):
    def _kw(self, tmp):
        return dict(acs=_ACA(_arrs(2), [0.0, 1.0]),
                    data_holder=SimpleNamespace(
                        data_res_arr=SimpleNamespace(arr=np.zeros(SHAPE))),
                    store_path=os.path.join(tmp, "s.h5"))

    def test_every_third_iteration_by_default_and_zero_turns_it_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if k != rs.SNAPSHOT_EVERY_ENV}
            with mock.patch.dict(os.environ, env, clear=True):
                got = [rs.maybe_snapshot(iteration=i, **self._kw(tmp)) is not None
                       for i in range(7)]
            self.assertEqual(got, [False, False, True, False, False, True, False])
            with mock.patch.dict(os.environ, {rs.SNAPSHOT_EVERY_ENV: "0"}):
                self.assertIsNone(rs.maybe_snapshot(iteration=2, **self._kw(tmp)))

    def test_a_failure_never_reaches_the_run(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {rs.SNAPSHOT_EVERY_ENV: "1"}), \
                mock.patch.object(rs, "take_snapshot", side_effect=RuntimeError("x")):
            with self.assertLogs(rs.logger, level="WARNING"):
                self.assertIsNone(rs.maybe_snapshot(iteration=0, **self._kw(tmp)))


class WiringTest(unittest.TestCase):
    def test_recipe_hook_and_pickling(self):
        from lisatools.globalfit.recipe import Recipe

        src = inspect.getsource(Recipe.__call__)
        self.assertIn("maybe_snapshot(", src)
        r = Recipe()
        r._init_runtime()
        r.snapshot_context = dict(acs=object(), data_holder=None, store_path="x")
        self.assertNotIn("snapshot_context", pickle.loads(pickle.dumps(r)).__dict__)

    def test_compute_ranks_serve_the_builtin_and_the_head_gets_a_context(self):
        from lisatools.globalfit import run

        src = inspect.getsource(run)
        self.assertIn("RESIDUAL_SNAPSHOT_OP: lambda payload, clock, model: walker_residual(", src)
        self.assertIn("self.recipe.snapshot_context = dict(", src)
        self.assertIn('store_iteration0=int(getattr(self, "_resume_store_iteration", 0))', src)
        self.assertIn("self._resume_store_iteration = int(backend.iteration)", src)

    def test_short_tar_ships_the_snapshot(self):
        from lisatools.globalfit.monitor import snapshot as snap

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "run_testing_residual_snapshot.npz")
            np.savez(p, a=np.zeros(1))
            self.assertIn(p, snap._short_members(tmp, None, 1))


if __name__ == "__main__":
    unittest.main()
