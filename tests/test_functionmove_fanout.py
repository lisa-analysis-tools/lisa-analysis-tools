"""FunctionMove re-syncs log_like from ALL walker blocks under several compute ranks."""

import types
import unittest

import numpy as np

from lisatools.globalfit.moves.functionmove import FunctionMove


class _Acs:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def likelihood(self, complex=False):
        return self.values


class _Fanout:
    """Stands in for WalkerFanout: the head's block plus one fixed worker block."""

    def __init__(self, single, worker_values=()):
        self.single = single
        self.worker_values = list(worker_values)
        self.calls = 0

    def gather_likelihood(self, acs):
        self.calls += 1
        return np.concatenate([acs.likelihood(complex=False), self.worker_values])


def _fn(model, state):
    return state, None


class FunctionMoveFanoutTest(unittest.TestCase):
    def _propose(self, fanout, acs_values):
        move = FunctionMove(_fn, name="f")
        move.fanout = fanout
        model = types.SimpleNamespace(analysis_container_arr=_Acs(acs_values))
        state = types.SimpleNamespace(log_like=np.zeros((2, 4)))
        new_state, accepted = move.propose(model, state)
        return new_state.log_like, accepted

    def test_no_fanout_reads_the_full_aca_directly(self):
        ll, accepted = self._propose(None, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(ll, [[1.0, 2.0, 3.0, 4.0]] * 2)
        self.assertEqual(accepted.shape, (2, 4))

    def test_single_compute_rank_reads_the_aca_directly(self):
        fo = _Fanout(single=True)
        ll, _ = self._propose(fo, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(ll, [[1.0, 2.0, 3.0, 4.0]] * 2)
        self.assertEqual(fo.calls, 0)

    def test_several_compute_ranks_gather_every_block(self):
        # the head's ACA holds walkers 0-1 only; the worker holds 2-3
        fo = _Fanout(single=False, worker_values=[3.0, 4.0])
        ll, _ = self._propose(fo, [1.0, 2.0])
        np.testing.assert_array_equal(ll, [[1.0, 2.0, 3.0, 4.0]] * 2)
        self.assertEqual(fo.calls, 1)

    def test_setup_captures_the_context_fanout_and_pickle_drops_it(self):
        import pickle

        move = FunctionMove(_fn, name="f")
        fo = _Fanout(single=False)
        ctx = types.SimpleNamespace(acs=_Acs([0.0]), ntemps=2, nwalkers=4, fanout=fo)
        self.assertIsNone(move.setup(ctx))
        self.assertIs(move.fanout, fo)
        self.assertEqual(move.accepted.shape, (2, 4))
        back = pickle.loads(pickle.dumps(move))
        self.assertIsNone(back.fanout)
        self.assertIsNone(back.acs)


if __name__ == "__main__":
    unittest.main()
